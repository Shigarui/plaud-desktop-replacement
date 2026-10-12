"""
meetingrec.recorder — simultaneous microphone + system-audio capture.

Three problems this solves, in order of how badly they bite:

1. TWO CLOCKS. The mic endpoint and the render endpoint are driven by separate
   hardware oscillators. Over an hour they drift seconds apart. Counting
   samples and assuming the tracks line up produces files that start in sync
   and end noticeably out. Instead each stream is pinned to wall time: every
   block is stamped on arrival, and if a stream falls behind where wall time
   says it should be, silence is inserted to catch it up. Both files then sit
   on the same timeline regardless of what the hardware did.

2. LATE DETECTION. The consent store updates a beat after a meeting actually
   starts, so by the time the detector fires you have already missed the part
   where someone says what the call is about. Capture therefore runs
   continuously, discarding into a ring buffer. When a meeting fires, the
   buffer is flushed into the file and you get the preceding ~30 seconds.
   Cost is roughly 12 MB of RAM held permanently.

3. DEVICES VANISH. Bluetooth headsets disconnect, docks get unplugged. A
   stream that dies must not take the other one with it, and must not take
   down the process. Each stream fails independently and keeps writing
   silence so the timeline stays intact.

Output per session, in staging:
    <slug>-mic.wav      what you said
    <slug>-system.wav   what everyone else said
    <slug>-mix.wav      both, for playback (optional)

Keeping mic and system separate is the whole point — it is free speaker
separation, and no diarizer will ever do as good a job of telling you apart
from the room as two physically distinct streams already do.
"""

from __future__ import annotations

import logging
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from .audio import (
    SoundcardReader,
    SounddeviceReader,
    conceal_dropouts,
    device_hostapi,
    downsample,
    enhance_speech,
    find_input_device,
    find_input_devices,
    high_frequency_fraction,
    list_input_devices,
    loopback_devices,
    resample,
)

log = logging.getLogger(__name__)

DEFAULT_RATE = 48000
# ~21 ms at 48 kHz. Measured, not guessed: on a Bluetooth headset mic, 4096
# frames (85 ms) produced 9-26% digital silence while 1024 produced 0.2% on the
# same device in the same minute. A WASAPI buffer that large starves between
# HFP packets and returns zeros rather than waiting, and those zeros are
# indistinguishable from packet loss after the fact. Smaller is safer here up
# to a point — 256 frames starves the other way (62% zeros).
DEFAULT_BLOCK = 1024
DRIFT_TOLERANCE = 0.05  # seconds of slip before padding kicks in
MAX_PAD_SECONDS = 5.0  # never paper over a stall longer than this


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _default_speaker_name() -> Optional[str]:
    """Current default render endpoint, or None if it cannot be read."""
    try:
        import soundcard as sc

        return str(sc.default_speaker().name)
    except Exception:
        return None


def to_int16(block: np.ndarray) -> np.ndarray:
    """float32 [-1, 1] -> int16, clipped. Clipping beats wrapping."""
    return (np.clip(block, -1.0, 1.0) * 32767.0).astype(np.int16)


@dataclass
class StreamStats:
    frames_written: int = 0
    frames_padded: int = 0
    blocks: int = 0
    errors: int = 0
    device: str = ""
    first_sample_monotonic: Optional[float] = None  # wall clock of frame 0
    dead_devices: tuple = ()  # opened fine but produced only digital silence
    rate: int = 0  # rate actually negotiated, which may not be the one requested
    failed: bool = False
    fallback: bool = False  # true if the preferred device did not work

    @property
    def padded_seconds(self) -> float:
        return self.frames_padded / DEFAULT_RATE


@dataclass
class RecordingResult:
    session_slug: str
    started_at: datetime
    mic_path: Optional[Path] = None
    system_path: Optional[Path] = None
    mix_path: Optional[Path] = None
    duration_seconds: float = 0.0
    preroll_seconds: float = 0.0
    stats: dict = field(default_factory=dict)

    @property
    def paths(self) -> list[Path]:
        return [p for p in (self.mic_path, self.system_path, self.mix_path) if p]


# --------------------------------------------------------------------------
# one stream
# --------------------------------------------------------------------------


class _Stream(threading.Thread):
    """
    Captures one device continuously.

    Idle: blocks go into a ring buffer and are dropped as they age out.
    Armed: the ring buffer is flushed to disk, then blocks are written live.
    """

    def __init__(
        self,
        role: str,
        rate: int = DEFAULT_RATE,
        channels: int = 2,
        block: int = DEFAULT_BLOCK,
        preroll_seconds: float = 30.0,
        loopback: bool = False,
        device_name: Optional[object] = None,  # str or list[str], in priority order
        lazy: bool = False,
        open_timeout: float = 5.0,
        warmup_seconds: float = 1.5,
        mic_rate: Optional[int] = None,
    ):
        super().__init__(name=f"capture-{role}", daemon=True)
        self.role = role
        self.rate = rate
        self.channels = channels
        self.block = block
        self.loopback = loopback
        self.device_name = device_name

        maxblocks = max(1, int(preroll_seconds * rate / block))
        self._ring: deque[tuple[float, np.ndarray]] = deque(maxlen=maxblocks)
        self._lock = threading.Lock()
        # NOT _stop: threading.Thread._stop is a real method that join() calls.
        self._stopping = threading.Event()

        # Lazy mode: hold the device closed until a meeting actually starts.
        # Bluetooth headsets drop from A2DP to HFP the instant anything opens
        # their microphone, so an always-open mic means mono 16 kHz music all
        # day. The cost is that the mic track has no pre-roll — the loopback
        # stream keeps its full buffer, so only your own first few seconds are
        # lost, which is usually you asking whether people can hear you.
        self.lazy = lazy
        self.open_timeout = open_timeout
        # A freshly opened Bluetooth input stream takes a second or two to
        # settle: measured 19.7% digital silence in the first 5 seconds, then
        # 0.0% for the following 45. Lazy opening lands every meeting squarely
        # in that window, so let the stream run into the ring buffer for a
        # moment before the file is opened. The audio is not discarded — it
        # goes to pre-roll, so the ragged part is still there if you want it.
        self.warmup_seconds = warmup_seconds
        # Windows moves playback to a different endpoint when a Bluetooth
        # headset switches profile — "Headphones (X)" for A2DP becomes
        # "Headset Earphone (X)" for HFP the moment a call opens the mic.
        # A loopback stream resolved once at startup then faithfully records
        # silence from the endpoint nobody is using any more, which is exactly
        # when the meeting you wanted is happening.
        self._switch_check_interval = 2.0
        # Force a capture rate rather than negotiating. Bluetooth endpoints
        # often accept 48 kHz and upsample a 16 kHz link behind the scenes,
        # which adds dropouts for no extra information.
        self.mic_rate = mic_rate
        # Silence watchdog. A device can open cleanly and return nothing at
        # all: a Bluetooth dongle whose headset is switched off does exactly
        # that, and so does a loopback endpoint the system stopped using. A
        # live microphone always has a noise floor, so a long run of EXACT
        # zeros means the device is dead rather than the room being quiet.
        # Without this the app records 128 seconds of nothing and reports
        # success, which is worse than failing.
        self._armed_frames = 0
        self._zero_frames = 0
        self._bad_devices: set[str] = set()
        self.silence_check_seconds = 12.0
        self.silence_threshold = 0.99
        self._active = threading.Event()
        self._opened = threading.Event()
        if not lazy:
            self._active.set()

        self._writer: Optional[wave.Wave_write] = None
        self._path: Optional[Path] = None
        self._file_rate: Optional[int] = None  # rate the open WAV was created at
        self._t0: Optional[float] = None  # wall clock of first written frame
        self.stats = StreamStats()

    # -- device ----------------------------------------------------------

    def _readers(self) -> list:
        """Readers to try, best first. Mics via PortAudio, loopback via soundcard."""
        if self.loopback:
            return [
                SoundcardReader(d, self.rate, self.channels, self.block)
                for d in loopback_devices()
            ]

        names = self.device_name
        if isinstance(names, str):
            names = [names]
        if self.mic_rate:
            self.rate = self.mic_rate
        return [
            SounddeviceReader(idx, name, self.rate, self.channels, self.block)
            for idx, name in find_input_devices(names)
        ]

    def _open_recorder(self):
        """Open the first reader that works. Returns the live reader."""
        readers = self._readers()
        if not readers:
            raise RuntimeError(f"no {self.role} devices present")

        live = [r for r in readers if r.device not in self._bad_devices]
        if live:
            readers = live
        elif self._bad_devices:
            # Everything has been marked dead at some point. Devices come back
            # (a headset gets switched on), so clear the slate and try again
            # rather than giving up for the rest of the session.
            log.warning("[%s] all devices previously silent — retrying from scratch",
                        self.role)
            self._bad_devices.clear()

        errors = []
        for i, reader in enumerate(readers):
            try:
                reader.__enter__()
            except Exception as exc:
                errors.append(f"{reader.device}: {exc!r}")
                log.warning("[%s] %s unusable — %r", self.role, reader.device, exc)
                continue

            self.stats.device = reader.device
            self.stats.fallback = i > 0
            self.stats.rate = reader.rate
            self.stats.dead_devices = tuple(sorted(self._bad_devices))
            # A device may open at a rate we did not ask for (Bluetooth HFP
            # runs at 16 kHz and cannot be persuaded otherwise). Track the
            # real rate so drift maths and the WAV header agree with reality.
            if reader.rate != self.rate:
                log.info("[%s] running at %d Hz, not %d", self.role, reader.rate, self.rate)
            self.rate = reader.rate
            self.channels = reader.channels
            if i > 0:
                log.warning("[%s] FELL BACK to %s (preferred device failed)",
                            self.role, reader.device)
            else:
                log.info("[%s] device: %s @ %d Hz", self.role, reader.device, reader.rate)
            return reader
        raise RuntimeError(f"no usable {self.role} device — tried: {'; '.join(errors)}")

    # -- lifecycle -------------------------------------------------------

    def run(self) -> None:
        backoff = 1.0
        while not self._stopping.is_set():
            # In lazy mode this parks the thread with the device closed.
            if not self._active.wait(0.25):
                continue

            reader = None
            try:
                reader = self._open_recorder()
                self.stats.failed = False
                backoff = 1.0
                self._opened.set()
                next_check = time.monotonic() + self._switch_check_interval
                switched = False
                while not self._stopping.is_set() and self._active.is_set():
                    data = reader.read(self.block)
                    self._handle(np.asarray(data, dtype=np.float32), time.monotonic())
                    if self._writer is not None and self._armed_frames >= \
                            self.rate * self.silence_check_seconds:
                        ratio = self._zero_frames / max(self._armed_frames, 1)
                        self._armed_frames = 0
                        self._zero_frames = 0
                        if ratio >= self.silence_threshold:
                            log.error("[%s] %s is producing pure silence (%.0f%% exact "
                                      "zeros over %.0fs) — treating it as dead",
                                      self.role, reader.device, ratio * 100,
                                      self.silence_check_seconds)
                            self._bad_devices.add(reader.device)
                            switched = True
                            break

                    if self.loopback and time.monotonic() >= next_check:
                        next_check = time.monotonic() + self._switch_check_interval
                        current = _default_speaker_name()
                        if current and current != reader.device:
                            log.warning("[%s] default output moved to %s — reopening",
                                        self.role, current)
                            switched = True
                            break
                if switched:
                    backoff = 0.0  # deliberate switch, not a failure: reopen at once
            except Exception as exc:
                self.stats.errors += 1
                self.stats.failed = True
                # repr, not str: AssertionError carries no message and str()
                # renders it as an empty string, which tells you nothing.
                log.warning("[%s] capture stopped — %r — retry in %.0fs",
                            self.role, exc, backoff)
            finally:
                self._opened.clear()
                if reader is not None:
                    try:
                        reader.__exit__(None, None, None)
                    except Exception:
                        pass
            if not self._stopping.is_set() and self._active.is_set():
                if backoff:
                    self._stopping.wait(backoff)
                    backoff = min(backoff * 2, 30.0)
                else:
                    backoff = 1.0  # reset after an endpoint switch

    def stop(self) -> None:
        self._stopping.set()

    # -- block handling --------------------------------------------------

    def _handle(self, block: np.ndarray, stamp: float) -> None:
        self.stats.blocks += 1
        with self._lock:
            if self._writer is None:
                self._ring.append((stamp, block))
                return
            if self._t0 is None:
                self._t0 = stamp
            self._catch_up(stamp)
            self._write(block)
            self._armed_frames += block.shape[0]
            self._zero_frames += int(np.count_nonzero(np.all(block == 0.0, axis=-1))
                                     if block.ndim > 1 else np.count_nonzero(block == 0.0))

    def _catch_up(self, stamp: float) -> None:
        """Insert silence if wall time has outrun the samples we have."""
        assert self._t0 is not None
        rate = self._file_rate or self.rate
        expected = int((stamp - self._t0) * rate)
        deficit = expected - self.stats.frames_written
        if deficit <= int(DRIFT_TOLERANCE * rate):
            return
        deficit = min(deficit, int(MAX_PAD_SECONDS * rate))
        self._write(np.zeros((deficit, self.channels), dtype=np.float32))
        self.stats.frames_padded += deficit
        log.debug("[%s] padded %d frames to hold sync", self.role, deficit)

    def _write(self, block: np.ndarray) -> None:
        if self._writer is None:
            return
        # A device can be swapped out mid-meeting and come back at a different
        # rate. The WAV header is already written, so conform to it rather than
        # producing a file whose back half plays at the wrong speed.
        if self._file_rate is not None and self.rate != self._file_rate and block.size:
            block = resample(block, self.rate, self._file_rate)
        if block.ndim == 1:
            block = block.reshape(-1, 1)
        if block.shape[1] != self.channels:
            block = (
                np.repeat(block[:, :1], self.channels, axis=1)
                if block.shape[1] == 1
                else block[:, : self.channels]
            )
        self._writer.writeframes(to_int16(block).tobytes())
        self.stats.frames_written += block.shape[0]

    # -- arm / disarm ----------------------------------------------------

    def arm(self, path: Path) -> float:
        """Open the file and flush pre-roll. Returns seconds of pre-roll saved."""
        if self.lazy:
            # Wake the capture thread and wait for the device before writing the
            # WAV header — the header has to carry the rate the device actually
            # negotiated, and for Bluetooth that is not the one we asked for.
            self._active.set()
            if not self._opened.wait(self.open_timeout):
                log.error("[%s] device did not open within %.0fs — writing header at "
                          "%d Hz and continuing to retry", self.role, self.open_timeout,
                          self.rate)
            elif self.warmup_seconds > 0:
                time.sleep(self.warmup_seconds)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            w = wave.open(str(path), "wb")
            w.setnchannels(self.channels)
            w.setsampwidth(2)
            w.setframerate(self.rate)
            self._writer = w
            self._path = path
            self._file_rate = self.rate
            # Carry the negotiated rate across the reset — otherwise the
            # summary reports the rate we asked for rather than the one the
            # device actually gave us, which is exactly backwards.
            self.stats = StreamStats(device=self.stats.device, rate=self.rate,
                                     fallback=self.stats.fallback)
            self._armed_frames = 0
            self._zero_frames = 0

            buffered = list(self._ring)
            self._ring.clear()
            if buffered:
                self._t0 = buffered[0][0]
                for _stamp, blk in buffered:
                    self._write(blk)
            else:
                self._t0 = None
            return self.stats.frames_written / self.rate

    def disarm(self) -> Optional[Path]:
        with self._lock:
            if self._writer is None:
                return None
            try:
                self._writer.close()
            except Exception:
                log.exception("[%s] error closing wav", self.role)
            # Keep the wall-clock time of the first sample: the two streams
            # flush different amounts of pre-roll, so their files start at
            # different absolute times and must be aligned afterwards.
            self.stats.first_sample_monotonic = self._t0
            path, self._writer, self._t0 = self._path, None, None
            self._path = None
            self._file_rate = None
        if self.lazy:
            # Release the device so the headset can return to A2DP.
            self._active.clear()
            log.info("[%s] device released", self.role)
        return path


# --------------------------------------------------------------------------
# both streams together
# --------------------------------------------------------------------------


class DualRecorder:
    """
    Owns the mic and system streams. Start once at launch and leave running;
    arm/disarm per meeting.
    """

    def __init__(
        self,
        staging_dir: Path,
        rate: int = DEFAULT_RATE,
        preroll_seconds: float = 30.0,
        block_frames: int = DEFAULT_BLOCK,
        write_mixdown: bool = True,
        mic_device: Optional[object] = None,  # str or list[str], priority order
        normalize_mic_rate: bool = True,
        clean_mic: bool = True,
        highpass_hz: float = 250.0,
        keep_raw_mic: bool = False,
        conceal_dropouts: bool = True,
        lazy_mic: bool = True,
        warmup_seconds: float = 1.5,
        mic_rate: Optional[int] = None,
    ):
        self.staging_dir = Path(staging_dir)
        self.rate = rate
        self.write_mixdown = write_mixdown
        self.preroll_seconds = preroll_seconds
        self.normalize_mic_rate = normalize_mic_rate
        self.clean_mic = clean_mic
        self.highpass_hz = highpass_hz
        self.keep_raw_mic = keep_raw_mic
        self.conceal_dropouts = conceal_dropouts
        self.lazy_mic = lazy_mic
        self.warmup_seconds = warmup_seconds

        self.block_frames = block_frames
        self.mic = _Stream(
            "mic", rate=rate, channels=1, block=block_frames,
            # In lazy mode the ring only has to hold the warm-up window, but it
            # must not be zero or that audio is lost rather than merely ragged.
            preroll_seconds=(warmup_seconds + 0.5) if lazy_mic else preroll_seconds,
            loopback=False, device_name=mic_device, lazy=lazy_mic,
            warmup_seconds=warmup_seconds if lazy_mic else 0.0,
            mic_rate=mic_rate,
        )
        self.system = _Stream("system", rate=rate, channels=2, block=block_frames,
                              preroll_seconds=preroll_seconds, loopback=True,
                              warmup_seconds=0.0)  # always open; already settled

        self._armed_at: Optional[float] = None
        self._slug: Optional[str] = None
        self._started_at: Optional[datetime] = None
        self._preroll_actual = 0.0

    @property
    def armed(self) -> bool:
        return self._armed_at is not None

    def start(self) -> None:
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.mic.start()
        self.system.start()
        if self.lazy_mic:
            log.info("capture running — system pre-roll %.0fs, mic opens on demand "
                     "(%.1fs warm-up)", self.preroll_seconds, self.warmup_seconds)
        else:
            log.info("capture running (pre-roll %.0fs, mic held open)", self.preroll_seconds)

    def shutdown(self) -> Optional[RecordingResult]:
        result = self.stop_recording() if self.armed else None
        self.mic.stop()
        self.system.stop()
        return result

    # -- per meeting ------------------------------------------------------

    def start_recording(self, slug: str) -> None:
        if self.armed:
            log.warning("already recording %s — ignoring start", self._slug)
            return
        self._slug = slug
        self._started_at = datetime.now()
        mic_pre = self.mic.arm(self.staging_dir / f"{slug}-mic.wav")
        sys_pre = self.system.arm(self.staging_dir / f"{slug}-system.wav")
        self._preroll_actual = min(mic_pre, sys_pre)
        self._armed_at = time.monotonic()
        log.info("RECORDING %s (recovered %.1fs of pre-roll)", slug, self._preroll_actual)

    def stop_recording(self) -> Optional[RecordingResult]:
        if not self.armed:
            return None
        duration = time.monotonic() - (self._armed_at or 0)
        mic_path = self.mic.disarm()
        sys_path = self.system.disarm()
        self._armed_at = None

        result = RecordingResult(
            session_slug=self._slug or "unknown",
            started_at=self._started_at or datetime.now(),
            mic_path=mic_path,
            system_path=sys_path,
            duration_seconds=duration + self._preroll_actual,
            preroll_seconds=self._preroll_actual,
            stats={"mic": vars(self.mic.stats), "system": vars(self.system.stats)},
        )

        for role, st in (("mic", self.mic.stats), ("system", self.system.stats)):
            if st.frames_padded:
                log.info("[%s] %.2fs of silence inserted to hold sync",
                         role, st.padded_seconds)
            if st.failed:
                log.warning("[%s] stream was failing at stop — check the file", role)

        # Align first: the mixdown and every later consumer assume both files
        # describe the same stretch of time.
        try:
            result.stats["alignment"] = align_tracks(
                mic_path, self.mic.stats.first_sample_monotonic,
                sys_path, self.system.stats.first_sample_monotonic)
        except Exception:
            log.exception("track alignment failed — files left unaligned")

        # Clean before the rate check: the rumble we are removing is exactly
        # the low-frequency energy that would otherwise skew the bandwidth
        # measurement, and before the mixdown so the mix uses the good track.
        if self.clean_mic and mic_path:
            try:
                result.stats.setdefault("mic", {}).update(
                    clean_mic_track(mic_path, self.highpass_hz, self.keep_raw_mic,
                                    self.conceal_dropouts)
                )
            except Exception:
                log.exception("mic cleanup failed — keeping the raw capture")

        if self.normalize_mic_rate and mic_path:
            try:
                rewritten, hf = rewrite_at_true_rate(mic_path)
                result.stats.setdefault("mic", {})["hf_fraction"] = round(hf, 5)
                result.stats["mic"]["rate_corrected"] = rewritten
            except Exception:
                log.exception("rate check failed — leaving %s as recorded", mic_path.name)

        if self.write_mixdown and mic_path and sys_path:
            try:
                result.mix_path = mixdown(mic_path, sys_path,
                                          self.staging_dir / f"{self._slug}-mix.wav")
            except Exception:
                log.exception("mixdown failed — separate tracks are unaffected")

        log.info("stopped %s (%.1fs)", result.session_slug, result.duration_seconds)
        return result


# --------------------------------------------------------------------------
# mixdown
# --------------------------------------------------------------------------


def _read_wav(path: Path) -> tuple[np.ndarray, int, int]:
    with wave.open(str(path), "rb") as w:
        rate, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        raw = w.readframes(n)
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return data.reshape(-1, ch) if ch > 1 else data.reshape(-1, 1), rate, ch


def mixdown(mic_path: Path, system_path: Path, out_path: Path,
            mic_gain: float = 1.0, system_gain: float = 1.0) -> Path:
    """
    Sum the two tracks into a stereo file for convenient playback.

    Both are attenuated before summing — two full-scale sources added together
    clip immediately, and clipping is not recoverable. The separate tracks
    remain the authoritative copies; this one is for listening.
    """
    mic, mrate, _ = _read_wav(mic_path)
    sysa, srate, _ = _read_wav(system_path)

    # The two tracks routinely differ now: a Bluetooth mic records at 16 kHz
    # while loopback stays at 48 kHz. Mix at the higher rate so the system
    # audio is not degraded to meet the mic.
    rate = max(mrate, srate)
    if mrate != rate:
        mic = resample(mic, mrate, rate)
    if srate != rate:
        sysa = resample(sysa, srate, rate)

    if mic.shape[1] == 1:
        mic = np.repeat(mic, 2, axis=1)
    if sysa.shape[1] == 1:
        sysa = np.repeat(sysa, 2, axis=1)

    n = max(mic.shape[0], sysa.shape[0])
    mic = np.pad(mic, ((0, n - mic.shape[0]), (0, 0)))
    sysa = np.pad(sysa, ((0, n - sysa.shape[0]), (0, 0)))

    mixed = mic * (mic_gain * 0.7) + sysa * (system_gain * 0.7)
    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > 1.0:
        mixed /= peak

    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(to_int16(mixed).tobytes())
    return out_path


def rewrite_at_true_rate(path: Path, cutoff_hz: float = 7000.0,
                         threshold: float = 0.01, target_rate: int = 16000) -> tuple[bool, float]:
    """
    If a track contains no real high-frequency content, rewrite it at a rate
    that matches what it actually holds.

    Bluetooth headset mics deliver ~16 kHz narrowband audio, but Windows
    presents the endpoint at 48 kHz and upsamples on the way out. The file then
    claims a fidelity it does not have: three times the size, no extra
    information, and interpolation artifacts layered on top of an already
    compressed codec. Measuring the spectrum and writing the honest rate costs
    nothing and produces a smaller, cleaner file.

    Returns (rewritten, measured_hf_fraction).
    """
    data, rate, channels = _read_wav(path)
    if rate <= target_rate or data.size == 0:
        return False, 1.0

    hf = high_frequency_fraction(data, rate, cutoff_hz)
    if hf >= threshold:
        log.info("%s is wideband (%.2f%% above %.0f kHz) — left at %d Hz",
                 path.name, hf * 100, cutoff_hz / 1000, rate)
        return False, hf

    reduced = downsample(data, rate, target_rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(target_rate)
        w.writeframes(to_int16(reduced).tobytes())
    log.info("%s is narrowband (%.2f%% above %.0f kHz) — rewritten at %d Hz",
             path.name, hf * 100, cutoff_hz / 1000, target_rate)
    return True, hf


def pad_front(path: Path, seconds: float) -> float:
    """
    Insert leading silence so a track starts at a given point on the timeline.

    Returns the seconds actually added. Rewrites the file in place.
    """
    if seconds <= 0:
        return 0.0
    data, rate, channels = _read_wav(path)
    frames = int(round(seconds * rate))
    if frames <= 0:
        return 0.0
    padded = np.vstack([np.zeros((frames, channels), dtype=np.float32), data])
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(to_int16(padded).tobytes())
    return frames / rate


def align_tracks(mic_path: Optional[Path], mic_t0: Optional[float],
                 system_path: Optional[Path], system_t0: Optional[float]) -> dict:
    """
    Put both tracks on one timeline.

    The streams hold different amounts of pre-roll — 30 s for system audio,
    a second or two for a lazily-opened mic — so the system file begins almost
    half a minute earlier in real time. Played together they appear wildly out
    of sync: you ask a question six seconds in and the answer arrives thirty
    seconds later. Both streams stamp their first sample against the same
    monotonic clock, so the later one is simply front-padded to match.
    """
    if not (mic_path and system_path and mic_t0 is not None and system_t0 is not None):
        return {"aligned": False, "reason": "missing track or timestamp"}

    offset = mic_t0 - system_t0
    if abs(offset) < 0.02:
        return {"aligned": True, "offset_seconds": 0.0}

    if offset > 0:
        added = pad_front(mic_path, offset)      # mic started later
        who = "mic"
    else:
        added = pad_front(system_path, -offset)  # system started later
        who = "system"
    log.info("aligned tracks: padded %s by %.2fs", who, added)
    return {"aligned": True, "offset_seconds": round(offset, 3),
            "padded": who, "padded_seconds": round(added, 3)}


def clean_mic_track(path: Path, highpass_hz: float = 250.0,
                    keep_raw: bool = False, conceal: bool = True) -> dict:
    """
    High-pass and normalise a mic track in place.

    Optionally keeps the untouched capture beside it as *-raw.wav. Off by
    default: the raw file is the same audio with inaudible rumble on top, and
    a second copy of every meeting is a lot of disk for something you will
    never listen to.
    """
    data, rate, channels = _read_wav(path)
    if data.size == 0:
        return {}
    if keep_raw:
        raw_path = path.with_name(path.stem + "-raw" + path.suffix)
        with wave.open(str(raw_path), "wb") as w:
            w.setnchannels(channels)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(to_int16(data).tobytes())

    # Conceal before enhancing: the high-pass would otherwise ring at every
    # step discontinuity, turning each click into a longer smear.
    stats: dict = {}
    if conceal:
        data, gap_stats = conceal_dropouts(data, rate)
        stats.update(gap_stats)
        if gap_stats.get("dropouts"):
            log.info("%s: %d dropouts (%.1f%%) — %d bridged, %d ramped",
                     path.name, gap_stats["dropouts"], gap_stats["dropout_pct"],
                     gap_stats["bridged"], gap_stats["ramped"])

    cleaned, enh = enhance_speech(data, rate, highpass_hz=highpass_hz)
    stats.update(enh)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(to_int16(cleaned).tobytes())
    log.info("%s cleaned: speech band %.0f%% -> %.0f%% (gain %+.1f dB)",
             path.name, stats["speech_band_before"] * 100,
             stats["speech_band_after"] * 100, stats["gain_db"])
    return stats


def list_devices(probe: bool = True) -> str:
    """
    What each library can see, and — with probe=True — what actually opens.

    Mics are probed through PortAudio because that is what records them, and
    loopback through soundcard for the same reason. A device listed by Windows
    is not necessarily one either library can drive, which is the failure worth
    catching before a meeting rather than during one.
    """
    import soundcard as sc

    lines = ["Speakers (system audio is captured from the default):"]
    try:
        default_spk = sc.default_speaker().name
    except Exception:
        default_spk = ""
    for spk in sc.all_speakers():
        lines.append(f"  {'*' if spk.name == default_spk else ' '} {spk.name}")

    lines.append("\nMicrophones (via PortAudio):")
    try:
        default_in = find_input_device(None)
        default_idx = default_in[0] if default_in else -1
    except Exception:
        default_idx = -1

    for idx, name, ch, sr in list_input_devices():
        mark = "*" if idx == default_idx else " "
        note = f"  [{device_hostapi(idx)}, {ch}ch @ {sr} Hz]"
        if probe:
            reader = SounddeviceReader(idx, name, DEFAULT_RATE, 1, 1024)
            try:
                with reader:
                    reader.read(1024)
                note += f"  [works @ {reader.rate} Hz]"
            except Exception as exc:
                note += f"  [FAILS: {type(exc).__name__}]"
        lines.append(f"  {mark} {name}{note}")

    lines.append("\nLoopback (via soundcard):")
    for dev in loopback_devices():
        note = ""
        if probe:
            try:
                with SoundcardReader(dev, DEFAULT_RATE, 2, 1024) as r:
                    r.read(1024)
                note = "  [works]"
            except Exception as exc:
                note = f"  [FAILS: {type(exc).__name__}]"
        lines.append(f"    {dev.name}{note}")

    lines.append("\n* = default")
    if probe:
        lines.append("A device marked [FAILS] is skipped automatically at record time.")
        lines.append("Duplicates across host APIs are collapsed; WASAPI wins where available.")
        lines.append("A mic opening below 48000 Hz is normal for Bluetooth (HFP);")
        lines.append("it is recorded at its true rate and resampled at mixdown.")
    return "\n".join(lines)
