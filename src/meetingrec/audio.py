"""
meetingrec.audio — device readers, one per library.

Why two libraries instead of one:

soundcard drives WASAPI loopback flawlessly, but it asserts that every
endpoint reports WAVEFORMATEXTENSIBLE with float32 before it will open
anything. Bluetooth headset microphones report plain WAVEFORMATEX with
cbSize = 0, so soundcard refuses them outright — not a rate problem, a
struct-shape problem, and it fails identically at every rate. Native apps
avoid this because they handle whatever format the endpoint reports.

sounddevice wraps PortAudio, which does exactly that. So mics go through
sounddevice and loopback stays on soundcard, which is the arrangement where
each library does the part it is good at.

The second thing PortAudio buys us is honest sample rates. A Bluetooth mic in
HFP mode runs at 16 kHz and cannot be talked into 48 kHz. Rather than fail, a
reader negotiates: it asks for the rate we want, and if the device says no, it
opens at the device's native rate and reports back what it actually got. The
file is then written at that true rate and resampled only at mixdown, so the
raw track is never silently wrong about what it contains.

Every reader is a context manager exposing:
    .device    str   what actually opened
    .rate      int   the rate actually in use, which may not be the one asked
    .channels  int
    .read(n)   -> float32 array, shape (n, channels)
"""

from __future__ import annotations

import logging
from typing import Optional, Protocol

import numpy as np

log = logging.getLogger(__name__)


class AudioReader(Protocol):
    device: str
    rate: int
    channels: int

    def __enter__(self) -> "AudioReader": ...
    def __exit__(self, *exc) -> bool: ...
    def read(self, frames: int) -> np.ndarray: ...


# --------------------------------------------------------------------------
# sounddevice / PortAudio — microphones
# --------------------------------------------------------------------------


class SounddeviceReader:
    """
    PortAudio input. Accepts any format the device reports, and falls back to
    the device's native sample rate rather than failing when the requested one
    is unavailable.
    """

    def __init__(self, device_index: int, device_name: str, rate: int,
                 channels: int, block: int):
        self.device_index = device_index
        self.device = device_name
        self.requested_rate = rate
        self.rate = rate
        self.channels = channels
        self.block = block
        self._stream = None

    def __enter__(self) -> "SounddeviceReader":
        import sounddevice as sd

        info = sd.query_devices(self.device_index)
        max_ch = int(info.get("max_input_channels", 1))
        if max_ch < 1:
            raise RuntimeError(f"{self.device} has no input channels")
        self.channels = min(self.channels, max_ch)

        native = int(info.get("default_samplerate") or self.requested_rate)
        for rate in dict.fromkeys([self.requested_rate, native, 48000, 16000, 44100]):
            try:
                stream = sd.InputStream(
                    device=self.device_index,
                    samplerate=rate,
                    channels=self.channels,
                    dtype="float32",
                    blocksize=self.block,
                    latency="high",  # favour not dropping blocks over low latency
                )
                stream.start()
            except Exception as exc:
                log.debug("[%s] %d Hz rejected: %r", self.device, rate, exc)
                continue
            self._stream = stream
            self.rate = int(stream.samplerate)
            if self.rate != self.requested_rate:
                log.warning("%s opened at %d Hz (asked for %d) — resampled at mixdown",
                            self.device, self.rate, self.requested_rate)
            return self
        raise RuntimeError(f"{self.device}: no usable sample rate")

    def __exit__(self, *exc) -> bool:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                log.debug("error closing %s", self.device, exc_info=True)
            self._stream = None
        return False

    def read(self, frames: int) -> np.ndarray:
        assert self._stream is not None
        data, overflowed = self._stream.read(frames)
        if overflowed:
            # Blocks were dropped by the driver. The drift padding will patch
            # the hole so the timeline stays honest.
            log.debug("[%s] input overflow", self.device)
        return np.asarray(data, dtype=np.float32)


# PortAudio exposes the same physical mic once per host API. They are not
# equivalent: WASAPI is the modern shared-mode path and the one Windows itself
# uses; WDM-KS wants exclusive access and usually fails while anything else
# holds the device; MME and DirectSound work but resample in software and add
# latency. Preference order matters more than it looks — picking the wrong
# duplicate is the difference between a mic that works and one that throws.
# Host API preference for MICROPHONES.
#
# MME first is deliberate and measured. Raw WASAPI shared mode hands over the
# unprocessed stream: on a Bluetooth headset that meant 54% of energy below
# 300 Hz and 34% in the speech band. The same device through MME gave 14% and
# 77% — a completely different recording, before any filtering of ours.
#
# The reason is that MME routes through the Windows audio engine, which applies
# the endpoint's own capture processing; WASAPI shared bypasses much of it.
# Commercial recorders sound better on these devices for exactly this reason,
# not because of anything clever downstream. MME's higher latency is irrelevant
# when the job is writing a file.
#
# WDM-KS stays last: it wants exclusive access and usually fails outright.
MIC_HOSTAPI_RANK = {
    "MME": 0,
    "Windows DirectSound": 1,
    "Windows WASAPI": 2,
    "Windows WDM-KS": 3,
}

# Kept for callers that want the old ordering (diagnostics, comparisons).
WASAPI_FIRST_RANK = {
    "Windows WASAPI": 0,
    "Windows DirectSound": 1,
    "MME": 2,
    "Windows WDM-KS": 3,
}

HOSTAPI_RANK = MIC_HOSTAPI_RANK  # default used by list_input_devices

# Not real devices — aliases that route to whatever Windows feels like.
GENERIC_NAMES = ("microsoft sound mapper", "primary sound capture driver")


def _hostapi_names() -> dict[int, str]:
    import sounddevice as sd

    return {i: a["name"] for i, a in enumerate(sd.query_hostapis())}


def list_input_devices(all_hostapis: bool = False,
                       rank: Optional[dict] = None) -> list[tuple[int, str, int, int]]:
    """
    Real input devices, de-duplicated across host APIs.

    Returns (index, name, max_input_channels, default_samplerate). With
    all_hostapis=True, returns everything PortAudio reports, duplicates and
    unusable WDM-KS entries included — useful for diagnosis, not for picking.
    """
    import sounddevice as sd

    apis = _hostapi_names()
    rows = []
    for i, d in enumerate(sd.query_devices()):
        if int(d.get("max_input_channels", 0)) < 1:
            continue
        name = str(d["name"]).strip()
        api = apis.get(int(d.get("hostapi", -1)), "")
        rows.append((i, name, int(d["max_input_channels"]),
                     int(d.get("default_samplerate") or 0), api))

    if all_hostapis:
        return [(i, f"{n}  [{a}]", c, s) for i, n, c, s, a in rows]

    best: dict[str, tuple] = {}
    for i, name, ch, sr, api in rows:
        if any(g in name.lower() for g in GENERIC_NAMES):
            continue
        # PortAudio truncates MME names to 31 chars, so the same mic appears
        # under slightly different strings. Key on the truncated form.
        key = name.lower()[:28]
        rank_value = (rank or HOSTAPI_RANK).get(api, 9)
        if key not in best or rank_value < best[key][0]:
            best[key] = (rank_value, i, name, ch, sr)

    return [(i, name, ch, sr) for _rank, i, name, ch, sr in
            sorted(best.values(), key=lambda r: (r[0], r[2].lower()))]


def device_hostapi(index: int) -> str:
    import sounddevice as sd

    try:
        return _hostapi_names().get(int(sd.query_devices(index)["hostapi"]), "?")
    except Exception:
        return "?"


def match_input_device(name: str, devices: list[tuple[int, str, int, int]]):
    """
    Resolve one configured name to a device.

    Matching is deliberately loose: PortAudio truncates names to 31 characters
    on MME and renders WASAPI names slightly differently from what Windows
    shows, so exact matching fails more often than it succeeds. Exact first,
    then prefix, then substring.
    """
    target = name.strip().lower()
    for idx, dname, _ch, _sr in devices:
        if dname.strip().lower() == target:
            return idx, dname
    for idx, dname, _ch, _sr in devices:
        low = dname.strip().lower()
        if low.startswith(target[:28]) or target.startswith(low[:28]):
            return idx, dname
    for idx, dname, _ch, _sr in devices:
        if target[:20] in dname.lower() or dname.lower()[:20] in target:
            return idx, dname
    return None


def find_input_devices(names: Optional[list[str]] = None) -> list[tuple[int, str]]:
    """
    Build the mic preference order: configured names first, in the order given,
    then the system default, then everything else as a last resort.

    The ordering matters more than it looks. With a single configured name and
    a blind fallback, a headset that is switched off silently demotes you to
    whatever else exists — a webcam on the far side of the desk, for instance —
    and the recording sounds inexplicably terrible. Naming several devices in
    priority order means the fallback is a choice rather than an accident.
    """
    devices = list_input_devices()
    if not devices:
        return []

    out: list[tuple[int, str]] = []
    seen: set[int] = set()

    for name in names or []:
        if not name:
            continue
        hit = match_input_device(name, devices)
        if hit is None:
            log.warning("configured mic %r not found — skipping", name)
            continue
        if hit[0] not in seen:
            seen.add(hit[0])
            out.append(hit)

    try:
        import sounddevice as sd

        default_idx = sd.default.device[0]
        if default_idx is not None and default_idx >= 0:
            dname = str(sd.query_devices(default_idx)["name"]).strip().lower()
            for idx, name_, _ch, _sr in devices:
                if name_.strip().lower()[:28] == dname[:28] and idx not in seen:
                    seen.add(idx)
                    out.append((idx, name_))
                    break
    except Exception:
        pass

    for idx, name_, _ch, _sr in devices:
        if idx not in seen:
            seen.add(idx)
            out.append((idx, name_))
    return out


def find_input_device(name: Optional[str]) -> Optional[tuple[int, str]]:
    """Single-name convenience wrapper over find_input_devices."""
    found = find_input_devices([name] if name else None)
    return found[0] if found else None


# --------------------------------------------------------------------------
# soundcard — loopback
# --------------------------------------------------------------------------


class SoundcardReader:
    """WASAPI loopback via soundcard. Only used for system audio."""

    def __init__(self, device, rate: int, channels: int, block: int):
        self._device = device
        self.device = str(device.name)
        self.rate = rate
        self.channels = channels
        self.block = block
        self._cm = None
        self._rec = None

    def __enter__(self) -> "SoundcardReader":
        self._cm = self._device.recorder(
            samplerate=self.rate, channels=self.channels, blocksize=self.block
        )
        self._rec = self._cm.__enter__()
        return self

    def __exit__(self, *exc) -> bool:
        if self._cm is not None:
            try:
                self._cm.__exit__(None, None, None)
            except Exception:
                log.debug("error closing %s", self.device, exc_info=True)
            self._cm, self._rec = None, None
        return False

    def read(self, frames: int) -> np.ndarray:
        assert self._rec is not None
        return np.asarray(self._rec.record(numframes=frames), dtype=np.float32)


def loopback_devices() -> list:
    """Loopback endpoints, default speaker first."""
    import soundcard as sc

    out, seen = [], set()

    def add(d) -> None:
        if d is not None and d.name not in seen:
            seen.add(d.name)
            out.append(d)

    try:
        add(sc.get_microphone(str(sc.default_speaker().name), include_loopback=True))
    except Exception as exc:
        log.warning("no default speaker loopback: %r", exc)
    for m in sc.all_microphones(include_loopback=True):
        if getattr(m, "isloopback", False):
            add(m)
    return out


# --------------------------------------------------------------------------
# resampling
# --------------------------------------------------------------------------


def resample(data: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """
    Linear-interpolation resample.

    Adequate for speech going into a transcriber, and it keeps numpy as the
    only dependency. It is not a polyphase filter, so there is some aliasing
    on downsampling — which is why the raw tracks are stored at their native
    rates and only the mixdown is resampled. If transcription quality ever
    looks rate-related, this function is the thing to replace.
    """
    if src_rate == dst_rate or data.size == 0:
        return data
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    n_out = int(round(data.shape[0] * dst_rate / src_rate))
    if n_out <= 0:
        return np.zeros((0, data.shape[1]), dtype=np.float32)
    x_old = np.arange(data.shape[0], dtype=np.float64)
    x_new = np.linspace(0, data.shape[0] - 1, n_out, dtype=np.float64)
    return np.stack(
        [np.interp(x_new, x_old, data[:, c]) for c in range(data.shape[1])], axis=1
    ).astype(np.float32)


# --------------------------------------------------------------------------
# bandwidth measurement and honest downsampling
# --------------------------------------------------------------------------


def lowpass(data: np.ndarray, cutoff_hz: float, rate: int, taps: int = 127) -> np.ndarray:
    """
    Windowed-sinc low-pass. Needed before decimation: throwing away samples
    without filtering first folds everything above the new Nyquist back down
    into the audible band as aliasing, which sounds far worse than the
    bandwidth limit you were trying to respect.
    """
    if data.size == 0:
        return data
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    taps = min(taps, max(9, (data.shape[0] // 2) * 2 - 1))
    n = np.arange(taps) - (taps - 1) / 2.0
    h = np.sinc(2.0 * (cutoff_hz / rate) * n) * np.hamming(taps)
    h /= h.sum()

    # Reflect-pad before convolving. A plain "same" convolution treats the
    # signal as zero outside its bounds, which creates a step at each end and
    # rings for half the filter length — audible as a click at the start and
    # end of every file, and enough to defeat the stopband entirely.
    pad = taps // 2
    padded = np.pad(data, ((pad, pad), (0, 0)), mode="reflect")
    return np.stack(
        [np.convolve(padded[:, c], h, mode="valid") for c in range(data.shape[1])], axis=1
    ).astype(np.float32)


def downsample(data: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Anti-aliased downsample. Integer ratios decimate; others interpolate."""
    if dst_rate >= src_rate or data.size == 0:
        return data
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    filtered = lowpass(data, dst_rate * 0.45, src_rate)
    if src_rate % dst_rate == 0:
        return filtered[:: src_rate // dst_rate].astype(np.float32)
    return resample(filtered, src_rate, dst_rate)


def high_frequency_fraction(data: np.ndarray, rate: int, cutoff_hz: float = 7000.0,
                            rms_gate: float = 0.005) -> float:
    """
    Fraction of signal energy above cutoff, measured over voiced windows only.

    Silence has no spectrum worth measuring, so quiet windows are gated out —
    otherwise a recording that is mostly pauses reports whatever the noise
    floor happens to look like.
    """
    if data.ndim > 1:
        data = data[:, 0]
    win = 8192
    if data.size < win:
        return 1.0
    spec = np.zeros(win // 2 + 1)
    voiced = 0
    for i in range(0, data.size - win, win):
        chunk = data[i : i + win]
        if np.sqrt(np.mean(chunk**2)) < rms_gate:
            continue
        spec += np.abs(np.fft.rfft(chunk * np.hanning(win))) ** 2
        voiced += 1
    if voiced == 0 or spec.sum() <= 0:
        return 1.0  # nothing to judge — leave the file alone
    freqs = np.fft.rfftfreq(win, 1.0 / rate)
    return float(spec[freqs > cutoff_hz].sum() / spec.sum())


def highpass(data: np.ndarray, cutoff_hz: float, rate: int, taps: int = 511) -> np.ndarray:
    """High-pass by spectral inversion of the low-pass. Long kernel: the whole
    point is a steep skirt down at 250 Hz, and a short filter smears it."""
    if data.size == 0:
        return data
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return (data - lowpass(data, cutoff_hz, rate, taps)).astype(np.float32)


def enhance_speech(data: np.ndarray, rate: int, highpass_hz: float = 250.0,
                   target_peak_dbfs: float = -3.0) -> tuple[np.ndarray, dict]:
    """
    Make a raw mic track listenable, the way every voice recorder does.

    Desk microphones sit in a bath of infrasonic energy: HVAC, building
    vibration through the desk, handling, and mains hum. Measured on real
    captures here, 88% of total energy sat below 300 Hz and only 8% in the
    300-3000 Hz band where speech intelligibility actually lives. None of that
    low end is audible as content, but it dominates the peak, so normalising
    the raw file just makes the rumble louder and leaves the voice buried.

    High-passing at 250 Hz and then normalising moves the same recording to
    roughly 63% speech-band energy, which matches what a commercial recorder
    produces from the same room.

    Returns (processed, stats).
    """
    if data.size == 0:
        return data, {}
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    before = _band_share(data[:, 0], rate)
    out = highpass(data, highpass_hz, rate)

    peak = float(np.max(np.abs(out)))
    gain = 1.0
    if peak > 1e-6:
        gain = (10.0 ** (target_peak_dbfs / 20.0)) / peak
        out = out * gain
    after = _band_share(out[:, 0], rate)

    return out.astype(np.float32), {
        "highpass_hz": highpass_hz,
        "gain_db": round(20 * np.log10(max(gain, 1e-12)), 1),
        "speech_band_before": round(before, 3),
        "speech_band_after": round(after, 3),
    }


def _band_share(mono: np.ndarray, rate: int, lo: float = 300.0, hi: float = 3000.0) -> float:
    """Fraction of voiced energy in the speech band. The quality number."""
    W = 4096
    if mono.size < W:
        return 0.0
    spec = np.zeros(W // 2 + 1)
    for i in range(0, mono.size - W, W // 2):
        c = mono[i : i + W]
        if np.sqrt(np.mean(c**2)) < 0.003:
            continue
        spec += np.abs(np.fft.rfft(c * np.hanning(W))) ** 2
    total = spec.sum()
    if total <= 0:
        return 0.0
    f = np.fft.rfftfreq(W, 1.0 / rate)
    return float(spec[(f >= lo) & (f < hi)].sum() / total)


# --------------------------------------------------------------------------
# packet-loss concealment
# --------------------------------------------------------------------------


def find_dropouts(mono: np.ndarray, min_frames: int = 1) -> list[tuple[int, int]]:
    """Runs of exact digital zero, as (start, length). Real audio is never
    exactly zero for long, so these are lost packets, not quiet passages."""
    if mono.size == 0:
        return []
    z = mono == 0.0
    if not z.any():
        return []
    edges = np.diff(np.concatenate(([0], z.view(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [(int(a), int(b - a)) for a, b in zip(starts, ends) if b - a >= min_frames]


def conceal_dropouts(data: np.ndarray, rate: int, max_conceal_ms: float = 120.0,
                     ramp_ms: float = 8.0, fill_gain: float = 0.65) -> tuple[np.ndarray, dict]:
    """
    Fill Bluetooth packet loss so it stops sounding like clicks.

    A lost packet arrives as a run of exact zeros. The information is gone
    either way, but the *sound* of the loss depends entirely on the edges: a
    waveform cut to zero mid-cycle is a step discontinuity, and a step is a
    click. Dozens of those per minute is what "garbled" means.

    Short gaps are bridged by cross-fading the audio either side of the hole,
    which produces plausible texture at the right level. Long gaps cannot be
    invented, so those keep their silence and only get raised-cosine ramps at
    the edges — you still lose the words, but you do not get a click announcing
    it. Measured here: 81% of real gaps were under 50 ms and fully bridgeable.

    Returns (processed, stats).
    """
    if data.size == 0:
        return data, {}
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    out = data.copy()
    n, channels = out.shape

    max_frames = int(max_conceal_ms * rate / 1000)
    ramp = max(2, int(ramp_ms * rate / 1000))
    gaps = find_dropouts(out[:, 0], min_frames=max(2, int(0.002 * rate)))

    bridged = ramped = 0
    for start, length in gaps:
        end = start + length
        if start == 0 or end >= n:
            continue  # leading/trailing silence is not packet loss

        if length <= max_frames:
            take = min(length, start, n - end)
            if take < 2:
                continue
            # Mirror the neighbours into the hole. A reversed segment begins on
            # exactly the sample the audio left off at, so the splice is
            # continuous by construction — splicing the neighbour forwards
            # instead lands on an arbitrary phase and clicks just as loudly as
            # the silence it replaced.
            pre_rev = out[start - take : start][::-1]
            post_rev = out[end : end + take][::-1]
            if take != length:
                idx = np.linspace(0, take - 1, length)
                base = np.arange(take)
                pre_rev = np.stack([np.interp(idx, base, pre_rev[:, c])
                                    for c in range(channels)], axis=1)
                post_rev = np.stack([np.interp(idx, base, post_rev[:, c])
                                     for c in range(channels)], axis=1)
            fade = np.linspace(1.0, 0.0, length, dtype=np.float32).reshape(-1, 1)
            # Duck the middle of the fill, but keep unity at both edges so the
            # gain shaping cannot reintroduce a step.
            duck = (1.0 - (1.0 - fill_gain)
                    * np.sin(np.linspace(0, np.pi, length, dtype=np.float32))).reshape(-1, 1)
            out[start:end] = ((pre_rev * fade + post_rev * (1 - fade)) * duck).astype(np.float32)
            bridged += 1
        else:
            # Too long to invent. Ramp the edges so the silence arrives quietly.
            r = min(ramp, length // 2, start, n - end)
            if r >= 2:
                win = np.cos(np.linspace(0, np.pi / 2, r, dtype=np.float32)) ** 2
                out[start - r : start] *= win.reshape(-1, 1)
                out[end : end + r] *= win[::-1].reshape(-1, 1)
            ramped += 1

    total = sum(l for _, l in gaps)
    return out, {
        "dropouts": len(gaps),
        "dropout_pct": round(total / n * 100, 2) if n else 0.0,
        "bridged": bridged,
        "ramped": ramped,
    }
