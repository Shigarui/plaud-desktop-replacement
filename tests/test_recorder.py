"""Recorder tests. No audio hardware required — streams are driven by hand."""

import wave
from pathlib import Path

import numpy as np
import pytest

from meetingrec.recorder import DEFAULT_RATE, _Stream, mixdown, to_int16


def tone(frames: int, channels: int = 1, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0, frames / DEFAULT_RATE, frames, endpoint=False)
    sig = (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return np.repeat(sig.reshape(-1, 1), channels, axis=1)


def read(path):
    with wave.open(str(path), "rb") as w:
        return w.getnframes(), w.getnchannels(), w.getframerate()


def test_to_int16_clips_instead_of_wrapping():
    loud = np.array([[2.0], [-2.0]], dtype=np.float32)
    out = to_int16(loud)
    assert out.max() == 32767 and out.min() == -32767


def test_ring_buffer_drops_oldest_when_idle():
    s = _Stream("mic", channels=1, block=4800, preroll_seconds=1.0)  # 10 blocks
    for _ in range(25):
        s._handle(tone(4800), 0.0)
    assert len(s._ring) == 10
    assert s.stats.frames_written == 0  # nothing written while idle


def test_arm_flushes_preroll(tmp_path):
    s = _Stream("mic", channels=1, block=4800, preroll_seconds=1.0)
    for i in range(5):
        s._handle(tone(4800), i * 0.1)
    path = tmp_path / "m.wav"
    preroll = s.arm(path)
    assert preroll == pytest.approx(0.5, abs=0.01)  # 5 * 4800 frames
    assert len(s._ring) == 0
    s.disarm()
    frames, ch, rate = read(path)
    assert frames == 24000 and ch == 1 and rate == DEFAULT_RATE


def test_drift_padding_inserts_silence(tmp_path):
    """A stream running slow gets padded so it stays on wall time."""
    s = _Stream("sys", channels=2, block=4800, preroll_seconds=0.1)
    s.arm(tmp_path / "s.wav")
    # Feed 0.1s of audio but claim 1.0s of wall time has passed.
    s._handle(tone(4800, 2), 0.0)
    s._handle(tone(4800, 2), 1.0)
    s.disarm()
    assert s.stats.frames_padded > 0
    frames, _, _ = read(tmp_path / "s.wav")
    assert frames == pytest.approx(DEFAULT_RATE * 1.1, rel=0.05)


def test_no_padding_when_on_time(tmp_path):
    s = _Stream("mic", channels=1, block=4800, preroll_seconds=0.1)
    s.arm(tmp_path / "m.wav")
    for i in range(5):
        s._handle(tone(4800), i * 0.1)  # exactly real-time
    s.disarm()
    assert s.stats.frames_padded == 0


def test_padding_is_capped(tmp_path):
    """A 10-minute device stall must not write 10 minutes of silence."""
    s = _Stream("mic", channels=1, block=4800, preroll_seconds=0.1)
    s.arm(tmp_path / "m.wav")
    s._handle(tone(4800), 0.0)
    s._handle(tone(4800), 600.0)
    s.disarm()
    assert s.stats.frames_padded <= DEFAULT_RATE * 5


def test_mono_mic_upmixes_on_write(tmp_path):
    s = _Stream("sys", channels=2, block=4800, preroll_seconds=0.1)
    s.arm(tmp_path / "s.wav")
    s._handle(tone(4800, 1), 0.0)  # mono block into a stereo stream
    s.disarm()
    _, ch, _ = read(tmp_path / "s.wav")
    assert ch == 2


def test_mixdown_pads_shorter_track(tmp_path):
    def write(path, frames, ch):
        with wave.open(str(path), "wb") as w:
            w.setnchannels(ch)
            w.setsampwidth(2)
            w.setframerate(DEFAULT_RATE)
            w.writeframes(to_int16(tone(frames, ch)).tobytes())

    write(tmp_path / "mic.wav", 48000, 1)
    write(tmp_path / "sys.wav", 24000, 2)  # half as long
    out = mixdown(tmp_path / "mic.wav", tmp_path / "sys.wav", tmp_path / "mix.wav")
    frames, ch, _ = read(out)
    assert frames == 48000 and ch == 2


def test_mixdown_does_not_clip(tmp_path):
    def write(path, ch, amp):
        with wave.open(str(path), "wb") as w:
            w.setnchannels(ch)
            w.setsampwidth(2)
            w.setframerate(DEFAULT_RATE)
            w.writeframes(to_int16(tone(48000, ch, amp)).tobytes())

    write(tmp_path / "mic.wav", 1, 0.99)
    write(tmp_path / "sys.wav", 2, 0.99)
    out = mixdown(tmp_path / "mic.wav", tmp_path / "sys.wav", tmp_path / "mix.wav")
    with wave.open(str(out), "rb") as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    assert np.abs(data).max() <= 32767
    # Clipping looks like flat tops: long runs of samples pinned at the rail.
    # A normalized sine touches the peak only briefly.
    from itertools import groupby

    at_rail = np.abs(data) == np.abs(data).max()
    longest_run = max((sum(1 for _ in g) for k, g in groupby(at_rail) if k), default=0)
    assert longest_run <= 4, f"flat-topped for {longest_run} samples — signal is clipped"


def test_mixdown_resamples_to_the_higher_rate(tmp_path):
    """A 16 kHz Bluetooth mic track must not drag 48 kHz system audio down."""
    def write(path, rate, seconds, ch):
        with wave.open(str(path), "wb") as w:
            w.setnchannels(ch)
            w.setsampwidth(2)
            w.setframerate(rate)
            frames = int(rate * seconds)
            t = np.linspace(0, seconds, frames, endpoint=False)
            sig = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            w.writeframes(to_int16(np.repeat(sig.reshape(-1, 1), ch, axis=1)).tobytes())

    write(tmp_path / "mic.wav", 16000, 1.0, 1)   # HFP Bluetooth
    write(tmp_path / "sys.wav", 48000, 1.0, 2)   # loopback
    out = mixdown(tmp_path / "mic.wav", tmp_path / "sys.wav", tmp_path / "mix.wav")
    frames, ch, rate = read(out)
    assert rate == 48000, "mixed down to the lower rate"
    assert ch == 2
    assert frames == pytest.approx(48000, rel=0.01)


def test_resample_preserves_duration_and_shape():
    from meetingrec.audio import resample

    src = np.zeros((16000, 1), dtype=np.float32)
    out = resample(src, 16000, 48000)
    assert out.shape == (48000, 1)
    assert resample(src, 16000, 16000) is src  # no-op stays a no-op
    assert resample(np.zeros((0, 1), dtype=np.float32), 16000, 48000).shape[0] == 0


# --- device fallback -------------------------------------------------------


class _FakeReader:
    """Shaped like meetingrec.audio readers: context manager + read()."""

    def __init__(self, name, fail=False, rate=48000, channels=1):
        self.device = name
        self.fail = fail
        self.rate = rate
        self.channels = channels
        self.entered = False

    def __enter__(self):
        if self.fail:
            raise AssertionError()  # exactly how soundcard rejects a BT mic
        self.entered = True
        return self

    def __exit__(self, *a):
        self.entered = False
        return False

    def read(self, frames):
        return np.zeros((frames, self.channels), dtype=np.float32)


def test_falls_back_past_broken_device(monkeypatch):
    s = _Stream("mic", channels=1, block=1024, preroll_seconds=0.1)
    readers = [_FakeReader("Poly BT700", fail=True), _FakeReader("Razer Kiyo Pro")]
    monkeypatch.setattr(s, "_readers", lambda: readers)
    r = s._open_recorder()
    r.__exit__(None, None, None)
    assert s.stats.device == "Razer Kiyo Pro"
    assert s.stats.fallback is True


def test_no_fallback_flag_when_preferred_works(monkeypatch):
    s = _Stream("mic", channels=1, block=1024, preroll_seconds=0.1)
    monkeypatch.setattr(s, "_readers", lambda: [_FakeReader("Razer Kiyo Pro")])
    s._open_recorder().__exit__(None, None, None)
    assert s.stats.fallback is False


def test_adopts_device_native_rate(monkeypatch):
    """A 16 kHz HFP mic must not be written into a 48 kHz WAV header."""
    s = _Stream("mic", rate=48000, channels=1, block=1024, preroll_seconds=0.1)
    monkeypatch.setattr(s, "_readers",
                        lambda: [_FakeReader("Bluetooth HFP", rate=16000)])
    s._open_recorder().__exit__(None, None, None)
    assert s.rate == 16000
    assert s.stats.rate == 16000


def test_all_devices_broken_reports_each(monkeypatch):
    s = _Stream("mic", channels=1, block=1024, preroll_seconds=0.1)
    monkeypatch.setattr(s, "_readers",
                        lambda: [_FakeReader("A", True), _FakeReader("B", True)])
    with pytest.raises(RuntimeError, match="no usable mic device"):
        s._open_recorder()


def test_empty_device_list_raises(monkeypatch):
    s = _Stream("mic", channels=1, block=1024, preroll_seconds=0.1)
    monkeypatch.setattr(s, "_readers", lambda: [])
    with pytest.raises(RuntimeError, match="no mic devices"):
        s._open_recorder()


def test_rate_change_midstream_conforms_to_wav_header(tmp_path, monkeypatch):
    """Device swapped mid-meeting: the file must stay at one rate."""
    s = _Stream("mic", rate=48000, channels=1, block=4800, preroll_seconds=0.1)
    s.arm(tmp_path / "m.wav")
    s._handle(tone(4800), 0.0)
    s.rate = 16000  # device came back at a different rate
    s._handle(np.zeros((1600, 1), dtype=np.float32), 0.1)
    s.disarm()
    frames, _, rate = read(tmp_path / "m.wav")
    assert rate == 48000
    assert frames == pytest.approx(9600, rel=0.05)  # 1600 @16k -> 4800 @48k


# --- honest sample rates ---------------------------------------------------


def _write_wav(path, rate, seconds, ch, maker):
    frames = int(rate * seconds)
    t = np.linspace(0, seconds, frames, endpoint=False)
    sig = maker(t).astype(np.float32)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(to_int16(np.repeat(sig.reshape(-1, 1), ch, axis=1)).tobytes())


def test_narrowband_track_is_rewritten_at_true_rate(tmp_path):
    """A 48 kHz file holding only telephone-band speech should become 16 kHz."""
    from meetingrec.recorder import rewrite_at_true_rate

    p = tmp_path / "mic.wav"
    _write_wav(p, 48000, 2.0, 1,
               lambda t: 0.4 * np.sin(2 * np.pi * 500 * t) + 0.2 * np.sin(2 * np.pi * 1800 * t))
    before = p.stat().st_size
    rewritten, hf = rewrite_at_true_rate(p)
    assert rewritten and hf < 0.01
    frames, ch, rate = read(p)
    assert rate == 16000
    assert frames == pytest.approx(32000, rel=0.02)  # duration preserved
    assert p.stat().st_size < before / 2


def test_wideband_track_is_left_alone(tmp_path):
    from meetingrec.recorder import rewrite_at_true_rate

    p = tmp_path / "mic.wav"
    rng = np.random.default_rng(0)
    _write_wav(p, 48000, 2.0, 1, lambda t: 0.3 * rng.standard_normal(len(t)))
    rewritten, hf = rewrite_at_true_rate(p)
    assert not rewritten and hf > 0.01
    assert read(p)[2] == 48000


def test_silent_track_is_not_touched(tmp_path):
    """No voiced windows means nothing to judge — don't guess."""
    from meetingrec.recorder import rewrite_at_true_rate

    p = tmp_path / "mic.wav"
    _write_wav(p, 48000, 2.0, 1, lambda t: np.zeros(len(t)))
    rewritten, _ = rewrite_at_true_rate(p)
    assert not rewritten
    assert read(p)[2] == 48000


def test_downsample_suppresses_aliasing():
    """Decimating without a filter folds high tones down; ours must not."""
    from meetingrec.audio import downsample, high_frequency_fraction

    sr, secs = 48000, 1.0
    t = np.linspace(0, secs, int(sr * secs), endpoint=False)
    # 15 kHz tone is above the 8 kHz Nyquist of the target rate
    sig = (0.5 * np.sin(2 * np.pi * 15000 * t)).astype(np.float32).reshape(-1, 1)
    out = downsample(sig, sr, 16000).ravel()
    assert out.shape[0] == pytest.approx(16000, rel=0.01)
    # Measure the steady state. The first and last few hundred samples carry a
    # boundary transient from the pad, which a pure full-amplitude sine
    # exaggerates and real audio does not.
    interior = out[200:-200]
    assert np.abs(interior).max() < 0.001, "15 kHz tone folded down into the audible band"


# --- speech enhancement ----------------------------------------------------


def test_highpass_removes_rumble_and_keeps_speech():
    from meetingrec.audio import enhance_speech

    sr = 48000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    rumble = 0.6 * np.sin(2 * np.pi * 30 * t) + 0.3 * np.sin(2 * np.pi * 120 * t)
    speech = 0.15 * np.sin(2 * np.pi * 700 * t) + 0.1 * np.sin(2 * np.pi * 1600 * t)
    mixed = (rumble + speech).astype(np.float32).reshape(-1, 1)

    out, stats = enhance_speech(mixed, sr, highpass_hz=250.0)
    assert stats["speech_band_before"] < 0.25, "test signal should start rumble-dominated"
    assert stats["speech_band_after"] > 0.80, "high-pass should leave speech dominant"
    assert out.shape == mixed.shape


def test_enhance_normalises_to_target_peak():
    from meetingrec.audio import enhance_speech

    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    quiet = (0.01 * np.sin(2 * np.pi * 700 * t)).astype(np.float32).reshape(-1, 1)
    out, stats = enhance_speech(quiet, sr, target_peak_dbfs=-3.0)
    peak_db = 20 * np.log10(np.abs(out).max())
    assert peak_db == pytest.approx(-3.0, abs=0.5)
    assert stats["gain_db"] > 20  # it was very quiet


def test_enhance_handles_silence_without_dividing_by_zero():
    from meetingrec.audio import enhance_speech

    out, stats = enhance_speech(np.zeros((48000, 1), dtype=np.float32), 48000)
    assert np.all(out == 0)
    assert stats["gain_db"] == 0.0


def test_clean_mic_track_writes_raw_copy_when_asked(tmp_path):
    from meetingrec.recorder import clean_mic_track

    p = tmp_path / "mic.wav"
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sig = (0.5 * np.sin(2 * np.pi * 40 * t) + 0.1 * np.sin(2 * np.pi * 900 * t)).astype(np.float32)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(to_int16(sig.reshape(-1, 1)).tobytes())

    stats = clean_mic_track(p, highpass_hz=250.0, keep_raw=True)
    assert (tmp_path / "mic-raw.wav").exists()
    assert stats["speech_band_after"] > stats["speech_band_before"]
    assert read(p)[2] == sr  # rate untouched by cleaning


# --- packet-loss concealment -----------------------------------------------


def test_find_dropouts_locates_zero_runs():
    from meetingrec.audio import find_dropouts

    x = np.ones(1000, dtype=np.float32)
    x[100:150] = 0.0
    x[700:705] = 0.0
    gaps = find_dropouts(x, min_frames=2)
    assert gaps == [(100, 50), (700, 5)]


def test_conceal_bridges_short_gaps_and_removes_clicks():
    from meetingrec.audio import conceal_dropouts

    sr = 16000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    sig = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32).reshape(-1, 1)
    # punch out 12 ms mid-waveform, the median real gap
    gap = slice(int(sr * 0.5), int(sr * 0.5) + int(sr * 0.012))
    holed = sig.copy()
    holed[gap] = 0.0

    clicks_before = int(np.sum(np.abs(np.diff(holed.ravel())) > 0.25))
    fixed, stats = conceal_dropouts(holed, sr)
    clicks_after = int(np.sum(np.abs(np.diff(fixed.ravel())) > 0.25))

    assert stats["dropouts"] == 1 and stats["bridged"] == 1
    assert clicks_before > 0
    assert clicks_after == 0, "bridging should leave no step discontinuity"
    assert not np.any(fixed[gap] == 0.0), "gap should be filled"


def test_conceal_ramps_long_gaps_instead_of_inventing_audio():
    from meetingrec.audio import conceal_dropouts

    sr = 16000
    t = np.linspace(0, 3.0, sr * 3, endpoint=False)
    sig = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32).reshape(-1, 1)
    gap = slice(sr, sr + int(sr * 0.5))  # 500 ms — real speech, unrecoverable
    holed = sig.copy()
    holed[gap] = 0.0

    fixed, stats = conceal_dropouts(holed, sr)
    assert stats["ramped"] == 1 and stats["bridged"] == 0
    # middle of the gap stays silent — we don't fabricate half a second
    mid = (gap.start + gap.stop) // 2
    assert np.all(fixed[mid - 100 : mid + 100] == 0.0)
    assert int(np.sum(np.abs(np.diff(fixed.ravel())) > 0.25)) == 0


def test_conceal_ignores_leading_and_trailing_silence():
    from meetingrec.audio import conceal_dropouts

    sr = 16000
    x = np.zeros((sr, 1), dtype=np.float32)
    x[sr // 3 : 2 * sr // 3] = 0.4
    _, stats = conceal_dropouts(x, sr)
    assert stats["bridged"] == 0 and stats["ramped"] == 0


def test_arm_preserves_negotiated_rate_in_stats(tmp_path, monkeypatch):
    """The summary must report the rate the device gave, not the one we asked for."""
    s = _Stream("mic", rate=48000, channels=1, block=1024, preroll_seconds=0.1)
    monkeypatch.setattr(s, "_readers", lambda: [_FakeReader("BT HFP", rate=16000)])
    s._open_recorder().__exit__(None, None, None)
    s.arm(tmp_path / "m.wav")
    s.disarm()
    assert s.stats.rate == 16000
    assert s.stats.device == "BT HFP"


# --- lazy mic opening ------------------------------------------------------


def test_lazy_stream_does_not_open_device_until_armed(tmp_path, monkeypatch):
    """The whole point: a Bluetooth headset must stay in A2DP until a meeting."""
    import time as _time

    reader = _FakeReader("BT HFP", rate=16000)
    s = _Stream("mic", rate=48000, channels=1, block=256, preroll_seconds=0.0, lazy=True)
    monkeypatch.setattr(s, "_readers", lambda: [reader])
    s.start()
    try:
        _time.sleep(0.4)
        assert not reader.entered, "device opened while idle"
        assert not s._opened.is_set()

        s.arm(tmp_path / "m.wav")
        assert reader.entered, "device should open on arm"
        assert s.stats.rate == 16000

        s.disarm()
        _time.sleep(0.4)
        assert not reader.entered, "device should be released on disarm"
    finally:
        s.stop()
        s.join(timeout=2)


def test_lazy_arm_writes_header_at_negotiated_rate(tmp_path, monkeypatch):
    import time as _time

    s = _Stream("mic", rate=48000, channels=1, block=256, preroll_seconds=0.0, lazy=True)
    monkeypatch.setattr(s, "_readers", lambda: [_FakeReader("BT HFP", rate=16000)])
    s.start()
    try:
        _time.sleep(0.3)
        s.arm(tmp_path / "m.wav")
        _time.sleep(0.2)
        s.disarm()
    finally:
        s.stop()
        s.join(timeout=2)
    assert read(tmp_path / "m.wav")[2] == 16000


def test_lazy_arm_survives_a_device_that_never_opens(tmp_path, monkeypatch):
    """A mic that fails at meeting time must not take the session down."""
    import time as _time

    s = _Stream("mic", rate=48000, channels=1, block=256, preroll_seconds=0.0,
                lazy=True, open_timeout=0.3)
    monkeypatch.setattr(s, "_readers", lambda: [_FakeReader("broken", fail=True)])
    s.start()
    try:
        _time.sleep(0.2)
        s.arm(tmp_path / "m.wav")  # must return, not hang
        s.disarm()
    finally:
        s.stop()
        s.join(timeout=2)
    assert (tmp_path / "m.wav").exists()
    assert read(tmp_path / "m.wav")[2] == 48000  # fell back to requested rate


def test_non_lazy_stream_opens_immediately(tmp_path, monkeypatch):
    import time as _time

    reader = _FakeReader("USB mic")
    s = _Stream("sys", rate=48000, channels=1, block=256, preroll_seconds=0.1, lazy=False)
    monkeypatch.setattr(s, "_readers", lambda: [reader])
    s.start()
    try:
        _time.sleep(0.4)
        assert reader.entered, "non-lazy stream should open at startup"
    finally:
        s.stop()
        s.join(timeout=2)


# --- mic priority list -----------------------------------------------------


def test_priority_list_orders_configured_devices_first(monkeypatch):
    from meetingrec import audio

    devices = [(0, "Microphone (Razer Kiyo Pro)", 2, 48000),
               (1, "Headset (soundcore Liberty 5 Pro Max)", 1, 48000),
               (2, "Headset Microphone (Poly BT700)", 1, 48000)]
    monkeypatch.setattr(audio, "list_input_devices", lambda *a, **k: devices)
    order = audio.find_input_devices(["Poly BT700", "soundcore"])
    assert [n for _, n in order][:2] == ["Headset Microphone (Poly BT700)",
                                         "Headset (soundcore Liberty 5 Pro Max)"]
    # everything else still reachable, but only after the named ones
    assert "Microphone (Razer Kiyo Pro)" in [n for _, n in order]


def test_unknown_configured_name_is_skipped_not_fatal(monkeypatch):
    from meetingrec import audio

    devices = [(0, "Microphone (Razer Kiyo Pro)", 2, 48000)]
    monkeypatch.setattr(audio, "list_input_devices", lambda *a, **k: devices)
    order = audio.find_input_devices(["Nonexistent Headset", "Kiyo"])
    assert order[0][1] == "Microphone (Razer Kiyo Pro)"


def test_no_devices_returns_empty(monkeypatch):
    from meetingrec import audio

    monkeypatch.setattr(audio, "list_input_devices", lambda *a, **k: [])
    assert audio.find_input_devices(["anything"]) == []


def test_readers_builds_without_monkeypatching_itself(monkeypatch):
    """
    Exercise the real _readers() path.

    Every other lazy/fallback test patches _readers, which means a missing
    import inside it sails through the whole suite and only fails at record
    time. This one calls it for real with the device list stubbed.
    """
    from meetingrec import audio

    monkeypatch.setattr(audio, "list_input_devices",
                        lambda *a, **k: [(0, "Headset Microphone (Poly BT700)", 1, 48000),
                                         (1, "Headset (soundcore Liberty 5 Pro Max)", 1, 48000)])
    s = _Stream("mic", rate=48000, channels=1, block=1024, preroll_seconds=0.0,
                device_name=["Poly", "soundcore"], lazy=True)
    readers = s._readers()
    assert [r.device for r in readers][:2] == ["Headset Microphone (Poly BT700)",
                                               "Headset (soundcore Liberty 5 Pro Max)"]


def test_readers_accepts_a_bare_string_too(monkeypatch):
    from meetingrec import audio

    monkeypatch.setattr(audio, "list_input_devices",
                        lambda *a, **k: [(0, "Microphone (Razer Kiyo Pro)", 2, 48000)])
    s = _Stream("mic", rate=48000, channels=1, block=1024, preroll_seconds=0.0,
                device_name="Kiyo")
    assert s._readers()[0].device == "Microphone (Razer Kiyo Pro)"


def test_default_block_is_small_enough_for_bluetooth():
    """
    Guard rail for a measured finding.

    4096 frames produced 9-26% digital silence on a Bluetooth headset mic;
    1024 produced 0.2%. If someone raises this back up for throughput reasons,
    the dropouts come back and look exactly like packet loss.
    """
    from meetingrec.recorder import DEFAULT_BLOCK, DEFAULT_RATE

    ms = DEFAULT_BLOCK / DEFAULT_RATE * 1000
    assert 10 <= ms <= 30, f"block is {ms:.0f} ms — outside the measured safe range"


def test_block_frames_reaches_both_streams():
    from meetingrec.recorder import DualRecorder

    rec = DualRecorder(Path("/tmp/meetingrec-test"), block_frames=512)
    assert rec.mic.block == 512 and rec.system.block == 512


def test_lazy_arm_waits_for_warmup_before_opening_file(tmp_path, monkeypatch):
    """
    The mic must settle before the file opens.

    A freshly opened Bluetooth stream drops ~20% of its first seconds; lazy
    opening otherwise puts that window at the start of every meeting.
    """
    import time as _time

    s = _Stream("mic", rate=48000, channels=1, block=256, preroll_seconds=1.0,
                lazy=True, warmup_seconds=0.4)
    monkeypatch.setattr(s, "_readers", lambda: [_FakeReader("BT", rate=48000)])
    s.start()
    try:
        _time.sleep(0.2)
        t0 = _time.monotonic()
        s.arm(tmp_path / "m.wav")
        elapsed = _time.monotonic() - t0
        assert elapsed >= 0.35, f"arm returned in {elapsed:.2f}s — warm-up skipped"
        s.disarm()
    finally:
        s.stop()
        s.join(timeout=2)


def test_warmup_audio_goes_to_preroll_not_the_bin(tmp_path, monkeypatch):
    """Warm-up audio is ragged, not worthless — it should land in pre-roll."""
    import time as _time

    s = _Stream("mic", rate=48000, channels=1, block=4800, preroll_seconds=2.0,
                lazy=True, warmup_seconds=0.3)
    monkeypatch.setattr(s, "_readers", lambda: [_FakeReader("BT", rate=48000)])
    s.start()
    try:
        _time.sleep(0.2)
        preroll = s.arm(tmp_path / "m.wav")
        s.disarm()
    finally:
        s.stop()
        s.join(timeout=2)
    assert preroll > 0, "warm-up audio was discarded instead of buffered"


def test_non_lazy_stream_has_no_warmup_delay(tmp_path, monkeypatch):
    import time as _time

    s = _Stream("sys", rate=48000, channels=2, block=256, preroll_seconds=0.1,
                lazy=False, warmup_seconds=0.0)
    monkeypatch.setattr(s, "_readers", lambda: [_FakeReader("loopback", channels=2)])
    s.start()
    try:
        _time.sleep(0.3)
        t0 = _time.monotonic()
        s.arm(tmp_path / "s.wav")
        assert _time.monotonic() - t0 < 0.2
        s.disarm()
    finally:
        s.stop()
        s.join(timeout=2)


# --- output endpoint switching ---------------------------------------------


def test_loopback_reopens_when_default_output_changes(monkeypatch):
    """
    A Bluetooth headset moves playback to its HFP endpoint when a call starts.
    Capturing the old A2DP endpoint records silence for the whole meeting —
    the exact window the recording exists for.
    """
    import time as _time
    from meetingrec import recorder as rec_mod

    opened: list[str] = []
    current = {"name": "Headphones (soundcore)"}

    class _Reader(_FakeReader):
        def __enter__(self):
            opened.append(self.device)
            return super().__enter__()

    def readers():
        return [_Reader(current["name"], channels=2)]

    s = _Stream("system", rate=48000, channels=2, block=256,
                preroll_seconds=0.5, loopback=True)
    s._switch_check_interval = 0.05
    monkeypatch.setattr(s, "_readers", readers)
    monkeypatch.setattr(rec_mod, "_default_speaker_name", lambda: current["name"])

    s.start()
    try:
        _time.sleep(0.3)
        assert opened == ["Headphones (soundcore)"]
        # Teams starts: Windows moves playback to the HFP endpoint
        current["name"] = "Headset Earphone (soundcore)"
        _time.sleep(0.6)
        assert "Headset Earphone (soundcore)" in opened, "did not follow the switch"
    finally:
        s.stop()
        s.join(timeout=2)


def test_mic_stream_does_not_chase_output_changes(monkeypatch):
    """Only loopback follows the default output; the mic must stay put."""
    import time as _time
    from meetingrec import recorder as rec_mod

    opened: list[str] = []

    class _Reader(_FakeReader):
        def __enter__(self):
            opened.append(self.device)
            return super().__enter__()

    s = _Stream("mic", rate=48000, channels=1, block=256, preroll_seconds=0.5)
    s._switch_check_interval = 0.05
    monkeypatch.setattr(s, "_readers", lambda: [_Reader("Headset (soundcore)")])
    monkeypatch.setattr(rec_mod, "_default_speaker_name", lambda: "Something Else")
    s.start()
    try:
        _time.sleep(0.4)
        assert len(opened) == 1, "mic stream reopened on an output change"
    finally:
        s.stop()
        s.join(timeout=2)


def test_mme_is_preferred_over_wasapi_for_microphones():
    """
    Measured, not stylistic: the same Bluetooth mic gave 34% speech-band
    energy through WASAPI and 77% through MME, before any processing.
    """
    from meetingrec.audio import MIC_HOSTAPI_RANK

    assert MIC_HOSTAPI_RANK["MME"] < MIC_HOSTAPI_RANK["Windows WASAPI"]
    assert MIC_HOSTAPI_RANK["Windows WDM-KS"] == max(MIC_HOSTAPI_RANK.values())


def test_device_list_picks_the_mme_duplicate(monkeypatch):
    # sounddevice raises OSError (not ImportError) when the PortAudio binary
    # is missing, so importorskip does not catch it.
    try:
        import sounddevice as sd
    except Exception as exc:
        pytest.skip(f"PortAudio unavailable: {exc}")
    from meetingrec import audio

    rows = [{"name": "Headset (soundcore)", "max_input_channels": 1,
             "default_samplerate": 44100, "hostapi": 0},
            {"name": "Headset (soundcore)", "max_input_channels": 1,
             "default_samplerate": 48000, "hostapi": 1}]
    monkeypatch.setattr(sd, "query_devices", lambda *a, **k: rows)
    monkeypatch.setattr(sd, "query_hostapis",
                        lambda *a, **k: [{"name": "MME"}, {"name": "Windows WASAPI"}])
    found = audio.list_input_devices()
    assert len(found) == 1 and found[0][0] == 0, "should collapse onto the MME entry"


def test_mic_rate_override_is_applied(monkeypatch):
    from meetingrec import audio

    monkeypatch.setattr(audio, "list_input_devices",
                        lambda *a, **k: [(0, "Headset (soundcore)", 1, 44100)])
    s = _Stream("mic", rate=48000, channels=1, block=1024, preroll_seconds=0.0,
                device_name="soundcore", mic_rate=16000)
    readers = s._readers()
    assert s.rate == 16000
    assert readers[0].requested_rate == 16000


# --- silence watchdog ------------------------------------------------------


class _SilentReader(_FakeReader):
    """Opens fine, returns pure digital silence — a dongle with the headset off."""

    def read(self, frames):
        return np.zeros((frames, self.channels), dtype=np.float32)


class _LiveReader(_FakeReader):
    def read(self, frames):
        t = np.linspace(0, frames / 48000, frames, endpoint=False)
        sig = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        return np.repeat(sig.reshape(-1, 1), self.channels, axis=1)


def test_silent_device_is_marked_dead_and_skipped(tmp_path, monkeypatch):
    """
    The failure this exists for: a Bluetooth dongle whose headset is switched
    off opens successfully and records 100% zeros. Without a watchdog the app
    reports a successful meeting containing nothing at all.
    """
    import time as _time

    silent = _SilentReader("Poly BT700 (headset off)")
    live = _LiveReader("Headset (soundcore)")
    s = _Stream("mic", rate=48000, channels=1, block=4800, preroll_seconds=0.1)
    s.silence_check_seconds = 0.2  # 0.2s of audio, not wall time
    monkeypatch.setattr(s, "_readers",
                        lambda: [r for r in (silent, live)])
    s.start()
    try:
        s.arm(tmp_path / "m.wav")
        _time.sleep(1.5)
        assert "Poly BT700 (headset off)" in s._bad_devices, "silent device not flagged"
        assert s.stats.device == "Headset (soundcore)", "did not move to a live device"
        s.disarm()
    finally:
        s.stop()
        s.join(timeout=2)


def test_live_device_is_not_flagged(tmp_path, monkeypatch):
    import time as _time

    s = _Stream("mic", rate=48000, channels=1, block=4800, preroll_seconds=0.1)
    s.silence_check_seconds = 0.2
    monkeypatch.setattr(s, "_readers", lambda: [_LiveReader("good mic")])
    s.start()
    try:
        s.arm(tmp_path / "m.wav")
        _time.sleep(1.0)
        assert s._bad_devices == set()
        s.disarm()
    finally:
        s.stop()
        s.join(timeout=2)


def test_watchdog_only_runs_while_armed(tmp_path, monkeypatch):
    """Idle silence is normal — don't condemn a device nobody is recording from."""
    import time as _time

    s = _Stream("mic", rate=48000, channels=1, block=4800, preroll_seconds=1.0)
    s.silence_check_seconds = 0.2
    monkeypatch.setattr(s, "_readers", lambda: [_SilentReader("idle mic")])
    s.start()
    try:
        _time.sleep(1.0)
        assert s._bad_devices == set(), "flagged a device before recording started"
    finally:
        s.stop()
        s.join(timeout=2)


# --- track alignment -------------------------------------------------------


def _make_wav(path, rate, seconds, ch, amp=0.3):
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    sig = (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(to_int16(np.repeat(sig.reshape(-1, 1), ch, axis=1)).tobytes())


def test_pad_front_adds_leading_silence(tmp_path):
    from meetingrec.recorder import pad_front

    p = tmp_path / "a.wav"
    _make_wav(p, 48000, 1.0, 1)
    added = pad_front(p, 0.5)
    frames, ch, rate = read(p)
    assert added == pytest.approx(0.5, abs=0.01)
    assert frames == pytest.approx(48000 * 1.5, rel=0.01)
    with wave.open(str(p), "rb") as w:
        head = np.frombuffer(w.readframes(1000), dtype=np.int16)
    assert np.all(head == 0), "padding is not silent"


def test_align_pads_the_later_track(tmp_path):
    """
    The real bug: system audio carries 30s of pre-roll and the lazy mic ~1.5s,
    so the mic file starts ~28s later in real time and the conversation plays
    back with answers arriving before questions.
    """
    from meetingrec.recorder import align_tracks

    mic, sysp = tmp_path / "mic.wav", tmp_path / "sys.wav"
    _make_wav(mic, 48000, 2.0, 1)
    _make_wav(sysp, 48000, 2.0, 2)
    # system started 28.5s earlier on the shared monotonic clock
    info = align_tracks(mic, 1028.5, sysp, 1000.0)
    assert info["padded"] == "mic"
    assert info["padded_seconds"] == pytest.approx(28.5, abs=0.05)
    assert read(mic)[0] == pytest.approx(48000 * 30.5, rel=0.01)
    assert read(sysp)[0] == pytest.approx(48000 * 2.0, rel=0.01)


def test_align_pads_system_when_it_starts_later(tmp_path):
    from meetingrec.recorder import align_tracks

    mic, sysp = tmp_path / "mic.wav", tmp_path / "sys.wav"
    _make_wav(mic, 48000, 2.0, 1)
    _make_wav(sysp, 48000, 2.0, 2)
    info = align_tracks(mic, 1000.0, sysp, 1003.0)
    assert info["padded"] == "system"
    assert read(sysp)[0] == pytest.approx(48000 * 5.0, rel=0.01)


def test_align_is_a_noop_when_already_together(tmp_path):
    from meetingrec.recorder import align_tracks

    mic, sysp = tmp_path / "mic.wav", tmp_path / "sys.wav"
    _make_wav(mic, 48000, 2.0, 1)
    _make_wav(sysp, 48000, 2.0, 2)
    info = align_tracks(mic, 1000.0, sysp, 1000.005)
    assert info["offset_seconds"] == 0.0
    assert read(mic)[0] == pytest.approx(96000, rel=0.01)


def test_align_skips_when_a_timestamp_is_missing(tmp_path):
    from meetingrec.recorder import align_tracks

    mic = tmp_path / "mic.wav"
    _make_wav(mic, 48000, 1.0, 1)
    assert align_tracks(mic, None, None, None)["aligned"] is False


def test_stream_has_no_method_shadowing_thread_attributes():
    """Same guard as the screenshot capturer — _Stream is a Thread too."""
    import threading

    probe = threading.Thread()
    reserved = {n for n in vars(probe) if n.startswith("_")}
    ours = {n for n in vars(_Stream) if n.startswith("_")
            and callable(getattr(_Stream, n, None))}
    assert not (ours & reserved), f"shadowed by Thread: {sorted(ours & reserved)}"
