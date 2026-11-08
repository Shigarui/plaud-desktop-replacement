"""Transcription tests. No model, no audio — the merge and format logic only."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from transcribe import Segment, find_tracks, meeting_folders, merge, to_markdown


def test_merge_interleaves_speakers_on_one_timeline():
    """
    The whole point of two tracks: attribution is correct by construction,
    so a merge on timestamps reconstructs the conversation in order.
    """
    mic = [Segment("Me", 0.0, 2.0, "how are we on Q3"),
           Segment("Me", 8.0, 9.5, "makes sense")]
    sysa = [Segment("Others", 2.5, 7.0, "tracking about ten percent under")]
    out = merge([mic, sysa])
    assert [s.speaker for s in out] == ["Me", "Others", "Me"]
    assert [s.start for s in out] == [0.0, 2.5, 8.0]


def test_merge_is_stable_when_speakers_overlap():
    """People talk over each other; ordering must stay deterministic."""
    a = [Segment("Me", 5.0, 6.0, "sorry go ahead")]
    b = [Segment("Others", 5.0, 7.0, "I was going to say")]
    assert [s.speaker for s in merge([a, b])] == ["Me", "Others"]


def test_segment_stamp_is_readable():
    assert Segment("Me", 0, 1, "x").stamp == "0:00:00"
    assert Segment("Me", 3661, 3662, "x").stamp == "1:01:01"


def test_markdown_groups_consecutive_lines_by_speaker():
    segs = [Segment("Me", 0, 1, "one"), Segment("Me", 1, 2, "two"),
            Segment("Others", 3, 4, "three")]
    md = to_markdown(segs, "Test", {})
    assert md.count("**Me**") == 1, "should not repeat the speaker header"
    assert "**Others**" in md
    assert "`0:00:03` three" in md


def test_markdown_includes_summary_when_present():
    md = to_markdown([Segment("Me", 0, 1, "hi")], "T", {}, summary="It was short.")
    assert "## Summary" in md and "It was short." in md
    assert md.index("## Summary") < md.index("## Transcript")


def test_markdown_without_summary_has_no_empty_section():
    md = to_markdown([Segment("Me", 0, 1, "hi")], "T", {})
    assert "## Summary" not in md


def test_metadata_line_is_built_from_the_sidecar():
    meta = {"started_at": "2026-10-13T12:53:03", "duration_seconds": 1800,
            "app": "ms-teams.exe"}
    md = to_markdown([Segment("Me", 0, 1, "hi")], "T", meta)
    assert "ms-teams.exe" in md and "30 min" in md


def test_find_tracks_picks_up_both_and_the_sidecar(tmp_path):
    (tmp_path / "x-mic.wav").write_bytes(b"")
    (tmp_path / "x-system.wav").write_bytes(b"")
    (tmp_path / "x-mic.wav.json").write_text(json.dumps({"app": "zoom.exe"}))
    mic, sysa, single, meta = find_tracks(tmp_path)
    assert mic.name == "x-mic.wav"
    assert sysa.name == "x-system.wav"
    assert single is None
    assert meta["app"] == "zoom.exe"


def test_find_tracks_tolerates_a_missing_side(tmp_path):
    (tmp_path / "x-system.wav").write_bytes(b"")
    mic, sysa, single, meta = find_tracks(tmp_path)
    assert mic is None and sysa is not None and single is None


def test_meeting_folders_finds_nested_meetings(tmp_path):
    a = tmp_path / "MACHINE" / "2026" / "10" / "20261024-101500-zoom.exe"
    b = tmp_path / "MACHINE" / "2026" / "10" / "20261025-093000-ms-teams.exe"
    for d in (a, b):
        d.mkdir(parents=True)
        (d / "x-mic.wav").write_bytes(b"")
    (tmp_path / "MACHINE" / "notes.txt").write_text("ignore me")
    found = meeting_folders(tmp_path)
    assert set(found) == {a, b}


def test_meeting_folders_deduplicates_when_both_tracks_exist(tmp_path):
    d = tmp_path / "20261024-101500-zoom.exe"
    d.mkdir()
    (d / "x-mic.wav").write_bytes(b"")
    (d / "x-system.wav").write_bytes(b"")
    assert meeting_folders(tmp_path) == [d]


# --- speaker attribution ---------------------------------------------------

from transcribe import attribute, load_speakers  # noqa: E402


def _speakers():
    return [
        {"label": "Matt_Kaczor", "name": "Matt_Kaczor",
         "turns": [{"start": 0.0, "end": 10.0}, {"start": 30.0, "end": 40.0}]},
        {"label": "Angela_Combs", "name": "Angela_Combs",
         "turns": [{"start": 12.0, "end": 25.0}]},
        {"label": "Speaker 3", "name": None,
         "turns": [{"start": 50.0, "end": 60.0}]},
    ]


def test_system_segments_get_real_names():
    segs = [Segment("Others", 1.0, 5.0, "morning all"),
            Segment("Others", 14.0, 20.0, "on the variance report"),
            Segment("Grayson", 26.0, 28.0, "makes sense")]
    attribute(segs, _speakers())
    assert segs[0].speaker == "Matt_Kaczor"
    assert segs[1].speaker == "Angela_Combs"
    assert segs[2].speaker == "Grayson", "the mic track must not be reassigned"


def test_straddling_segment_goes_to_whoever_held_most_of_it():
    """
    Whisper follows words and the diariser follows energy, so boundaries never
    line up. A segment spanning a handover belongs to whoever said most of it.
    """
    seg = Segment("Others", 8.0, 16.0, "…and over to you")  # 2s Matt, 4s Angela
    attribute([seg], _speakers())
    assert seg.speaker == "Angela_Combs"


def test_unnamed_cluster_keeps_its_placeholder():
    """Better an honest "Speaker 3" than someone else's words under a name."""
    seg = Segment("Others", 52.0, 58.0, "one more thing")
    attribute([seg], _speakers())
    assert seg.speaker == "Speaker 3"


def test_segment_outside_every_turn_is_left_alone():
    seg = Segment("Others", 100.0, 105.0, "stray audio")
    attribute([seg], _speakers())
    assert seg.speaker == "Others"


def test_attribute_without_diarisation_changes_nothing():
    segs = [Segment("Others", 1.0, 5.0, "hello")]
    attribute(segs, [])
    assert segs[0].speaker == "Others"


def test_load_speakers_reads_the_file(tmp_path):
    (tmp_path / "speakers.json").write_text(json.dumps({"speakers": _speakers()}))
    assert len(load_speakers(tmp_path)) == 3


def test_load_speakers_tolerates_no_file(tmp_path):
    assert load_speakers(tmp_path) == []


def test_load_speakers_tolerates_a_corrupt_file(tmp_path):
    (tmp_path / "speakers.json").write_text("{not json")
    assert load_speakers(tmp_path) == []


def test_markdown_adds_a_who_spoke_summary_for_real_meetings():
    segs = [Segment("Grayson", 0, 60, "a"), Segment("Matt_Kaczor", 60, 300, "b"),
            Segment("Angela_Combs", 300, 360, "c")]
    md = to_markdown(segs, "T", {})
    assert "## Who spoke" in md
    assert md.index("Matt_Kaczor") < md.index("Angela_Combs"), "not ordered by talk time"


def test_two_party_transcript_has_no_who_spoke_section():
    """With just you and one other voice the summary adds nothing."""
    segs = [Segment("Grayson", 0, 10, "a"), Segment("Others", 10, 20, "b")]
    assert "## Who spoke" not in to_markdown(segs, "T", {})


def test_human_duration_does_not_round_short_speakers_to_zero():
    from transcribe import human_duration

    assert human_duration(9) == "9s"
    assert human_duration(45) == "45s"
    assert human_duration(60) == "1m"
    assert human_duration(185) == "3m 05s"


def test_who_spoke_shows_share_of_talk_time():
    segs = [Segment("Grayson", 0, 100, "a"), Segment("Matt", 100, 300, "b"),
            Segment("Angela", 300, 400, "c")]
    md = to_markdown(segs, "T", {})
    assert "50%" in md  # Matt holds half
    assert "25%" in md


# --- audio decoding --------------------------------------------------------

import wave as _wave  # noqa: E402

import numpy as np  # noqa: E402

from transcribe import decode_wav  # noqa: E402


def _write(path, seconds=1.0, rate=48000, channels=1, width=2):
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    sig = (0.4 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    with _wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(np.repeat(sig.reshape(-1, 1), channels, axis=1).tobytes())


def test_decode_wav_resamples_to_16k(tmp_path):
    """
    We decode ourselves rather than letting whisper use PyAV: a PyAV release
    changed av.open() and every transcription failed with an unexpected
    keyword. Our files are plain WAV, so the dependency is avoidable.
    """
    p = tmp_path / "a.wav"
    _write(p, seconds=2.0, rate=48000)
    audio = decode_wav(p)
    assert audio is not None
    assert audio.dtype == np.float32
    assert len(audio) == pytest.approx(32000, rel=0.01)


def test_decode_wav_mixes_stereo_down(tmp_path):
    p = tmp_path / "s.wav"
    _write(p, seconds=1.0, rate=16000, channels=2)
    audio = decode_wav(p)
    assert audio.ndim == 1
    assert len(audio) == pytest.approx(16000, rel=0.01)


def test_decode_wav_leaves_16k_mono_alone(tmp_path):
    p = tmp_path / "m.wav"
    _write(p, seconds=1.5, rate=16000)
    assert len(decode_wav(p)) == 24000


def test_decode_wav_returns_none_for_other_formats(tmp_path):
    """Anything unusual falls back to whisper's own decoder."""
    p = tmp_path / "odd.wav"
    _write(p, width=1)  # 8-bit
    assert decode_wav(p) is None

    missing = tmp_path / "nope.wav"
    assert decode_wav(missing) is None

    text = tmp_path / "notaudio.wav"
    text.write_text("hello")
    assert decode_wav(text) is None


def test_decode_wav_output_is_in_range(tmp_path):
    p = tmp_path / "a.wav"
    _write(p)
    audio = decode_wav(p)
    assert float(np.abs(audio).max()) <= 1.0


# --- single-track recordings (phone uploads) --------------------------------


def test_a_lone_recording_is_found_as_a_single_track(tmp_path):
    """
    The phone uploads one file containing everyone, including you — so there
    is no free attribution and every speaker must be named by the diariser.
    """
    (tmp_path / "20261024-143012-plaud.mp3").write_bytes(b"")
    mic, sysa, single, _ = find_tracks(tmp_path)
    assert mic is None and sysa is None
    assert single.name == "20261024-143012-plaud.mp3"


def test_a_desktop_pair_is_not_treated_as_single_track(tmp_path):
    (tmp_path / "x-mic.wav").write_bytes(b"")
    (tmp_path / "x-system.wav").write_bytes(b"")
    assert find_tracks(tmp_path)[2] is None


def test_derived_files_are_not_mistaken_for_the_recording(tmp_path):
    """A mixdown is not a separate recording."""
    (tmp_path / "x-mix.wav").write_bytes(b"")
    (tmp_path / "x-mic-raw.wav").write_bytes(b"")
    assert find_tracks(tmp_path)[2] is None


def test_compressed_uploads_are_discovered(tmp_path):
    # Timestamped names: support folders are deliberately excluded now.
    for name in ("20261024-101500-plaud/rec.opus",
                 "20261025-111000-plaud/rec.m4a",
                 "20261026-162117-plaud/rec.ogg"):
        p = tmp_path / name
        p.parent.mkdir(parents=True)
        p.write_bytes(b"")
    found = meeting_folders(tmp_path)
    assert len(found) == 3


def test_mixdowns_do_not_create_phantom_meetings(tmp_path):
    folder = tmp_path / "20261024-101500-zoom.exe"
    folder.mkdir()
    (folder / "x-mix.wav").write_bytes(b"")
    assert meeting_folders(tmp_path) == []


def test_shared_folders_are_excluded_from_transcription(tmp_path):
    """
    Enrollment samples and dictionaries live in the archive beside the
    recordings. Transcribing a folder of voice profiles is wasted work and
    produces a transcript nobody wants.
    """
    for name in ("_shared/speakers/sample.wav",
                 "_selftest/probe.wav",
                 "DESKTOP-1/2026/10/20261024-101500-zoom.exe/x-system.wav"):
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    found = meeting_folders(tmp_path)
    assert len(found) == 1
    assert found[0].name == "20261024-101500-zoom.exe"


def test_both_scripts_accept_a_date_cutoff():
    """
    Guard against the flag existing in one script and not the other — the
    console sends it to both, and a missing flag is an argparse error that
    aborts the whole run.
    """
    import subprocess
    import sys as _sys
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[1]
    for script in ("diarize.py", "transcribe.py"):
        out = subprocess.run([_sys.executable, str(root / "scripts" / script), "--help"],
                             capture_output=True, text=True, cwd=str(root)).stdout
        assert "--since" in out, f"{script} has no --since"
