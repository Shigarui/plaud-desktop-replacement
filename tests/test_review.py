"""Review-server tests — data handling only, no browser."""

import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import review
from review import (
    enroll_from_meeting,
    library_names,
    meeting_folders,
    meeting_payload,
    safe_folder_name,
    slice_wav,
)

RATE = 16000


def write_wav(path, seconds, rate=RATE, ch=1):
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    sig = (0.4 * np.sin(2 * np.pi * 300 * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.repeat(sig.reshape(-1, 1), ch, axis=1).tobytes())


@pytest.fixture
def archive(tmp_path):
    folder = tmp_path / "archive" / "M" / "2026" / "10" / "20261013-mtg"
    folder.mkdir(parents=True)
    write_wav(folder / "x-system.wav", 60)
    (folder / "x-mic.wav.json").write_text(json.dumps({"app": "ms-teams.exe"}))
    (folder / "speakers.json").write_text(json.dumps({
        "clusters": 2,
        "speakers": [
            {"label": "Matt_Kaczor", "name": "Matt_Kaczor", "confidence": 0.78,
             "z": 8.5, "seconds": 20.0,
             "turns": [{"start": 1.0, "end": 11.0}, {"start": 20.0, "end": 30.0}]},
            {"label": "Speaker 2", "name": None, "confidence": 0.29, "z": 2.9,
             "seconds": 15.0, "turns": [{"start": 35.0, "end": 50.0}]},
        ],
    }))
    profiles = tmp_path / "profiles"
    (profiles / "Matt_Kaczor").mkdir(parents=True)
    (profiles / "Angela_Combs").mkdir(parents=True)
    review.STATE["archive"] = tmp_path / "archive"
    review.STATE["profiles"] = profiles
    return folder


def test_finds_meetings_with_results(archive):
    assert meeting_folders() == [archive]


def test_library_names_come_from_profile_folders(archive):
    assert library_names() == ["Angela_Combs", "Matt_Kaczor"]


def test_payload_carries_what_the_page_needs(archive):
    p = meeting_payload(archive)
    assert p["meta"]["app"] == "ms-teams.exe"
    assert p["duration"] == pytest.approx(60.0, abs=0.1)
    assert len(p["speakers"]) == 2
    assert p["speakers"][1]["name"] is None


def test_slice_wav_cuts_the_requested_span(archive):
    data = slice_wav(archive / "x-system.wav", 10.0, 15.0)
    with wave.open(__import__("io").BytesIO(data), "rb") as w:
        assert w.getnframes() / w.getframerate() == pytest.approx(5.0, abs=0.05)


def test_slice_wav_clamps_a_negative_start(archive):
    data = slice_wav(archive / "x-system.wav", -5.0, 2.0)
    assert len(data) > 44  # more than just a header


def test_safe_folder_name_strips_path_characters():
    assert safe_folder_name("Laura Eich") == "Laura_Eich"
    assert safe_folder_name("../../etc/passwd") == "etcpasswd"
    assert safe_folder_name("") == "Unnamed"
    assert safe_folder_name("!!!") == "Unnamed"


# --- enrollment feedback loop ----------------------------------------------


def test_correcting_a_speaker_writes_a_new_profile_sample(archive):
    """
    The compounding part: a name supplied once becomes an enrollment sample
    recorded under real meeting conditions, so the same person is recognised
    without help next time.
    """
    speaker = json.loads((archive / "speakers.json").read_text())["speakers"][1]
    result = enroll_from_meeting(archive, speaker, "Angela Combs")
    assert result["ok"]
    out = Path(result["path"])
    assert out.exists()
    assert out.parent.name == "Angela_Combs"
    assert result["seconds"] == pytest.approx(15.0, abs=0.1)


def test_enrolled_sample_contains_only_that_speakers_turns(archive):
    speaker = {"turns": [{"start": 1.0, "end": 11.0}, {"start": 20.0, "end": 30.0}]}
    result = enroll_from_meeting(archive, speaker, "Matt_Kaczor")
    with wave.open(result["path"], "rb") as w:
        seconds = w.getnframes() / w.getframerate()
    assert seconds == pytest.approx(20.0, abs=0.2), "picked up audio outside the turns"


def test_enrollment_creates_the_folder_for_a_new_person(archive):
    speaker = {"turns": [{"start": 35.0, "end": 50.0}]}
    result = enroll_from_meeting(archive, speaker, "Brand New Person")
    assert result["ok"]
    assert (review.STATE["profiles"] / "Brand_New_Person").is_dir()


def test_enrollment_without_turns_is_refused(archive):
    assert enroll_from_meeting(archive, {"turns": []}, "X")["ok"] is False


def test_enrollment_without_a_system_track_is_refused(tmp_path):
    review.STATE["profiles"] = tmp_path / "p"
    empty = tmp_path / "empty"
    empty.mkdir()
    result = enroll_from_meeting(empty, {"turns": [{"start": 0, "end": 5}]}, "X")
    assert result["ok"] is False


# --- unnamed voices across meetings ----------------------------------------


def _meeting(root, name, speakers):
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    write_wav(folder / "x-system.wav", 60)
    (folder / "speakers.json").write_text(json.dumps({"speakers": speakers}))
    return folder


def _unknown(label, seconds, vec, turns=None):
    return {"label": label, "name": None, "confidence": 0.3, "z": 2.0,
            "seconds": seconds, "embedding": list(vec),
            "turns": turns or [{"start": 0.0, "end": seconds}]}


@pytest.fixture
def archive_many(tmp_path):
    root = tmp_path / "archive"
    voice_a = [1.0, 0.0, 0.0]
    voice_b = [0.0, 1.0, 0.0]
    _meeting(root, "m1", [_unknown("Speaker 1", 300, voice_a),
                          {"label": "Matt", "name": "Matt", "confidence": .8,
                           "z": 9, "seconds": 100, "turns": []}])
    _meeting(root, "m2", [_unknown("Speaker 2", 200, [0.98, 0.1, 0.0])])
    _meeting(root, "m3", [_unknown("Speaker 1", 150, voice_b)])
    _meeting(root, "m4", [{"label": "Angela", "name": "Angela", "confidence": .7,
                           "z": 5, "seconds": 90, "turns": []}])
    review.STATE["archive"] = root
    review.STATE["profiles"] = tmp_path / "profiles"
    (tmp_path / "profiles").mkdir()
    return root


def test_same_voice_in_two_meetings_groups_together(archive_many):
    """
    The point of the tab: one stranger across nine meetings is someone you
    work with, not nine separate mysteries.
    """
    from review import gather_unknowns

    result = gather_unknowns()
    biggest = result["groups"][0]
    assert biggest["meetings"] == 2
    assert biggest["seconds"] == pytest.approx(500.0)


def test_different_voices_stay_apart(archive_many):
    from review import gather_unknowns

    result = gather_unknowns()
    assert result["stats"]["likely_people"] == 2
    assert result["stats"]["unknown_clusters"] == 3


def test_stats_count_only_affected_meetings(archive_many):
    from review import gather_unknowns

    stats = gather_unknowns()["stats"]
    assert stats["meetings_total"] == 4
    assert stats["meetings_with_unknowns"] == 3
    assert stats["meetings_clean"] == 1
    assert stats["unknown_minutes"] == pytest.approx(10.8, abs=0.1)


def test_named_speakers_are_not_listed(archive_many):
    from review import gather_unknowns

    labels = [m["label"] for g in gather_unknowns()["groups"] for m in g["members"]]
    assert "Matt" not in labels and "Angela" not in labels


def test_groups_are_ordered_by_total_talk_time(archive_many):
    from review import gather_unknowns

    seconds = [g["seconds"] for g in gather_unknowns()["groups"]]
    assert seconds == sorted(seconds, reverse=True)


def test_clusters_without_embeddings_are_not_guessed_at(tmp_path):
    """Results from before centroids were stored must stay separate."""
    from review import gather_unknowns

    root = tmp_path / "a"
    _meeting(root, "m1", [{"label": "Speaker 1", "name": None, "confidence": .3,
                           "z": 2, "seconds": 100, "turns": []}])
    _meeting(root, "m2", [{"label": "Speaker 1", "name": None, "confidence": .3,
                           "z": 2, "seconds": 100, "turns": []}])
    review.STATE["archive"] = root
    review.STATE["profiles"] = tmp_path / "p"
    assert gather_unknowns()["stats"]["likely_people"] == 2


def test_empty_archive_reports_zeroes(tmp_path):
    from review import gather_unknowns

    review.STATE["archive"] = tmp_path
    review.STATE["profiles"] = tmp_path
    stats = gather_unknowns()["stats"]
    assert stats["likely_people"] == 0 and stats["unknown_clusters"] == 0
