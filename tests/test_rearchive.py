"""Re-archiving stranded recordings — grouping files back into meetings."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from rearchive import meeting_key


def test_every_file_of_one_meeting_groups_together():
    """
    Files are stranded individually but belong to a meeting. Their own names
    carry the answer, which matters because the folder they should have gone
    to was never created.
    """
    names = [
        "20261029-095659-ms-teams.exe-mic.wav",
        "20261029-095659-ms-teams.exe-system.wav",
        "20261029-095659-ms-teams.exe-mix.wav",
        "20261029-095659-ms-teams.exe-mic.wav.json",
        "20261029-095659-ms-teams.exe-shot001.jpg",
        "20261029-095659-ms-teams.exe-shots.json",
    ]
    keys = {meeting_key(n) for n in names}
    assert keys == {("20261029-095659", "ms-teams.exe")}


def test_different_meetings_stay_apart():
    a = meeting_key("20261029-095659-ms-teams.exe-mic.wav")
    b = meeting_key("20261029-133122-zoom.exe-mic.wav")
    assert a != b
    assert a[1] == "ms-teams.exe" and b[1] == "zoom.exe"


def test_a_single_track_upload_is_recognised():
    assert meeting_key("20261025-111000-plaud.mp3") == ("20261025-111000", "plaud")


def test_an_unrecognisable_name_is_left_alone():
    """Guessing a destination for a file we cannot place would scatter it
    somewhere nobody will look."""
    assert meeting_key("notes.txt") is None
    assert meeting_key("README") is None
    assert meeting_key("postprocess-queue.json") is None


def test_raw_mic_track_belongs_to_its_meeting():
    assert meeting_key("20261029-095659-zoom.exe-mic-raw.wav") \
        == ("20261029-095659", "zoom.exe")
