"""Storage tests that do not touch Drive."""

import json
from datetime import datetime

from meetingrec.storage import LocalArchive, safe_name, session_subdir


def test_safe_name_strips_problem_characters():
    assert safe_name('Q3 Review: Franchise/Ops <draft>') == "Q3 Review-Franchise-Ops-draft"
    assert safe_name("") == "untitled"


def test_session_subdir_is_machine_first():
    d = datetime(2026, 3, 9, 14, 30)
    assert session_subdir(d, "GTERRY-LT") == "GTERRY-LT/2026/03"
    assert session_subdir(d, "GTERRY-LT", "day") == "GTERRY-LT/2026/03/09"
    assert session_subdir(d, "GTERRY-LT", "flat") == "GTERRY-LT"


def test_local_archive_moves_and_writes_sidecar(tmp_path):
    src = tmp_path / "staging" / "call.wav"
    src.parent.mkdir()
    src.write_bytes(b"RIFF-not-really")

    backend = LocalArchive(tmp_path / "archive")
    result = backend.archive(src, "MACHINE/2026/03", {"app": "zoom.exe"})

    assert result.ok
    dest = tmp_path / "archive" / "MACHINE" / "2026" / "03" / "call.wav"
    assert dest.exists() and not src.exists()
    assert json.loads(dest.with_suffix(".wav.json").read_text())["app"] == "zoom.exe"


def test_local_archive_does_not_overwrite(tmp_path):
    backend = LocalArchive(tmp_path / "archive", write_sidecar=False)
    for _ in range(2):
        src = tmp_path / "call.wav"
        src.write_bytes(b"x")
        backend.archive(src, "", {})
    names = sorted(p.name for p in (tmp_path / "archive").iterdir())
    assert names == ["call-2.wav", "call.wav"]


def test_missing_source_fails_cleanly(tmp_path):
    result = LocalArchive(tmp_path).archive(tmp_path / "nope.wav", "", {})
    assert not result.ok and "missing" in result.error


def test_session_subdir_adds_a_folder_per_meeting():
    """
    With screenshots on, one meeting is three WAVs plus a dozen JPEGs plus
    sidecars. Flat month folders interleave meetings; a slug folder keeps each
    call self-contained.
    """
    from datetime import datetime

    from meetingrec.storage import session_subdir

    d = datetime(2026, 10, 12, 14, 5)
    slug = "20261012-140530-msteams_8wekyb3d8bbwe"
    assert session_subdir(d, "WINMAX2", "month", slug) == \
        "WINMAX2/2026/10/20261012-140530-msteams_8wekyb3d8bbwe"
    # unchanged when no slug is passed
    assert session_subdir(d, "WINMAX2", "month") == "WINMAX2/2026/10"


def test_session_subdir_slug_works_with_every_grouping():
    from datetime import datetime

    from meetingrec.storage import session_subdir

    d = datetime(2026, 10, 12)
    assert session_subdir(d, "M", "day", "mtg").startswith("M/2026/10/12/")
    assert session_subdir(d, "M", "flat", "mtg") == "M/mtg"


def test_session_subdir_sanitises_the_slug():
    from datetime import datetime

    from meetingrec.storage import session_subdir

    out = session_subdir(datetime(2026, 10, 12), "M", "flat", "Q3 Review: Ops/L&D")
    assert "/" not in out.split("M/")[1]
    assert ":" not in out


# --- Drive mounting late at logon -------------------------------------------


def test_drive_is_retried_rather_than_abandoned(tmp_path, monkeypatch):
    """
    The recorder starts shortly after logon, before Drive for Desktop has
    mounted. Resolving once and falling back to local meant a whole day of
    recordings silently never reached Drive.
    """
    from meetingrec import storage as mod

    mounted = {"yet": False}
    real_root = tmp_path / "DriveFS"

    def fake_find():
        return real_root if mounted["yet"] else None

    monkeypatch.setattr(mod, "find_drive_desktop_root", fake_find)
    backend = mod.DeferredDriveArchive(folder="Meeting Recordings")

    assert backend.available() is False, "claimed Drive before it mounted"

    real_root.mkdir()
    mounted["yet"] = True
    assert backend.available() is True, "never retried once Drive appeared"


def test_a_file_is_kept_in_staging_while_drive_is_missing(tmp_path, monkeypatch):
    from meetingrec import storage as mod

    monkeypatch.setattr(mod, "find_drive_desktop_root", lambda: None)
    backend = mod.DeferredDriveArchive()
    src = tmp_path / "a.wav"
    src.write_bytes(b"x")

    result = backend.archive(src, "M/2026/10/mtg", {})
    assert result.ok is False
    assert "staging" in result.error
    assert src.exists(), "the file must survive a failed archive"


def test_the_missing_mount_is_reported_once_not_per_file(tmp_path, monkeypatch, caplog):
    """A 40-screenshot meeting would otherwise write 40 identical errors."""
    import logging

    from meetingrec import storage as mod

    monkeypatch.setattr(mod, "find_drive_desktop_root", lambda: None)
    backend = mod.DeferredDriveArchive()
    src = tmp_path / "a.wav"
    src.write_bytes(b"x")

    with caplog.at_level(logging.WARNING, logger="meetingrec.storage"):
        for _ in range(5):
            backend.archive(src, "M/2026/10/mtg", {})
    warnings = [r for r in caplog.records if "not mounted yet" in r.message]
    assert len(warnings) == 1, f"logged {len(warnings)} times"
