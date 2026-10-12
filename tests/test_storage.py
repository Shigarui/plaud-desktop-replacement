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
