"""The master index — what an agent searches instead of opening recordings."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from index import gather, keywords, store_numbers, write_markdown


def meeting(root, name, speakers, classification=None, transcript=None):
    folder = root / "M" / "2026" / "10" / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "speakers.json").write_text(json.dumps({"speakers": speakers}))
    if classification:
        (folder / "classification.json").write_text(json.dumps(classification))
    if transcript:
        (folder / "transcript.md").write_text(transcript, encoding="utf-8")
    return folder


def test_keywords_skip_filler():
    """Handles to search on, not a summary — but filler makes useless handles."""
    text = ("we really think that the inventory variance is going to be "
            "the inventory problem, basically inventory")
    out = keywords(text)
    assert "inventory" in out
    assert "really" not in out and "basically" not in out


def test_store_numbers_are_extracted():
    assert store_numbers("visited Store 64 and store #338 today") == ["338", "64"]
    assert store_numbers("no stores mentioned") == []


def test_gather_reads_what_is_on_disk(tmp_path):
    meeting(tmp_path, "20261013-153011-ms-teams.exe",
            [{"name": "Matt Kaczor", "seconds": 600},
             {"name": None, "label": "Speaker 2", "seconds": 20}],
            {"title": "Oct-13 / Matt Kaczor / Coaching Call",
             "type": "Coaching Call", "category": "Field Team",
             "team_member": "Matt Kaczor", "flags": []},
            "## Summary\n\nTalked about territory coverage.\n")
    rows = gather(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["title"].endswith("Coaching Call")
    assert row["speakers"] == ["Matt Kaczor"]
    assert row["unnamed"] == 1
    assert "territory" in row["summary"].lower()
    assert row["classified"] is True


def test_unclassified_meetings_are_listed_not_invented(tmp_path):
    """
    A meeting with no classification has no title. Making one up here would
    put a confident wrong answer in front of whatever reads this.
    """
    meeting(tmp_path, "20261014-090000-zoom.exe", [{"name": "Dan Baron", "seconds": 300}])
    rows = gather(tmp_path)
    assert rows[0]["classified"] is False
    assert rows[0]["title"] == ""


def test_support_folders_are_excluded(tmp_path):
    folder = tmp_path / "_shared" / "speakers"
    folder.mkdir(parents=True)
    (folder / "speakers.json").write_text(json.dumps({"speakers": []}))
    assert gather(tmp_path) == []


def test_rows_come_back_newest_first(tmp_path):
    for name in ("20261001-090000-a", "20261020-090000-b", "20261010-090000-c"):
        meeting(tmp_path, name, [{"name": "X", "seconds": 100}])
    order = [r["id"][:8] for r in gather(tmp_path)]
    assert order == ["20261020", "20261010", "20261001"]


def test_the_markdown_groups_by_month_and_names_people(tmp_path):
    meeting(tmp_path, "20261013-153011-ms-teams.exe",
            [{"name": "Matt Kaczor", "seconds": 600}],
            {"title": "Oct-13 / Matt Kaczor / Coaching Call",
             "category": "Field Team", "flags": ["Trim"]},
            "## Summary\n\nCoverage and caseload.\n")
    rows = gather(tmp_path)
    out = tmp_path / "INDEX.md"
    write_markdown(rows, out, tmp_path)
    text = out.read_text(encoding="utf-8")
    assert "## October 2026" in text
    assert "Matt Kaczor" in text
    assert "Trim" in text
    assert "# Meeting index" in text


def test_the_index_says_the_transcript_is_the_record(tmp_path):
    """
    An index that reads like a substitute for the transcript invites someone
    to quote it as what was said.
    """
    meeting(tmp_path, "20261013-153011-x", [{"name": "A", "seconds": 10}])
    out = tmp_path / "INDEX.md"
    write_markdown(gather(tmp_path), out, tmp_path)
    assert "transcript is the record" in out.read_text(encoding="utf-8")
