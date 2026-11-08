"""Naming and taxonomy rules — every rule in the spec, exercised."""

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from classify import (
    Classification,
    classify,
    category_for,
    date_segment,
    folder_date,
    is_unnamed,
    load_rules,
    speaker_segment,
    talk_shares,
    team_member_folder,
    type_segment,
)

RULES = load_rules()
ME = RULES["people"]["me"]


def sp(name, seconds):
    return {"label": name, "name": None if name.startswith("Speaker") else name,
            "seconds": seconds, "turns": []}


def make(tmp_path, folder_name, speakers, transcript=""):
    folder = tmp_path / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "speakers.json").write_text(json.dumps({"speakers": speakers}))
    if transcript:
        (folder / "transcript.md").write_text(transcript, encoding="utf-8")
    return folder


# --- segment 1: date --------------------------------------------------------


def test_the_date_comes_from_the_folder_not_today():
    """The rule is explicit: never use today's date."""
    assert date_segment(folder_date(Path("20260901-143000-ms-teams.exe"))) == "Sept-01"
    assert date_segment(folder_date(Path("20260115-090000-zoom.exe"))) == "Jan-15"
    assert date_segment(folder_date(Path("20261103-120000-plaud"))) == "Nov-03"


def test_an_undated_folder_yields_no_date():
    assert folder_date(Path("_shared")) is None
    assert date_segment(None) == ""


# --- segment 2: speakers ----------------------------------------------------


def test_the_user_is_excluded_from_talk_time():
    """
    Percentages are of the time OTHER people spoke. On a call where the user
    holds most of it, a colleague with the remainder is dominant, not minor.
    """
    shares = talk_shares([sp(ME, 700), sp("Matt Kaczor", 300)], ME)
    assert ME not in shares
    assert shares["Matt Kaczor"] == 100.0


def test_one_other_speaker_is_named_in_full():
    result = Classification()
    shares = talk_shares([sp(ME, 500), sp("Matt Kaczor", 500)], ME)
    assert speaker_segment(shares, RULES, result) == "Matt Kaczor"


def test_up_to_three_dominant_speakers_in_talk_time_order():
    result = Classification()
    shares = talk_shares([sp(ME, 100), sp("Dan Baron", 400), sp("Jeff Bennett", 300),
                          sp("Katie Bacon", 200), sp("Adam Chandler", 100)], ME)
    out = speaker_segment(shares, RULES, result)
    assert out == "Dan Baron, Jeff Bennett, Katie Bacon"


def test_a_voice_under_twenty_percent_is_not_named():
    result = Classification()
    shares = talk_shares([sp(ME, 100), sp("Dan Baron", 900), sp("Jeff Bennett", 100)], ME)
    assert speaker_segment(shares, RULES, result) == "Dan Baron"


def test_more_than_four_team_members_becomes_the_team():
    result = Classification()
    speakers = [sp(ME, 100)] + [sp(n, 200) for n in
                                ("Angela Combs", "Adam Chandler", "Dan Baron",
                                 "Matt Kaczor", "Jeff Bennett")]
    shares = talk_shares(speakers, ME)
    assert speaker_segment(shares, RULES, result) == "FBC/OSS Team"


def test_nobody_dominant_uses_the_loudest_and_flags_it():
    """A title still gets made, but a human is told it was a fallback."""
    result = Classification()
    shares = talk_shares([sp(ME, 10)] + [sp(f"P{i}", 100) for i in range(8)], ME)
    out = speaker_segment(shares, RULES, result)
    assert out.startswith("P")
    assert "Flagged, Speaker Audit" in result.flags
    assert result.fallbacks


def test_a_solo_recording_is_notes_to_self():
    result = Classification()
    assert speaker_segment({}, RULES, result) == "Notes to Self"


# --- segment 3: type --------------------------------------------------------


def test_audit_beats_field_visit():
    """Precedence falls out of rule order rather than special cases."""
    result = Classification()
    shares = {"Dan Baron": 100.0}
    text = "we did a store visit and ran the store audit checklist"
    assert type_segment(text, shares, RULES, result) == "Audit"


def test_qbr_beats_ops_meeting():
    result = Classification()
    text = "quarterly business review agenda, operations numbers"
    assert type_segment(text, {"Dan Baron": 100.0}, RULES, result) == "QBR"


def test_coaching_beats_one_on_one():
    result = Classification()
    text = "this is a coaching call about your development plan"
    assert type_segment(text, {"Matt Kaczor": 100.0}, RULES, result) == "Coaching Call"


def test_a_plain_one_to_one_is_one_on_one():
    result = Classification()
    assert type_segment("just catching up", {"Matt Kaczor": 100.0},
                        RULES, result) == "One on One"


def test_a_big_team_call_is_a_group_meeting():
    result = Classification()
    shares = {n: 20.0 for n in ("Angela Combs", "Adam Chandler", "Dan Baron",
                                "Matt Kaczor", "Jeff Bennett")}
    assert type_segment("agenda", shares, RULES, result) == "Group Meeting"


def test_an_unmatched_type_falls_back_and_says_so():
    result = Classification()
    shares = {"A": 40.0, "B": 35.0, "C": 25.0}
    assert type_segment("hello everyone", shares, RULES, result) == "Ops Meeting"
    assert result.fallbacks


# --- categories -------------------------------------------------------------


def test_a_store_number_routes_to_franchisees():
    name, folder = category_for("we visited Store 64 last week",
                                {"Someone": 100.0}, RULES)
    assert folder == "5-Franchisees"


def test_technology_people_route_to_systems():
    name, folder = category_for("", {"Josh Collins": 100.0}, RULES)
    assert folder == "4-Biggby/Systems"


def test_leadership_routes_to_corporate():
    name, folder = category_for("", {"Laura Eich": 100.0}, RULES)
    assert folder == "4-Biggby/Corporate"


def test_a_team_member_routes_to_field_team():
    name, folder = category_for("", {"Dan Baron": 100.0}, RULES)
    assert folder == "3-Field Team"


def test_a_solo_recording_routes_to_notes_to_self():
    name, folder = category_for("reminder to myself", {}, RULES)
    assert folder == "1-Note To Self"


# --- the team-member routine ------------------------------------------------


def test_a_clean_one_to_one_files_under_that_person():
    result = Classification()
    assert team_member_folder({"Dan Baron": 100.0}, RULES, result) == "Dan Baron"


def test_a_quiet_unnamed_voice_does_not_disqualify():
    """The spec's own example: Speaker 3 at 10% or less is ignored."""
    result = Classification()
    assert team_member_folder({"Dan Baron": 92.0, "Speaker 3": 8.0},
                              RULES, result) == "Dan Baron"


def test_a_loud_unnamed_voice_does_disqualify():
    """Above 10% it is a third party, and filing that as a private 1:1 is
    the kind of mistake that matters."""
    result = Classification()
    assert team_member_folder({"Dan Baron": 85.0, "Speaker 3": 15.0},
                              RULES, result) is None
    assert result.fallbacks


def test_two_team_members_is_a_conflict_not_a_guess():
    result = Classification()
    assert team_member_folder({"Dan Baron": 55.0, "Matt Kaczor": 45.0},
                              RULES, result) is None
    assert any("Conflict" in f for f in result.flags)


# --- end to end -------------------------------------------------------------


def test_a_coaching_call_gets_the_full_title(tmp_path):
    folder = make(tmp_path, "20260901-143000-ms-teams.exe",
                  [sp(ME, 400), sp("Matt Kaczor", 600)],
                  "This is a coaching call about your development plan.")
    out = classify(folder, RULES)
    assert out.title == "Sept-01 / Matt Kaczor / Coaching Call"
    assert out.team_member == "Matt Kaczor"
    assert out.folder == "3-Field Team"


def test_a_solo_note_collapses_to_two_segments(tmp_path):
    folder = make(tmp_path, "20261103-081500-plaud", [sp(ME, 90)],
                  "Just wanted to note, don't forget the store order.")
    out = classify(folder, RULES)
    assert out.title == "Nov-03 / Notes to Self"
    assert out.folder == "1-Note To Self"


def test_a_long_recording_is_marked_for_trimming(tmp_path):
    folder = make(tmp_path, "20261020-090000-zoom.exe",
                  [sp(ME, 100), sp("Dan Baron", 900)], "ops meeting agenda")
    out = classify(folder, RULES, duration=4 * 3600)
    assert "Trim" in out.flags


def test_an_unidentified_meeting_is_skipped_not_guessed(tmp_path):
    folder = tmp_path / "20261020-090000-zoom.exe"
    folder.mkdir()
    out = classify(folder, RULES)
    assert out.skipped
    assert out.fallbacks


def test_only_generic_speaker_labels_gets_flagged(tmp_path):
    folder = make(tmp_path, "20261020-090000-zoom.exe",
                  [sp(ME, 100), sp("Speaker 1", 500), sp("Speaker 2", 400)],
                  "ops meeting")
    out = classify(folder, RULES)
    assert "Flagged, Speaker Audit" in out.flags
