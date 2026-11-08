"""Enrollment tests — parsing and matching maths, no model or audio needed."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from enroll import (
    Profile,
    score_detail,
    cosine,
    identify,
    load_library,
    parse_card,
    parse_index,
    save_library,
    scan,
)


# --- profile cards ---------------------------------------------------------


def test_parse_card_handles_the_formats_these_cards_come_in(tmp_path):
    card = tmp_path / "p.md"
    card.write_text(
        "---\n"
        "name: Laura\n"
        "---\n"
        "# Profile\n"
        "**Role:** Ops Director\n"
        "- Team: Field Operations\n"
        "Notes: joins most Monday calls\n",
        encoding="utf-8")
    out = parse_card(card)
    assert out["name"] == "Laura"
    assert out["role"] == "Ops Director"
    assert out["team"] == "Field Operations"
    assert out["notes"].startswith("joins most")


def test_parse_card_ignores_prose_without_pairs(tmp_path):
    card = tmp_path / "p.md"
    card.write_text("Just some notes about this person.\nNothing structured.\n",
                    encoding="utf-8")
    assert parse_card(card) == {}


def test_parse_card_keeps_the_first_value_for_a_repeated_key(tmp_path):
    card = tmp_path / "p.md"
    card.write_text("Role: Director\nRole: Former Director\n", encoding="utf-8")
    assert parse_card(card)["role"] == "Director"


def test_parse_card_survives_an_unreadable_file(tmp_path):
    card = tmp_path / "p.md"
    card.write_bytes(b"\xff\xfe\x00bad")
    assert isinstance(parse_card(card), dict)


# --- master index ----------------------------------------------------------


def test_parse_index_reads_aliases_and_rules(tmp_path):
    (tmp_path / "MasterIndex.csv").write_text(
        "Folder,Aliases,Assignment Rule\n"
        "Laura,Laura T;L. Terry,Route to Ops\n"
        "Mike,Michael,Route to Finance\n", encoding="utf-8")
    index = parse_index(tmp_path)
    assert index["laura"]["aliases"] == ["Laura T", "L. Terry"]
    assert "Route to Ops" in index["laura"]["rules"].values()
    assert index["mike"]["aliases"] == ["Michael"]


def test_parse_index_tolerates_a_missing_file(tmp_path):
    assert parse_index(tmp_path) == {}


def test_parse_index_matches_varied_column_names(tmp_path):
    (tmp_path / "speaker_index.csv").write_text(
        "Speaker Name,Also Known As\nLaura,LT\n", encoding="utf-8")
    assert parse_index(tmp_path)["laura"]["aliases"] == ["LT"]


# --- folder scanning -------------------------------------------------------


def _speaker(root, name, audio="sample.ogg", card=True):
    folder = root / name
    folder.mkdir()
    (folder / audio).write_bytes(b"")
    if card:
        (folder / "profile.md").write_text(f"Name: {name}\nRole: tester\n",
                                           encoding="utf-8")
    return folder


def test_scan_builds_one_profile_per_speaker_folder(tmp_path):
    _speaker(tmp_path, "Laura")
    _speaker(tmp_path, "Mike")
    (tmp_path / "MasterIndex.md").write_text("notes", encoding="utf-8")
    profiles = scan(tmp_path)
    assert [p.name for p in profiles] == ["Laura", "Mike"]
    assert all(p.samples for p in profiles)
    assert profiles[0].card["role"] == "tester"


def test_scan_skips_folders_without_audio(tmp_path):
    _speaker(tmp_path, "Laura")
    (tmp_path / "Notes").mkdir()
    (tmp_path / "Notes" / "readme.md").write_text("x", encoding="utf-8")
    assert [p.name for p in scan(tmp_path)] == ["Laura"]


def test_scan_merges_aliases_from_index_and_card(tmp_path):
    folder = _speaker(tmp_path, "Laura", card=False)
    (folder / "profile.md").write_text("Name: Laura\nAliases: Lu\n", encoding="utf-8")
    (tmp_path / "MasterIndex.csv").write_text("Folder,Aliases\nLaura,Laura T\n",
                                              encoding="utf-8")
    assert scan(tmp_path)[0].aliases == ["Laura T", "Lu"]


def test_scan_accepts_several_audio_formats(tmp_path):
    folder = tmp_path / "Laura"
    folder.mkdir()
    for name in ("a.ogg", "b.wav", "c.m4a", "notes.txt"):
        (folder / name).write_bytes(b"")
    assert len(scan(tmp_path)[0].samples) == 3


# --- matching --------------------------------------------------------------


def test_cosine_of_identical_vectors_is_one():
    v = np.array([0.3, 0.4, 0.5])
    assert cosine(v, v) == pytest.approx(1.0)


def test_cosine_of_orthogonal_vectors_is_zero():
    assert cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_cosine_handles_a_zero_vector():
    assert cosine(np.zeros(3), np.array([1.0, 2.0, 3.0])) == 0.0


def test_identify_returns_the_nearest_speaker():
    lib = {"speakers": [
        {"name": "Laura", "embedding": np.array([1.0, 0.0, 0.0])},
        {"name": "Mike", "embedding": np.array([0.0, 1.0, 0.0])},
    ]}
    name, score = identify(np.array([0.95, 0.1, 0.0]), lib, threshold=0.5)
    assert name == "Laura" and score > 0.9


def test_identify_refuses_a_stranger():
    """An unknown voice must come back unnamed rather than borrowing a name."""
    lib = {"speakers": [{"name": "Laura", "embedding": np.array([1.0, 0.0, 0.0])}]}
    name, score = identify(np.array([0.0, 0.0, 1.0]), lib, threshold=0.55)
    assert name is None
    assert score < 0.55


def test_identify_on_an_empty_library():
    name, score = identify(np.array([1.0, 0.0]), {"speakers": []})
    assert name is None


# --- library round trip ----------------------------------------------------


def test_library_round_trips(tmp_path):
    p = Profile(name="Laura", folder=tmp_path / "Laura",
                samples=[tmp_path / "s.ogg"], card={"role": "Ops"},
                aliases=["LT"])
    p.embedding = np.array([0.6, 0.8])
    out = tmp_path / "voiceprints.json"
    save_library([p], out)

    lib = load_library(out)
    assert lib["speakers"][0]["name"] == "Laura"
    assert lib["speakers"][0]["aliases"] == ["LT"]
    assert lib["speakers"][0]["card"]["role"] == "Ops"
    assert np.allclose(lib["speakers"][0]["embedding"], [0.6, 0.8])


def test_library_records_the_model_that_made_it(tmp_path):
    """
    Embeddings are only comparable within the model that produced them, so the
    library has to say which one it used.
    """
    p = Profile(name="L", folder=tmp_path, samples=[])
    p.embedding = np.array([1.0, 0.0])
    out = tmp_path / "v.json"
    save_library([p], out)
    assert "model" in json.loads(out.read_text())
    assert json.loads(out.read_text())["dimensions"] == 2


# --- threshold behaviour ---------------------------------------------------


def test_default_threshold_sits_above_a_realistic_worst_pair():
    """
    On a real 41-speaker library the closest two voices scored 0.55. A default
    at or below that would name one person as the other; it has to clear it.
    """
    import inspect

    from enroll import identify as fn

    default = inspect.signature(fn).parameters["threshold"].default
    assert default > 0.55, "default threshold does not clear a measured worst pair"


def test_identify_respects_a_raised_threshold():
    lib = {"speakers": [{"name": "Brie", "embedding": np.array([1.0, 0.0])}]}
    probe = np.array([0.9, 0.436])  # ~0.9 similarity
    assert identify(probe, lib, threshold=0.65)[0] == "Brie"
    assert identify(probe, lib, threshold=0.95)[0] is None


def test_identify_picks_the_best_of_several_close_candidates():
    """Two similar voices must resolve to the nearer one, not the first found."""
    lib = {"speakers": [
        {"name": "Brie", "embedding": np.array([1.0, 0.0, 0.0])},
        {"name": "Caitlin", "embedding": np.array([0.85, 0.53, 0.0])},
    ]}
    name, _ = identify(np.array([0.86, 0.51, 0.0]), lib, threshold=0.6)
    assert name == "Caitlin"


# --- capture-domain matching -----------------------------------------------


def test_simulate_capture_removes_what_the_recorder_removes():
    """
    Enrollment clips are clean and full-band; the recorder high-passes the mic
    at 300 Hz and HFP discards everything above ~4 kHz. Embedding the two
    without matching domains produced a measured inversion: a speaker scored
    0.455 against his own profile while two different people scored 0.55.
    """
    from enroll import TARGET_RATE, simulate_capture

    sr = TARGET_RATE
    t = np.linspace(0, 1.0, sr, endpoint=False)
    signal = (0.5 * np.sin(2 * np.pi * 120 * t)      # fundamental, should go
              + 0.3 * np.sin(2 * np.pi * 1000 * t)   # speech band, should stay
              + 0.3 * np.sin(2 * np.pi * 6000 * t))  # above HFP, should go
    out = simulate_capture(signal.astype(np.float32))

    spec = np.abs(np.fft.rfft(out)) ** 2
    freqs = np.fft.rfftfreq(len(out), 1 / sr)
    total = spec.sum()
    assert spec[freqs < 300].sum() / total < 0.01, "fundamental survived"
    assert spec[freqs > 3800].sum() / total < 0.01, "top end survived"
    assert spec[(freqs >= 300) & (freqs <= 3800)].sum() / total > 0.95


def test_simulate_capture_preserves_length_and_normalises():
    from enroll import simulate_capture

    audio = (np.random.default_rng(0).standard_normal(16000) * 0.01).astype(np.float32)
    out = simulate_capture(audio)
    assert out.shape == audio.shape
    assert 0.5 < float(np.max(np.abs(out))) <= 0.75


def test_simulate_capture_handles_empty_input():
    from enroll import simulate_capture

    assert simulate_capture(np.zeros(0, dtype=np.float32)).size == 0


def test_simulate_capture_respects_a_custom_highpass():
    from enroll import TARGET_RATE, simulate_capture

    sr = TARGET_RATE
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sig = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    # Measure energy, not peak: a filtered-out tone leaves numerical residue
    # that peak-normalising would scale straight back up.
    cut = simulate_capture(sig, highpass_hz=300)
    kept = simulate_capture(sig, highpass_hz=120)
    assert float(np.sqrt(np.mean(cut ** 2))) < 1e-4, "200 Hz tone survived a 300 Hz cut"
    assert float(np.sqrt(np.mean(kept ** 2))) > 0.1, "200 Hz tone lost to a 120 Hz cut"


def test_simulate_capture_does_not_amplify_silence():
    """A track with nothing in it must come back with nothing in it."""
    from enroll import simulate_capture

    out = simulate_capture(np.zeros(16000, dtype=np.float32))
    assert float(np.max(np.abs(out))) < 1e-6


# --- score normalisation ---------------------------------------------------


def _library(values: dict) -> dict:
    """Embeddings whose cosine against a fixed probe equals each given value."""
    lib = {"speakers": []}
    for name, v in values.items():
        e = np.zeros(8)
        e[0] = v
        e[1] = np.sqrt(max(0.0, 1 - v * v))
        lib["speakers"].append({"name": name, "embedding": e})
    return lib


PROBE = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])


def test_cross_channel_match_is_accepted_on_its_lead():
    """
    The real case. A speaker's own voice scored 0.455 against his own profile
    while impostor pairs inside the library reached 0.55. Absolute scoring
    rejects the true match; the statistical lead accepts it.
    """
    # Rebuilt from a real 41-speaker run: 0.455 for the right person, then
    # 0.262 / 0.242 / 0.203 / 0.199 and a long tail. A small fixture would
    # have an inflated spread and misrepresent the z-score entirely.
    tail = [0.262, 0.242, 0.203, 0.199] + list(np.linspace(0.19, 0.01, 36))
    lib = _library({"Grayson": 0.455,
                    **{f"P{i}": v for i, v in enumerate(tail)}})
    name, score = identify(PROBE, lib)
    assert name == "Grayson"
    assert score < 0.65, "accepted despite being below the absolute threshold"


def test_a_stranger_with_no_clear_lead_is_rejected():
    """An unknown voice scores middling against everyone and leads nobody."""
    lib = _library({f"P{i}": v for i, v in
                    enumerate([0.31, 0.30, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24])})
    name, _ = identify(PROBE, lib)
    assert name is None


def test_high_absolute_score_still_accepted_without_a_lead():
    """Same-domain audio can score high against several profiles at once."""
    lib = _library({"A": 0.80, "B": 0.74, "C": 0.72, "D": 0.70, "E": 0.69})
    assert identify(PROBE, lib)[0] == "A"


def test_score_detail_reports_the_decision_inputs():
    lib = _library({"A": 0.46, "B": 0.26, "C": 0.24, "D": 0.20})
    d = score_detail(PROBE, lib)
    assert d["ranked"][0][1] == "A"
    assert d["margin"] == pytest.approx(0.20, abs=0.01)
    assert d["z"] > 2.0


def test_tiny_library_falls_back_to_the_absolute_threshold():
    """With one or two profiles there is no field to measure a lead against."""
    lib = _library({"A": 0.50})
    assert identify(PROBE, lib)[0] is None
    assert identify(PROBE, _library({"A": 0.80}))[0] == "A"


def test_identify_handles_an_empty_library_cleanly():
    assert identify(PROBE, {"speakers": []}) == (None, 0.0)


# --- three-way classification ----------------------------------------------


def test_classify_separates_the_three_outcomes():
    from enroll import classify

    tail = list(np.linspace(0.20, 0.02, 38))

    clear = _library({"Matt": 0.78, "Next": 0.30,
                      **{f"P{i}": v for i, v in enumerate(tail)}})
    assert classify(PROBE, clear)["status"] == "named"

    tie = _library({"Kody": 0.60, "Sam": 0.56,
                    **{f"P{i}": v for i, v in enumerate(tail)}})
    assert classify(PROBE, tie)["status"] == "ambiguous"

    stranger = _library({f"P{i}": v for i, v in enumerate(np.linspace(0.30, 0.02, 40))})
    assert classify(PROBE, stranger)["status"] == "unknown"


def test_ambiguous_result_names_nobody_but_lists_both():
    from enroll import classify

    tail = list(np.linspace(0.20, 0.02, 38))
    out = classify(PROBE, _library({"Kody": 0.60, "Sam": 0.56,
                                    **{f"P{i}": v for i, v in enumerate(tail)}}))
    assert out["name"] is None
    assert [c["name"] for c in out["candidates"][:2]] == ["Kody", "Sam"]
    assert out["margin"] < 0.10


def test_margin_requirement_is_configurable():
    from enroll import classify

    tail = list(np.linspace(0.20, 0.02, 38))
    lib = _library({"Kody": 0.60, "Sam": 0.56,
                    **{f"P{i}": v for i, v in enumerate(tail)}})
    assert classify(PROBE, lib, min_margin=0.02)["status"] == "named"
    assert classify(PROBE, lib, min_margin=0.20)["status"] == "ambiguous"


def test_classify_on_an_empty_library():
    from enroll import classify

    out = classify(PROBE, {"speakers": []})
    assert out["status"] == "unknown" and out["candidates"] == []


def test_encoder_loading_asks_for_copies_not_symlinks():
    """
    Windows refuses symlinks without Developer Mode, so speechbrain warned on
    every run and then copied anyway — noise printed over working behaviour.
    Asking for copies up front removes the cause rather than hiding it.
    """
    import inspect

    from enroll import load_encoder

    source = inspect.getsource(load_encoder)
    assert "LocalStrategy" in source
    assert "COPY" in source
    assert "catch_warnings" in source, "no fallback for older speechbrain"
