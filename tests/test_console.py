"""Console tests — command building, job control, state. No browser."""

import json
import sys
import time
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import console
from console import JOB, build_commands, gather_unknowns, job_state, start_job


def wait_for_job(timeout=40.0):
    """
    Poll until the job finishes. Spawning an interpreter takes longer under a
    loaded test run than it does alone, and a fixed deadline turns that into
    an intermittent failure that looks like a bug.
    """
    deadline = time.time() + timeout
    while time.time() < deadline and job_state()["running"]:
        time.sleep(0.05)
    return job_state()


@pytest.fixture
def wired(tmp_path):
    archive = tmp_path / "archive"
    profiles = tmp_path / "profiles"
    archive.mkdir()
    (profiles / "Person_A").mkdir(parents=True)
    (profiles / "Person_B").mkdir(parents=True)
    console.STATE.update({"archive": archive, "profiles": profiles,
                          "root": tmp_path, "python": sys.executable})
    JOB.update({"name": None, "proc": None, "lines": console.deque(maxlen=400),
                "started": None, "finished": None, "code": None})
    return tmp_path


# --- commands --------------------------------------------------------------


def test_every_button_has_a_label_and_a_hint(wired):
    for key, cmd in build_commands().items():
        assert cmd["label"], key
        assert cmd["hint"], key
        assert cmd["group"] in {"Check", "Process", "Maintain"}, key


def test_commands_needing_a_library_are_marked_unavailable_without_one(wired):
    console.STATE["profiles"] = None
    cmds = build_commands()
    assert cmds["diarize"]["argv"] is None
    assert cmds["enroll"]["argv"] is None
    assert cmds["transcribe"]["argv"] is not None, "transcription needs no library"


def test_library_path_is_passed_to_diarize(wired):
    argv = build_commands()["diarize"]["argv"]
    assert "--library" in argv
    assert argv[argv.index("--library") + 1].endswith("voiceprints.json")


def test_force_variant_is_distinct(wired):
    cmds = build_commands()
    assert "--force" in cmds["diarize_force"]["argv"]
    assert "--force" not in cmds["diarize"]["argv"]


# --- job control -----------------------------------------------------------


def test_a_job_streams_its_output(wired):
    start_job("echo", [sys.executable, "-c", "print('hello from the job')"])
    state = wait_for_job()
    assert state["code"] == 0
    assert any("hello from the job" in ln for ln in state["lines"])


def test_only_one_job_runs_at_a_time(wired):
    """
    Transcription saturates the CPU. Letting a web page queue three of them
    is a way to wedge the machine.
    """
    start_job("slow", [sys.executable, "-c", "import time; time.sleep(5)"])
    second = start_job("another", [sys.executable, "-c", "print('nope')"])
    assert second["ok"] is False
    assert "still running" in second["error"]
    JOB["proc"].terminate()


def test_a_finished_job_frees_the_slot(wired):
    start_job("quick", [sys.executable, "-c", "pass"])
    wait_for_job()
    assert start_job("next", [sys.executable, "-c", "pass"])["ok"] is True


def test_a_failing_job_reports_its_exit_code(wired):
    start_job("fail", [sys.executable, "-c", "raise SystemExit(3)"])
    assert wait_for_job()["code"] == 3


# --- review data -----------------------------------------------------------


def _meeting(root, name, speakers):
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    with wave.open(str(folder / "x-system.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.zeros(16000, dtype=np.int16).tobytes())
    (folder / "speakers.json").write_text(json.dumps({"speakers": speakers}))
    return folder


def test_unknown_voices_group_across_meetings(wired):
    archive = console.STATE["archive"]
    voice = [1.0, 0.0, 0.0]
    for name in ("m1", "m2"):
        _meeting(archive, name, [{"label": "Speaker 1", "name": None,
                                  "confidence": 0.3, "z": 2.0, "seconds": 120,
                                  "embedding": voice, "turns": []}])
    result = gather_unknowns()
    assert result["stats"]["likely_people"] == 1
    assert result["groups"][0]["meetings"] == 2


def test_named_speakers_are_not_listed_as_unknown(wired):
    _meeting(console.STATE["archive"], "m1",
             [{"label": "Person_A", "name": "Person_A", "confidence": 0.8,
               "z": 9.0, "seconds": 100, "turns": []}])
    assert gather_unknowns()["stats"]["likely_people"] == 0


def test_library_names_come_from_profile_folders(wired):
    assert console.library_names() == ["Person_A", "Person_B"]


def test_a_quit_endpoint_exists():
    """
    Launched from a shortcut there is no console to Ctrl-C, so the page must
    be able to stop the server it is talking to — otherwise the only way out
    is Task Manager.
    """
    import inspect

    import console as mod

    source = inspect.getsource(mod.Handler.do_POST)
    assert "/api/quit" in source
    assert "/api/quit" in mod.PAGE, "no Quit control in the page"


def test_quit_warns_that_the_recorder_keeps_running():
    import console as mod

    assert "recorder keeps running" in mod.PAGE


# --- the help panel ---------------------------------------------------------


def test_help_panel_covers_every_button():
    """
    The panel is only useful if it stays in step with the buttons. A new
    command with no explanation is the failure this guards against.
    """
    import console as mod

    console.STATE.update({"archive": Path("/tmp"), "profiles": Path("/tmp"),
                          "root": Path("/tmp"), "python": sys.executable})
    # The help is HTML, so "&" is written "&amp;" there quite correctly.
    page = mod.PAGE.replace("&amp;", "&")
    for cmd in build_commands().values():
        assert cmd["label"] in page, f"no help text mentions '{cmd['label']}'"


def test_help_states_the_order_that_matters():
    """Transcription reads what identification wrote, so the order is not
    a preference."""
    import console as mod

    page = mod.PAGE
    assert "Order of operations" in page
    assert page.index("Identify speakers</strong> — clusters") < \
        page.index("Transcribe</strong> — slow"), "order section is inverted"


def test_help_markup_is_balanced():
    import console as mod

    page = mod.PAGE
    assert page.count("<details") == page.count("</details>")
    assert page.count('<div class="body">') == page.count("</div></details>")


def test_help_explains_single_track_phone_recordings():
    """The one behavioural difference a user would otherwise find confusing."""
    import console as mod

    assert "single track" in mod.PAGE
    assert "including you" in mod.PAGE


# --- playing clips from any recording ---------------------------------------


def test_a_phone_upload_is_playable(wired):
    """
    The clip endpoint used to look only for *-system.wav, so clips from phone
    recordings silently 404'd — you could see a voice on the timeline but
    never hear it, which is exactly when you need to.
    """
    from console import playable_track

    folder = console.STATE["archive"] / "m"
    folder.mkdir()
    (folder / "20261025-111000-plaud.mp3").write_bytes(b"")
    assert playable_track(folder).name == "20261025-111000-plaud.mp3"


def test_a_system_track_is_preferred_when_present(wired):
    from console import playable_track

    folder = console.STATE["archive"] / "m"
    folder.mkdir()
    (folder / "x-system.wav").write_bytes(b"")
    (folder / "x.mp3").write_bytes(b"")
    assert playable_track(folder).name == "x-system.wav"


def test_the_mic_track_is_never_served_as_the_timeline_audio(wired):
    """The timeline describes the far end, not your own microphone."""
    from console import playable_track

    folder = console.STATE["archive"] / "m"
    folder.mkdir()
    for name in ("x-mic.wav", "x-mix.wav"):
        (folder / name).write_bytes(b"")
    assert playable_track(folder) is None


def test_slicing_a_wav_returns_the_requested_span(wired):
    from console import slice_audio

    folder = console.STATE["archive"] / "m"
    folder.mkdir()
    path = folder / "x-system.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.zeros(16000 * 30, dtype=np.int16).tobytes())
    clip = slice_audio(path, 5.0, 11.0)
    import io as _io
    with wave.open(_io.BytesIO(clip), "rb") as w:
        assert w.getnframes() / w.getframerate() == pytest.approx(6.0, abs=0.05)


# --- date cutoff in the GUI -------------------------------------------------


def test_only_dated_commands_accept_a_cutoff(wired):
    """
    The value reaches a subprocess argument list, so only commands that
    understand --since should ever receive it.
    """
    cmds = build_commands()
    assert cmds["diarize"].get("dated") is True
    assert cmds["transcribe"].get("dated") is True
    assert cmds["diarize_force"].get("dated") is True
    for key in ("devices", "storage", "list", "testrecord", "enroll", "tests"):
        assert not cmds[key].get("dated"), f"{key} does not take a date"


def test_the_run_endpoint_validates_the_date_format():
    import inspect

    import console as mod

    source = inspect.getsource(mod.Handler.do_POST)
    assert "d{4}-" in source or "\\\\d{4}-" in source, "no date validation"
    assert '"--since"' in source


def test_the_page_offers_spans_and_a_date_picker():
    import console as mod

    page = mod.PAGE
    for label in ("Last 7 days", "Last month", "Last 3 months", "Everything"):
        assert label in page, f"no {label!r} option"
    assert 'type="date"' in page
    assert "Only meetings from" in page


# --- visual grouping --------------------------------------------------------


def test_each_control_group_gets_its_own_panel():
    """
    Without panels the page is one long column of buttons and the headings
    disappear into it.
    """
    import console as mod

    page = mod.PAGE
    assert ".panel {" in page
    for group in ("recorder", "check", "process", "maintain"):
        assert f"panel-{group}" in page, f"no accent defined for {group}"


def test_panels_are_balanced():
    import console as mod

    page = mod.PAGE
    assert page.count('<section class="panel') == page.count("</section>")


def test_speaker_state_is_visible_at_a_glance():
    """Named, unknown and ambiguous should be distinguishable without
    reading the score."""
    import console as mod

    page = mod.PAGE
    assert ".person.unknown" in page
    assert ".person.ambiguous" in page
    assert "border-left" in page


# --- meeting labels ---------------------------------------------------------


def test_metadata_is_read_from_either_sidecar_shape(wired):
    """
    The desktop writes <name>-mic.wav.json; the phone writes <name>.meta.json.
    Looking only for the desktop one left every phone recording titled
    "Meeting" with nothing else known.
    """
    from console import read_meta

    desktop = console.STATE["archive"] / "d"
    desktop.mkdir()
    (desktop / "x-mic.wav.json").write_text(json.dumps({"app": "zoom.exe"}))
    assert read_meta(desktop)["app"] == "zoom.exe"

    phone = console.STATE["archive"] / "p"
    phone.mkdir()
    (phone / "rec.meta.json").write_text(json.dumps({"title": "Standup"}))
    assert read_meta(phone)["title"] == "Standup"


def test_sidecars_we_write_ourselves_are_not_mistaken_for_metadata(wired):
    from console import read_meta

    folder = console.STATE["archive"] / "m"
    folder.mkdir()
    (folder / "speakers.json").write_text(json.dumps({"speakers": []}))
    (folder / "transcript.json").write_text(json.dumps({"segments": []}))
    (folder / "x-shots.json").write_text(json.dumps([]))
    assert read_meta(folder) == {}


def test_a_title_falls_back_through_sensible_options(wired):
    from console import meeting_title

    speakers = [{"name": "Laura_Eich"}, {"name": "Brie_Roper"}, {"name": "Dan_Baron"}]
    assert meeting_title(Path("m"), {"title": "Q3 review"}, []) == "Q3 review"
    assert meeting_title(Path("m"), {"app": "ms-teams.exe"}, []) == "Teams"
    # no title and no app: name the people in it
    assert "Laura Eich" in meeting_title(Path("m"), {}, speakers)
    assert meeting_title(Path("20261101-075933-plaud"), {}, []) == "Plaud recording"
    assert meeting_title(Path("m"), {}, []) == "Meeting"


# --- collections ------------------------------------------------------------


def test_a_rule_sorts_by_who_is_in_the_meeting(wired):
    from console import matches_rule

    meeting = {"named": ["Laura Eich", "Brie Roper"], "title": "Teams"}
    assert matches_rule(meeting, {"speakers_any": ["Laura Eich"]})
    assert matches_rule(meeting, {"speakers_all": ["Laura Eich", "Brie Roper"]})
    assert not matches_rule(meeting, {"speakers_all": ["Laura Eich", "Dan Baron"]})
    assert not matches_rule(meeting, {"speakers_any": ["Dan Baron"]})
    assert not matches_rule(meeting, {"speakers_none": ["Laura Eich"]})


def test_an_empty_rule_matches_nothing(wired):
    """A rule with no conditions would swallow the entire archive, which is
    never what someone meant to build."""
    from console import matches_rule

    assert not matches_rule({"named": ["Laura Eich"], "title": "x"}, {})


def test_rules_ignore_underscores_and_case(wired):
    from console import matches_rule

    meeting = {"named": ["Laura Eich"], "title": "Teams"}
    assert matches_rule(meeting, {"speakers_any": ["laura eich"]})


def test_a_hand_placed_meeting_is_not_overridden_by_a_rule(wired):
    """A rule should never move something someone filed deliberately."""
    from console import assign_collection

    data = {
        "collections": [{"id": "c1", "name": "Ops",
                         "rule": {"speakers_any": ["Laura Eich"]}}],
        "assignments": {"20261101-075933-plaud": "c2"},
    }
    meeting = {"name": "20261101-075933-plaud", "named": ["Laura Eich"], "title": ""}
    assert assign_collection(meeting, data) == "c2"


def test_the_first_matching_rule_wins(wired):
    from console import assign_collection

    data = {"collections": [
        {"id": "c1", "name": "A", "rule": {"speakers_any": ["Laura Eich"]}},
        {"id": "c2", "name": "B", "rule": {"speakers_any": ["Laura Eich"]}},
    ], "assignments": {}}
    meeting = {"name": "m", "named": ["Laura Eich"], "title": ""}
    assert assign_collection(meeting, data) == "c1"


def test_collections_round_trip_through_disk(wired):
    from console import load_collections, save_collections

    save_collections({"collections": [{"id": "c1", "name": "Ops", "rule": {}}],
                      "assignments": {"m1": "c1"}})
    data = load_collections()
    assert data["collections"][0]["name"] == "Ops"
    assert data["assignments"]["m1"] == "c1"


def test_a_corrupt_collections_file_does_not_crash(wired):
    from console import collections_path, load_collections

    path = collections_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    assert load_collections() == {"collections": [], "assignments": {}}


# --- clip slicing -----------------------------------------------------------


def _write_tone(path, seconds=20.0, rate=44100):
    """
    WAV only, written with the standard library.

    Writing ANY compressed format through libsndfile overflows the stack and
    takes the interpreter down on some Windows builds — observed with mp3,
    and again with ogg or flac. A hard crash cannot be caught, so there is no
    safe way to probe for it: tests simply never write compressed audio.

    Nothing in this project writes compressed audio either, so this costs no
    coverage of real behaviour. The compressed READ path is exercised against
    files the recorder actually receives, not ones invented here.
    """
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    sig = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(sig.tobytes())


def _clip_seconds(data):
    import io as _io

    with wave.open(_io.BytesIO(data), "rb") as w:
        return w.getnframes() / w.getframerate()


def test_a_clip_can_be_cut_from_a_wav(tmp_path):
    from console import slice_audio

    path = tmp_path / "rec.wav"
    _write_tone(path)
    assert _clip_seconds(slice_audio(path, 5.0, 11.0)) == pytest.approx(6.0, abs=0.1)


def test_compressed_recordings_go_through_the_decode_path(tmp_path, monkeypatch):
    """
    Exercised with a stubbed decoder rather than a real compressed file,
    because creating one crashes the interpreter on some builds. What matters
    is that a non-WAV file takes the soundfile route and gets sliced by frame.
    """
    import console as mod

    rate = 16000
    whole = (np.arange(rate * 10, dtype=np.int16)).reshape(-1, 1)
    monkeypatch.setattr(mod, "_decode_whole", lambda track: (whole, rate))

    class FakeInfo:
        samplerate = rate
        frames = rate * 10

    fake_sf = type("sf", (), {
        "info": staticmethod(lambda path: FakeInfo()),
        "read": staticmethod(lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no seek"))),
    })
    monkeypatch.setitem(sys.modules, "soundfile", fake_sf)

    path = tmp_path / "rec.mp3"
    path.write_bytes(b"not really an mp3")
    # seek fails, so it must fall back to decoding the whole file
    assert _clip_seconds(mod.slice_audio(path, 2.0, 5.0)) == pytest.approx(3.0, abs=0.1)


def test_a_range_past_the_end_is_clamped(tmp_path):
    """Diariser turns can run marginally past the decoded length; that should
    return the tail, not an error."""
    from console import slice_audio

    path = tmp_path / "rec.wav"
    _write_tone(path, seconds=20.0)
    assert _clip_seconds(slice_audio(path, 18.0, 25.0)) == pytest.approx(2.0, abs=0.1)


def test_an_empty_range_is_refused(tmp_path):
    from console import slice_audio

    path = tmp_path / "rec.wav"
    _write_tone(path, seconds=5.0)
    with pytest.raises(ValueError):
        slice_audio(path, 3.0, 3.0)
    with pytest.raises(ValueError):
        slice_audio(path, 30.0, 40.0)


def test_a_decoded_file_is_reused_between_clips(tmp_path, monkeypatch):
    """
    Without the cache, every clip in a meeting would decode the whole
    recording again — minutes of CPU to hear four seconds.
    """
    import console as mod

    path = tmp_path / "rec.wav"
    _write_tone(path)

    mod._DECODED.update({"key": None, "audio": None, "rate": 0})
    calls = {"n": 0}
    real = mod._decode_whole

    def counted(track):
        calls["n"] += 1
        return real(track)

    monkeypatch.setattr(mod, "_decode_whole", counted)
    for _ in range(3):
        mod._decode_whole(path)
    assert calls["n"] == 3
    assert mod._DECODED["audio"] is not None


def test_the_player_reports_the_real_reason():
    """
    One message for every failure — missing file, decode error, server down —
    made this impossible to diagnose from the page.
    """
    import console as mod

    page = mod.PAGE
    assert "Could not load that clip:" in page
    assert "Browser refused to play it:" in page
    assert "came back empty" in page


def test_the_whole_recording_can_be_served_for_browser_playback():
    """
    Server-side decoding of compressed audio depends on the libsndfile build
    installed — one of which crashes the interpreter outright. Browsers
    decode MP3 and Opus natively, so the reliable path is to hand over the
    file and seek in it.
    """
    import inspect

    import console as mod

    source = inspect.getsource(mod.Handler.do_GET)
    assert "/api/audio" in source
    assert "audio/mpeg" in source, "no content type for mp3"
    assert "playInPlace" in mod.PAGE
    assert "/api/audio?folder=" in mod.PAGE


def test_server_cutting_remains_as_a_fallback():
    """Some formats a browser will refuse; cutting server-side still works
    for those."""
    import console as mod

    assert "playCut" in mod.PAGE
    assert "Cutting that clip on the server" in mod.PAGE


def test_playback_stops_at_the_end_of_the_turn():
    """Seeking into a two-hour recording would otherwise play to the end of
    the meeting."""
    import console as mod

    assert "STOP_AT" in mod.PAGE
    assert "ontimeupdate" in mod.PAGE


def test_no_test_ever_writes_compressed_audio():
    """
    Guard against reintroducing the crash. Writing ogg, flac or mp3 through
    libsndfile overflows the stack on some Windows builds and takes the whole
    run down — and a hard crash cannot be caught or skipped around, so the
    only defence is never doing it.
    """
    from pathlib import Path as _Path

    tests_dir = _Path(__file__).resolve().parent
    offenders = []
    for path in tests_dir.glob("test_*.py"):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            # Built at runtime so this check does not match its own source.
            needles = ("sf" + ".write(", "soundfile" + ".write(")
            if any(n in stripped for n in needles):
                offenders.append(f"{path.name}:{number}")
    assert not offenders, f"compressed audio written in: {offenders}"


# --- listening from the Unnamed tab -----------------------------------------


def test_unknown_groups_carry_what_playback_needs(wired):
    """
    The Unnamed tab could not play anything: its entries need both the folder
    the audio lives in and the turn times to seek to, and a card without
    those is a dead button.
    """
    from console import gather_unknowns

    archive = console.STATE["archive"]
    folder = archive / "20261101-075933-plaud"
    folder.mkdir(parents=True)
    with wave.open(str(folder / "rec.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.zeros(16000 * 60, dtype=np.int16).tobytes())
    (folder / "speakers.json").write_text(json.dumps({"speakers": [
        {"label": "Speaker 1", "name": None, "confidence": 0.3, "z": 2.0,
         "seconds": 40.0, "embedding": [1.0, 0.0, 0.0],
         "turns": [{"start": 2.0, "end": 22.0}, {"start": 30.0, "end": 50.0}]},
    ]}))

    group = gather_unknowns()["groups"][0]
    member = group["members"][0]
    assert Path(member["folder"]).exists()
    assert member["turns"], "no turn times — nothing to play"
    assert all(t["end"] > t["start"] for t in member["turns"])


def test_the_unnamed_tab_plays_through_the_same_path_as_meetings():
    """
    It had its own Listen button on a separate code path, which is how it
    came to be broken while the Meetings tab worked.
    """
    import console as mod

    page = mod.PAGE
    assert ".listen" not in page, "the separate Listen button is back"
    assert "cliprow" in page
    assert page.count("play(member.folder") >= 1


def test_playback_refuses_impossible_requests_out_loud():
    import console as mod

    page = mod.PAGE
    assert "no recording attached" in page
    assert "no length recorded" in page


# --- saving a name must not depend on enrollment ----------------------------


def test_the_name_is_written_before_enrollment_is_attempted():
    """
    The real failure: enrollment decodes audio, which throws on some builds,
    and the exception aborted the whole request — so the name was silently
    never saved and the correction vanished.
    """
    import inspect

    import console as mod

    source = inspect.getsource(mod.Handler.do_POST)
    save_at = source.index("path.write_text(json.dumps(data, indent=2)")
    enroll_at = source.index("enroll_from_meeting(folder, speaker, person)")
    assert save_at < enroll_at, "enrollment still runs before the name is saved"
    assert "except Exception as exc:" in source


def test_aborted_connections_are_not_treated_as_errors():
    """
    The page polls every 1.2 seconds; a browser navigating away aborts the
    socket mid-write and printed a full traceback every time, burying
    anything that mattered.
    """
    import inspect

    import console as mod

    source = inspect.getsource(mod.Handler)
    assert "ConnectionAbortedError" in source
    assert "handle_one_request" in source


def test_the_player_can_be_stopped():
    """A clip can run a minute; there was no way to stop it short of
    reloading the page."""
    import console as mod

    page = mod.PAGE
    assert 'id="stop"' in page
    assert 'id="pp"' in page
    assert "stopAudio()" in page
    assert "Escape" in page, "no keyboard stop"


def test_a_meeting_folder_can_be_opened():
    import inspect

    import console as mod

    assert "/api/reveal" in inspect.getsource(mod.Handler.do_POST)
    assert "Open folder" in mod.PAGE


# --- calendar ---------------------------------------------------------------


def test_the_calendar_offers_week_and_month():
    import console as mod

    page = mod.PAGE
    assert 'id="tab-k"' in page
    assert "drawWeek" in page and "drawMonth" in page
    assert 'data-mode="week"' in page and 'data-mode="month"' in page


def test_the_week_starts_on_sunday_like_the_day_headers():
    """A grid whose columns disagree with its headers puts every meeting on
    the wrong day."""
    import console as mod

    page = mod.PAGE
    assert "x.setDate(x.getDate() - x.getDay())" in page
    assert "cols[at.getDay()]" in page


def test_only_dated_meetings_are_placed():
    """A folder without a timestamp has no position on a calendar; it must be
    left out rather than dropped at an arbitrary time."""
    import console as mod

    assert "filter(x => x.at)" in mod.PAGE


def test_clicking_a_calendar_block_opens_that_meeting():
    import console as mod

    page = mod.PAGE
    assert "openMeeting(i)" in page
    assert "setView('meetings')" in page


def test_opening_from_the_calendar_does_not_snap_back_to_the_newest():
    import console as mod

    assert "CURRENT ?? 0, DATA.meetings.length - 1" in mod.PAGE


# --- the page's own integrity -----------------------------------------------


def test_every_function_the_page_calls_is_defined():
    """
    The page is one big string, so a missing function is invisible until
    someone clicks the thing that needs it. That is exactly what happened:
    an edit silently failed to apply, the sidebar helpers never made it into
    the page, and the Calendar tab rendered nothing at all with no error
    anywhere on the server.
    """
    import re

    import console as mod

    script = mod.PAGE[mod.PAGE.index("<script>"):]

    defined = set(re.findall(r"function\s+([A-Za-z_]\w*)\s*\(", script))
    defined |= set(re.findall(r"(?:const|let|var)\s+([A-Za-z_]\w*)\s*=", script))
    defined |= set(re.findall(r"([A-Za-z_]\w*)\s*:\s*(?:async\s*)?(?:function|\()", script))
    # Shorthand object methods — `hide() { … }` — define a name rather than
    # calling one.
    defined |= set(re.findall(r"([A-Za-z_]\w*)\s*\([^()]*\)\s*\{", script))
    # Parameters of arrow functions are local names, not globals.
    for params in re.findall(r"\(([^()]*)\)\s*=>", script):
        defined |= {q.strip() for q in params.split(",") if q.strip().isidentifier()}

    builtins = {
        "if", "for", "while", "switch", "catch", "return", "typeof", "fetch",
        "Audio", "Date", "Set", "Map", "Array", "Object", "JSON", "String",
        "Number", "Math", "Promise", "URL", "RegExp", "parseInt", "parseFloat",
        "setTimeout", "setInterval", "clearInterval", "clearTimeout", "alert",
        "prompt", "confirm", "isNaN", "encodeURIComponent", "require", "exec",
        "async", "await", "new", "delete", "void", "in", "of", "do", "else",
    }

    # Strip string and template literals first: prose like "service up (pid"
    # otherwise looks exactly like a call to a function named "up".
    code = re.sub(r"`(?:[^`\\]|\\.)*`", "``", script)
    code = re.sub(r"'(?:[^'\\\n]|\\.)*'", "''", code)
    code = re.sub(r'"(?:[^"\\\n]|\\.)*"', '""', code)

    called = set(re.findall(r"(?<![.\w$])([a-z][A-Za-z0-9_]*)\s*\(", code))
    missing = sorted(called - defined - builtins)
    assert not missing, f"the page calls undefined functions: {missing}"


def test_the_sidebar_shows_day_date_title_and_people():
    """
    "Meeting / 20261101-0759 · 0s" told you nothing — with a dozen in the
    list the only way to tell them apart was the timestamp.
    """
    import console as mod

    page = mod.PAGE
    assert "meetingButton" in page
    assert 'class="when"' in page and 'class="what"' in page and 'class="who"' in page
    assert "nobody identified" in page


# --- port conflicts ---------------------------------------------------------


def test_a_taken_port_moves_to_the_next_one():
    """
    Another local app on the same port was invisible: binding failed, the
    browser opened anyway, and it showed whatever else was listening — which
    looks exactly like this app launching the wrong program.
    """
    from console import ExclusiveHTTPServer, Handler, bind

    # Held by a real listening server, exactly as another local app would.
    # No SO_REUSEADDR here: on Windows that flag is what let a second
    # process bind the same port, which is the bug this guards.
    held = ExclusiveHTTPServer(("127.0.0.1", 0), Handler)
    taken = held.server_address[1]
    try:
        server, port = bind(taken)
        assert server is not None
        assert port != taken, "claimed a port that was already in use"
        server.server_close()
    finally:
        held.server_close()


def test_a_free_port_is_used_as_asked():
    import socket

    from console import bind

    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    free = probe.getsockname()[1]
    probe.close()

    server, port = bind(free)
    assert port == free
    server.server_close()


def test_the_browser_is_only_opened_after_the_port_is_ours():
    import inspect

    import console as mod

    source = inspect.getsource(mod.main)
    bind_at = source.index("server, port = bind(")
    open_at = source.index("webbrowser.open")
    assert bind_at < open_at
    assert "if server is None:" in source, "no guard when binding fails"


def test_the_console_identifies_itself():
    """So a port can be told apart from another app answering on it."""
    import inspect

    import console as mod

    assert mod.IDENTITY == "meetingrec-console"
    assert "IDENTITY" in inspect.getsource(mod.Handler.do_GET)


def test_the_server_never_shares_a_port():
    """
    On Windows SO_REUSEADDR lets a second process bind a port another program
    is already listening on — both bound, and which one answers a given
    connection is arbitrary. That showed up as this console opening and a
    different local app's page appearing.
    """
    from console import ExclusiveHTTPServer

    assert ExclusiveHTTPServer.allow_reuse_address is False
    assert "SO_EXCLUSIVEADDRUSE" in __import__("inspect").getsource(
        ExclusiveHTTPServer.server_bind)


# --- seeking inside a recording ---------------------------------------------


def _serve(tmp_path, data=b"0123456789" * 1000):
    """A console bound to a temp archive with one audio file in it."""
    import threading

    import console as mod

    folder = tmp_path / "20261013-153011-zoom.exe"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "rec.mp3").write_bytes(data)
    mod.STATE.update({"archive": tmp_path, "profiles": tmp_path,
                      "root": tmp_path, "python": sys.executable})
    server, port = mod.bind(0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, port, folder


def test_the_server_advertises_and_honours_byte_ranges(tmp_path):
    """
    Without range support a browser cannot seek: it has the whole file but
    no way to ask for part of it, so every clip plays from the beginning and
    a speaker forty minutes in is unreachable.
    """
    import urllib.request
    from urllib.parse import quote

    data = bytes(range(256)) * 40
    server, port, folder = _serve(tmp_path, data)
    try:
        url = (f"http://127.0.0.1:{port}/api/audio?folder={quote(str(folder))}")

        whole = urllib.request.urlopen(url)
        assert whole.headers["Accept-Ranges"] == "bytes"
        assert whole.read() == data

        req = urllib.request.Request(url, headers={"Range": "bytes=100-199"})
        part = urllib.request.urlopen(req)
        assert part.status == 206
        assert part.headers["Content-Range"] == f"bytes 100-199/{len(data)}"
        body = part.read()
        assert len(body) == 100
        assert body == data[100:200]
    finally:
        server.shutdown()
        server.server_close()


def test_an_open_ended_range_runs_to_the_end(tmp_path):
    import urllib.request
    from urllib.parse import quote

    data = b"abcdefghij" * 50
    server, port, folder = _serve(tmp_path, data)
    try:
        url = f"http://127.0.0.1:{port}/api/audio?folder={quote(str(folder))}"
        req = urllib.request.Request(url, headers={"Range": "bytes=400-"})
        res = urllib.request.urlopen(req)
        assert res.status == 206
        assert res.read() == data[400:]
    finally:
        server.shutdown()
        server.server_close()


def test_a_range_past_the_end_is_refused_properly(tmp_path):
    import urllib.error
    import urllib.request
    from urllib.parse import quote

    server, port, folder = _serve(tmp_path, b"short")
    try:
        url = f"http://127.0.0.1:{port}/api/audio?folder={quote(str(folder))}"
        req = urllib.request.Request(url, headers={"Range": "bytes=9999-"})
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(req)
        assert caught.value.code == 416
    finally:
        server.shutdown()
        server.server_close()


def test_the_player_waits_for_the_seek_before_playing():
    """
    Setting currentTime is a request, not an instruction — playing before
    the browser reports 'seeked' starts from wherever it already was.
    """
    import console as mod

    page = mod.PAGE
    assert "'seeked'" in page
    assert "will not seek" in page, "no warning when a seek is refused"
