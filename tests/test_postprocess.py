"""Post-processor tests — queueing and deferral, no whisper involved."""

import json
import time
from pathlib import Path

import pytest

from meetingrec.postprocess import PostProcessor


def wait_until(predicate, timeout=15.0, interval=0.05):
    """
    Poll for a condition instead of sleeping a guessed amount.

    These tests drive a worker that polls four times a second; fixed sleeps
    pass on an idle machine and fail on a busy one. A flaky test gets ignored,
    and an ignored test catches nothing.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def hold(predicate, duration=0.6, interval=0.05):
    """Assert something stays true for a while — for 'must NOT happen yet'."""
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        if not predicate():
            return False
        time.sleep(interval)
    return True


@pytest.fixture
def project(tmp_path):
    (tmp_path / "scripts").mkdir()
    return tmp_path


def test_work_is_deferred_by_the_delay(project):
    """
    Nothing should start the instant a meeting ends — Drive may still be
    syncing, and the user may be about to join another call.
    """
    done = []
    p = PostProcessor(project, delay_seconds=0.4)
    p._process = lambda folder: done.append(folder)
    p.start()
    try:
        p.submit(project)
        assert hold(lambda: not done, 0.3), "started before the delay elapsed"
        assert wait_until(lambda: done == [project])
    finally:
        p.stop()
        p.join(timeout=2)


def test_processing_does_not_start_while_a_meeting_is_recording(project):
    """
    Whisper saturates the CPU; contention during capture shows up as dropouts
    in a recording that cannot be redone. Loading the model is heavy too, so
    the work must not even begin until the machine is free.
    """
    done = []
    recording = {"now": True}
    p = PostProcessor(project, delay_seconds=0.1, busy=lambda: recording["now"])
    p._process = lambda folder: done.append(folder)
    p.start()
    try:
        p.submit(project)
        assert hold(lambda: not done), "ran while a meeting was being recorded"
        recording["now"] = False
        assert wait_until(lambda: done == [project])
    finally:
        p.stop()
        p.join(timeout=2)


def test_several_meetings_queue_up(project):
    done = []
    p = PostProcessor(project, delay_seconds=0.2)
    p._process = lambda folder: done.append(folder.name)
    p.start()
    try:
        for name in ("m1", "m2", "m3"):
            folder = project / name
            folder.mkdir()
            p.submit(folder)
        assert wait_until(lambda: sorted(done) == ["m1", "m2", "m3"])
    finally:
        p.stop()
        p.join(timeout=2)


def test_a_failure_does_not_kill_the_worker(project):
    """One bad meeting must not stop every later one being processed."""
    done = []

    def flaky(folder):
        if folder.name == "bad":
            raise RuntimeError("torch exploded")
        done.append(folder.name)

    p = PostProcessor(project, delay_seconds=0.1)
    p._process = flaky
    p.start()
    try:
        for name in ("bad", "good"):
            folder = project / name
            folder.mkdir()
            p.submit(folder)
        assert wait_until(lambda: done == ["good"])
    finally:
        p.stop()
        p.join(timeout=2)


def test_a_vanished_folder_is_skipped(project):
    p = PostProcessor(project, delay_seconds=0.0)
    p._process(project / "not-there")  # must not raise


def test_missing_script_is_reported_not_fatal(project):
    p = PostProcessor(project, delay_seconds=0.0)
    assert p._run_script("nope.py", []) is False


def test_diarisation_is_skipped_without_a_library(project):
    calls = []
    p = PostProcessor(project, delay_seconds=0.0, library=None)
    p._run_script = lambda name, args: calls.append(name) or True
    folder = project / "m"
    folder.mkdir()
    p._process(folder)
    assert calls == ["transcribe.py"], "tried to diarise with no library"


def test_both_steps_run_when_a_library_exists(project):
    lib = project / "voiceprints.json"
    lib.write_text("{}")
    calls = []
    p = PostProcessor(project, delay_seconds=0.0, library=lib, me="Grayson_Terry")
    p._run_script = lambda name, args: calls.append((name, args)) or True
    folder = project / "m"
    folder.mkdir()
    p._process(folder)
    assert [c[0] for c in calls] == ["diarize.py", "transcribe.py"]
    assert "--library" in calls[0][1]
    assert "--me" in calls[0][1]


# --- suspend and resume ----------------------------------------------------


def test_running_work_is_suspended_not_killed(project, monkeypatch):
    """
    Whisper keeps no checkpoint. Killing a long transcription throws away
    every minute already spent and the retry starts from zero — under
    repeated interruption it may never finish. Freezing costs nothing.
    """
    import subprocess as sp

    (project / "scripts").mkdir(exist_ok=True)
    script = project / "scripts" / "slow.py"
    script.write_text("import time\nfor _ in range(60): time.sleep(0.05)\n")

    actions = []
    recording = {"now": False}
    p = PostProcessor(project, delay_seconds=0.0, busy=lambda: recording["now"])
    monkeypatch.setattr(PostProcessor, "_set_suspended",
                        staticmethod(lambda pid, suspend: actions.append(suspend) or True))

    thread = __import__("threading").Thread(
        target=lambda: p._run_script("slow.py", []), daemon=True)
    thread.start()
    time.sleep(0.5)
    recording["now"] = True
    assert wait_until(lambda: True in actions), "did not suspend when a meeting started"
    recording["now"] = False
    assert wait_until(lambda: False in actions), "did not resume after the meeting"
    thread.join(timeout=15)


def test_no_suspend_calls_when_nothing_interrupts(project, monkeypatch):
    (project / "scripts").mkdir(exist_ok=True)
    (project / "scripts" / "quick.py").write_text("print('done')\n")
    actions = []
    p = PostProcessor(project, delay_seconds=0.0, busy=lambda: False)
    monkeypatch.setattr(PostProcessor, "_set_suspended",
                        staticmethod(lambda pid, suspend: actions.append(suspend) or True))
    assert p._run_script("quick.py", []) is True
    assert actions == []


def test_thread_cap_is_passed_to_transcription(project):
    calls = []
    p = PostProcessor(project, delay_seconds=0.0, threads=6)
    p._run_script = lambda name, args: calls.append(args) or True
    folder = project / "m"
    folder.mkdir()
    p._process(folder)
    assert "--threads" in calls[-1] and "6" in calls[-1]


def test_default_threads_leaves_headroom():
    """Half the cores, so contention mostly never arises in the first place."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from transcribe import default_threads

    import os
    assert 2 <= default_threads() <= max(2, (os.cpu_count() or 4))
    assert default_threads() <= (os.cpu_count() or 4) // 2 or default_threads() == 2


# --- quiet hours -----------------------------------------------------------

from datetime import time as dtime  # noqa: E402

from meetingrec.postprocess import format_window, in_window, parse_window  # noqa: E402


def test_window_crossing_midnight_is_handled():
    """
    The case people actually configure. 18:00-07:00 wraps, and a naive
    start <= now <= end comparison gets every hour of it wrong.
    """
    w = parse_window("18:00-07:00")
    assert in_window(w, dtime(19, 30))
    assert in_window(w, dtime(2, 0))
    assert in_window(w, dtime(6, 59))
    assert not in_window(w, dtime(9, 0))
    assert not in_window(w, dtime(17, 59))


def test_same_day_window():
    w = parse_window("09:00-17:00")
    assert in_window(w, dtime(12, 0))
    assert not in_window(w, dtime(20, 0))


def test_no_window_means_any_time():
    assert in_window(None, dtime(3, 0))
    assert parse_window(None) is None
    assert parse_window("") is None


def test_a_malformed_window_is_ignored_not_fatal():
    assert parse_window("evenings please") is None
    assert parse_window("18:00") is None


def test_format_window_reads_back():
    assert format_window(parse_window("18:00-07:00")) == "18:00-07:00"
    assert format_window(None) == "any time"


def test_work_waits_for_the_window(project, monkeypatch):
    from meetingrec import postprocess as mod

    done = []
    allowed = {"now": False}
    monkeypatch.setattr(mod, "in_window", lambda w, now=None: allowed["now"])
    p = PostProcessor(project, delay_seconds=0.0, window="18:00-07:00")
    p._process = lambda folder: done.append(folder)
    p.start()
    try:
        p.submit(project)
        assert hold(lambda: not done), "ran outside the window"
        allowed["now"] = True
        assert wait_until(lambda: done == [project])
    finally:
        p.stop()
        p.join(timeout=2)


# --- durable queue ---------------------------------------------------------


def test_queue_survives_a_restart(project):
    """
    Without this, stopping the recorder before the evening window silently
    throws away every meeting waiting to be processed.
    """
    qfile = project / "queue.json"
    first = PostProcessor(project, delay_seconds=999, queue_file=qfile)
    folder = project / "m1"
    folder.mkdir()
    first.submit(folder)
    assert qfile.exists()

    done = []
    second = PostProcessor(project, delay_seconds=999, queue_file=qfile)
    second._process = lambda f: done.append(f)
    second.start()
    try:
        assert wait_until(lambda: done == [folder]), \
            "carried-over work was not picked up"
    finally:
        second.stop()
        second.join(timeout=2)


def test_processed_meetings_leave_the_queue(project):
    qfile = project / "queue.json"
    p = PostProcessor(project, delay_seconds=0.0, queue_file=qfile)
    p._process = lambda f: None
    p.start()
    try:
        folder = project / "m"
        folder.mkdir()
        p.submit(folder)
        assert wait_until(lambda: json.loads(qfile.read_text()) == [])
    finally:
        p.stop()
        p.join(timeout=2)


def test_a_failing_meeting_does_not_block_the_queue_forever(project):
    """A meeting that fails twice will fail a third time; drop it."""
    qfile = project / "queue.json"
    p = PostProcessor(project, delay_seconds=0.0, queue_file=qfile)

    def boom(folder):
        raise RuntimeError("nope")

    p._process = boom
    p.start()
    try:
        folder = project / "bad"
        folder.mkdir()
        p.submit(folder)
        assert wait_until(lambda: json.loads(qfile.read_text()) == [])
    finally:
        p.stop()
        p.join(timeout=2)


def test_a_completed_meeting_does_not_reappear(project):
    """
    Regression. submit() used to queue before persisting, so with no delay the
    worker could finish and clear the entry before submit wrote it — leaving a
    completed meeting in the file to be processed again after a restart.
    """
    qfile = project / "queue.json"
    processed = []
    p = PostProcessor(project, delay_seconds=0.0, queue_file=qfile)
    p._process = lambda f: processed.append(f.name)
    p.start()
    try:
        folder = project / "m"
        folder.mkdir()
        p.submit(folder)
        assert wait_until(lambda: processed == ["m"])
        assert hold(lambda: json.loads(qfile.read_text()) == []), \
            "a finished meeting was written back into the queue"
    finally:
        p.stop()
        p.join(timeout=2)


def test_the_queue_file_is_never_seen_half_written(project):
    """
    _persist_load treats a parse failure as "nothing queued", so a reader
    catching a half-written file would silently discard every pending
    meeting. The replace has to be atomic.
    """
    import json as _json
    import threading as _threading

    qfile = project / "queue.json"
    p = PostProcessor(project, delay_seconds=999, queue_file=qfile)
    folders = [project / f"m{i}" for i in range(40)]
    for f in folders:
        f.mkdir()

    bad = []
    stop = _threading.Event()

    def reader():
        while not stop.is_set():
            if qfile.exists():
                try:
                    _json.loads(qfile.read_text())
                except _json.JSONDecodeError:
                    bad.append(1)          # torn content — the real failure
                except OSError:
                    pass                   # Windows: unopenable during replace

    t = _threading.Thread(target=reader, daemon=True)
    t.start()
    for i in range(len(folders)):
        p._persist_write(folders[: i + 1])
    stop.set()
    t.join(timeout=5)
    assert not bad, f"{len(bad)} reads saw a partially written queue file"


def test_a_transient_open_failure_does_not_empty_the_queue(project, monkeypatch):
    """
    On Windows os.replace makes the target briefly unopenable. Treating that
    like corruption would return "nothing queued" and silently drop every
    pending meeting.
    """
    qfile = project / "queue.json"
    p = PostProcessor(project, delay_seconds=999, queue_file=qfile)
    folder = project / "m"
    folder.mkdir()
    p._persist_add(folder)

    calls = {"n": 0}
    real = Path.read_text

    def flaky(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError("busy")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky)
    assert p._persist_load() == [folder], "a transient failure emptied the queue"
