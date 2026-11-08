"""Single-instance tests — the lock that stops two recorders fighting."""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from meetingrec.singleton import AlreadyRunning, SingleInstance


def test_first_acquire_writes_our_pid(tmp_path):
    lock = SingleInstance(tmp_path / "x.pid")
    with lock:
        assert (tmp_path / "x.pid").read_text().strip() == str(os.getpid())
    assert not (tmp_path / "x.pid").exists(), "lock not cleared on exit"


def test_a_second_holder_is_refused_while_the_first_lives(tmp_path):
    """
    The whole point. Two recorders fight over the audio endpoints, Windows
    chimes on every acquisition, and both capture worse audio than one.
    """
    path = tmp_path / "x.pid"
    first = SingleInstance(path)
    first.acquire()
    try:
        with pytest.raises(AlreadyRunning):
            SingleInstance(path).acquire()
    finally:
        first.release()


def test_the_lock_is_free_again_after_release(tmp_path):
    path = tmp_path / "x.pid"
    with SingleInstance(path):
        pass
    with SingleInstance(path):  # must not raise
        pass


def test_the_holder_pid_is_recorded_for_the_log(tmp_path):
    path = tmp_path / "x.pid"
    with SingleInstance(path):
        assert path.read_text().strip() == str(os.getpid())


def test_a_crash_leaves_no_stale_lock_behind(tmp_path):
    """
    A kernel lock dies with the process, so there is no stale state to reason
    about — which is what every PID-file version of this got wrong.
    """
    path = tmp_path / "x.pid"
    code = (
        "import sys, os;"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parents[1] / 'src')!r});"
        "from meetingrec.singleton import SingleInstance;"
        f"l = SingleInstance({str(path)!r}); l.acquire();"
        "os._exit(1)"  # die holding it, no cleanup
    )
    subprocess.run([sys.executable, "-c", code], check=False, timeout=30)

    with SingleInstance(path):  # must not be blocked by the dead holder
        assert path.read_text().strip() == str(os.getpid())


def test_only_one_of_many_contenders_wins(tmp_path):
    """Simultaneous starts: exactly one claim, the rest told it is taken."""
    import threading

    path = tmp_path / "x.pid"
    winners, losers, errors = [], [], []
    start = threading.Barrier(8)

    def contend():
        start.wait()
        lock = SingleInstance(path)
        try:
            lock.acquire()
            winners.append(lock)
        except AlreadyRunning:
            losers.append(1)
        except Exception as exc:  # nothing else should ever come out
            errors.append(exc)

    threads = [threading.Thread(target=contend) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, f"unexpected errors: {errors}"
    assert len(winners) == 1, f"{len(winners)} contenders claimed the lock"
    assert len(losers) == 7
    for lock in winners:
        lock.release()


def test_the_pid_stays_readable_while_the_lock_is_held(tmp_path):
    """
    msvcrt.locking takes a MANDATORY lock on Windows — a locked byte range
    cannot be read by anyone, including the holder. Locking byte 0 therefore
    made the PID unreadable and broke every reader. The lock sits far past
    the content instead.
    """
    path = tmp_path / "x.pid"
    lock = SingleInstance(path)
    lock.acquire()
    try:
        assert path.read_text().strip() == str(os.getpid())
        # and a fresh handle, as another process would use
        with open(path, "r", encoding="utf-8") as handle:
            assert handle.read().strip() == str(os.getpid())
    finally:
        lock.release()


def test_the_refusal_names_the_holder(tmp_path):
    """The log has to say who has it, which needs the PID to be readable."""
    path = tmp_path / "x.pid"
    first = SingleInstance(path)
    first.acquire()
    try:
        with pytest.raises(AlreadyRunning) as caught:
            SingleInstance(path).acquire()
        assert caught.value.pid == os.getpid()
    finally:
        first.release()
