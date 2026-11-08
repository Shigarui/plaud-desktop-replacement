"""
meetingrec.singleton — refuse to run twice.

Two recorders on one machine is not a harmless duplicate. They fight over the
same audio endpoints, each acquisition makes Windows play its device-connect
chime, and both capture degraded audio because neither has exclusive access.
It is also easy to arrive at: a scheduled task plus a manual run, or a task
that fires more than once.

The lock is enforced by the operating system rather than inferred from a PID
file, because every inference-based version of this was wrong in a different
way. See SingleInstance for the specific failures.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


class AlreadyRunning(RuntimeError):
    def __init__(self, pid: int):
        super().__init__(f"meetingrec is already running (pid {pid})")
        self.pid = pid


class SingleInstance:
    """
    An OS-enforced exclusive lock, held for the lifetime of the run.

    Earlier attempts here were all variations on "look at a PID file and
    decide", and each failed in a way worth recording:

      exists() then create   — two processes starting in the same millisecond
                               both pass the check and both write.
      O_CREAT|O_EXCL         — atomic, but the loser has to decide whether the
                               file is stale, and "delete it and retry" means
                               under contention everyone deletes everyone
                               else's lock and all of them proceed.
      PID liveness checks    — PIDs are recycled, and probing them portably is
                               worse than it looks: os.kill(pid, 0) terminates
                               the process on Windows.

    A real lock asks the kernel. The file stays open for the whole run, and
    the lock dies with the process — so a crash leaves nothing to clean up
    and no stale state to reason about. The PID is still written, but only so
    the log can name who holds it.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.acquired = False
        self._fd: Optional[int] = None

    # The byte that is actually locked, chosen to sit far past any content.
    # msvcrt.locking takes a MANDATORY lock on Windows: a locked region cannot
    # be read by anyone, including us, so locking byte 0 makes the PID we just
    # wrote unreadable. flock on POSIX is advisory and has no such problem,
    # but locking the same far-away byte on both keeps the two paths honest.
    LOCK_OFFSET = 1 << 20

    def _lock_fd(self, fd: int) -> None:
        if os.name == "nt":
            import msvcrt

            os.lseek(fd, self.LOCK_OFFSET, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock_fd(self, fd: int) -> None:
        try:
            if os.name == "nt":
                import msvcrt

                os.lseek(fd, self.LOCK_OFFSET, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_UN)
        except Exception:
            log.debug("could not release the lock cleanly", exc_info=True)

    def _read_holder(self) -> Optional[int]:
        try:
            return int(self.path.read_text(encoding="utf-8").strip())
        except Exception:
            return None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR)
        try:
            self._lock_fd(fd)
        except OSError as exc:
            holder = self._read_holder()
            os.close(fd)
            raise AlreadyRunning(holder or 0) from exc

        try:
            os.truncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, str(os.getpid()).encode())
        except Exception:
            log.debug("could not record the pid in the lock file", exc_info=True)

        self._fd = fd
        self.acquired = True

    def release(self) -> None:
        if not self.acquired or self._fd is None:
            return
        self._unlock_fd(self._fd)
        try:
            os.close(self._fd)
        except Exception:
            pass
        self._fd = None
        self.acquired = False
        try:
            self.path.unlink()
        except Exception:
            pass  # another holder may already own it; the lock is what counts

    def __enter__(self) -> "SingleInstance":
        self.acquire()
        return self

    def __exit__(self, *exc) -> bool:
        self.release()
        return False
