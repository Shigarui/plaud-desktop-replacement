"""
meetingrec.postprocess — run diarisation and transcription after a meeting.

Deliberately out-of-process and deferred.

Transcribing a long meeting is many minutes of saturated CPU. Audio capture is
real-time and unforgiving: if the next call starts while Whisper is running,
the contention shows up as dropouts in a recording you cannot redo. So work is
queued, held until nothing is being recorded, and run as a separate process at
below-normal priority. A transcription that finishes late costs nothing; a
recording that drops audio cannot be recovered.

Running it as a subprocess also means a crash in torch or whisper cannot take
the recorder down with it.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger(__name__)

# Windows: keep the console clean and stay out of the recorder's way.
BELOW_NORMAL = 0x00004000
NO_WINDOW = 0x08000000


def parse_window(spec: Optional[str]) -> Optional[tuple[dtime, dtime]]:
    """Parse "18:00-07:00". None means any time."""
    if not spec:
        return None
    try:
        start_s, end_s = spec.split("-")
        start = dtime(*[int(x) for x in start_s.strip().split(":")])
        end = dtime(*[int(x) for x in end_s.strip().split(":")])
        return start, end
    except Exception:
        log.warning("could not read the processing window %r — ignoring", spec)
        return None


def format_window(window: Optional[tuple[dtime, dtime]]) -> str:
    if not window:
        return "any time"
    return f"{window[0].strftime('%H:%M')}-{window[1].strftime('%H:%M')}"


def in_window(window: Optional[tuple[dtime, dtime]],
              now: Optional[dtime] = None) -> bool:
    """
    Is the clock inside the allowed hours?

    An evening window normally crosses midnight — 18:00 to 07:00 — so a plain
    start <= now <= end comparison is wrong for exactly the case people
    actually configure.
    """
    if not window:
        return True
    now = now or datetime.now().time()
    start, end = window
    if start <= end:
        return start <= now <= end
    return now >= start or now <= end


class PostProcessor(threading.Thread):
    """Queues finished meetings and processes them when the machine is idle."""

    def __init__(
        self,
        project_root: Path,
        python_exe: Optional[str] = None,
        delay_seconds: float = 300.0,
        library: Optional[Path] = None,
        me: Optional[str] = None,
        mic_label: Optional[str] = None,
        whisper_model: str = "medium",
        threads: Optional[int] = None,
        window: Optional[str] = None,
        queue_file: Optional[Path] = None,
        summarise: Optional[str] = None,
        busy: Optional[Callable[[], bool]] = None,
    ):
        super().__init__(name="postprocess", daemon=True)
        self.root = Path(project_root)
        self.python = python_exe or sys.executable
        self.delay = delay_seconds
        self.library = Path(library) if library else None
        self.me = me
        self.mic_label = mic_label or "Me"
        self.whisper_model = whisper_model
        self.threads = threads
        # "18:00-07:00" — only work in these hours. Crossing midnight is the
        # normal case for an evening window, so the comparison has to handle
        # a start later than the end.
        self.window = parse_window(window)
        # The queue survives a restart. Without that, stopping the recorder at
        # five o'clock silently discards everything waiting for the evening,
        # and the meetings are never processed at all.
        self.queue_file = Path(queue_file) if queue_file else None
        self.summarise = summarise
        # Asked before each step, so a meeting that starts mid-queue wins.
        self.busy = busy or (lambda: False)

        self._queue: "queue.Queue[tuple[float, Path]]" = queue.Queue()
        self._stopping = threading.Event()

    def submit(self, folder: Path) -> None:
        when = time.monotonic() + self.delay
        # Persist BEFORE queueing. The other order races: the worker can pick
        # the item up, finish it and remove it from the file, and only then
        # does this write it back — so a completed meeting reappears on the
        # next restart and is processed twice.
        self._persist_add(Path(folder))
        self._queue.put((when, Path(folder)))
        if self.window:
            log.info("queued %s — will run in the %s window",
                     Path(folder).name, format_window(self.window))
        else:
            log.info("queued %s for processing in %.0f min",
                     Path(folder).name, self.delay / 60)

    # -- durable queue ---------------------------------------------------

    def _persist_load(self) -> list[Path]:
        """
        Read the queue, retrying a transient open failure.

        On Windows os.replace briefly makes the target unopenable, so a read
        landing in that window raises PermissionError. Treating that the same
        as corruption would return "nothing queued" and silently drop every
        pending meeting — so retry briefly, and only give up on content that
        is genuinely unreadable.
        """
        if not self.queue_file:
            return []
        for attempt in range(4):
            if not self.queue_file.exists():
                return []
            try:
                return [Path(p) for p in json.loads(
                    self.queue_file.read_text(encoding="utf-8"))]
            except OSError:
                time.sleep(0.05 * (attempt + 1))
            except Exception:
                log.warning("queue file is unreadable — starting empty")
                return []
        log.warning("could not read the queue file after several tries")
        return []

    def _persist_write(self, folders: list[Path]) -> None:
        """
        Replace the queue file atomically.

        Writing in place leaves a window where the file is truncated or half
        written. A reader sees invalid JSON, and since _persist_load treats a
        parse failure as "nothing queued", a crash at the wrong moment would
        silently discard every pending meeting. Write a sibling and rename:
        os.replace is atomic, so the file is only ever the old one or the new.
        """
        if not self.queue_file:
            return
        try:
            self.queue_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.queue_file.with_suffix(self.queue_file.suffix + ".tmp")
            tmp.write_text(json.dumps([str(p) for p in folders], indent=2),
                           encoding="utf-8")
            os.replace(tmp, self.queue_file)
        except Exception:
            log.debug("could not write the queue file", exc_info=True)

    def _persist_add(self, folder: Path) -> None:
        existing = self._persist_load()
        if folder not in existing:
            self._persist_write(existing + [folder])

    def _persist_remove(self, folder: Path) -> None:
        self._persist_write([p for p in self._persist_load() if p != folder])

    def stop(self) -> None:
        self._stopping.set()

    # -- worker ----------------------------------------------------------

    def run(self) -> None:
        # Anything left over from a previous session is due immediately.
        pending: list[tuple[float, Path]] = [
            (time.monotonic(), folder) for folder in self._persist_load()]
        if pending:
            log.info("%d meeting(s) carried over from last time", len(pending))
        # Poll briefly rather than blocking on the queue: an item already
        # waiting becomes due while we are parked, and a long block means the
        # delay is really "delay plus however long we were asleep".
        while not self._stopping.is_set():
            try:
                pending.append(self._queue.get(timeout=0.25))
            except queue.Empty:
                pass

            if not pending:
                continue
            now = time.monotonic()
            ready = [item for item in pending if item[0] <= now]
            if not ready:
                continue
            # Two guards, not one. Don't START during a meeting — loading the
            # model is itself heavy, and launching work only to freeze it
            # helps nobody. Work already running is suspended instead, in
            # _run_script, because killing it would discard real progress.
            if self.busy():
                continue
            if not in_window(self.window):
                continue  # outside working hours for this; it keeps

            when, folder = ready[0]
            pending.remove((when, folder))
            try:
                self._process(folder)
            except Exception:
                log.exception("post-processing failed for %s", folder.name)
            finally:
                # Removed either way: a meeting that fails twice will fail a
                # third time, and retrying forever would block everything behind it.
                self._persist_remove(folder)

    def _run_script(self, name: str, args: list[str]) -> bool:
        """
        Run a step, freezing it whenever a meeting starts.

        Suspending beats killing. Whisper keeps no checkpoint, so terminating
        a long transcription discards every minute of work already done and
        the retry begins at zero — under repeated interruption it might never
        finish at all. A suspended process uses no CPU, keeps its memory, and
        resumes exactly where it stopped.
        """
        script = self.root / "scripts" / name
        if not script.exists():
            log.warning("%s not found — skipping", name)
            return False

        flags = BELOW_NORMAL | NO_WINDOW if sys.platform == "win32" else 0
        started = time.time()
        proc = subprocess.Popen(
            [self.python, str(script), *args],
            cwd=str(self.root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, creationflags=flags,
        )

        suspended = False
        paused_for = 0.0
        paused_at = 0.0
        try:
            while proc.poll() is None:
                if self._stopping.is_set():
                    self._set_suspended(proc.pid, False)
                    proc.terminate()
                    return False
                want_paused = self.busy()
                if want_paused and not suspended:
                    if self._set_suspended(proc.pid, True):
                        suspended, paused_at = True, time.monotonic()
                        log.info("%s suspended — a meeting started", name)
                elif not want_paused and suspended:
                    if self._set_suspended(proc.pid, False):
                        suspended = False
                        paused_for += time.monotonic() - paused_at
                        log.info("%s resumed after %.0fs", name, paused_for)
                time.sleep(1.0)
            output = proc.stdout.read() if proc.stdout else ""
        finally:
            if suspended:
                self._set_suspended(proc.pid, False)

        if proc.returncode != 0:
            tail = (output or "").strip().splitlines()[-3:]
            log.error("%s failed (%d): %s", name, proc.returncode, " | ".join(tail))
            return False
        note = f" ({paused_for:.0f}s paused)" if paused_for else ""
        log.info("%s finished in %.0fs%s", name, time.time() - started, note)
        return True

    @staticmethod
    def _set_suspended(pid: int, suspend: bool) -> bool:
        try:
            import psutil

            proc = psutil.Process(pid)
            for child in proc.children(recursive=True):
                try:
                    child.suspend() if suspend else child.resume()
                except Exception:
                    pass
            proc.suspend() if suspend else proc.resume()
            return True
        except Exception as exc:
            log.debug("could not %s %d: %r",
                      "suspend" if suspend else "resume", pid, exc)
            return False

    def _process(self, folder: Path) -> None:
        if not folder.exists():
            log.warning("%s has gone — skipping", folder.name)
            return
        log.info("post-processing %s", folder.name)

        if self.library and self.library.exists():
            args = [str(folder), "--library", str(self.library)]
            if self.me:
                args += ["--me", self.me]
            self._run_script("diarize.py", args)
        else:
            log.info("no voiceprint library configured — skipping diarisation")

        args = [str(folder), "--model", self.whisper_model, "--me", self.mic_label]
        if self.threads:
            args += ["--threads", str(self.threads)]
        if self.summarise:
            args += ["--summarise", self.summarise]
        self._run_script("transcribe.py", args)

        transcript = folder / "transcript.md"
        if transcript.exists():
            log.info("ready: %s", transcript)
