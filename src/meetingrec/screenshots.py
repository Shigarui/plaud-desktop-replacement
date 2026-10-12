"""
meetingrec.screenshots — visual context from the meeting window.

Captures what was on screen during a call: shared slides, demos, whiteboards.
Audio tells you what was said; this tells you what was pointed at.

Three decisions worth understanding, because the obvious versions are wrong.

PINNED TO THE WINDOW, NOT THE MONITOR. The detector already knows which app
started the meeting, so only that window's rectangle is captured. Your inbox,
Slack and notes sit outside it and never reach the archive. Capturing the whole
screen would be simpler and considerably more invasive.

FOREGROUND ONLY. Teams and Zoom render through the GPU, and PrintWindow
returns black frames for composited surfaces, so capturing an occluded window
gives you nothing useful. Reading the screen where the window sits is reliable
— but only honest when the window is actually on top. So when you alt-tab away
capture pauses, which also means whatever you switched to is never captured.

CHANGE-TRIGGERED, NOT ON A TIMER. A fixed interval produces ninety near
identical frames of one slide. Each frame is perceptually hashed and kept only
when it differs enough from the last, with a floor so a video call does not
save a frame per second. The same meeting yields a dozen images that each show
something new.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# Consent-store identities map to the executables that own the window. The
# identity is what the detector reports; the process name is what Windows
# calls the running program, and they are rarely the same for packaged apps.
APP_PROCESSES: dict[str, tuple[str, ...]] = {
    "msteams_8wekyb3d8bbwe": ("ms-teams.exe", "msteams.exe", "Teams.exe"),
    "microsoft.teams_8wekyb3d8bbwe": ("Teams.exe", "ms-teams.exe"),
    "teams.exe": ("Teams.exe", "ms-teams.exe"),
    "ms-teams.exe": ("ms-teams.exe", "Teams.exe"),
    "zoom.exe": ("Zoom.exe", "zoom.exe"),
    "webexmta.exe": ("webexmta.exe", "CiscoCollabHost.exe"),
    "slack.exe": ("slack.exe",),
}

# Window titles worth preferring when a process owns several windows. Zoom in
# particular keeps its main panel open alongside the meeting itself.
TITLE_HINTS = ("meeting", "zoom", "teams", "call", "webex", "huddle")

# Never captured, whatever is in the foreground. Following the foreground means
# anything you look at during a call can be archived, so the obviously
# sensitive cases are excluded by default. Matched case-insensitively against
# the window title and the process name.
DEFAULT_EXCLUDE = (
    "password", "1password", "bitwarden", "keepass", "lastpass", "dashlane",
    "authenticator", "credential", "bank", "payroll", "incognito", "private browsing",
)


@dataclass
class ShotStats:
    captured: int = 0
    skipped_unchanged: int = 0
    skipped_background: int = 0
    skipped_excluded: int = 0
    errors: int = 0
    window_title: str = ""
    paths: list = field(default_factory=list)
    shots: list = field(default_factory=list)  # {file, title, process, offset_s}


# --------------------------------------------------------------------------
# window lookup (Win32)
# --------------------------------------------------------------------------


def _user32():
    return ctypes.WinDLL("user32", use_last_error=True)


def find_window_for_processes(process_names: tuple[str, ...]) -> Optional[tuple[int, str]]:
    """
    Largest visible top-level window belonging to any of these processes.

    Largest rather than first: meeting apps keep small helper windows around
    (toolbars, notification popups) and the meeting itself is the big one.
    Returns (hwnd, title) or None.
    """
    try:
        import psutil
    except ImportError:
        log.warning("psutil missing — screenshots need the [screenshots] extra")
        return None

    wanted = {n.lower() for n in process_names}
    pids = set()
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            if (proc.info["name"] or "").lower() in wanted:
                pids.add(proc.info["pid"])
        except Exception:
            continue
    if not pids:
        return None

    u = _user32()
    found: list[tuple[int, int, str]] = []  # (area, hwnd, title)

    CB = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)

    def visit(hwnd, _lparam):
        if not u.IsWindowVisible(hwnd):
            return True
        pid = wt.DWORD()
        u.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value not in pids:
            return True
        length = u.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        u.GetWindowTextW(hwnd, buf, length + 1)
        rect = wt.RECT()
        u.GetWindowRect(hwnd, ctypes.byref(rect))
        w, h = rect.right - rect.left, rect.bottom - rect.top
        if w < 200 or h < 200:
            return True  # toolbars and popups
        found.append((w * h, hwnd, buf.value))
        return True

    u.EnumWindows(CB(visit), 0)
    if not found:
        return None

    hinted = [f for f in found if any(h in f[2].lower() for h in TITLE_HINTS)]
    best = max(hinted or found, key=lambda f: f[0])
    return best[1], best[2]


def window_rect(hwnd: int) -> Optional[tuple[int, int, int, int]]:
    """(left, top, width, height), or None if the window has gone."""
    u = _user32()
    rect = wt.RECT()
    if not u.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None
    w, h = rect.right - rect.left, rect.bottom - rect.top
    if w <= 0 or h <= 0:
        return None
    return rect.left, rect.top, w, h


def is_foreground(hwnd: int) -> bool:
    return _user32().GetForegroundWindow() == hwnd


def foreground_window() -> Optional[tuple[int, str, str]]:
    """
    (hwnd, title, process_name) of whatever is on top, or None.

    Used when following the foreground rather than pinning to the meeting
    window — while you present, you are looking at the thing you are
    presenting, and that is a different window entirely.
    """
    u = _user32()
    hwnd = u.GetForegroundWindow()
    if not hwnd:
        return None
    length = u.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return None
    buf = ctypes.create_unicode_buffer(length + 1)
    u.GetWindowTextW(hwnd, buf, length + 1)

    name = ""
    try:
        import psutil

        pid = wt.DWORD()
        u.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        name = psutil.Process(pid.value).name()
    except Exception:
        pass
    return hwnd, buf.value, name


# --------------------------------------------------------------------------
# change detection
# --------------------------------------------------------------------------


def _box_downscale(gray: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Area-average down to a tiny grid.

    Sampling individual pixels instead is much faster and quite wrong: thin
    content — a line of text, a chart axis — falls between the sample points
    and vanishes, so two visibly different slides hash within a few bits of
    each other. Averaging every pixel into its cell means everything on screen
    contributes something.
    """
    h, w = gray.shape
    ys = np.linspace(0, h, height + 1).astype(int)
    xs = np.linspace(0, w, width + 1).astype(int)
    out = np.empty((height, width), dtype=np.float64)
    for i in range(height):
        for j in range(width):
            cell = gray[ys[i]:max(ys[i + 1], ys[i] + 1),
                        xs[j]:max(xs[j + 1], xs[j] + 1)]
            out[i, j] = cell.mean() if cell.size else 0.0
    return out


def dhash(image: np.ndarray, size: int = 8) -> int:
    """
    Difference hash: downscale, compare each cell to its right neighbour.

    Robust to the things that do not matter — a cursor moving, video noise,
    a timer ticking — and sensitive to the thing that does, which is the whole
    frame changing because someone advanced a slide.
    """
    if image.ndim == 3:
        image = image[..., :3].mean(axis=2)
    small = _box_downscale(image.astype(np.float64), size + 1, size)
    bits = small[:, 1:] > small[:, :-1]
    out = 0
    for bit in bits.flatten():
        out = (out << 1) | int(bit)
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# --------------------------------------------------------------------------
# capturer
# --------------------------------------------------------------------------


class ScreenshotCapturer(threading.Thread):
    """
    Captures the meeting window while armed. Start once, arm per meeting.

    Failures here must never affect the recording: a missing window, a closed
    monitor or a permissions problem logs and moves on.
    """

    def __init__(
        self,
        interval: float = 3.0,
        min_gap_seconds: float = 15.0,
        change_threshold: int = 12,
        quality: int = 70,
        max_per_meeting: int = 200,
        foreground_only: bool = True,
        follow_foreground: bool = False,
        exclude: tuple[str, ...] = DEFAULT_EXCLUDE,
    ):
        super().__init__(name="screenshots", daemon=True)
        self.interval = interval
        self.min_gap_seconds = min_gap_seconds
        self.change_threshold = change_threshold
        self.quality = quality
        self.max_per_meeting = max_per_meeting
        self.foreground_only = foreground_only
        # Capture whatever is on top rather than only the meeting window. Needed
        # to catch your own screen shares: the app you are sharing renders in
        # its own window, and the meeting window shows only a "you are sharing"
        # banner. The cost is that anything else you look at is fair game too,
        # which is what `exclude` is for.
        self.follow_foreground = follow_foreground
        self.exclude = tuple(e.lower() for e in exclude)

        self._stopping = threading.Event()
        self._active = threading.Event()
        self._lock = threading.Lock()
        self._out_dir: Optional[Path] = None
        self._slug = ""
        self._app = ""
        self._last_hash: Optional[int] = None
        self._last_saved = 0.0
        self._armed_at = 0.0
        self.stats = ShotStats()

    # -- lifecycle -------------------------------------------------------

    def arm(self, out_dir: Path, slug: str, app: str) -> None:
        with self._lock:
            self._out_dir = Path(out_dir)
            self._out_dir.mkdir(parents=True, exist_ok=True)
            self._slug = slug
            self._app = app
            self._last_hash = None
            self._last_saved = 0.0
            self._armed_at = time.monotonic()
            self.stats = ShotStats()
        self._active.set()
        log.info("screenshots armed for %s", app)

    def disarm(self) -> ShotStats:
        self._active.clear()
        with self._lock:
            stats = self.stats
        if stats.captured or stats.errors:
            log.info("screenshots: %d captured, %d unchanged, %d background, "
                     "%d excluded, %d errors", stats.captured, stats.skipped_unchanged,
                     stats.skipped_background, stats.skipped_excluded, stats.errors)

        # A manifest makes the images navigable later: which window each shot
        # came from and how far into the meeting it was taken.
        with self._lock:
            out_dir, slug = self._out_dir, self._slug
        if stats.shots and out_dir:
            try:
                import json

                manifest = out_dir / f"{slug}-shots.json"
                manifest.write_text(json.dumps(stats.shots, indent=2), encoding="utf-8")
                stats.paths.append(manifest)
            except Exception:
                log.exception("could not write screenshot manifest")
        return stats

    def stop(self) -> None:
        self._stopping.set()

    # -- capture loop ----------------------------------------------------

    def run(self) -> None:
        try:
            import mss  # noqa: F401
        except ImportError:
            log.warning("mss not installed — screenshots disabled "
                        "(pip install -e \".[screenshots]\")")
            return

        while not self._stopping.is_set():
            if not self._active.wait(0.5):
                continue
            try:
                self._tick()
            except Exception:
                with self._lock:
                    self.stats.errors += 1
                log.debug("screenshot tick failed", exc_info=True)
            self._stopping.wait(self.interval)

    def _is_excluded(self, title: str, process: str) -> bool:
        haystack = f"{title} {process}".lower()
        return any(term in haystack for term in self.exclude)

    def _pick_window(self) -> Optional[tuple[int, str, str]]:
        """
        The window to capture this tick, as (hwnd, title, process).

        NOT named _target: threading.Thread.__init__ assigns self._target for
        the callable a thread runs, so a method of that name is shadowed by an
        instance attribute of None and fails only when called.
        """
        with self._lock:
            app = self._app

        if self.follow_foreground:
            return foreground_window()

        processes = APP_PROCESSES.get(app.lower())
        if not processes:
            processes = (app,) if app.lower().endswith(".exe") else ()
            if not processes:
                return None
        hit = find_window_for_processes(processes)
        if hit is None:
            return None
        hwnd, title = hit
        if self.foreground_only and not is_foreground(hwnd):
            with self._lock:
                self.stats.skipped_background += 1
            return None
        return hwnd, title, ""

    def _tick(self) -> None:
        with self._lock:
            if self._out_dir is None or self.stats.captured >= self.max_per_meeting:
                return
            out_dir, slug, started = self._out_dir, self._slug, self._armed_at

        target = self._pick_window()
        if target is None:
            return
        hwnd, title, process = target

        if self._is_excluded(title, process):
            with self._lock:
                self.stats.skipped_excluded += 1
            return

        rect = window_rect(hwnd)
        if rect is None:
            return

        frame = self._grab(rect)
        if frame is None:
            return

        digest = dhash(frame)
        now = time.monotonic()
        with self._lock:
            unchanged = (self._last_hash is not None
                         and hamming(digest, self._last_hash) < self.change_threshold)
            too_soon = (now - self._last_saved) < self.min_gap_seconds
            if unchanged or (too_soon and self._last_hash is not None):
                self.stats.skipped_unchanged += 1
                self._last_hash = digest
                return
            index = self.stats.captured + 1

        path = out_dir / f"{slug}-shot{index:03d}.jpg"
        if self._save(frame, path):
            with self._lock:
                self._last_hash = digest
                self._last_saved = now
                self.stats.captured += 1
                self.stats.window_title = title
                self.stats.paths.append(path)
                self.stats.shots.append({
                    "file": path.name,
                    "title": title,
                    "process": process,
                    "offset_seconds": round(now - started, 1) if started else None,
                })

    @staticmethod
    def _grab(rect: tuple[int, int, int, int]) -> Optional[np.ndarray]:
        import mss

        left, top, width, height = rect
        with mss.mss() as sct:
            raw = sct.grab({"left": left, "top": top, "width": width, "height": height})
        return np.asarray(raw)[..., :3]  # BGRA -> BGR

    def _save(self, frame: np.ndarray, path: Path) -> bool:
        try:
            from PIL import Image

            Image.fromarray(frame[..., ::-1]).save(  # BGR -> RGB
                path, "JPEG", quality=self.quality, optimize=True)
            return True
        except Exception:
            with self._lock:
                self.stats.errors += 1
            log.exception("could not write %s", path)
            return False
