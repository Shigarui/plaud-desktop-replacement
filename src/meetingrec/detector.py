"""
meetingrec.detector — detect when a virtual meeting starts/stops on Windows.

Primary signal: the Windows capability-access consent store. Every time an app
opens the microphone, Windows writes LastUsedTimeStart/LastUsedTimeStop under

    HKCU\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager
        \\ConsentStore\\microphone

Desktop apps live under the NonPackaged subkey with '#' substituted for '\\' in
their path. Store/packaged apps sit directly under 'microphone', keyed by
package family name. LastUsedTimeStop == 0 means the app holds the mic *right
now*. Reading this needs no elevation.

Secondary signal (optional): an active WASAPI render session for the same
process, via pycaw. A process that is both capturing mic and playing audio is
almost certainly in a call rather than dictating a note. This mainly matters
for browsers, where mic access alone is ambiguous.

Usage:
    py -m meetingrec list      # dump every mic consent entry
    py -m meetingrec watch     # live state, Ctrl-C to quit

    from meetingrec.detector import MeetingDetector
    d = MeetingDetector(on_start=cb_start, on_stop=cb_stop)
    d.run()                    # blocking; or d.start() for a thread
"""

from __future__ import annotations

import itertools
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable, Optional, Sequence

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Registry locations
# --------------------------------------------------------------------------

CONSENT_STORE = (
    r"SOFTWARE\Microsoft\Windows\CurrentVersion"
    r"\CapabilityAccessManager\ConsentStore\microphone"
)

# Executables / package families that count as "a meeting is happening".
# Run --list during a real call to confirm the exact names on your machines;
# Zoom in particular has shipped under several paths.
DEFAULT_ALLOWLIST = {
    "zoom.exe",
    "teams.exe",
    "ms-teams.exe",
    "msteams.exe",
    "chrome.exe",
    "msedge.exe",
    "firefox.exe",
    "slack.exe",
    "webexmta.exe",
    "webex.exe",
    "ciscocollabhost.exe",
    "discord.exe",
    "gotomeeting.exe",
    # packaged (Store) identities are matched on the family-name prefix
    "msteams_8wekyb3d8bbwe",
    "microsoft.teams_8wekyb3d8bbwe",
    "microsoftteams_8wekyb3d8bbwe",
}

# Apps where mic access alone is weak evidence — a browser could just as easily
# be running a speech-to-text box. Require a playback session too.
AMBIGUOUS = {"chrome.exe", "msedge.exe", "firefox.exe"}

FILETIME_EPOCH = datetime(1601, 1, 1, tzinfo=timezone.utc)


def filetime_to_dt(ft: int) -> Optional[datetime]:
    """Convert a Windows FILETIME (100ns ticks since 1601) to an aware datetime."""
    if not ft:
        return None
    try:
        return FILETIME_EPOCH + timedelta(microseconds=ft // 10)
    except OverflowError:
        return None


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MicUser:
    """An app currently holding the microphone."""

    key: str  # registry subkey name, or "pid:<n>" for a live session
    name: str  # lowercase exe name, or package family name
    path: Optional[str]  # full exe path for non-packaged apps
    packaged: bool
    started_at: Optional[datetime]
    device: Optional[str] = None  # which microphone, when known

    def __str__(self) -> str:
        when = self.started_at.astimezone().strftime("%H:%M:%S") if self.started_at else "?"
        return f"{self.name} (since {when})"


@dataclass
class MeetingSession:
    """One detected meeting, from first trigger to confirmed end."""

    session_id: int
    app: str
    started_at: datetime  # when the app actually opened the mic (registry truth)
    detected_at: Optional[datetime] = None  # when WE noticed; later if we started mid-call
    last_active_at: Optional[datetime] = None  # last poll where the mic was still held
    ended_at: Optional[datetime] = None
    apps_seen: set = field(default_factory=set)

    @property
    def duration(self) -> timedelta:
        """Real meeting length. Excludes the stop-debounce tail."""
        end = self.ended_at or self.last_active_at or datetime.now(timezone.utc)
        return end - self.started_at

    @property
    def detection_lag(self) -> timedelta:
        """How much of the meeting happened before we saw it. Pre-roll covers
        some of this; anything beyond the pre-roll buffer is unrecoverable."""
        if not self.detected_at:
            return timedelta(0)
        return max(timedelta(0), self.detected_at - self.started_at)


# --------------------------------------------------------------------------
# Registry probe
# --------------------------------------------------------------------------


class ConsentStoreProbe:
    """Reads the mic consent store. Returns apps holding the mic right now."""

    def __init__(self, include_hklm: bool = True):
        try:
            import winreg  # noqa: F401
        except ImportError as exc:  # pragma: no cover - non-Windows
            raise RuntimeError("ConsentStoreProbe requires Windows") from exc
        self.include_hklm = include_hklm

    def __call__(self) -> list[MicUser]:
        return self.read_all(active_only=True)

    def read_all(self, active_only: bool = False) -> list[MicUser]:
        import winreg

        hives = [(winreg.HKEY_CURRENT_USER, "HKCU")]
        if self.include_hklm:
            hives.append((winreg.HKEY_LOCAL_MACHINE, "HKLM"))

        out: list[MicUser] = []
        for hive, _label in hives:
            out.extend(self._read_hive(hive, CONSENT_STORE, packaged=True, active_only=active_only))
            out.extend(
                self._read_hive(
                    hive,
                    CONSENT_STORE + r"\NonPackaged",
                    packaged=False,
                    active_only=active_only,
                )
            )
        # de-dupe on (name, started_at); HKLM and HKCU can both list a service
        seen, deduped = set(), []
        for u in out:
            sig = (u.name, u.started_at)
            if sig not in seen:
                seen.add(sig)
                deduped.append(u)
        return deduped

    def _read_hive(self, hive, subkey: str, packaged: bool, active_only: bool) -> list[MicUser]:
        import winreg

        results: list[MicUser] = []
        try:
            root = winreg.OpenKey(hive, subkey, 0, winreg.KEY_READ)
        except FileNotFoundError:
            return results
        except OSError as exc:
            log.debug("cannot open %s: %s", subkey, exc)
            return results

        with root:
            for i in itertools.count():
                try:
                    child = winreg.EnumKey(root, i)
                except OSError:
                    break
                if child.lower() == "nonpackaged":
                    continue
                entry = self._read_entry(root, child, packaged)
                if entry is None:
                    continue
                in_use, user = entry
                if active_only and not in_use:
                    continue
                results.append(user)
        return results

    @staticmethod
    def _read_entry(root, child: str, packaged: bool):
        import winreg

        try:
            with winreg.OpenKey(root, child) as k:
                def qword(name: str) -> int:
                    try:
                        value, _ = winreg.QueryValueEx(k, name)
                        return int(value or 0)
                    except (FileNotFoundError, OSError, TypeError, ValueError):
                        return 0

                start = qword("LastUsedTimeStart")
                stop = qword("LastUsedTimeStop")
        except OSError:
            return None

        # Never used at all: no start timestamp.
        if not start and not stop:
            return None

        in_use = start > 0 and stop == 0

        if packaged:
            name = child.lower()
            path = None
        else:
            path = child.replace("#", "\\")
            name = path.rsplit("\\", 1)[-1].lower()

        return in_use, MicUser(
            key=child,
            name=name,
            path=path,
            packaged=packaged,
            started_at=filetime_to_dt(start),
        )


# --------------------------------------------------------------------------
# Playback corroboration (optional, needs pycaw)
# --------------------------------------------------------------------------


class PlaybackProbe:
    """Names of processes with an ACTIVE WASAPI render session."""

    AUDIO_SESSION_STATE_ACTIVE = 1

    def __init__(self):
        self.available = False
        try:
            from pycaw.pycaw import AudioUtilities  # noqa: F401

            self.available = True
        except Exception as exc:  # pragma: no cover
            log.info("pycaw unavailable (%s) — playback corroboration disabled", exc)

    def __call__(self) -> set[str]:
        if not self.available:
            return set()
        try:
            from pycaw.pycaw import AudioUtilities

            names = set()
            for s in AudioUtilities.GetAllSessions():
                if not s.Process:
                    continue
                try:
                    if s.State != self.AUDIO_SESSION_STATE_ACTIVE:
                        continue
                except Exception:
                    pass  # older pycaw without .State — count it
                names.add(s.Process.name().lower())
            return names
        except Exception as exc:
            log.warning("playback probe failed: %s", exc)
            return set()


# --------------------------------------------------------------------------
# Detector
# --------------------------------------------------------------------------

IDLE = "idle"
IN_MEETING = "in_meeting"
PENDING_STOP = "pending_stop"


class MeetingDetector:
    """
    Polls for meeting activity and fires callbacks on start/stop.

    on_start(session: MeetingSession)
    on_stop(session: MeetingSession)

    mic_source / playback_source are injectable so the state machine can be
    tested without Windows. Leave them None for real use.
    """

    def __init__(
        self,
        poll_interval: float = 2.0,
        stop_debounce: float = 45.0,
        allowlist: Optional[Iterable[str]] = None,
        ambiguous: Optional[Iterable[str]] = None,
        require_playback_for_ambiguous: bool = True,
        on_start: Optional[Callable[[MeetingSession], None]] = None,
        on_stop: Optional[Callable[[MeetingSession], None]] = None,
        mic_source: Optional[Callable[[], Sequence[MicUser]]] = None,
        playback_source: Optional[Callable[[], set]] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.poll_interval = poll_interval
        self.stop_debounce = stop_debounce
        self.allowlist = {a.lower() for a in (allowlist or DEFAULT_ALLOWLIST)}
        self.ambiguous = {a.lower() for a in (ambiguous if ambiguous is not None else AMBIGUOUS)}
        self.require_playback_for_ambiguous = require_playback_for_ambiguous
        self.on_start = on_start
        self.on_stop = on_stop
        self._clock = clock

        self._mic_source = mic_source or ConsentStoreProbe()
        self._playback_source = playback_source or PlaybackProbe()

        self.state = IDLE
        self.session: Optional[MeetingSession] = None
        self._session_counter = 0
        self._pending_since: Optional[float] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # -- classification ----------------------------------------------------

    def _matches_allowlist(self, user: MicUser) -> bool:
        if user.name in self.allowlist:
            return True
        # packaged identities: match on family-name prefix
        return any(user.name.startswith(a) for a in self.allowlist if "_" in a)

    def qualifying_apps(self) -> list[MicUser]:
        """Apps holding the mic that we believe indicate a live meeting."""
        holders = [u for u in self._mic_source() if self._matches_allowlist(u)]
        if not holders:
            return []

        needs_check = [u for u in holders if u.name in self.ambiguous]
        if not needs_check or not self.require_playback_for_ambiguous:
            return holders

        playing = self._playback_source()
        if not playing:
            # No playback data available at all — don't discard on no evidence.
            return holders
        return [u for u in holders if u.name not in self.ambiguous or u.name in playing]

    # -- state machine -----------------------------------------------------

    def poll_once(self) -> str:
        now = self._clock()
        active = self.qualifying_apps()

        if active:
            if self.state == IDLE:
                self._begin(active[0])
            elif self.state == PENDING_STOP:
                log.info("meeting resumed (%s) — cancelling pending stop", active[0].name)
                self.state = IN_MEETING
                self._pending_since = None
            if self.session:
                self.session.apps_seen.update(u.name for u in active)
                self.session.last_active_at = datetime.now(timezone.utc)
        else:
            if self.state == IN_MEETING:
                log.info("mic released — holding %.0fs before ending", self.stop_debounce)
                self.state = PENDING_STOP
                self._pending_since = now
            elif self.state == PENDING_STOP:
                assert self._pending_since is not None
                if now - self._pending_since >= self.stop_debounce:
                    self._end()

        return self.state

    def _begin(self, trigger: MicUser) -> None:
        self._session_counter += 1
        now = datetime.now(timezone.utc)
        self.session = MeetingSession(
            session_id=self._session_counter,
            app=trigger.name,
            started_at=trigger.started_at or now,
            detected_at=now,
            last_active_at=now,
            apps_seen={trigger.name},
        )
        self.state = IN_MEETING
        self._pending_since = None
        log.info("MEETING START  #%d via %s", self.session.session_id, trigger.name)
        if self.on_start:
            try:
                self.on_start(self.session)
            except Exception:
                log.exception("on_start callback raised")

    def _end(self) -> None:
        session = self.session
        self.state = IDLE
        self._pending_since = None
        self.session = None
        if session is None:
            return
        # End at the last poll that saw the mic held, NOT now — otherwise the
        # 45s debounce tail gets counted as meeting time and, once the recorder
        # exists, as 45s of silence appended to every file.
        session.ended_at = session.last_active_at or datetime.now(timezone.utc)
        log.info(
            "MEETING END    #%d after %s", session.session_id, str(session.duration).split(".")[0]
        )
        if self.on_stop:
            try:
                self.on_stop(session)
            except Exception:
                log.exception("on_stop callback raised")

    # -- lifecycle ---------------------------------------------------------

    def run(self) -> None:
        """Blocking poll loop. Ctrl-C or stop() to exit."""
        log.info("watching for meetings (poll %.1fs)", self.poll_interval)
        try:
            while not self._stop_event.is_set():
                try:
                    self.poll_once()
                except Exception:
                    log.exception("poll failed; continuing")
                self._stop_event.wait(self.poll_interval)
        finally:
            if self.state in (IN_MEETING, PENDING_STOP):
                self._end()

    def start(self) -> threading.Thread:
        """Run the loop on a daemon thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.run, name="meeting-detector", daemon=True)
        self._thread.start()
        return self._thread

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout)


# --------------------------------------------------------------------------
# live capture sessions (WASAPI)
# --------------------------------------------------------------------------


class CaptureSessionProbe:
    """
    Asks the audio engine which processes are capturing right now.

    This exists because the consent store cannot be trusted. On one Windows
    build it updates per call and gives a true meeting start time; on another
    it had not been written for two months while Settings happily displayed
    the same day's activity from somewhere else. Identical code, identical
    approach, completely different result — so the registry is at best a
    bonus source, not the foundation.

    A live WASAPI session cannot go stale: it reflects what the audio engine
    is doing this instant. It also reports WHICH microphone an app is using,
    which the registry never did.

    The trade-off is that there is no history. We learn a meeting started when
    we notice it, not when it actually began, so the pre-roll buffer matters
    more here.
    """

    EXCLUDE_PIDS = {os.getpid()}
    AUDIO_SESSION_STATE_ACTIVE = 1
    DEVICE_STATE_ACTIVE = 0x1
    EDATAFLOW_CAPTURE = 1

    def __init__(self, exclude_self: bool = True):
        self.available = False
        self.exclude_self = exclude_self
        self._warned = False
        try:
            import comtypes  # noqa: F401
            import psutil  # noqa: F401
            from pycaw.pycaw import IAudioSessionManager2  # noqa: F401

            self.available = True
        except Exception as exc:
            log.info("capture-session probe unavailable (%s)", exc)

    def __call__(self) -> list[MicUser]:
        if not self.available:
            return []
        try:
            return self._read()
        except Exception as exc:
            if not self._warned:
                log.warning("capture-session probe failed (%r) — falling back", exc)
                self._warned = True
            return []

    def _read(self) -> list[MicUser]:
        import comtypes
        import psutil
        from comtypes import CLSCTX_ALL, CLSCTX_INPROC_SERVER
        from pycaw.constants import CLSID_MMDeviceEnumerator
        from pycaw.pycaw import (
            AudioUtilities,
            IAudioSessionControl2,
            IAudioSessionManager2,
            IMMDeviceEnumerator,
        )

        # Each poll may land on a thread COM has not seen before.
        try:
            comtypes.CoInitialize()
        except Exception:
            pass

        enumerator = comtypes.CoCreateInstance(
            CLSID_MMDeviceEnumerator, IMMDeviceEnumerator, CLSCTX_INPROC_SERVER)
        collection = enumerator.EnumAudioEndpoints(
            self.EDATAFLOW_CAPTURE, self.DEVICE_STATE_ACTIVE)

        now = datetime.now(timezone.utc)
        out: list[MicUser] = []
        seen: set[tuple[str, str]] = set()

        for i in range(collection.GetCount()):
            dev = collection.Item(i)
            try:
                device_name = AudioUtilities.CreateDevice(dev).FriendlyName
            except Exception:
                device_name = "?"
            try:
                mgr = dev.Activate(IAudioSessionManager2._iid_, CLSCTX_ALL, None)
                mgr = mgr.QueryInterface(IAudioSessionManager2)
                sessions = mgr.GetSessionEnumerator()
            except Exception:
                continue

            for j in range(sessions.GetCount()):
                try:
                    ctl = sessions.GetSession(j)
                    if ctl.GetState() != self.AUDIO_SESSION_STATE_ACTIVE:
                        continue
                    pid = ctl.QueryInterface(IAudioSessionControl2).GetProcessId()
                except Exception:
                    continue
                if not pid or (self.exclude_self and pid in self.EXCLUDE_PIDS):
                    continue
                try:
                    proc = psutil.Process(pid)
                    name, path = proc.name(), proc.exe()
                except Exception:
                    continue

                # The same app often holds several endpoints at once.
                sig = (name.lower(), device_name)
                if sig in seen:
                    continue
                seen.add(sig)
                out.append(MicUser(key=f"pid:{pid}", name=name.lower(), path=path,
                                   packaged=False, started_at=now, device=device_name))
        return out


def build_mic_source(mode: str = "auto"):
    """
    Choose how to detect microphone use.

    "sessions" is the reliable one and the default. "consent" reads the
    registry, which gives a true start time where it works. "auto" prefers
    sessions and falls back.
    """
    mode = (mode or "auto").lower()
    if mode == "consent":
        return ConsentStoreProbe()
    if mode == "sessions":
        return CaptureSessionProbe()

    probe = CaptureSessionProbe()
    if probe.available and probe():
        return probe
    if probe.available:
        # No active sessions right now is normal — nobody is in a meeting.
        return probe
    log.info("falling back to the consent store for detection")
    return ConsentStoreProbe()
