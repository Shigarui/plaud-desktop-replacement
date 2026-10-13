"""State-machine tests. Run anywhere — no Windows required."""

from datetime import datetime, timezone

import pytest

from meetingrec.detector import IDLE, IN_MEETING, PENDING_STOP, MeetingDetector, MicUser


def mic(name: str) -> MicUser:
    return MicUser(key=name, name=name, path=None, packaged=False,
                   started_at=datetime.now(timezone.utc))


class Fake:
    def __init__(self):
        self.mic: list[MicUser] = []
        self.play: set[str] = set()
        self.t = 0.0

    def mic_src(self):
        return list(self.mic)

    def play_src(self):
        return set(self.play)

    def clock(self):
        return self.t


@pytest.fixture
def rig():
    f = Fake()
    events: list[tuple] = []
    d = MeetingDetector(
        mic_source=f.mic_src, playback_source=f.play_src, clock=f.clock, stop_debounce=45.0,
        on_start=lambda s: events.append(("start", s.session_id, s.app)),
        on_stop=lambda s: events.append(("stop", s.session_id)),
    )
    return f, d, events


def test_ignores_unknown_apps(rig):
    f, d, events = rig
    f.mic = [mic("voicerecorder.exe")]
    assert d.poll_once() == IDLE
    assert events == []


def test_start_and_clean_stop(rig):
    f, d, events = rig
    f.mic = [mic("zoom.exe")]
    assert d.poll_once() == IN_MEETING
    f.mic = []
    assert d.poll_once() == PENDING_STOP
    f.t += 46
    assert d.poll_once() == IDLE
    assert [e[0] for e in events] == ["start", "stop"]


def test_blip_does_not_split_session(rig):
    """A mic drop inside the debounce window must not create a second file."""
    f, d, events = rig
    f.mic = [mic("zoom.exe")]
    d.poll_once()
    f.mic = []
    d.poll_once()
    f.t += 10
    f.mic = [mic("zoom.exe")]
    assert d.poll_once() == IN_MEETING
    f.mic = []
    d.poll_once()
    f.t += 46
    d.poll_once()
    assert [e[0] for e in events] == ["start", "stop"]
    assert events[0][1] == events[1][1] == 1


def test_browser_suppressed_without_playback(rig):
    f, d, events = rig
    f.mic = [mic("chrome.exe")]
    f.play = {"spotify.exe"}
    assert d.poll_once() == IDLE


def test_browser_allowed_with_playback(rig):
    f, d, events = rig
    f.mic = [mic("chrome.exe")]
    f.play = {"chrome.exe"}
    assert d.poll_once() == IN_MEETING


def test_no_playback_data_does_not_discard(rig):
    """Fail toward over-recording: absent evidence is not evidence of absence."""
    f, d, events = rig
    f.mic = [mic("chrome.exe")]
    f.play = set()
    assert d.poll_once() == IN_MEETING


def test_tracks_multiple_apps(rig):
    f, d, events = rig
    f.mic = [mic("zoom.exe")]
    d.poll_once()
    f.mic = [mic("zoom.exe"), mic("slack.exe")]
    d.poll_once()
    assert d.session.apps_seen == {"zoom.exe", "slack.exe"}


def test_duration_excludes_debounce_tail(rig):
    """The 45s stop debounce must not be counted as meeting time."""
    f, d, events = rig
    f.mic = [mic("zoom.exe")]
    d.poll_once()
    session = d.session
    f.mic = []
    d.poll_once()
    f.t += 46
    d.poll_once()
    assert session.ended_at == session.last_active_at
    assert session.duration.total_seconds() < 5  # not ~45


# --- detection source selection --------------------------------------------


def test_micuser_carries_the_device_it_was_seen_on():
    """Live sessions report which microphone an app is using; the registry
    never did. The field must be optional so consent-store entries still fit."""
    from meetingrec.detector import MicUser

    live = MicUser(key="pid:123", name="ms-teams.exe", path=None, packaged=False,
                   started_at=datetime.now(timezone.utc),
                   device="Headset (soundcore Liberty 5 Pro Max)")
    assert live.device.startswith("Headset")

    from_registry = MicUser(key="k", name="zoom.exe", path=None, packaged=False,
                            started_at=datetime.now(timezone.utc))
    assert from_registry.device is None


def test_build_mic_source_honours_an_explicit_choice():
    from meetingrec.detector import (
        CaptureSessionProbe,
        ConsentStoreProbe,
        build_mic_source,
    )

    try:
        assert isinstance(build_mic_source("consent"), ConsentStoreProbe)
    except RuntimeError:
        pass  # ConsentStoreProbe refuses to construct off Windows
    assert isinstance(build_mic_source("sessions"), CaptureSessionProbe)


def test_session_probe_degrades_quietly_without_its_dependencies():
    """Off Windows, or without pycaw, it must report nothing rather than raise."""
    from meetingrec.detector import CaptureSessionProbe

    probe = CaptureSessionProbe()
    assert probe() == [] or probe.available


def test_session_probe_excludes_our_own_process():
    """meetingrec opens the mic itself once recording starts; seeing that as
    a meeting would make it self-triggering."""
    import os

    from meetingrec.detector import CaptureSessionProbe

    assert os.getpid() in CaptureSessionProbe.EXCLUDE_PIDS


def test_detector_accepts_session_shaped_users():
    """The state machine must not care which probe produced its input."""
    from meetingrec.detector import IN_MEETING, MeetingDetector, MicUser

    user = MicUser(key="pid:99", name="ms-teams.exe", path="C:/x/ms-teams.exe",
                   packaged=False, started_at=datetime.now(timezone.utc),
                   device="Headset")
    d = MeetingDetector(mic_source=lambda: [user], playback_source=lambda: set())
    assert d.poll_once() == IN_MEETING
