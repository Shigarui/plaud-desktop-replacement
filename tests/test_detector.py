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
