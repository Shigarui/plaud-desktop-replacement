from datetime import datetime, timezone
from meeting_detector import MeetingDetector, MicUser, IDLE, IN_MEETING, PENDING_STOP

def mk(name):
    return MicUser(key=name, name=name, path=None, packaged=False,
                   started_at=datetime.now(timezone.utc))

class Fake:
    def __init__(self): self.mic=[]; self.play=set(); self.t=0.0
    def mic_src(self): return list(self.mic)
    def play_src(self): return set(self.play)
    def clock(self): return self.t

f=Fake(); events=[]
d=MeetingDetector(mic_source=f.mic_src, playback_source=f.play_src, clock=f.clock,
                  stop_debounce=45.0,
                  on_start=lambda s: events.append(("start",s.session_id,s.app)),
                  on_stop=lambda s: events.append(("stop",s.session_id)))

assert d.poll_once()==IDLE
# non-allowlisted app opens mic
f.mic=[mk("voicerecorder.exe")]; assert d.poll_once()==IDLE, "should ignore unknown app"
# zoom joins
f.mic=[mk("zoom.exe")]; assert d.poll_once()==IN_MEETING
# blip: mic released
f.mic=[]; assert d.poll_once()==PENDING_STOP
f.t+=10; assert d.poll_once()==PENDING_STOP
# reacquired -> resume, no new session
f.mic=[mk("zoom.exe")]; assert d.poll_once()==IN_MEETING
f.mic=[]; f.t+=1; d.poll_once(); f.t+=50
assert d.poll_once()==IDLE
# browser alone, no playback -> allowed (no evidence either way)
f.mic=[mk("chrome.exe")]; f.play=set()
assert d.poll_once()==IN_MEETING
f.mic=[]; d.poll_once(); f.t+=100; assert d.poll_once()==IDLE
# browser with playback data that excludes it -> suppressed
f.mic=[mk("chrome.exe")]; f.play={"spotify.exe"}
assert d.poll_once()==IDLE, "ambiguous app w/o playback should be suppressed"
# browser WITH playback -> meeting
f.play={"chrome.exe"}; assert d.poll_once()==IN_MEETING
print("events:", events)
print("ALL STATE MACHINE TESTS PASSED")
