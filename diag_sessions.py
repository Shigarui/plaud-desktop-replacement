"""
Ask the audio engine which processes have a microphone stream open.

The consent store in the registry is stale on some Windows builds — Settings
shows today's activity while the registry has nothing for months. This asks
WASAPI directly instead, which cannot go stale because it reflects live state.

Run it during a Teams or Zoom call.
"""

import sys
import time

try:
    import comtypes
    import psutil
    from comtypes import CLSCTX_ALL, CLSCTX_INPROC_SERVER
    from pycaw.pycaw import (
        AudioUtilities,
        IAudioSessionControl2,
        IAudioSessionManager2,
        IMMDeviceEnumerator,
    )
    from pycaw.constants import CLSID_MMDeviceEnumerator
except ImportError as exc:
    print(f"missing dependency: {exc}")
    print('fix:  .\\.venv\\Scripts\\python.exe -m pip install -e ".[detect]"')
    sys.exit(1)

EDATAFLOW = {0: "render", 1: "capture"}
STATE = {0: "inactive", 1: "ACTIVE", 2: "expired"}
DEVICE_STATE_ACTIVE = 0x1


def capture_endpoints():
    """Every enabled capture endpoint, not just the default one — a meeting
    app may well be using a different microphone than Windows defaults to."""
    enumerator = comtypes.CoCreateInstance(
        CLSID_MMDeviceEnumerator, IMMDeviceEnumerator, CLSCTX_INPROC_SERVER)
    collection = enumerator.EnumAudioEndpoints(1, DEVICE_STATE_ACTIVE)  # 1 = eCapture
    out = []
    for i in range(collection.GetCount()):
        dev = collection.Item(i)
        try:
            name = AudioUtilities.CreateDevice(dev).FriendlyName
        except Exception:
            name = "<unnamed>"
        out.append((name, dev))
    return out


def sessions_on(dev):
    mgr = dev.Activate(IAudioSessionManager2._iid_, CLSCTX_ALL, None)
    mgr = mgr.QueryInterface(IAudioSessionManager2)
    enum = mgr.GetSessionEnumerator()
    rows = []
    for i in range(enum.GetCount()):
        ctl = enum.GetSession(i)
        try:
            ctl2 = ctl.QueryInterface(IAudioSessionControl2)
            pid = ctl2.GetProcessId()
        except Exception:
            pid = 0
        try:
            state = ctl.GetState()
        except Exception:
            state = -1
        name = "<system>"
        if pid:
            try:
                name = psutil.Process(pid).name()
            except Exception:
                name = f"<pid {pid}>"
        rows.append((name, pid, state))
    return rows


def snapshot(label=""):
    print(f"\n{'=' * 68}")
    print(f"{label}{time.strftime('%H:%M:%S')}")
    print("=" * 68)
    any_active = False
    for name, dev in capture_endpoints():
        rows = sessions_on(dev)
        if not rows:
            continue
        print(f"\n  {name}")
        for proc, pid, state in rows:
            mark = "  <== CAPTURING" if state == 1 else ""
            if state == 1:
                any_active = True
            print(f"      {proc:<30} pid={pid:<7} {STATE.get(state, state)}{mark}")
    if not any_active:
        print("\n  (nothing is capturing right now)")


if __name__ == "__main__":
    comtypes.CoInitialize()
    try:
        if "--watch" in sys.argv:
            print("Polling every 3s. Join/leave a meeting to see it change. Ctrl-C to stop.")
            while True:
                snapshot()
                time.sleep(3)
        else:
            snapshot("one-shot at ")
            print("\nRun with --watch to poll continuously.")
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        comtypes.CoUninitialize()
