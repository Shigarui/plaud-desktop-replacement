"""Dump every capability-access location Windows might be using."""
import winreg, itertools
from datetime import datetime, timedelta, timezone

EPOCH = datetime(1601, 1, 1, tzinfo=timezone.utc)
BASE = r"SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore"

def ft(v):
    if not v:
        return "0"
    try:
        return (EPOCH + timedelta(microseconds=int(v) // 10)).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return f"?{v}"

def walk(hive, hivename, path, depth=0, maxdepth=4):
    try:
        key = winreg.OpenKey(hive, path, 0, winreg.KEY_READ)
    except OSError:
        return
    with key:
        vals = {}
        for i in itertools.count():
            try:
                n, v, _t = winreg.EnumValue(key, i)
                vals[n] = v
            except OSError:
                break
        if vals:
            interesting = {k: v for k, v in vals.items() if "LastUsed" in k or k == "Value"}
            if interesting:
                leaf = path.split("\\")[-1]
                start = ft(interesting.get("LastUsedTimeStart"))
                stop = interesting.get("LastUsedTimeStop")
                state = "OPEN NOW" if (interesting.get("LastUsedTimeStart") and not stop) else ""
                print(f"{'  '*depth}{leaf[:70]}")
                print(f"{'  '*depth}    start={start}  stop={ft(stop)}  {state}")
        if depth < maxdepth:
            for i in itertools.count():
                try:
                    sub = winreg.EnumKey(key, i)
                except OSError:
                    break
                walk(hive, hivename, path + "\\" + sub, depth + 1, maxdepth)

for hive, name in ((winreg.HKEY_CURRENT_USER, "HKCU"), (winreg.HKEY_LOCAL_MACHINE, "HKLM")):
    for cap in ("microphone", "webcam"):
        print(f"\n{'='*70}\n{name}\\...\\ConsentStore\\{cap}\n{'='*70}")
        walk(hive, name, f"{BASE}\\{cap}")

print(f"\n{'='*70}\nTop-level subkeys present under ConsentStore\n{'='*70}")
for hive, name in ((winreg.HKEY_CURRENT_USER, "HKCU"), (winreg.HKEY_LOCAL_MACHINE, "HKLM")):
    try:
        with winreg.OpenKey(hive, BASE) as k:
            subs = []
            for i in itertools.count():
                try:
                    subs.append(winreg.EnumKey(k, i))
                except OSError:
                    break
            print(f"{name}: {subs}")
    except OSError as e:
        print(f"{name}: {e}")
