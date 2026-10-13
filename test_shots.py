import time, pathlib
from meetingrec.screenshots import ScreenshotCapturer

out = pathlib.Path("recordings/shottest")
c = ScreenshotCapturer(follow_foreground=True, min_gap_seconds=5.0)
c.start()
c.arm(out, "shottest", "zoom.exe")
print("Capturing the foreground window for 30s.")
print("Switch between a few windows, scroll, change slides.")
for i in range(30, 0, -5):
    print(f"  {i}s left - captured {c.stats.captured}")
    time.sleep(5)
stats = c.disarm()
c.stop()
print(f"\ncaptured {stats.captured}, unchanged {stats.skipped_unchanged}, "
      f"excluded {stats.skipped_excluded}, errors {stats.errors}")
for s in stats.shots:
    print(f"  +{s['offset_seconds']:5.1f}s  {s['file']}  {s['title'][:60]}")
