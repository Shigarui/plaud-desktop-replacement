"""
rearchive.py — move recordings stranded in staging into the archive.

Files end up in staging when archiving failed at the time: Drive not mounted
yet, the laptop offline, the client restarting. The recording itself is fine —
it just never made the last hop.

    python rearchive.py --dry-run
    python rearchive.py

Groups files back into meetings by their timestamp prefix, rebuilds the
destination path from the recording's own name, and moves them. Anything it
cannot place is left alone and reported rather than guessed at.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from meetingrec.config import Config  # noqa: E402
from meetingrec.storage import build_backend  # noqa: E402

STAMP = re.compile(r"^(\d{8})-(\d{6})-(.+?)(?:-(?:mic|system|mix|mic-raw)\.wav"
                   r"|-shot\d+\.jpg|-shots\.json|\.wav\.json|\.json)?$")


def meeting_key(name: str) -> tuple[str, str] | None:
    """The timestamp+app a file belongs to, from its own name."""
    match = re.match(r"^(\d{8}-\d{6})-(.+?)(?:-(?:mic|system|mix|mic-raw)\.wav"
                     r"|-shot\d+\.jpg|-shots\.json|-mic\.wav\.json)$", name)
    if match:
        return match.group(1), match.group(2)
    match = re.match(r"^(\d{8}-\d{6})-(.+?)\.[a-z0-9]+$", name)
    return (match.group(1), match.group(2)) if match else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Re-archive stranded recordings.")
    ap.add_argument("--staging", default=None, help="override the staging folder")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would move, change nothing")
    args = ap.parse_args(argv)

    cfg = Config.load()
    staging = Path(args.staging) if args.staging else Path(cfg.staging_dir)
    if not staging.is_dir():
        print(f"no staging folder at {staging}")
        return 0

    files = [p for p in staging.iterdir() if p.is_file()]
    if not files:
        print("staging is empty — nothing stranded")
        return 0

    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    orphans: list[Path] = []
    for path in sorted(files):
        key = meeting_key(path.name)
        (groups[key].append(path) if key else orphans.append(path))

    print(f"{len(files)} file(s) in staging, {len(groups)} meeting(s)")

    backend = build_backend(cfg)
    if hasattr(backend, "available") and not backend.available():
        print("\nThe archive is not reachable right now — nothing moved.")
        print("Start Google Drive and try again; the files are safe meanwhile.")
        return 1

    moved = failed = 0
    for (stamp, app), paths in sorted(groups.items()):
        when = datetime.strptime(stamp, "%Y%m%d-%H%M%S")
        subdir = (f"{cfg.machine}/{when:%Y}/{when:%m}/{stamp}-{app}"
                  if cfg.folder_per_meeting
                  else f"{cfg.machine}/{when:%Y}/{when:%m}")
        print(f"\n{stamp}-{app}  ({len(paths)} file(s))")
        if args.dry_run:
            print(f"  would go to {subdir}")
            continue
        for path in paths:
            result = backend.archive(path, subdir, {})
            if result.ok:
                path.unlink(missing_ok=True)
                moved += 1
            else:
                print(f"  FAILED {path.name}: {result.error}")
                failed += 1

    if orphans:
        print(f"\n{len(orphans)} file(s) could not be placed and were left alone:")
        for path in orphans[:10]:
            print(f"  {path.name}")

    if args.dry_run:
        print("\ndry run — nothing moved")
    else:
        print(f"\n{moved} file(s) archived" + (f", {failed} failed" if failed else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
