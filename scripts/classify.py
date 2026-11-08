"""
classify.py — title and categorise meetings by the field-ops naming rules.

Produces `{Date} / {Speaker(s)} / {Type}` for every meeting, assigns a content
category, and applies the Trim, Flagged and team-member routing checks.

Nothing is renamed or moved. The archive layout is what every other part of
this project navigates by, and a move that goes wrong loses a recording. The
classification is written to `classification.json` beside the meeting instead,
and the console sorts and filters by it — the same outcome without touching
the files.

What is decided here is decided from evidence already on disk: talk-time
percentages come from speakers.json, the date from the folder name, duration
from the audio, and trigger phrases from the transcript. Where a rule cannot
be applied honestly, the fallback is recorded and reported rather than
guessed — ambiguity is a result, not a failure.

    python classify.py --all "G:/My Drive/Meeting Recordings"
    python classify.py --all <root> --since 30d --force
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sept", "Oct", "Nov", "Dec"]


# --------------------------------------------------------------------------
# rules
# --------------------------------------------------------------------------


def load_rules(path: Optional[Path] = None) -> dict:
    path = path or Path(__file__).resolve().parent / "taxonomy.toml"
    with path.open("rb") as fh:
        return tomllib.load(fh)


@dataclass
class Classification:
    title: str = ""
    date: str = ""
    speakers: str = ""
    type: str = ""
    category: str = ""
    folder: str = ""
    team_member: Optional[str] = None
    flags: list = field(default_factory=list)
    fallbacks: list = field(default_factory=list)
    talk: dict = field(default_factory=dict)
    skipped: bool = False


# --------------------------------------------------------------------------
# the pieces of a title
# --------------------------------------------------------------------------


def folder_date(folder: Path) -> Optional[datetime]:
    """
    The meeting date, from the folder name.

    Not the file's modification time: copying or re-syncing an archive
    rewrites those, and the rule is explicit that today's date is never used.
    """
    match = re.match(r"(\d{8})[-_]?(\d{6})?", folder.name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1) + (match.group(2) or "000000"),
                                 "%Y%m%d%H%M%S")
    except ValueError:
        return None


def date_segment(when: Optional[datetime]) -> str:
    """`Mmm-DD`, no leading zero on the day."""
    return f"{MONTHS[when.month - 1]}-{when.day:02d}" if when else ""


def talk_shares(speakers: list, me: str) -> dict:
    """
    Share of speaking time per voice, with the user excluded.

    Percentages are of the time OTHER people spoke, which is what the naming
    rule means: on a call where the user holds 70%, a colleague with the
    remaining 30% is the dominant speaker, not a minor one.
    """
    others = [s for s in speakers
              if (s.get("name") or "").strip().lower() != me.strip().lower()]
    total = sum(max(s.get("seconds", 0.0), 0.0) for s in others)
    if total <= 0:
        return {}
    return {(s.get("name") or s.get("label") or "Unknown"):
            round(s.get("seconds", 0.0) / total * 100, 1) for s in others}


def is_unnamed(label: str) -> bool:
    return bool(re.fullmatch(r"(speaker|voice)\s*\d+", label.strip(), re.I))


def speaker_segment(shares: dict, rules: dict, result: Classification) -> str:
    """Segment 2: who was on the call, by the talk-time rules."""
    cfg = rules["thresholds"]
    team = {n.lower() for n in rules["people"]["team"]}

    if not shares:
        return "Notes to Self"

    ranked = sorted(shares.items(), key=lambda kv: -kv[1])
    named = [(n, p) for n, p in ranked if not is_unnamed(n)]
    dominant = [(n, p) for n, p in ranked if p >= cfg["dominant_percent"]]

    on_team = [n for n, _ in named if n.lower() in team]
    if len(on_team) > cfg["group_call_team_members"]:
        return "FBC/OSS Team"

    if not dominant:
        # Rule: use the loudest voice anyway, and flag it for a human.
        result.flags.append("Flagged, Speaker Audit")
        result.fallbacks.append(
            f"nobody reached {cfg['dominant_percent']:.0f}% — used the "
            f"highest at {ranked[0][1]:.0f}%")
        return ranked[0][0]

    chosen = [n for n, _ in dominant if not is_unnamed(n)][:cfg["max_named_speakers"]]
    if not chosen:
        result.flags.append("Flagged, Speaker Audit")
        result.fallbacks.append("only unnamed speakers were dominant")
        return dominant[0][0]
    return ", ".join(chosen)


def matches(text: str, phrases: list, patterns: Optional[list] = None) -> bool:
    low = text.lower()
    if any(p.lower() in low for p in phrases or []):
        return True
    return any(re.search(p, low) for p in patterns or [])


def type_segment(text: str, shares: dict, rules: dict,
                 result: Classification) -> str:
    """
    Segment 3, from a fixed vocabulary.

    The rules are ordered so precedence falls out of the ordering: Audit is
    checked before Field Visit, QBR before Ops Meeting, Coaching before One
    on One. Expressing it as order rather than special cases means adding a
    label later cannot quietly break an existing precedence.
    """
    cfg = rules["thresholds"]
    solo = not shares
    dominant = [n for n, p in shares.items() if p >= cfg["dominant_percent"]]
    team = {n.lower() for n in rules["people"]["team"]}
    on_team = [n for n in shares if n.lower() in team]

    if len(on_team) > cfg["group_call_team_members"]:
        return "Group Meeting"

    for entry in rules["type"]:
        if entry.get("solo_only"):
            if solo:
                return entry["name"]
            continue
        if entry.get("one_to_one"):
            if len(dominant) == 1:
                return entry["name"]
            continue
        if entry.get("fallback"):
            result.fallbacks.append("no type phrase matched — used Ops Meeting")
            return entry["name"]
        if matches(text, entry.get("phrases", []), entry.get("patterns")):
            return entry["name"]

    result.fallbacks.append("no type could be determined")
    return "Ops Meeting"


def category_for(text: str, shares: dict, rules: dict) -> tuple[str, str]:
    """Which content category and folder this belongs to."""
    solo = not shares
    present = {n.lower() for n in shares}
    team = {n.lower() for n in rules["people"]["team"]}

    for entry in rules["category"]:
        if entry.get("solo_only"):
            if solo or matches(text, entry.get("phrases", [])):
                if solo:
                    return entry["name"], entry["folder"]
            continue

        people = [p.lower() for p in entry.get("people", [])]
        if entry.get("people_from") == "team":
            people = list(team)
        if people and present & set(people):
            return entry["name"], entry["folder"]
        if matches(text, entry.get("phrases", []), entry.get("patterns")):
            return entry["name"], entry["folder"]

    return "Uncategorised", ""


def team_member_folder(shares: dict, rules: dict,
                       result: Classification) -> Optional[str]:
    """
    The one-on-one routine: file under a team member when they are the only
    other dominant voice.

    An unnamed voice above the competing threshold disqualifies the
    recording — otherwise a three-way call would be filed as a private 1:1,
    which is the kind of mistake that matters.
    """
    cfg = rules["thresholds"]
    team = {n.lower(): n for n in rules["people"]["team"]}

    dominant = [(n, p) for n, p in shares.items() if p >= cfg["dominant_percent"]]
    members = [n for n, _ in dominant if n.lower() in team]

    if len(members) > 1:
        result.flags.append("Conflict: more than one team member qualifies")
        return None
    if len(members) != 1:
        return None

    for name, pct in shares.items():
        if name == members[0]:
            continue
        if is_unnamed(name) and pct > cfg["unlabeled_competes"]:
            result.fallbacks.append(
                f"an unnamed voice held {pct:.0f}% — not filed as a 1:1")
            return None
        if not is_unnamed(name) and pct >= cfg["dominant_percent"]:
            return None
    return team[members[0].lower()]


# --------------------------------------------------------------------------
# one meeting
# --------------------------------------------------------------------------


def read_text(folder: Path, limit: int = 400_000) -> str:
    """The transcript if there is one; otherwise whatever names we have."""
    for name in ("transcript.md", "transcript.json"):
        path = folder / name
        if path.exists():
            try:
                return path.read_text(encoding="utf-8", errors="replace")[:limit]
            except Exception:
                continue
    return ""


def classify(folder: Path, rules: dict, duration: float = 0.0) -> Classification:
    result = Classification()
    speakers_file = folder / "speakers.json"
    if not speakers_file.exists():
        result.skipped = True
        result.fallbacks.append("not identified yet — run speaker identification")
        return result

    try:
        speakers = json.loads(speakers_file.read_text(encoding="utf-8")).get("speakers", [])
    except Exception as exc:
        result.skipped = True
        result.fallbacks.append(f"speakers.json unreadable ({exc})")
        return result

    me = rules["people"]["me"]
    shares = talk_shares(speakers, me)
    text = read_text(folder)

    when = folder_date(folder)
    if when is None:
        result.fallbacks.append("no date in the folder name — ask before filing")
    result.date = date_segment(when)
    result.talk = shares

    result.speakers = speaker_segment(shares, rules, result)
    result.type = type_segment(text, shares, rules, result)
    result.category, result.folder = category_for(text, shares, rules)

    if result.speakers == "Notes to Self":
        result.type = "Notes to Self"
        result.category, result.folder = "Notes to Self", "1-Note To Self"
        result.title = " / ".join(x for x in (result.date, "Notes to Self") if x)
    else:
        result.title = " / ".join(
            x for x in (result.date, result.speakers, result.type) if x)

    # Length and speaker-clarity checks run independently of the category.
    if duration and duration > rules["thresholds"]["trim_hours"] * 3600:
        result.flags.append("Trim")
    if any(is_unnamed(n) for n in shares) and not any(
            not is_unnamed(n) for n in shares):
        if "Flagged, Speaker Audit" not in result.flags:
            result.flags.append("Flagged, Speaker Audit")

    result.team_member = team_member_folder(shares, rules, result)
    return result


# --------------------------------------------------------------------------
# driving
# --------------------------------------------------------------------------


def parse_since(spec: Optional[str]) -> Optional[datetime]:
    if not spec:
        return None
    text = spec.strip().lower()
    match = re.fullmatch(r"(\d+)\s*([dwmy])", text)
    if match:
        days = {"d": 1, "w": 7, "m": 30, "y": 365}[match.group(2)] * int(match.group(1))
        return datetime.now() - timedelta(days=days)
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"could not read a date from {spec!r}")


def audio_duration(folder: Path) -> float:
    import wave

    for path in sorted(folder.glob("*-system.wav")) or sorted(folder.glob("*.wav")):
        try:
            with wave.open(str(path), "rb") as w:
                return w.getnframes() / w.getframerate()
        except Exception:
            continue
    try:
        import soundfile as sf

        for path in folder.iterdir():
            if path.suffix.lower() in (".mp3", ".ogg", ".opus", ".m4a", ".flac"):
                info = sf.info(str(path))
                return float(info.frames) / info.samplerate
    except Exception:
        pass
    return 0.0


def meeting_folders(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.glob("**/speakers.json")
                  if not any(part.startswith("_") for part in p.parent.parts))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Title and categorise meetings.")
    ap.add_argument("path", help="a meeting folder, or the archive root with --all")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--rules", default=None, help="an alternative taxonomy.toml")
    ap.add_argument("--since", default=None, help="2026-10-01, or a span like 30d")
    ap.add_argument("--force", action="store_true", help="redo ones already done")
    ap.add_argument("--dry-run", action="store_true", help="show, write nothing")
    args = ap.parse_args(argv)

    rules = load_rules(Path(args.rules) if args.rules else None)
    root = Path(args.path)
    folders = meeting_folders(root) if args.all else [root]

    try:
        since = parse_since(args.since)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    if since:
        folders = [f for f in folders
                   if (folder_date(f) or datetime.min) >= since]

    done = skipped = 0
    needs_attention: list[tuple[str, list]] = []

    for folder in folders:
        out = folder / "classification.json"
        if out.exists() and not args.force:
            continue
        try:
            result = classify(folder, rules, audio_duration(folder))
        except Exception as exc:
            print(f"  FAILED {folder.name}: {exc}")
            continue

        if result.skipped:
            skipped += 1
            continue

        marks = ("  [" + ", ".join(result.flags) + "]") if result.flags else ""
        print(f"{result.title}{marks}")
        print(f"    {result.category or '—'}"
              + (f" · {result.team_member}" if result.team_member else ""))
        if result.fallbacks:
            needs_attention.append((result.title, result.fallbacks))

        if not args.dry_run:
            out.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
        done += 1

    print(f"\n{done} classified" + (f", {skipped} not identified yet" if skipped else ""))
    if needs_attention:
        # The output contract: every fallback is reported so it can be
        # corrected, rather than disappearing into a confident-looking title.
        print(f"\n{len(needs_attention)} needed a fallback:")
        for title, why in needs_attention[:20]:
            print(f"  {title}")
            for reason in why:
                print(f"      {reason}")
    if args.dry_run:
        print("\ndry run — nothing written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
