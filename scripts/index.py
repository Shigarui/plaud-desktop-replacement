"""
index.py — one searchable reference covering the whole archive.

Writes `_shared/INDEX.md` and `_shared/index.json` so an agent can answer
"when did we last discuss X", "what has Dan and I covered this quarter",
"which recordings mention Store 64" without opening a single recording.

Two formats because they serve different readers. The Markdown is for a
person or a model reading top to bottom: grouped by month, one line per
meeting, with the title, people and topics visible. The JSON is for anything
doing lookups, and carries the fields the Markdown summarises.

Only what is already on disk goes in. Nothing is inferred and nothing is
re-derived here: if a meeting has not been classified, it is listed as
unclassified rather than given a guessed title.

    python index.py --all "G:/My Drive/Meeting Recordings"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from classify import folder_date, load_rules  # noqa: E402

STOPWORDS = {
    "that", "this", "with", "have", "from", "they", "what", "were", "been",
    "them", "then", "than", "when", "will", "your", "about", "would", "there",
    "their", "which", "could", "should", "going", "really", "think", "just",
    "like", "know", "yeah", "okay", "right", "thing", "things", "kind", "sort",
    "actually", "basically", "because", "little", "maybe", "sure", "guess",
}


def keywords(text: str, limit: int = 12) -> list[str]:
    """
    The words this meeting is actually about.

    Deliberately crude — frequency minus filler. It is not a summary and
    should not read like one; it is a set of handles to search on, and a
    wrong handle costs a false hit rather than a false belief.
    """
    words = re.findall(r"[a-z][a-z'-]{3,}", text.lower())
    counts = Counter(w for w in words if w not in STOPWORDS)
    return [w for w, _ in counts.most_common(limit)]


def store_numbers(text: str) -> list[str]:
    found = sorted({m.group(1) for m in
                    re.finditer(r"\bstore\s+#?(\d{1,4})\b", text, re.I)})
    return found[:10]


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def summarise_transcript(folder: Path) -> dict:
    """First lines of the summary section, if the transcript has one."""
    path = folder / "transcript.md"
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {}

    summary = ""
    match = re.search(r"^##\s*Summary\s*$(.+?)(?=^##\s|\Z)", text,
                      re.M | re.S)
    if match:
        summary = " ".join(match.group(1).split())[:400]
    return {"summary": summary, "keywords": keywords(text),
            "stores": store_numbers(text), "words": len(text.split())}


def gather(root: Path) -> list[dict]:
    rows = []
    for speakers_file in sorted(root.glob("**/speakers.json")):
        folder = speakers_file.parent
        if any(part.startswith("_") for part in folder.parts):
            continue

        speakers = read_json(speakers_file).get("speakers", [])
        classification = read_json(folder / "classification.json")
        content = summarise_transcript(folder)
        when = folder_date(folder)

        rows.append({
            "folder": str(folder),
            "id": folder.name,
            "when": when.isoformat() if when else None,
            "title": classification.get("title") or "",
            "type": classification.get("type") or "",
            "category": classification.get("category") or "",
            "team_member": classification.get("team_member"),
            "flags": classification.get("flags", []),
            "classified": bool(classification),
            "speakers": [s["name"] for s in speakers if s.get("name")],
            "unnamed": sum(1 for s in speakers if not s.get("name")),
            "talk": classification.get("talk", {}),
            "transcript": (folder / "transcript.md").exists(),
            **content,
        })
    rows.sort(key=lambda r: r["when"] or "", reverse=True)
    return rows


def write_markdown(rows: list[dict], path: Path, root: Path) -> None:
    by_month: dict[str, list] = defaultdict(list)
    for row in rows:
        key = row["when"][:7] if row["when"] else "undated"
        by_month[key].append(row)

    people = Counter(p for r in rows for p in r["speakers"])
    categories = Counter(r["category"] for r in rows if r["category"])

    out = [
        "# Meeting index",
        "",
        f"{len(rows)} recordings · generated {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "Every classified recording in the archive, newest first. Titles follow",
        "`Date / Speaker(s) / Type`. Use this to find a recording; open the",
        "transcript for what was actually said — the transcript is the record,",
        "this is only an index of it.",
        "",
        "## People",
        "",
    ]
    out += [f"- **{name}** — {count} recording{'s' if count != 1 else ''}"
            for name, count in people.most_common()]
    out += ["", "## Categories", ""]
    out += [f"- **{name}** — {count}" for name, count in categories.most_common()]

    unclassified = [r for r in rows if not r["classified"]]
    if unclassified:
        out += ["", f"## Not yet classified ({len(unclassified)})", "",
                "Listed rather than guessed at. Run classification to title these.", ""]
        out += [f"- `{r['id']}`" for r in unclassified[:40]]

    for month in sorted(by_month, reverse=True):
        label = (datetime.strptime(month, "%Y-%m").strftime("%B %Y")
                 if month != "undated" else "Undated")
        out += ["", f"## {label}", ""]
        for row in by_month[month]:
            title = row["title"] or f"({row['id']})"
            out.append(f"### {title}")
            bits = []
            if row["speakers"]:
                bits.append("With " + ", ".join(row["speakers"]))
            if row["category"]:
                bits.append(row["category"])
            if row.get("stores"):
                bits.append("Stores " + ", ".join(row["stores"]))
            if row["flags"]:
                bits.append("⚑ " + ", ".join(row["flags"]))
            if bits:
                out.append(" · ".join(bits))
            if row.get("summary"):
                out += ["", row["summary"]]
            if row.get("keywords"):
                out += ["", "*" + ", ".join(row["keywords"][:10]) + "*"]
            try:
                rel = Path(row["folder"]).relative_to(root)
            except ValueError:
                rel = Path(row["folder"])
            out += ["", f"`{rel}`", ""]

    path.write_text("\n".join(out), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the master meeting index.")
    ap.add_argument("path", help="the archive root")
    ap.add_argument("--all", action="store_true", help="accepted for symmetry")
    ap.add_argument("--out", default=None, help="where to write (default _shared)")
    args = ap.parse_args(argv)

    root = Path(args.path)
    if not root.is_dir():
        print(f"not a folder: {root}", file=sys.stderr)
        return 1

    rows = gather(root)
    if not rows:
        print("no identified meetings found")
        return 0

    out_dir = Path(args.out) if args.out else root / "_shared"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "index.json").write_text(
        json.dumps({"generated": datetime.now().isoformat(),
                    "count": len(rows), "meetings": rows}, indent=2),
        encoding="utf-8")
    write_markdown(rows, out_dir / "INDEX.md", root)

    classified = sum(1 for r in rows if r["classified"])
    transcribed = sum(1 for r in rows if r["transcript"])
    print(f"{len(rows)} meetings indexed "
          f"({classified} classified, {transcribed} transcribed)")
    print(f"  {out_dir / 'INDEX.md'}")
    print(f"  {out_dir / 'index.json'}")
    if classified < len(rows):
        print(f"\n{len(rows) - classified} have no title yet — run classify.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
