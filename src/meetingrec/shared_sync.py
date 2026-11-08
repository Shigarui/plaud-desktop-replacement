"""
shared_sync.py: desktop half of the PlaudSync phone <-> meetingrec contract (schema 1).

Drop into meetingrec's src/meetingrec/ (or run standalone). Reads/writes
    <Drive>/My Drive/Meeting Recordings/_shared/...
through Google Drive for Desktop. See PlaudSync docs/SHARED_CONTRACT.md.

CLI:
    python shared_sync.py status
    python shared_sync.py apply            # dry run: shows what would change
    python shared_sync.py apply --write    # apply events, update dictionary/metas, queue corrections
    python shared_sync.py export-speakers --profiles "<path to BIGGBY_Speaker_Profiles>"
    python shared_sync.py plaud-queue      # phone recordings still needing a desktop transcript
    python shared_sync.py mark-transcribed <slug>

From meetingrec:
    from meetingrec.shared_sync import SharedSync
    ss = SharedSync()
    ss.apply_events(write=True)                       # start of every transcription run
    prompt = ss.whisper_prompt()                      # pass as Whisper initial_prompt
    segs = ss.apply_label_corrections(slug, segs)     # after diarization/naming
    ss.mark_transcribed(slug); ss.consume_corrections(slug)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import string
import sys
from datetime import datetime
from pathlib import Path

ROOT_NAME = "Meeting Recordings"
SHARED = "_shared"
STATE_DIR = Path(os.environ.get("MEETINGREC_STATE", Path(__file__).resolve().parent / "shared_state"))


def find_drive_root() -> Path:
    """Same auto-detect as plaud_sync_full: first drive letter with 'My Drive'."""
    env = os.environ.get("MEETINGREC_DRIVE_ROOT")
    if env:
        return Path(env)
    for letter in string.ascii_uppercase:
        p = Path(f"{letter}:/My Drive/{ROOT_NAME}")
        if p.exists():
            return p
    home = Path.home() / "My Drive" / ROOT_NAME
    if home.exists():
        return home
    raise SystemExit(f"Can't find 'My Drive/{ROOT_NAME}'. Set MEETINGREC_DRIVE_ROOT.")


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


class SharedSync:
    def __init__(self, drive_root: Path | None = None, state_dir: Path = STATE_DIR):
        self.root = Path(drive_root) if drive_root else find_drive_root()
        self.shared = self.root / SHARED
        self.state = state_dir
        self.ledger_path = self.state / "applied_events.json"
        self.corrections_path = self.state / "pending_corrections.json"

    # ---------------------------------------------------------------- events
    def pending_events(self) -> list[dict]:
        applied = set(_read_json(self.ledger_path, []))
        events = []
        for batch in sorted((self.shared / "events").glob("*/*.json")):
            for e in _read_json(batch, []):
                if isinstance(e, dict) and e.get("id") not in applied:
                    events.append(e)
        return sorted(events, key=lambda e: e.get("ts", ""))

    def apply_events(self, write: bool = False) -> list[str]:
        """Applies every unseen event. Returns a human-readable change log."""
        log: list[str] = []
        events = self.pending_events()
        if not events:
            return ["No new events."]
        dictionary = _read_json(self.shared / "dictionary.json", {"schema": 1, "terms": []})
        terms = {t["term"].lower(): t for t in dictionary.get("terms", [])}
        renames = _read_json(self.shared / "speaker_renames.json", {"schema": 1, "renames": []})
        corrections = _read_json(self.corrections_path, {})

        for e in events:
            t, p, rec, who = e.get("type"), e.get("payload") or {}, e.get("recording"), e.get("device")
            if t == "term_add":
                terms[p["term"].lower()] = {"term": p["term"], "aliases": p.get("aliases", []),
                                            "updatedAt": e["ts"], "updatedBy": who}
                log.append(f"dictionary + {p['term']}")
            elif t == "term_remove":
                terms.pop(p["term"].lower(), None)
                log.append(f"dictionary - {p['term']}")
            elif t == "speaker_rename":
                renames["renames"].append({"from": p["from"], "to": p["to"], "ts": e["ts"], "by": who})
                log.append(f"speaker rename (global) {p['from']} -> {p['to']}")
            elif t in ("label_correction", "flag") and rec:
                corrections.setdefault(rec, []).append(e)
                what = f"{p.get('from')} -> {p.get('to')}" if t == "label_correction" else f"{p.get('kind')}: {p.get('note', '')}"
                log.append(f"{rec}: {t} {what}")
                self._update_meta(rec, lambda m: m.setdefault("flags", []).append(
                    {"type": t, "detail": what, "at": e["ts"], "device": who}), write)
            elif t == "title_set" and rec:
                log.append(f"{rec}: title -> {p.get('title')}")
                self._update_meta(rec, lambda m: m.update(title=p.get("title"), tags=p.get("tags", m.get("tags", [])),
                                                          titledBy="manual"), write)
            else:
                log.append(f"skipped unknown event {t} ({e.get('id')})")

        if write:
            dictionary["terms"] = sorted(terms.values(), key=lambda x: x["term"].lower())
            _write_json(self.shared / "dictionary.json", dictionary)
            _write_json(self.shared / "speaker_renames.json", renames)
            _write_json(self.corrections_path, corrections)
            ledger = set(_read_json(self.ledger_path, [])) | {e["id"] for e in events}
            _write_json(self.ledger_path, sorted(ledger))
        return log

    # ---------------------------------------------------------------- recordings
    def recording_dir(self, slug: str) -> Path | None:
        for meta in self.root.glob(f"*/*/*/{slug}/{slug}.meta.json"):
            return meta.parent
        hits = [p for p in self.root.rglob(slug) if p.is_dir() and not p.is_relative_to(self.shared)]
        return hits[0] if hits else None

    def _update_meta(self, slug: str, fn, write: bool) -> None:
        d = self.recording_dir(slug)
        if not d or not write:
            return
        path = d / f"{slug}.meta.json"
        meta = _read_json(path, {"schema": 1, "slug": slug})
        fn(meta)
        meta["updatedAt"] = datetime.now().astimezone().isoformat(timespec="seconds")
        _write_json(path, meta)

    def plaud_queue(self) -> list[Path]:
        """Phone recordings (PLAUD-* machines) without a desktop transcript yet."""
        out = []
        for d in self.root.glob("PLAUD-*/*/*/*"):
            if not d.is_dir():
                continue
            meta = _read_json(d / f"{d.name}.meta.json", {})
            audio = sorted(d.glob("*.mp3")) + sorted(d.glob("*.ogg"))
            if audio and meta.get("status") != "transcribed":
                out.append(audio[0])
        return sorted(out)

    def mark_transcribed(self, slug: str) -> None:
        self._update_meta(slug, lambda m: m.update(status="transcribed"), write=True)

    # ---------------------------------------------------------------- transcription hooks
    def whisper_prompt(self, limit: int = 800) -> str:
        """Dictionary terms for Whisper's initial_prompt (bias toward correct spellings)."""
        terms = _read_json(self.shared / "dictionary.json", {}).get("terms", [])
        return ", ".join(t["term"] for t in terms)[:limit]

    def alias_map(self) -> dict[str, str]:
        """Mis-hearing -> correct term, for a post-transcription find/replace pass."""
        m = {}
        for t in _read_json(self.shared / "dictionary.json", {}).get("terms", []):
            for a in t.get("aliases", []):
                m[a.lower()] = t["term"]
        return m

    def corrections_for(self, slug: str) -> list[dict]:
        return _read_json(self.corrections_path, {}).get(slug, [])

    def apply_label_corrections(self, slug: str, segments: list[dict], speaker_key: str = "speaker") -> list[dict]:
        """Relabels speakers: per-recording corrections first, then global renames."""
        mapping = {}
        for r in _read_json(self.shared / "speaker_renames.json", {}).get("renames", []):
            mapping[r["from"]] = r["to"]
        for e in self.corrections_for(slug):
            if e.get("type") == "label_correction":
                mapping[e["payload"]["from"]] = e["payload"]["to"]
        for s in segments:
            if s.get(speaker_key) in mapping:
                s[speaker_key] = mapping[s[speaker_key]]
        return segments

    def consume_corrections(self, slug: str) -> None:
        c = _read_json(self.corrections_path, {})
        if c.pop(slug, None) is not None:
            _write_json(self.corrections_path, c)

    # ---------------------------------------------------------------- speakers
    AUDIO_EXT = (".wav", ".mp3", ".m4a", ".ogg", ".opus", ".flac")
    MAX_SAMPLE_BYTES = 8 * 1024 * 1024

    def _sample_for(self, profiles_dir: Path, name: str, row: dict | None) -> Path | None:
        """A speaker's voice sample: an audio path in their MasterIndex row, else the first audio file
        in a profile folder whose name matches theirs."""
        if row:
            for v in row.values():
                if isinstance(v, str) and v.lower().endswith(self.AUDIO_EXT):
                    for cand in (Path(v), profiles_dir / v):
                        if cand.exists():
                            return cand
                    hits = list(profiles_dir.rglob(Path(v).name))
                    if hits:
                        return hits[0]
        key = name.lower().replace(" ", "").replace("_", "")
        for d in profiles_dir.iterdir():
            if d.is_dir() and d.name.lower().replace(" ", "").replace("_", "") == key:
                audio = sorted(f for f in d.rglob("*") if f.suffix.lower() in self.AUDIO_EXT)
                if audio:
                    return audio[0]
        return None

    def export_speakers(self, profiles_dir: Path) -> int:
        """Writes _shared/speakers.json + _shared/speakers/<sample> so phones can list names and play voices."""
        profiles_dir = Path(profiles_dir)
        rows: dict[str, dict] = {}
        index = profiles_dir / "MasterIndex.csv"
        if index.exists():
            with index.open(encoding="utf-8-sig", newline="") as f:
                data = list(csv.DictReader(f))
            if data:
                col = next((c for c in data[0] if c and "name" in c.lower()), list(data[0])[0])
                rows = {r[col].strip(): r for r in data if r.get(col, "").strip()}
        names = set(rows) or {p.name for p in profiles_dir.iterdir() if p.is_dir()}
        for r in _read_json(self.shared / "speaker_renames.json", {}).get("renames", []):
            names.discard(r["from"]); names.add(r["to"])

        out_dir = self.shared / "speakers"
        out_dir.mkdir(parents=True, exist_ok=True)
        samples: dict[str, str] = {}
        for n in names:
            src = self._sample_for(profiles_dir, n, rows.get(n))
            if not src or src.stat().st_size > self.MAX_SAMPLE_BYTES:
                continue
            safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in n).strip() + src.suffix.lower()
            dst = out_dir / safe
            if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                dst.write_bytes(src.read_bytes())
            samples[n] = safe
        _write_json(self.shared / "speakers.json",
                    {"schema": 1, "names": sorted(names, key=str.lower), "samples": samples})
        self.samples_exported = len(samples)
        return len(names)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", help="override path to 'My Drive/Meeting Recordings'")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    a = sub.add_parser("apply"); a.add_argument("--write", action="store_true")
    e = sub.add_parser("export-speakers"); e.add_argument("--profiles", required=True)
    sub.add_parser("plaud-queue")
    m = sub.add_parser("mark-transcribed"); m.add_argument("slug")
    args = ap.parse_args(argv)
    ss = SharedSync(Path(args.root) if args.root else None)

    if args.cmd == "status":
        ev = ss.pending_events()
        print(f"Drive root: {ss.root}")
        print(f"Pending events: {len(ev)}")
        for t in sorted({x['type'] for x in ev}):
            print(f"  {t}: {sum(1 for x in ev if x['type'] == t)}")
        print(f"Phone recordings awaiting desktop transcript: {len(ss.plaud_queue())}")
        print(f"Dictionary terms: {len(_read_json(ss.shared / 'dictionary.json', {}).get('terms', []))}")
    elif args.cmd == "apply":
        for line in ss.apply_events(write=args.write):
            print(("  " if args.write else "  [dry run] ") + line)
    elif args.cmd == "export-speakers":
        n = ss.export_speakers(Path(args.profiles))
        print(f"Exported {n} speaker names, {ss.samples_exported} voice samples")
    elif args.cmd == "plaud-queue":
        for p in ss.plaud_queue():
            print(p)
    elif args.cmd == "mark-transcribed":
        ss.mark_transcribed(args.slug); print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
