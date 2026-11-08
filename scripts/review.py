"""
review.py — listen to each detected voice and put the right name to it.

Diarisation gets most speakers right and honestly declines the rest. This is
where you close the gap: play a clip, pick a name, move on.

The part that compounds: correcting a speaker also writes their audio into the
profile library as a new enrollment sample, taken from a real meeting rather
than a clean studio clip. So the name you supply once is recognised
automatically next time, and the profile gets better at matching meeting
conditions every time you use it.

    python review.py --archive "G:/My Drive/Meeting Recordings" \
                     --profiles ".\\BIGGBY_Speaker_Profiles"

Opens at http://127.0.0.1:8765 — local only, no account, nothing leaves the box.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import threading
import wave
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

STATE: dict = {"archive": None, "profiles": None}


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


def meeting_folders() -> list[Path]:
    root: Path = STATE["archive"]
    return sorted({p.parent for p in root.glob("**/speakers.json")}, reverse=True)


def library_names() -> list[str]:
    profiles: Path = STATE["profiles"]
    if not profiles or not profiles.is_dir():
        return []
    return sorted(p.name for p in profiles.iterdir() if p.is_dir())


def meeting_payload(folder: Path) -> dict:
    data = json.loads((folder / "speakers.json").read_text(encoding="utf-8"))
    meta = {}
    for sidecar in folder.glob("*-mic.wav.json"):
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:
            pass
        break
    system = next(iter(sorted(folder.glob("*-system.wav"))), None)
    total = 0.0
    if system:
        with wave.open(str(system), "rb") as w:
            total = w.getnframes() / w.getframerate()
    return {
        "folder": str(folder),
        "name": folder.name,
        "meta": meta,
        "duration": total,
        "speakers": data.get("speakers", []),
        "shots": sorted(p.name for p in folder.glob("*-shot*.jpg")),
    }


def cosine(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


def gather_unknowns(similarity: float = 0.55) -> dict:
    """
    Every unidentified voice across the archive, grouped by who they probably are.

    Unknowns are only useful in aggregate. One stranger in one meeting is a
    shrug; the same stranger in nine meetings across two months is someone you
    work with regularly and should enrol. Grouping by centroid similarity turns
    a list of anonymous clusters into a list of people, and naming a group
    labels every meeting at once.
    """
    entries = []
    for folder in meeting_folders():
        try:
            data = json.loads((folder / "speakers.json").read_text(encoding="utf-8"))
        except Exception:
            continue
        for sp in data.get("speakers", []):
            if sp.get("name"):
                continue
            entries.append({
                "folder": str(folder),
                "meeting": folder.name,
                "label": sp["label"],
                "seconds": sp.get("seconds", 0.0),
                "confidence": sp.get("confidence", 0.0),
                "status": sp.get("status", "unknown"),
                "candidates": sp.get("candidates", []),
                "turns": sp.get("turns", []),
                "embedding": sp.get("embedding"),
            })

    # Group what can be grouped; anything recorded before centroids were saved
    # stays on its own rather than being guessed at.
    groups: list[dict] = []
    for entry in sorted(entries, key=lambda e: -e["seconds"]):
        vec = entry.get("embedding")
        placed = False
        if vec:
            for group in groups:
                if group["embedding"] and cosine(vec, group["embedding"]) >= similarity:
                    group["members"].append(entry)
                    group["seconds"] += entry["seconds"]
                    placed = True
                    break
        if not placed:
            groups.append({"embedding": vec, "members": [entry],
                           "seconds": entry["seconds"]})

    groups.sort(key=lambda g: -g["seconds"])
    for i, group in enumerate(groups, 1):
        group["id"] = i
        group["meetings"] = len({m["meeting"] for m in group["members"]})
        group.pop("embedding", None)

    total_meetings = len(meeting_folders())
    affected = len({m["meeting"] for g in groups for m in g["members"]})
    return {
        "groups": groups,
        "stats": {
            "meetings_total": total_meetings,
            "meetings_with_unknowns": affected,
            "meetings_clean": total_meetings - affected,
            "unknown_clusters": len(entries),
            "likely_people": len(groups),
            "unknown_minutes": round(sum(e["seconds"] for e in entries) / 60, 1),
            "ambiguous": sum(1 for e in entries if e.get("status") == "ambiguous"),
        },
    }


def slice_wav(system: Path, start: float, end: float) -> bytes:
    """A playable WAV of one turn, cut without rewriting the source."""
    with wave.open(str(system), "rb") as w:
        rate, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        w.setpos(max(0, int(start * rate)))
        frames = w.readframes(max(1, int((end - start) * rate)))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(ch)
        out.setsampwidth(width)
        out.setframerate(rate)
        out.writeframes(frames)
    return buf.getvalue()


def safe_folder_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9 _-]+", "", name).strip().replace(" ", "_")
    return cleaned or "Unnamed"


def enroll_from_meeting(folder: Path, speaker: dict, person: str) -> dict:
    """
    Write this cluster's audio into the profile library as a new sample.

    Meeting-condition samples are more useful than clean ones: they carry the
    same codec, room and headset as the recordings they will be matched
    against, which is exactly where clean enrollment clips fall down.
    """
    profiles: Path = STATE["profiles"]
    if not profiles:
        return {"ok": False, "error": "no profiles folder configured"}
    system = next(iter(sorted(folder.glob("*-system.wav"))), None)
    if system is None:
        return {"ok": False, "error": "no system track"}

    turns = sorted(speaker.get("turns", []), key=lambda t: t["end"] - t["start"],
                   reverse=True)[:12]
    if not turns:
        return {"ok": False, "error": "no turns"}

    with wave.open(str(system), "rb") as w:
        rate, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        chunks = []
        for turn in sorted(turns, key=lambda t: t["start"]):
            w.setpos(max(0, int(turn["start"] * rate)))
            chunks.append(w.readframes(max(1, int((turn["end"] - turn["start"]) * rate))))

    target = profiles / safe_folder_name(person)
    target.mkdir(parents=True, exist_ok=True)
    out_path = target / f"{safe_folder_name(person)}_from_{folder.name[:20]}.wav"
    with wave.open(str(out_path), "wb") as out:
        out.setnchannels(ch)
        out.setsampwidth(width)
        out.setframerate(rate)
        out.writeframes(b"".join(chunks))

    seconds = sum(t["end"] - t["start"] for t in turns)
    return {"ok": True, "path": str(out_path), "seconds": round(seconds, 1)}


# --------------------------------------------------------------------------
# server
# --------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    def _send(self, body: bytes, content_type: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload, status: int = 200):
        self._send(json.dumps(payload).encode(), "application/json", status)

    def do_GET(self):
        url = urlparse(self.path)
        query = parse_qs(url.query)

        if url.path == "/":
            return self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")

        if url.path == "/api/meetings":
            return self._json({
                "library": library_names(),
                "meetings": [meeting_payload(f) for f in meeting_folders()],
            })

        if url.path == "/api/unknowns":
            return self._json(gather_unknowns())

        if url.path == "/api/clip":
            folder = Path(query.get("folder", [""])[0])
            system = next(iter(sorted(folder.glob("*-system.wav"))), None)
            if system is None:
                return self._json({"error": "no system track"}, 404)
            start = float(query.get("start", ["0"])[0])
            end = float(query.get("end", ["0"])[0])
            return self._send(slice_wav(system, start, end), "audio/wav")

        if url.path == "/api/shot":
            folder = Path(query.get("folder", [""])[0])
            name = query.get("name", [""])[0]
            path = folder / name
            if not path.exists() or path.parent != folder:
                return self._json({"error": "not found"}, 404)
            return self._send(path.read_bytes(), "image/jpeg")

        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        url = urlparse(self.path)

        if url.path == "/api/name-group":
            # One name, applied everywhere that voice appears.
            person = (payload.get("name") or "").strip()
            if not person:
                return self._json({"error": "no name"}, 400)
            enrolled, touched = [], 0
            for member in payload.get("members", []):
                folder = Path(member["folder"])
                path = folder / "speakers.json"
                if not path.exists():
                    continue
                data = json.loads(path.read_text(encoding="utf-8"))
                for speaker in data.get("speakers", []):
                    if speaker["label"] != member["label"]:
                        continue
                    speaker["name"] = person
                    speaker["label"] = person
                    speaker["corrected"] = True
                    if payload.get("enroll"):
                        result = enroll_from_meeting(folder, speaker, person)
                        if result.get("ok"):
                            enrolled.append(result["seconds"])
                    touched += 1
                path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            return self._json({"ok": True, "meetings": touched,
                               "enrolled_seconds": round(sum(enrolled), 1)})

        if url.path == "/api/save":
            folder = Path(payload["folder"])
            path = folder / "speakers.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            changes = {c["label"]: c for c in payload.get("changes", [])}
            enrolled = []
            for speaker in data.get("speakers", []):
                change = changes.get(speaker["label"])
                if not change:
                    continue
                person = (change.get("name") or "").strip()
                speaker["name"] = person or None
                speaker["label"] = person or speaker["label"]
                speaker["corrected"] = True
                if person and change.get("enroll"):
                    result = enroll_from_meeting(folder, speaker, person)
                    if result.get("ok"):
                        enrolled.append(f"{person} (+{result['seconds']}s)")
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            return self._json({"ok": True, "enrolled": enrolled})

        return self._json({"error": "not found"}, 404)


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Who spoke</title>
<style>
  :root {
    --paper:#F2F4F3; --ink:#16201C; --muted:#6E7B74; --rule:#D6DCD8;
    --known:#2F6F4E; --unknown:#B4541F; --warn:#8A6A12; --field:#FFFFFF;
  }
  * { box-sizing:border-box; }
  body {
    margin:0; background:var(--paper); color:var(--ink);
    font:15px/1.5 ui-sans-serif,-apple-system,"Segoe UI",system-ui,sans-serif;
    display:grid; grid-template-columns:270px 1fr; min-height:100vh;
  }
  aside { border-right:1px solid var(--rule); padding:22px 0; overflow-y:auto; }
  aside h1 { font-size:15px; font-weight:600; margin:0 20px 14px; letter-spacing:-0.01em; }
  .tabs { display:flex; gap:2px; margin:0 20px 16px; }
  .tabs button { flex:1; padding:7px 0; font-size:13px; background:none;
                 color:var(--muted); border:1px solid var(--rule); border-radius:3px; }
  .tabs button[aria-pressed="true"] { background:var(--ink); color:var(--paper);
                                      border-color:var(--ink); }
  .stats { margin:0 20px 18px; font-size:13px; color:var(--muted);
           font-variant-numeric:tabular-nums; }
  .stats b { display:block; font-size:30px; color:var(--unknown); font-weight:600;
             letter-spacing:-0.02em; line-height:1.1; }
  .stats div { margin-top:9px; }
  .group-head { display:flex; align-items:baseline; gap:12px; }
  .seen { color:var(--muted); font-size:13px; font-variant-numeric:tabular-nums; }
  .where { color:var(--muted); font-size:13px; margin:6px 0 10px; }
  .where span { display:inline-block; margin-right:14px; }
  .mtg { display:block; width:100%; text-align:left; background:none; border:0;
         padding:9px 20px; cursor:pointer; color:var(--ink); font:inherit;
         border-left:3px solid transparent; }
  .mtg:hover { background:#E7EBE9; }
  .mtg[aria-current="true"] { border-left-color:var(--ink); background:#E7EBE9; }
  .mtg time { display:block; color:var(--muted); font-size:12.5px;
              font-variant-numeric:tabular-nums; }
  main { padding:34px 40px 80px; max-width:900px; }
  h2 { font-size:24px; font-weight:600; margin:0 0 4px; letter-spacing:-0.02em; }
  .sub { color:var(--muted); font-size:13.5px; font-variant-numeric:tabular-nums;
         margin-bottom:30px; }
  .person { border-top:1px solid var(--rule); padding:20px 0; }
  .row { display:flex; align-items:baseline; gap:12px; }
  .who { font-size:18px; font-weight:600; letter-spacing:-0.01em; }
  .who.unidentified { color:var(--unknown); font-weight:500; }
  .who.ambiguous { color:var(--warn); font-weight:500; }
  .person.ambiguous .turn { background:var(--warn); }
  .pick { display:flex; gap:8px; align-items:center; margin-bottom:10px;
          flex-wrap:wrap; }
  .pick span { color:var(--muted); font-size:13px; }
  .pick button { background:none; color:var(--ink); border:1px solid var(--warn);
                 padding:5px 11px; font-size:13.5px; }
  .pick button:hover { background:var(--warn); color:var(--paper); }
  .score { color:var(--muted); font-size:13px; font-variant-numeric:tabular-nums;
           margin-left:auto; }
  /* The timeline is the point of this screen: regular turn-taking looks
     different from two people wrongly merged into one cluster. */
  .timeline { position:relative; height:26px; margin:12px 0 14px;
              background:#E7EBE9; border-radius:2px; overflow:hidden; cursor:pointer; }
  .turn { position:absolute; top:0; bottom:0; background:var(--known); opacity:.78;
          min-width:2px; }
  .unidentified .turn, .person.unknown .turn { background:var(--unknown); }
  .turn:hover { opacity:1; }
  .controls { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
  select, input[type=text] {
    font:inherit; padding:7px 9px; border:1px solid var(--rule);
    border-radius:3px; background:var(--field); color:var(--ink); min-width:230px;
  }
  button { font:inherit; padding:7px 13px; border:1px solid var(--ink);
           background:var(--ink); color:var(--paper); border-radius:3px; cursor:pointer; }
  button.ghost { background:none; color:var(--ink); border-color:var(--rule); }
  button:focus-visible, select:focus-visible, .mtg:focus-visible,
  .timeline:focus-visible { outline:2px solid var(--ink); outline-offset:2px; }
  label.enroll { display:flex; gap:6px; align-items:center; color:var(--muted);
                 font-size:13px; }
  .bar { position:sticky; bottom:0; background:var(--paper);
         border-top:1px solid var(--rule); padding:14px 40px; margin:0 -40px;
         display:flex; gap:14px; align-items:center; }
  .note { color:var(--muted); font-size:13px; }
  .empty { color:var(--muted); padding:60px 0; }
  .shots { display:flex; gap:8px; margin-top:16px; flex-wrap:wrap; }
  .shots img { height:64px; border:1px solid var(--rule); border-radius:2px; }
  @media (prefers-reduced-motion:no-preference) { .turn { transition:opacity .12s; } }
</style>
</head>
<body>
<aside>
  <h1>Who spoke</h1>
  <div class="tabs">
    <button id="tab-m" aria-pressed="true">Meetings</button>
    <button id="tab-u" aria-pressed="false">Unnamed</button>
  </div>
  <div id="stats" class="stats"></div>
  <div id="list"></div>
</aside>
<main id="main"><p class="empty">Loading…</p></main>

<script>
let DATA = null, CURRENT = null, CHANGES = {}, UNKNOWNS = null, VIEW = 'meetings';

const fmt = s => {
  s = Math.round(s);
  const m = Math.floor(s / 60);
  return m ? `${m}m ${String(s % 60).padStart(2,'0')}s` : `${s}s`;
};

async function load() {
  DATA = await (await fetch('/api/meetings')).json();
  const list = document.getElementById('list');
  if (!DATA.meetings.length) {
    document.getElementById('main').innerHTML =
      '<p class="empty">No reviewed meetings yet. Run diarize.py first.</p>';
    return;
  }
  renderList();
  show(0);
}

function show(i) {
  CURRENT = i; CHANGES = {};
  document.querySelectorAll('.mtg').forEach((b, j) =>
    b.setAttribute('aria-current', j === i ? 'true' : 'false'));
  const m = DATA.meetings[i];
  const main = document.getElementById('main');

  main.innerHTML = `<h2>${m.meta.app || m.name}</h2>
    <p class="sub">${m.name.slice(0,13)} · ${fmt(m.duration)} · ${m.speakers.length} voices</p>`;

  m.speakers.forEach(sp => main.appendChild(personCard(m, sp)));

  if (m.shots.length) {
    const d = document.createElement('div');
    d.className = 'shots';
    d.innerHTML = m.shots.slice(0,8).map(s =>
      `<img loading="lazy" src="/api/shot?folder=${encodeURIComponent(m.folder)}&name=${encodeURIComponent(s)}" alt="">`
    ).join('');
    main.appendChild(d);
  }

  const bar = document.createElement('div');
  bar.className = 'bar';
  bar.innerHTML = `<button id="save">Save names</button>
    <span class="note" id="status">Click a block to hear that moment.</span>`;
  main.appendChild(bar);
  document.getElementById('save').onclick = () => save(m);
}

function personCard(m, sp) {
  const el = document.createElement('section');
  const status = sp.status || (sp.name ? 'named' : 'unknown');
  el.className = 'person' + (status === 'named' ? '' : ' ' + status);
  const conf = status === 'named'
    ? `matched ${sp.confidence.toFixed(2)} · z ${sp.z.toFixed(1)}`
    : status === 'ambiguous'
      ? `too close to call · gap ${(sp.margin || 0).toFixed(2)}`
      : status === 'too-short'
        ? 'too short to match'
        : `no match · best ${sp.confidence.toFixed(2)}`;
  const cls = status === 'named' ? '' : status === 'ambiguous' ? 'ambiguous' : 'unidentified';

  el.innerHTML = `
    <div class="row">
      <span class="who ${cls}">${sp.label}</span>
      <span class="score">${fmt(sp.seconds)} · ${conf}</span>
    </div>
    <div class="timeline" tabindex="0" role="group" aria-label="turns"></div>
    ${status === 'ambiguous' && (sp.candidates || []).length ? `
    <div class="pick">
      <span>Sounds like one of these — listen, then pick:</span>
      ${sp.candidates.slice(0,3).map(c =>
        `<button data-name="${c.name}">${c.name} · ${c.score.toFixed(2)}</button>`).join('')}
    </div>` : ''}
    <div class="controls">
      <select></select>
      <input type="text" placeholder="or type a new name" hidden>
      <label class="enroll"><input type="checkbox" checked> add this audio to their profile</label>
    </div>`;

  const tl = el.querySelector('.timeline');
  (sp.turns || []).forEach(t => {
    const b = document.createElement('div');
    b.className = 'turn';
    b.style.left = (t.start / m.duration * 100) + '%';
    b.style.width = Math.max(0.3, (t.end - t.start) / m.duration * 100) + '%';
    b.title = `${fmt(t.start)} – ${fmt(t.end)}`;
    b.onclick = e => { e.stopPropagation(); play(m.folder, t.start, t.end); };
    tl.appendChild(b);
  });
  tl.onclick = () => {
    const longest = (sp.turns || []).slice().sort((a,b)=>(b.end-b.start)-(a.end-a.start))[0];
    if (longest) play(m.folder, longest.start, longest.end);
  };

  const sel = el.querySelector('select');
  sel.innerHTML = `<option value="">— leave unidentified —</option>`
    + DATA.library.map(n => `<option${n === sp.name ? ' selected' : ''}>${n}</option>`).join('')
    + `<option value="__new">+ someone not listed…</option>`;
  const free = el.querySelector('input[type=text]');
  const enroll = el.querySelector('.enroll input');

  const record = () => {
    const name = sel.value === '__new' ? free.value.trim() : sel.value;
    CHANGES[sp.label] = { label: sp.label, name, enroll: enroll.checked };
  };
  sel.onchange = () => { free.hidden = sel.value !== '__new'; if (!free.hidden) free.focus(); record(); };
  free.oninput = record;
  enroll.onchange = record;

  // One click resolves the common case: the right answer is usually one of
  // the two names the matcher could not separate.
  el.querySelectorAll('.pick button').forEach(b => {
    b.onclick = () => {
      sel.value = b.dataset.name;
      free.hidden = true;
      record();
      status(`${b.dataset.name} selected — Save names to confirm.`);
    };
  });
  return el;
}

let audio = null;
function play(folder, start, end) {
  if (audio) audio.pause();
  audio = new Audio(`/api/clip?folder=${encodeURIComponent(folder)}&start=${start}&end=${end}`);
  audio.play().catch(() => status('Could not play that clip.'));
  status(`Playing ${fmt(start)} – ${fmt(end)}`);
}
function status(text) { const s = document.getElementById('status'); if (s) s.textContent = text; }

async function save(m) {
  const changes = Object.values(CHANGES);
  if (!changes.length) return status('Nothing changed yet.');
  status('Saving…');
  const res = await (await fetch('/api/save', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ folder: m.folder, changes })
  })).json();
  if (!res.ok) return status('Save failed.');
  status(res.enrolled.length
    ? `Saved. Added to profiles: ${res.enrolled.join(', ')}. Re-run enroll.py to use them.`
    : 'Saved.');
  DATA = await (await fetch('/api/meetings')).json();
  show(CURRENT);
}

async function loadUnknowns() {
  UNKNOWNS = await (await fetch('/api/unknowns')).json();
  const s = UNKNOWNS.stats;
  document.getElementById('stats').innerHTML = VIEW === 'unknowns'
    ? `<b>${s.likely_people}</b> unnamed ${s.likely_people === 1 ? 'voice' : 'voices'}
       <div>${s.unknown_clusters} clusters · ${s.unknown_minutes} min of speech</div>
       ${s.ambiguous ? `<div>${s.ambiguous} too close to call</div>` : ''}
       <div>${s.meetings_with_unknowns} of ${s.meetings_total} meetings affected</div>`
    : '';
}

function showUnknowns() {
  const main = document.getElementById('main');
  document.getElementById('list').innerHTML = '';
  if (!UNKNOWNS || !UNKNOWNS.groups.length) {
    main.innerHTML = '<p class="empty">Every voice has a name. Nothing to do here.</p>';
    return;
  }
  main.innerHTML = `<h2>Unnamed voices</h2>
    <p class="sub">Grouped by how they sound, so one name covers every meeting
    that voice appears in.</p>`;

  UNKNOWNS.groups.forEach(g => main.appendChild(groupCard(g)));
}

function groupCard(g) {
  const el = document.createElement('section');
  el.className = 'person unknown';
  const longest = g.members.slice().sort((a,b)=>b.seconds-a.seconds)[0];
  el.innerHTML = `
    <div class="group-head">
      <span class="who unidentified">Voice ${g.id}</span>
      <span class="seen">${fmt(g.seconds)} across ${g.meetings}
        ${g.meetings === 1 ? 'meeting' : 'meetings'}</span>
    </div>
    <p class="where">${g.members.map(m =>
      `<span>${m.meeting.slice(0,13)} · ${fmt(m.seconds)}</span>`).join('')}</p>
    <div class="controls">
      <button class="ghost listen">Listen</button>
      <select></select>
      <input type="text" placeholder="or type a new name" hidden>
      <label class="enroll"><input type="checkbox" checked> add to their profile</label>
      <button class="apply">Name this voice</button>
    </div>`;

  const sel = el.querySelector('select');
  sel.innerHTML = `<option value="">— choose a name —</option>`
    + DATA.library.map(n => `<option>${n}</option>`).join('')
    + `<option value="__new">+ someone not listed…</option>`;
  const free = el.querySelector('input[type=text]');
  sel.onchange = () => { free.hidden = sel.value !== '__new'; if (!free.hidden) free.focus(); };

  el.querySelector('.listen').onclick = () => {
    const turn = (longest.turns || []).slice()
      .sort((a,b)=>(b.end-b.start)-(a.end-a.start))[0];
    if (turn) play(longest.folder, turn.start, turn.end);
  };

  el.querySelector('.apply').onclick = async () => {
    const name = sel.value === '__new' ? free.value.trim() : sel.value;
    if (!name) return status('Pick a name first.');
    status('Saving…');
    const res = await (await fetch('/api/name-group', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ name, members: g.members,
                             enroll: el.querySelector('.enroll input').checked })
    })).json();
    status(`Named ${name} in ${res.meetings} meeting${res.meetings===1?'':'s'}`
           + (res.enrolled_seconds ? `, +${res.enrolled_seconds}s to their profile.` : '.'));
    DATA = await (await fetch('/api/meetings')).json();
    await loadUnknowns();
    showUnknowns();
    const bar = document.createElement('div');
    bar.className = 'bar';
    bar.innerHTML = '<span class="note" id="status">Saved.</span>';
    document.getElementById('main').appendChild(bar);
  };
  return el;
}

function setView(v) {
  VIEW = v;
  document.getElementById('tab-m').setAttribute('aria-pressed', v === 'meetings');
  document.getElementById('tab-u').setAttribute('aria-pressed', v === 'unknowns');
  loadUnknowns().then(() => {
    if (v === 'unknowns') showUnknowns();
    else { renderList(); show(CURRENT ?? 0); }
  });
}

function renderList() {
  const list = document.getElementById('list');
  list.innerHTML = '';
  DATA.meetings.forEach((m, i) => {
    const b = document.createElement('button');
    b.className = 'mtg';
    const unnamed = m.speakers.filter(s => !s.name).length;
    b.innerHTML = `${m.meta.app || 'Meeting'}<time>${m.name.slice(0,13)} · ${fmt(m.duration)}`
      + (unnamed ? ` · ${unnamed} unnamed` : '') + `</time>`;
    b.onclick = () => show(i);
    list.appendChild(b);
  });
}

document.getElementById('tab-m').onclick = () => setView('meetings');
document.getElementById('tab-u').onclick = () => setView('unknowns');

load().then(loadUnknowns);
</script>
</body>
</html>
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Review and correct speaker labels.")
    ap.add_argument("--archive", required=True, help="the Meeting Recordings root")
    ap.add_argument("--profiles", required=True, help="the speaker profiles folder")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)

    STATE["archive"] = Path(args.archive)
    STATE["profiles"] = Path(args.profiles)
    if not STATE["archive"].is_dir():
        print(f"not a folder: {args.archive}", file=sys.stderr)
        return 1

    folders = meeting_folders()
    print(f"{len(folders)} reviewed meeting(s), {len(library_names())} profile(s)")
    if not folders:
        print("Nothing to review yet — run diarize.py first.")

    url = f"http://127.0.0.1:{args.port}"
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"open {url}   (Ctrl-C to stop)")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
