"""
console.py — one window for everything: run commands, review speakers.

Replaces remembering six PowerShell invocations and which order they go in.
Buttons run the same commands you would type, stream their output back, and
refuse to run two at once. The speaker review tabs from review.py are folded
in, so correcting a name and re-running identification happen in one place.

Local only — binds 127.0.0.1, no account, nothing leaves the machine.

    python console.py --archive "G:/My Drive/Meeting Recordings" \
                      --profiles ".\\Speaker_Profiles"
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
import wave
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

import numpy as np

STATE: dict = {"archive": None, "profiles": None, "root": None, "python": None}

# One job at a time, deliberately. Transcription saturates the CPU; letting
# someone queue three of them from a web page is a way to wedge the machine.
JOB: dict = {
    "name": None, "proc": None, "lines": deque(maxlen=400),
    "started": None, "finished": None, "code": None,
}
JOB_LOCK = threading.Lock()


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def build_commands() -> dict:
    """
    What the buttons do. Each entry is the argv the command would use if typed.

    `kind` drives the styling: primary for the things you run daily, danger
    for the one that stops recording.
    """
    py = STATE["python"]
    archive = str(STATE["archive"])
    profiles = str(STATE["profiles"]) if STATE["profiles"] else None
    library = str(STATE["profiles"] / "voiceprints.json") if STATE["profiles"] else None

    cmds = {
        "devices": {
            "label": "Audio devices",
            "hint": "What this machine has, and which ones actually open",
            "group": "Check", "kind": "ghost",
            "argv": [py, "-m", "meetingrec", "devices"],
        },
        "storage": {
            "label": "Test archiving",
            "hint": "Writes a throwaway file to the archive and reports where it landed",
            "group": "Check", "kind": "ghost",
            "argv": [py, "-m", "meetingrec", "storage"],
        },
        "list": {
            "label": "Who has the mic",
            "hint": "Live capture sessions — run this during a call",
            "group": "Check", "kind": "ghost",
            "argv": [py, "-m", "meetingrec", "list"],
        },
        "testrecord": {
            "label": "Test record (20s)",
            "hint": "Play audio and talk, then listen to all three files",
            "group": "Check", "kind": "ghost",
            "argv": [py, "-m", "meetingrec", "testrecord", "--seconds", "20"],
        },
        "diarize": {
            "label": "Identify speakers",
            "hint": "Cluster each system track and match against the voiceprint library",
            "group": "Process", "kind": "primary", "dated": True,
            "argv": ([py, "scripts/diarize.py", "--all", archive, "--library", library]
                     if library else None),
            "needs": "a voiceprint library",
        },
        "transcribe": {
            "label": "Transcribe",
            "hint": "Whisper over both tracks, merged with speaker names",
            "group": "Process", "kind": "primary", "dated": True,
            "argv": [py, "scripts/transcribe.py", "--all", archive],
        },
        "enroll": {
            "label": "Rebuild voiceprints",
            "hint": "Re-read the profile folder — run after correcting names",
            "group": "Process", "kind": "secondary",
            "argv": ([py, "scripts/enroll.py", profiles, "--verify"]
                     if profiles else None),
            "needs": "a profiles folder",
        },
        "diarize_force": {
            "label": "Re-identify everything",
            "hint": "Redo identification across the whole archive with the current library",
            "group": "Process", "kind": "secondary", "dated": True,
            "argv": ([py, "scripts/diarize.py", "--all", archive,
                      "--library", library, "--force"] if library else None),
            "needs": "a voiceprint library",
        },
        "classify": {
            "label": "Title & categorise",
            "hint": "Applies the naming rules — date, speakers, type — and sorts by category",
            "group": "Process", "kind": "primary", "dated": True,
            "argv": [py, "scripts/classify.py", "--all", archive],
        },
        "index": {
            "label": "Rebuild the index",
            "hint": "One searchable reference covering the whole archive, for an agent to read",
            "group": "Process", "kind": "secondary",
            "argv": [py, "scripts/index.py", archive],
        },
        "tests": {
            "label": "Run tests",
            "hint": "The full suite — use after updating",
            "group": "Maintain", "kind": "ghost",
            "argv": [py, "-m", "pytest", "-q"],
        },
    }
    return {k: v for k, v in cmds.items()}


def recorder_state() -> dict:
    """Is the recorder running, and how do we know."""
    running, pids = False, []
    try:
        import psutil

        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                joined = " ".join(proc.info["cmdline"] or [])
            except Exception:
                continue
            if "meetingrec" in joined and " run" in f" {joined} ":
                if proc.info["pid"] != os.getpid():
                    pids.append(proc.info["pid"])
        running = bool(pids)
    except Exception:
        pass

    lock = STATE["root"] / "logs" / "meetingrec.pid"
    return {
        "running": running,
        "pids": sorted(pids),
        "lock_pid": lock.read_text(encoding="utf-8").strip() if lock.exists() else None,
    }


def tail_log(lines: int = 40) -> list[str]:
    path = STATE["root"] / "logs" / "meetingrec.log"
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            return [ln.rstrip() for ln in deque(fh, maxlen=lines)]
    except Exception:
        return []


def start_job(name: str, argv: list[str]) -> dict:
    with JOB_LOCK:
        if JOB["proc"] and JOB["proc"].poll() is None:
            return {"ok": False, "error": f"{JOB['name']} is still running"}
        JOB.update({"name": name, "lines": deque(maxlen=400),
                    "started": time.time(), "finished": None, "code": None})

        flags = 0
        if sys.platform == "win32":
            # Below normal: these are background chores and the recorder,
            # if it is running, must win any contest for the CPU.
            flags = 0x00004000 | 0x08000000
        proc = subprocess.Popen(
            argv, cwd=str(STATE["root"]), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
            encoding="utf-8", errors="replace", creationflags=flags,
        )
        JOB["proc"] = proc

    def pump():
        for line in proc.stdout:  # type: ignore[union-attr]
            JOB["lines"].append(line.rstrip())
        proc.wait()
        JOB["code"] = proc.returncode
        JOB["finished"] = time.time()

    threading.Thread(target=pump, daemon=True).start()
    return {"ok": True}


def job_state() -> dict:
    proc = JOB["proc"]
    running = bool(proc and proc.poll() is None)
    return {
        "name": JOB["name"], "running": running, "code": JOB["code"],
        "lines": list(JOB["lines"]),
        "elapsed": round(time.time() - JOB["started"], 1) if JOB["started"] else 0,
    }


def control_recorder(action: str) -> dict:
    """Start or stop via the scheduled task, falling back to the process."""
    if sys.platform != "win32":
        return {"ok": False, "error": "Windows only"}
    verb = "Start-ScheduledTask" if action == "start" else "Stop-ScheduledTask"
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command",
                        f"{verb} -TaskName meetingrec"],
                       capture_output=True, text=True, timeout=30)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    if action == "stop":
        # Stopping the task does not always take the process with it.
        try:
            import psutil

            for pid in recorder_state()["pids"]:
                try:
                    psutil.Process(pid).terminate()
                except Exception:
                    pass
        except Exception:
            pass
    time.sleep(1.0)
    return {"ok": True, "state": recorder_state()}


# --------------------------------------------------------------------------
# review data (meetings and unnamed voices)
# --------------------------------------------------------------------------


def meeting_folders() -> list[Path]:
    return sorted({p.parent for p in STATE["archive"].glob("**/speakers.json")},
                  reverse=True)


def library_names() -> list[str]:
    profiles = STATE["profiles"]
    if not profiles or not profiles.is_dir():
        return []
    return sorted(p.name for p in profiles.iterdir() if p.is_dir())


SIDECAR_SKIP = ("speakers.json", "transcript.json")


def read_meta(folder: Path) -> dict:
    """
    Whatever metadata the recording came with.

    The desktop writes <name>-mic.wav.json; the phone writes <name>.meta.json
    with a different shape. Looking only for the desktop one meant every
    phone recording showed up titled "Meeting" with nothing else known.
    """
    candidates = sorted(folder.glob("*.json"),
                        key=lambda p: (not p.name.endswith("-mic.wav.json"),
                                       not p.name.endswith(".meta.json")))
    for sidecar in candidates:
        if sidecar.name in SIDECAR_SKIP or sidecar.name.endswith("-shots.json"):
            continue
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            return data
    return {}


def audio_duration(folder: Path) -> float:
    """
    How long the recording runs, from whichever track exists.

    Duration used to come from *-system.wav only, so every phone recording
    reported 0s — which also made the turn timeline collapse to nothing.
    """
    track = playable_track(folder)
    if track is None:
        return 0.0
    if track.suffix.lower() == ".wav":
        try:
            with wave.open(str(track), "rb") as w:
                return w.getnframes() / w.getframerate()
        except Exception:
            pass
    try:
        import soundfile as sf

        info = sf.info(str(track))
        return float(info.frames) / info.samplerate
    except Exception:
        return 0.0


def meeting_title(folder: Path, meta: dict, speakers: list) -> str:
    """
    A human label. Prefers a real title, then the app, then the people in it.

    "Meeting" told you nothing — with a dozen in the list the only way to
    tell them apart was the timestamp.
    """
    for key in ("title", "name", "subject", "summary_title"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:80]

    app = meta.get("app") or ""
    if app:
        return app.replace(".exe", "").replace("ms-teams", "Teams").title()

    named = [s["name"] for s in speakers if s.get("name")]
    if named:
        shown = ", ".join(n.replace("_", " ") for n in named[:2])
        return shown + (f" +{len(named) - 2}" if len(named) > 2 else "")
    if folder.name.endswith("-plaud"):
        return "Plaud recording"
    return "Meeting"


def meeting_payload(folder: Path) -> dict:
    data = json.loads((folder / "speakers.json").read_text(encoding="utf-8"))
    speakers = data.get("speakers", [])
    meta = read_meta(folder)

    # A title derived from the naming rules beats anything guessed from
    # metadata, so it wins when one exists.
    classification = {}
    path = folder / "classification.json"
    if path.exists():
        try:
            classification = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            classification = {}
    named = [s["name"].replace("_", " ") for s in speakers if s.get("name")]
    return {
        "folder": str(folder), "name": folder.name, "meta": meta,
        "title": classification.get("title") or meeting_title(folder, meta, speakers),
        "classified": bool(classification),
        "type": classification.get("type", ""),
        "category": classification.get("category", ""),
        "team_member": classification.get("team_member"),
        "flags": classification.get("flags", []),
        "duration": audio_duration(folder),
        "speakers": speakers,
        "named": named,
        "unnamed": sum(1 for s in speakers if not s.get("name")),
        "single_track": bool(data.get("single_track")),
        "shots": sorted(p.name for p in folder.glob("*-shot*.jpg")),
        "transcript": (folder / "transcript.md").exists(),
    }


# --------------------------------------------------------------------------
# collections
# --------------------------------------------------------------------------


def collections_path() -> Path:
    """
    Stored beside the recordings, not in the project folder.

    The archive is what syncs between machines, so collections follow the
    meetings they describe rather than living on one PC. "_shared" is already
    excluded from meeting scanning, so nothing here is mistaken for audio.
    """
    return STATE["archive"] / "_shared" / "collections.json"


def load_collections() -> dict:
    path = collections_path()
    if not path.exists():
        return {"collections": [], "assignments": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        log_warn("collections.json is unreadable — starting empty")
        return {"collections": [], "assignments": {}}
    data.setdefault("collections", [])
    data.setdefault("assignments", {})
    return data


def save_collections(data: dict) -> None:
    path = collections_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic: a torn write here would lose every rule and assignment, and
    # the reader treats a parse failure as "no collections at all".
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def log_warn(message: str) -> None:
    print(f"  {message}")


def matches_rule(meeting: dict, rule: dict) -> bool:
    """
    Does this meeting belong to a collection by rule?

    Speakers are the useful signal: a recurring call is recognisable by who
    is in it far more reliably than by its title, which is often just the
    meeting app's name.
    """
    if not rule:
        return False

    named = {n.lower() for n in meeting.get("named", [])}

    required = [n.lower() for n in rule.get("speakers_all", [])]
    if required and not all(n in named for n in required):
        return False

    any_of = [n.lower() for n in rule.get("speakers_any", [])]
    if any_of and not any(n in named for n in any_of):
        return False

    excluded = [n.lower() for n in rule.get("speakers_none", [])]
    if excluded and any(n in named for n in excluded):
        return False

    text = rule.get("title_contains", "").strip().lower()
    if text and text not in meeting.get("title", "").lower():
        return False

    # A rule of all-empty fields would match everything, which is never what
    # someone meant to build.
    return bool(required or any_of or excluded or text)


def assign_collection(meeting: dict, data: dict) -> Optional[str]:
    """
    Which collection a meeting is in.

    An explicit assignment always wins — a rule should never override a
    choice someone made by hand. Otherwise the first matching rule takes it,
    so order in the list is the priority order.
    """
    assigned = data.get("assignments", {}).get(meeting["name"])
    if assigned:
        return assigned
    for collection in data.get("collections", []):
        if matches_rule(meeting, collection.get("rule", {})):
            return collection["id"]
    return None


def cosine(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


def gather_unknowns(similarity: float = 0.55) -> dict:
    """Unidentified voices grouped across meetings, so one name fixes many."""
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
                "folder": str(folder), "meeting": folder.name,
                "label": sp["label"], "seconds": sp.get("seconds", 0.0),
                "confidence": sp.get("confidence", 0.0),
                "status": sp.get("status", "unknown"),
                "candidates": sp.get("candidates", []),
                "turns": sp.get("turns", []), "embedding": sp.get("embedding"),
            })

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

    total = len(meeting_folders())
    affected = len({m["meeting"] for g in groups for m in g["members"]})
    return {
        "groups": groups,
        "stats": {
            "meetings_total": total, "meetings_with_unknowns": affected,
            "meetings_clean": total - affected, "unknown_clusters": len(entries),
            "likely_people": len(groups),
            "unknown_minutes": round(sum(e["seconds"] for e in entries) / 60, 1),
            "ambiguous": sum(1 for e in entries if e.get("status") == "ambiguous"),
        },
    }


AUDIO_SUFFIXES = (".wav", ".mp3", ".ogg", ".opus", ".flac", ".m4a")


def playable_track(folder: Path) -> Optional[Path]:
    """
    The track the timeline refers to.

    Desktop meetings have a system track. Phone uploads are a single
    compressed file — looking only for *-system.wav meant clips from those
    meetings silently 404'd, so you could see a voice but never hear it,
    which is exactly when you most need to.
    """
    system = next(iter(sorted(folder.glob("*-system.wav"))), None)
    if system is not None:
        return system
    try:
        others = sorted(p for p in folder.iterdir()
                        if p.suffix.lower() in AUDIO_SUFFIXES
                        and not p.name.endswith(("-mic.wav", "-mix.wav",
                                                 "-mic-raw.wav")))
    except OSError:
        return None
    return others[0] if others else None


# One decoded recording, kept so that labelling a meeting does not re-decode
# a 90-minute MP3 for every clip. Only one, because these are large and the
# workflow walks through a single meeting at a time.
_DECODED: dict = {"key": None, "audio": None, "rate": 0}


def _decode_whole(track: Path):
    """Decode a file completely and remember it, keyed by path and mtime."""
    import soundfile as sf

    key = (str(track), track.stat().st_mtime_ns)
    if _DECODED["key"] == key:
        return _DECODED["audio"], _DECODED["rate"]
    audio, rate = sf.read(str(track), dtype="int16", always_2d=True)
    _DECODED.update({"key": key, "audio": audio, "rate": rate})
    return audio, rate


def slice_audio(track: Path, start: float, end: float) -> bytes:
    """
    A playable WAV of one turn, from any format.

    Seeking is tried first and is cheap when it works. It often does not:
    libsndfile's seek on compressed formats depends on the encoder, and MP3
    in particular can refuse or land in the wrong place. The fallback decodes
    the file once and slices from memory, which is why the result is cached —
    without that, every clip in a meeting would decode the whole recording
    again.
    """
    if end <= start:
        raise ValueError("empty time range")

    if track.suffix.lower() == ".wav":
        try:
            with wave.open(str(track), "rb") as w:
                rate, channels, width = (w.getframerate(), w.getnchannels(),
                                         w.getsampwidth())
                total = w.getnframes()
                first = max(0, int(start * rate))
                if first >= total:
                    raise ValueError("that range is past the end of the recording")
                w.setpos(first)
                frames = w.readframes(max(1, int((end - start) * rate)))
            if not frames:
                # An empty but valid WAV plays as silence with no explanation,
                # which is worse than saying nothing is there.
                raise ValueError("no audio in that range")
            return _wav_bytes(frames, channels, width, rate)
        except ValueError:
            raise            # a real answer: the range is wrong, not the codec
        except Exception:
            pass             # odd encoding — let soundfile deal with it

    import soundfile as sf

    info = sf.info(str(track))
    rate = info.samplerate
    first = max(0, int(start * rate))
    last = min(info.frames, int(end * rate)) if info.frames else int(end * rate)
    if last <= first:
        raise ValueError("that range is past the end of the recording")

    data = None
    try:
        data, rate = sf.read(str(track), start=first, stop=last,
                             dtype="int16", always_2d=True)
        if data.size == 0:
            data = None
    except Exception:
        data = None

    if data is None:
        whole, rate = _decode_whole(track)
        data = whole[first:last]
        if data.size == 0:
            raise ValueError("no audio in that range")

    return _wav_bytes(data.tobytes(), data.shape[1], 2, rate)


def _wav_bytes(frames: bytes, channels: int, width: int, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(width)
        out.setframerate(rate)
        out.writeframes(frames)
    return buf.getvalue()


def safe_folder_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9 _-]+", "", name).strip().replace(" ", "_")
    return cleaned or "Unnamed"


def enroll_from_meeting(folder: Path, speaker: dict, person: str) -> dict:
    """
    Write a corrected speaker's audio into the profile library.

    Samples captured under real meeting conditions match future meetings
    better than clean studio clips, so the library improves every time a
    correction is made.
    """
    profiles = STATE["profiles"]
    if not profiles:
        return {"ok": False, "error": "no profiles folder"}
    system = playable_track(folder)
    if system is None:
        return {"ok": False, "error": "no audio in that folder"}
    turns = sorted(speaker.get("turns", []),
                   key=lambda t: t["end"] - t["start"], reverse=True)[:12]
    if not turns:
        return {"ok": False, "error": "no turns"}

    # Built from the same slicer the player uses, so a correction made on a
    # phone recording enrolls the audio you actually listened to.
    pieces = [slice_audio(system, t["start"], t["end"])
              for t in sorted(turns, key=lambda t: t["start"])]
    chunks, rate, ch = [], 16000, 1
    for raw in pieces:
        with wave.open(io.BytesIO(raw), "rb") as w:
            rate, ch = w.getframerate(), w.getnchannels()
            chunks.append(w.readframes(w.getnframes()))
    width = 2

    target = profiles / safe_folder_name(person)
    target.mkdir(parents=True, exist_ok=True)
    out_path = target / f"{safe_folder_name(person)}_from_{folder.name[:20]}.wav"
    with wave.open(str(out_path), "wb") as out:
        out.setnchannels(ch)
        out.setsampwidth(width)
        out.setframerate(rate)
        out.writeframes(b"".join(chunks))
    return {"ok": True, "path": str(out_path),
            "seconds": round(sum(t["end"] - t["start"] for t in turns), 1)}


# --------------------------------------------------------------------------
# server
# --------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def handle_one_request(self):
        """
        A browser that navigates away mid-response aborts the socket, and the
        page polls every 1.2 seconds — so this happens constantly and printed
        a full traceback each time, burying anything that mattered.
        """
        try:
            super().handle_one_request()
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            self.close_connection = True

    def _send(self, body: bytes, content_type: str, status: int = 200):
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            self.close_connection = True

    def _send_range(self, path: Path, content_type: str):
        """
        Serve a file with byte-range support.

        Without this a browser cannot seek: it has the whole file but no way
        to ask for part of it, so setting currentTime on a streamed media
        element silently does nothing and playback stays at the beginning.
        The symptom is every clip playing from the start of the recording,
        which makes identifying a speaker forty minutes in impossible.
        """
        try:
            size = path.stat().st_size
        except OSError as exc:
            return self._json({"error": f"could not read it: {exc}"}, 500)

        start, end = 0, size - 1
        header = self.headers.get("Range", "")
        partial = False
        match = re.match(r"bytes=(\d*)-(\d*)", header.strip())
        if match:
            first, last = match.group(1), match.group(2)
            if first:
                start = int(first)
                if last:
                    end = min(int(last), size - 1)
            elif last:                      # a suffix range: the final N bytes
                start = max(0, size - int(last))
            if start >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            partial = True

        length = end - start + 1
        try:
            self.send_response(206 if partial else 200)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()

            # Streamed in chunks: a long recording is tens of megabytes and
            # reading it whole for every seek would stall the console.
            with path.open("rb") as fh:
                fh.seek(start)
                remaining = length
                while remaining > 0:
                    block = fh.read(min(256 * 1024, remaining))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            self.close_connection = True

    def _json(self, payload, status: int = 200):
        self._send(json.dumps(payload).encode(), "application/json", status)

    def do_GET(self):
        from urllib.parse import parse_qs, urlparse

        url = urlparse(self.path)
        query = parse_qs(url.query)

        if url.path == "/":
            return self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")

        if url.path == "/api/commands":
            cmds = {k: {kk: vv for kk, vv in v.items() if kk != "argv"}
                    | {"available": v.get("argv") is not None}
                    for k, v in build_commands().items()}
            return self._json({"commands": cmds})

        if url.path == "/api/status":
            return self._json({"app": IDENTITY,
                               "recorder": recorder_state(), "job": job_state(),
                               "log": tail_log(30)})

        if url.path == "/api/meetings":
            data = load_collections()
            meetings = [meeting_payload(f) for f in meeting_folders()]
            for meeting in meetings:
                meeting["collection"] = assign_collection(meeting, data)
            return self._json({"library": library_names(),
                               "collections": data["collections"],
                               "assignments": data["assignments"],
                               "meetings": meetings})

        if url.path == "/api/unknowns":
            return self._json(gather_unknowns())

        if url.path == "/api/audio":
            # The whole recording, for the browser to decode and seek.
            # Browsers handle MP3 and Opus natively and reliably, where
            # server-side decoding depends on whatever libsndfile build is
            # installed — one of which crashes the interpreter outright.
            folder = Path(query.get("folder", [""])[0])
            track = playable_track(folder)
            if track is None:
                return self._json({"error": "no audio in that folder"}, 404)
            types = {".mp3": "audio/mpeg", ".ogg": "audio/ogg",
                     ".opus": "audio/ogg", ".m4a": "audio/mp4",
                     ".flac": "audio/flac", ".wav": "audio/wav"}
            return self._send_range(
                track, types.get(track.suffix.lower(), "application/octet-stream"))

        if url.path == "/api/clip":
            folder = Path(query.get("folder", [""])[0])
            track = playable_track(folder)
            if track is None:
                return self._json({"error": "no audio in that folder"}, 404)
            try:
                clip = slice_audio(track,
                                   float(query.get("start", ["0"])[0]),
                                   float(query.get("end", ["0"])[0]))
            except Exception as exc:
                return self._json({"error": f"could not decode: {exc}"}, 500)
            return self._send(clip, "audio/wav")

        if url.path == "/api/shot":
            folder = Path(query.get("folder", [""])[0])
            path = folder / query.get("name", [""])[0]
            if not path.exists() or path.parent != folder:
                return self._json({"error": "not found"}, 404)
            return self._send(path.read_bytes(), "image/jpeg")

        if url.path == "/api/transcript":
            path = Path(query.get("folder", [""])[0]) / "transcript.md"
            if not path.exists():
                return self._json({"error": "no transcript"}, 404)
            return self._json({"text": path.read_text(encoding="utf-8")})

        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        from urllib.parse import urlparse

        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        url = urlparse(self.path)

        if url.path == "/api/run":
            name = payload.get("command", "")
            entry = build_commands().get(name)
            if not entry:
                return self._json({"ok": False, "error": "unknown command"}, 400)
            if not entry.get("argv"):
                return self._json({"ok": False,
                                   "error": f"needs {entry.get('needs', 'configuration')}"}, 400)

            argv = list(entry["argv"])
            since = (payload.get("since") or "").strip()
            # Validated here rather than passed through: this value reaches a
            # subprocess argument list, and only commands that understand it
            # should receive it.
            if since and entry.get("dated"):
                if not re.fullmatch(r"\d{4}-\d{2}-\d{2}|\d+[dwmy]", since):
                    return self._json({"ok": False,
                                       "error": "date must be YYYY-MM-DD or a span like 7d"}, 400)
                argv += ["--since", since]
            return self._json(start_job(name, argv))

        if url.path == "/api/collections":
            data = load_collections()
            action = payload.get("action")

            if action == "create":
                name = (payload.get("name") or "").strip()
                if not name:
                    return self._json({"ok": False, "error": "no name"}, 400)
                data["collections"].append({
                    "id": f"c{int(time.time() * 1000)}",
                    "name": name[:60],
                    "rule": payload.get("rule") or {},
                })
            elif action == "update":
                for collection in data["collections"]:
                    if collection["id"] == payload.get("id"):
                        if "name" in payload:
                            collection["name"] = (payload["name"] or "").strip()[:60]
                        if "rule" in payload:
                            collection["rule"] = payload["rule"] or {}
            elif action == "delete":
                target = payload.get("id")
                data["collections"] = [c for c in data["collections"]
                                       if c["id"] != target]
                # Assignments pointing at a deleted collection would strand
                # those meetings in a folder that no longer exists.
                data["assignments"] = {k: v for k, v in data["assignments"].items()
                                       if v != target}
            elif action == "assign":
                meeting = payload.get("meeting")
                target = payload.get("id")
                if not meeting:
                    return self._json({"ok": False, "error": "no meeting"}, 400)
                if target:
                    data["assignments"][meeting] = target
                else:
                    data["assignments"].pop(meeting, None)
            else:
                return self._json({"ok": False, "error": "unknown action"}, 400)

            save_collections(data)
            return self._json({"ok": True, "collections": data["collections"]})

        if url.path == "/api/reveal":
            # Opens the meeting's folder in the file manager — the quickest
            # way to get at the audio, screenshots and transcript directly.
            folder = Path(payload.get("folder", ""))
            if not folder.is_dir():
                return self._json({"ok": False, "error": "no such folder"}, 404)
            try:
                if sys.platform == "win32":
                    subprocess.Popen(["explorer", str(folder)])
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", str(folder)])
                else:
                    subprocess.Popen(["xdg-open", str(folder)])
            except Exception as exc:
                return self._json({"ok": False, "error": str(exc)}, 500)
            return self._json({"ok": True})

        if url.path == "/api/quit":
            # Launched from a shortcut there is no console to interrupt, so
            # the page needs a way to stop the server it is talking to.
            threading.Timer(0.3, lambda: os._exit(0)).start()
            return self._json({"ok": True})

        if url.path == "/api/recorder":
            return self._json(control_recorder(payload.get("action", "start")))

        if url.path == "/api/save":
            folder = Path(payload["folder"])
            path = folder / "speakers.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            changes = {c["label"]: c for c in payload.get("changes", [])}
            enrolled = []
            pending = []
            for speaker in data.get("speakers", []):
                change = changes.get(speaker["label"])
                if not change:
                    continue
                person = (change.get("name") or "").strip()
                speaker["name"] = person or None
                speaker["label"] = person or speaker["label"]
                speaker["corrected"] = True
                if person and change.get("enroll"):
                    pending.append((dict(speaker), person))

            # The name is written BEFORE enrollment is attempted. Enrollment
            # decodes audio, and on some builds that throws — which used to
            # abort the whole request, so the name was silently never saved.
            # Adding a sample is a bonus; recording the name is the point.
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")

            problems = []
            for speaker, person in pending:
                try:
                    result = enroll_from_meeting(folder, speaker, person)
                except Exception as exc:
                    problems.append(f"{person}: {exc}")
                    continue
                if result.get("ok"):
                    enrolled.append(f"{person} (+{result['seconds']}s)")
                else:
                    problems.append(f"{person}: {result.get('error')}")
            return self._json({"ok": True, "enrolled": enrolled,
                               "problems": problems})

        if url.path == "/api/name-group":
            person = (payload.get("name") or "").strip()
            if not person:
                return self._json({"error": "no name"}, 400)
            enrolled, touched, problems = [], 0, []
            for member in payload.get("members", []):
                folder = Path(member["folder"])
                path = folder / "speakers.json"
                if not path.exists():
                    continue
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except Exception as exc:
                    problems.append(f"{folder.name}: {exc}")
                    continue

                pending = []
                for speaker in data.get("speakers", []):
                    if speaker["label"] != member["label"]:
                        continue
                    speaker["name"] = person
                    speaker["label"] = person
                    speaker["corrected"] = True
                    touched += 1
                    if payload.get("enroll"):
                        pending.append(dict(speaker))

                # Name first, enrollment after. Enrollment decodes audio and
                # can throw; letting that abort the request meant the name
                # was never written and the correction silently vanished.
                try:
                    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                except Exception as exc:
                    problems.append(f"{folder.name}: could not save ({exc})")
                    continue

                for speaker in pending:
                    try:
                        result = enroll_from_meeting(folder, speaker, person)
                    except Exception as exc:
                        problems.append(f"{folder.name}: no sample ({exc})")
                        continue
                    if result.get("ok"):
                        enrolled.append(result["seconds"])

            return self._json({"ok": True, "meetings": touched,
                               "enrolled_seconds": round(sum(enrolled), 1),
                               "problems": problems})

        return self._json({"error": "not found"}, 404)


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>meetingrec console</title>
<style>
  :root {
    --bg:#2C3033; --bg-deep:#0F2940; --panel:rgba(255,255,255,.055);
    --panel-edge:rgba(255,255,255,.10); --rust:#A54F20; --rust-lift:#C25F28;
    --blue:#3270AD; --teal:#244E5A; --ember:#492F18;
    --text:#F2F4F5; --dim:#B9C0C4; --silver:#8E989E;
    --ok:#4F9D69; --warn:#C8912F; --bad:#C2553F; --teal-bright:#4A8FA3;
    --r:14px; --r-sm:10px;
  }
  * { box-sizing:border-box; }
  body {
    margin:0; min-height:100vh; color:var(--text);
    font:15px/1.55 "Segoe UI Variable Text","Segoe UI",system-ui,sans-serif;
    background:
      radial-gradient(1100px 620px at 12% -8%, rgba(165,79,32,.20), transparent 62%),
      radial-gradient(900px 520px at 92% 4%, rgba(50,112,173,.18), transparent 58%),
      linear-gradient(168deg,#2C3033 0%,#242A2E 46%,#0F2940 100%);
    background-attachment:fixed;
  }
  header {
    display:flex; align-items:center; gap:16px; flex-wrap:wrap;
    padding:18px 28px; border-bottom:1px solid var(--panel-edge);
    background:rgba(15,41,64,.34); backdrop-filter:blur(14px);
    position:sticky; top:0; z-index:20;
  }
  h1 { font-size:17px; font-weight:600; margin:0; letter-spacing:.2px; }
  .tabs { display:flex; gap:6px; margin-left:8px; }
  .tabs button {
    background:transparent; border:1px solid transparent; color:var(--dim);
    padding:8px 16px; border-radius:999px; font:inherit; font-size:14px;
    cursor:pointer; transition:background .16s, color .16s, border-color .16s;
  }
  .tabs button:hover { background:rgba(255,255,255,.07); color:var(--text); }
  .tabs button[aria-pressed="true"] {
    background:rgba(165,79,32,.26); border-color:rgba(165,79,32,.55); color:#fff;
  }
  .state { margin-left:auto; display:flex; align-items:center; gap:10px;
           font-size:13.5px; color:var(--dim); }
  .tabquit { background:rgba(255,255,255,.05); border:1px solid var(--panel-edge);
             color:var(--dim); padding:6px 13px; border-radius:999px;
             font:inherit; font-size:13px; cursor:pointer; margin-left:6px; }
  .tabquit:hover { background:rgba(194,85,63,.26); border-color:rgba(194,85,63,.5);
                   color:#fff; }
  .dot { width:9px; height:9px; border-radius:50%; background:var(--silver);
         box-shadow:0 0 0 4px rgba(255,255,255,.06); }
  .dot.on { background:var(--ok); box-shadow:0 0 0 4px rgba(79,157,105,.22); }
  .dot.off { background:var(--bad); box-shadow:0 0 0 4px rgba(194,85,63,.20); }
  main { padding:26px 28px 90px; max-width:1320px; }
  .withhelp { display:grid; grid-template-columns:minmax(0,1fr) 330px; gap:28px;
              align-items:start; }
  @media (max-width:1100px) { .withhelp { grid-template-columns:1fr; } }
  aside.help { position:sticky; top:92px; }
  aside.help > p { color:var(--silver); font-size:12.5px; margin:0 0 12px; }
  details {
    background:var(--panel); border:1px solid var(--panel-edge);
    border-radius:var(--r-sm); margin-bottom:8px; overflow:hidden;
    backdrop-filter:blur(10px);
  }
  details[open] { background:rgba(255,255,255,.075); }
  summary {
    cursor:pointer; padding:11px 14px; font-size:13.5px; font-weight:600;
    list-style:none; display:flex; align-items:center; gap:9px;
    transition:background .14s;
  }
  summary::-webkit-details-marker { display:none; }
  summary::before {
    content:"›"; display:inline-block; font-size:16px; line-height:1;
    color:var(--silver); transition:transform .16s;
  }
  details[open] summary::before { transform:rotate(90deg); }
  summary:hover { background:rgba(255,255,255,.05); }
  .body { padding:2px 14px 14px 30px; font-size:13px; color:var(--dim);
          line-height:1.6; }
  .body p { margin:0 0 9px; }
  .body ol, .body ul { margin:0 0 9px; padding-left:18px; }
  .body li { margin-bottom:5px; }
  .body code { background:rgba(0,0,0,.34); padding:1px 5px; border-radius:4px;
               font:12px "Cascadia Code",Consolas,monospace; color:#E0A878; }
  .body strong { color:var(--text); font-weight:600; }
  .body .warn { color:#E0BE63; }
  h2 { font-size:13px; font-weight:600; letter-spacing:.10em; text-transform:uppercase;
       color:var(--silver); margin:30px 0 14px; }
  h2:first-child { margin-top:0; }

  /* Each group of controls is its own surface. Without this the page reads
     as one long column of buttons and the headings get lost in it. */
  .panel {
    position:relative; background:rgba(255,255,255,.035);
    border:1px solid var(--panel-edge); border-radius:var(--r);
    padding:16px 18px 18px; margin:0 0 18px; backdrop-filter:blur(8px);
    overflow:hidden;
  }
  .panel::before {
    content:""; position:absolute; inset:0 auto 0 0; width:3px;
    background:var(--accent, var(--silver)); opacity:.9;
  }
  .panel > h2 { margin:0 0 13px; display:flex; align-items:center; gap:9px; }
  .panel > h2::before {
    content:""; width:7px; height:7px; border-radius:50%;
    background:var(--accent, var(--silver));
    box-shadow:0 0 0 3px color-mix(in srgb, var(--accent, var(--silver)) 22%, transparent);
  }
  .panel .lede { color:var(--silver); font-size:12.5px; margin:-6px 0 13px; }
  .panel-recorder { --accent:var(--rust); }
  .panel-check    { --accent:var(--teal-bright); }
  .panel-process  { --accent:var(--blue); }
  .panel-maintain { --accent:var(--silver); }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(248px,1fr)); gap:12px; }
  .range { display:flex; align-items:center; gap:8px; flex-wrap:wrap;
           margin:-4px 0 14px; }
  .range span { color:var(--silver); font-size:12.5px; }
  .range button {
    background:rgba(255,255,255,.05); border:1px solid var(--panel-edge);
    color:var(--dim); padding:6px 13px; border-radius:999px; font:inherit;
    font-size:13px; cursor:pointer; transition:background .15s,color .15s,border-color .15s;
  }
  .range button:hover { background:rgba(255,255,255,.09); color:var(--text); }
  .range button[aria-pressed="true"] {
    background:rgba(50,112,173,.30); border-color:rgba(50,112,173,.6); color:#fff;
  }
  .range input[type=date] {
    font:inherit; font-size:13px; padding:5px 9px; border-radius:var(--r-sm);
    border:1px solid var(--panel-edge); background:rgba(0,0,0,.26);
    color:var(--text); color-scheme:dark; min-width:0;
  }
  .card {
    background:var(--panel); border:1px solid var(--panel-edge);
    border-radius:var(--r); padding:15px 16px; backdrop-filter:blur(10px);
    text-align:left; color:var(--text); font:inherit; cursor:pointer;
    transition:transform .14s, background .16s, border-color .16s, box-shadow .16s;
  }
  .card:hover:not(:disabled) {
    transform:translateY(-2px); background:rgba(255,255,255,.085);
    border-color:rgba(255,255,255,.20); box-shadow:0 8px 26px rgba(0,0,0,.30);
  }
  .card:disabled { opacity:.42; cursor:not-allowed; }
  .card .t { font-weight:600; font-size:15px; display:block; margin-bottom:5px; }
  .card .h { font-size:12.5px; color:var(--dim); line-height:1.45; }
  .card.primary { background:linear-gradient(145deg,rgba(165,79,32,.34),rgba(165,79,32,.16));
                  border-color:rgba(194,95,40,.52); }
  .card.secondary { background:linear-gradient(145deg,rgba(50,112,173,.30),rgba(50,112,173,.13));
                    border-color:rgba(50,112,173,.48); }
  .card.danger { background:linear-gradient(145deg,rgba(194,85,63,.28),rgba(194,85,63,.12));
                 border-color:rgba(194,85,63,.46); }
  .console {
    margin-top:22px; background:rgba(0,0,0,.34); border:1px solid var(--panel-edge);
    border-radius:var(--r); overflow:hidden;
  }
  .console header {
    position:static; padding:11px 16px; background:rgba(255,255,255,.045);
    backdrop-filter:none; border-bottom:1px solid var(--panel-edge);
    font-size:13px; color:var(--dim); gap:10px;
  }
  .console pre {
    margin:0; padding:14px 16px; max-height:340px; overflow:auto;
    font:12.5px/1.6 "Cascadia Code",Consolas,monospace; color:var(--dim);
    white-space:pre-wrap; word-break:break-word;
  }
  .spin { width:12px; height:12px; border-radius:50%; border:2px solid rgba(255,255,255,.22);
          border-top-color:var(--rust-lift); animation:spin .8s linear infinite; }
  @keyframes spin { to { transform:rotate(360deg); } }
  @media (prefers-reduced-motion:reduce) { .spin { animation:none; } .card:hover { transform:none; } }
  /* review */
  .layout { display:grid; grid-template-columns:260px 1fr; gap:26px; }
  .side { border-right:1px solid var(--panel-edge); padding-right:18px; }
  .side h2 { margin:0 0 10px; }
  .meetinghead {
    background:rgba(255,255,255,.035); border:1px solid var(--panel-edge);
    border-radius:var(--r); padding:14px 17px; margin-bottom:16px;
  }
  .meetinghead h2 { margin:0 0 4px; color:var(--text); font-size:15px;
                    letter-spacing:0; text-transform:none; }
  .mtg { display:block; width:100%; text-align:left; background:none;
         border:0; border-left:2px solid transparent;
         padding:9px 11px; border-radius:var(--r-sm); color:var(--text);
         font:inherit; cursor:pointer; margin-bottom:3px; }
  .mtg:hover { background:rgba(255,255,255,.07); }
  .mtg[aria-current="true"] { background:rgba(165,79,32,.24);
                              border-left-color:var(--rust); }
  .mtg .when { color:var(--silver); font-size:11.5px; letter-spacing:.02em;
               text-transform:uppercase; }
  .mtg .what { display:block; font-size:13.5px; font-weight:600;
               margin:1px 0 2px; line-height:1.3; }
  .mtg .who { display:block; color:var(--dim); font-size:11.5px;
              overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .mtg .flag { color:#E08A5F; }

  .coll { margin-bottom:14px; }
  .coll > .head {
    display:flex; align-items:center; gap:7px; padding:5px 8px;
    color:var(--silver); font-size:11.5px; font-weight:600;
    letter-spacing:.09em; text-transform:uppercase; cursor:pointer;
    border-radius:var(--r-sm); user-select:none;
  }
  .coll > .head:hover { background:rgba(255,255,255,.05); color:var(--text); }
  .coll > .head .count { margin-left:auto; font-weight:400; letter-spacing:0;
                         text-transform:none; }
  .coll > .head .caret { transition:transform .16s; display:inline-block; }
  .coll.closed > .head .caret { transform:rotate(-90deg); }
  .coll.closed > .items { display:none; }
  .coll.rule > .head::after { content:"auto"; font-size:9.5px; font-weight:600;
                              letter-spacing:.06em; color:var(--blue);
                              border:1px solid rgba(50,112,173,.5);
                              padding:1px 5px; border-radius:999px; }
  .sidebtn { width:100%; margin-top:6px; background:rgba(255,255,255,.05);
             border:1px dashed var(--panel-edge); color:var(--dim);
             padding:8px; border-radius:var(--r-sm); font:inherit;
             font-size:12.5px; cursor:pointer; }
  .sidebtn:hover { background:rgba(255,255,255,.09); color:var(--text); }
  /* A speaker is a thing you act on — pick a name, listen, save — so it
     reads better as a card than as a row in a list. */
  .person {
    position:relative; background:rgba(255,255,255,.035);
    border:1px solid var(--panel-edge); border-left:3px solid var(--known);
    border-radius:var(--r); padding:15px 17px; margin-bottom:12px;
    backdrop-filter:blur(8px);
  }
  .person.unknown { border-left-color:var(--rust); }
  .person.ambiguous { border-left-color:var(--warn); }
  .person.too-short { border-left-color:var(--silver); opacity:.72; }
  .row { display:flex; align-items:baseline; gap:12px; }
  .who { font-size:17px; font-weight:600; }
  .who.unidentified { color:#E08A5F; }
  .who.ambiguous { color:#E0BE63; }
  .score { margin-left:auto; color:var(--silver); font-size:12.5px; }
  .timeline { position:relative; height:26px; margin:11px 0 12px;
              background:rgba(255,255,255,.07); border-radius:6px;
              overflow:hidden; cursor:pointer; }
  .turn { position:absolute; top:0; bottom:0; background:var(--blue);
          opacity:.82; min-width:2px; }
  .person.unknown .turn { background:var(--rust); }
  .person.ambiguous .turn { background:var(--warn); }
  .turn:hover { opacity:1; }
  .controls { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
  select, input[type=text] {
    font:inherit; font-size:14px; padding:8px 10px; border-radius:var(--r-sm);
    border:1px solid var(--panel-edge); background:rgba(0,0,0,.26);
    color:var(--text); min-width:230px;
  }
  .btn { font:inherit; font-size:14px; padding:8px 15px; border-radius:var(--r-sm);
         border:1px solid rgba(194,95,40,.55); cursor:pointer; color:#fff;
         background:linear-gradient(145deg,rgba(165,79,32,.44),rgba(165,79,32,.22)); }
  .btn.ghost { background:rgba(255,255,255,.055); border-color:var(--panel-edge);
               color:var(--text); }
  .btn:hover { filter:brightness(1.14); }
  .pick { display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-bottom:10px; }
  .pick span { color:var(--silver); font-size:13px; }
  label.enroll { display:flex; gap:7px; align-items:center; color:var(--dim); font-size:13px; }
  .where { color:var(--silver); font-size:12.5px; margin:6px 0 10px; }
  .where span { margin-right:14px; }
  .clips { margin:10px 0 13px; }
  .cliprow { display:flex; align-items:center; gap:6px; flex-wrap:wrap;
             margin-bottom:6px; }
  .cliplabel { color:var(--silver); font-size:11.5px; min-width:150px; }
  .clip {
    background:rgba(50,112,173,.22); border:1px solid rgba(50,112,173,.45);
    color:var(--text); font:inherit; font-size:11.5px; padding:4px 10px;
    border-radius:999px; cursor:pointer; font-variant-numeric:tabular-nums;
  }
  .clip:hover { background:rgba(50,112,173,.42); }
  .stats {
    background:rgba(255,255,255,.035); border:1px solid var(--panel-edge);
    border-left:3px solid var(--rust); border-radius:var(--r);
    padding:15px 18px; margin-bottom:16px; color:var(--dim); font-size:13px;
  }
  .stats b { display:block; font-size:34px; font-weight:600; color:#E08A5F;
             line-height:1.1; margin-bottom:3px; }
  /* calendar */
  .calbar { display:flex; align-items:center; gap:10px; margin-bottom:14px;
            flex-wrap:wrap; }
  .calbar h2 { margin:0; color:var(--text); font-size:17px; letter-spacing:-.01em;
               text-transform:none; }
  .calbar .spacer { flex:1 1 auto; }
  .navbtn { background:rgba(255,255,255,.055); border:1px solid var(--panel-edge);
            color:var(--dim); width:32px; height:32px; border-radius:50%;
            font:inherit; cursor:pointer; line-height:1; }
  .navbtn:hover { background:rgba(255,255,255,.1); color:var(--text); }
  .seg { display:flex; gap:2px; background:rgba(0,0,0,.25);
         border:1px solid var(--panel-edge); border-radius:999px; padding:2px; }
  .seg button { background:none; border:0; color:var(--dim); font:inherit;
                font-size:12.5px; padding:5px 13px; border-radius:999px;
                cursor:pointer; }
  .seg button[aria-pressed="true"] { background:var(--rust); color:#fff; }

  .week { display:grid; grid-template-columns:54px repeat(7, minmax(0,1fr));
          border:1px solid var(--panel-edge); border-radius:var(--r);
          overflow:hidden; background:rgba(255,255,255,.025); }
  .week .dayhead { padding:9px 6px; text-align:center; font-size:12px;
                   color:var(--silver); border-bottom:1px solid var(--panel-edge);
                   border-left:1px solid var(--panel-edge); }
  .week .dayhead b { display:block; color:var(--text); font-size:16px;
                     font-weight:600; }
  .week .dayhead.today b { color:var(--rust-lift); }
  .week .corner { border-bottom:1px solid var(--panel-edge); }
  .gut { border-left:0; }
  .gut div { height:44px; font-size:11px; color:var(--silver); text-align:right;
             padding-right:7px; transform:translateY(-7px); }
  .daycol { position:relative; border-left:1px solid var(--panel-edge); }
  .daycol .hr { height:44px; border-bottom:1px solid rgba(255,255,255,.055); }
  .ev { position:absolute; left:3px; right:3px; border-radius:7px;
        padding:4px 7px; font-size:11.5px; line-height:1.3; overflow:hidden;
        cursor:pointer; color:#EAF1F6; border-left:3px solid;
        background:rgba(50,112,173,.34); border-color:#5B9BD5; }
  .ev:hover { filter:brightness(1.2); z-index:5; }
  .ev b { display:block; font-weight:600; font-size:11.5px;
          overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .ev span { color:rgba(255,255,255,.72); font-size:10.5px; }
  .ev.unnamed { background:rgba(165,79,32,.34); border-color:#C96B33; }

  .month { display:grid; grid-template-columns:repeat(7, minmax(0,1fr));
           gap:5px; }
  .mcell { min-height:104px; background:rgba(255,255,255,.03);
           border:1px solid var(--panel-edge); border-radius:var(--r-sm);
           padding:7px; }
  .mcell.dim { opacity:.4; }
  .mcell .d { font-size:12px; color:var(--silver); margin-bottom:4px; }
  .mcell.today .d { color:var(--rust-lift); font-weight:700; }
  .pill { display:block; width:100%; text-align:left; background:rgba(50,112,173,.3);
          border:0; border-left:2px solid #5B9BD5; color:#EAF1F6; font:inherit;
          font-size:11px; padding:2px 5px; border-radius:4px; margin-bottom:3px;
          cursor:pointer; overflow:hidden; text-overflow:ellipsis;
          white-space:nowrap; }
  .pill.unnamed { background:rgba(165,79,32,.3); border-color:#C96B33; }
  .pill:hover { filter:brightness(1.25); }

  .shots { display:flex; gap:8px; flex-wrap:wrap; margin-top:16px;
           padding:13px; background:rgba(255,255,255,.03);
           border:1px solid var(--panel-edge); border-radius:var(--r); }
  .shots img { height:66px; border-radius:8px; border:1px solid var(--panel-edge); }
  .empty { color:var(--silver); padding:50px 0; }
  .note { color:var(--silver); font-size:13px; }
  button:focus-visible, select:focus-visible, .timeline:focus-visible,
  input:focus-visible { outline:2px solid var(--rust-lift); outline-offset:2px; }
</style>
</head>
<body>
<header>
  <h1>meetingrec</h1>
  <nav class="tabs">
    <button id="tab-c" aria-pressed="true">Control</button>
    <button id="tab-k" aria-pressed="false">Calendar</button>
    <button id="tab-m" aria-pressed="false">Meetings</button>
    <button id="tab-u" aria-pressed="false">Unnamed</button>
  </nav>
  <div class="state"><span class="dot" id="dot"></span><span id="statetext">checking…</span>
    <button class="tabquit" id="quit" title="Stop this console (the recorder keeps running)">Quit</button>
  </div>
</header>
<main id="main"><p class="empty">Loading…</p></main>

<div id="player" class="player" hidden>
  <button id="pp" class="pbtn" aria-label="Pause">❚❚</button>
  <button id="stop" class="pbtn" aria-label="Stop">■</button>
  <div class="ptrack"><div id="pfill"></div></div>
  <span id="ptime" class="ptime">0:00</span>
  <span id="pwhat" class="pwhat"></span>
</div>

<script>

const HELP = `
<p>What each button does, and the order that matters.</p>

<details open><summary>The short version</summary><div class="body">
<p>Day to day you need <strong>one</strong> thing: <code>Start recorder</code>.
Everything else happens on its own after 6pm.</p>
<p>The other buttons are for checking something, or for redoing work after you
correct a name.</p>
</div></details>

<details><summary>Order of operations</summary><div class="body">
<ol>
  <li><strong>Identify speakers</strong> — clusters each recording and puts
      names to the voices it recognises.</li>
  <li><strong>Transcribe</strong> — writes the transcript, using those names.</li>
  <li><strong>Meetings / Unnamed tabs</strong> — correct anything wrong.</li>
  <li><strong>Rebuild voiceprints</strong> — picks up audio from your corrections.</li>
  <li><strong>Re-identify everything</strong> — applies the improved library.</li>
</ol>
<p>Steps 1 and 2 are order-dependent: transcription reads what identification
wrote, so running it first gives you “Others” instead of names.</p>
</div></details>

<details><summary>Recorder</summary><div class="body">
<p><strong>Start recorder</strong> — watches for meetings and records them.
It also starts by itself when you log in, so you rarely need this.</p>
<p><strong>Stop recorder</strong> — stops the scheduled task and the running
process. Nothing is recorded while it is off.</p>
<p>The dot in the header shows whether it is up.</p>
</div></details>

<details><summary>Check</summary><div class="body">
<p><strong>Audio devices</strong> — lists every device and probes which ones
actually open. Run this when a recording comes out silent.</p>
<p><strong>Test archiving</strong> — writes a throwaway file to the archive
and reports where it landed.</p>
<p><strong>Who has the mic</strong> — live capture sessions. Run it
<em>during</em> a call to confirm detection sees it.</p>
<p><strong>Test record (20s)</strong> — no detector involved. Play something
and talk, then listen to all three files. Most audio problems show up here
before they ruin a real meeting.</p>
</div></details>

<details><summary>Process</summary><div class="body">
<p><strong>Only meetings from</strong> — limits the three buttons below to
recent work. Both scripts skip meetings they have already done, so this is
mostly about not walking an archive that grows forever — and it matters a lot
with <strong>Re-identify everything</strong>, which otherwise redoes every
meeting you have.</p>
<p><strong>Identify speakers</strong> — skips meetings already done, so it is
cheap to run repeatedly.</p>
<p><strong>Transcribe</strong> — slow. A 40-minute meeting takes several
minutes. It skips anything already transcribed.</p>
<p><strong>Rebuild voiceprints</strong> — re-reads the profile folder,
including samples added by your corrections.</p>
<p><strong>Re-identify everything</strong> — ignores previous results and
redoes the whole archive. Use after rebuilding voiceprints.</p>
<p><strong>Title &amp; categorise</strong> — applies the naming rules:
<code>Date / Speaker(s) / Type</code>, plus a content category and the Trim,
Flagged and one-on-one checks. Works best after transcription, since the
meeting type is decided from what was said.</p>
<p><strong>Rebuild the index</strong> — writes one searchable reference over
the whole archive to <code>_shared/INDEX.md</code> and
<code>index.json</code>, so an agent can answer "when did we last discuss
this" without opening recordings.</p>
</div></details>

<details><summary>Maintain</summary><div class="body">
<p><strong>Run tests</strong> — the full suite. Worth running after applying
an update, so a broken install shows up now rather than during a meeting.</p>
</div></details>

<details><summary>Fixing a wrong name</summary><div class="body">
<ol>
  <li>Open <strong>Meetings</strong> or <strong>Unnamed</strong>.</li>
  <li>Click a block on the timeline to hear that moment.</li>
  <li>Pick the right name, keep “add this audio to their profile” ticked.</li>
  <li>Save.</li>
  <li>Run <strong>Rebuild voiceprints</strong>, then
      <strong>Re-identify everything</strong>.</li>
</ol>
<p>Each correction also becomes an enrollment sample recorded in real meeting
conditions, so the same person is recognised without help next time.</p>
</div></details>

<details><summary>Calendar</summary><div class="body">
<p>Meetings laid out by when they happened, in <strong>Week</strong> or
<strong>Month</strong>. Click any block to open that meeting.</p>
<p>Blocks in orange have nobody identified in them yet — a quick way to see
which stretches still need labelling.</p>
<p>The week grid fits itself to the hours you actually record in, rather than
showing an empty midnight-to-midnight column.</p>
<p>Only meetings whose folder is named by timestamp appear; anything without
a date in its name cannot be placed.</p>
</div></details>

<details><summary>Titles and categories</summary><div class="body">
<p>Titles follow <code>Date / Speaker(s) / Type</code>. The date comes from
the recording, never from today. Speakers are whoever held at least 20% of
the talking — you are excluded from that count, so a colleague holding the
rest of a call you dominated still counts as the main speaker.</p>
<p>Categories sort the sidebar automatically. Folders you make by hand always
win over them.</p>
<p>A <span class="warn">⚑ flag</span> means a rule could not be applied
cleanly — nobody clearly dominant, only generic speaker labels, a recording
over three hours, or two team members who both qualify as the one-on-one
partner. Flagged is a request to look, not an error.</p>
<p>Rules live in <code>scripts/taxonomy.toml</code>: rosters, trigger
phrases and thresholds are all editable without touching code.</p>
<p>Nothing is renamed or moved on disk. The classification is stored beside
each recording and the sidebar sorts by it — a move that goes wrong loses a
recording, and the rest of this project navigates by the folder layout.</p>
</div></details>

<details><summary>Folders in the sidebar</summary><div class="body">
<p><strong>+ New folder</strong> at the bottom of the list makes one. You can
sort meetings into it by hand, or give it a rule.</p>
<p>A rule sorts by <strong>who is in the meeting</strong>. That works far
better than titles, which are usually just the name of the meeting app. Name
one or more people and anything with them in it lands there.</p>
<p>Folders with a rule are marked <strong>auto</strong>. Moving a meeting by
hand always wins over a rule, so nothing you file deliberately gets moved
back.</p>
<p>Rules are evaluated top to bottom and the first match takes the meeting, so
put the specific folders above the broad ones.</p>
<p>Folders live with the recordings rather than on this PC, so they follow the
archive between machines.</p>
</div></details>

<details><summary>Unnamed voices</summary><div class="body">
<p>Voices are grouped across meetings by how they sound, so naming one covers
every meeting that voice appears in.</p>
<p><strong>Too close to call</strong> means two people in the library scored
within a whisker of each other. Listen and pick — a wrong name gets believed
in a way “Speaker 3” does not.</p>
<p><strong>No match</strong> usually means that person is not enrolled yet.</p>
</div></details>

<details><summary>Phone recordings</summary><div class="body">
<p>Recordings uploaded from the phone are a <strong>single track</strong> with
everyone in it, including you.</p>
<p>Desktop meetings record you and the far end separately, so attribution is
free. A phone recording has no such split, which means every speaker has to be
identified — your own voice included.</p>
<p>They also fragment more: one microphone catching everyone at different
distances, lossily compressed, means the same person can sound different
across a long recording. Clustering is looser for these, and a merge pass
rejoins fragments afterwards — but if you still see one person as several
voices, name them all and they will combine.</p>
</div></details>

<details><summary>When something looks wrong</summary><div class="body">
<p><span class="warn">Silent recording</span> — run <strong>Audio
devices</strong>. A headset that is switched off still opens and returns
silence.</p>
<p><span class="warn">Nothing archived</span> — recordings under 30 seconds
are discarded. The log says so by name.</p>
<p><span class="warn">Heavy dropouts</span> — Bluetooth congestion. Fewer
paired devices helps more than any setting here.</p>
<p>Output from whatever ran last is in the panel below the buttons; when
nothing is running it shows the recorder log.</p>
</div></details>
`;

let DATA = null, UNKNOWNS = null, CURRENT = null, CHANGES = {}, VIEW = 'control';
let COMMANDS = null, POLL = null;
// Remembered across tab switches so a long session does not keep resetting
// to the whole archive.
let SINCE = '';
// Which sidebar folders are collapsed. Kept in memory so the list does not
// spring back open every time a save re-renders it.
const CLOSED = new Set();
// Categories come from the naming rules; hand-made folders always win.
let GROUP_BY_CATEGORY = true;

const fmt = s => { s = Math.round(s); const m = Math.floor(s / 60);
  return m ? `${m}m ${String(s % 60).padStart(2,'0')}s` : `${s}s`; };
const esc = t => String(t).replace(/[&<>"]/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

async function api(path, body) {
  const opts = body ? { method:'POST', headers:{'Content-Type':'application/json'},
                        body: JSON.stringify(body) } : {};
  return (await fetch(path, opts)).json();
}

/* ---------- control ---------- */

async function showControl() {
  if (!COMMANDS) COMMANDS = (await api('/api/commands')).commands;
  const groups = {};
  Object.entries(COMMANDS).forEach(([key, c]) =>
    (groups[c.group] = groups[c.group] || []).push([key, c]));

  const main = document.getElementById('main');
  const LEDE = {
    Check: 'Diagnostics. None of these touch your recordings.',
    Process: 'The work. Identification first, then transcription.',
    Maintain: 'Housekeeping.',
  };

  main.innerHTML = `<div class="withhelp"><div>
    <section class="panel panel-recorder">
    <h2>Recorder</h2>
    <p class="lede">Runs in the background and starts itself at log in.</p>
    <div class="grid">
      <button class="card primary" data-rec="start">
        <span class="t">Start recorder</span>
        <span class="h">Watch for meetings and record them automatically</span></button>
      <button class="card danger" data-rec="stop">
        <span class="t">Stop recorder</span>
        <span class="h">Stops the scheduled task and the running process</span></button>
    </div></section>
    ${Object.entries(groups).map(([name, items]) => `
      <section class="panel panel-${name.toLowerCase()}">
      <h2>${esc(name)}</h2>
      ${LEDE[name] ? `<p class="lede">${esc(LEDE[name])}</p>` : ''}
      ${name === 'Process' ? `
        <div class="range" id="range">
          <span>Only meetings from:</span>
          <button data-since="7d">Last 7 days</button>
          <button data-since="4w">Last month</button>
          <button data-since="3m">Last 3 months</button>
          <button data-since="" aria-pressed="true">Everything</button>
          <input type="date" id="sincedate" aria-label="or pick a date">
        </div>` : ''}
      <div class="grid">${items.map(([key, c]) => `
        <button class="card ${c.kind}" data-cmd="${key}" ${c.available ? '' : 'disabled'}>
          <span class="t">${esc(c.label)}</span>
          <span class="h">${esc(c.available ? c.hint : 'Needs ' + (c.needs || 'setup'))}</span>
        </button>`).join('')}</div></section>`).join('')}
    <div class="console">
      <header><span id="jobname">Output</span><span id="jobspin"></span>
        <span class="note" id="jobtime"></span></header>
      <pre id="out">Pick something above. Output appears here as it runs.</pre>
    </div></div>
    <aside class="help"><h2 style="margin-top:0">How this works</h2>${HELP}</aside>
    </div>`;

  const range = main.querySelector('#range');
  const dateBox = main.querySelector('#sincedate');
  const markRange = value => {
    SINCE = value;
    range.querySelectorAll('button').forEach(x =>
      x.setAttribute('aria-pressed', x.dataset.since === value ? 'true' : 'false'));
    if (!value || !/^\d{4}-/.test(value)) dateBox.value = '';
  };
  if (range) {
    range.querySelectorAll('button').forEach(b =>
      b.onclick = () => markRange(b.dataset.since));
    dateBox.onchange = () => {
      if (dateBox.value) {
        SINCE = dateBox.value;
        range.querySelectorAll('button').forEach(x =>
          x.setAttribute('aria-pressed', 'false'));
      }
    };
    markRange(SINCE);
  }

  main.querySelectorAll('[data-cmd]').forEach(b => b.onclick = async () => {
    const res = await api('/api/run', { command: b.dataset.cmd, since: SINCE });
    if (!res.ok) setOut(`Could not start: ${res.error}`);
  });
  main.querySelectorAll('[data-rec]').forEach(b => b.onclick = async () => {
    b.disabled = true;
    await api('/api/recorder', { action: b.dataset.rec });
    b.disabled = false;
    refresh();
  });
}

function setOut(text) { const o = document.getElementById('out'); if (o) o.textContent = text; }

async function refresh() {
  const s = await api('/api/status');
  const dot = document.getElementById('dot');
  dot.className = 'dot ' + (s.recorder.running ? 'on' : 'off');
  document.getElementById('statetext').textContent =
    s.recorder.running ? `recording service up (pid ${s.recorder.pids.join(', ')})`
                       : 'recorder not running';

  if (VIEW !== 'control') return;
  const name = document.getElementById('jobname');
  if (!name) return;
  const j = s.job;
  name.textContent = j.name ? j.name : 'Output';
  document.getElementById('jobspin').innerHTML = j.running ? '<span class="spin"></span>' : '';
  document.getElementById('jobtime').textContent = j.name
    ? (j.running ? `${j.elapsed}s` : (j.code === 0 ? 'finished' : `exit ${j.code}`)) : '';
  const out = document.getElementById('out');
  const text = j.lines.length ? j.lines.join('\n')
    : (s.log.length ? s.log.join('\n') : 'Pick something above.');
  if (out.textContent !== text) {
    const atBottom = out.scrollTop + out.clientHeight >= out.scrollHeight - 30;
    out.textContent = text;
    if (atBottom) out.scrollTop = out.scrollHeight;
  }
}

/* ---------- meetings ---------- */

async function loadMeetings() { DATA = await api('/api/meetings'); }

const DAYS = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

// A meeting placed by hand should not be described as sorted automatically.
const IS_MANUAL = m => (DATA.assignments || {})[m.name] !== undefined;

function meetingWhen(name) {
  // 20261101-075933-plaud
  const m = /^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})/.exec(name);
  if (!m) return { label: name.slice(0, 13), date: null };  // undated folder
  const d = new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5]);
  const hour = ((d.getHours() + 11) % 12) + 1;
  const ampm = d.getHours() < 12 ? 'am' : 'pm';
  return {
    label: `${DAYS[d.getDay()]} · ${MONTHS[d.getMonth()]} ${d.getDate()} · ${hour}:${m[5]}${ampm}`,
    date: d,
  };
}

function meetingButton(m, i) {
  const b = document.createElement('button');
  b.className = 'mtg';
  const when = meetingWhen(m.name);
  // Three names is enough to recognise a recurring call; more just truncates.
  const named = m.named || [];
  const who = named.slice(0, 3).join(', ')
    + (named.length > 3 ? ` +${named.length - 3}` : '');
  b.innerHTML =
    `<span class="when">${esc(when.label)}</span>`
    + `<span class="what">${esc(m.title || 'Meeting')}</span>`
    + `<span class="who">${who ? esc(who) : '<span class="flag">nobody identified</span>'}`
    + (m.duration ? ` · ${fmt(m.duration)}` : '')
    + (m.category ? ` · ${esc(m.category)}` : '')
    + (m.unnamed ? ` · <span class="flag">${m.unnamed} unnamed</span>` : '')
    + ((m.flags || []).length ? ` · <span class="flag">⚑ ${esc(m.flags.join(', '))}</span>` : '')
    + `</span>`;
  b.onclick = () => showMeeting(i);
  return b;
}

function renderList() {
  const side = document.getElementById('side');
  side.innerHTML = '';

  const byId = {};
  (DATA.collections || []).forEach(c => byId[c.id] = { ...c, items: [] });
  const loose = [];
  // Hand-made folders win; the rest fall back to the category the naming
  // rules assigned, so the list is organised even before anyone files
  // anything by hand.
  const byCategory = {};
  DATA.meetings.forEach((m, i) => {
    if (m.collection && byId[m.collection]) { byId[m.collection].items.push([m, i]); }
    else if (GROUP_BY_CATEGORY && m.category) {
      (byCategory[m.category] = byCategory[m.category] || []).push([m, i]);
    } else { loose.push([m, i]); }
  });

  const section = (title, items, collection) => {
    if (!items.length && !collection) return;
    const wrap = document.createElement('div');
    wrap.className = 'coll' + (collection && collection.rule &&
      Object.keys(collection.rule).length ? ' rule' : '');
    if (CLOSED.has(title)) wrap.classList.add('closed');
    wrap.innerHTML = `<div class="head" tabindex="0" role="button">
        <span class="caret">▾</span>${esc(title)}
        <span class="count">${items.length}</span></div>
      <div class="items"></div>`;
    const head = wrap.querySelector('.head');
    const toggle = () => {
      wrap.classList.toggle('closed');
      wrap.classList.contains('closed') ? CLOSED.add(title) : CLOSED.delete(title);
    };
    head.onclick = toggle;
    head.onkeydown = e => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); }
    };
    const box = wrap.querySelector('.items');
    items.forEach(([m, i]) => box.appendChild(meetingButton(m, i)));
    side.appendChild(wrap);
  };

  (DATA.collections || []).forEach(c => section(c.name, byId[c.id].items, c));
  Object.keys(byCategory).sort().forEach(name => section(name, byCategory[name], null));
  section((DATA.collections || []).length || Object.keys(byCategory).length
          ? 'Unsorted' : 'All meetings', loose, null);

  const add = document.createElement('button');
  add.className = 'sidebtn';
  add.textContent = '+ New folder';
  add.onclick = newCollection;
  side.appendChild(add);
}

async function newCollection() {
  const name = prompt('Folder name (e.g. "Lifecycle", "1:1s")');
  if (!name || !name.trim()) return;
  const who = prompt(
    'Sort meetings here automatically when these people are in them?\n'
    + 'Comma-separated, or leave blank to sort by hand.\n\n'
    + 'Known: ' + (DATA.library || []).slice(0, 12).join(', '));
  const rule = {};
  if (who && who.trim()) {
    rule.speakers_any = who.split(',').map(x => x.trim().replace(/_/g, ' '))
      .filter(Boolean);
  }
  await api('/api/collections', { action: 'create', name: name.trim(), rule });
  await loadMeetings();
  showMeetings();
}

function showMeetings() {
  if (!DATA || !DATA.meetings.length) {
    document.getElementById('main').innerHTML =
      '<p class="empty">No identified meetings yet. Run “Identify speakers” first.</p>';
    return;
  }
  document.getElementById('main').innerHTML =
    `<div class="layout"><div class="side" id="side"></div><div id="detail"></div></div>`;
  renderList();
  // CURRENT may have been set by a click in the calendar, so respect it
  // rather than snapping back to the newest meeting.
  showMeeting(Math.min(CURRENT ?? 0, DATA.meetings.length - 1));
}

function showMeeting(i) {
  CURRENT = i; CHANGES = {};
  document.querySelectorAll('.mtg').forEach((b, j) =>
    b.setAttribute('aria-current', j === i ? 'true' : 'false'));
  const current = document.querySelectorAll('.mtg')[i];
  if (current && current.scrollIntoView) {
    current.scrollIntoView({ block: 'nearest' });
  }
  const m = DATA.meetings[i], d = document.getElementById('detail');
  const when = meetingWhen(m.name);
  d.innerHTML = `<div class="meetinghead">
    <h2>${esc(m.title || m.name)}</h2>
    <p class="note" style="margin:0 0 10px">${esc(when.label)} ·
      ${m.duration ? fmt(m.duration) : 'length unknown'} ·
      ${m.speakers.length} voices${m.single_track ? ' · single track' : ''}${
        m.transcript ? ' · transcript ready' : ''}</p>
    <div class="controls">
      <button class="btn ghost" id="reveal">Open folder</button>
      <label class="enroll" for="coll">Folder</label>
      <select id="coll"></select>
      ${m.collection && !IS_MANUAL(m) ? '<span class="note">sorted automatically</span>' : ''}
    </div></div>`;

  d.querySelector('#reveal').onclick = async () => {
    const res = await api('/api/reveal', { folder: m.folder });
    if (!res.ok) setStatus(`Could not open it: ${res.error}`);
  };

  const sel = d.querySelector('#coll');
  sel.innerHTML = `<option value="">— unsorted —</option>`
    + (DATA.collections || []).map(c =>
        `<option value="${esc(c.id)}"${c.id === m.collection ? ' selected' : ''}>${esc(c.name)}</option>`).join('');
  sel.onchange = async () => {
    await api('/api/collections',
              { action: 'assign', meeting: m.name, id: sel.value });
    await loadMeetings();
    showMeetings();
  };
  m.speakers.forEach(sp => d.appendChild(personCard(m, sp)));
  if (m.shots.length) {
    const g = document.createElement('div');
    g.className = 'shots';
    g.innerHTML = m.shots.slice(0,10).map(s =>
      `<img loading="lazy" alt="" src="/api/shot?folder=${encodeURIComponent(m.folder)}&name=${encodeURIComponent(s)}">`).join('');
    d.appendChild(g);
  }
  const bar = document.createElement('p');
  bar.style.marginTop = '22px';
  bar.innerHTML = `<button class="btn" id="save">Save names</button>
    <span class="note" id="status" style="margin-left:12px">Click a block to hear that moment.</span>`;
  d.appendChild(bar);
  document.getElementById('save').onclick = () => save(m);
}

function personCard(m, sp) {
  const el = document.createElement('section');
  const status = sp.status || (sp.name ? 'named' : 'unknown');
  el.className = 'person' + (status === 'named' ? '' : ' ' + status);
  const conf = status === 'named' ? `matched ${sp.confidence.toFixed(2)} · z ${sp.z.toFixed(1)}`
    : status === 'ambiguous' ? `too close to call · gap ${(sp.margin||0).toFixed(2)}`
    : status === 'too-short' ? 'too short to match'
    : `no match · best ${sp.confidence.toFixed(2)}`;
  const cls = status === 'named' ? '' : status === 'ambiguous' ? 'ambiguous' : 'unidentified';

  el.innerHTML = `
    <div class="row"><span class="who ${cls}">${esc(sp.label)}</span>
      <span class="score">${fmt(sp.seconds)} · ${esc(conf)}</span></div>
    <div class="timeline" tabindex="0" role="group" aria-label="turns"></div>
    ${status === 'ambiguous' && (sp.candidates||[]).length ? `
      <div class="pick"><span>Sounds like one of these:</span>
        ${sp.candidates.slice(0,3).map(c =>
          `<button class="btn ghost" data-name="${esc(c.name)}">${esc(c.name)} · ${c.score.toFixed(2)}</button>`).join('')}
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
    const l = (sp.turns||[]).slice().sort((a,b)=>(b.end-b.start)-(a.end-a.start))[0];
    if (l) play(m.folder, l.start, l.end);
  };

  const sel = el.querySelector('select');
  sel.innerHTML = `<option value="">— leave unidentified —</option>`
    + DATA.library.map(n => `<option${n === sp.name ? ' selected' : ''}>${esc(n)}</option>`).join('')
    + `<option value="__new">+ someone not listed…</option>`;
  const free = el.querySelector('input[type=text]');
  const enroll = el.querySelector('.enroll input');
  const record = () => {
    const name = sel.value === '__new' ? free.value.trim() : sel.value;
    CHANGES[sp.label] = { label: sp.label, name, enroll: enroll.checked };
  };
  sel.onchange = () => { free.hidden = sel.value !== '__new'; if (!free.hidden) free.focus(); record(); };
  free.oninput = record; enroll.onchange = record;
  el.querySelectorAll('.pick button').forEach(b => b.onclick = () => {
    sel.value = b.dataset.name; free.hidden = true; record();
    setStatus(`${b.dataset.name} selected — Save names to confirm.`);
  });
  return el;
}

let audio = null, AUDIO_FOLDER = null, STOP_AT = null;

const PLAYER = {
  el: () => document.getElementById('player'),
  show(label) {
    const p = PLAYER.el();
    p.hidden = false;
    document.getElementById('pwhat').textContent = label || '';
    document.getElementById('pp').textContent = '❚❚';
  },
  hide() { PLAYER.el().hidden = true; document.getElementById('pfill').style.width = '0'; },
  tick() {
    if (!audio || CLIP_END === null) return;
    const span = CLIP_END - CLIP_START;
    const done = Math.min(Math.max(audio.currentTime - CLIP_START, 0), span);
    document.getElementById('pfill').style.width = (done / span * 100) + '%';
    document.getElementById('ptime').textContent = fmt(span - done) + ' left';
  },
};

let CLIP_START = 0, CLIP_END = null;

function stopAudio(hidePlayer) {
  if (audio) { audio.pause(); audio.ontimeupdate = null; }
  if (hidePlayer !== false) PLAYER.hide();
}

/* Play by seeking inside the whole recording.
   The browser decodes MP3 and Opus natively and caches the file, so after
   the first clip every other one starts instantly — and nothing depends on
   the server being able to decode compressed audio, which is exactly where
   this was failing. */
async function playInPlace(folder, start, end) {
  if (AUDIO_FOLDER !== folder || !audio) {
    stopAudio();
    audio = new Audio(`/api/audio?folder=${encodeURIComponent(folder)}`);
    AUDIO_FOLDER = folder;
    audio.preload = 'auto';
  }
  stopAudio();

  const begin = async () => {
    STOP_AT = end; CLIP_START = start; CLIP_END = end;

    // Wait for the seek to land before playing. Setting currentTime is a
    // request, not an instruction: the browser fetches the byte range and
    // fires 'seeked' when it is ready, and playing before that starts from
    // wherever it already was — which is the beginning.
    if (Math.abs(audio.currentTime - start) > 0.3) {
      await new Promise(done => {
        const ok = () => { audio.removeEventListener('seeked', ok); done(); };
        audio.addEventListener('seeked', ok);
        audio.currentTime = start;
        setTimeout(ok, 5000);   // never hang on a file that will not seek
      });
    }

    audio.ontimeupdate = () => {
      PLAYER.tick();
      if (STOP_AT !== null && audio.currentTime >= STOP_AT) stopAudio();
    };
    PLAYER.show(`${fmt(start)} – ${fmt(end)}`);
    await audio.play();

    // If the seek was refused outright, say so rather than quietly playing
    // the wrong part of a two-hour recording.
    if (start > 1 && audio.currentTime < 1) {
      setStatus('That recording will not seek — playing from the start.');
    }
    return true;
  };

  try {
    if (audio.readyState < 1) {
      await new Promise((ok, fail) => {
        audio.addEventListener('loadedmetadata', ok, { once: true });
        audio.addEventListener('error', () => fail(new Error('could not decode it')),
                               { once: true });
        setTimeout(() => fail(new Error('timed out loading')), 20000);
      });
    }
    await begin();
    setStatus(`Playing ${fmt(start)} – ${fmt(end)}`);
    return true;
  } catch (err) {
    audio = null; AUDIO_FOLDER = null;
    return false;
  }
}

async function play(folder, start, end) {
  if (!folder) return setStatus('That clip has no recording attached to it.');
  if (!(end > start)) return setStatus('That turn has no length recorded.');
  setStatus(`Loading ${fmt(start)} – ${fmt(end)}…`);
  if (await playInPlace(folder, start, end)) return;
  // Fall back to a server-cut clip: some formats a browser will not play.
  setStatus('Cutting that clip on the server…');
  return playCut(folder, start, end);
}

async function playCut(folder, start, end) {
  if (audio) { audio.pause(); if ((audio.src || '').startsWith('blob:')) URL.revokeObjectURL(audio.src); }

  // Fetched rather than handed to <audio> directly, so a failure can say
  // what actually went wrong instead of one generic message for everything
  // from a missing file to a decode error.
  let res;
  try {
    res = await fetch(`/api/clip?folder=${encodeURIComponent(folder)}&start=${start}&end=${end}`);
  } catch (e) {
    return setStatus('Could not reach the console — is it still running?');
  }
  if (!res.ok) {
    let why = `${res.status}`;
    try { why = (await res.json()).error || why; } catch (e) {}
    return setStatus(`Could not load that clip: ${why}`);
  }

  const blob = await res.blob();
  if (!blob.size) return setStatus('That clip came back empty.');
  audio = new Audio(URL.createObjectURL(blob));
  AUDIO_FOLDER = null;
  CLIP_START = 0; CLIP_END = end - start;
  audio.ontimeupdate = PLAYER.tick;
  audio.onended = () => PLAYER.hide();
  PLAYER.show(`${fmt(start)} – ${fmt(end)}`);
  audio.play()
    .then(() => setStatus(`Playing ${fmt(start)} – ${fmt(end)}`))
    .catch(err => setStatus(`Browser refused to play it: ${err.message}`));
}
function setStatus(t) { const s = document.getElementById('status'); if (s) s.textContent = t; }

async function save(m) {
  const changes = Object.values(CHANGES);
  if (!changes.length) return setStatus('Nothing changed yet.');
  setStatus('Saving…');
  const res = await api('/api/save', { folder: m.folder, changes });
  if (!res.ok) return setStatus('Save failed.');
  setStatus((res.enrolled.length
      ? `Saved. Added to profiles: ${res.enrolled.join(', ')}. Rebuild voiceprints to use them.`
      : 'Saved.')
    + ((res.problems || []).length ? ` No sample from ${res.problems.length}.` : ''));
  await loadMeetings(); showMeetings();
}

/* ---------- unnamed ---------- */

async function loadUnknowns() { UNKNOWNS = await api('/api/unknowns'); }

function showUnknowns() {
  const main = document.getElementById('main');
  if (!UNKNOWNS || !UNKNOWNS.groups.length) {
    main.innerHTML = '<p class="empty">Every voice has a name.</p>';
    return;
  }
  const s = UNKNOWNS.stats;
  main.innerHTML = `
    <div class="stats"><b>${s.likely_people}</b> unnamed ${s.likely_people === 1 ? 'voice' : 'voices'}
      · ${s.unknown_clusters} clusters · ${s.unknown_minutes} min
      · ${s.meetings_with_unknowns} of ${s.meetings_total} meetings
      ${s.ambiguous ? `· ${s.ambiguous} too close to call` : ''}</div>
    <p class="note">Grouped by how they sound, so one name covers every meeting that voice appears in.</p>`;
  UNKNOWNS.groups.forEach(g => main.appendChild(groupCard(g)));
  const bar = document.createElement('p');
  bar.style.marginTop = '18px';
  bar.innerHTML = '<span class="note" id="status"></span>';
  main.appendChild(bar);
}

function groupCard(g) {
  const el = document.createElement('section');
  el.className = 'person unknown';
  // Longest first: the best sample of this voice, and the one worth hearing.
  const members = g.members.slice().sort((a, b) => b.seconds - a.seconds);

  el.innerHTML = `
    <div class="row"><span class="who unidentified">Voice ${g.id}</span>
      <span class="score">${fmt(g.seconds)} across ${g.meetings} ${g.meetings===1?'meeting':'meetings'}</span></div>
    <div class="clips"></div>
    <div class="controls">
      <select></select>
      <input type="text" placeholder="or type a new name" hidden>
      <label class="enroll"><input type="checkbox" checked> add to their profile</label>
      <button class="btn apply">Name this voice</button>
    </div>`;

  /* One row per meeting, with a clickable block per turn — the same control
     the Meetings tab uses, rather than a separate "Listen" button that took
     its own path and did not work. Hearing several moments across different
     meetings is also how you actually recognise someone. */
  const clips = el.querySelector('.clips');
  members.slice(0, 4).forEach(member => {
    const turns = (member.turns || []).slice()
      .sort((a, b) => (b.end - b.start) - (a.end - a.start)).slice(0, 8);
    const row = document.createElement('div');
    row.className = 'cliprow';
    row.innerHTML = `<span class="cliplabel">${esc(member.meeting.slice(0, 13))}
      · ${fmt(member.seconds)}</span>`;
    if (!turns.length) {
      row.innerHTML += '<span class="note">no turn times recorded</span>';
    }
    turns.sort((a, b) => a.start - b.start).forEach(t => {
      const b = document.createElement('button');
      b.className = 'clip';
      b.textContent = fmt(t.end - t.start);
      b.title = `${member.meeting} · ${fmt(t.start)}`;
      b.onclick = () => play(member.folder, t.start, t.end);
      row.appendChild(b);
    });
    clips.appendChild(row);
  });

  const sel = el.querySelector('select');
  sel.innerHTML = `<option value="">— choose a name —</option>`
    + (DATA ? DATA.library : []).map(n => `<option>${esc(n)}</option>`).join('')
    + `<option value="__new">+ someone not listed…</option>`;
  const free = el.querySelector('input[type=text]');
  sel.onchange = () => { free.hidden = sel.value !== '__new'; if (!free.hidden) free.focus(); };

  el.querySelector('.apply').onclick = async () => {
    const name = sel.value === '__new' ? free.value.trim() : sel.value;
    if (!name) return setStatus('Pick a name first.');
    setStatus('Saving…');
    const res = await api('/api/name-group', { name, members: g.members,
      enroll: el.querySelector('.enroll input').checked });
    setStatus(`Named ${name} in ${res.meetings} meeting${res.meetings===1?'':'s'}`
      + (res.enrolled_seconds ? `, +${res.enrolled_seconds}s to their profile.` : '.')
      + ((res.problems || []).length
          ? ` (name saved; no sample from ${res.problems.length})` : ''));
    await loadMeetings(); await loadUnknowns(); showUnknowns();
  };
  return el;
}

/* ---------- calendar ---------- */

let CAL_MODE = 'week', CAL_ANCHOR = new Date();
const HOUR_PX = 44;

function startOfWeek(d) {
  const x = new Date(d.getFullYear(), d.getMonth(), d.getDate());
  x.setDate(x.getDate() - x.getDay());
  return x;
}
const sameDay = (a, b) => a.getFullYear() === b.getFullYear()
  && a.getMonth() === b.getMonth() && a.getDate() === b.getDate();

function datedMeetings() {
  // Only meetings whose folder name carries a timestamp can be placed.
  return DATA.meetings.map((m, i) => ({ m, i, at: meetingWhen(m.name).date }))
                      .filter(x => x.at);
}

function showCalendar() {
  const main = document.getElementById('main');
  main.innerHTML = `<div class="calbar">
      <button class="navbtn" id="prev" aria-label="Previous">‹</button>
      <button class="navbtn" id="next" aria-label="Next">›</button>
      <button class="btn ghost" id="today">Today</button>
      <h2 id="calrange"></h2>
      <span class="spacer"></span>
      <div class="seg">
        <button data-mode="week" aria-pressed="${CAL_MODE==='week'}">Week</button>
        <button data-mode="month" aria-pressed="${CAL_MODE==='month'}">Month</button>
      </div>
    </div><div id="calbody"></div>
    <p class="note" id="status" style="margin-top:14px"></p>`;

  main.querySelector('#prev').onclick = () => shiftCalendar(-1);
  main.querySelector('#next').onclick = () => shiftCalendar(1);
  main.querySelector('#today').onclick = () => { CAL_ANCHOR = new Date(); showCalendar(); };
  main.querySelectorAll('.seg button').forEach(b => b.onclick = () => {
    CAL_MODE = b.dataset.mode; showCalendar();
  });

  CAL_MODE === 'week' ? drawWeek() : drawMonth();
}

function shiftCalendar(direction) {
  const d = new Date(CAL_ANCHOR);
  CAL_MODE === 'week' ? d.setDate(d.getDate() + 7 * direction)
                      : d.setMonth(d.getMonth() + direction);
  CAL_ANCHOR = d;
  showCalendar();
}

function eventClass(m) { return m.unnamed && !(m.named || []).length ? ' unnamed' : ''; }

function openMeeting(index) {
  CURRENT = index;
  setView('meetings');
}

function drawWeek() {
  const start = startOfWeek(CAL_ANCHOR);
  const days = [...Array(7)].map((_, k) => {
    const d = new Date(start); d.setDate(d.getDate() + k); return d;
  });
  const last = days[6];
  document.getElementById('calrange').textContent =
    `${MONTHS[start.getMonth()]} ${start.getDate()} – `
    + (start.getMonth() === last.getMonth() ? '' : MONTHS[last.getMonth()] + ' ')
    + `${last.getDate()}, ${last.getFullYear()}`;

  const here = datedMeetings().filter(x => x.at >= days[0] &&
    x.at < new Date(last.getFullYear(), last.getMonth(), last.getDate() + 1));

  // Fit the grid to the day actually recorded rather than showing a dead
  // midnight-to-midnight column.
  let first = 8, lastHour = 18;
  here.forEach(x => {
    first = Math.min(first, x.at.getHours());
    const endH = x.at.getHours() + Math.ceil((x.m.duration || 1800) / 3600);
    lastHour = Math.max(lastHour, Math.min(23, endH));
  });

  const today = new Date();
  const grid = document.createElement('div');
  grid.className = 'week';
  grid.innerHTML = '<div class="corner"></div>'
    + days.map(d => `<div class="dayhead${sameDay(d, today) ? ' today' : ''}">`
        + `${DAYS[d.getDay()]}<b>${d.getDate()}</b></div>`).join('')
    + '<div class="gut">' + [...Array(lastHour - first + 1)].map((_, k) => {
        const h = first + k;
        const label = `${((h + 11) % 12) + 1}${h < 12 ? 'am' : 'pm'}`;
        return `<div>${label}</div>`;
      }).join('') + '</div>'
    + days.map(() => '<div class="daycol">'
        + '<div class="hr"></div>'.repeat(lastHour - first + 1) + '</div>').join('');

  const cols = grid.querySelectorAll('.daycol');
  here.forEach(({ m, i, at }) => {
    const col = cols[at.getDay()];
    if (!col) return;
    const top = ((at.getHours() - first) + at.getMinutes() / 60) * HOUR_PX;
    const height = Math.max(22, ((m.duration || 1800) / 3600) * HOUR_PX);
    const ev = document.createElement('button');
    ev.className = 'ev' + eventClass(m);
    ev.style.top = top + 'px';
    ev.style.height = height + 'px';
    const hour = ((at.getHours() + 11) % 12) + 1;
    ev.innerHTML = `<b>${esc(m.title || 'Meeting')}</b>`
      + `<span>${hour}:${String(at.getMinutes()).padStart(2,'0')}`
      + (m.duration ? ` · ${fmt(m.duration)}` : '') + `</span>`;
    ev.title = `${m.title}\n${(m.named || []).join(', ') || 'nobody identified'}`;
    ev.onclick = () => openMeeting(i);
    col.appendChild(ev);
  });

  const body = document.getElementById('calbody');
  body.innerHTML = '';
  body.appendChild(grid);
  if (!here.length) setStatus('Nothing recorded this week.');
}

function drawMonth() {
  const anchor = CAL_ANCHOR;
  const first = new Date(anchor.getFullYear(), anchor.getMonth(), 1);
  const gridStart = startOfWeek(first);
  document.getElementById('calrange').textContent =
    `${MONTHS[anchor.getMonth()]} ${anchor.getFullYear()}`;

  const byDay = {};
  datedMeetings().forEach(x => {
    const key = `${x.at.getFullYear()}-${x.at.getMonth()}-${x.at.getDate()}`;
    (byDay[key] = byDay[key] || []).push(x);
  });

  const today = new Date();
  const grid = document.createElement('div');
  grid.className = 'month';
  grid.innerHTML = DAYS.map(d =>
    `<div class="note" style="text-align:center;font-size:11.5px">${d}</div>`).join('');

  for (let k = 0; k < 42; k++) {
    const d = new Date(gridStart);
    d.setDate(d.getDate() + k);
    if (k >= 35 && d.getMonth() !== anchor.getMonth()) break;
    const cell = document.createElement('div');
    cell.className = 'mcell'
      + (d.getMonth() === anchor.getMonth() ? '' : ' dim')
      + (sameDay(d, today) ? ' today' : '');
    cell.innerHTML = `<div class="d">${d.getDate()}</div>`;
    (byDay[`${d.getFullYear()}-${d.getMonth()}-${d.getDate()}`] || [])
      .sort((a, b) => a.at - b.at)
      .forEach(({ m, i, at }) => {
        const pill = document.createElement('button');
        pill.className = 'pill' + eventClass(m);
        const hour = ((at.getHours() + 11) % 12) + 1;
        pill.textContent = `${hour}:${String(at.getMinutes()).padStart(2,'0')} ${m.title || 'Meeting'}`;
        pill.title = m.title;
        pill.onclick = () => openMeeting(i);
        cell.appendChild(pill);
      });
    grid.appendChild(cell);
  }

  const body = document.getElementById('calbody');
  body.innerHTML = '';
  body.appendChild(grid);
}

/* ---------- tabs ---------- */

async function setView(v) {
  VIEW = v;
  const map = { c:'control', k:'calendar', m:'meetings', u:'unknowns' };
  Object.keys(map).forEach(k => document.getElementById('tab-'+k)
    .setAttribute('aria-pressed', map[k] === v));
  if (v === 'control') return showControl();
  if (!DATA) await loadMeetings();
  if (v === 'calendar') return showCalendar();
  if (v === 'meetings') return showMeetings();
  await loadUnknowns(); showUnknowns();
}

document.getElementById('pp').onclick = () => {
  if (!audio) return;
  const btn = document.getElementById('pp');
  if (audio.paused) { audio.play(); btn.textContent = '❚❚'; btn.ariaLabel = 'Pause'; }
  else { audio.pause(); btn.textContent = '▶'; btn.ariaLabel = 'Play'; }
};
document.getElementById('stop').onclick = () => { stopAudio(); setStatus('Stopped.'); };
document.querySelector('.ptrack').addEventListener('click', e => {
  if (!audio || CLIP_END === null) return;
  const box = e.currentTarget.getBoundingClientRect();
  const frac = Math.min(Math.max((e.clientX - box.left) / box.width, 0), 1);
  audio.currentTime = CLIP_START + (CLIP_END - CLIP_START) * frac;
});
// Space pauses, Escape stops — a clip runs long enough to want both.
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.code === 'Space' && audio) { e.preventDefault(); document.getElementById('pp').click(); }
  if (e.code === 'Escape' && audio) { stopAudio(); setStatus('Stopped.'); }
});

document.getElementById('tab-c').onclick = () => setView('control');
document.getElementById('tab-k').onclick = () => setView('calendar');
document.getElementById('tab-m').onclick = () => setView('meetings');
document.getElementById('tab-u').onclick = () => setView('unknowns');

document.getElementById('quit').onclick = async () => {
  if (!confirm('Close the console? The recorder keeps running.')) return;
  clearInterval(POLL);
  try { await api('/api/quit', {}); } catch (e) {}
  document.body.innerHTML =
    '<main><p class="empty">Console closed. You can shut this tab.</p></main>';
};

setView('control');
refresh();
POLL = setInterval(refresh, 1200);
</script>
</body>
</html>
"""


IDENTITY = "meetingrec-console"


class ExclusiveHTTPServer(ThreadingHTTPServer):
    """
    A server that refuses to share a port.

    HTTPServer sets allow_reuse_address by default. On Linux that only
    skips the TIME_WAIT delay, but on Windows SO_REUSEADDR genuinely lets a
    second process bind a port another program is already listening on —
    both end up bound, and which one receives a given connection is
    arbitrary. The visible symptom is this console opening and another
    local app's page appearing instead.

    So: no address reuse, and on Windows ask for exclusive use explicitly,
    which makes a taken port fail loudly the way it already does elsewhere.
    """

    allow_reuse_address = False

    def server_bind(self):
        if os.name == "nt":
            import socket as _socket

            try:
                self.socket.setsockopt(_socket.SOL_SOCKET,
                                       _socket.SO_EXCLUSIVEADDRUSE, 1)
            except OSError:
                pass  # older Windows; allow_reuse_address = False still helps
        super().server_bind()


def bind(preferred: int, attempts: int = 12):
    """
    Claim a port, or explain who has it.

    Another local app on the same port used to be invisible: binding failed,
    but the browser was opened anyway and showed whatever was already
    listening — which looks for all the world like this app starting the
    wrong program.
    """
    for offset in range(attempts):
        port = preferred + offset
        try:
            server = ExclusiveHTTPServer(("127.0.0.1", port), Handler)
        except OSError as exc:
            if offset == 0:
                holder = whoever_holds(port)
                print(f"port {port} is already in use{holder}")
            continue
        # The port the OS actually gave us, which differs from the one asked
        # for whenever 0 is passed to mean "any free port".
        return server, server.server_address[1]

    print(f"could not find a free port between {preferred} and "
          f"{preferred + attempts - 1}", file=sys.stderr)
    return None, 0


def whoever_holds(port: int) -> str:
    """Name the process on a port, so the message is actionable."""
    try:
        import psutil

        for conn in psutil.net_connections(kind="inet"):
            if conn.laddr and conn.laddr.port == port and conn.pid:
                proc = psutil.Process(conn.pid)
                return f" by {proc.name()} (pid {conn.pid})"
    except Exception:
        pass
    return ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="meetingrec console.")
    ap.add_argument("--archive", required=True, help="the recordings archive root")
    ap.add_argument("--profiles", required=True, help="the speaker profiles folder")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    # python.exe, not pythonw: these jobs stream their output back to the page.
    venv_py = root / ".venv" / "Scripts" / "python.exe"
    STATE.update({
        "archive": Path(args.archive),
        "profiles": Path(args.profiles),
        "root": root,
        "python": str(venv_py if venv_py.exists() else sys.executable),
    })
    if not STATE["archive"].is_dir():
        print(f"not a folder: {args.archive}", file=sys.stderr)
        return 1

    print(f"project  : {root}")
    print(f"archive  : {STATE['archive']}  ({len(meeting_folders())} identified)")
    print(f"profiles : {STATE['profiles']}  ({len(library_names())} voices)")

    server, port = bind(args.port)
    if server is None:
        return 1

    url = f"http://127.0.0.1:{port}"
    if port != args.port:
        print(f"\nport {args.port} was taken by something else — using {port}")
    print(f"\nopen {url}   (Ctrl-C to stop)")
    if not args.no_browser:
        # Only after the port is ours. Opening the browser regardless meant
        # landing on whatever else was listening there, which looked exactly
        # like this app launching the wrong program.
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
