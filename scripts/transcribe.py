"""
transcribe.py — turn recorded meetings into labelled transcripts.

Deliberately separate from the recorder. Transcription has entirely different
dependencies, runs for minutes rather than milliseconds, and benefits from
being re-runnable over old recordings when you change models. Keeping it out
of the capture path means a transcription failure can never cost you audio.

The speaker labelling is close to free, and that is a consequence of how the
recording was made rather than anything clever here. Two physically separate
tracks mean the mic is unambiguously you and the system track is unambiguously
everyone else. No diarisation model has to guess, because the microphone
already knew. Transcribe each side on its own, merge on the shared timeline —
which the recorder aligned — and the attribution is correct by construction.

Usage:
    # one meeting
    python transcribe.py "G:/My Drive/Meeting Recordings/MACHINE/2026/10/20261013-.../"

    # every meeting not yet done
    python transcribe.py --all "G:/My Drive/Meeting Recordings"

    # bigger model, and summarise with a local Ollama model
    python transcribe.py --all "G:/..." --model large-v3 --summarise

Install:
    py -m pip install faster-whisper
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

OLLAMA_URL = "http://localhost:11434/api/generate"

# Recordings uploaded from the phone arrive in whatever the device produced.
AUDIO_SUFFIXES = (".wav", ".mp3", ".ogg", ".opus", ".flac", ".m4a")

SUMMARY_PROMPT = """You are summarising a meeting transcript.

Write, in plain language:
1. A two or three sentence summary of what the meeting was about.
2. Key decisions, as bullets. Only decisions that were actually made.
3. Action items as bullets, each with who owns it if the transcript says.
4. Open questions left unresolved.

If a section has nothing in it, write "None" rather than inventing content.
Do not speculate beyond what was said.

TRANSCRIPT:
{transcript}
"""


def parse_since(spec: Optional[str]) -> Optional[datetime]:
    """
    Accepts 2026-10-01, 20261001, or a span like 7d / 6w / 3m.

    Spans are what you actually want day to day — "the last week" stays
    correct tomorrow, where a fixed date quietly stops filtering anything.
    """
    if not spec:
        return None
    text = spec.strip().lower()

    match = re.fullmatch(r"(\d+)\s*([dwmy])", text)
    if match:
        count, unit = int(match.group(1)), match.group(2)
        days = {"d": 1, "w": 7, "m": 30, "y": 365}[unit] * count
        return datetime.now() - timedelta(days=days)

    for fmt in ("%Y-%m-%d", "%Y%m%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"could not read a date from {spec!r} "
                     "(try 2026-10-01, or a span like 7d)")


def folder_date(folder: Path) -> Optional[datetime]:
    """
    The meeting date, from the folder name rather than the filesystem.

    Copying an archive or re-syncing it rewrites every modification time, so
    mtime says when the file was last touched, not when the meeting happened.
    The name carries the real answer.
    """
    match = re.match(r"(\d{8})[-_]?(\d{6})?", folder.name)
    if not match:
        return None
    try:
        stamp = match.group(1) + (match.group(2) or "000000")
        return datetime.strptime(stamp, "%Y%m%d%H%M%S")
    except ValueError:
        return None


def after_cutoff(folder: Path, since: Optional[datetime]) -> bool:
    """Undated folders are always processed — better than skipping silently."""
    if since is None:
        return True
    when = folder_date(folder)
    return when is None or when >= since


def human_duration(seconds: float) -> str:
    """Minutes once there are minutes, seconds before that — so a speaker who
    said one sentence does not appear as "0 min"."""
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m {seconds % 60:02d}s" if seconds % 60 else f"{seconds // 60}m"


@dataclass
class Segment:
    speaker: str
    start: float
    end: float
    text: str

    @property
    def stamp(self) -> str:
        return str(timedelta(seconds=int(self.start)))


# --------------------------------------------------------------------------
# transcription
# --------------------------------------------------------------------------


def default_threads() -> int:
    """
    Half the cores, minimum two.

    Leaving headroom is cheaper than reacting to contention. Whisper will
    happily saturate every core, and a recorder sharing the machine then
    drops audio it cannot recapture. Transcription finishing in nine minutes
    instead of five costs nothing anyone notices.
    """
    import os

    return max(2, (os.cpu_count() or 4) // 2)


def load_model(size: str, compute_type: str = "int8", threads: Optional[int] = None):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("faster-whisper is not installed.\n"
              "  py -m pip install faster-whisper", file=sys.stderr)
        raise SystemExit(1)
    threads = threads or default_threads()
    print(f"loading whisper '{size}' ({compute_type}, {threads} threads) "
          "— first run downloads the model")
    return WhisperModel(size, device="cpu", compute_type=compute_type,
                        cpu_threads=threads)


def decode_audio(path: Path, target_rate: int = 16000) -> Optional[np.ndarray]:
    """
    Decode anything we might be handed, into a 16 kHz mono array.

    WAV goes through the standard library. Everything else goes through
    soundfile, which matters because phone uploads are Opus or MP3 — and
    because letting whisper open those files is exactly the coupling that
    broke when an upstream audio library changed its API.
    """
    if path.suffix.lower() == ".wav":
        decoded = decode_wav(path, target_rate)
        if decoded is not None:
            return decoded
    try:
        import soundfile as sf

        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        return None  # let whisper try

    audio = data.mean(axis=1)
    if rate != target_rate and audio.size:
        n_out = int(round(len(audio) * target_rate / rate))
        audio = np.interp(np.linspace(0, len(audio) - 1, n_out),
                          np.arange(len(audio)), audio).astype(np.float32)
    return np.ascontiguousarray(audio, dtype=np.float32)


def decode_wav(path: Path, target_rate: int = 16000) -> Optional[np.ndarray]:
    """
    Decode a WAV ourselves and hand whisper a plain array.

    faster-whisper normally decodes through PyAV, which couples us to that
    library's API across versions — one release changed av.open() and every
    transcription died with an unexpected-keyword error. Our recordings are
    uncompressed WAV written by this project, so the standard library can read
    them and the whole dependency drops out of the path.

    Returns None for anything that is not a plain 16-bit WAV, so the caller
    can fall back to letting whisper open the file.
    """
    try:
        with wave.open(str(path), "rb") as w:
            if w.getsampwidth() != 2:
                return None
            channels, rate, frames = w.getnchannels(), w.getframerate(), w.getnframes()
            raw = w.readframes(frames)
    except Exception:
        return None

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != target_rate and audio.size:
        n_out = int(round(len(audio) * target_rate / rate))
        audio = np.interp(np.linspace(0, len(audio) - 1, n_out),
                          np.arange(len(audio)), audio).astype(np.float32)
    return np.ascontiguousarray(audio, dtype=np.float32)


def transcribe_track(model, path: Path, speaker: str,
                     language: Optional[str] = None) -> list[Segment]:
    """
    One track, one speaker label.

    VAD filtering matters more here than usual. Bluetooth dropouts are
    concealed by bridging audio across the gap, and a transcriber handed a
    stretch of that will happily hallucinate words into it. Trimming
    non-speech before decoding removes most of the opportunity.
    """
    if not path.exists():
        return []
    print(f"  {speaker:<8} {path.name} ...", end="", flush=True)
    started = time.time()
    source = decode_audio(path)
    if source is None:
        source = str(path)  # let whisper try, for anything not a plain WAV
    segments, info = model.transcribe(
        source,
        language=language,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        beam_size=5,
        condition_on_previous_text=False,  # stops one bad guess cascading
    )
    out = [Segment(speaker, s.start, s.end, s.text.strip())
           for s in segments if s.text.strip()]
    print(f" {len(out)} segments in {time.time() - started:.0f}s "
          f"({info.language}, {info.duration:.0f}s audio)")
    return out


def load_speakers(folder: Path) -> list[dict]:
    """Named voices from diarize.py, if it has run on this meeting."""
    path = folder / "speakers.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("speakers", [])
    except Exception:
        return []


def attribute(segments: list[Segment], speakers: list[dict],
              fallback: str = "Others") -> list[Segment]:
    """
    Put names to the system track using the diarisation turns.

    Whisper and the diariser disagree about where a turn begins — one is
    following words, the other energy — so the boundaries never line up
    exactly. Each transcript segment is therefore assigned to whichever
    speaker it OVERLAPS most, rather than to whoever happens to own its start
    instant. A segment that straddles a handover goes to whoever held most of
    it, which is almost always the one who said the words.

    Clusters the matcher left unnamed or flagged ambiguous keep their
    placeholder. A transcript that says "Speaker 2" is honest; one that
    attributes someone's words to a colleague is not.
    """
    if not speakers:
        return segments

    turns: list[tuple[float, float, str]] = []
    for sp in speakers:
        label = sp.get("name") or sp.get("label") or fallback
        for turn in sp.get("turns", []):
            turns.append((turn["start"], turn["end"], label))
    if not turns:
        return segments

    for seg in segments:
        if seg.speaker != fallback:
            continue  # the mic track is already known to be you
        overlaps: dict[str, float] = {}
        for start, end, label in turns:
            amount = min(seg.end, end) - max(seg.start, start)
            if amount > 0:
                overlaps[label] = overlaps.get(label, 0.0) + amount
        if overlaps:
            seg.speaker = max(overlaps.items(), key=lambda kv: kv[1])[0]
    return segments


def merge(tracks: Iterable[list[Segment]]) -> list[Segment]:
    """Interleave both speakers on the shared timeline."""
    return sorted((s for t in tracks for s in t), key=lambda s: (s.start, s.speaker))


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------


def to_markdown(segments: list[Segment], title: str, meta: dict,
                summary: Optional[str] = None) -> str:
    lines = [f"# {title}", ""]
    if meta:
        when = meta.get("started_at", "")
        dur = meta.get("duration_seconds")
        bits = [b for b in (when, f"{dur / 60:.0f} min" if dur else "",
                            meta.get("app", "")) if b]
        lines += [" · ".join(bits), ""]
    if summary:
        lines += ["## Summary", "", summary.strip(), "", "---", ""]

    who = sorted({s.speaker for s in segments})
    if len(who) > 2:
        talk: dict[str, float] = {}
        for seg in segments:
            talk[seg.speaker] = talk.get(seg.speaker, 0.0) + (seg.end - seg.start)
        lines += ["## Who spoke", ""]
        total = sum(talk.values()) or 1.0
        for name, seconds in sorted(talk.items(), key=lambda kv: -kv[1]):
            lines.append(f"- **{name}** — {human_duration(seconds)} "
                         f"({seconds / total * 100:.0f}%)")
        lines += [""]

    lines += ["## Transcript", ""]
    last = None
    for seg in segments:
        if seg.speaker != last:
            lines += ["", f"**{seg.speaker}**", ""]
            last = seg.speaker
        lines.append(f"`{seg.stamp}` {seg.text}")
    return "\n".join(lines) + "\n"


def summarise(segments: list[Segment], model: str) -> Optional[str]:
    """
    Summarise with a local Ollama model.

    Transcription needs an audio model; summarising is a text job, which is
    what a local LLM is good at. Nothing leaves the machine either way.
    """
    text = "\n".join(f"[{s.stamp}] {s.speaker}: {s.text}" for s in segments)
    if not text.strip():
        return None
    # Keep the prompt inside a sane context. A long meeting will exceed it;
    # truncating the middle keeps the opening framing and the closing actions,
    # which is where decisions usually live.
    limit = 48000
    if len(text) > limit:
        head, tail = text[: limit // 2], text[-limit // 2:]
        text = head + "\n\n[... middle of the meeting omitted for length ...]\n\n" + tail

    payload = json.dumps({
        "model": model,
        "prompt": SUMMARY_PROMPT.format(transcript=text),
        "stream": False,
        "options": {"temperature": 0.2},
    }).encode()

    print(f"  summarising with {model} ...", end="", flush=True)
    started = time.time()
    try:
        req = urllib.request.Request(OLLAMA_URL, data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as resp:
            body = json.loads(resp.read())
        print(f" done in {time.time() - started:.0f}s")
        return body.get("response", "").strip()
    except urllib.error.URLError as exc:
        print(f" failed ({exc.reason}) — is Ollama running?")
    except Exception as exc:
        print(f" failed ({exc})")
    return None


# --------------------------------------------------------------------------
# driving
# --------------------------------------------------------------------------


def find_tracks(folder: Path) -> tuple[Optional[Path], Optional[Path], Optional[Path], dict]:
    """
    (mic, system, single, metadata).

    A desktop meeting has a mic/system pair. A recording uploaded from the
    phone is one file containing everyone, including you — so there is no
    free attribution and every speaker has to be named by the diariser.
    """
    mic = next(iter(sorted(folder.glob("*-mic.wav"))), None)
    sysa = next(iter(sorted(folder.glob("*-system.wav"))), None)

    single = None
    if mic is None and sysa is None:
        others = sorted(p for p in folder.iterdir()
                        if p.suffix.lower() in AUDIO_SUFFIXES
                        and not p.name.endswith(("-mix.wav", "-mic-raw.wav")))
        single = others[0] if others else None

    meta = {}
    for pattern in ("*-mic.wav.json", "*.json"):
        for sidecar in folder.glob(pattern):
            if sidecar.name in ("speakers.json", "transcript.json") \
                    or sidecar.name.endswith("-shots.json"):
                continue
            try:
                meta = json.loads(sidecar.read_text(encoding="utf-8"))
            except Exception:
                continue
            break
        if meta:
            break
    return mic, sysa, single, meta


def process(folder: Path, model, me: str, others: str,
            summary_model: Optional[str], force: bool,
            language: Optional[str]) -> bool:
    out_md = folder / "transcript.md"
    if out_md.exists() and not force:
        return False

    mic, sysa, single, meta = find_tracks(folder)
    if not mic and not sysa and not single:
        return False

    print(f"\n{folder.name}" + ("  (single track)" if single else ""))
    if single:
        # One file, everyone in it. Nothing is known for free, so every
        # segment starts unattributed and the diariser supplies the names.
        segments = transcribe_track(model, single, others, language)
    else:
        segments = merge([
            transcribe_track(model, mic, me, language) if mic else [],
            transcribe_track(model, sysa, others, language) if sysa else [],
        ])
    if not segments:
        print("  no speech found")
        return False

    speakers = load_speakers(folder)
    if speakers:
        attribute(segments, speakers, others)
        named = sorted({s.speaker for s in segments} - {me, others})
        print(f"  speakers: {', '.join(named) if named else 'none identified'}")
    else:
        print("  no speakers.json — run diarize.py to put names to the "
              "system track")

    summary = summarise(segments, summary_model) if summary_model else None
    title = meta.get("app", folder.name)
    out_md.write_text(to_markdown(segments, folder.name, meta, summary),
                      encoding="utf-8")
    (folder / "transcript.json").write_text(
        json.dumps({"meta": meta, "summary": summary,
                    "speakers": sorted({s.speaker for s in segments}),
                    "segments": [asdict(s) for s in segments]}, indent=2),
        encoding="utf-8")
    print(f"  wrote {out_md.name} ({len(segments)} segments)")
    return True


def is_meeting_folder(folder: Path) -> bool:
    """
    A meeting, or just audio that happens to live in the archive?

    Shared data — enrollment samples, dictionaries — sits alongside the
    recordings, and "any folder with audio in it" swept those in. Meetings
    are named by timestamp; anything under a segment starting with "_" is
    support data.
    """
    if any(part.startswith("_") for part in folder.parts):
        return False
    return folder_date(folder) is not None


def meeting_folders(root: Path) -> list[Path]:
    """Any folder holding audio — a desktop pair or a single phone upload."""
    seen = set()
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        if path.name.endswith(("-mix.wav", "-mic-raw.wav")):
            continue
        if is_meeting_folder(path.parent):
            seen.add(path.parent)
    return sorted(seen)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Transcribe recorded meetings.")
    ap.add_argument("path", help="a meeting folder, or the archive root with --all")
    ap.add_argument("--all", action="store_true", help="process every untranscribed folder")
    ap.add_argument("--model", default="medium",
                    help="whisper size: tiny base small medium large-v3 (default medium)")
    ap.add_argument("--compute", default="int8", help="int8 (default), int8_float16, float32")
    ap.add_argument("--threads", type=int, default=None,
                    help="CPU threads (default: half the cores, to leave "
                         "headroom for recording)")
    ap.add_argument("--me", default="Me", help="label for the microphone track")
    ap.add_argument("--others", default="Others", help="label for the system track")
    ap.add_argument("--language", default=None, help="force a language, e.g. en")
    ap.add_argument("--summarise", nargs="?", const="gpt-oss:20b", default=None,
                    metavar="OLLAMA_MODEL", help="summarise with a local Ollama model")
    ap.add_argument("--since", default=None,
                    help="only meetings on or after this date: 2026-10-01, or "
                         "a span like 7d / 4w / 3m")
    ap.add_argument("--force", action="store_true", help="redo folders already done")
    args = ap.parse_args(argv)

    root = Path(args.path)
    if not root.exists():
        print(f"not found: {root}", file=sys.stderr)
        return 1

    try:
        since = parse_since(args.since)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    folders = meeting_folders(root) if args.all else [root]
    if since and args.all:
        before = len(folders)
        folders = [f for f in folders if after_cutoff(f, since)]
        print(f"since {since:%Y-%m-%d}: {len(folders)} of {before} meeting(s)")
    pending = [f for f in folders
               if args.force or not (f / "transcript.md").exists()]
    if not pending:
        print("nothing to do — all folders already have transcript.md (use --force to redo)")
        return 0

    print(f"{len(pending)} meeting(s) to transcribe")
    model = load_model(args.model, args.compute, args.threads)
    failed: list[str] = []

    done = 0
    for folder in pending:
        try:
            if process(folder, model, args.me, args.others,
                       args.summarise, args.force, args.language):
                done += 1
        except KeyboardInterrupt:
            print("\ninterrupted")
            break
        except Exception as exc:
            failed.append(folder.name)
            print(f"  FAILED: {exc!r}")

    print(f"\n{done} transcript(s) written")
    if failed:
        # Named, because a silent skip in a long batch is indistinguishable
        # from a meeting that had nothing in it.
        print(f"{len(failed)} failed and were skipped: {', '.join(failed[:5])}"
              + (" …" if len(failed) > 5 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
