"""
diarize.py — work out who spoke when on the system track, and name them.

The approach, and why it is shaped this way:

A single meeting turn is often two or three seconds, and a speaker embedding
taken from that is noisy. Matching each turn against the library separately
would mean the same person identified correctly in one sentence and wrongly in
the next, which reads worse than no names at all.

So turns are clustered first, and only the CLUSTER is matched. Six turns of
four seconds each become one centroid built from twenty-four seconds of that
voice — back in the territory where identification measurably works. Averaging
also cancels the things that ruin individual clips: one turn off-mic, one with
crosstalk, one containing a cough. They pull in different directions.

Clusters that match nobody stay as "Speaker 2". A wrong name is worse than an
honest unknown, because a wrong name gets believed.

    python diarize.py "G:/.../20261013-140530-ms-teams.exe/" \
        --library ".\\BIGGBY_Speaker_Profiles\\voiceprints.json"

    python diarize.py --all "G:/My Drive/Meeting Recordings" --library ...
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import wave
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from enroll import (  # noqa: E402
    TARGET_RATE,
    classify,
    cosine,
    embed,
    load_encoder,
    load_library,
    score_detail,
)

# Anything the phone might upload. The desktop writes WAV; the recorder app
# uploads whatever the device produced, which is usually Opus or MP3.
AUDIO_SUFFIXES = (".wav", ".mp3", ".ogg", ".opus", ".flac", ".m4a")

MIN_SEGMENT_SECONDS = 1.5
MIN_CLUSTER_SECONDS = 4.0
CLUSTER_DISTANCE = 0.65  # cosine distance; lower splits more eagerly

# Single-track recordings fragment far more than two-track ones. Everyone is
# on one microphone at varying distances, the file is lossily compressed, and
# the same person genuinely sounds different across a long recording — so the
# first pass splits one speaker into several clusters. A looser threshold and
# a merge pass over the centroids put them back together.
SINGLE_TRACK_DISTANCE = 0.78

# Both centroids come from the same recording, so this is a same-channel
# comparison — the kind that scores HIGH. On a real 41-voice library the
# closest two different people reached 0.55 under exactly those conditions,
# so merging below that would start combining people. One speaker's
# fragments sit comfortably above it. 0.68 leaves room on both sides.
MERGE_SIMILARITY = 0.68


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


@dataclass
class Segment:
    start: float
    end: float
    embedding: Optional[np.ndarray] = None
    cluster: int = -1

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Speaker:
    cluster: int
    label: str
    name: Optional[str]
    confidence: float
    z: float
    seconds: float
    status: str = "unknown"  # named | ambiguous | unknown | too-short
    margin: float = 0.0
    candidates: list = field(default_factory=list)
    segments: list = field(default_factory=list)


# --------------------------------------------------------------------------
# audio
# --------------------------------------------------------------------------


def read_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    """
    Mono float32 at the embedding rate, from WAV or anything soundfile reads.

    The desktop recorder writes WAV, but recordings uploaded from the phone
    arrive in whatever the hardware produced — Opus or MP3 — and the stdlib
    wave module cannot open those. soundfile is already a dependency for
    enrollment, so compressed formats cost nothing extra here.
    """
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as w:
                ch, rate, n = w.getnchannels(), w.getframerate(), w.getnframes()
                raw = w.readframes(n)
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            mono = data.reshape(-1, ch).mean(axis=1) if ch > 1 else data
            return _to_target_rate(mono, rate), TARGET_RATE
        except Exception:
            pass  # odd bit depth or a mislabelled file — let soundfile try

    import soundfile as sf

    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    return _to_target_rate(data.mean(axis=1), rate), TARGET_RATE


def _to_target_rate(mono: np.ndarray, rate: int) -> np.ndarray:
    if rate == TARGET_RATE or mono.size == 0:
        return mono.astype(np.float32)
    n_out = int(round(len(mono) * TARGET_RATE / rate))
    return np.interp(np.linspace(0, len(mono) - 1, n_out),
                     np.arange(len(mono)), mono).astype(np.float32)


def is_meeting_folder(folder: Path) -> bool:
    """
    Does this folder hold a meeting, or just audio that lives nearby?

    The archive also carries shared data — enrollment samples, dictionaries —
    and scanning for "any audio file" swept those in: a folder of voice
    profiles was diarised as a meeting and confidently matched to the person
    whose samples it held. Meeting folders are named by timestamp, and
    anything under a path segment starting with "_" is support data.
    """
    if any(part.startswith("_") for part in folder.parts):
        return False
    return folder_date(folder) is not None


def find_audio(folder: Path) -> tuple[Optional[Path], bool]:
    """
    The track to analyse, and whether it is a single-track recording.

    A desktop meeting has a system track holding everyone but you. A phone
    upload is one file holding everyone INCLUDING you, so every cluster has
    to be matched — your own voice is just another speaker in it.
    """
    system = next(iter(sorted(folder.glob("*-system.wav"))), None)
    if system is not None:
        return system, False

    others = sorted(p for p in folder.iterdir()
                    if p.suffix.lower() in AUDIO_SUFFIXES
                    and not p.name.endswith(("-mic.wav", "-mix.wav",
                                             "-mic-raw.wav")))
    return (others[0], True) if others else (None, False)


def find_speech(audio: np.ndarray, rate: int, frame_ms: int = 30,
                min_silence_ms: int = 400) -> list[tuple[float, float]]:
    """
    Energy-based speech segmentation.

    Deliberately simple. The threshold adapts to the recording rather than
    being fixed, because these tracks vary enormously in level — some are
    normalised to -3 dBFS, some are raw Bluetooth at -30. An absolute
    threshold would find everything in one and nothing in the other.

    Dropouts concealed by the recorder read as speech-like energy, so short
    islands are discarded below MIN_SEGMENT_SECONDS.
    """
    frame = max(1, int(rate * frame_ms / 1000))
    usable = audio[: len(audio) // frame * frame]
    if usable.size == 0:
        return []
    energy = np.sqrt(np.mean(usable.reshape(-1, frame) ** 2, axis=1))

    floor = float(np.percentile(energy, 20))
    ceiling = float(np.percentile(energy, 95))
    if ceiling <= floor * 1.5:
        return []  # no dynamic range: silence, or a constant tone
    threshold = floor + (ceiling - floor) * 0.15
    voiced = energy > threshold

    gap_frames = max(1, int(min_silence_ms / frame_ms))
    segments: list[tuple[float, float]] = []
    start = None
    silence = 0
    for i, is_voiced in enumerate(voiced):
        if is_voiced:
            if start is None:
                start = i
            silence = 0
        elif start is not None:
            silence += 1
            if silence >= gap_frames:
                segments.append((start * frame / rate, (i - silence) * frame / rate))
                start = None
    if start is not None:
        segments.append((start * frame / rate, len(voiced) * frame / rate))

    return [(a, b) for a, b in segments if b - a >= MIN_SEGMENT_SECONDS]


# --------------------------------------------------------------------------
# clustering
# --------------------------------------------------------------------------


def cluster(segments: list[Segment], distance: float = CLUSTER_DISTANCE) -> int:
    """
    Agglomerative clustering on cosine distance, average linkage.

    The number of speakers is discovered rather than supplied — you do not
    know in advance who joined. One person can still land in two clusters if
    their audio changed mid-meeting (they moved, or switched headset); both
    clusters then match the same name and are merged afterwards, so the split
    costs nothing.
    """
    usable = [s for s in segments if s.embedding is not None]
    if not usable:
        return 0
    if len(usable) == 1:
        usable[0].cluster = 0
        return 1

    groups = [[i] for i in range(len(usable))]
    vectors = [s.embedding for s in usable]

    while len(groups) > 1:
        best = (None, None, -1.0)
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                sim = float(np.mean([cosine(vectors[a], vectors[b])
                                     for a in groups[i] for b in groups[j]]))
                if sim > best[2]:
                    best = (i, j, sim)
        i, j, sim = best
        if 1.0 - sim > distance:
            break
        groups[i] = groups[i] + groups[j]
        groups.pop(j)

    for cid, members in enumerate(groups):
        for idx in members:
            usable[idx].cluster = cid
    return len(groups)


def merge_similar(segments: list[Segment], n_clusters: int,
                  similarity: float = MERGE_SIMILARITY) -> int:
    """
    Join clusters whose centroids are close, and renumber.

    Clustering decides on individual segments, where a two-second turn is a
    poor sample of a voice. A centroid built from a cluster is a much better
    one, so comparing centroids afterwards catches splits the first pass could
    not see — which is most of why a single-track recording comes back with
    twenty speakers instead of five.
    """
    if n_clusters < 2:
        return n_clusters

    members = {cid: [s for s in segments if s.cluster == cid]
               for cid in range(n_clusters)}
    members = {cid: m for cid, m in members.items() if m}
    if len(members) < 2:
        return len(members)

    centroids = {cid: centroid(m) for cid, m in members.items()}
    # Largest first: small fragments should join a well-sampled voice rather
    # than the other way round.
    order = sorted(members, key=lambda c: -sum(s.duration for s in members[c]))

    groups: list[list[int]] = []
    for cid in order:
        for group in groups:
            if max(cosine(centroids[cid], centroids[other])
                   for other in group) >= similarity:
                group.append(cid)
                break
        else:
            groups.append([cid])

    for new_id, group in enumerate(groups):
        for cid in group:
            for seg in members[cid]:
                seg.cluster = new_id
    return len(groups)


def centroid(segments: list[Segment]) -> np.ndarray:
    """
    Average the embeddings of a cluster, weighted by duration.

    Weighting matters: a ten-second turn is a far better sample of a voice
    than a two-second one, and should count accordingly.
    """
    vecs = np.array([s.embedding for s in segments])
    weights = np.array([max(s.duration, 0.1) for s in segments])
    mean = np.average(vecs, axis=0, weights=weights)
    norm = np.linalg.norm(mean)
    return mean / norm if norm else mean


# --------------------------------------------------------------------------
# naming
# --------------------------------------------------------------------------


def name_clusters(segments: list[Segment], n_clusters: int, library: dict,
                  z_threshold: float = 4.0, absolute: float = 0.65,
                  min_seconds: float = MIN_CLUSTER_SECONDS,
                  min_margin: float = 0.10) -> list[Speaker]:
    speakers: list[Speaker] = []
    for cid in range(n_clusters):
        members = [s for s in segments if s.cluster == cid]
        if not members:
            continue
        seconds = sum(s.duration for s in members)

        verdict = {"status": "too-short", "name": None, "score": 0.0,
                   "z": 0.0, "margin": 0.0, "candidates": []}
        if seconds >= min_seconds:
            verdict = classify(centroid(members), library, absolute,
                               z_threshold, min_margin)
        speakers.append(Speaker(
            cluster=cid, label=verdict["name"] or f"Speaker {cid + 1}",
            name=verdict["name"], confidence=verdict["score"], z=verdict["z"],
            status=verdict["status"], margin=verdict["margin"],
            candidates=verdict["candidates"], seconds=seconds, segments=members))

    # One person split across two clusters matches the same profile twice.
    merged: dict[str, Speaker] = {}
    out: list[Speaker] = []
    for sp in speakers:
        if sp.name and sp.name in merged:
            first = merged[sp.name]
            first.segments += sp.segments
            first.seconds += sp.seconds
            continue
        if sp.name:
            merged[sp.name] = sp
        out.append(sp)
    return sorted(out, key=lambda s: -s.seconds)


# --------------------------------------------------------------------------
# driving
# --------------------------------------------------------------------------


def process(folder: Path, encoder, library: dict, args) -> bool:
    system, single = find_audio(folder)
    if system is None:
        return False
    out_path = folder / "speakers.json"
    if out_path.exists() and not args.force:
        return False

    print(f"\n{folder.name}" + ("  (single track)" if single else ""))
    try:
        audio, rate = read_audio_mono(system)
    except Exception as exc:
        # A recording still being written, a truncated upload, a zero-byte
        # file from a crashed session. Any of those used to abort the whole
        # run and skip every meeting after it, which is a far worse outcome
        # than one meeting going unprocessed.
        size = system.stat().st_size if system.exists() else 0
        if size < 1024:
            print(f"  skipped: {system.name} is {size} bytes — "
                  "still recording, or the capture failed")
        else:
            print(f"  skipped: could not read {system.name} ({exc})")
        return False
    spans = find_speech(audio, rate)
    print(f"  {len(audio) / rate:.0f}s audio, {len(spans)} speech segments")
    if not spans:
        print("  nothing to cluster")
        return False

    segments = [Segment(a, b) for a, b in spans]
    for seg in segments:
        clip = audio[int(seg.start * rate):int(seg.end * rate)]
        if clip.size:
            seg.embedding = embed(encoder, clip)

    # A single track needs a looser hand: see SINGLE_TRACK_DISTANCE.
    distance = args.distance
    if single and distance == CLUSTER_DISTANCE:
        distance = SINGLE_TRACK_DISTANCE
    n = cluster(segments, distance)
    merged = merge_similar(segments, n, args.merge)
    if merged != n:
        print(f"  {n} clusters merged into {merged} voice(s)")
        n = merged
    else:
        print(f"  {n} distinct voice(s)")

    speakers = name_clusters(segments, n, library, args.z, args.threshold,
                             min_margin=args.margin)
    for sp in speakers:
        if sp.status == "named":
            print(f"    {sp.label:<24} {sp.seconds:5.0f}s  "
                  f"score {sp.confidence:.3f}  z {sp.z:.1f}  margin {sp.margin:.3f}")
        elif sp.status == "ambiguous":
            rivals = " or ".join(c["name"] for c in sp.candidates[:2])
            print(f"    {sp.label:<24} {sp.seconds:5.0f}s  AMBIGUOUS: {rivals}"
                  f"  (margin {sp.margin:.3f}) — review")
        elif sp.status == "too-short":
            print(f"    {sp.label:<24} {sp.seconds:5.0f}s  too little audio")
        else:
            print(f"    {sp.label:<24} {sp.seconds:5.0f}s  "
                  f"no match (best {sp.confidence:.3f}, z {sp.z:.1f})")

    # Your own voice belongs on the mic track. Finding it here means it leaked
    # into the room — speakers instead of headphones — and both tracks now
    # carry you, which ruins the separation the whole design depends on.
    # On a two-track recording your own voice belongs on the mic track only;
    # finding it here means it leaked through the speakers. On a single-track
    # phone recording it is expected and not worth warning about.
    if args.me and not single and any(sp.name == args.me for sp in speakers):
        print(f"\n  WARNING: {args.me} appears on the SYSTEM track.")
        print("  Your voice is being picked up through the speakers. Use")
        print("  headphones, or the two tracks stop being separable.")

    out_path.write_text(json.dumps({
        "clusters": n,
        "single_track": single,
        "speakers": [
            {"label": sp.label, "name": sp.name, "confidence": round(sp.confidence, 4),
             "z": round(sp.z, 2), "seconds": round(sp.seconds, 1),
             "status": sp.status, "margin": round(sp.margin, 4),
             "candidates": sp.candidates,
             # The centroid is kept so unidentified voices can be compared
             # ACROSS meetings later. Without it every unknown is an island,
             # and the same person turns up as a different stranger each week.
             "embedding": [round(float(x), 5) for x in centroid(sp.segments)],
             "turns": [{"start": round(s.start, 2), "end": round(s.end, 2)}
                       for s in sorted(sp.segments, key=lambda x: x.start)]}
            for sp in speakers
        ],
    }, indent=2), encoding="utf-8")
    print(f"  wrote {out_path.name}")
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Identify speakers on the system track.")
    ap.add_argument("path", help="a meeting folder, or the archive root with --all")
    ap.add_argument("--library", required=True, help="voiceprints.json from enroll.py")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--distance", type=float, default=CLUSTER_DISTANCE,
                    help=f"cluster split distance (default {CLUSTER_DISTANCE}, "
                         f"{SINGLE_TRACK_DISTANCE} for single-track recordings)")
    ap.add_argument("--merge", type=float, default=MERGE_SIMILARITY,
                    help=f"merge clusters whose centroids score above this "
                         f"(default {MERGE_SIMILARITY}); raise it if two people "
                         f"get combined")
    ap.add_argument("--z", type=float, default=4.0, help="z-score to accept a name")
    ap.add_argument("--threshold", type=float, default=0.65, help="absolute score to accept")
    ap.add_argument("--margin", type=float, default=0.10,
                    help="minimum gap to the runner-up; below this the cluster "
                         "is flagged ambiguous rather than named (default 0.10)")
    ap.add_argument("--me", default=None,
                    help="your own profile name — warns if it shows up on the "
                         "system track, which means your voice is leaking")
    ap.add_argument("--since", default=None,
                    help="only meetings on or after this date: 2026-10-01, or "
                         "a span like 7d / 4w / 3m")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.path)
    lib_path = Path(args.library)
    if not lib_path.exists():
        print(f"no library at {lib_path} — run enroll.py first", file=sys.stderr)
        return 1
    library = load_library(lib_path)
    print(f"library: {len(library['speakers'])} speaker(s)")

    try:
        since = parse_since(args.since)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.all:
        folders = sorted({p.parent for p in root.rglob("*")
                          if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES
                          and not p.name.endswith(("-mic.wav", "-mix.wav",
                                                   "-mic-raw.wav"))})
        before = len(folders)
        folders = [f for f in folders if is_meeting_folder(f)]
        if len(folders) != before:
            print(f"ignoring {before - len(folders)} folder(s) that are not meetings")
        if since:
            before = len(folders)
            folders = [f for f in folders if after_cutoff(f, since)]
            print(f"since {since:%Y-%m-%d}: {len(folders)} of {before} meeting(s)")
    else:
        folders = [root]
    pending = [f for f in folders if args.force or not (f / "speakers.json").exists()]
    if not pending:
        print("nothing to do (use --force to redo)")
        return 0

    encoder = load_encoder()
    done, failed = 0, []
    for folder in pending:
        try:
            if process(folder, encoder, library, args):
                done += 1
        except KeyboardInterrupt:
            print("\ninterrupted")
            break
        except Exception as exc:
            failed.append(folder.name)
            print(f"  FAILED: {exc}")

    print(f"\n{done} meeting(s) processed")
    if failed:
        print(f"{len(failed)} failed and were skipped: {', '.join(failed[:5])}"
              + (" …" if len(failed) > 5 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
