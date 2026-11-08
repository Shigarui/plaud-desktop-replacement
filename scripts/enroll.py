"""
enroll.py — build a voiceprint library from an existing speaker-profile folder.

Expects the layout you already have:

    MasterFolder/
        MasterIndex.md
        MasterIndex.csv          rules and folder assignment
        Laura/
            laura-profile.md     profile card
            laura-sample.ogg     voice sample
        Mike/
            ...

Each sample is turned into a speaker embedding — a vector that encodes how a
voice sounds rather than what it said — and stored in voiceprints.json. Later,
segments from a meeting's system track can be embedded the same way and matched
by cosine similarity.

A note on why samples rather than embeddings: an embedding only means anything
inside the model that produced it. Vectors from two different speaker models
live in incompatible spaces and cannot be compared, so importing someone else's
embeddings would lock you to their exact model and version. Re-embedding from
the original audio takes seconds per speaker and keeps the library portable.

    python enroll.py "D:/Voice Profiles"              # build the library
    python enroll.py "D:/Voice Profiles" --verify     # check it can tell them apart

Install:
    py -m pip install speechbrain soundfile torch --index-url https://download.pytorch.org/whl/cpu
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

AUDIO_SUFFIXES = (".ogg", ".wav", ".mp3", ".m4a", ".flac", ".opus")
EMBED_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
TARGET_RATE = 16000
LIBRARY_NAME = "voiceprints.json"


@dataclass
class Profile:
    name: str
    folder: Path
    samples: list = field(default_factory=list)
    card: dict = field(default_factory=dict)
    aliases: list = field(default_factory=list)
    embedding: Optional[np.ndarray] = None


# --------------------------------------------------------------------------
# reading the library folder
# --------------------------------------------------------------------------


def parse_card(path: Path) -> dict:
    """
    Pull key/value pairs out of a profile card.

    Deliberately forgiving: YAML front matter, `**Role:** Ops Director`,
    `- name: Laura`, or a plain `Role: Ops Director` all work. These cards are
    written by hand and by other tools, so a strict parser would reject more
    than it read.
    """
    out: dict = {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return out

    for raw in text.splitlines():
        # Strip emphasis BEFORE trimming list markers: lstrip("-*# ") eats the
        # opening ** of "**Role:**" and leaves the closing pair stranded in
        # the value, so "Ops Director" arrives as "** Ops Director".
        line = raw.strip().replace("**", "").replace("__", "")
        line = line.lstrip("-*# ").strip()
        if not line or line.startswith("---"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower().replace(" ", "_")
        value = value.strip()
        if key and value and len(key) < 40 and key not in out:
            out[key] = value
    return out


def parse_index(folder: Path) -> dict:
    """
    Read MasterIndex.csv for aliases and folder rules.

    Column names vary, so match on what a header contains rather than equality.
    Anything unrecognised is kept as-is: a rule we do not understand is still
    worth carrying into the library for a human to read later.
    """
    index: dict = {}
    csv_path = next((p for p in folder.glob("*.csv")
                     if "index" in p.name.lower()), None)
    if csv_path is None:
        return index

    try:
        with csv_path.open(encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
    except Exception as exc:
        print(f"  could not read {csv_path.name}: {exc}")
        return index

    for row in rows:
        clean = {(k or "").strip().lower(): (v or "").strip()
                 for k, v in row.items() if k}
        name = next((clean[k] for k in clean
                     if "name" in k or "speaker" in k or "folder" in k), "")
        if not name:
            continue
        aliases = next((clean[k] for k in clean if "alias" in k or "also" in k), "")
        index[name.strip().lower()] = {
            "aliases": [a.strip() for a in re.split(r"[;,|]", aliases) if a.strip()],
            "rules": {k: v for k, v in clean.items() if v and "alias" not in k},
        }
    print(f"  index: {len(index)} entries from {csv_path.name}")
    return index


def scan(root: Path) -> list[Profile]:
    """One Profile per subfolder that contains at least one audio sample."""
    index = parse_index(root)
    profiles: list[Profile] = []

    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        samples = sorted(p for p in folder.iterdir()
                         if p.suffix.lower() in AUDIO_SUFFIXES)
        if not samples:
            continue

        card: dict = {}
        for md in sorted(folder.glob("*.md")):
            card.update(parse_card(md))

        name = card.get("name") or folder.name
        entry = index.get(folder.name.strip().lower(), {})
        aliases = list(entry.get("aliases", []))
        for key in ("aliases", "alias", "also_known_as"):
            if key in card:
                aliases += [a.strip() for a in re.split(r"[;,|]", card[key]) if a.strip()]
        if entry.get("rules"):
            card.setdefault("_index_rules", entry["rules"])

        profiles.append(Profile(name=name, folder=folder, samples=samples,
                                card=card, aliases=sorted(set(aliases))))
    return profiles


# --------------------------------------------------------------------------
# embedding
# --------------------------------------------------------------------------


def load_encoder():
    """
    Load the speaker-embedding model, quietly.

    SpeechBrain tries to symlink the downloaded model into its working
    folder. Windows refuses without Developer Mode or admin, so it warns on
    every single run and then copies instead — working perfectly while
    burying the output that matters. Asking for copies up front removes the
    cause rather than muting the symptom.
    """
    import warnings

    try:
        import torch  # noqa: F401
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        try:
            from speechbrain.pretrained import EncoderClassifier  # older layout
        except ImportError:
            print("speechbrain and torch are required.\n"
                  "  py -m pip install torch --index-url "
                  "https://download.pytorch.org/whl/cpu\n"
                  "  py -m pip install speechbrain soundfile", file=sys.stderr)
            raise SystemExit(1)

    print(f"loading {EMBED_MODEL} — first run downloads it")
    kwargs = {"source": EMBED_MODEL, "savedir": ".models/ecapa",
              "run_opts": {"device": "cpu"}}

    # Newer speechbrain accepts a fetch strategy; older versions do not know
    # the argument at all, so try it and fall back rather than version-sniff.
    try:
        from speechbrain.utils.fetching import LocalStrategy

        return EncoderClassifier.from_hparams(
            local_strategy=LocalStrategy.COPY, **kwargs)
    except Exception:
        pass

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*symlink.*")
        return EncoderClassifier.from_hparams(**kwargs)


def read_audio(path: Path) -> np.ndarray:
    """Mono float32 at 16 kHz, which is what the encoder expects."""
    import soundfile as sf

    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if rate != TARGET_RATE:
        n_out = int(round(len(mono) * TARGET_RATE / rate))
        mono = np.interp(np.linspace(0, len(mono) - 1, n_out),
                         np.arange(len(mono)), mono).astype(np.float32)
    return mono


def simulate_capture(audio: np.ndarray, highpass_hz: float = 300.0,
                     band_limit_hz: float = 3800.0) -> np.ndarray:
    """
    Make an enrollment clip resemble meeting audio before embedding it.

    This matters more than the model or the threshold. A clean 90-second
    enrollment clip and a Bluetooth meeting track are different kinds of
    audio: the recorder high-passes the mic at 300 Hz, which removes the very
    vocal fundamentals the encoder leans on, and HFP throws away everything
    above roughly 4 kHz before that.

    Measured on a real library: a speaker's own voice scored 0.455 against his
    own clean profile, while two *different* people scored 0.55 against each
    other. No threshold can survive that inversion. Embedding both sides in
    the same domain removes it.
    """
    from numpy.fft import irfft, rfft, rfftfreq

    if audio.size == 0:
        return audio
    spectrum = rfft(audio)
    freqs = rfftfreq(len(audio), 1.0 / TARGET_RATE)
    spectrum[freqs < highpass_hz] = 0.0
    spectrum[freqs > band_limit_hz] = 0.0
    out = irfft(spectrum, len(audio)).astype(np.float32)
    peak = float(np.max(np.abs(out)))
    # Only normalise real signal. Filtering can leave numerical residue a
    # thousand times below anything audible, and scaling that to full level
    # turns silence into a loud artefact the encoder will happily embed.
    if peak < 1e-4:
        return out
    return out / peak * 0.7


def embed(encoder, audio: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        vec = encoder.encode_batch(torch.from_numpy(audio).unsqueeze(0))
    vec = vec.squeeze().cpu().numpy().astype(np.float64)
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def embed_profile(encoder, profile: Profile, min_seconds: float = 3.0,
                  match_capture: bool = False, highpass_hz: float = 300.0) -> bool:
    """
    Average the embeddings of every sample for this speaker.

    Averaging normalised vectors pulls the result toward what is consistent
    across recordings and away from whatever was particular to one of them —
    a cold, a different room, a different headset.
    """
    vectors = []
    for sample in profile.samples:
        try:
            audio = read_audio(sample)
        except Exception as exc:
            print(f"    {sample.name}: unreadable ({exc})")
            continue
        seconds = len(audio) / TARGET_RATE
        if seconds < min_seconds:
            print(f"    {sample.name}: only {seconds:.1f}s — too short, skipped")
            continue
        if match_capture:
            audio = simulate_capture(audio, highpass_hz)
        vectors.append(embed(encoder, audio))
        print(f"    {sample.name}: {seconds:.1f}s"
              + ("  (capture-matched)" if match_capture else ""))

    if not vectors:
        return False
    mean = np.mean(vectors, axis=0)
    norm = np.linalg.norm(mean)
    profile.embedding = mean / norm if norm else mean
    return True


# --------------------------------------------------------------------------
# library
# --------------------------------------------------------------------------


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if not na or not nb:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def save_library(profiles: list[Profile], path: Path, model: str = EMBED_MODEL) -> None:
    payload = {
        # The model is recorded because embeddings are only comparable within
        # the model that made them. Matching against a library built by a
        # different encoder produces confident nonsense.
        "model": model,
        "dimensions": int(len(profiles[0].embedding)) if profiles else 0,
        "speakers": [
            {
                "name": p.name,
                "folder": p.folder.name,
                "aliases": p.aliases,
                "card": p.card,
                "samples": [s.name for s in p.samples],
                "embedding": [float(x) for x in p.embedding],
            }
            for p in profiles if p.embedding is not None
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_library(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for s in data.get("speakers", []):
        s["embedding"] = np.asarray(s["embedding"], dtype=np.float64)
    return data


def identify(vector: np.ndarray, library: dict,
             threshold: float = 0.65, z_threshold: float = 4.0
             ) -> tuple[Optional[str], float]:
    """
    Nearest speaker in the library, or (None, score) if nothing stands out.

    Scored two ways, because the absolute cosine on its own is misleading.

    Enrollment clips are all recorded the same way, so two different people
    compared against each other score higher than a real match compared across
    recordings. Measured here: the worst impostor pair inside the library hit
    0.55, while a speaker's own voice from a meeting matched his own profile at
    0.455. Judged on absolute score the impostors win, and no threshold fixes
    that, because the two numbers are not the same kind of measurement.

    What does survive the channel change is how far the top match stands above
    everyone else. That same 0.455 sat 0.19 clear of the runner-up and about
    five standard deviations above the field. So the decision is made on a
    z-score against the rest of the library — standard practice in speaker
    verification, and the thing that makes cross-channel matching work at all.

    A match needs EITHER a high absolute score (same-domain audio) OR a clear
    statistical lead (cross-domain, which is the normal case here).

    The z bar is 4, and it was set by being wrong rather than by taste. On a
    41-profile library one cluster matched the wrong person at z=3.8 —
    confidently, and only a human listening caught it. Measured against every
    case we have:

        8.6  correct      3.8  WRONG
        6.2  correct
        5.0  correct
        4.4  correct

    Four is the only line that separates them. With dozens of profiles
    something will land three sigma out by chance, so three is not evidence;
    five would have thrown away a correct match at 4.4.
    """
    speakers = library.get("speakers", [])
    if not speakers:
        return None, 0.0

    scores = sorted(((cosine(vector, sp["embedding"]), sp["name"])
                     for sp in speakers), reverse=True)
    best_score, best_name = scores[0]
    if len(scores) < 3:
        return (best_name, best_score) if best_score >= threshold else (None, best_score)

    rest = np.array([s for s, _ in scores[1:]])
    spread = float(rest.std())
    z = (best_score - float(rest.mean())) / spread if spread > 1e-9 else 0.0

    if best_score >= threshold or z >= z_threshold:
        return best_name, best_score
    return None, best_score


def classify(vector: np.ndarray, library: dict, threshold: float = 0.65,
             z_threshold: float = 4.0, min_margin: float = 0.10,
             max_candidates: int = 3) -> dict:
    """
    Decide between three outcomes rather than two.

        named      one person stands clearly above everyone else
        ambiguous  somebody matches, but a rival is close enough to doubt it
        unknown    nothing is close; probably not in the library at all

    The middle state exists because of a real failure. A speaker who was not
    enrolled at all was confidently named as the nearest voice present — the
    two people genuinely sound alike, so the score was perfectly reasonable
    for a nearest neighbour and completely wrong as an identification.

    Raising the acceptance bar does not fix that: it throws away correct
    matches at the same level. What distinguishes the two cases is the gap to
    the runner-up. A real match wins outright; a nearest-neighbour-by-default
    sits in a huddle of similar scores. When the gap is thin, the honest answer
    is to name both and let a human spend five seconds listening.
    """
    detail = score_detail(vector, library)
    if not detail:
        return {"status": "unknown", "name": None, "score": 0.0, "z": 0.0,
                "margin": 0.0, "candidates": []}

    best, z, margin = detail["best"], detail["z"], detail["margin"]
    candidates = [{"name": n, "score": round(sc, 4)}
                  for sc, n in detail["ranked"][:max_candidates]]
    qualifies = best >= threshold or z >= z_threshold

    if not qualifies:
        return {"status": "unknown", "name": None, "score": best, "z": z,
                "margin": margin, "candidates": candidates}
    if margin < min_margin:
        return {"status": "ambiguous", "name": None, "score": best, "z": z,
                "margin": margin, "candidates": candidates}
    return {"status": "named", "name": detail["ranked"][0][1], "score": best,
            "z": z, "margin": margin, "candidates": candidates}


def score_detail(vector: np.ndarray, library: dict) -> dict:
    """Everything behind an identify() decision, for inspection."""
    scores = sorted(((cosine(vector, sp["embedding"]), sp["name"])
                     for sp in library.get("speakers", [])), reverse=True)
    if not scores:
        return {}
    rest = np.array([s for s, _ in scores[1:]]) if len(scores) > 1 else np.array([0.0])
    spread = float(rest.std())
    return {
        "ranked": scores,
        "best": scores[0][0],
        "margin": scores[0][0] - scores[1][0] if len(scores) > 1 else scores[0][0],
        "field_mean": float(rest.mean()),
        "field_std": spread,
        "z": (scores[0][0] - float(rest.mean())) / spread if spread > 1e-9 else 0.0,
    }


def verify(profiles: list[Profile]) -> None:
    """
    Cross-compare every profile.

    This is the step that tells you whether the library is usable. Two people
    scoring 0.8 against each other means the samples cannot distinguish them,
    and no threshold will fix it — the fix is better or longer samples.
    """
    usable = [p for p in profiles if p.embedding is not None]
    if len(usable) < 2:
        print("\nneed at least two profiles to compare")
        return

    print(f"\n{'':<18}" + "".join(f"{p.name[:8]:>9}" for p in usable))
    for a in usable:
        row = f"{a.name[:17]:<18}"
        for b in usable:
            score = 1.0 if a is b else cosine(a.embedding, b.embedding)
            row += f"{score:>9.2f}"
        print(row)

    pairs = [(cosine(a.embedding, b.embedding), a.name, b.name)
             for i, a in enumerate(usable) for b in usable[i + 1:]]
    pairs.sort(reverse=True)
    print()
    for score, a, b in pairs[:5]:
        verdict = ("TOO SIMILAR — these will be confused" if score > 0.7
                   else "close, watch this pair" if score > 0.5
                   else "ok")
        print(f"  {a} vs {b}: {score:.2f}  {verdict}")

    worst = pairs[0][0] if pairs else 0.0
    scores = [p[0] for p in pairs]
    suggested = round(min(0.80, max(0.55, worst + 0.10)), 2)
    print(f"\n  {len(usable)} speakers, {len(pairs)} pairs")
    print(f"  worst pair {worst:.2f}   median {float(np.median(scores)):.2f}")
    print(f"\n  SUGGESTED THRESHOLD: {suggested}")
    print(f"  Above your worst pair ({worst:.2f}) so two different people are")
    print("  not confused, and low enough that a genuine match still lands.")
    if worst > 0.7:
        print("\n  WARNING: that worst pair is too close to separate reliably.")
        print("  No threshold fixes this — those two need longer or cleaner samples.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build a voiceprint library.")
    ap.add_argument("folder", help="the master folder holding one subfolder per speaker")
    ap.add_argument("--out", default=None, help=f"output path (default <folder>/{LIBRARY_NAME})")
    ap.add_argument("--verify", action="store_true",
                    help="cross-compare profiles and report how separable they are")
    ap.add_argument("--min-seconds", type=float, default=3.0,
                    help="ignore samples shorter than this (default 3)")
    ap.add_argument("--identify", metavar="AUDIO",
                    help="match one audio file against an existing library "
                         "instead of enrolling — a quick end-to-end check")
    ap.add_argument("--threshold", type=float, default=0.65,
                    help="match threshold for --identify (default 0.65)")
    ap.add_argument("--z", type=float, default=4.0,
                    help="standard deviations above the field needed to accept "
                         "a match on its lead alone (default 4)")
    ap.add_argument("--match-capture", action="store_true",
                    help="band-limit enrollment clips toward meeting audio. "
                         "Measured as HARMFUL on real data — it removes the "
                         "detail the encoder uses. Kept for experimentation.")
    ap.add_argument("--highpass", type=float, default=300.0,
                    help="high-pass used by --match-capture; set it to the "
                         "recorder's highpass_hz (default 300)")
    args = ap.parse_args(argv)

    root = Path(args.folder)
    if not root.is_dir():
        print(f"not a folder: {root}", file=sys.stderr)
        return 1

    if args.identify:
        lib_path = Path(args.out) if args.out else root / LIBRARY_NAME
        if not lib_path.exists():
            print(f"no library at {lib_path} — enroll first", file=sys.stderr)
            return 1
        library = load_library(lib_path)
        encoder = load_encoder()
        audio = read_audio(Path(args.identify))
        print(f"\n{Path(args.identify).name}: {len(audio) / TARGET_RATE:.1f}s")
        vector = embed(encoder, audio)
        detail = score_detail(vector, library)
        name, score = identify(vector, library, args.threshold, args.z)
        print("\n  closest matches:")
        for sc, nm in detail["ranked"][:5]:
            print(f"    {nm:<24} {sc:.3f}")
        print(f"\n  field: mean {detail['field_mean']:.3f} "
              f"sd {detail['field_std']:.3f}")
        print(f"  margin over runner-up: {detail['margin']:.3f}")
        print(f"  z-score above the field: {detail['z']:.2f}  "
              f"(need {args.z} to accept on lead alone)")
        print(f"\n  verdict: {name or 'NO MATCH'}")
        return 0

    print(f"scanning {root}")
    profiles = scan(root)
    if not profiles:
        print("no speaker folders with audio samples found")
        return 1
    print(f"  {len(profiles)} speaker folder(s) with samples")

    encoder = load_encoder()
    ok = []
    for profile in profiles:
        print(f"\n{profile.name}" + (f"  (aliases: {', '.join(profile.aliases)})"
                                     if profile.aliases else ""))
        if embed_profile(encoder, profile, args.min_seconds,
                         args.match_capture, args.highpass):
            ok.append(profile)
        else:
            print("    no usable samples — skipped")

    if not ok:
        print("\nnothing enrolled")
        return 1

    out = Path(args.out) if args.out else root / LIBRARY_NAME
    save_library(ok, out)
    print(f"\nenrolled {len(ok)} speaker(s) -> {out}")

    if args.verify:
        verify(ok)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
