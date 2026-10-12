"""
realign.py — fix the timing offset on recordings made before alignment existed.

Why they are out of sync: the system stream buffers up to 30 seconds of
pre-roll while a lazily-opened mic buffers only a second or two. Both files
begin at their own zero, so the system track starts earlier in real time and
the conversation plays back wrong — you ask something six seconds in and the
answer lands half a minute later.

Newer recordings are aligned automatically. This repairs the older ones.

    # 1. write candidate mixes and find the right offset by ear
    python realign.py --probe "G:/My Drive/.../20260917-150534-…-mic.wav"

    # 2. apply it (positive = mic starts later, which is the usual case)
    python realign.py --offset 28.5 "…-mic.wav"

There is deliberately no automatic detection. Two attempts at estimating the
offset from the audio both failed, for the same underlying reason: a
conversation contains no audio common to both tracks. Correlating the loudness
envelopes locks onto the turn-taking rhythm instead and reported -3s for a
synthetic pair built with a known zero offset — confidently wrong, which is
worse than no answer. Bleed from the speakers into the mic would work, but
closed earbuds produce none. Your ears settle it in under a minute.

Pass the -mic.wav path; the matching -system.wav is found beside it. Originals
are copied to *-unaligned.wav before anything is written.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import wave
from pathlib import Path

import numpy as np


def read_wav(path: Path) -> tuple[np.ndarray, int, int]:
    with wave.open(str(path), "rb") as w:
        ch, rate, n = w.getnchannels(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    d = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return (d.reshape(-1, ch) if ch > 1 else d.reshape(-1, 1)), rate, ch


def write_wav(path: Path, data: np.ndarray, rate: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(data.shape[1])
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(data, -1, 1) * 32767).astype(np.int16).tobytes())


def probe(mic_path: Path, sys_path: Path, out_dir: Path, offsets: list[float]) -> None:
    """Write candidate mixes at several offsets so you can pick by ear."""
    mic, mrate, _ = read_wav(mic_path)
    sysa, srate, _ = read_wav(sys_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    for off in offsets:
        m, s = mic.copy(), sysa.copy()
        if off > 0:
            m = np.vstack([np.zeros((int(off * mrate), m.shape[1]), np.float32), m])
        elif off < 0:
            s = np.vstack([np.zeros((int(-off * srate), s.shape[1]), np.float32), s])
        write_wav(out_dir / f"offset_{off:+06.1f}s.wav", mix(m, mrate, s, srate), max(mrate, srate))
        print(f"  offset_{off:+06.1f}s.wav")


def resample(data: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst or data.size == 0:
        return data
    n_out = int(round(data.shape[0] * dst / src))
    x_old = np.arange(data.shape[0], dtype=np.float64)
    x_new = np.linspace(0, data.shape[0] - 1, n_out, dtype=np.float64)
    return np.stack([np.interp(x_new, x_old, data[:, c])
                     for c in range(data.shape[1])], axis=1).astype(np.float32)


def mix(mic: np.ndarray, mrate: int, sysa: np.ndarray, srate: int) -> np.ndarray:
    rate = max(mrate, srate)
    m = resample(mic, mrate, rate)
    s = resample(sysa, srate, rate)
    if m.shape[1] == 1:
        m = np.repeat(m, 2, axis=1)
    if s.shape[1] == 1:
        s = np.repeat(s, 2, axis=1)
    n = max(m.shape[0], s.shape[0])
    m = np.pad(m, ((0, n - m.shape[0]), (0, 0)))
    s = np.pad(s, ((0, n - s.shape[0]), (0, 0)))
    out = m * 0.7 + s * 0.7
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    return out / peak if peak > 1.0 else out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Realign a mic/system recording pair.")
    ap.add_argument("mic", help="path to the -mic.wav file")
    ap.add_argument("--offset", type=float,
                    help="seconds to shift; positive = mic starts later")
    ap.add_argument("--probe", action="store_true",
                    help="write candidate mixes at several offsets to choose by ear")
    args = ap.parse_args(argv)

    mic_path = Path(args.mic)
    if not mic_path.exists():
        print(f"not found: {mic_path}", file=sys.stderr)
        return 1
    sys_path = Path(str(mic_path).replace("-mic.wav", "-system.wav"))
    if not sys_path.exists():
        print(f"no matching system track: {sys_path}", file=sys.stderr)
        return 1

    mic, mrate, _ = read_wav(mic_path)
    sysa, srate, _ = read_wav(sys_path)
    print(f"mic    : {mic.shape[0]/mrate:7.1f}s @ {mrate} Hz")
    print(f"system : {sysa.shape[0]/srate:7.1f}s @ {srate} Hz")

    if args.probe:
        out = mic_path.parent / (mic_path.stem.replace("-mic", "") + "-probe")
        print(f"\nwriting candidates to {out}")
        probe(mic_path, sys_path, out, [0, 5, 10, 15, 20, 25, 28.5, 30, 35])
        print("\nPlay each. The one where replies follow questions naturally is your offset.")
        print("Then rerun with:  --offset <that number>")
        return 0

    offset = args.offset
    if offset is None:
        print("\nRun with --probe first to find the offset by ear, then --offset <n>.")
        return 1

    for p in (mic_path, sys_path):
        backup = p.with_name(p.stem + "-unaligned" + p.suffix)
        if not backup.exists():
            shutil.copy2(p, backup)
    print(f"originals kept as *-unaligned.wav")

    if offset > 0:
        mic = np.vstack([np.zeros((int(offset * mrate), mic.shape[1]), np.float32), mic])
        write_wav(mic_path, mic, mrate)
        print(f"padded mic by {offset:.2f}s")
    elif offset < 0:
        sysa = np.vstack([np.zeros((int(-offset * srate), sysa.shape[1]), np.float32), sysa])
        write_wav(sys_path, sysa, srate)
        print(f"padded system by {-offset:.2f}s")

    mix_path = Path(str(mic_path).replace("-mic.wav", "-mix.wav"))
    write_wav(mix_path, mix(mic, mrate, sysa, srate), max(mrate, srate))
    print(f"rebuilt {mix_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
