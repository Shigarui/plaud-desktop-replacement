"""Diarization tests — segmentation, clustering and naming. No model needed."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from diarize import (
    MIN_SEGMENT_SECONDS,
    Segment,
    centroid,
    cluster,
    find_speech,
    name_clusters,
)

RATE = 16000


def speech(seconds, amp=0.3, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(RATE * seconds)) * amp).astype(np.float32)


def silence(seconds):
    return np.zeros(int(RATE * seconds), dtype=np.float32)


def unit(vec):
    v = np.asarray(vec, dtype=float)
    return v / np.linalg.norm(v)


# --- segmentation ----------------------------------------------------------


def test_find_speech_splits_on_silence():
    audio = np.concatenate([silence(1), speech(3), silence(2), speech(4), silence(1)])
    spans = find_speech(audio, RATE)
    assert len(spans) == 2
    assert spans[0][0] == pytest.approx(1.0, abs=0.3)
    assert spans[1][1] - spans[1][0] == pytest.approx(4.0, abs=0.5)


def test_find_speech_discards_blips():
    """
    Concealed dropouts read as speech-like energy. Anything shorter than the
    minimum is not a usable sample of a voice.
    """
    audio = np.concatenate([silence(1), speech(0.4), silence(1), speech(3)])
    spans = find_speech(audio, RATE)
    assert len(spans) == 1
    assert spans[0][1] - spans[0][0] >= MIN_SEGMENT_SECONDS


def test_find_speech_adapts_to_quiet_recordings():
    """A raw Bluetooth track at -30 dBFS must segment like a normalised one."""
    loud = np.concatenate([silence(1), speech(3, amp=0.5), silence(1)])
    quiet = np.concatenate([silence(1), speech(3, amp=0.01), silence(1)])
    assert len(find_speech(loud, RATE)) == len(find_speech(quiet, RATE)) == 1


def test_find_speech_returns_nothing_for_silence():
    assert find_speech(silence(10), RATE) == []


def test_find_speech_handles_empty_audio():
    assert find_speech(np.zeros(0, dtype=np.float32), RATE) == []


# --- clustering ------------------------------------------------------------


def _segs(vectors, duration=3.0):
    out = []
    for i, v in enumerate(vectors):
        s = Segment(start=i * 10.0, end=i * 10.0 + duration)
        s.embedding = unit(v)
        out.append(s)
    return out


def test_cluster_separates_two_distinct_voices():
    segs = _segs([[1, 0, 0], [0.98, 0.1, 0], [0, 1, 0], [0.05, 0.99, 0]])
    assert cluster(segs) == 2
    assert segs[0].cluster == segs[1].cluster
    assert segs[2].cluster == segs[3].cluster
    assert segs[0].cluster != segs[2].cluster


def test_cluster_keeps_one_voice_together():
    segs = _segs([[1, 0, 0], [0.99, 0.05, 0], [0.97, 0.1, 0]])
    assert cluster(segs) == 1


def test_cluster_discovers_the_count_without_being_told():
    segs = _segs([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.99, 0.02, 0]])
    assert cluster(segs) == 3


def test_cluster_handles_a_single_segment():
    segs = _segs([[1, 0, 0]])
    assert cluster(segs) == 1 and segs[0].cluster == 0


def test_cluster_ignores_segments_without_embeddings():
    segs = _segs([[1, 0, 0], [0, 1, 0]])
    segs.append(Segment(start=99, end=102))  # no embedding
    assert cluster(segs) == 2


# --- centroids -------------------------------------------------------------


def test_centroid_is_weighted_by_duration():
    """
    A ten-second turn samples a voice far better than a two-second one, so it
    must count for more when the cluster is averaged.
    """
    long = Segment(0, 10)
    long.embedding = unit([1, 0, 0])
    short = Segment(20, 22)
    short.embedding = unit([0, 1, 0])
    c = centroid([long, short])
    assert c[0] > c[1], "short turn outweighed a turn five times its length"


def test_centroid_is_normalised():
    segs = _segs([[1, 1, 0], [1, 0.9, 0]])
    assert np.linalg.norm(centroid(segs)) == pytest.approx(1.0)


# --- naming ----------------------------------------------------------------


def _library(values: dict) -> dict:
    lib = {"speakers": []}
    for name, v in values.items():
        e = np.zeros(8)
        e[0] = v
        e[1] = np.sqrt(max(0.0, 1 - v * v))
        lib["speakers"].append({"name": name, "embedding": e})
    return lib


def _cluster_of(duration, vector=(1, 0, 0, 0, 0, 0, 0, 0), cid=0):
    s = Segment(0, duration)
    s.embedding = unit(vector)
    s.cluster = cid
    return s


def test_named_when_the_centroid_leads_the_field():
    tail = [0.26, 0.24, 0.20, 0.199] + list(np.linspace(0.19, 0.01, 36))
    lib = _library({"Laura": 0.46, **{f"P{i}": v for i, v in enumerate(tail)}})
    out = name_clusters([_cluster_of(30)], 1, lib)
    assert out[0].name == "Laura"
    assert out[0].label == "Laura"


def test_unmatched_cluster_stays_numbered():
    """A stranger must not be given the nearest name in the library."""
    lib = _library({f"P{i}": v for i, v in
                    enumerate([0.31, 0.30, 0.29, 0.28, 0.27, 0.26])})
    out = name_clusters([_cluster_of(30)], 1, lib)
    assert out[0].name is None
    assert out[0].label == "Speaker 1"


def test_short_cluster_is_not_matched_at_all():
    """
    Too little audio makes a noisy embedding. Better unnamed than wrong —
    a wrong name gets believed.
    """
    lib = _library({"Laura": 0.9, "Mike": 0.1, "Dana": 0.1, "Sam": 0.1})
    out = name_clusters([_cluster_of(2.0)], 1, lib)
    assert out[0].name is None


def test_two_clusters_matching_one_person_are_merged():
    """
    One speaker can split in two if their audio changed mid-meeting. Both
    halves match the same profile and should appear once.
    """
    tail = [0.26, 0.24, 0.20, 0.199] + list(np.linspace(0.19, 0.01, 36))
    lib = _library({"Laura": 0.46, **{f"P{i}": v for i, v in enumerate(tail)}})
    segs = [_cluster_of(20, cid=0), _cluster_of(15, cid=1)]
    out = name_clusters(segs, 2, lib)
    assert len(out) == 1
    assert out[0].name == "Laura"
    assert out[0].seconds == pytest.approx(35.0)


def test_speakers_are_ordered_by_talk_time():
    lib = _library({f"P{i}": 0.2 for i in range(6)})
    segs = [_cluster_of(5, cid=0), _cluster_of(40, cid=1), _cluster_of(20, cid=2)]
    out = name_clusters(segs, 3, lib)
    assert [round(s.seconds) for s in out] == [40, 20, 5]


def test_a_match_at_the_level_that_was_wrong_is_rejected():
    """
    Regression for a real misidentification. On a 41-profile library a cluster
    was confidently named as the wrong person at z=3.8, and only a human
    listening caught it. Correct matches in the same meeting scored 5.0 and
    8.6; a correct cross-channel self-test scored 4.4. Four is the line.
    """
    tail = [0.30, 0.28, 0.26, 0.25] + list(np.linspace(0.24, 0.02, 36))
    lib = _library({"Kody": 0.40, **{f"P{i}": v for i, v in enumerate(tail)}})
    out = name_clusters([_cluster_of(900)], 1, lib)
    assert out[0].z < 4.0
    assert out[0].name is None, f"named at z={out[0].z:.1f}, which was wrong before"


# --- ambiguity -------------------------------------------------------------


def test_two_close_candidates_are_flagged_not_named():
    """
    The real failure this fixes: an unenrolled speaker was confidently named
    as the nearest voice in the library, because the two people genuinely
    sound alike. The score was reasonable for a nearest neighbour and wrong as
    an identification. A thin gap to the runner-up is the tell.
    """
    tail = list(np.linspace(0.20, 0.02, 38))
    lib = _library({"Kody": 0.60, "Sam": 0.56,
                    **{f"P{i}": v for i, v in enumerate(tail)}})
    out = name_clusters([_cluster_of(900)], 1, lib)
    assert out[0].status == "ambiguous"
    assert out[0].name is None
    assert {c["name"] for c in out[0].candidates[:2]} == {"Kody", "Sam"}


def test_a_clear_winner_is_still_named():
    tail = list(np.linspace(0.20, 0.02, 38))
    lib = _library({"Matt": 0.78, "Other": 0.30,
                    **{f"P{i}": v for i, v in enumerate(tail)}})
    out = name_clusters([_cluster_of(900)], 1, lib)
    assert out[0].status == "named" and out[0].name == "Matt"


def test_nothing_close_is_unknown_not_ambiguous():
    """A stranger with no near match is a different problem from a tie."""
    tail = list(np.linspace(0.30, 0.02, 40))
    lib = _library({f"P{i}": v for i, v in enumerate(tail)})
    out = name_clusters([_cluster_of(900)], 1, lib)
    assert out[0].status == "unknown"


def test_short_clusters_report_too_short(tmp_path):
    tail = list(np.linspace(0.20, 0.02, 40))
    lib = _library({"Matt": 0.9, **{f"P{i}": v for i, v in enumerate(tail)}})
    out = name_clusters([_cluster_of(2.0)], 1, lib)
    assert out[0].status == "too-short"


def test_candidates_are_recorded_for_review():
    tail = list(np.linspace(0.20, 0.02, 38))
    lib = _library({"Kody": 0.60, "Sam": 0.56,
                    **{f"P{i}": v for i, v in enumerate(tail)}})
    out = name_clusters([_cluster_of(900)], 1, lib)
    assert len(out[0].candidates) >= 2
    assert all("score" in c for c in out[0].candidates)


# --- single-track recordings ------------------------------------------------


def test_single_track_recording_is_found(tmp_path):
    """
    A phone upload is one file holding everyone including you, so there is no
    system track to look for — but it still needs identifying.
    """
    from diarize import find_audio

    (tmp_path / "20261024-143012-plaud.mp3").write_bytes(b"")
    track, single = find_audio(tmp_path)
    assert single is True
    assert track.name == "20261024-143012-plaud.mp3"


def test_a_system_track_wins_over_anything_else(tmp_path):
    from diarize import find_audio

    (tmp_path / "x-system.wav").write_bytes(b"")
    (tmp_path / "x-mic.wav").write_bytes(b"")
    track, single = find_audio(tmp_path)
    assert single is False and track.name == "x-system.wav"


def test_derived_tracks_are_never_chosen(tmp_path):
    """The mic track is you alone; the mixdown is a derivative. Neither is
    the thing to diarise."""
    from diarize import find_audio

    for name in ("x-mic.wav", "x-mix.wav", "x-mic-raw.wav"):
        (tmp_path / name).write_bytes(b"")
    assert find_audio(tmp_path) == (None, False)


def test_empty_folder_yields_nothing(tmp_path):
    from diarize import find_audio

    assert find_audio(tmp_path) == (None, False)


def test_wav_reader_resamples_to_the_embedding_rate(tmp_path):
    import wave as _wave

    from diarize import TARGET_RATE, read_audio_mono

    p = tmp_path / "x.wav"
    rate = 48000
    t = np.linspace(0, 1.0, rate, endpoint=False)
    sig = (0.4 * np.sin(2 * np.pi * 300 * t) * 32767).astype(np.int16)
    with _wave.open(str(p), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.repeat(sig.reshape(-1, 1), 2, axis=1).tobytes())

    audio, out_rate = read_audio_mono(p)
    assert out_rate == TARGET_RATE
    assert audio.ndim == 1
    assert len(audio) == pytest.approx(TARGET_RATE, rel=0.01)


# --- merging fragmented speakers -------------------------------------------


def _frag(cid, base, duration, rng, jitter=0.08):
    s = Segment(0.0, duration)
    v = np.asarray(base, dtype=float) + rng.standard_normal(len(base)) * jitter
    s.embedding = v / np.linalg.norm(v)
    s.cluster = cid
    return s


def test_one_speaker_split_into_fragments_is_rejoined():
    """
    Why a single-track recording comes back with twenty speakers instead of
    five: clustering decides on individual turns, where two seconds is a poor
    sample of a voice. A cluster centroid is a much better one, so comparing
    centroids afterwards catches splits the first pass could not see.
    """
    from diarize import merge_similar

    rng = np.random.default_rng(4)
    a, b = np.zeros(16), np.zeros(16)
    a[0], b[1] = 1.0, 1.0
    segs = [_frag(0, a, 30, rng), _frag(1, a, 20, rng), _frag(2, a, 15, rng),
            _frag(3, a, 8, rng), _frag(4, b, 25, rng), _frag(5, b, 12, rng)]
    assert merge_similar(segs, 6) == 2
    assert len({s.cluster for s in segs[:4]}) == 1
    assert segs[0].cluster != segs[4].cluster


def test_two_similar_but_different_people_are_not_combined():
    """
    Both centroids come from one recording, so this is a same-channel
    comparison — the kind that scores high. On a real library the closest two
    different people reached 0.55 under those conditions, so the threshold
    has to clear it.
    """
    from diarize import merge_similar

    rng = np.random.default_rng(1)
    a = np.zeros(16)
    a[0] = 1.0
    close = np.zeros(16)
    close[0], close[1] = 0.55, np.sqrt(1 - 0.55 ** 2)
    segs = [_frag(0, a, 30, rng), _frag(1, close, 25, rng)]
    assert merge_similar(segs, 2) == 2


def test_merge_leaves_a_single_cluster_alone():
    from diarize import merge_similar

    rng = np.random.default_rng(2)
    a = np.zeros(16)
    a[0] = 1.0
    assert merge_similar([_frag(0, a, 30, rng)], 1) == 1
    assert merge_similar([], 0) == 0


def test_merge_renumbers_clusters_contiguously():
    """Downstream code iterates range(n), so gaps would lose speakers."""
    from diarize import merge_similar

    rng = np.random.default_rng(3)
    a, b = np.zeros(16), np.zeros(16)
    a[0], b[1] = 1.0, 1.0
    segs = [_frag(0, a, 30, rng), _frag(1, a, 20, rng), _frag(2, b, 25, rng)]
    n = merge_similar(segs, 3)
    assert sorted({s.cluster for s in segs}) == list(range(n))


# --- date cutoff ------------------------------------------------------------


def test_spans_and_dates_both_parse():
    from diarize import parse_since

    from datetime import datetime

    assert parse_since("2026-10-01") == datetime(2026, 10, 1)
    assert parse_since("20261001") == datetime(2026, 10, 1)
    assert parse_since(None) is None
    assert parse_since("") is None
    # a span is relative, so only its distance from now is fixed
    assert 6 <= (datetime.now() - parse_since("7d")).days <= 7
    assert 27 <= (datetime.now() - parse_since("4w")).days <= 28


def test_an_unreadable_date_is_rejected_loudly():
    """Silently processing everything would be worse than refusing."""
    from diarize import parse_since

    with pytest.raises(ValueError):
        parse_since("last tuesday")


def test_the_date_comes_from_the_folder_name_not_the_filesystem(tmp_path):
    """
    Copying or re-syncing an archive rewrites every modification time, so
    mtime says when the file was last touched, not when the meeting happened.
    """
    from datetime import datetime

    from diarize import folder_date

    assert folder_date(Path("20261025-111000-plaud")) == datetime(2026, 10, 25, 11, 10)
    assert folder_date(Path("20261013-134658-ms-teams.exe")) \
        == datetime(2026, 10, 13, 13, 46, 58)
    assert folder_date(Path("_shared")) is None


def test_cutoff_keeps_recent_and_drops_older():
    from datetime import datetime

    from diarize import after_cutoff

    cut = datetime(2026, 10, 20)
    assert after_cutoff(Path("20261025-111000-plaud"), cut)
    assert not after_cutoff(Path("20261013-134658-ms-teams.exe"), cut)
    assert after_cutoff(Path("20261013-134658-ms-teams.exe"), None)


def test_an_undated_folder_is_never_skipped():
    """Skipping something because its name is unusual would lose work
    silently; processing it costs one pass."""
    from datetime import datetime

    from diarize import after_cutoff

    assert after_cutoff(Path("some-odd-folder"), datetime(2026, 10, 20))


# --- robustness over a whole archive ----------------------------------------


def test_support_folders_are_not_treated_as_meetings():
    """
    A folder of enrollment samples was diarised as a meeting and confidently
    matched to the person whose samples it held. Shared data lives alongside
    the recordings; only timestamped folders are meetings.
    """
    from diarize import is_meeting_folder

    assert is_meeting_folder(Path("DESKTOP-1/2026/10/20261027-085902-ms-teams.exe"))
    assert is_meeting_folder(Path("PLAUD-NOTEPRO/2026/10/20261025-111000-plaud"))
    assert not is_meeting_folder(Path("_shared/speakers"))
    assert not is_meeting_folder(Path("_selftest"))
    assert not is_meeting_folder(Path("GTERRY/2026/09/notes"))


def test_an_unreadable_recording_does_not_abort_the_batch(tmp_path, capsys):
    """
    A recording still being written killed the whole run and skipped every
    meeting after it — far worse than one meeting going unprocessed.
    """
    from types import SimpleNamespace

    from diarize import process

    folder = tmp_path / "20261027-085902-ms-teams.exe"
    folder.mkdir()
    (folder / "x-system.wav").write_bytes(b"")  # zero length, as if mid-write

    args = SimpleNamespace(force=True, distance=0.65, merge=0.68, z=4.0,
                           threshold=0.65, margin=0.10, me=None)
    assert process(folder, encoder=None, library={"speakers": []}, args=args) is False
    out = capsys.readouterr().out
    assert "skipped" in out
    assert "still recording" in out, "no hint about what went wrong"
