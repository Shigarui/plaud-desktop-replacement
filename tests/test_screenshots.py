"""Screenshot tests. Pure logic only — no windows, no screen, runs anywhere."""

import numpy as np
import pytest

from meetingrec.screenshots import (
    APP_PROCESSES,
    ScreenshotCapturer,
    ShotStats,
    dhash,
    hamming,
)


def slide(text_rows: int, seed: int = 0) -> np.ndarray:
    """A fake 'slide': mostly white with some dark bars, like text on a page."""
    rng = np.random.default_rng(seed)
    img = np.full((720, 1280, 3), 240, dtype=np.uint8)
    for i in range(text_rows):
        y = 80 + i * 60
        img[y:y + 24, 100:100 + int(rng.integers(300, 900))] = 30
    return img


def test_identical_frames_hash_identically():
    a = slide(5)
    assert dhash(a) == dhash(a.copy())


def test_cursor_sized_change_does_not_trip_the_threshold():
    """A moving cursor or ticking timer must not count as a new slide."""
    a = slide(5)
    b = a.copy()
    b[300:316, 600:616] = 0  # ~16px cursor
    assert hamming(dhash(a), dhash(b)) < 12


def test_different_slides_exceed_the_threshold():
    a, b = slide(4, seed=1), slide(9, seed=2)
    assert hamming(dhash(a), dhash(b)) >= 12


def test_dhash_is_64_bits():
    assert 0 <= dhash(slide(3)) < 2 ** 64


def test_hamming_counts_differing_bits():
    assert hamming(0b1010, 0b1001) == 2
    assert hamming(5, 5) == 0


def test_grayscale_input_is_accepted():
    gray = slide(4)[..., 0]
    assert isinstance(dhash(gray), int)


def test_teams_identity_maps_to_real_processes():
    """The consent store reports a package family name, not an exe."""
    procs = APP_PROCESSES["msteams_8wekyb3d8bbwe"]
    assert any(p.lower().endswith(".exe") for p in procs)
    assert any("teams" in p.lower() for p in procs)


def test_capturer_starts_disarmed():
    c = ScreenshotCapturer()
    assert not c._active.is_set()
    assert c.stats.captured == 0


def test_arm_resets_state_between_meetings(tmp_path):
    c = ScreenshotCapturer()
    c.stats = ShotStats(captured=7, errors=2)
    c._last_hash = 12345
    c.arm(tmp_path, "slug", "zoom.exe")
    assert c.stats.captured == 0 and c.stats.errors == 0
    assert c._last_hash is None
    assert c._active.is_set()
    c.disarm()
    assert not c._active.is_set()


def test_disarm_returns_stats(tmp_path):
    c = ScreenshotCapturer()
    c.arm(tmp_path, "slug", "zoom.exe")
    c.stats.captured = 3
    stats = c.disarm()
    assert stats.captured == 3


def test_unknown_app_without_exe_suffix_is_ignored(tmp_path):
    """A packaged identity we have no mapping for must not crash the tick."""
    c = ScreenshotCapturer()
    c.arm(tmp_path, "slug", "some.unknown_package_8wekyb3d8bbwe")
    c._tick()  # must return quietly
    assert c.stats.captured == 0


def test_video_tile_churn_does_not_trigger_a_capture():
    """
    The realistic false positive: a small webcam tile updating constantly
    while the slide behind it never changes.
    """
    rng = np.random.default_rng(0)
    base = slide(5)
    noisy = base.copy()
    noisy[600:640, 1000:1200] = rng.integers(0, 255, (40, 200, 3), dtype=np.uint8)
    assert hamming(dhash(base), dhash(noisy)) < 12


def test_thin_content_is_not_lost_in_downscaling():
    """
    Point-sampling an 8x8 grid misses thin text entirely and hashes two
    different slides within a few bits of each other. Area-averaging fixes it.
    """
    worst = min(hamming(dhash(slide(4, seed=s)), dhash(slide(9, seed=s + 10)))
                for s in range(1, 8))
    assert worst >= 12, f"closest pair differed by only {worst} bits"


# --- foreground following and exclusions -----------------------------------


def test_exclusion_matches_title_or_process():
    c = ScreenshotCapturer()
    assert c._is_excluded("Vault - 1Password", "1password.exe")
    assert c._is_excluded("Chase Online Banking", "chrome.exe")
    assert c._is_excluded("anything", "bitwarden.exe")
    assert not c._is_excluded("Q3 Review.pptx - PowerPoint", "POWERPNT.EXE")


def test_exclusion_is_case_insensitive():
    c = ScreenshotCapturer()
    assert c._is_excluded("MY PASSWORD VAULT", "")
    assert c._is_excluded("", "KeePassXC.exe")


def test_custom_exclude_list_replaces_the_default():
    c = ScreenshotCapturer(exclude=("secretproject",))
    assert c._is_excluded("secretproject roadmap", "chrome.exe")
    assert not c._is_excluded("1Password", "1password.exe")


def test_excluded_window_is_counted_not_captured(tmp_path, monkeypatch):
    from meetingrec import screenshots as mod

    c = ScreenshotCapturer(follow_foreground=True)
    c.arm(tmp_path, "slug", "zoom.exe")
    monkeypatch.setattr(mod, "foreground_window",
                        lambda: (1, "Vault - Bitwarden", "bitwarden.exe"))
    c._tick()
    assert c.stats.skipped_excluded == 1
    assert c.stats.captured == 0
    c.disarm()


def test_follow_foreground_uses_the_top_window_not_the_meeting(tmp_path, monkeypatch):
    """
    While you present, you are looking at the shared tab, not the meeting UI.
    """
    from meetingrec import screenshots as mod

    called = {"meeting_lookup": 0}
    monkeypatch.setattr(mod, "find_window_for_processes",
                        lambda *a: called.__setitem__("meeting_lookup", 1) or (9, "Teams"))
    monkeypatch.setattr(mod, "foreground_window",
                        lambda: (42, "Q3 dashboard - Chrome", "chrome.exe"))

    c = ScreenshotCapturer(follow_foreground=True)
    c.arm(tmp_path, "slug", "msteams_8wekyb3d8bbwe")
    assert c._pick_window() == (42, "Q3 dashboard - Chrome", "chrome.exe")
    assert called["meeting_lookup"] == 0, "should not have looked for the meeting window"
    c.disarm()


def test_pinned_mode_still_targets_the_meeting_window(tmp_path, monkeypatch):
    from meetingrec import screenshots as mod

    monkeypatch.setattr(mod, "find_window_for_processes", lambda *a: (9, "Zoom Meeting"))
    monkeypatch.setattr(mod, "is_foreground", lambda hwnd: True)
    c = ScreenshotCapturer(follow_foreground=False)
    c.arm(tmp_path, "slug", "zoom.exe")
    assert c._pick_window() == (9, "Zoom Meeting", "")
    c.disarm()


def test_manifest_is_written_on_disarm(tmp_path):
    import json

    c = ScreenshotCapturer()
    c.arm(tmp_path, "mtg", "zoom.exe")
    c.stats.shots.append({"file": "mtg-shot001.jpg", "title": "Zoom Meeting",
                          "process": "Zoom.exe", "offset_seconds": 12.0})
    stats = c.disarm()
    manifest = tmp_path / "mtg-shots.json"
    assert manifest.exists()
    assert json.loads(manifest.read_text())[0]["title"] == "Zoom Meeting"
    assert manifest in stats.paths


def test_no_manifest_when_nothing_captured(tmp_path):
    c = ScreenshotCapturer()
    c.arm(tmp_path, "mtg", "zoom.exe")
    c.disarm()
    assert not (tmp_path / "mtg-shots.json").exists()


def test_no_method_shadows_a_thread_attribute():
    """
    threading.Thread sets instance attributes like _target, _stop and _args in
    __init__. A method sharing one of those names is silently replaced by the
    attribute and fails only when something calls it.
    """
    import threading

    probe = threading.Thread()
    reserved = {n for n in vars(probe) if n.startswith("_")}
    ours = {n for n in vars(ScreenshotCapturer) if n.startswith("_")
            and callable(getattr(ScreenshotCapturer, n, None))}
    assert not (ours & reserved), f"shadowed by Thread: {sorted(ours & reserved)}"
