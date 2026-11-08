"""
meetingrec.config — per-machine settings.

Reads config.toml from the project root (or $MEETINGREC_CONFIG). That file is
gitignored on purpose: paths and allowlists differ across machines, and the
NAS mount letter is not the same everywhere. config.example.toml is the
committed template.

Design note on paths: capture always writes to `staging_dir` on a local disk.
Never point that at a network share — a Tailscale hiccup mid-call will stall
the writer thread and you will lose audio. Finished files get moved to
`archive_dir` afterward, and that is where the NAS belongs.
"""

from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.toml"


@dataclass
class Config:
    # --- detection ---
    poll_interval: float = 2.0
    stop_debounce: float = 45.0
    allowlist: Optional[list[str]] = None  # None -> detector defaults
    ambiguous: Optional[list[str]] = None
    require_playback_for_ambiguous: bool = True
    # How microphone use is detected:
    #   "sessions" — live WASAPI capture sessions. Reliable on every build.
    #   "consent"  — the registry consent store. Gives a true meeting start
    #                time, but silently stops updating on some builds.
    #   "auto"     — sessions where available, consent otherwise.
    detection_source: str = "auto"

    # --- capture ---
    sample_rate: int = 48000
    preroll_seconds: float = 30.0
    # Recordings shorter than this are discarded — accidental triggers, a
    # call joined and immediately left. Set discard_short_sessions = false to
    # archive them anyway; a two-minute check-in can still be worth keeping.
    min_session_seconds: float = 30.0
    discard_short_sessions: bool = True
    # One name, or a list in priority order:
    #   mic_device = ["Headset Microphone (Poly BT700)", "Headset (soundcore ...)"]
    # Matching is loose, so a distinctive fragment is enough.
    mic_device: Optional[object] = None
    # Rewrite a narrowband mic track at its true rate instead of storing
    # upsampled 48 kHz that carries no extra information.
    normalize_mic_rate: bool = True
    # Desk mics capture far more sub-300 Hz rumble than speech. High-passing
    # and normalising is what makes a raw capture sound like a voice recorder.
    clean_mic: bool = True
    highpass_hz: float = 250.0
    keep_raw_mic: bool = False
    # Bluetooth packet loss arrives as runs of exact zeros. Bridging short ones
    # and ramping long ones removes the clicks that make speech sound garbled.
    conceal_dropouts: bool = True
    # Keep the mic closed until a meeting starts. Bluetooth headsets drop to
    # 16 kHz mono the moment anything opens their mic, so holding it open all
    # day degrades every other sound on the machine. Costs mic pre-roll only.
    lazy_mic: bool = True
    # Capture buffer size in frames. 1024 (~21 ms) measured far better than
    # 4096 on Bluetooth: large buffers starve between packets and return
    # silence. Raise only if you see overflows in -v output.
    block_frames: int = 1024
    # Seconds to let a lazily-opened mic settle before the file is opened.
    # Bluetooth streams drop heavily for the first second or two after opening.
    warmup_seconds: float = 1.5
    # Force the mic capture rate. Bluetooth endpoints often accept 48000 and
    # upsample a 16 kHz link internally, which adds dropouts and no detail.
    # 16000 measured cleanest on a Bluetooth headset; leave unset for wired.
    mic_rate: Optional[int] = None

    # --- screenshots ---
    # Visual context from the meeting window. Pinned to that window only, and
    # only while it is in the foreground, so other apps are never captured.
    screenshots: bool = False
    screenshot_interval: float = 3.0        # how often to look
    screenshot_min_gap: float = 15.0        # never save more often than this
    screenshot_change_threshold: int = 12   # dhash bits (0-64); lower = fussier
    screenshot_quality: int = 70
    screenshot_max_per_meeting: int = 200
    screenshot_foreground_only: bool = True
    # Capture whatever is on top, not just the meeting window. Needed to catch
    # your own screen shares — the app you share renders in its own window.
    screenshot_follow_foreground: bool = True
    # Window titles / process names never captured, case-insensitive substrings.
    # None uses the built-in list (password managers, banking, private windows).
    screenshot_exclude: Optional[list] = None

    # --- post-processing ---
    # Diarise and transcribe automatically after a meeting. Runs out of
    # process, at low priority, and waits for any live recording to finish —
    # transcription saturates the CPU and a dropped recording cannot be redone.
    post_process: bool = False
    post_process_delay: float = 300.0        # seconds after the meeting ends
    post_process_library: Optional[str] = None   # voiceprints.json
    post_process_me: Optional[str] = None        # your profile name
    post_process_mic_label: str = "Me"           # label for your track
    whisper_model: str = "medium"
    # CPU threads for transcription. None uses half the cores, which keeps
    # permanent headroom so contention rarely becomes a problem at all.
    whisper_threads: Optional[int] = None
    # Only process during these hours, e.g. "18:00-07:00". Work queued outside
    # the window waits, and the queue survives restarts so nothing is lost.
    post_process_window: Optional[str] = None
    summarise_with: Optional[str] = None         # e.g. "gpt-oss:20b"

    # --- storage ---
    # backend: "drive" (Drive for Desktop), "driveapi" (API v3), or "local"
    backend: str = "drive"
    staging_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "recordings")
    archive_dir: Optional[Path] = None  # only used by backend = "local"
    group_by: str = "month"  # month | day | flat
    # One folder per meeting inside that, so a call's audio, screenshots and
    # metadata stay together instead of interleaving with every other meeting.
    folder_per_meeting: bool = True

    # Drive for Desktop
    drive_folder: str = "Meeting Recordings"
    drive_root: Optional[Path] = None  # auto-detected if unset (e.g. G:/My Drive)

    # Drive API v3 only
    drive_credentials: Path = field(default_factory=lambda: PROJECT_ROOT / "client_secret.json")
    drive_token: Path = field(default_factory=lambda: PROJECT_ROOT / "token.json")
    drive_shared_drive_id: Optional[str] = None

    # --- misc ---
    machine: str = field(default_factory=lambda: os.environ.get("COMPUTERNAME", "unknown"))
    notify_on_start: bool = True

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Config":
        path = path or Path(os.environ.get("MEETINGREC_CONFIG", DEFAULT_CONFIG_PATH))
        cfg = cls()
        if not path.exists():
            log.info("no config at %s — using defaults", path)
            cfg.staging_dir.mkdir(parents=True, exist_ok=True)
            return cfg

        with path.open("rb") as fh:
            raw: dict[str, Any] = tomllib.load(fh)

        flat: dict[str, Any] = {}
        for section in ("detection", "capture", "screenshots", "post_process",
                        "storage", "paths", "misc"):
            flat.update(raw.get(section, {}))
        flat.update({k: v for k, v in raw.items() if not isinstance(v, dict)})

        known = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(flat) - known
        if unknown:
            log.warning("ignoring unknown config keys: %s", ", ".join(sorted(unknown)))

        for key in ("staging_dir", "archive_dir", "drive_root", "drive_credentials", "drive_token"):
            if flat.get(key):
                flat[key] = Path(str(flat[key])).expanduser()

        cfg = replace(cfg, **{k: v for k, v in flat.items() if k in known})
        cfg.staging_dir.mkdir(parents=True, exist_ok=True)
        log.debug("loaded config from %s", path)
        return cfg
