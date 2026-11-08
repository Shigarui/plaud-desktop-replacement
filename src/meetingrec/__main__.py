"""CLI entry point:  py -m meetingrec <command>"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import Config
from .detector import (
    FILETIME_EPOCH,
    CaptureSessionProbe,
    ConsentStoreProbe,
    MeetingDetector,
    MeetingSession,
    build_mic_source,
)
from .postprocess import PostProcessor
from .singleton import AlreadyRunning, SingleInstance
from .recorder import DualRecorder, list_devices
from .screenshots import ScreenshotCapturer
from .storage import build_backend, find_drive_desktop_root, safe_name, session_subdir

log = logging.getLogger("meetingrec")


def _print_track_stats(result, cfg: Config) -> None:
    """Shared summary for testrecord and run."""
    alignment = result.stats.get("alignment") or {}
    if alignment.get("padded"):
        print(f"  aligned: padded {alignment['padded']} by "
              f"{alignment['padded_seconds']}s (offset {alignment['offset_seconds']}s)")

    for role, st in result.stats.items():
        if role == "alignment" or not isinstance(st, dict):
            continue
        rate = st.get("rate") or cfg.sample_rate
        print(f"  [{role}] device={st.get('device', '?')} @ {rate} Hz "
              f"blocks={st.get('blocks', 0)} "
              f"padded={st.get('frames_padded', 0) / max(rate, 1):.2f}s "
              f"errors={st.get('errors', 0)}"
              + ("  (FALLBACK)" if st.get("fallback") else ""))
        if st.get("discontinuities"):
            print(f"         driver gaps: {st['discontinuities']} buffer "
                  f"discontinuities reported")
        if st.get("dead_devices"):
            print(f"         DEAD DEVICES SKIPPED: {', '.join(st['dead_devices'])}")
        if st.get("dropouts"):
            print(f"         dropouts: {st['dropouts']} ({st['dropout_pct']}% of file) "
                  f"- {st['bridged']} bridged, {st['ramped']} ramped")
        if "speech_band_after" in st:
            print(f"         speech band: {st['speech_band_before'] * 100:.0f}%"
                  f" -> {st['speech_band_after'] * 100:.0f}% "
                  f"(HPF {st['highpass_hz']:.0f} Hz, gain {st['gain_db']:+.1f} dB)")


def cmd_list(cfg: Config) -> int:
    live = CaptureSessionProbe()
    if live.available:
        active = live()
        print("Capturing RIGHT NOW (live audio sessions):")
        if active:
            for u in active:
                print(f"    {u.name:<30} on {u.device}")
        else:
            print("    (nothing)")
        print()

    probe = ConsentStoreProbe()
    entries = probe.read_all(active_only=False)
    if not entries:
        print("No consent-store entries found. On some Windows builds this")
        print("store stops updating entirely — detection uses live sessions.")
        return 0
    active_keys = {(u.name, u.started_at) for u in probe.read_all(active_only=True)}
    print(f"{len(entries)} app(s) have used the microphone:\n")
    for u in sorted(entries, key=lambda x: (x.started_at or FILETIME_EPOCH), reverse=True):
        flag = "  <== MIC OPEN NOW" if (u.name, u.started_at) in active_keys else ""
        when = u.started_at.astimezone().strftime("%Y-%m-%d %H:%M:%S") if u.started_at else "never"
        print(f"  [{'pkg' if u.packaged else 'exe'}] {u.name:<40} last start {when}{flag}")
        if u.path:
            print(f"         {u.path}")
    return 0


def cmd_watch(cfg: Config) -> int:
    def on_start(s: MeetingSession) -> None:
        print(f"\n>>> START  #{s.session_id}  {s.app}  {s.started_at.astimezone():%H:%M:%S}")

    def on_stop(s: MeetingSession) -> None:
        print(f"<<< STOP   #{s.session_id}  {str(s.duration).split('.')[0]}  "
              f"apps={','.join(sorted(s.apps_seen))}\n")

    d = MeetingDetector(
        mic_source=build_mic_source(cfg.detection_source),
        poll_interval=cfg.poll_interval,
        stop_debounce=cfg.stop_debounce,
        allowlist=cfg.allowlist,
        ambiguous=cfg.ambiguous,
        require_playback_for_ambiguous=cfg.require_playback_for_ambiguous,
        on_start=on_start,
        on_stop=on_stop,
    )
    last = ""
    try:
        while True:
            state = d.poll_once()
            # Show the SESSION start, not each MicUser's stamp: a live capture
            # session is stamped "now" every poll, so printing that makes a
            # steady meeting look like it restarts every two seconds.
            names = ", ".join(sorted({u.name for u in d.qualifying_apps()})) or "-"
            if d.session:
                since = d.session.started_at.astimezone().strftime("%H:%M:%S")
                elapsed = str(d.session.duration).split(".")[0]
                line = f"{state:<12} {names} (since {since}, {elapsed})"
            else:
                line = f"{state:<12} {names}"
            if line != last:
                print(f"[{datetime.now():%H:%M:%S}] {line}")
                last = line
            time.sleep(cfg.poll_interval)
    except KeyboardInterrupt:
        print("\nstopped")
        return 0


def cmd_storage(cfg: Config) -> int:
    """Verify the archive target before trusting it with a real recording."""
    print(f"backend      : {cfg.backend}")
    print(f"staging dir  : {cfg.staging_dir}  (exists={cfg.staging_dir.exists()})")
    root = find_drive_desktop_root()
    print(f"Drive desktop: {root or 'NOT FOUND'}")
    try:
        backend = build_backend(cfg)
    except Exception as exc:
        print(f"backend init FAILED: {exc}")
        return 1
    print(f"resolved     : {backend.name}")

    probe = cfg.staging_dir / "meetingrec-storage-test.txt"
    probe.write_text(f"written {datetime.now():%Y-%m-%d %H:%M:%S} by meetingrec\n",
                     encoding="utf-8")
    result = backend.archive(probe, "_selftest", {"test": True, "machine": cfg.machine})
    print(f"archive      : {'OK' if result.ok else 'FAILED'}  {result.location or result.error}")
    return 0 if result.ok else 1


def cmd_devices(cfg: Config) -> int:
    print(list_devices())
    return 0


def _build_recorder(cfg: Config, mic: Optional[str] = None) -> DualRecorder:
    return DualRecorder(
        cfg.staging_dir,
        rate=cfg.sample_rate,
        preroll_seconds=cfg.preroll_seconds,
        mic_device=mic or cfg.mic_device,
        normalize_mic_rate=cfg.normalize_mic_rate,
        clean_mic=cfg.clean_mic,
        highpass_hz=cfg.highpass_hz,
        keep_raw_mic=cfg.keep_raw_mic,
        conceal_dropouts=cfg.conceal_dropouts,
        lazy_mic=cfg.lazy_mic,
        block_frames=cfg.block_frames,
        warmup_seconds=cfg.warmup_seconds,
        mic_rate=cfg.mic_rate,
    )


def cmd_testrecord(cfg: Config, seconds: float = 15.0, mic: Optional[str] = None) -> int:
    """Record a fixed window with no detector involved. Proves the audio path."""
    rec = _build_recorder(cfg, mic)
    rec.start()
    print(f"warming up pre-roll for 3s, then recording {seconds:.0f}s...")
    print("PLAY SOMETHING (a video, music) AND TALK so both tracks have signal.")
    time.sleep(3)
    rec.start_recording(f"testrecord-{datetime.now():%Y%m%d-%H%M%S}")
    time.sleep(seconds)
    result = rec.shutdown()
    if not result:
        print("nothing recorded")
        return 1
    print(f"\nduration: {result.duration_seconds:.1f}s "
          f"(includes {result.preroll_seconds:.1f}s pre-roll)")
    for path in result.paths:
        print(f"  {path.stat().st_size / 1_000_000:7.2f} MB  {path}")
    _print_track_stats(result, cfg)
    print("\nListen to all three before trusting this.")
    return 0


def cmd_run(cfg: Config) -> int:
    """The real thing: detect, record, archive."""
    lock = SingleInstance(Path(__file__).resolve().parents[2] / "logs" / "meetingrec.pid")
    try:
        lock.acquire()
    except AlreadyRunning as exc:
        log.error("%s — not starting a second recorder", exc)
        log.error("two recorders fight over the audio devices and both capture "
                  "worse audio than one")
        return 1

    try:
        return _run(cfg)
    finally:
        lock.release()


def _run(cfg: Config) -> int:
    backend = build_backend(cfg)
    rec = _build_recorder(cfg)
    rec.start()

    post: Optional[PostProcessor] = None
    if cfg.post_process:
        post = PostProcessor(
            project_root=Path(__file__).resolve().parents[2],
            python_exe=sys.executable,
            delay_seconds=cfg.post_process_delay,
            library=cfg.post_process_library,
            me=cfg.post_process_me,
            mic_label=cfg.post_process_mic_label,
            whisper_model=cfg.whisper_model,
            threads=cfg.whisper_threads,
            window=cfg.post_process_window,
            queue_file=cfg.staging_dir / "postprocess-queue.json",
            summarise=cfg.summarise_with,
            busy=lambda: rec.armed,
        )
        post.start()
        log.info("post-processing on: %.0f min after each meeting",
                 cfg.post_process_delay / 60)

    shots: Optional[ScreenshotCapturer] = None
    if cfg.screenshots:
        shots = ScreenshotCapturer(
            interval=cfg.screenshot_interval,
            min_gap_seconds=cfg.screenshot_min_gap,
            change_threshold=cfg.screenshot_change_threshold,
            quality=cfg.screenshot_quality,
            max_per_meeting=cfg.screenshot_max_per_meeting,
            foreground_only=cfg.screenshot_foreground_only,
            follow_foreground=cfg.screenshot_follow_foreground,
            **({"exclude": tuple(cfg.screenshot_exclude)}
               if cfg.screenshot_exclude else {}),
        )
        shots.start()

    def on_start(s: MeetingSession) -> None:
        slug = f"{s.started_at.astimezone():%Y%m%d-%H%M%S}-{safe_name(s.app, 30)}"
        rec.start_recording(slug)
        if shots:
            shots.arm(cfg.staging_dir, slug, s.app)

    def on_stop(s: MeetingSession) -> None:
        result = rec.stop_recording()
        shot_stats = shots.disarm() if shots else None
        if not result:
            return

        alignment = result.stats.get("alignment") or {}
        if alignment.get("padded"):
            log.info("aligned tracks: padded %s by %.2fs (offset %.2fs)",
                     alignment["padded"], alignment["padded_seconds"],
                     alignment["offset_seconds"])

        for role, st in result.stats.items():
            if role == "alignment" or not isinstance(st, dict):
                continue
            if st.get("discontinuities"):
                log.info("[%s] %d buffer discontinuities reported by the driver",
                         role, st["discontinuities"])
            if st.get("dead_devices"):
                log.error("[%s] recorded from a dead device during this meeting: %s",
                          role, ", ".join(st["dead_devices"]))

        if result.duration_seconds < cfg.min_session_seconds:
            # Loud, and it names the threshold. Deleting a recording quietly
            # is the kind of thing you discover weeks later when you go
            # looking for a call that was shorter than you remembered.
            if cfg.discard_short_sessions:
                log.warning("DISCARDING %.0fs recording of %s — shorter than "
                            "min_session_seconds (%.0fs). Set discard_short_sessions "
                            "= false to keep these.",
                            result.duration_seconds, s.app, cfg.min_session_seconds)
                for p in result.paths:
                    p.unlink(missing_ok=True)
                return
            log.info("keeping %.0fs recording (under min_session_seconds, "
                     "but discard_short_sessions is off)", result.duration_seconds)

        subdir = session_subdir(
            s.started_at.astimezone(), cfg.machine, cfg.group_by,
            slug=result.session_slug if cfg.folder_per_meeting else None)
        meta = {
            "app": s.app,
            "apps_seen": sorted(s.apps_seen),
            "started_at": s.started_at.isoformat(),
            "ended_at": s.ended_at.isoformat() if s.ended_at else None,
            "duration_seconds": round(result.duration_seconds, 1),
            "detection_lag_seconds": round(s.detection_lag.total_seconds(), 1),
            "machine": cfg.machine,
        }
        extra = list(shot_stats.paths) if shot_stats else []
        if extra:
            meta["screenshots"] = len(extra)
            meta["window_title"] = shot_stats.window_title

        archived_to: Optional[Path] = None
        for path in result.paths + extra:
            outcome = backend.archive(path, subdir, meta)
            if outcome.ok and archived_to is None:
                archived_to = Path(outcome.location).parent
            if not outcome.ok:
                log.error("archive failed for %s: %s — file kept in staging",
                          path.name, outcome.error)

        if post and archived_to:
            post.submit(archived_to)

    detector = MeetingDetector(
        mic_source=build_mic_source(cfg.detection_source),
        poll_interval=cfg.poll_interval,
        stop_debounce=cfg.stop_debounce,
        allowlist=cfg.allowlist,
        ambiguous=cfg.ambiguous,
        require_playback_for_ambiguous=cfg.require_playback_for_ambiguous,
        on_start=on_start,
        on_stop=on_stop,
    )
    print("meetingrec running. Ctrl-C to stop.")
    try:
        detector.run()
    except KeyboardInterrupt:
        pass
    finally:
        rec.shutdown()
        if shots:
            shots.stop()
        if post:
            post.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="meetingrec", description="Meeting detection and recording.")
    p.add_argument(
        "command",
        choices=["list", "watch", "storage", "devices", "testrecord", "run"],
        help="list: mic consent entries | watch: live detection | storage: test archive | "
             "devices: audio devices | testrecord: fixed-length capture | run: detect+record+archive",
    )
    p.add_argument("--seconds", type=float, default=15.0, help="testrecord length")
    p.add_argument("--mic", help="override mic_device for this run (partial name is fine)")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--log-file", default=None,
                   help="where to write the rotating log (default logs/meetingrec.log)")
    args = p.parse_args(argv)

    # Windows consoles default to a legacy code page (cp1252), so any log line
    # containing an em dash or arrow raises UnicodeEncodeError and is silently
    # dropped. Reconfigure before the first log call.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    level = logging.DEBUG if args.verbose else logging.INFO
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                            datefmt="%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(level)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    # Always log to a file as well. Under Task Scheduler there is no console
    # at all, so without this a meeting that fails to record leaves no trace
    # anywhere — and the whole point of running unattended is not watching.
    log_path = Path(args.log_file) if args.log_file else \
        Path(__file__).resolve().parents[2] / "logs" / "meetingrec.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        from logging.handlers import RotatingFileHandler

        rotating = RotatingFileHandler(log_path, maxBytes=2_000_000,
                                       backupCount=5, encoding="utf-8")
        rotating.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
        root.addHandler(rotating)
        log.info("logging to %s", log_path)
    except Exception as exc:
        log.warning("could not open the log file (%r) — console only", exc)
    if sys.platform != "win32":
        print("meetingrec only runs on Windows.", file=sys.stderr)
        return 2

    cfg = Config.load()
    if args.command == "testrecord":
        return cmd_testrecord(cfg, args.seconds, args.mic)
    return {
        "list": cmd_list,
        "watch": cmd_watch,
        "storage": cmd_storage,
        "devices": cmd_devices,
        "run": cmd_run,
    }[args.command](cfg)


if __name__ == "__main__":
    raise SystemExit(main())
