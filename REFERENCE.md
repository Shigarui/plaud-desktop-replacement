# meetingrec — complete reference

Everything in one place: what each module does, every command, every config
option, the daily workflows, and the findings behind the non-obvious choices.

Companion files: **INSTALL.md** for a new machine, **README.md** for the short
version. This document is the long one.

---

## Contents

1. [What it does](#1-what-it-does)
2. [Project layout](#2-project-layout)
3. [Install](#3-install)
4. [Commands](#4-commands)
5. [Workflows](#5-workflows)
6. [Configuration reference](#6-configuration-reference)
7. [Module reference](#7-module-reference)
8. [Scripts reference](#8-scripts-reference)
9. [Output layout](#9-output-layout)
10. [Running unattended](#10-running-unattended)
11. [Findings and design decisions](#11-findings-and-design-decisions)
12. [Troubleshooting](#12-troubleshooting)
13. [Tests](#13-tests)

---

## 1. What it does

Detects when a virtual meeting starts on Windows, records desktop audio and
microphone as **separate tracks**, captures screenshots of the meeting window,
archives everything to cloud storage, then transcribes it with named speakers.

```
meeting starts
   ↓  detector sees a live mic capture session
   ↓  mic stream opens; system stream already buffering (30s pre-roll)
   ↓  screenshots on visual change
meeting ends (+45s debounce)
   ↓  tracks aligned, mic cleaned, dropouts concealed
   ↓  archived to <machine>/<year>/<month>/<meeting>/
   ↓  (after the quiet-hours window) diarize → transcribe
transcript.md with real names
```

The architectural decision everything depends on: **two separate audio tracks**.
The mic is you; the system track is everyone else. Speaker attribution for a
two-party call is then correct by construction rather than inferred.

### Current state

| Piece | State |
|---|---|
| Detection | working |
| Dual-stream capture, pre-roll, drift correction | working |
| Device fallback, silence watchdog | working |
| Speech cleanup, packet-loss concealment | working |
| Track alignment | working |
| Screenshots | working |
| Cloud archiving | working |
| Transcription | working |
| Speaker identification + review GUI | working |
| Deferred post-processing | working |
| Unattended running | working |
| Tray icon | not built |

~6,200 lines across 10 modules and 5 scripts. 242 tests, none needing audio
hardware or network.

---

## 2. Project layout

```
meetingrec/
├── src/meetingrec/
│   ├── __main__.py        CLI entry point, run loop
│   ├── config.py          TOML config with defaults
│   ├── detector.py        meeting detection (two probes)
│   ├── recorder.py        dual-stream capture, cleanup, alignment
│   ├── audio.py           device readers, DSP, concealment
│   ├── screenshots.py     window capture
│   ├── storage.py         archive backends
│   ├── postprocess.py     deferred diarize + transcribe
│   └── singleton.py       single-instance lock
├── scripts/
│   ├── transcribe.py      Whisper, speaker-labelled transcripts
│   ├── enroll.py          build a voiceprint library
│   ├── diarize.py         cluster and name speakers
│   ├── review.py          local web GUI for corrections
│   ├── realign.py         fix pre-alignment recordings
│   └── install_task.ps1   Task Scheduler registration
├── tests/                 10 files, 242 tests
├── config.toml            per-machine, gitignored
├── config.example.toml    documented template
├── bootstrap.ps1          first-time setup
├── logs/                  rotating log + pid lock + queue
└── recordings/            local staging, emptied after each meeting
```

---

## 3. Install

```powershell
py --version          # 3.11+ required (tomllib)
git --version
Test-Path "G:\My Drive"
```

Then:

```powershell
mkdir "$HOME\dev" -Force | Out-Null
Expand-Archive "meetingrec.zip" -DestinationPath "$HOME\dev" -Force
cd "$HOME\dev\meetingrec"
powershell -ExecutionPolicy Bypass -File .\bootstrap.ps1
```

`bootstrap.ps1` creates the venv, installs the package editable, copies
`config.example.toml` to `config.toml`, runs `git init`, then runs the tests and
a storage self-test.

### Dependency groups

| Extra | Packages | For |
|---|---|---|
| (base) | `soundcard`, `sounddevice`, `numpy` | capture |
| `detect` | `pycaw`, `comtypes`, `psutil` | detection |
| `screenshots` | `mss`, `pillow`, `psutil` | window capture |
| `transcribe` | `faster-whisper` | transcription |
| (manual) | `speechbrain`, `torch`, `soundfile` | speaker ID |
| `dev` | `pytest` | tests |

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[detect,screenshots,transcribe,dev]"
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install speechbrain soundfile
```

Torch needs its own index. Combining both in one command fails with a
misleading "no matching distribution", because `--index-url` *replaces* PyPI
rather than adding to it.

### Shell shortcuts

Add to `$PROFILE`:

```powershell
function meetingrec { Push-Location "<project>"; & .\.venv\Scripts\python.exe -m meetingrec @args; Pop-Location }
function mrupdate   { Push-Location "<project>"; $t = Join-Path $env:TEMP ("mr_" + (Get-Date -Format "HHmmss")); Expand-Archive ".\meetingrec.zip" -DestinationPath $t -Force; Copy-Item "$t\meetingrec\*" . -Recurse -Force; Remove-Item $t -Recurse -Force; & .\.venv\Scripts\python.exe -m pytest -q; Pop-Location }
function transcribe { Push-Location "<project>"; & .\.venv\Scripts\python.exe scripts\transcribe.py --all "<archive>" --me "<your name>" @args; Pop-Location }
function diarize    { Push-Location "<project>"; & .\.venv\Scripts\python.exe scripts\diarize.py --all "<archive>" --library ".\<profiles>\voiceprints.json" @args; Pop-Location }
function enroll     { Push-Location "<project>"; & .\.venv\Scripts\python.exe scripts\enroll.py ".\<profiles>" @args; Pop-Location }
function recreview  { Push-Location "<project>"; & .\.venv\Scripts\python.exe scripts\review.py --archive "<archive>" --profiles ".\<profiles>" @args; Pop-Location }
```

---

## 4. Commands

### `meetingrec <command>`

| Command | Purpose |
|---|---|
| `devices` | List every audio device and probe which ones actually open |
| `storage` | Write a throwaway file to the archive and report where it landed |
| `list` | Mic consent entries + live capture sessions |
| `watch` | Live detection state, no recording |
| `testrecord --seconds N [--mic NAME]` | Fixed-length capture, no detector. `testrecord` never archives |
| `run` | The real thing: detect, record, archive, queue post-processing |

Global flags: `-v` for debug logging, `--log-file PATH` to redirect the log.

### Scripts

```powershell
# transcription
python scripts\transcribe.py <folder|--all root> [--model medium] [--me NAME]
                             [--threads N] [--summarise MODEL] [--force]

# voiceprint library
python scripts\enroll.py <profiles folder> [--verify] [--identify AUDIO]
                         [--threshold 0.65] [--z 4.0] [--min-seconds 3]

# speaker identification
python scripts\diarize.py <folder|--all root> --library <voiceprints.json>
                          [--distance 0.65] [--z 4.0] [--margin 0.10]
                          [--me YOUR_PROFILE] [--force]

# review GUI
python scripts\review.py --archive <root> --profiles <folder> [--port 8765]

# fix old misaligned recordings
python scripts\realign.py <...-mic.wav> --probe
python scripts\realign.py <...-mic.wav> --offset 28.5
```

---

## 5. Workflows

### First run on a new machine

```powershell
meetingrec devices      # 1. note the mic you'll wear; confirm it says [works]
                        # 2. set mic_device in config.toml, that device first
meetingrec storage      # 3. confirm archiving works
meetingrec testrecord --seconds 20
                        # 4. play audio, talk, then LISTEN to all three files
meetingrec list         # 5. during a real call — confirm detection sees it
```

Step 1 matters most. Device selection caused more lost recordings than any
audio problem.

### Daily

```powershell
meetingrec run
```

Leave it running. With `post_process = true` nothing else is needed — meetings
detect, record, archive, and transcribe themselves in the evening window.

### Manual post-processing

If `post_process` is off, after a meeting:

```powershell
diarize        # cluster and name speakers → speakers.json
transcribe     # Whisper both tracks → transcript.md
recreview      # correct anything wrong
```

Order matters: `transcribe` reads `speakers.json`, so `diarize` goes first.

### After correcting names in the GUI

```powershell
enroll                  # pick up the new audio samples
diarize --force         # re-identify with the improved library
transcribe --force      # rewrite transcripts with corrected names
```

### Building the voiceprint library

```powershell
enroll --verify
```

Expects one subfolder per speaker, each containing an audio sample and
optionally a `.md` profile card. A `MasterIndex.csv` in the root supplies
aliases. `--verify` cross-compares every profile and recommends a threshold —
run it before trusting the library.

```powershell
enroll --identify "recordings\<a known recording>-mic.wav"
```

Matches one file against the library. Using a recording of yourself is a free
end-to-end test, since the answer is known.

---

## 6. Configuration reference

`config.toml`, gitignored, one per machine. Every value below is the default.

### `[detection]`

| Key | Default | Notes |
|---|---|---|
| `poll_interval` | `2.0` | Seconds between checks |
| `stop_debounce` | `45.0` | Mute does not release the mic; this covers device switches |
| `require_playback_for_ambiguous` | `true` | Browsers also need an active playback session |
| `detection_source` | `"auto"` | `sessions` \| `consent` \| `auto` |
| `allowlist` | built-in | **Replaces** the defaults entirely |
| `ambiguous` | built-in | Must also appear in `allowlist` |

### `[capture]`

| Key | Default | Notes |
|---|---|---|
| `sample_rate` | `48000` | Requested; devices may negotiate lower |
| `preroll_seconds` | `30.0` | System audio only |
| `min_session_seconds` | `60.0` | Shorter meetings are **deleted** |
| `mic_device` | unset | String or list, in priority order |
| `mic_rate` | unset | Force a capture rate |
| `block_frames` | `1024` | ~21 ms. See finding 2 |
| `warmup_seconds` | `1.5` | Settle time for a lazily-opened mic |
| `lazy_mic` | `true` | Keep the mic closed until a meeting starts |
| `clean_mic` | `true` | High-pass + normalise |
| `highpass_hz` | `300.0` | 250–300 desk mic, 120–150 close mic |
| `keep_raw_mic` | `false` | Also write `*-mic-raw.wav` |
| `conceal_dropouts` | `true` | Bridge short gaps, ramp long ones |
| `normalize_mic_rate` | `false` | Rewrite narrowband tracks at their true rate |

### `[screenshots]`

| Key | Default | Notes |
|---|---|---|
| `screenshots` | `false` | Off by default — it captures your screen |
| `screenshot_follow_foreground` | `true` | Needed to catch your own screen shares |
| `screenshot_foreground_only` | `true` | |
| `screenshot_interval` | `3.0` | How often to look |
| `screenshot_min_gap` | `15.0` | Never save more often than this |
| `screenshot_change_threshold` | `12` | dHash bits 0–64; lower is fussier |
| `screenshot_quality` | `70` | JPEG |
| `screenshot_max_per_meeting` | `200` | |
| `screenshot_exclude` | built-in | Substrings; replaces the defaults |

### `[post_process]`

| Key | Default | Notes |
|---|---|---|
| `post_process` | `false` | Diarize + transcribe automatically |
| `post_process_delay` | `300.0` | Seconds after the meeting ends |
| `post_process_window` | unset | e.g. `"18:00-07:00"`; crosses midnight correctly |
| `post_process_library` | unset | `voiceprints.json`; unset skips diarisation |
| `post_process_me` | unset | Warns if your voice appears on the system track |
| `post_process_mic_label` | `"Me"` | Label for your track in transcripts |
| `whisper_model` | `"medium"` | `tiny`…`large-v3` |
| `whisper_threads` | unset | Default is half the cores |
| `summarise_with` | unset | Local LLM model name, e.g. an Ollama tag |

### `[storage]`

| Key | Default | Notes |
|---|---|---|
| `backend` | `"drive"` | `drive` \| `driveapi` \| `local` |
| `drive_folder` | `"Meeting Recordings"` | |
| `drive_root` | auto | Set with several accounts |
| `group_by` | `"month"` | `month` \| `day` \| `flat` |
| `folder_per_meeting` | `true` | One folder per call |
| `archive_dir` | unset | `local` backend only |
| `drive_credentials` / `drive_token` / `drive_shared_drive_id` | unset | `driveapi` only |

### `[misc]` and derived values

| Key | Default | Notes |
|---|---|---|
| `staging_dir` | `<project>/recordings` | Local only. **Never** point this at a synced folder |
| `machine` | `%COMPUTERNAME%` | Top folder in the archive; override to rename |
| `notify_on_start` | `true` | Placeholder — not yet implemented |

---

## 7. Module reference

### `config.py`

Dataclass with defaults, loaded from TOML. Unknown keys warn rather than fail.
Sections are flattened, so key names must be unique across the whole file.

### `detector.py`

| Export | Purpose |
|---|---|
| `MeetingDetector` | State machine: `idle` → `in_meeting` → `pending_stop` → `idle` |
| `MeetingSession` | One meeting; carries `started_at`, `detection_lag`, `apps_seen` |
| `MicUser` | An app holding the mic, with the device when known |
| `ConsentStoreProbe` | Reads the registry consent store |
| `CaptureSessionProbe` | Reads live WASAPI capture sessions |
| `build_mic_source(mode)` | Picks a probe |

Both probes are injectable, which is how the state machine is tested without
Windows. A re-acquire inside the debounce window continues the same session
rather than starting a second file.

### `audio.py`

| Export | Purpose |
|---|---|
| `SounddeviceReader` | PortAudio input; negotiates the rate |
| `SoundcardReader` | WASAPI loopback; counts driver discontinuities |
| `list_input_devices` / `find_input_devices` | De-duplicated across host APIs |
| `resample` / `downsample` / `lowpass` / `highpass` | DSP |
| `enhance_speech` | High-pass then normalise; returns before/after band shares |
| `conceal_dropouts` | Bridge short gaps by mirroring, ramp long ones |
| `find_dropouts` | Runs of exact zeros |
| `high_frequency_fraction` | Bandwidth measurement over voiced windows |

Host API preference for mics is **MME first**, WASAPI third. See finding 1.

### `recorder.py`

| Export | Purpose |
|---|---|
| `DualRecorder` | Owns both streams; `start()` once, `start_recording`/`stop_recording` per meeting |
| `_Stream` | One device; ring buffer, drift padding, device fallback, silence watchdog |
| `mixdown` | Sum both tracks at the higher rate, with headroom |
| `align_tracks` / `pad_front` | Put both files on one timeline |
| `clean_mic_track` | Conceal, then enhance |
| `rewrite_at_true_rate` | Downsample a narrowband track honestly |
| `list_devices` | Human-readable device survey with probing |

### `screenshots.py`

`ScreenshotCapturer` thread; `arm()`/`disarm()` per meeting. Captures the
foreground window's rectangle, perceptually hashes each frame, saves on change
with a minimum gap. Writes a `-shots.json` manifest.

### `storage.py`

`LocalArchive`, `DriveDesktopArchive`, `DriveApiArchive` behind one `archive()`
interface. `session_subdir()` builds the folder path. `safe_name()` sanitises
meeting titles for filenames.

### `postprocess.py`

`PostProcessor` thread. Queues finished meetings, waits out the delay and the
time window, defers while recording, runs `diarize.py` then `transcribe.py` as
subprocesses at below-normal priority. Suspends rather than kills if a meeting
starts mid-run. Queue persists to disk.

### `singleton.py`

`SingleInstance` — an OS-enforced file lock held for the run. Kernel-level, so
it dies with the process and there is no stale state.

---

## 8. Scripts reference

### `transcribe.py`

Transcribes each track separately and merges on the shared timeline. Reads
`speakers.json` if present and attributes system-track segments by **maximum
overlap** — Whisper follows words, the diariser follows energy, so boundaries
never line up.

Decodes WAV itself rather than letting Whisper open the file, which decouples
it from the transcriber's audio backend.

Writes `transcript.md` (readable, with a talk-time breakdown) and
`transcript.json` (structured).

### `enroll.py`

Builds `voiceprints.json` from a folder of speaker subfolders. Averages
multiple samples per person. Records the model name, because embeddings are
only comparable within the model that produced them.

`classify()` returns three outcomes: **named**, **ambiguous** (a rival is
within the margin), **unknown**. Scoring is by z-score against the library
distribution, not absolute similarity. See finding 5.

### `diarize.py`

Segments the system track on energy, embeds each segment, clusters them,
averages each cluster into a duration-weighted centroid, and matches the
**centroid** — not individual turns. Clusters under 4 seconds are never
matched. Stores centroids in `speakers.json` so unknowns can be compared across
meetings.

### `review.py`

Local web GUI at `127.0.0.1:8765`. Two tabs:

- **Meetings** — each voice with a timeline strip of its turns; click to hear.
- **Unnamed** — unidentified voices grouped across meetings by centroid
  similarity, so naming one labels every meeting that voice appears in.

Correcting a speaker also writes their audio into the profile library as a new
sample recorded under real meeting conditions, so the library improves with use.

### `realign.py`

Repairs recordings made before automatic alignment. `--probe` writes candidate
mixes at several offsets; `--offset N` applies the one you picked. There is
deliberately no automatic detection — see finding 7.

---

## 9. Output layout

```
<archive root>/<MACHINE>/<year>/<month>/<timestamp>-<app>/
    20261016-120116-zoom.exe-mic.wav         your voice
    20261016-120116-zoom.exe-system.wav      everyone else
    20261016-120116-zoom.exe-mix.wav         both, for playback
    20261016-120116-zoom.exe-mic.wav.json    metadata sidecar
    20261016-120116-zoom.exe-shot001.jpg     screenshots
    20261016-120116-zoom.exe-shots.json      screenshot manifest
    speakers.json                            who spoke, when, confidence
    transcript.md                            readable transcript
    transcript.json                          structured transcript
```

Machine-first because several machines share one archive. One folder per
meeting because screenshots otherwise interleave every call's files.

Local staging is `recordings/`, emptied after each meeting. Files left there
mean archiving failed, and the log says so.

### `speakers.json`

```json
{
  "clusters": 4,
  "speakers": [
    { "label": "<name>", "name": "<name>", "confidence": 0.78, "z": 8.5,
      "margin": 0.47, "status": "named", "seconds": 890.0,
      "embedding": [...], "turns": [{"start": 12.4, "end": 31.0}] }
  ]
}
```

`status` is `named`, `ambiguous`, `unknown` or `too-short`.

---

## 10. Running unattended

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_task.ps1
Start-ScheduledTask -TaskName meetingrec
Get-Content .\logs\meetingrec.log -Wait -Tail 20
```

Remove with `.\scripts\install_task.ps1 -Remove`; keep the console with
`-ShowConsole`.

**Never a Windows service, and never "run whether the user is logged on or
not".** Both detection signals live in the interactive session: the consent
store is under `HKCU`, and audio endpoints are per-session. Session 0 sees an
empty hive and no devices, and records nothing, silently.

**No restart policy.** Task Scheduler treats any exit as a failure worth
retrying, including a deliberate stop — so a restart count means every manual
stop spawns a replacement and you get two recorders fighting over the audio
devices.

---

## 11. Findings and design decisions

Each of these was measured, counter-intuitive, and changed the output more than
anything else.

### 1. Mics through MME, not WASAPI

Raw WASAPI bypasses the Windows audio engine and with it the endpoint's own
capture processing.

| Path | Below 300 Hz | Speech band |
|---|---|---|
| WASAPI | 54% | 34% |
| MME | 14% | 77% |

Same device, same minute, before any filtering. Loopback stays on WASAPI.

### 2. Small capture buffers, not large

| Block size | Digital silence |
|---|---|
| 256 | 62% |
| 1024 | 0.2% |
| 4096 | 9–26% |

A large buffer starves between Bluetooth packets and returns zeros rather than
waiting. The instinct that bigger is safer is backwards here.

### 3. Narrowband is a protocol ceiling

A Bluetooth headset in call mode carries ~16 kHz. No library or filter recovers
what was never transmitted. A commercial recorder measured *narrower* than this
build (0.01% vs 0.35% above 7 kHz) and sounded better — the difference was
processing, not bandwidth.

### 4. Rumble, not bandwidth, is what sounds bad

Measured on a real capture: 88% of energy below 300 Hz, 8% in the speech band.
High-pass then normalise moved the same file to 63% speech band. Normalising
without the high-pass just amplifies the rumble.

### 5. Score speakers by z-score, not similarity

On a 41-voice library, the closest impostor pair scored **0.55** while a
speaker's own voice from a meeting matched his own profile at **0.455**. Those
numbers are not comparable — one is same-channel, the other cross-channel.

What survives a channel change is the *lead* over the field. That 0.455 sat 4.4
standard deviations above a field averaging 0.095. Threshold on z ≥ 4.0, with
an `ambiguous` verdict when a rival is within 0.10.

Measured outcomes that set the threshold:

| z | Correct? |
|---|---|
| 8.6 | yes |
| 6.2 | yes |
| 5.0 | yes |
| 4.4 | yes |
| 3.8 | **no** |

### 6. Cluster before matching

A single turn is 2–3 seconds and its embedding is noisy. Six turns of four
seconds become one centroid built from 24 seconds, which is where
identification measurably works. Averaging also cancels one-off problems — a
turn off-mic, crosstalk, a cough.

### 7. Things tried that did not work

**Band-limiting enrollment audio** to match meeting conditions. Measured: the
correct match fell from 0.455 to 0.248, because it removed the detail the
encoder relies on.

**Automatic offset detection** for misaligned recordings. Correlating the two
tracks' loudness envelopes locks onto conversational turn-taking; it reported
−3 s for a pair built with exactly zero offset. A second attempt gating on
quiet passages killed the signal entirely. Removed — a conversation shares no
audio between tracks unless there is bleed.

**PID-file locking**, three times. `exists()` then create has a race;
`O_CREAT|O_EXCL` is atomic but the loser cannot tell stale from live, and
delete-and-retry means everyone deletes everyone else's lock; PID liveness
checks break on recycling, and `os.kill(pid, 0)` *terminates* the process on
Windows. The answer is a kernel file lock.

### 8. Two habits worth keeping

**Measure, don't infer.** Nearly every wrong turn came from reasoning about
what numbers implied instead of comparing two files through the same code path.
One ten-second sweep across four host APIs found in a single run what days of
theorising about codecs had missed.

**Be suspicious of a flaky test.** Three times an intermittent failure looked
like timing and was a real race. The instinct to loosen a timeout is usually
wrong.

---

## 12. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `python not recognized` | Store alias | Use `py`, or disable the alias in Settings → Apps → Advanced |
| Detection never fires | Consent store stale on this build | `detection_source = "sessions"` |
| Mic track silent | Device opened but dead (headset off) | Check `mic_device` order; the watchdog logs it |
| System track silent | Output went elsewhere | Set your headset as the Windows default |
| Heavy dropouts | Bluetooth congestion | Move the dongle off USB 3; prune paired devices |
| Muffled audio | `highpass_hz` too low | Raise toward 300 |
| Thin, telephone-ish | `highpass_hz` too high | Lower toward 150 |
| Tracks out of sync | Pre-alignment recording | `realign.py --probe` then `--offset` |
| Nothing archived | Under `min_session_seconds` | Lower it; short meetings are deleted by design |
| Two recorders | Task restart policy | Re-register the task; the lock also refuses the second |
| Device-connect chimes | Two instances, or endpoint churn | Check for duplicates; prune Bluetooth pairings |
| Speaker named wrongly | That person is not enrolled | Enroll them; raise `--z` or `--margin` |

### What is not solved

Bluetooth packet loss is environmental. Expect a few percent of any recording
to be reconstructed, with bursts when the 2.4 GHz band is busy. Concealment
hides the clicks; it cannot recover lost words.

A crowded Bluetooth stack degrades everything at once. If audio endpoints keep
invalidating, count the paired devices before blaming the capture code —
duplicate mouse input and audio dropouts can share one cause.

---

## 13. Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

242 tests, none needing audio hardware, a network, or Windows-only APIs beyond
what is stubbed. Device readers and detection probes are injectable; the
recorder's streams are driven by hand.

| File | Covers |
|---|---|
| `test_detector.py` | State machine, debounce, probe selection, session timing |
| `test_recorder.py` | Ring buffer, drift padding, fallback, watchdog, alignment, mixdown |
| `test_storage.py` | Archive backends, folder layout, name sanitising |
| `test_screenshots.py` | Change detection, exclusions, manifest |
| `test_transcribe.py` | Merge, attribution, WAV decoding, formatting |
| `test_enroll.py` | Card parsing, library round-trip, z-score classification |
| `test_diarize.py` | Segmentation, clustering, centroids, naming |
| `test_review.py` | Payloads, clip slicing, enrollment feedback |
| `test_postprocess.py` | Queue, deferral, suspension, windows, persistence |
| `test_singleton.py` | Lock contention, crash recovery, Windows mandatory locks |

Several tests exist specifically to prevent a bug returning — the Thread
attribute shadowing guards, the z-threshold regression, the lock race, the
loopback reopen loop. Those are worth keeping even though they look redundant.
