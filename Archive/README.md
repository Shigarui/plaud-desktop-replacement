# meetingrec

Detects when a virtual meeting starts on Windows, records desktop audio and
microphone as separate tracks, and archives the result to Google Drive.

## Status

| Piece | State |
|---|---|
| `detector.py` — meeting detection | working, tested |
| `storage.py` — local / Drive desktop / Drive API | working, tested |
| `config.py` — per-machine settings | working |
| `recorder.py` — dual-stream WASAPI capture | not written yet |
| tray UI, transcription handoff | not started |

## Setup

```powershell
cd $HOME\dev\meetingrec
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -e ".[detect,dev]"
Copy-Item config.example.toml config.toml
```

If `Activate.ps1` is blocked:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## First run, in this order

```powershell
py -m meetingrec list      # 1. confirm the app allowlist on THIS machine
py -m meetingrec storage   # 2. confirm the archive target actually works
py -m meetingrec watch     # 3. join a test call, watch the state changes
```

Step 1 matters most. Every machine reports meeting apps slightly differently —
new Teams shows as a packaged identity, Zoom has shipped under several paths.
Run it *during* a live call and look for `<== MIC OPEN NOW`. If your app isn't
in `DEFAULT_ALLOWLIST`, add it to `allowlist` in config.toml.

Step 2 writes a throwaway file to `_selftest/` in your archive and reports
where it landed. Do this before trusting it with a real recording.

## How detection works

Windows logs every microphone acquisition to the capability-access consent
store in `HKCU`. An entry whose `LastUsedTimeStop` is `0` has the mic open
right now. Polling that is far more reliable than scraping window titles, and
needs no elevation.

Browsers are ambiguous — mic access could be a meeting or a dictation box — so
those also require an active WASAPI playback session (via `pycaw`) before they
count. If pycaw is missing, browsers are allowed through: over-recording is
recoverable, a missed meeting is not.

Muting does not release the mic handle, so mute is not mistaken for leaving.
The 45s stop debounce covers device switches and tab reloads, and a re-acquire
inside that window continues the same session instead of starting a new file.

## Storage

Capture always writes to a **local** staging directory. Never point staging at
Drive or any synced folder — a sync client locking the file mid-write stalls
the writer thread, and a stalled writer during a call loses audio. Archiving
happens after the session ends and is allowed to fail without taking the
recording with it.

Three backends, set `backend` in config.toml:

- **`drive`** (default) — Google Drive for Desktop. Archiving is a filesystem
  move into `G:\My Drive\Meeting Recordings\`; Google's client uploads. No
  OAuth, no credentials to distribute. Start here.
- **`driveapi`** — Drive API v3 resumable upload. Gives file IDs and metadata,
  works on machines without the Drive client. Needs a Cloud project, an OAuth
  client (Desktop app type), and `client_secret.json`.
  **Gotcha:** if the consent screen is *External* and still in *Testing*,
  refresh tokens die after 7 days. Publish it, or use an Internal Workspace
  consent screen.
- **`local`** — plain directory. Set `archive_dir`.

Files land under `<machine>/<year>/<month>/`. Machine-first because three
machines writing into one Drive folder will otherwise collide, and because
you'll want to know which box produced a recording.

## Tests

```powershell
py -m pytest
```

12 tests, none of which need Windows or Drive — the detector's mic and
playback probes are injectable, and storage tests use `tmp_path`.

## Recording consent

Recording meetings has legal implications that vary by state and by employer
policy. `notify_on_start` is on by default; leave it on.
