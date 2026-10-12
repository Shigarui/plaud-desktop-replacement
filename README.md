# meetingrec

Detects when a virtual meeting starts on Windows, records desktop audio and
microphone as separate tracks, cleans them up, and archives to Google Drive.

New machine? See **INSTALL.md**.

## Status

| Piece | State |
|---|---|
| detection (mic consent store) | working |
| dual-stream capture, pre-roll, drift correction | working |
| device fallback + silence watchdog | working |
| speech cleanup, packet-loss concealment | working |
| track alignment | working |
| Drive archiving | working |
| tray icon, auto-start | not built |
| transcription handoff | not built |

70 tests, none needing audio hardware or Drive.

## Commands

```powershell
py -m meetingrec devices      # what the machine has, and what actually opens
py -m meetingrec storage      # prove the archive target works
py -m meetingrec list         # mic consent entries; run during a call
py -m meetingrec watch        # live detection state
py -m meetingrec testrecord --seconds 20 [--mic NAME]
py -m meetingrec run          # the real thing
```

## How detection works

Windows logs every microphone acquisition to the capability-access consent
store in `HKCU`. An entry whose `LastUsedTimeStop` is `0` has the mic open
right now. Polling that beats scraping window titles and needs no elevation.

Browsers are ambiguous — mic access could be a meeting or a dictation box —
so they also require an active playback session. If pycaw is missing they pass
through: over-recording is recoverable, a missed meeting is not.

Muting does not release the mic, so mute is not mistaken for leaving.

## Capture notes, all learned the hard way

**Mics go through MME, not WASAPI.** Raw WASAPI bypasses the Windows audio
engine and with it the endpoint's own capture processing. On a Bluetooth
headset that was 34% speech-band energy versus 77% through MME, before any
filtering. Loopback still uses WASAPI via soundcard, which it handles well.

**Block size is 1024 frames, not 4096.** Large buffers starve between HFP
packets and return digital silence: 9-26% versus 0.2% on the same device.

**The mic opens only when a meeting starts.** Holding a Bluetooth mic open
forces HFP, which would degrade all playback all day.

**A freshly opened stream settles for 1.5s** before the file opens. Measured
~20% dropouts in the first five seconds, 0.0% for the next 45.

**Streams that go silent are treated as dead.** A dongle whose headset is off
opens fine and records nothing; the watchdog moves to the next device.

**Tracks are aligned afterwards.** System audio carries 30s of pre-roll and
the mic ~1.5s, so without correction the two files describe different stretches
of time and replies arrive before questions.

## Storage

Capture always writes to a local staging directory, never straight to Drive —
a sync client locking the file mid-write stalls the writer thread and loses
audio. Archiving happens after the meeting and may fail without taking the
recording with it.

Files land in `<machine>/<year>/<month>/`, machine-first, with a JSON sidecar.

## What is not solved

Bluetooth packet loss is environmental. Expect a few percent of any recording
to be reconstructed, with bursts to 20% when the 2.4 GHz band is busy.

HFP narrowband is a hard ceiling on any Bluetooth headset — roughly telephone
quality. A wired USB mic is the only way past it.

## Recording consent

Recording meetings has legal implications that vary by state and employer.
Build the habit of saying so at the top of a call.
