# Phone ↔ Desktop shared contract (schema 1)

Everything lives under `My Drive/Meeting Recordings/` (same root as meetingrec).

## Recordings
`<MACHINE>/<YYYY>/<MM>/<slug>/`
- `<slug>.mp3`: audio (phone uploads; slug ends in `-plaud`). `<slug>-phone-excerpt.txt`: phone Whisper excerpt (first 10 min, used for titling). meetingrec keeps `-mic/-system/-mix.wav`.
- `<slug>.meta.json`: sidecar. Fields: `slug, sessionId, deviceSn, machine, startedAt,
  durationSec, source (plaud|phone|teams|zoom|meet|voip), title, tags[], titledBy,
  status (new|titled|transcribed), flags[], updatedAt`.
  Desktop sets `status = "transcribed"` after its run and may fill `title`/`tags`.

## Shared state: `_shared/`
- `dictionary.json`: `{schema, terms:[{term, aliases[], note, updatedAt, updatedBy}]}`.
  Both sides feed `term` + `aliases` into Whisper's initial prompt.
- `speakers.json`: `{"names": ["Laura Eich", ...]}`. **Desktop exports this** from its speaker
  library (BIGGBY_Speaker_Profiles + MasterIndex.csv) after each run; the phone reads it for
  autocomplete. The phone changes labels only via events.
- `events/<DEVICE>/<yyyyMMdd-HHmmss-SSS>.json`: JSON array of events. **Each device writes only
  its own folder**, so offline edits never collide.

### Event
```json
{"id":"uuid","schema":1,"ts":"2026-10-24T14:05:00-04:00","device":"PHONE-SM-F976U",
 "type":"label_correction","recording":"20261024-140530-plaud",
 "payload":{"from":"Speaker 2","to":"Laura Eich"}}
```
Types: `label_correction` (one recording), `speaker_rename` (global), `term_add`, `term_remove`,
`flag` (`kind`, `note`), `title_set` (`title`, `tags[]`).

## Desktop duties on each transcription run (meetingrec TODO)
1. Read all `events/*/*.json` not yet in its applied-ledger (by event `id`), in `ts` order.
2. Apply: term changes → dictionary.json; label/speaker events → transcript + speaker library;
   `flag` → surface in review.py.
3. Re-export `_shared/dictionary.json` and `_shared/speakers.json` so phones pick up the result.
4. Record applied ids locally (never delete other devices' files).
5. Transcribe phone recordings (`PLAUD-*` folders) like its own `-mix.wav`, then mark them `transcribed`.
