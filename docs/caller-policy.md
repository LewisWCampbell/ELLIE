# Repeat-caller escalation (cost protection)

Elle already recognizes callers by number (memory keyed on the phone number).
This adds an escalation ladder so someone who dials over and over — e.g. to run
up your per-minute cost — gets progressively less service.

## The ladder

Counting how many times a number has called within `ABUSE_WINDOW_HOURS`:

| Calls in window | Tier | What happens |
|---|---|---|
| ≤ `ABUSE_LIMITED_AFTER` (default 3) | **normal** | Full experience. |
| past that (default 4–6) | **limited** | The bot is told to keep the call short and steer them to email; the barge-in call is **hard-capped** at `ABUSE_LIMITED_MAX_SECONDS` (default 120s). |
| past `ABUSE_BLOCK_AFTER` (default 6) | **blocked** | A brief spoken message, then hang up — **the paid Deepgram/LLM/TTS pipeline is never started**, so a blocked call costs essentially nothing. |

Thresholds count the current call, so it escalates *with each call*: the 4th
call is limited, the 7th is blocked (with the defaults).

## Where it's enforced

- **`callerPolicy.ts`** — `assessCaller(from)` returns the tier from
  `db.countRecentCalls()`.
- **Realtime (`/voice-realtime`)** — blocked callers get `<Say>` + `<Hangup>`
  before the media stream opens (no cost). Otherwise the tier is passed to the
  bridge as a stream parameter; a limited caller gets the "keep it short" note
  appended to Elle's instructions plus the duration cap (closing the
  `<Connect>` stream ends the call).
- **Cascaded (`/voice`)** — blocked callers get `<Say>` + `<Hangup>`. (The
  turn-based path has no per-minute audio model, so blocking is the meaningful
  control there.)

Offline test (no phone, no network): `npm run verify:policy`.

## Tuning

All via env (see `.env.example`): `ABUSE_WINDOW_HOURS`, `ABUSE_LIMITED_AFTER`,
`ABUSE_BLOCK_AFTER`, `ABUSE_LIMITED_MAX_SECONDS`. To effectively disable it, set
the thresholds very high.

## Notes

- Anonymous / withheld numbers arrive as `unknown` and share one bucket, so
  repeated anonymous calls escalate together. That's usually what you want for
  abuse control, but it does lump distinct anonymous callers.
- Caller ID can be spoofed. This is a cost-blunting measure, not authentication.
  For hard guarantees, add Twilio-level limits (e.g. per-number rate limits or a
  verified-callers allowlist) in front of the app.
