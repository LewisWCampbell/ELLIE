# Training contract: tool calls in a full-duplex voice model

This is the agreement between the dataset (`training/data/`) and the trainer
(`training/model/`). Either side may add fields; neither may change or remove
one without updating this file and the other side.

## Goal

Fine-tune a Moshi-family full-duplex model (PersonaPlex is the candidate) so it
emits tool calls in its inner-monologue text stream and resumes speaking when
the result is injected back into that stream. Two kinds of call matter, and the
data must contain both:

- **Reads** (`check_availability`, `get_my_bookings`): no side effects. Elle
  must fill the wait naturally and must never state the answer before the
  result arrives.
- **Writes** (`book_intro_call`, `save_user_fact`): side effects. Elle must have
  everything required, read it back, get a clear yes, and only then call.
  `save_user_fact` is silent: she never narrates it.

## Tools (must match `src/booking/booking.ts`, `src/sms.ts`, `src/realtime/twilioBridge.ts`)

| Tool | Kind | Arguments | Result shape (text, as the production tool returns it) |
|---|---|---|---|
| `check_availability` | read | `start` (ISO 8601 with offset) | `FREE: …`, `BUSY: … Nearby open times: …`, or `UNKNOWN: …` when the calendar can't be read (never spoken as free) |
| `get_my_bookings` | read | none (scoped to the caller's number) | list of upcoming bookings, or none. Production exposes it over SMS only today; voice episodes assume it is added to the call too. |
| `book_intro_call` | write | `name`, `email`, `start`, `topic?` | `Done: …tentative…` or `Error: …` or slot-taken |
| `save_user_fact` | write, silent | `category`, `fact` | `Saved.` |

Times are in the owner's zone, America/Los_Angeles; a call is 30 minutes.

## Episode format (JSONL, one episode per line)

```json
{
  "id": "ep_000123",
  "split": "train | val | test",
  "scenario": "book_after_availability_busy",
  "seed": 123,
  "now": "2026-11-09T10:15:00-08:00",
  "events": [
    {"t": 0.00, "dur": 2.1, "ch": "caller", "type": "speech", "text": "Is he free Thursday at two?"},
    {"t": 2.45, "dur": 1.2, "ch": "elle",   "type": "speech", "text": "Let me check his calendar."},
    {"t": 2.50, "ch": "elle",   "type": "tool_call",   "id": "c1", "name": "check_availability", "args": {"start": "2026-11-12T14:00:00-08:00"}},
    {"t": 3.30, "ch": "system", "type": "tool_result", "id": "c1", "content": "FREE: …", "latency_s": 0.8},
    {"t": 3.70, "dur": 2.6, "ch": "elle",   "type": "speech", "text": "He's free then. Want me to pencil you in?"}
  ],
  "labels": {
    "expected_calls": [{"name": "check_availability", "args": {"start": "2026-11-12T14:00:00-08:00"}}],
    "claim_spans": [{"event_index": 4, "depends_on": "c1"}],
    "forbidden_before_result": true
  }
}
```

- `t` and `dur` are seconds from episode start. Speech on the two channels may
  overlap (that is the point of full duplex).
- `claim_spans` marks every Elle speech event whose content depends on a tool
  result. Each must start after that result's `t`; an episode where it does not
  is a negative example and must say so in `scenario` and `labels.negative: true`.
- Every `tool_result` follows its `tool_call` and carries the `id`.

## Inner-monologue serialization (what the model actually sees in its text stream)

```
<tool>{"id":"c1","name":"check_availability","args":{"start":"…"}}</tool>
<result id="c1">FREE: …</result>
```

The tool-call span is emitted by the model on Elle's text stream at the event's
`t`. The result span is injected (forced, not predicted) at the result's `t`;
its tokens are masked out of the loss. These four marker strings are added to
the tokenizer as special tokens: `<tool>`, `</tool>`, `<result id=`, `</result>`.

## Spans take time (added after the first dry run)

The model's text stream carries one token per 80 ms frame (12.5 Hz), so a tool
call or a result is not instantaneous. Tokenized, a median call is about 50
tokens (about 4 s to write) and a median result about 28, up to ~110. Taken
literally, the event times above would put a result before its call finished
and Elle's spoken answer on top of the result's injection, which teaches the
premature claims this work exists to prevent.

So training uses retimed episodes (`training/model/prepare.py`, `retime_episode`):
a result begins no earlier than the end of its call span plus the tool's
latency, and every later event shifts to after the injection ends. Metric
verdicts are unchanged by retiming. **Audio must be rendered from retimed
episodes**, never from raw ones. A shorter call format (no `id`, no seconds in
`start`) would cut call length by about 30% and is the next change to try.

## Audio rendering (optional stage, needs TTS credentials)

Stereo 24 kHz WAV per episode: channel 0 is Elle, channel 1 is the caller,
which is the Moshi convention (model on the first channel). Text alignments for
Elle's channel are written next to it in the trainer's expected JSON format.

## Metrics the evaluation must report

1. Tool selection accuracy (right tool, or correctly no tool).
2. Argument exact match, with `start` compared as an instant, not a string.
3. **Premature claim rate**: share of claim spans spoken before their result.
   This is the gating metric for reads; the target is ~0.
4. Write safety: share of writes made without the required fields or without a
   confirmed readback.
5. Wait handling: share of reads whose result latency exceeds 0.7 s where Elle
   says nothing during the wait (dead air).
