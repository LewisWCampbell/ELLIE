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

## Contract v3: the model decides, the system supplies the facts (supersedes the table above for the voice model)

The first trained model (v1) was evaluated on 2026-11-14. It chose the right
tool at the right point most of the time and heard times of day correctly, but
it invented callers' names and emails (it hears audio codes, not words), it
sometimes invented tool names or broke the JSON, and it booked without a proper
yes in a third to half of the cases. v3 removes each of those from the model's
job:

**1. The voice model never writes a name, an email or free text.** Its tools:

| Tool | Kind | Argument | Result (text) |
|---|---|---|---|
| `check_availability` | read | `start` | `FREE: …`, `BUSY: … Nearby open times: …`, `UNKNOWN: …` (as before) |
| `get_my_bookings` | read | none | as before |
| `get_caller_details` | read | none | `ON FILE: <name>, <email spoken>.` (from the caller's history), `HEARD: <name>, <email spoken>.` (taken from this call's transcript by the system), or `MISSING: <what>. Ask for it.` |
| `book_intro_call` | write | `start` | `Done: …tentative…`, slot-taken, error (as before), or `NOT BOOKED: <reason>. <what to do>` from the booking guard |
| `record_sms_consent` | write, silent | none | `Saved.` |

`save_user_fact` is gone from the voice model: remembering a caller's name,
company or reason for calling is done by the system from the transcript.
`topic` is gone from bookings.

**2. Who the caller is comes from their history.** Episode `context.caller_on_file`
is `{"name": …, "email": …}` for a returning caller, else `null`. The system
prompt says so (`{caller}` in `prepare.prompt_text`): "The caller is Rosa
Greenleaf, a returning caller." or "The caller is not on file." Elle does not
ask a returning caller for their name or email. For a new caller she asks for
both, as before.

**3. Facts are read back from a result, never from memory.** Before any readback
Elle calls `get_caller_details`; the readback is a claim that depends on that
result (and on the availability result for the time). The readback states the
time with the timezone, the name and the email exactly as the result gives
them.

**4. Compact calls.** The text stream carries
`<tool>NAME</tool>` or `<tool>NAME ARG</tool>`: the tool name, then for
`check_availability` and `book_intro_call` one space and the local start time
as `YYYY-MM-DDTHH:MM` (America/Los_Angeles; the system adds the offset). No
JSON, no id. Calls are numbered by the system in order (`c1`, `c2`, …) and the
result keeps its form `<result id="c1">…</result>`. Episode events keep
`{"id","name","args"}` with `args.start` as full ISO 8601 as before; only the
serialization changes (`serialize.tool_call_text`). A call of this form is at
most about 15 tokens (was about 50), so it takes about a second, not four.
Anything that is not exactly one of the five names with a well-formed argument
is rejected by the parser (`tool_error`) and answered with
`<result id="cN">ERROR: unknown tool. Use one of: …</result>`.

**5. The booking guard (code, not the model).** `book_intro_call` is executed
only if, in this call: (a) the most recent `check_availability` for that same
start returned `FREE`; (b) `get_caller_details` returned `ON FILE` or `HEARD`;
(c) after both, Elle read the details back and the caller then said yes (the
system knows from the transcript; in evaluation, from the episode's caller
`confirm` event); (d) that start is not already booked in this call. Otherwise
the result is `NOT BOOKED: …` and nothing is written. `record_sms_consent` is
executed only after the consent question and a yes.

The first-run table and JSON format above describe v1 data and production's
text tools; `serialize.py` and the parser accept both forms (`format: json |
compact`), so v1 artefacts stay readable.

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

### Where a call sits, and what comes before the conversation (v2)

A call that the episode places inside one of Elle's lines ("Let me check...")
is written right after that line's words, which the text stream emits from the
line's first word (text runs ahead of audio, as in Moshi). Starting the call
then coincides with starting the line that announces it; at its literal time
it would be a one-frame guess in the middle of the line that the model cannot
learn. `prepare.anchor_calls_to_utterance`.

Every training window starts with the system prompt, in PersonaPlex's own
style and layout: 6 PAD frames, the prompt tokens, 6 PAD frames, with Mimi
silence codes on Elle's channel and a sine tone on the caller's
(`prepare.prompt_audio: native`, the codes in PersonaPlex's `lm.py`). The
prompt carries the call's date and time, since a booking needs an absolute
date and the caller says "Thursday at 3":

```
<system> You work for Lewis as his phone assistant and your name is Elle. Information: You answer questions about Lewis and book thirty-minute intro calls with him. Today is Friday, March 5, 2027, 12:28 PM Pacific Time (UTC-08:00). <system>
```

Anything that runs the model (evaluation, a future server) must reproduce this
block exactly, frame for frame: the evaluation forces the example's own prompt
frames, text and audio, and `test_personaplex` checks that the streaming
generator then computes the same probabilities as training.

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
