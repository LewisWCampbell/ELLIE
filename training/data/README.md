# training/data: synthetic tool-call episodes for Elle

This directory builds the dataset side of [`../CONTRACT.md`](../CONTRACT.md): two-channel
episodes (Elle and a caller) with tool calls, results, latencies and overlap. They
are used to fine-tune a Moshi-family full-duplex model (PersonaPlex is the
candidate) so it emits tool calls in its inner-monologue text stream.

Everything runs on the Python 3.11 standard library, with no network and no API keys.

```
python3 training/data/generate.py                 # writes out/sample_{train,val,test}.jsonl + out/STATS.md
python3 training/data/validate.py training/data/out/sample_*.jsonl --holdout-family slot_conflict
python3 -m unittest discover training/data        # 69 tests, about 3 s
python3 training/data/serialize.py training/data/out/sample_val.jsonl 2   # look at the text stream
```

## Files

| file | what it is |
|---|---|
| `generate.py` | CLI and library entry point: `build_episode(scenario, seed)`, `generate_split(...)`, the STATS.md writer |
| `scenarios.py` | the scenario grammar: beats (opening, check, details, readback, book, SMS consent, ...) and the 33 registered scenarios |
| `episode.py` | `Ep`, the timeline builder: speech timing, overlap and barge-in, tool calls and results, fillers, claim spans and write gates |
| `banks.py` | hand-written paraphrase banks (English and Spanish), fictional names, companies and topics |
| `elle_time.py` | the owner's zone, spoken time phrases, `resolve()` against `now` (DST-correct), and ports of the production result formats |
| `validate.py` | the contract checker (stdlib only, no sibling imports) |
| `serialize.py` | episode to inner-monologue text stream plus loss mask (stdlib only, no sibling imports) |
| `expand.py` | optional LLM paraphrase pass; passthrough with no key |
| `render_audio.py` | optional audio stage: pluggable TTS (Deepgram Aura intended), stereo 24 kHz WAV and alignment JSON |
| `test_*.py` | determinism, every scenario validates, validator catches each violation, serializer markers and mask, DST |
| `out/` | the sample: 2000 train, 200 val, 200 test (about 9 MB) and `STATS.md` |

## Generating

```
python3 training/data/generate.py --out training/data/out --prefix sample_ \
    --train 2000 --val 200 --test 200 --seed 7 \
    --holdout-family slot_conflict --holdout-share 0.3
```

- **Deterministic.** Each episode's randomness is `random.Random(f"{base_seed}:{scenario}:{seed}")`.
  Python hashes string seeds with SHA-512, so the same arguments give the same bytes on any
  machine (a test checks this).
- **Splits are disjoint by (scenario, seed).** Train seeds start at 0, val at 1,000,000 and test at 2,000,000.
- **Held-out family.** One family is never in train or val and makes up about 30% of test.
  The default is `slot_conflict` (`book_slot_taken_race`, `book_busy_at_write`): the write comes
  back "taken" after a clean readback and yes. The test asks whether the model can combine skills it
  did learn (BUSY handling on reads, honest error recovery, re-checking before a second readback).
  `--holdout-family none` turns this off. For a production run you may want to rotate the held-out
  family, or hold out nothing, so that no required behaviour is absent from training.
- **Scale.** About 1,500 episodes a second. 100k episodes is roughly 370 MB of JSONL.

## Episode format

This follows the contract exactly. It also adds fields, which the contract allows ("either side may
add fields"):

- Speech events: `act` (dialogue act, e.g. `readback`, `confirm`, `filler`, `claim`, `pre_call`,
  `barge_in`, `backchannel`), `lang` (when not English), `refers` (on readbacks: the `start`, `name`
  and `email` read back), and on a cut-off utterance `interrupted: true` plus `full_text`. On a
  cut-off line, `text` holds only the words actually spoken and `dur` is the spoken part.
- Top level `context`: `channel`, `caller_number` (always `+1NNN55501NN`), `notes` (prior facts about
  this phone number, as production injects them) and `notes_names`.
- `labels`: `negative` and `negative_kind` (`premature_claim` | `write_without_readback` | `dead_air`),
  `write_gates` (`[{call_id, readback_index, confirm_index}]`), `family`, `lang` (`en` | `es` | `en+es`),
  `kind` (`read` | `write` | `mixed` | `none`), `held_out_family` (test only), and `expanded` (set by expand.py).

Tool result strings mirror production word for word: `src/booking/availability.ts` (FREE, BUSY with
the ported `nearbyOpenings`, UNKNOWN), `src/booking/booking.ts` (Done, busy at write, slot-lock race),
`src/realtime/twilioBridge.ts` (`Saved.` and the booking error), and `src/sms.ts` (`get_my_bookings` rows,
`fmtWhen`). Times are spoken to the caller with the zone, and every `start` is ISO 8601 with the correct
Los Angeles offset for that date. About 25% of episodes have `now` in the 10 days before a DST change.

### Time phrases

The caller's phrase is generated from a structured `TimeRef`. `elle_time.resolve(ref, now)`
independently maps it back to an instant, and the generator asserts that the two agree.
These are the conventions:

- "Thursday" means the next Thursday strictly after today. Generated only for 1 to 6 days ahead.
- "next week on Tuesday" means Tuesday of the following Monday-to-Sunday week.
- "the 14th" means the next 14th strictly after today. Generated only within 27 days.
- "November 19th" means this year's, or next year's if it has passed.
- A bare hour follows business hours: 1 to 6 means p.m., 7 to 11 means a.m., and 12 means noon.

Elle's readbacks always say "a.m." or "p.m." and "Pacific time".

## Serializer API (for the trainer)

```python
from training.data.serialize import serialize_episode, to_text_and_mask, extract_tool_calls, SPECIAL_TOKENS
# or: sys.path.insert(0, "training/data"); import serialize

segs = serialize_episode(ep)                       # list[Segment]
segs = serialize_episode(ep, word_level=True)      # one Segment per word, times spread over dur
segs = serialize_episode(ep, include_caller=True)  # caller words too, forced (not predicted)
text, mask = to_text_and_mask(segs)                # mask[i] True = char i is predicted (in the loss)
```

`Segment(t, end, text, is_forced, kind, event_index)`, sorted by `t`. Ties keep event order.

| kind | text | is_forced | at |
|---|---|---|---|
| `speech` | Elle's words | False | speech `t` |
| `tool_call` | `<tool>{"id":"c1","name":"check_availability","args":{"start":"…"}}</tool>` | False (predicted) | the call's `t` |
| `tool_result` | `<result id="c1">FREE: …</result>` | True (masked) | the result's `t` |
| `caller` | caller words (only with `include_caller`) | True | speech `t` |

The tool-call JSON is compact, with keys in the order `id`, `name`, `args`. `SPECIAL_TOKENS` is the four
marker strings to add to the tokenizer. With `word_level=True` a call that lands mid-utterance is
interleaved between the surrounding words, which is where it sits in a time-aligned 12.5 Hz text
stream. The whole result span, markers included, is forced. `extract_tool_calls(text)` parses calls
back out of generated text, returning unparseable bodies as `{"raw", "error"}`, and
`strip_markers(text)` returns just the spoken words. Both are there for the evaluator.

## Validator

`validate_episode(ep) -> [Issue(code, message)]` and `violations(ep) -> {kinds}` are importable.
It checks the following:

- **Schema.** Top-level and per-event-type fields, `now` with an offset, events sorted by `t`, and no
  speaker overlapping their own previous line.
- **Ids.** Call ids are `c1..cN` in time order. Each call has exactly one result, which comes after it
  and carries the same id, and `t = call.t + latency_s`. Episode ids increase within a file and are unique across files.
- **Arguments.** Exact argument sets per tool. `start` must parse, have an offset, use the correct LA
  offset for its date (this catches DST bugs), be strictly after `now`, be no more than 365 days out,
  and land on :00 or :30. Emails must match the production `EMAIL_RE`.
- **`expected_calls`.** Must equal the tool calls, in order.
- **Claim spans.** Each must point at Elle speech, and must start strictly after its result unless the
  episode is a labelled negative.
- **Write gates.** Every `book_intro_call` needs a gate. The readback is an Elle `readback` whose
  `refers` matches the args (`start` compared as an instant) and that says the timezone. The yes is a
  caller `confirm` that ends before the call. There is no decline or correction in between, and the
  readback is the last one before the write. `save_user_fact` with category `sms_consent` needs a gate
  too, on the consent offer. Other `save_user_fact` calls are silent and ungated, and Elle must never
  narrate them (marker phrases like "make a note" or "saving").
- **Names.** Elle never says a name from `context.notes_names`, or from a bookings lookup, before the caller says it.
- **Dead air.** For each read with latency over 0.7 s, some Elle speech must overlap the wait, and no
  silence on either channel inside the wait may exceed 2.0 s.
- **Negatives.** `labels.negative` must match a `neg` scenario name. The declared `negative_kind` must
  actually occur, and no other kind may.
- **Splits.** (scenario, seed) must not repeat across or within splits, the file's split label must be
  consistent, and `--holdout-family` must not leak into train or val.

## Scenarios (33, in 15 families)

| family | scenarios |
|---|---|
| read_basic | `availability_free`, `availability_busy_alternatives` |
| bookings_read | `bookings_none`, `bookings_one`, `bookings_two` |
| wait_handling | `availability_slow_filler`, `bookings_slow_filler`, `availability_caller_keeps_talking` |
| mind_change | `availability_mind_change_mid_wait` (the result arrives for a time no longer wanted, and Elle checks the new one without reporting the stale result), `mind_change_after_readback` |
| read_error | `availability_unknown_recovery` (UNKNOWN, then retry or book as a request), `availability_timeout` (slow UNKNOWN with fillers, then retry) |
| booking | `book_after_availability_free`, `book_after_availability_busy`, `book_direct_details_first` |
| write_guard | `book_readback_declined` (no write), `book_readback_corrected`, `book_missing_email` (Elle asks; a refusal means no write), `book_invalid_email`, `book_past_time` (Elle catches it, so no past-time call is ever emitted) |
| slot_conflict | `book_slot_taken_race`, `book_busy_at_write` (held out by default) |
| write_error | `book_write_error` (honest recovery, then a silent note) |
| silent_save | `save_fact_midconversation` |
| followup | `book_then_when_again_memory`, `book_then_when_again_lookup` |
| identity | `returning_caller_notes` (notes name an earlier caller, and Elle asks "who should I put this under?") |
| overlap | `barge_in_then_book` (plus random barge-ins on greetings and offers, and backchannels, across scenarios) |
| bilingual | `es_switch_midcall`; 10 other scenarios also render 15 to 20% of their episodes fully in Spanish |
| negative | `neg_premature_availability_claim`, `neg_premature_bookings_claim`, `neg_book_without_readback`, `neg_dead_air_wait` |

Most booking flows end with the production SMS consent flow. It covers every compliance element and
ends with "yes or no?". Only a clear yes triggers `save_user_fact(sms_consent, …)`, followed by "a confirmation text is on its way".

### Adding a scenario

1. Write `def sc_my_case(ep): ...` in `scenarios.py`, composing the beats. The usual pattern is
   `slot_and_ask`, `check`, `ep.claim`, `details`, `readback`, `confirm`, `book`, `booked`, `sms_consent`
   and `bye`. Lines come from `ep.E(key)` / `ep.C(key)` using bank keys, so add new phrasings to
   `banks.py` (in `ES` too if the scenario supports Spanish).
2. Use `ep.claim(...)` for anything that depends on a result, `ep.gate(...)` (through `book`) for writes,
   and `quiet_save` for silent facts. Set `ep.negative_kind` only for `neg_` scenarios.
3. Register it in `SCENARIOS` with a family, a weight and its languages.
4. Run the tests. `test_every_scenario_and_language_validates` builds 25 seeds of every scenario in every language.

## Optional stages

- **`expand.py in.jsonl out.jsonl`.** With `ANTHROPIC_API_KEY` set and the `anthropic` SDK installed
  (default model `claude-opus-5-5`, override with `EXPAND_MODEL`), it rewrites only low-stakes lines:
  greetings, fillers, acks, chit-chat and goodbyes. It never touches readbacks, confirmations, claims,
  times, names, emails or compliance text. A rewrite is kept only if the word count stays within ±30%
  (so the timing still holds) and the episode re-validates. With no key, the output is byte-identical to the input.
- **`render_audio.py episodes.jsonl out_dir --backend deepgram|dry`.** Writes stereo 24 kHz 16-bit WAV
  (channel 0 Elle, channel 1 caller) plus `<id>.json` with
  `{"alignments": [[word, [start, end], "SPEAKER_MAIN"], ...], "tool_spans": [...]}`. Deepgram needs
  `DEEPGRAM_API_KEY`; without it the script exits with status 2 and a clear message. The Deepgram
  request code is written but has never been run, and voice ids are configurable through environment
  variables and should be checked against Deepgram's catalogue. `--backend dry` renders tones offline so the pipeline can be tested.

## Known limitations

- **Synthetic text is not real conversation.** The phrasing comes from templates, about 350 English
  and about 135 Spanish lines. Callers are cooperative and fluent, with no disfluencies, ASR errors,
  crosstalk, noise, wrong numbers or hostile callers. The model can learn the templates rather than
  the behaviour. expand.py helps a little; real data helps a lot.
- **Timing is estimated.** Speech `dur` comes from word count (2.6 words/s, ±12% jitter) and gaps are
  Gaussian. Latencies are sampled per tool (reads: 55% at 0.18 to 0.7 s and 45% at 0.7 to 2.6 s, plus
  dedicated slow and timeout scenarios), not measured.
- **No audio yet.** Rendered TTS will not match the estimated `dur`. A real render must re-time each
  episode from the clip lengths: shift later events, and keep calls and results at the same offsets
  relative to the speech that anchors them. It should also take alignments from TTS timestamps or forced
  alignment rather than the uniform per-word spread used here. The validator should then be re-run on the re-timed episodes.
- **Spanish coverage is narrower.** 11 scenarios, smaller banks, and Spanish for Elle but English tool results (as in production).
- **Real data from the production supervisor would add** real caller language and ASR transcripts,
  measured tool latencies (including calendar slowness and real UNKNOWNs), real barge-in and
  backchannel timing, the actual distribution of intents and outcomes, and the failure cases the
  templates do not imagine. Those logs should be converted into this format (same events, labels and
  validator), labelled for claims and gates, and used at least for val/test, so the metrics are measured on real behaviour.

## Contract ambiguities and how they were resolved

None of these required changing CONTRACT.md, but the trainer and evaluator should know them.

1. **`UNKNOWN:` results.** The contract lists `FREE:` / `BUSY:` only, but production
   `check_availability` also returns `UNKNOWN: …` when the calendar cannot be read or is not connected.
   Read-error and timeout scenarios use it, and Elle never states availability from it. The contract table should probably list it.
2. **`save_user_fact` result.** The contract says `saved`; production returns `Saved.`. The data uses `Saved.`.
3. **Timeouts.** Production has no timeout path for reads. A timeout is modelled as a slow (3.2 to 6 s) `UNKNOWN: couldn't read the calendar just now…`.
4. **Which writes need readback and yes.** `book_intro_call` needs a full readback of time with zone,
   name and email, then a yes. `save_user_fact` is "write, silent", so ordinary facts are ungated, but
   `sms_consent` is gated on the consent offer and a clear yes, as the persona requires. All of this is in `labels.write_gates`.
5. **What counts as a claim span.** Any Elle line whose content depends on a result. That covers
   availability answers, bookings answers, "I can't see his calendar" after UNKNOWN, the post-booking
   "you're penciled in", the slot-taken apology, and "a confirmation text is on its way". A readback
   given right after a re-check is both a readback and a claim.
6. **Dead air.** Metric 5's "Elle says nothing during the wait" is operationalised as no Elle speech
   overlapping the wait, and the validator also flags more than 2.0 s of total silence. Elle's
   "let me check" usually starts just before the call and runs into the wait, as in the contract example.
7. **`expected_calls` in negatives** lists the calls actually made, including the wrong ones.
   Evaluators should score tool selection and arguments (metrics 1 and 2) on positive episodes only.
8. **"Monotonic ids".** Read as: call ids `c1..cN` in time order, and episode ids increasing within a file and unique across files.
9. **`get_my_bookings` on voice.** Production exposes it over SMS (`src/sms.ts`) but not in the voice
   bridge (`twilioBridge.ts`). The contract lists it for this model, so the voice episodes use it.
   `reschedule_booking` and `cancel_booking` are not in the contract, so they are not generated.
10. **Alignment JSON format.** "The trainer's expected JSON format" is not specified. render_audio.py writes
    the moshi-finetune style `alignments` list plus a separate `tool_spans` list; the trainer should confirm or say otherwise.
11. **Interrupted speech.** `text` holds what was actually said (truncated), and `full_text` holds the intended line.
