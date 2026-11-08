# Runtime: plugging the trained model into the Node service

This describes a Python inference server for the fine-tuned Moshi-family model
and the WebSocket protocol a fourth `RealtimeModel` implementation in Node
(`src/realtime/moshiDuplex.ts`, next to `openaiRealtime.ts`, `geminiLive.ts`,
`deepgramCascade.ts`) can be written against. The frame-level logic the server
needs (tool-span parsing, forced result injection, banned tokens) is already
implemented and tested in `runtime.py` (`ToolStreamParser`, `Injector`) and is
what `evaluate.py --mode model` runs in closed loop.

Status: the protocol and the frame logic are designed and unit-tested here; the
server itself is a sketch. Everything that touches the `moshi` package
(`LMGen`, Mimi streaming encode/decode, PersonaPlex's `text_token=` forcing) is
**UNVERIFIED**: written from source read on 2026-11-07, never run (no GPU, no
`moshi` install, Hugging Face blocked from this machine).

## 1. What the Node side expects (read from the repo)

`src/realtime/types.ts`:

```ts
interface RealtimeModel {
  readonly currentItemId: string | null;
  appendAudio(base64ulaw: string): void;          // caller audio, μ-law 8 kHz, base64
  sendText?(text: string): void;                  // typed text from the paired browser
  cancelResponse(): void;                          // barge-in
  truncate(itemId: string, audioEndMs: number): void;
  sendFunctionResult(callId: string, output: string): void;
  greet?(text: string): void;                      // speak an opening line now
  close(): void;
}
type RealtimeFactory = (config: RealtimeSessionConfig, handlers: RealtimeHandlers) => RealtimeModel;
```

Handlers the model calls: `onReady`, `onAudioDelta(base64ulaw, itemId)`,
`onSpeechStarted`, `onGreetingSpoken`, `onUserTranscript`, `onAssistantTranscript`,
`onFunctionCall(name, args, callId)`, `onError`, `onClose`.

`src/realtime/twilioBridge.ts` `handleFunctionCall(name, args, callId)` executes
the tool and answers with `this.rs.sendFunctionResult(callId, out)`:
`save_user_fact` replies `"Saved."` synchronously; `check_availability` and
`book_intro_call` reply asynchronously when the calendar/DB call resolves;
`get_my_bookings` is **not** handled on the voice bridge today (it exists only in
`src/sms.ts`); it falls into `default: reply("Unknown tool: …")`. The bridge
must get a `get_my_bookings` case before this model goes live, or the model must
not be trained to call it on voice (see README, contract issues).

## 2. Server process

```
caller μ-law 8k ──► resample 24k ──► Mimi.encode (streaming, 1920 samples = 80 ms = 1 frame)
                                         │ 8 user codes / frame
                                         ▼
                 ┌──────────── LMGen.step (one frame) ─────────────┐
 forced token? ──┤ text token: sampled (with banned ids) or forced │
 (Injector)      └──────────── text token + 8 Elle codes ──────────┘
       ▲                   │ text token                 │ Elle codes
       │                   ▼                            ▼
       │          ToolStreamParser.feed      Mimi.decode ──► 24k ──► 8k μ-law ──► audio msg
       │                   │ </tool>  ─────────► tool_call msg ──► Node executes tool
       └─── add_result ◄── tool_result msg ◄────────────────────────────┘
```

Per 80 ms frame:

1. Take 1920 samples of caller audio at 24 kHz (pad with silence if the network
   is late; never block the clock), encode to 8 user codes.
2. `forced = injector.forced(frame, last_text_token, parser)`.
   * if `forced` is not None: run the step with that text token forced. In
     PersonaPlex's fork, `LMGen.step(input_tokens, text_token=...)` exists for
     its system prompt (UNVERIFIED that it behaves identically mid-stream). In
     Kyutai's `LMGen`, force it through `on_text_logits_hook` by setting every
     logit except the forced id to -inf (works with sampling and greedy).
   * else sample normally, with `injector.banned_ids(parser)` masked to -inf
     (`<result id=` and `</result>` are never sampled; `<tool>` is banned inside
     a span).
3. `parser.feed(text_token, frame)`; on a `tool_call` event send a `tool_call`
   message (below) and remember `call_id`. On `tool_error` (malformed JSON),
   inject `<result id="…">Error: malformed tool call.</result>` so the model
   recovers in-band, and log it.
4. Decode Elle's 8 codes to audio, resample to 8 kHz μ-law, send. Audio keeps
   flowing during the call span and the wait.
5. Plain text tokens are detokenized into `transcript.assistant` deltas.

### Waiting policy

* **Audio never pauses.** The model was trained on episodes where Elle keeps
  talking ("one sec, checking…") through the call span and the wait, so it
  generates that itself. Pausing generation would freeze the frame clock and
  desync from the caller's audio. The dead-air metric checks this behaviour.
* **Result injection** happens at the first word boundary (last token PAD/EPAD
  or a marker) at or after arrival, at most `max_wait_frames` (6 = 0.48 s) late,
  preceded by EPAD when the previous token was PAD, exactly as `prepare.py`
  builds training streams. A result occupies one frame per token (~2-3 s for a
  typical availability result), during which Elle's own text cannot advance.
* **Timeouts.** If no result arrives within 8 s, inject
  `UNKNOWN: couldn't read the calendar just now.` (the dataset's timeout
  convention) so the model has something to recover from.
* **Premature-claim guard (optional).** Text leads audio by only about a frame
  in Moshi, so the server cannot reliably catch a claim before it is voiced.
  If the eval gate is not met, train with text ahead of audio (Moshi-finetune's
  interleaver has an `audio_delay`), and have the server run
  `metrics.looks_like_claim` on the pending text while a read is in flight and
  force PAD instead. Not built: the gate should be met by training.

### Barge-in

Moshi is full duplex: when the caller talks over Elle, the model hears it and
stops on its own. So the server sends **no** `speech_started` events by default,
and `cancel`/`truncate` only drop audio that is already queued on the server
side (they do not touch model state; there is no "response" to cancel).
`twilioBridge` still flushes Twilio's buffer on `onSpeechStarted`; if the server
should drive that, it can emit `speech_started` when the model's own output
energy drops to silence within 200 ms of caller speech onset (a heuristic,
UNVERIFIED).

### Sessions and prompts

* `instructions` + `instructionsDynamic` become the PersonaPlex text prompt
  (`<system> … <system>`, pinned in context). Keep it the same as training
  (`prepare.prompt_text`); a very different prompt is off-distribution.
  The long OpenAI-style system prompt (knowledge base, policies) does not fit:
  PersonaPlex's pinned prompt is at most `--max-pinned-frames` (512 frames, one
  token per frame). Per-call caller facts can be injected as a forced
  `<result id="ctx">Caller notes: …</result>` at session start.
* `voice` maps to a PersonaPlex voice prompt file (e.g. `NATF2.pt`).
* `tools` is informational: the tool set is baked in by training. The server
  rejects a session whose tool names are not a subset of the trained ones.
* `sendText` (typed web messages): inject as a forced
  `<result id="web">Caller typed: …</result>`. Not trained; expect weak handling
  until such episodes are added to the dataset.
* `greet(text)`: force the greeting's text tokens on Elle's stream at their
  natural pace (one word start every ~5 frames, as `prepare.spread_words` lays
  out); the audio follows the text. Send `greeting_spoken` after the last
  frame of it has been sent. UNVERIFIED that forced text yields clean audio;
  the fallback is to let the model greet on its own (it was trained to) and
  not implement `greet`.

## 3. WebSocket protocol (JSON text frames)

Endpoint: `wss://<host>/v1/duplex`, one connection per call. Every message is a
JSON object with a `type`. Audio is base64 μ-law 8 kHz mono (what Twilio and
the bridge already use), 20 ms or more per message; the server resamples.
Server-side frames are 80 ms; the server buffers.

### Client (Node) → server

| type | fields | when |
|---|---|---|
| `session.start` | `instructions`, `instructions_dynamic?`, `voice`, `tools: [{name, description, parameters}]`, `call_sid`, `sample_rate: 8000`, `encoding: "mulaw"` | first message |
| `audio.append` | `audio` (base64 μ-law) | `appendAudio` |
| `tool.result` | `call_id`, `output` (string, exactly what the tool returned) | `sendFunctionResult` |
| `text.append` | `text` | `sendText` (optional) |
| `greet` | `text` | `greet` (optional) |
| `audio.cancel` | | `cancelResponse`: drop server-side queued audio |
| `audio.truncate` | `item_id`, `audio_end_ms` | `truncate`: same, plus marks the transcript |
| `session.close` | | `close` |

### Server → client (Node)

| type | fields | maps to |
|---|---|---|
| `session.ready` | `session_id`, `model`, `frame_ms: 80` | `onReady` |
| `audio.delta` | `item_id`, `audio` (base64 μ-law 8 kHz) | `onAudioDelta(audio, item_id)` |
| `transcript.assistant` | `item_id`, `delta` | `onAssistantTranscript` (accumulate per item) |
| `tool.call` | `call_id`, `name`, `arguments` (object) | `onFunctionCall(name, arguments, call_id)` |
| `tool.result_injected` | `call_id`, `frame` | (log only: latency accounting) |
| `greeting.done` | | `onGreetingSpoken` |
| `speech.started` | | `onSpeechStarted` (off by default, see barge-in) |
| `error` | `code`, `message`, `fatal: bool` | `onError` |
| `session.closed` | `reason` | `onClose` |

`item_id` is a new id each time Elle starts speaking after at least 0.64 s of
text-stream silence (the same rule `runtime.decode_text_stream` uses), so
`currentItemId` and `truncate` keep their meaning.
`call_id` is the `id` the model wrote in the JSON (`c1`, `c2`, …), made unique
per session by the server if the model repeats one; the Node side passes it
back unchanged. There is no `onUserTranscript` source in Moshi (it does not
transcribe the caller); run Deepgram STT on the caller audio in Node if the
transcript is needed (the cascade already has the code).

### Example exchange

```
→ {"type":"session.start","instructions":"…","voice":"NATF2","tools":[…],"call_sid":"CA…","sample_rate":8000,"encoding":"mulaw"}
← {"type":"session.ready","session_id":"s_19","model":"elle-tools-v1","frame_ms":80}
→ {"type":"audio.append","audio":"//7+…"}            (continuous)
← {"type":"audio.delta","item_id":"i3","audio":"…"}   (continuous)
← {"type":"transcript.assistant","item_id":"i3","delta":" Let me check"}
← {"type":"tool.call","call_id":"c1","name":"check_availability","arguments":{"start":"2026-11-12T14:00:00-08:00"}}
→ {"type":"tool.result","call_id":"c1","output":"FREE: Thursday, November 12 at 2:00 PM PST is open. Offer it to the caller."}
← {"type":"tool.result_injected","call_id":"c1","frame":412}
← {"type":"transcript.assistant","item_id":"i4","delta":" He's free then"}
```

## 4. Node implementation notes (`moshiDuplex.ts`)

* Mirror `openaiRealtime.ts`: open the socket in the factory, queue
  `appendAudio` until `session.ready`, translate messages to handlers.
* `sendFunctionResult` sends `tool.result` and nothing else (no
  `response.create`: the model is always generating).
* `cancelResponse`/`truncate` send `audio.cancel`/`audio.truncate`.
* Expose a `REALTIME_PROVIDER=moshi` option in `factory.ts`.
* `handleFunctionCall` needs a `get_my_bookings` case for voice (see section 1).
* Latency budget per tool call: call-span emission (~2-4 s of 80 ms frames,
  see README), tool latency, then result injection (~1 frame per token). The
  bridge should not add its own timeouts shorter than 8 s.

## 5. Server sketch (UNVERIFIED APIs)

```python
# serve.py (sketch, not in this repo as runnable code)
from moshi.models import loaders, LMGen          # PersonaPlex or Kyutai fork
from runtime import ToolStreamParser, Injector
from textvocab import SentencePieceTokenizer

info = loaders.CheckpointInfo.from_hf_repo("nvidia/personaplex-7b-v1")
mimi, lm = info.get_mimi(device="cuda"), info.get_moshi(device="cuda")
resize_moshi_text_vocab(lm, 4); load merged LoRA + new rows (train.py checkpoint)
tok = SentencePieceTokenizer(info.tokenizer)
lm_gen = LMGen(lm, temp=0.8, temp_text=0.7, on_text_logits_hook=apply_ban_and_force)

async def session(ws):
    parser, inj = ToolStreamParser(tok), Injector(tok, max_wait_frames=6)
    with mimi.streaming(1), lm_gen.streaming(1):
        while True:
            pcm = await caller_audio.read_frame()              # 1920 samples @ 24 kHz
            codes = mimi.encode(pcm)                           # [1, 8, 1]
            forced = inj.forced(frame, last, parser)           # read by apply_ban_and_force
            out = lm_gen.step(codes)                           # [1, 1 + 8, 1] or None during warm-up
            ...feed parser, send tool.call / audio.delta / transcript ...
```
