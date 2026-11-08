# Removing the OpenAI dependency

<!-- doc-check: proposal — this document sketches work that was never built.
     The settings and code below are illustrative, not part of the system. -->

This is the detailed migration guide for taking the app fully off external model APIs, so no third-party AI vendor ever sees call audio (the strongest HIPAA/privacy posture). It assumes the current build, which uses OpenAI two places:

1. **Realtime voice** (`VOICE_MODE=realtime`) — the OpenAI Realtime API is the speech-to-speech model behind the phone bridge.
2. **Cascaded LLM** (optional, `LLM_PROVIDER=openai`) — the text brain, if you point it at OpenAI rather than Claude or a local model.

The codebase was structured so this swap is small: both places already talk to an **interface**, not to OpenAI directly. You are replacing implementations, not rewriting the app.

> **First, the honest nuance.** "No API" means *no external/hosted model API*. A self-hosted model is still reached over a local interface — an inference server on `localhost`, or in-process bindings. What you remove is the internet round-trip, the per-token bill, and the third party. Everything below keeps audio and text inside infrastructure you control.

---

## Where OpenAI is used today (inventory)

| Location | File | What it does | Interface it already implements |
|---|---|---|---|
| Realtime model | `src/realtime/openaiRealtime.ts` | Dials `wss://api.openai.com/v1/realtime`; relays audio; native barge-in | `RealtimeModel` / `RealtimeFactory` (`src/realtime/types.ts`) |
| Realtime wiring | `src/realtime/twilioBridge.ts` | `createMediaWss(factory = openAIRealtimeFactory)` — factory is injectable | — |
| Cascaded brain | `src/llm/openai.ts` | OpenAI-compatible chat client | `LlmProvider` (`src/llm/types.ts`) |
| Config | `src/config.ts` | `realtime.*` (OpenAI key/model/voice), `openai*` (cascaded) | — |

Nothing else imports OpenAI. The bridge (`Bridge` in `twilioBridge.ts`) calls only the `RealtimeModel` interface methods (`appendAudio`, `cancelResponse`, `truncate`, `sendFunctionResult`, `close`, `currentItemId`) — it has no idea OpenAI exists.

---

## Path A — the cascaded brain (LLM only): **already done, zero code**

The text LLM can be fully local *today*. `src/llm/openai.ts` speaks the OpenAI-compatible protocol, and Ollama / vLLM / llama.cpp all serve that protocol on localhost. Point the existing provider at a local server:

```bash
LLM_PROVIDER=openai
OPENAI_BASE_URL=http://localhost:11434/v1   # Ollama
OPENAI_MODEL=llama3.1:8b
OPENAI_API_KEY=ignored                       # required by the client, unused locally
```

No file changes. This removes OpenAI (and Anthropic) as the *brain* in cascaded mode. STT/TTS in cascaded mode are still Twilio's built-in services — see Path C to make those local too.

---

## Path B — the realtime voice model: swap the factory

This is the main event: replace OpenAI Realtime with a **self-hosted full-duplex speech-to-speech model** (e.g. **Moshi** — ~200 ms latency, single GPU, native barge-in). Because the bridge depends on `RealtimeFactory`, you add one new file and change one line.

### B1. Write a `RealtimeModel` implementation for your local model

Create `src/realtime/moshiRealtime.ts`, mirroring `openaiRealtime.ts` but pointing at your server:

```ts
import WebSocket from "ws";
import { config } from "../config";
import { RealtimeFactory, RealtimeHandlers, RealtimeModel, RealtimeSessionConfig } from "./types";

export class MoshiRealtime implements RealtimeModel {
  currentItemId: string | null = null;
  private ws: WebSocket;

  constructor(sessionConfig: RealtimeSessionConfig, private h: RealtimeHandlers) {
    // Your self-hosted endpoint — inside your VPC, no internet:
    this.ws = new WebSocket(config.realtime.localUrl); // e.g. ws://moshi.internal:8998/api/chat
    this.ws.on("open", () => {
      /* send your model's session/config message; set instructions + voice */
    });
    this.ws.on("message", (data) => {
      /* map your model's events onto the handlers:
         - audio out chunk (μ-law 8k)      -> this.h.onAudioDelta?.(b64, itemId)
         - user-speech-onset / VAD event   -> this.h.onSpeechStarted?.()   // barge-in
         - user transcript                 -> this.h.onUserTranscript?.(t)
         - assistant transcript            -> this.h.onAssistantTranscript?.(t)
         - tool/function call              -> this.h.onFunctionCall?.(name, args, callId)
      */
    });
    this.ws.on("error", (e) => this.h.onError?.(e));
    this.ws.on("close", () => this.h.onClose?.());
  }

  appendAudio(b64ulaw: string): void { /* send caller audio frame to your model */ }
  cancelResponse(): void { /* stop current generation */ }
  truncate(itemId: string, audioEndMs: number): void { /* trim model memory to heard audio */ }
  sendFunctionResult(callId: string, output: string): void { /* return tool result, continue */ }
  close(): void { try { this.ws.close(); } catch {} }
}

export const moshiRealtimeFactory: RealtimeFactory = (cfg, h) => new MoshiRealtime(cfg, h);
```

**The audio contract is the leverage.** The bridge sends and expects **G.711 μ-law @ 8 kHz base64** (what Twilio speaks). If your model wants 16-bit PCM at 16/24 kHz (most do), do the μ-law↔PCM conversion and resampling *inside this file* — nowhere else changes. That conversion is the main real work in Path B.

**Barge-in still works the same way.** The bridge already flushes Twilio + truncates + cancels on `onSpeechStarted`. You just need to emit `onSpeechStarted()` when your model/VAD detects caller speech onset. Full-duplex models like Moshi surface this naturally because they're always listening.

### B2. Point the bridge at your factory (one line)

In `src/index.ts`:

```ts
// before
const mediaWss = createMediaWss();
// after
import { moshiRealtimeFactory } from "./realtime/moshiRealtime";
const mediaWss = createMediaWss(moshiRealtimeFactory);
```

(Or make it config-driven: pick the factory from an env var like `REALTIME_BACKEND=openai|moshi`.)

### B3. Config

Add to `src/config.ts` under `realtime`:

```ts
localUrl: process.env.REALTIME_LOCAL_URL ?? "ws://127.0.0.1:8998",
backend: (process.env.REALTIME_BACKEND ?? "openai") as "openai" | "moshi",
```

Then `openaiRealtime.ts` is no longer imported once `backend=moshi` — you can keep it for fallback or delete it.

### B4. Infrastructure

- A **GPU host** for the full-duplex model (Moshi runs on a single modern GPU; a cloud GPU instance or an owned card). Serve it on a private address the app can reach.
- Keep it **inside your network / VPC** — that's the whole point: audio never leaves.
- The app itself stays CPU-only and cheap.

---

## Path C — remove Twilio's built-in speech (cascaded mode, optional)

If you use cascaded mode and want *its* speech to be local too (not just the brain), you must stop using Twilio's Gather STT and Say TTS, because those are external services:

1. **Switch cascaded to Media Streams** like the realtime path — replace the `<Gather>`/`<Say>` loop in `src/voice.ts` with a `<Connect><Stream>` bridge (you can copy the structure from `src/realtime/`).
2. **Self-host STT**: run **whisper.cpp / faster-whisper** (8 kHz phone audio → a small model is plenty; runs on CPU or a small GPU). Feed it the Media Stream audio, get transcripts, call `runTurn()` in `ai.ts`.
3. **Self-host TTS**: run **Piper** (fast, CPU-friendly); convert its output to μ-law 8 kHz and stream back to Twilio.
4. **Barge-in here is hand-built**: add VAD (Silero) on the incoming audio; on speech onset, stop the Piper playback and discard the queued audio.

This is more work than Path B because you're assembling three models and writing the turn-taking yourself — which is exactly why the realtime full-duplex model (Path B) is the better route to good barge-in.

---

## What still isn't "yours" after all this

Removing OpenAI does **not** remove the phone network. You still rent a **DID + SIP trunk** from a carrier (Twilio/Telnyx) to receive calls — that's telephony, not a model API, and it's covered by a BAA, not by self-hosting the model. If you want to drop the telephony vendor too, that's the self-hosted SIP path in `course/` (jambonz on your own server). The layering:

| Layer | External API? | How to self-host |
|---|---|---|
| Phone number / trunk | Carrier (always) | Own SIP server + a bare trunk (course capstone) |
| Speech-to-speech model | OpenAI Realtime | Moshi on your GPU (Path B) |
| — or STT + LLM + TTS | Twilio STT/TTS + OpenAI/Claude | whisper.cpp + Ollama + Piper (Paths A+C) |
| App / memory | none (already local) | — |

---

## Cost reframe

You trade **per-minute API fees for fixed compute**:

- Today (managed): ~$0.02–0.05/min for the realtime model, scaling linearly with call minutes.
- Self-hosted: a GPU instance (~$0.50–2/hr, or owned hardware) that serves many concurrent calls, plus your ops time. Effectively **free per minute** once running.

Below a few hundred concurrent-call-hours a month the API is cheaper; past that, self-hosting wins — and the privacy property (no third party sees PHI) may justify it at any volume for a health product.

---

## Migration checklist

- [ ] **Brain local** (cascaded): point `LLM_PROVIDER=openai` at Ollama/vLLM (`OPENAI_BASE_URL=localhost`). No code change.
- [ ] **Stand up the model server**: Moshi on a GPU (Path B) or whisper+llm+piper (Path C), on a private address.
- [ ] **Implement `RealtimeModel`** for your model (`moshiRealtime.ts`), handling the μ-law↔PCM/resample conversion and emitting `onSpeechStarted` for barge-in.
- [ ] **Swap the factory** in `index.ts` (or via `REALTIME_BACKEND`).
- [ ] **Verify offline**: adapt `src/realtime/bridge.verify.ts` — the bridge contract is unchanged, so the same test proves your new model wires up correctly (`npm run verify:bridge`).
- [ ] **Test a real call**; tune the barge-in VAD threshold and truncation.
- [ ] **Delete `openaiRealtime.ts`** (and the OpenAI config) once you're confident, or keep it behind `REALTIME_BACKEND=openai` as a fallback.
- [ ] **Confirm no egress**: with the model local and the brain local, the only external hop left is the carrier trunk — under a BAA.

The key takeaway: because the bridge depends on `RealtimeModel` (not OpenAI), removing the dependency is **one new file + one changed line + a model server** — not a rewrite.
