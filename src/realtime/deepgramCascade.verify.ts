/**
 * Offline verification of the Deepgram cascade orchestration — no network, no
 * Deepgram, no LLM. Fakes the STT stream, the TTS, and the LLM to assert the
 * full turn pipeline: caller utterance → LLM → tool call round-trip → final
 * text → paced μ-law frames out; plus barge-in stops playback mid-utterance.
 *
 * Run: npx tsx src/realtime/deepgramCascade.verify.ts
 */
import assert from "node:assert";
import { DeepgramCascade } from "./deepgramCascade";
import { SttFactory, SttHandlers, SttStream } from "./deepgram/stt";
import { Tts } from "./deepgram/tts";
import { RealtimeHandlers, RealtimeSessionConfig } from "./types";
import { ChatRequest, ChatResponse, LlmProvider } from "../llm/types";

// --- fakes ------------------------------------------------------------------

class FakeStt implements SttStream {
  sent = 0;
  constructor(public handlers: SttHandlers) {}
  send(): void {
    this.sent++;
  }
  close(): void {}
}
function fakeSttFactory(): { factory: SttFactory; get: () => FakeStt } {
  let made: FakeStt | null = null;
  return {
    factory: { open: (h) => (made = new FakeStt(h)) },
    get: () => made as FakeStt,
  };
}

/**
 * Streaming TTS fake: emits `chunks` μ-law pieces of `chunkBytes` each, yielding
 * between them so it behaves like a real stream. Non-frame-aligned on purpose so
 * the cascade's re-framing/carry logic is exercised.
 */
function fakeTts(chunks: number, chunkBytes: number): Tts {
  return {
    async stream(_text, onAudio, isCancelled) {
      for (let i = 0; i < chunks; i++) {
        if (isCancelled()) return;
        onAudio(Buffer.alloc(chunkBytes, 0x7f));
        await Promise.resolve();
      }
    },
  };
}

/** LLM that plays back a queued script of responses, one per chat() call. */
function scriptedLlm(script: ChatResponse[]): LlmProvider & { calls: ChatRequest[] } {
  const calls: ChatRequest[] = [];
  return {
    name: "fake",
    calls,
    async chat(req: ChatRequest): Promise<ChatResponse> {
      calls.push(req);
      return script.shift() ?? { text: "", toolCalls: [], refused: false, raw: null };
    },
  };
}

const sessionConfig: RealtimeSessionConfig = {
  instructions: "You are Elle. Keep it short.",
  voice: "aura-2-thalia-en",
  tools: [
    { name: "search_resume", description: "look up the resume", parameters: { type: "object", properties: {} } },
  ],
};

// --- Scenario A: full happy path with a tool call ---------------------------

async function happyPath(): Promise<void> {
  const audioOut: string[] = [];
  const userTranscripts: string[] = [];
  const assistantTranscripts: string[] = [];
  const toolCalls: Array<{ name: string; id: string }> = [];

  const stt = fakeSttFactory();
  const llm = scriptedLlm([
    // 1st LLM call → wants to search the resume
    { text: "", toolCalls: [{ id: "c1", name: "search_resume", input: { query: "projects" } }], refused: false, raw: null },
    // 2nd LLM call (after tool result) → final spoken answer
    { text: "Lewis built Movie Rails and Moneyball Creator Marketing.", toolCalls: [], refused: false, raw: null },
  ]);

  const handlers: RealtimeHandlers = {
    onAudioDelta: (b64) => audioOut.push(b64),
    onUserTranscript: (t) => userTranscripts.push(t),
    onAssistantTranscript: (t) => assistantTranscripts.push(t),
    onFunctionCall: (name, _args, callId) => {
      toolCalls.push({ name, id: callId });
      // Mirror the bridge: answer the tool synchronously.
      cascade.sendFunctionResult(callId, "Movie Rails; Moneyball Creator Marketing");
    },
  };

  // 5 streamed chunks of 100 bytes = 500 μ-law bytes → 3 full 160-byte frames
  // emitted mid-stream + a 20-byte remainder flushed at the end = 4 frames.
  const cascade = new DeepgramCascade(sessionConfig, handlers, {
    sttFactory: stt.factory,
    tts: fakeTts(5, 100),
    llm,
  });

  // Caller audio flows to STT.
  cascade.appendAudio(Buffer.from("hi").toString("base64"));
  assert.strictEqual(stt.get().sent, 1, "caller audio forwarded to STT");

  // Caller finishes an utterance.
  stt.get().handlers.onUtterance?.("what are your projects");
  await new Promise((r) => setImmediate(r)); // let the async turn settle

  assert.deepStrictEqual(userTranscripts, ["what are your projects"], "user transcript surfaced");
  assert.strictEqual(toolCalls.length, 1, "one tool call surfaced to the bridge");
  assert.strictEqual(toolCalls[0].name, "search_resume", "correct tool");
  assert.strictEqual(llm.calls.length, 2, "LLM called twice (tool loop then answer)");
  assert.deepStrictEqual(
    assistantTranscripts,
    ["Lewis built Movie Rails and Moneyball Creator Marketing."],
    "assistant transcript surfaced"
  );
  assert.strictEqual(audioOut.length, 4, "streamed chunks re-framed into 4 μ-law frames");
  assert.ok(
    audioOut.every((f) => Buffer.from(f, "base64").length <= 160),
    "every frame is <= 160 bytes (20 ms)"
  );
  assert.strictEqual(
    audioOut.reduce((n, f) => n + Buffer.from(f, "base64").length, 0),
    500,
    "no audio bytes dropped in re-framing"
  );
}

// --- Scenario B: barge-in stops playback mid-utterance ----------------------

async function bargeIn(): Promise<void> {
  const audioOut: string[] = [];
  const stt = fakeSttFactory();
  const llm = scriptedLlm([
    { text: "Here is a very long answer that would take many frames to speak.", toolCalls: [], refused: false, raw: null },
  ]);

  let cascade: DeepgramCascade;
  let cancelledSeenByTts = false;

  // TTS that emits one frame, then the caller barges in, then tries to emit more.
  const bargeTts: Tts = {
    async stream(_text, onAudio, isCancelled) {
      onAudio(Buffer.alloc(160, 0x7f)); // frame 1 → forwarded
      cascade.cancelResponse(); // what the bridge does on SpeechStarted
      cancelledSeenByTts = isCancelled(); // stream is told to stop
      onAudio(Buffer.alloc(160, 0x7f)); // frame 2 → dropped by the cascade guard
    },
  };

  cascade = new DeepgramCascade(sessionConfig, { onAudioDelta: (b64) => audioOut.push(b64) }, {
    sttFactory: stt.factory,
    tts: bargeTts,
    llm,
  });

  stt.get().handlers.onUtterance?.("tell me everything");
  await new Promise((r) => setImmediate(r));

  assert.strictEqual(audioOut.length, 1, "barge-in halted playback after the first frame");
  assert.ok(cancelledSeenByTts, "TTS stream is signalled to stop synthesizing");

  // And SpeechStarted passes straight through for the bridge to act on.
  let sawSpeechStarted = false;
  const stt2 = fakeSttFactory();
  new DeepgramCascade(
    sessionConfig,
    { onSpeechStarted: () => (sawSpeechStarted = true) },
    { sttFactory: stt2.factory, tts: fakeTts(0, 0), llm: scriptedLlm([]) }
  );
  stt2.get().handlers.onSpeechStarted?.();
  assert.ok(sawSpeechStarted, "SpeechStarted surfaced for barge-in");
}

// --- Scenario C: bilingual voice selection ----------------------------------
//
// The greeting's Spanish invite and any Spanish reply must go out through the
// Spanish TTS voice; English replies through the English voice. The fake TTS
// records which model each chunk was synthesized with (no openSession, so the
// cascade passes the per-language model to stream()).

async function bilingual(): Promise<void> {
  const spoken: Array<{ text: string; model?: string }> = [];
  const recordingTts: Tts = {
    async stream(text, onAudio, _isCancelled, model) {
      spoken.push({ text, model });
      onAudio(Buffer.alloc(160, 0x7f));
    },
  };

  const stt = fakeSttFactory();
  const llm = scriptedLlm([
    { text: "Lewis built Movie Rails.", toolCalls: [], refused: false, raw: null },
    { text: "Claro, encantada de hablar en español contigo. ¿En qué te ayudo?", toolCalls: [], refused: false, raw: null },
  ]);
  const cascade = new DeepgramCascade(sessionConfig, {}, {
    sttFactory: stt.factory,
    tts: recordingTts,
    llm,
  });

  cascade.greet("Hi there, this is Elle, Lewis's AI assistant.");
  await new Promise((r) => setTimeout(r, 0));

  stt.get().handlers.onUtterance?.("what are your projects", "en");
  await new Promise((r) => setTimeout(r, 0));
  stt.get().handlers.onUtterance?.("hola, prefiero hablar en español", "es");
  await new Promise((r) => setTimeout(r, 0));

  assert.strictEqual(spoken.length, 4, "greeting + invite + two replies spoken");
  assert.ok(spoken[0].model?.endsWith("-en"), "greeting spoken by the English voice");
  assert.ok(spoken[1].model?.endsWith("-es"), "Spanish invite spoken by the Spanish voice");
  assert.ok(spoken[1].text.includes("español"), "invite is the Spanish sentence");
  assert.ok(spoken[2].model?.endsWith("-en"), "English reply spoken by the English voice");
  assert.ok(spoken[3].model?.endsWith("-es"), "Spanish reply spoken by the Spanish voice");
}

async function main(): Promise<void> {
  await happyPath();
  await bargeIn();
  await bilingual();
  console.log("Deepgram cascade verification passed:");
  console.log("  ✓ caller audio → STT; utterance → LLM turn");
  console.log("  ✓ tool call round-trips through the bridge and the LLM continues");
  console.log("  ✓ final answer is streamed and re-framed into 20 ms μ-law frames");
  console.log("  ✓ barge-in (cancelResponse) stops playback mid-utterance");
  console.log("  ✓ SpeechStarted passes through for the bridge to flush Twilio");
  console.log("  ✓ bilingual: greeting invite + Spanish replies use the Spanish voice");
}

void main();
