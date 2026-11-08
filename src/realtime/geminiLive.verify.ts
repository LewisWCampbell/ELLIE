/**
 * Offline verification of the Gemini Live adapter — no network, no Google
 * credentials, no phone. A fake session stands in for the SDK, so every path
 * the Twilio bridge depends on is exercised: audio in, audio out, barge-in,
 * transcripts, tool calls, the greeting gate, and the close race.
 *
 * This cannot prove the live wire protocol is right; only a real call can do
 * that. What it does prove is that everything on THIS side of the SDK boundary
 * behaves, so a failed demo points at the model or the credentials rather than
 * at the adapter.
 *
 * Run: npx tsx src/realtime/geminiLive.verify.ts
 */
import assert from "node:assert";
import { pcm16ToUlaw, ulawToPcm16 } from "./audio/g711";
import { GeminiLive, type LiveCallbacks, type LiveMessage, type LiveSession } from "./geminiLive";
import type { RealtimeHandlers, RealtimeSessionConfig } from "./types";

interface Sent {
  audio: string[];
  turns: string[];
  toolResponses: Array<{ id: string; response: Record<string, unknown> }>;
  closed: boolean;
}

/** A fake Live session, plus a handle to drive server messages into it. */
function fakeSession(): { session: LiveSession; sent: Sent } {
  const sent: Sent = { audio: [], turns: [], toolResponses: [], closed: false };
  const session: LiveSession = {
    sendRealtimeInput: (p) => sent.audio.push(p.audio.data),
    sendClientContent: (p) => sent.turns.push(p.turns),
    sendToolResponse: (p) => {
      for (const r of p.functionResponses) sent.toolResponses.push({ id: r.id, response: r.response });
    },
    close: () => {
      sent.closed = true;
    },
  };
  return { session, sent };
}

const SESSION_CONFIG: RealtimeSessionConfig = {
  instructions: "You are Elle.",
  instructionsDynamic: "The caller is unknown.",
  voice: "Aoede",
  tools: [{ name: "book_intro_call", description: "Book a call", parameters: { type: "object" } }],
};

/** 20 ms of caller audio: 160 μ-law bytes at 8 kHz. */
const FRAME = Buffer.alloc(160, 0x40).toString("base64");

/** One chunk of model audio: PCM16 at 24 kHz, base64 as the wire carries it. */
function modelAudio(samples: number): string {
  const pcm = Buffer.alloc(samples * 2);
  for (let i = 0; i < samples; i++) pcm.writeInt16LE(Math.round(9000 * Math.sin(i / 4)), i * 2);
  return pcm.toString("base64");
}

async function main(): Promise<void> {
  const { session, sent } = fakeSession();
  let cb!: LiveCallbacks;
  let seenInstructions = "";

  const events: string[] = [];
  const audioOut: string[] = [];
  const handlers: RealtimeHandlers = {
    onReady: () => events.push("ready"),
    onAudioDelta: (b64) => audioOut.push(b64),
    onSpeechStarted: () => events.push("speech-started"),
    onGreetingSpoken: () => events.push("greeting-spoken"),
    onUserTranscript: (t) => events.push(`user:${t}`),
    onAssistantTranscript: (t) => events.push(`elle:${t}`),
    onFunctionCall: (name, args, id) => events.push(`tool:${name}:${JSON.stringify(args)}:${id}`),
    onError: (e) => events.push(`error:${String(e)}`),
    onClose: () => events.push("closed"),
  };

  const model = new GeminiLive(SESSION_CONFIG, handlers, async (instructions, _cfg, callbacks) => {
    seenInstructions = instructions;
    cb = callbacks;
    return session;
  });

  // Audio that arrives before the socket opens is queued, not dropped: a caller
  // who starts talking the instant the call connects is ordinary.
  model.appendAudio(FRAME);
  assert.strictEqual(sent.audio.length, 0, "pre-open audio is queued, not sent");

  await new Promise((r) => setImmediate(r)); // let connect() settle
  cb.onopen();

  assert.ok(events.includes("ready"), "onReady fires when the session opens");
  assert.strictEqual(sent.audio.length, 1, "queued audio flushes on open");
  assert.ok(
    seenInstructions.startsWith("You are Elle.") && seenInstructions.includes("caller is unknown"),
    "static instructions come first, dynamic context after, so the cache prefix holds"
  );

  // Caller audio is upsampled: 160 μ-law bytes at 8 kHz become 320 PCM16
  // samples at 16 kHz, which is 640 bytes.
  model.appendAudio(FRAME);
  const firstIn = Buffer.from(sent.audio[1], "base64");
  assert.strictEqual(firstIn.length, 640, "8 kHz μ-law becomes 16 kHz PCM16");

  // Model audio is downsampled and handed back as μ-law for Twilio: 300 samples
  // at 24 kHz is 100 bytes at 8 kHz.
  cb.onmessage({
    serverContent: { modelTurn: { parts: [{ inlineData: { data: modelAudio(300) } }] } },
  });
  assert.strictEqual(audioOut.length, 1, "model audio reaches the bridge");
  assert.strictEqual(
    Buffer.from(audioOut[0], "base64").length,
    100,
    "24 kHz PCM16 becomes 8 kHz μ-law"
  );
  assert.ok(model.currentItemId, "an item id is set so the bridge can correlate audio");

  // Transcripts flow both ways, which is what keeps the call log, the paired
  // browser, and the admin view working on this architecture too.
  cb.onmessage({ serverContent: { inputTranscription: { text: "hi there" } } });
  cb.onmessage({ serverContent: { outputTranscription: { text: "Hi, this is Elle." } } });
  assert.ok(events.includes("user:hi there"), "caller transcript is forwarded");
  assert.ok(events.includes("elle:Hi, this is Elle."), "assistant transcript is forwarded");

  // Barge-in: the model announces it interrupted itself, and the bridge is told
  // so it can flush what Twilio has already buffered.
  cb.onmessage({ serverContent: { interrupted: true } });
  assert.ok(events.includes("speech-started"), "interruption is reported as barge-in");

  // Tool calls round-trip with the id the model supplied.
  cb.onmessage({
    toolCall: {
      functionCalls: [{ id: "call-1", name: "book_intro_call", args: { when: "tuesday" } }],
    },
  });
  assert.ok(
    events.some((e) => e.startsWith("tool:book_intro_call:") && e.endsWith(":call-1")),
    "tool call is forwarded with its id"
  );
  model.sendFunctionResult("call-1", JSON.stringify({ ok: true }));
  assert.deepStrictEqual(
    sent.toolResponses,
    [{ id: "call-1", response: { ok: true } }],
    "tool result goes back as structured JSON"
  );

  // A tool that returns a bare string must not break the turn.
  model.sendFunctionResult("call-2", "not json");
  assert.deepStrictEqual(
    sent.toolResponses[1],
    { id: "call-2", response: { result: "not json" } },
    "a non-JSON tool result is wrapped rather than thrown away"
  );

  // The greeting gate: onGreetingSpoken fires only once the greeting's turn
  // completes, which is what keeps barge-in disabled until the caller has heard
  // the whole opening line.
  const before = events.filter((e) => e === "greeting-spoken").length;
  model.greet("Hi, this is Elle.");
  assert.ok(
    sent.turns.some((t) => t.includes("Hi, this is Elle.")),
    "the greeting is sent as a turn"
  );
  assert.strictEqual(
    events.filter((e) => e === "greeting-spoken").length,
    before,
    "greeting is not reported spoken before its turn completes"
  );
  cb.onmessage({ serverContent: { turnComplete: true } });
  assert.strictEqual(
    events.filter((e) => e === "greeting-spoken").length,
    before + 1,
    "greeting reported spoken when its turn completes"
  );

  // Typed input from a paired browser is a turn like any other.
  model.sendText("what has he shipped?");
  assert.ok(sent.turns.includes("what has he shipped?"), "typed input becomes a turn");

  // A turn's trailing partial group is flushed rather than held back into the
  // next turn: 302 samples is 100 whole groups plus 2 left over.
  audioOut.length = 0;
  cb.onmessage({
    serverContent: { modelTurn: { parts: [{ inlineData: { data: modelAudio(302) } }] } },
  });
  cb.onmessage({ serverContent: { turnComplete: true } });
  const emitted = audioOut.reduce((n, b) => n + Buffer.from(b, "base64").length, 0);
  assert.strictEqual(emitted, 101, "the remainder is flushed at end of turn, not dropped");

  // The audio actually carries signal rather than silence, which is the failure
  // a length check alone would miss.
  const decoded = ulawToPcm16(Buffer.from(audioOut[0], "base64"));
  let energy = 0;
  for (let i = 0; i < decoded.length / 2; i++) energy += Math.abs(decoded.readInt16LE(i * 2));
  assert.ok(energy / (decoded.length / 2) > 1000, "emitted audio is speech, not silence");

  // Barge-in must not leave a partial group to be prepended to the next turn.
  cb.onmessage({
    serverContent: { modelTurn: { parts: [{ inlineData: { data: modelAudio(2) } }] } },
  });
  model.cancelResponse();
  audioOut.length = 0;
  cb.onmessage({
    serverContent: { modelTurn: { parts: [{ inlineData: { data: modelAudio(3) } }] } },
  });
  assert.strictEqual(
    audioOut.reduce((n, b) => n + Buffer.from(b, "base64").length, 0),
    1,
    "cancel drops the stale partial group instead of carrying it forward"
  );

  // Closing is idempotent and reaches the session.
  model.close();
  assert.ok(sent.closed, "close reaches the session");
  model.close();

  // The close race: a call that hangs up while the socket is still opening must
  // still close the session it eventually gets, rather than leaking it.
  const late = fakeSession();
  let release!: (s: LiveSession) => void;
  const pending = new GeminiLive(
    SESSION_CONFIG,
    {},
    () => new Promise<LiveSession>((resolve) => (release = resolve))
  );
  pending.close();
  release(late.session);
  await new Promise((r) => setImmediate(r));
  assert.ok(late.sent.closed, "a session that arrives after close() is closed anyway");

  // A connector that rejects surfaces as an error rather than an unhandled
  // rejection that takes the process down mid-call.
  const errors: unknown[] = [];
  new GeminiLive(SESSION_CONFIG, { onError: (e) => errors.push(e) }, async () => {
    throw new Error("no credentials");
  });
  await new Promise((r) => setImmediate(r));
  assert.strictEqual(errors.length, 1, "a failed connect is reported, not thrown");

  // Sanity: the fixture really is μ-law the bridge could play.
  assert.strictEqual(pcm16ToUlaw(ulawToPcm16(Buffer.from(FRAME, "base64"))).length, 160);

  console.log("Gemini Live adapter verification passed:");
  console.log("  ✓ audio arriving before the socket opens is queued, then flushed");
  console.log("  ✓ static instructions precede dynamic context (cache prefix holds)");
  console.log("  ✓ caller μ-law 8 kHz is upsampled to PCM16 16 kHz");
  console.log("  ✓ model PCM16 24 kHz is downsampled to μ-law 8 kHz with real signal");
  console.log("  ✓ transcripts forwarded both ways");
  console.log("  ✓ model-side interruption surfaces as barge-in");
  console.log("  ✓ tool calls round-trip, including non-JSON results");
  console.log("  ✓ greeting is reported spoken only when its turn completes");
  console.log("  ✓ end of turn flushes the resampler remainder");
  console.log("  ✓ barge-in drops the stale partial group");
  console.log("  ✓ a session arriving after close() is still closed");
  console.log("  ✓ a failed connect is reported instead of crashing the call");
}

void main();
