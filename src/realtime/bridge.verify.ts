/**
 * Offline verification of the Twilio<->realtime bridge logic — no network,
 * no OpenAI, no phone. Uses a fake Twilio socket and a fake RealtimeModel to
 * assert: caller audio is forwarded, model audio is played back, barge-in
 * flushes + truncates + cancels, and a tool call saves a fact.
 *
 * Run: npx tsx src/realtime/bridge.verify.ts
 */
import assert from "node:assert";
import { EventEmitter } from "node:events";
import os from "node:os";
import path from "node:path";

// Point the DB at a throwaway file BEFORE importing anything that opens it.
process.env.SQLITE_PATH = path.join(os.tmpdir(), `letstalk-verify-${process.pid}.db`);

import type { WebSocket } from "ws";
import { Bridge } from "./twilioBridge";
import { RealtimeHandlers, RealtimeModel, RealtimeSessionConfig } from "./types";
import * as db from "../db";

// --- fakes ------------------------------------------------------------------

class FakeTwilio extends EventEmitter {
  readyState = 1; // WebSocket.OPEN
  sent: Array<Record<string, unknown>> = [];
  send(data: string): void {
    this.sent.push(JSON.parse(data));
  }
}

class FakeModel implements RealtimeModel {
  currentItemId: string | null = null;
  calls: string[] = [];
  handlers: RealtimeHandlers;
  config: RealtimeSessionConfig;
  constructor(config: RealtimeSessionConfig, handlers: RealtimeHandlers) {
    this.config = config;
    this.handlers = handlers;
  }
  appendAudio(b64: string): void {
    this.calls.push(`append:${b64}`);
  }
  cancelResponse(): void {
    this.calls.push("cancel");
  }
  truncate(itemId: string, audioEndMs: number): void {
    this.calls.push(`truncate:${itemId}:${audioEndMs}`);
  }
  sendFunctionResult(callId: string, output: string): void {
    this.calls.push(`funcResult:${callId}:${output}`);
  }
  close(): void {
    this.calls.push("close");
  }
}

// --- test -------------------------------------------------------------------

let model: FakeModel | null = null;
const factory = (config: RealtimeSessionConfig, handlers: RealtimeHandlers): RealtimeModel => {
  model = new FakeModel(config, handlers);
  return model;
};

const twilio = new FakeTwilio();
new Bridge(twilio as unknown as WebSocket, factory);
const CALLER = "+15550009999";

// 1. Twilio 'start' — should create the model and record the call.
twilio.emit(
  "message",
  JSON.stringify({
    event: "start",
    start: { streamSid: "MZ123", customParameters: { from: CALLER, callSid: "CA123" } },
  })
);
assert.ok(model, "model was created on start");
assert.match(model!.config.instructions, /phone call/i, "instructions were built");

// 2. Caller audio 'media' — should be forwarded to the model.
twilio.emit(
  "message",
  JSON.stringify({ event: "media", media: { payload: "Y2FsbGVy" } })
);
assert.ok(model!.calls.includes("append:Y2FsbGVy"), "caller audio forwarded to model");

// 3. Model speaks — bridge should play it back to Twilio.
const frame = Buffer.alloc(160).toString("base64"); // 160 bytes μ-law = 20 ms
model!.currentItemId = "item_1";
model!.handlers.onAudioDelta?.(frame, "item_1");
const played = twilio.sent.find((m) => m.event === "media");
assert.ok(played, "model audio played back to Twilio");
assert.deepStrictEqual((played!.media as { payload: string }).payload, frame, "same audio forwarded");

// 4. BARGE-IN — caller talks over the bot.
twilio.sent = [];
model!.calls = [];
model!.handlers.onSpeechStarted?.();
assert.ok(
  twilio.sent.some((m) => m.event === "clear"),
  "barge-in flushes Twilio playback (clear)"
);
assert.ok(
  model!.calls.some((c) => c.startsWith("truncate:item_1:")),
  "barge-in truncates model context to what was heard"
);
assert.ok(model!.calls.includes("cancel"), "barge-in cancels the in-progress response");

// 5. Tool call — should save a fact and return a result.
model!.calls = [];
model!.handlers.onFunctionCall?.(
  "save_user_fact",
  { category: "name", fact: "Their name is Priya." },
  "call_1"
);
assert.ok(
  model!.calls.some((c) => c.startsWith("funcResult:call_1:")),
  "tool result returned to model"
);
const facts = db.getFactsForCaller(CALLER);
assert.ok(
  facts.some((f) => f.category === "name" && f.fact === "Their name is Priya."),
  "fact persisted to the database"
);

console.log("Bridge verification passed:");
console.log("  ✓ caller audio forwarded to the model");
console.log("  ✓ model audio played back to the caller");
console.log("  ✓ barge-in: flush playback + truncate context + cancel response");
console.log("  ✓ tool call saved a fact to the database");
