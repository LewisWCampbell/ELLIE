/**
 * Offline verification of the runtime voice-architecture switch — no network,
 * no phone. The interesting property is the deadline on a demo override: it is
 * enforced when the provider is READ, not by a timer, so it survives the
 * process that set it dying. The same read-per-call rule is what lets two
 * overlapping Cloud Run instances agree about which architecture is live, so
 * that is asserted here too by writing the store behind the module's back.
 *
 * Run: npx tsx src/realtime/factory.verify.ts
 */
import assert from "node:assert";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

async function main(): Promise<void> {
  // Throwaway DB and a known env default, set before config/db load. Static
  // imports hoist above this, so the modules are pulled in dynamically below.
  const dbPath = path.join(os.tmpdir(), `letstalk-factory-${process.pid}.db`);
  for (const f of [dbPath, `${dbPath}-wal`, `${dbPath}-shm`]) {
    try {
      fs.rmSync(f);
    } catch {
      /* not there */
    }
  }
  process.env.SQLITE_PATH = dbPath;
  process.env.REALTIME_PROVIDER = "deepgram";

  const factory = await import("./factory");

  // The deployed default stands when nothing is overridden.
  assert.strictEqual(factory.currentVoiceProvider(), "deepgram", "env default applies");
  assert.strictEqual(factory.voiceProviderOverridden(), false, "no override yet");
  assert.strictEqual(factory.voiceProviderMinutesLeft(), null, "no deadline yet");

  // A timed demo switch takes effect and reports its remaining time.
  factory.setVoiceProvider("openai", 30);
  assert.strictEqual(factory.currentVoiceProvider(), "openai", "demo override applies");
  assert.strictEqual(factory.voiceProviderOverridden(), true, "reported as an override");
  assert.strictEqual(factory.voiceProviderMinutesLeft(), 30, "30 minutes left");

  // An expired deadline reverts on the next read, with no timer involved.
  // The deadline is written straight into the store to simulate time passing,
  // which is also exactly what a second instance would see.
  const { setSetting, getSetting } = await import("../db");
  factory.setVoiceProvider("openai", 30);
  setSetting("voice_provider_expires_at", String(Date.now() - 1000));
  assert.strictEqual(factory.currentVoiceProvider(), "deepgram", "expired override reverts");
  assert.strictEqual(factory.voiceProviderOverridden(), false, "expiry clears the override");
  assert.strictEqual(factory.voiceProviderMinutesLeft(), null, "no deadline after expiry");

  // ...and the revert is persisted, not just in-memory, so the next instance
  // does not resurrect a demo that already ended.
  assert.strictEqual(getSetting("voice_provider"), "", "expiry cleared the stored override");

  // A switch made elsewhere (another instance, the admin surface) is picked up
  // without a restart, because the store is read per call rather than cached.
  setSetting("voice_provider", "openai");
  setSetting("voice_provider_expires_at", "");
  assert.strictEqual(factory.currentVoiceProvider(), "openai", "a switch made elsewhere is seen");

  // An override with no deadline stands until it is cleared by hand.
  factory.setVoiceProvider("openai");
  assert.strictEqual(factory.voiceProviderMinutesLeft(), null, "no deadline when none was asked for");
  assert.strictEqual(factory.currentVoiceProvider(), "openai", "open-ended override stands");
  factory.setVoiceProvider(null);
  assert.strictEqual(factory.currentVoiceProvider(), "deepgram", "clearing reverts to the env default");

  // A live deadline is reported in whole minutes, rounded up, so "1 min left"
  // never displays as 0.
  factory.setVoiceProvider("openai", 45);
  assert.ok((factory.voiceProviderMinutesLeft() ?? 0) > 40, "a live deadline is reported");
  setSetting("voice_provider_expires_at", String(Date.now() + 5_000));
  assert.strictEqual(factory.voiceProviderMinutesLeft(), 1, "a few seconds left rounds up to 1");
  factory.setVoiceProvider(null);

  console.log("Voice factory verification passed:");
  console.log("  ✓ deployed default applies with no override");
  console.log("  ✓ timed demo override applies and reports minutes left");
  console.log("  ✓ an expired deadline reverts on read, with no timer");
  console.log("  ✓ the revert is persisted, so a recycle can't resurrect it");
  console.log("  ✓ a switch made on another instance is picked up per call");
  console.log("  ✓ an override with no deadline stands until cleared");
  console.log("  ✓ remaining time rounds up, so it never reads as 0 min left");
}

void main();
