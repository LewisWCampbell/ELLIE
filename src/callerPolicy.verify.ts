/**
 * Offline verification of repeat-caller escalation — no network, no phone.
 * Records calls in a throwaway DB and asserts the tier climbs normal → limited
 * → blocked as the same number keeps calling, and that it's per-number.
 *
 * Run: npx tsx src/callerPolicy.verify.ts
 */
import assert from "node:assert";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

async function main(): Promise<void> {
  // Throwaway DB + explicit thresholds must be set BEFORE the config/db modules
  // load. Static `import` is hoisted above this code, so load them dynamically
  // (after the env is set) instead.
  const dbPath = path.join(os.tmpdir(), `letstalk-policy-${process.pid}.db`);
  for (const f of [dbPath, `${dbPath}-wal`, `${dbPath}-shm`]) {
    try {
      fs.rmSync(f);
    } catch {
      /* not there */
    }
  }
  process.env.SQLITE_PATH = dbPath;
  process.env.ABUSE_LIMITED_AFTER = "3";
  process.env.ABUSE_BLOCK_AFTER = "6";
  process.env.ABUSE_WINDOW_HOURS = "24";

  const db = await import("./db");
  const { assessCaller } = await import("./callerPolicy");

  const A = "+15551110000";
  const B = "+15552220000";

  // First contact → normal.
  assert.strictEqual(assessCaller(A).tier, "normal", "1st call is normal");
  assert.strictEqual(assessCaller(A).callNumber, 1, "callNumber counts this call");

  // After 3 recorded calls, the 4th is limited.
  for (let i = 0; i < 3; i++) db.recordCallStart(`A-${i}`, A);
  assert.strictEqual(assessCaller(A).tier, "limited", "past limitedAfter → limited");
  assert.strictEqual(assessCaller(A).callNumber, 4, "4th call in the window");

  // After 6 recorded calls, the 7th is blocked.
  for (let i = 3; i < 6; i++) db.recordCallStart(`A-${i}`, A);
  assert.strictEqual(assessCaller(A).tier, "blocked", "past blockAfter → blocked");
  assert.strictEqual(assessCaller(A).callNumber, 7, "7th call in the window");

  // Escalation is keyed on the number — a different caller is unaffected.
  assert.strictEqual(assessCaller(B).tier, "normal", "a different number is still normal");

  console.log("Caller policy verification passed:");
  console.log("  ✓ first-time caller is normal");
  console.log("  ✓ escalates to limited past the limited threshold");
  console.log("  ✓ escalates to blocked past the block threshold");
  console.log("  ✓ escalation is per phone number");
}

void main();
