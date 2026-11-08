/**
 * Offline verification of the owner's "has anyone called?" report — no network,
 * no phone, no intake platform. Seeds a throwaway database with calls, notes
 * and a booking, then asserts what the summary says about them.
 *
 * The two behaviours worth pinning down are the ones that would be wrong in a
 * way nobody notices. Notes have to be read PER CALL, because reporting a note
 * from a previous conversation as something this caller said is the same
 * shared-phone bug the persona already guards against. And an empty result has
 * to distinguish "nobody rang" from "this instance has no memory of it",
 * because the call log lives on the instance and a deploy takes it away.
 *
 * Run: npx tsx src/intake.verify.ts
 */
import assert from "node:assert";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

async function main(): Promise<void> {
  const dbPath = path.join(os.tmpdir(), `letstalk-intake-${process.pid}.db`);
  for (const f of [dbPath, `${dbPath}-wal`, `${dbPath}-shm`]) {
    try {
      fs.rmSync(f);
    } catch {
      /* not there */
    }
  }
  process.env.SQLITE_PATH = dbPath;
  process.env.BOOKING_TIMEZONE = "America/Los_Angeles";

  const db = await import("./db");
  const { summarizeRecentCalls } = await import("./telegram");

  const NOW = new Date("2026-08-26T20:00:00Z");
  const A = "+15551110000";
  const B = "+15552220000";

  // Nothing recorded yet, and the process has been up longer than the window:
  // that is a real "nobody called", with no caveat to add.
  assert.match(
    summarizeRecentCalls(24, 10, NOW, 48 * 3600),
    /^No calls in the last 24 hours\.$/,
    "an empty log on a long-lived instance reports plainly"
  );

  // Same empty log, but the instance is younger than the window. The answer has
  // to say the history is missing rather than assert nobody rang.
  const young = summarizeRecentCalls(24, 10, NOW, 900);
  assert.match(young, /No calls in the last 24 hours\./, "still reports the window");
  assert.match(young, /only been up 15 minutes/, "names how long this instance has existed");
  assert.match(young, /doesn't survive a redeploy/, "explains why the history may be short");

  // A call that went nowhere.
  db.recordCallStart("CA-1", A);
  db.recordCallEnd("CA-1");

  // A call that produced notes and a booking.
  db.recordCallStart("CA-2", B);
  db.saveFact(B, "identity", "Dana Reyes, recruiter at Acme", "CA-2");
  db.saveFact(B, "intent", "hiring for a senior analytics role", "CA-2");
  db.saveBooking({
    phoneNumber: B,
    callSid: "CA-2",
    name: "Dana Reyes",
    email: "dana@example.com",
    topic: "analytics role",
    startAt: "2026-08-28T17:00:00.000Z",
    status: "pending",
    eventLink: null,
    eventId: null,
  });
  db.recordCallEnd("CA-2");

  const report = summarizeRecentCalls(24, 10, NOW, 48 * 3600);

  assert.match(report, /^2 calls in the last 24 hours, 1 of them booked:/m, "counts calls and bookings");
  assert.ok(report.includes(A), "lists the first caller's number");
  assert.ok(report.includes(B), "lists the second caller's number");
  assert.match(report, /nothing noted, no booking/, "a call that went nowhere says so");
  assert.match(report, /Dana Reyes, recruiter at Acme \(identity\)/, "notes are quoted with their category");
  // The booking's start is shown in the owner's timezone, not UTC: 17:00Z is
  // 10:00 in America/Los_Angeles, and reading a UTC stamp off a phone at a
  // glance is exactly the sort of thing that gets a meeting missed.
  assert.match(report, /BOOKED: Dana Reyes, Fri, Aug 28, 10:00 AM \(pending\)/, "the booking is attributed, in local time");

  // Notes belong to the CALL, not the number. A later call from the same phone
  // must not inherit what the previous caller said.
  db.recordCallStart("CA-3", B);
  db.recordCallEnd("CA-3");
  const shared = summarizeRecentCalls(24, 10, NOW, 48 * 3600);
  const thirdLine = shared.split("\n").find((l) => l.includes("nothing noted") && l.includes(B));
  assert.ok(thirdLine, "the later call from the same number reports nothing noted");
  assert.ok(
    !thirdLine.includes("Dana Reyes"),
    "a note from the earlier call is NOT attributed to the later caller"
  );

  // Repeat callers are flagged, which is what turns a list of numbers into
  // something worth acting on.
  assert.match(shared, /earlier calls? in the last 30 days/, "repeat callers are surfaced");

  // A call still in progress is not reported as a zero-length call.
  db.recordCallStart("CA-4", A);
  const live = summarizeRecentCalls(24, 10, NOW, 48 * 3600);
  assert.match(live, /still on the call/, "an unfinished call is described as live");

  // The window actually filters. Nothing was seeded older than an hour, so a
  // one-hour window still sees everything, but the header must follow the ask.
  assert.match(
    summarizeRecentCalls(168, 10, NOW, 48 * 3600),
    /in the last 168 hours/,
    "the window is echoed back so the answer is unambiguous"
  );

  console.log("Owner call-report verification passed:");
  console.log("  ✓ an empty log on a long-lived instance reports plainly");
  console.log("  ✓ an empty log on a young instance says the history is missing");
  console.log("  ✓ counts calls and how many booked");
  console.log("  ✓ quotes per-call notes with their category");
  console.log("  ✓ attributes the booking to the call it came from");
  console.log("  ✓ notes are per call, never inherited from a shared phone");
  console.log("  ✓ repeat callers are flagged");
  console.log("  ✓ a call in progress is not reported as zero length");
  console.log("  ✓ the requested window is echoed back");
}

void main();
