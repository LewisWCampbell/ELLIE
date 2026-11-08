/**
 * Offline verification of intro-call booking — no network, no Google Calendar.
 * Uses a throwaway DB and a fake CalendarClient to check: a free slot creates a
 * tentative hold and a pending booking row with the "confirm" messaging; a busy
 * slot is refused; bad input is rejected; and with no calendar it still captures
 * the lead.
 *
 * Run: npx tsx src/booking/booking.verify.ts
 */
import assert from "node:assert";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

async function main(): Promise<void> {
  const dbPath = path.join(os.tmpdir(), `letstalk-booking-${process.pid}.db`);
  for (const f of [dbPath, `${dbPath}-wal`, `${dbPath}-shm`]) {
    try {
      fs.rmSync(f);
    } catch {
      /* not there */
    }
  }
  process.env.SQLITE_PATH = dbPath;
  process.env.PROFILE_NAME = "Lewis";
  process.env.BOOKING_DURATION_MINUTES = "30";
  process.env.BOOKING_TIMEZONE = "America/Los_Angeles";

  const db = await import("../db");
  const { handleBookIntroCall, parseStart } = await import("./booking");

  // --- start-time parsing -------------------------------------------------
  // A string carrying its own offset is trusted exactly as written.
  assert.strictEqual(
    parseStart("2026-08-28T14:00:00-07:00", "America/Los_Angeles")?.toISOString(),
    "2026-08-28T21:00:00.000Z",
    "an explicit offset is honoured"
  );

  // One WITHOUT an offset is read as wall-clock time in the owner's timezone,
  // not as UTC. This is the bug that matters: JavaScript parses a bare ISO
  // string as local time, local time on Cloud Run is UTC, and a caller asking
  // for 2pm was being booked at 7am their time, after Elle read "two o'clock"
  // back to them and they said yes.
  assert.strictEqual(
    parseStart("2026-08-28T14:00:00", "America/Los_Angeles")?.toISOString(),
    "2026-08-28T21:00:00.000Z",
    "a bare time means 2pm in the owner's zone, not 2pm UTC"
  );
  assert.strictEqual(
    parseStart("2026-08-28 14:00", "America/Los_Angeles")?.toISOString(),
    "2026-08-28T21:00:00.000Z",
    "a space instead of T parses the same way"
  );

  // Winter, so the same wall-clock time is a different instant: PST not PDT.
  assert.strictEqual(
    parseStart("2026-01-15T14:00:00", "America/Los_Angeles")?.toISOString(),
    "2026-01-15T22:00:00.000Z",
    "the zone offset is taken at that date, so daylight saving is handled"
  );

  // A zone that never shifts, to prove the offset is not hardcoded anywhere.
  assert.strictEqual(
    parseStart("2026-08-28T14:00:00", "UTC")?.toISOString(),
    "2026-08-28T14:00:00.000Z",
    "UTC wall clock is UTC"
  );
  assert.strictEqual(parseStart("next tuesdayish", "UTC"), null, "unparseable input is rejected");

  const CALLER = "+15551234567";
  // Fixture times are relative to now. They used to be hardcoded dates, which
  // is why this suite started failing the moment the guard against booking in
  // the past was added: the fixture had quietly aged into the past.
  const inDays = (n: number): string => new Date(Date.now() + n * 86_400_000).toISOString();
  const good = { name: "Priya", email: "priya@example.com", start: inDays(3), topic: "data role" };

  // --- free slot → tentative hold + pending booking ---
  const created: Array<{ summary: string; startISO: string; attendeeEmail?: string }> = [];
  const freeCal = {
    async isFree(): Promise<boolean> {
      return true;
    },
    async createTentative(d: {
      summary: string;
      startISO: string;
      attendeeEmail?: string;
    }): Promise<{ htmlLink: string; id: string }> {
      created.push({ summary: d.summary, startISO: d.startISO, attendeeEmail: d.attendeeEmail });
      return { htmlLink: "https://calendar.google.com/event?eid=abc", id: "abc" };
    },
  };
  const msg = await handleBookIntroCall(good, { fromNumber: CALLER, callSid: "CA1" }, freeCal);
  assert.strictEqual(created.length, 1, "a tentative event was created");
  assert.match(created[0].summary, /tentative|pending/i, "event is marked pending");
  assert.strictEqual(created[0].attendeeEmail, "priya@example.com", "caller attached as a guest");
  assert.match(msg, /tentative/i, "reply tells Elle it's tentative");
  assert.match(msg, /confirm/i, "reply tells Elle to say it needs confirmation");
  const rows = db.getBookings(CALLER) as Array<{ status: string; email: string; event_link: string }>;
  assert.strictEqual(rows.length, 1, "a booking row was saved");
  assert.strictEqual(rows[0].status, "pending", "booking is pending");
  assert.strictEqual(rows[0].email, "priya@example.com", "email captured");
  assert.ok(rows[0].event_link.includes("calendar.google.com"), "event link stored");

  // --- busy slot → refused, no new booking ---
  const busyCal = {
    async isFree(): Promise<boolean> {
      return false;
    },
    async createTentative(): Promise<{ htmlLink: string; id: string }> {
      throw new Error("should not be called when busy");
    },
  };
  const busyMsg = await handleBookIntroCall(good, { fromNumber: CALLER, callSid: "CA2" }, busyCal);
  assert.match(busyMsg, /busy|different time/i, "busy slot asks for another time");
  assert.strictEqual((db.getBookings(CALLER) as unknown[]).length, 1, "no booking saved for a busy slot");

  // --- bad input → rejected before any calendar call ---
  const badEmail = await handleBookIntroCall({ name: "X", email: "not-an-email", start: good.start }, {}, freeCal);
  assert.match(badEmail, /^Error/, "invalid email rejected");
  const noTime = await handleBookIntroCall({ name: "X", email: "x@y.com" }, {}, freeCal);
  assert.match(noTime, /^Error/, "missing start rejected");
  assert.strictEqual(created.length, 1, "no extra calendar events from bad input");

  // --- a time in the past is refused before touching the calendar ---
  // Models get the year wrong, and a caller reading back "the 28th" cannot tell
  // which year Elle heard. A hold silently landing in the past is invisible.
  const past = await handleBookIntroCall(
    { name: "X", email: "x@y.com", start: inDays(-1) },
    {},
    freeCal
  );
  assert.match(past, /^Error/, "a past time is rejected");
  assert.match(past, /in the past/i, "and says why");

  const tooFar = await handleBookIntroCall(
    { name: "X", email: "x@y.com", start: inDays(400) },
    {},
    freeCal
  );
  assert.match(tooFar, /^Error/, "an absurd lead time is rejected");
  assert.match(tooFar, /year/i, "and points at the likely cause");
  assert.strictEqual(created.length, 1, "neither reached the calendar");

  // --- no calendar configured → capture the lead anyway ---
  const captureMsg = await handleBookIntroCall(
    { name: "Sam", email: "sam@co.com", start: inDays(5) },
    { fromNumber: "+15559998888" },
    null
  );
  assert.match(captureMsg, /confirm/i, "capture-only still promises confirmation");
  assert.strictEqual((db.getBookings("+15559998888") as unknown[]).length, 1, "lead captured without a calendar");

  // --- pending-bookings view lists everything awaiting confirmation ---
  const pending = db.getPendingBookings();
  assert.ok(pending.length >= 2, "pending view lists the outstanding requests");
  assert.ok(pending.every((b) => b.status === "pending"), "only pending rows are listed");
  assert.ok(pending.some((b) => b.email === "priya@example.com"), "the calendar booking is pending");

  console.log("Booking verification passed:");
  console.log("  ✓ free slot creates a tentative hold + pending booking, with 'confirm' messaging");
  console.log("  ✓ busy slot is refused and asks for another time");
  console.log("  ✓ invalid email / missing time are rejected before any calendar call");
  console.log("  ✓ a bare time is read in the owner's zone, not as UTC");
  console.log("  ✓ daylight saving is taken at the booked date");
  console.log("  ✓ times in the past, and a year out, are refused");
  console.log("  ✓ with no calendar, the lead is still captured");
  console.log("  ✓ pending-bookings view lists everything awaiting confirmation");

  // --- check_availability: the read half -----------------------------------
  const { handleCheckAvailability, nearbyOpenings } = await import("./availability");
  // A fixed Monday morning in Los Angeles, so weekdays and hours are exact.
  const NOW = new Date("2026-11-09T09:00:00-08:00");
  const at = (s: string) => new Date(s).toISOString();
  const busyCalOf = (ranges: Array<{ start: string; end: string }>) => ({
    async isFree(): Promise<boolean> {
      throw new Error("busy() should be used when available");
    },
    async busy() {
      return ranges;
    },
    async createTentative(): Promise<{ htmlLink: string; id: string }> {
      throw new Error("a read must never write");
    },
  });

  // Free.
  const free = await handleCheckAvailability({ start: "2026-11-12T14:00:00-08:00" }, busyCalOf([]), NOW);
  assert.match(free, /^FREE: Thursday, November 12 at 2:00 PM PST/, `free result: ${free}`);

  // Busy, with alternatives that skip the busy block, stay in hours, and come back in order.
  const blocked = busyCalOf([{ start: at("2026-11-12T13:00:00-08:00"), end: at("2026-11-12T15:00:00-08:00") }]);
  const busyRes = await handleCheckAvailability({ start: "2026-11-12T14:00:00-08:00" }, blocked, NOW);
  assert.match(busyRes, /^BUSY: Thursday, November 12 at 2:00 PM PST is taken\. Nearby open times: /, busyRes);
  assert.match(busyRes, /3:00 PM/, "the first slot after the block is offered");
  assert.match(busyRes, /12:30 PM/, "the last slot before the block is offered");
  assert.doesNotMatch(busyRes, /\b(1:00|1:30|2:30) PM/, "nothing inside the busy block is offered");

  // A bare time with no offset means the owner's wall clock, as it does when booking.
  const bare = await handleCheckAvailability({ start: "2026-11-12T14:00" }, busyCalOf([]), NOW);
  assert.match(bare, /2:00 PM PST/, "a bare time is read in the owner's zone");

  // Suggestions never land on a weekend or outside working hours.
  const friLate = new Date("2026-11-13T16:30:00-08:00");
  const alts = nearbyOpenings(friLate, [{ start: friLate.toISOString(), end: at("2026-11-13T17:00:00-08:00") }], NOW, {
    timeZone: "America/Los_Angeles",
    durationMs: 30 * 60_000,
    hours: { start: 9, end: 17 },
  });
  assert.strictEqual(alts.length, 3, "three suggestions");
  for (const d of alts) {
    const local = d.toLocaleString("en-US", { timeZone: "America/Los_Angeles", weekday: "short", hour: "numeric", hourCycle: "h23" });
    assert.ok(!/Sat|Sun/.test(local), `no weekend suggestion: ${local}`);
    const hour = Number(local.split(" ").pop());
    assert.ok(hour >= 9 && hour < 17, `inside working hours: ${local}`);
  }

  // A calendar error is UNKNOWN, never FREE: a failed read must not be spoken as a yes.
  const broken = {
    async isFree(): Promise<boolean> {
      throw new Error("down");
    },
    async busy(): Promise<Array<{ start: string; end: string }>> {
      throw new Error("freeBusy 503");
    },
    async createTentative(): Promise<{ htmlLink: string; id: string }> {
      throw new Error("unused");
    },
  };
  const unk = await handleCheckAvailability({ start: "2026-11-12T14:00:00-08:00" }, broken, NOW);
  assert.match(unk, /^UNKNOWN: /, "a failed read is UNKNOWN");
  assert.match(unk, /Don't tell the caller whether the time is free/, "and says what not to claim");
  assert.match(
    await handleCheckAvailability({ start: "2026-11-12T14:00:00-08:00" }, null, NOW),
    /^UNKNOWN: no calendar/,
    "no calendar connected is UNKNOWN too"
  );

  // A client without busy ranges still answers yes or no.
  const simple = { ...freeCal };
  assert.match(await handleCheckAvailability({ start: "2026-11-12T14:00:00-08:00" }, simple, NOW), /^FREE: /);

  // Bad input is refused before the calendar is touched.
  assert.match(await handleCheckAvailability({ start: "" }, broken, NOW), /^Error/);
  assert.match(await handleCheckAvailability({ start: "2026-11-01T14:00:00-08:00" }, broken, NOW), /^Error: .*past/);

  console.log("  ✓ check_availability: free, busy with nearby openings, bare times in the owner's zone");
  console.log("  ✓ suggestions skip weekends and stay inside working hours");
  console.log("  ✓ a failed or missing calendar is UNKNOWN, never FREE");
}

void main();
