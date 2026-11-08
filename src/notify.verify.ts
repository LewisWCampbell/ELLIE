/**
 * Offline verification of owner notifications and reminders.
 *
 * The property under test is restraint as much as delivery. A channel that
 * pings for every wrong number gets muted, and then the booking notification
 * that actually mattered is muted with it, so "does not send" is checked as
 * carefully as "sends".
 *
 * Run: npx tsx src/notify.verify.ts
 */
import assert from "node:assert";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

async function main(): Promise<void> {
  const dbPath = path.join(os.tmpdir(), `notify-verify-${process.pid}.db`);
  process.env.SQLITE_PATH = dbPath;
  process.env.TELEGRAM_BOT_TOKEN = "test-token";
  process.env.TELEGRAM_OWNER_ID = "424242";
  process.env.PROFILE_NAME = "Lewis Campbell";
  process.env.ASSISTANT_NAME = "Elle";
  process.env.BOOKING_TIMEZONE = "America/Los_Angeles";
  process.env.TWILIO_ACCOUNT_SID = "ACtest";
  process.env.TWILIO_FROM_NUMBER = "+13412013991";
  process.env.TWILIO_AUTH_TOKEN = "tok";

  const sent: Array<{ url: string; body: string }> = [];
  const realFetch = globalThis.fetch;
  globalThis.fetch = (async (url: string | URL, init?: RequestInit) => {
    sent.push({ url: String(url), body: String(init?.body ?? "") });
    return new Response(JSON.stringify({ ok: true, sid: "SMtest" }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }) as typeof fetch;

  try {
    const db = await import("./db");
    const notify = await import("./notify");
    const { sendDueReminders, reminderText } = await import("./booking/reminders");

    // --- a booking reaches the owner, with what he needs to act on ---------
    sent.length = 0;
    notify.notifyBooking({
      name: "Dana Reyes",
      email: "dana@example.com",
      topic: "contract work",
      startISO: "2026-09-03T21:00:00.000Z",
      phoneNumber: "+14155550134",
      placed: "penciled in as TENTATIVE",
      eventLink: "https://calendar.google.com/x",
    });
    await new Promise((r) => setTimeout(r, 10));
    assert.strictEqual(sent.length, 1, "a booking notifies the owner");
    const booking = decodeURIComponent(sent[0].body);
    for (const needle of ["Dana Reyes", "dana@example.com", "contract work", "+14155550134"]) {
      assert.ok(booking.includes(needle), `the booking notice carries ${needle}`);
    }
    assert.ok(/2:00.PM/.test(booking), "the time is in the owner's timezone, not UTC");

    // --- a call with something in it is reported ---------------------------
    sent.length = 0;
    notify.notifyCallEnded({
      phoneNumber: "+15550100142",
      seconds: 253,
      turns: 9,
      facts: [{ category: "interest", fact: "Asked about contract availability." }],
      booked: false,
    });
    await new Promise((r) => setTimeout(r, 10));
    assert.strictEqual(sent.length, 1, "a real conversation is reported");
    assert.ok(decodeURIComponent(sent[0].body).includes("4m 13s"), "duration is human-readable");

    // --- silence is the point: three cases that must NOT notify -----------
    sent.length = 0;
    notify.notifyCallEnded({ phoneNumber: "+1", seconds: 3, turns: 0, facts: [], booked: false });
    notify.notifyCallEnded({
      phoneNumber: "+1",
      seconds: 300,
      turns: 12,
      facts: [{ category: "name", fact: "x" }],
      booked: true,
    });
    await new Promise((r) => setTimeout(r, 10));
    assert.strictEqual(sent.length, 0, "a silent wrong number and an already-notified booking stay quiet");

    // --- reminders ---------------------------------------------------------
    const soon = new Date(Date.now() + 20 * 3_600_000).toISOString();
    const later = new Date(Date.now() + 20 * 24 * 3_600_000).toISOString();
    db.saveBooking({
      phoneNumber: "+14155550134", callSid: null, name: "Dana Reyes", email: "d@example.com",
      topic: null, startAt: soon, status: "pending", eventLink: null,
    });
    db.saveBooking({
      phoneNumber: "+13105550111", callSid: null, name: "Far Future", email: "f@example.com",
      topic: null, startAt: later, status: "pending", eventLink: null,
    });

    sent.length = 0;
    const run = await sendDueReminders();
    assert.strictEqual(run.sent, 1, "only the booking inside the window is reminded");
    const sms = sent.find((x) => x.url.includes("Messages.json"));
    assert.ok(sms, "the reminder goes out as an SMS");
    const body = decodeURIComponent(sms.body.replace(/\+/g, " "));
    assert.ok(body.includes("Dana"), "addressed by first name");
    assert.ok(/STOP/.test(body), "carries the opt-out instruction");

    // --- running twice does not text twice ---------------------------------
    sent.length = 0;
    const again = await sendDueReminders();
    assert.strictEqual(again.sent, 0, "a second run sends nothing: the claim is the idempotence key");

    // --- an opted-out number is never reminded -----------------------------
    const optOut = new Date(Date.now() + 21 * 3_600_000).toISOString();
    db.saveBooking({
      phoneNumber: "+15550100177", callSid: null, name: "Opted Out", email: "o@example.com",
      topic: null, startAt: optOut, status: "pending", eventLink: null,
    });
    db.setSmsConsent("+15550100177", "opted_out", "sms_stop");
    sent.length = 0;
    const third = await sendDueReminders();
    assert.strictEqual(third.sent, 0, "consent is checked at send time, not at booking time");
    assert.strictEqual(third.skippedNoConsent, 1, "and the skip is counted, not silent");

    // --- the text itself ---------------------------------------------------
    const text = reminderText("Dana Reyes", soon);
    assert.ok(text.startsWith("Hi Dana,"), "first name only, the way a person writes");
    assert.ok(text.length <= 320, `reminder fits a registered keyword reply (${text.length})`);

    console.log("Notification and reminder verification passed:");
    console.log("  ✓ a booking reaches the owner with name, email, topic, number and local time");
    console.log("  ✓ a real conversation is reported with a readable duration");
    console.log("  ✓ a silent wrong number is NOT reported");
    console.log("  ✓ a call that booked is not reported twice");
    console.log("  ✓ only bookings inside the reminder window are texted");
    console.log("  ✓ running the job twice texts once");
    console.log("  ✓ an opted-out number is skipped and counted");
    console.log("  ✓ the reminder is first-name, opted-out-able, and inside 320 characters");
  } finally {
    globalThis.fetch = realFetch;
    fs.rmSync(dbPath, { force: true });
  }
}

void main();
