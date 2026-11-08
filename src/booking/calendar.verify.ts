/**
 * Offline verification of the Calendar REST client — no Google, no credentials.
 * Stubs the token source and fetch, then checks what goes on the wire.
 *
 * The behaviour being pinned down is the guest fallback. Google refuses to let
 * a plain service account add attendees to an event, and that refusal applies
 * to any calendar merely SHARED with the account, which is the normal setup for
 * a personal Gmail calendar. Every booking attaches the caller as a guest, so
 * without a fallback every booking on such a calendar throws, gets caught one
 * layer up, and degrades to a lead: no hold ever reaches the calendar, while
 * the caller has been told their time is held.
 *
 * Run: npx tsx src/booking/calendar.verify.ts
 */
import assert from "node:assert";

interface Sent {
  url: string;
  method: string;
  body: Record<string, unknown>;
}

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

async function main(): Promise<void> {
  process.env.BOOKING_CALENDAR_ID = "lewis@example.com";
  process.env.BOOKING_TIMEZONE = "America/Los_Angeles";

  const { createCalendarClient } = await import("./calendar");

  const realFetch = globalThis.fetch;
  const sent: Sent[] = [];
  /** Replies in order; each entry answers one request. */
  let replies: Array<() => Response> = [];

  globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
    sent.push({
      url: String(url),
      method: init?.method ?? "GET",
      body: init?.body ? JSON.parse(String(init.body)) : {},
    });
    const next = replies.shift();
    if (!next) throw new Error(`unexpected request: ${String(url)}`);
    return next();
  }) as typeof fetch;

  try {
    const cal = createCalendarClient(async () => "fake-token");
    assert.ok(cal, "a client is built when a calendar id is configured");

    const details = {
      summary: "Intro call",
      description: "Caller: Dana <dana@example.com>",
      startISO: "2026-09-01T17:00:00.000Z",
      endISO: "2026-09-01T17:30:00.000Z",
      timezone: "America/Los_Angeles",
      attendeeEmail: "dana@example.com",
    };

    // --- the happy path: guests are allowed, one request, guest attached -----
    sent.length = 0;
    replies = [() => jsonResponse(200, { id: "ev1", htmlLink: "https://cal/ev1" })];
    const ok = await cal.createTentative(details);
    assert.strictEqual(ok.id, "ev1", "returns the created event id");
    assert.strictEqual(sent.length, 1, "one request when guests are permitted");
    assert.deepStrictEqual(
      sent[0].body.attendees,
      [{ email: "dana@example.com" }],
      "the caller is attached as a guest"
    );
    assert.strictEqual(sent[0].body.status, "tentative", "the hold is tentative, not confirmed");
    assert.strictEqual(ok.guestAttached, true, "reports that the guest was attached");
    assert.ok(String(sent[0].url).includes("sendUpdates=none"), "nobody is emailed on insert");

    // --- the personal-calendar path: guests refused, hold still created ------
    sent.length = 0;
    replies = [
      () =>
        jsonResponse(403, {
          error: {
            code: 403,
            message:
              "Service accounts cannot invite attendees without Domain-Wide Delegation of Authority.",
          },
        }),
      () => jsonResponse(200, { id: "ev2", htmlLink: "https://cal/ev2" }),
    ];
    const fallback = await cal.createTentative(details);
    assert.strictEqual(fallback.id, "ev2", "the hold is created on the retry");
    assert.strictEqual(fallback.guestAttached, false, "reports that the guest was dropped");
    assert.strictEqual(sent.length, 2, "exactly one retry, not a loop");
    assert.ok("attendees" in sent[0].body, "the first attempt did include the guest");
    assert.ok(!("attendees" in sent[1].body), "the retry drops the guest");
    assert.strictEqual(
      sent[1].body.description,
      details.description,
      "the caller's email survives in the description, so nothing is lost"
    );

    // --- any other failure still throws rather than silently retrying -------
    sent.length = 0;
    replies = [() => jsonResponse(403, { error: { message: "Insufficient permissions" } })];
    await assert.rejects(
      () => cal.createTentative(details),
      /events\.insert 403/,
      "a read-only share is a real error, not something to retry around"
    );
    assert.strictEqual(sent.length, 1, "no retry on an unrelated 403");

    sent.length = 0;
    replies = [() => jsonResponse(404, { error: { message: "Not Found" } })];
    await assert.rejects(() => cal.createTentative(details), /events\.insert 404/, "a wrong calendar id throws");

    // --- free/busy reads the right calendar ---------------------------------
    sent.length = 0;
    replies = [() => jsonResponse(200, { calendars: { "lewis@example.com": { busy: [] } } })];
    assert.strictEqual(await cal.isFree(details.startISO, details.endISO), true, "empty busy means free");

    sent.length = 0;
    replies = [
      () =>
        jsonResponse(200, {
          calendars: { "lewis@example.com": { busy: [{ start: details.startISO, end: details.endISO }] } },
        }),
    ];
    assert.strictEqual(await cal.isFree(details.startISO, details.endISO), false, "a busy block means busy");

    // --- an unreadable calendar is an ERROR, never "free" --------------------
    // freeBusy answers 200 and reports the problem per calendar. Reading the
    // empty busy list out of that response makes an inaccessible calendar look
    // wide open, which is how the live check came back "OK, slot is free" about
    // a calendar the account could not see at all.
    sent.length = 0;
    replies = [
      () =>
        jsonResponse(200, {
          calendars: {
            "lewis@example.com": { errors: [{ domain: "global", reason: "notFound" }], busy: [] },
          },
        }),
    ];
    await assert.rejects(
      () => cal.isFree(details.startISO, details.endISO),
      /could not read|not shared/i,
      "a calendar the account cannot see is an error, not a free slot"
    );

    sent.length = 0;
    replies = [() => jsonResponse(200, { calendars: {} })];
    await assert.rejects(
      () => cal.isFree(details.startISO, details.endISO),
      /cannot see this calendar/i,
      "no entry for the calendar is also an error, not a free slot"
    );

    // --- cancel tolerates an event that is already gone ---------------------
    sent.length = 0;
    replies = [() => new Response("", { status: 410 })];
    await cal.cancelEvent?.("ev1");
    assert.strictEqual(sent.length, 1, "410 Gone is treated as already cancelled");

    console.log("Calendar client verification passed:");
    console.log("  ✓ a tentative hold is created with the caller as a guest");
    console.log("  ✓ nobody is emailed on insert (sendUpdates=none)");
    console.log("  ✓ when guests are refused, the hold is retried without one");
    console.log("  ✓ the result says whether the guest survived");
    console.log("  ✓ that retry happens exactly once, and keeps the description");
    console.log("  ✓ unrelated 403s and 404s still throw rather than retry");
    console.log("  ✓ free/busy reads free and busy correctly");
    console.log("  ✓ an unreadable calendar raises rather than reporting free");
    console.log("  ✓ cancelling an already-gone event is not an error");
  } finally {
    globalThis.fetch = realFetch;
  }
}

void main();
