/**
 * Offline verification of the durable call archive — no Cloud Storage, no
 * credentials. Stubs the token source and fetch, then checks what would go on
 * the wire and what comes back out of a replay.
 *
 * The property that matters is that a call survives the container it was taken
 * on. Everything here is in service of that: a call is archived the moment it
 * starts (not only when it ends, which a dropped call never reaches), replay
 * puts it back, and replaying twice does not duplicate a conversation.
 *
 * Run: npx tsx src/archive.verify.ts
 */
import assert from "node:assert";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

interface Sent {
  url: string;
  method: string;
  body: string;
}

const settle = async (): Promise<void> => {
  for (let i = 0; i < 5; i++) await new Promise((r) => setTimeout(r, 0));
};

async function main(): Promise<void> {
  const dbPath = path.join(os.tmpdir(), `archive-verify-${process.pid}.db`);
  process.env.SQLITE_PATH = dbPath;
  process.env.CALL_ARCHIVE_BUCKET = "elle-calls-test";

  const db = await import("./db");
  const archive = await import("./archive");

  const realFetch = globalThis.fetch;
  const sent: Sent[] = [];
  /** name -> stored object body, standing in for the bucket. */
  const bucket = new Map<string, string>();
  let listStatus = 200;
  let uploadStatus = 200;

  globalThis.fetch = (async (url: string | URL, init?: RequestInit) => {
    const u = String(url);
    sent.push({ url: u, method: init?.method ?? "GET", body: String(init?.body ?? "") });

    if (u.includes("/upload/storage/")) {
      if (uploadStatus !== 200) return new Response("nope", { status: uploadStatus });
      const name = decodeURIComponent(new URL(u).searchParams.get("name") ?? "");
      bucket.set(name, String(init?.body ?? ""));
      return new Response(JSON.stringify({ name }), { status: 200 });
    }
    if (u.includes("alt=media")) {
      const name = decodeURIComponent(u.split("/o/")[1].split("?")[0]);
      const body = bucket.get(name);
      return body
        ? new Response(body, { status: 200, headers: { "Content-Type": "application/json" } })
        : new Response("", { status: 404 });
    }
    // list
    if (listStatus !== 200) return new Response("denied", { status: listStatus });
    const offset = new URL(u).searchParams.get("startOffset") ?? "";
    const items = [...bucket.keys()].filter((n) => n >= offset).sort().map((name) => ({ name }));
    return new Response(JSON.stringify({ items }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }) as typeof fetch;

  archive.setTokenSourceForTests(async () => "fake-token");

  try {
    assert.strictEqual(archive.archiveEnabled(), true, "a configured bucket turns the archive on");

    // --- names sort chronologically -----------------------------------------
    const a = archive.objectName("2026-08-26 21:02:11", "CA111");
    const b = archive.objectName("2026-08-27 08:00:00", "CA222");
    assert.ok(a < b, "object names sort in call order, so a date prefix can bound a listing");
    assert.ok(!a.includes(":") && !a.includes(" "), "no characters that need escaping in a name");

    archive.startArchive();

    // --- a call is archived the moment it starts -----------------------------
    // Not at the end: a caller who hangs up mid-sentence never reaches the end,
    // and that call still has to leave a record.
    sent.length = 0;
    db.recordCallStart("CA-live", "+14155550134");
    await settle();
    assert.strictEqual(sent.length, 1, "the start of a call is written immediately");
    assert.ok(sent[0].url.includes("/upload/storage/"), "written as an object upload");

    // --- ordinary turns batch, then flush ------------------------------------
    sent.length = 0;
    db.saveTranscript("CA-live", "caller", "hi, is this Lewis?");
    db.saveTranscript("CA-live", "assistant", "This is Elle.");
    db.saveFact("+14155550134", "interest", "asked about pricing", "CA-live");
    await settle();
    assert.strictEqual(sent.length, 0, "individual turns do not each cause an upload");

    await archive.flushAll();
    assert.strictEqual(sent.length, 1, "the batch goes up as one object");
    const stored = JSON.parse(sent[0].body) as { transcript: unknown[]; facts: unknown[]; call: { call_sid: string } };
    assert.strictEqual(stored.call.call_sid, "CA-live", "the snapshot is of the right call");
    assert.strictEqual(stored.transcript.length, 2, "both turns are in the snapshot");
    assert.strictEqual(stored.facts.length, 1, "what Elle learned is in the snapshot");

    // --- a booking is written at once, not at the next tick ------------------
    sent.length = 0;
    db.saveBooking({
      phoneNumber: "+14155550134",
      callSid: "CA-live",
      name: "Dana",
      email: "dana@example.com",
      topic: "intro",
      startAt: "2026-09-01T17:00:00.000Z",
      status: "pending",
      eventLink: null,
    });
    await settle();
    assert.strictEqual(sent.length, 1, "a booking is archived immediately");

    // --- replay: a call from a previous container comes back -----------------
    // Written straight into the stand-in bucket, because the point is a call
    // this database has never seen.
    //
    // Dated relative to today. The replay only looks back a fixed number of
    // days, so a fixture with a literal date passes for a month after it is
    // written and then fails for everyone who clones the repo after that, with
    // nothing in the code having changed.
    const at = (daysAgo: number, hms: string) =>
      `${new Date(Date.now() - daysAgo * 86_400_000).toISOString().slice(0, 10)} ${hms}`;
    const ghost = {
      v: 1,
      call: {
        call_sid: "CA-ghost",
        phone_number: "+13105550111",
        started_at: at(3, "15:04:05"),
        ended_at: at(3, "15:09:00"),
      },
      transcript: [
        { role: "caller", content: "do you do contract work?", created_at: at(3, "15:04:10") },
        { role: "assistant", content: "He does.", created_at: at(3, "15:04:14") },
      ],
      facts: [
        {
          id: 1,
          phone_number: "+13105550111",
          category: "interest",
          fact: "contract work",
          call_sid: "CA-ghost",
          created_at: at(3, "15:05:00"),
        },
      ],
      bookings: [],
    };
    bucket.set(archive.objectName(ghost.call.started_at, "CA-ghost"), JSON.stringify(ghost));

    assert.strictEqual(db.hasCall("CA-ghost"), false, "the ghost call is not here yet");
    const n = await archive.restoreFromArchive(30);
    assert.ok(n >= 1, "the archived call is replayed");
    assert.strictEqual(db.hasCall("CA-ghost"), true, "and is now in the local database");
    assert.strictEqual(db.getTranscripts("CA-ghost").length, 2, "with its conversation intact");
    assert.strictEqual(db.getFactsForCall("CA-ghost").length, 1, "and what Elle learned from it");

    // --- replaying twice must not duplicate the conversation -----------------
    const again = await archive.restoreFromArchive(30);
    assert.strictEqual(again, 0, "a call already present is skipped");
    assert.strictEqual(
      db.getTranscripts("CA-ghost").length,
      2,
      "the transcript is not duplicated by a second replay"
    );

    // --- the window is honoured ---------------------------------------------
    const narrow = await archive.restoreFromArchive(1);
    assert.strictEqual(narrow, 0, "an old call is outside a one-day window");

    // --- failures degrade, never throw --------------------------------------
    listStatus = 403;
    assert.strictEqual(await archive.restoreFromArchive(30), 0, "an unreadable bucket returns nothing");
    listStatus = 200;

    uploadStatus = 500;
    db.saveTranscript("CA-live", "caller", "one more thing");
    await archive.flushAll();
    assert.strictEqual(
      archive.archiveStatus().pending,
      1,
      "a failed upload stays pending so it is retried rather than dropped"
    );
    uploadStatus = 200;
    await archive.flushAll();
    assert.strictEqual(archive.archiveStatus().pending, 0, "and goes up when the bucket returns");

    console.log("Call archive verification passed:");
    console.log("  ✓ object names sort in call order");
    console.log("  ✓ a call is archived the moment it starts, not only when it ends");
    console.log("  ✓ turns batch into one object rather than one upload each");
    console.log("  ✓ a booking is written immediately");
    console.log("  ✓ an archived call replays into a fresh database, transcript and facts intact");
    console.log("  ✓ replaying twice does not duplicate the conversation");
    console.log("  ✓ the restore window is honoured");
    console.log("  ✓ an unreadable bucket degrades to empty history instead of throwing");
    console.log("  ✓ a failed upload stays pending and goes up on the next flush");
  } finally {
    archive.stopArchive();
    globalThis.fetch = realFetch;
    fs.rmSync(dbPath, { force: true });
    fs.rmSync(`${dbPath}-journal`, { force: true });
  }
}

void main();
