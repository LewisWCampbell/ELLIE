/**
 * Offline verification of the query log and the knowledge-gap report.
 *
 * Two properties matter more than the report's arithmetic:
 *
 *  1. What does NOT get logged. This table records what people say, so the
 *     filter is a privacy boundary as much as a signal-to-noise one. Names,
 *     email addresses and phone numbers spoken during booking must never
 *     reach it, and the schema must have nowhere to put a caller's identity
 *     even if some future call site tried.
 *
 *  2. That a miss is really a miss. The whole report is worthless if it cries
 *     gap over questions the corpus answers fine, because a digest that is
 *     mostly false alarms gets ignored, and then the real gap in it is ignored
 *     too.
 *
 * Runs against a small corpus built here rather than the real one, so it works
 * in a clone that does not have the private knowledge/ directory.
 *
 * Run: npx tsx src/gaps.verify.ts
 */
import assert from "node:assert";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

async function main(): Promise<void> {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "gaps-verify-"));
  const corpusDir = path.join(tmp, "knowledge");
  fs.mkdirSync(corpusDir);
  fs.writeFileSync(
    path.join(corpusDir, "projects.md"),
    `# Hollywood
A computer vision project that reconstructs film sets from archival photographs.
Built with photogrammetry and a custom camera solver.

# Elle
A telephone assistant that answers questions about the portfolio and books
intro calls. Runs on Cloud Run with a realtime voice bridge.
`
  );
  fs.writeFileSync(
    path.join(corpusDir, "education.md"),
    `# Northfield University
Studied computer science. Focused on graphics and interactive systems.
`
  );

  process.env.SQLITE_PATH = path.join(tmp, "gaps.db");
  process.env.KNOWLEDGE_DIR = corpusDir;
  process.env.TELEGRAM_BOT_TOKEN = "test-token";
  process.env.TELEGRAM_OWNER_ID = "424242";
  process.env.GAP_DIGEST_DAYS = "7";
  process.env.GAP_RETENTION_DAYS = "90";

  const sent: string[] = [];
  const realFetch = globalThis.fetch;
  globalThis.fetch = (async (_url: string | URL, init?: RequestInit) => {
    sent.push(String(init?.body ?? ""));
    return new Response(JSON.stringify({ ok: true }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }) as typeof fetch;

  try {
    const knowledge = await import("./knowledge");
    knowledge.useRetriever(knowledge.buildRetrieverFrom(corpusDir));

    const gaps = await import("./gaps");
    const db = await import("./db");

    // --- 1. the filter ----------------------------------------------------
    const shouldLog = [
      "What did he build at Northfield?",
      "tell me about the hollywood project",
      "how does the voice bridge handle interruptions",
      "Does he have experience with computer vision?",
      "im curious about his photogrammetry work", // no question mark, no interrogative
    ];
    for (const q of shouldLog) {
      assert.ok(gaps.isQuestion(q), `should log: ${q}`);
    }

    const shouldNotLog = [
      "yeah",
      "okay sounds good",
      "thanks so much",
      "My name is Lewis Campbell",
      "this is Jordan from Acme",
      "I'm Sam",
      "lewis@example.com",
      "my number is 555-0142",
      "how about 3pm",
      "sure",
      "",
    ];
    for (const q of shouldNotLog) {
      assert.ok(!gaps.isQuestion(q), `should NOT log: ${q}`);
    }

    // --- 2. the schema holds no identity ----------------------------------
    gaps.logQuestion("phone", "What did he study at Northfield?");
    const rows = db.getQueriesSince(new Date(Date.now() - 86400000).toISOString());
    assert.strictEqual(rows.length, 1, "the question was logged");
    const columns = Object.keys(rows[0]);
    for (const forbidden of ["phone_number", "call_sid", "caller", "from"]) {
      assert.ok(!columns.includes(forbidden), `query log must not carry ${forbidden}`);
    }

    // The bound conversion: a row written seconds ago must fall inside a
    // window that starts today. An ISO bound compared raw against SQLite's
    // "YYYY-MM-DD HH:MM:SS" sorts wrong on the T and silently drops it.
    const todayOnly = db.getQueriesSince(
      new Date(Date.now() - 60 * 1000).toISOString()
    );
    assert.strictEqual(todayOnly.length, 1, "a fresh row is inside a one-minute window");

    // --- 3. hits and misses -----------------------------------------------
    const hit = gaps.scoreQuestion("what is the hollywood photogrammetry project");
    assert.ok(hit.answered, "an on-corpus question is answerable");
    assert.strictEqual(hit.topSource, "projects.md");

    const miss = gaps.scoreQuestion("does he know anything about maritime insurance law");
    assert.ok(!miss.answered, "an off-corpus question is a gap");

    // Coverage, not score, is what catches this one: "project" is in the
    // corpus, so BM25 returns something, but the subject is absent.
    const sneaky = gaps.scoreQuestion("what was his sailing project in Norway");
    assert.ok(!sneaky.answered, "one incidental word match is not an answer");

    // --- 4. clustering ----------------------------------------------------
    for (const q of [
      "What did he do at Northfield?",
      "tell me about his time at Northfield",
      "Northfield University, what did he study there",
      "does he know anything about maritime insurance law",
      "any experience with maritime insurance",
    ]) {
      gaps.logQuestion("phone", q);
    }
    const report = gaps.gapReport(7);
    assert.ok(report.total >= 6, `logged questions: ${report.total}`);

    const northfield = report.topics.find((t) => /northfield/i.test(t.example));
    assert.ok(northfield, "Northfield questions appear in the report");
    assert.ok(northfield.count >= 3, `Northfield variants cluster together, got ${northfield.count}`);

    const maritime = report.gaps.find((g) => /maritime/i.test(g.example));
    assert.ok(maritime, "the uncovered topic is reported as a gap");
    assert.strictEqual(maritime.count, 2, "both phrasings cluster into one gap");
    assert.ok(
      !report.gaps.some((g) => /northfield/i.test(g.example)),
      "a covered topic is not reported as a gap"
    );

    // --- 5. the digest ----------------------------------------------------
    const text = gaps.formatGapReport(report);
    assert.ok(text.includes("Could not answer"), "the digest leads with gaps");
    assert.ok(/maritime/i.test(text), "the gap is named in the digest");
    assert.ok(!/\+1|555|@/.test(text), "the digest carries no contact details");

    const first = await gaps.maybeSendDigest();
    assert.strictEqual(first, "sent", "the first digest sends");
    assert.strictEqual(sent.length, 1, "exactly one message went out");
    assert.ok(/maritime/i.test(sent[0]), "the sent message is the report");

    const second = await gaps.maybeSendDigest();
    assert.strictEqual(second, "not-due", "a second digest inside the window does not send");
    assert.strictEqual(sent.length, 1, "and no second message went out");

    // --- 6. retention -----------------------------------------------------
    const kept = db.getQueriesSince(new Date(Date.now() - 7 * 86400000).toISOString()).length;
    assert.ok(kept > 0, "recent rows survive the prune");
    const dropped = db.pruneQueries(new Date(Date.now() + 86400000).toISOString());
    assert.strictEqual(dropped, kept, "a prune past every row removes them all");
    assert.strictEqual(
      db.getQueriesSince(new Date(0).toISOString()).length,
      0,
      "the log is empty after a full prune"
    );

    // --- 7. the off switch ------------------------------------------------
    const { config } = await import("./config");
    config.gaps.enabled = false;
    gaps.logQuestion("phone", "What did he build at Northfield?");
    assert.strictEqual(
      db.getQueriesSince(new Date(0).toISOString()).length,
      0,
      "QUERY_LOG=0 logs nothing at all"
    );
    config.gaps.enabled = true;

    console.log("gaps.verify: all checks passed");
  } finally {
    globalThis.fetch = realFetch;
    fs.rmSync(tmp, { recursive: true, force: true });
  }
}

void main().catch((e) => {
  console.error(e);
  process.exit(1);
});
