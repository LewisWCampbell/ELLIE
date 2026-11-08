/**
 * Offline verification of the spoken greeting.
 *
 * This suite exists because of a bug that shipped and stayed shipped. The
 * opening line read perfectly on screen and stumbled out loud: "this is Elle,
 * Lewis Campbell's AI assistant" puts a comma between two names, the TTS
 * lengthens comma pauses, and callers heard the owner's name break in half.
 * Thirteen suites passed throughout, because every one of them stubbed the
 * audio away and none could even see the string.
 *
 * The checks below are about PROSODY, which is the property nothing else in
 * the test suite looks at: where the pauses land, how often a name is said,
 * how long the caller waits before they can speak.
 *
 * Run: npx tsx src/greeting.verify.ts
 */
import assert from "node:assert";

/** Aura speaks at roughly 150 words a minute; a comma adds a beat. */
const WORDS_PER_MINUTE = 150;
const COMMA_PAUSE_MS = 300;

function spokenSeconds(text: string): number {
  const words = text.trim().split(/\s+/).length;
  const commas = (text.match(/,/g) ?? []).length;
  return (words / WORDS_PER_MINUTE) * 60 + (commas * COMMA_PAUSE_MS) / 1000;
}

async function main(): Promise<void> {
  process.env.PROFILE_NAME = "Lewis Campbell";
  process.env.ASSISTANT_NAME = "Elle";
  process.env.SITE_URL = "https://lewiswcampbell.com";
  process.env.SQLITE_PATH = "/tmp/greeting-verify.db";

  const { buildGreeting } = await import("./realtime/twilioBridge");
  const greeting = buildGreeting(false);
  const owner = "Lewis Campbell";

  // --- the bug that shipped ------------------------------------------------
  // A name pressed against a comma gets split by the pause. This is the exact
  // shape that produced "Elle... Lewis... Campbell's".
  assert.ok(
    !new RegExp(`,\\s*${owner}`).test(greeting),
    `the owner's full name must not follow a comma (the pause splits it):\n${greeting}`
  );
  assert.ok(
    !new RegExp(`${owner}\\s*,`).test(greeting),
    `the owner's full name must not be followed by a comma`
  );

  // --- said once, then first name ------------------------------------------
  // Repeating a full name is how a form reads, not how a person talks.
  const fullNameCount = greeting.split(owner).length - 1;
  assert.strictEqual(fullNameCount, 1, `the full name is said exactly once, not ${fullNameCount}`);
  // The property, not one phrasing of it: the first name appears on its own
  // somewhere after the single full-name introduction.
  const afterIntro = greeting.slice(greeting.indexOf(owner) + owner.length);
  assert.ok(
    /\bLewis\b/.test(afterIntro),
    "after introducing him once by full name, later mentions use the first name"
  );

  // --- the caller is not made to wait --------------------------------------
  // The greeting is uninterruptible by design, so its length is dead air the
  // caller cannot escape. Past about twenty-five seconds people hang up.
  const seconds = spokenSeconds(greeting);
  assert.ok(seconds < 25, `greeting takes ${seconds.toFixed(1)}s to speak, which is too long to sit through`);

  // --- it must actually invite a reply -------------------------------------
  assert.ok(/\?\s*$/.test(greeting.trim()), "the greeting ends on a question, so the caller knows it is their turn");

  // --- speakable ------------------------------------------------------------
  assert.ok(!/https?:\/\//.test(greeting), "no raw URL scheme: the voice would read out 'h t t p s colon'");
  assert.ok(!/[<>*_#`]/.test(greeting), "no markup characters, which the voice reads literally");

  // --- the recorded variant leads with the notice --------------------------
  const recorded = buildGreeting(true);
  assert.ok(
    recorded.startsWith("Just so you know, this call is recorded."),
    "when recording is on, consent is the first thing said, before anything else"
  );
  assert.ok(recorded.includes(greeting), "the notice is added to the greeting rather than replacing part of it");

  console.log("Greeting verification passed:");
  console.log(`  ✓ no name sits against a comma pause (the bug that shipped)`);
  console.log(`  ✓ the full name is spoken once; later mentions are first-name`);
  console.log(`  ✓ ${seconds.toFixed(1)}s to speak, inside the 25s a caller will sit through`);
  console.log(`  ✓ ends on a question, so the caller knows it is their turn`);
  console.log(`  ✓ nothing in it that a voice would read literally`);
  console.log(`  ✓ the recording notice leads, and adds rather than replaces`);
}

void main();
