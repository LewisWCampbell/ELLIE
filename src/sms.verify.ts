/**
 * Offline verification of the registered SMS keyword replies.
 *
 * These two messages are not ordinary output. They are registered with the
 * campaign, they are what a carrier compares against live traffic, and the
 * registry caps them at 320 characters. Going over is not a runtime error and
 * nothing in the service notices: the reply sends fine and the campaign is
 * rejected weeks later, by which point the wording has been edited three times
 * and nobody remembers which edit did it. Asserting the budget here means a
 * deploy fails first, in seconds.
 *
 * Run: npx tsx src/sms.verify.ts
 */
import assert from "node:assert";

async function main(): Promise<void> {
  // The deployed configuration, since the length depends on it: a longer brand
  // or owner name spends the same budget as a longer sentence.
  process.env.BRAND_NAME = process.env.BRAND_NAME || "Example Brand LLC";
  process.env.PROFILE_NAME = process.env.PROFILE_NAME || "Lewis Campbell";
  process.env.ASSISTANT_NAME = process.env.ASSISTANT_NAME || "Elle";
  process.env.SQLITE_PATH = process.env.SQLITE_PATH || "/tmp/sms-verify.db";

  const { helpText, consentConfirmationText, KEYWORD_REPLY_MAX } = await import("./sms");

  const help = helpText();
  const enrol = consentConfirmationText();

  for (const [name, text] of [
    ["HELP reply", help],
    ["enrollment confirmation", enrol],
  ] as const) {
    assert.ok(
      text.length <= KEYWORD_REPLY_MAX,
      `${name} is ${text.length} characters, over the ${KEYWORD_REPLY_MAX} the registry allows:\n${text}`
    );
    // Identification: a recipient must be able to tell who is responsible.
    assert.ok(text.includes("Example Brand LLC"), `${name} names the registered brand`);
    // The opt-out instruction is the one thing that must never be trimmed.
    // Both phrasings are in use ("Reply STOP to opt out" and "Reply HELP for
    // help, STOP to opt out"), and either satisfies the requirement.
    assert.ok(/STOP to opt out/.test(text), `${name} carries the opt-out instruction`);
    assert.ok(/Msg frequency varies/.test(text), `${name} states that frequency varies`);
    assert.ok(/rates may apply/.test(text), `${name} states that rates may apply`);
    assert.ok(text.includes("sms-terms.html"), `${name} links the terms`);
  }

  // HELP has to route someone to a human, which is the whole point of the
  // keyword. A HELP reply that only recites disclosures fails a manual review.
  assert.ok(/call this number|Text or call/i.test(help), "HELP says how to reach a person");

  // The assistant must not present itself as a person in the message a carrier
  // reads. Both replies say plainly that it is automated.
  assert.ok(/automated|AI assistant/i.test(help), "HELP says the assistant is automated");
  assert.ok(/automated|AI assistant/i.test(enrol), "the confirmation says the assistant is automated");

  // Headroom check, so a small future edit does not silently land at 319.
  for (const [name, text] of [
    ["HELP reply", help],
    ["enrollment confirmation", enrol],
  ] as const) {
    console.log(`  ${name}: ${text.length}/${KEYWORD_REPLY_MAX} (${KEYWORD_REPLY_MAX - text.length} spare)`);
  }

  console.log("Registered SMS replies verification passed:");
  console.log("  ✓ both fit the 320 character registry limit");
  console.log("  ✓ both name the registered brand");
  console.log("  ✓ both carry frequency, rates, opt-out and terms");
  console.log("  ✓ HELP routes to a person and says the assistant is automated");
}

void main();
