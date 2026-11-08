/**
 * Offline verification of the RAG retriever — no network, no LLM. Builds the
 * index from the knowledge/ corpus and asserts that queries return the relevant
 * sections. Run: npx tsx src/knowledge.verify.ts
 */
import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";

// The corpus is private: it lives in the owner's working tree and in deploys,
// never in the public repository. A fresh clone has no knowledge/ directory,
// and that is expected, not a failure — so this suite tests the retriever only
// where there is a corpus to retrieve from.
if (!fs.existsSync(path.join(__dirname, "..", "knowledge"))) {
  console.log("Knowledge verification skipped: no knowledge/ directory.");
  console.log("The corpus is private and not part of the repository; the retriever");
  console.log("is exercised wherever the corpus exists (the owner's checkout, deploys).");
  process.exit(0);
}
import { buildRetrieverFrom } from "./knowledge";

const r = buildRetrieverFrom("knowledge");
assert.ok(r.size > 0, "corpus loaded some chunks");

function hits(query: string) {
  const h = r.search(query, 3);
  assert.ok(h.length > 0, `query returned results: ${query}`);
  return h;
}

// Each check names the section a question should land in, never what the
// corpus says. The corpus is private, and a public test that quoted it would
// publish it one assertion at a time.
function landsIn(query: string, file: string): boolean {
  return hits(query).some((h) => h.source === file);
}

assert.ok(landsIn("intern researcher consultant owner roles", "experience.md"), "work experience");
assert.ok(landsIn("machine learning projects he built", "projects.md"), "projects");
assert.ok(landsIn("master of science bachelor degree", "education.md"), "education");
assert.ok(landsIn("coach organizer league leadership", "leadership.md"), "leadership");
assert.ok(landsIn("technical skills soft skills certifications awards", "skills.md"), "skills");
assert.ok(landsIn("personality and values", "about.md"), "personality");
assert.ok(landsIn("interests and hobbies", "about.md"), "interests");
assert.strictEqual(r.search("xyzzy quux flibbertigibbet", 3).length, 0, "irrelevant query returns nothing");

console.log("Knowledge (RAG) verification passed:");
console.log(`  ✓ loaded ${r.size} chunks from knowledge/`);
console.log("  ✓ work / projects / education / leadership / skills queries hit the right sections");
console.log("  ✓ personality & interests queries hit the About section");
console.log("  ✓ irrelevant query returns nothing (no hallucinated grounding)");
console.log(`  overview: ${r.overview()}`);
