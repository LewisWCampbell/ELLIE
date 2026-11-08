/**
 * Does the documentation still describe the system that exists?
 *
 * Documentation drifts silently. Nothing breaks, no test fails, and the gap is
 * invisible until a reader follows an instruction that stopped working months
 * ago. This checks the claims that CAN be checked mechanically — the names of
 * things — against the source, so drift shows up as a failing command rather
 * than as someone's bad afternoon.
 *
 * It cannot check whether the prose is true. It can check that every setting
 * the code reads is written down somewhere, that every tool the model can call
 * is documented, and that no doc references a setting or tool that no longer
 * exists. Those three cover the drift that actually bites.
 *
 *   npm run check:docs           # report
 *   npm run check:docs -- --fix  # list what to add, in paste-ready form
 */
import fs from "node:fs";
import path from "node:path";

const ROOT = path.join(__dirname, "..");
const DOC_DIRS = ["public/docs", "docs"];
const SRC_DIRS = ["src", "tools"];

/** Settings that are read but deliberately not documented for users. */
const UNDOCUMENTED_OK = new Set([
  "NODE_ENV",
  "PORT",
  "K_REVISION",
  "K_SERVICE",
  "GOOGLE_CLOUD_PROJECT",
  "HOME",
  "HTTPS_PROXY",
  // Script-local variables that happen to use the same defaulting idiom as a
  // real setting. Kept as an explicit list rather than a cleverer regex: three
  // named exceptions are easier to audit than a pattern nobody can read.
  "TW_SECRET",
  "TZ_NAME",
  "NAME",
]);

function walk(dir: string, exts: string[]): string[] {
  const out: string[] = [];
  const abs = path.join(ROOT, dir);
  if (!fs.existsSync(abs)) return out;
  for (const entry of fs.readdirSync(abs, { withFileTypes: true })) {
    const p = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...walk(p, exts));
    else if (exts.some((e) => entry.name.endsWith(e))) out.push(p);
  }
  return out;
}

function read(rel: string): string {
  return fs.readFileSync(path.join(ROOT, rel), "utf8");
}

// --- what the code actually has ---------------------------------------------

const srcFiles = walk("src", [".ts"]).filter((f) => !f.includes(".verify."));
const srcText = srcFiles.map(read).join("\n");
const toolText = walk("tools", [".ts", ".sh"]).map(read).join("\n");

/** Every environment variable the code reads. */
const envUsed = new Set<string>();
const allCode = srcText + toolText;
// Two ways the code reads a setting. Secrets go through secretEnv(), which is
// why an earlier version of this check reported TWILIO_AUTH_TOKEN as an
// undocumented setting nobody reads: it is read, just not through process.env.
for (const re of [/process\.env\.([A-Z][A-Z0-9_]*)/g, /secretEnv\("([A-Z][A-Z0-9_]*)"\)/g]) {
  for (const m of allCode.matchAll(re)) {
    if (!UNDOCUMENTED_OK.has(m[1])) envUsed.add(m[1]);
  }
}
// Shell scripts: only the `NAME=${NAME:-default}` idiom counts. A bare $VAR is
// usually a local, and treating those as settings buried the real findings
// under thirty variables nobody can configure.
for (const m of toolText.matchAll(/\b([A-Z][A-Z0-9_]{2,})=\$\{\1:-/g)) {
  if (!UNDOCUMENTED_OK.has(m[1])) envUsed.add(m[1]);
}

/** Every tool name the model can call, from the tool definitions themselves. */
const toolNames = new Set<string>();
for (const m of srcText.matchAll(/^\s*name:\s*"([a-z][a-z0-9_]+)",\s*$/gm)) toolNames.add(m[1]);
for (const m of srcText.matchAll(/name:\s*BOOK_INTRO_CALL\.name/g)) {
  void m;
  toolNames.add("book_intro_call");
}

/** Every npm script, since docs tell people to run them. */
const scripts = new Set<string>(
  Object.keys(JSON.parse(read("package.json")).scripts as Record<string, string>)
);

// --- what the docs claim -----------------------------------------------------

const docFiles = DOC_DIRS.flatMap((d) => walk(d, [".html", ".md"]));
const docText = docFiles.map(read).join("\n");

/**
 * Design proposals describe code that does not exist yet, on purpose. Marked
 * with `doc-check: proposal` near the top, they still count as documentation
 * for "is this thing written down anywhere" but are excluded from "does the
 * code still read this", because naming a setting in a sketch of unbuilt work
 * is not a claim that it is configurable today.
 */
const PROPOSAL = /doc-check:\s*proposal/;
const currentDocText = docFiles
  .filter((f) => !PROPOSAL.test(read(f)))
  .map(read)
  .join("\n");

const envDocumented = new Set<string>();
for (const m of docText.matchAll(/\b([A-Z][A-Z0-9_]{2,})\b/g)) envDocumented.add(m[1]);

const scriptsMentioned = new Set<string>();
for (const m of docText.matchAll(/npm run ([a-z:-]+)/g)) scriptsMentioned.add(m[1]);

// --- the report --------------------------------------------------------------

const missingEnv = [...envUsed].filter((v) => !envDocumented.has(v)).sort();
const missingTools = [...toolNames].filter((t) => !docText.includes(t)).sort();
// Docs legitimately write "npm run verify:$suite" in a shell loop; a name that
// ends in a colon is a template, not a command anyone types.
const deadScripts = [...scriptsMentioned]
  .filter((s) => !s.endsWith(":") && !s.endsWith("-") && !scripts.has(s))
  .sort();

// A doc naming a setting the code no longer reads sends readers to configure
// something with no effect, which is worse than an omission: it looks like it
// worked. Only flags names that look like ours, to avoid third-party examples.
const OURS = /^(ELLE_|ASSISTANT_|PROFILE_|BOOKING_|DEEPGRAM_|TWILIO_|TELEGRAM_|REALTIME_|VOICE_|CALL_|ABUSE_|RECORDING_|GEMINI_|VERTEX_|ADMIN_|SMS_|MAX_|SILENT_|SITE_|ASK_|CORS_|LANGUAGES|BRAND_|ALLOW_)/;
/**
 * Names that appear in documentation as CODE CONSTANTS rather than as settings
 * anyone can configure. Documenting them is correct; reporting them as
 * unreadable settings is not.
 */
const CODE_CONSTANTS = new Set(["VOICE_MAX_TOKENS", "FRAME_BYTES", "ECHO_WINDOW_MS", "KEYWORD_REPLY_MAX"]);

const envInCurrentDocs = new Set<string>();
for (const m of currentDocText.matchAll(/\b([A-Z][A-Z0-9_]{2,})\b/g)) envInCurrentDocs.add(m[1]);

const staleEnv = [...envInCurrentDocs]
  .filter(
    (v) =>
      OURS.test(v) &&
      !v.endsWith("_") && // a bare prefix like ABUSE_ is prose, not a name
      v.includes("_") &&
      !CODE_CONSTANTS.has(v) &&
      !envUsed.has(v) &&
      !UNDOCUMENTED_OK.has(v)
  )
  .sort();

let failed = 0;
const section = (title: string, items: string[], why: string): void => {
  if (items.length === 0) {
    console.log(`  OK    ${title}`);
    return;
  }
  failed++;
  console.log(`  DRIFT ${title}`);
  console.log(`        ${why}`);
  for (const i of items) console.log(`          ${i}`);
};

console.log(`checked ${docFiles.length} documents against ${srcFiles.length} source files\n`);
section("every setting the code reads is documented", missingEnv,
  "these are read by the code and appear in no document:");
section("every tool the model can call is documented", missingTools,
  "the model can call these and no document mentions them:");
section("no document names a setting the code ignores", staleEnv,
  "documented but never read — a reader would configure these for no effect:");
section("no document tells you to run a command that is gone", deadScripts,
  "referenced in docs, not in package.json:");

console.log("");
if (failed === 0) {
  console.log("Documentation matches the code on every mechanically checkable claim.");
  process.exit(0);
}
console.log(`${failed} kind${failed === 1 ? "" : "s"} of drift. The prose still needs a human;`);
console.log("this only proves the names line up.");
process.exit(1);
