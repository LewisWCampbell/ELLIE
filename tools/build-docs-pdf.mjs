/**
 * Bind the documentation site into one printable book.
 *
 * The site is seventeen pages that link to each other, which is right for
 * reading on a screen and useless for handing someone a document. This walks
 * the same page order the sidebar uses, lifts each page's <main>, and renders
 * the lot as a single PDF with a title page, a table of contents, and numbered
 * parts.
 *
 *   node tools/build-docs-pdf.mjs            # PDF + the single-file HTML
 *   node tools/build-docs-pdf.mjs --html     # skip the PDF (no browser needed)
 *
 * Cross-page links are rewritten to internal anchors, so the PDF's own table of
 * contents and every "see the voice pipeline" link jump within the document
 * rather than opening a browser. The site's stylesheet is inlined and then
 * overridden for print: the sidebar goes away, colours are forced light, and
 * code blocks and tables are told not to break across a page.
 *
 * The diagrams survive because they were always inline SVG in the page rather
 * than external images, which is also why the docs site needs no assets.
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const DOCS = path.join(ROOT, "public", "docs");
const OUT_HTML = path.join(ROOT, "Elle-Documentation.html");
const OUT_PDF = path.join(ROOT, "Elle-Documentation.pdf");

/** Same order as the sidebar in public/docs/docs.js. */
const PAGES = [
  ["index.html", "Overview"],
  ["architecture.html", "Architecture"],
  ["diagrams.html", "Diagrams"],
  ["voice-pipeline.html", "The voice pipeline"],
  ["languages.html", "Bilingual mode"],
  ["channels.html", "Web, SMS and intake"],
  ["booking.html", "Booking and compliance"],
  ["data.html", "Data and knowledge"],
  ["database.html", "The database"],
  ["security.html", "Security"],
  ["compliance.html", "Compliance and BAAs"],
  ["cost.html", "Cost engineering"],
  ["scaling.html", "Architectures by volume"],
  ["operations.html", "Operations"],
  ["reference.html", "Code reference"],
  ["features.html", "Feature index"],
  ["glossary.html", "Glossary"],
];

const slug = (file) => "sec-" + file.replace(/\.html$/, "");

/** Pull the <main>…</main> body out of a docs page. */
function extractMain(html) {
  const start = html.indexOf("<main>");
  const end = html.lastIndexOf("</main>");
  if (start === -1 || end === -1) throw new Error("no <main> block");
  return html.slice(start + "<main>".length, end);
}

/**
 * Rewrite links between docs pages into anchors within this document, and
 * prefix each page's own ids so two pages using the same id (a numbered
 * citation, say) cannot collide once they share one document.
 */
function localize(body, file) {
  const prefix = slug(file) + "-";
  return (
    body
      // id="ref1" -> id="sec-index-ref1"
      .replace(/\bid="([^"]+)"/g, (_m, id) => `id="${prefix}${id}"`)
      // href="architecture.html#foo" -> href="#sec-architecture-foo"
      .replace(
        /\bhref="([a-z0-9-]+)\.html#([^"]+)"/g,
        (_m, page, frag) => `href="#${slug(page + ".html")}-${frag}"`
      )
      // href="architecture.html" -> href="#sec-architecture"
      .replace(/\bhref="([a-z0-9-]+)\.html"/g, (_m, page) => `href="#${slug(page + ".html")}"`)
      // href="#ref1" (same page) -> href="#sec-index-ref1"
      .replace(/\bhref="#([^"]+)"/g, (_m, frag) =>
        frag.startsWith("sec-") ? `href="#${frag}"` : `href="#${prefix}${frag}"`
      )
      // The per-page footer repeats seventeen times in one document.
      .replace(/<footer class="docfoot">[\s\S]*?<\/footer>/g, "")
      // The mobile nav button has nothing to toggle here.
      .replace(/<button class="menu-btn">[\s\S]*?<\/button>/g, "")
  );
}

const PRINT_CSS = `
@page {
  size: Letter;
  margin: 16mm 15mm 18mm;
}

/* Print overrides. The screen stylesheet above still supplies every colour
   token, type scale and diagram rule; this only undoes the app chrome and
   teaches the content where it may break across a page. */
:root, :root:not([data-theme="light"]) {
  --bg: #ffffff; --panel: #f6f6f4; --ink: #14161a; --soft: #55606e;
  --line: #d8dce2; --accent: #7c5cff; --accent-ink: #4b2fd6;
  --good: #1f8a5c; --warn: #b4690e; --code-bg: #1b1f27; --code-ink: #e6e9ef;
  --code-soft: #8b94a5;
}
body { background: #fff; color: var(--ink); }
.layout { display: block; }
.sidebar, .menu-btn, .copy-btn { display: none !important; }
main { max-width: none; padding: 0; margin: 0; }

.cover { text-align: center; padding: 22vh 0 0; page-break-after: always; }
.cover h1 { font-size: 2.6rem; margin: 0 0 0.4rem; letter-spacing: -0.02em; }
.cover .sub { color: var(--soft); font-size: 1.05rem; margin: 0 0 2.4rem; }
.cover .meta { color: var(--soft); font-size: 0.85rem; line-height: 1.9; }
.cover .rule { width: 4rem; height: 3px; background: var(--accent); margin: 1.6rem auto; }

.toc { page-break-after: always; }
.toc h2 { margin-top: 0; }
.toc ol { list-style: none; padding: 0; counter-reset: toc; }
.toc li { counter-increment: toc; padding: 0.45rem 0; border-bottom: 1px solid var(--line); }
.toc li a { text-decoration: none; color: var(--ink); font-weight: 600; }
.toc li a::before { content: counter(toc) ".  "; color: var(--accent-ink); }
.toc li .blurb { display: block; font-weight: 400; font-size: 0.85rem; color: var(--soft); }

section.chapter { page-break-before: always; }
section.chapter > h1 { border-bottom: 2px solid var(--accent); padding-bottom: 0.4rem; }
section.chapter > h1::before {
  content: "Part " counter(chapter);
  color: var(--accent-ink); font-weight: 600; font-size: 0.62em;
  letter-spacing: 0.08em; text-transform: uppercase; display: block;
  margin-bottom: 0.35rem;
}
body { counter-reset: chapter; }
section.chapter { counter-increment: chapter; }

/* Nothing that reads as a unit may be split down the middle. */
.codeblock, .diagram, .card, .note, .figwrap, table, .tablewrap { page-break-inside: avoid; }
h1, h2, h3 { page-break-after: avoid; }
tr { page-break-inside: avoid; }
/* Diagrams must FIT the page, not run off it.
   The screen stylesheet now scales .sdg to its column too, so print and screen
   finally agree; these rules stay because print must not inherit any future
   screen-only override, and because a 900px-wide ERD at a 186mm content width
   loses its right-hand quarter if anything goes wrong here. They need at least
   the specificity of .sdg to win, hence svg.sdg rather than a bare svg. Every
   generated diagram carries a viewBox, so setting the width and letting the
   height follow scales them correctly.

   The figure is also allowed to bleed a few millimetres into the page margins.
   Diagram text is 11px in a 660-900px coordinate space, so every point of scale
   given back is legibility on the printed page. */
.figwrap {
  overflow: visible;
  margin-left: -6mm;
  margin-right: -6mm;
  padding-left: 0.3rem;
  padding-right: 0.3rem;
}
svg, svg.sdg, .figwrap svg {
  display: block;
  width: 100%;
  max-width: 100%;
  height: auto;
}
.grid2 { display: block; }
.grid2 .card { margin-bottom: 0.8rem; }
a { color: var(--accent-ink); }

/* Long code lines wrap rather than vanish off the right edge of the paper. */
.codeblock pre { white-space: pre-wrap; word-break: break-word; font-size: 0.72rem; }
`;

function build() {
  const css = fs.readFileSync(path.join(DOCS, "docs.css"), "utf8");
  const generated = new Date().toISOString().slice(0, 10);

  const chapters = PAGES.map(([file, title]) => {
    const raw = fs.readFileSync(path.join(DOCS, file), "utf8");
    const body = localize(extractMain(raw), file);
    return { file, title, body };
  });

  // The lead paragraph of each page doubles as its table-of-contents blurb.
  const blurb = (body) => {
    const m = body.match(/<p class="lead">([\s\S]*?)<\/p>/);
    const text = (m ? m[1] : "").replace(/<[^>]+>/g, "").replace(/\s+/g, " ").trim();
    return text.length > 190 ? text.slice(0, 187).trimEnd() + "..." : text;
  };

  const toc = chapters
    .map(
      (c) =>
        `<li><a href="#${slug(c.file)}">${c.title}</a>` +
        `<span class="blurb">${blurb(c.body)}</span></li>`
    )
    .join("\n");

  const bodies = chapters
    .map((c) => `<section class="chapter" id="${slug(c.file)}">\n${c.body}\n</section>`)
    .join("\n");

  const html = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Elle · System Documentation</title>
<style>
${css}
${PRINT_CSS}
</style>
</head>
<body>
<main>

<div class="cover">
  <h1>Elle</h1>
  <p class="sub">An AI phone, SMS and web assistant, documented end to end</p>
  <div class="rule"></div>
  <p class="meta">
    Lewis W. Campbell<br>
    lewiswcampbell.com<br>
    github.com/LewisWCampbell/ELLIE<br><br>
    Generated ${generated} from the LetsTalk codebase<br>
    ${chapters.length} parts
  </p>
</div>

<div class="toc">
  <h2>Contents</h2>
  <ol>
${toc}
  </ol>
  <div class="note">Code excerpts are copied from the repository and lightly trimmed for
  width. Nothing here contains a credential: every secret lives in GCP Secret Manager and
  reaches the process only as an environment variable at runtime, and configuration is
  documented by variable <em>name</em> only.</div>
</div>

${bodies}

</main>
</body>
</html>`;

  fs.writeFileSync(OUT_HTML, html);
  console.log(`html  ${OUT_HTML}  (${(html.length / 1024).toFixed(0)} KB, ${chapters.length} parts)`);
  return html;
}

/**
 * Candidate Chromium binaries, in preference order. The headless shell is what
 * this environment ships for automation; a normal Chrome or Chromium install is
 * the fallback on a laptop. Deliberately no Playwright dependency: this tool
 * exists to print a document, and adding a browser-automation package to the
 * service's dependencies to do it would be a poor trade.
 */
function findChromium() {
  const fromEnv = process.env.CHROMIUM_BIN;
  const roots = process.env.PLAYWRIGHT_BROWSERS_PATH
    ? fs
        .readdirSync(process.env.PLAYWRIGHT_BROWSERS_PATH)
        .flatMap((d) => [
          path.join(process.env.PLAYWRIGHT_BROWSERS_PATH, d, "chrome-linux", "chrome"),
          path.join(process.env.PLAYWRIGHT_BROWSERS_PATH, d, "chrome-linux", "headless_shell"),
        ])
    : [];
  const candidates = [
    fromEnv,
    ...roots,
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/usr/bin/google-chrome",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  ].filter(Boolean);
  return candidates.find((c) => {
    try {
      fs.accessSync(c, fs.constants.X_OK);
      return true;
    } catch {
      return false;
    }
  });
}

async function toPdf() {
  const bin = findChromium();
  if (!bin) {
    console.error(
      "\nNo Chromium found, so only the HTML was written. Open it and print to PDF\n" +
        "from the browser, or set CHROMIUM_BIN to a Chrome/Chromium binary and rerun."
    );
    return;
  }

  const { execFileSync } = await import("node:child_process");
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), "docs-pdf-"));
  try {
    execFileSync(
      bin,
      [
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        `--user-data-dir=${profile}`,
        // Chromium's own header/footer would fight the @page margins and print
        // the file:// URL across the bottom of every page.
        "--no-pdf-header-footer",
        `--print-to-pdf=${OUT_PDF}`,
        "file://" + OUT_HTML,
      ],
      { stdio: ["ignore", "ignore", "pipe"] }
    );
  } finally {
    fs.rmSync(profile, { recursive: true, force: true });
  }

  const kb = fs.statSync(OUT_PDF).size / 1024;
  console.log(
    `pdf   ${OUT_PDF}  (${kb < 1024 ? kb.toFixed(0) + " KB" : (kb / 1024).toFixed(1) + " MB"})`
  );
  console.log(`      rendered with ${bin}`);
}

build();
if (!process.argv.includes("--html")) await toPdf();
