import fs from "fs";
import path from "path";
import { config } from "./config";

/**
 * A small, dependency-free RAG retriever over the owner's resume/portfolio.
 *
 * Design choices, deliberately aligned with this project's constraints:
 *  - Local & offline: a BM25 lexical index over markdown chunks. No embedding
 *    API, no vector database, no GPU, no per-query cost — and nothing leaves
 *    the box (good for the privacy/HIPAA story). For a personal corpus (a few
 *    hundred chunks) brute-force scoring is instant.
 *  - Pluggable: `Retriever` is an interface; swapping in an embedding-based
 *    retriever later is a drop-in (see docs/rag.md).
 */

export interface Chunk {
  id: number;
  source: string; // file name, e.g. "projects.md"
  heading: string; // nearest markdown heading
  text: string;
}

export interface Retrieved extends Chunk {
  score: number;
}

const STOPWORDS = new Set(
  ("a an the and or but of to in on at for with without is are was were be been being do does did " +
    "have has had i you he she it we they them me him his her hers us its my your our their " +
    "this that these those as by from " +
    "about into over under out up down so than then too very can could would should will just also " +
    "who what when where why how which whose")
    .split(" ")
);

/**
 * A deliberately small suffix stripper, not a full stemmer.
 *
 * Without it the index and the query have to agree on grammar: "what did he
 * study at Northfield" scores zero against a corpus that says "studied", and the
 * gap report then files a well-answered question as a hole in the knowledge
 * base. Five rules cover the endings that actually cause that on a corpus of
 * prose about someone's work, and stopping there keeps the transform easy to
 * predict - an over-eager stemmer collides unrelated words and makes retrieval
 * worse in a way nobody can see from the outside.
 *
 * Applied to both sides, so index and query are always stemmed alike.
 */
function stem(t: string): string {
  if (t.length <= 4) return t;
  if (t.endsWith("ied")) return t.slice(0, -3) + "y"; // studied -> study
  if (t.endsWith("ies")) return t.slice(0, -3) + "y"; // studies -> study
  if (t.endsWith("sses")) return t.slice(0, -2); // classes -> class
  if (t.endsWith("ing") && t.length > 6) return t.slice(0, -3); // building -> build
  if (t.endsWith("ed") && t.length > 5) return t.slice(0, -2); // worked -> work
  if (t.endsWith("s") && !t.endsWith("ss")) return t.slice(0, -1); // projects -> project
  return t;
}

function tokenize(text: string): string[] {
  return (text.toLowerCase().match(/[a-z0-9]+/g) ?? [])
    .filter((t) => t.length > 1 && !STOPWORDS.has(t))
    .map(stem);
}

/** The corpus tokenizer, shared so callers group words the same way it does. */
export function terms(text: string): string[] {
  return tokenize(text);
}

/** Split a markdown document into heading-scoped chunks, capped in size. */
function chunkMarkdown(source: string, content: string): Omit<Chunk, "id">[] {
  const lines = content.replace(/\r\n/g, "\n").split("\n");
  const chunks: Omit<Chunk, "id">[] = [];
  let heading = source.replace(/\.md$/, "");
  let buf: string[] = [];

  const flush = () => {
    const text = buf.join("\n").trim();
    if (!text) return;
    // Keep chunks readable: split overly long sections on blank lines.
    if (text.length <= 1000) {
      chunks.push({ source, heading, text });
    } else {
      let acc = "";
      for (const para of text.split(/\n\s*\n/)) {
        if ((acc + "\n\n" + para).length > 1000 && acc) {
          chunks.push({ source, heading, text: acc.trim() });
          acc = para;
        } else {
          acc = acc ? `${acc}\n\n${para}` : para;
        }
      }
      if (acc.trim()) chunks.push({ source, heading, text: acc.trim() });
    }
    buf = [];
  };

  for (const line of lines) {
    const m = /^(#{1,6})\s+(.*)$/.exec(line);
    if (m) {
      flush();
      heading = m[2].trim();
    } else {
      buf.push(line);
    }
  }
  flush();
  return chunks;
}

/** BM25 lexical retriever. */
export class LexicalRetriever {
  private chunks: Chunk[] = [];
  private docTokens: string[][] = [];
  private df = new Map<string, number>();
  private avgdl = 0;
  private readonly k1 = 1.5;
  private readonly b = 0.75;

  constructor(chunks: Chunk[]) {
    this.chunks = chunks;
    this.docTokens = chunks.map((c) => tokenize(`${c.heading} ${c.text}`));
    for (const toks of this.docTokens) {
      for (const t of new Set(toks)) this.df.set(t, (this.df.get(t) ?? 0) + 1);
    }
    const totalLen = this.docTokens.reduce((s, t) => s + t.length, 0);
    this.avgdl = this.docTokens.length ? totalLen / this.docTokens.length : 0;
  }

  private idf(term: string): number {
    const n = this.docTokens.length;
    const df = this.df.get(term) ?? 0;
    return Math.log(1 + (n - df + 0.5) / (df + 0.5));
  }

  search(query: string, k = 4): Retrieved[] {
    const qTerms = tokenize(query);
    if (!qTerms.length || !this.chunks.length) return [];
    const scored = this.docTokens.map((toks, i) => {
      const dl = toks.length;
      const tf = new Map<string, number>();
      for (const t of toks) tf.set(t, (tf.get(t) ?? 0) + 1);
      let score = 0;
      for (const term of qTerms) {
        const f = tf.get(term);
        if (!f) continue;
        const denom = f + this.k1 * (1 - this.b + (this.b * dl) / (this.avgdl || 1));
        score += this.idf(term) * ((f * (this.k1 + 1)) / denom);
      }
      return { ...this.chunks[i], score };
    });
    return scored
      .filter((c) => c.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, k);
  }

  get size(): number {
    return this.chunks.length;
  }

  /**
   * Does this (already tokenized) term appear anywhere in the corpus?
   *
   * Presence is a different question from score, and the gap report needs
   * both. BM25 will happily return a chunk that matched one common word, so
   * score alone reads "his sailing trip in Norway" as answered because
   * "project" turns up somewhere nearby. Absence of the distinctive terms is
   * what a knowledge gap actually is.
   */
  has(term: string): boolean {
    return this.df.has(term);
  }

  /** A short, human-readable list of the topics the corpus covers. */
  overview(): string {
    const bySource = new Map<string, Set<string>>();
    for (const c of this.chunks) {
      if (!bySource.has(c.source)) bySource.set(c.source, new Set());
      bySource.get(c.source)!.add(c.heading);
    }
    const parts: string[] = [];
    for (const [source, headings] of bySource) {
      parts.push(`${source.replace(/\.md$/, "")}: ${[...headings].slice(0, 12).join(", ")}`);
    }
    return parts.join(" | ");
  }
}

function loadChunks(dir: string): Chunk[] {
  const abs = path.resolve(dir);
  let files: string[] = [];
  try {
    files = fs.readdirSync(abs).filter((f) => f.endsWith(".md"));
  } catch {
    return [];
  }
  const chunks: Chunk[] = [];
  let id = 0;
  for (const file of files.sort()) {
    const content = fs.readFileSync(path.join(abs, file), "utf8");
    for (const c of chunkMarkdown(file, content)) chunks.push({ id: id++, ...c });
  }
  return chunks;
}

// Build once at startup from the configured knowledge directory.
let corpus = loadChunks(config.knowledgeDir);
let retriever = new LexicalRetriever(corpus);
console.log(`Knowledge base: ${retriever.size} chunks from ${config.knowledgeDir}/`);

export interface ResumeSection {
  source: string;
  title: string;
  items: Array<{ heading: string; text: string }>;
}

function titleCase(name: string): string {
  return name
    .replace(/\.md$/, "")
    .replace(/[-_]/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

/** The knowledge base as ordered, display-ready sections — for the resume pages. */
export function resumeSections(): ResumeSection[] {
  const bySource = new Map<string, ResumeSection>();
  for (const c of corpus) {
    // Internal-only file: visual and live-page URLs for the paired-browser show_work tool,
    // not a resume section to render.
    if (c.source === "visuals.md") continue;
    if (!bySource.has(c.source)) {
      bySource.set(c.source, { source: c.source, title: titleCase(c.source), items: [] });
    }
    // Skip an empty top-level intro chunk whose heading == the section title.
    if (c.text.trim()) bySource.get(c.source)!.items.push({ heading: c.heading, text: c.text });
  }
  return [...bySource.values()];
}

/** Rebuild the index (used by tests, or if you hot-reload content). */
export function buildRetrieverFrom(dir: string): LexicalRetriever {
  return new LexicalRetriever(loadChunks(dir));
}

export function useRetriever(r: LexicalRetriever): void {
  retriever = r;
}

/** Retrieve the top-k resume/portfolio chunks relevant to a query. */
export function searchKnowledge(query: string, k = 4): Retrieved[] {
  return retriever.search(query, k);
}

/** Does the corpus use this term at all? Terms must come from `terms()`. */
export function corpusHas(term: string): boolean {
  return retriever.has(term);
}

/** A short description of what the knowledge base covers, for the system prompt. */
export function knowledgeOverview(): string {
  return retriever.overview();
}

/**
 * The entire knowledge base as one formatted block, for embedding directly in
 * the system prompt. Only sensible for a small corpus (a resume/portfolio) —
 * it lets a latency-sensitive channel (the phone) skip the per-turn retrieval
 * tool round-trip and answer in a single model call. Cache it via cache_control.
 */
export function knowledgeFullText(): string {
  return corpus
    .map((c) => `[${c.source.replace(/\.md$/, "")} › ${c.heading}]\n${c.text}`)
    .join("\n\n");
}

/** Format retrieved chunks for a tool result the model reads. */
export function formatRetrieved(results: Retrieved[]): string {
  if (!results.length) {
    return "No matching information found in the resume/portfolio. Tell the caller you don't have that detail and offer to take a message.";
  }
  return results.map((r) => `[${r.source.replace(/\.md$/, "")} › ${r.heading}]\n${r.text}`).join("\n\n");
}
