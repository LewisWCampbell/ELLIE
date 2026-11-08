/**
 * The knowledge-gap report.
 *
 * The system was throwing away its most valuable output. Every call produced a
 * list of things a real person wanted to know about Lewis, and once the call
 * ended that list existed only inside a transcript nobody was going to reread.
 * The corpus therefore only ever grew by guessing: he would decide what was
 * worth writing down, and find out whether he was right never.
 *
 * This closes that loop. Every question is scored against the knowledge base as
 * it arrives, the score is recorded without the asker, and questions the corpus
 * could not answer are surfaced in a digest. The output is not a model update
 * or an index rebuild, it is a writing queue: "four people asked about the
 * Northfield research and there are two sentences about it" is an instruction to
 * go write a paragraph. The system improves because a human edits markdown,
 * which at this corpus size is both cheaper and more truthful than anything
 * automated.
 *
 * Two deliberate non-goals:
 *
 *  - No vector database. Retrieval is not the bottleneck being measured here.
 *    On the phone path the entire corpus is inlined into the cached prompt and
 *    BM25 is not consulted at all, so a miss recorded here is almost never
 *    "the retriever failed to find it" and almost always "it isn't written
 *    down anywhere". Embeddings would improve the first number and leave the
 *    second one exactly where it is.
 *
 *  - No identity. The log carries no phone number and no call SID, and it is
 *    not joinable back to a caller. What is worth knowing is the distribution
 *    of questions across everyone, not who asked which one.
 */
import { config } from "./config";
import * as db from "./db";
import { searchKnowledge, corpusHas, terms } from "./knowledge";
import { notifyOwner } from "./notify";

export type Channel = "phone" | "web" | "sms";

/** Questions are stored for the report, not reread in full; keep rows small. */
const MAX_QUESTION_CHARS = 300;

/**
 * Words that open a question, including the ones speech-to-text hands over
 * with no question mark at all. Punctuation is the exception on the phone
 * rather than the rule, so it cannot be the test.
 */
const INTERROGATIVE =
  /^(who|what|whats|when|where|why|how|hows|which|whose|does|do|did|is|are|was|were|can|could|would|will|should|has|have|had|any|anything|tell|explain|describe|walk|talk|say|know)\b/i;

/**
 * Self-disclosure, not enquiry. "My name is Lewis Campbell" clears every
 * length test and scores zero against the corpus, so without this every caller
 * introducing themselves would be filed as a knowledge gap named after them,
 * which is both useless and the one thing this table is not supposed to hold.
 *
 * The opener alone cannot decide it: "I'm Sam" and "I'm curious about the
 * photogrammetry work" start identically. Length after the opener separates
 * them, because an introduction is short by nature - a name and maybe a
 * company - while anything with three content words after "I'm" has moved on
 * to a topic.
 */
const SELF_INTRO = /^(my name is|my name's|this is|i'm|i am|im|it's|its)\b\s*/i;

function isSelfIntro(text: string): boolean {
  const m = SELF_INTRO.exec(text);
  if (!m) return false;
  return contentTerms(text.slice(m[0].length)).length <= 2;
}

/**
 * Booking chatter: email addresses, phone numbers, times of day. These are
 * how a caller hands over their contact details, which is precisely what this
 * table must not accumulate, and none of them is a question about Lewis. The
 * digit pattern is deliberately loose enough to catch a seven-digit number
 * ("555-0142") as well as a full one.
 */
const CONTACT_DETAIL = /@|\d{3}[-.\s]?\d{4}|\b\d{1,2}(:\d{2})?\s?(a\.?m|p\.?m)\b/i;

/**
 * Conversational filler on top of the corpus tokenizer's own stopwords. These
 * are the words that survive "is this a question" but say nothing about what
 * was asked, and they would otherwise pull unrelated questions into the same
 * cluster. Listed in stemmed form, since that is what `terms` returns.
 */
const FILLER = new Set(
  ("please thank okay yeah yes hey hello sure right well like really actually " +
    "kind sort mean guess wondering curious know tell say maybe thing stuff")
    .split(" ")
);

/** The corpus tokenizer plus conversational filler, so both sides agree. */
function contentTerms(text: string): string[] {
  return terms(text).filter((t) => t.length > 2 && !FILLER.has(t));
}

/**
 * Is this utterance worth logging?
 *
 * A transcript is mostly not questions. It is "yeah", "mm-hm", "sounds good",
 * a name, an email address being spelled out. Logging all of it would bury the
 * signal and store far more of the conversation than this table has any
 * business holding, so the filter is intentionally strict and the cost of it
 * being wrong is asymmetric: a dropped question is one missing row in a report
 * that has plenty of others, a kept non-question is noise in every digest.
 */
export function isQuestion(text: string): boolean {
  const t = text.trim().replace(/\s+/g, " ");
  if (t.length < 8 || t.length > 500) return false;
  if (isSelfIntro(t)) return false;
  if (CONTACT_DETAIL.test(t)) return false;

  // One content word is enough when the sentence is plainly a question: "what
  // did he do at Northfield?" is five function words and a topic, and it is the
  // most useful row this table can hold. Without the question shape, demand
  // enough substance that a topic can still be read off it, which is what
  // keeps "sounds good to me" and a spelled-out name out.
  const found = contentTerms(t);
  if (t.endsWith("?") || INTERROGATIVE.test(t)) return found.length >= 1;
  return found.length >= 3;
}

/**
 * The share of a question's content terms the corpus uses at all.
 *
 * Filler is excluded before counting, which matters more than it sounds: "tell
 * me about his time at Northfield" is a well-answered question, and counting
 * "tell" and "time" against it drags a perfectly good hit below the bar and
 * files it as a gap. A question with no content terms left scores 1, so a bare
 * "what about that?" is never blamed on the corpus.
 */
function coverageOf(qterms: string[]): number {
  const uniq = new Set(qterms);
  if (!uniq.size) return 1;
  let present = 0;
  for (const t of uniq) if (corpusHas(t)) present++;
  return present / uniq.size;
}

export interface Scored {
  answered: boolean;
  topScore: number;
  coverage: number;
  topSource: string | null;
  topHeading: string | null;
}

/**
 * Score a question against the corpus without answering it.
 *
 * Both tests have to pass. Score alone lets a single common word through;
 * coverage alone would flag a well-answered question that happens to use one
 * unusual synonym. Together they mean "the corpus contains this vocabulary and
 * some chunk is actually about it".
 */
export function scoreQuestion(question: string): Scored {
  const found = searchKnowledge(question, 1);
  const top = found[0];
  const topScore = top?.score ?? 0;
  const coverage = coverageOf(contentTerms(question));
  return {
    answered: topScore >= config.gaps.minScore && coverage >= 0.5,
    topScore,
    coverage,
    topSource: top?.source ?? null,
    topHeading: top?.heading ?? null,
  };
}

/**
 * The hook every channel calls with whatever the person just said.
 *
 * Never throws. This runs on the live audio path, between a caller finishing a
 * sentence and the model starting to answer it, so it must be incapable of
 * costing anyone a turn: the work is a few hundred string comparisons against
 * an in-memory index and one local INSERT, with no network of any kind.
 */
export function logQuestion(channel: Channel, text: string): void {
  if (!config.gaps.enabled) return;
  try {
    if (!isQuestion(text)) return;
    const question = text.trim().replace(/\s+/g, " ").slice(0, MAX_QUESTION_CHARS);
    const s = scoreQuestion(question);
    db.logQuery({
      channel,
      question,
      topScore: s.topScore,
      topSource: s.topSource,
      topHeading: s.topHeading,
      answered: s.answered,
    });
  } catch (e) {
    console.error("[gaps] failed to log question:", e);
  }
}

/* -------------------------------------------------------------------------
 * Reporting
 * ---------------------------------------------------------------------- */

export interface Cluster {
  /** The clearest question in the group, shown as the group's label. */
  example: string;
  count: number;
  answered: number;
  /** Where the corpus answered from, when it did. */
  source: string | null;
}

export interface GapReport {
  days: number;
  total: number;
  answered: number;
  /** Everything asked, most-asked first. */
  topics: Cluster[];
  /** The subset the corpus could not answer, most-asked first. */
  gaps: Cluster[];
  /** Sections the corpus answered from, most-cited first. */
  sources: Array<{ source: string; heading: string; count: number }>;
  byChannel: Record<string, number>;
}

/**
 * Group questions about the same thing.
 *
 * "What did he do at Northfield?" and "tell me about his time at Northfield" are one
 * topic, not two, and a report that lists them separately buries the fact that
 * it is the most-asked thing on the page - which is the entire point of the
 * report.
 *
 * Pairwise similarity is the obvious approach and it does not work here.
 * Spoken questions about one subject share very little beyond the subject
 * itself: those two sentences have exactly one content word in common out of
 * six, so any overlap threshold loose enough to merge them merges everything
 * else too. So group on the subject directly - each question is filed under
 * its most-repeated content word across the whole window, which is a decent
 * definition of what it is about, and ties go to the longer word because
 * "insurance" names a topic and "any" does not.
 *
 * A question sharing no word with any other lands in a group of one, which is
 * correct: it was asked once.
 */
function cluster(rows: db.QueryRow[]): Cluster[] {
  const df = new Map<string, number>();
  const termsFor = new Map<number, string[]>();
  for (const row of rows) {
    const t = contentTerms(row.question);
    termsFor.set(row.id, t);
    for (const term of new Set(t)) df.set(term, (df.get(term) ?? 0) + 1);
  }

  const groups = new Map<string, db.QueryRow[]>();
  for (const row of rows) {
    const t = termsFor.get(row.id) ?? [];
    if (!t.length) continue;
    let key = t[0];
    for (const term of t) {
      const better =
        (df.get(term) ?? 0) > (df.get(key) ?? 0) ||
        ((df.get(term) ?? 0) === (df.get(key) ?? 0) && term.length > key.length);
      if (better) key = term;
    }
    const bucket = groups.get(key) ?? [];
    bucket.push(row);
    groups.set(key, bucket);
  }

  return [...groups.values()]
    .map((group) => {
      // Label with the shortest question in the group: the one that got to the
      // point is the one that reads best in a digest.
      const example = [...group].sort((a, b) => a.question.length - b.question.length)[0].question;
      const answered = group.filter((r) => r.answered).length;
      const source = group.find((r) => r.answered && r.top_source)?.top_source ?? null;
      return { example, count: group.length, answered, source };
    })
    .sort((a, b) => b.count - a.count || a.example.localeCompare(b.example));
}

export function gapReport(days = config.gaps.digestDays || 7, limit = 500): GapReport {
  const since = new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString();
  const rows = db.getQueriesSince(since, limit);

  const byChannel: Record<string, number> = {};
  const sourceCounts = new Map<string, { source: string; heading: string; count: number }>();
  for (const r of rows) {
    byChannel[r.channel] = (byChannel[r.channel] ?? 0) + 1;
    if (r.answered && r.top_source) {
      const key = `${r.top_source}#${r.top_heading ?? ""}`;
      const entry = sourceCounts.get(key) ?? {
        source: r.top_source,
        heading: r.top_heading ?? "",
        count: 0,
      };
      entry.count++;
      sourceCounts.set(key, entry);
    }
  }

  return {
    days,
    total: rows.length,
    answered: rows.filter((r) => r.answered).length,
    topics: cluster(rows).slice(0, 10),
    gaps: cluster(rows.filter((r) => !r.answered)).slice(0, 10),
    sources: [...sourceCounts.values()].sort((a, b) => b.count - a.count).slice(0, 8),
    byChannel,
  };
}

/** The digest as the owner reads it: the gaps first, because they are the ask. */
export function formatGapReport(r: GapReport): string {
  const lines: string[] = [];
  const channels = Object.entries(r.byChannel)
    .map(([c, n]) => `${n} ${c}`)
    .join(", ");
  const rate = r.total ? Math.round((r.answered / r.total) * 100) : 0;

  lines.push(`What people asked Elle, last ${r.days} days`);
  lines.push("");
  lines.push(`${r.total} questions${channels ? ` (${channels})` : ""}, ${rate}% answerable from the corpus`);

  if (r.gaps.length) {
    lines.push("");
    lines.push("Could not answer:");
    for (const g of r.gaps) {
      lines.push(`  ${g.count > 1 ? `${g.count}x ` : ""}${g.example}`);
    }
    lines.push("");
    lines.push("Each of those is a paragraph missing from knowledge/.");
  }

  if (r.topics.length) {
    lines.push("");
    lines.push("Most asked:");
    for (const t of r.topics.slice(0, 5)) {
      const where = t.source ? ` [${t.source.replace(/\.md$/, "")}]` : " [not covered]";
      lines.push(`  ${t.count}x ${t.example}${where}`);
    }
  }

  if (r.sources.length) {
    lines.push("");
    lines.push("Answered from:");
    for (const s of r.sources) {
      lines.push(`  ${s.count}x ${s.source.replace(/\.md$/, "")} / ${s.heading}`);
    }
  }

  return lines.join("\n");
}

/** Drop log rows past the retention window. */
export function pruneQueryLog(): number {
  const before = new Date(
    Date.now() - config.gaps.retentionDays * 24 * 60 * 60 * 1000
  ).toISOString();
  return db.pruneQueries(before);
}

const DIGEST_KEY = "gap_digest_at";

/**
 * Send the digest if one is due, and prune the log while we are here.
 *
 * Called from the hourly sweep rather than a scheduler, for the same reason
 * the reminder sweep is: a feature that needs someone to remember to create a
 * Cloud Scheduler job is a feature that silently does not run. The last-sent
 * timestamp lives in the settings table, so a restart does not resend and two
 * instances do not both send.
 */
export async function maybeSendDigest(now = new Date()): Promise<"sent" | "not-due" | "empty" | "off"> {
  if (!config.gaps.enabled || config.gaps.digestDays <= 0) return "off";

  const last = db.getSetting(DIGEST_KEY);
  const dueAfter = config.gaps.digestDays * 24 * 60 * 60 * 1000;
  if (last && now.getTime() - new Date(last).getTime() < dueAfter) return "not-due";

  const report = gapReport(config.gaps.digestDays);
  // Claim the window before sending. A digest lost to a failed send is worth
  // less than a digest sent twice from two instances.
  db.setSetting(DIGEST_KEY, now.toISOString());
  pruneQueryLog();

  if (!report.total) return "empty";
  await notifyOwner(formatGapReport(report));
  return "sent";
}
