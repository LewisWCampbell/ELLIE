# The knowledge base (RAG) & conversation template

This is the piece that makes the bot answer as *you* — grounded in your real
resume and abilities, on both the phone and the website — with a consistent
conversation flow that adapts to each caller.

## The idea

```
        knowledge/*.md  (your resume, projects, skills — plain markdown)
                 │  chunked by heading, indexed (BM25)
                 ▼
  caller question ──▶ search_resume tool ──▶ top matching chunks ──▶ the model
                                                                       │
                            answers ONLY from what was retrieved ──────┘
```

The model never free-associates about you. When a caller asks something, it
calls the `search_resume` tool, gets the most relevant sections of your resume,
and answers from those — or says it doesn't know and offers to take a message.
The **same** retriever and persona power the phone bot (cascaded and realtime)
and the website chat, so the answers are consistent everywhere.

## Adding your content

Put markdown files in `knowledge/` — e.g. `experience.md`, `projects.md`,
`skills.md` (example files are included; replace their content). Rules of thumb:

- **Every `##` / `###` heading becomes a retrievable chunk.** Give sections
  clear, specific headings ("Cloud and infrastructure", "Realtime voice
  project") — the heading is part of what's searched.
- Keep entries factual and concrete. The bot answers *only* from this text.
- Small, focused sections retrieve better than giant walls of text (long
  sections are auto-split at ~1000 characters).

Set `PROFILE_NAME` so the persona says "You are <your name>'s AI assistant".
No rebuild or re-index step is needed for content — it loads at startup;
restart the app after editing.

Verify retrieval offline any time:

```bash
npm run verify:knowledge
```

## The conversation template

`src/persona.ts` holds one persona + template used by every channel. It's a
fixed arc that the model adapts to each caller:

1. Greet warmly, say it's your AI assistant, ask who's calling and what they want.
2. Answer questions by **first calling `search_resume`**, only from what comes back.
3. Encourage a visit to your website for the interactive display (and, on a paired
   phone call, use `show_image` / `show_choices` to walk through projects).
4. Adapt to the caller — recruiter vs. collaborator vs. curious.
5. Quietly remember who called (`save_user_fact`) and take messages.

Editing the arc or tone is a single file; it changes phone and web together.

## The website chat

`POST /api/chat { message, history }` → `{ reply, sources }` runs the same
persona + RAG for anonymous web visitors. `public/resume.html` is a minimal chat
UI on top of it (suggested questions, source labels). This is the "interactive
display of my abilities" the voice bot points callers to — and it works
standalone, without a phone call. It's also, itself, a demonstration of building
a RAG system.

## Design: why lexical (BM25) by default

The default retriever is **local BM25** — pure TypeScript, no embedding API, no
vector database, no GPU, no per-query cost, and nothing leaves the machine
(good for the privacy/HIPAA posture). For a personal corpus (tens to a few
hundred chunks) it's instant and, because resume content is keyword-rich, it
retrieves well.

Queries and chunks pass through a five-rule suffix stripper first (`studied`
and `study` become one term, as do `projects` and `project`). It is not a real
stemmer, on purpose: without it, a query scores zero for using a different
tense than the corpus, and with an aggressive one unrelated words collide and
retrieval gets quietly worse.

### Upgrading to embeddings (optional, semantic matching)

If you want paraphrase-robust semantic matching ("do they know ops?" → "on-call,
Terraform, CI/CD"), swap the retriever — the rest of the app is unchanged
because everything calls `searchKnowledge()` behind the `Retriever` shape:

1. Add an `Embedder` (OpenAI `text-embedding-3-small` is ~$0.02/1M tokens; or a
   **local** model via Ollama `nomic-embed-text` for zero external dependency).
2. At startup, embed each chunk once; store the vectors (in memory or a column
   in SQLite).
3. Implement `search(query)` as: embed the query, cosine-similarity against the
   chunk vectors, return top-k. For a few hundred chunks, brute-force cosine in
   JS is instant — no vector DB needed.
4. Point `searchKnowledge` at the new retriever.

Keeping it local (Ollama embeddings) preserves the "no third party sees my data"
property; using OpenAI embeddings is cheaper to run but sends chunk/query text to
OpenAI. Pick per your constraints — the interface makes it a contained change.

## Closing the loop: the query log and the gap report

Retrieval tells you what the corpus *has*. The more useful question is what it
is missing, and the only place that answer exists is in what people actually
ask. Every question that arrives on any channel is scored against the index as
it comes in, and the score is written to a `queries` table:

```
caller question ──▶ is this a question? ──▶ score it (BM25 + term coverage)
                            │  no                    │
                            ▼                        ▼
                        dropped                queries table
                                                     │  weekly
                                                     ▼
                                          digest to the owner:
                                          "3x asked about X, not covered"
```

Two things make it worth having:

- **It records the miss, not just the hit.** A question the corpus cannot
  answer is the signal; a question it can answer is only confirmation. The
  digest leads with the misses because each one is a paragraph that should
  exist in `knowledge/` and does not.
- **Near-duplicate questions are grouped**, so "what did he do at Northfield" and
  "tell me about his time at Northfield" show up as one topic asked twice rather
  than two singletons buried in a list.

The output is not a model update or an index rebuild. It is a writing queue:
you read the digest, you write the missing paragraph, and the next caller gets
an answer. On a corpus this size that loop is faster and more truthful than any
automated alternative, because a person decides what is true.

**A miss is usually not a retrieval failure.** On the phone the whole corpus is
already inside the prompt, so nothing was retrieved and nothing could have been
missed - the question simply had no answer written down. This is the specific
reason a vector database would not help here: better matching cannot find text
that does not exist. When the corpus grows past what fits in a prompt, that
changes, and the `Retriever` seam above is where it changes.

### Judging a miss

A question counts as answered only if both tests pass:

- **Score.** The top BM25 chunk scores at least `GAP_MIN_SCORE`.
- **Coverage.** At least half the question's content words appear somewhere in
  the corpus at all.

Either alone gives a bad report. Score alone passes "what was his sailing trip
in Norway" because "trip" happens to appear somewhere. Coverage alone flags a
well-answered question that used one unusual synonym. A digest full of false
alarms gets ignored, and then the real gap in it gets ignored too, so the bar
for calling something a gap is deliberately high.

### Privacy

The table holds the question, the score, and the channel. It holds **no phone
number and no call SID**, and there is no column to put one in, so it cannot be
joined back to a caller. Utterances that look like contact details (emails,
phone numbers, times) and self-introductions ("my name is...") are dropped
before they are written. Rows are deleted after `GAP_RETENTION_DAYS`.

What is worth knowing here is the distribution of questions across everyone,
not who asked which one.

### Settings

| Setting | Default | What it does |
|---|---|---|
| `QUERY_LOG` | `1` | Set to `0` to turn the log off entirely; nothing is written |
| `GAP_MIN_SCORE` | `1` | BM25 score below which a question counts as unanswered |
| `GAP_RETENTION_DAYS` | `90` | How long logged questions are kept |
| `GAP_DIGEST_DAYS` | `7` | Days between digests; `0` keeps the log but never sends one |

Read it any time at `/admin/gaps` (or `/admin/gaps.json?days=30`), and force a
digest with `POST /admin/send-digest`. Both need the admin token.

## Files

| File | Role |
|---|---|
| `knowledge/*.md` | Your resume/portfolio content (the corpus) |
| `src/knowledge.ts` | Chunking + BM25 retriever + `searchKnowledge()` |
| `src/gaps.ts` | Query log, gap report, weekly digest |
| `src/persona.ts` | Shared persona, conversation template, `search_resume` tool schema |
| `src/webchat.ts` | Website chat brain (RAG + persona, text in/out) |
| `public/interactive.html` | Visual resume + live chat with the assistant |
| `public/cv.html` | Traditional, print-friendly resume |
| `public/resume.html` | Minimal chat-only page (widget-style) |
| `src/ai.ts`, `src/realtime/twilioBridge.ts` | Phone bots — both call `search_resume` |

Pages render from `GET /api/resume` (sections derived from `knowledge/`) and
`GET /api/profile` (owner + assistant name), so they're a single source of truth
that grows with your content.
