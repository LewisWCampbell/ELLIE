# Elle

This is the system behind **Elle**, the AI assistant who answers my phone at
**+1 (341) 201-3991**. Call her and she answers questions about my work, shows
my projects in your browser while you talk, and books intro calls onto my real
calendar. Text the same number and she picks the
thread up by SMS. Try her at
[lewiswcampbell.com/ask.html](https://lewiswcampbell.com/ask.html).

This repository is the whole service: one Node process on Cloud Run that
answers the phone, the texts, the web chat, and my private intake channel.

```
                        ┌─────────┐  media stream (WebSocket, μ-law 8 kHz)
 Caller ──── PSTN ────▶ │ Twilio  │ ◀━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
                        └─────────┘                                      ┃
                                                              ┌──────────┸──────────┐
 Texter ──── SMS ─────▶ webhooks ────────────────────────────▶│  LetsTalk (Express) │
                                                              │                     │
 Browser ─── /ws ─────▶ pairing socket ──────────────────────▶│  bridge · sessions  │
                                                              │  booking · memory   │
 Me ──────── intake ──▶ private channel ─────────────────────▶│  knowledge (BM25)   │
                                                              └──────────┰──────────┘
                                            ┌──────────────┬─────────────┸───┬──────────────┐
                                            ▼              ▼                 ▼              ▼
                                        Deepgram      Claude (LLM)     Google Calendar   GCS archive
                                        STT / TTS     via API/Vertex   free/busy + holds  every call
```

## The interesting part

A voice agent is drawn as three boxes: speech to text, a language model, text to
speech. The hard fifth of this codebase is the layer between them, the one that
decides *when to talk*: turn-taking, barge-in (interrupt her mid-sentence and she
stops, and her memory is truncated to what you actually heard), echo suppression
so she doesn't answer her own voice, and a drift-corrected 20 ms pacer so
playback doesn't slur. That layer is documented end to end:

- **[The orchestration, explained](https://talk.lewiswcampbell.com/docs/orchestration.html)**: every timer, gate
  and cancellation token, with the measured latency of one turn.
- **[Full engineering docs](https://talk.lewiswcampbell.com/docs/)**:
  eighteen chapters, from architecture to the database schema.
- **[The blog post](https://lewiswcampbell.com/blog/building-elle.html)**: why a
  three-layer stack can hold an exchange but not a conversation.
- **[Elle at call-center scale](https://lewiswcampbell.com/elle-at-scale.html)**: what this costs at
  1K to 1M minutes a month, every rate footnoted.

The voice brain is pluggable. `RealtimeModel` is one interface with three
implementations (Deepgram cascade, OpenAI Realtime, Gemini Live on Vertex),
chosen per call from a runtime switch with an expiry, so a live call can be
moved between architectures without a deploy.

## What she can do

- **Answer as my assistant**, grounded in a local BM25 knowledge base
  (`knowledge/*.md`), so she cites real work instead of free-associating. The
  corpus itself is private and not in this repository; a fresh clone runs with
  an empty knowledge base and says so at boot.
- **Book intro calls**: free/busy across every configured calendar, a tentative
  hold, a unique-index slot lock so two concurrent callers cannot take the same
  time, and an SMS reminder the day before (consent re-checked at send time).
- **Pair with a browser** mid-call: a spoken code links a web session; she can
  run live project demos in a sandboxed iframe, show images, and put tappable
  choices on screen. The page also shows what she knows about you, editable.
- **Text**: full A2P-compliant SMS with STOP/HELP handling, booking management
  by text, and a registered campaign.
- **Report to me** on a private channel: every booking immediately, every real
  conversation when it ends, and nothing for silent wrong numbers.
- **Tell me what she could not answer**: every question is scored against the
  knowledge base as it arrives and logged without the asker, so the weekly
  digest is a list of paragraphs I should go write. The corpus grows from what
  people actually ask rather than from my guesses about it.
- **Survive deploys**: every call is archived to Cloud Storage as it happens and
  replayed on startup, because Cloud Run containers are disposable.
- **Fail politely**: at capacity callers are asked to text rather than hearing
  silence; a failed turn apologises out loud instead of going quiet.

## Running it

```bash
cp .env.example .env      # fill in what you have; the file documents every setting
npm install
npm run dev               # local server; Twilio needs a public URL (ngrok, etc.)
```

Deploying is one script that refuses to ship anything broken:

```bash
./tools/deploy.sh         # typecheck, docs drift check, 15 offline suites, deploy
```

The offline suites are the house style: no mocks of my own code, just stubbed
vendors, and each one pins behaviour that once broke or must never break.
`verify:greeting` checks *prosody* (where the pauses land in the opening line);
`verify:booking` books across a DST boundary; `verify:archive` replays a call
into a fresh database twice and proves the transcript doesn't duplicate.

```bash
npm run check:calendar    # asks the DEPLOYED service to test its own calendar
npm run check:calls       # who called: Twilio's record, then Elle's transcripts
npm run check:docs        # does the documentation still match the code
```

## Layout

```
src/
  index.ts              bootstrap, boot-time credential checks, reminder sweep
  realtime/
    twilioBridge.ts     the call: audio clock, barge-in, capacity, pairing
    deepgramCascade.ts  the conversation: turns, tools, sentence streaming, echo
    geminiLive.ts       fused speech-to-speech provider (Vertex)
    factory.ts          per-call provider selection with runtime override
  booking/              calendar client, slot lock, reminders, self-check
  sms.ts                A2P compliance gate + SMS conversation
  telegram.ts           my private intake channel
  archive.ts            durable per-call snapshots on GCS
  notify.ts             notifications to me (restraint included)
  knowledge.ts          BM25 RAG over knowledge/ (corpus private, not in the repo)
  gaps.ts               anonymous query log and the knowledge-gap digest
public/docs/            the engineering documentation site
tools/                  deploy, operator checks, doc drift detection
```

## Next steps

Everything above treats turn-taking as an engineering problem: timers, gates
and cancellation tokens standing in for a judgment people make without
thinking. The version of this that is actually good will be a model that
learned that judgment, and the seam for it already exists in this codebase.

So the next stage is deliberate: take an off-the-shelf full-duplex interaction
model, NVIDIA's PersonaPlex being the current candidate (Moshi-derived, MIT
code, openly licensed weights, persona set by role prompt and voice by audio
conditioning), and engineer tool calling on top of it. That is the part nobody
ships today. A speech-to-speech model will happily discuss a Thursday
appointment and cannot make one: there is no text bottleneck in it to hang a
function call on.

The design I intend to build is a supervisor. The duplex model owns the
conversation, listening and speaking with turn-taking in the weights instead
of in my timers. Its transcript stream feeds a small text model that owns the
tools: the calendar, the booking lock, the SMS confirmations, everything in
`booking/` that already works. Tool results are written back into the duplex
model's context as they resolve, so the caller hears an immediate, natural
voice while the booking lands a beat later. The interesting failure modes are
the point: the supervisor can lag the conversation, and a caller can be told a
slot is held a moment before the write fails. Designing those handoffs, rather
than discovering them in production, is the work.

The pieces are staged. `RealtimeModel` means a fourth brain is one file and
one enum value; the mu-law and resampling audio path is built and verified;
the [scaling study](https://lewiswcampbell.com/elle-at-scale.html) prices the
GPU hosting and names the one number that decides it, concurrent streams per
GPU, which I will benchmark before committing to hardware.

## License

All rights reserved. I've made the code public to be read, not reused; if you
want to build on it, ask me. The number is at the top, and she'll pass it on.
