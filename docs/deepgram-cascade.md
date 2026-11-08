# Cheaper barge-in: the Deepgram cascade

The barge-in phone bot can run two ways. Both let the caller talk over Elle and
have her stop and adapt; they differ in cost and vendors.

| `REALTIME_PROVIDER` | How it works | ~cost / 3-min call | Vendors |
|---|---|---|---|
| `openai` | One speech-to-speech model (audio in → audio out) | ~$0.40–0.50 | OpenAI |
| `deepgram` | **Deepgram STT → text LLM → Deepgram Aura TTS** | **~$0.06–0.09** | Deepgram + your LLM |

The cascade is cheaper because speech-to-speech models bill **audio tokens**, and
output audio is the dominant cost. Splitting the pipeline means the LLM only ever
sees/emits text (cheap), and the audio legs are a low-cost STT/TTS vendor.

The reasoning still runs on **your `LLM_PROVIDER`** — so with `LLM_PROVIDER=vertex`
the thinking stays Claude-on-Vertex (inside your GCP BAA); only Deepgram is added
for ears and voice. Deepgram signs a HIPAA BAA.

## How it's wired

It's a drop-in `RealtimeModel` (see `src/realtime/types.ts`), so the Twilio
bridge, barge-in flush/truncate, tool calls (`search_resume`, `save_user_fact`,
`show_image`, `show_choices`), and browser pairing are all unchanged.

```
Twilio μ-law 8k ─▶ Deepgram STT ─▶ utterance ─▶ text LLM (+ tools) ─▶ reply text
                        │                                                  │
                   SpeechStarted ──▶ barge-in                     Deepgram Aura TTS
                        │                                                  │
                   cancelResponse()  ◀── bridge ──▶  μ-law 8k frames ◀─────┘
```

- **No transcoding.** Both Deepgram legs are configured for G.711 μ-law @ 8 kHz,
  the exact format Twilio Media Streams use.
- **Streaming TTS.** Aura is driven over its WebSocket, so audio starts flowing
  as the first words are synthesized rather than after the whole reply — the
  caller hears Elle begin almost immediately. Chunks are re-framed into exact
  20 ms μ-law frames on the way to Twilio.
- **Barge-in.** Deepgram's VAD emits `SpeechStarted`; the bridge flushes Twilio
  playback and calls `cancelResponse()`, which invalidates the in-flight turn so
  the LLM loop and the TTS frame pump stop immediately.
- **Short replies.** Voice turns are capped (`VOICE_MAX_TOKENS`) so Elle stays
  crisp — which is also the biggest cost lever, since output audio dominates.

Files: `src/realtime/deepgram/stt.ts`, `deepgram/tts.ts`,
`deepgramCascade.ts`, `factory.ts`. Offline test (no network):
`npm run verify:cascade`.

## Turning it on

```bash
VOICE_MODE=realtime
REALTIME_PROVIDER=deepgram
DEEPGRAM_API_KEY=...              # one key: STT + TTS
DEEPGRAM_STT_MODEL=nova-2-phonecall   # tuned for 8 kHz phone audio
DEEPGRAM_TTS_MODEL=aura-2-thalia-en   # bright American female; any Aura voice
# reasoning provider (unchanged) — e.g. Claude on Vertex:
LLM_PROVIDER=vertex
VERTEX_PROJECT_ID=...
```
Point the Twilio number's Voice webhook at `/voice-realtime` as usual.

On Cloud Run, add the Deepgram key as a secret and set the env vars:
```bash
printf 'DG_KEY' | gcloud secrets create deepgram-key --data-file=-
gcloud secrets add-iam-policy-binding deepgram-key \
  --member="serviceAccount:${PROJNUM}-compute@developer.gserviceaccount.com" \
  --role=roles/secretmanager.secretAccessor
gcloud run services update letstalk-elle --region us-central1 \
  --update-env-vars VOICE_MODE=realtime,REALTIME_PROVIDER=deepgram,DEEPGRAM_STT_MODEL=nova-2-phonecall,DEEPGRAM_TTS_MODEL=aura-2-thalia-en \
  --update-secrets DEEPGRAM_API_KEY=deepgram-key:latest
```

## Tuning

- **Latency vs. eagerness:** `DEEPGRAM_ENDPOINTING_MS` (default **1000** ms) sets
  how much trailing silence Elle waits out before deciding the caller is done.
  Higher = more patient (won't cut off a mid-thought pause) but slower to reply;
  lower = snappier but more likely to interrupt. The `utterance_end_ms` fallback
  in `stt.ts` tracks it automatically (clamped to Deepgram's 1000 ms minimum).
- **Cost:** shorter replies are the biggest lever. `VOICE_MAX_TOKENS` in
  `deepgramCascade.ts` caps reply length.
- **Voice:** any Aura voice via `DEEPGRAM_TTS_MODEL`.

## Cost note

At demo volumes the per-call difference is a few dollars a month total — so this
matters most when call volume grows. If you want the absolute floor later, swap
the TTS for Cartesia (edit `deepgram/tts.ts` behind the same `Tts` interface).

## Bilingual mode (English + Spanish)

`LANGUAGES=en,es` (the default) makes the whole cascade bilingual; set
`LANGUAGES=en` to run the original English-only stack. What changes:

- **STT** switches to the code-switching model (`DEEPGRAM_STT_MULTI_MODEL`,
  default `nova-3`, with `language=multi`). One stream transcribes both
  languages, even mid-sentence, and tags every word with the language it was
  spoken in. `stt.ts` reports each utterance's dominant language to the cascade.
- **TTS** holds one warm socket per voice: the English voice
  (`DEEPGRAM_TTS_MODEL`) plus a Spanish voice (`DEEPGRAM_TTS_MODEL_ES`, default
  `aura-2-celeste-es`). The Spanish socket connects lazily the first time it's
  needed, so English-only calls pay nothing.
- **Voice selection** per reply chunk: the caller's detected language is the
  default, and a lightweight text scan (`ES_HINT` in `deepgramCascade.ts`)
  overrides it when the reply itself is clearly the other language — e.g.
  answering "do you speak Spanish?" in Spanish.
- **The LLM** needs nothing: Claude is natively multilingual, and the persona
  gains a mirror-the-caller rule (reply entirely in the caller's language,
  switch when they switch, never mix within a reply).
- **The greeting** ends with one Spanish sentence — "Y si prefieres hablar en
  español, solo dímelo." — spoken by the Spanish voice, so Spanish-only callers
  hear in their own language that they can switch. It's part of the
  uninterruptible greeting, and the greeting note tells the model the offer was
  already made.
- **SMS** mirrors too: texts in Spanish get answered in Spanish.

Compliance note: the verbal SMS-consent flow must cover every element in the
caller's language; the persona says so. If a regulated deployment needs
word-for-word approved Spanish scripts, store them pre-translated — don't let
the model translate on the fly.
