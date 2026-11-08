# Realtime voice MVP — architecture, deploy, and HIPAA readiness

A small, quickly-deployable version of LetsTalk that lets callers **talk over the bot** and have it adapt, built for a **health-sector demo on synthetic data** with a **compliant-by-design** architecture.

## What this is

A phone number that, when called, connects the live audio to the **OpenAI Realtime API** (speech-to-speech). The model listens, reasons, speaks, and handles turn-taking in one place — so **barge-in is native**: the moment the caller starts talking, the bot stops, and because we truncate the model's context to what the caller actually heard, it *adapts* to the interruption instead of barrelling on.

It reuses the existing app's persona (`profile.md` → interview mode) and memory (facts saved to SQLite, recalled by phone number on the next call).

```
 Caller ──PSTN──▶ Twilio ──<Connect><Stream> (wss)──▶  /media  (this server)
                                                          │  relays audio both ways
                                                          ▼
                                              OpenAI Realtime API (wss)
                                              STT + LLM + TTS + turn-taking
                                                          │
                                     save_user_fact tool ─┴─▶ SQLite (facts, transcripts)
```

## The barge-in mechanism (the part you asked for)

1. Twilio streams caller audio (μ-law 8 kHz) to `/media`; we forward each frame to the Realtime API.
2. The model speaks; we forward its audio back to Twilio.
3. When the caller talks over the bot, the Realtime API's server-side VAD fires `input_audio_buffer.speech_started`. On that event the bridge (`src/realtime/twilioBridge.ts`):
   - sends Twilio a `clear` to **flush the bot's queued audio** (instant silence),
   - sends `conversation.item.truncate` so the model's memory of its reply is trimmed to the milliseconds actually played — **this is what makes it "retain and adapt"**, not just stop,
   - cancels the in-progress response.

Both legs speak G.711 μ-law @ 8 kHz, so the bridge is a passthrough — no transcoding.

## Files

| File | Role |
|---|---|
| `src/realtime/route.ts` | Twilio webhook → `<Connect><Stream>` TwiML |
| `src/realtime/twilioBridge.ts` | Twilio Media Stream ↔ OpenAI Realtime, per call; barge-in + fact saving |
| `src/realtime/openaiRealtime.ts` | Minimal OpenAI Realtime WebSocket client |
| `src/index.ts` | Routes WS upgrades: `/ws` (browser pairing) and `/media` (Twilio) |

Select the mode with `VOICE_MODE=realtime` (default stays `cascaded`, the original Gather/Say path).

> **API drift note:** the Realtime API has evolved since its 2024 preview. Field names in `openaiRealtime.ts` follow the widely-used preview shape (`input_audio_format`, `turn_detection`, event names). If you're on a newer GA model, check the current reference and adjust that one file.

## Deploy quickly

Twilio Media Streams needs a **public `wss://` endpoint**, so deploy somewhere with real TLS (Fly.io, Render, Railway, a VPS behind Caddy/nginx). Local dev: `ngrok http 3000` and set `PUBLIC_URL` to the https URL.

```bash
# 1. Configure
cp .env.example .env
#   set: VOICE_MODE=realtime
#        OPENAI_REALTIME_API_KEY=sk-...
#        PUBLIC_URL=https://your-public-host
#        TWILIO_AUTH_TOKEN=...            (enables webhook signature checks)
cp profile.example.md profile.md          # interview persona (optional)

# 2. Run
npm install && npm run build && npm start
#   or: docker build -t letstalk . && docker run -p 3000:3000 --env-file .env letstalk

# 3. Point your Twilio number's Voice webhook at:
#        https://your-public-host/voice-realtime   (HTTP POST)
```

Call the number and interrupt the bot mid-sentence — it should stop instantly and respond to what you said.

## HIPAA readiness (compliant-by-design; the demo runs on synthetic data)

**This demo uses synthetic/test data, so HIPAA does not bind it.** The point is to demonstrate an architecture that becomes fully compliant by flipping on the controls below — *before* any real PHI flows. HIPAA is not "self-hosted vs cloud"; it's **a signed BAA with every vendor that touches PHI, plus safeguards**. Managed vendors are fine if they sign.

### BAAs to sign before real PHI

- **OpenAI** — email `baa@openai.com`; use an API tier under the BAA with **zero data retention** on the Realtime endpoints. (Consumer/standard tiers are *not* covered.)
- **Twilio** — Programmable Voice + SIP are HIPAA-eligible under a BAA; requires the **Security or Enterprise edition**. A standard/trial account is not eligible.
- **Your hosting/storage provider** — AWS/GCP/Azure all sign BAAs for their HIPAA-eligible services; put the app and database on those.

### Safeguards this architecture needs (checklist)

- [ ] **Encryption in transit** — already true: Twilio streams and the OpenAI connection are both `wss://` (TLS). Terminate TLS at your edge; never expose `/media` over plain `ws`.
- [ ] **Encryption at rest** — transcripts and facts in SQLite are PHI when real. Use an encrypted volume / encrypted DB, or move to a managed encrypted datastore under a BAA.
- [ ] **Access control** — authenticate and least-privilege everyone who can reach the DB, logs, and admin surfaces.
- [ ] **Audit logging** — record who accessed PHI and when; keep it tamper-evident.
- [ ] **No PHI in application logs** — scrub transcripts/identifiers from stdout and error logs (they land in log aggregators that may not be covered). Today the app logs errors, not content — keep it that way.
- [ ] **Retention & deletion** — define how long transcripts/facts live and honor deletion requests (the `facts`/`transcripts` tables are keyed by phone number, which makes per-subject deletion straightforward).
- [ ] **Zero data retention on OpenAI** — configure it, so patient audio/text isn't retained vendor-side.
- [ ] **Consent & recording notice** — if you record or transcribe, disclose it at call start per state law.
- [ ] **Minimum necessary** — the interview persona already answers from a profile and saves only what's needed; keep the fact schema minimal.

### Where this lands for the pitch

You can tell the health company, truthfully: *"Patient data is handled only by BAA-covered vendors (OpenAI, Twilio, our cloud) under zero-retention terms, encrypted end to end, with audit logging and defined retention — and the demo you're seeing runs entirely on synthetic data."* If they require that **no third-party AI vendor ever sees patient audio**, that's the trigger to switch the model layer to the self-hosted full-duplex option (Moshi on your own HIPAA-eligible GPU cloud) — the bridge and app stay the same; only the endpoint the bridge dials changes.
