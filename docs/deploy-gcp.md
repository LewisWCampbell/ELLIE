# Deploy LetsTalk on Google Cloud Run

Runs Elle's brain + API (and the phone bot) at **talk.lewiswcampbell.com** on
Cloud Run, using the repo's Dockerfile. The reasoning is **Claude on Vertex AI**,
so it stays inside your GCP BAA with no model API key to manage. Your main site
(`lewiswcampbell.com`) is untouched — this is a separate service on its own
subdomain that the "Talk to Elle" page and the phone number call.

```
  lewiswcampbell.com  (static site)
      └─ ask.html ── fetch ──▶  https://talk.lewiswcampbell.com/api/chat   (Cloud Run)
  Twilio number ───────────▶  https://talk.lewiswcampbell.com/voice-realtime
```

Prereqs: a Google Cloud account with **billing enabled**, and the `gcloud` CLI
(`gcloud init`, then `gcloud auth login`).

---

## Phase 1 — Web chat live (Claude on Vertex)

### 1. Pick a project, confirm billing
```bash
gcloud config set project YOUR_PROJECT_ID
gcloud billing projects describe YOUR_PROJECT_ID     # billingEnabled: true
```

### 2. Enable services (Vertex, not Secret Manager — auth is the service account)
```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com aiplatform.googleapis.com
```

### 3. Enable the Claude model in Vertex Model Garden (one console click)
Console → **Vertex AI → Model Garden** → search **Claude** → open the model
(e.g. *Claude Sonnet 4.5*, or *Haiku 4.5* for cheaper) → **Enable**. Note its
exact **model id** (e.g. `claude-sonnet-4-5@20250929`) and a **region** it's
served in (usually `us-east5`). This can't be done from the CLI.

### 4. Let Cloud Run's service account call Vertex
```bash
PROJNUM=$(gcloud projects describe YOUR_PROJECT_ID --format='value(projectNumber)')
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:${PROJNUM}-compute@developer.gserviceaccount.com" \
  --role=roles/aiplatform.user
```

### 5. Deploy from source (run from the repo root — Cloud Build reads the Dockerfile)
Set `VERTEX_MODEL` / `VERTEX_REGION` to what step 3 gave you:
```bash
gcloud run deploy letstalk-elle \
  --source . \
  --region us-central1 \
  --port 3000 \
  --min-instances 1 \                 # stay warm — no cold start on the demo
  --timeout 3600 \                    # long-lived WebSockets for Phase 2 phone audio
  --allow-unauthenticated \
  --set-env-vars LLM_PROVIDER=vertex,VERTEX_PROJECT_ID=YOUR_PROJECT_ID,VERTEX_REGION=us-east5,VERTEX_MODEL=claude-sonnet-4-5@20250929,PROFILE_NAME="Your Name",ASSISTANT_NAME=Elle,PUBLIC_URL=https://talk.lewiswcampbell.com,CORS_ORIGIN="https://lewiswcampbell.com,https://www.lewiswcampbell.com",SQLITE_PATH=/tmp/letstalk.db
```
Notes:
- **No API key** — Vertex auth is the service account from step 4 (ADC).
- `SQLITE_PATH=/tmp/letstalk.db` — Cloud Run's filesystem is ephemeral and `/tmp`
  is always writable. Caller memory resets on redeploy/scale; fine for the demo.
  For durable memory later, use a GCE VM or swap SQLite → Firestore/Cloud SQL.
- Cloud Run in `us-central1` while Vertex is `us-east5` is fine (cross-region call).
- If prompted to create an Artifact Registry repo, answer **Y**.

Test the `*.run.app` URL it prints:
```bash
curl -s RUN_URL/healthz          # {"ok":true,...}
curl -s RUN_URL/api/chat -H 'Content-Type: application/json' \
  -d '{"message":"What are Lewis'\''s ML projects?","history":[]}'
```
A reply naming real projects with `sources` proves the brain + Vertex + resume
grounding all work — before DNS. (A 404 here almost always means `VERTEX_MODEL`
doesn't match Model Garden for your region.)

### 6. Map the subdomain
```bash
gcloud beta run domain-mappings create \
  --service letstalk-elle --region us-central1 \
  --domain talk.lewiswcampbell.com
```
Add the DNS record(s) it prints at your DNS provider (name = `talk`). TLS issues
automatically; once green, `https://talk.lewiswcampbell.com/healthz` works.

> If your region doesn't support domain mappings, put a global external
> Application Load Balancer with a serverless NEG in front and point `talk` at it.

### 7. The website page is already wired
`ask.html` calls `https://talk.lewiswcampbell.com/api/chat`, and `CORS_ORIGIN`
lists your site origins — so the moment step 6 goes green, the live page works.

**Phase 1 done.**

---

## Phase 2 — The phone number (barge-in, Deepgram cascade)

The cheap barge-in stack: **Twilio + Deepgram (STT + Aura TTS) + Claude on
Vertex** (see `docs/deepgram-cascade.md`).

### 8. Add the Deepgram key as a secret
```bash
gcloud services enable secretmanager.googleapis.com
printf 'DG_KEY' | gcloud secrets create deepgram-key --data-file=-
gcloud secrets add-iam-policy-binding deepgram-key \
  --member="serviceAccount:${PROJNUM}-compute@developer.gserviceaccount.com" \
  --role=roles/secretmanager.secretAccessor
```

### 9. Turn on realtime + the cascade
```bash
gcloud run services update letstalk-elle --region us-central1 \
  --update-env-vars VOICE_MODE=realtime,REALTIME_PROVIDER=deepgram,DEEPGRAM_STT_MODEL=nova-2-phonecall,DEEPGRAM_TTS_MODEL=aura-2-thalia-en,DEEPGRAM_ENDPOINTING_MS=1000 \
  --update-secrets DEEPGRAM_API_KEY=deepgram-key:latest
```

### 10. Point the Twilio number at the app
Buy a number, then in the Twilio console → your number → **Voice → A call comes
in → Webhook (HTTP POST)** → `https://talk.lewiswcampbell.com/voice-realtime`.
Also store the Twilio auth token so the app verifies request signatures:
```bash
printf 'YOUR_TWILIO_AUTH_TOKEN' | gcloud secrets create twilio-token --data-file=-
gcloud secrets add-iam-policy-binding twilio-token \
  --member="serviceAccount:${PROJNUM}-compute@developer.gserviceaccount.com" \
  --role=roles/secretmanager.secretAccessor
gcloud run services update letstalk-elle --region us-central1 \
  --update-secrets TWILIO_AUTH_TOKEN=twilio-token:latest
```

### 11. Call it
Dial the number and talk to Elle — interrupt her mid-sentence and she adapts.

---

## Repeat-caller cost protection

On by default: a number that keeps calling within `ABUSE_WINDOW_HOURS` escalates
normal → limited (short + time-capped) → blocked (message + hang up, no paid
pipeline). Tune with `ABUSE_LIMITED_AFTER`, `ABUSE_BLOCK_AFTER`,
`ABUSE_LIMITED_MAX_SECONDS` (add them to `--update-env-vars`). See
`docs/caller-policy.md`.

## Redeploying
- Code or `knowledge/` change: `gcloud run deploy letstalk-elle --source . --region us-central1`.
- Env/secret only: `gcloud run services update letstalk-elle --region us-central1 ...`.

## Cost (rough)
- **Web chat only:** Cloud Run warm instance ~$8–15/mo + a few cents per chat.
- **+ Phone (Deepgram cascade):** + ~$1.15/mo number + ~$0.06–0.09 per 3-min call.

## Compliance
Reasoning is already on Vertex (your GCP BAA). For real PHI, also sign the GCP
BAA and Deepgram's BAA, keep PHI out of logs, and lock down IAM. For the current
synthetic-data demo, none of this binds yet.
