# Claude on Vertex AI — BAA-ready from day one

This is the compliance-native LLM path. It runs the **same Claude models** as the
default provider, but through **Google Vertex AI** instead of `api.anthropic.com`.
Because Vertex is a Google Cloud service, its calls fall under the **GCP BAA** that
already covers your Cloud Run compute, Secret Manager, and logging — so there's no
separate model-vendor BAA to sign. One vendor, one agreement, one security perimeter.

> The adapter doesn't make you compliant on its own — **you** accept the GCP BAA.
> What it does is route model traffic through a BAA-*covered* Google service, so the
> agreement you sign actually applies to it. Without it, the LLM call would hit a
> non-covered endpoint and sit outside the BAA. See the "Why this matters" note below.

## What's already built

- `src/llm/vertex.ts` — a `LlmProvider` that talks to Claude on Vertex. It shares the
  exact message-transform and response-parsing as the direct Anthropic path
  (`src/llm/anthropicShared.ts`), so behavior is identical; only the client and auth
  differ. The client is built lazily on first request.
- Wired into `src/llm/provider.ts` and `src/config.ts` under `LLM_PROVIDER=vertex`.
- `npm run verify:vertex` — offline check (no GCP creds, no network) that the provider
  wires up and the transforms round-trip.

## Turning it on

Nothing to write — just configuration:

```bash
LLM_PROVIDER=vertex
VERTEX_PROJECT_ID=your-gcp-project        # or set GOOGLE_CLOUD_PROJECT
VERTEX_REGION=us-east5                     # a region where the model is served
VERTEX_MODEL=claude-sonnet-4-5@20250929    # exact Model Garden id (see below)
```

**Auth is ADC (Application Default Credentials) — no API key is stored anywhere:**
- On **Cloud Run / GCE**: the machine's service account is used automatically. Grant it
  the `roles/aiplatform.user` role.
- **Locally**: run `gcloud auth application-default login` once.

### One-time GCP setup

1. Enable the API: `gcloud services enable aiplatform.googleapis.com`.
2. In the Cloud console → **Vertex AI → Model Garden**, find the Claude model you want
   and click **Enable** (Anthropic models are enable-on-first-use).
3. Confirm the model id and region. Vertex model ids are pinned and **region-specific**
   — `claude-sonnet-4-5@20250929`, `claude-opus-4-1@20250805`, etc. Set `VERTEX_MODEL`
   to exactly what Model Garden lists for your `VERTEX_REGION`; a wrong id or a region
   that doesn't serve it returns a 404 at request time.
4. Give the runtime service account `roles/aiplatform.user`.

That's the whole switch. Point `LLM_PROVIDER` at `vertex`, and the website chat and the
cascaded phone brain both run on Claude-in-your-GCP-project.

## Why this matters (the compliance mechanism)

A vendor's BAA only covers PHI when three things are true: you've signed that vendor's
BAA, the **specific service** is on their HIPAA-covered list, and you use it per their
config rules.

- Sending PHI to the **consumer** Gemini endpoint (`generativelanguage.googleapis.com`)
  is *not* covered by the GCP BAA, even inside a GCP project.
- Sending it to **Anthropic/OpenAI directly** requires a *separate* BAA from that company.
- Sending it through **Vertex AI** — a GCP-covered service — puts it under the same GCP
  BAA as the rest of your stack. That's the "native" part.

## What this does NOT cover

- **Realtime voice (Phase 2).** The OpenAI Realtime leg is a different vendor and needs
  its own BAA, or a move to a GCP-covered speech path / self-hosted model. The LLM being
  on Vertex doesn't extend to the audio vendor.
- **Your own obligations.** Keep PHI out of logs, lock down IAM (and VPC-SC if you want
  the network perimeter), and actually accept the GCP BAA for your organization.

## For the current demo

The demo uses **synthetic data only**, so nothing here binds yet — keep Phase 1 on the
default Anthropic (or Gemini) key if you like. This path exists so that the day you move
toward real PHI, going compliant is a config change plus signing the GCP BAA, not a
re-architecture.
