# Call recording + blob storage

Records realtime (barge-in) phone calls and stores them in Google Cloud Storage,
in the same GCP project as the app, so recordings stay inside your GCP BAA.

Because the call audio already flows through the bridge as G.711 μ-law 8 kHz
frames, recording is done **app-side**: no Twilio recording fees, and the audio
never leaves your infrastructure. Each call produces two objects:

```
gs://<bucket>/<prefix>/<YYYY-MM-DD>/<callSid>.wav    stereo audio: caller = left, Elle = right
gs://<bucket>/<prefix>/<YYYY-MM-DD>/<callSid>.json   metadata + turn-by-turn transcript
```

The WAV is standard 16-bit PCM stereo (μ-law decoded on the way out) so it plays
everywhere. The caller and Elle are on separate channels, aligned on the caller's
real-time frame clock (Elle's channel is silence-padded so her speech lands at the
right moment).

## What's built

- `src/recording/wav.ts` — μ-law → 16-bit stereo PCM WAV encoder (pure).
- `src/recording/recorder.ts` — `CallRecorder`: accumulates both directions and
  renders the WAV, aligning Elle's channel to the caller clock.
- `src/recording/blob.ts` — `BlobStore` interface + a GCS implementation
  (loaded lazily, only when recording is on). Swapping to S3/R2 is one new class.
- Wired into `twilioBridge.ts`: captures caller frames (`media`) and Elle's
  frames (`onAudioDelta`), and on hang-up uploads the WAV + JSON sidecar once.
- `npm run verify:recording` — offline test (no network): WAV header, channel
  alignment, and the BlobStore contract via a fake.

## Turning it on

```bash
RECORDING_ENABLED=true
RECORDING_BUCKET=your-recordings-bucket
RECORDING_PREFIX=recordings            # optional
```

On Google Cloud:

```bash
# create a bucket (uniform access; pick your region/retention)
gcloud storage buckets create gs://your-recordings-bucket --location=us-central1

# let the Cloud Run service account write to it
gcloud storage buckets add-iam-policy-binding gs://your-recordings-bucket \
  --member="serviceAccount:${PROJNUM}-compute@developer.gserviceaccount.com" \
  --role=roles/storage.objectAdmin

gcloud run services update letstalk-elle --region us-central1 \
  --update-env-vars RECORDING_ENABLED=true,RECORDING_BUCKET=your-recordings-bucket
```

Auth is ADC (the same service account that reaches Vertex) — no keys.

## Compliance & privacy (read before enabling)

Recording is **off by default** on purpose:

- **Consent / notice.** Many jurisdictions require all-party consent. When
  `RECORDING_ENABLED=true`, Elle is instructed to announce the recording in her
  opening turn automatically (via `recordingNotice()` in the persona). That
  covers the common case, but it relies on the model saying it — for hard legal
  compliance, prefer a **deterministic pre-roll**: a short `<Say>` before the
  `<Connect><Stream>` in `route.ts`, so the notice is guaranteed verbatim.
- **PHI / HIPAA.** GCS is a GCP-covered service, so recordings sit under your GCP
  BAA — but only once you've signed it and locked the bucket down (uniform
  bucket-level access, no public access, a retention/lifecycle policy, and
  restricted IAM). The transcript sidecar can contain whatever the caller said,
  so treat it as sensitive too.
- **Retention.** Set a bucket lifecycle rule to auto-delete after N days rather
  than keeping call audio forever.

## Notes / limits

- Only the **realtime** path is recorded (its audio streams through the bridge).
  The cascaded `<Gather>`/`<Say>` path has no raw audio here; use Twilio-side
  recording there if you need it.
- Audio is buffered in memory for the call's duration and flushed once on
  hang-up — fine for demo-length calls; for very long calls you'd stream to
  storage instead.
