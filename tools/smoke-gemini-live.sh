#!/usr/bin/env bash
# Prove Gemini Live actually works for THIS project before trusting it on a
# call. Opens a real session, says one thing, and writes the reply to a WAV.
#
#   ./tools/smoke-gemini-live.sh
#   ./tools/smoke-gemini-live.sh gemini-live-2.5-flash-preview-native-audio
#
# Run it in Cloud Shell, or anywhere `gcloud auth application-default login`
# has been done. It reads VERTEX_PROJECT_ID, GEMINI_LIVE_LOCATION, and
# GEMINI_LIVE_MODEL from the environment or .env, with the same defaults the
# service uses.
#
# WHY THIS EXISTS: the Live API's model ids move, and availability is per
# project, per region, and sometimes per allowlist. Every one of those failures
# looks the same from a phone call (silence), so check it here where the error
# message is readable.
#
# Output lands in ./voice-previews/gemini-live-smoke.wav — 24 kHz mono, which is
# what the model emits before the service resamples it down to phone audio.

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-${GEMINI_LIVE_MODEL:-}}"
# Dependencies have to be installed locally. Without them the failure is a
# module-resolution stack trace from inside the config loader, which points at
# dotenv and not at the actual problem.
if [ ! -d node_modules ]; then
  echo "node_modules is missing in $(pwd)." >&2
  echo "Run 'npm install' first, then try again." >&2
  exit 1
fi

exec ./node_modules/.bin/tsx tools/smoke-gemini-live.ts "$MODEL"
