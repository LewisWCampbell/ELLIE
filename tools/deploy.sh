#!/usr/bin/env bash
# Deploy Elle to Cloud Run.
#
#   ./tools/deploy.sh              # verify, then deploy
#   ./tools/deploy.sh --skip-checks    # deploy without running the suites first
#   ./tools/deploy.sh --dry-run        # print the command, change nothing
#
# Everything that has to be set at deploy time lives here rather than in a
# command someone retypes from memory. Retyping is how a variable gets dropped,
# and a dropped variable does not fail loudly: it falls back to a default and
# the service comes up subtly wrong.
#
# SECRETS ARE NEVER IN THIS FILE. They are Secret Manager references, resolved
# by Cloud Run at boot. To add or rotate one, pipe the value in so it never
# reaches your shell history or this repository:
#
#   printf '%s' 'THE-VALUE' | gcloud secrets create NAME --data-file=- --replication-policy=automatic
#   printf '%s' 'THE-VALUE' | gcloud secrets versions add NAME --data-file=-

set -euo pipefail
# Resolved before the cd, so the helper is found however this was invoked.
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE/.."
. "$HERE/_service.sh"

# ── Plain configuration ──────────────────────────────────────────────────────
# Values here are not sensitive. Anything that is belongs in the secrets list.
#
# Note the "^@^" prefix. gcloud splits this list on commas by default, and
# LANGUAGES=en,es contains one, so a comma-delimited list would set LANGUAGES to
# "en" and then try to parse "es" as another assignment. The prefix picks @ as
# the delimiter instead. This is exactly the kind of thing that does not error,
# it just quietly turns bilingual mode off.
#
# Only the variables listed here are touched. --update-env-vars leaves every
# other variable already on the service alone, which is why things like
# BRAND_NAME, BOOKING_CALENDAR_ID, BOOKING_BUSY_CALENDARS, CALL_ARCHIVE_BUCKET, VERTEX_PROJECT_ID,
# TWILIO_ACCOUNT_SID, TWILIO_FROM_NUMBER and TELEGRAM_OWNER_ID are deliberately
# absent: they are set once and survive every deploy.
ENV_VARS='^@^'$(tr -d '\n' <<'EOF'
SITE_URL=https://lewiswcampbell.com@
ASK_PATH=/ask.html@
CORS_ORIGIN=https://lewiswcampbell.com@
VOICE_MODE=realtime@
REALTIME_PROVIDER=deepgram@
DEEPGRAM_STT_MODEL=nova-2-phonecall@
DEEPGRAM_STT_MULTI_MODEL=nova-3@
DEEPGRAM_TTS_MODEL=aura-2-thalia-en@
DEEPGRAM_TTS_MODEL_ES=aura-2-celeste-es@
DEEPGRAM_ENDPOINTING_MS=1000@
LANGUAGES=en,es@
GEMINI_LIVE_LOCATION=us-central1@
BOOKING_TIMEZONE=America/Los_Angeles@
MAX_CONCURRENT_CALLS=20@
BOOKING_DURATION_MINUTES=30@
MAX_CALL_SECONDS=600@
SILENT_HANGUP_SECONDS=75
EOF
)

# ── Secret Manager references ────────────────────────────────────────────────
# Which environment variable is fed by which secret.
#
# The two names are chosen independently, and here they differ: TWILIO_AUTH_TOKEN
# is served by a secret called twilio-token. Assuming they match is how this
# script once reported every secret missing while all of them were sitting in
# Secret Manager under their real names. Bindings already on the service are
# read back and left alone, so this list is a fallback for anything not yet
# bound rather than the source of truth.
SECRET_ENV_NAMES=(
  DEEPGRAM_API_KEY
  ANTHROPIC_API_KEY
  TWILIO_AUTH_TOKEN
  TELEGRAM_BOT_TOKEN
  TELEGRAM_WEBHOOK_SECRET
  ADMIN_TOKEN
  OPENAI_REALTIME_API_KEY
)

# Dependencies have to be installed locally. Without them the failure is a
# module-resolution stack trace from inside the config loader, which points at
# dotenv and not at the actual problem.
if [ ! -d node_modules ]; then
  echo "node_modules is missing in $(pwd)." >&2
  echo "Run 'npm install' first, then try again." >&2
  exit 1
fi

# The knowledge corpus is gitignored (private) but REQUIRED at runtime: it
# reaches Cloud Run only because .gcloudignore deliberately does not exclude
# it. A checkout without it — a fresh clone of the public repo — would deploy
# an Elle who knows nothing, and nothing downstream would error. Refuse here.
if [ -z "$(ls knowledge/*.md 2>/dev/null)" ]; then
  echo "knowledge/ is missing or empty in $(pwd)." >&2
  echo "The corpus is private and not in the repository; restore it from your" >&2
  echo "backup before deploying, or Elle ships knowing nothing." >&2
  exit 1
fi

DRY_RUN=0
SKIP_CHECKS=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --skip-checks) SKIP_CHECKS=1 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

# ── Pre-flight ───────────────────────────────────────────────────────────────
# Deploying a build that does not compile wastes several minutes of Cloud Build
# to tell you something tsc knows in seconds.
if [ "$SKIP_CHECKS" -eq 0 ]; then
  echo "==> typecheck"
  npm run --silent typecheck

  echo "==> documentation"
  # Names only, and it takes a second. A doc that names a setting the code
  # stopped reading is worse than an omission: it looks like it worked.
  if npm run --silent check:docs >/tmp/check-docs.log 2>&1; then
    echo "    matches the code"
  else
    echo "    DRIFT"
    echo
    sed 's/^/    /' /tmp/check-docs.log
    echo
    echo "Not deploying. Fix the documentation, or pass --skip-checks."
    exit 1
  fi
  echo

  echo "==> offline verification"
  for suite in bridge cascade knowledge booking policy recording factory audio gemini intake archive calendar sms notify greeting gaps; do
    printf '    %-10s ' "$suite"
    if npm run --silent "verify:$suite" >/tmp/verify-$suite.log 2>&1; then
      echo "ok"
    else
      echo "FAILED"
      echo
      tail -20 /tmp/verify-$suite.log
      echo
      echo "Not deploying. Fix the above, or pass --skip-checks to deploy anyway."
      exit 1
    fi
  done
  echo
fi

# Attach only what is needed. A binding the service already has is left
# exactly as it is: re-stating it would be a no-op at best, and pointing it at
# a differently named secret would be an outage.
echo "==> secrets"
SPEC=$(gcloud run services describe "$SERVICE" --region "$REGION" --format=json 2>/dev/null || true)
ATTACH=()
PLAIN=()
for name in "${SECRET_ENV_NAMES[@]}"; do
  bound=""
  literal=0
  if [ -n "$SPEC" ]; then
    read -r bound literal <<<"$(printf '%s' "$SPEC" | python3 -c '
import json, sys
want = sys.argv[1]
d = json.load(sys.stdin)
for c in d.get("spec", {}).get("template", {}).get("spec", {}).get("containers", []):
    for e in c.get("env", []):
        if e.get("name") == want:
            ref = (e.get("valueFrom") or {}).get("secretKeyRef") or {}
            print(ref.get("name") or "-", 1 if isinstance(e.get("value"), str) and e["value"] else 0)
            raise SystemExit
print("-", 0)
' "$name" 2>/dev/null || echo "- 0")"
  fi

  if [ "${bound:--}" != "-" ] && [ -n "${bound:-}" ]; then
    printf '    %-26s already bound to %s\n' "$name" "$bound"
    continue
  fi

  if [ "${literal:-0}" = "1" ]; then
    PLAIN+=("$name")
    printf '    %-26s set as PLAIN TEXT on the service\n' "$name"
    continue
  fi

  found=""
  for cand in $(secret_candidates "$name"); do
    if gcloud secrets describe "$cand" >/dev/null 2>&1; then found="$cand"; break; fi
  done
  if [ -n "$found" ]; then
    ATTACH+=("$name=$found:latest")
    printf '    %-26s attaching %s\n' "$name" "$found"
  else
    printf '    %-26s not configured\n' "$name"
  fi
done
SECRETS=$(IFS=,; echo "${ATTACH[*]}")

if [ ${#PLAIN[@]} -gt 0 ]; then
  echo
  echo "    The values above live in the service spec, which means they are"
  echo "    readable by anyone with Viewer on the project and are kept in every"
  echo "    revision's history. ./tools/secrets-migrate.sh moves them into"
  echo "    Secret Manager without ever printing them."
fi
echo

CMD=(gcloud run deploy "$SERVICE"
  --source .
  --region "$REGION"
  --min-instances 1
  --update-env-vars "$ENV_VARS"
)
if [ -n "$SECRETS" ]; then
  CMD+=(--update-secrets "$SECRETS")
fi

if [ "$DRY_RUN" -eq 1 ]; then
  echo "==> dry run, not deploying:"
  printf '%q ' "${CMD[@]}"; echo
  exit 0
fi

echo "==> deploying $SERVICE to $REGION"
"${CMD[@]}"

URL=$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)')
echo
echo "Deployed: $URL"
echo
echo "Worth checking now, both runnable as-is:"
echo
echo "  ./tools/check-calendar.sh"
echo "      asks the service to check its own booking calendar, as the account"
echo "      it books with. This is the authoritative one."
echo
echo "  curl -sS -H \"X-Admin-Token: \$ADMIN_TOKEN\" $URL/admin/status"
echo "      live probes: calendar, intake bot webhook, active architecture"
