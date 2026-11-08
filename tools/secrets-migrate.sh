#!/usr/bin/env bash
# Move API keys off the Cloud Run service config and into Secret Manager.
#
#   ./tools/secrets-migrate.sh           # show the plan, change nothing
#   ./tools/secrets-migrate.sh --apply   # do it
#
# An env var set with --set-env-vars holds its value in the service spec, where
# it is visible to anyone with Viewer on the project, sits in every revision's
# history, and comes back in plain text from 'gcloud run services describe'. A
# Secret Manager reference holds only a pointer; the value is fetched at boot by
# the runtime service account and never appears in the spec.
#
# The values are piped straight from the service description into gcloud. They
# are never printed, never written to a file, and never touch shell history.
#
# The switch happens in ONE service update: the plain variable is removed and
# the secret reference is added together. Doing it in two steps would either
# leave a gap where the variable is gone and the secret is not yet attached, or
# hit Cloud Run's refusal to have both a literal and a secret for one name.
set -euo pipefail
# Resolved before the cd, so the helper is found however this was invoked.
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE/.."
. "$HERE/_service.sh"

NAMES=(
  DEEPGRAM_API_KEY
  ANTHROPIC_API_KEY
  TWILIO_AUTH_TOKEN
  TELEGRAM_BOT_TOKEN
  TELEGRAM_WEBHOOK_SECRET
  ADMIN_TOKEN
  OPENAI_REALTIME_API_KEY
)

APPLY=0
for arg in "$@"; do
  case "$arg" in
    --apply) APPLY=1 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

require_gcloud

SPEC=$(gcloud run services describe "$SERVICE" --region "$REGION" --format=json 2>/dev/null || true)
if [ -z "$SPEC" ]; then
  echo "Could not read service '$SERVICE' in $REGION." >&2
  exit 1
fi

# Which of these are literals on the service right now.
PLAIN=$(printf '%s' "$SPEC" | python3 -c '
import json, sys
want = set(sys.argv[1:])
d = json.load(sys.stdin)
out = []
for c in d.get("spec", {}).get("template", {}).get("spec", {}).get("containers", []):
    for e in c.get("env", []):
        if e.get("name") in want and isinstance(e.get("value"), str) and e["value"]:
            out.append(e["name"])
print(" ".join(sorted(set(out))))
' "${NAMES[@]}")

if [ -z "$PLAIN" ]; then
  echo "Nothing to move: none of the known key names are set as literals on $SERVICE."
  exit 0
fi

# The secret's name follows this project's existing convention rather than the
# variable's spelling, so ADMIN_TOKEN lands next to twilio-token and
# deepgram-key instead of introducing a second naming style in one store.
# No associative array: this has to run on the bash that ships with macOS,
# which is 3.2 and does not have them. target_secret_name is deterministic, so
# recomputing it costs one lookup and keeps the plan and the apply in step.
echo "Set as plain text on $SERVICE:"
for n in $PLAIN; do
  t=$(target_secret_name "$n")
  if gcloud secrets describe "$t" >/dev/null 2>&1; then
    printf '  %-26s -> %s (exists, a new version is added)\n' "$n" "$t"
  else
    printf '  %-26s -> %s (created)\n' "$n" "$t"
  fi
done
echo

MOVE=$(echo $PLAIN | tr ' ' ',')
UPDATE=""
for n in $PLAIN; do
  UPDATE="${UPDATE:+$UPDATE,}$n=$(target_secret_name "$n"):latest"
done

if [ "$APPLY" -eq 0 ]; then
  echo "Plan (nothing has changed):"
  echo
  echo "  1. copy each value into Secret Manager, piped, never displayed"
  echo "  2. one atomic service update:"
  echo
  echo "     gcloud run services update $SERVICE --region $REGION \\"
  echo "       --remove-env-vars $MOVE \\"
  echo "       --update-secrets $UPDATE"
  echo
  echo "  3. confirm the service still works:"
  echo "     npm run check:calendar"
  echo
  echo "Run with --apply to do it."
  exit 0
fi

echo "==> copying values into Secret Manager"
for n in $PLAIN; do
  # printed to a pipe only, with no trailing newline: a stray \n at the end of
  # an API key is a 401 that takes an hour to find.
  read_value() {
    printf '%s' "$SPEC" | python3 -c '
import json, sys
want = sys.argv[1]
d = json.load(sys.stdin)
for c in d.get("spec", {}).get("template", {}).get("spec", {}).get("containers", []):
    for e in c.get("env", []):
        if e.get("name") == want and isinstance(e.get("value"), str):
            sys.stdout.write(e["value"]); raise SystemExit
' "$n"
  }
  t=$(target_secret_name "$n")
  if gcloud secrets describe "$t" >/dev/null 2>&1; then
    read_value | gcloud secrets versions add "$t" --data-file=- >/dev/null
    printf '    %-26s new version of %s\n' "$n" "$t"
  else
    read_value | gcloud secrets create "$t" --data-file=- --replication-policy=automatic >/dev/null
    printf '    %-26s created %s\n' "$n" "$t"
  fi
done

# The runtime account has to be able to read them, or the next revision fails to
# start. Granting is idempotent, so this is safe to repeat.
RUNTIME=$(printf '%s' "$SPEC" | python3 -c '
import json, sys
d = json.load(sys.stdin)
print(d.get("spec", {}).get("template", {}).get("spec", {}).get("serviceAccountName") or "")
')
if [ -n "$RUNTIME" ]; then
  echo
  echo "==> granting $RUNTIME read access"
  for n in $PLAIN; do
    gcloud secrets add-iam-policy-binding "$(target_secret_name "$n")" \
      --member="serviceAccount:$RUNTIME" \
      --role=roles/secretmanager.secretAccessor >/dev/null 2>&1 \
      && printf '    %-26s ok\n' "$n" \
      || printf '    %-26s could not grant, do it in the console\n' "$n"
  done
else
  echo
  echo "The service has no explicit service account, so it runs as the Compute"
  echo "Engine default account. That account usually can read secrets already."
fi

echo
echo "==> switching the service over in one update"
gcloud run services update "$SERVICE" --region "$REGION" \
  --remove-env-vars "$MOVE" \
  --update-secrets "$UPDATE"

echo
echo "Done. Confirm nothing broke:"
echo "  npm run check:calendar"
echo "  ./tools/check-calls.sh"
