#!/usr/bin/env bash
# Make call history survive deploys.
#
#   ./tools/enable-archive.sh              # show the plan, change nothing
#   ./tools/enable-archive.sh --apply      # do it
#   RETAIN_DAYS=365 ./tools/enable-archive.sh --apply   # and expire old calls
#
# Cloud Run hands every container a fresh, empty disk, and the call database
# lives on it. A deploy, a scale to zero, or an ordinary instance replacement
# takes every transcript with it, and nothing errors when that happens: the
# call list simply comes back empty, which reads as a quiet week rather than as
# data loss.
#
# This creates a bucket, lets the service write to it, and points the service
# at it. From then on each call is written to the bucket as it happens and
# replayed into the new container on startup.
#
# The bucket holds recordings of conversations with real people, so it is
# created private, with public access prevented at the bucket level, and
# uniform access so a stray object ACL cannot open a single file.
set -euo pipefail
# Resolved before the cd, so the helper is found however this was invoked.
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE/.."
. "$HERE/_service.sh"

APPLY=0
for arg in "$@"; do
  case "$arg" in
    --apply) APPLY=1 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

require_gcloud

PROJECT=$(gcloud config get-value project 2>/dev/null || true)
if [ -z "$PROJECT" ] || [ "$PROJECT" = "(unset)" ]; then
  echo "No gcloud project is set. gcloud config set project YOUR-PROJECT" >&2
  exit 1
fi

BUCKET=${CALL_ARCHIVE_BUCKET:-${PROJECT}-elle-calls}

SPEC=$(gcloud run services describe "$SERVICE" --region "$REGION" --format=json 2>/dev/null || true)
if [ -z "$SPEC" ]; then
  echo "Could not read service '$SERVICE' in $REGION." >&2
  exit 1
fi

RUNTIME=$(printf '%s' "$SPEC" | python3 -c '
import json, sys
d = json.load(sys.stdin)
print(d.get("spec", {}).get("template", {}).get("spec", {}).get("serviceAccountName") or "")
')
if [ -z "$RUNTIME" ]; then
  NUM=$(gcloud projects describe "$PROJECT" --format='value(projectNumber)' 2>/dev/null || true)
  RUNTIME="${NUM}-compute@developer.gserviceaccount.com"
  RUNTIME_NOTE="(the Compute Engine default account, which is what the service runs as)"
else
  RUNTIME_NOTE=""
fi

CURRENT=$(service_env CALL_ARCHIVE_BUCKET)

echo "project   $PROJECT"
echo "service   $SERVICE ($REGION)"
echo "bucket    gs://$BUCKET"
echo "writer    $RUNTIME $RUNTIME_NOTE"
echo "currently ${CURRENT:-archive is off}"
echo

if [ "$APPLY" -eq 0 ]; then
  echo "Plan (nothing has changed):"
  echo
  echo "  1. create gs://$BUCKET in $REGION, private, uniform access,"
  echo "     public access prevented"
  echo "  2. grant $RUNTIME roles/storage.objectAdmin on it"
  [ -n "${RETAIN_DAYS:-}" ] && echo "  3. expire archived calls after ${RETAIN_DAYS} days"
  echo "  $([ -n "${RETAIN_DAYS:-}" ] && echo 4 || echo 3). point the service at it:"
  echo "       CALL_ARCHIVE_BUCKET=$BUCKET"
  echo
  echo "Existing calls on the running container are archived on the next write,"
  echo "and anything already lost to an earlier deploy stays lost. Twilio still"
  echo "has the record that those calls happened."
  echo
  echo "Run with --apply to do it."
  exit 0
fi

echo "==> bucket"
if gcloud storage buckets describe "gs://$BUCKET" >/dev/null 2>&1; then
  echo "    already exists"
else
  gcloud storage buckets create "gs://$BUCKET" \
    --location="$REGION" \
    --uniform-bucket-level-access \
    --public-access-prevention
  echo "    created"
fi

echo "==> access"
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \
  --member="serviceAccount:$RUNTIME" \
  --role=roles/storage.objectAdmin >/dev/null
echo "    $RUNTIME can read and write it"

if [ -n "${RETAIN_DAYS:-}" ]; then
  echo "==> retention"
  TMP=$(mktemp)
  printf '{"rule":[{"action":{"type":"Delete"},"condition":{"age":%s}}]}\n' "$RETAIN_DAYS" > "$TMP"
  gcloud storage buckets update "gs://$BUCKET" --lifecycle-file="$TMP" >/dev/null
  rm -f "$TMP"
  echo "    archived calls are deleted after $RETAIN_DAYS days"
fi

echo "==> pointing the service at it"
gcloud run services update "$SERVICE" --region "$REGION" \
  --update-env-vars "CALL_ARCHIVE_BUCKET=$BUCKET" >/dev/null
echo "    CALL_ARCHIVE_BUCKET=$BUCKET"

echo
echo "Done. Confirm it took:"
echo "  ./tools/check-calls.sh"
echo
echo "The status page's 'Call archive' row shows how many calls were replayed"
echo "on the last start, and names any error rather than failing quietly."
