#!/usr/bin/env bash
# Who has called the number, and what was said.
#
#   ./tools/check-calls.sh          # or: npm run check:calls
#   DAYS=30 ./tools/check-calls.sh  # widen the window (default 7 days)
#
# Two records, because they answer different questions and disagree on purpose:
#
#   Twilio    every call that ever reached the number, permanent. This is the
#             one to trust for "has anyone called".
#   Instance  what Elle remembers: transcripts, and the facts she captured.
#             It lives on the container's ephemeral disk, so a deploy or a
#             cold start wipes it. An empty instance record next to a
#             non-empty Twilio record means calls happened before this
#             revision started, not that nothing happened.
#
# No placeholders. Everything is looked up, so the command pastes as-is.
set -euo pipefail
. "$(dirname "$0")/_service.sh"

DAYS=${DAYS:-7}

require_gcloud
lookup_url

SINCE=$(python3 -c "
import datetime
print((datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=$DAYS)).strftime('%Y-%m-%d'))
")
TZ_NAME=$(service_env BOOKING_TIMEZONE)
TZ_NAME=${TZ_NAME:-America/Los_Angeles}

# ── Twilio: the permanent record ─────────────────────────────────────────────
echo "Twilio's record  ·  last $DAYS days  ·  times in $TZ_NAME"
echo

SID=$(service_env TWILIO_ACCOUNT_SID)
NUMBER=$(service_env TWILIO_FROM_NUMBER)

if [ -z "$SID" ]; then
  echo "  TWILIO_ACCOUNT_SID is not set on the service, so Twilio cannot be queried."
  echo "  The Console at console.twilio.com > Monitor > Logs > Calls has the same record."
else
  # Read the auth token straight into the curl call. It is never echoed, never
  # stored in a variable that gets printed, and never written to disk.
  # Secret Manager first, then the service's own env, because on this project
  # the value may still be a literal on the service rather than a secret.
  # Ask the service which secret feeds TWILIO_AUTH_TOKEN rather than assuming
  # the secret shares the variable's name. On this project it does not.
  TW_SECRET=$(service_secret TWILIO_AUTH_TOKEN)
  TW_SECRET=${TW_SECRET:-TWILIO_AUTH_TOKEN}
  TW_TOKEN=$(gcloud secrets versions access latest --secret="$TW_SECRET" 2>/dev/null || true)
  [ -z "$TW_TOKEN" ] && TW_TOKEN=$(service_env TWILIO_AUTH_TOKEN)
  if [ -z "$TW_TOKEN" ]; then
    echo "  Could not read TWILIO_AUTH_TOKEN from Secret Manager or from the service,"
    echo "  so Twilio was not queried. The Console has the same record."
  fi

  if [ -n "$TW_TOKEN" ]; then
    TO_ENC=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1], safe=''))" "$NUMBER")
    URL_TW="https://api.twilio.com/2010-04-01/Accounts/$SID/Calls.json?PageSize=50&StartTime%3E=$SINCE"
    if [ -n "$NUMBER" ]; then
      URL_TW="$URL_TW&To=$TO_ENC"
    else
      echo "  (TWILIO_FROM_NUMBER is not set, so this lists every call on the account,"
      echo "   outbound ones included. The direction column tells them apart.)"
      echo
    fi

    HTTP=$(curl -sS -u "$SID:$TW_TOKEN" -o /tmp/tw-calls.$$ -w '%{http_code}' "$URL_TW" || echo "000")
    unset TW_TOKEN
    if [ "$HTTP" = "200" ]; then
      python3 - "$TZ_NAME" /tmp/tw-calls.$$ <<'PY'
import json, sys, datetime
from email.utils import parsedate_to_datetime
try:
    from zoneinfo import ZoneInfo
    tz = ZoneInfo(sys.argv[1])
except Exception:
    tz = datetime.timezone.utc

with open(sys.argv[2]) as fh:
    calls = json.load(fh).get("calls", [])
if not calls:
    print("  No calls to the number in this window.")
    raise SystemExit

def when(c):
    try:
        return parsedate_to_datetime(c["start_time"]).astimezone(tz).strftime("%a %b %-d  %-I:%M %p")
    except Exception:
        return c.get("start_time") or "?"

def dur(c):
    try:
        s = int(c.get("duration") or 0)
    except Exception:
        return ""
    return f"{s//60}m {s%60:02d}s" if s >= 60 else f"{s}s"

for c in calls:
    # "completed" only means Twilio connected it. A 0s completed call is
    # someone hanging up before Elle said anything, which is worth seeing.
    print(f"  {when(c):<22} {c.get('from',''):<16} {c.get('direction',''):<13} {c.get('status',''):<11} {dur(c)}")
print()
print(f"  {len(calls)} call(s).")
PY
    elif [ "$HTTP" = "401" ]; then
      echo "  Twilio rejected the credentials (401). The stored TWILIO_AUTH_TOKEN may be"
      echo "  stale; rotating it in the Console requires updating the secret too."
    else
      echo "  Twilio returned HTTP $HTTP."
      head -c 300 /tmp/tw-calls.$$; echo
    fi
    rm -f /tmp/tw-calls.$$
  fi
fi

# ── The instance: transcripts, if this revision took the call ────────────────
echo
lookup_token

# Setting the bucket and RUNNING the code that writes to it are two different
# things, and the gap between them is a revision. Reporting "survives deploys"
# from the environment variable alone would announce durability that the
# running revision may not have, which is worse than reporting none. Ask the
# service what it is actually doing.
ARCHIVE=$(service_env CALL_ARCHIVE_BUCKET)
ARCHIVE_STATE=$(curl -sS -H "X-Admin-Token: $TOKEN" "$URL/admin/status.json" 2>/dev/null \
  | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    raise SystemExit
for c in d.get("checks", []):
    if c.get("feature") == "Call archive":
        print(str(c.get("detail")))
        raise SystemExit
' 2>/dev/null || true)

if [ -z "$ARCHIVE_STATE" ]; then
  if [ -n "$ARCHIVE" ]; then
    echo "Elle's record  ·  bucket set, but this revision does not report an archive"
    echo
    echo "  CALL_ARCHIVE_BUCKET is $ARCHIVE, and the running revision knows nothing"
    echo "  about it. The archive code has not been deployed yet, so calls are still"
    echo "  being kept only on the container. Run ./tools/deploy.sh"
    echo
  else
    echo "Elle's record  ·  this instance only, LOST on the next deploy"
  fi
else
  echo "Elle's record  ·  $ARCHIVE_STATE"
fi
echo

HTTP=$(curl -sS -o /tmp/calls.$$ -w '%{http_code}' \
  -H "X-Admin-Token: $TOKEN" "$URL/admin/calls.json" || echo "000")

case "$HTTP" in
  200)
    # The rows come from python; every line of explanation is written here, so
    # an empty list cannot end up printing a heading with nothing under it.
    if python3 - "$TZ_NAME" /tmp/calls.$$ <<'ROWS'
import json, sys, datetime
try:
    from zoneinfo import ZoneInfo
    tz = ZoneInfo(sys.argv[1])
except Exception:
    tz = datetime.timezone.utc

with open(sys.argv[2]) as fh:
    calls = json.load(fh).get("calls", [])
if not calls:
    raise SystemExit(3)

def when(s):
    try:
        d = datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc)
        return d.astimezone(tz).strftime("%a %b %-d  %-I:%M %p")
    except Exception:
        return s or "?"

for c in calls:
    turns = c.get("transcript") or []
    print(f"  {when(c.get('started_at','')):<22} {c.get('phone_number',''):<16} {len(turns)} turns")
print()
print(f"  {len(calls)} call(s) with a transcript.")
ROWS
    then
      echo
      echo "  Read the conversations:"
      echo "    $URL/admin/calls?token=$TOKEN"
      echo
      echo "  That link carries the admin token, so treat it like a password."
    else
      echo "  Nothing recorded here."
      if [ -n "$ARCHIVE" ]; then
        echo "  Calls taken before the archive was turned on are not in it."
      else
        echo "  If Twilio shows calls above, an earlier container took them and the"
        echo "  transcripts went with it."
      fi
    fi
    if [ -z "$ARCHIVE" ]; then
      echo
      echo "  Transcripts are NOT durable right now. ./tools/enable-archive.sh"
      echo "  creates a private bucket and points the service at it, after which"
      echo "  each call is written as it happens and replayed on startup."
    fi
    ;;
  403) echo "  403: the admin token did not match the one the service is deployed with." ;;
  404) echo "  404: the service sets no ADMIN_TOKEN, so /admin routes are off by design." ;;
  000) echo "  The service could not be reached." ;;
  *)   echo "  Unexpected HTTP $HTTP."; head -c 300 /tmp/calls.$$; echo ;;
esac
rm -f /tmp/calls.$$
