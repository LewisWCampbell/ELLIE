# Shared lookup for the deployed service: its URL and its admin token.
# Sourced, not executed. Both check-calendar.sh and check-calls.sh use it so
# there is one place where "which service, and how do I authenticate to it"
# is answered, rather than two copies that drift apart.

SERVICE=${SERVICE:-letstalk-elle}
REGION=${REGION:-us-central1}

require_gcloud() {
  if ! command -v gcloud >/dev/null 2>&1; then
    echo "gcloud is required to look up the service URL and token." >&2
    exit 1
  fi
}

# Sets URL, or exits.
#
# gcloud's stderr is captured and interpreted rather than discarded. "The
# service does not exist" and "gcloud could not even ask" are different
# answers with different fixes, and printing one message for both sent its
# owner hunting for a missing service when the actual problem was an unset
# default project.
lookup_url() {
  local err_file="/tmp/svc-lookup-err.$$"
  URL=$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)' 2>"$err_file" || true)
  if [ -n "$URL" ]; then
    rm -f "$err_file"
    return
  fi

  local err project
  err=$(cat "$err_file" 2>/dev/null); rm -f "$err_file"
  project=$(gcloud config get-value project 2>/dev/null)

  echo "Could not read service '$SERVICE' in $REGION." >&2
  if [ -z "$project" ] || [ "$project" = "(unset)" ]; then
    echo >&2
    echo "gcloud has NO default project set, which is the usual cause after a" >&2
    echo "gcloud update or a new machine. Fix:" >&2
    echo "  gcloud config set project letstalk-elle" >&2
  elif printf '%s' "$err" | grep -qiE "reauth|credential|expired|unauthenticated|login"; then
    echo >&2
    echo "gcloud's login has expired (project is '$project'). Fix:" >&2
    echo "  gcloud auth login" >&2
  elif printf '%s' "$err" | grep -qiE "not found|could not be found"; then
    echo >&2
    echo "gcloud asked project '$project' and it has no such service. If the" >&2
    echo "project is wrong: gcloud config set project letstalk-elle" >&2
    echo "If the service name or region differ: SERVICE=other REGION=other $0" >&2
  else
    echo >&2
    echo "gcloud said (project '$project'):" >&2
    printf '%s\n' "$err" | head -4 | sed 's/^/  /' >&2
  fi
  exit 1
}

# Sets TOKEN, or explains what to do and exits 2.
#
# Three places the token can be, tried in order: the caller's environment, a
# literal on the service, and the secret the service reads it from. That last
# one matters after the token is moved into Secret Manager, which is the right
# place for it: without resolving the binding, every owner tool would start
# reporting "no token" the moment the setup got more secure.
lookup_token() {
  TOKEN="${ADMIN_TOKEN:-}"
  [ -z "$TOKEN" ] && TOKEN=$(service_env ADMIN_TOKEN)

  SECRET_BACKED=0
  if [ -z "$TOKEN" ]; then
    admin_secret=$(service_secret ADMIN_TOKEN)
    if [ -n "$admin_secret" ]; then
      SECRET_BACKED=1
      TOKEN=$(gcloud secrets versions access latest --secret="$admin_secret" 2>/dev/null || true)
    fi
  fi

  if [ -z "$TOKEN" ]; then
    echo "No admin token available."
    echo
    if [ "$SECRET_BACKED" = "1" ]; then
      echo "The service reads ADMIN_TOKEN from the secret '$admin_secret', which this"
      echo "account cannot access. Ask for roles/secretmanager.secretAccessor on it,"
      echo "or pass the value in directly:"
      echo
      echo "  ADMIN_TOKEN=... $0"
    else
      echo "The service does not set ADMIN_TOKEN at all, so every /admin route returns"
      echo "404 by design: owner views are off unless a token is configured. To turn"
      echo "them on, invent a long random value and store it, then redeploy:"
      echo
      echo "  printf '%s' \"\$(openssl rand -hex 24)\" | \\"
      echo "    gcloud secrets create admin-token --data-file=- --replication-policy=automatic"
      echo "  ./tools/deploy.sh"
    fi
    exit 2
  fi
}

# Reads a plain (non-secret) env var off the deployed service. Prints nothing
# when the variable is unset or is secret-backed.
service_env() {
  gcloud run services describe "$SERVICE" --region "$REGION" --format=json 2>/dev/null \
    | python3 -c '
import json, sys
want = sys.argv[1]
try:
    d = json.load(sys.stdin)
except Exception:
    raise SystemExit
for c in d.get("spec", {}).get("template", {}).get("spec", {}).get("containers", []):
    for e in c.get("env", []):
        if e.get("name") == want and isinstance(e.get("value"), str):
            print(e["value"]); raise SystemExit
' "$1" 2>/dev/null || true
}

# Prints the Secret Manager secret bound to env var NAME on the service, or
# nothing if that variable is a literal or unset.
#
# Worth resolving rather than assuming: the env var and the secret behind it
# are named independently, and on this project they differ (TWILIO_AUTH_TOKEN
# is served by a secret called twilio-token). Guessing the name is how a tool
# concludes a secret does not exist when it is sitting right there.
service_secret() {
  gcloud run services describe "$SERVICE" --region "$REGION" --format=json 2>/dev/null \
    | python3 -c '
import json, sys
want = sys.argv[1]
try:
    d = json.load(sys.stdin)
except Exception:
    raise SystemExit
for c in d.get("spec", {}).get("template", {}).get("spec", {}).get("containers", []):
    for e in c.get("env", []):
        if e.get("name") == want:
            ref = (e.get("valueFrom") or {}).get("secretKeyRef") or {}
            if ref.get("name"):
                print(ref["name"])
            raise SystemExit
' "$1" 2>/dev/null || true
}

# Names to try in Secret Manager for one environment variable, in order. The
# first that exists wins. The pairing is not mechanical: this project's secret
# for TWILIO_AUTH_TOKEN is called twilio-token, which no transformation of the
# variable name produces.
secret_candidates() {
  case "$1" in
    DEEPGRAM_API_KEY)  echo "deepgram-key DEEPGRAM_API_KEY" ;;
    ANTHROPIC_API_KEY) echo "anthropic-key ANTHROPIC_API_KEY" ;;
    TWILIO_AUTH_TOKEN) echo "twilio-token TWILIO_AUTH_TOKEN" ;;
    *) echo "$(printf '%s' "$1" | tr 'A-Z_' 'a-z-') $1" ;;
  esac
}

# The secret an env var should live in: an existing one under any known name,
# otherwise this project's convention, which is lower-case-with-hyphens.
target_secret_name() {
  for cand in $(secret_candidates "$1"); do
    if gcloud secrets describe "$cand" >/dev/null 2>&1; then echo "$cand"; return; fi
  done
  printf '%s' "$1" | tr 'A-Z_' 'a-z-'
}
