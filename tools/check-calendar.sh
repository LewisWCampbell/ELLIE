#!/usr/bin/env bash
# Ask the DEPLOYED service to check its own booking calendar.
#
#   ./tools/check-calendar.sh      # or: npm run check:calendar
#
# This is the authoritative check. It runs inside the service, as the account
# Elle actually books with, which is the only identity whose access matters.
# The local tools/smoke-calendar.ts runs as whoever is at the terminal.
#
# No placeholders: the service URL and the admin token are looked up, so the
# command can be pasted as-is. A command with a <placeholder> in it is a command
# that gets run verbatim by mistake, and bash reads < as a redirect.
set -euo pipefail
. "$(dirname "$0")/_service.sh"

require_gcloud
lookup_url
lookup_token

echo "checking $URL/admin/calendar-check"
echo

# No --fail. The body IS the report, and --fail discards the body on any error
# status, leaving nothing but a number to look at. The status is captured
# separately and interpreted below.
HTTP=$(curl -sS -o /tmp/calendar-check.$$ -w '%{http_code}' \
  -H "X-Admin-Token: $TOKEN" "$URL/admin/calendar-check" || echo "000")
cat /tmp/calendar-check.$$
rm -f /tmp/calendar-check.$$

case "$HTTP" in
  200) exit 0 ;;
  403)
    echo
    echo "403: the admin token did not match the one the service is deployed with."
    exit 1 ;;
  404)
    echo
    echo "404: the service sets no ADMIN_TOKEN, so every /admin route is off by"
    echo "design. Store one and redeploy to turn owner views on."
    exit 1 ;;
  000)
    echo
    echo "The service could not be reached at all."
    exit 1 ;;
  *)
    echo
    echo "Unexpected HTTP $HTTP. The body above is whatever the service returned."
    exit 1 ;;
esac
