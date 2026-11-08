#!/usr/bin/env bash
# End-to-end check of the booking calendar against the REAL Google Calendar.
#
#   ./tools/smoke-calendar.sh      # or: npm run smoke:calendar
#
# Run it in Cloud Shell, or anywhere `gcloud auth application-default login`
# has been done.
#
# This is only a wrapper. Every decision, including reading the calendar id off
# the deployed Cloud Run service when it is not set locally, lives in the
# TypeScript, so that running it through npm behaves identically. Logic that
# lives in one of two entry points is logic that half the callers never get.
#
# It creates one throwaway hold six months out and deletes it before exiting.
# Nobody is emailed: every write uses sendUpdates=none.
set -euo pipefail
cd "$(dirname "$0")/.."

# Dependencies have to be installed locally. Without them the failure is a
# module-resolution stack trace from inside the config loader, which points at
# dotenv and not at the actual problem.
if [ ! -d node_modules ]; then
  echo "node_modules is missing in $(pwd)." >&2
  echo "Run 'npm install' first, then try again." >&2
  exit 1
fi

exec ./node_modules/.bin/tsx tools/smoke-calendar.ts
