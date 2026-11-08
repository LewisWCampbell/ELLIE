#!/usr/bin/env bash
# Render Elle's greeting through several Deepgram voices so they can be
# compared by ear. Voice choice is subjective and the marketing adjectives in
# the catalog ("energetic", "warm") do not survive the phone band, so the only
# honest way to pick one is to listen.
#
#   export DEEPGRAM_API_KEY=...        # or run in Cloud Shell after fetching it
#   ./tools/preview-voices.sh          # renders the default shortlist
#   ./tools/preview-voices.sh aura-2-juno-en aura-2-iris-en   # or your own list
#
# Output lands in ./voice-previews/ as one WAV per voice.
#
# NOTE the sample rate below. These render at 24 kHz so they are pleasant on
# laptop speakers, which is NOT what a caller hears: on a real call the audio is
# G.711 mu-law at 8 kHz, which rolls off everything under roughly 300 Hz and
# above 3.4 kHz. A voice that sounds rich here can sound thin on the phone, and
# a voice that sounds thin here often sounds fine. Pass --phone to render at
# 8 kHz mu-law instead, which is what the caller actually gets.

set -euo pipefail

: "${DEEPGRAM_API_KEY:?set DEEPGRAM_API_KEY first}"

# What Elle actually opens a call with, so the comparison is on real copy
# rather than a neutral test sentence.
# Matches the deployed greeting's opening. The earlier wording put the two
# names either side of a comma ("Elle, Lewis Campbell's") and Aura's comma
# pause made the intro stumble; keep this in step with twilioBridge.ts so what
# you audition is what callers hear.
LINE="Hi there! I'm Elle, the AI assistant for Lewis Campbell. I can tell you about his work, or get you booked for a quick intro call. What brings you in today?"

PHONE=0
VOICES=()
for arg in "$@"; do
  case "$arg" in
    --phone) PHONE=1 ;;
    -*) echo "unknown flag: $arg" >&2; exit 2 ;;
    *) VOICES+=("$arg") ;;
  esac
done

# Bright-to-mellow shortlist. The first is the current default.
if [ ${#VOICES[@]} -eq 0 ]; then
  VOICES=(
    aura-2-thalia-en      # bright, confident, American   <- current default
    aura-2-electra-en     # energetic, dynamic
    aura-stella-en        # fast, casual, high energy
    aura-2-amalthea-en    # warm and approachable
    aura-2-juno-en        # natural, melodic
    aura-athena-en        # the previous voice, British, lower register
  )
fi

if [ "$PHONE" -eq 1 ]; then
  ENCODING="encoding=mulaw&sample_rate=8000&container=wav"
  SUFFIX="-phone"
else
  ENCODING="encoding=linear16&sample_rate=24000&container=wav"
  SUFFIX=""
fi

OUT=voice-previews
mkdir -p "$OUT"

for voice in "${VOICES[@]}"; do
  dest="$OUT/${voice}${SUFFIX}.wav"
  code=$(curl -sS -o "$dest" -w '%{http_code}' \
    -X POST "https://api.deepgram.com/v1/speak?model=${voice}&${ENCODING}" \
    -H "Authorization: Token ${DEEPGRAM_API_KEY}" \
    -H "Content-Type: application/json" \
    -d "$(printf '{"text": %s}' "$(printf '%s' "$LINE" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')")")

  if [ "$code" = "200" ]; then
    printf '  ok    %-24s %s\n' "$voice" "$dest"
  else
    # A bad voice id comes back as JSON in the .wav, so show it rather than
    # leaving a file that fails to play for no visible reason.
    printf '  FAIL  %-24s HTTP %s: %s\n' "$voice" "$code" "$(head -c 200 "$dest")"
    rm -f "$dest"
  fi
done

echo
echo "Listen, then set the winner:"
echo "  gcloud run services update letstalk-elle --region us-central1 \\"
echo "    --update-env-vars DEEPGRAM_TTS_MODEL=<voice>"
