#!/usr/bin/env bash
# On the VM, in this order. Each step resumes if interrupted (a Spot stop):
# rerun the same command.
#
#   bash prepare.sh check      speak a sample in every voice; must report ok: true
#   bash prepare.sh synth      speak every line once, record real lengths  -> data/measured.jsonl
#   bash prepare.sh retime     make room for tool calls in the text stream -> data/retimed.jsonl
#   bash prepare.sh assemble   place the speech into stereo audio files    -> audio/
#   bash prepare.sh mimi       encode the audio for training; build the plain-conversation set
#
# For a trial run first: LIMIT=50 bash prepare.sh synth (and assemble), then
# listen to a few files in ~/elle-work/audio before doing all of it.
source "$(dirname "$0")/_env.sh"
TTS="$ELLE_WORK/tts-venv/bin/python"
LIMIT_ARGS=()
if [ -n "${LIMIT:-}" ]; then LIMIT_ARGS=(--limit "$LIMIT"); fi

case "${1:-}" in
  check)
    "$TTS" training/data/render_audio.py check-kokoro --out "$ELLE_WORK/kokoro_check" ;;
  synth)
    "$TTS" training/data/render_audio.py synth "$ELLE_WORK/data/episodes.jsonl" \
      --cache "$ELLE_WORK/clips" --out "$ELLE_WORK/data/measured.jsonl" \
      --backend kokoro --workers "${WORKERS:-6}" "${LIMIT_ARGS[@]}" ;;
  retime)
    python3 training/model/prepare.py --config training/gcp/config/smoke.yaml \
      --episodes "$ELLE_WORK/data/measured.jsonl" --out "$ELLE_WORK/runs/prepared_zero" \
      --write-retimed "$ELLE_WORK/data/retimed.jsonl" ;;
  assemble)
    "$TTS" training/data/render_audio.py assemble "$ELLE_WORK/data/retimed.jsonl" \
      --cache "$ELLE_WORK/clips" --out "$ELLE_WORK/audio" \
      --delete-clips --chunk 500 "${LIMIT_ARGS[@]}" ;;
  mimi)
    python3 training/model/prepare.py --config training/gcp/config/gcp.yaml
    python3 training/model/plain.py synth --config training/gcp/config/gcp.yaml --n-train 2000 --n-val 200 ;;
  *)
    echo "usage: bash prepare.sh check|synth|retime|assemble|mimi"; exit 2 ;;
esac
