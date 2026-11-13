#!/usr/bin/env bash
# On the VM, in this order. Each step resumes if interrupted (a Spot stop):
# rerun the same command.
#
#   bash prepare.sh check      speak a sample in every voice; must report ok: true
#   bash prepare.sh synth      speak every line once, record real lengths  -> data/measured.jsonl
#   bash prepare.sh retime     make room for tool calls in the text stream -> data/retimed.jsonl
#   bash prepare.sh assemble   place the speech into stereo audio files    -> audio/
#   bash prepare.sh mimi       encode the audio for training
#   bash prepare.sh plain      voice and encode the plain-conversation (no tools) set
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
    python3 training/model/prepare.py --config "$ELLE_CFG" ;;
  plain)
    # The plain-conversation set (no tools) goes through the same voice
    # pipeline as the tool data, so it has real audio too.
    D="$ELLE_WORK/data"
    # Voiced once (v1 did it under the marker .done_plain), reused by every run.
    if [ ! -f "$ELLE_WORK/.done_plain_audio" ] && [ ! -f "$ELLE_WORK/.done_plain" ]; then
      [ -f "$D/plain_episodes.jsonl" ] || python3 training/model/plain.py episodes \
        --n-train 2000 --n-val 200 --out "$D/plain_episodes.jsonl"
      "$TTS" training/data/render_audio.py synth "$D/plain_episodes.jsonl" \
        --cache "$ELLE_WORK/clips_plain" --out "$D/plain_measured.jsonl" --backend kokoro --workers "${WORKERS:-6}"
      python3 training/model/prepare.py --config training/gcp/config/smoke.yaml \
        --episodes "$D/plain_measured.jsonl" --out "$ELLE_WORK/runs/plain_zero" --write-retimed "$D/plain_retimed.jsonl"
      "$TTS" training/data/render_audio.py assemble "$D/plain_retimed.jsonl" \
        --cache "$ELLE_WORK/clips_plain" --out "$ELLE_WORK/audio" --delete-clips --chunk 500
      touch "$ELLE_WORK/.done_plain_audio"
    fi
    python3 training/model/plain.py synth --config "$ELLE_CFG" --episodes "$D/plain_retimed.jsonl" ;;
  *)
    echo "usage: bash prepare.sh check|synth|retime|assemble|mimi|plain"; exit 2 ;;
esac
