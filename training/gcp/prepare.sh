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
  gen)
    # The conversations themselves, and a check that the real tokenizer can
    # write every tool call exactly (an unknown character would silently
    # corrupt the calls in training and make them unparseable later).
    mkdir -p "$DATA_DIR"
    python3 training/data/generate.py --out "$DATA_DIR" --prefix full_ \
      --train "${N_TRAIN:-20000}" --val "${N_VAL:-1000}" --test "${N_TEST:-1000}" --seed 7 --holdout-family slot_conflict
    python3 training/data/validate.py "$DATA_DIR"/full_*.jsonl --holdout-family slot_conflict
    cat "$DATA_DIR"/full_{train,val,test}.jsonl > "$DATA_DIR/episodes.jsonl"
    python3 training/gcp/tokenizer_check.py --config "$ELLE_CFG" --episodes "$DATA_DIR/episodes.jsonl" ;;
  synth)
    "$TTS" training/data/render_audio.py synth "$DATA_DIR/episodes.jsonl" \
      --cache "$CLIPS_DIR" --out "$DATA_DIR/measured.jsonl" \
      --backend kokoro --workers "${WORKERS:-6}" "${LIMIT_ARGS[@]}" ;;
  retime)
    python3 training/model/prepare.py --config "${ELLE_CFG%.yaml}_retime.yaml" \
      --episodes "$DATA_DIR/measured.jsonl" --out "$ELLE_WORK/runs/retime_$DATA_TAG" \
      --write-retimed "$DATA_DIR/retimed.jsonl" ;;
  assemble)
    "$TTS" training/data/render_audio.py assemble "$DATA_DIR/retimed.jsonl" \
      --cache "$CLIPS_DIR" --out "$AUDIO_DIR" \
      --delete-clips --chunk 500 "${LIMIT_ARGS[@]}"
    python3 training/gcp/repair_audio.py "$DATA_DIR/retimed.jsonl" "$AUDIO_DIR" ;;
  plain_audio)
    # The plain-conversation set (no tools) goes through the same voice
    # pipeline as the tool data, so it has real audio too.
    [ -f "$DATA_DIR/plain_episodes.jsonl" ] || python3 training/model/plain.py episodes \
      --n-train 2000 --n-val 200 --out "$DATA_DIR/plain_episodes.jsonl"
    "$TTS" training/data/render_audio.py synth "$DATA_DIR/plain_episodes.jsonl" \
      --cache "$CLIPS_DIR" --out "$DATA_DIR/plain_measured.jsonl" --backend kokoro --workers "${WORKERS:-6}"
    python3 training/model/prepare.py --config "${ELLE_CFG%.yaml}_retime.yaml" \
      --episodes "$DATA_DIR/plain_measured.jsonl" --out "$ELLE_WORK/runs/retime_plain_$DATA_TAG" \
      --write-retimed "$DATA_DIR/plain_retimed.jsonl"
    "$TTS" training/data/render_audio.py assemble "$DATA_DIR/plain_retimed.jsonl" \
      --cache "$CLIPS_DIR" --out "$AUDIO_DIR" --delete-clips --chunk 500 ;;
  mimi)
    # Encodes the audio (cached per file, so an interrupted run resumes) and
    # builds the training examples.
    python3 training/model/prepare.py --config "$ELLE_CFG" ;;
  plain)
    python3 training/model/plain.py synth --config "$ELLE_CFG" --episodes "$DATA_DIR/plain_retimed.jsonl" ;;
  *)
    echo "usage: bash prepare.sh check|gen|synth|retime|assemble|plain_audio|mimi|plain"; exit 2 ;;
esac
