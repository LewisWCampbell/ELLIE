#!/usr/bin/env bash
# On the VM: every data step for the current dataset generation (DATA_TAG) and
# config (ELLE_CFG), in order, skipping what is done. Each step resumes if a
# Spot stop interrupts it. Called by run_all.sh; safe to run by hand.
source "$(dirname "$0")/_env.sh"

# Room for the new audio: the first run's audio and encodings (v1) are
# retired; its checkpoints and reports are in the bucket. KEEP_V1=1 keeps them.
if [ "${KEEP_V1:-0}" != 1 ] && [ "$DATA_TAG" != v1 ] && [ -d "$ELLE_WORK/audio" ]; then
  echo "prepare_all: removing the first run's audio and encodings to free disk space"
  rm -rf "$ELLE_WORK/audio" "$ELLE_WORK/audio_codes" "$ELLE_WORK/clips" "$ELLE_WORK/clips_plain" \
         "$ELLE_WORK/runs/prepared_v1" "$ELLE_WORK/runs/prepared_v2" "$ELLE_WORK/runs/prepared_zero" \
         "$ELLE_WORK/runs/plain_v1" "$ELLE_WORK/runs/plain_v2" "$ELLE_WORK/runs/plain_zero"
fi
df -h "$ELLE_WORK" | tail -1

step() {  # step NAME MARKER
  if [ -f "$ELLE_WORK/.done_$2" ]; then echo "prepare_all: $1 already done, skipping"; return; fi
  echo "prepare_all: $1 starting $(date)"
  bash "$REPO/training/gcp/prepare.sh" "$1"
  touch "$ELLE_WORK/.done_$2"
  echo "prepare_all: $1 finished $(date)"
}
# The audio steps belong to the dataset generation; the encoding steps also
# depend on the config (prompt, call format, layout).
step gen "gen_$DATA_TAG"
step synth "synth_$DATA_TAG"
step retime "retime_${DATA_TAG}_$CFG_TAG"
step assemble "assemble_${DATA_TAG}_$CFG_TAG"
step plain_audio "plain_audio_${DATA_TAG}_$CFG_TAG"
step mimi "mimi_${DATA_TAG}_$CFG_TAG"
step plain "plain_${DATA_TAG}_$CFG_TAG"
