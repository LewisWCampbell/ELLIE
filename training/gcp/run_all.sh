#!/usr/bin/env bash
# On the VM: the whole pipeline, unattended, start to finish.
#   tmux new -d -s elle 'bash ~/ELLE/training/gcp/run_all.sh'
# synth -> retime -> assemble -> mimi -> train. Every step resumes, and the
# boot service reruns this script after a Spot stop (train.sh hands over to
# it while the .pipeline marker exists). On success train.sh shuts the VM
# down; on any failure this script shuts it down too, so nothing bills idle.
source "$(dirname "$0")/_env.sh"
# Always log to ~/run_all.log, also when the boot service (not tmux) runs us.
if [ -z "${RUN_ALL_LOGGING:-}" ]; then
  export RUN_ALL_LOGGING=1
  exec > >(tee -a "$HOME/run_all.log") 2>&1
fi
set -E  # let the ERR trap fire inside functions too
touch "$ELLE_WORK/.pipeline" "$ELLE_WORK/.training"

fail() {
  echo "run_all: FAILED at step '$STEP' ($(date)). Log: ~/run_all.log. Shutting down in 5 minutes."
  rm -f "$ELLE_WORK/.pipeline" "$ELLE_WORK/.training"
  # Copy the log to the bucket so the error can be read without a GPU.
  tail -c 200000 "$HOME/run_all.log" | tr -d '\000' > "$HOME/run_all.tail.log" || true
  gcloud storage cp "$HOME/run_all.tail.log" "$BUCKET/logs/run_all_failed_$(date +%Y%m%d_%H%M).log" || true
  stop_watchdog
  sudo shutdown -h +5
  exit 1
}
trap fail ERR

done_mark() { [ -f "$ELLE_WORK/.done_$1" ]; }
step() {
  STEP="$1"
  if done_mark "$1"; then echo "run_all: $1 already done, skipping"; return; fi
  echo "run_all: $1 starting $(date)"
  bash "$REPO/training/gcp/prepare.sh" "$1"
  touch "$ELLE_WORK/.done_$1"
  echo "run_all: $1 finished $(date)"
}

step synth
step retime
step assemble
step mimi
step plain

STEP=train
rm -f "$ELLE_WORK/.pipeline"
echo "run_all: training starting $(date)"
bash "$REPO/training/gcp/train.sh"
