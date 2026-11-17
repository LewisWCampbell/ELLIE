#!/usr/bin/env bash
# On the VM: the whole pipeline, unattended, start to finish.
#   tmux new -d -s elle 'bash ~/ELLE/training/gcp/run_all.sh pilot'   data, then the go/no-go pilot
#   tmux new -d -s elle 'bash ~/ELLE/training/gcp/run_all.sh full'    data (skipped if done), train, evaluate
# Every step resumes, and the
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
touch "$ELLE_WORK/.training"
start_progress
# A sick GPU is not a failed run: keep the resume markers and stop, so the
# restart job brings the VM back (usually on other hardware) and this resumes.
if ! check_gpu; then
  echo "run_all: GPU failed its health check ($(date)); stopping for a restart on other hardware."
  sudo shutdown -h +1
  exit 1
fi

fail() {
  echo "run_all: FAILED at step '$STEP' ($(date)). Log: ~/run_all.log. Shutting down in 5 minutes."
  notify "FAILED at step $STEP. The log is in the bucket under logs/."
  rm -f "$ELLE_WORK/.pipeline" "$ELLE_WORK/.training"
  # Copy the log to the bucket so the error can be read without a GPU.
  tail -c 200000 "$HOME/run_all.log" | tr -d '\000' > "$HOME/run_all.tail.log" || true
  gcloud storage cp "$HOME/run_all.tail.log" "$BUCKET/logs/run_all_failed_$(date +%Y%m%d_%H%M).log" || true
  stop_watchdog
  sudo shutdown -h +5
  exit 1
}
trap fail ERR

# What to do after the data is ready: "pilot" (the go/no-go check) or "full"
# (train and evaluate). Given once on the command line and remembered in the
# .pipeline marker, so a resume after a Spot stop continues the same job.
MODE="${1:-$(cat "$ELLE_WORK/.pipeline" 2>/dev/null || true)}"
MODE="${MODE:-full}"
echo "$MODE" > "$ELLE_WORK/.pipeline"
echo "run_all: mode $MODE, data $DATA_TAG, config $CFG_TAG ($(date))"

STEP=data
bash "$REPO/training/gcp/prepare_all.sh"

if [ "$MODE" = pilot ]; then
  STEP=pilot
  bash "$REPO/training/gcp/pilot.sh"
  exit 0
fi

STEP=train
rm -f "$ELLE_WORK/.pipeline"
echo "run_all: training starting $(date)"
bash "$REPO/training/gcp/train.sh"
