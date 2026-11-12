#!/usr/bin/env bash
# On the VM: start (or resume) the full run.
#   nohup bash train.sh > ~/train.log 2>&1 &
# Checkpoints are copied to the bucket every 10 minutes. If Spot stops the VM,
# start it again and the boot service resumes from the last checkpoint. When
# the run finishes, results go to the bucket and the VM shuts itself down so
# it stops billing.
source "$(dirname "$0")/_env.sh"
# After a Spot stop during data preparation, the boot service lands here:
# hand over to run_all.sh, which finishes preparation before training.
if [ "${1:-}" = "--resumed" ] && [ -f "$ELLE_WORK/.pipeline" ]; then
  exec bash "$REPO/training/gcp/run_all.sh"
fi
RUN_DIR="$ELLE_WORK/runs/elle_tools_v1"
mkdir -p "$RUN_DIR"
touch "$ELLE_WORK/.training"

# Restore checkpoints from the bucket if this disk has none (a fresh VM).
if [ -z "$(ls -A "$RUN_DIR/checkpoints" 2>/dev/null)" ]; then
  gcloud storage rsync -r "$BUCKET/runs/elle_tools_v1" "$RUN_DIR" 2>/dev/null || true
fi

( while sleep 600; do gcloud storage rsync -r "$RUN_DIR" "$BUCKET/runs/elle_tools_v1" >/dev/null 2>&1 || true; done ) &
SYNC=$!
trap 'kill $SYNC 2>/dev/null || true' EXIT

python3 training/model/train.py --config training/gcp/config/gcp.yaml --resume

kill $SYNC 2>/dev/null || true
gcloud storage rsync -r "$RUN_DIR" "$BUCKET/runs/elle_tools_v1"
echo "Training finished; results are in $BUCKET/runs/elle_tools_v1."

# Evaluate on this same VM before shutting down: the GPU is already here and
# Spot capacity is scarce. The .training marker stays until the evaluation is
# over, so a Spot stop in the middle resumes here (training is at its last
# step and returns at once). A failed evaluation is not retried: its log goes
# to the bucket and the VM shuts down.
if [ "${EVAL_AFTER:-1}" = 1 ] && [ ! -f "$ELLE_WORK/.done_eval" ]; then
  echo "Evaluation starting $(date) (${EVAL_EPISODES:-40} test conversations); log: ~/eval.log"
  if EVAL_EPISODES="${EVAL_EPISODES:-40}" bash "$REPO/training/gcp/eval.sh" > "$HOME/eval.log" 2>&1; then
    echo "Evaluation finished $(date)"
  else
    echo "Evaluation FAILED $(date); see the uploaded log"
  fi
  touch "$ELLE_WORK/.done_eval"
  grep -av "UserWarning\|extern_kernels\|^  return Variable" "$HOME/eval.log" | tr -d '\000' | tail -c 300000 > "$HOME/eval.short.log" || true
  gcloud storage cp "$HOME/eval.short.log" "$BUCKET/logs/eval_$(date +%Y%m%d_%H%M).log" || true
  gcloud storage rsync -r "$RUN_DIR" "$BUCKET/runs/elle_tools_v1" || true
fi

rm -f "$ELLE_WORK/.training"
stop_watchdog
echo "Shutting down to stop billing."
sudo shutdown -h +2
