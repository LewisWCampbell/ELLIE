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
rm -f "$ELLE_WORK/.training"
stop_watchdog
echo "Training finished; results are in $BUCKET/runs/elle_tools_v1. Shutting down to stop billing."
sudo shutdown -h +2
