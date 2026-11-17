#!/usr/bin/env bash
# The go/no-go pilot, run by `run_all.sh pilot` once the data is ready (it can
# also be run by hand). For each learning rate: train 200 steps on 32
# conversations, then
#   - markers.py on those conversations: <tool> accuracy/probability, teacher-forced
#   - a closed-loop evaluation on 16 of them: calls made, guard refusals
# Go for the full run if, for at least one learning rate, <tool> accuracy is
# >= 0.95 and the closed loop makes >= 80% of the expected calls.
# Each part is skipped if already done, so a resume after a Spot stop
# continues. At the end: upload, pause the restart job, shut down.
source "$(dirname "$0")/_env.sh"
set +e  # one failing step must not stop the rest, the upload or the shutdown
LOG="$HOME/pilot.log"
STAMP=$(date +%Y%m%d_%H%M)
start_progress
if ! check_gpu >> "$LOG" 2>&1; then
  gcloud storage cp "$LOG" "$BUCKET/logs/pilot_${STAMP}_gpu_failed.log" || true
  sudo shutdown -h +1; exit 1   # markers stay: the restart job resumes on other hardware
fi
notify "Pilot started"
filter() { grep -av "UserWarning\|extern_kernels\|^  return Variable"; }
{
  for lr in ${PILOT_LRS:-2e-5 4e-6}; do
    C=training/gcp/config/pilot_lr$lr.yaml
    if [ -f "$ELLE_WORK/.done_pilot_lr${lr}_$DATA_TAG" ]; then echo "=== pilot lr $lr already done"; continue; fi
    echo "=== pilot lr $lr: train $(date)"
    python3 training/model/train.py --config $C --resume 2>&1 | filter | grep -aE "step [0-9]+:|pilot:|trainable|resumed|Error|error|Traceback"
    echo "=== pilot lr $lr: markers (teacher-forced) $(date)"
    python3 training/model/markers.py --config $C --split train --limit-episodes 32 2>&1 | filter | grep -aE '"tool_|"examples"|Error|Traceback'
    echo "=== pilot lr $lr: closed loop $(date)"
    python3 training/model/evaluate.py --config $C --mode model --split train --max-episodes 16 \
        --temperature-text 0 --tag pilot 2>&1 | filter | tail -5
    cat "$ELLE_WORK/runs/pilot_lr$lr/eval/pilot/model_summary.md" 2>/dev/null
    gcloud storage rsync -r "$ELLE_WORK/runs/pilot_lr$lr/eval" "$BUCKET/runs/pilot_lr$lr/eval" || true
    gcloud storage cp "$ELLE_WORK/runs/pilot_lr$lr/train_log.jsonl" "$BUCKET/runs/pilot_lr$lr/" || true
    [ -f "$ELLE_WORK/runs/pilot_lr$lr/eval/pilot/model_summary.md" ] && touch "$ELLE_WORK/.done_pilot_lr${lr}_$DATA_TAG"
  done
  echo "=== done $(date)"
} >> "$LOG" 2>&1
gcloud storage cp "$LOG" "$BUCKET/logs/pilot_$STAMP.log" || true
notify "Pilot finished. Read logs/pilot_$STAMP.log"
rm -f "$ELLE_WORK/.pipeline" "$ELLE_WORK/.training"
stop_watchdog
sudo shutdown -h +2
