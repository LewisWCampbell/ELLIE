#!/usr/bin/env bash
# On the VM, before the full v2 run: the go/no-go pilot. Unattended; uploads
# everything and shuts the VM down at the end.
#   tmux new -d -s pilot 'bash ~/ELLE/training/gcp/pilot.sh'
#
# 1. Encode the v2 data once (same audio as v1, new text stream; about 1 h).
# 2. For each learning rate: train 200 steps on 32 conversations, then
#    - markers.py on those conversations: <tool> accuracy/probability, teacher-forced
#    - closed-loop evaluation on 16 of them, greedy: calls made, parse errors
# Go for the full run if, for at least one learning rate, <tool> accuracy is
# >= 0.95 and the closed loop makes >= 80% of the expected calls with no errors.
source "$(dirname "$0")/_env.sh"
set +e  # one failing step must not stop the rest, the upload or the shutdown
LOG="$HOME/pilot.log"
STAMP=$(date +%Y%m%d_%H%M)
start_progress
if ! check_gpu > "$LOG" 2>&1; then
  gcloud storage cp "$LOG" "$BUCKET/logs/pilot_${STAMP}_gpu_failed.log" || true
  sudo shutdown -h +2; exit 1
fi
notify "Pilot started"
filter() { grep -av "UserWarning\|extern_kernels\|^  return Variable"; }
{
  for s in mimi plain; do
    if [ -f "$ELLE_WORK/.done_${s}_$CFG_TAG" ]; then echo "=== $s ($CFG_TAG) already done"; continue; fi
    echo "=== encode: $s ($CFG_TAG) $(date)"
    bash training/gcp/prepare.sh $s 2>&1 | filter
    [ "${PIPESTATUS[0]}" = 0 ] && touch "$ELLE_WORK/.done_${s}_$CFG_TAG"
  done
  for lr in ${PILOT_LRS:-2e-5 4e-6}; do
    C=training/gcp/config/pilot_lr$lr.yaml
    echo "=== pilot lr $lr: train $(date)"
    python3 training/model/train.py --config $C --resume 2>&1 | filter | grep -aE "step [0-9]+:|pilot:|trainable|Error|error|Traceback" 
    echo "=== pilot lr $lr: markers (teacher-forced) $(date)"
    python3 training/model/markers.py --config $C --split train --limit-episodes 32 2>&1 | filter | grep -aE '"tool_|"examples"|Error|Traceback'
    echo "=== pilot lr $lr: closed loop $(date)"
    python3 training/model/evaluate.py --config $C --mode model --split train --max-episodes 16 \
        --temperature-text 0 --tag pilot 2>&1 | filter | tail -5
    cat "$ELLE_WORK/runs/pilot_lr$lr/eval/pilot/model_summary.md" 2>/dev/null
    gcloud storage rsync -r "$ELLE_WORK/runs/pilot_lr$lr/eval" "$BUCKET/runs/pilot_lr$lr/eval" || true
    gcloud storage cp "$ELLE_WORK/runs/pilot_lr$lr/train_log.jsonl" "$BUCKET/runs/pilot_lr$lr/" || true
  done
  echo "=== done $(date)"
} >> "$LOG" 2>&1
gcloud storage cp "$LOG" "$BUCKET/logs/pilot_$STAMP.log" || true
notify "Pilot finished. Read logs/pilot_$STAMP.log"
sudo shutdown -h +2
