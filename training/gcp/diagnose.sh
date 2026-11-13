#!/usr/bin/env bash
# On the VM: re-evaluate an existing checkpoint with the fixed evaluation, no
# training. Unattended; uploads everything and shuts the VM down at the end.
#   tmux new -d -s diag 'bash ~/ELLE/training/gcp/diagnose.sh'
#
#  1. markers: teacher-forced, did the checkpoint learn to START calls?
#     (<tool> accuracy/probability where the data has one; GPU minutes)
#  2. closed loop, greedy text, 10 test calls, 5 saved as audio
#  3. closed loop, sampled text (temp 0.7, top-k 25, NVIDIA's defaults), 2 seeds
#  4. closed loop with Elle's recorded audio forced (separates text from audio drift)
# Each closed-loop report also says how many calls were started, how many
# parsed, how many failed, and p(<tool>) around every place a call was due.
source "$(dirname "$0")/_env.sh"
set +e  # one failing step must not stop the rest, the upload or the shutdown
N="${DIAG_EPISODES:-10}"
# Defaults to the first run (v1); DIAG_CFG/DIAG_RUN point it at another.
CFG="${DIAG_CFG:-training/gcp/config/gcp.yaml}"
DRUN="${DIAG_RUN:-elle_tools_v1}"
OUT="$ELLE_WORK/runs/$DRUN/eval"
LOG="$HOME/diagnose.log"
STAMP=$(date +%Y%m%d_%H%M)
start_progress
notify "Diagnostic of $DRUN started"
run() { echo "=== $* ($(date))"; "$@" 2>&1 | grep -av "UserWarning\|extern_kernels\|^  return Variable"; }
{
  run python3 training/model/markers.py --config $CFG --split val --max-examples 200
  run python3 training/model/evaluate.py --config $CFG --mode model --split test --max-episodes "$N" \
      --temperature-text 0 --listen 5 --tag greedy
  for s in 0 1; do
    run python3 training/model/evaluate.py --config $CFG --mode model --split test --max-episodes "$N" \
        --temperature-text 0.7 --top-k-text 25 --seed-offset $s --tag sampled_s$s
  done
  run python3 training/model/evaluate.py --config $CFG --mode model --split test --max-episodes 5 \
      --temperature-text 0 --elle-audio teacher --tag teacher_audio
  echo "=== done $(date)"
} > "$LOG" 2>&1
gcloud storage cp "$LOG" "$BUCKET/logs/diagnose_$STAMP.log" || true
notify "Diagnostic finished: $(grep -a 'tool calls made' "$LOG" | head -1 | cut -c1-120)"
gcloud storage rsync -r "$OUT" "$BUCKET/runs/$DRUN/eval" || true
[ "${NO_SHUTDOWN:-0}" = 1 ] || sudo shutdown -h +2   # NO_SHUTDOWN=1 when another job follows
