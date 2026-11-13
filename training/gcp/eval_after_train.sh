#!/usr/bin/env bash
# On the VM, while training is still running:
#   tmux new -d -s eval 'EVAL_EPISODES=40 bash ~/ELLE/training/gcp/eval_after_train.sh'
# Waits for training to finish, cancels its shutdown (the GPU is already
# here; Spot capacity is scarce), runs the evaluation, uploads the log and
# the results, then shuts the VM down, whether the evaluation passed or not.
ELLE_WORK="$HOME/elle-work"
BUCKET_LOG="gs://letstalk-elle-elle-training/logs/eval_$(date +%Y%m%d_%H%M).log"
while [ -f "$ELLE_WORK/.training" ]; do sleep 3; done
# A newer train.sh evaluates by itself; then there is nothing to do here.
if [ -f "$ELLE_WORK/.done_eval_${RUN_NAME:-elle_tools_v1}" ]; then exit 0; fi
touch "$ELLE_WORK/.done_eval_${RUN_NAME:-elle_tools_v1}"
( for _ in $(seq 60); do sudo shutdown -c 2>/dev/null; sleep 3; done ) &
bash "$HOME/ELLE/training/gcp/eval.sh" > "$HOME/eval.log" 2>&1
echo "eval exit code: $?" >> "$HOME/eval.log"
wait
grep -av "UserWarning\|extern_kernels\|^  return Variable" "$HOME/eval.log" | tail -c 300000 > "$HOME/eval.short.log"
gcloud storage cp "$HOME/eval.short.log" "$BUCKET_LOG"
gcloud storage rsync -r "$ELLE_WORK/runs/${RUN_NAME:-elle_tools_v1}/eval" "gs://letstalk-elle-elle-training/runs/${RUN_NAME:-elle_tools_v1}/eval" || true
sudo shutdown -h +3
