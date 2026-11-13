#!/usr/bin/env bash
# Background progress reporter for the VM jobs (started by diagnose.sh,
# pilot.sh and run_all.sh; one copy at a time). Every PROGRESS_EVERY seconds:
#   - writes a short status to gs://<bucket>/status.txt
#       read it with:  gcloud storage cat gs://letstalk-elle-elle-training/status.txt
#   - if NTFY_TOPIC is set, pushes a phone notification (ntfy app, topic name
#     = NTFY_TOPIC) whenever the latest milestone changes: a new stage, a
#     training progress line, a result, a failure. The messages hold stage
#     names and numbers only, and go through the public ntfy.sh server.
source "$(dirname "$0")/_env.sh"
set +e
EVERY="${PROGRESS_EVERY:-300}"
last=""
milestones='^=== |^run_all: |step [0-9]+:|tool calls made|call attempts|"tool_open_acc"|FAILED|Traceback|Error|done'
while true; do
  log=$(ls -t "$HOME"/pilot.log "$HOME"/diagnose.log "$HOME"/run_all.log "$HOME"/eval.log 2>/dev/null | head -1)
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null)
  step=$(ls -t "$ELLE_WORK"/runs/*/train_log.jsonl 2>/dev/null | head -1 | xargs -r tail -n 1 | cut -c1-60)
  mile=$( [ -n "$log" ] && tr -d '\000' < "$log" | grep -aE "$milestones" | tail -1 | cut -c1-200)
  status="$(date -u '+%Y-%m-%d %H:%M UTC') | $(basename "${log:-none}")
latest: ${mile:-starting}
training: ${step:-n/a}
gpu: ${gpu:-n/a}"
  echo "$status" | gcloud storage cp - "$BUCKET/status.txt" >/dev/null 2>&1
  if [ -n "${NTFY_TOPIC:-}" ] && [ -n "$mile" ] && [ "$mile" != "$last" ]; then
    curl -s -m 10 -H "Title: Elle training" -d "$mile" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null
    last="$mile"
  fi
  sleep "$EVERY"
done
