#!/usr/bin/env bash
# On the VM. Reference mode scores the dataset itself (a sanity check on the
# metrics); model mode runs the trained model through every test call,
# answers its tool calls with the scripted results, and gates the scores. It
# also saves 5 of Elle's replies as WAV under eval/listen/: listen to them.
source "$(dirname "$0")/_env.sh"
python3 training/model/evaluate.py --config training/gcp/config/gcp.yaml --mode reference --split test
# Teacher-forced: did training teach the model to start calls at all?
python3 training/model/markers.py --config training/gcp/config/gcp.yaml --split val --max-examples 200
# Closed loop, greedy and with NVIDIA's default text sampling.
python3 training/model/evaluate.py --config training/gcp/config/gcp.yaml --mode model --split test --listen 5 \
  --max-episodes "${EVAL_EPISODES:-500}" --temperature-text 0 --tag greedy
python3 training/model/evaluate.py --config training/gcp/config/gcp.yaml --mode model --split test \
  --max-episodes "${EVAL_EPISODES:-500}" --temperature-text 0.7 --top-k-text 25 --tag sampled
gcloud storage rsync -r "$ELLE_WORK/runs/elle_tools_v1/eval" "$BUCKET/runs/elle_tools_v1/eval"
