#!/usr/bin/env bash
# On the VM. The real 7B model, 8 episodes, silent audio, two steps. Expect it
# to fail the first time: the real-model backend has never run. Fix it here,
# where a failure costs minutes, before starting the long run.
source "$(dirname "$0")/_env.sh"
python3 training/model/prepare.py --config training/gcp/config/smoke.yaml --limit 8
python3 training/model/train.py --config training/gcp/config/smoke.yaml --max-steps 2
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
echo "SMOKE OK: the real model loaded, resized its vocab, and took two steps."
