#!/usr/bin/env bash
# End-to-end CPU dry run: prepare -> train (few steps) -> evaluate (reference + model).
# Uses the dataset sample (training/data/out/sample_*.jsonl) when present,
# otherwise the hand-built fixtures.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SAMPLE="$HERE/../data/out"
if ls "$SAMPLE"/sample_train.jsonl >/dev/null 2>&1; then
  CFG="$HERE/config/dryrun_sample.yaml"
  mkdir -p "$HERE/runs/dryrun_sample"
  cat "$SAMPLE"/sample_train.jsonl "$SAMPLE"/sample_val.jsonl "$SAMPLE"/sample_test.jsonl > "$HERE/runs/dryrun_sample/sample_all.jsonl"
else
  CFG="$HERE/config/dryrun.yaml"
fi
echo "== config: $CFG"
python3 "$HERE/prepare.py" --config "$CFG" --rebuild-vocab
python3 "$HERE/plain.py" synth --config "$CFG" --n-train 60 --n-val 12
python3 "$HERE/evaluate.py" --config "$CFG" --mode reference --split all
python3 "$HERE/train.py" --config "$CFG"
python3 "$HERE/evaluate.py" --config "$CFG" --mode model --listen 2
