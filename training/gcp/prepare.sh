#!/usr/bin/env bash
# On the VM.
#   bash prepare.sh retime   -> data/retimed.jsonl, the ONLY input audio may be rendered from
#   bash prepare.sh mimi     -> Mimi-encode rendered audio from $ELLE_WORK/audio (GPU)
source "$(dirname "$0")/_env.sh"
case "${1:-}" in
  retime)
    python3 training/model/prepare.py --config training/gcp/config/smoke.yaml \
      --episodes "$ELLE_WORK/data/episodes.jsonl" --out "$ELLE_WORK/runs/prepared_zero" \
      --write-retimed "$ELLE_WORK/data/retimed.jsonl"
    echo "Render audio from $ELLE_WORK/data/retimed.jsonl" ;;
  mimi)
    python3 training/model/prepare.py --config training/gcp/config/gcp.yaml
    # The plain-conversation set mixed into training (see training/model/README.md).
    python3 training/model/plain.py synth --config training/gcp/config/gcp.yaml --n-train 2000 --n-val 200 ;;
  *) echo "usage: prepare.sh retime|mimi"; exit 2 ;;
esac
