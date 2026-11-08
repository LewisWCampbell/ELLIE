# Sourced by the VM-side scripts.
set -euo pipefail
source "$HOME/gcp.env"
export ELLE_WORK="$HOME/elle-work"
export HF_HOME="$ELLE_WORK/hf"
export HF_HUB_OFFLINE=1
REPO="$HOME/ELLE"
source "$ELLE_WORK/venv/bin/activate"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
