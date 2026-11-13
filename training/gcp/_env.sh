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
# Which run the scripts work on: its config and its folder name (under runs/
# on the VM and in the bucket). The first run (v1) keeps its own config.
export ELLE_CFG="${ELLE_CFG:-training/gcp/config/gcp_v2.yaml}"
export RUN_NAME="${RUN_NAME:-elle_tools_v2}"
CFG_TAG="$(basename "$ELLE_CFG" .yaml)"

# Pause the Cloud Scheduler job that restarts this VM after Spot stops (if one
# was set up), so a finished or failed run is not booted again every 10 min.
stop_watchdog() {
  gcloud scheduler jobs pause elle-watchdog --location us-central1 --quiet >/dev/null 2>&1 \
    && echo "watchdog paused" || true
}
