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

# Pause the Cloud Scheduler job that restarts this VM after Spot stops (if one
# was set up), so a finished or failed run is not booted again every 10 min.
stop_watchdog() {
  gcloud scheduler jobs pause elle-watchdog --location us-central1 --quiet >/dev/null 2>&1 \
    && echo "watchdog paused" || true
}
