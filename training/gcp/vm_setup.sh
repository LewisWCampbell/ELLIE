#!/usr/bin/env bash
# Run ONCE on the VM. Clones the repo, builds the Python environment,
# installs PersonaPlex's moshi package, downloads the weights, generates the
# dataset, and installs a service that resumes training whenever the VM boots
# (so a Spot preemption costs a restart, not the run).
#
#   export HF_TOKEN=...   # after accepting the PersonaPlex licence on huggingface.co
#   bash vm_setup.sh
set -euo pipefail
source "$HOME/gcp.env"
: "${HF_TOKEN:?export HF_TOKEN first}"
export ELLE_WORK="$HOME/elle-work"
REPO="$HOME/ELLE"
mkdir -p "$ELLE_WORK"

[ -d "$REPO" ] || git clone "$REPO_URL" "$REPO"
cp "$HOME/gcp.env" "$REPO/training/gcp/gcp.env"

nvidia-smi --query-gpu=name,memory.total --format=csv
python3 -m venv "$ELLE_WORK/venv"
source "$ELLE_WORK/venv/bin/activate"
pip install --upgrade pip
pip install -r "$REPO/training/model/requirements.txt"
sudo apt-get install -y libopus-dev
[ -d "$ELLE_WORK/personaplex" ] || git clone https://github.com/NVIDIA/personaplex "$ELLE_WORK/personaplex"
pip install "$ELLE_WORK/personaplex/moshi/."

export HF_HOME="$ELLE_WORK/hf"
python3 - <<'PY'
from huggingface_hub import snapshot_download
for repo in ("nvidia/personaplex-7b-v1", "kyutai/moshiko-pytorch-bf16"):
    print("downloading", repo, "->", snapshot_download(repo))
PY

python3 "$REPO/training/data/generate.py" --out "$ELLE_WORK/data" --prefix full_ \
  --train 20000 --val 1000 --test 1000 --seed 7 --holdout-family slot_conflict
python3 "$REPO/training/data/validate.py" "$ELLE_WORK"/data/full_*.jsonl
cat "$ELLE_WORK"/data/full_{train,val,test}.jsonl > "$ELLE_WORK/data/episodes.jsonl"

# The resume-on-boot service. It does nothing until train.sh creates the
# "training" marker, so booting the VM for other work never starts a run.
sudo tee /etc/systemd/system/elle-train.service >/dev/null <<UNIT
[Unit]
Description=Resume Elle tool-call training after a reboot or Spot preemption
After=network-online.target
ConditionPathExists=$ELLE_WORK/.training

[Service]
User=$USER
ExecStart=/bin/bash $REPO/training/gcp/train.sh --resumed
Restart=no

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable elle-train.service
echo "Setup done. Next: bash $REPO/training/gcp/smoke.sh"
