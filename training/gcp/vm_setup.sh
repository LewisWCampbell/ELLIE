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

# Check the token before spending 20 minutes on installs: is it a valid token,
# and does it unlock the gated PersonaPlex weights? (Status codes only; the
# token is never printed.)
hf_status() { curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $HF_TOKEN" "$1"; }
if [ "$(hf_status https://huggingface.co/api/whoami-v2)" != "200" ]; then
  echo "HF_TOKEN is not a valid Hugging Face token (it may have been pasted into the wrong prompt)." >&2
  echo "Run the read line on its own, paste the token at the prompt, then rerun this script." >&2
  exit 1
fi
case "$(hf_status https://huggingface.co/nvidia/personaplex-7b-v1/resolve/main/.gitattributes)" in
  200|302|307) ;;
  *) echo "The token is valid but cannot download nvidia/personaplex-7b-v1." >&2
     echo "Accept the licence on that model's page, signed in as the token's owner." >&2
     exit 1 ;;
esac
echo "Hugging Face token OK, PersonaPlex access OK."

export ELLE_WORK="$HOME/elle-work"
REPO="$HOME/ELLE"
mkdir -p "$ELLE_WORK"

[ -d "$REPO" ] || git clone "$REPO_URL" "$REPO"
cp "$HOME/gcp.env" "$REPO/training/gcp/gcp.env"

nvidia-smi --query-gpu=name,memory.total --format=csv
# Ubuntu packages, all up front: venv support (the Deep Learning VM image ships
# Python without it), Python's C headers (PersonaPlex compiles layers with
# torch.compile, whose Triton backend builds a C extension and fails without
# them), Opus for Moshi's audio, espeak-ng for Kokoro's Spanish.
sudo apt-get update -q
sudo apt-get install -y python3-venv python3-dev libopus-dev espeak-ng
python3 -m venv "$ELLE_WORK/venv"
source "$ELLE_WORK/venv/bin/activate"
pip install --upgrade pip
# PersonaPlex first: it pins the PyTorch version it needs (below 2.5). Installed
# second, it made pip download a newer PyTorch and then replace it, ~3 GB wasted.
[ -d "$ELLE_WORK/personaplex" ] || git clone https://github.com/NVIDIA/personaplex "$ELLE_WORK/personaplex"
pip install "$ELLE_WORK/personaplex/moshi/."
pip install -r "$REPO/training/model/requirements.txt"

export HF_HOME="$ELLE_WORK/hf"
python3 - <<'PY'
from huggingface_hub import snapshot_download
for repo in ("nvidia/personaplex-7b-v1", "kyutai/moshiko-pytorch-bf16", "hexgrad/Kokoro-82M"):
    print("downloading", repo, "->", snapshot_download(repo))
PY

# The open speech model (Kokoro) that voices the training audio, in its own
# venv: its dependencies can upgrade packages the training venv relies on.
python3 -m venv "$ELLE_WORK/tts-venv"
"$ELLE_WORK/tts-venv/bin/pip" install --upgrade pip
"$ELLE_WORK/tts-venv/bin/pip" install -r "$REPO/training/data/requirements.txt" "kokoro==0.9.4" "misaki[en]==0.9.4"
"$ELLE_WORK/tts-venv/bin/python" -m spacy download en_core_web_sm

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
