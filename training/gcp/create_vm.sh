#!/usr/bin/env bash
# Run on YOUR laptop (where gcloud is signed in). Creates the bucket and the
# GPU VM. Nothing here has been run yet: read each command before you do.
#
#   cp gcp.env.example gcp.env && $EDITOR gcp.env
#   bash training/gcp/create_vm.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/gcp.env"
: "${GCP_PROJECT:?set GCP_PROJECT in gcp.env}"
REGION="${GCP_ZONE%-*}"

gcloud services enable compute.googleapis.com storage.googleapis.com --project "$GCP_PROJECT"

if ! gcloud storage buckets describe "$BUCKET" --project "$GCP_PROJECT" >/dev/null 2>&1; then
  gcloud storage buckets create "$BUCKET" --project "$GCP_PROJECT" --location "$REGION" --uniform-bucket-level-access
fi

if [ -z "$IMAGE_FAMILY" ]; then
  IMAGE_FAMILY="$(gcloud compute images list --project deeplearning-platform-release \
    --filter='family~^common-cu12 AND family~debian' --sort-by='~creationTimestamp' \
    --limit=1 --format='value(family)')"
  echo "Using Deep Learning VM image family: $IMAGE_FAMILY"
fi

SPOT_FLAGS=()
if [ "$SPOT" = "1" ]; then
  SPOT_FLAGS=(--provisioning-model=SPOT --instance-termination-action=STOP)
fi

gcloud compute instances create "$VM_NAME" \
  --project "$GCP_PROJECT" --zone "$GCP_ZONE" \
  --machine-type "$MACHINE_TYPE" \
  --image-project deeplearning-platform-release --image-family "$IMAGE_FAMILY" \
  --boot-disk-size "${BOOT_DISK_GB}GB" --boot-disk-type pd-ssd \
  --maintenance-policy TERMINATE \
  --scopes cloud-platform \
  --metadata install-nvidia-driver=True \
  "${SPOT_FLAGS[@]}"

echo
echo "Created. Next:"
echo "  gcloud compute scp training/gcp/gcp.env $VM_NAME:~ --zone $GCP_ZONE --project $GCP_PROJECT"
echo "  gcloud compute ssh $VM_NAME --zone $GCP_ZONE --project $GCP_PROJECT"
echo "  then on the VM:  git clone $REPO_URL ~/ELLE && export HF_TOKEN=... && bash ~/ELLE/training/gcp/vm_setup.sh"
