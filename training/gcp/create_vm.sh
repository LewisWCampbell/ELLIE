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

gcloud services enable compute.googleapis.com storage.googleapis.com iam.googleapis.com --project "$GCP_PROJECT"

if ! gcloud storage buckets describe "$BUCKET" --project "$GCP_PROJECT" >/dev/null 2>&1; then
  gcloud storage buckets create "$BUCKET" --project "$GCP_PROJECT" --location "$REGION" --uniform-bucket-level-access \
    --default-storage-class STANDARD
  gcloud storage buckets update "$BUCKET" --update-labels purpose=elle-training
fi

# The VM gets its own identity, allowed to read and write the training bucket
# and nothing else. Without this it would run as the project's default compute
# account, which is the account Elle's live service runs as, with access to
# every one of her secrets, on a machine that runs downloaded model code.
SA_NAME="elle-train"
SA_EMAIL="$SA_NAME@$GCP_PROJECT.iam.gserviceaccount.com"
if ! gcloud iam service-accounts describe "$SA_EMAIL" --project "$GCP_PROJECT" >/dev/null 2>&1; then
  gcloud iam service-accounts create "$SA_NAME" --project "$GCP_PROJECT" \
    --display-name "Elle training VM (training bucket only)"
fi
gcloud storage buckets add-iam-policy-binding "$BUCKET" \
  --member "serviceAccount:$SA_EMAIL" --role roles/storage.objectAdmin >/dev/null
# Lets the VM's logs and GPU metrics reach Cloud Logging/Monitoring; reads nothing.
for role in roles/logging.logWriter roles/monitoring.metricWriter; do
  gcloud projects add-iam-policy-binding "$GCP_PROJECT" \
    --member "serviceAccount:$SA_EMAIL" --role "$role" --condition=None >/dev/null
done

if [ -z "$IMAGE_FAMILY" ]; then
  # Newest CUDA 12 "common" Deep Learning VM image, whatever OS Google currently
  # builds it on (it moved from Debian to Ubuntu, which broke an OS filter here).
  IMAGE_FAMILY="$(gcloud compute images list --project deeplearning-platform-release \
    --filter='family~^common-cu12' --sort-by='~creationTimestamp' \
    --limit=1 --format='value(family)')"
fi
if [ -z "$IMAGE_FAMILY" ]; then
  echo "No Deep Learning VM image found. List them with:" >&2
  echo "  gcloud compute images list --project deeplearning-platform-release --filter='family~^common-cu' --format='value(family)' | sort -u" >&2
  echo "and set IMAGE_FAMILY in gcp.env to one of them." >&2
  exit 1
fi
echo "Using Deep Learning VM image family: $IMAGE_FAMILY"

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
  --service-account "$SA_EMAIL" --scopes cloud-platform \
  --metadata install-nvidia-driver=True \
  --labels purpose=elle-training \
  "${SPOT_FLAGS[@]}"

echo
echo "Created. Next:"
echo "  gcloud compute scp training/gcp/gcp.env $VM_NAME:~ --zone $GCP_ZONE --project $GCP_PROJECT"
echo "  gcloud compute ssh $VM_NAME --zone $GCP_ZONE --project $GCP_PROJECT"
echo "  then on the VM:  git clone $REPO_URL ~/ELLE && export HF_TOKEN=... && bash ~/ELLE/training/gcp/vm_setup.sh"
echo
echo "Costs: everything here carries the label purpose=elle-training. A budget alert"
echo "(Billing > Budgets & alerts) filtered on that label emails you before a surprise."
