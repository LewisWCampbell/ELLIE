# Training on Google Cloud

One VM with one A100 80GB (or one H100), in the project Elle already runs in.
A full run is roughly 40 to 50 A100-hours (about 20 on an H100). Checkpoints
go to a Cloud Storage bucket every 10 minutes, so a Spot VM being stopped costs
a restart, not the run. **None of these scripts has been run yet**: this
session cannot sign in to GCP. Read each command before running it.

## The one piece of red tape: GPU quota

New projects usually have a GPU quota of zero, and creating the VM fails until
it is raised. In the console: **IAM & Admin → Quotas**, then request:

- **GPUs (all regions)**: at least 1 (a global cap that is often 0).
- **Preemptible NVIDIA A100 80GB GPUs** in your region, at least 1. Spot VMs
  draw on the *preemptible* quota. For on-demand, request **NVIDIA A100 80GB
  GPUs** instead. For an H100, the H100 equivalents.

Approval ranges from minutes to a couple of days, and a project with little
billing history is sometimes asked for justification. "Fine-tuning a speech
model for research, one GPU, about 50 hours" is enough.

## Cost

Approximate; the console shows the exact hourly price when you create the VM.

| | per hour | full run |
|---|---|---|
| A100 80GB, Spot | roughly $1.50 to $2.50 | roughly $70 to $115 |
| A100 80GB, on-demand | roughly $5 to $6 | roughly $225 to $270 |
| H100, on-demand | roughly $10 to $12 | roughly $200 to $240 |

Add the smoke test and a few short debugging sessions, plus about $50 a month
while a 500 GB SSD boot disk exists. **Delete the VM when you are done**; a
stopped VM still bills for its disk. The training script shuts the VM down
when a run finishes, so it never idles on a GPU.

Speech audio for training (about 700 hours) is the other cost: roughly $400 to
$850 through Deepgram at assumed prices, or GPU time on this same VM if an open
TTS backend is added to `render_audio.py`. Elle's channel should be voiced
with her production voice, so the model learns to sound like her.

## Order of operations

```
laptop  1. Request the GPU quota (above). Wait for approval.
        2. Accept the PersonaPlex licence on huggingface.co; create a read token.
        3. cp training/gcp/gcp.env.example training/gcp/gcp.env; fill it in.
        4. bash training/gcp/create_vm.sh
        5. gcloud compute scp training/gcp/gcp.env elle-train:~ ...; gcloud compute ssh elle-train ...
VM      6. git clone the repo; export HF_TOKEN=...; bash ~/ELLE/training/gcp/vm_setup.sh
        7. bash training/gcp/smoke.sh          expect to fix the real-model backend here
        8. bash training/gcp/prepare.sh retime
anywhere 9. render audio from retimed.jsonl; copy it to ~/elle-work/audio on the VM
VM     10. bash training/gcp/prepare.sh mimi
       11. nohup bash training/gcp/train.sh > ~/train.log 2>&1 &
       12. bash training/gcp/eval.sh
laptop 13. gcloud storage rsync -r gs://<bucket>/runs ./runs   then delete the VM
```

If Spot stops the VM mid-run: `gcloud compute instances start elle-train
--zone ...`. The boot service resumes training from the last checkpoint by
itself.

## What the training data contains

Only synthetic conversations, generated on the VM. No caller audio, no
transcripts and no personal data are uploaded. Keep it that way: production
call logs belong to real people, and moving them into a training bucket is a
separate decision.

## Not built yet

- An open-source TTS backend for `render_audio.py`, if you'd rather not pay an
  API for the audio.

The real-model code (loading, the evaluation loop's model adapter, Mimi
decoding) was written from Moshi and PersonaPlex source and has never run.
The smoke test and the first `eval.sh` are where that gets fixed.
