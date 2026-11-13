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

Add the smoke test and a few short debugging sessions, plus roughly $50
a month while the 490 GB boot disk exists (balanced persistent disk). **Delete the VM when you are
done**; a stopped VM still bills for its disk. The training script shuts the VM
down when a run finishes, so it never idles on a GPU.

The training speech (about 700 hours of stereo audio) is generated on this same
VM by Kokoro, an open, Apache-licensed speech model: no API, no key, and no
conversation text leaving the project. It is estimated at 1 to 3 hours of GPU
time with several workers; `prepare.sh check` prints the real speed. Elle's
training voice is a Kokoro voice, not the Deepgram voice production uses today;
the duplex model generates its own voice, so what matters is that this voice
matches the voice prompt she is given at run time.

## Order of operations

```
laptop  1. Request the GPU quota (above). Wait for approval.
        2. Accept the PersonaPlex licence on huggingface.co; create a read token.
        3. cp training/gcp/gcp.env.example training/gcp/gcp.env; fill it in.
        4. bash training/gcp/create_vm.sh
        5. gcloud compute scp training/gcp/gcp.env elle-train:~ ...; gcloud compute ssh elle-train ...
VM      6. git clone the repo; export HF_TOKEN=...; bash ~/ELLE/training/gcp/vm_setup.sh
        7. bash training/gcp/smoke.sh            expect to fix the real-model backend here
        8. bash training/gcp/prepare.sh check    one sample per voice; listen to them
        9. bash training/gcp/prepare.sh synth    speak every line once (try LIMIT=50 first)
       10. bash training/gcp/prepare.sh retime
       11. bash training/gcp/prepare.sh assemble listen to a few files in ~/elle-work/audio
       12. bash training/gcp/prepare.sh mimi
       13. nohup bash training/gcp/train.sh > ~/train.log 2>&1 &
       14. bash training/gcp/eval.sh
laptop 15. gcloud storage rsync -r gs://<bucket>/runs ./runs   then delete the VM
```

If Spot stops the VM mid-run: `gcloud compute instances start elle-train
--zone ...`. The boot service resumes **training** from the last checkpoint by
itself. The audio steps (9 to 11) resume too, but not on their own: start the
VM and rerun the same command.

## The second run (v2)

The first run's checkpoint made zero tool calls in evaluation. That came from
the evaluation feeding the model different audio than training during the
system prompt, now fixed and covered by `test_personaplex`.
`diagnose.sh` re-evaluates that checkpoint. v2 also fixes what only a retrain
can: the prompt has the date (and Lewis's name), PersonaPlex's own prompt
layout, calls placed right after the line that announces them, float32 rows for
the new tokens, `<tool>` weighted on its own, and no unlikelihood term. Same
audio as v1; only the text stream is re-encoded (`gcp_v2.yaml`). Order:

```
VM  1. git pull
    2. tmux new -d -s pilot 'bash ~/ELLE/training/gcp/pilot.sh'
         encodes v2 (~1 h), then 200 steps on 32 calls at two learning rates,
         each followed by markers.py and a closed-loop check; uploads
         logs/pilot_*.log and shuts down
    3. read the log. Go if a learning rate reaches <tool> accuracy >= 0.95 and
       >= 80% of the expected calls in closed loop with no parse errors; put
       that rate in base_gpu.yaml (train.optim.lr)
    4. tmux new -d -s elle 'bash ~/ELLE/training/gcp/run_all.sh'
         skips the finished steps, trains v2 and evaluates it; the restart
         job brings it back after Spot stops
```

## Same project as Elle, separate identity

Training runs in Elle's own project, where the GPU quota was granted, but the
VM never runs as the project's default compute account. That account is the
one Elle's live service uses, with access to all of her secrets, and this VM
runs downloaded model code. `create_vm.sh` creates an `elle-train` service
account that can read and write the training bucket and write logs, nothing
else, and labels the VM and bucket `purpose=elle-training` so costs can be
budgeted and filtered separately.

## What the training data contains

Only synthetic conversations, generated on the VM. No caller audio, no
transcripts and no personal data are uploaded. Keep it that way: production
call logs belong to real people, and moving them into a training bucket is a
separate decision.

## Untested until the VM

Kokoro's voices, real clip lengths and speed are checked by `prepare.sh check`,
which needs the model weights and so could not run where this was written. The real-model code (loading, the evaluation loop's model adapter, Mimi
decoding) was written from Moshi and PersonaPlex source and has never run.
The smoke test and the first `eval.sh` are where that gets fixed.
