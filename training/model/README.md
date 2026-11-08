# training/model: teaching a full-duplex voice model to call tools

This directory is the trainer side of [`../CONTRACT.md`](../CONTRACT.md). It
fine-tunes a Moshi-family LM (NVIDIA PersonaPlex is the candidate; Kyutai's
Moshiko is the fallback) so that Elle writes tool calls in her inner-monologue
text stream, waits naturally, and resumes once the result is injected into
that stream.

Everything here runs on a CPU box with a tiny stand-in model, so the real code
paths (data prep, masking, LoRA, new tokens, checkpoints, closed-loop
generation with result injection, every metric) are tested. The real 7B model
sits behind `moshi_backend.py`, which is **UNVERIFIED**: it was written from
source code, never run (see [Verified and unverified](#verified-and-unverified)).

```
python3 -m unittest discover training/model          # 43 tests, about 30 s on 4 CPUs
training/model/run_dryrun.sh                          # prepare -> plain set -> reference eval -> train -> model eval
```

## Files

| file | what it does |
|---|---|
| `config/base_gpu.yaml` | the real run: PersonaPlex, LoRA r64, 100 s windows, lr schedule, eval gates |
| `config/dryrun.yaml` | CPU dry run on `fixtures/` (TinyDuplexLM, chunk tokenizer, placeholder audio) |
| `config/dryrun_sample.yaml` | CPU dry run on the dataset sample (`training/data/out/sample_*.jsonl`) |
| `prepare.py` | episodes -> retimed episodes -> frame-aligned text stream + loss mask + audio codes |
| `serializer_adapter.py` | uses `training/data/serialize.py` (`serialize_episode`), checks it against the contract |
| `textvocab.py` | control ids, the four markers, tokenizers (byte, chunk for dry runs, SentencePiece for real) |
| `tiny_model.py` | TinyDuplexLM (temporal + depth transformer, Moshi's interface), LoRA, vocab extension |
| `moshi_backend.py` | real Moshi / PersonaPlex: training adapter, checkpoints, inference loader, `MoshiFrameGenerator` over LMGen (all UNVERIFIED) |
| `train.py` | training loop: masked loss, unlikelihood on negatives, tool/plain batch mixing, forgetting check, checkpoints, resume, seeding |
| `plain.py` | builds the plain-conversation set: `synth` (tool-free calls from training/data's builder) or `external` (moshi-finetune format, UNVERIFIED) |
| `genloop.py` | the closed-loop evaluation loop shared by both backends, the `FrameGenerator` interface, the tiny generator, window stitching |
| `metrics.py` | the five contract metrics over events |
| `runtime.py` | frame-level tool-span parser, result injector, stream-to-events decoder |
| `evaluate.py` | reference mode (tests the metrics on the dataset) and model mode (closed loop, either backend, `--listen`) |
| `infer_runtime.md` | inference server design and the WebSocket protocol for a Node `RealtimeModel` |
| `fixtures/` | 11 hand-built episodes (6 positives, 5 negatives) and their generator |
| `test_*.py` | tests |

## Approach

* **LoRA on the LM**, base frozen. Moshi-finetune (Kyutai's official recipe)
  fine-tunes the same way: LoRA on every linear layer of the temporal (7B,
  32 layers, d=4096) and depth (6 layers) transformers, rank <= 128, lr about
  2e-6 to 4e-6 with a OneCycle schedule. We use rank 64 for a narrow skill.
* **Four new text tokens**: `<tool>`, `</tool>`, `<result id=`, `</result>`
  are appended after the 32,000 SentencePiece ids, so `text_card` becomes
  32,004 and the initial-token row (`text_card` in Moshi) moves to the end. The
  new rows of `text_emb`, `depformer_text_emb` and the `text_linear` head are
  trained (gradient-masked to just those rows, higher lr, no weight decay);
  every pretrained row stays fixed. A new row starts as the mean embedding of
  its marker's literal pieces; its head bias starts low, so the model's
  behaviour at step 0 is unchanged (tested).
* **Calls are predicted, results are forced.** On Elle's text stream the
  `<tool>{json}</tool>` span is a training target at the call's frame. The
  `<result id="c1">…</result>` span is teacher-forced at the result's frame
  with its loss masked, and so is the EPAD frame that announces it (the model
  cannot know when a result will arrive). At inference the result markers are
  banned from sampling and only ever injected.
* **Audio**: Mimi codes, 8 codebooks per channel at 12.5 Hz. Channel 0 is Elle
  (predicted, semantic codebook weighted x100 as in moshi-finetune), channel 1
  the caller (input only). Text loss down-weights PAD/EPAD (0.5, as
  moshi-finetune) and up-weights marker targets (2.0).
* **Negatives** (`labels.negative`) are never trained as targets. By default
  their bad frames (the premature claim's words, the unsafe write's `<tool>`)
  get an unlikelihood loss, `-log(1 - p)`, and their other frames get the
  ordinary loss. The bad frames are found with the same code the evaluator uses.

## Data flow

```
episodes.jsonl (CONTRACT.md, from training/data)
   │  prepare.retime_episode     make it physically possible at 12.5 tokens/s (see below)
   ▼
retimed episodes ──────────────► training/data/render_audio.py (optional TTS) ──► stereo 24 kHz WAV
   │                                                                               + alignment JSON
   │  training/data/serialize.serialize_episode  (via serializer_adapter)              │
   ▼                                                                                   ▼
Elle text segments (speech / tool_call / tool_result, is_forced) ───► frame placement ◄─ word timings
   │                                                                     │
   │                               Mimi.encode both channels (GPU) ──────┤  or placeholder codes
   ▼                                                                     ▼
example.npz: text[T], loss_mask[T], neg_mask[T], audio[16,T], audio_mask[T]   (T frames at 12.5 Hz)
```

**Frame placement** (`prepare.place_items`) follows Moshi's interleaver: a word
starts at the frame containing its start time, one token per frame, EPAD on
the frame before a word, PAD elsewhere. Unlike Moshi's default, nothing is ever
dropped (tokens queue, like `keep_and_shift=True`), and `<tool>`/`<result>`
spans are atomic: before a span starts, the rest of the utterance in progress
is flushed, so text runs a little *ahead* of its audio (as Moshi's text
normally does) rather than behind it. On the sample, the worst word lag per
example is 1 frame at the median and 3 at p90; 5% of examples have a word more
than 6 frames late (up to 42 frames, 3.4 s), always an utterance that *starts*
while a span is being written. That is the bandwidth problem below, made
visible.

**Retiming** (`prepare.retime_episode`) is the one substantive addition. The
contract's events treat a call and a result as instantaneous, but at one token
per 80 ms frame a call span takes about 4 s to write and a result 2 to 8 s to
inject. Taken literally, a result would arrive before its call finished and
Elle's *audio* would state the answer while the result was still being
injected, which is exactly the premature-claim behaviour the data must not
teach. Retiming moves each result to no earlier than call-span end plus the
tool's latency, shifts everything after a result to after its injection, and
moves a call that comes just before an Elle utterance to just after it starts.
Indices and labels are unchanged; all five metrics give identical verdicts
before and after (tested). On the sample it lengthens the data from 43.6 h to
58.4 h. Audio must be rendered from the retimed episodes: `prepare.py
--write-retimed out.jsonl`, then `render_audio.py out.jsonl …`; the `mimi`
audio mode refuses non-retimed audio unless told otherwise.

**Alignment JSON**: the format `render_audio.py` writes is accepted:
`{"alignments": [[word, [start, end], "SPEAKER_MAIN"], …], "tool_spans": […]}`
(moshi-finetune's format plus a span list). Word timings come from
`alignments`. Span text and times come from the episode itself, so
`tool_spans` is informational.

**PersonaPlex system prompt**: `prepare.prompt_text` puts a pinned
`<system> … <system>` text prompt at the start of every window, forced and
loss-masked, with no audio (PersonaPlex also has a voice prompt; how it was
trained with one is UNVERIFIED, so the voice prompt is applied at inference only).

## Runtime inference (summary; details in `infer_runtime.md`)

Each 80 ms frame: encode the caller's audio with Mimi, step the LM, and get one
text token plus 8 Elle codes. The text token is either sampled, with
`<result id=` and `</result>` banned, or forced by the injector. A
`ToolStreamParser` watches the stream; on `</tool>` it parses the JSON and the
server sends a `tool.call` message to the Node service, which runs the
existing `handleFunctionCall` and replies with `tool.result`. The injector then
forces `EPAD <result id="c1">…</result>` at the next word boundary. **Audio
never pauses**: the model was trained to keep talking through the call span
and the wait, which the dead-air metric checks. A fourth `RealtimeModel`
implementation in Node speaks a small JSON WebSocket protocol, specified in
`infer_runtime.md`.

## Model-mode evaluation (both backends)

`evaluate.py --mode model` runs `genloop.closed_loop` over whole episodes
(windows of long episodes are stitched back together). Each 80 ms frame:

1. the caller's Mimi codes for that frame come from the prepared example
   (rows 8-15), so the caller is replayed;
2. the text token is forced on prompt frames and while a result is being
   injected; otherwise the backend samples it with `<result id=`/`</result>`
   (and `<tool>` inside a span) banned;
3. a completed `</tool>` is parsed and answered by an oracle from the
   episode's own scripted result, ready `latency_s × 12.5` frames later;
   `runtime.Injector` forces `EPAD <result id="c1">…</result>` at the next
   word boundary (at most `eval.gen.max_wait_frames` late);
4. the emitted stream is decoded into events and scored by `metrics.py`.

A backend only implements `FrameGenerator.start/step` (and optionally
`decode_audio`). `TinyFrameGenerator` (CPU) is tested, the loop is tested with
a scripted generator (injection timing, EPAD + span, bans, caller codes,
premature claims caught, a backend that ignores forcing is an error). The real
adapter, `moshi_backend.MoshiFrameGenerator`, is UNVERIFIED. It was written
from LMGen's source: with PersonaPlex's fork it forces through
`LMGen.step(input_tokens=…, text_token=…)`, bans through a subclass that edits
the logits in `process_transformer_output`, and reads the emitted token back
from the streaming cache; with Kyutai's `LMGen` it forces and bans through
`on_text_logits_hook` and reads `on_text_hook`. Settings: `eval.gen`
(`max_wait_frames`, `temperature_text` (0 = near-greedy), `top_k_text`,
`temperature_audio`, `top_k_audio`, `voice_prompt`).

`--listen N` (or `eval.listen: N`) decodes Elle's generated codes with Mimi
and writes `run_dir/eval/listen/<episode>.wav` for the first N episodes. On
the CPU/tiny backend it is a no-op and the report says so.

The report also gives the number of calls made and injected, the frames from
"result ready" to injection, and the plain-conversation forgetting line (below).

## Plain-conversation mixing and forgetting

Set `data.plain_prepared_dir` and `data.plain_fraction` (0.3 in
`base_gpu.yaml`). Each batch then draws exactly that share of examples from
the plain set over time (a carried remainder: 0.3 × 4 = 1, 1, 2, 1, 1, 2, …),
from seeded RNGs whose state is saved in every checkpoint, so `--resume`
continues the same mixture. Tool and plain examples go through the model
separately; the step loss weights them by count, and the log has
`tool_loss`/`tool_text_loss`/`plain_loss`/`plain_text_loss` per step, plus
`plain_examples_seen` and `tool_examples_seen`.

Forgetting: before training (and on resume, from the checkpoint), the base
model's text loss on the held-out plain split (`data.plain_eval_split`) is
measured with LoRA switched off. At every eval, the current loss and the
delta are logged (`eval_plain_text_loss*`) and written to
`run_dir/plain_forgetting.json`, gated by `eval.gates.plain_text_loss_delta`
(0.05). The model report includes the line. The new tokens' trained rows stay
in place for the baseline; plain text never uses them.

Building the plain set (run `prepare.py` first, so the tokenizer exists):

```
python3 training/model/plain.py synth --config <cfg> --n-train 2000 --n-val 200
python3 training/model/plain.py external --config <cfg> --jsonl /data/dailytalk/dailytalk.jsonl   # UNVERIFIED
```

* `synth` builds tool-free calls (greeting, who is calling, small talk about
  Lewis, goodbye) with training/data's `Ep` builder and phrase banks
  (read-only use of its library); they pass its validator. This is a weak
  stand-in that shares the tool data's templates. It guards the text stream's
  general behaviour, not real conversational quality.
* `external` reads moshi-finetune's format (`{"path","duration"}` JSONL,
  stereo 24 kHz WAV with the model's speaker on the left, `x.json`
  alignments); the main speaker's words (`SPEAKER_MAIN`) become the text
  stream, and both channels are Mimi-encoded. Written from moshi-finetune's
  README and interleaver, never run. kyutai/DailyTalkContiguous is the
  intended source (14 GB on Hugging Face). Its speaker labels and whether its
  left channel is the "main" speaker were not checked.

## Metric definitions

All five are computed by `metrics.py` over events. Model output is first
decoded into events by `runtime.decode_text_stream`. Tool selection and
argument metrics count positives only, because the dataset's negatives list
the calls actually made.

1. **Tool selection accuracy**: per episode, the predicted sequence of tool
   names equals `expected_calls` (empty = correctly no tool). Call precision
   and recall come from an order-preserving name alignment.
2. **Argument exact match** over name-aligned calls: every key equal; `start`
   is parsed and compared as an instant (`18:00Z` == `10:00-08:00`; no offset
   counts as wrong); absent and blank optional args are equal.
3. **Premature claim rate** = premature claims / claims. Reference mode uses
   `claim_spans`. For model output a claim detector (`looks_like_claim`, regex
   over English and Spanish, hypotheticals stripped) finds claims, and each
   depends on the latest non-silent call before it. A claim before that
   call's result, or with no call at all, is premature.
4. **Write unsafe rate** = unsafe writes / writes. `book_intro_call` needs
   name, a valid email and an ISO `start` with offset, an Elle readback
   (time + name or full email) before the call, a caller yes after the
   readback, and no objection after the yes. `save_user_fact(sms_consent)`
   needs an SMS offer and a yes; other facts are ungated. Reference mode uses
   `labels.write_gates`. Also reported: `silent_write_narration_rate`, the
   share of ungated `save_user_fact` calls Elle narrates.
5. **Dead-air rate**: among reads with latency > 0.7 s, the share where Elle's
   speech covers less than 0.3 s of the wait (call to result). The 0.3 s, not
   0, absorbs the timing error of text-derived speech; for audio models, use
   VAD on the generated channel. Also reported: `long_silence_in_wait_rate`
   (> 2 s continuous).

**The metrics are tested on the dataset itself** (`evaluate.py --mode
reference`, run by the tests on val+test and by `run_dryrun.sh` on all 2,400
sample episodes). On the sample:

* all 2,240 positives score perfectly;
* all 160 negatives are caught, each by the metric matching its
  `negative_kind`;
* the claim detector has precision 0.995 and recall 0.992 against `claim_spans`;
* the readback heuristic agrees 100% with `write_gates`;
* the round trip (prepared training stream -> decoded events -> detector,
  i.e. the path a model's output takes) is also perfect on positives, with
  160/160 negatives caught.

## Dry-run results (this machine, 4 CPUs, no GPU)

* **Tests**: 43 pass in about 30 s. They cover token handling (markers are
  single ids, round-trip; resizing keeps old logits identical; the gradient
  reaches only the new rows), frame alignment (Moshi's placement rule; EPAD;
  spans contiguous, tool predicted, result plus its EPAD masked; windows never
  split a span; retiming is physical and metric-invariant), training (loss
  falls on a tiny overfit set; forced-frame targets do not affect the loss;
  checkpoint round-trip; bit-identical reruns; resume reproduces the run) and
  the evaluator (hand-built cases for every metric, the parser and injector,
  and reference mode on the fixtures and on the sample), the shared
  closed-loop loop (scripted generator: injection timing and content, bans,
  caller codes, prompt frames, premature claims, a backend ignoring forcing),
  window stitching, the WAV writer and the `--listen` no-op, and mixing (exact
  ratio, determinism, resumed runs reproduce the per-step tool and plain
  losses, separate loss logging, the forgetting baseline and file, and
  `model_report` end to end on the tiny backend).
* **Fixtures** (`config/dryrun.yaml`: 300 base "pretraining" steps on tool
  data with the markers stripped plus 60 plain calls, then 300 LoRA + new-row
  steps on 11 episodes with 30% plain mixing, about 75 s in total): eval text
  loss reaches 0.03 and marker accuracy 1.0. In closed loop (`elle_audio:
  teacher`), 8 calls were made and all 8 results injected, each one frame (the
  EPAD) after it was ready. Tool selection was 0.83 (5 of 6 positives), and
  the other five gated metrics were perfect. Held-out plain text loss rose
  from 2.34 (base) to 2.42, so the forgetting gate (+0.05) **fails**, which is
  what a 300-step overfit on 11 episodes should do; the check sees it. Before
  plain mixing was added, the same run scored 6 of 6. This is an overfit and
  only checks the plumbing.
* **Dataset sample** (`config/dryrun_sample.yaml`, `run_dryrun.sh`, about
  90 s in total): prepare 2,400 episodes -> 2,421 examples, 2.66 M frames;
  60 + 12 plain calls (0 validator issues); reference eval; 20 training steps
  with 30% plain (text loss 3.5 to 3.1; plain loss 2.84 -> 2.78); closed-loop
  model eval on 4 val episodes with `--listen 2`, which is a no-op. A 20-step
  model calls nothing yet; the run shows the full pipeline works on real
  episode shapes.

`elle_audio: teacher` replays Elle's own (placeholder) codes during closed-loop
generation, because placeholder audio is noise that cannot be generated. It
leaks timing, so it is an upper bound and only for dry runs. Real evaluation
uses `generate`.

## Running it for real (GPU box)

1. Environment: CUDA, Python 3.10+, `pip install -r requirements.txt` plus
   **one** `moshi` package (PersonaPlex's fork for PersonaPlex; the two
   packages share a name). Accept the PersonaPlex licence on Hugging Face and
   set `HF_TOKEN`.
2. Generate a full dataset (`training/data/generate.py`, e.g. 20k+ episodes),
   then `python3 training/model/prepare.py --config config/base_gpu.yaml
   --write-retimed retimed.jsonl` with `prepare.audio.mode: zero`, to check the
   stats and get retimed episodes.
3. Render audio from `retimed.jsonl` (`training/data/render_audio.py`, needs
   TTS credentials), then set `prepare.audio.mode: mimi`, `audio_dir`,
   `audio_is_retimed: true` and prepare again (Mimi encodes on GPU).
4. Build the plain-conversation set with `plain.py synth` (or `external` for
   DailyTalkContiguous, UNVERIFIED) and set `data.plain_prepared_dir`. See
   "Plain-conversation mixing and forgetting".
5. Smoke test the backend first: in a Python shell, run
   `moshi_backend.load_real_model(cfg)` and one forward/backward on a single
   example. Fix what breaks (UNVERIFIED: attribute names, the LoRA path for
   PersonaPlex, the dep_q=16 logits, row-mask hooks under FSDP; use a single
   GPU or DDP, not FSDP, at first).
6. `torchrun --nproc-per-node 1 training/model/train.py --config
   training/model/config/base_gpu.yaml`. `train.py` itself does not yet set
   up DDP; single GPU is the tested shape.
7. Evaluate: `evaluate.py --config … --mode model --split test --listen 5`
   runs the real `MoshiFrameGenerator` (UNVERIFIED; expect to fix attribute
   names on the first run, as with training). It needs Mimi-encoded caller
   audio in the prepared examples; zero-audio examples feed Mimi silence. Gate
   on `eval.gates` (including `plain_text_loss_delta`) and listen to
   `eval/listen/*.wav`.

## Hardware and cost estimate

Assumptions (state them, then check them on the first run):

* **Throughput**: moshi-finetune's README reports about 12k tokens/s on
  1×H100 (LoRA r128, batch 16 × 100 s, gradient checkpointing, 39.6 GB peak),
  counting 9 tokens per 12.5 Hz frame. That is 12,000 / (9 × 12.5) ≈ 107 s of
  audio per second, or **~107 h of audio per H100-hour**. An A100 does roughly
  0.4 to 0.5× that (assumed, not measured).
* **Data**: 20k episodes × ~88 s retimed (the sample's mean) ≈ 490 h, plus 30%
  plain conversation ≈ 700 h per epoch. Each hour is 45,000 frames
  (≈ 405k tokens at 9 per frame).
* **Schedule**: 3 epochs ≈ 2,100 h of audio, so 2,100 / 107 ≈ **20 H100-hours**
  (≈ 2.5 h on 8×H100, ≈ 20 h on 1×H100, ≈ 40 to 50 h on 1×A100-80GB).
  moshi-finetune's quick-start recipe (2,000 steps × 16 × 100 s ≈ 890 h) is
  about 8 H100-hours.
* **Memory**: bf16 weights ≈ 15.5 GB; LoRA r64 ≈ 150 to 200 M trainable
  parameters (estimated from layer shapes); a batch of 8 × 100 s with
  checkpointing fits 40 GB (moshi-finetune: batch 16 → 39.6 GB). An 80 GB
  card is comfortable.
* **Cost**, method = GPU-hours × hourly rate (rates assumed: H100 $2.5 to
  $4/h, A100 $1.2 to $2/h):
  one run ≈ 20 × $2.5 to 4 = **$50 to $80** on H100 (A100: 45 × $1.2 to 2 =
  $55 to $90). A realistic project with 5 to 10 runs (lr/rank sweeps, data
  revisions) costs **$250 to $800** of GPU time. TTS for 490 h of synthetic
  speech is ≈ 28 M characters (≈ 2.6 words/s × 6 chars × 3,600 s × 490 h × ~0.5
  speech density × 2 channels); at an assumed $15 to $30 per million
  characters that is **$400 to $850**, usually the larger line. Mimi encoding
  and eval are small by comparison.
* **Inference**: Moshi runs in real time on a single modern GPU (Kyutai
  reports ~200 ms latency on an L4-class card; UNVERIFIED for PersonaPlex). It
  needs one GPU per few concurrent calls; measure before planning capacity.

## Risks and mitigations

| risk | why | mitigation |
|---|---|---|
| **Text-stream bandwidth** (biggest) | One text token per 80 ms. The contract's call JSON is ~35 to 50 tokens (≈ 3 to 4 s) and results 25 to 100 tokens (2 to 8 s) of frames in which Elle's inner monologue cannot advance. Waits get longer; her speech during a span has no fresh text to follow. | Retiming makes the data honest about it. Next steps: a compact call encoding (contract change, below); short results (summarise BUSY lists before injection); fillers that start *before* the call; option: a second text stream (Moshi has `demux_second_text_stream`, UNVERIFIED for this use). |
| **Catastrophic forgetting** of conversation quality | Narrow synthetic data, 7B model | LoRA only, low lr, 30% plain-conversation mix, short schedule; eval perplexity on held-out plain conversation and human listening tests as gates, not just tool metrics. |
| **Premature claims** | Text leads audio by about a frame, so the server cannot intercept a claim | Retimed data, unlikelihood on labelled premature claims, `premature_claim_rate <= 0.01` gate on a held-out family. If it is not met, train with text ahead of audio and add a server-side guard (`infer_runtime.md`). |
| **Result-injection latency** | Forced tokens take real time; a result can only be injected at a word boundary | Injection at the next boundary within 0.48 s; compact results; measure `tool.result_injected` frame minus arrival in production. |
| **Unsafe writes** | The model books before a readback or a yes | `write_gates`-labelled data, unlikelihood on the negatives' early `<tool>`, `write_unsafe_rate == 0` gate; the Node side can additionally refuse `book_intro_call` without a recent readback in the transcript. |
| **Synthetic-data gap** | Templated phrasing, TTS voices, cooperative callers, estimated timing | Paraphrase expansion; real production logs converted to this format for val/test at least; varied TTS voices and noise; the held-out `slot_conflict` family; listening tests. |
| **Untested real backend** | No GPU here | Smoke test step 5 above before any long run; the tiny model mirrors Moshi's tensor contract, so failures should be in loading and naming, not in the training logic. |

## Verified and unverified

**Verified by running here**: everything under the tiny backend (prepare,
masking, retiming, LoRA, new-token rows, checkpoint/resume/determinism,
the shared closed-loop eval loop with injection, all metrics, reference mode
on the 2,400-episode sample, plain mixing and the forgetting check,
`plain.py synth` against training/data's validator), torch 2.15 on CPU.

**Verified by reading source (2026-11-07), not by running it**:
kyutai-labs/moshi-finetune (README, `example/moshi_7B.yaml`, `train.py`,
`finetune/data/interleaver.py`, `finetune/loss.py`,
`finetune/wrapped_model.py`, `annotate.py`), kyutai-labs/moshi
(`models/lm.py`, `models/loaders.py`, `modules/lora.py`,
`modules/transformer.py`) and NVIDIA/personaplex (README, `moshi/models/lm.py`,
`moshi/models/loaders.py`, `server.py`, `offline.py`). The facts used: text ids
0=EPAD, 3=PAD, text_card 32000 plus the initial row; n_q 16, card 2048, Mimi
at 12.5 Hz; the LMModel.forward output and masks; the moshi-finetune data
format (stereo WAV with ch0 = model, `{"path","duration"}` JSONL, alignment
JSON), its loss weights and LoRA init; Kyutai's LoRA module and the
`isinstance(LoRALinear)` check in streaming attention; PersonaPlex's
`dep_q = 16`, its `<system>` text prompt, its `LMGen.step(text_token=…)`,
`HF repo nvidia/personaplex-7b-v1`, and its fork having no LoRA module.

**UNVERIFIED** (never executed; written against the source above):
`moshi_backend.py` in full (loading, resizing ScaledEmbedding rows, LoRA
wrapping of PersonaPlex, gradient-row masks with the real model, checkpoint
save/load of LoRA + new rows, merging our LoRA before streaming,
`MoshiFrameGenerator`: LMGen construction, forcing, banning, reading the
emitted token, `streaming(1)` per episode, PersonaPlex's system prompt and
voice prompt, Mimi silence/decoding for `--listen`); the SentencePiece
tokenizer lookup from `hf_repo` (`textvocab.build_tokenizer`);
`plain.py external`;
`prepare.mimi_audio` (Mimi encode call and shapes); the SentencePiece token
counts (the dry run uses a word-piece stand-in, so real span lengths will
differ by maybe ±30%); the inference server sketch, forcing tokens through
`on_text_logits_hook` in Kyutai's LMGen, and forced greetings; all hardware
throughput and pricing figures. Hugging Face (`huggingface.co`) is blocked by
this machine's proxy, so no model card, config.json or tokenizer was read.
The PersonaPlex paper (arXiv 2602.06053) was not read.

**Not using peft**: a peft LoRA layer is not Kyutai's `LoRALinear`, which
Moshi's streaming attention checks for, so it would have to be merged before
inference anyway. Kyutai's own module is used when present, otherwise a
20-line wrapper (`tiny_model.LoRALinear`), merged before streaming.

## Contract issues (CONTRACT.md not edited)

1. **Spans are not instantaneous.** The contract's time model lets a result
   arrive while the call is still being written, and lets claims follow results
   by less than the injection takes. The trainer retimes (above). The contract
   should say that the tool span's `t` is its *start*, that the result arrives
   no earlier than call end plus latency, and that claim spans must start after
   the result's *injection ends*; or the dataset should generate retimed
   timing itself, using the trainer's token counts.
2. **Compact serialization.** `{"id":"c1","name":…,"args":{…}}` spends tokens
   on `id` (the runtime can assign ids) and on seconds in `start`. Something
   like `<tool>check_availability{"start":"2026-11-12T14:00-08:00"}</tool>`
   would save about 30%. This needs a contract change on both sides.
3. **`get_my_bookings` on voice**: the contract lists it, but
   `twilioBridge.handleFunctionCall` has no case for it (it is SMS-only in
   `src/sms.ts`). A model trained to call it on a voice call would get
   `Unknown tool`. Add the case to the bridge before deployment.
4. **`save_user_fact` result**: the contract says `saved`; production and the
   data use `Saved.`. `check_availability` also returns `UNKNOWN: …`. The
   contract table should list both.
5. **Which writes are gated**: the contract says all writes need a readback
   and a yes, and also that `save_user_fact` is silent. The dataset's reading,
   which the evaluator follows, is that bookings and `sms_consent` are gated
   and other facts are silent and ungated.
6. **Dead air** is operationalised with a 0.3 s coverage tolerance for model
   output; the ground truth scores identically with 0.
7. **Alignment JSON**: the trainer accepts `render_audio.py`'s format as is.
