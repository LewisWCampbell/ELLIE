"""Did a checkpoint learn to start tool calls? Teacher-forced, no generation.

    python3 training/model/markers.py --config <cfg> [--checkpoint DIR] [--split val] [--max-examples 200]

Loads the model exactly as training does (base + LoRA + new token rows from
trainable.pt, unmerged) and runs train.eval_loss on prepared examples:

  tool_open_acc    how often <tool> is the most likely token where the data
                   has <tool> (the call decision; must be near 1)
  tool_open_p      mean probability given to <tool> there
  tool_open_rank   mean number of tokens ranked above <tool> there (0 = top)
  tool_close_acc   the same for </tool> (easy; near 1 says spans are learned)
  tool_open_false_positive_rate   <tool> predicted where the data has none

Read with the closed-loop report: high tool_open_acc here but no calls in
evaluate.py --mode model means generation differs from training (a harness
problem); low tool_open_acc means the model did not learn the decision.
Writes <run_dir>/eval/markers_<checkpoint>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch  # noqa: E402

from config_util import load_config, resolve_path  # noqa: E402
from prepare import load_examples  # noqa: E402
from textvocab import SPECIAL_TOKENS, build_tokenizer  # noqa: E402


def marker_report(cfg: dict, base: str, ckpt: str = None, split: str = "val", max_examples: int = 200,
                  device: str = None) -> dict:
    from train import build_model, eval_loss, latest_checkpoint, set_trainable

    device = device or cfg.get("train", {}).get("backend_device", "cuda" if torch.cuda.is_available() else "cpu")
    run_dir = resolve_path(cfg["run_dir"], base)
    ckpt = ckpt or latest_checkpoint(run_dir)
    if ckpt is None:
        raise FileNotFoundError(f"no checkpoint under {run_dir}")
    tok = build_tokenizer(cfg.get("tokenizer", {}), base)
    model, new_ids, _ = build_model(cfg, tok, device, base, None, os.path.join(run_dir, "base.pt"))
    assert new_ids == [tok.special_ids[s] for s in SPECIAL_TOKENS], (new_ids, tok.special_ids)
    set_trainable(model, torch.load(os.path.join(ckpt, "trainable.pt"), map_location=device), new_ids)
    # As train.run's eval set: no negative episodes (their unsafe calls were
    # trained to be suppressed, so they are not places <tool> should win).
    exs = [e for e in load_examples(resolve_path(cfg["data"]["prepared_dir"], base), split)
           if not e["meta"].get("negative")]
    open_id = new_ids[0]
    # Only examples that contain a <tool> target say anything about the decision.
    with_calls = [e for e in exs if (e["text"] == open_id).any()]
    exs = with_calls[:max_examples]
    res = eval_loss(model, exs, cfg, new_ids, device)
    return {"checkpoint": ckpt, "split": split, "examples": len(exs), **res}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint")
    ap.add_argument("--split", default="val")
    ap.add_argument("--max-examples", type=int, default=200)
    args = ap.parse_args(argv)
    cfg, base = load_config(args.config)
    rep = marker_report(cfg, base, args.checkpoint, args.split, args.max_examples)
    out_dir = os.path.join(resolve_path(cfg["run_dir"], base), "eval")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"markers_{os.path.basename(rep['checkpoint'].rstrip('/'))}.json")
    with open(path, "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
