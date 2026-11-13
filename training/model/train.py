"""LoRA fine-tuning loop for tool calls on Elle's inner-monologue text stream.

  python3 training/model/train.py --config training/model/config/dryrun.yaml
  torchrun --nproc-per-node 1 training/model/train.py --config training/model/config/base_gpu.yaml   # GPU, UNVERIFIED

Backends: `tiny` (TinyDuplexLM, CPU, used by tests and dry runs) and `moshi`
(real Moshi/PersonaPlex via moshi_backend.py, UNVERIFIED, gated on `import
moshi`). Both expose forward(text [B,T], audio [B,16,T]) -> text logits and
Elle-audio logits aligned with the input frames.

Loss = text CE (result spans masked; PAD/EPAD down-weighted; marker targets
up-weighted) + audio CE on Elle's 8 codebooks (semantic codebook up-weighted)
[+ unlikelihood on the frames a labelled negative gets wrong].
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from config_util import load_config, resolve_path  # noqa: E402
from prepare import load_examples  # noqa: E402
from textvocab import EPAD_ID, PAD_ID, SPECIAL_TOKENS, ZERO_TOKEN_ID, build_tokenizer  # noqa: E402
from tiny_model import (N_CB, TinyDuplexLM, build_tiny, load_trainable_state_dict, setup_trainable,  # noqa: E402
                        trainable_state_dict)


# --- determinism ----------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


# --- batching ---------------------------------------------------------------------

def collate(examples: List[dict], seq_frames: int, rng: random.Random) -> Dict[str, torch.Tensor]:
    """Random crop to seq_frames (never cutting the prompt prefix), right-pad
    with the zero token (no input, no loss)."""
    T = min(seq_frames, max(len(e["text"]) for e in examples))
    B = len(examples)
    text = torch.full((B, T), ZERO_TOKEN_ID, dtype=torch.long)
    audio = torch.full((B, 2 * N_CB, T), ZERO_TOKEN_ID, dtype=torch.long)
    loss_mask = torch.zeros(B, T)
    neg_mask = torch.zeros(B, T)
    audio_mask = torch.zeros(B, T)
    for i, e in enumerate(examples):
        n = len(e["text"])
        a = rng.randint(0, n - T) if n > T else 0
        m = min(T, n - a)
        text[i, :m] = torch.from_numpy(e["text"][a:a + m].astype(np.int64))
        audio[i, :, :m] = torch.from_numpy(e["audio"][:, a:a + m].astype(np.int64))
        loss_mask[i, :m] = torch.from_numpy(e["loss_mask"][a:a + m].astype(np.float32))
        neg_mask[i, :m] = torch.from_numpy(e["neg_mask"][a:a + m].astype(np.float32))
        audio_mask[i, :m] = torch.from_numpy(e["audio_mask"][a:a + m].astype(np.float32))
    return {"text": text, "audio": audio, "loss_mask": loss_mask, "neg_mask": neg_mask, "audio_mask": audio_mask}


def strip_specials(ex: dict, base_card: int) -> dict:
    """For simulating a pretrained base: marker frames become masked PAD."""
    ex = dict(ex)
    t = ex["text"].copy()
    special = t >= base_card
    t[special] = PAD_ID
    ex["text"] = t
    ex["loss_mask"] = np.where(special, 0, ex["loss_mask"]).astype(np.uint8)
    ex["neg_mask"] = np.zeros_like(ex["neg_mask"])
    return ex


# --- loss -------------------------------------------------------------------------

def compute_loss(model_out: Tuple, batch: Dict[str, torch.Tensor], lcfg: dict, special_ids: List[int],
                 negatives: str = "exclude", ul_weight: float = 0.0) -> Dict[str, torch.Tensor]:
    text_logits, audio_logits = model_out[0], model_out[1]
    text_valid = model_out[2] if len(model_out) > 2 else None  # real backend's masks
    audio_valid = model_out[3] if len(model_out) > 3 else None
    text = batch["text"]
    V = text_logits.shape[-1]

    # text CE
    w = batch["loss_mask"] * (text != ZERO_TOKEN_ID).float()
    if negatives == "unlikelihood":
        w = w * (1.0 - batch["neg_mask"])  # bad frames get UL instead of CE
    if text_valid is not None:
        w = w * text_valid.float()
    pad_w = float(lcfg.get("text_padding_weight", 0.5))
    w = torch.where((text == PAD_ID) | (text == EPAD_ID), w * pad_w, w)
    if special_ids:
        is_special = torch.zeros_like(text, dtype=torch.bool)
        for s in special_ids:
            is_special |= text == s
        w = torch.where(is_special, w * float(lcfg.get("special_token_weight", 1.0)), w)
    tgt = text.clamp(min=0)
    ce = F.cross_entropy(text_logits.reshape(-1, V).float(), tgt.reshape(-1), reduction="none").view_as(w)
    text_loss = (ce * w).sum() / w.sum().clamp(min=1e-6)

    # audio CE on Elle's codebooks
    codes = batch["audio"][:, :N_CB]
    aw = batch["audio_mask"][:, None, :].expand_as(codes).float() * (codes != ZERO_TOKEN_ID).float()
    if audio_valid is not None:
        aw = aw * audio_valid[:, :N_CB].float()
    aw = aw.clone()
    aw[:, 0] *= float(lcfg.get("first_codebook_weight_multiplier", 1.0))
    al = audio_logits[:, :N_CB]
    ace = F.cross_entropy(al.reshape(-1, al.shape[-1]).float(), codes.clamp(min=0).reshape(-1),
                          reduction="none").view_as(aw)
    audio_loss = (ace * aw).sum() / aw.sum().clamp(min=1e-6)
    total = text_loss + float(lcfg.get("audio_weight", 1.0)) * audio_loss

    ul = torch.zeros(())
    if negatives == "unlikelihood" and batch["neg_mask"].sum() > 0:
        p = torch.softmax(text_logits.float(), -1).gather(-1, tgt[..., None])[..., 0]
        ul_terms = -torch.log((1.0 - p).clamp(min=1e-6)) * batch["neg_mask"]
        ul = ul_terms.sum() / batch["neg_mask"].sum()
        total = total + ul_weight * ul
    return {"loss": total, "text_loss": text_loss.detach(), "audio_loss": audio_loss.detach(), "ul_loss": ul.detach()}


# --- schedule -------------------------------------------------------------------------

def one_cycle(step: int, total: int, pct_start: float) -> float:
    """Linear warm-up for pct_start of the steps, then cosine to zero (the shape
    of PyTorch OneCycleLR that moshi-finetune uses)."""
    warm = max(1, int(total * pct_start))
    if step < warm:
        return (step + 1) / warm
    prog = (step - warm) / max(1, total - warm)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))


# --- model construction ------------------------------------------------------------

def marker_init_rows(model: TinyDuplexLM, tok) -> torch.Tensor:
    """Initialise each new row as the mean embedding of the marker's literal
    pieces (e.g. '<', 'tool', '>'), a common trick so a new token starts near
    what the model already associates with that text."""
    rows = []
    with torch.no_grad():
        for s in SPECIAL_TOKENS:
            ids = [i for i in tok.encode_plain(s) if 0 <= i < model.text_card]
            rows.append(model.text_emb.weight[ids].mean(0) if ids else model.text_emb.weight[: model.text_card].mean(0))
    return torch.stack(rows)


def build_model(cfg: dict, tok, device: str, base_dir: str, train_examples: Optional[List[dict]] = None,
                base_path: Optional[str] = None, log=print):
    """Returns (model, new_ids, param_groups). For `tiny`, the base is either
    loaded from base_path or created (and optionally pretrained on the training
    stream with markers stripped, to stand in for a pretrained model)."""
    backend = cfg["model"]["backend"]
    if backend == "moshi":  # pragma: no cover - GPU only
        from moshi_backend import load_real_model

        model, _tok_path, new_ids, groups = load_real_model(cfg, device)
        return model, new_ids, groups
    if backend != "tiny":
        raise ValueError(backend)
    model = build_tiny(cfg, tok.base_card).to(device)
    if base_path and os.path.exists(base_path):
        model.load_state_dict(torch.load(base_path, map_location=device))
    else:
        steps = int(cfg.get("tiny", {}).get("pretrain_steps", 0))
        if steps and train_examples:
            stripped = [strip_specials(e, tok.base_card) for e in train_examples]
            opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["tiny"].get("pretrain_lr", 3e-3)))
            rng = random.Random(cfg.get("seed", 0) + 1)
            bs = int(cfg["train"]["batch_size"])
            for s in range(steps):
                batch = collate(rng.sample(stripped, min(bs, len(stripped))), int(cfg["train"]["seq_frames"]), rng)
                out = compute_loss(model(batch["text"].to(device), batch["audio"].to(device)),
                                   {k: v.to(device) for k, v in batch.items()}, cfg.get("loss", {}), [])
                opt.zero_grad()
                out["loss"].backward()
                opt.step()
            log(f"[tiny] pretrained base for {steps} steps, final loss {out['loss'].item():.3f}")
        if base_path:
            os.makedirs(os.path.dirname(base_path), exist_ok=True)
            torch.save(model.state_dict(), base_path)
    new_ids = model.resize_text_vocab(len(SPECIAL_TOKENS), marker_init_rows(model, tok))
    assert new_ids == [tok.special_ids[s] for s in SPECIAL_TOKENS], (new_ids, tok.special_ids)
    groups = setup_trainable(model, new_ids, cfg.get("lora", {}))
    return model.to(device), new_ids, groups


# --- checkpoints -------------------------------------------------------------------

def get_trainable(model, new_ids) -> Dict[str, torch.Tensor]:
    if isinstance(model, TinyDuplexLM):
        return trainable_state_dict(model, new_ids)
    from moshi_backend import trainable_state_dict as real_sd  # pragma: no cover

    return real_sd(model.lm, new_ids)


def set_trainable(model, sd, new_ids) -> None:
    if isinstance(model, TinyDuplexLM):
        load_trainable_state_dict(model, sd, new_ids)
        return
    from moshi_backend import load_trainable_state_dict as real_load  # pragma: no cover

    real_load(model.lm, sd, new_ids)


def save_checkpoint(run_dir: str, step: int, model, new_ids, opt, sched_state: dict, keep: int, cfg: dict) -> str:
    d = os.path.join(run_dir, "checkpoints", f"step_{step:06d}")
    os.makedirs(d, exist_ok=True)
    torch.save(get_trainable(model, new_ids), os.path.join(d, "trainable.pt"))
    torch.save({"step": step, "optimizer": opt.state_dict(), "sched": sched_state,
                "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}},
               os.path.join(d, "state.pt"))
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump({"step": step, "new_token_ids": new_ids, "special_tokens": SPECIAL_TOKENS,
                   "backend": cfg["model"]["backend"]}, f)
    cks = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "step_*")))
    for old in cks[:-keep] if keep > 0 else []:
        shutil.rmtree(old, ignore_errors=True)
    return d


def latest_checkpoint(run_dir: str) -> Optional[str]:
    cks = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "step_*")))
    return cks[-1] if cks else None


def load_for_eval(cfg: dict, base_dir: str, ckpt: Optional[str] = None, device: str = "cpu"):
    """Rebuild the tiny model exactly as trained: base.pt + trainable rows/LoRA.
    (The real backend loads through moshi_backend.load_for_inference.)"""
    tok = build_tokenizer(cfg.get("tokenizer", {}), base_dir)
    run_dir = resolve_path(cfg["run_dir"], base_dir)
    ckpt = ckpt or latest_checkpoint(run_dir)
    if ckpt is None:
        raise FileNotFoundError(f"no checkpoint under {run_dir}")
    model, new_ids, _ = build_model(cfg, tok, device, base_dir, None, os.path.join(run_dir, "base.pt"))
    load_trainable_state_dict(model, torch.load(os.path.join(ckpt, "trainable.pt"), map_location=device), new_ids)
    model.eval()
    return model, tok, new_ids, ckpt


# --- tool / plain mixing -----------------------------------------------------------

class MixedSampler:
    """Draws each batch from the tool set and, for `plain_fraction` of its
    examples, from the plain-conversation set. The plain count per batch is
    exact in the long run (a carried remainder: 0.3 x 4 gives 1,1,2,1,1,2,...),
    every draw comes from a seeded RNG, and the whole state (RNGs + remainder)
    goes into checkpoints, so a resumed run continues the same sequence."""

    def __init__(self, tool: List[dict], plain: List[dict], fraction: float, seed: int):
        self.tool, self.plain = tool, plain
        self.fraction = float(fraction) if plain else 0.0
        self.rng_tool = random.Random(seed)
        self.rng_plain = random.Random(seed + 7919)
        self.rng_crop = random.Random(seed + 104729)
        self.carry = 0.0
        self.drawn = {"tool": 0, "plain": 0}

    def next(self, bs: int) -> Tuple[List[dict], List[dict]]:
        self.carry += bs * self.fraction
        n_plain = min(bs, int(math.floor(self.carry + 1e-9)))
        self.carry -= n_plain
        tool = [self.tool[self.rng_tool.randrange(len(self.tool))] for _ in range(bs - n_plain)]
        plain = [self.plain[self.rng_plain.randrange(len(self.plain))] for _ in range(n_plain)]
        self.drawn["tool"] += len(tool)
        self.drawn["plain"] += len(plain)
        return tool, plain

    def state(self) -> dict:
        return {"tool": self.rng_tool.getstate(), "plain": self.rng_plain.getstate(),
                "crop": self.rng_crop.getstate(), "carry": self.carry, "drawn": dict(self.drawn)}

    def load(self, st: dict) -> None:
        self.rng_tool.setstate(st["tool"]); self.rng_plain.setstate(st["plain"]); self.rng_crop.setstate(st["crop"])
        self.carry = float(st["carry"]); self.drawn = dict(st["drawn"])


class lora_scale:
    """Context manager: multiply every LoRA's scaling (ours or Kyutai's) by s;
    s=0 gives the base model back, for a forgetting baseline."""

    def __init__(self, model, s: float):
        self.mods = [m for m in model.modules() if hasattr(m, "lora_A") and hasattr(m, "scaling")]
        self.s = s

    def __enter__(self):
        self.saved = [m.scaling for m in self.mods]
        for m in self.mods:
            m.scaling = m.scaling * self.s
        return self

    def __exit__(self, *exc):
        for m, v in zip(self.mods, self.saved):
            m.scaling = v


def load_plain(cfg: dict, base_dir: str) -> Tuple[List[dict], List[dict]]:
    """(train, held-out) plain-conversation examples, or ([], []) if none configured."""
    d = cfg.get("data", {}).get("plain_prepared_dir")
    if not d:
        return [], []
    d = resolve_path(d, base_dir)
    return load_examples(d, cfg["data"].get("plain_train_split", "train")), \
        load_examples(d, cfg["data"].get("plain_eval_split", "val"))


# --- eval loss ----------------------------------------------------------------------

@torch.no_grad()
def eval_loss(model, examples, cfg, special_ids, device) -> Dict[str, float]:
    model.eval()
    rng = random.Random(0)
    tot: Dict[str, float] = {"loss": 0.0, "text_loss": 0.0, "audio_loss": 0.0}
    hit = n = 0
    # <tool> (deciding to call: the scarce, hard decision) and </tool> (ending a
    # call: easy) are reported separately; pooled, a <tool> that is never the
    # argmax hides behind a perfect </tool> at ~0.5.
    c = {"open_hit": 0, "open_n": 0, "open_p": 0.0, "open_rank": 0.0, "close_hit": 0, "close_n": 0,
         "fp": 0, "other_n": 0}
    bs = int(cfg["train"]["batch_size"])
    for i in range(0, len(examples), bs):
        batch = {k: v.to(device) for k, v in collate(examples[i:i + bs], int(cfg["train"]["seq_frames"]), rng).items()}
        out = model(batch["text"], batch["audio"])
        l = compute_loss(out, batch, cfg.get("loss", {}), special_ids)
        for k in tot:
            tot[k] += float(l[k]) * len(examples[i:i + bs])
        logits, tgt = out[0].float(), batch["text"]
        # Frames trained by unlikelihood (neg_mask) are not targets to get right.
        valid = (batch["loss_mask"] > 0) & (tgt >= 0) & (batch["neg_mask"] == 0)
        pred = logits.argmax(-1)
        sel = torch.zeros_like(tgt, dtype=torch.bool)
        for s in special_ids[:2]:  # <tool>, </tool>: the predicted ones
            sel |= tgt == s
        sel &= valid
        if sel.any():
            hit += int((pred[sel] == tgt[sel]).sum())
            n += int(sel.sum())
        if special_ids:
            open_id, close_id = special_ids[0], special_ids[1]
            o = valid & (tgt == open_id)
            if o.any():
                lo = logits[o]
                c["open_hit"] += int((lo.argmax(-1) == open_id).sum())
                c["open_n"] += int(o.sum())
                c["open_p"] += float(torch.softmax(lo, -1)[:, open_id].sum())
                c["open_rank"] += float((lo > lo[:, open_id:open_id + 1]).sum())
            cl = valid & (tgt == close_id)
            if cl.any():
                c["close_hit"] += int((pred[cl] == close_id).sum())
                c["close_n"] += int(cl.sum())
            other = valid & (tgt != open_id)
            c["fp"] += int((pred[other] == open_id).sum())
            c["other_n"] += int(other.sum())
    model.train()
    res = {k: v / max(1, len(examples)) for k, v in tot.items()}
    nan = float("nan")
    res["tool_marker_acc"] = hit / n if n else nan
    res["tool_open_acc"] = c["open_hit"] / c["open_n"] if c["open_n"] else nan
    res["tool_open_p"] = c["open_p"] / c["open_n"] if c["open_n"] else nan
    res["tool_open_rank"] = c["open_rank"] / c["open_n"] if c["open_n"] else nan  # 0 = argmax
    res["tool_close_acc"] = c["close_hit"] / c["close_n"] if c["close_n"] else nan
    res["tool_open_false_positive_rate"] = c["fp"] / c["other_n"] if c["other_n"] else nan
    res["tool_open_targets"] = c["open_n"]
    return res


# --- main loop -------------------------------------------------------------------------

def run(cfg: dict, base_dir: str, resume: bool = False, log=print) -> Dict[str, Any]:
    seed = int(cfg.get("seed", 0))
    seed_everything(seed)
    tcfg = cfg["train"]
    torch.set_num_threads(int(tcfg.get("num_threads", 4)))
    device = tcfg.get("backend_device", "cpu")
    run_dir = resolve_path(cfg["run_dir"], base_dir)
    prepared = resolve_path(cfg["data"]["prepared_dir"], base_dir)
    os.makedirs(run_dir, exist_ok=True)
    tok = build_tokenizer(cfg.get("tokenizer", {}), base_dir)

    train_split = tcfg.get("split", "train")
    train_ex = load_examples(prepared, train_split)
    negatives = tcfg.get("negatives", "exclude")
    if negatives == "exclude":
        train_ex = [e for e in train_ex if not e["meta"].get("negative")]
    if not train_ex:
        raise RuntimeError(f"no training examples in {prepared} split={train_split}")
    eval_split = cfg.get("eval", {}).get("split", "val")
    eval_ex = [e for e in load_examples(prepared, eval_split) if not e["meta"].get("negative")] or train_ex

    plain_train, plain_eval = load_plain(cfg, base_dir)
    plain_fraction = float(cfg.get("data", {}).get("plain_fraction", 0.0)) if plain_train else 0.0
    if cfg.get("data", {}).get("plain_prepared_dir") and not plain_train:
        raise RuntimeError("data.plain_prepared_dir is set but holds no training examples (see plain.py)")

    model, new_ids, groups = build_model(cfg, tok, device, base_dir, train_ex + plain_train,
                                         os.path.join(run_dir, "base.pt"), log)
    special_ids = list(new_ids)
    ocfg = tcfg["optim"]
    param_groups = [{"params": groups["lora"], "lr": float(ocfg["lr"]), "weight_decay": float(ocfg.get("weight_decay", 0.0))},
                    {"params": groups["new_tokens"], "lr": float(ocfg.get("new_token_lr", ocfg["lr"])), "weight_decay": 0.0}]
    param_groups = [g for g in param_groups if g["params"]]
    opt = torch.optim.AdamW(param_groups, betas=tuple(ocfg.get("betas", (0.9, 0.95))), eps=1e-8)
    base_lrs = [g["lr"] for g in opt.param_groups]
    n_trainable = sum(int(p.numel()) for g in param_groups for p in g["params"])
    log(f"trainable tensors: lora={len(groups['lora'])} new_token_tensors={len(groups['new_tokens'])} "
        f"(params incl. full rows of masked tensors: {n_trainable}); "
        f"tool examples {len(train_ex)}, plain {len(plain_train)} (fraction {plain_fraction})")

    step, max_steps = 0, int(tcfg["max_steps"])
    sampler = MixedSampler(train_ex, plain_train, plain_fraction, seed)
    ecfg = cfg.get("eval", {})
    forgetting_path = os.path.join(run_dir, "plain_forgetting.json")
    plain_base = None
    if resume and latest_checkpoint(run_dir):
        ck = latest_checkpoint(run_dir)
        set_trainable(model, torch.load(os.path.join(ck, "trainable.pt")), new_ids)
        st = torch.load(os.path.join(ck, "state.pt"), weights_only=False)
        opt.load_state_dict(st["optimizer"])
        step = int(st["step"])
        random.setstate(st["rng"]["python"]); np.random.set_state(st["rng"]["numpy"]); torch.set_rng_state(st["rng"]["torch"])
        sampler.load(st["sched"]["sampler"])
        plain_base = st["sched"].get("plain_base")
        log(f"resumed from {ck} at step {step}")
    if plain_eval and plain_base is None:
        # Forgetting baseline: the base model (LoRA off) on held-out plain conversation.
        with lora_scale(model, 0.0):
            plain_base = eval_loss(model, plain_eval, cfg, special_ids, device)["text_loss"]
        log(f"plain-conversation text loss of the base model: {plain_base:.4f}")

    log_path = os.path.join(run_dir, "train_log.jsonl")
    history: List[Dict[str, float]] = []
    bs, accum = int(tcfg["batch_size"]), int(tcfg.get("grad_accum", 1))
    clip = float(ocfg.get("grad_clip", 1.0))
    lcfg = cfg.get("loss", {})
    ul_w = float(tcfg.get("unlikelihood_weight", 0.0))
    seq = int(tcfg["seq_frames"])
    model.train()
    t0 = time.time()
    with open(log_path, "a") as logf:
        while step < max_steps:
            scale = one_cycle(step, max_steps, float(ocfg.get("pct_start", 0.05)))
            for g, lr in zip(opt.param_groups, base_lrs):
                g["lr"] = lr * scale
            opt.zero_grad()
            agg = {"loss": 0.0, "text_loss": 0.0, "audio_loss": 0.0, "ul_loss": 0.0}
            parts: Dict[str, Dict[str, float]] = {}
            for _ in range(accum):
                tool_b, plain_b = sampler.next(bs)
                n_total = len(tool_b) + len(plain_b)
                # Tool and plain examples go through separately so each loss is
                # logged on its own; the step loss weights them by example count.
                for name, exs, negs in (("tool", tool_b, negatives), ("plain", plain_b, "exclude")):
                    if not exs:
                        continue
                    batch = {k: v.to(device) for k, v in collate(exs, seq, sampler.rng_crop).items()}
                    out = compute_loss(model(batch["text"], batch["audio"]), batch, lcfg, special_ids, negs, ul_w)
                    w = len(exs) / n_total
                    (out["loss"] * w / accum).backward()
                    for k in agg:
                        agg[k] += float(out[k].detach()) * w / accum
                    pa = parts.setdefault(name, {"loss": 0.0, "text_loss": 0.0, "audio_loss": 0.0, "n": 0})
                    pa["loss"] += float(out["loss"].detach()); pa["text_loss"] += float(out["text_loss"])
                    pa["audio_loss"] += float(out["audio_loss"]); pa["n"] += 1
            torch.nn.utils.clip_grad_norm_([p for g in param_groups for p in g["params"]], clip)
            opt.step()
            step += 1
            rec = {"step": step, "lr_scale": scale, **agg, "elapsed_s": round(time.time() - t0, 2)}
            for name, pa in parts.items():
                for k in ("loss", "text_loss", "audio_loss"):
                    rec[f"{name}_{k}"] = pa[k] / pa["n"]
            rec["plain_examples_seen"] = sampler.drawn["plain"]
            rec["tool_examples_seen"] = sampler.drawn["tool"]
            if ecfg.get("every_steps") and (step % int(ecfg["every_steps"]) == 0 or step == max_steps):
                rec.update({f"eval_{k}": v for k, v in eval_loss(model, eval_ex, cfg, special_ids, device).items()})
                msg = (f"step {step}: loss {agg['loss']:.4f} text {agg['text_loss']:.4f} audio {agg['audio_loss']:.4f} "
                       f"| eval text {rec['eval_text_loss']:.4f} <tool> acc {rec['eval_tool_open_acc']:.2f}"
                       f" p {rec['eval_tool_open_p']:.2f} </tool> acc {rec['eval_tool_close_acc']:.2f}"
                       f" false <tool> {rec['eval_tool_open_false_positive_rate']:.4f}")
                if plain_eval:
                    pl = eval_loss(model, plain_eval, cfg, special_ids, device)["text_loss"]
                    rec.update({"eval_plain_text_loss": pl, "eval_plain_text_loss_base": plain_base,
                                "eval_plain_text_loss_delta": pl - plain_base})
                    msg += f" | plain text {pl:.4f} (base {plain_base:.4f}, delta {pl - plain_base:+.4f})"
                    write_forgetting(forgetting_path, step, plain_base, pl, ecfg.get("gates", {}))
                log(msg)
            history.append(rec)
            logf.write(json.dumps(rec) + "\n")
            if ecfg.get("ckpt_every_steps") and (step % int(ecfg["ckpt_every_steps"]) == 0 or step == max_steps):
                save_checkpoint(run_dir, step, model, new_ids, opt,
                                {"step": step, "sampler": sampler.state(), "plain_base": plain_base},
                                int(ecfg.get("keep_checkpoints", 3)), cfg)
    return {"history": history, "run_dir": run_dir, "new_ids": new_ids, "model": model, "tokenizer": tok,
            "checkpoint": latest_checkpoint(run_dir), "sampler": sampler}


def write_forgetting(path: str, step: int, base: float, current: float, gates: dict) -> dict:
    """run_dir/plain_forgetting.json: held-out plain-conversation text loss now
    vs the base model. evaluate.py includes it in the model report."""
    delta = current - base
    g = (gates or {}).get("plain_text_loss_delta")
    rec = {"step": step, "plain_text_loss_base": base, "plain_text_loss": current, "plain_text_loss_delta": delta,
           "gate": None if g is None else {"max": g.get("max"), "pass": delta <= g.get("max")}}
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
    return rec


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-steps", type=int)
    args = ap.parse_args(argv)
    cfg, base = load_config(args.config)
    if args.max_steps:
        cfg["train"]["max_steps"] = args.max_steps
    res = run(cfg, base, resume=args.resume)
    h = res["history"]
    print(json.dumps({"first_loss": h[0]["loss"] if h else None, "last_loss": h[-1]["loss"] if h else None,
                      "steps": len(h), "checkpoint": res["checkpoint"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
