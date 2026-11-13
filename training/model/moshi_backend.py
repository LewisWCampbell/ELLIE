"""Real-model backend: Kyutai Moshi or NVIDIA PersonaPlex behind the same
forward interface as TinyDuplexLM.

EVERYTHING IN THIS FILE IS UNVERIFIED. It was written against source read on
2026-11-07 (kyutai-labs/moshi `moshi/models/{lm,loaders}.py`,
`moshi/modules/{lora,transformer}.py`; kyutai-labs/moshi-finetune
`train.py`, `finetune/wrapped_model.py`; NVIDIA/personaplex
`moshi/moshi/models/{lm,loaders}.py`, `server.py`) but never executed: this
machine has no GPU, the `moshi` package is not installed, and huggingface.co
is blocked by the proxy, so no checkpoint or tokenizer could be loaded.
Facts taken from that source (verified by reading, not by running):
  * LMModel.forward(codes=[B, 1 + n_q, T]) -> LMOutput(logits [B, dep_q, T, card],
    mask, text_logits [B, 1, T, text_card], text_mask); logits are aligned with
    codes (delays undone), invalid positions are NaN and masked out.
  * n_q = 16 (8 model + 8 user codebooks), card = 2048, text_card = 32000,
    dim 4096, 32 layers; Moshi dep_q = 8, PersonaPlex dep_q = 16.
  * text_emb has text_card + 1 rows (row text_card = initial token); the input
    id -1 is a zero embedding.
  * Kyutai's moshi ships modules/lora.py (replace_all_linear_with_lora, LoRALinear
    with lora_A/lora_B and `frozen_W`); its attention checks
    `isinstance(in_proj, LoRALinear)` when building the streaming KV cache, so a
    foreign LoRA wrapper must be MERGED before streaming inference.
  * PersonaPlex's fork has no modules/lora.py and its get_moshi_lm takes no
    LoRA arguments; its LMGen.step accepts `text_token=` to force a text token
    (used for its system prompt), which is what result injection needs.
  * Both packages are named `moshi`; install one per environment.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from tiny_model import RowMask, apply_lora


def available() -> Tuple[bool, str]:
    try:
        import moshi  # type: ignore  # noqa: F401
        from moshi.models import loaders  # type: ignore  # noqa: F401
    except Exception as e:  # pragma: no cover - not installed on the CPU box
        return False, f"moshi package not importable: {e!r}"
    return True, "ok"


def _has_kyutai_lora() -> bool:
    try:
        import moshi.modules.lora  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


@torch.no_grad()
def resize_moshi_text_vocab(lm: nn.Module, n_new: int, init_rows: Optional[torch.Tensor] = None) -> List[int]:
    """Append n_new text ids to a Moshi LMModel. UNVERIFIED: assumes text_emb /
    depformer_text_emb expose `.weight` used via F.embedding (ScaledEmbedding
    subclasses nn.Embedding) and that nothing else caches text_card."""
    old = int(lm.text_card)

    def grow_emb(emb: nn.Module) -> None:
        w = emb.weight
        assert w.shape[0] == old + 1, ("unexpected text embedding rows", w.shape, old)
        new = torch.empty(old + n_new + 1, w.shape[1], device=w.device, dtype=w.dtype)
        new[:old] = w[:old]
        new[old + n_new] = w[old]  # initial token moves to the end
        new[old:old + n_new] = init_rows.to(w) if init_rows is not None else w[:old].float().mean(0).to(w.dtype)
        emb.weight = nn.Parameter(new, requires_grad=False)
        if hasattr(emb, "num_embeddings"):
            emb.num_embeddings = old + n_new + 1

    grow_emb(lm.text_emb)
    if getattr(lm, "depformer_text_emb", None) is not None:
        grow_emb(lm.depformer_text_emb)
    lin = lm.text_linear
    base_w = lin.frozen_W.weight if hasattr(lin, "frozen_W") else lin.weight
    new_w = torch.cat([base_w, base_w.float().mean(0, keepdim=True).repeat(n_new, 1).to(base_w.dtype)], 0)
    target = lin.frozen_W if hasattr(lin, "frozen_W") else lin
    target.weight = nn.Parameter(new_w, requires_grad=False)
    target.out_features = new_w.shape[0]
    if getattr(target, "bias", None) is not None:
        target.bias = nn.Parameter(torch.cat([target.bias, target.bias.new_full((n_new,), float(target.bias.min()) - 4.0)]))
    if hasattr(lin, "lora_B"):  # kyutai LoRA wraps text_linear too: grow B with zero rows
        b = lin.lora_B.weight
        lin.lora_B.weight = nn.Parameter(torch.cat([b, b.new_zeros(n_new, b.shape[1])], 0))
        lin.lora_B.out_features = b.shape[0] + n_new
    lm.text_card = old + n_new
    return list(range(old, old + n_new))


class MoshiAdapter(nn.Module):
    """Gives a Moshi LMModel TinyDuplexLM's training interface plus validity masks."""

    def __init__(self, lm: nn.Module):
        super().__init__()
        self.lm = lm

    @property
    def text_card(self) -> int:
        return int(self.lm.text_card)

    def forward(self, text: torch.Tensor, audio: torch.Tensor):
        codes = torch.cat([text[:, None], audio], dim=1)  # [B, 17, T]
        # Kyutai's LMModel trains through forward(codes, condition_tensors);
        # PersonaPlex's has no forward() and names the same pass forward_train.
        # Both return LMOutput(logits, mask, text_logits, text_mask).
        if hasattr(self.lm, "forward_train"):
            out = self.lm.forward_train(codes)
        else:
            out = self.lm(codes=codes, condition_tensors=None)
        # NaNs at invalid positions: zero them; the masks remove them from the loss.
        text_logits = torch.nan_to_num(out.text_logits[:, 0])
        audio_logits = torch.nan_to_num(out.logits)
        return text_logits, audio_logits, out.text_mask[:, 0], out.mask


def load_real_model(cfg: Dict[str, Any], device: str = "cuda"):  # pragma: no cover - GPU only
    """Returns (adapter, tokenizer_path, new_ids, param_groups). UNVERIFIED."""
    ok, why = available()
    if not ok:
        raise RuntimeError(f"real backend unavailable: {why}. Use model.backend=tiny for dry runs.")
    from moshi.models import loaders  # type: ignore

    mcfg = cfg["model"]
    dtype = getattr(torch, mcfg.get("dtype", "bfloat16"))
    from moshi_compat import checkpoint
    info = checkpoint(mcfg["hf_repo"], loaders)
    lcfg = cfg.get("lora", {})
    use_kyutai_lora = _has_kyutai_lora() and lcfg.get("impl", "auto") in ("auto", "kyutai")
    kwargs = {}
    if use_kyutai_lora:
        kwargs["lm_kwargs_overrides"] = {"gradient_checkpointing": bool(mcfg.get("gradient_checkpointing", True)),
                                         "lora": True, "lora_rank": int(lcfg.get("rank", 128)),
                                         "lora_scaling": float(lcfg.get("scaling", 2.0))}
    lm = info.get_moshi(device=device, dtype=dtype, **kwargs)
    tokenizer_path = str(info.tokenizer)
    for p in lm.parameters():
        p.requires_grad = False
    if not use_kyutai_lora:
        # PersonaPlex path: LoRA on the attention weights themselves
        # (param_lora), because PersonaPlex reads its weights directly instead
        # of calling layers, so wrapping layers cannot work. Merged into the
        # weights (merge_param_lora) before LMGen streaming.
        from param_lora import apply_param_lora
        wrapped = apply_param_lora(lm, int(lcfg.get("rank", 128)), float(lcfg.get("scaling", 2.0)),
                                   include_prefixes=tuple(lcfg.get("include_prefixes", ["transformer.", "depformer."])))
        if not wrapped:
            raise RuntimeError("found no attention weights to attach LoRA to; the model layout has changed")
    else:
        # moshi-finetune's initialize_lora_parameters: A kaiming, B zero.
        for n, m in lm.named_modules():
            if n.endswith("lora_A"):
                nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5)
            elif n.endswith("lora_B"):
                nn.init.zeros_(m.weight)
    new_ids = resize_moshi_text_vocab(lm, int(cfg.get("special_tokens_count", 4)))
    groups: Dict[str, List[nn.Parameter]] = {"lora": [], "new_tokens": []}
    for n, p in lm.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            p.requires_grad = True
            groups["lora"].append(p)
    for mod in (lm.text_emb, getattr(lm, "depformer_text_emb", None), lm.text_linear):
        if mod is None:
            continue
        w = mod.frozen_W.weight if hasattr(mod, "frozen_W") else mod.weight
        w.requires_grad = True
        w.register_hook(RowMask(w.shape[0], new_ids))
        groups["new_tokens"].append(w)
    lm.train()
    adapter = MoshiAdapter(lm)
    adapter.info = info  # CheckpointInfo: get_mimi() for evaluation / listening
    return adapter, tokenizer_path, new_ids, groups


# --- checkpoints of the real model -------------------------------------------------

def _row_tensors(lm: nn.Module) -> Dict[str, Optional[torch.Tensor]]:
    lin = lm.text_linear
    target = lin.frozen_W if hasattr(lin, "frozen_W") else lin
    dte = getattr(lm, "depformer_text_emb", None)
    return {"text_emb": lm.text_emb.weight, "depformer_text_emb": None if dte is None else dte.weight,
            "text_linear.weight": target.weight, "text_linear.bias": getattr(target, "bias", None)}


def trainable_state_dict(lm: nn.Module, new_ids: List[int]) -> Dict[str, torch.Tensor]:
    """LoRA A/B plus the new tokens' rows: everything a checkpoint must carry on
    top of the frozen base (same layout as tiny_model.trainable_state_dict)."""
    sd = {k: v.detach().cpu() for k, v in lm.state_dict().items() if "lora_A" in k or "lora_B" in k}
    ids = torch.tensor(new_ids, dtype=torch.long)
    for name, w in _row_tensors(lm).items():
        if w is not None:
            sd[f"new_tokens.{name}"] = w.detach()[ids.to(w.device)].cpu()
    return sd


@torch.no_grad()
def load_trainable_state_dict(lm: nn.Module, sd: Dict[str, torch.Tensor], new_ids: List[int]) -> None:
    lora = {k: v for k, v in sd.items() if not k.startswith("new_tokens.")}
    _missing, unexpected = lm.load_state_dict(lora, strict=False)
    assert not unexpected, unexpected
    for name, w in _row_tensors(lm).items():
        key = f"new_tokens.{name}"
        if w is not None and key in sd:
            w[torch.tensor(new_ids, device=w.device)] = sd[key].to(w)


def merge_foreign_lora(module: nn.Module) -> int:
    """Replace our tiny_model.LoRALinear wrappers by plain merged nn.Linear, as
    Moshi's streaming attention requires (it accepts nn.Linear or Kyutai's own
    LoRALinear). Returns the number merged."""
    from tiny_model import LoRALinear

    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            setattr(module, name, child.merged().to(child.base.weight.device, child.base.weight.dtype))
            n += 1
        else:
            n += merge_foreign_lora(child)
    return n


def load_for_inference(cfg: Dict[str, Any], ckpt_dir: str, device: str = "cuda"):  # pragma: no cover - GPU only
    """(lm, mimi, tokenizer, new_ids) for evaluation/serving. UNVERIFIED."""
    import os

    from textvocab import SentencePieceTokenizer

    adapter, tok_path, new_ids, _groups = load_real_model(cfg, device)
    lm = adapter.lm
    load_trainable_state_dict(lm, torch.load(os.path.join(ckpt_dir, "trainable.pt"), map_location=device), new_ids)
    merge_foreign_lora(lm)
    from param_lora import merge_param_lora
    merge_param_lora(lm)
    for p in lm.parameters():
        p.requires_grad = False
    lm.eval()
    mimi = adapter.info.get_mimi(device=device)
    mimi.eval()
    return lm, mimi, SentencePieceTokenizer(tok_path), new_ids


# --- streaming generation ---------------------------------------------------------

def _frame_generator_base():
    from genloop import FrameGenerator

    return FrameGenerator


class MoshiFrameGenerator(_frame_generator_base()):  # pragma: no cover - GPU only
    """The real LMGen behind genloop.FrameGenerator. UNVERIFIED (never run).

    Read from source (2026-11-07):
      * Kyutai `LMGen(lm, use_sampling, temp, temp_text, top_k, top_k_text, ...,
        on_text_hook, on_text_logits_hook)`; `step(input_tokens[B, 8, 1])` returns
        `[B, 1 + dep_q, 1]` delayed by max(delays) = 1 frame, or None while
        warming up. `on_text_logits_hook(text_logits)` runs just before the text
        token is sampled from that same tensor, so editing it in place forces
        or bans tokens; `on_text_hook(token[B])` gets the sampled token.
      * PersonaPlex `LMGen(lm, device, use_sampling, temp, temp_text, top_k,
        top_k_text, ..., text_prompt_tokens, ...)`; `step(input_tokens=,
        moshi_tokens=, text_token=)` forces the text token at this frame; no
        logits hook, but `step` calls `self.process_transformer_output(
        transformer_out, text_logits, ...)`, so a subclass can edit the logits
        (bans) first. The emitted token is read back from the state cache at
        (offset - 1). At offset 0, step consumes the frame and samples nothing.
      * `mimi.encode([B, 1, samples]) -> [B, 8, frames]`, `mimi.decode(codes)
        -> [B, 1, samples]`, `mimi.sample_rate`, `mimi.frame_rate`.
    """

    def __init__(self, lm: nn.Module, mimi: nn.Module, tok, gcfg: Dict[str, Any], device: str = "cuda"):
        import inspect

        from moshi.models import LMGen  # type: ignore

        self.lm, self.mimi, self.tok, self.device = lm, mimi, tok, device
        self.flavor = "personaplex" if "text_token" in inspect.signature(LMGen.step).parameters else "kyutai"
        temp_text = float(gcfg.get("temperature_text", 0.0))
        # temperature 0 = (near-)greedy text; audio keeps sampling, greedy audio degrades Moshi.
        kw = dict(use_sampling=True, temp=float(gcfg.get("temperature_audio", 0.8)), temp_text=max(temp_text, 1e-3),
                  top_k=int(gcfg.get("top_k_audio", 250)), top_k_text=int(gcfg.get("top_k_text", 25)))
        self._forced: Optional[int] = None
        self._banned: List[int] = []
        self._last_text: Optional[int] = None
        gen_self = self
        if self.flavor == "kyutai":
            self.lm_gen = LMGen(lm, on_text_logits_hook=self._edit_logits, on_text_hook=self._on_text, **kw)
        else:
            class _HookedLMGen(LMGen):  # type: ignore[misc, valid-type]
                def process_transformer_output(self_, transformer_out, text_logits, *a, **k):
                    gen_self._edit_logits(text_logits)
                    return super().process_transformer_output(transformer_out, text_logits, *a, **k)

            # sample_rate / frame_rate / audio_silence_frame_cnt as NVIDIA's
            # offline.py passes them (the defaults assume 32 kHz audio and
            # break a .wav voice prompt with 24 kHz Mimi).
            self.lm_gen = _HookedLMGen(lm, device=device, sample_rate=int(mimi.sample_rate),
                                       frame_rate=mimi.frame_rate,
                                       audio_silence_frame_cnt=int(0.5 * mimi.frame_rate), **kw)
        frame = int(round(mimi.sample_rate / mimi.frame_rate))
        with torch.no_grad():
            self.silence = mimi.encode(torch.zeros(1, 1, frame, device=device))[:, :, :1]  # [1, 8, 1]
        self.voice_prompt = gcfg.get("voice_prompt")  # PersonaPlex .pt / .wav voice prompt
        self.native_prompt = self.flavor == "personaplex" and bool(self.voice_prompt)

    # hooks
    def _edit_logits(self, text_logits: torch.Tensor) -> None:
        if self.track_ids is not None:
            # What the model wanted before any forcing/banning: lets the report
            # separate "never considered <tool>" from "considered it but lost".
            p = torch.softmax(text_logits.reshape(-1, text_logits.shape[-1])[0].float(), -1)
            t = self.track_ids
            self._probe = {"p_tool": float(p[t["tool"]]), "p_pad": float(p[t["pad"]]),
                           "p_epad": float(p[t["epad"]]), "argmax": int(p.argmax())}
        if self._forced is not None:
            text_logits.fill_(float("-inf"))
            text_logits[..., self._forced] = 0.0
        elif self._banned:
            text_logits[..., self._banned] = float("-inf")

    def _on_text(self, token: torch.Tensor) -> None:
        self._last_text = int(token[0])

    # FrameGenerator
    def start(self, ex: dict, ep: dict, gcfg: dict) -> None:
        import contextlib

        torch.manual_seed(int(ep.get("seed", 0)) + int(gcfg.get("seed_offset", 0)))
        self._stack = contextlib.ExitStack()
        self._stack.enter_context(torch.no_grad())
        self._stack.enter_context(self.lm_gen.streaming(1))
        self._codes: List[torch.Tensor] = []
        self._probe = None
        if self.flavor == "personaplex" and not self.native_prompt:
            # PersonaPlex's LMGen does not run the model on its very first
            # step: it overwrites that frame with the initial token and returns
            # None. Spend that step on a dummy frame here, so loop frame f lands
            # exactly where forward_train put frame f in training (otherwise the
            # first prompt token is silently lost and everything shifts by one).
            neg = torch.full((1, 8, 1), -1, dtype=torch.long, device=self.device)
            from textvocab import PAD_ID
            assert self.lm_gen.step(input_tokens=neg, moshi_tokens=neg, text_token=PAD_ID) is None
            assert int(self.lm_gen._streaming_state.offset) == 1, "LMGen priming step did not advance"
        if self.native_prompt:
            P = int(ex["meta"].get("prompt_frames", 0))
            prompt = [int(t) for t in ex["text"][:P] if int(t) > 3]
            self.lm_gen.text_prompt_tokens = prompt or None
            if str(self.voice_prompt).endswith(".pt"):
                self.lm_gen.load_voice_prompt_embeddings(self.voice_prompt)
            else:
                self.lm_gen.load_voice_prompt(self.voice_prompt)
            self.lm_gen.step_system_prompts(self.mimi)

    def step(self, frame, caller_codes, forced, banned, elle_codes=None) -> int:
        from textvocab import PAD_ID

        self._forced, self._banned, self._last_text, self._probe = forced, list(banned), None, None
        if self.flavor == "kyutai":
            # Kyutai's LMGen takes only the user's codes; it has no zero-token
            # input, so -1 frames are fed as silence (unused path).
            codes = self.silence if caller_codes is None or (np.asarray(caller_codes) < 0).any() else \
                torch.as_tensor(np.asarray(caller_codes), device=self.device).view(1, -1, 1)
            out = self.lm_gen.step(codes)
            tk = self._last_text
            if tk is None:
                tk = forced if forced is not None else PAD_ID
        else:
            # -1 goes in as -1: the zero embedding, exactly what forward_train
            # saw on those frames (prompt, audio tail). Elle's codes are forced
            # when given (prompt frames; teacher mode), otherwise sampled.
            neg = torch.full((1, 8, 1), -1, dtype=torch.long, device=self.device)
            codes = neg if caller_codes is None else \
                torch.as_tensor(np.asarray(caller_codes, dtype=np.int64), device=self.device).view(1, -1, 1)
            moshi = None if elle_codes is None else \
                torch.as_tensor(np.asarray(elle_codes, dtype=np.int64), device=self.device).view(1, -1, 1)
            st = self.lm_gen._streaming_state
            if int(st.offset) == 0:
                raise RuntimeError("LMGen was not primed in start(); frame alignment would be off by one")
            out = self.lm_gen.step(input_tokens=codes, moshi_tokens=moshi, text_token=forced)
            CT = st.cache.shape[2]
            tk = int(st.cache[0, 0, (int(st.offset) - 1) % CT])
        if out is not None:
            c = out[:, 1:9].detach()
            # Frames where Elle's codes were forced to -1 (the prompt) decode as silence.
            self._codes.append(torch.where(c < 0, self.silence.to(c.dtype).expand_as(c), c))
        if self.frame_stats is not None:
            self.frame_stats.append(self._probe)
        return tk

    def finish(self) -> None:
        self._stack.close()

    def decode_audio(self):
        if not self._codes:
            return None
        codes = torch.cat(self._codes, dim=-1)  # [1, 8, T]
        with torch.no_grad():
            pcm = self.mimi.decode(codes)
        return pcm[0, 0].float().cpu().numpy(), int(self.mimi.sample_rate)
