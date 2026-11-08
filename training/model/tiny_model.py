"""TinyDuplexLM: a CPU-sized stand-in with Moshi's *interface*, plus LoRA and
the special-token vocabulary extension.

Shape conventions follow Moshi's LMModel (lm.py): the text stream has
`text_card + 1` input embeddings (the last row is the initial token), the text
head outputs `text_card` logits, each audio codebook has `card + 1` input rows
(last = initial), and -1 (zero token) contributes no embedding. A temporal
transformer runs over frames; a small depth transformer predicts Elle's 8
codebooks per frame conditioned on the temporal state and that frame's text
token. Simplifications vs Moshi (fine for a dry run, wrong for real quality):
no acoustic-codebook delays, no KV cache, tiny sizes, learned positions.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

N_CB = 8  # codebooks per channel


# --- LoRA ---------------------------------------------------------------------

class LoRALinear(nn.Module):
    """y = W x + scaling * B(A(x)); W frozen, A Kaiming-uniform, B zeros
    (so training starts exactly at the base model). Same scheme as
    moshi/modules/lora.py and moshi-finetune's initialize_lora_parameters."""

    def __init__(self, base: nn.Linear, rank: int, scaling: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.scaling = scaling
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(self.dropout(x)))

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        lin = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        lin.weight.copy_(self.base.weight + self.scaling * self.lora_B.weight @ self.lora_A.weight)
        if self.base.bias is not None:
            lin.bias.copy_(self.base.bias)
        return lin


def apply_lora(model: nn.Module, rank: int, scaling: float, target: Iterable[str], dropout: float = 0.0,
               include_prefixes: Iterable[str] = ()) -> List[str]:
    """Wrap nn.Linear layers with LoRA. A layer is wrapped when its last name
    component is in `target` (or `target` contains "*"), and, if
    `include_prefixes` is given, its qualified name starts with one of them.
    Returns the wrapped names."""
    target = tuple(target)
    prefixes = tuple(include_prefixes)
    wrapped = []
    for name, module in list(model.named_modules()):
        for cname, child in list(module.named_children()):
            full = f"{name}.{cname}" if name else cname
            name_ok = "*" in target or full.split(".")[-1] in target
            prefix_ok = not prefixes or full.startswith(prefixes)
            if isinstance(child, nn.Linear) and name_ok and prefix_ok:
                setattr(module, cname, LoRALinear(child, rank, scaling, dropout))
                wrapped.append(full)
    return wrapped


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if ".lora_A." in k or ".lora_B." in k}


# --- vocab extension -------------------------------------------------------------

class RowMask:
    """Gradient hook: only rows in `rows` of a weight receive gradient, so a
    frozen pretrained embedding can learn just its new special-token rows."""

    def __init__(self, n_rows: int, rows: List[int]):
        self.mask = torch.zeros(n_rows, 1)
        self.mask[rows] = 1.0

    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        return grad * self.mask.to(grad.device, grad.dtype).view(-1, *([1] * (grad.dim() - 1)))


@torch.no_grad()
def _grow_embedding(emb: nn.Embedding, old_card: int, n_new: int, init_rows: Optional[torch.Tensor]) -> nn.Embedding:
    """Rows: [0, old_card) real tokens, old_card = initial token. New layout:
    [0, old_card) unchanged, [old_card, old_card+n_new) new, old_card+n_new = initial."""
    w = emb.weight
    assert w.shape[0] == old_card + 1, (w.shape, old_card)
    new = nn.Embedding(old_card + n_new + 1, w.shape[1]).to(w.device, w.dtype)
    new.weight[:old_card] = w[:old_card]
    new.weight[old_card + n_new] = w[old_card]
    new.weight[old_card:old_card + n_new] = init_rows if init_rows is not None else w[:old_card].mean(0, keepdim=True)
    return new


@torch.no_grad()
def _grow_head(lin: nn.Linear, n_new: int) -> nn.Linear:
    new = nn.Linear(lin.in_features, lin.out_features + n_new, bias=lin.bias is not None).to(lin.weight.device, lin.weight.dtype)
    new.weight[: lin.out_features] = lin.weight
    new.weight[lin.out_features:] = lin.weight.mean(0, keepdim=True)
    if lin.bias is not None:
        new.bias[: lin.out_features] = lin.bias
        # Start new tokens unlikely so the base behaviour is unchanged at step 0.
        new.bias[lin.out_features:] = lin.bias.min() - 4.0
    return new


# --- model ----------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, dim: int, heads: int, ff: int, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        self.norm2 = nn.LayerNorm(dim)
        self.ff_in = nn.Linear(dim, ff, bias=False)
        self.ff_out = nn.Linear(ff, dim, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        h = self.norm1(x)
        q, k, v = (m(h).view(B, T, self.heads, D // self.heads).transpose(1, 2) for m in (self.q, self.k, self.v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, D))
        return x + self.ff_out(F.gelu(self.ff_in(self.norm2(x))))


class TinyDuplexLM(nn.Module):
    def __init__(self, text_card: int, card: int = 64, dim: int = 64, layers: int = 2, heads: int = 4,
                 depth_dim: int = 64, depth_layers: int = 1, max_frames: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.text_card, self.card, self.dim, self.max_frames = text_card, card, dim, max_frames
        self.text_emb = nn.Embedding(text_card + 1, dim)
        self.audio_emb = nn.ModuleList([nn.Embedding(card + 1, dim) for _ in range(2 * N_CB)])
        self.pos = nn.Embedding(max_frames, dim)
        self.blocks = nn.ModuleList([Block(dim, heads, 4 * dim, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.text_linear = nn.Linear(dim, text_card, bias=True)
        # depth transformer over Elle's 8 codebooks of one frame
        self.depth_in = nn.Linear(dim, depth_dim, bias=False)
        self.depth_text_emb = nn.Embedding(text_card + 1, depth_dim)
        self.depth_audio_emb = nn.ModuleList([nn.Embedding(card + 1, depth_dim) for _ in range(N_CB - 1)])
        self.depth_pos = nn.Embedding(N_CB, depth_dim)
        self.depth_blocks = nn.ModuleList([Block(depth_dim, 2, 4 * depth_dim) for _ in range(depth_layers)])
        self.depth_norm = nn.LayerNorm(depth_dim)
        self.audio_heads = nn.ModuleList([nn.Linear(depth_dim, card, bias=False) for _ in range(N_CB)])

    # ids
    @property
    def text_initial_token_id(self) -> int:
        return self.text_card

    @property
    def initial_token_id(self) -> int:
        return self.card

    @staticmethod
    def _emb(emb: nn.Embedding, ids: torch.Tensor) -> torch.Tensor:
        """Embedding where -1 (zero token) contributes a zero vector."""
        keep = (ids >= 0).unsqueeze(-1)
        return emb(ids.clamp(min=0)) * keep

    def temporal(self, text: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        """text [B,T], audio [B,16,T] -> hidden [B,T,D]; position t sees frames < t."""
        B, T = text.shape
        assert T <= self.max_frames, (T, self.max_frames)
        t_in = torch.cat([torch.full((B, 1), self.text_initial_token_id, dtype=text.dtype, device=text.device),
                          text[:, :-1]], dim=1)
        a_in = torch.cat([torch.full((B, audio.shape[1], 1), self.initial_token_id, dtype=audio.dtype, device=audio.device),
                          audio[:, :, :-1]], dim=2)
        x = self._emb(self.text_emb, t_in)
        for k in range(audio.shape[1]):
            x = x + self._emb(self.audio_emb[k], a_in[:, k])
        x = x + self.pos(torch.arange(T, device=text.device))[None]
        for b in self.blocks:
            x = b(x)
        return self.norm(x)

    def depth(self, h: torch.Tensor, text_t: torch.Tensor, elle_audio_t: torch.Tensor) -> torch.Tensor:
        """Teacher-forced depth step. h [N,D], text_t [N], elle_audio_t [N,8] -> logits [N,8,card]."""
        N = h.shape[0]
        steps = [self.depth_in(h) + self._emb(self.depth_text_emb, text_t)]
        for k in range(1, N_CB):
            steps.append(self._emb(self.depth_audio_emb[k - 1], elle_audio_t[:, k - 1]))
        x = torch.stack(steps, dim=1) + self.depth_pos.weight[None]
        for b in self.depth_blocks:
            x = b(x)
        x = self.depth_norm(x)
        return torch.stack([self.audio_heads[k](x[:, k]) for k in range(N_CB)], dim=1)

    def forward(self, text: torch.Tensor, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Training forward. Returns text_logits [B,T,text_card] and Elle audio
        logits [B,8,T,card], aligned with the input frames (no shifting needed)."""
        h = self.temporal(text, audio)
        B, T, D = h.shape
        elle = audio[:, :N_CB].permute(0, 2, 1).reshape(B * T, N_CB)
        al = self.depth(h.reshape(B * T, D), text.reshape(B * T), elle)
        return self.text_linear(h), al.view(B, T, N_CB, -1).permute(0, 2, 1, 3)

    @torch.no_grad()
    def sample_depth(self, h: torch.Tensor, text_t: torch.Tensor) -> torch.Tensor:
        """Greedy autoregressive depth decoding for one frame. h [N,D] -> codes [N,8]."""
        codes = torch.full((h.shape[0], N_CB), -1, dtype=torch.long, device=h.device)
        for k in range(N_CB):
            logits = self.depth(h, text_t, codes)[:, k]
            codes[:, k] = logits.argmax(-1)
        return codes

    # --- special tokens ---------------------------------------------------------
    def resize_text_vocab(self, n_new: int, init_rows: Optional[torch.Tensor] = None) -> List[int]:
        """Append n_new text tokens (the four markers). Returns their ids."""
        old = self.text_card
        self.text_emb = _grow_embedding(self.text_emb, old, n_new, init_rows)
        self.depth_text_emb = _grow_embedding(self.depth_text_emb, old, n_new, None)
        self.text_linear = _grow_head(self.text_linear, n_new)
        self.text_card = old + n_new
        return list(range(old, old + n_new))


def build_tiny(cfg: dict, text_card: int) -> TinyDuplexLM:
    m = cfg.get("tiny", {})
    return TinyDuplexLM(text_card=text_card, card=int(m.get("card", 64)), dim=int(m.get("dim", 64)),
                        layers=int(m.get("layers", 2)), heads=int(m.get("heads", 4)),
                        depth_dim=int(m.get("depth_dim", 64)), depth_layers=int(m.get("depth_layers", 1)),
                        max_frames=int(m.get("max_frames", 2048)), dropout=float(m.get("dropout", 0.0)))


def setup_trainable(model: TinyDuplexLM, new_ids: List[int], lora_cfg: dict) -> Dict[str, List[nn.Parameter]]:
    """Freeze everything, add LoRA, and unfreeze only the new tokens' rows
    (embedding rows via a gradient mask; head rows likewise). Returns param
    groups: 'lora' (weight decay) and 'new_tokens' (no weight decay)."""
    for p in model.parameters():
        p.requires_grad = False
    wrapped = []
    if lora_cfg.get("enable", True):
        wrapped = apply_lora(model, int(lora_cfg.get("rank", 8)), float(lora_cfg.get("scaling", 2.0)),
                             lora_cfg.get("target", ["q", "k", "v", "o", "ff_in", "ff_out"]),
                             float(lora_cfg.get("dropout", 0.0)))
    groups: Dict[str, List[nn.Parameter]] = {"lora": [], "new_tokens": []}
    for n, p in model.named_parameters():
        if ".lora_A." in n or ".lora_B." in n:
            p.requires_grad = True
            groups["lora"].append(p)
    if new_ids and lora_cfg.get("train_new_token_rows", True):
        for w in (model.text_emb.weight, model.depth_text_emb.weight, model.text_linear.weight):
            w.requires_grad = True
            w.register_hook(RowMask(w.shape[0], new_ids))
            groups["new_tokens"].append(w)
        b = model.text_linear.bias
        b.requires_grad = True
        b.register_hook(lambda g, ids=list(new_ids): g * torch.zeros_like(g).index_fill_(0, torch.tensor(ids), 1.0))
        groups["new_tokens"].append(b)
    model._lora_wrapped = wrapped  # type: ignore[attr-defined]
    return groups


def trainable_state_dict(model: TinyDuplexLM, new_ids: List[int]) -> Dict[str, torch.Tensor]:
    """What a checkpoint must carry on top of the frozen base: LoRA A/B and the
    new tokens' rows."""
    sd = lora_state_dict(model)
    ids = torch.tensor(new_ids, dtype=torch.long)
    if new_ids:
        sd["new_tokens.text_emb"] = model.text_emb.weight.detach()[ids].cpu()
        sd["new_tokens.depth_text_emb"] = model.depth_text_emb.weight.detach()[ids].cpu()
        sd["new_tokens.text_linear.weight"] = model.text_linear.weight.detach()[ids].cpu()
        sd["new_tokens.text_linear.bias"] = model.text_linear.bias.detach()[ids].cpu()
    return sd


@torch.no_grad()
def load_trainable_state_dict(model: TinyDuplexLM, sd: Dict[str, torch.Tensor], new_ids: List[int]) -> None:
    lora = {k: v for k, v in sd.items() if not k.startswith("new_tokens.")}
    missing, unexpected = model.load_state_dict(lora, strict=False)
    assert not unexpected, unexpected
    if new_ids and "new_tokens.text_emb" in sd:
        ids = torch.tensor(new_ids, dtype=torch.long)
        model.text_emb.weight[ids] = sd["new_tokens.text_emb"]
        model.depth_text_emb.weight[ids] = sd["new_tokens.depth_text_emb"]
        model.text_linear.weight[ids] = sd["new_tokens.text_linear.weight"]
        model.text_linear.bias[ids] = sd["new_tokens.text_linear.bias"]
