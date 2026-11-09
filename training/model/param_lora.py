"""LoRA on the weights themselves, for models that read weights directly.

`tiny_model.LoRALinear` adds LoRA by replacing an nn.Linear with a wrapper, which
only works when the model *calls* that layer. PersonaPlex mostly does not:

  * its feed-forward gating hands `linear_in.weight` and `linear_out.weight`
    straight to a fused kernel;
  * its attention stores the q/k/v projection as a bare `in_proj_weight`
    parameter (no layer object at all) and multiplies it with F.linear;
  * its depth transformer reads `out_proj.weight` directly (`multi_linear`).

So a wrapped layer is either never seen or breaks the model ("'LoRALinear'
object has no attribute 'weight'"). Here LoRA is a PyTorch parametrization of
the weight tensor instead: every read of the weight, fused or not, returns
W + scaling * B @ A, and gradients flow to A and B only.

Same scheme as LoRALinear: A Kaiming-uniform, B zeros (training starts exactly
at the base model), and the same `scaling` meaning. The parameters are named
lora_A / lora_B, so checkpointing that collects those names works unchanged.

Only attention is targeted (q/k/v and the output projection), the classic LoRA
placement. Adapting the feed-forward weights too would keep a merged copy of
them for the backward pass, roughly 13 GB more on the 7B model.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn.utils import parametrize


class LoRAWeight(nn.Module):
    """Parametrization: W -> W + scaling * B @ A."""

    def __init__(self, weight: torch.Tensor, rank: int, scaling: float):
        super().__init__()
        out_f, in_f = weight.shape
        self.lora_A = nn.Parameter(torch.empty(rank, in_f, device=weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, device=weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = scaling

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        return W + (self.scaling * (self.lora_B @ self.lora_A)).to(W.dtype)


def attention_weights(model: nn.Module, include_prefixes: Tuple[str, ...] = ()) -> List[Tuple[str, nn.Module, str]]:
    """(qualified name, owning module, attribute) for every attention projection:
    `in_proj_weight` on the attention module, `weight` on its `out_proj`."""
    found = []
    for name, mod in model.named_modules():
        if include_prefixes and not (name + ".").startswith(include_prefixes):
            continue
        w = getattr(mod, "in_proj_weight", None)
        if isinstance(w, nn.Parameter) and w.dim() == 2:
            found.append((f"{name}.in_proj_weight", mod, "in_proj_weight"))
            out = getattr(mod, "out_proj", None)
            if isinstance(out, nn.Linear):
                found.append((f"{name}.out_proj.weight", out, "weight"))
    return found


def apply_param_lora(model: nn.Module, rank: int, scaling: float,
                     include_prefixes: Tuple[str, ...] = ()) -> List[str]:
    """Attach LoRA to every attention projection weight. Returns the names."""
    names = []
    for qual, mod, attr in attention_weights(model, tuple(include_prefixes)):
        if parametrize.is_parametrized(mod, attr):
            continue
        getattr(mod, attr).requires_grad = False
        parametrize.register_parametrization(mod, attr, LoRAWeight(getattr(mod, attr), rank, scaling))
        names.append(qual)
    return names


@torch.no_grad()
def merge_param_lora(model: nn.Module) -> int:
    """Bake every LoRA parametrization into its weight and remove it, leaving a
    plain model for streaming inference. Returns the number merged."""
    n = 0
    for mod in list(model.modules()):
        if not parametrize.is_parametrized(mod):
            continue
        for attr in list(mod.parametrizations.keys()):
            if any(isinstance(p, LoRAWeight) for p in mod.parametrizations[attr]):
                parametrize.remove_parametrizations(mod, attr, leave_parametrized=True)
                n += 1
    return n
