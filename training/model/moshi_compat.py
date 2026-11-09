"""One way to load a Moshi-family checkpoint, whichever `moshi` package is installed.

Two packages share the name `moshi` and load checkpoints differently:

  * Kyutai's (kyutai-labs/moshi) has `loaders.CheckpointInfo.from_hf_repo(repo)`,
    an object with `.tokenizer`, `.get_mimi(device=...)` and `.get_moshi(...)`.
  * NVIDIA's PersonaPlex fork has no CheckpointInfo. It exposes file-name
    constants (TEXT_TOKENIZER_NAME, MOSHI_NAME, MIMI_NAME) and the functions
    `get_mimi(filename, device)` and `get_moshi_lm(filename, ..., device, dtype)`.

The training code was first written against Kyutai's API, and the first run on
the GPU VM (PersonaPlex installed) failed with "module 'moshi.models.loaders'
has no attribute 'CheckpointInfo'". Every call site now goes through
`checkpoint(hf_repo)`, which returns an object with the Kyutai shape either way.

Files are fetched with `hf_hub_download`, which reads the local cache, so this
works offline (HF_HUB_OFFLINE=1) once vm_setup.sh has downloaded the weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class _PersonaPlexCheckpoint:
    """CheckpointInfo-shaped view of a PersonaPlex checkpoint."""

    def __init__(self, loaders: Any, hf_repo: str):
        from huggingface_hub import hf_hub_download  # type: ignore

        self._loaders = loaders
        self.hf_repo = hf_repo
        self._download = hf_hub_download
        self.tokenizer = Path(hf_hub_download(hf_repo, loaders.TEXT_TOKENIZER_NAME))

    def _file(self, name: str) -> Path:
        # Fetched on demand: prepare.py needs only the codec, not the 15 GB LM.
        return Path(self._download(self.hf_repo, name))

    def get_mimi(self, device: str = "cuda"):
        return self._loaders.get_mimi(self._file(self._loaders.MIMI_NAME), device=device)

    def get_moshi(self, device: str = "cuda", dtype: Any = None, **kwargs: Any):
        if kwargs:
            # Kyutai-only options (its built-in LoRA, gradient checkpointing
            # overrides). PersonaPlex's loader has no equivalent; refuse rather
            # than silently train without them.
            raise TypeError(f"PersonaPlex's get_moshi_lm does not accept {sorted(kwargs)}")
        extra = {"dtype": dtype} if dtype is not None else {}
        return self._loaders.get_moshi_lm(self._file(self._loaders.MOSHI_NAME), device=device, **extra)


def checkpoint(hf_repo: str, loaders: Any = None):
    """A checkpoint with `.tokenizer`, `.get_mimi(device=)` and `.get_moshi(device=, dtype=)`."""
    if loaders is None:
        from moshi.models import loaders  # type: ignore
    if hasattr(loaders, "CheckpointInfo"):
        return loaders.CheckpointInfo.from_hf_repo(hf_repo)
    return _PersonaPlexCheckpoint(loaders, hf_repo)


def flavour(loaders: Any = None) -> str:
    """'kyutai' or 'personaplex': which `moshi` package is installed."""
    if loaders is None:
        from moshi.models import loaders  # type: ignore
    return "kyutai" if hasattr(loaders, "CheckpointInfo") else "personaplex"
