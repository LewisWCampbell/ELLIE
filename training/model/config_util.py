"""YAML config loading with `extends:` (deep merge) and ${ENV} expansion.
Relative paths in a config resolve against that config file's directory."""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import yaml


def _merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str) -> Tuple[Dict[str, Any], str]:
    path = os.path.abspath(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = os.path.dirname(path)
    parent = cfg.pop("extends", None)
    if parent:
        pcfg, _ = load_config(os.path.join(base, parent))
        cfg = _merge(pcfg, cfg)
    return cfg, base


def resolve_path(p: str, base: str) -> str:
    p = os.path.expandvars(os.path.expanduser(str(p)))
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))
