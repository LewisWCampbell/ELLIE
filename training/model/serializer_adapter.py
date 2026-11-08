"""Thin adapter between the dataset's serializer and the trainer.

The trainer needs Elle's text stream as *timed segments*:

    {"kind": "speech" | "tool" | "result", "t": float, "dur": float | None,
     "text": str, "loss": bool, "id": str | None}

`loss` is False for forced (injected) spans, i.e. results. If
`training/data/serialize.py` exists and exposes `serialize_episode`, its output
is normalized into this shape; otherwise (or if its shape is not recognised)
the contract's serialization is applied here. Either way `contract_segments`
is computed, so a disagreement between the two is reported, not hidden.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from textvocab import RESULT_CLOSE, RESULT_OPEN, TOOL_CLOSE, TOOL_OPEN

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "data"))

Segment = Dict[str, Any]


def tool_call_text(ev: dict) -> str:
    payload = {"id": ev["id"], "name": ev["name"], "args": ev.get("args") or {}}
    return TOOL_OPEN + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + TOOL_CLOSE


def tool_result_text(ev: dict) -> str:
    return f'{RESULT_OPEN}"{ev["id"]}">{ev.get("content", "")}{RESULT_CLOSE}'


def contract_segments(ep: dict) -> List[Segment]:
    """CONTRACT.md 'Inner-monologue serialization', applied directly."""
    segs: List[Segment] = []
    for i, ev in enumerate(ep.get("events", [])):
        typ, ch = ev.get("type"), ev.get("ch")
        if typ == "speech" and ch == "elle":
            segs.append({"kind": "speech", "t": float(ev["t"]), "dur": float(ev.get("dur") or 0.0),
                         "text": ev.get("text", ""), "loss": True, "id": None, "event_index": i})
        elif typ == "tool_call":
            segs.append({"kind": "tool", "t": float(ev["t"]), "dur": None, "text": tool_call_text(ev),
                         "loss": True, "id": ev.get("id"), "event_index": i})
        elif typ == "tool_result":
            segs.append({"kind": "result", "t": float(ev["t"]), "dur": None, "text": tool_result_text(ev),
                         "loss": False, "id": ev.get("id"), "event_index": i})
    segs.sort(key=lambda s: (s["t"], s["event_index"]))
    return segs


# --- dataset serializer ------------------------------------------------------

_loaded: Optional[Tuple[Any, str]] = None


def _load_dataset_serializer():
    """Import training/data/serialize.py without putting the whole data dir on
    sys.path permanently (module names there could shadow ours)."""
    global _loaded
    if _loaded is not None:
        return _loaded
    path = os.path.join(DATA_DIR, "serialize.py")
    if not os.path.exists(path):
        _loaded = (None, "absent")
        return _loaded
    try:
        added = DATA_DIR not in sys.path
        if added:
            sys.path.insert(0, DATA_DIR)  # their module may import siblings (elle_time, ...)
        try:
            spec = importlib.util.spec_from_file_location("elle_dataset_serialize", path)
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)
        finally:
            if added:
                sys.path.remove(DATA_DIR)
        fn = getattr(mod, "serialize_episode", None)
        _loaded = (fn, "ok" if fn else "no serialize_episode")
    except Exception as e:  # pragma: no cover - depends on the other agent's code
        _loaded = (None, f"import failed: {e!r}")
    return _loaded


def dataset_serializer_status() -> str:
    return _load_dataset_serializer()[1]


_KIND_ALIASES = {"speech": "speech", "say": "speech", "text": "speech", "elle": "speech",
                 "tool": "tool", "tool_call": "tool", "call": "tool",
                 "result": "result", "tool_result": "result"}


def _normalize(obj: Any) -> Optional[List[Segment]]:
    """Accept the plausible shapes a serializer could return; None if unknown."""
    if isinstance(obj, dict):
        for key in ("segments", "spans", "items", "stream", "chunks"):
            if isinstance(obj.get(key), list):
                return _normalize(obj[key])
        return None
    if not isinstance(obj, list):
        return None
    out: List[Segment] = []
    for it in obj:
        if hasattr(it, "_asdict"):  # training/data/serialize.py returns NamedTuple Segments
            it = it._asdict()
        if not isinstance(it, dict) or "text" not in it:
            return None
        if str(it.get("kind", "")).lower() == "caller":
            continue  # Elle's stream only; the caller is heard through audio
        t = it.get("t", it.get("start"))
        if t is None:
            return None
        kind = _KIND_ALIASES.get(str(it.get("kind", it.get("type", ""))).lower())
        text = it["text"]
        if kind is None:
            kind = "tool" if text.startswith(TOOL_OPEN) else "result" if text.startswith(RESULT_OPEN) else "speech"
        if "loss" in it:
            loss = bool(it["loss"])
        elif "mask" in it:
            loss = bool(it["mask"])
        elif "is_forced" in it:
            loss = not bool(it["is_forced"])
        elif "forced" in it:
            loss = not bool(it["forced"])
        elif "loss_mask" in it and not isinstance(it["loss_mask"], list):
            loss = bool(it["loss_mask"])
        else:
            loss = kind != "result"
        dur = it.get("dur")
        if dur is None and it.get("end") is not None:
            dur = float(it["end"]) - float(t)
        out.append({"kind": kind, "t": float(t), "dur": None if dur is None else float(dur), "text": text,
                    "loss": loss, "id": it.get("id"), "event_index": it.get("event_index")})
    out.sort(key=lambda s: s["t"])
    return out


def episode_segments(ep: dict, prefer_dataset: bool = True) -> Tuple[List[Segment], str]:
    """Returns (segments, source) where source says which serializer was used."""
    ref = contract_segments(ep)
    if prefer_dataset:
        fn, status = _load_dataset_serializer()
        if fn is not None:
            try:
                segs = _normalize(fn(ep))
            except Exception as e:  # pragma: no cover
                return ref, f"contract (dataset serializer raised {e!r})"
            if segs is not None:
                mismatch = check_against_contract(segs, ref)
                return segs, "dataset" if not mismatch else f"dataset (differs from contract: {mismatch})"
            return ref, "contract (dataset serializer output shape not recognised)"
    return ref, "contract"


def check_against_contract(segs: List[Segment], ref: List[Segment]) -> str:
    """Empty string when the dataset's serialization agrees with the contract on
    the tool/result spans (text + loss flag); speech is allowed to differ."""
    a = [(s["kind"], s["text"], s["loss"]) for s in segs if s["kind"] != "speech"]
    b = [(s["kind"], s["text"], s["loss"]) for s in ref if s["kind"] != "speech"]
    if a == b:
        return ""
    for x, y in zip(a, b):
        if x != y:
            return f"first difference {x!r} vs {y!r}"
    return f"span count {len(a)} vs {len(b)}"
