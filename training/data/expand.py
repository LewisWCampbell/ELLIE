#!/usr/bin/env python3
"""Optional paraphrase/expansion pass: an LLM rewrites low-stakes utterances for
lexical diversity, leaving tool calls, results, labels and timing untouched.

    python3 training/data/expand.py in.jsonl out.jsonl

With no API key configured this is a byte-for-byte passthrough (and says so on
stderr), so pipelines can call it unconditionally. To enable it:

    pip install anthropic          # optional; see requirements.txt
    export ANTHROPIC_API_KEY=...   # or any credential the Anthropic SDK resolves
    export EXPAND_MODEL=claude-opus-5-5   # optional override

What may change: only speech whose `act` is in REWRITABLE_ACTS (greetings,
fillers, acknowledgements, chit-chat, goodbyes). Readbacks, confirmations,
claims, times, names, emails, compliance wording and anything a label points at
are never sent to the model. A rewrite is accepted only if it keeps the language,
stays within +-30% of the word count (so the estimated durations still hold),
contains no serialization markers, and the episode still passes validate.py;
otherwise the original episode is kept.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validate import MARKERS, validate_episode  # noqa: E402

DEFAULT_MODEL = "claude-opus-5-5"
REWRITABLE_ACTS = {"greeting", "intent", "ask_about", "ramble", "chat", "decline", "thanks", "bye",
                   "filler", "ack", "ask_time", "pitch", "info", "explain"}

PROMPT = (
    "You are paraphrasing lines of a phone conversation to add variety to a training set. "
    "Elle is a warm AI assistant who speaks in one or two short spoken sentences, with contractions "
    "and no lists or markup. Rewrite each line so it means the same thing in the same language, "
    "with a similar number of words. Never add names, times, dates, email addresses or facts, and "
    "never say anything about saving or noting information. Return only a JSON array of objects "
    '{"i": <same i>, "text": <rewritten line>} for every input line.\n\nLines:\n'
)


def has_credentials() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def make_client():
    """An Anthropic client, or None when no key or SDK is available."""
    if not has_credentials():
        return None
    try:
        import anthropic  # type: ignore
    except ImportError:
        print("expand.py: anthropic SDK not installed; passing episodes through unchanged", file=sys.stderr)
        return None
    return anthropic.Anthropic()


def _ask(client, model: str, lines: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    content = PROMPT + json.dumps(lines, ensure_ascii=False)
    try:
        resp = client.beta.messages.create(
            model=model, max_tokens=16000, betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            messages=[{"role": "user", "content": content}])
    except TypeError:  # older SDK without the fallback parameter
        resp = client.messages.create(model=model, max_tokens=16000,
                                      messages=[{"role": "user", "content": content}])
    if getattr(resp, "stop_reason", None) == "refusal":
        return None
    text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text").strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1]
    try:
        out = json.loads(text)
    except ValueError:
        return None
    return out if isinstance(out, list) else None


def _protected(ep: Dict[str, Any]) -> set:
    lab = ep["labels"]
    idx = {c["event_index"] for c in lab.get("claim_spans", [])}
    for g in lab.get("write_gates", []):
        idx |= {g["readback_index"], g["confirm_index"]}
    return idx


def expand_episode(ep: Dict[str, Any], client=None, model: Optional[str] = None) -> Dict[str, Any]:
    """Return a (possibly) paraphrased copy of `ep`; `ep` itself when there is no client."""
    if client is None:
        return ep
    model = model or os.environ.get("EXPAND_MODEL", DEFAULT_MODEL)
    protected = _protected(ep)
    todo = [{"i": i, "speaker": ev["ch"], "text": ev["text"]} for i, ev in enumerate(ep["events"])
            if ev["type"] == "speech" and ev.get("act") in REWRITABLE_ACTS and i not in protected
            and not ev.get("interrupted")]
    if not todo:
        return ep
    try:
        got = _ask(client, model, todo)
    except Exception as e:  # network/API errors never break the pipeline
        print(f"expand.py: {ep['id']}: {e!r}; keeping original", file=sys.stderr)
        return ep
    if not got:
        return ep
    new = json.loads(json.dumps(ep))
    orig = {d["i"]: d["text"] for d in todo}
    changed = 0
    for d in got:
        i, text = d.get("i"), d.get("text")
        if i not in orig or not isinstance(text, str) or not text.strip():
            continue
        n0, n1 = len(orig[i].split()), len(text.split())
        if not (0.7 * n0 <= n1 <= 1.3 * n0 + 1) or any(m in text for m in MARKERS):
            continue
        new["events"][i]["text"] = text.strip()
        changed += 1
    if not changed or validate_episode(new):
        return ep
    new["labels"]["expanded"] = {"model": model, "rewritten": changed}
    return new


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--limit", type=int, default=0, help="only expand the first N episodes (rest pass through)")
    a = ap.parse_args(argv)
    client = make_client()
    if client is None:
        print("expand.py: no API key configured; passthrough (output == input)", file=sys.stderr)
    n = changed = 0
    with open(a.src, encoding="utf-8") as fi, open(a.dst, "w", encoding="utf-8") as fo:
        for line in fi:
            if not line.strip():
                continue
            if client is None:
                fo.write(line if line.endswith("\n") else line + "\n")
                n += 1
                continue
            ep = json.loads(line)
            out = expand_episode(ep, client) if (not a.limit or n < a.limit) else ep
            changed += out is not ep
            fo.write(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
            n += 1
    print(f"expand.py: {n} episodes, {changed} rewritten", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
