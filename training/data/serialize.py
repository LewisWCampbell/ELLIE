"""Episode -> Elle's inner-monologue text stream (training/CONTRACT.md).

Standard library only, no sibling imports, so the trainer can import it from
anywhere (`from training.data.serialize import serialize_episode` from the repo
root, or `import serialize` with training/data on sys.path).

The stream is Elle's channel only. Each segment carries the time it belongs at
and whether it is *forced* (injected, excluded from the loss) or *predicted*:

    speech       Elle's words                                     predicted
    tool_call    <tool>{"id":"c1","name":...,"args":{...}}</tool>  predicted, at the call's t
    tool_result  <result id="c1">FREE: ...</result>               forced,    at the result's t
    caller       caller words (only with include_caller=True)      forced

`<tool>`, `</tool>`, `<result id=` and `</result>` are the four special tokens.
The tool-call JSON is compact (no spaces) with keys in the order id, name,
args, and args keys in the order they appear in the episode.

    segs = serialize_episode(ep)                 # utterance-level segments
    segs = serialize_episode(ep, word_level=True)  # one segment per word, times spread over dur
    text, mask = to_text_and_mask(segs)          # mask[i] is True where char i is predicted
    calls = extract_tool_calls(model_output_text)  # parse calls back out of a generated stream
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, NamedTuple, Sequence, Tuple

TOOL_OPEN = "<tool>"
TOOL_CLOSE = "</tool>"
RESULT_OPEN = "<result id="
RESULT_CLOSE = "</result>"
SPECIAL_TOKENS: Tuple[str, ...] = (TOOL_OPEN, TOOL_CLOSE, RESULT_OPEN, RESULT_CLOSE)


class Segment(NamedTuple):
    t: float          # start time in seconds from episode start
    end: float        # end time (t for zero-duration markers)
    text: str
    is_forced: bool   # True: injected/context, masked out of the loss
    kind: str         # "speech" | "tool_call" | "tool_result" | "caller"
    event_index: int  # index into ep["events"]


def tool_call_text(ev: Dict[str, Any]) -> str:
    body = json.dumps({"id": ev["id"], "name": ev["name"], "args": ev["args"]},
                      ensure_ascii=False, separators=(",", ":"))
    return f"{TOOL_OPEN}{body}{TOOL_CLOSE}"


def tool_result_text(ev: Dict[str, Any]) -> str:
    return f'{RESULT_OPEN}"{ev["id"]}">{ev["content"]}{RESULT_CLOSE}'


def _words(ev: Dict[str, Any], i: int, forced: bool, kind: str) -> List[Segment]:
    words = ev["text"].split()
    if not words:
        return []
    step = ev["dur"] / len(words)
    return [Segment(round(ev["t"] + k * step, 3), round(ev["t"] + (k + 1) * step, 3), w, forced, kind, i)
            for k, w in enumerate(words)]


def serialize_episode(ep: Dict[str, Any], *, word_level: bool = False,
                      include_caller: bool = False) -> List[Segment]:
    """Segments of Elle's text stream in time order.

    Ties at the same `t` keep event order; with `word_level`, a call or result
    that lands mid-utterance is interleaved between the words around it, which
    is where it sits in a time-aligned (12.5 Hz) Moshi text stream.
    """
    segs: List[Tuple[float, int, int, Segment]] = []
    for i, ev in enumerate(ep["events"]):
        typ = ev["type"]
        if typ == "speech" and (ev["ch"] == "elle" or include_caller):
            forced = ev["ch"] != "elle"
            kind = "speech" if ev["ch"] == "elle" else "caller"
            if word_level:
                for k, s in enumerate(_words(ev, i, forced, kind)):
                    segs.append((s.t, i, k, s))
            else:
                segs.append((ev["t"], i, 0, Segment(ev["t"], round(ev["t"] + ev["dur"], 3), ev["text"],
                                                    forced, kind, i)))
        elif typ == "tool_call":
            segs.append((ev["t"], i, 0, Segment(ev["t"], ev["t"], tool_call_text(ev), False, "tool_call", i)))
        elif typ == "tool_result":
            segs.append((ev["t"], i, 0, Segment(ev["t"], ev["t"], tool_result_text(ev), True, "tool_result", i)))
    segs.sort(key=lambda x: (x[0], x[1], x[2]))
    return [s for *_, s in segs]


def to_text_and_mask(segments: Sequence[Segment], sep: str = " ") -> Tuple[str, List[bool]]:
    """Join segments into one string plus a per-character loss mask (True =
    predicted, in the loss). Separators take the mask of the segment they precede."""
    parts: List[str] = []
    mask: List[bool] = []
    for k, s in enumerate(segments):
        if k:
            parts.append(sep)
            mask.extend([not s.is_forced] * len(sep))
        parts.append(s.text)
        mask.extend([not s.is_forced] * len(s.text))
    return "".join(parts), mask


def forced_spans(text: str, mask: Sequence[bool]) -> List[Tuple[int, int]]:
    """[start, end) character ranges that are forced (not predicted)."""
    out, start = [], None
    for i, m in enumerate(list(mask) + [True]):
        if not m and start is None:
            start = i
        elif m and start is not None:
            out.append((start, i))
            start = None
    return out


_CALL_RE = re.compile(re.escape(TOOL_OPEN) + r"(.*?)" + re.escape(TOOL_CLOSE), re.S)
_RESULT_RE = re.compile(re.escape(RESULT_OPEN) + r'"([^"]*)">(.*?)' + re.escape(RESULT_CLOSE), re.S)


def extract_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Parse every <tool>...</tool> span. Unparseable bodies come back as
    {"raw": body, "error": "..."} so an evaluator can count them as wrong calls."""
    out = []
    for m in _CALL_RE.finditer(text):
        body = m.group(1)
        try:
            obj = json.loads(body)
            if not isinstance(obj, dict):
                raise ValueError("not an object")
            out.append(obj)
        except ValueError as e:
            out.append({"raw": body, "error": str(e)})
    return out


def extract_results(text: str) -> List[Tuple[str, str]]:
    return [(m.group(1), m.group(2)) for m in _RESULT_RE.finditer(text)]


def strip_markers(text: str) -> str:
    """The spoken words only: removes tool and result spans."""
    return re.sub(r"\s+", " ", _RESULT_RE.sub("", _CALL_RE.sub("", text))).strip()


if __name__ == "__main__":  # quick look: python3 serialize.py file.jsonl [n]
    import sys
    path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    with open(path, encoding="utf-8") as f:
        for _, line in zip(range(n), f):
            ep = json.loads(line)
            for s in serialize_episode(ep):
                print(f"{s.t:7.2f} {'F' if s.is_forced else 'P'} {s.kind:11} {s.text}")
            print()
