"""Episode -> Elle's inner-monologue text stream (training/CONTRACT.md).

Standard library only, no sibling imports, so the trainer can import it from
anywhere (`from training.data.serialize import serialize_episode` from the repo
root, or `import serialize` with training/data on sys.path).

The stream is Elle's channel only. Each segment carries the time it belongs at
and whether it is *forced* (injected, excluded from the loss) or *predicted*:

    speech       Elle's words                                     predicted
    tool_call    <tool>check_availability 2026-11-12T14:00</tool>  predicted, at the call's t
    tool_result  <result id="c1">FREE: ...</result>               forced,    at the result's t
    caller       caller words (only with include_caller=True)      forced

`<tool>`, `</tool>`, `<result id=` and `</result>` are the four special tokens.

Two call forms (CONTRACT.md v3, point 4):

    compact (default)  <tool>NAME</tool> or <tool>NAME YYYY-MM-DDTHH:MM</tool>
                       NAME is one of the five tools; check_availability and
                       book_intro_call take the local start (America/Los_Angeles
                       wall time, no seconds, no offset), the others take nothing.
                       No id: calls are numbered c1, c2, ... in order.
    json (v1)          <tool>{"id":"c1","name":...,"args":{...}}</tool>, no spaces,
                       keys in the order id, name, args.

`extract_tool_calls` reads both and returns {"id", "name", "args"} with
args.start as full ISO 8601 with the Pacific offset for that date.

    segs = serialize_episode(ep)                 # utterance-level segments
    segs = serialize_episode(ep, word_level=True)  # one segment per word, times spread over dur
    text, mask = to_text_and_mask(segs)          # mask[i] is True where char i is predicted
    calls = extract_tool_calls(model_output_text)  # parse calls back out of a generated stream
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

TZ = ZoneInfo("America/Los_Angeles")
# The voice model's tools; True where the call carries a start time.
TOOLS: Dict[str, bool] = {
    "check_availability": True,
    "get_my_bookings": False,
    "get_caller_details": False,
    "book_intro_call": True,
    "record_sms_consent": False,
}

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


def compact_start(start: str) -> str:
    """Full ISO 8601 start -> local Los Angeles wall time `YYYY-MM-DDTHH:MM`."""
    dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
    if dt.tzinfo is not None:
        dt = dt.astimezone(TZ)
    return dt.strftime("%Y-%m-%dT%H:%M")


def expand_start(local: str) -> str:
    """`YYYY-MM-DDTHH:MM` Los Angeles wall time -> full ISO 8601 with the offset
    in force on that date (-08:00 or -07:00). Raises ValueError if malformed."""
    if not _LOCAL_RE.fullmatch(local):
        raise ValueError(f"start {local!r} is not YYYY-MM-DDTHH:MM")
    return datetime.strptime(local, "%Y-%m-%dT%H:%M").replace(tzinfo=TZ).isoformat()


def tool_call_text(ev: Dict[str, Any], fmt: str = "compact") -> str:
    """The call span. `fmt="compact"` is the v3 form; `fmt="json"` the v1 form."""
    if fmt == "json":
        body = json.dumps({"id": ev["id"], "name": ev["name"], "args": ev["args"]},
                          ensure_ascii=False, separators=(",", ":"))
    elif fmt == "compact":
        start = (ev.get("args") or {}).get("start")
        body = ev["name"] + (f" {compact_start(start)}" if start else "")
    else:
        raise ValueError(f"unknown call format {fmt!r}")
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
                      include_caller: bool = False, fmt: str = "compact") -> List[Segment]:
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
            segs.append((ev["t"], i, 0, Segment(ev["t"], ev["t"], tool_call_text(ev, fmt), False, "tool_call", i)))
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


_LOCAL_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")


def parse_call_body(body: str) -> Dict[str, Any]:
    """One call body (the text between <tool> and </tool>) -> {"name", "args"}.

    Compact form: exactly one of the five tool names, then one space and a
    well-formed local start for the two tools that take one, and nothing else.
    A body that starts with `{` is read as the v1 JSON form and returned as it
    is. Anything else raises ValueError (the caller answers with tool_error)."""
    if body.lstrip().startswith("{"):
        obj = json.loads(body)
        if not isinstance(obj, dict):
            raise ValueError("not an object")
        return obj
    name, sep, arg = body.partition(" ")
    if name not in TOOLS:
        raise ValueError(f"unknown tool {name!r}")
    if not TOOLS[name]:
        if sep:
            raise ValueError(f"{name} takes no argument")
        return {"name": name, "args": {}}
    if not sep:
        raise ValueError(f"{name} needs a start time")
    return {"name": name, "args": {"start": expand_start(arg)}}


def extract_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Parse every <tool>...</tool> span, compact or JSON, into
    {"id", "name", "args"}. Ids are assigned in order of appearance (c1, c2,
    ...) unless a JSON body carries its own. Unparseable bodies come back as
    {"id", "raw", "error"} so an evaluator can count them as wrong calls; they
    still take a number, as the system numbers every call it answers."""
    out = []
    for k, m in enumerate(_CALL_RE.finditer(text), 1):
        body = m.group(1)
        try:
            obj = parse_call_body(body)
            out.append(obj if "id" in obj else {"id": f"c{k}", **obj})
        except ValueError as e:
            out.append({"id": f"c{k}", "raw": body, "error": str(e)})
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
