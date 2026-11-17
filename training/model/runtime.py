"""Frame-level runtime logic shared by evaluation and the inference server.

  ToolStreamParser   watches Elle's text stream one token per frame and emits
                     tool calls (on </tool>), results, and speech tokens.
  Injector           decides, per frame, whether the text token is sampled or
                     forced (a pending tool result is injected at the next word
                     boundary, preceded by EPAD exactly as in training).
  decode_text_stream turns a whole generated stream into contract events, so
                     metrics.py can score a model like it scores ground truth.
  CallConstraint     (v3) the token ids allowed next inside an open <tool>
                     span, so a call can only become one of the five compact
                     calls with a real time.
  BookingGuard       (v3) decides in code whether a book_intro_call or a
                     record_sms_consent is executed (CONTRACT.md v3 point 5).

Two call formats are understood (CONTRACT.md v3 point 4):
  json     <tool>{"id":"c1","name":...,"args":{...}}</tool>   (v1)
  compact  <tool>NAME</tool> or <tool>NAME YYYY-MM-DDTHH:MM</tool>   (v3)

No torch here; the sampler passes in token ids and gets ids back.
"""

from __future__ import annotations

import calendar
import json
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

from textvocab import (EPAD_ID, FRAME_RATE, PAD_ID, RESULT_CLOSE, RESULT_OPEN, TOOL_CLOSE, TOOL_OPEN,
                       TextTokenizer)

MAX_SPAN_TOKENS = 256  # a span longer than this is abandoned as malformed
# A <tool> span that never closes must not swallow the rest of the episode (it
# would also keep <tool> banned and block every result): it is abandoned after
# this many frames. The longest call in the data is about 85 tokens, written
# back to back. Padding inside a span is skipped, not fatal: a sampled model may
# pause mid-call and still finish it correctly.
MAX_SPAN_FRAMES = 200
_RESULT_HEAD = re.compile(r'^\s*"([^"]*)">(.*)$', re.S)

# --- the v3 call format ----------------------------------------------------------

CALL_FORMATS = ("json", "compact")
# The voice model's five tools; the first two take the local start time.
COMPACT_TOOLS = ("check_availability", "get_my_bookings", "get_caller_details", "book_intro_call", "record_sms_consent")
TOOLS_WITH_START = ("check_availability", "book_intro_call")
TOOLS_NO_ARG = tuple(n for n in COMPACT_TOOLS if n not in TOOLS_WITH_START)
OWNER_TZ = "America/Los_Angeles"
_TIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$")
_TIME_SHAPE = "dddd-dd-ddTdd:dd"

ERROR_UNKNOWN_TOOL = "ERROR: unknown tool. Use one of: " + ", ".join(COMPACT_TOOLS) + "."
ERROR_BAD_TIME = "ERROR: bad time. Use YYYY-MM-DDTHH:MM."
ERROR_NO_ARGUMENT = "ERROR: this tool takes no argument. Write only its name."
ERROR_RESULTS = {"unknown_tool": ERROR_UNKNOWN_TOOL, "bad_time": ERROR_BAD_TIME, "bad_argument": ERROR_NO_ARGUMENT}


def _owner_zone():
    from zoneinfo import ZoneInfo

    return ZoneInfo(OWNER_TZ)


def local_time_to_iso(local: str) -> Optional[str]:
    """'2026-11-12T14:00' (the owner's wall clock) -> '2026-11-12T14:00:00-08:00',
    with the Pacific offset of that date. None if it is not a real date and time.
    A wall time skipped or repeated by a DST change takes the first offset."""
    if not _TIME_RE.match(local or ""):
        return None
    try:
        dt = datetime.strptime(local, "%Y-%m-%dT%H:%M").replace(tzinfo=_owner_zone())
    except ValueError:
        return None
    return dt.isoformat(timespec="seconds")


def iso_to_local_time(start: Any) -> str:
    """The compact argument for an event's args.start. A start without an offset
    is taken as already local. Raises ValueError for anything that the compact
    form cannot carry (not a time, or not on a whole minute)."""
    try:
        dt = datetime.fromisoformat(str(start).strip().replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"start is not ISO 8601: {start!r}") from None
    if dt.tzinfo is not None:
        dt = dt.astimezone(_owner_zone())
    if dt.second or dt.microsecond:
        raise ValueError(f"the compact call format has no seconds: {start!r}")
    return dt.strftime("%Y-%m-%dT%H:%M")


def compact_call_body(name: str, args: Optional[dict]) -> str:
    """What goes between <tool> and </tool> in the compact format."""
    if name in TOOLS_WITH_START:
        return f"{name} {iso_to_local_time((args or {}).get('start'))}"
    if name in TOOLS_NO_ARG:
        return name
    raise ValueError(f"{name!r} is not a v3 tool and has no compact form (prepare.call_format: json for v1 data)")


class CallError(ValueError):
    """A closed <tool> span that is not a call. `reason` is one of
    ERROR_RESULTS' keys, or 'malformed' for text that is no call in any format."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


def parse_compact_call(text: str) -> Tuple[str, dict]:
    """'NAME' or 'NAME YYYY-MM-DDTHH:MM' -> (name, args). Exactly the five
    names; the argument is present iff the tool takes one."""
    name, sep, arg = text.strip().partition(" ")
    if name not in COMPACT_TOOLS:
        raise CallError("unknown_tool", f"unknown tool {name[:40]!r}")
    if name in TOOLS_NO_ARG:
        if sep:
            raise CallError("bad_argument", f"{name} takes no argument")
        return name, {}
    iso = local_time_to_iso(arg)
    if iso is None:
        raise CallError("bad_time", f"{name}: bad time {arg[:40]!r}")
    return name, {"start": iso}


def parse_call_text(text: str) -> Tuple[str, dict, str, str]:
    """A closed span's text -> (name, args, id, format). Compact first, then the
    v1 JSON object (id may be '', the caller numbers it). Raises CallError."""
    try:
        name, args = parse_compact_call(text)
        return name, args, "", "compact"
    except CallError as compact_err:
        if not text.lstrip().startswith("{"):
            raise
        try:
            obj = json.loads(text)
        except (ValueError, TypeError) as e:
            raise CallError("malformed", str(e)) from compact_err
        if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
            raise CallError("malformed", "not a {name, args} object") from compact_err
        return (obj["name"], obj.get("args") if isinstance(obj.get("args"), dict) else {},
                str(obj.get("id") or ""), "json")


def time_prefix_ok(p: str) -> bool:
    """Can `p` still grow into a real YYYY-MM-DDTHH:MM (years 2000-2099)? Checked
    digit by digit, so that a constrained model can never write itself into a
    date that does not exist (month 13, February 30, hour 25)."""
    if len(p) > len(_TIME_SHAPE):
        return False
    for ch, want in zip(p, _TIME_SHAPE):
        if (want == "d") != ch.isdigit() or (want != "d" and ch != want) or not ch.isascii():
            return False
    if not "20".startswith(p[:2]):
        return False
    mo, day, hour, minute = p[5:7], p[8:10], p[11:13], p[14:16]
    if mo and (mo[0] not in "01" or (len(mo) == 2 and not 1 <= int(mo) <= 12)):
        return False
    if day:
        last = calendar.monthrange(int(p[:4]), int(mo))[1]
        if int(day[0]) > last // 10 or (len(day) == 2 and not 1 <= int(day) <= last):
            return False
    if hour and (hour[0] not in "012" or (len(hour) == 2 and int(hour) > 23)):
        return False
    if minute and minute[0] not in "012345":
        return False
    return True


def call_prefix_ok(s: str) -> bool:
    """Is `s` a prefix of some valid compact call body?"""
    if any(n.startswith(s) for n in COMPACT_TOOLS):
        return True
    return any(s.startswith(n + " ") and time_prefix_ok(s[len(n) + 1:]) for n in TOOLS_WITH_START)


def call_complete(s: str) -> bool:
    try:
        parse_compact_call(s)
    except CallError:
        return False
    return s == s.strip() and call_prefix_ok(s)


@dataclass
class StreamEvent:
    kind: str  # "token" | "tool_call" | "tool_error" | "result"
    frame: int
    token: Optional[int] = None
    start_frame: Optional[int] = None
    payload: Dict[str, Any] = field(default_factory=dict)


class ToolStreamParser:
    def __init__(self, tok: TextTokenizer, max_span_frames: int = MAX_SPAN_FRAMES):
        self.tok = tok
        self.ids = tok.special_ids
        self.mode = "speech"  # speech | tool | result
        self.buf: List[int] = []
        self.span_start = 0
        self.max_span_frames = max_span_frames
        # Calls are numbered by the system in the order they are closed (c1,
        # c2, ...). A closed span that is not a call takes a number too, so the
        # error that answers it has an id of its own.
        self.n_closed = 0

    def _abort(self, frame: int, token: Optional[int], error: str) -> List[StreamEvent]:
        """Abandon the open span: a tool_error carrying what was decoded so far."""
        mode, start, text = self.mode, self.span_start, self.tok.decode(self.buf)
        self.mode, self.buf = "speech", []
        return [StreamEvent("tool_error", frame, token, start, {"error": f"{mode} span: {error}", "text": text})]

    def flush(self, frame: int) -> List[StreamEvent]:
        """End of stream: a span still open becomes a tool_error (never silently lost)."""
        return self._abort(frame, None, "still open at end of stream") if self.in_span else []

    @property
    def in_span(self) -> bool:
        return self.mode != "speech"

    def feed(self, token: int, frame: int) -> List[StreamEvent]:
        ids = self.ids
        if self.mode == "speech":
            if token == ids[TOOL_OPEN]:
                self.mode, self.buf, self.span_start = "tool", [], frame
                return []
            if token == ids[RESULT_OPEN]:
                self.mode, self.buf, self.span_start = "result", [], frame
                return []
            if token in (ids[TOOL_CLOSE], ids[RESULT_CLOSE]):
                return [StreamEvent("tool_error", frame, token, frame, {"error": "stray close marker", "stray": True})]
            return [StreamEvent("token", frame, token)]
        close = ids[TOOL_CLOSE] if self.mode == "tool" else ids[RESULT_CLOSE]
        if token == close:
            text = self.tok.decode(self.buf)
            mode, start = self.mode, self.span_start
            self.mode, self.buf = "speech", []
            if mode == "tool":
                self.n_closed += 1
                cid = f"c{self.n_closed}"
                try:
                    name, args, own_id, fmt = parse_call_text(text)
                except CallError as e:
                    return [StreamEvent("tool_error", frame, token, start,
                                        {"error": str(e), "text": text, "reason": e.reason, "id": cid})]
                # A v1 JSON call carries its own id; a compact call is numbered here.
                return [StreamEvent("tool_call", frame, token, start,
                                    {"id": own_id or cid, "name": name, "args": args, "format": fmt})]
            m = _RESULT_HEAD.match(text)
            return [StreamEvent("result", frame, token, start,
                                {"id": m.group(1) if m else "", "content": m.group(2) if m else text})]
        if self.mode == "tool" and frame - self.span_start > self.max_span_frames:
            return self._abort(frame, token, f"not closed within {self.max_span_frames} frames")
        if token in (EPAD_ID, PAD_ID):
            return []  # padding inside a span carries nothing
        self.buf.append(token)
        if len(self.buf) > MAX_SPAN_TOKENS:
            return self._abort(frame, token, "too long")
        return []


@dataclass
class _Pending:
    call_id: str
    content: str
    ready_frame: int


class Injector:
    """Forced-token scheduler. Call `forced(frame, last_token)` before sampling
    each frame: an int means "emit this token, do not sample"; None means
    sample (with `banned_ids` masked out)."""

    def __init__(self, tok: TextTokenizer, max_wait_frames: int = 6):
        self.tok = tok
        self.queue: Deque[int] = deque()
        self.pending: List[_Pending] = []
        self.max_wait = max_wait_frames
        self._waited = 0

    def add_result(self, call_id: str, content: str, ready_frame: int) -> None:
        self.pending.append(_Pending(call_id, content, ready_frame))

    def banned_ids(self, parser: ToolStreamParser) -> List[int]:
        ids = self.tok.special_ids
        banned = [ids[RESULT_OPEN], ids[RESULT_CLOSE]]  # results are only ever forced
        if parser.in_span:
            banned.append(ids[TOOL_OPEN])  # no nesting
        return banned

    def forced(self, frame: int, last_token: Optional[int], parser: ToolStreamParser) -> Optional[int]:
        if self.queue:
            return self.queue.popleft()
        ready = [p for p in self.pending if p.ready_frame <= frame]
        if not ready or parser.in_span:
            self._waited = 0
            return None
        # Inject at a word boundary (as prepare.py does), or after max_wait frames.
        at_boundary = last_token is None or last_token in (PAD_ID, EPAD_ID) or self.tok.is_special(last_token)
        if not at_boundary and self._waited < self.max_wait:
            self._waited += 1
            return None
        p = ready[0]
        self.pending.remove(p)
        self._waited = 0
        span = self.tok.encode(f'{RESULT_OPEN}"{p.call_id}">{p.content}{RESULT_CLOSE}')
        if last_token == PAD_ID:
            # Training puts EPAD before a span that follows padding.
            self.queue.extend(span)
            return EPAD_ID
        self.queue.extend(span[1:])
        return span[0]

    @property
    def busy(self) -> bool:
        return bool(self.queue)


def decode_text_stream(tokens: Sequence[int], tok: TextTokenizer, frame_offset: int = 0,
                       gap_frames: int = 8, word_tail_frames: int = 4) -> List[dict]:
    """Generated text stream -> contract events (Elle side only).

    Speech: consecutive plain tokens form an utterance until `gap_frames`
    frames pass without one, or a span starts. Its time is its first token's
    frame; its duration runs to the last token + `word_tail_frames` (text runs
    slightly ahead of audio in Moshi, and a word's audio outlasts its tokens).
    That duration is an approximation; real evaluation should take speech
    timing from VAD on the generated audio channel.
    """
    parser = ToolStreamParser(tok)
    events: List[dict] = []
    unanswered: List[dict] = []  # calls (and answered errors) whose result has not come yet
    cur: List[int] = []
    cur_start = cur_last = None

    def flush():
        nonlocal cur, cur_start, cur_last
        if cur:
            text = tok.decode(cur).strip()
            if text:
                t0 = (cur_start + frame_offset) / FRAME_RATE
                t1 = (cur_last + 1 + word_tail_frames + frame_offset) / FRAME_RATE
                events.append({"t": round(t0, 3), "dur": round(t1 - t0, 3), "ch": "elle", "type": "speech", "text": text})
        cur, cur_start, cur_last = [], None, None

    for f, token in enumerate(tokens):
        token = int(token)
        was_span = parser.in_span
        for ev in parser.feed(token, f):
            if ev.kind == "token":
                if tok.is_control(token) or tok.is_special(token):
                    continue
                if cur_last is not None and f - cur_last > gap_frames:
                    flush()
                if cur_start is None:
                    cur_start = f
                cur.append(token)
                cur_last = f
            elif ev.kind == "tool_call":
                events.append({"t": round((ev.start_frame + frame_offset) / FRAME_RATE, 3), "ch": "elle",
                               "type": "tool_call", "id": ev.payload["id"], "name": ev.payload["name"],
                               "args": ev.payload["args"],
                               "t_end": round((f + 1 + frame_offset) / FRAME_RATE, 3)})
                unanswered.append(events[-1])
            elif ev.kind == "result":
                rid = ev.payload["id"]
                # Compact calls carry no id: the system numbers them c1, c2, ...
                # and so does the parser. Should a stream's results be numbered
                # differently (a window that starts mid-call, ids that skip),
                # the result still answers the oldest call without one.
                hit = next((c for c in unanswered if c["id"] == rid), None) or (unanswered[0] if unanswered else None)
                if hit is not None:
                    unanswered.remove(hit)
                    hit["id"] = rid
                events.append({"t": round((ev.start_frame + frame_offset) / FRAME_RATE, 3), "ch": "system",
                               "type": "tool_result", "id": rid, "content": ev.payload["content"]})
            elif ev.kind == "tool_error":
                events.append({"t": round((ev.start_frame + frame_offset) / FRAME_RATE, 3), "ch": "elle",
                               "type": "tool_call", "id": ev.payload.get("id", ""), "name": None, "args": {},
                               "error": ev.payload.get("error")})
                if ev.payload.get("id"):
                    unanswered.append(events[-1])
        if not was_span and parser.in_span:
            flush()  # a span ends the utterance in progress
    for ev in parser.flush(len(tokens)):
        events.append({"t": round((ev.start_frame + frame_offset) / FRAME_RATE, 3), "ch": "elle",
                       "type": "tool_call", "id": "", "name": None, "args": {}, "error": ev.payload.get("error")})
    flush()
    events.sort(key=lambda e: e["t"])
    return events


# --- constrained decoding inside a call (v3) ---------------------------------------

_CALL_ALPHABET = re.compile(r"^[a-z_ 0-9T:\-]+$")
_ALPHABET_CHARS = "abcdefghijklmnopqrstuvwxyz_ 0123456789T:-"


class CallConstraint:
    """Which text tokens may come next inside an open <tool> span, so that the
    span can only ever close as one of the five compact calls with a real time.

    It works on strings, not on one fixed tokenization, because a SentencePiece
    model can split the same text in more than one way: a candidate token is
    allowed iff decode(span + [candidate]) is still a prefix of a valid call
    body, and </tool> is allowed iff decode(span) is a complete one. PAD and
    EPAD are never allowed (training writes spans on consecutive frames).

    Cost: the candidates (ids whose text uses only [a-z_ 0-9T:-]) are found once
    per tokenizer and indexed by first character; per frame only those starting
    with a character that can come next are looked at, the survivors are
    confirmed with the tokenizer's real decode, and the answer for a span is
    remembered. Use `CallConstraint.for_tokenizer(tok)` to share that work."""

    def __init__(self, tok: TextTokenizer):
        self.tok = tok
        self.close_id = tok.special_ids[TOOL_CLOSE]
        # What a token adds to a span: after another token (`mid`), and as the
        # span's first token (`first`; SentencePiece drops a leading space there).
        anchor = tok.encode_plain("a")[-1]
        base = tok.decode_plain([anchor])
        self.mid: Dict[int, str] = {}
        self.first: Dict[int, str] = {}
        self.by_char_mid: Dict[str, List[int]] = {}
        self.by_char_first: Dict[str, List[int]] = {}
        for i in range(tok.base_card):
            if tok.is_control(i):
                continue
            joined = tok.decode_plain([anchor, i])
            if not joined.startswith(base):
                continue
            text = joined[len(base):]
            if not text or not _CALL_ALPHABET.match(text):
                continue
            self.mid[i] = text
            self.by_char_mid.setdefault(text[0], []).append(i)
            alone = tok.decode_plain([i])
            if alone and _CALL_ALPHABET.match(alone):
                self.first[i] = alone
                self.by_char_first.setdefault(alone[0], []).append(i)
        self.candidates: List[int] = sorted(self.mid)
        self._memo: Dict[Tuple[int, ...], Tuple[int, ...]] = {}
        self.fallbacks = 0  # spans where the indexed search found nothing (should stay 0)

    @classmethod
    def for_tokenizer(cls, tok: TextTokenizer) -> "CallConstraint":
        cached = getattr(tok, "_call_constraint", None)
        if cached is None:
            cached = cls(tok)
            try:
                tok._call_constraint = cached  # type: ignore[attr-defined]
            except AttributeError:  # pragma: no cover - a tokenizer with __slots__
                pass
        return cached

    def _confirmed(self, span: List[int], ids: Sequence[int]) -> List[int]:
        return [i for i in ids if call_prefix_ok(self.tok.decode_plain(span + [i]))]

    def allowed(self, span: Sequence[int]) -> Tuple[int, ...]:
        """Token ids allowed after `span` (the tokens since <tool>, markers and
        padding excluded). Never empty: a span that cannot continue may close."""
        key = tuple(int(t) for t in span)
        hit = self._memo.get(key)
        if hit is not None:
            return hit
        span = list(key)
        cur = self.tok.decode_plain(span) if span else ""
        nxt = [c for c in _ALPHABET_CHARS if call_prefix_ok(cur + c)]
        texts, index = (self.mid, self.by_char_mid) if span else (self.first, self.by_char_first)
        name = next((n for n in TOOLS_WITH_START if cur.startswith(n + " ")), None)
        if name is not None:  # inside the time: only the time is left to check
            done = cur[len(name) + 1:]
            room = len(_TIME_SHAPE) - len(done)
            likely = [i for c in nxt for i in index.get(c, ()) if len(texts[i]) <= room and time_prefix_ok(done + texts[i])]
        elif cur in TOOLS_WITH_START:  # the name is written: a space, then the time (about half of a
            # SentencePiece vocabulary starts with a space, so reject by the next character first)
            likely = [i for i in index.get(" ", ()) if texts[i] == " " or (texts[i][1] == "2" and time_prefix_ok(texts[i][1:]))]
        else:
            likely = [i for c in nxt for i in index.get(c, ()) if call_prefix_ok(cur + texts[i])]
        ok = self._confirmed(span, likely)
        if not ok and not call_complete(cur):
            # The index assumes decode(span + [i]) == decode(span) + text(i). If a
            # tokenizer breaks that, ask it about every candidate instead.
            self.fallbacks += 1
            ok = self._confirmed(span, self.candidates)
        if call_complete(cur) or not ok:
            ok.append(self.close_id)
        out = tuple(sorted(ok))
        if len(self._memo) < 50000:
            self._memo[key] = out
        return out


# --- the booking guard (v3) -----------------------------------------------------------

REFUSALS = {
    "not_checked": "NOT BOOKED: that time has not been checked as free in this call. Check it first.",
    "not_free": "NOT BOOKED: that time was not free when it was last checked. Offer another time.",
    "no_details": "NOT BOOKED: the caller's details are not known yet. Get the caller's details first.",
    "details_missing": "NOT BOOKED: the caller's details are incomplete. Ask for what is missing, "
                       "then get the caller's details again.",
    "no_intent": "NOT BOOKED: the caller has not asked to book. Ask whether they want this time.",
    "withdrawn": "NOT BOOKED: the caller changed their mind after asking. Ask what they want now.",
    "duplicate": "NOT BOOKED: this time is already booked in this call. Do not book it twice.",
    "update_needs_details": "NOT BOOKED: the caller corrected their details. Get the caller's details again, then book.",
    "no_consent_question": "NOT SAVED: the caller has not been asked about text messages. Ask the consent question first.",
    "no_consent": "NOT SAVED: the caller has not said yes to text messages. Do not record consent.",
}
GUARDED_TOOLS = ("book_intro_call", "record_sms_consent")
_BOOKED_PREFIXES = ("done", "updated")


def is_refusal(content: Any) -> bool:
    """A result that says the guard (or the parser) did not run the tool."""
    return str(content or "").lstrip().upper().startswith(("NOT BOOKED", "NOT SAVED", "ERROR: "))


def _instant(start: Any) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(str(start).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.utcoffset() is not None else None


@dataclass
class Refusal:
    reason: str  # a key of REFUSALS
    text: str    # the result to inject


class BookingGuard:
    """CONTRACT.md v3 point 5, as pure logic over what has happened in one call.

    Feed it, as they happen (or later, with their time `t`):
      tool_result(name, args, content)  every tool result the model was given,
                                        and every write the system executed
      caller(intent=, withdraw=, confirm=)  a caller utterance, as the system
                                        understood it: asked to book or accepted
                                        a time / declined, changed their mind or
                                        corrected something / said yes
      elle(sms_offer=True)              Elle asked the SMS consent question
    and ask `check(name, args)` before executing a write: None means execute, a
    Refusal carries the result to inject instead.

    It knows nothing about episodes or text: evaluation feeds it from the
    episode's caller acts (metrics.write_verdicts), a server from its transcript.

    book_intro_call(start) executes only if
      (a) the latest check_availability result for that instant began FREE,
      (b) the latest get_caller_details result began ON FILE or HEARD,
      (c) the caller asked to book or accepted, and has not declined, changed
          their mind or corrected something since,
      (d) that instant is not already booked in this call. A repeat is an
          update, allowed only once the caller corrected something after the
          booking and the details were fetched again after that; (c) is then
          not asked again.
    record_sms_consent executes only after Elle's consent question and a caller
    yes to it. A caller's no to that question is not a change of mind about a
    booking."""

    def __init__(self) -> None:
        self._events: List[tuple] = []  # (t, n, kind, data)
        self._n = 0

    def _add(self, t: Optional[float], kind: str, data: dict) -> None:
        self._n += 1
        # Without times, events are ordered as fed.
        self._events.append((float(t) if t is not None else float(self._n), self._n, kind, data))

    def tool_result(self, name: str, args: Optional[dict], content: Any, t: Optional[float] = None) -> None:
        self._add(t, "result", {"name": name, "args": args or {}, "content": str(content or "")})

    def caller(self, intent: bool = False, withdraw: bool = False, confirm: bool = False,
               t: Optional[float] = None) -> None:
        if intent or withdraw or confirm:
            self._add(t, "caller", {"intent": bool(intent), "withdraw": bool(withdraw), "confirm": bool(confirm)})

    def elle(self, sms_offer: bool = False, t: Optional[float] = None) -> None:
        if sms_offer:
            self._add(t, "offer", {})

    def _state(self) -> dict:
        st: Dict[str, Any] = {"avail": {}, "details": None, "intent": None, "withdraw": None, "booked": {},
                              "offer_open": False, "asked": False, "consent": False}
        for k, (_t, _n, kind, d) in enumerate(sorted(self._events, key=lambda e: e[:2])):
            if kind == "result":
                head = d["content"].lstrip()
                inst = _instant(d["args"].get("start"))
                if d["name"] == "check_availability" and inst is not None:
                    st["avail"][inst] = head
                elif d["name"] == "get_caller_details":
                    st["details"] = (head, k)
                elif d["name"] == "book_intro_call" and inst is not None and head.lower().startswith(_BOOKED_PREFIXES):
                    st["booked"][inst] = k
            elif kind == "offer":
                # Asking (again) does not undo a yes already given; the next answer replaces it.
                st["offer_open"], st["asked"] = True, True
            elif st["offer_open"] and (d["confirm"] or d["withdraw"]) and not d["intent"]:
                # The answer to the consent question, not a word about the booking.
                st["consent"], st["offer_open"] = d["confirm"] and not d["withdraw"], False
            else:
                if d["withdraw"]:
                    st["withdraw"] = k
                if d["intent"]:
                    # An utterance that takes something back and asks anew
                    # ("not Thursday, book Friday at three") leaves an intent.
                    st["intent"] = k + 0.5
        return st

    def _refuse(self, reason: str) -> Refusal:
        return Refusal(reason, REFUSALS[reason])

    def check(self, name: str, args: Optional[dict] = None) -> Optional[Refusal]:
        """None = execute. After executing, feed the write's result with
        tool_result (that is how a booking becomes 'already booked')."""
        st = self._state()
        if name == "record_sms_consent":
            if not st["asked"]:
                return self._refuse("no_consent_question")
            return None if st["consent"] else self._refuse("no_consent")
        if name != "book_intro_call":
            return None
        inst = _instant((args or {}).get("start"))
        det = st["details"]
        det_ok = det is not None and det[0].startswith(("ON FILE", "HEARD"))
        update = inst is not None and inst in st["booked"]
        if update:
            b = st["booked"][inst]
            if st["withdraw"] is None or st["withdraw"] < b:
                return self._refuse("duplicate")
            if not det_ok or det[1] < st["withdraw"]:
                return self._refuse("update_needs_details")
        free = st["avail"].get(inst) if inst is not None else None
        if free is None:
            return self._refuse("not_checked")
        if not free.startswith("FREE"):
            return self._refuse("not_free")
        if not det_ok:
            return self._refuse("details_missing" if det is not None and det[0].startswith("MISSING") else "no_details")
        if not update:
            if st["intent"] is None:
                return self._refuse("no_intent")
            if st["withdraw"] is not None and st["withdraw"] > st["intent"]:
                return self._refuse("withdrawn")
        return None
