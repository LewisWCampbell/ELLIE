"""Frame-level runtime logic shared by evaluation and the inference server.

  ToolStreamParser   watches Elle's text stream one token per frame and emits
                     tool calls (on </tool>), results, and speech tokens.
  Injector           decides, per frame, whether the text token is sampled or
                     forced (a pending tool result is injected at the next word
                     boundary, preceded by EPAD exactly as in training).
  decode_text_stream turns a whole generated stream into contract events, so
                     metrics.py can score a model like it scores ground truth.

No torch here; the sampler passes in token ids and gets ids back.
"""

from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence

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
                try:
                    obj = json.loads(text)
                    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
                        raise ValueError("not a {name, args} object")
                    return [StreamEvent("tool_call", frame, token, start,
                                        {"id": str(obj.get("id") or ""), "name": obj["name"],
                                         "args": obj.get("args") if isinstance(obj.get("args"), dict) else {}})]
                except (ValueError, TypeError) as e:
                    return [StreamEvent("tool_error", frame, token, start, {"error": str(e), "text": text})]
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
            elif ev.kind == "result":
                events.append({"t": round((ev.start_frame + frame_offset) / FRAME_RATE, 3), "ch": "system",
                               "type": "tool_result", "id": ev.payload["id"], "content": ev.payload["content"]})
            elif ev.kind == "tool_error":
                events.append({"t": round((ev.start_frame + frame_offset) / FRAME_RATE, 3), "ch": "elle",
                               "type": "tool_call", "id": "", "name": None, "args": {}, "error": ev.payload.get("error")})
        if not was_span and parser.in_span:
            flush()  # a span ends the utterance in progress
    for ev in parser.flush(len(tokens)):
        events.append({"t": round((ev.start_frame + frame_offset) / FRAME_RATE, 3), "ch": "elle",
                       "type": "tool_call", "id": "", "name": None, "args": {}, "error": ev.payload.get("error")})
    flush()
    events.sort(key=lambda e: e["t"])
    return events
