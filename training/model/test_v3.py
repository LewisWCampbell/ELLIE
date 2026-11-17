"""Contract v3: compact calls, constrained decoding inside a call, error
results for spans that are no call, the booking guard, the {caller} prompt,
and that the v1 JSON format behaves as before when it is the configured one.

The episodes here are written by hand in v3 form, so these tests do not depend
on training/data's generator."""

import copy
import os
import random
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import metrics as M  # noqa: E402
from evaluate import markdown, merge_events, model_report, reference_report  # noqa: E402
from genloop import FrameGenerator, closed_loop, loop_settings  # noqa: E402
from prepare import prepare_episode, prompt_text_for, retime_episode  # noqa: E402
from runtime import (COMPACT_TOOLS, ERROR_BAD_TIME, ERROR_NO_ARGUMENT, ERROR_UNKNOWN_TOOL, REFUSALS,  # noqa: E402
                     BookingGuard, CallConstraint, CallError, ToolStreamParser, call_complete, call_prefix_ok,
                     decode_text_stream, local_time_to_iso, parse_call_text, parse_compact_call, time_prefix_ok)
from serializer_adapter import contract_segments, episode_segments, tool_call_text  # noqa: E402
from textvocab import EPAD_ID, PAD_ID, TOOL_CLOSE, TOOL_OPEN, ChunkTokenizer  # noqa: E402

SPM = "/tmp/claude-0/-home-user-LetsTalk/04bfaa70-e1f5-560a-a8a9-b845d3e422a3/scratchpad/repro/spm/tok.model"
NOW = "2026-11-09T10:15:00-08:00"
THU2 = "2026-11-12T14:00:00-08:00"
FRI3 = "2026-11-13T15:00:00-08:00"
FREE = "FREE: Thursday, Nov 12 at 2:00 PM PST is open."
ON_FILE = "ON FILE: Rosa Greenleaf, rosa at example dot com."
DONE = "Done: tentative hold for Thursday, Nov 12 at 2:00 PM PST."
VOCAB_TEXT = ["check_availability 2026-11-12T14:00 get_my_bookings get_caller_details book_intro_call 2026-11-13T15:00 "
              "record_sms_consent", FREE, ON_FILE, DONE, "He's free then. Let me check. You're booked. "
              "Would you like a text reminder at this number?"] + list(REFUSALS.values()) + \
             [ERROR_UNKNOWN_TOOL, ERROR_BAD_TIME, ERROR_NO_ARGUMENT]


def sp(t, dur, ch, text, act=None, **kw):
    e = {"t": t, "dur": dur, "ch": ch, "type": "speech", "text": text, **kw}
    if act:
        e["act"] = act
    return e


def call(t, cid, name, args=None):
    return {"t": t, "ch": "elle", "type": "tool_call", "id": cid, "name": name, "args": args or {}}


def res(t, cid, content, lat):
    return {"t": t, "ch": "system", "type": "tool_result", "id": cid, "content": content, "latency_s": lat}


def v3_episode():
    """A returning caller asks for a time, accepts it, and is booked straight
    away (no readback, no yes); then the SMS consent question and a yes."""
    events = [
        sp(0.0, 2.0, "caller", "Hi, is Lewis free Thursday at two?", "ask_slot"),
        sp(2.3, 1.2, "elle", "Let me check his calendar.", "pre_call"),
        call(2.4, "c1", "check_availability", {"start": THU2}),
        res(3.4, "c1", FREE, 1.0),
        sp(3.8, 2.2, "elle", "He's free then. Want me to book it?", "claim"),
        sp(6.3, 0.9, "caller", "Yes please.", "accept"),
        sp(7.3, 1.0, "elle", "Great, one moment.", "pre_call"),
        call(7.4, "c2", "get_caller_details"),
        res(7.8, "c2", ON_FILE, 0.4),
        call(8.0, "c3", "book_intro_call", {"start": THU2}),
        res(9.0, "c3", DONE, 1.0),
        sp(9.4, 4.0, "elle", "You're booked for Thursday at 2 PM Pacific, tentative until Lewis confirms.", "claim"),
        sp(13.8, 2.2, "elle", "Would you like a text reminder at this number?", "sms_offer"),
        sp(16.3, 0.7, "caller", "Yes.", "confirm"),
        call(17.4, "c4", "record_sms_consent"),
        res(17.6, "c4", "Saved.", 0.2),
    ]
    return {"id": "ep_v3", "split": "train", "scenario": "book_returning_caller", "seed": 3, "now": NOW,
            "context": {"caller_on_file": {"name": "Rosa Greenleaf", "email": "rosa@example.com"}},
            "events": events,
            "labels": {"expected_calls": [{"name": e["name"], "args": e["args"]} for e in events if e["type"] == "tool_call"],
                       "claim_spans": [{"event_index": 4, "depends_on": "c1"}, {"event_index": 11, "depends_on": "c3"}],
                       "write_gates": [{"call_id": "c3", "intent_index": 5, "details_call_id": "c2"},
                                       {"call_id": "c4", "question_index": 12, "confirm_index": 13}]}}


def chunk_tok(extra=()):
    return ChunkTokenizer.build(VOCAB_TEXT + list(extra))


def spm_tok():
    """The real SentencePiece tokenizer class on a small model, or None."""
    try:
        import sentencepiece  # noqa: F401
    except ImportError:
        return None
    if not os.path.exists(SPM):
        return None
    from textvocab import SentencePieceTokenizer

    return SentencePieceTokenizer(SPM)


# --- 1. the compact format: parse, serialize, decode ---------------------------------

class CompactParseTest(unittest.TestCase):
    def test_valid_calls(self):
        self.assertEqual(parse_compact_call("check_availability 2026-11-12T14:00"), ("check_availability", {"start": THU2}))
        self.assertEqual(parse_compact_call("book_intro_call 2026-11-13T15:00"), ("book_intro_call", {"start": FRI3}))
        for name in ("get_my_bookings", "get_caller_details", "record_sms_consent"):
            self.assertEqual(parse_compact_call(name), (name, {}))
        self.assertEqual(len(COMPACT_TOOLS), 5)

    def test_dst_offsets(self):
        self.assertEqual(local_time_to_iso("2026-11-12T14:00"), "2026-11-12T14:00:00-08:00")  # PST
        self.assertEqual(local_time_to_iso("2026-07-01T09:30"), "2026-07-01T09:30:00-07:00")  # PDT
        self.assertEqual(local_time_to_iso("2027-03-13T23:30"), "2027-03-13T23:30:00-08:00")  # the night before the change
        self.assertEqual(local_time_to_iso("2027-03-14T12:00"), "2027-03-14T12:00:00-07:00")
        self.assertEqual(local_time_to_iso("2026-11-01T12:00"), "2026-11-01T12:00:00-08:00")  # back to standard time
        _n, args = parse_compact_call("check_availability 2026-07-01T09:30")
        self.assertTrue(M.args_exact_match(args, {"start": "2026-07-01T16:30:00Z"}))

    def test_unknown_tool_and_bad_arguments(self):
        cases = {"save_user_fact": "unknown_tool", "check_availabilty 2026-11-12T14:00": "unknown_tool",
                 "": "unknown_tool", "Book_intro_call 2026-11-12T14:00": "unknown_tool",
                 "check_availability": "bad_time", "check_availability Thursday": "bad_time",
                 "check_availability 2026-11-12T14:00:00-08:00": "bad_time", "check_availability 2026-02-30T10:00": "bad_time",
                 "book_intro_call 2026-11-12T25:00": "bad_time", "check_availability  2026-11-12T14:00": "bad_time",
                 "check_availability 2026-11-12T14:00 x": "bad_time",
                 "get_my_bookings 2026-11-12T14:00": "bad_argument", "record_sms_consent yes": "bad_argument"}
        for text, reason in cases.items():
            with self.assertRaises(CallError, msg=text) as cm:
                parse_compact_call(text)
            self.assertEqual(cm.exception.reason, reason, text)

    def test_legacy_json_still_parses(self):
        body = '{"id":"c7","name":"book_intro_call","args":{"name":"Jane Doe","email":"jane@example.com","start":"' + THU2 + '"}}'
        name, args, cid, fmt = parse_call_text(body)
        self.assertEqual((name, cid, fmt, args["email"]), ("book_intro_call", "c7", "json", "jane@example.com"))
        # v1 accepted any string name; so does the JSON path still
        self.assertEqual(parse_call_text('{"id":"c1","name":"save_user_fact","args":{"category":"role","fact":"x"}}')[0],
                         "save_user_fact")
        for bad in ('{oops', '["a"]', '{"name": 3}'):
            with self.assertRaises(CallError):
                parse_call_text(bad)
        self.assertEqual(parse_call_text("get_my_bookings")[3], "compact")

    def test_parser_numbers_calls_in_order(self):
        tok = chunk_tok()
        text = ("<tool>check_availability 2026-11-12T14:00</tool><tool>make_coffee</tool>"
                "<tool>book_intro_call 2026-11-32T14:00</tool><tool>get_caller_details</tool>"
                '<tool>{"id":"k9","name":"get_my_bookings","args":{}}</tool><tool>get_my_bookings</tool>')
        p = ToolStreamParser(tok)
        evs = [e for f, t in enumerate(tok.encode(text)) for e in p.feed(t, f)]
        got = [(e.kind, e.payload.get("id"), e.payload.get("name") or e.payload.get("reason")) for e in evs]
        self.assertEqual(got, [("tool_call", "c1", "check_availability"), ("tool_error", "c2", "unknown_tool"),
                               ("tool_error", "c3", "bad_time"), ("tool_call", "c4", "get_caller_details"),
                               ("tool_call", "k9", "get_my_bookings"), ("tool_call", "c6", "get_my_bookings")])
        self.assertEqual(evs[0].payload["args"], {"start": THU2})
        self.assertEqual(evs[1].payload["text"], "make_coffee")  # the raw text is kept
        self.assertEqual(evs[3].payload["args"], {})

    def test_serializer_formats(self):
        ev = call(1.0, "c1", "check_availability", {"start": THU2})
        self.assertEqual(tool_call_text(ev, "compact"), "<tool>check_availability 2026-11-12T14:00</tool>")
        # any offset: the compact form is the owner's wall clock
        self.assertEqual(tool_call_text(dict(ev, args={"start": "2026-07-01T16:30:00Z"}), "compact"),
                         "<tool>check_availability 2026-07-01T09:30</tool>")
        self.assertEqual(tool_call_text(call(1, "c2", "get_caller_details"), "compact"), "<tool>get_caller_details</tool>")
        # json is the default and is byte for byte what v1 wrote
        v1 = '<tool>{"id":"c1","name":"check_availability","args":{"start":"' + THU2 + '"}}</tool>'
        self.assertEqual(tool_call_text(ev), v1)
        self.assertEqual(tool_call_text(ev, "json"), v1)
        with self.assertRaises(ValueError):
            tool_call_text(call(1, "c1", "save_user_fact", {"category": "role", "fact": "x"}), "compact")
        with self.assertRaises(ValueError):
            tool_call_text(ev, "yaml")
        with self.assertRaises(ValueError):  # seconds cannot be written, so they are not silently dropped
            tool_call_text(dict(ev, args={"start": "2026-11-12T14:00:30-08:00"}), "compact")

    def test_segments_follow_the_configured_format(self):
        ep = v3_episode()
        for fmt in ("compact", "json"):
            ref = [s["text"] for s in contract_segments(ep, fmt) if s["kind"] == "tool"]
            self.assertEqual(ref[0], tool_call_text(ep["events"][2], fmt))
            # with or without the dataset's serializer, the spans are the contract's for this format
            segs, source = episode_segments(ep, prefer_dataset=True, fmt=fmt)
            self.assertEqual([s["text"] for s in segs if s["kind"] == "tool"], ref, source)
            self.assertEqual([s["loss"] for s in segs if s["kind"] != "speech"],
                             [s["kind"] == "tool" for s in segs if s["kind"] != "speech"])

    def test_prepare_and_round_trip_compact(self):
        ep = v3_episode()
        tok = ChunkTokenizer.build([s["text"] for s in contract_segments(ep, "compact")])
        cfg = {"prepare": {"max_frames": 4000, "audio": {"mode": "zero"}, "call_format": "compact"}}
        rt = retime_episode(ep, tok, call_format="compact")
        (ex,) = prepare_episode(rt, cfg, tok)
        self.assertEqual(ex["meta"]["call_format"], "compact")
        text = tok.decode(ex["text"].tolist())
        self.assertIn("<tool>check_availability 2026-11-12T14:00</tool>", text)
        self.assertIn("<tool>record_sms_consent</tool>", text)
        self.assertNotIn('"name"', text)
        # compact calls are short: a result still never starts before its call has been written
        spans = ex["meta"]["spans"]
        self.assertLessEqual(max(s["end_frame"] - s["start_frame"] for s in spans if s["kind"] == "tool"), 16)
        evs = decode_text_stream(ex["text"], tok, ex["meta"]["frame_offset"])
        calls = [e for e in evs if e["type"] == "tool_call"]
        self.assertEqual([(c["id"], c["name"]) for c in calls],
                         [("c1", "check_availability"), ("c2", "get_caller_details"), ("c3", "book_intro_call"),
                          ("c4", "record_sms_consent")])
        self.assertEqual(calls[2]["args"], {"start": THU2})
        # the metrics, on labels and on the decoded stream (the path a model's output takes)
        rep = reference_report([rt], [ex], tok, {})
        for part in ("ground_truth", "round_trip"):
            m = rep[part]["metrics"]
            self.assertEqual((m["tool_selection_accuracy"], m["arg_exact_match"]), (1.0, 1.0), part)
            self.assertEqual((m["write_unsafe_rate"], m["executed_unsafe_rate"], m["writes"]), (0.0, 0.0, 2), part)
            self.assertEqual(rep[part]["positives"]["flagged"], [], part)

    def test_decode_pairs_results_with_unnumbered_calls(self):
        # ids that do not run c1, c2, ... (a window cut from a longer call)
        tok = chunk_tok()
        stream = tok.encode('<tool>get_my_bookings</tool><result id="c5">none</result>')
        evs = decode_text_stream(stream, tok)
        self.assertEqual([e["id"] for e in evs], ["c5", "c5"])


# --- 3. constrained decoding ------------------------------------------------------------

class TimePrefixTest(unittest.TestCase):
    def test_prefixes(self):
        for ok in ["", "2", "20", "2026-1", "2026-11-12T14:00", "2026-02-2", "2028-02-29", "2026-11-30T23:59", "2026-12-31T0"]:
            self.assertTrue(time_prefix_ok(ok), ok)
        for bad in ["3", "19", "2026-13", "2026-00", "2026-02-3", "2026-02-29", "2026-11-31", "2026-11-00", "2026-11-12T24",
                    "2026-11-12T3", "2026-11-12T14:6", "2026-11-12 ", "2026/", "2026-11-12T14:000", "x"]:
            self.assertFalse(time_prefix_ok(bad), bad)

    def test_every_prefix_ok_string_can_be_completed(self):
        # No dead ends: from any allowed prefix some next character is allowed, until complete.
        rng = random.Random(1)
        for _ in range(300):
            s = rng.choice(["check_availability ", "book_intro_call "])
            while not call_complete(s):
                nxt = [c for c in "0123456789-T:" if call_prefix_ok(s + c)]
                self.assertTrue(nxt, s)
                s += rng.choice(nxt)
            self.assertIsNotNone(local_time_to_iso(s.split(" ")[1]), s)


class ConstraintMixin:
    """The same checks for any tokenizer (`self.tok`)."""

    def ids(self, text):
        return self.tok.encode_plain(text)

    def texts(self, span):
        con = CallConstraint.for_tokenizer(self.tok)
        return {self.tok.decode_plain(list(span) + [i]) for i in con.allowed(span) if i != con.close_id}

    def test_allowed_sets_at_prefixes(self):
        tok, con = self.tok, CallConstraint.for_tokenizer(self.tok)
        close = tok.special_ids[TOOL_CLOSE]
        # at the start: only beginnings of the five names
        start = self.texts([])
        self.assertTrue(start)
        self.assertTrue(all(any(n.startswith(t) for n in COMPACT_TOOLS) for t in start), start)
        self.assertTrue({"b", "c", "g", "r"} <= {t[0] for t in start})
        self.assertNotIn(close, con.allowed([]))
        for banned in (PAD_ID, EPAD_ID, tok.special_ids[TOOL_OPEN], tok.special_ids["<result id="]):
            self.assertNotIn(banned, con.allowed([]))
        # a complete no-arg call: only </tool>
        self.assertEqual(con.allowed(self.ids("get_my_bookings")), (close,))
        self.assertEqual(con.allowed(self.ids("record_sms_consent")), (close,))
        # "get_" can still become either get_ tool, nothing else
        nxt = self.texts(self.ids("get_"))
        self.assertTrue(all(t.startswith(("get_m", "get_c")) for t in nxt), nxt)
        self.assertTrue({t[:5] for t in nxt} == {"get_m", "get_c"}, nxt)
        # a tool that takes a time: a space next, never </tool>
        a = con.allowed(self.ids("check_availability"))
        self.assertNotIn(close, a)
        self.assertTrue(all(t.startswith("check_availability 2") or t == "check_availability " for t in
                            self.texts(self.ids("check_availability"))))
        # inside the time: only what keeps it a real date and time
        nxt = self.texts(self.ids("book_intro_call 2026-1"))
        self.assertTrue(nxt and all(call_prefix_ok(t) for t in nxt), nxt)
        self.assertEqual({t[len("book_intro_call 2026-1")] for t in nxt}, {"0", "1", "2"})
        self.assertNotIn(close, con.allowed(self.ids("book_intro_call 2026-11-12T14:0")))
        self.assertEqual(con.allowed(self.ids("book_intro_call 2026-11-12T14:00")), (close,))
        # the training tokenization of every kind of call is inside the constraint, step by step
        for body in ["check_availability 2026-11-12T14:00", "book_intro_call 2027-03-05T09:30", "get_my_bookings",
                     "get_caller_details", "record_sms_consent"]:
            toks = self.ids(body)
            for k, t in enumerate(toks):
                self.assertIn(t, con.allowed(toks[:k]), (body, k))
            self.assertIn(close, con.allowed(toks), body)
        self.assertEqual(con.fallbacks, 0)

    def test_allowed_is_exactly_the_prefix_rule(self):
        # Soundness and completeness against the definition, by brute force over
        # the candidates: allowed iff decode(span + [candidate]) is a valid prefix.
        tok, con = self.tok, CallConstraint.for_tokenizer(self.tok)
        for body in ["", "get_", "check_availability", "check_availability 20", "book_intro_call 2026-02-2"]:
            span = self.ids(body) if body else []
            here = tok.decode_plain(span) if span else ""
            brute = {i for i in con.candidates if call_prefix_ok(tok.decode_plain(span + [i]))}
            got = set(con.allowed(span)) - {con.close_id}
            self.assertTrue(got <= brute, body)  # nothing invalid is ever allowed
            # and everything valid that adds text is (a piece that decodes to nothing, like a lone "▁", is left out)
            self.assertEqual(got, {i for i in brute if tok.decode_plain(span + [i]) != here}, body)

    def test_random_walks_always_end_in_a_valid_call(self):
        tok, con = self.tok, CallConstraint.for_tokenizer(self.tok)
        rng = random.Random(7)
        seen = set()
        for _ in range(300):
            span = []
            for _step in range(80):
                t = rng.choice(con.allowed(span))
                if t == con.close_id:
                    break
                span.append(t)
            else:
                self.fail(f"span did not close: {tok.decode_plain(span)!r}")
            name, args = parse_compact_call(tok.decode_plain(span))
            seen.add(name)
            if args:
                self.assertIsNotNone(M.parse_instant(args["start"]))
        self.assertEqual(seen, set(COMPACT_TOOLS))

    def test_random_logits_through_the_closed_loop(self):
        tok = self.tok
        T = 700
        ex = {"id": "e", "episode_id": "e", "text": np.full(T, PAD_ID, dtype=np.int32),
              "audio": np.full((16, T), 5, dtype=np.int32), "meta": {"prompt_frames": 0, "frame_offset": 0}}
        gen = RandomLogits(tok, seed=11)
        out = closed_loop(gen, tok, ex, v3_episode(), {"call_format": "compact"})
        self.assertTrue(out["settings"]["constrain_calls"])
        self.assertGreater(len(out["calls"]), 10)
        self.assertGreater(out["constrained_frames"], 50)
        # every span that closed is a well-formed call; only a span cut off by the end of the stream may fail
        self.assertTrue(all("still open" in e["error"] for e in out["tool_errors"]), out["tool_errors"])
        self.assertEqual(out["tool_opens"], len(out["calls"]) + sum(e["error"].startswith("tool span") for e in out["tool_errors"]))
        for c in out["calls"]:
            self.assertIn(c["name"], COMPACT_TOOLS)
            self.assertEqual(bool(c["args"]), c["name"] in ("check_availability", "book_intro_call"))
        # spans are contiguous: no padding between <tool> and </tool>
        ids, inside = tok.special_ids, False
        for t in out["text"]:
            if t == ids[TOOL_OPEN]:
                inside = True
            elif t == ids[TOOL_CLOSE]:
                inside = False
            elif inside:
                self.assertNotIn(t, (PAD_ID, EPAD_ID))
        # and the same generator without the constraint writes garbage (the test has teeth)
        out2 = closed_loop(RandomLogits(tok, seed=11), tok, ex, v3_episode(),
                           {"call_format": "compact", "constrain_calls": False})
        self.assertEqual(len(out2["calls"]), 0)
        self.assertGreater(len(out2["tool_errors"]), 0)


class RandomLogits(FrameGenerator):
    """Random text logits every frame (a model that knows nothing), with <tool>
    made likely now and then so that many calls are started. Honours forcing,
    bans and the allowed set the way a real backend does."""

    def __init__(self, tok, seed=0):
        self.tok, self.seed = tok, seed

    def start(self, ex, ep, gcfg):
        self.rng = np.random.default_rng(self.seed)

    def step(self, frame, caller_codes, forced, banned, elle_codes=None, allowed=None):
        if forced is not None:
            return forced
        logits = self.rng.normal(size=self.tok.card)
        logits[[1, 2]] = -np.inf  # BOS/EOS are never text
        if self.rng.random() < 0.08:
            logits[self.tok.special_ids[TOOL_OPEN]] += 50.0
        if self.rng.random() < 0.05:
            logits[self.tok.special_ids[TOOL_CLOSE]] += 50.0
        logits[list(banned)] = -np.inf
        if allowed is not None:
            keep = np.zeros(self.tok.card, dtype=bool)
            keep[list(allowed)] = True
            logits[~keep] = -np.inf
        return int(np.argmax(logits))


class ChunkConstraintTest(ConstraintMixin, unittest.TestCase):
    def setUp(self):
        self.tok = chunk_tok()

    def test_tiny_generator_honours_allowed(self):
        import torch

        from genloop import TinyFrameGenerator
        from tiny_model import TinyDuplexLM
        torch.manual_seed(0)
        tok = self.tok
        model = TinyDuplexLM(text_card=tok.base_card, card=16, dim=32, layers=1, heads=2, depth_dim=16, depth_layers=1)
        model.resize_text_vocab(4)
        gen = TinyFrameGenerator(model.eval())
        ex = {"text": np.full(8, PAD_ID, dtype=np.int32), "audio": np.full((16, 8), 3, dtype=np.int32)}
        gen.start(ex, {"seed": 0}, {})
        con = CallConstraint.for_tokenizer(tok)
        span = []
        for f in range(8):
            allowed = con.allowed(span)
            tk = gen.step(f, None, None, [], allowed=allowed)
            self.assertIn(tk, allowed)
            if tk == con.close_id:
                break
            span.append(tk)


@unittest.skipUnless(spm_tok() is not None, "sentencepiece or the test model is not available")
class SentencePieceConstraintTest(ConstraintMixin, unittest.TestCase):
    """The same, with the real tokenizer class: pieces overlap ("▁20" + "2" + "6"
    and "▁2026" spell the same text), which is why the constraint works on strings."""

    @classmethod
    def setUpClass(cls):
        cls._tok = spm_tok()

    def setUp(self):
        self.tok = self._tok

    def test_more_than_one_tokenization_is_allowed(self):
        tok, con = self.tok, CallConstraint.for_tokenizer(self.tok)
        pieces = lambda span: {tok.sp.id_to_piece(i) for i in con.allowed(span) if i != con.close_id}  # noqa: E731
        first = pieces([])
        self.assertTrue(len(first) > 4, first)  # whole-word pieces and single letters
        self.assertTrue(any(len(p.replace("▁", "")) > 1 for p in first) and any(len(p.replace("▁", "")) == 1 for p in first))
        # a compact call round-trips through the real tokenizer and stays short
        for body in ["check_availability 2026-11-12T14:00", "record_sms_consent"]:
            self.assertEqual(tok.decode(tok.encode(f"<tool>{body}</tool>")), f"<tool>{body}</tool>")


# --- 4. the booking guard -----------------------------------------------------------------

class BookingGuardTest(unittest.TestCase):
    def ready(self, **skip):
        """A guard in the state where booking THU2 is allowed, minus `skip`."""
        g = BookingGuard()
        if not skip.get("check"):
            g.tool_result("check_availability", {"start": THU2}, FREE)
        if not skip.get("details"):
            g.tool_result("get_caller_details", {}, ON_FILE)
        if not skip.get("intent"):
            g.caller(intent=True)
        return g

    def reason(self, g, start=THU2, name="book_intro_call"):
        r = g.check(name, {"start": start} if name == "book_intro_call" else {})
        return None if r is None else r.reason

    def test_allow_path(self):
        g = self.ready()
        self.assertIsNone(g.check("book_intro_call", {"start": THU2}))
        # the same instant written with another offset is the same time
        self.assertIsNone(g.check("book_intro_call", {"start": "2026-11-12T22:00:00Z"}))
        # order of the three facts does not matter, and HEARD is as good as ON FILE
        g = BookingGuard()
        g.caller(intent=True)
        g.tool_result("get_caller_details", {}, "HEARD: Jane Doe, jane at example dot com.")
        g.tool_result("check_availability", {"start": THU2}, FREE)
        self.assertIsNone(g.check("book_intro_call", {"start": THU2}))
        # there is no readback and no yes in v3: nothing Elle says is needed
        self.assertIsNone(g.check("get_my_bookings"))  # reads are never the guard's business

    def test_a_time_must_have_been_checked_free(self):
        self.assertEqual(self.reason(self.ready(check=True)), "not_checked")
        self.assertEqual(self.reason(self.ready(), start=FRI3), "not_checked")  # another time was checked
        g = self.ready(check=True)
        g.tool_result("check_availability", {"start": THU2}, "BUSY: taken. Nearby open times: 3:00 PM.")
        self.assertEqual(self.reason(g), "not_free")
        g = self.ready(check=True)
        g.tool_result("check_availability", {"start": THU2}, "UNKNOWN: the calendar could not be read.")
        self.assertEqual(self.reason(g), "not_free")
        # the most recent check of that time decides
        g = self.ready()
        g.tool_result("check_availability", {"start": THU2}, "BUSY: taken.")
        self.assertEqual(self.reason(g), "not_free")
        g.tool_result("check_availability", {"start": THU2}, FREE)
        self.assertIsNone(self.reason(g))

    def test_details_must_be_known(self):
        self.assertEqual(self.reason(self.ready(details=True)), "no_details")
        g = self.ready(details=True)
        g.tool_result("get_caller_details", {}, "MISSING: email. Ask for it.")
        self.assertEqual(self.reason(g), "details_missing")
        g.tool_result("get_caller_details", {}, "HEARD: Jane Doe, jane at example dot com.")
        self.assertIsNone(self.reason(g))

    def test_the_caller_must_have_asked(self):
        self.assertEqual(self.reason(self.ready(intent=True)), "no_intent")
        g = self.ready(intent=True)
        g.caller(confirm=True)  # a bare yes to something is not a request to book
        self.assertEqual(self.reason(g), "no_intent")

    def test_withdraw_after_intent_voids_it(self):
        for _kind in ("decline", "mind_change", "correction"):
            g = self.ready()
            g.caller(withdraw=True)
            self.assertEqual(self.reason(g), "withdrawn")
            g.caller(intent=True)  # asking again restores it
            self.assertIsNone(self.reason(g))
        g = BookingGuard()  # a no BEFORE the yes does not matter
        g.caller(withdraw=True)
        g = self.ready() if g else g
        self.assertIsNone(self.reason(g))

    def test_duplicate_and_update_after_correction(self):
        g = self.ready()
        self.assertIsNone(self.reason(g))
        g.tool_result("book_intro_call", {"start": THU2}, DONE)
        self.assertEqual(self.reason(g), "duplicate")
        self.assertEqual(g.check("book_intro_call", {"start": THU2}).text,
                         "NOT BOOKED: this time is already booked in this call. Do not book it twice.")
        g.caller(intent=True)  # asking again does not make a second booking
        self.assertEqual(self.reason(g), "duplicate")
        # the caller corrects something: now it is an update, once the details were fetched again
        g.caller(withdraw=True)
        self.assertEqual(self.reason(g), "update_needs_details")
        g.tool_result("get_caller_details", {}, "MISSING: email. Ask for it.")
        self.assertEqual(self.reason(g), "update_needs_details")
        g.tool_result("get_caller_details", {}, "HEARD: Rosa Greenleaf, rosa at example dot org.")
        self.assertIsNone(self.reason(g))
        g.tool_result("book_intro_call", {"start": THU2}, "Updated: the invite now goes to rosa at example dot org.")
        self.assertEqual(self.reason(g), "duplicate")  # and again only once
        # a booking that failed is not a booking
        g = self.ready()
        g.tool_result("book_intro_call", {"start": THU2}, "Error: the slot was just taken.")
        self.assertIsNone(self.reason(g))

    def test_another_time_after_a_booking(self):
        g = self.ready()
        g.tool_result("book_intro_call", {"start": THU2}, DONE)
        self.assertEqual(self.reason(g, FRI3), "not_checked")
        g.tool_result("check_availability", {"start": FRI3}, "FREE: Friday at 3:00 PM is open.")
        self.assertIsNone(self.reason(g, FRI3))  # the earlier intent stands; the system replaces the hold
        g.caller(withdraw=True)  # "actually, a different day"
        self.assertEqual(self.reason(g, FRI3), "withdrawn")
        g.caller(intent=True)
        self.assertIsNone(self.reason(g, FRI3))
        g.caller(intent=True, withdraw=True)  # "not that one, book the other": one utterance, an intent remains
        self.assertIsNone(self.reason(g, FRI3))

    def test_sms_consent(self):
        g = BookingGuard()
        self.assertEqual(self.reason(g, name="record_sms_consent"), "no_consent_question")
        g.caller(confirm=True)  # a yes before the question is not consent
        self.assertEqual(self.reason(g, name="record_sms_consent"), "no_consent_question")
        g.elle(sms_offer=True)
        self.assertEqual(self.reason(g, name="record_sms_consent"), "no_consent")
        g.caller(confirm=True)
        self.assertIsNone(self.reason(g, name="record_sms_consent"))
        g.elle(sms_offer=True)  # asked again: the answer to that replaces the old yes
        g.caller(withdraw=True)
        self.assertEqual(self.reason(g, name="record_sms_consent"), "no_consent")

    def test_no_to_the_sms_question_is_not_a_change_of_mind(self):
        g = self.ready()
        g.elle(sms_offer=True)
        g.caller(withdraw=True)  # "no thanks" to texts
        self.assertIsNone(self.reason(g))
        g.caller(withdraw=True)  # a second no is about the booking
        self.assertEqual(self.reason(g), "withdrawn")

    def test_events_may_be_fed_late_with_their_time(self):
        g = BookingGuard()
        g.caller(confirm=True, t=20.0)
        g.elle(sms_offer=True, t=15.0)  # found later in the transcript, happened earlier
        self.assertIsNone(self.reason(g, name="record_sms_consent"))

    def test_refusal_texts(self):
        for reason, text in REFUSALS.items():
            head, _, rest = text.partition(": ")
            self.assertIn(head, ("NOT BOOKED", "NOT SAVED"), reason)
            self.assertEqual(len([p for p in rest.split(". ") if p]), 2, text)  # <reason>. <what to do>
            self.assertNotIn(chr(0x2014), text)
        self.assertEqual(len(set(REFUSALS.values())), len(REFUSALS))
        self.assertEqual(REFUSALS["not_checked"],
                         "NOT BOOKED: that time has not been checked as free in this call. Check it first.")


class WriteSafetyV3Test(unittest.TestCase):
    """metrics.write_problems for v3 calls is the guard's verdict."""

    def problems(self, ep, events=None):
        return {w["id"]: w["problems"] for w in M.score_episode(ep, events, "detected" if events is not None else "labels")["writes"]}

    def without_labels(self, ep):
        ep = copy.deepcopy(ep)
        ep["labels"].pop("write_gates")
        return ep

    def test_tool_sets(self):
        self.assertEqual(M.READ_TOOLS, {"check_availability", "get_my_bookings", "get_caller_details"})
        self.assertTrue({"book_intro_call", "record_sms_consent"} <= M.WRITE_TOOLS)
        self.assertIn("record_sms_consent", M.SILENT_TOOLS)
        self.assertIn("save_user_fact", M.WRITE_TOOLS)  # v1 still recognised
        self.assertTrue(M.is_gated({"name": "record_sms_consent", "args": {}}))
        self.assertEqual(M.missing_fields({"name": "book_intro_call", "args": {"start": THU2}}), [])
        self.assertEqual(M.missing_fields({"name": "book_intro_call", "args": {"name": "J", "start": THU2}}), ["email"])

    def test_reference_episode_is_safe_by_labels_and_by_guard(self):
        ep = v3_episode()
        self.assertEqual(self.problems(ep), {"c3": [], "c4": []})
        self.assertEqual(self.problems(self.without_labels(ep)), {"c3": [], "c4": []})
        sc = M.score_episode(ep)
        self.assertFalse(sc["flagged"], sc["flags"])
        self.assertTrue(all(w["executed"] for w in sc["writes"]))

    def test_labels_decide_when_present(self):
        ep = v3_episode()
        ep["labels"]["write_gates"][0]["intent_index"] = None
        self.assertEqual(self.problems(ep)["c3"], ["no_intent"])
        ep["labels"]["write_gates"][0] = {"call_id": "c3", "intent_index": 5, "details_call_id": None}
        self.assertEqual(self.problems(ep)["c3"], ["no_details"])
        ep["labels"]["write_gates"] = [{"call_id": "c3", "intent_index": 5, "details_call_id": "c2"}]  # none for c4
        self.assertEqual(self.problems(ep)["c4"], ["no_consent_question"])
        # a gate in a shape this code does not know falls back to the guard
        ep["labels"]["write_gates"] = [{"call_id": "c3", "something_else": 1}, {"call_id": "c4", "question_index": 12, "confirm_index": 13}]
        self.assertEqual(self.problems(ep), {"c3": [], "c4": []})

    def test_each_guard_rule_shows_up_as_the_problem(self):
        base = self.without_labels(v3_episode())

        def variant(fn):
            ep = copy.deepcopy(base)
            fn(ep["events"])
            return self.problems(ep)

        def drop(i):
            return lambda evs: evs.__delitem__(slice(i, i + 1))

        self.assertEqual(variant(lambda evs: evs[5].update(act="backchannel"))["c3"], ["no_intent"])
        self.assertEqual(variant(lambda evs: evs[3].update(content="BUSY: taken."))["c3"], ["not_free"])
        self.assertEqual(variant(lambda evs: evs[3].update(t=8.5))["c3"], ["not_checked"])  # result not yet there
        self.assertEqual(variant(lambda evs: evs[8].update(content="MISSING: email. Ask for it."))["c3"], ["details_missing"])
        self.assertEqual(variant(drop(8))["c3"], ["no_details"])
        self.assertEqual(variant(lambda evs: evs.insert(6, sp(7.25, 0.1, "caller", "Wait, no.", "decline")))["c3"], ["withdrawn"])
        self.assertEqual(variant(lambda evs: evs[13].update(act="decline"))["c4"], ["no_consent"])
        self.assertEqual(variant(lambda evs: evs[12].update(act="info"))["c4"], ["no_consent_question"])
        # book_intent on an ask counts as asking to book
        self.assertEqual(variant(lambda evs: (evs[5].update(act="backchannel"), evs[0].update(book_intent=True)))["c3"], [])
        # booking the same time twice
        def twice(evs):
            evs.insert(10, call(9.2, "c9", "book_intro_call", {"start": THU2}))
            evs.insert(11, res(9.3, "c9", REFUSALS["duplicate"], 0.1))
        p = variant(twice)
        self.assertEqual((p["c3"], p["c9"]), ([], ["duplicate"]))

    def test_a_caller_utterance_counts_once_it_has_been_said(self):
        base = self.without_labels(v3_episode())
        ep = copy.deepcopy(base)
        ep["events"][9]["t"] = 7.0  # booked while the caller is still saying "Yes please." (6.3 to 7.2)
        ep["events"][8]["t"] = 6.9
        self.assertEqual(self.problems(ep)["c3"], ["no_intent"])
        # "wait, cancel that" begun while the call is being written does not stop it (it went through first)
        ep = copy.deepcopy(base)
        ep["events"].insert(10, sp(7.9, 1.5, "caller", "Sorry, hold on, can you cancel that?", "cancel"))
        self.assertEqual(self.problems(ep)["c3"], [])
        ep["events"][10].update(t=7.2, dur=0.5)  # said before the call: it does
        self.assertEqual(self.problems(ep)["c3"], ["withdrawn"])

    def test_consent_question_is_found_in_decoded_speech(self):
        # No acts, cut into pieces, the caller's "mm-hm" in between: as a decoded stream looks.
        ev = [sp(80.0, 6.0, "elle", "One more thing. I can send you texts about your booking at this number."),
              sp(86.0, 0.5, "caller", "Okay.", "backchannel"),
              sp(90.0, 6.0, "elle", "Frequency varies, and message and data rates may apply."),
              sp(98.5, 0.5, "caller", "Mm-hm.", "backchannel"),
              sp(99.0, 5.0, "elle", "The terms are at lewiswcampbell.com. Would you like that, yes or no?"),
              sp(106.0, 1.4, "caller", "Yes, go ahead.", "confirm"),
              call(107.8, "c1", "record_sms_consent"), res(108.6, "c1", "Saved.", 0.2)]
        self.assertEqual(M.sms_offer_times(ev), [99.0])
        ep = {"id": "x", "scenario": "x", "events": ev, "labels": {"expected_calls": [], "claim_spans": []}}
        self.assertEqual(self.problems(ep), {"c1": []})
        ev[4]["text"] = "The terms are at lewiswcampbell.com."  # no question was asked
        self.assertEqual(self.problems(ep), {"c1": ["no_consent_question"]})
        self.assertEqual(M.sms_offer_times([sp(0, 2, "elle", "Want me to book it?")]), [])

    def test_v1_bookings_keep_the_readback_rule(self):
        args = {"name": "Jane Doe", "email": "jane@example.com", "start": THU2}
        ev = [sp(0, 2, "caller", "Book me Thursday at two. Jane Doe, jane at example dot com."),
              sp(6.0, 1, "caller", "Yes, go ahead."), call(7.5, "c1", "book_intro_call", args), res(8.5, "c1", "Done: ...", 1.0)]
        ep = {"id": "x", "scenario": "x", "events": ev, "labels": {"expected_calls": [], "claim_spans": []}}
        self.assertEqual(M.score_episode(ep)["writes"][0]["problems"], ["no_readback"])
        self.assertFalse(M.is_guarded(ev[2]))


# --- 2 + 4. the closed loop: error results and the guard --------------------------------

class TimedGenerator(FrameGenerator):
    """Starts writing `script[frame]` (text with markers) at that frame, one
    token per unforced frame; PAD otherwise."""

    def __init__(self, tok, script):
        self.tok, self.script = tok, script

    def start(self, ex, ep, gcfg):
        self.queue, self.allowed_seen = [], []

    def step(self, frame, caller_codes, forced, banned, elle_codes=None, allowed=None):
        if frame in self.script:
            self.queue += self.tok.encode(self.script[frame])
        if forced is not None:
            return forced
        tk = self.queue.pop(0) if self.queue else PAD_ID
        if allowed is not None:
            self.allowed_seen.append(tk in allowed)
        return tk


def example(T):
    return {"id": "ep_v3", "episode_id": "ep_v3", "text": np.full(T, PAD_ID, dtype=np.int32),
            "audio": np.full((16, T), 5, dtype=np.int32), "meta": {"prompt_frames": 0, "frame_offset": 0}}


def results_in(out, tok):
    return {e["id"]: e["content"] for e in decode_text_stream(out["text"], tok) if e["type"] == "tool_result"}


class ErrorResultTest(unittest.TestCase):
    def setUp(self):
        self.tok = chunk_tok(["make_coffee tomorrow"])

    def run_loop(self, script, gcfg):
        gen = TimedGenerator(self.tok, script)
        return closed_loop(gen, self.tok, example(260), v3_episode(), gcfg), gen

    def test_invalid_calls_are_answered_when_unconstrained(self):
        script = {5: "<tool>make_coffee</tool>", 60: "<tool>check_availability tomorrow</tool>",
                  120: "<tool>get_my_bookings 2026-11-12T14:00</tool>", 180: "<tool>check_availability 2026-11-12T14:00</tool>"}
        out, _ = self.run_loop(script, {"call_format": "compact", "constrain_calls": False})
        self.assertEqual([(e["reason"], e["result_id"]) for e in out["tool_errors"]],
                         [("unknown_tool", "c1"), ("bad_time", "c2"), ("bad_argument", "c3")])
        self.assertEqual(out["tool_errors"][0]["text"], "make_coffee")
        got = results_in(out, self.tok)
        self.assertEqual(got["c1"], "ERROR: unknown tool. Use one of: check_availability, get_my_bookings, "
                                    "get_caller_details, book_intro_call, record_sms_consent.")
        self.assertEqual(got["c2"], "ERROR: bad time. Use YYYY-MM-DDTHH:MM.")
        self.assertEqual(got["c3"], ERROR_NO_ARGUMENT)
        # the numbering continues: the good call is c4 and gets the episode's result
        (c,) = out["calls"]
        self.assertEqual((c["id"], c["name"], got["c4"]), ("c4", "check_availability", FREE))
        # the error is ready one frame after </tool> and injected at once (the model is at a boundary)
        e = out["tool_errors"][0]
        self.assertEqual(e["ready_frame"], e["frame"] + 2)
        stream = out["text"]
        self.assertEqual(stream[e["frame"] + 1], PAD_ID)
        self.assertEqual(stream[e["frame"] + 2], EPAD_ID)
        self.assertEqual(stream[e["frame"] + 3], self.tok.special_ids["<result id="])
        self.assertEqual(out["tool_opens"], 4)

    def test_constraint_makes_the_same_script_impossible(self):
        # A backend that ignores the allowed set is caught by the parser, but one
        # that honours it cannot write these: the scripted tokens are not allowed.
        out, gen = self.run_loop({5: "<tool>make_coffee</tool>"}, {"call_format": "compact"})
        self.assertIn(False, gen.allowed_seen)

    def test_error_results_can_be_switched_off(self):
        out, _ = self.run_loop({5: "<tool>make_coffee</tool>"},
                               {"call_format": "compact", "constrain_calls": False, "error_results": False})
        self.assertEqual(len(out["tool_errors"]), 1)
        self.assertNotIn("result_id", out["tool_errors"][0])
        self.assertEqual(out["forced_frames"], 0)


# Frames (12.5 per second) in the hand-written episode: the caller's "Yes
# please." ends at 7.2 s = frame 90; the consent "Yes." ends at 17.0 s = frame 212.5.
GOOD_SCRIPT = {30: "<tool>check_availability 2026-11-12T14:00</tool>", 95: "<tool>get_caller_details</tool>",
               125: "<tool>book_intro_call 2026-11-12T14:00</tool>", 175: " Would you like a text reminder at this number?",
               225: "<tool>record_sms_consent</tool>"}


class GuardInTheLoopTest(unittest.TestCase):
    def setUp(self):
        self.tok = chunk_tok()

    def run_loop(self, script, gcfg=None, ep=None, T=330):
        gen = TimedGenerator(self.tok, script)
        return closed_loop(gen, self.tok, example(T), ep or v3_episode(), dict({"call_format": "compact"}, **(gcfg or {})))

    def by_frame(self, out):
        return [(c["name"], c["guard"], c["executed"]) for c in out["calls"]]

    def test_allowed_writes_get_the_scripted_results(self):
        out = self.run_loop(GOOD_SCRIPT)
        self.assertEqual(self.by_frame(out), [("check_availability", None, True), ("get_caller_details", None, True),
                                              ("book_intro_call", None, True), ("record_sms_consent", None, True)])
        got = results_in(out, self.tok)
        self.assertEqual((got["c1"], got["c2"], got["c3"], got["c4"]), (FREE, ON_FILE, DONE, "Saved."))
        self.assertEqual(out["tool_errors"], [])

    def test_each_refusal_is_injected_and_the_scripted_result_is_kept(self):
        early = "<tool>book_intro_call 2026-11-12T14:00</tool>"
        check, details = GOOD_SCRIPT[30], GOOD_SCRIPT[95]
        # (what the model does first, the refusal, what it then does to put it right)
        cases = [
            ({5: early}, "not_checked", {200: check, 250: details, 290: early}),         # nothing checked yet
            ({30: check, 70: early}, "no_details", {250: details, 290: early}),          # free, but who is calling?
            ({5: check, 55: details, 95: early}, "no_intent", {290: early}),             # the caller has not said yes yet
            ({30: "<tool>check_availability 2026-11-13T15:00</tool>", 95: details, 125: early}, "not_checked", None),
        ]
        for script, reason, retry in cases:
            ep = v3_episode()
            ep["events"][5]["t"] = 12.0  # the caller's "Yes please." ends at 12.9 s, frame 161
            out = self.run_loop({**script, **(retry or {})}, ep=ep, T=360)
            books = [c for c in out["calls"] if c["name"] == "book_intro_call"]
            got = results_in(out, self.tok)
            self.assertEqual((books[0]["guard"], books[0]["executed"]), (reason, False), reason)
            self.assertEqual(got[books[0]["id"]], REFUSALS[reason])
            self.assertEqual(books[0]["ready_frame"], books[0]["end_frame"] + 1 + round(0.2 * 12.5))
            if retry:
                self.assertEqual((books[1]["guard"], books[1]["executed"]), (None, True), reason)
                self.assertEqual(got[books[1]["id"]], DONE)  # the refusal did not use up the episode's result

    def test_withdraw_and_duplicate_in_the_loop(self):
        ep = v3_episode()
        ep["events"].insert(6, sp(8.0, 0.8, "caller", "Actually, wait.", "mind_change"))  # frame 100
        out = self.run_loop({30: GOOD_SCRIPT[30], 95: GOOD_SCRIPT[95], 125: GOOD_SCRIPT[125]}, ep=ep)
        self.assertEqual(self.by_frame(out)[-1], ("book_intro_call", "withdrawn", False))
        # twice in a row, the second before the first result is even injected
        out = self.run_loop({30: GOOD_SCRIPT[30], 95: GOOD_SCRIPT[95], 125: GOOD_SCRIPT[125] * 2})
        self.assertEqual(self.by_frame(out)[-2:], [("book_intro_call", None, True), ("book_intro_call", "duplicate", False)])

    def test_update_after_a_correction(self):
        ep = v3_episode()
        evs = ep["events"]
        evs[12:] = [sp(14.0, 1.5, "caller", "No, it's dot org.", "correction"),
                    call(16.0, "c4", "get_caller_details"), res(16.4, "c4", "HEARD: Rosa Greenleaf, rosa at example dot org.", 0.4),
                    call(16.6, "c5", "book_intro_call", {"start": THU2}), res(17.6, "c5", "Updated: the invite goes to rosa at example dot org.", 1.0)]
        book = GOOD_SCRIPT[125]
        # frame 175 = 14.0 s: the correction begins. Before it a repeat is a duplicate; after it, without
        # fresh details, it is refused too; with them it is an update and gets the scripted "Updated:".
        out = self.run_loop({30: GOOD_SCRIPT[30], 95: GOOD_SCRIPT[95], 125: book, 145: book, 200: book,
                             245: "<tool>get_caller_details</tool>", 300: book}, ep=ep, T=380)
        books = [c for c in out["calls"] if c["name"] == "book_intro_call"]
        self.assertEqual([c["guard"] for c in books], [None, "duplicate", "update_needs_details", None])
        self.assertTrue(results_in(out, self.tok)[books[-1]["id"]].startswith("Updated:"))

    def test_sms_consent_needs_the_question_and_the_yes(self):
        # no question asked by the model: refused even though the caller's (replayed) yes is there
        out = self.run_loop({225: "<tool>record_sms_consent</tool>"})
        self.assertEqual(self.by_frame(out), [("record_sms_consent", "no_consent_question", False)])
        self.assertTrue(results_in(out, self.tok)["c1"].startswith("NOT SAVED: "))
        # asked, but recorded before the caller answered (the yes ends at frame 212.5)
        out = self.run_loop({175: GOOD_SCRIPT[175], 200: "<tool>record_sms_consent</tool>"})
        self.assertEqual(self.by_frame(out), [("record_sms_consent", "no_consent", False)])

    def test_guard_can_be_switched_off(self):
        out = self.run_loop({5: GOOD_SCRIPT[125]}, {"booking_guard": False})
        self.assertEqual(self.by_frame(out), [("book_intro_call", None, True)])

    def test_report_attempted_vs_executed(self):
        tok = self.tok
        book = GOOD_SCRIPT[125]  # two bookings too early (nothing checked; free but no details), then the good one
        script = {5: book, 60: GOOD_SCRIPT[30], 110: book, 165: GOOD_SCRIPT[95], 205: book, 260: GOOD_SCRIPT[175],
                  320: GOOD_SCRIPT[225]}
        ep = v3_episode()
        ep["events"][13]["t"] = 24.0  # the caller's yes to the consent question, after the model's (later) question
        with tempfile.TemporaryDirectory() as d:
            cfg = {"model": {"backend": "tiny"}, "run_dir": d, "prepare": {"call_format": "compact"},
                   "eval": {"gen": {"constrain_calls": False}}}
            rep = model_report(cfg, d, [ep], [example(360)], {"executed_unsafe_rate": {"max": 0.0}},
                               generator=TimedGenerator(tok, script), tok=tok)
        g = rep["guard"]
        self.assertTrue(g["enabled"])
        self.assertEqual((g["bookings_attempted"], g["bookings_blocked"], g["bookings_executed"]), (3, 2, 1))
        self.assertEqual(g["guard_refusals"], {"not_checked": 1, "no_details": 1})
        self.assertEqual((g["sms_consents_attempted"], g["sms_consents_executed"]), (1, 1))
        self.assertEqual(g["executed_unsafe"], 0)
        self.assertEqual(g["n_metric_disagreements"], 0)  # the metric replays the same guard and agrees with the loop
        m = rep["metrics"]
        self.assertEqual(m["writes"], 4)
        self.assertEqual(m["write_unsafe_rate"], 0.5)        # what the model attempted
        self.assertEqual(m["write_problem_counts"], {"not_checked": 1, "no_details": 1})
        self.assertEqual((m["executed_unsafe_rate"], m["writes_executed"]), (0.0, 2))  # what took effect
        self.assertTrue(rep["gates"]["executed_unsafe_rate"]["pass"])
        md = markdown(rep)
        self.assertIn("guard refusals by reason", md)
        self.assertIn("executed_unsafe_rate", md)

    def test_executed_unsafe_is_zero_whatever_the_model_does(self):
        # A model that books and records consent at random moments, guard on.
        tok, rng = self.tok, random.Random(5)
        bodies = ["<tool>book_intro_call 2026-11-12T14:00</tool>", "<tool>record_sms_consent</tool>",
                  "<tool>check_availability 2026-11-12T14:00</tool>", "<tool>get_caller_details</tool>",
                  "<tool>book_intro_call 2026-11-13T15:00</tool>", " Would you like a text reminder at this number?"]
        executed = 0
        for trial in range(25):
            script = {f: rng.choice(bodies) for f in sorted(rng.sample(range(5, 300, 5), 14))}
            ep = v3_episode()
            out = self.run_loop(script, ep=ep)
            evs = merge_events(ep, decode_text_stream(out["text"], tok))
            sc = M.score_episode(ep, evs, claim_source="detected")
            live = {c["id"]: c for c in out["calls"]}
            for w in sc["writes"]:
                if w["executed"]:
                    executed += 1
                    self.assertEqual(w["problems"], [], (trial, w))
                if w["id"] in live:  # loop and metric give the same verdict, reason included
                    self.assertEqual((w["problems"] or [None])[0], live[w["id"]]["guard"], (trial, w))
            self.assertIn(M.aggregate([sc])["executed_unsafe_rate"], (0.0, None))
        self.assertGreater(executed, 5)


# --- 5. the prompt -------------------------------------------------------------------------

class PromptTest(unittest.TestCase):
    TEMPLATE = ("<system> You work for Lewis as his phone assistant and your name is Elle. Information: You answer "
                "questions about Lewis and book thirty-minute intro calls with him. Today is {today}. {caller} <system>")

    def test_caller_on_file_or_not(self):
        ep = v3_episode()
        self.assertEqual(prompt_text_for(ep, self.TEMPLATE),
                         "<system> You work for Lewis as his phone assistant and your name is Elle. Information: You "
                         "answer questions about Lewis and book thirty-minute intro calls with him. Today is Monday, "
                         "November 9, 2026, 10:15 AM Pacific Time (UTC-08:00). The caller is Rosa Greenleaf, a returning "
                         "caller. <system>")
        for ctx in ({"caller_on_file": None}, {}, None, {"caller_on_file": {"email": "x@y.z"}}, {"caller_on_file": {"name": " "}}):
            e = dict(ep, context=ctx)
            self.assertIn("Today is Monday, November 9, 2026, 10:15 AM Pacific Time (UTC-08:00). The caller is not on file. <system>",
                          prompt_text_for(e, self.TEMPLATE))
        e = dict(ep)
        e.pop("context")
        self.assertIn("The caller is not on file.", prompt_text_for(e, "{caller}"))
        # templates without the placeholders are untouched
        self.assertEqual(prompt_text_for(ep, "<system> Hello. <system>"), "<system> Hello. <system>")
        self.assertEqual(prompt_text_for(ep, "{caller}"), "The caller is Rosa Greenleaf, a returning caller.")

    def test_base_config_is_v3(self):
        from config_util import load_config
        from prepare import call_format_of

        cfg, _ = load_config(os.path.join(HERE, "config", "base_gpu.yaml"))
        self.assertEqual(cfg["prepare"]["prompt_text"], self.TEMPLATE)
        self.assertEqual(call_format_of(cfg), "compact")
        self.assertTrue(loop_settings(dict(cfg["eval"]["gen"], call_format="compact"))["constrain_calls"])
        dry, _ = load_config(os.path.join(HERE, "config", "dryrun.yaml"))
        self.assertEqual(call_format_of(dry), "json")  # the v1 fixtures
        self.assertEqual(call_format_of({}), "json")
        with self.assertRaises(ValueError):
            call_format_of({"prepare": {"call_format": "xml"}})


# --- v1 stays v1 ---------------------------------------------------------------------------

class JsonFormatUnchangedTest(unittest.TestCase):
    V1_CALL = '<tool>{"id":"c1","name":"check_availability","args":{"start":"' + THU2 + '"}}</tool>'

    def test_defaults(self):
        self.assertEqual(loop_settings({}), {"call_format": "json", "constrain_calls": False, "error_results": False,
                                             "booking_guard": False, "guard_latency_s": 0.2})
        # the constraint only exists for compact calls, even if asked for
        self.assertFalse(loop_settings({"call_format": "json", "constrain_calls": True})["constrain_calls"])
        on = loop_settings({"call_format": "compact"})
        self.assertTrue(on["constrain_calls"] and on["error_results"] and on["booking_guard"])

    def test_json_loop_has_no_constraint_no_error_result_no_guard(self):
        tok = chunk_tok([self.V1_CALL, '{"id":"c2","name":"book_intro_call","args":{"name":"Jane Doe","email":"jane@example.com"}}'])
        v1_book = ('<tool>{"id":"c2","name":"book_intro_call","args":{"name":"Jane Doe","email":"jane@example.com",'
                   '"start":"' + THU2 + '"}}</tool>')

        class Strict(TimedGenerator):  # the v1 backend signature: no `allowed` argument at all
            def step(self, frame, caller_codes, forced, banned, elle_codes=None):
                return super().step(frame, caller_codes, forced, banned, elle_codes)

        gen = Strict(tok, {5: "<tool>{oops</tool>", 40: self.V1_CALL, 150: v1_book})
        out = closed_loop(gen, tok, example(330), v3_episode(), {})
        self.assertEqual(len(out["tool_errors"]), 1)
        self.assertNotIn("result_id", out["tool_errors"][0])  # a broken call still just vanishes, as in v1
        self.assertEqual([(c["id"], c["name"], c["guard"]) for c in out["calls"]],
                         [("c1", "check_availability", None), ("c2", "book_intro_call", None)])  # the calls' own ids
        got = results_in(out, tok)
        self.assertEqual((got["c1"], got["c2"]), (FREE, DONE))  # no guard: the scripted result
        self.assertEqual(out["constrained_frames"], 0)

    def test_prepare_json_is_the_default(self):
        ep = v3_episode()
        tok = ChunkTokenizer.build([s["text"] for s in contract_segments(ep)])
        cfg = {"prepare": {"max_frames": 4000, "audio": {"mode": "zero"}}}
        (ex,) = prepare_episode(ep, cfg, tok)
        self.assertIn(self.V1_CALL, tok.decode(ex["text"].tolist()))
        self.assertEqual(ex["meta"]["call_format"], "json")


if __name__ == "__main__":
    unittest.main()
