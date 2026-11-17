import json
import os
import sys
import tempfile
import unittest
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import expand  # noqa: E402
import render_audio  # noqa: E402
from generate import build_episode  # noqa: E402
from serialize import (SPECIAL_TOKENS, compact_start, expand_start, extract_results,  # noqa: E402
                       extract_tool_calls, forced_spans, parse_call_body, serialize_episode,
                       strip_markers, to_text_and_mask, tool_call_text)

CONTRACT_EXAMPLE = {
    "id": "ep_000123", "split": "train", "scenario": "x", "seed": 123, "now": "2026-11-09T10:15:00-08:00",
    "events": [
        {"t": 0.00, "dur": 2.1, "ch": "caller", "type": "speech", "text": "Is he free Thursday at two?"},
        {"t": 2.45, "dur": 1.2, "ch": "elle", "type": "speech", "text": "Let me check his calendar."},
        {"t": 2.50, "ch": "elle", "type": "tool_call", "id": "c1", "name": "check_availability",
         "args": {"start": "2026-11-12T14:00:00-08:00"}},
        {"t": 3.30, "ch": "system", "type": "tool_result", "id": "c1", "content": "FREE: ok", "latency_s": 0.8},
        {"t": 3.70, "dur": 2.6, "ch": "elle", "type": "speech", "text": "He's free then. Want me to pencil you in?"},
    ],
    "labels": {"expected_calls": [], "claim_spans": [], "forbidden_before_result": True},
}


class TestSerialize(unittest.TestCase):
    def test_contract_example_exact(self):
        segs = serialize_episode(CONTRACT_EXAMPLE)
        self.assertEqual([s.kind for s in segs], ["speech", "tool_call", "tool_result", "speech"])
        self.assertEqual(segs[1].text, "<tool>check_availability 2026-11-12T14:00</tool>")
        self.assertEqual(serialize_episode(CONTRACT_EXAMPLE, fmt="json")[1].text,
                         '<tool>{"id":"c1","name":"check_availability","args":{"start":"2026-11-12T14:00:00-08:00"}}</tool>')
        self.assertEqual(segs[2].text, '<result id="c1">FREE: ok</result>')
        self.assertEqual([s.t for s in segs], [2.45, 2.50, 3.30, 3.70])
        self.assertEqual([s.is_forced for s in segs], [False, False, True, False])

    def test_caller_excluded_by_default_and_forced_when_included(self):
        self.assertNotIn("caller", {s.kind for s in serialize_episode(CONTRACT_EXAMPLE)})
        segs = serialize_episode(CONTRACT_EXAMPLE, include_caller=True)
        self.assertEqual(segs[0].kind, "caller")
        self.assertTrue(segs[0].is_forced)

    def test_mask_covers_exactly_the_result_spans(self):
        ep = build_episode("book_after_availability_busy", 4)
        text, mask = to_text_and_mask(serialize_episode(ep))
        self.assertEqual(len(text), len(mask))
        spans = [text[a:b].strip() for a, b in forced_spans(text, mask)]
        n_results = sum(e["type"] == "tool_result" for e in ep["events"])
        self.assertEqual(len(spans), n_results)
        for s in spans:
            self.assertTrue(s.startswith('<result id="') and s.endswith("</result>"), s)
        for a, b in forced_spans(text, mask):
            self.assertNotIn("<tool>", text[a:b])

    def test_markers_at_event_times_and_order(self):
        for seed in range(20):
            ep = build_episode("availability_mind_change_mid_wait", seed)
            segs = serialize_episode(ep, word_level=True)
            self.assertEqual([s.t for s in segs], sorted(s.t for s in segs))
            calls = {e["id"]: e for e in ep["events"] if e["type"] == "tool_call"}
            res = {e["id"]: e for e in ep["events"] if e["type"] == "tool_result"}
            pos = {}
            order = iter(sorted(calls, key=lambda c: int(c[1:])))  # compact calls carry no id: c1, c2, ...
            for k, s in enumerate(segs):
                if s.kind == "tool_call":
                    cid = next(order)
                    self.assertEqual(s.t, calls[cid]["t"])
                    pos[("call", cid)] = k
                elif s.kind == "tool_result":
                    cid = s.text.split('"')[1]
                    self.assertEqual(s.t, res[cid]["t"])
                    pos[("res", cid)] = k
            for cid in calls:
                self.assertLess(pos[("call", cid)], pos[("res", cid)])

    def test_word_level_interleaves_call_inside_utterance(self):
        segs = serialize_episode(CONTRACT_EXAMPLE, word_level=True)
        kinds = [s.kind for s in segs]
        i = kinds.index("tool_call")
        self.assertEqual(kinds[i - 1], "speech")
        self.assertEqual(segs[i - 1].text, "Let")  # the call lands 0.05 s into "Let me check..."
        self.assertEqual(kinds[i + 1], "speech")

    def test_claims_follow_their_results_in_stream(self):
        ep = build_episode("book_slot_taken_race", 2)
        segs = serialize_episode(ep)
        where = {s.event_index: k for k, s in enumerate(segs)}
        res_idx = {e["id"]: i for i, e in enumerate(ep["events"]) if e["type"] == "tool_result"}
        for cs in ep["labels"]["claim_spans"]:
            self.assertGreater(where[cs["event_index"]], where[res_idx[cs["depends_on"]]])

    def test_round_trip_tool_calls(self):
        ep = build_episode("book_after_availability_free", 7)
        text, _ = to_text_and_mask(serialize_episode(ep))
        got = [{"name": c["name"], "args": c["args"]} for c in extract_tool_calls(text)]
        self.assertEqual(got, ep["labels"]["expected_calls"])
        self.assertEqual(len(extract_results(text)), len(got))
        plain = strip_markers(text)
        for tok in SPECIAL_TOKENS:
            self.assertNotIn(tok, plain)
        self.assertEqual(extract_tool_calls("<tool>{not json</tool>")[0].get("raw"), "{not json")

    def test_compact_and_json_forms_both_parse(self):
        for name in ("book_after_availability_free", "book_email_corrected", "bookings_one", "returning_caller"):
            for seed in range(5):
                ep = build_episode(name, seed)
                want = [{"id": e["id"], "name": e["name"], "args": e["args"]}
                        for e in ep["events"] if e["type"] == "tool_call"]
                for fmt in ("compact", "json"):
                    text, _ = to_text_and_mask(serialize_episode(ep, fmt=fmt))
                    self.assertEqual(extract_tool_calls(text), want, (name, seed, fmt))
                compact, _ = to_text_and_mask(serialize_episode(ep))
                self.assertNotIn("{", "".join(m for m in compact.split("<tool>")[1:] for m in [m.split("</tool>")[0]]))

    def test_compact_call_text(self):
        ev = {"id": "c3", "name": "book_intro_call", "args": {"start": "2026-11-12T14:00:00-08:00"}}
        self.assertEqual(tool_call_text(ev), "<tool>book_intro_call 2026-11-12T14:00</tool>")
        self.assertEqual(tool_call_text(ev, fmt="compact"), tool_call_text(ev))
        self.assertEqual(tool_call_text(ev, fmt="json"),
                         '<tool>{"id":"c3","name":"book_intro_call","args":{"start":"2026-11-12T14:00:00-08:00"}}</tool>')
        for name in ("get_my_bookings", "get_caller_details", "record_sms_consent"):
            self.assertEqual(tool_call_text({"id": "c1", "name": name, "args": {}}), f"<tool>{name}</tool>")
        # A start given in another zone is written as Los Angeles wall time.
        self.assertEqual(compact_start("2026-11-12T22:00:00+00:00"), "2026-11-12T14:00")
        with self.assertRaises(ValueError):
            tool_call_text(ev, fmt="yaml")

    def test_compact_start_gets_the_offset_of_its_date(self):
        for local, iso in (("2026-11-12T14:00", "2026-11-12T14:00:00-08:00"),   # standard time
                           ("2026-07-01T09:30", "2026-07-01T09:30:00-07:00"),   # daylight time
                           ("2026-10-31T16:00", "2026-10-31T16:00:00-07:00"),   # last day of PDT
                           ("2026-11-01T10:00", "2026-11-01T10:00:00-08:00"),   # clocks went back at 2 a.m.
                           ("2027-03-13T10:00", "2027-03-13T10:00:00-08:00"),   # day before spring forward
                           ("2027-03-14T10:00", "2027-03-14T10:00:00-07:00")):  # clocks went forward at 2 a.m.
            self.assertEqual(expand_start(local), iso)
            self.assertEqual(compact_start(iso), local)
            got = extract_tool_calls(f"<tool>check_availability {local}</tool>")
            self.assertEqual(got, [{"id": "c1", "name": "check_availability", "args": {"start": iso}}])

    def test_ids_are_assigned_in_order(self):
        text = ("Sure. <tool>check_availability 2026-11-12T14:00</tool> "
                '<result id="c1">FREE: ok</result> Great. <tool>get_caller_details</tool> '
                '<result id="c2">ON FILE: Rosa Greenleaf, rosa dot greenleaf at example dot org.</result> '
                "<tool>book_intro_call 2026-11-12T14:00</tool>")
        self.assertEqual([(c["id"], c["name"]) for c in extract_tool_calls(text)],
                         [("c1", "check_availability"), ("c2", "get_caller_details"), ("c3", "book_intro_call")])
        self.assertEqual([r[0] for r in extract_results(text)], ["c1", "c2"])

    def test_malformed_compact_calls_are_rejected(self):
        for body in ("reschedule_booking", "save_user_fact", "book_intro_call", "check_availability Thursday",
                     "check_availability 2026-11-12T14:00:00-08:00", "check_availability 2026-13-40T14:00",
                     "get_caller_details 2026-11-12T14:00", "get_my_bookings ", " get_my_bookings",
                     "book_intro_call 2026-11-12T14:00 Rosa", "book_intro_call  2026-11-12T14:00", ""):
            with self.assertRaises(ValueError, msg=body):
                parse_call_body(body)
            got = extract_tool_calls(f"<tool>get_my_bookings</tool><tool>{body}</tool>")
            self.assertEqual((got[1]["id"], got[1]["raw"]), ("c2", body))
            self.assertIn("error", got[1])


class TestOptionalStages(unittest.TestCase):
    def test_expand_is_passthrough_without_key(self):
        old = {k: os.environ.pop(k) for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN") if k in os.environ}
        try:
            self.assertIsNone(expand.make_client())
            ep = build_episode("availability_free", 1)
            self.assertIs(expand.expand_episode(ep, None), ep)
            with tempfile.TemporaryDirectory() as d:
                src, dst = os.path.join(d, "a.jsonl"), os.path.join(d, "b.jsonl")
                with open(src, "w") as f:
                    f.write(json.dumps(ep) + "\n")
                expand.main([src, dst])
                self.assertEqual(open(src, "rb").read(), open(dst, "rb").read())
        finally:
            os.environ.update(old)

    def test_render_audio_clear_error_without_key(self):
        old = os.environ.pop("DEEPGRAM_API_KEY", None)
        try:
            with self.assertRaises(render_audio.MissingCredentials) as cm:
                render_audio.DeepgramAuraBackend()
            self.assertIn("DEEPGRAM_API_KEY", str(cm.exception))
            with tempfile.TemporaryDirectory() as d:
                self.assertEqual(render_audio.main([os.devnull, d, "--backend", "deepgram"]), 2)
        finally:
            if old is not None:
                os.environ["DEEPGRAM_API_KEY"] = old

    def test_dry_render_writes_stereo_24k(self):
        ep = build_episode("barge_in_then_book", 1)
        with tempfile.TemporaryDirectory() as d:
            paths = render_audio.render_preview([ep], render_audio.DryBackend(), d)[0]
            with wave.open(paths["wav"]) as w:
                self.assertEqual((w.getnchannels(), w.getframerate(), w.getsampwidth()), (2, 24000, 2))
                self.assertGreater(w.getnframes() / 24000, 5.0)
            with open(paths["json"]) as f:
                align = json.load(f)
            self.assertTrue(align["alignments"])
            self.assertTrue(all(a[2] == "SPEAKER_MAIN" for a in align["alignments"]))
            self.assertTrue(any(s["kind"] == "result" for s in align["spans"]))


if __name__ == "__main__":
    unittest.main()
