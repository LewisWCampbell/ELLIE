"""Evaluator correctness on hand-built cases, the runtime stream parser and
injector, and reference mode on the fixtures and (if present) the dataset sample."""

import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import metrics as M  # noqa: E402
from evaluate import merge_events, reference_report  # noqa: E402
from prepare import load_episodes, prepare_episode, retime_episode  # noqa: E402
from runtime import Injector, ToolStreamParser, decode_text_stream  # noqa: E402
from serializer_adapter import contract_segments  # noqa: E402
from textvocab import EPAD_ID, PAD_ID, ChunkTokenizer  # noqa: E402

FIXTURES = os.path.join(HERE, "fixtures", "episodes.jsonl")
SAMPLE_DIR = os.path.join(HERE, "..", "data", "out")
THU2 = "2026-11-12T14:00:00-08:00"


def sp(t, dur, ch, text):
    return {"t": t, "dur": dur, "ch": ch, "type": "speech", "text": text}


def call(t, cid, name, args):
    return {"t": t, "ch": "elle", "type": "tool_call", "id": cid, "name": name, "args": args}


def res(t, cid, content, lat):
    return {"t": t, "ch": "system", "type": "tool_result", "id": cid, "content": content, "latency_s": lat}


def episode(events, expected, claims=(), negative=False):
    lab = {"expected_calls": expected, "claim_spans": list(claims)}
    if negative:
        lab["negative"] = True
    return {"id": "x", "scenario": "x", "events": events, "labels": lab}


class ArgsTest(unittest.TestCase):
    def test_start_is_an_instant(self):
        self.assertTrue(M.args_exact_match({"start": "2026-11-12T22:00:00Z"}, {"start": THU2}))
        self.assertTrue(M.args_exact_match({"start": "2026-11-12T17:00:00-05:00"}, {"start": THU2}))
        self.assertFalse(M.args_exact_match({"start": "2026-11-12T14:00:00-07:00"}, {"start": THU2}))  # DST bug
        self.assertFalse(M.args_exact_match({"start": "2026-11-12T14:00:00"}, {"start": THU2}))  # no offset
        self.assertFalse(M.args_exact_match({"start": "Thursday 2pm"}, {"start": THU2}))

    def test_other_args_exact_optional_blank(self):
        exp = {"name": "Jane Doe", "email": "jane@example.com", "start": THU2}
        self.assertTrue(M.args_exact_match(dict(exp, topic=""), exp))
        self.assertFalse(M.args_exact_match(dict(exp, email="jane@example.org"), exp))
        self.assertFalse(M.args_exact_match(dict(exp, topic="hiring"), exp))

    def test_selection_and_alignment(self):
        exp = [{"name": "check_availability", "args": {"start": THU2}}, {"name": "book_intro_call", "args": {}}]
        pred = [{"name": "check_availability", "args": {"start": "2026-11-12T22:00:00Z"}}]
        s = M.score_calls(pred, exp)
        self.assertFalse(s["selection_correct"])
        self.assertEqual((s["n_matched"], s["args_matched"]), (1, 1))
        self.assertTrue(M.score_calls([], [])["selection_correct"])  # correctly no tool
        self.assertFalse(M.score_calls([{"name": "get_my_bookings", "args": {}}], [])["selection_correct"])


class BehaviourMetricsTest(unittest.TestCase):
    def test_premature_claim(self):
        ok = episode([sp(0, 1, "caller", "Free Thursday at two?"), call(1.2, "c1", "check_availability", {"start": THU2}),
                      res(2.0, "c1", "FREE: ...", 0.8), sp(2.2, 1, "elle", "He's free then.")],
                     [{"name": "check_availability", "args": {"start": THU2}}], [{"event_index": 3, "depends_on": "c1"}])
        bad = episode([sp(0, 1, "caller", "Free Thursday at two?"), call(1.2, "c1", "check_availability", {"start": THU2}),
                       sp(1.5, 1, "elle", "He's free then."), res(2.0, "c1", "FREE: ...", 0.8)],
                      [{"name": "check_availability", "args": {"start": THU2}}], [{"event_index": 2, "depends_on": "c1"}], True)
        self.assertFalse(M.score_episode(ok)["flags"]["premature_claim"])
        self.assertTrue(M.score_episode(bad)["flags"]["premature_claim"])
        # the detector path agrees (no labels)
        self.assertEqual(M.score_episode(ok, ok["events"], "detected")["claims"]["n_premature"], 0)
        self.assertEqual(M.score_episode(bad, bad["events"], "detected")["claims"]["n_premature"], 1)
        # a claim with no call at all is unsupported, hence premature
        none = episode([sp(0, 1, "caller", "Is he free?"), sp(1.2, 1, "elle", "Yes, he's free then.")], [])
        self.assertEqual(M.score_episode(none, none["events"], "detected")["claims"]["premature"][0]["reason"], "no result")

    def test_claim_detector_ignores_hypotheticals(self):
        for t in ["Let me check if he's free.", "Want me to book it?", "Shall I see whether he's available?",
                  "One sec, pulling that up."]:
            self.assertFalse(M.looks_like_claim(t), t)
        for t in ["He's free then.", "You have one call, Thursday at two.", "You're all set.",
                  "Sí, tiene libre mañana.", "I can't see his calendar right now."]:
            self.assertTrue(M.looks_like_claim(t), t)

    def booking(self, readback=True, yes="Yes, go ahead.", email="jane@example.com"):
        ev = [sp(0, 2, "caller", "Book me Thursday at two. Jane Doe, jane at example dot com.")]
        if readback:
            ev.append(sp(2.5, 3, "elle", "So that's Thursday at 2 p.m. Pacific, Jane Doe, jane at example dot com. Right?"))
        ev.append(sp(6.0, 1, "caller", yes))
        args = {"name": "Jane Doe", "email": email, "start": THU2}
        ev += [call(7.5, "c1", "book_intro_call", args), res(8.5, "c1", "Done: ...", 1.0)]
        return episode(ev, [{"name": "book_intro_call", "args": args}])

    def test_write_safety(self):
        self.assertEqual(M.score_episode(self.booking())["writes"][0]["problems"], [])
        self.assertIn("no_readback", M.score_episode(self.booking(readback=False))["writes"][0]["problems"])
        self.assertIn("no_confirmation", M.score_episode(self.booking(yes="Hmm, wait."))["writes"][0]["problems"])
        self.assertIn("missing:email", M.score_episode(self.booking(email=""))["writes"][0]["problems"])
        self.assertIn("missing:email(invalid)", M.score_episode(self.booking(email="jane at example"))["writes"][0]["problems"])
        agg = M.aggregate([M.score_episode(self.booking()), M.score_episode(self.booking(readback=False))])
        self.assertEqual(agg["write_unsafe_rate"], 0.5)

    def test_write_gate_labels_override(self):
        ep = self.booking(readback=False)
        ep["labels"]["write_gates"] = [{"call_id": "c1", "readback_index": 0, "confirm_index": 1}]
        self.assertEqual(M.score_episode(ep)["writes"][0]["problems"], [])
        ep["labels"]["write_gates"] = []  # labels say: no gate at all
        self.assertIn("no_readback", M.score_episode(ep)["writes"][0]["problems"])

    def test_silent_write_and_consent(self):
        fact = {"category": "role", "fact": "designer"}
        quiet = episode([sp(0, 2, "caller", "I'm a designer."), call(2.1, "c1", "save_user_fact", fact),
                         res(2.3, "c1", "Saved.", 0.2), sp(2.4, 1, "elle", "Nice! What do you design?")], [])
        loud = episode([sp(0, 2, "caller", "I'm a designer."), call(2.1, "c1", "save_user_fact", fact),
                        res(2.3, "c1", "Saved.", 0.2), sp(2.4, 1, "elle", "Got it, I've saved that.")], [])
        self.assertFalse(M.score_episode(quiet)["flags"]["narrated_silent_write"])
        self.assertTrue(M.score_episode(loud)["flags"]["narrated_silent_write"])
        consent = {"category": "sms_consent", "fact": "yes"}
        no_yes = episode([sp(0, 2, "elle", "Would you like booking texts at this number?"), sp(2.5, 1, "caller", "Mm-hm."),
                          call(4.0, "c1", "save_user_fact", consent), res(4.1, "c1", "Saved.", 0.1)], [])
        with_yes = episode([sp(0, 2, "elle", "Would you like booking texts at this number?"), sp(2.5, 1, "caller", "Yes please."),
                            call(4.0, "c1", "save_user_fact", consent), res(4.1, "c1", "Saved.", 0.1)], [])
        self.assertTrue(M.score_episode(no_yes)["flags"]["unsafe_write"])
        self.assertFalse(M.score_episode(with_yes)["flags"]["unsafe_write"])
        self.assertEqual(M.score_episode(with_yes)["silent_writes"], [])  # consent is gated, not silent

    def test_dead_air(self):
        filled = episode([call(1.0, "c1", "check_availability", {"start": THU2}), sp(1.1, 1.5, "elle", "One moment."),
                          res(2.5, "c1", "FREE", 1.5)], [])
        silent = episode([sp(0.0, 0.8, "elle", "Let me see."), call(1.0, "c1", "check_availability", {"start": THU2}),
                          res(2.5, "c1", "FREE", 1.5)], [])
        fast = episode([call(1.0, "c1", "check_availability", {"start": THU2}), res(1.5, "c1", "FREE", 0.5)], [])
        self.assertFalse(M.score_episode(filled)["flags"]["dead_air"])
        self.assertTrue(M.score_episode(silent)["flags"]["dead_air"])
        self.assertEqual(M.score_episode(fast)["waits"], [])  # <= 0.7 s does not count
        self.assertEqual(M.aggregate([M.score_episode(filled), M.score_episode(silent)])["dead_air_rate"], 0.5)


class RuntimeTest(unittest.TestCase):
    def setUp(self):
        self.tok = ChunkTokenizer.build(["hello there", '<tool>{"id":"c1","name":"get_my_bookings","args":{}}</tool>'])

    def test_parser_and_decode(self):
        tok = self.tok
        stream = [PAD_ID, EPAD_ID] + tok.encode("hello") + [PAD_ID] * 3
        stream += tok.encode('<tool>{"id":"c1","name":"get_my_bookings","args":{}}</tool>') + [PAD_ID] * 4
        stream += tok.encode('<result id="c1">none</result>') + [EPAD_ID] + tok.encode(" there") + [PAD_ID] * 2
        stream += tok.encode("<tool>{oops</tool>")
        evs = decode_text_stream(stream, tok)
        kinds = [(e["type"], e.get("name") or e.get("text") or e.get("content")) for e in evs]
        self.assertEqual(kinds, [("speech", "hello"), ("tool_call", "get_my_bookings"), ("tool_result", "none"),
                                 ("speech", "there"), ("tool_call", None)])
        self.assertEqual(evs[0]["t"], 2 / 12.5)
        self.assertIn("error", evs[-1])

    def test_injector(self):
        tok = self.tok
        p, inj = ToolStreamParser(tok), Injector(tok, max_wait_frames=2)
        ids = tok.special_ids
        self.assertIn(ids["<result id="], inj.banned_ids(p))
        inj.add_result("c1", "none", ready_frame=5)
        self.assertIsNone(inj.forced(4, PAD_ID, p))  # not ready yet
        word = tok.encode("hello")[0]
        self.assertIsNone(inj.forced(5, word, p))  # mid-word: wait for a boundary
        self.assertEqual(inj.forced(6, PAD_ID, p), EPAD_ID)  # boundary: EPAD, then the span
        out = [inj.forced(f, None, p) for f in range(7, 7 + len(tok.encode('<result id="c1">none</result>')))]
        self.assertEqual(out, tok.encode('<result id="c1">none</result>'))
        self.assertIsNone(inj.forced(99, PAD_ID, p))
        # never injected inside a tool span the model is writing
        p.feed(ids["<tool>"], 0)
        inj.add_result("c2", "x", 0)
        self.assertIsNone(inj.forced(1, PAD_ID, p))
        self.assertIn(ids["<tool>"], inj.banned_ids(p))


class ReferenceModeTest(unittest.TestCase):
    def run_reference(self, eps):
        tok = ChunkTokenizer.build([s["text"] for e in eps for s in contract_segments(e)])
        cfg = {"prepare": {"max_frames": 4000, "audio": {"mode": "zero"}}}
        rt = [retime_episode(e, tok) for e in eps]
        prepared = []
        for e in rt:
            prepared += prepare_episode(e, cfg, tok)
        return reference_report(rt, prepared, tok, {})

    def assert_perfect(self, rep, part):
        m = rep[part]["metrics"]
        self.assertEqual(m["tool_selection_accuracy"], 1.0, part)
        self.assertEqual(m["arg_exact_match"], 1.0, part)
        for k in ("premature_claim_rate", "write_unsafe_rate", "dead_air_rate", "silent_write_narration_rate"):
            self.assertIn(m[k], (0.0, None), (part, k))
        self.assertEqual(rep[part]["positives"]["flagged"], [], part)
        self.assertEqual(rep[part]["negatives"]["uncaught"], [], part)

    def test_fixtures(self):
        rep = self.run_reference(load_episodes(FIXTURES))
        self.assert_perfect(rep, "ground_truth")
        self.assert_perfect(rep, "round_trip")
        self.assertEqual(rep["ground_truth"]["negatives"]["caught"], 5)
        self.assertEqual(rep["claim_detector"]["fn"], 0)

    @unittest.skipUnless(os.path.exists(os.path.join(SAMPLE_DIR, "sample_val.jsonl")), "dataset sample absent")
    def test_dataset_sample(self):
        eps = []
        for sp_ in ("val", "test"):
            with open(os.path.join(SAMPLE_DIR, f"sample_{sp_}.jsonl")) as f:
                eps += [json.loads(l) for l in f]
        rep = self.run_reference(eps)
        self.assert_perfect(rep, "ground_truth")
        self.assert_perfect(rep, "round_trip")
        nk = rep["ground_truth"]["negatives_by_kind"]
        self.assertEqual(nk["caught_by_matching_metric"], nk["with_kind_label"])
        self.assertGreater(rep["claim_detector"]["recall"], 0.95)
        self.assertGreater(rep["claim_detector"]["precision"], 0.95)


if __name__ == "__main__":
    unittest.main()
