"""prepare.py: frame alignment, loss masking, retiming, negatives, serializer use."""

import copy
import json
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import metrics as M  # noqa: E402
from prepare import build_items, load_episodes, place_items, prepare_episode, retime_episode  # noqa: E402
from serializer_adapter import contract_segments, dataset_serializer_status, episode_segments  # noqa: E402
from textvocab import EPAD_ID, PAD_ID, RESULT_CLOSE, RESULT_OPEN, TOOL_CLOSE, TOOL_OPEN, ChunkTokenizer, seconds_to_frame  # noqa: E402

FIXTURES = os.path.join(HERE, "fixtures", "episodes.jsonl")
SAMPLE = os.path.join(HERE, "..", "data", "out", "sample_val.jsonl")
CFG = {"prepare": {"max_frames": 4000, "tail_sec": 1.0, "audio": {"mode": "placeholder", "card": 64}}}


def fixtures():
    return {e["id"]: e for e in load_episodes(FIXTURES)}


def tokenizer(eps):
    return ChunkTokenizer.build([s["text"] for e in eps for s in contract_segments(e)])


class AlignmentTest(unittest.TestCase):
    def setUp(self):
        self.eps = fixtures()
        self.tok = tokenizer(self.eps.values())

    def test_word_placement_matches_moshi_rule(self):
        segs = [{"kind": "speech", "t": 1.0, "dur": 0.8, "text": "hello there", "loss": True, "event_index": 0}]
        items = build_items(segs, self.tok)
        placed = place_items(items, 40)
        f0 = seconds_to_frame(1.0)
        self.assertEqual(placed["text"][f0 - 1], EPAD_ID)  # EPAD announces the word
        self.assertEqual(int(placed["text"][f0]), items[0]["tokens"][0])
        self.assertTrue((placed["text"][: f0 - 1] == PAD_ID).all())
        self.assertTrue(placed["loss_mask"].all())

    def test_spans_masks_and_order(self):
        ep = self.eps["ep_fx_001"]
        ex = prepare_episode(ep, CFG, self.tok)[0]
        text, mask = ex["text"], ex["loss_mask"]
        ids = self.tok.special_ids
        spans = ex["meta"]["spans"]
        self.assertEqual([s["kind"] for s in spans], ["tool", "result", "tool", "result"])
        for s in spans:
            seg = text[s["start_frame"]:s["end_frame"]]
            if s["kind"] == "tool":
                self.assertEqual(seg[0], ids[TOOL_OPEN]); self.assertEqual(seg[-1], ids[TOOL_CLOSE])
                self.assertTrue(mask[s["start_frame"]:s["end_frame"]].all(), "tool spans are predicted")
                self.assertFalse(s["forced"])
            else:
                self.assertEqual(seg[0], ids[RESULT_OPEN]); self.assertEqual(seg[-1], ids[RESULT_CLOSE])
                self.assertFalse(mask[s["start_frame"]:s["end_frame"]].any(), "result spans are forced")
                self.assertTrue(s["forced"])
                if text[s["start_frame"] - 1] == EPAD_ID:
                    self.assertEqual(mask[s["start_frame"] - 1], 0, "EPAD before a result is masked too")
            self.assertTrue(all(not self.tok.is_special(t) for t in seg[1:-1]), "spans are contiguous")
        # exactly the result frames (+ their EPADs) are masked
        n_res = sum(s["end_frame"] - s["start_frame"] for s in spans if s["kind"] == "result")
        self.assertLessEqual(n_res, int((mask == 0).sum()))
        self.assertLessEqual(int((mask == 0).sum()), n_res + 2)
        self.assertEqual(ex["meta"]["dropped_tokens"], 0)
        self.assertEqual(ex["audio"].shape, (16, len(text)))
        # the stream decodes back to the same call
        self.assertIn('"name":"check_availability"', self.tok.decode(text.tolist()))

    def test_retime_is_physical(self):
        tok = self.tok
        for ep in self.eps.values():
            rt = retime_episode(ep, tok)
            evs = rt["events"]
            self.assertEqual(len(evs), len(ep["events"]))
            calls = {e["id"]: e for e in evs if e["type"] == "tool_call"}
            for r in (e for e in evs if e["type"] == "tool_result"):
                c = calls[r["id"]]
                from serializer_adapter import tool_call_text, tool_result_text

                call_frames = len(tok.encode(tool_call_text(c)))
                self.assertGreaterEqual(r["t"] + 1e-6, c["t"] + call_frames / 12.5 + r["latency_s"] - 0.01)
            # labelled claims of positives still start after their result's injection
            if not ep["labels"].get("negative"):
                res = M.results_by_id(evs)
                for cl in ep["labels"]["claim_spans"]:
                    r = res[cl["depends_on"]]
                    end = r["t"] + len(tok.encode(f'<result id="{r["id"]}">{r["content"]}</result>')) / 12.5
                    self.assertGreaterEqual(evs[cl["event_index"]]["t"] + 1e-6, end)
            # metric verdicts are invariant under retiming
            a, b = M.score_episode(ep)["flags"], M.score_episode(rt)["flags"]
            self.assertEqual(a, b, ep["id"])

    def test_prepared_text_never_lags_far(self):
        for ep in self.eps.values():
            ex = prepare_episode(ep, CFG, self.tok)[0]
            self.assertLessEqual(ex["meta"]["max_word_delay_frames"], 6, ep["id"])

    def test_neg_mask(self):
        ex = prepare_episode(self.eps["ep_fx_007"], CFG, self.tok)[0]  # premature claim
        self.assertGreater(int(ex["neg_mask"].sum()), 0)
        claim = self.tok.decode([int(t) for t, m in zip(ex["text"], ex["neg_mask"]) if m])
        self.assertIn("free", claim)
        ex = prepare_episode(self.eps["ep_fx_008"], CFG, self.tok)[0]  # write without readback
        self.assertEqual([int(t) for t, m in zip(ex["text"], ex["neg_mask"]) if m], [self.tok.special_ids[TOOL_OPEN]])
        ex = prepare_episode(self.eps["ep_fx_001"], CFG, self.tok)[0]
        self.assertEqual(int(ex["neg_mask"].sum()), 0)

    def test_windows_never_split_spans(self):
        cfg = copy.deepcopy(CFG)
        cfg["prepare"]["max_frames"] = 120
        exs = prepare_episode(self.eps["ep_fx_001"], cfg, self.tok)
        self.assertGreater(len(exs), 1)
        ids = self.tok.special_ids
        for ex in exs:
            opens = sum(1 for t in ex["text"] if t in (ids[TOOL_OPEN], ids[RESULT_OPEN]))
            closes = sum(1 for t in ex["text"] if t in (ids[TOOL_CLOSE], ids[RESULT_CLOSE]))
            self.assertEqual(opens, closes)
            self.assertLessEqual(len(ex["text"]), 120)

    def test_prompt_prefix_is_forced(self):
        cfg = copy.deepcopy(CFG)
        cfg["prepare"]["prompt_text"] = "<system> You are Elle. <system>"
        ex = prepare_episode(self.eps["ep_fx_005"], cfg, self.tok)[0]
        P = ex["prompt_frames"]
        self.assertGreater(P, 0)
        self.assertFalse(ex["loss_mask"][:P].any())
        self.assertTrue((ex["audio"][:, :P] == -1).all())
        self.assertEqual(ex["frame_offset"], -P)


class SerializerTest(unittest.TestCase):
    def test_dataset_serializer_agrees_with_contract(self):
        eps = list(fixtures().values())
        if os.path.exists(SAMPLE):
            with open(SAMPLE) as f:
                eps += [json.loads(l) for _, l in zip(range(50), f)]
        status = dataset_serializer_status()
        for ep in eps:
            segs, src = episode_segments(ep)
            if status == "ok":
                self.assertEqual(src, "dataset", (ep["id"], src))
            else:
                self.assertEqual(src, "contract")
            self.assertTrue(all(s["loss"] == (s["kind"] != "result") for s in segs))


if __name__ == "__main__":
    unittest.main()
