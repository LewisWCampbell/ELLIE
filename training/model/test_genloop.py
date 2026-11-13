"""The shared closed-loop evaluation loop (genloop.closed_loop) with result
injection, driven by a scripted generator so every frame is predictable, plus
window stitching and the CPU no-op of --listen."""

import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import metrics as M  # noqa: E402
from evaluate import merge_events, write_wav  # noqa: E402
from genloop import FrameGenerator, closed_loop, stitch_windows  # noqa: E402
from runtime import decode_text_stream  # noqa: E402
from textvocab import EPAD_ID, PAD_ID, ChunkTokenizer  # noqa: E402

THU2 = "2026-11-12T14:00:00-08:00"
CALL = '<tool>{"id":"c1","name":"check_availability","args":{"start":"' + THU2 + '"}}</tool>'
RESULT = "FREE: Thursday at 2:00 PM is open."


def episode(latency: float = 1.0):
    return {"id": "ep_t", "seed": 1, "events": [
        {"t": 0.0, "dur": 1.5, "ch": "caller", "type": "speech", "text": "Is he free Thursday at two?"},
        {"t": 2.0, "ch": "elle", "type": "tool_call", "id": "c1", "name": "check_availability", "args": {"start": THU2}},
        {"t": 3.0, "ch": "system", "type": "tool_result", "id": "c1", "content": RESULT, "latency_s": latency},
        {"t": 9.0, "dur": 1.2, "ch": "elle", "type": "speech", "text": "He's free then."}],
        "labels": {"expected_calls": [{"name": "check_availability", "args": {"start": THU2}}],
                   "claim_spans": [{"event_index": 3, "depends_on": "c1"}]}}


class ScriptedGenerator(FrameGenerator):
    """Emits `script[f]` on unforced frames (PAD when exhausted). After the
    tool call it waits with PAD until a result has been injected, then says
    `after` (or says it right away if `eager`, which is a premature claim)."""

    def __init__(self, tok, before, after, eager=False):
        self.tok, self.before, self.after, self.eager = tok, before, after, eager

    def start(self, ex, ep, gcfg):
        self.queue = list(self.before)
        self.after_q = list(self.after)
        self.log = []
        self.seen_result = False
        self.caller = []
        self.elle = []

    def step(self, frame, caller_codes, forced, banned, elle_codes=None):
        self.caller.append(None if caller_codes is None else int(caller_codes[0]))
        self.elle.append(None if elle_codes is None else [int(x) for x in elle_codes])
        self.log.append((frame, forced, tuple(banned)))
        if forced is not None:
            if forced == self.tok.special_ids["</result>"]:
                self.seen_result = True
            return forced
        if self.queue:
            return self.queue.pop(0)
        if (self.seen_result or self.eager) and self.after_q:
            return self.after_q.pop(0)
        return PAD_ID


class ClosedLoopTest(unittest.TestCase):
    def setUp(self):
        self.tok = ChunkTokenizer.build([CALL, RESULT, "He's free then. Let me check."])
        T = 160
        audio = np.full((16, T), 5, dtype=np.int32)
        audio[8:, :3] = -1  # caller silent (no codes) for the first frames
        self.ex = {"id": "ep_t", "episode_id": "ep_t", "text": np.full(T, PAD_ID, dtype=np.int32),
                   "audio": audio, "meta": {"prompt_frames": 0, "frame_offset": 0}}

    def run_loop(self, eager=False, latency=1.0, max_wait=6):
        tok = self.tok
        before = [PAD_ID] * 5 + tok.encode(" Let me check.") + [PAD_ID] * 2 + tok.encode(CALL)
        gen = ScriptedGenerator(tok, before, tok.encode(" He's free then.") + [PAD_ID], eager)
        out = closed_loop(gen, tok, self.ex, episode(latency), {"max_wait_frames": max_wait})
        return gen, out

    def test_injection_after_latency(self):
        gen, out = self.run_loop()
        tok = self.tok
        (call,) = out["calls"]
        self.assertEqual(call["name"], "check_availability")
        self.assertEqual(call["ready_frame"], call["end_frame"] + 1 + round(1.0 * 12.5))
        self.assertIsNotNone(call["inject_frame"])
        self.assertGreaterEqual(call["inject_frame"], call["ready_frame"])
        self.assertLessEqual(call["inject_frame"] - call["ready_frame"], 7)  # EPAD + boundary wait
        # the forced frames are exactly EPAD + the result span, starting at a boundary
        forced = [(f, t) for f, t, _ in gen.log if t is not None]
        span = tok.encode(f'<result id="c1">{RESULT}</result>')
        self.assertEqual([t for _, t in forced], [EPAD_ID] + span)
        self.assertEqual(out["forced_frames"], len(span) + 1)
        self.assertEqual(out["text"][forced[0][0]:forced[-1][0] + 1], [EPAD_ID] + span)
        # result markers were banned on every sampled frame
        for _f, t, banned in gen.log:
            if t is None:
                self.assertIn(tok.special_ids["<result id="], banned)
                self.assertIn(tok.special_ids["</result>"], banned)
        # caller codes come from the example, -1 (no audio) included: the model
        # was trained on -1 there, so it must not be replaced by anything else
        self.assertEqual(gen.caller[:4], [-1, -1, -1, 5])
        # outside the prompt Elle's audio is the model's own (not forced)
        self.assertTrue(all(e is None for e in gen.elle))
        # decoded and scored with the shared metrics: no premature claim
        evs = merge_events(episode(), decode_text_stream(out["text"], tok, 0))
        sc = M.score_episode(episode(), evs, claim_source="detected")
        self.assertTrue(sc["selection"]["selection_correct"])
        self.assertEqual(sc["selection"]["args_matched"], 1)
        self.assertEqual(sc["claims"]["n_claims"], 1)
        self.assertEqual(sc["claims"]["n_premature"], 0)

    def test_premature_claim_is_caught(self):
        _gen, out = self.run_loop(eager=True)
        evs = merge_events(episode(), decode_text_stream(out["text"], self.tok, 0))
        sc = M.score_episode(episode(), evs, claim_source="detected")
        self.assertEqual(sc["claims"]["n_premature"], 1)

    def test_backend_must_honour_forcing(self):
        class Rebel(ScriptedGenerator):
            def step(self, frame, caller_codes, forced, banned, elle_codes=None):
                return PAD_ID if forced is not None else super().step(frame, caller_codes, forced, banned, elle_codes)

        tok = self.tok
        gen = Rebel(tok, tok.encode(CALL), [])
        with self.assertRaises(RuntimeError):
            closed_loop(gen, tok, self.ex, episode(0.1), {})

    def test_prompt_frames(self):
        tok = self.tok
        audio = self.ex["audio"].copy()
        audio[:, :2] = -1  # prepare.py: all 16 audio rows are -1 on prompt frames
        ex = dict(self.ex, text=np.array(tok.encode("hi") + [PAD_ID] * 158, dtype=np.int32), audio=audio,
                  meta={"prompt_frames": 2, "frame_offset": -2})
        gen = ScriptedGenerator(tok, [], [])
        out = closed_loop(gen, tok, ex, episode(), {})
        self.assertEqual([t for _, t, _ in gen.log[:2]], tok.encode("hi"))  # forced by the loop
        # The audio the model hears during the prompt is exactly the training
        # input: -1 on both channels (Elle's forced, not generated).
        self.assertEqual(gen.caller[:2], [-1, -1])
        self.assertEqual(gen.elle[:2], [[-1] * 8, [-1] * 8])
        self.assertIsNone(gen.elle[2])
        # teacher mode forces Elle's recorded audio on every frame
        gen_t = ScriptedGenerator(tok, [], [])
        closed_loop(gen_t, tok, ex, episode(), {"elle_audio": "teacher"})
        self.assertEqual(gen_t.elle[5], [5] * 8)
        gen2 = ScriptedGenerator(tok, [], [])
        gen2.native_prompt = True
        out2 = closed_loop(gen2, tok, ex, episode(), {})
        self.assertEqual(gen2.log[0][0], 2)  # the backend ran its own prompt; frames 0-1 skipped
        self.assertEqual(out["text"][:2], out2["text"][:2])


class CallAttemptTest(unittest.TestCase):
    """Every call the model starts is accounted for: a span that never closes,
    or loses itself in padding, is reported (and does not block later calls)."""

    def setUp(self):
        self.tok = ChunkTokenizer.build([CALL, RESULT, "He's free then. Let me check."])
        T = 400
        self.ex = {"id": "ep_t", "episode_id": "ep_t", "text": np.full(T, PAD_ID, dtype=np.int32),
                   "audio": np.full((16, T), 5, dtype=np.int32), "meta": {"prompt_frames": 0, "frame_offset": 0}}

    def test_abandoned_call_then_good_call(self):
        tok = self.tok
        open_id = tok.special_ids["<tool>"]
        # A call that stalls and never closes is abandoned after MAX_SPAN_FRAMES...
        partial = [open_id] + tok.encode('{"id":"c1","na')[:3] + [PAD_ID] * 205
        before = [PAD_ID] * 3 + partial + [PAD_ID] * 3 + tok.encode(CALL)
        gen = ScriptedGenerator(tok, before, [])
        out = closed_loop(gen, tok, self.ex, episode(), {})
        self.assertEqual(out["tool_opens"], 2)
        self.assertEqual(len(out["tool_errors"]), 1)
        self.assertIn("not closed within", out["tool_errors"][0]["error"])
        self.assertEqual(len(out["calls"]), 1)  # <tool> was not left banned after the abort

    def test_call_with_a_pause_inside_still_parses(self):
        tok = self.tok
        body = tok.encode(CALL)
        before = [PAD_ID] * 3 + body[:6] + [PAD_ID] * 4 + body[6:]
        gen = ScriptedGenerator(tok, before, [])
        out = closed_loop(gen, tok, self.ex, episode(), {})
        self.assertEqual((out["tool_opens"], len(out["calls"]), len(out["tool_errors"])), (1, 1, 0))
        self.assertEqual(out["calls"][0]["name"], "check_availability")

    def test_unclosed_call_is_reported_at_end(self):
        tok = self.tok
        open_id = tok.special_ids["<tool>"]
        before = [PAD_ID] * 3 + [open_id] + tok.encode('{"id":"c1"')
        gen = ScriptedGenerator(tok, before, [])
        n = len(before)  # the stream ends mid-call (no padding after it)
        ex = dict(self.ex, text=self.ex["text"][:n], audio=self.ex["audio"][:, :n])
        out = closed_loop(gen, tok, ex, episode(), {})
        self.assertEqual(out["tool_opens"], 1)
        self.assertEqual(len(out["calls"]), 0)
        (err,) = out["tool_errors"]
        self.assertIn("still open", err["error"])
        # decode_text_stream reports it too, as a failed call
        evs = decode_text_stream(out["text"], tok, 0)
        self.assertTrue(any(e["type"] == "tool_call" and e.get("error") for e in evs))

    def test_tiny_generator_frame_stats(self):
        import torch

        from genloop import TinyFrameGenerator
        from tiny_model import TinyDuplexLM
        torch.manual_seed(0)
        tok = self.tok
        model = TinyDuplexLM(text_card=tok.base_card, card=16, dim=32, layers=1, heads=2, depth_dim=16, depth_layers=1)
        model.resize_text_vocab(4)
        ex = dict(self.ex, text=self.ex["text"][:20], audio=np.full((16, 20), 3, dtype=np.int32))
        out = closed_loop(TinyFrameGenerator(model.eval()), tok, ex, episode(), {})
        self.assertEqual(len(out["frame_stats"]), 20)
        self.assertTrue(all(0.0 <= s["p_tool"] <= 1.0 for s in out["frame_stats"]))


class StitchAndListenTest(unittest.TestCase):
    def test_stitch_windows(self):
        def part(w, a, b, P=2):
            text = np.concatenate([[1, 1], np.arange(a, b)])
            return {"id": f"e#w{w}", "episode_id": "e", "text": text, "loss_mask": np.ones_like(text),
                    "neg_mask": np.zeros_like(text), "audio_mask": np.ones_like(text),
                    "audio": np.tile(text, (16, 1)), "meta": {"frame_offset": a - P, "prompt_frames": P}}

        (ex,) = stitch_windows([part(1, 10, 20), part(0, 0, 10)])
        self.assertEqual(ex["id"], "e")
        self.assertEqual(ex["text"].tolist(), [1, 1] + list(range(20)))
        self.assertEqual(ex["audio"].shape, (16, 22))
        self.assertEqual(ex["meta"]["frame_offset"], -2)

    def test_listen_noop_on_cpu_and_wav_writer(self):
        self.assertIsNone(FrameGenerator().decode_audio())
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x", "a.wav")
            write_wav(p, np.sin(np.linspace(0, 100, 2400)) * 0.5, 24000)
            import wave

            with wave.open(p) as w:
                self.assertEqual((w.getframerate(), w.getnframes(), w.getnchannels()), (24000, 2400, 1))


if __name__ == "__main__":
    unittest.main()
