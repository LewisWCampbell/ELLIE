"""Plain-conversation mixing: exact ratio, determinism, resumable sampler state,
separate tool/plain losses, and the forgetting baseline."""

import copy
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:
    import torch  # noqa: F401

    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    HAVE_TORCH = False

DATA_LIB = os.path.exists(os.path.join(HERE, "..", "data", "episode.py"))


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class SamplerTest(unittest.TestCase):
    def test_ratio_determinism_resume(self):
        from train import MixedSampler

        tool, plain = [{"i": i} for i in range(50)], [{"p": i} for i in range(20)]
        a = MixedSampler(tool, plain, 0.3, seed=3)
        counts = [len(a.next(4)[1]) for _ in range(100)]
        self.assertEqual(sum(counts), 120)  # exactly 30% of 400
        self.assertEqual(set(counts), {1, 2})
        # determinism
        b, c = MixedSampler(tool, plain, 0.3, 3), MixedSampler(tool, plain, 0.3, 3)
        self.assertEqual([b.next(4) for _ in range(10)], [c.next(4) for _ in range(10)])
        # resume from a saved state continues the exact same sequence
        st = json.loads(json.dumps(b.state(), default=list))  # survives a round trip like torch.save
        st = {k: (tuple(tuple(x) if isinstance(x, list) else x for x in v) if k in ("tool", "plain", "crop") else v)
              for k, v in st.items()}
        d = MixedSampler(tool, plain, 0.3, 99)
        d.load(st)
        self.assertEqual([b.next(4) for _ in range(10)], [d.next(4) for _ in range(10)])
        # no plain set = no plain draws
        e = MixedSampler(tool, [], 0.5, 1)
        self.assertEqual(sum(len(e.next(4)[1]) for _ in range(10)), 0)


@unittest.skipUnless(HAVE_TORCH and DATA_LIB, "torch or training/data library missing")
class MixedTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import plain
        import prepare
        from test_train import FIXTURES, make_cfg

        cls.tmp = tempfile.mkdtemp(prefix="elle_mix_test_")
        cfg = make_cfg(cls.tmp, steps=20)
        cfg["data"]["plain_prepared_dir"] = os.path.join(cls.tmp, "plain")
        cfg["data"]["plain_fraction"] = 0.34
        cfg["eval"]["every_steps"] = 10
        cfg["eval"]["ckpt_every_steps"] = 10
        cfg["eval"]["gates"] = {"plain_text_loss_delta": {"max": 0.5}}
        eps = prepare.load_episodes(FIXTURES)[:4]
        tok = prepare.get_tokenizer(cfg, cls.tmp, eps)
        prepare.save_examples([x for ep in eps for x in prepare.prepare_episode(ep, cfg, tok, cls.tmp)],
                              cfg["data"]["prepared_dir"])
        cls.plain_info = plain.build_synth(cfg, cls.tmp, n_train=6, n_val=3)
        cls.cfg = cfg
        import train

        cls.res = train.run(copy.deepcopy(cfg), cls.tmp, log=lambda *a: None)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_plain_set_is_tool_free_and_valid(self):
        self.assertIn(self.plain_info["validator_issues"], (0, -1))
        self.assertGreater(self.plain_info["examples"]["train"], 0)
        self.assertGreater(self.plain_info["examples"]["val"], 0)
        from prepare import load_examples

        for ex in load_examples(self.cfg["data"]["plain_prepared_dir"]):
            self.assertEqual(ex["meta"]["spans"], [])
            self.assertTrue(ex["loss_mask"].all())  # nothing forced: no tools, no results
            self.assertFalse(ex["neg_mask"].any())

    def test_separate_losses_and_ratio(self):
        h = self.res["history"]
        self.assertTrue(all("tool_loss" in r and "plain_loss" in r for r in h))  # bs 3 x 0.34: >= 1 plain per batch
        self.assertTrue(all(r["tool_text_loss"] > 0 and r["plain_text_loss"] > 0 for r in h))
        last = h[-1]
        total = last["plain_examples_seen"] + last["tool_examples_seen"]
        self.assertEqual(total, 20 * 3)
        self.assertEqual(last["plain_examples_seen"], 20)  # floor-carry of 0.34 * 3 per step = 1 each step

    def test_forgetting_tracked(self):
        h = [r for r in self.res["history"] if "eval_plain_text_loss" in r]
        self.assertEqual([r["step"] for r in h], [10, 20])
        for r in h:
            self.assertAlmostEqual(r["eval_plain_text_loss_delta"], r["eval_plain_text_loss"] - r["eval_plain_text_loss_base"])
        with open(os.path.join(self.res["run_dir"], "plain_forgetting.json")) as f:
            pf = json.load(f)
        self.assertEqual(pf["step"], 20)
        self.assertIn("pass", pf["gate"])

    def test_resume_continues_mixture(self):
        import train

        tmp2 = tempfile.mkdtemp(prefix="elle_mix_resume_")
        try:
            cfg = copy.deepcopy(self.cfg)
            cfg["run_dir"] = os.path.join(tmp2, "run")
            full = train.run(copy.deepcopy(cfg), self.tmp, log=lambda *a: None)["history"]
            shutil.rmtree(os.path.join(cfg["run_dir"], "checkpoints", "step_000020"))
            resumed = train.run(copy.deepcopy(cfg), self.tmp, resume=True, log=lambda *a: None)["history"]
            self.assertEqual(resumed[0]["step"], 11)
            for a, b in zip(full[10:], resumed):
                self.assertEqual(a["plain_examples_seen"], b["plain_examples_seen"])
                self.assertAlmostEqual(a["plain_loss"], b["plain_loss"], delta=2e-3)
                self.assertAlmostEqual(a["tool_loss"], b["tool_loss"], delta=2e-3)
            self.assertAlmostEqual(full[-1]["eval_plain_text_loss_base"], resumed[-1]["eval_plain_text_loss_base"])
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)

    def test_model_report_with_tiny_backend(self):
        """evaluate.py --mode model end to end on the tiny backend, incl. the
        --listen no-op and the forgetting line."""
        import evaluate
        from prepare import load_examples
        from test_train import FIXTURES

        with open(FIXTURES) as f:
            eps = [json.loads(l) for l in f][:2]
        prepared = load_examples(self.cfg["data"]["prepared_dir"])
        cfg = copy.deepcopy(self.cfg)
        cfg["eval"]["max_episodes"] = 2
        out = os.path.join(self.tmp, "eval")
        rep = evaluate.model_report(cfg, self.tmp, eps, prepared, {}, listen=2, out_dir=out)
        self.assertEqual(rep["metrics"]["episodes"], 2)
        self.assertEqual(rep["listen"]["files"], [])
        self.assertIn("no-op", rep["listen"]["note"])
        self.assertIn("plain_forgetting", rep)
        md = evaluate.markdown(rep)
        self.assertIn("plain-conversation text loss", md)
        self.assertIn("listen: no-op", md)


if __name__ == "__main__":
    unittest.main()
