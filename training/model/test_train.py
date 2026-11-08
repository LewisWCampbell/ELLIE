"""Dry-run training on CPU: loss falls on a tiny overfit set, masking holds in
the loss, checkpoints round-trip, runs are deterministic, resume works."""

import copy
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:
    import torch

    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    HAVE_TORCH = False

FIXTURES = os.path.join(HERE, "fixtures", "episodes.jsonl")


def make_cfg(tmp: str, steps: int = 60) -> dict:
    return {
        "seed": 7, "run_dir": os.path.join(tmp, "run"),
        "data": {"episodes": FIXTURES, "prepared_dir": os.path.join(tmp, "prepared")},
        "tokenizer": {"kind": "chunk", "vocab": os.path.join(tmp, "prepared", "vocab.json")},
        "model": {"backend": "tiny"},
        "tiny": {"card": 32, "dim": 48, "layers": 2, "heads": 4, "depth_dim": 16, "depth_layers": 1,
                 "max_frames": 512, "pretrain_steps": 15, "pretrain_lr": 3e-3},
        "lora": {"rank": 8, "scaling": 2.0, "target": ["q", "k", "v", "o", "ff_in", "ff_out"]},
        "prepare": {"max_frames": 512, "audio": {"mode": "placeholder", "card": 32}},
        "train": {"backend_device": "cpu", "max_steps": steps, "batch_size": 3, "seq_frames": 512, "num_threads": 2,
                  "split": "train", "negatives": "unlikelihood", "unlikelihood_weight": 0.5,
                  "optim": {"lr": 2e-2, "new_token_lr": 3e-2, "weight_decay": 0.0, "pct_start": 0.1}},
        "loss": {"text_padding_weight": 0.5, "special_token_weight": 2.0, "audio_weight": 0.25,
                 "first_codebook_weight_multiplier": 2.0},
        "eval": {"every_steps": steps, "split": "train", "ckpt_every_steps": steps // 2, "keep_checkpoints": 2,
                 "max_episodes": 3, "gen": {"elle_audio": "teacher"}},
    }


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class DryRunTrainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import prepare

        cls.tmp = tempfile.mkdtemp(prefix="elle_train_test_")
        cls.cfg = make_cfg(cls.tmp)
        eps = prepare.load_episodes(FIXTURES)[:4]  # tiny overfit set incl. two calls and a no-tool episode
        tok = prepare.get_tokenizer(cls.cfg, cls.tmp, eps)
        exs = [x for ep in eps for x in prepare.prepare_episode(ep, cls.cfg, tok, cls.tmp)]
        prepare.save_examples(exs, cls.cfg["data"]["prepared_dir"])
        import train

        cls.res = train.run(copy.deepcopy(cls.cfg), cls.tmp, log=lambda *a: None)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_loss_decreases(self):
        h = self.res["history"]
        first = sum(r["text_loss"] for r in h[:3]) / 3
        last = sum(r["text_loss"] for r in h[-3:]) / 3
        self.assertLess(last, first * 0.8, (first, last))
        self.assertIn("eval_text_loss", h[-1])

    def test_result_frames_do_not_affect_loss(self):
        """Changing the forced result tokens' *targets* must not change the loss."""
        import train
        from prepare import load_examples

        ex = load_examples(self.cfg["data"]["prepared_dir"], "train")
        import random

        batch = train.collate(ex[:2], 512, random.Random(0))
        model = self.res["model"]
        model.eval()
        with torch.no_grad():
            out = model(batch["text"], batch["audio"])
            l1 = train.compute_loss(out, batch, self.cfg["loss"], self.res["new_ids"])["text_loss"]
            b2 = dict(batch)
            t2 = batch["text"].clone()
            forced = batch["loss_mask"] == 0
            t2[forced & (t2 >= 0)] = 5  # garbage targets at forced positions
            b2["text"] = t2
            l2 = train.compute_loss(out, b2, self.cfg["loss"], self.res["new_ids"])["text_loss"]
        self.assertAlmostEqual(float(l1), float(l2), places=5)
        self.assertTrue(forced.any())

    def test_checkpoint_roundtrip(self):
        import train

        model2, _tok, _ids, ck = train.load_for_eval(self.cfg, self.tmp)
        self.assertTrue(ck.endswith("step_000060"))
        text = torch.randint(4, 100, (1, 40))
        audio = torch.randint(0, 32, (1, 16, 40))
        m1 = self.res["model"]
        m1.eval()
        with torch.no_grad():
            a, b = m1(text, audio)[0], model2(text, audio)[0]
        self.assertTrue(torch.allclose(a, b, atol=1e-5))
        kept = os.listdir(os.path.join(self.cfg["run_dir"], "checkpoints"))
        self.assertEqual(sorted(kept), ["step_000030", "step_000060"])

    def test_deterministic_and_resume(self):
        import train

        tmp2 = tempfile.mkdtemp(prefix="elle_train_det_")
        try:
            cfg = make_cfg(self.tmp, steps=60)
            cfg["run_dir"] = os.path.join(tmp2, "run")
            r2 = train.run(copy.deepcopy(cfg), self.tmp, log=lambda *a: None)
            l1 = [round(r["loss"], 5) for r in self.res["history"]]
            l2 = [round(r["loss"], 5) for r in r2["history"]]
            self.assertEqual(l1, l2)
            # resume from step 15 reproduces steps 16..30
            shutil.rmtree(os.path.join(cfg["run_dir"], "checkpoints", "step_000060"))
            r3 = train.run(copy.deepcopy(cfg), self.tmp, resume=True, log=lambda *a: None)
            self.assertEqual(r3["history"][0]["step"], 31)
            l3 = [r["loss"] for r in r3["history"]]
            self.assertEqual(len(l3), 30)
            for a, b in zip(l3, l1[30:]):
                self.assertAlmostEqual(a, b, delta=2e-3)
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)

    def test_closed_loop_generation_runs(self):
        import json

        from genloop import TinyFrameGenerator, closed_loop
        from prepare import load_examples
        import train

        model, tok, _ids, _ = train.load_for_eval(self.cfg, self.tmp)
        ex = load_examples(self.cfg["data"]["prepared_dir"], "train")[0]
        with open(FIXTURES) as f:
            ep = json.loads(f.readline())
        out = closed_loop(TinyFrameGenerator(model), tok, ex, ep, {"elle_audio": "generate"})
        self.assertEqual(len(out["text"]), len(ex["text"]))
        ids = tok.special_ids
        # the sampler can never emit a result marker by itself
        n_open = out["text"].count(ids["<result id="])
        self.assertLessEqual(n_open, out["text"].count(ids["<tool>"]) if n_open else 0)

if __name__ == "__main__":
    unittest.main()
