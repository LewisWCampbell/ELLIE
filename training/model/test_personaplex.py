"""The real-model path, run against PersonaPlex's own code at miniature size.

Skipped unless NVIDIA's PersonaPlex `moshi` package is installed (it is on the
GPU VM). It builds PersonaPlex's real LMModel with tiny dimensions, so it runs
on a CPU in seconds, and drives moshi_backend's real loader through it: the
checkpoint lookup, weight-level LoRA, the four new tool tokens, the training
forward, our loss, an optimiser step, then save -> reload -> merge.

Every failure this guards against was found by running it, not by reading
code: PersonaPlex has no CheckpointInfo, no training forward() (it is called
forward_train), and reads its attention and feed-forward weights directly, so
wrapping layers for LoRA cannot work.
"""

import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:
    import torch
    from moshi.models import loaders, lm as moshi_lm
    HAVE_PERSONAPLEX = not hasattr(loaders, "CheckpointInfo") and hasattr(moshi_lm.LMModel, "forward_train")
except Exception:  # noqa: BLE001 - any import failure means "not installed here"
    HAVE_PERSONAPLEX = False


@unittest.skipUnless(HAVE_PERSONAPLEX, "PersonaPlex's moshi package is not installed")
class MiniPersonaPlex(unittest.TestCase):
    def setUp(self):
        import huggingface_hub

        self.tmp = tempfile.mkdtemp()
        self._real_download = huggingface_hub.hf_hub_download
        self._real_get_lm = loaders.get_moshi_lm

        def fake_download(repo, name, *a, **k):
            path = os.path.join(self.tmp, name)
            open(path, "ab").close()
            return path

        mini = dict(loaders._lm_kwargs)
        mini.update(dim=64, num_heads=4, num_layers=2, hidden_scale=2.0, context=64, depformer_dim=32,
                    depformer_dim_feedforward=64, depformer_num_heads=2, depformer_num_layers=1)
        self.mini = mini

        def mini_lm(filename, copy_missing_weights=True, device="cpu", dtype=torch.float32, delays=None, cpu_offload=False):
            torch.manual_seed(0)  # the same "pretrained" base on every load
            return moshi_lm.LMModel(device=device, dtype=dtype, **mini)

        huggingface_hub.hf_hub_download = fake_download
        loaders.get_moshi_lm = mini_lm
        self.cfg = {"model": {"hf_repo": "nvidia/personaplex-7b-v1", "dtype": "float32"},
                    "lora": {"rank": 4, "scaling": 2.0, "impl": "auto",
                             "include_prefixes": ["transformer.", "depformer."]},
                    "special_tokens_count": 4}

    def tearDown(self):
        import huggingface_hub

        huggingface_hub.hf_hub_download = self._real_download
        loaders.get_moshi_lm = self._real_get_lm

    def test_train_step_save_reload_merge(self):
        import moshi_backend as mb
        from param_lora import merge_param_lora
        from torch.nn.utils import parametrize
        from train import compute_loss

        adapter, _tok, new_ids, groups = mb.load_real_model(self.cfg, device="cpu")
        lm = adapter.lm
        self.assertEqual(new_ids, [32000, 32001, 32002, 32003])
        self.assertGreater(len(groups["lora"]), 0)
        self.assertGreater(sum(1 for m in lm.modules() if parametrize.is_parametrized(m)), 0)

        B, T = 2, 12
        torch.manual_seed(1)
        text = torch.randint(0, adapter.text_card, (B, T))
        text[0, 3] = new_ids[0]  # make sure a tool token appears in the input
        audio = torch.randint(0, self.mini["card"], (B, self.mini["n_q"], T))
        out = adapter(text, audio)
        self.assertEqual(out[1].shape[1], 8, "PersonaPlex predicts 8 audio codebooks")

        batch = {"text": text, "audio": audio, "loss_mask": torch.ones(B, T),
                 "neg_mask": torch.zeros(B, T), "audio_mask": torch.ones(B, T)}
        lcfg = {"text_padding_weight": 0.5, "special_token_weight": 2.0, "audio_weight": 1.0,
                "first_codebook_weight_multiplier": 100.0}
        losses = compute_loss(out, batch, lcfg, new_ids, negatives="unlikelihood", ul_weight=0.5)
        self.assertTrue(torch.isfinite(losses["loss"]))
        tables = {n: t.detach().clone() for n, t in self._base_tables(lm).items()}
        before = [p.detach().clone() for p in groups["new_tokens"]]
        losses["loss"].backward()
        frozen_with_grad = [n for n, p in lm.named_parameters() if not p.requires_grad and p.grad is not None]
        self.assertEqual(frozen_with_grad, [])
        # The new rows are float32 parameters of their own, one per table, 4 rows each.
        self.assertEqual([(p.dtype, p.shape[0]) for p in groups["new_tokens"]], [(torch.float32, 4)] * 3)
        self.assertTrue(all(p.grad is not None and p.grad.abs().sum() > 0 for p in groups["new_tokens"]))

        # Exactly as train.py builds it: no weight decay on the new-token rows.
        opt = torch.optim.AdamW([{"params": groups["lora"], "lr": 1e-3, "weight_decay": 0.1},
                                 {"params": groups["new_tokens"], "lr": 1e-3, "weight_decay": 0.0}],
                                betas=(0.9, 0.95), eps=1e-8)
        opt.step()
        for n, t in self._base_tables(lm).items():
            self.assertTrue(torch.equal(t[:new_ids[0]], tables[n][:new_ids[0]]), f"{n}: pretrained vocabulary must not move")
        self.assertTrue(all(not torch.equal(p, b) for p, b in zip(groups["new_tokens"], before)), "new rows must train")
        # ...and every read of the table sees them.
        self.assertTrue(torch.equal(lm.text_emb.weight[new_ids].float(), groups["new_tokens"][0].detach()))

        sd = mb.trainable_state_dict(lm, new_ids)
        lm.eval()
        codes = torch.cat([text[:, None], audio], 1)
        with torch.no_grad():
            ref = lm.forward_train(codes)
        fresh, _tok2, ids2, _g = mb.load_real_model(self.cfg, device="cpu")
        mb.load_trainable_state_dict(fresh.lm, sd, ids2)
        self.assertGreater(merge_param_lora(fresh.lm), 0)
        from param_lora import merge_spliced_rows
        self.assertEqual(merge_spliced_rows(fresh.lm), 3)
        self.assertEqual(sum(1 for m in fresh.lm.modules() if parametrize.is_parametrized(m)), 0)
        fresh.lm.eval()
        with torch.no_grad():
            got = fresh.lm.forward_train(codes)
        for a, b in ((ref.text_logits, got.text_logits), (ref.logits, got.logits)):
            ok = ~torch.isnan(a)
            self.assertTrue(torch.allclose(a[ok], b[ok], atol=1e-4), "merged model must match the trained one")

    @staticmethod
    def _base_tables(lm):
        from torch.nn.utils import parametrize
        out = {}
        for n in ("text_emb", "depformer_text_emb", "text_linear"):
            m = getattr(lm, n)
            out[n] = m.parametrizations.weight.original if parametrize.is_parametrized(m, "weight") else m.weight
        return out

    def test_native_prompt_codes_match_personaplex(self):
        from prepare import NATIVE_AGENT_SILENCE, NATIVE_USER_SINE
        self.assertEqual(NATIVE_AGENT_SILENCE, moshi_lm.SILENCE_TOKENS.tolist())
        self.assertEqual(NATIVE_USER_SINE, moshi_lm.SINE_TOKENS.tolist())

    def test_first_run_checkpoint_still_loads(self):
        """v1 checkpoints saved the new rows as bfloat16 slices of the tables;
        they load into the float32 rows (and a LoRA mismatch is an error)."""
        import moshi_backend as mb

        adapter, _tok, new_ids, groups = mb.load_real_model(self.cfg, device="cpu")
        sd = mb.trainable_state_dict(adapter.lm, new_ids)
        for k in [k for k in sd if k.startswith("new_tokens.")]:
            sd[k] = (torch.randn_like(sd[k]) * 0.01).to(torch.bfloat16)
        fresh, _t, ids2, g2 = mb.load_real_model(self.cfg, device="cpu")
        mb.load_trainable_state_dict(fresh.lm, sd, ids2)
        self.assertTrue(torch.equal(g2["new_tokens"][0].detach(), sd["new_tokens.text_emb"].float()))
        bad = {k: v for k, v in sd.items() if not k.endswith("lora_A")}
        with self.assertRaises(AssertionError):
            mb.load_trainable_state_dict(fresh.lm, bad, ids2)

    def test_streaming_generator_matches_training_forward_native_prompt(self):
        from prepare import NATIVE_AGENT_SILENCE, NATIVE_USER_SINE
        self._parity(prompt=(NATIVE_AGENT_SILENCE, NATIVE_USER_SINE))

    def test_streaming_generator_matches_training_forward(self):
        self._parity(prompt=None)

    def _parity(self, prompt):
        """The live generator must see exactly what training saw. Drive the real
        MoshiFrameGenerator frame by frame with an example's own text and audio
        (prompt frames with -1 audio, as prepare.py writes them) and compare the
        probabilities it computes before sampling with forward_train's on the
        same example. This is the check that would have caught the 7B run's
        zero tool calls before any GPU time: the generator fed Mimi silence and
        self-generated audio where training had -1, and LMGen's first step
        swallowed frame 0."""
        import numpy as np

        import moshi_backend as mb
        from param_lora import merge_param_lora

        adapter, _tok, new_ids, _g = mb.load_real_model(self.cfg, device="cpu")
        lm = adapter.lm
        torch.manual_seed(3)
        for p in lm.parameters():  # make the LoRA deltas non-zero, as after training
            if p.requires_grad:
                p.data.normal_(0, 0.05)
        merge_param_lora(lm)
        lm.eval()

        T, P = 24, 6
        g = torch.Generator().manual_seed(4)
        text = torch.randint(4, 300, (T,), generator=g)
        text[P + 5] = new_ids[0]
        text[P + 9] = new_ids[1]
        audio = torch.randint(0, self.mini["card"], (16, T), generator=g)
        if prompt is None:
            audio[:, :P] = -1  # prompt frames, zero layout: zero embedding on all 16 streams
        else:  # native layout: Mimi silence on Elle's channel, sine on the caller's
            audio[:8, :P] = torch.tensor(prompt[0])[:, None]
            audio[8:, :P] = torch.tensor(prompt[1])[:, None]
        audio[8:, T - 3:] = -1  # caller audio ran out (Mimi tail)
        with torch.no_grad():
            ref = lm.forward_train(torch.cat([text[None, None], audio[None]], 1))
        ref_p = torch.softmax(ref.text_logits[0, 0].float(), -1)  # [T, V]

        class FakeMimi:
            sample_rate, frame_rate = 24000, 12.5

            def encode(self, x):
                return torch.zeros(1, 8, max(1, x.shape[-1] // 1920), dtype=torch.long)

        gen = mb.MoshiFrameGenerator(lm, FakeMimi(), None, {"temperature_text": 0.0}, device="cpu")
        gen.track_ids = {"tool": new_ids[0], "pad": 3, "epad": 0}
        gen.frame_stats = []
        ex = {"text": text.numpy(), "audio": audio.numpy(), "meta": {"prompt_frames": P}}
        gen.start(ex, {"seed": 0}, {})
        try:
            for f in range(T):
                tk = gen.step(f, audio[8:, f].numpy(), int(text[f]), [], elle_codes=audio[:8, f].numpy())
                self.assertEqual(tk, int(text[f]))
        finally:
            gen.finish()
        self.assertEqual(len(gen.frame_stats), T)
        for f, st in enumerate(gen.frame_stats):
            self.assertIsNotNone(st, f"frame {f}: no logits seen")
            for key, tid in (("p_tool", new_ids[0]), ("p_pad", 3), ("p_epad", 0)):
                self.assertAlmostEqual(st[key], float(ref_p[f, tid]), delta=1e-4,
                                       msg=f"frame {f} {key}: streaming {st[key]:.6f} vs training {float(ref_p[f, tid]):.6f}")
            self.assertEqual(st["argmax"], int(ref_p[f].argmax()), f"frame {f}: argmax differs")


if __name__ == "__main__":
    unittest.main()
