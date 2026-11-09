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
        before = {id(p): p.detach().clone() for p in groups["new_tokens"]}
        losses["loss"].backward()
        frozen_with_grad = [n for n, p in lm.named_parameters() if not p.requires_grad and p.grad is not None]
        self.assertEqual(frozen_with_grad, [])

        # Exactly as train.py builds it: no weight decay on the new-token rows.
        opt = torch.optim.AdamW([{"params": groups["lora"], "lr": 1e-3, "weight_decay": 0.1},
                                 {"params": groups["new_tokens"], "lr": 1e-3, "weight_decay": 0.0}],
                                betas=(0.9, 0.95), eps=1e-8)
        opt.step()
        for w in groups["new_tokens"]:
            old = [i for i in range(w.shape[0]) if i not in new_ids]
            self.assertTrue(torch.equal(w.detach()[old], before[id(w)][old]), "pretrained vocabulary must not move")

        sd = mb.trainable_state_dict(lm, new_ids)
        lm.eval()
        codes = torch.cat([text[:, None], audio], 1)
        with torch.no_grad():
            ref = lm.forward_train(codes)
        fresh, _tok2, ids2, _g = mb.load_real_model(self.cfg, device="cpu")
        mb.load_trainable_state_dict(fresh.lm, sd, ids2)
        self.assertGreater(merge_param_lora(fresh.lm), 0)
        self.assertEqual(sum(1 for m in fresh.lm.modules() if parametrize.is_parametrized(m)), 0)
        fresh.lm.eval()
        with torch.no_grad():
            got = fresh.lm.forward_train(codes)
        for a, b in ((ref.text_logits, got.text_logits), (ref.logits, got.logits)):
            ok = ~torch.isnan(a)
            self.assertTrue(torch.allclose(a[ok], b[ok], atol=1e-4), "merged model must match the trained one")


if __name__ == "__main__":
    unittest.main()
