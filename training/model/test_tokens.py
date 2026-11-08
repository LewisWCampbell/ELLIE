"""Special-token handling: tokenizer markers, vocab extension, LoRA, row masks."""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from textvocab import (EPAD_ID, PAD_ID, RESULT_CLOSE, RESULT_OPEN, SPECIAL_TOKENS, TOOL_CLOSE, TOOL_OPEN,  # noqa: E402
                       ByteTokenizer, ChunkTokenizer, seconds_to_frame)

try:
    import torch

    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    HAVE_TORCH = False

CALL = '<tool>{"id":"c1","name":"check_availability","args":{"start":"2026-11-12T14:00:00-08:00"}}</tool>'
RESULT = '<result id="c1">FREE: Thursday, Nov 12 at 2:00 PM PST is open.</result>'


class TokenizerTest(unittest.TestCase):
    def check(self, tok):
        ids = tok.special_ids
        self.assertEqual(list(ids), SPECIAL_TOKENS)
        self.assertEqual(sorted(ids.values()), list(range(tok.base_card, tok.base_card + 4)))
        enc = tok.encode(CALL)
        self.assertEqual(enc[0], ids[TOOL_OPEN])
        self.assertEqual(enc[-1], ids[TOOL_CLOSE])
        self.assertEqual(sum(1 for t in enc if tok.is_special(t)), 2)  # each marker is ONE id
        enc_r = tok.encode(RESULT)
        self.assertEqual(enc_r[0], ids[RESULT_OPEN])
        self.assertEqual(enc_r[-1], ids[RESULT_CLOSE])
        self.assertEqual(tok.decode(enc), CALL)
        self.assertEqual(tok.decode(enc_r), RESULT)
        # plain text never produces a marker id or a control id
        plain = tok.encode_plain("<tool> </result> hello")
        self.assertFalse(any(tok.is_special(t) or t in (0, 1, 2, 3) for t in plain))
        # control ids are skipped on decode
        self.assertEqual(tok.decode([PAD_ID, EPAD_ID] + tok.encode("hi") + [PAD_ID]), "hi")

    def test_byte(self):
        self.check(ByteTokenizer())

    def test_chunk(self):
        tok = ChunkTokenizer.build([CALL, RESULT, "He's free then. Want me to book it?"])
        self.check(tok)
        self.assertLess(len(tok.encode(CALL)), len(ByteTokenizer().encode(CALL)) / 1.5)
        self.assertEqual(tok.decode(tok.encode("Sí, mañana 10:30 ✓")), "Sí, mañana 10:30 ✓")  # byte fallback

    def test_frame_rule(self):
        self.assertEqual(seconds_to_frame(0.0), 0)
        self.assertEqual(seconds_to_frame(0.079), 0)
        self.assertEqual(seconds_to_frame(0.08), 1)  # 12.5 Hz: frame f covers [f*0.08, (f+1)*0.08)
        self.assertEqual(seconds_to_frame(2.5), 31)


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class VocabExtensionTest(unittest.TestCase):
    def setUp(self):
        from tiny_model import TinyDuplexLM

        torch.manual_seed(0)
        self.tok = ByteTokenizer()
        self.model = TinyDuplexLM(text_card=self.tok.base_card, card=16, dim=32, layers=1, heads=2,
                                  depth_dim=16, max_frames=64)
        self.text = torch.randint(4, self.tok.base_card, (2, 20))
        self.audio = torch.randint(0, 16, (2, 16, 20))

    def test_resize_preserves_base(self):
        m = self.model
        old_emb = m.text_emb.weight.detach().clone()
        m.eval()
        before = m(self.text, self.audio)[0].detach()
        new_ids = m.resize_text_vocab(4)
        self.assertEqual(new_ids, [self.tok.special_ids[s] for s in SPECIAL_TOKENS])
        self.assertEqual(m.text_card, self.tok.card)
        self.assertEqual(m.text_emb.weight.shape[0], self.tok.card + 1)
        self.assertTrue(torch.equal(m.text_emb.weight[: self.tok.base_card], old_emb[: self.tok.base_card]))
        # the initial-token row moved to the new end
        self.assertTrue(torch.equal(m.text_emb.weight[self.tok.card], old_emb[self.tok.base_card]))
        self.assertEqual(m.text_initial_token_id, self.tok.card)
        after = m(self.text, self.audio)[0].detach()
        self.assertEqual(after.shape[-1], self.tok.card)
        self.assertTrue(torch.allclose(after[..., : self.tok.base_card], before, atol=1e-5))  # old logits unchanged
        # new tokens start unlikely
        self.assertTrue((after[..., self.tok.base_card:].max(-1).values < after[..., : self.tok.base_card].max(-1).values).all())

    def test_lora_and_row_mask(self):
        from tiny_model import LoRALinear, setup_trainable

        m = self.model
        m.eval()
        new_ids = m.resize_text_vocab(4)
        before = m(self.text, self.audio)[0].detach()
        groups = setup_trainable(m, new_ids, {"rank": 4, "scaling": 2.0, "target": ["q", "k", "v", "o", "ff_in", "ff_out"]})
        self.assertTrue(any(isinstance(x, LoRALinear) for x in m.modules()))
        self.assertTrue(torch.allclose(m(self.text, self.audio)[0], before, atol=1e-5))  # B=0 at init
        trainable = {id(p) for g in groups.values() for p in g}
        for n, p in m.named_parameters():
            self.assertEqual(p.requires_grad, id(p) in trainable, n)
        text = self.text.clone()
        text[:, 5] = new_ids[0]
        out = m(text, self.audio)
        (out[0].sum() + out[1].sum()).backward()
        g = m.text_emb.weight.grad
        self.assertGreater(g[new_ids].abs().sum().item(), 0)
        mask = torch.ones(g.shape[0], dtype=torch.bool)
        mask[new_ids] = False
        self.assertEqual(g[mask].abs().sum().item(), 0.0)  # pretrained rows get no gradient
        self.assertEqual(m.text_linear.weight.grad[mask[:-1]].abs().sum().item(), 0.0)
        self.assertEqual(m.text_linear.bias.grad[mask[:-1]].abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
