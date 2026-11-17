"""Can the real tokenizer write every tool call exactly?

Encodes the call and result text of a sample of episodes with the configured
tokenizer and decodes it again. An unknown character (SentencePiece's <unk>
shares id 0 with the text stream's EPAD) would silently corrupt calls in the
training data and make them unparseable afterwards. Exits non-zero on any
mismatch. Runs on the VM as part of `prepare.sh gen`.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model"))

from config_util import load_config  # noqa: E402
from prepare import call_format_of  # noqa: E402
from runtime import parse_call_text  # noqa: E402
from serializer_adapter import contract_segments  # noqa: E402
from textvocab import EPAD_ID, TOOL_CLOSE, TOOL_OPEN, build_tokenizer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--limit", type=int, default=2000)
    a = ap.parse_args()
    cfg, base = load_config(a.config)
    tok = build_tokenizer(cfg.get("tokenizer", {}), base)
    fmt = call_format_of(cfg)
    bad, n, lens = [], 0, []
    with open(a.episodes) as f:
        for i, line in enumerate(f):
            if i >= a.limit:
                break
            for s in contract_segments(json.loads(line), fmt=fmt):
                if s["kind"] not in ("tool", "result"):
                    continue
                ids = tok.encode(s["text"])
                n += 1
                if s["kind"] == "tool":
                    lens.append(len(ids))
                    body = s["text"][len(TOOL_OPEN):-len(TOOL_CLOSE)]
                    try:
                        parse_call_text(tok.decode(ids[1:-1]))
                        ok = tok.decode(ids[1:-1]) == body
                    except Exception:  # noqa: BLE001
                        ok = False
                else:
                    ok = tok.decode(ids) == s["text"]
                if (EPAD_ID in ids[1:-1]) or not ok:
                    bad.append((s["text"][:80], tok.decode(ids)[:80]))
    print(json.dumps({"format": fmt, "spans_checked": n, "bad": len(bad),
                      "call_tokens_mean": round(sum(lens) / max(1, len(lens)), 1),
                      "call_tokens_max": max(lens, default=0)}))
    for t, d in bad[:5]:
        print("tokenizer_check: MISMATCH", repr(t), "->", repr(d))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
