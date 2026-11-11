"""Build the plain-conversation set (no tools) that train.py mixes into every
batch (data.plain_fraction) and evaluates on (forgetting check).

Two sources, both written in prepare.py's example format to
data.plain_prepared_dir (splits "train" and "val"):

  synth     Tool-free calls built with training/data's own episode builder and
            phrase banks (read-only use of its library): greeting, who is
            calling, small talk about Lewis, goodbye. Runs anywhere, no audio
            needed. A weak stand-in: same templates and TTS-free timing as the
            tool data, so it guards the text stream's general behaviour more
            than real conversational quality.

            python3 training/model/plain.py synth --config <cfg> --n-train 400 --n-val 50

  external  UNVERIFIED (never run here: needs moshi + a GPU + the data). A
            dataset in kyutai/moshi-finetune's format, e.g.
            kyutai/DailyTalkContiguous: a JSONL of {"path": "x.wav",
            "duration": s} with stereo 24 kHz WAVs (left = the model's
            speaker) and x.json next to each, {"alignments": [[word, [start,
            end], speaker], ...]}. Words of `--main-speaker` (default
            SPEAKER_MAIN, as moshi-finetune's keep_main_only) become the text
            stream; both channels are Mimi-encoded.

            python3 training/model/plain.py external --config <cfg> --jsonl /data/dailytalk/dailytalk.jsonl

Use the same tokenizer as the tool data: run prepare.py first (for the
dry-run chunk tokenizer, its vocabulary is built there).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from typing import Any, Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "data"))

from config_util import load_config, resolve_path  # noqa: E402
from prepare import (FRAME_RATE, N_CODEBOOKS_PER_CHANNEL, ZERO_TOKEN_ID, build_items, choose_windows,  # noqa: E402
                     get_tokenizer, place_items, prepare_episode, save_examples)

# --- synth: tool-free calls from training/data's builder ---------------------------

# (caller bank key, Elle bank key) exchanges; a call uses a random subset.
EXCHANGES = [("caller_reveal_company", "elle_after_company"), ("caller_reveal_role", "elle_after_role"),
             ("caller_ask_about", "elle_chitchat"), ("caller_chitchat", "elle_chitchat"),
             ("caller_ramble", "elle_backchannel")]


def _data_lib():
    """Import training/data's builder without letting its module names shadow ours."""
    if DATA_DIR not in sys.path:
        sys.path.append(DATA_DIR)
    import elle_time  # noqa: F401
    from episode import Ep  # type: ignore

    return Ep, elle_time


def synth_episode(index: int, split: str, seed: int = 0) -> Dict[str, Any]:
    """One tool-free call, deterministic in (seed, split, index)."""
    Ep, T = _data_lib()
    key = f"plain:{seed}:{split}:{index}"
    rng = random.Random(key)
    now = T.random_now(rng)
    ep = Ep("plain_smalltalk", "plain", index, rng, now, "en")
    ep.E("elle_greet", act="greeting", at=round(rng.uniform(0.1, 0.4), 2))
    for c_key, e_key in rng.sample(EXCHANGES, rng.randint(2, 4)):
        ep.C(c_key, act="plain")
        ep.E(e_key, act="plain")
    ep.C("caller_bye", act="bye")
    ep.E("elle_bye", act="bye")
    # ep_NNNNNN ids (the validator's format), in a range the tool data does not use.
    out = ep.finish(f"ep_{(800000 if split == 'train' else 900000) + index:06d}", split)
    out["labels"]["plain"] = True
    assert not out["labels"]["expected_calls"], "plain episodes must not call tools"
    return out


def validate_with_dataset(eps: List[dict]) -> int:
    """Run training/data's contract validator if it is importable; returns the
    number of episodes with issues (0 expected)."""
    try:
        if DATA_DIR not in sys.path:
            sys.path.append(DATA_DIR)
        from validate import validate_episode  # type: ignore
    except Exception:
        return -1
    return sum(1 for e in eps if validate_episode(e))


def synth_episodes(n_train: int, n_val: int, seed: int = 0) -> List[dict]:
    return [synth_episode(i, "train", seed) for i in range(n_train)] + [synth_episode(i, "val", seed) for i in range(n_val)]


def build_synth(cfg: dict, base: str, n_train: int, n_val: int, seed: int = 0, out: Optional[str] = None,
                episodes: Optional[str] = None) -> Dict[str, Any]:
    """With real audio (audio.mode mimi), pass `episodes`: the plain episodes
    after the same TTS pipeline as the tool data (synth -> retime -> assemble),
    i.e. the retimed JSONL the audio was assembled from. Otherwise they are
    generated here and get placeholder/zero audio."""
    out_dir = resolve_path(out or cfg["data"]["plain_prepared_dir"], base)
    if episodes:
        with open(episodes, encoding="utf-8") as f:
            eps = [json.loads(l) for l in f if l.strip()]
    else:
        eps = synth_episodes(n_train, n_val, seed)
    bad = validate_with_dataset(eps)
    tok = get_tokenizer(cfg, base, eps)
    examples = [x for ep in eps for x in prepare_episode(ep, cfg, tok, base)]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "episodes.jsonl"), "w") as f:
        for ep in eps:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")
    counts = save_examples(examples, out_dir)
    return {"source": "synth", "examples": counts, "validator_issues": bad,
            "frames": int(sum(len(e["text"]) for e in examples)), "out": out_dir}


# --- external: moshi-finetune format (UNVERIFIED) ----------------------------------

def build_external(cfg: dict, base: str, jsonl: str, main_speaker: str = "SPEAKER_MAIN",
                   val_share: float = 0.05, limit: int = 0, out: Optional[str] = None) -> Dict[str, Any]:  # pragma: no cover
    """UNVERIFIED: written against moshi-finetune's README/data format and
    prepare.mimi_audio (itself UNVERIFIED); needs moshi, sphn and a GPU."""
    from prepare import mimi_audio

    out_dir = resolve_path(out or cfg["data"]["plain_prepared_dir"], base)
    pcfg = cfg.get("prepare", {})
    acfg = dict(pcfg.get("audio", {}))
    max_frames = int(pcfg.get("max_frames", 1250))
    tok = get_tokenizer(cfg, base, [])
    root = os.path.dirname(os.path.abspath(jsonl))
    examples = []
    with open(jsonl) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if limit:
        rows = rows[:limit]
    for row in rows:
        wav = row["path"] if os.path.isabs(row["path"]) else os.path.join(root, row["path"])
        with open(os.path.splitext(wav)[0] + ".json") as af:
            al = json.load(af)
        al["alignments"] = [a for a in al.get("alignments", []) if len(a) < 3 or a[2] == main_speaker]
        T = int(math.ceil(float(row["duration"]) * FRAME_RATE))
        items = build_items([], tok, {"alignments": [[a[0], a[1], "SPEAKER_MAIN"] for a in al["alignments"]]})
        placed = place_items(items, T)
        audio = mimi_audio(wav, T, acfg)
        split = "val" if int(hashlib.sha1(row["path"].encode()).hexdigest(), 16) % 10000 < val_share * 10000 else "train"
        name = os.path.splitext(os.path.basename(row["path"]))[0]
        for w, (a, b) in enumerate(choose_windows(T, max_frames, [])):
            examples.append({
                "id": f"plain_{name}#w{w}", "episode_id": f"plain_{name}", "split": split, "frame_offset": a,
                "prompt_frames": 0, "text": placed["text"][a:b].astype(np.int32), "loss_mask": placed["loss_mask"][a:b],
                "neg_mask": np.zeros(b - a, dtype=np.uint8), "audio": audio[:, a:b].astype(np.int32),
                "audio_mask": (audio[0, a:b] != ZERO_TOKEN_ID).astype(np.uint8),
                "meta": {"frame_offset": a, "prompt_frames": 0, "serializer": "moshi-finetune alignments",
                         "audio_mode": "mimi", "spans": [], "dropped_tokens": placed["dropped_tokens"],
                         "max_word_delay_frames": placed["max_word_delay_frames"], "max_span_delay_frames": 0,
                         "negative": False, "scenario": "plain_external"}})
    counts = save_examples(examples, out_dir)
    return {"source": "external", "examples": counts, "out": out_dir}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth")
    s.add_argument("--config", required=True)
    s.add_argument("--n-train", type=int, default=400)
    s.add_argument("--n-val", type=int, default=50)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out")
    s.add_argument("--episodes", help="voiced, retimed plain episodes (JSONL) instead of generating them")
    g = sub.add_parser("episodes", help="write the plain episodes as JSONL, for the TTS pipeline")
    g.add_argument("--n-train", type=int, default=400)
    g.add_argument("--n-val", type=int, default=50)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", required=True)
    e = sub.add_parser("external")
    e.add_argument("--config", required=True)
    e.add_argument("--jsonl", required=True)
    e.add_argument("--main-speaker", default="SPEAKER_MAIN")
    e.add_argument("--val-share", type=float, default=0.05)
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--out")
    args = ap.parse_args(argv)
    if args.cmd == "episodes":
        eps = synth_episodes(args.n_train, args.n_val, args.seed)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for ep in eps:
                f.write(json.dumps(ep, ensure_ascii=False) + "\n")
        print(json.dumps({"episodes": len(eps), "validator_issues": validate_with_dataset(eps), "out": args.out}))
        return 0
    cfg, base = load_config(args.config)
    if args.cmd == "synth":
        res = build_synth(cfg, base, args.n_train, args.n_val, args.seed, args.out, args.episodes)
    else:
        res = build_external(cfg, base, args.jsonl, args.main_speaker, args.val_share, args.limit, args.out)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
