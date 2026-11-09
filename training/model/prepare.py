"""Episodes (JSONL, CONTRACT.md) -> frame-aligned training examples.

One example = one window of an episode, at Mimi's 12.5 Hz frame rate:

  text       int32 [T]      Elle's inner-monologue text stream, one token per frame
  loss_mask  uint8 [T]      1 where the text token is predicted, 0 where forced
                            (result spans and the EPAD that announces them)
  neg_mask   uint8 [T]      1 on frames a labelled negative gets wrong (unlikelihood)
  audio      int32 [16, T]  rows 0-7 Elle's Mimi codebooks, rows 8-15 the caller's
                            (Moshi convention: model channel first), -1 = no input
  audio_mask uint8 [T]      1 where Elle's audio codes are real targets

Text placement follows Moshi's Interleaver (kyutai-labs/moshi-finetune
`finetune/data/interleaver.py`, read 2026-11-07): a word starts at the frame
containing its start time, its tokens go out one per frame, the frame before a
word start becomes EPAD if it was PAD, everything else is PAD. Two deliberate
differences, both needed for tool spans:
  * nothing is ever dropped (Moshi's default discards a word's leftover tokens
    when the next word starts; that would truncate a tool call). Late tokens
    queue, like Moshi's `keep_and_shift=True`.
  * <tool>/<result> spans are atomic: they start at the first free frame at or
    after their time and occupy consecutive frames; speech words that land
    inside a span are deferred behind it. The displacement is reported.

Audio: `audio.mode` is `placeholder` (deterministic fake codes, dry runs only),
`zero` (all -1: text-only training), or `mimi` (encode the stereo WAV rendered
by the dataset's TTS stage; needs the moshi package, UNVERIFIED here).

Usage:
  python3 training/model/prepare.py --config training/model/config/dryrun.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from config_util import load_config, resolve_path  # noqa: E402
from serializer_adapter import episode_segments  # noqa: E402
from textvocab import (EPAD_ID, FRAME_RATE, PAD_ID, ZERO_TOKEN_ID, ChunkTokenizer, TextTokenizer,  # noqa: E402
                       build_tokenizer, seconds_to_frame)

N_CODEBOOKS_PER_CHANNEL = 8


# --- word timing -------------------------------------------------------------

def spread_words(text: str, t: float, dur: float) -> List[Tuple[str, float, float]]:
    """No TTS alignment available: spread words over [t, t+dur] in proportion
    to their length (+1 for the gap). Approximate, and labelled as such."""
    words = text.split()
    if not words:
        return []
    dur = max(dur, 0.08 * len(words))
    weights = [len(w) + 1 for w in words]
    total = float(sum(weights))
    out, cur = [], t
    for w, wt in zip(words, weights):
        d = dur * wt / total
        out.append((w, cur, cur + d))
        cur += d
    return out


def load_alignment(json_path: str) -> Optional[dict]:
    """Trainer's expected alignment format (moshi-finetune's, extended):
    {"alignments": [[word, [start, end], "SPEAKER_MAIN"], ...],   # Elle only
     "spans": [{"kind": "tool"|"result", "t": s, "text": "...", "id": "c1"}]}  # optional
    """
    if not os.path.exists(json_path):
        return None
    with open(json_path) as f:
        return json.load(f)


# --- retiming: make the episode physically possible on a 12.5 Hz text stream -----

def retime_episode(ep: dict, tok: TextTokenizer, margin_frames: int = 1,
                   call_after_utterance_window_s: float = 0.6) -> dict:
    """The contract's events treat <tool> and <result> spans as instantaneous,
    but on Moshi's text stream every token takes one 80 ms frame: a ~45-token
    call takes ~3.6 s to emit and a ~35-token result ~2.8 s to inject. Without
    retiming, a result would arrive before its call finished, and Elle's audio
    would state the answer while the result was still being injected, which is
    exactly the premature-claim behaviour we must not teach. This returns a
    copy where:
      * a call that comes just before an Elle utterance (within the window)
        moves to just after that utterance starts, so her words go out first;
      * a result arrives no earlier than call-span end + latency_s;
      * everything after a result shifts so it starts after the injection ends,
        keeping its original gap.
    Event order, ids, labels and indices are unchanged. Renderers of stereo
    audio must use the retimed episode (prepare.py --write-retimed)."""
    import copy

    from serializer_adapter import tool_call_text, tool_result_text

    ep = copy.deepcopy(ep)
    evs = ep["events"]
    elle = [e for e in evs if e.get("type") == "speech" and e.get("ch") == "elle"]
    result_t = {e.get("id"): float(e["t"]) for e in evs if e.get("type") == "tool_result"}
    for e in evs:
        if e.get("type") == "tool_call":
            limit = min(e["t"] + call_after_utterance_window_s, result_t.get(e.get("id"), float("inf")))
            nxt = [u["t"] for u in elle if e["t"] < u["t"] < limit]
            if nxt:
                e["t"] = round(min(nxt) + 0.001, 3)
    order = sorted(range(len(evs)), key=lambda i: (evs[i]["t"], i))
    orig = {i: float(evs[i]["t"]) for i in order}
    call_end: Dict[str, float] = {}
    shift = 0.0
    for i in order:
        e = evs[i]
        t = orig[i] + shift
        if e.get("type") == "tool_call":
            # words of an Elle utterance in progress are flushed before the span
            pending = 0
            for u in elle:
                ut = float(u["t"])
                if ut <= t < ut + float(u.get("dur") or 0.0):
                    for w, ws, _ in spread_words(u["text"], ut, float(u.get("dur") or 0.0)):
                        if ws > t:
                            pending += len(tok.encode_word(w, first=False))
            n = len(tok.encode(tool_call_text(e))) + margin_frames
            call_end[e.get("id")] = t + (pending + n) / FRAME_RATE
        elif e.get("type") == "tool_result":
            lat = float(e.get("latency_s") or 0.0)
            t = max(t, call_end.get(e.get("id"), t) + lat)
            n = len(tok.encode(tool_result_text(e))) + margin_frames
            shift = t + n / FRAME_RATE - orig[i]
        e["t"] = round(t, 3)
    ep.setdefault("meta", {})["retimed"] = True
    return ep


# --- text stream ---------------------------------------------------------------

def _event_at(segments: List[dict], t: float) -> Optional[int]:
    for s in segments:
        if s["kind"] == "speech" and s["t"] - 0.05 <= t <= s["t"] + (s.get("dur") or 0.0) + 0.05:
            return s.get("event_index")
    return None


def bad_event_indices(ep: dict) -> set:
    """Event indices a negative episode gets wrong, from the same metric code the
    evaluator uses: premature claim utterances, the tool_call of an unsafe write,
    and utterances that narrate a silent write. Used for the unlikelihood loss."""
    if not ep.get("labels", {}).get("negative"):
        return set()
    from metrics import narration_events, score_episode

    sc = score_episode(ep)
    bad = {c["event_index"] for c in sc["claims"]["premature"]}
    unsafe = {w["id"] for w in sc["writes"] if w["problems"]}
    narrated = {s["id"] for s in sc["silent_writes"] if s["narrated"]}
    evs = ep["events"]
    for i, e in enumerate(evs):
        if e.get("type") == "tool_call" and e.get("id") in unsafe:
            bad.add(i)
        if e.get("type") == "tool_call" and e.get("id") in narrated:
            for s in narration_events(e, evs):
                bad.add(next(j for j, x in enumerate(evs) if x is s))
    return bad


def build_items(segments: List[dict], tok: TextTokenizer, alignment: Optional[dict] = None) -> List[dict]:
    """Segments -> placement items (one per word, one per span)."""
    items: List[dict] = []
    words: List[Tuple[str, float, float]] = []
    if alignment and alignment.get("alignments"):
        words = [(a[0], float(a[1][0]), float(a[1][1])) for a in alignment["alignments"]
                 if len(a) < 3 or a[2] in ("SPEAKER_MAIN", "elle", 0)]
        words = [(w, a, b, _event_at(segments, a)) for w, a, b in words]
    else:
        for s in segments:
            if s["kind"] == "speech":
                words.extend((w, a, b, s.get("event_index")) for w, a, b in spread_words(s["text"], s["t"], s.get("dur") or 0.0))
    for n, (w, start, end, ev_idx) in enumerate(words):
        ids = tok.encode_word(w, first=(n == 0))
        if ids:
            items.append({"kind": "word", "t": start, "end": end, "tokens": ids, "loss": True,
                          "atomic": False, "text": w, "event_index": ev_idx})
    for s in segments:
        if s["kind"] in ("tool", "result"):
            items.append({"kind": s["kind"], "t": s["t"], "end": None, "tokens": tok.encode(s["text"]),
                          "loss": bool(s["loss"]), "atomic": True, "text": s["text"], "id": s.get("id"),
                          "event_index": s.get("event_index")})
    # Stable chronological order; at equal times a span goes after the word that
    # was already being spoken.
    items.sort(key=lambda it: (seconds_to_frame(it["t"]), 0 if it["kind"] == "word" else 1, it["t"]))
    return items


def place_items(items: List[dict], T: int, bad_events: frozenset = frozenset()) -> Dict[str, Any]:
    """Frame placement. Returns text, loss_mask, neg_mask (frames to push
    *down* in a negative episode) and per-span placement info."""
    text = np.full(T, PAD_ID, dtype=np.int64)
    loss = np.ones(T, dtype=np.uint8)
    neg = np.zeros(T, dtype=np.uint8)
    queue: deque = deque()  # (token, loss, item_index, is_first)
    starts: Dict[int, int] = {}
    ends: Dict[int, int] = {}
    consumed = [False] * len(items)
    started_utts: set = set()

    def enqueue(j: int) -> None:
        it = items[j]
        consumed[j] = True
        if it["kind"] == "word" and it.get("event_index") is not None:
            started_utts.add(it["event_index"])
        for k, tk in enumerate(it["tokens"]):
            queue.append((tk, it["loss"], j, k == 0))

    i = 0
    for f in range(T):
        while i < len(items) and seconds_to_frame(items[i]["t"]) <= f:
            if not consumed[i]:
                if items[i]["atomic"]:
                    # Finish the utterance in progress first (its words go out
                    # early, i.e. text runs ahead of audio, which Moshi does
                    # anyway) so the span never splits a sentence and text
                    # never lags far behind its audio.
                    for j in range(i + 1, len(items)):
                        if (not consumed[j] and items[j]["kind"] == "word"
                                and items[j].get("event_index") in started_utts):
                            enqueue(j)
                enqueue(i)
            i += 1
        if not queue:
            continue
        tk, l, idx, first = queue.popleft()
        if first:
            starts[idx] = f
            if f > 0 and text[f - 1] == PAD_ID:
                text[f - 1] = EPAD_ID
                if not l:  # the model cannot know when a result will arrive
                    loss[f - 1] = 0
        text[f] = tk
        loss[f] = 1 if l else 0
        ends[idx] = f + 1
        it = items[idx]
        if it.get("event_index") is not None and it["event_index"] in bad_events and (first or not it["atomic"]):
            neg[f] = 1
    dropped = len(queue) + sum(len(it["tokens"]) for j, it in enumerate(items) if not consumed[j])
    spans, max_word_delay, max_span_delay = [], 0, 0
    for idx, it in enumerate(items):
        if idx not in starts:
            continue
        delay = starts[idx] - seconds_to_frame(it["t"])
        if it["atomic"]:
            max_span_delay = max(max_span_delay, delay)
            spans.append({"kind": it["kind"], "id": it.get("id"), "nominal_frame": seconds_to_frame(it["t"]),
                          "start_frame": starts[idx], "end_frame": ends[idx], "forced": not it["loss"]})
        else:
            max_word_delay = max(max_word_delay, delay)
    return {"text": text, "loss_mask": loss, "neg_mask": neg, "spans": spans, "dropped_tokens": int(dropped),
            "max_word_delay_frames": int(max_word_delay), "max_span_delay_frames": int(max_span_delay)}


# --- audio ---------------------------------------------------------------------

def speech_intervals(ep: dict, ch: str) -> List[Tuple[float, float]]:
    return [(float(e["t"]), float(e["t"]) + float(e.get("dur") or 0.0))
            for e in ep.get("events", []) if e.get("type") == "speech" and e.get("ch") == ch]


def placeholder_audio(ep: dict, T: int, card: int, seed: int) -> np.ndarray:
    """Deterministic fake Mimi codes: random codes where a channel speaks, a fixed
    per-codebook 'silence' code elsewhere. Exists only so the dry run exercises
    the audio path; it carries no acoustic information."""
    rng = np.random.default_rng(seed)
    out = np.empty((2 * N_CODEBOOKS_PER_CHANNEL, T), dtype=np.int64)
    for c, ch in enumerate(("elle", "caller")):
        speaking = np.zeros(T, dtype=bool)
        for a, b in speech_intervals(ep, ch):
            speaking[seconds_to_frame(a):min(T, int(math.ceil(b * FRAME_RATE)))] = True
        for k in range(N_CODEBOOKS_PER_CHANNEL):
            row = np.full(T, k % card, dtype=np.int64)
            row[speaking] = rng.integers(0, card, size=int(speaking.sum()))
            out[c * N_CODEBOOKS_PER_CHANNEL + k] = row
    return out


def mimi_audio(wav_path: str, T: int, mimi_cfg: dict) -> np.ndarray:  # pragma: no cover - needs GPU stack
    """Encode a stereo 24 kHz WAV (ch0 Elle, ch1 caller) with Mimi.

    UNVERIFIED: written against kyutai-labs/moshi `models/loaders.py`
    (`CheckpointInfo.from_hf_repo(...).get_mimi()`, `mimi.set_num_codebooks`,
    `mimi.encode(x[B,1,samples]) -> [B, n_q, frames]`) and moshi-finetune's
    InterleavedTokenizer, which encodes both channels as a batch of 2.
    Not run on this machine (no moshi, no GPU, Hugging Face blocked).
    """
    import torch
    from moshi.models import loaders  # type: ignore

    import sphn  # type: ignore  # moshi's audio IO helper

    global _MIMI
    if _MIMI is None:
        from moshi_compat import checkpoint
        info = checkpoint(mimi_cfg.get("hf_repo", "kyutai/moshiko-pytorch-bf16"), loaders)
        _MIMI = info.get_mimi(device=mimi_cfg.get("device", "cuda"))
        _MIMI.set_num_codebooks(N_CODEBOOKS_PER_CHANNEL)
        _MIMI.eval()
    wav, sr = sphn.read(wav_path)  # [channels, samples]
    assert sr == 24000 and wav.shape[0] == 2, (wav_path, sr, wav.shape)
    with torch.no_grad():
        x = torch.tensor(wav, device=_MIMI.device if hasattr(_MIMI, "device") else mimi_cfg.get("device", "cuda"))
        codes = _MIMI.encode(x[:, None])  # [2, 8, frames]
    codes = codes.reshape(2 * N_CODEBOOKS_PER_CHANNEL, -1).cpu().numpy()
    out = np.full((2 * N_CODEBOOKS_PER_CHANNEL, T), ZERO_TOKEN_ID, dtype=np.int64)
    n = min(T, codes.shape[1])
    out[:, :n] = codes[:, :n]
    return out


_MIMI = None


# --- episode -> examples ------------------------------------------------------

def episode_duration(ep: dict) -> float:
    end = 0.0
    for e in ep.get("events", []):
        end = max(end, float(e["t"]) + float(e.get("dur") or 0.0))
    return end


def choose_windows(T: int, max_frames: int, spans: List[dict]) -> List[Tuple[int, int]]:
    """Cut [0, T) into windows <= max_frames, never inside a span."""
    out, a = [], 0
    while a < T:
        b = min(T, a + max_frames)
        if b < T:
            for s in spans:
                if s["start_frame"] < b < s["end_frame"] and s["start_frame"] > a:
                    b = s["start_frame"]
        out.append((a, b))
        a = b
    return out


def prepare_episode(ep: dict, cfg: dict, tok: TextTokenizer, base_dir: str = ".") -> List[Dict[str, Any]]:
    pcfg = cfg.get("prepare", {})
    acfg = pcfg.get("audio", {"mode": "placeholder"})
    max_frames = int(pcfg.get("max_frames", 1250))
    tail = float(pcfg.get("tail_sec", 1.0))

    if pcfg.get("retime", True) and not ep.get("meta", {}).get("retimed"):
        if acfg.get("mode") == "mimi" and not acfg.get("audio_is_retimed", False):
            raise ValueError("mimi audio must be rendered from retimed episodes (prepare.py --write-retimed); "
                             "set prepare.audio.audio_is_retimed: true once it is, or prepare.retime: false")
        ep = retime_episode(ep, tok, call_after_utterance_window_s=float(pcfg.get("call_after_utterance_window_s", 0.6)))
    T = int(math.ceil((episode_duration(ep) + tail) * FRAME_RATE))
    segments, source = episode_segments(ep, prefer_dataset=pcfg.get("prefer_dataset_serializer", True))
    alignment, wav_path = None, None
    if acfg.get("mode") == "mimi":
        audio_dir = resolve_path(acfg["audio_dir"], base_dir)
        wav_path = os.path.join(audio_dir, f"{ep['id']}.wav")
        alignment = load_alignment(os.path.splitext(wav_path)[0] + ".json")
    items = build_items(segments, tok, alignment)
    # Place with room to spare, then trim: queued tokens are never dropped.
    total_tokens = sum(len(it["tokens"]) for it in items)
    placed = place_items(items, T + total_tokens + 2, frozenset(bad_event_indices(ep)))
    used = max([s["end_frame"] for s in placed["spans"]] + [int(np.nonzero(placed["text"] != PAD_ID)[0].max()) + 1
                                                           if (placed["text"] != PAD_ID).any() else 0])
    T = max(T, used + int(tail * FRAME_RATE))
    for k in ("text", "loss_mask", "neg_mask"):
        placed[k] = placed[k][:T]

    mode = acfg.get("mode", "placeholder")
    if mode == "placeholder":
        audio = placeholder_audio(ep, T, int(acfg.get("card", 2048)), int(ep.get("seed", 0)) * 7919 + 17)
        audio_mask = np.ones(T, dtype=np.uint8)
    elif mode == "zero":
        audio = np.full((2 * N_CODEBOOKS_PER_CHANNEL, T), ZERO_TOKEN_ID, dtype=np.int64)
        audio_mask = np.zeros(T, dtype=np.uint8)
    elif mode == "mimi":
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"{wav_path}: run the dataset's TTS stage or use audio.mode=zero")
        audio = mimi_audio(wav_path, T, acfg)
        audio_mask = (audio[0] != ZERO_TOKEN_ID).astype(np.uint8)
    else:
        raise ValueError(f"unknown audio mode {mode!r}")

    # Optional pinned system prompt (PersonaPlex style): forced text tokens at the
    # start of every window, loss-masked, no audio. UNVERIFIED that this matches
    # how PersonaPlex was trained; its server forces `<system> ... <system>` text
    # tokens (and a voice prompt) before the conversation.
    prompt_ids = tok.encode_plain(pcfg["prompt_text"]) if pcfg.get("prompt_text") else []
    P = len(prompt_ids) + (int(pcfg.get("prompt_pad_frames", 11)) if prompt_ids else 0)
    pre_text = np.array(prompt_ids + [PAD_ID] * (P - len(prompt_ids)), dtype=np.int64)
    zeros = np.zeros(P, dtype=np.uint8)
    pre_audio = np.full((2 * N_CODEBOOKS_PER_CHANNEL, P), ZERO_TOKEN_ID, dtype=np.int64)

    examples = []
    for w, (a, b) in enumerate(choose_windows(T, max_frames - P, placed["spans"])):
        spans = [dict(s, start_frame=s["start_frame"] - a + P, end_frame=s["end_frame"] - a + P,
                      nominal_frame=s["nominal_frame"] - a + P)
                 for s in placed["spans"] if a <= s["start_frame"] < b]
        examples.append({
            "id": ep["id"] if w == 0 and b == T else f"{ep['id']}#w{w}",
            "episode_id": ep["id"], "split": ep.get("split", "train"), "frame_offset": a - P,
            "prompt_frames": P,
            "text": np.concatenate([pre_text, placed["text"][a:b]]).astype(np.int32),
            "loss_mask": np.concatenate([zeros, placed["loss_mask"][a:b]]),
            "neg_mask": np.concatenate([zeros, placed["neg_mask"][a:b]]),
            "audio": np.concatenate([pre_audio, audio[:, a:b]], axis=1).astype(np.int32),
            "audio_mask": np.concatenate([zeros, audio_mask[a:b]]),
            "meta": {"frame_offset": a - P, "prompt_frames": P, "serializer": source, "audio_mode": mode, "spans": spans,
                     "dropped_tokens": placed["dropped_tokens"],
                     "max_word_delay_frames": placed["max_word_delay_frames"],
                     "max_span_delay_frames": placed["max_span_delay_frames"],
                     "negative": bool(ep.get("labels", {}).get("negative", False)),
                     "scenario": ep.get("scenario")},
        })
    return examples


def get_tokenizer(cfg: dict, base: str, eps: List[dict], rebuild: bool = False) -> TextTokenizer:
    """The chunk (dry-run) tokenizer's vocabulary is built here from the training
    split's Elle-side text, the way a SentencePiece model would be trained."""
    tcfg = cfg.get("tokenizer", {})
    if tcfg.get("kind") == "chunk":
        path = resolve_path(tcfg["vocab"], base)
        if rebuild or not os.path.exists(path):
            texts = [s["text"] for ep in eps if ep.get("split", "train") == "train"
                     for s in episode_segments(ep)[0]]
            ChunkTokenizer.build(texts, int(tcfg.get("max_pieces", 4000))).save(path)
    return build_tokenizer(tcfg, base)


def load_episodes(path: str) -> List[dict]:
    eps = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                eps.append(json.loads(line))
    return eps


def save_examples(examples: List[dict], out_dir: str) -> Dict[str, int]:
    os.makedirs(out_dir, exist_ok=True)
    counts: Dict[str, int] = {}
    with open(os.path.join(out_dir, "manifest.jsonl"), "w") as man:
        for ex in examples:
            sp = ex["split"]
            d = os.path.join(out_dir, sp)
            os.makedirs(d, exist_ok=True)
            fn = ex["id"].replace("#", "_") + ".npz"
            np.savez_compressed(os.path.join(d, fn), text=ex["text"], loss_mask=ex["loss_mask"], neg_mask=ex["neg_mask"],
                                audio=ex["audio"], audio_mask=ex["audio_mask"])
            man.write(json.dumps({"id": ex["id"], "episode_id": ex["episode_id"], "split": sp,
                                  "path": os.path.join(sp, fn), "frames": int(len(ex["text"])),
                                  "frame_offset": ex["frame_offset"],
                                  "prompt_frames": ex["prompt_frames"], **ex["meta"]}) + "\n")
            counts[sp] = counts.get(sp, 0) + 1
    return counts


def load_examples(out_dir: str, split: Optional[str] = None) -> List[dict]:
    out = []
    with open(os.path.join(out_dir, "manifest.jsonl")) as f:
        for line in f:
            m = json.loads(line)
            if split is not None and m["split"] != split:
                continue
            z = np.load(os.path.join(out_dir, m["path"]))
            out.append({"id": m["id"], "episode_id": m["episode_id"], "split": m["split"],
                        "text": z["text"], "loss_mask": z["loss_mask"], "neg_mask": z["neg_mask"], "audio": z["audio"],
                        "audio_mask": z["audio_mask"], "meta": m})
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--episodes", help="override data.episodes")
    ap.add_argument("--out", help="override data.prepared_dir")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--write-retimed", help="also write the retimed episodes here (input for TTS rendering)")
    ap.add_argument("--rebuild-vocab", action="store_true", help="chunk tokenizer: rebuild the vocab from these episodes")
    args = ap.parse_args(argv)
    cfg, base = load_config(args.config)
    eps_path = resolve_path(args.episodes or cfg["data"]["episodes"], base)
    out_dir = resolve_path(args.out or cfg["data"]["prepared_dir"], base)
    random.seed(cfg.get("seed", 0))
    eps = load_episodes(eps_path)
    if args.limit:
        eps = eps[: args.limit]
    tok = get_tokenizer(cfg, base, eps, rebuild=args.rebuild_vocab)
    pcfg = cfg.get("prepare", {})
    if pcfg.get("retime", True):
        eps = [ep if ep.get("meta", {}).get("retimed") else
               retime_episode(ep, tok, call_after_utterance_window_s=float(pcfg.get("call_after_utterance_window_s", 0.6)))
               for ep in eps]
    os.makedirs(out_dir, exist_ok=True)
    # The episodes as trained on (retimed); evaluate.py scores against these and
    # a TTS renderer must render audio from them.
    for path in [os.path.join(out_dir, "episodes.jsonl")] + ([args.write_retimed] if args.write_retimed else []):
        with open(path, "w") as f:
            for ep in eps:
                f.write(json.dumps(ep, ensure_ascii=False) + "\n")
    examples = []
    sources: Dict[str, int] = {}
    for ep in eps:
        for ex in prepare_episode(ep, cfg, tok, base):
            examples.append(ex)
            sources[ex["meta"]["serializer"]] = sources.get(ex["meta"]["serializer"], 0) + 1
    counts = save_examples(examples, out_dir)
    frames = sum(len(e["text"]) for e in examples)
    summary = {"episodes": len(eps), "examples": counts, "frames": frames, "hours": frames / FRAME_RATE / 3600,
               "serializer_sources": sources,
               "max_span_delay_frames": max((e["meta"]["max_span_delay_frames"] for e in examples), default=0),
               "max_word_delay_frames": max((e["meta"]["max_word_delay_frames"] for e in examples), default=0),
               "dropped_tokens": sum(e["meta"]["dropped_tokens"] for e in examples), "out": out_dir}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
