"""Text-stream vocabulary: control ids, the four tool markers, and tokenizers.

Layout (mirrors Moshi's LMModel, read from kyutai-labs/moshi `moshi/models/lm.py`
on 2026-11-07; VERIFIED by reading source, not by running it):

  * text ids 0..text_card-1 are real tokens; Moshi's SentencePiece model
    `tokenizer_spm_32k_3.model` has text_card = 32000.
  * id 0 = end-of-text-padding (EPAD), 3 = text padding (PAD)
    (`existing_text_end_padding_id=0`, `existing_text_padding_id=3`), 1/2 are
    the SentencePiece BOS/EOS. PersonaPlex's server prints
    `['EPAD','BOS','EOS','PAD']` for ids 0..3, i.e. the same layout.
  * `text_emb` has text_card + 1 rows; row `text_card` is the *initial* token
    fed at t=0 (`text_initial_token_id == text_card`).
  * zero_token_id = -1 means "no input, no loss" for that position.

Our extension: the four contract markers are appended as NEW ids
text_card_base .. text_card_base+3, so text_card becomes text_card_base + 4 and
the initial-token row moves to the end (see tiny_model.resize_text_vocab and
moshi_backend.resize_moshi_text_vocab). Appending (rather than reusing rare
SentencePiece pieces) keeps every pretrained row untouched.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence

FRAME_RATE = 12.5  # Mimi frames per second (VERIFIED: moshi-finetune README, Mimi docs)

EPAD_ID = 0
BOS_ID = 1
EOS_ID = 2
PAD_ID = 3
ZERO_TOKEN_ID = -1  # "no input, no loss", as in Moshi

TOOL_OPEN = "<tool>"
TOOL_CLOSE = "</tool>"
RESULT_OPEN = "<result id="
RESULT_CLOSE = "</result>"
SPECIAL_TOKENS: List[str] = [TOOL_OPEN, TOOL_CLOSE, RESULT_OPEN, RESULT_CLOSE]

# Longest marker first, so "</tool>" never splits as "<" + "/tool>" etc.
_SPECIAL_RE = re.compile("(" + "|".join(re.escape(s) for s in sorted(SPECIAL_TOKENS, key=len, reverse=True)) + ")")


class TextTokenizer:
    """Interface. `base_card` is the vocab size before the markers were added."""

    base_card: int

    @property
    def card(self) -> int:  # text_card after extension
        return self.base_card + len(SPECIAL_TOKENS)

    @property
    def special_ids(self) -> dict:
        return {s: self.base_card + i for i, s in enumerate(SPECIAL_TOKENS)}

    def id_of(self, marker: str) -> int:
        return self.special_ids[marker]

    def is_special(self, tok: int) -> bool:
        return self.base_card <= tok < self.card

    def is_control(self, tok: int) -> bool:
        return tok in (EPAD_ID, BOS_ID, EOS_ID, PAD_ID) or tok < 0

    # --- to implement -------------------------------------------------------
    def encode_plain(self, text: str) -> List[int]:
        raise NotImplementedError

    def decode_plain(self, ids: Sequence[int]) -> str:
        raise NotImplementedError

    def encode_word(self, word: str, first: bool) -> List[int]:
        """Tokenize one aligned word (Moshi tokenizes words individually)."""
        return self.encode_plain(word if first else " " + word)

    # --- shared -------------------------------------------------------------
    def encode(self, text: str) -> List[int]:
        """Encode text that may contain markers; markers become single ids and are
        never produced by the plain tokenizer."""
        out: List[int] = []
        for part in _SPECIAL_RE.split(text):
            if not part:
                continue
            if part in self.special_ids:
                out.append(self.special_ids[part])
            else:
                out.extend(self.encode_plain(part))
        return out

    def decode(self, ids: Iterable[int]) -> str:
        out: List[str] = []
        buf: List[int] = []
        inv = {v: k for k, v in self.special_ids.items()}
        for t in ids:
            if t in inv:
                if buf:
                    out.append(self.decode_plain(buf))
                    buf = []
                out.append(inv[t])
            elif self.is_control(t):
                continue
            else:
                buf.append(t)
        if buf:
            out.append(self.decode_plain(buf))
        return "".join(out)


class ByteTokenizer(TextTokenizer):
    """Dry-run tokenizer: ids 0..3 are control (same as Moshi), 4..259 are bytes.
    Not the real tokenizer; it only exists so CPU tests exercise the same code
    paths (control ids, appended specials, initial-token row)."""

    OFFSET = 4

    def __init__(self) -> None:
        self.base_card = self.OFFSET + 256

    def encode_plain(self, text: str) -> List[int]:
        return [b + self.OFFSET for b in text.encode("utf-8")]

    def decode_plain(self, ids: Sequence[int]) -> str:
        bs = bytes(i - self.OFFSET for i in ids if self.OFFSET <= i < self.OFFSET + 256)
        return bs.decode("utf-8", errors="replace")


_PIECE_RE = re.compile(r"\s?[A-Za-z\u00C0-\u024F]+|\s?\d+|\s?[^\sA-Za-z\d]|\s+")


class ChunkTokenizer(TextTokenizer):
    """Dry-run tokenizer with roughly SentencePiece's density (about one id per
    word): ids 0..3 control, 4..259 byte fallback, then a word-piece vocabulary
    built from the training corpus (`build`). Fully invertible, so tool-call
    JSON decodes exactly. Still not the real tokenizer."""

    OFFSET = 4

    def __init__(self, pieces: Sequence[str]):
        self.pieces = list(pieces)
        self.piece_id = {p: self.OFFSET + 256 + i for i, p in enumerate(self.pieces)}
        self.base_card = self.OFFSET + 256 + len(self.pieces)

    @classmethod
    def build(cls, texts: Iterable[str], max_pieces: int = 4000, min_count: int = 1) -> "ChunkTokenizer":
        counts: dict = {}
        for t in texts:
            for p in _PIECE_RE.findall(t):
                if len(p.encode("utf-8")) > 1:
                    counts[p] = counts.get(p, 0) + 1
        ranked = sorted((p for p, c in counts.items() if c >= min_count), key=lambda p: (-counts[p], p))
        return cls(ranked[:max_pieces])

    def save(self, path: str) -> None:
        import json
        import os

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"kind": "chunk", "pieces": self.pieces}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "ChunkTokenizer":
        import json

        with open(path) as f:
            return cls(json.load(f)["pieces"])

    def encode_plain(self, text: str) -> List[int]:
        out: List[int] = []
        for p in _PIECE_RE.findall(text):
            if p in self.piece_id:
                out.append(self.piece_id[p])
            else:
                out.extend(b + self.OFFSET for b in p.encode("utf-8"))
        return out

    def decode_plain(self, ids: Sequence[int]) -> str:
        inv = {v: k for k, v in self.piece_id.items()}
        buf = bytearray()
        for i in ids:
            if i in inv:
                buf.extend(inv[i].encode("utf-8"))
            elif self.OFFSET <= i < self.OFFSET + 256:
                buf.append(i - self.OFFSET)
        return buf.decode("utf-8", errors="replace")


class SentencePieceTokenizer(TextTokenizer):
    """Real tokenizer: Moshi/PersonaPlex `tokenizer_spm_32k_3.model`.

    UNVERIFIED in this repo (sentencepiece is not installed on the CPU box and the
    model file is behind Hugging Face, which the proxy blocks). The calls used
    (`encode`, `decode`, `get_piece_size`) are the standard sentencepiece API.
    """

    def __init__(self, model_path: str) -> None:
        import sentencepiece  # noqa: WPS433 (optional dependency)

        self.sp = sentencepiece.SentencePieceProcessor(model_file=model_path)
        self.base_card = int(self.sp.get_piece_size())

    def encode_plain(self, text: str) -> List[int]:
        return list(self.sp.encode(text))

    def encode_word(self, word: str, first: bool) -> List[int]:
        # SentencePiece marks word starts itself ("▁"), as Moshi's interleaver does.
        return list(self.sp.encode(word.strip()))

    def decode_plain(self, ids: Sequence[int]) -> str:
        return self.sp.decode(list(ids))


def build_tokenizer(cfg: dict, base_dir: str = ".") -> TextTokenizer:
    """cfg: {kind: byte} | {kind: chunk, vocab: path} | {kind: sentencepiece, path: ...}."""
    import os

    cfg = cfg or {}
    kind = cfg.get("kind", "byte")
    if kind == "byte":
        return ByteTokenizer()
    if kind == "chunk":
        p = cfg["vocab"]
        p = p if os.path.isabs(p) else os.path.normpath(os.path.join(base_dir, p))
        return ChunkTokenizer.load(p)
    if kind == "sentencepiece":
        path = cfg.get("path")
        if not path:
            # The checkpoint's own tokenizer (tokenizer_spm_32k_3.model). UNVERIFIED:
            # moshi's CheckpointInfo.from_hf_repo(...).tokenizer, read from loaders.py.
            from moshi.models import loaders  # type: ignore

            path = str(loaders.CheckpointInfo.from_hf_repo(cfg["hf_repo"]).tokenizer)
        return SentencePieceTokenizer(path)
    raise ValueError(f"unknown tokenizer kind {kind!r}")


def frames_to_seconds(f: float) -> float:
    return f / FRAME_RATE


def seconds_to_frame(t: float) -> int:
    """Frame index whose interval [f/12.5, (f+1)/12.5) contains t. Matches the
    Moshi interleaver's placement rule (`start * frame_rate < t + 1`)."""
    import math

    return max(0, int(math.floor(t * FRAME_RATE + 1e-9)))
