#!/usr/bin/env python3
"""Optional audio rendering stage (training/CONTRACT.md, "Audio rendering").

    python3 training/data/render_audio.py episodes.jsonl out_dir [--backend deepgram|dry] [--limit N]

Writes, per episode:
    <id>.wav   stereo, 24 kHz, 16-bit PCM; channel 0 = Elle, channel 1 = caller (Moshi convention)
    <id>.json  {"alignments": [[word, [start, end], "SPEAKER_MAIN"], ...],   Elle's words
                "tool_spans": [{"t", "text", "forced"}, ...],                 serialize.py markers
                "episode_id", "sample_rate"}

Backends implement `TTSBackend.synthesize(text, speaker, lang) -> bytes` (mono
int16 little-endian PCM at 24 kHz):

  - `deepgram`: Deepgram Aura, credentials from DEEPGRAM_API_KEY. Not exercised by
    the tests or by this repo's sample generation; it exits with a clear message
    when no key is set. Voice ids are configurable (ELLE_VOICE_EN, ELLE_VOICE_ES,
    CALLER_VOICES_EN, CALLER_VOICES_ES) because the right voice list should be
    checked against Deepgram's current catalogue before a real run.
  - `dry`: an offline tone generator whose length matches each utterance's `dur`,
    for testing the mixing, channel layout and alignment output with no network.

Known gap: TTS audio will not match the generator's estimated `dur`. This stub
places each clip at its `t` and truncates interrupted lines at `dur`; a real run
should re-time the episode from the rendered durations (shift later events and
recompute tool `t`s relative to speech) before writing alignments. See README.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.parse
import urllib.request
import wave
import zlib
from array import array
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from serialize import serialize_episode  # noqa: E402

SAMPLE_RATE = 24_000
ELLE_CH, CALLER_CH = 0, 1


class MissingCredentials(RuntimeError):
    pass


class TTSBackend:
    name = "base"

    def synthesize(self, text: str, speaker: str, lang: str) -> bytes:
        """Mono int16 LE PCM at 24 kHz for `text`. `speaker` is "elle" or a caller voice key."""
        raise NotImplementedError


class DryBackend(TTSBackend):
    """Offline stand-in: a quiet tone per word, sized by the 2.6 words/s estimate."""
    name = "dry"

    def synthesize(self, text: str, speaker: str, lang: str, dur: Optional[float] = None) -> bytes:
        n = int(SAMPLE_RATE * (dur if dur is not None else max(0.35, len(text.split()) / 2.6)))
        freq = 220.0 if speaker == "elle" else 160.0
        a = array("h", (int(1200 * math.sin(2 * math.pi * freq * k / SAMPLE_RATE)) for k in range(n)))
        return a.tobytes()


class DeepgramAuraBackend(TTSBackend):
    """Deepgram Aura text-to-speech over HTTPS (untested here; needs DEEPGRAM_API_KEY)."""
    name = "deepgram"
    URL = "https://api.deepgram.com/v1/speak"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("DEEPGRAM_API_KEY", "")
        if not self.api_key:
            raise MissingCredentials(
                "render_audio.py: the Deepgram backend needs DEEPGRAM_API_KEY in the environment. "
                "Set it, or run with --backend dry to test the pipeline offline. Nothing was rendered.")
        self.voices = {
            "elle:en": os.environ.get("ELLE_VOICE_EN", "aura-2-thalia-en"),
            "elle:es": os.environ.get("ELLE_VOICE_ES", "aura-2-celeste-es"),
        }
        self.caller_voices = {
            "en": os.environ.get("CALLER_VOICES_EN", "aura-2-orion-en,aura-2-luna-en,aura-2-arcas-en").split(","),
            "es": os.environ.get("CALLER_VOICES_ES", "aura-2-nestor-es,aura-2-estrella-es").split(","),
        }

    def voice(self, speaker: str, lang: str) -> str:
        lang = "es" if lang.startswith("es") else "en"
        if speaker == "elle":
            return self.voices[f"elle:{lang}"]
        pool = self.caller_voices[lang]
        return pool[zlib.crc32(speaker.encode()) % len(pool)]  # stable across runs

    def synthesize(self, text: str, speaker: str, lang: str, dur: Optional[float] = None) -> bytes:
        q = urllib.parse.urlencode({"model": self.voice(speaker, lang), "encoding": "linear16",
                                    "sample_rate": SAMPLE_RATE, "container": "none"})
        req = urllib.request.Request(f"{self.URL}?{q}", data=json.dumps({"text": text}).encode(),
                                     headers={"Authorization": f"Token {self.api_key}",
                                              "Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()


BACKENDS = {"dry": DryBackend, "deepgram": DeepgramAuraBackend}


def make_backend(name: str) -> TTSBackend:
    if name not in BACKENDS:
        raise SystemExit(f"render_audio.py: unknown backend {name!r} (choose from {', '.join(BACKENDS)})")
    return BACKENDS[name]()


def render_episode(ep: Dict, backend: TTSBackend, out_dir: str) -> Dict[str, str]:
    """Render one episode; returns {"wav": path, "json": path}."""
    events = ep["events"]
    total = max(e["t"] + e.get("dur", 0.0) for e in events) + 0.5
    n = int(total * SAMPLE_RATE)
    chans = [array("h", bytes(2 * n)), array("h", bytes(2 * n))]
    caller_voice = f"caller-{ep.get('seed', 0)}"
    for ev in events:
        if ev["type"] != "speech":
            continue
        speaker = "elle" if ev["ch"] == "elle" else caller_voice
        lang = ev.get("lang", "en")
        if isinstance(backend, DryBackend):
            pcm = backend.synthesize(ev["text"], speaker, lang, dur=ev["dur"])
        else:
            pcm = backend.synthesize(ev["text"], speaker, lang)
        clip = array("h")
        clip.frombytes(pcm[: len(pcm) // 2 * 2])
        if ev.get("interrupted"):
            clip = clip[: int(ev["dur"] * SAMPLE_RATE)]
        start = int(ev["t"] * SAMPLE_RATE)
        ch = chans[ELLE_CH if ev["ch"] == "elle" else CALLER_CH]
        if start + len(clip) > len(ch):
            ch.extend([0] * (start + len(clip) - len(ch)))
        for k, s in enumerate(clip):
            v = ch[start + k] + s
            ch[start + k] = max(-32768, min(32767, v))
    m = max(len(chans[0]), len(chans[1]))
    for c in chans:
        c.extend([0] * (m - len(c)))
    inter = array("h", bytes(4 * m))
    inter[0::2] = chans[ELLE_CH]
    inter[1::2] = chans[CALLER_CH]
    os.makedirs(out_dir, exist_ok=True)
    wav_path = os.path.join(out_dir, f"{ep['id']}.wav")
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(inter.tobytes())
    words = serialize_episode(ep, word_level=True)
    align = {
        "episode_id": ep["id"],
        "sample_rate": SAMPLE_RATE,
        "alignments": [[s.text, [s.t, s.end], "SPEAKER_MAIN"] for s in words if s.kind == "speech"],
        "tool_spans": [{"t": s.t, "text": s.text, "forced": s.is_forced} for s in words
                       if s.kind in ("tool_call", "tool_result")],
    }
    json_path = os.path.join(out_dir, f"{ep['id']}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(align, f, ensure_ascii=False)
    return {"wav": wav_path, "json": json_path}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episodes")
    ap.add_argument("out_dir")
    ap.add_argument("--backend", default="deepgram", choices=sorted(BACKENDS))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    try:
        backend = make_backend(a.backend)
    except MissingCredentials as e:
        print(e, file=sys.stderr)
        return 2
    n = 0
    with open(a.episodes, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if a.limit and n >= a.limit:
                break
            paths = render_episode(json.loads(line), backend, a.out_dir)
            n += 1
            print(paths["wav"])
    print(f"rendered {n} episode(s) with backend {backend.name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
