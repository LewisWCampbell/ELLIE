"""TTS backends and the voice plan for render_audio.py.

A backend turns (text, voice, lang, speed) into a `Clip`: mono 16-bit PCM at
24 kHz plus optional word timings. Backends:

  kokoro    hexgrad/Kokoro-82M (Apache-2.0), run locally (GPU if CUDA is available).
            The API used here was read from the installed `kokoro` 0.9.4 source:
            KPipeline(lang_code, repo_id, model=KModel), pipeline(text, voice=, speed=,
            split_pattern=None) yields Results with .audio (float tensor, 24 kHz) and,
            for English ('a'/'b') only, .tokens whose MToken.start_ts/end_ts are
            seconds within that Result's audio. Spanish ('e') goes through misaki's
            EspeakG2P and yields no tokens, so its word timings are proportional.
            UNVERIFIED here (no weights: Hugging Face is blocked in this sandbox):
            that the voice names below exist, the audio/timestamp values, speed.
            `render_audio.py check-kokoro` verifies all of it on the VM.
  deepgram  Deepgram Aura over HTTPS (needs DEEPGRAM_API_KEY; never run here).
  dry       Offline tone, length from word count. Pipeline smoke tests.
  fake      Offline, deterministic, deliberately variable clip lengths (0.6x to 1.5x
            the 2.6 words/s estimate) with word timings. For tests.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import urllib.parse
import urllib.request
import zlib
from array import array
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

SAMPLE_RATE = 24_000
Word = Tuple[str, float, float]  # word, start s, end s (relative to clip start)


class MissingCredentials(RuntimeError):
    pass


@dataclass
class Clip:
    pcm: bytes                         # mono int16 little-endian, 24 kHz
    words: Optional[List[Word]] = None  # None -> proportional timings are derived
    timing: str = "proportional"        # "kokoro" | "fake" | "proportional"

    @property
    def samples(self) -> int:
        return len(self.pcm) // 2


def proportional_words(text: str, dur: float) -> List[Word]:
    """Spread words over [0, dur] by (len + 1), the same rule as the trainer's spread_words."""
    words = text.split()
    if not words:
        return []
    weights = [len(w) + 1 for w in words]
    total = float(sum(weights))
    out, cur = [], 0.0
    for w, wt in zip(words, weights):
        d = dur * wt / total
        out.append((w, round(cur, 4), round(cur + d, 4)))
        cur += d
    return out


def float_to_pcm16(samples) -> bytes:
    """Float audio in [-1, 1] (list, numpy array or torch tensor) -> int16 LE bytes."""
    if hasattr(samples, "detach"):
        samples = samples.detach().cpu().numpy()
    try:
        import numpy as np  # noqa: WPS433
        a = np.asarray(samples, dtype=np.float32).reshape(-1)
        return (np.clip(a, -1.0, 1.0) * 32767.0).round().astype("<i2").tobytes()
    except ImportError:  # pragma: no cover
        return array("h", (int(round(max(-1.0, min(1.0, float(x))) * 32767)) for x in samples)).tobytes()


class TTSBackend:
    name = "base"
    model_id = ""
    elle_voices: Dict[str, str] = {}
    caller_pools: Dict[str, List[str]] = {}

    def elle_voice(self, lang: str) -> str:
        return self.elle_voices[lang]

    def caller_pool(self, lang: str) -> List[str]:
        return self.caller_pools[lang]

    def synthesize(self, text: str, voice: str, lang: str, speed: float = 1.0) -> Clip:
        raise NotImplementedError

    def spec(self) -> Dict[str, Any]:
        """Enough to rebuild this backend in a worker process."""
        return {"name": self.name}


# --- offline backends -------------------------------------------------------------

class DryBackend(TTSBackend):
    """A quiet tone per utterance, sized by the 2.6 words/s estimate."""
    name = "dry"
    model_id = "tone-v2"
    elle_voices = {"en": "elle", "es": "elle"}
    caller_pools = {"en": ["caller-a", "caller-b"], "es": ["caller-a", "caller-b"]}

    def synthesize(self, text: str, voice: str, lang: str, speed: float = 1.0) -> Clip:
        dur = (max(0.35, len(text.split()) / 2.6) + 0.15) / max(0.5, speed)
        n = int(SAMPLE_RATE * dur)
        freq = 220.0 if voice == "elle" else 160.0
        a = array("h", (int(1200 * math.sin(2 * math.pi * freq * k / SAMPLE_RATE)) for k in range(n)))
        return Clip(a.tobytes(), None, "proportional")


class FakeBackend(TTSBackend):
    """Deterministic, variable-length clips with word timings, for tests.

    Length = words / 2.6 s, scaled by a factor in [lo, hi] hashed from
    (voice, text), divided by speed. Each sample's value encodes the clip
    (low bits of a hash) so tests can find where a clip landed in a mix."""
    name = "fake"
    model_id = "fake-v1"
    elle_voices = {"en": "fake_elle_en", "es": "fake_elle_es"}
    caller_pools = {"en": ["fake_f_us", "fake_m_us", "fake_f_uk", "fake_m_uk"],
                    "es": ["fake_m_es", "fake_f_es"]}

    def __init__(self, lo: float = 0.6, hi: float = 1.5, with_words: bool = True):
        self.lo, self.hi, self.with_words = lo, hi, with_words
        self.calls = 0

    @staticmethod
    def marker(voice: str, text: str) -> int:
        return 1 + zlib.crc32(f"{voice}|{text}".encode()) % 30000

    def synthesize(self, text: str, voice: str, lang: str, speed: float = 1.0) -> Clip:
        self.calls += 1
        u = zlib.crc32(f"len|{voice}|{text}".encode()) / 0xFFFFFFFF
        factor = self.lo + (self.hi - self.lo) * u
        dur = max(0.3, len(text.split()) / 2.6 * factor + 0.1) / max(0.5, speed)
        n = int(round(SAMPLE_RATE * dur))
        a = array("h", [self.marker(voice, text)]) * n
        words = proportional_words(text, n / SAMPLE_RATE) if self.with_words else None
        return Clip(a.tobytes(), words, "fake" if words else "proportional")

    def spec(self) -> Dict[str, Any]:
        return {"name": self.name, "lo": self.lo, "hi": self.hi, "with_words": self.with_words}


# --- Deepgram (kept from the first version; never run here) ------------------------

class DeepgramAuraBackend(TTSBackend):
    """Deepgram Aura text-to-speech over HTTPS (untested here; needs DEEPGRAM_API_KEY).
    Aura has no speed control, so `speed` is ignored."""
    name = "deepgram"
    model_id = "aura"
    URL = "https://api.deepgram.com/v1/speak"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("DEEPGRAM_API_KEY", "")
        if not self.api_key:
            raise MissingCredentials(
                "render_audio.py: the Deepgram backend needs DEEPGRAM_API_KEY in the environment. "
                "Set it, or use --backend kokoro (local, no key) or --backend dry. Nothing was rendered.")
        self.elle_voices = {"en": os.environ.get("ELLE_VOICE_EN", "aura-2-thalia-en"),
                            "es": os.environ.get("ELLE_VOICE_ES", "aura-2-celeste-es")}
        self.caller_pools = {
            "en": os.environ.get("CALLER_VOICES_EN", "aura-2-orion-en,aura-2-luna-en,aura-2-arcas-en").split(","),
            "es": os.environ.get("CALLER_VOICES_ES", "aura-2-nestor-es,aura-2-estrella-es").split(","),
        }

    def synthesize(self, text: str, voice: str, lang: str, speed: float = 1.0) -> Clip:
        q = urllib.parse.urlencode({"model": voice, "encoding": "linear16",
                                    "sample_rate": SAMPLE_RATE, "container": "none"})
        req = urllib.request.Request(f"{self.URL}?{q}", data=json.dumps({"text": text}).encode(),
                                     headers={"Authorization": f"Token {self.api_key}",
                                              "Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=60) as r:
            pcm = r.read()
        return Clip(pcm[: len(pcm) // 2 * 2], None, "proportional")


# --- Kokoro -----------------------------------------------------------------------------

KOKORO_REPO = "hexgrad/Kokoro-82M"

# Voice names from Kokoro-82M's VOICES.md (v1.0). UNVERIFIED in this sandbox;
# `check-kokoro` fails if any configured voice file is missing from the snapshot.
# Elle: bf_isabella (British English), chosen by ear from the check-kokoro
# samples. British voices get British G2P (lang_code 'b'). Spanish has one
# native female voice, ef_dora. Override with KOKORO_ELLE_EN / KOKORO_ELLE_ES.
KOKORO_ELLE = {"en": os.environ.get("KOKORO_ELLE_EN", "bf_isabella"),
               "es": os.environ.get("KOKORO_ELLE_ES", "ef_dora")}
# Callers: never Elle's voice; US and UK accents; both genders. Excludes
# bm_lewis (the owner's name) and the novelty *_santa voices.
KOKORO_CALLERS_EN = ["af_heart", "af_bella", "af_nicole", "af_sarah", "af_aoede", "af_kore", "af_jessica",
                     "am_michael", "am_fenrir", "am_puck", "am_eric", "am_liam", "am_onyx",
                     "bf_emma", "bf_alice", "bm_george", "bm_fable", "bm_daniel"]
# Spanish has only em_alex as a native non-Elle voice, so the pool adds Portuguese,
# Italian and French voices speaking Spanish phonemes (kokoro allows a voice from
# another language, with a warning): accented Spanish, plausible for callers.
# UNVERIFIED how natural this sounds; check-kokoro writes samples to listen to.
KOKORO_CALLERS_ES = ["em_alex", "pf_dora", "pm_alex", "if_sara", "im_nicola", "ff_siwis"]


def _env_list(name: str, default: List[str]) -> List[str]:
    v = os.environ.get(name, "")
    return [x.strip() for x in v.split(",") if x.strip()] or list(default)


class KokoroBackend(TTSBackend):
    name = "kokoro"

    def __init__(self, device: Optional[str] = None, repo_id: str = KOKORO_REPO,
                 pipeline_factory: Optional[Callable[..., Any]] = None,
                 model_factory: Optional[Callable[..., Any]] = None):
        self.repo_id = repo_id
        self.model_id = f"{repo_id}@kokoro"
        self.device = device
        self.elle_voices = dict(KOKORO_ELLE)
        self.caller_pools = {"en": _env_list("KOKORO_CALLERS_EN", KOKORO_CALLERS_EN),
                             "es": _env_list("KOKORO_CALLERS_ES", KOKORO_CALLERS_ES)}
        self._pipeline_factory = pipeline_factory
        self._model_factory = model_factory
        self._model = None
        self._pipelines: Dict[str, Any] = {}

    # Construction is lazy so a worker process builds its own model on its own device.
    def _ensure(self) -> None:
        if self._model is not None or self._pipeline_factory is not None and self._model_factory is None:
            return
        if self._model_factory is None:
            try:
                from kokoro import KModel  # type: ignore
                import torch  # type: ignore
            except ImportError as e:
                raise SystemExit("render_audio.py: the kokoro backend needs `pip install kokoro` "
                                 f"(and torch); import failed: {e}")
            dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.device = dev
            self._model = KModel(repo_id=self.repo_id).to(dev).eval()
        else:
            self._model = self._model_factory()

    @staticmethod
    def lang_code(voice: str, lang: str) -> str:
        """Kokoro G2P language: 'e' for Spanish text; for English, British G2P for
        British voices ('b' prefix), American otherwise."""
        if lang == "es":
            return "e"
        return "b" if voice[:1] == "b" else "a"

    def pipeline(self, code: str):
        if code not in self._pipelines:
            self._ensure()
            if self._pipeline_factory is not None:
                self._pipelines[code] = self._pipeline_factory(lang_code=code, repo_id=self.repo_id, model=self._model)
            else:
                from kokoro import KPipeline  # type: ignore
                self._pipelines[code] = KPipeline(lang_code=code, repo_id=self.repo_id, model=self._model)
        return self._pipelines[code]

    def synthesize(self, text: str, voice: str, lang: str, speed: float = 1.0) -> Clip:
        pipe = self.pipeline(self.lang_code(voice, lang))
        pcm = bytearray()
        tokens: List[Tuple[str, str, Optional[float], Optional[float]]] = []  # text, ws, start, end (abs s)
        have_ts = True
        for res in pipe(text, voice=voice, speed=speed, split_pattern=None):
            audio = getattr(res, "audio", None)
            if audio is None:
                continue
            offset = len(pcm) / 2 / SAMPLE_RATE
            chunk = float_to_pcm16(audio)
            toks = getattr(res, "tokens", None)
            if not toks:
                have_ts = False
            else:
                for t in toks:
                    s, e = getattr(t, "start_ts", None), getattr(t, "end_ts", None)
                    tokens.append((t.text, t.whitespace,
                                   None if s is None else offset + float(s),
                                   None if e is None else offset + float(e)))
            pcm += chunk
        if not pcm:
            raise RuntimeError(f"kokoro produced no audio for {text!r} ({voice}, {lang})")
        dur = len(pcm) / 2 / SAMPLE_RATE
        words = tokens_to_words(text, tokens, dur) if have_ts and tokens else None
        return Clip(bytes(pcm), words, "kokoro" if words else "proportional")

    def spec(self) -> Dict[str, Any]:
        return {"name": self.name, "device": self.device, "repo_id": self.repo_id}


def tokens_to_words(text: str, tokens: Sequence[Tuple[str, str, Optional[float], Optional[float]]],
                    dur: float) -> Optional[List[Word]]:
    """Merge Kokoro MTokens (punctuation is its own token; `whitespace` marks a word
    end) into the whitespace-separated words of `text`. Returns None (caller falls
    back to proportional timing) if the words don't line up or no token has a time."""
    words: List[Tuple[str, List[Optional[float]], List[Optional[float]]]] = []
    cur, starts, ends = "", [], []
    for tx, ws, s, e in tokens:
        cur += tx
        starts.append(s)
        ends.append(e)
        if ws:
            words.append((cur, starts, ends))
            cur, starts, ends = "", [], []
    if cur:
        words.append((cur, starts, ends))
    target = text.split()
    if [w for w, _, _ in words] != target:
        return None
    out: List[Word] = []
    timed = [(next((x for x in st if x is not None), None), next((x for x in reversed(en) if x is not None), None))
             for _, st, en in words]
    if all(a is None for a, _ in timed):
        return None
    prev_end = 0.0
    for k, (w, _, _) in enumerate(words):
        a, b = timed[k]
        if a is None:
            a = prev_end
        if b is None:
            nxt = next((timed[j][0] for j in range(k + 1, len(words)) if timed[j][0] is not None), dur)
            b = max(a, nxt)
        a, b = max(0.0, min(a, dur)), max(0.0, min(b, dur))
        out.append((w, round(a, 4), round(max(a, b), 4)))
        prev_end = max(a, b)
    return out


BACKENDS = {"kokoro": KokoroBackend, "deepgram": DeepgramAuraBackend, "dry": DryBackend, "fake": FakeBackend}


def make_backend(spec: Dict[str, Any]) -> TTSBackend:
    spec = dict(spec)
    name = spec.pop("name")
    if name not in BACKENDS:
        raise SystemExit(f"render_audio.py: unknown backend {name!r} (choose from {', '.join(BACKENDS)})")
    return BACKENDS[name](**spec)


# --- voice plan ------------------------------------------------------------------------------

ELLE_SPEED = float(os.environ.get("ELLE_SPEED", "1.0"))


def _u(s: str) -> float:
    return zlib.crc32(s.encode()) / 0xFFFFFFFF


def episode_key(ep: Dict[str, Any]) -> str:
    """Stable across id renumbering and splits: (scenario, seed) is unique per dataset."""
    return f"{ep.get('scenario')}:{ep.get('seed')}"


def event_voice(ep: Dict[str, Any], ev: Dict[str, Any], backend: TTSBackend) -> Tuple[str, str, float]:
    """(voice, lang, speed) for one speech event. Elle: a fixed voice per language.
    Caller: one voice per episode from the pool (Spanish pool when the episode has
    any Spanish), a per-episode base speed in [0.94, 1.06], and +-0.02 per line."""
    lang = "es" if str(ev.get("lang", "en")).startswith("es") else "en"
    if ev["ch"] == "elle":
        return backend.elle_voice(lang), lang, ELLE_SPEED
    ep_lang = (ep.get("labels") or {}).get("lang", "en")
    pool = backend.caller_pool("es" if ep_lang != "en" else "en")
    k = episode_key(ep)
    voice = pool[zlib.crc32(f"voice|{k}".encode()) % len(pool)]
    base = 0.94 + 0.12 * _u(f"speed|{k}")
    jitter = 0.04 * (_u(f"jit|{k}|{ev['text']}") - 0.5)
    return voice, lang, round(min(1.12, max(0.88, base + jitter)), 3)


def clip_key(backend: TTSBackend, voice: str, lang: str, speed: float, text: str) -> str:
    """Content address of a clip: same backend, voice, language, speed and text ->
    same file, so repeated lines ("Yes.") are synthesized once."""
    h = hashlib.sha1(json.dumps([backend.name, backend.model_id, voice, lang, round(speed, 3), text],
                                ensure_ascii=False).encode()).hexdigest()
    return f"{h[:2]}/{h}"
