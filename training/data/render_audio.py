#!/usr/bin/env python3
"""Audio rendering stage (training/CONTRACT.md: "Audio rendering", "Spans take time").

Audio must be placed at the trainer's *retimed* event times, but retiming needs
the real speech durations. So rendering is two passes around the trainer:

  1. synth     episodes.jsonl -> clip cache + measured.jsonl
               Synthesizes every speech event once (content-addressed clips,
               resumable), then writes the episodes with each speech `dur`
               replaced by its clip's real length and later events re-placed so
               gaps, overlaps, barge-ins and tool latencies are preserved
               (audio_timing.py). Every episode is re-validated; any that fails
               is dropped and reported.
  2. (trainer) python3 training/model/prepare.py ... --episodes measured.jsonl
               --write-retimed retimed.jsonl
  3. assemble  retimed.jsonl + clip cache -> <id>.wav + <id>.json
               Places the cached clips (no re-synthesis) at the retimed times:
               stereo 24 kHz 16-bit, channel 0 Elle, channel 1 caller. Fails
               loudly if a clip is missing or a duration no longer matches its clip.

    python3 render_audio.py synth    full.jsonl --cache CLIPS --out measured.jsonl [--backend kokoro] [--workers 4]
    python3 render_audio.py assemble retimed.jsonl --cache CLIPS --out AUDIO_DIR [--delete-clips --chunk 500]
    python3 render_audio.py check-kokoro --out /tmp/kokoro_check     # one-command verification on the VM
    python3 render_audio.py render   episodes.jsonl OUT_DIR --backend dry   # quick preview, NOT for training

Alignment JSON (the trainer's format, training/model/prepare.py load_alignment):
    {"alignments": [[word, [start, end], "SPEAKER_MAIN"], ...],      Elle's words
     "spans": [{"kind": "tool"|"result", "t", "text", "id"}, ...],
     "episode_id", "sample_rate", "word_timing": "kokoro"|"proportional"|"mixed"|...}
Word times come from Kokoro's token timestamps where it gives them (English),
otherwise proportional to word length within the clip (Spanish, Deepgram, dry).

Backends: kokoro (default, local, no key), deepgram (DEEPGRAM_API_KEY), dry, fake.
See tts_backends.py for voices and what is UNVERIFIED.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import wave
import zlib
from array import array
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio_timing import apply_measured  # noqa: E402
from serialize import tool_call_text, tool_result_text  # noqa: E402
from tts_backends import (BACKENDS, KOKORO_REPO, SAMPLE_RATE, Clip, DeepgramAuraBackend, DryBackend,  # noqa: E402,F401
                          FakeBackend, KokoroBackend, MissingCredentials, TTSBackend, clip_key, event_voice,
                          make_backend, proportional_words)
from validate import validate_episode  # noqa: E402

DUR_TOLERANCE_S = 0.0015  # a retimed `dur` must equal its clip length to the millisecond
ELLE_CH, CALLER_CH = 0, 1


class AssembleError(RuntimeError):
    pass


# --- io helpers ----------------------------------------------------------------------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def clip_paths(cache: str, key: str) -> Tuple[str, str]:
    base = os.path.join(cache, key)
    return base + ".wav", base + ".json"


def clip_done(cache: str, key: str) -> bool:
    w, j = clip_paths(cache, key)
    return os.path.exists(w) and os.path.exists(j)


def write_clip(cache: str, key: str, clip: Clip, meta: Dict[str, Any]) -> Dict[str, Any]:
    """Atomic: the .json sidecar is written last and marks the clip complete."""
    w, j = clip_paths(cache, key)
    os.makedirs(os.path.dirname(w), exist_ok=True)
    tmp = w + f".tmp{os.getpid()}"
    with wave.open(tmp, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(clip.pcm)
    os.replace(tmp, w)
    side = dict(meta, samples=clip.samples, sample_rate=SAMPLE_RATE, timing=clip.timing,
                words=clip.words if clip.words is not None else
                proportional_words(meta["text"], clip.samples / SAMPLE_RATE))
    tmp = j + f".tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(side, f, ensure_ascii=False)
    os.replace(tmp, j)
    return side


def read_sidecar(cache: str, key: str) -> Dict[str, Any]:
    with open(clip_paths(cache, key)[1], encoding="utf-8") as f:
        return json.load(f)


def read_clip_pcm(cache: str, key: str) -> array:
    w, _ = clip_paths(cache, key)
    with wave.open(w, "rb") as f:
        if f.getframerate() != SAMPLE_RATE or f.getnchannels() != 1 or f.getsampwidth() != 2:
            raise AssembleError(f"{w}: expected mono 16-bit {SAMPLE_RATE} Hz")
        a = array("h")
        a.frombytes(f.readframes(f.getnframes()))
    return a


def parse_shard(s: Optional[str]) -> Optional[Tuple[int, int]]:
    if not s:
        return None
    i, n = (int(x) for x in s.split("/"))
    if not 0 <= i < n:
        raise SystemExit(f"--shard {s}: need 0 <= i < n")
    return i, n


def in_shard(key: str, shard: Optional[Tuple[int, int]]) -> bool:
    return shard is None or zlib.crc32(key.encode()) % shard[1] == shard[0]


class Progress:
    def __init__(self, label: str, total: int, every_s: float = 10.0):
        self.label, self.total, self.every = label, total, every_s
        self.t0 = self.last = time.time()
        self.done = 0
        self.audio_s = 0.0

    def tick(self, n: int = 1, audio_s: float = 0.0, force: bool = False) -> None:
        self.done += n
        self.audio_s += audio_s
        now = time.time()
        if (force and now != self.last) or now - self.last >= self.every or (n and self.done == self.total):
            self.last = now
            wall = max(1e-6, now - self.t0)
            rate = self.done / wall
            eta = (self.total - self.done) / rate if rate else float("inf")
            rt = f", {self.audio_s / 3600:.2f} h audio, {self.audio_s / wall:.1f}x real time" if self.audio_s else ""
            print(f"[{self.label}] {self.done}/{self.total}{rt}, {rate:.1f}/s, ETA {eta / 60:.1f} min",
                  file=sys.stderr, flush=True)


# --- synth ---------------------------------------------------------------------------------

def plan_episode(ep: Dict[str, Any], backend: TTSBackend) -> Dict[int, Dict[str, Any]]:
    """Speech event index -> {key, text, voice, lang, speed}."""
    plan = {}
    for i, ev in enumerate(ep["events"]):
        if ev["type"] != "speech":
            continue
        voice, lang, speed = event_voice(ep, ev, backend)
        # An interrupted line is synthesized as the words actually said (`text`).
        plan[i] = {"key": clip_key(backend, voice, lang, speed, ev["text"]), "text": ev["text"],
                   "voice": voice, "lang": lang, "speed": speed}
    return plan


_WORKER_BACKEND: Optional[TTSBackend] = None
_WORKER_CACHE = ""


def _worker_init(spec: Dict[str, Any], cache: str) -> None:
    global _WORKER_BACKEND, _WORKER_CACHE
    _WORKER_BACKEND = make_backend(spec)
    _WORKER_CACHE = cache


def _synth_job(job: Dict[str, Any], backend: Optional[TTSBackend] = None, cache: Optional[str] = None) -> Tuple[str, float]:
    backend = backend or _WORKER_BACKEND
    cache = cache or _WORKER_CACHE
    if clip_done(cache, job["key"]):
        return job["key"], 0.0
    clip = backend.synthesize(job["text"], job["voice"], job["lang"], job["speed"])
    meta = {k: job[k] for k in ("text", "voice", "lang", "speed")}
    meta.update(backend=backend.name, model_id=backend.model_id)
    write_clip(cache, job["key"], clip, meta)
    return job["key"], clip.samples / SAMPLE_RATE


def synthesize_jobs(jobs: List[Dict[str, Any]], backend: TTSBackend, cache: str, workers: int = 1,
                    progress_every: float = 10.0) -> float:
    """Synthesize missing clips. Returns seconds of audio produced."""
    prog = Progress("synth", len(jobs), progress_every)
    if not jobs:
        return 0.0
    if workers <= 1:
        for job in jobs:
            _, a = _synth_job(job, backend, cache)
            prog.tick(1, a)
    else:
        import multiprocessing as mp
        # spawn: each worker loads its own model (CUDA cannot be forked).
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_worker_init, initargs=(backend.spec(), cache)) as pool:
            for _, a in pool.imap_unordered(_synth_job, jobs, chunksize=8):
                prog.tick(1, a)
    return prog.audio_s


def measure_episode(ep: Dict[str, Any], plan: Dict[int, Dict[str, Any]], cache: str) -> Dict[str, Any]:
    durs = {}
    ep = json.loads(json.dumps(ep))
    for i, p in plan.items():
        side = read_sidecar(cache, p["key"])
        durs[i] = side["samples"] / SAMPLE_RATE
        ep["events"][i]["audio"] = {"clip": p["key"], "voice": p["voice"], "speed": p["speed"],
                                    "samples": side["samples"]}
    return apply_measured(ep, durs)


def cmd_synth(a: argparse.Namespace) -> int:
    try:
        backend = make_backend(backend_spec(a))
    except MissingCredentials as e:
        print(e, file=sys.stderr)
        return 2
    eps = read_jsonl(a.episodes)
    if a.limit:
        eps = eps[: a.limit]
    shard = parse_shard(a.shard)
    plans = [plan_episode(ep, backend) for ep in eps]
    jobs: Dict[str, Dict[str, Any]] = {}
    for plan in plans:
        for p in plan.values():
            jobs.setdefault(p["key"], p)
    todo = [j for k, j in sorted(jobs.items()) if in_shard(k, shard) and not clip_done(a.cache, k)]
    print(f"[synth] {len(eps)} episodes, {len(jobs)} unique clips, {len(jobs) - len(todo)} cached"
          + (f", shard {a.shard}" if shard else "") + f", {len(todo)} to synthesize with {backend.name}",
          file=sys.stderr, flush=True)
    t0 = time.time()
    audio_s = synthesize_jobs(todo, backend, a.cache, a.workers, a.progress_every)
    wall = time.time() - t0
    summary: Dict[str, Any] = {"episodes": len(eps), "unique_clips": len(jobs), "synthesized": len(todo),
                               "synth_audio_h": round(audio_s / 3600, 3), "synth_wall_s": round(wall, 1),
                               "realtime_factor": round(audio_s / wall, 1) if wall > 0 and audio_s else None}
    if shard or a.no_measure:
        summary["measured"] = "skipped (run synth again without --shard to write measured.jsonl)"
        print(json.dumps(summary, indent=2))
        return 0
    missing = [k for k in jobs if not clip_done(a.cache, k)]
    if missing:
        print(f"[synth] {len(missing)} clips still missing (e.g. {missing[0]}); not writing {a.out}", file=sys.stderr)
        return 1
    dropped: Counter = Counter()
    kept = 0
    examples: Dict[str, str] = {}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".tmp"
    total_h = 0.0
    with open(tmp, "w", encoding="utf-8") as f:
        for ep, plan in zip(eps, plans):
            m = measure_episode(ep, plan, a.cache)
            issues = validate_episode(m)
            if issues:
                dropped[issues[0].code] += 1
                examples.setdefault(issues[0].code, f"{ep['id']} [{ep['scenario']}]: {issues[0].message}")
                continue
            kept += 1
            total_h += max(e["t"] + e.get("dur", 0.0) for e in m["events"]) / 3600
            f.write(json.dumps(m, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(tmp, a.out)
    summary.update(measured=a.out, kept=kept, dropped=dict(dropped), dropped_examples=examples,
                   episode_hours_before_retime=round(total_h, 2))
    print(json.dumps(summary, indent=2))
    if sum(dropped.values()) > max(1, 0.01 * len(eps)):
        print(f"[synth] WARNING: dropped {sum(dropped.values())} of {len(eps)} episodes", file=sys.stderr)
        return 1
    return 0


# --- assemble -------------------------------------------------------------------------------

def _fsync(path: str) -> None:
    with open(path, "rb") as f:
        os.fsync(f.fileno())


def assemble_episode(ep: Dict[str, Any], cache: str, out_dir: str) -> Dict[str, str]:
    """Place cached clips at the episode's times; write <id>.wav and <id>.json."""
    evs = ep["events"]
    speech = [(i, e) for i, e in enumerate(evs) if e["type"] == "speech"]
    clips: Dict[int, array] = {}
    for i, e in speech:
        ref = (e.get("audio") or {}).get("clip")
        if not ref:
            raise AssembleError(f"{ep['id']} event {i}: no audio.clip (assemble needs episodes from `synth`)")
        if not clip_done(cache, ref):
            raise AssembleError(f"{ep['id']} event {i}: clip {ref} missing from {cache}")
        pcm = read_clip_pcm(cache, ref)
        clip_s = len(pcm) / SAMPLE_RATE
        if abs(float(e["dur"]) - clip_s) > DUR_TOLERANCE_S:
            raise AssembleError(f"{ep['id']} event {i}: dur {e['dur']} != clip length {clip_s:.4f} s ({ref}); "
                                "the episode was changed after synth")
        clips[i] = pcm
    end = max(e["t"] + e.get("dur", 0.0) for e in evs)
    n = int(math.ceil((end + 0.5) * SAMPLE_RATE))
    chans = [array("h", bytes(2 * n)), array("h", bytes(2 * n))]
    for i, e in speech:
        pcm = clips[i]
        start = int(round(float(e["t"]) * SAMPLE_RATE))
        ch = chans[ELLE_CH if e["ch"] == "elle" else CALLER_CH]
        if start + len(pcm) > len(ch):
            ch.extend(array("h", bytes(2 * (start + len(pcm) - len(ch)))))
        ch[start:start + len(pcm)] = pcm
    m = max(len(chans[0]), len(chans[1]))
    for c in chans:
        if len(c) < m:
            c.extend(array("h", bytes(2 * (m - len(c)))))
    inter = array("h", bytes(4 * m))
    inter[0::2] = chans[ELLE_CH]
    inter[1::2] = chans[CALLER_CH]
    os.makedirs(out_dir, exist_ok=True)
    wav_path = os.path.join(out_dir, f"{ep['id']}.wav")
    tmp = wav_path + ".tmp"
    with wave.open(tmp, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(inter.tobytes())
    _fsync(tmp)  # a Spot preemption is a power cut: unsynced files come back empty
    os.replace(tmp, wav_path)

    words, sources = [], set()
    for i, e in speech:
        if e["ch"] != "elle":
            continue
        side = read_sidecar(cache, e["audio"]["clip"])
        sources.add(side.get("timing", "proportional"))
        for w, s, t in side["words"]:
            words.append([w, [round(e["t"] + s, 4), round(e["t"] + t, 4)], "SPEAKER_MAIN"])
    spans = []
    for e in evs:
        if e["type"] == "tool_call":
            spans.append({"kind": "tool", "t": e["t"], "text": tool_call_text(e), "id": e["id"]})
        elif e["type"] == "tool_result":
            spans.append({"kind": "result", "t": e["t"], "text": tool_result_text(e), "id": e["id"]})
    align = {"episode_id": ep["id"], "sample_rate": SAMPLE_RATE, "alignments": words, "spans": spans,
             "word_timing": sources.pop() if len(sources) == 1 else ("mixed" if sources else "none")}
    json_path = os.path.join(out_dir, f"{ep['id']}.json")
    with open(json_path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(align, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(json_path + ".tmp", json_path)
    return {"wav": wav_path, "json": json_path}


def _clips_of(ep: Dict[str, Any]) -> List[str]:
    return [e["audio"]["clip"] for e in ep["events"] if e["type"] == "speech" and e.get("audio")]


def cmd_assemble(a: argparse.Namespace) -> int:
    eps = read_jsonl(a.episodes)
    if a.limit:
        eps = eps[: a.limit]
    shard = parse_shard(a.shard)
    if shard and a.delete_clips:
        raise SystemExit("--delete-clips cannot be combined with --shard (other shards may still need a clip)")
    eps = [ep for ep in eps if in_shard(ep["id"], shard)]
    for ep in eps:
        meta = ep.get("meta") or {}
        if not meta.get("retimed") and not a.allow_unretimed:
            raise SystemExit(f"{ep['id']}: not retimed. Assemble the output of "
                             "`training/model/prepare.py --write-retimed`, never raw or measured episodes "
                             "(or pass --allow-unretimed for a preview).")
        if not meta.get("measured"):
            raise SystemExit(f"{ep['id']}: has no measured durations; run `render_audio.py synth` first.")
    refs = Counter(k for ep in eps for k in _clips_of(ep))
    prog = Progress("assemble", len(eps), a.progress_every)
    done = skipped = deleted = 0
    for c0 in range(0, len(eps), a.chunk):
        chunk = eps[c0:c0 + a.chunk]
        for ep in chunk:
            wav = os.path.join(a.out, f"{ep['id']}.wav")
            if not a.force and os.path.exists(wav) and os.path.exists(os.path.splitext(wav)[0] + ".json"):
                skipped += 1
            else:
                assemble_episode(ep, a.cache, a.out)
                done += 1
            prog.tick(1, max(e["t"] + e.get("dur", 0.0) for e in ep["events"]))
        if a.delete_clips:
            for ep in chunk:
                for k in _clips_of(ep):
                    refs[k] -= 1
                    if refs[k] == 0:
                        for p in clip_paths(a.cache, k):
                            if os.path.exists(p):
                                os.remove(p)
                        try:
                            os.rmdir(os.path.dirname(clip_paths(a.cache, k)[0]))  # only if now empty
                        except OSError:
                            pass
                        deleted += 1
    print(json.dumps({"assembled": done, "skipped_existing": skipped, "clips_deleted": deleted,
                      "audio_h": round(prog.audio_s / 3600, 3), "out": a.out}, indent=2))
    return 0


# --- render (preview) -----------------------------------------------------------------------

def render_preview(eps: Sequence[Dict[str, Any]], backend: TTSBackend, out_dir: str) -> List[Dict[str, str]]:
    """synth + measure + assemble with no retiming: for listening, NOT for training."""
    cache = os.path.join(out_dir, "clips")
    out = []
    for ep in eps:
        plan = plan_episode(ep, backend)
        jobs = {p["key"]: p for p in plan.values()}
        for job in jobs.values():
            _synth_job(job, backend, cache)
        m = measure_episode(ep, plan, cache)
        out.append(assemble_episode(m, cache, out_dir))
    return out


def cmd_render(a: argparse.Namespace) -> int:
    try:
        backend = make_backend(backend_spec(a))
    except MissingCredentials as e:
        print(e, file=sys.stderr)
        return 2
    eps = read_jsonl(a.episodes)
    if a.limit:
        eps = eps[: a.limit]
    for p in render_preview(eps, backend, a.out_dir):
        print(p["wav"])
    print(f"rendered {len(eps)} preview episode(s) with {backend.name} (not retimed: not for training)",
          file=sys.stderr)
    return 0


# --- check-kokoro ----------------------------------------------------------------------------

CHECK_LINES = {
    "en": "Good news, he's open Thursday at 2 p.m. Want me to pencil you in?",
    "es": "Buenas noticias, está libre el jueves a las dos de la tarde. ¿Te lo aparto?",
}


def cmd_check_kokoro(a: argparse.Namespace) -> int:
    """Verify everything marked UNVERIFIED about the Kokoro backend, on the VM."""
    report: Dict[str, Any] = {"ok": True, "checks": []}

    def check(name: str, ok: bool, detail: Any = "") -> bool:
        report["checks"].append({"check": name, "ok": bool(ok), "detail": detail})
        report["ok"] &= bool(ok)
        print(f"[check-kokoro] {'PASS' if ok else 'FAIL'} {name}: {detail}", file=sys.stderr, flush=True)
        return ok

    try:
        import kokoro  # type: ignore
        import torch  # type: ignore
        check("import kokoro + torch", True, f"kokoro {getattr(kokoro, '__version__', '?')}, torch {torch.__version__}, "
              f"cuda {torch.cuda.is_available()} {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
    except Exception as e:  # noqa: BLE001
        check("import kokoro + torch", False, repr(e))
        print(json.dumps(report, indent=2))
        return 1
    try:
        from misaki import espeak  # type: ignore
        ps, _ = espeak.EspeakG2P(language="es")("Hola, ¿qué tal?")
        check("espeak-ng Spanish G2P (misaki)", bool(ps), ps)
    except Exception as e:  # noqa: BLE001
        check("espeak-ng Spanish G2P (misaki)", False, repr(e))
    try:
        import spacy  # type: ignore
        check("spaCy en_core_web_sm installed (else misaki downloads it at first use)",
              spacy.util.is_package("en_core_web_sm"), "python3 -m spacy download en_core_web_sm")
    except Exception as e:  # noqa: BLE001
        check("spaCy en_core_web_sm installed", False, repr(e))
    backend = KokoroBackend(device=a.device)
    try:
        from huggingface_hub import snapshot_download  # type: ignore
        snap = snapshot_download(KOKORO_REPO, local_files_only=True)
        have = {f[:-3] for f in os.listdir(os.path.join(snap, "voices")) if f.endswith(".pt")}
        wanted = set(backend.elle_voices.values()) | set(backend.caller_pools["en"]) | \
            set(backend.caller_pools["es"]) | {"af_bella"}
        check("Kokoro weights in local HF cache", os.path.exists(os.path.join(snap, "config.json")), snap)
        check("every configured voice exists", wanted <= have, sorted(wanted - have) or f"{len(wanted)} voices")
        report["available_voices"] = sorted(have)
    except Exception as e:  # noqa: BLE001
        check("Kokoro weights in local HF cache", False, f"{e!r}; download with snapshot_download('{KOKORO_REPO}')")
    os.makedirs(a.out, exist_ok=True)

    def synth(text: str, voice: str, lang: str, speed: float = 1.0, name: str = "") -> Optional[Clip]:
        try:
            clip = backend.synthesize(text, voice, lang, speed)
        except Exception as e:  # noqa: BLE001
            check(f"synthesize {voice}/{lang}", False, repr(e))
            return None
        if name:
            write_wav_mono(os.path.join(a.out, f"{name}.wav"), clip.pcm)
        return clip

    for lang in ("en", "es"):
        text = CHECK_LINES[lang]
        c = synth(text, backend.elle_voice(lang), lang, name=f"elle_{lang}_{backend.elle_voice(lang)}")
        if c is None:
            continue
        est = len(text.split()) / 2.6
        d = c.samples / SAMPLE_RATE
        check(f"Elle {lang} clip length plausible", 0.4 * est <= d <= 2.5 * est, f"{d:.2f} s vs estimate {est:.2f} s")
        if lang == "en":
            check("English word timestamps from Kokoro tokens", c.timing == "kokoro",
                  c.words[:4] if c.words else "none: falls back to proportional")
            if c.words:
                check("timestamps within the clip and increasing",
                      all(0 <= s <= t <= d + 0.05 for _, s, t in c.words)
                      and all(c.words[k][1] <= c.words[k + 1][1] for k in range(len(c.words) - 1)), "")
        else:
            check("Spanish timing is proportional (Kokoro gives no tokens for 'e')", c.timing == "proportional", c.timing)
        fast = synth(text, backend.elle_voice(lang), lang, 1.2)
        if fast is not None:
            check(f"speed=1.2 shortens the {lang} clip", fast.samples < c.samples, f"{fast.samples} < {c.samples}")
    synth(CHECK_LINES["en"], "af_bella", "en", name="elle_en_alt_af_bella")
    for lang in ("en", "es"):
        for v in backend.caller_pool(lang):
            synth("Sure, Thursday at two works for me." if lang == "en" else "Sí, el jueves a las dos me funciona.",
                  v, lang, name=f"caller_{lang}_{v}")
    # Throughput on typical lines.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from generate import build_episode  # noqa: WPS433
    lines = []
    for s in range(12):
        ep = build_episode("book_after_availability_free", s)
        lines += [(e["text"], e["ch"]) for e in ep["events"] if e["type"] == "speech"]
    lines = lines[:120]
    t0, audio = time.time(), 0.0
    for k, (text, ch) in enumerate(lines):
        v = backend.elle_voice("en") if ch == "elle" else backend.caller_pool("en")[k % len(backend.caller_pool("en"))]
        c = synth(text, v, "en")
        audio += c.samples / SAMPLE_RATE if c else 0.0
    wall = time.time() - t0
    rtf = audio / wall if wall else 0.0
    report["throughput"] = {"lines": len(lines), "audio_s": round(audio, 1), "wall_s": round(wall, 1),
                            "x_real_time_one_process": round(rtf, 1)}
    check("throughput measured", audio > 0, report["throughput"])
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[check-kokoro] listen to the samples in {a.out}", file=sys.stderr)
    return 0 if report["ok"] else 1


def write_wav_mono(path: str, pcm: bytes) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)


# --- cli ---------------------------------------------------------------------------------------

def backend_spec(a: argparse.Namespace) -> Dict[str, Any]:
    spec: Dict[str, Any] = {"name": a.backend}
    if a.backend == "kokoro" and getattr(a, "device", None):
        spec["device"] = a.device
    return spec


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmds = ("synth", "assemble", "render", "check-kokoro")
    if argv and argv[0] not in cmds and not argv[0].startswith("-"):
        argv = ["render"] + argv  # the first version's `render_audio.py episodes out_dir` form
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="synthesize clips and write measured.jsonl")
    s.add_argument("episodes")
    s.add_argument("--cache", required=True, help="clip cache directory (content-addressed, resumable)")
    s.add_argument("--out", required=True, help="measured.jsonl to write")
    s.add_argument("--backend", default="kokoro", choices=sorted(BACKENDS))
    s.add_argument("--device", default=None, help="kokoro: cuda or cpu (default: cuda if available)")
    s.add_argument("--workers", type=int, default=1, help="processes, each with its own model (all on the GPU)")
    s.add_argument("--shard", help="i/n: synthesize only this shard of clips (for several machines); "
                                   "then run once more without --shard to write measured.jsonl")
    s.add_argument("--no-measure", action="store_true")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--progress-every", type=float, default=10.0)
    s.set_defaults(fn=cmd_synth)

    s = sub.add_parser("assemble", help="place cached clips at retimed times into stereo WAV + alignment JSON")
    s.add_argument("episodes", help="retimed.jsonl from training/model/prepare.py --write-retimed")
    s.add_argument("--cache", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--chunk", type=int, default=500, help="episodes per chunk (clip deletion happens per chunk)")
    s.add_argument("--delete-clips", action="store_true", help="delete each clip once its last episode is assembled")
    s.add_argument("--shard", help="i/n: assemble only this shard of episodes")
    s.add_argument("--force", action="store_true", help="re-assemble episodes whose WAV already exists")
    s.add_argument("--allow-unretimed", action="store_true", help="preview only")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--progress-every", type=float, default=10.0)
    s.set_defaults(fn=cmd_assemble)

    s = sub.add_parser("render", help="quick preview (synth + assemble, no retiming); not for training")
    s.add_argument("episodes")
    s.add_argument("out_dir")
    s.add_argument("--backend", default="kokoro", choices=sorted(BACKENDS))
    s.add_argument("--device", default=None)
    s.add_argument("--limit", type=int, default=0)
    s.set_defaults(fn=cmd_render)

    s = sub.add_parser("check-kokoro", help="verify the Kokoro backend end to end (run on the VM)")
    s.add_argument("--out", default="kokoro_check")
    s.add_argument("--device", default=None)
    s.set_defaults(fn=cmd_check_kokoro)

    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except AssembleError as e:
        print(f"render_audio.py: ERROR: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
