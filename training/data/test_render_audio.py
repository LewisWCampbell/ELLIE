"""Audio stage: measured durations, synth cache, assembly, alignments, Kokoro adapter.
CPU only and offline: the fake backend stands in for TTS, a mock for Kokoro."""

import contextlib
import copy
import io
import json
import os
import sys
import tempfile
import unittest
import wave
from array import array
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import render_audio as R  # noqa: E402
from audio_timing import apply_measured  # noqa: E402
from generate import build_episode, generate_split, write_jsonl  # noqa: E402
from scenarios import SCENARIOS  # noqa: E402
from tts_backends import (KOKORO_CALLERS_EN, KOKORO_CALLERS_ES, KOKORO_ELLE, SAMPLE_RATE, FakeBackend, KokoroBackend,  # noqa: E402
                          clip_key, event_voice, tokens_to_words)
from validate import validate_episode  # noqa: E402

SR = SAMPLE_RATE


def fake_durs(ep, be=None):
    be = be or FakeBackend()
    out = {}
    for i, e in enumerate(ep["events"]):
        if e["type"] == "speech":
            v, l, s = event_voice(ep, e, be)
            out[i] = be.synthesize(e["text"], v, l, s).samples / SR
    return out


def slurp(path, mode="rb"):
    with open(path, mode) as f:
        return f.read()


def run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        code = R.main(argv)
    out = buf.getvalue()
    try:
        return code, json.loads(out)
    except ValueError:
        return code, out


def fake_retime(ep, after_index, shift):
    """Stand-in for prepare.py's retime: shift every event from `after_index` on."""
    ep = copy.deepcopy(ep)
    for e in ep["events"][after_index:]:
        e["t"] = round(e["t"] + shift, 3)
    ep.setdefault("meta", {})["retimed"] = True
    return ep


# --- measured durations ---------------------------------------------------------------------

def sp(t, dur, ch, text, **kw):
    return dict({"t": t, "dur": dur, "ch": ch, "type": "speech", "text": text}, **kw)


HAND = {
    "id": "ep_000001", "split": "train", "scenario": "hand", "seed": 1, "now": "2026-11-09T10:15:00-08:00",
    "events": [
        sp(0.0, 2.0, "caller", "Is he free Thursday at two?"),                          # 0
        sp(2.4, 1.2, "elle", "Let me check his calendar.", act="pre_call"),             # 1
        {"t": 2.5, "ch": "elle", "type": "tool_call", "id": "c1", "name": "check_availability",
         "args": {"start": "2026-11-12T14:00:00-08:00"}},                                # 2
        {"t": 3.3, "ch": "system", "type": "tool_result", "id": "c1", "content": "FREE: ok", "latency_s": 0.8},  # 3
        sp(3.9, 2.6, "elle", "He's free then. Want me to pencil you in?", act="claim"),  # 4
        sp(4.94, 0.5, "caller", "Mm-hm.", act="backchannel"),                            # 5
        sp(6.8, 1.0, "elle", "So shall I", act="pitch", interrupted=True),              # 6
        sp(7.55, 1.5, "caller", "Yes, book it please.", act="barge_in"),                # 7
    ],
    "labels": {"expected_calls": [{"name": "check_availability", "args": {"start": "2026-11-12T14:00:00-08:00"}}],
               "claim_spans": [{"event_index": 4, "depends_on": "c1"}], "forbidden_before_result": True,
               "negative": False, "write_gates": []},
}


class TestMeasured(unittest.TestCase):
    def test_hand_episode_exact(self):
        new = {0: 3.0, 1: 2.4, 4: 2.0, 5: 0.4, 6: 1.5, 7: 2.0}
        m = apply_measured(HAND, new)
        t = [e["t"] for e in m["events"]]
        self.assertEqual(t[0], 0.0)
        self.assertAlmostEqual(t[1], 3.0 + 0.4)                      # gap after the caller kept
        self.assertAlmostEqual(t[2], 3.4 + (0.1 / 1.2) * 2.4)        # same fraction into "let me check"
        self.assertAlmostEqual(t[3], t[2] + 0.8)                     # latency exact
        # floor = "let me check" (it ended last, at 3.6 > result 3.3): keep its 0.3 s gap
        self.assertAlmostEqual(t[4], max(3.4 + 2.4 + 0.3, t[3] + 0.15), places=3)
        self.assertAlmostEqual(t[5], t[4] + (1.04 / 2.6) * 2.0, places=3)        # backchannel fraction
        self.assertAlmostEqual(t[6], max(t[4] + 2.0 + 0.3, t[5] + 0.4 + 0.15), places=3)
        # Barge-in: the caller still starts 0.25 s before Elle's (cut) line ends.
        self.assertAlmostEqual((t[6] + 1.5) - t[7], 0.25, places=3)
        self.assertEqual([e["dur"] for e in m["events"] if e["type"] == "speech"], [3.0, 2.4, 2.0, 0.4, 1.5, 2.0])
        self.assertEqual(validate_episode(m), [])

    def test_all_scenarios_stay_valid_with_variable_lengths(self):
        be = FakeBackend(0.6, 1.5)
        dropped = total = 0
        for s in SCENARIOS:
            for lang in s.langs:
                for seed in range(15):
                    ep = build_episode(s.name, seed, lang=lang)
                    m = apply_measured(ep, fake_durs(ep, be))
                    total += 1
                    issues = validate_episode(m)
                    if issues:
                        # Only a premature-claim negative may lose its violation, when Elle's
                        # "let me check" grows past the result; synth drops those.
                        self.assertEqual(s.name, "neg_premature_availability_claim", issues[:2])
                        dropped += 1
        self.assertLess(dropped / total, 0.01)

    def test_latencies_and_order_preserved(self):
        for seed in range(30):
            ep = build_episode("availability_mind_change_mid_wait", seed)
            m = apply_measured(ep, fake_durs(ep))
            calls = {e["id"]: e for e in m["events"] if e["type"] == "tool_call"}
            for r in (e for e in m["events"] if e["type"] == "tool_result"):
                self.assertAlmostEqual(r["t"] - calls[r["id"]]["t"], r["latency_s"], delta=0.002)
            self.assertEqual([e["t"] for e in m["events"]], sorted(e["t"] for e in m["events"]))
            self.assertEqual(m["labels"]["expected_calls"], ep["labels"]["expected_calls"])

    def test_gaps_after_the_floor_are_kept(self):
        for name in ("book_after_availability_busy", "save_fact_midconversation", "bookings_two"):
            for seed in range(10):
                ep = build_episode(name, seed)
                m = apply_measured(ep, fake_durs(ep))
                old, new = ep["events"], m["events"]
                self.assertEqual([e.get("text") for e in old], [e.get("text") for e in new])  # no reorder here
                # A speech turn that started after everything before it ended keeps its gap
                # (or more, when it must not overlap something that grew, or its own channel).
                for i in range(1, len(old)):
                    if old[i]["type"] != "speech":
                        continue
                    ends = [p["t"] + p.get("dur", 0) for p in old[:i]]
                    if max(ends) <= old[i]["t"]:
                        gap = old[i]["t"] - max(ends)
                        new_gap = new[i]["t"] - max(p["t"] + p.get("dur", 0) for p in new[:i])
                        self.assertGreaterEqual(new_gap, min(gap, 0.02) - 0.002)
                        a = max(range(i), key=lambda j: (ends[j], j))
                        self.assertGreaterEqual(new[i]["t"] - (new[a]["t"] + new[a].get("dur", 0)), gap - 0.002)

    def test_unchanged_durations_are_identity(self):
        ep = build_episode("book_slot_taken_race", 2)
        same = {i: e["dur"] for i, e in enumerate(ep["events"]) if e["type"] == "speech"}
        m = apply_measured(ep, same)
        self.assertEqual([e["t"] for e in m["events"]], [round(e["t"], 3) for e in ep["events"]])


# --- voices ---------------------------------------------------------------------------------

class TestVoices(unittest.TestCase):
    def test_plan_is_deterministic_and_sane(self):
        be = KokoroBackend(pipeline_factory=lambda **k: None)
        seen = set()
        for seed in range(80):
            for name in ("book_after_availability_free", "es_switch_midcall"):
                ep = build_episode(name, seed)
                voices = {}
                for e in ep["events"]:
                    if e["type"] != "speech":
                        continue
                    v, lang, speed = event_voice(ep, e, be)
                    self.assertEqual((v, lang, speed), event_voice(ep, e, be))
                    if e["ch"] == "elle":
                        self.assertEqual(v, KOKORO_ELLE["en"] if lang == "en" else KOKORO_ELLE["es"])
                        self.assertEqual(speed, 1.0)
                    else:
                        voices.setdefault("caller", v)
                        self.assertEqual(v, voices["caller"])  # one caller voice per episode
                        self.assertTrue(0.88 <= speed <= 1.12)
                        pool = KOKORO_CALLERS_ES if ep["labels"]["lang"] != "en" else KOKORO_CALLERS_EN
                        self.assertIn(v, pool)
                        seen.add(v)
        self.assertTrue(any(v.startswith(("af_", "bf_")) for v in seen))
        self.assertTrue(any(v.startswith(("am_", "bm_")) for v in seen))
        self.assertTrue(any(v.startswith("b") for v in seen))
        self.assertNotIn(KOKORO_ELLE["en"], seen)  # callers never use Elle's voice
        self.assertNotIn(KOKORO_ELLE["es"], seen)

    def test_clip_key_depends_on_everything(self):
        be = FakeBackend()
        k = clip_key(be, "v", "en", 1.0, "Hi.")
        self.assertEqual(k, clip_key(be, "v", "en", 1.0, "Hi."))
        for other in (clip_key(be, "w", "en", 1.0, "Hi."), clip_key(be, "v", "es", 1.0, "Hi."),
                      clip_key(be, "v", "en", 1.01, "Hi."), clip_key(be, "v", "en", 1.0, "Hi!")):
            self.assertNotEqual(k, other)


# --- synth / assemble -------------------------------------------------------------------------

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        self.cache = os.path.join(self.d, "clips")
        self.eps = generate_split("val", 12, base_seed=5, holdout_family=None)
        self.src = os.path.join(self.d, "eps.jsonl")
        write_jsonl(self.src, self.eps)
        self.measured = os.path.join(self.d, "measured.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def synth(self, *extra):
        return run(["synth", self.src, "--cache", self.cache, "--out", self.measured, "--backend", "fake", *extra])

    def retimed(self):
        out = [fake_retime(ep, len(ep["events"]) // 2, 1.25) for ep in R.read_jsonl(self.measured)]
        p = os.path.join(self.d, "retimed.jsonl")
        write_jsonl(p, out)
        return p, out

    def test_synth_measures_and_resumes(self):
        code, s = self.synth()
        self.assertEqual(code, 0, s)
        self.assertEqual(s["kept"] + sum(s["dropped"].values()), 12)
        self.assertGreater(s["synthesized"], 0)
        m = R.read_jsonl(self.measured)
        for ep in m:
            self.assertEqual(validate_episode(ep), [])
            self.assertTrue(ep["meta"]["measured"])
            for e in ep["events"]:
                if e["type"] == "speech":
                    side = R.read_sidecar(self.cache, e["audio"]["clip"])
                    self.assertAlmostEqual(e["dur"], side["samples"] / SR, delta=0.0006)
        before = slurp(self.measured)
        code, s2 = self.synth()
        self.assertEqual((code, s2["synthesized"]), (0, 0))         # resume: everything cached
        self.assertEqual(slurp(self.measured), before)  # deterministic

    def test_interrupted_line_synthesizes_spoken_words_only(self):
        ep = next(e for s in range(40) for e in [build_episode("barge_in_then_book", s)]
                  if any(x.get("interrupted") for x in e["events"]))
        plan = R.plan_episode(ep, FakeBackend())
        i = next(i for i, e in enumerate(ep["events"]) if e.get("interrupted"))
        self.assertEqual(plan[i]["text"], ep["events"][i]["text"])
        self.assertNotEqual(plan[i]["text"], ep["events"][i]["full_text"])

    def test_sharded_synth_then_measure(self):
        for i in range(3):
            code, s = self.synth("--shard", f"{i}/3")
            self.assertEqual(code, 0)
            self.assertFalse(os.path.exists(self.measured))
        code, s = self.synth()
        self.assertEqual((code, s["synthesized"]), (0, 0))
        self.assertTrue(os.path.exists(self.measured))

    def test_assemble_places_clips_per_channel(self):
        self.synth()
        path, eps = self.retimed()
        out = os.path.join(self.d, "audio")
        code, s = run(["assemble", path, "--cache", self.cache, "--out", out])
        self.assertEqual(code, 0, s)
        for ep in eps:
            with wave.open(os.path.join(out, f"{ep['id']}.wav")) as w:
                self.assertEqual((w.getnchannels(), w.getframerate(), w.getsampwidth()), (2, SR, 2))
                a = array("h")
                a.frombytes(w.readframes(w.getnframes()))
            ch = {"elle": a[0::2], "caller": a[1::2]}
            for e in ep["events"]:
                if e["type"] != "speech":
                    continue
                side = R.read_sidecar(self.cache, e["audio"]["clip"])
                mark = FakeBackend.marker(side["voice"], side["text"])
                start = int(round(e["t"] * SR))
                n = side["samples"]
                self.assertEqual(ch[e["ch"]][start], mark, (ep["id"], e["text"]))
                self.assertEqual(ch[e["ch"]][start + n - 1], mark)
                if start > 0 and ch[e["ch"]][start - 1] != 0:
                    self.fail(f"{ep['id']}: something else on {e['ch']} right before {e['text']!r}")
            align = json.loads(slurp(os.path.join(out, f"{ep['id']}.json"), "r"))
            elle = [e for e in ep["events"] if e["type"] == "speech" and e["ch"] == "elle"]
            self.assertEqual([w[0] for w in align["alignments"]], [w for e in elle for w in e["text"].split()])
            first = elle[0]
            side = R.read_sidecar(self.cache, first["audio"]["clip"])
            self.assertAlmostEqual(align["alignments"][0][1][0], first["t"] + side["words"][0][1], places=3)
            self.assertEqual(align["word_timing"], "fake")
            self.assertEqual({s["kind"] for s in align["spans"]} <= {"tool", "result"}, True)

    def test_assemble_resume_and_delete_clips(self):
        self.synth()
        path, eps = self.retimed()
        out = os.path.join(self.d, "audio")
        code, s = run(["assemble", path, "--cache", self.cache, "--out", out, "--limit", "4"])
        self.assertEqual(s["assembled"], 4)
        code, s = run(["assemble", path, "--cache", self.cache, "--out", out, "--delete-clips", "--chunk", "5"])
        self.assertEqual((code, s["assembled"], s["skipped_existing"]), (0, 8, 4))
        left = [f for _, _, fs in os.walk(self.cache) for f in fs]
        self.assertEqual(left, [])
        self.assertGreater(s["clips_deleted"], 0)

    def test_missing_clip_fails_loudly(self):
        self.synth()
        path, eps = self.retimed()
        key = next(e["audio"]["clip"] for e in eps[0]["events"] if e["type"] == "speech")
        os.remove(R.clip_paths(self.cache, key)[0])
        code, out = run(["assemble", path, "--cache", self.cache, "--out", os.path.join(self.d, "a")])
        self.assertEqual(code, 3)

    def test_duration_mismatch_fails_loudly(self):
        self.synth()
        _, eps = self.retimed()
        i = next(i for i, e in enumerate(eps[0]["events"]) if e["type"] == "speech")
        eps[0]["events"][i]["dur"] += 0.05
        with self.assertRaises(R.AssembleError) as cm:
            R.assemble_episode(eps[0], self.cache, os.path.join(self.d, "a"))
        self.assertIn("clip length", str(cm.exception))

    def test_refuses_unretimed(self):
        self.synth()
        with self.assertRaises(SystemExit) as cm:
            R.main(["assemble", self.measured, "--cache", self.cache, "--out", os.path.join(self.d, "a")])
        self.assertIn("retimed", str(cm.exception))

    def test_workers_produce_the_same_clips(self):
        jobs = {}
        for ep in self.eps[:4]:
            for p in R.plan_episode(ep, FakeBackend()).values():
                jobs[p["key"]] = p
        c1, c2 = os.path.join(self.d, "w1"), os.path.join(self.d, "w2")
        with contextlib.redirect_stderr(io.StringIO()):
            R.synthesize_jobs(list(jobs.values()), FakeBackend(), c1, workers=1)
            R.synthesize_jobs(list(jobs.values()), FakeBackend(), c2, workers=2)
        for k in jobs:
            with open(R.clip_paths(c1, k)[0], "rb") as f1, open(R.clip_paths(c2, k)[0], "rb") as f2:
                self.assertEqual(f1.read(), f2.read())

    def test_with_trainer_retime(self):
        """measured.jsonl -> training/model/prepare.py retime_episode -> assemble."""
        sys.path.insert(0, os.path.join(HERE, "..", "model"))
        try:
            from prepare import retime_episode  # noqa: WPS433
            from serializer_adapter import contract_segments
            from textvocab import ChunkTokenizer
        except Exception as e:  # noqa: BLE001
            self.skipTest(f"trainer not importable here: {e!r}")
        self.synth()
        m = R.read_jsonl(self.measured)
        tok = ChunkTokenizer.build([s["text"] for e in m for s in contract_segments(e)])
        rt = [retime_episode(ep, tok) for ep in m]
        p = os.path.join(self.d, "retimed.jsonl")
        write_jsonl(p, rt)
        code, s = run(["assemble", p, "--cache", self.cache, "--out", os.path.join(self.d, "audio")])
        self.assertEqual(code, 0, s)
        self.assertEqual(s["assembled"], len(rt))


# --- kokoro adapter (pipeline mocked) ------------------------------------------------------------

class Tok(SimpleNamespace):
    pass


class MockPipeline:
    instances = []

    def __init__(self, lang_code, repo_id, model):
        self.lang_code, self.repo_id, self.model = lang_code, repo_id, model
        self.calls = []
        MockPipeline.instances.append(self)

    def __call__(self, text, voice=None, speed=1.0, split_pattern=None):
        self.calls.append((text, voice, speed, split_pattern))
        if self.lang_code == "e":  # espeak path: no tokens
            yield SimpleNamespace(audio=[0.5] * 12000, tokens=None)
            return
        # Two chunks, like KPipeline splitting a long English line. Punctuation is its own token.
        yield SimpleNamespace(audio=[0.25] * 24000, tokens=[
            Tok(text="Good", whitespace=" ", start_ts=0.10, end_ts=0.40),
            Tok(text="news", whitespace="", start_ts=0.40, end_ts=0.70),
            Tok(text=",", whitespace=" ", start_ts=0.70, end_ts=0.75),
            Tok(text="he's", whitespace=" ", start_ts=None, end_ts=None),
            Tok(text="open", whitespace="", start_ts=0.80, end_ts=0.95),
            Tok(text=".", whitespace=" ", start_ts=0.95, end_ts=0.97)])
        yield SimpleNamespace(audio=[-2.0] * 12000, tokens=[
            Tok(text="Want", whitespace=" ", start_ts=0.05, end_ts=0.20),
            Tok(text="it", whitespace="", start_ts=0.20, end_ts=0.40),
            Tok(text="?", whitespace="", start_ts=0.40, end_ts=0.45)])


class TestKokoroAdapter(unittest.TestCase):
    def setUp(self):
        MockPipeline.instances = []
        self.be = KokoroBackend(pipeline_factory=MockPipeline, model_factory=lambda: "MODEL")

    def test_english_chunks_concatenated_with_timestamps(self):
        clip = self.be.synthesize("Good news, he's open. Want it?", "af_heart", "en", 1.0)
        self.assertEqual(clip.samples, 36000)
        pcm = array("h")
        pcm.frombytes(clip.pcm)
        self.assertEqual(pcm[0], round(0.25 * 32767))
        self.assertEqual(pcm[-1], -32767)  # clipped to [-1, 1]
        self.assertEqual(clip.timing, "kokoro")
        words = [w for w, _, _ in clip.words]
        self.assertEqual(words, ["Good", "news,", "he's", "open.", "Want", "it?"])
        self.assertAlmostEqual(clip.words[1][2], 0.75)       # punctuation joins its word
        self.assertAlmostEqual(clip.words[4][1], 1.0 + 0.05)  # second chunk offset by the first's length
        self.assertAlmostEqual(clip.words[2][1], 0.75)        # untimed token: interpolated
        p = MockPipeline.instances[0]
        self.assertEqual((p.lang_code, p.repo_id, p.model), ("a", "hexgrad/Kokoro-82M", "MODEL"))
        self.assertEqual(p.calls[0][1:], ("af_heart", 1.0, None))

    def test_spanish_is_proportional_and_uses_e(self):
        clip = self.be.synthesize("Buenas noticias, está libre.", "ef_dora", "es", 0.97)
        self.assertEqual(clip.timing, "proportional")
        self.assertIsNone(clip.words)
        self.assertEqual(MockPipeline.instances[0].lang_code, "e")
        self.assertEqual(MockPipeline.instances[0].calls[0][2], 0.97)

    def test_british_voice_uses_british_g2p_and_pipelines_are_reused(self):
        self.be.synthesize("Good news, he's open. Want it?", "bm_george", "en")
        self.be.synthesize("Good news, he's open. Want it?", "bf_emma", "en")
        self.be.synthesize("Good news, he's open. Want it?", "am_eric", "en")
        self.assertEqual([p.lang_code for p in MockPipeline.instances], ["b", "a"])

    def test_token_mismatch_falls_back_to_proportional(self):
        clip = self.be.synthesize("Totally different words here", "af_heart", "en")
        self.assertEqual(clip.timing, "proportional")
        self.assertIsNone(clip.words)

    def test_tokens_to_words_none_when_untimed(self):
        toks = [("Hi", " ", None, None), ("there", "", None, None)]
        self.assertIsNone(tokens_to_words("Hi there", toks, 1.0))

    def test_backend_in_full_synth_flow(self):
        with tempfile.TemporaryDirectory() as d:
            ep = build_episode("availability_free", 3, lang="en")
            plan = R.plan_episode(ep, self.be)
            for p in plan.values():
                p["text"] = "Good news, he's open. Want it?"  # the mock only speaks this line
                R._synth_job(p, self.be, d)
            side = R.read_sidecar(d, next(iter(plan.values()))["key"])
            self.assertEqual(side["samples"], 36000)


if __name__ == "__main__":
    unittest.main()


class TestCheckKokoro(unittest.TestCase):
    """Runs the VM verification command against stub modules, so its own logic
    (not Kokoro) is tested here. On the VM it runs against the real thing."""

    def test_check_kokoro_with_stubs(self):
        import types
        from tts_backends import KOKORO_CALLERS_EN, KOKORO_CALLERS_ES, KOKORO_ELLE

        class StubPipeline:
            def __init__(self, lang_code, repo_id, model):
                self.lang_code = lang_code

            def __call__(self, text, voice=None, speed=1.0, split_pattern=None):
                n = int(SR * len(text.split()) / 2.6 / speed)
                toks = None
                if self.lang_code in "ab":
                    words = text.split()
                    step = n / SR / len(words)
                    toks = [Tok(text=w, whitespace=" " if k < len(words) - 1 else "",
                                start_ts=k * step, end_ts=(k + 1) * step) for k, w in enumerate(words)]
                yield SimpleNamespace(audio=[0.1] * n, tokens=toks)

        class StubModel:
            def __init__(self, repo_id=None):
                pass

            def to(self, dev):
                return self

            def eval(self):
                return self

        with tempfile.TemporaryDirectory() as d:
            snap = os.path.join(d, "snap")
            os.makedirs(os.path.join(snap, "voices"))
            open(os.path.join(snap, "config.json"), "w").close()
            for v in set(KOKORO_ELLE.values()) | set(KOKORO_CALLERS_EN) | set(KOKORO_CALLERS_ES) | {"af_bella"}:
                open(os.path.join(snap, "voices", v + ".pt"), "w").close()
            mods = {
                "kokoro": types.SimpleNamespace(__version__="0.9.4", KModel=StubModel, KPipeline=StubPipeline),
                "torch": types.SimpleNamespace(__version__="stub", cuda=types.SimpleNamespace(
                    is_available=lambda: False, get_device_name=lambda i: "")),
                "misaki": types.SimpleNamespace(),
                "misaki.espeak": types.SimpleNamespace(EspeakG2P=lambda language: (lambda t: ("ˈola", None))),
                "spacy": types.SimpleNamespace(util=types.SimpleNamespace(is_package=lambda n: True)),
                "huggingface_hub": types.SimpleNamespace(snapshot_download=lambda repo, local_files_only: snap),
            }
            mods["misaki"].espeak = mods["misaki.espeak"]
            saved = {k: sys.modules.get(k) for k in mods}
            sys.modules.update(mods)
            try:
                code, out = run(["check-kokoro", "--out", os.path.join(d, "out")])
            finally:
                for k, v in saved.items():
                    if v is None:
                        sys.modules.pop(k, None)
                    else:
                        sys.modules[k] = v
            self.assertEqual(code, 0, out)
            self.assertTrue(out["ok"])
            self.assertGreater(out["throughput"]["audio_s"], 0)
            wavs = os.listdir(os.path.join(d, "out"))
            self.assertIn(f"elle_en_{KOKORO_ELLE['en']}.wav", wavs)
            self.assertIn("caller_es_em_alex.wav", wavs)
