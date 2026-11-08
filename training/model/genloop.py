"""The closed-loop evaluation loop, shared by every backend.

One loop drives any `FrameGenerator` frame by frame over an episode:

  * the caller's audio codes for the frame come from the prepared example
    (rows 8-15), i.e. the caller is replayed;
  * the text token is either forced (prompt frames, injected results) or left
    to the generator, which samples it with banned ids masked out;
  * the text stream is watched by runtime.ToolStreamParser; a completed
    `</tool>` is answered by an oracle from the episode's own scripted results
    after their latency (in frames), and runtime.Injector forces the
    `EPAD <result id=...>...</result>` tokens at the next word boundary.

Backends implement only `start` and `step` (and optionally `decode_audio`):
`TinyFrameGenerator` here (CPU, tested) and `moshi_backend.MoshiFrameGenerator`
(real LMGen, UNVERIFIED). So everything except the real adapter is tested.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import metrics as M
from runtime import Injector, ToolStreamParser
from textvocab import FRAME_RATE

N_CB = 8


class FrameGenerator:
    """Interface. A generator holds the model's streaming state for one episode."""

    #: True when the backend runs the system prompt itself (PersonaPlex's
    #: step_system_prompts); the loop then skips the example's prompt frames.
    native_prompt: bool = False

    def start(self, ex: dict, ep: dict, gcfg: dict) -> None:
        raise NotImplementedError

    def step(self, frame: int, caller_codes: Optional[np.ndarray], forced: Optional[int],
             banned: Sequence[int]) -> int:
        """Advance one 80 ms frame. `caller_codes`: int[8] (or None = silence).
        Returns the text token actually emitted at this frame (== forced when
        forced is not None)."""
        raise NotImplementedError

    def finish(self) -> None:
        """Leave streaming mode, free state."""

    def decode_audio(self) -> Optional[tuple]:
        """(pcm float32 [samples] for Elle, sample_rate) or None if the backend
        has no audio decoder (CPU/tiny)."""
        return None


class EpisodeOracle:
    """Answers the model's tool calls from the episode's own results: the first
    unused scripted call with the same name gives the content and latency.
    Anything else gets a plausible error after 0.5 s."""

    def __init__(self, ep: dict):
        evs = ep["events"]
        res = M.results_by_id(evs)
        self.avail = [(c, res.get(c.get("id"))) for c in M.calls(evs)]

    def answer(self, name: str, args: dict):
        for k, (c, r) in enumerate(self.avail):
            if c.get("name") == name and r is not None:
                self.avail.pop(k)
                return r.get("content", ""), float(r.get("latency_s") or 0.5)
        return f"Error: {name} could not be completed.", 0.5


def closed_loop(gen: FrameGenerator, tok, ex: dict, ep: dict, gcfg: dict) -> Dict[str, Any]:
    """Run one episode. Returns the emitted text stream (one token per frame),
    the calls the model made and when each result was ready and injected."""
    T = len(ex["text"])
    P = int(ex["meta"].get("prompt_frames", 0))
    parser, inj, oracle = ToolStreamParser(tok), Injector(tok, int(gcfg.get("max_wait_frames", 6))), EpisodeOracle(ep)
    gen.start(ex, ep, gcfg)
    stream: List[int] = []
    calls: List[dict] = []
    pending: Dict[str, int] = {}
    last: Optional[int] = None
    forced_frames = 0
    try:
        for f in range(T):
            if f < P and gen.native_prompt:
                stream.append(int(ex["text"][f]))  # the backend consumed the prompt itself
                continue
            caller = ex["audio"][N_CB:2 * N_CB, f]
            caller = None if (caller < 0).any() else caller.astype(np.int64)
            forced = int(ex["text"][f]) if f < P else inj.forced(f, last, parser)
            if forced is not None:
                forced_frames += 1
            tk = int(gen.step(f, caller, forced, inj.banned_ids(parser)))
            if forced is not None and tk != forced:
                raise RuntimeError(f"backend ignored a forced token at frame {f}: {tk} != {forced}")
            stream.append(tk)
            for ev in parser.feed(tk, f):
                if ev.kind == "tool_call":
                    cid = ev.payload["id"] or f"c{f}"
                    content, lat = oracle.answer(ev.payload["name"], ev.payload["args"])
                    ready = f + 1 + int(round(lat * FRAME_RATE))
                    inj.add_result(cid, content, ready)
                    pending[cid] = len(calls)
                    calls.append({"id": cid, "name": ev.payload["name"], "args": ev.payload["args"],
                                  "start_frame": ev.start_frame, "end_frame": f, "ready_frame": ready,
                                  "inject_frame": None})
                elif ev.kind == "result" and ev.payload.get("id") in pending:
                    calls[pending.pop(ev.payload["id"])]["inject_frame"] = ev.start_frame
            last = tk
    finally:
        gen.finish()
    return {"text": stream, "forced_frames": forced_frames, "calls": calls}


# --- tiny backend ---------------------------------------------------------------

class TinyFrameGenerator(FrameGenerator):
    """TinyDuplexLM behind the FrameGenerator interface (full recompute per
    frame; no KV cache, fine for CPU-sized episodes).

    gcfg.elle_audio: "generate" (closed loop: Elle's codes come from the depth
    transformer) or "teacher" (replay the example's Elle codes and generate
    text only; for dry runs, where the audio is placeholder noise)."""

    def __init__(self, model, device: str = "cpu"):
        self.model, self.device = model, device

    def start(self, ex: dict, ep: dict, gcfg: dict) -> None:
        import torch

        T = len(ex["text"])
        self.text = torch.full((1, T), -1, dtype=torch.long, device=self.device)
        self.audio = torch.from_numpy(ex["audio"].astype("int64"))[None].to(self.device).clone()
        self.teacher = gcfg.get("elle_audio", "generate") == "teacher"
        if not self.teacher:
            self.audio[:, :N_CB] = -1
        self.audio[:, N_CB:] = -1  # filled frame by frame from what the loop passes in
        self.temp = float(gcfg.get("temperature_text", 0.0))
        self.rng = torch.Generator().manual_seed(int(ep.get("seed", 0)))

    def step(self, frame, caller_codes, forced, banned) -> int:
        import torch

        f = frame
        if caller_codes is not None:
            self.audio[0, N_CB:, f] = torch.from_numpy(np.asarray(caller_codes, dtype=np.int64))
        with torch.no_grad():
            h = self.model.temporal(self.text[:, : f + 1], self.audio[:, :, : f + 1])[:, -1]
            if forced is not None:
                tk = int(forced)
            else:
                logits = self.model.text_linear(h)[0].float()
                logits[list(banned)] = float("-inf")
                if self.temp > 0:
                    tk = int(torch.multinomial(torch.softmax(logits / self.temp, -1), 1, generator=self.rng))
                else:
                    tk = int(logits.argmax())
            self.text[0, f] = tk
            if not self.teacher:
                self.audio[0, :N_CB, f] = self.model.sample_depth(h, torch.tensor([tk], device=self.device))[0]
        return tk


def stitch_windows(examples: List[dict]) -> List[dict]:
    """Rejoin an episode's windows (`ep#w0`, `ep#w1`, ...) into one example so
    evaluation runs whole episodes. Later windows lose their repeated prompt
    prefix; the frame offsets stay consistent."""
    by_ep: Dict[str, List[dict]] = {}
    order: List[str] = []
    for ex in examples:
        if ex["episode_id"] not in by_ep:
            order.append(ex["episode_id"])
        by_ep.setdefault(ex["episode_id"], []).append(ex)
    out = []
    for eid in order:
        parts = sorted(by_ep[eid], key=lambda e: e["meta"]["frame_offset"])
        if len(parts) == 1:
            out.append(parts[0])
            continue
        first = parts[0]
        keys = ("text", "loss_mask", "neg_mask", "audio_mask")
        joined = {k: [first[k]] for k in keys}
        audio = [first["audio"]]
        for p in parts[1:]:
            P = int(p["meta"].get("prompt_frames", 0))
            for k in keys:
                joined[k].append(p[k][P:])
            audio.append(p["audio"][:, P:])
        ex = dict(first, id=eid, **{k: np.concatenate(v) for k, v in joined.items()})
        ex["audio"] = np.concatenate(audio, axis=1)
        ex["meta"] = dict(first["meta"], id=eid)
        out.append(ex)
    return out
