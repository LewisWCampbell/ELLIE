"""The closed-loop evaluation loop, shared by every backend.

One loop drives any `FrameGenerator` frame by frame over an episode:

  * the caller's audio codes for the frame come from the prepared example
    (rows 8-15), i.e. the caller is replayed, -1 included: -1 is the zero
    embedding the model was trained on (prompt frames, audio tail), and
    replacing it with "silence" codes puts the model in a context it never saw;
  * during the prompt frames Elle's own audio is also forced from the example
    (-1, as in training) instead of being sampled, and with
    gen.elle_audio=teacher it is forced on every frame (a diagnostic);
  * the text token is either forced (prompt frames, injected results) or left
    to the generator, which samples it with banned ids masked out;
  * the text stream is watched by runtime.ToolStreamParser; a completed
    `</tool>` is answered by an oracle from the episode's own scripted results
    after their latency (in frames), and runtime.Injector forces the
    `EPAD <result id=...>...</result>` tokens at the next word boundary.

This loop once replaced -1 by silence codes and let the model sample its own
audio during the prompt. The model had been trained on -1 there, so every
conversation started off-distribution; on a miniature PersonaPlex trained to
near-zero loss that alone gave 0 tool calls out of 10, and 9-10 once fixed
(see test_genloop / test_personaplex).

Backends implement only `start` and `step` (and optionally `decode_audio`):
`TinyFrameGenerator` here (CPU, tested) and `moshi_backend.MoshiFrameGenerator`
(real LMGen, UNVERIFIED). So everything except the real adapter is tested.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import metrics as M
from runtime import Injector, ToolStreamParser
from textvocab import EPAD_ID, FRAME_RATE, PAD_ID, TOOL_OPEN

N_CB = 8


class FrameGenerator:
    """Interface. A generator holds the model's streaming state for one episode."""

    #: True when the backend runs the system prompt itself (PersonaPlex's
    #: step_system_prompts); the loop then skips the example's prompt frames.
    native_prompt: bool = False

    def start(self, ex: dict, ep: dict, gcfg: dict) -> None:
        raise NotImplementedError

    #: Per-frame diagnostics filled by backends that can see the text logits:
    #: one dict per step ({"p_tool", "p_pad", "p_epad", "argmax"}) or None.
    frame_stats: Optional[List[Optional[dict]]] = None
    #: Text ids to track in frame_stats; set by closed_loop before start().
    track_ids: Optional[Dict[str, int]] = None

    def step(self, frame: int, caller_codes: Optional[np.ndarray], forced: Optional[int],
             banned: Sequence[int], elle_codes: Optional[np.ndarray] = None) -> int:
        """Advance one 80 ms frame. `caller_codes`: int[8], -1 = the zero
        embedding (as in training), None = all -1. `elle_codes`: when given,
        Elle's audio for this frame is forced to these codes instead of being
        generated (-1 allowed). Returns the text token actually emitted at this
        frame (== forced when forced is not None)."""
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
    the calls the model made and when each result was ready and injected, and
    what happened to every call the model started: `tool_opens` (how many
    <tool> it emitted), `tool_errors` (spans that did not parse, padded out,
    timed out or were still open at the end, with their text)."""
    T = len(ex["text"])
    P = int(ex["meta"].get("prompt_frames", 0))
    teacher = gcfg.get("elle_audio", "generate") == "teacher"
    parser, inj, oracle = ToolStreamParser(tok), Injector(tok, int(gcfg.get("max_wait_frames", 6))), EpisodeOracle(ep)
    ids = tok.special_ids
    gen.track_ids = {"tool": ids[TOOL_OPEN], "pad": PAD_ID, "epad": EPAD_ID}
    gen.frame_stats = []
    gen.start(ex, ep, gcfg)
    stream: List[int] = []
    calls: List[dict] = []
    errors: List[dict] = []
    pending: Dict[str, int] = {}
    last: Optional[int] = None
    forced_frames = tool_opens = 0
    try:
        for f in range(T):
            if f < P and gen.native_prompt:
                stream.append(int(ex["text"][f]))  # the backend consumed the prompt itself
                continue
            caller = ex["audio"][N_CB:2 * N_CB, f].astype(np.int64)
            elle = ex["audio"][:N_CB, f].astype(np.int64) if (f < P or teacher) else None
            forced = int(ex["text"][f]) if f < P else inj.forced(f, last, parser)
            if forced is not None:
                forced_frames += 1
            tk = int(gen.step(f, caller, forced, inj.banned_ids(parser), elle_codes=elle))
            if forced is not None and tk != forced:
                raise RuntimeError(f"backend ignored a forced token at frame {f}: {tk} != {forced}")
            if forced is None and tk == ids[TOOL_OPEN]:
                tool_opens += 1
            stream.append(tk)
            for ev in parser.feed(tk, f):
                if ev.kind == "tool_error":
                    errors.append({"frame": f, "start_frame": ev.start_frame, **ev.payload})
                elif ev.kind == "tool_call":
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
        for ev in parser.flush(T):
            errors.append({"frame": T, "start_frame": ev.start_frame, **ev.payload})
    finally:
        gen.finish()
    return {"text": stream, "forced_frames": forced_frames, "calls": calls, "tool_opens": tool_opens,
            "tool_errors": errors, "frame_stats": list(gen.frame_stats or [])}


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

    def step(self, frame, caller_codes, forced, banned, elle_codes=None) -> int:
        import torch

        f = frame
        if caller_codes is not None:
            self.audio[0, N_CB:, f] = torch.from_numpy(np.asarray(caller_codes, dtype=np.int64))
        if elle_codes is not None:
            self.audio[0, :N_CB, f] = torch.from_numpy(np.asarray(elle_codes, dtype=np.int64))
        with torch.no_grad():
            h = self.model.temporal(self.text[:, : f + 1], self.audio[:, :, : f + 1])[:, -1]
            logits = self.model.text_linear(h)[0].float()
            if self.track_ids is not None and self.frame_stats is not None:
                p = torch.softmax(logits, -1)
                self.frame_stats.append({"p_tool": float(p[self.track_ids["tool"]]), "p_pad": float(p[self.track_ids["pad"]]),
                                         "p_epad": float(p[self.track_ids["epad"]]), "argmax": int(p.argmax())})
            if forced is not None:
                tk = int(forced)
            else:
                logits[list(banned)] = float("-inf")
                if self.temp > 0:
                    tk = int(torch.multinomial(torch.softmax(logits / self.temp, -1), 1, generator=self.rng))
                else:
                    tk = int(logits.argmax())
            self.text[0, f] = tk
            if elle_codes is None and not self.teacher:
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
