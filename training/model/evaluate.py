"""Evaluation: the five CONTRACT.md metrics, as a JSON report + markdown summary.

Modes
  reference  Score the dataset itself, to test the metrics:
               * ground truth with the dataset's claim_spans labels: every
                 positive must score perfectly, every labelled negative must be
                 flagged;
               * the claim *detector* (used for model outputs, which have no
                 labels) against those labels: precision / recall;
               * round trip: the prepared training text stream decoded back into
                 events and scored with the detector, i.e. the exact path a
                 model's output takes.
  model      Closed-loop generation with a trained checkpoint: caller audio is
             replayed from the episode, Elle's text and audio are generated
             frame by frame, a </tool> triggers the tool (an oracle answering
             from the episode's own results, after its latency) and the result
             is injected into the text stream exactly as at runtime
             (runtime.Injector). The output is decoded into events and scored.
             The loop is genloop.closed_loop, shared by both backends: tiny
             (TinyFrameGenerator, CPU, tested) and real Moshi/PersonaPlex
             (moshi_backend.MoshiFrameGenerator over LMGen, UNVERIFIED).
             Windows of long episodes are stitched back into whole episodes.
             --listen N writes Elle's generated audio for N episodes as WAV
             (real backend; a no-op on CPU).

  python3 training/model/evaluate.py --config training/model/config/dryrun.yaml --mode reference
  python3 training/model/evaluate.py --config training/model/config/dryrun.yaml --mode model [--listen 5]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import metrics as M  # noqa: E402
from config_util import load_config, resolve_path  # noqa: E402
from runtime import decode_text_stream  # noqa: E402
from textvocab import build_tokenizer  # noqa: E402

GATED = ["tool_selection_accuracy", "arg_exact_match", "premature_claim_rate", "write_unsafe_rate",
         "silent_write_narration_rate", "dead_air_rate"]


def load_jsonl(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


# dataset labels.negative_kind -> the flag that must catch it
NEG_KIND_FLAG = {"premature_claim": "premature_claim", "write_without_readback": "unsafe_write",
                 "dead_air": "dead_air", "narrated_silent_write": "narrated_silent_write"}


def _neg_kind(ep_score: dict, eps_by_id: Dict[str, dict]) -> Optional[str]:
    ep = eps_by_id.get(ep_score["id"], {})
    return ep.get("labels", {}).get("negative_kind")


def _summarize_group(per: List[dict], eps_by_id: Optional[Dict[str, dict]] = None) -> Dict[str, Any]:
    pos = [p for p in per if not p["negative"]]
    neg = [p for p in per if p["negative"]]
    right = wrong_metric = 0
    for p in neg:
        k = _neg_kind(p, eps_by_id or {})
        if k in NEG_KIND_FLAG:
            if p["flags"].get(NEG_KIND_FLAG[k]):
                right += 1
            elif p["flagged"]:
                wrong_metric += 1
    return {
        "negatives_by_kind": {"caught_by_matching_metric": right, "caught_only_by_other_metric": wrong_metric,
                              "with_kind_label": sum(1 for p in neg if _neg_kind(p, eps_by_id or {}) in NEG_KIND_FLAG)},
        "positives": {"n": len(pos), "flagged": [{"id": p["id"], "flags": [k for k, v in p["flags"].items() if v]}
                                                for p in pos if p["flagged"]]},
        "negatives": {"n": len(neg), "caught": sum(1 for p in neg if p["flagged"]),
                      "uncaught": [{"id": p["id"], "scenario": p["scenario"]} for p in neg if not p["flagged"]],
                      "by_flag": M._count(k for p in neg for k, v in p["flags"].items() if v)},
    }


def detector_agreement(eps: List[dict]) -> Dict[str, Any]:
    """Event-level agreement of metrics.looks_like_claim with claim_spans."""
    tp = fp = fn = 0
    misses, false_alarms = [], []
    for ep in eps:
        labelled = {c.get("event_index") for c in ep.get("labels", {}).get("claim_spans", []) or []}
        for i, e in enumerate(ep["events"]):
            if e.get("type") != "speech" or e.get("ch") != "elle":
                continue
            det = M.looks_like_claim(e.get("text", ""))
            if det and i in labelled:
                tp += 1
            elif det:
                fp += 1
                if len(false_alarms) < 20:
                    false_alarms.append({"id": ep["id"], "text": e.get("text")})
            elif i in labelled:
                fn += 1
                if len(misses) < 20:
                    misses.append({"id": ep["id"], "text": e.get("text")})
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "misses": misses, "false_alarms": false_alarms}


def readback_agreement(eps: List[dict]) -> Dict[str, Any]:
    """The readback/yes heuristic (used on model outputs) vs labels.write_gates."""
    agree = disagree = 0
    examples = []
    for ep in eps:
        gates = {g.get("call_id"): g for g in ep.get("labels", {}).get("write_gates") or []}
        for c in M.calls(ep["events"]):
            if not M.is_gated(c) or c.get("id") not in gates:
                continue
            lab = bool(M.write_problems(c, ep["events"], gates[c.get("id")]))
            heu = bool(M.write_problems(c, ep["events"]))
            if lab == heu:
                agree += 1
            else:
                disagree += 1
                if len(examples) < 10:
                    examples.append({"id": ep["id"], "call": c.get("id"), "label_unsafe": lab, "heuristic_unsafe": heu,
                                     "heuristic": M.write_problems(c, ep["events"])})
    return {"agree": agree, "disagree": disagree,
            "agreement": agree / (agree + disagree) if agree + disagree else None, "disagreements": examples}


def reference_report(eps: List[dict], prepared: Optional[List[dict]], tok, gates: dict) -> Dict[str, Any]:
    by_id = {ep["id"]: ep for ep in eps}
    per = [M.score_episode(ep) for ep in eps]
    rep: Dict[str, Any] = {"mode": "reference", "ground_truth": {"metrics": M.aggregate([p for p in per if not p["negative"]]),
                                                                  "metrics_all": M.aggregate(per),
                                                                  **_summarize_group(per, by_id)},
                           "claim_detector": detector_agreement(eps), "readback_detector": readback_agreement(eps)}
    rep["ground_truth"]["gates_on_positives"] = M.check_gates(rep["ground_truth"]["metrics"], gates)
    if prepared:
        rt = []
        for ex in prepared:
            ep = by_id.get(ex["episode_id"])
            if ep is None or "#w" in ex["id"]:
                continue  # windows of long episodes are skipped in the round trip
            P = int(ex["meta"].get("prompt_frames", 0))
            rt.append(M.score_episode(ep, merge_events(ep, decode_text_stream(ex["text"][P:], tok,
                                                                              ex["meta"]["frame_offset"] + P)),
                                      claim_source="detected"))
        rep["round_trip"] = {"metrics": M.aggregate([p for p in rt if not p["negative"]]), **_summarize_group(rt, by_id)}
    rep["per_episode"] = per
    return rep


def merge_events(ep: dict, elle_events: List[dict]) -> List[dict]:
    """Caller side from the episode (it is replayed), Elle side + results from
    the decoded stream."""
    caller = [e for e in ep["events"] if e.get("type") == "speech" and e.get("ch") != "elle"]
    return sorted(caller + elle_events, key=lambda e: e["t"])


# --- model mode -------------------------------------------------------------------

def make_generator(cfg: dict, base: str, ckpt: Optional[str] = None):
    """(FrameGenerator, tokenizer, checkpoint path) for the configured backend.
    Both backends then run the same genloop.closed_loop."""
    from genloop import TinyFrameGenerator
    from train import latest_checkpoint, load_for_eval

    gcfg = cfg.get("eval", {}).get("gen", {})
    if cfg["model"]["backend"] == "tiny":
        model, tok, _new_ids, ckpt = load_for_eval(cfg, base, ckpt)
        return TinyFrameGenerator(model), tok, ckpt
    # Real Moshi / PersonaPlex. UNVERIFIED (never run): moshi_backend.load_for_inference
    # and MoshiFrameGenerator are written from source, see their docstrings.
    from moshi_backend import MoshiFrameGenerator, load_for_inference  # pragma: no cover

    ckpt = ckpt or latest_checkpoint(resolve_path(cfg["run_dir"], base))
    if ckpt is None:
        raise FileNotFoundError("no checkpoint to evaluate")
    device = cfg.get("train", {}).get("backend_device", "cuda")
    lm, mimi, tok, _ids = load_for_inference(cfg, ckpt, device)
    return MoshiFrameGenerator(lm, mimi, tok, gcfg, device), tok, ckpt


def write_wav(path: str, pcm, sample_rate: int) -> None:
    """Mono 16-bit WAV with the standard library (no soundfile dependency)."""
    import wave

    import numpy as np

    x = (np.clip(np.asarray(pcm, dtype=np.float32), -1.0, 1.0) * 32767).astype("<i2")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(x.tobytes())


def model_report(cfg: dict, base: str, eps: List[dict], prepared: List[dict], gates: dict,
                 ckpt: Optional[str] = None, listen: int = 0, out_dir: Optional[str] = None,
                 generator=None, tok=None) -> Dict[str, Any]:
    from genloop import closed_loop, stitch_windows

    if generator is None:
        generator, tok, ckpt = make_generator(cfg, base, ckpt)
    ecfg = cfg.get("eval", {})
    gcfg = ecfg.get("gen", {})
    by_id = {ep["id"]: ep for ep in eps}
    per, samples, listened, calls = [], [], [], []
    exs = stitch_windows([ex for ex in prepared if ex["episode_id"] in by_id])[: int(ecfg.get("max_episodes", 100))]
    listen_note = None
    diag = []
    for ex in exs:
        ep = by_id[ex["episode_id"]]
        out = closed_loop(generator, tok, ex, ep, gcfg)
        P = int(ex["meta"].get("prompt_frames", 0))
        # Score what the model said after the system prompt (the prompt is forced, not speech).
        elle = decode_text_stream(out["text"][P:], tok, ex["meta"]["frame_offset"] + P)
        sc = M.score_episode(ep, merge_events(ep, elle), claim_source="detected")
        per.append(sc)
        calls += [dict(c, episode=ep["id"]) for c in out["calls"]]
        d = episode_diagnostics(ex, out, tok)
        diag.append(dict(d, id=ep["id"]))
        if len(samples) < 5:
            samples.append({"id": ep["id"], "decoded": tok.decode(out["text"][P:]),
                            "reference": tok.decode([int(t) for t in ex["text"][P:]]),
                            "events": elle, "calls": out["calls"], "tool_errors": out["tool_errors"],
                            "diagnostics": d})
        if listen and len(listened) < listen and out_dir:
            audio = generator.decode_audio()
            if audio is None:
                listen_note = "listen: no-op, this backend has no audio decoder (CPU / tiny model, placeholder codes)"
            else:
                path = os.path.join(out_dir, "listen", f"{ep['id']}.wav")
                write_wav(path, audio[0], audio[1])
                listened.append(path)
    pos = [p for p in per if not p["negative"]]
    m = M.aggregate(pos)
    waits = [c["inject_frame"] - c["ready_frame"] for c in calls if c.get("inject_frame") is not None]
    rep = {"mode": "model", "backend": cfg["model"]["backend"], "checkpoint": ckpt, "metrics": m,
           "metrics_all": M.aggregate(per), "gates": M.check_gates(m, gates), **_summarize_group(per, by_id),
           "injection": {"calls": len(calls), "injected": len(waits),
                         "mean_wait_frames_after_ready": (sum(waits) / len(waits)) if waits else None,
                         "max_wait_frames_after_ready": max(waits) if waits else None},
           "attempts": summarize_diagnostics(diag), "per_episode_diagnostics": diag,
           "decoding": {k: gcfg.get(k) for k in ("temperature_text", "top_k_text", "seed_offset", "elle_audio")},
           "listen": {"files": listened, "note": listen_note}, "samples": samples, "per_episode": per}
    fpath = os.path.join(resolve_path(cfg["run_dir"], base), "plain_forgetting.json")
    if os.path.exists(fpath):
        with open(fpath) as f:
            rep["plain_forgetting"] = json.load(f)
    return rep


def episode_diagnostics(ex: dict, out: dict, tok, window: int = 10) -> Dict[str, Any]:
    """What the model did around each place the reference has a call: the
    highest p(<tool>) it gave within +-window frames (from the frame stats the
    generator records before sampling), whether it emitted <tool> there, and
    the first frame after the prompt where its text left the reference."""
    from textvocab import TOOL_OPEN

    tool = tok.special_ids[TOOL_OPEN]
    P = int(ex["meta"].get("prompt_frames", 0))
    ref = [int(t) for t in ex["text"]]
    gen = [int(t) for t in out["text"]]
    stats = out.get("frame_stats") or []
    neg = ex.get("neg_mask")
    # A call the model was trained NOT to make (an unsafe write in a negative
    # episode, neg_mask set) is not a place where <tool> is expected.
    ref_calls = [f for f, t in enumerate(ref) if t == tool and f >= P and not (neg is not None and neg[f])]
    near = []
    for f in ref_calls:
        lo, hi = max(P, f - window), min(len(gen), f + window + 1)
        ps = [stats[g]["p_tool"] for g in range(lo, hi)
              if g < len(stats) and stats[g] and math.isfinite(stats[g]["p_tool"])]
        near.append({"ref_frame": f, "max_p_tool": max(ps) if ps else None,
                     "emitted": any(gen[g] == tool for g in range(lo, hi))})
    nonfinite = sum(1 for st in stats if st and not math.isfinite(st["p_tool"]))
    free = [st["p_tool"] for st in stats[P:] if st and math.isfinite(st["p_tool"])]
    first_div = next((f for f in range(P, min(len(ref), len(gen))) if ref[f] != gen[f]), None)
    return {"reference_calls": len(ref_calls), "tool_opens": out.get("tool_opens", 0),
            "calls": len(out.get("calls", [])), "tool_errors": len(out.get("tool_errors", [])),
            "stray_closes": out.get("stray_closes", 0),
            "error_samples": [{"error": e.get("error"), "text": (e.get("text") or "")[:200]}
                              for e in out.get("tool_errors", [])[:3]],
            "max_p_tool": max(free) if free else None, "near_reference_calls": near,
            "first_divergence_frame": first_div, "prompt_frames": P,
            "nonfinite_frames": nonfinite, "frames": len(stats)}


def summarize_diagnostics(diag: List[dict]) -> Dict[str, Any]:
    near = [n for d in diag for n in d["near_reference_calls"]]
    ps = [n["max_p_tool"] for n in near if n["max_p_tool"] is not None]
    div = [d["first_divergence_frame"] - d["prompt_frames"] for d in diag if d["first_divergence_frame"] is not None]
    return {
        "reference_calls": sum(d["reference_calls"] for d in diag),
        "tool_opens": sum(d["tool_opens"] for d in diag),
        "calls_parsed": sum(d["calls"] for d in diag),
        "tool_errors": sum(d["tool_errors"] for d in diag),
        "stray_closes": sum(d.get("stray_closes", 0) for d in diag),
        "emitted_near_reference_call": sum(1 for n in near if n["emitted"]),
        "near_reference_p_tool_mean": (sum(ps) / len(ps)) if ps else None,
        "near_reference_p_tool_over_0.5": sum(1 for p in ps if p > 0.5),
        "near_reference_p_tool_over_0.1": sum(1 for p in ps if p > 0.1),
        "probed_reference_calls": len(ps),
        "median_frames_to_first_divergence": sorted(div)[len(div) // 2] if div else None,
        # NaN/inf text probabilities: never produced by a healthy model; points at the GPU or numerics.
        "nonfinite_frames": sum(d.get("nonfinite_frames", 0) for d in diag),
        "frames": sum(d.get("frames", 0) for d in diag),
    }


# --- output ----------------------------------------------------------------------

def _fmt(v: Any) -> str:
    return "n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def markdown(rep: Dict[str, Any]) -> str:
    lines = [f"# Evaluation ({rep['mode']})", ""]
    if rep["mode"] == "reference":
        gt = rep["ground_truth"]
        lines += ["## Ground truth (positives, labelled claims)", "", "| metric | value |", "|---|---|"]
        lines += [f"| {k} | {_fmt(gt['metrics'].get(k))} |" for k in GATED]
        lines += ["", f"- positives: {gt['positives']['n']}, flagged (should be 0): {len(gt['positives']['flagged'])}"
                  f" {gt['positives']['flagged'][:10]}",
                  f"- negatives: {gt['negatives']['n']}, caught: {gt['negatives']['caught']}"
                  f" (by flag: {gt['negatives']['by_flag']}), uncaught: {gt['negatives']['uncaught'][:10]}"]
        d = rep["claim_detector"]
        lines += ["", f"## Claim detector vs labels", "",
                  f"precision {_fmt(d['precision'])}, recall {_fmt(d['recall'])} (tp {d['tp']}, fp {d['fp']}, fn {d['fn']})",
                  "", f"Readback/yes heuristic vs labels.write_gates: agreement {_fmt(rep['readback_detector']['agreement'])}"
                  f" ({rep['readback_detector']['disagree']} disagreements)",
                  "", f"Negatives caught by the metric matching their negative_kind: {gt['negatives_by_kind']}"]
        if "round_trip" in rep:
            rt = rep["round_trip"]
            lines += ["", "## Round trip (prepared stream -> decoded events -> detector)", "", "| metric | value |", "|---|---|"]
            lines += [f"| {k} | {_fmt(rt['metrics'].get(k))} |" for k in GATED]
            lines += ["", f"- positives flagged: {len(rt['positives']['flagged'])} {rt['positives']['flagged'][:10]}",
                      f"- negatives caught: {rt['negatives']['caught']}/{rt['negatives']['n']}"]
    else:
        lines += [f"checkpoint: `{rep['checkpoint']}`", "", "| metric | value | gate |", "|---|---|---|"]
        for k in GATED:
            g = rep["gates"].get(k, {})
            if not g:
                gate = ""
            elif g.get("value") is None:
                gate = "no data"
            else:
                gate = ("pass" if g["pass"] else "FAIL") + f" ({'<=' if 'max' in g else '>='} {g.get('max', g.get('min'))})"
            lines.append(f"| {k} | {_fmt(rep['metrics'].get(k))} | {gate} |")
        inj = rep.get("injection", {})
        lines += ["", f"- tool calls made: {inj.get('calls')}, results injected: {inj.get('injected')},"
                  f" mean frames from result ready to injection: {_fmt(inj.get('mean_wait_frames_after_ready'))}"]
        at = rep.get("attempts")
        if at:
            lines += [f"- decoding: {rep.get('decoding')}",
                      f"- call attempts: reference calls {at['reference_calls']}, <tool> emitted {at['tool_opens']},"
                      f" parsed calls {at['calls_parsed']}, failed/unclosed calls {at['tool_errors']},"
                      f" stray </tool> {at['stray_closes']},"
                      f" emitted within 10 frames of a reference call {at['emitted_near_reference_call']}",
                      f"- p(<tool>) near reference calls: mean {_fmt(at['near_reference_p_tool_mean'])},"
                      f" >0.5 at {at['near_reference_p_tool_over_0.5']}/{at['probed_reference_calls']},"
                      f" >0.1 at {at['near_reference_p_tool_over_0.1']}/{at['probed_reference_calls']}",
                      f"- median frames after the prompt before the text first leaves the reference:"
                      f" {at['median_frames_to_first_divergence']}",
                      f"- frames with NaN/inf probabilities: {at.get('nonfinite_frames')} of {at.get('frames')}"
                      + ("  <-- NUMBERS UNRELIABLE (GPU or numerics fault)" if at.get("nonfinite_frames") else "")]
        pf = rep.get("plain_forgetting")
        if pf:
            g = pf.get("gate") or {}
            lines.append(f"- plain-conversation text loss (held out): {pf['plain_text_loss']:.4f} vs base "
                         f"{pf['plain_text_loss_base']:.4f}, delta {pf['plain_text_loss_delta']:+.4f}"
                         + (f" ({'pass' if g.get('pass') else 'FAIL'}, max {g.get('max')})" if g else "")
                         + f" at step {pf['step']}")
        else:
            lines.append("- plain-conversation forgetting: not measured (no data.plain_prepared_dir)")
        if rep.get("listen", {}).get("note"):
            lines.append(f"- {rep['listen']['note']}")
        for pth in rep.get("listen", {}).get("files", []):
            lines.append(f"- listen: `{pth}`")
        lines += ["", f"- episodes scored: {rep['metrics']['episodes']} positives"
                  f" (+{rep['negatives']['n']} negatives replayed, reported in metrics_all only)",
                  f"- positives with any flag: {len(rep['positives']['flagged'])}"]
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["reference", "model"], default="reference")
    ap.add_argument("--episodes", help="episodes JSONL (default: the prepared, retimed episodes)")
    ap.add_argument("--checkpoint")
    ap.add_argument("--split", help="override eval.split ('all' for every split)")
    ap.add_argument("--out", help="output directory (default: <run_dir>/eval)")
    ap.add_argument("--listen", type=int, default=0,
                    help="model mode: save Elle's generated audio for the first N episodes as WAV "
                         "(decoded with Mimi; a no-op on the CPU/tiny backend)")
    ap.add_argument("--max-episodes", type=int, default=0, help="model mode: override eval.max_episodes")
    ap.add_argument("--temperature-text", type=float, help="model mode: text sampling temperature (0 = greedy)")
    ap.add_argument("--top-k-text", type=int, help="model mode: text top-k")
    ap.add_argument("--seed-offset", type=int, help="model mode: added to each episode's seed")
    ap.add_argument("--elle-audio", choices=["generate", "teacher"],
                    help="model mode: teacher = force Elle's recorded audio (diagnostic: separates text from audio drift)")
    ap.add_argument("--tag", help="model mode: write into <out>/<tag>/ (several decoding settings side by side)")
    args = ap.parse_args(argv)
    cfg, base = load_config(args.config)
    if args.max_episodes:
        cfg.setdefault("eval", {})["max_episodes"] = args.max_episodes
    gen_over = {"temperature_text": args.temperature_text, "top_k_text": args.top_k_text,
                "seed_offset": args.seed_offset, "elle_audio": args.elle_audio}
    cfg.setdefault("eval", {}).setdefault("gen", {}).update({k: v for k, v in gen_over.items() if v is not None})
    prepared_dir = resolve_path(cfg["data"]["prepared_dir"], base)
    eps_path = args.episodes or os.path.join(prepared_dir, "episodes.jsonl")
    if not os.path.exists(eps_path):
        eps_path = resolve_path(cfg["data"]["episodes"], base)
    eps = load_jsonl(eps_path)
    split = args.split or cfg.get("eval", {}).get("split", "val")
    if args.mode == "model" and split != "all":
        eps = [e for e in eps if e.get("split", "train") == split]
    prepared = None
    if os.path.exists(os.path.join(prepared_dir, "manifest.jsonl")):
        from prepare import load_examples

        prepared = load_examples(prepared_dir)
    gates = cfg.get("eval", {}).get("gates", {})
    tok = build_tokenizer(cfg.get("tokenizer", {}), base) if prepared is not None or args.mode == "model" else None
    out_dir = args.out or os.path.join(resolve_path(cfg["run_dir"], base), "eval")
    if args.tag:
        out_dir = os.path.join(out_dir, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    if args.mode == "reference":
        rep = reference_report(eps, prepared, tok, gates)
    else:
        rep = model_report(cfg, base, eps, prepared or [], gates, args.checkpoint,
                           args.listen or int(cfg.get("eval", {}).get("listen", 0)), out_dir)
    with open(os.path.join(out_dir, f"{args.mode}_report.json"), "w") as f:
        json.dump(rep, f, indent=2, default=str)
    md = markdown(rep)
    with open(os.path.join(out_dir, f"{args.mode}_summary.md"), "w") as f:
        f.write(md)
    print(md)
    print(f"wrote {out_dir}/{args.mode}_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
