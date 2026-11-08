"""Replace estimated speech durations with measured clip lengths, re-placing
every event so the episode's structure survives.

The generator placed each event relative to something: after a silence gap
following earlier speech or a tool event, partway into another utterance (a
backchannel, a caller talking over "let me check", a tool call emitted
mid-sentence), just before the end of an utterance it cut off (a barge-in), or
exactly `latency_s` after its tool call (a result). `apply_measured` keeps
those relations when the durations change:

  - tool_result:  t = new t of its call + latency_s (latencies are exact).
  - an event that starts inside another speaker's utterance (its "host", the
    latest-starting one containing it): the same fraction of the way through the
    host, or, if the host was cut off (`interrupted`), the same distance before
    the host's end, so Elle still stops just after the caller barges in.
  - anything else: the generator placed it a gap after the "floor", whatever
    ended last before it (a speech end, or a tool call/result point); that gap
    is kept. Other events still in progress at that moment must not newly
    overlap it (each keeps min(its old gap, 0.15 s)). Claims therefore stay
    after their results, and writes after the yes.
  - a speaker never overlaps their own previous line (pushed 20 ms after it).
  - in a labelled premature-claim negative, the premature line keeps its offset
    from its call, so it stays premature where the timing allows.

Events are then re-sorted by time with label indices remapped (and call ids
renumbered if two calls swapped). The result should still pass validate.py;
render_audio.py checks every episode and drops (and reports) any that do not.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

EPS = 1e-6
SELF_GAP = 0.02


def _end(ev: Dict[str, Any], t: float, dur: Optional[float] = None) -> float:
    return t + (dur if dur is not None else ev.get("dur", 0.0))


def apply_measured(ep: Dict[str, Any], new_dur: Dict[int, float]) -> Dict[str, Any]:
    """`new_dur` maps speech event index -> measured duration (s). Returns a new episode."""
    ep = copy.deepcopy(ep)
    evs: List[Dict[str, Any]] = ep["events"]
    n = len(evs)
    old_t = [float(e["t"]) for e in evs]
    old_end = [_end(e, old_t[i]) if e["type"] == "speech" else old_t[i] for i, e in enumerate(evs)]
    call_idx = {e["id"]: i for i, e in enumerate(evs) if e["type"] == "tool_call"}
    res_idx = {e["id"]: i for i, e in enumerate(evs) if e["type"] == "tool_result"}
    lab = ep.get("labels", {})
    premature = set()
    if lab.get("negative") and lab.get("negative_kind") == "premature_claim":
        for cs in lab.get("claim_spans", []):
            r = res_idx.get(cs["depends_on"])
            if r is not None and old_t[cs["event_index"]] <= old_t[r]:
                premature.add(cs["event_index"])
    claim_dep = {cs["event_index"]: cs["depends_on"] for cs in lab.get("claim_spans", [])}

    new_t = [0.0] * n
    new_end = [0.0] * n
    last_end: Dict[str, float] = {}
    for i, e in enumerate(evs):
        t0 = old_t[i]
        is_speech = e["type"] == "speech"
        dur = round(float(new_dur[i]), 3) if is_speech and i in new_dur else float(e.get("dur", 0.0))
        if e["type"] == "tool_result" and e["id"] in call_idx:
            t = new_t[call_idx[e["id"]]] + float(e["latency_s"])
        elif i in premature and claim_dep.get(i) in call_idx:
            c = call_idx[claim_dep[i]]
            t = new_t[c] + (t0 - old_t[c])
        else:
            host = None
            for j in range(i):
                ej = evs[j]
                if ej["type"] != "speech" or (is_speech and ej["ch"] == e["ch"]):
                    continue
                if old_t[j] <= t0 < old_end[j] - EPS and (host is None or old_t[j] >= old_t[host]):
                    host = j
            if host is not None:
                h = evs[host]
                if h.get("interrupted"):
                    t = max(new_t[host], new_end[host] - (old_end[host] - t0))
                else:
                    frac = (t0 - old_t[host]) / max(EPS, old_end[host] - old_t[host])
                    t = new_t[host] + frac * (new_end[host] - new_t[host])
            else:
                # Anchors are the frontier: earlier events that end (or happen) at or
                # after the latest earlier start. Anything older is already
                # accounted for through the events placed after it.
                latest = max(old_t[:i], default=0.0)
                front = [j for j in range(i) if latest - EPS <= old_end[j] <= t0 + EPS]
                if front:
                    # The generator placed this event a gap after the floor (whatever
                    # ended last); keep that gap. Other frontier events must not
                    # start overlapping it: keep min(their old gap, 0.15 s).
                    a = max(front, key=lambda j: (old_end[j], j))
                    t = new_end[a] + (t0 - old_end[a])
                    for j in front:
                        t = max(t, new_end[j] + min(t0 - old_end[j], 0.15))
                else:
                    t = t0
        if is_speech and e["ch"] in last_end:
            t = max(t, last_end[e["ch"]] + SELF_GAP)
        t = round(max(0.0, t), 3)
        new_t[i] = t
        if is_speech:
            e["dur"] = dur
            new_end[i] = t + dur
            last_end[e["ch"]] = new_end[i]
        else:
            new_end[i] = t
        e["t"] = t
    # Exact latencies after rounding.
    for cid, r in res_idx.items():
        if cid in call_idx:
            evs[r]["t"] = round(evs[call_idx[cid]]["t"] + float(evs[r]["latency_s"]), 3)
    _resort(ep)
    ep.setdefault("meta", {})["measured"] = True
    return ep


def _resort(ep: Dict[str, Any]) -> None:
    evs = ep["events"]
    order = sorted(range(len(evs)), key=lambda i: (evs[i]["t"], i))
    if order != list(range(len(evs))):
        pos = {old: new for new, old in enumerate(order)}
        ep["events"] = [evs[i] for i in order]
        lab = ep.get("labels", {})
        for c in lab.get("claim_spans", []):
            c["event_index"] = pos[c["event_index"]]
        lab.get("claim_spans", []).sort(key=lambda c: c["event_index"])
        for g in lab.get("write_gates", []):
            g["readback_index"] = pos[g["readback_index"]]
            g["confirm_index"] = pos[g["confirm_index"]]
    # Call ids must stay c1..cN in time order.
    remap: Dict[str, str] = {}
    for e in ep["events"]:
        if e["type"] == "tool_call":
            remap[e["id"]] = f"c{len(remap) + 1}"
    if any(k != v for k, v in remap.items()):
        for e in ep["events"]:
            if e["type"] in ("tool_call", "tool_result") and e["id"] in remap:
                e["id"] = remap[e["id"]]
        lab = ep.get("labels", {})
        for c in lab.get("claim_spans", []):
            c["depends_on"] = remap.get(c["depends_on"], c["depends_on"])
        for g in lab.get("write_gates", []):
            g["call_id"] = remap.get(g["call_id"], g["call_id"])
        calls = [e for e in ep["events"] if e["type"] == "tool_call"]
        lab["expected_calls"] = [{"name": c["name"], "args": c["args"]} for c in calls]
