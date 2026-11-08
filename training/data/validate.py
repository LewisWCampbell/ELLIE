#!/usr/bin/env python3
"""Validate episodes against training/CONTRACT.md.

    python3 training/data/validate.py training/data/out/sample_*.jsonl [--holdout-family slot_conflict]

Exit status 0 when every episode passes, 1 otherwise. Standard library only and
independent of the generator, so the evaluator can import it:

    from validate import validate_episode, violations
    issues = validate_episode(ep)          # [Issue(code, message), ...]
    kinds = violations(ep)                 # {"premature_claim", "write_without_readback", "dead_air"}

Behavioural violations (premature claims, ungated writes, dead air) are errors
in positive episodes. In a labelled negative (`labels.negative: true`, scenario
name containing "neg") the declared `labels.negative_kind` must actually be
present, and no other kind may be.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple
from zoneinfo import ZoneInfo

TZ = ZoneInfo("America/Los_Angeles")
READ_TOOLS = {"check_availability", "get_my_bookings"}
WRITE_TOOLS = {"book_intro_call", "save_user_fact"}
TOOLS = READ_TOOLS | WRITE_TOOLS
SPLITS = {"train", "val", "test"}
MARKERS = ("<tool>", "</tool>", "<result id=", "</result>")
NEGATIVE_KINDS = {"premature_claim", "write_without_readback", "dead_air"}
DEAD_AIR_LATENCY_S = 0.7     # contract metric 5 threshold
DEAD_AIR_MAX_SILENCE_S = 2.0  # longest silence tolerated inside a slow read's wait
TIME_EPS = 0.011
MAX_LEAD_DAYS = 365
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
ZONE_WORDS = ("pacific", "pacífico", "pst", "pdt")
NARRATION_MARKERS = (
    "make a note", "making a note", "noting that", "i'll note", "saving", "i've saved",
    "save that", "jot that", "write that down", "remember that for", "in my notes",
    "lo anoto", "lo guardo", "tomo nota",
)
BOOKING_ROW_RE = re.compile(r"^#\d+: (.+?) — ", re.M)


class Issue(NamedTuple):
    code: str
    message: str


# --- helpers ---------------------------------------------------------------------

def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def parse_start(s: Any) -> Tuple[Optional[datetime], Optional[str]]:
    """(instant, error_code). Requires an explicit offset that is correct for Los
    Angeles at that instant."""
    if not isinstance(s, str) or not s:
        return None, "E_ARG_START_PARSE"
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None, "E_ARG_START_PARSE"
    if dt.tzinfo is None or dt.utcoffset() is None:
        return None, "E_ARG_START_OFFSET"
    if dt.utcoffset() != dt.astimezone(TZ).utcoffset():
        return dt, "E_ARG_START_TZ"
    return dt, None


def _speech(ev: Dict[str, Any]) -> bool:
    return ev.get("type") == "speech"


def _end(ev: Dict[str, Any]) -> float:
    return ev["t"] + ev.get("dur", 0.0)


# --- schema --------------------------------------------------------------------------

def _schema(ep: Any) -> List[Issue]:
    out: List[Issue] = []
    if not isinstance(ep, dict):
        return [Issue("E_SCHEMA", "episode is not an object")]
    for k, typ in (("id", str), ("split", str), ("scenario", str), ("seed", int), ("now", str),
                   ("events", list), ("labels", dict)):
        if k not in ep:
            out.append(Issue("E_SCHEMA", f"missing top-level field {k!r}"))
        elif not isinstance(ep[k], typ) or (typ is int and isinstance(ep[k], bool)):
            out.append(Issue("E_SCHEMA", f"{k!r} should be {typ.__name__}"))
    if out:
        return out
    if not re.fullmatch(r"ep_\d{6,}", ep["id"]):
        out.append(Issue("E_SCHEMA", f"id {ep['id']!r} is not ep_NNNNNN"))
    if ep["split"] not in SPLITS:
        out.append(Issue("E_SCHEMA", f"split {ep['split']!r} not in {sorted(SPLITS)}"))
    now, err = parse_start(ep["now"])
    if now is None or err == "E_ARG_START_OFFSET":
        out.append(Issue("E_SCHEMA", f"now {ep['now']!r} must be ISO 8601 with offset"))
    lab = ep["labels"]
    for k, typ in (("expected_calls", list), ("claim_spans", list), ("forbidden_before_result", bool)):
        if not isinstance(lab.get(k), typ):
            out.append(Issue("E_SCHEMA", f"labels.{k} missing or not {typ.__name__}"))
    if "negative" in lab and not isinstance(lab["negative"], bool):
        out.append(Issue("E_SCHEMA", "labels.negative must be a bool"))
    for i, ev in enumerate(ep["events"]):
        out += _event_schema(i, ev)
    return out


def _event_schema(i: int, ev: Any) -> List[Issue]:
    bad = lambda m: [Issue("E_EVENT_SCHEMA", f"event {i}: {m}")]  # noqa: E731
    if not isinstance(ev, dict):
        return bad("not an object")
    if not _num(ev.get("t")) or ev["t"] < 0:
        return bad("t must be a non-negative number")
    typ = ev.get("type")
    if typ == "speech":
        if ev.get("ch") not in ("elle", "caller"):
            return bad("speech ch must be elle or caller")
        if not _num(ev.get("dur")) or ev["dur"] <= 0:
            return bad("speech needs dur > 0")
        if not isinstance(ev.get("text"), str) or not ev["text"].strip():
            return bad("speech needs non-empty text")
    elif typ == "tool_call":
        if ev.get("ch") != "elle":
            return bad("tool_call must be on ch elle")
        if not isinstance(ev.get("id"), str) or not isinstance(ev.get("name"), str):
            return bad("tool_call needs string id and name")
        if not isinstance(ev.get("args"), dict):
            return bad("tool_call args must be an object")
    elif typ == "tool_result":
        if ev.get("ch") != "system":
            return bad("tool_result must be on ch system")
        if not isinstance(ev.get("id"), str) or not isinstance(ev.get("content"), str):
            return bad("tool_result needs string id and content")
        if not _num(ev.get("latency_s")) or ev["latency_s"] <= 0:
            return bad("tool_result needs latency_s > 0")
    else:
        return bad(f"unknown type {typ!r}")
    return []


# --- args ------------------------------------------------------------------------------

def check_args(name: str, args: Dict[str, Any], now: datetime) -> List[Issue]:
    out: List[Issue] = []
    if name not in TOOLS:
        return [Issue("E_UNKNOWN_TOOL", f"unknown tool {name!r}")]
    allowed = {"check_availability": ({"start"}, set()),
               "get_my_bookings": (set(), set()),
               "book_intro_call": ({"name", "email", "start"}, {"topic"}),
               "save_user_fact": ({"category", "fact"}, set())}[name]
    req, opt = allowed
    keys = set(args)
    if not req <= keys or not keys <= req | opt:
        out.append(Issue("E_ARGS_SCHEMA", f"{name} args {sorted(keys)} != required {sorted(req)} + optional {sorted(opt)}"))
    for k in keys & (req | opt):
        if not isinstance(args[k], str) or not args[k].strip():
            out.append(Issue("E_ARGS_SCHEMA", f"{name}.{k} must be a non-empty string"))
    if "start" in args and isinstance(args["start"], str):
        dt, err = parse_start(args["start"])
        if err:
            out.append(Issue(err, f"{name}.start {args['start']!r}"))
        if dt is not None:
            if dt <= now:
                out.append(Issue("E_ARG_START_PAST", f"{name}.start {args['start']} is not after now {now.isoformat()}"))
            if dt > now + timedelta(days=MAX_LEAD_DAYS):
                out.append(Issue("E_ARG_START_FAR", f"{name}.start more than a year out"))
            loc = dt.astimezone(TZ)
            if loc.minute not in (0, 30) or loc.second or loc.microsecond:
                out.append(Issue("E_ARG_START_BOUNDARY", f"{name}.start {args['start']} not on a :00/:30 boundary"))
    if name == "book_intro_call" and isinstance(args.get("email"), str) and not EMAIL_RE.match(args["email"]):
        out.append(Issue("E_ARG_EMAIL", f"invalid email {args['email']!r}"))
    return out


# --- behaviour ----------------------------------------------------------------------------

def _index(ep: Dict[str, Any]):
    evs = ep["events"]
    calls = {ev["id"]: (i, ev) for i, ev in enumerate(evs) if ev["type"] == "tool_call"}
    results = {ev["id"]: (i, ev) for i, ev in enumerate(evs) if ev["type"] == "tool_result"}
    return evs, calls, results


def _premature(ep: Dict[str, Any]) -> List[str]:
    evs, calls, results = _index(ep)
    out = []
    for cs in ep["labels"].get("claim_spans", []):
        i, dep = cs.get("event_index"), cs.get("depends_on")
        if isinstance(i, int) and 0 <= i < len(evs) and dep in results:
            if evs[i]["t"] <= results[dep][1]["t"]:
                out.append(f"claim event {i} at t={evs[i]['t']} starts before result {dep} at t={results[dep][1]['t']}")
    return out


def _gate_problems(ep: Dict[str, Any], cid: str, gate: Dict[str, Any]) -> List[str]:
    evs, calls, _ = _index(ep)
    ci, call = calls[cid]
    probs = []
    rbi, cfi = gate.get("readback_index"), gate.get("confirm_index")
    if not (isinstance(rbi, int) and isinstance(cfi, int) and 0 <= rbi < len(evs) and 0 <= cfi < len(evs)):
        return ["gate indices out of range"]
    rb, cf = evs[rbi], evs[cfi]
    want_rb = "sms_offer" if call["name"] == "save_user_fact" else "readback"
    if not (_speech(rb) and rb["ch"] == "elle" and rb.get("act") == want_rb):
        probs.append(f"gate readback_index {rbi} is not an Elle {want_rb}")
    if not (_speech(cf) and cf["ch"] == "caller" and cf.get("act") == "confirm"):
        probs.append(f"gate confirm_index {cfi} is not a caller confirm")
    if probs:
        return probs
    if not rb["t"] < cf["t"]:
        probs.append("confirm does not follow the readback")
    if _end(cf) > call["t"] + 0.05:
        probs.append(f"write {cid} at t={call['t']} before the caller finished saying yes (ends {_end(cf):.2f})")
    for j in range(cfi + 1, ci):
        e = evs[j]
        if _speech(e) and e["ch"] == "caller" and e.get("act") in ("decline", "correction", "mind_change"):
            probs.append(f"caller {e['act']} at event {j} between the yes and the write")
    if call["name"] == "book_intro_call":
        later_rbs = [j for j in range(rbi + 1, ci) if _speech(evs[j]) and evs[j].get("act") == "readback"]
        if later_rbs:
            probs.append(f"gate readback {rbi} is not the last readback before the write")
        ref = rb.get("refers") or {}
        args = call["args"]
        a_dt, _ = parse_start(args.get("start"))
        r_dt, _ = parse_start(ref.get("start"))
        if a_dt is None or r_dt is None or a_dt != r_dt:
            probs.append(f"readback time {ref.get('start')} != booked start {args.get('start')}")
        if ref.get("email") != args.get("email") or ref.get("name") != args.get("name"):
            probs.append("readback name/email differ from the booked ones")
        if not any(w in rb["text"].lower() for w in ZONE_WORDS):
            probs.append("readback does not say the timezone")
    return probs


def _ungated(ep: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """(missing-gate writes, invalid-gate problems)."""
    evs, calls, _ = _index(ep)
    gates = {g.get("call_id"): g for g in ep["labels"].get("write_gates", []) if isinstance(g, dict)}
    missing, invalid = [], []
    for cid, (_, call) in calls.items():
        needs = call["name"] == "book_intro_call" or (
            call["name"] == "save_user_fact" and call["args"].get("category") == "sms_consent")
        if not needs:
            continue
        if cid not in gates:
            missing.append(f"{call['name']} {cid} has no readback+yes gate")
        else:
            invalid += [f"{cid}: {p}" for p in _gate_problems(ep, cid, gates[cid])]
    return missing, invalid


def _dead_air(ep: Dict[str, Any]) -> List[str]:
    evs, calls, results = _index(ep)
    speech = [e for e in evs if _speech(e)]
    out = []
    for cid, (_, call) in calls.items():
        if call["name"] not in READ_TOOLS or cid not in results:
            continue
        res = results[cid][1]
        if res["latency_s"] <= DEAD_AIR_LATENCY_S:
            continue
        a, b = call["t"], res["t"]
        if not any(e["ch"] == "elle" and e["t"] < b and _end(e) > a for e in speech):
            out.append(f"read {cid} waits {res['latency_s']}s with Elle silent")
            continue
        spans = sorted((max(a, e["t"]), min(b, _end(e))) for e in speech if e["t"] < b and _end(e) > a)
        cur, gap = a, 0.0
        for s0, s1 in spans:
            gap = max(gap, s0 - cur)
            cur = max(cur, s1)
        gap = max(gap, b - cur)
        if gap > DEAD_AIR_MAX_SILENCE_S:
            out.append(f"read {cid}: {gap:.2f}s of silence inside the wait")
    return out


def violations(ep: Dict[str, Any]) -> Set[str]:
    """Behavioural violation kinds present in a (schema-valid) episode."""
    v = set()
    if _premature(ep):
        v.add("premature_claim")
    missing, _ = _ungated(ep)
    if missing:
        v.add("write_without_readback")
    if _dead_air(ep):
        v.add("dead_air")
    return v


def _names_check(ep: Dict[str, Any]) -> List[Issue]:
    """Elle never uses a name from stored notes (or from a bookings lookup) until
    the caller has said it in this conversation."""
    names: Set[str] = set()
    ctx = ep.get("context") or {}
    for n in ctx.get("notes_names") or []:
        names.update(w for w in re.findall(r"\w+", n) if len(w) > 1)
    for ev in ep["events"]:
        if ev["type"] == "tool_result":
            for m in BOOKING_ROW_RE.findall(ev["content"]):
                names.update(w for w in re.findall(r"\w+", m) if len(w) > 1)
    out = []
    said: Set[str] = set()
    for i, ev in enumerate(ep["events"]):
        if not _speech(ev):
            continue
        words = set(re.findall(r"\w+", ev["text"]))
        if ev["ch"] == "caller":
            said |= words
        else:
            for n in (words & names) - said:
                out.append(Issue("E_NOTES_NAME", f"event {i}: Elle says {n!r} before the caller gave it"))
    return out


def validate_episode(ep: Any) -> List[Issue]:
    out = _schema(ep)
    if out:
        return out
    now, _ = parse_start(ep["now"])
    evs, lab = ep["events"], ep["labels"]

    # Order, overlap, markers.
    for i in range(1, len(evs)):
        if evs[i]["t"] < evs[i - 1]["t"]:
            out.append(Issue("E_ORDER", f"event {i} at t={evs[i]['t']} before event {i - 1} at t={evs[i - 1]['t']}"))
    last_end: Dict[str, float] = {}
    for i, ev in enumerate(evs):
        if _speech(ev):
            if ev["t"] < last_end.get(ev["ch"], -1.0) - 0.01:
                out.append(Issue("E_SELF_OVERLAP", f"event {i}: {ev['ch']} speech overlaps their own previous line"))
            last_end[ev["ch"]] = max(last_end.get(ev["ch"], 0.0), _end(ev))
        text = ev.get("text", "") + ev.get("content", "")
        if any(m in text for m in MARKERS):
            out.append(Issue("E_MARKER_IN_TEXT", f"event {i} contains a serialization marker"))

    # Calls and results.
    call_order = [ev for ev in evs if ev["type"] == "tool_call"]
    for k, ev in enumerate(call_order, 1):
        if ev["id"] != f"c{k}":
            out.append(Issue("E_CALL_ID", f"tool_call ids must be c1..cN in order; got {ev['id']!r} at position {k}"))
    ids = Counter(ev["id"] for ev in call_order)
    for cid, n in ids.items():
        if n > 1:
            out.append(Issue("E_CALL_ID", f"duplicate tool_call id {cid}"))
    calls = {ev["id"]: (i, ev) for i, ev in enumerate(evs) if ev["type"] == "tool_call"}
    seen_results: Set[str] = set()
    for i, ev in enumerate(evs):
        if ev["type"] != "tool_result":
            continue
        cid = ev["id"]
        if cid in seen_results:
            out.append(Issue("E_DUP_RESULT", f"second result for {cid}"))
        seen_results.add(cid)
        if cid not in calls:
            out.append(Issue("E_RESULT_ORPHAN", f"result {cid} has no tool_call"))
            continue
        ci, call = calls[cid]
        if i < ci or ev["t"] <= call["t"]:
            out.append(Issue("E_RESULT_BEFORE_CALL", f"result {cid} at t={ev['t']} not after its call at t={call['t']}"))
        if abs(call["t"] + ev["latency_s"] - ev["t"]) > TIME_EPS:
            out.append(Issue("E_LATENCY_MISMATCH", f"result {cid}: t != call.t + latency_s"))
    for cid in calls:
        if cid not in seen_results:
            out.append(Issue("E_RESULT_MISSING", f"tool_call {cid} has no result"))
    for _, ev in calls.values():
        out += [Issue(c, f"{ev['id']}: {m}") for c, m in check_args(ev["name"], ev["args"], now)]

    exp = [{"name": ev["name"], "args": ev["args"]} for ev in call_order]
    if lab["expected_calls"] != exp:
        out.append(Issue("E_EXPECTED_CALLS", "labels.expected_calls does not match the tool_call events"))

    # Claim spans.
    for cs in lab["claim_spans"]:
        i, dep = cs.get("event_index") if isinstance(cs, dict) else None, cs.get("depends_on") if isinstance(cs, dict) else None
        if not isinstance(i, int) or not 0 <= i < len(evs) or not _speech(evs[i]) or evs[i]["ch"] != "elle":
            out.append(Issue("E_CLAIM_INDEX", f"claim span {cs} does not point at Elle speech"))
        elif dep not in calls:
            out.append(Issue("E_CLAIM_INDEX", f"claim span {cs} depends on unknown call"))
    if lab["claim_spans"] and not lab["forbidden_before_result"]:
        out.append(Issue("E_SCHEMA", "labels.forbidden_before_result must be true when there are claim spans"))
    if any(o.code in ("E_CLAIM_INDEX", "E_RESULT_ORPHAN", "E_RESULT_MISSING", "E_CALL_ID") for o in out):
        return out

    # Silent saves, names.
    for i, ev in enumerate(evs):
        if _speech(ev) and ev["ch"] == "elle" and any(m in ev["text"].lower() for m in NARRATION_MARKERS):
            out.append(Issue("E_NARRATED_SAVE", f"event {i}: Elle narrates bookkeeping: {ev['text']!r}"))
    out += _names_check(ep)

    # Behaviour vs. negative labels.
    missing, invalid = _ungated(ep)
    out += [Issue("E_GATE_INVALID", m) for m in invalid]
    found = {"premature_claim": _premature(ep), "write_without_readback": missing, "dead_air": _dead_air(ep)}
    code = {"premature_claim": "E_PREMATURE_CLAIM", "write_without_readback": "E_UNGATED_WRITE",
            "dead_air": "E_DEAD_AIR"}
    negative = bool(lab.get("negative"))
    named_neg = "neg" in ep["scenario"]
    if negative != named_neg:
        out.append(Issue("E_NEGATIVE_LABEL", "labels.negative and a 'neg' scenario name must go together"))
    declared = lab.get("negative_kind")
    if negative:
        if declared not in NEGATIVE_KINDS:
            out.append(Issue("E_NEGATIVE_LABEL", f"negative episode needs negative_kind in {sorted(NEGATIVE_KINDS)}"))
        elif not found[declared]:
            out.append(Issue("E_NEGATIVE_LABEL", f"labelled {declared} but no such violation is present"))
    for kind, msgs in found.items():
        if negative and kind == declared:
            continue
        out += [Issue(code[kind], m) for m in msgs]
    return out


# --- files & splits --------------------------------------------------------------------

def validate_files(paths: Sequence[str], holdout_family: Optional[str] = None,
                   max_errors: int = 50) -> Tuple[int, List[str], Counter]:
    """Validate JSONL files. Returns (episode_count, error lines, error-code counts)."""
    errs: List[str] = []
    codes: Counter = Counter()
    seen_pairs: Dict[Tuple[str, int], str] = {}
    seen_ids: Set[str] = set()
    fam_by_split: Dict[str, Set[str]] = defaultdict(set)
    n = 0

    def err(code: str, msg: str) -> None:
        codes[code] += 1
        if len(errs) < max_errors:
            errs.append(f"{code}: {msg}")

    for path in paths:
        prev_num = -1
        file_split = None
        with open(path, encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                if not line.strip():
                    continue
                n += 1
                where = f"{path}:{ln}"
                try:
                    ep = json.loads(line)
                except json.JSONDecodeError as e:
                    err("E_JSON", f"{where}: {e}")
                    continue
                for iss in validate_episode(ep):
                    err(iss.code, f"{where} {ep.get('id') if isinstance(ep, dict) else ''} [{ep.get('scenario') if isinstance(ep, dict) else ''}]: {iss.message}")
                if not isinstance(ep, dict) or not isinstance(ep.get("id"), str):
                    continue
                m = re.fullmatch(r"ep_(\d+)", ep["id"])
                num = int(m.group(1)) if m else -1
                if num <= prev_num:
                    err("E_EPISODE_ID", f"{where}: id {ep['id']} not increasing within the file")
                prev_num = num
                if ep["id"] in seen_ids:
                    err("E_EPISODE_ID", f"{where}: duplicate id {ep['id']}")
                seen_ids.add(ep["id"])
                split = ep.get("split")
                if file_split is None:
                    file_split = split
                elif split != file_split:
                    err("E_SPLIT_LABEL", f"{where}: split {split} in a {file_split} file")
                pair = (ep.get("scenario"), ep.get("seed"))
                if pair in seen_pairs and seen_pairs[pair] != split:
                    err("E_SPLIT_OVERLAP", f"{where}: scenario/seed {pair} appears in both {seen_pairs[pair]} and {split}")
                elif pair in seen_pairs:
                    err("E_SPLIT_OVERLAP", f"{where}: scenario/seed {pair} duplicated")
                seen_pairs.setdefault(pair, split)
                fam = (ep.get("labels") or {}).get("family")
                if fam:
                    fam_by_split[split].add(fam)
    if holdout_family:
        for split in ("train", "val"):
            if holdout_family in fam_by_split.get(split, set()):
                err("E_HOLDOUT_LEAK", f"held-out family {holdout_family} appears in {split}")
    return n, errs, codes


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--holdout-family", default=None)
    ap.add_argument("--max-errors", type=int, default=50)
    a = ap.parse_args(argv)
    n, errs, codes = validate_files(a.files, a.holdout_family, a.max_errors)
    for e in errs:
        print(e)
    total = sum(codes.values())
    if total:
        print(f"\nFAIL: {total} issue(s) in {n} episodes: " + ", ".join(f"{k}={v}" for k, v in codes.most_common()))
        return 1
    print(f"OK: {n} episodes valid across {len(a.files)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
