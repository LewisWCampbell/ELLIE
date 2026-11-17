#!/usr/bin/env python3
"""Validate episodes against training/CONTRACT.md.

    python3 training/data/validate.py training/data/out/sample_*.jsonl [--holdout-family slot_conflict]

Exit status 0 when every episode passes, 1 otherwise. Standard library only and
independent of the generator, so the evaluator can import it:

    from validate import validate_episode, violations
    issues = validate_episode(ep)          # [Issue(code, message), ...]
    kinds = violations(ep)                 # {"premature_claim", "write_without_intent", "dead_air"}

Contract v3: five tools, none of which takes a name, an email or free text;
`context.caller_on_file`; no readback. A booking is gated by the caller's
booking intent, a get_caller_details result and a FREE check (`_booking_gate`).

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
READ_TOOLS = {"check_availability", "get_my_bookings", "get_caller_details"}
WRITE_TOOLS = {"book_intro_call", "record_sms_consent"}
TOOLS = READ_TOOLS | WRITE_TOOLS
# Exact argument sets (contract v3): a start for the two calendar tools, nothing else anywhere.
TOOL_ARGS = {"check_availability": {"start"}, "get_my_bookings": set(), "get_caller_details": set(),
             "book_intro_call": {"start"}, "record_sms_consent": set()}
# The model never writes these; the system attaches them.
FORBIDDEN_ARGS = {"name", "email", "topic", "category", "fact"}
SPLITS = {"train", "val", "test"}
MARKERS = ("<tool>", "</tool>", "<result id=", "</result>")
NEGATIVE_KINDS = {"premature_claim", "write_without_intent", "dead_air"}
DEAD_AIR_LATENCY_S = 0.7     # contract metric 5 threshold
DEAD_AIR_MAX_SILENCE_S = 2.0  # longest silence tolerated inside a slow read's wait
TIME_EPS = 0.011
MAX_LEAD_DAYS = 365
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
ZONE_WORDS = ("pacific", "pacífico", "pst", "pdt")
TENTATIVE_WORDS = ("tentative", "pending", "penciled", "provisional", "pendiente")
INTENT_ACTS = ("accept", "pick")                      # plus any caller event with book_intent: true
VOID_ACTS = ("decline", "mind_change", "correction")  # a later one of these voids the intent
DETAILS_RE = re.compile(r"^(ON FILE|HEARD): (.+), ([^,]+)\.$")
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


def spoken_email(email: str, lang: str = "en") -> str:
    """Same as episode.spoken_email (this file stays independent of the generator)."""
    dot, at = (" punto ", " arroba ") if lang == "es" else (" dot ", " at ")
    local, _, dom = email.partition("@")
    return local.replace(".", dot) + at + dom.replace(".", dot)


def parse_details(content: str) -> Optional[Tuple[str, str, str]]:
    """(status, name, spoken email) of an ON FILE / HEARD result, else None (MISSING)."""
    m = DETAILS_RE.match(content)
    return (m.group(1), m.group(2), m.group(3)) if m else None


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
    ctx = ep.get("context")
    if ctx is not None:
        cof = ctx.get("caller_on_file") if isinstance(ctx, dict) else 0
        ok = cof is None or (isinstance(cof, dict) and set(cof) == {"name", "email"}
                             and isinstance(cof["name"], str) and cof["name"].strip()
                             and isinstance(cof["email"], str) and EMAIL_RE.match(cof["email"]))
        if not ok:
            out.append(Issue("E_SCHEMA", "context.caller_on_file must be null or {name, email} with a valid email"))
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
    keys = set(args)
    if keys & FORBIDDEN_ARGS:
        out.append(Issue("E_ARG_FREE_TEXT", f"{name} carries {sorted(keys & FORBIDDEN_ARGS)}: the model never "
                                            f"writes a name, an email or free text"))
    if name == "save_user_fact":
        return out + [Issue("E_SAVE_USER_FACT", "save_user_fact is not a voice tool in contract v3")]
    if name not in TOOLS:
        return out + [Issue("E_UNKNOWN_TOOL", f"unknown tool {name!r}")]
    req = TOOL_ARGS[name]
    if keys != req:
        out.append(Issue("E_ARGS_SCHEMA", f"{name} args {sorted(keys)} != {sorted(req)}"))
    for k in keys & req:
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


def _is_intent(ev: Dict[str, Any]) -> bool:
    """A caller event that asks to book a time or accepts one (contract v3, 5c)."""
    return _speech(ev) and ev["ch"] == "caller" and (ev.get("act") in INTENT_ACTS or ev.get("book_intent") is True)


def _same_start(a: Any, b: Any) -> bool:
    da, _ = parse_start(a)
    db, _ = parse_start(b)
    return da is not None and da == db


def _last_check(ep: Dict[str, Any], start: str, before: int) -> Optional[Tuple[int, str]]:
    """(result index, content) of the most recent check_availability for `start`
    whose result is in before event `before`."""
    evs, calls, results = _index(ep)
    best = None
    for cid, (_, c) in calls.items():
        if c["name"] == "check_availability" and _same_start(c["args"].get("start"), start) and cid in results:
            ri, res = results[cid]
            if ri < before and (best is None or ri > best[0]):
                best = (ri, res["content"])
    return best


def _booking_gate(ep: Dict[str, Any], cid: str, gate: Dict[str, Any]) -> List[str]:
    """Why a book_intro_call would not pass the booking guard. All of:

    1. details: `details_call_id` is a get_caller_details call whose result is
       in before the booking and starts with ON FILE or HEARD, and it is the
       latest details result before the booking (a stale one does not count).
    2. intent: `intent_index` is a caller event before the booking with act
       accept or pick, or with book_intent true, whose refers.start is the
       booked start. An accept must come after that time was put to the caller:
       after a check_availability call for it, or after an Elle line whose
       refers.start is that time. No caller decline, mind_change or correction
       lies between the intent and the booking.
    3. availability: the most recent check_availability result for that start
       before the booking is FREE. No exception: after UNKNOWN or BUSY, or with
       no check of that start at all, nothing may be booked.
    4. repeat: if the same start was already booked in this call (a Done or
       Updated result), this is an update: the intent event must be a caller
       correction made after that earlier result, and this call's result must
       start with "Updated:". A first booking must not return "Updated:".
    """
    evs, calls, results = _index(ep)
    ci, call = calls[cid]
    start = call["args"].get("start")
    probs: List[str] = []

    # 1. details
    did = gate.get("details_call_id")
    if did not in calls or calls[did][1]["name"] != "get_caller_details" or did not in results:
        probs.append(f"details_call_id {did!r} is not an answered get_caller_details call")
    else:
        ri, res = results[did]
        if ri > ci:
            probs.append(f"details result {did} arrives after the booking")
        elif parse_details(res["content"]) is None:
            probs.append(f"details result {did} is not ON FILE or HEARD: {res['content']!r}")
        later = [c for c, (i, c_ev) in calls.items() if c_ev["name"] == "get_caller_details"
                 and c in results and ri < results[c][0] < ci]
        if later:
            probs.append(f"details call {did} is not the latest before the booking ({later[-1]} is)")

    # Earlier successful bookings of the same start (rule 4).
    earlier = [results[c][0] for c, (i, c_ev) in calls.items()
               if i < ci and c_ev["name"] == "book_intro_call" and _same_start(c_ev["args"].get("start"), start)
               and c in results and results[c][1]["content"].startswith(("Done:", "Updated:"))]
    updated = cid in results and results[cid][1]["content"].startswith("Updated:")

    # 2. intent
    ii = gate.get("intent_index")
    if not (isinstance(ii, int) and 0 <= ii < ci and _speech(evs[ii]) and evs[ii]["ch"] == "caller"):
        return probs + [f"intent_index {ii!r} is not a caller event before the booking"]
    intent = evs[ii]
    if earlier:
        if intent.get("act") != "correction" or ii < max(earlier):
            probs.append("a repeat booking of the same start needs a caller correction after the first one")
        if not updated:
            probs.append("a repeat booking of the same start must return 'Updated:'")
    else:
        if updated:
            probs.append("'Updated:' result but this start was not booked before in this call")
        if not _is_intent(intent):
            probs.append(f"intent event {ii} is not an accept, a pick or a book_intent ask")
        elif not _same_start((intent.get("refers") or {}).get("start"), start):
            probs.append(f"intent event {ii} is not about the booked start {start}")
        elif intent.get("act") == "accept" and not intent.get("book_intent"):
            put = any((e["type"] == "tool_call" and e["name"] == "check_availability"
                       and _same_start(e["args"].get("start"), start))
                      or (_speech(e) and e["ch"] == "elle" and _same_start((e.get("refers") or {}).get("start"), start))
                      for e in evs[:ii])
            if not put:
                probs.append(f"caller accept at event {ii} comes before the time was put to them")
    for j in range(ii + 1, ci):
        e = evs[j]
        if _speech(e) and e["ch"] == "caller" and e.get("act") in VOID_ACTS:
            probs.append(f"caller {e['act']} at event {j} between the intent and the booking")

    # 3. availability
    last = _last_check(ep, start, ci)
    if last is None:
        probs.append(f"no check_availability result for {start} before the booking")
    elif not last[1].startswith("FREE"):
        probs.append(f"last check for {start} before the booking is not FREE: {last[1][:40]!r}")
    return probs


def _consent_gate(ep: Dict[str, Any], cid: str, gate: Dict[str, Any]) -> List[str]:
    """record_sms_consent: after the consent question (an Elle sms_offer) and a caller yes."""
    evs, calls, _ = _index(ep)
    ci, call = calls[cid]
    qi, cfi = gate.get("question_index"), gate.get("confirm_index")
    if not (isinstance(qi, int) and isinstance(cfi, int) and 0 <= qi < len(evs) and 0 <= cfi < len(evs)):
        return ["gate indices out of range"]
    q, cf = evs[qi], evs[cfi]
    probs = []
    if not (_speech(q) and q["ch"] == "elle" and q.get("act") == "sms_offer"):
        probs.append(f"gate question_index {qi} is not an Elle sms_offer")
    if not (_speech(cf) and cf["ch"] == "caller" and cf.get("act") == "confirm"):
        probs.append(f"gate confirm_index {cfi} is not a caller confirm")
    if probs:
        return probs
    if not q["t"] < cf["t"]:
        probs.append("the yes does not follow the consent question")
    if _end(cf) > call["t"] + 0.05:
        probs.append(f"write {cid} at t={call['t']} before the caller finished saying yes (ends {_end(cf):.2f})")
    for j in range(cfi + 1, ci):
        e = evs[j]
        if _speech(e) and e["ch"] == "caller" and e.get("act") in VOID_ACTS:
            probs.append(f"caller {e['act']} at event {j} between the yes and the write")
    return probs


def _ungated(ep: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """(missing-gate writes, invalid-gate problems). Every write needs a gate."""
    evs, calls, _ = _index(ep)
    gates = {g.get("call_id"): g for g in ep["labels"].get("write_gates", []) if isinstance(g, dict)}
    missing, invalid = [], []
    for cid, (_, call) in calls.items():
        if call["name"] not in WRITE_TOOLS:
            continue
        if cid not in gates:
            what = "caller intent + details" if call["name"] == "book_intro_call" else "consent question + yes"
            missing.append(f"{call['name']} {cid} has no gate ({what})")
        elif call["name"] == "book_intro_call":
            invalid += [f"{cid}: {p}" for p in _booking_gate(ep, cid, gates[cid])]
        else:
            invalid += [f"{cid}: {p}" for p in _consent_gate(ep, cid, gates[cid])]
    return missing, invalid


def _said(ev: Dict[str, Any]) -> str:
    """What Elle meant to say: the whole line even when a barge-in cut it short."""
    return (ev.get("full_text") or ev["text"]).lower()


def _v3_checks(ep: Dict[str, Any]) -> List[Issue]:
    """Contract v3 rules that are not gates: no readback, a caller on file is
    never asked who they are, and what Elle says after a booking comes from results."""
    evs, calls, results = _index(ep)
    out: List[Issue] = []
    on_file = (ep.get("context") or {}).get("caller_on_file")
    lang_of = lambda e: e.get("lang", "en")  # noqa: E731
    for i, ev in enumerate(evs):
        if not (_speech(ev) and ev["ch"] == "elle"):
            continue
        if ev.get("act") == "readback":
            out.append(Issue("E_READBACK", f"event {i}: there is no readback before a booking in contract v3"))
        if on_file and ev.get("act") in ("ask_name", "ask_email"):
            out.append(Issue("E_ON_FILE_ASK", f"event {i}: Elle asks a caller on file for their {ev['act'][4:]}"))
    deps: Dict[int, Set[str]] = defaultdict(set)
    for cs in ep["labels"].get("claim_spans", []):
        deps[cs["event_index"]].add(cs["depends_on"])
    for cid, (ci, call) in calls.items():
        if call["name"] != "book_intro_call" or cid not in results:
            continue
        content = results[cid][1]["content"]
        if not content.startswith(("Done:", "Updated:")):
            continue
        said = [(i, e) for i, e in enumerate(evs) if i > results[cid][0] and _speech(e) and e["ch"] == "elle"
                and e.get("act") in ("booked", "booked_update") and cid in deps.get(i, ())]
        if not said:
            out.append(Issue("E_BOOKED_CLAIM", f"{cid}: no post-booking sentence (act booked) depends on this booking"))
            continue
        i, ev = said[0]
        ref = ev.get("refers") or {}
        text = _said(ev)
        did = ref.get("details")
        det = parse_details(results[did][1]["content"]) if did in results and did in calls \
            and calls[did][1]["name"] == "get_caller_details" else None
        if det is None or did not in deps[i] or results[did][0] > ci:
            out.append(Issue("E_BOOKED_CLAIM", f"event {i}: refers.details {did!r} is not an ON FILE/HEARD result "
                                               f"before the booking that this sentence depends on"))
        elif not isinstance(ref.get("email"), str) or spoken_email(ref["email"]) != det[2]:
            out.append(Issue("E_BOOKED_CLAIM", f"event {i}: refers.email is not the email of details result {did}"))
        if not _same_start(ref.get("start"), call["args"].get("start")):
            out.append(Issue("E_BOOKED_CLAIM", f"event {i}: refers.start {ref.get('start')} != booked start"))
        if isinstance(ref.get("email"), str) and spoken_email(ref["email"], lang_of(ev)).lower() not in text:
            out.append(Issue("E_BOOKED_CLAIM", f"event {i}: does not say where the invite goes (spoken email)"))
        if not any(w in text for w in TENTATIVE_WORDS):
            out.append(Issue("E_BOOKED_CLAIM", f"event {i}: does not say the booking is tentative"))
        if content.startswith("Done:") and not any(w in text for w in ZONE_WORDS):
            out.append(Issue("E_BOOKED_CLAIM", f"event {i}: does not say the time with the timezone"))
    return out


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
        v.add("write_without_intent")
    if _dead_air(ep):
        v.add("dead_air")
    return v


def _names_check(ep: Dict[str, Any]) -> List[Issue]:
    """Elle says a caller's name only if it is on file, the caller has said it in
    this call, or a get_caller_details result has given it to her. Names she
    could otherwise only know from a bookings lookup or from memory are checked."""
    names: Set[str] = set()
    on_file = (ep.get("context") or {}).get("caller_on_file") or {}
    allowed: Set[str] = set(re.findall(r"\w+", on_file.get("name", "")))
    evs, calls, _ = _index(ep)
    for ev in evs:
        if ev["type"] == "tool_result":
            for m in BOOKING_ROW_RE.findall(ev["content"]):
                names.update(w for w in re.findall(r"\w+", m) if len(w) > 1)
            det = parse_details(ev["content"])
            if det and ev["id"] in calls and calls[ev["id"]][1]["name"] == "get_caller_details":
                names.update(w for w in re.findall(r"\w+", det[1]) if len(w) > 1)
    out = []
    for i, ev in enumerate(evs):
        if ev["type"] == "tool_result" and ev["id"] in calls and calls[ev["id"]][1]["name"] == "get_caller_details":
            det = parse_details(ev["content"])
            if det:
                allowed |= set(re.findall(r"\w+", det[1]))
        if not _speech(ev):
            continue
        words = set(re.findall(r"\w+", ev["text"]))
        if ev["ch"] == "caller":
            allowed |= words
        else:
            for n in (words & names) - allowed:
                out.append(Issue("E_NAME_SOURCE", f"event {i}: Elle says {n!r} before the caller or a result gave it"))
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

    # Silent writes, names.
    for i, ev in enumerate(evs):
        if _speech(ev) and ev["ch"] == "elle" and any(m in ev["text"].lower() for m in NARRATION_MARKERS):
            out.append(Issue("E_NARRATED_SAVE", f"event {i}: Elle narrates bookkeeping: {ev['text']!r}"))
    out += _names_check(ep)
    out += _v3_checks(ep)

    # Behaviour vs. negative labels.
    missing, invalid = _ungated(ep)
    out += [Issue("E_GATE_INVALID", m) for m in invalid]
    found = {"premature_claim": _premature(ep), "write_without_intent": missing, "dead_air": _dead_air(ep)}
    code = {"premature_claim": "E_PREMATURE_CLAIM", "write_without_intent": "E_UNGATED_WRITE",
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
