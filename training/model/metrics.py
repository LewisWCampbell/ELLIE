"""The five CONTRACT.md metrics, computed over events.

Everything here works on the contract's event format, so the same code scores
(a) the dataset's own ground truth ("reference" mode) and (b) a model's output
decoded back into events (events.py). Pure Python, no torch.

Where the contract leaves a definition open, the choice is written next to the
code and in README.md ("Metric definitions").
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

READ_TOOLS = {"check_availability", "get_my_bookings"}
WRITE_TOOLS = {"book_intro_call", "save_user_fact"}
SILENT_TOOLS = {"save_user_fact"}
REQUIRED_ARGS = {
    "check_availability": ["start"],
    "get_my_bookings": [],
    "book_intro_call": ["name", "email", "start"],
    "save_user_fact": ["category", "fact"],
}
INSTANT_ARGS = {"start"}
DEAD_AIR_LATENCY_S = 0.7  # contract: reads whose latency exceeds 0.7 s
LONG_SILENCE_S = 2.0  # secondary, stricter wait metric (not in the contract)
# "Says nothing": Elle's speech covers less than this much of the wait. Not 0,
# because speech timing decoded from the text stream is approximate (+-0.3 s);
# ground-truth events behave identically with 0 or 0.3.
DEAD_AIR_MIN_COVER_S = 0.3
NARRATION_WINDOW_S = (-0.5, 4.0)  # speech this close to a silent write is "narrating" it

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# --- small helpers -----------------------------------------------------------

def parse_instant(s: Any) -> Optional[datetime]:
    if not isinstance(s, str):
        return None
    try:
        dt = datetime.fromisoformat(s.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None or dt.utcoffset() is None:
        return None  # contract: ISO 8601 *with offset*
    return dt


def _blank(v: Any) -> bool:
    return v is None or (isinstance(v, str) and not v.strip())


def args_exact_match(pred: Dict[str, Any], exp: Dict[str, Any]) -> bool:
    """Exact match on every key in either dict; `start` compared as an instant
    (so 18:00Z == 10:00-08:00). Absent and blank are equal (optional `topic`)."""
    pred, exp = pred or {}, exp or {}
    for k in set(pred) | set(exp):
        a, b = pred.get(k), exp.get(k)
        if _blank(a) and _blank(b):
            continue
        if k in INSTANT_ARGS:
            da, db = parse_instant(a), parse_instant(b)
            if da is None or db is None or da != db:
                return False
        elif isinstance(a, str) and isinstance(b, str):
            if a.strip() != b.strip():
                return False
        elif a != b:
            return False
    return True


def speech(events: Iterable[dict], ch: str) -> List[dict]:
    return sorted((e for e in events if e.get("type") == "speech" and e.get("ch") == ch), key=lambda e: e["t"])


def calls(events: Iterable[dict]) -> List[dict]:
    return sorted((e for e in events if e.get("type") == "tool_call"), key=lambda e: e["t"])


def results_by_id(events: Iterable[dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for e in events:
        if e.get("type") == "tool_result" and e.get("id") not in out:
            out[e.get("id")] = e
    return out


def _end(e: dict) -> float:
    return float(e["t"]) + float(e.get("dur") or 0.0)


def turns(events: Iterable[dict], ch: str = "elle", max_gap_s: float = 1.5) -> List[dict]:
    """Merge one channel's consecutive utterances (gap < max_gap_s, no other
    speaker starting in between) into turns. Utterances decoded from a text
    stream are split at pauses; a readback must be judged as one turn."""
    evs = list(events)
    other = [float(e["t"]) for e in evs if e.get("type") == "speech" and e.get("ch") != ch]
    out: List[dict] = []
    for e in speech(evs, ch):
        if out and float(e["t"]) - _end(out[-1]) < max_gap_s and not any(_end(out[-1]) <= o < float(e["t"]) for o in other):
            last = out[-1]
            last["text"] = (last["text"] + " " + e.get("text", "")).strip()
            last["dur"] = _end(e) - float(last["t"])
        else:
            out.append({"t": float(e["t"]), "dur": float(e.get("dur") or 0.0), "ch": ch, "type": "speech",
                        "text": e.get("text", "")})
    return out


# --- 1 & 2: tool selection and arguments -------------------------------------

def align_calls(pred: List[dict], expected: List[dict]) -> List[Tuple[Optional[dict], Optional[dict]]]:
    """Order-preserving alignment by tool name (LCS); unmatched sides are None."""
    n, m = len(pred), len(expected)
    L = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            L[i][j] = L[i + 1][j + 1] + 1 if pred[i].get("name") == expected[j].get("name") else max(L[i + 1][j], L[i][j + 1])
    out, i, j = [], 0, 0
    while i < n and j < m:
        if pred[i].get("name") == expected[j].get("name"):
            out.append((pred[i], expected[j])); i += 1; j += 1
        elif L[i + 1][j] >= L[i][j + 1]:
            out.append((pred[i], None)); i += 1
        else:
            out.append((None, expected[j])); j += 1
    out += [(p, None) for p in pred[i:]] + [(None, e) for e in expected[j:]]
    return out


def score_calls(pred_calls: List[dict], expected: List[dict]) -> Dict[str, Any]:
    pairs = align_calls(pred_calls, expected)
    matched = [(p, e) for p, e in pairs if p is not None and e is not None]
    return {
        # Episode-level selection: the exact sequence of tool names, where an
        # empty sequence on both sides is "correctly no tool".
        "selection_correct": [c.get("name") for c in pred_calls] == [c.get("name") for c in expected],
        "n_pred": len(pred_calls), "n_expected": len(expected), "n_matched": len(matched),
        "args_matched": sum(1 for p, e in matched if args_exact_match(p.get("args"), e.get("args"))),
        "arg_mismatches": [{"name": p.get("name"), "pred": p.get("args"), "expected": e.get("args")}
                           for p, e in matched if not args_exact_match(p.get("args"), e.get("args"))],
    }


# --- 3: premature claims --------------------------------------------------------

_HYPOTHETICAL = re.compile(r"\b(if|whether|let me|i'?ll|i will|going to|gonna|want me to|shall i|should i|"
                           r"si|déjame|voy a)\b[^.?!,]*", re.I)
CLAIM_PATTERNS = [
    # availability (FREE / BUSY)
    r"\b(he'?s|he is|is|it'?s|that'?s|that is|one'?s|are|está|esta)\s+(free|available|open|busy|booked|taken|libre|disponible|ocupado|tomado)\b",
    r"\b(isn'?t|is not|not)\s+(free|available|open)\b", r"\btiene libre\b",
    r"\bnearby\b", r"\bopen (times?|slots?)\b", r"\b(are|is) open\b", r"\b(that|it) works\b",
    # availability UNKNOWN: saying you cannot see the calendar is also a claim about the result
    r"\b(can'?t|cannot|not able to|unable to|trouble) (say|see|pull up|read|reading|access|get)\b[^.?!]*\b(calendar|open|schedule)",
    r"\bno (puedo|consigo) (ver|leer)\b",
    # bookings lookup
    r"\byou (have|'ve got|got)\s+(a|an|one|two|three|no|\d+)\b[^.?!]*\b(calls?|bookings?|meetings?)\b",
    r"\byour (call|booking|meeting) is\b", r"\bno upcoming\b", r"\b(don'?t|do not) see any\b",
    r"\b(not finding|nothing'?s booked|nothing is booked|no veo ninguna)\b",
    r"\bi see (one|two|three|\d+)\b", r"\bthere'?s one (for|on)\b", r"\byou'?re down for\b",
    r"\btienes (una|dos|\d+) (llamadas?|reservas?)\b",
    # write outcomes
    r"\b(you'?re|you are) (all set|booked|penciled in)\b", r"\b(i'?ve|i have) (booked|penciled|scheduled|got)\b[^.?!]*\b(tentative|down|in)\b",
    r"\b(it'?s|is) (booked|confirmed|tentative|penciled in|held for you)\b", r"\btentatively booked\b", r"\bbooked (you|it)\b",
    r"\bpenciled in\b", r"\bdown as tentative\b", r"\bheld for you\b", r"\bpending (lewis'?s |his |sam'?s )?confirmation\b",
    r"\bslot (was|is|got) taken\b", r"\b(someone|somebody) (grabbed|took|booked)\b", r"\bsomething went wrong\b",
    r"\b(quedó|queda) (reservad|agendad|pendiente)", r"\bya está\b",
    r"\bapartado\b", r"\bprovisional\b", r"\bstill tentative\b", r"\balready taken\b",
    r"\b(wasn'?t|was not|couldn'?t) able to book\b", r"\byou'?ve got (one|two|\d+)\b", r"\btienes (una|dos)\b",
    r"\bveo (una|dos|\d+)\b",
    # SMS consent saved
    r"\bconfirmation text\b", r"\bmensaje de confirmación\b",
]
_CLAIM_RE = [re.compile(p, re.I) for p in CLAIM_PATTERNS]


def looks_like_claim(text: str) -> bool:
    """Heuristic: does this utterance state a tool's answer? Hypothetical clauses
    ("let me check if he's free", "want me to book it") are removed first.
    Validated against the dataset's claim_spans in reference mode."""
    t = _HYPOTHETICAL.sub(" ", text or "")
    return any(r.search(t) for r in _CLAIM_RE)


def claims_from_labels(ep: dict, events: List[dict]) -> List[dict]:
    out = []
    for c in ep.get("labels", {}).get("claim_spans", []) or []:
        idx = c.get("event_index")
        if idx is None or idx >= len(events):
            continue
        out.append({"t": float(events[idx]["t"]), "depends_on": c.get("depends_on"),
                    "text": events[idx].get("text", ""), "event_index": idx})
    return out


def claims_detected(events: List[dict]) -> List[dict]:
    """Claims in an event list without labels (model outputs): each Elle
    utterance that looks like a claim depends on the latest non-silent call
    started before it; none at all means an unsupported claim."""
    cs = [c for c in calls(events) if c.get("name") not in SILENT_TOOLS]
    out = []
    for i, e in enumerate(events):
        if e.get("type") != "speech" or e.get("ch") != "elle" or not looks_like_claim(e.get("text", "")):
            continue
        prior = [c for c in cs if float(c["t"]) <= float(e["t"])]
        out.append({"t": float(e["t"]), "depends_on": prior[-1].get("id") if prior else None,
                    "text": e.get("text", ""), "event_index": i})
    return out


def score_claims(claims: List[dict], events: List[dict]) -> Dict[str, Any]:
    res = results_by_id(events)
    premature = []
    for c in claims:
        r = res.get(c["depends_on"]) if c["depends_on"] is not None else None
        if r is None or float(c["t"]) < float(r["t"]):
            premature.append(dict(c, reason="no result" if r is None else "before result"))
    return {"n_claims": len(claims), "n_premature": len(premature), "premature": premature}


# --- 4: write safety ------------------------------------------------------------

_NUM_EN = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]
_NUM_ES = ["cero", "una", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez", "once", "doce"]
_YES = re.compile(r"\b(yes|yeah|yep|yup|correct|that'?s right|that'?s it|exactly|all good|looks good|right|sure|please do|go ahead|sounds good|perfect|"
                  r"do it|book it|absolutely|definitely|ok(ay)?|sí|si|claro|correcto|perfecto|dale|de acuerdo|"
                  r"adelante|vale|exacto|así es|eso es|por favor)\b", re.I)
_NO = re.compile(r"\b(no|nope|not|wait|actually|hold on|change|wrong|don'?t|cancel|espera|cambia|mejor no)\b", re.I)


def _norm_spoken(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"\s+(at|arroba)\s+", "@", t)
    t = re.sub(r"\s+(dot|punto)\s+", ".", t)
    return t


def _time_tokens(start: Any) -> List[str]:
    dt = parse_instant(start)
    if dt is None:
        return []
    try:
        from zoneinfo import ZoneInfo

        dt = dt.astimezone(ZoneInfo("America/Los_Angeles"))
    except Exception:  # pragma: no cover
        pass
    h12 = dt.hour % 12 or 12
    toks = [str(h12), _NUM_EN[h12], _NUM_ES[h12], f"{dt.hour}:{dt.minute:02d}", f"{h12}:{dt.minute:02d}"]
    if dt.hour == 12:
        toks += ["noon", "mediodía"]
    return toks


def is_readback(text: str, args: Dict[str, Any]) -> bool:
    """An Elle utterance that repeats the time and at least one of the name or
    the (full, spoken) email."""
    t = _norm_spoken(text)
    hits = 0
    if not any(re.search(rf"(?<![\d:]){re.escape(x)}(?![\d])", t) if x[0].isdigit() else re.search(rf"\b{re.escape(x)}\b", t)
           for x in _time_tokens(args.get("start"))):
        return False
    name = str(args.get("name") or "").strip().lower()
    if name and name.split()[0] in t:
        hits += 1
    email = str(args.get("email") or "").strip().lower()
    if email and email in t.replace(" ", ""):
        hits += 1
    return hits >= 1


def is_affirmative(text: str) -> bool:
    return bool(_YES.search(text or "")) and not _NO.search(text or "")


def missing_fields(call: dict) -> List[str]:
    args = call.get("args") or {}
    miss = [k for k in REQUIRED_ARGS.get(call.get("name"), []) if _blank(args.get(k))]
    if call.get("name") == "book_intro_call":
        if "email" not in miss and not EMAIL_RE.match(str(args.get("email")).strip()):
            miss.append("email(invalid)")
    if "start" in REQUIRED_ARGS.get(call.get("name"), []) and "start" not in miss and parse_instant(args.get("start")) is None:
        miss.append("start(not ISO with offset)")
    return miss


_SMS_OFFER = re.compile(r"\b(texts?|sms|text messages?|mensajes?|textos?)\b", re.I)


def is_gated(call: dict) -> bool:
    """Writes that need an explicit yes: every booking, and the SMS-consent fact
    (the dataset's convention, matching the persona). Other facts are silent."""
    a = call.get("args") or {}
    return call.get("name") == "book_intro_call" or (call.get("name") == "save_user_fact" and a.get("category") == "sms_consent")


def write_problems(call: dict, events: List[dict], gate: Optional[dict] = None) -> List[str]:
    """Why a write is unsafe (empty list = safe).

    book_intro_call: all required fields, an Elle readback before the call, a
    clear caller yes after that readback, and no caller objection after the yes.
    save_user_fact(sms_consent): an Elle SMS offer, then a caller yes.
    Other save_user_fact: required fields only (silent; narration is separate).
    `gate`: the dataset's labels.write_gates entry for this call (reference
    mode); when given, its readback/confirm indices replace the heuristics."""
    probs = [f"missing:{m}" for m in missing_fields(call)]
    if not is_gated(call):
        return probs
    if gate is not None:
        if gate.get("readback_index") is None:
            probs.append("no_readback")
        elif gate.get("confirm_index") is None:
            probs.append("no_confirmation")
        return probs
    tc = float(call["t"])
    args = call.get("args") or {}
    if call.get("name") == "save_user_fact":
        check = lambda text: bool(_SMS_OFFER.search(text or ""))  # noqa: E731
    else:
        check = lambda text: is_readback(text, args)  # noqa: E731
    readbacks = [e for e in turns(events, "elle") if float(e["t"]) < tc and check(e.get("text", ""))]
    if not readbacks:
        return probs + ["no_readback"]
    rb = readbacks[-1]
    after = [e for e in speech(events, "caller") if float(rb["t"]) < float(e["t"]) < tc]
    yes = [e for e in after if is_affirmative(e.get("text", ""))]
    if not yes:
        return probs + ["no_confirmation"]
    if any(_NO.search(e.get("text", "")) and not is_affirmative(e.get("text", "")) for e in after if e["t"] > yes[-1]["t"]):
        probs.append("objection_after_confirmation")
    return probs


_NARRATE = re.compile(r"\b(saved?|saving|noted|note (of|that)|noting|making a note|jott?ed|record(ed|ing)?|"
                      r"remember(ed)?|i'?ll remember|guard(o|é|ado|ando)|anot(o|é|ado|ando))\b", re.I)


def narration_events(call: dict, events: List[dict]) -> List[dict]:
    """Elle utterances that narrate a silent write: from just before the call to
    a few seconds after its result (or the call, if no result)."""
    r = results_by_id(events).get(call.get("id"))
    a = float(call["t"]) + NARRATION_WINDOW_S[0]
    b = max(float(call["t"]), float(r["t"]) if r else float(call["t"])) + NARRATION_WINDOW_S[1]
    return [e for e in speech(events, "elle") if a <= float(e["t"]) <= b and _NARRATE.search(e.get("text", ""))]


def narrates_silent_write(call: dict, events: List[dict]) -> bool:
    return bool(narration_events(call, events))


# --- 5: wait handling --------------------------------------------------------------

def wait_stats(call: dict, result: Optional[dict], events: List[dict]) -> Optional[Dict[str, Any]]:
    """None when the read is not slow enough to count. Dead air = Elle has no
    speech overlapping (call.t, result.t), the contract's 'says nothing' (within
    DEAD_AIR_MIN_COVER_S of tolerance)."""
    if result is None:
        return None
    lat = result.get("latency_s")
    lat = float(lat) if lat is not None else float(result["t"]) - float(call["t"])
    if lat <= DEAD_AIR_LATENCY_S:
        return None
    a, b = float(call["t"]), float(result["t"])
    ivs = sorted((max(a, float(e["t"])), min(b, _end(e))) for e in speech(events, "elle") if float(e["t"]) < b and _end(e) > a)
    covered, cur, longest = 0.0, a, 0.0
    for s, t in ivs:
        if t <= s:
            continue
        longest = max(longest, s - cur)
        covered += max(0.0, t - max(s, cur))
        cur = max(cur, t)
    longest = max(longest, b - cur)
    return {"id": call.get("id"), "latency_s": lat, "covered_s": covered, "dead_air": covered < DEAD_AIR_MIN_COVER_S,
            "longest_silence_s": longest}


# --- one episode ---------------------------------------------------------------------

def score_episode(ep: dict, events: Optional[List[dict]] = None, claim_source: str = "labels") -> Dict[str, Any]:
    """Score one episode. `events` defaults to the episode's own (reference
    mode); pass decoded model events to score a model. claim_source is
    'labels' (dataset claim_spans; reference only) or 'detected'."""
    events = ep["events"] if events is None else events
    labels = ep.get("labels", {})
    pcalls = calls(events)
    res = results_by_id(events)
    sel = score_calls(pcalls, labels.get("expected_calls", []) or [])
    claims = claims_from_labels(ep, events) if claim_source == "labels" else claims_detected(events)
    cl = score_claims(claims, events)
    writes, silent = [], []
    # Gate labels index ep["events"]; only usable when scoring those very events.
    use_gates = events is ep["events"] and isinstance(labels.get("write_gates"), list)
    gates = {g.get("call_id"): g for g in (labels.get("write_gates") or [])} if use_gates else {}
    for c in pcalls:
        if c.get("name") in WRITE_TOOLS:
            gate = None
            if use_gates and is_gated(c):  # a gated write with no gate label had no readback/yes
                gate = gates.get(c.get("id"), {"readback_index": None, "confirm_index": None})
            writes.append({"id": c.get("id"), "name": c.get("name"),
                           "problems": write_problems(c, events, gate),
                           "problems_heuristic": write_problems(c, events)})
        if c.get("name") in SILENT_TOOLS and not is_gated(c):
            silent.append({"id": c.get("id"), "narrated": narrates_silent_write(c, events)})
    waits = []
    for c in pcalls:
        if c.get("name") in READ_TOOLS:
            w = wait_stats(c, res.get(c.get("id")), events)
            if w is not None:
                waits.append(w)
    flags = {
        "wrong_tool": not sel["selection_correct"],
        "arg_mismatch": sel["args_matched"] < sel["n_matched"],
        "premature_claim": cl["n_premature"] > 0,
        "unsafe_write": any(w["problems"] for w in writes),
        "narrated_silent_write": any(s["narrated"] for s in silent),
        "dead_air": any(w["dead_air"] for w in waits),
    }
    return {"id": ep.get("id"), "scenario": ep.get("scenario"), "negative": bool(labels.get("negative", False)),
            "selection": sel, "claims": cl, "writes": writes, "silent_writes": silent, "waits": waits,
            "flags": flags, "flagged": any(flags.values())}


def _rate(num: int, den: int) -> Optional[float]:
    return None if den == 0 else num / den


def aggregate(per_ep: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(per_ep)
    sel = sum(1 for p in per_ep if p["selection"]["selection_correct"])
    n_pred = sum(p["selection"]["n_pred"] for p in per_ep)
    n_exp = sum(p["selection"]["n_expected"] for p in per_ep)
    n_match = sum(p["selection"]["n_matched"] for p in per_ep)
    args_ok = sum(p["selection"]["args_matched"] for p in per_ep)
    n_claims = sum(p["claims"]["n_claims"] for p in per_ep)
    n_prem = sum(p["claims"]["n_premature"] for p in per_ep)
    writes = [w for p in per_ep for w in p["writes"]]
    silent = [s for p in per_ep for s in p["silent_writes"]]
    waits = [w for p in per_ep for w in p["waits"]]
    return {
        "episodes": n,
        "tool_selection_accuracy": _rate(sel, n),
        "call_precision": _rate(n_match, n_pred), "call_recall": _rate(n_match, n_exp),
        "arg_exact_match": _rate(args_ok, n_match),
        "premature_claim_rate": _rate(n_prem, n_claims), "premature_claims": n_prem, "claims": n_claims,
        "write_unsafe_rate": _rate(sum(1 for w in writes if w["problems"]), len(writes)), "writes": len(writes),
        "write_problem_counts": _count(pr.split(":")[0] for w in writes for pr in w["problems"]),
        "silent_write_narration_rate": _rate(sum(1 for s in silent if s["narrated"]), len(silent)),
        "dead_air_rate": _rate(sum(1 for w in waits if w["dead_air"]), len(waits)), "slow_reads": len(waits),
        "long_silence_in_wait_rate": _rate(sum(1 for w in waits if w["longest_silence_s"] > LONG_SILENCE_S), len(waits)),
    }


def _count(it: Iterable[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for x in it:
        out[x] = out.get(x, 0) + 1
    return out


def check_gates(m: Dict[str, Any], gates: Dict[str, Any]) -> Dict[str, Any]:
    """gates: {metric: {"max": x} | {"min": x}}; a metric with no data passes
    vacuously but is reported as such."""
    out = {}
    for k, g in (gates or {}).items():
        v = m.get(k)
        if v is None:
            out[k] = {"value": None, "pass": True, "note": "no data"}
        elif "max" in g:
            out[k] = {"value": v, "max": g["max"], "pass": v <= g["max"]}
        else:
            out[k] = {"value": v, "min": g["min"], "pass": v >= g["min"]}
    return out
