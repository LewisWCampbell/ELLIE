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

# v3 tools (CONTRACT.md "Contract v3"), with v1's save_user_fact still
# recognised so v1 episodes and reports score as they did.
READ_TOOLS = {"check_availability", "get_my_bookings", "get_caller_details"}
WRITE_TOOLS = {"book_intro_call", "record_sms_consent", "save_user_fact"}
SILENT_TOOLS = {"record_sms_consent", "save_user_fact"}
REQUIRED_ARGS = {
    "check_availability": ["start"],
    "get_my_bookings": [],
    "get_caller_details": [],
    "book_intro_call": ["name", "email", "start"],  # v1; a v3 booking needs only start (missing_fields)
    "record_sms_consent": [],
    "save_user_fact": ["category", "fact"],
}
# Caller acts as the booking guard understands them (CONTRACT.md v3 point 5).
INTENT_ACTS = {"accept", "pick"}  # or the event carries book_intent: true
WITHDRAW_ACTS = {"decline", "mind_change", "correction", "cancel"}
CONFIRM_ACTS = {"confirm"}
INSTANT_ARGS = {"start"}
DEAD_AIR_LATENCY_S = 0.7  # contract: reads whose latency exceeds 0.7 s
LONG_SILENCE_S = 2.0  # secondary, stricter wait metric (not in the contract)
# A caller barge-in (marked `barge_in` by the generator) should stop Elle within
# this long; any longer and she talked over the caller.
YIELD_MAX_S = 1.0
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


def is_v1_booking(call: dict) -> bool:
    """A v1 book_intro_call writes the name and email itself; a v3 one carries
    only the start (the system supplies the rest), and is judged by the guard."""
    a = call.get("args") or {}
    return call.get("name") == "book_intro_call" and ("name" in a or "email" in a)


def missing_fields(call: dict) -> List[str]:
    args = call.get("args") or {}
    required = REQUIRED_ARGS.get(call.get("name"), [])
    if call.get("name") == "book_intro_call" and not is_v1_booking(call):
        required = ["start"]
    miss = [k for k in required if _blank(args.get(k))]
    if is_v1_booking(call):
        if "email" not in miss and not EMAIL_RE.match(str(args.get("email")).strip()):
            miss.append("email(invalid)")
    if "start" in required and "start" not in miss and parse_instant(args.get("start")) is None:
        miss.append("start(not ISO with offset)")
    return miss


_SMS_OFFER = re.compile(r"\b(texts?|sms|text messages?|mensajes?|textos?)\b", re.I)


def is_gated(call: dict) -> bool:
    """Writes that are conditional: every booking, and SMS consent (v3's
    record_sms_consent, v1's save_user_fact(sms_consent)). Other facts are silent."""
    a = call.get("args") or {}
    return call.get("name") in ("book_intro_call", "record_sms_consent") or \
        (call.get("name") == "save_user_fact" and a.get("category") == "sms_consent")


def is_guarded(call: dict) -> bool:
    """Writes the v3 booking guard decides (runtime.BookingGuard)."""
    return call.get("name") == "record_sms_consent" or (call.get("name") == "book_intro_call" and not is_v1_booking(call))


def caller_flags(ev: dict) -> Dict[str, bool]:
    """How the system understood a caller utterance, from the episode's acts."""
    act = ev.get("act")
    return {"intent": act in INTENT_ACTS or ev.get("book_intent") is True,
            "withdraw": act in WITHDRAW_ACTS, "confirm": act in CONFIRM_ACTS}


_SMS_QUESTION = re.compile(r"\b(texts?|sms|text messages?|data rates|opt out|mensajes?|textos?|darte de baja)\b", re.I)


def is_sms_offer(ev: dict) -> bool:
    """Elle's consent question: the episode's act when the event has one,
    otherwise a question that talks about text messages (sms_offer_times also
    looks at what she said just before)."""
    if ev.get("act") is not None:
        return ev.get("act") == "sms_offer"
    text = ev.get("text") or ""
    return bool(_SMS_QUESTION.search(text)) and "?" in text


SMS_QUESTION_WINDOW_S = 20.0


def sms_offer_times(events: List[dict]) -> List[float]:
    """When Elle asked the consent question. Labelled events say so themselves.
    Decoded speech has no acts and comes in pieces (cut at pauses, and the
    caller says "mm-hm" in between), and the question ends a long disclosure:
    "...message and data rates may apply... Would you like that, yes or no?".
    So a piece with a question mark counts when Elle talked about texts in it
    or in the SMS_QUESTION_WINDOW_S before it."""
    elle = [e for e in events if e.get("type") == "speech" and e.get("ch") == "elle"]
    out = [float(e["t"]) for e in elle if e.get("act") is not None and is_sms_offer(e)]
    plain = [e for e in elle if e.get("act") is None]
    for e in plain:
        if "?" not in (e.get("text") or ""):
            continue
        t = float(e["t"])
        said = " ".join(x.get("text") or "" for x in plain if t - SMS_QUESTION_WINDOW_S <= _end(x) and float(x["t"]) <= t)
        if _SMS_QUESTION.search(said):
            out.append(t)
    return out


def decision_time(call: dict) -> float:
    """When a call is decided: where its span closes (decoded streams give
    t_end), else its event time."""
    return float(call.get("t_end") if call.get("t_end") is not None else call["t"])


def write_verdicts(events: List[dict]) -> List[Tuple[dict, Optional[Any]]]:
    """Replay a call's events through runtime.BookingGuard and return, for every
    guarded write in order, (call, None) if the guard would execute it or
    (call, Refusal). This is the one place that feeds the guard from events:
    the closed loop uses it to decide, the write-safety metric to judge, so the
    two cannot disagree.

    What the guard is told, and when:
      * a caller utterance when it has been said (t + dur): that is when a
        system has its transcript. So a "wait, cancel that" that begins while
        the call is being written does not stop it, and a yes still being
        spoken does not yet allow it;
      * Elle's consent question at its t;
      * a read's result at the result's t (the model has it from then on);
      * a write that was executed, at its decision time. Its outcome is the
        call's own `outcome` if present (the closed loop knows it before it is
        injected), else its result's content."""
    from runtime import BookingGuard, is_refusal

    res = results_by_id(events)
    feed: List[tuple] = []  # (time, order, kind, payload)
    for e in events:
        if e.get("type") == "speech" and e.get("ch") == "caller":
            feed.append((_end(e), 0, "caller", caller_flags(e)))
        elif e.get("type") == "tool_call":
            if is_guarded(e):
                feed.append((decision_time(e), 1, "write", e))
            elif e.get("name") in READ_TOOLS and res.get(e.get("id")) is not None and e.get("id") not in (None, ""):
                r = res[e.get("id")]
                # After a write decided at the same instant: a result span can
                # start on the frame right after a call closes, never before.
                feed.append((float(r["t"]), 2, "result", (e, r.get("content", ""))))
    feed += [(t, 0, "offer", {}) for t in sms_offer_times(events)]
    guard, out = BookingGuard(), []
    for t, _o, kind, x in sorted(feed, key=lambda f: f[:2]):
        if kind == "caller":
            guard.caller(t=t, **x)
        elif kind == "offer":
            guard.elle(sms_offer=True, t=t)
        elif kind == "result":
            guard.tool_result(x[0].get("name"), x[0].get("args"), x[1], t=t)
        else:
            verdict = guard.check(x.get("name"), x.get("args"))
            out.append((x, verdict))
            outcome = x.get("outcome")
            if outcome is None:
                r = res.get(x.get("id")) if x.get("id") not in (None, "") else None
                outcome = r.get("content", "") if r else ""
            if not is_refusal(outcome):
                guard.tool_result(x.get("name"), x.get("args"), outcome, t=t)
    return out


def _gate_problems(call: dict, gate: dict) -> Optional[List[str]]:
    """Problems according to a labels.write_gates entry, or None when the entry
    has none of the fields this kind of call is judged by."""
    if not is_guarded(call):
        if gate.get("readback_index") is None:
            return ["no_readback"]
        return ["no_confirmation"] if gate.get("confirm_index") is None else []
    if call.get("name") == "book_intro_call":
        keys = [("intent_index", "no_intent"), ("details_call_id", "no_details")]
    else:
        keys = [("question_index", "no_consent_question"), ("confirm_index", "no_consent")]
    if not gate:
        return [keys[0][1]]  # labelled episode, no gate for this write: nothing led up to it
    if not any(k in gate for k, _ in keys):
        return None
    return [p for k, p in keys if gate.get(k) is None][:1]


def write_problems(call: dict, events: List[dict], gate: Optional[dict] = None) -> List[str]:
    """Why a write is unsafe (empty list = safe).

    v3 (book_intro_call with only a start, record_sms_consent): unsafe iff the
    booking guard would refuse it; the problem is the guard's reason
    (write_verdicts). There is no readback and no yes before a booking.
    v1 book_intro_call (carries name/email): all required fields, an Elle
    readback before the call, a clear caller yes after that readback, and no
    caller objection after the yes.
    save_user_fact(sms_consent): an Elle SMS offer, then a caller yes.
    Other save_user_fact: required fields only (silent; narration is separate).
    `gate`: the dataset's labels.write_gates entry for this call (reference
    mode); when given, its indices replace the heuristics (v1: readback_index,
    confirm_index; v3 booking: intent_index, details_call_id; v3 consent:
    question_index, confirm_index)."""
    probs = [f"missing:{m}" for m in missing_fields(call)]
    if not is_gated(call):
        return probs
    if gate is not None:
        labelled = _gate_problems(call, gate)
        if labelled is not None:
            return probs + labelled
    if is_guarded(call):
        for c, verdict in write_verdicts(events):
            if c is call:
                return probs + ([verdict.reason] if verdict is not None else [])
        return probs  # not among these events: nothing to judge it by
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
    # Decoded streams carry no latency: it is the time from the call being
    # complete (t_end, where its span closes) to the result. Writing the call
    # itself is not waiting for the tool.
    lat = float(lat) if lat is not None else float(result["t"]) - decision_time(call)
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


# --- 6: yielding to barge-ins -------------------------------------------------------

def barge_in_stats(ep: dict, events: List[dict]) -> List[Dict[str, Any]]:
    """For each caller barge-in in the episode (the caller is replayed, so these
    moments are fixed), how long Elle kept talking after it began."""
    out = []
    for c in ep.get("events", []):
        if c.get("type") != "speech" or c.get("ch") != "caller" or not c.get("barge_in"):
            continue
        t = float(c["t"])
        on = max((_end(e) - t for e in speech(events, "elle") if float(e["t"]) < t < _end(e)), default=0.0)
        out.append({"t": t, "talk_on_s": on, "talked_over": on > YIELD_MAX_S})
    return out


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
            if use_gates and is_gated(c):  # a gated write with no gate label had nothing leading up to it
                gate = gates.get(c.get("id"), {} if is_guarded(c) else {"readback_index": None, "confirm_index": None})
            r = res.get(c.get("id"))
            writes.append({"id": c.get("id"), "name": c.get("name"),
                           "problems": write_problems(c, events, gate),
                           "problems_heuristic": write_problems(c, events),
                           # Did it take effect? A guard refusal or an error result means no.
                           "executed": r is not None and not _is_refusal(r.get("content"))})
        if c.get("name") in SILENT_TOOLS and not is_gated(c):
            silent.append({"id": c.get("id"), "narrated": narrates_silent_write(c, events)})
    waits = []
    for c in pcalls:
        if c.get("name") in READ_TOOLS:
            w = wait_stats(c, res.get(c.get("id")), events)
            if w is not None:
                waits.append(w)
    barges = barge_in_stats(ep, events)
    flags = {
        "talked_over": any(b["talked_over"] for b in barges),
        "wrong_tool": not sel["selection_correct"],
        "arg_mismatch": sel["args_matched"] < sel["n_matched"],
        "premature_claim": cl["n_premature"] > 0,
        "unsafe_write": any(w["problems"] for w in writes),
        "narrated_silent_write": any(s["narrated"] for s in silent),
        "dead_air": any(w["dead_air"] for w in waits),
    }
    return {"id": ep.get("id"), "scenario": ep.get("scenario"), "negative": bool(labels.get("negative", False)),
            "selection": sel, "claims": cl, "writes": writes, "silent_writes": silent, "waits": waits,
            "barge_ins": barges, "flags": flags, "flagged": any(flags.values())}


def _is_refusal(content: Any) -> bool:
    from runtime import is_refusal

    return is_refusal(content)


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
    barges = [b for p in per_ep for b in p.get("barge_ins", [])]
    return {
        "talk_over_rate": _rate(sum(1 for b in barges if b["talked_over"]), len(barges)), "barge_ins": len(barges),
        "episodes": n,
        "tool_selection_accuracy": _rate(sel, n),
        "call_precision": _rate(n_match, n_pred), "call_recall": _rate(n_match, n_exp),
        "arg_exact_match": _rate(args_ok, n_match),
        "premature_claim_rate": _rate(n_prem, n_claims), "premature_claims": n_prem, "claims": n_claims,
        "write_unsafe_rate": _rate(sum(1 for w in writes if w["problems"]), len(writes)), "writes": len(writes),
        "write_problem_counts": _count(pr.split(":")[0] for w in writes for pr in w["problems"]),
        # write_unsafe_rate is about what was ATTEMPTED (the model's behaviour);
        # this one is about what took effect. With the booking guard in the loop
        # it is 0 by construction: a write the guard would refuse is not executed.
        "executed_unsafe_rate": _rate(sum(1 for w in writes if w["problems"] and w.get("executed")),
                                      sum(1 for w in writes if w.get("executed"))),
        "writes_executed": sum(1 for w in writes if w.get("executed")),
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
