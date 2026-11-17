"""Scenario grammar: reusable beats composed into named scenarios.

A *beat* is a short exchange (opening, ask for a slot, check availability,
collect details, get the caller's details, book, say what was booked, SMS
consent, goodbye). A *scenario* is a
function that composes beats with random choices. Every scenario is registered
in `SCENARIOS` with its family (used for held-out splits), relative weight, and
the languages it supports.

Adding a scenario: write `def sc_<name>(ep): ...` using the beats below, then
add a `Scenario(...)` row to `SCENARIOS`. `python3 -m unittest discover
training/data` builds every scenario in every supported language and validates
the output, so a broken scenario fails loudly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import banks
import elle_time as T
from episode import OWNER, Ep, Person, spoken_email

# --- production result strings (src/booking/*.ts, src/sms.ts, twilioBridge.ts) ----

# Contract v3: a booking needs a FREE check, so the advice no longer offers a request.
UNKNOWN_ADVICE = "Don't tell the caller whether the time is free. Offer to try again in a moment."


def res_free(dt: datetime) -> str:
    return f"FREE: {T.speakable_time(dt)} is open. Offer it to the caller."


def res_busy(dt: datetime, alts: Sequence[datetime]) -> str:
    said = T.speakable_time(dt)
    if not alts:
        return f"BUSY: {said} is taken, and nothing nearby in working hours is open. Ask for another day."
    return (f"BUSY: {said} is taken. Nearby open times: {'; '.join(T.speakable_time(a) for a in alts)}. "
            f"Offer one or two of these, not the whole list.")


def res_unknown(kind: str = "read") -> str:
    if kind == "nocal":
        return f"UNKNOWN: no calendar is connected, so availability can't be checked. {UNKNOWN_ADVICE}"
    return f"UNKNOWN: couldn't read the calendar just now. {UNKNOWN_ADVICE}"


def res_done(email: str, placed: str = f"penciled in as TENTATIVE on {OWNER}'s calendar, pending his confirmation") -> str:
    return (f"Done: {placed}. IMPORTANT — tell the caller this time is tentative until {OWNER} confirms, "
            f"and that he'll confirm by email to {email}. Do not imply it is final. Then tell them they "
            f"can text this same number any time to change or cancel it — say this HERE rather than in the "
            f"greeting, because it only matters once they actually have a booking.")


RES_TAKEN = ("Someone else booked that exact time a moment ago. Apologize briefly, offer the caller "
             "another time, and book again.")
RES_BUSY_AT_WRITE = f"That time is already busy on {OWNER}'s calendar. Offer the caller a different time and book again."
RES_WRITE_ERROR = "Error booking the call; offer to note their details for a follow-up instead."
RES_SAVED = "Saved."

# --- Contract v3 result strings (the system, not production's text tools) ----------

RES_MISSING_EMAIL = "MISSING: email. Ask for it."
RES_INVALID_EMAIL = "MISSING: email (what was heard was not a valid address). Ask for it again."


def res_details(ep: Ep) -> str:
    """get_caller_details. ON FILE when the caller's history has both and nothing
    was corrected in this call; HEARD when the system took both from this call's
    transcript; MISSING otherwise. The email is always in the English spoken
    form of spoken_email(), whatever language the call is in."""
    said = f"{ep.p.name}, {spoken_email(ep.p.email)}."
    on_file = ep.caller_on_file
    if on_file and on_file["email"] == ep.p.email:
        return f"ON FILE: {said}"
    missing = [w for w, have in (("name", ep.heard_name or bool(on_file)), ("email", ep.heard_email)) if not have]
    if missing:
        return f"MISSING: {' and '.join(missing)}. Ask for it."
    return f"HEARD: {said}"


def res_updated(email: str) -> str:
    """A repeat book_intro_call for a start already booked in this call, after
    the caller corrected their details: the system updates the booking."""
    return (f"Updated: the booking keeps its time and the invite now goes to {spoken_email(email)}. "
            f"Tell the caller it is fixed. It is still tentative until {OWNER} confirms.")
RES_NO_BOOKINGS = "No bookings found for this phone number."


def res_bookings(rows: Sequence[Tuple[int, str, datetime, str, str]]) -> str:
    if not rows:
        return RES_NO_BOOKINGS
    return "\n".join(f"#{bid}: {name} — {T.fmt_when(dt)} ({status})" + (f" — topic: {topic}" if topic else "")
                     for bid, name, dt, status, topic in rows)


# --- latency samplers --------------------------------------------------------------

def lat_read(ep: Ep) -> float:
    r = ep.rng.random()
    return ep.rng.uniform(0.18, 0.7) if r < 0.55 else ep.rng.uniform(0.7, 2.6)


def lat_slow(ep: Ep) -> float:
    return ep.rng.uniform(1.0, 3.0)


def lat_write(ep: Ep) -> float:
    return ep.rng.uniform(0.6, 2.2)


def lat_save(ep: Ep) -> float:
    return ep.rng.uniform(0.05, 0.3)


def lat_details(ep: Ep) -> float:
    """get_caller_details is a lookup in the system's own state: a fast read."""
    return ep.rng.uniform(0.12, 0.5) if ep.rng.random() < 0.75 else ep.rng.uniform(0.5, 1.1)


# --- slots & calendar ---------------------------------------------------------------

def busy_around(ep: Ep, slot: datetime) -> Tuple[List[Tuple[datetime, datetime]], List[datetime]]:
    """A plausible busy calendar that blocks `slot`, and the alternatives the
    production nearbyOpenings would suggest for it."""
    for _ in range(20):
        before = ep.rng.choice([0, 0, 1, 2])
        after = ep.rng.choice([1, 1, 2, 3])
        blocks = [(slot - timedelta(minutes=30 * before), slot + timedelta(minutes=30 * after))]
        for _ in range(ep.rng.randint(0, 3)):
            o = ep.rng.randint(-10, 30) * 30
            blocks.append((slot + timedelta(minutes=o), slot + timedelta(minutes=o + 30 * ep.rng.randint(1, 3))))
        alts = T.nearby_openings(slot, blocks, ep.now)
        if len(alts) >= 2 or ep.rng.random() < 0.05:
            return blocks, alts
    return blocks, alts


def other_slot(ep: Ep, avoid: Sequence[datetime], near: Optional[datetime] = None) -> datetime:
    for _ in range(200):
        if near is not None:
            cand = near + timedelta(minutes=30 * ep.rng.choice([-4, -2, -1, 1, 2, 3, 4, 48, 96]))
            if not T.is_bookable(cand, ep.now):
                cand = T.random_slot(ep.rng, ep.now)
        else:
            cand = T.random_slot(ep.rng, ep.now)
        if all(cand != a for a in avoid):
            return cand
    raise RuntimeError("no other slot")


# --- beats ---------------------------------------------------------------------------

def greet(ep: Ep, at: Optional[float] = None):
    """Elle's greeting. A caller on file is greeted by first name and never asked
    who they are (the prompt already says who is calling)."""
    at = at if at is not None else round(ep.rng.uniform(0.2, 0.6), 2)
    if ep.on_file:
        return ep.E("elle_greet_known", act="greeting_known", at=at)
    return ep.E("elle_greet", act="greeting", at=at)


def set_intent(ep: Ep, ev, slot: datetime, **extra) -> None:
    """Mark a caller event as asking to book `slot` or accepting it. The booking
    guard reads these events (CONTRACT.md v3, point 5c)."""
    ev["refers"] = {"start": T.iso(slot)}
    ev.update(extra)
    ep.intent, ep.intent_slot = ev, slot


def has_intent(ep: Ep, slot: datetime) -> bool:
    return ep.intent is not None and ep.intent_slot == slot


def opening(ep: Ep, intent: str = "book") -> bool:
    """Optional greeting and caller opener. Returns True when the caller's
    opener already contains the availability question (direct ask)."""
    greeted = ep.rng.random() < 0.55
    g = None
    if greeted:
        g = greet(ep)
    if ep.rng.random() < 0.3:
        return True  # caller goes straight to the time question
    key = "caller_open_book" if intent == "book" else "caller_open_check"
    if g is not None and g["dur"] > 2.5 and ep.rng.random() < 0.2:
        ep.interrupt(g, key, frac=ep.rng.uniform(0.55, 0.85), act="intent")  # caller talks over the greeting
    else:
        ep.C(key, act="intent", at=0.0 if not greeted else None)
    return False


def ask_for_slot(ep: Ep, slot: datetime, direct: bool, stated: Optional[bool] = False) -> bool:
    """The caller names a time. `stated`: True, they ask to book it ("can you
    book me in for..."); False, they only ask whether it is open; None, either.
    Returns whether booking intent was stated (the event carries book_intent)."""
    when, _ = T.caller_phrase(slot, ep.now, ep.rng, ep.lang)
    ep.vars["when"] = when
    ep.set_slot("slot", slot)
    at = 0.0 if direct and not ep.events else None
    if not direct:
        ep.E("elle_ask_time", act="ask_time")
    if stated is None:
        stated = ep.rng.random() < 0.4
    ev = ep.C("caller_want_book_at" if stated else "caller_ask_avail", act="ask_slot", at=at)
    if stated:
        set_intent(ep, ev, slot, book_intent=True)
    return stated


def check(ep: Ep, slot: datetime, outcome: str, latency: Optional[float] = None,
          during=None, pre_key: Optional[str] = "elle_check_pre", fill: bool = True,
          content: Optional[str] = None) -> Tuple[str, List[datetime]]:
    alts: List[datetime] = []
    if content is None:
        content, alts = outcome_content(ep, slot, outcome)
    lat = latency if latency is not None else lat_read(ep)
    cid, _ = ep.read("check_availability", {"start": T.iso(slot)}, content, lat,
                     pre_key=pre_key, during=during, fill=fill)
    return cid, alts


def outcome_content(ep: Ep, slot: datetime, outcome: str) -> Tuple[str, List[datetime]]:
    """Result text for a check of `slot`, exposing any alternatives as {alt1}/{alt2}."""
    alts: List[datetime] = []
    if outcome == "free":
        content = res_free(slot)
    elif outcome == "busy":
        _, alts = busy_around(ep, slot)
        content = res_busy(slot, alts)
    else:
        content = res_unknown("nocal" if ep.rng.random() < 0.25 else "read")
    for i, a in enumerate(alts[:2]):
        ep.set_slot(f"alt{i + 1}", a)
    return content, alts


def say_busy(ep: Ep, cid: str, alts: List[datetime]):
    key = "elle_busy" if len(alts) >= 2 else "elle_busy_one" if alts else "elle_busy_none"
    return ep.claim(key, cid)


def say_yes(ep: Ep, slot: datetime):
    ev = ep.C("caller_yes_book", act="accept")
    set_intent(ep, ev, slot)
    return ev


def accept_offer(ep: Ep, offer, slot: datetime) -> None:
    """Caller says yes to Elle's offer; sometimes eagerly, over its last words."""
    if offer is not None and offer["dur"] > 2.5 and ep.rng.random() < 0.2:
        ev = ep.interrupt(offer, "caller_yes_book", frac=ep.rng.uniform(0.75, 0.92), act="accept")
        set_intent(ep, ev, slot)
    else:
        say_yes(ep, slot)


def after_free(ep: Ep, cid: str, slot: datetime) -> None:
    """Report a FREE result. If the caller already asked to book this time (or
    picked or accepted it), Elle goes ahead; otherwise she offers it and the
    caller accepts. Which of the two happens depends only on what the caller
    said, never on a coin flip."""
    ep.set_slot("slot", slot)
    if has_intent(ep, slot):
        ep.claim("elle_free_go", cid)
    else:
        accept_offer(ep, ep.claim("elle_free", cid), slot)


def hold_off(ep: Ep) -> None:
    ep.C("caller_hold_off", act="decline")
    ep.intent = ep.intent_slot = None
    ep.E("elle_hold_off_ack", act="ack")


def pick_alt(ep: Ep, alts: List[datetime]) -> datetime:
    if len(alts) >= 2 and ep.rng.random() < 0.4:
        ev = ep.C("caller_pick_alt2", act="pick")
        chosen = alts[1]
    else:
        ev = ep.C("caller_pick_alt1", act="pick")
        chosen = alts[0]
    set_intent(ep, ev, chosen)
    ep.set_slot("slot", chosen)
    return chosen


def book_pick(ep: Ep, chosen: datetime, **kw) -> str:
    """The caller picked one of the alternatives from a BUSY result. The guard
    books only a time whose own check came back FREE, so Elle checks the pick
    (always), then books it."""
    cid, _ = check(ep, chosen, "free")
    after_free(ep, cid, chosen)
    return full_booking(ep, chosen, **kw)


def details(ep: Ep) -> None:
    """Ask a new caller for whatever is still missing. A caller on file is never asked."""
    if ep.on_file:
        return
    if not ep.heard_name and not ep.heard_email and ep.rng.random() < 0.25:
        ep.E("elle_ask_name", act="ask_name")
        ep.C("caller_give_all", act="give_details")
        ep.heard_name = ep.heard_email = True
        return
    if not ep.heard_name:
        ep.E("elle_ask_name", act="ask_name")
        ep.C("caller_give_name", act="give_name")
        ep.heard_name = True
    if not ep.heard_email:
        ep.E("elle_ask_email", act="ask_email")
        ep.C("caller_give_email", act="give_email")
        ep.heard_email = True


def caller_details(ep: Ep, content: Optional[str] = None, pre_key: Optional[str] = "elle_book_pre",
                   quiet: bool = False) -> str:
    """The get_caller_details read that precedes every booking. A fast read:
    usually under half a second, sometimes around a second, and then Elle is
    always mid-sentence (the pre-line) so the wait is covered. `quiet` drops the
    pre-line and keeps the read fast."""
    lat = lat_details(ep)
    said = ep.rng.random() < 0.6
    if quiet:
        lat = min(lat, 0.5)
    key = None if quiet or not (said or lat > 0.6) else pre_key
    cid, _ = ep.read("get_caller_details", {}, content if content is not None else res_details(ep),
                     lat, pre_key=key)
    return cid


def book(ep: Ep, slot: datetime, details_cid: str, content: Optional[str] = None,
         latency: Optional[float] = None, gated: bool = True,
         at: Optional[float] = None) -> str:
    """book_intro_call right after the details result. The only argument is the
    start: name and email are attached by the system."""
    if at is None:
        at = ep.results[details_cid]["t"] + ep.rng.uniform(0.05, 0.25)
    cid = ep.call("book_intro_call", {"start": T.iso(slot)}, at)
    ep.result(cid, content if content is not None else res_done(ep.p.email), latency or lat_write(ep))
    if gated:
        ep.gate_booking(cid, ep.intent, details_cid)
    return cid


def booked(ep: Ep, cid: str, details_cid: str, slot: datetime, key: str = "elle_booked",
           act: str = "booked", backchannel: bool = True):
    """What Elle says once the booking result is in: the time with the timezone,
    that it is tentative, and where the invite goes. A claim on the booking
    result and on the details result the email came from."""
    ep.set_slot("slot", slot)
    ev = ep.claim(key, cid, act=act, also=[details_cid])
    ev["refers"] = {"start": T.iso(slot), "email": ep.p.email}
    ev["_details"] = details_cid
    if backchannel:
        ep.backchannel(ev)
    return ev


def sms_consent(ep: Ep, yes: Optional[bool] = None) -> None:
    if yes is None:
        yes = ep.rng.random() < 0.6
    a = ep.E("elle_sms_offer_a", act="sms_offer_intro", gap=ep.gap(0.5, 0.15))
    ep.backchannel(a)
    b = ep.E("elle_sms_offer_b", act="sms_offer", gap=ep.gap(0.2, 0.08))
    ep.backchannel(b)
    if yes:
        cf = ep.C("caller_sms_yes", act="confirm")
        # Silent write: emitted as the floor frees up, never mentioned.
        cid = ep.call("record_sms_consent", {}, ep.write_at())
        ep.result(cid, RES_SAVED, lat_save(ep))
        ep.gate_consent(cid, b, cf)
        ep.claim("elle_sms_yes_after", cid)
    else:
        ep.C("caller_sms_no", act="decline")
        ep.E("elle_sms_no_after", act="ack")


def bye(ep: Ep, p: float = 0.8) -> None:
    if ep.rng.random() < p:
        ep.C("caller_bye", act="bye")
        ep.E("elle_bye", act="bye")


def full_booking(ep: Ep, slot: datetime, sms: Optional[bool] = None, end: bool = True,
                 key: str = "elle_booked") -> str:
    """(ask name and email) -> get_caller_details -> book -> say what was booked -> (sms) -> (bye).
    The caller has already asked for or accepted `slot`; there is no readback."""
    details(ep)
    d = caller_details(ep)
    cid = book(ep, slot, d)
    booked(ep, cid, d, slot, key=key)
    if sms is None:
        sms = ep.rng.random() < 0.65
    if sms:
        sms_consent(ep)
    if end:
        bye(ep)
    return cid


def slot_and_ask(ep: Ep, intent: str = "book", stated: Optional[bool] = False,
                 slot: Optional[datetime] = None) -> datetime:
    slot = slot or T.random_slot(ep.rng, ep.now)
    direct = opening(ep, intent)
    ask_for_slot(ep, slot, direct, stated)
    return slot


# --- scenarios: reads -------------------------------------------------------------------

def sc_availability_free(ep: Ep) -> None:
    slot = slot_and_ask(ep, "check")
    cid, _ = check(ep, slot, "free")
    if ep.rng.random() < 0.3:
        ep.claim("elle_free_noop", cid)
        bye(ep, 1.0)
    else:
        ep.claim("elle_free", cid)
        hold_off(ep)
        bye(ep, 0.5)


def sc_availability_busy_alternatives(ep: Ep) -> None:
    slot = slot_and_ask(ep, "check")
    cid, alts = check(ep, slot, "busy")
    say_busy(ep, cid, alts)
    hold_off(ep)
    bye(ep, 0.5)


def _booking_rows(ep: Ep, n: int, name: Optional[str] = None) -> List[Tuple[int, str, datetime, str, str]]:
    slots: List[datetime] = []
    while len(slots) < n:
        s = T.random_slot(ep.rng, ep.now, max_days=20)
        if s not in slots:
            slots.append(s)
    slots.sort()
    rows = []
    for s in slots:
        status = "pending" if ep.rng.random() < 0.8 else "confirmed"
        rows.append((ep.rng.randint(12, 480), name or ep.p.name, s, status,
                     ep.p.topic if ep.rng.random() < 0.5 else ""))
    return rows


def bookings_read(ep: Ep, n: int, latency: Optional[float] = None, during=None,
                  rows=None) -> Tuple[str, list]:
    if ep.rng.random() < 0.5:
        greet(ep)
    ep.C("caller_ask_bookings", act="ask_bookings", at=0.0 if not ep.events else None)
    rows = rows if rows is not None else _booking_rows(ep, n)
    cid, _ = ep.read("get_my_bookings", {}, res_bookings(rows), latency or lat_read(ep),
                     pre_key="elle_bookings_pre", during=during)
    return cid, rows


def say_bookings(ep: Ep, cid: str, rows) -> None:
    if not rows:
        ep.claim("elle_bookings_none", cid)
        if ep.rng.random() < 0.5:
            ep.C("caller_bookings_thanks", act="decline")
            ep.E("elle_hold_off_ack", act="ack")
        return
    ep.set_slot("slot", rows[0][2])
    if len(rows) == 1:
        key = "elle_bookings_one_confirmed" if rows[0][3] == "confirmed" and ep.lang == "en" else "elle_bookings_one"
        ep.claim(key, cid)
        ep.C("caller_bookings_thanks", act="thanks")
        if ep.rng.random() < 0.6:
            ep.E("elle_text_to_change", act="info")
    else:
        ep.vars["b2_long"] = T.elle_long(rows[1][2], ep.lang)
        ep.claim("elle_bookings_two", cid)
        ep.C("caller_pick_booking", act="pick_booking")
        ep.E("elle_text_to_change", act="info")
    bye(ep, 0.6)


def sc_bookings_none(ep: Ep) -> None:
    cid, rows = bookings_read(ep, 0)
    say_bookings(ep, cid, rows)


def sc_bookings_one(ep: Ep) -> None:
    cid, rows = bookings_read(ep, 1)
    say_bookings(ep, cid, rows)


def sc_bookings_two(ep: Ep) -> None:
    cid, rows = bookings_read(ep, 2)
    say_bookings(ep, cid, rows)


def sc_availability_slow_filler(ep: Ep) -> None:
    slot = slot_and_ask(ep, "check")
    outcome = "free" if ep.rng.random() < 0.6 else "busy"
    cid, alts = check(ep, slot, outcome, latency=lat_slow(ep))
    if outcome == "free":
        ep.claim("elle_free", cid)
        if ep.rng.random() < 0.5:
            say_yes(ep, slot)
            full_booking(ep, slot)
            return
        hold_off(ep)
    else:
        say_busy(ep, cid, alts)
        hold_off(ep)
    bye(ep, 0.4)


def sc_bookings_slow_filler(ep: Ep) -> None:
    cid, rows = bookings_read(ep, ep.rng.choice([0, 1, 1, 2]), latency=lat_slow(ep))
    say_bookings(ep, cid, rows)


def _ramble(ep: Ep, call_t: float, res_t: float) -> None:
    ev = ep.C("caller_ramble", act="ramble", at=call_t + ep.rng.uniform(0.15, 0.6))
    if ev["t"] + ev["dur"] < res_t - 0.6 and ep.rng.random() < 0.5:
        ep.C("caller_ramble", act="ramble", gap=ep.gap(0.25, 0.1))


def sc_availability_caller_keeps_talking(ep: Ep) -> None:
    slot = slot_and_ask(ep, "check")
    outcome = "free" if ep.rng.random() < 0.6 else "busy"
    cid, alts = check(ep, slot, outcome, latency=ep.rng.uniform(0.8, 2.6), during=_ramble)
    if outcome == "free":
        after_free(ep, cid, slot)
        full_booking(ep, slot)
    else:
        say_busy(ep, cid, alts)
        if alts:
            book_pick(ep, pick_alt(ep, alts))
        else:
            hold_off(ep)


def sc_availability_mind_change_mid_wait(ep: Ep) -> None:
    slot = slot_and_ask(ep, "check")
    slot2 = other_slot(ep, [slot], near=slot)
    when2, _ = T.caller_phrase(slot2, ep.now, ep.rng, ep.lang)
    ep.vars["when2"] = when2
    ep.set_slot("slot2", slot2)
    lat1 = ep.rng.uniform(1.2, 2.8)
    first_outcome = "free" if ep.rng.random() < 0.5 else "busy"

    def change(e: Ep, call_t: float, res_t: float) -> None:
        e.C("caller_mind_change", act="mind_change", at=call_t + e.rng.uniform(0.3, 0.9))

    c1, _ = check(ep, slot, first_outcome, latency=lat1, during=change, fill=False)
    # Elle acknowledges the change and checks the new time; whatever c1 says is
    # now about a time the caller no longer wants, so she doesn't report it.
    outcome2 = "free" if ep.rng.random() < 0.7 else "busy"
    c2, alts = check(ep, slot2, outcome2, pre_key="elle_mind_ack")
    ep.set_slot("slot", slot2)
    if outcome2 == "free":
        ep.claim("elle_free", c2)
        if ep.rng.random() < 0.6:
            say_yes(ep, slot2)
            full_booking(ep, slot2)
            return
        hold_off(ep)
    else:
        say_busy(ep, c2, alts)
        if alts:
            book_pick(ep, pick_alt(ep, alts))
            return
        hold_off(ep)
    bye(ep, 0.5)


def sc_time_change_after_booking(ep: Ep) -> None:
    """The caller moves the call after it was booked. That is a new request: a
    new check and a new booking for the new time; the system replaces the
    earlier tentative hold, and Elle says so."""
    slot = slot_and_ask(ep, "book", stated=None)
    c1, _ = check(ep, slot, "free")
    after_free(ep, c1, slot)
    full_booking(ep, slot, sms=False, end=False)
    slot2 = _new_when2(ep, slot)
    ev = ep.C("caller_change_time_after", act="ask_slot")
    set_intent(ep, ev, slot2, book_intent=True, change=True)
    c2, _ = check(ep, slot2, "free", pre_key="elle_mind_ack")
    after_free(ep, c2, slot2)
    full_booking(ep, slot2, sms=False, key="elle_booked_moved")


def sc_availability_unknown_recovery(ep: Ep) -> None:
    """UNKNOWN: Elle offers to try again. A FREE or BUSY retry continues as
    usual. If the retry is UNKNOWN too, or the caller does not want to wait,
    she says she cannot see the calendar and asks them to call or text back:
    nothing is booked without a FREE check, and no further call is made."""
    slot = slot_and_ask(ep, "book")
    c1, _ = check(ep, slot, "unknown")
    ep.claim("elle_unknown", c1)
    if ep.rng.random() < 0.3:
        ep.C("caller_no_wait", act="decline")
        ep.intent = ep.intent_slot = None
        ep.E("elle_unknown_later", act="ack")
        bye(ep, 0.6)
        return
    ep.C("caller_retry", act="retry")
    r = ep.rng.random()
    outcome = "free" if r < 0.55 else "busy" if r < 0.8 else "unknown"
    c2, alts = check(ep, slot, outcome)
    if outcome == "free":
        after_free(ep, c2, slot)
        full_booking(ep, slot)
    elif outcome == "busy":
        say_busy(ep, c2, alts)
        if alts:
            book_pick(ep, pick_alt(ep, alts))
        else:
            hold_off(ep)
    else:
        ep.claim("elle_unknown_again", c2)
        bye(ep, 0.7)


def sc_availability_timeout(ep: Ep) -> None:
    slot = slot_and_ask(ep, "check")
    c1, _ = check(ep, slot, "unknown", latency=ep.rng.uniform(3.2, 6.0),
                  content=res_unknown("read"))
    ep.claim("elle_unknown", c1)
    ep.C("caller_retry", act="retry")
    outcome = "free" if ep.rng.random() < 0.75 else "busy"
    c2, alts = check(ep, slot, outcome, latency=ep.rng.uniform(0.2, 0.9))
    if outcome == "free":
        ep.claim("elle_free", c2)
        if ep.rng.random() < 0.5:
            say_yes(ep, slot)
            full_booking(ep, slot)
            return
        hold_off(ep)
    else:
        say_busy(ep, c2, alts)
        hold_off(ep)
    bye(ep, 0.5)


# --- scenarios: writes and mixed -------------------------------------------------------------

def sc_book_after_availability_free(ep: Ep) -> None:
    slot = slot_and_ask(ep, "book", stated=None)
    cid, _ = check(ep, slot, "free")
    after_free(ep, cid, slot)
    full_booking(ep, slot)


def sc_book_after_availability_busy(ep: Ep) -> None:
    slot = slot_and_ask(ep, "book", stated=None)
    cid, alts = check(ep, slot, "busy")
    if not alts:  # this scenario needs something to pick
        raise RetryScenario()
    say_busy(ep, cid, alts)
    book_pick(ep, pick_alt(ep, alts))


def sc_book_direct_details_first(ep: Ep) -> None:
    ep.on_file = False  # the caller introduces themselves, so they are new
    slot = T.random_slot(ep.rng, ep.now)
    if ep.rng.random() < 0.5:
        greet(ep)
    when, _ = T.caller_phrase(slot, ep.now, ep.rng, ep.lang)
    ep.vars["when"] = when
    ep.set_slot("slot", slot)
    ev = ep.C("caller_want_book_at", act="ask_slot", at=0.0 if not ep.events else None)
    set_intent(ep, ev, slot, book_intent=True)
    ep.C("caller_give_all", act="give_details", gap=ep.gap(0.15, 0.08))
    ep.heard_name = ep.heard_email = True
    cid, _ = check(ep, slot, "free", pre_key="elle_check_first")
    after_free(ep, cid, slot)
    full_booking(ep, slot, sms=ep.rng.random() < 0.6)


def sc_book_declined_before_write(ep: Ep) -> None:
    """The caller backs out after hearing the time is free: nothing is booked
    and get_caller_details is never called. A new caller backs out while being
    asked for their name; a caller on file at the offer (once they have
    accepted, Elle books at once, so there is no later moment)."""
    slot = slot_and_ask(ep, "book", stated=False if ep.on_file else None)
    cid, _ = check(ep, slot, "free")
    if ep.on_file:
        ep.claim("elle_free", cid)
    else:
        after_free(ep, cid, slot)
        ep.E("elle_ask_name", act="ask_name")
    ep.C("caller_decline_book", act="decline")
    ep.intent = ep.intent_slot = None
    ep.E("elle_decline_ack", act="ack")
    bye(ep, 0.6)


def _other_email(ep: Ep) -> str:
    local, dom = ep.p.email.split("@")
    doms = [d for d in banks.EMAIL_DOMAINS if d != dom] or [dom]
    return f"{local}@{ep.rng.choice(doms)}"


def _after_email(ev) -> float:
    """A moment inside Elle's post-booking line just after she has said the email
    (its last word is the top-level domain), so the caller has heard what they correct."""
    words = ev["text"].split()
    k = max(i for i, w in enumerate(words) if w.strip(".,").lower() in ("com", "org", "net"))
    return ev["t"] + ev["dur"] * (k + 1) / len(words) + 0.05


def fix_email(ep: Ep, slot: datetime, said, cut: bool) -> None:
    """The caller corrects the email after hearing what was booked. Elle calls
    get_caller_details again (the system now has the corrected address, HEARD)
    and book_intro_call for the same start; the system treats that as an update."""
    ep.p.email = _other_email(ep)
    ep.heard_email = True
    ep.refresh_vars()
    if cut:
        fix = ep.interrupt(said, "caller_cut_fix_email", at=_after_email(said), act="correction")
    else:
        fix = ep.C("caller_cut_fix_email", act="correction")
    ep.intent, ep.intent_slot = fix, slot  # the correction is what licenses the repeat booking
    d = caller_details(ep, pre_key="elle_email_fix_pre")
    b = book(ep, slot, d, content=res_updated(ep.p.email))
    booked(ep, b, d, slot, key="elle_email_updated", act="booked_update", backchannel=False)


def sc_book_email_corrected(ep: Ep) -> None:
    slot = slot_and_ask(ep, "book", stated=None)
    cid, _ = check(ep, slot, "free")
    after_free(ep, cid, slot)
    details(ep)
    d = caller_details(ep)
    b = book(ep, slot, d)
    said = booked(ep, b, d, slot, backchannel=False)
    fix_email(ep, slot, said, cut=False)
    bye(ep)


def sc_book_missing_email(ep: Ep) -> None:
    ep.on_file = False
    slot = slot_and_ask(ep, "book")
    cid, _ = check(ep, slot, "free")
    ep.claim("elle_free", cid)
    ev = ep.C("caller_push_book", act="accept")  # gives a name and pushes to book, no email
    set_intent(ep, ev, slot)
    ep.heard_name = True
    d = caller_details(ep, quiet=True)  # MISSING: email. Ask for it.
    ep.claim("elle_need_email", d, act="ask_email")
    r = ep.rng.random()
    if r >= 0.5:
        ep.C("caller_email_hesitate", act="hesitate")
        ep.E("elle_email_why", act="explain")
        if r >= 0.75:
            ep.C("caller_refuse_email", act="decline")
            ep.intent = ep.intent_slot = None
            ep.E("elle_no_email_fallback", act="ack")
            bye(ep, 0.7)
            return
    ep.C("caller_give_email", act="give_email")
    ep.heard_email = True
    full_booking(ep, slot, sms=False)


def _bad_email(ep: Ep) -> str:
    """A spoken address that can't be valid: no TLD, no 'at', or cut off."""
    local, dom = ep.p.email.split("@")
    local_sp, dom_sp = local.replace(".", " dot "), dom.replace(".", " dot ")
    return ep.rng.choice([f"{local_sp} at {dom.split('.')[0]}", f"{local_sp} {dom_sp}", f"{local_sp} at"])


def sc_book_invalid_email(ep: Ep) -> None:
    ep.on_file = False
    slot = slot_and_ask(ep, "book", stated=None)
    cid, _ = check(ep, slot, "free")
    after_free(ep, cid, slot)
    ep.E("elle_ask_name", act="ask_name")
    ep.C("caller_give_name", act="give_name")
    ep.heard_name = True
    ep.E("elle_ask_email", act="ask_email")
    ep.vars["bad_email_sp"] = _bad_email(ep)
    ep.C("caller_bad_email", act="give_email_invalid")
    # Elle cannot judge an address she only heard: the details result tells her.
    d = caller_details(ep, content=RES_INVALID_EMAIL, quiet=True)
    ep.claim("elle_email_clarify", d, act="clarify")
    ep.C("caller_email_fix", act="give_email")
    ep.heard_email = True
    full_booking(ep, slot, sms=False)


def sc_book_past_time(ep: Ep) -> None:
    # Pin `now` to a weekday late morning / afternoon so "this morning at nine" is past.
    n = ep.now
    while n.weekday() >= 5:
        n += timedelta(days=1)
    n = n.replace(hour=ep.rng.randint(11, 15), minute=ep.rng.randrange(0, 60))
    ep.now = n
    past_h = ep.rng.randint(9, n.hour - 1)
    past = T.local_dt(n.date(), past_h, ep.rng.choice([0, 30]))
    ep.vars["past_clock"] = T.clock_en(past.hour, past.minute, ep.rng)
    ep.vars["past_slot"] = ep.vars["past_clock"] + (" this morning" if past.hour < 12 else " today")
    direct = opening(ep, "book")
    ep.vars["when"] = T.render_ref(T.TimeRef("today", past.hour, past.minute), ep.rng)
    if not direct:
        ep.E("elle_ask_time", act="ask_time")
    ep.C("caller_ask_avail", act="ask_slot_past", at=0.0 if not ep.events else None)
    slot = T.random_slot(ep.rng, ep.now, max_days=7)
    ep.set_slot("slot", slot)
    ep.E("elle_past", act="propose")["refers"] = {"start": T.iso(slot)}
    set_intent(ep, ep.C("caller_ok_alt", act="accept"), slot)
    cid, _ = check(ep, slot, "free")
    after_free(ep, cid, slot)
    full_booking(ep, slot)


def _conflict(ep: Ep, result_text: str, key: str) -> None:
    slot = slot_and_ask(ep, "book", stated=None)
    c1, _ = check(ep, slot, "free")
    after_free(ep, c1, slot)
    details(ep)
    d1 = caller_details(ep)
    b1 = book(ep, slot, d1, content=result_text)
    alt = other_slot(ep, [slot], near=slot)
    ep.set_slot("alt1", alt)
    ep.claim(key, b1)["refers"] = {"start": T.iso(alt)}
    set_intent(ep, ep.C("caller_ok_check_alt", act="accept"), alt)
    c2, _ = check(ep, alt, "free")
    after_free(ep, c2, alt)
    full_booking(ep, alt, sms=False)


def sc_book_slot_taken_race(ep: Ep) -> None:
    _conflict(ep, RES_TAKEN, "elle_slot_taken")


def sc_book_busy_at_write(ep: Ep) -> None:
    _conflict(ep, RES_BUSY_AT_WRITE, "elle_busy_at_write")


def sc_book_write_error(ep: Ep) -> None:
    slot = slot_and_ask(ep, "book", stated=None)
    c1, _ = check(ep, slot, "free")
    after_free(ep, c1, slot)
    details(ep)
    d = caller_details(ep)
    b = book(ep, slot, d, content=RES_WRITE_ERROR)
    # No follow-up call: the system notes the failed request from the transcript.
    ep.claim("elle_write_error", b, also=[d])
    bye(ep)


def sc_chat_then_book(ep: Ep) -> None:
    """Small talk about the caller's company and role, then maybe a booking.
    Nothing is saved by Elle: the system keeps such facts from the transcript."""
    greet(ep)
    ep.C("caller_reveal_company", act="reveal")
    ep.E("elle_after_company", act="chat")
    ep.C("caller_reveal_role", act="reveal")
    ep.E("elle_after_role", act="pitch")
    if ep.rng.random() < 0.3:
        ep.C("caller_chitchat", act="chat")
        ep.E("elle_chitchat", act="chat")
    if ep.rng.random() < 0.55:
        slot = T.random_slot(ep.rng, ep.now)
        ask_for_slot(ep, slot, direct=True)
        cid, _ = check(ep, slot, "free")
        after_free(ep, cid, slot)
        full_booking(ep, slot, sms=False)
    else:
        hold_off(ep)
        bye(ep, 0.6)


def sc_book_then_when_again_memory(ep: Ep) -> None:
    slot = slot_and_ask(ep, "book", stated=None)
    c1, _ = check(ep, slot, "free")
    after_free(ep, c1, slot)
    b = full_booking(ep, slot, sms=False, end=False)
    ep.C("caller_when_again", act="ask_recall")
    ep.claim("elle_when_again", b)
    bye(ep)


def sc_book_then_when_again_lookup(ep: Ep) -> None:
    slot = slot_and_ask(ep, "book", stated=None)
    c1, _ = check(ep, slot, "free")
    after_free(ep, c1, slot)
    full_booking(ep, slot, sms=ep.rng.random() < 0.4, end=False)
    ep.C("caller_ask_bookings", act="ask_bookings", gap=ep.gap(0.8, 0.3, 0.2, 2.0))
    rows = [(ep.rng.randint(12, 480), ep.p.name, slot, "pending", "")]
    cid, _ = ep.read("get_my_bookings", {}, res_bookings(rows), lat_read(ep), pre_key="elle_bookings_pre")
    ep.set_slot("slot", slot)
    ep.claim("elle_bookings_one", cid)
    bye(ep)


def sc_returning_caller(ep: Ep) -> None:
    """A caller on file: greeted by name, never asked for name or email, and
    booked with the details the system has (ON FILE)."""
    ep.on_file = True
    if ep.rng.random() < 0.5:
        n = ep.rng.choice([0, 1])
        cid, rows = bookings_read(ep, n)
        if rows:
            say_bookings(ep, cid, rows)
            return
        # No booking yet: Elle offers to make one and the caller takes her up on it.
        ep.claim("elle_bookings_none", cid)
        ep.C("caller_yes_book", act="intent")  # wants a call; no time named yet
        slot = T.random_slot(ep.rng, ep.now)
        ask_for_slot(ep, slot, direct=False, stated=None)
    else:
        slot = slot_and_ask(ep, "book", stated=None)
    c1, _ = check(ep, slot, "free")
    after_free(ep, c1, slot)
    full_booking(ep, slot)


def sc_barge_in_then_book(ep: Ep) -> None:
    greet(ep)
    ep.C("caller_ask_about", act="ask_about")
    pitch = ep.E("elle_interrupted_long", act="pitch")
    ep.interrupt(pitch, "caller_interrupt", act="barge_in")
    slot = T.random_slot(ep.rng, ep.now)
    ask_for_slot(ep, slot, direct=False)
    cid, alts = check(ep, slot, "free" if ep.rng.random() < 0.7 else "busy")
    if ep.results[cid]["content"].startswith("FREE"):
        after_free(ep, cid, slot)
        full_booking(ep, slot)
    else:
        say_busy(ep, cid, alts)
        hold_off(ep)


def sc_es_switch_midcall(ep: Ep) -> None:
    ep.lang_label = "en+es"
    greet(ep)
    ep.C("caller_switch_es", act="switch_lang")
    ep.set_lang("es")
    ep.E("elle_switch_ack", act="ask_time")
    slot = T.random_slot(ep.rng, ep.now)
    when, _ = T.caller_phrase(slot, ep.now, ep.rng, "es")
    ep.vars["when"] = when
    ep.set_slot("slot", slot)
    ep.C("caller_ask_avail", act="ask_slot")
    cid, _ = check(ep, slot, "free")
    after_free(ep, cid, slot)
    full_booking(ep, slot, sms=ep.rng.random() < 0.5)


# --- scenarios: intricate interruptions (family "interrupt") ----------------------------------
#
# Full duplex means the caller can talk at any moment, including while a tool
# call is in flight or while Elle is reading out its result. Each scenario
# below teaches one rule:
#   * a barge-in takes the floor: Elle stops within a few hundred ms;
#   * a correction makes the earlier result stale: Elle never reports it, she
#     checks the new time (a second read while the first may still be pending);
#   * a correction of the email while Elle says what she booked is fixed with
#     a fresh get_caller_details and a repeat booking for the same start (an
#     update), never from memory;
#   * a write that already went out can't be taken back by voice: Elle says so
#     honestly once the result lands (tentative; text to cancel);
#   * a backchannel ("mm-hm") is not a barge-in: Elle keeps going;
#   * a side question during a slow read is answered without guessing the result;
#   * when both start at once, Elle yields.

def _new_when2(ep: Ep, slot: datetime) -> datetime:
    slot2 = other_slot(ep, [slot], near=slot)
    when2, _ = T.caller_phrase(slot2, ep.now, ep.rng, ep.lang)
    ep.vars["when2"] = when2
    ep.set_slot("slot2", slot2)
    return slot2


def _report_and_continue(ep: Ep, cid: str, slot: datetime, alts: List[datetime]) -> None:
    """Report check `cid` for `slot`, then book (free), pick an alternative (busy) or wrap up."""
    ep.set_slot("slot", slot)
    if ep.results[cid]["content"].startswith("FREE"):
        offer = ep.claim("elle_free", cid)
        if ep.rng.random() < 0.75:
            accept_offer(ep, offer, slot)
            full_booking(ep, slot)
            return
        hold_off(ep)
    else:
        say_busy(ep, cid, alts)
        if alts:
            book_pick(ep, pick_alt(ep, alts))
            return
        hold_off(ep)
    bye(ep, 0.5)


def _last_elle(ep: Ep) -> Optional[Dict]:
    sp = [e for e in ep.events if e["type"] == "speech" and e["ch"] == "elle"]
    return max(sp, key=lambda e: e["t"] + e["dur"]) if sp else None


def sc_interrupt_claim_correct_time(ep: Ep) -> None:
    """The caller cuts into Elle's answer with a different time."""
    slot = slot_and_ask(ep, "check")
    c1, alts = check(ep, slot, "free" if ep.rng.random() < 0.55 else "busy")
    if ep.results[c1]["content"].startswith("FREE"):
        ev = ep.claim("elle_free", c1)
    else:
        ev = say_busy(ep, c1, alts)
    slot2 = _new_when2(ep, slot)
    ep.interrupt(ev, "caller_cut_correct_time", frac=ep.rng.uniform(0.3, 0.7), act="correction")
    c2, alts2 = check(ep, slot2, "free" if ep.rng.random() < 0.65 else "busy", pre_key="elle_yield_ack")
    _report_and_continue(ep, c2, slot2, alts2)


def sc_interrupt_wait_correction(ep: Ep) -> None:
    """The caller talks over Elle *while the read is in flight* to change the
    time. The first result is stale when it lands; Elle checks the new time."""
    slot = slot_and_ask(ep, "check")
    slot2 = _new_when2(ep, slot)

    def cut(e: Ep, call_t: float, res_t: float) -> None:
        host = _last_elle(e)
        t = call_t + e.rng.uniform(0.15, 0.4)
        if host is None or host["t"] + host["dur"] - t < 0.45:
            host = e.say("elle", e.line("elle_filler"), at=e.floor() + e.rng.uniform(0.25, 0.5), act="filler")
            t = host["t"] + e.rng.uniform(0.3, 0.6) * host["dur"]
        e.interrupt(host, "caller_cut_correct_time", at=t, act="mind_change")

    check(ep, slot, "free" if ep.rng.random() < 0.5 else "busy",
          latency=ep.rng.uniform(1.6, 3.2), during=cut, fill=False)
    c2, alts = check(ep, slot2, "free" if ep.rng.random() < 0.65 else "busy", pre_key="elle_yield_ack")
    _report_and_continue(ep, c2, slot2, alts)


def sc_interrupt_booked_fix_email(ep: Ep) -> None:
    """The caller cuts into Elle's post-booking sentence, right after hearing
    the email, to fix it. Elle stops, gets the corrected details from the
    system and books the same start again (an update)."""
    slot = slot_and_ask(ep, "book", stated=None)
    c1, _ = check(ep, slot, "free")
    after_free(ep, c1, slot)
    details(ep)
    d = caller_details(ep)
    b = book(ep, slot, d)
    said = booked(ep, b, d, slot, backchannel=False)
    fix_email(ep, slot, said, cut=True)
    bye(ep)


def sc_interrupt_offer_eager_yes(ep: Ep) -> None:
    """The caller says yes over the last words of Elle's offer ("...want me to
    pencil you in?"). That yes counts; Elle goes on to book once they finish."""
    slot = slot_and_ask(ep, "book")
    c1, _ = check(ep, slot, "free")
    offer = ep.claim("elle_free", c1)
    ev = ep.say("caller", ep.line("caller_yes_book"),
                at=offer["t"] + offer["dur"] - ep.rng.uniform(0.15, 0.5), act="accept")
    set_intent(ep, ev, slot)
    full_booking(ep, slot, sms=ep.rng.random() < 0.5)


def sc_interrupt_cancel_during_write(ep: Ep) -> None:
    """The caller says 'wait, don't book it' after the write already went out.
    Elle can't unsend it: she waits for the result, then says honestly that it
    went in as a tentative request and how to cancel (text this number)."""
    slot = slot_and_ask(ep, "book", stated=None)
    c1, _ = check(ep, slot, "free")
    after_free(ep, c1, slot)
    details(ep)
    d = caller_details(ep, quiet=True)
    pre = ep.E("elle_book_pre", act="pre_call",
               at=max(ep.floor() + ep.gap(0.2, 0.08), ep.results[d]["t"] + 0.08))
    at = pre["t"] + ep.rng.uniform(0.05, min(0.5, pre["dur"] * 0.5))
    cid = ep.call("book_intro_call", {"start": T.iso(slot)}, at)
    lat = ep.rng.uniform(1.3, 2.6)
    t = at + ep.rng.uniform(0.2, 0.6)
    if pre["t"] + pre["dur"] - t > 0.3:
        ep.interrupt(pre, "caller_cut_cancel", at=t, act="cancel")
    else:
        ep.say("caller", ep.line("caller_cut_cancel"), at=t, act="cancel")
    if at + lat - ep.floor() > 0.9:
        ep.E("elle_hold_on_write", act="filler", gap=ep.gap(0.25, 0.08))
    ep.result(cid, res_done(ep.p.email), lat)
    ep.gate_booking(cid, ep.intent, d)
    booked(ep, cid, d, slot, key="elle_write_landed_cancel", backchannel=False)
    if ep.rng.random() < 0.5:
        ep.C("caller_keep_it", act="keep")
        ep.E("elle_keep_ack", act="ack")
    else:
        ep.C("caller_will_text_cancel", act="ack")
        ep.E("elle_text_cancel_ack", act="ack")
    bye(ep, 0.7)


def sc_interrupt_pick_alt_early(ep: Ep) -> None:
    """Elle lists alternatives; the caller either grabs the first one before
    she finishes (a barge-in: she stops) or just says 'mm-hm' along the way (a
    backchannel: she keeps going)."""
    slot = slot_and_ask(ep, "book", stated=None)
    cid, alts = check(ep, slot, "busy")
    if len(alts) < 2:
        raise RetryScenario()
    ep.set_slot("slot", slot)
    ev = ep.claim("elle_busy", cid)
    if ep.rng.random() < 0.55:
        pick = ep.interrupt(ev, "caller_cut_pick_alt1", frac=ep.rng.uniform(0.72, 0.88), act="pick")
        set_intent(ep, pick, alts[0])
        ep.set_slot("slot", alts[0])
        book_pick(ep, alts[0])
    else:
        ep.backchannel(ev)
        book_pick(ep, pick_alt(ep, alts))


def sc_side_question_during_wait(ep: Ep) -> None:
    """A slow read; the caller asks something unrelated meanwhile. Elle answers
    it (no availability guess), keeps the wait covered, then reports."""
    slot = slot_and_ask(ep, "check")
    qs = banks.BANKS[ep.lang]["caller_side_question"]
    ans = banks.BANKS[ep.lang]["elle_side_answer"]
    i = ep.rng.randrange(len(qs))

    def ask(e: Ep, call_t: float, res_t: float) -> None:
        e.say("caller", qs[i], at=max(call_t + e.rng.uniform(0.2, 0.7), e.end["caller"] + 0.3), act="side_question")
        e.say("elle", ans[i], act="answer", gap=e.gap(0.3, 0.1))

    outcome = "free" if ep.rng.random() < 0.6 else "busy"
    cid, alts = check(ep, slot, outcome, latency=ep.rng.uniform(2.6, 4.5), during=ask)
    _report_and_continue(ep, cid, slot, alts)


def sc_talk_collision(ep: Ep) -> None:
    """Both start talking at the same moment, at the greeting and again after
    an offer. Elle yields each time; the caller keeps the floor."""
    g = greet(ep, at=round(ep.rng.uniform(0.2, 0.5), 2))
    ep.interrupt(g, "caller_open_book", at=g["t"] + ep.rng.uniform(0.0, 0.3), act="intent")
    slot = T.random_slot(ep.rng, ep.now)
    ask_for_slot(ep, slot, direct=False)
    cid, _ = check(ep, slot, "free")
    ep.claim("elle_free", cid)
    t0 = ep.floor() + ep.gap(0.95, 0.2, 0.6, 1.4)
    e = ep.say("elle", ep.line("elle_followup_offer"), at=t0, act="offer")
    set_intent(ep, ep.interrupt(e, "caller_yes_book", at=t0 + ep.rng.uniform(0.0, 0.2), act="accept"), slot)
    if ep.rng.random() < 0.3:
        ep.E("elle_after_collision", act="ack", gap=ep.gap(0.25, 0.08))
    full_booking(ep, slot)


# --- negatives (labelled; the evaluator must flag these) --------------------------------------

def sc_neg_premature_availability_claim(ep: Ep) -> None:
    ep.negative_kind = "premature_claim"
    slot = slot_and_ask(ep, "check")
    outcome = "free" if ep.rng.random() < 0.6 else "busy"
    lat = ep.rng.uniform(1.1, 2.8)
    pre = ep.E("elle_check_pre", act="pre_call") if ep.rng.random() < 0.5 else None
    at = (pre["t"] + 0.1) if pre else ep.floor() + ep.gap(0.25, 0.1)
    content, alts = outcome_content(ep, slot, outcome)
    cid = ep.call("check_availability", {"start": T.iso(slot)}, at)
    # WRONG: the answer is stated while the read is still in flight.
    bad = ep.say("elle", ep.line("elle_premature_free"), at=max(ep.end["elle"] + 0.1, at + 0.15),
                 act="premature_claim")
    ep.claims.append((bad, cid))
    if bad["t"] >= at + lat:  # keep it genuinely premature
        lat = bad["t"] - at + ep.rng.uniform(0.3, 1.0)
    ep.result(cid, content, lat)
    if outcome == "busy" and alts:
        ep.claim("elle_premature_retract", cid)
        hold_off(ep)
    elif outcome == "busy":
        ep.claim("elle_busy_none", cid)
    else:
        hold_off(ep)
    bye(ep, 0.4)


def sc_neg_premature_bookings_claim(ep: Ep) -> None:
    ep.negative_kind = "premature_claim"
    rows = _booking_rows(ep, 1)
    ep.C("caller_ask_bookings", act="ask_bookings", at=0.0)
    at = ep.floor() + ep.gap(0.25, 0.1)
    cid = ep.call("get_my_bookings", {}, at)
    ep.set_slot("slot", rows[0][2])
    bad = ep.say("elle", ep.line("elle_premature_bookings"), at=at + 0.15, act="premature_claim")
    ep.claims.append((bad, cid))
    ep.result(cid, res_bookings(rows), ep.rng.uniform(1.0, 2.5))
    ep.C("caller_bookings_thanks", act="thanks")
    bye(ep, 0.5)


def sc_neg_book_without_intent(ep: Ep) -> None:
    ep.negative_kind = "write_without_intent"
    ep.on_file = True  # details are known, so the missing intent is the only fault
    slot = slot_and_ask(ep, "check")
    c1, _ = check(ep, slot, "free")
    ep.claim("elle_free_noop", c1)
    # WRONG: the caller only asked whether he is free; nobody asked to book.
    d = caller_details(ep)
    b = book(ep, slot, d, gated=False)
    booked(ep, b, d, slot)
    bye(ep)


def sc_neg_dead_air_wait(ep: Ep) -> None:
    ep.negative_kind = "dead_air"
    slot = slot_and_ask(ep, "check")
    pre = ep.E("elle_check_pre", act="pre_call") if ep.rng.random() < 0.5 else None
    at = ep.floor() + ep.rng.uniform(0.1, 0.3)
    # WRONG: the read is slow and nobody says anything while it runs.
    outcome = "free" if ep.rng.random() < 0.6 else "busy"
    content, alts = outcome_content(ep, slot, outcome)
    cid = ep.call("check_availability", {"start": T.iso(slot)}, at)
    ep.result(cid, content, ep.rng.uniform(1.8, 3.0))
    if outcome == "free":
        ep.claim("elle_free", cid)
    else:
        ep.set_slot("slot", slot)
        say_busy(ep, cid, alts)
    hold_off(ep)


class RetryScenario(Exception):
    """Raised by a scenario when the random draw can't produce its shape; the
    generator retries with a derived seed."""


# --- registry --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Scenario:
    name: str
    fn: Callable[[Ep], None]
    family: str
    weight: float
    langs: Tuple[str, ...] = ("en",)
    es_share: float = 0.0


EN_ES = ("en", "es")

SCENARIOS: List[Scenario] = [
    # reads
    Scenario("availability_free", sc_availability_free, "read_basic", 6, EN_ES, 0.2),
    Scenario("availability_busy_alternatives", sc_availability_busy_alternatives, "read_basic", 6, EN_ES, 0.2),
    Scenario("bookings_none", sc_bookings_none, "bookings_read", 3, EN_ES, 0.2),
    Scenario("bookings_one", sc_bookings_one, "bookings_read", 4, EN_ES, 0.2),
    Scenario("bookings_two", sc_bookings_two, "bookings_read", 3, EN_ES, 0.15),
    Scenario("availability_slow_filler", sc_availability_slow_filler, "wait_handling", 6, EN_ES, 0.15),
    Scenario("bookings_slow_filler", sc_bookings_slow_filler, "wait_handling", 3),
    Scenario("availability_caller_keeps_talking", sc_availability_caller_keeps_talking, "wait_handling", 5),
    Scenario("availability_mind_change_mid_wait", sc_availability_mind_change_mid_wait, "mind_change", 5),
    Scenario("availability_unknown_recovery", sc_availability_unknown_recovery, "read_error", 4),
    Scenario("availability_timeout", sc_availability_timeout, "read_error", 3),
    # writes / mixed
    Scenario("book_after_availability_free", sc_book_after_availability_free, "booking", 9, EN_ES, 0.2),
    Scenario("book_after_availability_busy", sc_book_after_availability_busy, "booking", 7, EN_ES, 0.15),
    Scenario("book_direct_details_first", sc_book_direct_details_first, "booking", 4, EN_ES, 0.15),
    Scenario("book_declined_before_write", sc_book_declined_before_write, "write_guard", 4),
    Scenario("book_email_corrected", sc_book_email_corrected, "write_guard", 3),
    Scenario("book_missing_email", sc_book_missing_email, "write_guard", 4),
    Scenario("book_invalid_email", sc_book_invalid_email, "write_guard", 3),
    Scenario("book_past_time", sc_book_past_time, "write_guard", 3),
    Scenario("time_change_after_booking", sc_time_change_after_booking, "mind_change", 3),
    Scenario("book_slot_taken_race", sc_book_slot_taken_race, "slot_conflict", 4),
    Scenario("book_busy_at_write", sc_book_busy_at_write, "slot_conflict", 3),
    Scenario("book_write_error", sc_book_write_error, "write_error", 2),
    Scenario("chat_then_book", sc_chat_then_book, "small_talk", 5),
    Scenario("book_then_when_again_memory", sc_book_then_when_again_memory, "followup", 3),
    Scenario("book_then_when_again_lookup", sc_book_then_when_again_lookup, "followup", 3),
    Scenario("returning_caller", sc_returning_caller, "identity", 3),
    Scenario("barge_in_then_book", sc_barge_in_then_book, "overlap", 3),
    Scenario("es_switch_midcall", sc_es_switch_midcall, "bilingual", 2),
    # intricate interruptions (tool calls in flight, results being read out)
    Scenario("interrupt_claim_correct_time", sc_interrupt_claim_correct_time, "interrupt", 4),
    Scenario("interrupt_wait_correction", sc_interrupt_wait_correction, "interrupt", 4),
    Scenario("interrupt_booked_fix_email", sc_interrupt_booked_fix_email, "interrupt", 3),
    Scenario("interrupt_offer_eager_yes", sc_interrupt_offer_eager_yes, "interrupt", 2),
    Scenario("interrupt_cancel_during_write", sc_interrupt_cancel_during_write, "interrupt", 3),
    Scenario("interrupt_pick_alt_early", sc_interrupt_pick_alt_early, "interrupt", 3),
    Scenario("side_question_during_wait", sc_side_question_during_wait, "interrupt", 3),
    Scenario("talk_collision", sc_talk_collision, "interrupt", 3),
    # negatives
    Scenario("neg_premature_availability_claim", sc_neg_premature_availability_claim, "negative", 3, EN_ES, 0.15),
    Scenario("neg_premature_bookings_claim", sc_neg_premature_bookings_claim, "negative", 1),
    Scenario("neg_book_without_intent", sc_neg_book_without_intent, "negative", 2),
    Scenario("neg_dead_air_wait", sc_neg_dead_air_wait, "negative", 2),
]

BY_NAME: Dict[str, Scenario] = {s.name: s for s in SCENARIOS}
FAMILIES = sorted({s.family for s in SCENARIOS})
