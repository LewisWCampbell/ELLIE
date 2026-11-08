"""Scenario grammar: reusable beats composed into named scenarios.

A *beat* is a short exchange (opening, ask for a slot, check availability,
collect details, read back, book, SMS consent, goodbye). A *scenario* is a
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

import elle_time as T
from episode import OWNER, Ep, Person, spoken_email

# --- production result strings (src/booking/*.ts, src/sms.ts, twilioBridge.ts) ----

UNKNOWN_ADVICE = (f"Don't tell the caller whether the time is free. Offer to book it as a request "
                  f"pending {OWNER}'s confirmation, or to try again in a moment.")


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

def opening(ep: Ep, intent: str = "book") -> bool:
    """Optional greeting and caller opener. Returns True when the caller's
    opener already contains the availability question (direct ask)."""
    greet = ep.rng.random() < 0.55
    g = None
    if greet:
        g = ep.E("elle_greet", act="greeting", at=round(ep.rng.uniform(0.2, 0.6), 2))
    if ep.rng.random() < 0.3:
        return True  # caller goes straight to the time question
    key = "caller_open_book" if intent == "book" else "caller_open_check"
    if g is not None and g["dur"] > 2.5 and ep.rng.random() < 0.2:
        ep.interrupt(g, key, frac=ep.rng.uniform(0.55, 0.85), act="intent")  # caller talks over the greeting
    else:
        ep.C(key, act="intent", at=0.0 if not greet else None)
    return False


def ask_for_slot(ep: Ep, slot: datetime, direct: bool, book_intent: bool = False) -> None:
    when, _ = T.caller_phrase(slot, ep.now, ep.rng, ep.lang)
    ep.vars["when"] = when
    ep.set_slot("slot", slot)
    at = 0.0 if direct and not ep.events else None
    if not direct:
        ep.E("elle_ask_time", act="ask_time")
    key = "caller_want_book_at" if book_intent and ep.rng.random() < 0.4 else "caller_ask_avail"
    ep.C(key, act="ask_slot", at=at)


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


def say_busy(ep: Ep, cid: str, alts: List[datetime]) -> None:
    key = "elle_busy" if len(alts) >= 2 else "elle_busy_one" if alts else "elle_busy_none"
    ep.claim(key, cid)


def pick_alt(ep: Ep, alts: List[datetime]) -> datetime:
    if len(alts) >= 2 and ep.rng.random() < 0.4:
        ep.C("caller_pick_alt2", act="pick")
        chosen = alts[1]
    else:
        ep.C("caller_pick_alt1", act="pick")
        chosen = alts[0]
    ep.set_slot("slot", chosen)
    return chosen


def quiet_save(ep: Ep, category: str, fact: str, anchor=None) -> str:
    """save_user_fact, emitted silently: mid-way through Elle's anchor line, or
    right as the floor frees up. Elle never mentions it."""
    if anchor is not None and anchor.get("ch") == "elle":
        at = anchor["t"] + ep.rng.uniform(0.1, 0.6) * anchor["dur"]
    else:
        at = ep.write_at()
    cid = ep.call("save_user_fact", {"category": category, "fact": fact}, at)
    ep.result(cid, RES_SAVED, lat_save(ep))
    return cid


def details(ep: Ep, have_name: bool = False, have_email: bool = False, save_name: Optional[bool] = None) -> None:
    if save_name is None:
        save_name = ep.rng.random() < 0.5
    if not have_name and not have_email and ep.rng.random() < 0.25:
        ep.E("elle_ask_name", act="ask_name")
        ep.C("caller_give_all", act="give_details")
        have_name = have_email = True
        if save_name:
            nxt = None
            quiet_save(ep, "name", f"Their name is {ep.p.name}.", anchor=nxt)
        return
    if not have_name:
        ep.E("elle_ask_name", act="ask_name")
        ep.C("caller_give_name", act="give_name")
    if not have_email:
        ev = ep.E("elle_ask_email", act="ask_email")
        if save_name and not have_name:
            quiet_save(ep, "name", f"Their name is {ep.p.name}.", anchor=ev)
        ep.C("caller_give_email", act="give_email")


def readback(ep: Ep, slot: datetime, key: str = "elle_readback", claim_cid: Optional[str] = None):
    ep.set_slot("slot", slot)
    refers = {"start": T.iso(slot), "name": ep.p.name, "email": ep.p.email}
    if claim_cid is not None:
        rb = ep.claim(key, claim_cid, act="readback")
    else:
        rb = ep.E(key, act="readback")
    rb["refers"] = refers
    return rb


def confirm(ep: Ep):
    return ep.C("caller_confirm", act="confirm", gap=ep.gap(0.3, 0.1))


def book(ep: Ep, slot: datetime, rb, cf, content: Optional[str] = None,
         latency: Optional[float] = None, gated: bool = True) -> str:
    pre = ep.E("elle_book_pre", act="pre_call") if ep.rng.random() < 0.7 else None
    at = ep.write_at() if pre is None else pre["t"] + ep.rng.uniform(0.05, min(0.5, pre["dur"] * 0.5))
    args = {"name": ep.p.name, "email": ep.p.email, "start": T.iso(slot)}
    if ep.rng.random() < 0.35:
        args["topic"] = ep.p.topic
    cid = ep.call("book_intro_call", args, at)
    ep.result(cid, content if content is not None else res_done(ep.p.email), latency or lat_write(ep))
    if gated:
        ep.gate(cid, rb, cf)
    return cid


def booked(ep: Ep, cid: str, slot: datetime) -> None:
    ep.set_slot("slot", slot)
    ev = ep.claim("elle_booked", cid)
    ep.backchannel(ev)


def sms_consent(ep: Ep, yes: Optional[bool] = None) -> None:
    if yes is None:
        yes = ep.rng.random() < 0.6
    a = ep.E("elle_sms_offer_a", act="sms_offer_intro", gap=ep.gap(0.5, 0.15))
    ep.backchannel(a)
    b = ep.E("elle_sms_offer_b", act="sms_offer", gap=ep.gap(0.2, 0.08))
    ep.backchannel(b)
    if yes:
        cf = ep.C("caller_sms_yes", act="confirm")
        cid = quiet_save(ep, "sms_consent", "Verbally consented on a call to receive booking texts")
        ep.gate(cid, b, cf)
        ep.claim("elle_sms_yes_after", cid)
    else:
        ep.C("caller_sms_no", act="decline")
        ep.E("elle_sms_no_after", act="ack")


def bye(ep: Ep, p: float = 0.8) -> None:
    if ep.rng.random() < p:
        ep.C("caller_bye", act="bye")
        ep.E("elle_bye", act="bye")


def full_booking(ep: Ep, slot: datetime, check_cid: Optional[str] = None,
                 sms: Optional[bool] = None, end: bool = True) -> str:
    """details -> readback -> yes -> book -> tentative -> (sms) -> (bye)."""
    details(ep)
    rb = readback(ep, slot)
    cf = confirm(ep)
    cid = book(ep, slot, rb, cf)
    booked(ep, cid, slot)
    if sms is None:
        sms = ep.rng.random() < 0.65
    if sms:
        sms_consent(ep)
    if end:
        bye(ep)
    return cid


def slot_and_ask(ep: Ep, intent: str = "book", book_intent: bool = False,
                 slot: Optional[datetime] = None) -> datetime:
    slot = slot or T.random_slot(ep.rng, ep.now)
    direct = opening(ep, intent)
    ask_for_slot(ep, slot, direct, book_intent)
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
        ep.C("caller_hold_off", act="decline")
        ep.E("elle_hold_off_ack", act="ack")
        bye(ep, 0.5)


def sc_availability_busy_alternatives(ep: Ep) -> None:
    slot = slot_and_ask(ep, "check")
    cid, alts = check(ep, slot, "busy")
    say_busy(ep, cid, alts)
    if alts:
        chosen = pick_alt(ep, alts)
        ep.vars["pick"] = T.elle_short(chosen, ep.now, ep.lang)
        ep.E("elle_alt_offer", act="offer")
    ep.C("caller_hold_off", act="decline")
    ep.E("elle_hold_off_ack", act="ack")
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
        ep.E("elle_greet", act="greeting", at=round(ep.rng.uniform(0.2, 0.6), 2))
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
        ep.C("caller_pick_booking", act="pick")
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
            ep.C("caller_yes_book", act="accept")
            full_booking(ep, slot)
            return
        ep.C("caller_hold_off", act="decline")
        ep.E("elle_hold_off_ack", act="ack")
    else:
        say_busy(ep, cid, alts)
        ep.C("caller_hold_off", act="decline")
        ep.E("elle_hold_off_ack", act="ack")
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
        ep.claim("elle_free", cid)
        ep.C("caller_yes_book", act="accept")
        full_booking(ep, slot)
    else:
        say_busy(ep, cid, alts)
        if alts:
            chosen = pick_alt(ep, alts)
            full_booking(ep, chosen)
        else:
            ep.C("caller_hold_off", act="decline")
            ep.E("elle_hold_off_ack", act="ack")


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
            ep.C("caller_yes_book", act="accept")
            full_booking(ep, slot2)
            return
        ep.C("caller_hold_off", act="decline")
        ep.E("elle_hold_off_ack", act="ack")
    else:
        say_busy(ep, c2, alts)
        if alts:
            chosen = pick_alt(ep, alts)
            full_booking(ep, chosen)
            return
        ep.C("caller_hold_off", act="decline")
        ep.E("elle_hold_off_ack", act="ack")
    bye(ep, 0.5)


def sc_mind_change_after_readback(ep: Ep) -> None:
    slot = slot_and_ask(ep, "book", book_intent=True)
    c1, _ = check(ep, slot, "free")
    ep.claim("elle_free", c1)
    ep.C("caller_yes_book", act="accept")
    details(ep)
    readback(ep, slot)
    slot2 = other_slot(ep, [slot], near=slot)
    when2, _ = T.caller_phrase(slot2, ep.now, ep.rng, ep.lang)
    ep.vars["when2"] = when2
    ep.set_slot("slot2", slot2)
    ep.C("caller_mind_change", act="mind_change")
    c2, _ = check(ep, slot2, "free", pre_key="elle_mind_ack")
    rb = readback(ep, slot2, key="elle_free_readback", claim_cid=c2)
    cf = confirm(ep)
    cid = book(ep, slot2, rb, cf)
    booked(ep, cid, slot2)
    bye(ep)


def sc_availability_unknown_recovery(ep: Ep) -> None:
    slot = slot_and_ask(ep, "book")
    c1, _ = check(ep, slot, "unknown")
    ep.claim("elle_unknown", c1)
    if ep.rng.random() < 0.6:
        ep.C("caller_retry", act="retry")
        outcome = "free" if ep.rng.random() < 0.7 else "busy"
        c2, alts = check(ep, slot, outcome)
        if outcome == "free":
            ep.claim("elle_free", c2)
            ep.C("caller_yes_book", act="accept")
            full_booking(ep, slot)
        else:
            say_busy(ep, c2, alts)
            if alts:
                full_booking(ep, pick_alt(ep, alts))
            else:
                ep.C("caller_hold_off", act="decline")
                ep.E("elle_hold_off_ack", act="ack")
    else:
        # Book it as a request; the write tool re-checks at write time anyway.
        ep.C("caller_request_ok", act="accept")
        full_booking(ep, slot)


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
            ep.C("caller_yes_book", act="accept")
            full_booking(ep, slot)
            return
        ep.C("caller_hold_off", act="decline")
        ep.E("elle_hold_off_ack", act="ack")
    else:
        say_busy(ep, c2, alts)
        ep.C("caller_hold_off", act="decline")
        ep.E("elle_hold_off_ack", act="ack")
    bye(ep, 0.5)


# --- scenarios: writes and mixed -------------------------------------------------------------

def accept_offer(ep: Ep, offer) -> None:
    """Caller says yes to Elle's offer; sometimes eagerly, over its last words."""
    if offer is not None and offer["dur"] > 2.5 and ep.rng.random() < 0.2:
        ep.interrupt(offer, "caller_yes_book", frac=ep.rng.uniform(0.75, 0.92), act="accept")
    else:
        ep.C("caller_yes_book", act="accept")


def sc_book_after_availability_free(ep: Ep) -> None:
    slot = slot_and_ask(ep, "book", book_intent=True)
    cid, _ = check(ep, slot, "free")
    accept_offer(ep, ep.claim("elle_free", cid))
    full_booking(ep, slot)


def sc_book_after_availability_busy(ep: Ep) -> None:
    slot = slot_and_ask(ep, "book", book_intent=True)
    cid, alts = check(ep, slot, "busy")
    if not alts:  # this scenario needs something to pick
        raise RetryScenario()
    say_busy(ep, cid, alts)
    chosen = pick_alt(ep, alts)
    full_booking(ep, chosen)


def sc_book_direct_details_first(ep: Ep) -> None:
    slot = T.random_slot(ep.rng, ep.now)
    if ep.rng.random() < 0.5:
        ep.E("elle_greet", act="greeting", at=round(ep.rng.uniform(0.2, 0.6), 2))
    when, _ = T.caller_phrase(slot, ep.now, ep.rng, ep.lang)
    ep.vars["when"] = when
    ep.set_slot("slot", slot)
    ep.C("caller_want_book_at", act="ask_slot", at=0.0 if not ep.events else None)
    ep.C("caller_give_all", act="give_details", gap=ep.gap(0.15, 0.08))
    cid, _ = check(ep, slot, "free", pre_key="elle_check_first")
    if ep.rng.random() < 0.5:
        quiet_save(ep, "name", f"Their name is {ep.p.name}.")
    rb = readback(ep, slot, key="elle_free_readback", claim_cid=cid)
    cf = confirm(ep)
    b = book(ep, slot, rb, cf)
    booked(ep, b, slot)
    if ep.rng.random() < 0.6:
        sms_consent(ep)
    bye(ep)


def sc_book_readback_declined(ep: Ep) -> None:
    slot = slot_and_ask(ep, "book", book_intent=True)
    cid, _ = check(ep, slot, "free")
    ep.claim("elle_free", cid)
    ep.C("caller_yes_book", act="accept")
    details(ep)
    readback(ep, slot)
    ep.C("caller_decline_readback", act="decline")
    ep.E("elle_decline_ack", act="ack")
    bye(ep, 0.6)


def sc_book_readback_corrected(ep: Ep) -> None:
    slot = slot_and_ask(ep, "book", book_intent=True)
    cid, _ = check(ep, slot, "free")
    ep.claim("elle_free", cid)
    ep.C("caller_yes_book", act="accept")
    details(ep)
    readback(ep, slot)
    slot2 = other_slot(ep, [slot], near=slot)
    when2, _ = T.caller_phrase(slot2, ep.now, ep.rng, ep.lang)
    ep.vars["when2"] = when2
    ep.set_slot("slot2", slot2)
    ep.C("caller_correct_readback", act="correction")
    c2, _ = check(ep, slot2, "free", pre_key="elle_mind_ack")
    rb = readback(ep, slot2, key="elle_free_readback", claim_cid=c2)
    cf = confirm(ep)
    b = book(ep, slot2, rb, cf)
    booked(ep, b, slot2)
    bye(ep)


def sc_book_missing_email(ep: Ep) -> None:
    slot = slot_and_ask(ep, "book", book_intent=True)
    cid, _ = check(ep, slot, "free")
    ep.claim("elle_free", cid)
    ep.C("caller_push_book", act="give_name")
    ep.E("elle_need_email", act="ask_email")
    r = ep.rng.random()
    if r < 0.5:
        ep.C("caller_give_email", act="give_email")
    else:
        ep.C("caller_email_hesitate", act="hesitate")
        ep.E("elle_email_why", act="explain")
        if r < 0.75:
            ep.C("caller_give_email", act="give_email")
        else:
            ep.C("caller_refuse_email", act="decline")
            ev = ep.E("elle_no_email_fallback", act="ack")
            quiet_save(ep, "reason_for_calling",
                       f"Wanted an intro call around {T.fmt_when(slot)} about {ep.p.topic}; declined to give an email.",
                       anchor=ev)
            bye(ep, 0.7)
            return
    rb = readback(ep, slot)
    cf = confirm(ep)
    b = book(ep, slot, rb, cf)
    booked(ep, b, slot)
    bye(ep)


def _bad_email(ep: Ep) -> str:
    """A spoken address that can't be valid: no TLD, no 'at', or cut off."""
    local, dom = ep.p.email.split("@")
    local_sp, dom_sp = local.replace(".", " dot "), dom.replace(".", " dot ")
    return ep.rng.choice([f"{local_sp} at {dom.split('.')[0]}", f"{local_sp} {dom_sp}", f"{local_sp} at"])


def sc_book_invalid_email(ep: Ep) -> None:
    slot = slot_and_ask(ep, "book", book_intent=True)
    cid, _ = check(ep, slot, "free")
    ep.claim("elle_free", cid)
    ep.C("caller_yes_book", act="accept")
    ep.E("elle_ask_name", act="ask_name")
    ep.C("caller_give_name", act="give_name")
    ep.E("elle_ask_email", act="ask_email")
    ep.vars["bad_email_sp"] = _bad_email(ep)
    ep.vars["email_guess_sp"] = ep.vars["email_sp"]
    ep.C("caller_bad_email", act="give_email_invalid")
    ep.E("elle_email_clarify", act="clarify")
    ep.C("caller_email_fix", act="give_email")
    rb = readback(ep, slot)
    cf = confirm(ep)
    b = book(ep, slot, rb, cf)
    booked(ep, b, slot)
    bye(ep)


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
    ep.E("elle_past", act="propose")
    ep.C("caller_ok_alt", act="accept")
    cid, _ = check(ep, slot, "free")
    accept_offer(ep, ep.claim("elle_free", cid))
    full_booking(ep, slot)


def _conflict(ep: Ep, result_text: str, key: str) -> None:
    slot = slot_and_ask(ep, "book", book_intent=True)
    c1, _ = check(ep, slot, "free")
    ep.claim("elle_free", c1)
    ep.C("caller_yes_book", act="accept")
    details(ep)
    rb = readback(ep, slot)
    cf = confirm(ep)
    b1 = book(ep, slot, rb, cf, content=result_text)
    alt = other_slot(ep, [slot], near=slot)
    ep.set_slot("alt1", alt)
    ep.claim(key, b1)
    ep.C("caller_ok_check_alt", act="accept")
    c2, _ = check(ep, alt, "free")
    rb2 = readback(ep, alt, key="elle_free_readback", claim_cid=c2)
    cf2 = confirm(ep)
    b2 = book(ep, alt, rb2, cf2)
    booked(ep, b2, alt)
    bye(ep)


def sc_book_slot_taken_race(ep: Ep) -> None:
    _conflict(ep, RES_TAKEN, "elle_slot_taken")


def sc_book_busy_at_write(ep: Ep) -> None:
    _conflict(ep, RES_BUSY_AT_WRITE, "elle_busy_at_write")


def sc_book_write_error(ep: Ep) -> None:
    slot = slot_and_ask(ep, "book", book_intent=True)
    c1, _ = check(ep, slot, "free")
    ep.claim("elle_free", c1)
    ep.C("caller_yes_book", act="accept")
    details(ep)
    rb = readback(ep, slot)
    cf = confirm(ep)
    b = book(ep, slot, rb, cf, content=RES_WRITE_ERROR)
    ev = ep.claim("elle_write_error", b)
    quiet_save(ep, "booking_request",
               f"Wanted an intro call on {T.fmt_when(slot)}; email {ep.p.email}; booking failed, needs follow-up.",
               anchor=ev)
    bye(ep)


def sc_save_fact_midconversation(ep: Ep) -> None:
    ep.E("elle_greet", act="greeting", at=round(ep.rng.uniform(0.2, 0.6), 2))
    ep.C("caller_reveal_company", act="reveal")
    ev = ep.E("elle_after_company", act="chat")
    quiet_save(ep, "company", f"They work at {ep.p.company}.", anchor=ev)
    ep.C("caller_reveal_role", act="reveal")
    ev = ep.E("elle_after_role", act="pitch")
    quiet_save(ep, "role", f"They are hiring for {ep.p.role}.", anchor=ev)
    if ep.rng.random() < 0.3:
        ep.C("caller_chitchat", act="chat")
        ep.E("elle_chitchat", act="chat")
    if ep.rng.random() < 0.55:
        slot = T.random_slot(ep.rng, ep.now)
        ask_for_slot(ep, slot, direct=True)
        cid, _ = check(ep, slot, "free")
        ep.claim("elle_free", cid)
        ep.C("caller_yes_book", act="accept")
        full_booking(ep, slot, sms=False)
    else:
        ep.C("caller_hold_off", act="decline")
        ep.E("elle_hold_off_ack", act="ack")
        bye(ep, 0.6)


def sc_book_then_when_again_memory(ep: Ep) -> None:
    slot = slot_and_ask(ep, "book", book_intent=True)
    c1, _ = check(ep, slot, "free")
    ep.claim("elle_free", c1)
    ep.C("caller_yes_book", act="accept")
    b = full_booking(ep, slot, sms=False, end=False)
    ep.C("caller_when_again", act="ask_recall")
    ep.claim("elle_when_again", b)
    bye(ep)


def sc_book_then_when_again_lookup(ep: Ep) -> None:
    slot = slot_and_ask(ep, "book", book_intent=True)
    c1, _ = check(ep, slot, "free")
    ep.claim("elle_free", c1)
    ep.C("caller_yes_book", act="accept")
    full_booking(ep, slot, sms=ep.rng.random() < 0.4, end=False)
    ep.C("caller_ask_bookings", act="ask_bookings", gap=ep.gap(0.8, 0.3, 0.2, 2.0))
    rows = [(ep.rng.randint(12, 480), ep.p.name, slot, "pending", "")]
    cid, _ = ep.read("get_my_bookings", {}, res_bookings(rows), lat_read(ep), pre_key="elle_bookings_pre")
    ep.set_slot("slot", slot)
    ep.claim("elle_bookings_one", cid)
    bye(ep)


def sc_returning_caller_notes(ep: Ep) -> None:
    other = Person(ep.rng, ep.lang)
    same = ep.rng.random() < 0.4
    known = ep.p if same else other
    ep.notes = [{"category": "name", "fact": f"Their name is {known.name}."},
                {"category": "company", "fact": f"They work at {known.company}."}]
    ep.notes_names = [known.name]
    if ep.rng.random() < 0.5:
        cid, rows = bookings_read(ep, ep.rng.choice([0, 1]), rows=None)
        say_bookings(ep, cid, rows)
        if rows:
            return
        # No booking: make one, asking for the name rather than using the notes.
        ep.C("caller_yes_book", act="accept")
        slot = T.random_slot(ep.rng, ep.now)
        ask_for_slot(ep, slot, direct=False)
    else:
        slot = slot_and_ask(ep, "book", book_intent=True)
    c1, _ = check(ep, slot, "free")
    ep.claim("elle_free", c1)
    ep.C("caller_yes_book", act="accept")
    full_booking(ep, slot)


def sc_barge_in_then_book(ep: Ep) -> None:
    ep.E("elle_greet", act="greeting", at=round(ep.rng.uniform(0.2, 0.6), 2))
    ep.C("caller_ask_about", act="ask_about")
    pitch = ep.E("elle_interrupted_long", act="pitch")
    ep.interrupt(pitch, "caller_interrupt", act="barge_in")
    slot = T.random_slot(ep.rng, ep.now)
    ask_for_slot(ep, slot, direct=False)
    cid, alts = check(ep, slot, "free" if ep.rng.random() < 0.7 else "busy")
    if ep.results[cid]["content"].startswith("FREE"):
        ep.claim("elle_free", cid)
        ep.C("caller_yes_book", act="accept")
        full_booking(ep, slot)
    else:
        say_busy(ep, cid, alts)
        ep.C("caller_hold_off", act="decline")
        ep.E("elle_hold_off_ack", act="ack")


def sc_es_switch_midcall(ep: Ep) -> None:
    ep.lang_label = "en+es"
    ep.E("elle_greet", act="greeting", at=round(ep.rng.uniform(0.2, 0.6), 2))
    ep.C("caller_switch_es", act="switch_lang")
    ep.set_lang("es")
    ep.E("elle_switch_ack", act="ask_time")
    slot = T.random_slot(ep.rng, ep.now)
    when, _ = T.caller_phrase(slot, ep.now, ep.rng, "es")
    ep.vars["when"] = when
    ep.set_slot("slot", slot)
    ep.C("caller_ask_avail", act="ask_slot")
    cid, _ = check(ep, slot, "free")
    ep.claim("elle_free", cid)
    ep.C("caller_yes_book", act="accept")
    full_booking(ep, slot, sms=ep.rng.random() < 0.5)


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
        ep.C("caller_hold_off", act="decline")
        ep.E("elle_hold_off_ack", act="ack")
    elif outcome == "busy":
        ep.claim("elle_busy_none", cid)
    else:
        ep.C("caller_hold_off", act="decline")
        ep.E("elle_hold_off_ack", act="ack")
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


def sc_neg_book_without_readback(ep: Ep) -> None:
    ep.negative_kind = "write_without_readback"
    slot = slot_and_ask(ep, "book", book_intent=True)
    c1, _ = check(ep, slot, "free")
    ep.claim("elle_free", c1)
    ep.C("caller_yes_book", act="accept")
    details(ep)
    # WRONG: books straight away, no readback and no yes.
    b = book(ep, slot, None, None, gated=False)
    booked(ep, b, slot)
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
    ep.C("caller_hold_off", act="decline")
    ep.E("elle_hold_off_ack", act="ack")


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
    Scenario("book_readback_declined", sc_book_readback_declined, "write_guard", 4),
    Scenario("book_readback_corrected", sc_book_readback_corrected, "write_guard", 3),
    Scenario("book_missing_email", sc_book_missing_email, "write_guard", 4),
    Scenario("book_invalid_email", sc_book_invalid_email, "write_guard", 3),
    Scenario("book_past_time", sc_book_past_time, "write_guard", 3),
    Scenario("mind_change_after_readback", sc_mind_change_after_readback, "mind_change", 3),
    Scenario("book_slot_taken_race", sc_book_slot_taken_race, "slot_conflict", 4),
    Scenario("book_busy_at_write", sc_book_busy_at_write, "slot_conflict", 3),
    Scenario("book_write_error", sc_book_write_error, "write_error", 2),
    Scenario("save_fact_midconversation", sc_save_fact_midconversation, "silent_save", 5),
    Scenario("book_then_when_again_memory", sc_book_then_when_again_memory, "followup", 3),
    Scenario("book_then_when_again_lookup", sc_book_then_when_again_lookup, "followup", 3),
    Scenario("returning_caller_notes", sc_returning_caller_notes, "identity", 3),
    Scenario("barge_in_then_book", sc_barge_in_then_book, "overlap", 3),
    Scenario("es_switch_midcall", sc_es_switch_midcall, "bilingual", 2),
    # negatives
    Scenario("neg_premature_availability_claim", sc_neg_premature_availability_claim, "negative", 3, EN_ES, 0.15),
    Scenario("neg_premature_bookings_claim", sc_neg_premature_bookings_claim, "negative", 1),
    Scenario("neg_book_without_readback", sc_neg_book_without_readback, "negative", 2),
    Scenario("neg_dead_air_wait", sc_neg_dead_air_wait, "negative", 2),
]

BY_NAME: Dict[str, Scenario] = {s.name: s for s in SCENARIOS}
FAMILIES = sorted({s.family for s in SCENARIOS})
