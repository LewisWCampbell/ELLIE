"""Time handling for the synthetic episodes: the owner's zone, spoken time
phrases (English and Spanish), their resolution against an episode's `now`, and
exact mirrors of the production result formats.

Everything here is pure and deterministic. `resolve()` is deliberately written
independently of the code that *chooses* a slot, so the generator can assert
that the phrase a caller says really means the ISO `start` it emits.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, List, Optional, Sequence, Tuple

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python < 3.9
    raise SystemExit("Python 3.9+ with zoneinfo is required")

TZ_NAME = "America/Los_Angeles"
TZ = ZoneInfo(TZ_NAME)
SLOT_MIN = 30  # an intro call is 30 minutes
WORK_START_H = 9  # config.booking.hours default "9-17"
WORK_END_H = 17

WEEKDAYS_EN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEEKDAYS_EN_SHORT = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTHS_EN = ["January", "February", "March", "April", "May", "June", "July",
             "August", "September", "October", "November", "December"]
MONTHS_EN_SHORT = [m[:3] for m in MONTHS_EN]
WEEKDAYS_ES = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
MONTHS_ES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
             "agosto", "septiembre", "octubre", "noviembre", "diciembre"]

NUM_EN = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
          "nine", "ten", "eleven", "twelve"]
NUM_ES = ["cero", "una", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho",
          "nueve", "diez", "once", "doce"]


def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


# --- instants ----------------------------------------------------------------

def local_dt(d: date, hour: int, minute: int) -> datetime:
    """Wall-clock time in the owner's zone. zoneinfo picks the right offset for
    that date, which is the whole point near a DST change."""
    return datetime(d.year, d.month, d.day, hour, minute, tzinfo=TZ)


def iso(dt: datetime) -> str:
    """ISO 8601 with offset, seconds included: 2026-11-12T14:00:00-08:00."""
    return dt.astimezone(TZ).isoformat(timespec="seconds")


def parse_iso(s: str) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return dt


def has_offset(s: str) -> bool:
    dt = parse_iso(s)
    return dt is not None and dt.tzinfo is not None and dt.utcoffset() is not None


def offset_is_correct(s: str) -> bool:
    """True when the string's offset is the one Los Angeles actually uses at that
    instant (catches -08:00 written for a summer date, the classic DST bug)."""
    dt = parse_iso(s)
    if dt is None or dt.utcoffset() is None:
        return False
    return dt.utcoffset() == dt.astimezone(TZ).utcoffset()


def tz_abbrev(dt: datetime) -> str:
    return dt.astimezone(TZ).tzname() or "PT"


# --- production result formats (mirrors of src/) -----------------------------

def _clock12(dt: datetime) -> str:
    h = dt.hour % 12 or 12
    return f"{h}:{dt.minute:02d} {'AM' if dt.hour < 12 else 'PM'}"


def speakable_time(dt: datetime) -> str:
    """src/booking/availability.ts speakableTime: 'Thursday, November 12 at 2:00 PM PST'."""
    d = dt.astimezone(TZ)
    return f"{WEEKDAYS_EN[d.weekday()]}, {MONTHS_EN[d.month - 1]} {d.day} at {_clock12(d)} {tz_abbrev(d)}"


def fmt_when(dt: datetime) -> str:
    """src/sms.ts fmtWhen: 'Thu, Nov 12, 2:00 PM PST'."""
    d = dt.astimezone(TZ)
    return f"{WEEKDAYS_EN_SHORT[d.weekday()]}, {MONTHS_EN_SHORT[d.month - 1]} {d.day}, {_clock12(d)} {tz_abbrev(d)}"


def nearby_openings(requested: datetime, busy: Sequence[Tuple[datetime, datetime]], now: datetime,
                    max_alts: int = 3, search_days: int = 3) -> List[datetime]:
    """Port of availability.ts nearbyOpenings: 30-minute grid (UTC-aligned) from
    12 h before the request to `search_days` after, weekdays inside working hours,
    not overlapping busy ranges, nearest first, then back in time order."""
    slot = 30 * 60
    req = int(requested.timestamp())
    start = req - (req % slot) - 12 * 3600
    cands: List[int] = []
    t = start
    while t < req + search_days * 86400:
        if t > now.timestamp() and t != req:
            loc = datetime.fromtimestamp(t, TZ)
            hour = loc.hour + loc.minute / 60
            if loc.weekday() < 5 and hour >= WORK_START_H and hour + 0.5 <= WORK_END_H:
                if not any(b0.timestamp() < t + slot and b1.timestamp() > t for b0, b1 in busy):
                    cands.append(t)
        t += slot
    cands.sort(key=lambda x: abs(x - req))
    return [datetime.fromtimestamp(x, TZ) for x in sorted(cands[:max_alts])]


# --- slots -------------------------------------------------------------------

# Days the synthetic owner keeps clear (US holidays inside the sampled window).
HOLIDAYS = {date(2026, 11, 26), date(2026, 11, 27), date(2026, 12, 24), date(2026, 12, 25),
            date(2026, 12, 31), date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15),
            date(2027, 5, 31)}


def is_bookable(dt: datetime, now: datetime, lead_min: int = 45) -> bool:
    d = dt.astimezone(TZ)
    return (d.weekday() < 5 and d.date() not in HOLIDAYS and d.minute in (0, 30) and d.second == 0
            and WORK_START_H <= d.hour + d.minute / 60 <= WORK_END_H - 0.5
            and d > now + timedelta(minutes=lead_min))


def random_slot(rng: random.Random, now: datetime, max_days: int = 24,
                min_days: int = 0) -> datetime:
    """A weekday, working-hours, :00/:30 slot comfortably after `now`."""
    for _ in range(500):
        dd = rng.choice(range(min_days, max_days + 1)) if rng.random() < 0.3 else \
            min(max_days, max(min_days, int(rng.expovariate(1 / 4.0))))
        d = now.astimezone(TZ).date() + timedelta(days=dd)
        h = rng.choice([9, 9, 10, 10, 10, 11, 11, 12, 13, 13, 14, 14, 14, 15, 15, 16])
        m = rng.choice([0, 0, 0, 30])
        dt = local_dt(d, h, m)
        if is_bookable(dt, now):
            return dt
    raise RuntimeError("no bookable slot found")


# --- spoken references ---------------------------------------------------------

@dataclass(frozen=True)
class TimeRef:
    """What the caller said, structurally. `resolve(ref, now)` gives the instant."""
    kind: str  # today | tomorrow | weekday | next_week | dom | month_day | weekday_dom
    hour: int
    minute: int
    weekday: int = -1
    day: int = -1
    month: int = -1


def resolve(ref: TimeRef, now: datetime) -> datetime:
    """Turn a spoken reference into the owner-zone instant it denotes, relative to
    `now`. Conventions (documented in README):
      - "Thursday": the next Thursday strictly after today (1-7 days ahead).
      - "next week on Tuesday": Tuesday of the calendar week (Mon-Sun) after this one.
      - "the 14th": the next 14th strictly after today.
      - "November 19th": this year's, or next year's if already passed.
    """
    today = now.astimezone(TZ).date()
    if ref.kind == "today":
        d = today
    elif ref.kind == "tomorrow":
        d = today + timedelta(days=1)
    elif ref.kind == "weekday":
        delta = (ref.weekday - today.weekday()) % 7 or 7
        d = today + timedelta(days=delta)
    elif ref.kind == "next_week":
        monday_next = today + timedelta(days=7 - today.weekday())
        d = monday_next + timedelta(days=ref.weekday)
    elif ref.kind in ("dom", "weekday_dom"):
        d = today + timedelta(days=1)
        while d.day != ref.day:
            d += timedelta(days=1)
    elif ref.kind == "month_day":
        d = date(today.year, ref.month, ref.day)
        if d <= today:
            d = date(today.year + 1, ref.month, ref.day)
    else:
        raise ValueError(ref.kind)
    return local_dt(d, ref.hour, ref.minute)


def candidate_refs(slot: datetime, now: datetime) -> List[TimeRef]:
    """Every way of referring to `slot` that `resolve` maps back to it exactly."""
    s = slot.astimezone(TZ)
    today = now.astimezone(TZ).date()
    delta = (s.date() - today).days
    h, m = s.hour, s.minute
    out: List[TimeRef] = []
    if delta == 0:
        out.append(TimeRef("today", h, m))
    if delta == 1:
        out += [TimeRef("tomorrow", h, m)] * 3
    if 1 <= delta <= 6:
        out += [TimeRef("weekday", h, m, weekday=s.weekday())] * 3
    nxt = TimeRef("next_week", h, m, weekday=s.weekday())
    if 2 <= delta <= 13 and resolve(nxt, now) == s:
        out.append(nxt)
    if 2 <= delta <= 27:
        out.append(TimeRef("dom", h, m, day=s.day))
        out.append(TimeRef("weekday_dom", h, m, weekday=s.weekday(), day=s.day))
    if delta >= 2:
        out.append(TimeRef("month_day", h, m, day=s.day, month=s.month))
    return [r for r in out if resolve(r, now) == s]


def clock_en(h: int, m: int, rng: random.Random, style: str = "any") -> str:
    """Spoken clock time. Bare hours follow the business-hours convention: 1-6
    means PM, 7-11 means AM, 12 is noon."""
    h12 = h % 12 or 12
    pm = h >= 12
    ap = "p.m." if pm else "a.m."
    word = NUM_EN[h12]
    if h == 12 and m == 0:
        opts = ["noon", "twelve", "12 noon", "midday"]
    elif m == 0:
        opts = [word, f"{word} o'clock", f"{h12} {ap}", f"{h12}{'pm' if pm else 'am'}",
                f"{word} in the {'afternoon' if pm else 'morning'}"]
    else:
        opts = [f"{word} thirty", f"{h12}:30", f"half past {word}", f"{h12}:30 {ap}",
                f"{word} thirty in the {'afternoon' if pm else 'morning'}"]
    if style == "explicit":  # unambiguous am/pm, used by Elle in readbacks
        if h == 12 and m == 0:
            return "noon"
        return f"{h12}{'' if m == 0 else ':30'} {ap}"
    return rng.choice(opts)


def clock_es(h: int, m: int, rng: random.Random, style: str = "any") -> str:
    h12 = h % 12 or 12
    art = "la" if h12 == 1 else "las"
    base = f"{art} {NUM_ES[h12]}"
    if m == 30:
        base += " y media"
    if h == 12 and m == 0:
        return "al mediodía" if style != "bare" else "las doce"
    part = "de la mañana" if h < 12 else "de la tarde"
    if style == "explicit" or rng.random() < 0.5:
        return f"a {base} {part}"
    return f"a {base}"


def render_ref(ref: TimeRef, rng: random.Random, lang: str = "en") -> str:
    """The caller's phrase for a TimeRef ('Thursday at two', 'the 14th at 2pm')."""
    if lang == "es":
        c = clock_es(ref.hour, ref.minute, rng)
        wd = WEEKDAYS_ES[ref.weekday] if ref.weekday >= 0 else ""
        return {
            "today": f"hoy {c}",
            "tomorrow": f"mañana {c}",
            "weekday": f"el {wd} {c}",
            "next_week": f"la próxima semana, el {wd} {c}",
            "dom": f"el {ref.day} {c}",
            "weekday_dom": f"el {wd} {ref.day} {c}",
            "month_day": f"el {ref.day} de {MONTHS_ES[ref.month - 1] if ref.month > 0 else ''} {c}",
        }[ref.kind]
    c = clock_en(ref.hour, ref.minute, rng)
    wd = WEEKDAYS_EN[ref.weekday] if ref.weekday >= 0 else ""
    if ref.kind == "today":
        part = "this afternoon" if ref.hour >= 12 else "this morning"
        return rng.choice([f"today at {c}", f"later today at {c}", f"{part} at {c}"])
    if ref.kind == "tomorrow":
        return rng.choice([f"tomorrow at {c}", f"{c} tomorrow", f"tomorrow, {c}"])
    if ref.kind == "weekday":
        return rng.choice([f"{wd} at {c}", f"this {wd} at {c}", f"{c} on {wd}", f"{wd}, {c}"])
    if ref.kind == "next_week":
        return rng.choice([f"next week on {wd} at {c}", f"{wd} next week at {c}",
                           f"{c} on {wd} next week"])
    if ref.kind == "dom":
        return rng.choice([f"the {ordinal(ref.day)} at {c}", f"{c} on the {ordinal(ref.day)}"])
    if ref.kind == "weekday_dom":
        return rng.choice([f"{wd} the {ordinal(ref.day)} at {c}", f"{wd}, the {ordinal(ref.day)}, at {c}"])
    if ref.kind == "month_day":
        mo = MONTHS_EN[ref.month - 1]
        return rng.choice([f"{mo} {ordinal(ref.day)} at {c}", f"{mo} {ref.day} at {c}",
                           f"{c} on {mo} {ordinal(ref.day)}"])
    raise ValueError(ref.kind)


def caller_phrase(slot: datetime, now: datetime, rng: random.Random, lang: str = "en") -> Tuple[str, TimeRef]:
    refs = candidate_refs(slot, now)
    if not refs:
        raise RuntimeError(f"no phrase for {slot} at {now}")
    ref = rng.choice(refs)
    assert resolve(ref, now) == slot.astimezone(TZ), (ref, now, slot)
    return render_ref(ref, rng, lang), ref


def elle_short(slot: datetime, now: datetime, lang: str = "en") -> str:
    """How Elle refers to a slot in passing: 'Thursday at 2 p.m.', 'tomorrow at 10:30 a.m.'."""
    s = slot.astimezone(TZ)
    delta = (s.date() - now.astimezone(TZ).date()).days
    r = random.Random(0)
    if lang == "es":
        c = clock_es(s.hour, s.minute, r, "explicit")
        if delta == 0:
            return f"hoy {c}"
        if delta == 1:
            return f"mañana {c}"
        if delta <= 6:
            return f"el {WEEKDAYS_ES[s.weekday()]} {c}"
        return f"el {WEEKDAYS_ES[s.weekday()]} {s.day} {c}"
    c = clock_en(s.hour, s.minute, r, "explicit")
    if delta == 0:
        return f"today at {c}"
    if delta == 1:
        return f"tomorrow at {c}"
    if delta <= 6:
        return f"{WEEKDAYS_EN[s.weekday()]} at {c}"
    return f"{WEEKDAYS_EN[s.weekday()]} the {ordinal(s.day)} at {c}"


def elle_long(slot: datetime, lang: str = "en") -> str:
    """Full readback form, without the zone: 'Thursday, November 12th at 2 p.m.'."""
    s = slot.astimezone(TZ)
    r = random.Random(0)
    if lang == "es":
        return f"el {WEEKDAYS_ES[s.weekday()]} {s.day} de {MONTHS_ES[s.month - 1]} {clock_es(s.hour, s.minute, r, 'explicit')}"
    return f"{WEEKDAYS_EN[s.weekday()]}, {MONTHS_EN[s.month - 1]} {ordinal(s.day)} at {clock_en(s.hour, s.minute, r, 'explicit')}"


def zone_words(lang: str = "en") -> str:
    return "hora del Pacífico" if lang == "es" else "Pacific time"


# --- episode clock -------------------------------------------------------------

DST_CHANGES = [datetime(2026, 11, 1, tzinfo=TZ), datetime(2027, 3, 14, tzinfo=TZ),
               datetime(2027, 11, 7, tzinfo=TZ)]


def random_now(rng: random.Random) -> datetime:
    """An episode's `now`: mostly weekday business hours between Sep 2026 and
    May 2027, with ~25% of episodes in the 10 days before a DST change so the
    booked slot often lies on the other side of it."""
    if rng.random() < 0.25:
        anchor = rng.choice(DST_CHANGES[:2])
        d = (anchor - timedelta(days=rng.randint(1, 10))).date()
    else:
        d = date(2026, 9, 14) + timedelta(days=rng.randint(0, 255))
    if d.weekday() >= 5 and rng.random() < 0.75:
        d += timedelta(days=7 - d.weekday())
    h = rng.choice([8, 9, 9, 10, 10, 11, 11, 12, 13, 14, 15, 16, 17, 18])
    m = rng.randrange(0, 60)
    return datetime(d.year, d.month, d.day, h, m, tzinfo=TZ)
