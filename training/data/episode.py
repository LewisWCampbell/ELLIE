"""The episode builder: a two-channel timeline plus labels.

`Ep` is the small runtime that scenario functions drive. It places speech on
the caller and Elle channels with realistic gaps and overlap, emits tool calls
and results with latencies, keeps Elle talking through slow reads, and records
which Elle utterances depend on which result (claim spans) and which readback
and yes gate each write (write gates). `finish()` sorts the events and turns
object references into the integer indices the contract uses.
"""

from __future__ import annotations

import math
import random
import re
import unicodedata
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import banks
import elle_time as T

OWNER = "Lewis"
SPEECH_RATE_WPS = 2.6  # words per second, before jitter


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def ascii_fold(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def spoken_email(email: str, lang: str = "en") -> str:
    dot, at = (" punto ", " arroba ") if lang == "es" else (" dot ", " at ")
    local, dom = email.split("@")
    return local.replace(".", dot) + at + dom.replace(".", dot)


class Person:
    def __init__(self, rng: random.Random, lang: str = "en"):
        self.first = rng.choice(banks.FIRST_NAMES)
        self.last = rng.choice(banks.LAST_NAMES)
        self.name = f"{self.first} {self.last}"
        f, l = ascii_fold(self.first).lower(), ascii_fold(self.last).lower()
        dom = rng.choice(banks.EMAIL_DOMAINS)
        local = rng.choice([f"{f}.{l}", f"{f[0]}{l}", f"{f}{l[0]}", f"{f}", f"{f}.{l}", f"{f}{l}"])
        self.email = f"{local}@{dom}"
        self.company = rng.choice(banks.COMPANIES)
        self.role = rng.choice(banks.ROLES)
        self.topic = rng.choice(banks.TOPICS)
        self.phone = "+1" + rng.choice(["415", "510", "628", "650", "707", "916", "213", "206", "503"]) \
            + "55501" + f"{rng.randrange(0, 100):02d}"


class Ep:
    def __init__(self, scenario: str, family: str, seed: int, rng: random.Random,
                 now: datetime, lang: str = "en"):
        self.scenario, self.family, self.seed, self.rng = scenario, family, seed, rng
        self.now = now
        self.lang = lang
        self.lang_label = lang
        self.events: List[Dict[str, Any]] = []
        self.end = {"elle": 0.0, "caller": 0.0}
        self.calls: Dict[str, Dict[str, Any]] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.claims: List[Tuple[Dict[str, Any], str]] = []
        self.gates: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
        self.negative_kind: Optional[str] = None
        self.notes: List[Dict[str, str]] = []
        self.notes_names: List[str] = []
        self.channel = "voice"
        self.p = Person(rng, lang)
        self.vars: Dict[str, str] = {}
        self._seq = 0
        self.refresh_vars()

    # --- variables & banks ------------------------------------------------------
    def refresh_vars(self) -> None:
        p = self.p
        self.vars.update(first=p.first, last=p.last, name=p.name, email=p.email,
                         email_sp=spoken_email(p.email, self.lang), company=p.company,
                         role=p.role, topic=p.topic, owner=OWNER, zone=T.zone_words(self.lang))

    def set_lang(self, lang: str) -> None:
        self.lang = lang
        self.refresh_vars()

    def set_slot(self, key: str, dt: datetime) -> None:
        """Expose a slot to templates as {key} (short), {key}_cap, {key}_long, {key}_long_cap."""
        short = T.elle_short(dt, self.now, self.lang)
        long = T.elle_long(dt, self.lang)
        self.vars[key] = short
        self.vars[key + "_cap"] = _cap(short)
        self.vars[key + "_long"] = long
        self.vars[key + "_long_cap"] = _cap(long)

    def line(self, key: str, **kw: Any) -> str:
        bank = banks.BANKS[self.lang]
        if key not in bank:
            raise KeyError(f"bank key {key!r} missing for lang {self.lang!r}")
        tpl = self.rng.choice(bank[key])
        v = dict(self.vars)
        v.update({k: str(x) for k, x in kw.items()})
        for k in list(v):
            v.setdefault(k + "_cap", _cap(v[k]))
        text = re.sub(r"\.\.(?!\.)", ".", tpl.format_map(v))  # "2 p.m.." -> "2 p.m."
        return _cap(text.strip())

    # --- timing ---------------------------------------------------------------------
    def dur(self, text: str) -> float:
        n = max(1, len(text.split()))
        return round(max(0.35, n / SPEECH_RATE_WPS * self.rng.uniform(0.88, 1.15) + 0.15), 2)

    def gap(self, mu: float = 0.38, sd: float = 0.15, lo: float = 0.08, hi: float = 0.95) -> float:
        return min(hi, max(lo, self.rng.gauss(mu, sd)))

    def floor(self) -> float:
        return max(self.end.values())

    def _add(self, ev: Dict[str, Any]) -> Dict[str, Any]:
        ev["_seq"] = self._seq
        self._seq += 1
        self.events.append(ev)
        return ev

    def say(self, ch: str, text: str, at: Optional[float] = None, gap: Optional[float] = None,
            act: Optional[str] = None, **extra: Any) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        start = at if at is not None else self.floor() + (self.gap() if gap is None else gap)
        start = round(max(start, self.end[ch] + 0.08 if self.end[ch] > 0 else start), 2)
        ev: Dict[str, Any] = {"t": start, "dur": self.dur(text), "ch": ch, "type": "speech", "text": text}
        if act:
            ev["act"] = act
        if self.lang != "en":
            ev["lang"] = self.lang
        ev.update(extra)
        self.end[ch] = max(self.end[ch], round(start + ev["dur"], 2))
        return self._add(ev)

    def E(self, key: str, act: Optional[str] = None, at: Optional[float] = None,
          gap: Optional[float] = None, **kw: Any) -> Optional[Dict[str, Any]]:
        return self.say("elle", self.line(key, **kw) if key else "", at=at, gap=gap, act=act)

    def C(self, key: str, act: Optional[str] = None, at: Optional[float] = None,
          gap: Optional[float] = None, **kw: Any) -> Optional[Dict[str, Any]]:
        return self.say("caller", self.line(key, **kw), at=at, gap=gap, act=act)

    # --- tools -------------------------------------------------------------------------
    def call(self, name: str, args: Dict[str, Any], at: float) -> str:
        cid = f"c{len(self.calls) + 1}"
        ev = {"t": round(at, 2), "ch": "elle", "type": "tool_call", "id": cid, "name": name, "args": args}
        self.calls[cid] = self._add(ev)
        return cid

    def result(self, cid: str, content: str, latency: float) -> Dict[str, Any]:
        c = self.calls[cid]
        latency = round(max(0.03, latency), 2)
        ev = {"t": round(c["t"] + latency, 2), "ch": "system", "type": "tool_result", "id": cid,
              "content": content, "latency_s": latency}
        self.results[cid] = ev
        return self._add(ev)

    def cover_wait(self, until: float, max_gap: float = 1.25) -> None:
        """Keep Elle talking (fillers) so no silence longer than `max_gap` opens
        before `until` (the result time). This is the wait handling the contract
        measures; dead-air negatives simply skip it."""
        used = set()
        for _ in range(4):
            if until - self.floor() <= max_gap:
                return
            start = self.floor() + self.rng.uniform(0.45, 0.85)
            if start > until - 0.35:
                return
            for _ in range(5):
                text = self.line("elle_filler")
                if text not in used:
                    break
            used.add(text)
            self.say("elle", text, at=start, act="filler")

    def read(self, name: str, args: Dict[str, Any], content: str, latency: float,
             pre_key: Optional[str] = "elle_check_pre", fill: bool = True,
             during=None, pre_text: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """Elle says a short pre-line, emits the read call while saying it, the
        optional `during(ep, call_t, res_t)` hook adds caller speech in the wait,
        then fillers cover any remaining gap, then the result lands."""
        pre = None
        if pre_text is not None:
            pre = self.say("elle", pre_text, act="pre_call")
        elif pre_key:
            pre = self.E(pre_key, act="pre_call")
        if pre is not None:
            at = pre["t"] + self.rng.uniform(0.05, max(0.06, min(0.6, pre["dur"] * 0.6)))
        else:
            at = self.floor() + self.gap(0.25, 0.1)
        cid = self.call(name, args, at)
        res_t = self.calls[cid]["t"] + latency
        if during is not None:
            during(self, self.calls[cid]["t"], res_t)
        if fill:
            self.cover_wait(res_t)
        return cid, self.result(cid, content, latency)

    def claim(self, key: str, cid: str, act: Optional[str] = "claim", text: Optional[str] = None,
              **kw: Any) -> Dict[str, Any]:
        """Elle speech that depends on result `cid`: it starts after the result,
        after Elle's own previous line, and after the caller stops talking."""
        res = self.results[cid]
        start = max(res["t"] + self.rng.uniform(0.18, 0.55), self.end["elle"] + 0.12,
                    self.end["caller"] + self.gap(0.3, 0.1))
        ev = self.say("elle", text if text is not None else self.line(key, **kw), at=start, act=act)
        self.claims.append((ev, cid))
        return ev

    def gate(self, cid: str, readback: Dict[str, Any], confirm: Dict[str, Any]) -> None:
        self.gates.append((cid, readback, confirm))

    def write_at(self) -> float:
        return self.floor() + self.gap(0.2, 0.08, 0.06, 0.4)

    # --- overlap ---------------------------------------------------------------------
    def interrupt(self, ev: Dict[str, Any], key: str, frac: Optional[float] = None,
                  act: Optional[str] = None, text: Optional[str] = None,
                  at: Optional[float] = None) -> Dict[str, Any]:
        """The caller barges in on Elle's `ev`; Elle stops ~0.25 s later and her
        utterance is truncated to the words actually spoken. The caller line is
        marked `barge_in` so the evaluator can measure how fast a model yields.
        `at` overrides the start time (it must fall inside `ev`)."""
        frac = frac if frac is not None else self.rng.uniform(0.35, 0.7)
        start = at if at is not None else ev["t"] + frac * ev["dur"]
        c = self.say("caller", text if text is not None else self.line(key), at=start,
                     act=act or "barge_in")
        cut = round(c["t"] + self.rng.uniform(0.15, 0.35) - ev["t"], 2)
        if cut < ev["dur"]:
            c["barge_in"] = True
            words = ev["text"].split()
            keep = max(1, math.ceil(len(words) * cut / ev["dur"]))
            ev["full_text"] = ev["text"]
            ev["text"] = " ".join(words[:keep])
            ev["dur"] = cut
            ev["interrupted"] = True
            self.end["elle"] = round(ev["t"] + cut, 2)
        return c

    def backchannel(self, ev: Dict[str, Any], ch: str = "caller") -> None:
        if ev is None or ev["dur"] < 2.5:
            return
        at = ev["t"] + self.rng.uniform(0.35, 0.65) * ev["dur"]
        key = "caller_backchannel" if ch == "caller" else "elle_backchannel"
        saved = dict(self.end)
        self.say(ch, self.line(key), at=at, act="backchannel")
        # A backchannel never takes the floor.
        self.end = {k: max(saved[k], self.end[k]) if k == ch else saved[k] for k in saved}

    # --- finish ---------------------------------------------------------------------------
    def finish(self, ep_id: str, split: str) -> Dict[str, Any]:
        evs = sorted(self.events, key=lambda e: (e["t"], e["_seq"]))
        # Call ids are c1..cN in time order, whatever order the beats created them.
        remap: Dict[str, str] = {}
        for e in evs:
            if e["type"] == "tool_call":
                remap[e["id"]] = f"c{len(remap) + 1}"
        for e in evs:
            if e["type"] in ("tool_call", "tool_result"):
                e["id"] = remap[e["id"]]
        idx = {id(e): i for i, e in enumerate(evs)}
        claim_spans = [{"event_index": idx[id(e)], "depends_on": remap[cid]} for e, cid in self.claims]
        claim_spans.sort(key=lambda c: c["event_index"])
        gates = [{"call_id": remap[cid], "readback_index": idx[id(rb)], "confirm_index": idx[id(cf)]}
                 for cid, rb, cf in self.gates]
        calls = [e for e in evs if e["type"] == "tool_call"]
        reads = {"check_availability", "get_my_bookings"}
        n_r = sum(1 for c in calls if c["name"] in reads)
        n_w = len(calls) - n_r
        kind = "none" if not calls else "read" if not n_w else "write" if not n_r else "mixed"
        out_events = []
        for e in evs:
            out_events.append({k: v for k, v in e.items() if not k.startswith("_")})
        labels: Dict[str, Any] = {
            "expected_calls": [{"name": c["name"], "args": c["args"]} for c in calls],
            "claim_spans": claim_spans,
            "forbidden_before_result": bool(claim_spans),
            "negative": self.negative_kind is not None,
            "write_gates": gates,
            "family": self.family,
            "lang": self.lang_label,
            "kind": kind,
        }
        if self.negative_kind:
            labels["negative_kind"] = self.negative_kind
        return {
            "id": ep_id,
            "split": split,
            "scenario": self.scenario,
            "seed": self.seed,
            "now": T.iso(self.now),
            "context": {"channel": self.channel, "caller_number": self.p.phone, "notes": self.notes,
                        "notes_names": self.notes_names},
            "events": out_events,
            "labels": labels,
        }
