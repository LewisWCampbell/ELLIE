"""Writes fixtures/episodes.jsonl: a tiny hand-built episode set in the CONTRACT.md
format, used by the tests and by the dry run when the dataset sample is absent.
Six positives covering every tool and the no-tool case, five labelled negatives
(one per failure the metrics must catch). Run: python3 make_fixtures.py"""

import json
import os

NOW = "2026-11-09T10:15:00-08:00"
THU2 = "2026-11-12T14:00:00-08:00"
FRI10 = "2026-11-13T10:00:00-08:00"


def sp(t, dur, ch, text):
    return {"t": t, "dur": dur, "ch": ch, "type": "speech", "text": text}


def call(t, cid, name, args):
    return {"t": t, "ch": "elle", "type": "tool_call", "id": cid, "name": name, "args": args}


def res(t, cid, content, latency):
    return {"t": t, "ch": "system", "type": "tool_result", "id": cid, "content": content, "latency_s": latency}


def ep(n, scenario, events, expected, claims, split="train", negative=False, forbidden=True):
    labels = {"expected_calls": expected, "claim_spans": claims, "forbidden_before_result": forbidden}
    if negative:
        labels["negative"] = True
    return {"id": f"ep_fx_{n:03d}", "split": split, "scenario": scenario, "seed": n, "now": NOW,
            "events": events, "labels": labels}


EPISODES = [
    ep(1, "book_after_availability_free", [
        sp(0.0, 2.0, "caller", "Hi, is Sam free Thursday at two?"),
        sp(2.3, 1.2, "elle", "Let me check his calendar."),
        call(2.4, "c1", "check_availability", {"start": THU2}),
        res(3.4, "c1", "FREE: Thursday, Nov 12 at 2:00 PM PST is open.", 1.0),
        sp(3.8, 2.2, "elle", "He's free then. Want me to book it?"),
        sp(6.3, 0.9, "caller", "Yes please."),
        sp(7.4, 1.8, "elle", "Great. What's your name and email?"),
        sp(9.5, 2.5, "caller", "Jane Doe, jane at example dot com."),
        sp(12.2, 4.0, "elle", "So that's Jane Doe, jane at example dot com, Thursday at two. Shall I book it?"),
        sp(16.5, 1.0, "caller", "Yes, go ahead."),
        call(17.6, "c2", "book_intro_call", {"name": "Jane Doe", "email": "jane@example.com", "start": THU2}),
        sp(17.7, 1.0, "elle", "Booking that now."),
        res(18.9, "c2", "Done: booked a tentative intro call for Thursday, Nov 12 at 2:00 PM PST.", 1.3),
        sp(19.2, 2.0, "elle", "You're all set, it's tentatively booked."),
    ], [{"name": "check_availability", "args": {"start": THU2}},
        {"name": "book_intro_call", "args": {"name": "Jane Doe", "email": "jane@example.com", "start": THU2}}],
        [{"event_index": 4, "depends_on": "c1"}, {"event_index": 13, "depends_on": "c2"}]),
    ep(2, "availability_busy", [
        sp(0.0, 1.8, "caller", "Can I get Friday at ten?"),
        sp(2.0, 1.3, "elle", "One moment, checking."),
        call(2.1, "c1", "check_availability", {"start": "2026-11-13T18:00:00Z"}),
        res(3.0, "c1", "BUSY: Friday, Nov 13 at 10:00 AM PST is taken. Nearby open times: 10:30 AM, 11:00 AM.", 0.9),
        sp(3.4, 3.0, "elle", "He's busy then, but ten thirty or eleven are open."),
    ], [{"name": "check_availability", "args": {"start": FRI10}}],
        [{"event_index": 4, "depends_on": "c1"}]),
    ep(3, "get_my_bookings", [
        sp(0.0, 1.6, "caller", "Do I have anything booked?"),
        sp(1.9, 1.4, "elle", "One sec, pulling that up."),
        call(2.0, "c1", "get_my_bookings", {}),
        res(2.9, "c1", "Upcoming: #12 Thursday, Nov 12 at 2:00 PM PST (tentative).", 0.9),
        sp(3.5, 2.2, "elle", "You have one call, Thursday at two."),
    ], [{"name": "get_my_bookings", "args": {}}], [{"event_index": 4, "depends_on": "c1"}]),
    ep(4, "save_fact_silent", [
        sp(0.0, 2.0, "caller", "By the way, I'm a product designer."),
        call(2.2, "c1", "save_user_fact", {"category": "role", "fact": "product designer"}),
        res(2.4, "c1", "saved", 0.2),
        sp(2.3, 1.8, "elle", "Nice! What kind of products?"),
    ], [{"name": "save_user_fact", "args": {"category": "role", "fact": "product designer"}}], [],
        forbidden=False),
    ep(5, "no_tool_chitchat", [
        sp(0.0, 1.5, "caller", "What does Sam do?"),
        sp(1.8, 2.6, "elle", "He builds voice tools and phone assistants."),
    ], [], [], forbidden=False),
    ep(6, "availability_slow_with_filler", [
        sp(0.0, 1.8, "caller", "Is Thursday at two open?"),
        call(2.0, "c1", "check_availability", {"start": THU2}),
        sp(2.1, 2.0, "elle", "Sure, checking now, just a second."),
        res(3.6, "c1", "FREE: Thursday, Nov 12 at 2:00 PM PST is open.", 1.6),
        sp(4.2, 1.6, "elle", "Yes, he's free then."),
    ], [{"name": "check_availability", "args": {"start": THU2}}], [{"event_index": 4, "depends_on": "c1"}]),
    # --- negatives ---------------------------------------------------------
    ep(7, "neg_premature_claim", [
        sp(0.0, 2.0, "caller", "Is Sam free Thursday at two?"),
        call(2.2, "c1", "check_availability", {"start": THU2}),
        sp(2.3, 1.5, "elle", "Yes, he's free then."),
        res(3.4, "c1", "FREE: Thursday, Nov 12 at 2:00 PM PST is open.", 1.2),
        sp(4.0, 1.4, "elle", "Want me to book it?"),
    ], [{"name": "check_availability", "args": {"start": THU2}}], [{"event_index": 2, "depends_on": "c1"}],
        negative=True),
    ep(8, "neg_write_no_readback", [
        sp(0.0, 3.0, "caller", "Book me Thursday at two. Jane Doe, jane at example dot com."),
        call(3.3, "c1", "book_intro_call", {"name": "Jane Doe", "email": "jane@example.com", "start": THU2}),
        sp(3.4, 1.0, "elle", "Booking that now."),
        res(4.4, "c1", "Done: booked a tentative intro call for Thursday, Nov 12 at 2:00 PM PST.", 1.0),
        sp(4.8, 1.8, "elle", "You're all set, it's tentatively booked."),
    ], [{"name": "book_intro_call", "args": {"name": "Jane Doe", "email": "jane@example.com", "start": THU2}}],
        [{"event_index": 4, "depends_on": "c1"}], negative=True),
    ep(9, "neg_write_missing_field", [
        sp(0.0, 2.2, "caller", "Book me Thursday at two, I'm Jane Doe."),
        sp(2.5, 2.4, "elle", "So that's Jane Doe, Thursday at two. Shall I book it?"),
        sp(5.2, 0.8, "caller", "Yes."),
        call(6.2, "c1", "book_intro_call", {"name": "Jane Doe", "start": THU2}),
        sp(6.3, 1.0, "elle", "Booking that now."),
        res(7.3, "c1", "Error: email is required.", 1.0),
        sp(7.6, 1.8, "elle", "Sorry, what's your email?"),
    ], [{"name": "book_intro_call", "args": {"name": "Jane Doe", "start": THU2}}], [], negative=True),
    ep(10, "neg_dead_air", [
        sp(0.0, 1.8, "caller", "Is Thursday at two open?"),
        sp(2.0, 0.5, "elle", "Let me see."),
        call(2.6, "c1", "check_availability", {"start": THU2}),
        res(4.1, "c1", "FREE: Thursday, Nov 12 at 2:00 PM PST is open.", 1.5),
        sp(4.4, 1.6, "elle", "Yes, he's free then."),
    ], [{"name": "check_availability", "args": {"start": THU2}}], [{"event_index": 4, "depends_on": "c1"}],
        negative=True),
    ep(11, "neg_silent_write_narrated", [
        sp(0.0, 2.0, "caller", "By the way, I'm a product designer."),
        call(2.2, "c1", "save_user_fact", {"category": "role", "fact": "product designer"}),
        res(2.4, "c1", "saved", 0.2),
        sp(2.6, 2.2, "elle", "Got it, I've saved that you're a designer."),
    ], [{"name": "save_user_fact", "args": {"category": "role", "fact": "product designer"}}], [],
        negative=True, forbidden=False),
]

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "episodes.jsonl")
    with open(out, "w") as f:
        for e in EPISODES:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"wrote {len(EPISODES)} episodes to {out}")
