"""The validator must catch each violation type. Each test takes a valid
generated episode, breaks one rule, and checks the expected error code."""

import copy
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate import build_episode  # noqa: E402
from validate import validate_episode, validate_files, violations  # noqa: E402


INDEX_KEYS = ("intent_index", "question_index", "confirm_index")


def codes(ep):
    return {i.code for i in validate_episode(ep)}


def resort(ep):
    """Re-sort events by t after a timing edit, remapping label indices."""
    evs = ep["events"]
    order = sorted(range(len(evs)), key=lambda i: (evs[i]["t"], i))
    new_pos = {old: new for new, old in enumerate(order)}
    ep["events"] = [evs[i] for i in order]
    for c in ep["labels"]["claim_spans"]:
        c["event_index"] = new_pos[c["event_index"]]
    for g in ep["labels"].get("write_gates", []):
        for k in INDEX_KEYS:
            if k in g:
                g[k] = new_pos[g[k]]
    return ep


def delete_events(ep, idxs):
    idxs = set(idxs)
    keep = [i for i in range(len(ep["events"])) if i not in idxs]
    pos = {old: new for new, old in enumerate(keep)}
    ep["events"] = [ep["events"][i] for i in keep]
    ep["labels"]["claim_spans"] = [dict(c, event_index=pos[c["event_index"]])
                                   for c in ep["labels"]["claim_spans"] if c["event_index"] in pos]
    ep["labels"]["write_gates"] = [dict(g, **{k: pos[g[k]] for k in INDEX_KEYS if k in g})
                                   for g in ep["labels"]["write_gates"]
                                   if all(g[k] in pos for k in INDEX_KEYS if k in g)]
    return ep


def find(ep, pred):
    return next(i for i, e in enumerate(ep["events"]) if pred(e))


def call_of(ep, name):
    return find(ep, lambda e: e["type"] == "tool_call" and e["name"] == name)


def set_arg(ep, name, key, value):
    i = call_of(ep, name)
    if value is None:
        ep["events"][i]["args"].pop(key)
    else:
        ep["events"][i]["args"][key] = value
    return ep


class Base(unittest.TestCase):
    def setUp(self):
        self.book = build_episode("book_after_availability_free", 3)
        self.assertEqual(codes(self.book), set())

    def fresh(self):
        return copy.deepcopy(self.book)


class TestStructure(Base):
    def test_valid_episode_passes(self):
        self.assertEqual(validate_episode(self.book), [])

    def test_missing_field(self):
        ep = self.fresh()
        del ep["now"]
        self.assertIn("E_SCHEMA", codes(ep))

    def test_bad_event(self):
        ep = self.fresh()
        ep["events"][0]["dur"] = 0
        self.assertIn("E_EVENT_SCHEMA", codes(ep))

    def test_events_out_of_order(self):
        ep = self.fresh()
        ep["events"][0], ep["events"][1] = ep["events"][1], ep["events"][0]
        self.assertIn("E_ORDER", codes(ep))

    def test_call_ids_not_sequential(self):
        ep = self.fresh()
        i = call_of(ep, "check_availability")
        cid = ep["events"][i]["id"]
        for e in ep["events"]:
            if e.get("id") == cid:
                e["id"] = "c9"
        for c in ep["labels"]["claim_spans"]:
            if c["depends_on"] == cid:
                c["depends_on"] = "c9"
        self.assertIn("E_CALL_ID", codes(ep))

    def test_result_before_call(self):
        ep = self.fresh()
        r = find(ep, lambda e: e["type"] == "tool_result")
        c = find(ep, lambda e: e["type"] == "tool_call" and e["id"] == ep["events"][r]["id"])
        ep["events"][r]["t"] = ep["events"][c]["t"] - 0.01
        self.assertIn("E_RESULT_BEFORE_CALL", codes(ep))

    def test_latency_mismatch(self):
        ep = self.fresh()
        r = find(ep, lambda e: e["type"] == "tool_result")
        ep["events"][r]["latency_s"] += 0.5
        self.assertIn("E_LATENCY_MISMATCH", codes(ep))

    def test_missing_and_orphan_results(self):
        ep = self.fresh()
        r = find(ep, lambda e: e["type"] == "tool_result")
        delete_events(ep, [r])
        self.assertIn("E_RESULT_MISSING", codes(ep))
        ep = self.fresh()
        r = find(ep, lambda e: e["type"] == "tool_result")
        ep["events"][r]["id"] = "c77"
        self.assertIn("E_RESULT_ORPHAN", codes(ep))

    def test_expected_calls_mismatch(self):
        ep = self.fresh()
        ep["labels"]["expected_calls"].pop()
        self.assertIn("E_EXPECTED_CALLS", codes(ep))

    def test_marker_in_text(self):
        ep = self.fresh()
        i = find(ep, lambda e: e["type"] == "speech" and e["ch"] == "elle")
        ep["events"][i]["text"] += " <tool>{}</tool>"
        self.assertIn("E_MARKER_IN_TEXT", codes(ep))

    def test_self_overlap(self):
        ep = self.fresh()
        i = find(ep, lambda e: e["type"] == "speech" and e["ch"] == "elle")
        ep["events"][i]["dur"] += 400
        self.assertIn("E_SELF_OVERLAP", codes(ep))

    def test_claim_span_pointing_at_caller(self):
        ep = self.fresh()
        ep["labels"]["claim_spans"][0]["event_index"] = find(ep, lambda e: e["type"] == "speech" and e["ch"] == "caller")
        self.assertIn("E_CLAIM_INDEX", codes(ep))


class TestArgs(Base):
    def test_start_without_offset(self):
        ep = set_arg(self.fresh(), "check_availability", "start", "2026-11-12T14:00:00")
        self.assertIn("E_ARG_START_OFFSET", codes(ep))

    def test_start_unparseable(self):
        ep = set_arg(self.fresh(), "check_availability", "start", "Thursday at two")
        self.assertIn("E_ARG_START_PARSE", codes(ep))

    def test_start_wrong_dst_offset(self):
        ep = self.fresh()
        i = call_of(ep, "check_availability")
        s = ep["events"][i]["args"]["start"]
        wrong = s[:-6] + ("-08:00" if s.endswith("-07:00") else "-07:00")
        ep["events"][i]["args"]["start"] = wrong
        self.assertIn("E_ARG_START_TZ", codes(ep))

    def test_start_in_past(self):
        ep = self.fresh()
        ep["now"] = "2027-12-01T10:00:00-08:00"
        self.assertIn("E_ARG_START_PAST", codes(ep))

    def test_start_off_boundary(self):
        ep = self.fresh()
        i = call_of(ep, "check_availability")
        s = ep["events"][i]["args"]["start"]
        ep["events"][i]["args"]["start"] = s[:14] + "15" + s[16:]
        self.assertIn("E_ARG_START_BOUNDARY", codes(ep))

    def test_start_too_far(self):
        ep = self.fresh()
        ep["now"] = "2025-01-01T10:00:00-08:00"
        self.assertIn("E_ARG_START_FAR", codes(ep))

    def test_name_email_topic_are_never_arguments(self):
        for key, value in (("name", "Dana Whitfield"), ("email", "dana@example.com"), ("topic", "a backend role")):
            ep = set_arg(self.fresh(), "book_intro_call", key, value)
            self.assertLessEqual({"E_ARG_FREE_TEXT", "E_ARGS_SCHEMA"}, codes(ep), key)

    def test_missing_or_extra_arg(self):
        ep = set_arg(self.fresh(), "book_intro_call", "start", None)
        self.assertIn("E_ARGS_SCHEMA", codes(ep))
        ep = set_arg(self.fresh(), "get_caller_details", "start", "2026-11-12T14:00:00-08:00")
        self.assertIn("E_ARGS_SCHEMA", codes(ep))

    def test_save_user_fact_is_gone(self):
        ep = self.fresh()
        i = call_of(ep, "get_caller_details")
        ep["events"][i]["name"] = "save_user_fact"
        ep["events"][i]["args"] = {"category": "name", "fact": "Their name is Dana."}
        ep["labels"]["expected_calls"] = [{"name": e["name"], "args": e["args"]}
                                          for e in ep["events"] if e["type"] == "tool_call"]
        self.assertLessEqual({"E_SAVE_USER_FACT", "E_ARG_FREE_TEXT"}, codes(ep))

    def test_unknown_tool(self):
        ep = self.fresh()
        ep["events"][call_of(ep, "check_availability")]["name"] = "reschedule_booking"
        self.assertIn("E_UNKNOWN_TOOL", codes(ep))


class TestBehaviour(Base):
    def test_premature_claim(self):
        ep = self.fresh()
        cs = ep["labels"]["claim_spans"][0]
        claim = ep["events"][cs["event_index"]]
        r = find(ep, lambda e: e["type"] == "tool_result" and e["id"] == cs["depends_on"])
        res = ep["events"][r]
        res["t"] = round(claim["t"] + 0.4, 2)
        c = find(ep, lambda e: e["type"] == "tool_call" and e["id"] == cs["depends_on"])
        res["latency_s"] = round(res["t"] - ep["events"][c]["t"], 2)
        resort(ep)
        self.assertIn("E_PREMATURE_CLAIM", codes(ep))
        self.assertIn("premature_claim", violations(ep))

    def book_gate(self, ep):
        names = {e["id"]: e["name"] for e in ep["events"] if e["type"] == "tool_call"}
        return next(g for g in ep["labels"]["write_gates"] if names[g["call_id"]] == "book_intro_call")

    def test_write_without_gate(self):
        ep = self.fresh()
        ep["labels"]["write_gates"].remove(self.book_gate(ep))
        self.assertIn("E_UNGATED_WRITE", codes(ep))
        self.assertIn("write_without_intent", violations(ep))

    def test_intent_must_be_an_accept_pick_or_book_intent(self):
        ep = self.fresh()
        intent = ep["events"][self.book_gate(ep)["intent_index"]]
        intent["act"] = "thanks"
        intent.pop("book_intent", None)
        self.assertIn("E_GATE_INVALID", codes(ep))

    def test_intent_for_another_time(self):
        ep = self.fresh()
        intent = ep["events"][self.book_gate(ep)["intent_index"]]
        s = intent["refers"]["start"]
        intent["refers"]["start"] = s[:11] + ("09" if s[11:13] != "09" else "10") + s[13:]
        self.assertIn("E_GATE_INVALID", codes(ep))

    def test_intent_voided_by_a_later_decline(self):
        ep = self.fresh()
        g = self.book_gate(ep)
        j = next(i for i in range(g["intent_index"] + 1, len(ep["events"]))
                 if ep["events"][i]["type"] == "speech" and ep["events"][i]["ch"] == "caller"
                 and i < call_of(ep, "book_intro_call"))
        ep["events"][j]["act"] = "decline"
        self.assertIn("E_GATE_INVALID", codes(ep))

    def test_booking_needs_details_on_file_or_heard(self):
        ep = self.fresh()
        d = call_of(ep, "get_caller_details")
        r = find(ep, lambda e: e["type"] == "tool_result" and e["id"] == ep["events"][d]["id"])
        ep["events"][r]["content"] = "MISSING: email. Ask for it."
        self.assertIn("E_GATE_INVALID", codes(ep))
        ep = self.fresh()
        self.book_gate(ep)["details_call_id"] = ep["events"][call_of(ep, "check_availability")]["id"]
        self.assertIn("E_GATE_INVALID", codes(ep))

    def test_booking_before_the_details_result(self):
        ep = self.fresh()
        d = call_of(ep, "get_caller_details")
        b = call_of(ep, "book_intro_call")
        rb = find(ep, lambda e: e["type"] == "tool_result" and e["id"] == ep["events"][b]["id"])
        ep["events"][b]["t"] = round(ep["events"][d]["t"] + 0.01, 2)
        ep["events"][rb]["latency_s"] = round(ep["events"][rb]["t"] - ep["events"][b]["t"], 2)
        resort(ep)
        self.assertIn("E_GATE_INVALID", codes(ep))

    def test_booking_needs_a_free_check_for_that_start(self):
        ep = self.fresh()
        c = call_of(ep, "check_availability")
        r = find(ep, lambda e: e["type"] == "tool_result" and e["id"] == ep["events"][c]["id"])
        ep["events"][r]["content"] = "BUSY: that time is taken. Ask for another day."
        self.assertIn("E_GATE_INVALID", codes(ep))
        ep = self.fresh()
        r = find(ep, lambda e: e["type"] == "tool_result" and e["id"] == ep["events"][c]["id"])
        ep["events"][r]["content"] = "UNKNOWN: couldn't read the calendar just now."
        self.assertIn("E_GATE_INVALID", codes(ep))  # UNKNOWN never licenses a booking

    def test_nothing_is_booked_after_unknown(self):
        ends = set()
        for seed in range(80):
            ep = build_episode("availability_unknown_recovery", seed)
            self.assertEqual(codes(ep), set())
            self.assertNotIn("request", json.dumps(ep["labels"]))
            names = {e["id"]: e for e in ep["events"] if e["type"] == "tool_call"}
            last = {}
            for e in ep["events"]:
                if e["type"] == "tool_result" and names[e["id"]]["name"] == "check_availability":
                    last[names[e["id"]]["args"]["start"]] = e["content"].split(":")[0]
                elif e["type"] == "tool_call" and e["name"] == "book_intro_call":
                    self.assertEqual(last.get(e["args"]["start"]), "FREE")
            checks = [e["content"].split(":")[0] for e in ep["events"] if e["type"] == "tool_result"
                      and names[e["id"]]["name"] == "check_availability"]
            booked = any(e["name"] == "book_intro_call" for e in names.values())
            ends.add((tuple(checks[:2]), booked))
            if checks[:2] in (["UNKNOWN"], ["UNKNOWN", "UNKNOWN"]):
                self.assertEqual(len(names), len(checks))  # no call after giving up
        self.assertLessEqual({(("UNKNOWN",), False), (("UNKNOWN", "UNKNOWN"), False), (("UNKNOWN", "FREE"), True)}, ends)

    def test_repeat_booking_needs_a_correction_and_updated(self):
        ep = build_episode("book_email_corrected", 2)
        self.assertEqual(codes(ep), set())
        books = [e for e in ep["events"] if e["type"] == "tool_call" and e["name"] == "book_intro_call"]
        self.assertEqual(len(books), 2)
        broken = copy.deepcopy(ep)
        r = find(broken, lambda e: e["type"] == "tool_result" and e["id"] == books[1]["id"])
        broken["events"][r]["content"] = broken["events"][r]["content"].replace("Updated:", "Done:")
        self.assertIn("E_GATE_INVALID", codes(broken))
        broken = copy.deepcopy(ep)
        fix = find(broken, lambda e: e.get("act") == "correction")
        broken["events"][fix]["act"] = "thanks"
        self.assertIn("E_GATE_INVALID", codes(broken))
        first = copy.deepcopy(self.book)
        r = find(first, lambda e: e["type"] == "tool_result" and e["content"].startswith("Done:"))
        first["events"][r]["content"] = "Updated: the invite now goes to dana at example dot com."
        self.assertIn("E_GATE_INVALID", codes(first))

    def test_consent_write_needs_the_question_and_a_yes(self):
        ep = None
        for seed in range(40):
            cand = build_episode("book_after_availability_free", seed)
            if any(e["type"] == "tool_call" and e["name"] == "record_sms_consent" for e in cand["events"]):
                ep = cand
                break
        self.assertEqual(codes(ep), set())
        cid = ep["events"][call_of(ep, "record_sms_consent")]["id"]
        gate = next(g for g in ep["labels"]["write_gates"] if g["call_id"] == cid)
        self.assertEqual(set(gate), {"call_id", "question_index", "confirm_index"})
        broken = copy.deepcopy(ep)
        broken["events"][gate["confirm_index"]]["act"] = "decline"
        self.assertIn("E_GATE_INVALID", codes(broken))
        broken = copy.deepcopy(ep)
        broken["labels"]["write_gates"] = [g for g in broken["labels"]["write_gates"] if g["call_id"] != cid]
        self.assertIn("E_UNGATED_WRITE", codes(broken))

    def booked_event(self, ep):
        return ep["events"][find(ep, lambda e: e.get("act") == "booked")]

    def test_post_booking_sentence_is_required_and_comes_from_results(self):
        ep = self.fresh()
        self.booked_event(ep)["act"] = "claim"
        self.assertIn("E_BOOKED_CLAIM", codes(ep))
        ep = self.fresh()
        self.booked_event(ep)["refers"]["email"] = "someone.else@example.com"
        self.assertIn("E_BOOKED_CLAIM", codes(ep))
        ep = self.fresh()
        ev = self.booked_event(ep)
        ev["text"] = "You're all booked. See you then!"
        self.assertIn("E_BOOKED_CLAIM", codes(ep))
        ep = self.fresh()
        ev = self.booked_event(ep)
        i = ep["events"].index(ev)
        ep["labels"]["claim_spans"] = [c for c in ep["labels"]["claim_spans"]
                                       if not (c["event_index"] == i and c["depends_on"] == ev["refers"]["details"])]
        self.assertIn("E_BOOKED_CLAIM", codes(ep))

    def test_readback_act_is_rejected(self):
        ep = self.fresh()
        i = find(ep, lambda e: e["type"] == "speech" and e["ch"] == "elle" and e.get("act") == "pre_call")
        ep["events"][i]["act"] = "readback"
        self.assertIn("E_READBACK", codes(ep))

    def test_caller_on_file_is_not_asked(self):
        ep = None
        for seed in range(40):
            cand = build_episode("book_after_availability_free", seed)
            if cand["context"]["caller_on_file"] is None:
                ep = cand
                break
        self.assertEqual(codes(ep), set())
        self.assertTrue(any(e.get("act") == "ask_name" for e in ep["events"]))
        ep["context"]["caller_on_file"] = {"name": "Rosa Greenleaf", "email": "rosa.greenleaf@example.org"}
        self.assertIn("E_ON_FILE_ASK", codes(ep))
        ep["context"]["caller_on_file"] = {"name": "Rosa Greenleaf"}
        self.assertIn("E_SCHEMA", codes(ep))

    def test_dead_air(self):
        for seed in range(50):
            ep = build_episode("availability_slow_filler", seed)
            c = call_of(ep, "check_availability")
            call = ep["events"][c]
            r = find(ep, lambda e: e["type"] == "tool_result" and e["id"] == call["id"])
            if ep["events"][r]["latency_s"] > 1.0:
                break
        res = ep["events"][r]
        inside = [i for i, e in enumerate(ep["events"]) if e["type"] == "speech"
                  and e["t"] < res["t"] and e["t"] + e["dur"] > call["t"]]
        delete_events(ep, inside)
        self.assertIn("E_DEAD_AIR", codes(ep))
        self.assertIn("dead_air", violations(ep))

    def test_narrated_save(self):
        ep = self.fresh()
        i = find(ep, lambda e: e["type"] == "speech" and e["ch"] == "elle" and e.get("act") == "ask_name")
        ep["events"][i]["text"] = "Got it, I'll make a note of that. What's your name?"
        self.assertIn("E_NARRATED_SAVE", codes(ep))

    def test_name_before_a_result_or_the_caller_gave_it(self):
        ep = None
        for seed in range(40):
            cand = build_episode("book_after_availability_free", seed)
            if cand["context"]["caller_on_file"] is None:
                ep = cand
                break
        d = call_of(ep, "get_caller_details")
        r = find(ep, lambda e: e["type"] == "tool_result" and e["id"] == ep["events"][d]["id"])
        first = ep["events"][r]["content"].split(": ", 1)[1].split(" ")[0]
        i = find(ep, lambda e: e["type"] == "speech" and e["ch"] == "elle")
        ep["events"][i]["text"] = f"Hi {first}, good to hear from you again."
        self.assertIn("E_NAME_SOURCE", codes(ep))

    def test_negative_flag_on_positive(self):
        ep = self.fresh()
        ep["labels"]["negative"] = True
        ep["labels"]["negative_kind"] = "premature_claim"
        self.assertIn("E_NEGATIVE_LABEL", codes(ep))

    def test_negative_without_violation(self):
        ep = build_episode("neg_book_without_intent", 1)
        self.assertEqual(codes(ep), set())
        ep["labels"]["negative_kind"] = "dead_air"
        self.assertIn("E_NEGATIVE_LABEL", codes(ep))
        self.assertIn("E_UNGATED_WRITE", codes(ep))

    def test_negatives_are_detected(self):
        for name, kind in (("neg_premature_availability_claim", "premature_claim"),
                           ("neg_premature_bookings_claim", "premature_claim"),
                           ("neg_book_without_intent", "write_without_intent"),
                           ("neg_dead_air_wait", "dead_air")):
            ep = build_episode(name, 0)
            self.assertEqual(violations(ep), {kind}, name)
            pos = copy.deepcopy(ep)
            pos["scenario"] = name[4:]
            pos["labels"]["negative"] = False
            self.assertTrue(codes(pos) & {"E_PREMATURE_CLAIM", "E_UNGATED_WRITE", "E_DEAD_AIR"}, name)


class TestFiles(unittest.TestCase):
    def _write(self, d, name, eps):
        p = os.path.join(d, name)
        with open(p, "w") as f:
            for e in eps:
                f.write(json.dumps(e) + "\n")
        return p

    def test_split_overlap_and_holdout_leak(self):
        a = build_episode("availability_free", 5, split="train", ep_id="ep_000001")
        b = build_episode("availability_free", 5, split="test", ep_id="ep_000002")
        c = build_episode("book_slot_taken_race", 6, split="train", ep_id="ep_000003")
        with tempfile.TemporaryDirectory() as d:
            p1 = self._write(d, "train.jsonl", [a, c])
            p2 = self._write(d, "test.jsonl", [b])
            _, _, cnt = validate_files([p1, p2], holdout_family="slot_conflict")
            self.assertGreater(cnt["E_SPLIT_OVERLAP"], 0)
            self.assertGreater(cnt["E_HOLDOUT_LEAK"], 0)

    def test_ids_must_increase(self):
        a = build_episode("availability_free", 1, ep_id="ep_000005")
        b = build_episode("availability_free", 2, ep_id="ep_000004")
        with tempfile.TemporaryDirectory() as d:
            _, _, cnt = validate_files([self._write(d, "x.jsonl", [a, b])])
            self.assertGreater(cnt["E_EPISODE_ID"], 0)

    def test_split_label_mismatch(self):
        a = build_episode("availability_free", 1, split="train", ep_id="ep_000001")
        b = build_episode("availability_free", 2, split="val", ep_id="ep_000002")
        with tempfile.TemporaryDirectory() as d:
            _, _, cnt = validate_files([self._write(d, "x.jsonl", [a, b])])
            self.assertGreater(cnt["E_SPLIT_LABEL"], 0)


if __name__ == "__main__":
    unittest.main()
