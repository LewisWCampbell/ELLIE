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
        g["readback_index"] = new_pos[g["readback_index"]]
        g["confirm_index"] = new_pos[g["confirm_index"]]
    return ep


def delete_events(ep, idxs):
    idxs = set(idxs)
    keep = [i for i in range(len(ep["events"])) if i not in idxs]
    pos = {old: new for new, old in enumerate(keep)}
    ep["events"] = [ep["events"][i] for i in keep]
    ep["labels"]["claim_spans"] = [dict(c, event_index=pos[c["event_index"]])
                                   for c in ep["labels"]["claim_spans"] if c["event_index"] in pos]
    ep["labels"]["write_gates"] = [dict(g, readback_index=pos[g["readback_index"]], confirm_index=pos[g["confirm_index"]])
                                   for g in ep["labels"]["write_gates"]
                                   if g["readback_index"] in pos and g["confirm_index"] in pos]
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

    def test_invalid_email(self):
        ep = set_arg(self.fresh(), "book_intro_call", "email", "dana at example")
        self.assertIn("E_ARG_EMAIL", codes(ep))

    def test_missing_required_arg(self):
        ep = set_arg(self.fresh(), "book_intro_call", "email", None)
        self.assertIn("E_ARGS_SCHEMA", codes(ep))

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

    def test_write_without_gate(self):
        ep = self.fresh()
        ep["labels"]["write_gates"] = [g for g in ep["labels"]["write_gates"]
                                       if not any(e.get("id") == g["call_id"] and e["name"] == "book_intro_call"
                                                  for e in ep["events"] if e["type"] == "tool_call")]
        self.assertIn("E_UNGATED_WRITE", codes(ep))

    def test_write_before_yes(self):
        ep = self.fresh()
        g = next(g for g in ep["labels"]["write_gates"]
                 if any(e["type"] == "tool_call" and e["id"] == g["call_id"] and e["name"] == "book_intro_call"
                        for e in ep["events"]))
        confirm = ep["events"][g["confirm_index"]]
        c = find(ep, lambda e: e["type"] == "tool_call" and e["id"] == g["call_id"])
        r = find(ep, lambda e: e["type"] == "tool_result" and e["id"] == g["call_id"])
        ep["events"][c]["t"] = round(confirm["t"] + 0.01, 2)
        ep["events"][r]["latency_s"] = round(ep["events"][r]["t"] - ep["events"][c]["t"], 2)
        resort(ep)
        self.assertIn("E_GATE_INVALID", codes(ep))

    def test_readback_mismatch(self):
        ep = self.fresh()
        g = ep["labels"]["write_gates"][0]
        rb = ep["events"][g["readback_index"]]
        if "refers" in rb:
            rb["refers"]["email"] = "someone.else@example.com"
            self.assertIn("E_GATE_INVALID", codes(ep))

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

    def test_name_from_notes(self):
        ep = self.fresh()
        ep["context"]["notes_names"] = ["Zephyrine Quux"]
        i = find(ep, lambda e: e["type"] == "speech" and e["ch"] == "elle")
        ep["events"][i]["text"] = "Hi Zephyrine, good to hear from you again."
        self.assertIn("E_NOTES_NAME", codes(ep))

    def test_negative_flag_on_positive(self):
        ep = self.fresh()
        ep["labels"]["negative"] = True
        ep["labels"]["negative_kind"] = "premature_claim"
        self.assertIn("E_NEGATIVE_LABEL", codes(ep))

    def test_negative_without_violation(self):
        ep = build_episode("neg_book_without_readback", 1)
        self.assertEqual(codes(ep), set())
        ep["labels"]["negative_kind"] = "dead_air"
        self.assertIn("E_NEGATIVE_LABEL", codes(ep))
        self.assertIn("E_UNGATED_WRITE", codes(ep))

    def test_negatives_are_detected(self):
        for name, kind in (("neg_premature_availability_claim", "premature_claim"),
                           ("neg_premature_bookings_claim", "premature_claim"),
                           ("neg_book_without_readback", "write_without_readback"),
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
