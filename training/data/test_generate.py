import json
import os
import sys
import tempfile
import unittest
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import generate  # noqa: E402
from generate import build_episode, generate_split  # noqa: E402
from episode import spoken_email  # noqa: E402
from scenarios import SCENARIOS  # noqa: E402
from serialize import extract_tool_calls, serialize_episode, to_text_and_mask  # noqa: E402
from validate import TENTATIVE_WORDS, parse_details, validate_episode, validate_files, violations  # noqa: E402

V3_TOOLS = {"check_availability", "get_my_bookings", "get_caller_details", "book_intro_call", "record_sms_consent"}


def all_episodes(seeds=range(8)):
    """Every scenario in every language it supports: 8 seeds each is 400 episodes."""
    return [build_episode(s.name, seed, lang=lang) for s in SCENARIOS for lang in s.langs for seed in seeds]


def calls_of(ep):
    return [e for e in ep["events"] if e["type"] == "tool_call"]


def results_of(ep):
    return {e["id"]: (i, e) for i, e in enumerate(ep["events"]) if e["type"] == "tool_result"}


def dumps(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True)


class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_episode(self):
        for s in SCENARIOS:
            self.assertEqual(dumps(build_episode(s.name, 42)), dumps(build_episode(s.name, 42)), s.name)

    def test_different_seeds_differ(self):
        a = build_episode("book_after_availability_free", 1)
        b = build_episode("book_after_availability_free", 2)
        self.assertNotEqual(dumps(a["events"]), dumps(b["events"]))

    def test_base_seed_changes_output(self):
        a = build_episode("availability_free", 5, base_seed=0)
        b = build_episode("availability_free", 5, base_seed=1)
        self.assertNotEqual(dumps(a), dumps(b))

    def test_split_generation_is_reproducible(self):
        a = generate_split("val", 40, base_seed=3)
        b = generate_split("val", 40, base_seed=3)
        self.assertEqual([dumps(e) for e in a], [dumps(e) for e in b])

    def test_cli_writes_identical_files(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            for d in (d1, d2):
                generate.main(["--out", d, "--train", "30", "--val", "10", "--test", "20", "--seed", "11", "--no-stats"])
            for name in ("sample_train.jsonl", "sample_val.jsonl", "sample_test.jsonl"):
                with open(os.path.join(d1, name), "rb") as f1, open(os.path.join(d2, name), "rb") as f2:
                    self.assertEqual(f1.read(), f2.read(), name)


class TestCoverage(unittest.TestCase):
    def test_at_least_twenty_scenarios(self):
        self.assertGreaterEqual(len(SCENARIOS), 20)

    def test_every_scenario_and_language_validates(self):
        for s in SCENARIOS:
            for lang in s.langs:
                for seed in range(25):
                    ep = build_episode(s.name, seed, lang=lang)
                    issues = validate_episode(ep)
                    self.assertEqual(issues, [], f"{s.name}/{lang}/{seed}: {issues[:3]}")

    def test_negatives_really_violate_and_positives_dont(self):
        for s in SCENARIOS:
            for seed in range(10):
                ep = build_episode(s.name, seed)
                v = violations(ep)
                if ep["labels"]["negative"]:
                    self.assertTrue(s.name.startswith("neg_"))
                    self.assertEqual(v, {ep["labels"]["negative_kind"]}, s.name)
                else:
                    self.assertEqual(v, set(), s.name)

    def test_required_behaviours_present(self):
        eps = generate_split("train", 400, base_seed=9, holdout_family=None)
        sc = Counter(e["scenario"] for e in eps)
        self.assertGreater(len(sc), 20)
        lat = [ev["latency_s"] for e in eps for ev in e["events"] if ev["type"] == "tool_result"]
        self.assertTrue(any(0.7 < x <= 3.0 for x in lat))
        self.assertTrue(any(ev.get("interrupted") for e in eps for ev in e["events"]))
        self.assertTrue(any(e["labels"]["lang"] != "en" for e in eps))
        self.assertTrue(any(e["labels"]["negative"] for e in eps))
        kinds = Counter(e["labels"]["kind"] for e in eps)
        self.assertGreater(kinds["read"], 0)
        self.assertGreater(kinds["mixed"], 0)
        results = " ".join(ev["content"] for e in eps for ev in e["events"] if ev["type"] == "tool_result")
        for marker in ("FREE:", "BUSY:", "Nearby open times:", "UNKNOWN:", "Done:", "Someone else booked",
                       "No bookings found", "Saved.", "ON FILE:", "HEARD:", "MISSING: email. Ask for it.",
                       "MISSING: email (what was heard was not a valid address). Ask for it again.", "Updated:"):
            self.assertIn(marker, results)
        share = sum(e["context"]["caller_on_file"] is not None for e in eps) / len(eps)
        self.assertTrue(0.3 < share < 0.5, share)

    def test_no_real_contact_details(self):
        eps = generate_split("train", 200, base_seed=4)
        for e in eps:
            self.assertRegex(e["context"]["caller_number"], r"^\+1\d{3}55501\d{2}$")
            if e["context"]["caller_on_file"]:
                self.assertRegex(e["context"]["caller_on_file"]["email"], r"@(mail\.)?example\.(com|org|net)$")
            for ev in e["events"]:
                if ev["type"] == "speech" and "email" in (ev.get("refers") or {}):
                    self.assertRegex(ev["refers"]["email"], r"@(mail\.)?example\.(com|org|net)$")


class TestContractV3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eps = all_episodes()

    def test_only_the_five_tools_and_no_free_text_args(self):
        self.assertGreaterEqual(len(self.eps), 300)
        seen = set()
        for ep in self.eps:
            for c in calls_of(ep):
                seen.add(c["name"])
                self.assertIn(c["name"], V3_TOOLS, ep["scenario"])
                self.assertNotEqual(c["name"], "save_user_fact")
                want = {"start"} if c["name"] in ("check_availability", "book_intro_call") else set()
                self.assertEqual(set(c["args"]), want, (ep["scenario"], c))
            self.assertEqual(ep["labels"]["expected_calls"],
                             [{"name": c["name"], "args": c["args"]} for c in calls_of(ep)])
            self.assertNotIn("save_user_fact", json.dumps(ep))
        self.assertEqual(seen, V3_TOOLS)

    def test_caller_on_file_is_never_asked_name_or_email(self):
        n_on = n_new_asked = 0
        for ep in self.eps:
            acts = [e.get("act") for e in ep["events"] if e["type"] == "speech" and e["ch"] == "elle"]
            on_file = ep["context"]["caller_on_file"]
            self.assertNotIn("notes", ep["context"])
            if on_file:
                n_on += 1
                self.assertEqual(set(on_file), {"name", "email"})
                self.assertNotIn("ask_name", acts, ep["scenario"])
                self.assertNotIn("ask_email", acts, ep["scenario"])
                self.assertNotIn("greeting", acts, ep["scenario"])  # "who am I speaking with?"
                for _, r in results_of(ep).values():
                    self.assertFalse(r["content"].startswith("MISSING"), ep["scenario"])
            elif any(c["name"] == "book_intro_call" for c in calls_of(ep)):
                n_new_asked += "ask_email" in acts
                self.assertNotIn("greeting_known", acts)
        self.assertGreater(n_on, 100)
        self.assertGreater(n_new_asked, 50)

    def test_no_readback_and_no_confirm_before_a_booking(self):
        for ep in self.eps:
            names = {c["id"]: c["name"] for c in calls_of(ep)}
            for e in ep["events"]:
                self.assertNotEqual(e.get("act"), "readback", ep["scenario"])
            for g in ep["labels"]["write_gates"]:
                self.assertNotIn("readback_index", g)
                if names[g["call_id"]] == "book_intro_call":
                    self.assertEqual(set(g), {"call_id", "intent_index", "details_call_id"})
                    intent = ep["events"][g["intent_index"]]
                    self.assertEqual(intent["ch"], "caller")
                    self.assertTrue(intent.get("act") in ("accept", "pick", "correction") or intent.get("book_intent"))
                    self.assertNotEqual(intent.get("act"), "confirm")
                else:
                    self.assertEqual(set(g), {"call_id", "question_index", "confirm_index"})
                    self.assertEqual(ep["events"][g["question_index"]]["act"], "sms_offer")

    def test_every_booking_follows_a_details_result_and_a_free_check(self):
        for ep in self.eps:
            res = results_of(ep)
            evs = ep["events"]
            for i, c in enumerate(evs):
                if c["type"] != "tool_call" or c["name"] != "book_intro_call":
                    continue
                before = [e for e in evs[:i] if e["type"] == "tool_result"]
                names = {x["id"]: x for x in calls_of(ep)}
                det = [r for r in before if names[r["id"]]["name"] == "get_caller_details"]
                self.assertTrue(det, ep["scenario"])
                self.assertIsNotNone(parse_details(det[-1]["content"]), (ep["scenario"], det[-1]["content"]))
                chk = [r for r in before if names[r["id"]]["name"] == "check_availability"
                       and names[r["id"]]["args"]["start"] == c["args"]["start"]]
                self.assertTrue(chk, ep["scenario"])
                self.assertTrue(chk[-1]["content"].startswith("FREE"), ep["scenario"])

    def test_every_booking_is_followed_by_a_post_booking_claim(self):
        """After each Done/Updated result Elle says, as a claim on that result and
        on the details result, where the invite goes (the email exactly as the
        details result spelled it, in the call's language) and that it is tentative."""
        n = 0
        for ep in self.eps:
            res = results_of(ep)
            evs = ep["events"]
            deps = {}
            for cs in ep["labels"]["claim_spans"]:
                deps.setdefault(cs["event_index"], set()).add(cs["depends_on"])
            for c in calls_of(ep):
                if c["name"] != "book_intro_call" or not res[c["id"]][1]["content"].startswith(("Done:", "Updated:")):
                    continue
                n += 1
                said = [(i, e) for i, e in enumerate(evs) if e.get("act") in ("booked", "booked_update")
                        and c["id"] in deps.get(i, ())]
                self.assertEqual(len(said), 1, ep["scenario"])
                i, ev = said[0]
                self.assertGreater(ev["t"], res[c["id"]][1]["t"])
                did = ev["refers"]["details"]
                self.assertIn(did, deps[i])
                self.assertGreater(ev["t"], res[did][1]["t"])
                status, name, email_sp = parse_details(res[did][1]["content"])
                self.assertEqual(email_sp, spoken_email(ev["refers"]["email"]))
                text = (ev.get("full_text") or ev["text"]).lower()
                self.assertIn(spoken_email(ev["refers"]["email"], ev.get("lang", "en")).lower(), text)
                self.assertTrue(any(w in text for w in TENTATIVE_WORDS), text)
                self.assertNotIn(name.lower(), text)  # the name is not repeated back
                if ep["context"]["caller_on_file"] and not any(e.get("act") == "correction" for e in evs):
                    self.assertEqual(status, "ON FILE")
                    self.assertEqual(name, ep["context"]["caller_on_file"]["name"])
        self.assertGreater(n, 150)

    def test_email_correction_is_an_update_of_the_same_start(self):
        for name in ("book_email_corrected", "interrupt_booked_fix_email"):
            for seed in range(10):
                ep = build_episode(name, seed)
                res = results_of(ep)
                books = [c for c in calls_of(ep) if c["name"] == "book_intro_call"]
                self.assertEqual(len(books), 2)
                self.assertEqual(books[0]["args"], books[1]["args"])
                self.assertTrue(res[books[0]["id"]][1]["content"].startswith("Done:"))
                self.assertTrue(res[books[1]["id"]][1]["content"].startswith("Updated:"))
                det = [res[c["id"]][1]["content"] for c in calls_of(ep) if c["name"] == "get_caller_details"]
                self.assertEqual(len(det), 2)
                self.assertTrue(det[1].startswith("HEARD:"))
                self.assertNotEqual(det[0].split(": ", 1)[1], det[1].split(": ", 1)[1])
                fix = [i for i, e in enumerate(ep["events"]) if e.get("act") == "correction"]
                self.assertEqual(len(fix), 1)
                gate = {g["call_id"]: g for g in ep["labels"]["write_gates"]}[books[1]["id"]]
                self.assertEqual(gate["intent_index"], fix[0])

    def test_time_change_is_a_new_check_and_a_new_booking(self):
        for seed in range(10):
            ep = build_episode("time_change_after_booking", seed)
            names = [c["name"] for c in calls_of(ep)]
            self.assertEqual(names, ["check_availability", "get_caller_details", "book_intro_call"] * 2)
            starts = [c["args"]["start"] for c in calls_of(ep) if c["name"] == "book_intro_call"]
            self.assertNotEqual(starts[0], starts[1])
            self.assertTrue(any("earlier" in e["text"] for e in ep["events"] if e.get("act") == "booked"))

    def test_no_booking_when_the_caller_declines_or_refuses(self):
        for seed in range(20):
            ep = build_episode("book_declined_before_write", seed)
            self.assertEqual([c["name"] for c in calls_of(ep)], ["check_availability"])
        refused = 0
        for seed in range(40):
            ep = build_episode("book_missing_email", seed)
            contents = [r["content"] for _, r in results_of(ep).values()]
            self.assertIn("MISSING: email. Ask for it.", contents)
            if any(e.get("act") == "decline" for e in ep["events"]):
                refused += 1
                self.assertNotIn("book_intro_call", [c["name"] for c in calls_of(ep)])
        self.assertGreater(refused, 0)

    def test_compact_stream_round_trips_every_episode(self):
        lens = []
        for ep in self.eps:
            text, _ = to_text_and_mask(serialize_episode(ep))
            got = extract_tool_calls(text)
            self.assertEqual([{"name": c["name"], "args": c["args"]} for c in got], ep["labels"]["expected_calls"])
            self.assertEqual([c["id"] for c in got], [c["id"] for c in calls_of(ep)])
            lens += [len(s.text) for s in serialize_episode(ep) if s.kind == "tool_call"]
        self.assertLessEqual(max(lens), len("<tool>check_availability 2026-11-12T14:00</tool>"))


def _situation(ep, upto):
    """What the conversation looked like when a decision was made: the caller's
    dialogue acts and booking-intent marks, and the first word of every tool
    result (FREE, BUSY, HEARD, MISSING, ...), in order, up to event `upto`."""
    sig = [("on_file", ep["context"]["caller_on_file"] is not None)]
    for e in ep["events"][:upto]:
        if e["type"] == "speech" and e["ch"] == "caller" and e.get("act") not in ("backchannel", "ramble", "bye"):
            sig.append(("caller", e.get("act"), bool(e.get("book_intent"))))
        elif e["type"] == "tool_result":
            sig.append(("result", e["content"].split(":")[0].split(" ")[0]))
    return tuple(sig)


class TestCallPresenceIsDeterministic(unittest.TestCase):
    def test_same_scenario_and_seed_same_calls(self):
        for s in SCENARIOS:
            for seed in range(5):
                a = build_episode(s.name, seed)["labels"]["expected_calls"]
                b = build_episode(s.name, seed)["labels"]["expected_calls"]
                self.assertEqual(dumps(a), dumps(b), s.name)

    def test_no_coin_flip_about_whether_a_tool_is_called(self):
        """Proxy for "the same situation always leads to the same calls".

        A situation is the sequence of caller dialogue acts (with their
        book_intent marks), whether the caller is on file, and the
        first word of each tool result so far (`_situation`). Wording, timing,
        latency and Elle's own lines are left out on purpose: they may vary.
        For every caller turn in a few hundred positive episodes per scenario
        we record which tools Elle calls between that turn and the caller's
        next turn. If call presence were a coin flip (as save_user_fact after a
        name and the optional topic were in v1), two episodes of the same
        scenario with the same situation would differ in that list. They must
        not. The check is per scenario because a scenario fixes the flow, and
        the flow (for example whether the caller goes on to book) is allowed
        to be random."""
        for name in ("book_after_availability_free", "book_after_availability_busy", "chat_then_book",
                     "book_direct_details_first", "book_missing_email", "book_write_error",
                     "returning_caller", "availability_unknown_recovery", "book_email_corrected"):
            seen = {}
            for seed in range(150):
                ep = build_episode(name, seed)
                evs = ep["events"]
                turns = [i for i, e in enumerate(evs) if e["type"] == "speech" and e["ch"] == "caller"
                         and e.get("act") not in ("backchannel", "ramble", "bye")]
                for k, i in enumerate(turns):
                    nxt = turns[k + 1] if k + 1 < len(turns) else len(evs)
                    called = tuple(e["name"] for e in evs[i:nxt] if e["type"] == "tool_call")
                    key = _situation(ep, i + 1)
                    self.assertEqual(seen.setdefault(key, called), called, (name, seed, key))

    def test_details_call_precedes_every_booking_call_directly(self):
        """No booking without its own details read: the call before each
        book_intro_call is always get_caller_details, in every scenario."""
        for ep in all_episodes(range(4)):
            names = [c["name"] for c in calls_of(ep)]
            for k, n in enumerate(names):
                if n == "book_intro_call":
                    self.assertEqual(names[k - 1], "get_caller_details", ep["scenario"])


class TestSplits(unittest.TestCase):
    def test_holdout_and_disjoint(self):
        with tempfile.TemporaryDirectory() as d:
            generate.main(["--out", d, "--train", "150", "--val", "30", "--test", "60", "--seed", "2",
                           "--holdout-family", "slot_conflict"])
            paths = [os.path.join(d, f"sample_{s}.jsonl") for s in ("train", "val", "test")]
            n, errs, codes = validate_files(paths, holdout_family="slot_conflict")
            self.assertEqual(n, 240)
            self.assertEqual(sum(codes.values()), 0, errs[:5])
            fams = {s: {json.loads(l)["labels"]["family"] for l in open(p)} for s, p in zip(("train", "val", "test"), paths)}
            self.assertNotIn("slot_conflict", fams["train"] | fams["val"])
            self.assertIn("slot_conflict", fams["test"])
            self.assertTrue(os.path.exists(os.path.join(d, "STATS.md")))


if __name__ == "__main__":
    unittest.main()
