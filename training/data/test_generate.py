import json
import os
import sys
import tempfile
import unittest
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import generate  # noqa: E402
from generate import build_episode, generate_split  # noqa: E402
from scenarios import SCENARIOS  # noqa: E402
from validate import validate_episode, validate_files, violations  # noqa: E402


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
                       "No bookings found", "Saved."):
            self.assertIn(marker, results)

    def test_no_real_contact_details(self):
        eps = generate_split("train", 200, base_seed=4)
        for e in eps:
            self.assertRegex(e["context"]["caller_number"], r"^\+1\d{3}55501\d{2}$")
            for ev in e["events"]:
                if ev["type"] == "tool_call" and "email" in ev["args"]:
                    self.assertRegex(ev["args"]["email"], r"@(mail\.)?example\.(com|org|net)$")


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
