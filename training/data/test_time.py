import os
import random
import re
import sys
import unittest
from datetime import date, datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import elle_time as T  # noqa: E402

LA = T.TZ


class TestDST(unittest.TestCase):
    def test_offsets_either_side_of_fall_back(self):
        # DST ends 2026-11-01 at 02:00 local.
        self.assertEqual(T.iso(T.local_dt(date(2026, 10, 30), 14, 0)), "2026-10-30T14:00:00-07:00")
        self.assertEqual(T.iso(T.local_dt(date(2026, 11, 2), 14, 0)), "2026-11-02T14:00:00-08:00")

    def test_offsets_either_side_of_spring_forward(self):
        # DST starts 2027-03-14 at 02:00 local.
        self.assertEqual(T.iso(T.local_dt(date(2027, 3, 12), 9, 30)), "2027-03-12T09:30:00-08:00")
        self.assertEqual(T.iso(T.local_dt(date(2027, 3, 15), 9, 30)), "2027-03-15T09:30:00-07:00")

    def test_weekday_reference_across_fall_back(self):
        now = datetime(2026, 10, 29, 10, 15, tzinfo=LA)  # Thursday, PDT
        ref = T.TimeRef("weekday", 10, 0, weekday=0)  # "Monday at ten"
        self.assertEqual(T.iso(T.resolve(ref, now)), "2026-11-02T10:00:00-08:00")

    def test_tomorrow_across_spring_forward(self):
        now = datetime(2027, 3, 13, 16, 0, tzinfo=LA)  # Saturday before the change, PST
        ref = T.TimeRef("tomorrow", 14, 30)
        self.assertEqual(T.iso(T.resolve(ref, now)), "2027-03-14T14:30:00-07:00")

    def test_month_day_and_dom_across_dst(self):
        now = datetime(2026, 10, 20, 9, 0, tzinfo=LA)
        self.assertEqual(T.iso(T.resolve(T.TimeRef("month_day", 15, 0, day=12, month=11), now)),
                         "2026-11-12T15:00:00-08:00")
        self.assertEqual(T.iso(T.resolve(T.TimeRef("dom", 14, 0, day=3), now)),
                         "2026-11-03T14:00:00-08:00")  # the next 3rd after Oct 20

    def test_offset_checker(self):
        self.assertTrue(T.offset_is_correct("2026-07-01T10:00:00-07:00"))
        self.assertFalse(T.offset_is_correct("2026-07-01T10:00:00-08:00"))
        self.assertFalse(T.offset_is_correct("2026-12-01T10:00:00-07:00"))
        self.assertFalse(T.has_offset("2026-12-01T10:00:00"))


class TestResolutionConventions(unittest.TestCase):
    def test_weekday_is_strictly_after_today(self):
        now = datetime(2026, 11, 12, 9, 0, tzinfo=LA)  # Thursday
        self.assertEqual(T.resolve(T.TimeRef("weekday", 14, 0, weekday=3), now).date(), date(2026, 11, 19))
        self.assertEqual(T.resolve(T.TimeRef("weekday", 14, 0, weekday=4), now).date(), date(2026, 11, 13))

    def test_next_week_is_following_calendar_week(self):
        now = datetime(2026, 11, 10, 9, 0, tzinfo=LA)  # Tuesday
        self.assertEqual(T.resolve(T.TimeRef("next_week", 10, 0, weekday=1), now).date(), date(2026, 11, 17))

    def test_dom_rolls_into_next_month(self):
        now = datetime(2026, 11, 20, 9, 0, tzinfo=LA)
        self.assertEqual(T.resolve(T.TimeRef("dom", 14, 0, day=14), now).date(), date(2026, 12, 14))

    def test_month_day_rolls_into_next_year(self):
        now = datetime(2026, 12, 20, 9, 0, tzinfo=LA)
        self.assertEqual(T.resolve(T.TimeRef("month_day", 9, 0, day=5, month=1), now).date(), date(2027, 1, 5))

    def test_every_generated_phrase_resolves_to_its_slot(self):
        rng = random.Random(1234)
        for _ in range(3000):
            now = T.random_now(rng)
            slot = T.random_slot(rng, now)
            for lang in ("en", "es"):
                _, ref = T.caller_phrase(slot, now, rng, lang)
                got = T.resolve(ref, now)
                self.assertEqual(got, slot)
                self.assertTrue(T.offset_is_correct(T.iso(got)))
                self.assertGreater(got, now)
                self.assertIn(got.minute, (0, 30))


class TestProductionFormats(unittest.TestCase):
    def test_speakable_time_matches_availability_ts(self):
        dt = datetime(2026, 11, 12, 14, 0, tzinfo=LA)
        self.assertEqual(T.speakable_time(dt), "Thursday, November 12 at 2:00 PM PST")
        self.assertEqual(T.speakable_time(datetime(2026, 7, 1, 9, 30, tzinfo=LA)), "Wednesday, July 1 at 9:30 AM PDT")

    def test_fmt_when_matches_sms_ts(self):
        self.assertEqual(T.fmt_when(datetime(2026, 11, 12, 14, 0, tzinfo=LA)), "Thu, Nov 12, 2:00 PM PST")
        self.assertEqual(T.fmt_when(datetime(2026, 11, 12, 12, 30, tzinfo=LA)), "Thu, Nov 12, 12:30 PM PST")

    def test_nearby_openings_matches_booking_verify(self):
        now = datetime(2026, 11, 9, 9, 0, tzinfo=LA)
        req = datetime(2026, 11, 12, 14, 0, tzinfo=LA)
        busy = [(datetime(2026, 11, 12, 13, 0, tzinfo=LA), datetime(2026, 11, 12, 15, 0, tzinfo=LA))]
        alts = [T.speakable_time(a) for a in T.nearby_openings(req, busy, now)]
        self.assertTrue(any("3:00 PM" in a for a in alts))
        self.assertTrue(any("12:30 PM" in a for a in alts))
        self.assertFalse(any(re.search(r"\b(1:00|1:30|2:30) PM", a) for a in alts))
        late = datetime(2026, 11, 13, 16, 30, tzinfo=LA)  # Friday 4:30 PM, busy
        alts = T.nearby_openings(late, [(late, late + timedelta(minutes=30))], now)
        self.assertEqual(len(alts), 3)
        for a in alts:
            self.assertLess(a.weekday(), 5)
            self.assertTrue(9 <= a.hour < 17)


if __name__ == "__main__":
    unittest.main()
