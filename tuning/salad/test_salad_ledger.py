# Tests for salad_ledger.py's matching of Salad's daily usage to its invoice lines, with no Salad calls.
# Run: python3 -m unittest discover -s tuning/salad
import importlib.util, os, unittest

spec = importlib.util.spec_from_file_location("salad_ledger", os.path.join(os.path.dirname(os.path.abspath(__file__)), "salad_ledger.py"))
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)

PERIOD = {"inclusive_start_date": "2026-10-10T00:00:00+00:00", "exclusive_end_date": "2026-11-01T00:00:00+00:00"}


def metric(name, *days):
    return {"billable_metric": {"name": name}, "usage": [{"start": f"2026-10-{d:02d}T00:00:00+00:00", "metric_value": str(h)}
                                                         for d, h in days]}


class TestDailyCosts(unittest.TestCase):
    def test_hours_at_their_lines_price(self):
        daily, rates, missed = L.daily_costs([(PERIOD, [("RTX 3060 (12 GB) - Low", 3.0, 5.3, 15.9)])],
                                             [metric("RTX 3060 (12 GB) - Low", (11, 1.0), (12, 2.0))])
        self.assertEqual(daily["2026-10-12"]["RTX 3060 (12 GB) - Low"], (2.0, 10.6))
        self.assertEqual((rates, missed), ({"RTX 3060 (12 GB) - Low": 5.3}, []))

    def test_a_corrected_metric_takes_its_plain_line_and_its_empty_twin_none(self):
        # As on 24 Oct: the plain metric had no hours, the "(corrected)" one all of the line's.
        line = [("RTX 5070 (12 GB) - Low", 3.0, 10.8, 32.4)]
        use = [metric("RTX 5070 (12 GB) - Low", (11, 0)), metric("RTX 5070 (12 GB) - Low (corrected)", (11, 1.0), (13, 2.0))]
        daily, _, missed = L.daily_costs([(PERIOD, line)], use)
        self.assertEqual(daily["2026-10-13"]["RTX 5070 (12 GB) - Low"], (2.0, 21.6))
        self.assertEqual(missed, [])

    def test_an_exact_name_wins_over_a_corrected_one(self):
        lines = [("RTX 5060 (8 GB) - Lowest (corrected)", 1.0, 6.5, 6.5), ("RTX 5060 (8 GB) - Lowest", 2.0, 7.0, 14.0)]
        use = [metric("RTX 5060 (8 GB) - Lowest (corrected)", (11, 1.0)), metric("RTX 5060 (8 GB) - Lowest", (12, 2.0))]
        daily, _, _ = L.daily_costs([(PERIOD, lines)], use)
        self.assertEqual(daily["2026-10-11"], {"RTX 5060 (8 GB) - Lowest (corrected)": (1.0, 6.5)})
        self.assertEqual(daily["2026-10-12"], {"RTX 5060 (8 GB) - Lowest": (2.0, 14.0)})

    def test_a_metric_billed_under_another_name_takes_the_line_its_hours_match(self):
        # "RTX 5060 (16 GB) - Low" usage was billed as "RTX 5060 Ti (16 GB) - Low" on 24 Oct.
        lines = [("RTX 5060 Ti (16 GB) - Low", 3.0, 9.7, 29.1), ("RTX 2080 (8 GB) - Low", 5.0, 6.0, 30.0)]
        use = [metric("RTX 5060 (16 GB) - Low", (11, 1.0), (12, 2.0)), metric("RTX 2080 (8 GB) - Low", (11, 5.0))]
        daily, _, missed = L.daily_costs([(PERIOD, lines)], use)
        self.assertEqual(daily["2026-10-12"]["RTX 5060 Ti (16 GB) - Low"], (2.0, 19.4))
        self.assertEqual(missed, [])

    def test_hours_with_no_line_are_reported_not_priced(self):
        daily, _, missed = L.daily_costs([(PERIOD, [("RTX 2080 (8 GB) - Low", 5.0, 6.0, 30.0)])],
                                         [metric("RTX 9999 - Low", (11, 1.5))])
        self.assertEqual((daily, missed), ({}, [("RTX 9999 - Low", "2026-10-11", 1.5)]))

    def test_days_outside_the_period_are_left_to_their_own_period(self):
        daily, _, missed = L.daily_costs([(PERIOD, [("RTX 2080 (8 GB) - Low", 1.0, 6.0, 6.0)])],
                                         [{"billable_metric": {"name": "RTX 2080 (8 GB) - Low"},
                                           "usage": [{"start": "2026-11-01T00:00:00+00:00", "metric_value": "4"},
                                                     {"start": "2026-10-31T00:00:00+00:00", "metric_value": "1"}]}])
        self.assertEqual(list(daily), ["2026-10-31"])
        self.assertEqual(missed, [])


if __name__ == "__main__":
    unittest.main()
