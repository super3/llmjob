# Tests for bench.py's parsing and table, with no Salad calls.
# Run: python3 -m unittest discover -s tuning/salad
import importlib.util, os, tempfile, unittest

os.environ.setdefault("SALAD_MINE_DIR", tempfile.mkdtemp())
spec = importlib.util.spec_from_file_location("bench", os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench.py"))
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)

LINES = [
    "[bench] 19:40:01 card NVIDIA GeForce GTX 1080 Ti, 550.54, 250.00 W, 250.00 W, 11264 MiB, 6.1",
    "[bench] 19:42:00 ready PeakMiner 2.17.6, SRBMiner 3.7.1, ours v0.6.0",
    "[bench] 19:43:15 start peak 300s",
    "[bench] 19:43:30 sample peak t=15 hs=40000000000000 acc=0 rej=0 gpu=240,1800,70",
    "[bench] 19:48:15 sample peak t=300 hs=42500000000000 acc=3 rej=0 gpu=248,1810,74",
    "[bench] 19:48:30 sample srb t=15 hs=? acc=? rej=? gpu=0,139,40",
    "[bench] 19:48:45 exited srb t=30: unsupported GPU",
    "[bench] 19:49:00 exited ours t=15: no code for compute 6.1",
    "[bench] 19:49:10 all done",
]


class TestBench(unittest.TestCase):
    def test_row_for_prefers_the_longest_name(self):
        rows = ["GTX 1080 Ti", "GTX 1080", "GTX 1070"]
        self.assertEqual(b.row_for("NVIDIA GeForce GTX 1080 Ti", rows), "GTX 1080 Ti")
        self.assertEqual(b.row_for("NVIDIA GeForce GTX 1080", rows), "GTX 1080")
        self.assertEqual(b.row_for("NVIDIA GeForce GTX 1660 SUPER", ["GTX 1660 Super", "GTX 1660"]), "GTX 1660 Super")
        self.assertEqual(b.row_for("NVIDIA GeForce GTX 1660", ["GTX 1660 Super", "GTX 1660"]), "GTX 1660")
        self.assertIsNone(b.row_for("NVIDIA GeForce GTX 1050 Ti", rows))
        self.assertIsNone(b.row_for("NVIDIA GeForce GTX 1660 Ti", ["GTX 1660"]))   # a different card
        self.assertEqual(b.row_for("NVIDIA GeForce GTX 1660 Ti", ["GTX 1660", "GTX 1660 Ti"]), "GTX 1660 Ti")
        self.assertIsNone(b.row_for("NVIDIA GeForce GTX 1080 Ti", ["GTX 1080"]))

    def test_parse_reads_card_samples_exits_and_done(self):
        r = b.parse(LINES)
        self.assertEqual(r["card"]["name"], "NVIDIA GeForce GTX 1080 Ti")
        self.assertEqual(r["card"]["power"], "250.00 W")
        self.assertTrue(r["done"])
        self.assertEqual(b.figure(r["miners"]["peak"]), (42.5, None))       # its own figure at the end
        self.assertEqual(b.figure(r["miners"]["srb"]), (None, "no rate"))
        self.assertEqual(b.figure(r["miners"].get("ours"), mean_after=60), (None, "didn't start"))

    def test_a_rate_only_in_the_first_seconds_is_no_rate(self):
        # SRBMiner on a GTX 1660 whose PC lost the pool: 1.15 TH/s at t=15, then 0 for the rest of the run.
        m = {"samples": [{"t": 15, "th": 1.15}] + [{"t": t, "th": 0.0} for t in range(60, 301, 15)]}
        self.assertEqual(b.figure(m), (None, "no rate"))
        m = {"samples": [{"t": t, "th": 1.1 if t < 200 else 0.0} for t in range(15, 301, 15)]}
        self.assertEqual(b.figure(m), (1.1, None))       # most of the run counts

    def test_ours_is_the_mean_after_the_first_minute(self):
        m = {"samples": [{"t": 15, "th": 10.0}, {"t": 60, "th": 20.0}, {"t": 75, "th": 22.0}]}
        self.assertEqual(b.figure(m, mean_after=60), (21.0, None))

    def test_render_fills_rows_in_order_and_leaves_the_rest_empty(self):
        st = {"results": {"GTX 1080 Ti": [{"machine": "9318a331-xx", "power": "250.00 W", "ours": (None, "no build"),
                                            "peak": (42.5, None), "srb": (40.1, None)}]}}
        text = b.render(st)
        self.assertIn("| GTX 1080 Ti | Salad (250 W) | 9318a331 | no build | 42.5 | 40.1 | — |", text)
        self.assertIn("| GTX 1080 Ti | Salad (— W) | — | not run | not run | not run | — |", text)
        self.assertIn("| GTX 1650 | Salad (— W, 4 GB) | — | not run | not run | not run | — |", text)
        st["results"]["GTX 1660"] = [{"machine": "abcd1234", "power": "120.00 W", "ours": (30.0, None),
                                      "peak": (40.0, None), "srb": (38.0, None)}]
        self.assertIn("| GTX 1660 | Salad (120 W, 6 GB) | abcd1234 | 30.0 | 40.0 | 38.0 | 75% |", b.render(st))
        self.assertNotIn("GTX 1660 Ti", b.render(st))     # a card no class is for shows only when it turned up
        st["results"]["GTX 1660 Ti"] = [{"machine": "741f9c56", "power": "120.00 W", "ours": (1.2, None),
                                         "peak": (1.2, None), "srb": (1.2, None)}]
        self.assertEqual(b.render(st).count("| GTX 1660 Ti |"), 1)
        st["results"]["GTX 1660 Ti"][0]["power"] = "[N/A]"
        self.assertIn("| GTX 1660 Ti | Salad (— W, 6 GB) | 741f9c56 |", b.render(st))

    def test_a_group_stopped_before_a_pc_is_started_again_every_5_minutes(self):
        calls, now = [], [10_000]
        def http(method, url, body=None, timeout=None):
            calls.append((method, url))
            if url.endswith("/instances"):
                return 200, {"instances": []}
            if method == "GET":
                return 200, {"current_state": {"status": "stopped"}}
            return 202, None
        old = (b.w.http, b.w.now)
        b.w.http, b.w.now = http, lambda: now[0]
        try:
            bench = b.Bench()
            bench.st.update(cls=0, group="bench-x-1", created=now[0], pc_t=None, machine=None)
            bench.tick()
            self.assertIn(("POST", f"{b.w.CB}/bench-x-1/start"), calls)
            calls.clear(); now[0] += 60
            bench.tick()
            self.assertNotIn(("POST", f"{b.w.CB}/bench-x-1/start"), calls)
            calls.clear(); now[0] += 5 * 60
            bench.tick()
            self.assertIn(("POST", f"{b.w.CB}/bench-x-1/start"), calls)
        finally:
            b.w.http, b.w.now = old

    def test_a_pc_where_no_miner_gets_work_is_not_recorded(self):
        calls, now = [], [20_000]
        def http(method, url, body=None, timeout=None):
            calls.append((method, url))
            if url.endswith("/instances"):
                return 200, {"instances": [{"machine_id": "e3babbec-xx", "id": "i1", "state": "running"}]}
            if url.endswith("/log-entries"):
                return 200, {"items": []}
            return 202, None
        lines = ["[bench] 21:27:00 card NVIDIA GeForce GTX 1660, 617.42, 130.00 W, 130.00 W, 6144 MiB, 7.5"]
        for name in ("peak", "srb", "ours"):
            lines += [f"[bench] 21:30:00 sample {name} t={t} hs=0 acc=0 rej=0 gpu=22,1530,46" for t in range(15, 301, 15)]
        lines.append("[bench] 21:43:41 all done")
        old = (b.w.http, b.w.now)
        b.w.http, b.w.now = http, lambda: now[0]
        try:
            bench = b.Bench()
            cls = next(i for i, p in enumerate(b.PLAN) if p[0].startswith("GTX 1660 ("))
            bench.st.update(cls=cls, tries=0, group="bench-gtxgtx1660-1", created=now[0] - 900, machine="e3babbec-xx",
                            instance="i1", pc_t=now[0] - 900, log_t=now[0] - 60, lines=[f"t|{x}" for x in lines],
                            results={}, notes=[])
            bench.tick()
            self.assertNotIn("GTX 1660", bench.st["results"])
            self.assertIn("e3babbec-xx", bench.st["seen"])
            self.assertIn("no miner got work", bench.st["notes"][-1])
            self.assertIn(("POST", f"{b.w.CB}/bench-gtxgtx1660-1/instances/i1/reallocate"), calls)
        finally:
            b.w.http, b.w.now = old


if __name__ == "__main__":
    unittest.main()
