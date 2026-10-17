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

    def test_parse_reads_card_samples_exits_and_done(self):
        r = b.parse(LINES)
        self.assertEqual(r["card"]["name"], "NVIDIA GeForce GTX 1080 Ti")
        self.assertEqual(r["card"]["power"], "250.00 W")
        self.assertTrue(r["done"])
        self.assertEqual(b.figure(r["miners"]["peak"]), (42.5, None))       # its own figure at the end
        self.assertEqual(b.figure(r["miners"]["srb"]), (None, "no rate"))
        self.assertEqual(b.figure(r["miners"].get("ours"), mean_after=60), (None, "didn't start"))

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


if __name__ == "__main__":
    unittest.main()
