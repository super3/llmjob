import importlib.util, os, unittest

_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dash", "spot_merge.py")
_s = importlib.util.spec_from_file_location("spot_merge", _p)
sm = importlib.util.module_from_spec(_s)
_s.loader.exec_module(sm)

MD = """## Fleet spot checks: SRBMiner 3.8.9 (17 November)

Bench mode in the Salad watcher races it inside production boxes: v0.7.3 against PeakMiner 2.20.9.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner 3.8.9 | OneZeroMiner | % of best |
|---|---|---|---|---|---|---|---|
| RTX 3060 (12 GB) | Salad | aaa | 52.6 | 53.6 | 53.4 | 51.8 | 98.2% |
| RTX 4080 SUPER (16 GB; Salad lists it as RTX 4080) | Salad | bbb | 219.1 (v0.7.5) | 223.0 | 223.6 | 221.7 | 98.0% |
| RTX 5060 (8 GB) | Salad | ccc | 82.8 (35a6a708) | 83.6 | 84.0 | 82.6 | 98.6% |

## Next
"""


def row(card, machine, pct):
    return {"key": "", "series": "RTX 30" if "30" in card else "RTX 40", "count": "", "card": card, "host": "Salad (170 W)",
            "machine": machine, "ours": "1", "peak": "1", "srb": "1", "ozm": "1", "pct": pct}


class SpotMerge(unittest.TestCase):
    def test_a_spot_check_replaces_the_older_row_and_says_its_versions(self):
        rows = [row("RTX 3060 · 12 GB", "o1", "99.5%"), row("RTX 3060 · 12 GB", "o2", "98.0%"),
                row("RTX 4080 Super", "o3", "98.5%"), row("RTX 4080 Super", "o4", "98.6%"),
                row("RTX 4070", "o5", "99.0%")]
        out = sm.merge(rows, MD)
        self.assertEqual([r["machine"] for r in out], ["aaa", "o1", "bbb", "o3", "o5"])
        self.assertEqual(out[0]["ver"]["srb"], "3.8.9")
        self.assertEqual(out[0]["ver"]["ours"], "0.7.3")
        self.assertEqual(out[2]["ver"]["ours"], "0.7.5")
        self.assertEqual(out[2]["ours"], "219.1")
        self.assertEqual(out[1]["ver"]["srb"], "3.8.8")
        self.assertTrue(out[0]["spot"])
        self.assertEqual(out[0]["count"], "0 of 2 above 100%, median 98.8%")

    def test_a_build_commit_tag_is_read_as_its_version(self):
        s = sm.spot_rows(MD)[-1]
        self.assertEqual((s["machine"], s["ours"], s["ver"]["ours"]), ("ccc", "82.8", "35a6a708"))


if __name__ == "__main__":
    unittest.main()
