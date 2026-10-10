# Tests for watcher.py's rules, with Salad, the pool and the clock faked.
# Run: python3 -m unittest discover -s tuning/salad
import importlib.util, os, re, tempfile, unittest

TMP = tempfile.mkdtemp()
os.environ["SALAD_MINE_DIR"] = TMP
os.environ["PRL_WALLET"] = "prl1" + "q" * 59
spec = importlib.util.spec_from_file_location("watcher", os.path.join(os.path.dirname(os.path.abspath(__file__)), "watcher.py"))
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)

T0 = 1_800_000_000.0
USD_TH_HR = 0.00117
PRICE = 1.39
CLASSES = {
    "RTX 4080 (16 GB)": {"id": "c4080", "prices": {"batch": 0.11, "low": 0.15, "medium": 0.19, "high": 0.23}},
    "RTX 2060 (6 GB)": {"id": "c2060", "prices": {"batch": 0.02, "low": 0.03, "medium": 0.04, "high": 0.05}},
    "RTX 4070 (12 GB)": {"id": "c4070", "prices": {"batch": 0.07, "low": 0.09, "medium": 0.123, "high": 0.15}},
}


class Fake:
    """Answers HTTP calls by (method, URL regex); the last route added wins."""
    def __init__(self):
        self.calls, self.routes = [], []

    def on(self, method, frag, resp):
        self.routes.insert(0, (method, frag, resp))

    def __call__(self, method, url, body=None, ctype="application/json", timeout=60):
        self.calls.append((method, url, body))
        for m, frag, resp in self.routes:
            if m == method and re.search(frag, url):
                return resp(body) if callable(resp) else resp
        return 200, {}

    def made(self, method, frag):
        return [c for c in self.calls if c[0] == method and re.search(frag, c[1])]


class Base(unittest.TestCase):
    def setUp(self):
        for f in os.listdir(TMP):
            os.remove(os.path.join(TMP, f))
        self.clock = [T0]
        w.now = lambda: self.clock[0]
        self.http = Fake()
        w.http = self.http
        self.x = w.W()
        self.x.cfg = dict(w.DEFAULT_CFG)
        st = self.x.st
        st["classes"], st["classes_t"] = CLASSES, T0 + 10 ** 6
        st["mkt"], st["mkt_t"] = {"price": PRICE, "prl_th_hr": USD_TH_HR / PRICE, "usd_th_hr": USD_TH_HR}, T0 + 10 ** 6
        self.http.on("POST", "/containers$", (201, {}))

    def group(self, cls="RTX 4080 (16 GB)", pri="batch", status="running"):
        self.assertTrue(self.x.create(cls, pri))
        g = list(self.x.st["groups"].values())[-1]
        g["status"] = status
        return g

    def step(self, g, minutes, machine=None):
        """Advance the clock and run the rules with the group on `machine` (None: no PC)."""
        self.clock[0] += minutes * 60
        g["has_pc"], g["_machine"] = machine is not None, machine
        if machine:
            g["instance"] = "inst-" + machine
        self.x.rules(g, self.clock[0])

    def mine(self, g, machine, ths, minutes, shares=True):
        """Mine on `machine` at `ths`: one reading (and a share, unless shares=False) and one round a minute."""
        for _ in range(minutes):
            self.clock[0] += 60
            g["readings"].append([self.clock[0], machine, float(ths), 0])
            if shares:
                g["last_share"] = self.clock[0]
            g["has_pc"], g["_machine"], g["instance"] = True, machine, "inst-" + machine
            self.x.rules(g, self.clock[0])
            if g["ended"] or g["machine"] is None:
                return

    def events(self):
        try:
            with open(os.path.join(TMP, "events.log")) as f:
                return f.read()
        except FileNotFoundError:
            return ""


class TestCommand(Base):
    def test_command_sets_worker_wallet_and_version(self):
        cmd = w.cmd_for("biz-rtx4080", "prl1abc", "v0.5.13")
        self.assertIn("s/^WORKER=__WORKER__$/WORKER=biz-rtx4080/", cmd)
        self.assertIn("s/^W=prl1.*$/W=prl1abc/", cmd)
        self.assertNotIn("; curl", cmd)  # Salad refuses "; curl"
        self.assertNotIn("VER=", cmd)
        self.assertIn("s/^VER=v0.5.13;/VER=v0.5.14;/", w.cmd_for("biz-rtx4080", "prl1abc", "v0.5.14"))

    def test_slug(self):
        self.assertEqual(w.slug("RTX 4080 (16 GB)"), "rtx4080")
        self.assertEqual(w.slug("RTX 4070 Ti (12 GB)"), "rtx4070ti")

    def test_create_body(self):
        self.group()
        body = self.http.made("POST", "/containers$")[0][2]
        self.assertEqual(body["name"], "biz-rtx4080")
        self.assertEqual(body["container"]["image"], "nvidia/cuda:12.2.2-base-ubuntu22.04")
        self.assertEqual(body["container"]["priority"], "batch")
        self.assertEqual(body["container"]["resources"]["gpu_classes"], ["c4080"])
        self.assertEqual((body["restart_policy"], body["autostart_policy"], body["replicas"]), ("always", True, 1))

    def test_second_group_of_a_class_gets_its_own_name(self):
        self.group()
        self.group()
        self.assertEqual(self.http.made("POST", "/containers$")[1][2]["name"], "biz-rtx4080-2")


class TestMargin(Base):
    def test_margin_is_profit_over_earnings(self):
        earn = 199 * USD_TH_HR
        self.assertAlmostEqual(self.x.margin("RTX 4080 (16 GB)", "medium"), (earn - 0.19) / earn)


class TestRule2NoPc(Base):
    def test_moves_up_after_10_min(self):
        g = self.group(status="deploying")
        self.step(g, 9)
        self.assertFalse(self.http.made("PATCH", ""))
        self.step(g, 2)
        patch = self.http.made("PATCH", "/containers/biz-rtx4080")
        self.assertEqual(patch[0][2], {"container": {"priority": "low"}})
        self.assertEqual((g["pri"], g["price"]), ("low", 0.15))

    def test_waits_while_pending(self):
        g = self.group(status="pending")
        self.step(g, 15)
        self.assertFalse(self.http.made("PATCH", ""))

    def test_the_10_minutes_start_when_the_image_is_ready(self):
        g = self.group(status="pending")
        self.step(g, 15)
        g["status"] = "deploying"
        self.step(g, 9)
        self.assertFalse(self.http.made("PATCH", ""))
        self.step(g, 2)
        self.assertTrue(self.http.made("PATCH", ""))

    def test_deletes_when_no_higher_priority_clears_10_percent(self):
        g = self.group(pri="medium", status="deploying")  # high is +2% for a 4080
        self.step(g, 11)
        self.assertTrue(self.http.made("DELETE", "/containers/biz-rtx4080"))
        self.assertEqual(g["end_kind"], "released")


class TestRule3LostPc(Base):
    def test_deletes_20_min_after_losing_its_pc(self):
        g = self.group()
        self.step(g, 1, "m1")
        self.step(g, 5, None)
        self.step(g, 19, None)
        self.assertFalse(self.http.made("DELETE", ""))
        self.step(g, 2, "m2")  # got another PC in time
        self.assertIsNone(g["ended"])
        self.step(g, 1, None)
        self.step(g, 21, None)
        self.assertEqual(g["end_kind"], "released")


class TestRule4NoShare(Base):
    def test_paid_class_is_reallocated(self):
        g = self.group()
        self.step(g, 1, "m1")
        self.mine(g, "m1", 199, 19, shares=False)
        self.assertFalse(self.http.made("POST", "/reallocate"))
        self.mine(g, "m1", 199, 2, shares=False)
        self.assertTrue(self.http.made("POST", "/instances/inst-m1/reallocate"))
        self.assertEqual(g["reallocs"], 1)

    def test_shares_keep_it(self):
        g = self.group()
        self.step(g, 1, "m1")
        self.mine(g, "m1", 199, 30)
        self.assertFalse(self.http.made("POST", "/reallocate"))

    def test_class_that_never_paid_is_deleted(self):
        g = self.group(cls="RTX 4070 (12 GB)", pri="low")
        self.step(g, 1, "m1")
        self.mine(g, "m1", 120, 21, shares=False)
        self.assertEqual(g["end_kind"], "failed")
        self.assertNotIn("RTX 4070 (12 GB)", self.x.st.get("paid_classes", []))  # a good rate with no shares isn't paying

    def test_slow_card_waits_8_share_gaps(self):
        g = self.group(cls="RTX 2060 (6 GB)", pri="low")  # 40 TH/s: a share every ~4 min, 8 gaps ~31 min
        self.step(g, 1, "m1")
        self.mine(g, "m1", 40, 25, shares=False)
        self.assertFalse(self.http.made("POST", "/reallocate"))
        self.mine(g, "m1", 40, 8, shares=False)
        self.assertTrue(self.http.made("POST", "/reallocate"))


class TestNotMining(Base):
    def test_a_pc_with_no_reading_after_10_min_is_left(self):
        g = self.group()
        self.step(g, 1, "m1")
        self.step(g, 9, "m1")
        self.assertFalse(self.http.made("POST", "/reallocate"))
        self.step(g, 1, "m1")
        self.assertTrue(self.http.made("POST", "/instances/inst-m1/reallocate"))


class TestRule5Profit(Base):
    def test_an_unprofitable_pc_is_left_at_once(self):
        g = self.group(pri="medium")  # $0.19/hr; 100 TH/s earns $0.117
        self.step(g, 1, "m1")
        self.mine(g, "m1", 100, 4)  # 2 min of warm-up, then 2 readings: not enough yet
        self.assertFalse(self.http.made("POST", "/reallocate"))
        self.mine(g, "m1", 100, 1)  # the 3rd reading after warm-up
        self.assertTrue(self.http.made("POST", "/reallocate"))
        self.assertIn("against its $0.190/hr (-62%, under 10%)", self.events())

    def test_a_pc_that_pays_under_10_percent_is_left(self):
        g = self.group(pri="medium")  # $0.19/hr; 170 TH/s earns $0.199: +4.5%
        self.step(g, 1, "m1")
        self.mine(g, "m1", 170, 5)
        self.assertTrue(self.http.made("POST", "/reallocate"))

    def test_the_bar_comes_from_config(self):
        self.x.cfg["min_margin"] = 0.0
        g = self.group(pri="medium")
        self.step(g, 1, "m1")
        self.mine(g, "m1", 170, 10)
        self.assertFalse(self.http.made("POST", "/reallocate"))

    def test_warm_up_is_ignored(self):
        g = self.group(pri="medium")
        self.step(g, 1, "m1")
        self.mine(g, "m1", 50, 2)
        self.mine(g, "m1", 199, 30)
        self.assertFalse(self.http.made("POST", "/reallocate"))

    def test_it_keeps_checking_while_it_mines(self):
        g = self.group(pri="medium")
        self.step(g, 1, "m1")
        self.mine(g, "m1", 199, 30)
        self.mine(g, "m1", 120, 3)  # the owner starts using the card
        self.assertTrue(self.http.made("POST", "/reallocate"))

    def test_one_slow_reading_is_averaged_out(self):
        g = self.group(pri="medium")
        self.step(g, 1, "m1")
        self.mine(g, "m1", 199, 10)
        self.mine(g, "m1", 160, 1)  # the 3 readings average 186 TH/s: +13%
        self.mine(g, "m1", 199, 5)
        self.assertFalse(self.http.made("POST", "/reallocate"))

    def test_class_that_never_paid_is_deleted(self):
        g = self.group(cls="RTX 4070 (12 GB)", pri="low")  # $0.09/hr; 50 TH/s earns $0.059
        self.step(g, 1, "m1")
        self.mine(g, "m1", 50, 5)
        self.assertEqual(g["end_kind"], "failed")


class TestPaidClasses(Base):
    def test_a_thin_class_that_pays_here_counts_as_paid(self):
        g = self.group(cls="RTX 4070 (12 GB)", pri="low")  # never paid on the first account
        self.step(g, 1, "m1")
        self.mine(g, "m1", 120, 12)  # $0.140/hr > $0.09/hr, with shares
        self.assertIn("RTX 4070 (12 GB)", self.x.st["paid_classes"])
        self.assertIn("PAID biz-rtx4070", self.events())
        self.mine(g, "m1", 60, 3)  # then this PC slows down: the PC is the problem now, not the class
        self.assertTrue(self.http.made("POST", "/reallocate"))
        self.assertIsNone(g["ended"])


class TestDeleteReason(Base):
    def test_a_deletion_carries_the_boxes_last_lines(self):
        g = self.group(cls="RTX 4070 (12 GB)", pri="low")
        self.step(g, 1, "m1")
        lines = ["[run] g0 miner exited 08:30:00 (restart 1): out of memory", "[mining g0] 08:29:00 0.0 TH/s"]
        self.http.on("POST", "/log-entries", lambda b: (200, {"items": [{"text_log": l} for l in lines]}))
        self.mine(g, "m1", 50, 5)
        ev = self.events()
        self.assertIn("DELETED biz-rtx4070", ev)
        self.assertIn("WHY biz-rtx4070: [mining g0] 08:29:00 0.0 TH/s\n    WHY biz-rtx4070: [run] g0 miner exited", ev)
        body = self.http.made("POST", "/log-entries")[0][2]
        self.assertIn('container_group_name = "biz-rtx4070"', body["query"])


class TestRule6PortBlocked(Base):
    def test_reallocates_then_deletes_after_3(self):
        g = self.group()
        self.step(g, 1, "m1")
        g["blocked"].append("m1")
        self.step(g, 1, "m1")
        self.assertEqual(g["reallocs"], 1)
        g["reallocs"] = 3
        self.step(g, 1, "m2")
        g["blocked"].append("m2")
        self.step(g, 1, "m2")
        self.assertEqual(g["end_kind"], "failed")

    def test_old_pc_still_showing_after_a_reallocation_is_not_a_new_pc(self):
        g = self.group()
        self.step(g, 1, "m1")
        g["blocked"].append("m1")
        self.step(g, 1, "m1")
        self.step(g, 1, "m1")  # Salad still lists the old PC
        self.assertEqual(g["reallocs"], 1)
        self.assertIsNone(g["machine"])
        self.step(g, 1, "m2")
        self.assertEqual(g["machine"], "m2")
        self.assertIn("NEW PC biz-rtx4080", self.events())


class TestRule7Credit(Base):
    def test_stopped_after_mining_means_credit_ran_out(self):
        g = self.group()
        self.step(g, 1, "m1")
        g["status"] = "stopped"
        n = len(self.http.calls)
        self.step(g, 1, None)
        self.assertTrue(self.x.st["credits_out"])
        self.assertEqual(len(self.http.calls), n)  # left alone
        self.assertIsNone(g["ended"])

    def test_stopped_before_any_pc_is_started(self):
        g = self.group(status="stopped")
        self.step(g, 1, None)
        self.assertTrue(self.http.made("POST", "/containers/biz-rtx4080/start"))
        self.assertFalse(self.x.st["credits_out"])

    def test_no_credits_on_create(self):
        self.http.on("POST", "/containers$", (400, {"errors": {"code": ["no_credits_available"]}}))
        self.assertFalse(self.x.create("RTX 4080 (16 GB)", "batch"))
        self.assertTrue(self.x.st["credits_out"])


class TestSearch(Base):
    def test_picks_the_paid_class_at_its_cheapest_free_priority(self):
        avail = {"c4080": {"available_gpu_batch": 0, "available_gpu_low": 5},
                 "c2060": {"available_gpu_low": 3},
                 "c4070": {"available_gpu_batch": 9}}
        self.http.on("POST", "/availability/", lambda b: (200, avail.get(b["gpu_classes"][0], {})))
        self.http.on("GET", "/quotas", (200, {"container_groups_quotas": {"container_replicas_quota": 10}}))
        self.x.cfg["max_active"] = 2
        self.x.search()
        made = [(c[2]["name"], c[2]["container"]["priority"]) for c in self.http.made("POST", "/containers$")]
        self.assertEqual(made, [("biz-rtx4080", "low"), ("biz-rtx2060", "low")])

    def test_a_class_is_tried_once(self):
        self.http.on("POST", "/availability/", (200, {"available_gpu_low": 5}))
        self.http.on("GET", "/quotas", (200, {"container_groups_quotas": {"container_replicas_quota": 10}}))
        g = self.group(cls="RTX 4080 (16 GB)", pri="low")
        g["ended"], g["end_kind"] = self.clock[0], "released"
        self.x.cfg["max_active"] = 10
        self.x.search()
        names = [c[2]["name"] for c in self.http.made("POST", "/containers$")]
        self.assertEqual(names.count("biz-rtx4080"), 1)  # only the first one
        self.assertEqual(names[1:], ["biz-rtx2060", "biz-rtx4070"])  # the classes not tried yet

    def test_respects_max_active(self):
        self.http.on("POST", "/availability/", (200, {"available_gpu_low": 5}))
        self.group(cls="RTX 2060 (6 GB)", pri="low")
        n = len(self.http.made("POST", "/containers$"))
        self.x.search()
        self.assertEqual(len(self.http.made("POST", "/containers$")), n)


class TestRound(Base):
    def test_logs_and_pool_feed_the_ledger(self):
        self.x.cfg["search"] = False
        self.x.cfg["wallet"] = os.environ["PRL_WALLET"]
        g = self.group()
        self.http.on("GET", "/containers/biz-rtx4080", (200, {"current_state": {"status": "running"}}))
        self.http.on("GET", "/instances", (200, {"instances": [{"id": "i1", "machine_id": "m1", "state": "running"}]}))
        logs = [("2027-01-15T08:05:00.1234Z", "[card] 0, NVIDIA GeForce RTX 4080, 570.1, 320.00 W, 3105 MHz, 8.9"),
                ("2027-01-15T08:06:00Z", "[mining g0] 08:06:00 198.0 TH/s · 0 accepted · 0 rejected · up 0h 1m"),
                ("2027-01-15T08:07:00Z", "[mining g0] 08:07:00 200.0 TH/s · 2 accepted · 0 rejected · up 0h 2m")]
        items = [{"text_log": txt, "time": tm, "resource": {"labels": {"container_group_name": "biz-rtx4080", "machine_id": "m1"}}}
                 for tm, txt in logs]
        self.http.on("POST", "/log-entries", (200, {"items": items}))
        self.http.on("POST", "/log-entries", lambda b: (200, {"items": items}) if b["page_size"] <= 100 else (400, {}))
        self.http.on("GET", "herominers", (200, {"workers": [{"name": "biz-rtx4080-g0", "hashrate": 44500,
                                                               "lastShare": int(T0) + 400}]}))
        g["created"] = T0 - 3600
        self.clock[0] = T0 + 500
        self.x.round()
        self.x.round()  # the same log lines again are not counted twice
        self.assertAlmostEqual(g["prl"], 398 / 60 * USD_TH_HR / PRICE)
        self.assertEqual(g["mined_min"], 2)
        self.assertEqual(g["last_share"], T0 + 420)  # the log's share at 08:07 is later than the pool's
        self.assertEqual(g["machine"], "m1")
        self.assertAlmostEqual(g["pool_ths"], 44500 * 419 / 93644)
        self.assertEqual(len(self.http.made("POST", "/log-entries")), 2)
        ev = self.events()
        self.assertIn("CARD biz-rtx4080", ev)
        self.assertIn("FIRST SHARE biz-rtx4080", ev)
        with open(os.path.join(TMP, "report.md")) as f:
            report = f.read()
        self.assertIn("⚡ RTX 4080 ×1 (Salad, batch)", report)
        self.assertIn("1 GPU mining", report)

    def test_a_busy_round_reads_more_pages(self):
        g = self.group()
        g["created"] = T0 - 3600
        lines = [{"text_log": f"[mining g0] 07:{i // 60:02d}:{i % 60:02d} 199.0 TH/s · {i} accepted · 0 rejected · up 1h",
                  "time": w.iso(T0 - 600 + i), "resource": {"labels": {"container_group_name": "biz-rtx4080", "machine_id": "m1"}}}
                 for i in range(150)]
        def page(b):
            start = w.parse_iso(b["start_time"])
            return 200, {"items": [x for x in lines if w.parse_iso(x["time"]) >= start][:b["page_size"]]}
        self.http.on("POST", "/log-entries", page)
        got = self.x.logs([g])
        self.assertEqual(len(got), 150)  # the line on the page boundary is not counted twice
        self.assertEqual(len(self.http.made("POST", "/log-entries")), 2)
        self.assertEqual(self.x.st["last_log_end"], self.clock[0])

    def test_billing_counts_only_minutes_on_a_pc(self):
        self.x.cfg["search"] = False
        g = self.group()
        self.http.on("GET", "/containers/biz-rtx4080", (200, {"current_state": {"status": "running"}}))
        self.http.on("GET", "/instances", (200, {"instances": [{"id": "i1", "state": "allocating"}]}))
        self.x.round()
        self.clock[0] += 300
        self.x.round()
        self.assertEqual(g["spend"], 0)
        self.http.on("GET", "/instances", (200, {"instances": [{"id": "i1", "machine_id": "m1", "state": "running"}]}))
        self.clock[0] += 600
        self.x.round()
        self.assertAlmostEqual(g["spend"], 0.11 * 600 / 3600)


if __name__ == "__main__":
    unittest.main()
