# Tests for watcher.py's rules, with Salad, the pool and the clock faked.
# Run: python3 -m unittest discover -s tuning/salad
import importlib.util, json, os, re, tempfile, unittest

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
    "RTX 3060 Ti (8 GB)": {"id": "c3060ti", "prices": {"batch": 0.035, "low": 0.047, "medium": 0.06, "high": 0.07}},
}


class Fake:
    """Answers HTTP calls by (method, URL regex); the last route added wins."""
    def __init__(self):
        self.calls, self.routes = [], []

    def on(self, method, frag, resp):
        self.routes.insert(0, (method, frag, resp))

    def __call__(self, method, url, body=None, ctype="application/json", timeout=60, headers=None):
        self.calls.append((method, url, body))
        self.headers = headers
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

    def test_a_name_salad_still_holds_is_skipped(self):
        self.http.on("POST", "/containers$", lambda b: (400, {"type": "name_conflict"}) if b["name"] == "biz-rtx4080"
                     else (201, {}))
        self.assertTrue(self.x.create("RTX 4080 (16 GB)", "low"))
        names = [c[2]["name"] for c in self.http.made("POST", "/containers$")]
        self.assertEqual(names, ["biz-rtx4080", "biz-rtx4080-2"])
        body = self.http.made("POST", "/containers$")[1][2]
        self.assertIn("WORKER=biz-rtx4080-2/", body["container"]["command"][2])

    def test_live_groups_dont_use_up_the_name_retries(self):
        # 9 live groups and Salad still holding biz-rtx4080 to biz-rtx4080-12, as on 2026-10-11
        for i in range(9):
            self.x.st["groups"][f"g{i}"] = {"name": f"biz-other-{i}", "ended": None}
        held = {"biz-rtx4080"} | {f"biz-rtx4080-{i}" for i in range(2, 13)}
        self.http.on("POST", "/containers$", lambda b: (400, {"type": "name_conflict"}) if b["name"] in held
                     else (201, {}))
        self.assertTrue(self.x.create("RTX 4080 (16 GB)", "low"))
        self.assertEqual(self.http.made("POST", "/containers$")[-1][2]["name"], "biz-rtx4080-13")

    def test_names_of_deleted_groups_are_skipped_too(self):
        # 21 deleted 4080 groups, all names still held by Salad, as on 2026-10-12
        names = ["biz-rtx4080"] + [f"biz-rtx4080-{i}" for i in range(2, 22)]
        for n in names:
            self.x.st["groups"][n] = {"name": n, "ended": T0}
        self.http.on("POST", "/containers$", lambda b: (400, {"type": "name_conflict"}) if b["name"] in names
                     else (201, {}))
        self.assertTrue(self.x.create("RTX 4080 (16 GB)", "low"))
        self.assertEqual([c[2]["name"] for c in self.http.made("POST", "/containers$")], ["biz-rtx4080-22"])

    def test_name_retries_stop_after_20(self):
        self.http.on("POST", "/containers$", (400, {"type": "name_conflict"}))
        self.assertFalse(self.x.create("RTX 4080 (16 GB)", "low"))
        self.assertEqual(len(self.http.made("POST", "/containers$")), 21)

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


class TestMinerStops(Base):
    def test_a_pc_whose_miner_stops_is_left(self):
        g = self.group(pri="medium")
        self.step(g, 1, "m1")
        self.mine(g, "m1", 199, 10)
        self.step(g, 5, "m1")  # no readings since
        self.assertFalse(self.http.made("POST", "/reallocate"))
        self.step(g, 1, "m1")
        self.assertTrue(self.http.made("POST", "/instances/inst-m1/reallocate"))
        self.assertIn("no mining reading for 6 min", self.events())


class TestRepeatedLine(Base):
    def test_a_repeated_line_from_a_dead_miner_is_not_a_reading(self):
        self.x.cfg["search"] = False
        g = self.group()
        g["created"] = T0 - 3600
        self.http.on("GET", "/containers/biz-rtx4080", (200, {"current_state": {"status": "running"}}))
        self.http.on("GET", "/instances", (200, {"instances": [{"id": "i1", "machine_id": "m1", "state": "running"}]}))
        lines = [("2027-01-15T07:5%d:00Z" % i, "[mining g0] 07:5%d:00 112.1 TH/s · 326 accepted · 0 rejected · up 1h 02m" % i)
                 for i in range(5)]  # the same line five times, uptime frozen
        self.http.on("POST", "/log-entries", (200, {"items": [
            {"text_log": txt, "time": tm, "resource": {"labels": {"container_group_name": "biz-rtx4080", "machine_id": "m1"}}}
            for tm, txt in lines]}))
        self.x.round()
        self.assertEqual(len(g["readings"]), 1)
        self.assertAlmostEqual(g["prl"], 112.1 / 60 * USD_TH_HR / PRICE)

    def test_a_steady_live_miner_still_counts(self):
        self.x.cfg["search"] = False
        g = self.group()
        g["created"] = T0 - 3600
        self.http.on("GET", "/containers/biz-rtx4080", (200, {"current_state": {"status": "running"}}))
        self.http.on("GET", "/instances", (200, {"instances": [{"id": "i1", "machine_id": "m1", "state": "running"}]}))
        lines = [("2027-01-15T07:5%d:00Z" % i, "[mining g0] 07:5%d:00 45.0 TH/s · 12 accepted · 0 rejected · up 1h 0%dm" % (i, i))
                 for i in range(5)]  # same rate and shares, but the uptime moves
        self.http.on("POST", "/log-entries", (200, {"items": [
            {"text_log": txt, "time": tm, "resource": {"labels": {"container_group_name": "biz-rtx4080", "machine_id": "m1"}}}
            for tm, txt in lines]}))
        self.x.round()
        self.assertEqual(len(g["readings"]), 5)


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
        lines = [("2027-01-15T08:30:00Z", "[run] g0 miner exited 08:30:00 (restart 1): out of memory"),
                 ("2027-01-15T08:29:00Z", "[mining g0] 08:29:00 0.0 TH/s")]  # newest first: the watcher sorts them
        self.http.on("POST", "/log-entries", lambda b: (200, {"items": [{"time": tm, "text_log": l} for tm, l in lines]}))
        self.mine(g, "m1", 50, 5)
        ev = self.events()
        self.assertIn("DELETED biz-rtx4070", ev)
        self.assertIn("WHY biz-rtx4070: [mining g0] 08:29:00 0.0 TH/s\n    WHY biz-rtx4070: [run] g0 miner exited", ev)
        body = self.http.made("POST", "/log-entries")[0][2]
        self.assertIn('container_group_name = "biz-rtx4070"', body["query"])


class TestSweep(Base):
    def listing(self, *names):
        self.http.on("GET", "/containers$", (200, {"items": [{"name": n} for n in names]}))

    def sweep_at(self, minutes):
        self.clock[0] = T0 + minutes * w.MIN
        self.x.sweep(self.clock[0])

    def test_a_group_the_ledger_let_go_is_deleted_after_10_min(self):
        g = self.group()
        g["ended"] = T0  # the watcher deleted it, Salad answered 202 and kept it
        self.listing(g["name"])
        self.sweep_at(0)
        self.sweep_at(5)
        self.assertEqual(self.http.made("DELETE", "/containers/"), [])
        self.sweep_at(10)
        self.assertEqual(len(self.http.made("DELETE", f"/containers/{g['name']}$")), 1)
        self.assertIn(f"SWEPT {g['name']}: on Salad for 10 min", self.events())

    def test_live_groups_and_other_names_are_left_alone(self):
        g = self.group()
        self.listing(g["name"], "someone-elses-group")
        for m in (0, 10, 20):
            self.sweep_at(m)
        self.assertEqual(self.http.made("DELETE", "/containers/"), [])

    def test_a_stray_that_goes_away_by_itself_is_not_deleted(self):
        self.listing("biz-rtx3070-8")
        self.sweep_at(0)
        self.listing()
        self.sweep_at(5)
        self.listing("biz-rtx3070-8")  # back again: the 10 minutes start over
        self.sweep_at(10)
        self.assertEqual(self.http.made("DELETE", "/containers/"), [])

    def test_runs_at_most_every_5_min(self):
        self.listing()
        self.sweep_at(0)
        self.sweep_at(2)
        self.assertEqual(len(self.http.made("GET", "/containers$")), 1)

    def test_says_it_once_and_keeps_trying(self):
        self.listing("biz-rtx3070-8")
        self.http.on("DELETE", "/containers/", (202, {}))
        for m in (0, 10, 15, 20):
            self.sweep_at(m)
        self.assertEqual(len(self.http.made("DELETE", "/containers/biz-rtx3070-8$")), 3)
        self.assertEqual(self.events().count("SWEPT biz-rtx3070-8"), 1)



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
        # Out of moves means bad PCs, not a bad class: released, so the class can still get new groups.
        self.assertEqual(g["ended"] is not None, True)
        self.assertEqual(g["end_kind"], "released")
        self.assertIn("already reallocated 3 times", g["end_reason"])

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

    def searched(self, kind, minutes_ago):
        """Search after a 4080 group ended `minutes_ago` with `kind`; return the groups created by the search."""
        self.http.on("POST", "/availability/", (200, {"available_gpu_low": 5}))
        self.http.on("GET", "/quotas", (200, {"container_groups_quotas": {"container_replicas_quota": 10}}))
        g = self.group(cls="RTX 4080 (16 GB)", pri="low")
        g["ended"], g["end_kind"] = self.clock[0], kind
        self.clock[0] += minutes_ago * 60
        self.x.cfg["max_active"] = 10
        self.x.search()
        return [c[2]["name"] for c in self.http.made("POST", "/containers$")][1:]

    def test_a_class_that_failed_is_not_tried_again(self):
        made = self.searched("failed", 300)
        self.assertNotIn("biz-rtx4080", made)
        self.assertIn("biz-rtx2060", made)  # classes not tried yet still are

    def test_a_class_that_got_no_pc_is_tried_again_an_hour_later(self):
        self.assertEqual(self.searched("released", 59), ["biz-rtx2060", "biz-rtx4070"])
        self.setUp()
        self.assertEqual(self.searched("released", 61), ["biz-rtx4080-2", "biz-rtx2060", "biz-rtx4070"])  # its old name is skipped

    def test_a_class_with_a_live_group_is_not_doubled(self):
        self.http.on("POST", "/availability/", (200, {"available_gpu_low": 5}))
        self.http.on("GET", "/quotas", (200, {"container_groups_quotas": {"container_replicas_quota": 10}}))
        self.group(cls="RTX 4080 (16 GB)", pri="low")
        self.x.cfg["max_active"] = 10
        self.x.search()
        self.assertNotIn("biz-rtx4080-2", [c[2]["name"] for c in self.http.made("POST", "/containers$")])

    def test_at_most_3_new_groups_an_hour(self):
        self.http.on("POST", "/availability/", (200, {"available_gpu_low": 5}))
        self.http.on("GET", "/quotas", (200, {"container_groups_quotas": {"container_replicas_quota": 10}}))
        self.x.cfg["max_active"] = 10
        self.group(cls="RTX 4080 (16 GB)", pri="low")
        self.group(cls="RTX 2060 (6 GB)", pri="low")  # two made this hour already
        self.x.search()
        made = [c[2]["name"] for c in self.http.made("POST", "/containers$")]
        self.assertEqual(made[2:], ["biz-rtx4070"])  # one more this hour, though the 3060 Ti also qualifies
        self.clock[0] += 3600  # the next hour
        self.x.search()
        made = [c[2]["name"] for c in self.http.made("POST", "/containers$")]
        self.assertEqual(made[3:], ["biz-rtx3060ti"])

    def paying_4080(self):
        self.http.on("POST", "/availability/", (200, {"available_gpu_low": 5}))
        self.http.on("GET", "/quotas", (200, {"container_groups_quotas": {"container_replicas_quota": 10}}))
        self.x.cfg.update(max_active=10, max_new_per_hour=10, max_per_class=2)
        g = self.group(cls="RTX 4080 (16 GB)", pri="low")
        self.step(g, 1, "m1")
        return g

    def test_a_class_paying_on_its_pc_gets_a_second_group(self):
        g = self.paying_4080()
        self.mine(g, "m1", 199, 6)
        self.x.search()
        names = [c[2]["name"] for c in self.http.made("POST", "/containers$")]
        self.assertIn("biz-rtx4080-2", names)
        self.x.search()  # but not a third
        self.assertNotIn("biz-rtx4080-3", [c[2]["name"] for c in self.http.made("POST", "/containers$")])

    def test_no_second_group_before_the_first_proves_itself(self):
        g = self.paying_4080()
        self.mine(g, "m1", 199, 3)  # still warming up
        self.x.search()
        self.assertNotIn("biz-rtx4080-2", [c[2]["name"] for c in self.http.made("POST", "/containers$")])

    def test_the_hourly_cap_comes_from_config(self):
        self.http.on("POST", "/availability/", (200, {"available_gpu_low": 5}))
        self.http.on("GET", "/quotas", (200, {"container_groups_quotas": {"container_replicas_quota": 10}}))
        self.x.cfg.update(max_active=10, max_new_per_hour=10)
        self.x.search()
        self.assertEqual(len(self.http.made("POST", "/containers$")), 4)  # every class in the fixture

    def test_the_retry_wait_comes_from_config(self):
        self.x.cfg["retry_after_min"] = 10
        self.assertIn("biz-rtx4080-2", self.searched("released", 11))

    def test_a_class_that_keeps_getting_no_pc_goes_last(self):
        self.http.on("POST", "/availability/", (200, {"available_gpu_low": 5}))
        self.http.on("GET", "/quotas", (200, {"container_groups_quotas": {"container_replicas_quota": 10}}))
        self.x.cfg.update(max_active=10, max_new_per_hour=1, retry_after_min=10, max_per_class=3)
        for _ in range(3):
            g = self.group(cls="RTX 4080 (16 GB)", pri="low")
            g["ended"], g["end_kind"] = self.clock[0], "released"   # never had a PC
            self.clock[0] += 3600
        self.x.search()
        self.assertEqual(self.http.made("POST", "/containers$")[-1][2]["name"], "biz-rtx2060")  # not the 4080
        self.clock[0] += 181 * 60  # the hold ends
        self.x.search()
        self.assertEqual(self.http.made("POST", "/containers$")[-1][2]["name"], "biz-rtx4080-4")

    def test_a_class_that_mined_isnt_held_back(self):
        self.http.on("POST", "/availability/", (200, {"available_gpu_low": 5}))
        self.http.on("GET", "/quotas", (200, {"container_groups_quotas": {"container_replicas_quota": 10}}))
        self.x.cfg.update(max_active=10, max_new_per_hour=1, retry_after_min=10)
        for had in (False, True, False):
            g = self.group(cls="RTX 4080 (16 GB)", pri="low")
            g["ended"], g["end_kind"], g["had_pc"] = self.clock[0], "released", had
            self.clock[0] += 3600
        self.x.search()
        self.assertEqual(self.http.made("POST", "/containers$")[-1][2]["name"], "biz-rtx4080-4")

    def start_after(self, tries):
        """Search after 4080 groups ended as (priority, had_pc), oldest first; return the new 4080's priority."""
        self.http.on("POST", "/availability/", (200, {f"available_gpu_{p}": 5 for p in w.PRIS}))
        self.http.on("GET", "/quotas", (200, {"container_groups_quotas": {"container_replicas_quota": 10}}))
        self.x.cfg.update(max_active=10, max_new_per_hour=10, retry_after_min=10)
        for pri, had in tries:
            g = self.group(cls="RTX 4080 (16 GB)", pri=pri)
            g["ended"], g["end_kind"], g["had_pc"] = self.clock[0], "released", had
            self.clock[0] += 3600
        self.x.search()
        return next(c[2]["container"]["priority"] for c in self.http.made("POST", "/containers$")[len(tries):]
                    if c[2]["name"].startswith("biz-rtx4080"))

    def test_a_class_whose_last_try_got_no_pc_starts_where_it_last_got_one(self):
        self.assertEqual(self.start_after([("medium", True), ("high", False)]), "medium")
        self.setUp()
        self.assertEqual(self.start_after([("high", True), ("medium", True), ("high", False)]), "medium")  # the latest PC

    def test_a_class_whose_last_try_got_a_pc_starts_at_batch(self):
        self.assertEqual(self.start_after([("high", False), ("medium", True)]), "batch")
        self.setUp()
        self.assertEqual(self.start_after([("high", False)]), "batch")  # never placed: nothing to go on

    def test_respects_max_active(self):
        self.http.on("POST", "/availability/", (200, {"available_gpu_low": 5}))
        self.group(cls="RTX 2060 (6 GB)", pri="low")
        n = len(self.http.made("POST", "/containers$"))
        self.x.search()
        self.assertEqual(len(self.http.made("POST", "/containers$")), n)


class TestUpgrade(Base):
    def full_with_a_2060(self, avail, placed=True):
        """Every slot taken by one 2060 mining at 40 TH/s at low: $0.017/hr. A 4080 at low should make $0.083/hr,
        less about $0.018 to switch (the 2060's profit for 20 min, the 4080's bill for 5): $0.048 more.
        With placed, an earlier 4080 group had a PC here, at low."""
        self.http.on("POST", "/availability/", lambda b: (200, avail if b["gpu_classes"][0] == "c4080" else {}))
        self.http.on("GET", "/quotas", (200, {"container_groups_quotas": {"container_replicas_quota": 10}}))
        self.x.cfg.update(max_active=1, max_new_per_hour=10)
        old = self.group(cls="RTX 4080 (16 GB)", pri="low")
        old.update(ended=self.clock[0], end_kind="released", had_pc=placed)
        self.clock[0] += 3600
        g = self.group(cls="RTX 2060 (6 GB)", pri="low")
        self.step(g, 1, "m1")
        self.mine(g, "m1", 40, 6)
        return g

    def made(self):
        return [(c[2]["name"], c[2]["container"]["priority"]) for c in self.http.made("POST", "/containers$")][2:]

    def test_the_weakest_group_is_swapped_for_a_class_that_should_make_more(self):
        g = self.full_with_a_2060({"available_gpu_low": 5})
        self.x.cfg["upgrade_gain"] = 0.04
        self.x.upgrade(self.clock[0])
        self.assertEqual(g["end_kind"], "released")
        self.assertTrue(self.http.made("DELETE", "/containers/biz-rtx2060$"))
        self.assertEqual(self.made(), [("biz-rtx4080-2", "low")])
        self.assertIn("UPGRADE biz-rtx2060", self.events())

    def test_the_cost_of_switching_counts(self):
        # $0.066 more an hour, but $0.018 of it goes on switching: under a $0.05 bar.
        g = self.full_with_a_2060({"available_gpu_low": 5})
        self.x.upgrade(self.clock[0])
        self.assertIsNone(g["ended"])
        self.x.cfg["swap_wait_min"] = 0
        self.x.upgrade(self.clock[0])
        self.assertEqual(g["end_kind"], "released")

    def test_a_class_is_priced_where_it_last_got_a_pc(self):
        # The last 4080 only got a PC at medium: $0.043/hr there, not $0.083 at low.
        g = self.full_with_a_2060({"available_gpu_low": 5})
        [old] = [x for x in self.x.st["groups"].values() if x["class"] == "RTX 4080 (16 GB)"]
        old["pri"] = "medium"
        self.x.cfg.update(upgrade_gain=0.0, swap_wait_min=0)
        self.x.upgrade(self.clock[0])
        self.assertIn("a RTX 4080 (16 GB) should make $+0.043/hr", self.events())
        self.assertEqual(self.made(), [("biz-rtx4080-2", "low")])  # still asked for at its free priority

    def test_no_swap_to_a_class_whose_latest_group_got_no_pc(self):
        g = self.full_with_a_2060({"available_gpu_low": 5})
        self.x.cfg["upgrade_gain"] = 0.04
        self.clock[0] += 60
        nopc = self.group(cls="RTX 4080 (16 GB)", pri="medium")
        nopc.update(ended=self.clock[0] - 3600, end_kind="released")
        self.clock[0] += 3600
        self.x.upgrade(self.clock[0])
        self.assertIsNone(g["ended"])

    def test_no_swap_to_a_class_that_never_had_a_pc_here(self):
        g = self.full_with_a_2060({"available_gpu_low": 5}, placed=False)
        self.x.upgrade(self.clock[0])
        self.assertIsNone(g["ended"])
        self.assertEqual(self.made(), [])

    def test_no_swap_when_the_gain_is_too_small(self):
        g = self.full_with_a_2060({"available_gpu_low": 5})
        self.x.cfg["upgrade_gain"] = 0.07
        self.x.upgrade(self.clock[0])
        self.assertIsNone(g["ended"])
        self.assertEqual(self.made(), [])

    def test_a_free_batch_pc_is_counted_at_the_low_price(self):
        # At batch a 4080 would make $0.123/hr, $0.106 more; counted at low it's $0.066 more, $0.048 after switching.
        g = self.full_with_a_2060({"available_gpu_batch": 5})
        self.x.cfg["upgrade_gain"] = 0.05
        self.x.upgrade(self.clock[0])
        self.assertIsNone(g["ended"])
        self.x.cfg["upgrade_gain"] = 0.04
        self.x.upgrade(self.clock[0])
        self.assertEqual(self.made(), [("biz-rtx4080-2", "batch")])  # still asked for at batch

    def test_a_group_still_warming_up_is_left_alone(self):
        self.http.on("POST", "/availability/", (200, {"available_gpu_low": 5}))
        self.x.cfg.update(max_active=1, max_new_per_hour=10)
        g = self.group(cls="RTX 2060 (6 GB)", pri="low")
        self.step(g, 1, "m1")
        self.mine(g, "m1", 40, 2)
        self.x.upgrade(self.clock[0])
        self.assertIsNone(g["ended"])

    def test_off_when_upgrade_gain_is_unset(self):
        g = self.full_with_a_2060({"available_gpu_low": 5})
        self.x.cfg["upgrade_gain"] = None
        self.x.upgrade(self.clock[0])
        self.assertIsNone(g["ended"])

    def test_it_checks_every_15_min_once_full_and_not_while_a_slot_is_empty(self):
        self.http.on("GET", "/quotas", (200, {"container_groups_quotas": {"container_replicas_quota": 10}}))
        for full in (True, False):
            self.setUp()
            w.save(w.CFG, {"max_active": 0 if full else 10, "search_every_min": 5})
            checks = []
            self.x.upgrade = lambda t: (checks.append(t), self.x.st.update(last_upgrade=t))
            self.x.search = lambda: None
            for _ in range(31):
                self.clock[0] += 60
                self.x.round()
            self.assertEqual(len(checks), 2 if full else 0)

    def test_salads_quota_caps_the_slots_when_max_active_is_higher(self):
        # max_active is set above the quota so a raised quota is used at once. Until then, full means full.
        self.group(cls="RTX 2060 (6 GB)", pri="low")
        self.x.st["quota"] = 1
        w.save(w.CFG, {"max_active": 30, "search_every_min": 5})
        calls = []
        self.x.search = lambda: calls.append(1)
        self.x.upgrade = lambda t: self.x.st.update(last_upgrade=t)
        for _ in range(16):
            self.clock[0] += 60
            self.x.round()
        self.assertEqual(len(calls), 1)  # only the hourly search
        self.x.st["quota"] = 2           # Salad raises the quota
        for _ in range(16):
            self.clock[0] += 60
            self.x.round()
        self.assertGreater(len(calls), 2)  # now a slot is empty


class TestSearchTiming(Base):
    def rounds(self, minutes, empty=True):
        w.save(w.CFG, {"search_every_min": 5, "max_active": 10 if empty else 0})  # round() re-reads config.json
        self.http.on("GET", "/quotas", (200, {"container_groups_quotas": {"container_replicas_quota": 10}}))
        calls = []
        self.x.search = lambda: calls.append(self.clock[0])
        for _ in range(minutes):
            self.clock[0] += 60
            self.x.round()
        return calls

    def test_it_searches_every_5_min_while_a_slot_is_empty(self):
        self.assertEqual(len(self.rounds(16)), 4)  # at 08:01 (the hour's), 08:06, 08:11 and 08:16

    def test_full_means_only_the_hourly_search(self):
        self.assertEqual(len(self.rounds(16, empty=False)), 1)


class TestLogOutage(Base):
    def test_a_log_outage_doesnt_drop_a_mining_pc(self):
        g = self.group(pri="medium")
        self.step(g, 1, "m1")
        self.mine(g, "m1", 199, 10, shares=False)
        self.x.st["last_log_ok"] = self.clock[0]  # then Salad's log reads fail
        self.step(g, 8, "m1")
        self.assertFalse(self.http.made("POST", "/reallocate"))
        self.x.st["last_log_ok"] = self.clock[0]  # reads work again, and still no reading: the miner stopped
        self.step(g, 1, "m1")
        self.assertTrue(self.http.made("POST", "/instances/inst-m1/reallocate"))

    def test_pool_shares_keep_a_pc_whose_readings_stopped(self):
        g = self.group(pri="medium")
        self.step(g, 1, "m1")
        self.mine(g, "m1", 199, 10)
        for _ in range(8):
            g["last_share"] = self.clock[0]  # the pool still gets shares, though no reading comes
            self.step(g, 1, "m1")
        self.assertFalse(self.http.made("POST", "/reallocate"))

    def test_a_long_outage_is_reported_once(self):
        g = self.group()
        self.x.st["last_log_ok"] = self.clock[0] - 11 * 60
        self.http.on("POST", "/log-entries", (408, {}))
        self.x.logs([g])
        self.x.logs([g])
        self.assertEqual(self.events().count("WARN Salad log reads have failed for 11 min"), 1)


class TestPoolTotals(Base):
    POOL = {"stats": {"paid": "156064257", "balance": "20000000", "hashes": "8875147264", "hashrate_1h": 264765.8},
            "unconfirmed": [{"reward": "102092647"}, {"reward": "50000000"}],
            "unlocked": ["126028:4f27:2262:2266:92794907", "1791673994", "126003:bbff:2260:2266:56925295", "1791670000"],
            "workers": []}

    def test_the_pool_totals_are_read(self):
        self.x.cfg["wallet"] = os.environ["PRL_WALLET"]
        self.http.on("GET", "herominers", (200, self.POOL))
        self.x.pool()
        pt = self.x.st["pool_tot"]
        self.assertAlmostEqual(pt["paid"], 1.56064257)
        self.assertAlmostEqual(pt["balance"], 0.2)
        self.assertAlmostEqual(pt["unconfirmed"], 1.52092647)
        self.assertEqual(pt["blocks"], 4)
        self.assertAlmostEqual(pt["ths_1h"], 264765.8 * 419 / 93644)

    def test_the_report_shows_what_the_pool_credited(self):
        g = self.group()
        g["prl"], g["spend"] = 8.0, 6.0
        self.x.st["pool_tot"] = {"t": T0, "paid": 1.5, "balance": 0.5, "unconfirmed": 4.0, "blocks": 6,
                                 "hashes": 0, "ths_1h": 0}
        self.x.write_report()
        with open(os.path.join(TMP, "report.md")) as f:
            report = f.read()
        # 8 PRL mined for $6.00: $0.750 each. The pool has credited 6 of them. Nothing sold: 8 × (1.39 − 0.75) = +5.12.
        self.assertIn("| **+5.12\\*** |\n", report)  # the All row
        self.assertIn("Spent $6.00* on Salad for 8.000* PRL, $0.750* each.", report)
        self.assertIn("| ⛏️ PRL | 0 TH/s | 0.500 | 4.000 | 1.500 |", report)  # straight from the pool: no stars
        self.assertIn("The PRL pool has credited 75%* of our estimate over 6 blocks, and pays out from 1 PRL "
                      "(from 10 blocks on the table counts on that share).", report)
        self.assertNotIn("Locked", report)


class TestCreditRatio(Base):
    """Once the pool has credited us in 10 blocks, the table counts on the share of our estimate it credited."""
    def report(self, blocks=20, cfg=None):
        g = self.group()
        g["prl"], g["spend"] = 8.0, 6.0
        self.x.st["pool_tot"] = {"t": T0, "paid": 1.5, "balance": 0.5, "unconfirmed": 4.0, "blocks": blocks,
                                 "hashes": 0, "ths_1h": 0}
        self.x.cfg.update(cfg or {})
        self.x.write_report()
        with open(os.path.join(TMP, "report.md")) as f:
            return f.read()

    def test_the_first_line_starts_with_a_readable_time(self):
        self.assertTrue(self.report().startswith("**03:00 EST, Fri 15 Jan** · PRL $1.390 · "))

    def test_the_time_is_new_york_time_and_says_edt_in_summer(self):
        t = w.parse_iso("2026-10-17T15:53:00Z")
        self.assertEqual(w.stamp(t), "11:53 EDT, Sat 17 Oct")

    def test_the_first_line_leaves_out_what_the_all_row_shows(self):
        first = self.report().split("\n")[0]
        self.assertEqual(first, "**03:00 EST, Fri 15 Jan** · PRL $1.390 · 0/1 mining")

    def test_the_table_counts_on_what_the_pool_credits(self):
        report = self.report()
        # 75% of 8 PRL is 6, mined for $6.00: $1.000 each. Worth 6 × 1.39 − 6.00 = +2.34.
        self.assertIn("| **+2.34\\*** |\n", report)  # the All row
        self.assertIn("Spent $6.00* on Salad for 6.000* PRL (8.000* at the network's rate × 75%* credited), $1.000* each.", report)
        self.assertIn("The PRL pool has credited 75%* of our estimate over 20 blocks, and pays out from 1 PRL.", report)
        self.assertNotIn("from 10 blocks on", report)

    def test_the_ratio_has_a_floor(self):
        self.assertIn("× 90%* credited", self.report(cfg={"yield_floor": 0.9}))

    def test_the_rules_use_config_yield(self):
        self.report()
        self.assertAlmostEqual(self.x.usd(), USD_TH_HR * 0.75)        # "auto": the credit ratio
        self.x.cfg["yield"] = 1.0
        self.assertAlmostEqual(self.x.usd(), USD_TH_HR)               # fixed: the network's rate
        self.x.write_report()
        with open(os.path.join(TMP, "report.md")) as f:
            self.assertIn("× 75%* credited", f.read())                 # the table still shows the ratio

    def test_a_pc_judged_at_the_credited_rate(self):
        # 199 TH/s at medium clears 10% at the network's rate (+18%), not at 75% of it (−9%).
        self.report()
        g = list(self.x.st["groups"].values())[0]
        g.update(pri="medium", price=0.19)
        self.x.st["paid_classes"] = []
        self.step(g, 1, "m1")
        self.mine(g, "m1", 199, 6)
        self.assertIn("REALLOCATED", self.events())


class TestMeasuredRates(Base):
    """A class is judged on the median TH/s of the PCs it has mined on here, not its benchmark."""
    def mined(self, cls, pcs, pri="low"):
        for i, ths in enumerate(pcs):
            g = self.group(cls=cls, pri=pri)
            self.step(g, 1, f"p{len(self.x.st['groups'])}")
            self.mine(g, g["machine"], ths, 6)
            self.x.record_rates(self.clock[0])

    def test_the_benchmark_until_a_pc_has_mined(self):
        self.assertEqual(self.x.rate("RTX 4080 (16 GB)"), 199)

    def test_the_median_of_the_pcs_it_mined_on(self):
        self.mined("RTX 4080 (16 GB)", [150, 210, 220], pri="medium")
        self.assertEqual(self.x.rate("RTX 4080 (16 GB)"), 210)

    def test_a_faster_measured_rate_lets_a_class_move_up(self):
        # At its 40 TH/s benchmark a 2060 clears 10% only up to medium ($0.040, +15%) and not high ($0.050).
        # Measured at 60 here, high clears too: 60 × 0.00117 = $0.070 against $0.050 (+29%).
        self.assertLess(self.x.margin("RTX 2060 (6 GB)", "high"), 0.10)
        self.mined("RTX 2060 (6 GB)", [60])
        self.assertGreater(self.x.margin("RTX 2060 (6 GB)", "high"), 0.10)

    def test_old_pcs_are_forgotten(self):
        self.mined("RTX 4080 (16 GB)", [150], pri="medium")
        self.clock[0] += 8 * 86400
        self.x.record_rates(self.clock[0])
        self.assertEqual(self.x.rate("RTX 4080 (16 GB)"), 199)

    def test_the_first_time_it_fills_from_the_ledger(self):
        g = self.group(pri="medium")
        g["readings"] = [[self.clock[0] + 60 * i, "m1", 180.0, 0] for i in range(6)]
        g["ended"] = self.clock[0]
        self.x.record_rates(self.clock[0] + 600)
        self.assertEqual(self.x.rate("RTX 4080 (16 GB)"), 180)

    def test_a_new_group_is_created_at_the_measured_rate(self):
        self.mined("RTX 4080 (16 GB)", [210], pri="medium")
        g = self.group(pri="low")
        self.assertEqual(g["rate"], 210)
        self.assertIn("margin at 210 TH/s", self.events())


class TestStartingBoxes(Base):
    def test_a_box_starting_on_a_new_pc_is_left_out_of_the_profit_total(self):
        a, b = self.group(pri="medium"), self.group(cls="RTX 2060 (6 GB)", pri="low")
        self.step(a, 1, "m1")
        self.mine(a, "m1", 200, 5)
        self.step(b, 0, "m2")
        self.x.write_report()
        with open(os.path.join(TMP, "report.md")) as f:
            report = f.read()
        self.assertIn(f"| ⛏️{w.NB}2060{w.NB}low | 0 | 0.03 | — |", report)
        # $/hr counts both bills; Profit/hr only the 4080: 200 × 0.00117 − 0.19 = +0.044.
        self.assertIn("| **All** | **200** | **0.22** | **+0.04\\* (+19%\\*)** |", report)
        self.assertIn(" · 1/2 mining\n", report)  # the starting box counts as not mining yet

    def test_stars_inside_bold_are_escaped(self):
        # "**+0.27* (+24%*)**" doesn't render as bold: Markdown takes the stars for more emphasis.
        self.assertEqual(w.bold("+0.27* (+24%*)"), "**+0.27\\* (+24%\\*)**")

    def test_ended_groups_are_one_row_with_their_count(self):
        for kind in ("released", "released", "failed"):
            g = self.group()
            g["ended"], g["end_kind"], g["prl"], g["spend"] = self.clock[0], kind, 1.0, 1.0
        self.x.write_report()
        with open(os.path.join(TMP, "report.md")) as f:
            report = f.read()
        # 3 PRL at $1.39 less $3.00 spent
        self.assertIn(f"| Others{w.NB}×3 | — | — | — | +1.17* |", report)

    def test_estimates_are_starred_and_api_figures_are_not(self):
        a = self.group(pri="medium")
        self.step(a, 1, "m1")
        self.mine(a, "m1", 200, 5)
        self.x.write_report()
        with open(os.path.join(TMP, "report.md")) as f:
            report = f.read()
        # TH/s (the miner's log) and $/hr (Salad's price list) come from an API; the rest is our own count.
        self.assertIn(f"| ⛏️{w.NB}4080{w.NB}med | 200 | 0.19 | +0.04* (+19%*) |", report)
        self.assertIn("* our own count or estimate, or entered by hand", report)
        self.assertRegex(report, r"Salad credit left \$[\d.]+\* of \$[\d.]+\*, \d+\* h at \$[\d.]+/hr \(our count")

    def test_a_box_with_no_reading_after_15_min_counts_again(self):
        a = self.group(pri="medium")
        self.step(a, 1, "m1")
        self.clock[0] += 16 * 60
        self.x.write_report()
        with open(os.path.join(TMP, "report.md")) as f:
            self.assertIn("| 0 | 0.19 | -0.19* |", f.read())


class TestBadPcs(Base):
    def test_a_pc_another_group_left_is_left_at_once(self):
        a = self.group(pri="medium")
        self.step(a, 1, "m1")
        self.mine(a, "m1", 120, 6)   # slow: left
        self.assertIn("REALLOCATED biz-rtx4080 off PC m1", self.events())
        b = self.group(pri="medium")
        self.step(b, 1, "m1")        # another group gets the same PC
        self.assertEqual(len(self.http.made("POST", "/instances/inst-m1/reallocate")), 2)
        self.assertIn("REALLOCATED biz-rtx4080-2 off PC m1 (biz-rtx4080 left this PC at", self.events())
        self.assertIn("not counted toward its 3", self.events())
        self.assertEqual(b["reallocs"], 0)

    def test_the_bad_list_forgets_after_a_day(self):
        a = self.group(pri="medium")
        self.step(a, 1, "m1")
        self.mine(a, "m1", 120, 6)
        self.clock[0] += 25 * 3600
        b = self.group(pri="medium")
        self.step(b, 1, "m1")
        self.assertEqual(len(self.http.made("POST", "/instances/inst-m1/reallocate")), 1)

    def test_a_pc_salad_took_back_isnt_bad(self):
        a = self.group(pri="medium")
        self.step(a, 1, "m1")
        self.mine(a, "m1", 200, 6)
        self.step(a, 1, None)       # Salad takes the PC back
        self.assertEqual(self.x.st.get("bad_pcs", {}), {})


class TestCritic(Base):
    def setUp(self):
        super().setUp()
        self.x.st["created"] = T0 - 7200
        self.g = self.group()

    def pool(self, ths_1h=200.0, work=None, cred=None, blocks=100):
        work = self.g["prl"] if work is None else work
        hashes = work * 3600 / (419 / 93644 * USD_TH_HR / PRICE)
        cred = work if cred is None else cred
        self.x.st["pool_tot"] = {"t": self.clock[0], "paid": cred, "balance": 0.0, "unconfirmed": 0.0,
                                 "blocks": blocks, "hashes": hashes, "ths_1h": ths_1h}

    def mined_an_hour(self, ths=200.0):
        self.g["readings"] = [[self.clock[0] - 60 * i, "m1", ths, 0] for i in range(60)]
        self.g["prl"] = 5.0

    def test_no_drift_when_the_pool_agrees(self):
        self.mined_an_hour()
        self.pool(ths_1h=190.0)
        self.x.critic(self.clock[0])
        self.assertNotIn("DRIFT", self.events())

    def test_the_pool_seeing_less_hashrate_is_drift(self):
        self.mined_an_hour()
        self.pool(ths_1h=150.0)
        self.x.critic(self.clock[0])
        self.assertIn("DRIFT hashrate: the pool saw 150 TH/s over the last hour; the miners reported 200 (-25%)", self.events())
        self.x.critic(self.clock[0] + 60)  # once an hour
        self.assertEqual(self.events().count("DRIFT"), 1)

    def test_a_ledger_ahead_of_the_pools_work_is_drift(self):
        self.mined_an_hour()
        self.pool(work=4.0)
        self.x.critic(self.clock[0])
        self.assertIn("DRIFT ledger: the pool's count of our work is worth 4.000 PRL; the ledger says 5.000 (-20%)",
                      self.events())

    def test_credit_short_by_more_than_luck_is_drift(self):
        self.mined_an_hour()
        self.pool(cred=3.0, blocks=100)  # 2 / sqrt(100) = 20%; 40% short
        self.x.critic(self.clock[0])
        self.assertIn("DRIFT credit: the pool has credited 3.000 PRL for work worth 5.000 (-40%", self.events())

    def test_credit_short_over_few_blocks_is_luck(self):
        self.mined_an_hour()
        self.pool(cred=3.0, blocks=6)  # 2 / sqrt(6) = 82%
        self.x.critic(self.clock[0])
        self.assertNotIn("DRIFT", self.events())


class TestTraderStatus(Base):
    CFG = {"project": "p1", "service": "s1", "environment": "e1"}
    STATUS = {"t": 1800003900, "market": "prlusdt", "prl": 0.0, "prl_locked": 0.0, "usdt": 7.08, "usdt_locked": 0.0,
              "sold": 5.2564, "received": 7.1, "fees": 0.0142, "trades": 2,
              "last": {"order": 77, "sold": 5.2564, "average": 1.3507}}

    def railway(self, deployments=("d2", "d1"), logs=None, status=200):
        logs = logs if logs is not None else {"d2": ["starting", "STATUS " + json.dumps(self.STATUS)]}
        def answer(body):
            q = body["query"]
            if "deployments(" in q:
                return status, {"data": {"deployments": {"edges": [{"node": {"id": d}} for d in deployments]}}}
            return status, {"data": {"deploymentLogs": [{"message": m, "timestamp": "2027-01-15T08:05:01Z"}
                                                        for m in logs.get(body["variables"]["id"], [])]}}
        self.http.on("POST", "backboard.railway.com", answer)

    def test_the_newest_status_line_is_read(self):
        self.x.cfg["trader"] = self.CFG
        self.railway()
        self.x.trader()
        self.assertEqual(self.x.st["trader_status"]["usdt"], 7.08)
        q = self.http.made("POST", "backboard")[0][2]
        self.assertEqual(q["variables"]["i"], {"projectId": "p1", "serviceId": "s1", "environmentId": "e1"})

    def test_an_older_deployment_is_read_when_the_newest_has_no_status(self):
        self.x.cfg["trader"] = self.CFG
        self.railway(logs={"d2": ["building"], "d1": ["STATUS " + json.dumps(self.STATUS)]})
        self.x.trader()
        self.assertEqual(self.x.st["trader_status"]["sold"], 5.2564)

    def test_read_at_most_every_5_min(self):
        self.x.cfg["trader"] = self.CFG
        self.railway()
        self.x.trader()
        self.clock[0] += 120
        self.x.trader()
        self.assertEqual(len(self.http.made("POST", "backboard")), 2)   # deployments + one deployment's logs
        self.clock[0] += 300
        self.x.trader()
        self.assertEqual(len(self.http.made("POST", "backboard")), 4)

    def test_the_token_is_sent_as_a_project_token(self):
        self.x.cfg["trader"] = self.CFG
        self.railway()
        os.environ["RAILWAY_TOKEN"] = "tok"
        try:
            self.x.trader()
        finally:
            del os.environ["RAILWAY_TOKEN"]
        self.assertEqual(self.http.headers, {"Project-Access-Token": "tok"})

    def test_off_without_config(self):
        self.railway()
        self.x.trader()
        self.assertFalse(self.http.made("POST", "backboard"))

    def test_a_refused_read_warns_once_and_keeps_the_last_status(self):
        self.x.cfg["trader"] = self.CFG
        self.x.st["trader_status"] = dict(self.STATUS)
        self.http.on("POST", "backboard.railway.com", (200, {"errors": [{"message": "Not Authorized"}]}))
        self.x.trader()
        self.clock[0] += 600
        self.x.trader()
        self.assertEqual(self.events().count("WARN can't read the trader's Railway logs"), 1)
        self.assertEqual(self.x.st["trader_status"]["usdt"], 7.08)

    def test_the_report_shows_safetrade(self):
        g = self.group()
        g["prl"], g["spend"] = 10.0, 8.0   # $0.80 a PRL
        self.x.st["trader_status"] = dict(self.STATUS)
        self.x.write_report()
        with open(os.path.join(TMP, "report.md")) as f:
            report = f.read()
        self.assertIn("SafeTrade at the 09:05Z run: 5.256 PRL sold for 7.09 USDT net (avg 1.3507), holds 7.08 USDT and "
                      "0.000 PRL; last run sold 5.2564 PRL at 1.3507.", report)
        # The box's Profit $ counts its sold PRL at what it sold for: 7.0858 USDT net + 4.7436 unsold PRL × $1.39
        # − $8.00 = +5.68, not 10 × 1.39 − 8.
        self.assertIn("| — | +5.68* |\n", report)
        self.assertNotIn("Locked", report)
        self.assertIn("| **+5.68\\*** |\n", report)  # the All row

    def test_sales_entered_by_hand_count_until_the_logs_can_be_read(self):
        g = self.group()
        g["prl"], g["spend"] = 10.0, 8.0
        self.x.cfg["safetrade"] = {"sold": 5.2564, "received": 7.1, "fees": 0.0142, "usdt": 7.08}
        self.x.write_report()
        with open(os.path.join(TMP, "report.md")) as f:
            report = f.read()
        self.assertIn("SafeTrade entered by hand: 5.256* PRL sold for 7.09* USDT net (avg 1.3507*)", report)
        self.assertNotIn("Locked", report)


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
        self.assertIn(f"| ⛏️{w.NB}4080{w.NB}batch | 199 |", report)
        self.assertIn(" · 1/1 mining\n", report)
        self.assertTrue(all(l.count("|") == 6 for l in report.splitlines() if l.startswith("|")))

    def test_boxes_are_sorted_by_profit_an_hour(self):
        slow = self.group(cls="RTX 2060 (6 GB)", pri="low")
        fast = self.group(cls="RTX 4080 (16 GB)", pri="low")
        self.group(cls="RTX 4070 (12 GB)", pri="low")  # waiting: no PC
        for g, m, ths in ((slow, "m1", 40), (fast, "m2", 199)):
            g["has_pc"], g["machine"] = True, m
            g["readings"] = [[self.clock[0] - 60, m, float(ths), 0]]
        self.x.write_report()
        with open(os.path.join(TMP, "report.md")) as f:
            boxes = [l.split(" | ")[0] for l in f.read().splitlines() if l.startswith("| ⛏") or l.startswith("| ⏳")]
        self.assertEqual([b.split("\u00a0")[1] for b in boxes], ["4080", "2060", "4070"])

    def test_medium_is_shortened_in_the_table(self):
        g = self.group(pri="medium")
        g["has_pc"], g["machine"] = True, "abcdef1234"
        self.x.write_report()
        with open(os.path.join(TMP, "report.md")) as f:
            self.assertIn(f"| ⛏️{w.NB}4080{w.NB}med |", f.read())

    def test_a_busy_round_reads_more_pages(self):
        g = self.group()
        g["created"] = T0 - 3600
        lines = [{"text_log": f"[mining g0] 07:{i // 60:02d}:{i % 60:02d} 199.0 TH/s · {i} accepted · 0 rejected · up 1h",
                  "time": w.iso(T0 - 600 + 4 * i), "resource": {"labels": {"container_group_name": "biz-rtx4080", "machine_id": "m1"}}}
                 for i in range(150)]  # a line every 4 s: busier than 10 boxes starting at once
        def page(b):  # like Salad: at most page_size lines from the window, and not the oldest ones
            start, end = w.parse_iso(b["start_time"]), w.parse_iso(b["end_time"])
            return 200, {"items": [x for x in lines if start <= w.parse_iso(x["time"]) <= end][-b["page_size"]:]}
        self.http.on("POST", "/log-entries", page)
        got = self.x.logs([g])
        self.assertEqual(len(got), 150)  # every line, and the one on the split counted once
        self.assertLess(len(self.http.made("POST", "/log-entries")), 10)
        self.assertEqual(self.x.st["last_log_end"], self.clock[0])

    def test_a_failed_read_is_tried_again_next_round(self):
        g = self.group()
        self.x.st["last_log_end"] = self.clock[0] - 120
        self.http.on("POST", "/log-entries", (408, {}))
        self.assertEqual(self.x.logs([g]), [])
        self.assertEqual(self.x.st["last_log_end"], self.clock[0] - 120)

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


XMR = "8" + "a" * 94


class TestMonero(Base):
    """Monero on the box's CPU: the groups config names get release v0.6.0, more vCPUs and the address."""
    def setUp(self):
        super().setUp()
        self.x.cfg.update(xmr_wallet=XMR, xmr_groups=["biz-rtx4080"])

    def test_the_command_adds_the_worker_and_the_address(self):
        cmd = w.cmd_for("biz-rtx4080", "prl1abc", "v0.6.0", XMR)
        sed = re.search(r"(sed .*?) > /root/box\.sh", cmd).group(1)
        # The three box.sh lines it changes, as they are at the pinned commit.
        box = ("  declare -A dl; for f in llmjob-earn-cli-linux pearl_core.node pearl_core_cu13.node; do   # all three at once\n"
               "BASE='{\"core\":\"release\",\"env\":{},\"flags\":\"\"}'\n"
               "  rm -rf $R && mkdir -p $R && cp $D/rel/cli $R/\n")
        import subprocess
        out = subprocess.run(["sh", "-c", sed], input=box, capture_output=True, text=True).stdout
        self.assertIn("pearl_core_cu13.node llmjob-xmr-worker; do", out)
        self.assertIn(f"BASE='{{\"core\":\"release\",\"env\":{{}},\"flags\":\"--xmr-address {XMR}\"}}'", out)
        self.assertIn("cp $D/rel/cli $D/rel/llmjob-xmr-worker $R/ && chmod +x $R/llmjob-xmr-worker", out)
        self.assertIn("--cpu-threads 3", w.cmd_for("biz-rtx4080", "prl1abc", "v0.6.0", XMR, 3))
        self.assertNotIn("xmr", w.cmd_for("biz-rtx4080", "prl1abc", "v0.6.0"))

    def test_a_named_group_is_created_with_monero(self):
        g = self.group()
        body = self.http.made("POST", "/containers$")[-1][2]
        res = body["container"]["resources"]
        self.assertEqual((res["cpu"], res["memory"], res["gpu_classes"]), (8, 8192, ["c4080"]))
        self.assertIn(f"--xmr-address {XMR}", body["container"]["command"][2])
        self.assertIn("VER=v0.6.0;", body["container"]["command"][2])
        self.assertTrue(g["xmr"])
        self.assertEqual(g["ver"], "v0.6.0")
        self.assertIn("Monero on 8 vCPUs", self.events())

    def test_other_groups_are_created_as_before(self):
        self.x.cfg["xmr_groups"] = []
        g = self.group()
        body = self.http.made("POST", "/containers$")[-1][2]
        self.assertEqual((body["container"]["resources"]["cpu"], body["container"]["resources"]["memory"]), (2, 4096))
        self.assertNotIn("xmr", body["container"]["command"][2])
        self.assertFalse(g["xmr"])

    def test_a_running_group_is_switched_on_and_off(self):
        self.x.cfg["xmr_groups"] = []
        g = self.group()
        self.x.cfg["xmr_groups"] = ["*"]
        g.update(has_pc=True, instance="i1")
        self.http.on("PATCH", "/containers/biz-rtx4080", (200, {}))
        self.http.on("POST", "/instances/i1/restart", (202, {}))
        self.x.xmr_sync([g], self.clock[0])
        patch = self.http.made("PATCH", "/containers/biz-rtx4080")[-1][2]["container"]
        self.assertEqual(patch["resources"], {"cpu": 8, "memory": 8192})
        self.assertIn(XMR, patch["command"][2])
        self.assertTrue(g["xmr"])
        # A running instance keeps its old command until it restarts, so the box is restarted.
        self.assertEqual(len(self.http.made("POST", "/containers/biz-rtx4080/instances/i1/restart")), 1)
        self.assertIn("MONERO ON biz-rtx4080: miner v0.6.0 on 8 vCPUs and 8 GB; box restarted",
                      self.events())
        g["has_pc"] = False  # no PC: nothing to restart
        self.x.xmr_sync([g], self.clock[0] + 700)  # already on: nothing sent
        self.assertEqual(len(self.http.made("PATCH", "/containers/biz-rtx4080")), 1)
        self.x.cfg["xmr_wallet"] = ""
        self.x.xmr_sync([g], self.clock[0] + 1400)
        patch = self.http.made("PATCH", "/containers/biz-rtx4080")[-1][2]["container"]
        self.assertEqual(patch["resources"], {"cpu": 2, "memory": 4096})
        self.assertNotIn("xmr", patch["command"][2])
        self.assertFalse(g["xmr"])
        self.assertIn("MONERO OFF biz-rtx4080: miner v0.5.13 on 2 vCPUs and 4 GB; it starts that way on its next PC",
                      self.events())
        self.assertEqual(len(self.http.made("POST", "/restart")), 1)

    def test_a_box_salad_dropped_on_the_switch_is_said_so(self):
        # A PC that can't give the new vCPUs loses the instance at the PATCH, so the restart finds none (404).
        self.x.cfg["xmr_groups"] = []
        g = self.group()
        self.x.cfg["xmr_groups"] = ["*"]
        g.update(has_pc=True, instance="i1", machine="m1")
        self.http.on("PATCH", "/containers/biz-rtx4080", (200, {}))
        self.http.on("POST", "/instances/i1/restart", (404, {}))
        self.x.xmr_sync([g], self.clock[0])
        self.assertTrue(g["xmr"])
        self.assertIn("Salad dropped its instance (the PC can't take the new size); it starts that way on its next PC",
                      self.events())

    def test_a_refused_switch_is_tried_again_after_10_minutes(self):
        self.x.cfg["xmr_groups"] = []
        g = self.group()
        self.x.cfg["xmr_groups"] = ["biz-rtx4080"]
        self.http.on("PATCH", "/containers/biz-rtx4080", (400, {"title": "nope"}))
        self.x.xmr_sync([g], self.clock[0])
        self.x.xmr_sync([g], self.clock[0] + 300)
        self.assertEqual(len(self.http.made("PATCH", "/containers/biz-rtx4080")), 1)
        self.assertIn("WARN Monero on for biz-rtx4080 refused: HTTP 400", self.events())
        self.http.on("PATCH", "/containers/biz-rtx4080", (200, {}))
        self.x.xmr_sync([g], self.clock[0] + 600)
        self.assertTrue(g["xmr"])

    def pool(self, hashes, hashrate=900, name="biz-rtx4080-g0"):
        return 200, {"stats": {"balance": "5000000000", "paid": "0", "hashrate_1h": 32078.2},
                     "unconfirmed": [{"reward": "1000000000"}],
                     "workers": [{"name": name, "hashes": str(hashes), "hashrate": hashrate},
                                 {"name": "rig01", "hashes": "999999999", "hashrate": 700}]}

    def test_the_pool_counts_each_box_s_new_hashes(self):
        g = self.group()
        self.x.st["xmr_mkt"] = {"t": self.clock[0], "price": 500.0, "per_hash": 1e-12, "usd_khs_hr": 1e-12 * 3.6e6 * 500}
        self.http.on("GET", "stats_address", self.pool(4_000_000))
        self.x.xmr_pool([g], self.clock[0])
        self.assertAlmostEqual(g["xmr_coins"], 4e-6)
        self.assertEqual(g["xmr_khs"], 0.9)
        self.http.on("GET", "stats_address", self.pool(5_000_000))
        self.x.xmr_pool([g], self.clock[0] + 60)
        self.assertAlmostEqual(g["xmr_coins"], 5e-6)  # only the million new hashes
        self.http.on("GET", "stats_address", self.pool(200_000))  # the pool restarted its count
        self.x.xmr_pool([g], self.clock[0] + 120)
        self.assertAlmostEqual(g["xmr_coins"], 5.2e-6)
        self.assertAlmostEqual(self.x.st["xmr_tot"]["balance"], 0.005)
        self.assertAlmostEqual(self.x.st["xmr_tot"]["pending"], 0.001)   # the unconfirmed rewards
        self.assertAlmostEqual(self.x.st["xmr_tot"]["khs_1h"], 32.0782)  # the pool's own 1-hour hashrate

    def test_the_market_gives_what_a_hash_earns(self):
        self.http.on("GET", "monero.herominers.com/api/stats$",
                     (200, {"network": {"difficulty": 600_000_000_000}, "lastblock": {"reward": 600_000_000_000}}))
        self.http.on("GET", "coingecko", (200, {"monero": {"usd": 500}}))
        self.x.xmr_market()
        mk = self.x.st["xmr_mkt"]
        self.assertAlmostEqual(mk["per_hash"], 0.6 / 6e11 * 0.991)
        self.assertAlmostEqual(mk["usd_khs_hr"], 0.6 / 6e11 * 0.991 * 3.6e6 * 500)

    def test_the_table_shows_monero(self):
        g = self.group()
        plain = self.group(cls="RTX 2060 (6 GB)")
        self.x.st["xmr_mkt"] = {"t": self.clock[0], "price": 500.0, "per_hash": 1e-12, "usd_khs_hr": 0.002}
        self.x.st["xmr_tot"] = {"balance": 0.005, "paid": 0.0, "pending": 0.0001, "khs_1h": 1.5}
        g.update(has_pc=True, machine="m1", readings=[[self.clock[0] - 60, "m1", 200.0, 0]],
                 xmr_khs=2.0, xmr_seen=self.clock[0], xmr_coins=0.0001)
        self.x.write_report()
        with open(os.path.join(TMP, "report.md")) as f:
            report = f.read()
        self.assertIn("| Box | TH/s | kH/s | $/hr | Profit/hr (margin) | Profit $ |", report)
        # 200 TH/s × 0.00117 + 2 kH/s × 0.002 = 0.238 an hour, less 0.11 billed. 0.0001 XMR is $0.05, in its Profit $.
        self.assertIn(f"| ⛏️{w.NB}4080{w.NB}batch | 200 | 2.0 | 0.11 | +0.13* (+54%*) | +0.05* |", report)
        self.assertIn(f"| ⏳{w.NB}2060{w.NB}batch | — | — | — | — | +0.00* |", report)
        self.assertIn("| **All** | **200** | **2.0** | **0.11** |", report)
        self.assertIn("| Pool | Hashrate (1h) | Confirmed | Pending | Paid |", report)
        self.assertIn("| 🪙 XMR | 1.5 kH/s | 0.005000 | 0.000100 | 0.000000 |", report)  # straight from the pool: no stars
        self.assertIn("Our boxes have mined 0.000100* XMR ($0.05* at $500)", report)
        rows = [l for l in report.split("| Pool |")[0].splitlines() if l.startswith("|")]
        self.assertTrue(all(l.count("|") == 7 for l in rows))  # one table, with a kH/s column
        self.assertTrue(all(len(l) <= 90 for l in rows), max(rows, key=len))  # each row on one line in a chat window

    def test_without_an_address_the_table_is_unchanged(self):
        self.x.cfg["xmr_wallet"] = ""
        self.group()
        self.x.write_report()
        with open(os.path.join(TMP, "report.md")) as f:
            report = f.read()
        self.assertNotIn("kH/s", report)
        self.assertNotIn("Monero", report)


class TestKryptex(Base):
    """Up to kryptex_groups new groups mine Pearl on Kryptex's PPS+ pool through a port forward on the box."""
    def setUp(self):
        super().setUp()
        self.x.cfg.update(wallet=os.environ["PRL_WALLET"], kryptex_groups=2)

    def report(self):
        self.x.write_report()
        with open(os.path.join(TMP, "report.md")) as f:
            return f.read()

    def test_the_command_points_every_herominers_region_at_a_forward_to_kryptex(self):
        cmd = w.cmd_for("biz-rtx4080", "prl1abc", "v0.6.0", kryptex="prl-us.kryptex.network:7048")
        self.assertIn("install -y -qq curl ca-certificates socat", cmd)
        self.assertIn("socat TCP-LISTEN:1200,fork,reuseaddr TCP:prl-us.kryptex.network:7048", cmd)
        for r in w.HM_REGIONS:
            self.assertIn(f" {r}.pearl.herominers.com", cmd)
        self.assertLess(cmd.index("$E/hosts"), cmd.index("bash /root/box.sh"))
        self.assertNotIn("; curl", cmd)  # Salad refuses "; curl"
        self.assertNotIn("/etc/hosts", cmd)  # and its firewall refuses this path (HTTP 403)
        self.assertNotIn("socat", w.cmd_for("biz-rtx4080", "prl1abc", "v0.6.0"))

    def test_only_the_first_groups_up_to_the_count_mine_on_kryptex(self):
        a, b, c = self.group(), self.group(), self.group()
        self.assertEqual([g["kx"] for g in (a, b, c)], [True, True, False])
        cmds = [p[2]["container"]["command"][2] for p in self.http.made("POST", "/containers$")]
        self.assertEqual(["socat" in x for x in cmds], [True, True, False])
        self.assertIn("CREATED biz-rtx4080: RTX 4080 (16 GB) at batch", self.events())
        self.assertIn(", Pearl on Kryptex", self.events())
        a["ended"] = self.clock[0]
        self.assertTrue(self.group()["kx"])  # one ended, so the next takes its place

    def test_kryptex_workers_and_totals_are_read(self):
        wl = os.environ["PRL_WALLET"]
        self.http.on("GET", f"v3/miner/workers/{wl}", (200, {"results": [
            {"worker": "biz-rtx4080-g0", "hashrate": 2.0e14, "shares": 12, "last_share": int(T0)}]}))
        self.http.on("GET", f"v1/miner/balance/{wl}", (200, {"total": 0.6, "confirmed": 0.4, "unconfirmed": 0.2}))
        self.http.on("GET", f"v1/miner/payouts/{wl}", (200, {"results": [{"received": 1.5}, {"received": 1.0}]}))
        self.http.on("GET", "v1/pool/info", (200, {"net_hashrate": 1.0e19, "block_reward": 2000.0, "block_time": 200}))
        ws = self.x.kx_pool()
        self.assertEqual(ws["biz-rtx4080-g0"]["shares"], 12)
        kt = self.x.st["kx_tot"]
        self.assertEqual((kt["total"], kt["confirmed"], kt["unconfirmed"], kt["paid"]), (0.6, 0.4, 0.2, 2.5))
        # The network: 2000 PRL × 18 blocks an hour ÷ PRL per TH-hr, over Kryptex's 1e19 for it.
        self.assertAlmostEqual(kt["unit_th"], 2000 * 18 / (USD_TH_HR / PRICE) / 1.0e19)
        self.clock[0] += 60
        self.x.kx_pool()
        self.assertEqual(len(self.http.made("GET", "v1/miner/balance")), 1)  # totals at most every 5 minutes
        self.assertEqual(len(self.http.made("GET", "v3/miner/workers")), 2)

    def test_each_pool_has_its_own_credit_ratio(self):
        kg, hg = self.group(), self.group()
        hg["kx"] = False
        kg["prl"], hg["prl"] = 4.0, 10.0
        self.x.st["pool_tot"] = {"t": T0, "paid": 6.0, "balance": 1.0, "unconfirmed": 1.0, "blocks": 20,
                                 "hashes": 0, "ths_1h": 0}
        self.assertAlmostEqual(self.x.credit_ratio(), 0.8)  # 8 credited for HeroMiners' 10, not for all 14
        self.assertAlmostEqual(self.x.kx_ratio(), 0.97)     # nothing read from Kryptex yet: 1 less its fee
        self.x.st["kx_tot"] = {"t": T0, "total": 1.88, "confirmed": 1.0, "unconfirmed": 0.88, "paid": 2.0}
        self.assertAlmostEqual(self.x.kx_ratio(), 0.97)
        self.assertAlmostEqual(self.x.ratio(kg), 0.97)
        self.assertAlmostEqual(self.x.ratio(hg), 0.8)

    def test_the_report_marks_kryptex_boxes_and_shows_both_pools(self):
        kg = self.group()
        hg = self.group()
        hg["kx"] = False
        kg["prl"], kg["spend"], hg["prl"], hg["spend"] = 4.0, 3.0, 10.0, 8.0
        kg["has_pc"], kg["pool_ths"] = True, 199.6
        self.x.st["pool_tot"] = {"t": T0, "paid": 6.0, "balance": 1.0, "unconfirmed": 1.0, "blocks": 20,
                                 "hashes": 0, "ths_1h": 0}
        self.x.st["kx_tot"] = {"t": T0, "total": 1.88, "confirmed": 1.0, "unconfirmed": 0.88, "paid": 2.0}
        report = self.report()
        self.assertIn(f"| ⏳{w.NB}4080{w.NB}batch{w.NB}(K) |", report.replace("⛏️", "⏳"))
        self.assertIn(f"| ⏳{w.NB}4080-2{w.NB}batch |", report)
        self.assertIn(f"| ⛏️ PRL{w.NB}HeroMiners | 0 TH/s | 1.000 | 1.000 | 6.000 |", report)
        self.assertIn(f"| ⛏️ PRL{w.NB}Kryptex | 200 TH/s | 1.000 | 0.880 | 2.000 |", report)
        self.assertIn("(K) mines Pearl on Kryptex (PPS+), the rest on HeroMiners.", report)
        self.assertIn("The PRL pool HeroMiners has credited 80%* of our estimate over 20 blocks", report)
        self.assertIn("Kryptex has credited 3.880 PRL for 4.000* at the network's rate (97%*; PPS+ pays each share "
                      "less its 3% fee), and pays out from 1 PRL", report)
        # 4 × 0.97 + 10 × 0.8 = 11.88 PRL credited, for $11.00: (11.88 × 1.39) − 11 = +5.51.
        self.assertIn("| **+5.51\\*** |\n", report)
        self.assertIn("Spent $11.00* on Salad for 11.880* PRL (14.000* at the network's rate × 85%* credited)", report)

    def test_a_round_reads_a_kryptex_box_from_kryptex_and_logs_its_miner_lines(self):
        self.x.cfg["search"] = False
        g = self.group()
        self.http.on("GET", "/containers/biz-rtx4080", (200, {"current_state": {"status": "running"}}))
        self.http.on("GET", "/instances", (200, {"instances": [{"id": "i1", "machine_id": "m1", "state": "running"}]}))
        items = [{"text_log": "[minerlog g0] pool accepted login as biz-rtx4080-g0", "time": "2027-01-15T08:06:00Z",
                  "resource": {"labels": {"container_group_name": "biz-rtx4080", "machine_id": "m1"}}}]
        self.http.on("POST", "/log-entries", (200, {"items": items}))
        # HeroMiners has a worker of that name too (an earlier group): a Kryptex box must not take its figures.
        self.http.on("GET", "herominers", (200, {"workers": [{"name": "biz-rtx4080-g0", "hashrate": 44500,
                                                               "lastShare": int(T0) + 450}]}))
        self.http.on("GET", "v3/miner/workers", (200, {"results": [
            {"worker": "biz-rtx4080-g0", "hashrate": 5.0, "shares": 3, "last_share": (int(T0) + 400) * 1000}]}))
        self.x.st["kx_tot"] = {"t": T0 + 400, "total": 0, "confirmed": 0, "unconfirmed": 0, "paid": 0, "unit_th": 40.0}
        g["created"] = T0 - 3600
        self.clock[0] = T0 + 500
        self.x.round()
        self.assertEqual(g["last_share"], T0 + 400)
        self.assertAlmostEqual(g["pool_ths"], 200.0)
        self.assertIn("LOG biz-rtx4080 PC m1: [minerlog g0] pool accepted login as biz-rtx4080-g0", self.events())

    def test_without_kryptex_groups_the_report_is_unchanged(self):
        self.x.cfg["kryptex_groups"] = 0
        g = self.group()
        g["prl"], g["spend"] = 8.0, 6.0
        self.x.st["pool_tot"] = {"t": T0, "paid": 1.5, "balance": 0.5, "unconfirmed": 4.0, "blocks": 6,
                                 "hashes": 0, "ths_1h": 0}
        report = self.report()
        self.assertIn("| ⛏️ PRL | 0 TH/s | 0.500 | 4.000 | 1.500 |", report)
        self.assertNotIn("Kryptex", report)
        self.assertNotIn("(K)", report)


if __name__ == "__main__":
    unittest.main()
