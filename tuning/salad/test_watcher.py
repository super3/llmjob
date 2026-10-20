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


def shown(row):
    """A Markdown table row as a chat window shows it: without bold marks or the backslashes before stars."""
    return row.replace("**", "").replace("\\*", "*")


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

    def test_a_blocked_pc_goes_on_the_bad_list_and_the_next_group_leaves_it_free(self):
        a = self.group()
        self.step(a, 1, "m1")
        a["blocked"].append("m1")
        self.step(a, 1, "m1")
        self.assertEqual(self.x.st["bad_pcs"]["m1"]["why"], "PC blocks outbound port 1200")
        b = self.group()
        self.step(b, 1, "m1")                      # Salad hands the same PC to another group
        self.assertEqual(b["reallocs"], 0)         # moved off at once, without using one of its 3
        self.assertIn("(biz-rtx4080 left this PC at 08:02Z: PC blocks outbound port 1200); not counted toward its 3",
                      self.events())

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

    def test_a_class_that_failed_is_not_tried_again_within_a_day(self):
        made = self.searched("failed", 300)
        self.assertNotIn("biz-rtx4080-2", made)
        self.assertIn("biz-rtx2060", made)  # classes not tried yet still are

    def test_a_class_that_failed_is_tried_again_after_retry_failed_h(self):
        self.assertNotIn("biz-rtx4080-2", self.searched("failed", 24 * 60 - 1))
        self.setUp()
        self.assertIn("biz-rtx4080-2", self.searched("failed", 24 * 60 + 1))
        self.setUp()
        self.x.cfg["retry_failed_h"] = 48
        self.assertNotIn("biz-rtx4080-2", self.searched("failed", 24 * 60 + 1))

    def test_an_old_failure_doesnt_stop_a_paying_class_getting_a_second_group(self):
        old = self.group(cls="RTX 4080 (16 GB)", pri="low")
        old["ended"], old["end_kind"] = self.clock[0], "failed"
        self.clock[0] += 25 * 3600
        g = self.paying_4080()
        self.mine(g, "m1", 199, 6)
        self.x.search()
        self.assertIn("biz-rtx4080-3", [c[2]["name"] for c in self.http.made("POST", "/containers$")])

    def test_a_recent_failure_still_stops_a_second_group(self):
        old = self.group(cls="RTX 4080 (16 GB)", pri="low")
        old["ended"], old["end_kind"] = self.clock[0], "failed"
        self.clock[0] += 3600
        g = self.paying_4080()
        self.mine(g, "m1", 199, 6)
        self.x.search()
        self.assertNotIn("biz-rtx4080-3", [c[2]["name"] for c in self.http.made("POST", "/containers$")])

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

    def dear(self, high=1.0, medium=0.19):
        """The 4080 at a high price that clears no margin, and at `medium`."""
        self.x.st["classes"] = dict(CLASSES, **{"RTX 4080 (16 GB)": {"id": "c4080", "prices": {
            "batch": 0.11, "low": 0.15, "medium": medium, "high": high}}})

    def test_a_start_priority_that_no_longer_clears_the_margin_falls_back_to_a_cheaper_one(self):
        self.dear()
        self.assertEqual(self.start_after([("high", True), ("high", False)]), "medium")  # the dearest that clears it
        self.setUp()
        self.dear(medium=1.0)
        self.assertEqual(self.start_after([("high", True), ("high", False)]), "low")

    def test_the_fallback_needs_a_gpu_free(self):
        self.dear()
        self.http.on("POST", "/availability/", lambda b: (200, {"available_gpu_high": 5, "available_gpu_low": 5}))
        self.x.cfg.update(max_active=10, max_new_per_hour=10, retry_after_min=10)
        self.http.on("GET", "/quotas", (200, {"container_groups_quotas": {"container_replicas_quota": 10}}))
        for had in (True, False):
            g = self.group(cls="RTX 4080 (16 GB)", pri="high")
            g["ended"], g["end_kind"], g["had_pc"] = self.clock[0], "released", had
            self.clock[0] += 3600
        self.x.search()
        made = [(c[2]["name"], c[2]["container"]["priority"]) for c in self.http.made("POST", "/containers$")][2:]
        self.assertIn(("biz-rtx4080-3", "low"), made)   # none free at medium

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

    def test_each_boxs_work_at_herominers_is_valued_as_it_comes_in(self):
        g = self.group()
        unit = USD_TH_HR / PRICE * 2 ** 32 / 3.6e15                                     # PRL for one unit of work (2^32 hashes)
        row = lambda shares: {"hashes": str(shares * 2 ** 21), "lastShare": g["created"] + 60}
        self.x.hm_count(g, row(1))                                       # one share
        self.assertAlmostEqual(g["hm_gross"], 2 ** 21 * unit)
        self.x.st["mkt"]["prl_th_hr"] *= 2                               # the network pays twice as much now
        self.x.hm_count(g, row(3))                                       # two more shares, at the new pay
        self.assertAlmostEqual(g["hm_gross"], 5 * 2 ** 21 * unit)
        self.x.hm_count(g, row(1))                                       # the pool started the worker's count again
        self.assertAlmostEqual(g["hm_gross"], 7 * 2 ** 21 * unit)
        self.x.hm_count(g, None)                                         # no row: nothing changes
        self.assertAlmostEqual(g["hm_gross"], 7 * 2 ** 21 * unit)

    def test_work_an_earlier_group_of_the_same_name_left_doesnt_count(self):
        g = self.group()
        self.x.hm_count(g, {"hashes": str(5 * 2 ** 21), "lastShare": g["created"] - 600})
        self.assertEqual(g["hm_gross"], 0)
        self.x.hm_count(g, {"hashes": str(6 * 2 ** 21), "lastShare": g["created"] + 60})   # this group's first share
        self.assertAlmostEqual(g["hm_gross"], 2 ** 21 * USD_TH_HR / PRICE * 2 ** 32 / 3.6e15)

    def test_the_report_shows_what_the_pool_credited(self):
        g = self.group()
        g["prl"], g["spend"], g["hm_units"], g["hm_gross"] = 8.0, 6.0, 1.0, 8.0
        self.x.st["pool_tot"] = {"t": T0, "paid": 1.5, "balance": 0.5, "unconfirmed": 4.0, "blocks": 6,
                                 "hashes": 0, "ths_1h": 0}
        self.x.write_report()
        with open(os.path.join(TMP, "report.md")) as f:
            report = f.read()
        # Work worth 8 PRL at the network's rate; in under 10 blocks the pool's credit ratio isn't used yet.
        self.assertIn("| **8.000\\*** |\n", report)  # the All row
        self.assertIn("| ⛏️ PRL | 0 TH/s | 0.500 | 4.000 | 1.500 |", report)  # straight from the pool: no stars
        self.assertNotIn("has credited", report)   # the ratio is in the notes, not the report, since 20 October
        self.assertNotIn("Locked", report)


class TestCreditRatio(Base):
    """Once the pool has credited us in 10 blocks, the table counts on the share of our estimate it credited."""
    def report(self, blocks=20, cfg=None):
        g = self.group()
        g["prl"], g["spend"], g["hm_units"], g["hm_gross"] = 8.0, 6.0, 1.0, 8.0
        self.x.st["pool_tot"] = {"t": T0, "paid": 1.5, "balance": 0.5, "unconfirmed": 4.0, "blocks": blocks,
                                 "hashes": 0, "ths_1h": 0}
        self.x.cfg.update(cfg or {})
        self.x.write_report()
        with open(os.path.join(TMP, "report.md")) as f:
            return f.read()

    def test_the_first_line_starts_with_a_readable_time(self):
        self.assertTrue(self.report().startswith("**3:00 am EST, Fri, Jan 15** · PRL $1.39 "))  # no rule above it

    def test_the_time_is_new_york_time_and_says_edt_in_summer(self):
        t = w.parse_iso("2026-10-17T15:53:00Z")
        self.assertEqual(w.stamp(t), "11:53 am EDT, Sat, Oct 17")

    def test_the_date_is_in_us_style(self):
        self.assertEqual(w.stamp(w.parse_iso("2026-10-20T00:09:00Z")), "8:09 pm EDT, Mon, Oct 19")
        self.assertEqual(w.stamp(w.parse_iso("2026-11-01T15:00:00Z")), "10:00 am EST, Sun, Nov 1")  # no 0 before the day

    def test_the_time_is_on_the_12_hour_clock(self):
        at = lambda s: w.clock(w.parse_iso(s))
        self.assertEqual(at("2026-10-20T00:00:00Z"), "8:00 pm EDT")     # 20:00 in New York
        self.assertEqual(at("2026-10-19T16:05:00Z"), "12:05 pm EDT")    # noon
        self.assertEqual(at("2026-10-20T04:30:00Z"), "12:30 am EDT")    # midnight
        self.assertEqual(at("2027-01-15T14:00:00Z"), "9:00 am EST")

    def test_the_first_line_leaves_out_what_the_all_row_shows(self):
        first = self.report().split("\n")[0]
        # The credit left ($100 less $6.00 spent); no box is billing, so no hours.
        self.assertEqual(first, "**3:00 am EST, Fri, Jan 15** · PRL $1.39 · 0/1 mining · Salad $94.00* left")

    def test_every_slot_has_a_row_and_an_empty_one_says_so(self):
        report = self.report(cfg={"max_active": 10})
        empty = f"| ⬜{w.NB}empty | — | — | — | — |"
        self.assertEqual(report.count(empty + "\n"), 9)              # 1 group, 10 slots
        self.assertIn(" · 0/10 mining · Salad $", report)                    # counted against the slots, not the groups
        rows =[l for l in report.split("\n") if l.startswith("| ")]
        self.assertLess(rows.index(empty), rows.index(next(l for l in rows if l.startswith("| **All**"))))
        self.assertTrue(all(len(shown(l)) <= 90 for l in rows))

    def test_nothing_but_the_tables_under_the_heading(self):
        self.x.st["kx_tot"] = {"t": T0, "total": 1.0, "confirmed": 1.0, "unconfirmed": 0.0, "paid": 0.0}
        self.x.cfg["safetrade"] = {"sold": 1.0, "received": 1.5, "fees": 0.0, "usdt": 1.5}
        report = self.report(cfg={"max_active": 10, "xmr_wallet": "4" + "a" * 94, "xmr_groups": ["*"]})
        lines = report.split("\n")
        self.assertEqual(lines[next(i for i, l in enumerate(lines) if l.startswith("| **All**")) + 1], "")  # All ends it
        self.assertTrue(report.endswith(" |\n"))                     # nothing under the tables
        for gone in ("⛏️ mining ·", "our own count or estimate", "(K) mines Pearl on Kryptex", "is Salad's bill",
                     "has credited", "pays out from", "SafeTrade", "have mined", "credit_offset", "on Salad for",
                     "spent", "Cost per", "Credit left", "costs us"):
            self.assertNotIn(gone, report)

    def test_the_heading_says_how_many_hours_the_credit_lasts(self):
        self.group(pri="low")["has_pc"] = True                      # $0.15/hr against $94.00 of credit
        self.assertIn(" · Salad $94.00* left (627* h)\n", self.report(cfg={"max_active": 2}))

    def test_a_full_account_has_no_empty_rows(self):
        report = self.report()                                       # 1 group, 1 slot
        self.assertNotIn("empty", report)

    def test_salads_quota_caps_the_rows_too(self):
        self.x.st["quota"] = 3
        report = self.report(cfg={"max_active": 10})
        self.assertEqual(report.count(f"| ⬜{w.NB}empty |"), 2)
        self.assertIn(" · 0/3 mining · Salad $", report)

    def test_an_empty_slot_has_a_cell_for_monero_when_it_is_on(self):
        self.x.cfg["xmr_wallet"] = "4abc"
        self.assertIn(f"| ⬜{w.NB}empty | — | — | — | — | — | — |\n", self.report(cfg={"max_active": 2}))

    def test_the_table_counts_on_what_the_pool_credits(self):
        report = self.report()
        # The box's work is worth 8 PRL at the network's rate; the pool credits 75%: 6.
        self.assertIn(f"| ⏳{w.NB}4080{w.NB}batch | — | — | — | 6.000* |\n", report)
        self.assertIn("| **6.000\\*** |\n", report)  # the All row

    def test_the_ratio_has_a_floor(self):
        # 8 × 90% = 7.20 PRL, not 8 × 75%
        self.assertIn("| **7.200\\*** |\n", self.report(cfg={"yield_floor": 0.9}))
        self.assertAlmostEqual(self.x.credit_ratio(), 0.9)

    def test_the_rules_use_config_yield(self):
        self.report()
        self.assertAlmostEqual(self.x.usd(), USD_TH_HR * 0.75)        # "auto": the credit ratio
        self.x.cfg["yield"] = 1.0
        self.assertAlmostEqual(self.x.usd(), USD_TH_HR)               # fixed: the network's rate
        self.x.write_report()
        with open(os.path.join(TMP, "report.md")) as f:
            # the table still counts on the ratio: 6 PRL, not 8
            self.assertIn("| **6.000\\*** |\n", f.read())

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
        self.assertIn(" · 1/2 mining · Salad $", report)  # the starting box counts as not mining yet

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
        # They ended before the pools' counts were kept: 1 PRL each by the ledger's estimate, before 10 blocks.
        self.assertIn(f"| Others{w.NB}×3 | — | — | — | 3.000* |", report)

    def test_estimates_are_starred_and_api_figures_are_not(self):
        a = self.group(pri="medium")
        self.step(a, 1, "m1")
        self.mine(a, "m1", 200, 5)
        self.x.write_report()
        with open(os.path.join(TMP, "report.md")) as f:
            report = f.read()
        # TH/s (the miner's log) and $/hr (Salad's price list) come from an API; the rest is our own count.
        self.assertIn(f"| ⛏️{w.NB}4080{w.NB}med | 200 | 0.19 | +0.04* (+19%*) |", report)
        self.assertRegex(report, r" · Salad \$[\d.]+\* left \(\d+\* h\)\n")

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

    def test_sales_dont_change_the_report(self):
        g = self.group()
        g.update(prl=10.0, spend=8.0, hm_units=1.0, hm_gross=10.0)
        for status, manual in ((dict(self.STATUS), None), (None, {"sold": 5.2564, "received": 7.1, "fees": 0.0142})):
            self.x.st["trader_status"], self.x.cfg["safetrade"] = status, manual
            self.x.write_report()
            with open(os.path.join(TMP, "report.md")) as f:
                report = f.read()
            # The table counts coins earned (10 PRL of work, before 10 blocks), not dollars: no Profit $ since 20 October.
            self.assertIn("| **10.000\\*** |\n", report)
            for gone in ("SafeTrade", "Profit $", "Locked"):
                self.assertNotIn(gone, report)


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
                                                               "lastShare": int(T0) + 400, "hashes": str(2 ** 22)}]}))
        g["created"] = T0 - 3600
        self.clock[0] = T0 + 500
        self.x.round()
        self.x.round()  # the same log lines again are not counted twice
        self.assertAlmostEqual(g["prl"], 398 / 60 * USD_TH_HR / PRICE)
        self.assertAlmostEqual(g["hm_gross"], 2 ** 22 * USD_TH_HR / PRICE * 2 ** 32 / 3.6e15)   # its 2 shares, once
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
        self.assertIn(" · 1/1 mining · Salad $", report)
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
        self.assertIn("| Box | TH/s | kH/s | $/hr | Profit/hr (margin) | PRL | XMR |", report)
        # 200 TH/s × 0.00117 + 2 kH/s × 0.002 = 0.238 an hour, less 0.11 billed. No Pearl counted at a pool yet;
        # 0.0001 XMR from HeroMiners' count of its Monero work, starred.
        self.assertIn(f"| ⛏️{w.NB}4080{w.NB}batch | 200 | 2.0 | 0.11 | +0.13* (+54%*) | 0.000 | 0.000100* |", report)
        self.assertIn(f"| ⏳{w.NB}2060{w.NB}batch | — | — | — | — | 0.000 | 0.000000* |", report)
        self.assertIn("| **All** | **200** | **2.0** | **0.11** |", report)
        self.assertIn("| Pool | Hashrate (1h) | Confirmed | Pending | Paid |", report)
        self.assertIn("| 🪙 XMR HeroMiners | 1.5 kH/s | 0.005000 | 0.000100 | 0.000000 |", report)  # straight from the pool: no stars
        rows =[l for l in report.split("| Pool |")[0].splitlines() if l.startswith("|")]
        self.assertTrue(all(l.count("|") == 8 for l in rows))  # one table, with kH/s and XMR columns
        # each row on one line in a chat window, as shown (bold marks and escapes aren't)
        self.assertTrue(all(len(shown(l)) <= 90 for l in rows), max(rows, key=len))

    def test_without_an_address_the_table_is_unchanged(self):
        self.x.cfg["xmr_wallet"] = ""
        self.group()
        self.x.write_report()
        with open(os.path.join(TMP, "report.md")) as f:
            report = f.read()
        self.assertNotIn("kH/s", report)
        self.assertNotIn("Monero", report)


class TestKryptex(Base):
    """kryptex_groups groups mine Pearl on Kryptex's PPS+ pool."""
    def setUp(self):
        super().setUp()
        self.x.cfg.update(wallet=os.environ["PRL_WALLET"], kryptex_groups=1)
        self.x.st["kx_read_t"] = self.clock[0]   # Kryptex's workers were just read

    def group(self, *a, kx=False, **kw):
        """A new group on HeroMiners, or on Kryptex with kx. (create puts a new group on Kryptex while fewer than
        kryptex_groups are; see test_new_groups_start_on_kryptex_while_fewer_than_the_count_are.)"""
        want = self.x.cfg["kryptex_groups"]
        self.x.cfg["kryptex_groups"] = 99 if kx else 0
        try:
            return super().group(*a, **kw)
        finally:
            self.x.cfg["kryptex_groups"] = want

    def config_file(self, **kw):
        """config.json for a whole round(), which re-reads it: one group on Kryptex, no search."""
        w.save(w.CFG, dict(dict(wallet=os.environ["PRL_WALLET"], kryptex_groups=1, search=False), **kw))

    def report(self):
        self.x.write_report()
        with open(os.path.join(TMP, "report.md")) as f:
            return f.read()

    def mining(self, machine, since_min, ths=199.0, pri="medium"):
        """A HeroMiners group on `machine` for `since_min` minutes, with a reading a minute for the last 6."""
        g = self.group(pri=pri)
        t = self.clock[0]
        g.update(has_pc=True, had_pc=True, machine=machine, pc_since=t - since_min * 60, instance="inst-" + machine,
                 readings=[[t - k * 60, machine, ths, 10 - k, f"1h {60 - k}m"] for k in range(6, 0, -1)])
        return g

    def commands(self):
        return [c[2]["container"]["command"][2] for c in self.http.made("PATCH", "/containers/")]

    def test_the_command_points_the_miner_at_kryptex_with_pool(self):
        cmd = w.cmd_for("biz-rtx4080", "prl1abc", "v0.6.1", kryptex="prl-us.kryptex.network:7048")
        sed = re.search(r"(sed .*?) > /root/box\.sh", cmd).group(1)
        import subprocess
        base = "BASE='{\"core\":\"release\",\"env\":{},\"flags\":\"\"}'\n"
        out = subprocess.run(["sh", "-c", sed], input=base, capture_output=True, text=True).stdout
        self.assertEqual(out, "BASE='{\"core\":\"release\",\"env\":{},\"flags\":\"--pool prl-us.kryptex.network:7048\"}'\n")
        # With Monero too, both flags go in, and the CPU worker is downloaded as before.
        both = w.cmd_for("biz-rtx4080", "prl1abc", "v0.6.1", XMR, 0, "prl-us.kryptex.network:7048")
        sed = re.search(r"(sed .*?) > /root/box\.sh", both).group(1)
        out = subprocess.run(["sh", "-c", sed], input=base, capture_output=True, text=True).stdout
        self.assertIn(f"\"flags\":\"--pool prl-us.kryptex.network:7048 --xmr-address {XMR}\"", out)
        self.assertIn("llmjob-xmr-worker", both)
        # No /etc/hosts redirect or local forward any more (the file is read-only on Salad), and no "; curl".
        for c in (cmd, both):
            self.assertNotIn("socat", c)
            self.assertNotIn("hosts", c)
            self.assertNotIn("; curl", c)

    # box.sh's port check, as it is at the pinned commit.
    PORT_CHECK = ("  ok=0; for h in us us2 ca de; do timeout 5 bash -c \"exec 3<>/dev/tcp/$h.pearl.herominers.com/1200\" "
                  "2>/dev/null && { ok=1; break; }; done\n"
                  "  [ $ok = 1 ] || { echo \"[run] FAIL: outbound port 1200 blocked\"; sleep infinity; }\n")

    def port_check(self, cmd):
        import subprocess
        sed = re.search(r"(sed .*?) > /root/box\.sh", cmd).group(1)
        return subprocess.run(["sh", "-c", sed], input=self.PORT_CHECK, capture_output=True, text=True).stdout

    def test_a_kryptex_box_checks_kryptexs_port_and_others_check_1200_as_before(self):
        out = self.port_check(w.cmd_for("biz-rtx4080", "prl1abc", "v0.6.1", XMR, 0, "prl-us.kryptex.network:7048"))
        self.assertIn("for h in prl-us.kryptex.network; do timeout 5 bash -c \"exec 3<>/dev/tcp/$h/7048\" 2>/dev/null", out)
        self.assertIn("echo \"[run] FAIL: outbound port 7048 blocked\"", out)
        self.assertNotIn("1200", out)
        self.assertEqual(self.port_check(w.cmd_for("biz-rtx4080", "prl1abc", "v0.6.0", XMR)), self.PORT_CHECK)

    def test_the_round_reads_which_port_a_box_found_blocked(self):
        self.config_file()
        g = self.group(kx=True)
        self.http.on("GET", "/containers/biz-rtx4080", (200, {"current_state": {"status": "running"}}))
        self.http.on("GET", "/instances", (200, {"instances": [{"id": "i1", "machine_id": "m1", "state": "running"}]}))
        items = [{"text_log": "[run] FAIL: outbound port 7048 blocked", "time": "2027-01-15T08:01:00Z",
                  "resource": {"labels": {"container_group_name": "biz-rtx4080", "machine_id": "m1"}}}]
        self.http.on("POST", "/log-entries", (200, {"items": items}))
        self.clock[0] += 120
        self.x.round()
        self.assertEqual(g["blocked_port"], {"m1": 7048})
        self.assertIn("PORT BLOCKED biz-rtx4080 on PC m1: [run] FAIL: outbound port 7048 blocked", self.events())

    def blocked_on_kryptex(self, works=True):
        """A Kryptex box on m1 that found Kryptex's port blocked; `works`: another box's work reached Kryptex lately."""
        if works:
            self.group(kx=True)["kx_share"] = self.clock[0]
        g = self.group(kx=True)
        self.step(g, 1, "m1")
        g["blocked"].append("m1")
        g["blocked_port"] = {"m1": 7048}
        self.step(g, 1, "m1")
        return g

    def test_a_kryptex_box_leaves_a_pc_blocking_kryptexs_port_and_lists_it_for_kryptex_boxes_only(self):
        g = self.blocked_on_kryptex()
        self.assertEqual(g["reallocs"], 1)
        self.assertTrue(g["kx"])                                          # a PC problem: it stays on Kryptex
        self.assertEqual(self.x.st["bad_pcs"]["m1"]["ports"], [7048])
        hm = self.group()                                                 # a HeroMiners box checks 1200: m1 may suit it
        self.step(hm, 1, "m1")
        self.assertEqual(hm["machine"], "m1")
        kb = self.group(kx=True)                                          # another Kryptex box doesn't take it
        self.step(kb, 1, "m1")
        self.assertIsNone(kb["machine"])
        self.assertEqual(kb["reallocs"], 0)

    def test_a_pc_that_blocks_both_ports_stays_listed_for_both(self):
        # On 19 October e8c084ab blocked 1200 for a HeroMiners box, then 7048 for a Kryptex box.
        self.x.st["bad_pcs"] = {"m1": {"t": self.clock[0], "group": "biz-rtx3060", "why": "PC blocks outbound port 1200",
                                       "ports": [1200]}}
        self.blocked_on_kryptex()
        self.assertEqual(self.x.st["bad_pcs"]["m1"]["ports"], [1200, 7048])
        self.assertEqual(self.x.st["bad_pcs"]["m1"]["why"], "PC blocks outbound ports 1200 and 7048")
        for kx in (False, True):
            g = self.group(kx=kx)
            self.step(g, 1, "m1")
            self.assertIsNone(g["machine"])                               # neither pool's boxes take it

    def test_a_pc_bad_for_every_box_isnt_narrowed_to_a_port(self):
        self.x.st["bad_pcs"] = {"m1": {"t": self.clock[0], "group": "biz-rtx3060", "why": "no mining reading"}}
        g = self.group(kx=True)
        self.step(g, 1, "m1")                                             # moved off at once, as before
        self.assertIsNone(g["machine"])
        # A box that still runs on it (the list was written after it got the PC) and finds a port blocked
        # leaves the entry as it was.
        h = self.group()
        h.update(has_pc=True, machine="m1", pc_since=self.clock[0], instance="inst-m1", blocked=["m1"],
                 blocked_port={"m1": 1200})
        self.step(h, 1, "m1")
        self.assertEqual(self.x.st["bad_pcs"]["m1"]["why"], "no mining reading")
        self.assertIsNone(w.W.bad_ports(self.x.st["bad_pcs"]["m1"]))

    def test_a_kryptex_box_keeps_a_pc_that_blocks_only_1200(self):
        self.x.st["bad_pcs"] = {"m1": {"t": self.clock[0], "group": "biz-rtx3060", "why": "PC blocks outbound port 1200",
                                       "port": 1200}}
        kb = self.group(kx=True)
        self.step(kb, 1, "m1")
        self.assertEqual(kb["machine"], "m1")
        hm = self.group()
        self.step(hm, 1, "m1")                                            # a HeroMiners box still leaves it
        self.assertIsNone(hm["machine"])

    def test_kryptexs_port_unreachable_with_no_proof_kryptex_works_is_a_kryptex_failure(self):
        self.http.on("PATCH", "/containers/", (200, {}))
        self.http.on("POST", "/reallocate", (202, {}))
        g = self.blocked_on_kryptex(works=False)
        self.assertFalse(g["kx"])                                         # back to HeroMiners, as a box with no shares
        self.assertEqual(self.x.st["kx_fails"], 1)
        self.assertNotIn("m1", self.x.st.get("bad_pcs", {}))              # the PC isn't blamed
        self.assertNotIn("--pool", w.cmd_for("biz-rtx4080", "prl1abc", "v0.6.0"))
        self.assertNotIn("--pool", w.cmd_for("biz-rtx4080", "prl1abc", "v0.6.0", XMR))

    def test_a_kryptex_box_gets_at_least_the_release_with_pool(self):
        self.x.cfg.update(box_ver="v0.5.13", xmr_ver="v0.6.0", xmr_wallet=XMR, xmr_groups=["*"])
        cmd = self.x.box("biz-rtx4080", True, True)[1][2]
        self.assertIn(f"--pool prl-us.kryptex.network:7048 --xmr-address {XMR}", cmd)
        self.assertIn("VER=v0.6.1;", cmd)
        self.assertNotIn("--pool", self.x.box("biz-rtx4080", True, False)[1][2])
        self.assertEqual(self.x.box("biz-rtx4080", False, True)[0], "v0.6.1")
        self.assertEqual(self.x.box("biz-rtx4080", True, True)[0], "v0.6.1")
        self.assertEqual(self.x.box("biz-rtx4080", True, False)[0], "v0.6.0")
        self.assertEqual(self.x.box("biz-rtx4080", False, False)[0], "v0.5.13")
        self.assertIn("VER=v0.6.1;", self.x.box("biz-rtx4080", False, True)[1][2])
        self.x.cfg["box_ver"] = "v0.6.2"
        self.assertEqual(self.x.box("biz-rtx4080", True, True)[0], "v0.6.2")   # never lowered

    def test_turning_monero_on_or_off_keeps_a_kryptex_box_on_kryptex(self):
        g = self.mining("m1", 300)
        g.update(kx=True, ver="v0.6.1", xmr=False)
        self.x.cfg.update(xmr_wallet=XMR, xmr_groups=["*"])
        self.x.xmr_sync([g], self.clock[0])
        cmd = self.commands()[-1]
        self.assertIn(f"--pool prl-us.kryptex.network:7048 --xmr-address {XMR}", cmd)
        self.assertIn("VER=v0.6.1;", cmd)
        self.assertEqual(g["ver"], "v0.6.1")
        self.x.cfg.update(xmr_wallet="", xmr_groups=[])
        g["xmr_try"] = 0
        self.clock[0] += 3600
        self.x.xmr_sync([g], self.clock[0])
        cmd = self.commands()[-1]
        self.assertIn("--pool prl-us.kryptex.network:7048", cmd)
        self.assertNotIn("--xmr-address", cmd)

    def test_new_groups_start_on_kryptex_while_fewer_than_the_count_are(self):
        self.x.cfg["kryptex_groups"] = 2
        a, b, c = (Base.group(self) for _ in range(3))
        self.assertEqual([g["kx"] for g in (a, b, c)], [True, True, False])
        self.assertEqual((a["kx_since"], c["kx_since"]), (a["created"], None))
        posts = [p[2]["container"]["command"][2] for p in self.http.made("POST", "/containers$")]
        self.assertEqual(["--pool" in p for p in posts], [True, True, False])
        self.assertEqual(self.events().count(", Pearl on Kryptex"), 2)   # in their CREATED lines
        a["ended"] = self.clock[0]                          # a is deleted: the next new group takes its place
        self.assertTrue(Base.group(self)["kx"])
        self.x.st["kx_fails"] = 2                           # the test stopped: none until kryptex_token changes
        b["ended"] = self.clock[0]
        self.assertFalse(Base.group(self)["kx"])

    def test_a_box_that_is_mining_is_never_moved_to_kryptex(self):
        a, b = self.mining("m1", 300, ths=150.0), self.mining("m2", 30)
        for _ in range(3):
            self.x.kx_sync([a, b], self.clock[0])
            self.clock[0] += 16 * 60
        self.assertFalse(a["kx"] or b["kx"])
        self.assertFalse(self.http.made("PATCH", "/containers/"))
        self.assertFalse(self.http.made("POST", "/reallocate$"))

    def test_a_group_waiting_for_a_pc_is_switched_by_its_command_the_cheapest_first(self):
        box, dear, cheap = self.mining("m1", 300), self.group(), self.group()
        dear["price"], cheap["price"] = 0.20, 0.10
        self.x.kx_sync([box, dear, cheap], self.clock[0])
        self.assertTrue(cheap["kx"])
        self.assertFalse(dear["kx"] or box["kx"])
        patch = self.http.made("PATCH", "/containers/biz-rtx4080-3$")
        self.assertEqual(len(patch), 1)
        self.assertEqual(list(patch[0][2]["container"]), ["command"])  # resources stay as they are
        self.assertIn("--pool", patch[0][2]["container"]["command"][2])
        self.assertEqual(len(self.http.made("PATCH", "/containers/")), 1)
        self.assertFalse(self.http.made("POST", "/reallocate$"))      # no PC to leave
        self.assertEqual(cheap["kx_since"], self.clock[0])
        self.assertIn("KRYPTEX ON biz-rtx4080-3 (waiting for a PC): Pearl on Kryptex; it starts that way on its "
                      "next PC", self.events())

    def test_only_as_many_waiting_groups_as_are_missing_are_switched(self):
        self.x.cfg["kryptex_groups"] = 5
        kx = [self.group(kx=True) for _ in range(4)]
        waiting = [self.group() for _ in range(3)]
        for i, g in enumerate(waiting):
            g["price"] = 0.10 + i / 100
        self.x.kx_sync(kx + waiting, self.clock[0])
        self.assertEqual([g["kx"] for g in waiting], [True, False, False])   # 4 + 1: the cheapest
        self.assertEqual(len(self.http.made("PATCH", "/containers/")), 1)

    def test_with_none_on_kryptex_the_cheapest_waiting_groups_fill_every_slot_at_once(self):
        self.x.cfg["kryptex_groups"] = 5
        waiting = [self.group() for _ in range(7)]
        for i, g in enumerate(waiting):
            g["price"] = 0.20 - i / 100                   # the last five are the cheapest
        self.x.kx_sync(waiting, self.clock[0])
        self.assertEqual([g["kx"] for g in waiting], [False, False, True, True, True, True, True])
        self.clock[0] += 60
        self.x.kx_sync(waiting, self.clock[0])
        self.assertEqual(len(self.http.made("PATCH", "/containers/")), 5)   # none the round after

    def test_a_group_still_preparing_its_image_is_not_switched(self):
        self.x.cfg["kryptex_groups"] = 4
        gs = [self.group(status=s) for s in ("pending", "failed", "deploying", "stopped")]
        self.x.kx_sync(gs, self.clock[0])
        # Salad refuses an update while the image is prepared; a deploying group is waiting for a PC (rule 2).
        self.assertEqual([g["kx"] for g in gs], [False, False, True, True])

    def test_a_refused_switch_back_by_command_is_tried_again_in_15_minutes(self):
        g = self.group(kx=True)
        g["kx_since"] = self.clock[0] - 3600
        self.x.cfg["kryptex_groups"] = 0
        self.http.on("PATCH", "/containers/", (502, {}))
        for _ in range(15):
            self.x.kx_sync([g], self.clock[0])
            self.clock[0] += 60
        self.assertEqual(len(self.http.made("PATCH", "/containers/")), 1)
        self.x.kx_sync([g], self.clock[0])
        self.assertEqual(len(self.http.made("PATCH", "/containers/")), 2)

    def test_a_refused_update_is_tried_again_in_15_minutes_and_the_next_group_takes_the_slot(self):
        cheap, dear = self.group(), self.group()
        cheap["price"], dear["price"] = 0.05, 0.20
        self.http.on("PATCH", "/containers/biz-rtx4080$", (502, {}))
        self.x.kx_sync([cheap, dear], self.clock[0])
        self.assertFalse(cheap["kx"])
        self.assertTrue(dear["kx"])                       # the slot is filled this round
        self.x.cfg["kryptex_groups"] = 2
        for _ in range(14):
            self.clock[0] += 60
            self.x.kx_sync([cheap, dear], self.clock[0])
        self.assertEqual(len(self.http.made("PATCH", "/containers/biz-rtx4080$")), 1)
        self.assertEqual(self.events().count("WARN Kryptex on for biz-rtx4080 refused"), 1)
        self.clock[0] += 60
        self.x.kx_sync([cheap, dear], self.clock[0])
        self.assertEqual(len(self.http.made("PATCH", "/containers/biz-rtx4080$")), 2)

    def test_a_waiting_group_whose_read_failed_is_not_switched(self):
        g = self.group()
        g["_skip"] = True                                 # it may have a PC by now
        self.x.kx_sync([g], self.clock[0])
        self.assertFalse(g["kx"])
        self.assertFalse(self.http.made("PATCH", "/containers/"))

    def test_a_group_whose_state_salad_did_not_answer_is_left_alone_that_round(self):
        self.config_file()
        box, w = self.mining("m1", 300), self.group()
        box.update(machine="m1", pc_since=self.clock[0] - 300 * 60)
        self.http.on("GET", "/containers/biz-rtx4080(-2)?$", (502, {}))
        self.clock[0] += 60
        self.x.round()                                    # no rule runs on last round's state
        self.assertEqual((box["machine"], box["has_pc"], box["_skip"]), ("m1", True, True))
        self.assertFalse(w["kx"])                         # nor a switch: w may have got a PC
        self.assertFalse(self.http.made("PATCH", "/containers/"))
        self.assertNotIn("NEW PC", self.events())

    def test_a_late_line_from_the_pc_a_group_just_left_counts_to_its_old_pool(self):
        g = self.mining("m0", 300)
        self.x.reallocate(g, "slow")                      # it left m0 and waits for a PC
        self.x.kx_sync([g], self.clock[0])
        self.assertTrue(g["kx"])
        self.assertEqual((g["sw_from"], g["sw_prev"]), ("m0", False))

    def test_a_kryptex_group_keeps_kryptex_when_it_loses_its_pc(self):
        k, box = self.mining("m1", 300), self.mining("m2", 300, ths=150.0)
        k.update(kx=True, kx_since=self.clock[0] - 3600)
        k.update(has_pc=False, machine=None, pc_since=None)     # Salad took its PC back
        for _ in range(4):
            self.clock[0] += 15 * 60
            self.x.kx_sync([k, box], self.clock[0])
        self.assertTrue(k["kx"])                                 # it starts on Kryptex at its next PC
        self.assertFalse(box["kx"])                              # and the box that is mining is left alone
        self.assertFalse(self.http.made("PATCH", "/containers/"))
        self.assertFalse(self.http.made("POST", "/reallocate$"))

    def test_lowering_the_count_switches_groups_back_once_they_have_no_pc(self):
        k, w = self.mining("m1", 300), self.group()
        for g in (k, w):
            g.update(kx=True, kx_since=self.clock[0] - 3600)
        self.x.cfg["kryptex_groups"] = 0
        self.x.kx_sync([k, w], self.clock[0])
        self.assertFalse(w["kx"])                       # waiting for a PC: its command is updated
        self.assertTrue(k["kx"])                        # mining: left on its PC
        self.assertIn("KRYPTEX OFF biz-rtx4080-2 (more Kryptex groups than config asks): Pearl on HeroMiners; "
                      "it starts that way on its next PC", self.events())
        k.update(has_pc=False, machine=None)            # Salad takes its PC back
        self.clock[0] += 60
        self.x.kx_sync([k, w], self.clock[0])
        self.assertFalse(k["kx"])
        self.assertNotIn("--pool", self.commands()[-1])
        self.assertFalse(self.http.made("POST", "/reallocate$"))

    def test_a_kryptex_box_with_no_shares_goes_back_and_its_pc_stays_off_the_bad_list(self):
        g = self.mining("m1", 300)
        g.update(kx=True, kx_since=self.clock[0] - 3600)
        self.x.leave(g, "no share for 25 min on PC m1")
        self.assertFalse(g["kx"])
        self.assertFalse(g["ended"])
        self.assertNotIn("m1", self.x.st.get("bad_pcs", {}))
        self.assertEqual(g["reallocs"], 0)   # not counted toward its 3
        self.assertNotIn("--pool", self.commands()[-1])
        self.assertIn("KRYPTEX OFF biz-rtx4080 (on Kryptex: no share for 25 min on PC m1)", self.events())
        self.assertEqual(self.x.st["kx_fails"], 1)
        # A HeroMiners box leaves its PC as before.
        g.update(has_pc=True, machine="m3", pc_since=self.clock[0], instance="inst-m3")
        self.x.leave(g, "no share for 25 min on PC m3")
        self.assertIn("m3", self.x.st["bad_pcs"])

    def test_two_failures_stop_the_test_until_the_token_changes(self):
        self.x.cfg["kryptex_groups"] = 3
        a, b, c = self.mining("m1", 300), self.mining("m2", 200), self.mining("m3", 100)
        for g in (a, b, c):
            g.update(kx=True, kx_since=self.clock[0] - 3600)
        for g in (a, b):
            self.x.leave(g, "no mining reading 12 min after getting PC")
        self.assertIn("WARN Kryptex test stopped: 2 Kryptex boxes mined no shares", self.events())
        # c is still on Kryptex with a PC, and Kryptex isn't taking the work: it is moved at once.
        self.x.kx_sync([a, b, c], self.clock[0])
        self.assertFalse(c["kx"])
        self.assertTrue(self.http.made("POST", "/instances/inst-m3/reallocate"))
        self.assertIn("KRYPTEX OFF biz-rtx4080-3 (the Kryptex test stopped after 2 boxes mined no shares there)",
                      self.events())
        # While the test is stopped, groups waiting for a PC and new groups stay on HeroMiners.
        self.clock[0] += 16 * 60
        d = Base.group(self)
        self.x.kx_sync([a, b, c, d], self.clock[0])
        self.assertFalse(a["kx"] or b["kx"] or c["kx"] or d["kx"])
        self.x.cfg["kryptex_token"] = 1
        self.assertTrue(Base.group(self)["kx"])

    def test_a_refused_move_leaves_the_box_as_it_was(self):
        g = self.mining("m1", 300)
        g.update(kx=True, kx_since=self.clock[0] - 3600)
        self.x.st["kx_fails"] = 2                       # the test stopped: a Kryptex box with a PC is moved
        self.http.on("POST", "/reallocate$", (500, {}))
        self.x.kx_sync([g], self.clock[0])
        self.assertTrue(g["kx"])
        self.assertEqual((g["machine"], g["has_pc"]), ("m1", True))
        cmds = self.commands()
        self.assertEqual(len(cmds), 2)                  # the HeroMiners command, then the Kryptex one put back
        self.assertNotIn("--pool", cmds[0])
        self.assertIn("--pool", cmds[1])
        self.assertIn("WARN Kryptex off for biz-rtx4080: Salad refused to move it to a new PC", self.events())
        self.assertNotIn("KRYPTEX OFF", self.events())

    def test_an_instance_salad_no_longer_has_counts_as_moved(self):
        g = self.mining("m1", 300)
        g.update(kx=True, kx_since=self.clock[0] - 3600)
        self.x.st["kx_fails"] = 2                       # the test stopped: a box with a PC is moved
        self.http.on("POST", "/reallocate$", (404, {"type": "container_group_instance_not_found"}))
        self.x.kx_sync([g], self.clock[0])
        self.assertFalse(g["kx"])                         # the switch stands: the next instance starts on HeroMiners
        self.assertEqual(len(self.commands()), 1)        # nothing put back
        self.assertIn("KRYPTEX OFF biz-rtx4080", self.events())

    def test_a_failed_switch_back_is_no_strike_and_the_pc_is_left_as_usual(self):
        g = self.mining("m1", 300)
        g.update(kx=True, kx_since=self.clock[0] - 3600)
        self.http.on("PATCH", "/containers/", (503, {}))
        self.x.leave(g, "no share for 25 min on PC m1")
        self.assertEqual(self.x.st.get("kx_fails", 0), 0)
        self.assertIn("m1", self.x.st["bad_pcs"])      # nothing mined there: the PC goes on the list as for any box

    def test_a_group_that_failed_on_kryptex_is_not_switched_again(self):
        a = self.mining("m1", 300)
        a.update(kx=True, kx_since=self.clock[0] - 3600, price=0.01)
        self.x.leave(a, "no share for 25 min on PC m1")
        self.assertFalse(a["kx"])
        self.assertFalse(a["has_pc"])                    # moved, and waiting for its new PC
        b = self.group()
        self.x.kx_sync([a, b], self.clock[0] + 16 * 60)
        self.assertFalse(a["kx"])                        # cheaper, but it failed on Kryptex: b is taken instead
        self.assertTrue(b["kx"])

    def test_a_kryptex_box_that_sends_nothing_while_others_reach_kryptex_has_a_pc_problem(self):
        self.x.cfg["kryptex_groups"] = 2
        g, other = self.mining("m1", 300), self.mining("m2", 300)
        for x in (g, other):
            x.update(kx=True, kx_since=self.clock[0] - 3600)
        other["kx_share"] = self.clock[0] - 600          # Kryptex took other's work 10 min ago
        self.x.leave(g, "no mining reading 12 min after getting PC m1")
        self.assertTrue(g["kx"])                         # still on Kryptex, moved off a bad PC
        self.assertEqual(self.x.st.get("kx_fails", 0), 0)
        self.assertIn("m1", self.x.st["bad_pcs"])
        other["kx_share"] = self.clock[0] - 31 * 60      # nothing reached Kryptex for 30 min: a Kryptex failure
        g.update(has_pc=True, machine="m3", pc_since=self.clock[0] - 720, instance="inst-m3")
        self.x.leave(g, "no mining reading 12 min after getting PC m3")
        self.assertFalse(g["kx"])
        self.assertEqual(self.x.st["kx_fails"], 1)
        self.assertNotIn("m3", self.x.st["bad_pcs"])

    def test_a_kryptex_box_whose_read_failed_keeps_its_slot(self):
        k, h = self.mining("m1", 300), self.group()
        k.update(kx=True, kx_since=self.clock[0] - 3600, _skip=True)   # its instances read failed this round
        self.x.kx_sync([k, h], self.clock[0])
        self.assertTrue(k["kx"])
        self.assertFalse(h["kx"])
        self.assertFalse(self.http.made("PATCH", "/containers/"))

    def test_a_kryptex_box_that_had_shares_and_stops_is_a_pc_problem(self):
        g = self.mining("m1", 300)
        g.update(kx=True, kx_since=self.clock[0] - 3600, kx_share=self.clock[0] - 1800)   # Kryptex took its work
        self.x.leave(g, "no mining reading for 7 min on PC m1")
        self.assertTrue(g["kx"])                         # still a Kryptex box, moved off a bad PC
        self.assertEqual(self.x.st.get("kx_fails", 0), 0)
        self.assertIn("m1", self.x.st["bad_pcs"])

    def test_a_late_herominers_share_is_no_proof_kryptex_works(self):
        self.x.cfg["search"] = False
        g = self.group()
        g.update(kx=True, kx_since=T0 + 300, sw_t=T0 + 300, sw_from="m0", sw_prev=False, created=T0 - 3600,
                 acc={"m0": 10})
        self.http.on("GET", "/containers/biz-rtx4080", (200, {"current_state": {"status": "running"}}))
        self.http.on("GET", "/instances", (200, {"instances": [{"id": "i1", "machine_id": "m1", "state": "running"}]}))
        # The PC it left sends one more HeroMiners share after the switch.
        items = [{"text_log": "[mining g0] 08:06:00 198.0 TH/s · 11 accepted · 0 rejected · up 4h 3m",
                  "time": "2027-01-15T08:06:00Z",
                  "resource": {"labels": {"container_group_name": "biz-rtx4080", "machine_id": "m0"}}}]
        self.http.on("POST", "/log-entries", (200, {"items": items}))
        self.clock[0] = T0 + 500
        self.x.round()
        self.assertGreater(g["last_share"], g["kx_since"])
        self.assertIsNone(g.get("kx_share"))
        g.update(has_pc=True, machine="m1", pc_since=self.clock[0], instance="inst-m1")
        self.x.leave(g, "no mining reading 10 min after getting PC m1")
        self.assertFalse(g["kx"])                        # a Kryptex failure: back to HeroMiners
        self.assertEqual(self.x.st["kx_fails"], 1)
        self.assertNotIn("m1", self.x.st.get("bad_pcs", {}))

    def test_while_kryptex_cannot_be_read_the_miners_own_counts_stand_in(self):
        self.x.cfg["kryptex_groups"] = 2
        g, other = self.mining("m1", 300), self.mining("m2", 300)
        for x in (g, other):
            x.update(kx=True, kx_since=self.clock[0] - 3600)
        self.x.st["kx_read_t"] = self.clock[0] - 31 * 60   # Kryptex's API unread for 31 min,
        # but other's miner counts accepted shares on Kryptex (mining() gives it a rising count).
        self.x.leave(g, "no mining reading 12 min after getting PC m1")
        self.assertTrue(g["kx"])                             # Kryptex's pool works: a PC problem
        self.assertEqual(self.x.st.get("kx_fails", 0), 0)
        self.assertIn("m1", self.x.st["bad_pcs"])

    def test_an_outage_of_kryptex_s_api_and_pool_is_still_a_kryptex_failure(self):
        self.x.cfg["kryptex_groups"] = 2
        g, other = self.mining("m1", 300), self.mining("m2", 300)
        for x in (g, other):
            x.update(kx=True, kx_since=self.clock[0] - 3600)
        self.x.st["kx_read_t"] = self.clock[0] - 31 * 60
        other["kx_share"] = self.clock[0] - 35 * 60           # Kryptex's count, from before its API went down
        for r in g["readings"] + other["readings"]:
            r[3] = 0                                         # and no miner has counted an accepted share
        self.x.leave(g, "no mining reading 12 min after getting PC m1")
        self.assertFalse(g["kx"])                            # no box's miner counts a share there either
        self.assertEqual(self.x.st["kx_fails"], 1)
        self.assertNotIn("m1", self.x.st.get("bad_pcs", {}))

    def test_a_lone_kryptex_box_s_own_log_shares_stand_in_while_kryptex_cannot_be_read(self):
        g = self.mining("m1", 60)                         # a rising accepted count on m1, an hour on it
        g.update(kx=True, kx_since=self.clock[0] - 3000)
        self.x.st["kx_read_t"] = self.clock[0] - 40 * 60
        self.x.leave(g, "no mining reading for 7 min on PC m1")
        self.assertTrue(g["kx"])                          # it mined on Kryptex: a PC problem
        self.assertEqual(self.x.st.get("kx_fails", 0), 0)
        self.assertIn("m1", self.x.st["bad_pcs"])

    def test_log_shares_count_from_the_moment_kryptex_s_api_stops_answering(self):
        self.x.cfg["kryptex_groups"] = 2
        g, other = self.mining("m1", 300), self.mining("m2", 300)
        for x in (g, other):
            x.update(kx=True, kx_since=self.clock[0] - 3600)
        for r in g["readings"]:
            r[3] = 5
        other["kx_share"] = self.clock[0] - 35 * 60
        self.x.st["kx_read_t"] = self.clock[0]            # read this round: Kryptex's own count decides
        self.assertFalse(self.x.kx_works(g, self.clock[0]))
        self.x.st["kx_read_t"] = self.clock[0] - 3 * 60   # no answer for 3 min: other's log shares since then count
        self.assertTrue(self.x.kx_works(g, self.clock[0]))
        self.x.st["kx_read_t"] = self.clock[0] - 90       # one missed round isn't an outage
        self.assertFalse(self.x.kx_works(g, self.clock[0]))
        self.x.st["kx_read_t"] = self.clock[0] - 3 * 60
        for r in other["readings"]:
            r[0] -= 10 * 60                               # its log shares are from before Kryptex's last answer,
        self.assertFalse(self.x.kx_works(g, self.clock[0]))   # which had no share from it

    def test_the_miner_s_own_count_is_proof_only_on_a_pc_mining_on_kryptex(self):
        g = self.mining("m1", 60)
        g.update(kx=True, kx_since=self.clock[0] - 7200)
        last = g["readings"][-1][0]
        self.assertEqual(self.x.kx_log_proof(g), last)
        g["pc_pool"] = {"m1": False}                      # seen mining on HeroMiners there
        self.assertEqual(self.x.kx_log_proof(g), 0)
        g["pc_pool"] = {}
        g["pc_since"] = self.clock[0] - 4 * 60            # its first 5 minutes there: not yet
        self.assertEqual(self.x.kx_log_proof(g), 0)
        g["machine"] = "m9"                               # once it has left m1, m1's shares count again,
        self.assertEqual(self.x.kx_log_proof(g), last)    # from 5 minutes after its first reading there
        g["readings"] = [r for r in g["readings"] if r[0] >= last - 4 * 60]
        self.assertEqual(self.x.kx_log_proof(g), 0)       # left within 5 minutes: never seen, so no proof
        g["pc_pool"] = {"m1": True}                       # unless seen mining on Kryptex
        self.assertEqual(self.x.kx_log_proof(g), last)

    def test_a_box_that_just_switched_shows_no_hashrate_until_its_new_pool_has_a_row(self):
        self.config_file()
        k, h = self.two_on_pcs()
        k["pool_ths"] = 160.0                             # from HeroMiners, before its switch
        k["readings"] = [[self.clock[0] - n * 60, "m1", 160.0, 10 - n, f"1h {60 - n}m"] for n in range(5, 0, -1)]
        k["last_share"] = self.clock[0] - 60              # mining on Kryptex since its switch
        self.http.on("GET", "herominers", (200, {"workers": [{"name": "biz-rtx4080-2-g0", "hashrate": 33000,
                                                               "lastShare": int(self.clock[0]) - 30}]}))
        self.x.round()                                    # Kryptex has no row for it yet
        self.assertIsNone(k["pool_ths"])
        self.assertAlmostEqual(h["pool_ths"], 33000 * w.UNIT_TH)
        self.x.st["kx_tot"] = {"t": T0, "total": 0.5, "confirmed": 0.3, "unconfirmed": 0.2, "paid": 0.0}
        self.assertIn(f"| ⛏️ PRL{w.NB}Kryptex | 0 TH/s |", self.report())

    def test_the_kryptex_row_sums_the_boxes_that_mine_there(self):
        k, h = self.group(kx=True), self.group()
        k.update(has_pc=True, machine="m1", pc_pool={"m1": False}, pool_ths=199.0)   # its box mines on HeroMiners
        h.update(has_pc=True, machine="m2", pc_pool={"m2": True}, pool_ths=150.0)    # its box mines on Kryptex
        self.x.st["pool_tot"] = {"t": T0, "paid": 6.0, "balance": 1.0, "unconfirmed": 1.0, "blocks": 20,
                                 "hashes": 0, "ths_1h": 199}
        self.x.st["kx_tot"] = {"t": T0, "total": 0.5, "confirmed": 0.3, "unconfirmed": 0.2, "paid": 0.0}
        self.assertIn(f"| ⛏️ PRL{w.NB}Kryptex | 150 TH/s |", self.report())

    def test_kryptex_s_workers_read_is_dated_only_when_it_works(self):
        self.config_file()
        g = self.group(kx=True)
        self.http.on("GET", "/containers/biz-rtx4080$", (200, {"current_state": {"status": "running"}}))
        self.http.on("GET", "/instances", (200, {"instances": []}))
        self.x.st.pop("kx_read_t")
        self.x.round()
        self.assertEqual(self.x.st["kx_read_t"], self.clock[0])
        t0 = self.clock[0]
        self.http.on("GET", "v3/miner/workers", (503, {}))
        self.clock[0] += 60
        self.x.round()
        self.assertEqual(self.x.st["kx_read_t"], t0)

    def test_a_kryptex_box_seen_mining_on_herominers_at_its_pc_counts_there(self):
        g = self.mining("m1", 20)
        g.update(kx=True, kx_since=self.clock[0] - 25 * 60, sw_t=self.clock[0] - 25 * 60)
        t = self.clock[0]
        self.x.pc_pool(g, None, {"lastShare": g["pc_since"] + 4 * 60})   # within 5 min of the PC: not yet
        self.assertNotIn("pc_pool", g)
        self.x.pc_pool(g, None, {"lastShare": t - 60})
        self.assertEqual(g["pc_pool"], {"m1": False})
        self.assertIn("WARN biz-rtx4080 mines Pearl on HeroMiners on PC m1, though its command is for Kryptex: it "
                      "got the PC as its command changed. This PC's readings count to HeroMiners; its next PC starts "
                      "on Kryptex", self.events())
        self.x.pc_pool(g, None, {"lastShare": t})
        self.assertEqual(self.events().count("mines Pearl on HeroMiners on PC m1"), 1)   # said once
        # Its PC failing is no Kryptex failure: the box wasn't on Kryptex.
        self.x.leave(g, "no mining reading for 7 min on PC m1")
        self.assertTrue(g["kx"])
        self.assertEqual(self.x.st.get("kx_fails", 0), 0)
        self.assertIn("m1", self.x.st["bad_pcs"])

    def test_a_herominers_box_seen_mining_on_kryptex_at_its_pc_counts_there(self):
        g = self.mining("m1", 20)
        self.x.pc_pool(g, {"last_share": int(self.clock[0] - 60) * 1000}, None)
        self.assertEqual(g["pc_pool"], {"m1": True})
        self.assertIn("WARN biz-rtx4080 mines Pearl on Kryptex on PC m1, though its command is for HeroMiners",
                      self.events())

    def two_on_pcs(self):
        """A Kryptex group on m1 and a HeroMiners group on m2, both an hour on their PCs, for whole rounds."""
        k, h = self.group(kx=True), self.group()
        for g, m in ((k, "m1"), (h, "m2")):
            g.update(has_pc=True, had_pc=True, machine=m, pc_since=self.clock[0] - 3600, created=self.clock[0] - 7200)
            self.http.on("GET", f"/containers/{g['name']}/instances",
                         (200, {"instances": [{"id": "i-" + m, "machine_id": m, "state": "running"}]}))
        self.http.on("GET", "/containers/biz-rtx4080(-2)?$", (200, {"current_state": {"status": "running"}}))
        return k, h

    def test_a_round_sees_which_pool_a_box_mines_on(self):
        self.config_file()
        k, h = self.two_on_pcs()
        self.http.on("GET", "v3/miner/workers", (200, {"results": [
            {"worker": "biz-rtx4080-2-g0", "avg_hashrate_30m": "199000000000000", "last_share": int(self.clock[0]) - 60}]}))
        self.x.round()
        self.assertEqual(h["pc_pool"], {"m2": True})         # its Kryptex row: the box on m2 mines there
        self.assertNotIn("pc_pool", k)
        self.assertEqual(h["kx_share"], self.clock[0] - 60)  # a HeroMiners group's Kryptex share is proof too
        self.assertIn("WARN biz-rtx4080-2 mines Pearl on Kryptex on PC m2", self.events())

    def test_kryptex_is_read_for_a_switched_group_with_none_on_kryptex(self):
        self.config_file(kryptex_groups=0)
        k, h = self.two_on_pcs()
        k["ended"] = self.clock[0]                           # no group on Kryptex
        h.update(sw_t=self.clock[0] - 3600, sw_from="m0", sw_prev=True)   # switched off an hour ago
        self.http.on("GET", "v3/miner/workers", (200, {"results": [
            {"worker": "biz-rtx4080-2-g0", "avg_hashrate_30m": "199000000000000", "last_share": int(self.clock[0]) - 60}]}))
        self.x.round()
        self.assertTrue(self.http.made("GET", "v3/miner/workers"))
        self.assertEqual(h["pc_pool"], {"m2": True})
        # Two hours after the switch the workers are still read: its box on m2 mines on Kryptex.
        self.clock[0] += 3600 + 60
        n = len(self.http.made("GET", "v3/miner/workers"))
        self.x.round()
        self.assertEqual(len(self.http.made("GET", "v3/miner/workers")), n + 1)
        # Once it is on a PC mining on HeroMiners, they aren't.
        h["pc_pool"] = {}
        self.clock[0] += 60
        self.x.round()
        self.assertEqual(len(self.http.made("GET", "v3/miner/workers")), n + 1)

    def test_a_box_seen_on_herominers_takes_its_last_share_from_there(self):
        self.config_file()
        k, h = self.two_on_pcs()
        k["pc_pool"], k["last_share"] = {"m1": False}, self.clock[0] - 900
        self.http.on("GET", "herominers", (200, {"workers": [{"name": "biz-rtx4080-g0", "hashrate": 199,
                                                               "lastShare": int(self.clock[0]) - 30}]}))
        self.x.round()
        self.assertEqual(k["last_share"], self.clock[0] - 30)

    def test_a_new_box_on_a_pc_seen_before_starts_unseen(self):
        g = self.mining("m1", 300)
        g.update(kx=True, machine=None, has_pc=False, pc_pool={"m1": False})
        self.step(g, 1, "m1")                                # Salad hands it m1 again: a new box there
        self.assertEqual(g["pc_pool"], {})
        self.assertTrue(self.x.on_kx(g, "m1", self.clock[0]))

    def test_a_stopped_test_leaves_a_box_already_on_herominers_on_its_pc(self):
        g = self.mining("m1", 300)
        g.update(kx=True, kx_since=self.clock[0] - 3600, pc_pool={"m1": False})
        self.x.st["kx_fails"] = 2
        self.x.kx_sync([g], self.clock[0])
        self.assertFalse(g["kx"])
        self.assertFalse(self.http.made("POST", "/reallocate$"))
        self.assertIn("KRYPTEX OFF biz-rtx4080 (the Kryptex test stopped after 2 boxes mined no shares there): Pearl "
                      "on HeroMiners; it mines that way on this PC already", self.events())
        self.assertEqual((g["sw_from"], g["sw_prev"]), ("m1", False))   # m1's box mined on HeroMiners all along
        g.update(machine=None, has_pc=False)              # Salad takes m1 back, and hands it back later
        self.step(g, 1, "m1")
        self.assertIsNone(g["sw_from"])                   # the new box there runs the group's command
        self.assertFalse(self.x.on_kx(g, "m1", self.clock[0]))

    def test_the_hashrate_check_counts_a_box_seen_on_herominers(self):
        a, k = self.group(), self.group()
        t = self.clock[0]
        self.x.st["created"], self.x.st["kx_any_t"] = t - 7200, t - 5 * 3600
        a["readings"] = [[t - n * 60, "m0", 200.0, 1, "1h"] for n in range(60)]
        k.update(kx=True, pc_pool={"m1": False}, readings=[[t - n * 60, "m1", 199.0, 1, "1h"] for n in range(60)])
        self.x.st["pool_tot"] = {"t": t, "paid": 0, "balance": 0, "unconfirmed": 0, "blocks": 0, "hashes": 0,
                                 "ths_1h": 399.0}
        self.x.critic(t)
        self.assertNotIn("DRIFT hashrate", self.events())

    def test_the_ledger_follows_salad_back_to_herominers(self):
        self.config_file()
        g = self.group(kx=True)
        g.update(kx_since=self.clock[0] - 3600, kx_share=self.clock[0] - 600)
        box = "apt-get ...; curl -fsSL x | sed > /root/box.sh; bash /root/box.sh"
        self.http.on("GET", "/containers/biz-rtx4080$", (200, {"current_state": {"status": "running"},
                                                              "container": {"command": ["/bin/sh", "-c", box]}}))
        self.http.on("GET", "/instances", (200, {"instances": [{"id": "i1", "machine_id": "m1", "state": "running"}]}))
        self.clock[0] += 60
        self.x.round()
        self.assertFalse(g["kx"])
        self.assertIsNone(g["kx_since"])
        self.assertEqual(g["kx_share"], self.clock[0] - 660)   # its Kryptex shares stay proof
        self.assertTrue(g["sw_prev"])                           # the PC it's on stays booked to Kryptex
        self.assertIn("WARN biz-rtx4080: Salad has the HeroMiners command for it, though the ledger said Kryptex",
                      self.events())

    def test_following_salad_keeps_what_the_pc_it_left_or_is_on_was_booked_to(self):
        g = self.group()
        g.update(left=["m0"])                                 # waiting, just left m0
        on = {"container": {"command": ["/bin/sh", "-c", "bash /root/box.sh --pool prl-us.kryptex.network:7048"]}}
        self.x.kx_follow(g, on, self.clock[0])
        self.assertEqual((g["sw_from"], g["sw_prev"]), ("m0", False))
        # A switch off whose move Salad refused and whose put-back timed out but went through: the box on m1 kept
        # its Kryptex command, and stays booked there when the ledger follows Salad.
        h = self.mining("m1", 300)
        h.update(kx=False, sw_t=self.clock[0] - 300, sw_from="m1", sw_prev=True)
        self.x.kx_follow(h, on, self.clock[0])
        self.assertTrue(h["kx"])
        self.assertTrue(self.x.on_kx(h, "m1", self.clock[0]))

    def test_readings_from_a_pc_seen_on_herominers_count_there(self):
        self.config_file()
        g = self.group()
        g.update(kx=True, kx_since=T0, created=T0 - 3600, pc_pool={"m1": False})
        self.http.on("GET", "/containers/biz-rtx4080", (200, {"current_state": {"status": "running"}}))
        self.http.on("GET", "/instances", (200, {"instances": [{"id": "i1", "machine_id": "m1", "state": "running"}]}))
        items = [{"text_log": "[mining g0] 08:06:00 198.0 TH/s · 2 accepted · 0 rejected · up 4h 3m",
                  "time": "2027-01-15T08:06:00Z",
                  "resource": {"labels": {"container_group_name": "biz-rtx4080", "machine_id": "m1"}}}]
        self.http.on("POST", "/log-entries", (200, {"items": items}))
        self.clock[0] = T0 + 500
        self.x.round()
        self.assertGreater(g["prl"], 0)
        self.assertEqual(g.get("prl_kx", 0.0), 0.0)
        self.assertEqual(self.x.kx_log_proof(g), 0)        # its shares went to HeroMiners

    def test_only_kryptex_s_own_count_proves_it_takes_a_box_s_work(self):
        self.config_file()
        g = self.group()
        g.update(kx=True, kx_since=T0, created=T0 - 3600)
        self.http.on("GET", "/containers/biz-rtx4080", (200, {"current_state": {"status": "running"}}))
        self.http.on("GET", "/instances", (200, {"instances": [{"id": "i1", "machine_id": "m1", "state": "running"}]}))
        items = [{"text_log": "[mining g0] 08:06:00 198.0 TH/s · 2 accepted · 0 rejected · up 4h 3m",
                  "time": "2027-01-15T08:06:00Z",
                  "resource": {"labels": {"container_group_name": "biz-rtx4080", "machine_id": "m1"}}}]
        self.http.on("POST", "/log-entries", (200, {"items": items}))
        self.clock[0] = T0 + 500
        self.x.round()
        self.assertGreater(g.get("prl_kx", 0.0), 0)       # booked to Kryptex, as its command says
        self.assertIsNone(g.get("kx_share"))
        self.http.on("GET", "v3/miner/workers", (200, {"results": [
            {"worker": "biz-rtx4080-g0", "avg_hashrate_30m": "198000000000000", "last_share": int(T0) + 520}]}))
        self.clock[0] += 60
        self.x.round()
        self.assertEqual(g["kx_share"], T0 + 520)

    def test_the_ledger_follows_the_command_salad_has(self):
        self.config_file()
        g = self.group()
        box = "apt-get ...; curl -fsSL x | sed > /root/box.sh; bash /root/box.sh"
        cmd = {"current_state": {"status": "running"},
               "container": {"command": ["/bin/sh", "-c", box.replace("sed", "sed -e '--pool prl-us.kryptex.network:7048'")]}}
        self.http.on("GET", "/containers/biz-rtx4080$", (200, cmd))
        self.http.on("GET", "/instances", (200, {"instances": [{"id": "i1", "machine_id": "m1", "state": "running"}]}))
        g["sw_t"] = self.clock[0]                          # just switched: Salad may not show it yet
        self.clock[0] += 60
        self.x.round()
        self.assertFalse(g["kx"])
        self.clock[0] += 61
        self.x.round()                                     # an update that timed out went through
        self.assertTrue(g["kx"])
        self.assertEqual(g["kx_since"], self.clock[0])
        self.assertEqual((g["sw_from"], g["sw_prev"]), ("m1", False))   # the box on m1 started on HeroMiners
        self.assertIn("WARN biz-rtx4080: Salad has the Kryptex command for it, though the ledger said HeroMiners",
                      self.events())
        self.assertFalse(self.http.made("PATCH", "/containers/"))   # one on Kryptex: nothing else switched

    def test_without_a_box_on_kryptex_only_the_totals_are_read_every_5_minutes(self):
        self.x.cfg["wallet"] = os.environ["PRL_WALLET"]
        self.http.on("GET", "v1/miner/balance", (503, {}))
        for _ in range(7):
            self.x.kx_pool(workers=False)
            self.clock[0] += 60
        self.assertFalse(self.http.made("GET", "v3/miner/workers"))
        self.assertEqual(len(self.http.made("GET", "v1/miner/balance")), 2)   # 08:00 and 08:05, though both failed

    def test_if_salad_refuses_the_move_and_the_put_back_the_ledger_follows_salad(self):
        g = self.mining("m1", 300)
        g.update(kx=True, kx_since=self.clock[0] - 3600)
        self.x.st["kx_fails"] = 2
        self.http.on("POST", "/reallocate$", (500, {}))
        self.http.on("PATCH", "/containers/", lambda b: (503, {}) if "--pool" in b["container"]["command"][2] else (200, {}))
        self.x.kx_sync([g], self.clock[0])
        self.assertFalse(g["kx"])                        # Salad has the HeroMiners command: the next PC starts on it
        self.assertEqual((g["machine"], g["sw_from"], g["sw_prev"]), ("m1", "m1", True))   # this PC stays Kryptex
        self.assertIn("refused to move it and to put its command back (HTTP 503)", self.events())

    def test_a_refused_switch_back_waits_15_minutes(self):
        g = self.mining("m1", 300)
        g.update(kx=True, kx_since=self.clock[0] - 3600)
        self.x.st["kx_fails"] = 2
        self.http.on("POST", "/reallocate$", (409, {}))
        for _ in range(5):
            self.x.kx_sync([g], self.clock[0])
            self.clock[0] += 60
        self.assertEqual(len(self.http.made("POST", "/reallocate$")), 1)
        self.clock[0] += 11 * 60
        self.x.kx_sync([g], self.clock[0])
        self.assertEqual(len(self.http.made("POST", "/reallocate$")), 2)

    def test_a_box_that_changed_pc_this_round_waits_a_round(self):
        g = self.mining("m1", 300)
        g.update(kx=True, kx_since=self.clock[0] - 3600)
        self.x.st["kx_fails"] = 2
        g["_machine"] = "m5"                             # Salad moved it; rules() hasn't seen the new PC yet
        self.x.kx_sync([g], self.clock[0])
        self.assertTrue(g["kx"])
        self.assertFalse(self.http.made("PATCH", "/containers/"))

    def test_a_late_reading_after_a_switch_back_counts_as_kryptex(self):
        self.x.cfg["search"] = False
        g = self.group()
        g.update(kx=False, sw_t=T0 + 300, sw_from="m0", sw_prev=True, created=T0 - 3600)
        self.http.on("GET", "/containers/biz-rtx4080", (200, {"current_state": {"status": "running"}}))
        self.http.on("GET", "/instances", (200, {"instances": [{"id": "i1", "machine_id": "m1", "state": "running"}]}))
        items = [{"text_log": "[mining g0] 08:06:00 198.0 TH/s · 2 accepted · 0 rejected · up 4h 3m",
                  "time": "2027-01-15T08:06:00Z",
                  "resource": {"labels": {"container_group_name": "biz-rtx4080", "machine_id": "m0"}}}]
        self.http.on("POST", "/log-entries", (200, {"items": items}))
        self.clock[0] = T0 + 500
        self.x.round()
        self.assertAlmostEqual(g["prl_kx"], 198 / 60 * USD_TH_HR / PRICE)

    def test_kryptex_totals_are_still_read_after_the_last_box_leaves(self):
        g = self.group()
        g.update(kx=False, prl=1.0, prl_kx=0.5)
        self.x.cfg["search"] = False
        self.http.on("GET", "/containers/biz-rtx4080", (200, {"current_state": {"status": "running"}}))
        self.http.on("GET", "/instances", (200, {"instances": []}))
        self.x.round()
        self.assertTrue(self.http.made("GET", "v1/miner/balance"))

    def test_the_hashrate_check_waits_an_hour_after_a_switch(self):
        g = self.group()
        t = self.clock[0]
        self.x.st["created"] = t - 7200
        g["readings"] = [[t - k * 60, "m1", 100.0, 1, "1h"] for k in range(60)]
        self.x.st["pool_tot"] = {"t": t, "paid": 0, "balance": 0, "unconfirmed": 0, "blocks": 0, "hashes": 0,
                                 "ths_1h": 200.0}
        self.x.st["kx_any_t"] = t - 600
        self.x.critic(t)
        self.assertNotIn("DRIFT hashrate", self.events())
        self.x.st["kx_any_t"], self.x.st["critic_hour"] = t - 4000, None
        self.x.critic(t)
        self.assertIn("DRIFT hashrate", self.events())

    def test_kryptex_workers_and_totals_are_read(self):
        wl = os.environ["PRL_WALLET"]
        self.http.on("GET", f"v3/miner/workers/{wl}", (200, {"results": [
            {"worker": "biz-rtx4080-g0", "valid": 12, "avg_hashrate_30m": "200000000000000.5", "last_share": int(T0)}]}))
        self.http.on("GET", f"v1/miner/balance/{wl}", (200, {"total": 0.6, "confirmed": 0.4, "unconfirmed": 0.2}))
        self.http.on("GET", f"v1/miner/payouts/{wl}", (200, {"results": [{"received": 1.5}, {"received": 1.0}]}))
        ws = self.x.kx_pool()
        self.assertEqual(ws["biz-rtx4080-g0"]["valid"], 12)
        kt = self.x.st["kx_tot"]
        self.assertEqual((kt["total"], kt["confirmed"], kt["unconfirmed"], kt["paid"]), (0.6, 0.4, 0.2, 2.5))
        self.clock[0] += 60
        self.x.kx_pool()
        self.assertEqual(len(self.http.made("GET", "v1/miner/balance")), 1)  # totals at most every 5 minutes
        self.assertEqual(len(self.http.made("GET", "v3/miner/workers")), 2)

    def test_each_pool_has_its_own_credit_ratio_and_a_switched_box_is_split(self):
        kg, hg = self.group(), self.group()
        kg.update(kx=True, prl=6.0, prl_kx=4.0)   # 2 PRL mined on HeroMiners before the switch, 4 on Kryptex
        hg["prl"] = 8.0
        self.x.st["pool_tot"] = {"t": T0, "paid": 6.0, "balance": 1.0, "unconfirmed": 1.0, "blocks": 20,
                                 "hashes": 0, "ths_1h": 0}
        self.assertAlmostEqual(self.x.credit_ratio(), 0.8)  # 8 credited for HeroMiners' 10, not for all 14
        self.assertAlmostEqual(self.x.kx_ratio(), 0.94)     # nothing read from Kryptex yet: what it was measured to pay
        self.x.st["kx_tot"] = {"t": T0, "total": 1.76, "confirmed": 1.0, "unconfirmed": 0.76, "paid": 2.0}
        self.assertAlmostEqual(self.x.kx_ratio(), 0.94)
        self.assertAlmostEqual(self.x.ratio(kg), 0.94)
        self.assertAlmostEqual(self.x.ratio(hg), 0.8)
        self.assertAlmostEqual(self.x.credited(kg), 2 * 0.8 + 4 * 0.94)

    def test_prl_mined_while_on_kryptex_is_counted_apart(self):
        self.x.cfg["search"] = False
        g = self.group()
        g.update(kx=True, kx_since=T0, created=T0 - 3600)
        self.http.on("GET", "/containers/biz-rtx4080", (200, {"current_state": {"status": "running"}}))
        self.http.on("GET", "/instances", (200, {"instances": [{"id": "i1", "machine_id": "m1", "state": "running"}]}))
        items = [{"text_log": "[mining g0] 08:06:00 198.0 TH/s · 2 accepted · 0 rejected · up 0h 1m",
                  "time": "2027-01-15T08:06:00Z",
                  "resource": {"labels": {"container_group_name": "biz-rtx4080", "machine_id": "m1"}}}]
        self.http.on("POST", "/log-entries", (200, {"items": items}))
        self.clock[0] = T0 + 500
        self.x.round()
        self.assertAlmostEqual(g["prl_kx"], 198 / 60 * USD_TH_HR / PRICE)
        self.assertAlmostEqual(g["prl"], g["prl_kx"])

    def test_a_late_reading_from_the_pc_a_box_left_counts_as_herominers(self):
        self.x.cfg["search"] = False
        g = self.group()
        g.update(kx=True, kx_since=T0 + 300, sw_t=T0 + 300, sw_from="m0", sw_prev=False, created=T0 - 3600)
        self.http.on("GET", "/containers/biz-rtx4080", (200, {"current_state": {"status": "running"}}))
        self.http.on("GET", "/instances", (200, {"instances": [{"id": "i1", "machine_id": "m1", "state": "running"}]}))
        line = lambda tm, mach, up: {"text_log": f"[mining g0] {tm[11:19]} 198.0 TH/s · 2 accepted · 0 rejected · up {up}",
                                     "time": tm, "resource": {"labels": {"container_group_name": "biz-rtx4080",
                                                                         "machine_id": mach}}}
        # The switch was at 08:05. 08:04: the old PC, before it; 08:06: the old PC again, a late line after it;
        # 08:07: the new PC.
        items = [line("2027-01-15T08:04:00Z", "m0", "4h 1m"), line("2027-01-15T08:06:00Z", "m0", "4h 3m"),
                 line("2027-01-15T08:07:00Z", "m1", "0h 1m")]
        self.http.on("POST", "/log-entries", (200, {"items": items}))
        self.clock[0] = T0 + 500
        self.x.round()
        one = 198 / 60 * USD_TH_HR / PRICE
        self.assertAlmostEqual(g["prl"], 3 * one)
        self.assertAlmostEqual(g["prl_kx"], one)   # only the new PC's reading

    def test_each_kryptex_boxs_shares_are_read_from_its_chart(self):
        g, h = self.group(kx=True), self.group()
        share = USD_TH_HR / PRICE * 2 ** 32 / 3.6e15 * 2 ** 21                         # PRL the network pays for one share
        t0 = int(g["created"]) * 1000
        chart = [{"timestamp": t0, "valid": 3}, {"timestamp": t0 + 600000, "valid": 2}, {"timestamp": t0 + 1200000, "valid": 0}]
        self.http.on("GET", f"/v1/miner/chart/.*/{g['name']}-g0", lambda body: (200, chart))
        t = self.clock[0]
        self.x.kx_count([g, h], t)
        self.assertEqual(len(self.http.made("GET", "/miner/chart/")), 1)   # not for the box on HeroMiners
        self.assertEqual(g["kx_shares"], 5)
        self.assertAlmostEqual(g["kx_gross"], 5 * share)
        self.x.kx_count([g, h], t + 4 * 60)                                  # at most every 5 minutes
        self.assertEqual(len(self.http.made("GET", "/miner/chart/")), 1)
        # The bucket still filling gets 2 more shares, and a new one starts, while the network pays twice as much:
        # each bucket keeps the pay of when it was first seen.
        self.x.st["mkt"]["prl_th_hr"] *= 2
        chart[1]["valid"], chart[2]["valid"] = 4, 1
        self.x.kx_count([g, h], t + 5 * 60)
        self.assertEqual(g["kx_shares"], 8)
        self.assertAlmostEqual(g["kx_gross"], 7 * share + 2 * share)
        # Once the group ends its buckets go; what it earned stays.
        g["ended"] = t
        self.x.kx_count([], t + 6 * 60)
        self.assertNotIn("kx_b", g)
        self.assertEqual(g["kx_shares"], 8)

    def test_a_box_that_just_left_kryptex_is_read_for_an_hour(self):
        g = self.group()
        g["sw_t"] = self.clock[0] - 50 * 60
        self.http.on("GET", "/miner/chart/", (200, [{"timestamp": int(g["created"]) * 1000, "valid": 2}]))
        self.x.kx_count([g], self.clock[0])
        self.assertEqual(g["kx_shares"], 2)
        g["sw_t"] = self.clock[0] - 61 * 60
        self.x.kx_count([g], self.clock[0] + 5 * 60)
        self.assertEqual(len(self.http.made("GET", "/miner/chart/")), 1)

    def test_a_box_that_mined_on_kryptex_before_the_counts_began_is_read_once(self):
        g = self.group()
        g["prl_kx"] = 0.5                                            # in the ledger, from before 20 October
        self.http.on("GET", "/miner/chart/", (200, [{"timestamp": int(g["created"]) * 1000, "valid": 2}]))
        self.x.kx_count([g], self.clock[0])
        self.assertEqual(g["kx_shares"], 2)
        self.x.kx_count([g], self.clock[0] + 5 * 60)
        self.assertEqual(len(self.http.made("GET", "/miner/chart/")), 1)

    def test_the_prl_column_is_by_the_pools_counts(self):
        kg, mixed, old, new = self.group(kx=True), self.group(kx=True), self.group(), self.group()
        kg.update(kx_shares=100, kx_gross=1.0)                       # only Kryptex: 94% of 1.0, not starred
        mixed.update(kx_shares=100, kx_gross=1.0, hm_units=1.0, hm_gross=1.0)   # and HeroMiners before it switched
        old.update(ended=self.clock[0], prl=2.0)                     # ended before the counts: the ledger's estimate
        new.update(ended=self.clock[0], hm_units=1.0, hm_gross=0.5)  # ended since
        report = self.report()
        self.assertRegex(report, rf"4080{w.NB}batch{w.NB}\(K\) \|[^\n]*\| 0\.940 \|\n")
        self.assertRegex(report, rf"4080-2{w.NB}batch{w.NB}\(K\) \|[^\n]*\| 1\.940\* \|\n")  # 0.94 + 1.0 before 10 blocks
        self.assertIn(f"| Others{w.NB}×2 | — | — | — | 2.500* |\n", report)
        self.assertIn("| **5.380\\*** |\n", report)

    def test_the_report_marks_kryptex_boxes_and_shows_both_pools(self):
        kg = self.group()
        hg = self.group()
        kg.update(kx=True, prl=4.0, prl_kx=4.0, spend=3.0, has_pc=True, pool_ths=199.6, kx_shares=2000, kx_gross=4.0)
        hg.update(prl=10.0, spend=8.0, hm_units=1.0, hm_gross=10.0)
        self.x.st["pool_tot"] = {"t": T0, "paid": 6.0, "balance": 1.0, "unconfirmed": 1.0, "blocks": 20,
                                 "hashes": 0, "ths_1h": 0}
        self.x.st["kx_tot"] = {"t": T0, "total": 1.88, "confirmed": 1.0, "unconfirmed": 0.88, "paid": 2.0}
        report = self.report()
        self.assertIn(f"| ⏳{w.NB}4080{w.NB}batch{w.NB}(K) |", report.replace("⛏️", "⏳"))
        self.assertIn(f"| ⏳{w.NB}4080-2{w.NB}batch |", report)
        self.assertIn(f"| ⛏️ PRL{w.NB}HeroMiners | 0 TH/s | 1.000 | 1.000 | 6.000 |", report)
        self.assertIn(f"| ⛏️ PRL{w.NB}Kryptex | 200 TH/s | 1.000 | 0.880 | 2.000 |", report)
        self.assertAlmostEqual(self.x.kx_ratio(), 0.97)
        self.assertAlmostEqual(self.x.credit_ratio(), 0.8)
        # Kryptex pays 94% of a share's value, so its shares (4 PRL at the network's rate) are 3.76 PRL, not starred.
        # HeroMiners' count (10 PRL of work) at its credit ratio is 8, starred: it pays by its luck.
        self.assertRegex(report, r"\(K\) \|[^\n]*\| 3\.760 \|\n")
        self.assertRegex(report, rf"4080-2{w.NB}batch \|[^\n]*\| 8\.000\* \|\n")
        self.assertIn("| **11.760\\*** |\n", report)

    def test_a_round_reads_a_kryptex_box_from_kryptex_and_logs_its_miner_lines(self):
        self.x.cfg["search"] = False
        g = self.group()
        g.update(kx=True, kx_since=T0)
        self.http.on("GET", "/containers/biz-rtx4080", (200, {"current_state": {"status": "running"}}))
        self.http.on("GET", "/instances", (200, {"instances": [{"id": "i1", "machine_id": "m1", "state": "running"}]}))
        items = [{"text_log": "[minerlog g0] pool accepted login as biz-rtx4080-g0", "time": "2027-01-15T08:06:00Z",
                  "resource": {"labels": {"container_group_name": "biz-rtx4080", "machine_id": "m1"}}}]
        self.http.on("POST", "/log-entries", (200, {"items": items}))
        # HeroMiners has a worker of that name too (an earlier group): a Kryptex box must not take its figures.
        self.http.on("GET", "herominers", (200, {"workers": [{"name": "biz-rtx4080-g0", "hashrate": 44500,
                                                               "lastShare": int(T0) + 450}]}))
        self.http.on("GET", "v3/miner/workers", (200, {"results": [
            {"worker": "biz-rtx4080-g0", "valid": 3, "avg_hashrate_30m": "200000000000000", "last_share": (int(T0) + 400) * 1000}]}))
        # Its chart: 4 shares in a bucket since the group began, and 9 in one that ended before (the earlier group's).
        self.http.on("GET", "/miner/chart/", (200, [{"timestamp": int(T0 - 7200) * 1000, "valid": 9},
                                                     {"timestamp": int(T0 - 600) * 1000, "valid": 4}]))
        self.x.st["kx_tot"] = {"t": T0 + 400, "total": 0, "confirmed": 0, "unconfirmed": 0, "paid": 0}
        g["created"] = T0 - 3600
        self.clock[0] = T0 + 500
        self.x.round()
        self.assertEqual(g["kx_shares"], 4)
        self.assertEqual(g["last_share"], T0 + 400)
        self.assertAlmostEqual(g["pool_ths"], 200.0)
        self.assertTrue(g["kx"])   # it has a PC: nothing to switch
        self.assertIn("LOG biz-rtx4080 PC m1: [minerlog g0] pool accepted login as biz-rtx4080-g0", self.events())

    def test_an_ended_kryptex_group_that_never_mined_there_leaves_the_table_as_before(self):
        g = self.group()
        g.update(kx=True, ended=self.clock[0], prl=0.0)
        h = self.group()
        h["prl"], h["spend"] = 8.0, 6.0
        self.x.st["pool_tot"] = {"t": T0, "paid": 1.5, "balance": 0.5, "unconfirmed": 4.0, "blocks": 6,
                                 "hashes": 0, "ths_1h": 0}
        self.x.st["kx_tot"] = {"t": T0, "total": 0.0, "confirmed": 0.0, "unconfirmed": 0.0, "paid": 0.0}
        report = self.report()
        self.assertIn("| ⛏️ PRL | 0 TH/s | 0.500 | 4.000 | 1.500 |", report)
        self.assertNotIn("Kryptex", report)

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


class TestGemma(Base):
    """Config "llm": the box runs Gemma beside the miner, and mining that pauses for a prompt isn't read as dead."""
    # The box.sh lines it changes, as they are at the pinned commit.
    BOX = ("  cd $R; env $envs ./cli -a $W --mode mining -w $WORKER-g$g --no-serve --no-update $flags >> $D/g$g.log 2>&1 &\n"
           "    line=$(tr -d '\\r' < $D/g$g.log | grep -oE '[0-9.]+ TH/s · [0-9,]+ accepted · [0-9,]+ rejected · up "
           "[0-9]+[hm] [0-9]+[ms]' | tail -1)\n"
           "    echo \"[mining g$g] $(now) $line\"\n")

    def sed(self, cmd, text):
        import subprocess
        sed = re.search(r"(sed .*?) > /root/box\.sh", cmd).group(1)
        return subprocess.run(["sh", "-c", sed], input=text, capture_output=True, text=True).stdout

    def test_the_command_runs_auto_mode_serving_and_says_when_mining_pauses(self):
        out = self.sed(w.cmd_for("biz-rtx4080", "prl1abc", "v0.6.5", XMR, 0, "prl-us.kryptex.network:7048", True), self.BOX)
        self.assertIn("./cli -a $W --mode auto -w $WORKER-g$g --no-update $flags", out)
        self.assertNotIn("--no-serve", out)
        # box.sh's status line, run on a miner's log: its hashrate while mining, "paused" while the LLM has the card.
        import subprocess, tempfile
        loop = out.split("\n", 1)[1].replace("$(now)", "02:30:00")
        with tempfile.TemporaryDirectory() as d:
            log = os.path.join(d, "g0.log")
            def line(text):
                with open(log, "a") as f:
                    f.write(text)
                return subprocess.run(["bash", "-c", loop], env={"D": d, "g": "0", "PATH": os.environ["PATH"]},
                                      capture_output=True, text=True).stdout.strip()
            mine = "121.0 TH/s · 22 accepted · 0 rejected · up 0h 31m"
            self.assertEqual(line("⛏  " + mine + "\n"), "[mining g0] 02:30:00 " + mine)
            self.assertEqual(line("mining paused on GPU 0 while the local LLM answers a prompt\n"),
                             "[mining g0] 02:30:00 paused for a prompt")
            again = "119.0 TH/s · 23 accepted · 0 rejected · up 0h 33m"
            self.assertEqual(line("mining resumed on GPU 0\n⛏  " + again + "\n"), "[mining g0] 02:30:00 " + again)

    def test_without_it_the_command_is_unchanged(self):
        for args in (("v0.6.0",), ("v0.6.0", XMR), ("v0.6.1", XMR, 0, "prl-us.kryptex.network:7048")):
            self.assertEqual(w.cmd_for("biz-rtx4080", "prl1abc", *args), w.cmd_for("biz-rtx4080", "prl1abc", *args, llm=False)
                             if len(args) == 4 else w.cmd_for("biz-rtx4080", "prl1abc", *args))
            self.assertEqual(self.sed(w.cmd_for("biz-rtx4080", "prl1abc", *args), self.BOX), self.BOX)
        self.x.cfg["xmr_wallet"], self.x.cfg["xmr_groups"] = XMR, ["*"]
        self.assertNotIn("--mode auto", self.x.box("biz-rtx4080", True, True)[1][2])

    def test_a_box_gets_release_llm_ver_or_later(self):
        self.x.cfg.update(llm=True, xmr_wallet=XMR)
        for xmr, kx in ((False, False), (True, False), (True, True)):
            ver, cmd, _ = self.x.box("biz-rtx4080", xmr, kx)
            self.assertEqual(ver, "v0.6.5")
            self.assertIn("VER=v0.6.5;", cmd[2])
            self.assertIn("--mode auto", cmd[2])
        self.x.cfg["ver_override"] = {"biz-rtx4080": "v0.7.0"}
        self.assertEqual(self.x.box("biz-rtx4080", True, True)[0], "v0.7.0")   # never lowered

    def test_a_new_group_runs_gemma(self):
        self.x.cfg["llm"] = True
        g = self.group()
        self.assertTrue(g["llm"])
        self.assertIn("--mode auto", self.http.made("POST", "/containers$")[-1][2]["container"]["command"][2])
        self.assertIn(", Gemma", self.events())

    def test_turning_it_on_updates_each_group_without_moving_a_box_that_mines(self):
        mining, waiting, preparing = self.group(), self.group(status="deploying"), self.group(status="pending")
        mining.update(has_pc=True, machine="m1", instance="inst-m1")
        self.http.on("PATCH", "/containers/", (200, {}))
        self.x.cfg["llm"] = True
        t = self.clock[0]
        self.x.llm_sync([mining, waiting, preparing], t)
        patched = [c[1] for c in self.http.made("PATCH", "/containers/")]
        self.assertEqual(len(patched), 2)                       # not the group still preparing its image
        self.assertTrue(all("--mode auto" in c[2]["container"]["command"][2] for c in self.http.made("PATCH", "/containers/")))
        self.assertFalse(self.http.made("POST", "/restart") or self.http.made("POST", "/reallocate"))   # no hash lost
        self.assertTrue(mining["llm"] and waiting["llm"] and not preparing.get("llm"))
        self.assertIn(f"GEMMA ON {mining['name']}: miner v0.6.5; it keeps mining on this PC as it was", self.events())
        self.assertIn(f"GEMMA ON {waiting['name']}: miner v0.6.5; it starts that way on its next PC", self.events())
        self.x.llm_sync([mining, waiting], t + 60)              # done: nothing more to send
        self.assertEqual(len(self.http.made("PATCH", "/containers/")), 2)
        # A refused update is tried again after 10 minutes.
        preparing["status"] = "deploying"
        self.http.on("PATCH", "/containers/", (409, {"error": "busy"}))
        self.x.llm_sync([preparing], t + 60)
        self.assertIn(f"WARN Gemma on for {preparing['name']} refused: HTTP 409", self.events())
        self.x.llm_sync([preparing], t + 5 * 60)
        self.assertEqual(len(self.http.made("PATCH", "/containers/")), 3)
        self.x.llm_sync([preparing], t + 11 * 60)
        self.assertEqual(len(self.http.made("PATCH", "/containers/")), 4)

    def test_a_pause_shows_only_while_box_sh_says_so_and_for_15_minutes_at_most(self):
        g, t = self.group(), self.clock[0]
        g.update(pause_since=t, paused_t=t)
        self.assertTrue(self.x.paused(g, t + 4 * w.MIN))
        self.assertFalse(self.x.paused(g, t + 5 * w.MIN))          # box.sh stopped saying so: not paused any more
        g["paused_t"] = t + 15 * w.MIN
        self.assertTrue(self.x.paused(g, t + 15 * w.MIN - 1))
        self.assertFalse(self.x.paused(g, t + 15 * w.MIN))         # longer than the CLI ever pauses

    def test_a_new_pc_starts_with_no_pause(self):
        g = self.group(pri="medium")
        self.step(g, 1, "m1")
        self.mine(g, "m1", 199, 5)
        self.paused(g, 3)
        self.step(g, 1, "m2")                                   # Salad hands it another PC
        self.assertIsNone(g["pause_since"])
        self.assertIsNone(g["paused_t"])
        self.step(g, 10, "m2")                                  # never mines there: left at 10 min, as on any PC
        self.assertIn("no mining reading 10 min after getting PC m2", self.events())

    def test_switching_a_groups_pool_or_monero_keeps_gemma(self):
        self.x.cfg.update(llm=True, xmr_wallet=XMR, xmr_groups=["*"], kryptex_groups=1)
        g = self.group(status="deploying")
        g["llm"] = False                                        # as if made before Gemma was on
        self.http.on("PATCH", "/containers/", (200, {}))
        self.x.kx_switch(g, not g["kx"], self.clock[0], "test")
        self.assertIn("--mode auto", self.http.made("PATCH", "/containers/")[-1][2]["container"]["command"][2])
        self.assertTrue(g["llm"])
        g["llm"] = False
        self.x.cfg["xmr_groups"] = []
        self.x.xmr_sync([g], self.clock[0])
        self.assertIn("--mode auto", self.http.made("PATCH", "/containers/")[-1][2]["container"]["command"][2])
        self.assertTrue(g["llm"])

    def paused(self, g, minutes, machine="m1"):
        """box.sh says "paused" a minute at a time for `minutes`, with no readings, the rules running each minute."""
        for _ in range(minutes):
            self.clock[0] += 60
            g["paused_t"], g["pause_since"] = self.clock[0], g.get("pause_since") or self.clock[0]
            g["has_pc"], g["_machine"], g["instance"] = True, machine, "inst-" + machine
            self.x.rules(g, self.clock[0])

    def test_a_box_paused_for_a_prompt_keeps_its_pc(self):
        g = self.group(pri="medium")
        self.step(g, 1, "m1")
        self.mine(g, "m1", 199, 10)
        self.paused(g, 9)                                       # a long prompt: no readings, no shares
        self.assertFalse(self.http.made("POST", "/reallocate"))
        g["pause_since"] = None                                 # mining again
        self.mine(g, "m1", 199, 5)
        self.assertFalse(self.http.made("POST", "/reallocate"))

    def test_a_pause_counts_for_15_minutes_at_most(self):
        g = self.group(pri="medium")
        self.step(g, 1, "m1")
        self.mine(g, "m1", 199, 10)
        self.paused(g, 20)                                      # the CLI resumes after 10: this miner isn't coming back
        self.assertFalse(self.http.made("POST", "/reallocate"))  # 15 min of the pause, then 6 with nothing
        self.paused(g, 2)
        self.assertTrue(self.http.made("POST", "/instances/inst-m1/reallocate"))
        self.assertIn("no mining reading for", self.events())

    def test_the_readings_just_after_a_pause_get_a_warm_up(self):
        g = self.group(pri="medium")                            # $0.19/hr: 100 TH/s is under 10%
        self.step(g, 1, "m1")
        self.mine(g, "m1", 199, 10)
        self.paused(g, 2)
        g["pause_since"] = None
        self.mine(g, "m1", 100, 2)                              # the card coming back up to speed
        self.assertFalse(self.http.made("POST", "/reallocate"))
        self.mine(g, "m1", 199, 5)
        self.assertFalse(self.http.made("POST", "/reallocate"))

    def test_the_share_clock_starts_again_after_a_pause(self):
        g = self.group(pri="medium")
        self.step(g, 1, "m1")
        self.mine(g, "m1", 199, 21)
        self.paused(g, 9)
        g["pause_since"] = None
        self.mine(g, "m1", 199, 9, shares=False)                # 18 min since its last share, 9 since the pause
        self.assertNotIn("no share", self.events())

    def test_the_round_reads_paused_lines_and_a_reading_ends_the_pause(self):
        self.x.cfg["search"] = False
        g = self.group()
        g.update(created=T0 - 3600, machine="m1", has_pc=True, had_pc=True, pc_since=T0 - 1800)
        self.http.on("GET", "/containers/biz-rtx4080", (200, {"current_state": {"status": "running"}}))
        self.http.on("GET", "/instances", (200, {"instances": [{"id": "i1", "machine_id": "m1", "state": "running"}]}))
        lab = {"labels": {"container_group_name": "biz-rtx4080", "machine_id": "m1"}}
        items = [{"text_log": "[mining g0] 08:05:00 199.0 TH/s · 9 accepted · 0 rejected · up 0h 30m",
                  "time": "2027-01-15T08:05:00Z", "resource": lab},
                 {"text_log": "[mining g0] 08:06:00 paused for a prompt", "time": "2027-01-15T08:06:00Z", "resource": lab},
                 {"text_log": "[mining g0] 08:07:00 paused for a prompt", "time": "2027-01-15T08:07:00Z", "resource": lab}]
        self.http.on("POST", "/log-entries", (200, {"items": items}))
        self.clock[0] = T0 + 500
        self.x.round()
        self.assertEqual((g["pause_since"], g["paused_t"]), (T0 + 360, T0 + 420))
        self.assertEqual(self.events().count("PAUSED biz-rtx4080: mining stopped on PC m1 while Gemma answers a prompt"), 1)
        self.assertTrue(self.x.paused(g, self.clock[0]))
        with open(os.path.join(TMP, "report.md")) as f:
            self.assertIn(f"| 🧠{w.NB}4080{w.NB}batch |", f.read())
        items.append({"text_log": "[mining g0] 08:09:00 199.0 TH/s · 10 accepted · 0 rejected · up 0h 34m",
                      "time": "2027-01-15T08:09:00Z", "resource": lab})
        self.clock[0] = T0 + 600
        self.x.round()
        self.assertIsNone(g["pause_since"])
        self.assertIn("RESUMED biz-rtx4080: mining again on PC m1 after 3 min paused for Gemma", self.events())
        self.assertEqual(g["paused_t"], T0 + 420)               # its end still gives the next readings their warm-up
        self.assertFalse(self.x.paused(g, self.clock[0]))


if __name__ == "__main__":
    unittest.main()
