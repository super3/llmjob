# Tests for watcher.py's rules, with Salad, the pool and the clock faked.
# Run: python3 -m unittest discover -s tuning/salad
import importlib.util, io, json, os, re, shutil, tempfile, unittest

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


# The box command's sed and shell run on Salad's Linux boxes; a host without sh (Windows) skips the tests that run them.
HAS_SH = bool(shutil.which("sh")) and os.path.exists("/bin/sh")
needs_sh = unittest.skipUnless(HAS_SH, "needs sh, as the boxes have")


def shown(row):
    """A Markdown table row as a chat window shows it: without bold marks or the backslashes before stars, and a
    link as its text."""
    return re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", row.replace("**", "").replace("\\*", "*"))


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
            with open(os.path.join(TMP, "events.log"), encoding="utf-8") as f:
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

    # The lines of box.sh (at BOX's commit) that an AMD box's command changes.
    AMD_LINES = "  nvidia-smi --query-gpu=index,name,driver_version,power.limit,clocks.max.sm,compute_cap --format=csv,noheader | sed 's/^/[card] /'\n  declare -A dl; for f in llmjob-earn-cli-linux pearl_core.node pearl_core_cu13.node; do   # all three at once\nif [ ! -f $D/tools ]; then\nN=$(nvidia-smi -L | wc -l)\nCC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')\n    release) cp $D/rel/pearl_core.node $D/rel/pearl_core_cu13.node $R/ ;;\n  [ $((tick % 5)) = 0 ] && nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu --format=csv,noheader,nounits |\n"

    def amd_box(self, cmd):
        if not HAS_SH:
            self.skipTest("needs sh, as the boxes have")
        import subprocess
        sed = re.search(r"(sed .*?) > /root/box\.sh", cmd).group(1)
        out = subprocess.run(["sh", "-c", sed], input=self.AMD_LINES, capture_output=True, encoding="utf-8")
        self.assertEqual(out.returncode, 0, out.stderr)
        return out.stdout

    def test_slug_and_is_amd(self):
        self.assertEqual(w.slug("AMD RX 9070 XT (16GB)"), "rx9070xt")
        self.assertEqual(w.slug("AMD RX 7900 XT (20GB)"), "rx7900xt")
        self.assertTrue(w.is_amd("biz-rx9070xt-2"))
        self.assertFalse(w.is_amd("biz-rtx4080"))

    def test_an_amd_box_runs_the_amd_core_and_needs_no_nvidia_smi(self):
        out = self.amd_box(w.cmd_for("biz-rx9070xt", "prl1abc", "v0.6.8", XMR, 0, "prl-us.kryptex.network:7048", amd=True))
        self.assertNotIn("nvidia-smi", out)
        self.assertIn('echo "[card] 0, $(rocminfo 2>/dev/null | grep -m1 -oE "AMD Radeon[ A-Za-z0-9]+")', out)
        self.assertIn("for f in pearl_core_hip.node llmjob-earn-cli-linux ", out)
        self.assertIn("release) cp $D/rel/pearl_core_hip.node $D/rel/pearl_core.node ", out)
        self.assertIn("\nN=1\n", out)
        self.assertIn("if false; then", out)   # no CUDA build tools
        self.assertIn("&& true --query-gpu=index,clocks.sm", out)
        self.assertIn("--core amd", w.cmd_for("biz-rx9070xt", "prl1abc", "v0.6.8", amd=True))
        # An NVIDIA box's command leaves these lines alone.
        self.assertEqual(self.amd_box(w.cmd_for("biz-rtx4080", "prl1abc", "v0.6.6", kryptex="prl-us.kryptex.network:7048")), self.AMD_LINES)

    def test_an_amd_group_gets_the_rocm_image_v0_6_8_and_no_gemma(self):
        self.x.cfg["llm"] = True
        self.x.st["classes"]["AMD RX 9070 XT (16GB)"] = {"id": "c9070", "prices": {"batch": 0.07, "low": 0.11,
                                                                                   "medium": 0.155, "high": 0.196}}
        self.assertTrue(self.x.create("AMD RX 9070 XT (16GB)", "batch"))
        body = self.http.made("POST", "/containers$")[-1][2]
        self.assertEqual(body["name"], "biz-rx9070xt")
        self.assertEqual(body["container"]["image"], "rocm/dev-ubuntu-24.04:7.2")
        cmd = body["container"]["command"][2]
        self.assertIn("--core amd", cmd)
        self.assertIn("s/^VER=v0.5.13;/VER=v0.6.8;/", cmd)
        self.assertNotIn("--mode auto", cmd)
        g = list(self.x.st["groups"].values())[-1]
        self.assertEqual((g["ver"], g["llm"]), ("v0.6.8", False))
        # Gemma stays off it: llm_sync doesn't keep trying to turn it on.
        self.x.llm_sync([g], w.now() + 3600)
        self.assertEqual(len(self.http.made("PATCH", "/containers/biz-rx9070xt")), 0)
        # An NVIDIA group still gets Gemma and its own image.
        self.assertTrue(self.x.create("RTX 4080 (16 GB)", "batch"))
        nv = self.http.made("POST", "/containers$")[-1][2]
        self.assertEqual(nv["container"]["image"], "nvidia/cuda:12.2.2-base-ubuntu22.04")
        self.assertIn("--mode auto", nv["container"]["command"][2])

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
    def test_moves_up_20_min_after_losing_its_pc_then_deletes_when_nothing_higher_clears(self):
        g = self.group()  # batch; low clears 10% for a 4080, medium too, high doesn't
        self.step(g, 1, "m1")
        self.step(g, 5, None)
        self.step(g, 21, None)
        self.assertEqual(self.http.made("PATCH", "/containers/biz-rtx4080")[0][2], {"container": {"priority": "low"}})
        self.assertIsNone(g["ended"])
        self.step(g, 19, None)  # 20 more minutes at low before the next step
        self.assertEqual(len(self.http.made("PATCH", "")), 1)
        self.step(g, 2, None)
        self.assertEqual(g["pri"], "medium")
        self.step(g, 21, None)
        self.assertEqual(g["end_kind"], "released")
        self.assertIn("no PC since losing one 20 min ago, moved to low", self.events())

    def test_a_refused_priority_change_waits_for_the_next_round(self):
        g = self.group()
        self.step(g, 1, "m1")
        self.step(g, 5, None)
        self.http.on("PATCH", "/containers/biz-rtx4080", (400, {}))
        self.step(g, 21, None)
        self.assertIsNone(g["ended"])
        self.assertEqual(g["pri"], "batch")

    def test_deletes_20_min_after_losing_its_pc(self):
        g = self.group(pri="medium")  # high is +2% for a 4080
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

    def test_the_reading_clock_starts_when_the_container_runs(self):
        # 28 October: rx9060xt was left 10 minutes after getting a PC that was still downloading the 1.16 GB AMD image,
        # with no container output at all, and its class barred for a day as failed.
        g = self.group(cls="RTX 4070 (12 GB)", pri="low")       # a class that never paid: leaving deletes it
        g["_run"] = False                                       # Salad shows the PC, downloading the image
        self.step(g, 1, "m1")
        self.step(g, 14, "m1")
        self.assertIsNone(g["ended"])
        self.assertFalse(self.http.made("POST", "/reallocate"))
        g["_run"] = True                                        # running from minute 15
        self.step(g, 1, "m1")
        self.step(g, 9, "m1")
        self.assertIsNone(g["ended"])
        self.step(g, 1, "m1")                                   # 10 min after it started: judged as usual
        self.assertEqual(g["end_kind"], "failed")
        self.assertIn("no mining reading 10 min after it started on PC m1 (15 min loading first)", self.events())

    def test_a_pc_that_never_starts_the_container_is_left_after_30_min(self):
        g = self.group(cls="RTX 4070 (12 GB)", pri="low")
        g["_run"] = False
        self.step(g, 1, "m1")
        self.step(g, 29, "m1")
        self.assertFalse(self.http.made("POST", "/reallocate"))
        self.step(g, 1, "m1")
        self.assertTrue(self.http.made("POST", "/instances/inst-m1/reallocate"))
        self.assertIsNone(g["ended"])                           # the PC's fault: not deleted, class not barred
        self.assertEqual(g["reallocs"], 0)
        self.assertIn("not running 30 min after getting PC m1 (image still loading)", self.x.st["bad_pcs"]["m1"]["why"])

    def test_a_group_on_its_pc_before_the_run_clock_keeps_its_clock(self):
        g = self.group()
        self.step(g, 1, "m1")
        del g["run_since"]                                      # as saved by the watcher before this rule
        self.step(g, 9, "m1")
        self.assertEqual(g["run_since"], g["pc_since"])
        self.step(g, 1, "m1")
        self.assertTrue(self.http.made("POST", "/instances/inst-m1/reallocate"))


class TestPCsLeftBefore(Base):
    def test_the_pc_just_left_is_skipped_for_10_minutes_only(self):
        g = self.group()
        self.step(g, 1, "m1")
        self.step(g, 10, "m1")                       # no reading for 10 minutes: left
        self.assertTrue(self.http.made("POST", "/instances/inst-m1/reallocate"))
        n = len(self.http.calls)
        self.step(g, 1, "m1")                        # Salad still shows the old PC: not a new PC, nothing done
        self.assertEqual(len(self.http.calls), n)
        self.assertNotIn("NEW PC biz-rtx4080", self.events())
        self.step(g, 10, "m1")                       # still there 11 minutes on: judged like any PC
        self.assertIn("NEW PC biz-rtx4080 (RTX 4080 (16 GB), batch): PC m1", self.events())

    def test_a_pc_left_hours_ago_is_a_new_pc(self):
        # 22 October: 2070-3 was moved off 452ceb99, then handed 6091f458, a bad-listed PC it had left hours before.
        # Every PC it had ever left was skipped, so no rule ran on it and it billed at 0 TH/s for 4.5 hours.
        g = self.group()
        self.step(g, 1, "m1")
        self.step(g, 10, "m1")
        self.assertIn("REALLOCATED biz-rtx4080 off PC m1", self.events())
        self.step(g, 1, "m2")
        self.mine(g, "m2", 199, 3)
        self.x.reallocate(g, "slow")                 # moved off m2; m1 is its last PC but one
        self.clock[0] += 3 * 3600
        self.x.st["bad_pcs"]["m1"] = {"t": self.clock[0] - 600, "group": "biz-rtx4080-2", "why": "slow"}
        n = len(self.http.made("POST", "/reallocate"))
        self.step(g, 1, "m1")                        # handed m1 again: a new PC, and on the bad list
        self.assertIn("NEW PC biz-rtx4080 (RTX 4080 (16 GB), batch): PC m1", self.events())
        self.assertEqual(len(self.http.made("POST", "/reallocate")), n + 1)
        self.assertIn("(biz-rtx4080-2 left this PC at", self.events())


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


class TestRestartOnTheSamePC(Base):
    def test_a_restart_on_the_same_pc_starts_the_reading_clock_again(self):
        # 12 days to 22 October: 4 of the 39 "no mining reading for 6 min" moves came just after box.sh started
        # again on the same PC, a good PC given up while the box was starting (3090ti-19, 5060-46, 3060-54).
        g = self.group(pri="medium")
        self.step(g, 1, "m1")
        self.mine(g, "m1", 199, 12)                              # (so the 20-minute share rule runs below)
        g["restart_t"], g["restart_pc"] = self.clock[0], "m1"   # box.sh started again on m1 (see the round test)
        g["last_share"] = self.clock[0] - 5 * 60                 # and its last share came 5 minutes before that
        self.step(g, 9, "m1")                                    # 9 quiet minutes: past the 6-minute rule
        self.assertFalse(self.http.made("POST", "/reallocate"))
        self.step(g, 2, "m1")                                    # 11: no reading 10 min after it, as on a new PC
        self.assertTrue(self.http.made("POST", "/instances/inst-m1/reallocate"))
        self.assertIn("no mining reading 11 min after its restart on PC m1", self.events())
        self.step(g, 1, "m2")                                    # a new PC: its own clock, not the restart's
        self.step(g, 10, "m2")
        self.assertIn("no mining reading 10 min after getting PC m2", self.events())

    def test_readings_after_a_restart_are_judged_from_a_fresh_warm_up(self):
        g = self.group(pri="medium")
        self.step(g, 1, "m1")
        self.mine(g, "m1", 199, 10)
        g["restart_t"], g["restart_pc"] = self.clock[0], "m1"
        self.mine(g, "m1", 100, 2)                               # warm-up readings after the restart: low, not judged
        self.assertFalse(self.http.made("POST", "/reallocate"))
        self.mine(g, "m1", 199, 5)
        self.assertFalse(self.http.made("POST", "/reallocate"))

    def round_with(self, lines, readings=True):
        """One round with the group on m1 (mining there before, if `readings`) and these log lines from m1."""
        self.x.cfg["search"] = False
        g = self.group()
        g.update(created=T0 - 3600, machine="m1", pc_since=T0 - 3000, had_pc=True, has_pc=True, last_share=T0 - 60)
        if readings:
            g["readings"] = [[T0 - 600 + 60 * i, "m1", 199.0, i, f"up 0h {i}m"] for i in range(5)]
        self.http.on("GET", "/containers/biz-rtx4080", (200, {"current_state": {"status": "running"}}))
        self.http.on("GET", "/instances", (200, {"instances": [{"id": "i1", "machine_id": "m1", "state": "running"}]}))
        self.http.on("POST", "/log-entries", (200, {"items": [
            {"text_log": txt, "time": tm, "resource": {"labels": {"container_group_name": "biz-rtx4080", "machine_id": "m1"}}}
            for tm, txt in lines]}))
        self.x.round()
        return g

    def test_the_round_notices_a_restart_once_a_pc(self):
        g = self.round_with([("2027-01-15T07:57:00Z", "[run] release v0.6.6 ready"),
                             ("2027-01-15T07:59:00Z", "[run] release v0.6.6 ready")])
        self.assertEqual((g["restart_t"], g["restart_pc"]), (T0 - 180, "m1"))
        self.assertEqual(self.events().count("RESTARTED biz-rtx4080 on PC m1"), 1)

    def test_a_second_running_instance_is_reported_once(self):
        self.x.cfg["search"] = False
        g = self.group()
        g.update(created=T0 - 3600, machine="m1", pc_since=T0 - 3000, had_pc=True, has_pc=True, last_share=T0 - 60)
        self.http.on("GET", "/containers/biz-rtx4080", (200, {"current_state": {"status": "running"}}))
        two = [{"id": "i1", "machine_id": "m1", "state": "running"}, {"id": "i2", "machine_id": "m2", "state": "running"}]
        self.http.on("GET", "/instances", (200, {"instances": two}))
        for _ in range(2):
            self.clock[0] += 60
            self.x.round()
        self.assertEqual(self.events().count("WARN biz-rtx4080: Salad runs 2 instances of it, on PCs m1, m2; only m1 is "
                                             "judged"), 1)
        self.assertEqual(g["machine"], "m1")

    def two_pc_round(self, pc_since, rows):
        """One round with the group listed on m2 since `pc_since`, and these (time, PC, group, line) log lines."""
        self.x.cfg["search"] = False
        g = self.group()
        g.update(created=T0 - 7200, machine="m2", pc_since=pc_since, had_pc=True, has_pc=True, last_share=T0 - 60)
        self.http.on("GET", "/containers/biz-rtx4080", (200, {"current_state": {"status": "running"}}))
        self.http.on("GET", "/instances", (200, {"instances": [{"id": "i2", "machine_id": "m2", "state": "running"}]}))
        self.http.on("POST", "/log-entries", (200, {"items": [
            {"text_log": txt, "time": tm, "resource": {"labels": {"container_group_name": name, "machine_id": m}}}
            for tm, m, name, txt in rows]}))
        self.x.round()
        return g

    # 3060-71 on 23 Oct: Salad listed a new copy on a new PC while the first mined on, on its own PC, unlisted.
    def test_a_first_copy_still_mining_beside_the_listed_one_is_reported_once(self):
        rows = [("2027-01-15T07:58:00Z", "m1", "biz-rtx4080", "[mining g0] 07:58:00 199.0 TH/s · 29 accepted · 0 rejected · up 1h 15m"),
                ("2027-01-15T07:58:10Z", "m2", "biz-rtx4080", "[mining g0] 07:58:10 150.0 TH/s · 2 accepted · 0 rejected · up 08m 01s"),
                ("2027-01-15T07:59:00Z", "m1", "biz-rtx4080", "[mining g0] 07:59:00 199.0 TH/s · 30 accepted · 0 rejected · up 1h 16m")]
        g = self.two_pc_round(T0 - 600, rows)
        self.clock[0] += 60
        self.x.round()   # the same lines again
        self.assertEqual(self.events().count("WARN biz-rtx4080: PC m1 still mines for it (199 TH/s, up 1h 15m), though "
                                             "Salad lists it on PC m2: two copies, and only m2 is judged"), 1)
        self.assertEqual(g["machine"], "m2")

    # A move's old PC can log for a minute or two while Salad stops it there.
    def test_the_old_pc_logging_just_after_a_move_is_not_a_second_copy(self):
        rows = [("2027-01-15T07:58:00Z", "m1", "biz-rtx4080", "[mining g0] 07:58:00 199.0 TH/s · 29 accepted · 0 rejected · up 1h 15m")]
        self.two_pc_round(T0 - 200, rows)
        self.assertNotIn("still mines", self.events())

    # 3060-50 on 21 Oct mined 16 min after its delete, and 3060-71 on 23 Oct over 10 min.
    def test_a_box_mining_after_its_group_was_deleted_is_reported_once(self):
        self.x.cfg["search"] = False
        old = self.group()
        old.update(ended=T0 - 600, end_kind="released", had_pc=True)
        mining = lambda tm, up: (tm, "m9", "biz-rtx4080", f"[mining g0] {tm[11:19]} 199.0 TH/s · 31 accepted · 0 rejected · up {up}")
        rows = [mining("2027-01-15T07:52:00Z", "1h 15m"),     # 2 min after the delete: Salad may still be stopping it
                mining("2027-01-15T07:57:00Z", "1h 20m"),
                mining("2027-01-15T07:58:00Z", "1h 21m")]
        # The round reads logs only while some group is live: biz-rtx4080-2, on m2.
        self.x.cfg["search"] = False
        g = self.group()
        g.update(created=T0 - 3600, machine="m2", pc_since=T0 - 3000, had_pc=True, has_pc=True, last_share=T0 - 60)
        self.http.on("GET", "/containers/biz-rtx4080-2", (200, {"current_state": {"status": "running"}}))
        self.http.on("GET", "/instances", (200, {"instances": [{"id": "i2", "machine_id": "m2", "state": "running"}]}))
        self.http.on("POST", "/log-entries", (200, {"items": [
            {"text_log": txt, "time": tm, "resource": {"labels": {"container_group_name": name, "machine_id": m}}}
            for tm, m, name, txt in rows]}))
        self.x.round()
        self.clock[0] += 60
        self.x.round()
        self.assertEqual(self.events().count("WARN biz-rtx4080: deleted at 07:50Z, but PC m9 still mines for it (199 TH/s, "
                                             "up 1h 21m); Salad no longer lists it, so it can't be stopped from here"), 1)

    # box.sh repeats a dead miner's last line, uptime and all: that is not a box mining.
    def test_a_deleted_groups_repeated_last_line_is_not_mining(self):
        self.x.cfg["search"] = False
        old = self.group()
        old.update(ended=T0 - 900, end_kind="released", had_pc=True)
        line = "[mining g0] 07:44:00 199.0 TH/s · 31 accepted · 0 rejected · up 1h 15m"
        rows = [("2027-01-15T07:57:00Z", "m9", "biz-rtx4080", line), ("2027-01-15T07:58:00Z", "m9", "biz-rtx4080", line)]
        g = self.group()
        g.update(created=T0 - 3600, machine="m2", pc_since=T0 - 3000, had_pc=True, has_pc=True, last_share=T0 - 60)
        self.http.on("GET", "/containers/biz-rtx4080-2", (200, {"current_state": {"status": "running"}}))
        self.http.on("GET", "/instances", (200, {"instances": [{"id": "i2", "machine_id": "m2", "state": "running"}]}))
        self.http.on("POST", "/log-entries", (200, {"items": [
            {"text_log": txt, "time": tm, "resource": {"labels": {"container_group_name": name, "machine_id": m}}}
            for tm, m, name, txt in rows]}))
        self.x.round()
        self.assertNotIn("still mines", self.events())

    def test_a_return_to_a_pc_mined_on_hours_ago_is_not_a_restart(self):
        self.x.cfg["search"] = False
        g = self.group()
        g.update(created=T0 - 30000, machine="m1", pc_since=T0 - 120, had_pc=True, has_pc=True, last_share=None)
        g["readings"] = [[T0 - 20000 + 60 * i, "m1", 199.0, i, f"up 0h {i}m"] for i in range(5)]   # an earlier stint
        self.http.on("GET", "/containers/biz-rtx4080", (200, {"current_state": {"status": "running"}}))
        self.http.on("GET", "/instances", (200, {"instances": [{"id": "i1", "machine_id": "m1", "state": "running"}]}))
        self.http.on("POST", "/log-entries", (200, {"items": [
            {"text_log": "[run] release v0.6.6 ready", "time": "2027-01-15T07:59:00Z",
             "resource": {"labels": {"container_group_name": "biz-rtx4080", "machine_id": "m1"}}}]}))
        self.x.round()
        self.assertIsNone(g.get("restart_t"))
        self.assertNotIn("RESTARTED", self.events())

    def test_a_pause_s_warm_up_reading_isn_t_kept_as_the_pc_s_rate(self):
        # 23 October: 3060-70 read 0.1 TH/s on 60b629b5 right after a pause; had the PC gone then, 32 TH/s would
        # have stood as its rate for 7 days, and the 7-day check would have moved good groups off it.
        g = self.group(pri="low")
        self.step(g, 1, "m1")
        self.mine(g, "m1", 199, 6)
        self.x.record_rates(self.clock[0])
        g["pause_since"] = g["paused_t"] = self.clock[0] + 30
        self.mine(g, "m1", 0.1, 1)
        self.x.record_rates(self.clock[0])
        self.mine(g, "m1", 199, 1)
        self.x.record_rates(self.clock[0])
        self.assertAlmostEqual(self.x.st["pc_rates"]["RTX 4080 (16 GB)"]["m1"][0], 199)

    def test_the_first_start_on_a_pc_is_not_a_restart(self):
        g = self.round_with([("2027-01-15T07:57:00Z", "[run] release v0.6.6 ready")], readings=False)
        self.assertIsNone(g.get("restart_t"))
        self.assertNotIn("RESTARTED", self.events())


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

    def test_a_margin_just_under_the_bar_isn_t_shown_as_the_bar(self):
        # 22 October: 3060-69 was moved off a PC as "+10%, under 10%".
        self.assertEqual(w.under(0.0996, 0.10), "+9.9%")
        self.assertEqual(w.under(0.096, 0.10), "+9.6%")
        self.assertEqual(w.under(0.094, 0.10), "+9%")
        self.assertEqual(w.under(-0.62, 0.10), "-62%")
        g = self.group(pri="medium")                 # $0.19/hr: 180 TH/s earns $0.2106, +9.78%
        self.step(g, 1, "m1")
        self.mine(g, "m1", 180, 5)
        self.assertIn("against its $0.190/hr (+9.7%, under 10%)", self.events())

    def test_a_pc_that_pays_under_10_percent_is_left(self):
        g = self.group(pri="medium")  # $0.19/hr; 170 TH/s earns $0.199: +4.5%
        self.step(g, 1, "m1")
        self.mine(g, "m1", 170, 5)
        self.assertTrue(self.http.made("POST", "/reallocate"))

    def pools(self, hm=0.90, kx=0.97):
        """HeroMiners crediting `hm` of its estimate (in 20 blocks) and Kryptex `kx`, from an ended group's PRL."""
        old = self.group(cls="RTX 2060 (6 GB)")
        old.update(ended=True, prl=20.0, prl_kx=10.0)
        self.x.st["pool_tot"] = {"t": self.clock[0], "paid": 10 * hm, "balance": 0.0, "unconfirmed": 0.0,
                                 "blocks": 20, "hashes": 0, "ths_1h": 0}
        self.x.st["kx_tot"] = {"t": self.clock[0], "total": 0.0, "confirmed": 0.0, "unconfirmed": 0.0, "paid": 10 * kx}
        self.assertAlmostEqual((self.x.credit_ratio(), self.x.kx_ratio()), (hm, kx))

    def test_each_box_is_judged_at_its_own_pool_s_credit_ratio(self):
        # 22 October: the rules valued every box at HeroMiners' 0.93 while every box mined on Kryptex at 0.97, so
        # boxes at a real +10-14% read under 10% and were left: 63 of 125 margin moves and deletions in 48 hours.
        # 193 TH/s on a $0.19/hr 4080: +13% at Kryptex's 0.97, +6% at HeroMiners' 0.90.
        self.pools()
        k, h = self.group(pri="medium"), self.group(pri="medium")
        k["kx"] = True
        for g, m in ((k, "mk"), (h, "mh")):
            self.step(g, 1, m)
            self.mine(g, m, 193, 10)
        self.assertFalse(self.http.made("POST", "/instances/inst-mk/reallocate"))   # Kryptex box kept
        self.assertTrue(self.http.made("POST", "/instances/inst-mh/reallocate"))    # HeroMiners box left
        self.assertAlmostEqual(self.x.pc_margin(k), 1 - 0.19 / (193 * USD_TH_HR * 0.97))

    def test_a_new_group_and_a_climb_are_priced_at_the_pool_they_mine_on(self):
        self.pools()
        rate = 193
        self.x.st["pc_rates"] = {"RTX 4080 (16 GB)": {"m1": [rate, self.clock[0]]}}
        self.x.cfg["kryptex_groups"] = 10              # a new group starts on Kryptex
        self.assertTrue(self.x.new_on_kx())
        self.assertGreater(self.x.margin("RTX 4080 (16 GB)", "medium"), 0.10)
        self.assertLess(self.x.margin("RTX 4080 (16 GB)", "medium", kx=False), 0.10)
        self.x.cfg["kryptex_groups"] = 0               # a new group starts on HeroMiners
        self.assertLess(self.x.margin("RTX 4080 (16 GB)", "medium"), 0.10)
        # A Kryptex group with no PC at low for 10 minutes climbs to medium, which clears 10% on Kryptex.
        g = self.group(pri="low")
        g.update(kx=True, status="deploying")
        self.step(g, 11)
        self.assertEqual(g["pri"], "medium")
        self.assertFalse(g["ended"])

    def left_at(self, ths, pri="medium", kx=False, m="m1"):
        """A 4080 group that mines at `ths` on `m` and leaves it for its margin; the bad-list entry it leaves."""
        g = self.group(pri=pri)
        g["kx"] = kx
        self.step(g, 1, m)
        self.mine(g, m, ths, 5)
        self.assertIn("under 10%", self.x.st["bad_pcs"][m]["why"])
        return self.x.st["bad_pcs"][m]

    def test_a_pc_left_for_its_rate_is_kept_by_a_group_that_rate_clears_for(self):
        # 22 October: a PC left at high was refused at a cheaper priority where it cleared, 21 of 681 bad-list moves to
        # 23 October at PRL's 02:34Z price.
        bad = self.left_at(170)                    # $0.199/hr at medium's $0.19: +4.5%, left
        self.assertEqual(bad["ths"], 170)
        g = self.group(pri="low")                  # $0.15/hr: 170 TH/s is +25%
        n = len(self.http.made("POST", "/reallocate"))
        self.step(g, 1, "m1")
        self.assertIn("NOTE biz-rtx4080-2 keeps PC m1: biz-rtx4080 left it at 170 TH/s, which clears 10% at this "
                      "group's $0.150/hr", self.events())
        self.mine(g, "m1", 170, 10)               # and its own readings keep it there
        self.assertEqual(len(self.http.made("POST", "/reallocate")), n)
        self.assertEqual(g["machine"], "m1")

    def test_a_pc_left_for_its_rate_is_still_left_by_a_group_it_doesn_t_clear_for(self):
        self.left_at(170)
        g = self.group(pri="medium")               # the same price: still +4.5%
        n = len(self.http.made("POST", "/reallocate"))
        self.step(g, 1, "m1")
        self.assertEqual(len(self.http.made("POST", "/reallocate")), n + 1)
        self.assertIn("(biz-rtx4080 left this PC at", self.events())
        self.assertNotIn("NOTE biz-rtx4080-2 keeps", self.events())

    def test_a_pc_left_on_herominers_is_kept_by_a_kryptex_group_it_clears_for(self):
        # 193 TH/s on a $0.19/hr 4080: +6.5% at HeroMiners' 0.90, +13% at Kryptex's 0.97.
        self.pools()
        self.left_at(193, kx=False)
        g = self.group(pri="medium")
        g["kx"] = True
        n = len(self.http.made("POST", "/reallocate"))
        self.step(g, 1, "m1")
        self.assertEqual(len(self.http.made("POST", "/reallocate")), n)
        self.assertIn("NOTE biz-rtx4080-2 keeps PC m1", self.events())

    def test_an_entry_from_before_the_rate_was_kept_is_read_from_its_reason(self):
        self.x.st.setdefault("bad_pcs", {})["m1"] = {"t": self.clock[0], "group": "biz-rtx4080-9", "why": (
            "earns $0.199/hr at 170 TH/s on PC m1 against its $0.190/hr (+4%, under 10%)")}
        self.assertEqual(w.W.bad_ths(self.x.st["bad_pcs"]["m1"]), 170.0)
        self.assertIsNone(w.W.bad_ths({"t": 0, "group": "g", "why": "no mining reading 10 min after getting PC m1"}))
        self.assertIsNone(w.W.bad_ths({"t": 0, "group": "g", "why": "PC blocks outbound port 1200", "ports": [1200]}))
        g = self.group(pri="low")
        self.step(g, 1, "m1")
        self.assertFalse(self.http.made("POST", "/reallocate"))
        self.assertIn("NOTE biz-rtx4080 keeps PC m1: biz-rtx4080-9 left it at 170 TH/s", self.events())

    def test_a_pc_left_for_another_reason_is_left_whatever_the_price(self):
        self.x.st.setdefault("bad_pcs", {})["m1"] = {"t": self.clock[0], "group": "biz-rtx4080-9",
                                                     "why": "no mining reading 10 min after getting PC m1"}
        g = self.group(pri="batch")
        self.step(g, 1, "m1")
        self.assertTrue(self.http.made("POST", "/instances/inst-m1/reallocate"))

    def test_a_pc_whose_measured_rate_fails_is_left_after_its_bad_list_entry_expires(self):
        # Bad-list entries last 24 h; a PC's measured rate is kept 7 days. 150 TH/s at medium's $0.19 is -8%.
        self.x.st["pc_rates"] = {"RTX 4080 (16 GB)": {"m1": [150.0, self.clock[0] - 2 * 86400]}}
        g = self.group(pri="medium")
        self.step(g, 1, "m1")
        self.assertTrue(self.http.made("POST", "/instances/inst-m1/reallocate"))
        self.assertIn("off PC m1 (PC m1 made 150 TH/s 48 h ago, -8% at this group's $0.190/hr, under 10%); "
                      "not counted toward its 3", self.events())
        self.assertEqual(g["reallocs"], 0)

    def test_a_pc_whose_measured_rate_clears_for_this_group_is_kept(self):
        self.x.st["pc_rates"] = {"RTX 4080 (16 GB)": {"m1": [150.0, self.clock[0] - 2 * 86400]}}
        g = self.group(pri="batch")                # $0.11/hr: 150 TH/s is +37%
        self.step(g, 1, "m1")
        self.assertFalse(self.http.made("POST", "/reallocate"))

    def test_a_measured_rate_older_than_rate_days_is_not_used(self):
        self.x.st["pc_rates"] = {"RTX 4080 (16 GB)": {"m1": [150.0, self.clock[0] - 8 * 86400]}}
        g = self.group(pri="medium")
        self.step(g, 1, "m1")
        self.assertFalse(self.http.made("POST", "/reallocate"))

    def monero(self, *khs):
        """Monero on, at $0.005 a kH/s-hour, with a live box the pools saw at each of `khs` kH/s just now."""
        self.x.cfg.update(xmr_wallet="88" + "x" * 93, xmr_groups=["*"])
        self.x.st["xmr_mkt"] = {"t": self.clock[0], "price": 300.0, "per_hash": 0, "usd_khs_hr": 0.005}
        for k in khs:
            o = self.group(cls="RTX 2060 (6 GB)")
            o.update(has_pc=True, xmr_khs=k, xmr_seen=self.clock[0])

    def test_the_rules_count_a_box_s_monero(self):
        # 22 October: the rules counted Pearl only, so boxes that cleared 10% with their Monero were left.
        # 175 TH/s at medium's $0.19: $0.205 from Pearl is +7%; with $0.010 of Monero, $0.215 is +12%.
        self.monero(1.0, 2.0, 6.0)                  # the median box: 2 kH/s, $0.010/hr (the mean is 3)
        self.assertAlmostEqual(self.x.xmr_usd(), 0.010)
        g = self.group(pri="medium")
        self.assertTrue(g["xmr"])
        self.step(g, 1, "m1")
        self.mine(g, "m1", 175, 10)
        self.assertFalse(self.http.made("POST", "/instances/inst-m1/reallocate"))
        self.assertAlmostEqual(self.x.pc_margin(g), 1 - 0.19 / (175 * USD_TH_HR + 0.010))
        self.assertAlmostEqual(self.x.margin("RTX 4080 (16 GB)", "medium", rate=175),
                               1 - 0.19 / (175 * USD_TH_HR + 0.010))

    def test_a_box_under_the_margin_with_its_monero_is_still_left(self):
        self.monero(2.0)
        g = self.group(pri="medium")
        self.step(g, 1, "m1")
        self.mine(g, "m1", 165, 5)                  # $0.193 + $0.010 is +6%
        self.assertTrue(self.http.made("POST", "/instances/inst-m1/reallocate"))
        self.assertIn("earns $0.203/hr at 165 TH/s (Monero $0.010 of it) on PC m1 against its $0.190/hr (+6%, "
                      "under 10%)", self.events())
        self.assertEqual(self.x.st["bad_pcs"]["m1"]["ths"], 165)

    def test_the_report_values_kryptex_monero_at_what_kryptex_pays(self):
        # 200 TH/s ($0.234) and 100 kH/s on Kryptex at $0.002 a kH/s-hour, 94% of the network's pay against
        # HeroMiners' 99.1%: $0.190, so $0.424 earned against $0.19 is +$0.23 (+55%); at 99.1% it read +$0.24.
        self.x.cfg.update(xmr_wallet="88" + "x" * 93, xmr_groups=["*"])
        self.x.st["xmr_mkt"] = {"t": self.clock[0], "price": 300.0, "per_hash": 0, "usd_khs_hr": 0.002}
        g = self.group(pri="medium")
        self.step(g, 1, "m1")
        self.mine(g, "m1", 200, 5)
        g.update(kxx_khs=100.0, kxx_seen=self.clock[0])
        self.x.write_report()
        with open(os.path.join(TMP, "report.md"), encoding="utf-8") as f:
            self.assertIn("| 0.19 | +0.23* (+55%*) |", f.read())

    def test_the_median_of_an_even_count_is_the_mean_of_the_middle_two(self):
        self.monero(1.0, 3.0)
        self.assertAlmostEqual(self.x.xmr_usd(), 0.010)

    def test_kryptex_monero_is_valued_at_what_kryptex_pays(self):
        self.monero()
        o = self.group(cls="RTX 2060 (6 GB)")
        o.update(has_pc=True, kxx_khs=2.0, kxx_seen=self.clock[0])
        self.assertAlmostEqual(self.x.xmr_usd(), 2.0 * 0.94 / 0.991 * 0.005)

    def test_a_box_long_on_its_pc_counts_its_own_monero(self):
        # The median stands in only while the pools' 30-minute averages still hold the old PC: after that a box
        # whose Monero doesn't reach the pool counts none, and one that mines more counts its own.
        self.monero(2.0, 2.0, 2.0)
        g = self.group(pri="medium")
        g.update(has_pc=True, machine="m1", pc_since=self.clock[0])
        for i in range(8):                           # reads every 5 minutes for 40 minutes, at 0.2 kH/s
            self.clock[0] += 300
            g.update(xmr_khs=0.2, xmr_seen=self.clock[0])
            for o in self.x.st["groups"].values():
                if o is not g:
                    o["xmr_seen"] = self.clock[0]
            self.x.record_rates(self.clock[0])
        self.assertAlmostEqual(self.x.xmr_usd(g), 0.2 * 0.005)
        # A new PC: its class's figures again (its own on m1, not the 2060s' median) until 30 minutes of its own.
        g.update(machine="m2", pc_since=self.clock[0])
        self.assertAlmostEqual(self.x.xmr_usd(g), 0.2 * 0.005)

    def test_a_box_the_pools_never_see_counts_no_monero_after_45_minutes(self):
        self.monero(2.0, 2.0, 2.0)
        g = self.group(pri="medium")
        g.update(has_pc=True, machine="m1", pc_since=self.clock[0])
        def others_seen():
            for o in self.x.st["groups"].values():
                if o is not g:
                    o["xmr_seen"] = self.clock[0]
        self.clock[0] += 44 * 60
        others_seen()
        self.assertAlmostEqual(self.x.xmr_usd(g), 2.0 * 0.005)   # still the median: it may not show yet
        self.clock[0] += 2 * 60
        others_seen()
        self.assertEqual(self.x.xmr_usd(g), 0.0)
        g.update(xmr_khs=0.4, xmr_seen=self.clock[0])                # seen at last, 46 minutes in: its own figure
        self.x.record_rates(self.clock[0])
        self.assertAlmostEqual(self.x.xmr_usd(g), 0.4 * 0.005)

    def test_a_new_box_s_monero_comes_from_its_own_class(self):
        # 23 October: the fleet median (3060s at 3–5 kH/s) credited a new 4060, whose PCs make 0.5–1.3, with about
        # twice its Monero, so it kept a PC it left 8 minutes later (OPEN.md #65).
        self.monero(4.0, 4.0, 4.0)                                  # three 2060s at 4 kH/s: the fleet median
        t = self.clock[0]
        self.x.st["xmr_rates"] = {"RTX 4080 (16 GB)": {"m1": [0.5, t], "m2": [1.0, t], "m3": [1.3, t]}}
        self.assertAlmostEqual(self.x.xmr_usd(cls="RTX 4080 (16 GB)"), 1.0 * 0.005)
        self.assertAlmostEqual(self.x.xmr_usd(cls="RTX 4070 (12 GB)"), 4.0 * 0.005)   # none stored: the fleet's
        self.assertAlmostEqual(self.x.xmr_usd(), 4.0 * 0.005)
        g = self.group(pri="medium")
        g.update(has_pc=True, machine="m9", pc_since=self.clock[0])   # on a PC, no reads of its own yet
        self.assertAlmostEqual(self.x.xmr_usd(g), 1.0 * 0.005)
        self.assertAlmostEqual(self.x.margin("RTX 4080 (16 GB)", "medium", rate=175),
                               1 - 0.19 / (175 * USD_TH_HR + 0.005))

    def test_a_box_s_own_monero_is_stored_for_its_class_for_rate_days(self):
        self.monero()
        g = self.group(pri="medium")
        g.update(has_pc=True, machine="m1", pc_since=self.clock[0])
        for i in range(8):                           # reads every 5 minutes for 40 minutes, at 0.6 kH/s
            self.clock[0] += 300
            g.update(xmr_khs=0.6, xmr_seen=self.clock[0])
            self.x.record_rates(self.clock[0])
        self.assertAlmostEqual(self.x.st["xmr_rates"]["RTX 4080 (16 GB)"]["m1"][0], 0.6)
        g.update(has_pc=False, machine=None, pc_since=None)
        self.clock[0] += 7 * 86400
        self.x.record_rates(self.clock[0])
        self.assertEqual(self.x.st["xmr_rates"]["RTX 4080 (16 GB)"], {})

    def test_monero_counts_only_for_a_box_that_mines_it_and_from_fresh_figures(self):
        self.assertEqual(self.x.xmr_usd(), 0.0)    # Monero off
        self.monero(2.0)
        g = self.group()
        g["xmr"] = False
        self.assertEqual(self.x.xmr_usd(g), 0.0)
        self.clock[0] += 16 * 60                    # the pools last saw the Monero box 16 minutes ago
        self.assertEqual(self.x.xmr_usd(), 0.0)

    def test_a_box_is_priced_at_the_pool_its_pc_mines_on(self):
        # A Kryptex group whose box was seen mining on HeroMiners at its PC (a command change that landed as Salad
        # handed it the PC: 3060-45 on 19 October) is judged at HeroMiners' ratio there.
        self.pools()
        g = self.group(pri="medium")
        g["kx"] = True
        self.step(g, 1, "mh")
        g["pc_pool"] = {"mh": False}                   # its shares were seen at HeroMiners on this PC
        self.mine(g, "mh", 193, 10)
        self.assertTrue(self.http.made("POST", "/instances/inst-mh/reallocate"))

    def test_kryptex_s_ratio_holds_while_its_api_is_down(self):
        # The estimate keeps growing while Kryptex's totals can't be read; the ratio stays at its last read.
        self.pools()
        self.x.st["kx_tot"]["prl_kx"] = 10.0
        live = self.group()
        live["prl_kx"] = 10.0                          # mined since the last read
        self.assertAlmostEqual(self.x.kx_ratio(), 0.97)

    def test_the_search_checks_the_margin_again_after_a_create_fills_kryptex(self):
        # Two slots, one Kryptex slot left: the second group starts on HeroMiners, so it needs to clear 10% there.
        self.pools()
        self.x.cfg.update(kryptex_groups=1, max_active=2, max_new_per_hour=5)
        self.http.on("POST", "availability", (200, {f"available_gpu_{p}": 1 for p in w.PRIS}))
        self.x.st["pc_rates"] = {"RTX 4080 (16 GB)": {m: [110, self.clock[0]] for m in ("m1", "m2", "m3")}}   # batch: +12% Kryptex, +5% HM
        self.assertGreater(self.x.margin("RTX 4080 (16 GB)", "batch", kx=True), 0.10)
        self.assertLess(self.x.margin("RTX 4080 (16 GB)", "batch", kx=False), 0.10)
        self.x.search()
        live = [g for g in self.x.st["groups"].values() if not g["ended"]]
        for g in live:
            self.assertGreaterEqual(self.x.margin(g["class"], g["pri"], kx=bool(g["kx"])), 0.10, g["name"])
        self.assertTrue(live)
        # The CREATED line gives the margin at the pool the group was made for.
        for g in live:
            line = next(l for l in self.events().splitlines() if f"CREATED {g['name']}:" in l)
            self.assertIn(f"{self.x.margin(g['class'], g['pri'], kx=bool(g['kx'])) * 100:+.0f}% margin", line)

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

    def test_class_that_never_paid_is_deleted_released_and_the_pc_barred(self):
        # Left for its rate, the PC was at fault: it's barred, and the class may be tried again on another PC.
        g = self.group(cls="RTX 4070 (12 GB)", pri="low")  # $0.09/hr; 50 TH/s earns $0.059
        self.step(g, 1, "m1")
        self.mine(g, "m1", 50, 5)
        self.assertEqual(g["end_kind"], "released")
        self.assertEqual(self.x.st["bad_pcs"]["m1"]["group"], g["name"])
        self.assertEqual(self.x.st["pc_rates"]["RTX 4070 (12 GB)"]["m1"][0], 50.0)


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
    def test_reallocates_without_using_a_move_then_counts_after_6(self):
        # 22 October: 4060-54 was deleted on its fourth PC for a blocked port, the PC's fault, not the group's.
        g = self.group()
        self.step(g, 1, "m1")
        g["blocked"].append("m1")
        self.step(g, 1, "m1")
        self.assertEqual(g["reallocs"], 0)
        self.assertIn("REALLOCATED biz-rtx4080 off PC m1 (PC blocks outbound port 1200); not counted toward its 3",
                      self.events())
        # A group that keeps landing on blocked PCs counts them from the seventh, so it can't move for ever.
        g["reallocs"], g["port_moves"] = 3, 6
        self.step(g, 11, "m2")
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

    def test_a_pc_left_on_an_earlier_day_says_which_day(self):
        # At 15:04Z on 23 Oct an event said a PC was left "at 15:54Z": it was 15:54Z the day before.
        a = self.group()
        self.step(a, 1, "m1")
        a["blocked"].append("m1")
        self.step(a, 1, "m1")
        self.x.st["bad_pcs"]["m1"]["t"] = T0 - (16 * 3600 + 6 * 60)   # 15:54Z on 14 Jan; the clock is 15 Jan
        b = self.group()
        self.step(b, 1, "m1")
        self.assertIn("(biz-rtx4080 left this PC at 14 Jan 15:54Z: PC blocks outbound port 1200)", self.events())

    def test_old_pc_still_showing_after_a_reallocation_is_not_a_new_pc(self):
        g = self.group()
        self.step(g, 1, "m1")
        g["blocked"].append("m1")
        self.step(g, 1, "m1")
        self.step(g, 1, "m1")  # Salad still lists the old PC
        self.assertEqual(g["port_moves"], 1)
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
        self.assertIn("POST /start -> HTTP 200\n", self.events())

    def test_a_refused_start_logs_salad_s_reason(self):
        self.http.on("POST", "/start$", (400, {"type": "invalid_state", "detail": "the image is still downloading"}))
        g = self.group(status="stopped")
        self.step(g, 1, None)
        self.assertIn("POST /start -> HTTP 400 {'type': 'invalid_state', 'detail': 'the image is still downloading'}",
                      self.events())

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

    def test_the_search_trusts_salad_s_count_of_replicas_in_use(self):
        # 22 October: Salad counted one replica more than we had at times all day, so every search sent creates it then
        # refused with created_replicas_quota_exceeded. With 1 group and 9 of 10 replicas in use, there's room for 1.
        self.http.on("POST", "/availability/", (200, {"available_gpu_low": 5}))
        self.http.on("GET", "/quotas", (200, {"container_groups_quotas": {"container_replicas_quota": 10,
                                                                         "container_replicas_used": 9}}))
        self.group(cls="RTX 4070 (12 GB)", pri="low")
        n = len(self.http.made("POST", "/containers$"))
        self.x.cfg["max_active"] = 10
        self.x.search()
        self.assertEqual(len(self.http.made("POST", "/containers$")) - n, 1)

    def test_the_search_stops_at_the_first_quota_refusal(self):
        self.http.on("POST", "/availability/", (200, {"available_gpu_low": 5}))
        self.http.on("GET", "/quotas", (200, {"container_groups_quotas": {"container_replicas_quota": 10}}))
        self.http.on("POST", "/containers$", (400, {"type": "created_replicas_quota_exceeded"}))
        self.x.cfg["max_active"] = 10
        self.x.search()
        self.assertEqual(len(self.http.made("POST", "/containers$")), 1)   # not one refused create per class
        self.assertEqual(self.events().count("CREATE FAILED"), 1)
        self.http.on("POST", "/containers$", (201, {}))
        self.x.search()                                                    # the next search tries again
        self.assertGreater(len(self.http.made("POST", "/containers$")), 2)

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

    def held_4080(self, tries="low", avail=("medium",), placed="medium"):
        """A 4080 that last got a PC at `placed`, then three groups with no PC at `tries` (a priority or a list)."""
        self.http.on("POST", "/availability/", lambda b: (200, {f"available_gpu_{p}": 5 for p in avail}
                                                          if b["gpu_classes"][0] == "c4080" else {}))
        self.http.on("GET", "/quotas", (200, {"container_groups_quotas": {"container_replicas_quota": 10}}))
        self.x.cfg.update(max_active=10, retry_after_min=10)
        p = self.group(cls="RTX 4080 (16 GB)", pri=placed)
        p.update(had_pc=True, ended=self.clock[0], end_kind="released")
        for pri in ([tries] * 3 if isinstance(tries, str) else tries):
            self.clock[0] += 900
            g = self.group(cls="RTX 4080 (16 GB)", pri=pri)
            g["ended"], g["end_kind"] = self.clock[0], "released"
        self.clock[0] += 900
        n = len(self.http.made("POST", "/containers$"))
        self.x.search()
        return [(c[2]["name"], c[2]["container"]["priority"]) for c in self.http.made("POST", "/containers$")[n:]]

    def test_a_class_held_for_cheaper_tries_is_tried_at_the_priority_it_places_at(self):
        self.assertEqual(self.held_4080(), [("biz-rtx4080-5", "medium")])

    def test_not_while_that_priority_fails_the_margin(self):
        self.x.st["classes"] = json.loads(json.dumps(CLASSES))               # a copy: CLASSES is shared
        self.x.st["classes"]["RTX 4080 (16 GB)"]["prices"]["medium"] = 0.23   # medium no longer clears
        self.assertEqual(self.held_4080(avail=("medium", "low")), [])

    def test_a_class_with_a_no_pc_try_at_its_priority_stays_held(self):
        self.assertEqual(self.held_4080(tries=["low", "medium", "low"]), [])

    def test_a_class_that_keeps_getting_no_pc_is_left_out(self):
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

    def test_a_class_that_keeps_getting_no_pc_doesnt_take_a_slot_nothing_else_wants(self):
        # 22 October: such classes went last but still took any free slot, so for 324 of the 722-723 minutes every
        # allowed slot was full that day a group of one held a slot, and 3 times a class that gets PCs was refused.
        avail = {"c4080": {"available_gpu_low": 5}}            # only the 4080 has a free GPU
        self.http.on("POST", "/availability/", lambda b: (200, avail.get(b["gpu_classes"][0], {})))
        self.http.on("GET", "/quotas", (200, {"container_groups_quotas": {"container_replicas_quota": 10}}))
        self.x.cfg.update(max_active=10, retry_after_min=10)
        for _ in range(3):
            g = self.group(cls="RTX 4080 (16 GB)", pri="low")
            g["ended"], g["end_kind"] = self.clock[0], "released"   # never had a PC
            self.clock[0] += 3600
        n = len(self.http.made("POST", "/containers$"))
        self.x.search()
        self.assertEqual(len(self.http.made("POST", "/containers$")), n)
        self.assertIn("SEARCH found nothing to create", self.events())
        self.clock[0] += 121 * 60                                # 3 h after the last one ended
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
        with open(os.path.join(TMP, "report.md"), encoding="utf-8") as f:
            report = f.read()
        # Work worth 8 PRL at the network's rate; in under 10 blocks the pool's credit ratio isn't used yet.
        self.assertIn("| **8.000\\*** |\n", report)  # the All row
        self.assertIn("| ⛏️ PRL | 0 TH/s | 0.500 | 4.000 | 1.500 |", shown(report))  # straight from the pool: no stars
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
        with open(os.path.join(TMP, "report.md"), encoding="utf-8") as f:
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

    def test_the_heading_says_how_many_days_the_credit_lasts(self):
        self.group(pri="low")["has_pc"] = True                      # $0.15/hr against $94.00 of credit: 627 h
        self.assertIn(" · Salad $94.00* left (26* days)\n", self.report(cfg={"max_active": 2}))

    def test_under_a_day_of_credit_shows_in_hours(self):
        self.assertEqual([w.lasts(h) for h in (627, 36, 24, 23.6, 5)], ["26 days", "2 days", "1 day", "24 h", "5 h"])

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
        with open(os.path.join(TMP, "report.md"), encoding="utf-8") as f:
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
        for m in ("m1", "m2", "m3"):
            g = self.group(pri="medium")
            g["readings"] = [[self.clock[0] + 60 * i, m, 180.0, 0] for i in range(6)]
            g["ended"] = self.clock[0]
        self.x.record_rates(self.clock[0] + 600)
        self.assertEqual(self.x.rate("RTX 4080 (16 GB)"), 180)

    def test_one_or_two_slow_pcs_don_t_lower_the_benchmark(self):
        # OPEN.md #62: a 5090 left at 131 TH/s would have set the class's rate to 131 against its 418.
        self.mined("RTX 4080 (16 GB)", [150], pri="medium")
        self.assertEqual(self.x.rate("RTX 4080 (16 GB)"), 199)
        self.mined("RTX 4080 (16 GB)", [160], pri="medium")
        self.assertEqual(self.x.rate("RTX 4080 (16 GB)"), 199)
        self.mined("RTX 4080 (16 GB)", [170], pri="medium")   # the third PC: now the median rules
        self.assertEqual(self.x.rate("RTX 4080 (16 GB)"), 160)

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
        with open(os.path.join(TMP, "report.md"), encoding="utf-8") as f:
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
        with open(os.path.join(TMP, "report.md"), encoding="utf-8") as f:
            report = f.read()
        # They ended before the pools' counts were kept: 1 PRL each by the ledger's estimate, before 10 blocks.
        self.assertIn(f"| Ended{w.NB}×3 | — | — | — | 3.000* |", report)

    def test_estimates_are_starred_and_api_figures_are_not(self):
        a = self.group(pri="medium")
        self.step(a, 1, "m1")
        self.mine(a, "m1", 200, 5)
        self.x.write_report()
        with open(os.path.join(TMP, "report.md"), encoding="utf-8") as f:
            report = f.read()
        # TH/s (the miner's log) and $/hr (Salad's price list) come from an API; the rest is our own count.
        self.assertIn(f"| ⛏️{w.NB}4080{w.NB}med | 200 | 0.19 | +0.04* (+19%*) |", report)
        self.assertRegex(report, r" · Salad \$[\d.]+\* left \(\d+\* (h|days?)\)\n")

    def test_a_box_with_no_reading_after_15_min_counts_again(self):
        a = self.group(pri="medium")
        self.step(a, 1, "m1")
        self.clock[0] += 16 * 60
        self.x.write_report()
        with open(os.path.join(TMP, "report.md"), encoding="utf-8") as f:
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

    def test_after_a_day_the_rate_a_pc_was_left_at_still_counts(self):
        # The bad list forgets a PC after 24 h, but the rate it was left at is kept 7 days (pc_rates), even when it
        # was left at its first judgment: until the 22 October review it was never recorded then.
        a = self.group(pri="medium")
        self.step(a, 1, "m1")
        self.mine(a, "m1", 120, 6)                   # $0.140/hr against $0.19: left at its first judgment
        self.assertEqual(self.x.st["pc_rates"]["RTX 4080 (16 GB)"]["m1"][0], 120)
        self.clock[0] += 25 * 3600
        b = self.group(pri="medium")
        self.step(b, 1, "m1")
        self.assertEqual(len(self.http.made("POST", "/instances/inst-m1/reallocate")), 2)
        self.assertIn("(PC m1 made 120 TH/s 25 h ago,", self.events())
        self.assertEqual(b["reallocs"], 0)
        c = self.group(pri="batch")                  # at $0.11/hr 120 TH/s is +22%: the PC is kept
        self.step(c, 1, "m1")
        self.assertEqual(len(self.http.made("POST", "/instances/inst-m1/reallocate")), 2)

    def test_a_pc_salad_took_back_isnt_bad(self):
        a = self.group(pri="medium")
        self.step(a, 1, "m1")
        self.mine(a, "m1", 200, 6)
        self.step(a, 1, None)       # Salad takes the PC back
        self.assertEqual(self.x.st.get("bad_pcs", {}), {})

    def test_one_low_reading_doesnt_move_a_box_but_two_do(self):
        # Rule 5 judges the median of the last 3 readings: on 20 October 5070-31 was moved off a PC at 133 TH/s for
        # one reading of 3 between two of 133.
        g = self.group(pri="medium")               # $0.19/hr: 200 TH/s is +19%, 120 is -35%
        self.step(g, 1, "m1")
        self.mine(g, "m1", 200, 5)
        self.mine(g, "m1", 50, 1)                  # one bad minute: the mean of 200, 200, 50 would be under 10%
        self.mine(g, "m1", 200, 3)
        self.assertEqual(self.http.made("POST", "/reallocate"), [])
        self.assertEqual(self.x.pc_rate(g), 200)
        self.mine(g, "m1", 120, 2)                 # the PC really slows: left on the second low reading
        self.assertEqual(len(self.http.made("POST", "/reallocate")), 1)
        self.assertIn("at 120 TH/s on PC m1", self.events())

    def test_a_pc_salad_takes_back_three_times_in_a_day_goes_on_the_bad_list(self):
        # 17-19 October: 3323a855 was handed to 3090 Ti groups 19 times in 38 hours and taken back within minutes
        # nearly every time.
        a = self.group(pri="medium")
        for i in range(3):
            self.step(a, 1, "m1")
            self.step(a, 3, None)    # taken back 3 minutes after it was handed over
            self.assertEqual("m1" in self.x.st.get("bad_pcs", {}), i == 2)
        self.assertIn("NOTE PC m1 goes on the bad list: Salad took it back within 10 min of handing it over 3 times "
                      "in 24 h", self.events())
        self.assertNotIn("m1", self.x.st["pc_drops"])
        b = self.group(pri="medium")
        self.step(b, 1, "m1")        # the next group that gets it moves off at once, uncounted
        self.assertEqual(len(self.http.made("POST", "/instances/inst-m1/reallocate")), 1)
        self.assertIn("REALLOCATED biz-rtx4080-2 off PC m1 (biz-rtx4080 left this PC at", self.events())
        self.assertEqual(b["reallocs"], 0)

    def test_only_quick_drops_within_a_day_count(self):
        a = self.group(pri="medium")
        self.step(a, 1, "m1")
        self.step(a, 3, None)                        # a quick drop
        self.step(a, 1, "m1")
        self.mine(a, "m1", 200, 12)
        self.step(a, 1, None)                        # taken back after 13 minutes: not a drop
        self.assertEqual(len(self.x.st["pc_drops"]["m1"]), 1)
        self.clock[0] += 25 * 3600                   # the first drop is over a day old
        b = self.group(pri="medium")
        for _ in range(2):
            self.step(b, 1, "m1")
            self.step(b, 2, None)
        self.assertEqual(len(self.x.st["pc_drops"]["m1"]), 2)
        self.assertNotIn("m1", self.x.st.get("bad_pcs", {}))


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


class TestSafeTrade(Base):
    BAL = [{"currency": "pearl", "balance": "1.1461474", "locked": "0.5"},
           {"currency": "usdt", "balance": "9.0505421", "locked": "0"},
           {"currency": "xmr", "balance": "0.000008", "locked": "0"}]

    def fill(self, fid, market="pearlusdt", amount=2.0, price=1.9, order_side="sell", side="buy", order=None):
        return {"id": fid, "order_id": order or fid * 10, "market": market, "price": str(price), "amount": str(amount),
                "total": str(amount * price), "side": side, "order_side": order_side, "fee_currency": "usdt",
                "fee": str(amount * price / 1000), "created_at": w.iso(self.clock[0] - 3600)}

    def setUp(self):
        super().setUp()
        os.environ["SAFETRADE_API_KEY"], os.environ["SAFETRADE_API_SECRET"] = "k" * 16, "s" * 32
        self.addCleanup(os.environ.pop, "SAFETRADE_API_KEY", None)
        self.addCleanup(os.environ.pop, "SAFETRADE_API_SECRET", None)
        self.x.cfg.update(xmr_wallet="4" + "a" * 94)
        # Newest first, as SafeTrade lists them. The buy of USDC isn't a sale.
        self.fills = [self.fill(5), self.fill(4, amount=1.0, price=2.0), self.fill(3, price=1.5),
                      self.fill(2, market="xmrusdt", amount=0.01, price=620.1),
                      self.fill(1, market="usdcusdt", amount=10.0, price=0.999, order_side="buy", side="sell")]
        self.http.on("GET", r"safetrade\.com/api/v2/trade/account/balances/spot$", (200, self.BAL))
        self.http.on("GET", r"safetrade\.com/api/v2/trade/market/trades\?", self.page)
        # Deposits by coin: two pages of PRL (one rejected), one of XMR.
        self.deposits = {"pearl": [{"amount": "1.5", "status": "collected"}] * 100 + [{"amount": "2.0", "status": "collected"},
                                   {"amount": "9.0", "status": "rejected"}],
                         "xmr": [{"amount": "0.010368", "status": "collected"}]}
        self.http.on("GET", r"safetrade\.com/api/v2/trade/account/deposits\?", self.deposit_page)
        # The USDC bought (fill 1) withdrawn, some USDT withdrawn, and a failed withdrawal that doesn't count.
        self.http.on("GET", r"safetrade\.com/api/v2/trade/account/withdraws\?", (200, [
            {"currency_id": "usdt", "amount": "2.0", "fee": "1", "status": "succeed"},
            {"currency_id": "usdt", "amount": "50.0", "fee": "1", "status": "failed"},
            {"currency_id": "usdc", "amount": "9.0", "fee": "0.9", "status": "succeed"}]))
        st = self.x.st
        st["pool_tot"] = {"t": T0, "ths_1h": 0, "balance": 0.744, "unconfirmed": 0.0, "paid": 151.622, "blocks": 0}
        st["xmr_tot"] = {"khs_1h": 5.1, "balance": 0.000949, "pending": 0.000094, "paid": 0.010368}

    def page(self, _):
        n = int(re.search(r"page=(\d+)", self.http.calls[-1][1]).group(1))
        return 200, self.fills[(n - 1) * 100:n * 100]

    def deposit_page(self, _):
        url = self.http.calls[-1][1]
        coin, n = re.search(r"currency=(\w+)", url).group(1), int(re.search(r"page=(\d+)", url).group(1))
        return 200, self.deposits.get(coin, [])[(n - 1) * 100:n * 100]

    def row(self, coin):
        return next(shown(l) for l in self.report().splitlines() if l.startswith(f"| {coin} |"))

    def report(self):
        self.x.write_report()
        with open(os.path.join(TMP, "report.md"), encoding="utf-8") as f:
            return f.read()

    def test_off_without_keys(self):
        del os.environ["SAFETRADE_API_KEY"]
        self.x.safetrade()
        self.assertFalse(self.http.made("GET", "safetrade"))
        self.assertNotIn("SafeTrade", self.report())

    def test_one_table_from_pools_paid_to_the_usdt_left(self):
        self.x.st["kx_tot"] = {"confirmed": 0.0, "unconfirmed": 2.6, "paid": 45.121}
        self.x.safetrade()
        rows = [shown(l) for l in self.report().splitlines() if l.startswith("| ")]
        head = "| 💵 SafeTrade | Paid | Received | Waiting | Sold | Fees | USDT |"
        pools = rows[rows.index("| Pool | Hashrate (1h) | Confirmed | Pending | Paid |") + 1:rows.index(head)]
        self.assertFalse([r for r in pools if "💵" in r])   # no SafeTrade rows among the pools any more
        # Paid is both pools'; Received the deposits less the rejected one; Waiting held plus in open orders; Sold every
        # sale: 2.0 at 1.9, 1.0 at 2.0 and 2.0 at 1.5 make 8.8 USDT, of which 0.1% fees.
        self.assertEqual(rows[rows.index(head) + 1:], [
            "| PRL | 196.743 | 152.000 | 1.646 | 5.000 | $0.01 | +$8.79 |",
            "| XMR | 0.010368 | 0.010368 | 0.000008 | 0.010000 | $0.01 | +$6.19 |",
            "| USDC out | | | | | $0.91 | −$9.00 |",    # the 9 USDC that arrived: its trade fee and the network's
            "| USDT out | | | | | $1.00 | −$2.00 |",    # 2 USDT arrived, its fee on top; the failed one doesn't count
            "| Total out | | | | | $1.91 | −$11.00 |",  # what reached the bank
            "| Left | | | | | $1.93 | $9.05 |"])         # every fee, and the USDT the account holds
        self.assertEqual(len(self.http.made("GET", "deposits")), 3)   # two pages of PRL, one of XMR

    def test_usdt_sent_to_the_salad_wallet_is_its_own_row_not_the_banks(self):
        # 28 October: 5 USDT to the top-up wallet (1 of the 6 asked went in fees) paid for $5 of Salad credit; it was
        # counted in USDT out, the money that reached the bank.
        self.http.on("GET", r"safetrade\.com/api/v2/trade/account/withdraws\?", (200, [
            {"currency_id": "usdt", "amount": "2.0", "fee": "1", "status": "succeed", "rid": "0xbank"},
            {"currency_id": "usdt", "amount": "5.0", "fee": "1", "status": "succeed", "rid": self.x.cfg["salad_wallet"]},
            {"currency_id": "usdc", "amount": "9.0", "fee": "0.9", "status": "succeed"}]))
        self.x.safetrade()
        rows = [shown(l) for l in self.report().splitlines() if l.startswith("| ")]
        self.assertEqual(rows[-5:], [
            "| USDC out | | | | | $0.91 | −$9.00 |",
            "| USDT out | | | | | $1.00 | −$2.00 |",
            "| Total out | | | | | $1.91 | −$11.00 |",   # still only what reached the bank
            "| USDT to Salad | | | | | $1.00 | −$5.00 |",
            "| Left | | | | | $2.93 | $9.05 |"])        # its fee counts with the rest

    def test_only_our_own_sell_orders_count(self):
        # A fill whose taker sold into our buy order is a buy of ours.
        self.fills = [self.fill(1, order_side="buy", side="sell")]
        self.x.safetrade()
        self.assertEqual(self.row("PRL"), "| PRL | 151.622 | 152.000 | 1.646 | 0.000 | $0.00 | +$0.00 |")

    def test_every_page_of_fills_counts(self):
        self.fills = [self.fill(300 - i, amount=1.0, price=2.0) for i in range(250)]
        self.x.safetrade()
        self.assertEqual(len(self.http.made("GET", "market/trades")), 3)
        self.assertIn("| 250.000 | $0.50 | +$499.50 |", self.row("PRL"))

    def test_a_new_sale_is_an_event_with_its_usdt_and_old_ones_are_not(self):
        self.x.safetrade()
        self.assertNotIn("SOLD", self.events())
        # One order filled twice, then a newer order: one event each, oldest first.
        self.fills = [self.fill(8, amount=0.5, price=2.0, order=200), self.fill(7, amount=1.0, price=2.1, order=100),
                      self.fill(6, amount=1.0, price=1.9, order=100)] + self.fills
        self.clock[0] += 60 * self.x.cfg["report_min"]   # the next report's read
        self.x.safetrade()
        sold = [l for l in self.events().splitlines() if " SOLD " in l]
        self.assertEqual(len(sold), 2)
        self.assertIn("SOLD 2.000 PRL on SafeTrade for $4.00 USDT after fees ($2.00 each)", sold[0])
        self.assertIn("SOLD 0.500 PRL on SafeTrade for $1.00 USDT after fees ($2.00 each)", sold[1])

    def test_requests_are_signed(self):
        self.x.safetrade()
        h = self.http.headers
        self.assertEqual(h["X-Auth-Apikey"], "k" * 16)
        want = w.hmac.new(b"s" * 32, (h["X-Auth-Nonce"] + "k" * 16).encode(), w.hashlib.sha256).hexdigest()
        self.assertEqual(h["X-Auth-Signature"], want)
        self.assertFalse([c for c in self.http.calls if c[0] != "GET"])

    def test_read_once_a_report_in_its_first_round(self):
        self.x.cfg["report_min"] = 15
        reads = lambda: len(self.http.made("GET", "balances"))
        self.x.safetrade()
        for _ in range(14):   # the rest of the quarter hour
            self.clock[0] += 60
            self.x.safetrade()
        self.assertEqual(reads(), 1)
        self.clock[0] += 60   # T0 + 15 min: the next report's first round
        self.x.safetrade()
        self.assertEqual(reads(), 2)

    def test_a_failed_read_is_tried_again_5_minutes_later(self):
        self.x.cfg["report_min"] = 15
        self.http.on("GET", r"safetrade\.com", (500, {}))
        self.x.safetrade()
        for _ in range(5):
            self.clock[0] += 60
            self.x.safetrade()
        self.assertEqual(len(self.http.made("GET", "balances")), 2)   # T0 and T0 + 5 min

    def test_a_refused_read_warns_once_keeps_the_last_and_says_when_it_was(self):
        self.x.safetrade()
        self.http.on("GET", r"safetrade\.com", (403, {"raw": ""}))
        for _ in range(4):
            self.clock[0] += 300
            self.x.safetrade()
        self.assertEqual(self.events().count("WARN can't read SafeTrade /account/balances/spot: HTTP 403"), 1)
        self.assertIn(f"| 💵 SafeTrade{w.NB}(read{w.NB}3:00{w.NB}am) | Paid |", shown(self.report()))
        self.assertEqual(self.row("PRL"), "| PRL | 151.622 | 152.000 | 1.646 | 5.000 | $0.01 | +$8.79 |")

    def test_a_read_from_before_deposits_and_withdrawals_were_read_shows_what_it_has(self):
        self.x.st["st_acct"] = {"t": self.clock[0], "coins": {"pearl": [1.0, 0.0], "usdt": [3.8, 0.0]},
                                "sold": {"pearl": [2.0, 3.8]}}
        self.assertEqual(self.row("PRL"), "| PRL | 151.622 | — | 1.000 | 2.000 | $0.00 | +$3.80 |")
        self.assertEqual(self.row("**Left**"), "| Left | | | | | $0.00 | $3.80 |")

    def test_a_read_from_before_the_totals_shows_no_row_until_the_next(self):
        self.x.st["st_acct"] = {"t": self.clock[0], "coins": {}, "sales": {}}
        self.assertNotIn("SafeTrade", self.report())


class TestClassPrices(Base):
    def test_a_running_group_takes_a_new_price(self):
        g = self.group(cls="RTX 4080 (16 GB)", pri="low")
        old = self.group(cls="RTX 4080 (16 GB)", pri="low")
        old["ended"] = True
        items = [{"name": n, "id": c["id"], "prices": [{"priority": p, "price": v} for p, v in c["prices"].items()]}
                 for n, c in CLASSES.items()]
        items[0]["prices"][1]["price"] = 0.163   # 4080 at low: 0.15 → 0.163
        self.http.on("GET", "/gpu-classes$", (200, {"items": items}))
        self.x.st["classes_t"] = 0
        self.x.classes()
        self.assertEqual(g["price"], 0.163)
        self.assertEqual(old["price"], 0.15)          # an ended group keeps what it cost
        self.assertEqual(self.events().count("PRICE"), 1)
        self.assertIn("now $0.163/hr (was $0.150)", self.events())
        self.x.classes()                              # within the hour: no read, no second event
        self.assertEqual(self.events().count("PRICE"), 1)


class TestSell(Base):
    """The PRL autoseller, moved in from the Railway trader service (removed); the book and numbers are its tests'."""
    BOOK = {"bids": [{"price": "1.40", "remaining_volume": "3"}, {"price": "1.38", "remaining_volume": "2"},
                     {"price": "1.30", "remaining_volume": "50"}], "asks": []}
    MKT = {"id": "pearlusdt", "base_unit": "pearl", "state": "enabled", "amount_precision": 2, "price_precision": 4,
           "min_amount": "0.1"}

    def setUp(self):
        super().setUp()
        os.environ["SAFETRADE_API_KEY"], os.environ["SAFETRADE_API_SECRET"] = "k" * 16, "s" * 32
        self.addCleanup(os.environ.pop, "SAFETRADE_API_KEY", None)
        self.addCleanup(os.environ.pop, "SAFETRADE_API_SECRET", None)
        w.time.sleep, real_sleep = (lambda s: None), w.time.sleep
        self.addCleanup(setattr, w.time, "sleep", real_sleep)
        self.x.cfg["sell"] = {"live": True}
        self.book(self.BOOK)
        self.http.on("GET", r"/public/markets/pearlusdt$", (200, self.MKT))
        self.http.on("GET", r"/market/orders\?market=pearlusdt&state=wait&side=sell$", (200, []))
        self.http.on("GET", r"/account/balances/spot$", (200, [{"currency": "usdt", "balance": "9"},
                                                               {"currency": "pearl", "balance": "4.123"}]))
        self.http.on("POST", r"/market/orders$", (201, {"id": 77, "state": "wait"}))
        self.http.on("POST", r"/cancel$", (200, {}))
        self.clock[0] = T0 - T0 % 3600 + 5 * 60          # :05 past the hour

    def book(self, b):
        self.http.on("GET", r"/public/markets/pearlusdt/depth\?limit=50$", (200, b))

    def orders(self):
        return [c[2] for c in self.http.made("POST", r"/market/orders$")]

    def test_sells_what_the_book_takes_within_slippage(self):
        self.x.sell(self.clock[0])
        # 3% under the best bid of 1.40 is 1.358: the bids at 1.40 and 1.38 (5 PRL) qualify; we have 4.123
        self.assertEqual(self.orders(), [{"market": "pearlusdt", "side": "sell", "amount": 4.12, "price": 1.358}])
        self.assertIn("SELL: order 77 for 4.12 PRL with a limit at 1.358 (best bid 1.4, balance 4.123)", self.events())
        self.assertTrue(self.http.headers["X-Auth-Apikey"])

    def test_sells_only_what_the_bids_above_the_floor_take(self):
        self.http.on("GET", r"/account/balances/spot$", (200, [{"currency": "pearl", "balance": "20"}]))
        self.x.sell(self.clock[0])
        self.assertEqual(self.orders()[0]["amount"], 5.0)

    def test_never_under_min_price(self):
        self.x.cfg["sell"]["min_price"] = 1.45
        self.x.sell(self.clock[0])
        self.assertEqual(self.orders(), [])
        self.assertIn("nothing to sell (best bid 1.4 is under the floor 1.45)", self.events())

    def test_dust_and_the_markets_minimum_are_left(self):
        self.http.on("GET", r"/account/balances/spot$", (200, [{"currency": "pearl", "balance": "0.3"}]))
        self.x.sell(self.clock[0])
        self.assertEqual(self.orders(), [])
        self.assertIn("nothing to sell (only 0.3 PRL", self.events())
        self.assertEqual(w.sell_plan(1.5, [(1.4, 10)], dict(self.MKT, min_amount="2"), {}),
                         (None, "only 1.5 PRL to sell (balance 1.5, bids down to 1.358 take 10.0000)"))

    def test_last_runs_open_sell_is_cancelled_first(self):
        self.http.on("GET", r"/market/orders\?market=pearlusdt&state=wait&side=sell$",
                     (200, [{"id": 5, "side": "sell", "state": "wait", "origin_amount": "2", "filled_amount": "1", "price": "1.5"},
                            {"id": 6, "side": "sell", "state": "done"}]))
        self.x.sell(self.clock[0])
        posts = [c[1].split("/api/v2/trade")[1] for c in self.http.calls if c[0] == "POST"]
        self.assertEqual(posts, ["/market/orders/5/cancel", "/market/orders"])   # only what's open, then the sale
        self.assertIn("SELL: cancelled open sell 5, 1 PRL at 1.5", self.events())

    def test_book_as_price_volume_pairs(self):
        self.book({"bids": [["1.40", "10"]], "asks": []})
        self.x.sell(self.clock[0])
        self.assertEqual(self.orders()[0]["amount"], 4.12)

    def test_amounts_round_down_never_up(self):
        self.assertEqual(w.floor_to(4.12999999999995, 4), 4.1299)
        self.assertEqual(w.floor_to(4.12, 2), 4.12)
        self.assertEqual(w.floor_to(0.30000000000000004, 1), 0.3)

    def test_a_dry_run_places_and_cancels_nothing(self):
        self.x.cfg["sell"] = {}                              # live isn't set
        self.http.on("GET", r"/market/orders\?market=pearlusdt&state=wait&side=sell$",
                     (200, [{"id": 5, "side": "sell", "state": "wait", "origin_amount": "1", "price": "1.5"}]))
        self.x.sell(self.clock[0])
        self.assertFalse([c for c in self.http.calls if c[0] == "POST"])
        self.assertIn("SELL DRY RUN: would cancel open sell 5, 1 PRL at 1.5", self.events())
        self.assertIn("SELL DRY RUN: would sell 4.12 PRL with a limit at 1.358 (best bid 1.4, balance 4.123)", self.events())

    def test_once_an_hour_from_its_minute(self):
        self.x.cfg["sell"]["minute"] = 10
        self.x.sell(self.clock[0])                           # :05, before :10
        self.assertEqual(self.orders(), [])
        self.clock[0] += 5 * 60
        self.x.sell(self.clock[0])
        self.x.sell(self.clock[0] + 600)                     # same hour: no second run
        self.assertEqual(len(self.orders()), 1)
        self.x.sell(self.clock[0] + 3600)
        self.assertEqual(len(self.orders()), 2)

    def test_off_without_its_config_or_the_keys(self):
        del self.x.cfg["sell"]
        self.x.sell(self.clock[0])
        self.x.cfg["sell"] = {"live": True}
        del os.environ["SAFETRADE_API_SECRET"]
        self.x.sell(self.clock[0])
        self.assertFalse(self.http.calls)

    def test_the_retired_prl_market_is_passed_over_and_errors_are_one_warning(self):
        self.http.on("GET", r"/public/markets/pearlusdt$", (200, dict(self.MKT, state="disabled")))
        self.http.on("GET", r"/public/markets/prlusdt$", (404, {}))
        self.x.sell(self.clock[0])
        self.assertEqual(self.orders(), [])
        self.assertIn("WARN SELL failed: no enabled PRL/USDT market", self.events())
        self.http.on("GET", r"/public/markets/pearlusdt$", (200, self.MKT))
        self.http.on("POST", r"/market/orders$", (422, {"errors": ["market.order.insufficient_funds"]}))
        self.x.sell(self.clock[0] + 3600)
        self.assertIn("WARN SELL failed: sell order: HTTP 422", self.events())


class TestSaladBalance(Base):
    def setUp(self):
        super().setUp()
        os.environ["SALAD_USER"], os.environ["SALAD_PASS"] = "u@x", "pw"
        self.addCleanup(os.environ.pop, "SALAD_USER", None)
        self.addCleanup(os.environ.pop, "SALAD_PASS", None)
        self.reads, self.answer = [], (200, 22865, None)
        self.real_balance = real = w.portal_balance
        w.portal_balance = lambda u, p, orgs=None: self.reads.append((u, p)) or self.orgs.append(orgs) or self.answer
        self.orgs = []
        self.addCleanup(setattr, w, "portal_balance", real)
        self.x.cfg.update(credit_start=100.0, credit_offset=401.01)

    def head(self):
        self.x.write_report()
        with open(os.path.join(TMP, "report.md"), encoding="utf-8") as f:
            return f.readline()

    def test_salads_own_balance_shows_without_a_star(self):
        self.x.salad_balance()
        self.assertEqual(self.reads, [("u@x", "pw")])
        self.assertIn("Salad $228.65 left", self.head())

    def test_shows_it_out_of_all_the_credit_bought(self):
        self.answer = (200, 20237, 500.0)
        self.x.salad_balance()
        self.assertIn("Salad $202.37/$500 left", self.head())
        self.answer = (200, 20137, None)   # the invoices didn't read: keep the last total
        self.clock[0] += 900
        self.x.salad_balance()
        self.assertIn("Salad $201.37/$500 left", self.head())

    def test_credit_bought_is_every_grant_not_voided(self):
        # A crypto top-up makes a Metronome credit grant and no purchase invoice (28 October: $5 in USDT).
        grants = [{"amount_granted": "40000", "voided_at": None}, {"amount_granted": "40000", "voided_at": "2026-10-14"},
                  {"amount_granted": "10000", "voided_at": None}, {"amount_granted": "500", "voided_at": None}]
        asked = []

        class R(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        real = w.urllib.request.urlopen
        w.urllib.request.urlopen = lambda req, timeout=None: asked.append(json.loads(req.data)["query"]) or \
            R(json.dumps({"data": {"Customer": [{"CreditGrants": grants}]}}).encode())
        self.addCleanup(setattr, w.urllib.request, "urlopen", real)
        call = lambda m, p, b=None: (200, {"embed_url": "https://x/embed?token=t1"}) if p.endswith("/dashboard-sessions/credits") else (404, None)
        self.assertEqual(w.bought(call), 505.0)
        self.assertIn("CreditGrants", asked[0])
        self.assertIsNone(w.bought(lambda m, p, b=None: (500, "down")))

    def test_every_org_in_salad_orgs_adds_up(self):
        # 28 October: llmjob mines, llmjob2 got $10 of its own; the heading is both.
        self.x.salad_balance()
        self.assertEqual(self.orgs, [[w.ORG]])                 # none set: just the org the boxes run in
        self.x.cfg["salad_orgs"] = ["llmjob", "llmjob2"]
        self.clock[0] += 900
        self.x.salad_balance()
        self.assertEqual(self.orgs[-1], ["llmjob", "llmjob2"])
        bal = {"llmjob": 17533, "llmjob2": 1000}
        grants = {"llmjob": 505.0, "llmjob2": 10.0}

        def call(method, path, body=None):
            if path == "/users/login":
                return 204, None
            org = path.split("/")[2]
            if path.endswith("/credits-balance"):
                return 200, {"amount": bal[org]}
            return 404, None
        real = w.bought
        w.bought = lambda c, org=None: grants[org]
        self.addCleanup(setattr, w, "bought", real)
        self.assertEqual(self.real_balance("u", "p", ["llmjob", "llmjob2"], call),
                         (200, 18533, 515.0, {"llmjob": (17533, 505.0), "llmjob2": (1000, 10.0)}))
        grants["llmjob2"] = None                               # one org's grants didn't read: no total
        self.assertEqual(self.real_balance("u", "p", ["llmjob", "llmjob2"], call)[:3], (200, 18533, None))
        bal.pop("llmjob2")                                     # one org's balance didn't read: no balance at all
        self.assertEqual(self.real_balance("u", "p", ["llmjob", "llmjob2"], lambda m, p, b=None:
                                           (204, None) if p == "/users/login" else
                                           (200, {"amount": 1}) if "/llmjob/" in p else (500, "down"))[:2], (500, "down"))

    def two_orgs(self, steps):
        """Reads of llmjob and llmjob2, a quarter-hour apart: steps is [(llmjob $, llmjob2 $, llmjob2 bought $)]."""
        self.x.cfg["salad_orgs"] = ["llmjob", "llmjob2"]
        for i, (a, b, paid2) in enumerate(steps):
            if i:
                self.clock[0] += 900
            self.answer = (200, round((a + b) * 100), 505.0 + paid2,
                           {"llmjob": (round(a * 100), 505.0), "llmjob2": (round(b * 100), paid2)})
            self.x.salad_balance()

    def test_each_org_lasts_on_its_own_and_one_not_spending_isnt_shown(self):
        # The fleet (all in llmjob) is one 4080 on a PC; llmjob2 spends nothing yet.
        g = self.group()
        self.step(g, 1, "m1")
        self.two_orgs([(175.0, 10.0, 10.0)] * 9)
        head = self.head()
        self.assertIn("Salad $185.00/$515 left (llmjob ", head)
        self.assertNotIn("llmjob2", head)

    def test_an_org_the_watcher_doesnt_run_is_timed_by_its_measured_spend(self):
        # llmjob2 falls $1 an hour (0.25 a quarter), with a $10 top-up two hours in that isn't spend.
        steps = [(175.0, 10.0 - 0.25 * i + (10.0 if i >= 8 else 0.0), 10.0 + (10.0 if i >= 8 else 0.0)) for i in range(12)]
        self.two_orgs(steps)
        self.assertAlmostEqual(self.x.org_burn("llmjob2", self.clock[0]), 1.0)
        self.assertIn("(llmjob2 17 h)", self.head())          # $17.25 at $1/hr; the fleet here has no box, so no llmjob
        # And it warns once it's under 12 h, for that org.
        self.x.st["org_bal"] = {}
        self.two_orgs([(175.0, 9.0 - 0.25 * i, 20.0) for i in range(12)])
        self.head()
        self.assertIn("ALERT llmjob2's credit $6.25 lasts about 6.2 h", self.events())

    def test_off_without_the_login(self):
        del os.environ["SALAD_PASS"]
        self.x.salad_balance()
        self.assertEqual(self.reads, [])
        self.assertIn("Salad $501.01* left", self.head())

    def test_read_once_a_report_in_its_first_round(self):
        self.x.cfg["report_min"] = 15
        for _ in range(16):   # T0 to T0 + 15 min, a round a minute
            self.x.salad_balance()
            self.clock[0] += 60
        self.assertEqual(len(self.reads), 2)   # at T0 and at T0 + 15 min

    def test_a_read_sets_our_count_so_a_stale_read_falls_back_close_to_it(self):
        self.x.salad_balance()
        g = self.group()
        g["spend"] = 0.5
        self.clock[0] += 3601
        # No read for over an hour: our count, starred, from Salad's last figure less what was spent since.
        self.assertIn("Salad $228.15* left", self.head())

    def test_a_failed_read_warns_once_and_keeps_the_last(self):
        self.x.salad_balance()
        self.answer = (401, "Unauthorized", None)
        for _ in range(3):
            self.clock[0] += 600
            self.x.salad_balance()
        self.assertEqual(self.events().count("WARN can't read Salad's balance from the portal: HTTP 401"), 1)
        self.assertIn("Salad $228.65 left", self.head())


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
        with open(os.path.join(TMP, "report.md"), encoding="utf-8") as f:
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
        with open(os.path.join(TMP, "report.md"), encoding="utf-8") as f:
            boxes = [l.split(" | ")[0] for l in f.read().splitlines() if l.startswith("| ⛏") or l.startswith("| ⏳")]
        self.assertEqual([b.split("\u00a0")[1] for b in boxes], ["4080", "2060", "4070"])

    def test_medium_is_shortened_in_the_table(self):
        g = self.group(pri="medium")
        g["has_pc"], g["machine"] = True, "abcdef1234"
        self.x.write_report()
        with open(os.path.join(TMP, "report.md"), encoding="utf-8") as f:
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


    def test_billing_skips_a_pc_still_downloading_but_the_rules_see_it(self):
        # Salad bills only a running instance; a downloading one has a PC for the rules (bad list, NEW PC) already.
        self.x.cfg["search"] = False
        g = self.group()
        self.http.on("GET", "/containers/biz-rtx4080", (200, {"current_state": {"status": "running"}}))
        self.http.on("GET", "/instances", (200, {"instances": [{"id": "i1", "machine_id": "m1", "state": "downloading"}]}))
        self.x.round()
        self.clock[0] += 120
        self.x.round()
        self.assertTrue(g["has_pc"])
        self.assertEqual(g["machine"], "m1")
        self.assertEqual(g["spend"], 0)
        self.http.on("GET", "/instances", (200, {"instances": [{"id": "i1", "machine_id": "m1", "state": "running"}]}))
        self.clock[0] += 60
        self.x.round()
        self.assertAlmostEqual(g["spend"], 0.11 * 60 / 3600)

XMR = "8" + "a" * 94


class TestMonero(Base):
    """Monero on the box's CPU: the groups config names get release v0.6.0, more vCPUs and the address."""
    def setUp(self):
        super().setUp()
        self.x.cfg.update(xmr_wallet=XMR, xmr_groups=["biz-rtx4080"])

    @needs_sh
    def test_the_command_adds_the_worker_and_the_address(self):
        cmd = w.cmd_for("biz-rtx4080", "prl1abc", "v0.6.0", XMR)
        sed = re.search(r"(sed .*?) > /root/box\.sh", cmd).group(1)
        # The three box.sh lines it changes, as they are at the pinned commit.
        box = ("  declare -A dl; for f in llmjob-earn-cli-linux pearl_core.node pearl_core_cu13.node; do   # all three at once\n"
               "BASE='{\"core\":\"release\",\"env\":{},\"flags\":\"\"}'\n"
               "  rm -rf $R && mkdir -p $R && cp $D/rel/cli $R/\n")
        import subprocess
        out = subprocess.run(["sh", "-c", sed], input=box, capture_output=True, encoding="utf-8").stdout
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
        self.assertAlmostEqual(mk["per_hash"] / (0.6 / 6e11 * 0.991), 1)
        self.assertAlmostEqual(mk["usd_khs_hr"] / (0.6 / 6e11 * 0.991 * 3.6e6 * 500), 1)

    def test_a_failed_market_read_is_tried_again_5_minutes_later(self):
        self.http.on("GET", "monero.herominers.com/api/stats$",
                     (200, {"network": {"difficulty": 600_000_000_000}, "lastblock": {"reward": 600_000_000_000}}))
        self.http.on("GET", "coingecko", (429, {}))
        self.x.xmr_market()
        self.clock[0] += 240
        self.x.xmr_market()
        self.assertEqual(len(self.http.made("GET", "coingecko")), 1)
        self.clock[0] += 61
        self.x.xmr_market()
        self.assertEqual(len(self.http.made("GET", "coingecko")), 2)

    def market(self, reward, price=600):
        self.http.on("GET", "monero.herominers.com/api/stats$",
                     (200, {"network": {"difficulty": 800_000_000_000}, "lastblock": {"reward": reward}}))
        self.http.on("GET", "coingecko", (200, {"monero": {"usd": price}}))

    def test_a_far_off_market_read_is_set_aside_until_it_repeats(self):
        # 22 October: at least 15 reads of HeroMiners' stats gave a hash about 8 to 2,200 times too little.
        self.market(620_000_000_000)
        self.x.xmr_market()
        good = dict(self.x.st["xmr_mkt"])
        self.market(12_400_000_000)                  # a reward 50 times too small, read for 19 minutes
        self.clock[0] += 300
        for _ in range(20):
            self.x.xmr_market()
            self.assertEqual(self.x.st["xmr_mkt"], good)
            self.clock[0] += 60
        self.market(620_000_000_000, price=650)      # then a right read again
        self.x.xmr_market()
        self.assertAlmostEqual(self.x.st["xmr_mkt"]["price"], 650)
        self.assertNotIn("xmr_odd", self.x.st)
        self.clock[0] += 300
        self.market(62_000_000_000)                  # a real 10-fold fall is taken once it has held 20 minutes
        for _ in range(20):
            self.x.xmr_market()
            self.assertAlmostEqual(self.x.st["xmr_mkt"]["per_hash"] / (0.62 / 8e11 * 0.991), 1)
            self.clock[0] += 60
        self.x.xmr_market()
        self.assertAlmostEqual(self.x.st["xmr_mkt"]["per_hash"] / (0.062 / 8e11 * 0.991), 1)

    def test_a_box_paused_for_a_prompt_or_without_a_pearl_reading_has_no_margin(self):
        # 22 October: 3060-70, paused for Gemma's speed test 20 minutes into its PC, showed "-0.05* (-3180%*)"
        # from its Monero alone, and its bill turned the All row's margin negative.
        t = self.clock[0]
        self.x.st["xmr_mkt"] = {"t": t, "price": 500.0, "per_hash": 1e-12, "usd_khs_hr": 0.002}
        p, d, ok = self.group(), self.group(cls="RTX 2060 (6 GB)"), self.group(cls="RTX 4070 (12 GB)")
        for g in (p, d, ok):
            g.update(has_pc=True, machine="m1", pc_since=t - 1800, xmr=True, xmr_khs=1.0, xmr_seen=t)
        p.update(readings=[[t - 900, "m1", 200.0, 0]], pause_since=t - 600, paused_t=t - 60)
        d["readings"] = [[t - 900, "m1", 40.0, 0]]       # no reading for 15 minutes, not paused: a quiet miner
        ok["readings"] = [[t - 60, "m1", 200.0, 0]]
        self.x.write_report()
        with open(os.path.join(TMP, "report.md"), encoding="utf-8") as f:
            report = f.read()
        self.assertIn(f"| 🧠{w.NB}4080{w.NB}batch | 0 | 1.0 | 0.11 | — |", report)   # like a box starting
        self.assertIn(f"| ⛏️{w.NB}2060{w.NB}batch | 0 | 1.0 | {d['price']:.2f} | {0.002 - d['price']:+.2f}* |", report)  # no margin
        rows = [l for l in report.splitlines() if l.startswith("| ") and "%" in l.split("|")[5]]
        self.assertEqual(len(rows), 2)                     # the box with a reading, and the All row
        # The paused box's bill is left out of the total, as a starting box's is; the quiet one's counts.
        earn = 200 * self.x.st["mkt"]["usd_th_hr"] * self.x.credit_ratio() + 3 * 0.002
        hr = earn - ok["price"] - d["price"]
        self.assertIn(f"| **All** | **200** | **3.0** | **{p['price'] + d['price'] + ok['price']:.2f}** | "
                      f"**{hr:+.2f}\\* ({hr / earn * 100:+.0f}%\\*)** |", report)

    def test_the_table_shows_monero(self):
        g = self.group()
        plain = self.group(cls="RTX 2060 (6 GB)")
        self.x.st["xmr_mkt"] = {"t": self.clock[0], "price": 500.0, "per_hash": 1e-12, "usd_khs_hr": 0.002}
        self.x.st["xmr_tot"] = {"balance": 0.005, "paid": 0.0, "pending": 0.0001, "khs_1h": 1.5}
        g.update(has_pc=True, machine="m1", readings=[[self.clock[0] - 60, "m1", 200.0, 0]],
                 xmr_khs=2.0, xmr_seen=self.clock[0], xmr_coins=0.0001)
        self.x.write_report()
        with open(os.path.join(TMP, "report.md"), encoding="utf-8") as f:
            report = f.read()
        self.assertIn("| Box | TH/s | kH/s | $/hr | Profit/hr (margin) | PRL | XMR |", report)
        # 200 TH/s × 0.00117 + 2 kH/s × 0.002 = 0.238 an hour, less 0.11 billed. No Pearl counted at a pool yet;
        # 0.0001 XMR from HeroMiners' count of its Monero work, starred.
        self.assertIn(f"| ⛏️{w.NB}4080{w.NB}batch | 200 | 2.0 | 0.11 | +0.13* (+54%*) | 0.000 | 0.000100* |", report)
        self.assertIn(f"| ⏳{w.NB}2060{w.NB}batch | — | — | — | — | 0.000 | 0.000000 |", report)   # nothing to star
        self.assertIn("| **All** | **200** | **2.0** | **0.11** |", report)
        self.assertIn("| Pool | Hashrate (1h) | Confirmed | Pending | Paid |", report)
        self.assertIn("| 🪙 XMR HeroMiners | 1.5 kH/s | 0.005000 | 0.000100 | 0.000000 |", shown(report))  # straight from the pool: no stars
        rows =[l for l in report.split("| Pool |")[0].splitlines() if l.startswith("|")]
        self.assertTrue(all(l.count("|") == 8 for l in rows))  # one table, with kH/s and XMR columns
        # each row on one line in a chat window, as shown (bold marks and escapes aren't)
        self.assertTrue(all(len(shown(l)) <= 90 for l in rows), max(rows, key=len))

    def kx_api(self, balance=(200, {"confirmed": 2.061e-06, "unconfirmed": 5e-07, "threshold": 0.1})):
        self.x.cfg["xmr_wallet"] = "4xyz"
        self.http.on("GET", "kryptex.com/xmr/api/v1/miner/balance/4xyz$", balance)
        self.http.on("GET", "kryptex.com/xmr/api/v1/miner/payouts/4xyz$",
                     (200, {"count": 2, "results": [{"received": "0.1"}, {"received": "0.12"}]}))
        self.http.on("GET", "kryptex.com/xmr/api/v3/miner/workers/4xyz$",
                     (200, {"results": [{"worker": "a", "avg_hashrate_30m": "1500"}, {"worker": "b", "avg_hashrate_30m": "500"}]}))

    def test_box_rates_on_kryptex_are_read_even_when_the_balance_fails(self):
        self.kx_api(balance=(503, {}))
        self.http.on("GET", "kryptex.com/xmr/api/v3/miner/workers/4xyz$",
                     (200, {"results": [{"worker": "biz-rtx4080-g0", "avg_hashrate_30m": "1500"}]}))
        g = self.group()
        self.x.kx_xmr([g], self.clock[0])
        self.assertEqual((g["kxx_khs"], g["kxx_seen"]), (1.5, self.clock[0]))
        self.assertNotIn("kxx_tot", self.x.st)

    def test_kryptexs_monero_pool_is_read_at_most_every_5_minutes(self):
        self.kx_api()
        t = self.clock[0]
        self.x.kx_xmr([], t)
        kxx = self.x.st["kxx_tot"]
        self.assertAlmostEqual(kxx["confirmed"], 2.061e-06)
        self.assertAlmostEqual(kxx["unconfirmed"], 5e-07)
        self.assertAlmostEqual(kxx["paid"], 0.22)  # the sum of its payouts
        self.assertAlmostEqual(kxx["khs"], 2.0)    # the workers' 30-minute hashrates, in kH/s
        self.x.kx_xmr([], t + 240)
        self.assertEqual(len(self.http.made("GET", "xmr/api/v1/miner/balance")), 1)
        self.x.kx_xmr([], t + 300)
        self.assertEqual(len(self.http.made("GET", "xmr/api/v1/miner/balance")), 2)

    def test_every_page_of_kryptexs_monero_payouts_is_counted(self):
        self.kx_api()
        self.http.on("GET", "kryptex.com/xmr/api/v1/miner/payouts/4xyz$",
                     (200, {"count": 3, "next": "https://x/?page=2", "results": [{"received": "0.1"}, {"received": "0.12"}]}))
        self.http.on("GET", r"kryptex.com/xmr/api/v1/miner/payouts/4xyz\?page=2$", (200, {"results": [{"received": "0.05"}]}))
        self.x.kx_xmr([], self.clock[0])
        self.assertAlmostEqual(self.x.st["kxx_tot"]["paid"], 0.27)

    def test_a_failed_read_of_kryptexs_monero_pool_keeps_the_last_and_waits_5_minutes(self):
        self.kx_api()
        t = self.clock[0]
        self.x.kx_xmr([], t)
        self.kx_api(balance=(500, {}))
        self.x.kx_xmr([], t + 300)
        self.assertEqual(self.x.st["kxx_tot"]["t"], t)
        self.x.kx_xmr([], t + 540)  # 4 minutes after the failed try
        self.assertEqual(len(self.http.made("GET", "xmr/api/v1/miner/balance")), 2)

    def test_the_pool_table_shows_kryptexs_monero_pool(self):
        self.group()
        self.x.cfg["xmr_wallet"] = "4xyz"
        self.x.st["xmr_tot"] = {"balance": 0.005, "paid": 0.0, "pending": 0.0001, "khs_1h": 1.5}
        self.x.st["kxx_tot"] = {"t": self.clock[0], "confirmed": 2.061e-06, "unconfirmed": 0.0, "paid": 0.0, "khs": 0.0}
        self.x.write_report()
        with open(os.path.join(TMP, "report.md"), encoding="utf-8") as f:
            report = f.read()
        self.assertIn("| 🪙 [XMR Kryptex](https://pool.kryptex.com/xmr/miner/stats/4xyz) | 0.0 kH/s "
                      "| 0.000002 | 0.000000 | 0.000000 |", report)
        rows = report.split("| Pool |")[1].splitlines()
        self.assertTrue(rows.index(next(l for l in rows if "XMR Kryptex" in l))
                        > rows.index(next(l for l in rows if "XMR HeroMiners" in l)))  # under HeroMiners' row

    def test_with_xmr_kx_the_command_sends_monero_to_kryptex_on_v0_6_6(self):
        cmd = w.cmd_for("biz-rtx4080", "prl1abc", "v0.6.6", XMR, xmr_pool="xmr-us.kryptex.network:7029")
        self.assertIn(f"--xmr-address {XMR} --xmr-pool xmr-us.kryptex.network:7029", cmd)
        self.assertNotIn("--xmr-pool", w.cmd_for("biz-rtx4080", "prl1abc", "v0.6.6", XMR))
        self.x.cfg["xmr_kx"] = True
        for kx in (False, True):
            ver, c, res = self.x.box("biz-rtx4080", True, kx)
            self.assertEqual(ver, "v0.6.6")
            self.assertIn("VER=v0.6.6;", c[2])
            self.assertIn("--xmr-pool xmr-us.kryptex.network:7029", c[2])
            self.assertEqual(res, {"cpu": 8, "memory": 8192})
        self.assertNotIn("--xmr", self.x.box("biz-rtx4080", False)[1][2])   # no Monero, no Monero pool
        self.x.cfg["ver_override"] = {"biz-rtx4080": "v0.7.0"}
        self.assertEqual(self.x.box("biz-rtx4080", True)[0], "v0.7.0")       # never lowered
        self.x.cfg.update(xmr_kx=False, ver_override={})
        self.assertNotIn("--xmr-pool", self.x.box("biz-rtx4080", True)[1][2])

    def test_the_ledgers_pearl_pool_isnt_fooled_by_the_monero_pool_flag(self):
        self.x.cfg["xmr_kx"] = True
        g = self.group()
        self.assertFalse(g["kx"])
        self.x.kx_follow(g, {"container": {"command": self.x.box(g["name"], True)[1]}}, self.clock[0] + 600)
        self.assertFalse(g["kx"])   # --xmr-pool isn't --pool: Pearl stays on HeroMiners

    def test_a_new_monero_group_sends_it_to_kryptex(self):
        self.x.cfg["xmr_kx"] = True
        g = self.group()
        self.assertTrue(g["xmr_kx"])
        self.assertIn("--xmr-pool xmr-us.kryptex.network:7029",
                      self.http.made("POST", "/containers$")[-1][2]["container"]["command"][2])
        self.assertIn(", Monero on 8 vCPUs to Kryptex", self.events())
        plain = self.group(cls="RTX 2060 (6 GB)")   # not in xmr_groups
        self.assertFalse(plain["xmr_kx"])

    def test_turning_xmr_kx_on_updates_each_monero_group_without_moving_a_box_that_mines(self):
        self.x.cfg["xmr_groups"] = ["biz-rtx4080", "biz-rtx4080-2", "biz-rtx4080-3"]
        mining, waiting, preparing = self.group(), self.group(status="deploying"), self.group(status="pending")
        plain = self.group(cls="RTX 2060 (6 GB)")
        mining.update(has_pc=True, machine="m1", instance="inst-m1")
        self.http.on("PATCH", "/containers/", (200, {}))
        self.x.cfg["xmr_kx"] = True
        t = self.clock[0]
        self.x.xmr_kx_sync([mining, waiting, preparing, plain], t)
        made = self.http.made("PATCH", "/containers/")
        self.assertEqual(len(made), 2)   # not the group still preparing its image, nor the one without Monero
        self.assertTrue(all("--xmr-pool xmr-us.kryptex.network:7029" in c[2]["container"]["command"][2] for c in made))
        self.assertTrue(all(list(c[2]["container"]) == ["command"] for c in made))   # same vCPUs and RAM
        self.assertFalse(self.http.made("POST", "/restart") or self.http.made("POST", "/reallocate"))   # keeps its PC
        self.assertTrue(mining["xmr_kx"] and waiting["xmr_kx"] and not preparing.get("xmr_kx"))
        self.assertEqual(mining["ver"], "v0.6.6")
        self.assertIn(f"MONERO ON KRYPTEX {mining['name']}: miner v0.6.6; it keeps this PC, where Salad usually "
                      "restarts it that way within minutes", self.events())
        self.assertIn(f"MONERO ON KRYPTEX {waiting['name']}: miner v0.6.6; it starts that way on its next PC",
                      self.events())
        self.x.xmr_kx_sync([mining, waiting, plain], t + 60)   # done: nothing more to send
        self.assertEqual(len(self.http.made("PATCH", "/containers/")), 2)
        # A refused update is tried again after 10 minutes.
        preparing["status"] = "deploying"
        self.http.on("PATCH", "/containers/", (409, {"error": "busy"}))
        self.x.xmr_kx_sync([preparing], t + 60)
        self.assertIn(f"WARN Monero to Kryptex for {preparing['name']} refused: HTTP 409", self.events())
        self.x.xmr_kx_sync([preparing], t + 5 * 60)
        self.assertEqual(len(self.http.made("PATCH", "/containers/")), 3)
        self.x.xmr_kx_sync([preparing], t + 11 * 60)
        self.assertEqual(len(self.http.made("PATCH", "/containers/")), 4)
        # Off again: back to HeroMiners.
        self.http.on("PATCH", "/containers/", (200, {}))
        self.x.cfg["xmr_kx"] = False
        self.x.xmr_kx_sync([mining], t + 20 * 60)
        self.assertNotIn("--xmr-pool", self.http.made("PATCH", "/containers/")[-1][2]["container"]["command"][2])
        self.assertFalse(mining["xmr_kx"])
        self.assertIn(f"MONERO ON HEROMINERS {mining['name']}: miner v0.6.0", self.events())   # config's release again

    def test_any_command_update_records_where_its_monero_goes(self):
        self.x.cfg["xmr_groups"] = ["*"]
        a, b, c = self.group(), self.group(), self.group()
        self.http.on("PATCH", "/containers/", (200, {}))
        self.x.cfg.update(xmr_kx=True, llm=True)
        t = self.clock[0]
        self.x.llm_sync([a], t)                         # Gemma's update carries --xmr-pool too
        self.assertTrue(a["xmr_kx"])
        self.assertTrue(self.x.kx_switch(b, True, t, "test", in_place=True))
        self.assertTrue(b["xmr_kx"])
        c["xmr"] = False
        self.x.xmr_sync([c], t)                         # Monero back on, to Kryptex
        self.assertTrue(c["xmr_kx"])
        n = len(self.http.made("PATCH", "/containers/"))
        self.x.xmr_kx_sync([a, b, c], t + 11 * 60)      # so none is sent again
        self.assertEqual(len(self.http.made("PATCH", "/containers/")), n)

    def test_a_newer_release_in_config_moves_each_group_in_place(self):
        self.x.cfg["xmr_groups"] = ["*"]
        mining, waiting, preparing = self.group(), self.group(status="deploying"), self.group(status="pending")
        mining.update(has_pc=True, machine="m1", instance="inst-m1")
        self.http.on("PATCH", "/containers/", (200, {}))
        t = self.clock[0]
        self.x.ver_sync([mining, waiting, preparing], t)
        self.assertFalse(self.http.made("PATCH", "/containers/"))   # each runs config's release already
        self.x.cfg["box_ver"] = "v0.6.7"
        self.x.ver_sync([mining, waiting, preparing], t)
        made = self.http.made("PATCH", "/containers/")
        self.assertEqual(len(made), 2)   # not the group still preparing its image
        self.assertTrue(all("VER=v0.6.7;" in c[2]["container"]["command"][2] for c in made))
        self.assertTrue(all(list(c[2]["container"]) == ["command"] for c in made))   # same vCPUs and RAM
        self.assertFalse(self.http.made("POST", "/restart") or self.http.made("POST", "/reallocate"))   # keeps its PC
        self.assertEqual((mining["ver"], waiting["ver"], preparing["ver"]), ("v0.6.7", "v0.6.7", "v0.6.0"))
        self.assertIn(f"RELEASE {mining['name']}: miner v0.6.0 -> v0.6.7; it keeps this PC, where Salad usually "
                      "restarts it that way within minutes", self.events())
        self.assertIn(f"RELEASE {waiting['name']}: miner v0.6.0 -> v0.6.7; it starts that way on its next PC",
                      self.events())
        self.x.ver_sync([mining, waiting], t + 11 * 60)   # done: nothing more to send
        self.assertEqual(len(self.http.made("PATCH", "/containers/")), 2)

    def test_one_group_can_go_first_and_no_release_is_lowered(self):
        a, b = self.group(), self.group(cls="RTX 2060 (6 GB)")   # b mines no Monero: config's v0.5.13
        self.http.on("PATCH", "/containers/", (200, {}))
        t = self.clock[0]
        self.x.cfg["ver_override"] = {b["name"]: "v0.6.7"}
        self.x.ver_sync([a, b], t)
        self.assertEqual([c[1].rsplit("/", 1)[1] for c in self.http.made("PATCH", "/containers/")], [b["name"]])
        self.assertEqual((a["ver"], b["ver"]), ("v0.6.0", "v0.6.7"))
        self.x.cfg.update(ver_override={a["name"]: "v0.5.0"}, box_ver="v0.5.0")
        self.x.ver_sync([a, b], t + 11 * 60)
        self.assertEqual(len(self.http.made("PATCH", "/containers/")), 1)   # neither goes back
        self.assertEqual((a["ver"], b["ver"]), ("v0.6.0", "v0.6.7"))
        # Named in ver_override, a group does go back, to leave a bad release.
        self.x.cfg["ver_override"] = {b["name"]: "v0.6.6"}
        self.x.ver_sync([a, b], t + 22 * 60)
        self.assertIn("VER=v0.6.6;", self.http.made("PATCH", "/containers/")[-1][2]["container"]["command"][2])
        self.assertEqual(len(self.http.made("PATCH", "/containers/")), 2)
        self.assertEqual(b["ver"], "v0.6.6")
        self.assertIn(f"RELEASE {b['name']}: miner v0.6.7 -> v0.6.6", self.events())

    def test_a_refused_release_update_is_tried_again_after_10_minutes(self):
        g = self.group()
        self.http.on("PATCH", "/containers/", (409, {"error": "busy"}))
        self.x.cfg["box_ver"] = "v0.6.7"
        t = self.clock[0]
        self.x.ver_sync([g], t)
        self.assertIn(f"WARN release v0.6.7 for {g['name']} refused: HTTP 409", self.events())
        self.assertEqual(g["ver"], "v0.6.0")
        self.x.ver_sync([g], t + 5 * 60)
        self.assertEqual(len(self.http.made("PATCH", "/containers/")), 1)
        self.http.on("PATCH", "/containers/", (200, {}))
        self.x.ver_sync([g], t + 11 * 60)
        self.assertEqual(len(self.http.made("PATCH", "/containers/")), 2)
        self.assertEqual(g["ver"], "v0.6.7")

    def test_a_round_moves_groups_to_a_newer_release(self):
        self.x.cfg["search"] = False
        g = self.group()
        g["created"] = T0 - 3600
        self.http.on("GET", "/containers/biz-rtx4080", (200, {"current_state": {"status": "running"}}))
        self.http.on("PATCH", "/containers/biz-rtx4080", (200, {}))
        w.save(w.CFG, {"search": False, "xmr_wallet": XMR, "xmr_groups": ["biz-rtx4080"], "box_ver": "v0.6.7"})
        self.x.round()   # it re-reads config.json
        self.assertIn("VER=v0.6.7;", self.http.made("PATCH", "/containers/biz-rtx4080")[-1][2]["container"]["command"][2])
        self.assertEqual(g["ver"], "v0.6.7")

    def test_a_round_switches_monero_to_kryptex_and_reads_kryptex(self):
        self.x.cfg["search"] = False
        g = self.group()
        g["created"] = T0 - 3600
        self.http.on("GET", "/containers/biz-rtx4080", (200, {"current_state": {"status": "running"}}))
        self.http.on("PATCH", "/containers/biz-rtx4080", (200, {}))
        self.http.on("GET", "kryptex.com/xmr/api/v1/miner/(balance|payouts)/", (200, {}))
        self.http.on("GET", "kryptex.com/xmr/api/v3/miner/workers/",
                     (200, {"results": [{"worker": "biz-rtx4080-g0", "avg_hashrate_30m": "1500"}]}))
        w.save(w.CFG, {"search": False, "xmr_wallet": XMR, "xmr_groups": ["biz-rtx4080"], "xmr_kx": True})
        self.x.round()   # it re-reads config.json
        self.assertIn("--xmr-pool", self.http.made("PATCH", "/containers/biz-rtx4080")[-1][2]["container"]["command"][2])
        self.assertEqual(g["kxx_khs"], 1.5)   # its rate on Kryptex

    def xmr_chart(self, g, *buckets):
        """Kryptex's chart for the box's Monero worker: (seconds after the group began, hashrate in H/s) a bucket."""
        self.http.on("GET", rf"kryptex.com/xmr/api/v1/miner/chart/4xyz/{g['name']}-g0\?days=2$",
                     (200, [{"timestamp": (g["created"] + s) * 1000, "valid": 1, "hashrate": hr} for s, hr in buckets]))

    def kx_box(self, per_hash=1e-10, **kw):
        """A box mining Monero on Kryptex, with what a hash earns on the network (HeroMiners' figure, less its fee)."""
        g = self.group()
        g.update(dict(xmr_kx=True, had_pc=True), **kw)
        self.kx_api()
        self.x.st["xmr_mkt"] = {"t": self.clock[0], "price": 500.0, "per_hash": per_hash * (1 - w.XMR_FEE),
                                "usd_khs_hr": 0.002}
        return g

    def xmr_row(self, name):
        self.x.write_report()
        with open(os.path.join(TMP, "report.md"), encoding="utf-8") as f:
            return next(l for l in f.read().splitlines() if name in l)

    def test_each_boxs_work_on_kryptex_is_counted_from_its_worker_chart(self):
        g = self.kx_box()
        t = self.clock[0]
        # A bucket that ended before the group began, one it began in, one filling and one with no work.
        self.xmr_chart(g, (-1200, 5000.0), (-300, 1000.0), (300, 2000.0), (900, 0.0))
        self.x.kx_xmr([g], t)
        self.assertAlmostEqual(g["kxx_gross"] / ((600_000 + 1_200_000) * 1e-10), 1, places=9)   # hashrate × 600
        self.assertEqual(g["xmr_coins"], 0.0)
        self.assertEqual(len(g["kxx_b"]), 2)
        # 5 minutes on, a hash pays twice as much: the filling bucket keeps its first value, the new one gets the new.
        self.x.st["xmr_mkt"]["per_hash"] *= 2
        self.xmr_chart(g, (-300, 1000.0), (300, 3000.0), (900, 1000.0))
        self.x.kx_xmr([g], t + 300)
        self.assertAlmostEqual(g["kxx_gross"] / ((600_000 + 1_800_000) * 1e-10 + 600_000 * 2e-10), 1, places=9)
        self.xmr_chart(g, (300, 1500.0))   # a smaller reading of a bucket never lowers it
        self.x.kx_xmr([g], t + 600)
        self.assertAlmostEqual(g["kxx_gross"] / 3.6e-4, 1, places=9)
        # At what Kryptex pays for the network's work (94%), with no star: 0.000360 × 0.94.
        self.assertIn("| 0.000 | 0.000338 |", self.xmr_row("4080"))
        self.x.cfg["kryptex_xmr_pay"] = 0.5
        self.assertIn("| 0.000 | 0.000180 |", self.xmr_row("4080"))

    def test_buckets_valued_from_a_bad_market_read_are_valued_again(self):
        # 22 October: at least 35 buckets held a pay about 8 to 2,200 times too low, kept since a bucket keeps its first.
        g = self.kx_box()
        start = int(g["created"]) * 1000
        g["kxx_b"] = {str(start + 600_000 * i): [600_000.0, 1e-10] for i in range(1, 13)}
        g["kxx_b"].update({str(start + 600_000 * i): [600_000.0, 4e-14] for i in (13, 14)})
        self.x.st["xmr_mkt"]["per_hash"] = 5e-14 * (1 - w.XMR_FEE)   # and the market read is off now
        self.xmr_chart(g, (600 * 15, 1000.0))
        self.x.kx_xmr([g], self.clock[0])
        self.assertEqual({ph for _, ph in g["kxx_b"].values()}, {1e-10})
        self.assertAlmostEqual(g["kxx_gross"], (14 * 600_000 + 600_000) * 1e-10)

    def test_the_first_chart_read_drops_the_estimates_from_the_kryptex_rate(self):
        # 0.0005 XMR so far: HeroMiners' count of its 2 million hashes there, and estimates from its rate on Kryptex.
        g = self.kx_box(xmr_coins=0.0005, xmr_hashes=2e6)
        self.xmr_chart(g, (300, 1000.0))
        t = self.clock[0]
        self.x.kx_xmr([g], t)
        self.assertAlmostEqual(g["xmr_coins"], 2e6 * 1e-10 * (1 - w.XMR_FEE))   # HeroMiners' count alone
        g["xmr_coins"] += 1e-5   # more work on HeroMiners (xmr_pool)
        self.x.kx_xmr([g], t + 300)
        self.assertAlmostEqual(g["xmr_coins"], 2e6 * 1e-10 * (1 - w.XMR_FEE) + 1e-5)   # not dropped again
        # Starred, for its HeroMiners part: 0.0001982 + 0.00001 + 600,000 hashes × 1e-10 × 94% = 0.0002646.
        self.assertIn("| 0.000 | 0.000265* |", self.xmr_row("4080"))

    def test_a_box_is_read_once_it_has_had_a_pc_with_its_monero_on_kryptex(self):
        g = self.kx_box(had_pc=False)
        t = self.clock[0]
        self.x.kx_xmr([g], t)
        g.update(had_pc=True, xmr_kx=False)          # Monero on HeroMiners
        self.x.kx_xmr([g], t + 300)
        self.assertFalse(self.http.made("GET", "xmr/api/v1/miner/chart/"))
        self.assertNotIn("kxx_b", g)
        g["xmr_kx"] = True
        self.http.on("GET", "xmr/api/v1/miner/chart/", (500, []))   # a failed read changes nothing
        self.x.kx_xmr([g], t + 600)
        self.assertEqual(len(self.http.made("GET", "xmr/api/v1/miner/chart/")), 1)
        self.assertNotIn("kxx_b", g)
        self.xmr_chart(g, (300, 1000.0))
        self.x.kx_xmr([g], t + 900)
        g["xmr_kx"] = False                          # back on HeroMiners: its last Kryptex buckets are still read
        self.x.kx_xmr([g], t + 1200)
        self.assertEqual(len(self.http.made("GET", "xmr/api/v1/miner/chart/")), 3)
        self.x.st["xmr_mkt"] = None                  # with nothing to value it at, the chart isn't read
        self.x.kx_xmr([g], t + 1500)
        self.assertEqual(len(self.http.made("GET", "xmr/api/v1/miner/chart/")), 3)

    def test_an_ended_box_keeps_its_kryptex_total_without_its_buckets(self):
        g = self.kx_box()
        self.xmr_chart(g, (300, 1000.0))
        self.x.kx_xmr([g], self.clock[0])
        g["ended"] = self.clock[0] + 60
        self.x.kx_xmr([], self.clock[0] + 120)
        self.assertNotIn("kxx_b", g)
        self.assertAlmostEqual(g["kxx_gross"], 6e-5)
        old = self.group()
        old.update(ended=self.clock[0], xmr_coins=0.0001)   # ended before the charts: its rate over time, starred
        self.assertIn("| 0.000 | 0.000156* |", self.xmr_row(f"Ended{w.NB}×2"))   # 0.0001 + 0.00006 × 94%

    def test_a_boxs_khs_counts_each_monero_pool_that_saw_it_lately(self):
        g = self.group()
        t = self.clock[0]
        self.x.st["xmr_mkt"] = {"t": t, "price": 500.0, "per_hash": 1e-12, "usd_khs_hr": 0.002}
        g.update(has_pc=True, machine="m1", readings=[[t - 60, "m1", 200.0, 0]], kxx_khs=2.0, kxx_seen=t)

        def row():
            self.x.write_report()
            with open(os.path.join(TMP, "report.md"), encoding="utf-8") as f:
                return next(l for l in f.read().splitlines() if "4080" in l)
        self.assertIn("| 200 | 2.0 |", row())
        g.update(xmr_khs=1.0, xmr_seen=t - 60)   # HeroMiners still sees it just after a switch
        self.assertIn("| 200 | 3.0 |", row())
        g["kxx_seen"] = t - 15 * w.MIN           # Kryptex hasn't seen it for 15 minutes
        self.assertIn("| 200 | 1.0 |", row())

    def test_without_an_address_the_table_is_unchanged(self):
        self.x.cfg["xmr_wallet"] = ""
        self.group()
        self.x.st["kxx_tot"] = {"t": self.clock[0], "confirmed": 2.061e-06, "unconfirmed": 0.0, "paid": 0.0, "khs": 0.0}
        self.x.write_report()
        with open(os.path.join(TMP, "report.md"), encoding="utf-8") as f:
            report = f.read()
        self.assertNotIn("kH/s", report)
        self.assertNotIn("Monero", report)
        self.assertNotIn("XMR", report)


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
        with open(os.path.join(TMP, "report.md"), encoding="utf-8") as f:
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

    @needs_sh
    def test_the_command_points_the_miner_at_kryptex_with_pool(self):
        cmd = w.cmd_for("biz-rtx4080", "prl1abc", "v0.6.1", kryptex="prl-us.kryptex.network:7048")
        sed = re.search(r"(sed .*?) > /root/box\.sh", cmd).group(1)
        import subprocess
        base = "BASE='{\"core\":\"release\",\"env\":{},\"flags\":\"\"}'\n"
        out = subprocess.run(["sh", "-c", sed], input=base, capture_output=True, encoding="utf-8").stdout
        self.assertEqual(out, "BASE='{\"core\":\"release\",\"env\":{},\"flags\":\"--pool prl-us.kryptex.network:7048\"}'\n")
        # With Monero too, both flags go in, and the CPU worker is downloaded as before.
        both = w.cmd_for("biz-rtx4080", "prl1abc", "v0.6.1", XMR, 0, "prl-us.kryptex.network:7048")
        sed = re.search(r"(sed .*?) > /root/box\.sh", both).group(1)
        out = subprocess.run(["sh", "-c", sed], input=base, capture_output=True, encoding="utf-8").stdout
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
        if not HAS_SH:
            self.skipTest("needs sh, as the boxes have")
        import subprocess
        sed = re.search(r"(sed .*?) > /root/box\.sh", cmd).group(1)
        return subprocess.run(["sh", "-c", sed], input=self.PORT_CHECK, capture_output=True, encoding="utf-8").stdout

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
        self.assertEqual((g["reallocs"], g["port_moves"]), (0, 1))     # moved, without using one of its 3
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

    def test_with_kx_in_place_a_mining_box_is_switched_by_its_command_on_its_pc(self):
        self.x.cfg.update(kx_in_place=True, kryptex_groups=2)
        dear, cheap, waiting = self.mining("m1", 300, ths=150.0), self.mining("m2", 30), self.group()
        dear["price"], cheap["price"] = 0.20, 0.05
        t = self.clock[0]
        self.x.kx_sync([dear, cheap, waiting], t)
        self.assertTrue(waiting["kx"] and cheap["kx"])          # the group waiting first, then the cheaper box
        self.assertFalse(dear["kx"])                            # only as many as are missing
        self.assertFalse(self.http.made("POST", "/reallocate$"))   # it keeps its PC
        patch = self.http.made("PATCH", f"/containers/{cheap['name']}$")
        self.assertIn("--pool", patch[0][2]["container"]["command"][2])
        self.assertIn(f"KRYPTEX ON {cheap['name']} (mining on HeroMiners): Pearl on Kryptex; switched by its command; "
                      "it keeps this PC, where Salad usually restarts it that way within minutes", self.events())
        # Its readings count to Kryptex from the switch: until Salad restarts it the PC still mines on HeroMiners.
        self.assertFalse(self.x.on_kx(cheap, "m2", t - 60))
        self.assertTrue(self.x.on_kx(cheap, "m2", t + 60))
        self.assertEqual(cheap["sw_in_place"], t)

    def test_after_a_switch_in_place_the_old_pools_shares_wait_for_the_restart(self):
        self.x.cfg.update(kx_in_place=True, kryptex_groups=1)
        g = self.mining("m2", 30)
        t = self.clock[0]
        self.x.kx_sync([g], t)
        g["_machine"] = None
        hm = lambda m: {"name": f"{g['name']}-g0", "lastShare": int(t + m * 60), "hashrate": 1000}
        self.x.pc_pool(g, None, hm(10))                         # Salad hasn't restarted it yet: no conclusion
        self.assertFalse(g.get("pc_pool"))
        self.assertNotIn("mines Pearl on HeroMiners", self.events())
        self.x.pc_pool(g, None, hm(16))                         # still on HeroMiners 16 min on: it stayed there
        self.assertEqual(g["pc_pool"], {"m2": False})
        self.assertIn(f"WARN {g['name']} mines Pearl on HeroMiners on PC m2", self.events())

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
        self.assertIn(f"| ⛏️ PRL{w.NB}Kryptex | 0 TH/s |", shown(self.report()))

    def test_the_kryptex_row_sums_the_boxes_that_mine_there(self):
        k, h = self.group(kx=True), self.group()
        k.update(has_pc=True, machine="m1", pc_pool={"m1": False}, pool_ths=199.0)   # its box mines on HeroMiners
        h.update(has_pc=True, machine="m2", pc_pool={"m2": True}, pool_ths=150.0)    # its box mines on Kryptex
        self.x.st["pool_tot"] = {"t": T0, "paid": 6.0, "balance": 1.0, "unconfirmed": 1.0, "blocks": 20,
                                 "hashes": 0, "ths_1h": 199}
        self.x.st["kx_tot"] = {"t": T0, "total": 0.5, "confirmed": 0.3, "unconfirmed": 0.2, "paid": 0.0}
        self.assertIn(f"| ⛏️ PRL{w.NB}Kryptex | 150 TH/s |", shown(self.report()))

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

    def test_a_release_update_keeps_a_kryptex_box_on_kryptex(self):
        g = self.group(kx=True)
        self.http.on("PATCH", "/containers/", (200, {}))
        self.x.cfg["box_ver"] = "v0.6.7"
        self.x.ver_sync([g], self.clock[0])
        cmd = self.http.made("PATCH", "/containers/")[-1][2]["container"]["command"][2]
        self.assertIn("--pool prl-us.kryptex.network:7048", cmd)
        self.assertEqual(w.cmd_ver(cmd), "v0.6.7")
        self.assertEqual(w.cmd_ver(w.cmd_for("biz-rtx4080", "prl1abc", "v0.5.13")), "v0.5.13")   # box.sh's own

    def test_the_ledger_follows_the_command_salad_has(self):
        self.config_file()
        g = self.group()
        box = "apt-get ...; curl -fsSL x | sed > /root/box.sh; bash /root/box.sh"
        cmd = {"current_state": {"status": "running"},
               "container": {"command": ["/bin/sh", "-c", box.replace(
                   "sed", "sed -e 's/^VER=v0.5.13;/VER=v0.6.1;/' -e '--pool prl-us.kryptex.network:7048'")]}}
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
        self.assertEqual(g["ver"], "v0.6.1")                            # and the release Salad has
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

    def test_every_page_of_kryptexs_payouts_is_counted(self):
        # Kryptex lists 15 payouts a page, newest first. Summing only the first page made the paid total drop
        # as old payouts left it (16.677, then 16.655 after a payout, of 23.452 paid on 22 October).
        wl = os.environ["PRL_WALLET"]
        self.http.on("GET", f"v1/miner/balance/{wl}", (200, {"total": 0.5, "confirmed": 0.3, "unconfirmed": 0.2}))
        self.http.on("GET", f"v1/miner/payouts/{wl}$", (200, {"count": 17, "next": f"https://x/?page=2",
                                                             "results": [{"received": 1.0}] * 15}))
        self.http.on("GET", rf"v1/miner/payouts/{wl}\?page=2$", (200, {"count": 17, "next": None,
                                                                      "results": [{"received": 0.5}] * 2}))
        self.x.kx_pool()
        self.assertAlmostEqual(self.x.st["kx_tot"]["paid"], 16.0)
        # Our own API base with ?page=N, not the "next" link, which names another host; nothing past the last page.
        self.assertEqual([c[1].split("/payouts/")[1] for c in self.http.made("GET", "v1/miner/payouts")],
                         [wl, f"{wl}?page=2"])
        self.assertTrue(all(c[1].startswith(w.KX_API) for c in self.http.made("GET", "v1/miner/payouts")))
        # A page that fails leaves the last totals as they were, rather than a sum short of some payouts.
        self.http.on("GET", rf"v1/miner/payouts/{wl}\?page=2$", (502, None))
        self.http.on("GET", f"v1/miner/balance/{wl}", (200, {"total": 9.0, "confirmed": 9.0, "unconfirmed": 0}))
        self.clock[0] += 300
        self.x.kx_pool()
        self.assertEqual((self.x.st["kx_tot"]["paid"], self.x.st["kx_tot"]["total"]), (16.0, 0.5))
        self.assertEqual(w.kx_paid(w.KX_API, wl, most=1), None)  # more pages than it reads: no total

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
        self.assertRegex(report, rf"4080{w.NB}batch \|[^\n]*\| 0\.940 \|\n")
        self.assertRegex(report, rf"4080-2{w.NB}batch \|[^\n]*\| 1\.940\* \|\n")  # 0.94 + 1.0 before 10 blocks
        self.assertIn(f"| Ended{w.NB}×2 | — | — | — | 2.500* |\n", report)
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
        self.assertIn(f"| ⛏️ PRL{w.NB}HeroMiners | 0 TH/s | 1.000 | 1.000 | 6.000 |", shown(report))
        self.assertIn(f"| ⛏️ PRL{w.NB}Kryptex | 200 TH/s | 1.000 | 0.880 | 2.000 |", shown(report))
        self.assertAlmostEqual(self.x.kx_ratio(), 0.97)
        self.assertAlmostEqual(self.x.credit_ratio(), 0.8)
        # Kryptex pays 94% of a share's value, so its shares (4 PRL at the network's rate) are 3.76 PRL, not starred.
        # HeroMiners' count (10 PRL of work) at its credit ratio is 8, starred: it pays by its luck.
        self.assertRegex(report, r"\(K\) \|[^\n]*\| 3\.760 \|\n")
        self.assertRegex(report, rf"4080-2{w.NB}batch \|[^\n]*\| 8\.000\* \|\n")
        self.assertIn("| **11.760\\*** |\n", report)

    def test_each_pools_name_links_to_its_page_for_our_address(self):
        self.x.cfg.update(wallet="prl1abc", xmr_wallet="4xyz", xmr_groups=["*"])
        self.group(kx=True)
        self.x.st["pool_tot"] = {"t": T0, "paid": 1.0, "balance": 0.0, "unconfirmed": 0.0, "blocks": 20,
                                 "hashes": 0, "ths_1h": 0}
        self.x.st["kx_tot"] = {"t": T0, "total": 0.0, "confirmed": 0.0, "unconfirmed": 0.0, "paid": 0.0}
        self.x.st["xmr_tot"] = {"t": T0, "balance": 0.0, "paid": 0.0, "pending": 0.0, "khs_1h": 0.0}
        report = self.report()
        self.assertIn(f"| ⛏️ [PRL{w.NB}HeroMiners](https://pearl.herominers.com/?wallet=prl1abc) |", report)
        self.assertIn(f"| ⛏️ [PRL{w.NB}Kryptex](https://pool.kryptex.com/prl/miner/stats/prl1abc) |", report)
        self.assertIn("| 🪙 [XMR HeroMiners](https://monero.herominers.com/?wallet=4xyz) |", report)
        self.assertIn(f"| ⛏️ PRL{w.NB}Kryptex | 0 TH/s |", shown(report))   # in a chat window: just the name

    def test_with_every_box_on_kryptex_the_report_drops_the_k_mark(self):
        a, b, gone = self.group(kx=True), self.group(kx=True), self.group()
        gone["ended"] = self.clock[0]                     # an ended HeroMiners group doesn't bring the mark back
        report = self.report()
        self.assertIn(f"| ⏳{w.NB}4080{w.NB}batch |", report)
        self.assertIn(f"| ⏳{w.NB}4080-2{w.NB}batch |", report)
        self.assertNotIn("(K)", report)
        b["kx"] = False                                   # one box back on HeroMiners: Kryptex boxes are marked again
        self.assertIn(f"| ⏳{w.NB}4080{w.NB}batch{w.NB}(K) |", self.report())

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
        self.assertIn("| ⛏️ PRL | 0 TH/s | 0.500 | 4.000 | 1.500 |", shown(report))
        self.assertNotIn("Kryptex", report)

    def test_without_kryptex_groups_the_report_is_unchanged(self):
        self.x.cfg["kryptex_groups"] = 0
        g = self.group()
        g["prl"], g["spend"] = 8.0, 6.0
        self.x.st["pool_tot"] = {"t": T0, "paid": 1.5, "balance": 0.5, "unconfirmed": 4.0, "blocks": 6,
                                 "hashes": 0, "ths_1h": 0}
        report = self.report()
        self.assertIn("| ⛏️ PRL | 0 TH/s | 0.500 | 4.000 | 1.500 |", shown(report))
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
        if not HAS_SH:
            self.skipTest("needs sh, as the boxes have")
        import subprocess
        sed = re.search(r"(sed .*?) > /root/box\.sh", cmd).group(1)
        return subprocess.run(["sh", "-c", sed], input=text, capture_output=True, encoding="utf-8").stdout

    @needs_sh
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
                with open(log, "a", encoding="utf-8") as f:
                    f.write(text)
                return subprocess.run(["bash", "-c", loop], env={"D": d, "g": "0", "PATH": os.environ["PATH"]},
                                      capture_output=True, encoding="utf-8").stdout.strip()
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
        self.assertFalse(self.http.made("POST", "/restart") or self.http.made("POST", "/reallocate"))   # keeps its PC
        self.assertTrue(mining["llm"] and waiting["llm"] and not preparing.get("llm"))
        self.assertIn(f"GEMMA ON {mining['name']}: miner v0.6.5; it keeps this PC, where Salad usually restarts it "
                      "that way within minutes", self.events())
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
        with open(os.path.join(TMP, "report.md"), encoding="utf-8") as f:
            self.assertIn(f"| 🧠{w.NB}4080{w.NB}batch |", f.read())
        items.append({"text_log": "[mining g0] 08:09:00 199.0 TH/s · 10 accepted · 0 rejected · up 0h 34m",
                      "time": "2027-01-15T08:09:00Z", "resource": lab})
        self.clock[0] = T0 + 600
        self.x.round()
        self.assertIsNone(g["pause_since"])
        self.assertIn("RESUMED biz-rtx4080: mining again on PC m1 after 3 min paused for Gemma", self.events())
        self.assertEqual(g["paused_t"], T0 + 420)               # its end still gives the next readings their warm-up
        self.assertFalse(self.x.paused(g, self.clock[0]))


class TestPickMargin(Base):
    """The search, and a waiting group moving up a priority, price a class at a slower-than-median PC (OPEN.md #69)."""
    def setUp(self):
        super().setUp()
        self.x.cfg["yield"] = 1.0
        t = self.clock[0]
        self.x.st["pc_rates"] = {"RTX 4080 (16 GB)": {f"m{i}": [r, t] for i, r in enumerate((150, 170, 185, 190, 200))}}
        self.http.on("GET", "/quotas", (200, {"container_groups_quotas": {"container_replicas_quota": 10}}))

    def test_the_rate_at_a_share_of_the_stored_pcs(self):
        self.assertEqual(self.x.rate("RTX 4080 (16 GB)"), 185)                       # the median
        self.assertAlmostEqual(self.x.rate("RTX 4080 (16 GB)", 0.35), 176)            # 170 + 0.4 of the way to 185
        self.assertAlmostEqual(self.x.pick_margin("RTX 4080 (16 GB)", "medium"), 1 - 0.19 / (176 * USD_TH_HR))

    def test_the_search_skips_a_price_only_the_median_pc_clears(self):
        # At medium the median PC (185 TH/s) makes +12%, but one at 176 makes +8%: 2 of the 5 stored PCs fail.
        self.assertGreater(self.x.margin("RTX 4080 (16 GB)", "medium"), 0.10)
        avail = {"c4080": {"available_gpu_medium": 5}}
        self.http.on("POST", "/availability/", lambda b: (200, avail.get(b["gpu_classes"][0], {})))
        self.x.cfg["max_active"] = 2
        self.x.search()
        self.assertFalse(self.http.made("POST", "/containers$"))
        self.assertIn("SEARCH found nothing to create", self.events())
        avail["c4080"]["available_gpu_low"] = 5                                     # low clears at 176: +27%
        self.x.search()
        made = [(c[2]["name"], c[2]["container"]["priority"]) for c in self.http.made("POST", "/containers$")]
        self.assertEqual(made, [("biz-rtx4080", "low")])

    def test_a_waiting_group_moves_up_only_to_a_price_a_slower_pc_clears(self):
        g = self.group(pri="low")
        self.step(g, 10, None)
        self.assertFalse(self.http.made("PATCH", ""))
        self.assertEqual(g["end_kind"], "released")
        self.assertIn("no PC at low and no higher priority clears 10%", self.events())


class TestMoveCountReset(Base):
    """A group's count of moves starts again after 2 h above the bar on one PC (OPEN.md #68)."""
    def test_the_count_starts_again_after_2_h_on_a_pc(self):
        g = self.group(pri="medium")
        g["reallocs"] = 2
        self.step(g, 1, "m1")
        self.mine(g, "m1", 199, 119)                 # 119 min on the PC
        self.assertEqual(g["reallocs"], 2)
        self.mine(g, "m1", 199, 1)
        self.assertEqual(g["reallocs"], 0)
        self.assertIn("MOVES biz-rtx4080: 2 h above 10% on PC m1, so its count of moves starts again (it had used 2 "
                      "of 3)", self.events())

    def test_a_restart_on_the_pc_starts_the_2_h_again(self):
        g = self.group(pri="medium")
        g["reallocs"] = 1
        self.step(g, 1, "m1")
        self.mine(g, "m1", 199, 60)
        g["restart_t"] = self.clock[0]               # Salad restarted the box here an hour in
        self.mine(g, "m1", 199, 70)
        self.assertEqual(g["reallocs"], 1)
        self.mine(g, "m1", 199, 50)
        self.assertEqual(g["reallocs"], 0)


class TestNodeIdentity(Base):
    """A Gemma box keeps one LLMJob node identity across its PCs, so the server's speed test (and its pause in
    mining) comes once a day a group, not once a PC (OPEN.md #11)."""
    def node(self, cmd):
        m = re.search(r"printf '%s' '(\{.*?\})' > \$d/node.json", cmd)
        return json.loads(m.group(1)) if m else None

    def test_a_gemma_box_keeps_one_identity_across_commands(self):
        self.x.cfg["llm"] = True
        a = self.node(self.x.box("biz-rtx4080", False)[1][2])
        b = self.node(self.x.box("biz-rtx4080", True)[1][2])
        self.assertEqual(a, b)
        self.assertEqual(set(a), {"nodeId", "publicKey", "secretKey", "name", "connected", "createdAt"})
        self.assertIsNone(a["name"])
        self.assertFalse(a["connected"])
        # The CLI's form: tweetnacl's 64-byte secret key (seed, then public key) and the id the server computes.
        pub, sec = w.base64.b64decode(a["publicKey"]), w.base64.b64decode(a["secretKey"])
        self.assertEqual((len(pub), len(sec), sec[32:]), (32, 64, pub))
        self.assertEqual(a["nodeId"], w.hashlib.sha256(a["publicKey"].encode()).hexdigest()[:16])
        sk = w.Ed25519PrivateKey.from_private_bytes(sec[:32])
        sk.public_key().verify(sk.sign(b"id:123"), b"id:123")
        self.assertEqual(sk.public_key().public_bytes(w.serialization.Encoding.Raw, w.serialization.PublicFormat.Raw),
                         pub)
        self.assertNotEqual(self.node(self.x.box("biz-rtx4080-2", False)[1][2]), a)   # another group, another node

    def test_no_identity_without_gemma_or_on_an_amd_box(self):
        self.assertNotIn("node.json", self.x.box("biz-rtx4080", False)[1][2])
        self.x.cfg["llm"] = True
        self.assertNotIn("node.json", self.x.box("biz-rx9070xt", False)[1][2])
        self.assertNotIn("biz-rx9070xt", self.x.st.get("node_keys", {}))

    def test_create_sends_the_group_s_identity(self):
        self.x.cfg["llm"] = True
        self.group()
        cmd = self.http.made("POST", "/containers$")[-1][2]["container"]["command"][2]
        self.assertEqual(self.node(cmd)["nodeId"], self.x.st["node_keys"]["biz-rtx4080"]["nodeId"])

    @needs_sh
    def test_the_step_writes_the_cli_s_node_json(self):
        import subprocess, stat
        k = self.x.node_key("biz-rtx4080")
        home = tempfile.mkdtemp()
        subprocess.run(["/bin/sh", "-c", w.node_step(k)], env={"HOME": home, "PATH": os.environ["PATH"]}, check=True)
        path = os.path.join(home, ".local", "share", "llmjob-earn", "node.json")
        with open(path, encoding="utf-8") as f:
            self.assertEqual(json.load(f), {"nodeId": k["nodeId"], "publicKey": k["publicKey"], "secretKey": k["secretKey"],
                                            "name": None, "connected": False, "createdAt": k["createdAt"]})
        self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)

    def test_an_identity_no_live_group_uses_goes_after_rate_days(self):
        self.x.node_key("biz-rtx4080")
        self.clock[0] += 7 * 86400 - 60
        self.x.record_rates(self.clock[0])
        self.assertIn("biz-rtx4080", self.x.st["node_keys"])
        self.clock[0] += 60
        self.x.record_rates(self.clock[0])
        self.assertNotIn("biz-rtx4080", self.x.st["node_keys"])


class TestFilesAreUtf8(Base):
    """Python on Windows (before 3.15) opens text in cp1252 unless told otherwise, which has no ⛏️: the watcher's
    first report there would have crashed it."""
    def test_every_file_the_watcher_writes_or_reads_is_opened_as_utf8(self):
        import builtins
        seen = []
        def spy(file, mode="r", *a, **kw):
            seen.append((os.path.basename(str(file)), mode, kw.get("encoding")))
            return builtins.open(file, mode, *a, **kw)
        w.open = spy
        try:
            w.event("PORT BLOCKED — ⛏️ ×3")
            w.save(w.STATE, {"note": "⏳"})
            w.load(w.STATE, None)
            self.group()
            self.x.write_report()
        finally:
            del w.open
        self.assertTrue(seen)
        self.assertEqual([s for s in seen if s[2] != "utf-8"], [])
        with open(os.path.join(TMP, "report.md"), encoding="utf-8") as f:
            self.assertIn("⏳", f.read())


if __name__ == "__main__":
    unittest.main()
