# Tests for bench.py's parsing and table, with no Salad calls, and for bench.sh: its helpers on their own, and the
# whole script in a dry run with every box command stubbed.
# Run: python3 -m unittest discover -s tuning/salad
import importlib.util, json, os, shutil, signal, subprocess, sys, tempfile, unittest
from unittest import mock

# Never an exported folder: these tests write status and log files, and a lane's live ones may be there.
TMP = tempfile.mkdtemp()
os.environ["SALAD_MINE_DIR"] = TMP
HERE = os.path.dirname(os.path.abspath(__file__))


def no_network(method, url, *a, **k):
    """Every bench module here gets this as its Salad client until a test puts in a stand-in: the proxy adds
    the Salad key, so a test that reached the real client could create a group."""
    raise AssertionError(f"a test reached the network: {method} {url}")


spec = importlib.util.spec_from_file_location("bench", os.path.join(HERE, "bench.py"))
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
b.w.http = no_network


def read(path):
    with open(path) as f:
        return f.read()


def load_bench(**env):
    """bench.py loaded afresh with these BENCH_ variables set (and every other one unset), its files in TMP
    whatever another test module set SALAD_MINE_DIR to."""
    keys = ("BENCH_SET", "BENCH_LANE", "BENCH_COMMIT", "BENCH_BASE_SHA", "BENCH_VERIFY_SECS", "SALAD_MINE_DIR")
    old = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ.pop(k, None)
    os.environ.update(env, SALAD_MINE_DIR=TMP)
    try:
        s = importlib.util.spec_from_file_location("bench_" + "_".join(env.values()), os.path.join(HERE, "bench.py"))
        m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)
        m.w.http = no_network
        return m
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

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

    def test_the_tests_leave_an_exported_salad_mine_dir_alone(self):
        # The README has the operator export SALAD_MINE_DIR for the lanes. Tests run in that shell must not write
        # their status and log lines into the live files there.
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "bench-amd-a.log"), "w") as f:
            f.write("live\n")
        r = subprocess.run([sys.executable, "-m", "unittest",
                            "test_bench.TestBenchAMD.test_a_result_keeps_the_copy_rate_hit_check_core_and_order",
                            "test_bench.TestBench.test_a_pc_where_no_miner_gets_work_is_not_recorded"],
                           cwd=HERE, env=dict(os.environ, SALAD_MINE_DIR=d), capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(os.listdir(d), ["bench-amd-a.log"])
        self.assertEqual(read(os.path.join(d, "bench-amd-a.log")), "live\n")


# An AMD box's lines: our core passes its hit check, then ours (first, with the warm-up) and SRBMiner.
AMD_LINES = [
    "[bench] 10:00:01 card AMD Radeon RX 9070 XT, ?, , , ? MiB, gfx1201",
    "[bench] 10:01:05 fetch ok 0123456789ab",
    "[bench] 10:01:10 hip | copy: 564 GB/s (read + write)",
    "[bench] 10:01:10 copy 564 GB/s",
    "[bench] 10:01:11 start build gfx1201",
    "[bench] 10:01:22 start hitcheck 600s",
    "[bench] 10:11:30 hitcheck PASS secs=600 hits=2202 verified=400 failed=0 salts=277",
    "[bench] 10:11:31 ready SRBMiner 3.7.1, ours HIP core 0123456789ab for gfx1201",
    "[bench] 10:12:00 start ours 60s",
    "[bench] 10:13:10 start ours 300s",
] + [f"[bench] 10:14:00 sample ours t={t} hs={90 if t < 60 else 94}000000000000 acc=1 rej=0 gpu=" for t in range(15, 301, 15)] \
  + ["[bench] 10:18:20 start srb 300s"] \
  + [f"[bench] 10:20:00 sample srb t={t} hs=80000000000000 acc=2 rej=0 gpu=" for t in range(15, 301, 15)] \
  + ["[bench] 10:23:40 all done"]
FULL_SHA = "002deae58de57e7311cd07031380aa4281601247"
FAILED_CHECK = ["[bench] 10:11:30 hitcheck FAIL secs=600 hits=410 verified=400 failed=2 salts=60 "
                "why=jackpot hash mismatch (fold/transcript)", "[bench] 10:11:30 skip ours: hit check failed"]


def salad(calls, instances=None, classes=None, logs=None):
    """A stand-in for watcher.http: records each call and answers like Salad. logs: (machine, line) pairs."""
    def http(method, url, body=None, timeout=None):
        calls.append((method, url, body))
        if url.endswith("/gpu-classes"):
            return 200, {"items": [{"name": n, "id": f"c{i}"} for i, n in enumerate(classes or [])]}
        if url.endswith("/instances"):
            return 200, {"instances": instances or []}
        if url.endswith("/log-entries"):
            return 200, {"items": [{"time": f"2026-10-22T{line.split()[1]}.{i:03d}Z", "text_log": line,
                                    "resource": {"labels": {"machine_id": mach}}}
                                   for i, (mach, line) in enumerate(logs or [])]}
        return 201, {}
    return http


class TestBenchRTX(unittest.TestCase):
    """The RTX 20- and 30-series rerun (BENCH_SET=rtx): its lanes, classes, box, miner order and table."""
    @classmethod
    def setUpClass(cls):
        cls.r = load_bench(BENCH_SET="rtx", BENCH_COMMIT="c0ffee")
        cls.l20 = load_bench(BENCH_SET="rtx", BENCH_LANE="20", BENCH_COMMIT="c0ffee")
        cls.l30 = load_bench(BENCH_SET="rtx", BENCH_LANE="30", BENCH_COMMIT="c0ffee")

    def bench(self, m, calls=None, classes=None, **st):
        old = m.w.http
        m.w.http = salad(calls if calls is not None else [], classes=classes)
        self.addCleanup(lambda: setattr(m.w, "http", old))
        x = m.Bench()
        x.wallet = "prl1x"
        x.st.update(**st)
        return x

    def test_each_lane_takes_one_series_with_its_own_files(self):
        self.assertEqual([os.path.basename(f) for f in (self.l20.STATE, self.l30.STATE)],
                         ["bench-rtx-20.json", "bench-rtx-30.json"])
        self.assertEqual([p[0] for p in self.l20.PLAN],
                         ["RTX 2080 Ti (11 GB)", "RTX 2080 (8 GB)", "RTX 2070 (8 GB)", "RTX 2060 (6 GB)"])
        self.assertEqual([p[0] for p in self.l30.PLAN][0], "RTX 3090 Ti (24 GB)")
        self.assertEqual(len(self.l30.PLAN), 8)
        self.assertEqual({p[2] for p in self.r.PLAN}, {300})
        # Each lane writes its own series only, and every class is in exactly one lane.
        self.assertEqual([t for t, _ in self.l20.SECTIONS], ["RTX 20-series (on Salad)"])
        self.assertEqual([t for t, _ in self.l30.SECTIONS], ["RTX 30-series (on Salad)"])
        self.assertEqual(len(self.l20.PLAN) + len(self.l30.PLAN), len(self.r.PLAN))

    def test_a_lane_from_another_set_is_refused(self):
        for env in ({"BENCH_SET": "rtx", "BENCH_LANE": "a"}, {"BENCH_LANE": "20"}):
            m = load_bench(BENCH_COMMIT="c0ffee", **env)
            with mock.patch.dict(os.environ, {"PRL_WALLET": "prl1x"}), self.assertRaises(SystemExit) as e:
                m.Bench().run()
            self.assertIn("BENCH_LANE", str(e.exception))

    def test_a_box_runs_the_release_on_salad_s_cuda_image(self):
        calls = []
        self.bench(self.l20, calls, classes=["RTX 2080 Ti (11 GB)"], cls=0, tries=0, group=None).create()
        body = next(c[2] for c in calls if c[0] == "POST")
        self.assertEqual(body["name"], "bench-20-rtx2080ti-1")      # not "biz-": the watcher leaves it alone
        self.assertEqual(body["container"]["image"], self.l20.w.IMAGE)
        self.assertEqual(body["container"]["priority"], "high")
        cmd = body["container"]["command"][2]
        self.assertIn("ORDER=ours,peak,srb OURS_SECS=300 bash /root/bench.sh", cmd)
        self.assertNotIn("amdtest.sh", cmd)

    def test_the_two_pcs_of_a_class_take_turns_going_first_each_in_a_group_of_its_own(self):
        res = {"machine": "m1" * 8, "card": "NVIDIA GeForce RTX 2080 Ti", "power": "250.00 W", "ours": (90.0, None),
               "peak": (92.0, None), "srb": (91.0, None)}
        x = self.bench(self.l20, cls=0, tries=0, group="bench-20-rtx2080ti-1")
        self.assertEqual(x.order(), "ours,peak,srb")
        x.st["results"] = {"RTX 2080 Ti": [res]}
        self.assertEqual(x.order(), "srb,peak,ours")
        x.move_on("done")                       # a moved PC would keep the first group's order
        self.assertIsNone(x.st["group"])

    def test_a_super_card_from_a_class_gets_a_row_of_its_own(self):
        self.assertEqual(self.r.row_for("NVIDIA GeForce RTX 2070 SUPER",
                                        ["RTX 2070"] + self.r.EXTRA["RTX 2070 (8 GB)"]), "RTX 2070 Super")
        st = {"results": {"RTX 2070": [{"machine": "abcdef1234", "power": "175.00 W", "ours": (50.0, None),
                                         "peak": (52.0, None), "srb": (51.0, None)}]}}
        text = self.l20.render(st)
        self.assertIn("| RTX 2070 | Salad (175 W) | abcdef12 | 50.0 | 52.0 | 51.0 | 96% |", text)
        self.assertNotIn("RTX 2070 Super", text)          # not waited for: no empty rows for it
        self.assertIn("| RTX 2060 | Salad (— W, 6 GB) | — | not run | not run | not run | — |", text)

    def test_table_puts_the_two_lanes_together(self):
        d = tempfile.mkdtemp()
        for lane, row in (("20", "RTX 2080"), ("30", "RTX 3070")):
            with open(f"{d}/{lane}.json", "w") as f:
                json.dump({"results": {row: [{"machine": "0123456789", "power": "200.00 W", "ours": (70.0, None),
                                              "peak": (69.0, None), "srb": (71.5, None)}]}}, f)
        text = self.r.table([f"{d}/20.json", f"{d}/30.json"])
        self.assertLess(text.index("## RTX 20-series"), text.index("## RTX 30-series"))
        self.assertIn("| RTX 2080 | Salad (200 W) | 01234567 | 70.0 | 69.0 | 71.5 | 98% |", text)
        self.assertIn("| RTX 3070 | Salad (200 W) | 01234567 | 70.0 | 69.0 | 71.5 | 98% |", text)


class TestBenchAMD(unittest.TestCase):
    """The Radeon set (BENCH_SET=amd): its plan, files, lanes, box command, lines, results and table."""
    @classmethod
    def setUpClass(cls):
        cls.a = load_bench(BENCH_SET="amd", BENCH_COMMIT="c0ffee")
        cls.la = load_bench(BENCH_SET="amd", BENCH_LANE="a", BENCH_COMMIT="c0ffee", BENCH_BASE_SHA="beef",
                            BENCH_VERIFY_SECS="900")
        cls.lb = load_bench(BENCH_SET="amd", BENCH_LANE="b", BENCH_COMMIT="c0ffee")
        cls.lc = load_bench(BENCH_SET="amd", BENCH_LANE="c", BENCH_COMMIT="c0ffee")

    def with_salad(self, m, calls, now=30_000, **kw):
        old = (m.w.http, m.w.now)
        m.w.http, m.w.now = salad(calls, **kw), (lambda: now)
        self.addCleanup(lambda: setattr(m.w, "http", old[0]) or setattr(m.w, "now", old[1]))

    def bench(self, m, **st):
        x = m.Bench()
        x.wallet = "prl1x"
        x.st.update(**st)
        return x

    def run_alone(self, m, clock=None):
        """For a test that calls run(): its state, status and log files in a folder of their own (run() saves its
        state, which the other tests' Bench() would load), a wallet, and sleep moving clock on, or stopping a
        loop that doesn't end."""
        d, start = tempfile.mkdtemp(), clock[0] if clock else 0
        def sleep(s):
            if clock is None or clock[0] - start > 2 * 3600:
                raise RuntimeError("run() didn't finish")
            clock[0] += s
        stack = mock.patch.multiple(m, STATE=f"{d}/s.json", STATUS=f"{d}/s.md", LOG=f"{d}/s.log")
        stack.start()
        self.addCleanup(stack.stop)
        for p in (mock.patch.dict(os.environ, {"PRL_WALLET": "prl1x"}), mock.patch.object(m.time, "sleep", sleep)):
            p.start()
            self.addCleanup(p.stop)

    def test_its_own_files_and_salad_s_class_names(self):
        self.assertTrue(self.a.STATE.endswith("bench-amd.json"))
        self.assertTrue(b.STATE.endswith("bench.json"))           # the GTX set keeps its files
        self.assertEqual([p[0] for p in self.a.PLAN][0], "AMD RX 9070 XT (16GB)")
        self.assertEqual(len(self.a.PLAN), 5)
        self.assertEqual({p[2] for p in self.a.PLAN}, {300})     # ours runs 5 minutes, like SRBMiner
        self.assertEqual(self.a.RUN_MIN, 50)
        self.assertEqual(b.RUN_MIN, 45)

    def test_each_lane_has_its_own_files_and_classes(self):
        for m, lane in ((self.la, "a"), (self.lb, "b"), (self.lc, "c")):
            self.assertEqual([os.path.basename(f) for f in (m.STATE, m.STATUS, m.LOG)],
                             [f"bench-amd-{lane}.json", f"bench-amd-{lane}_status.md", f"bench-amd-{lane}.log"])
        # Lane a keeps its first two classes in the same places, so a run already on them carries on.
        self.assertEqual([p[0] for p in self.la.PLAN], ["AMD RX 9070 XT (16GB)", "AMD RX 9060 XT (16GB)"])
        self.assertEqual([p[0] for p in self.lb.PLAN], ["AMD RX 7900 XTX (24GB)", "AMD RX 7900 XT (20GB)"])
        self.assertEqual([p[0] for p in self.lc.PLAN], ["AMD RX 7800 XT (16GB)"])
        self.assertEqual(self.la.SECTIONS[0][1], ["RX 9070 XT", "RX 9060 XT"])
        # Every card is in exactly one lane.
        self.assertEqual(sorted(c for cs in self.la.LANES.values() for c in cs), sorted(self.a.SECTIONS[0][1]))
        self.assertEqual(len(self.la.ALL_SECTIONS[0][1]), 5)
        gtx_a = load_bench(BENCH_LANE="a")            # a lane means nothing to the GTX set
        self.assertEqual(len(gtx_a.PLAN), len(b.PLAN))

    def test_a_lane_outside_the_amd_set_is_refused(self):
        m = load_bench(BENCH_LANE="a", BENCH_COMMIT="c0ffee")
        with mock.patch.dict(os.environ, {"PRL_WALLET": "prl1x"}), self.assertRaises(SystemExit) as e:
            m.Bench().run()
        self.assertIn("BENCH_LANE", str(e.exception))

    def test_the_box_builds_ours_and_runs_it_with_srbminer_on_the_rocm_image(self):
        calls = []
        self.with_salad(self.la, calls, classes=["AMD RX 9070 XT (16GB)"])
        self.bench(self.la, cls=0, tries=0, group=None).create()
        body = next(c[2] for c in calls if c[0] == "POST")
        self.assertEqual(body["name"], "bench-a-rx9070xt-1")
        c = body["container"]
        self.assertEqual(c["image"], "rocm/dev-ubuntu-24.04:7.2")
        self.assertEqual(c["resources"], {"cpu": 4, "memory": 8192, "gpu_classes": ["c0"],
                                          "storage_amount": 10737418240})
        cmd = c["command"][2]
        raw = "https://raw.githubusercontent.com/super3/llmjob/c0ffee/tuning/salad"
        self.assertIn(f"curl -fsSL {raw}/bench.sh -o /root/bench.sh && curl -fsSL {raw}/amdtest.sh -o "
                      "/root/amdtest.sh && W=prl1x ORDER=ours,srb OURS_SECS=300 BASE_SHA=beef VERIFY_SECS=900 "
                      "bash /root/bench.sh", cmd)
        self.assertNotIn("peak", cmd)
        self.assertNotIn("; curl", cmd)          # Salad refuses it

    def test_base_sha_and_hit_check_length_default_to_the_commit_and_600_s(self):
        calls = []
        self.with_salad(self.lb, calls, classes=["AMD RX 7900 XTX (24GB)"])
        self.bench(self.lb, cls=0, tries=0, group=None).create()
        body = next(c[2] for c in calls if c[0] == "POST")
        self.assertEqual(body["name"], "bench-b-rx7900xtx-1")
        self.assertIn("BASE_SHA=c0ffee VERIFY_SECS=600 ", body["container"]["command"][2])

    def test_the_two_pcs_of_a_class_take_turns_going_first(self):
        x = self.bench(self.la, cls=0, tries=0, group=None, results={})
        self.assertEqual(x.order(), "ours,srb")
        x.st["tries"] = 3                       # PCs skipped don't count
        self.assertEqual(x.order(), "ours,srb")
        x.st["results"] = {"RX 9070 XT": [{"machine": "m1"}]}
        self.assertEqual(x.order(), "srb,ours")

    def test_after_a_recorded_pc_the_next_one_gets_a_new_group_with_the_other_order(self):
        calls = []
        self.with_salad(self.la, calls, classes=["AMD RX 9070 XT (16GB)"])
        x = self.bench(self.la, cls=0, tries=0, group="bench-a-rx9070xt-1", instance="i1",
                       results={"RX 9070 XT": [{"machine": "m1"}]})
        x.move_on("done")
        self.assertIn(("DELETE", f"{self.la.w.CB}/bench-a-rx9070xt-1", None), calls)
        self.assertFalse(any(u.endswith("/reallocate") for _, u, _ in calls))
        self.assertIsNone(x.st["group"])
        x.create()
        body = [c[2] for c in calls if c[0] == "POST"][-1]
        self.assertEqual(body["name"], "bench-a-rx9070xt-2")
        self.assertIn("ORDER=srb,ours ", body["container"]["command"][2])
        # A PC that wasn't recorded is moved as before, keeping the group and its order.
        calls.clear()
        x.st.update(group="bench-a-rx9070xt-2", instance="i2")
        x.move_on("PC 1234abcd was benchmarked already")
        self.assertIn(("POST", f"{self.la.w.CB}/bench-a-rx9070xt-2/instances/i2/reallocate", None), calls)

    def test_amd_card_names_match_their_rows_and_xt_is_not_xtx(self):
        rows = ["RX 9070 XT", "RX 9060 XT", "RX 7900 XTX", "RX 7900 XT", "RX 7800 XT"]
        self.assertEqual(self.a.row_for("Radeon RX 7900 XTX", rows), "RX 7900 XTX")
        self.assertEqual(self.a.row_for("AMD Radeon RX 7900 XT", rows), "RX 7900 XT")
        self.assertIsNone(self.a.row_for("AMD Radeon RX 7900 GRE", rows))

    def test_parse_reads_the_copy_rate_hit_check_and_core(self):
        r = self.a.parse(AMD_LINES)
        self.assertEqual(r["copy"], 564.0)
        self.assertEqual(r["core"], "0123456789ab")
        self.assertEqual(r["hitcheck"], {"pass": True, "secs": 600, "hits": 2202, "verified": 400, "failed": 0,
                                         "salts": 277})
        self.assertEqual(r["skipped"], {})
        self.assertIsNone(r["fail"])
        self.assertEqual(self.a.figure(r["miners"]["ours"], mean_after=60), (94.0, None))
        self.assertEqual(self.a.figure(r["miners"]["srb"]), (80.0, None))
        self.assertIsNone(r["miners"].get("peak"))

    def test_parse_reads_a_failed_hit_check_without_failing_the_box(self):
        r = self.a.parse(FAILED_CHECK)
        self.assertEqual(r["hitcheck"], {"pass": False, "secs": 600, "hits": 410, "verified": 400, "failed": 2,
                                         "salts": 60, "why": "jackpot hash mismatch (fold/transcript)"})
        self.assertEqual(r["skipped"], {"ours": "hit check failed"})
        self.assertIsNone(r["fail"])            # not the box's FAIL: SRBMiner still runs
        r = self.a.parse(["[bench] 10:11:30 hitcheck FAIL secs=600 error=no result: Error: no HIP device"])
        self.assertEqual(r["hitcheck"], {"pass": False, "secs": 600, "error": "no result: Error: no HIP device"})
        self.assertEqual(self.a.parse(["[bench] 10:01:10 copy ? GB/s"])["copy"], None)

    def done(self, m, lines, order="ours,srb"):
        """One tick on a box whose lines are all in: what it recorded."""
        calls = []
        self.with_salad(m, calls, now=50_000, instances=[{"machine_id": "83151dba-xx", "id": "i1",
                                                          "state": "running"}])
        x = self.bench(m, cls=0, tries=0, group="bench-a-rx9070xt-1", created=48_000, machine="83151dba-xx",
                       instance="i1", pc_t=48_000, log_t=49_900, lines=[f"t|{l}" for l in lines], results={},
                       notes=[], order=order)
        x.tick()
        return x, calls

    def test_a_result_keeps_the_copy_rate_hit_check_core_and_order(self):
        x, calls = self.done(self.la, AMD_LINES)
        h = x.st["results"]["RX 9070 XT"][0]
        self.assertEqual(h["ours"], (94.0, None))         # the mean after the first minute
        self.assertEqual(h["peak"], (None, "not run"))
        self.assertEqual(h["srb"], (80.0, None))
        self.assertEqual(h["copy"], 564.0)
        self.assertEqual(h["hitcheck"]["hits"], 2202)
        self.assertEqual((h["core"], h["order"]), ("0123456789ab", "ours,srb"))
        self.assertIn(("DELETE", f"{self.la.w.CB}/bench-a-rx9070xt-1", None), calls)   # next PC: a new group

    def test_a_failed_hit_check_records_srbminer_and_says_why_ours_is_missing(self):
        lines = [l for l in AMD_LINES if " ours " not in l and "hitcheck" not in l] + FAILED_CHECK
        x, _ = self.done(self.la, lines)
        h = x.st["results"]["RX 9070 XT"][0]
        self.assertEqual(h["ours"], (None, "hit check failed"))
        self.assertEqual(h["srb"], (80.0, None))
        self.assertFalse(h["hitcheck"]["pass"])
        self.assertNotIn("83151dba", " ".join(x.st["notes"]))   # a result, not a failed PC

    def test_render_has_peakminer_not_run_percent_of_srbminer_and_a_line_for_each_pc(self):
        x, _ = self.done(self.la, AMD_LINES)
        lines = [l for l in AMD_LINES if " ours " not in l and "hitcheck" not in l] + FAILED_CHECK
        x2, _ = self.done(self.la, lines, order="srb,ours")
        x.st["results"]["RX 9070 XT"].append(dict(x2.st["results"]["RX 9070 XT"][0], machine="3d6cb276-yy"))
        text = self.la.render(x.st)
        self.assertIn("| RX 9070 XT | Salad (— W, 16 GB) | 83151dba | 94.0 | not run | 80.0 | 118% |", text)
        self.assertIn("| RX 9070 XT | Salad (— W, 16 GB) | 3d6cb276 | hit check failed | not run | 80.0 | — |", text)
        self.assertIn("- `83151dba` (RX 9070 XT): copy 564 GB/s; hit check PASS, 400 of 2202 hits verified, 0 failed, "
                      "277 salts in 600 s, 7.95 hits a salt; ours ran first.", text)
        self.assertIn("- `3d6cb276` (RX 9070 XT): copy 564 GB/s; hit check FAIL, 400 of 410 hits verified, 2 failed, "
                      "60 salts in 600 s, 6.83 hits a salt (jackpot hash mismatch (fold/transcript)); "
                      "SRBMiner ran first.", text)
        self.assertIn("| RX 9060 XT | Salad (— W, 16 GB) | — | not run | not run | not run | — |", text)
        self.assertNotIn("RX 7900", text)                 # lane b's rows
        self.assertNotIn("RX 7800", text)                 # lane c's
        self.assertNotIn("hits a salt", b.render({"results": {}}))

    def test_table_puts_the_two_lanes_together_in_benchmark_md_s_order(self):
        d = tempfile.mkdtemp()
        ha = {"machine": "aaaa1111", "power": "", "ours": [93.8, None], "peak": [None, "not run"],
              "srb": [79.4, None], "copy": 564, "hitcheck": None}
        hb = dict(ha, machine="bbbb2222", ours=[50.4, None], srb=[52.8, None])
        b.w.save(os.path.join(d, "a.json"), {"results": {"RX 9070 XT": [ha]}})
        b.w.save(os.path.join(d, "b.json"), {"results": {"RX 7900 XTX": [hb]}})
        text = self.la.table([os.path.join(d, "a.json"), os.path.join(d, "b.json"), os.path.join(d, "none.json")])
        rows = [l for l in text.splitlines() if l.startswith("| RX")]
        self.assertEqual(len(rows), 10)
        self.assertEqual(rows[0], "| RX 9070 XT | Salad (— W, 16 GB) | aaaa1111 | 93.8 | not run | 79.4 | 118% |")
        self.assertEqual(rows[4], "| RX 7900 XTX | Salad (— W, 24 GB) | bbbb2222 | 50.4 | not run | 52.8 | 95% |")
        self.assertIn("- `aaaa1111` (RX 9070 XT): copy 564 GB/s; no hit check.", text)

    def test_a_failed_check_on_the_pc_meant_to_go_first_says_srbminer_went_first(self):
        # ORDER=ours,srb, but bench.sh drops ours after the failed check, so SRBMiner takes the warm-up.
        lines = [l for l in AMD_LINES if " ours " not in l and "hitcheck" not in l] + FAILED_CHECK
        x, _ = self.done(self.la, lines, order="ours,srb")
        h = x.st["results"]["RX 9070 XT"][0]
        self.assertEqual(h["order"], "srb")
        self.assertIn("(fold/transcript)); SRBMiner ran first.", self.la.render(x.st))
        x, _ = self.done(self.la, AMD_LINES, order="ours,srb")      # one that passed keeps its order
        self.assertEqual(x.st["results"]["RX 9070 XT"][0]["order"], "ours,srb")

    def test_a_pc_salad_moves_the_box_to_by_itself_gets_a_row_of_its_own_lines(self):
        # PC aaaaaaaa printed some lines, then Salad moved the instance to bbbbbbbb with no word from us, and B
        # printed a whole passing run. B's row must not keep A's skip or average in A's samples.
        a_fail = ["[bench] 09:30:01 card AMD Radeon RX 9070 XT, ?, , , ? MiB, gfx1201",
                  "[bench] 09:31:10 copy 120 GB/s"] + FAILED_CHECK \
            + [f"[bench] 09:42:00 sample srb t={t} hs=70000000000000 acc=1 rej=0 gpu=" for t in (15, 30)]
        a_ours = AMD_LINES[:7] + [f"[bench] 09:45:00 sample ours t={t} hs=50000000000000 acc=1 rej=0 gpu="
                                  for t in range(15, 301, 15)]
        for a_lines in (a_fail, a_ours):
            calls = []
            logs = [("aaaaaaaa-pc", l) for l in a_lines[-3:]] + [("bbbbbbbb-pc", l) for l in AMD_LINES]
            self.with_salad(self.la, calls, now=50_000, logs=logs,
                            instances=[{"machine_id": "bbbbbbbb-pc", "id": "i2", "state": "running"}])
            x = self.bench(self.la, cls=0, tries=0, group="bench-a-rx9070xt-1", created=40_000,
                           machine="aaaaaaaa-pc", instance="i1", pc_t=40_000, log_t=49_000,
                           lines=[f"t|{l}" for l in a_lines], results={}, notes=[], seen=[], order="ours,srb")
            x.tick()
            h = x.st["results"]["RX 9070 XT"][0]
            self.assertEqual((h["machine"], h["ours"], h["srb"], h["copy"]),
                             ("bbbbbbbb-pc", (94.0, None), (80.0, None), 564.0))
            self.assertTrue(h["hitcheck"]["pass"])
            self.assertEqual(h["order"], "ours,srb")
            self.assertIn("PC bench-a-rx9070xt-1: bbbbbbbb (was aaaaaaaa)", read(self.la.LOG))
            body = next(c[2] for c in calls if c[1].endswith("/log-entries"))
            self.assertEqual(body["start_time"], self.la.w.iso(50_000 - 120))    # B's first lines, read again

    def test_an_amd_run_needs_a_full_base_sha(self):
        # A short one makes git fetch fail on every box ("couldn't find remote ref"), so the run is refused.
        for env in ({"BENCH_COMMIT": "002deae"}, {"BENCH_COMMIT": FULL_SHA, "BENCH_BASE_SHA": "5aa0069"}):
            m = load_bench(BENCH_SET="amd", BENCH_LANE="a", **env)
            self.run_alone(m)
            with self.assertRaises(SystemExit) as e:
                m.Bench().run()
            self.assertIn("full 40-character commit", str(e.exception))
        for set_, env in (("amd", {"BENCH_COMMIT": "002deae", "BENCH_BASE_SHA": FULL_SHA}),
                          ("amd", {"BENCH_COMMIT": FULL_SHA}),
                          ("gtx", {"BENCH_COMMIT": "002deae"})):       # the GTX set only puts it in the URL
            m = load_bench(BENCH_SET=set_, **env)
            self.run_alone(m)
            x = m.Bench()
            with mock.patch.object(x, "tick", return_value=False):
                x.run()
            self.assertIn("FINISHED", read(m.LOG))

    def test_the_time_given_a_box_grows_with_the_hit_check(self):
        self.assertEqual(self.a.RUN_MIN, 50)
        self.assertEqual(self.la.RUN_MIN, 55)                    # BENCH_VERIFY_SECS=900
        self.assertEqual(load_bench(BENCH_SET="amd", BENCH_VERIFY_SECS="1800").RUN_MIN, 70)
        self.assertEqual(load_bench(BENCH_SET="amd", BENCH_VERIFY_SECS="601").RUN_MIN, 51)
        self.assertEqual(load_bench(BENCH_VERIFY_SECS="1800").RUN_MIN, 45)     # the GTX set runs no check

    def test_a_deleted_group_is_looked_at_until_salad_says_it_is_gone(self):
        now, calls, gone_at = [60_000], [], [None]
        def http(method, url, body=None, timeout=None):
            calls.append((method, url))
            if method == "GET":
                return (404, None) if gone_at[0] and now[0] >= gone_at[0] else (200, {"name": "x"})
            return (202, None) if method == "DELETE" else (201, {})
        old = (self.la.w.http, self.la.w.now)
        self.la.w.http, self.la.w.now = http, (lambda: now[0])
        self.addCleanup(lambda: setattr(self.la.w, "http", old[0]) or setattr(self.la.w, "now", old[1]))
        g = "bench-a-rx9070xt-1"
        x = self.bench(self.la, group=g, notes=[], gone={})
        x.delete()                                # Salad says 202 and keeps it
        self.assertEqual(list(x.st["gone"]), [g])
        seen = []
        for _ in range(16):                       # a minute a tick
            calls.clear()
            now[0] += 60
            x.check_gone()
            seen.append([m for m, _ in calls])
        self.assertEqual(seen[:5], [[], [], [], [], ["GET", "DELETE"]])        # 5 min after each delete
        self.assertEqual(seen[9], ["GET", "DELETE"])
        self.assertEqual(len([n for n in x.st["notes"] if g in n]), 1)        # said once, at 10 min
        self.assertIn("still on Salad 10 min after its delete", x.st["notes"][-1])
        gone_at[0] = now[0] + 60
        now[0] += 5 * 60
        x.check_gone()
        self.assertEqual(x.st["gone"], {})
        self.assertIn(f"GONE {g}", read(self.la.LOG))
        # One Salad has already removed (404) isn't looked at again.
        self.la.w.http = lambda method, url, body=None, timeout=None: (404, None)
        x.st["group"] = "bench-a-rx9070xt-2"
        x.delete()
        self.assertEqual(x.st["gone"], {})

    def test_the_run_ends_only_once_salad_has_removed_its_last_group(self):
        now, calls = [70_000], []
        def http(method, url, body=None, timeout=None):
            calls.append((method, url))
            if method == "GET":
                return (404, None) if now[0] >= 70_000 + 11 * 60 else (200, {})
            return 0, {"error": "timed out"}       # a delete that didn't get through
        m = load_bench(BENCH_SET="amd", BENCH_LANE="a", BENCH_COMMIT=FULL_SHA)
        m.w.http, m.w.now = http, (lambda: now[0])
        self.run_alone(m, clock=now)
        x = self.bench(m, cls=len(m.PLAN), group="bench-a-rx7800xt-2")
        x.delete()
        self.assertEqual(x.now_line(), "All classes done. Waiting for Salad to remove bench-a-rx7800xt-2.")
        x.run()
        self.assertGreaterEqual(now[0], 70_000 + 11 * 60)
        self.assertEqual(x.st["gone"], {})
        self.assertEqual(x.now_line(), "All classes done.")
        self.assertEqual(calls.count(("DELETE", f"{m.w.CB}/bench-a-rx7800xt-2")), 3)   # at 0, 5 and 10 min
        self.assertIn("FINISHED", read(m.LOG))

    def test_the_gtx_command_is_unchanged(self):
        calls = []
        old = (b.w.http, b.w.now)
        b.w.http, b.w.now = salad(calls, classes=["GTX 1660 (6 GB)"]), (lambda: 30_000)
        try:
            x = b.Bench()
            x.wallet = "prl1x"
            x.st.update(cls=next(i for i, p in enumerate(b.PLAN) if p[0] == "GTX 1660 (6 GB)"), tries=1, group=None)
            x.create()
        finally:
            b.w.http, b.w.now = old
        body = next(c[2] for c in calls if c[0] == "POST")
        self.assertEqual(body["name"], "bench-gtxgtx1660-2")
        self.assertEqual(body["container"]["resources"]["cpu"], 2)
        self.assertEqual(body["container"]["resources"]["memory"], 4096)
        self.assertEqual(body["container"]["command"][2],
                         "apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq curl ca-certificates >/dev/null "
                         f"2>&1 && curl -fsSL https://raw.githubusercontent.com/super3/llmjob/{b.COMMIT}/tuning/salad/"
                         "bench.sh -o /root/bench.sh && W=prl1x ORDER=srb,peak,ours OURS_SECS=300 bash /root/bench.sh "
                         "2>&1 | tee -a /root/bench.log")


def bench_sh_part(start, end):
    """bench.sh's text from the line starting `start` up to the one starting `end`."""
    src = read(os.path.join(HERE, "bench.sh"))
    i = src.index("\n" + start) + 1
    return src[i:src.index("\n" + end, i) + 1]


def bash(script, cwd=None, env=None):
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, cwd=cwd,
                          env=dict(os.environ, **(env or {})))


@unittest.skipUnless(shutil.which("jq"), "needs jq, as the box has")
class TestBenchShPieces(unittest.TestCase):
    """bench.sh's AMD helpers, cut out of the script and run here."""
    HELPERS = None

    @classmethod
    def setUpClass(cls):
        cls.HELPERS = bench_sh_part("say() {", "fail() {") + bench_sh_part("# drop NAME", "# Our miner on AMD")

    def test_syntax(self):
        r = subprocess.run(["bash", "-n", os.path.join(HERE, "bench.sh")], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_drop_leaves_a_miner_out_of_order(self):
        r = bash(self.HELPERS + 'ORDER=ours,srb; drop ours "hit check failed"; echo "ORDER=$ORDER"\n'
                 'unset ORDER; drop peak; echo "ORDER=$ORDER"\nORDER=srb,ours; drop peak; echo "ORDER=$ORDER"')
        out = r.stdout.splitlines()
        self.assertRegex(out[0], r"^\[bench\] \d\d:\d\d:\d\d skip ours: hit check failed$")
        self.assertEqual(out[1:], ["ORDER=srb", "ORDER=srb,ours", "ORDER=srb,ours"])

    def test_copy_rate_reads_the_hip_test_s_copy_line(self):
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "hip.log"), "w") as f:
            f.write("devices 1 (no error)\nmadd: 0 of 16777216 wrong\ncopy: 564 GB/s (read + write)\nPASS\n")
        self.assertEqual(bash(self.HELPERS + f"copy_rate {d}/hip.log").stdout, "564\n")
        self.assertEqual(bash(self.HELPERS + f"copy_rate {d}/none.log").stdout, "")

    def hitcheck(self, log_text):
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "verify.log"), "w") as f:
            f.write(log_text)
        r = bash(self.HELPERS + f"hitcheck {d}/verify.log; echo rc=$?", env={"VERIFY_SECS": "600"})
        line, rc = r.stdout.splitlines()
        self.assertLess(len(line), 400)          # Salad's log service drops long lines
        return line.split(" ", 2)[2], rc

    def test_hitcheck_line_from_verify_hits_json(self):
        res = {"file": "build", "hits": 2202, "verified": 400, "failed": 0, "salts": 277, "fails": [], "PASS": True,
               "device": {"fold": "x" * 300}, "saltSeenS": list(range(277))}
        self.assertEqual(self.hitcheck("warning: x\n" + json.dumps(res) + "\n"),
                         ("hitcheck PASS secs=600 hits=2202 verified=400 failed=0 salts=277", "rc=0"))
        res.update(failed=2, PASS=False, fails=[{"nonce": 1, "salt": 2, "why": "jackpot hash mismatch (fold/transcript)"}])
        line, rc = self.hitcheck(json.dumps(res))
        self.assertEqual(line, "hitcheck FAIL secs=600 hits=2202 verified=400 failed=2 salts=277 "
                               "why=jackpot hash mismatch (fold/transcript)")
        self.assertEqual(rc, "rc=1")
        self.assertEqual(b.hit_check(line)["why"], "jackpot hash mismatch (fold/transcript)")
        # PASS needs at least 20 verified and 3 salts too: verify-hits says so, and that's what counts.
        res.update(failed=0, PASS=False, fails=[], verified=5, hits=5, salts=1)
        self.assertEqual(self.hitcheck(json.dumps(res))[0],
                         "hitcheck FAIL secs=600 hits=5 verified=5 failed=0 salts=1")

    def test_hitcheck_says_why_there_is_no_result(self):
        self.assertEqual(self.hitcheck('{"error":"Error: hipErrorNoDevice"}\n'),
                         ("hitcheck FAIL secs=600 error=Error: hipErrorNoDevice", "rc=1"))
        self.assertEqual(self.hitcheck("Error: Cannot find module 'x'\n  at y\n" + "z" * 500 + "\n")[0],
                         "hitcheck FAIL secs=600 error=no result: " + "z" * 160)

    def test_sample_reads_the_cli_s_rate_line(self):
        # The line as the CLI prints it (earn/test/earnCli.test.js), with a thousands comma.
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "ours.log"), "w") as f:
            f.write("LLMJob Earn CLI v0.6.7\n⛏  93.8 TH/s · 1,234 accepted · 1 rejected · up 05m 00s\n")
        r = bash(bench_sh_part("say() {", "fail() {") + bench_sh_part("# One sample:", "run() {") +
                 "AMD=1; SMI=true; sample ours 75", cwd=d, env={"LC_ALL": "C"})
        self.assertRegex(r.stdout, r"sample ours t=75 hs=93800000000000 acc=1234 rej=1 gpu=$")


# A dry run of bench.sh: every command it runs on a box is a stub here (apt-get, curl, the miners, nvidia-smi or
# rocminfo and amd-smi, git, npm, node, hip-clang), the box's folders are moved into a temporary one, and date
# and sleep share a fake clock, so a whole run takes a few seconds. The stubs log what they were asked to CALLS.
STUBS = {
    "apt-get": "exit 0",
    "md5sum": "cat >/dev/null",
    "sha256sum": "cat >/dev/null",
    "pkill": 'echo "pkill $*" >> "$CALLS"',
    "date": 'case "$*" in "+%s") cat "$CLOCK" ;; *%T*) "$REAL_DATE" -u -d "@$(cat "$CLOCK")" +%T ;; '
            '*) exec "$REAL_DATE" "$@" ;; esac',
    "sleep": '[ "$1" = infinity ] && { kill -TERM $PPID; exit 0; }\n'
             'echo $(( $(cat "$CLOCK") + ${1%.*} )) > "$CLOCK"; "$REAL_SLEEP" 0.02',
    "curl": r'''out=; url=
while [ $# -gt 0 ]; do case "$1" in -o) out=$2; shift ;; -m|--retry) shift ;; -*) ;; *) url=$1 ;; esac; shift; done
case "$url" in http://127.0.0.1*) ;; *) echo "curl $url${out:+ -> $out}" >> "$CALLS" ;; esac
case "$url" in
  *127.0.0.1:21550*) echo '{"algorithms":[{"hashrate":{"gpu":{"total":80000000000000}},"shares":{"accepted":3,"rejected":0}}]}' ;;
  *127.0.0.1:4068*) echo '{"hashrate":40000000000000,"accepted":1,"rejected":0}' ;;
  *SRBMiner*) mkdir -p SRBMiner-Multi-3-7-1 && cp "$STUBS/srbminer" SRBMiner-Multi-3-7-1/SRBMiner-MULTI \
                && tar czf "$out" SRBMiner-Multi-3-7-1 && rm -rf SRBMiner-Multi-3-7-1 ;;
  *peakminer*|*llmjob-earn-cli-linux) cp "$STUBS/miner" "$out" ;;
  *) : > "$out" ;;
esac''',
    "miner": 'echo "$(basename "$0") $*" >> "$CALLS"\necho "⛏  30.0 TH/s · 1 accepted · 0 rejected · up 00m 05s"\n'
             'exec "$REAL_SLEEP" 30',
    "srbminer": 'echo "srbminer $*" >> "$CALLS"\ncase " $* " in *" --list-devices "*) echo "GPU0 Radeon"; exit 0 ;; esac\n'
                'echo "SRBMiner-MULTI mining"; exec "$REAL_SLEEP" 30',
    "rocminfo": "printf 'Agent 1\\n  Name: AMD Ryzen 7\\n  Marketing Name: AMD Ryzen 7\\n  Device Type: CPU\\n"
                "Agent 2\\n  Name: gfx1201\\n  Marketing Name: AMD Radeon RX 9070 XT\\n  Device Type: GPU\\n'",
    "amd-smi": 'case "$1" in monitor) echo "GPU,POWER,GPU_TEMP" ;; esac',
    "clinfo": 'echo "Platform #0: AMD Accelerated Parallel Processing"',
    "git": r'''[ "$1" = -C ] && { cd "$2"; shift 2; }
echo "git $*" >> "$CALLS"
case "$1" in
  checkout) mkdir -p earn/native/amd tuning/salad && : > tuning/salad/amdtest_hip.cpp
    printf '#!/bin/bash\necho "build-amd.sh $*"\n[ "${FAKE_BUILD_RC:-0}" = 0 ] || { echo "error: fake"; exit 1; }\nmkdir -p build && : > build/pearl_core_hip.node\n' > earn/native/amd/build-amd.sh
    chmod +x earn/native/amd/build-amd.sh ;;
  rev-parse) echo 0123456789ab ;;
esac''',
    "npm": 'mkdir -p "$AMDTEST_NM/node_modules/node-addon-api" && : > "$AMDTEST_NM/node_modules/node-addon-api/napi.h"',
    "node": r'''case "$1" in
  -v) echo v22.22.0 ;;
  ../probes/verify-hits.js) echo "node $*" >> "$CALLS"; cat "$FAKE_VERIFY" ;;
  src/cli/earn-cli.js) echo "node $* | PEARL_CORE_VARIANT=$PEARL_CORE_VARIANT PEARL_CORE_PATH=$PEARL_CORE_PATH NODE_PATH=$NODE_PATH" >> "$CALLS"
    echo "LLMJob Earn CLI v0.6.7"; echo "⛏  93.8 TH/s · 4 accepted · 0 rejected · up 00m 05s"; exec "$REAL_SLEEP" 30 ;;
esac''',
    "clang++": r'''while [ $# -gt 0 ]; do [ "$1" = -o ] && out=$2; shift; done
printf '#!/bin/bash\necho "devices 1 (no error)"\necho "copy: 564 GB/s (read + write)"\necho PASS\n' > "$out"; chmod +x "$out"''',
    "ldd": 'echo "libamdhip64.so.7 => /opt/rocm/lib/libamdhip64.so.7 (0x0)"',
    "nvidia-smi": 'case "$*" in *power.draw*) echo "100.5, 1800, 60" ;; '
                  '*) echo "NVIDIA GeForce GTX 1660, 550.54, 120.00 W, 120.00 W, 6144 MiB, 7.5" ;; esac',
}
VERIFY_PASS = {"file": "build", "hits": 2202, "verified": 400, "failed": 0, "salts": 277, "fails": [], "PASS": True}


def dry_run(src, amd, env=None, verify=None, amdtest_src=None):
    """Runs bench.sh's text src with the stubs, as an AMD box or an NVIDIA one. Returns its output and the stubs'
    log, with the temporary folder shown as <T>."""
    t = tempfile.mkdtemp()
    stubs = os.path.join(t, "stubs")
    os.makedirs(stubs)
    for d in ("root", "b"):
        os.makedirs(os.path.join(t, d))
    for name, body in STUBS.items():
        if name == "nvidia-smi" and amd:
            continue
        with open(os.path.join(stubs, name), "w") as f:
            f.write("#!/bin/bash\n" + body + "\n")
        os.chmod(os.path.join(stubs, name), 0o755)
    src = (src.replace("/opt/b", f"{t}/b").replace("/root/", f"{t}/root/").replace("test -c /dev/dxg", "true")
           .replace("/opt/rocm-wsl/bin/amd-smi", f"{stubs}/amd-smi"))
    with open(os.path.join(t, "bench.sh"), "w") as f:
        f.write(src)
    amdtest = amdtest_src or read(os.path.join(HERE, "amdtest.sh"))
    with open(os.path.join(t, "amdtest.sh"), "w") as f:
        f.write(amdtest.replace("/root/", f"{t}/root/").replace("/opt/rocm/lib/llvm/bin/clang++", f"{stubs}/clang++")
                .replace("/opt/node", f"{t}/node"))
    with open(os.path.join(t, "verify.json"), "w") as f:
        f.write(json.dumps(verify or VERIFY_PASS) + "\n")
    with open(os.path.join(t, "clock"), "w") as f:
        f.write("1000000\n")
    e = {"PATH": f"{stubs}:/usr/bin:/bin", "HOME": f"{t}/root", "CALLS": f"{t}/calls", "CLOCK": f"{t}/clock",
         "STUBS": stubs, "REAL_DATE": shutil.which("date"), "REAL_SLEEP": shutil.which("sleep"),
         "AMDTEST_ROOT": f"{t}/llmjob", "AMDTEST_NM": f"{t}/nm", "FAKE_VERIFY": f"{t}/verify.json", "W": "prl1test"}
    e.update(env or {})
    p = subprocess.Popen(["bash", f"{t}/bench.sh"], cwd=t, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, start_new_session=True)
    try:
        out = p.communicate(timeout=120)[0]
    finally:
        try:
            os.killpg(p.pid, signal.SIGKILL)       # the miners' stubs, still waiting
        except ProcessLookupError:
            pass
    calls = read(f"{t}/calls") if os.path.exists(f"{t}/calls") else ""
    shutil.rmtree(t, ignore_errors=True)
    return out.replace(t, "<T>"), calls.replace(t, "<T>")


@unittest.skipUnless(shutil.which("jq") and shutil.which("timeout") and shutil.which("tar"), "needs jq, timeout, tar")
@unittest.skipIf(shutil.which("nvidia-smi", path="/usr/bin:/bin") or shutil.which("rocminfo", path="/usr/bin:/bin"),
                 "a real GPU tool would be found before the stubs")
class TestBenchShDryRun(unittest.TestCase):
    """bench.sh from start to end with stubs: what an NVIDIA box and an AMD box run, in order."""
    SRC = None

    @classmethod
    def setUpClass(cls):
        cls.SRC = read(os.path.join(HERE, "bench.sh"))

    def bench_lines(self, out):
        return [l for l in out.splitlines() if l.startswith("[bench] ")]

    def test_an_nvidia_box_runs_the_three_miners_as_before(self):
        out, calls = dry_run(self.SRC, amd=False, env={"ORDER": "peak,srb,ours", "OURS_SECS": "300"})
        rel = "https://github.com/super3/llmjob/releases/download/v0.6.9"
        self.assertEqual([l for l in calls.splitlines() if not l.startswith("pkill")], [
            "curl https://github.com/peakminer/peakminer/releases/download/v2.20.3/peakminer-2.20.3-linux-x86_64 -> peak",
            "curl https://github.com/doktor83/SRBMiner-Multi/releases/download/3.8.0/SRBMiner-Multi-3-8-0-Linux.tar.gz"
            " -> srb.tgz",
            f"curl {rel}/llmjob-earn-cli-linux -> llmjob-earn-cli-linux",
            f"curl {rel}/pearl_core.node -> pearl_core.node",
            f"curl {rel}/pearl_core_cu13.node -> pearl_core_cu13.node",
            "peak -o stratum+tcp://us.pearl.herominers.com:1200 -u prl1test.bench-peak -p x -c pearl",
            "peak -o stratum+tcp://us.pearl.herominers.com:1200 -u prl1test.bench-peak -p x -c pearl",
            "srbminer --algorithm-gpu pearlhash --pool us.pearl.herominers.com:1200 --wallet prl1test --worker "
            "bench-srb --disable-cpu --api-enable --api-port 21550",
            "cli -a prl1test --mode mining -w bench-ours --no-report --no-serve --no-update"])
        self.assertNotIn("src/cli/earn-cli.js", calls)
        lines = self.bench_lines(out)
        self.assertIn("card NVIDIA GeForce GTX 1660, 550.54, 120.00 W, 120.00 W, 6144 MiB, 7.5", lines[0])
        self.assertIn("ready PeakMiner 2.20.3, SRBMiner 3.8.0, ours v0.6.9", lines[1])
        self.assertEqual([l.split(" ", 2)[2] for l in lines if " start " in l],
                         ["start peak 60s", "start peak 300s", "start srb 300s", "start ours 300s"])
        r = b.parse(lines)
        self.assertTrue(r["done"])
        self.assertEqual(b.figure(r["miners"]["peak"]), (40.0, None))
        self.assertEqual(b.figure(r["miners"]["srb"]), (80.0, None))
        self.assertEqual(b.figure(r["miners"]["ours"], mean_after=60), (30.0, None))

    def test_an_amd_box_builds_and_checks_our_core_then_runs_it_and_srbminer(self):
        out, calls = dry_run(self.SRC, amd=True, env={"ORDER": "ours,srb", "OURS_SECS": "300",
                                                      "BASE_SHA": "0123456789abcdef", "VERIFY_SECS": "600"})
        self.assertNotIn("peakminer", calls)         # PeakMiner isn't downloaded or run
        self.assertFalse([l for l in calls.splitlines() if l.startswith("peak ")])
        self.assertIn("git fetch -q --depth 1 origin 0123456789abcdef", calls)
        self.assertIn("node ../probes/verify-hits.js build/pearl_core_hip.node 600", calls)
        self.assertIn("node src/cli/earn-cli.js --address prl1test --mode mining --pool us.pearl.herominers.com:1200 "
                      "-w bench-ours --no-report --no-serve --no-update | PEARL_CORE_VARIANT=amd PEARL_CORE_PATH="
                      "<T>/llmjob/earn/native/amd/build/pearl_core_hip.node NODE_PATH=<T>/nm/node_modules", calls)
        self.assertIn("srbminer --algorithm-gpu pearlhash --pool us.pearl.herominers.com:1200 --wallet prl1test "
                      "--worker bench-srb --disable-cpu --api-enable --api-port 21550 --disable-gpu-nvidia "
                      "--disable-gpu-intel", calls)
        self.assertIn("pkill -f src/cli/earn-cli\\.js", calls)
        lines = self.bench_lines(out)
        bodies = [l.split(" ", 2)[2] for l in lines]
        self.assertIn("card AMD Radeon RX 9070 XT, ?, , , ? MiB, gfx1201", bodies)
        self.assertIn("copy 564 GB/s", bodies)
        self.assertIn("hitcheck PASS secs=600 hits=2202 verified=400 failed=0 salts=277", bodies)
        self.assertIn("ready SRBMiner 3.8.0, ours HIP core 0123456789ab for gfx1201", bodies)
        self.assertEqual([x for x in bodies if x.startswith("start ")],
                         ["start build gfx1201", "start hitcheck 600s", "start ours 60s", "start ours 300s",
                          "start srb 300s"])
        self.assertTrue(all(len(l) < 400 for l in lines))
        r = self.parse_amd(lines)
        self.assertEqual(r["hitcheck"]["salts"], 277)
        self.assertEqual(TestBenchAMD.a.figure(r["miners"]["ours"], mean_after=60), (93.8, None))
        self.assertEqual(TestBenchAMD.a.figure(r["miners"]["srb"]), (80.0, None))

    def parse_amd(self, lines):
        if not hasattr(TestBenchAMD, "a"):
            TestBenchAMD.setUpClass()
        return TestBenchAMD.a.parse(lines)

    def test_an_amd_box_whose_hit_check_fails_runs_srbminer_only(self):
        bad = dict(VERIFY_PASS, failed=1, PASS=False, fails=[{"why": "seed chain mismatch"}])
        out, calls = dry_run(self.SRC, amd=True, verify=bad, env={"ORDER": "ours,srb", "BASE_SHA": "abc"})
        self.assertNotIn("earn-cli.js --address", calls)
        bodies = [l.split(" ", 2)[2] for l in self.bench_lines(out)]
        self.assertIn("hitcheck FAIL secs=600 hits=2202 verified=400 failed=1 salts=277 why=seed chain mismatch", bodies)
        self.assertIn("skip ours: hit check failed", bodies)
        self.assertIn("ready SRBMiner 3.8.0", bodies)
        self.assertEqual([x for x in bodies if x.startswith("start srb")], ["start srb 60s", "start srb 300s"])
        self.assertEqual(bodies[-1], "all done")
        r = self.parse_amd(self.bench_lines(out))
        self.assertIsNone(r["fail"])
        self.assertEqual(r["skipped"], {"ours": "hit check failed"})

    def test_an_amd_box_whose_core_does_not_build_runs_srbminer_only(self):
        out, calls = dry_run(self.SRC, amd=True, env={"ORDER": "srb,ours", "BASE_SHA": "abc", "FAKE_BUILD_RC": "1"})
        self.assertNotIn("verify-hits", calls)
        bodies = [l.split(" ", 2)[2] for l in self.bench_lines(out)]
        self.assertIn("skip ours: build failed", bodies)
        self.assertEqual([x for x in bodies if x.startswith("start ")],
                         ["start build gfx1201", "start srb 60s", "start srb 300s"])

    def test_an_amd_box_with_no_base_sha_fails(self):
        out, _ = dry_run(self.SRC, amd=True, env={"ORDER": "ours,srb"})
        bodies = [l.split(" ", 2)[2] for l in self.bench_lines(out)]
        self.assertEqual(bodies[-2:], ["FAIL no BASE_SHA", "all done"])


if __name__ == "__main__":
    unittest.main()
