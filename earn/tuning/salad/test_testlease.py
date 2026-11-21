# Tests for testlease.py: leases, names, heartbeats, approvals, verified deletes and the reaper, against a fake
# Salad API with no network. Run: python -m unittest discover -s earn/tuning/salad
import contextlib, importlib.util, io, json, os, shutil, tempfile, unittest

os.environ["SALAD_MINE_DIR"] = tempfile.mkdtemp()
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("testlease", os.path.join(HERE, "testlease.py"))
tl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tl)

T0 = tl.parse_iso("2026-11-21T12:00:00Z")
MIN = 60


class FakeSalad:
    """llmjob2's containers in memory. groups: name -> status. Scripted answers: delete_codes and get_codes are
    lists used up front to back (then normal behaviour)."""

    def __init__(self, groups=None, logs=None, list_ok=True, org="llmjob2"):
        self.org = org
        self.groups = dict(groups or {})
        self.logs = dict(logs or {})     # name -> True (job line lately), False, None (log read fails)
        self.list_ok = list_ok
        self.delete_codes, self.get_codes, self.calls = [], [], []
        self.gone_after_delete = 0       # GETs that still answer 200 after a delete

    def list_groups(self):
        self.calls.append(("list",))
        if not self.list_ok:
            return False, []
        return True, [{"name": n, "current_state": {"status": s}} for n, s in self.groups.items()]

    def get_status(self, name):
        self.calls.append(("get", name))
        if self.get_codes:
            return self.get_codes.pop(0)
        if name in self.groups:
            return 200
        return 404

    def stop(self, name):
        self.calls.append(("stop", name))
        self.groups[name] = "stopped"
        return 202

    def delete(self, name):
        self.calls.append(("delete", name))
        if self.delete_codes:
            c = self.delete_codes.pop(0)
            if c >= 300:
                return c
        self.groups.pop(name, None)
        return 202

    def has_job_line(self, name, start, end):
        self.calls.append(("logs", name))
        return self.logs.get(name, False)

    def acted(self):
        return [c for c in self.calls if c[0] in ("stop", "delete")]


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        os.environ["SALAD_LOCKS_DIR"] = self.dir

    def tearDown(self):
        del os.environ["SALAD_LOCKS_DIR"]
        shutil.rmtree(self.dir, ignore_errors=True)

    def lease_with(self, slot, agent, groups, hb_age):
        tl.take(agent, slot, now=T0 - hb_age)
        for g in groups:
            tl.set_state(slot, g, "live", now=T0 - hb_age)

    def reap(self, api, live=True, now=T0):
        c = Clock()
        return tl.reap(api, live=live, now=now, sleep=c.sleep, clock=c)


class TestLease(Base):
    def test_take_writes_owner_and_lease(self):
        self.assertEqual(tl.take("q4nv30", now=T0), 5)
        lease = tl.read_lease(5)
        self.assertEqual(lease, {"agent": "q4nv30", "slot": 5, "groups": [], "heartbeat_utc": "2026-11-21T12:00:00Z"})
        with open(os.path.join(self.dir, "salad-5", "owner"), encoding="utf-8") as f:
            self.assertIn("q4nv30", f.read())
        self.assertEqual(tl.take("q4amd", now=T0), 6)   # the next free slot

    def test_take_refuses_an_existing_lock_even_stale_and_a_second_lock(self):
        tl.take("old", 7, now=T0 - 10 * 3600)            # stale for hours
        with self.assertRaises(SystemExit):
            tl.take("q4nv30", 7, now=T0)
        os.mkdir(os.path.join(self.dir, "salad-8"))      # an old-style lock with no lease
        with self.assertRaises(SystemExit):
            tl.take("q4nv30", 8, now=T0)
        tl.take("q4nv30", 5, now=T0)
        with self.assertRaises(SystemExit):
            tl.take("q4nv30", 6, now=T0)                 # one lock per agent
        for bad in ("Q4", "q4-nv", "", "a" * 17):
            with self.assertRaises(SystemExit):
                tl.take(bad, now=T0)

    def test_no_free_lock(self):
        for i, s in enumerate(tl.SLOTS):
            tl.take(f"a{i}", s, now=T0)
        with self.assertRaises(SystemExit):
            tl.take("late", now=T0)

    def test_names(self):
        self.assertEqual(tl.parse_name("t5-q4nv30-3090-1"), (5, "q4nv30", "3090", 1))
        self.assertEqual(tl.parse_name("t9-amd-7900xtx-42"), (9, "amd", "7900xtx", 42))
        for bad in ("t4-a-3090-1", "t10-a-3090-1", "t5-a-3090-0", "t5-a-3090", "t5-a-b-3090-1", "biz-rtx3090-1",
                    "nvtest-rtx3090", "t5-A-3090-1", "t5-a-3090-100", ""):
            self.assertIsNone(tl.parse_name(bad), bad)

    def test_require_owner_ties_the_name_to_the_lock(self):
        with self.assertRaises(SystemExit):
            tl.require_owner("t5-q4nv30-3090-1")         # no lease
        tl.take("q4nv30", 5, now=T0)
        self.assertEqual(tl.require_owner("t5-q4nv30-3090-1"), (5, "q4nv30"))
        for bad in ("t5-other-3090-1", "t6-q4nv30-3090-1", "nvtest-rtx3090"):
            with self.assertRaises(SystemExit):
                tl.require_owner(bad)

    def test_group_states_and_heartbeat(self):
        tl.take("q4nv30", 5, now=T0)
        tl.set_state(5, "t5-q4nv30-3090-1", "creating", now=T0 + 1)
        self.assertEqual(tl.read_lease(5)["groups"], [{"name": "t5-q4nv30-3090-1", "state": "creating"}])
        tl.set_state(5, "t5-q4nv30-3090-1", "live", now=T0 + 2)
        self.assertEqual(tl.read_lease(5)["groups"], [{"name": "t5-q4nv30-3090-1", "state": "live"}])
        self.assertTrue(tl.heartbeat(5, now=T0 + 300))
        self.assertEqual(tl.read_lease(5)["heartbeat_utc"], "2026-11-21T12:05:00Z")
        self.assertEqual(tl.heartbeat_agent("q4nv30", now=T0 + 600), [5])
        self.assertEqual(tl.heartbeat_agent("nobody", now=T0), [])
        self.assertFalse(tl.heartbeat(9, now=T0))
        self.assertEqual(tl.remove_group(5, "t5-q4nv30-3090-1"), 0)

    def test_release_only_when_empty_and_own(self):
        self.lease_with(5, "q4nv30", ["t5-q4nv30-3090-1"], 0)
        with self.assertRaises(SystemExit):
            tl.release("q4nv30", 5)
        with self.assertRaises(SystemExit):
            tl.release("other", 5)
        tl.remove_group(5, "t5-q4nv30-3090-1")
        tl.release("q4nv30", 5)
        self.assertFalse(os.path.exists(os.path.join(self.dir, "salad-5")))

    def test_stale_mutex_is_broken(self):
        tl.take("q4nv30", 5, now=T0)
        m = os.path.join(self.dir, "salad-5", "lease.mutex")
        os.mkdir(m)
        os.utime(m, (T0 - 3600, T0 - 3600))
        self.assertTrue(tl.heartbeat(5, now=T0))


class TestVerifiedDelete(Base):
    def test_retries_429_and_5xx_then_waits_for_404(self):
        api = FakeSalad({"t5-a-3090-1": "running"})
        api.delete_codes = [429, 503]
        api.get_codes = [200, 500, 429]                  # still there, then errors, then the real answer (404)
        c = Clock()
        ok, detail = tl.verified_delete(api, "t5-a-3090-1", c.sleep, c)
        self.assertTrue(ok, detail)
        self.assertEqual([x for x in api.calls if x[0] == "delete"], [("delete", "t5-a-3090-1")] * 3)
        self.assertEqual(sum(x[0] == "get" for x in api.calls), 4)

    def test_other_4xx_fails_at_once(self):
        api = FakeSalad({"t5-a-3090-1": "running"})
        api.delete_codes = [403]
        c = Clock()
        ok, detail = tl.verified_delete(api, "t5-a-3090-1", c.sleep, c)
        self.assertFalse(ok)
        self.assertIn("403", detail)

    def test_never_404_times_out(self):
        api = FakeSalad({"t5-a-3090-1": "running"})
        api.get_codes = [200] * 1000
        c = Clock()
        ok, detail = tl.verified_delete(api, "t5-a-3090-1", c.sleep, c)
        self.assertFalse(ok)
        self.assertIn("still answers HTTP 200", detail)

    def test_delete_leased_takes_it_out_of_the_lease_but_keeps_the_lock(self):
        self.lease_with(5, "q4nv30", ["t5-q4nv30-3090-1", "t5-q4nv30-3090-2"], 0)
        api = FakeSalad({"t5-q4nv30-3090-1": "running", "t5-q4nv30-3090-2": "running"})
        c = Clock()
        ok, detail = tl.delete_leased(api, "t5-q4nv30-3090-1", c.sleep, c)
        self.assertTrue(ok, detail)
        self.assertEqual([g["name"] for g in tl.read_lease(5)["groups"]], ["t5-q4nv30-3090-2"])

    def test_failed_delete_stays_in_the_lease_as_deleting(self):
        self.lease_with(5, "q4nv30", ["t5-q4nv30-3090-1"], 0)
        api = FakeSalad({"t5-q4nv30-3090-1": "running"})
        api.get_codes = [200] * 1000
        c = Clock()
        ok, _ = tl.delete_leased(api, "t5-q4nv30-3090-1", c.sleep, c)
        self.assertFalse(ok)
        self.assertEqual(tl.read_lease(5)["groups"], [{"name": "t5-q4nv30-3090-1", "state": "deleting"}])

    def test_delete_leased_refuses_another_agents_box(self):
        self.lease_with(5, "q4nv30", ["t5-q4nv30-3090-1"], 0)
        with self.assertRaises(SystemExit):
            tl.delete_leased(FakeSalad(), "t5-thief-3090-1")


class TestReaper(Base):
    G = "t5-q4nv30-3090-1"

    def log_lines(self):
        with open(os.path.join(self.dir, "reaper.log"), encoding="utf-8") as f:
            return [json.loads(x) for x in f]

    def test_listing_failure_does_nothing(self):
        self.lease_with(5, "q4nv30", [self.G], 3600)
        api = FakeSalad({self.G: "running"}, list_ok=False)
        recs = self.reap(api)
        self.assertEqual([r["action"] for r in recs], ["abort"])
        self.assertEqual(api.calls, [("list",)])
        self.assertIsNotNone(tl.read_lease(5))
        self.assertEqual(self.log_lines()[0]["action"], "abort")

    def test_never_another_org(self):
        api = FakeSalad({self.G: "running"}, org="llmjob")
        self.assertEqual([r["action"] for r in self.reap(api)], ["abort"])
        self.assertEqual(api.calls, [])

    def test_fresh_heartbeat_keeps_the_box(self):
        self.lease_with(5, "q4nv30", [self.G], 9 * MIN)
        api = FakeSalad({self.G: "running"})
        recs = self.reap(api)
        self.assertEqual([r["action"] for r in recs], ["keep"])
        self.assertEqual(api.acted(), [])

    def test_stale_heartbeat_but_job_lines_keep_the_box(self):
        self.lease_with(5, "q4nv30", [self.G], 60 * MIN)
        api = FakeSalad({self.G: "running"}, logs={self.G: True})
        self.assertEqual([r["action"] for r in self.reap(api)], ["keep"])
        api.logs[self.G] = None                         # logs unreadable: not reaped either
        self.assertEqual([r["action"] for r in self.reap(api)], ["keep"])
        self.assertEqual(api.acted(), [])

    def test_two_passes_stop_then_delete_then_free_the_lock(self):
        self.lease_with(5, "q4nv30", [self.G], 11 * MIN)
        api = FakeSalad({self.G: "running"})
        recs = self.reap(api)
        self.assertEqual([r["action"] for r in recs], ["stop"])
        self.assertEqual(api.acted(), [("stop", self.G)])
        self.assertIn("reaper_stopped_utc", tl.read_lease(5)["groups"][0])
        recs = self.reap(api, now=T0 + 15 * MIN)
        self.assertEqual([r["action"] for r in recs], ["delete", "free_lock"])
        self.assertEqual(api.acted(), [("stop", self.G), ("delete", self.G)])
        self.assertNotIn(self.G, api.groups)
        self.assertFalse(os.path.exists(os.path.join(self.dir, "salad-5")))
        self.assertEqual([r["action"] for r in self.log_lines()], ["stop", "delete", "free_lock"])

    def test_agent_back_between_passes_clears_the_stop(self):
        self.lease_with(5, "q4nv30", [self.G], 11 * MIN)
        api = FakeSalad({self.G: "running"})
        self.reap(api)
        tl.heartbeat(5, now=T0 + 14 * MIN)
        self.assertEqual([r["action"] for r in self.reap(api, now=T0 + 15 * MIN)], ["keep"])
        self.assertNotIn("reaper_stopped_utc", tl.read_lease(5)["groups"][0])
        self.assertEqual(api.acted(), [("stop", self.G)])

    def test_failed_verified_delete_keeps_the_lock(self):
        self.lease_with(5, "q4nv30", [self.G], 11 * MIN)
        api = FakeSalad({self.G: "running"})
        self.reap(api)
        api.delete_codes = [500] * 1000
        recs = self.reap(api, now=T0 + 15 * MIN)
        self.assertEqual([r["action"] for r in recs], ["delete_failed"])
        self.assertTrue(os.path.isdir(os.path.join(self.dir, "salad-5")))
        self.assertEqual(tl.read_lease(5)["groups"][0]["state"], "deleting")

    def test_dry_run_changes_nothing(self):
        self.lease_with(5, "q4nv30", [self.G], 11 * MIN)
        self.lease_with(6, "q4amd", [], 60 * MIN)        # stale and empty
        before = {s: tl.read_lease(s) for s in (5, 6)}
        api = FakeSalad({self.G: "running"})
        for t in (T0, T0 + 15 * MIN):
            recs = self.reap(api, live=False, now=t)
            self.assertEqual([r["action"] for r in recs], ["stop", "free_lock"])
            self.assertTrue(all(r["dry_run"] for r in recs))
        self.assertEqual(api.acted(), [])
        self.assertEqual({s: tl.read_lease(s) for s in (5, 6)}, before)
        self.assertIn("would stop t5-q4nv30-3090-1", tl.summary(recs))
        self.assertIn("dry run", tl.summary(recs))

    def test_dry_run_is_the_default_on_day_one(self):
        self.assertFalse(tl.REAP_LIVE_DEFAULT)
        self.lease_with(5, "q4nv30", [self.G], 11 * MIN)
        api = FakeSalad({self.G: "running"})
        recs = tl.reap(api, now=T0)
        self.assertTrue(recs[0]["dry_run"])
        self.assertEqual(api.acted(), [])

    def test_other_names_and_unleased_groups_are_only_reported(self):
        self.lease_with(5, "q4nv30", [self.G], 60 * MIN)
        api = FakeSalad({"biz-rtx3090-1": "running", "amdtest-rdna3": "running", "t6-ghost-3090-1": "running",
                         "t5-other-3090-1": "running", "t5-q4nv30-3090-9": "running"})
        recs = self.reap(api)
        rep = {r["group"] for r in recs if r["action"] == "report"}
        self.assertEqual(rep, {"biz-rtx3090-1", "amdtest-rdna3", "t6-ghost-3090-1", "t5-other-3090-1",
                               "t5-q4nv30-3090-9"})
        self.assertEqual(api.acted(), [])
        # The lease's own group is gone from Salad, so the stale lease is emptied and freed.
        self.assertIn("forget", [r["action"] for r in recs])
        self.assertFalse(os.path.exists(os.path.join(self.dir, "salad-5")))

    def test_stale_lease_with_a_creating_group_salad_never_made(self):
        tl.take("q4nv30", 5, now=T0 - 30 * MIN)
        tl.set_state(5, self.G, "creating", now=T0 - 30 * MIN)
        api = FakeSalad({})
        acts = [r["action"] for r in self.reap(api)]
        self.assertEqual(acts, ["forget", "free_lock"])

    def test_fresh_empty_lease_and_old_style_lock_are_left(self):
        tl.take("q4nv30", 5, now=T0 - 2 * MIN)
        os.mkdir(os.path.join(self.dir, "salad-7"))
        recs = self.reap(FakeSalad({}))
        self.assertTrue(os.path.isdir(os.path.join(self.dir, "salad-5")))
        self.assertTrue(os.path.isdir(os.path.join(self.dir, "salad-7")))
        self.assertEqual([(r["action"], r.get("slot")) for r in recs], [("report", 7)])

    def test_stop_on_an_already_stopped_group_counts(self):
        self.lease_with(5, "q4nv30", [self.G], 11 * MIN)
        api = FakeSalad({self.G: "stopped"})
        api.stop = lambda name: api.calls.append(("stop", name)) or 409
        self.assertEqual([r["action"] for r in self.reap(api)], ["stop"])
        self.assertEqual([r["action"] for r in self.reap(api, now=T0 + 15 * MIN)], ["delete", "free_lock"])

    def test_approvals_and_requests_are_reported(self):
        tl.request_extension(self.G, 6, "long race", now=T0)
        tl.approve(self.G, "+6h", now=T0)
        tl.request_extension("t6-x-3090-1", 2, "more", now=T0)
        recs = self.reap(FakeSalad({}))
        got = {(r["action"], r["group"]) for r in recs}
        self.assertEqual(got, {("approval", self.G), ("request", self.G), ("request", "t6-x-3090-1")})
        self.assertTrue(next(r for r in recs if r["action"] == "approval")["valid"])
        s = tl.summary(recs)
        self.assertIn("approval t5-q4nv30-3090-1", s)
        self.assertIn("request t6-x-3090-1", s)


class TestApprovals(Base):
    G = "t5-q4nv30-3090-1"

    def test_no_approval_means_three_hours(self):
        self.assertEqual(tl.cap_secs(self.G, now=T0), 3 * 3600)

    def test_a_request_alone_changes_nothing(self):
        p = tl.request_extension(self.G, 8, "overnight soak", now=T0)
        self.assertTrue(p.endswith(os.path.join("requests", self.G + ".json")))
        self.assertEqual(tl.cap_secs(self.G, now=T0), 3 * 3600)
        with self.assertRaises(SystemExit):
            tl.request_extension(self.G, 8, "", now=T0)
        with self.assertRaises(SystemExit):
            tl.request_extension("biz-x", 8, "why", now=T0)

    def test_approval_raises_the_cap_until_it_ends(self):
        tl.request_extension(self.G, 8, "overnight soak", now=T0)
        a = tl.approve(self.G, "+8h", now=T0)
        self.assertEqual(a, {"group": self.G, "until_utc": "2026-11-21T20:00:00Z", "reason": "overnight soak",
                             "approved_by": "coordinator"})
        self.assertEqual(tl.cap_secs(self.G, now=T0), 8 * 3600)
        self.assertEqual(tl.cap_secs(self.G, now=T0 + 7 * 3600), 3 * 3600)   # never below the normal cap
        self.assertEqual(tl.cap_secs(self.G, now=T0 + 9 * 3600), 3 * 3600)   # expired
        tl.approve(self.G, "2026-11-22T00:00Z", "x", now=T0)
        self.assertEqual(tl.cap_secs(self.G, now=T0), 12 * 3600)
        with self.assertRaises(SystemExit):
            tl.approve(self.G, "2026-11-20T00:00Z", now=T0)

    def test_an_approval_not_from_the_coordinator_is_ignored(self):
        tl.write_json(tl.approval_path(self.G), {"group": self.G, "until_utc": "2026-11-22T00:00:00Z",
                                                  "reason": "self-approved", "approved_by": "q4nv30"})
        self.assertEqual(tl.cap_secs(self.G, now=T0), 3 * 3600)
        tl.write_json(tl.approval_path(self.G), {"group": "t5-other-1-1", "until_utc": "2026-11-22T00:00:00Z",
                                                  "approved_by": "coordinator"})
        self.assertEqual(tl.cap_secs(self.G, now=T0), 3 * 3600)

    def test_cli_extend_says_to_ask_the_coordinator(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            tl.main(["extend", "--group", self.G, "--hours", "5", "--reason", "soak"])
        self.assertIn("message the coordinator", out.getvalue())
        self.assertFalse(os.path.exists(tl.approval_path(self.G)))


class TestCli(Base):
    def test_take_show_reap(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            tl.main(["take", "--agent", "q4nv30"])
            tl.main(["show"])
            tl.main(["reap"], http=lambda *a, **k: (200, {"items": []}))
        s = out.getvalue()
        self.assertIn("took salad-5", s)
        self.assertIn("salad-5: q4nv30", s)
        self.assertIn("reaper (dry run)", s)

    def test_reap_with_a_failing_listing(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            tl.main(["reap", "--live"], http=lambda *a, **k: (0, {"error": "timeout"}))
        self.assertIn("abort", out.getvalue())

    def test_salad_client_reads_job_lines(self):
        calls = []

        def http(m, url, body=None, **k):
            calls.append((m, url))
            if url.endswith("/log-entries"):
                return 200, {"items": [{"text_log": "x [nvtest] 12:00:00 boot job running"}]}
            return 200, {"items": [{"name": "t5-a-3090-1"}]}
        api = tl.Salad(http)
        self.assertTrue(api.has_job_line("t5-a-3090-1", T0 - 600, T0))
        self.assertTrue(api.list_groups()[0])
        self.assertTrue(all("/llmjob2/" in u for _, u in calls))
        self.assertFalse(tl.Salad(lambda *a, **k: (200, {"items": [{"text_log": "other"}]})).has_job_line("g", 0, 1))
        self.assertIsNone(tl.Salad(lambda *a, **k: (408, None)).has_job_line("g", 0, 1))
        self.assertFalse(tl.Salad(lambda *a, **k: (200, {"oops": 1})).list_groups()[0])


class TestAmdtestUsesLeases(Base):
    """amdtest.py's side: names tied to locks, the cap from approvals, verified deletes, heartbeats."""

    @classmethod
    def setUpClass(cls):
        sp = importlib.util.spec_from_file_location("amdtest", os.path.join(HERE, "amdtest.py"))
        cls.a = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(cls.a)

    def setUp(self):
        super().setUp()
        a = self.a
        self.calls = []
        self.salad = {}

        def http(m, url, body=None, **k):
            self.calls.append((m, url, body))
            name = url.rstrip("/").split("/")[-1]
            if url.endswith("/gpu-classes"):
                return 200, {"items": [{"name": "RTX 3090 (24 GB)", "id": "c3090"},
                                       {"name": "AMD RX 9070 XT (16GB)", "id": "c9070"}]}
            if url.endswith("/log-entries"):
                return 200, {"items": []}
            if m == "POST" and url == a.w.CB:
                self.salad[body["name"]] = body
                return 201, {}
            if m == "DELETE":
                self.salad.pop(name, None)
                return 202, {}
            if m == "GET" and name in self.salad:
                return 200, {"current_state": {"status": "running"},
                             "container": {"command": self.salad[name]["container"]["command"],
                                           "environment_variables": self.salad[name]["container"][
                                               "environment_variables"]}}
            if m == "PATCH":
                return 200, {"container": body["container"]}
            return 404, {}
        self.old = a.w.http, a.time.sleep, a.w.ORG, a.lease.time.sleep
        a.w.http, a.time.sleep, a.w.ORG, a.lease.time.sleep = http, (lambda s: None), "llmjob2", (lambda s: None)

    def tearDown(self):
        a = self.a
        a.w.http, a.time.sleep, a.w.ORG, a.lease.time.sleep = self.old
        super().tearDown()

    def run_cmd(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.a.main(list(argv))
        return out.getvalue()

    def test_create_refuses_without_the_agents_lease(self):
        for argv in (["create", "nvtest-rtx3090", "3090", "--sha", "x"],     # old names can't be created
                     ["create", "t5-q4nv30-3090-1", "--sha", "x"]):           # no lease
            with self.assertRaises(SystemExit):
                self.run_cmd(*argv)
        self.a.lease.take("other", 5)
        with self.assertRaises(SystemExit):
            self.run_cmd("create", "t5-q4nv30-3090-1", "--sha", "x")         # someone else's lock
        with self.assertRaises(SystemExit):
            self.run_cmd("create", "t5-other-gtx9-1", "--sha", "x")          # not a known card
        self.assertEqual([c for c in self.calls if c[0] == "POST"], [])

    def test_create_only_in_llmjob2(self):
        self.a.lease.take("q4nv30", 5)
        self.a.w.ORG = "llmjob"
        with self.assertRaises(SystemExit):
            self.run_cmd("create", "t5-q4nv30-3090-1", "--sha", "x")
        self.assertEqual(self.calls, [])

    def test_create_writes_the_lease_and_the_cap(self):
        self.a.lease.take("q4amd", 7)
        self.run_cmd("create", "t7-q4amd-9070xt-1", "--sha", "f" * 40)
        body = self.salad["t7-q4amd-9070xt-1"]
        self.assertEqual(body["container"]["resources"]["gpu_classes"], ["c9070"])
        self.assertEqual(body["container"]["image"], self.a.IMAGE)
        self.assertEqual(body["restart_policy"], "never")
        self.assertEqual(body["container"]["environment_variables"]["BOX_MAX_SECS"], "10800")
        self.assertEqual(self.a.lease.read_lease(7)["groups"], [{"name": "t7-q4amd-9070xt-1", "state": "live"}])

    def test_cap_cannot_be_set_by_hand_but_an_approval_raises_it(self):
        self.a.lease.take("q4nv30", 5)
        for k in self.a.BOX_VARS:
            with self.assertRaises(SystemExit):
                self.run_cmd("create", "t5-q4nv30-3090-1", "--sha", "x", "--env", f"{k}=99999")
        self.a.lease.approve("t5-q4nv30-3090-1", "+6h")
        self.run_cmd("create", "t5-q4nv30-3090-1", "--sha", "x")
        cap = int(self.salad["t5-q4nv30-3090-1"]["container"]["environment_variables"]["BOX_MAX_SECS"])
        self.assertAlmostEqual(cap, 6 * 3600, delta=5)
        # set-env reads the cap again, and refuses it by hand
        with self.assertRaises(SystemExit):
            self.run_cmd("set-env", "t5-q4nv30-3090-1", "BOX_MAX_SECS=1", "--no-restart")
        os.remove(self.a.lease.approval_path("t5-q4nv30-3090-1"))
        self.run_cmd("set-env", "t5-q4nv30-3090-1", "TASKS=card", "--no-restart")
        patch = [c[2] for c in self.calls if c[0] == "PATCH"][-1]
        self.assertEqual(patch["container"]["environment_variables"]["BOX_MAX_SECS"], "10800")

    def test_failed_create_forgets_a_group_salad_never_made(self):
        self.a.lease.take("q4nv30", 5)
        old = self.a.w.http
        self.a.w.http = lambda m, url, body=None, **k: (
            (200, {"items": [{"name": "RTX 3090 (24 GB)", "id": "c3090"}]}) if url.endswith("/gpu-classes")
            else (400, {"error": "bad"}) if m == "POST" else (404, {}))
        try:
            with self.assertRaises(SystemExit):
                self.run_cmd("create", "t5-q4nv30-3090-1", "--sha", "x")
        finally:
            self.a.w.http = old
        self.assertEqual(self.a.lease.read_lease(5)["groups"], [])

    def test_commands_refresh_the_heartbeat(self):
        self.a.lease.take("q4nv30", 5, now=T0)
        self.run_cmd("status", "t5-q4nv30-3090-1")
        self.assertNotEqual(self.a.lease.read_lease(5)["heartbeat_utc"], "2026-11-21T12:00:00Z")
        tl_old = self.a.lease.read_lease(5)
        self.a.lease.write_json(self.a.lease.lease_path(5), dict(tl_old, heartbeat_utc="2026-11-21T12:00:00Z"))
        with self.assertRaises(SystemExit):
            self.run_cmd("stop", "t5-thief-3090-1")      # mutating someone else's box is refused
        self.run_cmd("logs", "t5-q4nv30-3090-1")
        self.assertNotEqual(self.a.lease.read_lease(5)["heartbeat_utc"], "2026-11-21T12:00:00Z")

    def test_delete_is_verified_and_leaves_the_lease(self):
        self.a.lease.take("q4nv30", 5)
        self.run_cmd("create", "t5-q4nv30-3090-1", "--sha", "x")
        out = self.run_cmd("delete", "t5-q4nv30-3090-1")
        self.assertIn("gone (404)", out)
        self.assertEqual(self.a.lease.read_lease(5)["groups"], [])
        self.assertTrue(os.path.isdir(os.path.join(self.dir, "salad-5")))   # the lock stays until released


if __name__ == "__main__":
    unittest.main()
