# Tests for guard.py, with Salad, the pool and the clock faked.
# Run: python3 -m unittest discover -s tuning/salad
import importlib.util, os, re, unittest

spec = importlib.util.spec_from_file_location("guard", os.path.join(os.path.dirname(os.path.abspath(__file__)), "guard.py"))
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
W = guard.W

T0 = 1_800_000_000.0
WALLET = "prl1" + "q" * 59


class Fake:
    def __init__(self, routes):
        self.calls, self.routes = [], routes

    def __call__(self, method, url, body=None, ctype="application/json", timeout=60):
        self.calls.append((method, url))
        for m, frag, resp in self.routes:
            if m == method and re.search(frag, url):
                return resp
        return 404, {}


def group(name, age_min, status="running", kx=False):
    cmd = ["/bin/sh", "-c", "... -e 's|\"flags\":\"\"}|\"flags\":\"--pool prl-us.kryptex.network:7048\"}|' ..." if kx else "..."]
    return {"name": name, "create_time": W.iso(T0 - age_min * 60), "current_state": {"status": status},
            "container": {"command": cmd}}


class TestGuard(unittest.TestCase):
    def run_guard(self, groups, workers, instances=None, dry=False, kx=None, kx_status=200, kx_body=None):
        inst = instances if instances is not None else [{"state": "running", "machine_id": "m1"}]
        self.fake = Fake([("GET", r"/containers$", (200, {"items": groups})),
                          ("GET", r"/instances$", (200, {"instances": inst})),
                          ("GET", "herominers", (200, {"workers": workers})),
                          ("GET", "kryptex", (kx_status, kx_body if kx_body is not None else {"results": kx or []})),
                          ("POST", r"/stop$", (202, None))])
        W.http, W.now = self.fake, lambda: T0
        return guard.check(WALLET, dry=dry)

    def stops(self):
        return [u for m, u in self.fake.calls if m == "POST"]

    def test_a_group_on_a_pc_with_no_pool_work_for_an_hour_is_stopped(self):
        out, stopped = self.run_guard([group("biz-rtx4080", 90)], [{"name": "biz-rtx4080-g0", "hashrate": 0, "hashrate_1h": 0}])
        self.assertEqual(stopped, ["biz-rtx4080"])
        self.assertTrue(self.stops()[0].endswith("/containers/biz-rtx4080/stop"))
        self.assertIn("STOPPED biz-rtx4080", out[0])

    def test_a_group_the_pool_has_never_seen_is_stopped(self):
        _, stopped = self.run_guard([group("biz-rtx4080", 90)], [])
        self.assertEqual(stopped, ["biz-rtx4080"])

    def test_a_mining_group_is_left_alone(self):
        _, stopped = self.run_guard([group("biz-rtx4080", 300)], [{"name": "biz-rtx4080-g0", "hashrate": 0, "hashrate_1h": 44000}])
        self.assertEqual(stopped, [])

    def test_a_group_mining_on_kryptex_is_left_alone(self):
        # Kryptex's rows: the worker as "wallet/worker", averages in H/s as strings, last_share in ms.
        row = {"worker": f"{WALLET}/biz-rtx4080-g0", "avg_hashrate_30m": "199000000000000.5", "last_share": 0}
        _, stopped = self.run_guard([group("biz-rtx4080", 300)], [], kx=[row])
        self.assertEqual(stopped, [])
        row = {"worker": f"{WALLET}/biz-rtx4080-g0", "avg_hashrate_30m": "0", "last_share": int(T0 - 600) * 1000}
        _, stopped = self.run_guard([group("biz-rtx4080", 300)], [], kx=[row])
        self.assertEqual(stopped, [])
        row = {"worker": f"{WALLET}/biz-rtx4080-g0", "avg_hashrate_30m": "0", "last_share": int(T0 - 7200) * 1000}
        _, stopped = self.run_guard([group("biz-rtx4080", 300)], [], kx=[row])
        self.assertEqual(stopped, ["biz-rtx4080"])

    def test_when_kryptex_cant_be_read_only_its_boxes_are_left_alone(self):
        groups = [group("biz-rtx4080", 90, kx=True), group("biz-rtx3090", 90)]
        out, stopped = self.run_guard(groups, [], kx_status=503)
        self.assertEqual(stopped, ["biz-rtx3090"])      # a dead HeroMiners box is still stopped
        self.assertIn("Kryptex: workers failed (HTTP 503); boxes on Kryptex left alone", out[0])
        # A 200 without a results list is a failed read too, not "no Kryptex workers".
        _, stopped = self.run_guard(groups, [], kx_body={"error": "busy"})
        self.assertEqual(stopped, ["biz-rtx3090"])

    def test_a_young_group_is_left_alone(self):
        _, stopped = self.run_guard([group("biz-rtx4080", 50)], [])
        self.assertEqual(stopped, [])

    def test_a_group_waiting_for_a_pc_is_left_alone(self):
        _, stopped = self.run_guard([group("biz-rtx4080", 90)], [], instances=[{"state": "allocating"}])
        self.assertEqual(stopped, [])

    def test_other_groups_and_stopped_groups_are_left_alone(self):
        _, stopped = self.run_guard([group("development1-x", 90), group("biz-rtx2060", 90, status="stopped")], [])
        self.assertEqual(stopped, [])
        self.assertFalse(any("/instances" in u for _, u in self.fake.calls))

    def test_a_dry_run_stops_nothing(self):
        out, stopped = self.run_guard([group("biz-rtx4080", 90)], [], dry=True)
        self.assertEqual(stopped, [])
        self.assertEqual(self.stops(), [])
        self.assertIn("WOULD STOP biz-rtx4080", out[0])

    def test_nothing_is_done_when_the_pool_cant_be_read(self):
        self.fake = Fake([("GET", r"/containers$", (200, {"items": [group("biz-rtx4080", 90)]}))])
        W.http, W.now = self.fake, lambda: T0
        out, stopped = guard.check(WALLET)
        self.assertEqual(stopped, [])
        self.assertIn("did nothing", out[0])


if __name__ == "__main__":
    unittest.main()
