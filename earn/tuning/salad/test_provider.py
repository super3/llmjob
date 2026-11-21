# Tests for provider.py, with our server and Salad faked.
# Run: python3 -m unittest discover -s earn/tuning/salad
import importlib.util, os, unittest

spec = importlib.util.spec_from_file_location("provider", os.path.join(os.path.dirname(os.path.abspath(__file__)), "provider.py"))
provider = importlib.util.module_from_spec(spec)
spec.loader.exec_module(provider)

IMAGE = "ghcr.io/super3/llmjob-bench@sha256:" + "a" * 64


class Fake:
    def __init__(self, answers):
        self.answers, self.calls = answers, []

    def __call__(self, method, url, body=None, headers=None, timeout=60, ctype=None):
        self.calls.append((method, url, body, headers))
        for (m, frag), resp in self.answers.items():
            if m == method and frag in url:
                return resp
        return 404, {"error": "no route"}


class NetworkTest(unittest.TestCase):
    def test_create_status_logs_delete(self):
        job = {"id": "cj_1", "status": "running",
               "runs": [{"id": "cr_1", "nodeId": "abcdef0123456789", "card": 0, "status": "running", "minutes": 2.5}]}
        fake = Fake({
            ("POST", "/api/containers"): (201, {"job": {"id": "cj_1"}}),
            ("GET", "/api/containers/cj_1/logs"): (200, {"lines": [
                {"ts": 5, "nodeId": "abcdef0123456789", "card": 0, "line": "[bench] hi"}], "next": 7}),
            ("GET", "/api/containers/cj_1"): (200, {"job": job}),
            ("DELETE", "/api/containers/cj_1"): (200, {"job": job}),
        })
        p = provider.Network(server="http://srv/", token="tok", call=fake)
        h = p.create({"name": "bench-1", "image": IMAGE, "env": {"A": "1"}, "duration_sec": 600,
                      "target": {"gpu": "RTX 4090"}})
        self.assertEqual(h, "cj_1")
        method, url, body, headers = fake.calls[0]
        self.assertEqual(url, "http://srv/api/containers")
        self.assertEqual(headers, {"Authorization": "Bearer tok"})
        self.assertEqual(body, {"name": "bench-1", "image": IMAGE, "env": {"A": "1"}, "durationSec": 600,
                                "replicas": 1, "target": {"gpu": "RTX 4090"}, "payRateUsdHour": 0})
        st = p.status(h)
        self.assertEqual(st["state"], "running")
        self.assertEqual(st["instances"], [{"id": "cr_1", "where": "abcdef0123456789/gpu0", "state": "running", "minutes": 2.5}])
        lines, nxt = p.logs(h, 3)
        self.assertEqual(lines, [{"t": 5, "where": "abcdef0123456789/gpu0", "text": "[bench] hi"}])
        self.assertEqual(nxt, 7)
        self.assertIn("since=3", fake.calls[-1][1])
        self.assertTrue(p.delete(h))

    def test_states_and_errors(self):
        for status, want in (("pending", "pending"), ("done", "done"), ("stopping", "stopped"), ("stopped", "stopped")):
            fake = Fake({("GET", "/api/containers/x"): (200, {"job": {"status": status}})})
            self.assertEqual(provider.Network(server="http://s", token="t", call=fake).status("x")["state"], want)
        p = provider.Network(server="http://s", token="t", call=Fake({("POST", "/api/containers"): (400, {"error": "bad image"})}))
        with self.assertRaises(provider.ProviderError) as e:
            p.create({"name": "n", "image": "x"})
        self.assertIn("HTTP 400", str(e.exception))
        self.assertFalse(provider.Network(server="http://s", token="t", call=Fake({("DELETE", "/x"): (500, {})})).delete("x"))


class FakeW:
    CB = "https://salad/c"
    BASE = "https://salad/org"

    def __init__(self, answers):
        self.http = Fake(answers)

    @staticmethod
    def iso(t):
        return f"iso{int(t)}"

    @staticmethod
    def parse_iso(s):
        return 42.0


class SaladTest(unittest.TestCase):
    def test_create_status_logs_delete(self):
        w = FakeW({
            ("POST", "/c"): (200, {}),
            ("GET", "/c/g1/instances"): (200, {"instances": [{"instance_id": "i1", "machine_id": "m1", "state": "running"}]}),
            ("GET", "/c/g1"): (200, {"current_state": {"status": "running"}}),
            ("POST", "/log-entries"): (200, {"items": [
                {"time": "b", "text_log": "two", "resource": {"labels": {"machine_id": "m1"}}},
                {"time": "a", "text_log": "one"}]}),
            ("DELETE", "/c/g1"): (202, None),
        })
        p = provider.Salad(w=w)
        self.assertEqual(p.create({"name": "g1", "image": "img", "gpu_classes": ["c1"]}), "g1")
        body = w.http.calls[0][2]
        self.assertEqual(body["container"]["resources"]["gpu_classes"], ["c1"])
        self.assertEqual(body["restart_policy"], "never")
        st = p.status("g1")
        self.assertEqual(st["state"], "running")
        self.assertEqual(st["instances"], [{"id": "i1", "where": "m1", "state": "running"}])
        lines, nxt = p.logs("g1", 10)
        self.assertEqual([x["text"] for x in lines], ["one", "two"])
        self.assertEqual(lines[0]["where"], None)
        self.assertEqual(nxt, 42.0)
        self.assertTrue(p.delete("g1"))

    def test_quiet_logs_and_stopped_group(self):
        w = FakeW({("POST", "/log-entries"): (200, {"items": []}),
                   ("GET", "/c/g2/instances"): (500, {}),
                   ("GET", "/c/g2"): (200, {"current_state": {"status": "stopped"}})})
        p = provider.Salad(w=w)
        self.assertEqual(p.logs("g2", 9), ([], 9))
        self.assertEqual(p.status("g2"), {"state": "stopped", "instances": [], "raw": {"current_state": {"status": "stopped"}}})


class PickTest(unittest.TestCase):
    def test_pick(self):
        self.assertIsInstance(provider.pick("network", server="http://s", token="t"), provider.Network)
        self.assertIsInstance(provider.pick("salad"), provider.Salad)
        old = os.environ.pop("BENCH_PROVIDER", None)
        try:
            self.assertIsInstance(provider.pick(), provider.Salad)
            os.environ["BENCH_PROVIDER"] = "NETWORK"
            self.assertIsInstance(provider.pick(token="t"), provider.Network)
        finally:
            os.environ.pop("BENCH_PROVIDER", None)
            if old is not None:
                os.environ["BENCH_PROVIDER"] = old
        with self.assertRaises(provider.ProviderError):
            provider.pick("vast")

    def test_http_never_raises(self):
        s, j = provider.http("GET", "http://127.0.0.1:9/nothing", timeout=2)
        self.assertEqual(s, 0)
        self.assertIn("error", j)


if __name__ == "__main__":
    unittest.main()
