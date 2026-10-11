# Tests for trader.py with SafeTrade faked. Run: python3 -m unittest discover -s trader
import importlib.util, os, re, unittest

os.environ.update(SAFETRADE_API_KEY="k", SAFETRADE_API_SECRET="s")
spec = importlib.util.spec_from_file_location("trader", os.path.join(os.path.dirname(os.path.abspath(__file__)), "trader.py"))
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
t.time.sleep = lambda s: None
HTTP, LOG = t.http, t.log  # the real ones, before tests swap them

MARKETS = [{"id": "prlusdt", "amount_precision": 2, "price_precision": 4, "min_amount": "0.1"}]
BOOK = {"bids": [{"price": "1.40", "remaining_volume": "3"}, {"price": "1.38", "remaining_volume": "2"},
                 {"price": "1.30", "remaining_volume": "50"}], "asks": []}


class Fake:
    def __init__(self, balance="4.123", orders=(), book=BOOK):
        self.calls = []
        self.routes = [("GET", r"/public/markets$", (200, MARKETS)),
                       ("GET", r"/order-book$", (200, book)),
                       ("GET", r"/market/orders$", (200, list(orders))),
                       ("GET", r"/account/balances$", (200, [{"currency": "usdt", "balance": "9"},
                                                              {"currency": "prl", "balance": balance}])),
                       ("POST", r"/market/orders$", (201, {"id": 77, "state": "wait"})),
                       ("GET", r"/market/orders/77$", (200, {"id": 77, "state": "done", "executed_volume": "4.12",
                                                              "avg_price": "1.392"})),
                       ("POST", r"/cancel$", (200, {}))]

    def __call__(self, method, path, params=None, auth=False):
        self.calls.append((method, path, params, auth))
        for m, frag, resp in self.routes:
            if m == method and re.search(frag, path):
                return resp
        return 404, {}

    def made(self, method, frag):
        return [c for c in self.calls if c[0] == method and re.search(frag, c[1])]


class Base(unittest.TestCase):
    def setUp(self):
        self.logs = []
        t.log = self.logs.append
        t.MAX_SLIPPAGE, t.MIN_PRICE, t.MIN_VOLUME = 0.03, 0.0, 0.5
        os.environ["DRY_RUN"] = "0"
        os.environ["RAILWAY_ENVIRONMENT_NAME"] = "production"

    def go(self, fake):
        self.fake = fake
        t.http = fake
        return t.main()


class TestSell(Base):
    def test_sells_what_the_book_takes_within_slippage(self):
        self.assertEqual(self.go(Fake()), 0)
        order = self.fake.made("POST", r"/market/orders$")[0]
        # 3% under the best bid of 1.40 is 1.358: the bids at 1.40 and 1.38 (5 PRL) qualify; we have 4.123
        self.assertEqual(order[2], {"market": "prlusdt", "side": "sell", "volume": "4.12", "ord_type": "limit",
                                    "price": "1.358"})
        self.assertTrue(order[3])
        self.assertIn("SOLD 4.12 of 4.12 PRL at about 1.392", self.logs[-1])

    def test_sells_only_what_the_bids_above_the_floor_take(self):
        self.go(Fake(balance="20"))
        self.assertEqual(self.fake.made("POST", r"/market/orders$")[0][2]["volume"], "5.0")

    def test_never_under_min_price(self):
        t.MIN_PRICE = 1.45
        self.go(Fake())
        self.assertFalse(self.fake.made("POST", r"/market/orders$"))
        self.assertIn("under the floor", self.logs[-1])

    def test_dust_is_left(self):
        self.go(Fake(balance="0.3"))
        self.assertFalse(self.fake.made("POST", r"/market/orders$"))
        self.assertIn("nothing to sell", self.logs[-1])

    def test_last_runs_open_sell_is_cancelled_first(self):
        self.go(Fake(orders=[{"id": 5, "side": "sell", "remaining_volume": "1", "price": "1.5"}]))
        self.assertTrue(self.fake.made("POST", r"/market/orders/5/cancel$"))
        posts = [c[1] for c in self.fake.calls if c[0] == "POST"]
        self.assertEqual(posts, ["/market/orders/5/cancel", "/market/orders"])

    def test_book_as_price_volume_pairs(self):
        self.go(Fake(book={"bids": [["1.40", "10"]], "asks": []}))
        self.assertEqual(self.fake.made("POST", r"/market/orders$")[0][2]["volume"], "4.12")


class TestSafety(Base):
    def test_dry_run_by_default(self):
        del os.environ["DRY_RUN"]
        self.go(Fake())
        self.assertEqual([c for c in self.fake.calls if c[0] == "POST"], [])
        self.assertIn("DRY RUN: would sell 4.12 PRL with a limit at 1.358", self.logs[-1])

    def test_no_trading_outside_production(self):
        os.environ["RAILWAY_ENVIRONMENT_NAME"] = "llmjob-pr-261"
        self.go(Fake(orders=[{"id": 5, "side": "sell"}]))
        self.assertEqual([c for c in self.fake.calls if c[0] == "POST"], [])
        self.assertIn("not production", self.logs[0])

    def test_no_trading_off_railway(self):
        del os.environ["RAILWAY_ENVIRONMENT_NAME"]
        self.go(Fake())
        self.assertEqual([c for c in self.fake.calls if c[0] == "POST"], [])

    def test_an_api_error_fails_the_run(self):
        f = Fake()
        f.routes.insert(0, ("GET", r"/account/balances$", (401, {"errors": ["authz.invalid_signature"]})))
        self.assertEqual(self.go(f), 1)
        self.assertIn("FAILED: balances failed: HTTP 401", self.logs[-1])
        self.assertFalse(f.made("POST", r"/market/orders$"))

    def test_signature(self):
        # hex HMAC-SHA256(secret, nonce + key), as the Peatio docs give it
        import hashlib, hmac
        want = hmac.new(b"s", b"1584087661035k", hashlib.sha256).hexdigest()
        seen = {}

        class Resp:
            status = 200
            def read(self): return b"[]"
            def __enter__(self): return self
            def __exit__(self, *a): return False

        def urlopen(req, timeout):
            seen.update(req.headers)
            return Resp()

        real_http, real_now, real_open = HTTP, t.now_ms, t.urllib.request.urlopen
        t.now_ms, t.urllib.request.urlopen = (lambda: "1584087661035"), urlopen
        try:
            self.assertEqual(real_http("GET", "/account/balances", auth=True), (200, []))
        finally:
            t.now_ms, t.urllib.request.urlopen = real_now, real_open
        self.assertEqual(seen["X-auth-apikey"], "k")
        self.assertEqual(seen["X-auth-nonce"], "1584087661035")
        self.assertEqual(seen["X-auth-signature"], want)

if __name__ == "__main__":
    unittest.main()
