# Tests for trader.py with SafeTrade faked. Run: python3 -m unittest discover -s trader
import importlib.util, json, os, re, select, unittest

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
                       ("GET", r"/depth$", (200, book)),
                       ("GET", r"/market/orders$", (200, list(orders))),
                       ("GET", r"/account/balances$", (200, [{"currency": "usdt", "balance": "9"},
                                                              {"currency": "prl", "balance": balance}])),
                       ("POST", r"/market/orders$", (201, {"id": 77, "state": "wait"})),
                       ("GET", r"/market/orders/77$", (200, {"id": 77, "state": "done", "filled_amount": 4.12,
                                                              "origin_amount": 4.12, "price": 1.358})),
                       ("GET", r"/market/trades$", (200, [
                           {"order_id": 77, "amount": 3, "total": 4.2, "fee": 0.0084, "fee_currency": "usdt"},
                           {"order_id": 77, "amount": 1.12, "total": 1.5456, "fee": 0.0031, "fee_currency": "usdt"},
                           {"order_id": 12, "amount": 9, "total": 9, "fee": 0, "fee_currency": "usdt"}])),
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
        t.MARKET = "prlusdt"
        os.environ["DRY_RUN"] = "0"
        os.environ["RAILWAY_ENVIRONMENT_NAME"] = "production"

    def saw(self, text):
        self.assertTrue(any(text in line for line in self.logs), f"{text!r} not in {self.logs}")

    def go(self, fake):
        self.fake = fake
        t.http = fake
        return t.main()


class TestSell(Base):
    def test_sells_what_the_book_takes_within_slippage(self):
        self.assertEqual(self.go(Fake()), 0)
        order = self.fake.made("POST", r"/market/orders$")[0]
        # 3% under the best bid of 1.40 is 1.358: the bids at 1.40 and 1.38 (5 PRL) qualify; we have 4.123
        self.assertEqual(order[1], "/trade/market/orders")
        self.assertEqual(order[2], {"market": "prlusdt", "side": "sell", "amount": 4.12, "price": 1.358})
        self.assertTrue(order[3])
        # 5.7456 USDT for 4.12 PRL from this order's two trades; the other order's trade isn't counted
        self.saw("SOLD 4.12 of 4.12 PRL at 1.3946 USDT average for 5.75 USDT (fee 0.0115 USDT")

    def test_sells_only_what_the_bids_above_the_floor_take(self):
        self.go(Fake(balance="20"))
        self.assertEqual(self.fake.made("POST", r"/market/orders$")[0][2]["amount"], 5.0)

    def test_never_under_min_price(self):
        t.MIN_PRICE = 1.45
        self.go(Fake())
        self.assertFalse(self.fake.made("POST", r"/market/orders$"))
        self.saw("under the floor")

    def test_dust_is_left(self):
        self.go(Fake(balance="0.3"))
        self.assertFalse(self.fake.made("POST", r"/market/orders$"))
        self.saw("nothing to sell")

    def test_last_runs_open_sell_is_cancelled_first(self):
        self.go(Fake(orders=[{"id": 5, "side": "sell", "remaining_volume": "1", "price": "1.5"}]))
        self.assertTrue(self.fake.made("POST", r"/market/orders/5/cancel$"))
        posts = [c[1] for c in self.fake.calls if c[0] == "POST"]
        self.assertEqual(posts, ["/trade/market/orders/5/cancel", "/trade/market/orders"])

    def test_book_as_price_volume_pairs(self):
        self.go(Fake(book={"bids": [["1.40", "10"]], "asks": []}))
        self.assertEqual(self.fake.made("POST", r"/market/orders$")[0][2]["amount"], 4.12)


class TestPaths(Base):
    def test_the_documented_paths(self):
        self.go(Fake())
        paths = {c[1] for c in self.fake.calls}
        for p in ("/trade/public/markets", "/trade/public/markets/prlusdt/depth", "/trade/account/balances/spot",
                  "/trade/market/orders"):
            self.assertIn(p, paths)
        self.assertFalse([p for p in paths if p.startswith("/peatio")])

    def test_the_old_public_paths_are_a_fallback(self):
        f = Fake()
        f.routes = [r for r in f.routes if r[1] not in (r"/public/markets$", r"/depth$")]
        f.routes.insert(0, ("GET", r"^/peatio/public/markets$", (200, MARKETS)))
        f.routes.insert(0, ("GET", r"^/peatio/public/markets/prlusdt/order-book$", (200, BOOK)))
        self.go(f)
        self.assertEqual(f.made("POST", r"/market/orders$")[0][2]["amount"], 4.12)

    def test_the_market_by_id(self):
        f = Fake()
        f.routes.insert(0, ("GET", r"^/trade/public/markets/prlusdt$", (200, {"id": "prlusdt", "amount_precision": 1,
                                                                              "price_precision": 3})))
        self.go(f)
        self.assertFalse(f.made("GET", r"/public/markets$"))
        self.assertEqual(f.made("POST", r"/market/orders$")[0][2]["amount"], 4.1)

    def test_balances_come_from_the_first_path_that_answers(self):
        f = Fake(balance="99")
        f.routes.insert(0, ("GET", r"/trade/account/balances/spot$", (200, [{"currency": "prl", "balance": "4.123"}])))
        self.go(f)
        self.assertEqual(f.made("POST", r"/market/orders$")[0][2]["amount"], 4.12)
        self.assertFalse(f.made("GET", r"/trade/account/balances$"))

    def test_a_cancelled_rest_is_reported_from_its_amounts(self):
        self.go(Fake(orders=[{"id": 5, "side": "sell", "market": "prlusdt", "origin_amount": 2, "filled_amount": 0.5,
                              "price": 1.5}]))
        self.assertIn("cancelled open sell 5: 1.5 at 1.5", self.logs)

    def test_an_unpaged_market_list_is_read_once(self):
        f = Fake()
        many = [{"id": f"x{i}usdt"} for i in range(150)]
        f.routes.insert(0, ("GET", r"^/trade/public/markets$", (200, many)))
        f.routes.insert(0, ("GET", r"^/peatio/public/markets$", (404, {})))
        self.assertEqual(self.go(f), 1)
        self.assertEqual(len(f.made("GET", r"^/trade/public/markets$")), 2)   # page 2 repeats page 1, so it stops
        self.saw("among 150 listed")


class TestStatus(Base):
    def run_capturing(self, fake):
        import contextlib, io
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.go(fake)
        return [json.loads(l[7:]) for l in out.getvalue().splitlines() if l.startswith("STATUS ")]

    def test_every_run_reports_the_account_and_lifetime_sales(self):
        st = self.run_capturing(Fake())
        self.assertEqual(len(st), 1)
        # balances: PRL 4.123, USDT 9; lifetime: every sell trade on the market, the other order's included
        self.assertEqual((st[0]["prl"], st[0]["usdt"], st[0]["trades"]), (4.123, 9.0, 3))
        self.assertAlmostEqual(st[0]["sold"], 13.12)
        self.assertAlmostEqual(st[0]["received"], 14.7456)
        self.assertEqual(st[0]["last"]["order"], 77)
        self.assertAlmostEqual(st[0]["last"]["average"], 1.394563, places=5)
        self.saw("ACCOUNT 4.123 PRL free + 0 in orders · 9.00 USDT free + 0.00 in orders · sold so far 13.12 PRL for 14.75 USDT")

    def test_a_dry_run_reports_too(self):
        del os.environ["DRY_RUN"]
        st = self.run_capturing(Fake())
        self.assertEqual(len(st), 1)
        self.assertIsNone(st[0]["last"])


class TestMarket(Base):
    def test_the_market_is_found_on_a_later_page(self):
        page1 = [{"id": f"x{i}usdt", "base_unit": f"x{i}", "quote_unit": "usdt"} for i in range(100)]

        class Paged(Fake):
            def __call__(self, method, path, params=None, auth=False):
                if method == "GET" and path == "/trade/public/markets":
                    self.calls.append((method, path, params, auth))
                    return 200, page1 if params["page"] == 1 else MARKETS
                return super().__call__(method, path, params, auth)

        self.go(Paged())
        self.assertEqual([c[2]["page"] for c in self.fake.made("GET", r"/public/markets$")], [1, 2])
        self.assertEqual(self.fake.made("POST", r"/market/orders$")[0][2]["market"], "prlusdt")

    def test_the_market_is_matched_on_its_currencies(self):
        f = Fake()
        f.routes.insert(0, ("GET", r"/public/markets$", (200, [{"id": "PRL_USDT", "base_unit": "prl", "quote_unit": "usdt",
                                                                 "amount_precision": 2, "price_precision": 4}])))
        self.go(f)
        self.assertEqual(f.made("POST", r"/market/orders$")[0][2]["market"], "PRL_USDT")
        self.assertTrue(f.made("GET", r"^/trade/public/markets/PRL_USDT/depth$"))

    def test_a_missing_market_says_what_is_listed(self):
        f = Fake()
        f.routes.insert(0, ("GET", r"/public/markets$", (200, [{"id": "prlbtc"}, {"id": "ethusdt"}])))
        self.assertEqual(self.go(f), 1)
        self.saw("no PRL/USDT market among 2 listed; ids with 'prl': ['prlbtc']")


class TestSafety(Base):
    def test_dry_run_by_default(self):
        del os.environ["DRY_RUN"]
        self.go(Fake())
        self.assertEqual([c for c in self.fake.calls if c[0] == "POST"], [])
        self.saw("DRY RUN: would sell 4.12 PRL with a limit at 1.358")

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
        self.assertIn("FAILED: balances failed: /trade/account/balances/spot HTTP 404; /trade/account/balances HTTP 401",
                      self.logs[-1])
        self.assertFalse(f.made("POST", r"/market/orders$"))

    def test_a_hung_run_is_stopped(self):
        # A call that never answers must not hold the cron service: Railway skips every run while one is going.
        def hangs(method, path, params=None, auth=False):
            select.select([], [], [], 5)
            return 200, MARKETS
        t.MAX_RUN_SECONDS = 1
        try:
            self.assertEqual(self.go(hangs), 1)
        finally:
            t.MAX_RUN_SECONDS = 300
        self.saw("run started")
        self.saw("FAILED: run took over 1 s")

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
