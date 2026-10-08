"""Tests for sell_prl.py against a fake SafeTrade server. Run: python3 tuning/test_sell_prl.py"""
import contextlib, csv, hashlib, hmac, io, json, os, sys, tempfile, threading, unittest
from decimal import Decimal as D
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sell_prl

KEY, SECRET = "test-key-123", "test-secret-456"


class Fake:
    """Order book, balances and orders, matched the way an exchange does: a sell fills against bids at or
    above its price, best bid first, at each bid's price."""
    def __init__(self, bids, balance, last="1.42", fill=True):
        self.bids = [[D(p), D(a)] for p, a in bids]; self.prl = D(balance); self.usdt = D("0")
        self.last, self.fill, self.orders, self.posts, self.nonces, self.bad_auth = D(last), fill, {}, 0, set(), 0

    def place(self, body):
        amount, price = D(body["amount"]), D(body["price"])
        assert body["side"] == "sell" and body["type"] == "limit" and amount <= self.prl
        self.prl -= amount; got = D("0"); value = D("0")
        if self.fill:
            for lv in self.bids:
                if lv[0] < price or got == amount: break
                take = min(lv[1], amount - got); lv[1] -= take; got += take; value += take * lv[0]
            self.bids = [lv for lv in self.bids if lv[1] > 0]
        self.usdt += value
        oid = len(self.orders) + 1
        self.orders[oid] = dict(id=oid, market=body["market"], side="sell", price=str(price), origin_amount=str(amount),
                                filled_amount=str(got), avg_price=str((value / got) if got else price),
                                state="done" if got == amount else "wait", _left=amount - got)
        return self.orders[oid]

    def cancel(self, oid):
        o = self.orders[oid]
        if o["state"] == "wait": self.prl += o["_left"]; o["state"] = "cancel"
        return o


def serve(fake):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def send(self, code, obj):
            b = json.dumps(obj, default=str).encode(); self.send_response(code)
            self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(b)
        def authed(self):
            k, n, s = (self.headers.get(h) for h in ("X-Auth-Apikey", "X-Auth-Nonce", "X-Auth-Signature"))
            ok = k == KEY and n and n not in fake.nonces and s == hmac.new(SECRET.encode(), (n + k).encode(), hashlib.sha256).hexdigest()
            if not ok: fake.bad_auth += 1; self.send(401, {"errors": ["authz.invalid_signature"]}); return False
            fake.nonces.add(n); return True
        def do_GET(self):
            p = self.path.split("?")[0]
            if p == "/api/v2/trade/public/markets/prlusdt":
                return self.send(200, {"id": "prlusdt", "min_amount": "0.1", "amount_precision": 2, "price_precision": 4})
            if p == "/api/v2/trade/public/tickers/prlusdt":
                return self.send(200, {"ticker": {"last": str(fake.last), "buy": str(fake.bids[0][0]) if fake.bids else None, "sell": "1.43"}})
            if p == "/api/v2/trade/public/markets/prlusdt/depth":
                return self.send(200, {"asks": [["1.43", "10"]], "bids": [[str(a), str(b)] for a, b in fake.bids]})
            if not self.authed(): return
            if p == "/api/v2/trade/account/balances/spot":
                return self.send(200, [{"currency": "prl", "balance": str(fake.prl), "locked": "0"}, {"currency": "usdt", "balance": str(fake.usdt)}])
            if p.startswith("/api/v2/trade/market/orders/"):
                return self.send(200, {k: v for k, v in fake.orders[int(p.rsplit("/", 1)[1])].items() if not k.startswith("_")})
            self.send(404, {"errors": ["not found"]})
        def do_POST(self):
            p = self.path; body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"null")
            if not self.authed(): return
            if p == "/api/v2/trade/market/orders":
                fake.posts += 1; return self.send(201, {k: v for k, v in fake.place(body).items() if not k.startswith("_")})
            if p.endswith("/cancel"):
                return self.send(201, {k: v for k, v in fake.cancel(int(p.split("/")[-2])).items() if not k.startswith("_")})
            self.send(404, {"errors": ["not found"]})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class SellPrlTest(unittest.TestCase):
    def setUp(self):
        os.environ["SAFETRADE_API_KEY"], os.environ["SAFETRADE_API_SECRET"] = KEY, SECRET
        self.dir = tempfile.mkdtemp()
        self.sales, self.state = os.path.join(self.dir, "sales.csv"), os.path.join(self.dir, "state.json")

    def run_main(self, fake, *args):
        srv = serve(fake); out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                sell_prl.main(["--base-url", f"http://127.0.0.1:{srv.server_address[1]}/api/v2", "--sales-log", self.sales,
                               "--state", self.state, "--once", *args])
        finally: srv.shutdown(); srv.server_close()
        text = out.getvalue()
        self.assertNotIn(SECRET, text); self.assertNotIn(KEY, text)       # credentials never printed
        return text

    def sales_rows(self):
        if not os.path.exists(self.sales): return []
        with open(self.sales) as fh: return list(csv.DictReader(fh))

    def test_check_places_nothing(self):
        f = Fake([("1.41", "2"), ("1.405", "5")], "3.257")
        out = self.run_main(f, "--check")
        self.assertIn("balance: 3.257 prl", out); self.assertIn("would sell now: 3.25 prl at 1.4050 (ok)", out)
        self.assertEqual(f.posts, 0); self.assertEqual(f.bad_auth, 0)

    def test_dry_run_places_nothing(self):
        f = Fake([("1.41", "2"), ("1.405", "5")], "3.257")
        out = self.run_main(f)
        self.assertIn("dry run: would sell 3.25 prl at 1.4050", out); self.assertEqual(f.posts, 0); self.assertEqual(f.prl, D("3.257"))

    def test_live_sells_the_balance_within_the_slippage_limit(self):
        f = Fake([("1.41", "2"), ("1.405", "5"), ("1.30", "100")], "3.257")
        out = self.run_main(f, "--live", "--order-ttl", "2")
        self.assertEqual(f.posts, 1); self.assertEqual(f.bad_auth, 0)
        self.assertEqual(f.prl, D("0.007"))                                # under the 0.1 minimum: left for later
        self.assertEqual(f.usdt, D("2") * D("1.41") + D("1.25") * D("1.405"))  # never reached the 1.30 bid
        rows = self.sales_rows(); self.assertEqual(len(rows), 1); self.assertEqual(rows[0]["amount_prl"], "3.25")
        self.assertIn("sold 3.25 prl", out); self.assertFalse(os.path.exists(self.state))

    def test_thin_book_sells_only_what_the_bids_take(self):
        f = Fake([("1.41", "0.5"), ("1.20", "50")], "3")
        self.run_main(f, "--live", "--order-ttl", "2")
        self.assertEqual(f.prl, D("2.5")); self.assertEqual(f.usdt, D("0.5") * D("1.41"))

    def test_refuses_when_best_bid_is_far_under_the_last_trade(self):
        f = Fake([("1.41", "10")], "3", last="1.60")
        out = self.run_main(f, "--live")
        self.assertEqual(f.posts, 0); self.assertIn("more than 5% under the last trade", out)

    def test_floor_blocks_sales_under_it(self):
        f = Fake([("1.41", "10")], "3")
        out = self.run_main(f, "--live", "--floor", "1.5")
        self.assertEqual(f.posts, 0); self.assertIn("under the floor", out)

    def test_unfilled_order_is_cancelled_after_its_ttl(self):
        f = Fake([("1.41", "10")], "3", fill=False)
        out = self.run_main(f, "--live", "--order-ttl", "2")
        self.assertEqual(f.orders[1]["state"], "cancel"); self.assertEqual(f.prl, D("3"))
        self.assertEqual(self.sales_rows(), []); self.assertIn("left unsold", out); self.assertFalse(os.path.exists(self.state))

    def test_restart_finishes_only_its_own_order(self):
        f = Fake([("1.41", "10")], "3", fill=False)
        mine = f.place({"market": "prlusdt", "side": "sell", "type": "limit", "amount": "1", "price": "1.41"})
        theirs = f.place({"market": "prlusdt", "side": "sell", "type": "limit", "amount": "1", "price": "2.00"})
        with open(self.state, "w") as fh: json.dump({"id": mine["id"], "amount": "1", "price": "1.41"}, fh)
        f.fill = True
        out = self.run_main(f, "--live", "--order-ttl", "2")
        self.assertEqual(f.orders[mine["id"]]["state"], "cancel"); self.assertEqual(f.orders[theirs["id"]]["state"], "wait")
        self.assertIn(f"finishing order {mine['id']}", out)

    def test_nonces_never_repeat(self):
        c = sell_prl.SafeTrade("http://x", KEY, SECRET)
        ns = [int(c._nonce()) for _ in range(1000)]
        self.assertEqual(len(set(ns)), 1000); self.assertEqual(ns, sorted(ns))

    def test_plan_reads_other_field_shapes(self):
        self.assertEqual(sell_prl.parse_bids({"bids": [{"price": "1.4", "amount": "2"}, {"price": "1.5", "amount": "1"}]})[0], (D("1.5"), D("1")))
        self.assertEqual(sell_prl.parse_balance({"data": [{"currency_id": "PRL", "available": "4.2"}]}, "prl"), D("4.2"))
        self.assertEqual(sell_prl.parse_ticker({"last_price": "1.4", "bid": "1.39"})["bid"], D("1.39"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
