#!/usr/bin/env python3
"""Sell mined PRL on SafeTrade as it arrives, so each payout is locked in at the dollar price of the moment.

Every --poll seconds:
  1. Read the PRL balance that is free to trade.
  2. If it is at least the market's minimum order, read the order book and sell with a limit order that
     reaches no lower than --max-slip percent under the best bid, and never under --floor.
  3. Wait up to --order-ttl seconds for the order to fill, then cancel what is left and start again.
It also refuses to sell when the best bid is more than --max-gap percent under the last trade, which
guards against selling into an empty book.

Dry run is the default: it logs what it would sell and places nothing. Add --live to trade.
--check reads the market, the order book, the ticker and the balance once, prints what it understood,
and exits. Run it first.

Credentials come from SAFETRADE_API_KEY and SAFETRADE_API_SECRET in the environment and are never
printed. Give the key trading rights only, no withdrawals.

Every sale is appended to sales.csv next to this script (time, amount, price, value). The id of the order
in flight is kept in sell_prl_state.json, so a restarted run finishes that order (and only that one: orders
you place yourself are never touched).
Standard library only; Python 3.8 or newer.
"""
import argparse, csv, decimal, hashlib, hmac, json, os, sys, time, urllib.error, urllib.parse, urllib.request

D = decimal.Decimal
HERE = os.path.dirname(os.path.abspath(__file__))


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + " " + msg, flush=True)


class SafeTrade:
    """Signed client for the SafeTrade REST API, the way SafeTrade's example client signs:
    X-Auth-Nonce is a millisecond timestamp and X-Auth-Signature is HMAC-SHA256(secret, nonce + key) in hex."""

    def __init__(self, base_url, key, secret, timeout=20):
        self.base, self.key, self.secret, self.timeout = base_url.rstrip("/"), key, secret, timeout
        self.last_nonce = 0

    def _nonce(self):
        n = int(time.time() * 1000)
        self.last_nonce = n if n > self.last_nonce else self.last_nonce + 1   # never repeat a nonce
        return str(self.last_nonce)

    def _headers(self, signed):
        h = {"Accept": "application/json", "Content-Type": "application/json;charset=utf-8", "User-Agent": "llmjob-sell-prl/1"}
        if signed:
            nonce = self._nonce()
            sig = hmac.new(self.secret.encode(), (nonce + self.key).encode(), hashlib.sha256).hexdigest()
            h.update({"X-Auth-Apikey": self.key, "X-Auth-Nonce": nonce, "X-Auth-Signature": sig})
        return h

    def call(self, method, path, query=None, body=None, signed=False):
        url = self.base + path + ("?" + urllib.parse.urlencode(query) if query else "")
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers=self._headers(signed))
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode() or "null")
        except urllib.error.HTTPError as e:
            text = e.read().decode(errors="replace")
            if "cloudflare" in text.lower():
                text = "blocked by Cloudflare (this network's address is refused; run from another machine)"
            raise RuntimeError(f"{method} {path}: HTTP {e.code}: {text[:300]}")

    # Public
    def market(self, m): return self.call("GET", f"/trade/public/markets/{m}")
    def ticker(self, m): return self.call("GET", f"/trade/public/tickers/{m}")
    def depth(self, m): return self.call("GET", f"/trade/public/markets/{m}/depth")
    # Private
    def balances(self): return self.call("GET", "/trade/account/balances/spot", signed=True)
    def orders(self, m, state): return self.call("GET", "/trade/market/orders", {"market": m, "state": state}, signed=True)
    def order(self, oid): return self.call("GET", f"/trade/market/orders/{oid}", signed=True)
    def cancel(self, oid): return self.call("POST", f"/trade/market/orders/{oid}/cancel", signed=True)
    def sell(self, m, amount, price):
        return self.call("POST", "/trade/market/orders", body={"market": m, "side": "sell", "type": "limit",
                                                               "amount": str(amount), "price": str(price)}, signed=True)


# The API's exact field names aren't published outside its Swagger page, so read them defensively and
# let --check show what was understood.
def num(x):
    try: return D(str(x))
    except (decimal.InvalidOperation, TypeError): return None


def pick(d, *keys):
    for k in keys:
        if isinstance(d, dict) and d.get(k) not in (None, ""): return d[k]
    return None


def parse_ticker(t):
    t = t.get("ticker", t) if isinstance(t, dict) else {}
    return dict(last=num(pick(t, "last", "last_price", "price")), bid=num(pick(t, "buy", "bid", "best_bid")),
                ask=num(pick(t, "sell", "ask", "best_ask")))


def parse_levels(levels):
    out = []
    for lv in levels or []:
        p, a = (lv[0], lv[1]) if isinstance(lv, (list, tuple)) else (pick(lv, "price"), pick(lv, "amount", "volume", "quantity"))
        p, a = num(p), num(a)
        if p and a and p > 0 and a > 0: out.append((p, a))
    return out


def parse_bids(depth):
    bids = parse_levels((depth or {}).get("bids"))
    return sorted(bids, key=lambda x: -x[0])          # best (highest) first, whatever order the API used


def parse_market(m):
    m = m or {}
    return dict(min_amount=num(pick(m, "min_amount", "min_order_amount", "minimum_amount")) or D("0"),
                amount_precision=int(pick(m, "amount_precision", "base_precision") or 4),
                price_precision=int(pick(m, "price_precision", "quote_precision") or 6))


def parse_balance(balances, currency):
    rows = balances.get("data", balances) if isinstance(balances, dict) else balances
    for r in rows or []:
        if str(pick(r, "currency", "currency_id", "asset") or "").lower() == currency:
            return num(pick(r, "balance", "available", "free")) or D("0")
    return D("0")


def plan_sale(balance, bids, last, rules, max_slip, max_gap, floor):
    """What to sell now: (amount, limit price, reason). amount is 0 when nothing should be sold."""
    if balance < max(rules["min_amount"], D("0.00000001")): return D("0"), None, "balance under the minimum order"
    if not bids: return D("0"), None, "no bids"
    best = bids[0][0]
    if last and best < last * (1 - max_gap / 100):
        return D("0"), None, f"best bid {best} is more than {max_gap}% under the last trade {last}"
    lowest = max(best * (1 - max_slip / 100), floor or D("0"))
    if best < lowest: return D("0"), None, f"best bid {best} is under the floor {floor}"
    room, price = D("0"), best
    for p, a in bids:                                  # sweep bids down to the slippage limit
        if p < lowest: break
        room += a; price = p
    q = D(1).scaleb(-rules["amount_precision"])
    amount = min(balance, room).quantize(q, rounding=decimal.ROUND_DOWN)
    if amount < rules["min_amount"] or amount <= 0: return D("0"), None, "bids within the slippage limit are smaller than the minimum order"
    price = price.quantize(D(1).scaleb(-rules["price_precision"]), rounding=decimal.ROUND_DOWN)
    return amount, price, "ok"


def order_state(o):
    return str(pick(o or {}, "state", "status") or "").lower()


def filled(o):
    return num(pick(o or {}, "executed_volume", "filled_amount", "filled")) or D("0")


def record_sale(path, amount, price, quote):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["time_utc", "amount_prl", "price", "value_" + quote])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), str(amount), str(price), str((amount * price).quantize(D("0.0001")))])


def finish(st, oid, amt, price, sales_log, quote, base, state_path):
    """Cancel what is left of our order, record what filled, and forget the order."""
    o = st.order(oid)
    if order_state(o) in ("wait", "pending", "open", ""):
        st.cancel(oid); o = st.order(oid)
    got = filled(o)
    avg = num(pick(o, "avg_price", "average_price")) or price
    if got > 0:
        record_sale(sales_log, got, avg, quote); log(f"sold {got} {base} at {avg} = {(got * avg).quantize(D('0.0001'))} {quote}")
    if amt is not None and got < amt: log(f"{amt - got} {base} left unsold; trying again next round")
    if os.path.exists(state_path): os.remove(state_path)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Sell PRL on SafeTrade as it arrives.")
    ap.add_argument("--market", default="prlusdt", help="SafeTrade market id (default prlusdt; prlusdc for USDC)")
    ap.add_argument("--base-url", default="https://safe.trade/api/v2")
    ap.add_argument("--live", action="store_true", help="place real orders (default: dry run)")
    ap.add_argument("--check", action="store_true", help="read everything once, print it, place nothing, exit")
    ap.add_argument("--poll", type=float, default=60, help="seconds between checks (default 60)")
    ap.add_argument("--max-slip", type=D, default=D("1"), help="lowest price as %% under the best bid (default 1)")
    ap.add_argument("--max-gap", type=D, default=D("5"), help="refuse if the best bid is this %% under the last trade (default 5)")
    ap.add_argument("--floor", type=D, default=None, help="never sell under this price")
    ap.add_argument("--order-ttl", type=float, default=120, help="seconds before an unfilled order is cancelled (default 120)")
    ap.add_argument("--once", action="store_true", help="one round, then exit")
    ap.add_argument("--sales-log", default=os.path.join(HERE, "sales.csv"))
    ap.add_argument("--state", default=os.path.join(HERE, "sell_prl_state.json"))
    a = ap.parse_args(argv)

    key, secret = os.environ.get("SAFETRADE_API_KEY", ""), os.environ.get("SAFETRADE_API_SECRET", "")
    if not key or not secret: sys.exit("Set SAFETRADE_API_KEY and SAFETRADE_API_SECRET in the environment.")
    base = a.market[:3] if a.market.endswith(("usdt", "usdc")) else "prl"
    quote = a.market[len(base):]
    st = SafeTrade(a.base_url, key, secret)
    rules = parse_market(st.market(a.market))

    if a.check:
        t, bids = parse_ticker(st.ticker(a.market)), parse_bids(st.depth(a.market))
        bal = parse_balance(st.balances(), base)
        print(f"market {a.market}: min order {rules['min_amount']} {base}, amount precision {rules['amount_precision']}, price precision {rules['price_precision']}")
        print(f"ticker: last {t['last']}, bid {t['bid']}, ask {t['ask']}")
        print(f"order book: {len(bids)} bid levels; best {bids[0] if bids else None}")
        print(f"balance: {bal} {base} free to trade")
        amt, price, why = plan_sale(bal, bids, t["last"], rules, a.max_slip, a.max_gap, a.floor)
        print(f"would sell now: {amt} {base} at {price} ({why})")
        return

    log(f"selling {base.upper()} for {quote.upper()} on {a.market} as it arrives; {'LIVE' if a.live else 'dry run, nothing will be placed'}")
    if a.live and os.path.exists(a.state):   # an earlier run stopped with an order in flight: finish it
        with open(a.state) as f: prev = json.load(f)
        log(f"finishing order {prev['id']} left by an earlier run")
        finish(st, prev["id"], None, num(prev.get("price")), a.sales_log, quote, base, a.state)
    while True:
        try:
            t, bids = parse_ticker(st.ticker(a.market)), parse_bids(st.depth(a.market))
            bal = parse_balance(st.balances(), base)
            amt, price, why = plan_sale(bal, bids, t["last"], rules, a.max_slip, a.max_gap, a.floor)
            if amt > 0 and not a.live:
                log(f"dry run: would sell {amt} {base} at {price} (best bid {bids[0][0]}, last {t['last']})")
            elif amt > 0:
                o = st.sell(a.market, amt, price); oid = pick(o, "id")
                if oid is None: raise RuntimeError(f"order response has no id: {str(o)[:300]}")
                with open(a.state, "w") as f: json.dump({"id": oid, "amount": str(amt), "price": str(price)}, f)
                log(f"placed sell {oid}: {amt} {base} at {price}")
                deadline = time.time() + a.order_ttl
                while time.time() < deadline:
                    time.sleep(min(5, max(1, a.order_ttl / 10)))
                    if order_state(st.order(oid)) in ("done", "filled", "cancel", "cancelled", "canceled", "reject", "rejected"): break
                finish(st, oid, amt, price, a.sales_log, quote, base, a.state)
            elif bal > 0:
                log(f"holding {bal} {base}: {why}")
        except Exception as ex:
            log(f"error: {ex}")
        if a.once: return
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
