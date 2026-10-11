#!/usr/bin/env python3
# Sells mined PRL for USDT on SafeTrade, once an hour (a Railway cron service; see README.md).
# Each run: cancel our open orders on the market (last run's unfilled rest), read the PRL balance, then sell
# what the order book can take without going more than MAX_SLIPPAGE below the best bid, and never under
# MIN_PRICE. It's one limit order at that floor price, so it fills against every bid at or above it, best
# first. Anything the book can't take now waits for the next run.
# It trades only when DRY_RUN=0 and it runs in Railway's production environment; anything else just logs
# what it would do. Standard library only.
import hashlib, hmac, json, math, os, sys, time, urllib.error, urllib.parse, urllib.request

BASE = os.environ.get("SAFETRADE_BASE", "https://safe.trade/api/v2/peatio").rstrip("/")
MARKET = os.environ.get("MARKET", "prlusdt")
COIN = os.environ.get("COIN", "prl")
KEY = os.environ.get("SAFETRADE_API_KEY", "")
SECRET = os.environ.get("SAFETRADE_API_SECRET", "")
MAX_SLIPPAGE = float(os.environ.get("MAX_SLIPPAGE", "0.03"))   # sell down to 3% under the best bid
MIN_PRICE = float(os.environ.get("MIN_PRICE", "0"))           # never sell under this (USDT per PRL)
MIN_VOLUME = float(os.environ.get("MIN_VOLUME", "0.5"))        # don't bother with less PRL than this


def now_ms():
    return str(int(time.time() * 1000))


def http(method, path, params=None, auth=False):
    """Returns (status, parsed JSON or None). Signs with the API key when auth is set:
    X-Auth-Signature = hex HMAC-SHA256(secret, nonce + key), nonce a millisecond timestamp."""
    url = BASE + path
    data = None
    if method == "GET" and params:
        url += "?" + urllib.parse.urlencode(params)
    elif params is not None:
        data = json.dumps(params).encode()
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", "llmjob-trader/1.0")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if auth:
        nonce = now_ms()
        req.add_header("X-Auth-Apikey", KEY)
        req.add_header("X-Auth-Nonce", nonce)
        req.add_header("X-Auth-Signature", hmac.new(SECRET.encode(), (nonce + KEY).encode(), hashlib.sha256).hexdigest())
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            txt = r.read().decode()
            return r.status, (json.loads(txt) if txt.strip() else None)
    except urllib.error.HTTPError as e:
        txt = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(txt)
        except ValueError:
            return e.code, {"raw": txt[:300]}
    except Exception as e:
        return 0, {"error": str(e)[:300]}


def log(msg):
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


class Fail(Exception):
    pass


def need(status, body, what):
    if not 200 <= status < 300:
        raise Fail(f"{what} failed: HTTP {status} {json.dumps(body)[:300]}")
    return body


def floor_to(x, places):
    f = 10 ** places
    return math.floor(x * f + 1e-9) / f


def ceil_to(x, places):
    f = 10 ** places
    return math.ceil(x * f - 1e-9) / f


def levels(side):
    """Order book levels as (price, volume). Peatio's order-book gives orders with price and remaining_volume;
    some versions give [price, volume] pairs."""
    out = []
    for o in side or []:
        if isinstance(o, dict):
            p, v = o.get("price"), o.get("remaining_volume", o.get("volume"))
        else:
            p, v = o[0], o[1]
        out.append((float(p), float(v)))
    return out


def market_info():
    s, j = http("GET", "/public/markets")
    need(s, j, "markets")
    for m in j or []:
        if m.get("id") == MARKET:
            return {"amount_precision": int(m.get("amount_precision", 4)), "price_precision": int(m.get("price_precision", 6)),
                    "min_amount": float(m.get("min_amount") or 0)}
    raise Fail(f"market {MARKET} not listed")


def plan(balance, bids, info):
    """What to sell: (volume, price), or None with the reason."""
    if not bids:
        return None, "no bids"
    best = max(p for p, _ in bids)
    price = ceil_to(max(best * (1 - MAX_SLIPPAGE), MIN_PRICE), info["price_precision"])
    if price > best:
        return None, f"best bid {best} is under the floor {price}"
    room = sum(v for p, v in bids if p >= price)
    vol = floor_to(min(balance, room), info["amount_precision"])
    if vol < max(MIN_VOLUME, info["min_amount"]):
        return None, f"only {vol} {COIN.upper()} to sell (balance {balance}, bids down to {price} take {room:.4f})"
    return (vol, price), None


def run(live):
    info = market_info()

    # Last run's unfilled rest: cancel it, so its PRL counts in the balance again.
    s, orders = http("GET", "/market/orders", {"market": MARKET, "state": "wait"}, auth=True)
    need(s, orders, "open orders")
    for o in orders or []:
        if o.get("side") == "sell":
            if live:
                need(*http("POST", f"/market/orders/{o['id']}/cancel", {}, auth=True), f"cancel order {o['id']}")
            log(f"{'cancelled' if live else 'would cancel'} open sell {o['id']}: {o.get('remaining_volume')} at {o.get('price')}")
    if orders and live:
        time.sleep(2)

    s, bal = http("GET", "/account/balances", auth=True)
    need(s, bal, "balances")
    row = next((b for b in bal or [] if str(b.get("currency", "")).lower() == COIN), {})
    balance = float(row.get("balance") or 0)

    s, book = http("GET", f"/public/markets/{MARKET}/order-book", {"asks_limit": 1, "bids_limit": 50})
    need(s, book, "order book")
    bids = levels((book or {}).get("bids"))

    what, why = plan(balance, bids, info)
    if what is None:
        log(f"nothing to sell: {why}")
        return
    vol, price = what
    if not live:
        log(f"DRY RUN: would sell {vol} {COIN.upper()} with a limit at {price} (best bid {max(p for p, _ in bids)}, "
            f"balance {balance})")
        return
    s, o = http("POST", "/market/orders", {"market": MARKET, "side": "sell", "volume": str(vol), "ord_type": "limit",
                                            "price": str(price)}, auth=True)
    need(s, o, "sell order")
    time.sleep(3)
    s, o2 = http("GET", f"/market/orders/{o['id']}", auth=True)
    o = o2 if 200 <= s < 300 and isinstance(o2, dict) else o
    done = float(o.get("executed_volume") or 0)
    avg = o.get("avg_price") or o.get("price")
    log(f"SOLD {done} of {vol} {COIN.upper()} at about {avg} USDT (limit {price}, order {o.get('id')}, "
        f"state {o.get('state')}); the rest waits for the next run")


def main():
    env = os.environ.get("RAILWAY_ENVIRONMENT_NAME")
    live = os.environ.get("DRY_RUN", "1") == "0" and env == "production"
    if os.environ.get("DRY_RUN", "1") == "0" and not live:
        log(f"DRY_RUN=0 but this is environment {env!r}, not production: dry run only")
    if not KEY or not SECRET:
        log("SAFETRADE_API_KEY and SAFETRADE_API_SECRET aren't set; nothing to do")
        return 0
    try:
        run(live)
    except Fail as e:
        log(f"FAILED: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
