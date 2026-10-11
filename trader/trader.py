#!/usr/bin/env python3
# Sells mined PRL for USDT on SafeTrade, once an hour (a Railway cron service; see README.md).
# Each run: cancel our open orders on the market (last run's unfilled rest), read the PRL balance, then sell
# what the order book can take without going more than MAX_SLIPPAGE below the best bid, and never under
# MIN_PRICE. It's one limit order at that floor price, so it fills against every bid at or above it, best
# first. Anything the book can't take now waits for the next run.
# It trades only when DRY_RUN=0 and it runs in Railway's production environment; anything else just logs
# what it would do. Standard library only.
import hashlib, hmac, json, math, os, sys, time, urllib.error, urllib.parse, urllib.request

# Paths from SafeTrade's REST API page (safe.trade/api/v2): private calls under /trade (balances at
# /trade/account/balances/spot; orders at /trade/market/orders, /{id} and /{id}/cancel), public data under
# /trade/public. /peatio/market/orders answers 404 there. The old /peatio/public paths answered on the first run,
# so they stay as a fallback for market data.
BASE = os.environ.get("SAFETRADE_BASE", "https://safe.trade/api/v2").rstrip("/")
PUBS, PRIV = ["/trade/public", "/peatio/public"], "/trade"
BALANCE_PATHS = ["/trade/account/balances/spot", "/trade/account/balances"]
COIN = os.environ.get("COIN", "prl")
QUOTE = os.environ.get("QUOTE", "usdt")
MARKET = os.environ.get("MARKET", COIN + QUOTE)   # replaced by the id SafeTrade uses, once found
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
    return round(math.floor(x * f + 1e-9) / f, places)   # round(): no 4.1200000001 in the JSON


def ceil_to(x, places):
    f = 10 ** places
    return round(math.ceil(x * f - 1e-9) / f, places)


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


def market_fields(m):
    return {"amount_precision": int(m.get("amount_precision", 4)), "price_precision": int(m.get("price_precision", 6)),
            "min_amount": float(first(m, "min_amount", "min_volume") or 0)}


def market_info():
    """The PRL/USDT market: by id first, then from the market list (100 to a page, every page read), matched on
    its id or on its base and quote currencies, since the id's spelling can differ."""
    global MARKET
    seen = []
    for pub in PUBS:
        s, m = http("GET", f"{pub}/markets/{MARKET}")
        if 200 <= s < 300 and isinstance(m, dict) and m.get("id"):
            MARKET = str(m["id"])
            return pub, market_fields(m)
        for page in range(1, 21):
            s, j = http("GET", pub + "/markets", {"limit": 100, "page": page})
            if not 200 <= s < 300 or not isinstance(j, list) or not j:
                break
            if page > 1 and j[0].get("id") == top:
                break   # the trade API returns the whole list on every page
            top = j[0].get("id")
            for m in j:
                mid = str(m.get("id", ""))
                base = str(m.get("base_unit") or m.get("base_currency") or "").lower()
                quote = str(m.get("quote_unit") or m.get("quote_currency") or "").lower()
                seen.append(mid)
                if mid.lower() == MARKET.lower() or (base == COIN and quote == QUOTE):
                    MARKET = mid
                    return pub, market_fields(m)
            if len(j) < 100:
                break
    near = sorted({i for i in seen if COIN in i.lower()})
    raise Fail(f"no {COIN.upper()}/{QUOTE.upper()} market among {len(set(seen))} listed; ids with {COIN!r}: {near or 'none'}")


def order_book(pub):
    """Bids as (price, volume): /markets/{id}/depth on the trade API, /order-book on the old one."""
    for path, params in ((f"{pub}/markets/{MARKET}/depth", {"limit": 50}),
                         (f"{pub}/markets/{MARKET}/order-book", {"asks_limit": 1, "bids_limit": 50})):
        s, book = http("GET", path, params)
        if 200 <= s < 300 and isinstance(book, dict) and "bids" in book:
            return levels(book.get("bids"))
    raise Fail(f"order book failed for {MARKET}")


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


def first(d, *keys):
    """The first of these fields that's set: Peatio and SafeTrade's trade API name some differently."""
    return next((d[k] for k in keys if d.get(k) not in (None, "")), None)


def balance_of():
    tried = []
    for path in BALANCE_PATHS:
        s, bal = http("GET", path, auth=True)
        if 200 <= s < 300 and isinstance(bal, (list, dict)):
            rows = bal if isinstance(bal, list) else bal.get("balances") or bal.get("data") or [bal]
            row = next((b for b in rows if isinstance(b, dict) and str(b.get("currency", "")).lower() == COIN), {})
            log(f"balances from {path}")
            return float(first(row, "balance", "available", "free") or 0)
        tried.append(f"{path} HTTP {s}")
    raise Fail("balances failed: " + "; ".join(tried))


def run(live):
    pub, info = market_info()

    # Last run's unfilled rest: cancel it, so its PRL counts in the balance again.
    s, orders = http("GET", PRIV + "/market/orders", {"market": MARKET, "state": "wait", "side": "sell"}, auth=True)
    need(s, orders, "open orders")
    for o in orders or []:
        if o.get("side") == "sell" and str(o.get("market", MARKET)).lower() == MARKET.lower():
            if live:
                need(*http("POST", f"{PRIV}/market/orders/{o['id']}/cancel", {}, auth=True), f"cancel order {o['id']}")
            rest = float(o.get("origin_amount") or 0) - float(o.get("filled_amount") or 0)
            log(f"{'cancelled' if live else 'would cancel'} open sell {o['id']}: {rest:g} at {o.get('price')}")
    if orders and live:
        time.sleep(2)

    balance = balance_of()

    bids = order_book(pub)

    what, why = plan(balance, bids, info)
    if what is None:
        log(f"nothing to sell: {why}")
        return
    vol, price = what
    if not live:
        log(f"DRY RUN: would sell {vol} {COIN.upper()} with a limit at {price} (best bid {max(p for p, _ in bids)}, "
            f"balance {balance})")
        return
    # CreateOrderParams: market, side, amount and price as numbers; a price makes it a limit order (no type field).
    s, o = http("POST", PRIV + "/market/orders", {"market": MARKET, "side": "sell", "amount": vol, "price": price}, auth=True)
    need(s, o, "sell order")
    time.sleep(3)
    s, o2 = http("GET", f"{PRIV}/market/orders/{o['id']}", auth=True)
    o = o2 if 200 <= s < 300 and isinstance(o2, dict) else o
    done = float(o.get("filled_amount") or 0)
    # The order doesn't carry an average price; its trades do, with the fee.
    s, trades = http("GET", PRIV + "/market/trades", {"market": MARKET, "side": "sell", "limit": 100}, auth=True)
    mine = [t for t in trades or [] if str(t.get("order_id")) == str(o.get("id"))] if 200 <= s < 300 else []
    got = sum(float(t.get("total") or 0) for t in mine)
    fee = sum(float(t.get("fee") or 0) for t in mine)
    amt = sum(float(t.get("amount") or 0) for t in mine)
    avg = f"{got / amt:.4f}" if amt else "?"
    log(f"SOLD {done} of {vol} {COIN.upper()} at {avg} USDT average for {got:.2f} USDT (fee {fee:g} "
        f"{(mine[0].get('fee_currency') or '').upper() if mine else ''}; limit {price}, order {o.get('id')}, "
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
