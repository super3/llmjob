#!/usr/bin/env python3
# Sells mined PRL for USDT on SafeTrade, once an hour (a Railway cron service; see README.md).
# Each run: cancel our open orders on the market (last run's unfilled rest), read the PRL balance, then sell
# what the order book can take without going more than MAX_SLIPPAGE below the best bid, and never under
# MIN_PRICE. It's one limit order at that floor price, so it fills against every bid at or above it, best
# first. Anything the book can't take now waits for the next run.
# It trades only when DRY_RUN=0 and it runs in Railway's production environment; anything else just logs
# what it would do. Standard library only.
import hashlib, hmac, json, math, os, re, signal, sys, time, urllib.error, urllib.parse, urllib.request
from decimal import Decimal, ROUND_DOWN

# Paths from SafeTrade's REST API page (safe.trade/api/v2): private calls under /trade (balances at
# /trade/account/balances/spot; orders at /trade/market/orders, /{id} and /{id}/cancel), public data under
# /trade/public. /peatio/market/orders answers 404 there. The old /peatio/public paths answered on the first run,
# so they stay as a fallback for market data.
BASE = os.environ.get("SAFETRADE_BASE", "https://safe.trade/api/v2").rstrip("/")
PUBS, PRIV = ["/trade/public", "/peatio/public"], "/trade"
BALANCE_PATHS = ["/trade/account/balances/spot", "/trade/account/balances"]
QUOTE = os.environ.get("QUOTE", "usdt")


def coin_ids(first=""):
    """SafeTrade's ids for the coin, newest first. It listed Pearl as PRL until 18 October 2026, when it opened a new
    PEARL coin with its own markets (pearlusdt) and disabled PRL's (prlusdt). A run sells on the first id with an
    enabled market and counts past sales on all of them. COIN, if set, goes first."""
    return list(dict.fromkeys(c for c in (str(first).strip().lower(), "pearl", "prl") if c))


COINS = coin_ids(os.environ.get("COIN", ""))
MARKET = os.environ.get("MARKET", "")   # SafeTrade's id for the market, if set; otherwise found from COINS and QUOTE
COIN_ID = COINS[0]   # the coin's id on the market a run sells on, once found
# What the logs call the coin, and the STATUS line's key for it (lower case), whatever SafeTrade's id: the Salad
# watcher reads that line.
NAME = "PRL"
KEY = os.environ.get("SAFETRADE_API_KEY", "")
SECRET = os.environ.get("SAFETRADE_API_SECRET", "")
MAX_SLIPPAGE = float(os.environ.get("MAX_SLIPPAGE", "0.03"))   # sell down to 3% under the best bid
MIN_PRICE = float(os.environ.get("MIN_PRICE", "0"))           # never sell under this (USDT per PRL)
MIN_VOLUME = float(os.environ.get("MIN_VOLUME", "0.5"))        # don't bother with less PRL than this
# A run that hangs keeps the cron service busy, and Railway skips every later run until it ends, so a run
# is stopped after this long.
MAX_RUN_SECONDS = int(os.environ.get("MAX_RUN_SECONDS", "300"))
PAGES = 50   # trade history pages read for the lifetime totals, 100 trades each; past that, the ACCOUNT line says so
# SafeTrade takes the key only from the IPs on its whitelist. SAFETRADE_PROXY (http://user:pass@host:port) sends
# every call through a proxy with a static IP, so the whitelist needs just that address, whatever IP Railway uses.
PROXY = os.environ.get("SAFETRADE_PROXY", "")
OUT_IP = "?"   # the IP SafeTrade sees, read at the start of each run


def now_ms():
    return str(int(time.time() * 1000))


def urlopen(req, timeout):
    if PROXY:
        proxies = urllib.request.ProxyHandler({"http": PROXY, "https": PROXY})
        return urllib.request.build_opener(proxies).open(req, timeout=timeout)
    return urllib.request.urlopen(req, timeout=timeout)


def proxy_name():
    """The proxy's host and port, without its login. A password with '#' or '?' in it makes urlsplit end the address
    there (urllib's own proxy parsing is fine), so its "host" is the user name and its port part of the password, and
    the ValueError's text holds that part: then nothing of the address is shown."""
    u = urllib.parse.urlsplit(PROXY)
    try:
        port = u.port
    except ValueError:
        return "(an address this log can't show)"
    return f"{u.hostname}:{port}" if port else str(u.hostname)


def out_ip():
    """The IP our calls leave from (through the proxy, if there is one), so a 403 can be checked against the whitelist."""
    try:
        req = urllib.request.Request("https://api.ipify.org", headers={"User-Agent": "llmjob-trader/1.0"})
        with urlopen(req, timeout=15) as r:
            return r.read().decode().strip()[:45]
    except Exception as e:
        return f"unknown ({str(e)[:80]})"


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
        with urlopen(req, timeout=30) as r:
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


class TooLong(BaseException):
    """The run went past MAX_RUN_SECONDS. Not an Exception, so http()'s and out_ip()'s `except Exception` can't catch
    it: the alarm usually fires inside a network call, and a caught one let the run carry on with no limit at all,
    since the alarm fires once. The 12 October hang, 15 hours with no sale, was inside urlopen."""


def need(status, body, what):
    if not 200 <= status < 300:
        hint = f" (calls go out from {OUT_IP}: is it on the key's IP whitelist?)" if status in (401, 403) else ""
        raise Fail(f"{what} failed: HTTP {status} {json.dumps(body)[:300]}{hint}")
    return body


def floor_to(x, places):
    # Decimal from the float's shortest repr: 4.12 stays 4.12 (x * 100 is 411.99999...), and 4.12999999999995 rounds
    # down to 4.1299, not up to 4.13, which would ask to sell more than the balance.
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-places), rounding=ROUND_DOWN))


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


def enabled(m):
    """SafeTrade disables a market it retires, as it did prlusdt. The old public API gives no state at all."""
    return str(m.get("state") or "enabled").lower() == "enabled"


def base_of(m):
    return str(m.get("base_unit") or m.get("base_currency") or "").lower()


def market_info():
    """The market to sell on: the first enabled one in our order, which is MARKET if set, then each of COINS against
    QUOTE. Each pub is asked by id first, then its market list is read (100 to a page, every page), matched on the
    id or on the base and quote currencies, since the id's spelling can differ. Sets MARKET and COIN_ID."""
    global MARKET, COIN_ID
    ids = [i.lower() for i in dict.fromkeys(([MARKET] if MARKET else []) + [c + QUOTE for c in COINS])]

    def rank(m):
        """Where a market comes in our order, or None if it isn't one of ours."""
        mid = str(m.get("id", "")).lower()
        quote = str(m.get("quote_unit") or m.get("quote_currency") or "").lower()
        if mid in ids:
            return ids.index(mid)
        if base_of(m) in COINS and quote == QUOTE:
            return len(ids) + COINS.index(base_of(m))
        return None

    def use(pub, m):
        global MARKET, COIN_ID
        MARKET = str(m["id"])
        COIN_ID = base_of(m) or next((c for c in COINS if MARKET.lower() == c + QUOTE), COINS[0])
        return pub, market_fields(m)

    seen, off = [], set()
    for pub in PUBS:
        for mid in ids:
            s, m = http("GET", f"{pub}/markets/{mid}")
            if 200 <= s < 300 and isinstance(m, dict) and m.get("id") and rank(m) is not None:
                if enabled(m):
                    return use(pub, m)
                off.add(str(m["id"]))
        found = []
        for page in range(1, 21):
            s, j = http("GET", pub + "/markets", {"limit": 100, "page": page})
            if not 200 <= s < 300 or not isinstance(j, list) or not j:
                break
            if page > 1 and j[0].get("id") == top:
                break   # the trade API returns the whole list on every page
            top = j[0].get("id")
            for m in j:
                mid = str(m.get("id", ""))
                seen.append(mid)
                r = rank(m)
                if r is not None and enabled(m):
                    found.append((r, len(found), m))
                elif r is not None:
                    off.add(mid)
            if len(j) < 100:
                break
        if found:
            return use(pub, min(found)[2])
    near = sorted({i for i in seen if any(c in i.lower() for c in COINS)})
    raise Fail(f"no enabled {NAME}/{QUOTE.upper()} market among {len(set(seen))} listed (SafeTrade ids tried: "
               f"{', '.join(COINS)}); ids with them: {near or 'none'}" + (f"; disabled: {sorted(off)}" if off else ""))


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
        return None, f"only {vol} {NAME} to sell (balance {balance}, bids down to {price} take {room:.4f})"
    return (vol, price), None


def first(d, *keys):
    """The first of these fields that's set: Peatio and SafeTrade's trade API name some differently."""
    return next((d[k] for k in keys if d.get(k) not in (None, "")), None)


def balances():
    """{currency: (available, locked)} for the spot account."""
    tried = []
    for path in BALANCE_PATHS:
        s, bal = http("GET", path, auth=True)
        if 200 <= s < 300 and isinstance(bal, (list, dict)):
            rows = bal if isinstance(bal, list) else bal.get("balances") or bal.get("data") or [bal]
            return {str(b.get("currency", "")).lower(): (float(first(b, "balance", "available", "free") or 0),
                                                         float(b.get("locked") or 0))
                    for b in rows if isinstance(b, dict)}
        tried.append(f"{path} HTTP {s}")
    raise Fail("balances failed: " + "; ".join(tried))


def lifetime():
    """Every sell of the coin so far, summed: (PRL sold, USDT received, fees, trades, what couldn't be read). From
    SafeTrade's trade history, so it needs no state of its own.

    The history is read whole and each trade is kept by its own market: the one a run sells on, or any of COINS
    against QUOTE (a trade without a market counts as ours). It can't be asked for by market, because Cloudflare in
    front of SafeTrade rewrites "prl" to "PEARL" anywhere in a URL, so a request for prlusdt's trades reaches
    SafeTrade as PEARLusdt, which isn't a market. After the rename SafeTrade's public trade lists show PRL's old trades
    under the pearl markets; whichever id a trade carries, it's counted. Each trade counts once, by its id."""
    ours = {MARKET.lower()} | {c + QUOTE for c in COINS}
    prl = usdt = fees = 0.0
    n, counted, unread, top, full = 0, set(), [], None, False
    for page in range(1, PAGES + 1):
        s, j = http("GET", PRIV + "/market/trades", {"side": "sell", "page": page, "limit": 100}, auth=True)
        if not 200 <= s < 300:
            unread.append(f"trade history page {page}: HTTP {s}")
            break
        if page > 1 and isinstance(j, list) and j and j[0].get("id") == top:
            # A full page then the same page again: SafeTrade ignored `page`, so the rest can't be read this way.
            if full:
                unread.append(f"trade history past {100 * (page - 1)} trades (page {page} repeats page {page - 1})")
            break
        if not isinstance(j, list) or not j:
            break
        top = j[0].get("id")
        for t in j:
            key = t.get("id") if t.get("id") is not None else json.dumps(t, sort_keys=True)
            if t.get("side", "sell") != "sell" or str(t.get("market") or MARKET).lower() not in ours or key in counted:
                continue
            counted.add(key)
            prl += float(t.get("amount") or 0)
            usdt += float(t.get("total") or 0)
            fees += float(t.get("fee") or 0)
            n += 1
        full = len(j) >= 100
        if not full:
            break
    else:
        unread.append(f"trade history past {100 * PAGES} trades (read {PAGES} pages)")
    return prl, usdt, fees, n, unread


def status(last=None):
    """One readable line and one STATUS line of JSON with the account after this run: what's on the exchange and
    everything sold so far. The JSON line is for tools that read the logs."""
    b = balances()
    c, q = b.get(COIN_ID, (0.0, 0.0)), b.get(QUOTE, (0.0, 0.0))
    prl, usdt, fees, n, unread = lifetime()
    log(f"ACCOUNT {c[0]:g} {NAME} free + {c[1]:g} in orders · {q[0]:.2f} {QUOTE.upper()} free + {q[1]:.2f} "
        f"in orders · sold so far {prl:g} {NAME} for {usdt:.2f} {QUOTE.upper()}"
        + (f" (average {usdt / prl:.4f}, fees {fees:.4f})" if prl else "")
        + (f" · not counted: {'; '.join(unread)}" if unread else ""))
    key = NAME.lower()
    print("STATUS " + json.dumps({"t": int(time.time()), "market": MARKET, "coin": COIN_ID, key: c[0], key + "_locked": c[1],
                                  QUOTE: q[0], QUOTE + "_locked": q[1], "sold": prl, "received": usdt, "fees": fees,
                                  "trades": n, "last": last}), flush=True)


def run(live):
    pub, info = market_info()

    # Last run's unfilled rest: cancel it, so its PRL counts in the balance again.
    s, orders = http("GET", PRIV + "/market/orders", {"market": MARKET, "state": "wait", "side": "sell"}, auth=True)
    need(s, orders, "open orders")
    if orders is not None and not isinstance(orders, list):
        raise Fail(f"open orders: expected a list, got {json.dumps(orders)[:200]}")
    for o in orders or []:
        # Only what is still open: a finished order can't be cancelled, and asking fails the run.
        if o.get("state", "wait") != "wait":
            continue
        if o.get("side") == "sell" and str(o.get("market", MARKET)).lower() == MARKET.lower():
            if live:
                need(*http("POST", f"{PRIV}/market/orders/{o['id']}/cancel", {}, auth=True), f"cancel order {o['id']}")
            rest = float(o.get("origin_amount") or 0) - float(o.get("filled_amount") or 0)
            log(f"{'cancelled' if live else 'would cancel'} open sell {o['id']}: {rest:g} at {o.get('price')}")
    if orders and live:
        time.sleep(2)

    held = balances()
    balance = held.get(COIN_ID, (0.0, 0.0))[0]
    for c in COINS:
        left = sum(held.get(c, (0.0, 0.0)))
        if c != COIN_ID and left > 0:
            log(f"{left:g} more {NAME} is held as {c!r}, which {MARKET} doesn't trade; it isn't sold")

    bids = order_book(pub)

    what, why = plan(balance, bids, info)
    if what is None:
        log(f"nothing to sell: {why}")
        return status()
    vol, price = what
    if not live:
        log(f"DRY RUN: would sell {vol} {NAME} with a limit at {price} (best bid {max(p for p, _ in bids)}, "
            f"balance {balance})")
        return status()
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
    log(f"SOLD {done} of {vol} {NAME} at {avg} USDT average for {got:.2f} USDT (fee {fee:g} "
        f"{(mine[0].get('fee_currency') or '').upper() if mine else ''}; limit {price}, order {o.get('id')}, "
        f"state {o.get('state')}); the rest waits for the next run")
    status({"order": o.get("id"), "sold": done, "of": vol, "received": round(got, 6), "fee": fee,
            "average": round(got / amt, 6) if amt else None, "limit": price})


def main():
    env = os.environ.get("RAILWAY_ENVIRONMENT_NAME")
    live = os.environ.get("DRY_RUN", "1") == "0" and env == "production"
    if os.environ.get("DRY_RUN", "1") == "0" and not live:
        log(f"DRY_RUN=0 but this is environment {env!r}, not production: dry run only")
    if not KEY or not SECRET:
        log("SAFETRADE_API_KEY and SAFETRADE_API_SECRET aren't set; nothing to do")
        # A live service without its key sells nothing; failing the run makes Railway show it.
        return 1 if live else 0

    def too_long(*_):
        raise TooLong()
    signal.signal(signal.SIGALRM, too_long)
    # Before the first call: out_ip() goes through the proxy too, and can hang like any other call.
    signal.alarm(MAX_RUN_SECONDS)
    global OUT_IP
    try:
        OUT_IP = out_ip()
        log(f"run started: calls go out from {OUT_IP}" + (f" through the proxy at {proxy_name()}" if PROXY else ""))
        run(live)
    except Fail as e:
        log(f"FAILED: {e}")
        return 1
    except TooLong:
        log(f"FAILED: run took over {MAX_RUN_SECONDS} s; stopped so the next hourly run isn't skipped")
        return 1
    except Exception as e:
        # An answer of a shape nothing expected (a 2xx order without an id, say). Logged in one line, any login in a
        # URL masked, instead of a traceback.
        log(f"FAILED: unexpected {type(e).__name__}: {re.sub(r'//[^/@ ]+@', '//***@', str(e))[:200]}")
        return 1
    finally:
        signal.alarm(0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
