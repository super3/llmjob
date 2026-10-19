# PRL autoseller

Sells the PRL that mining pays out for USDT on SafeTrade, once an hour, as a Railway cron service.

## What a run does

1. Finds the market to sell on: the first enabled one for SafeTrade's ids for the coin against USDT,
   `pearl` then `prl` (see below). Each is asked for by id, then matched by id or by its base and quote
   currencies, reading every page of SafeTrade's market list (it comes 100 to a page). Then cancels our
   open sell orders there: whatever the last run couldn't sell.
2. Reads the coin's balance under that id. Any left under another id is logged, not sold.
3. Reads the order book and works out a floor price: 3% under the best bid (`MAX_SLIPPAGE`), and never
   under `MIN_PRICE`.
4. Places one limit sell at that floor for as much PRL as the bids at or above it can take. A limit
   order at the floor fills against every bid above it, best first, so it acts like a market order
   that won't sink the price on a thin book. What the book can't take waits for the next run.
5. Logs what sold, the average price and the fee, read from the order's trades.
6. Logs the account and exits. That's an `ACCOUNT` line for people (PRL and USDT, free and in orders,
   and everything sold so far) and a `STATUS` line of JSON for tools that read the logs. "Sold so far"
   is summed from SafeTrade's trade history on the coin's markets, the retired `prlusdt` included, so the
   trader keeps no state of its own. A history it can't read is named on the `ACCOUNT` line. Dry runs and
   runs with nothing to sell log the account too.

**PRL became PEARL on SafeTrade on 18 October 2026.** At 16:53 UTC SafeTrade opened a new `pearl` coin
with its own markets (`pearlusdt` among them, same 2-coin minimum), and at 17:51 it disabled `prl` and its
markets. A trader that only knew `prl` had no market to sell on from then. It now tries `pearl` first,
passes over disabled markets, and still counts the sales made on `prlusdt`. The logs still call the coin PRL,
and the `STATUS` line keeps its `prl` and `prl_locked` keys, since the Salad watcher reads them. Its `coin`
field says which id the run used.

Less than `MIN_VOLUME` (0.5 PRL), or the market's minimum, isn't sold.

Each run logs `run started` first and is stopped after `MAX_RUN_SECONDS` (300). Railway skips a cron run while
the last one is still going, so one hung run used to stop all selling: on 12 October a run started at 21:00,
never finished, and no run happened for the next 15 hours.

## Safety

- **Dry run by default.** It places or cancels nothing until `DRY_RUN=0`. A dry run still makes the
  read-only calls (markets, open orders, balance, order book), so the logs show whether the key and
  the API paths work.
- **Production only.** It trades only when `RAILWAY_ENVIRONMENT_NAME` is `production`. PR preview
  environments may copy the service; there it only logs. Previews also lack the static IP, so the
  SafeTrade whitelist refuses them anyway.
- **The key** should allow trading and reading only, never withdrawals, and be whitelisted to the
  static proxy's IP only.
- **One fixed IP.** With `SAFETRADE_PROXY` set, every SafeTrade call goes through that proxy, so SafeTrade
  always sees the proxy's IP, whatever IP Railway gives the service. Each run starts with the IP its calls
  leave from (`run started: calls go out from …`), and a 401 or 403 names it, so a refused key can be checked
  against the whitelist. The log shows the proxy's host and port, never its login.

## Railway setup

Service `trader` in the `llmjob` project:

| Setting | Value |
|---|---|
| Source | `super3/llmjob`, Root Directory `/trader`, Watch Paths `/trader/**` |
| Networking | SafeTrade calls go through the static proxy in `SAFETRADE_PROXY`; its IP is the only one in SafeTrade's IP whitelist |
| Start and schedule | From `railway.toml` here: `python3 trader.py` at minute 5 of every hour, no restarts |
| Variables | `SAFETRADE_API_KEY`, `SAFETRADE_API_SECRET` (sealed), `SAFETRADE_PROXY` (sealed: `http://user:pass@host:port` from the static proxy provider), `DRY_RUN` (`1` until the dry run looks right, then `0`) |

Optional variables: `MAX_SLIPPAGE` (0.03), `MIN_PRICE` (0), `MIN_VOLUME` (0.5), `MAX_RUN_SECONDS` (300), `MARKET`
(unset: found from the coin's ids), `COIN` (unset: `pearl`, then `prl`; a value here is tried first), `QUOTE` (`usdt`),
`SAFETRADE_BASE` (`https://safe.trade/api/v2`). A disabled market is passed over even when `MARKET` or `COIN` names it.

## The API

From SafeTrade's REST API page (base `https://safe.trade/api/v2`) and its example client
(github.com/safetrade-exchange/example-client):

- **Private calls** are under `/trade`: `GET /trade/account/balances/spot` (`currency`, `balance`,
  `locked`), `GET` and `POST /trade/market/orders`, `GET /trade/market/orders/{id}`, `POST
  /trade/market/orders/{id}/cancel`, `GET /trade/market/trades`. An order is
  `{market, side, amount, price}` as numbers. A price makes it a limit order; there's no `type` field.
  An order comes back with `origin_amount`, `filled_amount` and `state` but no average price, so the
  average and the fee come from its trades. `/peatio/market/orders` answers 404.
- **Public data** is under `/trade/public`: `markets` (the whole list, not paged), `markets/{id}`
  (`amount_precision`, `price_precision`, `min_amount`, `base_unit`, `quote_unit`) and
  `markets/{id}/depth` (`bids` and `asks` as `[price, amount]`). The older `/peatio/public` paths
  answered on the first run and stay as a fallback.
- **Signing:** `X-Auth-Apikey`, `X-Auth-Nonce` (a millisecond timestamp) and `X-Auth-Signature`, the
  hex HMAC-SHA256 of nonce + key under the secret.

These come from the swagger file at `safetrade.com/api/v2/trade/public/swagger.json`. SafeTrade's
Cloudflare blocks the cloud session this was written in, so it was pasted in from a browser.

Tests: `python3 -m unittest discover -s trader`. They fake SafeTrade.
