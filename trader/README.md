# PRL autoseller

Sells the PRL that mining pays out for USDT on SafeTrade, once an hour, as a Railway cron service.

## What a run does

1. Finds the PRL/USDT market: by id, or by its base and quote currencies, reading every page of
   SafeTrade's market list (it comes 100 to a page). Then cancels our open sell orders there:
   whatever the last run couldn't sell.
2. Reads the PRL balance.
3. Reads the order book and works out a floor price: 3% under the best bid (`MAX_SLIPPAGE`), and never
   under `MIN_PRICE`.
4. Places one limit sell at that floor for as much PRL as the bids at or above it can take. A limit
   order at the floor fills against every bid above it, best first, so it acts like a market order
   that won't sink the price on a thin book. What the book can't take waits for the next run.
5. Logs what sold and at what average price, then exits.

Less than `MIN_VOLUME` (0.5 PRL), or the market's minimum, isn't sold.

## Safety

- **Dry run by default.** It places or cancels nothing until `DRY_RUN=0`. A dry run still makes the
  read-only calls (markets, open orders, balance, order book), so the logs show whether the key and
  the API paths work.
- **Production only.** It trades only when `RAILWAY_ENVIRONMENT_NAME` is `production`. PR preview
  environments may copy the service; there it only logs. Previews also lack the static IP, so the
  SafeTrade whitelist refuses them anyway.
- **The key** should allow trading and reading only, never withdrawals, and be whitelisted to the
  service's static IPs.

## Railway setup

Service `trader` in the `llmjob` project:

| Setting | Value |
|---|---|
| Source | `super3/llmjob`, Root Directory `/trader`, Watch Paths `/trader/**` |
| Networking | Static outbound IPs on; all of them in SafeTrade's IP whitelist |
| Start and schedule | From `railway.toml` here: `python3 trader.py` at minute 5 of every hour, no restarts |
| Variables | `SAFETRADE_API_KEY`, `SAFETRADE_API_SECRET` (sealed), `DRY_RUN` (`1` until the dry run looks right, then `0`) |

Optional variables: `MAX_SLIPPAGE` (0.03), `MIN_PRICE` (0), `MIN_VOLUME` (0.5), `MARKET` (`prlusdt`),
`COIN` (`prl`), `QUOTE` (`usdt`), `SAFETRADE_BASE` (`https://safe.trade/api/v2/peatio`).

## The API

SafeTrade runs OpenDAX/Peatio. Private calls send `X-Auth-Apikey`, `X-Auth-Nonce` (a millisecond
timestamp) and `X-Auth-Signature`, the hex HMAC-SHA256 of nonce + key under the secret. Paths used:
`GET /public/markets`, `GET /public/markets/prlusdt/order-book`, `GET /account/balances`,
`GET /market/orders?state=wait`, `POST /market/orders`, `POST /market/orders/{id}/cancel`,
`GET /market/orders/{id}`. These come from Openware's Peatio docs; SafeTrade's Cloudflare blocks the
cloud session they were written in, so the first dry run on Railway is the real check.

Tests: `python3 -m unittest discover -s trader`. They fake SafeTrade.
