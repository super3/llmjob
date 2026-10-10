# Tuning mining on rented Vast boxes

The goal is to earn more per dollar of rent on the boxes we rent on Vast, on-demand and
spot (interruptible). Earnings scale with hashrate, so every gain in TH/s on a box we already
pay for is profit.

## How a box is tuned

Each box runs [`box.sh`](box.sh). It starts one miner per GPU (`PEARL_GPU_INDEX`), each
under its own pool worker name (`<worker>-g<N>`), on the published v0.5.12 release. Every
2 minutes it fetches `tuning/control/<worker>.json` from this branch and restarts only the
GPUs whose settings changed. Everything else keeps mining.

What a control file can set, per GPU or as the box default:

| Setting | What it does |
|---|---|
| `"core": "release"` | The CLI's own pick of its CUDA 12 or CUDA 13 core (the default) |
| `"core": "cu12"` / `"cu13"` | Force one of the release's two cores |
| `"core": "build:<name>"` | A core built on the box from this branch, listed under `"builds"` with a commit and `-D` defines. Built with CUDA 12.8 and run as the cu12 core, or with CUDA 13.3 and run as the cu13 core when its entry has `"cuda": "13"` (needs a 580+ driver; on Blackwell, 12.8 builds run about 3% slower) |
| `"flags"` | Extra CLI flags, e.g. `--mine-mem-clock 0` |
| `"env"` | Extra environment for the miner |

A built core runs only after it passes the hit check on its GPU (400 hits recomputed from
scratch, the way the pool verifies them). Settings whose miner keeps exiting are dropped and
the GPU goes back to the release default.

## Rules for a test

1. **Change one or two GPUs at a time.** The box's other GPUs are the control group: same
   host, same power and cooling, measured over the same minutes.
2. **Measure after warm-up.** Ignore each miner's first 2 minutes, then compare at least
   10 minutes of readings from the test GPUs against the control GPUs.
3. **A gain counts when it holds.** At least +1% over the control GPUs across the whole
   window, with no rejected or invalid shares at the pool. Then roll it out to the rest of
   the box and record it in [`RESULTS.md`](RESULTS.md).
4. **Single-GPU boxes have no control group,** so they only take settings already verified
   on the same card elsewhere, compared before and after over at least 10 minutes each.
5. **Code changes go on this branch** under `earn/native`, are built on the box through
   `"builds"`, and must pass the hit check and `npm test` in `earn/`.
6. **Write down what didn't work too,** so no one tries it again.

## Files

- [`box.sh`](box.sh): the script each rented box runs.
- `control/<worker>.json`: each box's live settings. These steer running boxes; they are an
  experiment log, not part of the miner.
- [`RESULTS.md`](RESULTS.md): what was tried on which card, and the result.
- [`sell_prl.py`](sell_prl.py): sells mined PRL on SafeTrade as it arrives. See below.

## Selling PRL as it arrives

`sell_prl.py` checks your SafeTrade PRL balance every minute and sells what is there for USDT,
so each payout is turned into dollars at that moment's price instead of riding the PRL price.
It uses only the Python standard library.

It will not sell into a bad book:

- The lowest price it accepts is 1% under the best bid (`--max-slip`). Bids below that are left
  alone, and the rest of the balance waits for the next round.
- It sells nothing when the best bid is more than 5% under the last trade (`--max-gap`).
- `--floor 1.20` stops all sales under $1.20.
- An order that hasn't filled after 2 minutes is cancelled and tried again next round.

Each sale is added to `sales.csv`. The order in flight is saved in `sell_prl_state.json`, so a
restarted run finishes it. Orders you place yourself on SafeTrade are never touched.

To set it up:

1. On SafeTrade, check that its PRL is the coin you mine. Then point your pool payouts at your
   SafeTrade PRL deposit address, or send PRL there yourself.
2. Make an API key with trading rights only. Leave withdrawals off, so the key can't move funds
   out even if it leaks.
3. Run it on your own computer or a server. SafeTrade sits behind Cloudflare, which refuses many
   cloud addresses, including the ones Claude Code sessions run in.

```sh
export SAFETRADE_API_KEY=...  SAFETRADE_API_SECRET=...
python3 sell_prl.py --check   # read market, book, ticker and balance; print them; place nothing
python3 sell_prl.py           # dry run: logs what it would sell, places nothing
python3 sell_prl.py --live    # sells for real
```

`--check` shows what the script understood from the API. If a balance or price reads as `None` or
0 when it shouldn't, stop and fix the field names before going live. Use `--market prlusdc` to sell
for USDC. Tests: `python3 test_sell_prl.py` (runs against a fake SafeTrade server).
