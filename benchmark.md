# RTX benchmark

Our miner against PeakMiner 2.17.6 and SRBMiner 3.7.1 on rented Vast hosts, one
or two hosts per card. Each host runs all three miners one after another on the
same pool, so rates within a row compare directly. Don't compare across rows:
power limits and cooling differ from host to host.

## How a host is tested

1. Build our core from source for the card, and run the CLI from the same
   source, so the run is exactly the commit under test.
2. Hit check: `earn/native/probes/verify-hits.js` runs the core for 90 s and
   recomputes the first 400 hits from scratch the way the pool's verifier would.
   Every one must match. It checks that the answers are right, not the speed.
   It runs on every host but isn't a column in the tables: a host that passes
   shows nothing, and a failure is marked in that row's "Ours" cell.
3. Our miner, PeakMiner and SRBMiner each mine for 5 minutes on
   `us.pearl.herominers.com`. The order alternates between hosts, because
   whoever runs first gets the coolest card.

Rates are in TH/s: each miner's own reading at the end of its 5 minutes. For
our miner that last reading covers only about 0.5 s, so it can be off by one
batch (see the RTX 2070 note). Later runs will average every reading after
the first minute instead. On a multi-GPU rental, every miner is pinned to
GPU 0 and GPU 1 is checked to stay idle.

## How hosts are picked

- The driver supports CUDA 12.8, which our miner needs.
- Reliability is 90% or higher.
- The power limit is at least 80% of the card's stock limit, so each card is
  measured close to how it's sold.
- Vast's performance score is at least half the median for that card. This
  screens out fake hosts, like the Texas "4090s" that ran at 38 TH/s.
- Not in mainland China. Each box downloads from GitHub and mines on a US pool.
- Cheapest first. A host we've already tested wins when it's listed. The second
  host for a card comes from a different operator.
- One GPU per rental where possible.

The 20-series hosts were picked before the power and performance-score rules
existed, so the Thailand 2080 Ti (170 W of 250 W) is on the list.

## RTX 20-series (PR #250 build)

Run on 2026-10-06, 09:15–09:38 UTC, at commit `f66f80c`.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner |
|---|---|---|---|---|---|
| RTX 2080 Ti | Thailand (170 W) | 95392 | 60.5 | 70.9 | 70.3 |
| RTX 2080 Ti | Pennsylvania (260 W) | 150735 | 85.7 | 94.2 | 94.2 |
| RTX 2080 | Colorado (275 W) | 149439 | 70.9 | didn't run | 80.3 |
| RTX 2070 Super | Alberta (215 W) | 31798 | 55.0 | 60.9 | 59.1 |
| RTX 2070 | South Korea (150 W) | 139007 | ~43 ⚠️ | 47.8 | 46.5 |
| RTX 2060 Super | Germany (175 W) | 149900 | 41.1 | 46.5 | 45.0 |
| RTX 2060 | Australia (190 W, 6 GB) | 152547 | 43.4 | 50.3 | 47.3 |
| RTX 2060 | South Korea (184 W, 12 GB) | 27568 | 46.3 | 52.9 | 52.3 |

Every host passed the hit check, and every run had 0 rejected shares.

- **Against PeakMiner:** we're at 85–91% on every card. PeakMiner keeps the
  tensor cores busier: about 92% of peak against our 78–82%.
- **⚠️ RTX 2070:** the CLI's last reading was 57.2 TH/s, which can't be
  right: at the 1395 MHz it ran at, the card's peak is 51.4. The hit check's
  own work count puts it at about 43 (55 operand redraws in 90 s), in line
  with the other cards. The cause is in the CLI's reading. It covers only
  about 0.5 s, and with two batches in flight a batch can be counted in the
  next window. On a slow card that one batch is about 40% of a reading. The
  long-run average is exact and shares are unaffected.
- **RTX 2080:** PeakMiner wrote nothing to its log and the GPU stayed idle, so
  there's no PeakMiner number for this card.
- **SRBMiner** logs an OpenCL error at start on every 20-series card, then
  mines normally on CUDA.

## RTX 30-series (PR #250 build)

Not run yet. Hosts picked 2026-10-06; none has been tested before.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner |
|---|---|---|---|---|---|
| RTX 3060 | Thailand (170 W) | 146320 | | | |
| RTX 3060 | Vietnam (170 W) | 138808 | | | |
| RTX 3060 Ti | Japan (180 W) | 137800 | | | |
| RTX 3060 Ti | New Zealand (220 W) | 142449 | | | |
| RTX 3070 | Quebec (220 W) | 148988 | | | |
| RTX 3070 | Quebec (180 W) | 152549 | | | |
| RTX 3070 Ti | Ontario (310 W) | 43435 | | | |
| RTX 3070 Ti | Pennsylvania (310 W) | 136798 | | | |
| RTX 3080 | Kentucky (320 W) | 29108 | | | |
| RTX 3080 | France (320 W) | 153103 | | | |
| RTX 3080 Ti | Japan (330 W) | 137807 | | | |
| RTX 3080 Ti | Portugal (350 W) | 56596 | | | |
| RTX 3090 | Quebec (350 W) | 16146 | | | |
| RTX 3090 | Argentina (280 W) | 54987 | | | |
| RTX 3090 Ti | Vietnam (450 W) | 27934 | | | |
| RTX 3090 Ti | Greece (450 W) | 152830 | | | |

## RTX 40-series (PR #250 build)

Not run yet. Hosts picked 2026-10-06; none has been tested before.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner |
|---|---|---|---|---|---|
| RTX 4060 | New Zealand (115 W) | 148383 | | | |
| RTX 4060 | Australia (115 W) | 143986 | | | |
| RTX 4060 Ti | Ontario (160 W) | 37799 | | | |
| RTX 4060 Ti | Brazil (160 W) | 152073 | | | |
| RTX 4070 | Mexico (135 W) | 136612 | | | |
| RTX 4070 Super | Delaware (220 W) | 142006 | | | |
| RTX 4070 Super | California (220 W) | 153237 | | | |
| RTX 4070 Ti | North Macedonia (285 W) | 150347 | | | |
| RTX 4070 Ti | Delaware (285 W) | 39901 | | | |
| RTX 4070 Ti Super | Ontario (285 W) | 29907 | | | |
| RTX 4070 Ti Super | Romania (250 W) | 150551 | | | |
| RTX 4080 | Utah (320 W) | 150424 | | | |
| RTX 4080 | Nevada (320 W) | 147894 | | | |
| RTX 4080 Super | Japan (320 W) | 36413 | | | |
| RTX 4080 Super | California (275 W) | 138449 | | | |
| RTX 4090 | Estonia (450 W) | 56522 | | | |
| RTX 4090 | Poland (400 W) | 151409 | | | |
| RTX 4090D | Tanzania (425 W) | 70632 | | | |
| RTX 4090D | South Carolina (425 W) | 142279 | | | |

## RTX 50-series (PR #250 build)

Not run yet. Hosts picked 2026-10-06; none has been tested before.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner |
|---|---|---|---|---|---|
| RTX 5060 | United States (145 W) | 68005 | | | |
| RTX 5060 | Virginia (125 W) | 151478 | | | |
| RTX 5060 Ti | Virginia (150 W) | 151123 | | | |
| RTX 5060 Ti | France (180 W) | 146674 | | | |
| RTX 5070 | New York (200 W) | 142292 | | | |
| RTX 5070 | Kansas (250 W) | 151873 | | | |
| RTX 5070 Ti | South Korea (250 W) | 39891 | | | |
| RTX 5070 Ti | South Korea (300 W) | 18149 | | | |
| RTX 5080 | Georgia (360 W) | 147973 | | | |
| RTX 5080 | New Jersey (360 W) | 152066 | | | |
| RTX 5090 | British Columbia (575 W) | 38389 | | | |
| RTX 5090 | Romania (600 W) | 9105 | | | |
| RTX 5090D | Taiwan (575 W) | 151179 | | | |

## Notes on the 30/40/50-series hosts

- **RTX 4070:** no host meets the rules. Mexico is the only one with a new
  enough driver, and it's held to 135 W of a stock 200 W, so it will read low.
- **RTX 5090D:** only one host is listed.
- **4090s we ran before:** those were interruptible rentals of v0.5.9. None is
  listed today as a one-GPU on-demand rental.
- **Laptop GPUs** (3060, 4070 and 4080 laptop) are on Vast but left out.
- **Cost:** the 48 hosts together are about $9.60/hr. At about 30 minutes each,
  a full run is about $5.
