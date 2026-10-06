# RTX benchmark

Our miner against PeakMiner 2.17.6 and SRBMiner 3.7.1 on rented Vast hosts, one
or two hosts per card. Each host runs all three miners one after another on the
same pool, so rates within a row compare directly. Don't compare across rows:
power limits and cooling differ from host to host.

## How a host is tested

1. Download our CLI and core from the published release (now v0.5.11) on
   GitHub, the same files a user gets. The first 20-series run predates the
   release: it built PR #250 from source, which is the code that shipped as
   v0.5.11. On an RTX 50-series card with driver 580 or newer, the CLI loads
   the release's CUDA 13 core, as a user's rig would, and the hit check runs
   on that core.
2. Hit check: `earn/native/probes/verify-hits.js` runs the core for 90 s and
   recomputes the first 400 hits from scratch the way the pool's verifier would.
   Every one must match. It checks that the answers are right, not the speed.
   It runs on every host but isn't a column in the tables: a host that passes
   shows nothing, and a failure is marked in that row's "Ours" cell.
3. Our miner, PeakMiner and SRBMiner each mine for 5 minutes on
   `us.pearl.herominers.com`. The order alternates between hosts, because
   whoever runs first gets the coolest card.

Rates are in TH/s. For PeakMiner and SRBMiner it's the miner's own figure at
the end of its 5 minutes. For our miner it's the mean of every reading after
the first minute. A single reading covers only about 0.5 s and can be off by
one batch (see the RTX 2070 note). The first 20-series run used the last
reading, so its "Ours" figures are single readings. On a multi-GPU rental,
every miner is pinned to GPU 0 and GPU 1 is checked to stay idle.

"% of best" is our rate divided by the faster of PeakMiner and
SRBMiner on the same host.

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

## RTX 20-series (v0.5.11)

Run on 2026-10-06, 09:15–09:38 UTC, with PR #250 built from source at
`f66f80c`: the code that shipped as v0.5.11. The RTX 2070 was rerun at
10:24 UTC with the v0.5.11 release, averaging our readings.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RTX 2080 Ti | Thailand (170 W) | 95392 | 60.5 | 70.9 | 70.3 | 85% |
| RTX 2080 Ti | Pennsylvania (260 W) | 150735 | 85.7 | 94.2 | 94.2 | 91% |
| RTX 2080 | Colorado (275 W) | 149439 | 70.9 | waiting | 80.3 | 88% |
| RTX 2070 Super | Alberta (215 W) | 31798 | 55.0 | 60.9 | 59.1 | 90% |
| RTX 2070 | South Korea (150 W) | 139007 | 44.5 | 50.1 | 48.9 | 89% |
| RTX 2060 Super | Germany (175 W) | 149900 | 41.1 | 46.5 | 45.0 | 88% |
| RTX 2060 | Australia (190 W, 6 GB) | 152547 | 43.4 | 50.3 | 47.3 | 86% |
| RTX 2060 | South Korea (184 W, 12 GB) | 27568 | 46.3 | 52.9 | 52.3 | 88% |

Every host passed the hit check, and every run had 0 rejected shares.

- **Against the top competitor:** we're at 85–91% on every card. The top
  competitor is PeakMiner everywhere except the 2080, where PeakMiner didn't
  run and SRBMiner is the comparison. On the Pennsylvania 2080 Ti the two
  tie at 94.2. PeakMiner keeps the tensor cores busier: about 92% of peak
  against our 78–82%.
- **RTX 2070:** the first run's last CLI reading was 57.2 TH/s, above the
  card's peak of 51.4 at the clock it ran at. A single reading covers about
  0.5 s, and with two batches in flight one batch can land in the next
  reading. The rerun averages every reading after the first minute: 44.5,
  with single readings from 29 to 59. Shares were never affected.
- **RTX 2080:** PeakMiner wrote nothing to its log and the GPU stayed idle, so
  there's no PeakMiner number yet. No RTX 2080 has been listed on Vast since,
  so the rerun is waiting.
- **SRBMiner** logs an OpenCL error at start on every 20-series card, then
  mines normally on CUDA.

## RTX 30-series (v0.5.11)

Run on 2026-10-06, 10:24–11:10 UTC, with the v0.5.11 release.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RTX 3090 Ti | Greece (450 W) | 152830 | 116.9 | 145.9 | 144.3 | 80% |
| RTX 3090 Ti | Washington (450 W) | 151121 | 119.8 | 149.7 | 153.1 | 78% |
| RTX 3090 | Ukraine (390 W) | 141130 | 103.9 | 129.9 | 131.2 | 79% |
| RTX 3090 | Quebec (300 W) | 152641 | 87.3 | 117.6 | 114.9 | 74% |
| RTX 3080 Ti | Japan (330 W) | 137807 | 97.2 | didn't start | 124.5 | 78% |
| RTX 3080 Ti | Portugal (350 W) | 56596 | 99.7 | 126.7 | 127.3 | 78% |
| RTX 3080 | Kentucky (320 W) | 29108 | 84.9 | 107.6 | 108.1 | 79% |
| RTX 3080 | Washington (280 W) | 25433 | 70.6 | 91.5 | 91.3 | 77% |
| RTX 3070 Ti | Ontario (310 W) | 43435 | 63.5 | 85.8 | 85.6 | 74% |
| RTX 3070 | Quebec (220 W) | 148988 | 59.7 | 75.6 | 75.6 | 79% |
| RTX 3070 | Quebec (180 W) | 152549 | 57.1 | 72.7 | 72.6 | 79% |
| RTX 3060 Ti | Japan (180 W) | 137800 | 50.0 | 63.7 | 63.9 | 78% |
| RTX 3060 Ti | New Zealand (220 W) | 144095 | 51.7 | 65.7 | no rate | 79% |
| RTX 3060 | Poland (170 W) | 149975 | 39.1 | 50.0 | 49.8 | 78% |
| RTX 3060 | Vietnam (170 W) | 138808 | 38.0 | 48.9 | 48.7 | 78% |

Every host passed the hit check, and every run had 0 rejected shares.

- **Against the top competitor:** we're at 74–80% on every card, well behind
  the 20-series (85–91%). PeakMiner and SRBMiner finish within 3% of each
  other on every card, and all three miners run at the same power cap,
  so the gap is ours. Ampere is where our kernel has the most to gain.
- **Our readings swing.** Single readings land 20–35% either side of the
  mean, for example 69.8 to 105.5 around 87.3 on the Quebec 3090. The mean is what the
  card does; one reading on the screen can mislead.
- **RTX 3080 Ti, Japan:** PeakMiner exited at once with code 127 and wrote
  nothing, the same way it failed on the RTX 2080. The comparison there is
  SRBMiner.
- **RTX 3060 Ti, New Zealand:** SRBMiner mined (the card drew 216 W), but its
  rate wasn't in a form the script could read. The script now reads it
  reliably. The comparison there is PeakMiner.
- **RTX 3070 Ti:** one host. The second host never started its container in
  27 minutes, and no other 3070 Ti could be rented.
- **Replaced hosts.** Six hosts couldn't run the test and were swapped for
  the next one by the same rules:
  - 3090 Quebec (16146), 3090 Illinois (142650), 3060 Ti New Zealand
    (142449) and 3060 Thailand (146320): the host couldn't attach the GPU
    to the container ("failed to inject CDI devices").
  - 3060 Poland (48503): the host's Docker had no NVIDIA runtime.
  - 3090 Czechia (24191): the container started but CUDA saw no GPU, so no
    miner could run.
  - Three hosts planned earlier (3090 Ti Vietnam, 3090 Argentina, 3080
    France) were no longer listed when the run started.

## RTX 40-series (v0.5.11)

Next to run. Hosts rechecked 2026-10-06 at 10:48 UTC; seven planned earlier
were no longer listed and were replaced by the same rules.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RTX 4090 | Estonia (450 W) | 56522 | | | | |
| RTX 4090 | Ukraine (450 W) | 138134 | | | | |
| RTX 4090D | Tanzania (425 W) | 70632 | | | | |
| RTX 4090D | South Carolina (425 W) | 142279 | | | | |
| RTX 4080 Super | Japan (320 W) | 36413 | | | | |
| RTX 4080 Super | California (275 W) | 138449 | | | | |
| RTX 4080 | Utah (320 W) | 145428 | | | | |
| RTX 4080 | Nevada (320 W) | 147894 | | | | |
| RTX 4070 Ti Super | Texas (295 W) | 137944 | | | | |
| RTX 4070 Ti Super | Texas (285 W) | 142663 | | | | |
| RTX 4070 Ti | United Kingdom (285 W) | 149163 | | | | |
| RTX 4070 Ti | Delaware (285 W) | 39901 | | | | |
| RTX 4070 Super | California (209 W) | 145255 | | | | |
| RTX 4070 Super | California (220 W) | 153237 | | | | |
| RTX 4070 | Kentucky (200 W) | 34040 | | | | |
| RTX 4060 Ti | Ontario (160 W) | 37799 | | | | |
| RTX 4060 Ti | Brazil (160 W) | 152073 | | | | |
| RTX 4060 | New Zealand (115 W) | 148383 | | | | |
| RTX 4060 | Australia (115 W) | 143986 | | | | |

## RTX 50-series (v0.5.11)

Next to run. Hosts rechecked 2026-10-06 at 10:48 UTC; four planned earlier
were no longer listed and were replaced by the same rules.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RTX 5090 | British Columbia (575 W) | 38389 | | | | |
| RTX 5090 | Romania (575 W) | 137732 | | | | |
| RTX 5080 | Georgia (360 W) | 147973 | | | | |
| RTX 5080 | New Jersey (360 W) | 152066 | | | | |
| RTX 5070 Ti | South Korea (250 W) | 27661 | | | | |
| RTX 5070 Ti | South Korea (300 W) | 18149 | | | | |
| RTX 5070 | Colombia (250 W) | 145974 | | | | |
| RTX 5070 | Kansas (250 W) | 151873 | | | | |
| RTX 5060 Ti | Virginia (150 W) | 151123 | | | | |
| RTX 5060 Ti | Ontario (180 W) | 153080 | | | | |
| RTX 5060 | United States (145 W) | 68005 | | | | |
| RTX 5060 | Virginia (125 W) | 151478 | | | | |

## Notes on the 40/50-series hosts

- **RTX 4070:** the Kentucky host runs at the card's stock 200 W. The Mexico
  host planned earlier (135 W) is no longer listed.
- **RTX 5090D:** no host is listed.
- **4090s we ran before:** those were interruptible rentals of v0.5.9. None is
  listed today as a one-GPU on-demand rental.
- **Laptop GPUs** (3060, 4070 and 4080 laptop) are on Vast but left out.
