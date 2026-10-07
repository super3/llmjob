# RTX benchmark

Our miner against PeakMiner 2.17.6 and SRBMiner 3.7.1 on rented Vast hosts, one
or two hosts per card. Each host runs all three miners one after another on the
same pool, so rates within a row compare directly. Don't compare across rows:
power limits and cooling differ from host to host.

## Summary

How close our miner gets to the faster of PeakMiner and SRBMiner on the same
host:

- RTX 20-series: 85–91%.
- RTX 30-series: 74–80%. Ampere is where we're furthest behind.
- RTX 40-series: 97–101%.
- RTX 50-series: 90–102%. The 5090 and 5080 are close to even; the 5070 Ti,
  5070, 5060 Ti and 5060 trail by 5–10%.
- A100-class (sm_80, PR #253, not in a release yet): 87–89%.

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
every miner is pinned to GPU 0 and checked to use only that card.

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

Run on 2026-10-06, 11:12–12:10 UTC, with the v0.5.11 release.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RTX 4090 | Australia (450 W) | 17334 | 314.9 | 314.7 | 314.2 | 100% |
| RTX 4090 | British Columbia (370 W) | 152545 | 289.3 | 287.5 | 289.6 | 100% |
| RTX 4090D | South Carolina (425 W) | 142279 | 272.2 | 264.1 | 271.1 | 100% |
| RTX 4080 Super | Florida (320 W) | 142085 | 196.5 | didn't start | 196.3 | 100% |
| RTX 4080 Super | California (275 W) | 138449 | 193.4 | 192.9 | 192.6 | 100% |
| RTX 4080 | Taiwan (280 W) | 149135 | 189.6 | 187.4 | 188.1 | 101% |
| RTX 4080 | Nevada (320 W) | 147894 | 194.7 | 197.4 | 197.1 | 99% |
| RTX 4070 Ti Super | Texas (285 W) | 142663 | 171.6 | 171.9 | 171.7 | 100% |
| RTX 4070 Ti Super | United States (285 W) | 43741 | 167.4 | 169.5 | 168.8 | 99% |
| RTX 4070 Ti | United Kingdom (285 W) | 149163 | 159.8 | 160.6 | 159.7 | 100% |
| RTX 4070 Ti | Delaware (285 W) | 39901 | 156.1 | 157.5 | 156.8 | 99% |
| RTX 4070 Super | California (209 W) | 145255 | 138.5 | 139.6 | 139.3 | 99% |
| RTX 4070 Super | California (220 W) | 153237 | 139.9 | 139.0 | 141.3 | 99% |
| RTX 4070 | New York (200 W) | 19053 | 122.0 | 120.9 | 119.4 | 101% |
| RTX 4060 Ti | Ontario (160 W) | 37799 | 86.3 | 86.1 | 86.6 | 100% |
| RTX 4060 Ti | Brazil (160 W) | 152073 | 89.1 | 88.3 | 89.0 | 100% |
| RTX 4060 | New Zealand (115 W) | 148383 | 59.9 | 60.3 | 58.7 | 99% |
| RTX 4060 | Australia (120 W) | 143986 | 46.5 | 47.7 | 47.5 | 97% |

Every host passed the hit check, and every run had 0 rejected shares.

- **Against the top competitor:** we're at 97–101% on every card. All three
  miners finish within about 3% of each other on Ada, so there's little left
  to gain here.
- **RTX 4080 Super, Florida:** PeakMiner exited at once with code 127 and
  wrote nothing (see the 50-series notes). The comparison there is SRBMiner.
  It's a 2-GPU rental. Every miner was pinned to GPU 0, and each one's rate
  matches a single card.
- **RTX 4090D:** one host. The Tanzania host (70632) never started its
  container, in two tries of 16 and 30 minutes, and no other 4090D is listed.

## RTX 50-series (v0.5.11)

Run in the same window as the 40-series.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RTX 5090 | South Korea (500 W) | 145183 | 384.5 | didn't start | 378.0 | 102% |
| RTX 5090 | India (500 W) | 151861 | 369.0 | driver too old | 375.3 | 98% |
| RTX 5080 | New Jersey (360 W) | 152066 | 223.5 | 227.0 | 229.0 | 98% |
| RTX 5080 | Poland (360 W) | 153231 | 223.6 | 227.8 | 228.3 | 98% |
| RTX 5070 Ti | South Korea (250 W) | 27661 | 89.5 | 93.6 | 94.6 | 95% |
| RTX 5070 Ti | South Korea (250 W) | 28852 | 89.2 | 93.6 | 94.5 | 94% |
| RTX 5070 | Colombia (250 W) | 145974 | 125.7 | 132.3 | 133.5 | 94% |
| RTX 5070 | Poland (250 W) | 31379 | 126.1 | 132.7 | 134.2 | 94% |
| RTX 5060 Ti | Virginia (150 W) | 151123 | 88.0 | 94.5 | 94.3 | 93% |
| RTX 5060 Ti | Ontario (180 W) | 153080 | 87.6 | 95.3 | 95.7 | 92% |
| RTX 5060 | Virginia (125 W) | 151478 | 70.7 | 76.8 | 77.1 | 92% |
| RTX 5060 | India (145 W) | 119163 | 68.8 | 76.4 | didn't mine | 90% |

Every host passed the hit check, and every run had 0 rejected shares.

- **Against the top competitor:** 98–102% on the 5090 and 5080, 94–95% on the
  5070 Ti and 5070, and 90–93% on the 5060 Ti and 5060. On the power-capped
  cards below the 5080, our miner ran 60–185 MHz slower than the other two at
  the same power. It uses more power per clock, so the card clocks down to
  stay under its cap. That's where the gap on the smaller cards comes from.
- **Which core we load:** on a 50-series card with driver 580 or newer, the
  CLI loads the CUDA 13 core. The India 5090 has driver 570, so it loaded the
  CUDA 12.8 core. The South Korea 5090, on CUDA 13, ran 4% faster than the
  India 5090 at the same power and clock, while SRBMiner scored within 1% on
  both. That points to the CUDA 13 core being faster on Blackwell, but it's
  two different hosts.
  - The first batch of boxes printed the core line before our miner started,
    so it doesn't name a core. Every one of those hosts has driver 580 or
    newer, so the CLI's rule picks CUDA 13 there, and their hit checks ran on
    the CUDA 13 core.
- **PeakMiner:**
  - South Korea 5090: exited at once with code 127 and wrote nothing, the
    same as on the Florida 4080 Super, the RTX 2080 and the Japan 3080 Ti.
    PeakMiner's binary is packed with UPX, which exits with code 127 when it
    can't unpack itself in memory. A host restriction is the likely cause,
    but we haven't confirmed it.
  - India 5090: refused to start, with "driver too old for this CUDA runtime".
    PeakMiner needs driver 580 or newer on 50-series cards. Our miner ran on
    that host with the CUDA 12.8 core.
- **SRBMiner, India 5060:** left the GPU idle (180 MHz, 10 W) and printed no
  rate and no error. The comparison there is PeakMiner.
- **RTX 5070 Ti:** both hosts hold the core at about 1340 MHz; the card's
  maximum is 3120. All three miners ran at 89–95 TH/s on under 140 W, far
  below a stock 5070 Ti. The percentage is still a fair same-host
  comparison. At that clock our miner drew 135–139 W against 94–99 W for the
  other two. Two 300 W hosts were tried and couldn't attach the GPU.
- **Not tested:** the RTX 5090D, which no host lists, and laptop GPUs (3060,
  4070 and 4080 laptop), which are on Vast but left out.

## A100-class (sm_80, PR #253)

Run on 2026-10-07, 14:53–15:15 UTC, with PR #253's core at `feba1ca`. No
release supports these cards yet. Same harness, pool and miner versions as
the RTX tables.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| A100 PCIe 40 GB | Japan (250 W) | 146405 | 192.0 | 214.8 | 216.0 | 89% |
| CMP 170HX | Georgia (250 W) | 147823 | 146.9 | 167.9 | 165.0 | 87% |

- **Both cards:** all three miners ran at the 250 W limit. PeakMiner and
  SRBMiner get about 85% of the tensor cores' peak work per clock, and we get
  about 70%. We run at a higher clock (on the A100, 1236 MHz against their
  1151–1155), so the gap is work per clock, not power.
- **CMP 170HX:** SRBMiner's rate fell near the end of its run. Its average
  after the first minute was 168.9, which also puts us at 87%. The card has
  74 SMs and a 32 MB L2, with the A100's full tensor rate per SM.

## Replaced hosts (40/50-series)

Hosts that couldn't run the test were swapped for the next one by the same
rules.

- The host couldn't attach the GPU to the container ("failed to inject CDI
  devices"): 4090 Estonia (56522), 5090 British Columbia (38389), 5090 Czechia
  (144726), and 5070 Ti South Korea (18149 and 18370).
- The host's Docker had no NVIDIA runtime: 4080 Utah (145428).
- The host couldn't resolve DNS: 4080 Super Japan (36413).
- Docker Hub's download limit stopped the image pull: 4090 Ukraine (138134).
- Outbound port 1200 was blocked: 5060 Hong Kong (149228).
- No longer rentable when the run started: 4070 Ti Super Texas (137944), 4070
  Kentucky (34040), 5090 Romania (137732), 5080 Georgia (147973), 5070 Kansas
  (151873) and 5060 United States (68005).
- The first 5060 Virginia box was stopped too early, while still
  downloading. It was rerun on the same host.
