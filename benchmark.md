# RTX benchmark

Our miner against PeakMiner 2.17.6 and SRBMiner 3.7.1 on rented Vast hosts, one
or two hosts per card. Each host runs all three miners one after another on the
same pool, so rates within a row compare directly. Don't compare across rows:
power limits and cooling differ from host to host.

## Summary

How close our miner gets to the faster of PeakMiner and SRBMiner on the same
host:

- RTX 20-series: 94–98% on the six hosts re-measured with v0.5.13's kernel.
  Two rows are still from v0.5.11: the RTX 2070 (89%) and the 2060 Super
  (88%).
- RTX 30-series: 97–104% on the eleven hosts re-measured with v0.5.13's
  kernel. Four rows are still from v0.5.11, at 78–80%: both 3090 Tis, the
  Ukraine 3090 and the 220 W Quebec 3070.
- RTX 40-series: 99–103% on the eight hosts re-measured with v0.5.13's
  kernel, and 97–101% on the ten still from v0.5.11.
- RTX 50-series: 99–100% on the four hosts re-measured with v0.5.13's kernel.
  Of the eight still from v0.5.11, the 5090s and 5080s are close to even
  (98–102%); the 5070 Tis, the Poland 5070 and the India 5060 trail by
  5–10%.
- A100-class (sm_80, new in v0.5.13): 97–107% on the three rows with
  v0.5.13's kernel: the A30 107%, the A100 PCIe 99% and the New York A100
  SXM4 97%. The California A100 SXM4 (89%, on a host that cools the card
  poorly) and the CMP 170HX (87%) are from an earlier build. A speed test of
  v0.5.13's kernel on the CMP 170HX read 98%.
- Hopper (sm_90a, new in v0.5.13): 93–94% on the H100 NVL, H100 PCIe, H200
  and H200 NVL, re-measured with the published v0.5.13 release. The H100 SXM
  (83%) is from a build before v0.5.13, without the fold change that gained
  6.1–8.0% on the other four. These cards couldn't mine at all before
  v0.5.13.
- Ada workstation and data-center cards (sm_89: L40S, L40, L4, RTX 6000,
  5000, 4500, 4000 and 2000 Ada): 101–111%.
- RTX PRO Blackwell (sm_120): 99–101%. The 6000 Max-Q and the RTX 6000D are
  level; the RTX PRO 5000 and 6000 Server are 0.7% short.
- RTX A4000 (sm_86): 102%.

## How a host is tested

1. Get our CLI and core. The v0.5.11 runs downloaded them from the
   published release on GitHub, the same files a user gets. So did the
   re-measures of 17 RTX rows and four Hopper rows on 2026-10-08,
   17:03–18:47 UTC, from the published v0.5.13 release; those boxes also
   checked the files against the release digests. Each section names those
   rows. Other later runs built a commit from source, or as CI builds it;
   each section names the commit and says whether its kernel is
   byte-identical to v0.5.13's. The first 20-series run predates the
   release: it built PR #250 from source, which is the code that shipped as
   v0.5.11. On an RTX 50-series card with driver 580 or newer, the CLI loads
   the CUDA 13 core, as a user's rig would, and the hit check runs on that
   core.
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
the first minute. Up to v0.5.11 a single reading covered only about 0.5 s
and could be off by one batch (see the RTX 2070 note); from v0.5.12 each
reading is the average since the previous one. The first 20-series run used
the last reading, so the one row still from it, the 2060 Super, is a
single reading. On a multi-GPU rental, every miner is pinned to GPU 0 and
checked to use only that card.

"% of best" is our rate divided by the faster of PeakMiner and
SRBMiner on the same host.

A re-measure is a pool run of only our miner, as above, on the row's own
host, and the row keeps its PeakMiner and SRBMiner figures. A speed test
isn't a re-measure: `hashrate.js` runs the core alone, with no pool, for
3–4 rounds of 60 s after a 15 s warm-up, and the figure is the mean of the
rounds. Speed tests go in the section notes and never change a row. On the
hosts that have both a speed test and a pool run of the same kernel, the two
agree within 1% when the card holds the same clock in both. When it
doesn't, the rate follows the clock: the Alberta 2070 Super tested 57.3 at
about 1605 MHz and read 60.0 in its pool run at 1680 MHz. The Kentucky
A4000, at 92–94 C, tested 54.7 and read 56.9 in its pool run.

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

## Keeping this file current

- A change that re-measures a host in this file updates its row in the same
  PR. A number that only appears in chat or in a PR description doesn't
  count as recorded.
- Only a run of our miner on the row's own host (same machine ID) replaces
  its "Ours" and "% of best". Give the earlier figure in that section's
  notes, with the date and the commit that ran.
- PeakMiner and SRBMiner figures are fixed targets. Don't rerun them for a
  row that has them. A new host runs all three miners.
- A run on a different host, or by a different method than "How a host is
  tested", goes in the section's notes and leaves the row alone.
- When a row changes, check that the Summary and the section heading still
  match it.
- Before tagging a release, check that the headings, the Summary and "How a
  host is tested" name the right release, and that no section calls a
  shipped card unreleased.

## RTX 20-series (v0.5.13 and v0.5.11)

Run on 2026-10-06, 09:15–09:38 UTC, with PR #250 built from source at
`f66f80c`: the code that shipped as v0.5.11. The RTX 2070 was rerun at
10:24 UTC with the v0.5.11 release, averaging our readings.

Six rows were re-measured later with 5-minute pool runs. The Pennsylvania
2080 Ti ran on 2026-10-08, 14:10–14:19 UTC, with the v0.5.13 release
candidate (`26c50b3`) built as CI builds it, which has the same sm_75
kernel as v0.5.12 and v0.5.13, byte for byte. The Thailand 2080 Ti, the
2080, the 2070 Super and both 2060s ran on 2026-10-08, 18:21–18:37 UTC,
with the published v0.5.13 release. The RTX 2070 and the 2060 Super are
still from v0.5.11.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RTX 2080 Ti | Thailand (170 W) | 95392 | 67.7 | 70.9 | 70.3 | 95% |
| RTX 2080 Ti | Pennsylvania (260 W) | 150735 | 89.6 | 94.2 | 94.2 | 95% |
| RTX 2080 | Colorado (275 W) | 149439 | 78.3 | waiting | 80.3 | 97% |
| RTX 2070 Super | Alberta (215 W) | 31798 | 60.0 | 60.9 | 59.1 | 98% |
| RTX 2070 | South Korea (150 W) | 139007 | 44.5 | 50.1 | 48.9 | 89% |
| RTX 2060 Super | Germany (175 W) | 149900 | 41.1 | 46.5 | 45.0 | 88% |
| RTX 2060 | Australia (190 W, 6 GB) | 152547 | 48.8 | 50.3 | 47.3 | 97% |
| RTX 2060 | South Korea (184 W, 12 GB) | 27568 | 49.8 | 52.9 | 52.3 | 94% |

Every host passed the hit check, and every run had 0 rejected shares.

- **Against the top competitor:** the six re-measured hosts are at 94–98%.
  On v0.5.11 every card was at 85–91%, and the RTX 2070 and the 2060 Super
  still show that. The top competitor is PeakMiner everywhere except the
  2080, where PeakMiner didn't run and SRBMiner is the comparison. On the
  Pennsylvania 2080 Ti the two tie at 94.2. On v0.5.11, PeakMiner kept the
  tensor cores busier: about 92% of peak against our 78–82%.
- **On v0.5.11:** the re-measured rows read 60.5 (85%) on the Thailand
  2080 Ti, 85.7 (91%) on the Pennsylvania 2080 Ti, 70.9 (88%) on the 2080,
  55.0 (90%) on the 2070 Super, and 43.4 (86%) and 46.3 (88%) on the
  Australia and South Korea 2060s.
- **RTX 2060 Super:** a speed test of the v0.5.13 kernel on this host read
  44.0 (95%) on 2026-10-06, on `dc97cbd` with a patch and build flags. It
  isn't a pool run, so the row keeps its v0.5.11 figure.
- **RTX 2080 Ti, Thailand:** the host holds the card at about 1095 MHz,
  below its 170 W limit. It did for all three miners on 2026-10-06 and for
  the v0.5.13 pool run, which drew about 148 W.
- **RTX 2070 Super:** in the v0.5.13 pool run the card drew about 210 W at
  1680 MHz. On 2026-10-06 the three miners drew 198–206 W, so this row may
  be a little high.
- **RTX 2070:** the first run's last CLI reading was 57.2 TH/s, above the
  card's peak of 51.4 at the clock it ran at. On v0.5.11 a single reading
  covered about 0.5 s, and with two batches in flight one batch could land in
  the next reading. The rerun averages every reading after the first minute:
  44.5, with single readings from 29 to 59. Shares were never affected.
- **RTX 2080:** PeakMiner wrote nothing to its log and the GPU stayed idle, so
  there's no PeakMiner number yet. The same host was rented again later that
  day for a speed test and on 2026-10-08 for the pool run in the row, but
  PeakMiner hasn't been rerun on it.
- **SRBMiner** logs an OpenCL error at start on every 20-series card, then
  mines normally on CUDA.

## RTX 30-series (v0.5.13 and v0.5.11)

Run on 2026-10-06, 10:24–11:10 UTC, with the v0.5.11 release.

Eleven rows were re-measured later with pool runs. Eight ran on 2026-10-07,
02:39–16:18 UTC, built from source at `a7247f1`, or with a patch on
`79173e2`, or with a patch and build flags on `1c0db74`. All three builds
have the same sm_86 kernel as v0.5.12 and v0.5.13, byte for byte. Those are
5-minute pool runs, except the Portugal 3080 Ti's, which ran 10 minutes.
The Quebec 3090, the 3070 Ti and the 180 W Quebec 3070 ran 5-minute pool
runs on 2026-10-08, 18:29–18:47 UTC, with the published v0.5.13 release.
Still from v0.5.11: both 3090 Tis, the Ukraine 3090 and the 220 W Quebec
3070.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RTX 3090 Ti | Greece (450 W) | 152830 | 116.9 | 145.9 | 144.3 | 80% |
| RTX 3090 Ti | Washington (450 W) | 151121 | 119.8 | 149.7 | 153.1 | 78% |
| RTX 3090 | Ukraine (390 W) | 141130 | 103.9 | 129.9 | 131.2 | 79% |
| RTX 3090 | Quebec (300 W) | 152641 | 121.8 | 117.6 | 114.9 | 104% |
| RTX 3080 Ti | Japan (330 W) | 137807 | 127.1 | didn't start | 124.5 | 102% |
| RTX 3080 Ti | Portugal (350 W) | 56596 | 123.2 | 126.7 | 127.3 | 97% |
| RTX 3080 | Kentucky (320 W) | 29108 | 106.9 | 107.6 | 108.1 | 99% |
| RTX 3080 | Washington (280 W) | 25433 | 90.6 | 91.5 | 91.3 | 99% |
| RTX 3070 Ti | Ontario (310 W) | 43435 | 85.9 | 85.8 | 85.6 | 100% |
| RTX 3070 | Quebec (220 W) | 148988 | 59.7 | 75.6 | 75.6 | 79% |
| RTX 3070 | Quebec (180 W) | 152549 | 74.3 | 72.7 | 72.6 | 102% |
| RTX 3060 Ti | Japan (180 W) | 137800 | 63.7 | 63.7 | 63.9 | 100% |
| RTX 3060 Ti | New Zealand (220 W) | 144095 | 64.7 | 65.7 | no rate | 98% |
| RTX 3060 | Poland (170 W) | 149975 | 49.2 | 50.0 | 49.8 | 98% |
| RTX 3060 | Vietnam (170 W) | 138808 | 48.4 | 48.9 | 48.7 | 99% |

Every host passed the hit check, and every run had 0 rejected shares.

- **Against the top competitor:** the eleven re-measured hosts are at
  97–104%. On v0.5.11 every card was at 74–80%, and both 3090 Tis, the
  Ukraine 3090 and the 220 W Quebec 3070 still show that. PeakMiner and
  SRBMiner finish within 3% of each other on every card, and in the v0.5.11
  run all three miners ran at the same power cap, so that gap was ours.
- **On v0.5.11:** the re-measured rows read:
  - 3090 Quebec 87.3 (74%).
  - 3080 Ti Japan 97.2 (78%) and Portugal 99.7 (78%).
  - 3080 Kentucky 84.9 (79%) and Washington 70.6 (77%).
  - 3070 Ti 63.5 (74%), and 3070 Quebec 57.1 (79%) at 180 W.
  - 3060 Ti Japan 50.0 (78%) and New Zealand 51.7 (79%).
  - 3060 Poland 39.1 (78%) and Vietnam 38.0 (78%).
- **Speed tests, not pool runs:** on 2026-10-07 a speed test of the v0.5.13
  kernel on the Washington 3090 Ti read 153.2 (100%), on `79173e2` with a
  patch, and one on the 220 W Quebec 3070 read 74.7 (99%), on `a7247f1`.
  They aren't pool runs, so both rows keep their v0.5.11 figures.
- **RTX 3080 Ti, Portugal:** in the re-measure the card drew about 314 W.
  In the v0.5.11 run all three miners drew 348–349 W, so this row is
  probably low.
- **RTX 3060 Ti, Japan:** the rate follows the card's temperature. This pool
  run was at 70 C. An earlier one of the same kernel on this host, at 74 C,
  read 62.8 (98%).
- **RTX 3060, Vietnam:** the pool run got no share accepted in its 5
  minutes, and none rejected. It passed the hit check. The rate is the
  CLI's own figure. The box kept only the run's summary, not the CLI log.
- **Our readings swung on v0.5.11.** Single readings landed 20–35% either
  side of the mean, for example 69.8 to 105.5 around 87.3 on the Quebec 3090.
  The mean is what the card does; a single reading could mislead.
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

## RTX 40-series (v0.5.13 and v0.5.11)

Run on 2026-10-06, 11:12–12:10 UTC, with the v0.5.11 release.

Eight rows were re-measured later with 5-minute pool runs. The Australia
4090 ran on 2026-10-08, 14:11–14:22 UTC, with the v0.5.13 release candidate
(`26c50b3`) built as CI builds it, which has the same sm_89 kernel as
v0.5.12 and v0.5.13, byte for byte. Both 4080 Supers, the Taiwan 4080, both
4070 Supers, the Brazil 4060 Ti and the New Zealand 4060 ran on 2026-10-08,
18:22–18:37 UTC, with the published v0.5.13 release. Still from v0.5.11:
the British Columbia 4090, the 4090D, the Nevada 4080, both 4070 Ti Supers,
both 4070 Tis, the 4070, the Ontario 4060 Ti and the Australia 4060.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RTX 4090 | Australia (450 W) | 17334 | 323.5 | 314.7 | 314.2 | 103% |
| RTX 4090 | British Columbia (370 W) | 152545 | 289.3 | 287.5 | 289.6 | 100% |
| RTX 4090D | South Carolina (425 W) | 142279 | 272.2 | 264.1 | 271.1 | 100% |
| RTX 4080 Super | Florida (320 W) | 142085 | 197.9 | didn't start | 196.3 | 101% |
| RTX 4080 Super | California (275 W) | 138449 | 198.0 | 192.9 | 192.6 | 103% |
| RTX 4080 | Taiwan (280 W) | 149135 | 190.2 | 187.4 | 188.1 | 101% |
| RTX 4080 | Nevada (320 W) | 147894 | 194.7 | 197.4 | 197.1 | 99% |
| RTX 4070 Ti Super | Texas (285 W) | 142663 | 171.6 | 171.9 | 171.7 | 100% |
| RTX 4070 Ti Super | United States (285 W) | 43741 | 167.4 | 169.5 | 168.8 | 99% |
| RTX 4070 Ti | United Kingdom (285 W) | 149163 | 159.8 | 160.6 | 159.7 | 100% |
| RTX 4070 Ti | Delaware (285 W) | 39901 | 156.1 | 157.5 | 156.8 | 99% |
| RTX 4070 Super | California (209 W) | 145255 | 138.5 | 139.6 | 139.3 | 99% |
| RTX 4070 Super | California (220 W) | 153237 | 141.7 | 139.0 | 141.3 | 100% |
| RTX 4070 | New York (200 W) | 19053 | 122.0 | 120.9 | 119.4 | 101% |
| RTX 4060 Ti | Ontario (160 W) | 37799 | 86.3 | 86.1 | 86.6 | 100% |
| RTX 4060 Ti | Brazil (160 W) | 152073 | 88.7 | 88.3 | 89.0 | 100% |
| RTX 4060 | New Zealand (115 W) | 148383 | 60.2 | 60.3 | 58.7 | 100% |
| RTX 4060 | Australia (120 W) | 143986 | 46.5 | 47.7 | 47.5 | 97% |

Every host passed the hit check, and every run had 0 rejected shares.

- **Against the top competitor:** 99–103% on the eight re-measured hosts and
  97–101% on the rest. All three miners finish within about 3% of each other
  on Ada, so there's little left to gain here.
- **On v0.5.11:** the re-measured rows read 314.9 (100%) on the Australia
  4090, 196.5 (100%) and 193.4 (100%) on the Florida and California 4080
  Supers, 189.6 (101%) on the Taiwan 4080, 138.5 (99%) and 139.9 (99%) on
  the 209 W and 220 W 4070 Supers, 89.1 (100%) on the Brazil 4060 Ti, and
  59.9 (99%) on the New Zealand 4060.
- **Speed tests, not pool runs:** on 2026-10-07 a speed test of the v0.5.13
  kernel, on `79173e2`, read 198.1 (100%) on the Nevada 4080, 171.6 (100%)
  on the Texas 4070 Ti Super and 160.0 (100%) on the United Kingdom
  4070 Ti. They aren't pool runs, so those rows keep their v0.5.11 figures.
- **RTX 4060, Australia:** the host runs the card in two clock states. It
  sometimes locks it at 1995 MHz and sometimes leaves it power-capped at
  about 2600 MHz. In the v0.5.11 session all three miners ran at 1995 MHz,
  so the fixed competitor figures come from that state. At
  1995 MHz a speed test of the v0.5.13 kernel, on `79173e2`, read 47.1
  (99%) on 2026-10-07; it isn't a pool run. A pool run of the published
  v0.5.13 release on 2026-10-08 read 61.1 at about 2600 MHz. That's the
  other state, so it can't be compared with this row, which keeps its
  v0.5.11 figure. Both later runs had a 115 W limit; the v0.5.11 session
  had 120 W.
- **RTX 4060 Ti, Brazil:** the v0.5.13 pool run got no share accepted in
  its 5 minutes, and none rejected. It connected to the pool and passed the
  hit check. The rate is the CLI's own figure.
- **RTX 4070 Super, California (220 W):** in the v0.5.13 pool run the card
  ran at a median 81 C, reached 85 C, and slowed for heat in 8 of 49
  samples, so its lowest reading was 138.5 against a median of 142.1. This
  row may be a little low.
- **RTX 4080 Super, Florida:** PeakMiner exited at once with code 127 and
  wrote nothing (see the 50-series notes). The comparison there is SRBMiner.
  It's a 2-GPU rental. Every miner was pinned to GPU 0, and each one's rate
  matches a single card.
- **RTX 4090D:** one host. The Tanzania host (70632) never started its
  container, in two tries of 16 and 30 minutes, and no other 4090D is listed.

## RTX 50-series (v0.5.13 and v0.5.11)

Run in the same window as the 40-series.

Four rows were re-measured later with 5-minute pool runs, all on the CUDA 13
core. The Virginia 5060 Ti ran on 2026-10-08, 13:24–13:34 UTC, with the
v0.5.13 release candidate (`26c50b3`) built as CI builds it, and the
Virginia 5060 on 2026-10-07 on `3005e42`. Both have the same sm_120 kernel
as v0.5.12 and v0.5.13, byte for byte. The Colombia 5070 and the Ontario
5060 Ti ran on 2026-10-08, 18:35–18:47 UTC, with the published v0.5.13
release. Still from v0.5.11: both 5090s, both 5080s, both 5070 Tis, the
Poland 5070 and the India 5060.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RTX 5090 | South Korea (500 W) | 145183 | 384.5 | didn't start | 378.0 | 102% |
| RTX 5090 | India (500 W) | 151861 | 369.0 | driver too old | 375.3 | 98% |
| RTX 5080 | New Jersey (360 W) | 152066 | 223.5 | 227.0 | 229.0 | 98% |
| RTX 5080 | Poland (360 W) | 153231 | 223.6 | 227.8 | 228.3 | 98% |
| RTX 5070 Ti | South Korea (250 W) | 27661 | 89.5 | 93.6 | 94.6 | 95% |
| RTX 5070 Ti | South Korea (250 W) | 28852 | 89.2 | 93.6 | 94.5 | 94% |
| RTX 5070 | Colombia (250 W) | 145974 | 132.1 | 132.3 | 133.5 | 99% |
| RTX 5070 | Poland (250 W) | 31379 | 126.1 | 132.7 | 134.2 | 94% |
| RTX 5060 Ti | Virginia (150 W) | 151123 | 94.1 | 94.5 | 94.3 | 100% |
| RTX 5060 Ti | Ontario (180 W) | 153080 | 95.3 | 95.3 | 95.7 | 100% |
| RTX 5060 | Virginia (125 W) | 151478 | 76.8 | 76.8 | 77.1 | 100% |
| RTX 5060 | India (145 W) | 119163 | 68.8 | 76.4 | didn't mine | 90% |

Every host passed the hit check, and every run had 0 rejected shares.

- **Against the top competitor:** 99–100% on the four re-measured hosts. The
  v0.5.11 run was at 98–102% on the 5090 and 5080, 94–95% on the 5070 Ti and
  5070, and 90–93% on the 5060 Ti and 5060. On the power-capped cards below
  the 5080, v0.5.11 ran 60–185 MHz slower than the other two at the same
  power. It used more power per clock, so the card clocked down to stay under
  its cap. That's where the gap on the smaller cards came from. On v0.5.13
  the four re-measured cards ran 30–53 MHz faster than the other two did.
- **On v0.5.11:** the re-measured rows read 88.0 (93%) on the Virginia
  5060 Ti, 87.6 (92%) on the Ontario 5060 Ti, 70.7 (92%) on the Virginia
  5060 and 125.7 (94%) on the Colombia 5070.
- **Speed tests, not pool runs:** on 2026-10-07 a speed test of the v0.5.13
  kernel read 229.7 (101%) on the Poland 5080, on `3005e42`, and 92.3 (98%)
  on the 5070 Ti on 27661, on `79173e2` with a patch and a build flag. They
  aren't pool runs, so both rows keep their v0.5.11 figures.
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
  comparison. On v0.5.11, at that clock our miner drew 135–139 W against
  94–99 W for the other two; in the speed test on 27661 it drew 87–88 W.
  Two 300 W hosts were tried and couldn't attach the GPU.
- **Not tested:** the RTX 5090D, which no host lists, and laptop GPUs (3060,
  4070 and 4080 laptop), which are on Vast but left out.

## A100-class (sm_80, v0.5.13 and an earlier build)

Run on 2026-10-07, 14:53–22:52 UTC. The A100 PCIe and the New York A100
SXM4 ran `3804a7c` and the A30 `1f699da`, each with PeakMiner and SRBMiner
re-run on the same box in the same session. Both builds have the same sm_80
kernel as v0.5.13, byte for byte. The CMP 170HX and the California A100
SXM4 ran the first build, `feba1ca`, whose sm_80 kernel differs.
v0.5.13 is the first release that supports these cards. Same harness, pool
and miner versions as the RTX tables.

One row was re-measured later, with v0.5.13's sm_80 kernel: the A100 PCIe,
a 5-minute pool run on 2026-10-08, 13:13–13:22 UTC, with the v0.5.13
release candidate (`26c50b3`) built as CI builds it. The CMP 170HX and the
California A100 SXM4 are still from `feba1ca`.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| A100 PCIe 40 GB | Japan (250 W) | 146405 | 212.8 | 212.2 | 213.9 | 99% |
| A100 SXM4 40 GB | New York (400 W) | 152837 | 246.5 | 252.2 | 254.7 | 97% |
| A100 SXM4 40 GB | California (400 W) | 152356 | 177.3 | 197.4 | 199.8 | 89% |
| A30 24 GB | Australia (155 W) | 145742 | 104.7 | 98.1 | 96.2 | 107% |
| CMP 170HX | Georgia (250 W) | 147823 | 146.9 | 167.9 | 165.0 | 87% |

- **A100 PCIe and SXM4:** in the same-session runs, all three miners ran at
  the power limit. SRBMiner does 4.0–4.3% more work per clock than we do,
  and we run a 1–2.6% higher clock at the same watts, so the gap is work per
  clock. On `feba1ca` the PCIe read 192.0 (89%) and on `dd64b2c` the SXM4
  read 240.3 (95%). The copy points (`dd64b2c`), the hash in its own kernel
  (`1f699da`) and band 32 (`3804a7c`) closed most of it.
- **A100 PCIe, re-measured:** in the session with PeakMiner and SRBMiner it
  read 209.5 (98%). This host runs the card at about 1170 MHz in some
  sessions and about 1200 MHz in others, at the same 250 W, and the same
  build reads about 209 and 213. The re-measure ran at 1200 MHz. PeakMiner
  and SRBMiner ran in a 1170 MHz session, at 1136–1143 MHz themselves, so
  this row is probably high.
- **CMP 170HX:** all three miners ran at the 250 W limit on `feba1ca`, when
  we got about 70% of the tensor peak per clock to their 85%. A speed test
  of the v0.5.13 kernel on this host read 164.6 (98%) on 2026-10-07, on
  `e07b1ab` at the same 250 W, and an earlier one of the same kernel read
  165.0. They aren't pool runs, so the row keeps its `feba1ca` figure.
- **A100 SXM4, New York:** all three miners ran at the 400 W limit, at
  75–77 C. This is the row that shows the card at its rated power.
- **A100 SXM4, California:** the host cools the card poorly. All three miners
  ran at 85 C and the thermal limit, drawing about 215 W of the 400 W allowed,
  so this row says more about the host than the card. The first SXM4 host
  tried (149846) failed during setup.
- **A30:** the host enforces 155 W, not its listed 165 W, and all three
  miners were power-capped at 82 C. This row is a second run on `1f699da`,
  with PeakMiner and SRBMiner re-run on the same box at the same 154 W. On
  `feba1ca` we read 94.7 against 98.0 and 96.1 (97%).
- **CMP 170HX, first run:** SRBMiner's rate fell near the end of its run.
  Its average after the first minute was 168.9, which also puts us at 87%.
  The card has 74 SMs and a 32 MB L2, with the A100's full tensor rate per
  SM. Its owner has unlocked it: it reports 64 GB, where a stock card has
  8 GB and 70 SMs. Every CMP 170HX on Vast is like this, so a stock card
  hasn't been tested.

## Hopper (sm_90a, v0.5.13 and a build before it)

Run on 2026-10-08, 06:35–07:17 UTC, with PR #253's release build at `e8c04b4`,
which mines with Hopper's wgmma fold by default. Each box built the core as
native-core.yml does (CUDA 12.8, sm_75/80/86/89/90a/120, no `-D` flags).
v0.5.13 supports these cards, with a faster fold than this build (see "Since
`e8c04b4`" below). Same harness, pool and miner versions as the RTX tables.
PeakMiner and SRBMiner on the H100 SXM and NVL hosts are the 2026-10-07
runs; the other three hosts are new, so all three miners ran there.

Four rows were re-measured on 2026-10-08 with the published v0.5.13
release: the H100 NVL, the H100 PCIe and the H200 NVL at 17:03–17:09 UTC,
and the H200 at 18:21–18:26 UTC. Each box downloaded the CLI and core from
GitHub, checked them against the release digests, and ran the hit check and
a 5-minute pool run. The H100 SXM is still from `e8c04b4`.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| H100 SXM 80 GB | New York (700 W) | 153443 | 630.8 | 757.7 | 734.4 | 83% |
| H100 NVL 94 GB | Japan (400 W) | 29785 | 505.1 | 533.0 | 537.6 | 94% |
| H100 PCIe 80 GB | Czechia (350 W) | 147981 | 454.4 | 391.6 | 481.0 | 94% |
| H200 141 GB | Saudi Arabia (700 W) | 131919 | 669.7 | 718.6 | 720.4 | 93% |
| H200 NVL 141 GB | Quebec (600 W) | 153365 | 627.4 | 669.8 | 675.3 | 93% |

- **Every host:** passed the hit check and picked the wgmma fold (2-CTA
  clusters, 128x256 tiles). The CLI got 26, 16, 18, 20 and 26 shares
  accepted, none rejected.
- **The gap is work per clock.** Every miner ran at the power cap. On
  `e8c04b4` we ran 70–290 MHz faster than the other two at the same watts.
  Per clock that fold did 71–75% of the wgmma peak and the competitors
  87–94%; most of the difference was the fold's readout. On the four hosts
  re-measured on v0.5.13, the fold does 9–13% more work per clock than on
  `e8c04b4`. The card clocks 30–60 MHz lower at the same cap, which is still
  30–250 MHz faster than the other two, and per clock we do 90–92% of
  SRBMiner's work.
- **On `e8c04b4`:** the re-measured rows read 471.0 (88%) on the H100 NVL,
  423.1 (88%) on the H100 PCIe, 619.9 (86%) on the H200 and 591.5 (88%) on
  the H200 NVL.
- **Before `e8c04b4`:** the cp.async port, run on 2026-10-07, did 436.2 on the
  H100 SXM (58%) and 341.0 on the H100 NVL (63%), and 299.4 against 478.4 and
  485.1 on an H100 PCIe in the United States (81035, 62%).
- **Since `e8c04b4`:** v0.5.13 adds `7040f89`, which removes a bank conflict
  in the fold's readout and the descriptor copies that took issue slots in
  its chunk loop. On the release, the H100 NVL read 7.2% more than on
  `e8c04b4`, the H100 PCIe 7.4%, the H200 8.0% and the H200 NVL 6.1%. The
  H100 SXM host hasn't been run on it yet. On two other hosts,
  `hashrate.js` speed tests on 2026-10-08 of a build with v0.5.13's sm_90a
  kernel read 7.6% more than without `7040f89` on an H100 SXM (California,
  152422, 700 W: 670.2 against 623.0) and 7.1% more on an H100 NVL (South
  Korea, 58970, 400 W: 509.4 against 475.4). Those are different hosts and
  a different method, so the H100 SXM row is unchanged.
- **H100 PCIe:** 81035 wasn't offered, so this is a new host. PeakMiner ran
  at 915 MHz and fell from about 404 to 391.6 by 5:00. SRBMiner is the faster
  competitor here either way.
- **H200:** the same chip as the H100 with faster memory, but on `e8c04b4`
  the fold does about 10% less per clock on it than on the H100 SXM (about
  2770 against 3063 int8 MAC/clock/SM). Not looked into yet.
- **Order:** on the three new hosts the miners ran ours, then SRBMiner, then
  PeakMiner, not alternating between hosts.
- **Skipped hosts:** H200 Massachusetts (153354, reliability 0.68), H200 New
  Jersey (153539, performance score 2), H200 NVL Czechia (43532, rented at
  $5.61/h against the $3.74 listed, destroyed after 14 s), and H100 PCIe
  France (153139, restarts the container).

## Ada workstation and data-center cards (sm_89, v0.5.13)

Run on 2026-10-07, 16:55–21:45 UTC, with PR #253's core at `5f427dd`,
`54c993d` or `e0bee03`. Their sm_89 kernels are byte-identical to each other
and to v0.5.13's, and no change was made for these cards. They run the same
sm_89 code as the RTX 40-series. Same harness, pool and miner versions as the
RTX tables.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| L40S | Texas (350 W) | 96563 | 288.4 | 257.1 | 281.8 | 102% |
| L40S | Taiwan (350 W) | 151291 | 288.9 | 250.8 | 270.7 | 107% |
| RTX 6000 Ada | New York (300 W) | 142572 | 227.5 | 195.9 | 210.4 | 108% |
| RTX 6000 Ada | Germany (300 W) | 146274 | 194.2 | 167.5 | 179.2 | 108% |
| RTX 5000 Ada | Minnesota (250 W) | 151333 | 215.4 | 193.4 | 201.1 | 107% |
| L40 | Vietnam (300 W) | 114318 | 173.8 | 166.5 | 170.4 | 102% |
| L40 | New York (250 W) | 56199 | 153.6 | 131.8 | 139.0 | 111% |
| RTX 4500 Ada | France (210 W) | 148971 | 146.7 | 134.3 | 145.1 | 101% |
| RTX 4000 Ada | Norway (130 W) | 149238 | 96.6 | 94.0 | 95.0 | 102% |
| RTX 4000 Ada | France (130 W) | 152701 | 93.1 | 91.3 | 91.8 | 101% |
| L4 | Czechia (72 W) | 116596 | 79.4 | 74.4 | 77.8 | 102% |
| L4 | Washington (72 W) | 152541 | 78.5 | 74.2 | 77.2 | 102% |
| RTX 2000 Ada | Romania (70 W) | 68269 | 47.8 | 47.1 | 43.6 | 102% |

- **Every host:** every miner ran at the power limit. We ran 30–310 MHz
  faster than the other two at the same watts, so we use less power for the
  same work. Per clock we get about 96% of the tensor peak and SRBMiner about
  97%.
- **L40:** half the L40S's int8 tensor rate per SM, so its rates are lower.
  Every other card here has the full rate, the L4 included. The New York
  host sets the L40 to 250 W; its maximum is 300 W.
- **Hosts matter:** at the same 300 W, the Germany RTX 6000 Ada ran at
  83–85 C and about 1400 MHz, and the New York one at 77–79 C and 1640 MHz,
  so every miner was about 17% faster in New York. The RTX 4000 Ada read 4%
  higher in Norway (74 C) than in France (85 C).
- **RTX 2000 Ada:** PeakMiner is the faster competitor here.
- **Replaced hosts:** L4 Brazil (151023) blocks outbound port 1200, so the
  competitors couldn't run. L4 Utah (109523) is thermally limited: it drew
  about 50 W of its 72 W at 87–88 C, and every miner's rate swung by ±4 TH/s.

## RTX PRO Blackwell (sm_120, v0.5.13)

Run on 2026-10-07, 19:20–21:15 UTC, with PR #253's core at `54c993d` or
`e0bee03` (their sm_120 kernels are byte-identical to each other and to
v0.5.13's), on the CUDA 13 core, as a rig with driver 580 or newer loads it.
No change was made for these cards. Same harness, pool and miner versions as
the RTX tables.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RTX PRO 6000 Workstation | Mississippi (600 W) | 148552 | 397.0 | 389.9 | 393.9 | 101% |
| RTX PRO 6000 Server | California (600 W) | 150951 | 378.8 | 377.2 | 381.5 | 99% |
| RTX PRO 6000 Max-Q | Illinois (300 W) | 153399 | 311.4 | 307.4 | 311.5 | 100% |
| RTX PRO 5000 48 GB | California (250 W) | 150611 | 211.4 | 208.7 | 212.8 | 99% |
| RTX PRO 4500 | Arizona (200 W) | 148915 | 175.5 | 170.8 | 172.9 | 101% |
| RTX PRO 4000 | Pennsylvania (145 W) | 150214 | 127.4 | 126.2 | 126.9 | 100% |
| RTX 6000D | Czechia (550 W) | 149788 | 144.6 | 144.8 | 144.8 | 100% |

- **Per clock:** we get 94–96% of the tensor peak on every card, and SRBMiner
  97–98%. At the power cap we run 2.6–4.1% faster at the same watts, which
  is enough on most cards. The RTX PRO 5000 (2.6%) is 0.7% short, and the
  6000 Max-Q (99.97%) and the RTX 6000D (99.9%) round to 100%.
- **RTX PRO 6000 Server:** a passive server card. Once it reached 85 C it
  held about 2065 MHz and 440 W of its 600 W, for every miner, so heat, not
  power, set its rate on this host.
- **RTX PRO 5000 48 GB:** the host sets 250 W, 83% of the card's 300 W.
- **RTX 6000D:** Vast's name for a cut-down card with 156 SMs and 84 GB. Its
  int8 tensor rate is capped at about 37% of the RTX PRO 6000's per SM per
  clock. All three miners tie at its top clock and about 253 W.
- **Not in the table:** the RTX PRO 5000 72 GB. Its only host (California,
  153314) blocks outbound port 1200, so no miner could reach the pool. Our
  speed test read 239.4 at 300 W. Vast lists no RTX PRO 2000 or 4000 SFF.

## RTX A4000 (sm_86, v0.5.13)

Run on 2026-10-07, 16:53–17:47 UTC, with PR #253's core at `5f427dd`, whose
sm_86 kernel is byte-identical to v0.5.13's. The A4000 is a GA104 card and
runs the same sm_86 code as the RTX 30-series. Same harness, pool and miner
versions as the RTX tables.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RTX A4000 | Kazakhstan (140 W) | 147920 | 59.5 | 58.2 | 56.6 | 102% |
| RTX A4000 | Kentucky (140 W) | 29102 | 56.9 | 55.8 | 53.5 | 102% |

- **Both hosts:** every miner ran at the 140 W limit. We ran 60–110 MHz
  faster than the other two.
- **Kentucky:** the card runs at 92–94 C. In this 5-minute run all three
  miners stayed at the power limit, but in our longer speed tests the card
  hit its thermal limit and read 54.7–55.1 TH/s.
- **PeakMiner, Kazakhstan:** got no share accepted in its 5 minutes. The
  rate is its own figure.
- **Replaced host:** Germany (14335). Another workload was using the GPU,
  and the tensor probe read 37 T-MAC/s against 92 on a clean card.

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
