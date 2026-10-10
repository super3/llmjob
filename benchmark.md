# RTX benchmark

Our miner against PeakMiner 2.17.6 and SRBMiner 3.7.1 on rented Vast hosts, one
or two hosts per card. Each host runs all three miners one after another on the
same pool, so rates within a row compare directly. Don't compare across rows:
power limits and cooling differ from host to host.

## Summary

How close our miner gets to the faster of PeakMiner and SRBMiner on the same
host:

- RTX 20-series: 94–98% on all eight hosts, on v0.5.13's kernel. The RTX
  2070 (95%) and the 2060 Super (94%) ran v0.5.14's core (PR #256's build
  at `50b2d6b`), which has the same sm_75 kernel.
- RTX 30-series: 97–104% on the eleven hosts re-measured with v0.5.13's
  kernel, two of them on v0.5.14's core (PR #256's build at `50b2d6b`),
  which has the same kernel. Four rows are still from v0.5.11, at 78–80%:
  both 3090 Tis, the Ukraine 3090 and the 220 W Quebec 3070.
- RTX 40-series: 96–103% on the twelve hosts re-measured with v0.5.14's
  code (PR #256's build at `1040668` or `50b2d6b`), and 99–101% on the six
  still from v0.5.11. Against v0.5.13 in the same rental, `1040668` gained
  0.5% on the 209 W 4070 Super and was level on the rest. That card is at
  96% because its host has run slower in each later rental: 138.5 on
  v0.5.13, then 136.0 on `1040668` and 134.5 on `50b2d6b`.
- RTX 50-series: 99–100% on the three hosts re-measured with v0.5.13's
  kernel, and 95–99% on the four re-measured with v0.5.14's core (PR #256's
  build at `fec1e8b`), which has the same kernel. The India 5090
  (95%) held a lower clock than in its earlier rental, and the 5070 Ti and
  the Poland 5070 may share their GPU with another workload. Of the five
  still from v0.5.11, the South Korea 5090 and the 5080s are close to even
  (98–102%); the 5070 Ti on 28852 and the India 5060 trail by 6–10%.
- A100-class (sm_80, new in v0.5.13): 94–107% on the four rows with
  v0.5.13's kernel: the A30 107%, the A100 PCIe 98%, the New York A100
  SXM4 96% and the California A100 SXM4 94%, on a host that cools the card
  poorly. The three A100s are from v0.5.14's core (PR #256's build at
  `50b2d6b`), which has the same kernel. The CMP 170HX (87%) is from an
  earlier build; a speed test of v0.5.13's kernel on it read 98%.
- Hopper (sm_90a, new in v0.5.13): 100–102% on the H100 NVL, H100 PCIe, H200
  and H200 NVL, re-measured with v0.5.14's Hopper fold (PR #256's build at
  `466ccf0`). That's 6.5–8.8% above their v0.5.13 pool runs. The H100 SXM
  (83%) is still from a build before v0.5.13, without the fold changes
  since; a speed test of v0.5.14's fold on another H100 SXM host read 96%.
  These cards couldn't mine at all before v0.5.13.
- Ada workstation and data-center cards (sm_89: L40S, L40, L4, RTX 6000,
  5000, 4500, 4000 and 2000 Ada): 101–111%. Ten rows are from v0.5.14's
  code (PR #256's build at `1040668`), which gained 0.7–1.7% over v0.5.13
  in the same rental on the L4s and the RTX 2000, 4000 and 4500 Ada, and
  was level on the rest. Three are still from 2026-10-07, on v0.5.13's
  kernel.
- RTX PRO Blackwell (sm_120): 98–101%. The 6000 Max-Q and the RTX 6000D are
  level; the RTX PRO 5000 is 0.7% short and the 6000 Server 1.5%. The RTX
  6000D and 6000 Server rows are from v0.5.14's core (PR #256's build at
  `50b2d6b`), which has v0.5.13's kernel.
- Ampere workstation and data-center cards (sm_86: RTX A6000, A5000, A4000
  and A2000, and the A40): 97–102% on v0.5.13's kernel. Seven of the ten
  rows are at 100–102%. The Kansas A6000 (99.8%) and the Belgium A40
  (99.4%) are just short, and the 6 GB A2000 is at 97%.
- B200 (sm_100, new in v0.5.14): 33%. Its hits are correct, but it mines on
  `mma.sync`, which tops out at about 41% of the competitors on this card.
  Matching them needs a fold on the B200's own tensor instructions
  (tcgen05). Before v0.5.14 a B200 couldn't mine at all.
- Tesla T4 (sm_75): 27.7 TH/s at its 70 W limit in one 5-minute pool run
  of v0.5.13. PeakMiner and SRBMiner weren't run, so there is no % of best.

## How a host is tested

1. Get our CLI and core. The v0.5.11 runs downloaded them from the published
   release on GitHub, the same files a user gets. So did the re-measures of
   nine RTX rows on 2026-10-08, 18:21–18:47 UTC, and the eight new sm_86 rows
   on 2026-10-09, from the published v0.5.13 release; those boxes also
   checked the files against the release digests. The 20-, 30-, 50-series
   and sm_86 sections name those rows. Other later runs
   built a commit from source, or as CI builds it; each section names the
   commit and says whether its kernel is byte-identical to v0.5.13's. The
   rows from PR #256's build ran a commit that has v0.5.14's core for that
   card; none ran the published v0.5.14 files. Those commits are named by their
   hashes after the rebase onto main on 2026-10-10; each one's native sources
   are byte-identical to the commit that ran. The first 20-series run
   predates v0.5.11: it built PR #250 from source,
   which is the code that shipped as v0.5.11. On an RTX 50-series card with
   driver 580 or newer, the CLI loads the CUDA 13 core, as a user's rig
   would, and the hit check runs on that core.
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
the last reading; no row is still from it. On a multi-GPU rental, every
miner is pinned to GPU 0 and checked to use only that card.

"% of best" is our rate divided by the faster of PeakMiner and
SRBMiner on the same host.

A re-measure is a pool run of only our miner, as above, on the row's own
host, and the row keeps its PeakMiner and SRBMiner figures. A speed test
isn't a re-measure: `hashrate.js` runs the core alone, with no pool, for
3–4 rounds of 60 s after a 15 s warm-up, and the figure is the mean of the
rounds. Speed tests go in the section notes and never change a row. On the
hosts that have both a speed test and a pool run of the same kernel, the two
agree within 1% when the card holds the same clock in both, except the Japan
H100 NVL: its pool run read 1.7% below its speed test, both at 1155 MHz.
When the clock differs, the rate follows it: the Alberta 2070 Super tested
57.3 at about 1605 MHz and read 60.0 in its pool run at 1680 MHz. The
Kentucky A4000, at 92–94 C, tested 54.7 and read 56.9 in its pool run.

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

## RTX 20-series (v0.5.14 and v0.5.13)

Run on 2026-10-06, 09:15–09:38 UTC, with PR #250 built from source at
`f66f80c`: the code that shipped as v0.5.11. The RTX 2070 was rerun at
10:24 UTC with the v0.5.11 release, averaging our readings.

Every row was re-measured later with a 5-minute pool run. The Pennsylvania
2080 Ti ran on 2026-10-08, 14:10–14:19 UTC, with the v0.5.13 release
candidate (`26c50b3`) built as CI builds it, which has the same sm_75
kernel as v0.5.12 and v0.5.13, byte for byte. The Thailand 2080 Ti, the
2080, the 2070 Super and both 2060s ran on 2026-10-08, 18:21–18:37 UTC,
with the published v0.5.13 release. The RTX 2070 and the 2060 Super ran on
2026-10-09, 04:28–06:02 UTC, with PR #256's build at `50b2d6b`, built on
each box with CI's flags. That build has v0.5.14's core for these cards.
Its sm_75 kernel is v0.5.13's, byte for byte.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RTX 2080 Ti | Thailand (170 W) | 95392 | 67.7 | 70.9 | 70.3 | 95% |
| RTX 2080 Ti | Pennsylvania (260 W) | 150735 | 89.6 | 94.2 | 94.2 | 95% |
| RTX 2080 | Colorado (275 W) | 149439 | 78.3 | waiting | 80.3 | 97% |
| RTX 2070 Super | Alberta (215 W) | 31798 | 60.0 | 60.9 | 59.1 | 98% |
| RTX 2070 | South Korea (150 W) | 139007 | 47.8 | 50.1 | 48.9 | 95% |
| RTX 2060 Super | Germany (175 W) | 149900 | 43.8 | 46.5 | 45.0 | 94% |
| RTX 2060 | Australia (190 W, 6 GB) | 152547 | 48.8 | 50.3 | 47.3 | 97% |
| RTX 2060 | South Korea (184 W, 12 GB) | 27568 | 49.8 | 52.9 | 52.3 | 94% |

Every host passed the hit check, and every run had 0 rejected shares.

- **Against the top competitor:** every host is at 94–98%. On v0.5.11
  every card was at 85–91%. The top competitor is PeakMiner everywhere
  except the 2080, where PeakMiner didn't run and SRBMiner is the
  comparison. On the Pennsylvania 2080 Ti the two tie at 94.2. On v0.5.11,
  PeakMiner kept the tensor cores busier: about 92% of peak against our
  78–82%.
- **On v0.5.11:** on 2026-10-06 the rows read 60.5 (85%) on the Thailand
  2080 Ti, 85.7 (91%) on the Pennsylvania 2080 Ti, 70.9 (88%) on the 2080,
  55.0 (90%) on the 2070 Super, 44.5 (89%) on the 2070 (the 10:24 UTC rerun
  with the v0.5.11 release), 41.1 (88%) on the 2060 Super (on `f66f80c`, a
  single reading), and 43.4 (86%) and 46.3 (88%) on the Australia and South
  Korea 2060s.
- **RTX 2060 Super:** the row is the pool run that came right after the hit
  check. An earlier rental of the same host that day, at 04:29 UTC, ran the
  same build and read 42.9 in its pool run, which came after 4 minutes of
  speed tests with the card at 77 C. On 2026-10-06 a speed test of the
  v0.5.13 kernel on this host read 44.0 (95%), on `dc97cbd` with a patch
  and build flags.
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

## RTX 30-series (v0.5.14, v0.5.13 and v0.5.11)

Run on 2026-10-06, 10:24–11:10 UTC, with the v0.5.11 release.

Eleven rows were re-measured later with 5-minute pool runs. Six ran on
2026-10-07, 05:09–16:18 UTC, built from source at `a7247f1`, which has the
same sm_86 kernel as v0.5.12 and v0.5.13, byte for byte. The Quebec 3090,
the 3070 Ti and the 180 W Quebec 3070 ran on 2026-10-08, 18:29–18:47 UTC,
with the published v0.5.13 release. The Portugal 3080 Ti and the Vietnam
3060 ran on 2026-10-09, 04:46–05:18 UTC, with PR #256's build at
`50b2d6b`, built on each box with CI's flags. That build has v0.5.14's core
for these cards. Its sm_86 kernel is v0.5.13's, byte for byte. Still from
v0.5.11: both 3090 Tis, the Ukraine 3090 and the 220 W Quebec 3070.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RTX 3090 Ti | Greece (450 W) | 152830 | 116.9 | 145.9 | 144.3 | 80% |
| RTX 3090 Ti | Washington (450 W) | 151121 | 119.8 | 149.7 | 153.1 | 78% |
| RTX 3090 | Ukraine (390 W) | 141130 | 103.9 | 129.9 | 131.2 | 79% |
| RTX 3090 | Quebec (300 W) | 152641 | 121.8 | 117.6 | 114.9 | 104% |
| RTX 3080 Ti | Japan (330 W) | 137807 | 127.1 | didn't start | 124.5 | 102% |
| RTX 3080 Ti | Portugal (350 W) | 56596 | 123.4 | 126.7 | 127.3 | 97% |
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
- **Before PR #256's build:** on 2026-10-07 the Portugal 3080 Ti read
  123.2 (97%) in a 10-minute pool run on `79173e2` with a patch, and the
  Vietnam 3060 read 48.4 (99%) on `1c0db74` with a patch and build flags.
- **Speed tests, not pool runs:** on 2026-10-07 a speed test of the v0.5.13
  kernel on the Washington 3090 Ti read 153.2 (100%), on `79173e2` with a
  patch, and one on the 220 W Quebec 3070 read 74.7 (99%), on `a7247f1`.
  They aren't pool runs, so both rows keep their v0.5.11 figures.
- **RTX 3080 Ti, Portugal:** the card drew about 314 W in the 2026-10-07
  pool run, and 313–314 W in the speed tests just before the 2026-10-09
  one. In the v0.5.11 run all three miners drew 348–349 W, so this row is
  probably low.
- **RTX 3060 Ti, Japan:** the rate follows the card's temperature. This pool
  run was at 70 C. An earlier one of the same kernel on this host, at 74 C,
  read 62.8 (98%).
- **RTX 3060, Vietnam:** neither re-measure got a share accepted in its 5
  minutes, and none was rejected. Both passed the hit check. The rate is
  the CLI's own figure.
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

## RTX 40-series (v0.5.14 and v0.5.11)

Run on 2026-10-06, 11:12–12:10 UTC, with the v0.5.11 release.

Eight rows were re-measured on 2026-10-08, 19:19–20:53 UTC, with 5-minute
pool runs of PR #256's build at `1040668`, which has v0.5.14's core for
these cards: the Australia 4090, both 4080 Supers, both 4080s, both 4060
Tis and the New Zealand 4060.
Each box built the core from source (CUDA 12.8, sm_89, native-core.yml's
flags) with `-DPEARL_LOG_ADA_L2=1`, which only prints the card's L2 and the
batch width, and ran it under the v0.5.13 CLI. The only change from v0.5.13
is host code that picks the batch width from the L2 (see "Ada batch width
(v0.5.14)" below), so the sm_89 kernel is v0.5.13's.
Four more ran on 2026-10-09, 06:19–08:12 UTC, with 5-minute pool runs of
PR #256's build at `50b2d6b`, built on each box with CI's flags: the Texas
4070 Ti Super, both 4070 Supers and the Australia 4060. It picks the same
batch widths as `1040668`, and its sm_89 kernel is v0.5.13's too. Still
from v0.5.11: the British Columbia 4090, the 4090D, the United States 4070
Ti Super, both 4070 Tis and the 4070.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RTX 4090 | Australia (450 W) | 17334 | 317.7 | 314.7 | 314.2 | 101% |
| RTX 4090 | British Columbia (370 W) | 152545 | 289.3 | 287.5 | 289.6 | 100% |
| RTX 4090D | South Carolina (425 W) | 142279 | 272.2 | 264.1 | 271.1 | 100% |
| RTX 4080 Super | Florida (320 W) | 142085 | 198.5 | didn't start | 196.3 | 101% |
| RTX 4080 Super | California (275 W) | 138449 | 198.6 | 192.9 | 192.6 | 103% |
| RTX 4080 | Taiwan (280 W) | 149135 | 190.4 | 187.4 | 188.1 | 101% |
| RTX 4080 | Nevada (320 W) | 147894 | 197.8 | 197.4 | 197.1 | 100% |
| RTX 4070 Ti Super | Texas (285 W) | 142663 | 172.3 | 171.9 | 171.7 | 100% |
| RTX 4070 Ti Super | United States (285 W) | 43741 | 167.4 | 169.5 | 168.8 | 99% |
| RTX 4070 Ti | United Kingdom (285 W) | 149163 | 159.8 | 160.6 | 159.7 | 100% |
| RTX 4070 Ti | Delaware (285 W) | 39901 | 156.1 | 157.5 | 156.8 | 99% |
| RTX 4070 Super | California (209 W) | 145255 | 134.5 | 139.6 | 139.3 | 96% |
| RTX 4070 Super | California (220 W) | 153237 | 140.0 | 139.0 | 141.3 | 99% |
| RTX 4070 | New York (200 W) | 19053 | 122.0 | 120.9 | 119.4 | 101% |
| RTX 4060 Ti | Ontario (160 W) | 37799 | 85.9 | 86.1 | 86.6 | 99% |
| RTX 4060 Ti | Brazil (160 W) | 152073 | 89.0 | 88.3 | 89.0 | 100% |
| RTX 4060 | New Zealand (115 W) | 148383 | 60.3 | 60.3 | 58.7 | 100% |
| RTX 4060 | Australia (120 W) | 143986 | 47.0 | 47.7 | 47.5 | 99% |

Every host passed the hit check, and every run had 0 rejected shares.

- **Against the top competitor:** 96–103% on the twelve hosts re-measured
  with PR #256's build and 99–101% on the six still from v0.5.11. All three
  miners finish within about 3% of each other on Ada, so there's little left
  to gain here.
- **On v0.5.13:** before PR #256's build, eight of the re-measured rows read,
  on 2026-10-08: 323.5 (103%) on the Australia 4090, on `26c50b3` at
  14:11–14:22 UTC; and with the published release at 18:22–18:37 UTC,
  197.9 (101%) and 198.0 (103%) on the Florida and California 4080 Supers,
  190.2 (101%) on the Taiwan 4080, 138.5 (99%) and 141.7 (100%) on the
  209 W and 220 W 4070 Supers, 88.7 (100%) on the Brazil 4060 Ti, and 60.2
  (100%) on the New Zealand 4060.
- **On v0.5.11:** the re-measured rows read 314.9 (100%) on the Australia
  4090, 196.5 (100%) and 193.4 (100%) on the Florida and California 4080
  Supers, 189.6 (101%) on the Taiwan 4080, 194.7 (99%) on the Nevada 4080,
  171.6 (100%) on the Texas 4070 Ti Super, 138.5 (99%) and 139.9 (99%) on
  the 209 W and 220 W 4070 Supers, 86.3 (100%) and 89.1 (100%) on the
  Ontario and Brazil 4060 Tis, and 59.9 (99%) and 46.5 (97%) on the New
  Zealand and Australia 4060s.
- **On `1040668`:** on 2026-10-08 the 4070 Supers read 136.0 (97%) at
  209 W and 140.1 (99%) at 220 W.
- **The release in the same rental:** each box ran the v0.5.13 release
  before and after `1040668` (A1 and A2, in "Ada batch width (v0.5.14)"
  below). Four hosts read lower than in their earlier rental because the
  host was slower, not because of the build:
  - Australia 4090: the release read 317.4 both times, against 323.5 on
    `26c50b3`.
  - 209 W 4070 Super: 135.4 and 135.3, against 138.5.
  - 220 W 4070 Super: 140.1 both times, against 141.7.
  - Ontario 4060 Ti: 85.8 and 85.9, against 86.3 on v0.5.11. The card ran
    at 83–84 C and slowed for heat in every run.
- **Speed tests, not pool runs:** on 2026-10-07 a speed test of the v0.5.13
  kernel, on `79173e2`, read 198.1 (100%) on the Nevada 4080, 171.6 (100%)
  on the Texas 4070 Ti Super and 160.0 (100%) on the United Kingdom
  4070 Ti. They aren't pool runs. The United Kingdom row keeps its v0.5.11
  figure; the Nevada and Texas rows are now pool runs of PR #256's build.
- **RTX 4060, Australia:** the host runs the card in two clock states. It
  sometimes locks it at 1995 MHz and sometimes leaves it power-capped at
  about 2600 MHz. In the v0.5.11 session all three miners ran at 1995 MHz,
  so the fixed competitor figures come from that state. At
  1995 MHz a speed test of the v0.5.13 kernel, on `79173e2`, read 47.1
  (99%) on 2026-10-07; it isn't a pool run. On 2026-10-08 two later
  rentals ran in the other state, at about 2600 MHz: a pool run of the
  published v0.5.13 release read 61.1, and at 19:18–19:39 UTC PR #256's
  build read 61.9, between release runs of 61.9 and 62.0 (62.0 with the
  width forced to 1024). They can't be compared with this row. On
  2026-10-09 the host was back at 1995 MHz, and the row is that pool run
  of `50b2d6b`. All the later runs had a 115 W limit; the v0.5.11 session
  had 120 W.
- **RTX 4060 Ti, Brazil:** the earlier v0.5.13 pool run (88.7) got no share
  accepted in its 5 minutes, and none rejected. The run in the row got 3.
- **RTX 4070 Super, California (220 W):** in the earlier v0.5.13 pool run
  (141.7) the card ran at a median 81 C, reached 85 C, and slowed for heat
  in 8 of 49 samples. In the `1040668` rental it ran at 70–72 C and only
  the power limit held it, yet every run read about 140.1. In the row's
  run on 2026-10-09 it ran at 84 C and slowed for heat in 15 of 49
  samples, and read 140.0.
- **RTX 4070 Super, California (209 W):** in the row's run the card held
  2445 MHz at 85 C, against 2475 MHz at 79 C for `1040668` on 2026-10-08,
  both at about 208 W. The rate fell by about as much as the clock.
- **RTX 4080 Super, Florida:** PeakMiner exited at once with code 127 and
  wrote nothing (see the 50-series notes). The comparison there is SRBMiner.
  It's a 2-GPU rental. Every miner was pinned to GPU 0, and each one's rate
  matches a single card.
- **RTX 4090D:** one host. The Tanzania host (70632) never started its
  container, in two tries of 16 and 30 minutes, and no other 4090D is listed.

## RTX 50-series (v0.5.14, v0.5.13 and v0.5.11)

Run in the same window as the 40-series.

Seven rows were re-measured later with 5-minute pool runs. The Virginia
5060 Ti ran on 2026-10-08, 13:24–13:34 UTC, with the v0.5.13 release
candidate (`26c50b3`) built as CI builds it, and the Virginia 5060 on
2026-10-07 on `3005e42`. Both have the same sm_120 kernel as v0.5.12 and
v0.5.13, byte for byte. The Colombia 5070 ran on 2026-10-08, 18:35–18:40
UTC, with the published v0.5.13 release. The India 5090, the 5070 Ti on
27661, the Poland 5070 and the Ontario 5060 Ti ran on 2026-10-09,
07:33–10:31 UTC, with PR #256's build at `fec1e8b`, built on each box
with CI's flags. That build has v0.5.14's core for these cards. Its sm_120
kernel is v0.5.13's, byte for byte. Every re-measure ran on the CUDA 13
core except the India 5090's: that host has driver 570, so the CLI loads
the CUDA 12.8 core there. Still from v0.5.11: the South Korea 5090, both
5080s, the 5070 Ti on 28852 and the India 5060.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RTX 5090 | South Korea (500 W) | 145183 | 384.5 | didn't start | 378.0 | 102% |
| RTX 5090 | India (500 W) | 151861 | 358.1 | driver too old | 375.3 | 95% |
| RTX 5080 | New Jersey (360 W) | 152066 | 223.5 | 227.0 | 229.0 | 98% |
| RTX 5080 | Poland (360 W) | 153231 | 223.6 | 227.8 | 228.3 | 98% |
| RTX 5070 Ti | South Korea (250 W) | 27661 | 92.3 | 93.6 | 94.6 | 98% |
| RTX 5070 Ti | South Korea (250 W) | 28852 | 89.2 | 93.6 | 94.5 | 94% |
| RTX 5070 | Colombia (250 W) | 145974 | 132.1 | 132.3 | 133.5 | 99% |
| RTX 5070 | Poland (250 W) | 31379 | 130.7 | 132.7 | 134.2 | 97% |
| RTX 5060 Ti | Virginia (150 W) | 151123 | 94.1 | 94.5 | 94.3 | 100% |
| RTX 5060 Ti | Ontario (180 W) | 153080 | 94.9 | 95.3 | 95.7 | 99% |
| RTX 5060 | Virginia (125 W) | 151478 | 76.8 | 76.8 | 77.1 | 100% |
| RTX 5060 | India (145 W) | 119163 | 68.8 | 76.4 | didn't mine | 90% |

Every host passed the hit check, and every run had 0 rejected shares.

- **Against the top competitor:** 99–100% on the three hosts re-measured
  with v0.5.13's kernel, and 95–99% on the four re-measured with PR #256's
  build. The v0.5.11 run was at 98–102% on the 5090 and 5080, 94–95% on the
  5070 Ti and 5070, and 90–93% on the 5060 Ti and 5060. On the power-capped
  cards below the 5080, v0.5.11 ran 60–185 MHz slower than the other two at
  the same power. It used more power per clock, so the card clocked down to
  stay under its cap. That's where the gap on the smaller cards came from.
  On v0.5.13's kernel, the two Virginia cards, the Colombia 5070 and the
  Ontario 5060 Ti (in its 2026-10-08 run) ran 30–53 MHz faster than the
  other two did.
- **On v0.5.11:** the re-measured rows read 369.0 (98%) on the India 5090,
  89.5 (95%) on the 5070 Ti on 27661, 126.1 (94%) on the Poland 5070,
  88.0 (93%) on the Virginia 5060 Ti, 87.6 (92%) on the Ontario 5060 Ti,
  70.7 (92%) on the Virginia 5060 and 125.7 (94%) on the Colombia 5070.
- **On v0.5.13:** on 2026-10-08 the published release read 95.3 (100%) on
  the Ontario 5060 Ti, at 2715 MHz and 76 C. The row's run held 2700 MHz
  at 77 C.
- **Speed tests, not pool runs:** on 2026-10-07 a speed test of the v0.5.13
  kernel read 229.7 (101%) on the Poland 5080, on `3005e42`, and 92.3 (98%)
  on the 5070 Ti on 27661, on `79173e2` with a patch and a build flag. They
  aren't pool runs. The Poland 5080 row keeps its v0.5.11 figure; the 5070
  Ti row on 27661 is now a pool run of PR #256's build.
- **RTX 5090, India:** the row's run held 2250 MHz at the 500 W limit,
  against 2340 MHz in the v0.5.11 session, also at 500 W. Per clock it did
  0.9% more work than v0.5.11, so the rate fell because the clock did. These
  runs don't show whether the host or the newer kernel set the lower clock.
- **RTX 5070 Ti on 27661 and RTX 5070, Poland:** before our miner started
  on 2026-10-09, each GPU already showed 100% use: at 28 W on the 5070 Ti,
  and at 43 W and 2925 MHz on the 5070. Another workload may share these
  cards.
- **Which core we load:** on a 50-series card with driver 580 or newer, the
  CLI loads the CUDA 13 core. The India 5090 has driver 570, so it loaded the
  CUDA 12.8 core. In the v0.5.11 run the South Korea 5090, on CUDA 13, ran
  4% faster than the India 5090 at the same power and clock, while SRBMiner
  scored within 1% on both. That points to the CUDA 13 core being faster on
  Blackwell, but it's two different hosts.
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
  94–99 W for the other two; in the speed test on 27661 it drew 87–88 W,
  and in the row's pool run 87 W at 1342 MHz.
  Two 300 W hosts were tried and couldn't attach the GPU.
- **Not tested:** the RTX 5090D, which no host lists, and laptop GPUs (3060,
  4070 and 4080 laptop), which are on Vast but left out.

## A100-class (sm_80, v0.5.14, v0.5.13 and an earlier build)

Run on 2026-10-07, 14:53–22:52 UTC. The A100 PCIe and the New York A100
SXM4 ran `3804a7c` and the A30 `1f699da`, each with PeakMiner and SRBMiner
re-run on the same box in the same session. Both builds have the same sm_80
kernel as v0.5.13, byte for byte. The CMP 170HX and the California A100
SXM4 ran the first build, `feba1ca`, whose sm_80 kernel differs.
v0.5.13 is the first release that supports these cards. Same harness, pool
and miner versions as the RTX tables.

The three A100s were re-measured on 2026-10-09, 11:11–11:18 UTC, with
5-minute pool runs of PR #256's build at `50b2d6b`, built on each box with
CI's flags. That build has v0.5.14's core for these cards. Its sm_80 kernel
is v0.5.13's, byte for byte. The A30 is still from `1f699da` and the CMP
170HX from `feba1ca`.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| A100 PCIe 40 GB | Japan (250 W) | 146405 | 209.5 | 212.2 | 213.9 | 98% |
| A100 SXM4 40 GB | New York (400 W) | 152837 | 244.5 | 252.2 | 254.7 | 96% |
| A100 SXM4 40 GB | California (400 W) | 152356 | 188.1 | 197.4 | 199.8 | 94% |
| A30 24 GB | Australia (155 W) | 145742 | 104.7 | 98.1 | 96.2 | 107% |
| CMP 170HX | Georgia (250 W) | 147823 | 146.9 | 167.9 | 165.0 | 87% |

- **A100 PCIe and SXM4:** in the same-session runs, all three miners ran at
  the power limit. SRBMiner does 4.0–4.3% more work per clock than we do,
  and we run a 1–2.6% higher clock at the same watts, so the gap is work per
  clock. On `feba1ca` the PCIe read 192.0 (89%) and on `dd64b2c` the SXM4
  read 240.3 (95%). The copy points (`dd64b2c`), the hash in its own kernel
  (`1f699da`) and band 32 (`3804a7c`) closed most of it.
- **A100 PCIe, re-measured:** in the session with PeakMiner and SRBMiner,
  on 2026-10-07 on `3804a7c`, it read 209.5 (98%). This host runs the card
  at about 1170 MHz in some sessions and about 1200 MHz in others, at the
  same 250 W, and the same build reads about 209 and 213. On 2026-10-08, in
  a 1200 MHz session, `26c50b3` read 212.8 (99%). The row's run on
  2026-10-09 was at 1170 MHz, the same state as the session PeakMiner and
  SRBMiner ran in (at 1136–1143 MHz themselves).
- **CMP 170HX:** all three miners ran at the 250 W limit on `feba1ca`, when
  we got about 70% of the tensor peak per clock to their 85%. A speed test
  of the v0.5.13 kernel on this host read 164.6 (98%) on 2026-10-07, on
  `e07b1ab` at the same 250 W, and an earlier one of the same kernel read
  165.0. They aren't pool runs, so the row keeps its `feba1ca` figure.
- **A100 SXM4, New York:** all three miners ran at the 400 W limit, at
  75–77 C. This is the row that shows the card at its rated power. In that
  session, on 2026-10-07 on `3804a7c`, ours read 246.5 (97%). The row's
  run on 2026-10-09 was also at the 400 W limit, at 1380 MHz and 70 C.
- **A100 SXM4, California:** the host cools the card poorly. All three miners
  ran at 85 C and the thermal limit, drawing about 215 W of the 400 W allowed,
  so this row says more about the host than the card. In that session, on
  2026-10-07 on `feba1ca`, ours read 177.3 (89%). The row's run on
  2026-10-09 was held by heat too: 84 C, 1050 MHz and about 194 W, slowed
  for heat in every sample. The first SXM4 host tried (149846) failed
  during setup.
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

## Hopper (sm_90a, v0.5.14 and a build before v0.5.13)

Run on 2026-10-08, 06:35–07:17 UTC, with PR #253's release build at `e8c04b4`,
which mines with Hopper's wgmma fold by default. Each box built the core as
native-core.yml does (CUDA 12.8, sm_75/80/86/89/90a/120, no `-D` flags).
v0.5.13 supports these cards, with a faster fold than this build (see "Since
`e8c04b4`" below). Same harness, pool and miner versions as the RTX tables.
PeakMiner and SRBMiner on the H100 SXM and NVL hosts are the 2026-10-07
runs; the other three hosts are new, so all three miners ran there.

Four rows were re-measured on 2026-10-09, 00:33–00:48 UTC, with 5-minute
pool runs of PR #256's build at `466ccf0`, which has v0.5.14's Hopper
fold: the H100 NVL, the H100 PCIe, the H200 and the H200 NVL. Each box
built it as native-core.yml does and ran the hit check and the pool run on
that core. The boxes built a local commit that differs from `466ccf0`
only in comments.
Its sm_90a fold is not v0.5.13's: it runs the fold's three-warpgroup form
(see "After v0.5.13" below). The H100 SXM is still from `e8c04b4`.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| H100 SXM 80 GB | New York (700 W) | 153443 | 630.8 | 757.7 | 734.4 | 83% |
| H100 NVL 94 GB | Japan (400 W) | 29785 | 537.9 | 533.0 | 537.6 | 100% |
| H100 PCIe 80 GB | Czechia (350 W) | 147981 | 490.7 | 391.6 | 481.0 | 102% |
| H200 141 GB | Saudi Arabia (700 W) | 131919 | 719.9 | 718.6 | 720.4 | 100% |
| H200 NVL 141 GB | Quebec (600 W) | 153365 | 682.9 | 669.8 | 675.3 | 101% |

- **Every host:** passed the hit check and picked the wgmma fold with 2-CTA
  clusters: 192x256 tiles on `466ccf0`, 128x256 in the H100 SXM's
  `e8c04b4` run. The CLI got 26, 19, 21, 20 and 23 shares accepted, none
  rejected.
- **The gap is work per clock.** Every miner ran at the power cap. On
  `e8c04b4` we ran 70–290 MHz faster than the other two at the same watts.
  Per clock that fold did 71–75% of the wgmma peak and the competitors
  87–94%; most of the difference was the fold's readout. On the four hosts
  re-measured on v0.5.13, the fold does 9–13% more work per clock than on
  `e8c04b4`. The card clocks 30–60 MHz lower at the same cap, which is still
  30–250 MHz faster than the other two, and per clock we do 90–92% of
  SRBMiner's work.
- **On v0.5.13:** before PR #256's build, the re-measured rows read, on
  2026-10-08 with the published release: 505.1 (94%) on the H100 NVL,
  454.4 (94%) on the H100 PCIe and 627.4 (93%) on the H200 NVL at
  17:03–17:09 UTC, and 669.7 (93%) on the H200 at 18:21–18:26 UTC.
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
- **After v0.5.13:** v0.5.14 (PR #256) makes the fold's three-warpgroup
  form the default (192x256 tiles, band 8; `PEARL_HOPPER_WG3` in
  `pearl_config.h`). In the table's pool runs of `466ccf0`, the H100 NVL
  read 6.5% more than in its v0.5.13 pool run, the H100 PCIe 8.0%, the
  H200 7.5% and the H200 NVL 8.8%. Each compares two rentals of the same
  host. The H200 is at 99.9% of SRBMiner, which rounds to 100%.
- **Speed tests of the new fold:** `hashrate.js` speed tests on 2026-10-08,
  22:46–23:39 UTC, of a build with `466ccf0`'s sm_90a kernel against
  v0.5.13's in the same rental, 3 rounds each, ahead in every round: 547.1
  against 501.0 (+9.2%) on the H100 NVL (Japan, 29785, this table's host);
  and 729.3 against 670.8 (+8.7%) on an H100 SXM (California, 152422,
  700 W; New York 153443 had no offer), 96.3% of PeakMiner's 757.7. The
  H100 NVL's pool run read 1.7% below its speed test, at the same 1155 MHz.
  The SXM test is a different method on a different host, so the H100 SXM
  row is unchanged. 152422 blocks outbound port 1200, so it can't do a pool
  run.
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

## Ada workstation and data-center cards (sm_89, v0.5.14 and v0.5.13)

Run on 2026-10-07, 16:55–21:45 UTC, with PR #253's core at `5f427dd`,
`54c993d` or `e0bee03`. Their sm_89 kernels are byte-identical to each other
and to v0.5.13's, and PR #253 made no change for these cards. They run the
same sm_89 code as the RTX 40-series. Same harness, pool and miner versions as
the RTX tables.

Ten rows were re-measured on 2026-10-08, 19:23–20:17 UTC, with 5-minute
pool runs of PR #256's build at `1040668`, built on each box as for the
RTX 40-series. That build has v0.5.14's core for these cards. It has
v0.5.13's sm_89 kernel and changes only the batch width the host picks
from the card's L2 (see "Ada batch width (v0.5.14)" below). Still from
PR #253's core: the Germany RTX 6000 Ada, the New York L40 and the France
RTX 4000 Ada.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| L40S | Texas (350 W) | 96563 | 288.2 | 257.1 | 281.8 | 102% |
| L40S | Taiwan (350 W) | 151291 | 288.6 | 250.8 | 270.7 | 107% |
| RTX 6000 Ada | New York (300 W) | 142572 | 228.1 | 195.9 | 210.4 | 108% |
| RTX 6000 Ada | Germany (300 W) | 146274 | 194.2 | 167.5 | 179.2 | 108% |
| RTX 5000 Ada | Minnesota (250 W) | 151333 | 222.3 | 193.4 | 201.1 | 111% |
| L40 | Vietnam (300 W) | 114318 | 172.9 | 166.5 | 170.4 | 101% |
| L40 | New York (250 W) | 56199 | 153.6 | 131.8 | 139.0 | 111% |
| RTX 4500 Ada | France (210 W) | 148971 | 148.1 | 134.3 | 145.1 | 102% |
| RTX 4000 Ada | Norway (130 W) | 149238 | 97.7 | 94.0 | 95.0 | 103% |
| RTX 4000 Ada | France (130 W) | 152701 | 93.1 | 91.3 | 91.8 | 101% |
| L4 | Czechia (72 W) | 116596 | 79.5 | 74.4 | 77.8 | 102% |
| L4 | Washington (72 W) | 152541 | 80.0 | 74.2 | 77.2 | 104% |
| RTX 2000 Ada | Romania (70 W) | 68269 | 48.5 | 47.1 | 43.6 | 103% |

- **Before PR #256's build:** on 2026-10-07, with PR #253's core, the
  re-measured rows read 288.4 (102%) and 288.9 (107%) on the Texas and
  Taiwan L40S, 227.5 (108%) on the New York RTX 6000 Ada, 215.4 (107%) on
  the RTX 5000 Ada, 173.8 (102%) on the Vietnam L40, 146.7 (101%) on the
  RTX 4500 Ada, 96.6 (102%) on the Norway RTX 4000 Ada, 79.4 (102%) and
  78.5 (102%) on the Czechia and Washington L4s, and 47.8 (102%) on the
  RTX 2000 Ada.
- **The release in the same rental:** each box ran the v0.5.13 release
  before and after PR #256's build (A1 and A2, in "Ada batch width
  (v0.5.14)" below).
  - Both L40S and the L40 went down 0.1–0.5%. The build keeps the
    release's own width there, and the release read about the same as the
    build: 288.7 and 288.1 in Texas, 288.6 and 288.5 in Taiwan, and 172.8
    and 172.9 on the L40.
  - The RTX 5000 Ada's rise from 215.4 is mostly the host: the release read
    221.7 and 221.6 in the same rental, and the build added 0.3%.
  - On the Czechia L4 the release read 78.3 and 78.0, 1.5% below
    2026-10-07, so the build's 1.7% gain moved the row only from 79.4 to
    79.5.
- **Every host:** every miner ran at the power limit. We ran 30–310 MHz
  faster than the other two at the same watts, so we use less power for the
  same work. Per clock we get about 96% of the tensor peak and SRBMiner about
  97%.
- **L40:** half the L40S's int8 tensor rate per SM, so its rates are lower.
  Every other card here has the full rate, the L4 included. The New York
  host sets the L40 to 250 W; its maximum is 300 W.
- **Hosts matter:** at the same 300 W, the Germany RTX 6000 Ada ran at
  83–85 C and about 1400 MHz, and the New York one at 77–79 C and 1640 MHz,
  so every miner was about 17% faster in New York. On 2026-10-07 the RTX
  4000 Ada read 4% higher in Norway (74 C) than in France (85 C).
- **RTX 2000 Ada:** PeakMiner is the faster competitor here.
- **Replaced hosts:** L4 Brazil (151023) blocks outbound port 1200, so the
  competitors couldn't run. L4 Utah (109523) is thermally limited: it drew
  about 50 W of its 72 W at 87–88 C, and every miner's rate swung by ±4 TH/s.

## Ada batch width (v0.5.14)

PR #256's `1040668` ports a host-only rule from PR #255 for the sm_89 tall
fold. At v0.5.13's batch width (`col_batch` 2048), one launch keeps
re-reading 64 MB of one input (B') and a 6 MB band of the other (A'). When
the L2 can't hold that, B' comes back from memory once a band, and on a
power-capped card that memory power comes out of the SM clock. The rule
halves `col_batch` until B' and one band fit the L2 the card reports. On
these cards that gave 512 on 24 and 32 MB, 1024 on 40–64 MB, and 2048,
v0.5.13's width, on 72 MB and up. The kernel is unchanged. The rule ships
in v0.5.14. Below, "the release" is v0.5.13.

Each host ran three to five 5-minute pool runs in one rental on 2026-10-08,
19:18–20:53 UTC: the v0.5.13 release (A1), `1040668` (B), on some hosts
`1040668` with the width forced to another value (C), then the release
again (A2). B and C were built on the box (see the RTX 40-series intro);
C also set `-DPEARL_ADA_COL_BATCH=N`. Every built core passed the hit
check, and every run had 0 rejected shares. "B vs release" is B over the
mean of A1 and A2, and "A2 vs A1" is how far the release itself moved in
the rental.

| Card | Host | L2 | Width | Release, A1 / A2 | B | B vs release | A2 vs A1 | C (vs release) |
|---|---|---|---|---|---|---|---|---|
| RTX 4060 | New Zealand | 24 MB | 512 | 60.2 / 59.9 | 60.3 | +0.4% | -0.5% | 1024: 60.2 (+0.2%) |
| RTX 4060 | Australia | 24 MB | 512 | 61.9 / 62.0 | 61.9 | 0.0% | 0.0% | 1024: 62.0 (0.0%) |
| RTX 2000 Ada | Romania | 24 MB | 512 | 47.9 / 47.7 | 48.5 | +1.4% | -0.2% | 256: 48.4 (+1.2%); 512: 48.3 (+1.1%) |
| RTX 4060 Ti | Ontario | 32 MB | 512 | 85.8 / 85.9 | 85.9 | +0.1% | +0.1% | 1024: 86.0 (+0.2%) |
| RTX 4060 Ti | Brazil | 32 MB | 512 | 88.7 / 88.7 | 89.0 | +0.4% | 0.0% | 1024: 89.0 (+0.4%) |
| RTX 4000 Ada | Norway | 40 MB | 1024 | 96.6 / 96.6 | 97.7 | +1.1% | 0.0% | 512: 97.9 (+1.3%) |
| L4 | Czechia | 48 MB | 1024 | 78.3 / 78.0 | 79.5 | +1.7% | -0.4% | 512: 79.1 (+1.2%) |
| L4 | Washington | 48 MB | 1024 | 78.7 / 78.6 | 80.0 | +1.7% | -0.1% | |
| RTX 4500 Ada | France | 48 MB | 1024 | 147.1 / 147.0 | 148.1 | +0.7% | 0.0% | |
| RTX 4070 Super | California (209 W) | 48 MB | 1024 | 135.4 / 135.3 | 136.0 | +0.5% | -0.1% | |
| RTX 4070 Super | California (220 W) | 48 MB | 1024 | 140.1 / 140.1 | 140.1 | 0.0% | 0.0% | |
| RTX 4080 | Taiwan | 64 MB | 1024 | 190.2 / 190.2 | 190.4 | +0.1% | 0.0% | 512: 190.1 (0.0%) |
| RTX 4080 | Nevada | 64 MB | 1024 | 197.7 / 197.4 | 197.8 | +0.1% | -0.2% | |
| RTX 4080 Super | Florida | 64 MB | 1024 | 198.9 / 197.9 | 198.5 | 0.0% | -0.5% | |
| RTX 4080 Super | California | 64 MB | 1024 | 198.0 / 198.0 | 198.6 | +0.3% | 0.0% | |
| RTX 5000 Ada | Minnesota | 64 MB | 1024 | 221.7 / 221.6 | 222.3 | +0.3% | -0.1% | |
| RTX 4090 | Australia | 72 MB | 2048 | 317.4 / 317.4 | 317.7 | +0.1% | 0.0% | 1024: 318.3 (+0.3%) |
| L40S | Texas | 96 MB | 2048 | 288.7 / 288.1 | 288.2 | -0.1% | -0.2% | 1024: 288.1 (-0.1%) |
| L40S | Taiwan | 96 MB | 2048 | 288.6 / 288.5 | 288.6 | 0.0% | 0.0% | 1024: 288.2 (-0.1%) |
| L40 | Vietnam | 96 MB | 2048 | 172.8 / 172.9 | 172.9 | 0.0% | 0.0% | 1024: 172.8 (-0.1%) |
| RTX 6000 Ada | New York | 96 MB | 2048 | 227.9 / 228.2 | 228.1 | 0.0% | +0.1% | 1024: 226.1 (-0.9%) |

- **Where the rule narrows the batch,** it gained 0.7–1.7% on the
  power-capped workstation and data-center cards (both L4s and the RTX
  2000, 4000 and 4500 Ada) and 0.5% on the 209 W 4070 Super. Their median
  SM clock rose 15–45 MHz at the same power. On the other cards it narrows,
  B was 0.0–0.4% above the release, within the 0.5% the release itself
  moved between A1 and A2 on two hosts.
- **At 72 MB and up** the rule keeps 2048, so B runs the release's batch
  and read within 0.1% of it.
- **Forced widths:** none beat the rule's pick by more than 0.2% (1024 on
  the 4090 and the Ontario 4060 Ti, 512 on the RTX 4000 Ada). 1024 instead
  of 2048 cost 0.9% on the RTX 6000 Ada, and 512 instead of 1024 cost 0.5%
  on the Czechia L4. The RTX 2000 Ada's C at 512 is the rule's own width,
  forced; it read 0.3% below B, which is about the run-to-run noise.
- **RTX 4060, Australia:** ran in its 2600 MHz state, so these runs don't
  set its row (see the 40-series notes).

## RTX PRO Blackwell (sm_120, v0.5.14 and v0.5.13)

Run on 2026-10-07, 19:20–21:15 UTC, with PR #253's core at `54c993d` or
`e0bee03` (their sm_120 kernels are byte-identical to each other and to
v0.5.13's), on the CUDA 13 core, as a rig with driver 580 or newer loads it.
No change was made for these cards. Same harness, pool and miner versions as
the RTX tables.

Two rows were re-measured on 2026-10-09, 09:51–10:06 UTC, with 5-minute
pool runs of PR #256's build at `50b2d6b`, built on each box with CI's
flags for the CUDA 13 core: the RTX PRO 6000 Server and the RTX 6000D. That
build has v0.5.14's core for these cards. Its sm_120 kernel is v0.5.13's,
byte for byte.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RTX PRO 6000 Workstation | Mississippi (600 W) | 148552 | 397.0 | 389.9 | 393.9 | 101% |
| RTX PRO 6000 Server | California (600 W) | 150951 | 375.7 | 377.2 | 381.5 | 98% |
| RTX PRO 6000 Max-Q | Illinois (300 W) | 153399 | 311.4 | 307.4 | 311.5 | 100% |
| RTX PRO 5000 48 GB | California (250 W) | 150611 | 211.4 | 208.7 | 212.8 | 99% |
| RTX PRO 4500 | Arizona (200 W) | 148915 | 175.5 | 170.8 | 172.9 | 101% |
| RTX PRO 4000 | Pennsylvania (145 W) | 150214 | 127.4 | 126.2 | 126.9 | 100% |
| RTX 6000D | Czechia (550 W) | 149788 | 144.6 | 144.8 | 144.8 | 100% |

- **Per clock:** we get 94–96% of the tensor peak on every card, and SRBMiner
  97–98%. At the power cap we run 2.6–4.1% faster at the same watts, which
  is enough on most cards. The RTX PRO 5000 (2.6%) is 0.7% short, and the
  6000 Max-Q (99.97%) and the RTX 6000D (99.9%) round to 100%.
- **Earlier figures:** on 2026-10-07, on `e0bee03`, ours read 378.8 (99%)
  on the RTX PRO 6000 Server and 144.6 (100%) on the RTX 6000D.
- **RTX PRO 6000 Server:** a passive server card. Once it reached 85 C it
  held about 2065 MHz and 440 W of its 600 W, for every miner, so heat, not
  power, set its rate on this host. In the row's run on 2026-10-09 it held
  2055 MHz and about 438 W at 85 C, and the row is 1.5% short of SRBMiner.
  The driver reported no slowdown reason in 48 of the 49 samples.
- **RTX PRO 5000 48 GB:** the host sets 250 W, 83% of the card's 300 W.
- **RTX 6000D:** Vast's name for a cut-down card with 156 SMs and 84 GB. Its
  int8 tensor rate is capped at about 37% of the RTX PRO 6000's per SM per
  clock. All three miners tie at its top clock and about 253 W. The row's
  run on 2026-10-09 held the same 2422 MHz at about 264 W.
- **Not in the table:** the RTX PRO 5000 72 GB. Its only host (California,
  153314) blocks outbound port 1200, so no miner could reach the pool. Our
  speed test read 239.4 at 300 W. Vast lists no RTX PRO 2000 or 4000 SFF.

## Ampere workstation and data-center cards (sm_86, v0.5.13)

These cards run the same sm_86 code as the RTX 30-series. The A4000 rows ran
on 2026-10-07, 16:53–17:47 UTC, with PR #253's core at `5f427dd`, whose sm_86
kernel is byte-identical to v0.5.13's. The other eight ran on 2026-10-09,
04:54–05:24 UTC, from the published v0.5.13 files, checked against the
release digests. v0.5.14 doesn't change this kernel: all 20 sm_86 functions
built from PR #256's `50b2d6b` match the published v0.5.13 core. Same
harness, pool and miner versions as the RTX tables.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RTX A6000 | Italy (300 W) | 47281 | 123.4 | 122.3 | 122.6 | 101% |
| RTX A6000 | Kansas (300 W) | 38333 | 127.7 | 127.9 | 127.8 | 99.8% |
| A40 | United Kingdom (300 W) | 152158 | 123.2 | 123.1 | 120.5 | 100% |
| A40 | Belgium (300 W) | 150273 | 124.5 | 124.2 | 125.2 | 99% |
| RTX A5000 | North Carolina (230 W) | 147999 | 93.8 | 92.1 | 87.0 | 102% |
| RTX A5000 | Italy (200 W) | 151139 | 84.5 | 83.2 | 81.8 | 102% |
| RTX A4000 | Kazakhstan (140 W) | 147920 | 59.5 | 58.2 | 56.6 | 102% |
| RTX A4000 | Kentucky (140 W) | 29102 | 56.9 | 55.8 | 53.5 | 102% |
| RTX A2000 12 GB | Poland (70 W) | 28227 | 29.2 | 28.1 | 28.8 | 101% |
| RTX A2000 6 GB | Japan (70 W) | 152199 | 27.7 | 28.5 | 28.1 | 97% |

- **Every host:** passed the hit check, and every miner sat at the power
  limit after the first minute. No miner had a share rejected.
- **Order:** on the 2026-10-09 hosts, the first host of each card ran ours,
  PeakMiner, SRBMiner; the second ran SRBMiner, PeakMiner, ours.
- **RTX A5000, Italy:** the host sets 200 W, 87% of the card's 230 W stock
  limit. SRBMiner's rate swung between 75 and 84.
- **RTX A2000s:** at their speed, the 90 s hit check found only 264 (Poland)
  and 248 (Japan) hits, all correct. A 240 s rerun on each machine found 764
  and 736, and the first 400 of each checked out. Poland's rerun was a
  different offer on the same machine, with a x4 PCIe link against x16, so
  it probably ran on another card of the same model.
- **RTX A2000 12 GB, Poland:** neither our miner nor SRBMiner got a share
  accepted in its 5 minutes. Both rates are the miners' own figures.
- **RTX A2000 6 GB, Japan:** the card ran at 86–91 C for every miner.
  PeakMiner averaged 28.22 over its run, against which we'd be 98%. The box
  logged only every 15th CLI reading; those average 27.91, against 27.69 for
  the box's mean over all 217.
- **A40, Belgium:** a 2-GPU rental, with every miner pinned to GPU 0. GPU 1
  stayed idle throughout.
- **No RTX A4500 row:** its only host (Czechia, 149635) sets 130 W of the
  card's 200 W, below the power rule.
- **A4000, both hosts:** every miner ran at the 140 W limit. We ran 60–110
  MHz faster than the other two.
- **Kentucky:** the card runs at 92–94 C. In this 5-minute run all three
  miners stayed at the power limit, but in our longer speed tests the card
  hit its thermal limit and read 54.7–55.1 TH/s.
- **PeakMiner, Kazakhstan:** got no share accepted in its 5 minutes. The
  rate is its own figure.
- **Replaced host:** Germany (14335). Another workload was using the GPU,
  and the tensor probe read 37 T-MAC/s against 92 on a clean card.

## B200 (sm_100, v0.5.14)

Run on 2026-10-09, 05:28–06:09 UTC, with PR #256's build at `ec0e38d`,
which has v0.5.14's sm_100 core. The box built it as native-core.yml does:
CUDA 12.8, v0.5.13's architectures plus sm_100. Our miner ran from source
with `PEARL_CORE_PATH` set to that core; the app picks the same CUDA 12.8
core for a compute 10.0 card. Same pool and miner versions as the RTX tables.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| B200 180 GB | Oregon (1000 W) | 96942 | 460.2 | 1376.9 | 1384.1 | 33% |

- **Hit check:** 400/400, in both operand orders.
- **Why it's slow:** the B200 runs the TMA tall fold that RTX 50 cards mine
  on, built for sm_100. That fold uses `mma.sync`, and a probe on this card
  topped out at about 568 TH/s with it, 41% of SRBMiner. PeakMiner and
  SRBMiner did about 6,300 int8 MACs per clock per SM, three times what
  `mma.sync` reached, so they use the B200's own tensor instructions
  (tcgen05). Matching them needs a tcgen05 fold.
- **Clocks and power:** ours ran at 1965 MHz and 872 W, under the 1000 W
  limit. PeakMiner and SRBMiner ran at the limit, at 1470 and 1492 MHz.
- **Speed tests** (`hashrate.js`, 3 rounds): the shipped fold 461.6, and the
  cp.async fold (`-DPEARL_TALL_TMA=0`) 468.8, 1.5% faster in every round.
  Batch widths of 1024 and 512 were level with 2048.
- **Shares:** 19 accepted for ours, 46 for PeakMiner and 43 for SRBMiner,
  none rejected.
- **Order:** ours, then PeakMiner, then SRBMiner. The competitors' figures
  are their last reading, at 295 s.
- **Host:** the cheapest 1-GPU offer, reliability 0.997, driver 580.126.09,
  148 SMs and 126.5 MiB of L2.

## Tesla T4 (sm_75, v0.5.13)

Run on 2026-10-09, 08:48–08:53 UTC, from the published v0.5.13 files, checked
against the release digests. v0.5.14's sm_75 code is byte-identical to
v0.5.13's. This was one 5-minute pool run of our miner only: PeakMiner and
SRBMiner weren't run and there was no hit check, so there is no "% of best"
and no table row.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| Tesla T4 16 GB | Czechia (70 W) | 28909 | 27.7 | not run | not run | — |

- **Ours:** 27.65 TH/s, the mean of the CLI's 188 readings after the first
  minute (median 27.5, range 26.8–29.1). Per minute: 30.2, 28.3, 27.5, 27.4,
  27.4. The rate fell as the card warmed and held about 27.4 from minute 4.
  No share was accepted or rejected in the 5 minutes; about 1.5 were
  expected at this rate.
- **Core:** the CLI loaded the CUDA 12.8 core ("GPU 0 is compute 7.5 (the
  CUDA 13 build has no code for it)"), on driver 580.178.04.
- **Clocks and power:** the T4's 70 W limit is its default and maximum. The
  card ran at a median 750 MHz of its 1590 MHz maximum, 67.5 W and 73 C, at
  the power cap (0x4) in every sample after the first minute.
- **Per clock:** at a steady 750 MHz, 27.4 TH/s is 913 int8 MACs per clock
  per SM, 89% of Turing's IMMA peak, in line with the 20-series cards.

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
