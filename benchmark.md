# Benchmarks

Our miner against PeakMiner and SRBMiner on rented GPUs. The baseline is a full run on Vast on 10–11 October 2026 with
v0.5.14. Later releases didn't change those cards' kernels, so the figures still hold. From 28 October the consumer
cards were run again on Salad PCs with v0.6.9 against newer competitors. Each section also lists the experiments we
raced against our own release. Before it was shortened, this file held every run's full notes:
[benchmark.md at 16192d68](https://github.com/super3/llmjob/blob/16192d68f40b9c7ba11048e1128e7ba2c5509dca/benchmark.md).

## Summary

"% of best" is our rate divided by the faster of PeakMiner and SRBMiner on the same host.

| Cards | Vast, 10–11 October (v0.5.14) | Latest Salad run (v0.6.9) | Note |
|---|---|---|---|
| RTX 20-series | 92–99% | 95.7–99.7%, median 97.4% (30 Oct, SRBMiner 3.8.2) | PeakMiner fastest on every PC |
| RTX 30-series | 97–101% | 91–97%, median 92.5% (28 Oct) | Our weakest series now |
| RTX 40-series | 98–102% | 93.5–95.8%, median 94.5% (29–30 Oct) | Competitors got faster, we didn't |
| RTX 50-series | 96–101% | 87.6–103.2%, median 95.2% (30 Oct, SRBMiner 3.8.2) | 5090 trails most |
| A100-class | 97–111% | - | A30's 111% is too high; about 107% |
| Hopper | 98–99% | - | |
| Ada workstation and data-center | 96–100% | - | RTX 6000 Ada at 96% |
| RTX PRO Blackwell | 95–100% | - | |
| Ampere workstation and data-center | 89–100% | - | A2000s 89–91%, A4000s 93–94% |
| B200 | 33% | - | Needs a tcgen05 fold |
| Tesla T4 | 91% | - | |
| GTX 10-series | not run | - | Competitors mine 0 |
| GTX 16-series | 96–104% (Salad, 17 Oct) | - | 0.7–1.3 TH/s; not worth mining |
| AMD Radeon | - | 92–110% against SRBMiner 3.8.0 (30 Oct) | PeakMiner mines 0; v0.6.12 adds about 13% on RDNA 4 |

**Why we trail where we do.** On most Ampere and Ada hosts and the RTX PRO 6000 Workstation, SRBMiner held a higher
SM clock than ours at the same power, while per clock the two were within 2%. On the RTX 20-series, the T4, the A100s
and the H100s the clocks matched or ours was higher, so the gap there is work per clock.

## How a host is tested

1. **Files.** Each box downloads the published release files and checks their sha256 digests. PeakMiner and SRBMiner
   come from their own releases and run at stock. A compute 12.0 card with driver 580 or newer loads the CUDA 13 core;
   every other card loads the CUDA 12.8 core.
2. **Hit check.** `earn/native/probes/verify-hits.js` runs the core for 90 s and recomputes the first 400 hits the way
   the pool's verifier would. Every one must match. It checks correctness, not speed, and every row passed. Three slow
   cards (two RTX A2000s and the T4) needed longer runs to reach 400.
3. **Runs.** After a 60 s warm-up, the three miners each mine for 5 minutes on `us.pearl.herominers.com`, back to back
   on the same card. The order alternates between hosts, since whoever runs first gets the coolest card. nvidia-smi
   samples clock, power, temperature and use every 5 s. On a multi-GPU rental every miner is pinned to GPU 0.

Our rate is the mean of every reading after the first minute. A competitor's rate is its own API figure at the end of
its 5 minutes. A competitor that didn't run has the reason in its cell, and the row compares with the other one.
Clocks and power in the notes are medians after the first minute. "% of best" is computed from unrounded rates, and
shows one decimal from 99.5% to just under 100%, so a row that's behind never shows as 100%.

Every row was checked against the files its host sent back. On nine rows (the Ukraine 3090, Portugal 3080 Ti, RTX
4090D, the three A100s and the three H100s) Vast's 500-character log limit lost the raw files, so those figures come
from the box's own result line, checked against its per-minute log lines. Our miner had no share rejected on any row.

A re-measure is a pool run of only our miner, as above, on the row's own machine; the row keeps its PeakMiner and
SRBMiner figures. A speed test (`hashrate.js`, no pool) isn't a re-measure: it goes in the notes and never changes a
row.

### Salad runs

Salad PCs are home computers that run containers under WSL2, named by the first 8 characters of the machine ID.
Salad doesn't say where a PC is. Each PC runs the three miners one after another, 5 minutes each after a 60-second
warm-up, on `us.pearl.herominers.com` (`tuning/salad/bench.py` and `bench.sh`). A card's first PC runs ours first and
its second runs SRBMiner first. Release files are checked against their published checksums, and there is no hit
check. These are different PCs from the Vast rows, tested by a different method, so they never change those rows:
compare "% of best", not TH/s.

"Our own builds" sections race builds of our core against the release on the same card, either in race mode
(`earn/native/probes/race.cu`, every build in one process taking 4 s turns, each scored against the release's turn in
the same round) or with `hashrate.js` turns. No competitor runs, so they are never rows.

## How hosts are picked

- The driver supports CUDA 12.8.
- Reliability is 90% or higher.
- The power limit is at least 80% of the card's stock limit.
- Vast's performance score is at least half the median for that card, which screens out fake hosts.
- Not in mainland China.
- Cheapest first. A host already tested wins when it's listed, and a card's second host is from another operator.
- One GPU per rental where possible.

A row stays on its own machine when that machine can be rented; otherwise it moves to the cheapest offer that meets
these rules. On 10 October 54 rows ran on their own machine and 38 on new hosts. Exceptions: own machines stay even
where they no longer meet the rules (the 170 W Thailand 2080 Ti, the India 5090 at 58% reliability); the British
Columbia 4080 runs at 225 W and the France 4070 Ti is unverified, because nothing else was for rent. Four rows are on
2-GPU boxes.

## Keeping this file current

- A change that re-measures a host in this file updates its row in the same PR. A number that's only in chat or a PR
  description doesn't count as recorded.
- Only a run of our miner on the row's own host (same machine ID) replaces its "Ours" and "% of best". Give the earlier
  figure in that section's notes, with the date and the commit that ran.
- PeakMiner and SRBMiner figures are fixed targets until the next full rerun. Don't rerun them for a row that has them.
  A new host runs all three miners.
- A run on a different host, or by a different method than "How a host is tested", goes in the section's notes and
  leaves the row alone.
- When a row changes, check that the Summary and the section heading still match it.
- Before tagging a release, check that the intro, the Summary and "How a host is tested" name the right release, and
  that no section calls a shipped card unreleased.

The 10 October run on v0.5.14 reran all three miners on 92 of the 95 rows and is this file's baseline. On 11 October
six RTX 20-series rows were rerun against SRBMiner 3.7.5; the other sections compare with SRBMiner 3.7.4.

## RTX 20-series

Six rows were rerun on 11 October on their own machines, against PeakMiner 2.18.3 and SRBMiner 3.7.5. The
Pennsylvania 2080 Ti and Colorado 2080 keep their 10 October figures against SRBMiner 3.7.4.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RTX 2080 Ti | Thailand (170 W) | 95392 | 63.4 | 64.1 | 64.2 | 99% |
| RTX 2080 Ti | Pennsylvania (260 W) | 150735 | 89.0 | 93.2 | 92.0 | 96% |
| RTX 2080 | Colorado (275 W) | 149439 | 77.6 | 81.7 | 82.8 | 94% |
| RTX 2070 Super | Alberta (215 W) | 31798 | 60.3 | 62.4 | 64.0 | 94% |
| RTX 2070 | South Korea (150 W) | 139007 | 47.8 | 50.4 | 51.7 | 93% |
| RTX 2060 Super | Germany (175 W) | 149900 | 42.7 | 45.3 | 46.4 | 92% |
| RTX 2060 | Australia (190 W, 6 GB) | 152547 | 47.8 | 49.9 | 48.4 | 96% |
| RTX 2060 | South Korea (184 W, 12 GB) | 27568 | 49.8 | 52.9 | 52.4 | 94% |

- **The gap is work per clock.** Our clock is never more than 30 MHz below the faster competitor's; we do 5–7.5% less
  per clock.
- **Thailand 2080 Ti:** the host held the card at 975–1020 MHz on 11 October, so all three miners read 6–11% lower
  than on 10 October (ours 67.8, 94%). The competitors lost more, which is why the row moved to 99%.
- **RTX 2060 Super:** held at 76–78 °C and 118–136 W of 175 W, below the card's base clock, for every miner.
- **Since v0.5.14:** a band of 5 column groups on 4 MB cards (+0.30% on a 2070, +0.21% on a 2080), three fold changes
  (about +4.6% to +6.6%) and a second fold kernel on 4 MB cards (+0.17 to +0.19%) are in the core after v0.6.6.
  Pool runs of that core on 21 October read 94.0 TH/s on the Pennsylvania 2080 Ti (100.9% of the row's best), 82.1 on
  the Colorado 2080 (99.2%) and 50.8 on the South Korea 2070 (98.3%). Details are in earn/native/probes/README.md.

### On Salad, 28 October, against PeakMiner 2.20.3 and SRBMiner 3.8.0

v0.6.9 by the Salad method in "How a host is tested". Salad's 2070 and 2060 classes also handed out Super cards,
which got rows of their own.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RTX 2080 Ti | Salad (280 W) | ec9e6c87 | 96.5 | 89.0 | 94.4 | 102% |
| RTX 2080 Ti | Salad (260 W) | ef08a260 | 94.9 | 94.7 | 86.4 | 100% |
| RTX 2080 | Salad (245 W) | 83f7bc7e | 65.6 | 67.6 | 65.5 | 97% |
| RTX 2080 | Salad (129 W) | f765012b | 54.8 | 54.5 | 53.8 | 100% |
| RTX 2070 Super | Salad (215 W) | 2d10f218 | 61.6 | 62.7 | 62.4 | 98% |
| RTX 2070 Super | Salad (215 W) | ffd709c4 | 62.6 | 63.9 | 63.1 | 98% |
| RTX 2070 | Salad (175 W) | 2aaa0e3e | 48.0 | 49.3 | 47.4 | 97% |
| RTX 2070 | Salad (184 W) | 0416482b | 51.1 | 49.5 | 49.3 | 103% |
| RTX 2060 Super | Salad (200 W) | 413b2e3d | 48.1 | 49.5 | 48.2 | 97% |
| RTX 2060 Super | Salad (184 W) | ab1e6b0e | 46.9 | 47.5 | 47.3 | 99% |
| RTX 2060 | Salad (168 W, 6 GB) | 81031a48 | 46.1 | 47.3 | 45.4 | 98% |
| RTX 2060 | Salad (160 W, 6 GB) | a1fd5f37 | 47.4 | 48.9 | 46.7 | 97% |

- **Median 98% (97–103%), up from 94% on Vast,** from the core after v0.6.6. PeakMiner 2.20.3 gained about 3% and is
  the faster competitor on 11 of 12 PCs.
- **The rows above 100% are odd PCs:** ec9e6c87 swung between 207 and 266 W for every miner; f765012b is capped at
  129 W; 0416482b ran ours at a lower clock and still hashed 3% more, for no visible reason.
- **Our 2080 Ti readout is coarse:** a single 15-second reading can be about ±8% off. The 5-minute mean is sound.

### On Salad, 30 October, against SRBMiner 3.8.2

The same method, after SRBMiner 3.8.1 and 3.8.2 listed small pearlhash gains on RTX 20 to 50.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RTX 2080 Ti | Salad (260 W) | ef08a260 | 94.8 | 95.0 | 93.6 | 99.7% |
| RTX 2080 Ti | Salad (260 W) | 671bb26b | 78.3 | 79.1 | 75.5 | 99.1% |
| RTX 2080 | Salad (245 W) | 83f7bc7e | 66.9 | 68.8 | 68.0 | 97.4% |
| RTX 2080 | Salad (260 W) | b2de4b63 | 78.2 | 79.0 | 78.0 | 99.0% |
| RTX 2070 Super | Salad (255 W) | 32c563bc | 62.1 | 63.2 | 62.4 | 98.2% |
| RTX 2070 | Salad (185 W) | 46bfd62d | 55.9 | 57.6 | 56.4 | 97.0% |
| RTX 2070 | Salad (225 W) | 1e19bb79 | 61.3 | 63.8 | 62.5 | 96.1% |
| RTX 2060 | Salad (6 GB) | 4af24fcc | 35.7 | 37.3 | 35.1 | 95.7% |
| RTX 2060 | Salad (167 W, 6 GB) | 0338617b | 46.9 | 48.3 | 45.8 | 97.3% |

- **Median 97.4%.** PeakMiner 2.20.3 was fastest on every PC; SRBMiner 3.8.2 led us on four of nine, by 2% at most.
- On ef08a260, the 28 October row's own PC, SRBMiner rose from 86.4 to 93.6 (+8.3%) and ours was flat.
- Our figure carries about ±1.5–2.4% of noise on fast cards, so 1–2 points between PCs are within noise.

### Our own builds, 1 November: RTX 20 experiments

- **Transcript hash and the L0 instruction cache (research item 11).** A stub hash a fifth the size bought at most
  +0.10% on a 2080 Ti, and a rolled hash lost about 0.4%. Instruction fetch doesn't limit the RTX 20 fold. Dropped.
- **k 4096 on RTX 20 (research item 9).** The band k 4096 would need costs 1.24% on a 2080 Ti and 2.76% on a 2070
  Super in DRAM traffic alone, more than k 4096 could win back (about 0.9%). Dropped.

### Monero on the CPU

SRBMiner mines Monero on the CPU alongside Pearl in one process; PeakMiner doesn't mine Monero. On 11 October each RTX
20 rental also ran SRBMiner 3.7.5 on Monero alone and on both at once.

| Card | Host | CPU (CPUs the container got) | Monero alone | Monero with PRL | SRBMiner PRL alone → with Monero |
|---|---|---|---|---|---|
| RTX 2080 Ti | Thailand | EPYC 7B13 (17.6 of 256) | 504 H/s | 556 H/s | 64.2 → 44.0 |
| RTX 2080 Ti | Florida (stand-in) | Ryzen 9 5900X (11.5 of 24) | 4,133 H/s | 3,869 H/s | 32.8 → 40.1 |
| RTX 2070 Super | Alberta | Ryzen 5 5500 (11.5 of 12) | 2,677 H/s | 2,509 H/s | 64.0 → 64.2 |
| RTX 2070 | South Korea | Xeon E5-2680 v4 (13.4 of 56) | 3,672 H/s | 3,760 H/s | 51.7 → 50.5 |
| RTX 2060 Super | Germany | Core i5-10400 (7.7 of 8) | 1,845 H/s | 1,810 H/s | 46.4 → 45.3 |
| RTX 2060 | Australia | Xeon E5-2699 v3 (34.6 of 72) | 4,128 H/s | 4,209 H/s | 48.4 → 49.4 |
| RTX 2060 | South Korea (12 GB) | Xeon E5-2680 v4 (9.0 of 56) | 2,399 H/s | 2,575 H/s | 52.4 → 52.6 |

- No container allowed huge pages or MSR tuning, so these Monero rates are low for the CPUs.
- SRBMiner ignores the container's CPU quota and ran 2–15 times more threads than CPUs on some hosts. On Thailand that
  cut its Pearl rate by 31%.

Our CPU miner (new in v0.6.0) against SRBMiner 3.7.5 on Monero alone, same threads and pool, 11 October:

| CPU (CPUs the container got) | GPU | Threads | Ours | SRBMiner 3.7.5 | % of best |
|---|---|---|---|---|---|
| Core i5-11400 (11.5 of 12) | RTX 3060 | 6 | 2,457 H/s | 1,880 H/s | 131% |
| Ryzen 9 5950X (15.4 of 32) | RTX 2060 | 14 | 6,439 H/s | 4,527 H/s | 142% |

- Ours used transparent huge pages; SRBMiner couldn't get the reserved huge pages it asks for. That likely explains
  much of the gap.
- With both mining, our Pearl rate moved by under 1%.

## RTX 30-series

Run on 10 October. Three rows are on their own machine; twelve moved to new hosts.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RTX 3090 Ti | Vietnam (450 W) | 21444 | 153.1 | 149.3 | 153.9 | 99% |
| RTX 3090 Ti | Oman (450 W) | 12840 | 150.4 | 150.7 | didn't start | 99.8% |
| RTX 3090 | Ukraine (290 W) | 141130 | 115.8 | 113.3 | 118.4 | 98% |
| RTX 3090 | Quebec (325 W) | 67901 | 125.1 | 123.9 | 123.0 | 101% |
| RTX 3080 Ti | Japan (330 W) | 140510 | 125.9 | 124.9 | 128.1 | 98% |
| RTX 3080 Ti | Portugal (350 W) | 56596 | 128.1 | 127.9 | 127.1 | 100% |
| RTX 3080 | Taiwan (320 W) | 146647 | 110.9 | 110.8 | 109.0 | 100% |
| RTX 3080 | Austria (290 W) | 143869 | 111.1 | 110.4 | 109.9 | 101% |
| RTX 3070 Ti | Quebec (290 W) | 137703 | 85.3 | 85.8 | 86.0 | 99% |
| RTX 3070 | Quebec (220 W) | 56506 | 72.7 | 73.3 | 75.1 | 97% |
| RTX 3070 | Quebec (180 W) | 152549 | 72.1 | 72.9 | 74.6 | 97% |
| RTX 3060 Ti | Colorado (200 W) | 151268 | 59.7 | 60.1 | 60.4 | 99% |
| RTX 3060 Ti | Argentina (180 W) | 150126 | 58.1 | 58.9 | 57.5 | 99% |
| RTX 3060 | New Brunswick (170 W) | 15303 | 50.8 | 51.4 | 51.9 | 98% |
| RTX 3060 | South Korea (170 W) | 137223 | 48.2 | 48.8 | 49.2 | 98% |

- **Oman 3090 Ti:** SRBMiner exited at once on this host, so the row compares with PeakMiner.
- **Dropped runs:** a Spain 3090 Ti had another workload on the GPU, and a Utah 3080 and a US 3060 were held by heat.
  Those rows were rerun on other hosts.
- **Quebec 3090 and Colorado 3060 Ti:** SRBMiner's end figure was a dip. Against its mean the rows read 99.1% and
  98.1%.
- **Argentina 3060 Ti:** the host locks the clock at 1560 MHz for every miner.

### On Salad, 28 October, against PeakMiner 2.20.3 and SRBMiner 3.8.0

v0.6.9 by the Salad method.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RTX 3090 Ti | Salad (336 W) | b687d663 | 126.0 | 149.7 | 152.7 | 83% |
| RTX 3090 Ti | Salad (450 W) | ef44d3cd | 148.3 | 160.5 | 161.3 | 92% |
| RTX 3090 | Salad (350 W) | 66666c27 | 113.7 | 121.3 | 120.3 | 94% |
| RTX 3080 Ti | Salad (298 W) | 344adcb0 | 121.5 | 133.2 | 132.8 | 91% |
| RTX 3080 Ti | Salad (350 W) | 4ec4131f | 126.9 | 137.4 | 136.4 | 92% |
| RTX 3080 | Salad (285 W) | 93c805c0 | 105.8 | 114.3 | 113.4 | 93% |
| RTX 3080 | Salad (340 W) | 9fbd405b | 111.8 | 115.0 | 115.0 | 97% |
| RTX 3070 Ti | Salad (310 W) | b7d4d9a2 | 88.3 | 93.6 | 93.1 | 94% |
| RTX 3070 Ti | Salad (310 W) | cdc3c2a9 | 89.9 | 94.8 | 94.6 | 95% |
| RTX 3070 | Salad (270 W) | 46aae7db | 83.4 | 89.6 | 89.4 | 93% |
| RTX 3070 | Salad (216 W) | 22bd1d8a | 77.4 | 83.6 | 83.8 | 92% |
| RTX 3060 Ti | Salad (225 W) | abef70e6 | 65.9 | 71.2 | 70.6 | 93% |
| RTX 3060 Ti | Salad (210 W) | 3a481d90 | 66.4 | 72.5 | 71.8 | 92% |
| RTX 3060 | Salad (170 W, 12 GB) | 0771915f | 48.4 | 50.5 | 50.3 | 96% |
| RTX 3060 | Salad (170 W, 12 GB) | 85c1c681 | 50.1 | 54.3 | 54.3 | 92% |

- **Median 92.5%, down from 97–101% on Vast,** mostly because the competitors got 5–7% faster where power limits match.
- **The gap has two parts:** per clock, ours does 94.9–96.7% of the best competitor's work (95.8% on average); and at
  the power limit ours runs a 1.6–4.6% lower clock. Where something other than power set the clock, ours reached
  96–97%.
- **b687d663 (83%):** the host caps the card at 336 W, and ours ran at 1560 MHz against their 1800–1815 MHz. Per clock
  it matches the 450 W 3090 Ti.

### Our own builds, 31 October: holding B' in the L2 (the Ampere flip)

Research item 1, raced in race mode. The flip holds the B' of a narrow batch in the L2's persisting slice and streams
A', instead of reading B' again every band. It ships in `f421841` (`PEARL_AMPERE_FLIP`) on the 3060 Ti and the 3080
12 GB, 3080 Ti, 3090 and 3090 Ti, the cards where its modelled DRAM bytes are at most 80% of the band walk's.

| Card | Batch | vs release | Rounds ahead |
|---|---|---|---|
| RTX 3060 Ti (200 W) | 64 | +1.92% | 35/35 |
| RTX 3090 (350 W) | 128 | +0.66% | 35/35 |
| RTX 3090 (297.5 W), shipped build | 128 | +1.52% | 49/49 |
| RTX 3080 10 GB (320 W), flip forced on | 64 | −0.71% | 0/44 |

- A tighter power limit gains more: watts that DRAM no longer takes go to the SM clock.
- `verify-hits` passed 400 of 400 on every build. Tiles from a counter, 64 offsets on the 3090, half-held B' and wider
  strip walks all did worse and were dropped.

### Our own builds, 1–2 November: the tile hash on RTX 30 and 40

- **Copy groups and a readout fence (research item 10).** The copy change bought at most +0.10%; the fences cost
  2.4–2.7%. Dropped.
- **What the tile hash costs (research item 12).** Removing it gains about 1.4–1.7% on a 3060 Ti and a 4070, far more
  than the earlier +0.15–0.2% estimate.
- **Where the cost comes from (step 0, 2 November).** It's the hash running past the partner warp's window, not issue
  contention (0.03% on a 3060 Ti, 0.06% on a 4070 Super). Past about 2,300 cycles after the hand-off, each cycle the
  hasher runs costs a tile 0.9–1.0 cycle; the hash releases at about 3,150. That overturns item 12's reading. Dropped:
  hash control bits, warp turn-taking (#18) and IMMAs during the hash. Parked: readout pacing (#17).
- **A wider window (g6, 2 November).** `PEARL_TALL_APT=6u` moves the partner's wait later, but moving the copy points
  costs 1,380 cycles a tile on the 3060 Ti, nearly four times what an infinite window could free (+355). Dropped.
- **A shorter hash (Z6 and f39, 2 November, 3060 Ti).** No transcript clearing at all saves only 20 cycles a tile, so
  Z6 was dropped. Any edit to the hash's code makes ptxas reallocate the chunk loop, which then costs 3,383 cycles a
  tile; f39 (`PEARL_TALL_HASH_FMA`) lost 3,517 (−3.4%). All store-side work is dropped too.
- **That closes the RTX 30/40 hash line.** On the 3060 Ti the hash costs 1.3% of a tile, and none of the ways to reach
  it pays.

## RTX 40-series

Run on 10 October. Ten rows are on their own machine; eight moved to new hosts.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RTX 4090 | Australia (450 W) | 17334 | 318.7 | 316.2 | 317.7 | 100% |
| RTX 4090 | Texas (450 W) | 153411 | 314.7 | 310.8 | 314.1 | 100% |
| RTX 4090D | South Carolina (425 W) | 142279 | 276.7 | 268.0 | 277.3 | 99.8% |
| RTX 4080 Super | Florida (320 W) | 142085 | 198.0 | 196.5 | 198.9 | 99.5% |
| RTX 4080 Super | California (275 W) | 138449 | 198.1 | 196.7 | 198.6 | 99.8% |
| RTX 4080 | Japan (320 W) | 6706 | 199.8 | 198.6 | 199.0 | 100% |
| RTX 4080 | British Columbia (225 W) | 43318 | 169.7 | 158.3 | 172.9 | 98% |
| RTX 4070 Ti Super | Texas (285 W) | 142663 | 172.2 | 171.5 | 171.6 | 100% |
| RTX 4070 Ti Super | Quebec (285 W) | 36427 | 167.8 | 166.9 | 167.6 | 100% |
| RTX 4070 Ti | North Carolina (285 W) | 137392 | 155.6 | 156.2 | 155.3 | 99.6% |
| RTX 4070 Ti | France (285 W) | 138259 | 160.4 | 160.1 | 159.2 | 100% |
| RTX 4070 Super | California (209 W) | 145255 | 134.8 | 132.2 | 131.3 | 102% |
| RTX 4070 Super | California (220 W) | 153237 | 138.6 | 137.7 | 138.9 | 99.8% |
| RTX 4070 | Colorado (200 W) | 144340 | 119.2 | 118.5 | 119.7 | 99.6% |
| RTX 4060 Ti | Ohio (150 W) | 27068 | 86.0 | 84.1 | 86.6 | 99% |
| RTX 4060 Ti | Brazil (160 W) | 152073 | 88.7 | 87.7 | 88.4 | 100% |
| RTX 4060 | New Zealand (115 W) | 148383 | 60.4 | 60.1 | 58.5 | 100% |
| RTX 4060 | Australia (115 W) | 143986 | 47.0 | 47.8 | 45.7 | 98% |

- **RTX 4090D:** its raw files were lost and SRBMiner's figure is only known to be 277.30–277.39, so the row is
  99.7–99.8% either way.
- **Australia 4060:** the host locks the card at 1995 MHz.

### On Salad, 29–30 October, against PeakMiner 2.20.3 and SRBMiner 3.8.0

v0.6.9 by the Salad method. SRBMiner 3.8.1 and 3.8.2 were already out, so its figures here may be a little below its
newest.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RTX 4090 | Salad (450 W) | adbabd35 | 321.8 | 342.6 | 344.0 | 94% |
| RTX 4090 | Salad (450 W) | 2982dea7 | 313.3 | 331.7 | 329.3 | 94% |
| RTX 4080 | Salad (320 W) | 2ffd8ac3 | 197.9 | 209.3 | 209.3 | 95% |
| RTX 4080 | Salad (320 W) | 9850190a | 190.9 | 202.4 | 203.0 | 94% |
| RTX 4070 Ti Super | Salad (254 W) | 8445b495 | 165.0 | 175.8 | 175.4 | 94% |
| RTX 4070 Ti Super | Salad (228 W) | 42d4fae9 | 169.7 | 181.2 | 179.8 | 94% |
| RTX 4070 Ti | Salad (242 W) | cf64ae47 | 153.6 | 163.7 | 163.4 | 94% |
| RTX 4070 Ti | Salad (285 W) | 2a9e8d3d | 160.2 | 168.7 | 168.1 | 95% |
| RTX 4070 | Salad (200 W) | 24656adc | 114.2 | 121.2 | 121.1 | 94% |
| RTX 4070 | Salad (200 W) | 8e6114f1 | 122.5 | 128.6 | 128.5 | 95% |
| RTX 4060 Ti | Salad (165 W, 16 GB) | 4138e3e4 | 91.1 | 96.0 | 95.8 | 95% |
| RTX 4060 Ti | Salad (165 W, 16 GB) | 7e70e38b | 88.3 | 93.4 | 93.1 | 95% |
| RTX 4060 | Salad (115 W) | 4e31b49a | 59.4 | 62.0 | 61.8 | 96% |
| RTX 4060 | Salad (115 W) | 59d34bc2 | 59.7 | 62.8 | 62.6 | 95% |

- **Median 94.5%, down from 98–102% on Vast.** Where power limits match, the competitors are 4–7% faster than on Vast
  and ours is flat to 3% slower. On a 4090 at 450 W: ours +0.3%, the best competitor +6.9%.
- These boxes don't record clocks, so the gap can't be split into per-clock work and clock.

## RTX 50-series

Run on 10 October on the CUDA 13 core, except the India 5090, whose driver 570 gets the CUDA 12.8 core. Seven rows
are on their own machine; five moved to new hosts.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RTX 5090 | South Korea (500 W) | 148959 | 380.6 | 374.0 | 384.8 | 99% |
| RTX 5090 | India (500 W) | 151861 | 362.8 | driver too old | 378.2 | 96% |
| RTX 5080 | United States (330 W) | 137656 | 198.7 | 201.2 | 200.2 | 99% |
| RTX 5080 | Poland (360 W) | 153231 | 229.8 | 228.5 | 230.4 | 99.7% |
| RTX 5070 Ti | South Korea (250 W) | 28072 | 92.0 | 93.5 | 91.0 | 98% |
| RTX 5070 Ti | South Korea (250 W) | 28852 | 92.3 | 93.5 | 92.7 | 99% |
| RTX 5070 | Colombia (250 W) | 145974 | 130.9 | 131.2 | 132.1 | 99% |
| RTX 5070 | Texas (250 W) | 147729 | 131.8 | 131.9 | 133.0 | 99% |
| RTX 5060 Ti | Finland (165 W) | 140025 | 93.3 | 92.5 | 94.3 | 99% |
| RTX 5060 Ti | Ontario (180 W) | 153080 | 95.2 | 95.4 | 88.4 | 99.8% |
| RTX 5060 | Virginia (125 W) | 151478 | 76.2 | didn't start | 76.8 | 99% |
| RTX 5060 | India (145 W) | 119163 | 77.4 | 76.4 | didn't start | 101% |

- **India 5090 (96%):** per clock it does 3.4% less than SRBMiner, against 0.6% less on the South Korea 5090 on the
  CUDA 13 core. They're different hosts, so this doesn't prove the core is the cause. PeakMiner needs driver 580 or
  newer on these cards.
- **RTX 5070 Ti:** both hosts lock the SM clock at 1342 MHz, so the cards run far below a stock 5070 Ti.
- **Virginia 5060 and United States 5080:** competitors' end figures were dips; against their means the rows read
  98.4% and about 98%.

### On Salad, 29–30 October, against PeakMiner 2.20.3 and SRBMiner 3.8.0

v0.6.9 by the Salad method. Every PC had driver 591 or newer, so all ran the CUDA 13 core.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RTX 5090 | Salad (575 W) | 3fda699b | 404.5 | 438.4 | 440.1 | 92% |
| RTX 5090 | Salad (450 W) | 01bdc2fc | 354.8 | 394.2 | 398.9 | 89% |
| RTX 5080 | Salad (360 W) | cc442fff | 225.8 | 232.9 | 234.3 | 96% |
| RTX 5080 | Salad (360 W) | f6e3944e | 228.5 | 237.0 | 237.4 | 96% |
| RTX 5070 Ti | Salad (300 W) | ebdd97df | 186.5 | 194.7 | 195.2 | 96% |
| RTX 5070 Ti | Salad (300 W) | 770088cd | 185.8 | 195.1 | 195.5 | 95% |
| RTX 5070 | Salad (250 W) | 71aa43da | 134.2 | 141.3 | 141.3 | 95% |
| RTX 5070 | Salad (225 W) | a163e4f3 | 130.0 | 137.0 | 137.3 | 95% |
| RTX 5060 Ti | Salad (180 W, 16 GB) | 6c52b860 | 94.1 | 98.6 | 97.1 | 95% |
| RTX 5060 Ti | Salad (153 W, 16 GB) | 78d368bf | 92.7 | 97.7 | 97.5 | 95% |
| RTX 5060 | Salad (145 W) | 616220a5 | 74.0 | 77.7 | 78.1 | 95% |
| RTX 5060 | Salad (145 W) | 0db26521 | 78.5 | 77.4 | 77.7 | 101% |

- **94.6–96.4% on 10 of 12 PCs, down from 96–101% on Vast.** The competitors are 2–7% faster than on Vast where power
  limits match; ours is −1% to +2%.
- **The RTX 5090 trails most:** 92% at 575 W and 89% at 450 W.

### On Salad, 30 October, against SRBMiner 3.8.2

The same method. HeroMiners was retired during the run, so later boxes mined on Kryptex; all three miners of a row
share its pool.

| Card | Host | Machine ID | Pool | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|---|
| RTX 5090 | Salad (575 W) | 541c8dda | HeroMiners | 411.6 | 418.2 | 430.4 | 95.6% |
| RTX 5090 | Salad (437 W) | 83be133f | HeroMiners | 330.9 | 366.4 | 377.8 | 87.6% |
| RTX 5080 | Salad (360 W) | b57ed5de | HeroMiners | 213.2 | 223.7 | 229.0 | 93.1% |
| RTX 5080 | Salad (360 W) | f6e3944e | HeroMiners | 228.6 | 237.0 | 239.3 | 95.5% |
| RTX 5070 Ti | Salad (300 W) | afe54169 | HeroMiners | 186.1 | 195.9 | 196.3 | 94.8% |
| RTX 5070 Ti | Salad (300 W) | ca7ddba0 | HeroMiners | 190.5 | 200.6 | 201.3 | 94.7% |
| RTX 5070 | Salad (250 W) | cbdb57ce | HeroMiners | 127.9 | 132.8 | 133.8 | 95.6% |
| RTX 5070 | Salad (250 W) | 5599f5d3 | Kryptex | 128.3 | 133.4 | 134.0 | 95.8% |
| RTX 5060 Ti | Salad (150 W, 16 GB) | f1e711c2 | Kryptex | 92.1 | 97.9 | 86.9 | 94.1% |
| RTX 5060 Ti | Salad (180 W, 16 GB) | 3926805c | Kryptex | 96.7 | 101.0 | 101.3 | 95.5% |
| RTX 5060 | Salad (145 W) | 960ed7a7 | Kryptex | 78.6 | 76.1 | 71.8 | 103.2% |
| RTX 5060 | Salad (123 W) | dd173265 | Kryptex | 74.9 | 79.7 | 80.7 | 92.8% |

- **Median 95.2%.** SRBMiner 3.8.2 was the faster competitor on 10 of 12 PCs.
- **The 437 W 5090 read 87.6%,** our lowest NVIDIA row. Two 145 W 5060 PCs (960ed7a7 here, 0db26521 on 29–30
  October) beat both competitors, while another at 145 W read 95%. What separates them isn't known.
- The two 5070 rows, one on each pool, agree (95.6% and 95.8%).

### Our own builds, 1–2 November: RTX 50 and RTX 40 experiments

- **k 4096 on RTX 50 and 40 (research item 8).** It does more work a clock (about +1.0% on a 5090, +0.4% on a 4090),
  but its doubled DRAM traffic takes watts from the SM clock at the power cap: −0.78% on a 5090, −0.29% on a 4090.
  Dropped. A card held at a fixed clock below its limit might still gain.
- **What moving the operands costs (research item 16).** Skipping the operand loads (timing only) let a stock 5090 at
  600 W clock 8.16% higher (+8.33% hashrate) with the same work per clock. The mid cards gain less (+3.29% on a 5080,
  +1.95% on a 5070 Ti, +2.84% on a 5060), because they reach their top clock first.
- **Neither way to move fewer operands works.** Drawing B' on the SM gained only +1.81% against a +2.5% bar. Sharing B'
  between two CTAs by TMA multicast hangs: the multicast's bytes or its transaction count never arrive on sm_120, and
  CUTLASS keeps RTX 50 to one-CTA clusters. Item 16 stops here.

## A100-class (sm_80)

Run on 10 October. Four rows are on their own machine; one moved.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| A100 PCIe 40 GB | Japan (250 W) | 146405 | 211.1 | 213.7 | 215.3 | 98% |
| A100 SXM4 40 GB | New York (400 W) | 152837 | 245.5 | 251.8 | 253.4 | 97% |
| A100 SXM4 40 GB | Mississippi (400 W) | 117301 | 246.6 | 252.8 | 253.8 | 97% |
| A30 24 GB | Australia (144 W) | 145742 | 100.5 | 90.1 | 90.8 | 111% |
| CMP 170HX | Georgia (250 W) | 147823 | 163.4 | 165.5 | 165.3 | 99% |

- **A100s:** all miners ran at the power limit and we held 15–30 MHz more than SRBMiner, so its 2–3% lead is work per
  clock.
- **A30:** the host lowered the power limit during the competitors' runs. Against PeakMiner at 144 W, ours is about
  107%, not 111%.
- **CMP 170HX:** reports 64 GB, so it's an unlocked card.

## Hopper (sm_90a)

Run on 10 October. Two rows are on their own machine; the H100 SXM moved.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| H100 SXM 80 GB | Czechia (575 W) | 148671 | 682.8 | 697.7 | 688.1 | 98% |
| H100 NVL 94 GB | Japan (400 W) | 29785 | 536.5 | 548.6 | 539.1 | 98% |
| H100 PCIe 80 GB | Czechia (350 W) | 147981 | 484.9 | 448.3 | 488.0 | 99% |
| H200 141 GB | — | — | — | — | — | not run: over budget |
| H200 NVL 141 GB | — | — | — | — | — | not run: over budget |

- On the H100 SXM and NVL all miners ran at about the same clock, so PeakMiner's 2% lead is work per clock.
- **H200s, last figures (8–9 October, v0.5.14's Hopper fold):** H200 141 GB 719.9 against SRBMiner 3.7.1's 720.4
  (99.9%); H200 NVL 682.9 against SRBMiner's 675.3 (101%).

## Ada workstation and data-center cards (sm_89)

Run on 10 October. Eight rows are on their own machine and four moved; the Vietnam L40 wasn't run.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| L40S | Texas (350 W) | 96563 | 290.7 | 253.7 | 291.9 | 99.6% |
| L40S | Romania (350 W) | 150627 | 288.1 | 252.8 | 290.4 | 99% |
| RTX 6000 Ada | New York (300 W) | 142572 | 210.0 | 180.4 | 219.0 | 96% |
| RTX 6000 Ada | Germany (300 W) | 152478 | 216.6 | 186.6 | 225.8 | 96% |
| RTX 5000 Ada | Missouri (250 W) | 107776 | 208.4 | 185.9 | 210.7 | 99% |
| L40 | — | — | — | — | — | not run: none for rent |
| L40 | New York (250 W) | 56199 | 153.2 | 132.8 | 156.6 | 98% |
| RTX 4500 Ada | France (210 W) | 148971 | 139.7 | 134.2 | 139.6 | 100% |
| RTX 4000 Ada | Norway (130 W) | 149238 | 98.0 | 94.5 | 98.5 | 99.5% |
| RTX 4000 Ada | France (130 W) | 152701 | 96.1 | 92.4 | 97.4 | 99% |
| L4 | Czechia (72 W) | 109442 | 79.3 | 73.8 | 80.9 | 98% |
| L4 | Washington (72 W) | 152541 | 80.5 | 75.0 | 81.6 | 99% |
| RTX 2000 Ada | Romania (70 W) | 68269 | 48.3 | 46.9 | 48.7 | 99% |

- **Vietnam L40, last figures (7–8 October):** ours 172.9 against SRBMiner 3.7.1's 170.4, 101%.
- **RTX 4500 Ada:** the card ran at 88 °C for every miner, so heat likely held this run.

## RTX PRO Blackwell (sm_120)

Run on 10 October on the CUDA 13 core. Four rows are on their own machine; three moved.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RTX PRO 6000 Workstation | Mississippi (600 W) | 148552 | 398.7 | 389.1 | 419.2 | 95% |
| RTX PRO 6000 Server | Australia (600 W) | 139338 | 435.1 | 432.4 | 435.9 | 99.8% |
| RTX PRO 6000 Max-Q | Australia (300 W) | 142021 | 308.0 | 302.9 | 306.7 | 100% |
| RTX PRO 5000 48 GB | California (250 W) | 150611 | 211.5 | 208.3 | 214.8 | 98% |
| RTX PRO 4500 | Arizona (200 W) | 148915 | 175.7 | 171.2 | 176.0 | 99.8% |
| RTX PRO 4000 | France (145 W) | 147490 | 127.9 | 126.5 | 128.8 | 99% |
| RTX 6000D | Czechia (550 W) | 149788 | 144.6 | 144.6 | 144.6 | 100% |

- **RTX PRO 6000 Max-Q:** SRBMiner's end figure is a dip; against its mean the row reads 98.4%.
- **RTX 6000D:** a cut-down card. All three miners sat just under its top clock at 234–254 W, so they tie.

## Ampere workstation and data-center cards (sm_86)

Run on 10 October. Six rows are on their own machine; four moved.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RTX A6000 | Vietnam (300 W) | 140998 | 121.6 | 122.3 | 124.8 | 97% |
| RTX A6000 | Romania (300 W) | 37274 | 126.1 | 125.7 | 124.9 | 100% |
| A40 | United Kingdom (300 W) | 152158 | 125.8 | 124.6 | 128.2 | 98% |
| A40 | Belgium (300 W) | 151322 | 123.2 | 122.5 | 127.3 | 97% |
| RTX A5000 | North Carolina (230 W) | 147999 | 93.7 | 92.8 | 94.0 | 99.7% |
| RTX A5000 | Italy (200 W) | 151139 | 84.6 | 83.4 | 87.7 | 97% |
| RTX A4000 | Estonia (140 W) | 145734 | 57.9 | 56.0 | 61.6 | 94% |
| RTX A4000 | Kentucky (140 W) | 29102 | 58.2 | 56.8 | 62.3 | 93% |
| RTX A2000 12 GB | Poland (70 W) | 28227 | 28.7 | 29.0 | 32.3 | 89% |
| RTX A2000 6 GB | Japan (70 W) | 152199 | 27.6 | 28.5 | 30.4 | 91% |

- **RTX A2000s and A4000s trail by clock:** SRBMiner ran the A2000s at 1282 and 1200 MHz against our 1147 and 1102,
  and the A4000s 90 MHz higher. Both A4000s ran at 89–93 °C.

### CMP 90HX on Clore, 4 November (our miner only)

A five-minute check that the mining-only CMP 90HX (GA102, 10 GB) runs our miner. Release v0.6.11 on three cards of a
Clore server in Sweden, on HeroMiners. Not a row: only our miner ran, and not on Vast.

| Card | Power limit | TH/s after 5 min | SM clock |
|---|---|---|---|
| CMP 90HX (GPU 0) | 200 W of 250 W | 72.7 | 1485 MHz |
| CMP 90HX (GPU 1) | 200 W of 250 W | 72.7 | 1500 MHz |
| CMP 90HX (GPU 2) | 200 W of 250 W | 71.0 | 1380 MHz |

- Every card sat at its 200 W cap. A 250 W limit would give more, by an amount we didn't measure.
- At PRL $1.45 and difficulty 52.2M, 72 TH/s earns about $0.056 an hour, below Clore's cheapest CMP 90HX ($0.073
  on-demand, $0.063 spot). It doesn't pay to rent there.

## B200 (sm_100)

Run on 10 October on the row's own machine, on the CUDA 12.8 core.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| B200 180 GB | Oregon (1000 W) | 96942 | 457.9 | 1380.6 | 1394.7 | 33% |

- **Why it's slow:** the B200 runs the RTX 50 fold built for sm_100, on `mma.sync`. A probe on this host topped out at
  about 568 TH/s with it, about 41% of SRBMiner. The competitors do three times our rate at a lower clock, so they use
  the B200's own tensor instructions (tcgen05). Matching them needs a tcgen05 fold.

## Tesla T4 (sm_75)

Run on 10 October on the row's own machine, on the CUDA 12.8 core.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| Tesla T4 16 GB | Czechia (70 W) | 28909 | 30.1 | 32.5 | 33.1 | 91% |

- All miners ran at the card's 70 W limit, ours at 825 MHz and the competitors at 840, so we do about 7% less work
  per clock.
- The core after v0.6.6 mines 4 MB cards like the T4 at band 5 and with the second RTX 20 fold kernel. No T4 has run
  it.

## GTX 10-series (Pascal, sm_61, on Salad)

PeakMiner and SRBMiner can't mine Pearl on these cards: PeakMiner's README says Pearl needs RTX 20xx or newer, and
both got nothing on a GTX 1080 on Salad on 17 October. v0.6.8 adds a Pascal core (a fold on `dp4a`), but the release
hasn't been run on these cards by "How a host is tested", so "Ours" is "not run".

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| GTX 1080 Ti | Salad | not run | not run | 0 | 0 | — |
| GTX 1080 | Salad (180 W) | c868132d | not run | 0 | 0 | — |
| GTX 1070 | Salad | not run | not run | 0 | 0 | — |
| GTX 1060 | Salad (6 GB) | not run | not run | 0 | 0 | — |
| GTX 1050 Ti | Salad (4 GB) | not run | not run | 0 | 0 | — |

- **Before its release (17–18 October),** the Pascal core mined at 17.8 TH/s on a 250 W GTX 1080 Ti on Salad, with
  every checked hit correct and the pool accepting its share.

## GTX 16-series (Turing without tensor cores, sm_75, on Salad)

Run on 17 October on Salad, much as in "Salad runs", with our release v0.6.0 against PeakMiner 2.17.6 and SRBMiner 3.7.1.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| GTX 1660 Super | Salad (106 W, 6 GB) | 1535c250 | 1.1 | 1.13 | 1.13 | 97% |
| GTX 1660 Super | Salad (119 W, 6 GB) | 535c608c | 1.0 | 1.01 | 0.96 | 96% |
| GTX 1660 Ti | Salad (6 GB) | 741f9c56 | 1.2 | 1.14 | 1.16 | 104% |
| GTX 1660 Ti | Salad (88 W, 6 GB) | 30b3c19c | 1.3 | 1.28 | 1.29 | 101% |
| GTX 1660 | Salad (120 W, 6 GB) | 095beb7b | 1.1 | 1.05 | 1.04 | 104% |
| GTX 1660 | Salad (6 GB) | not run | not run | not run | not run | — |
| GTX 1650 | Salad (75 W, 4 GB) | 522e3507 | 0.7 | 0.68 | 0.69 | 99% |
| GTX 1650 | Salad (4 GB) | not run | not run | not run | not run | — |

- **All three miners are equally slow.** These cards have no tensor cores, so every miner's tensor instructions run on
  a slow shared unit. A share takes about 2.5 hours at 1 TH/s.
- A dp4a path could reach 5–9 TH/s, but that still earns less than Salad charges for these cards, so it isn't being
  built.
- Our CLI prints one decimal, so "% of best" is only good to about ±5%.

## AMD Radeon (RDNA 3 and 4, on Salad)

Run on 30 October by the Salad method: the v0.6.9 release's AMD core (`--core amd`, built on each PC with ROCm 7.2)
against SRBMiner 3.8.0, after a 2-minute hit check. Every PC passed it. PeakMiner mines nothing on AMD (its README
lists AMD only for other coins), so "% of best" is ours divided by SRBMiner's rate. The release build of the AMD core
(`ubuntu-latest-hip` in `native-core.yml`) uses the same image and compiler, ROCm 7.2.0, as these boxes. Salad's
`amd-smi` reports nothing, so there are no power figures.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RX 9070 XT | Salad (16 GB) | a68b3382 | 114.4 | 0 | 111.3 | 103% |
| RX 9070 XT | Salad (16 GB) | 21001ac3 | 117.4 | 0 | 117.4 | 99.9% |
| RX 9060 XT | Salad (16 GB) | f9b8be8b | 61.2 | 0 | 56.1 | 109% |
| RX 9060 XT | Salad (16 GB) | cff5c764 | 61.7 | 0 | 56.2 | 110% |
| RX 7900 XTX | Salad (24 GB) | 9a57035a | 48.8 | 0 | 53.0 | 92% |
| RX 7900 XTX | Salad (24 GB) | 45c94e40 | 53.1 | 0 | 53.9 | 99% |
| RX 7900 XT | Salad (20 GB) | cc2f03b6 | 53.1 | 0 | 51.4 | 103% |
| RX 7900 XT | Salad (20 GB) | 67c308d2 | 47.2 | 0 | 47.8 | 99% |
| RX 7800 XT | Salad (16 GB) | d0dbba63 | 30.2 | 0 | 31.2 | 97% |
| RX 7800 XT | Salad (16 GB) | 53c8eb50 | 31.2 | 0 | 30.9 | 101% |

- **SRBMiner 3.8.0 closed most of our lead on RDNA 4.** On 23 October, against SRBMiner 3.7.1 on other PCs, ours led on
  every card (100.7–147.7%); SRBMiner read 69–92 on the 9070 XT and 43 on the 9060 XT then. On RDNA 3 the cards are now
  within a few percent either way.
- **The PC matters as much as the card.** RDNA 3 cards slow as they heat, so which miner runs first matters by a few
  percent. Some PCs were far slower for both miners (one 7900 XTX ran 12 times slower, with something else on its GPU);
  those aren't rows.
- **v0.6.12 is about 13% faster on RDNA 4** than these rows (the 4 November sections). The rows stay on v0.6.9 until
  the release is run against SRBMiner on them.
- **Before release (22–23 October),** PR #261's core went from 91–94% of SRBMiner on RDNA 3 and about half its rate on
  the RX 9060 XT to ahead on all ten PCs. The 9060 XT's gap was its cache: reading operands in per-tile order
  took the card from 22 to about 50 TH/s. Mining on Kryptex, a 9070 XT ran 92–94 TH/s for 31 minutes with every share
  accepted.

### Our own builds, 1 November: readout, k 4096 and batch width

- **Readout probe for two-lane regions (research item 2).** Skipping the readout's cross-lane steps (timing only) gave
  about +1.5% on a 9070 XT and +1.6% on a 9060 XT. But the real two-lane regions were level, and a second readout (v2,
  research item 4) lost 0.4–0.9%. Combining the lanes' words costs what the cross-lane steps cost. Both dropped; the
  probe stays as the gate for the next readout idea.
- **k 4096 on RDNA 4 (research item 5).** +1.03 to +1.11% on a 9070 XT and +1.14 to +1.24% on a 9060 XT, because the hash
  kernel's work halves. In the app it read +0.75% and +1.3%, with no cost from job switches. Kryptex and HeroMiners
  accepted k 4096 shares, and the spec's verifier passed every captured share. Shipped: RDNA 4 mines k 4096 on Linux.
  At rank 128 a MAC earns the same at either k.
- **RDNA 3 (research item 13).** k 4096 gained under +0.3% (+0.34% and +0.15%), so RDNA 3 stays at k 2048. Batch 128 was
  ahead of the release's 256 in every round on a 7900 XTX and a 7800 XT (+0.1 to +1.8% a round) and 64 was slower on
  both chips, so 128 is the default on gfx1100 and gfx1101. RDNA 4 was level at 128 and keeps 256. The hash kernel is
  about 0.8% of a batch on RDNA 3.
- **Ablations (item 13).** The slab barrier is 1.3–1.9% of fold time on a 7900 XTX and about three times that on RDNA
  4. Cutting LDS fragment loads from 24 a slab to 3 made a 9070 XT's fold 6–12% faster, an upper bound.
- **Item 15, built and dropped.** Moving the next slab's LDS store and barrier a k-step earlier was slower on both
  RDNA 4 cards in every round (by up to 4.1%): the global loads that feed it get less time to arrive.
- **Item 14, built and dropped.** Skewing each block's start helps one generation of blocks, then the barriers pull them
  back into step; at the miner's launch every wait was slower.

### Our own builds, 2 November: a fold for RDNA 2 (dot4)

RDNA 2 (RX 6000, gfx103x) has no matrix instruction and got the slow reference fold. The new `dot4` fold uses
`v_dot4` instead. Salad rents no RDNA 2 card, so it ran in RDNA 2's loop on an RX 7800 XT.

| Fold | Fold alone (TMAC/s) | `hashrate.js` (TH/s) |
|---|---|---|
| `wmma11`, the card's own | 32.81, 32.85 | 32.41, 32.61 |
| `dot4`, RDNA 2's loop | 18.09, 17.99, 18.00 | 17.88, 17.87 |
| `ref`, what RDNA 2 got until now | 4.22, 4.21 | 4.31, 4.32 |

- `dot4` matched the CPU bit for bit, and `verify-hits` re-verified 400 of 400 hits. It's 4.1x the reference fold.
- RDNA 3 and RDNA 4 release builds leave `dot4` out: it's slower there than their WMMA folds.

**Projected for RDNA 2 cards. A projection, not a measurement: no RDNA 2 card has run it.** It assumes one
`v_dot4_i32_i8` a lane a clock and the 96% of peak this 7800 XT reached, at each card's game clock.

| Card | CUs | Game clock (MHz) | Projected TH/s |
|---|---|---|---|
| RX 6600 | 28 | 2044 | 14.0 |
| RX 6700 XT | 40 | 2424 | 23.7 |
| RX 6800 | 60 | 1815 | 26.6 |
| RX 6800 XT | 72 | 2015 | 35.4 |
| Radeon PRO V620 | 72 | 2015 (the 6800 XT's) | 35.4 |
| RX 6900 XT | 80 | 2015 | 39.3 |
| RX 6950 XT | 80 | 2100 | 41.0 |

If RDNA 2's `v_dot4` runs at half rate, as RDNA 3's does, every figure halves (7.0 to 20.5 TH/s).

### Our own builds, 4 November: where an RX 9070 XT's fold time goes, and two cuts

A WMMA peak probe (`test/pearl_amd_peak.cpp`) shows the instruction does 1,009 MAC a CU a clock, 98.6% of AMD's 1,024.
The v0.6.11 fold ran at 70% of that. Each VALU instruction beside the WMMAs costs about a cycle of WMMA time (1.1–1.2
cycles if it waits on nothing, about 4.5 if chained), so the cuts target VALU. Two cuts are now RDNA 4's default:

- `PEARL_AMD_KPAIR`: each fragment pair is one 16-byte LDS load, with no address VALU.
- `PEARL_AMD_BUFLD`: the slab loads are buffer loads, with no VALU a load.

| Card | `hashrate.js` (TH/s): release | KPAIR | KPAIR + BUFLD |
|---|---|---|---|
| RX 9070 XT (`9a3e3210`) | 115.93, 115.79, 115.71 | 122.72, 122.60, 122.62 (+5.9%) | 125.58, 125.63, 125.61 (+8.5%) |
| RX 9070 XT (`635a1ec0`) | 107.17, 107.34 | 113.64, 113.57 (+5.9%) | not run |
| RX 9060 XT (`a4e51ee5`) | 62.41, 62.30, 62.30 | 66.20, 66.16, 66.18 (+6.2%) | 67.60, 67.58, 67.59 (+8.4%) |

- KPAIR's fold alone was 5.9–6.8% faster on an RX 9070 XT (+6.18% and +6.47%), and `verify-hits` passed 400 of 400 on
  both cards. RDNA 3 is unchanged.
- In a duel on Kryptex, our CLI with both cuts read 125.75 TH/s against SRBMiner 3.8.2's 117.89 on the same 9070 XT
  (106.7%). Not a row: it's another PC.
- **Dropped:** a scalar slab base (`PEARL_AMD_SADDR`, −4.0 to −4.6%), a 64-byte slab (−10.9%), and a fold with no LDS
  (it only matched KPAIR + BUFLD before its hash).

### Our own builds, 4 November: six more cuts on RDNA 4

Six switches, on by default for gfx12 (`PEARL_AMD_RDNA4_SET`) and raced against KPAIR + BUFLD:

- `PEARL_AMD_RDSEL=1` and `PEARL_AMD_RDPERM=2`: a cheaper wrap readout: a scalar branch where the compiler had three
  `v_cndmask` a word (+1.15% alone on an RX 9070 XT).
- `PEARL_AMD_CRANK=1` with `PEARL_AMD_SCHEDB=1`: rank 128 at compile time, with a `sched_barrier` that stops the
  compiler moving the next slab's stores above the WMMAs. The host refuses any other rank on such a card.
- `PEARL_AMD_LDS32=1` with `PEARL_AMD_LDS32_PIN=1`: 32-byte LDS rows with pinned offsets, 24,576 bytes a block instead
  of 36,864.

| Card | PC | `hashrate.js`: KPAIR + BUFLD | The set | Gain |
|---|---|---|---|---|
| RX 9070 XT | `5353d048` | 110.05, 109.71, 109.57 | 115.06, 114.76, 114.75 | +4.63% |
| RX 9070 XT | `d481bdd9` | 115.30, 115.26, 115.25, 115.18 | 120.37, 120.33, 120.33, 120.33 | +4.42% |
| RX 9070 XT (fresh PC, ABBA) | `3d6cb276` | 112.43 | 117.26 | +4.29% |
| RX 9060 XT | `d706e2f4` | 50.67, 50.93, 50.95 | 53.71, 53.24, 53.60 | +5.24% |
| RX 9060 XT | `d706e2f4` | 50.79, 50.72, 51.33, 50.68 | 53.73, 53.67, 53.47, 53.65 | +5.40% |
| RX 9060 XT (fresh PC, ABBA) | `b9b8ec1e` | 68.63 | 71.80 | +4.63% |

- **Ahead in every scored round on every PC,** and +4.29% and +4.63% on fresh PCs, with verify-hits 400 of 400 on both
  cards every time (at k 4096 and k 8192). Every fold check was bit-exact.
- **The parts don't add up:** alone they read +1.2 to +3.4%, together +4.3 to +5.4%. The cards are at their power
  limit, so fewer cycles a slab also means a lower clock.
- **Tried and dropped:** operand fill (level from 48 to 63); the keeping lane by exec mask (+0.51 and +0.73% fold alone,
  below RDSEL alone at +0.78%); a split slab barrier (−1.1% and −1.4% on top of the set); 8 waves a SIMD (−2.1 to −2.6% on the set);
  batch width and hash-kernel changes outside the fold (none worth keeping).
- The Windows core leaves the set out (`-DPEARL_AMD_RDNA4_SET=0`): it was only timed with ROCm's clang. RDNA 3 code is
  unchanged.

### k 8192 by default on the big gfx1201 cards (4 November)

On top of the set, k 8192 read +0.45% and +0.82% on an RX 9070 XT but −5.8% and −5.9% on an RX 9060 XT, whose fold
alone is about 7% slower there. So the host picks k 8192 on gfx1201 with at least 56 CUs (RX 9070 XT, R9700, RX 9070)
and k 4096 on the rest of RDNA 4. Checked with the v0.6.12 release build:

| Card (PC) | Default k | `hashrate.js` | Pool run | `verify-hits` |
|---|---|---|---|---|
| RX 9070 XT (`0e2d082f`) | 8192 | 132.09 against 131.17 TH/s at k 4096 (+0.70%) | HeroMiners, 15 min: 13 of 13 accepted | 400/400 at k 8192 |
| RX 9060 XT (`f029e99c`) | 4096 | 69.49 to 69.52 TH/s | HeroMiners, 10 min: 8 of 8; at k 8192, 3 of 3 | 400/400 at k 4096 |

- **HIP counts WGPs:** on a third 9070 XT, HIP's `multiProcessorCount` read 32 where rocminfo says 64 CUs, so the gate
  takes 28–39 as WGPs or 56 and up as CUs.
- The 48-CU RX 9070 GRE stays at k 4096 (nothing between 32 and 64 CUs was timed). The RX 9070 and R9700 get k 8192
  untimed, on the 9070 XT's result.

## Hosts replaced in this run

On 10 October these hosts were dropped and the row ran on the next one by the same rules:

- Container didn't start within 10 minutes: RTX 3070 Ti Pennsylvania, RTX 3060 Romania, RTX 4090 Finland, RTX 4080
  Super California.
- CUDA found no GPU: RTX 4070 Ti Super Arizona. Setup failed: A100 SXM4 California.
- Outbound port 1200 blocked: RTX 5070 Quebec, L4 Brazil. Went offline mid-run: RTX 5070 France.
- Another workload on the GPU: RTX 3090 Ti Spain. Held by heat: RTX 3080 Utah, RTX 3060 United States.
- Our test script failed and the row moved: RTX 3060 Utah, H100 SXM Czechia, RTX 4080 Super Indiana, L40S Taiwan.
