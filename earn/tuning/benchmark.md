# Benchmarks

Our miner against PeakMiner and SRBMiner, 5 minutes each on the same card and pool, and since 9 November against
OneZeroMiner too, a top-5 Pearl miner by downloads (NVIDIA only). WildRig Multi isn't measured: it mines NVIDIA
cards through OpenCL, which Salad's WSL2 containers don't have. "% of best" is our rate over the fastest competitor's. Vast rows ran v0.5.14 on 10–11 October; Salad rows ran v0.6.9 (October), v0.6.13 (9–10 November), PR #296's build (the RTX rows, at b822bb6e, and the Radeon rows on Salad, at 21a6e31b, both 14 November) and next/after-v0.7.2 at 1975683d (every Salad row of the official table, 15 November) on home PCs, so compare "% of
best" between them, not TH/s. The old file with every run's notes:
[benchmark.md at 16192d68](https://github.com/super3/llmjob/blob/16192d68f40b9c7ba11048e1128e7ba2c5509dca/benchmark.md).

## Summary

| Cards | Latest Salad run | Note |
|---|---|---|
| RTX 20-series | 96.5–101.4%, median 99.8%, 4 of 11 above 100% (15 Nov, next/after-v0.7.2 at 1975683d; PeakMiner 2.20.9, SRBMiner 3.8.8, OneZeroMiner 1.8.6; whole-run means) | 11 of 11 rows new; lowest RTX 2070 Super at 96.5% |
| RTX 30-series | 97.4–100.2%, median 98.4%, 1 of 16 above 100% (15 Nov, next/after-v0.7.2 at 1975683d; PeakMiner 2.20.9, SRBMiner 3.8.8, OneZeroMiner 1.8.6; whole-run means) | 16 of 16 rows new; lowest RTX 3080 Ti at 97.4% |
| RTX 40-series | 96.7–99.6%, median 98.8%, 0 of 18 above 100% (15 Nov, next/after-v0.7.2 at 1975683d; PeakMiner 2.20.9, SRBMiner 3.8.8, OneZeroMiner 1.8.6; whole-run means) | 17 of 18 rows new; lowest RTX 4070 Super at 96.7% |
| RTX 50-series | 98.5–99.8%, median 99.3%, 0 of 12 above 100% (15 Nov, next/after-v0.7.2 at 1975683d; PeakMiner 2.20.9, SRBMiner 3.8.8, OneZeroMiner 1.8.6; whole-run means) | 12 of 12 rows new; lowest RTX 5090 at 98.5% |
| GTX 16-series | 96–104% (17 Oct, v0.6.0) | 0.7–1.3 TH/s; not worth mining |
| AMD Radeon | 94.0–120.9%, median 95.9% (15 Nov: Salad rows at 1975683d against SRBMiner 3.8.8, whole-run means; RDNA 2 on Chedr's lent HiveOS rig, 1400 MHz cap) | Salad RDNA 3 and 4 rows RX 9070 XT 95.5%, RX 9070 XT 94.0%, RX 9060 XT 94.2%, RX 9060 XT 96.3%, RX 7900 XTX 95.2%, RX 7900 XTX 94.9%, RX 7900 XT 95.9%, RX 7900 XT 94.3%, RX 7800 XT 108.0%, RX 7800 XT 98.6%; RX 6800 120.9% and two RX 6600s 120.7% and 120.4% against 3.8.8 |
| Intel Arc | Not Salad: one rented rig of six Arc A750s. 124% against IAminer 1.0.0 on one card, 128% on six (10 Nov, PR #290's SYCL core, in no release). With the fragment fold of 11 Nov (branch `goal/arc-kernel`, not on the PR): 470% of IAminer's 10 Nov figure on a cool card, 423 to 449% on six | 16.9 TH/s a card on the PR; 64 on a cool card with the fragment fold and 55 to 58 a card with six running hot, at the card's 190 W limit; the matrix engines alone do 113 T-MAC/s; 10.2 on HiveOS's own 2024 driver (first fold). SRBMiner and WildRig mine nothing on Arc; the pool rejected both shares IAminer found |

## The official table, 15 November (next/after-v0.7.2 at 1975683d; PeakMiner 2.20.9, SRBMiner 3.8.8, OneZeroMiner 1.8.6)

**How rows are scored (from 15 November):** every miner's figure, ours and each competitor's, is the average of
the hashrate it displays over its whole 5-minute run, startup included (`salad/bench.py`, the founder's rule). No
warm-up cut, no single reading, no credited shares. Rows before this table used our mean after the first minute
against each competitor's last reading after the first minute. Each result keeps every sample, so old rows can be
re-scored.

Ours is built on each Salad PC from next/after-v0.7.2 at `1975683dcb6d` (v0.7.3 plus the new competitor pins): the
RDNA 4 and RDNA 3 kernel wins, the RTX 50 fold files (`pearl_rot_sm120.cubin` `8584f90b22f0` and the 5090's
`pearl_rot_sm120w.cubin` `c580e56056be`), and the fair scoring. Competitors as pinned in `salad/bench.sh`: PeakMiner
2.20.9 (15 November, "Pearl hashrate efficiency improved on RTX 20/30/40/50"), SRBMiner 3.8.8 and OneZeroMiner
1.8.6; on AMD only SRBMiner runs. Each miner runs 5 minutes on the same PC; the order alternates between the two PCs of
a card. The boxes ran `bench.sh` from 1975683d, then f41d96cd and 5a44d2ab (fixes for a failed build and for
stopping PeakMiner; the two rows f41d96cd spoiled were rerun); only `bench.sh` differs, our code is 1975683d on every
PC. Groups ran in the llmjob2 org, up to six at a time. A row measured on an older version than its
column shows that version in its cell. At most two rows a card.

- 70 rows: 66 new on Salad, 1 old, and Chedr's three RDNA 2 rows (raced on 15 November).
- RTX 20: 4 of 11 above 100%, 96.5–101.4%, median 99.8%. RTX 30: 1 of 16 above 100%, 97.4–100.2%, median 98.4%. RTX 40: 0 of 18 above 100%,
  96.7–99.6%, median 98.8%. RTX 50: 0 of 12 above 100%, 98.5–99.8%, median 99.3%. AMD Radeon: 4 of 13 above 100% (the three RDNA 2 rows and one RX 7800 XT; 1 of 10 Salad rows), 94.0–120.9%, median 95.9%.
- The best competitor on the 56 new RTX rows: PeakMiner 2.20.9 on 44, SRBMiner 3.8.8 on 9, OneZeroMiner 1.8.6 on 3.
- Lowest RTX rows: RTX 2070 Super `50834930` 96.5%, RTX 2080 `885c458c` 97.1%, RTX 3080 Ti `fbc84995` 97.4%, RTX 3080 Ti `ca4c98d6` 97.7%, RTX 2060 `e2c5c0fd` 97.8%, RTX 3060 Ti `808fba0f` 97.9%. Lowest Radeon rows: RX 9070 XT `22f73721` 94.0%, RX 9060 XT `78285119` 94.2%, RX 7900 XT `c2b8614b` 94.3%.
- Left out, each rerun on another PC: RTX 2080 `7dde7a7e` (86 °C in the first leg; the miners after it fell 2-9%),
  `e188cb9d` (throttled to about 420 MHz and 70 W: every miner 17.8-19.3 TH/s) and `11a76764` (its card line never
  reached Salad's logs, and the class also hands out 2080 Supers); RTX 4060 class `f9712745` (no card line and no
  power limit, so the card is unconfirmed); RTX 5080 `36ce74d8` (our build failed when node-gyp could not download
  Node's headers, and `bench.sh` then could not find PeakMiner: fixed in f41d96cd); RTX 5090 `48c67dd4` (PeakMiner
  2.20.9 exited at once with code 127, and the PC was erratic: SRBMiner 351.7 against 439 on `17f5597a`); RTX 4070
  Ti Super `14528113` and RTX 5090 `2b31ad9b` (f41d96cd left PeakMiner running after its leg, so SRBMiner and
  OneZeroMiner shared the card with it; fixed in 5a44d2ab).
- Kept with a note: RX 7800 XT `dc17e5be` (108.0%): SRBMiner mined steadily the whole run at 29.7-30.5 TH/s and read
  30.75 on the same PC earlier on 15 November (goal-rdna3c); the PC's copy rate is 452 GB/s against 476-488 where
  SRBMiner read 34.1-34.5. RTX 3070 Ti `0718be56` was recorded by hand from its samples (no card line; Salad's 3070 Ti
  class and its 290 W limit). The second RTX 2070 and RTX 4070 Super rows were rerun on 16 November at 5a44d2ab (0.7.3 build, same miners):
  the 2070 replaces `5a7c776a`, a slow PC (39.7-41.4 TH/s for every miner), and its PC reported no power limit
  either; the 4070 Super replaces `4a38a790`, raced on 0.7.2 and PeakMiner 2.20.8.

#### RTX 20-series

Dev fees in the headers: ours 0% (`devFeePct: 0` in earn/src/shared/config.js), PeakMiner 2% (its README's "Supported coins & dev fees" table), SRBMiner 2% (its README's algorithm list, "[2.00%] pearlhash"), OneZeroMiner 1% (its README's table, "pearlhash 1%"). "% of best" compares displayed hashrates.

4 of 11 above 100%, median 99.8%.

| Card | Host | Machine ID | Ours 0.7.3 (0% fee) | PeakMiner 2.20.9 (2% fee) | SRBMiner 3.8.8 (2% fee) | OneZeroMiner 1.8.6 (1% fee) | % of best |
|---|---|---|---|---|---|---|---|
| RTX 2080 Ti | Salad (200 W) | 49981401 | 83.5 | 81.3 | 80.5 | 83.0 | 100.7% |
| RTX 2080 Ti | Salad (260 W) | 131cf457 | 107.0 | 104.4 | 102.6 | 105.5 | 101.4% |
| RTX 2080 | Salad (215 W) | 6c2ac902 | 68.4 | 68.5 | 67.9 | 68.2 | 99.9% |
| RTX 2080 | Salad (260 W) | 885c458c | 79.0 | 81.3 | 80.0 | 80.1 | 97.1% |
| RTX 2070 Super | Salad (215 W) | 50834930 | 62.7 | 65.0 | 60.7 | 63.7 | 96.5% |
| RTX 2070 Super | Salad (215 W) | 22b920ea | 64.5 | 64.2 | 59.0 | 60.9 | 100.4% |
| RTX 2070 | Salad (175 W) | 0e693eaa | 52.2 | 53.0 | 52.1 | 52.8 | 98.6% |
| RTX 2070 | Salad | b533245d | 53.0 | 53.5 | 52.7 | 53.3 | 99.1% |
| RTX 2060 Super | Salad (175 W) | dd94ad76 | 54.1 | 54.4 | 54.0 | 54.5 | 99.4% |
| RTX 2060 | Salad (160 W, 6 GB) | 77b0b863 | 39.4 | 39.5 | 38.9 | 38.8 | 99.8% |
| RTX 2060 | Salad (172.50 W, 6 GB) | e2c5c0fd | 46.4 | 47.5 | 46.5 | 46.1 | 97.8% |

#### RTX 30-series

Dev fees in the headers: ours 0% (`devFeePct: 0` in earn/src/shared/config.js), PeakMiner 2% (its README's "Supported coins & dev fees" table), SRBMiner 2% (its README's algorithm list, "[2.00%] pearlhash"), OneZeroMiner 1% (its README's table, "pearlhash 1%"). "% of best" compares displayed hashrates.

1 of 16 above 100%, median 98.4%.

| Card | Host | Machine ID | Ours 0.7.3 (0% fee) | PeakMiner 2.20.9 (2% fee) | SRBMiner 3.8.8 (2% fee) | OneZeroMiner 1.8.6 (1% fee) | % of best |
|---|---|---|---|---|---|---|---|
| RTX 3090 Ti | Salad (450 W) | 2ddb782a | 165.4 | 167.3 | 166.7 | 166.0 | 98.9% |
| RTX 3090 Ti | Salad (450 W) | 2a7c8faa | 160.2 | 162.8 | 162.1 | 161.2 | 98.4% |
| RTX 3090 | Salad (420 W) | a1653a9c | 141.4 | 142.8 | 141.9 | 142.0 | 99.0% |
| RTX 3090 | Salad (350 W) | f5b2cc5b | 139.6 | 141.9 | 141.4 | 140.2 | 98.3% |
| RTX 3080 Ti | Salad (350 W) | ca4c98d6 | 97.6 | 100.0 | 99.6 | 99.2 | 97.7% |
| RTX 3080 Ti | Salad (350 W) | fbc84995 | 114.5 | 116.7 | 117.6 | 115.5 | 97.4% |
| RTX 3080 | Salad (370 W) | 23cb961e | 102.9 | 103.6 | 103.0 | 101.4 | 99.3% |
| RTX 3080 | Salad (288.60 W) | 87a7b4d1 | 117.0 | 119.0 | 118.8 | 117.9 | 98.3% |
| RTX 3070 Ti | Salad (290 W) | 0718be56 | 95.0 | 94.8 | 94.8 | 93.4 | 100.2% |
| RTX 3070 Ti | Salad (310 W) | 241f649d | 93.5 | 94.5 | 94.5 | 93.1 | 99.0% |
| RTX 3070 | Salad (280 W) | 23ef6a93 | 88.9 | 89.3 | 89.2 | 88.6 | 99.5% |
| RTX 3070 | Salad (204 W) | bdfe76b9 | 82.7 | 84.4 | 84.1 | 81.8 | 98.0% |
| RTX 3060 Ti | Salad (216 W) | 808fba0f | 71.8 | 72.5 | 73.4 | 72.3 | 97.9% |
| RTX 3060 Ti | Salad (200 W) | 15c747f0 | 64.8 | 65.7 | 66.0 | 65.4 | 98.1% |
| RTX 3060 | Salad (170 W, 12 GB) | c7080b06 | 53.6 | 53.9 | 50.6 | 52.6 | 99.5% |
| RTX 3060 | Salad (170 W, 12 GB) | 9144f25d | 54.9 | 55.0 | 55.8 | 54.8 | 98.4% |

#### RTX 40-series

Dev fees in the headers: ours 0% (`devFeePct: 0` in earn/src/shared/config.js), PeakMiner 2% (its README's "Supported coins & dev fees" table), SRBMiner 2% (its README's algorithm list, "[2.00%] pearlhash"), OneZeroMiner 1% (its README's table, "pearlhash 1%"). "% of best" compares displayed hashrates.

0 of 18 above 100%, median 98.8%.

| Card | Host | Machine ID | Ours 0.7.3 (0% fee) | PeakMiner 2.20.9 (2% fee) | SRBMiner 3.8.8 (2% fee) | OneZeroMiner 1.8.6 (1% fee) | % of best |
|---|---|---|---|---|---|---|---|
| RTX 4090 | Salad (450 W) | 30ab123f | 341.4 | 346.0 | 345.0 | 341.8 | 98.7% |
| RTX 4090 | Salad (450 W) | 6c6c50a9 | 341.9 | 346.7 | 345.1 | 343.4 | 98.6% |
| RTX 4080 Super | Salad (320 W) | 14c7fd58 | 221.8 | 225.3 | 225.2 | 223.4 | 98.5% |
| RTX 4080 Super | Salad (320 W) | dc17aa71 | 218.0 | 220.0 | 218.5 | 217.6 | 99.1% |
| RTX 4080 | Salad (320 W) | bb1bf9b5 | 213.1 | 215.9 | 215.3 | 204.7 | 98.7% |
| RTX 4080 | Salad (320 W) | c5e1b53b | 203.4 | 204.9 | 203.8 | 204.0 | 99.3% |
| RTX 4070 Ti Super | Salad (285 W) | 0026caee | 180.2 | 181.4 | 181.2 | 172.5 | 99.3% |
| RTX 4070 Ti Super | Salad (285 W) | 31cb1001 | 182.6 | 184.9 | 184.6 | 175.9 | 98.8% |
| RTX 4070 Ti | Salad (285 W) | e7bb5e5a | 161.3 | 161.9 | 161.5 | 160.0 | 99.6% |
| RTX 4070 Ti | Salad (285 W) | d0a29348 | 160.0 | 161.8 | 162.0 | 158.9 | 98.8% |
| RTX 4070 Super | Salad (220 W) | 370e67d1 | 147.2 | 148.8 | 148.8 | 148.2 | 98.9% |
| RTX 4070 Super | Salad (245 W) | 19f6e26a | 154.2 | 156.4 | 156.0 | 156.1 | 98.6% |
| RTX 4070 | Salad (200 W) | 1d3e1dbd | 122.9 | 124.4 | 124.5 | 122.8 | 98.7% |
| RTX 4070 | Salad (200 W) | 3edcf921 | 126.8 | 127.9 | 128.0 | 127.6 | 99.0% |
| RTX 4060 Ti | Salad (165 W, 16 GB) | d8d9e2e9 | 87.2 | 87.8 | 87.5 | 86.9 | 99.3% |
| RTX 4060 Ti | Salad (165 W, 16 GB) | a955a654 | 94.5 | 95.0 | 94.9 | 94.2 | 99.5% |
| RTX 4060 | Salad (115 W) | e5b24ded | 65.1 | 65.3 | 65.4 | 64.3 | 99.5% |
| RTX 4060 | Salad (115 W) | 186e28cc | 65.6 | 66.1 | 66.6 | 63.2 | 98.5% |

#### RTX 50-series

Dev fees in the headers: ours 0% (`devFeePct: 0` in earn/src/shared/config.js), PeakMiner 2% (its README's "Supported coins & dev fees" table), SRBMiner 2% (its README's algorithm list, "[2.00%] pearlhash"), OneZeroMiner 1% (its README's table, "pearlhash 1%"). "% of best" compares displayed hashrates.

0 of 12 above 100%, median 99.3%.

| Card | Host | Machine ID | Ours 0.7.3 (0% fee) | PeakMiner 2.20.9 (2% fee) | SRBMiner 3.8.8 (2% fee) | OneZeroMiner 1.8.6 (1% fee) | % of best |
|---|---|---|---|---|---|---|---|
| RTX 5090 | Salad (600 W) | 17f5597a | 432.8 | 439.2 | 438.7 | 430.2 | 98.5% |
| RTX 5090 | Salad (517.50 W) | 6b3ba69b | 426.9 | 433.0 | 431.8 | 390.0 | 98.6% |
| RTX 5080 | Salad (360 W) | 1b88850e | 229.7 | 230.6 | 228.9 | 223.2 | 99.6% |
| RTX 5080 | Salad (360 W) | ca402047 | 237.8 | 239.9 | 238.3 | 227.9 | 99.1% |
| RTX 5070 Ti | Salad (300 W) | 3e0262af | 197.5 | 198.1 | 196.9 | 195.0 | 99.7% |
| RTX 5070 Ti | Salad (300 W) | 9b414265 | 180.3 | 180.7 | 179.8 | 162.7 | 99.8% |
| RTX 5070 | Salad (250 W) | 513660a3 | 135.7 | 136.3 | 135.5 | 134.8 | 99.5% |
| RTX 5070 | Salad (200 W) | fcba3f97 | 133.3 | 134.6 | 134.1 | 133.0 | 99.0% |
| RTX 5060 Ti | Salad (180 W, 16 GB) | 210dd643 | 92.1 | 92.4 | 91.5 | 87.2 | 99.7% |
| RTX 5060 Ti | Salad (180 W, 16 GB) | 09852d28 | 99.3 | 100.2 | 99.8 | 97.7 | 99.1% |
| RTX 5060 | Salad (145 W) | 04c3db84 | 75.5 | 75.9 | 75.5 | 73.8 | 99.6% |
| RTX 5060 | Salad (145 W) | b3ed56f3 | 78.5 | 79.4 | 79.1 | 78.1 | 98.9% |

#### AMD Radeon

Dev fees in the headers: ours 0%, SRBMiner 2% (its README's algorithm list, "[2.00%] pearlhash"). PeakMiner and OneZeroMiner have no AMD kernel and aren't run.

4 of 13 above 100%, median 95.9%.

| Card | Host | Machine ID | Ours 0.7.3 (0% fee) | PeakMiner 2.20.9 (2% fee) | SRBMiner 3.8.8 (2% fee) | OneZeroMiner 1.8.6 (1% fee) | % of best |
|---|---|---|---|---|---|---|---|
| RX 9070 XT | Salad (16 GB) | 0798158d | 143.7 | — | 150.4 | — | 95.5% |
| RX 9070 XT | Salad (16 GB) | 22f73721 | 141.5 | — | 150.6 | — | 94.0% |
| RX 9060 XT | Salad (16 GB) | 78285119 | 73.7 | — | 78.3 | — | 94.2% |
| RX 9060 XT | Salad (16 GB) | 48343577 | 70.4 | — | 73.1 | — | 96.3% |
| RX 7900 XTX | Salad (24 GB) | 91701547 | 54.3 | — | 57.0 | — | 95.2% |
| RX 7900 XTX | Salad (24 GB) | 5ae23c47 | 59.5 | — | 62.6 | — | 94.9% |
| RX 7900 XT | Salad (20 GB) | bb1b51f0 | 48.2 | — | 50.3 | — | 95.9% |
| RX 7900 XT | Salad (20 GB) | c2b8614b | 48.7 | — | 51.6 | — | 94.3% |
| RX 7800 XT | Salad (16 GB) | dc17e5be | 32.4 | — | 30.0 | — | 108.0% |
| RX 7800 XT | Salad (16 GB) | 0faaa5fc | 32.3 | — | 32.7 | — | 98.6% |
| RX 6800 | Chedr | GPU 3 | 19.36 (d2c8e93a) | — | 16.01 | — | 120.9% |
| RX 6600 | Chedr | GPU 0 | 8.91 (d2c8e93a) | — | 7.38 | — | 120.7% |
| RX 6600 | Chedr | GPU 1 | 8.88 (d2c8e93a) | — | 7.37 | — | 120.4% |

The RX 6800 and RX 6600 rows are Chedr's lent HiveOS rig with the core clock capped at 1,400 MHz, raced on 15 November against SRBMiner 3.8.8, and scored as each miner's mean after a 1-minute warm-up, not the whole-run mean of the Salad rows.

### Salad, 16 November: displayed against proved hashrates (the measurement audit)

Every row above compares what each miner displays. To check that the displays are honest, five runs put every
miner on bench.sh's stand-in pool at 2^209 (about 1 share/s at 140 TH/s), 15 minutes each, so each run also has
the rate its accepted shares prove. Proved is the rate over the window less its first minute (`ths60`); a rate
from n shares is good to about 1/sqrt(n), so about ±3.5% at 140 TH/s and ±5% on the 2070s. bench.sh at bad49716,
ours built at 1975683d with the rotation fold, PeakMiner 2.20.9, SRBMiner 3.8.8, OneZeroMiner 1.8.6.

| Card | PC | Ours shown → proved | PeakMiner shown → proved | SRBMiner shown → proved | OneZeroMiner shown → proved |
|---|---|---|---|---|---|
| RTX 2070 Super | 988d0fe7 | 66.9 → 68.4 (+2.2%) | 68.7 → 70.1 (+2.1%) | 67.3 → 71.0 (+5.6%) | 67.6 → 68.0 (+0.5%) |
| RTX 3080 Ti | 78416115 | 132.2 → 132.1 (−0.1%) | 134.4 → 131.4 (−2.2%) | 135.3 → 121.5 (−10.2%) | 133.7 → 132.1 (−1.2%) |
| RTX 4070 Super | 8ff75fdc | 139.0 → 138.4 (−0.4%) | 142.1 → 146.9 (+3.4%) | 141.8 → 144.8 (+2.1%) | 141.7 → 142.4 (+0.5%) |
| RTX 4070 | adf17705 | 122.5 → 121.4 (−0.9%) | 124.4 → 109.0 (−12.4%) | 124.3 → 119.5 (−3.9%) | no rate |
| RTX 2070 | 5616fc2b | 47.2 → 48.2 (+2.1%) | 47.9 → 45.1 (−5.8%) | 47.3 → 48.0 (+1.5%) | no rate |
| Mean | | +0.6% | −3.0% | −1.0% | −0.1% (3 runs) |

- No miner's display runs high beyond what its fee explains. On the mean, PeakMiner proves 3.0% under its display
  and SRBMiner 1.0%, close to their 2% dev fees, which mine to their own pool; ours proves what it shows.
- Ours is steady run to run (−0.9% to +2.2%). The competitors swing much more (PeakMiner −12.4% to +3.4%, SRBMiner
  −10.2% to +5.6%), most likely because their fee time comes in blocks a 15-minute window catches or misses.
- So the gaps in the tables are real: the displayed figures are a fair basis. What a pool credits differs from them
  by about each miner's fee.
- OneZeroMiner mined nothing on two of the five PCs (no rate, no shares).

## Fleet spot checks: SRBMiner 3.8.9 (17 November)

SRBMiner 3.8.9 came out on 17 November with "minor efficiency improvements" on pearlhash for RTX 30, 40 and 50 cards.
Bench mode in the Salad watcher races it inside production boxes: v0.7.3 against PeakMiner 2.20.9, SRBMiner 3.8.9 and
OneZeroMiner 1.8.6, 5 minutes each, on the box's own PC (spot.sh at 5143004c). Up to two checks per GPU class.

The dashboard and llmjob.com/benchmarks show these in place of the official table's older rows: a card keeps its two
newest races, and a figure is yellow (amber on the site) only when its own row ran an older version than the column names.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner 3.8.9 | OneZeroMiner | % of best |
|---|---|---|---|---|---|---|---|
| RTX 3060 (12 GB) | Salad | 38873e10 | 52.6 | 53.6 | 53.4 | 51.8 | 98.2% |
| RTX 3060 (12 GB) | Salad | a8da9ded | 52.1 | 52.5 | 52.7 | 49.7 | 98.8% |
| RTX 3060 Ti (8 GB) | Salad | 13a0f425 | 69.4 | 70.1 | 70.3 | 69.5 | 98.7% |
| RTX 3060 Ti (8 GB) | Salad | 64a64b5c | 68.7 | 69.1 | 69.6 | 69.0 | 98.7% |
| RTX 2080 Ti (11 GB) | Salad | 5499bafa | 84.9 | 86.3 | 88.2 | 87.3 | 96.2% |
| RTX 2080 Ti (11 GB) | Salad | dbcb4344 | 86.4 | 85.0 | 87.2 | 86.1 | 99.0% |
| RTX 4070 Ti (12 GB) | Salad | 3ce02762 | 172.0 | 163.7 | 164.5 | 162.8 | 104.6% |
| RTX 4070 Ti (12 GB) | Salad | 61e44847 | 157.5 | 158.7 | 158.6 | 157.1 | 99.3% |
| RTX 3090 Ti (24 GB) | Salad | efd04f2b | 159.6 | 160.7 | 160.6 | 160.0 | 99.3% |
| RTX 5090 (32 GB) | Salad | 98706283 | 428.1 | 442.4 | 447.9 | 391.5 | 95.6% |
| RTX 3080 (10 GB) | Salad | f4db02b9 | 116.8 (v0.7.4) | 119.0 | 119.4 | 113.5 | 97.8% |
| RTX 3080 (10 GB) | Salad | 097ebb95 | 99.6 (v0.7.5) | 100.0 | 99.6 | 97.7 | 99.6% |
| RTX 3090 Ti (24 GB) | Salad | 2f47a848 | 161.5 (v0.7.5) | 163.5 | 162.9 | 155.1 | 98.7% |
| RTX 4080 SUPER (16 GB; Salad lists it as RTX 4080) | Salad | f55874c9 | 219.1 (v0.7.5) | 223.0 | 223.6 | 221.7 | 98.0% |
| RTX 4080 (16 GB) | Salad | 5c95670b | 198.3 (v0.7.5) | 201.3 | 201.5 | 200.0 | 98.4% |
| RTX 5080 (16 GB) | Salad | ef98e320 | 232.8 (v0.7.5) | 234.7 | 235.8 | 230.0 | 98.7% |
| RTX 5070 (12 GB) | Salad | 3d945622 | 136.0 (v0.7.5) | 136.7 | 137.1 | 134.0 | 99.2% |
| RTX 5070 (12 GB) | Salad | 3224f357 | 137.3 (v0.7.5) | 137.6 | 137.8 | 123.8 | 99.6% |

On the official table's two RTX 3060 PCs SRBMiner 3.8.8 ran 50.6 and 55.8; here 3.8.9 is 1.5% above ours, between
those two, so one PC can't yet say how much 3.8.9 gained. PeakMiner leads on this PC.

## RTX 2080 Ti: SRBMiner 3.8.8 against 3.8.9 on one PC (18 November)

One llmjob2 RTX 2080 Ti (g18-rtx20-2080ti-3, about 211 W, ours v0.7.5 built at db089e9e), 3 minutes a turn, in the
order SRB 3.8.9, SRB 3.8.8, ours, SRB 3.8.9, ours. "Shown" is each miner's displayed rate; "credited" is what our
share decoder scores from the shares the pool accepted over the same 3 minutes (noisy over so short a window).

| Turn | Shown TH/s | Credited TH/s | Shares |
|---|---|---|---|
| SRBMiner 3.8.9 | 84.4 | 78.1 | 202 |
| SRBMiner 3.8.8 | 81.6 | 75.0 | 193 |
| Ours | 82.9 | 89.8 | 231 |
| SRBMiner 3.8.9 | 84.4 | 82.4 | 212 |
| Ours | 82.6 | 83.6 | 215 |

On the shown rate SRBMiner 3.8.9 is 3.4% above 3.8.8 on this card, and ours is 98.0% of 3.8.9. On credited work ours
is ahead of both SRB versions in both of its turns. Nsight Systems traced no kernels for either SRBMiner version, so
their launch shapes can't be compared this way; Nsight Compute is blocked on Salad (ERR_NVGPUCTRPERM). Our 60 s hit
check failed with "no result" on the box before the race (a tooling fault to fix; both race turns of ours ran).

## bpm on a Vast RTX PRO 4500 (18 November)

A one-off look at blackwell_pearl_miner (bpm, 08d9c76a, built with CUDA 13.0.88) against the usual four, on one Vast
RTX PRO 4500 (machine 51804, 200 W cap, driver 580.173.02), one after another, 5 minutes each, scored by bench.py's
rule (the mean of every 15 s sample).

| Miner | TH/s |
|---|---|
| Ours v0.7.5 | 196.8 |
| PeakMiner 2.20.9 | 198.5 |
| SRBMiner 3.8.9 | 200.5 |
| OneZeroMiner 1.8.6 | 193.9 |
| bpm (default kernel u6) | about 176, 77 s only |

Ours is 98.1% of the best (SRBMiner) here; every miner read 2 to 5 TH/s above the official row's machine. bpm stopped
itself at 77 s at its default 85 C limit (the card ran 83-85 C under every miner), after a steady 176 TMAC/s, about
12% under SRBMiner at the same clocks, so it isn't a competitor to race on this card and wasn't rerun.

## RTX 20 to 50: the 14 November table, official until 15 November (PR #296 build; PeakMiner 2.20.8, SRBMiner 3.8.8, OneZeroMiner 1.8.6)

Scored the old way (our mean after the first minute against each competitor's last reading after the first minute). It replaced the 12–13 November table (PR #293 build against PeakMiner 2.20.7, SRBMiner 3.8.7
and OneZeroMiner 1.8.5), which stays under each series below as history.
Ours is built on each Salad PC from PR #296's branch at `b822bb6e`: the commit that pins the new competitor versions
(`34ab6b73`) plus the `bench.sh` fix that builds the RTX 5090's own fold file (`pearl_rot_sm120w.cubin`, hash
`c580e560`), so the 5090 rows mine on the file a release ships for them. Competitors: PeakMiner 2.20.8, SRBMiner 3.8.8
and OneZeroMiner 1.8.6, as pinned in `salad/bench.sh`. Each miner runs 5 minutes on the same PC; the order alternates
between the two PCs of a card. Ours mines with the rotation fold, on by default since v0.7.0 (RTX 20 has its own
kernel). "Run" says where a row comes from: "new" is this run, all four miners on that PC; "old" is a 12–13 November
row kept because Salad gave no PC of that card this time. Groups ran in the llmjob2 org, up to seven lanes at a time.

- 56 rows: 55 new, 1 old (the suspect 4080 Super row; Salad's 4080 class gave no 4080 Super this time).
- 2 new rows are above 100%: the RTX 5070 `a1047664` (100.2%) and the RTX 5060 Ti `81cce3b7` (100.1%). The RTX 5060
  `e4f17758` is at 99.97% and the RTX 5070 `1306abc3` at 99.9%.
- Median of the new rows 98.8% (12–13 November: 99.2% against the older competitor versions). 46 of 55 are 98–100%.
  Per series: RTX 20 98.5%, RTX 30 98.7%, RTX 40 98.5%, RTX 50 99.4%.
- The best competitor is now OneZeroMiner 1.8.6 on 32 of 55 rows, SRBMiner 3.8.8 on 18 and PeakMiner 2.20.8 on 5.
- Lowest: the RTX 2080 `6b76def0` at 94.2% (OneZeroMiner 77.2, PeakMiner 76.8, ours 72.7), then the RTX 4070 Super
  `4a38a790` (96.7%), RTX 3090 Ti `04f2a421` (96.8%), RTX 5060 `6f073405` (96.9%), RTX 3080 Ti `41a7e7b3` and RTX
  4080 `e28b91ec` (97.0%), RTX 4060 `4b77372c` (97.6%) and the RTX 5090 `2b31ad9b` (98.0%, SRBMiner 446.9 against
  our 438.0).
- Slow PCs, kept because every miner ran on the same card: the RTX 5090 `a409f43c` (every miner 338–345 at 575 W,
  against 423–447 on `2b31ad9b`) and the RTX 3080 Ti `1e0e40a7` (102–107).
- Left out: `dcc6dd89` (RTX 5090), a broken PC on which every miner read about 31 TH/s; `942e256e` (RTX 2080), at
  87 °C under load; `416ca6d6` (RTX 4060 Ti class), whose card line never reached Salad's logs, so the card is
  unconfirmed (it drew about 70 W at 2010 MHz; the miners read 68.7–69.5). Four PCs got no work from the pool and
  recorded nothing.
- The 4070 class gave two RTX 4070 Supers, the 2070 class two 2070 Supers and the 2060 class one 2060 Super; they
  are rows of their own.

#### RTX 20-series

11 of 11 rows rerun. Above 100%: 0 of 11. Median 98.5%.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | OneZeroMiner | % of best | Run |
|---|---|---|---|---|---|---|---|---|
| RTX 2080 Ti | Salad (222.50 W) | 99914c2d | 87.8 | 86.2 | 85.4 | 88.4 | 99.3% | new |
| RTX 2080 Ti | Salad (221 W) | 183b3f22 | 89.2 | 86.4 | 86.1 | 90.9 | 98.2% | new |
| RTX 2080 | Salad (225 W) | 6b76def0 | 72.7 | 76.8 | 75.8 | 77.2 | 94.2% | new |
| RTX 2080 | Salad (225 W) | d46a74f1 | 53.7 | 54.3 | 53.4 | 53.1 | 98.9% | new |
| RTX 2070 Super | Salad (215 W) | b44931b9 | 67.8 | 68.5 | 68.0 | 67.8 | 99.0% | new |
| RTX 2070 Super | Salad (163.40 W) | bb37dfde | 51.9 | 52.7 | 51.7 | 52.1 | 98.6% | new |
| RTX 2070 | Salad (175 W) | 70ce62f9 | 54.0 | 55.0 | 53.5 | 54.8 | 98.2% | new |
| RTX 2070 | Salad (140 W) | 701abaca | 51.8 | 52.0 | 51.4 | 52.8 | 98.1% | new |
| RTX 2060 Super | Salad (175 W) | dd94ad76 | 54.2 | 54.3 | 53.9 | 55.0 | 98.5% | new |
| RTX 2060 | Salad (170 W, 6 GB) | 856ba561 | 46.2 | 46.5 | 45.4 | 46.6 | 99.2% | new |
| RTX 2060 | Salad (153.52 W, 6 GB) | 4e3fb584 | 42.8 | 43.1 | 41.8 | 43.5 | 98.4% | new |

#### RTX 30-series

16 of 16 rows rerun. Above 100%: 0 of 16. Median 98.7%.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | OneZeroMiner | % of best | Run |
|---|---|---|---|---|---|---|---|---|
| RTX 3090 Ti | Salad (480 W) | 7f29b442 | 165.0 | 164.6 | 165.6 | 166.7 | 98.9% | new |
| RTX 3090 Ti | Salad (450 W) | 04f2a421 | 147.1 | 146.9 | 152.0 | 149.4 | 96.8% | new |
| RTX 3090 | Salad (315 W) | 26ffa9ba | 128.4 | 128.6 | 130.3 | 130.7 | 98.2% | new |
| RTX 3090 | Salad (357 W) | b65a1341 | 140.5 | 140.3 | 142.0 | 142.1 | 98.9% | new |
| RTX 3080 Ti | Salad (350 W) | 1e0e40a7 | 107.0 | 102.0 | 107.4 | 106.0 | 99.6% | new |
| RTX 3080 Ti | Salad (350 W) | 41a7e7b3 | 122.0 | 122.5 | 125.8 | 124.6 | 97.0% | new |
| RTX 3080 | Salad (320 W) | 4d08e2ef | 114.2 | 113.8 | 115.7 | 116.2 | 98.3% | new |
| RTX 3080 | Salad (340 W) | f5dc88ec | 122.4 | 122.1 | 123.3 | 123.6 | 99.0% | new |
| RTX 3070 Ti | Salad (229.10 W) | ca1fd23b | 86.7 | 86.2 | 87.9 | 87.9 | 98.6% | new |
| RTX 3070 Ti | Salad (310 W) | e7b729e8 | 85.6 | 85.5 | 87.2 | 86.7 | 98.2% | new |
| RTX 3070 | Salad (210.60 W) | 087be162 | 84.8 | 84.1 | 85.7 | 85.7 | 99.0% | new |
| RTX 3070 | Salad (220 W) | 26a8bf34 | 84.6 | 85.1 | 86.2 | 86.0 | 98.1% | new |
| RTX 3060 Ti | Salad (200 W) | 7ada1993 | 70.0 | 70.0 | 70.7 | 70.8 | 98.8% | new |
| RTX 3060 Ti | Salad (240 W) | d3ead61a | 73.2 | 72.9 | 73.6 | 73.6 | 99.5% | new |
| RTX 3060 | Salad (170 W, 12 GB) | 0acae8d0 | 52.0 | 52.0 | 52.9 | 52.8 | 98.2% | new |
| RTX 3060 | Salad (170 W, 12 GB) | 6be91d5b | 54.1 | 53.8 | 54.4 | 54.2 | 99.4% | new |

#### RTX 40-series

16 of 17 rows rerun. Above 100%: 0 of 16. Median 98.5%.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | OneZeroMiner | % of best | Run |
|---|---|---|---|---|---|---|---|---|
| RTX 4090 | Salad (450 W) | 98d76dd7 | 347.1 | 345.4 | 350.4 | 352.5 | 98.5% | new |
| RTX 4090 | Salad (450 W) | 75e726c8 | 347.4 | 346.3 | 351.7 | 353.6 | 98.2% | new |
| RTX 4080 Super | Salad (256 W) | 030da504 | 206.7 | 235.5 (suspect) | 209.1 | 204.8 | 87.8% | old |
| RTX 4080 | Salad (307.20 W) | c5c0cfdc | 212.1 | 211.2 | 215.3 | 212.9 | 98.5% | new |
| RTX 4080 | Salad (272 W) | e28b91ec | 201.1 | 200.2 | 203.3 | 207.3 | 97.0% | new |
| RTX 4070 Ti Super | Salad (305 W) | 35a46f0e | 138.5 | 138.1 | 138.9 | 138.6 | 99.7% | new |
| RTX 4070 Ti Super | Salad (285 W) | b84f4f0a | 179.0 | 178.0 | 179.2 | 179.8 | 99.5% | new |
| RTX 4070 Ti | Salad (285 W) | 5098c9b4 | 170.4 | 169.3 | 171.2 | 172.1 | 99.1% | new |
| RTX 4070 Ti | Salad (285 W) | c60a79c1 | 163.5 | 163.4 | 163.5 | 164.9 | 99.1% | new |
| RTX 4070 Super | Salad (176 W) | 7239f89b | 148.4 | 148.3 | 150.9 | 150.7 | 98.3% | new |
| RTX 4070 Super | Salad (220 W) | 4a38a790 | 148.6 | 147.9 | 152.1 | 153.7 | 96.7% | new |
| RTX 4070 | Salad (215 W) | 5edc8cc1 | 129.0 | 127.7 | 130.4 | 130.4 | 98.9% | new |
| RTX 4070 | Salad (200 W) | 8a2ff801 | 126.0 | 125.4 | 128.1 | 128.1 | 98.3% | new |
| RTX 4060 Ti | Salad (165 W, 16 GB) | 39c86a93 | 90.4 | 89.7 | 90.8 | 91.0 | 99.3% | new |
| RTX 4060 Ti | Salad (165 W, 16 GB) | 0b9d5c5d | 92.3 | 91.7 | 93.5 | 93.3 | 98.7% | new |
| RTX 4060 | Salad (115 W) | b2662123 | 56.4 | 55.7 | 57.4 | 56.5 | 98.4% | new |
| RTX 4060 | Salad (115 W) | 4b77372c | 64.2 | 63.9 | 64.9 | 65.8 | 97.6% | new |

##### Vast, 18-19 November: RTX 4080 SUPER and RTX 4080 against SRBMiner 3.8.9, and goal-nvk24's exchange trims shipped (+0.1-0.2%) (branch `goal/rtx40c`)

Two Vast boxes, one at a time: an RTX 4080 SUPER (machine 137067, 320 W, driver 580.173) and an RTX 4080
(machine 35189, 320 W, driver 615.71). bench.sh on Kryptex, 60 s warm-up, 5 minutes each; ours is the v0.7.5
release with its fold cubin. Competitors by observation only (their API, `nvidia-smi` once a second). Ours is the
mean of its lines; a competitor's figure is its last sample.

| Card | Ours (two legs) | PeakMiner 2.20.9 | SRBMiner 3.8.9 | OneZeroMiner 1.8.6 | % of best |
|---|---|---|---|---|---|
| RTX 4080 SUPER | 223.65, 223.38 (2758 MHz, 318 W, power cap 100%) | 226.69 (2790 MHz, 309 W) | 226.08 (2790 MHz, 305 W, cap 3%) | 225.92 (2789 MHz, 309 W) | 98.6% |
| RTX 4080 | 210.31, 209.57 (2726 MHz, 319 W, cap 100%) | 212.58 (2744 MHz, 319 W) | 212.20 (2749 MHz, 319 W) | 212.65 (2743 MHz, 319 W) | 98.7% |

- The gap is clock at the same watts, as goal-nvk25 and nvk26 found; the work a clock is level. On the 4080 SUPER,
  TH/s a GHz: ours 81.03, SRBMiner 81.03, OneZeroMiner 80.97, PeakMiner 81.25 (the card's peak is 81.92). There the
  three competitors reach the card's top clock (2790 MHz, held by voltage and temperature) at 305-309 W and are
  hardly capped. Ours needs 318 W for 2758 MHz and sits on the cap. On the 4080 all four miners are capped. We lead
  OneZeroMiner by 0.5% in clock (2726 against 2743 MHz) and 0.6% in work a clock (77.01 against 77.53; peak 77.82).
- The memory controller reads 0% for all four miners on both cards, so on RTX 40 no DRAM traffic is left to save
  (RTX 20's lead).
- Our core alone (`hashrate.js`) reads what the CLI reads on Kryptex: 223.6 against 223.5 on the 4080 SUPER. Shares
  and job switches cost under 0.1% (`linedips` 0.12-0.15 s in 225 s).

**Raced in the miner's core** (nv34box.sh product stage, `hashrate.js` 120 s legs in turns at the 320 W cap):

| Change | RTX 4080 SUPER | RTX 4080 |
|---|---|---|
| shipped loop | 223.53-223.72, mean 223.63 (10 legs) | 209.66-209.97, mean 209.79 (6 legs) |
| `bxu=1 bimm=1` (s1, goal-nvk24's trims: 8 counter and address LOP3 fewer a chunk) | 223.77-223.96, mean 223.90 (+0.12%, 6 legs) | 210.09-210.39, mean 210.23 (+0.21%, 5 legs) |
| the RTX 30 loop on sm_89 (`r30`: no B exchange, verify-hits 298 of 298) | 223.68, 223.70, 223.70 (+0.03%; 2745 MHz at 320 W, more work a clock) | |
| `PEARL_ROT_TILES=128` | 223.54, 223.35, 223.39 (-0.1%) | |
| `PEARL_PROFILE_K=131072` | no result: hashrate.js exits at once | |

- s1 is ahead in every pair, on both cards, by more than twice the spread of the legs (s.d. 0.05-0.15 TH/s). It is
  correct: verify-hits 300 of 300 on the 4080 SUPER and 400 of 400 (560 hits, 29 salts) on the 4080. It ships
  here: `build_rot.py`'s sm_89 keys add `bxu=1 bimm=1`. The CUDA 12.8 cubin is 93d56520..., and it is in
  `pearl_rot_sm89.ref` and `kRotCubins`. The old lines stay, because the offsets records are unchanged.
- s1 is worth 0.1-0.2%. It closes about a tenth of the gap.
- The lead from RDNA 3 (B' straight from memory into registers) was built and run correctly on RTX 40 by
  goal-nvk27 (`bs3z`). It lost 19%: the L2 bytes cost more energy than the draw they replace (about 32 pJ a byte,
  goal-nvk28), and the loads took the pipe to 82%. B' kept in shared memory and filled from the L2 lost too (nvk28).
  The 96 x 512 block, which is more reuse for each shared read of B, was level at best (nvk24). These were not run
  again.
- The fold's loop is 529 instructions (about 8.5 KB) from its label to its branch. It fits the L0 instruction cache,
  so instruction fetch was not priced.

Tools: in this session's scratchpad, `goals/rtx40c` (vast.py, run.sh, run2.sh, runc.sh, run3.sh, pw.sh, pwsplit.py).

**Power every 100 ms, ours (s1, ef9dfcc5) against SRBMiner 3.8.9** (Vast RTX 4080, machine 152267, 340 W limit,
driver 580.178, 19 November; bench.sh on Kryptex, ours, srb, ours, 5 minutes each; `nvidia-smi -lms 100`, the
first 60 s of each leg left out):

| Leg | TH/s | Instant W: mean, s.d., p1-p99 | SM clock | SW power cap set |
|---|---|---|---|---|
| ours | 211.45 | 323.7, 4.7, 288-326 | 2760 MHz in every sample | 100% |
| SRBMiner | 211.38-211.51 | 306.9, 3.2, 301-313 | 2760 MHz in every sample | 0% |
| ours | 211.31 | 325.3, 4.7, 289-329 | 2760 MHz in every sample | 99% |

- The 5090's swing doesn't show here. No sample is over the limit, and the s.d. of 1 s means is the same (3.0 W
  ours, 2.9 W SRBMiner). Ours has a few short dips to 271-289 W that SRBMiner doesn't (salt or share events);
  they cost no clock.
- With 340 W both reach the card's top clock, so the hashrate is level (-0.05%). But ours draws 17-18 W (5.5%)
  more for the same clock and the same work. On a 320 W card that is the clock we lose (2726 against 2749 MHz
  above). The gap on RTX 40 is energy per MMA, not swings.
- RmProfilingAdminOnly is 1 on this host, so no ncu bank-conflict count.

**Where the extra watts go: timing-only cubins, each leaving one cost out** (the same machine, 19 November;
nv34box.sh product stage at c382e604, `hashrate.js` 120 s legs in turns with ship, which is s1; power every
100 ms, each leg's first 20 s left out; ab.sh and abs.py). Every leg held 2760 MHz, so watts measure energy per
unit of work. The eight ship legs read 324.3-325.9 W (mean 325.1) at 211.14-211.64 TH/s.

| Left out (rfold.py key) | TH/s | W | Against the ship legs either side | SW power cap set |
|---|---|---|---|---|
| the readout's XORs (`nord=1`, 96 LOP3 a chunk) | 212.12 | 310.8 | -14.3 W, -4.7% W per TH/s | 0% |
| B's last gather pair (`nbr=1`) | 211.72 | 315.4 | -9.7 W, -3.1% | 0% |
| A's last 3 gathers a chunk, loaded from the stage (`ah=3`) | 211.15 | 315.3 | -9.5 W, -2.7% | 0% |
| the gathers' PRMT (`noprm=1`) | 210.25 | 315.7 | -9.6 W, -2.3% | 0% |
| the readout's sum reads (`rdx=1`) | 211.87 | 317.5 | -7.9 W, -2.6% | 42% |
| the transcript word (`ntr=1`) | 211.58 | 324.3 | -0.7 W (the same cubin as ship here) | 100% |

- SRBMiner drew 306.9 W at 211.4 TH/s, 18 W under ship. The readout is the biggest single part, and no one
  part closes the gap. The readout, the B and A gathers and the regroup each cost 8-14 W.
- These do not add up. The card is on its SW power cap at 325 W even under the 340 W limit, and every variant
  under about 317 W drops it, so the first 8-10 W any of them saves may be the cap's own margin (voltage).
- `nlg=1` (three of four offsets loads from one record) failed rfold.py's assert in this configuration, so that
  leg ran without the fold. There is no separate hash kernel a batch (the fold tests its transcripts itself). The
  per-salt prep and `ROT_BKEEP` (written for the 96 x 512 block only) were not priced.

##### rig01, 15 November: the readout, the PRMT and the transcript word have no cheaper correct form; small trims priced (branch `goal/nv-k29`)

Nothing ships; the RTX 30 and RTX 40 cubins are unchanged (c89 rebuilds to 810df5cd... from `build_rot.py`'s keys).
rig01's RTX 4090, `gprobe` k 65536, 25 s runs in turn, the shipped RTX 40 loop `c89` as control. TIMING ONLY unless
a row says it passed the CPU check.

**Why no cheaper correct form exists (counted from the loop: a chunk is 192 IMMA and 328 others).**

- Readout, 96 XOR3 a chunk. The protocol XORs every entry of a region into the region's word at every chunk. A
  thread's region is its 192 sums, so a chunk has 191 XOR inputs, and a three-input step takes two: 96 is the
  floor. Every computed entry has to be in some region (else its MACs earn nothing), so no region shape lowers the
  count. RDNA 4's two-lane regions won there because a region spanned lanes and paid cross-lane steps; here a
  region is one thread's, and a two-lane region would add a shuffle and a select a chunk.
- PRMT, one per fragment register. sm_89's ldmatrix moves 16-bit elements, and a table row is one rank index, so a
  transposed load gives a lane two columns at two k. The IMMA wants one column at four k in a register, for A and B
  alike (any k order works if both agree, but the two columns still have to be split). That is a byte interleave
  of two registers. A 16-bit element of one column at two k is the table by pair (127x the size); masks need a
  byte-swapped second table (no room) and still cost a step a register.
- Transcript word, one LDL and one STL a chunk. The loop writes 252 registers. Shared memory holds tables 57,344 +
  exchange 16,384 + stage 24,576 bytes, leaving about 3 KB; the transcripts need 16 KB. Word pairs need one more
  register and lost on the 5090 (any added step costs more than half the load/store saves). k 2048 would make each
  word a plain store, but the tile's overhead (0.12% of the pipe at k 65536) would grow 32x.

**What the card said** (pipe; watts at a locked 2010 MHz; clock at 250 W and 450 W; gain is pipe x clock against
the mean of the c89 runs):

| Build | Check | 2010 MHz: pipe, W | 250 W | 450 W |
|---|---|---|---|---|
| c89, six / four / two runs | 512/512 | 99.15-99.17%, 247-254 W | 1957-1980 MHz, -0.68 to +0.49% | 2614-2631 MHz, -0.32 to +0.32% |
| s1: `bxu=1 bimm=1` (8 counter/address LOP3 fewer a chunk) | 16,384/16,384 at k 2048, orders 6 and 0 | 99.26%, 249-254 W | +0.20, -0.51% | +0.05, +0.05% |
| s2: s1 + `trl=2` | 16,384/16,384 (orders 6, 0) | 99.38%, 249-254 W | +0.49, +0.39% | +0.08, +0.08% |
| t2: `trl=2` alone | **16,224/16,384 at order 6** (one tile wrong), 16,384 at order 0 | 99.26%, 254-255 W | +0.13, -0.14% | -0.16, -0.08% |
| no readout (`nord`) | wrong | 99.44%, 243-244 W | +5.6, +8.0% | |
| readout reads no sums (`rdx`) / reads one sum a step (`rdx=2`) | wrong | 99.26%, 246-249 W / 247-253 W | | |
| no PRMT (`noprm`) | wrong | 98.68%, 246-248 W | +3.8, +1.3% | |
| no transcript LDL/STL (`dbg=3`) | wrong | 99.26%, 252 W | +1.28, +1.28% | |
| c89 with B constant (gprobe fake 128, the control) | wrong | 99.16%, 201 W | | |
| the RTX 30 loop on this card (`r30`) | 512/512 | 99.60-99.61%, 253-258 W | | |

- The prices stand as earlier rounds found them: the readout is 8 W at 2010 MHz and 6-8% at 250 W, the PRMT 6 W
  and 1-4%, the transcript's two memory steps 1.3% at 250 W. None has a correct form with fewer steps.
- The trims that are correct move nothing outside the twin's spread: s1 is +0.05% at 450 W and inside c89's 1.2%
  spread at 250 W. `trl=2`, reported correct by goal-nvk25, failed one tile of 64 here once in six checks: a race,
  cause not found. Do not ship `trl` on the RTX 40 loop.
- No Salad race was run: nothing measured here could show on a Salad card.

Tools: `S/nvk29` (mk.sh, go.sh, h.sh, l1.txt, out1.txt, osum.py).

##### rig01, 15 November: B' kept in shared memory and filled from the L2 loses; what an L2 byte costs (branch `goal/nv-k28`)

Nothing ships; the RTX 30 and RTX 40 cubins are unchanged (the RTX 40 loop still rebuilds to 6e5f2350...). rig01's
RTX 4090, watts at a locked 2010 MHz unless named. TIMING ONLY except the shipped loop `c89`.

**The idea.** goal-nvk27 found B with no draw worth 22-25 W at 2010 MHz, and B' read from global memory every tile
too dear. So: keep each 64-column slice of B' in shared memory and let several row bands of a block read it, so
each B' byte crosses the L2 once every few row bands.

**On paper it has little room.** A band's sums run over all of k, so a slice can only be shared by bands that run
at the same time: Rb bands x Sb slots = 8 warps (one block an SM; 256 threads x 255 registers fill the register
file). Steps a warp a k-step (shared-memory / ALU), and L2 bytes an SM a k-step:

| Shape | A side | B side | L2 | Shared memory |
|---|---|---|---|---|
| c89 (B drawn once a slot, `bx`) | 9 / 12 | 8 / 16 | 0 | 96 KB |
| B' resident, Rb 2 x Sb 4 (today's shape) | 9 / 12 | 4 LDSM + 2 copies / 0 | 8 KB | fits |
| Rb 4 x Sb 2 (A shared by 2 warps, not 4) | 12 / 24 | 4 + 1 / 0 | 4 KB | 90-105 KB |
| Rb 8 x Sb 1 (A not shared) | 12 / 48 | 4 + 0.5 / 0 | 2 KB | A's tables alone 96 KB: does not fit |

Only Rb 2 saves anything on the chip; taller blocks lose A's sharing, which costs what B's draw saves.

**What an L2 byte costs (`probes/hand89/tring.cu`, new).** Plain CUDA: 48 mma a warp a k-step with A fixed in
registers and B's 16 words loaded with four plain ldmatrix a k-step; all runs at 99.1-99.9% of the pipe.

| B from | random B (W) | constant B (W) |
|---|---|---|
| a shared-memory table, a new 2 KB window every k-step | 187-188 | 140 |
| the same, plus cp.async copies to a junk area: 4 KB an SM a k-step | 215 | |
| 8 KB | 231-232 | 162-164 |
| 16 KB | 276-277 | |
| a ring that cp.async fills, 8 KB an SM a k-step, B read from it | 230-233 | 162-164 |
| the same, 16 KB | 274-275 | 182-185 |

So 8 KB an SM a k-step from the L2 costs 44 W with random bytes (22 W with constant bytes), about 32 pJ a byte at
1.37 TB/s; 4 KB costs 27 W. Reading B from shared memory itself costs almost nothing (the table rows draw what
goal-nvk26's tbench drew with B fixed in registers, 186 W). Rb 2's 8 KB (44 W) is twice what B's draw costs in c89
(about 20 W: `bpl=1`, B as plain LDSM of table bytes spread as B', drew 231 W against c89's 248-252). Rb 4's 4 KB
(27 W) and Rb 8's 2 KB (about 14 W, and no fit) have no on-chip saving to set against it. The family loses.

**The hand-loop skeleton misled again.** `rfold.py bstr=1` with new keys `bstn` (copies a lane a turn) and `bsu`
(every warp its own bytes) prices the ring in the shipped loop. At 250 W it read +11% (8 KB) to +17% (4 KB), but at
2010 MHz it drew 192-194 W whatever B' held: iid random (`gprobe -DGP_BIGB=1`), the real B' stream (`GP_BIGB=3`)
or all 63 (`GP_BIGB=2`), against the plain C test's 230 W for the same traffic. Its sums do change with B' (the
transcripts differ), but its operands toggle like constant B, for a reason not found. The control caught it, as the
brief asked; no gain from that loop counts. Its pipe: 96.3% (4 KB), 92.0% (8 KB), 77.9% (16 KB); at 450 W the 8 KB
form was -4.2% against c89 even with the cheap operands. `gprobe` fake 64 / 128 (new): B's tables only spread as B' /
constant, for `bpl=1`.

Also: goal-nvk27's correct streamed fold (bs3z) drew 250-255 W at 81.5-81.7% pipe (256 W last round).

##### rig01, 15 November: two folds that do not draw B, priced and checked; both lose (branch `goal/nv-k27`)

Nothing ships; the RTX 30 and RTX 40 cubins are unchanged. rig01's RTX 4090, `gprobe` 30 s runs at k 65536 with
the shipped RTX 40 loop `c89` as the control (its own spread: -1.3 to +2.0% at 250 W, +-0.2% at 450 W). Gain is
pipe x clock against c89. TIMING ONLY unless a row says it passed the CPU check.

**B grouped by its rank index (goal-nvk26's proposal).** A chunk's sums are `sum_p G[r][p] (U_B[c][p] + 32)` with
`G[r][p] = sum(k: p0B = p) A'[r][k] - sum(k: p1B = p) A'[r][k]` and A's fill 0. Checked on the CPU with the
protocol's own noise generators (`S/nvk27/alg.js`): the transcript matches the reference for 6 salts and three
fill choices. But G leaves int8 in 0.31-0.36% of entries (largest |G| 219; never with 2 or fewer k on a rank
index in a chunk, 0.25% at 3, 1.3% at 4), and that cannot be designed away: a chunk has 512 dense-factor terms of
up to 32 over 128 slots, the whole int8 range on average. Splitting the heavy slots needs about 159 slots (+24%
IMMA). Its timing skeleton, before any overflow fix:

| Build (rfold.py keys) | 250 W | 450 W |
|---|---|---|
| B as plain loads of a fixed 6-bit table, no draw, A as now (`bpl=1`) | 2208-2228 MHz, +10.4 to +12.0% | 2685 MHz (top), 427 W, +2.6% |
| the same plus A's G draw, 3x / 4x A's table loads, spread over the chunk (`arep asp`) | +1.1..2.6% / -4.7..-5.0% | +0.5% / -1.4% |
| A's G draw packed into the first turn | 93.3% / 90.5% pipe: +0.2..0.8% / -2.4..-3.3% | -3.8% / -7.0% |

So A's side costs about what B's draw saves, with no room left for the overflow.

**B' made once a job and read from global memory** (B' is the protocol's noised B, fixed for a job; the next
design those numbers point to). `gprobe -DGP_BIGB=3` builds the real B' on the host in the loop's layout and hands
the tile's stream to the loop in shared memory; rfold.py `bstr=3 nob=1 bsb=1 bpl0=1` reads each warp's 2 KB a
k-step straight into B's registers (four LDG.E.128 a lane, the next turn's lines touched ahead). It is correct:
16,384 of 16,384 lane transcripts at k 2048 and 8192 (searched and definition order), 512 of 512 at k 65536.

| Build | 250 W | 450 W | locked 2010 MHz |
|---|---|---|---|
| c89 | 1959-2039 MHz, 99.15% pipe | 2614-2631 MHz, 450 W | 247-254 W |
| B as plain table loads (`bpl=1`, not a fold) | +10.4 to +12.0% | +2.6% | 225-226 W |
| B' streamed, correct (`bs3z`) | 1942-1975 MHz at 82% pipe: -18.7 to -19.9% | 2648 MHz, 446 W, 79.8% pipe: -18.9% | 256 W at 81.7% pipe |

The correct fold spends about 25% more energy a unit of work than c89: what the L2 delivers costs more than the
draw it replaces (B's draw is worth 22 W at 2010 MHz, by `bpl=1`). Its pipe is lost waiting on the loads.

**A skeleton that misled, and the control that caught it.** The earlier streamed skeletons (`bstr=2`, B' as random
bytes behind the offsets, `gprobe -DGP_BIGB=1`, never checked) showed +9 to +14% at 250 W (2350-2380 MHz at 92-94%
pipe). At a locked 2010 MHz the same build drew 191 W with random B' and 185 W with B' all 63 (`-DGP_BIGB=2`);
random against constant B is about 45 W on this card. So its B operands hardly toggled, and the gain was a
constant-operand effect. On the real stream it still drew 200 W while the correct fold drew 256 W. A timing
skeleton that feeds the tensor unit needs a constant-operand control at a locked clock before its gain counts.

Also measured: the definition's order of k costs the streamed loop 0.4% against the searched one; c89 at k 8192 runs
98.55% pipe (k 65536: 99.15%).

Tools: rfold.py keys `bpl arep asp bstr bsg0 bpe bpf bsb nob bpl0`; `gprobe -DGP_BIGB=1|2|3`; asm89 assembles the
untransposed LDSM; `S/nvk27` (alg.js, mk.sh, go.sh, h9.sh, osum.py, the hold lists). The miner side of the
streamed fold (pearl_rot.cu `ROT_BS` with `rot_aux_bprime`, the host's B' at k 8192) is on `goal/nv-k27` only,
never run in the miner.

#### RTX 50-series (14 November)

12 of 12 rows rerun. Above 100%: 2 of 12. Median 99.4%.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | OneZeroMiner | % of best | Run |
|---|---|---|---|---|---|---|---|---|
| RTX 5090 | Salad (575 W) | 2b31ad9b | 438.0 | 422.5 | 446.9 | 435.6 | 98.0% | new |
| RTX 5090 | Salad (575 W) | a409f43c | 342.0 | 341.1 | 337.9 | 345.3 | 99.1% | new |
| RTX 5080 | Salad (300 W) | 57d1c912 | 251.9 | 250.7 | 251.7 | 254.3 | 99.1% | new |
| RTX 5080 | Salad (360 W) | 60b3d702 | 235.0 | 233.9 | 235.2 | 236.8 | 99.3% | new |
| RTX 5070 Ti | Salad (300 W) | d4084405 | 199.3 | 199.4 | 198.8 | 199.7 | 99.8% | new |
| RTX 5070 Ti | Salad (300 W) | 26c6a233 | 193.9 | 193.1 | 195.0 | 195.7 | 99.1% | new |
| RTX 5070 | Salad (250 W) | a1047664 | 133.0 | 132.0 | 132.7 | 131.9 | 100.2% | new |
| RTX 5070 | Salad (250 W) | 1306abc3 | 137.1 | 137.0 | 136.5 | 137.3 | 99.9% | new |
| RTX 5060 Ti | Salad (180 W, 16 GB) | d8da36ad | 103.4 | 103.0 | 103.1 | 103.8 | 99.6% | new |
| RTX 5060 Ti | Salad (180 W, 16 GB) | 81cce3b7 | 100.6 | 100.5 | 100.3 | 99.2 | 100.1% | new |
| RTX 5060 | Salad (145 W) | e4f17758 | 83.6 | 83.5 | 83.6 | 82.5 | 100.0% | new |
| RTX 5060 | Salad (145 W) | 6f073405 | 80.4 | 81.8 | 81.2 | 83.0 | 96.9% | new |

## RTX 20-series

Vast, 10–11 October.

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

The gap is work per clock: our clock is within 30 MHz of theirs, and we do 5–7.5% less per clock.

### Salad, 28 October (PeakMiner 2.20.3, SRBMiner 3.8.0)

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

The rows over 100% are odd PCs: one swung between 207 and 266 W, one is capped at 129 W.

### Salad, 30 October (SRBMiner 3.8.2)

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

### Salad, 9–10 November (v0.6.13; PeakMiner 2.20.7, SRBMiner 3.8.2, OneZeroMiner 1.8.5)

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | OneZeroMiner | % of best |
|---|---|---|---|---|---|---|---|
| RTX 2080 Ti | Salad (300 W) | 1ff934b7 | 105.4 | 105.5 | 104.2 | 105.5 | 100% |
| RTX 2080 Ti | Salad (260 W) | 7e24f884 | 97.4 | 97.9 | 94.8 | 94.6 | 99% |
| RTX 2080 | Salad (260 W) | a5d2c787 | 79.0 | 80.9 | 76.6 | 76.0 | 98% |
| RTX 2070 Super | Salad (215 W) | ef050aaa | 64.7 | 65.4 | 64.8 | 62.2 | 99% |
| RTX 2070 Super | Salad (215 W) | 842bcd6e | 63.9 | 65.3 | 65.2 | 61.5 | 98% |
| RTX 2070 | Salad (140 W) | b6b2b0c9 | 51.1 | 52.5 | 51.4 | 50.3 | 97% |
| RTX 2070 | Salad (175 W) | f4e53154 | 53.5 | 55.1 | 53.5 | 52.0 | 97% |
| RTX 2060 | Salad (160 W, 6 GB) | 67fbcdbf | 40.6 | 42.0 | 40.3 | 39.9 | 97% |
| RTX 2060 | Salad (160 W, 6 GB) | 92c0cd67 | 36.7 | 37.4 | 36.7 | 37.4 | 98% |

PeakMiner is fastest on every PC; OneZeroMiner is the slowest of the three here.

Read again on 10 November from the boxes' own sample lines, which Salad's log service still held (the result files
kept two PCs'), with the 94de741e and dbbceed1 PCs that the table above leaves out. Each miner is the mean of its
samples from the first minute on; the tensor peak is the card's SMs × 1024 MAC a clock.

| Card | Machine ID | Ours TH/s, MHz, W | PeakMiner TH/s, MHz, W | Ours, of the tensor peak | PeakMiner | Memory controller, ours and PeakMiner |
|---|---|---|---|---|---|---|
| RTX 2080 Ti | 1ff934b7 | 105.4, 1645, 292 | 105.8, 1649, 292 | 92.0% | 92.2% | 11% and 25% |
| RTX 2080 Ti | 7e24f884 | 97.4, 1520, 254 | 97.9, 1524, 254 | 92.0% | 92.3% | 11% and 25% |
| RTX 2080 | a5d2c787 | 79.0, 1812, 258 | 80.7, 1806, 255 | 92.5% | 94.8% | 19% and 31% |
| RTX 2080 | c6b154db | 125.5, 1402, 166 | 128.8, 1406, 165 | 190% (two cards) | 195% | 15% and 23% |
| RTX 2070 Super | ef050aaa | 64.7, 1707, 192 | 65.4, 1716, 188 | 92.5% | 93.0% | 15% and 21% |
| RTX 2070 Super | 842bcd6e | 63.9, 1703, 197 | 64.8, 1706, 195 | 91.6% | 92.8% | 16% and 21% |
| RTX 2070 | b6b2b0c9 | 51.1, 1546, 137 | 52.3, 1546, 137 | 89.7% | 91.8% | 11% and 18% |
| RTX 2070 | 94de741e | 52.5, 1543, 112 | 53.4, 1541, 112 | 92.3% | 93.9% | 15% and 24% |
| RTX 2060 | 67fbcdbf | 40.6, 1459, 125 | 42.0, 1430, 127 | 90.7% | 95.5% | 19% and 29% |
| RTX 2060 | 92c0cd67 | 36.7, 1360, 100 | 37.6, 1346, 98 | 87.8% | 90.8% | 19% and 24% |
| RTX 2070 Super | dbbceed1 | 65.9, 1759, 211 | 68.1, 1758, 212 | 91.4% | 94.6% | 16% and 25% |
| RTX 2070 | f4e53154 | 53.5, 1578, 169 | 55.2, 1572, 172 | 91.9% | 95.2% | 13% and 19% |

- Per clock we do 99.8% of PeakMiner's work on the two 2080 Tis and 97.4% (±0.45; 94.9–99.4%) on the nine other
  single-card PCs. Its memory controller is busier than ours on every PC, and its VRAM use is ours plus 1,166 MiB on
  every PC.
- **`c6b154db` is not one RTX 2080.** OneZeroMiner's log on it reads `GPU 1 Share accepted`, and 125.5 TH/s is 190% of
  what GPU 0's 46 SMs can do at the 1,402 MHz it ran at. The PC has at least two cards; every miner mined them all,
  and `bench.sh` read the first card's power and clock. Its "% of best" stands, its TH/s and watts are not one card's.
  So on 11 November its row (ours 125.5 against 128.8, 126.6 and 121.8, 97%) was taken out of the 9–10 November
  table above, which holds single cards, and out of the count of rows: 53 NVIDIA rows, not 54.

### Salad, 12–13 November: the official table until 14 November (PR #293 build; PeakMiner 2.20.7, SRBMiner 3.8.7, OneZeroMiner 1.8.5)

Replaced by the 14 November table at the top of this file.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | OneZeroMiner | % of best |
|---|---|---|---|---|---|---|---|
| RTX 2080 Ti | Salad (260 W) | 7e24f884 | 99.2 | 97.8 | 96.5 | 94.8 | 101.4% |
| RTX 2080 Ti | Salad (260 W) | 60bcfc7b | 100.7 | 95.8 | 99.3 | 93.3 | 101.4% |
| RTX 2080 | Salad (215 W) | 4cd16f9f | 66.6 | 66.9 | 65.7 | 63.4 | 99.5% |
| RTX 2080 | Salad (215 W) | 27083522 | 61.2 | 64.5 | 62.2 | 59.8 | 94.9% |
| RTX 2070 Super | Salad (215 W) | 31ebd1e6 | 67.7 | 68.5 | 68.2 | 64.2 | 98.9% |
| RTX 2070 Super | Salad (215 W) | 6f70af1f | 67.9 | 69.7 | 68.2 | 64.8 | 97.5% |
| RTX 2070 | Salad (175 W) | c4614cb9 | 45.2 | 45.6 | 45.0 | 45.8 | 98.6% |
| RTX 2070 | Salad (175 W) | d2064b2f | 52.6 | 53.5 | 52.3 | 50.3 | 98.4% |
| RTX 2060 | Salad (184 W, 6 GB) | 75bea93c | 52.4 | 53.9 | 51.8 | 50.9 | 97.2% |

### Salad, 10 November: what PeakMiner changed on RTX 20 at 2.20.1, and our fold in its shape (PR #290 goal work)

**The capture.** One RTX 2070 Super PC (`588f9e55`, 215 W, driver 616.92, 1.77 GB of VRAM held by its desktop). Each
miner mined on the stand-in pool (`verify-pool.js serve`, target 2^211, a job every 44 s as Kryptex sends them)
through `sharetap.py`, so every share was checked and decoded. Two passes, forward then reverse, 4 minutes a run with
the first minute left out. Every figure is the mean of the miner's own samples: its rate every 5 s, the card every
2 s. The two passes agreed to 0.2% for every miner. The scripts are in the session's scratchpad (`g20ctl/g20/`), not
committed.

| Miner | m | n | k | Tile | Fill | Memory controller | VRAM over idle | SM MHz | W | °C | TH/s | TH/s per GHz | Of the tensor peak |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PeakMiner 2.20.0 | 16384 | 65536 | 8192 | 8×16 | 64 | 35% | 1,193 MiB | 1749 | 213 | 73 | 65.01 | 37.17 | 90.7% |
| PeakMiner 2.20.1 | 1024 | 245760 | 2048 | 8×16 | 64 | 22% | 2,153 MiB | 1750 | 212 | 73 | 67.80 | 38.75 | 94.6% |
| PeakMiner 2.20.7 | 1024 | 245760 | 2048 | 8×16 | 64 | 23% | 2,153 MiB | 1748 | 212 | 73 | 68.19 | 39.01 | 95.2% |
| Ours v0.6.15 | 131072 | 262144 | 2048 | 16×16 | 48 | 16% | 987 MiB | 1749 | 213 | 73 | 66.07 | 37.78 | 92.2% |

The tensor peak is 40 SMs × 1024 MAC a clock. With `--gpu-lmem 810` (the lock fails on Salad) 2.20.7 ran the same:
68.23 TH/s, the same shape, memory controller 24%.

- **What 2.20.1 changed is in its shares: the shape.** A went from 16384 rows to 1024, k from 8192 to 2048, and B from
  65536 columns to 245760. A' was 134 MB and is now 2 MB, which fits the card's 4 MB L2; B' is 503 MB. That came with
  +4.3% work a clock on this card at the same clock and power, the memory controller 35% busy before and 22% after.
  2.20.7 mines 2.20.1's shape and shows 0.6% more. Before 2.20.1 we were 1.6% ahead of it a clock; now we are 3.2%
  behind.
- Every release fills A and B with 64 and mines rank 128. It re-draws only A between salts, as we do: a share's root
  of B is the same all job and its root of A is new almost every share (80–95 roots in a job's ~90 shares; ours 32–39
  in ~77, a salt every 1.06 s). Its stamp is in A's first bytes, written as 63s and 64s.
- Its tile is one thread's accumulators, 8 rows × 16 columns (128 sums); ours is 16×16 over four lanes. 2.20.1 moved
  the tile's rows from every 16th to pairs 8 apart every 32. It hashes twice as often a MAC as we do.
- **Ours is 96.9% of 2.20.7 here, all of it work a clock**: 1749 and 1748 MHz, 213 and 212 W, both 73 °C.
- Shares against the shown rate, over all runs: the pool credited PeakMiner 1.01 ±0.02 of what it showed (2,369
  shares in seven runs) and ours 0.97 ±0.04 (655 in two). One PeakMiner run had a 12 s gap in its shares, where every
  other run's longest was 4 s: its dev fee, most likely. Without that run it read 1.03. So nothing here says its shown
  rate is more than a user gets, and 3 minutes a run is too short to say more.

**Our fold in PeakMiner's shape, same PC.** Race mode with an A/A twin, 17 to 20 rounds a race. A copy of `race.cu`
was given per-build m, n, batch width and launch-sweep settings for this (in the scratchpad; the repo's file is
unchanged), so one build of the release's core ran every arm but the row bands. Rates are MACs a second with re-draws
counted.

| Arm | Against the release shape | Twin | SM MHz | Memory controller |
|---|---|---|---|---|
| Release shape (m 131072, n 262144, column bands of 5) | 66.3 TH/s | | 1750 | 16% |
| m 1024, n 262144, band 5 | −3.01% ±0.05 | +0.01% ±0.03 | 1728 | 34% |
| m 1024, n 131072, band 5 | −4.64% ±0.05 | | 1743 | 33% |
| m 1024, n 524288, band 5 | −2.30% ±0.08 | | 1720 | 36% |
| m 2048, n 262144, band 5 | −1.51% ±0.04 | 0.00% ±0.03 | 1740 | 27% |
| m 512, n 262144, band 5 | −5.21% ±0.06 | | 1721 | 43% |
| m 1024, n 262144, band 5, batches of 2048 offsets (1 ms) | −6.70% ±0.10 | | 1746 | 38% |
| m 1024, n 262144, band 1 | −1.91% ±0.04 | −0.05% ±0.07 | 1748 | 23% |
| m 1024, n 262144, band 2 | −3.06% ±0.09 | | 1725 | 34% |
| Our m and n, row bands of 8 (`-DPEARL_BD_WALK=0`) | −0.33% ±0.13 | | 1744 | 20% |
| Our m and n, row bands of 6 | −1.10% ±0.05 | | 1725 | 26% |
| Our m and n, row bands of 4 | −1.71% ±0.02 | | 1720 | 38% |

- **PeakMiner's shape does not help our fold.** At m 1024 with a band of 1 our memory controller reads 23%, the same
  as PeakMiner's, so this is very likely how it walks (inferred): A' resident, and each 256-column panel of B' read
  from DRAM once by the eight blocks that share it. Our fold does the same work a clock that way as with B' resident:
  row bands of 8 at our own m read −0.33%, all of it clock (0.3% lower, from the extra DRAM traffic).
- m 1024 itself costs us 1.6% more. A salt lasts 8.6 ms there, and our re-draw between salts is queued kernels that
  run with the fold stopped, about 0.14 ms. PeakMiner re-draws as often and shows no such loss.
- So the shape is not what puts PeakMiner ahead of us. On this card it was behind us a clock at 2.20.0; with 2.20.1
  it reached a walk as good as ours, and what it has beyond that is inside its loop, which its shares don't show.

**Where our fold's last points go, same PC.** Timing-only builds (wrong results on purpose, never shipped) against the
release build, 9 and 11 rounds, twins −0.02% ±0.04:

| Switched off | TH/s | SM MHz | Per clock |
|---|---|---|---|
| The readout (`PEARL_ABLATE_READOUT`) | +1.96% ±0.10 | 1764 | +1.5% |
| The readout and the hash | +2.37% ±0.11 | 1764 | +1.9% |
| The readout's shuffles only | −2.88% ±0.05 | 1751 | −2.6% |
| The hash only | −4.05% ±0.08 | 1741 | −3.3% |
| B loaded once a tile (`PEARL_ABLATE_BLOAD`) | +4.12% ±0.05 | 1822 | +0.2% |
| Every B load an L1 hit (`PEARL_ABLATE_BHIT`) | +3.18% ±0.05 | 1807 | +0.1% |
| Readout, hash and B loads | +9.06% ±0.05 | 1830 | +4.5% |

- Bringing B' from the L2 into the SMs costs this card about 4% of its clock at the power limit and almost no work a
  clock. The readout and hash cost 1.9% of the work a clock. With all three free the fold would run at 96.6% of the
  tensor peak and 1830 MHz, 9% above today.
- PeakMiner is 3.2% ahead at our clock and our watts, so it spends less energy a MAC somewhere in those 9 points.
  Turning off the shuffles or the hash alone makes our fold slower: its warps then drift apart and stop sharing B' in
  L1 (memory controller 32% with the hash off), as October's rounds found.

**The review's two cheap races, same PC** (16 rounds, twin −0.01% ±0.03): operand fill 56 −0.02% ±0.03 and fill 63
−0.04% ±0.04 against the 48 that ships; band 4 −0.21% ±0.03 and band 6 −0.21% ±0.06 against the rule's 5. Nothing to
change.

**The same on an RTX 2060** (`59be6855`, 6 GB, 170 W, driver 617.42, 0.52 GB of VRAM held by its desktop), where
PeakMiner's lead in the table above is largest. PeakMiner 2.20.7 and ours twice each, forward then reverse, 200 s a
run with the first 50 s left out; 2.20.1 and 2.20.0 once, 100 s with the first 40 s left out. Target 2^213.

| Miner | m | n | k | Tile | Fill | Memory controller | VRAM over idle | SM MHz | W | °C | TH/s | TH/s per GHz | Of the tensor peak |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PeakMiner 2.20.0 | 16384 | 393216 | 8192 | 8×16 | 64 | 51% | 4,177 MiB | 1692 | 167 | 70 | 48.54 | 28.69 | 93.4% |
| PeakMiner 2.20.1 | 1024 | 245760 | 2048 | 8×16 | 64 | 30% | 2,135 MiB | 1696 | 163 | 69 | 49.71 | 29.31 | 95.4% |
| PeakMiner 2.20.7 | 1024 | 245760 | 2048 | 8×16 | 64 | 31% | 2,135 MiB | 1691 | 162 | 70 | 49.41 | 29.23 | 95.1% |
| Ours v0.6.15 | 131072 | 262144 | 2048 | 16×16 | 48 | 22% | 969 MiB | 1707 | 165 | 71 | 47.83 | 28.03 | 91.2% |

- Ours is 96.8% of 2.20.7 on this card: 95.9% of its work a clock, at a 0.9% higher clock and 3 W more.
- 2.20.1 made the same change here (m 1024, k 2048, the same tile), worth +2.2% a clock. But 2.20.0 was already 2.4%
  ahead of us a clock on this card with its old shape, both operands in DRAM (B' was 3.2 GB) and the memory
  controller 51% busy. So its lead does not come from the shape.
- Over both PCs the pool credited PeakMiner 0.99 ±0.02 of its shown rate (4,011 shares) and ours 0.98 ±0.02 (2,264).
- Our fold in PeakMiner's regime on this card's 3 MB L2 (race, 12 rounds, twin −0.03% ±0.07; the release shape ran
  48.19 TH/s at 1705 MHz with the memory controller 22% busy): row bands of 8 −1.00% ±0.05, row bands of 4 −1.09%
  ±0.14, m 1024 with a band of 1 −1.78% ±0.17, each at about 1690 MHz with the memory controller 37–40% busy
  (PeakMiner: 31%). It costs us more here than on the 2070 Super.

**What this leaves.** Nothing we can copy from PeakMiner's shares (m, n, k, the walk they imply, a higher fill) gains
on our fold; the band rule and fill 48 hold. Its tile, one thread a region, is the one visible difference not
tried. It would let every warp do the same work, where our fold has four warps hash and four not and loses up to 4%
when they drift apart. Whether that is where its 3% comes from is a guess; building it is a new readout and hash for
the Turing fold, and October's ROWPAIR build, which cut the readout's shuffles from six to one, read −0.4 to 0.0%.

### Salad, 12 November: the rotation fold priced on RTX 20, and where we stand (branch goal/nv-20)

The rotation fold that lifted RTX 40 and 30 by about 6% was priced on Turing before anything was built, with a
timing-only probe (`native/probes/turing/price75.cu`; tables in `native/probes/README.md`, "The rotation fold priced
on Turing"). Its operand stream alone, with no readout and no hash, mines about what the release fold mines with
both, so it is not built for sm_75:

| PC | Rotation stream, no readout or hash (by count) | Release fold, `hashrate.js` |
|---|---|---|
| RTX 2070 Super `d52b8b17` (235 W) | 84.9% of the pipe at 1799-1827 MHz: 62.6-63.5 TH/s | 64.24, 64.24 at 1767 MHz |
| RTX 2080 Ti `63284c7a` (220 W) | 87.0% at 1334-1342 MHz: 80.9-81.3 | 81.88-82.14 at 1293-1295 MHz |
| RTX 2060 `58719891` (6 GB) | 87.0% at 1451-1467 MHz: 38.8-39.2 | 39.02-39.06 at 1385-1389 MHz |

Turing pays for every shared-memory access and ALU step in tensor pipe (32 ALU steps a warp a k-step: -2.8 to -2.9
points; past about 12 shared accesses of 16 bytes a k-step the pipe falls away), where Ada paid mostly in clock. The
fold needs 14 shared accesses and 40 ALU steps a k-step on Turing's 64x64 warp tiles; the release needs 8 global
loads.

The other miners on the same PCs (`tuning/salad/nv20box.sh`: bench.sh on the stand-in pool at 2^210, 180 s after a
60 s warm-up, each miner's sample means from 60 s on; ours is the v0.6.15 release):

| PC | Ours | PeakMiner 2.20.7 | SRBMiner | OneZeroMiner 1.8.5 | Ours, % of best | Per clock, of PeakMiner |
|---|---|---|---|---|---|---|
| RTX 2070 Super `d52b8b17` | 64.61 (1767 MHz, 228 W) | 65.78 (1778, 229 W) | 64.51 (1758) | did not mine (its dev-pool connection failed) | 98.2% | 98.8% |
| RTX 2080 Ti `63284c7a` | 81.54 (1297 MHz, 212 W) | 82.24 (1313, 212 W) | 80.95 (1297) | 79.57 (1328) | 99.1% | 100.4% |
| RTX 2060 6 GB `58719891` (160 W limit, held near 110 W) | 38.54 (1413 MHz, 111 W) | 39.44 (1373, 107 W) | 38.84 (1400) | 37.82 (1435) | 97.7% | 94.9% (our CLI's samples here spread 32.6-45.7; `hashrate.js` read 39.03 at 1388 MHz, 97.9% of PeakMiner a clock) |

Launch-shape races on the release core, `hashrate.js` legs taking turns on one PC:

- 2080 Ti: the band rule (8) 81.88 / 82.04 / 82.04 / 82.14 TH/s; band 6 81.88 / 82.04 (level); band 10 79.35 / 79.56
  (-3.0%); a batch of 512 column offsets in place of 2048, 82.02 (level).
- 2060: the rule (band 4) 39.03 / 39.06 / 39.03; band 3 39.02 / 39.05 (level); a batch of 512, 39.02 / 39.03 (level).

What it leaves: on the 2080 Ti PeakMiner's lead is clock at the same watts (1.2%), not work a clock. On the 2070
Super it is 0.6% clock and 1.2% work a clock. The release fold's chunk loop is 512 IMMA and 149 other instructions a
chunk, 64 of them the readout's xors at their floor; by the probe's ALU price those and the hash explain most of what
it loses a clock. A batch of 512 column offsets costs no shown rate on any of the cards measured, which makes the review's
job-switch change (fewer hits dropped when a job changes) free to try; it was not measured on a pool.

### Salad, 12 November: L1 prefetches and the loop's load schedule on an RTX 2070 Super (branch goal/nv-20b)

One PC throughout: RTX 2070 SUPER, driver 621.14, power limit 172 W (215 W default), held at it in every leg
(166-169 W, 72-74 °C). The release core and each build from the same source with switches, `hashrate.js` legs of 90 s
taking turns, an A/A twin (the release built again) in each race. The card drifted down about 2% over a race, so each
leg is compared with the control legs beside it ("paired"). Every build passed `verify-hits.js` with no failures (400
of 400 in 90 s; the 60 s runs verified all 357-375 of their hits).

| Build | What it changes | Paired raw | Per clock | SM MHz against the release |
|---|---|---|---|---|
| A/A twin | nothing | −0.08% (−0.42 to +0.16), +0.02% | −0.1%, +0.3% | level |
| `PEARL_BD_PF=3` | L1 prefetches two k-steps ahead, shared out among the warps that read the same A and B lines | −3.45% | −2.2% | −20 |
| `PEARL_BD_PF=2` | B only | −1.23% (−1.32, −1.13) | −0.7% | −10 |
| `PEARL_BD_PF=1` | A only | −1.46% (−1.47, −1.46) | −0.6% | −14 |
| `PEARL_BD_P64=3` | 64-bit pointers a pass, so the 32 loads take immediates (8 fewer IADD3 a chunk, 653 instructions) | −0.51% (−0.78 to +0.29), 5 legs | −1.0% | +8 |
| `PEARL_BD_P64=1` | A only | −1.08% | −1.1% | level |
| `PEARL_BD_P64=2` | B only | −2.60% | −3.3% | +10 |
| `PEARL_BD_TADDR=1` | round 6's transcript index in a register (649 instructions) | −0.41% (−0.41, −0.40) | −0.5% | level |
| `-Xptxas --register-usage-level=8` | another ptxas schedule: 647 instructions, 251 registers | +0.31% (−0.06 to +1.08), 4 legs; level in the card's steady half (60.87-60.96 against 60.90-60.91) | +0.2 to +0.6% | −3 |
| `-Xptxas --register-usage-level=3` | another schedule, 661 instructions | −0.27% (−1.13 to +0.11); level in the steady half | −0.3% | level |

The last race's twin read −0.58% (−1.17 to +0.01): the card stepped up 2% between two legs, at the same clock, and
the pairs across that step carry it. In the steady half all four builds read 60.87-60.96 TH/s.

- **The warps' shared lines already reach L1 once.** A row group's four warps load the same A lines and a scheduler's
  two warps the same B lines. If their loads missed L1 together, one warp's prefetch would save the others the trip
  from L2 and the energy that costs. Instead every prefetch build is slower, at a lower clock and less work a clock:
  each prefetch (`CCTL.E.PF1`) costs about what a load does, far more than the ALU price of an instruction.
- **The loop's speed follows where ptxas puts its loads, not the instruction count.** Cutting 8 or 12 instructions a
  chunk lost every time. P64 moved the loads (ptxas spreads them out) and lost 1% a clock, though fewer instructions
  let the card run 0.5% faster at the same watts. A source that issues one load before each B pair's IMMA, instead of
  four at the top of each pass, compiles to the release's chunk loop instruction for instruction: ptxas places the
  loads itself, so the source cannot spread them.
- So no schedule measured on this card beats the release's; ptxas's level-8 schedule ties it. What is left on Turing
  needs the loads placed in SASS (a cubin patch or a hand schedule), not another source change.
- Box time: one group, `nv20b-2070-1`, about 2 h (machine `e7703412`). The switches are in commits c6fe3964, e2c50a5f
  and afd01fba and were taken out again (c1f1e88e): the Turing tests compare the fold with copies of its source.

### Salad, 12 November: B's loads kept in L1 (+0.9% on an RTX 2080 Ti), and a SASS load placement (branch goal/nv-20c)

Four PCs, at most two at once: an RTX 2080 Ti (driver 621.14, 250 W limit, held at 241-244 W, 76-78 °C,
1376-1392 MHz, machine `5a02bd58`), a plain RTX 2070 (621.14, 225 W, 220-224 W, 68-70 °C, 1830-1847 MHz), an RTX
2060 (581.42, 170 W limit, 150-152 W, 84-85 °C, 1594-1605 MHz, machine `dff31915`) and an RTX 2070 SUPER (616.92, 215 W, 211-213 W, 73-74 °C, 1747-1751 MHz). `nv20box.sh`: the core built on
the box from the branch, `hashrate.js` legs of 90 s taking turns, an A/A twin of the control in every race, each leg
compared with the control legs beside it ("paired"). Every build passed `verify-hits.js` 400 of 400 in 90 s.

**The cache operator of the chunk loop's loads** (`PEARL_BD_LD_B` / `PEARL_BD_LD_A` at the time; sm_75 encodes each
as different bits of the same LDG, and ptxas schedules the loop identically):

| Build | 2080 Ti | 2070 | 2060 |
|---|---|---|---|
| A/A twin | −0.06% (round 1), −0.69% (round 2) | −0.17% | +0.06% |
| B evict_last as shipped (050b284f) against two release builds, 3 rounds | **+0.58%** (89.40 against 88.88; the two release builds agree within 0.1%) | 2070 SUPER: **0.00%** (66.01 against 66.01 and 65.97) | |
| B `.nc.L1::evict_last` (`LDG.E.EL`) | **+0.92%** (+0.76 to +1.03), **+0.95%**; per clock +0.1%, +10 to 13 MHz | +0.30% (+0.18 to +0.52); per clock −0.2%, +8 MHz | +0.06% (−0.18 to +0.26) |
| A and B evict_last | +0.91% | | +0.14% |
| B evict_last + `L2::128B` | +0.87% | | +0.15% |
| A evict_last | +0.16% | +0.08% | |
| A and B coherent (`ld.global`) | −0.14% | +0.16% | |
| A and B `L2::128B` | −0.11% | +0.17% | |
| `-Xptxas --register-usage-level=8` | +0.29% (per clock +0.41%) | +0.29% | |
| B evict_last + level 8 | +1.11% | | +0.54% |

- **B evict_last is the win.** On the 2080 Ti every build with it read 89.2-89.6 TH/s and both release builds
  88.3-88.9, in all legs of two races. The gain is clock at the same work a clock: B's lines stay in L1 for the
  scheduler's other warp, which reads the same B, so less comes from L2 and the card runs 10-13 MHz faster under
  its power limit. It shrinks where the card has more headroom or less L2 traffic to save: +0.3% on the 2070, level
  on the 2060. It is now the default (commit 050b284f): the machine code is the release's instruction for
  instruction with the B loads' cache bits set (rg kernel `54c72c5e58a9`, first kernel `a0eab2bdc6f4`), and
  `-DPEARL_BD_LD_B=0` builds the release fold byte for byte. (050b284f's message says +0.18% on the 2060; the
  finished race says +0.06%.)
- **register-usage-level 8 is +0.3 to +0.5% on all three cards, on top of evict_last.** It does not ship from
  here: CI compiles every architecture in one nvcc call, so the flag would rebuild the sm_80 to sm_120 kernels too.
  Shipping it needs the sm_75 cubin built on its own (as a patched cubin is) or a per-arch build step.
- **`nvidia-smi -lgc` is refused on Salad** ("Unable to set GPU locked clocks ... Unknown Error"), so the
  watts-at-a-locked-clock instrument (goal-nvk14) cannot be used there.

**A SASS load placement.** `S/nv20c/mover.py` (scratchpad) moves LDGs inside the rg kernel's chunk loop, re-derives
their waits, and checks that every register dependence over two loop turns and the loop exit keeps the protection
(scoreboard wait or cycle distance) it had. What it found about the loop: the 8 loads at its top are pinned (their
address register is reused at once); the 8 for the next turn sit in the readout as latency filler, and moving them
out costs 30-68 cycles of stall repairs; a moved load may not cross the six SHFLs (offsets the driver rewrites);
IMMA stall fields above 4 do not decode. The one placement with no repairs, `rgp4` (four loads moved later
between IMMAs), lost 29.4% on the 2080 Ti: it put a load on scoreboard 4, which the loop's loads use to release
their address registers, so the next turn's address writes waited for its data. `rgp4s` uses scoreboard 5 instead:
−35.4% on the 2070 SUPER (42.6 against 66.0 TH/s, at 1886 MHz: the card idles). So scoreboard 4 was not the
whole story: something else in a moved load costs a full load latency per turn, which my checker does not model
(it guards correctness, and verify-hits passed 400/400 / 383/383). Not found; both lists stay in
`probes/turing/sm75/` for whoever looks next. Placing loads by hand is not a cheap route on Turing.

**Where the rows stand.** `bench.sh` on the same PCs right after the races (stand-in pool at 2^210, 180 s after
60 s warm-up, sample means from 60 s; ours is the published v0.6.15, which still loads B without evict_last):

| PC | ours v0.6.15 | PeakMiner 2.20.7 | ours / PeakMiner | with the shipped change |
|---|---|---|---|---|
| RTX 2080 Ti (`5a02bd58`, 250 W) | 87.99 TH/s, 1384 MHz, 241 W, 78 °C | 89.16, 1365 MHz, 242 W, 77 °C | 98.7% | about 99.3% (+0.58%) to 99.6% (+0.92%) |
| RTX 2070 SUPER (215 W) | 66.03, 1749 MHz, 211 W, 73 °C | 68.25, 1748 MHz, 213 W, 73 °C | 96.7% | 96.7% (level) |

- On this 2080 Ti, unlike goal/nv-20's (where PeakMiner's lead was 1.2% of clock), PeakMiner runs 19 MHz *lower*
  and does 2.7% more work a clock; on this 2070 SUPER it does 3.3% more at the same clock and watts. So what is
  left on RTX 20 is work a clock, not energy, and evict_last (an energy gain) cannot close it on the 2070 SUPER.
- Not measured: the 2060 against PeakMiner today (the box stopped logging and was deleted). Its race said +0.06%.
- Box time: nv20c-2070-1 47 min, nv20c-2080ti-1 2 h 0 min, nv20c-2060-1 48 min, nv20c-2070-2 41 min; about 4 h
  15 min, including about 25 min idle after I restarted two boxes twice within 13 s.

### Salad, 12 November: the RTX 20 folds rebuilt at ptxas level 8 with ROWPAIR + LAST_RO, +0.5-0.6% (branch goal/nv-20d)

**What shipped.** The two RTX 20 folds are built a second time, for sm_75 alone, with ptxas
`--register-usage-level=8` and with `PEARL_BD_ROWPAIR` and `PEARL_BD_LAST_RO` on. nvcc hands `-Xptxas` to every
architecture of a call, and CI builds sm_61 to sm_120 in one call, so this build is its own object:
`src/pearl_kernel_t75.cu` includes `pearl_kernel.cu` whole with every name that has linkage renamed (the kernels, the
`__device__` variables, `blake3_chunk_cv` and the tall fold's host helpers), and turns the two switches on first. CI
compiles it on the two CUDA 12.8 entries and links it beside `pearl_kernel.o`, which does not change, so every other
card's machine code is byte for byte what it was. The host (`PEARL_T75_RUL8=1`, which CI passes) launches
`pearl_t75_tile_fold_wmma(_rg)` on an RTX 20 card when that kernel's sm_75 body loaded, and then draws A' with the
same object's `pearl_t75_materialize16_afrag` (ROWPAIR permutes A's rows). The fold name ends in `RUL8 ROWPAIR
LAST_RO`. A CI step fails the job if either fold spills: both 249 registers, 0 spill. Levels 6 to 10 give this fold
the same machine code; level 5 differs. Commits 6945261d, c8cf5f27, b2b8ad08; CI native core runs 43208653892 and
43226561346 green on all nine jobs; `turingT75.test.js` checks the renames against every linked name in the kernel's
unit, the CI step, the host's pick and the A' draw. Full earn suite: 134 suites, 3,408 tests passed, 65 skipped.

**Races** (`nv20box.sh`; `%t75` builds the new object, `-D` flags apply to both objects; hashrate.js legs of 90 s
taking turns, the first leg left out; "paired" = each leg against the control legs on either side; every build in a
row below passed verify-hits.js 400 of 400 unless the row says otherwise):

Level 8 alone (the object without the two switches, commit 6945261d), against the default:

| PC | default | level 8 | A/A twin | level 8, paired |
|---|---|---|---|---|
| RTX 2080 Ti (260 W, held at 256-257 W, 57-66 °C, 1555-1590 MHz, driver 621.14) | 95.51 TH/s | 95.74 | 95.54 | +0.06% (−0.14 to +0.23); twin −0.28% |
| RTX 2070 SUPER (204 W, 198-202 W, 66-67 °C, 1773-1787 MHz, 621.01) | 67.16 | 67.25 | 67.15 | +0.12% (+0.04 to +0.26); twin −0.06% |
| RTX 2060 6 GB (170 W, but 81-86 W, 64 °C, 1258-1309 MHz, 617.14; a noisy PC) | 35.09 | 35.28 | 35.04 | +0.25% (−1.18 to +1.63); twin −0.62%; verify 381/381 in 90 s |

The old fold switches at level 8, against level 8, on a second RTX 2080 Ti (250 W, 239-242 W, 72-73 °C, 1441-1451
MHz, 617.42; the steadiest card of the day: the twin read +0.00%, −0.02 to +0.02):

| build | TH/s | paired | per clock |
|---|---|---|---|
| level 8 | 93.43 | | |
| + ROWPAIR | 93.20 | −0.25% | −0.40% |
| + LAST_RO | 88.66 | −5.12% | −4.67% |
| + ROWPAIR + LAST_RO | 93.84 | **+0.44%** (+0.43 to +0.44) | +0.35% |
| + SNAKE | 92.12 | −1.41% | −1.36% |

ROWPAIR + LAST_RO against the default, means of three legs each:

| PC | default | level 8 | level 8 + ROWPAIR + LAST_RO | the same, built again (twin) | ROWPAIR + LAST_RO at level 5 |
|---|---|---|---|---|---|
| RTX 2070 (140 W of 175, 134-135 W, 63 °C, 1286-1296 MHz, 610.88) | 43.32 | 43.33 (+0.02%) | **43.55 (+0.53%)** | 43.54 (+0.52%) | 43.45 (+0.29%) |
| RTX 2080 Ti, a third PC (250 W, 240-242 W, 84 °C, 1399-1421 MHz, 621.14) | 87.75 | 87.92 (+0.17% paired) | **88.34 (+0.64% paired, +0.55 to +0.68)** | 88.31 (+0.61%) | 88.22 (+0.50%) |

The shipped form (b2b8ad08, built on the box exactly as CI builds it) against the default, with an A/A twin of the
default:

| PC | default | shipped | twin | shipped, paired | per clock | verify-hits |
|---|---|---|---|---|---|---|
| RTX 2070 (140 W, the PC above) | 43.32 | 43.58 | 43.37 | **+0.52%** (+0.39 to +0.60); twin −0.03% | +0.92% | 400/400 twice |
| RTX 2080 Ti, a fourth PC (250 W, 242 W, 77 °C, 1423-1432 MHz, 621.14) | 90.36 | 91.37 | 90.34 | **+1.00%** (+0.70 to +1.14); twin −0.26% | +0.81% | 400/400 twice |

Each switch alone loses at level 8, as each did at level 5 in rounds 3 to 6; together they win, and most of the
win holds at level 5 too. The gain is work a clock (+0.8% a clock on the 2080 Ti) at the same watts.

**Where the rows stand** (`bench.sh` on the same PCs after the races: stand-in pool at 2^210, 180 s after 60 s
warm-up, sample means from 60 s; "ours" is the published v0.6.15 unless the row says otherwise):

| PC | ours | PeakMiner 2.20.7 | ours / PeakMiner | PeakMiner's work a clock |
|---|---|---|---|---|
| RTX 2080 Ti (first PC) | 92.75 TH/s, 1558 MHz, 256 W | 94.82, 1564 MHz, 257 W | 97.8% | +1.8% |
| RTX 2080 Ti (third PC) | 84.00, 1401 MHz, 242 W | 87.38, 1404 MHz, 241 W | 96.1% | +3.8% |
| RTX 2070 SUPER | 66.94, 1774 MHz, 193 W | 69.21, 1783 MHz, 199 W | 96.7% | +2.9% |
| RTX 2070 (140 W) | 41.46, 1303 MHz, 133 W | 44.40, 1294 MHz, 136 W | 93.4% | +7.8% |
| RTX 2060 | 35.41, 1279 MHz, 85 W | 36.27, 1279 MHz, 84 W | 97.6% | +2.4% |
| RTX 2070 (140 W), the release's CLI on the shipped core | 45.31, 1296 MHz, 135 W | 44.78, 1288 MHz, 136 W | 101.2%, but see below: our samples read 35.1-51.2 | ours +0.6% |
| RTX 2080 Ti (fourth PC), the release's CLI on the shipped core | **91.28**, 1436 MHz, 244 W, 77 °C | 88.64, 1438 MHz, 242 W, 77 °C | **103.0%** | ours +3.1% |

On the fourth 2080 Ti the release's CLI with the shipped core is above PeakMiner by 3.0% (pool credit over the last
2 minutes agrees: 91.30 against 87.23 TH/s), and hashrate.js of the shipped core read 91.37 on the same PC. Read by
hashrate.js of the shipped core (or the default plus its measured gain) against PeakMiner's own figure on the same
PC: 2080 Ti about 101.3%, 101.0% and 103.1% on three PCs; 2070 SUPER about 97.5%; 140 W 2070 97.3-98.2%; 2060 about
97.2%. So the RTX 2080 Ti row is above PeakMiner with this change; the 2070 SUPER, 2070 and 2060 rows are still 2-3%
behind, all of it work a clock.

**bench.sh's figure for our miner moves about ±4% between runs.** Against hashrate.js of the default core on the
same PC, the release's CLI read −2.9% (first 2080 Ti, 92.75 against 95.51), −4.3% (third 2080 Ti, 84.00 against
87.75), −4.3% (140 W 2070, 41.46 against 43.32), within 0.5% on the 2070 SUPER and the 2060, and then, on the shipped
core, +4.0% on the same 140 W 2070 (45.31 against 43.58) and −0.1% on the fourth 2080 Ti (91.28 against 91.37). Its
15 s samples read only the CLI's latest 1 s line, and on RTX 20 those swing with batch boundaries (35.1 to 51.2 TH/s
on the 2070, where PeakMiner's stay within 44.3-44.8), so 8 samples do not settle. Most of the earlier "3-4% under
PeakMiner" on these PCs was this, not the kernel: compare our rows by hashrate.js, by a long run, or by the new
`linemean ours` line (the mean of every 1 s line past the first 60, which is the run's own mean; not yet run on a
box). `bench.sh` also takes `OURS_CORE` (`nv20box.sh OURS_CORE=<build>`): the release's CLI with a core built on the
box, as in the two rows above.

Box time: nv20d-2080ti-1 2 h 55 min (11:23-14:18 over four PCs, about 35 min of it idle: twice Salad did not restart
the box after a set-env and I reallocated it, and it went silent once at 11:56); nv20d-2070-1 37 min; nv20d-2060-1
42 min; nv20d-2070-2 75 min. About 5.5 h in all. Tools: `S/nv20d/` in the scratchpad (hrsum.py takes the control and
twin names as arguments).

### Salad, 12 November: the RTX 20 rows settled, where the fold's time goes, and k 4096 (branch goal/nv-20e)

**The rows, measured the same way for both miners.** `nv20box.sh` bench on the stand-in pool (2^210), our miner as
the v0.6.15 CLI on the shipped core (the t75 object: level 8, ROWPAIR + LAST_RO, built on the box as CI builds it),
each miner's runs of 300 s taking turns. Ours is the `linemean` figure (every 1 s rate line past the first 60);
PeakMiner 2.20.7's is the mean of its 15 s samples, which stay within 1%. Our own 15 s samples and the pool's credit
(about 200 shares a run, so ±7%) do not settle a run; linemean does.

| PC | ours (linemean) | PeakMiner 2.20.7 | ours / PeakMiner | hashrate.js of the same core |
|---|---|---|---|---|
| RTX 2070 SUPER, 215 W, driver 610.88, 82 °C | 64.75, 65.23 TH/s, about 1713 MHz, 200 W | 65.80, 65.16; 1718 MHz, 200 W | **99.25%** | 64.91 |
| RTX 2070 SUPER, 215 W, 617.42, 86 °C | 63.58, 63.66; 1660 MHz, 207 W | 65.18; 1663 MHz, 211 W | **97.6%** | 64.09 |
| RTX 2060 6 GB, 183 W, 610.88 | 47.19, 47.03; 1669 MHz, 179 W | 47.94, 47.96; 1672 MHz, 180 W | **98.2%** | 46.85 |
| RTX 2060 6 GB, 160 W limit, drawing 106 W, 621.14 | 37.69, 37.67; 1371 MHz, 106 W | 37.88; 1345 MHz, 103 W | **99.5%** | 37.70 |

So the 2070 SUPER and 2060 rows are 0.5-2.5% short by PC, smaller than the 2.5-3% estimated before. The plain RTX
2070 row was not re-run (goal-nv20d: 97.3-98.2% by hashrate.js).

**What does not help** (`hashrate.js` legs of 90 s taking turns, each against the shipped core beside it, A/A twin
in every race):

| Change | RTX 2070 SUPER | RTX 2060 | Note |
|---|---|---|---|
| A/A twin | +0.10%, -0.05%, -0.02% | -0.05%, +0.01%, -0.09% | |
| The default fold (before the t75 object) | -0.96% | -0.86% | the shipped form's gain, again |
| Band one down / one up | -0.22% / -0.61% (4 / 6) | -0.21% / -1.19% (3 / 5) | |
| The other kernel (`PEARL_BD_SWEEP_HASH_RG`) | +0.08% | -0.05% | |
| One stream | +0.31% | -0.06% | one leg; the 2070 SUPER drifted 1% in that race |
| ptxas level 4 on the t75 object | +0.08% (per clock -0.4%) | -0.43% | levels 0 and 2 build a longer fold (3,304 instructions against 2,680) |
| SNAKE on top of ROWPAIR + LAST_RO | -1.55% | -1.03% | 247 registers |
| k 4096 (below) | -7.59% | -7.68% (card held at 1680 MHz) | bands 2, 3 and 4 alike |

**k 4096.** A tile at k 4096 does twice the MACs for one hash and one hand-off, and a share's k is the miner's
choice. Built (commits 030c3dbf, be5e019a, 304f6d0d): chunk c < 16 stores its word rotated, chunk c + 16 xors into it
with `red.shared`, and with `PEARL_T75_K=4096` the host answers cardK with 4096 for an RTX 20 card on the t75 folds.
It verified (320 of 320 and 227 of 227 hits at k 4096 in 90 s) and lost 7.6% on both cards, at every band; the
memory controller went from 16% to 32-45%. Reverted (7b905bb9). The hash ablation below says why there was little
to gain.

**Where the shipped fold's time goes.** Timing-only builds of the t75 object (wrong results on purpose, never
shipped), two legs each, paired against the shipped build, raw and per clock:

| Switched off | RTX 2070 SUPER (215 W, 85 °C, 1680 MHz) | RTX 2060 (106 W, 1379 MHz) |
|---|---|---|
| The readout (`PEARL_ABLATE_READOUT`) | +2.74% / +2.41% | +2.24% / +2.87% |
| The transcript hash (`_TRANSCRIPT_HASH`) | +0.32% / -0.14% | +0.31% / +0.31% |
| The readout's shuffle (`_TRANSCRIPT`) | -0.27% / -0.11% | -0.01% / +0.13% |
| Every B load an L1 hit (`_BHIT`) | +3.29% / +0.63% (+43 MHz) | +6.71% / +1.97% (+62 MHz) |
| B loaded once a tile (`_BLOAD`) | +4.53% / +1.92% | +8.22% / +3.99% |

- The hash is not where the time goes any more (0.3%), so nothing that spreads it (a larger k, a split hash) can
  pay. The readout's xor fold, 64 LOP3 a chunk at their floor, costs 2.2-2.8%: about three times their issue slots,
  so it is how they sit among the IMMA, not their number. B's traffic from L2 costs 3-7%, mostly as clock.
- At most the readout and B are where PeakMiner's last 1-2% can come from.

**What shipped: each B pair gated on its own readout fold** (`PEARL_BD_RO_PAIRGATE`, commits ae96818e and
19657553). In the k-lo pass of a chunk's first k-step, each B pair's IMMA wait (`pearl_bd_after`) for the readout's
fold of the accumulators they overwrite. They waited on the fold chain through every pair before them; now each
waits on its own pair's fold, and that is xored onto the chain after. Only `pearl_kernel_t75.cu` turns it on, so
every other card's machine code is unchanged; the RTX 20 folds are 253 registers, no spill, the same instruction
count (rg kernel e04dae371a55 under CUDA 12.6), and the fold name ends in `RUL8 ROWPAIR LAST_RO PAIRGATE`. CI native
core green; full earn suite 134 suites, 3,408 passed, 65 skipped.

| PC | shipped t75 | with PAIRGATE | paired | twin | verify-hits |
|---|---|---|---|---|---|
| RTX 2070 SUPER (255 W limit, drawing 178 W, 83 °C, 1630 MHz, driver 580.88) | 62.39 TH/s | 62.70 | **+0.44%** (6 legs, +0.30 to +0.60), per clock +0.52% | +0.06% | 400/400 |
| RTX 2060 6 GB (160 W, held at 1680 MHz, 153 W, 617.14) | 47.87 | 47.99 | **+0.10%** (5 legs, +0.08 to +0.14), per clock +0.33% | -0.08% | 400/400 |

On the same 2070 SUPER, the release CLI on the PAIRGATE core read 62.46 TH/s (linemean) against PeakMiner's 63.01:
**99.1%**. On the 2060 it read 47.88 against PeakMiner's 48.85 (1677 and 1684 MHz, 153 and 152 W): **98.0%**,
98.4% a clock. So with this change the rows stand at about 98-99.7% (the four PCs above plus +0.1 to +0.4%), still
under PeakMiner on every 2070 SUPER and 2060 PC measured. What is left to try is the readout (2.2-2.8% of the rate,
of which this recovered 0.4) and B's L2 traffic (3-7%, mostly clock); the hash is spent.

Box time: nv20e-2070-1 and nv20e-2060-1, 14:28-18:02 each over four PCs (about 12 min of each idle: set-env did not
restart a running box three times, and each time I reallocated it). About 7 h. Tools: `S/nv20e/` in the scratchpad
(r2.sh to r4.sh are the rounds' settings).

### Salad, 12 November: the lag between a hasher and its B partner, and a 2060 SUPER row (branch goal/nv-20f)

**Where the rows stand.** `nv20box.sh` bench on the stand-in pool (2^210), the v0.6.15 CLI on the shipped t75 core
(PAIRGATE, built on the box as CI builds it) against PeakMiner 2.20.7, 300 s runs taking turns twice. Ours is the
`linemean` figure, PeakMiner's the mean of its 15 s samples from 60 s on.

| PC | ours (linemean) | PeakMiner 2.20.7 | ours / PeakMiner |
|---|---|---|---|
| RTX 2060 SUPER, 175 W, driver 621.01, 65-67 °C | 53.90, 53.84 TH/s; 1692-1695 MHz, 168-169 W | 53.94, 54.07; 1681 MHz, 170 W | **99.75%** (99.0% a clock) |
| RTX 2070 SUPER, 215 W, 621.01, 69-79 °C | 60.50, 60.40; 1736-1746 MHz, 211 W | 61.83, 61.96; 1736-1738 MHz, 211 W | **97.7%** |

Pool credit (about 200 shares a run, ±7%) read ours 55.9 and 55.2 against PeakMiner's 49.8 and 48.2 on the 2060
SUPER, and 63.4 and 62.0 against 56.9 and 63.4 on the 2070 SUPER; it does not settle a run. That 2070 SUPER PC does
about 9% less work a clock than the first one this round (34.7 against 38.3 TH/s per GHz, same card and watts), and
PeakMiner with it.

**What I raced** (hashrate.js legs of 90 s taking turns, each against the shipped t75 core beside it, A/A twin in
every race; every build listed passed verify-hits.js 400 of 400 where verified). The idea: each scheduler holds one
warp that hashes after every hand-off and one that goes on, and the two read the same B. The hash (about 2.3K
clocks) leaves the second warp about 4.5 k-steps ahead for the rest of the tile, with their readouts half
overlapping.

| Build | What it changes | RTX 2070 SUPER (first PC: 211-215 W, 1716-1759 MHz) | RTX 2060 SUPER (168-171 W, 1673-1711 MHz) |
|---|---|---|---|
| A/A twin | nothing | -0.10% (second PC) | +0.06% |
| `HDELAY=1` | each hasher waits 1 chained shared load (the delay builds' control: same chunk loop) | -0.31% (3 legs) | -0.15% (3 legs) |
| `HDELAY=16` | ~500 clocks more lag | -0.64% | -0.20% |
| `HDELAY=32` | ~1,000 | -2.14% | -2.55% |
| `HDELAY=48` | ~1,500 | -3.70% | -4.58% |
| `HDELAY=64` | ~2,000 | -3.93% | -5.30% |
| `LD_A=3` | A's loads evict_first (`LDG.E.EF`; `.cs` is the same SASS on sm_75) | -0.73% | -1.09% |
| `WMAP` | warps re-paired so the two readers of a B slot hash together: no B lag | -2.26% (second PC, 3 legs) | -2.06% (2 legs) |
| `WMAP` + one-sided delay 1 / 16 / 32 / 48 | the B pair that far apart again | -2.40 / -3.37 / -3.53 / -4.41% (second PC) | -2.17 / -2.91 / -3.24 / -4.17% |

- **Lag costs past what the hash makes.** Up to ~500 clocks more is about level; beyond it the loss grows fast and
  the clock falls with it, which fits the lagging warp's B and A lines leaving L1. Less lag does not help either:
  with none (WMAP) both warps ask for the same lines at once, and every lag from there up only adds loss. The ~2.3K
  clocks the hash makes are on the flat part, so there is nothing here to win.
- **B's L2 traffic is at its floor for this tile.** The block's 128x256 int32 accumulators are half the register
  file, so no larger block fits, and the hash-off ablation (+0.3%, goal/nv-20e) leaves little room for lines lost
  while the hasher lags. B through shared memory would add stores and shared reads on the same 64 B/clock path, for
  at most that. Not built.
- **The readout's xor fold is at its floor in instructions:** the chunk loop is 512 IMMA and 127 others (70 LOP3,
  32 LDG, 10 IADD3, 8 PRMT, 1 SHFL, 1 STS, 2 IMAD, LEA, ISETP, BRA), 64 of the LOP3 for 128 values a lane. What it
  costs (2.2-2.8%) is where those sit among the IMMA.
- A clock64 spin or a `NANOSLEEP` as the delay made ptxas rebuild the tile loop (`NANOSLEEP.WARP` put divergence
  guards on the shuffles), so the delay is a chain of dependent shared loads in the hashers' existing branch.
- The switches are in commits 81c72c76, 03bcf71b and d65715f7 and were taken out again (d0a3cc1f): the Turing tests
  compare the fold with copies of its source. Full earn suite on the branch: 134 suites passed (1 skipped), 3,416
  tests passed, 65 skipped.

Box time: nv20f-2070-1 18:24-20:03 (two PCs; set-env did not restart it once and I reallocated it) and nv20f-2060-1
18:24-19:59, about 3.3 h in all. Tools: `S/nv20f/` in the scratchpad (r1.sh to r4.sh are the rounds' settings,
benchsum.py reads a bench).

### Salad, 12 November: a hand-scheduled chunk loop for the RTX 20 folds (branches goal/nv-20g and goal/nv-20h)

**What was raced.** `sched75.py` (`earn/native/probes/turing/sched/`) rewrites the t75 folds' chunk loop (639
instructions, 512 IMMA) in the built cubin: same instruction words and registers, a new order, and new stall counts,
scoreboards, waits and reuse flags. `sm75patch.py` applies the lists in `probes/turing/sm75g/`. Every list passed the
scheduler's dependence checker, a register dataflow check over four turns (9,412 reads, none changed) and nvdisasm,
and every build raced passed `verify-hits.js` 400 of 400. `hashrate.js` legs of 90 s taking turns against the shipped
t75 core, with an A/A twin.

| Build | What it is | RTX 2070 SUPER (167.7 W limit, 166 W, 1484 MHz) | RTX 2060 6 GB (172.5 W, 169 W, 1676 MHz) | RTX 2070 (175 W, 166-172 W, 1575 MHz) |
|---|---|---|---|---|
| A/A twin | nothing | -0.02% | -0.04%, +0.02% | -0.11% |
| s40 | new order, planned for a 40-clock IMMA result | -0.66% (per clock -0.78%) | -0.62% (-0.15%) | -0.46% (-0.35%) |
| s60 | the same at 60 clocks | -0.22% (-0.04%) | -0.51% (+0.31%) | +0.25% (+0.36%), 4 legs, -0.01 to +0.43 |
| m1 | ptxas's order, all control bits re-derived | -2.25% | -2.39% | -1.64% |
| s40b | s40 with bank-aware reuse and ptxas's yield bits | | -0.19% (-0.24%) | |
| s60b | s60 the same | | -1.49% (-1.28%) | |
| m1b | m1 the same | | -7.73% | |
| m1p | ptxas's order, reuse and yield bits; only scoreboards and waits re-derived | | -7.69% | |

The RTX 2070 row is a second PC that the 2070 group moved to on a restart; it re-ran the first round.

- **Nothing beats the shipped loop.** s60 on the plain 2070 (+0.25%) is inside two twins' spread and lost on the
  other two cards. Nothing shipped.
- **The scoreboards cost the most.** m1p keeps ptxas's order, reuse and yield bits and changes only which scoreboards
  the loads and IMMA use and where the waits go (126 waits against ptxas's 78): -7.7%. So the scheduler's scoreboard
  plan is far worse than ptxas's, and the new orders (s40b at -0.2%) win back about 7.5% of that. A new order with
  ptxas-quality scoreboards is the untested case that could pay.
- **Yield bits matter.** m1 differs from m1p only in its yield bits (set on every instruction with a stall under 5,
  43 more than ptxas) and its reuse flags, and it lost 2.4% where m1p lost 7.7%. With bad waits, letting the other
  warp issue covers much of the loss.
- **Reuse flags are per register bank.** ptxas keeps an IMMA operand's reuse flag across an IMMA that reads the
  other bank (register number mod 2) in that slot, so the cache holds one entry per slot per bank, and only IMMAs
  evict it. The first lists kept a flag only when the very next IMMA read the same register, which left 96-100 IMMAs
  with both uncached operands in one bank (ptxas 74). `reuse2.py` applies the bank rule: on ptxas's own loop it gives
  every flag ptxas set and 28 more (57 such IMMAs). The race did not separate this from the yield change.
- The encoding has no yield bit with a stall of 12 (nvdisasm: undefined opex value 0x1c); ptxas uses yield with
  stalls up to 8.
- The cards ran at their power limits in every leg, so a raw gain also needs the power per hash to hold.

Box time: nv20g-2070-3 21:15-22:40 (two PCs) and nv20g-2060-3 21:15-22:43, about 3 h in all. Stopped there because the
four Salad slots were needed for the PR #293 re-benchmark. Tools: `S/nv20g/` and `S/nv20h/` in the scratchpad
(r1.sh, r2.sh, hrsum.py, miss.py).

### Salad, 13 November: waits cost nothing, yield bits do; why the two-warp model was wrong (branch goal/nv-20j)

**What was raced.** Patch lists for both RTX 20 folds that keep ptxas's chunk loop as it is (order, stalls, reuse,
registers) and change only control bits (`probes/turing/sm75i/`, `sm75j/`): r0 drops the 28 of its 78 waits that
`sched75.py`'s dependence checker finds unneeded (goal-nv20i's `climb.py`; its race had stopped after the builds);
c1 also moves three IMMA scoreboards; the y* lists change only yield bits. nw is timing only: it drops 64 waits, every
one that covers only an IMMA's result or operand read, so it computes wrong values on purpose. ro is
`PEARL_ABLATE_READOUT`, also timing only. `hashrate.js` legs of 90 s taking turns against the shipped t75 core,
each leg paired with the control legs beside it, A/A twin in every race, first leg left out. r0 (on all three
cards), c1, ya, y0 and yp passed `verify-hits.js` 400 of 400; the other yield lists change only yield bits, which
cannot change a value, and were not verified.

| Build | What changes | Model | RTX 2080 SUPER (250 W, driver 616.56, held at 246-248 W, 67-69 C, 1784-1791 MHz) | RTX 2060 6 GB (160 W, driver 616.92, 155-156 W, 81 C, 1645-1666 MHz) | RTX 2080 Ti (260 W, driver 621.01, 257-259 W, 59-62 C, 1511-1528 MHz) |
|---|---|---|---|---|---|
| A/A twin | nothing | | -0.00% (2 legs) | -0.01%, +0.00% | -0.09% (2 legs) |
| r0 | 28 of 78 waits dropped | +0.6% | -0.03% (4 legs, -0.08 to +0.01) | -0.00% (3 legs), +0.02% | +0.02% (3 legs, -0.01 to +0.05) |
| c1 | r0 and three IMMA scoreboards moved from 0 to 1 | +1.2% | -0.04% (4 legs, -0.12 to +0.02) | | |
| nw (timing only) | 64 of 78 waits dropped | +4.0% | | -0.02% (2 legs) | -0.00% (3 legs, -0.11 to +0.13) |
| ro (timing only) | no readout | | | +1.86% (per clock +2.37%) | +2.63% (3 legs; per clock +2.39%) |
| ya | yield set on all 639 instructions (ptxas: 593) | +1.7% | | -1.20% (3 legs; per clock -0.32%, clock -14 MHz) | -1.03% (3 legs; per clock -0.20%, clock -12 MHz) |
| y0 | yield cleared on all (but the driver-listed SHFL) | +0.7% | | -3.86% (per clock -3.11%) | |
| yx | cleared on every non-IMMA | +0.1% | | -0.79% (per clock -0.61%) | |
| yi | cleared on every IMMA | +0.4% | | -3.57% (per clock -3.11%) | |
| yp | cleared on every other IMMA (a warp issues its IMMA in pairs) | | | -2.78% (3 legs; per clock -2.17%) | |
| yl | cleared on the 32 loads | | | -0.36% (3 legs; per clock -0.05%) | |
| yc | cleared on the 48 other non-IMMA that have it | | | -0.32% (3 legs; per clock -0.50%) | |

The model is `simlist.py`'s copy of climb.py's two-warp simulation at an IMMA result of 32 clocks and an L2 load of
250 (the setting that fits nv20h's m1 and m1p best).

- **Waits cost nothing on the card.** Dropping 64 of the loop's 78 waits, which the model says is 4.0% of the tensor
  pipe and most of what ptxas's loop loses, changes nothing on either fold (-0.02% on the 2060's, -0.00% on the 2080
  Ti's, inside the twins). So no scoreboard or wait plan can gain here, and r0 and c1 (level on all three cards) do
  not ship.
- **Yield bits are the control bits that matter, and ptxas's beat every group change tried.** Clearing yield on the
  IMMA costs 3.6-3.9% (2.8% on every other one), so the two warps of a scheduler must take turns IMMA by IMMA
  (why is not measured; one guess is that it keeps them close enough to share B's lines in L1, as goal-nv20f's lag
  races found matters). Setting it on the 46 instructions where ptxas leaves it off (the readout's xors, the address
  adds) costs 1.0-1.2% on both cards, most of it clock at the same watts; clearing it on the loads or on the other
  non-IMMA costs 0.3-0.4% each, 0.8% together.
- **Why the model was wrong.** Its premise holds: a pure IMMA stream fills Turing's pipe with one warp a scheduler
  (price75 `ceil` with 128 threads, below: 99.32%) as with two (99.997%). But it charges each wait as lost issue
  time when both warps are stalled, and the card does not lose that time: October's `sass.sh` measured ptxas's own
  loop at 98.9% of the IMMA peak with two warps a scheduler, where the model says 94-96%, and nw now shows the waits
  are free inside the whole fold. The 94.4% that matched the model was the whole fold, readout and hash included. Its
  yield rule (a set bit keeps issuing from the same warp) gets the sign of all four yield changes wrong, and with the
  rule inverted it still has y0 at +1.4 to +1.8%. So the model cannot rank schedules: the re-ordered loops of
  goal/nv-20g and -20h were level or worse on the card because there was nothing in the waits for them to win.
- **Where the loop does lose** is in the instructions themselves: no readout is +2.4% a clock on the 2060 and the
  2080 Ti (its 64 xors, 8 PRMT, a shuffle and a store a chunk), as goal-nv20e found, while its waits are free.

**price75: one warp a scheduler against two** (`price75 <stream> 25 1500 <threads>`, two RTX 2060 PCs, each
stream twice; pipe is IMMA a clock an SM over the peak of one):

| Stream | 256 threads (two warps a scheduler) | 128 threads (one) |
|---|---|---|
| `ceil`: operands in registers | 99.997% (1786-1879 MHz) | 99.32% (1792-1842 MHz) |
| `rel`: the probe's own A and B loads from global (ptxas schedules it badly, 255 registers) | 67.5-67.6% (1700-1746 MHz) | 82.3-82.4% (1678-1707 MHz) |

One warp alone keeps the pipe full. With loads in the stream, two warps a scheduler did much worse than one; that is
the probe's own poorly scheduled stream, not the fold, but it says the two warps of a scheduler can get in each
other's way when both load.

**Where the rows stand on these PCs.** `nv20box.sh` bench on the stand-in pool (2^210): the v0.7.0 CLI on the
shipped t75 core, PeakMiner 2.20.7, SRBMiner 3.8.7 and OneZeroMiner 1.8.5, 300 s each after a 60 s warm-up, in the
order PeakMiner, ours, SRBMiner, OneZeroMiner, ours, PeakMiner. Each figure is the mean of the miner's 15 s samples
over its runs, as in the official table; ours also as `linemean` (the mean of the CLI's 1 s lines past the first 60,
which its 15 s samples swing around on RTX 20) and as `hashrate.js` of the same core in the race above.

| PC | PeakMiner | Ours: samples / linemean / hashrate.js | SRBMiner | OneZeroMiner | Ours, % of PeakMiner (samples) | PeakMiner's work a clock over ours |
|---|---|---|---|---|---|---|
| RTX 2080 SUPER (250 W, driver 616.56, 68-69 C) | 82.87 TH/s, 1797 MHz, 247 W | 81.36 / 80.75 / 81.62-82.03, 1785 MHz, 248 W | 81.62, 1784 MHz | 76.16, 1832 MHz | 98.2% (hashrate.js 98.5-99.0%) | +1.2% (hashrate.js +0.9%) |
| RTX 2060 6 GB (160 W, driver 616.92, 81-82 C) | 48.79, 1654 MHz, 156 W | 47.91 / 47.77 / 47.85-47.88, 1657 MHz, 156 W | 46.57, 1646 MHz | 45.52, 1718 MHz | 98.2% (hashrate.js 98.1%) | +2.0% |
| RTX 2080 Ti (260 W, driver 621.01, 60 C) | 98.04, 1509 MHz, 258 W | 99.24 / 99.17 / 99.16-99.48, 1530 MHz, 258 W | 96.45, 1510 MHz | 94.34, 1573 MHz | **101.2%** | +0.2% (ours runs 1.4% faster at the same watts) |

- Against PeakMiner, ours is 98.2% on the 2080 SUPER and the 2060 and 101.2% on the 2080 Ti, as in the official
  table (2060 97.2%, 2080 Tis 101.4%). PeakMiner does 1.2-2.0% more work a clock on the smaller cards at the same
  clock and watts; on the 2080 Ti its work a clock is ours and ours holds the higher clock. SRBMiner and OneZeroMiner
  are behind both on all three PCs.

**What is left.** Nothing in the chunk loop's control bits pays: waits and scoreboards are free, ptxas's yield bits
beat every group change, and re-ordering (goal/nv-20g, -20h) was level. What still costs on the card is
instructions: the readout's (2.4% a clock) and B's loads (goal-nv20e: 1.9-4.0% a clock with B loaded once a tile).
So the next tries are fewer instructions a chunk, not a new schedule. Beside the 64 xors (their floor), the readout
has a shuffle, a store and the 8 `PRMT ..., 0x3210, ...` that tie each B pair's IMMA to its fold (`pearl_bd_after`,
PAIRGATE); a region held in one thread's own accumulators (PeakMiner's tile) would need no shuffle. And fewer B
loads a MAC. Single yield bits could still be searched on a card, but every group of them lost.

Box time: nv20j-2080-1 17:16-18:21 (an RTX 2080 SUPER from the 2080 class), nv20j-2060-1 18:21-20:51 (two PCs:
Salad moved it at 18:35, and set-env restarted it twice on the second), nv20j-2080ti-1 20:51-21:58; about 4 h 40
min. Tools: `S/nv20j/` in the scratchpad (r1.sh to r6.sh, simlist.py, mkyield.py, climbj.py,
hrsum.py, benchpc.py).

### Salad, 14 November: a cheaper readout for RTX 20, tried four ways; nothing ships (branch goal/nv-20k)

The aim was to win back some of the readout's 2.4% a clock without breaking anything. The t75 chunk loop
(CUDA 12.8, sm_75, ptxas level 8) is 639 instructions a chunk: 512 IMMA, 70 LOP3, 32 LDG, 10 IADD3, 8 PRMT, 2
IMAD, LEA, SHFL, STS, ISETP, BRA. The readout's share is 68 LOP3 (64 is the floor for 128 sums), the 8 PRMT gates,
the shuffle and the store.

Checked on rig01 by compiling only (registers and spills):

- **Fewer gates spill.** Every form that drops PRMT gates spills at 255 registers: no gates 72 bytes, gates on one
  row half only 60-72, alternate gates 96, two gates 72. One chain a B pair (64 LOP3, 4 gates) spills 72, and 68 with
  the odd B word gated too. NVVM re-associates the xors, so the tree's shape in the source does not change the code.
  The 8 gates stay.

Raced with `nv20box.sh` (hashrate.js legs of 90 s taking turns, each leg paired with the control legs beside it, an
A/A twin of the shipped t75 core). Group nv20k-2060-1 in llmjob2.

| Build | What it changes | Instructions a chunk | RTX 2060 6 GB (170 W, 610.62; 1655 MHz, 166 W, 75 C) | RTX 2060 SUPER (175 W, 610.88; 1563 MHz, 162 W, 82 C) |
|---|---|---|---|---|
| twin | nothing | 639 | -0.12% (-0.32 to +0.07), per clock -0.02% | +0.03% (-0.01 to +0.05), per clock +0.15% |
| im (`PEARL_BD_IMMOFF`) | each k-step's load offset goes into the load's immediate: 8 IADD3 fewer | 631 | -0.13% (-0.28 to -0.04), per clock -0.13% | |
| pg2 (`PEARL_BD_RO_PAIRGATE=2`) | the readout's xors at their floor (one sum moved between the halves' folds, one chain): 66 LOP3 | 635 | | +0.03% (-0.00 to +0.07), per clock +0.32% |
| ll (timing only) | every thread hashes, no shuffle: what a tile of one thread's sums would cost | 637 | -2.21% (-2.35 to -2.05), per clock -3.12% | |
| ah (timing only) | every thread hashes, shuffle kept | 639 | -2.43% (-2.73 to -2.23), per clock -2.23% | |
| ro (timing only, `PEARL_ABLATE_READOUT`) | no readout in the loop | 558 | about +1.1% raw, +1.8% per clock (3 legs, a run cut short) | +2.65% (+2.63 to +2.68), per clock +2.26% |

verify-hits 400 of 400 for im (623 hits) and pg2 (660 hits).

- **Address steps cost nothing.** im drops 8 IADD3 a chunk and is level. So the loop's ALU steps don't each cost the
  same amount of pipe time. pg2 drops 4 LOP3, which read no sums, and is level too. What makes the readout cost is
  reading the 128 sums, or the gates' wait, and neither can go: every sum has to be read once a chunk, and fewer
  gates spill.
- **A tile of one thread's sums (PeakMiner's) loses about 2% on our fold.** It would drop the shuffle, which is worth
  nothing (goal-nv20e). But a 128-sum tile doubles the hash work, and with every thread hashing no warp is left on
  the tensor pipe while the others hash. Today one warp of each scheduler hashes and the other keeps the pipe fed.
- **B loads.** goal-nv20e priced B once a tile at +1.9 to 4.0% a clock and every B load an L1 hit at +0.6 to 1.9%.
  Fewer B loads a MAC needs a larger warp tile. That needs registers the fold doesn't have (253 used, and fewer
  gates already spill), so nothing was built.

Bench on the RTX 2060 6 GB, same PC (v0.7.0 CLI on the im core, stand-in pool 2^210, 300 s turns, PeakMiner,
ours, ours, PeakMiner, sample means): PeakMiner 2.20.7 48.10 TH/s at 1660 MHz, 167 W; ours 47.30 at 1665 MHz, 166
W (linemean 47.64, 47.52). **Ours 98.3%**, and PeakMiner does 2.0% more work a clock, as in goal-nv20j.

The switches (`PEARL_BD_IMMOFF`, `PEARL_BD_RO_PAIRGATE=2`, `PEARL_ABLATE_ALLHASH`) are on goal/nv-20k, off by
default and not on this branch. Box time: nv20k-2060-1 00:57-03:00 (two PCs: set-env did not restart the box, so it
was reallocated at 02:28). Tools: `S/nv20k/` in the scratchpad (bld.ps1, census.py, r1.sh, r2.sh).

### Salad, 14 November: freeing registers in the RTX 20 loop; nothing ships (branch goal/nv-20m)

The aim was to free registers so the readout could drop gates without spilling, and to price what the readout's
cost is made of. A liveness count over the t75 chunk loop (CUDA 12.8, sm_75, level 8; `S/nv20m/live.py`) puts its
peak at 234 live registers, in the readout k-step, and 164 to 212 elsewhere.

New switch `PEARL_BD_SB` (off by default): single-buffer chosen operand halves (bit 0 A's k-hi quads, 1 B's k-hi,
2 A's k-lo, 3 B's k-lo). A single half loads one pass ahead of its use, into the registers its last pass read,
instead of two passes ahead into a second buffer. Same 639 instructions. Loop peak: sb1 226 (246 registers), sb2
224 (242), sb3 223, sb15 223. On single buffers a readout with 4 gates (one row half) or one gate builds with no
spill inside the loop; on the shipped buffers the same forms spill 72 bytes with loads and stores in the loop.
Also `PEARL_BD_GATEMASK` (drop chosen gates) and `PEARL_ABLATE_FOLD2` (timing only: the chunk readouts fold 2 of
each pair's 16 sums, every gate kept; the tile's last readout still folds all, so no IMMA is dropped).

Raced with `nv20box.sh` (hashrate.js legs of 90 s taking turns, paired with the control legs beside them, A/A twin
of the shipped t75 core). Group nv20m-2060-1 in llmjob2.

| Build | What it changes | RTX 2060 SUPER (184 W, 617.14; 1556-1577 MHz, 155-162 W, 86 C) | RTX 2060 6 GB (172.5 W, 610.88; 1704-1726 MHz, 167-170 W, 72 C) |
|---|---|---|---|
| twin | nothing | +0.01% (-0.04 to +0.06) | -0.07% (-0.10 to -0.03) |
| sba (`SB=1`) | A's k-hi single | | -0.68% (-0.88 to -0.53), per clock -0.66% |
| sbb (`SB=3`) | A's and B's k-hi single | -1.53% (-1.70 to -1.36), per clock -1.77% | -3.50% (-3.58 to -3.42), per clock -3.92% |
| sbt (`SB=2`) | B's k-hi single | -4.25%, per clock -4.19% | |
| sbf (`SB=15`) | all four halves single | | -3.29%, per clock -3.91% |
| gla | sbb, gates on row half 0 only (4 PRMT) | -3.68%, per clock -3.26% | |
| glb | sbt, gates on row half 1 only (4 PRMT) | -4.69%, per clock -4.11% | |
| gone | sbt, one gate | -10.36% | |
| gno | sbt, no gates (one LDL a chunk) | -10.34% | |
| fda (timing only) | sba, FOLD2 | | +0.79%, per clock +0.55% |
| fdb (timing only) | sbb, FOLD2 | | +0.10%, per clock -0.31% |
| ro (timing only) | no readout | +2.40%, per clock +2.09% | +1.52%, per clock +1.91% |
| ros (timing only) | sbb, no readout | +1.91%, per clock +1.81% | +0.45%, per clock -0.39% |

verify-hits PASS (60 s) for sba, sbb, sbt, sbf, gla, glb, gno and gone (248 to 318 hits each, none failed).

- **Single buffers free registers but cost time.** A's k-hi alone costs 0.7%, both k-hi halves 1.5 to 3.5%, B's
  k-hi alone 4.3%. Loading one pass ahead is too late for these loads, B's above all.
- **The gates are not a cost.** On the same buffers, gates on one row half lose 0.4 to 2.2% and one gate or none
  lose about 6%. Without them ptxas runs the next k-step's IMMA ahead of the fold.
- **The readout's cost is reading the sums, not the gates' wait.** On sbb, FOLD2 gets back 3.6% of the full
  readout's 3.95% (ros against sbb), with every gate still waiting. On sba, FOLD2 gets back 1.5%. Every sum has to be
  read once a chunk and the xors are at their floor (goal-nv20k), so the cost can't be cut by instruction count.
- **A larger warp tile does not fit.** 64 x 96 needs 192 accumulators; with every half single-buffered (40
  operand registers, and that alone costs 3.3%) the loop would sit at about 240 before the readout's peak.
- The losing builds read 22 to 38% memory utilisation where t75 reads 12 to 23%. The 2060 6 GB's numbers moved
  between legs of one build, so this is a hint, not a measurement.

Nothing is closer to PeakMiner (98.3% on the same 2060, goal-nv20k); no bench this time, since nothing won. The
switches (`PEARL_BD_SB`, `PEARL_BD_GATEMASK`, `PEARL_ABLATE_FOLD2`, with goal-nv20k's three cherry-picked) are on
goal/nv-20m, off by default (default sm_75 machine code unchanged) and not on this branch. Box time: nv20m-2060-1
03:18-04:52 (two PCs: set-env did not restart the box, so it was reallocated at 04:06). Tools: `S/nv20m/` in the
scratchpad (bld.ps1, census via S/nv20k/census.py, live.py, r1.sh, r2.sh).

### Salad, 14 November: what the RTX 20 readout's reads cost, and a hand-placed readout; nothing ships (branch goal/nv-20n)

The aim was to make the readout's reads of the 128 sums cheaper with hand-placed SASS. Every build is a patch list
on the shipped t75 loop (CUDA 12.8, sm_75, level 8), both fold sections, in `probes/turing/sm75n/` (tools in
`sm75n/tools/`). Raced with `nv20box.sh` (hashrate.js legs of 90 s taking turns, paired with the control legs beside
them, A/A twin of the shipped t75 core). Groups nv20n-2060-1 and nv20n-2060-2 in llmjob2. Timing-only builds compute
wrong values on purpose.

**Where the cost is.** Found on rig01 first: ptxas already keeps every readout LOP3 off three reads in one bank (0 of
70), and it drops IMMA reuse flags across every instruction with its yield bit clear.

| Build | What it changes | RTX 2060 SUPER (175 W, 616.92; 1685-1693 MHz, 164 W, 73 C) | RTX 2060 6 GB (about 150 W; 1514-1605 MHz, 150 W, 56-80 C) | RTX 2060 6 GB (172.5 W, 621.01; 1672-1682 MHz, 169 W, 75 C) |
|---|---|---|---|---|
| twin | nothing | -0.00% (4 legs, -0.03 to +0.01) | per clock -0.15% | +0.05% (4 legs, +0.02 to +0.06) |
| rz (timing only) | the 68 readout LOP3 read RZ: same issue, control bits and waits | +1.79% (+1.78 to +1.80) | per clock +1.18% | +1.66% (+1.59 to +1.71) |
| nop (timing only) | they are NOPs | +1.96% | | |
| ro (timing only) | no readout | +2.37% | | |
| ru | 28 IMMA reuse flags ptxas left off in the readout k-step | -0.04% (-0.05 to -0.02) | | |
| rzs (timing only) | the 128 sum reads read RZ | | per clock +1.14% | |
| rzt (timing only) | the 67 temporary reads read RZ | | per clock +0.51% | |
| rss (timing only) | the 128 sum reads moved to registers no IMMA writes, banks kept | | per clock -0.18% | |
| rsb (timing only) | the same, all three of a LOP3's reads in one bank | | per clock -1.48% | |
| c0 (timing only) | slot c of each readout LOP3 reads RZ: one read fewer each | | | +0.85% (+0.80 to +0.95) |

The 150 W PC heated from 56 to 80 C during the race and its clock fell, so only its per-clock figures (3 legs each)
are given. verify-hits 400 of 400 for ru (688 hits).

- **The cost is register-file cycles, not which registers are read.** The same reads moved to registers no IMMA
  writes cost the same (rss), and three reads in one bank cost 1.3% more (rsb). A LOP3 with three reads over two
  banks takes two cycles; with one read fewer, two banks, one cycle (c0 gets back half of rz). Issue slots are worth
  0.15% (nop against rz).
- **IMMA reads don't count the same way.** ru saves 28 IMMA operand reads in the same k-step and is level.
- **Ceiling:** free reads (rz) are +1.7 to +1.8%. PeakMiner does 2.0% more work a clock on the 2060, so the readout
  alone can't put the 2060 over 100%.

**The hand-placed readout.** The only way to give a LOP3 a free operand without more instructions is the reuse
cache. nvdisasm accepts a reuse flag on IMMA's C operand. Two probes on ptxas's loop passed verify-hits: cr (a LOP3's
reuse flag before the IMMA that reads that register as C; 555 hits) and cb (the IMMA's C reuse flag before a LOP3 that
reads that register from slot c; 502 and 634 hits). `hr.py` and `hr2.py` drop ptxas's 68 readout LOP3 and fold the
128 sums in two xor chains (R246, even bank, and R1, odd bank: the only registers free across the chains; the
kernel's highest is R250, and the first lists' R251/R252 stopped the card with an illegal instruction). Each chain
LOP3 sits next to the IMMA that next reads its pair as C. The 8 PRMT gates read RZ instead of the fold (their value
never depended on it). Every other instruction keeps its order, registers, waits and stalls (a dropped LOP3's wait
and stall move to its neighbours). Both tools check that the chains fold the same 128 values and that every other
instruction still reads the same writer.

| Build | Chain LOP3 placement | verify-hits | Timing |
|---|---|---|---|
| hr | after the IMMA; one half from its C cache, the other read before the IMMA's in-place result lands | FAIL, 29 of 400 | -1.11% raw, -0.22% per clock (1 leg, 168 W 2060, twin -0.04%) |
| hrn | the same, no cache | FAIL, 30 of 400 | -0.66%, -0.44% per clock (1 leg) |
| hr0 | hr with the chain LOP3s' yield bit clear | FAIL, 34 of 400 | -0.44%, -0.41% per clock (1 leg) |
| hs | after the IMMA; one half from its C cache, the other read by the chain's previous LOP3 before its own IMMA | FAIL, 25 of 387 | |
| hb | before the IMMA, both halves from the register file (two cycles a LOP3, as ptxas) | PASS, 400 of 400 (497 hits) | -0.49% raw (-1.00 to -0.13), +0.76% per clock (3 legs; an RTX 2060 held at 80-84 W, 1231-1278 MHz; twin -0.31%, per clock -0.66%) |

c0 beside hr on the 168 W PC: +1.32% (1 leg); beside hb on the 80 W PC: +0.61% (3 legs, -0.20 to +1.39).

- **Reading a register just after an in-place IMMA is a race** (hr, hrn, hr0): the other warp can hold this one back
  until the IMMA's result lands.
- **The IMMA C reuse cache doesn't hand its value to a LOP3** (hs fails where hb, with every other change, passes).
  cb passed only because its register was not overwritten. So there is no free operand for the readout's LOP3s, and
  in place they stay at two register-file cycles each.
- hb, ptxas's reads in a hand order, is level at best. Nothing ships.

Not done: idea 2 of the brief (L1/L2 hit rates for goal-nv20m's losing builds) needs a profiler, which the Salad boxes
don't have; idea 3 (the hash split over more warps beside a one-thread tile) was not built. No PeakMiner bench this
time (nothing won; goal-nv20k: 98.3% on a 2060). No 2080-class card was tried. Box time: nv20n-2060-1 05:00-07:10
(three PCs; the third throttled at 86 C and 70 W, so the group was deleted), nv20n-2060-2 07:11-09:37 (four PCs:
set-env does not restart a running box; realloc does, and moves it). Tools: `S/nv20n/` in the scratchpad (r1.sh to
r7.sh, mklist.py, hr.py, hr2.py, hrcheck.py, freeregs.py).

### Salad, 14 November: a hash spread over the loop, and what A and B loads cost; nothing ships (branch goal/nv-20p)

One race on an RTX 2060 6 GB in llmjob2 (160 W limit, driver 621.14, 1650 MHz, 137-140 W, 80-82 C). The PC drifted
from 44 to 39.5 TH/s during it and the A/A twin read -1.5% to +2% against the control, so each figure is against the
mean of control and twin (40.16 TH/s), 3 legs each. All but the twin are timing only (the twin: verify-hits 271/271).

| Build | What it is | Against control and twin |
|---|---|---|
| sp2 | 2 BLAKE3 G steps a chunk inside the loop | -4.5% (clock -15 MHz) |
| sp4 | 4 G steps a chunk, what a one-thread tile would need | -6.0% (clock -27 MHz) |
| ho | hash off | +0.3% |
| hd16 | hash off, hasher about one k-step behind its partner | -1.9% |
| hd32 | hash off, about two k-steps behind | -2.8% |
| al | A loaded once a tile | +3.5% (clock +60 MHz; per clock 0%) |
| bl | B loaded once a tile | +5.5% (clock +75 MHz; per clock +1.0%) |

- **A one-thread tile with the hash spread over the loop is out.** It could save at most the shuffle (0-0.27%,
  goal-nv20e) and the hash after the hand-off (+0.3%); spreading the hash alone costs 4.5-6%. More lag between a
  scheduler's two warps only loses.
- **A smaller xor tree is capped at about +0.1%:** any tree over 128 sums takes about 127 register-file cycles (3-input
  LOP3s at 2, or 2-input at 1) against today's 136; at goal-nv20n's price (136 cycles = +1.7%) that is 9 cycles. Not
  built.
- **Fewer B loads within 255 registers: no layout cuts the bytes.** A 64 x 64 warp tile reads 32 bytes from L1 per IMMA
  whatever the layout; fewer bytes need a bigger warp tile (64 x 96: 192 sums and 40 operand registers single-buffered,
  -3.3% in goal-nv20m); B through shared memory adds stores on the same path; a 256 x 128 block only swaps which
  operand misses L1. Not built.
- A's and B's loads together are worth about 9% raw on the 2060, mostly as clock at about the same watts. PeakMiner
  loads the same bytes per IMMA, so this is not plainly where its 2% a clock comes from.

Not done: the RTX 2080 SUPER round (its box failed setup in node-gyp, then the run stopped); no PeakMiner bench. The
switches (`PEARL_BD_HDELAY`, `PEARL_ABLATE_SPREAD=N`, `PEARL_ABLATE_ALOAD`) are on `goal/nv-20p` only, off by default.

### Salad, 15 November: why OneZeroMiner 1.8.6 leads on RTX 20, and our rotation fold on sm_75; nothing ships (branch goal/nv-20q)

**What OneZeroMiner does, seen from outside.** `bench.sh` on the stand-in pool (2^210) with two new measuring options:
`SMI1=1` reads the card once a second through each run, and `ozm1`..`ozm3` run OneZeroMiner with extra flags
(`OZM1_X`..). Shares decoded through `sharetap.py`. Each figure is the run's mean past its first 60 s.

| PC | Miner | TH/s | SM MHz | W | Memory controller | VRAM | Shape (m, n, k, a thread's sums) |
|---|---|---|---|---|---|---|---|
| RTX 2080 `d46a74f1` (225 W limit, SM clock held at 1200 MHz by the host) | ours v0.7.2 | 53.92, 53.78 | 1200 | 118.0, 118.6 | 27% | 1,228 MiB | 131072, 262144, 2048, 16x16 over 4 lanes |
| | OneZeroMiner 1.8.6 | 52.9 (own count, both runs) | 1200 | 100.5, 101.0 | 0% | 630 MiB | 5888 (46 SMs x 128), 49152, 65536, 8x16 |
| | PeakMiner 2.20.8 | 54.3 | 1200 | 113.0 | 19% | 2,394 MiB | |
| RTX 2060 `3fd5a9cf` (168 W, 79 C) | ours v0.7.2 | 47.01 | 1637 | 159.1 | 23% | | |
| | OneZeroMiner 1.8.6 | about 46.4 | 1661 | 155.5 | 0% | | |

- Its log says `kernel 2 (compact)`. At k 65536 its operands would be 386 MB (A) and 3.2 GB (B), yet it holds 400 MB
  over idle and its memory controller reads 0%: it draws the noise on the chip, as our rotation fold does on RTX
  30 to 50.
- At one clock it does 1.9% less work than ours and draws 15% less power. On a power-capped card that becomes
  clock, which is its 1-2% lead on most RTX 20 rows. PeakMiner does 0.7% more work a clock than ours at 4% less power.
- `--power-save 1` and `2` do nothing on Salad ("Power save level 1 unavailable: required controls unavailable
  (PS002)"); `--mclk 810` fails ("Failed to apply locked memory clock: insufficient permission"). So none of its new
  settings works on Salad, and neither would our memory clock lock.
- The 14 November row of the RTX 2080 `6b76def0` (94.2%): our 15 s samples swung 84.7 / 74.7 / 67.5 TH/s there while
  every competitor held steady, so part of that row is the PC.

**Our rotation fold for sm_75** (`goal/nv-20q`, off unless `PEARL_ROT_T75=1` and `PEARL_ROT_CUBIN` names the cubin).
`pearl_rot.cu` built for sm_75 with each m16n8k32 mma as four m8n8k16 on the same fragments, so the tile, the
transcript and the host are the RTX 30/40 ones. Turing's 64 KB holds only the tables, so the first form has every
warp gather its own A (no stage, no barriers). `hashrate.js` 90 s legs taking turns with the release fold; verify-hits
passed on every real build (92 to 195 hits, no failure; the staged builds saw 2 salts in 120 s, so PASS=false on
the salt count only).

| Build | RTX 2060 6 GB (`3fd5a9cf`, 170 W) | RTX 2080 (225 W, 87 C) | RTX 2060 (180 W) |
|---|---|---|---|
| release fold | 43.78, 43.55 at 1537-1541 MHz, 149-152 W (92.5% of the pipe) | 68.37, 67.42 at 1556-1581, 183-190 W (92%); 73.23, 67.80 on a cooler PC | 46.92 x3 at 1680, 180 W |
| 12 x 16 tile (192 sums, 255 registers), three schedules | 33.49-33.60 at 1666-1672, 143 W (65%) | | |
| 8 x 16 tile (128 sums), plain / A a tile ahead / B a turn ahead and readout spread | | 44.28 / 44.33 / 44.22 at 1706-1713, 182 W (55%); 46.0, 44.2 | 31.03 / 31.01 / 30.59 at 1819-1845 |
| 8 x 16, no B gathers (timing only) | | 73.66 at 1769, 173 W (88%) | 50.01 at 1861, 165 W |
| 8 x 16, no A gathers (timing only) | | 66.47 at 1632, 180 W (87%) | 46.49, 46.34 |
| 8 x 16, B's gathers on conflict-free rows (timing only) | | 53.08, 51.88 (67%) | |
| 8 x 16, A's on conflict-free rows (timing only) | | 55.08, 54.25 (70%) | |
| 8 x 16, both conflict-free (timing only) | | 62.10 at 1611 (82%) | |
| 8 x 16 staged (A shared by the band through a 16 KB stage): barrier a k-step / two a chunk / and B ahead / and readout spread | | | 31.83 / 30.07 / 34.07 / 35.07 at 1837-1884, 168 W, against 47.38 at 1689, 166 W |

- The noise drawn on the chip does save power, as on OneZeroMiner: the real builds ran 8-10% higher clocks at the same or lower watts,
  with the memory controller at 0-6%. But the gathers' shared-memory reads cap the pipe: either half alone leaves
  87-88%, both 55%. Bank conflicts are most of it: the host's order search leaves 1.7-1.9 wavefronts a matrix
  (simulated on random rank pairs; 2.6 in the definition's order), close to the floor that the residues' random
  imbalance sets, and with no conflict at all the fold still reaches only 82% against the release's 92%.
- Sharing A through a stage cuts a chunk's LDSM from 64 to 40 but adds barriers: 60-62% at best.
- So this fold does not carry over to Turing (as the 12 November price probe said), and OneZeroMiner's kernel 2 must
  draw its operands some cheaper way, or with fewer reads a MAC, than these gathers.

### Salad, 15 November: what the release fold's DRAM reads cost on RTX 20, and the hybrid fold priced; nothing ships (branch goal/nv-20r)

Two ideas from goal-nv20q, priced on one Salad group in llmjob2 (`nv20r-2080-1`, two PCs). `nv20box.sh`, hashrate.js
legs of 90 s taking turns, each leg paired with the control legs beside it, an A/A twin of the shipped t75 core.
Timing-only builds compute wrong values on purpose. Switches on `goal/nv-20r` only, off by default.

**Idea 1: fewer DRAM reads.** RTX 2080 (215 W limit, driver 595.95; 209-211 W, 77-79 C). The shipped fold reads
about 51 KB of A' from DRAM a tile (A streams past a column band of 5 groups of B held in L2): memory controller 17%.

| Build | What it changes | Paired raw | Per clock | SM MHz | Memory controller |
|---|---|---|---|---|---|
| t75 | shipped | 68.54-69.06 TH/s | | 1587-1600 | 17% |
| twin | nothing | -0.37% (-0.38, -0.36) | +0.01% | | 17% |
| al2 (timing only, `PEARL_ABLATE_AL2=2`) | every tile reads one of 2 row groups' A', so A' never leaves L2 | **+1.08%** (+0.96, +1.20) | -0.42% | +27 | 0% |
| aef (`PEARL_BD_LD_A=7`) | A's loads `.cs` (LDG.E.EF) | -2.12% | -1.44% | | 21% |
| aef, band 6 | | -1.85% | -1.44% | | 18% |
| aef, band 8 | | -4.65% | -2.50% | | 52% |
| alu, band 8 (`PEARL_BD_LD_A=8`) | A's loads `.lu` (LDG.E.LU) | -21.74% | -20.44% | | 74% |
| t75, band 8 | | -2.68% | -0.90% | | 45% |

verify-hits 400/400 (aef), 340/340 (alu), 400/400 (twin).

- **All of the fold's DRAM traffic is worth about 1% on a power-capped 2080**, as clock. That is idea 1's ceiling,
  and it is less than the gap on most RTX 20 rows.
- No real change gets near it. Evict-first A does not keep B in L2 (it loses L1 reuse among the four warps that
  read the same A), and a deeper column band only adds DRAM traffic. The band walk is close to the fewest DRAM bytes
  a 4 MB L2 allows for a 128 x 256 tile: holding B and streaming A, or the other way round, read the same bytes for
  the same L2 budget, and PeakMiner's shape (A resident, B streamed) reads more a tile (about 64 KB).

**Idea 2: the hybrid (A drawn on the chip, B loaded as the release does).** goal-nv20q's "no B gathers" build
(88% of the pipe at +12% clock) left B at zero: no B traffic, and an idle multiplier input. New timing-only form
`ROT_T75 | 65536` (with 2048) loads B the release's way: four 16-byte loads a lane a k-step, one k-step ahead, from
2.6 MB of L2-resident bytes, the two warps of a column slot on the same lines. RTX 2080 SUPER (250 W, driver 621.24;
203-224 W, 55-61 C), 3 legs each:

| Build | TH/s | SM MHz | Memory controller | Paired raw |
|---|---|---|---|---|
| t75 (shipped) | 75.84-76.23 | 1602-1658 | 17% | |
| twin | 75.84-75.92 | 1655-1659 | 17% | -0.16% (-0.24 to -0.09) |
| g: rotation fold 8 x 16, A gathered, B zero (timing only) | 84.46-84.70 | 1890-1946 | 3% | +11.37% |
| gb: the same, B loaded (timing only) | 66.11-66.22 | 1714-1723 | 12% | **-12.85%** |
| r: the real rotation fold 8 x 16 | 49.82-49.88 | 1848-1866 | 1% | -34.34% |

- **The hybrid loses 13%.** With B really loaded, the LDSM gathers of A and the B loads share the SM's load path,
  and the build runs below the release at a similar clock. Most of the "no B gathers" build's lead came from B
  being zero. Idea 3 (each table entry twice, to halve bank conflicts) was to go only with idea 2, so it was not built.

**Same PC, the first 2080** (bench.sh, stand-in pool 2^210, 240 s each after 60 s, means of the 15 s samples from 60
s on): ours (v0.7.2) 69.22 TH/s (linemean 68.97) at 1595 MHz, 209.8 W, memory controller 17%; OneZeroMiner 1.8.6
68.60 at 1611 MHz, 211.0 W, 0%; PeakMiner 2.20.8 68.55 at 1595 MHz, 210.0 W, 28%. **Ours 100.6% of the best** (100.5%
by linemean) on this PC.

Box time: nv20r-2080-1 02:52-04:13, about 1 h 20 min over two PCs. Tools: `S/nv20r/` in the scratchpad (r1.sh,
r2.sh, hrsum.py).

### Salad, 15 November: what a tile with fewer L2 bytes a MAC could win on RTX 20; nothing ships (branch goal/nv-20s)

goal-nv20r left one route: a new tile shape that reads fewer L2 bytes a MAC than 128 x 256, within 255 registers.

**On paper.** An M x N block reads 1/M bytes of B and 1/N of A from L2 a MAC. Its sums are 4-byte registers, so
M x N = 32 x warps x sums a lane, and 8 warps of 255 registers are the whole register file. At 128 sums a lane
M x N = 32768: 128 x 256 reads 3/256 = 0.0117 bytes a MAC, and the best shape of that size (square) 2/181 = 0.0110,
6% fewer. 256 x 128 only swaps A and B. Fewer bytes need more sums a lane: 80 x 64 warp tiles (160 sums, a 160 x 256
block) cut B's bytes by 20%, 96 x 64 (192 sums, 192 x 256) by 33%. The loop now peaks at 234 live registers: 128
sums, 64 operand registers (two k-steps of A and B) and about 42 others. 192 sums with every operand half
single-buffered (40) and the same 42 others is about 274, so it spills, and single-buffering alone cost 0.7 to 4.3%
(goal-nv20m). 160 sums needs most of that too, and an 80-row or 80-column warp tile no longer holds whole 16 x 16
regions.

**The gain side, measured first** (timing only, wrong results on purpose). New switches `PEARL_ABLATE_BHITN=N` and
`_AHITN=N`: k-steps t with (t + 1) % N == 0 re-read the k-step before, so 1/N of B's (or A's) L2 traffic goes, as a
block 1 / (1 - 1/N) times as tall (or wide) would save. `nv20box.sh`, hashrate.js legs of 90 s taking turns, each
paired with the control legs beside it, an A/A twin of the shipped t75 core. Group nv20s-2080-1 in llmjob2.

Round 1, RTX 2080 SUPER (225 W limit, driver 610.60; 212-219 W, 85-86 C, 1625-1680 MHz), 3 paired legs each. Built at
139b44ef, where ptxas merged some of the re-reads with the load before them (it merges equal `ld.global.nc`), so
these also drop some L1 reads: upper bounds.

| Build | What goes | TH/s | SM MHz | Memory controller | Paired raw | Per clock |
|---|---|---|---|---|---|---|
| t75 | nothing (shipped) | 74.08 | 1630 | 18% | | |
| twin | nothing | 73.99 | 1627 | 18% | -0.12% (-0.16 to -0.09) | +0.03% |
| bh4 | 1/4 of B's L2 traffic (a block 171 rows tall) | 74.50 | 1641 | 17% | +0.63% (+0.59 to +0.68) | -0.11% |
| bh2 | 1/2 of B's (256 rows) | 75.38 | 1659 | 17% | +1.82% (+1.72 to +1.88) | -0.05% |
| bh (`PEARL_ABLATE_BHIT`) | all of B's | 76.47 | 1676 | 17% | +3.29% (+3.15 to +3.43) | +0.34% |
| ah2 | 1/2 of A's (512 columns) | 74.94 | 1657 | 9% | +1.22% (+1.17 to +1.29) | -0.54% |

verify-hits 400/400 (twin).

Round 2 (1399794d) kept every load: a re-read starts 16 bytes before the k-step it repeats, so ptxas can't merge it.
It also asked one more thing: `PEARL_ABLATE_ALDS` loads A from the transcripts' shared memory instead of global, at
the same offsets (no A traffic to L2 or DRAM, and a shared load in place of an L1 hit). Salad's realloc gave two
PCs with two GPUs each (left out: the core ran on the card nvidia-smi does not read first), then an RTX 2080
(195.65 W limit, driver 621.24; 191-194 W, 52 C, 1566-1581 MHz). Legs pinned to that card with
`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0`. In the third pass every build but the twin fell 1-2%, so the
first two passes are given beside all three.

| Build | Paired raw, 3 passes | First two passes | Per clock, 3 passes | Memory controller |
|---|---|---|---|---|
| t75 | 69.38 TH/s | 69.52, 69.38 | | 17% |
| twin | -0.16% (-0.27 to -0.10) | -0.1% | -0.01% | 17% |
| bh4 | -0.23% (-0.90 to +0.13) | +0.1% | -0.47% | 16% |
| bh2 | -0.26% (-0.87 to +0.07) | +0.0% | -1.00% | 16% |
| ah2 | -0.71% (-1.52 to -0.30) | -0.4% | -1.24% | 10% |
| alds | +0.28% (-1.16 to +1.04) | +0.9 to +1.0% | +0.01% | 0% |

- **A taller tile can't pay.** Round 1 (re-reads merged, so fewer L1 reads too) puts B's L2 traffic at about 0.07%
  of the rate per percent of it: a 192 x 256 block (B -33%) at most about +1.1%, a 160 x 256 block about +0.6%, all
  as clock. With every load kept (round 2) the same cuts are level or worse: the 16-byte offset costs each re-read
  a fifth L1 line, and that costs about what the L2 bytes saved. So the true worth is between the two, at most about
  1%, before the registers a taller tile needs. Not built.
- **A through shared memory can't pay either.** alds, with no A traffic past the SM at all, is worth +0.9 to +1.0%
  (two clean passes). A staged A would keep A's L2 reads and add shared stores and barriers, so it would get less.
- All of A's and B's L2 traffic together is worth about 5% (round 1: all of B's +3.3%, half of A's +1.2%), and no
  block that fits the register file reaches more than a fifth of it. What is left above 2% is drawing the operands
  on the chip, as OneZeroMiner does; on Turing that lost to shared-memory gathers (goal-nv20q, goal-nv20r).

The switches (`PEARL_ABLATE_BHITN`, `_AHITN`, `_AHIT4` (it spills, so it was not raced) and `_ALDS`) are on
goal/nv-20s only, off by default. Box time: nv20s-2080-1 04:26-06:03 over four PCs, about 1 h 40 min. Tools:
`S/nv20s/` in the scratchpad (r1.sh, r2.sh, hrsum.py).

### Salad, 15 November: what drawing the operands on the chip costs Turing's load path; nothing ships (branch goal/nv-20t)

goal-nv20s left one route above 2%: draw the operands on the chip, as OneZeroMiner does, with fewer shared-memory
reads a MAC than goal-nv20q's gathers. This round counted each scheme on paper, then priced the steps and the
schemes with `price75.cu` (timing only, wrong sums on purpose: the release's geometry, 64x64 warp tiles, 128
IMMA.8816 a warp a k-step, no readout and no hash). New streams and offsets options are on goal/nv-20t.

**On paper**, in 128-byte wavefronts a warp a k-step (128 IMMA):

| Scheme | Wavefronts |
|---|---|
| release: 8 loads of 16 bytes | 32 |
| 8 x 16 rotation fold: 16 ldmatrix gathers, c collisions (1.7-1.9 after the order search) | 64c, about 115 |
| 12 x 16 rotation fold | 53c |
| hybrid, A drawn by each warp, B loaded (goal-nv20r) | 32c + 16 |
| hybrid, B drawn by each warp, A loaded | 32c + 16 |
| both drawn once a block into a stage, ldmatrix-ready | 32 + 24 reads + 12 stores = 68, and a barrier |

Every draw needs 2 table bytes for each operand byte (the plus and the minus entry), and the registers still have to
be filled, so no scheme gets under the release's 32. It can only win on power, by taking the L2 and DRAM traffic off.
An order search (`cfsim.py`) reaches about 1.5 collisions a matrix with one gathered operand and 1.9 with two.

**Round 1, the price of each step** (nv20t-2080-1, llmjob2, RTX 2080, 215 W, driver 595.95, 75-81 C, every run at
the power cap; 30 s each). "Count" is the rate the stream would mine; the release mines about 69 TH/s with readout
and hash on 2080s like this (goal-nv20r).

| Stream | Pipe | MHz | Count |
|---|---|---|---|
| operands in registers | 99.5% | 1744 | 81.7 |
| 8 ldmatrix of ready fragments (sm8) | 99.3% | 1688 | 79.0 |
| sm8 + 8 / 16 ld.shared of 16 bytes | 91.7% / 65.9% | 1647 / 1696 | 71.1 / 52.7 |
| sm8 + 8 ldmatrix, ready / .trans ready | 90.3% / 90.3% | 1650 | 70.2 |
| sm8 + 8 ldmatrix.trans, rows spread / random rows | 90.7% / 54.3% | 1614 / 1763 | 69.0 / 45.1 |
| sm8 + 16 / 32 ld.shared of 4 bytes | 95.1% / 96.2% | 1640 | 73.7 / 74.2 |
| sm8 + 16 ld.shared of 8 bytes | 91.1% | 1629 | 69.9 |
| sm8 + 16 ld.shared of 16 bytes, one address a warp | 90.9% | 1634 | 70.0 |
| sm8 + 16 / 32 shfl | 95.9% / 85.5% | 1625 | 73.3 / 65.5 |
| sm8 + 8 / 16 texture fetches of 16 bytes | 77.5% / 42.9% | 1618 | 59.1 / 32.6 |
| sm8 + 8 global loads from L1 | 88.9% | 1592 | 66.7 |
| both gathered by each warp, rows spread / random rows | 85.5% / 37.8% | 1605 / 1816 | 64.7 / 32.3 |
| both drawn once a block into a stage, block barrier a k-step | 72.6% | 1695 | 58.0 |
| the same with no barrier (would be wrong) | 88.7% | 1628 | 68.0 |
| B alone drawn into a stage, barrier | 84.6% | 1660 | 66.1 |

- Turing's load path takes about 64 bytes a clock an SM whatever the instruction. 4-byte loads are a little
  cheaper per byte; broadcast loads, shuffles and texture fetches are not cheaper. So there is no side path for a
  gather.
- Collisions cost far more than their count: random rows on 8 ldmatrix take the pipe from 91% to 54%.
- A stage needs a block barrier every k-step, which costs 16 points: 48 KB of tables leave 16 KB, too little for a
  deeper stage. With no barrier at all it is still below the release.

**Round 2, B drawn by each warp against B loaded, A the same** (nv20t-2080-2, RTX 2080, 225 W, driver 621.14). The
first group never restarted after set-env and was replaced. The card heated from 77 to 87 C during the run and its
watts fell from 220 to 180, so compare neighbours, and the pipe (work a clock). "pair" puts two rows of every other
matrix in one bank group (1.5 collisions a matrix, about what the order search reaches).

| A from | B loaded | B gathered, no collisions | B gathered, 1.5 | B gathered, random rows |
|---|---|---|---|---|
| ready fragments in shared | 99.0%: 76.5, 73.0 | 95.8%: 74.6, 72.2 | 91.8%: 71.8, 69.4 | |
| registers | 94.5%: 72.6, 70.5 | 91.1%: 70.7, 69.0 | 93.2%: 71.9 | |
| global (the probe's `rel`, 255 registers, badly scheduled) | 67.3%: 56.1, 51.8 | 94.2%: 72.6, 68.9 | 91.9%: 71.4, 66.2 | 63.0%: 51.6 |

- With A held the same, gathering B with no collisions at all loses 1-3% against loading it (76.5 -> 74.6 and
  73.0 -> 72.2), and at the 1.5 collisions an order search can reach it loses 5-6%. The gather gains 1-2% clock at
  the cap, not enough to pay for the pipe.
- The `rel` row is not a fair base: its stream spills at 255 registers and ptxas schedules it badly (as goal-nv20
  found); `gathB` has 236 and schedules well.

**What this says.** On Turing no on-chip draw tried here beats loading the operands: not each warp drawing both
(85.5% even with no collisions), not a shared stage (barriers), not the hybrid either way round (goal-nv20r for A
drawn, round 2 for B drawn). The load path, at about 64 bytes a clock, is the limit, and drawing needs twice the
operand bytes through it. OneZeroMiner's power at one clock (15% under ours, goal-nv20q) does not come from a draw
like these: our draw streams ran no higher clock than the release at the cap. Not found yet: what it does instead.

Box time: nv20t-2080-1 06:22-08:10 (idle from 06:37 after set-env failed to restart it), nv20t-2080-2 08:10-08:25.
Tools: `S/nv20t/` in the scratchpad (r1.sh, cfsim.py, ed1.py, ed2.py).

### Salad, 15 November: narrow global loads for the RTX 20 fold; nothing ships (branch goal/nv-20u)

goal-nv20t saw 4-byte shared loads cost Turing's load path a little less per byte than 16-byte ones. This round
asked whether the same holds for the fold's global loads. `PEARL_BD_LDW` (goal/nv-20u, off by default) loads A's or
B's fragments in 8- or 4-byte pieces, with A' and B' written word-major so the narrow loads stay coalesced.

RTX 2080 `6b76def0` (the 94.2% row of the 14 November table; 225 W limit, driver 617.42; 220-223 W, 70-71 C).
hashrate.js 90 s legs, one leg each, taking turns with the shipped t75 core and its A/A twin.

| Build | What it changes | Instructions a chunk | TH/s | SM MHz | Against t75 |
|---|---|---|---|---|---|
| t75 | shipped | | 76.37 | 1738 | |
| t75b | nothing (twin) | | 76.38 | 1741 | +0.01% |
| w4b (`PEARL_BD_LDW=8`) | B in 4-byte loads | 687 | 72.98 | 1724 | **-4.4%** |
| w2b (`PEARL_BD_LDW=4`) | B in 8-byte loads | 655 | 75.15 | 1735 | **-1.6%** |

verify-hits 400/400 for w4ab (A and B narrow, 442 hits) and w2b (460 hits): the word-major A' and B' are right.
w4a and w4ab were not raced before the box was deleted.

- Turing's global path costs per instruction, not per byte. The 4-byte saving goal-nv20t saw is shared memory only.
  Narrow global loads are closed.
- REG2 does not carry over to the RTX 20 fold: ROWPAIR already folds the two-lane regions with one shuffle.
- The readout is within about 9 register-file cycles a chunk of its floor (127 against 136), about 0.1% of the
  rate, as goal-nv20p computed.

Box time: nv20u-2080-1, 08:37-09:00. Tools: `S/nv20u/` in the scratchpad (r1.sh, salad.py, hrsum.py).

### Salad, 15 November: B passed between a pair's warps through a shared ring, priced; nothing ships (branch goal/nv-20v)

The idea: the two warps of a column slot run on one scheduler and use the same B. Let row slot 0's warp load B
from global and store it into a ring in shared memory, and row slot 1's warp read it there with 4-byte shared
loads, which goal-nv20t found the cheapest step on Turing's load path. A ring of 6 k-steps a pair is 48 KB, so with
the transcripts' 16 KB it takes all 64 KB. Priced first in `price75.cu` (timing only, wrong sums on purpose), as new
B sources: W stores word-major with 16 `st.shared` of 4 bytes and reads with 16 `ld.shared` of 4 bytes (no bank
conflicts); Q stores and reads 4 x 16 bytes in fragment order; F adds flow control (each warp publishes its k-step
count in shared memory and waits on its partner's: the writer at most 6 k-steps ahead, the reader behind it).

nv20v-2080-1 (llmjob2): RTX 2080 SUPER, 250 W, driver 581.80, 52-68 C, at the power cap. nv20v-2080-2: RTX 2080,
225 W, driver 621.14, 77-87 C, held by heat (pipe figures the same to 0.0002, counts lower). 30 s each, every
stream two to four times; pipe is the tensor pipe's share, the same within 0.0002 on every repeat.

| Stream | Pipe | Count (2080 SUPER) | SM MHz |
|---|---|---|---|
| A ready in shared, B loaded by both warps (smArB) | 99.0% | 88.4, 85.9, 85.9 | 1765-1816 |
| the same, B through the ring, W | 84.8% | 77.9 | 1869 |
| the same, Q | 89.6% | 81.0 | 1840 |
| the same, W with flow control | 79.7% | 72.4 | 1846 |
| the same, Q with flow control | 72.4% | 66.8 | 1877 |
| A in registers, B loaded by both warps (relB) | 94.5% | 83.6, 83.3, 83.6 | 1793-1802 |
| the same, B through the ring, W with flow control | 89.9% | 80.3, 80.4 | 1818-1820 |
| A and B loaded (the probe's `rel`: spills, badly scheduled, not a fair base) | 65.3-66.9% | 58.8-58.9 | 1829-1835 |
| A loaded, B through the ring, W / Q with flow control | 73.9% / 73.7-74.0% | 65.3 / 65.6-65.8 | 1797-1816 |

- **The ring loses 5 to 27 points of pipe**, more than the twin spread by a wide margin (repeats agree to 0.02
  points), so it was not built in the fold. Even with no flow control it is 9-14 points down; the flow control's
  volatile load, fence and store each k-step take 5-17 more. The clock gain at the cap is 1-6%, far short.
- Why: the second warp's B loads are L1 hits that cost the load path little when A comes from shared (smArB is
  within 0.3 points of 8 ldmatrix alone), while the ring adds the writer's stores to the same path, and the reader
  waits on the writer. Loading B once a pair saves an L1 read, not an L2 one, so it has no power to win either.

**Round 2: is OneZeroMiner's lower power at a clock its missing memory traffic?** goal-nv20q found OneZeroMiner
1.8.6 draws about 15% less power than ours at one clock on RTX 20, with its memory controller at 0%, and no gather
explained it. Our shipped core with its memory traffic switched off (timing only, wrong results on purpose; built at
goal/nv-20s 04e4e2af, which has the switches): `bhit` makes every B load after the first two an L1 hit (no B
traffic past the SM), `alds` loads A from the transcripts' shared memory (no A traffic), `nomem` both. hashrate.js
legs of 90 s, three of each, taking turns with the shipped t75 core and its twin; then the three miners on the
stand-in pool (2^210, 240 s each after 60 s, means of the 15 s samples from 60 s on, the card read once a second).
Legs pinned to nvidia-smi's first card.

| Build | RTX 2080 SUPER (250 W, driver 581.80, 68 C) | RTX 2080 (215 W, driver 617.14, 83 C) |
|---|---|---|
| t75 (shipped) and twin, 6 legs | 82.16 TH/s (82.03-82.57) at 1764-1775 MHz, memory controller 19% | 68.09 (67.94-68.56) at 1549-1565 MHz, 17% |
| twin against t75 | -0.24% | -0.29% |
| bhit (no B traffic) | 85.18, **+3.7%**, 1829-1830 MHz (+3.3%) | 71.13, **+4.5%**, 1618-1619 MHz (+4.1%) |
| alds (no A traffic) | 82.96, **+1.0%**, 1782-1785 MHz (+0.8%), 0% | 69.30, **+1.8%**, 1581-1585 MHz (+1.8%), 1% |
| nomem (neither) | 87.12, **+6.0%**, 1856-1858 MHz (+4.9%), 0% | 73.21, **+7.5%**, 1649-1653 MHz (+6.1%), 0% |

All at the power cap (249 W and 211-213 W). Each build's three legs agree within 0.1 TH/s.

The three miners on the same PCs, right after (displayed rate; "credit" is what the stand-in pool credited, about
250 shares a run, so about ±6%):

| PC | Miner | TH/s | SM MHz | W | Memory controller | Credit |
|---|---|---|---|---|---|---|
| RTX 2080 SUPER | ours v0.7.2 | 82.47, 82.99 | 1762-1765 | 245-246 | 19% | 78.84, 79.42 (270, 272 shares) |
| | OneZeroMiner 1.8.6 | 82.01, 81.99 | 1791-1794 | 248 | 0% | 73.89, 70.37 |
| | PeakMiner 2.20.8 | 82.54 | 1779 | 249 | 21% | 68.61 |
| RTX 2080 | ours v0.7.2 | 68.02, 67.75 | 1546-1555 | 210 | 17% | 63.36, 64.53 |
| | OneZeroMiner 1.8.6 | 68.46, 68.43 | 1585-1592 | 210 | 0% | 58.74, 57.52 |
| | PeakMiner 2.20.8 | 68.77 | 1544 | 212 | 28% | 63.65 |

Ours is 100.2% of the best by displayed rate on the 2080 SUPER and 98.7% on the 2080; by credit it leads on the
2080 SUPER and ties PeakMiner on the 2080.

- **OneZeroMiner's power edge comes from its missing memory traffic.** At the cap it runs 1.7% (2080 SUPER) and 2.5%
  (2080) higher clocks than ours, with the memory controller at 0%. Our own core with no L2 or DRAM traffic runs 4.9%
  and 6.1% higher than ours (5.2% and 6.5% above ours in the bench), higher than OneZeroMiner. So taking the
  traffic off is worth more clock than OneZeroMiner gets; its on-chip draw spends part of it back, and it does about
  2% less work a clock (goal-nv20q). Nothing else needs explaining.
- **B's traffic is most of it** (+3.7% and +4.5% of the rate, against +1.0% and +1.8% for A's), as goal-nv20s found.
  The whole of it, +6.0 to +7.5%, is the most any fold that keeps today's arithmetic could win on these cards, and
  only if it removed the traffic for free. Every way of removing it priced so far costs more on Turing's load path
  than it saves: B gathered by each warp (goal-nv20t: -1 to -2.5% with no bank conflicts at all, -5 to -6% at what
  an order search reaches), shared stages (block barriers), and this round's ring (which saves only L1 reads).

Box time: nv20v-2080-1 09:05-10:16 (2080 SUPER), nv20v-2080-2 09:06-09:16, nv20v-2080-3 09:17-09:24 (an RTX 2070 at
88 C under hardware slowdown, 947 MHz: replaced), nv20v-2080-4 09:25-10:17, about 2 h 20 min in all. Tools:
`S/nv20v/` in the scratchpad (p1.sh, r2.sh, bsum.py).

### Salad, 19 November: SRBMiner 3.8.9 on an RTX 2080 Ti and a 2070 SUPER, and band, stream and ptxas variants; nothing ships (branch goal/rtx20b)

**Profiling.** Nsight Compute can't read counters on Salad (ERR_NVGPUCTRPERM: the hosts are Windows under WSL2), and
Nsight Systems traced no SRBMiner kernels (lead session, g18-rtx20-2080ti-3). So the comparison below is from the
outside: each miner's shown rate and the card read once a second. `nv20box.sh` at 1c84c032, stand-in pool at 2^210,
180 s a turn after a 60 s warm-up; ours is the v0.7.5 release (its 12.8 core, the t75 fold), shown as `linemean`;
the others are the means of their 15 s samples from 60 s on.

| PC | Miner | Shown TH/s | SM MHz | W | Memory controller | TH/s per GHz |
|---|---|---|---|---|---|---|
| RTX 2080 Ti, 260 W, driver 621.24, 53 C | SRBMiner 3.8.9 | 103.79, 103.78 | 1617 | 257.5 | 0% | 64.2 |
| | ours | 102.50, 102.47 | 1581 | 258.5 | 11% | 64.8 |
| RTX 2070 SUPER, 235 W, driver 621.24, 80 C | PeakMiner 2.20.9 | 66.11, 66.06 | 1734 | 228.4 | 23-26% | 38.1 |
| | SRBMiner 3.8.9 | 65.74, 65.86 | 1766 | 227.5 | 1% | 37.3 |
| | ours | 64.68, 64.67 | 1737 | 227.0 | 16% | 37.2 |

- **2080 Ti: ours 98.8% of SRBMiner 3.8.9.** SRBMiner does 1.0% less work a clock than ours but runs 36 MHz (2.3%)
  faster at the same watts, with the memory controller at 0%. 3.8.9 now draws its operands on the chip on Turing,
  or keeps them in L2; either way its DRAM watts go to the clock.
- **2070 SUPER: ours 97.9% of PeakMiner, 98.3% of SRBMiner.** SRBMiner's shape is the same as on the 2080 Ti
  (level per clock, +1.7% clock, no memory traffic). PeakMiner runs at our clock with more memory traffic, and does
  2.3% more work a clock.
- **What our DRAM traffic costs**, from the band races below: on the 2080 Ti band 10 put the memory controller at
  27% against 11% and lost 1.5%; on the 2070 SUPER band 7 put it at 33% against 16% and lost 1.0%, all of it as clock
  (per clock level). About 0.06-0.1% a point, so our 11-16% is worth 1.0-1.5%, about half of SRBMiner's clock lead.
  goal-nv20r's al2 ablation (A never leaves L2) said the same on a 2080 (+1.1%).
- **Our core already loads both operands straight from global into registers** (A-direct and B-direct; shared
  holds only the transcripts), so the RDNA 3 fix (B' to registers without an LDS stage) is the shipped form here.

**Variants raced** (hashrate.js legs of 90 s taking turns with the shipped t75 core and its A/A twin, three of each,
the first leg left out; "paired" is each leg against the control legs beside it). verify-hits passed on every
variant (388 to 400 of 388 to 400 verified, no failure).

| Build | RTX 2080 Ti, paired raw (per clock) | RTX 2070 SUPER, paired raw (per clock) | Memory controller |
|---|---|---|---|
| twin | -0.07% (-0.14%) | -0.34% (-0.22%), -0.82 to -0.08 | same |
| band 6 (`-DPEARL_BD_BAND=6`) | -0.49% (+0.22%) | +0.12% (+0.27%), -0.07 to +0.40 | 2080 Ti 15% |
| band 7 | -0.20% (+0.15%) | -0.97% (+0.30%) | 13% / 33% |
| band 10 | -1.53% (+0.23%) | | 27% |
| one stream (`-DPEARL_FOLD_STREAMS=0`) | +0.09% (+0.12%), +0.06 to +0.12 | | 11% |
| t75 object at ptxas register-usage level 6 | | +0.12% (+0.18%) | 16% |
| level 10 | | +0.06% (+0.18%) | 16% |

- The shipped bands (8 on a 5.5 MB L2, 5 on 4 MB) are the best of these on both cards.
- One stream is level on the mean (+0.09% against a twin spread of 0.07). Its 1 s lines are steady (sd 0.37 TH/s
  against 11.8 with two streams) but the mean is what's shown.
- Levels 6 and 10 give the same 253 registers as 8 and race level with it.
- The 60 s hit check before the race failed with "no result" on every RTX 2080 Ti because it also checked the CUDA 13
  core, which has no compute 7.5 code and which the app never loads on RTX 20 ("no kernel image is available"). The
  12.8 core passed 400 of 400. bench.sh now skips the CUDA 13 core there (1c84c032).

Next, from this: the gap on both cards is clock bought with DRAM watts (SRBMiner) or work a clock (PeakMiner on the
2070 SUPER). The only route to SRBMiner's clock is fewer DRAM bytes, and the L2 bands can't give them; goal-nv20q
and nv20t found no on-chip draw on Turing that keeps the pipe fed.

Box time: g18-rtx20-2080ti-4 02:20-03:15, g18-rtx20-2070-1 03:17-04:16 (a 2070 SUPER), about $0.30 in all.
Tools: `g20b/` in the rtx20 scratchpad (r2.sh, r3.sh, poll.sh, hrsum.py).

**Round 3: the fold's DRAM traffic and its tile walk** (g18-rtx20-2080ti-5, the same 260 W 2080 Ti as above).

The walk is already the one that reads the fewest DRAM bytes for loaded operands. A band of column groups of B
(8 on a 5.5 MB L2, 512 KB a 256-column group at k 2048, so 4 MB) stays in L2 while every row group's A
streams past it, and a band's tiles go out row group by row group, so the 8 blocks on one row group's A run at the
same time and DRAM serves each A line about once a band. That is 1/(256 x band) bytes a MAC. Grouping or swizzling the
tile order can't go below it: any 2-D blocking that fits the L2 (A rows R and B columns C with (R + C) x k at most
the L2) reads (R + C)/(R x C) a MAC, three times as much at R = C. Only a deeper band, a smaller k, or operands
made on the chip cut it.

The memory controller follows that sum: 11, 13, 15% at bands 8, 7, 6 (8/7 and 8/6 of 11 are 12.6 and 14.7), so
nothing else in the fold reads DRAM to speak of. At band 9 the L2 overflows:

| Build | Legs (TH/s) | SM MHz | Memory controller | Against t75 |
|---|---|---|---|---|
| t75 (shipped, band 8) | 103.01 (first leg, left out), 102.59 | 1580-1585 | 12% | |
| twin | 102.68, 102.62 | 1579-1582 | 12% | level |
| band 9 (`-DPEARL_BD_BAND=9`) | 102.05, 101.96 | 1570 | 16% | -0.6% |

verify-hits 400 of 400 on band 9. So band 8 is the L2's edge; a deeper band only adds traffic. A smaller k halves
the traffic per halving but doubles the readout's share per MAC (the readout is worth 2.7% at k 2048; k 4096 lost
7.6% the other way, goal-nv20e). What's left of the DRAM traffic is worth 1.0-1.5% as clock, and only an operand
drawn on the chip removes it, which on Turing has so far cost more pipe than it saves (goal-nv20q, nv20t).
Box time: g18-rtx20-2080ti-5 04:20-04:34, about $0.05.

**Round 4: one operand made on the chip, priced** (g18-rtx20-2080ti-6, an RTX 2080 Ti, 250 W limit, driver 621.24,
73 C rising to 87 C; `price75.cu` at the release fold's geometry, timing only, no readout or hash; each run's clock and
watts from nvidia-smi beside it; "count" is SMs x 1024 MACs x clock x pipe).

Why it has to be a gather. The noise of A is A'[r][kk] = fill + E_AL[r][p0(kk)] - E_AL[r][p1(kk)]: p0 and p1 are the
same for every row, but they index a 128-byte table of the row's own. A lane's A fragments cover 8 rows of a warp
tile, so making them in registers would need 1 KB of tables a lane (256 registers, on top of the 128 sums), and a GPU
can't index registers by a value anyway. So the tables go in shared memory and each fragment is gathered from them
(two ldmatrix and eight ALU steps a 16-byte fragment), which is what `gathA` and `rotrel` do. Making B on the chip
instead would save no DRAM: the fold's DRAM traffic is A's (round 3); B stays in the L2.

| Stream | Pipe | SM MHz | W | Memory controller | Count |
|---|---|---|---|---|---|
| operands in registers (ceiling) | 97.9% | 1613 | 244 | 1% | 110.0 |
| B' loaded as the release does, A free in registers (the most any A-on-chip fold can reach) | 91.8%, 91.7% | 1442, 1384 | 230, 234 | 1% | 92.1, 88.4 |
| **A gathered every k-step, B' loaded** (`gathA`) | 89.1%, 88.9% | 1376, 1332 | 229, 232 | 1% | 85.3, 82.5 |
| A made the rotation fold's way (gathered at one turn in four, shared through a stage), B' loaded (`rotrel`) | 83.9%, 83.9% | 1438, 1403 | 231, 228 | 1% | 84.0, 81.9 |
| B gathered every k-step, A free (`rgAgB`) | 88.9% | 1419 | 236 | 1% | 87.8 |
| the release core mined (hashrate.js, with readout and hash) | | 1367-1368 | 234 | 11% | 86.63, 86.54 |

- **It loses.** The A-on-chip stream alone, with no readout and no hash, counts 82.5-85.3 TH/s; the release mines
  86.6 on the same PC with both. The readout and hash would cost it another 2-4%.
- **The clock goes down, not up.** Taking A's DRAM reads away (memory controller 1%) should buy clock, but the
  gathers cost more watts than the DRAM reads did: `gathA` ran 50-66 MHz below `relB` at the same 230 W, and lost
  2.8 points of pipe to the shared-memory reads on top. SRBMiner's 0% memory controller with a higher clock means
  its draw is cheaper than any gather tried here; how is not known from outside.
- Nothing was built into the fold: by the brief's rule (stop if the stream can't feed the pipe), the stream lost
  before any mining code existed.

Box time: g18-rtx20-2080ti-6 04:38-04:50, about $0.04.

**Round 5: the config each miner declares on an RTX 2080 Ti** (g18-rtx20-2080ti-13, 260 W, driver 621.14, 52-61 C).
`bench.sh` with TAP=1 at 1c84c032: every miner mined the stand-in pool (2^210) through `sharetap.py`, which decoded
each share's plain proof. 120 s a run after a 30 s warm-up, SRBMiner twice, then ours (v0.7.5), then PeakMiner.
Every share verified (678 of 678).

| Miner | m | n | k | Region (rows x columns) | Region rows | Region columns | Fill | VRAM over idle | Memory controller |
|---|---|---|---|---|---|---|---|---|---|
| SRBMiner 3.8.9 | 4354 | 32768 | **65536** | 8 x 16 | +0, 1, 8, 9, 16, 17, 24, 25 | every 8th of 128 | 64 | about 410 MiB | 2% |
| PeakMiner 2.20.9 | 1024 | 245760 | 2048 | 8 x 16 | +0, 8, 32, 40, 64, 72, 96, 104 | pairs every 32, over 256 | 64 | about 2,210 MiB | 24% |
| ours v0.7.5 | 131072 | 262144 | 2048 | 16 x 16 | +0-3, 8-11, 16-19, 24-27 | pairs every 8, over 64 | 48 | about 1,040 MiB | 13% |

Shown rates (means of the 15 s samples from 60 s on; ours is `linemean`): SRBMiner 101.3 and 100.3 TH/s at 1631 and
1620 MHz, ours 98.5 at 1571, PeakMiner 96.8 at 1571, all at 257 W. Two minutes a run is too short to rank the
miners; these runs were for the decode.

- **PeakMiner's tile is ours.** A thread of an m16n8k32 accumulator holds rows g and g + 8 and columns 2t and 2t + 1
  of each 16 x 8 block. PeakMiner's region is exactly one thread's: four row blocks 32 apart and eight column blocks
  32 apart. So each warp covers 64 rows and 64 columns, interleaved with its neighbours, over a block 128 rows by 256
  columns: eight 64 x 64 warp tiles, as ours. Its shape is the same as on 10 November (2.20.1 to 2.20.7). Every part
  of it that shares can show has been raced on our fold and lost: m 1024 with A held in L2 (-1.9% on a 2070 SUPER,
  -1.8% on a 2060, 10 November) and a region of one thread's sums (-2.2%, goal-nv20k).
- **SRBMiner 3.8.9 declares k 65536**, 32 times ours. The transcript is 16 words whatever k is (chunk c xors into
  word c mod 16), so it hashes 32 times less often a MAC. Its operands, as declared, would take 2.4 GB (A' 285 MB,
  B' 2.1 GB), but it holds about 410 MiB and reads almost nothing from DRAM. So it does not store B', and very likely
  not A' either. It draws them on the chip from the noise tables (E_AL is m x 128 bytes, 557 KB here; E_BR n x 128,
  4 MB), and at k 65536 nothing else would fit. Round 4's gathers cost us more pipe than this saves, so its draw is
  a cheaper one; what it is cannot be read from the shares.
- Its regions are rows in adjacent pairs (+0, 1, 8, 9, ...) and columns every 8th: the transposed accumulator, so it
  very likely computes B' times A'^T.

Box time: g18-rtx20-2080ti-10 06:59-07:27 (three PCs, each stalled after boot: apt-get update hung in the boot
command), g18-rtx20-2080ti-11 and -12 07:27-07:41 (apt timed out at 240 s), g18-rtx20-2080ti-13 07:41-08:02 (apt
over IPv4 worked). About $0.21 in all.

**Round 6: does our loop cost more a clock on the smaller cards?** PeakMiner does as much work a clock as ours on
the 2080 Tis and 2.0-2.6% more on 2070s, 2070 SUPERs and 2060s (10 November, goal-nv20k, round 1 above). On
12 November (goal-nv20e, an older core) B's loads cost us +1.9% a clock on a 2070 SUPER and +4.0% on a 2060, which
looked like the same pattern. So the same timing-only builds were raced on both chips, on the same build and the
same day: `nv20box.sh` product stage, hashrate.js legs of 90 s taking turns with t75 and its A/A twin, 15 legs,
the first left out. "Per clock" is the TH/s per GHz change, paired with the control legs beside each leg.

| Switched off | RTX 2080 Ti (240 W, 596.21; 1502 MHz at 235 W), raw / per clock | RTX 2070 SUPER (215 W, 616.92; 1748 MHz at 213 W), raw / per clock |
|---|---|---|
| twin | -0.90% / -0.27% | -0.56% / -0.03% |
| B loaded once a tile (`PEARL_ABLATE_BLOAD`) | +6.51% / +0.70% | +4.35% / +0.66% |
| Every B load an L1 hit (`_BHIT`) | +5.59% / +0.34% | +3.48% / +0.35% |
| The readout (`_READOUT`) | +2.32% / +1.97% | +2.39% / +2.27% |

The shipped core ran 64.1 TH/s per GHz on the 2080 Ti and 38.2 on the 2070 SUPER.

- **No.** On today's core, B's loads cost the same a clock on both chips (0.7% and 0.35%), and the readout about the
  same (2.0% and 2.3%). The 12 November figures were from the core before A-direct. So nothing in our loop costs
  more on the smaller chips, and the loop can't explain why PeakMiner leads only there.
- B's traffic is still worth 4-6% of the rate, all of it as clock at the power limit, as before.
- What's left is outside the loop or in how the rate is shown. On this 2070 SUPER the core ran 38.2 TH/s per GHz;
  round 1's 2070 SUPER (another PC) showed 37.2 for our CLI and 38.1 for PeakMiner. Round 5's CLI lost 1.2 s of
  120 to dips (`linemean ... dips=15 lost_s=1.17`). Next is to time the CLI and hashrate.js on one PC, back to back.
  If the CLI shows about 2% less than the core, the per-clock gap is the redraws and launch gaps between salts, not
  the fold.

Box time: g18-rtx20-2080ti-14 08:06-08:40, g18-rtx20-2070-2 08:39-09:23. About $0.23.

### Monero on the CPU, 11 October (our CPU miner against SRBMiner 3.7.5)

| CPU (CPUs the container got) | GPU | Threads | Ours | SRBMiner 3.7.5 | % of best |
|---|---|---|---|---|---|
| Core i5-11400 (11.5 of 12) | RTX 3060 | 6 | 2,457 H/s | 1,880 H/s | 131% |
| Ryzen 9 5950X (15.4 of 32) | RTX 2060 | 14 | 6,439 H/s | 4,527 H/s | 142% |

Ours used transparent huge pages and SRBMiner got none, which likely explains much of the gap.

## RTX 30-series

Vast, 10 October.

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

SRBMiner exited at once on the Oman host, so that row compares with PeakMiner.

### Salad, 28 October (PeakMiner 2.20.3, SRBMiner 3.8.0)

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

We do 94.9–96.7% of their work per clock and run a 1.6–4.6% lower clock at the same power.

### Salad, 9–10 November (v0.6.13; PeakMiner 2.20.7, SRBMiner 3.8.2, OneZeroMiner 1.8.5)

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | OneZeroMiner | % of best |
|---|---|---|---|---|---|---|---|
| RTX 3090 Ti | Salad (400 W) | 9f31aaad | 136.4 | 142.3 | 143.1 | 144.1 | 95% |
| RTX 3090 Ti | Salad (450 W) | 23132fc1 | 128.4 | 147.6 | 147.2 | 151.9 | 85% |
| RTX 3090 | Salad (315 W) | 36cab5e7 | 112.5 | 118.1 | 118.0 | 117.5 | 95% |
| RTX 3090 | Salad (319 W) | 219edba8 | 113.2 | 121.9 | 119.9 | 121.5 | 93% |
| RTX 3080 Ti | Salad (350 W) | c09c68c3 | 111.2 | 129.6 | 116.0 | 124.2 | 86% |
| RTX 3080 Ti | Salad (350 W) | a4519d9b | 116.0 | 121.3 | 122.0 | 122.1 | 95% |
| RTX 3080 | Salad (309 W) | 121e592d | 113.2 | 121.8 | 122.0 | 122.3 | 93% |
| RTX 3080 | Salad (320 W) | a04f3c19 | 110.4 | 118.3 | 118.6 | 119.0 | 93% |
| RTX 3070 Ti | Salad (290 W) | c203e4e5 | 82.8 | 87.5 | 87.4 | 88.7 | 93% |
| RTX 3070 Ti | Salad (290 W) | c5d9c25f | 85.1 | 90.4 | 91.4 | 91.3 | 93% |
| RTX 3070 | Salad (240 W) | ff74eb4b | 81.4 | 86.5 | 74.8 | 87.9 | 93% |
| RTX 3070 | Salad (240 W) | f0552404 | 84.0 | 88.1 | 81.6 | 88.8 | 95% |
| RTX 3060 Ti | Salad (220 W) | 9107bf3b | 67.2 | 72.0 | 67.6 | 72.4 | 93% |
| RTX 3060 Ti | Salad (200 W) | 8ccdb586 | 65.1 | 71.0 | 71.1 | 71.0 | 92% |
| RTX 3060 | Salad (170 W, 12 GB) | 6dfca07e | 49.9 | 54.1 | exited (73) | 54.3 | 92% |
| RTX 3060 | Salad (170 W, 12 GB) | 669dc77c | 49.9 | 53.7 | 53.5 | 54.0 | 92% |

A first 3080 Ti PC (`b64f0e45`) stalled for 36 minutes during our run, so it was rerun on `c09c68c3`. SRBMiner
3.8.2 exited with code 73 on one 3060. OneZeroMiner is fastest on most PCs.

### Salad, 12–13 November: the official table until 14 November (PR #293 build; PeakMiner 2.20.7, SRBMiner 3.8.7, OneZeroMiner 1.8.5)

Replaced by the 14 November table at the top of this file. Ours mined with the rotation fold; "Ours, old kernel" is the same build without it.

| Card | Host | Machine ID | Ours | Ours, old kernel | PeakMiner | SRBMiner | OneZeroMiner | % of best | Fold gain |
|---|---|---|---|---|---|---|---|---|---|
| RTX 3090 Ti | Salad (450 W) | aee86d74 | 166.4 | 153.9 | 165.7 | 166.5 | 166.8 | 99.8% | +8.2% |
| RTX 3090 Ti | Salad (450 W) | 22e3868d | 164.1 | 153.7 | 164.1 | 164.3 | 164.5 | 99.7% | +6.8% |
| RTX 3090 | Salad (420 W) | d5edd4e5 | 149.8 | 138.7 | 149.8 | 150.3 | 150.4 | 99.6% | +8.0% |
| RTX 3090 | Salad (350 W) | f618bdee | 127.0 | 119.8 | 127.1 | 128.9 | 128.0 | 98.5% | +6.1% |
| RTX 3080 Ti | Salad (370 W) | e48ec725 | 140.3 | 132.2 | 139.5 | 140.8 | 140.2 | 99.7% | +6.1% |
| RTX 3080 Ti | Salad (350 W) | d6d29abf | 137.9 | 127.7 | 137.6 | 139.1 | 138.2 | 99.2% | +8.0% |
| RTX 3080 | Salad (320 W) | 9965e895 | 116.1 | 108.5 | 115.6 | 116.7 | 116.6 | 99.5% | +7.0% |
| RTX 3080 | Salad (370 W) | 7ec45f05 | 127.4 | 118.8 | 127.6 | 127.9 | 127.9 | 99.6% | +7.2% |
| RTX 3070 Ti | Salad (246.50 W) | 1da95a13 | 81.3 | 74.9 | 80.6 | 81.7 | 81.5 | 99.5% | +8.5% |
| RTX 3070 | Salad (220 W) | 446be81f | 73.1 | 66.6 | 73.5 | 74.4 | 73.6 | 98.2% | +9.7% |
| RTX 3070 | Salad (240 W) | 78c53df0 | 85.0 | 78.0 | 84.3 | 85.7 | 84.9 | 99.2% | +9.0% |
| RTX 3060 Ti | Salad (200 W) | 1f2c4858 | 68.5 | 62.8 | 67.9 | 68.6 | 68.3 | 99.9% | +9.0% |
| RTX 3060 Ti | Salad (220 W) | c898f3c0 | 70.0 | 66.0 | 69.9 | 70.5 | 70.3 | 99.3% | +6.1% |
| RTX 3060 | Salad (170 W, 12 GB) | 6389d2b2 | 53.5 | 49.1 | 46.6 | 53.8 | 53.5 | 99.5% | +9.0% |
| RTX 3060 | Salad (170 W, 12 GB) | f726f676 | 48.6 | 44.7 | 41.9 | 49.4 | 48.1 | 98.4% | +8.7% |

### Salad, 12 November: the rotation fold on RTX 30 (branch `goal/nv-30`)

The rotation fold (rig01's 4090: 335.8 against 317.4 TH/s) built for sm_86 and run on two Salad PCs by
`earn/tuning/salad/nv30box.sh`: the probe, then the fold and the release fold in one core (`hashrate.js`, 120 s
legs taking turns, after 10 s), then the other miners through `bench.sh` on the stand-in pool (180 s after a 60 s
warm-up; the mean of each miner's samples from 30 s on). ptxas 12.6 and 12.8 compile the fold's kernel for sm_86
into the same code as for sm_89, byte for byte, so the hand loop and `asm89.py` carry over unchanged (its check:
0 wrong of 2,365 instructions on the PC's 12.8).

| Card, PC, limit | Rotation fold | Release fold | PeakMiner 2.20.7 | OneZeroMiner 1.8.5 | SRBMiner 3.8.6 | Fold, % of best | Release, % of best |
|---|---|---|---|---|---|---|---|
| RTX 3090, 22ce6093, 390 W | 141.63 (`iy=0`; 141.00 without) at 1710 MHz | 133.13 at 1677 MHz | 143.35 at 1729 MHz | 143.34 at 1726 MHz | 142.70 at 1740 MHz | 98.8% (98.4% without `iy=0`) | 92.9% |
| RTX 3060, 41137306, 170 W | 51.15 (without `iy=0`) at 1810 MHz | 47.60 at 1737 MHz | 51.77 at 1830 MHz | 51.96 at 1835 MHz | 51.94 at 1837 MHz | 98.4% | 91.6% |

- All at the card's limit (389 W, 170 W). Twins: the 3090's release legs 133.12 / 133.13, fold legs 140.97 /
  140.97 / 140.99 and 141.00 / 141.00 / 140.99 in a later run; the 3060's 47.60 / 47.59 and 51.13 / 51.16. Our
  release v0.6.15 through `bench.sh` on the 3090: 133.17. On the 3060 its CLI printed one sample and ran 12 minutes
  (end line 47.4 TH/s, 473 accepted): its row uses `hashrate.js`'s release legs.
- Correct: `verify-hits.js` 400 of 400 on the 3090 for both the file without `iy=0` (499 hits) and with it (506
  hits); 37 of 37 on the 3060 (300 s at 2^231 is all a 51 TH/s card hits). The probe's CPU check 16,384 of 16,384 at
  k 2048 and 2,048 of 2,048 over 8 tiles at k 65536 on both cards. Every share the stand-in pool saw was accepted.
- The probe: the fold holds 98.90% of the pipe on the 3090 and 98.91% on the 3060 (the 4090: 98.96%); the gather
  fold as compiled 89.3% and 88.9%. k 65536 holds more than k 8192 (98.54%) and the memory controller reads 0 to 1%
  on both, so the offsets table (4.2 MB a salt) needs no banding even in the 3060's 3 MB of L2. The release fold
  reads 40 to 42% on the 3060.
- So the gap to the other miners is clock at the same watts (1710 against 1726 to 1740 MHz on the 3090, 1810
  against 1830 to 1837 on the 3060; they hold about 98.7% of the pipe there by their rate), and 0.6% lost between
  the probe's count and the miner on the 3090 (0.1% on the 4090).
- Keys raced (the probe, 45 s at k 65536, 3090): `iy=0` 99.05% against 98.89 to 98.90; `ly=0`, `bt=13`, `sbo=9`
  level; `sbo=5`, `py=1`, `cq=0`, `ae=0`, `ist=4`, `ist=6` and orders 1, 4, 5 lower; `lead=2` WRONG on sm_86 (0 of
  16,384). In the miner `iy=0` gave 141.61 / 141.65 and 141.65 / 141.63 against 141.00 / 141.00, so the sm_86 build
  takes it (`build_rot.py` ARCH_KEYS); the RTX 40 file is unchanged. `PEARL_ROT_TILES` 64 and 128 read 141.70 to
  141.85 against 141.63 to 141.65, inside their own spread (sd 4 to 11 against 1 to 2); 256 read 144.3 and 144.5
  with sd 29 to 32, above the 143.6 a 3090 can do at 1710 MHz with a full pipe, so a sampling artifact of batches
  near the sample window. The default stays 32.
- Shipped as the opt-in for RTX 30 (`--rotation-fold`, `LLMJOB_ROTATION_FOLD=1`): `pearl_rot_sm86.cubin`, built
  and hash-checked in CI beside the sm_89 file, loaded by the host on compute 8.6.

### Salad, 12 November: the readout without bank collisions on RTX 30 (branch `goal/nv-15`)

The rotation fold's readout now reads each IMMA's sums as (0, 2) and (1, 3), on the chain of the other register
parity, with U8 IMMA (`rfold.py rpr=2 u8=1`, with RTX 30's `iy=0`). One Salad RTX 3090 (PC 62b4dc19, 290 W
limit, 84 C, 1506 to 1515 MHz), hashrate.js, alternating 120 s legs, CUDA 12.8 build:

| Loop | TH/s |
|---|---|
| `iy=0` (shipped before) | 123.62, 123.91, 123.81 |
| `iy=0 rpr=2 u8=1` | 124.35, 124.46, 124.42 (+0.47%) |

verify-hits 400 of 400 (469 hits, 24 salts). Now the RTX 30 build (cubin 1b391752...).

### Salad, 12 November: the readout first in its gap on RTX 30 (branch `goal/nv-16`)

The readout's XOR3 steps go first in their gap, right behind the IMMA, and may use more of the chunk's end
(`rfold.py rgf=1 rb0=3 rb1=0`, on top of `iy=0 rpr=2 u8=1`). One Salad RTX 3090 (280 W limit, 74 C, 1456 to
1465 MHz), hashrate.js, alternating 120 s legs, CUDA 12.8 build:

| Loop | TH/s |
|---|---|
| `iy=0 rpr=2 u8=1` (shipped before) | 120.74, 120.81, 120.87 |
| `+ rgf=1 rb0=3 rb1=0` | 121.16, 121.17, 121.25 (+0.36%) |

verify-hits 400 of 400 (454 hits, 23 salts). Now the RTX 30 build (cubin cb50068c...).

### Salad, 12 November: the RTX 40 changes on RTX 30, shared records and B drawn once (branch `goal/nv-30b`)

The two changes the RTX 4090 took that day: records the eight warps share (`sr=1`, kernel built with ROT_SR=1),
and B drawn once with its exchange keys (`bx=1 bsym=4 bsg=8 bsg3=24 bss=2 bord=2 barv=1 bxt=1`, ROT_BX=1). Two
Salad PCs, CUDA 12.8 builds, hashrate.js, 120 s legs in turn (rot s y y3 yn, three rounds):

| Loop | RTX 3090 (280 W, about 1330 MHz) | RTX 3060 (170 W, about 1886 MHz) |
|---|---|---|
| shipped (`iy=0 rpr=2 u8=1 rgf=1 rb0=3 rb1=0`, cb50068c) | 107.76, 107.34, 107.33 | 53.33, 53.32, 53.32 |
| `+ sr=1` (da15a69d) | 108.09, 107.95, 107.98 (+0.49%) | 53.52, 53.55, 53.53 (+0.39%) |
| `+ sr=1` + B drawn once, rb0=6 rb1=3 (the 4090's keys) | 105.30, 105.22, 105.25 | 52.10, 52.11, 52.11 |
| `+ sr=1` + B drawn once, rb0=3 rb1=0 | 105.87, 105.85 | 52.40, 52.39 |
| B drawn once with sr, without `iy=0` | 107.34, 107.29 | 53.32, 53.32 |

verify-hits 400 of 400 on the sr file (451 hits, 12 salts) and on the B-drawn-once file (432 hits) on the 3090.
The probe's CPU check passes for both on both cards. So sr ships for RTX 30 (cubin da15a69d...). B drawn once
does not: on sm_86 its pipe is 94.1% (3090) and 96.4% (3060) at k 65536 against 97.0% and 99.2% for the loop
without it, and `iy=0`, which is worth +0.45% on the plain loop, costs it 2.3%. Without `iy=0` it is only level
with the shipped loop, below sr.

The other miners on the same PCs (bench.sh, stand-in pool at 2^210, means of the 15 s samples from 30 s):

| PC | OneZeroMiner | SRBMiner | PeakMiner | Ours, sr | % of best |
|---|---|---|---|---|---|
| RTX 3090 (280 W) | 108.92 (1362 MHz) | 109.23 (1362 MHz) | 108.46 (1343 MHz) | 108.01 (1330 MHz) | 98.9% |
| RTX 3060 (170 W) | 53.41 (1895 MHz) | 53.31 (1899 MHz) | 53.14 (1890 MHz) | 53.53 (1886 MHz) | 100.2% |

The 3060 is now above the best of them, the 3090 not: at 280 W our loop runs 30 MHz below theirs, so it
spends more energy a MAC. That, not the pipe, is what is left on a watt-capped RTX 30.

B drawn once without `iy=0`, its keys retuned (same 3060 PC, the probe, k 65536, 60 s, two passes each): the
pipe is 98.7 to 99.5% against 99.25% for the sr loop, and pipe x clock lands within the passes' noise of it
(bsg=10, bss=3 and py=1 the best, each one pass above and one below). Raced in the miner on a second, heat-held
3060 PC (80 C, about 1820 MHz), legs in turn: sr loop 52.10 (first leg, card still cooler), 51.23, 51.38;
`bsg=10` 51.07, 51.05; `bss=3` 51.10, 51.24 TH/s: 0.3 to 0.5% below sr at the same heat. Not shipped.

### Vast, 19 November: an RTX 3080 Ti re-baselined against SRBMiner 3.8.9, six loop keys raced, A taken off its stage, power and bank conflicts profiled; nothing ships (branch `goal/rtx30c`)

Nothing ships; the RTX 30 file stays da15a69d. One Vast RTX 3080 Ti (instance 61361305, machine 154213, driver
595.99.02, 350 W limit, power cap the only limiter, 73-80 C). Nsight Compute is refused there (ERR_NVGPUCTRPERM).

The race, `bench.sh` at 94e1e36c on Kryptex, 60 s warm-up, then 5 minutes each in this order; whole-run means of
the 15 s samples:

| Miner | TH/s | MHz | W |
|---|---|---|---|
| ours v0.7.5 (0% fee) | 136.85 | 1684 | 346 |
| PeakMiner 2.20.9 (2% fee) | 140.03 | 1714 | 346 |
| SRBMiner 3.8.9 (2% fee) | 140.03 | 1715 | 346 |
| OneZeroMiner 1.8.6 (1% fee) | 138.31 | 1720 | 347 |

97.7% of best, the same as the official table's two 3080 Ti rows. The gap is clock at the same watts again
(30-36 MHz), as goal-nv-k25 and goal/rtx30 found.

Six `rfold.py` keys not raced on sm_86 before, each on the shipped keys (`nv34box.sh`, `hashrate.js` 100 s legs in
turn, two rounds). Each passed `verify-hits` (75-82 of 75-82 in 60 s; `txm` 362 of 362 in 300 s):

| Loop | TH/s |
|---|---|
| shipped (`ship`, and `re`, its rebuild through `rfold.py`) | 136.42, 136.44, 136.31, 136.45, 136.66, 136.27, 136.51, 136.25, 136.68 |
| `txm=1` (the transcript pointer one ALU step shorter) | 136.84, 137.02, 136.33, 136.82, 136.52, 136.62, 136.65 |
| `txm=1 prr=1` | 137.08, 136.50, 136.47 |
| `prr=1` (reuse marks on PRMT pairs) | 136.47, 136.67 |
| `dpm=1 dpg=3` (over-waits moved) | 136.36, 136.23 |
| `lgg=1` (the record load a gap earlier) | 136.50, 136.44 |
| `cq=0` (no sixth gather quad) | 136.49, 136.32 |
| `pf=1` | does not build with `sr=1` (rfold.py assertion); its legs ran the release fold |

`txm` reads +0.18% on the means, but this host's cooling cycles moved the card between 73 and 76 C, and the rate
follows the temperature at about 0.2 TH/s a degree. `txm`'s legs ran 0.8 C cooler on average, which accounts for
all of it. None of the six is a win.

Box: Vast 61361305, 1.65 h, about $0.33.

Round 2, the same day: RDNA 3's lever in NVIDIA form. RDNA 3 won 2-4% by loading B' fragments straight from memory
into registers, so that only A' goes through LDS (PR #310, 31fb20b6). On the rotation fold, B never goes through a
shared-memory stage: it is gathered from tables that stay resident, and B' made once a job and streamed from the L2
has already lost on the 4090 (goal-nv-k27: -19%, because L2 bytes cost more energy than the on-chip draw). A is the
operand that goes through a stage, so the race takes A off it. Same Vast machine, a second card (instance 61373106,
350 W, 68-69 C the whole time, so the legs are steady). `hashrate.js` 100 s legs in turn:

| Loop | TH/s | MHz |
|---|---|---|
| shipped and its twin (`ship`, `re`) | 139.37, 139.41, 139.42, 139.41, 139.40, 139.40 | 1710-1713 |
| `ga=1 bt=0`: every turn gathers its own six A tiles, with no stage, store or barrier (verify-hits 139 of 139) | 135.77, 135.81, 135.82 (-2.6%) | 1665 |
| `bt=0` alone (WRONG: verify-hits 69 of 138; timing only) | 139.38, 139.37 | 1710 |
| `nax=2`: two more STS.128 a chunk (timing only, results correct) | 139.23, 139.20 (-0.13%) | 1710 |
| `nax=4`: six more LDS.128 a chunk (timing only, results correct) | 139.19, 139.16 (-0.16%) | 1710 |
| `txm=1`, raced again on this steady card | 139.41, 139.36, 139.41 (+0.05%) | 1711 |
| `txm=1 prr=1` (verify-hits 142 of 142) | 139.34, 139.45, 139.41 (+0.06%) | 1712 |

(The ship and twin legs of the `txm` round: 139.26, 139.25, 139.43, 139.34, 139.36, 139.29.)

- Taking A off its stage costs 36 more LDSM and 72 more PRMT a chunk, and saves 7 STS and 19 LDS. It loses 2.6%,
  all of it clock, as `ga` lost on the 5090 (-3.6%).
- On sm_86 a shared-memory instruction is cheap: about 0.065% for an STS.128 and 0.027% for an LDS.128 a chunk,
  against about 0.3% for each on a capped 5090. All 70 of the loop's shared-memory instructions a chunk come to only
  about 2-4% by that price, and each one carries work that would cost more done any other way. On RTX 30 the
  energy is not in shared memory.
- The loop's gathers are already LDSM x4 and its stage loads LDS.128. A wider register tile has no room: the loop
  writes 252 registers, and goal-nv50x's 128 x 384 block lost. `txm` and `txm prr` are under twice their error, so
  neither ships.

Box: Vast 61373106, 1.1 h, about $0.22.

Round 3, the same day: the two leads from the RTX 5090 profile, checked on RTX 30. Neither holds here.

**Power swings.** The 5090 profile found our instant power swinging much wider than SRBMiner's, with the cap
reacting to the peaks. `nvidia-smi -lms 100` (power.draw.instant, which this driver updates about every 0.5 s)
beside `bench.sh` legs, ours (the shipped cubin) and SRBMiner 3.8.9 in turn on the stand-in pool:

| Card | Leg | W mean | W SD | W min-max | MHz | Shown TH/s |
|---|---|---|---|---|---|---|
| 3080 Ti, 350 W (Vast 61388348) | ours | 345.3 | 7.70 | 315-364 | 1722 | 140.8 |
| | SRB | 346.4 | 4.63 | 322-360 | 1759 | 143.7 |
| | ours | 348.4 | 4.32 | 340-357 | 1721 | 141.2 |
| | SRB | 346.8 | 4.29 | 339-360 | 1758 | 143.7 |
| 3080 Ti VM, 400 W (Vast 61392502) | ours | 398.4 | 5.31 | 385-408 | 1814 | 147.7 |
| | SRB | 398.6 | 5.75 | 383-410 | 1846 | 150.6 |

- Both draw the same average power. Ours runs 1.7-2.1% lower clock at that power, which is the whole gap
  (98.0-98.3% shown).
- Our power swings no wider than SRB's in three legs out of four. The wide leg (SD 7.7) is the controller stepping
  between 1710, 1725 and 1740 MHz: at our load one 15 MHz step is about 9 W (340, 348, 353 W), at SRB's about
  4 W. Within one clock bin our SD is 5-7 W against SRB's 4-5 W. The prep kernels are on a side stream and take
  about 0.15 ms a salt (`pearl_gen_dense` 76 us, `rot_aux_tab` 69 us) against 74 ms for each fold launch. A salt
  lasts about 20 s.
- On RTX 30 the gap is energy per clock, not peaks that the cap answers.

**Shared-memory bank conflicts.** Nsight Compute is refused in a Vast container (ERR_NVGPUCTRPERM) but works in a
Vast VM (KVM image, ncu 2025.1.1 from NVIDIA's apt repo). The fold kernel, per `IMMA.16832`:

| Counter (fold kernel, 3080 Ti) | per MMA |
|---|---|
| shared wavefronts | 2.40 (ideal 1.46) |
| bank conflicts (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared`) | 0.95 (loads 0.87, stores 0.08) |
| shared bank reads (4-byte words) | 42.5, which is the ideal for the instructions issued |
| DRAM bytes / L2 bytes | 0.032 / 0.77 |

Source counters, by instruction:

| Instructions | a MMA | wavefronts each (ideal) | excess a MMA |
|---|---|---|---|
| LDSM gathers (54 sites) | 0.229 | 6.6-7.35 (4) | 0.64 |
| transcript `LDS`/`STS [R251]` (32-bit) | 0.005 each | 16 (1) | 0.157 |
| stage `LDS.128` / `STS.128` | 0.094 / 0.031 | 4 (4) | 0 |

- The stage is conflict-free, so padding or swizzling it gains nothing.
- The gather conflicts come from the permuted rows. The bank-pass order refine (`PEARL_ROT_ORDER=2`) already cuts
  them from 1.76 to 1.57 passes a matrix, and on a capped 3080 Ti it raced -0.17% (17 November). So fewer gather
  passes do not pay on sm_86.
- The transcript's word is 16-way: each thread's 16 words are 64 bytes apart, so a warp hits 2 banks. Removing both
  instructions outright (`ntr`, timing only) priced at +0.2% on a 3080 Ti ("RTX 30, priced on a capped Salad RTX 3080 Ti", below), which caps what a conflict-free
  layout could gain. A layout of [word][thread] needs the 16 KB area 16 KB-aligned so that the loop's 3-step pointer
  wrap still works. The area sits at 56 KB. Moving it means moving the tables, and the 99 KB limit leaves no room
  to add a new area. Not built for at most 0.2%.
- A conflict replays a wavefront, but the bank reads stay at the ideal, so it costs issue slots, not data moved.
  That fits the earlier prices on sm_86.
- SRBMiner 3.8.9 exits under Nsight Compute ("Runtime diagnostic code: S01 / Unsupported runtime environment"), so
  there is no SRB figure for sm_86.

Boxes: Vast 61388348 (0.5 h, about $0.10) and VM 61392502 (0.63 h, about $0.14), both destroyed.

## RTX 40-series

Vast, 10 October.

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

### Salad, 29–30 October (PeakMiner 2.20.3, SRBMiner 3.8.0)

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

Where power limits match Vast, the competitors got 4–7% faster and ours didn't.

### Salad, 9–10 November (v0.6.13; PeakMiner 2.20.7, SRBMiner 3.8.2, OneZeroMiner 1.8.5)

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | OneZeroMiner | % of best |
|---|---|---|---|---|---|---|---|
| RTX 4090 | Salad (400 W) | 2aeb962b | 292.2 | 322.0 | 323.1 | 334.1 | 87% |
| RTX 4090 | Salad (450 W) | c1e9408c | 319.6 | 342.1 | 343.4 | 345.4 | 93% |
| RTX 4080 Super | Salad (320 W) | 5e5391f8 | 207.1 | 220.4 | 220.4 | 222.3 | 93% |
| RTX 4080 Super | Salad (256 W) | 549668e2 | 191.2 | 216.5 | 216.2 | 214.6 | 88% |
| RTX 4080 | Salad (340 W) | 6e3fff63 | 201.0 | 213.8 | 213.0 | 217.5 | 92% |
| RTX 4080 | Salad (320 W) | d43ec5f2 | 187.3 | 195.4 | 191.8 | 196.8 | 95% |
| RTX 4070 Ti Super | Salad (285 W) | b3484cf9 | 168.9 | 179.8 | 181.1 | 179.1 | 93% |
| RTX 4070 Ti Super | Salad (285 W) | 2ac34122 | 168.9 | 178.8 | 179.1 | 180.9 | 93% |
| RTX 4070 Ti | Salad (285 W) | a4683a51 | 153.3 | 176.3 | 161.7 | 163.1 | 87% |
| RTX 4070 Ti | Salad (285 W) | 61e44847 | 152.8 | 157.2 | 157.5 | 158.4 | 96% |
| RTX 4070 | Salad (160 W) | 907f31cc | 112.9 | 120.4 | 121.1 | 121.8 | 93% |
| RTX 4070 | Salad (152 W) | 884c62d6 | 109.2 | 116.7 | 117.2 | 118.1 | 92% |
| RTX 4060 Ti | Salad (165 W, 16 GB) | ea966099 | 88.0 | 92.0 | 92.3 | 91.3 | 95% |
| RTX 4060 Ti | Salad (165 W, 16 GB) | b6c25a5b | 89.7 | 93.7 | 94.7 | 95.2 | 94% |
| RTX 4060 | Salad (115 W) | 2caeec23 | 63.8 | 66.2 | 66.8 | 66.6 | 95% |
| RTX 4060 | Salad (115 W) | 4ad13b38 | 64.5 | 66.7 | 70.0 | 67.3 | 92% |

Our gap widens as the power limit drops: 87% on the 400 W 4090 against 93% at 450 W, 88% on the 256 W 4080 Super
against 93% at 320 W.

### Salad, 12–13 November: the official table until 14 November (PR #293 build; PeakMiner 2.20.7, SRBMiner 3.8.7, OneZeroMiner 1.8.5)

Replaced by the 14 November table at the top of this file. Ours mined with the rotation fold; "Ours, old kernel" is the same build without it.

| Card | Host | Machine ID | Ours | Ours, old kernel | PeakMiner | SRBMiner | OneZeroMiner | % of best | Fold gain |
|---|---|---|---|---|---|---|---|---|---|
| RTX 4090 | Salad (450 W) | 0a25ed1e | 356.3 | 337.1 | 355.4 | 356.4 | 357.9 | 99.6% | +5.7% |
| RTX 4090 | Salad (450 W) | fd67d2b9 | 339.2 | 316.8 | 337.9 | 343.1 | 341.3 | 98.9% | +7.1% |
| RTX 4080 Super | Salad (256 W) | 030da504 | 206.7 | 192.0 | 235.5 (suspect) | 209.1 | 204.8 | 87.8% | +7.6% |
| RTX 4080 | Salad (340 W) | 6076eb5a | 211.1 | 199.6 | 212.0 | 211.5 | 213.2 | 99.0% | +5.7% |
| RTX 4080 | Salad (272 W) | a556cac8 | 202.9 | 190.1 | 203.0 | 204.9 | 204.3 | 99.0% | +6.7% |
| RTX 4070 Ti Super | Salad (285 W) | 93c7f101 | 172.7 | 163.5 | 174.2 | 173.1 | 170.4 | 99.1% | +5.6% |
| RTX 4070 Ti Super | Salad (228 W) | 665d7118 | 172.4 | 160.2 | 170.8 | 173.9 | 173.1 | 99.1% | +7.6% |
| RTX 4070 Ti | Salad (242.25 W) | e46a64e0 | 167.2 | 159.5 | 166.9 | 168.0 | 168.2 | 99.4% | +4.9% |
| RTX 4070 Ti | Salad (285 W) | cbc5b9d4 | 165.0 | 156.5 | 164.5 | 165.5 | 165.8 | 99.5% | +5.4% |
| RTX 4070 | Salad (200 W) | 1f57498d | 124.1 | 117.0 | 123.5 | 125.1 | 124.9 | 99.2% | +6.0% |
| RTX 4070 | Salad (200 W) | 133d9fd8 | 126.0 | 120.1 | 125.7 | 127.5 | 127.3 | 98.8% | +4.9% |
| RTX 4060 Ti | Salad (165 W, 16 GB) | 06ae98ec | 94.0 | 89.2 | 94.5 | 95.2 | 95.5 | 98.5% | +5.4% |
| RTX 4060 Ti | Salad (165 W, 16 GB) | ea966099 | 92.2 | 88.6 | 91.8 | 92.6 | 92.4 | 99.5% | +4.0% |
| RTX 4060 | Salad (— W) | 3239c03f | 65.0 | 61.6 | 64.1 | 65.0 | 64.4 | 99.9% | +5.6% |
| RTX 4060 | Salad (— W) | 70b185b0 | 63.2 | 59.7 | 61.9 | 63.3 | 62.4 | 99.8% | +5.8% |

### rig01, native Windows, 10 November (v0.6.14; PeakMiner 2.20.7, SRBMiner 3.8.2, OneZeroMiner 1.8.5)

Every Salad row runs in WSL2; our users run Windows. This is rig01's RTX 4090 (450 W, driver 610.88) on Windows 11,
with the installed app stopped. Each miner mined 5 minutes on Kryptex (prl-us:7048) after a 60 s warm-up, two
rounds, the second in reverse order (PeakMiner, SRBMiner, OneZeroMiner, ours, then back). TH/s is each miner's own
figure, as bench.sh reads it: the competitors' API at the end of the run, ours the mean of its status line after the
first minute. The power, SM clock and temperature columns are nvidia-smi means after the first minute.

| Miner | Round 1 | Round 2 | Mean | Power | SM clock | Temp | Shares (acc/rej) |
|---|---|---|---|---|---|---|---|
| Ours (v0.6.14 CLI, the installed app's runtime and cores) | 317.3 | 317.6 | 317.5 | 449 W | 2531 MHz | 77 °C | 21/0 |
| PeakMiner 2.20.7 | 339.1 | 339.1 | 339.1 | 450 W | 2624 MHz | 77 °C | 27/0 |
| SRBMiner 3.8.2 | 339.3 | 337.9 | 338.6 | 445 W | 2619 MHz | 77 °C | 24/0 |
| OneZeroMiner 1.8.5 | 341.9 | 341.8 | 341.9 | 450 W | 2625 MHz | 77 °C | 20/0 |

Ours is 92.9% of the best, the same gap as Salad's 450 W 4090 (c1e9408c, 92.5%), so WSL2 doesn't explain it.
The competitors also land within 1% of that box. All four hold the 450 W cap with the memory at 10251 MHz, but
ours runs about 90 MHz lower on the SM clock (2531 against 2619–2625): it spends more power per clock. Per MHz it
does 0.1254 TH/s against OneZeroMiner's 0.1302, so the gap splits about evenly between clock (3.6%) and work per
clock (3.7%). SRBMiner's own rate dipped to 239 and 291 TH/s for two samples early in round 1; its end figure
didn't. No miner rejected a share.

Downloads were checked against the SHA-256 digests GitHub lists for each release asset, and SRBMiner's zip also
against its published .md5. Windows Defender quarantines SRBMiner on download, so the competitors ran from
`C:\miners\item16-1110`, rig01's excluded folder. A short Python script ran bench.sh's method, as Git Bash has no jq.

### rig01, 10 November: memory clock, core clock and power limit on the RTX 4090 (PR #290 items 1 and 6)

An elevated session on rig01 could set clocks, so the sweep that #287 item 11 couldn't run was run. Our v0.6.16 CI
core (`pearl_core.node`), `hashrate.js` for 90 s after 20 s of warm-up, native Windows, driver 610.88. Each locked run
had a stock run before and after it; stock stayed within 317.0–317.9 TH/s all afternoon.

The card has four memory speeds. A lock at 1000, 1200, 1500, 2000, 3000 or 5001 MHz runs at 5001 MHz; a lock at
6000, 7001 or 9000 MHz stays at 10251 MHz and changes nothing.

| Memory speed the card ran | TH/s | Against stock | SM clock | Power | Memory busy |
|---|---|---|---|---|---|
| 10251 MHz (stock) | 317.0–317.9 | — | 2525–2536 MHz | 449 W | 3% |
| 5001 MHz | 320.0–320.3 (six runs) | +0.8% | 2552–2554 MHz | 449 W | 5% |
| 810 MHz | 283.1 | −11% | 2640 MHz | 437 W | 32% |
| 405 MHz | 78.5 | −75% | 645 MHz | 93 W | 20% |

The lock frees power for the SM clock, as guessed, but only 5001 MHz is usable: at 810 MHz the fold's DRAM reads
become the limit, and at 405 MHz the whole card drops to a low power state. PeakMiner tunes for 405 and 810 MHz
because its kernel no longer reads DRAM (item 2); ours does.

The lock gains more under a lower power limit. Two runs of each, alternating; repeats agreed within 0.1%:

| Power limit | Stock memory | 5001 MHz lock | Gain | SM clock, stock → locked |
|---|---|---|---|---|
| 480 W (draws 451–463 W) | 319.7, 320.2 | 321.7, 321.7 | +0.5% | 2546–2550 → 2565 MHz |
| 450 W | 317.5 | 320.2 | +0.8% | 2531 → 2553 MHz |
| 400 W | 304.8, 304.8 | 309.1, 309.3 | +1.4% | 2427 → 2468 MHz |
| 350 W | 290.1, 290.1 | 294.1, 294.0 | +1.3% | 2310 → 2345 MHz |
| 300 W | 272.2, 272.2 | 276.9 (one run) | +1.7% | 2168 → 2210 MHz |

At 480 W the card stops at about 460 W, so it is no longer power capped and the lock has less to give. At 300 W the
card still makes 86% of its 450 W rate on two-thirds of the power.

A core clock lock does nothing here (item 6). `nvidia-smi -lgc 2600,2600` and `2700,2700` on top of the memory lock
gave 319.7 and 319.8 TH/s with the SM clock still at 2549–2552 MHz, and `-lgc 2600,2600` alone gave 317.4. The power
limit sets the clock, not the lock. nvidia-smi has no voltage control; a clock offset through NVML does the
job and was run later the same day (next section).

`verify-hits.js` under the 5001 MHz lock: 400 of 400 hits verified.

What it means: where the miner can set clocks (it needs Administrator on Windows or root on Linux, so HiveOS and
mining rigs, not Salad or Vast containers), a 5001 MHz memory lock is worth about +0.8% on a 4090 at its stock limit
and up to +1.7% on a power-limited one. It is measured on one 4090 only.

Built as `--mine-mem-clock auto` (`LLMJOB_MINE_MEM_CLOCK=auto` in the app), off unless asked for. A card with a
lock to try (RTX 40 at 5001 MHz, RTX 50 other than the 5090 at 7001 MHz) mines four 75 s turns, driver's clock and
locked in turn, and keeps the lock only if it was 0.3% faster and ahead in every turn. Run for real on rig01 from
this branch's CLI on Kryptex: the trial measured 320.0 TH/s locked against 317.4 (+0.8%) and kept the lock, with 8
shares accepted and none rejected in under six minutes. The lock is dropped when an LLM co-runs on the card, as
every memory clock lock is.

Changed on 11 November: an RTX 5090 runs this trial too, at 7001 MHz, and runs it by default. It used to be locked
at 7001 MHz outright, on one run on one card (+8.0% at 600 W) whose earlier run had the two clocks level. Not
measured: no 5090 has run the trial. The trial now also hands the card's clock to the driver before its first turn,
so a lock left by a killed miner is not timed against itself. The reset alone was checked on rig01's RTX 4090 while
it mined: `nvidia-smi -i 0 -rmc` on a card with no lock exits 0 and leaves the memory clock at 10251 MHz. A whole
trial with the reset in it has run only in the tests. A trial's answer is kept for a week a card
(`mem-clock-trials.json`), so a start within the week locks at once or leaves the card alone, with no trial.

### rig01, 10 November: core clock offset on the RTX 4090 (item 6, second pass)

Item 6's first pass tried a core clock lock (`nvidia-smi -lgc`), which changed nothing: a lock can't change the
volts the card needs for a clock, and the power limit sets the clock. A clock offset can. nvidia-smi has no
switch for it, but NVML does (`nvmlDeviceSetClockOffsets`, GPU clock, P-state 0; this card allows −1000 to
+1000 MHz), and PeakMiner (`--gpu-core`) and SRBMiner (`--gpu-coffset`) both expose it. The RTX 40 review of
10 November pointed this out.

Same method as the memory clock sweep: our v0.6.16 CI core, `hashrate.js` for 90 s after 20 s of warm-up, a stock
run between every setting, 450 W limit, native Windows, driver 610.88, Administrator. Each offset first ran
`verify-hits.js`; every one verified 400 of 400 hits. The offset was set back to 0 afterwards.

| Setting | TH/s | Against stock | SM clock | Power |
|---|---|---|---|---|
| Stock (five runs) | 317.3–317.5 | — | 2530–2533 MHz | 449 W |
| +50 MHz | 320.9 | +1.1% | 2559 MHz | 450 W |
| +100 MHz | 325.3 | +2.5% | 2593 MHz | 449 W |
| +150 MHz | 331.2 | +4.3% | 2640 MHz | 449 W |
| +150 MHz and the 5001 MHz memory lock | 333.3 | +5.0% | 2660 MHz | 450 W |
| 5001 MHz memory lock alone | 319.9 | +0.8% | 2554 MHz | 449 W |

A sixth stock run, the first of the sweep, read 307.4 TH/s with a dip in it (sd 41); it is left out.

About 70% of each offset step comes back as clock at the same watts (+107 MHz for +150), and the gain was still
rising at +150 MHz, where the sweep stopped by choice. With the offset and the memory lock together this card runs
at 97.5–98.4% of what the three competitors did on it at stock settings on 10 November (338.6–341.9 TH/s), up from
92.9%.

What it does not show. It is one card: how far a card can be offset before it computes wrong differs card to
card, so a shipped version would have to step up and hit-check each step, as the memory clock trial does. No
long run was made, so stability over hours at +150 MHz is not known. It needs Administrator or root, so it moves
no Salad or Vast row. And the competitors have the same switch: against a competitor tuned the same way it keeps
pace, it does not pass. Nothing is built into the app.

### rig01, 10 November: `--mine-core-offset auto`, the offset trial run in the miner

The sweep above as a switch in the miner, built on branch `goal/offset` and not yet in a release:
`--mine-core-offset auto` (`LLMJOB_MINE_CORE_OFFSET=auto` in the desktop app). Each RTX 20, 30, 40 or 50 card
mines 75 s at offset 0, at +50, +100 and +150 MHz, and at 0 again. Before a step's turn counts the card is handed
a job with an easy target and 100 of its hits are recomputed in JS; one wrong hit ends the climb. It keeps the
highest step that passed, beat the step before it by 0.3% and beat offset 0 by 0.3%. Off unless asked for. The
earn README has the rest.

Run on rig01's RTX 4090: the branch's CLI on the Kryptex pool, mining only, with the core CI built from the
branch (`native core` run 42904856120), 450 W limit, native Windows, driver 610.88, Administrator. The rates are
the trial's own turn means; clock, power and temperature are nvidia-smi's, read every 5 s.

Run 1, `--mine-core-offset auto` alone:

| Turn | TH/s | Against 0 | SM clock | Power | Temperature |
|---|---|---|---|---|---|
| 0 (two turns) | 317.4 | — | 2533–2535 MHz | 449 W | 76 °C |
| +50 MHz | 320.7 | +1.0% | 2554 MHz | 450 W | 78 °C |
| +100 MHz | 325.2 | +2.5% | 2592 MHz | 450 W | 77 °C |
| +150 MHz | 331.2 | +4.3% | 2641 MHz | 449 W | 76 °C |

It kept +150 MHz, the step the sweep found best, and the rates agree with the sweep's to 0.2 TH/s. The miner's
line: `core clock offset kept at +150 MHz on GPU 0 while mining (+4.4%): 317.4 TH/s at 0, 320.7 at +50 MHz,
325.2 at +100 MHz, 331.2 at +150 MHz` (its 4.4% is from the unrounded means). The verdict came 6 min 27 s after
the start: five turns and about 11 s of hit checks in all. After it the card mined at 331.3 TH/s with NVML
reading offset 150 MHz, at 2640 MHz and 450 W. 13 shares accepted, none rejected; 10 of them were found with the
card above offset 0, so each was recomputed before it was sent. Stopped with Ctrl+C, the miner logged `core clock
offset back to 0 on GPU 0` and exited 0, and NVML then read offset 0.

Run 2, with `--mine-mem-clock auto` as well. The memory clock trial ran first and kept its lock (319.9 TH/s at
5001 MHz against 317.6, +0.7%). The offset trial started on the sample that ended it and ran with the lock in
place:

| Turn, memory locked at 5001 MHz | TH/s | Against 0 | SM clock | Power |
|---|---|---|---|---|
| 0 (two turns) | 320.0 | — | 2553–2555 MHz | 449 W |
| +50 MHz | 324.0 | +1.3% | 2586 MHz | 449 W |
| +100 MHz | 328.4 | +2.6% | 2624 MHz | 449 W |
| +150 MHz | 333.6 | +4.3% | 2662 MHz | 449 W |

It kept +150 MHz again (the miner's line says +4.2%, from the unrounded means): 333.6 TH/s, +5.0% on the
317.6 TH/s the run began at, the same as the sweep's 333.3 for the two together. 29 shares accepted, none
rejected, 16 of them with the card above offset 0. Stopped with Ctrl+C:
`memory clock released on GPU 0`, `core clock offset back to 0 on GPU 0`, exit 0, and NVML read offset 0.

The hit check on its own, run as the miner runs it (the card's rate first, then the check's job with the bound
set for 30 hits a second: 2^231.7 at 317.8 TH/s): 100 hits in 3.2 and 3.3 s (31.4 and 30.0 a second), all 100
right both times. Recomputing a hit took 4.3 to 4.4 ms, so one check is 0.43 s of the miner's event loop.

Also checked on this card:

- `verify-hits.js` on the new core at offset 0: 400 of 400, from 1,397 hits over 177 salts. The core mines as
  before.
- The core's read answers offset 0 and a range of −1000 to +1000 MHz, as pynvml does. It takes 35 ms the first
  time and 12 to 16 ms after, and works without administrator rights. Its set, from a process without them,
  answers `Insufficient Permissions` (NVML code 4), and the miner then says the trial needs root or administrator
  rights and mines on at the card's own clock.
- Started with +50 MHz already on the card, as a run killed hard leaves it: the miner said so and the trial
  started from 0. The miner killed hard at +50 MHz left the card at +50 MHz (NVML read it), as the README warns.

What it does not show. It is still one card, and no card that fails a step has been run, so the failing paths
are covered by the unit tests only. Each run spent about two minutes at +150 MHz, so stability over hours is not
known. A check of 100 hits catches a card that computes a few percent of its results wrong; a card wrong once in
several hundred passes it, and is caught later by the check on each share it sends. The trial takes six and a
half minutes of every start.

### rig01, 11 November: CPU use per card, and a search thread that sleeps (branch `goal/cpu-per-card`)

Why: on 11 November a rig of 8 NVIDIA cards on a 2-core Celeron G1840 (HiveOS, v0.6.15) made 31.8 TH/s on each
RTX 3070 at 77 W where OneZeroMiner made 74.9 at about 101 W; the load average reached 6.8 and HiveOS's watchdog
rebooted it. It ran `--mode auto` with one LLM server at the time.

Where the CPU went, by reading and by OS thread counters (`Get-Process` thread times over 55 s of a 60 s run; one
core on a synthetic job, hardest target, nothing else in the process; scripts in the session's `cpu/` folder):
one native thread a card, `PearlCore::SearchLoop`, and all of it in the wait. It keeps two launches queued and
waits for the older one in `cudaEventSynchronize` on a plain event, which busy-waits. A launch on the 4090 is 28 ms
(36 a second), and queueing one and reading its hit count takes about 0.1 ms. Every other thread in the process
(Node's, the driver's) read under 0.002 cores. The installed app's mining process read 1.00 core the same way.

RTX 4090, 450 W, same build each row, the wait chosen by `PEARL_WAIT`, 60 s after a 12 s warm-up:

| Wait | Cores used | TH/s | MHz | W |
|---|---|---|---|---|
| spin (v0.6.15's), twin runs | 0.995, 0.999 | 317.44, 317.41 | 2534, 2532 | 449 |
| blocking event | 0.003 | 317.37 | 2530 | 449 |
| blocking event + `cudaDeviceScheduleBlockingSync` | 0.003 | 317.19 | 2532 | 449 |
| poll (`cudaEventQuery`, 0.5 ms sleep) | 0.006 | 317.11 | 2531 | 449 |
| spin, a new job every 5 s | 0.991 | 317.56 | 2535 | 449 |
| blocking event, a new job every 5 s | 0.002 | 317.16 | 2535 | 449 |
| blocking event + device flag, a new job every 5 s | 0.002 | 317.14 | 2528 | 449 |
| **shipped form** (blocking event, thread priority raised), twin runs | 0.004, 0.003 | 317.37, 317.13 | 2534, 2528 | 449 |
| shipped build with `PEARL_WAIT=spin` | 0.998 | 317.26 | 2527 | 449 |

The spread of like runs is 0.08% (317.13 to 317.37), so no wait differs from another in rate. The device flag added
nothing, with or without job switches, and is not set.

A starved CPU, imitated: the miner pinned to one core (affinity mask 1), alone or beside three busy loops pinned to
the same core at normal priority. Four busy threads on a core is what 8 spinning cards on 2 cores are.

| Wait | One core, alone: TH/s | One core + 3 busy loops: TH/s | W | GPU busy |
|---|---|---|---|---|
| spin | 317.73 | 173.82 and 184.12 (55% and 58%) | 299, 312 | 53%, 58% |
| blocking event, normal priority | 317.21 | 208.65 | 331 | 60% |
| poll, normal priority | not run | 260.33 | 409 | 88% |
| blocking event, priority raised | not run | 317.49 and 317.52 | 449 | 100% |
| poll, priority raised | not run | 317.28 | 449 | 100% |

The spinning miner on the busy core lost what the field report lost: under 60% of the rate, the card under 60% busy
and well under its power limit. Sleeping alone did not cure it, because a thread at normal priority is woken late on
a busy core; with its priority raised (`THREAD_PRIORITY_HIGHEST`; nice -10 on Linux) it ran at full rate.

`verify-hits.js` on the shipped build: 400 of 400 with the default wait, with `poll` and with `spin`.

Not measured: Linux (the code is the same file, but the driver's blocking wait, the scheduler, and the nice value,
which needs root, are not; nothing here was compiled or run on Linux), a rig of many cards, a real 2-core CPU, and
any card but the 4090. The busy loops stand in for other cards' spinning threads on Windows' scheduler, not Linux's.
The AMD and Intel hosts wait their own way and were not touched. `nvidia-smi`, which the app runs every 5 s for
temperatures, took 39 ms of CPU a call here (0.008 cores); on a slow CPU with 8 cards it will be more.

### rig01, 12 November: the readout without bank collisions, and B drawn once at the cap (branch `goal/nv-15`)

RTX 4090, 450 W. `rfold.py rpr=2 u8=1` (see the RTX 30 entry) against the loop shipped before, hashrate.js,
alternating 120 s legs, one addon: 335.92, 336.10, 336.07 TH/s before; 336.92, 336.92, 336.91 after (+0.26%,
2597 MHz). verify-hits 400 of 400 on the 12.6 and 12.8 files. On Kryptex through the CLI with the core finding
the file beside it: 336.41 TH/s by the miner's count over 300 s, 18 shares accepted, none rejected. Now the
RTX 40 build (cubin a5a7931e..., which CI makes on Linux and Windows). OneZeroMiner's 341.9 is 1.5% ahead.

B drawn once with the same readout draws OneZeroMiner's 339 W at a locked 2400 MHz (ours 345), but at the cap it
runs 98.35% of the pipe at 2612 to 2615 MHz against 99.18% at 2597: level within 0.2%. Not shipped. Tables in
`native/probes/README.md`, "The readout without bank collisions, shipped; B drawn once level at the cap".

### rig01, 12 November: the readout first in its gap (branch `goal/nv-16`)

RTX 4090, 450 W. `rfold.py rgf=1 rb0=3 rb1=0` on top of `rpr=2 u8=1`: the readout's steps first in their gap,
not right ahead of the next IMMA, and allowed into gaps 41 to 44 of the chunk's last turn and the next chunk's
first three. The probe holds 99.54% of the pipe against 99.18%, the same as a loop with no readout at all.
hashrate.js, alternating 120 s legs, one addon: 336.93, 337.07, 337.10 TH/s before; 338.00, 338.04, 337.93
after (+0.29%, 2596 to 2597 MHz). verify-hits 400 of 400 on the 12.6 and 12.8 files. On Kryptex through the CLI
with the core finding the file beside it: 337.48 TH/s by the miner's count over 298 s, 14 shares accepted, none
rejected. Now the RTX 40 build (cubin c7219d7d...). OneZeroMiner's 341.9 is 1.1% ahead. What is left of the
readout's cost is energy: its 96 XOR3 a chunk are the least the protocol allows.

### rig01, 12 November: operand encodings, and the memory clock on the rotation fold (branch `goal/nv-17`)

RTX 4090, 450 W, the shipped rotation fold (`rpr=2 u8=1 rgf=1 rb0=3 rb1=0`).

Operand encodings, probe, watts at a locked 2400 MHz core clock (pipe 99.54% in every run): the shipped
unsigned operands (fill 63 plus the noise, 0 to 126) 346.7 to 347.4 W once settled. Fill 64: 346.5 to 347.1 W
(level). Signed operands cost more: A or B as the signed noise (timing only) +10 W each, both +28 W, fill -64 read
as signed +10 W. Each chunk's k sorted so neighbouring operands are alike (orders 10, 11): more watts per point
of pipe, not fewer. No encoding is cheaper than the shipped one.

The memory clock, hashrate.js, alternating 120 s legs:

| Memory clock | TH/s | SM clock | Power |
|---|---|---|---|
| driver's (10251 MHz) | 337.90, 337.91, 337.87 | 2594 to 2596 MHz | 449 W |
| 810 MHz | 341.55, 341.40, 341.44 (+1.05%) | 2624 to 2626 MHz | 445 to 446 W |
| 5001 MHz | 340.34, 340.37 (+0.7%) | 2616 to 2617 MHz | 449 W |

405 MHz puts the card in a low power state (645 MHz SM clock). On the release fold 810 MHz lost 11% (10 November,
above); the rotation fold reads almost no DRAM, so it gains. OneZeroMiner, same card and pool, its own API over
90 s: 342.08 and 341.89 TH/s at the driver's clock, 342.76 and 342.87 at 810 MHz, 344.47 at 5001 MHz. Like for
like ours is 99.6% of it at 810 MHz and 99.1% at each miner's best setting: the lock does not close the gap.
On the rotation fold an RTX 40 card now tries 810 MHz by default through the existing memory clock trial.
On Kryptex through this branch's CLI with `--rotation-fold` and no memory flag: the trial kept 810 MHz (340.5
TH/s locked against 337.5, +0.9%), then 340.8 TH/s by the miner's count; 20 shares accepted, none rejected in
669 s; verify-hits 400 of 400 beforehand.

### rig01, 12 November: records the eight warps share (branch `goal/nv-19`)

RTX 4090, 450 W. `rfold.py sr=1` on the shipped loop: the offsets records carry no warp's base, so all eight
warps of a block read one stream (`rot_aux_offs` built with `ROT_SR=1`, gprobe `rot 4`) and seven of a k-step's
eight record loads find their bytes in the SM's cache, not the L2. The bases sit in uniform registers named in the
shared-memory address (`[R+UR+imm]`, no extra step); the record pointer moves by a uniform step a turn.
Probe at 810 MHz memory: 99.32% of the pipe at 2633 to 2639 MHz against 99.43% at 2625 to 2626 (pipe x clock
2615 to 2621 against 2610 to 2611); +0.5% tiles in 60 s. hashrate.js, alternating 120 s legs, one addon:

| Memory clock | Shipped loop (d18246d5) | With sr (e0e8a527) | Gain |
|---|---|---|---|
| 810 MHz | 341.34, 341.37, 341.34 (2626 MHz, 443 W) | 343.37, 343.40, 343.02 (2644 MHz, 448.5 W) | +0.56% |
| driver's (10251 MHz) | 337.97, 338.08, 337.97 | 339.56, 339.64, 339.56 | +0.5% |

verify-hits 400 of 400 on the 12.6 and 12.8 files. On Kryptex through the CLI with `--rotation-fold` (the trial
kept 810 MHz): 342.56 TH/s by the miner's count over 299 s at 2644 MHz, 11 shares accepted, none rejected.
Against OneZeroMiner's 342.8 at 810 MHz (above) that is 100.1% by hashrate.js; against its best, 344.5 at
5001 MHz, 99.6%.

Then B drawn once with sr (`y24` + `sr=1`, now the RTX 40 build, cubin 3a354cac...). Every B-drawn-once form with
`bxt=1` had faulted in some placements (an illegal address, only in repeated launches). The cause: the transcript
store in the loop's tail had no read scoreboard, and the function's return zeroes R3 right after the epilogue; a
store queued behind the four exchange stores read R3 after that and wrote to local address 0. With the store
holding its read scoreboard, all five faulting placements run clean. sr helps this form more than the old loop
(probe pipe 98.86 -> 99.09%, +17 to +25 MHz). hashrate.js against the sr loop above, alternating 120 s legs:

| Memory clock | sr loop (e0e8a527) | B drawn once + sr (3a354cac) | Gain |
|---|---|---|---|
| 810 MHz | 343.03, 343.05, 343.05 (2642 MHz) | 344.51, 344.38, 344.37 (2665 MHz) | +0.40% |
| driver's (10251 MHz) | 339.15, 339.51, 339.57 | 340.75, 340.79, 340.61 | +0.37% |

verify-hits 400 of 400 on the 12.6 and 12.8 files. Kryptex through the CLI (the trial kept 810 MHz): 343.79
TH/s by the miner's count over 299 s at 2667 MHz, 7 shares accepted, none rejected. Against OneZeroMiner:
100.5% of its 342.8 at 810 MHz, 99.99% of its best (344.47 at 5001 MHz). Level with its best, not above it.

### rig01, 12 November: over-waits, the swing in B drawn once's rate, and batch size (branch `goal/nv-k20`)

RTX 4090, 450 W, 810 MHz memory, the shipped B-drawn-once loop (3a354cac). Nothing here beats it.

- Over-waits. goal-nv50f's fix for RTX 50 (`dpm=1 dpg=3`: a gather's subtract that waits on a scoreboard a newer
  load also set has the wait moved onto that newer load) is now an `rfold.py` key for sm_89 too, off by default.
  It moves 12 waits a chunk in the shipped loop and changes nothing: probe pipe 99.08-99.10% against 99.08-99.09%,
  pipe x clock 2635-2639 against 2637-2644 (40 s runs). With the scoreboards handed out in turn (`sbo=0`) the loop
  drops to 97.58%, and `dpm=1` brings it back to 99.08%: the two fixes do the same job, and on sm_89 the loop's
  remaining 0.9% of pipe is not in the gathers' waits.
- The swing. hashrate.js samples of B drawn once have sd 1.6-1.8 against 0.6-0.8 for the older loop. Every sample
  kept (100 ms clock and watts beside them): it is one 0.5 s window 1.5% high once a salt (every 6.5 s), the rest
  as steady as the older loop's. The time-weighted rate equals the plain mean (344.55 and 344.55), so the swing
  hides no loss. One of four 70 s runs had a 0.13 s stall and a 2 s pause of the whole process (host side).
- Batch size (`PEARL_ROT_TILES`, tiles a block a batch, default 32), hashrate.js legs of 120 s in the order
  32, 64, 128, 128, 64, 32: 343.80 (first leg), 344.72, 344.05, 344.23, 344.62, 344.72 TH/s. 128 is 0.15% lower
  (its sd 0.6: bigger batches smooth the salt's window); 64 is level with 32. Default kept.

### rig01, 12 November: A shared by all eight warps, priced; the shipped loop at three memory clocks (branch `goal/nv-k21`)

RTX 4090, 450 W. Nothing ships; the shipped cubin (3a354cac) is unchanged.

- The one block shape with fewer shared-memory steps. Warps of 96 x 64 in a 96 x 512 block (one row band, eight
  column slots) would share A eight ways and draw B alone: 15.5 shared-memory steps a warp a k-step against 17,
  but 38 ALU steps against B drawn once's 28, 64 KB of B tables, and all eight warps tied to one stage. Priced with
  the new timing-only key `ah=3` (a first turn gathers and stores only A's tiles 0-2 and loads 3-5 from the stage),
  probe at 810 MHz memory, 40 s, pipe x clock: shipped loop 2636-2645; its A-unshared parent s30 2621-2624;
  s30 + ah=3 (that block's steps) 2655-2657 (+0.6%); the same with one barrier for all eight warps 2649, twice,
  at 2685 MHz, the card's top clock, 437 W (+0.35%). The real form also needs a mid-turn exchange between two
  warps and a new tile shape for the host and the pool. At most +0.35% before those costs, and the 2685 MHz
  ceiling caps any form at about +1.2%: not built.
- `ah` on B drawn once (not a buildable form, only a price): 2664 at the 2685 MHz ceiling. A first try left
  tiles 3-5 unstored, so the IMMA folded constants and drew 379 W: a timing-only key must keep operands fresh.
- The shipped loop by memory clock, hashrate.js 120 s legs in the order 810, 5001, driver, twice: 810 MHz 344.04
  and 343.86 (2662 MHz, 448 W); 5001 MHz 341.88 and 342.21 (2645 MHz); driver's 10251 MHz 340.42 and 340.41
  (2621 MHz, 449 W). Against OneZeroMiner at the same clocks (342.8, 344.47, 342.0): 100.3%, 99.3%, 99.5%. The
  fold reads almost nothing from DRAM, so the loss at higher memory clocks is the memory's own power taking watts
  from the core; it falls with the clock and no kernel change touches it.

### rig01, 12-13 November: the 96 x 512 block built and raced (branch `goal/nv-k22`)

RTX 4090, 450 W, 810 MHz memory, gprobe 30 s legs, against the shipped loop (ys24f, the same code as 3a354cac) in
the same holds. Nothing ships; the shipped cubin is unchanged (rebuilt: 3a354cac, as listed).

- Built in the probe: gprobe `-DGP_W8=1` (96 x 512 tiles, A 12 KB, B 64 KB, transcript in the frame, 12 KB
  stage) and rfold.py `w8=1`. Each warp pair makes a k-step's six A tiles at its first turn, three each, and
  swaps halves through the stage. CPU check 16,384/16,384 at k 2048, 8,192/8,192 at 8192, 1,024/1,024 at 65536.
  15.5 shared-memory steps a warp a k-step against 17.
- With the barriers it needs (one of all eight warps in the first turn, one at the chunk's end, the pair's):
  97.33% of the pipe at 2685 MHz (the card's top clock), 435 W: pipe x clock 2613 against the shipped loop's
  2643-2653 (-1.2%). It does take 2.4% less energy a tile, but at 810 MHz memory the shipped loop already runs
  at 2668-2685 MHz, so there is no clock left to win.
- What the barriers cost (timing-only key `w8nb`): none at all 99.34-99.36% (2649-2669 pipe x clock, wrong
  sums); without the chunk-end barrier 98.18%; the pair's barrier or the first turn's alone about 0.1 each. A
  barrier of all eight warps stops every scheduler at once, about 0.8 points a barrier a chunk.
- Group barriers instead (warps 0-3 and 4-7, arrive/sync as bord=2, keys `w8g`): 97.8% with a group sync at
  the chunk's start (`w8g=1`); 98.6-98.8% (pipe x clock 2634-2639) with "slot free" barriers that only the
  slot's own pair syncs on (`w8g=3`). Both are still below the shipped loop, and `w8g=3` is not correct yet: 1
  to 3 tiles of 64 wrong and an occasional hang at k 65536, timing-dependent. A simulation of its barrier
  protocol (random warp speeds, the card's counting) finds no race, so the fault is in how the card runs it.
- Found on the way: in a patched cubin only barriers 0 to 7 work. Numbers 8 to 11 from a register give "an
  illegal instruction"; barrier 8 as an immediate orders nothing (wrong tiles). A ptxas-built kernel uses 8 to
  15 without trouble, so it is something the patch does not set. And a barrier that reads a predicate set 6
  cycles earlier hung the card: set predicates for barriers early.
- Ceiling: even the barrier-free timing form is only +0.3 to +0.7% over the shipped loop here. The form needs
  at least 0.5 points of pipe for barriers. Not worth finishing on this card at this memory clock.

### rig01 and Salad, 13 November: the 96 x 512 block at Salad's memory clock, and what shares and job switches cost (branch `goal/nv-k23`)

**The 96 x 512 block at the driver's memory clock.** The k22 race ran at 810 MHz memory, where the card sits at
its top clock. Salad's cards run at the driver's memory clock, power-capped. Same probe builds (k 65536, 30 s
legs, rig01's 4090), pipe x clock:

| Form | 450 W | 350 W |
|---|---|---|
| shipped loop (ys24f, correct) | 99.32% x 2624-2627 MHz = 2607 | 99.32% x 2435-2438 = 2418-2422 |
| 96 x 512, all barriers (`w8=1`, correct) | 97.82% x 2643-2645 = 2586 | 97.82% x 2470-2476 = 2416-2422 |
| 96 x 512, no barriers (`w8nb=7`, TIMING ONLY) | 99.49% x 2620-2622 = 2608 | 99.49% x 2430-2431 = 2418-2419 |
| 96 x 512, group barriers (`w8g=3`, wrong in 1-3 tiles) | 98.52% x 2628-2638 = 2589-2599 | 98.51% x 2448-2452 = 2411-2416 |

Even with no barriers the block only ties the shipped loop: its 1.5 fewer shared-memory steps a warp a k-step
are paid for by 2.5 more gathers (10 ALU steps). At 350 W the clock gives back lost pipe almost one for one, so
on a deeply capped card only the energy a unit of work counts, and the two forms spend the same. Closed. Also
raced on the shipped loop, all correct, nothing gained: `iy=0` (RTX 30's and 50's key) 96.19% x 2644 = 2544
against 2607-2611 at 450 W; `bt=13` 2606-2607; `ly=0` 2600; `py=1` 2606.

Where the shipped loop's energy goes at 350 W and the driver's memory clock (TIMING ONLY forms, pipe x clock
against the loop's 2422-2428): no readout (`nord`) 2472 (+1.9%); B's gathers' LDSM as moves (`nold=3`) 2449
(+1.0%); every gather's (`nold=1`) 2464 (+1.6%); A's alone (`nold=2`) 2421 (nothing); no PRMT (`noprm`) 2441
(+0.6%); LDSM.T as LDS.128 (`ldk=3`) 2417 (nothing); the exchange's syncs left out (`bnsy`) 2408 (worse); no
stage stores (`nost`, A's operands then zero three turns in four) 2565 (+5.7%, operand values again). The
readout is every miner's cost; what is ours alone and movable is about 1.6% in the gathers' loads and 0.6% in
their PRMT.

**What our miner loses outside the fold.** In the official RTX 40 runs above our CLI's 1 s lines average 0.17 to
0.65% under their own median on 9 of 13 PCs (a few lines 2-6.5% low), where each competitor's mean sits on its
median. On rig01 those low lines come at shares and at job switches. Measured with our CLI on the stand-in pool
(`verify-pool.js serve`), 170 s legs, the mean of every 1 s line past 40 s, 450 W, driver's memory clock:

| Legs (TH/s) | shipped host code | reads deduped, search threaded | and A's tree from B's |
|---|---|---|---|
| no shares, no new jobs | 340.36, 340.32 | | |
| a share every ~3.8 s | 338.75, 338.84 | 339.74, 339.77 (+0.29%) | 339.54 (shares 47 of 47 checked, a job every 30 s too) |
| a new job every 10 s | 339.38, 339.38, 339.43, 339.33 | 339.63, 339.49, 339.51, 339.45 | 339.66, 339.56 |

Repeats of one build agree within 0.1%.

- A share cost about 15 ms of the GPU's time. A hit's device reads (`HitReads`) queued one 1 KB copy a proof
  leaf: 1,792 for the rotation fold's 12 rows and 16 columns at k 65536, though under the constant fill a
  compact operand has only two distinct chunks. The search thread was busy queueing them while the one batch
  behind it ran out. Now a read of bytes an earlier read already fetches takes that read's copy: about 160
  copies a share, and a share costs about 1 ms.
- A new job cost about 28 ms: the draw (both operand trees, 16-17 ms on the 4090) and then the job's first salt
  mining in the definition's order (about 4.5% slower) for the 220 ms its order search took on one thread. The
  first salt's search now runs on up to 8 threads (31 ms here), and A's tree takes B's chunk CVs: under the
  constant fill every chunk of A past the stamp's has the CV of B's chunk of the same index, so a third of the
  chunk hashing goes (draw 13-15 ms). 47 of 47 shares across 6 jobs passed the stand-in pool's full check with
  it, and 47 of 47 again on the core rebuilt after rebasing. A job now costs about 20 ms here.
- On a Salad RTX 4070 (4 vCPU; ROTLOG=1 lines): draw 39-41 ms and 210-240 ms unordered (search 171-224 ms on one
  thread) with the old code; draw 30-31 ms and 62-71 ms unordered with the new. Raced on that PC on Kryptex,
  the same build with and without the three changes (`bench.sh` `oursa` legs, the old behaviour by
  environment), 300 s legs taking turns, the PC steady at 196 W, 2715 MHz, 64 C: new 126.39, 126.42, 126.37,
  126.37 TH/s; old 126.32, 126.28, 126.28, 126.31 (+0.07%). Lines under 98.5% of the median lost 0.10-0.19 s a
  leg against 0.19-0.33 s. A first 4070 PC lost 7% to another workload halfway (its clock up, its VRAM use up
  200 MiB) and was dropped.
- So these were worth about 0.1% on a capped RTX 40 card on Kryptex, and 0.3% where shares are frequent. A job
  still costs about 30 ms on a 4070, nearly all of it hashing B's tree (16.8M chunks at n 262144).

Switches for racing (all on by default): `PEARL_HIT_DEDUP=0`, `PEARL_ROT_SEARCH_THREADS=1`, `PEARL_ROT_TREES=0`
put the old behaviour back; `PEARL_ROT_LOG=1` logs each job's draw and each unordered salt. `bench.sh` has an
`oursa` leg (the same build with `OURS_A`'s environment), dips in its `linemean` line and a `rotlog` summary.

### rig01, 14 November: the 5090's 96 x 512 block on RTX 30 and 40, and the RTX 40 loop's last plain steps (branch `goal/nv-k24`)

Nothing ships; the RTX 40 and RTX 30 cubins are unchanged (the loop rebuilds to 3a354cac as listed). Probe
figures (`gprobe`, k 65536, 30 s legs, rig01's RTX 4090, the driver's memory clock), pipe x clock (pxc), or watts at
a locked 2400 MHz core where the capped clock's steps are too coarse.

**The 96 x 512 block (`w8`) against the RTX 30 loop.** On the 5090 the block won against a loop without the B
exchange (sm_120 has none). RTX 30 runs that kind of loop too (sm_86: `sr=1 iy=0`, no `bx`), and goal-nvk23 had
raced the block only against the RTX 40 loop. Same ISA, so priced on the 4090 at two power limits:

| Loop | 450 W | 350 W |
|---|---|---|
| RTX 40 shipped (`bx` + `sr`) | 2596, 2586 | 2400, 2398 |
| RTX 30 shipped (`sr`, `iy=0`) | 2589, 2590 | 2395 |
| 96 x 512, group barriers (`w8g=4`, correct) | 2571 (98.22% pipe) | 2384 |
| the same with `iy=0` | 2568, 2569 | 2383 |
| 96 x 512, barriers of all eight warps (`w8=1`, correct) | 2569, 2569 | 2397 |
| 96 x 512, no barriers (TIMING ONLY) | 2590, 2587 | 2397 |

Even its ceiling is only level with both loops. Closed for RTX 30 and 40. (`w8g=4` on sm_89 had left the cubin's
barrier count at 5 while it uses barriers 1 to 6: the card stopped with "an illegal instruction". `rfold.py` now
writes 8; with that `w8g=4` is correct on sm_89, 512/512 at k 65536 in every run.)

**The RTX 40 loop's plain steps.** A chunk is 192 IMMA and 328 other steps. New `rfold.py` keys, off by default:

- `bxu`: the exchange's loads read `[R251 + UR30 + imm]` (UR30 = +0x800 for band 0, -0x800 for band 1), so the two
  XORs a make turn that moved R251 to the partner's area and back go. 4 LOP3 a chunk fewer. Correct: 16,384/16,384
  at k 2048 (orders 6 and 1), 4,096/4,096 at k 8192, 512/512 at k 65536.
- `bimm`: the exchange's syncs as immediates under the band predicate, like its arrives (6 + band ahead of the
  stores, 5 - band ahead of the loads), so the count's two XORs a make turn go. 4 LOP3 fewer. With `bxu`
  (`bxum`): correct at k 2048 in orders 6 and 0, k 8192 and 65536. (Its first build hung: the epilogue's last sync
  still read the count's register. Fixed.)
- `tx`: the transcript word's address worked out once a chunk. 3 ALU fewer, but 2.3 points of pipe lost (its later
  load sits on the scoreboard other steps wait on): -1.3% at 350 W. Dropped.
- `brr`: the band barrier as one `BAR.SYNC R0, 0x80` (R0's low bits carry 1 + band, the stage's URs take it off).
  Correct on the RTX 30 loop but no gain (locked 2400 MHz: 344.8-345.0 W against 344.0-344.5 W). On the RTX 40
  loop wrong in 4 of 64 tiles and 2 points of pipe lower. Dropped. (`txm`, the RTX 30 tail one ALU step shorter:
  built, not raced.)

| Loop, in turn | Watts at a locked 2400 MHz (pipe) | pxc at 350 W |
|---|---|---|
| shipped | 343.2, 340.9, 341.0 (99.32-99.33%) | 2399.8, 2397.6, 2394.8, 2396.6 |
| `bxu` | 340.6, 340.8, 340.8 (99.35-99.36%) | 2397.3, 2396.3, 2400.5, 2397.6, 2416.4 |
| `bxu bimm` | 343.2, 341.0, 340.7 (99.37%) | 2398.8, 2396.8, 2412.7, 2400.8 |

At a locked clock 8 fewer plain steps a chunk move the watts by nothing measurable (run-to-run drift is about
0.5 W); the pipe rises 0.05 points. So the price of a step is in what it reads: steps on a counter or an address
cost about nothing, where the readout's 96 XOR3 (which read the sums) cost 1.9% and the gathers' PRMT 0.6%.
At most +0.1%, under what a race can show. Not shipped. The RTX 40 loop's plain steps are at their floor: the
rest are gathers, stage and exchange traffic, the readout, the records and their parting, and the loop count.

### rig01 and Salad, 15 November: why OneZeroMiner 1.8.6 and SRBMiner 3.8.8 lead on RTX 30 and 40, and the transcript word loaded early (branch `goal/nv-k25`)

Nothing ships; the RTX 30 and RTX 40 cubins are unchanged. Competitors by ordinary observation only: run, read
their rate (their API), the card (`nvidia-smi` every second, `-q` for the limiters), and their own `--help`.

**From outside, the competitors spend less energy per unit of work.** They do the same work a clock as ours, or a
little less, and get more clock from the same power. Memory clock, power limit and voltage are the same; nothing is
set. Stand-in pool (`verify-pool.js`) at about one share a second:

| PC, limit | Miner | TH/s | MHz | W | TH/s a GHz |
|---|---|---|---|---|---|
| rig01 RTX 4090, 450 W (Windows) | ours (shipped RTX 40 fold, `hashrate.js`) | 337.51, 337.42 | 2614, 2616 | 448.7 | 129.1 |
| | OneZeroMiner 1.8.6 | 341.23 (+1.1%) | 2636 | 448.1 | 129.45 |
| rig01 RTX 4090, 250 W | ours | 258.43, 260.22, 259.72, 259.46 | 2011-2020 | 249.9 | 128-129 |
| | OneZeroMiner 1.8.6 | 271.80, 270.91 (+4.6%) | 2102, 2085 | 249.7 | 129.3-129.9 |
| Salad RTX 3080 Ti, 350 W | ours (release CLI, our core, shipped fold) | 136.02 | 1678 | 346.7 | 81.08 |
| | SRBMiner 3.8.8 | 138.57 | 1705 | 347.1 | 81.27 |
| | OneZeroMiner 1.8.6 | 137.20 | 1700 | 346.8 | 80.71 |
| Salad RTX 4080, 320 W | ours (release CLI, our core, shipped fold) | 215.80 | 2687 | 319.2 | 80.32 |
| | SRBMiner 3.8.8 | 218.89 | 2715 | 318.4 | 80.62 |
| | OneZeroMiner 1.8.6 | 217.98 | 2710 | 318.3 | 80.44 |

- The deeper the cap, the wider the gap: on the 4090 OneZeroMiner leads by 1.1% at 450 W and 4.6% at 250 W, almost
  all of it clock (+22 and +80 MHz). The same few percent less energy a unit of work buys little clock high on the
  voltage curve and nearly its whole share low on it.
- `hashrate.js` on the same Salad PCs: 81.34 TH/s a GHz on the 3080 Ti, 80.85 on the 4080. So our kernel does more
  work a clock than either competitor; the CLI's 0.3-0.6% under it is the stand-in pool's two to three shares a
  second (Kryptex gave our CLI two shares in five minutes in the official runs).
- OneZeroMiner 1.8.6's `--help`: its pearlhash kernel on SM 86/89/120 is a "compact kernel" by default,
  "regardless of memory clock" ("supports 405/810 MHz memory"). Its new `--power-save 1-3` changed nothing visible
  on the 4090 at 450 W: 2646-2648 MHz, 439-441 W against 2644 MHz, 439 W without it.
- A Salad RTX 4070 (200 W) sat on its voltage and temperature limits ("Reliability", "SW Thermal Slowdown", 83 C),
  not the power cap. There ours does the most work a clock (46.26 TH/s a GHz against 46.07 for SRBMiner and 45.71
  for OneZeroMiner) and runs 20 MHz lower, hotter: SRBMiner 123.24, ours 122.83 (`hashrate.js`), OneZeroMiner
  122.19.
- On a hold where the stand-in pool's target gave about 77 shares a second, SRBMiner (Windows) fell to 85-150 TH/s
  and OneZeroMiner lost 2%: competitor rates need a target near one share a second.

**How the table reads the competitors.** `bench.py` takes a competitor's last 15 s sample and our mean. OneZeroMiner
averages over 10 s, so its last sample jumps: on the RTX 4080 `e28b91ec` it read 204.0-204.3 for most of the run and
207.3 at the end, the table's figure. Against its mean from 150 s on, the four lowest RTX 30/40 rows are 98.1%
(`4a38a790`), 98.8% (`04f2a421`), 97.9% (`41a7e7b3`) and 98.0% (`e28b91ec`), not 96.7-97.0%. SRBMiner and PeakMiner
report long averages that include a run's first minutes, when each miner runs faster after the 8 s gap (on
`04f2a421` OneZeroMiner fell from 154 to 149 over 150 s): the miner that runs first reads high, and that can't be
taken out of their samples. Not changed here.

**The transcript word loaded early (`trl`) on the RTX 40 loop.** goal-nv50q's lever: the 5090's word came from the
L2 four gaps ahead of its reader. The RTX 40 loop (`bx bxt`) keeps its transcript in local memory too; the RTX 30
loop keeps it in shared memory. `rfold.py` `trl=1|2` now works on the RTX 40 loop: the next tail's word loaded in
gap `trg` of turn 1 or 2 on the records' scoreboard, its address from the count less one chunk (+4). Correct: probe
16,384/16,384 at k 2048 (orders 6 and 0), 4,096/4,096 at k 8192; `verify-hits.js` 291/291 and 289/289 on a 4080.

| Loop (probe, rig01, pxc, three runs each) | 450 W | 300 W |
|---|---|---|
| RTX 40 shipped | 2591 (2593, 2594, 2588) | 2266 (2264, 2268, 2267) |
| the word's LDL a move and STL gone (`dbg=3`, TIMING ONLY) | 2605 (+0.52%) | 2280 (+0.63%) |
| `trl=1` | 2592 (+0.03%) | 2262 (-0.16%) |
| `trl=2` | 2594 (+0.12%) | 2265 (-0.04%) |
| `cq=0` | 2587 (-0.16%) | 2262 (-0.19%) |
| `trl=1 cq=0` | 2584 (-0.30%) | 2259 (-0.31%) |
| two of four record LDG gone (`lgh`, TIMING ONLY) | 2592 (0) | 2264 (-0.1%) |

`trl` raises the pipe 0.21-0.24 points, and the clock gives it back: under the cap the tail's wait was free. On a
capped Salad RTX 4080 (320 W, `hashrate.js`, 100 s legs in turn): shipped 217.13, 217.16, 217.18, 217.14; A/A twin
217.18, 217.11, 217.05; `trl=1` 217.14, 217.07, 217.11; `trl=2` 217.23, 217.20; `dbg=3` 217.74, 217.74, 217.89
(+0.28%). So the RTX 40 transcript's load and store cost 0.3-0.6% of energy and only taking them out would pay; word
pairs (half of them) need a register the loop does not have (it writes all 252 the compiled function writes).
`cq=0` loses on the RTX 40 loop.

**RTX 30, priced on a capped Salad RTX 3080 Ti** (350 W, 1676-1702 MHz, `hashrate.js` 100 s legs in turn, TIMING
ONLY unless named; shipped 136.56-136.75 after the first leg, A/A twin 136.49-136.71):

| Change | TH/s | vs twin |
|---|---|---|
| no readout (`nord`) | 138.06, 138.28 | +1.1% |
| readout steps read no sums (`rdx`) | 137.31, 137.45 | +0.5% |
| no gather PRMT (`noprm`) | 137.61, 137.67 | +0.7% |
| two of four record LDG gone (`lgh`) | 137.84, 137.68 | +0.8% |
| transcript LDS/STS gone (`ntr`) | 136.92, 137.03 | +0.2% |
| B's / A's LDSM.T as LDS.128 (`ldk=1`, `ldk=2`) | 136.67, 136.67 / 136.53, 136.59 | 0 |
| `cq=0` (correct by design) | 136.70, 136.73 | 0 |
| the RTX 40 loop (B drawn once, `bx` with goal-nvk19's exchange keys; `verify-hits` 141/141) | 135.83, 135.82 | -0.5% |
| the same with `iy=0` | 132.98, 133.03 | -2.6% |

The RTX 40 loop still loses on RTX 30. The parts our loop could lose and stay correct add up to well under the
1.5-2% the competitors save on these cards.

Tools: `salad/nv34box.sh` (one box script for RTX 30 and 40: the shipped file built by `build_rot.py`, `rfold.py`
variants, `verify-hits.js`, `hashrate.js` legs in turn, smiq lines, then `bench.sh` with the release CLI on our core).

### rig01, 15 November: where the RTX 40 loop's energy goes, against OneZeroMiner at a locked clock (branch `goal/nv-k26`)

Nothing ships; the RTX 30 and RTX 40 cubins are unchanged. rig01's RTX 4090, the driver's memory clock. Competitors
by ordinary observation only (run, read their API and `nvidia-smi`).

**At the same clock OneZeroMiner draws 3.4% less power for the same work.** Stand-in pool, legs of 110 s in turn,
`nvidia-smi` every 200 ms (`S/nvk26/obs.sh`, `osum.py`):

| Leg | TH/s | MHz | W | C | J/TH |
|---|---|---|---|---|---|
| ours (shipped RTX 40 fold), locked 2010 MHz | 259.59, 259.61 | 2010 | 249.4, 253.8 | 61.8, 60.6 | 0.961, 0.978 |
| OneZeroMiner 1.8.6, locked 2010 MHz | 260.32, 260.71 | 2010 | 244.6, 245.3 | 59.5, 59.2 | 0.940, 0.941 |
| SRBMiner 3.8.8, locked 2010 MHz | 249.50 | 2010 | 235.1 | 58.5 | 0.942 |
| ours, locked 2400 MHz | 309.70 | 2400 | 340.9 | 68.0 | 1.101 |
| OneZeroMiner, locked 2400 MHz | 310.83 | 2400 | 329.3 | 67.4 | 1.060 |
| ours, 250 W cap | 259.59, 260.73 | 2004, 2019 | 249.9 | 60.6 | 0.960 |
| OneZeroMiner, 250 W cap | 271.42, 268.71 | 2096, 2070 | 249.9 | 60.3 | 0.925 |

Work a clock is the same (129.2 against 129.5-129.7 TH/s a GHz); ours draws 8.5 W more at 2010 MHz and 11.6 W more
at 2400. Under a 250 W cap that is OneZeroMiner's 3-4% more clock. It is energy a unit of work, not clock control.

**The tensor unit alone, and how it spends.** `probes/hand89/tbench.cu` (new): nothing but m16n8k32 u8 mma in the
fold's order (8 in a row share A), 16.01 cycles an mma, the whole pipe; the code is the same in every mode, only the
fragments' values differ. Locked 2010 MHz:

| tbench operands | W |
|---|---|
| random, drawn as the fold's (63 + d0 - d1) | 186.0-186.4 |
| B's eight columns equal inside each fragment | 142.0, 142.1 |
| B constant | 141.1 |
| consecutive mma share B | 186.0 |
| A constant | 188.7 |
| B uniform 0..63 (6 bits, a dense factor + 32) | 179.3, 179.3 |
| B uniform 48..79 | 185.7, 185.9 |
| A and B constant | 120.8 |

So the unit holds A and steps through B's eight columns inside each mma; the column-to-column change of B costs
44 W at full pipe, changes between mma cost nothing, and A's values cost nothing. B's values as 6-bit numbers
would save 7 W. No affine encoding of the protocol's operands flips fewer bits (best 3.488 against 3.500 a byte).

**The loop on the same footing.** Probe (`gprobe`, the shipped RTX 40 loop `c89`), locked 2010 MHz, 30 s runs,
two rounds; run-to-run drift is about 3 W. Operand data from a `PD_INPUT` file (`probes/hand89/mkin.py`, new), so
the code and the sums are the real ones (checks 512/512):

| Data | W |
|---|---|
| true random (as the miner's) | 253.9, 254.2 |
| B's factors constant (B' = 63) | 200.9, 201.4 |
| A's constant | 244.7, 245.0 |
| B's columns equal in eights (one n8 fragment = one column) | 209.3, 209.8 |
| B's columns equal in pairs | 230.0, 229.8 |
| a column slot's eight n8 tiles equal | 248.9, 247.8 |
| A's rows equal in eights / a band's six m16 tiles equal | 248.0, 245.7 / 250.3, 246.8 |
| every k with the same rank indices | 237.7, 237.4 |

The loop draws 254 W where the tensor unit alone draws 186 W: about 68 W (27%) is everything that is not the mma.
Of that about 17 W follows the operand values as they move (A constant -9 W where tbench shows A's values cost the
unit nothing; B -53 W against tbench's -45 W). If OneZeroMiner's tensor cost is ours (same data, same
instruction), its non-mma part is about 9 W (13%) smaller.

**What the parts cost at the 250 W cap** (probe, two rounds, TIMING ONLY unless named; c89 1966-2000 MHz at 99.18%
of the pipe; gain in pipe x clock):

| Change | Clock (MHz) | Gain |
|---|---|---|
| no readout (`nord`) | 2074, 2095 | +5.8% |
| readout steps read no sums (`rdx`) | 2030, 2018 | +2.9% |
| all gather loads as moves (`nold=1`, operand values change too) | 2106, 2103 | +7% |
| B's gather loads as moves (`nold=3`) | 2020, 2020, 2023, 2035 | +2.5% |
| A's (`nold=2`) | 1992, 1978 | +0.4% |
| no PRMT (`noprm`) | 2063, 2032, 2050, 2054 | +3.0% (pipe -0.5 points) |
| B's PRMT only (`noprm=2`, new) | 2006, 2033 | +1.0% (pipe -0.3) |
| B's gathers only two IADD3 from moves (`noprm=2 nold=3`) | 2057, 2053 | +3.6% |
| no transcript load/store (`dbg=3`) | 2003, 1985 | +1.1% |
| two of four record loads gone (`lgh`, correct) | 1976, 1971 | 0 |
| B constant (`bcz`, new without `b8`) | 2340 | +17% |
| B's columns equal in eights (data) | 2305 | +15% |
| a slot's n8 tiles equal (data) | 2063 | +3% |

At a deep cap the readout (5.8%), the PRMT (3%) and B's gather loads (2.5%) are the big movable parts, but each is
a minimum of this design: 96 three-input XOR for a thread's 192 sums, a PRMT pair for each fragment word (sm_89's
ldmatrix moves 16-bit elements, so a gather hands a lane two rows at two k), and B drawn once already halves B's
gathers.

**A design these prices point to, not built.** Group the chunk's k by B's rank index: a chunk's sums are
`63 Σk A' + Σp G[r][p] D_B[c][p]` with `G[r][p] = Σ(k: p0=p) a - Σ(k: p1=p) a` (A's fill 0, which no share's credit
reads) and `D_B` the dense factor biased +32 (the bias adds 32 Σp G = 0). The B operand is then `D_B`: fixed for the
job, four consecutive p bytes a register, a plain load with no gather, no subtract, no PRMT and no exchange, and
6-bit values (-7 W in the unit). Priced from the table: B's side about +6% at 250 W. Its costs: A's side becomes
sums of a variable count of terms (each p holds Poisson(2) k; padded, 2 to 4 times today's A gathers, +1.5-3%),
`G` leaves int8 in about 0.4% of entries (98 a tile a chunk at k 65536 by count; each needs a 256-column fix on the
CUDA cores, perhaps 1-2%), the row term `63 Σk A'` (a dense-factor dot product a row a chunk, once a block), and
new offsets from the host. Net by these prices +1 to +3%, with large error bars: worth a timing-only skeleton
before anything else.

Tools: `S/nvk26/obs.sh` (competitor and our legs at a locked clock or a cap), `run26.py` (`run24.py` with `!env`
lines for `PD_INPUT`), `probes/hand89/tbench.cu` and `mkin.py`; `gfold.py` `acz`/`bcz` now also work without `b8`.

## RTX 50-series

Vast, 10 October, on the CUDA 13 core except the India 5090 (driver 570, CUDA 12.8 core).

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

Both 5070 Ti hosts lock the clock at 1342 MHz, far below a stock card.

### Salad, 29–30 October (PeakMiner 2.20.3, SRBMiner 3.8.0)

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

### Salad, 30 October (SRBMiner 3.8.2)

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

### Salad, 9–10 November (v0.6.13; PeakMiner 2.20.7, SRBMiner 3.8.2, OneZeroMiner 1.8.5)

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | OneZeroMiner | % of best |
|---|---|---|---|---|---|---|---|
| RTX 5090 | Salad (575 W) | 4b8c60bb | 371.3 | 417.1 | 410.5 | 424.7 | 87% |
| RTX 5090 | Salad (450 W) | 2210b832 | 349.9 | 407.3 | 406.3 | 408.3 | 86% |
| RTX 5080 | Salad (360 W) | 183c762b | 224.3 | 235.2 | 225.6 | 236.1 | 95% |
| RTX 5080 | Salad (360 W) | 38fd5fd3 | 187.6 | 196.0 | 196.4 | 192.7 | 96% |
| RTX 5070 Ti | Salad (300 W) | 23e86137 | 160.7 | 167.7 | 168.9 | 167.6 | 95% |
| RTX 5070 Ti | Salad (300 W) | 51e7db39 | 187.8 | 196.0 | 196.8 | 197.7 | 95% |
| RTX 5070 | Salad (200 W) | 75cb2a16 | 124.7 | 133.4 | 133.5 | 133.7 | 93% |
| RTX 5070 | Salad (200 W) | 749878e7 | 126.2 | 134.3 | 134.5 | 136.9 | 92% |
| RTX 5060 Ti | Salad (180 W, 16 GB) | f5eacc1c | 93.4 | 96.5 | 96.6 | 96.7 | 97% |
| RTX 5060 Ti | Salad (180 W, 16 GB) | 5ef6ad18 | 92.9 | 97.7 | 101.7 | 97.9 | 91% |
| RTX 5060 | Salad (145 W) | b784c119 | 78.9 | 83.9 | 83.9 | 84.3 | 94% |
| RTX 5060 | Salad (135 W) | 796c35b1 | 73.4 | 77.9 | 78.4 | 77.3 | 94% |

The 5090 trails most, as before.

Which core ran (#290 item 15): every box here reported a driver of 580 or newer (595.97 to 621.14; the two 5090s
621.01 and 621.14), so the app picked the CUDA 13 core on all twelve. None should have been on the CUDA 12.8 core,
which is 3.2% slower on a 5090. The bench didn't log the core itself, so a CUDA 13 core that failed to start and fell
back would not show; nothing points to that. So the 5090s' 86–87% is the CUDA 13 core's real gap on these boxes, not
an old driver. bench.sh now logs our CLI's "Pearl core:" line as "loaded ...", and bench.py keeps it per row.

### Salad, 12–13 November: the official table until 14 November (PR #293 build; PeakMiner 2.20.7, SRBMiner 3.8.7, OneZeroMiner 1.8.5)

Replaced by the 14 November table at the top of this file. Ours mined with the rotation fold; "Ours, old kernel" is the same build without it.

| Card | Host | Machine ID | Ours | Ours, old kernel | PeakMiner | SRBMiner | OneZeroMiner | % of best | Fold gain |
|---|---|---|---|---|---|---|---|---|---|
| RTX 5090 | Salad (600 W) | 6043ec7a | 430.6 | 414.8 | 432.0 | 443.9 | 423.6 | 97.0% | +3.8% |
| RTX 5090 | Salad (600 W) | 0e704bd1 | 427.9 | 415.4 | 425.9 | 445.5 | 427.9 | 96.1% | +3.0% |
| RTX 5080 | Salad (360 W) | 0467cb92 | 234.1 | 221.5 | 234.9 | 232.8 | 228.7 | 99.7% | +5.7% |
| RTX 5080 | Salad (396 W) | 3363804f | 241.5 | 231.3 | 240.8 | 244.5 | 241.4 | 98.8% | +4.4% |
| RTX 5070 Ti | Salad (300 W) | 0761bce7 | 202.4 | 192.1 | 202.5 | 204.8 | 200.1 | 98.8% | +5.3% |
| RTX 5070 Ti | Salad (300 W) | 90649426 | 200.2 | 189.7 | 198.6 | 200.2 | 200.0 | 100.0% | +5.5% |
| RTX 5070 | Salad (250 W) | 1124253e | 137.7 | 130.1 | 137.5 | 138.3 | 138.0 | 99.6% | +5.9% |
| RTX 5070 | Salad (250 W) | dcb9402a | 130.0 | 126.6 | 133.7 | 130.2 | 129.7 | 97.3% | +2.7% |
| RTX 5060 Ti | Salad (180 W, 16 GB) | 34780021 | 102.1 | 96.7 | 101.2 | 102.2 | 102.1 | 99.9% | +5.6% |
| RTX 5060 Ti | Salad (180 W, 16 GB) | de92bd0e | 94.9 | 91.4 | 95.7 | 97.1 | 95.7 | 97.7% | +3.7% |
| RTX 5060 | Salad (145 W) | dd173265 | 77.0 | 73.5 | 77.7 | 77.1 | 77.1 | 99.2% | +4.7% |
| RTX 5060 | Salad (145 W) | 11d68c15 | 85.6 | 80.9 | 85.6 | 86.6 | 85.9 | 98.9% | +5.9% |

### Salad, 11 November: the tensor-pipe probe on RTX 30, 40 and 50 cards (PR #290 goal work)

The 4090 study ("What fills the tensor pipe", Research notes) is one card. This is its second probe,
`earn/native/probes/pipeprobe2.cu`, on Salad PCs of the other architectures: sm_86 (RTX 3090, 3060), sm_89 (RTX 4070)
and sm_120 (RTX 5090, 5070 Ti, 5070). **Timing only: the streams compute nothing real, and a figure "by count" is not
a mining rate.** Branch `goal/nv-probe-cards`: the probe's new switches, the box script
`earn/tuning/salad/pipebox.sh`, and the boxes' logs with the scripts that made these tables in
`earn/tuning/pipeprobe/`.

How it was run. One PC a card. The probe was built on the box for the card's architecture, with CUDA 12.8 for sm_86
and sm_89 and CUDA 13.3 for sm_120 (the toolkits our two cores ship with), and for sm_120 again with 12.8. Each warp
times itself with the SM's cycle counter; pipe use is 16 cycles an IMMA a scheduler over the cycles taken. Every
stream ran 1 s for its pipe use. The main ones ran 30 s with the card read every 0.8 s: clock, watts and flags are
means after the first 8 s, and the rate by count is IMMA x 4096 over the time of the launches after the first 8 s.
The fragments are loaded from mostly-zero bytes as in the study's last round, and again from hashed bytes: the bytes
change the watts, not the pipe use. Before the probe, on the same PC in the same session: OneZeroMiner 1.8.5 and
our release v0.6.15 on the stand-in pool (60 s warm-up, 3 minutes each, the mean of each miner's own figure from
60 s in), then the release's core under `hashrate.js` (180 s after 20 s).

| Card | PC | SMs | Power limit | What held it |
|---|---|---|---|---|
| RTX 3090 | 71128c8f | 82 | 390 W | Heat: 86 to 88 C with the thermal-slowdown flag on in every run, at 340 to 359 W |
| RTX 3060 | 89183448 | 28 | 170 W | The limit, once the bytes are hashed or the draw is in; 1962 to 1987 MHz below it |
| RTX 4070 | 1aaa2daa | 46 | 200 W | The same; 2700 to 2760 MHz below the limit |
| RTX 5090 | a8a49612 | 170 | 600 W | Watts; 2880 MHz until a stream nears the limit |
| RTX 5090 | e57a5671 | 170 | 450 W | Watts, in every stream but the lightest (probe only) |
| RTX 5070 Ti | 3f7ee3a4 | 70 | 300 W | Its clock: 2805 to 2857 MHz from 166 to 296 W |
| RTX 5070 | 949fe6d1 | 48 | 250 W | Its clock: 2887 to 2909 MHz from 140 to 215 W (probe only) |

Two of the RTX 50 PCs were not clean. On the 5070 Ti every stream, even one warp alone on an SM, lost about 6% of its
cycles (17.0 cycles an IMMA where the 5090 takes 16.0), and both miners ran about 6% under the 5070 Ti of the gate
(10 November). Something else was using that card: it had 1130 MiB in use and 3% load before we started (inferred,
not seen). The 5070 lost 0.7% the same way. Their rows below are shares of their own loads-only row.

**What a full pipe is.** IMMA only unless said; cycles an IMMA a scheduler, and pipe use at 16.

| Warps in the block | RTX 3090 | RTX 3060 | RTX 4070 | RTX 5090 |
|---|---|---|---|---|
| One warp alone on the SM | 16.01 (99.93%) | 16.01 (99.93%) | 16.01 (99.93%) | 16.10 (99.35%) |
| 4 (one a scheduler) | 16.04 (99.78%) | 16.01 (99.93%) | 16.01 (99.93%) | 16.11 (99.32%) |
| 8 (two a scheduler) | 16.01 (99.93%) | 16.01 (99.93%) | 16.01 (99.93%) | 16.01 (99.93%) |
| 12 (three), 32 IMMA a k-step | 16.02 (99.90%) | 16.02 (99.90%) | 16.02 (99.90%) | 16.00 (99.99%) |
| 16 (four), 24 IMMA a k-step | 16.02 (99.87%) | 16.02 (99.87%) | 16.02 (99.87%) | 16.00 (99.99%) |
| 4, with the fragment loads | 16.14 (99.13%) | 16.14 (99.13%) | 16.16 (99.03%) | 16.05 (99.68%) |
| 8, with the fragment loads | 16.00 (100.00%) | 16.00 (100.00%) | 16.00 (100.00%) | 16.00 (100.00%) |
| Warps 0 and 4 only, each warp's own cycles an IMMA | 32.02 | 32.02 | 32.02 | 32.02 |
| Warps 0 and 1 only (and 0 with 2, 3, 5, 6 or 7) | 16.01 | 16.01 | 16.01 | 16.10 |

- A pipe is one IMMA every 16.0 cycles a scheduler on all three architectures, an SM has four, and warp w runs on
  scheduler w % 4 (warps 0 and 4 share a pipe; 0 and any other do not). The SM's cycle counter agrees with
  nvidia-smi's clock within 0.6% on every PC but the 450 W 5090 (below).
- Two warps a scheduler fill it everywhere: 99.93% on IMMA alone, 100.00% with the fragment loads. Three or four
  add nothing.
- One warp a scheduler also holds 99.3 to 99.9% of an IMMA-only stream, on sm_89 too. The study's "one warp alone
  gets 91%" was read with its first probe, whose run-time tests stand between the IMMA; this probe has none
  (the cause is inferred: I did not rerun the first probe). What holds is narrower: on sm_86 and sm_89 a lone
  warp loses 0.8 to 0.9 points as soon as the four ldmatrix a k-step are in its stream, and two warps lose nothing.

**Pipe use by stream** (1 s runs; 8 warps, two a scheduler, 48 IMMA a k-step, 4 k-steps a chunk).

| Stream | RTX 4090 (the study) | RTX 3090 | RTX 3060 | RTX 4070 | RTX 5090, CUDA 13.3 | RTX 5090, CUDA 12.8 | RTX 5070 Ti, 13.3, of its loads row | RTX 5070, 13.3, of its loads row |
|---|---|---|---|---|---|---|---|---|
| IMMA only | 99.93% | 99.93% | 99.93% | 99.93% | 99.93% | 99.99% | 99.7% | 99.8% |
| + fragment loads (kept in every row below) | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100% | 100% |
| + readout, one run of 96 XOR3 | 98.72% | 98.75% | 98.75% | 98.71% | 98.33% | 97.67% | 98.2% | 98.5% |
| + readout, three runs of 32 | 99.67% | 99.68% | 99.68% | 99.68% | 98.19% | | 98.5% | 97.9% |
| the same with a shared XOR | 99.71% | 99.71% | 99.71% | 99.71% | 98.40% | 98.21% | 98.3% | 98.6% |
| + readout, six runs of 16 | 99.68% | 99.60% | 99.68% | 99.68% | 97.94% | | 97.8% | 98.1% |
| + 4 draw units a k-step, after the 24th IMMA | 99.61% | 99.52% | 99.52% | 99.56% | 96.92% | 96.39% | 96.9% | 96.7% |
| the same at the k-step's end | 99.52% | 99.71% | 99.71% | 99.61% | 96.99% | | 96.9% | 97.1% |
| the same in two halves / one unit every 12 IMMA | 98.44% | 97.03 / 97.37% | 97.36% | 97.30% | 96.34% | | 96.2% | 96.4% |
| + 4 units that read their offsets first | 98.80% | 99.03% | 99.03% | 98.95% | 96.47% | 97.82% | 96.2% | 96.3% |
| loads, three-run readout, 4 such units | 95.15% | 95.17% | 95.17% | 95.15% | 94.82% | 95.26% | 94.6% | 94.8% |
| the same with one shared add and test a k-step | 94.71% | 95.23% | 95.22% | 95.27% | 93.93% | 95.04% | 93.7% | 94.1% |
| the same with 3 units | 96.97% | 96.73% | 96.73% | 96.76% | 95.03% | 96.32% | 94.8% | 94.7% |

- **sm_86 and sm_89 repeat the 4090 to a few hundredths of a point**, on three cards from 28 to 82 SMs. The
  compiler gives sm_86 and sm_89 the same loop (same counts, same order: two XOR3 after each IMMA in a readout run).
  Where a row differs from the study's by half a point or more (the draw in halves, the offsets-first units, the
  whole stream with 4 units), the build differs: CUDA 12.8 here, 12.6 there.
- **sm_120 does not.** The readout costs 1.6 to 2.1 points however it is cut (0.3 on the others), four draw units 3.1
  to 3.6 (0.4 to 0.5), and the whole stream holds 93.9 to 96.3%. Both compilers, three cards. Both give sm_120 a
  different order (six or seven XOR3 at a time between two IMMA), so part of this may be the order and not the chip;
  the box cannot run an sm_89 build on an sm_120 card to tell.

**What a step between the IMMA costs, by count and kind.** Points of pipe use lost against the same PC's loads-only
row. A draw unit is 18 instructions (7 adds, 4 byte selects, 2 XOR, 4 shared loads, 1 shared store).

| Stream | RTX 3060 | RTX 4070 | RTX 5090 (450 W PC), 13.3 | RTX 5070, 13.3 (12.8) |
|---|---|---|---|---|
| 1 draw unit a k-step | 0.07 | 0.02 | 0.78 | 0.27 (0.64) |
| 2 draw units | 0.02 | 0.16 | 1.55 | 1.12 (1.30) |
| 3 draw units | 0.11 | 0.11 | 2.00 | 2.16 (2.88) |
| 4 draw units | 0.48 | 0.44 | 3.08 (600 W PC) | 3.23 (5.73) |
| One-run readout kept in 4 / 8 / 16 XOR chains | 0.84 / 1.25 / 2.68 | 0.84 / 1.29 / 2.68 | 1.54 / 1.67 (600 W PC) / 2.36 | 1.32 / 1.52 / 2.05 |
| Three-run readout kept in 4 / 8 / 16 chains | 0.58 / 0.29 / 0.71 | 0.58 / 0.29 / 0.71 | 1.77 / 1.60 / 2.03 | 1.77 / 1.43 / 1.54 |

| One kind of step after every IMMA, loads kept | RTX 4070 Super (PC 7f4f5e71), 12.8 | RTX 5070, 13.3 (12.8) |
|---|---|---|
| An independent add every 2 IMMA | 0.09 | 1.06 (0.51) |
| An independent add an IMMA | 0.36 | 2.06 (2.33) |
| 2 independent adds an IMMA | 0.54 | 4.63 (5.06) |
| An add an IMMA, each waiting on the one before | 0.42 | 1.46 (4.85) |
| An XOR3 an IMMA of two sums the IMMA 8 earlier wrote | 3.40 | 1.54 (1.04) |
| 2 such XOR3 an IMMA | 4.57 | 5.68 (6.37) |

| An add an IMMA, IMMA only, RTX 5070, 13.3 | Without the add | With it | With 2 adds an IMMA |
|---|---|---|---|
| One warp a scheduler | 98.28% | 95.80% | |
| Two | 99.29% | 96.34% | 94.10% |
| Three (24 IMMA a k-step) | 99.25% | 97.09% | 94.48% |
| Four (16 IMMA a k-step) | 99.10% | 96.81% | 94.96% |

- **On sm_120 a plain add between the IMMA costs a third to a half of a cycle (0.33 to 0.50), wherever it is and
  however many warps share the scheduler. On sm_89 it costs a twentieth or less (0.03 to 0.06).** That one
  difference accounts for the draw: a unit costs about 6 cycles on sm_120 and under 1 on sm_86 and sm_89, and up to
  three units a k-step are free on those two. It is linear in the count on sm_120, so there is nothing to weave
  into: each instruction is paid for.
- A NOP is not paid for: ptxas 12.8 puts one after every IMMA of the sm_120 IMMA-only loop, and it holds 99.99%.
- Three or four warps a scheduler do not absorb the adds on sm_120 (2.2 to 3.0 points for one add an IMMA at any
  warp count), so more warps is not the way out there.
- An XOR3 of sums an IMMA wrote 8 IMMA earlier is dear on sm_89 too when there is one after every IMMA (0.55 cycle
  each), yet the three-run readout's XOR3, placed 9 to 25 IMMA after their sums' last IMMA, cost about 0.1
  each. Distance from the IMMA that wrote the sum seems to matter. Not followed up.
- The same pass had a shared load and a shared store after every IMMA. I left their rows out: my addresses put four
  lanes on one bank, so they price a conflicted access (9 to 18 points on both architectures), not a clean one.

**Clock, watts and rate by count** (30 s runs; T-MAC/s by count beside the mining rates on the same PC).

| Stream | RTX 3090, 390 W, heat-held | RTX 3060, 170 W | RTX 4070, 200 W |
|---|---|---|---|
| IMMA only | 1844 MHz, 340 W, 154.7 | 1987 MHz, 105 W, 56.8 | 2760 MHz, 133 W, 129.5 |
| + fragment loads | 1777, 344 W, 148.9 | 1987, 125 W, 56.8 | 2758, 150 W, 129.4 |
| + readout, one run | 1754, 346 W, 145.3 | 1977, 130 W, 55.9 | 2745, 161 W, 127.4 |
| + readout, three runs | 1744, 345 W, 146.2 | 1972, 131 W, 56.3 | 2745, 162 W, 128.7 |
| + 4 draw units | 1662, 354 W, 138.7 | 1968, 161 W, 56.1 | 2715, 196 W (cap flag at times), 127.0 |
| + 4 units, offsets first | 1635, 353 W, 136.0 | 1965, 167 W, 55.7 | 2715, 198 W (cap flag), 125.9 |
| all together, 3 units | 1643, 348 W, 133.3 | 1965, 162 W, 54.4 | 2700, 191 W, 122.7 |
| all together, 4 units | 1622, 352 W, 129.6 | 1962, 170 W (cap), 53.4 | 2679, 200 W (cap), 120.1 |
| loads, hashed bytes | 1644, 349 W, 137.9 | 1965, 162 W, 56.2 | 2730, 184 W, 128.1 |
| three-run readout, hashed bytes | 1620, 351 W, 135.5 | 1964, 170 W (cap), 56.0 | 2700, 184 W, 126.5 |
| 4 draw units, hashed bytes | 1548, 359 W, 129.2 | 1897, 170 W (cap), 54.1 | 2615, 200 W (cap), 122.6 |
| loads again, last (the run's drift) | 1747, 331 W, 146.3 | 1968, 126 W, 56.3 | 2745, 151 W, 128.7 |
| Our release v0.6.15, `hashrate.js` | 125.8 TH/s, 1571 MHz, 370 W | 50.6 TH/s, 1845 MHz, 170 W | 115.9 TH/s, 2568 MHz, 200 W |
| OneZeroMiner 1.8.5 | 131.8 to 137.2 TH/s, 1590 to 1658 MHz, 354 to 378 W | 54.9 TH/s, 1938 MHz, 170 W | 123.1 TH/s, 2652 MHz, 200 W |
| Release / OneZeroMiner, share of the pipe at their clocks | 95.3% / 98.5 to 98.7% | 95.6% / 98.7% | 95.8% / 98.5% |

| Stream | RTX 5090, 600 W | RTX 5090, 450 W (the SM's counter in MHz) | RTX 5070 Ti, 300 W | RTX 5070, 250 W |
|---|---|---|---|---|
| IMMA only | 2880 MHz, 445 W, 501.5 | | 2857 MHz, 167 W, 192.3 | 2909 MHz, 140 W, 141.9 |
| + fragment loads | 2880, 485 W, 501.6 | 2837, 424 W, 492.9 | 2857, 178 W, 192.7 | 2895, 150 W, 141.4 |
| + readout, one run | 2880, 531 W, 492.9 | | 2827, 191 W, 187.4 | |
| + readout, three runs | 2880, 523 W, 493.2 | 2816, 450 W, 481.8 | 2827, 191 W, 187.7 | 2895, 164 W, 139.1 |
| + 1 / 2 draw units | | 2771 / 2691, 450 W, 479.5 / 462.0 | | |
| + 4 draw units | 2710, 547 W (cap flag), 456.5 | 2573, 450 W, 433.3 | 2812, 243 W, 183.5 | 2887, 212 W, 136.5 |
| + 4 units, offsets first | 2596, 504 W (cap flag), 435.8 | 2519, 450 W, 421.7 | 2812, 251 W, 182.5 | |
| all together, 3 units | 2662, 529 W (cap flag), 440.4 | 2564, 450 W, 423.5 | 2812, 245 W, 179.8 | 2887, 215 W, 133.8 |
| all together, 4 units | 2547, 493 W (cap flag), 416.4 | | 2810, 263 W, 177.9 | |
| loads, hashed bytes | 2827, 600 W, 492.1 | 2654, 450 W, 461.2 | 2812, 225 W, 189.5 | 2887, 196 W, 140.9 |
| three-run readout, hashed bytes | 2790, 600 W, 477.7 | | 2812, 239 W, 186.3 | |
| 4 draw units, hashed bytes | 2601, 600 W, 438.6 | 2259, 450 W, 380.0 | 2805, 296 W (cap), 183.4 | 2840, 250 W, 134.2 |
| all together, 3 units, CUDA 12.8 build | 2624, 514 W (cap flag), 439.9 | | 2814, 245 W, 181.0 | |
| loads again, last | 2880, 486 W, 501.6 | | 2857, 179 W, 192.6 | 2895, 152 W, 141.4 |
| Our release v0.6.15, `hashrate.js` (CUDA 13 core) | 425.9 TH/s, 2571 MHz, 600 W | | 179.4 TH/s, 2805 MHz, 299 W | |
| OneZeroMiner 1.8.5 | 429.8 TH/s, 2532 MHz, 510 W | | 184.8 TH/s, 2805 MHz, 275 W | |
| Release / OneZeroMiner, share of the pipe at their clocks | 95.2% / 97.5% | | 94.6% / 97.4% of this PC's loads row | |

A first 5090 PC (892b7450, 522 W limit) ran the miners only before it went silent: OneZeroMiner 422.3 TH/s at
2504 MHz and 486 W, ours 406.2 TH/s at 2453 MHz and 522 W.

**By architecture**

- **sm_86 (RTX 30) and sm_89 (RTX 40): the 4090's rules hold, and so does its reading of the competitors.**
  Two warps a scheduler fill the pipe (100.00%). The three-run readout costs 0.3% of the pipe and 6 to 12 W; four
  draw units cost 0.5% of the pipe and 36 to 47 W (a quarter to a third of the loads stream's watts). Watts are what
  these cards run out of: with hashed bytes the draw-only stream sits on the limit at 1897 MHz (3060) and 2615 MHz
  (4070), 3.5 and 4.2% under the clock of the loads stream on the same bytes. That stream is OneZeroMiner's
  signature on both PCs, as it was on the 4090: 54.1 by count at 1897 MHz and 170 W against 54.9 TH/s at 1938 MHz
  and 170 W, and 122.6 at 2615 MHz and 200 W against 123.1 TH/s at 2652 MHz and 200 W. On the heat-held 3090: 129.2
  at 1548 MHz against 131.8 to 137.2 TH/s. Against our release the stream is +7.0% (3060), +5.8% (4070) and +2.7%
  (3090).
- So on RTX 30 and 40 the design carries over as it is, and it buys parity, not a lead: their whole miner is level
  with or just above our draw-only stream, which has no readout in it. A lead needs the two things the study already
  named, and these PCs put numbers on them: the order pinned so the pieces together keep what they keep apart (the
  whole stream with 3 units holds 96.7%, where the three units alone hold 99.7 to 99.8% and the readout alone 99.7%:
  about 2.8 points), and a draw that costs fewer watts than the probe's unit (each watt saved is clock on every one
  of these cards).
- **sm_120 (RTX 50): the pipe fills the same way, and nothing between the IMMA is free.** Two warps a scheduler hold
  100.00%, at 16.0 cycles an IMMA. But the readout costs 1.6 to 2.1% of the pipe and four draw units 3.1 to 3.6%,
  because every plain instruction costs a third to a half of a cycle there. The whole stream as the compiler orders it
  holds 93.9 to 96.3%, about what our release holds (94.6 to 95.2%), where OneZeroMiner holds 97.4 to 97.5% on these
  PCs and the three competitors 98.2 to 99.1% on the gate's 5070 Ti.
- What limits an RTX 50 card differs by card. The 5070 Ti and the 5070 are held by their clock: it moves 1.8% and
  0.8% over every stream short of the limit, so there a stream is worth its pipe use and little else. The 5090 is
  held by watts: loads on hashed bytes already draw the 600 W limit (2827 MHz against a 2880 MHz top), the
  three-run readout adds 38 W, and four draw units take 8% of the clock at 600 W and 15% at 450 W.
- By count the whole stream with 3 units gives 440 on the 600 W 5090 against 425.9 TH/s for our release and
  429.8 for OneZeroMiner (mostly-zero bytes; the draw alone on hashed bytes gives 438.6), 423.5 on the 450 W 5090
  (380.0 for the draw alone on hashed bytes; the benchmark table's 450 W 5090 has the competitors at 406 to 408 and
  us at 349.9 on v0.6.13), and 180 to 181 on the 5070 Ti against 179.4 and 184.8. **So on RTX 50 the 4090 design, as
  the probe has it, is not enough**: about level with the release on the clock-held cards, and short of the
  competitors' watts on a 450 W 5090. It needs a stream with far fewer instructions an IMMA, or an order we have not
  found. The competitors' 98 to 99% on the 5070 Ti says one exists.

**The RTX 5070 Ti: why free operands gained nothing.** The card is held by its clock, not its watts. Under a 300 W
limit every stream ran at 2805 to 2857 MHz, from 166 W to 296 W. Free operands save watts (54 W on the gate's PC), and
on this card watts buy no clock. Its 5% gap to the competitors is pipe time: we hold about 94 to 95% of the pipe at
that clock and they hold 98 to 99%. And on sm_120 pipe time is not got back by weaving: the readout and the draw each
cost real cycles there.

**Not verified**

- Why a plain instruction costs a third to a half of a cycle on sm_120, and whether the compiler's sm_120 order or the chip is
  behind it. How the competitors hold 98 to 99% there.
- On the 600 W 5090 the streams with draw units ran with the power-cap flag on at 493 to 547 W and 2547 to 2710 MHz,
  below the limit; the hashed-byte ones drew the full 600 W. The competitors look the same on three 5090s (2470 to
  2530 MHz at 486 to 513 W under limits of 522 to 600 W; the third is the gate's). On the 450 W 5090 every capped
  stream drew exactly 450 W and the instant draw stayed within 13 W of it. I don't know what limit the 600 W PC's
  streams hit.
- On the 450 W 5090 (driver 581.57) nvidia-smi's clock read up to 15% above the SM's own cycle counter under the cap
  (2602 against 2259 MHz), so a rate over nvidia-smi's clock understates pipe use on such a PC. The other PCs agree
  within 0.6%. The competitors' shares of the pipe above use nvidia-smi's clock.
- What a real fold's operand bytes cost in watts: the probe has mostly-zero bytes and hashed bytes, not a draw's.
- One PC a card and one 30 s run a stream. The loads stream run again at the end moved by -1.7% of clock on the
  3090 (heat), -1.0% on the 3060, -0.5% on the 4070 and 0.0% on the RTX 50 cards. On the 3090 that last run read
  99.66% where the three before it read 100.00%; I don't know why.
- The 3090 PC is hot: both miners and every stream ran under thermal slowdown, and the miners' order matters there
  (OneZeroMiner read 131.8 TH/s on a hot card and 137.2 after the box's restart).
- The 5070 Ti PC's 6% and the 5070's 0.7% are read as another load from what the card showed at idle. The miners
  were not run on the 5070, the 450 W 5090 or the 4070 Super.
- On the RTX 50 PCs with driver 616 to 621 a flag bit 0x400 was set in every run; I can't name it.
- The miners' figures are their own displays. The stand-in pool's credited rate over 3 minutes (100 to 1,100
  shares) is too coarse to check them.

Box time: 3.1 box-hours (RTX 3090 31 min, RTX 5090 60 min over three PCs, RTX 5070 Ti 27 min, RTX 3060 24 min,
RTX 4070 28 min, RTX 5070 16 min).

### Salad, 11 November: what holds the tensor pipe on RTX 50 (PR #290 goal work)

The probe round above left one question: on sm_120 a plain instruction between two IMMA costs a third to a half of
a cycle, so the stream holds 94 to 96% of the pipe where the three competitors hold 98 to 99%. Is that the chip or
the order ptxas picks? **It is the order.** Same probe, `earn/native/probes/pipeprobe2.cu`, with switches that place
steps; **timing only: the streams compute nothing real, and a figure "by count" is not a mining rate.** Branch
`goal/nv-probe-cards`; logs in `earn/tuning/pipeprobe/` (`logs2-*`).

How it was run. RTX 5070 Ti PCs, CUDA 13.3, 8 warps (two a scheduler), 48 IMMA a k-step, 1 s runs; pipe use is 16
cycles an IMMA a scheduler over the cycles taken. Three PCs that held 100.00% on the loads-only stream (acefdd76,
801f7b6a, f54436e4) agreed on every stream as compiled to 0.01 point, and the rows below are theirs unless said;
three other PCs had another load on the card (98.4 to 99.9% on loads only) and were left or are marked. ptxas
moves steps, so each build's loop was read back from the binary. To get orders the source cannot ask for,
`earn/tuning/pipeprobe/perm.py` moves whole instructions inside the compiled loop: it keeps every register and
shared-memory dependency, clears the operand-reuse flags (the card takes a flagged operand from a latch without
checking which register the latch holds), and the probe launches the reordered cubin. A checksum of the kernel's
output says whether a reordered loop computed what the compiler's did.

**The three competitors, from outside** (normal runs through the share relay, 100 s each; PeakMiner 2.20.7,
SRBMiner 3.8.2, OneZeroMiner 1.8.5). Same outline as on the RTX 4090.

| Card, PC, limit | Miner | m x n x k | Tile | Memory controller | W | SM MHz | Shown TH/s | Of the tensor peak |
|---|---|---|---|---|---|---|---|---|
| RTX 5070 Ti, eaa66a4e, 300 W | PeakMiner | 30720 x 16384 x 65536 | 24 x 8 | 0% | 300 | 2865 | 202.8 | 98.7% |
| | SRBMiner | 16514 x 62720 x 65536 | 24 x 8 | 0% | 300 | 2857 to 2880 | 200.4 | 97.6% |
| | OneZeroMiner | 13440 x 32768 x 65536 | 24 x 8 | 0 to 1% | 300 | 2865 to 2880 | 200.3 | 97.3% |
| RTX 5090, 218578cc, 800 W | PeakMiner | 30720 x 16384 x 65536 | 24 x 8 | 0% | 470 | 2527 | 436.4 | 99.2% |
| | SRBMiner | 16514 x 62720 x 65536 | 24 x 8 | 0% | 469 | 2505 to 2512 | 429.9 | 98.4% |
| | OneZeroMiner | 32640 x 32768 x 65536 | 24 x 8 | 0% | 463 | 2520 to 2527 | 429.1 | 97.7% |

- 24 x 8 is 192 sums a thread: 48 IMMA a k-step a warp and two warps a scheduler, as on the 4090. k 65536, rank
  128, fill 64, no memory traffic, 0.3 to 1.0 GiB in use. SRBMiner's tile was 10 x 16 on the 4090.
- So they do not get their 98 to 99% from another shape. The rate is their own display; the stand-in pool's
  credit over 100 s is too coarse to check it.

**What a step costs by its place** (cycles an IMMA a scheduler; 16.000 is a full pipe).

| Stream | Order | Cycles an IMMA | Pipe use |
|---|---|---|---|
| Loads only | | 16.000 | 100.00% |
| 192 independent adds a chunk | as compiled: 7 in each of 27 gaps | 16.323 | 98.02% |
| | reordered: 1 after every IMMA (PC with another load: 16.098 without the adds) | 16.131 | |
| | reordered: 2 in each of 89 gaps | 16.167 | 98.97% |
| | reordered: 3 in each of 56 gaps | 16.167 | 98.97% |
| | reordered: 4 in each of 43 gaps | 16.317 | 98.06% |
| | reordered: 8 in each of 19 gaps | 16.549 | 96.68% |
| | reordered: one run of 184 | 16.672 | 95.97% |
| 192 multiplies, shifts or byte selects | as compiled: 7 a gap | 16.323 each | 98.02% |
| | reordered: 1 after every IMMA (the PC with another load) | 16.105 to 16.111 | |
| 192 XOR3 of sums, one after every IMMA | as compiled | 16.057 | 99.64% |
| | the same without the never-taken adds ptxas put in | 16.797 | 95.26% |
| | each XOR3 moved to another gap, 0 to 39 IMMA later | 16.67 to 17.00 | 94.2 to 96.0% |
| 96 XOR3 of sums, one after every second IMMA | as compiled | 16.036 to 16.051 | 99.68 to 99.77% |
| 2 XOR3 of sums after every IMMA | as compiled | 16.91 to 17.24 | 92.8 to 94.6% |
| XOR3 and an add that waits on it, after every IMMA | as compiled | 17.380 | 92.06% |
| XOR3 and 2 shared stores after every IMMA | as compiled | 49.3 | 32.4% |
| XOR3 and 2 loads that each wait on the last, after every IMMA | as compiled | 29.3 | 54.7% |
| 4 warps fold, 4 only help (11 plain steps a pass, flat out) | helpers got 1.7 steps an IMMA | 16.872 (16.130 idle) | 94.83% |

- **One plain step after an IMMA is free on sm_120, whatever its kind. The second in the same gap costs about 0.2
  cycle if it is independent and about a cycle if it reads sums or waits on the first; four cost 1.4 and eight
  5.2.** On sm_89 a k-step's worth weaves in for nothing. ptxas does not know the difference: it packs seven or
  eight steps behind one IMMA and leaves the next gaps empty, and that packing is the 2 to 5% the last round saw.
- A warp that only helps gets under two issue slots an IMMA beside a folding warp and costs it 0.4 cycle a step.
  Helper warps are no way out.
- With fewer than eight warps running, the same one-XOR3-a-gap loop costs 0.3 to 0.7 cycle an IMMA: 16.83 with
  one warp (16.14 without the XOR3), 16.46 with one warp on each of two or four schedulers (16.14), 16.64 with
  two warps on each of one to three schedulers (16.02), 16.06 with all eight (16.00). I don't know why.
- Shared stores are dear when they come close together: 48 wide stores a chunk, two in a gap, cost nothing
  (16.000), 96 narrow ones at one every second IMMA cost 1.3 cycles an IMMA (17.336), and four in a gap about 20
  cycles a gap (18.369).
- An XOR3 of sums is free only where ptxas puts it. In its loops the instruction right after an IMMA never names a
  register in the aligned block of eight that holds the IMMA's four sums; where an XOR3 would, ptxas puts a
  never-taken uniform add before it. Take those adds out and each such XOR3 costs about 1.5 cycles.
- `-Xptxas -O1` does not keep the source's order, and is slower (draw 92.1%, all together 90.1%).

**The readout and the draw** (a draw unit is the probe's 18 instructions for 16 bytes).

| Stream | As compiled | Reordered, one step a gap |
|---|---|---|
| Readout, three runs of 32 XOR3 | 98.40% (6 or 7 a gap) | 97.5 to 97.7% |
| Readout written one XOR3 after every IMMA of two k-steps (`-DRD=4`) | 97.96% (ptxas bunches them again, and spills) | |
| A readout's count of XOR3, one after every second IMMA | 99.68 to 99.77% | |
| 1 draw unit a k-step | 99.64% | 99.97%, same checksum |
| 2 draw units | 98.86% | 99.68%, same checksum (99.89% with the reuse flags kept, not checked) |
| 3 draw units | 97.9 to 98.0% | faults |
| 4 draw units | 96.92% | 96.02% (one or two a gap; not checked) |
| XOR3 after every second IMMA, and 1 draw unit | 98.98% | 97.77%, checksum differs: not the same program |
| the same with 2 draw units | 98.29% | |
| Three-run readout and 2 / 3 / 4 draw units | 97.04 / 96.32 / 94.08% | faults, or 95.14% with 3 |

- Loads and a readout's count of XOR3 hold 99.7%, as compiled. With one draw unit 99.0%. Two draw units alone,
  one step a gap, hold 99.7%.
- Reordered draws of three or four units fault (illegal memory access) in most orders, with every dependency the
  listing shows kept. `perm.py` now also keeps the step after an IMMA off that IMMA's block of eight registers
  (the draw's offset register shares a block with the first sums), which would explain it. That rule is not
  tested: the box budget ran out.

**Watts on an RTX 5090** (PC 218578cc, 800 W limit, operand bytes like a real fold's: 64 + u - v with u and v in 0
to 63; 30 s runs). This card ran everything at 2506 to 2532 MHz, miners too, so nothing was capped. No 450 W PC
turned up in four tries (575, 600, 600 and 800 W limits).

| Stream | Pipe use | By count, T-MAC/s | W |
|---|---|---|---|
| Loads only | 100.00% | 439.2 | 355 |
| + a readout's count of XOR3 | 99.77% | 437.7 | 384 |
| + readout in three runs, as compiled | 98.37% | 430.6 | 375 |
| Loads and 1 draw unit, as compiled / one step a gap | 99.64 / 99.97% | 435.8 / 437.0 | 395 / 403 |
| Loads and 2 draw units, as compiled / one step a gap | 98.77 / 99.68% | 432.3 / 434.8 | 427 / 426 |
| Loads and 4 draw units, as compiled | 96.92% | 422.3 | 473 |
| XOR3 and 1 draw unit, as compiled | 98.98% | 432.3 | 421 |
| XOR3 and 2 draw units, as compiled | 98.29% | 429.1 | 448 |
| Three-run readout and 4 draw units, as compiled | 94.08% | 410.2 | 481 |
| PeakMiner / SRBMiner / OneZeroMiner | 99.2 / 98.4 / 97.7% | 436.4 / 429.9 / 429.1 TH/s | 470 / 469 / 463 |

- A readout's XOR3 cost 29 W and a draw unit 30 to 40 W at this clock. The competitors' whole stream draws 79 to
  86 W more than loads and XOR3 alone, which is two to three of the probe's units; a real fold needs 3.5 units'
  worth of bytes a thread a k-step. So the probe's unit is too dear in watts by about half, as well as too long.

**What it means for an RTX 50 kernel**

- The budget is about one plain instruction an IMMA, each in its own gap: 48 a k-step a warp. The readout takes
  24 of them on average and the fragment loads 4. The probe's draw is 63 (3.5 units of 18), which cannot fit. A
  draw that writes the fragment registers directly (wide loads and three-input adds, no store and no ldmatrix) is
  about 24 instructions for a thread's 64 bytes, and 24 + 24 is 48. That count is the only thing the competitors'
  98 to 99% tells us about their draw.
- ptxas will not lay such a stream out by itself. It kept one step a gap only where each step read sums the IMMA
  before it wrote. So either every step is tied to an IMMA by a value it reads, or the loop is reordered after
  ptxas, as `perm.py` does.
- Keep eight warps running the loop, and keep an XOR3 of sums where ptxas puts it.

**Not verified**

- Why reordered draws of three or four units fault, and whether the block-of-eight rule stops it.
- A true readout (each IMMA's two XOR3 inside the 48 gaps after the chunk's last k-step) at one XOR3 a gap: the
  99.7% is for the count, with XOR3 that read sums on the compiler's schedule.
- Readout and draw together above 99.0%.
- The watts under a 450 W cap. On a card that clocks to 2880 MHz a cap would bite every stream here.
- One-step-a-gap figures for adds and other kinds come from the PC with another load (16.098 cycles on loads
  only); on a clean PC the same build had 56 adds held in one gap by a false dependency in the tool.
- Why a card with an 800 W limit sits at 2520 MHz from 355 W up.

Box time: 3.4 box-hours for 3 budgeted (RTX 5070 Ti 71 min over seven PCs; RTX 5090 135 min, of which 118 were an
idle box: this PC stalled for two hours with the delete waiting behind a log poll).

### Salad, 12 November: the rotation fold on RTX 50, compiled from C (branch `goal/nv-50`)

- **What ran.** The rotation fold (RTX 30 and 40's opt-in, `--rotation-fold`) has a chunk loop written by hand in
  sm_89 instructions; sm_120 encodes IADD3, LDG, BRA, MOV, LDC/LDCU and IADD.64 another way, keeps kernel
  parameters at `c[0x0][0x380]`, and starts the dynamic shared memory at 0x400, not 0. So the fold was written in C
  (`earn/native/src/pearl_rot_c.inc`, included by `pearl_rot.cu` for sm_120 and by the probe as `GP_ROTC`) and
  shipped as ptxas compiles it. One Salad RTX 5090 throughout: PC 3e7100e6, 575 W limit, driver 617.14, memory
  clock 14001 MHz (not locked; the app's 5090 memory-clock trial was not in play). Box script
  `earn/tuning/salad/nv50box.sh`; the release fold is the CUDA 13.3 core's (`pearl_core_cu13.node`, the core RTX
  50 rigs on driver 580+ load), built on the box from the same commit.
- **The miner.** `hashrate.js`, 120 s legs taking turns, the shipped form (B gathered a turn ahead, barriers after a
  chunk's first and last turn only, loop inlined; the exact file CI builds, hash ab55669f...): 401.60, 401.61,
  401.31 TH/s (2557 to 2560 MHz, 575 W) against 393.19, 390.97, 390.71 for the release fold (2408 to 2420 MHz,
  575 W). Mean 401.5 against 391.6: **+2.5%**. `verify-hits.js` 400 of 400 (688 hits, 35 salts). An earlier race of
  four forms (90 s legs): release 393.68 / 391.77 / 391.33; B ahead and inlined 398.19 / 397.74; the shipped form
  401.92 / 401.44; B ahead with the readout spread (outlined) 401.52 / 400.98; that plus barriers trimmed and the
  readout over two turns 390.17 / 389.95. Before the loop was inlined the miner's cubin spilled one register a chunk
  (one STL and one LDL in the loop) and got 384.4 / 384.6 where the probe's cubin of the same form, which did not
  spill, held 92% of the pipe.
- **The probe** (`gprobe.cu` with `GP_ROTC`, k 65536, 128 tiles a block, 30 to 60 s, the CPU check passing on every
  form run for real): the loop as ptxas orders it holds 84.0% (no B ahead), 92.1% (B ahead), 90.6% (B ahead,
  barriers trimmed), 85.4 to 89.2% (readout spread). Price list on the 92.1% form (timing only, wrong results on
  purpose): no bank collisions 92.1%; no barriers 91.1%; no readout in the loop 92.4%; neither 94.0%; B gathered once
  a chunk 91.6% (and 2685 MHz against 2593, fewer steps under the power cap); no stage loads 85.0%; nearly
  everything but the first turn's gathers and the IMMA out 95.0%. No one part holds the 8 points; they are in how
  ptxas orders the IMMA stream on sm_120.
- **Our own order: tried and dropped for now.** `earn/native/probes/hand89/sched120.py` keeps ptxas's instructions
  and registers, reorders the loop, and writes the control fields again (stall, yield, scoreboards, waits, reuse);
  nothing is assembled, and every result is read back through nvdisasm. Every scheduled cubin computed right
  (16,384 of 16,384), and none beat ptxas. Calibration on ptxas's own order: our first control fields cost 11 points
  (scoreboards 10, stalls and yields 5, cleared reuse flags 3.7); written as ptxas writes them (only the newest
  awaited shared load gets a scoreboard, since a warp's shared loads finish in order; IMMA 11 cycles apart; PRMT
  pairs marked for reuse) they come within 3 points (89.2 against 92.1%). Reorders on top (steps spread one or two a
  gap, loads first, over-full gaps smoothed) lost 1 to 10 points. Two sm_120 rules learnt on the way: nvdisasm (and
  so presumably the card) refuses a yield with a stall of 0 or of 12 and more; and ptxas spaces a warp's IMMA 11
  cycles apart.
- **The other miners on the same PC** (bench.sh, 180 s each after 60 s, sample means): SRBMiner 3.8.6 434.3 TH/s (567 W, 2619 MHz),
  PeakMiner 2.20.7 427.0 (530 W, 2476 MHz), OneZeroMiner 1.8.5 419.0 (533 W, 2485 MHz), our v0.6.15 CLI 386.6
  (575 W, 2457 MHz); every share the stand-in pool checked passed. So the fold is at 92.4% of the best (401.5 by
  `hashrate.js`, a different meter from bench.sh's samples), the release core at 90.2%, the v0.6.15 CLI at 89.0%.
  The other miners run under the 575 W cap; ours sits on it. Not measured: an RTX 5080 or 5070 (no Salad slot was
  free for one), and a 450 W 5090.
- **Shipped (opt-in, off by default).** `pearl_rot_sm120.cubin`, built by CI's CUDA 13 job with `build_rot.py
  --arch sm_120` against `src/pearl_rot_sm120.ref`, carried by the installer, AppImage, CLI, HiveOS package,
  release and self-updater like the RTX 30 file; the host opens the fold on compute 12.0. A CUDA 13 cubin needs
  driver 580 or newer; on an older driver the load fails and the card mines on the release fold.

### Salad, 12 November: the 4090's hand loop on RTX 50, and an RTX 5080 (branch `goal/nv-50b`)

- **Where the 5090's pipe went.** A stream of nothing but int8 mma (`probes/hand89/mmabench.cu`, 48 sums, fixed
  fragments) holds 99.99% of the tensor pipe on an RTX 5090 in every order (A shared by the mma in a row, B shared,
  neither) and 99.79% with one warp a scheduler. The probe with every part but the mma taken out (timing only)
  holds 99.57%; with B gathered once a chunk 97.94%; with the first turn's A gathers and stores 96.60%; the shipped
  C form 91.27%. So the 8 points are in how ptxas places the gathers, stage loads and barriers, not in the mma.
  Reordering the C instead (a turn's barrier ahead of its last tile's mma, A gathers a tile ahead, both, with the
  readout forms) never beat form 11: 84.3 to 90.5% in the probe; the barrier move lost 8% in the miner on a 5090
  and 6% on a 5080.
- **The hand loop, encoded for sm_120.** `probes/hand89/rfold120.py` takes rfold.py's loop (the one RTX 40 ships)
  as rfold.py builds it and writes it into an sm_120 build: shared addresses through the base ptxas computes
  (0x400 + the CTA's cluster rank << 24) in a uniform register, parameters through LDC/LDCU from c[0x0][0x390]
  and 0x398, LDG with desc[URn], sm_120's branch offsets. What the card taught on the way, each found by cutting
  the loop short (`cut=N`) and running the cut on the card: IMAD.WIDE.U32 with a 64-bit register addend gave a
  wrong address (ptxas never writes it); a NOP's wait does not hold the warp; a 255-register kernel has no R253
  (illegal instruction); registers the kernel keeps across the call are put back before the return. Every word is
  read back through nvdisasm. build_rot.py --arch sm_120 now builds the file this way.
- **Correct.** Probe CPU check 16,384 of 16,384 at k 2048 and at k 8192, 512 of 512 at k 65536. verify-hits.js on
  the miner's cubin: 400 of 400 (498 hits, 26 salts; again 510 hits on another PC). rfold.py's lead=2 is WRONG on
  sm_120 (the check fails): the default lead=3 is needed.
- **Probe, clean PC 02e173db (limit 488.75 W), k 65536, 30 s:** hand loop 95.94 / 95.94% at 2375 / 2356 MHz
  (count 396.6 / 393.5 TH/s); shipped C form 11 91.30 / 91.28% at 2482 / 2435 MHz (394.4 / 386.9). Keys: ist=7
  96.08%, pad=11 95.93, cq=0 95.76, ae=0 95.57. At the power cap the pipe gain is mostly paid back in clock.
- **Miner, same PC, hashrate.js 120 s legs taking turns:** hand loop 389.36, 389.55, 389.22, 389.47 TH/s (2345
  MHz, 489 W); form 11 (the opt-in shipped now) 384.24, 384.43, 383.97 (2412 MHz); release 367.82, 367.45 (2236
  MHz). Hand loop +1.35% over form 11, +5.9% over the release. The other miners on that PC (bench.sh, 180 s after
  60 s, sample means, 489 W): SRBMiner 3.8.6 422.2 (2484 MHz), PeakMiner 2.20.7 416.3 (2418 MHz), OneZeroMiner
  1.8.5 412.2 (2440 MHz), our v0.6.15 CLI 368.1 (2243 MHz): the hand loop is at 92.2% of the best (form 11 91.0%).
  What is left is energy: SRBMiner holds about 97.8% of the pipe at 2484 MHz and we 95.4% at 2345 MHz on the same
  489 W, so a cycle of ours draws about 6% more. A first race on a PC whose GPU had another load
  (spreads 9 to 23 TH/s): hand 378.5 / 390.0 / 376.6 against form 11 358.9 / 376.9 / 364.9.
- **RTX 5080** (PC 6d1af811, 300 W, driver 616): release 204.4 / 205.9 / 205.8 TH/s (2500 to 2515 MHz); form 11
  203.1 / 203.3 (2570 MHz): the opt-in loses 1.1% there. bench.sh means: SRBMiner 3.8.6 221.1 (2621 MHz),
  PeakMiner 2.20.7 219.5 (2575 MHz), our v0.6.15 202.8 (2498 MHz); OneZeroMiner 1.8.5 could not reach its pool.
  92% of the best. Their pipe share works out at 98 to 99% against our 92%.
- **Shipped.** build_rot.py --arch sm_120 makes the hand loop's file (hash e439e1ca..., the same from a Salad PC's
  Linux and from Windows); on an RTX 5090 (PC 944d2f0a, 600 W) that file passed verify-hits 400 of 400 (570 hits,
  29 salts), and it is now the one hash in `src/pearl_rot_sm120.ref` and `kRotCubins`. Still opt-in
  (`--rotation-fold`).
- **CI's file raced.** RTX 5080 (Salad, 360 W, clean PC, driver 621.14), hashrate.js 120 s legs taking turns: hand
  loop 223.50, 223.48, 223.46 TH/s (2767 MHz, 360 W); form 11 215.01, 214.95, 215.01 (2775 MHz); release 220.57,
  220.36 (2750 MHz). verify-hits 284 of 284. Same PC, bench.sh means: PeakMiner 232.0 (2775 MHz, 352 W), SRBMiner
  230.6 (2776 MHz, 337 W). So on the 5080 the hand loop is +3.9% over form 11 and +1.4% over the release (the opt-in
  no longer loses there), and at 96.3% of the best. The 5080 runs at its clock limit, not its power limit, so the
  gap there is pipe share: about 94.0% of the pipe in the miner against PeakMiner's 97.3%. An RTX 5090 held at 1965
  MHz (PC 944d2f0a, about 300 W, noisy): hand 317.6, 321.9, 324.1 against form 11 312.4, 293.8, 311.1 and release
  320.3, 318.9.

### Salad, 12 November: the RTX 50 hand loop retuned, ist=7 and iy=0 (branch `goal/nv-50c`)

- **Shipped.** sm_120's rfold keys are now `rpr=2 u8=1 rgf=1 rb0=3 rb1=0 ist=7 iy=0` (build_rot.py ARCH_KEYS):
  the 4090's readout keys, the IMMA's stall 7 cycles (not 5) and the IMMA without the yield flag. The file
  (hash 0768c6f2..., the same from a Salad PC's Linux, CI and Windows) passed verify-hits 400 of 400 on an RTX 5090
  (549 hits) and an RTX 5080 (452 hits), and is now the one sm_120 hash in `src/pearl_rot_sm120.ref` and
  `kRotCubins`. Still opt-in (`--rotation-fold`).
- **Probe, k 65536, pipe share.** The readout keys alone change nothing on sm_120 (5090 95.87 against 96.02%); ist
  is what matters, with a sharp best at 7. 5080 at 2685 MHz, with the readout keys: ist 6 95.43, 7 97.70, 8 97.37,
  9 93.60, 10 94.83% (pad 10 or 11 changes nothing; ist=7 without the readout keys 96.27%). 5090 (600 W PC):
  96.02 -> 97.60%. iy=0 on a 510 W 5090: 97.90 against 97.64%. All correct builds 16,384 of 16,384 at k 2048 and
  8192, 512 of 512 at k 65536.
- **Miner, hashrate.js 120 s legs taking turns, same PC.** RTX 5090, 488.75 W PC (the one goal-nv50b raced, the old
  file 389.4 again): old file 389.44, 389.46, 389.39; ist=7 399.18, 399.45, 399.11; with iy=0 402.11, 402.17,
  401.88 (+3.3%, 2374 against 2346 MHz). RTX 5090, 575 W PC: ist=7 416.96, 416.99, 416.97; with iy=0 418.05,
  418.13, 418.07. RTX 5090, 600 W PC (drew about 510 W): old 430.30, 430.22, 430.35; ist=7 434.44, 434.46. RTX
  5080, 360 W PC: old 232.34, 231.99, 232.13, 232.34; ist=7 236.89, 236.65, 236.93 (+2.0%). RTX 5080, 450 W PC:
  old 233.75, 233.69; ist=7 237.53, 237.49, 237.54 (+1.6%). Tile batch (PEARL_ROT_TILES) 128: 233.92 against 233.75
  at 32; 8: 232.59. The default stays.
- **Against the other miners, same PC (bench.sh sample means).** 5080 360 W: SRBMiner 3.8.6 242.2, PeakMiner 2.20.7
  242.1; ours 236.8 = 97.8% (was 95.9%). 5080 450 W: PeakMiner 241.6, SRBMiner 238.6; ours 237.5 = 98.3%. 5090
  488.75 W: SRBMiner 422.3 (2485 MHz), PeakMiner 416.5; ours 402.1 = 95.2% (was 92.2%). 5090 600 W PC: SRBMiner
  450.3 (2666 MHz, 498 W), PeakMiner 441.8; ours (ist=7, before iy=0) 434.4 = 96.5%.
  5090 575 W PC (driver 621.14, 78 C): SRBMiner 434.8 (2559 MHz, 555 W), PeakMiner 424.8 (2474 MHz, 522 W);
  ours 418.1 (2475 MHz, 538 W) = 96.2%. A 5080 PC with another load on it (2627 MiB in use at idle): ist=7
  213.06-213.22, with iy=0 213.51-213.73 (+0.25%); verify-hits passes there, its rates are not used above.
  A clean 5080 PC (360 W, driver 610.88, 2797 MHz, 331 W): ist=7 235.00, 234.99, 235.02; with iy=0 235.58, 235.59,
  235.59 (pipe 97.9%, the probe's 97.93%). PeakMiner 238.2 (2797 MHz, 321 W), SRBMiner 237.4: ours 98.9% of the best.
- **Priced after shipping (probe, shipped keys, timing only).** Offsets records from the SM's cache (nlg) change
  nothing on a 5090 (97.84%) or a 5080 (97.93%): the L2 trip is hidden. Without the gathers' PRMT (noprm) 99.05% /
  99.15%: the PRMT are the largest pipe item left (1.2 points). One barrier instruction (brg=1, correct) nothing.
- **Where the rest is.** On the 5090 our pipe now equals SRBMiner's (97.2 against 96.9% by count on the 600 W PC);
  the gap is clock at the same watts (2567 against 2666 MHz), i.e. energy per cycle. On the 5080 it is about one
  point of pipe (97.6 against 98.6%) at the same clock.
- **The 5080's "2 points between probe and miner" were not real.** On one 5080 the probe's count was 234.5 / 234.0
  and the miner 233.75: 0.2%. The earlier figure set the 5090's probe pipe beside a 5080 miner. The 5080 loop holds
  less pipe at higher clocks (97.7% at 2690 MHz, 96.5% at 2865).
- **Energy price list, 5090 at its cap (probe, timing only unless said; clock against the shipped keys at the
  same watts).** No readout +2.9 to +3.0%; A's gathers as moves +2.2%; B's gathers as moves +7.7%; half of B's
  gathers gone (bsym=4 with no stores or loads) +4.9% count; B shared through the exchange without its syncs
  (bsym=4) +1.5% count (the ceiling of B drawn once, before syncs); every operand byte the fill +6.8% at 389 W;
  fill 0 (S8) -1.5 to -2.7% (worse), fill 32 level; B kept in place across tiles (snk 1 and 2) 0; LDS.128 for
  LDSM 0; reuse marks on PRMT pairs (prr, correct) 0; py=1, ly=0 0. `nvidia-smi -lgc` is refused on Salad 5090s
  ("Unknown Error"), so watts at a locked clock cannot be had there.

### Salad, 12 November: the RTX 50 gathers without PRMT, byte-transposing ldmatrix (branch `goal/nv-50d`)

- **Shipped.** Every gather of the sm_120 hand loop is now one `ldmatrix.m16n16.x2.trans.b8` (SASS
  `LDSM.8.MT1616.2`, an sm_120a instruction that runs from the plain sm_120 cubin). It transposes bytes, so each
  lane gets rows g and g + 8 in registers of their own and the two subtracts write the fragment words: the 22 PRMT
  a k-step are gone (698 instructions in the loop against 802). The tables are stored in plain row order
  (pearl_rot.cu `ROT_TAB8`); offsets and the k order search are unchanged. rfold key `b8=1`, in sm_120's ARCH_KEYS.
  The file (hash 8e56197c..., the same from a Salad PC's Linux, CI and Windows) passed verify-hits 400 of 400 on two
  RTX 5080s (496 and 503 hits) and an RTX 5090 (738 hits), and replaces 0768c6f2 in `src/pearl_rot_sm120.ref` and
  `kRotCubins`. Still opt-in (`--rotation-fold`). RTX 40 and 30 cannot use it: ldmatrix has no byte form before
  sm_100a/sm_120a.
- **Probe, k 65536.** 5090 (575 W PC): 97.58, 97.59, 97.56% of the pipe for the loop before; 98.64, 98.61, 98.62%
  with b8, at a slightly higher clock on the same 520 W (count +1.5%). Without the PRMT and with wrong sums (noprm)
  98.80%: b8 gets nearly all of it. Correct at k 2048 and 8192 (16,384 of 16,384) and 65536 (512 of 512). On
  another 575 W PC: b8 98.96%; ist 6 96.85, 8 98.43 (7 stays); al=0 98.82, cq=36 99.00 (nothing).
- **Miner, hashrate.js 120 s legs taking turns, same PC.** RTX 5090, 575 W PC A: before 417.52, 417.43, 417.47;
  b8 424.68, 424.42, 424.26 (+1.7%). RTX 5090, 575 W PC B (draws 575 W): before 435.52, 436.06, 437.37; b8 445.48,
  442.45, 443.49 (+1.7%). RTX 5080, 295 W PC: before 231.94, 231.80, 231.80; b8 235.16, 235.13, 235.09 (+1.4%).
  RTX 5080, 360 W PC: before 234.18, 234.19, 234.18; b8 237.49, 237.44 (one disturbed leg, 233.47) (+1.4%).
- **Against the other miners, same PC (bench.sh sample means).** 5090 PC B: PeakMiner 448.46 (2599 MHz),
  SRBMiner 447.53 (2632 MHz); ours 443.8 = 99.0% (was 97.3%). 5080 360 W: SRBMiner 240.74 (2835 MHz), PeakMiner
  238.94; ours 237.47 = 98.6% (was 97.3%). 5080 295 W: SRBMiner 238.77 (2821 MHz), PeakMiner 236.88; ours 235.13 =
  98.5% (was 97.1%). **5080, clean 360 W PC that stays under its cap (driver 610.88, 2797 MHz, clock-bound):**
  b8 238.13, 238.12, 238.12 against 235.53 before (+1.1%); PeakMiner 237.88 (2797 MHz, 346 W), SRBMiner 236.40
  (2798 MHz, 329 W). Ours 100.1% of the best: the first RTX 50 row above the other miners.
- **Where the rest is.** The other four PCs ran at their power cap, and there our pipe is now above SRBMiner's (5080 at
  295 W: 98.75 against 98.4% by count) while our clock is 1 to 2% lower at the same watts. So the gap is energy per
  cycle. Priced on the b8 loop at a 5090's cap (timing only): B's gathers as moves +2.8% count (it was +7.7%
  before b8), no readout +1.0%. `nvidia-smi -lmc` is refused on Salad ("does not have permission"), so the memory
  clock, which gains 1% on the RTX 40 rotation fold, cannot be priced on these PCs.

### Salad, 12 November: where a capped RTX 50's watts go, and one barrier fewer (branch `goal/nv-50e`)

- **Shipped: bt=13.** The sm_120 hand loop leaves out the barrier after a chunk's second turn. No A fragment is
  stored in the second or third turn, so it ordered nothing; every load keeps a barrier between it and the stores
  around it. CPU check 16,384 of 16,384 at k 2048 and 8192, 512 of 512 at 65536. Hash 2bc71ae7 (CI, a Salad PC's
  Linux and Windows agree) replaces 8e56197c in `src/pearl_rot_sm120.ref` and `kRotCubins`: verify-hits 400 of 400
  on an RTX 5090 (1131 hits) and an RTX 5080 (624 hits). Probe pipe 99.03% against 98.95% (5080), 98.26% against
  98.18% (5090).
- **Miner, hashrate.js 120 s legs taking turns, same PC.** RTX 5080 (360 W limit, runs at 2775 MHz and ~350 W, so
  clock-bound): 8e56197c 235.30, 235.24, 235.24; 2bc71ae7 235.51, 235.49, 235.49, 235.47 (+0.10%). RTX 5090 (534 W
  limit, runs at ~2522 MHz and 433 W, clock-bound): 423.51, 423.93, 423.93; 2bc71ae7 423.96, 424.23, 424.68 (+0.13%).
- **Against the other miners, same PCs (bench.sh sample means).** 5080: PeakMiner 236.50 (2775 MHz, 342 W),
  SRBMiner 234.57 (2775 MHz, 328 W); ours 235.49 = 99.6% of the best. 5090: PeakMiner 426.00 (2526 MHz, 427 W),
  SRBMiner 419.35 (2545 MHz, 467 W); ours 424.29 = 99.6%. Both PCs are clock-bound and PeakMiner runs at our clock,
  so on these two the gap is pipe (about 0.4%), not energy.
- **Energy price list on the b8 loop (probe, k 65536, timing only unless said).** A 5080 PC running under its cap
  (watts read at a fixed 2775 MHz): b8 336-347 W. B's operands held at the fill with the loads kept (new key `bcz`)
  269-272 W; B's loads as moves (`nold=3`) 312 W; A's operands held (`acz`) 336 W; A's loads as moves 338 W; both
  operands held 246 W; no readout 330 W; no A stores (`nost`; the stage then holds zeros) 296 W; B shared through the
  exchange (`bsym=4`) 339-341 W against its ceiling (`bnst bnld`, half of B's gathers gone) 326-327 W, so the
  exchange eats ~85% of what B drawn once could save; B's bank collisions gone (gprobe fake 2) -2 to -4 W and +0.25
  points of pipe; records from the SM cache (`nlg`) nothing; IMMA S8.S8, S8.U8 or U8.S8 (correct with operands
  0..126) +7 W against U8.U8. A 5090 at its 510 W cap (count = pipe x clock against 430.2-431.5): B's loads as moves
  +5.3%, no readout +3.4%, A's operands held +3.0%, A's loads as moves +2.5%, `nlg` and bt=13 nothing.
- **What that says.** The tensor unit's energy follows the operands' values (held at 63 they cost much less, zeros
  less still), not their changes: B held within a tile (`snk=2`), both bands on the same k-step order (a gprobe
  test, which also broke half the tiles), every slot reading one slot's B and both bands reading one band's A all
  left the watts unchanged. The values are fixed: the protocol's noise is +-63 and the operands must stay 0..127
  as s8 (fill 63 or 64). B's bank collisions are at their floor: 128 random rank indices put 16 +- 3.7 rows on each
  of 8 bank residues, so some octets must collide; greedy and annealing searches (`S/nv50e/ordsim.py`) and gprobe's
  order modes 2, 4 and 9 leave 0.76-0.84 rows of 8 against mode 6's 0.83, and mode 9 is +0.03 points of pipe.
- **Memory clock on RTX 50: not priced.** Salad refuses `-lmc`, and no other rented 5090 was allowed (no Vast or
  Clore for this work), so the RTX 40 rotation fold's 810 MHz trial was not extended to Blackwell.

### Salad, 12 November: the gathers' over-waits, and a clock-bound RTX 5080 above PeakMiner (branch `goal/nv-50f`)

- **What was lost.** Three scoreboards serve up to six gathers in flight, and a wait on a scoreboard holds the warp
  until every load on it has landed. In the shipped loop 46 subtracts a chunk waited on a scoreboard whose newest
  load went out 0 or 1 gaps before, when the load they needed went out 9 to 15 gaps before.
- **Shipped: dpb=7 dpm=1 dpg=3 (rfold120.py).** Each such wait moves into the control field of the first newer load
  on that scoreboard, when its own load is 3 or more gaps older than that load (there it has long landed). No
  instruction is added. A walk of the listing with outstanding loads (prologue, three passes, epilogue) finds no
  hazard. CPU check 16,384 of 16,384 at k 2048 and 8192, 512 of 512 at 65536. Hash 0deef760 (a Salad PC's Linux and
  Windows agree) replaces 2bc71ae7 in `src/pearl_rot_sm120.ref` and `kRotCubins`: verify-hits 400 of 400 on an RTX
  5090 (1007 hits) and an RTX 5080 (636 hits).
- **Tried and dropped.** DEPBAR.LE in front of the subtracts (`dpb` alone): correct only with ptxas's stall 4 (stall
  1 gives wrong sums at every k), and 0.13 to 0.2 points of pipe worse: the 23 extra issues cost more than the waits.
  `dpg` 1 and 2 move more waits but onto loads whose producer has not landed (5090 pipe 97.77 / 97.87 against 98.0);
  `dpg` 4 and 6 equal 3 on a 5080. `sbo` 0, 3, 5 and `cq=36` with dpm: no change. One barrier instruction from a
  register (`brg=1`): correct on sm_120, 0.15 points worse. The loop with no barriers at all (timing only, wrong)
  gains nothing on either card, so the barriers are not where the pipe goes.
- **Probe, k 65536, legs in turn.** RTX 5080 (360 W limit, clock-bound at 2835 MHz, 330-335 W): 98.54-98.61% of the
  pipe against 99.04-99.10% (count 240.3 against 241.5, +0.46%). RTX 5090 (437 W limit, at the cap, ~2227 MHz):
  97.50-97.53% against 97.97-98.03% (count 378.0 against 379.8-380.0, +0.5%).
- **Miner, hashrate.js 120 s legs taking turns, same PC.** 5080: 2bc71ae7 238.46, 239.20, 238.93, 238.94; 0deef760
  239.79, 239.93, 240.13, 240.41 (2825-2829 MHz, 336 W): +0.50%. 5090: 373.10, 373.02, 373.03, 373.24; 0deef760
  374.56, 374.32, 374.73, 374.84 (2212-2221 MHz, 437 W): +0.41%.
- **Against the other miners, same PCs (bench.sh sample means, PeakMiner 2.20.7, SRBMiner 3.8.6).** 5080: PeakMiner
  239.29 (2835 MHz, 330 W), SRBMiner 189.08 (2814 MHz, 258 W; it ran low on this PC); ours 240.07 = **100.3% of the
  best**. 5090 at its 437 W cap: SRBMiner 379.12 (2268 MHz), PeakMiner 378.07 (2235 MHz); ours 374.61 = 98.8%. On
  the capped card both run 20 to 50 MHz faster at the same watts: the gap there is still energy per cycle.

### Salad, 12 November, evening: shared records on sm_120 are wrong, and IMMA operand reuse saves nothing (branch `goal/nv-50g`)

Nothing shipped; the sm_120 file stays 0deef760. Both PCs were clock-bound, so energy shows as watts at a fixed clock:
an RTX 5080 (360 W limit, 2790 MHz) and an RTX 5090 (400 W limit, held at 2025 MHz, 340-356 W).

- **Shared records (`sr`, the RTX 40 form from `goal/nv-19`) ported to sm_120.** rfold120.py now writes it: the
  shared base goes into sr's address registers, R2UR uses sm_120's encoding, and gprobe `rot 5` gives b8 tables with
  shared records. Every port reads back as asked and the hazard walk is clean. **It is wrong on the card.** Each check
  run has 1 to 9 bad tiles out of 64 (k 2048, 8192 and 65536, on both cards). Each bad tile has exactly one band (128
  threads) wrong, which points at one warp's stage stores or first-turn gathers. Ruled out, each still wrong: without
  b8 (the sm_89 form), without bt=13, without dpm, the uniform registers at UR24 or UR40, the pointer step 1, 2 or 4
  gaps behind the load, the record load at gap 6, plain LDG in place of the constant path, R2UR without a scoreboard
  (as ptxas writes it), a 16-cycle settle after each R2UR group, and every R2UR issued twice. Not yet tested: sr's
  registers below UR20, where the kernel's own uniform registers sit.
- **What sr would be worth (timing runs, probe k 65536, legs taking turns).** 5080: pipe 99.38% against 99.56%,
  345.9-349.8 W against 351.3-358.8 W, count 238.5 against 238.9 (-0.17%). 5090 at 2025 MHz: 99.40% against
  99.56%, 339.3-347.3 W against 343.1-356.0 W, count 350.4 against 351.0. About 7 W less at the same clock, for 0.17
  points of pipe. On a card held by its power cap that could come back as clock, but no capped PC was on hand to
  measure it.
- **IMMA operand reuse (`ru=1`, A's register marked for reuse on 7 of each 8 IMMA).** Correct (16,384 of 16,384 at k
  2048 and 8192). No saving: on the 5080 355.0, 357.9 and 359.1 W against 351.8, 356.7 and 358.4 W, same pipe and
  clock. The flag changes nothing here, as on sm_89; ptxas never sets it on IMMA either.

### Salad, 12 November, night: the sm_120 shared-records fault found and fixed, and +0.7% on a capped RTX 5090 (branch `goal/nv-50h`)

Nothing shipped yet; the sm_120 file stays 0deef760. One PC: an RTX 5090 with a 575 W limit that the fold holds at
its power cap (cap active in every sample, 510-535 W, 2500-2550 MHz), driver 621.14.

- **The fault.** gprobe's new PD_DIAG (k 2048: one chunk per transcript word) showed it in every bad tile:
  always chunk 0. Changing one k-step of the band's A to another k-step of chunk 0 makes 96 of 128 threads match. Read
  through the rot order, that means one warp folded chunk 0's first turn with its second turn's record. That warp's
  store fed bad A to the three other warps of its band. Its own B was wrong too, which is why the fourth warp did
  not match. Cause: in the prologue, the pointer step (waiting on the load's read scoreboard) issued one cycle
  behind the first record load. On sm_120 it can go ahead, and the load reads the stepped pointer. sr's uniform
  registers below UR24 (srm, UR0-3 and UR18-23, or UR18-27) and stores without a uniform register (srw) were still
  wrong, as expected.
- **The fix, `srp=1`:** the step goes behind the prologue's existing 8-cycle stall. Probe: 16,384 of 16,384 at k
  2048 (5 runs) and 8192 (3), and at k 65536. The miner's cubin with sr and srp (hand-built on the PC):
  verify-hits 400 of 400 (1163 hits, 57 salts). The shipped keys: 400 of 400 (1147 hits).
- **Probe timing at the cap (k 65536, legs taking turns).** Shipped keys: pipe 98.99, 98.98, 99.01%; 2543, 2533,
  2524 MHz; count 438.2, 436.4, 435.0. sr+srp: 98.82, 98.81, 98.84%; 2556, 2553, 2554 MHz; count 439.8, 439.2,
  439.4. That is 0.17 points of pipe for about 25 MHz, a net +0.7%.
- **Race, hashrate.js 120 s legs taking turns.** Shipped keys (rold) 431.25, 431.27, 431.41, 431.38 TH/s (2497
  MHz, 530 W). sr+srp (rsr) 434.38, 434.35, 434.31, 434.29 (2520 MHz, 527 W). Means 431.33 -> 434.33, **+0.70%**.
  Not measured: competitors on this PC, which was deleted before bench.sh ran. A clock-bound card (5080) where sr
  costs 0.17 points, and CI's own bytes (build_rot.py with sr=1 srp=1 gives 8584f90b on Windows).

### Salad, 13 November, early: shared records shipped on RTX 50, and where SRBMiner's 5090 lead comes from (branch `goal/nv-50i`)

**Shipped:** sm_120's ARCH_KEYS gain `sr=1 srp=1`. The file is 8584f90b (build_rot.py on a Salad PC's Linux and on
Windows: the same hash), listed in pearl_rot_sm120.ref and kRotCubins in place of 0deef760.

- **RTX 5090, 480 W limit PC, memory at 810 MHz, driver 621.14, at its power cap (SW power cap on).** verify-hits 400
  of 400 (1528 hits, 74 salts). hashrate.js 120 s legs taking turns: 0deef760 423.39, 423.47, 423.82, 423.66 (2690-2699
  MHz, 433-435 W); 8584f90b 426.65, 426.49, 426.71, 426.84 (2705-2709 MHz, 428-429 W). Means 423.59 -> 426.67,
  **+0.73%**. Same PC, bench.sh: SRBMiner 429.9 (2769 MHz, 417 W), PeakMiner 420.3 (2678 MHz, 424 W). Ours 99.2% of
  the best.
- **RTX 5080, 360 W limit PC, driver 610.88.** verify-hits 400 of 400 (850 hits, 43 salts). Legs: 0deef760 245.23,
  245.61, 245.80, 245.57 (2870-2879 MHz, 360 W, at the cap); 8584f90b 245.55, 245.52, 245.50, 245.50 (2880 MHz,
  353-357 W, under the cap at the clock ceiling). Level (-0.01%). Same PC: PeakMiner 244.7, SRBMiner 243.6, all three
  at 2880 MHz. Ours **100.3%** of the best. On a 5080 that stays well under its cap, expect sr's -0.17 points
  (goal-nv50g); no PC like that came up this round. The host loads one file per architecture and cannot tell a
  capped card from a clock-bound one before mining, so there is no clean way to make sr conditional.
- **Why SRBMiner leads the 600 W 5090s (official rerun of 12 November, bench.sh samples read back from Salad).**
  Every miner does the same work per clock: TH/s per GHz 171.6-172.6 for ours, PeakMiner, OneZeroMiner and SRBMiner on
  both PCs. The lead is all clock. PC 6043ec7a: ours 2506 MHz at 533 W, PeakMiner 2505 at 525 W, SRBMiner 2590 at
  559 W. PC 0e704bd1: ours 2481 at 531 W, SRBMiner 2594 at 549 W. None of them reached 600 W, and our release's C
  fold (ours0) did, at 600 W and 2518 MHz. So the 5090s at 600 W are not held by board power. Something else sets
  the clock, and SRBMiner gets about 3.5% more of it while drawing more watts. On the 810 MHz-memory PC above, SRBMiner
  also ran 60 MHz higher than ours at 12 W less. On the 5080 it drew 336 W at the same 2880 MHz as ours (~355 W):
  about 5% less energy per hash. Observation only (nvidia-smi beside each run); nothing about how it does that.
  nv50box.sh's new `SMIQ=<secs>` logs the active clock limiters and the power samples' spread beside each run, to
  name the limit.
- **The limit, named (600 W RTX 5090 PC, driver 610.88, SMIQ=10, every miner on the same PC).** All four show "SW
  Power Cap" as the only active clock limiter. Ours (8584f90b) 430.98 and 428.02 TH/s at 2507-2529 MHz, 522 W
  average, smooth (one 2.4 s window spans ~497-535 W). The release C fold 413.5 and 414.1 at 2535 MHz, pinned at
  600 W (550-615 W). PeakMiner 420.5 at 2486 MHz, 519 W, smooth (475-545 W). SRBMiner 436.3 at 2591 MHz, 538 W,
  very spiky (385-670 W in one window). Board power holds only the C fold; our loop and PeakMiner stop ~80 W under
  it at the same reason. So the binding limit is a part of the card's power (most likely the core's own rail),
  not the board total. A loop that does more of its work off that rail, or less work on it, gets the clock.
- **Gaps priced, and they lose.** A test-only knob (PEARL_ROT_GAP=every:us: the card drained and left idle every
  N batches; not in the repo) to copy SRBMiner's spiky draw, legs taking turns on that PC: base 431.89, 427.62,
  427.82, 427.84; 16:2000 426.66 and 426.70; 8:2000 425.68; 4:2000 424.07; 2:1000 423.69; 8:8000 419.83. The clock
  rose 5-20 MHz and the average stayed at 521-524 W: the idle time costs more than it buys. SRBMiner's spikes are not
  how it gets its clock. What it has is a lower cost on the limited rail per hash (about 3% clock at equal work
  per clock). Our own price list (goal-nv50e) puts the movable share in B's shared-memory loads (+5.3% as moves),
  the readout (+3.4%) and A's operand values (+3.0%); operand encodings are closed (goal-nvk17). So the lever is
  fewer shared-memory steps per IMMA, a tiling change. A price list re-run against this limit (nold, nord, acz,
  bcz) was sent to the PC but the box went silent after its restart and was deleted; not measured.

### Salad, 13 November, evening: the 96 x 512 block on RTX 5090s, +0.55% and +1.50% in the miner (branch `goal/nv-50k`)

**Shipped:** a file of its own for RTX 5090s, `pearl_rot_sm120w.cubin` (build_rot.py `--arch sm_120w`: sm_120's keys
plus `w8=1 w8g=4`, kernel built with `-DROT_W8=1 -DROT_C=49155`). The host takes it on compute 12.0 cards with 128
SMs or more when it is listed and the profile's n is a multiple of 512; `PEARL_ROT_W8=0` keeps a 5090 on
`pearl_rot_sm120.cubin`. Other RTX 50 cards are unchanged (8584f90b). The file is 9da716e2 (build_rot.py on two
Salad PCs' Linux and on Windows: the same hash), listed in the new `pearl_rot_sm120w.ref` and in kRotCubins.

The 96 x 512 block is goal-nvk22's: A shared by all eight warps, so each warp gathers and stores half as much of
it. It closed on RTX 40 at Salad's memory clock (section above). On a 5090 at its power cap the saving comes back as
clock.

- **Probe, timing first** (gprobe, k 65536, 30 s legs, 575 W 5090, SW power cap in every sample; each run against
  the shipped loop's runs either side of it). `ah=3` (a warp's first turn draws three A tiles, as in the block): +1.4%
  by count (442.5 and 442.0 against 435.7-438.8), 2560 MHz against 2525, same pipe. `nord` (no readout) +2.1% for
  scale. Barriers of all eight warps cost 1.8 points of pipe whether a chunk has one, two or three (97.43-97.50%
  against 99.25%): they put both warps of each scheduler in step.
- **The block on sm_120, three ways to order its 12 KB stage, all correct** (16,384/16,384 at k 2048, 8,192/8,192 at
  8192, 512/512 at 65536). rfold120.py now gives sr's uniform registers the shared base, restores the kernel's
  registers from its own sm_120 words, and keeps the transcript in the frame.
  - `w8m`: a "full" and an "empty" mbarrier a stage slot (SYNCS.ARRIVE and SYNCS.PHASECHK.TRANS64.TRYWAIT, words
    from ptxas 13.3), with a branch to a retry block where a warp needs the data. Best form (`w8m1` lane-0 arrives,
    `w8msb=1` waits on scoreboards nothing else waits for): +0.41 to +0.84% on two PCs, pipe 98.2-98.4%. Its cost is
    the waits: arrives alone +1.3% (pipe 99.03%), TRYWAITs without branches +1.0%, waited on without branches +0.67%.
    A check that does not suspend (`w8mnt`) is the same; a uniform branch (`w8mu`) hung on the card.
  - `w8mc`: one "all full" wait a chunk for turns 1-3: 96.9% pipe, +0.25% / +0.69%. A wait every warp shares couples
    them as a barrier does.
  - `w8g=4`: goal-nvk22's group protocol (each group of four warps, one a scheduler, syncs with itself or on the
    other group's arrive) on barriers 1-6: 97.36% / 95.81% pipe but 50-80 MHz more clock: +0.89 to +1.01% (PC 1),
    +1.26% (PC 2). On sm_120 a fourth barrier is an illegal instruction unless the compiled kernel has eight
    (`pearl_rot_c.inc` flag 32768, a `bar.sync 7` in the C fold's room); the code section's count alone is not
    enough. The form's ceiling (no ordering, timing only): +1.1 to +1.6%.
- **Clock-bound cards lose.** A 300 W RTX 5070 Ti at 2805 MHz (cap flag on, 289 W): `ah=3` saves 4.4 W at the same
  rate, and w8m's lost pipe is lost rate (-0.9%). w8g=4 gives up 2-3.5 points of pipe, so about -2% there. Hence the
  5090-only file.
- **The miner** (hashrate.js, 120 s legs taking turns with the shipped keys `rold`, same PC, then bench.sh's SRBMiner
  3.8.7 and PeakMiner 2.20.7 on that PC; verify-hits 400 of 400 for each new file first).
  - 575 W 5090, driver 621.01, memory 14201 MHz: `rold` 431.37, 431.53, 432.00, 431.75 (mean 431.66, spread 0.63;
    2553-2559 MHz, 543-546 W); **w8g=4** 438.04, 437.87, 438.05, 437.78 (mean 437.94, spread 0.27; 2635-2638 MHz,
    562-563 W): **+1.45%**. verify-hits 400 of 400 (764 hits, 39 salts). SRBMiner 442.9 (2647 MHz, 559 W),
    PeakMiner 425.6. Ours 97.5% of the best before, 98.9% after.
  - 575 W 5090, driver 610.88, memory 13801 MHz: `rold` 432.52, 432.98, 432.82, 432.73 (mean 432.76; 2498-2506 MHz,
    532-536 W); w8m 434.84, 434.62, 434.84, 434.80 (mean 434.78; 2540-2542 MHz, 558-559 W): +0.47%. verify-hits 400
    of 400 (759 hits). SRBMiner 438.6 (2570 MHz, 560 W), PeakMiner 423.7. Ours 98.7% before, 99.1% with w8m.
  - The board draws more, not less: 562 W against 545. What the block saves is on the rail the cap holds (the core's;
    see the section of 13 November, early), and the cap gives it back as clock. SRBMiner sits at the same 559-560 W
    and about the same clock.
- **The shipped file itself** (9da716e2, built on each box by build_rot.py `--arch sm_120w`; verify-hits 400 of
  400 on both: 931 and 940 hits, 47 salts each). Legs taking turns with `rold` again:
  - PC of driver 621.01: `rold` 431.86, 431.96, 431.69, 431.82 (mean 431.83, spread 0.27; 2554-2557 MHz, 543-545 W);
    the file 438.30, 438.17, 438.38, 438.36 (mean 438.30, spread 0.21; 2636-2640 MHz, 562-564 W): **+1.50%**, **99.0%**
    of SRBMiner's 442.9 on that PC (97.5% before).
  - PC of driver 610.88: `rold` 432.52, 432.71, 432.58, 432.55 (mean 432.59, spread 0.19; 2498-2501 MHz, 532-533 W);
    the file 434.99, 434.92, 434.79, 435.12 (mean 434.96, spread 0.33; 2568-2571 MHz, 566-569 W): **+0.55%**, **99.2%**
    of SRBMiner's 438.6 (98.6% before). Here the clock rises less (+2.8% against +3.2%) and the barriers cost a
    little more pipe (about 2.2 points against 1.7, from rate over clock).
- **Left:** the 5090 rows are at 99.0-99.2% of SRBMiner on these two PCs. The ordering costs 2-3.5 points of pipe
  that the clock only partly pays back; the form's ceiling is +1.1-1.6% in the probe, so w8g=4 has most of it. Untested: RTX PRO 6000
  (188 SMs, takes the file under the SM rule), and 5090s at a 450-500 W limit (more capped, should gain more).

### Salad, 14 November: the 5090 file on other RTX 50 cards and limits, and B's gathers priced for a taller warp tile (branch `goal/nv-50m`)

Nothing ships. The 5090 file stays 5090-only, and no rule by power limit is added. One timing-only rfold.py key,
`nbm=N` (B's last N pairs' gathers as moves: N=1 leaves six of B's eight LDSM a k-step), prices the B half of a
128 x 48 warp tile. All numbers are hashrate.js legs taking turns on one PC (100 s, or 120 s where said), then
bench.sh's SRBMiner 3.8.7 and PeakMiner 2.20.7 on the same PC (sample means). sm_120 is the shipped RTX 50 file's
keys (8584f90b), sm_120w the 5090 file's (9da716e2); both rebuild to their listed hashes with nbm in rfold.py.

- **The 5090 file on capped cards below the 5090: it loses.**
  - RTX 5080, limit 324 W (default 360), driver 621.01, "SW Power Cap" in every sample: sm_120 236.67, 236.06,
    236.07, 236.06 (mean 236.22, spread 0.61; 2775-2782 MHz, 324 W); sm_120w 234.69, 234.69, 234.74, 235.02 (mean
    234.79, spread 0.33; 2810-2815 MHz, 324 W): **-0.61%**. SRBMiner 237.9 (2816 MHz, 324 W), PeakMiner 236.2.
    Ours 99.3% of the best.
  - RTX 5070, limit 200 W (default 250), driver 610.60, at 200 W: sm_120 135.26, 134.97, 135.00, 134.97 (mean 135.05,
    spread 0.29; 2767-2772 MHz); sm_120w 133.47, 133.44, 133.46, 133.45 (mean 133.46, spread 0.03; 2782 MHz):
    **-1.18%**. SRBMiner 135.0 (2786 MHz), PeakMiner 134.0. Ours 100.0%.
  - The block's saving comes back as only 0.5-1.3% of clock on these board-capped cards, against its 2 points
    of pipe.
- **5090s at 575 W and 471.5 W: a tie, not the gain goal-nv50k expected at lower limits.** 120 s legs:
  - 575 W PC, driver 621.01, 79-81 C: sm_120 436.15, 435.67, 434.94, 435.33 (mean 435.52, spread 1.21; 2521-2529
    MHz, 546-552 W); sm_120w 435.05, 435.08, 434.74, 434.61 (mean 434.87, spread 0.47; 2576-2581 MHz, **575 W**):
    -0.15%. The 5090 file reaches the board limit on this hot card; on goal-nv50k's two 575 W PCs it stopped at
    562-569 W and won. SRBMiner 440.2 (2588 MHz, 575 W), PeakMiner 430.9 (2495 MHz, 538 W). Ours 98.9%.
  - 471.5 W PC, driver 617.14, warming from 64 to 79 C through the run: sm_120 407.64 (cold), 390.02, 389.36,
    388.61; sm_120w 397.23, 392.02, 388.76, 388.38 (2308-2359 MHz). Warm legs 389.33 against 388.57: -0.2%. The
    board limit binds both files there. SRBMiner 398.6 (2354 MHz, 471.5 W), PeakMiner 389.3. Ours 97.7%.
  - So the 5090 file wins where the board has about 10 W to spare at the core-rail cap, and ties where it has not.
    A rule by power limit would gain nothing measurable; the host's "128 SMs or more" stays.
- **B's gathers priced (nbm=1, TIMING ONLY).**
  - 575 W 5090: sm_120 + nbm 442.70, 447.6 against 435.5 (**+1.7%**, at 548-550 W); sm_120w + nbm 443.24, 442.75
    against 434.7 (+1.9%, at 575 W).
  - 471.5 W 5090: sm_120 + nbm 405.31, 404.25 (warm) against about 389.3 warm (**+4%**); sm_120w + nbm 409.32, 406.23.
  - Clock-bound cards: RTX 5080 (360 W limit, clock held by "Reliability" at 2760-2775 MHz, driver 621.24):
    sm_120 225.96, 225.69, 225.79, 225.53 at 316 W; nbm 225.69, 225.78, 225.80, 225.78 at 296 W (level, -20 W).
    SRBMiner 226.0, PeakMiner 225.6: ours 99.9%. RTX 5070 (250 W limit, 2812 MHz, driver 610.88): sm_120 137.55,
    137.40, 137.41, 137.35 at 220 W; nbm 136.98 (four legs) at 204 W (-0.3%: pipe). PeakMiner 136.9, SRBMiner
    136.7: ours 100.4%.
  - What a real 128 x 48 warp tile (block 256 x 192, or 128 x 384 with A shared by all eight warps) would pay back
    on the A side, per warp a k-step in the rotation fold: one more A gather (two LDSM; goal-nv50e's nold=2 puts an
    A LDSM at about 0.8% on a capped 5090), half a 16-byte store and 1.5 stage loads. Estimate: +0.5 to +1% on a
    575 W 5090, about +2% on a 471 W one, nothing on clock-bound cards. Registers allow it: B's fragments drop
    from 16 registers to 12 and A's stay two sets of four. Not built: it changes rfold.py's B loop (four pairs are
    written into about 30 places), pearl_rot.cu's block and readout, the thread's 12 x 16 pattern (16 x 12), and
    the host's hit mapping and tile counts.
- **A broken PC to skip:** one 575 W 5090 ran at 877 MHz with "Reliability" as the clock limiter (145 TH/s). Deleted.

### Salad, 14 November: the taller warp tile's A side, and A gathered by every warp; both lose (branches `goal/nv-50n`, `goal/nv-50p`)

Nothing ships. The shipped files are unchanged (sm_120 8584f90b, sm_120w 9da716e2). All runs are hashrate.js 100 s
legs taking turns on capped RTX 5090s in llmjob2 ("SW Power Cap" in every sample). rw is the shipped 5090 file,
rold the shipped sm_120 keys.

- **The 128 x 48 tile's A side, priced (rfold.py key `nax`, timing only, results still correct; verify-hits 400 of
  400, 552 and 544 hits).** nax repeats A's gather, store and stage loads so a warp pays what two more A tiles cost.
  On top of `nbm=1` (B's saving):

  | PC (driver) | rw | nbm=1 | nbm=1 + nax=1 |
  |---|---|---|---|
  | 575 W (621.14) | 433.22, 432.97 | 437.89, 436.36 | 422.27, 421.83 (2465 MHz, 491 W) |
  | 575 W (621.01) | 425.41, 425.20, 424.97 | 429.09, 429.15 | 414.19, 413.94 (2413 MHz) |

  The A side costs 3.5% against nbm=1, all as clock lost under the cap. So a real 128 x 48 tile would land about
  2-2.5% below the shipped files. Not built.
- **`ga=1` (with bt=0): every warp gathers its own six A tiles each turn, no stage, stores or barriers.** Correct
  (verify-hits 400 of 400 on two 5090s, 565 and 537 hits), and loses:
  - 575 W PC (621.14): rw 444.73, 443.84 (2632 MHz, 548 W), twin 444.29; rold 442.54, 441.14 (~2560 MHz, 510 W);
    ga 427.05, 424.74 (~2480 MHz, 456 W). **-3.6%** against rold.
  - 510 W PC (610.88): rw 431.93, 430.86, 431.30, 430.03, 430.24; rold 430.48 twice (~2496 MHz); ga 411.90, 411.89
    (~2410 MHz, 507 W). **-4.3%**.
  - ga's TH/s per GHz equals rold's (172.5): the pipe is the same, the whole loss is clock. On the 575 W PC it hit
    the cap at 456 W board power against rold's 510 W, so the capped quantity is not board watts.
- **Shared-memory instruction count, not bytes, sets the clock.** nax as a bitmask, timing only, 570 W 5090
  (621.24): rold 441.86, 442.02; nax=2 (+0.5 STS.128 a warp a k-step) 439.37, -0.6%; nax=4 (+1.5 LDS.128) 433.86,
  433.56, -1.9%. So an LDS.128 costs about 1.2%, an STS.128 about 1.1%, and from ga a gather (two LDSM and their
  subtracts) about 1.2%, although an LDSM moves half the bytes. A rule of 3-4% per KB (goal-nv50n) does not hold.
- **What follows:** by that count the shipped 96 x 64 warp tile (17 shared-memory instructions a k-step) already
  beats 128 x 48 (18) and 64 x 96 (18), and B shared through the exchange is level at best before its syncs. The
  next lever is fewer shared-memory instructions per IMMA at the same bytes each, for example an A tile held in
  registers across two turns where the k order allows. Not run: SRBMiner and PeakMiner on these PCs (a box move cut
  bench.sh off) and ga on a clock-bound 5080 or 5070.

### Salad, 14 November: the 5090 file's transcript word loaded a chunk early, and no sixth gather quad (branch `goal/nv-50q`)

The RTX 5090 file gains 1.0-1.6% and ships: build_rot.py's sm_120w keys add `trl=1 cq=0` (hash c580e560, was
9da716e2). The sm_120 file is unchanged (8584f90b). All numbers are hashrate.js 100 s legs taking turns on capped
RTX 5090s in llmjob2 (575 W limit, "SW Power Cap" in every sample), then bench.sh's SRBMiner 3.8.7 and PeakMiner
2.20.7 on the same PC (sample means). rw is the file before (9da716e2's keys), rw2 its twin.

- **Where the shipped 5090 loop stands.** A chunk of four k-steps: 192 IMMA, 38 LDSM, 21 LDS.128, 3 STS.128, 1 LDL,
  1 STL, 4 LDG.E.64, 5 barriers issued. That is 15.5 shared-memory instructions a k-step, the floor for an 8-warp
  block (255 registers a thread allow no more) with these gathers: for a warp tile of m A tiles by n B tiles
  (m n = 48), A shared by eight warps costs 1.25 m and B gathered costs n, least at 96 x 64; the transposed form
  (B shared by eight, A gathered: 2 m + 0.625 n) is also 15.5 at best. Every instruction already moves 512 bytes a
  warp. An A tile held in registers across two turns cannot save anything: each turn is another k-step, whose A
  fragment is new noise (rank indices drawn per k), and each is loaded once a k-step already.
- **The narrow memory steps, priced (timing only, run1).**

  | PC (driver) | rw | rw2 | ntr: no transcript LDL/STL | lgh: 2 of 4 record LDG gone | SRBMiner | PeakMiner |
  |---|---|---|---|---|---|---|
  | 575 W, board limit (610.88) | 427.71, 427.37, 427.16 (2690 MHz) | 427.71, 427.51, 427.01 | 434.37, 434.13 (+1.6%) | 434.65, 434.24, 433.79 (+1.6%) | 433.5 (2712 MHz) | 421.2 |
  | 510 W (610.88) | 431.37, 430.83 (2555 MHz) | 431.30, 430.52, 430.96 | 438.37, 437.98 (+1.7%) | 442.18, 441.22, 441.40 (+2.4%) | 435.7 (2587 MHz) | 423.8 |

  ntr's gain is all per clock (TH/s per GHz 158.7 to 161.1, 168.7 to 171.8): the tail stood waiting for the word's
  LDL, which went out four gaps ahead of its reader and often came from the L2. lgh's is mostly clock.
- **trl=T (rfold.py, correct):** the word's LDL goes out in gap 26 of turn T on the records' scoreboard, whose
  waits are placed for an L2 trip; the prologue zeroes the register for chunk 0's tail. trl=1: verify-hits 400 of
  400 (580 hits); 430.72, 429.53 against rw 425.81, 425.83, 425.61 (+1.0%) on the 575 W PC. It gets most of ntr's
  per-clock gain; the rest is the LDL and STL's own power.
- **Three more prices on that PC (run3):** lgs (two record LDG as LDS.64 from shared memory, timing only) 424.99,
  424.07 (-0.3%, it loses pipe); lgh=2 (one record LDG a chunk, timing only) 440.26, 440.00 (+3.4%, +70 MHz);
  cq=0 (no sixth gather quad, correct) 429.00, 428.82 (+0.7%).
- **pk (rfold.py key, correct, not shipped):** a chunk's four records in one LDG.128 a lane, packed as 7-bit rank
  indices (pearl_rot.cu ROT_PK, gprobe rot 6), the turn's word picked with three SEL on the pair's bits and parted
  with two to six ALU steps. CPU check 16,384 of 16,384 at k 2048, 8,192 of 8,192 at 8192, 512 of 512 at 65536 on two
  boxes; verify-hits 400 of 400 on two. But it saves no clock (pk 427.33 against rw 426.07; pk with trl 429.30,
  429.18 against trl alone 429.67, 429.53). So lgh's clock gain came from re-reading the last turn's B rows (the same
  data again), not from the load count. Its first build faulted ("misaligned address": pb1 = w >> 21 kept pb0's top
  bits); fixed with a mask.
- **The probe's w8 base needs eight barriers.** gprobe built with GP_ROTC=16387 and GP_W8=1 now fails "an illegal
  instruction" even with the shipped keys; with GP_ROTC=49155 (as the miner's base) the checks pass.
- **The shipped candidate, trl=1 cq=0 (run7), on a clean 575 W PC (driver 610.88, 559 MiB at idle):**
  verify-hits 400 of 400 (589 hits). rw 435.05, 435.21, 435.16, 437.48, 437.47, rw2 437.46 (2594 MHz); trl=1 cq=0
  441.81, 444.20, 444.19 (2581 MHz): +1.5% against the rw legs either side. trl=1 alone 441.34, 441.26, 443.54;
  pk with trl 440.58, 442.00, 442.90. SRBMiner 442.7 (2598 MHz, 575 W), PeakMiner 436.1. The new file: 99.8-100.3%
  of SRBMiner on that PC (the file before 98.3-98.8%).
- **Boxes to skip:** two 5090 PCs with 2.5-5.6 GB of card memory in use and 4-10% load at idle (one failed every
  probe, the other's legs had sd 8). And Salad twice did not restart a box after set-env (it kept the old container).
- **The file CI ships (build_rot.py --arch sm_120w at 4273f85a), built on two boxes (run8):** hash c580e560 on both
  Linux PCs and on Windows. verify-hits 400 of 400 on both (781 and 788 hits). Legs against rw:
  - 575 W PC above (driver 610.88): rw 437.41, 437.50, 437.50 (2594 MHz); new file 444.28, 444.30, 444.31 (2581
    MHz): **+1.57%** (spreads 0.02 / 0.01%). SRBMiner 442.7 (run7): **100.4%**.
  - second 575 W PC (driver 610.88, 514 MiB at idle): rw 441.26, 441.22, 441.27 (2622 MHz); new file 447.45, 447.52,
    447.51 (2602 MHz): **+1.41%** (spreads 0.01 / 0.02%). SRBMiner 446.9 (2634 MHz), PeakMiner 443.7: **100.1%**.
- **What is left on the 5090:** the gain is per clock; the new file runs 13-20 MHz below the old under the same cap.
  Next to try: fewer transcript steps (the LDL/STL's own power: about 0.6%, ntr against trl), and a k order inside
  each chunk that makes consecutive turns of a warp re-read B rows they just read (lgh showed that repeated B data
  saves clock), as long as it keeps the bank-collision order.

### Salad, 14 November: re-read B rows and a transcript in word pairs; neither wins (branch `goal/nv-50r`)

Nothing ships. The 5090 file stays c580e560 and the sm_120 file 8584f90b. All numbers are hashrate.js 100 s legs
taking turns on RTX 5090s in llmjob2. rship is the shipped 5090 file built on the box (build_rot.py, hash
c580e560), rt its twin built by hand from the same keys.

- **Re-read B rows save nothing (idea 1 dropped).** The order of k inside a chunk is the host's
  (rot_order_chunk), and any order gives the same sums, so a k order that repeats B rows across a warp's turns
  would be a host change only. In the 5090 loop pair p folds k-steps p, p^2, p^3, p^1. Priced first with
  pearl_rot.cu `ROT_DUPB` (timing only, every record load kept; on branch `goal/nv-50r` ddd36e8e, not on this one): k-steps 2 and 3 take k-steps 0 and 1's B rank
  indices, so two of every three turn changes re-read all B rows (`d1`; `d2` only the first sign's rows).
  - 460 W PC (driver 591.86, capped): rt 399.19, 398.92, 398.26, 398.31; d1 400.13, 399.00, 398.13, 398.06;
    d2 399.26, 399.10. All within 0.3%.
  - 600 W PC (driver 621.14, 545-554 W drawn): rt 441.01, 441.06, 440.97; d1 441.01, 441.11, 440.99.
  - So goal-nv50q's lgh gain did not come from repeated B data. And a real order could repeat far less anyway: a
    simulation of moving k inside their octets (bank cost unchanged) repeats 5.5% of rows, against d1's 67%.
- **The transcript's load and store cost about 1% on a capped card, nothing on an uncapped one.** Timing only:
  `ntr` (no transcript LDL/STL), `trh` (rfold.py, new: both run every other chunk, under P3 from the word
  pointer's bit 2).

  | PC | rship | rt | trh | ntr |
  |---|---|---|---|---|
  | 460 W (591.86), "capped" | 398.25, 398.00 | 398.26, 398.31 | 400.86, 399.74, 399.56 (+0.46%) | 402.43, 402.02 (+1.0%) |
  | 600 W (621.14), 545-554 W | 441.26, 440.70 | 441.01, 441.06, 440.97 | 440.66, 440.35, 440.49 | 440.70, 440.93, 440.75 |

- **Word pairs built correctly, twice; both lose.** An even chunk loads its word and the next with one LDL.64,
  the odd chunk stores both with one STL.64. asm89 learned LDL.64/STL.64, LOP3 into a predicate, UIADD3 with an
  immediate and ISETP against a uniform register (ptxas 13.3's words, read back through nvdisasm).
  - `trp=1`: the pair in R250:R251, the word pointer gone, each address worked out from the count (four ALU
    steps, twice). CPU check 16,384/16,384 at k 2048, 8,192/8,192 at 8192, 512/512 at 65536; verify-hits 400 of
    400 (713 hits). 460 W PC: tp 397.49, 397.78, 397.23 against rship/rt 398.20-398.84 (**-0.25%**). 600 W PC
    drawing 457 W (driver 610.88): tp 451.96, 452.52, 452.45, 452.42 against 452.89-453.07 (-0.12%).
  - `trq=U`: the chunk count in uniform register U (UR36 in the miner's file), which frees R250 for the pair and
    keeps the pointer: three steps more a loop, not eight. verify-hits 400 of 400 on two 5090s (721 and 776 hits;
    the probe's base names UR36, so its CPU check did not build). 460 W PC: tq 398.18, 398.28 against rship 399.64
    (cold), 398.94 and rt 398.97 (**-0.18%**). Cut short when the slots went to the RTX table re-run.
  - So trh's +0.46% is the price of half the loads before any extra step, and three extra issued steps a chunk
    (two predicated-off) already cost more than that. Whatever replaces the transcript's LDL/STL must add no
    steps to the loop.
- **SRBMiner 3.8.7 on the 600 W PC:** 440.7 TH/s (2609 MHz, 555 W), PeakMiner 2.20.7 426.9; the shipped file
  441.0 (2583 MHz): **100.1%**.
- **Not done:** the sm_120 file (5080/5070 class), since nothing won on the 5090. Boxes to skip: a 431 W 5090 with
  1.5 GB in use at idle (legs sd 8-29), and two 600 W boards that never reached their cap.

### Salad, 15 November: why the competitors lead on RTX 50, seen from outside; nothing ships (branch `goal/nv-50s`)

Nothing ships. The sm_120 file stays 8584f90b and the 5090 file c580e560. The competitors were only run and
watched: their own hashrate, `nvidia-smi` beside them (clock, power, the clock's limiter every 20-30 s), and their
shares through the stand-in pool (`sharetap.py`). Ours is hashrate.js on the shipped file built on the box
(rship), 100 s legs; "per GHz" is TH/s per GHz of SM clock. All boxes in llmjob2.

- **The 14 November RTX 5060 row `6f073405` (96.9%) is a noisy PC and an unequal figure.** bench.py takes a
  competitor's last reading and our mean after the first minute. On that PC OneZeroMiner's samples ran 78.5-83.1
  and its last was 83.0; its mean was 80.6. Means for all four: SRBMiner 81.6, PeakMiner 81.5, OneZeroMiner 80.6,
  ours 80.4 (ours swung 73.9-83.4 and held 83.4 for 8 samples in a row).
- **A clean RTX 5060 (155 W limit, driver 617.42) is clock-bound, not power-bound.** Every miner runs 2812 MHz,
  held by "Reliability" (the voltage limit). Ours 85.60, 85.55 at 145 W; SRBMiner 3.8.8 85.77 at 137.5 W;
  OneZeroMiner 1.8.6 85.42; PeakMiner 2.20.8 85.44. Ours 99.8% of the best: 0.3% of work per clock.
- **On capped cards the best competitor runs a higher clock at the same watts.**

  | Card (limit, driver) | Ours | SRBMiner 3.8.8 | OneZeroMiner 1.8.6 | PeakMiner 2.20.8 |
  |---|---|---|---|---|
  | RTX 5090 (575 W, 610.88) | 436.1 at 2584 MHz (168.8 per GHz) | 437.1 at 2636 MHz (165.8) | 430.8 at 2602 MHz, 564 W (165.5) | 423.4 at 2529 MHz, 548 W (167.4) |
  | RTX 5080 (288 W of 360, 621.14) | 226.65 at 2662 MHz (85.2) | 231.2 at 2737 MHz (84.5) | 230.3 at 2709 MHz (85.0) | 227.2 at 2666 MHz (85.2) |

  "SW Power Cap" in nearly every sample for every miner. On the 5090 SRBMiner does 2.1% less work per clock and runs
  2.0% more clock: per hash the two are level (1.315 against 1.317 J/TH; ours 99.8%). On the 5080 SRBMiner does
  0.8% less per clock and runs 2.8% more clock: about 2% less energy a hash (1.246 against 1.271 J/TH; ours 98.0%).
  So it is lower power per hash at the cap, not more work per clock and not a setting.
- **SRBMiner 3.8.8 changed its tile.** Its shares: k 65536, rank 128, a thread's region 16 A rows (8 apart, over
  128) by 12 B columns (pairs 64 apart, over 384): a 128 x 48 warp tile, a 128 x 384 block with A shared by all
  eight warps. SRBMiner 3.8.2 sent 24 x 8. Ours is 12 x 16 (96 x 64; the 5090 file's block 96 x 512). OneZeroMiner
  on the 5060: 12 x 16. goal-nv50k priced this 128 x 48 tile on paper (+0.5-1% on a 575 W 5090) and did not build it.
- **The 128 x 48 tile in the C form loses (`ROT_T128`, pearl_rot.cu, TIMING ONLY).** The sm_120 C form (ROT_C=11,
  two bands of four warps) with 8 m16 by 6 n8 tiles a warp (c8) against 6 by 8 (c6), same work a tile:
  5090 c6 403.21, 402.54, 402.24, 402.31; c8 411.41 (cold), 400.61, 400.48, 400.23 (**-0.5%**, same clock). 5080
  c6 211.21, 211.25, 211.38, 211.38; c8 209.97, 210.03, 210.15, 210.13 (**-0.6%**, 2676 against 2678 MHz). In the band
  form A is shared by four warps only, so this is not SRBMiner's block; the A-shared-by-eight form (a 128 x 384
  hand loop) is not priced.
- **Our CLI loses nothing against hashrate.js** (5090, the 575 W PC, CLI from source on the stand-in pool at 2^206,
  one new job each 30 s): hashrate.js 435.26, 435.93, 436.09; CLI 434.97, 435.73 (-0.1%). The 2b31ad9b row's gap is
  the card's, not the CLI's.
- **Cache-eviction hints on the 5090 loop: nothing** (rfold120.py `lgev` on the record LDG, `trev` on the transcript
  LDL/STL, correct by construction; verify-hits 400 of 400 on both with `trev`): rship 435.46, 435.03, 435.05, twin
  435.36, 434.95; records .EF 435.25, 435.14; records .NA 432.51, 432.21 (-0.65%); transcript .EL 435.09, 434.98;
  both 435.25, 435.21. The transcript word is not slowed by evictions from the L1. On the 5080 file records .EF
  225.74, 225.75 and .NA 224.39, 224.38 against rship 226.11, 225.74, 228.09 and twin 225.86, 227.15: lose.
- **goal-nv50q's trl/cq lever on the sm_120 file is not worth porting** (5080 at 288 W, TIMING ONLY): rship 225.72,
  225.71, 225.70; twin 225.70, 225.71; ntr (no transcript LDS/STS: trl's ceiling) 226.34, 226.33 (+0.28%); cq=0
  224.74, 224.86 (-0.4%); both 225.19, 225.21.
- **The 5090 file now beats the sm_120 file on a capped 5080, and still loses where the card is not capped**
  (the file as it ships, rw: c580e560's keys, verify-hits 400 of 400 on the 5080):
  - RTX 5080, 288 W of 360 (621.14), "SW Power Cap": rship 228.83, 229.05, 228.94; twin r0 228.81, 229.01; rw 230.00,
    230.02, 230.14, 230.11 (**+0.49%**, 2715 against 2691 MHz). goal-nv50k measured the 5090 file before trl/cq at
    -0.61% on a capped 5080.
  - RTX 5070, 250 W at its default (610.88), at its top clock (no limiter, 243 W): rship 139.03, 138.94, 139.22; r0
    139.19, 139.00; rw 138.82, 138.85, 138.82, 138.84 (-0.17%).
  - RTX 5070 Ti, 300 W, clocks locked by the owner at 2328 MHz (163 W): rship 165.74, 165.81, 165.90; r0 165.75,
    165.88; rw 165.03, 165.21, 165.14, 165.19 (-0.4%).
  - So taking the 5090 file on RTX 50 cards whose power cap holds the clock would gain about 0.5% there. The host
    picks by SM count (128 or more); a rule by "power limit set below the default" would catch this 5080 and miss
    the uncapped 5070. Not shipped on one PC.
- **OneZeroMiner `--power-save`** (level 1, an `ozmps` leg on the goal branch; bench.sh's `ozm1`-`ozm3` legs now do this; its notes: a V/F adjustment on top of overclock
  settings, for the 5070 Ti, 5080 and 5090): 428.1 at 2577 MHz, 556 W on the 5090 against 430.8 without. The
  official table runs it without, so it explains no row.
- **Stand-in pool:** `VPOOL`'s bits are the target's exponent, so 214 gives 16 times more shares than 210, not fewer.
  At 12 shares a second SRBMiner fell to 327 TH/s (GPU use 60-80%). Use 206 or lower for a low share rate.
- **Tools added here:** nv50box.sh stage `cli`, rfold120.py `lgev`/`trev`, pearl_rot.cu `ROT_T128` (all default
  off; the shipped files rebuild to their listed hashes). bench.sh's OURS_SRC and release legs fail inside nv50box.sh ("pearl_core.node not
  found"); not chased.
- **Next:** build SRBMiner's 128 x 384 block (A shared by eight, 8 m16 by 6 n8 a warp) as a hand loop on the w8 base:
  rfold.py's B loop at six n8 tiles, a 16 KB stage, the host's tile layout, hit index and proof tile at 16 x 12.
  Registers allow it (B's fragments 12 registers, not 16). And the host rule above, with a second capped card first.

### Salad, 15 November: RTX 50 cards time both fold files, and the 128 x 384 block priced with drawn operands (branches `goal/nv-50t`, `goal/nv-50u`)

The two shipped files are unchanged (sm_120 8584f90b, sm_120w c580e560, both built on every box). What changes is
the host: an RTX 50 card with both files beside the core loads both, and the startup launch-shape trial (the RTX 30
one) times the two tiles on the card and keeps the rule's file unless the other wins by 0.3% in two thirds of the
rounds. All numbers are hashrate.js 100 s legs taking turns, or our CLI on the stand-in pool (2^206, a new job every
30 s, mean after the first minute), in llmjob2. rshipw/rw is the 5090 file, rship the sm_120 file. Proxies are TIMING
ONLY: rn = rfold.py `nbm=1` (B's last pair's gathers made moves), rx = `nbm=1 nax=7` (the 128 x 384 block's step mix
less one stage load a chunk), rx1 = `nbm=1 nax=1`, r4 = `nbm=1 nax=4`, ro = `nax=7`, rbr = `nbr=1` (new, below),
rz = `nbr=1 nax=7`.

- **The 5090 file on capped 5080s, again (goal-nv50t):** 5080 at 288 W of 360 (621.01), "SW Power Cap": rship
  228.08, 228.06, 227.99 (2677 MHz), twin 228.08, 228.03; rw 229.24, 229.22, 229.15 (2701 MHz): **+0.51%** (twin
  0.04%). 5080 at 252 W (621.14), CLI: f:n (192 x 256 forced) 219.90, f:w (96 x 512) 220.95: +0.5%. A 5070 Ti at its
  300 W default near its top clock: rship 191.11, 191.33, 190.39, rw 190.90, 190.65, 190.69 (-0.1%). A 5080 held at
  2700 MHz by its voltage limit (306 W of 360, 621.24), CLI: f:n 229.35, trial 228.48.
- **5090s, CLI:** 450 W PC (617.42, 80-85 C): f:n 373.69, 373.98; f:w 379.04, 379.16 (+1.4%).
- **nbm flatters the B side.** Its moves make the last pair's fragments the fill, a constant, and constant operands
  cost less energy at the cap (goal-nv50e). `nbr=1` (rfold.py, new) drops the pair's gathers and lets its IMMA read
  pair 1's drawn fragments (pair 3's unwritten registers in a k-step's last tile, 2 of 48 IMMA).

  | PC | rshipw | rn / rbr | rx | rz | other |
  |---|---|---|---|---|---|
  | 5080, 288 W (621.01; goal-nv50t) | rw 229.24, 229.22, 229.15 | rn 230.74, 230.72 | 232.14, 232.02 | | |
  | 5080, 252 W (621.14), run3 | 220.96, 221.28, 221.14 | rn 224.19, 223.40 | 222.99, 223.39 | | rx1 224.57, 224.73 |
  | same PC, run4 | 221.27, 220.96, 221.50 | rbr 223.62, 224.10 (+1.2%) | 223.96, 222.98 | 222.78, 222.86 (+0.7%) | |
  | 5090, 575 W (621.14), ~537 W drawn | 435.17, 435.18 (cold first 438.69) | rbr 445.93, 446.15 (+2.5%) | 432.48, 432.27 (-0.6%) | 436.39, 436.37 (+0.3%) | |
  | 5090, 450 W, 80-85 C (617.42) | 378.97, 378.76 (cold first 386.61) | rn 394.10, 394.04 | 389.41, 389.18 | | r4 392.30, 392.01; ro 371.30, 371.15 |
  | 5080 at its 2700 MHz voltage limit | 228.68, 228.67, 228.73 | rn 225.82, 225.94 | 227.77, 227.72 | | r4 227.91, 227.86; ro 228.61 twice |

  The real 128 x 384 block would land near rz less one stage load a chunk (about -0.3%): level on a normal 575 W
  5090 and about +0.4% on a capped 5080, before any schedule it would lose. Not built. B's two gathers a k-step are
  worth 1.2-2.5% at the cap, but the taller tile's extra A work takes it back. On a clock-bound card nbm costs 1.2%
  of work per clock.
- **Competitors on the same PCs (mean of samples):** 5080 at 252 W: OneZeroMiner 1.8.6 222.73 (2645 MHz), SRBMiner
  3.8.8 221.32 (2696 MHz); ours rshipw 221.24 (2652 MHz): 99.3%, behind OneZeroMiner by work per clock (84.2 against
  83.4 per GHz), not clock. 5090 at 575 W: SRBMiner 3.8.8 444.01 (2613 MHz, 571 W drawn), OneZeroMiner 433.00; ours
  435.18 (2543 MHz, 539 W): 98.0%. Both under "SW Power Cap", but ours is held at 539 W of board power and SRBMiner
  at 571 W, and rbr drew 565 W at 2600 MHz: what binds is not board watts but the share of them our loop puts on
  the limited rail. A 2-GPU 5090 box's competitor legs mined both cards: not used.
- **The trial needed two fixes, then worked.** A trial turn ended only when nonce reached the salt's span, and the
  rotation fold's batches do not divide it (2688 tiles a batch on a 5080, 697,344 a salt), so the core redrew the
  salt first and the trial never left its first turn: no verdict in 300 or 600 s, and no tile switch. A turn now
  ends where the salt is redrawn. And since those salts take 4-7 s, rounds counted for RTX 30's 4 s turns would
  take 7-10 minutes, so the trial also ends at a round's end once shapeTrialSecs have passed and 3 rounds are done.
  Our CLI, 600 s, stand-in pool, after both fixes:
  - 5090, 600 W limit (621.01; ~500 W drawn, capped): "kept the default rotation 96 x 512, against the default:
    rotation 192 x 256 -1.44% (ahead 0 of 5)". 243 shares, 243 accepted, 0 rejected.
  - 5080 at 252 W: "kept the default rotation 192 x 256, against the default: rotation 96 x 512 +0.18% (ahead 2 of
    3)". 137 shares, 137 accepted, 0 rejected. Forced legs on this PC had 96 x 512 at +0.5%, so three rounds read
    the gain short and the trial kept the rule's file: a missed gain, not a loss. A longer trial for the rotation
    fold is the next step.
- **Records, priced (TIMING ONLY but rpk).** `pk=1` (a chunk's four records in one LDG.128, correct): 5080 at 252
  W 220.17, 220.48 against rshipw 220.93, 221.85, 221.08 (-0.4%); 600 W 5090 449.62, 446.48 against 450.15, 447.43,
  446.19 (level). `lgh=2` (one record load a chunk; turns 1-3 reuse turn 0's record, so their B gathers read the
  same rows): 5080 227.42, 227.76, 227.69, 228.07 (+2.9%, 2723 MHz against 2650); 5090 461.57, 456.28, 460.46,
  459.65 (+2.6%). `lgh=3` (rfold.py, new: lgh=2's repeated records with every load kept): 5080 226.10 (+2.2%), 5090
  456.63, 456.27 (+1.8%). So about two thirds of lgh's gain is the repeated B rows, which no correct fold can have,
  and a third (0.7-0.8%) the three record loads, which pk saves but gives back in its extra steps.
- **Boxes to skip:** a 5090 at "SW Power Cap" with 295 W drawn of 575 (2085 MHz), a 5090 stuck at 886 MHz with 405
  MHz memory.

### Salad, 15 November: the shipped trial host code verified, a 600 s trial for RTX 50, and what work costs under the cap (branch `goal/nv-50w`)

Shipped: RTX 50 and RTX PRO Blackwell cards time their two rotation fold files for 600 s, not 150 s (ef136b4a,
shapePick.js `ROT_TRIAL_SECS`, picked by the card's name). The two cubins are unchanged (sm_120 8584f90b, sm_120w
c580e560). Boxes in llmjob2. Ours is hashrate.js 100 s legs taking turns, or our CLI on the stand-in pool (2^206).
Competitors are bench.sh sample means. powcap.cu (new, probes/hand89) runs one kind of work at a time at full
occupancy and reads NVML every 100 ms.

- **verify-hits, shipped host code (881bbfb4), both files beside the addon:** RTX 5090 (480 W of 600, 621.14):
  sm_120 file 400 of 400 (564 hits), sm_120w 400 of 400 (560). RTX 5080 (360 W, 621.01): 400 of 400 (543), 400 of
  400 (533). Earlier, on a noisy 5080 (621.24): 288 of 288 on each in 150 s.
- **The 600 s trial, in our CLI (900 s):** 5090 at 480 W: 21 rounds, kept 96 x 512, with 192 x 256 at -0.31%
  (ahead 2 of 21). Forced legs agree: rshipw 424.43, 424.25, 424.28 against rship 423.25, 422.93. 353 of 353 shares
  accepted. 5080 at 360 W, near its voltage limit (2770-2782 MHz): 11 rounds, kept 192 x 256, with 96 x 512 at
  -0.25% (ahead 0 of 11). Legs agree: rship 235.86, 236.06 against rshipw 233.86 (sd 4.6), 235.64, 235.68. 180 of
  180 accepted. A capped 5080, the case the longer trial is for, did not come up.
- **Same PCs, against the competitors:**

  | Card (limit, driver) | Ours (best file) | SRBMiner 3.8.8 | OneZeroMiner 1.8.6 | % of best |
  |---|---|---|---|---|
  | RTX 5090 (480 W, 621.14) | 424.32 at 2469 MHz, 480 W | 427.03 at 2507 MHz, 480 W | 425.60 at 2475 MHz, 480 W | 99.4% |
  | RTX 5080 (360 W, 621.01) | 235.96 at 2770 MHz, 360 W | 234.87 at 2782 MHz, 337 W | 236.07 at 2781 MHz, 349 W | 99.95% |

  On the 480 W 5090, every miner sits at exactly 480 W. There the board total binds, and SRBMiner gets 1.5% more
  clock from the same watts: less energy a hash (1.124 against 1.131 J/TH). rbr (timing only) 429.21, 429.11 at
  2504 MHz.
- **What binds a high-limit 5090 below its board limit.** NVML on RTX 50 answers only the GPU scope: module and
  memory power return "not supported", and there are no rail readings. Driver 621's nvidia-smi has "Board Limit" as
  a reason of its own beside "SW Power Cap". Under our fold, SW Power Cap is active and Board Limit is not. With a
  480 W limit we reach 480 W; goal-nv50v's 575 W and 600 W PCs held us at 539 W and 502 W. So at high limits a
  budget inside the chip binds, not the board total.
- **powcap on a 5080 held near 2782 MHz by its voltage limit (watts at the same clock = energy):**

  | Work | W | Work | W |
  |---|---|---|---|
  | idle | 15 | IMMA, random operands, A held for 4 (as our loop holds it) | 277.3 |
  | IMMA, zero operands | 133.5 | the same in a snake / B held for 6 / both changing | 277.7 / 272.8 / 283.8 |
  | IMMA, bytes 0-127 (as ours), A held | 260.5 | the same, snake / B held | 259.8 / 256.5 |
  | + one LDS.128 feeding the operands every 16 IMMA | 272.9 | every 8 / every 4 | 295.4 / 324.4 (capped) |
  | every 8, snake / B held | 291.8 / 291.2 | FP32 FMA | 320.5 (capped) |

  The operands' values set the IMMA's power. Zeros cost half as much, and order moves it by 1-2%. In goal-nv50e the
  real loop gained nothing from holding B, so the order is not worth a loop. Every shared-memory step that feeds
  operands is dear: one LDS.128 per 8 IMMA adds 13% to the IMMA's watts. Our 5090 loop issues 62 shared-memory steps
  per 192 IMMA. (The first run's plain LDS mode drew 289 W at 2785 MHz. In the second build the compiler emptied
  the plain LDS and ldmatrix modes, so they are not listed.) A repeat of the random-operand IMMA read 283.5 W
  against 277.3 W the first time. The order differences (272.8 to 283.8 W) are within that drift.
- **powcap on a 575 W 5090 (621.01; 86 C, a noisy PC): which work the cap holds below the board limit.**

  | Work | Board W | MHz | IMMA work a second |
  |---|---|---|---|
  | IMMA, bytes 0-127 | 547.9, 564.8, 566.7 (SW Power Cap, below 575) | 2805-2814 | 840-843 |
  | IMMA, zero operands | 293.6 (no cap: voltage limit) | 2863 | 861 |
  | + one LDS.128 every 16 / 8 / 4 IMMA | 575 / 575 / 575 | 2785 / 2739-2746 / 2692 | 832 / 808-809 / 777 |
  | + one LDG.128 every 8 IMMA from the L1 / the L2 | 575 / 563.1 | 2766 / 2827 | 713 / 265 (the loads' latency) |
  | + a local word's LDL/STL / a shared word's LDS/STS every 8 | 575 / 575 | 2743 / 2768 | 778 / 707 |
  | FP32 FMA | 575 | 2676 | |
  | 16-byte loads of a 32 MB / 1 GB buffer | 523.2 / 408.3 (below 575) | 2850 / 2812 | |

  IMMA alone and loads from the L2 or DRAM are held below the board limit. Add shared-memory or FP32 work and the
  card reaches it. So the budget that binds below the board limit is loaded by the tensor work and by L2/DRAM
  traffic, much less by the rest. (The plain load kernel's byte rate is too high to be real, so take its two rows
  only as a sign. The fed mixes stall on their loads, so their work rates are not prices.)
- **This round's 5090s all reached their board limit with our fold:** 480 W (621.14) and 575 W (621.01, 86 C).
  goal-nv50v's 575 W PC held us at 539 W and a 600 W one at 502 W. So the missing 30 W belongs to some PCs, not
  to the fold. Where the board total binds, SRBMiner gets 1.5-1.6% more clock from the same watts on both PCs.
  We do about 1% more work per clock, so it ends 0.6-0.9% ahead.

  | Card (limit, driver) | Ours | SRBMiner 3.8.8 | OneZeroMiner 1.8.6 | % of best |
  |---|---|---|---|---|
  | RTX 5090 (575 W, 621.01, 86 C) | 420.01 at 2625 MHz, 575 W | 423.41 at 2667 MHz, 575 W | 423.91 at 2640 MHz, 575 W | 99.1% |
  | RTX 5080 (360 W, 610.88, clock-bound 2790 MHz) | 219.17, 339 W | 217.08, 319 W | 218.57, 332 W | 100.3% |

  rbr (timing only) on the 575 W PC 426.09, 426.13 at 2665 MHz (+1.45%). On the 5080: rship 219.35, 218.95, 219.21
  against rshipw 218.48, 217.94, 218.26. The 600 s trial kept 192 x 256 (96 x 512 -0.27%, ahead 4 of 10), and 183
  of 183 shares were accepted.
- **The loop's L1/L2 steps where the board total binds (same 575 W PC, timing only):** rshipw 422.07, 420.11, 419.80
  (2622-2636 MHz). lgh=2 (one record LDG a chunk, turns 1-3 reusing turn 0's record) 434.33, 434.49 at 2700 MHz:
  +3.3%. lgs (two record LDG as LDS.64) 419.35, 419.50: -0.3%. ntr does not assemble with trl=1 (rfold.py's
  assert), so its legs fell back to the release fold and are not counted. lgh's gain is mostly repeated operands
  (goal-nv50v's lgh=3), which no correct fold has. pk, which keeps the loads' saving and is correct, was level
  before (goal-nv50q). powcap says the L2 bytes are dear, but the record stream mostly hits the L1.
- **Where this leaves RTX 50:** on clock-bound 5080s we are level or ahead (99.95%, 100.3%). On 5090s held at their
  board limit we are 99.1-99.4%. SRBMiner turns the same watts into 1.5% more clock, and we give back about 1% in
  work per clock. No order, value or load change priced here closes that correctly. The changes that do (rbr's
  fewer B gathers, lgh's repeated records) are wrong on purpose. The 128 x 384 block, which takes one B pair of
  four away for good, is the design left. goal-nv50v priced it at about level to +0.4% with drawn operands.

### Salad, 15 November: SRBMiner's 128 x 384 block as a real hand loop; it loses (branch `goal/nv-50x`)

Nothing ships. The shipped files are unchanged (sm_120 8584f90b, sm_120w c580e560; both still rebuild from the
branch, and the sm_89 and sm_86 loops give the same bytes as before). All numbers are hashrate.js 100 s legs taking
turns in llmjob2, then bench.sh's SRBMiner 3.8.8 and OneZeroMiner 1.8.6 on the same PC (mean of samples from 30 s).
rshipw is the shipped 5090 file built on the box, rw2 the same keys built by hand (the A/A twin).

- **The block, built (rfold.py `t128`).** 8 m16 by 6 n8 tiles a warp (a tile is 6 gaps, a turn still 48 IMMA), A
  shared by all eight warps through a 16 KB stage (a k-step's slot 4 KB, a warp's half its four tiles), three B
  pairs a warp, the 5090 file's group barriers at the new gaps (pbg=18 wag=19 rag=40). A chunk a warp: 32 LDSM (the
  5090 file 38), 28 LDS.128 (21), 4 STS.128 (3), 192 IMMA. `t128=1` runs it in the 96 x 512 kernel: the same work a
  tile and the same bank pattern, but the transcripts are wrong, so TIMING ONLY. `t128=2` is the layout a real
  128 x 384 kernel would have; it was not needed. `t128s=1` moves B's gathers four gaps earlier.
- **Results.**

  | PC | rshipw | rw2 (twin) | rt (t128) | rt1 (t128s) | rbr (one B pair fewer) |
  |---|---|---|---|---|---|
  | 5090, 600 W limit, ~531 W drawn, SW Power Cap (596.36), run1 | 444.01 (cold), 442.67 (2583 MHz) | 442.81, 442.55 | 440.06, 440.42, 440.50 (2576 MHz, 526 W): **-0.53%** | | 453.46, 453.52 (2647 MHz, 558 W) |
  | same PC, run2 | 444.64 (cold), 442.61 | 442.64 | 440.13, 440.33: **-0.55%** | 439.96, 440.32: -0.55% | |
  | 5080, 360 W, at its cap and top clock (2797 MHz, 616.92) | 237.75 (cold), 236.77 | 236.93, 236.83 | 236.44, 236.44, 236.39: -0.17% | | 236.69, 236.67 |
  | 5080, 360 W, SW Power Cap at 2775 MHz (621.24) | 235.06 (cold), 234.82 | 234.85 | 234.24, 234.21: **-0.27%** | 234.09, 234.12 | |

  Twin spreads 0.03-0.06%. The block runs 7 MHz lower on the 5090 at 5 W less: it costs more energy a tile.
- **Why.** A warp's shared-memory steps a k-step are 1.25 m + n for m A tiles (A shared by eight: 7/8 of a load and
  1/8 of a gather pair and a store each) by n B tiles (one gather each). 6 x 8 is 15.5, 8 x 6 is 16. B's two fewer
  gathers a k-step are worth what rbr shows, but the A side takes 7 more LDS.128, 2 more LDSM and 1 more STS.128 a
  chunk. At about 0.3% a shared-memory step a chunk under the cap (goal-nv50n's nax, this run's rbr), that is the
  -0.5% measured. The gather schedule is not the cause (rt1 the same). Of the shapes A shared by eight allows, 96 x 64
  a warp (the 5090 file) has the fewest steps; 64 x 96 is 17.
- **The transcript, priced again (goal brief item 4).** rnt is the 5090 file without trl (trl=0), rntr the same with
  ntr=1 (no transcript load or store, TIMING ONLY).
  - 5090: rnt 440.59, 440.78 (2600 MHz, 536 W; -0.43%); rntr 444.48, 444.38 (2595 MHz, 531 W): **+0.40%** against
    rshipw/rw2 442.6.
  - Capped 5080: rnt 233.49, 233.53 (-0.57%); rntr 235.23, 235.35 (+0.20%).
  - So trl already has about half of what the transcript's two local-memory steps cost; the other 0.2-0.4% would need
    the transcript in shared memory, and the 96 x 512 kernel has 11 KB free of the 16 KB that takes.
- **Competitors on the same PCs.**

  | Card (limit, driver) | Ours (rshipw) | SRBMiner 3.8.8 | OneZeroMiner 1.8.6 | % of best |
  |---|---|---|---|---|
  | RTX 5090 (600 W, 596.36) | 442.6 at 2583 MHz, 531 W | 452.57 at 2661 MHz, 564 W | 439.21 at 2561 MHz, 513 W | **97.8%** |
  | RTX 5080 (360 W, 621.24) | 234.8 at 2775 MHz, 360 W | 235.76 at 2792 MHz, 360 W | 235.69 at 2776 MHz, 360 W | 99.6% |

  On this 5090 every miner is held by SW Power Cap well below the 600 W limit, and SRBMiner gets 3.0% more clock
  while drawing 33 W more from the board: whatever budget binds there, SRBMiner puts less of its watts on it. We do
  0.8% more work per clock (171.4 against 170.1 TH/s per GHz). This is the worst case seen for us (goal-nv50v's
  98.0% was the same kind of PC); where the board total binds we are 99.1-99.4%.
- **Left.** By count of shared-memory steps the 5090 file's shape is the floor for A shared by eight warps and B
  gathered, and the steps that remain (B's gathers, A's stage loads, the readout, the records, the transcript) are
  each priced and either needed or already at their best form. Not tried: A shared across a cluster of two blocks
  (distributed shared memory: about 1.5 steps a chunk fewer on paper, before the cluster's barriers and the remote
  stores).

### 17 Nov 2026: three ideas from blackwell_pearl_miner on an RTX 5090 (branch goal/rtx50b)

blackwell_pearl_miner (bpm, ISC, an open sm_120 miner) gave three ideas: ptxas -O levels, a column order inside each
128-k block that avoids shared-memory bank conflicts, and its main loop (TMA table loads, 8-bit transposing ldmatrix,
a 160 x 256 tile, a per-GPU placement sweep). Keep rule taken from bpm: a gain counts only above twice its error and
0.05%. One Salad 5090 in llmjob2 (g16-rtx50-5090-5, 480 W limit, driver 581.57), `nv50box.sh` at 10936d15,
`hashrate.js` 120 s legs taking turns, all on the same addon (CUDA 13.3) and the shipped `pearl_rot_sm120w.cubin`
(c580e560) unless named.

**Idea 2, the k order: +0.21% and +0.11% on capped RTX 50 cards; on by default for RTX 50.** Our host already searches each chunk's k order against bank conflicts
(`rot_order_chunk`, squared cost), and that already beats bpm's greedy. Simulated on random index lists (scratch
`sim.js`, passes per ldmatrix matrix, 1.0 = no conflict): definition order 2.60, ours 1.76, bpm's `reorder_pairs`
1.86. More swap sweeps scored on the real pass count (the fullest bank of each matrix, squared cost as tie-break)
reach 1.57 (B's matrices 1.70 to 1.49). That is `PEARL_ROT_ORDER=2` (214f74ef, host only; raced as a setting).

| Legs | TH/s |
|---|---|
| shipped order | 426.65, 426.67, 426.58, 426.55 (mean 426.61; 426.53-427.03 in the legs around) |
| `PEARL_ROT_ORDER=2` | 427.45, 427.63, 427.22, 427.65 (mean 427.49) |

+0.88 TH/s, **+0.21%**, standard error of the difference about 0.02%. verify-hits 400 of 400 (571 hits, 29 salts).
Clock and watts the same (2575-2591 MHz, 480 W, the card at its power cap): fewer bank passes give more work a
clock. The order search serves only the rotation fold (sm_86, sm_89, sm_120); the tall folds and RTX 20's core do
not use it.

**A second 5090 shows no gain** (g16-rtx50-5090-6, PC a78c5263, 575 W limit, driver 621.24, memory 14332 MHz,
commit e36085b6). This card is clock-bound: 2630 MHz at 457 W, under its cap. Shipped order 441.81, 442.34, 442.04,
441.90 (mean 442.02); `PEARL_ROT_ORDER=2` 441.81, 442.14, 442.63, 441.86 (mean 442.11): +0.02%, noise. So the gain
needs a card the power cap holds, as the first PC was. It costs CPU: the core's own count put the search at 580 ms a
salt against 142 ms (27 searches a 120 s leg), about 13% of one core a card more. In the simulation one refine
sweep instead of up to eight keeps most of the gain (1.586 passes against 1.566, from 1.753) at about 40% of the
extra time. On the same PC through bench.sh's meter (not hashrate.js's): SRBMiner 3.8.8 428.1 TH/s, PeakMiner
2.20.9 429.3. Three PCs were passed over first: one logged nothing, one held 877 MHz at 128 W, one had 4.3 GB in use
and 7% load at idle.

**An RTX 3080 Ti loses** (the same search serves the sm_86 rotation fold). Salad had no PCs for an hour (three RTX
50 and two RTX 3080 Ti groups in llmjob2 stayed pending 6-15 minutes each, never allocated), so this ran on Vast:
instance 61095094, Quebec, 300 W limit, driver 580.126.09, i5-10400F, `nv30box.sh` at 7a866b2f, power-capped (299 W,
1629 MHz). verify-hits 167 of 167 with the new order. Shipped order 131.72, 131.76, 131.78, 131.81, 131.79 (mean
131.77, sd of a leg's samples 0.17-0.24); `PEARL_ROT_ORDER=2` 131.46, 131.80, 131.44, 131.50, 131.57 (mean 131.55,
sd 0.33-0.39): **-0.17%**. The wider spread inside its legs points at the longer search on the CPU, not the order
itself, but that is not shown.

**An RTX 5060 Ti gains +0.11%** (g16-rtx50-5060ti-4, Salad, 153 W limit, driver 591.74, at its cap: 153 W, 79-80 C;
`pearl_rot_sm120.cubin` 8584f90b, commit e36085b6). verify-hits 126 of 126. Shipped order 97.71, 97.66, 97.61, 97.57,
97.54, 97.52 (drifting down as the card warmed); `PEARL_ROT_ORDER=2` 97.79, 97.72, 97.68, 97.69, 97.67. Each against
the mean of the legs either side: +0.105, +0.085, +0.09, +0.135, +0.14 TH/s, **+0.11%**, standard error 0.01%. The
search: 848 ms a salt against 250 (a salt is about 17 s here). On the same PC through bench.sh's meter: SRBMiner
3.8.8 97.84 TH/s, PeakMiner 2.20.9 98.39. (An earlier 5060 Ti, g16-rtx50-5060ti-3, 180 W limit and clock-bound at
150 W, gave 91.14 against 91.18 before my own timeout deleted the group.)

**Where this leaves the order:** on RTX 50 cards held by their power cap +0.21% (5090) and +0.11% (5060 Ti), level on
a clock-bound 5090, -0.17% on a capped RTX 3080 Ti. So it is on by default for compute 12.0 only (1ac45a07,
`rot_order_true`); `PEARL_ROT_ORDER=2` or `1` forces it on or off for any card. RTX 30 and 40 keep the old search.
That build checked on the same 5060 Ti: verify-hits 126 of 126 with no setting (the refine on, 841 ms searches), and
97.69, 97.70, 97.67 against 97.64, 97.58, 97.56 with `PEARL_ROT_ORDER=1`: +0.10%.
Its cost is CPU on each card's helper thread: about 13% of one core for a 5090, 5% for a 5060 Ti. In the simulation
one refine sweep instead of up to eight keeps most of the gain at about half that; not raced.

**Idea 1, ptxas -O levels: nothing to win.** The shipped file is a hand loop patched into the compiled kernel, so -O
reaches only the kernel around it. With -O1 and -O0 the compiled kernel uses registers the loop needs, and
`rfold120.py` refuses both. -O2 builds: 426.42 and 426.09 against 426.61, 426.25 and an -O3 rebuild's 426.24 on either
side: level. (pearl2 dev found the same on the RTX 40 tall fold on rig01: -O1 6.5% slower, -O2 and -O3 the same
cubin; an earlier sm_120 probe found -O1 slower too.) bpm's +38% for -O1 is about how ptxas schedules its C loop;
our loop is not ptxas's. The C-fold and release-fold -O legs were skipped once this was clear.

**Idea 3, the tables by TMA bulk copy:** each tile starts with every thread copying 78 KB of tables in 19 rounds of
a load and a store (`pearl_rot.cu`); bpm starts its tiles with two `cp.async.bulk` copies. `ROT_TMA=1` does the
same, and `ROT_TABSKIP=1` (timing only) copies a block's first tile only, to price the copy. On this box
`rfold120.py` refused both: their prologues put tile values in UR16-34, where the hand loop keeps its bases, and
then kept the barrier's phase live across the loop's call. With the loop's uniform registers at UR40-54 (`sr_ua=40
sr_ub=54`) and the phase worked out from the tile index (e36085b6), the TMA build goes through; the timing-only
one still does not (it keeps three registers the loop writes), so the copy's ceiling is not priced.
On the second 5090 (above): verify-hits 400 of 400 (589 hits) for the TMA build. Against the shipped kernel with the
same moved registers (`hU`, verify-hits 400 of 400): **TMA 440.90, 440.65, 440.73 against 441.64, 441.72, 442.04,
-0.24%**. One thread waiting on a bulk copy is slower here than 256 threads copying. Dropped.
The rest of bpm's main loop is not new to us or is worse by our measure. Its 8-bit transposing ldmatrix has shipped
since goal-nv50d (`b8=1`, +1.7%). Its 160 x 256 tile has every warp rebuild A and B into a stage and load both back:
about 0.5 shared-memory instructions an IMMA against our 96 x 512 block's 0.32, and at the power cap that count sets
the clock (goal-nv50n). Its per-GPU placement sweep is what our hand loop's keys (`ist`, `iy`, `dpb`, `dpg`, `cq`,
`trl`, ...) have had on 5090 and 5080 cards since 12 November.

Other numbers from the box: the release fold (k 2048) 370.5-370.9 TH/s at 2650 MHz against the rotation fold's
426-427 at 2580 MHz.

### 19 Nov 2026: the RTX 5090 against SRBMiner 3.8.9, kept B tables, and a key sweep; nothing ships (branch goal/rtx50c)

One Salad 5090 group in llmjob2 (g18-rtx50-5090-1; three PCs, as stop/start moved it), `nv50box.sh`, `hashrate.js`
legs taking turns on the shipped `pearl_rot_sm120w.cubin` (c580e560) built on the box with main's host (v0.7.5:
the k order refine on). Competitors through bench.sh on the stand-in pool (2^206), mean of samples from 60 s.

**Baseline (PC 4241f62d, 600 W limit, driver 610.88, SW Power Cap throughout):**

| Miner | TH/s | MHz | W |
|---|---|---|---|
| Ours (rshipw, 4 legs of 120 s; twin hB 444.41, 444.75, 443.89, 443.96) | 444.4 (445.02, 444.41, 444.35, 444.37) | 2603 | 502 |
| SRBMiner 3.8.9 | 461.5 | 2688 | 534 |
| PeakMiner 2.20.9 | 448.3 | 2615 | 500 |

Ours is **96.3%** of SRBMiner 3.8.9. TH/s per GHz: ours 170.7, SRBMiner 171.7, PeakMiner 171.4. So the work per clock
is level and the gap is clock: SRBMiner runs 3.3% more clock while drawing 32 W more, under the same SW Power Cap
and well below the 600 W limit. This is the inner-budget case of 15 November, wider than with 3.8.8 (97.8% then).
Its shares still show the 128 x 384 block (16 A rows 8 apart by 12 B columns in pairs 64 apart): 3.8.9 did not
change its tile.

- **PEARL_ROT_TILES=128** (four times the tiles a batch): 444.02, 443.84, 444.43 against 444.4: level.
- **ROT_BKEEP** (pearl_rot.cu, new, off by default): a block copies B's 64 KB of tables only when its next tile is
  in another column block (on the 131072-row profile 7 tiles in 8 keep them). The first two forms were refused by
  rfold120.py (ptxas kept the tile count and the copy address across the hand loop's call); the committed form
  keeps v in the frame and a shared word at ROT_MB. verify-hits 400 of 400 (571 hits, 29 salts). PC be10d586, 480 W
  limit, at the board limit: hU (twin, same moved uniform registers) 431.05, 430.98, 430.96; hK 430.97, 430.91:
  **-0.01%**, level. The per-tile table copy costs nothing measurable.
- **The 5090 file's loop keys, one at a time** (third PC, 600 W, 610.88, inner-budget bound; 100 s legs, each
  against the shipped file's legs either side; the card drifted from 442 to 438.6 over the run):

  | Key | TH/s | Shipped either side | Change |
  |---|---|---|---|
  | lgg=1 | 438.25 | 442.37 (cold), 438.72 | level to -0.1% |
  | lgg=3 | 438.77 | 438.72, 439.03 | level |
  | ist=6 | 436.95 (2620 MHz, 518 W) | 439.03, 439.84 | -0.6% |
  | ist=8 | 439.81 | 439.84, 439.75 | level |
  | dpb=6 | 439.09 | 439.75, 438.99 | -0.1% |
  | dpb=8 | 437.95 (2604 MHz, 512 W) | 438.99, 438.58 | -0.2% |
  | dpg=2 | 438.53 | 438.58, 438.58 | level |
  | dpg=4 | 438.38 | 438.58 | level |

  The shipped keys stay. The keys that raise the clock (ist=6, dpb=8) draw more watts and do less work a clock.
- **The k order search is near its limit.** A simulated annealing search (scratch sa.js, 100 random chunks)
  reaches 1.576 passes a matrix with 100k moves a chunk, the shipped refine's 1.57; the bound from the fullest
  residue count is 1.38 and is not reachable that way.

- **Memory-controller use (nvidia-smi utilization.memory), the RTX 20 lead:** ours 0% in every rotation-fold leg on
  both 600 W PCs (hashrate.js memUtil 0, 1 on the first cold leg); SRBMiner 3.8.9 0% and PeakMiner 0% in every bench
  sample. On Turing ours ran 11-16% against SRBMiner's ~0%; on the 5090 the fold already runs from the L2, so there
  is no DRAM power to win back and a tile walk for L2 reuse has nothing to save here.

**A from global memory, priced before a hand loop (the lead's design; `powcap.cu` wfold modes, new).** A
fold-shaped kernel: one block an SM, eight warps, 48 IMMA a warp a k-step with each A fragment held for 8, the same
four B LDS.128 a k-step in every mode, operand bytes 0-127 in every mode. Only A's source differs: `fstn` the stage
(warps 0-5 load a table entry and store it, every warp loads six tiles; no barrier), `fgl2` six LDG.128 a k-step of
an A' stream every SM reads at once (in the L2; the SM's eight warps share each line through the L1), `fgdr` the
same but each SM its own stream (from DRAM), `fnoa` no A loads. 40 s a mode after 10 s of warm-up.

| 5090 (limit) | fstn | fgl2 | fgdr | fnoa |
|---|---|---|---|---|
| 600 W (610.88), board limit | 419.89 (2747 MHz), 417.18 (2729), 417.17 (2730) | 408.47 (2696), 408.36 (2695) | | 408.95 (2790), 409.14 (2791) |
| 575 W (621.24), board limit | | 405.59 (2685), 403.30 (2671), 404.04 (2675) | 377.21 (2517), memory controller 44.7% | 406.74 (2777), 406.73 (2776) |

(Launches a second; every row at its board limit with SW Power Cap on. The 575 W PC's real fold drew 535 W at 2559
MHz: 439.41.) The memory controller read 0% in every mode but `fgdr`.

- **The stream loses to the stage by 2.2-2.8%** in work and 1.3-1.9% in clock on the same PC. L2 bytes cost more
  of the cap than the shared-memory steps they replace, as goal-nv-k28 found on RTX 40 (8 KB an SM a k-step from
  the L2 cost 44 W there).
- **From DRAM it is 7% worse again** (2517 MHz, memory controller 45%). An A' stream would need the column-fastest
  tile walk to stay in the L2.
- `fnoa`'s rate per clock is lower because its operands change by ALU steps (an IMAD and a LOP3 a word), so it
  bounds the clock (2790 MHz), not the rate.
- What `fstn` leaves out is the stage's barriers. The 13 November probe put the whole ordering cost of the 96 x 512
  block at +1.1-1.6% (no ordering, timing only). So A from global memory would gain at most that and lose
  2.2-2.8%: about -1%. The hand loop was not built.
- Two earlier forms of these modes are not counted: `fst` (one barrier a k-step, which starves this kernel's pipe:
  357-361 launches a second at 2803-2841 MHz) and a first `fstn` whose stage operands were 63 plus six random bits,
  which toggle less (3019 MHz at 232 W against `fnoa`'s 2913 MHz at 342 W on a 460 W PC).

**The configs each miner declares (the decode the RDNA 4 agent used, from run 1's tapped shares on the 600 W PC;
sharetap.py on the stand-in pool).** Ours is the profile in shared/miner/pearlhash.js and the fold's region.

| Miner | m | n | k | Region (a thread's sums) | Shares decoded |
|---|---|---|---|---|---|
| Ours | 131072 | 262144 | 65536 | 12 rows 8 apart (span 96) x 16 columns, pairs 8 apart (span 64) | (from the code) |
| SRBMiner 3.8.9 | 17922 | 62976 | 65536 | 16 rows 8 apart (span 128) x 12 columns, pairs 64 apart (span 384) | 83 of 83 ok |
| PeakMiner 2.20.9 | 5121 | 261120 | 32768 | the same 16 x 12 as SRBMiner | 69 of 69 ok |

- Every miner keeps 192 sums a thread, so the same accumulator registers and the same readout (96 three-input XOR
  a chunk). SRBMiner's region is our 12 x 16 turned on its side; it implies the 128 x 384 block (8 m16 by 6 n8 tiles
  a warp, A shared by eight warps) that goal-nv50x built as a real hand loop on 15 November: -0.5% on two PCs.
  PeakMiner declares the same region and runs level with us, so the region is not where SRBMiner's clock comes from.
- m and k change no cost a MAC that matters here. A smaller m means smaller A tables, but ours already fit in the L2
  (memory controller 0%), and the table copy a tile is free (ROT_BKEEP, level). A smaller k means more tiles, so more
  copies and hashes a MAC; RDNA 4 priced k 16384 at -0.65%. Nothing here is worth a powcap price or a launch setting.

**The NVIDIA profiler on a Vast RTX 5090 VM (machine 51518, 575 W cap, 19 Nov; run by the lead session, recorded
here because this is its only record).** Ours v0.7.6 against SRBMiner 3.8.9 and PeakMiner 2.20.9, the same PC.

| | Ours | SRBMiner 3.8.9 | PeakMiner 2.20.9 |
|---|---|---|---|
| IMMA a cycle | 41.7 | 41.9 | 41.9 |
| Tensor pipe active | 98.4% | 98.6% | |
| Clock at a 575 W average | 2469 MHz | 2515 MHz | 2484 MHz |
| Instant power, 100 ms samples | 486-647 W, SD 7-13 W; about 5% above 590 W | 534-587 W, SD 2-4 W; none above 590 W | 572-593 W |
| DRAM bytes / L2 sectors a IMMA | 5-6x / 2.8x SRBMiner's | 1 / 1 | |
| Shared-memory bank conflicts a IMMA | 0.41 (25% of shared-load wavefronts are replays) | 0.245 | 0.27 |

Instructions a IMMA were the same for all three. Our kernel as ncu saw it: pearl_rot_fold, 255 registers, 90.5 KB of
dynamic shared memory, 256 threads, a grid of 170, 51.5 ms a launch at 2.0 GHz. The lead's reading: the same work a
clock, so the gap is clock; the cap reacting to our power peaks may cost about 46 MHz (1.8%), the peaks perhaps from
the per-salt and per-job work between fold launches.

**The profiler's two leads, checked on Salad 5090s (19 Nov).**

*Lead 1, power spikes.* nv50box.sh `HRPW=1` (new) samples `power.draw.instant` every 100 ms through each
hashrate.js leg. The core's stderr is timestamped, and `PEARL_ROT_LOG=2` (new) logs when the next salt's tables are
queued on the side stream ("prep") and when a prepared salt takes over ("switch"). `pwsum.py` (new) sums the trace,
leaving out the first 20 s.

| PC (limit, driver) | Leg | TH/s | Mean W | SD W | Max W | Top 2% within 300 ms of a salt event | MHz in the second after an event / the rest |
|---|---|---|---|---|---|---|---|
| 575 W (617.42), board limit | shipped | 455.16 | 575.0 | 3.3 | 608 | 23 of 23 (chance 24.5%) | |
| same | PEARL_ROT_TILES=128 | 452.15 | 575.0 | 5.7 | 616 | 18 of 23 | |
| 575 W (610.88), held at 531-541 W | shipped (cold) | 441.65 | 540.7 | 7.4 | 569 | 13 of 23 (prep 6, switch 9) | 2566 / 2563 |
| same | PEARL_ROT_NOCOMMIT=1 | 434.87 | 531.2 | 11.9 | 638 | 23 of 23 (prep 22) | 2537 / 2529 |
| same | shipped | 434.41 | 531.8 | 10.6 | 634 | 23 of 23 (prep 18) | 2533 / 2527 |
| same | PEARL_ROT_NOCOMMIT=1 | 434.15 | 531.1 | 9.6 | 631 | 23 of 23 (prep 22) | 2533 / 2527 |

- The swings the profiler saw are real and come from the salt events: every top-2% sample of a warm leg lies within
  300 ms of one, mostly of the side stream's prep (the next salt's dense factor, tables and offsets, about every
  4.5 s). Peaks reach 630-640 W where the rest of the leg tops out at 540-555 W.
- **They cost no clock that can be measured.** In the second after each event the clock is the same as elsewhere,
  or 4-6 MHz higher. The SW Power Cap's average holds and the peaks are too short to move it. So the 1.8% estimate is
  not borne out on these PCs. On the PC held at its 575 W board limit the spread is only 3.3 W.
- `PEARL_ROT_NOCOMMIT=1` (new, timing only: no restamp commit, which runs on the legacy default stream and drains
  every queued batch at a salt switch) is level: 434.87 and 434.15 against 434.41. Bigger batches (128 tiles an SM)
  widen the spread and are level to lower, as before.
- Nothing to change in the prep. Moving it to its own SMs would cost 1.2% of the fold (2 of 170 SMs) for nothing
  measured.

*Lead 2, bank conflicts (0.41 a IMMA against SRBMiner's 0.245).* Our conflicts are the gathers'. Each ldmatrix row is
a table entry picked by a random rank index, and two rows collide when their indices agree mod 8. The k order
search already holds that to 1.57 passes a matrix, and annealing reaches no lower (above). A warp issues 8 B
gathers and 1.5 A gathers a k-step: about 0.37 extra wavefronts a IMMA, close to the 0.41 measured. A padded or
swizzled table only renames the slots, so random indices still collide at the same rate. Only a second copy of the
tables would give the order a choice, and B's 64 KB has no room. SRBMiner's 128 x 384 block issues 6 B gathers a
warp a k-step, not 8, which matches its 0.245 almost exactly (0.75 x ours). That block was built as a hand loop on
15 November and lost 0.5%: the A side's extra stage traffic costs more than the conflicts it saves.

**Where this leaves the 5090:** level with PeakMiner, 3.7% behind SRBMiner 3.8.9, all of it clock at the same
SW Power Cap. Nothing tried here moves that budget: kept tables, batch size, loop keys, and feeding A from the L2
(priced at about -1%). The memory controller sits at 0% for every miner, so DRAM is not where SRBMiner's clock
comes from on this card.

## A100-class (sm_80)

Clore and Vast, 16 November.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | OneZeroMiner | % of best |
|---|---|---|---|---|---|---|---|
| A100 PCIe 40 GB | Vast (250 W) | 57527860 | 211.9 | 225.1 | 206.7 | 217.6 | 94.1% |
| A100 PCIe 80 GB | Clore (300 W) | 124543 | 223.1 | 239.7 | 232.0 | 237.9 | 93.1% |
| A100 SXM4 40 GB | Vast (400 W) | 60140204 | 250.0 | 277.4 | 271.7 | 274.3 | 90.1% |
| A100 SXM4 40 GB | Vast (400 W) | 57328781 | 226.2 | 246.6 | 243.5 | 240.0 | 91.7% |
| CMP 170HX | Vast (250 W) | 59377421 | 173.0 | 191.4 | 186.5 | 183.8 | 90.4% |
| CMP 170HX | Vast (250 W) | 59377417 | 172.0 | 188.1 | 192.7 | 192.4 | 89.3% |

The A100 PCIe 80 GB row is a rerun on Clore, 16 November: v0.7.3 against PeakMiner 2.20.9, SRBMiner 3.8.8
and OneZeroMiner 1.8.6, 5 minutes each on Kryptex, each miner's mean of its displayed rate.
sm_80 has no rotation-fold kernel, so ours mines on the
normal kernel there.

The two CMP 170HX rows are reruns on Vast, 16 November, the same way, on two PCs in Finland (Vast offer IDs, as
the machine IDs weren't recorded). They replace the 10 October Georgia row (163.4 against 165.5 and 165.3), raced
on older builds.

The A100 PCIe 40 GB row is a rerun on Vast (Japan, offer 57527860), 16 November, the same way; it replaces
the 10 October row (211.1 against 213.7 and 215.3).

The two A100 SXM4 40 GB rows are reruns on Vast, 16 November, the same way (offer IDs); they replace the 10
October rows (245.5 and 246.6 against bests of 253.4 and 253.8). The second PC drew only about 270 W of its 400 W
limit for every miner, so its figures are lower across the board.

The A30's host lowered the power limit during the competitors' runs; at equal power ours is about 107%.

## Hopper (sm_90a)

Vast, 10 October.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| H100 SXM 80 GB | Vast (575 W) | 148671 | 682.8 | 697.7 | 688.1 | 98% |
| H100 NVL 94 GB | Vast (400 W) | 29785 | 536.5 | 548.6 | 539.1 | 98% |
| H100 PCIe 80 GB | Vast (350 W) | 147981 | 484.9 | 448.3 | 488.0 | 99% |
| H200 141 GB | Vast | — | — | — | — | not run: over budget |
| H200 NVL 141 GB | Vast | — | — | — | — | not run: over budget |

Last H200 figures (8–9 October): 99.9% on the H200 141 GB and 101% on the H200 NVL.

## Ada workstation and data-center cards (sm_89)

Vast, 10 October; rows with a OneZeroMiner figure rerun on Vast, 16 November.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | OneZeroMiner | % of best |
|---|---|---|---|---|---|---|---|
| L40S | Vast (350 W) | 58081963 | 314.4 | 323.5 | 323.3 | 274.6 | 97.2% |
| L40S | Vast (350 W) | 150627 | 288.1 | 252.8 | 290.4 | not run | 99% |
| RTX 6000 Ada | Vast (300 W) | 60786704 | 262.9 | 238.4 | 230.3 | 224.9 | 110.3% |
| RTX 6000 Ada | Vast (300 W) | 152478 | 216.6 | 186.6 | 225.8 | not run | 96% |
| RTX 5000 Ada | Vast (220 W) | 53457490 | 220.8 | 225.4 | 225.4 | 221.8 | 98.0% |
| L40 | Vast (300 W) | 53285135 | 180.3 | 180.7 | 180.7 | 27.1 | 99.8% |
| L40 | Vast (250 W) | 56199 | 153.2 | 132.8 | 156.6 | not run | 98% |
| RTX 4500 Ada | Vast (210 W) | 148971 | 139.7 | 134.2 | 139.6 | not run | 100% |
| RTX 4000 Ada | Vast (130 W) | 59281335 | 102.9 | 104.6 | 104.5 | 104.6 | 98.4% |
| RTX 4000 Ada | Vast (130 W) | 152701 | 96.1 | 92.4 | 97.4 | not run | 99% |
| L4 | Vast (40 W) | 59651494 | 48.4 | 50.2 | 50.0 | 46.6 | 96.4% |
| L4 | Vast (72 W) | 152541 | 80.5 | 75.0 | 81.6 | not run | 99% |
| RTX 2000 Ada | Vast (70 W) | 58892707 | 52.4 | 53.2 | 53.1 | 52.9 | 98.5% |

## RTX PRO Blackwell (sm_120)

Vast, 10 October, on the CUDA 13 core; rows with a OneZeroMiner figure rerun on Vast, 16 November.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | OneZeroMiner | % of best |
|---|---|---|---|---|---|---|---|
| RTX PRO 6000 Workstation | Vast (600 W) | 60116258 | 483.2 | 486.0 | 485.4 | 485.9 | 99.4% |
| RTX PRO 6000 Server | Vast (600 W) | 139338 | 435.1 | 432.4 | 435.9 | not run | 99.8% |
| RTX PRO 6000 Max-Q | Vast (300 W) | 60105109 | 333.8 | 337.2 | 336.9 | 320.3 | 99.0% |
| RTX PRO 5000 48 GB | Vast (300 W) | 49233740 | 260.3 | 262.4 | 263.3 | 243.1 | 98.9% |
| RTX PRO 4500 | Vast (200 W) | 57478866 | 193.4 | 194.9 | 195.4 | 194.5 | 99.0% |
| RTX PRO 4000 | Vast (145 W) | 58966937 | 134.0 | 135.5 | 136.0 | 135.2 | 98.5% |
| RTX 6000D | Vast (600 W) | 58409092 | 144.9 | 144.3 | 144.1 | 141.4 | 100.4% |

## Ampere workstation and data-center cards (sm_86)

Vast, 10 October. Rows with a OneZeroMiner figure rerun on Vast, 17 November, on v0.7.3 against PeakMiner 2.20.9, SRBMiner 3.8.8 and OneZeroMiner 1.8.6.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | OneZeroMiner | % of best |
|---|---|---|---|---|---|---|---|
| RTX A6000 | Vast (300 W) | 60043321 | 132.3 | 134.0 | 133.7 | 134.0 | 98.7% |
| RTX A6000 | Vast (300 W) | 37274 | 126.1 | 125.7 | 124.9 | not run | 100% |
| A40 | Vast (300 W) | 60270897 | 137.3 | 139.2 | 138.6 | 139.0 | 98.6% |
| A40 | Vast (300 W) | 151322 | 123.2 | 122.5 | 127.3 | not run | 97% |
| RTX A5000 | Vast (230 W) | 147999 | 93.7 | 92.8 | 94.0 | not run | 99.7% |
| RTX A5000 | Vast (200 W) | 60249679 | 95.8 | 97.2 | 97.4 | 96.8 | 98.4% |
| RTX A4000 | Vast (140 W) | 15970378 | 71.9 | 73.2 | 73.4 | 73.3 | 98.0% |
| RTX A4000 | Vast (140 W) | 29102 | 58.2 | 56.8 | 62.3 | not run | 93% |
| RTX A2000 12 GB | Vast (70 W) | 28227 | 28.7 | 29.0 | 32.3 | not run | 89% |
| RTX A2000 6 GB | Vast (70 W) | 59654526 | 36.4 | 37.1 | 37.1 | 37.1 | 98.1% |

On 10 October SRBMiner held a higher clock than ours at the same power on the A2000s and A4000s. On the 17 November A4000 all four miners ran at 1470–1500 MHz.

### CMP 90HX on Clore, 4 November (v0.6.11, our miner only)

| Card | Power limit | TH/s after 5 min | SM clock |
|---|---|---|---|
| CMP 90HX (GPU 0) | 200 W of 250 W | 72.7 | 1485 MHz |
| CMP 90HX (GPU 1) | 200 W of 250 W | 72.7 | 1500 MHz |
| CMP 90HX (GPU 2) | 200 W of 250 W | 71.0 | 1380 MHz |

At 72 TH/s it earns about $0.056 an hour, below Clore's cheapest rent for the card.

## B200 (sm_100)

Vast, 10 October.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| B200 180 GB | Vast (1000 W) | 96942 | 457.9 | 1380.6 | 1394.7 | 33% |

We mine it on `mma.sync`, which tops out near 568 TH/s here; the competitors use tcgen05.

## Tesla T4 (sm_75)

Vast, 17 November, on v0.7.3 against PeakMiner 2.20.9, SRBMiner 3.8.8 and OneZeroMiner 1.8.6.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | OneZeroMiner | % of best |
|---|---|---|---|---|---|---|---|
| Tesla T4 16 GB | Vast (70 W) | 56283824 | 34.3 | 34.8 | 33.4 | 36.1 | 95.0% |

## Tesla V100 (Volta, sm_70)

Vast, 10 November (PR #290 item 9). PeakMiner 2.20.7, SRBMiner 3.8.2 and OneZeroMiner 1.8.5 ran as bench.sh runs
them, on Kryptex. No release has V100 code yet, so "Ours" is this branch's core built for sm_70 alone (the dp4a fold
GTX 10 and GTX 16 cards mine on), run by `earn/native/probes/turing/pool.sh`. Not a release row.

| Card | Host | Machine ID | Ours (dp4a, this branch) | PeakMiner | SRBMiner | OneZeroMiner | % of best |
|---|---|---|---|---|---|---|---|
| Tesla V100 SXM2 32 GB | Vast (300 W) | 155858 | 29.2 | exits | 40.0 | 50.6 | 58% |

On 17 November v0.7.3 skipped a Tesla V100 SXM2 32 GB on Vast ("no CUDA device found": the release has no sm_70 code), so this row still stands for the branch build only. On that box SRBMiner and OneZeroMiner also failed to start.

- PeakMiner retires the card at start ("all GPUs retired", exit 70) twice. SRBMiner drew 289–295 W, OneZeroMiner
  292–297 W at 1410 MHz; ours 298 W at 1507 MHz.
- Ours passed verify-hits 248 of 248 and ran 29.15 TH/s for 5 minutes, about 242 MAC an SM a clock: 95% of the 256 a
  dp4a allows on Volta's 64 integer lanes an SM. So the dp4a fold is at its ceiling here. A further run of about
  12 minutes on Kryptex had 2 shares accepted and 0 rejected (none in the 5-minute run). The box cost $0.06.
- OneZeroMiner's 50.6 TH/s is about 450 MAC an SM a clock, 88% of the fp16 tensor cores' 512, so it most likely mines
  on HMMA with fp16 operands: |A'|, |B'| ≤ 63 are exact in fp16, and its fp32 sums are exact integers under 2^24
  (2048 × 63² is 8.1M). A Volta fold like that would be about +70% over dp4a.
- Why it matters: Clore listed about 1,900 V100 SXM2 cards on 4 November, 1,240 of them idle, at a median $2.06 a
  card-day. At 29 TH/s a V100 earns about $0.02 an hour, so like the CMP 90HX it doesn't pay to rent; it's for owners.
  Until this branch our core had no sm_70 code at all, so a V100 couldn't mine on it.

### The fp16 tensor-core fold, 10 November (PR #290 item 20)

Same Vast machine (155858, V100 SXM2 32 GB, 300 W). `pearl_tile_fold_hmma` is the dp4a fold's 128x128 tile on the
fp16 tensor cores: A' and B' converted to fp16 as they are staged, `mma.sync` m8n8k4 into fp32 accumulators. Each
build was raced against the dp4a fold (`-DPEARL_VOLTA_HMMA=0`) in one process (`compilers.sh` race stage, 4 s turns),
then checked with verify-hits.

| Build | Vast | vs dp4a | SM clock | Power | verify-hits |
|---|---|---|---|---|---|
| dp4a fold (item 9) | Vast | — | 1530 MHz | 291 W | 248/248 (item 9) |
| wmma 16x16x16 | Vast | +19.2% | 1523 MHz | 295 W | 314/314 |
| m8n8k4, k outermost | Vast | +36.0% | 1500 MHz | 299 W | 359/359 |
| + no bank conflicts on A' | Vast | +39.6% | 1485 MHz | 299 W | 232/232 |
| + accumulators start at 2^23 (shipped) | Vast | +47.9% | 1477 MHz | 298 W | 400 verified, 0 failed |

- The shipped build's race ran 32 rounds, 32 ahead, ci95 0.2%; a copy of the dp4a build ran +0.000%. On Kryptex (`probes/turing/pool.sh`, prl-us), 20 minutes measured after a minute's warm-up: 43.85 TH/s, 3 shares accepted and 0 rejected, 1477 MHz, 299 W; its verify-hits passed 398 of 398. The box cost about $0.20 for two hours.
- 43.8 TH/s is 87% of OneZeroMiner's 50.6 here (was 58%), and ahead of SRBMiner's 40.0.
- Exactness: fp16 holds every int8, and fp32 sums of integers are exact below 2^24. (The note above that |A'| and
  |B'| stay under 64 was wrong: at fill 48 they reach 111. The sums are what matter.) The accumulators start at 2^23,
  so the readout takes the float's bits as the integer, which needs each one in [0, 2^23) at every chunk's end. At
  fill 48 the nearest end is 14 sd away (chunk 0); `pearl_config.h` has the numbers. `probes/hmma-check.cc` runs the
  fold's layout and readout on the CPU against a plain fold.
- What each step did. wmma's 16x16x16 runs its four k-steps of an 8x8 tile back to back, each waiting on the last;
  m8n8k4 with k outermost puts 16 other mma between them. A' rows 32 apart sat on the same banks, so every 16-byte
  load of A' was a 2-way conflict; swapping rows 0-3 and 4-7 of every other quad pair's tile fixes it (+3.0%). The
  readout first added a magic constant and subtracted it again (2 ops an accumulator); starting the accumulators at
  2^23 leaves only XORs (+6.0%), and 45 fewer registers (210).
- Ablations on the 41.45 build (timing only, wrong results): mma and shared loads alone 58.4 TH/s, 487 MAC an SM a
  clock, 95% of the tensor cores' 512. Without the int8-to-fp16 staging +12.0%, without the old readout +10.5%,
  without the global loads +8.7% (and the clock rises to 1530), without the barriers -3.2%.
- Tried and dropped (each against the shipped build, same race): converting and storing half the next stage between
  the two k-steps -7%; loading two stages ahead +0.3%; fragment loads of 8 or 4 halfs at a time +0.2%; tile walk
  band 4 -0.1%; 1024 column offsets a batch +0.4%. All inside the noise or worse.
- What's left: the card holds its 300 W cap at 1477 MHz, 371 MAC an SM a clock (72% of peak) against OneZeroMiner's
  ~449 at 1410 MHz. The staging (conversion and shared stores) and the global loads are the remaining ~20%. A
  128x256 tile would cut both by a quarter but fits only one block an SM; fp16 operands in global memory would drop
  the conversion but double the bytes loaded. Neither is tried.

## AMD Radeon (RDNA 3 and 4, on Salad)

### Salad, 19 November: RDNA 3's B' straight from memory into registers, +2.0 to +2.6% (`PEARL_AMD_GB11`, branch `goal/rdna3d`)

RDNA 4's `PEARL_AMD_GB` (goal-rdna4u) on the wmma11 fold, aimed at the LDS energy the note below found: each wave
loads its next slab's B' fragments (both k-steps, 8 buffer loads of 16 bytes a lane, lanes l and l + 16 the same row)
straight into VGPRs after the slab's last WMMA, and only A' goes through LDS. 212 VGPRs against 226, 6 waves, no
spill; the fold check bit-exact on gfx1100 and gfx1101. Boxes `g18-rdna3-xtx2` (RX 7900 XTX, PC `de6b6d38`; Salad
gave `a8cae450`, the RTX 5080, first) and `g18-rdna3-x78a` (RX 7800 XT, PC `59b5beab`), `hashrate.js` 50 s after 15 s,
builds in turn:

| Card, PC | Release | Twin | GB11 | GB11 against the release |
|---|---|---|---|---|
| RX 7900 XTX, `de6b6d38`, 3 rounds | 56.61, 57.31, 57.32 | 57.19, 57.40, 57.45 | 58.75, 59.02, 58.83 | +2.6 to +3.8% (+2.4 to +2.7% against the twin) |
| RX 7800 XT, `59b5beab`, 4 rounds | 32.84, 32.76, 32.74, 32.74 | 32.79, 32.76, 32.74, 32.75 | 33.51, 33.49, 33.48, 33.49 | +2.0 to +2.3% |

- Stamps of the fold alone on the XTX: 6,391 cycles a slab against 6,636 (-3.7%), clock 2,624 against 2,630 MHz.
- verify-hits with GB11: 400 of 400 on the XTX (1,957 hits, 248 salts) and the 7800 XT (1,117 hits, 143 salts).
- **Against SRBMiner 3.8.9 on the XTX** (`de6b6d38`, Kryptex, turns of 300 s with the clock probe, GB11 first):
  ours 58.21 and 57.89 (mean after the first minute; 2,580 and 2,568 MHz), SRBMiner 58.48 and 58.48 (its last
  figure; means 57.77 and 58.11; 2,578 and 2,596 MHz). 99.3% by the table's rule, 100.2% by both miners' means. The
  release on this PC read 97.1% of SRBMiner 3.8.8 (goal-rdna3b).
- On by default where RDNA 3's set is (gfx1100 and gfx1101, Linux core); `-DPEARL_AMD_GB11=0` builds the old path.
  The Windows core has no RDNA 3 buffer loads, so it keeps B' through LDS.
- **Cost:** about $0.21 for the two boxes (02:53-04:03 UTC).

### Salad, 18-19 November: RDNA 3 against SRBMiner 3.8.9, where the XTX's power goes, and twelve tries that didn't pay (branch `goal/rdna3d`)

One box in `llmjob2`, `g18-rdna3-xtx1` (RX 7900 XTX, high priority, 4 vCPU, 8 GB), PC `6a402b7b` throughout (Salad
first gave it `a8cae450`, an RTX 5080 listed as an XTX). Ours is main at 94e1e36c (v0.7.5 + #305). Races are
`hashrate.js` 50 s after 15 s, builds taking turns, with a twin of the release.

**When we fell behind.** On 11 November v0.6.16 was 101-102% of SRBMiner 3.8.6 on the XTX and 102-107% on the 7900 XT.
SRBMiner 3.8.8 (13 November, "improves pearlhash on RDNA 3 and RDNA 4") took those rows to 94-98% on 14 November;
our rates didn't fall. 3.8.7 and 3.8.9 name NVIDIA only. So the drop is SRBMiner 3.8.8's.

**Baseline against 3.8.9** (same PC, ours first):

| | Ours | SRBMiner 3.8.9 | Ours / SRB |
|---|---|---|---|
| Kryptex, 2 x 300 s, shown | 56.52, 56.25 (mean after 1 min) | 59.92, 59.98 (its last figure; means 59.73, 59.98) | 94.1% |
| Stand-in pool 2^213, 2 x 600 s, shown | 56.53, 56.44 | 59.91, 58.90 (means 58.90, 58.99) | 95.1% (95.8% by means) |
| Stand-in, credited shares after the first minute | 3,444 + 3,457 = 6,901 in 1,096 s | 3,554 + 3,422 = 6,976 in 1,098 s | 98.9% (+-1.2%) |
| Clock (probe beside the miner, 2 turns of 240 s) | 2,566 and 2,562 MHz, 899-909 mV | 2,653 and 2,653 MHz, 930-968 mV | SRB +3.4% |

SRBMiner shows about 4% more than ours but is credited 1% more; less its 2% fee, it does about 3% more work, all of
it from clock: per clock the two are level. It holds 3.4% more clock at a higher voltage, so it spends less energy a
MAC. The founder's displayed-rate gap is that 3% plus about 1.4% of display.

**Profiling.** rocprofv3 doesn't run on Salad (WSL2: no `/dev/kfd`, no KFD topology), and `amd-smi` gives the GFX
voltage only. So the profile is the fold's clock stamps, `pearl_amd_peak`, timing-only ablations and two new probes.
- Fold alone, stamps (`PEARL_AMD_STAMP`, 600 launches): release 2,692-2,700 MHz, 6,638 cycles a slab, 30.5% of the
  loop at the barrier. No LDS fragment reads (ablation 8): same clock, 6,434 cycles (-3.1%). No staging (32): +1.2%
  clock, same cycles. Operands in L2 (16): +3.1% clock; A' alone in L2 (new bit 256): +0%; B' alone (new bit 512):
  +0.6%.
- But in `hashrate.js` (50 s, the card warm) operands in L2 is only +0.7% (57.09 against 56.70, twin 56.59), A' alone
  +0.6%, B' alone 0. Off-chip traffic is worth under 1% in the miner.
- `pearl_amd_peak` (cool card, 4 s runs): register-fed WMMAs 64.5-65.6 TMAC/s at 2,815-2,838 MHz; the fold's LDS reads
  61.1-62.8 at 2,706-2,775 (-2 to -4% clock); fold-shaped reads with no staging and the readout 58.8.
- The card is power-bound: a build that cuts cycles a slab gives the time back as clock (below, `IGLP=2`).

**What gfx11's WMMA reads** (`test/pearl_amd_lay11.cpp`, new): A's odd lanes of 0-15 and even lanes of 16-31 are never
read (row 2g + h of C takes A row r from lane r + 16h; zero or 0x55 there leaves all 256 sums right). Every B lane is
read. So half of A's lanes are free.

**Tried, each against the release and a twin on `6a402b7b`** (TH/s, `hashrate.js`; all bit-exact unless timing-only):

| Try | Result |
|---|---|
| Cache policy on the slab loads (`AUXA`/`AUXB`, box patch): A' glc / slc / dlc, B' slc / dlc | 56.41 / 56.39 / 56.61, 55.58 / 56.41 against 56.70: none ahead, B' slc -2% (B' reuse in L2 matters) |
| `PEARL_AMD_C4=1` and `=2` on RDNA 3 (chunk of 4 slabs known at compile time) | 3 rounds: 56.78, 56.82, 56.76 and 56.78, 56.63, 56.52 against base 56.70, 57.05, 56.77 and twin 56.59, 56.60, 56.77: level |
| Fill 63 / fill 63 with A' and B' unsigned (`W11_UNS`, timing only) | 56.49 / 56.53: level |
| LDS reads placed among the WMMAs (`W11_IGLP`, `sched_group_barrier`): =1 one read per 2 WMMAs, =2 reads in two groups | =1 214 VGPRs, 6,695 cycles a slab, 56.50 / 56.51; =2 234 VGPRs, 6,551 cycles (-1.3%) but -0.4% clock, 56.65 / 56.75 against base 56.80 / 56.44, twin 56.76 / 56.60: level |
| Wave64 fold shape (`test/pearl_amd_peak11w64.cpp`, new: 64 x 128 a wave in 128 accumulator VGPRs, 25% fewer read instructions a WMMA) | LDS-fed 60.2-61.3 TMAC/s against wave32's 61.2-62.8: loses (each wave64 read delivers 1,024 bytes) |
| A's unread lanes: zero by an exec-masked read / zero in registers / each pair reading one row | 60.0-60.4 against 61.3-62.3 / level with registers / 61.1-61.6 against 61.1-62.2: none pays |
| Read forms (probe): upper half-wave at its own addresses (more bank reads) | -2.2% clock: LDS energy follows the bank reads |

- **Not reached.** No try beat the release by more than its twin, so nothing was raced on an RX 7800 XT and nothing
  ships. The switches above were box patches on 94e1e36c; the branch keeps only the two probes and this note.
- **Why it is hard:** our fold already runs at SRBMiner's rate a clock. What is left is energy a MAC: the WMMAs' own
  operand bits (set by the protocol), the LDS reads (at their least for a 64 x 64 wave tile in 256 VGPRs), staging
  and off-chip traffic (under 1% in the miner), and the readout and hash (under 1%, goal-rdna3c).
- **Cost:** one box, 00:03-02:49 UTC, about $0.53; lock salad-6.

### Salad, 15 November: RDNA 3, the readout's and the hash's share, REG2's tile, larger k, and two more PCs against SRBMiner (goal-rdna3c, branch `goal/rdna3c`)

Boxes in `llmjob2`: `rdna3c-x1` (RX 7900 XTX, PC `5ae23c47`) and `rdna3c-y1` (RX 7800 XT, PC `dc17e5be`, moved by
Salad to `770b1eff` at 15:18 UTC). Races are `hashrate.js` 50 s after 15 s, 2-4 rounds, builds in turn, a twin of the
release (next/after-v0.7.2 at 72408267); clocks by the probe in one diag run each. Every build that changes the fold
passed the fold check bit-exact unless it is an ablation.

**What the parts outside the WMMAs cost** (timing-only ablations; the 7800 XT's twin within 0.1%, the XTX's runs had
sd about 1 TH/s):

| Build | RX 7800 XT `dc17e5be` | RX 7900 XTX `5ae23c47` |
|---|---|---|
| release / twin | 32.96-33.01 / 32.96-33.02 | 59.05-59.64 / 58.95-59.69 |
| ablation 64 (new on RDNA 3): the readout without its cross-lane steps or selects | 33.25-33.27 (+0.8%) | 59.67-59.82 (+0.5%) |
| ablation 128 (new): no fused hash | 33.28-33.30 (+0.9%) | 59.62-59.91 (+0.4%) |
| ablation 8: no LDS fragment reads | | 60.36-60.67 (+2.0%) |
| ablation 32: no staging | | 60.29-60.46 (+1.8%) |
| ablation 2: no slab barrier | | 59.47-59.62 (+0.3%) |
| operand fill 32 / 40 / 56 / 63 (48 ships) | | -0.6% / level / level / -0.4% |

**Larger k loses on RDNA 3.** The wrap readout (`-DPEARL_AMD_KWRAP=1`, bit-exact, 226 VGPRs) at k 2048 was level on
both cards; at k 4096 -0.5% (7800 XT) and -0.4% (XTX); at k 8192 -1.2% and -0.9%, at the same clock (2459 against
2456 MHz on the 7800 XT). So the hash and the block's prologue can't be paid for by a longer tile, as RDNA 4 does.

**REG2's 32 x 8 tile on RDNA 3** (`PEARL_AMD_REG2_11=1`, off). In the RDNA 3 layout a lane holds rows 16 mt + 2 g +
l / 16, so the tile's region is four lanes (l, l ^ 1, l ^ 16, l ^ 17): a chunk XORs the lane's 128 sums to two words,
one per row offset, then one `v_permlanex16` step and one DPP step, and keeps RDNA 4's queue of 8 words. The fused
hash takes the regions from LDS as before. Bit-exact; 226 VGPRs, 6 waves, no spills; the slab pair 273 instructions
against 292. verify-hits 400 of 400 on both cards (7800 XT 2,261 hits, 281 salts; XTX 4,033 hits, 506 salts), once
`verify-hits.js` took the core's tile at k 2048 too (it did only when the core chose another k, so the first runs failed
every hit; fixed on the branch).

| Build | RX 7800 XT | RX 7900 XTX |
|---|---|---|
| REG2_11=1 | 33.00-33.03 (+0.1%) | 59.22-59.43 (level) |
| REG2_11=2: its cross-lane steps a slab pair later (236 VGPRs) | 32.64-32.66 (-1.0%) | 58.53-59.08 (-0.8%) |
| REG2_11=3: the same, no rotl 13 at k 2048 | 32.63-32.65 (-1.0%) | 58.53-58.80 (-1.0%) |
| `PEARL_AMD_HASH2`: two fused hashes a thread, on 64 threads | 32.86-32.88 (-0.3%) | 58.85-59.17 (-0.4%) |

REG2 takes 0.1% of ablation 64's 0.8%: under the bar, so it stays off.

**Against SRBMiner 3.8.8, same PC** (ours = 72408267; each miner's mean of its 10 s samples after the first minute;
credit = shares the stand-in pool accepted after the first minute, 2^213, all k 2048, none rejected):

| Card, PC | Pool | Ours | SRBMiner 3.8.8 | Ours / SRB |
|---|---|---|---|---|
| RX 7800 XT, `dc17e5be` | Kryptex, 2 x 300 s | 33.06, 33.05 (2456 MHz) | 30.75, 30.79 (2429 MHz) | **107.4%** |
| RX 7800 XT, `dc17e5be` | stand-in, 1 x 600 s, credited | 2,040 shares | 1,831 | **111%** |
| RX 7800 XT, `770b1eff` | stand-in, 2 x 600 s, shown | 32.86, 32.91 | 34.38, 34.40 | 95.6% |
| RX 7800 XT, `770b1eff` | stand-in, credited | 2,039 + 2,044 = 4,083 | 2,106 + 2,176 = 4,282 | 95.4% (+-1.6%) |
| RX 7900 XTX, `5ae23c47` | Kryptex, 2 x 300 s | 59.41, 59.05 (2673 MHz, 870 mV) | 62.68, 62.68 (2779 MHz, 896 mV) | 94.5% |
| RX 7900 XTX, `5ae23c47` | stand-in, 2 x 600 s, shown | 59.25, 59.18 | 61.27, 61.10 | 96.8% |
| RX 7900 XTX, `5ae23c47` | stand-in, credited | 3,706 + 3,619 = 7,325 | 3,679 + 3,752 = 7,431 | 98.6% (+-1.2%) |

- Ours is steady across PCs (7800 XT 32.9-33.1 on both); SRBMiner read 30.8 on one 7800 XT and 34.4 on the other.
- On the XTX the gap is mostly clock: SRBMiner held 4% more clock at a higher voltage, and did 1.8% more a clock.
  None of the ablations moved our clock that far (all within 2669-2775 MHz in one-run probes).
- **Not reached:** RDNA 3 over 100% on every PC. Nothing here beat the release by more than its twin. No RX 7900 XT ran.
- **Cost:** two boxes, 12:22-16:20 UTC, high priority, `llmjob2`.

### Salad, 15 November: RDNA 3, a persistent fold and a lagged slab barrier (goal-rdna3b, branch `goal/rdna3b`)

Picking up goal-rdna3a's list (below): the cost a launch and a block pay outside the slab loop, then the LDS fragment
reads. Boxes in `llmjob2`: `rdna3b-a1` (RX 7800 XT, PC `362bd356`) and `rdna3b-b1` (RX 7900 XTX, PC `de6b6d38`).
Races are the fold alone (`pearl_amd_foldtest` at the miner's launch, 400 launches, miner operands, builds in turn)
and `hashrate.js` 50 s after 15 s, 3-4 rounds in turn, each with a twin of the release; every build below passed the
fold check bit-exact, and the persistent ones also with 5, 7 and 13 blocks over 32-128 tiles.

**Where a 7800 XT's fold loses, stamps of the release** (fold alone, 2383 MHz): slab 6,647 cycles, 236.6 MAC a CU a
clock, the register-fed WMMA peak at 6 waves and SRBMiner's whole rate. A block spends 1.70% of its time before the
slab loop and 0.80% after it; the launch's gap is 0.39% and the card is busy 97.7% of the launch.

**A persistent fold** (`PEARL_AMD_PERSIST`, off): the host launches one block a slot (3 a WGP; HIP's
`multiProcessorCount` counts WGPs on RDNA 3, and `hipOccupancyMaxActiveBlocksPerMultiprocessor` answered 1, not 3),
and each block's slab loop runs on into its next tile: the last two slabs of a tile load the next tile's first two,
and the fused hash puts the tile's transcripts in the B' rows' unused padding. Taking the prologue and epilogue away
did not pay:

| Build | VGPRs, waves | RX 7900 XTX, fold alone | RX 7900 XTX, hashrate.js | RX 7800 XT, fold alone |
|---|---|---|---|---|
| release | 226, 6 | 55.6-56.4 TMAC/s | 57.1-57.2 TH/s | 32.6-32.9 |
| persistent, tiles b, b + 144, ... | 232, 6 | -3.2% | -2.8% | |
| persistent, tiles from a counter (`PDYN`) | 240, 6 | -3.0% | -2.9% (2639 MHz against 2608) | -3.0% |
| the same, blocks on a WGP started a third or a sixth of a slab apart (`PSKEW`) | 240, 6 | -2.6 to -3.2% | -2.6% | |
| the same, each tile hashed in the next tile's second slab (`PHASH`) | 243, 5 | -5% | -5.4% | -6% |
| the same at a 64-byte slab, 2 blocks a WGP | 256, spills 92 | -40% | | |
| one block a group of 2 tiles, dispatched as the release's (`PGRP=2`) | 231, 6 | -0.5% | -1.1% | level |
| groups of 4 tiles (`PGRP=4`) | 231, 6 | -0.6% | -1.6% | -0.9% |

- The persistent builds run at a higher clock than the release (2639 against 2608 MHz in `hashrate.js` on the XTX)
  and still do 3% less: they lose cycles, not watts. Starting the three blocks of a WGP apart changed nothing, so it
  isn't the blocks running in step. Their slab loops are not the release's (325-372 instructions a slab pair against
  292, and with the counter's store in the loop the compiler moved the first slab's WMMAs below its barrier).
- The fold check passes for all of them; verify-hits of an early counter build: 400 of 400 on the 7800 XT (2,154
  hits, 272 salts).

**A lagged slab barrier** (`PEARL_AMD_LAG3`, off): three LDS buffers of 32-byte rows (the same 36,864 bytes a block,
the 16-byte halves of a row swapped on row bit 3, which keeps 8- and 16-lane passes conflict-free), so slab s is
stored two steps ahead and its fragment reads are issued before step s's barrier, landing while the wave waits there.

| Build | VGPRs, waves | RX 7900 XTX, fold alone | RX 7900 XTX, hashrate.js | RX 7800 XT |
|---|---|---|---|---|
| both k-steps' reads before the barrier | 256, 5 (29 spilled) | -5.2% | -5.8% | -6.0% |
| the first k-step's before, the second after | 256, 5 (29 spilled) | -3.2% | -3.3% | -4.0% |
| the same held to 6 waves | 240, 6 (43 spilled) | -0.4% | -1.4% (2547 MHz against 2608) | -1.3% |
| both before, 6 waves | 240, 6 (56 spilled) | -0.8% | -3.4% | |
| the first k-step's before, slab loads one step ahead in one register set (`LAG3=2 LAG3PF=1`) | 226, 6 | +0.2 to +0.5% | 57.28, 57.28, 57.23 against 57.38, 57.21, 57.18 (level) | +0.25%: hashrate.js 32.81-32.85 in 7 of 7 rounds against 32.73-32.77 (twin within 0.06%); verify-hits 400 of 400 (2,231 hits, 280 salts) |
| both before, loads one step ahead | 226, 6 | 0 to +0.2% | -0.4% | -0.1% |

- Reads before the barrier land while the wave waits there, but at most +0.25% shows on either card: the release's
  fragment reads were hidden by the other two blocks on the WGP already. lagpf's clock in `hashrate.js` was 2598 MHz
  against the release's 2608-2615 on the XTX. Under the +0.3% bar on the 7800 XT and level on the XTX, so it stays
  off (branch `goal/rdna3b`), as do the persistent switches; neither went to this branch.

**Also tried:** RDNA 4's serpentine WMMA order (`PEARL_AMD_SERP=1`) on RDNA 3: XTX `hashrate.js` +0.05%, 7800 XT fold
-0.3 to 0% (level); batch 256 on the 7800 XT -0.65%; loads straight into LDS (`buffer_load ... lds`): ROCm 7.2's
clang refuses them for gfx1100 and gfx1101 ("needs target feature vmem-to-lds-load-insts").

**The release against SRBMiner 3.8.8 on a clean RX 7900 XTX** (`de6b6d38`; ours is next/after-v0.7.2 at 3abe0e96, the
fused hash on gfx1100, built on the box; each miner's mean of its 10 s samples after the first minute; clock probe in
a 180 s turn of each):

| Pool | Ours | SRBMiner 3.8.8 | Ours / SRB |
|---|---|---|---|
| Kryptex, 2 turns of 300 s, shown TH/s | 56.09, 55.98 (2544 MHz) | 57.68, 57.78 (2576 MHz) | 97.1% |
| Stand-in pool, 2^213, 2 turns of 600 s, shown | 55.89, 55.96 | 57.11, 56.97 | 98.0% |
| Stand-in pool, credited shares after the first minute | 3,492 + 3,467 = 6,959 (54.95, 54.65 TH/s) | 3,586 + 3,501 = 7,087 (56.43, 55.09) | 98.2% (±1.2%) |

All shares k 2048, none rejected. On this PC SRBMiner showed about what it was credited (its Kryptex figure 1.5% above
its stand-in one), unlike goal-rdna3a's XTX `c0a9622d`, where it showed 3-4% more. SRBMiner runs 1.3% higher clock and
1.7% more a clock.

- **Not reached:** RDNA 3 over 100%. Nothing here beat the release by its twin's spread on either card except lagpf's
  +0.25% on the 7800 XT. No RX 7900 XT was run.
- **Cost:** about 9.5 box-hours (two boxes, 07:18-12:02 UTC), high priority, `llmjob2`.

### Salad, 15 November: RDNA 3 against SRBMiner 3.8.8, where the rate goes (goal-rdna3a)

The 14 November rows put RDNA 3 at 94-98% of SRBMiner 3.8.8. This compares the two miners on the same PCs with clocks,
then prices what the comparison pointed to. Boxes in `llmjob2`: RX 7900 XTX `c0a9622d`, RX 7800 XT `fce3ea37`, RX 7900
XT (second box, PC not logged). Salad's `amd-smi` now reads clock and voltage. Ours is goal/rdna3a built on each box;
SRBMiner 3.8.8 from its release. Duels: Kryptex, 2 turns of 300 s each, mean of the 10 s samples after the first
minute. Not rows: one PC a class.

**Same PC, old fold (21a6e31b's), with clocks:**

| Card, PC | Ours (TH/s; MHz, mV) | SRBMiner 3.8.8 | Ours / SRB | Per clock |
|---|---|---|---|---|
| RX 7800 XT `fce3ea37` | 33.45, 33.33; 2424-2427, 944 | 34.30, 34.25; 2412-2414, 933 | 97.4% | SRB +2.9% |
| RX 7900 XTX `c0a9622d` | 56.95, 56.95; 2687-2700, 845 | 60.99, 61.15; 2709-2712, 855 | 93.2% | SRB +6.2% |
| RX 7900 XT | 47.76, 47.70; 2480-2496, 869-897 | 49.97, 49.98; 2561-2562, 885-895 | 95.5% | SRB +1.4% |

The clock probe (SHADER_CYCLES over REALTIME) agrees: 7800 XT ours 2422, SRB 2416 MHz; XTX ours 2668, SRB 2721.

**Is SRBMiner credited what it shows?** Each miner on `verify-pool.js` (every share checked as the spec's verifier
does), turns of 600 s, credit from the shares accepted after the first minute:

| Card | Target | Ours: shares (credited TH/s) | SRBMiner: shares (credited TH/s) | Credited ratio | Shown ratio |
|---|---|---|---|---|---|
| RX 7900 XTX | 2^212, 2 turns | 3,561 (57.21, 54.85) | 3,592 (56.05, 56.89) | 99.1% | 97.1% (56.8-57.3 vs 58.4-58.5) |
| RX 7900 XTX | 2^213, 3 turns | 10,565 (56.96, 55.77, 53.61) | 10,685 (56.87, 55.53, 55.54) | 98.9% | 97.3% (55.4-56.6 vs 57.8) |
| RX 7800 XT | 2^213, 2 turns | 4,114 (32.54, 32.26) | 4,247 (32.15, 34.68) | 96.9% | 97.7% |

All shares k 2048, none rejected. On the XTX SRBMiner shows 3-4% more than it is credited (and showed 61 TH/s on
Kryptex against 58 on the stand-in pool), ours 1-2%; credited, the two are within 1.1% (±1.4%). On the 7800 XT the
credited gap matches the shown one. Our XTX PC also slowed over the run (56.6 to 55.4 TH/s shown).

**Where our rate goes.** The fold test's stamps on the XTX (fold alone, 400 launches): 6,707 cycles a slab, 234.5 MAC a
CU a clock, which is SRBMiner's whole rate per clock in the duel (234.7). The fold alone loses 2.3-2.8% of each 9.2 ms
launch to its tail (0.15-0.17 ms) and the gap before the next (0.07-0.12 ms); in the miner the XTX's hash kernel runs
0.40 ms beside the fold on its own queue and the clock is lower (fold alone 2768 MHz, in the miner 2650-2690).
`pearl_amd_peak` on the 7800 XT (MAC a CU a clock, miner data): WMMA from registers 236.6-238.6 at 6 waves, 239.5 at 4;
LDS reads and barrier 235.9; 16 lone VALU a slab 236.1; 16 dependent VALU 225.1; RDNA 4's readout 231.2; LDS, barrier
and readout 225.2-226.1. The kernels ran at 2550-2700 MHz against 2420 in the miner: the card mines power-limited.
The operand fill moves its clock: fill 0 runs 2325 MHz against 2424 at 48 (below).

**What was tried** (`hashrate.js` 50 s after 15 s, rounds in turn with a twin; the 7800 XT's twin within 0.1%, the
XTX's runs had sd 1 TH/s):

| Build | RX 7800 XT `fce3ea37` (base 33.33-33.41) | RX 7900 XT (base 47.69-47.93) | RX 7900 XTX `c0a9622d` (base 56.8-57.9) |
|---|---|---|---|
| 256x256 blocks of 16 waves at 8 a SIMD (`W11_WR=4`, 173 VGPRs, bit-exact) | -0.3% | | -1.5% |
| the same at 4 waves (1 block a WGP) | -8% | | -8% |
| ablation 32, no slab staging (timing only) | +2.8%, clock 2476 vs 2427 MHz | | +2.5% |
| ablation 8, no LDS fragment reads | +2.2%, same clock | | +1% |
| ablation 2, no slab barrier | +1.0% | | +0.5% |
| readout as 2, 3 or 4 chains a region | -0.9, -0.3, -0.6% | | |
| hash in the fold (`PEARL_AMD_FUSE=1`) | -0.1% to level | +0.6% (batch 11.32 vs 11.50 ms) | +0.6% (+1.2% vs twin; batch 9.58 vs 9.72 ms) |
| slot 1's fold on a second queue (`PEARL_AMD_FOLDQ2`) | +0.2 to +0.3% (clock 2405 vs 2424) | -1.0% | -0.8% (folds overlap whole, 2525 vs 2649 MHz) |
| both | +0.3 to +0.5% | | |
| hash on the fold's queue (`PEARL_AMD_HASHQ=0`) | | -0.6% | -0.3% |
| k 4096 | +0.1% | | |
| 4 waves a SIMD (LDS padded to 2 blocks a WGP) | -2.6% | | |
| operand fill 0 / 16 / 32 / 63 / -32 (timing only) | -4.0 / -1.3 / -0.6 / -0.1 / -1.1% | | |
| no-LDS fold (`pearl_amd_peak` direct, directrd) | 32.8 / 31.7 TMAC/s against the fold's ~33.6 | | |

- **256x256 blocks lose on RDNA 3**, where they won 1.2-1.9% on RDNA 4. The staging is worth +2.5-2.8% if free, but a
  third fewer bytes did not pay for the 16-wave barrier and twice-longer blocks.
- **More readout chains lose.** `s_delay_alu` in the loop fell from 25 to 8, and the rate fell: the four interleaved
  chains were not the cost.
- **Shipped on goal/rdna3a:** `PEARL_AMD_FUSE` on by default in gfx1100 and gfx1101 code, `PEARL_AMD_FOLDQ2` on
  gfx1101. verify-hits 400 of 400: fused on the 7900 XT (3,294 hits, 411 salts), fused with FOLDQ2 on the 7800 XT
  (2,247 hits, 282 salts; FOLDQ2 alone 2,249 hits). 226 VGPRs, 6 waves, no spills, as before.

**The new default against SRBMiner 3.8.8, same PCs:**

| Card | Ours, new (TH/s; MHz) | SRBMiner 3.8.8 | Ours / SRB | Old fold on that PC |
|---|---|---|---|---|
| RX 7800 XT `fce3ea37` | 33.62, 33.54; 2437-2447 | 34.25, 34.31; 2413-2417 | 97.9% | 97.4% |
| RX 7900 XT | 48.40, 48.11; 2489-2572 | 49.93, 50.25; 2561-2564 | 96.2% | 95.5% |

`hashrate.js`, new against old in turn, 4 rounds: RX 7800 XT 33.48-33.51 against 33.38 (+0.3 to +0.4%, twin within 0.1%); RX 7900 XT 47.76-48.36 against 47.82-47.89 (+0.4% on the means, sd 0.7 a run).

- **Not verified:** the XTX with the new default (its box was gone by then), and an XTX on a quieter PC.
- **Cost:** about 11 box-hours, high priority, at most two groups at a time.

### Salad, 14 November: the Radeon rows against SRBMiner 3.8.8 (PR #296's build at 21a6e31b)

The current Radeon table. SRBMiner 3.8.8 came out on 13 November and says it improves pearlhash on RDNA 3 and RDNA 4.
Our rows were measured against 3.8.2 and 3.8.6, so every Salad Radeon class was run again with `bench.py`
(`BENCH_SET=amd`) and `bench.sh` at 21a6e31b, which pins 3.8.8. Ours is the HIP core at 21a6e31b, built on each box.
Each miner ran 5 minutes on the same PC after a 10-minute hit check; the two PCs of a class took turns going first.
The 9–10 and 11 November rows below stay as history.

**Above 100%: 3 of 13 (the three RDNA 2 rows, against 3.8.8 since 15 November). Median 96% over all 13; 94% over
the 10 Salad rows.**

| Card | Host | Machine ID | Ours | SRBMiner | % of best | Run |
|---|---|---|---|---|---|---|
| RX 9070 XT | Salad (16 GB) | 0c93191c | 140.1 | 154.8 (3.8.8) | 90% | new |
| RX 9070 XT | Salad (16 GB) | 8c091bb8 | 121.0 | 136.5 (3.8.8) | 89% | new |
| RX 9060 XT | Salad (16 GB) | ec4f1585 | 52.3 | 58.0 (3.8.8) | 90% | new |
| RX 9060 XT | Salad (16 GB) | 250513ab | 73.3 | 80.6 (3.8.8) | 91% | new |
| RX 7900 XTX | Salad (24 GB) | 0e34e7a1 | 58.3 | 59.2 (3.8.8) | 98% | new |
| RX 7900 XTX | Salad (24 GB) | 91701547 | 54.3 | 56.5 (3.8.8) | 96% | new |
| RX 7900 XT | Salad (20 GB) | 76f2e5bb | 48.7 | 50.1 (3.8.8) | 97% | new |
| RX 7900 XT | Salad (20 GB) | c2b8614b | 48.4 | 51.6 (3.8.8) | 94% | new |
| RX 7800 XT | Salad (16 GB) | 362bd356 | 32.2 | 34.1 (3.8.8) | 94% | new |
| RX 7800 XT | Salad (16 GB) | 83dd9640 | 33.6 | 34.5 (3.8.8) | 97% | new |
| RX 6800 | Chedr's HiveOS rig (1400 MHz core cap) | bus 0d:00.0 | 19.36 | 16.01 (3.8.8) | 121% | new (15 Nov) |
| RX 6600 | Chedr's HiveOS rig (1400 MHz core cap) | bus 03:00.0 | 8.91 | 7.38 (3.8.8) | 121% | new (15 Nov) |
| RX 6600 | Chedr's HiveOS rig (1400 MHz core cap) | bus 07:00.0 | 8.88 | 7.37 (3.8.8) | 120% | new (15 Nov) |

The RDNA 2 rows are means of both miners' samples after the first minute, over 4 legs each; the other two RX 6600s are in
"Against SRBMiner 3.8.8 (15 November)" under "RDNA 2 on a lent HiveOS rig". "Ours" is the mean of our samples after the first minute and "SRBMiner" its own last figure, as in every table here.
The means of both miners' samples after the first minute, one every 15 s, give the same picture:

| Machine ID | Ran first | Ours: mean (lowest–highest) | SRBMiner: mean (lowest–highest) | Mean over mean | Copy GB/s | Hit check |
|---|---|---|---|---|---|---|
| 0c93191c | ours | 140.21 (137.4–141.6) | 154.02 (151.14–155.15) | 91.0% | 562 | 400 of 2,309 |
| 8c091bb8 | SRBMiner | 120.88 (118.2–123.5) | 132.97 (130.44–136.51) | 90.9% | 562 | 400 of 2,030 |
| ec4f1585 | ours | 52.33 (51.1–53.3) | 58.07 (57.71–58.59) | 90.1% | 261 | 400 of 1,806 |
| 250513ab | SRBMiner | 73.25 (73.1–73.4) | 80.48 (79.78–80.63) | 91.0% | 277 | 400 of 2,496 |
| 0e34e7a1 | ours | 58.26 (57.8–58.4) | 59.12 (58.89–59.25) | 98.6% | 749 | 400 of 3,981 |
| 91701547 | SRBMiner | 54.26 (53.9–54.9) | 56.70 (55.07–57.30) | 95.7% | 756 | 400 of 3,691 |
| 76f2e5bb | ours | 48.74 (48.2–49.5) | 50.09 (50.04–50.22) | 97.3% | 605 | 400 of 3,308 |
| c2b8614b | SRBMiner | 48.45 (48.1–48.6) | 51.55 (51.06–51.82) | 94.0% | 572 | 400 of 3,318 |
| 362bd356 | ours | 32.19 (32.0–32.3) | 34.14 (34.05–34.30) | 94.3% | 488 | 400 of 2,194 |
| 83dd9640 | SRBMiner | 33.61 (33.5–33.7) | 34.55 (34.36–34.68) | 97.3% | 476 | 400 of 2,291 |

- **SRBMiner 3.8.8 is ahead on every Salad Radeon class.** RDNA 4 is furthest behind, at 89–91% on all four PCs; the
  four RDNA 4 rows agree within 2 points though the PCs' rates differ by 16% (9070 XT) and 40% (9060 XT). RDNA 3 is
  94–98%. Against 3.8.2 and 3.8.6 (9–11 November) the same classes read 101–143%.
- **Our rates did not fall.** On the 9070 XT, ours is 121–140 TH/s, against 121–131 on 9–10 November; SRBMiner went
  from 91.5–105.9 (3.8.2) to 136.5–154.8. On the RX 7900 XTX SRBMiner went from 53.8–54.1 (3.8.6) to 56.5–59.2.
  These are different PCs, so read the change from the ratios, not the TH/s.
- **Hit checks:** all ten PCs PASS, 400 hits verified, 0 failed, 7.95 to 8.03 hits a salt.
- **RDNA 2 is not on Salad** (no RX 6000 class). Its rows come from the lent HiveOS rig, raced against 3.8.8 on 15
  November: 120-121%, the same as against 3.8.6. 3.8.8's notes name RDNA 3 and 4 only, and on RDNA 2 it is no faster.
  Salad also lists no RX 9070 (non-XT) or RX 7900 GRE.
- **No clock, power or temperature:** Salad's `amd-smi` gave no reading on any box, as before.
- **PCs passed over:** RX 7900 XTX `33ec7cef` (Salad moved the box before it set up) and `c0a9622d` (not done 50
  minutes after getting the PC, so `bench.py` moved on). Neither gave a row.
- **Time and cost:** 13:28 to 16:20 UTC, two groups at a time in the `llmjob2` org (`bamd-t1-a-*`, `-b-*`, `-c-*`),
  about 5.5 group-hours. About $1 at Salad's high-priority prices ($0.126 to $0.196 an hour a GPU), plus 4 vCPUs
  and 8 GB a box.

### Salad, 30 October (v0.6.9 against SRBMiner 3.8.0)

PeakMiner mines nothing on AMD.

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

v0.6.12 is about 13% faster on RDNA 4 than these rows; it hasn't been raced against SRBMiner yet.

### Salad, 9–10 November (v0.6.13 against SRBMiner 3.8.2); RX 7900 rows re-run 11 November (v0.6.16 against SRBMiner 3.8.6)

PeakMiner and OneZeroMiner don't mine on AMD.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | OneZeroMiner | % of best |
|---|---|---|---|---|---|---|---|
| RX 9070 XT | Salad (16 GB) | 28f11ef2 | 131.0 | 0 | 91.5 | 0 | 143% |
| RX 9070 XT | Salad (16 GB) | c1940c4c | 120.8 | 0 | 105.9 | 0 | 114% |
| RX 9060 XT | Salad (16 GB) | d3ae7070 | 70.0 | 0 | 54.9 | 0 | 128% |
| RX 9060 XT | Salad (16 GB) | 3540c3e2 | 59.7 | 0 | 54.0 | 0 | 111% |
| RX 7900 XTX | Salad (24 GB) | 6f96bc89 | 54.3 | 0 | 53.8 | 0 | 101% |
| RX 7900 XTX | Salad (24 GB) | ae456453 | 55.3 | 0 | 54.1 | 0 | 102% |
| RX 7900 XT | Salad (20 GB) | dc8fcef5 | 45.9 | 0 | 43.0 | 0 | 107% |
| RX 7900 XT | Salad (20 GB) | 1d5cab04 | 48.5 | 0 | 47.6 | 0 | 102% |
| RX 7800 XT | Salad (16 GB) | 559a11d6 | 31.8 | 0 | 30.9 | 0 | 103% |
| RX 7800 XT | Salad (16 GB) | 16f1685c | 32.9 | 0 | 32.0 | 0 | 103% |
| RX 6800 | Chedr's HiveOS rig (1400 MHz core cap) | bus 0d:00.0 | 19.28 | 0 | 16.01 | 0 | 120% |
| RX 6600 | Chedr's HiveOS rig (1400 MHz core cap) | bus 03:00.0 | 8.90 | 0 | 7.38 | 0 | 121% |
| RX 6600 | Chedr's HiveOS rig (1400 MHz core cap) | bus 07:00.0 | 8.85 | 0 | 7.37 | 0 | 120% |

Every PC passed a 10-minute hit check: 400 hits verified, 0 failed, 7.94–8.05 hits a salt.

The three RDNA 2 rows aren't Salad. They come from a community member's HiveOS rig on 12 November: our CLI v0.7.0
from `next/intel-and-kernel` (in no release yet) with the RDNA 2 fold of `goal/rdna2b` (A' words by DPP, 64-byte
slabs) against SRBMiner 3.8.6, with the cards capped at 1400 MHz by their owner, so compare their "% of best", not
their TH/s. Each is the mean of two 4-minute legs per miner, taken in turns, with four cards mining at once. Both
cards passed verify-hits (400 of 400). Before that fold the same rows read 108% (17.26, 7.94 and 7.93 TH/s).
On 13 November the fold's 8-byte fragment reads and earlier store (`goal/rdna2c`) took them from 115%, 115% and
114% to 118%, and then RDNA 2's own slab loop (`goal/rdna2d`: no address VALU in the loop, an LDS swizzle) to 120%,
121% and 120%: the CLI v0.7.0 with the CI core of that loop against SRBMiner 3.8.6 run again, three cards at once,
two 4-minute legs each. Details are in "RDNA 2 on a lent HiveOS rig", below.

The four RX 7900 rows are the 11 November re-run: the same PCs, v0.6.16 against SRBMiner 3.8.6, both miners run
again ("Salad, 11 November: the RX 7900 rows re-run", below). The other six rows are still v0.6.13 against
SRBMiner 3.8.2. What the four rows read on 9–10 November:

| Card | Machine ID | Ours (v0.6.13) | SRBMiner 3.8.2 | % of best |
|---|---|---|---|---|
| RX 7900 XTX | 6f96bc89 | 49.8 | 54.0 | 92% |
| RX 7900 XTX | ae456453 | 50.8 | 54.1 | 94% |
| RX 7900 XT | dc8fcef5 | 46.3 | 45.4 | 102% |
| RX 7900 XT | 1d5cab04 | 48.4 | 47.7 | 101% |

In that run the 7900 XTX was the one AMD card where SRBMiner led, on both PCs and in both run orders.

Rerun on the first of those PCs (`6f96bc89`), 10 November (item 7), with v0.6.13's AMD kernel (the same fold code,
built at 1c78804b): the duel (ours first, then SRBMiner, 5 minutes each on Kryptex, after a 30-minute sweep had
warmed the card) read ours 54.37 TH/s, mean after the first minute, and SRBMiner 54.40, its last figure (mean 53.89).
That is 99.9%, not the row's 92%. `hashrate.js` read 53.48 to 54.35 in three rounds, and the CLI's trace shows the
fold at 99% of each batch (10.07 of 10.17 ms; hash 0.8%, idle 0.2%). So the row's 49.8 was a low reading of this
PC, not a gap in the kernel. The row stayed until the full rerun of 11 November (above). What off-chip
operand traffic costs here, from the same sweep (each build in turn, three rounds of `hashrate.js`, four of the
fold alone):

| Build | `hashrate.js` TH/s | Fold alone | Clock (stamps) |
|---|---|---|---|
| v0.6.13 | 53.48, 54.27, 54.35 | 54.8-57.3 TMAC/s | 2564 MHz |
| Operands held in the L2 (`PEARL_AMD_ABLATE=16`, timing only) | 55.59, 55.94, 55.93 (+3.3%) | +2.3 to +3.4% | 2640 MHz (+3.0%) |
| No slab staging at all (`PEARL_AMD_ABLATE=32`, timing only) | 57.19, 56.83, 56.79 (+5.4%) | +1.3 to +4.4% | 2698 MHz (+5.2%) |

Cycles a slab are the same in all three (6,699-6,745); waves wait 30-31% of the slab loop at the barrier and 0.7% on
memory. So on the 7900 XTX operands from memory cost about 3% of clock, and a kernel that drew them on chip could get
at most that, less what the draw costs.

RDNA 3's buffer loads (`PEARL_AMD_RDNA3_BUFLD`, on by default on gfx1100 and gfx1101; issue #287, item 2), 7 November:
the release and the new build taking turns on one PC, order rotating, rounds 2-5. The twin is a second copy of the
release, for the noise.

| Card | PC | Fold alone | `hashrate.js` | Twin: fold, `hashrate.js` |
|---|---|---|---|---|
| RX 7900 XTX | `57cc1801` | +1.17, +1.02, +1.25, +1.04% | +0.77, +0.76, +0.77, +0.66% (release 54.27-54.34) | −0.22 to +0.14%, 0 to +0.17% |
| RX 7800 XT | `72dd708c` | +0.81, +1.34, +0.86, +0.88% | +0.82, +0.79, +0.79, +0.75% (release 31.81-31.82) | 0 to +0.62%, 0 to +0.03% |

- TID alone (XTX): fold +0.16%, `hashrate.js` +0.04%. The gain is the buffer loads'.
- verify-hits checked 400 of 400 hits on both PCs. On Kryptex the XTX got 8 accepted shares and 0 rejected in 20
  minutes at 54.5-54.8 TH/s. The 7800 XT's default build (the shipping code) got 1 accepted and 0 rejected in 15
  minutes, but the pool reset the connection 4 times, 3 of them just after a share went out, so those shares got
  no answer. Through the stand-in pool (`verifypool:cli`) the same build's 86 shares all passed the spec's checks,
  and on another 7800 XT (`70b39a06`) it got 2 accepted, 0 rejected, with no reset in 10 minutes. So the resets
  were that PC's link to the pool.
- The 7800 XT's +0.62% twin round is one low release reading (31.765 TMAC/s against 31.97-32.01 in the others).
- Not yet raced against SRBMiner, so the rows above stay as they are.

RDNA 3's staging cuts on top of those loads (`PEARL_AMD_RDNA3_SET`: PFX, CRANK and SCHEDB, on by default on gfx1100
and gfx1101; issue #287, item 8), 7 November: item 2's build (base) and the new one taking turns on one PC, order
rotating, scored from round 2. The twin is a second copy of the base.

| Card | PC | Fold alone | `hashrate.js` | Twin: fold, `hashrate.js` |
|---|---|---|---|---|
| RX 7900 XTX | `28a12f15` | +0.71, +0.60, +0.88, +0.62% | +0.89, +1.01, +1.20% (base 55.22-55.70) | 0 to +0.13%, not run |
| RX 7800 XT | `72dd708c` | +0.83, +0.89, +0.85, +0.87% | +0.90, +0.86, +0.83, +0.93% (base 31.25-31.28) | −0.02 to +0.08%, −0.10 to +0.10% |

- The same with the slab loads pinned two slabs ahead (`PEARL_AMD_PF2PIN`, which the set leaves off) was slower on
  both: fold +0.52% and `hashrate.js` +0.38% on the XTX, +0.67% and +0.66% on the 7800 XT. CRANK and SCHEDB alone
  lost 1.22% in the fold on the 7800 XT.
- verify-hits checked 400 of 400 hits on the 7800 XT. On Kryptex the XTX got 6 accepted shares and 0 rejected in
  10 minutes at 55.6-57.1 TH/s.
- The pinned build was also raced on two more XTX PCs: on `fd4ebc44` the fold +0.87 to +1.37% and `hashrate.js` +0.36
  to +1.15% (mean +0.68%; that PC drifted, its twin read up to +0.75%), verify-hits 400 of 400 and 4 accepted, 0
  rejected in 15 minutes on Kryptex; on `c15f2792` the fold +0.24 to +0.53%, before Salad moved the box.

### RDNA 2 on a lent HiveOS rig, 12 November: the first RX 6000 run

A community member (Chedr) lent his HiveOS rig: five RX 6600 (gfx1032, 28 CUs, 8 GB) and one RX 6800 (gfx1030,
60 CUs, 16 GB), a Pentium G4400 (2 cores), 8 GB RAM, Ubuntu 22.04 (glibc 2.35), kernel 6.12.0-hiveos, ROCm 6.4.4 in
/opt/rocm with no HIP. We ran as an unprivileged user in its home folder: the CI HiveOS package of
`next/intel-and-kernel` at 165f765d (CLI v0.7.0) with its own ROCm 7.2 runtime (`rocm/` beside the core; `ldd` showed
none of the rig's ROCm 6.4.4 libraries loaded), and Node 22.11. The default build, unchanged: `dot4` on both chips,
every hit checked in JS (gfx103x is still an untested kind of card).

**Every rate here is at a 1400 MHz core cap, memory and power stock.** The owner's reset left `OD_SCLK` at 1400 MHz
(the cards ran 1345-1374 MHz). Stock boost is about 2490 MHz on an RX 6600 and 2105 on an RX 6800, so these
understate both cards. Power caps were the default 100 W and 215 W, memory 875 and 1000 MHz. SRBMiner ran later
at the same cap ("Same rig, same cap: against SRBMiner", below). Those are the rows in the AMD table.

| Card | Fold check | verify-hits | Fold alone (TMAC/s) | `hashrate.js` (TH/s) | Power, clock |
|---|---|---|---|---|---|
| RX 6600 (each of 5) | PASS (256x512, 2048x2048) | 400 of 400, 0 failed | 7.96 | 7.91, 7.92, 7.95, 7.99, 7.92 (sd 0.02) | 48-59 W, 1345-1358 MHz |
| RX 6800 | PASS (256x512, 2048x2048) | 400 of 400, 0 failed | 17.32 | 17.26 (sd 0.03) | 120 W, 1371-1374 MHz |
| All six at once | | | | 56.92 (7.91, 7.91, 7.94, 17.25, 7.99, 7.92) | |

- **RDNA 2's `v_dot4_i32_i8` is full rate.** Per CU and clock both chips do about 210 int8 MACs (RX 6600: 7.96 TMAC/s
  over 28 CUs at about 1.35 GHz; RX 6800: 17.32 over 60 at 1.37), 82% of the 256 a full-rate dot4 allows. The
  half-rate case the 2 November projection warned of (RDNA 3's dot4 did 140 on an RX 7800 XT) is ruled out. At the
  projection's game clocks that is about 12.0 TH/s for an RX 6600 (2044 MHz) and 22.8 for an RX 6800 (1815 MHz);
  the projection said 14.0 and 26.6 at 96% of peak. That assumes the rate keeps following the clock. Before the owner's reset, at an 1800 MHz
  cap, one RX 6600 read 10.14 TH/s in verify-hits (7.92 at 1400 MHz): +28% for +29% clock.
- The fold is 99.5% of each batch (`PEARL_AMD_TRACE`: 138.2 ms of 138.9, hash 0.6 ms, idle 0.02 ms on an RX 6600);
  the six cards together lost nothing against each alone, on this 2-core CPU.
- **Kryptex** (US, worker `chedr-rdna2`, all six cards): 57.0 TH/s, 4 shares accepted and 0 rejected in 3 minutes,
  each checked card's first hit right (the RX 6800's among them).
- **Stopped for heat.** Three minutes into that run the RX 6600s reached 91-101 C edge and 101-113 C junction, every
  one reading 0 rpm at full fan PWM, and throttled to 486-1250 MHz (the six read 42.5 TH/s). One (bus 10:00.0)
  dropped off the PCIe bus. The RX 6800, whose fan read about 1100 rpm, stayed at 62 C. Idle afterwards, one RX
  6600 still read 93 C at 24 W. In the first runs, at 53 W and 18 W, the same cards read 35-47 C, so their cooling
  changed during the owner's clock reset and reboot. Earlier runs here were short (40 s to 3 min); runs on this rig
  now go under a guard that stops ours at 85 C edge, 95 C junction, or 10 s of 0 rpm under load.
- **Fold variants, raced after the owner set the fans to 85% (15:02-15:29 UTC), under the guard, same 1400 MHz
  cap.** Built on the rig with ROCm 7.2's clang (the CI release core's twin there, "release", read the same as
  CI's to 0.06%). Three rounds each, order rotating; `hashrate.js` 30 s after 10 s, fold alone in TMAC/s:

  | Build | RX 6800 TH/s | RX 6800 fold | RX 6600 TH/s | RX 6600 fold |
  |---|---|---|---|---|
  | CI release | 17.23, 17.23, 17.23 | 17.32-17.33 | 7.92, 7.92, 7.92 | 7.962-7.963 |
  | release, built on the rig | 17.22, 17.24, 17.23 | 17.31-17.34 | 7.92, 7.92, 7.92 | 7.962-7.964 |
  | DPP readout (`PEARL_AMD_DPP=1`) | 17.29, 17.29, 17.28 (+0.35%) | 17.38 | 7.95, 7.95, 7.95 (+0.38%) | 7.989-7.992 |
  | `PEARL_AMD_DOT4_K=8` (225 VGPRs) | 16.95 x3 (-1.6%) | 17.03 | | |
  | `PEARL_AMD_DOT4_WAVES=5` (191 VGPRs) | 16.04-16.05 (-6.9%) | 16.13 | | |
  | `-mcumode` | 15.84-15.85 (-8.0%) | 15.92 | | |

  The DPP build passed the fold check and verify-hits 400 of 400 on both cards, so it is RDNA 2's default now (200
  VGPRs, 4 waves; `goal/rdna2`). `PEARL_AMD_DOT4_WAVES=6`, which would fit a third block a WGP, caps the fold at 160
  VGPRs and spills 50, so it wasn't run.
- **What the card's dot4 can do:** a kernel of nothing but `v_dot4_i32_i8` from registers (64 accumulators a lane)
  did 20.33 TMAC/s at 4 waves a SIMD and 20.62 at 8 on the RX 6800 at 1388 MHz: 244-248 MAC a CU a clock, 96% of
  AMD's 256. The fold's 17.3 is 85% of that; the loads, barriers and readout take the rest.
- **Kryptex again, 10 minutes, fans at 85%** (15:35-15:45 UTC, CI release CLI, `--gpu-index 0,1,3,4`: three RX
  6600s and the RX 6800; the RX 6600 whose fan reads 0 rpm left out, and the one at 13:00.0 had gone from the bus
  after the first run): 6 shares accepted, 0 rejected, 40.96 TH/s mean after 30 s (`hashrate.js` read 40.95 for
  the four together), every card's first checked hit right. The guard never fired: at most 53 C edge and 61 C
  junction (an RX 6600), the RX 6800 47/57 C.
- Not measured: the rate at stock clocks, and anything longer than 10 minutes.
- Two display slips seen: the CLI's header named all six cards "RX 6600/6600 XT/6650 XT" (the per-card lines are
  right), and `hashrate.js` reports the AMD core's fold as "unresolved".
- CI's Linux fold check binary needed glibc 2.38 (`__isoc23_strtol` from `atoi`), so it couldn't start here: fixed
  on `goal/rdna2`, and CI now uploads it as `pearl-amd-foldtest-ubuntu-latest`.

#### Same rig, same cap: against SRBMiner (12 November, 18:03-19:09 UTC)

Chedr's miner was off: no SRBMiner process of his, HiveOS reporting 0, every card idle. Ours was the CI HiveOS
package of `next/intel-and-kernel` at 5ced0641 (CLI v0.7.0, the DPP readout on by default). SRBMiner was 3.8.6, the
version the Salad benchmark pins (md5 checked), run as `bench.sh` runs it (`--algorithm-gpu pearlhash`, `--gpu-id`
for the card). Both on Kryptex US with the same address, legs taking turns (ours, SRBMiner, ours, SRBMiner), 60 s
of warm-up then 240 s measured, a reading every 10 s, all under the guard. Ours is the mean of the CLI's stats file
(`--stats-file`, per card); SRBMiner's the mean of its own API (`hashrate.gpu`, per card). Same settings as above:
1400 MHz core cap, memory and power stock, fans at 85%. GPU 2 (bus 0a:00.0), whose fan reads 0 rpm, wasn't used.

| Card (bus) | Ours, legs 1 and 3 (TH/s) | SRBMiner 3.8.6, legs 2 and 4 | % of SRBMiner | Power, ours / SRBMiner | Clock | Max edge / junction |
|---|---|---|---|---|---|---|
| RX 6600 (03:00.0), alone | 7.940, 7.946 | 7.380, 7.371 | 107.7% | 46.0 / 48.4 W | 1350 / 1348 MHz | 46 / 51 C |
| RX 6800 (0d:00.0), alone | 17.266, 17.258 | 16.034, 16.023 | 107.7% | 118.9 / 118.0 W | 1370 MHz | 48 / 57 C |
| Four cards at once | 41.093, 41.085 | 38.183, 38.159 | 107.6% | | | 54 / 62 C |
| of which RX 6600 (03:00.0) | 7.946, 7.954 | 7.388, 7.380 | 107.7% | 46.0 / 48.5 W | 1350 MHz | 48 / 54 C |
| of which RX 6600 (07:00.0) | 7.926, 7.928 | 7.370, 7.366 | 107.6% | 53.2 / 55.0 W | 1347 MHz | 46 / 52 C |
| of which RX 6800 (0d:00.0) | 17.285, 17.257 | 16.035, 16.032 | 107.7% | 118.7 / 117.9 W | 1370 MHz | 48 / 58 C |
| of which RX 6600 (10:00.0) | 7.935, 7.946 | 7.389, 7.381 | 107.5% | 55.2 / 57.0 W | 1349 MHz | 54 / 62 C |

- **Ours is 7.6-7.7% ahead on every card**, alone or four at once, at the same clock, and draws 2 W less on each
  RX 6600 (the same on the RX 6800). Each miner's legs agree to 0.1%.
- SRBMiner autotunes nothing for pearlhash on AMD (its API's `gpu_autotune_results` reads "0") and printed only to a
  terminal (`--list-devices` showed nothing without one); its device order matched ours (GPU 0-4 = buses 03, 07, 0a,
  0d, 10).
- **SRBMiner 3.8.7** (11 November; its notes change pearlhash only on NVIDIA RTX 30, 40 and 50), the same way on
  the RX 6600 at 03:00.0 (19:09-19:36 UTC): ours 7.954 and 7.946, 3.8.7 7.383 and 7.381, then 3.8.6 7.380. So 3.8.7
  is no faster here (107.7% either way). The Salad benchmark now pins 3.8.7 for its NVIDIA rows.
- Shares in the 12 legs (about 2.1M difficulty): ours 5, SRBMiner 4, none rejected. Too few to compare.
- The guard never fired. Nothing of ours or SRBMiner's was left running after.

#### The dot4 fold, faster: A' words by DPP, 64-byte slabs (12 November, 19:45-21:00 UTC)

Same rig, same cap, under the guard. Builds made on the rig (ROCm 7.2 clang, gfx1030 and gfx1032) and raced with
`race.sh`'s method: each build's fold alone (fold check binary, the miner's operands) and `hashrate.js` 30 s after
10 s, builds taking turns, the CI release core of 5ced0641 ("CI") among them.

What held the old loop, from timing-only builds on the RX 6800 (fold alone; the release 17.33-17.34 TMAC/s):

| Left out | TMAC/s | |
|---|---|---|
| LDS fragment reads (each slab's read once, used at every k-step) | 19.21, 19.22 | +10.9% |
| the slab barrier | 18.41, 18.42 | +6.2% |
| the slab staging (global loads, LDS stores) | 17.99, 18.00 | +3.8% |
| staging and barrier | 19.25, 19.26 | +11.1% |

So two changes, now the RDNA 2 default (`goal/rdna2b`):

- **A' words by DPP** (`PEARL_AMD_DOT4_DPPA`). The eight lanes of a dpp8 group hold the same eight A' rows, so each
  lane reads one row's word and every `v_dot4c` takes the word it needs from that lane through its own dpp8 operand
  (`v_dot4c_i32_i8_dpp`, no extra instruction). LDS reads a k-step fall from 24 words a lane to 10; LDS instructions
  a slab from 104 to 40. 191 VGPRs.
- **64-byte slabs for the dot4 fold** (`PEARL_AMD_DOT4_SLAB`): half the barriers and stores. 61,440 bytes of LDS a
  block, still 2 blocks a WGP. The host lays out each fold's operands by that fold's slab. (On an RX 9060 XT, 64-byte
  slabs were 4% slower in the WMMA fold, so the other folds keep 32.)

| Build | RX 6800 `hashrate.js` (TH/s) | fold alone (TMAC/s) | RX 6600 (03:00.0) `hashrate.js` | fold alone |
|---|---|---|---|---|
| CI release (5ced0641) | 17.27, 17.26, 17.26 | | 7.95, 7.95, 7.95 | |
| release, built on the rig | 17.25-17.29 | 17.33-17.38 | 7.95, 7.95 | 7.988-7.990 |
| 64-byte slabs | 17.76-17.79 (+3.0%) | 17.84-17.90 | 8.19 (+3.0%) | 8.230 |
| A' by DPP | 18.04 (+4.5%) | 18.14 | 8.31, 8.31 (+4.5%) | 8.352-8.354 |
| **both (the new default)** | **18.35, 18.36, 18.36 (+6.4%)** | 18.43-18.47 | **8.45, 8.46, 8.46 (+6.4%)** | 8.499-8.506 |

- The new default passed the fold check (256x512 and 2048x2048) on both cards, and verify-hits 400 of 400, 0 failed,
  on both (RX 6800 120 s at 18.38 TH/s, RX 6600 180 s at 8.46). 202 VGPRs, 4 waves, no spills on every gfx103x
  target (`build-amd.sh check`).
- **Not what held it: VGPR bank conflicts.** RDNA 2 reads each VGPR bank (register number mod 4) once a clock. In a
  probe of nothing but `v_dot4c` from registers (RX 6800, 4 waves a SIMD) the three operands in three banks ran 253
  MAC a CU a clock; accumulator and A' word in one bank 194 (189 with A' changing); A' and B' in one bank, B'
  changing, 194; accumulator and B' in one bank 126. 643 of the old fold's 1032 dot4s read two operands from one
  bank. But a build that placed every operand by hand (physical register constraints, layout fitted to the paired
  LDS reads: 1029 of 1032 clean, 222 VGPRs, no spills) ran 17.15 against the release's 17.27 TH/s, and +0.1% on
  top of the DPP build. So the loop waits on LDS and barriers, not on the register file.
- Also tried: the next slab stored at the top of each slab instead of the bottom (16.66-16.68, -3.4%), and an
  LDS-only slab barrier (17.26-17.27, no change).
- The DPP form needs dpp8, which gfx10 and later have; RDNA 3 and RDNA 4 run their WMMA folds, whose operands can't
  come through a dpp8 operand, so this doesn't carry over. What may: their loops' slab barrier and staging, priced
  here at 6% and 4% of RDNA 2's.
- **Mining, against SRBMiner again** (21:02-21:24 UTC): the CLI of 5ced0641 with this core, on the four cards whose
  fans read rpm, Kryptex US, legs taking turns as above. Ours 43.689 and 43.770 TH/s, SRBMiner 3.8.6 38.184 and
  38.155: 114.6%. Per card: 03:00.0 8.446, 8.472 against 7.385, 7.376 (114.6%); 07:00.0 8.416, 8.448 against 7.377,
  7.372 (114.3%); 0d:00.0 (RX 6800) 18.398, 18.391 against 16.032, 16.025 (114.8%); 10:00.0 8.428, 8.460 against
  7.390, 7.382 (114.3%). Same power as before (46-55 W an RX 6600, 118 W the RX 6800), same clocks. Ours 3 shares
  accepted, SRBMiner 1, none rejected. Max 55/63 C; the guard never fired.

#### 8-byte fragment reads, and what else was tried (13 November, `goal/rdna2c`)

Same rig, same 1400 MHz cap, Chedr's miner off (no miner process of his, HiveOS 0), every run under the guard (it
never fired). Builds made on the rig (ROCm 7.2 clang). "Fold alone" is the fold check binary on the miner's
operands, 30 launches, builds taking turns, order rotating, two or three rounds; `hashrate.js` is 30 s after 10 s.
The base is the shipped fold of `goal/rdna2b` (A' by DPP, 64-byte slabs): 18.42-18.48 TMAC/s on the RX 6800.

**What held the shipped fold** (timing-only builds, `PEARL_AMD_DOT4_ABL`, RX 6800, fold alone):

| Left out | TMAC/s | |
|---|---|---|
| LDS fragment reads (each slab's read once) | 19.65, 19.66 | +6.7% |
| the slab barrier | 19.20, 19.21 | +4.2% |
| the slab staging (global loads, LDS stores) | 18.97, 18.98 | +3.0% |
| LDS stores only | 18.97, 18.98 | +2.9% |
| global loads only | 18.49, 18.51 | +0.3% |
| the readout (one at the end) | 18.82, 18.82 | +2.1% |
| barrier and staging | 19.62, 19.62 | +6.4% |
| the dpp8 operand (plain `v_dot4c`) | 18.46, 18.48 | +0.1% |

So the fragment reads again. **Shipped: 8 bytes a read** (`PEARL_AMD_DOT4_DPPA_KB=8`, the new default): each lane
reads its A' row and its eight B' rows with `ds_read_b64` and runs two k-steps on them, half the LDS instructions.
217 VGPRs (202 before), still 4 waves, no spills or scratch on gfx1030-1036 (`build-amd.sh check`).

| Card | Build | Fold alone (TMAC/s) | `hashrate.js` (TH/s) |
|---|---|---|---|
| RX 6800 (0d:00.0) | base | 18.43-18.48 | 18.34, 18.37, 18.37 |
| | **8-byte reads** | 18.99-19.01 | **18.88, 18.89, 18.89 (+2.9%)** |
| RX 6600 (03:00.0) | base | 8.496-8.504 | 8.46, 8.46, 8.46 |
| | **8-byte reads** | 8.746-8.747 | **8.70, 8.70, 8.70 (+2.8%)** |
| RX 6600 (07:00.0) | base | 8.470, 8.473 | 8.43, 8.43 |
| | **8-byte reads** | 8.705, 8.722 | **8.67, 8.68 (+2.9%)** |

With the store moved before the readout (next bullet), the shipped build of the branch against the same base, then
checked again: RX 6800 18.95, 18.97, 18.97 TH/s against 18.36, 18.37, 18.37 (+3.3%; fold 19.04-19.09); RX 6600
03:00.0 8.73, 8.74, 8.74 against 8.46 x3 (+3.3%); RX 6600 07:00.0 8.71, 8.72 against 8.43, 8.44 (+3.3%). Fold check
PASS on both chips, verify-hits 400 of 400, 0 failed, on both (RX 6800 19.02 TH/s at the end, RX 6600 8.71), 216
VGPRs, 4 waves, no spills or scratch on gfx1030-1036. Against SRBMiner's 12 November figures: 118.3%, 118.4% and
118.2%.

- Fold check PASS (256x512 and 2048x2048) on both chips; verify-hits 400 of 400, 0 failed, on the RX 6800 (120 s,
  18.79 TH/s at the end) and the RX 6600 (180 s, 8.70).
- **Against SRBMiner:** not re-run this time (no pool address in this session, so neither miner went to the pool).
  Against its 12 November figures on the same cards and cap (16.03 TH/s on the RX 6800, 7.38 and 7.37 on the RX
  6600s), `hashrate.js` puts the new fold at 118%. The previous fold read 18.36-18.39 TH/s mining in the CLI
  and 18.34-18.37 in `hashrate.js`, so the two agree within 0.2% on this rig. The goal was 120% (19.25 and 8.85).
- What the 8-byte loop still spends (timing-only, RX 6800, against 18.97): no barrier +3.1%, no staging +2.6%, no
  readout +1.7%.
- **Shipped too: the next slab's store before the readout**, not after, so the stores finish during the readout's
  VALU instead of in the wait before the barrier (RDNA 2's fold only; every other fold keeps the old order).
  `hashrate.js`, three rounds: RX 6800 18.95, 18.98, 18.97 against 18.89, 18.90, 18.89 (+0.4%); RX 6600 (03:00.0)
  8.73, 8.74, 8.74 against 8.70 x3 (+0.4%). An LDS-only slab barrier on top changed nothing (19.04 either way).

**Tried and dropped** (RX 6800, fold alone unless noted):

- 16-byte reads (`ds_read_b128`, four k-steps): 249 VGPRs, 16.31 (-14%; RX 6600 -14%).
- The next slab's LDS store at k-pair 2, 4 or 6 of the slab instead of after it: 18.91, 18.86, 18.81 (-0.3 to -0.8%).
- A' rows 72 bytes apart in LDS instead of 80 (fewer bank conflicts on the A' reads, stores split in two): 18.79 (-0.9%).
- No empty asm between k-pairs (`PEARL_AMD_DOT4_SPLIT=0`): 15.31 (-19%).
- **A' from SGPRs** (a wave holds 16 whole rows of A' against all 256 columns, so a k-step's A' words are the same
  in every lane: scalar loads straight from memory, no A' staging, `v_dot4c` with an SGPR operand, no dpp8; B' alone
  in LDS, 72-byte rows, conflict-free `ds_read2_b64`). Fold check PASS. In a probe of nothing but dot4s it ran 246 MAC
  a CU a clock against dpp8's 231-235, but in the fold the dpp8 operand turned out free (above). 18.18 as written;
  18.29 with every accumulator and B' word pinned to a register so none shares a bank; 18.60 with the next k-pair's
  scalar and LDS loads issued before each k-pair's dot4s (one `lgkmcnt(0)` wait a k-pair, since scalar loads return
  out of order); 18.67 (hashrate.js 18.56-18.58, +1.1%) with the store early. Below the 8-byte reads, and far more
  code. 6 waves a SIMD (its LDS allows 3 blocks a WGP) needs 160 VGPRs: it spilled 60-276 bytes and ran 18.11.
- **No LDS at all**: A' by scalar loads as above, B' straight from memory into VGPRs (operands laid out in 16-byte
  slabs, so a wave's load is 4 cache lines), no staging, no barrier. Fold check PASS. 18.31 (one k-pair of
  prefetch), 18.25 (three, 5 spills), 18.20-18.29 with an `s_barrier` every 2, 4 or 8 k-pairs to keep the waves
  together, 16.37-16.39 in CU mode. The same build reading one k-pair's B' over and over ran 20.12, so it is the B'
  traffic from L0 and L2, not the loop.
- The dot4 rate depends on the data: in the probe, all-zero operands ran 20.6 TMAC/s at 1389-1390 MHz and random
  ones 20.25 at 1359-1361 MHz, the same MACs a clock. The cards sit below the 1400 MHz cap under load, and ablations
  that leave LDS unwritten (so zeros) read high for that reason too.

#### RDNA 2's own slab loop: addresses out of the loop and an LDS swizzle (13 November, `goal/rdna2d`)

Same rig, same 1400 MHz cap, Chedr's miner off (no miner process of his, HiveOS reporting no miner, the cards idle),
every run under the guard (it never fired). Builds made on the rig with ROCm 7.2's clang. Races as above, builds
taking turns, order rotating; `hashrate.js` 30 s after 10 s, with the card's clock, power and temperatures read
every 2 s while it runs (the means are over the samples at 90% busy or more). The base is the shipped fold of
`goal/rdna2c` ("final2", 18.95-18.96 TH/s on the RX 6800), with a second build of the same source as its twin.

**Where the shipped loop went.** In its gfx1030 assembly (216 VGPRs, 4 waves) a slab is 2048 `v_dot4c_i32_i8_dpp`,
40 LDS reads, 6 global loads, 6 LDS stores and about 51 other VALU: 24 for the slab loads' 64-bit addresses, 10
for the LDS stores' and 12 for the LDS read bases, which the loop's parity (s & 1) made new every slab. The
readout adds about 94 VALU a chunk (2 slabs). RDNA 2 issues one VALU a clock a SIMD, and the timing-only builds
said the loop is close to bound by that, so those VALU are time. And by hand, the LDS banks: at 80-byte rows (20
banks apart) rows 8 apart share a bank, so a k-pair's A' read (rows 0, 4, 8, ... 28 and 1, 5, ... 29 in each 16
lanes) put 4 rows on a bank, and each 8 lanes of a 16-byte staging store (rows r and r + 1) 2. B' reads were
conflict-free.

**Shipped, the RDNA 2 default on Linux:**

- `PEARL_AMD_DOT4_ADDR`: the slab loads as buffer loads (PEARL_AMD_BUFLD, with gfx10's resource word 3,
  0x31014000, and PEARL_AMD_TID); LDS laid out [A' 0][B' 0][A' 1][B' 1], so a slab's two buffers are one SGPR
  offset apart for A' and B' alike; every LDS address a lane base plus that offset plus an immediate. 6 address
  VALU a slab, not about 51. 214 VGPRs (215 with the swizzle).
- `PEARL_AMD_DOT4_SWZ`: A' row r of a tile at LDS row r ^ (((r >> 3) & 3) << 1), which puts the A' reads 2 rows
  a bank (the best 8-byte reads can do at 80-byte rows), and staging lanes mapped so each 8 lanes of a store write
  rows 4 apart, which covers the 32 banks once. `test/amdDot4Addr.test.js` checks the map and the bank counts.

| Build (RX 6800, 0d:00.0) | `hashrate.js` (TH/s) | fold alone (TMAC/s) | Clock, power |
|---|---|---|---|
| shipped (final2) | 18.96, 18.95, 18.96; 18.96, 18.95; 18.96, 18.96 | 19.07 | 1368-1369 MHz, 120 W |
| its twin | 18.93, 18.95, 18.95 | 19.01-19.07 | |
| ADDR | 19.10, 19.11, 19.11; 19.08, 19.12 (+0.8%) | 19.18-19.24 | 1366-1368 MHz, 119-120 W |
| ADDR, twin | 19.08, 19.10 | 19.19-19.22 | 1367-1368 MHz, 119-120 W |
| **ADDR + SWZ** | **19.33, 19.34; 19.31, 19.34 (+2.0%)** | 19.41-19.46 | 1366-1368 MHz, 120 W |
| ADDR + SWZ, twin | 19.33, 19.34 | 19.45-19.46 | 1368 MHz, 120 W |

Max 46/55 C edge/junction in every leg of races b and c (race a read the sensors at idle).

**Tried and dropped** (RX 6800, `hashrate.js`, same races):

- Each k-pair's LDS reads one k-pair ahead, into a second set of fragment registers (219 VGPRs): 19.07-19.08 on
  ADDR, so the reads' wait was hidden by the other waves already.
- The readout after the barrier, once the next slab's first reads are issued (as RDNA 4's late readout): 18.46-18.47
  (-3.3% on ADDR); with the reads ahead too, 18.78-18.79.
- The readout's lane bit 4 step (v_mov, v_permlanex16, XOR) taken out of each chunk and done once after the loop,
  each lane keeping its half of four words a region (223 VGPRs, 12 VALU a chunk fewer): 19.07 on ADDR, 19.27 on
  ADDR + SWZ (-0.3%).
- A' reads with no bank conflict (A' rows 16-31 of each 32 starting 8 bytes into their row, so those rows' 16-byte
  stores are 8-byte aligned): 19.29, 19.30, no faster than 2 rows a bank.
- 16 waves a block of one 32x64 block of C each (64 sums a lane, 128 VGPRs, 8 waves a SIMD, 512-thread blocks):
  19.14, 19.14 (-1.0% on ADDR + SWZ), 19.21 with conflict-free A' reads (8 bytes of scratch); 123-124 W.
- The next slab's store issued after k-pair 6's or 7's reads, not after the slab's dot4s (`PEARL_AMD_DOT4_STKP`, an
  off-by-default switch): k-pair 6 read 19.36, 19.36 on the RX 6800 against 19.31-19.34 and 8.92 x3 on the RX 6600
  against 8.91 x4 (+0.1 to +0.2%, twin spreads 0.05% and 0%); k-pair 7 19.33, 19.33. Left off: too small to ship
  without its own pool run. (Its first builds put the store inside a preprocessor test of a C++ constant, so they
  stored nothing and failed the fold check; that race was stopped and thrown away.)

What the ADDR loop still spends (timing-only builds, RX 6800, against its 19.08-19.12): no slab barrier 19.80, 19.81
(+3.8%); each slab's fragments read once 19.66, 19.67 (+3.0%; the swizzle then took 1.3 of that); one readout at
the end 19.39, 19.40 (+1.6%); no staging 19.16, 19.17 (+0.4%).

**The shipped source** (d5702de8, built on the rig; "fin", and a twin of it), against the shipped fold of
`goal/rdna2c`, clock and power sampled under load:

| Card | Build | `hashrate.js` (TH/s) | fold alone (TMAC/s) | Clock, power, max edge/junction |
|---|---|---|---|---|
| RX 6800 (0d:00.0) | goal/rdna2c | 18.95, 18.96 | 19.07 | 1368 MHz, 120 W, 46/55 C |
| | **this** | **19.31, 19.34, 19.33, 19.31, 19.33 (+2.0%)** | 19.40-19.46 | 1365-1368 MHz, 120-121 W, 46/56 C |
| | twin | 19.32, 19.33 | 19.43-19.45 | 1367 MHz, 120 W |
| RX 6600 (03:00.0) | goal/rdna2c | 8.74, 8.73, 8.73 | 8.78 | 1349 MHz, 46 W, 43/48 C |
| | **this** | **8.91, 8.91, 8.91, 8.91 (+2.0%)** | 8.95-8.96 | 1348-1349 MHz, 47 W, 43/49 C |
| | twin | 8.91, 8.91, 8.91 | 8.96 | 1349 MHz, 47 W |

- Fold check PASS (256x512 and 2048x2048) on both chips; verify-hits 400 of 400, 0 failed, on the RX 6800 (120 s,
  19.34 TH/s at the end) and the RX 6600 (180 s). 215 VGPRs, 4 waves, no spills or scratch on gfx1030-1036
  (`build-amd.sh check`). The gfx1030 loop has 6 address VALU a slab: four read bases, the store's, and the flip.
- Against SRBMiner's 12 November figures on the same cards and cap (16.03 TH/s on the RX 6800, 7.38 on the RX 6600
  at 03:00.0): 120.5% and 120.7%. The goal was 120%.

**Mining, against SRBMiner again** (18:44-19:06 UTC): the release path, the CLI of the CI HiveOS package of 5ced0641
(v0.7.0) with the CI-built core of d5702de8 (`pearl-core-hip-ubuntu-latest`), against SRBMiner 3.8.6, three cards at
once (03:00.0, 07:00.0, 0d:00.0; 10:00.0 was in use by another test and 0a:00.0's fan reads 0 rpm), Kryptex US, legs
taking turns (ours, SRBMiner, ours, SRBMiner), 60 s warm-up then 240 s, means of each miner's own figures as above.

| Card (bus) | Ours, legs 1 and 3 (TH/s) | SRBMiner 3.8.6, legs 2 and 4 | % of SRBMiner | Power, ours / SRBMiner | Clock | Max edge / junction |
|---|---|---|---|---|---|---|
| RX 6600 (03:00.0) | 8.909, 8.893 | 7.376, 7.378 | 120.7% | 46.7 / 48.4 W | 1349-1350 / 1349 MHz | 45 / 51 C |
| RX 6600 (07:00.0) | 8.863, 8.833 | 7.363, 7.376 | 120.1% | 54.0 / 55.0 W | 1344-1345 / 1347 MHz | 44 / 50 C |
| RX 6800 (0d:00.0) | 19.340, 19.225 | 16.018, 16.009 | 120.4% | 119.8 / 117.9 W | 1367 MHz | 47 / 56 C |
| All three | 37.112, 36.950 | 30.758, 30.764 | 120.4% | | | |

- Every card is at or above the 120% goal. Ours 6 shares accepted, 0 rejected; SRBMiner 2 accepted, 1 rejected (stale).
  Each miner's legs agree to 0.4% (ours) and 0.02% (SRBMiner). The guard never fired.
- SRBMiner read the same as on 12 November (16.03, 7.38 and 7.37 TH/s then).
- In the CLI the RX 6800 read 19.23-19.34, `hashrate.js` 19.31-19.34: within 0.3%.

#### Against SRBMiner 3.8.8 (15 November, 20:33-21:29 UTC)

Chedr stopped his miner at 20:27 UTC and rebooted the rig. At 20:33 there was no miner process of his, HiveOS
reported 0, and every card was idle. His reset put the cores back at the 1400 MHz cap (pp_dpm_sclk level 1 = 1400,
OD_SCLK 1400, manual level), so these rows are at the same clock as 12-13 November. Memory and power stock (875 MHz
and 100 W on the RX 6600s, 1000 MHz and 215 W on the RX 6800). The RX 6600 at 13:00.0, which had left the bus on 12
November, was back. 0a:00.0, whose fan reads 0 rpm, was not used.

Ours: the CLI of the CI HiveOS package of 5ced0641 (v0.7.0, as on 13 November) with the CI HIP core of d2c8e93a, the
tip of `next/after-v0.7.2` (`pearl-core-hip-ubuntu-latest`, md5 cde362c2). SRBMiner 3.8.8 from its GitHub release,
run as `bench.sh` runs it. Its API mapped ids 0, 1, 3, 4 to buses 03, 07, 0d, 10, the same order as ours. Kryptex
US, same address, legs taking turns (ours, SRBMiner, ours, SRBMiner), 60 s warm-up then 240 s, a reading every 10 s,
means of each miner's own figures, everything under the guard (it never fired). Round a ran four cards, round b all
five.

| Card (bus) | Ours, 4 legs (TH/s) | SRBMiner 3.8.8, 4 legs | % of SRBMiner | Power, ours / SRBMiner | Clock, ours / SRBMiner | Max edge / junction |
|---|---|---|---|---|---|---|
| RX 6600 (03:00.0) | 8.918, 8.923, 8.920, 8.863 | 7.376, 7.384, 7.385, 7.373 | 120.7% | 46.0 / 47.9 W | 1348-1349 / 1349 MHz | 49 / 55 C |
| RX 6600 (07:00.0) | 8.889, 8.892, 8.901, 8.843 | 7.367, 7.379, 7.378, 7.371 | 120.4% | 54.0 / 54.8 W | 1346 / 1347-1348 MHz | 45 / 51 C |
| RX 6800 (0d:00.0) | 19.382, 19.376, 19.371, 19.299 | 15.983, 15.986, 16.039, 16.029 | 120.9% | 118.7 / 116.6 W | 1369 / 1369-1370 MHz | 48 / 57 C |
| RX 6600 (10:00.0) | 8.979, 9.006, 8.969, 8.938 | 7.438, 7.441, 7.441, 7.434 | 120.6% | 53.0 / 54.0-54.5 W | 1359 / 1359 MHz | 47 / 53 C |
| RX 6600 (13:00.0), round b only | 8.850, 8.829 | 7.384, 7.377 | 119.8% | 56.0 / 56.9 W | 1346-1347 / 1349 MHz | 54 / 62 C |
| Four cards, round a | 46.168, 46.197 | 38.164, 38.189 | 121.0% | | | |
| Five cards, round b | 55.012, 54.770 | 45.627, 45.583 | 120.4% | | | |

- **3.8.8 is no faster than 3.8.6 on RDNA 2.** It read 16.01 TH/s on the RX 6800 and 7.37-7.38 on the RX 6600s, as
  3.8.6 did on 12-13 November. Ours is 120-121% on every card, at the same clock and 1-2 W less on each RX 6600.
- Each miner's legs agree to 0.8% (ours; round b's second leg was 0.4-0.7% lower on every card) and 0.4% (SRBMiner).
- Kryptex shares over the 8 legs: ours 4 accepted, SRBMiner 6, none rejected. Too few to compare.
- **Credit check, RX 6800** (21:17-21:29 UTC, `verify-pool.js serve` at 2^213, ours then SRBMiner, 330 s each,
  shares counted after the first 60 s): ours 580 accepted, 0 rejected in 271 s (2.14 a second), showing 19.35 TH/s;
  at 2^213 that rate should give 2.20 shares a second, so ours is credited 97% of what it shows. SRBMiner 202
  accepted, 0 rejected (0.75 a second) while showing 15.95 TH/s, which should give 1.81: 41% of what it shows. Not
  explained yet. On Salad RDNA 3 (above) SRBMiner's stand-in credit matched its shown rate within 3-4%, and on Kryptex
  here it got as many shares as ours. So read this as "our shown rate is honest", not as SRBMiner losing 59%; the
  next step is to log the tile each miner's shares declare.


After the six RDNA 4 cuts the slab loop has no VALU but the readout's, about 88 a chunk, run after the chunk's last
WMMAs. `PEARL_AMD_RDLATE` moves it into the next slab, after that slab's fragment loads are issued (=2 splits it over
two slabs), and `PEARL_AMD_RDTREE` XORs each region's sums as three chains, not one. Both now default on with the set
(Linux gfx12). Against the release on one PC, `test/sweep.sh`: each build's fold checked bit for bit first, then
`hashrate.js` 60 s after 20 s, 4 rounds, the builds taking turns, round 1 not scored. ctl is the release under an
extra flag, the same code. Not rows: no competitor ran.

| Card, k | PC | Release (TH/s) | ctl | RDTREE | RDLATE=1 | RDLATE=2 | RDLATE=1 + RDTREE | Both (shipped) |
|---|---|---|---|---|---|---|---|---|
| RX 9070 XT, 8192 | `33042316` | 125.73–125.81 | −0.02, +0.01, +0.02% | +1.22, +1.28, +1.22% | +1.34, +1.25, +1.26% | +1.38, +1.28, +1.27% | not run | **+1.72, +1.66, +1.67%** |
| RX 9060 XT, 4096 | `f9b8be8b` | 70.91–70.92 | 0.00, 0.00, 0.00% | +1.03, +1.06, +1.04% | not run | not run | +1.62, +1.73, +1.69% | **+1.38, +1.42, +1.41%** |

- **The fold alone** (k 4096, miner operands, 4 rounds of 300 launches): on the 9070 XT both read +1.24, +1.34, +1.51%
  (ctl −0.08 to +0.19%). On the 9060 XT that timing was noisy (ctl +0.15, +1.75, +1.74%) and only the hashrate rounds
  count.
- **Clock stamps** (fold alone): a slab fell from 1888 to 1831 cycles on the 9070 XT (−3.0%) with the clock 2357 →
  2332 MHz, and from 1889 to 1831 on the 9060 XT (−3.1%) at 2700 → 2704 MHz. On both the wait at the barrier rose from
  622 to 721–722 cycles a slab while the rest fell from 1,266–1,267 to 1,109–1,110: the waves reach the barrier sooner and give
  part of that back waiting there.
- **Code** (ROCm 7.2.0): 213 VGPRs against 201, 24 KB of LDS, 7 waves by registers (6 run), no spills or scratch,
  the release's staging. On both boxes the default build's fold was byte for byte the measured flags' (md5
  `7706cb31`), and `-DPEARL_AMD_RDLATE=0 -DPEARL_AMD_RDTREE=0` gave the release's (`b6633aeb`). The Windows core (set
  off) is unchanged at 193 VGPRs and 36 KB.
- **Checks of the default build:** the fold test at k 2048, 4096 and 8192 on both chips; `verify-hits` 400 of 400 on
  the 9070 XT at k 8192 (130 salts, PC `c10c6949`) and on the 9060 XT at k 4096 (147 salts); Kryptex for 15 minutes,
  15 accepted and 0 rejected on the 9070 XT (all 15 decoded as valid k 8192 proofs, one connection) and
  7 accepted and 0 rejected on the 9060 XT (7 valid k 4096 proofs).
- **Left open:** RDLATE=1 with RDTREE read 0.24–0.31% more than the shipped =2 on the 9060 XT in every round, under the
  +0.3% median the plan asked for, on one PC, and never on a 9070 XT or a pool.
- **Cost:** a 9070 XT box for 74 minutes (10 of them a restart that went silent until `realloc`) and a 9060 XT box
  for 55, high priority.

RDNA 4's tile order (`PEARL_AMD_WALK=1`, on by default on gfx1200 and gfx1201 in the Linux core; issue #287, item 5),
7 November: supertiles about the size of what the card holds at once (4 x 12 tiles on an RX 9060 XT, 8 x 12 on an
RX 9070 XT), every other row of them backwards, against v0.6.13's order on the same PC, `hashrate.js`.

| Card, k | PC | `hashrate.js` | Fold alone |
|---|---|---|---|
| RX 9060 XT, 4096 | `9ac12ceb` | +0.38% | — |
| RX 9060 XT, 4096 | `ffa9f87b` | +1.85% | +1.91% |
| RX 9070 XT, 8192 | `c10c6949` | +0.56% | — |
| RX 9070 XT, 8192 | `1dba3ce9` | +0.75% | +0.57% |

- verify-hits checked 400 of 400, and Kryptex accepted 14 shares with none rejected.
- Dealing each supertile's blocks to one sub-tile per shader array (`=2`) read 0.04-0.20% above `=1`, under the bar.
- On an RX 7900 XTX (`45c94e40`) both were level (+0.06%, -0.04%), and the stamps put only 19% of blocks on array
  i % 12, so RDNA 3 keeps the old order. So does the Windows core, which mines k 2048.

### Our own builds on Salad, 7 November: RDNA 4's #287 items 4, 5 and 6 together

Each was raced alone against the release. Merged (exp/287-combined, 45b2c771), they were raced together on one PC per
card: base (all three off; its fold's md5 is the release's, b6633aeb, as main built on the same box), a no-op twin of
base, item 4 alone, items 4 and 5 (9060 XT only) and the merged default. `hashrate.js` 60 s after 20 s, order
rotating, round 1 not scored. Not rows: no competitor ran.

| Card, k | PC | Base (TH/s) | Twin | Item 4 | Items 4 + 5 | Items 4 + 5 + 6 |
|---|---|---|---|---|---|---|
| RX 9060 XT, 4096 | `85387649` | 70.63–71.31 | +0.06, −0.58, −0.07, −3.08, −0.55% | +1.71% (5 of 5) | **+2.20%** (5 of 5) | +1.71% (5 of 5) |
| RX 9070 XT, 8192 | `1dba3ce9` | 121.05–123.02 | −1.84, +0.03, −0.76, 0.00% | +0.92% (4 of 4) | **+1.66%** (4 of 4) | item 6 is off on gfx1201 |

- **Item 5 adds on top of item 4** on both: `hashrate.js` +0.49% (ahead in 4 of 5) and +0.74% (4 of 4); the fold alone
  +0.38% and +0.40% (5 of 5, 4 of 4).
- **Item 6 doesn't.** On top of items 4 and 5 it lost 0.47% on the 9060 XT (behind in 4 of 5; fold alone −0.39%, 0 of 5).
  On a second 9060 XT (`b1aeab8d`) the twin swung 1.4% and it read −0.12% by mean, +0.90% and +0.81% by median in
  the two settled rounds. So `PEARL_AMD_FUSE` is off by default; the gfx1200 that ships is the items 4 + 5 build.
- **Noise:** the 9060 XT's twin swung up to 3.1% from round 4 on, the 9070 XT's up to 1.8%. The bar, twice the twin's
  mean |delta|, was 1.73% and 1.31%: items 4 + 5 cleared it on both, items 4 + 5 + 6 missed by 0.02%.
- **Checks:** every build's fold bit-exact at 1024 and 4096 square (and k 8192 on the 9070 XT). The merged default:
  verify-hits 400 of 400 on both (on the 9060 XT with item 6 on), Kryptex 4 of 4 shares on the 9060 XT and 9 of 9 on
  the 9070 XT, none rejected, one connection each.
- **Cost:** three boxes, about 141 minutes, high priority.

v0.6.8's release tests (23 October), on two other RX 9070 XT PCs (`8081f47f`, `5e3d4671`): every checked hit verified
(400 of 3,589 and 400 of 3,703), 105.8 and 110.95 TH/s in `hashrate.js`, and 106.5 and 111.5 TH/s through the CLI with
`--core amd` for 300 s, 0 rejected. Another method and other PCs, so not rows.

### Our own builds on Salad, 10 November: NVIDIA ideas on AMD (PR #290 item 23)

Which of the NVIDIA miner's gains could carry to RDNA 2, 3 and 4. The desk review is in "Research notes" (NVIDIA work
on AMD). Two cheap ones were built behind switches and raced against the release on one PC each, `test/sweep.sh`:
the fold checked bit for bit first, then `hashrate.js` 60 s after 20 s, the builds taking turns, with a twin of the
release (an unused flag) for the noise. Not rows: no competitor ran.

- `PEARL_AMD_HASHQ`: the hash kernel, its hit count and the slot's event on a queue of their own, after the fold's
  event, so the next batch's fold follows this one with no hash between them (`pearl_host_hip.cpp`, Ctx). Each slot's
  fold waits for that slot's last hash first. The AMD form of the NVIDIA hash work: AMD's hash in the fold (#287 item 6)
  lost, and a stream per slot was level on an RX 9070 XT at k 2048 (23 October).
- `PEARL_AMD_RDTREE11`: RDNA 4's three XOR chains a region (`PEARL_AMD_RDTREE`) in RDNA 3's k 2048 readout.

| Card, k | PC | Release (TH/s) | Twin | `PEARL_AMD_HASHQ` | `PEARL_AMD_RDTREE11` |
|---|---|---|---|---|---|
| RX 7900 XTX, 2048 | `44d3716f` (run 1) | 51.39–51.64 | −0.15 to +0.06% | **+3.06 to +3.62%** (5 of 5) | −0.18 to −0.52% (5 of 5 behind) |
| RX 7900 XTX, 2048 | `44d3716f` (run 2, Salad gave the same PC) | 51.46–51.75 | −0.35 to +0.10% | **+2.99 to +3.71%** (6 of 6) | — |
| RX 7900 XTX, 2048 | `3bf70206` (4 rounds, then the PC restarted the container) | 55.97–56.72 | −0.13 to −0.29% | **+0.73 to +1.00%** (4 of 4) | — |
| RX 7900 XTX, 2048 | `3bf70206` (6 rounds) | 55.09–56.30 | −0.76 to +0.25% | **+0.21 to +0.62%**, first round +1.92% (6 of 6) | — |
| RX 7800 XT, 2048 | `72dd708c` | 32.62–32.66 | 0.00 to −0.03% | −0.06 to +0.03% | — |
| RX 7800 XT, 2048 | `16f1685c` | 32.84–33.01 | −0.18 to 0.00% | −0.03 to −0.27% (6 of 6 behind) | — |
| RX 9060 XT, 4096 | `8c96a2a7` | 73.14–73.23 | −0.03 to +0.04% | −0.21 to −0.36% (6 of 6 behind) | (gfx12 has RDTREE) |

- **The XTX gain is in the fold, not the hash.** `PEARL_AMD_TRACE` on `44d3716f`: the release's batch is 10.67 ms, the fold
  10.57 and the hash 0.084. With the hash on its own queue the batch is 10.33 ms and the fold itself 10.29–10.31 ms.
  That is 0.26 ms a batch from the fold, three times the hash it no longer waits for. On both 7800 XTs and the 9060 XT
  the fold got 0.1 ms slower with the hash beside it, which ate what the hash saved. This XTX runs the release at
  51.5 TH/s where item 7's XTX ran 54.4, so the size of the gain is this PC's, not the card's. The second XTX (`3bf70206`, 56.2 TH/s on the release) gained
  +0.2 to +1.0%: ahead of the release and of the twin in all 10 rounds, but its fold didn't get faster (9.68 ms either
  way); the batch lost about the hash's 0.08 ms and no more.
- **Checks.** Every build's fold bit-exact. verify-hits with the hash on its own queue: 400 of 400 on the XTX twice
  (3,646 and 3,655 hits, 8.0 a salt) and 400 of 400 on the second XTX (3,836 hits, 480 salts), 0 failed. On Kryptex the XTX got 6 accepted shares and 0 rejected in 10 minutes
  at 53.3 TH/s.
- **RDTREE11 loses** in the fold alone too (−0.25 to −0.59% a round, twin ±0.3%). RDNA 3's readout is a quarter of
  RDNA 4's share of a chunk (its WMMA is a quarter as fast), and the three chains' extra live registers cost more.
  Dropped; not in the code.
- **Shipped:** `PEARL_AMD_HASHQ` is on for gfx1100 (RX 7900) only: the default is 2. Two XTX PCs gained (+3% and
  +0.2 to +1.0%), 21 of 21 rounds ahead. No RX 7900 XT or GRE has run it. gfx1101 (RX 7800 XT, 7700 XT) and RDNA 4
  stay off. `-DPEARL_AMD_HASHQ=1` turns it on for every card, `=0` for none, and the environment variable `PEARL_AMD_HASHQ=1` (or `0`) overrides the build at run time.
- **Cost:** five boxes (one moved by Salad mid-run), about 4.5 box-hours, high priority.

### Salad, 10–11 November: the RX 7900 against SRBMiner again, and what a slow PC is (PR #290, AMD goal)

The 9–10 November rows put the RX 7900 XTX at 92% and 94% of SRBMiner. Since then the hash kernel moved to a queue of
its own on gfx1100 (`PEARL_AMD_HASHQ`, item 23). This is the duel with that on, on two XTX PCs (one of them the 94%
row's own PC) and one RX 7900 XT, and what a PC where our rate reads low is doing. Not rows: the method differs from
the table's (see "Keeping this file current").

**Method.** One box a PC (`amdtest.py`, the ROCm image). Ours is the PR head's AMD core (5d2b012e, the default build,
the same binary on all three PCs) through the app's CLI; both miners mine on Kryptex. Ours warms the card for 60 s,
then the miners take turns of 5 minutes, two each. Every turn is scored the same way: the mean of its 24 ten-second
samples after the first minute (the CLI's last printed rate, SRBMiner's API rate). Then one turn each of 3 minutes with
a clock probe beside the miner: one wave from another process that reads SHADER_CYCLES over REALTIME for a millisecond
or two every 10 s.

| Card | PC | SRBMiner | Ours, turns 1 and 2 (TH/s) | SRBMiner, turns 1 and 2 | Ours / SRBMiner | Clock under ours | Clock under SRBMiner |
|---|---|---|---|---|---|---|---|
| RX 7900 XTX | `45c94e40` | 3.8.2 | 57.06, 56.91 | 55.59, 55.50 | **102.6%** | 2581 MHz (2434–2667) | 2588 MHz (2561–2646) |
| RX 7900 XTX | `ae456453` (the 94% row's PC) | 3.8.6 | 55.04, 54.88 | 53.87, 54.07 | **101.8%** | no reading (see below) | 2494 MHz (2463–2531) |
| RX 7900 XT | `78024435` | 3.8.6 | 46.16, 46.48 | 46.27, 46.11 | **100.3%** | 2473 MHz (2394–2553) | 2424 MHz (2396–2456) |

- Both XTX PCs are above 100% now. `45c94e40` read 53.1 against 53.9 (99%) on 30 October, and `ae456453` 50.8 against
  54.1 on 9–10 November. The XT is level: 0.3% is inside the spread of our own two turns (0.7%).
- The CLI's own mean over every line it printed after the first minute, one a second: 57.10 and 57.17, 55.03 and 54.81,
  46.27 and 46.35. SRBMiner's own 15-minute average at each turn's end: 55.76 and 55.51, 54.07 and 53.99, 46.12 and
  46.19.
- **SRBMiner 3.8.6 runs in the ROCm image once `wget` is installed.** Without it 3.8.6 printed nothing and exited
  (given a terminal: "Make sure you have 'wget' installed on your system!"), which is what stopped it on Salad on 9
  November. One more turn of 3.8.2 on `ae456453` read 53.68, so the two versions are level on this card.
- **Clock.** On `45c94e40` the clock is the same under both miners, so we do 2.6% more per clock there. On the XT ours
  runs 2% more clock for the same rate, so SRBMiner does 1.7% more per clock there. `ae456453` holds 3.6% less clock
  than `45c94e40` (2494 MHz against 2588 under SRBMiner), and both miners read 3 to 4% lower on it.
- **No clock under ours on `ae456453`:** in that 3-minute turn ours read 50.21 TH/s, not 55.0, and the probe's readings
  ran from 1395 to 2809 MHz (median 2664). That is the rate the PC gives a queue it keeps stopping (below). SRBMiner
  beside the same probe read 54.41 by its own average. Twenty minutes later `hashrate.js` with the same probe beside it
  was level with the default (54.93–55.21), so it wasn't the probe alone: the PC has spells, and the hash queue doesn't
  keep our miner out of all of them. A 10-minute watch of ours alone, to count them, got 2 minutes in (55.0, 54.9)
  before Salad took the PC away. With the probe the other two turns read 56.93 (57.0 without) and 45.83 (46.3).
- Shares: every one accepted, but one of ours in turn 1 on `45c94e40` (the reason wasn't kept).

**What a slow PC is.** `ae456453` is one: the fold alone (the fold test's launches back to back on one queue) reads
52.5–53.6 TMAC/s there against 58.6–59.1 on `45c94e40`. That is three times what its lower clock explains, and it is
not the fold's cycles either:

| Fold alone, one queue | `45c94e40` (fast) | `ae456453` (slow) |
|---|---|---|
| The miner's launch, TMAC/s (3 rounds) | 59.1, 59.0, 58.6 | 53.6, 52.6, 52.5 |
| 2 batches in one launch (taller, so B' stays one batch's) | 58.6, 58.8, 58.9 | 52.9, 52.7, 52.8 |
| 4 batches in one launch | 59.2, 59.3, 58.8 | 52.8, 52.5, 52.8 |
| 16 tiles wide (batch 256) | 57.4, 57.9, 58.0 | 52.3, 52.0, 52.0 |
| 32 tiles wide (batch 512) | 55.9, 56.2, 56.4 | 51.1, 51.0, 51.0 |
| Clock in these 4 s runs, from a cool card (stamps) | 2707–2711 MHz | 2640–2782 MHz |
| Cycles a slab | 6,711 | 6,707–6,717 |
| Blocks whose waves changed SIMD mid-run, a launch | 0 of 8,192 | 115–119 |

On the slow PC **the whole queue stops about 30 times a second, for 3.6–4.0 ms each time.** In one 171 ms launch (64
batches wide, in strips) no block started or ended for 3.77, 3.66, 3.63, 3.64, 3.62 and 3.69 ms, at 3.9, 37.2, 67.1,
103.8, 137.2 and 161.5 ms. The 107–126 blocks running at each stop came back on other SIMDs (their HW_ID1 changed) and
took 3.96 ms where every other block took 0.163. Between stops the card is full and the fold runs as on the fast PC.
3.65 ms in every 33.3 is 11%. So something on that PC preempts a compute queue that stays busy, and the fold has about
1,150 waves to put away and bring back each time, which is probably where the 3.6 ms go. A launch longer than the 33 ms
between stops is always hit, so longer launches can't cure it; on the fast PC they gain nothing.

**The stops come while the miner has one busy queue.** A stamp build of the core can write the stamps of one of the
miner's own launches (`PEARL_AMD_STAMP_DUMP`, on branch `goal/amd`). On `ae456453`, launches of about 10 ms, in two
sittings an hour apart:

| `hashrate.js` on the stamp build | Launches stopped | The stops |
|---|---|---|
| Hash on the fold's queue (v0.6.13's way, `PEARL_AMD_HASHQ=0`) | 5 of 13 | 3.72–4.02 ms; 680–1,152 waves moved; the launch then spans 13.4–13.9 ms, not 9.5–10.0 |
| Hash on its own queue (the PR's default) | 0 of 13 | |
| Hash on the fold's queue, and a one-thread kernel on a third queue every 2 ms (`PEARL_AMD_TICK_US=2000`) | 0 of 8 | |
| The same every 20 ms | 1 of 5 | 3.57 ms |

Five of thirteen is about what 30 stops a second give a 10 ms launch. A second queue of the miner's with work on it,
the hash's or a ticker's, ends the stops.

**`hashrate.js` on `ae456453`**, each setting in turn, order rotating, 45 s after 12 s, scored from round 2, in three
runs of 4, 3 and 3 rounds. The control is the PR's default; the twin is a second copy of it.

| Setting | TH/s | Against the default |
|---|---|---|
| The PR's default (hash on its own queue) | 54.92–54.98, 55.03, 55.01–55.03 | twin +0.05 to +0.44%, −0.05 to +0.02%, −0.07 to −0.09% |
| Hash on the fold's queue (`PEARL_AMD_HASHQ=0`, v0.6.13's way) | 50.45–50.50 | **−8.05 to −8.24%** |
| Hash on the fold's queue, the core's ticker every 2 ms | 55.07–55.16 | +0.07 to +0.24% |
| The same, every 20 ms | 54.71–54.75 | −0.51 to −0.55% |
| The same, every 50 ms | 51.88–51.96 | −5.54 to −5.72% |
| The default with the core's ticker every 2 ms, every 0.5 ms | 55.09–55.15, 55.11 | +0.11 to +0.22%, +0.15% |
| The hit count zeroed on the hash queue, not before the fold | 54.99–55.10 | +0.09 to +0.29% |
| That, and no wait for the slot's last hash on the fold's queue | 54.92–55.07 | −0.11 to +0.27% |
| 2 batches a launch (20 ms), as strips | 54.89–54.91 | −0.22% |
| 4 batches a launch (41 ms), as strips | 51.51–51.52 | **−6.19 to −6.31%** |
| 4 batches a launch, the core's ticker every 2 ms | 51.50–51.54 | −6.34 to −6.41% |
| Batch 64 (a 5 ms launch) | 54.26–54.41 | −1.13 to −1.40% |
| The default beside another process's kernel every 2 ms | 54.24–54.26 | −1.40 to −1.44% |
| The default beside another process's kernel every 10 s | 54.93–55.21 | −0.18 to +0.36% |

- **The hash queue is worth +8.9% on the slow PC** (50.5 to 55.0), and 50.5 is the benchmark row's 50.8. On the fast PC
  one round read hash queue off −0.7% (56.73 against 57.15, twin 57.21), batch 256 −1.6% and batch 512 −3.3%. Item 23
  found the same two sizes on two other PCs (+3.0 to +3.7%, +0.2 to +1.0%). So the gain is the PC's, and it is the
  stops. What is left between the two PCs is their clock.
- **Longer launches lose** on the slow PC, because a 41 ms launch is always stopped, and the fold alone gains nothing
  from them on either XTX PC. The review's lead (several 128-offset batches a launch) isn't worth building as a default.
  The test builds are on branch `goal/amd`: `-DPEARL_AMD_STRIP=1` and `PEARL_AMD_STRIPS=n` launch n batches as strips
  of the grid's z, so B' in use stays one batch's.
- Nothing raced beats the PR's default on this PC by more than its twin.
- With `PEARL_AMD_TRACE=1` and the probe every 5 s the default read 50.97–51.09 there. Race on such a PC with the trace
  off.

**RX 7900 XT: the hash queue is right there too.** It is on for every gfx1100 card and no XT had raced it. One PC
(`78024435`), `hashrate.js` 45 s after 12 s, 6 rounds, order rotating:

| RX 7900 XT `78024435` | TH/s, rounds 1–6 | Against the default, rounds 2–6 |
|---|---|---|
| The PR's default (hash on its own queue) | 46.74, 46.36, 46.26, 46.26, 46.24, 45.79 | |
| Twin | 46.00, 46.33, 46.18, 46.38, 44.72, 46.23 | −0.06, −0.17, +0.26, −3.29, +0.96% |
| Hash on the fold's queue (`PEARL_AMD_HASHQ=0`) | 44.73, 44.59, 46.03, 45.96, 44.61, 44.76 | −3.82, −0.50, −0.65, −3.53, −2.25% |
| Hash on the fold's queue, the core's ticker every 2 ms | 46.45, 45.91, 46.41, 46.33, 46.36, 46.44 | −0.97, +0.32, +0.15, +0.26, +1.42% |

- Off is behind in 5 of 5 rounds: by 0.5–0.65% when the PC is steady and by 2.3–3.8% in the three rounds where it read
  44.6–44.8. The twin read 44.72 once. So this PC has a slow state too, 3.5% down, which hit the off setting in 4 runs
  of 6 and the default in 1 of 12. On is the right default for the XT.
- Its stamps caught no stop: fold alone 46.75 TMAC/s at 2548 MHz, 6,708 cycles a slab, 126 blocks at once, no wave
  moved in a 182 ms launch, nor in 12 of the miner's own launches (6 with the hash queue on, 6 off). What the slow state
  is on this PC is not known.
- Longer launches gain nothing on the XT either. Fold alone, 3 rounds: 47.8, 47.4, 47.6 TMAC/s at 1 batch a launch,
  47.8, 47.8, 47.8 at 2, and 48.2, 47.6, 47.7 at 4. In the miner, 3 rounds (twin −0.04 to −0.11%): 2 batches a launch
  −0.37 to −0.45%, 4 batches −0.54 to −0.65%.

- **Not known:** what on `ae456453` preempts the queue (another program on that PC's GPU, or its driver's scheduler),
  how many PCs do it, how long the spells last that reach our miner with the hash queue on, and whether the RX 7800 XT
  and RDNA 4, which keep the hash on the fold's queue, are stopped the same way on such a PC. The 9–10 November RX 9060
  XT row `3540c3e2` has the same shape: ours 15% under our other PC, SRBMiner 1.6% under. Its launch is 31 ms and the
  RX 9070 XT's 34 ms, as long as the gap between stops here.
- **Cost:** two boxes over four PCs (one of them with no usable GPU), about 4.5 box-hours, high priority.

### Salad, 11 November: the slow PC's queue stops on RDNA 4 and the RX 7800 XT (PR #290, AMD goal)

The entry above found that a slow RX 7900 XTX PC stops a busy compute queue 30 times a second, and that the hash on
its own queue ends the stops on gfx1100. gfx1101 (RX 7800 XT) and RDNA 4 still hash on the fold's queue. This asks
whether the stops cost us there too, and what ends them. Not rows.

**Method.** As above: one box a PC, the fold alone with clock stamps first (is the PC a slow one?), then `hashrate.js`
45 s after 12 s, the builds taking turns, with a twin for the noise, scored from round 2 (the first run on a cool
RX 9060 XT read 1 to 3% high). Then stamps of the miner's own launches, 6 or 8 of each setting: was the queue
stopped in that launch (a gap of 0.3 ms or more with no block starting or ending)? The run's CPU time is bash's
`time` on the whole `hashrate.js` process.

**RX 9060 XT, `3540c3e2`** (the 111% row's PC; k 4096, 256 column offsets a launch)

**It is a slow PC, with the same stops.** The fold alone at the miner's launch read 57.4 and 57.6 TMAC/s at 2561 and
2528 MHz; fast RX 9060 XT PCs mine at 70.0 to 73.2 TH/s. A launch is 38 ms here. In one 155 ms launch the queue was
stopped 5 times for 2.08 to 2.09 ms, 33.46 ms apart, and 181 to 251 waves came back on another SIMD each time. That
is the RX 7900 XTX PC's pattern with a shorter stop (2.1 ms, not 3.6 to 4.0).

| Setting | TH/s, rounds 2 on | Against the default | Launches stopped | CPU time of a 57 s run |
|---|---|---|---|---|
| Default (hash on the fold's queue) | 55.28–55.51 (9 rounds) | — | 8 of 8 | 229 s |
| Twin | 55.26–55.56 | −0.31 to +0.31% (9 rounds) | — | 229 s |
| `PEARL_AMD_HASHQ=1` (hash on its own queue) | 55.19–55.28 | −0.23, −0.04, −0.27% | 5 of 6 | — |
| 128 offsets a launch (19–22 ms) | 54.81–54.98 | −0.61, −0.65, −1.12% | 4 of 6 | — |
| 128 offsets and the hash queue | 54.93–55.01 | −0.70, −0.62, −0.76% | — | — |
| Ticker every 2 ms (`PEARL_AMD_TICK_US=2000`) | 56.76–57.43 | **+2.44 to +3.68%** (7 of 7) | 2 of 8 | 273–281 s |
| Ticker every 0.5 ms | 56.96–57.12 | +3.10, +2.80% | — | 308–309 s |
| Ticker every 10 ms | 56.03–56.66 | +1.14, +2.26% | 4 of 6 | 239–241 s |
| Ticker every 2 ms that does not wait (`PEARL_AMD_TICK_WAIT=0`) | 56.92–57.42 | **+2.54 to +3.66%** (4 of 4) | 0 of 6 | 250–258 s |
| That every 0.5 ms | 55.24–57.30 | +3.45, −0.31, +2.50, −0.45% | 3 of 4 | 272–278 s, or 237 s |
| That every 5 ms | 55.47–57.02 | +2.72, −0.04% | — | 252 s, or 230 s |

- **The hash queue does not end the stops on RDNA 4.** It was level in the rate (round 1 read −1.11%, the default
  running first on the cool card) and 5 of 6 launches were still stopped. A launch here is 38 ms, longer than the
  33 ms between stops, so every launch meets one, and the hash queue only gets work once a launch. Halving the launch
  did not help either: 4 of 6 launches stopped, and the shorter launch costs 0.6 to 1.1% by itself.
- **The ticker does:** a thread that launches a one-thread kernel on a queue of its own every 2 ms
  (`PEARL_AMD_TICK_US`, branch `goal/amd`). +2.4 to +3.7% in 7 of 7 rounds over three races. Every 0.5 ms is the
  same; every 10 ms gets a third to two thirds of it and leaves stops in 4 of 6 launches.
- **It costs CPU.** The ticker's thread waits for each tick's kernel, and HIP's wait spins. The default run takes
  229 s of CPU in 57 s (four cores busy before any ticker); the ticker every 2 ms takes 44 to 52 s more, 0.8 to 0.9
  of a core, every 0.5 ms 1.4 cores, every 10 ms 0.2.
- **A ticker that does not wait** (`PEARL_AMD_TICK_WAIT=0`: it records an event behind its kernel and skips a tick
  while the last one has not run) kept the gain every 2 ms at half the CPU cost, 0.4 to 0.5 of a core, and none of 6
  stamped launches was stopped. But it is not dependable at other intervals: every 0.5 ms 3 runs of 6 read the
  default's rate, and every 5 ms 1 of 3. Those runs took almost no extra CPU time either (237 s and 230 s), and a
  guard that waits for a tick not seen done in 50 ms never fired in them, so the ticks ran and did nothing. Why is
  not known. **Every run that gained took more CPU time (11 to 80 s more), and the four that took almost none (1 to
  8 s) gained nothing.**
- **The stops are the smaller part of what this PC loses.** With the ticker the stamped launches still ran only 20
  to 32 of the card's 48 blocks for about 4 ms every 17 ms: all 48 at once for 66 to 81% of a launch (8 launches;
  61% by default, 2 launches). That looks like another program drawing 60 frames a second. With a clock 5% under a
  fast PC's (2530–2560 MHz against about 2700) that is most of the 20% this PC is short of a fast one. Nothing of
  ours got it back.
- **Not a default.** `PEARL_AMD_TICK_US` stays 0 on every card. A default needs a fast RX 9060 XT PC to show the
  ticker is not behind there, an RX 9070 XT, the hit check, and a decision on the CPU it takes: a ticker on every
  RDNA 4 PC would spend half a core or more where there are no stops to end.

**Duel on this PC**, as in the entry above: SRBMiner 3.8.6, ours the PR head's core (5d2b012e, the default build), and
the same core with the ticker every 2 ms; 5-minute turns, two each, each scored by its 24 ten-second samples after the
first minute.

| | Turn 1 (TH/s) | Turn 2 | Against SRBMiner | Clock, probe beside a 3-minute turn |
|---|---|---|---|---|
| Ours, default | 55.34 | 55.37 | **112.4%** | 2550 MHz (2479–2569) |
| Ours, ticker every 2 ms | 57.10 | 57.19 | **116.0%** | not read |
| SRBMiner 3.8.6 | 49.25 | 49.24 | — | 2130 MHz (2057–2191) |

- The row says 59.7 against 54.0 for this PC on 9–10 November. Both miners are lower now (ours 7%, SRBMiner 9%), so
  the PC is doing more of something else than it was, and SRBMiner loses to it too.
- SRBMiner's own 15-minute average at its turns' ends: 49.30 and 49.19. All shares accepted: ours 3, with the ticker
  6, SRBMiner 3.
- The probe read 16% less clock under SRBMiner than under ours, in all 13 readings. On the RX 7900s the two were
  within 2%. Whether SRBMiner runs this card at a lower clock or the probe's one wave shares a SIMD with SRBMiner's
  is not known.

**RX 7800 XT** (k 2048, 128 column offsets a launch)

One box, three PCs in turn.

| PC | Fold alone at the miner's launch | Clock | Stops in one long launch | Default (TH/s) |
|---|---|---|---|---|
| `56ae1be5` | 34.51 TMAC/s, launch 15.9 ms | 2525 MHz | one of 1.05 ms in 128 ms | 34.25 (hash queue 34.30) |
| `16f1685c` | the PC could not fetch the repo; nothing ran | — | — | — |
| `559a11d6` (a 103% row's PC) | 29.66 TMAC/s, launch 18.5 ms | 2309 MHz | 4 of 3.12–3.24 ms in 150 ms, 42.3 ms apart | 29.52–29.55 |

`56ae1be5` is a fast PC, so the question has no answer there. `559a11d6` is a slow one: 9% less clock, and stops 24 times
a second with 520 to 602 waves coming back on another SIMD each time. The fold test's launches also sit 1.1 ms apart
there (none on the fast PC). The race ran on `559a11d6`, 4 rounds:

| Setting | TH/s, rounds 2 to 4 | Against the default | Stamped launches, 6 each | CPU time of a 57 s run |
|---|---|---|---|---|
| Default (hash on the fold's queue) | 29.54, 29.55, 29.52 | — | 1 stopped (3.27 ms) | 228 s |
| Twin | 29.55, 29.54, 29.53 | +0.03, −0.03, +0.03% | — | 228 s |
| `PEARL_AMD_HASHQ=1` | 29.42, 29.39, 29.52 | −0.41, −0.54, 0.00% | 6 begin with a stall of 1.3–1.7 ms | 285 s |
| Ticker every 2 ms | 28.33, 28.34, 28.33 | −4.10, −4.09, −4.03% | 6 have one or two stalls of 1.2–1.6 ms | 322 s |
| Ticker every 2 ms that does not wait | 28.35, 28.35, 28.33 | −4.03, −4.06, −4.03% | — | 281 s |

- **The hash queue does not win on a slow RX 7800 XT.** Level or behind in every round (round 1 read −0.30%). On this
  PC work on a second queue stops the fold's queue itself: with the hash queue every launch begins with a stall of
  1.3 to 1.7 ms, and 60 to 90 of the 90 blocks then running come back on another SIMD. The ticker's stalls come
  12 ms apart, and it loses 4%. So a second queue here trades the PC's own stop (one launch of 6) for a shorter one
  in every launch.
- **gfx1101 keeps the hash on the fold's queue.** It was level or 0.03 to 0.27% behind on two fast PCs on 10
  November, and it is level or behind on this slow one.
- **The hash queue takes one more CPU core.** 285 s of CPU against 228 s in a 57 s run, the same on both PCs. `ps -L`
  shows why: the miner's process runs 4 threads at 100% by default, 5 with the hash queue, 4 and two at 82% with the
  ticker, 4 and one at 91% with the ticker that does not wait. That is one busy thread for each HIP stream with work
  on it. The hash queue has been the default on gfx1100 since 10 November, and its CPU time was not measured there.

**What the three slow PCs say**

| Slow PC | Stops in the fold alone | A second queue with work on it |
|---|---|---|
| RX 7900 XTX `ae456453` | 3.6–4.0 ms, 30 a second | ends them: hash queue +8.9% |
| RX 9060 XT `3540c3e2` | 2.1 ms, 30 a second | ends them only if it has work every 2 ms: hash queue level, ticker +2.4 to +3.7% |
| RX 7800 XT `559a11d6` | 3.2 ms, 24 a second | stops the fold itself: hash queue 0 to −0.5%, ticker −4% |

- A slow PC is not one thing, and no setting is right on all three. No default changed. The ticker is a switch on
  branch `goal/amd` (`PEARL_AMD_TICK_US`, `PEARL_AMD_TICK_WAIT`), off on every card, not for merging as it is.
- Every box here is a WSL2 container on someone's Windows PC. Whether a miner run from Windows on these PCs is
  stopped the same way is not known.
- Not known either: why SRBMiner loses less than we do on the slow RX 9060 XT PC (its launch pattern was not
  measured); a slow RX 9070 XT; the ticker on a fast RDNA 4 PC; what the four busy threads of the default are for,
  and whether a blocking wait would free them.

### Salad, 11 November: what the hash queue costs in CPU on an RX 7900 XTX (PR #290, AMD goal)

The entry above found the hash on its own queue taking one more CPU core on the RX 7800 XT. It is the default on
gfx1100, so this is the same measurement there. One box, RX 7900 XTX `6f96bc89`, branch `goal/amd` (aff96401, then
8389f9d4; the default path is the PR's). `hashrate.js` 45 s after 12 s, the settings taking turns, two races of 3
rounds, every round shown. The CPU time is bash's `time` on the whole 57 s run, and the threads are `ps -L`.

**The PC is a fast one:** the fold alone at the miner's launch read 56.8 TMAC/s at 2635 MHz, and an 81 ms launch had
no queue stop and no wave moved.

| Race | Setting | TH/s, rounds 1 to 3 | Against the hash queue off | CPU time | Threads at 100% |
|---|---|---|---|---|---|
| 1 | `PEARL_AMD_HASHQ=0` | 54.16, 54.17, 54.03 | — | 228 s | 4 |
| 1 | `PEARL_AMD_HASHQ=1` | 54.59, 54.74, 54.65 | +0.79, +1.05, +1.15% | 285 s | 5 |
| 1 | Default (the hash queue, on gfx1100) | 54.50, 54.56, 54.43 | +0.63, +0.72, +0.74% | 285 s | — |
| 2 | `PEARL_AMD_HASHQ=0` | 54.10, 54.15, 54.28 | — | 228 s | 4 |
| 2 | Default | 54.58, 54.51, 54.73 | +0.89, +0.66, +0.83% | 285 s | 5 |
| 2 | Default, HIP's blocking wait | 54.29, 54.41, 54.58 | +0.35, +0.48, +0.55% | 285 s | 5 |
| 2 | Hash queue off, HIP's blocking wait | 53.93, 53.93, 54.02 | −0.31, −0.41, −0.48% | 228 s | 4 |

- **The hash queue takes one more CPU core on the RX 7900 XTX too:** 285 s of CPU against 228 s, the same in every
  run, for +0.6 to +1.2% on this fast PC (it was +8.9% on the slow PC `ae456453`). No twin ran: the default and
  `PEARL_AMD_HASHQ=1` are one setting on this card and read 0.2 to 0.4% apart in race 1, so that is the noise.
- **HIP's blocking wait frees no thread.** `PEARL_AMD_SCHED=block` (an environment switch on `goal/amd`) calls
  `hipSetDeviceFlags(hipDeviceScheduleBlockingSync)` before the first allocation. In ROCm 7.2's source that turns the
  device's active wait off, so a wait spins for 100 µs and then blocks (`hip_device_runtime.cpp`, and `WaitForSignal`
  in `rocvirtual.hpp`). The runtime took the call, the CPU time and the busy threads did not change, and the rate read
  0.2 to 0.5% lower.
- **What the busy threads are** (gdb, the call chain of each): the miner's own worker, waiting for a slot's event in
  `pearl_host_collect`, inside HIP and then libhsa-runtime64; one `wsl::thunk::ComputeQueue::AqlToPm4Thread` in
  `librocdxg.so` for each compute queue, a loop around `steady_clock::now()`; and two threads wholly inside
  libhsa-runtime64. The fifth thread the hash queue adds is its own queue's `AqlToPm4Thread`.
- **This is WSL2's cost.** `librocdxg.so` is ROCm's layer over Windows' GPU driver in a WSL2 container, which is
  what a Salad box is. The Windows installer's core runs on AMD's HIP SDK instead, where none of this was
  measured, and no Windows Radeon has run that core yet.
- No default changed. Not known: what the two libhsa-runtime64 threads are for, and whether any setting of the
  runtime's lets the three threads that are not ours sleep.

### Salad, 11 November: the RX 7900 rows re-run (v0.6.16 against SRBMiner 3.8.6)

The two RX 7900 XTX rows read 92% and 94% on 9–10 November, on v0.6.13. v0.6.16 runs the hash kernel on a queue of
its own on these cards (`PEARL_AMD_HASHQ`), which read 102% in a duel that was not a table run. This is the table
run: `bench.py`, both miners on the row's own PC, the same lengths as 9–10 November (60 s warm-up, 5 minutes each,
a 10-minute hit check first, Kryptex). The two RX 7900 XT rows were re-run the same way, as the nearest to 100%. All
four boxes landed on the old rows' PCs, so the four rows above are replaced and no host is new.

| Card | Machine ID | Ran first | Ours | SRBMiner 3.8.6 | % of best | Ours: mean (lowest–highest) | SRBMiner: mean (lowest–highest) | Mean over mean | On 9–10 Nov |
|---|---|---|---|---|---|---|---|---|---|
| RX 7900 XTX | 6f96bc89 | SRBMiner | 54.3 | 53.8 | **101%** | 54.31 (53.9–55.7) | 53.80 (53.19–54.11) | 100.9% | 49.8 / 54.0, 92% |
| RX 7900 XTX | ae456453 | ours | 55.3 | 54.1 | **102%** | 55.28 (54.9–56.8) | 54.34 (52.99–54.81) | 101.7% | 50.8 / 54.1, 94% |
| RX 7900 XT | dc8fcef5 | SRBMiner | 45.9 | 43.0 | **107%** | 45.88 (45.0–47.7) | 43.04 (42.91–43.28) | 106.6% | 46.3 / 45.4, 102% |
| RX 7900 XT | 1d5cab04 | ours | 48.5 | 47.6 | **102%** | 48.47 (48.2–48.7) | 47.65 (46.00–48.29) | 101.7% | 48.4 / 47.7, 101% |

"Ours" is the mean of our samples after the first minute and "SRBMiner" its own last figure, as in every table
here. The means are of each miner's 17 samples after the first minute, one every 15 s. `bench.py` now keeps every
sample with the result, so the rows can be scored again another way.

- **What ran as ours:** v0.6.16's source (commit bffee00a, the release tag), built on each box for gfx1100 by
  `bench.sh` and run through the CLI from source. It is not the installer's own `pearl_core_hip.node` file. The
  9–10 November run built v0.6.13's source the same way.
- **SRBMiner is 3.8.6, not the 3.8.2 of the other six rows.** `bench.sh` pins 3.8.2, so the run's driver swapped
  that one line on the box for 3.8.6 and its published MD5 (38128fa1…) and installed `wget`, which 3.8.6 needs. The
  "ready" line on all four boxes says SRBMiner 3.8.6. The two versions read level on `ae456453` on 10 November
  (53.68 against 53.87 and 54.07).
- **Both RX 7900 XTX rows are above 100% now, by a little.** `6f96bc89` is 1.0% ahead, which is inside what one
  5-minute run moves by (our own samples there spread 0.4 TH/s). Call that row level or just ahead.
- **Not all of the change on `6f96bc89` is the new code.** Its 49.8 of 9–10 November was a low reading: v0.6.13's
  kernel read 54.37 on the same PC a day later (99.9%, above). On `ae456453` the old code read low because that PC
  stops a busy queue 30 times a second, which the hash queue ends (+8.9% there).
- **`dc8fcef5`'s 107% is the PC, in part.** SRBMiner held 43.0 for its whole turn (42.9–43.3), 5% under its 45.4 of
  9–10 November. Ours then read 45.0 for 2.5 minutes and 46.4 after. So the PC was slower in SRBMiner's turn and
  the first half of ours than in the second half. Read the row as above 100%, not as a 5-point gain.
- **Hit checks:** all four PASS, 400 hits verified, 0 failed, 8.00 to 8.02 hits a salt (3,482 hits in 435 salts,
  3,776 in 472, 3,076 in 384, 3,320 in 414, in the table's order). Copy rates 755, 747, 622 and 627 GB/s.
- **No clock, power or temperature:** Salad's `amd-smi` gave no reading on any box, as on 9–10 November.
- **PCs passed over.** Each lane first moved off PCs that were not an old row's (up to 4 moves a class: XTX
  `a648c3ee`, `45c94e40`, `fd4ebc44`, `28a12f15`; XT `78024435`, `fa211c2d`, `4b1d9a8b`, `3f29b5c7`); none of those
  ran a miner. One XT PC, `8e4144bb`, passed the hit check (400 of 2,947) but neither miner got work from Kryptex
  there (SRBMiner 0.0 for 5 minutes, ours still "connecting" at the end), so `bench.py` did not record it and moved
  the box on, to `56de7dc2`. Salad then moved it by itself, before it had set up, to `dc8fcef5`.
- **Time and cost:** 21:33 to 23:08 UTC, four groups (`amdrerun-rx7900xtx-1` and `-2`, `amdrerun-rx7900xt-1` and
  `-2`), two at a time, 154 group-minutes in all. About $0.50 at Salad's high-priority prices ($0.168 an hour for
  the XTX class, $0.156 for the XT, plus 4 vCPUs and 8 GB).

## Intel Arc (Alchemist, on a rented rig)

The first Intel hardware, 10 November: a rig of six Arc A750 cards. Ours is the SYCL core in `earn/native/sycl`
(the `xmx8` fold: the XMX matrix engines through DPAS at sub-group 8), built on the rig from PR #290 and run by
that branch's CLI through `PEARL_CORE_PATH`. It is in no release. The competitors are IAminer, the Pearl miner
for Arc that could be had, and the two we race elsewhere (SRBMiner, WildRig). TH/s is each miner's own figure.

| Card | Host | Rig | Ours | IAminer | SRBMiner | WildRig | % of best |
|---|---|---|---|---|---|---|---|
| Arc A750 | HiveOS rig (8 GB) | arc750, card 0 | 105.3 | 13.6 | 0 | 0 | 774.3% |
| Arc A750 x 6 | HiveOS rig (8 GB each) | arc750, all six | 100.1 | 78.0 | 0 | 0 | 128% |
| Arc Pro B50 | Dev box (73 W, 16 GB) | devb50 | 46.5 | not run | not run | not run | — |

Updated 16 November. Card 0's "Ours" is now the kernel v0.7.2 shipped and v0.7.3 keeps: 105.2-105.3 TH/s at 190 W
in the final race on that card (13 November, goal/arc-31, below). The rig is no longer rented, so IAminer keeps its
10 November figure on the same card; the founder judged that good enough with ours this far ahead. The six-card row
is still the 10 November PR #290 build. The Arc Pro B50 (Battlemage) is v0.7.3's own figure on the founder's dev
box, 46.5 TH/s at its 73 W limit with every share accepted on Kryptex; no competitor has been raced on it yet.

- **IAminer 1.0.0** mines on the card. Its own tuning picked m = n = 131072, k 4096. The pool rejected the only two
  shares it found ("Invalid share"), so it has no accepted share on this rig.
- **SRBMiner 3.8.6 and 3.8.2** list the six cards but cannot mine Pearl on them: "No GPU devices available to mine
  with". Their algorithm list marks `pearlhash` for AMD and NVIDIA only.
- **WildRig 0.52.1 and 0.51.9** list the card, print "Start mining", then show "n/a TH/s" for four minutes with the
  card idle at 34 W.
- **ARCMiner** has no column: it could not be had. Kryptex's miner page lists it as Windows only, and its download
  page (github.com/jbman2025/ARC-miner) returned 404.

### Salad, 14 November: RDNA 4 against SRBMiner 3.8.8, where its lead comes from (goal-rdna4a, branch `goal/rdna4a`)

SRBMiner 3.8.8 (13 November) took the RX 9070 XT rows from 143% and 114% of SRBMiner 3.8.2 to 89-90% of 3.8.8
(bench-amd-1114). This run asked why, from the outside: SRBMiner's hashrate, its shares, and the card's clock read
two ways while it mined (Salad's `amd-smi metric` through WSL2, which reads the clock and voltage but no power, and
our clock probe, one wave reading SHADER_CYCLES against REALTIME beside the miner). Then it priced our fold's parts
with timing-only builds and tried four changes. Boxes in llmjob2, `rdna4a-x1` (RX 9070 XT) and `rdna4a-y1` (RX 9060
XT), each moved PC when restarted. Ours is next/after-v0.7.0's HIP core (8606534f, the same fold as 21a6e31b).

**Same PC, each miner 240 s on Kryptex, mean of the 10 s samples after the first minute; clock by amd-smi during the
turn:**

| Card | PC | Ours (TH/s, MHz, mV) | SRBMiner 3.8.8 (TH/s, MHz, mV) | Ours % of SRB | SRB's lead a clock |
|---|---|---|---|---|---|
| RX 9070 XT | 27ca922c | 131.6, 2368, 771 | 145.4, 2443, 800 | 90.5% | +7.1% |
| RX 9070 XT | b1332467 | 111.8, 2015, 759 | 127.5, 2146, 768 | 87.7% | +7.1% |
| RX 9070 XT | 354e321b (stand-in pool) | 124.9 | 135.3 | 92.3% | not read |
| RX 9060 XT | ec4f1585 (a slow PC) | 52.5, 2094, 737 | 58.0, 2154, 732 | 90.5% | +7.4% |
| RX 9060 XT | 7058e6f5 | 66.7, 2524, 797 | 74.6, 2534, 789 | 89.4% | +11.5% |
| RX 9060 XT | 28ddb67e | 74.7, 2751, 807 | 82.4, 2768, 815 | 90.7% | +9.6% |

- **Same k, honest figures.** On the stand-in pool (`verify-pool.js` at 2^210, two turns of 360 s each) SRBMiner was
  credited 133.9 and 129.9 TH/s (+-4%) against 135.3 shown, ours 127.0 and 124.4 against 124.9. Every share of both
  miners was k 8192. Its shares through sharetap.py are a compressed proof our decoder doesn't read ("proof ends early").
- **Per clock.** On 28ddb67e the clock probe read 2864 MHz under ours and 2808 under SRBMiner: SRBMiner did 916 MAC a
  CU a clock (91% of the WMMA's 1,009), ours 812 (80%). On the RX 9070 XT it also runs 3-6% higher clock, so it
  spends less energy a cycle as well.

**What our fold's parts cost** (timing-only builds, `hashrate.js` 45 s after 12 s, 3 rounds, builds taking turns,
twin of the default within 0.1-0.3%; clock by the probe):

| Build | RX 9070 XT 354e321b (k 8192) | RX 9060 XT 28ddb67e (k 4096) |
|---|---|---|
| default | 124.2-124.9 TH/s at 2260 MHz | 74.6-75.0 at 2856 |
| no slab barrier (ablation 2) | +0.3% | -0.2% |
| no LDS fragment reads (8) | +5.0% | +1.7% |
| every block reads tile 0, held in L2 (16) | +0.6% | +2.0% |
| no staging: no slab loads, no LDS stores (32) | +8.4%, 2456 MHz | +6.7%, 2975 MHz |
| no slab loads, LDS stores kept (128, new) | not run | +5.1% |
| no LDS stores, loads kept (256, new) | not run | +1.6% |
| readout without its cross-lane steps (64) | +0.6% | +0.5% |
| 2 + 8 + 32 | +8.0%, 2456 MHz | +7.6%, 2958 MHz |

On the power-capped RX 9070 XT the staging's whole cost is clock: the fold does the same work a clock without it.
Most of it is the slab loads from L2, not the LDS stores. Even with no staging, no fragment reads and no barrier
(2 + 8 + 32) our fold stays below SRBMiner a clock on the 9060 XT.

**`pearl_amd_peak` on both cards** (MAC a CU a clock, 6 waves a SIMD, miner operands; 9070 XT d30d0f2a / 9060 XT
28ddb67e): WMMA from registers 1000 / 1012; plus the fold's readout 886 / 887; readout without cross-lane steps 920 /
919; 16-byte LDS fragment reads and the barrier, no readout 908 / 914 (927 / 933 at 8 waves); both 786 / 789. Each
VALU beside the WMMAs costs about a WMMA clock (valui16: 935 on both). SRBMiner's 916 is what we get with either the
readout or the fragment reads, not both. The readout's 128 sums a chunk need at least 64 `v_xor3` (6.25% of the WMMA
time for any miner); our loop has 60 `v_xor3`, 16 `v_xor`, 8 DPP XORs, 4 `v_permlanex16` and 2 selects a chunk
(ROCm 7.2's code, counted from the assembly), close to that floor. So SRBMiner's operands cost it very little, and how
is not visible from outside.

**Tried:**

| Change | RX 9070 XT | RX 9060 XT | Notes |
|---|---|---|---|
| No LDS: each lane loads its fragments with buffer loads, one slab ahead (`PEARL_AMD_DIRECT=1`) | -5.7% (clock 2147 vs 2260) | -7.9% | bit-exact; 2.7x the bytes through the vector cache |
| The same two slabs ahead (`=2`) | -5.6% | -7.1% | |
| 256x256 blocks of 16 waves (`PEARL_AMD_W12_WR=4`), 4 waves a SIMD | not run | -7.8% (clock +5%) | a third fewer bytes loaded a MAC saves energy, but one block a WGP stalls |
| The same at 8 waves a SIMD (`PEARL_AMD_WAVES=8`), late readout off (192 VGPRs) | **+1.2 to +1.5%** | **+1.5 to +1.7%** | tile walk 8 x 6 and 4 x 6; RDTREE kept; with the late readout it spills |
| The same with KPAIR off (178 VGPRs) | not run | -17% | |
| Slab loads' cache policy (`PEARL_AMD_LDAUX`): non-temporal, last use | not run | -21% | high-temporal level |

**The 256x256 block as the default** (`goal/rdna4a` 639c85d7, `PEARL_AMD_W12_WR=4`: 8 waves a SIMD, late readout
off, tile walk 8 x 8 on gfx1201 and 4 x 8 on gfx1200; not in a release). Against the 128x256 block (the release's fold
with its 8 x 12 and 4 x 12 walks) on one PC, `hashrate.js` 45 s after 12 s, 4 rounds, a twin within 0.3%:

| Card | PC | Default (256x256) | 128x256 | Gain | Clock (probe) | verify-hits |
|---|---|---|---|---|---|---|
| RX 9070 XT (k 8192) | cbb31097 | 129.4-129.8 TH/s | 127.0-127.1 | +1.9 to +2.1% | 2374 vs 2290 MHz | 400 of 400 (2,211 hits, 275 salts) |
| RX 9060 XT (k 4096) | 28ddb67e | 76.25-76.35 | 74.75-74.83 | +1.9 to +2.1% | | 400 of 400 (2,543 hits, 318 salts) |

Same PC with SRBMiner 3.8.8, Kryptex, 240 s turns (the 256x256 flags on `fa1d916f`, the same fold): RX 9060 XT
28ddb67e, two turns: ours 74.64 and 74.62 TH/s, 256x256 76.10 and 76.09, SRBMiner 82.30 and 82.33, so 90.7% -> 92.4%.
RX 9070 XT d30d0f2a, one turn (the box moved PC after it): ours 132.9 (2404 MHz smi), 256x256 134.4 (2469), SRBMiner
147.0 (2480), so 90.4% -> 91.4%.

The default build against SRBMiner 3.8.8 on one PC, two 240 s turns each (the core built from 639c85d7): RX 9070 XT
cbb31097, ours 129.26-129.41 TH/s at 2349-2351 MHz (smi), SRBMiner 142.29 and 142.32 at 2387-2396: 90.9%, where the
release's fold on the same PC read 127.0 against 142.4 (89.2%). RX 9060 XT 28ddb67e: ours 74.7-75.9 (the PC got noisy,
SRBMiner 81.3-81.7): 92.3%.

**The readout's XORs as LDS atomics** (`pearl_amd_peak`, 5e862031): from registers it ran 917 and 904 MAC a CU a clock
on the two cards against 890 and 875 for the fold's readout (+3%), but with the fragment reads and the barrier 740 and
726 against 792 and 760 (-5 to -7%): the atomics contend with the LDS reads. Not built into the fold. A batch of 128
column offsets on the default build (9060 XT) read no better, on a PC whose rounds swung 3%.

**Next:** SRBMiner's lead a clock is still 7-11%, and from outside its operands look almost free. Ideas not yet
built: a wave64 fold (64x128 a wave in the same registers, a quarter fewer LDS bytes
a WMMA), and the readout's cross-lane and keep steps trimmed toward the 64-xor3 floor (about 1%). The Windows core keeps
the 128x256 block until a Windows Radeon has run either.

### Salad, 14 November: RDNA 4, a wave64 fold and five other tries (goal-rdna4b, branch `goal/rdna4b`)

Picking up from goal-rdna4a's list (above): a wave64 fold first, then the readout's cross-lane steps, then cheaper
operand delivery. Boxes in llmjob2: `rdna4b-x1` (RX 9070 XT, PC 354e321b, k 8192), `rdna4b-y1` (RX 9060 XT, PC
28ddb67e, dropped: its rounds swung 72-76 TH/s) and `rdna4b-y2` (RX 9060 XT, PC 7058e6f5, k 4096). Ours is
`goal/rdna4a`'s default, the 256x256 block. Races are `hashrate.js` 50 s after 15 s, 3 rounds, builds taking turns,
with a twin of the default; every build that changed the fold passed the fold check (bit-exact).

**Wave64, priced first** (`test/pearl_amd_peak64.cpp`, built with `-mwavefrontsize64`; MAC a CU a clock, 4 waves64
a SIMD). A wave64 holds a 64x128 tile in the same 128 accumulator VGPRs (4 a WMMA, not 8), so a WMMA needs a quarter
fewer operand bytes.
- From registers it runs at wave32's rate: 1005 against 1008 on the 9070 XT, 990 and 990 on the 9060 XT.
- Its readout is cheaper: registers plus the readout 910 against wave32's 886 (9070 XT).
- Fed from LDS it first hit bank conflicts (654: rows r and r + 8 of a half-wave in one bank); with LDS32's half swap
  it ran 960 with no barrier, but the slab barrier cost it 12-15% where wave32 loses 3%. With the barrier pinned after
  the WMMAs, plus the readout: 124.5 against wave32's 126.0 TMAC/s (9070 XT), 74.3 against 73.9 (9060 XT). Level.
- With no LDS (each wave loads its own fragments; the fold's grid and operands, k 4096) and the readout: 121.5
  against wave32's no-LDS 114.2 TMAC/s (9070 XT), 74.3 against 66.6 (9060 XT).
- The layout, read on a card (`test/pearl_amd_lay64.cpp`): lane l holds A row l % 16 at k 4 (l / 16) .. +3, and
  accumulator g of lane l is C[g + 8 ((l / 16) & 1) + 4 (l / 32)][l % 16]. So lane bit 5 picks a region's row offset
  and a region's eight lanes are bits 0, 3 and 4, as in wave32.

**Built into the fold** (`PEARL_AMD_W64`, bit-exact on both cards once the layout was right; the whole gfx12 code
object compiled wave64):

| Build | RX 9070 XT 354e321b | RX 9060 XT 28ddb67e |
|---|---|---|
| default (wave32, 256x256) | 125.5-125.6 TH/s | 74.1-76.3 (noisy PC) |
| `W64=1`: no LDS, buffer loads one slab ahead | 115.2-115.4 (-8.1%) | 67.1-68.1 (-11%) |
| `W64=2`: staged through LDS, 8-byte KPAIR reads | 123.9-124.0 (-1.2%) | 73.9-74.5 (-1.2%) |

**Ablations at 256x256** (timing only, RX 9070 XT, 2 rounds, default 125.43-125.76): no slab barrier (2) +2.1%, no
LDS fragment reads (8) +6.4%, no staging (32) +4.7%, readout without cross-lane steps (64) +1.9%. At 128x256 (goal-rdna4a)
they were +0.3%, +5.0%, +8.4% and +0.6%: the bigger block moved cost from staging into the barrier and the readout's
cross-lane steps.

**Tried at 256x256** (3 rounds, twin within 0.1% on both PCs):

| Change | RX 9070 XT 354e321b | RX 9060 XT 7058e6f5 | Notes |
|---|---|---|---|
| **Serpentine WMMA order** (`PEARL_AMD_SERP`) | **+0.3%** (126.15-126.23 vs 125.74-125.84) | **+0.2%** (69.58-69.75 vs 69.48-69.54) | each WMMA shares an operand with the one before |
| Readout as a reduce-scatter (`PEARL_AMD_RDRS`): the bit-4 step first, then 2 words | -0.3% | behind (noisy PC) | VALU 94 -> 80 a chunk, but more selects and moves |
| Late readout (`RDLATE=1`) | -3.6% | -3% | spills 18 VGPRs |
| Late readout with one-slab-ahead loads (`PF1`, `RDLATE=2`) | -1.4% | -1.3% | still spills 9 |
| Split barrier, stores after the first k-step | -23% | -22% | spills |
| Split barrier, stores first, signal before the WMMAs | -5.4% | -7% | the fence waits for all 8 reads before any WMMA |
| Split barrier, signal after the first k-step | -5.8% | | |
| Lagged barrier, three LDS buffers (`PEARL_AMD_LAGB=1`) | level | | 48 KB a block; a wave can run a step ahead |
| The same, loads a step earlier in two sets (`=2`) | -4.5% | | |
| Tile walks 4x16, 16x4, 8x16, 16x8 (9070 XT; 8x8 today) | -0.1 to -0.25% | | |
| CU mode (`-mcumode`) | -2.8% | -4.5% | |

None of the barrier changes recovered ablation 2's 2.1%: removing the s_barrier helps by letting waves drift apart,
not by saving the wait, and anything that delays the first WMMA until all of a slab's reads land costs more.

**Serpentine order by default** (`PEARL_AMD_SERP`, on with the RDNA 4 set; `goal/rdna4b` 0cacaff4 is goal-rdna4a's fold
plus this switch, and the other tries are reverted there, their code in that branch's history). verify-hits 400 of 400
on both cards (9070 XT: 2,171 hits, 270 salts at k 8192; 9060 XT: 2,393 hits, 297 salts at k 4096). The branch head
against itself with `-DPEARL_AMD_SERP=0`, 3 rounds: RX 9070 XT 126.17-126.21 against 125.92-125.93 (+0.2 to +0.3%,
twin 126.20-126.27); RX 9060 XT 69.34-69.57 against 69.26-69.33 (+0.1 to +0.3%, twin 69.35-69.45).

Same PC, each miner 240-300 s, mean of the 10 s samples after the first minute (ours = goal-rdna4a's default):

| Card, PC, pool | Ours | Serpentine | SRBMiner 3.8.8 | Ours % | Serpentine % |
|---|---|---|---|---|---|
| RX 9070 XT, 354e321b, stand-in pool (Kryptex sent this PC no work) | 125.99, 125.94 | 126.13, 126.13 | 134.73, 134.53 | 93.6% | 93.7% |
| RX 9060 XT, 7058e6f5, Kryptex | 69.34, 69.39 | 69.57, 69.58 | 74.02, 73.97 | 93.7% | 94.0% |
| RX 9060 XT, 7058e6f5, stand-in pool | 69.46, 69.19 | 69.25 | 73.39, 73.13 | 94.6% | 94.4% |

On the stand-in pool every share was credited (ours 450-467 a turn on the 9070 XT, k 8192). SRBMiner sent k 8192
shares there from the 9060 XT as well, where ours sends k 4096.

**Next:** the LDS fragment reads (6.4% at 256x256) are the largest single cost left, then staging (4.7%), the barrier
(2.1%) and the cross-lane steps (1.9%). Wave64 cuts the read bytes but loses as much at the barrier. Not tried: a
block of 512x256 (1024 threads, a third fewer slab bytes again, but one block a WGP), the readout's cross-lane steps
with EXEC masks instead of selects, and hand-scheduled interleaving of the readout's XORs with the last slab's WMMAs
(`sched_group_barrier`).

### Salad, 15 November: RDNA 4, five more tries and where the clock goes (goal-rdna4c, branch `goal/rdna4c`)

Picking up goal-rdna4b's list: a 512x256 block, the keep by exec mask, the readout beside the last slab's WMMAs, and
fewer LDS fragment reads. Then two more: one barrier a slab pair, and ablations with the clock read during each.
Boxes in llmjob2: `rdna4c-x1` (RX 9070 XT, PC 27ca922c, k 8192) and `rdna4c-y1` (RX 9060 XT, PC a0f5e858, k 4096).
Ours is next/after-v0.7.2's default (the 256x256 block with serpentine order). Races are `hashrate.js` 50 s after
15 s, builds taking turns, with a twin of the default. On 27ca922c the first build after a job's builds reads about
0.3% high (134.0 every time), so the rounds below leave that slot out where it matters. Every build here passed the
fold check (bit-exact) unless it says otherwise. Not rows: no competitor ran.

**RX 9070 XT, 27ca922c (base 133.4-133.6 TH/s, twin within 0.1% of it):**

| Build | TH/s | Against base | Notes |
|---|---|---|---|
| 512x256 block (`W12_WR=8`, 32 waves, one block a WGP) | 127.2-127.7 | -4.8% | 192 VGPRs, 48 KB LDS; the whole WGP waits at each barrier |
| A' straight from memory (`ADIR`, only B' through LDS) | 130.3-130.7 | -2.4% | 187 VGPRs, 16 KB; clock 2413 against 2450 MHz, and 1.0% less a clock |
| Four LDS buffers, one barrier a slab pair (`RING4`) | 126.1-126.9 | -5.5% | 187 VGPRs, 64 KB a block |
| Keep by exec mask (`RDEXEC`) | 133.67-133.89 | +0.2 to +0.3% | verify-hits 400 of 400 (2,262 hits, 285 salts); clock 2457 against 2450 |
| Readout in the chunk-end slab's block (`RDINT` 1 and 2) | 10.2-10.6 | | spills 500-568 bytes of scratch |
| `UNROLL4` at 256x256 | | | spills 696 bytes; not timed |

- `RDEXEC`: every lane runs the three cross-lane steps but leaves out the last XOR; the keeping lanes, as an exec-mask
  branch, XOR both halves into their keep with one xor3 a word. A first form with the permlanex16 inside the branch
  (fetch-inactive) failed the fold check. On the RX 9060 XT (a0f5e858, 4 rounds) it was level to +0.1%: 71.68, 71.47 and 71.47
  against base 71.51, 71.45 and 71.39, twin 71.54, 71.48 and 71.39. So it stays off (`-DPEARL_AMD_RDEXEC=1` builds it).
- The 512x256 block (idea 1) loses as 4 waves a SIMD did (goal-rdna4a): one block a WGP, so nothing runs while it
  waits at a barrier. `RING4`'s 64 KB a block probably leaves room for only one block a WGP too (not checked).

**Where the clock goes** (RX 9070 XT, 27ca922c, timing-only ablations, one run each with the clock probe beside it;
"a clock" is TH/s per GHz, base 54.3):

| Build | TH/s | Clock (MHz) | A clock |
|---|---|---|---|
| base | 133.64 | 2460 | 54.3 |
| twin | 133.61 | 2458 | 54.4 |
| no barrier (2) | 135.27 (+1.2%) | 2392 | 56.6 (+4.1%) |
| no cross-lane steps (64) | 135.45 (+1.4%) | 2423 | 55.9 (+2.9%) |
| no LDS fragment reads (8) | 136.24 (+1.9%) | 2480 | 54.9 (+1.1%) |
| no staging (32) | 135.21 (+1.2%) | 2528 | 53.5 (-1.5%) |
| none of those, and no readout in the loop (554) | 171.48 (+28%) | 2795 | 61.4 (+13%) |

- The card is held by its power limit. A part that wastes cycles (the barrier, the cross-lane steps) costs work a
  clock; a part that moves bytes (the staging, the LDS reads) costs clock. Each alone is worth only 1.2-1.9% of TH/s
  on this PC, because the other parts then take the freed power. With all of them gone the fold runs 93.7% of the
  WMMA peak at 2795 MHz.
- On this PC the LDS reads cost 1.9%, not the 6.4% goal-rdna4b measured on 354e321b. Idea 4 is worth at most that.
- SRBMiner 3.8.8 did 145.4 TH/s on this PC at 2443 MHz (goal-rdna4a): 59.5 a GHz against our 54.3, 9.5% more work
  a clock at about the same clock, so it spends less energy on everything that is not a WMMA.
- Bit 512 (no readout in the loop; in goal/rdna4c's history only) ran slower alone, 115.4 at 2476 MHz: the code
  changed shape, so read it as untimed. The 554 row is bits 2, 8, 32 and 512 together.

**Cost:** an RX 9070 XT box for about 2 hours and an RX 9060 XT box for about 20 minutes, high priority.

### Salad, 15 November: RDNA 4, one barrier a slab pair and what the ablations measure (goal-rdna4d, branch `goal/rdna4d`)

Picking up goal-rdna4c's RING4 (four LDS buffers, one barrier a slab pair, -5.5%): was it the loads' timing or the
64 KB of LDS? Boxes in llmjob2: `rdna4d-x1` (RX 9070 XT, PC fe255ef4, k 4096 in hashrate.js) and `rdna4d-y1` (RX
9060 XT, PC 26316a33, noisy: sd 1-2 TH/s). Ours is next/after-v0.7.2's default. Races are `hashrate.js` 50 s after
15 s, 3 rounds, builds taking turns, with a twin of the default (within 0.2%). Clock by the probe in one diag run
each. Every build here passed the fold check unless it is an ablation. Not rows: no competitor ran.

**RX 9070 XT, fe255ef4 (base 133.2-134.0 TH/s, twin 133.2-133.4):**

| Build | TH/s | Clock (MHz) | Notes |
|---|---|---|---|
| base | 133.2-134.0 | 2446 | 192 VGPRs, 32 KB |
| `RING4=1` (goal-rdna4c's: slab s + 3 loaded at the pair's second slab) | 125.8-125.9 (-5.8%) | | 187 VGPRs, 64 KB |
| `RING4=2` (both of a pair's slabs loaded at its first slab) | 125.2-125.8 (-6.1%) | 2480 | 182 VGPRs, 64 KB; a clock -7.4% |
| default padded to 48 KB of LDS | 133.6 | | |
| default padded to 64 KB | 133.9 | | two blocks a WGP still fit |

RX 9060 XT (26316a33, sd 1-2): RING4=1 and =2 65-68 against base 70-72.

So the 64 KB a block is not what RING4 loses on: the default padded to 64 KB runs level. RING4 does the same work in
fewer barriers and still loses 7% a clock.

**Ablations with the clock** (timing only, one diag run each, base 133.0 at 2446 MHz, 54.4 a GHz):

| Bits | Leaves out | TH/s | Clock (MHz) | A clock |
|---|---|---|---|---|
| 66 | barrier, cross-lane steps | 137.1 (+3.1%) | 2383 | +5.8% |
| 40 | LDS fragment reads, staging | 138.6 (+4.2%) | 2568 | -0.7% |
| 42 | barrier, LDS reads, staging | 140.1 (+5.3%) | 2558 | +0.7% |
| 106 | those three and cross-lane | 158.1 (+18.9%) | 2809 | +3.5% |

Reading: every ablation that drops the LDS reads or the staging also stops the WMMAs' operand data changing (the
fragments are then the same registers every k-step). The step from 42 to 106 drops only the cross-lane steps, about
1-2% of the work, yet adds 13% and 250 MHz. The likely cause is the operand registers going constant: less switching
in the WMMA, which a real fold can't have. goal-rdna4c's "+28% with everything removed" has the same confound. Bits
128 and 256 (operand bytes masked to zero or to their low bit, every load, store and WMMA kept) were built to
measure this; goal-rdna4e ran them (below).

### Salad, 15 November: RDNA 4, where the power goes, and two-lane regions (goal-rdna4e, branches `goal/rdna4e`, `goal/rdna4e-ship`)

Picking up goal-rdna4d's untimed builds (RING4=3, ablation bits 128 and 256, wave64 with RING4), then the readout
item from its list and what the ablations pointed to. Boxes in llmjob2: `rdna4e-x1` (RX 9070 XT; PC 8dd7fdc8, then
22f73721 from 04:14, k 4096 in `hashrate.js`) and `rdna4e-y1` (RX 9060 XT, PC 78285119). Ours is
next/after-v0.7.2's default plus the switch named. Races are `hashrate.js` 50 s after 15 s, builds taking turns, a
twin of the default; clock by the probe in one diag run each. Every build that changes the fold passed the fold
check (bit-exact) unless it is an ablation.

**RING4=3 and wave64.** RING4=3 (four LDS buffers, the ring half fixed at compile time) spills at the 192-VGPR cap:
256 bytes of scratch, also with `LPIN` 1 or 2 or `SCHEDB` 0. It ran 78.6-78.8 TH/s against 129.4-130.4 on the 9070
XT and 43.4-43.7 against 71.1-72.0 on the 9060 XT, so it does not test goal-rdna4d's idea. The assembly answers it
instead: RING4=1 and =2 add only four `v_add_nc_u32` and a few SALU instructions per 64 WMMAs to the loop (280 and
286 instructions against 266). The ring offset in a register is worth about 0.4%, not the 7% a clock RING4 lost.
RING4 is closed. Wave64 (`W64=2`) read 127.5 at 2415 MHz against base 129.4 at 2375 (-1.4%); with RING4=3 it spills.

**Where the power goes** (timing-only ablations, one diag run each with the clock):

| Build | RX 9070 XT 8dd7fdc8 | RX 9070 XT 22f73721 | RX 9060 XT 78285119 |
|---|---|---|---|
| base | 129.4 at 2375 MHz | 139.9 at 2683 | 71.1 at 2722 |
| operand bytes zeroed at the slab loads, all work kept (128) | 146.1 at 2873 | 158.3 at 3125 | 78.3 at 3145 |
| operand bytes masked to their low bit (256) | 141.1 at 2793 | | 75.6 at 3062 |
| fragments ANDed with zero just before the WMMAs (1024, new) | | 143.0 at 3118 | 70.7 at 3150 |
| the same ANDs with all ones (2048, new) | | 125.0 at 2736 | 62.3 at 2866 |

1024 and 2048 run the same instructions; only the values the WMMAs see differ. Zeros at the WMMAs alone gave +14%
clock and +14% TH/s on both cards, and 128 and 1024 reach about 3.1 GHz, the top clock, so the true share is larger.
In 1024 the staging, LDS and registers still carry real data, so almost all of the data-dependent power is in the
WMMAs. That also explains goal-rdna4c's "+28% with everything removed" and goal-rdna4d's 42-to-106 jump: those
ablations froze the operand registers. What a fold can change is the cycles and energy around the WMMAs, and the
operand values: a fill sweep (9070 XT 22f73721, one diag run each) read fill 0 123.9, -32 135.3, 16 133.3, 32 137.0,
48 (the default) 140.4, 63 140.8 TH/s. Fill 48 to 63 is the best, as the bit-flip model says. Neither s_setprio over
each slab's WMMAs (`PEARL_AMD_PRIO` 1 and 3: 138.6 and 138.9, -1.2%) nor B' outer in the WMMA order
(`PEARL_AMD_ORDER=1`: 140.1, level; 9060 XT 71.1, level) changed it.

**The readout's lane-bit-4 step once a tile** (`PEARL_AMD_RDK4`, goal-rdna4d's item 3; bit-exact): -5% on both cards
(9070 XT 123.2-124.0 against 129.8-130.3; 9060 XT 68.4-68.7 against 72.1-72.3), keeping by select (=1) or by exec
mask (=2). The assembly shows why: 8 more live VGPRs at the 192 cap, and the compiler loads the four A' fragments
of a k-pair one at a time into the same registers, each with a full LDS wait. The fold has no VGPRs to spare.

**Two-lane regions** (`PEARL_AMD_REG2`, new, opt-in). The miner chooses the tile pattern. In the RDNA 4 fold a lane
holds 32 rows by 4 columns of its wave's 64 x 64 of C, so a 32 x 8 region (rows {0-7, 16-23, 32-39, 48-55}, columns
{0, 1, 16, 17, 32, 33, 48, 49}, 256 sums like the profile's 16 x 16) is two lanes. A chunk's readout is then the
lane's 128 sums XORed in 6 chains to one word, one DPP step and one XOR into a queue of 8 keeps: about 68 VALU
against about 90, and fewer live words. The core reports the tile (`core.tile`) and hashes its pattern bytes
(0,7,1,3 and 0,1,7,3) into job_key, as the CUDA rotation fold does, so config52 and the proofs carry it.

| Build | RX 9070 XT 22f73721 (3 rounds) | RX 9060 XT 78285119 (3 rounds) |
|---|---|---|
| base | 140.18-140.76 | 72.03-72.20 |
| twin | 140.13-140.45 | 72.02-72.09 |
| REG2, 3 chains | 140.61-141.34 | 71.44-71.92 |
| REG2, 6 chains | **141.35-141.68 (+0.7%)** | **72.12-72.81 (+1.0% in rounds 2 and 3)** |
| REG2, 8 chains | 140.30-140.67 | 71.57-71.92 |
| REG2 with `LPIN=1` | 140.32-140.90 | 71.90-71.97 |
| readout without cross-lane steps or keep (ablation 64, the bound) | 142.89-143.32 (+1.9%) | 73.00-73.26 (+1.4%) |
| `RDEXEC` (diag) | 140.9 (+0.3%) | |

verify-hits: 400 of 400 on the 9060 XT (1,203 hits, 152 salts, k 4096) once the job key carried the tile (the first
run failed every hit at the Merkle check: the host had hashed the profile's pattern bytes).

**REG2 on by default** (`goal/rdna4e-ship`: next/after-v0.7.2 plus REG2 with 6 chains and nothing else of these tries;
`-DPEARL_AMD_REG2=0` builds the old tile), 4 rounds:

| Card, PC | default (REG2) | REG2=0 | twin of REG2=0 | verify-hits of REG2 |
|---|---|---|---|---|
| RX 9070 XT, 22f73721 | 141.17-141.68 | 140.04-140.86 | 140.13-140.47 | 400 of 400 (1,486 hits, 181 salts, k 8192) |
| RX 9060 XT, 78285119 | 72.11-72.78 | 71.71-72.03 | 71.42-72.05 | 400 of 400 (1,459 hits, 184 salts, k 4096) |

Same PC on Kryptex (9070 XT 22f73721, 240 s turns, mean of the 10 s samples after the first minute): REG2=0 140.39
and 140.46 TH/s (8 shares accepted), REG2 141.43 and 141.10 (9 accepted, 0 rejected: the pool takes the 32 x 8
tile), SRBMiner 3.8.8 152.99 and 153.08. Ours went from 91.7% to 92.3% of SRBMiner on this PC. (On goal-rdna4b's
PCs ours was 93.6-94.0%.)

**Where this leaves RDNA 4.** SRBMiner still does about 9% more work a clock at the same clock and power. These
runs show the WMMAs' own operand switching sets most of the card's power and is the same for any miner at fill 48,
so the difference has to be in cycles and energy around the WMMAs. Cutting the readout's VALU (REG2) bought 0.6-1.0%
of ablation 64's 1.4-1.9% bound; extra live registers lose at once (RDK4), and so did a ring of four buffers, wave64,
priorities and WMMA order. Not tried yet: a readout that keeps all 16 words in the lane pair without the queue's
moves (by an index in M0, `v_movrels`), and folding the readout's XORs into the slab's WMMA stream with
`sched_group_barrier`.

**Cost:** an RX 9070 XT box for about 3 h 10 min and an RX 9060 XT box for about 3 h, high priority, llmjob2.

### Salad, 15 November: RDNA 4, what an instruction beside the WMMAs costs, and the slab's reads row by row (goal-rdna4f, branches `goal/rdna4f`, `push/rdna4f-wgrp`)

Boxes in llmjob2: `rdna4f-x1` (RX 9070 XT, PC d30d0f2a, k 8192 in `hashrate.js`) and `rdna4f-y1` (RX 9060 XT, PC
af75e0bc, k 4096). Base is REG2 with 6 chains (now the default). `hashrate.js` 50 s after 15 s, builds taking turns,
a twin of the base.

**A cost probe** (`test/pearl_amd_cost.cpp`, new; timing only): the fold's slab of 32 WMMAs from registers, plus N
instructions of one kind at one place, in MAC a CU a clock (1,000 for the WMMAs alone; 9070 XT, 8 waves a SIMD).

| Added a slab | before or after the WMMAs | spread between them |
|---|---|---|
| 8 `ds_load_b128` (data unused) | 988 / 983 | 882 |
| 8 `buffer_load_b128` | 977 | 885 |
| 8 `ds_store_b128` | 967 (after) | 861 |
| 16 independent `v_xor_b32` | 946 / 940 | 870 |
| 16 `s_add_co_u32` | | 871 |
| 8 `s_nop 0`, or 8 `s_wait_dscnt` with nothing outstanding | | 1,005-1,007 |
| a dependent `v_xor` chain of 16 | 782 without `s_delay_alu`, 892 with | |
| 4 LDS reads each waited for at once | | 917 |
| the slab barrier | 1,006 | |

An SALU, VALU or LDS instruction between two WMMAs costs 3-4 WMMA cycles; the same in a burst about one. Waits that
are already satisfied are free. On the 9060 XT, bursts of 24, 32 and 48 b128 reads a slab gave 638, 496 and 328, and
32 b64 reads 507: the LDS takes about one wave instruction per 4 cycles a WGP whatever its width. The fold issues
about 160 LDS instructions a block-slab against 1,024 WMMA cycles a SIMD, so the LDS is about two thirds busy.

**WGRP=4, on by default** (`PEARL_AMD_WGRP`): the slab reads B' first, then A' row by row, and runs each row of WMMAs
after a wait for its own A' read; the compiler's order waited for all of A' and B' 0 before the first WMMA.
Bit-exact; verify-hits 400 of 400 on the 9070 XT (1,420 hits, 174 salts, k 8192) and the 9060 XT (1,457 hits, 184
salts, k 4096). `-DPEARL_AMD_WGRP=0` builds the old order.

| Card, PC | WGRP=4 | base (REG2) | twin |
|---|---|---|---|
| RX 9070 XT, d30d0f2a (12 rounds over 4 jobs) | 135.41-135.67 | 134.57-135.10 | 134.59-134.71 |
| RX 9060 XT, af75e0bc (noisy PC, sd 1-2; quiet round) | 71.97-75.27 (75.27) | 71.33-75.10 (75.10) | 70.94-75.09 |

The 9070 XT gains 0.6-0.7%; the 9060 XT about 0.3-0.5%. One VGPR spills (8 bytes), outside the slab loop.

**Tried and dropped** (all bit-exact unless said):
- WGRP=1 (one wait for all 8 reads, then 32 WMMAs unbroken): -0.2%. =2 (row 0, then the rest): +0.3%.
- PFR, three LDS buffers and each k-step's fragments read a k-step ahead (8-byte reads): 9070 XT 121.6-124.2 vs
  135. Its controls: the same loop reading after the barrier 113.9 (three buffers) and 119.2 (two buffers); with the
  default's 16-byte reads 131.8. Reading ahead is worth about 7%, but needs 8-byte reads (twice the LDS
  instructions) or 32 more VGPRs. A loop of 128 WMMAs (four slabs) spilled 750 bytes.
- Base padded to 48 KB of LDS: 134.7 (level), so two 48 KB blocks still fit a WGP.
- Wave64 through LDS (`W64=2`, 25% fewer LDS instructions a MAC): 131.5 against REG2=0's 133.4 (-1.4%).
- K64, 64-byte slabs (one barrier a 64 bytes of k, the second k-pair read mid-slab): 9070 XT 121.7, 9060 XT 66.9
  against 75.1; 12 bytes of spill whose reload waits on the load counter, and so on the staging loads, every slab.
- gfx12 has no LDS DMA, `ds_load_tr`, `v_xor_b64` or `v_bitop3` (llvm-mc on the box).

**The spill, fixed (b619b92a).** 43b20fa7 kept the reads in an empty asm as well, which cost one VGPR that
spilled (8 bytes, outside the slab loop), and CI's fold check, which allows no spill, failed. The reads return in
order, so each row's first WMMA already waits for its own A' read; the sched_barriers alone keep the rows apart.
Without the asm: 192 VGPRs, no spill or scratch on gfx1200 and gfx1201. RX 9070 XT (d30d0f2a, 2 rounds): 135.46-135.62
against WGRP=0's 134.76-134.79 and the asm form's twin 135.58-135.61. verify-hits of b619b92a 400 of 400 on the RX
9060 XT (1,205 hits, 152 salts, k 4096).

**Against SRBMiner 3.8.8 on the same PC, shown and credited.** RX 9070 XT d30d0f2a, Kryptex, 2 turns of 240 s:
SRBMiner 145.29 and 145.39 TH/s shown at 2515 MHz, ours (REG2 + WGRP=4) 135.56 at 2494 MHz: 93.3% shown. On the
stand-in pool (`verify-pool.js`, turns of about 560 s at 2^209 after one of 318 s at 2^210), the credited rates:

| Card, PC | Ours credited (shown) | SRBMiner 3.8.8 credited (shown) | Ours by credit |
|---|---|---|---|
| RX 9070 XT, d30d0f2a | 139.19, 134.70, 127.90 (135.6) | 132.13, 131.93, 138.72 (143.8) | 133.0 vs 134.6 by time: 98.8% +-2.5% |
| RX 9060 XT, af75e0bc | 73.40, 75.41 | 71.63, 67.86 | 74.4 vs 69.7: 107% +-4% |

Every share of both was accepted; ours were k 8192 on the 9070 XT and k 4096 on the 9060 XT, SRBMiner's k 8192 on
both. SRBMiner shows about 7% more than it is credited on the 9070 XT (145.3 shown, 134.6 credited); ours shows
about what it is credited. As goal-rdna3a found on RDNA 3, the shown figures overstate SRBMiner's lead: by credit
the two are level on this 9070 XT, within the runs' error.

**Cost:** an RX 9070 XT box and an RX 9060 XT box, high priority, llmjob2, from 07:15.

### Salad, 15 November: RDNA 4, firmer credit, two chunks a block and the readout's words in LDS (goal-rdna4g, branches `goal/rdna4g`, `push/rdna4g-c8q`)

Boxes in llmjob2: `rdna4g-x1` (RX 9070 XT, PC 8496ad78; it moved to a bad PC at 20:08), `rdna4g-x2` (RX 9070 XT,
PC cbb31097), `rdna4g-y1` (RX 9060 XT, PC ec4f1585, a slow PC: SRBMiner showed 38 TH/s, dropped) and `rdna4g-y2`
(RX 9060 XT, PC 8128b9d9). Races are `hashrate.js` 50 s after 15 s, builds taking turns, with a twin; every build
that changed the fold passed the fold check (bit-exact).

**Credit, longer runs** (`verify-pool.js`, each miner 4 turns of 600 s on the 9070 XT at 2^211, 3 on the 9060 XT at
2^212, about 2,100-2,300 shares a turn; shares after the first 60 s of a turn; error 1/sqrt(shares)). 2^207, tried
first, is a harder target and gave only about 150 shares a turn.

| Card, PC, ours | Ours credited (shown) | SRBMiner 3.8.8 credited (shown) | Ours by credit | Ours shown |
|---|---|---|---|---|
| RX 9070 XT, 8496ad78, 0490e409 | 134.4 +-1.1% (137.0) | 135.8 +-1.1% (142.6) | 98.9% +-1.5% | 96.1% |
| RX 9070 XT, 8496ad78, d2c8e93a | 136.2 +-1.1% (138.0) | 135.8 +-1.1% (142.5) | 100.3% +-1.5% | 96.9% |
| RX 9060 XT, 8128b9d9, d2c8e93a | 72.6 +-1.2% (73.5) | 71.0 +-1.2% (73.8) | 102.2% +-1.7% | 99.6% |
| RX 9060 XT, 8128b9d9, d2c8e93a again | 72.4 +-1.2% (73.3) | 71.5 +-1.2% (73.7) | 101.2% +-1.7% | 99.4% |
| RX 9070 XT, cbb31097, 00b15d75's fold | 133.1 +-1.1% (134.7) | 130.8 +-1.1% (138.0) | 101.7% +-1.5% | 97.6% |
| RX 9060 XT, 8128b9d9, 00b15d75 | 73.7 +-1.2% (74.4) | 70.7 +-1.2% (73.7) | 104.1% +-1.7% | 100.9% |

Every share was accepted. SRBMiner shows 4-5% more than it is credited; ours shows 1-2% more. Its shown rate fell
from 148 to 142-144 TH/s at 2^211 against 2^207 (more shares to prove), ours from about 139 to 137-138.

**What was tried** (9060 XT 8128b9d9 unless named; base is next's default at the time):

| Change | Result | Notes |
|---|---|---|
| `C4`: every chunk but the last as four slabs known at compile time, no per-slab clamp, store test or readout test | level | the loop's SALU fell from about 12 a slab to 1; the SALU beside the LDS phases costs nothing measurable |
| `C4=2`: two chunks at a time, so each chunk's lane parity is known | +0.4 to +0.6% | spilled 40 bytes (10 VGPRs) |
| `RFRAME`: REG2's words kept unrotated, each chunk's word rotated in by one `v_alignbit` | level | takes the every-16-chunks branch out |
| `QLDS`: REG2's 8 words in LDS, one `ds_xor_b32` a chunk, 16 KB more LDS a block (48 KB) | +0.4% | 184 VGPRs |
| `C4=2` + `QLDS` | +0.65% (9060 XT), +0.8 to +1.1% (9070 XT 8496ad78) | 190 VGPRs, no spill: shipped as d2c8e93a |
| `QLDS=2`: both lanes of a pair XOR their own half into the keeper's word, no DPP step or select | +0.5 to +0.7% on both | 191 VGPRs |
| `QLDS=2`, 5 readout chains, no load clobber (`C4LP=0`) | +1.3 to +1.5% (9060 XT), +1.9% (9070 XT cbb31097) | shipped as 00b15d75 |
| readout chains 4, 7, 8 (with QLDS) | -0.5 to -1% against 5 or 6 | |
| `AHEAD`: the next slab's A' read during this slab, four LDS buffers (64 KB), staging a slab further ahead | -5.7% | |
| `AHEAD=2` (its order and buffers, A' read after the barrier, timing only) | -1.2% | so the reads ahead cost about 4.5% more on their own; reading ahead is closed |

The first C4 build sank three slabs' WMMAs below their barriers and kept 24 fragments in scratch (700 bytes): the
barrier's fences don't hold instructions that touch no memory. An empty asm using every sum after each slab's
WMMAs (`acc_hold`) keeps them in their slab.

**Shipped:**
- d2c8e93a (C4=2, RFRAME, QLDS): RX 9070 XT 8496ad78, 4 rounds: 139.18-140.18 against the old loop 137.96-138.47,
  twin 139.37-139.68 (+0.8 to +1.2%). RX 9060 XT 8128b9d9, 3 rounds: 73.48-73.90 against 72.97-73.02, twin
  73.49-73.53 (+0.7%). verify-hits 400 of 400 (1,595 hits at k 8192; 1,655 at k 4096).
- 00b15d75 (QLDS=2, 5 chains, no C4LP): RX 9070 XT cbb31097, 4 rounds: 134.87-135.18 against d2c8e93a's settings
  132.40-132.60, twin 134.86-134.94 (+1.9%). RX 9060 XT 8128b9d9, 4 rounds: 74.29-74.45 against d2c8e93a's loop
  73.16-73.53, twin 74.21-74.40 (+1.2 to +1.6%). verify-hits 400 of 400 (1,568 hits at k 8192; 1,678 at k 4096).
  191 VGPRs, 48 KB LDS, no spill or scratch on gfx1200 and gfx1201. Earn suite 3478 passed on the exact commit.
- Together, against 0490e409's loop: about +2.7% on the 9070 XT and +2% on the 9060 XT.

**After shipping** (00b15d75's fold; RX 9070 XT cbb31097, 4 rounds; RX 9060 XT 8128b9d9, 3 rounds), all level or
worse, so the defaults stay:
- Operand fill 56 and 63 against 48: +0.2 to +0.3% on both cards, about the twin's spread (9070 XT 134.94-135.28
  against 134.63-134.92, twin 134.61-134.92); 40: -0.5%. Not changed (the fill is shared with the CUDA cores).
- `C4=1` with QLDS=2: -0.4%; with 6 chains -0.9%; 4 chains with C4=2: -0.5%.
- `WGRP=5` (A' row 0 read first, then B', row 0's WMMAs one at a time): 134.80-134.84 against 134.79-134.84, level.

So by credit both RDNA 4 cards are now over SRBMiner 3.8.8 on these PCs (9070 XT 101.7% +-1.5%, 9060 XT 104.1%
+-1.7%), and on the 9060 XT the shown rate is too (100.9%). On the 9070 XT SRBMiner still shows 2.4% more than ours.

**Cost:** two RX 9070 XT boxes and two RX 9060 XT boxes, high priority, llmjob2, from 12:50.

### Salad, 16 November: RDNA 4's lagged barrier without its spill, and fill 63 on gfx12 (goal-rdna4h, goal-rdna4i)

Same PC for every comparison, hashrate.js, 65 s runs taking turns. RX 9070 XT `38d1e6af` (a slow PC: 121 TH/s where
others make 135-142), RX 9060 XT `996de981`.

**The lagged barrier (`PEARL_AMD_LAG`, goal/rdna4h b0ff7e34 and c5d7ece1, opt-in, not on this branch).** Three LDS
buffers (64 KB a block, still two blocks a WGP) and gfx12's split barrier: a wave waits for the step before only after
its own WMMAs. Every build of b0ff7e34 spilled 248 bytes, but its ISA showed the 24-slab hot loop with no scratch at
all; the spill was in the tail chunks. c5d7ece1 holds the tail slabs' sums in their slab (`acc_hold`): 184 VGPRs for
`LAG=1`, 189 for `=2`, no scratch on gfx1200 or gfx1201. It still loses:
- RX 9070 XT, 3 rounds: base 120.72/120.72 (122.14 in the first slot), twin 120.71-120.76, `LAG=1` 119.19-120.13
  (-1.0 to -1.3%), `LAG=2` 119.23-119.44 (-1.2%).
- RX 9060 XT, 3 rounds: twin 79.42, `LAG=1` 78.65-78.78 (-0.8%), `LAG=2` 77.7-78.3 (noisy).
So the split barrier costs more than the waiting it saves, as goal-rdna4b found before C4. Closed.

The shipped hot loop (gfx1201), per slab: 32 WMMAs, 2 `buffer_load_b128`, 8 `ds_load_b128`, 2 `ds_store_b128`,
7 waits, the barrier's signal and wait, 2 SALU; per chunk of 4 slabs, 3 `v_xor` + 62 `v_xor3` + an `alignbit` + one
`ds_xor`. Nothing in it is spare.

**Fill 63 on gfx12 (`PEARL_AMD_GFX12_FILL`, shipped).** The operand fill on RDNA 4 is now 63; every other card keeps
48. At 63, A' = 63 + noise stays in [0, 126], so its sign bit never flips. Built as a host change (`Ctx::fill`), raced
against `-DPEARL_AMD_GFX12_FILL=48`, with a twin of each, 4 rounds (the first round's first slot reads high on both
PCs and is left out):

| Round | 9070 XT fill 63 / twin | 9070 XT fill 48 / twin | 9060 XT fill 63 / twin | 9060 XT fill 48 / twin |
|---|---|---|---|---|
| 1 | 122.15 / 121.34 | (122.52 first) / 121.19 | 79.60 / (78.93, sd 1.2) | (79.81 first) / 79.30 |
| 2 | 121.24 / 121.06 | 120.80 / 120.83 | 79.55 / 79.48 | (78.65, sd 1.2) / 79.38 |
| 3 | 120.92 / 120.95 | 120.71 / 120.78 | 79.48 / (78.41, sd 1.4) | 79.34 / 79.35 |
| 4 | 120.84 / 120.85 | 120.66 / 120.70 | 79.50 / (78.40, sd 1.4) | (78.81, sd 1.1) / 79.36 |

On the 9070 XT fill 63 led in every round, by 0.12 to 0.36%; each pair of twins was within 0.01 to 0.15%. On the
9060 XT runs with the PC's dips (sd over 1) are in brackets; of the clean runs (sd 0.12-0.13), every fill-63 run
(79.48-79.60) is above every fill-48 run (79.34-79.38): +0.15 to +0.25%. goal-rdna4g had the same +0.2 to +0.3% at
the default fold. verify-hits with fill 63: 400 of 400 on both (2,007 hits, 257 salts at k 8192 on the 9070 XT;
2,593 hits, 327 salts at k 4096 on the 9060 XT), 0 failed. The fold is unchanged: 191 VGPRs, 48 KB LDS, no spill or
scratch on gfx1200 and gfx1201. Small, but every round on both cards.

**Tried after it, all with fill 63, none shipped:**
- The second fold queue and the hash's own queue at run time (`PEARL_AMD_FOLDQ2=1`, `PEARL_AMD_HASHQ=1`), 4 rounds.
  RX 9070 XT: base and twin 120.89-121.09; FOLDQ2 118.79-119.25 (-1.7%, its runs unsteady, sd 1.6-2.0); HASHQ
  119.94-120.00 (-0.8%); both 118.56-118.65 (-1.9%). RX 9060 XT (base and twin 79.44-79.51 in clean runs): HASHQ
  76.5-76.9 (-3.5%), both 77.6 (-2.3%), FOLDQ2 79.87 and 79.90 in two runs and 76.65-76.89 in the other two. Each
  also costs the CPU a core more. Off on RDNA 4, as they were.
- Where a batch's time goes (`PEARL_AMD_TRACE`): RX 9070 XT, 36.46 ms a batch, the fold 36.09, the hash 0.232, idle
  0.10 (1.0% outside the fold), clock 2127 MHz on this PC; RX 9060 XT, 28.12 ms, fold 27.61, hash 0.457 (1.6%), idle
  0.02. The hash kernel reads the batch's 128 MiB of transcripts at about the cards' memory bandwidth.
- Loads straight into LDS (`global_load_lds_*`, `buffer_load ... lds`, `global_load_async_to_lds_b128`) and the
  transposing loads (`ds_load_tr16_b128`, `global_load_tr4_b64`): the gfx1200 and gfx1201 assembler takes none of
  them. The slab has to go through VGPRs.

**The shipped head (e3b1501e, fill 63) against SRBMiner 3.8.8 on these two PCs.** Displayed: Kryptex, 3 turns of
300 s each, taking turns, then one turn each of 180 s with the clock probe. Credit: the stand-in pool
(`verify-pool.js`), 2 turns each of 360 s, shares after the first 60 s; every share accepted.

| Card, PC | Ours shown | SRBMiner shown | Shown | Clock, ours / SRBMiner | Credit, ours / SRBMiner | By credit |
|---|---|---|---|---|---|---|
| RX 9070 XT `38d1e6af` | 120.99, 121.00, 120.94 | 128.22, 128.21, 128.42 | 94.3% | 2136 / 2210 MHz | 2,085 / 2,130 shares at 2^211 (115.2 / 117.5 TH/s) | 97.9% +-3.1% |
| RX 9060 XT `996de981` | 78.60, 79.53, 78.82 | 83.40, 83.43, 83.40 | 94.3-95.3% | 2844 / 2865 MHz | 2,744 / 2,772 shares at 2^212 (75.8 / 76.6 TH/s) | 99.0% +-2.7% |

These are slower PCs than goal-rdna4g's (9070 XT `cbb31097` 97.6% shown, 9060 XT `8128b9d9` 100.9%). The 9070 XT
here runs about 2,130 MHz, so the power cap binds harder, and SRBMiner gets 3.5% more clock and 2.4% more per clock.
On this 9060 XT both run the same clock and the gap is per clock; ours also dips to 76.3 TH/s for a minute at a time
on this PC, which SRBMiner's figures show less. By credit both are level with SRBMiner within the error. SRBMiner's
shares on the 9060 XT are k 8192; ours there are k 4096.

**k 8192 on the RX 9060 XT (shipped for 16 GB cards).** SRBMiner mines k 8192 there, and ours had stayed at k 4096
since 4 November, when the fold of then was about 6% slower at k 8192. With today's fold, `PEARL_AMD_K=8192` at run
time, 4 rounds on `996de981`: k 8192 and its twin 80.16-80.20 TH/s in every clean run (one dip run, 79.16 at sd 1.5,
left out), k 4096 and its twin 79.48-79.51: +0.9%. A larger k also halves the hash kernel's share of a batch (1.6% at
k 4096 on this card). The host now picks k 8192 on gfx1200 with 12 GB or more (`k8192_by_default`); the 8 GB model
keeps k 4096 so the smallest LLM still fits beside the miner. Built that way (52bbad38), 3 rounds: the default
(now k 8192) and its twin 80.16-80.21 in every clean run, `PEARL_AMD_K=4096` and its twin 79.48-79.49 (the PC dipped
in the other runs, sd 1.0-1.6). The trace at k 8192: 56.56 ms a batch, the hash 0.456 ms (0.8%). verify-hits at the
new default: 400 of 400 (1,295 hits, 165 salts, k 8192, gfx1200), 0 failed. On the 9070 XT the same commit verified 400 of 400 (2,027 hits, 259 salts, k 8192). 191
VGPRs, no spill or scratch on gfx1200 or gfx1201 (the fold is unchanged; only the host's choice of k moved).

**Column batch 512 on the RX 9070 XT** (half the launch tails a MAC), 4 rounds on `38d1e6af`: 120.76-121.18 against
the release's 121.06-121.28 (twin 121.08-121.54): -0.3%. Not changed.

### Salad, 16 November: RDNA 4 mines k 16384 on the big gfx1201 cards, and five tries that lost (goal-rdna4o, branches `goal/rdna4o`, `goal/rdna4o-ship`)

k 16384 was taken back out the same day: three more PCs did not agree (next section).

Boxes in llmjob2: `rdna4o-x1` (RX 9070 XT, PC 85a5c49c, a slow PC: base 133.7-134.2) and `rdna4o-y2` (RX 9060 XT,
PC 28ddb67e). `hashrate.js` 50 s after 15 s, builds taking turns with a twin of the default; clock from the probe in
one diag run.

**Shipped: k 16384 by default on gfx1201 with 56+ CUs** (`k16384_by_default` in `pearl_host_hip.cpp`). At k 8192 the
hash kernel was still 0.78% of a 9070 XT batch (0.254 of 32.88 ms); at k 16384 it is 0.35% (0.229 of 65.38 ms) and the
fold runs at the same rate (65.07 ms against 2 x 32.57).

| Build | RX 9070 XT | RX 9060 XT |
|---|---|---|
| base (k 8192) / twin | 133.73-133.78 / 133.72-133.85 | 78.48-79.12 / 78.47-78.55 |
| `PEARL_AMD_K=16384` / its twin | 134.60-134.70 / 134.62-134.73 (**+0.65%**) | 75.57-75.82 / 75.59-75.71 (-3.5%) |
| k 8192, a new job every 30 s | 133.64-133.74 | 78.40-78.44 |
| k 16384, a new job every 30 s | 134.46-134.57 (+0.6%) | 75.44-75.48 |

The 9060 XT's batch at k 16384 holds 64 MB of B', twice its 32 MB Infinity Cache, and its fold slowed to 115.87 ms
against 2 x 55.51; it keeps k 8192. verify-hits at k 16384: 400 of 400 on both cards (753 and 445 hits). Kryptex took
6 of 6 shares from the shipped core at k 16384. k 16384 needs about 6,961 MiB; short of it a card steps down one k
at a time.

**Shipped: a column batch of 128 on gfx1200 at k 8192 or more** (the 16 GB RX 9060 XT; `amd_col_batch`). At 256 a
batch's B' is 32 MB, all of the card's Infinity Cache. 5 rounds, twins of each: 128 read 78.83-78.98 against
78.43-78.56 (+0.55%, ahead in every round), clock 2827 against 2807 MHz; verify-hits 400 of 400 at 128 (891 hits). The
RX 9070 XT at k 16384 read 133.53-133.66 at 128 against 133.89-134.36 (-0.25%) and keeps 256.

**Tried and dropped** (all on `goal/rdna4o`, opt-in, none shipped):
- `PEARL_AMD_RDSPLIT` (half the waves run the readout at the chunk's end, half as a burst after the next barrier),
  9070 XT 3 rounds: =1 130.10-130.29 (-2.8%), =2 128.39-128.55, =3 130.67-130.70, and every wave late (RDLSB=2, no
  spill) 130.72-130.76 (-2.5%). The late burst itself costs 2.5%.
- `PEARL_AMD_RDBANK` (the readout's XORs as inline asm, two sums from different VGPR banks each; the release has 29
  of 65 XORs reading two sources from one bank): every mask 131.85-132.11 (-1.4%) on the 9070 XT. `RDBANK_H` (plain
  XORs held by empty asms) level on the 9060 XT (78.56-79.09 against twin 78.57-78.86). Bank clashes cost nothing.
- `PEARL_AMD_WUNS` (B', or both operands, read as unsigned; the same sums with fill 63): level on both cards
  (9070 XT 133.71-134.03 against base 133.69-134.17), same clock (2341 against 2343 MHz).
- gfx1201's assembler takes no `matrix_a_reuse`/`matrix_b_reuse` on v_wmma.
- SRBMiner 3.8.8 alone on the 9060 XT at --gpu-intensity 10, 20 and 31: 82.17-82.42 against 82.31 at its default.
  Its only public AMD knob changes nothing for pearlhash.

### Salad, 16 November: k 16384 taken back out on the big gfx1201 cards (goal-rdna4p)

The k 16384 default above rested on one PC. Three more RX 9070 XT PCs in llmjob2, at `6dfc421e`, with `hashrate.js`
taking turns with a twin build:

| PC | Rounds | k 16384 (default / twin) | k 8192 (`PEARL_AMD_K=8192`) | k 16384 against k 8192 |
|---|---|---|---|---|
| 85a5c49c (goal-rdna4o, above) | see above | 134.60-134.70 / 134.62-134.73 | 133.73-133.78 | +0.65% |
| 0798158d (2501 MHz) | 3 | 143.86-144.14 / 144.08-144.26 | 144.26-144.41 | -0.17%, level |
| 2b9f4078 (2365 MHz) | 4 | 133.71-134.74 / 133.74-135.26 | 136.30-137.08, ahead every round | -1.6% |
| 92aa4381 (2398-2412 MHz) | 4 | 127.47-136.31 / 131.91-137.00 | 127.72-136.47 / twin 128.69-137.46 | level (+0.45% on the means, noisy) |

92aa4381 slowed about 6% from its third round on, so its rounds disagree: k 8192 led by 0.1-0.5% in rounds 1 and 2
and trailed after. One gain, two level and one loss is not a safe default, so the big gfx1201 cards mine k 8192 again,
as in v0.7.3. `PEARL_AMD_K=16384` still asks for k 16384, and the one-k-at-a-time VRAM step-down stays. The 16 GB
RX 9060 XT's column batch of 128 (above) is unchanged.

**Shipped after v0.7.4: the 16 GB RX 9060 XT keeps the old tile order at batch 128** (`PEARL_AMD_WALK_MINGX 16` on
gfx1200 in `pearl_amd_fold.h`; goal-rdna4p). At batch 128 the grid is 8 column tiles wide, and the 4 x 8 supertile walk
was tuned at 256. Any walk 8 tiles wide is the old order, and it won on two PCs, ahead in every round:

| PC | Rounds | Old order (8-wide walks, or the build) / twin | 6dfc421e's walk / twin | |
|---|---|---|---|---|
| ac3f08f2 (2666 MHz) | 3 | walk 8x8 76.73-76.99 | 76.39-76.87 / 76.45-76.61 | +0.4% |
| 8aca3ffa (2684 MHz) | 4 | 8x8, 8x4, 8x16, 8x2: 76.66-76.81 | 76.32-76.49 / 76.33-76.39 | +0.45% |
| 8aca3ffa, ship check of the build | 3 | 76.75-77.03 / 76.73-76.79 | `MINGX=0` 75.70-76.48 / 75.71-76.36 | +0.8% |

verify-hits on the build: 400 of 400 (836 hits, batch 128, k 8192, gfx1200). The 8 GB card's batch of 256 still walks.

### Salad, 16 November: each RX 9070 XT PC times k 8192 against k 16384 and keeps the faster (goal-rdna4q)

k 16384 wins on some RX 9070 XT PCs and loses on others (the table above), and nothing the app reads tells them apart.
So the app now times both on the PC, as RTX 30 times its launch shapes (`shared/kTrial.js`, `PearlMiner._planK`). The
first time a gfx1201 card opens at k 8192 in a week, it mines 30 s uncounted, then four 30 s turns at k 8192, 16384,
16384 and 8192, the first 8 s of each left out. Each change of k opens the card's core again with `kPin`
(`pearl_host_card_k_pin`); the card is down about 1 s each time (CLI timestamps: card line to first rate line 0.5-1.1
s). k 16384 is kept only if it leads by 0.3% on the means, in every turn, and by twice the gap between the two turns of
each k. Otherwise the card stays at k 8192. The answer is kept a week per card (`k-trials.json`).
`LLMJOB_K_TRIAL=0` turns it off and `PEARL_AMD_K` overrides it.

Earlier designs, dropped: 20 s turns in the order 8192, 16384, 8192, 16384 with no warm-up kept k 16384 3 of 4 times on
d7020f76 (+0.27 to +0.67%); with a warm-up and the order 8192, 16384, 16384, 8192 it read -0.09, +1.04, +0.27 and +0.66%
there. The 30 s turns and the spread rule are what made the answers repeat.

All at `96b6244f`, Kryptex, llmjob2. `hashrate.js` rounds take turns with a twin build. The CLI A/B is the CLI itself
mining with `PEARL_AMD_K` and the trial off, 240 s a run, the mean of its rate line after 60 s, order alternating.

| PC | `hashrate.js`: k 8192 / twin, k 16384 / twin | CLI A/B, k 8192 / k 16384 | Trials (each timing afresh) |
|---|---|---|---|
| 1d0f121d | 119.86 / 119.64, 117.14 / 117.06 (4 rounds): **-2.3%** | | 4 of 4 kept k 8192: -2.13, -2.16, -2.13, -2.25% |
| e2598c10 | 135.23 / 135.07, 133.53 / 133.40 (3 rounds): **-1.2%** | | 3 of 3 kept k 8192: -1.24, -1.25, -1.21% |
| f3b6c003 | 141.63 / 141.51, 140.60 / 140.55 (3 rounds): **-0.7%** | | 3 of 3 kept k 8192: -0.80, -0.76, -0.78% |
| 8c091bb8 (noisy) | 129.81 / 129.47, 129.24 / 128.92 (3 rounds): -0.4% | | 4 of 4 kept k 8192: +1.39% (spread rule), +0.61% (not every turn), +0.09, -1.46% |
| d7020f76 | 124.49 / 123.68, 124.64 / 124.67 (4 rounds): +0.13%, twins 0.65% apart | 4 rounds: 122.91 / 122.94, level (medians +0.3%) | 4 of 4 kept k 16384: +0.57, +0.66, +0.85, +0.54% |

So the trial picked the faster k, or one level with it, every time: k 8192 on the four PCs where k 16384 loses
(-0.4 to -2.3%), and k 16384 on d7020f76, where the two are level in `hashrate.js` and in the CLI and the trial read
k 16384 0.5-0.9% ahead in each of its four runs. None of these five PCs showed 85a5c49c's clear +0.65%. A start after
a trial uses the answer and times nothing (f3b6c003: "k 8192, from its k trial of 2026-11-16"). What a trial costs:
on a PC where k 16384 loses 2.3%, 60 s at k 16384 is 1.4 s of mining, plus about 1 s for each of the two or three
reopenings.

verify-hits at k 16384 on `96b6244f` (d7020f76): 400 of 400 (1052 hits, 133 salts). Every trial run's shares were
accepted at both k, none rejected. Ours against SRBMiner 3.8.8 on the RX 9070 XT is still about 95.5% (goal-rdna4o):
this adds at most the +0.65% a PC like 85a5c49c gives, and nothing on the others.

### Salad, 17 November: RDNA 4, the RX 9060 XT hashes in the fold, and three energy tries that lost (goal-rdna4r, branches `goal/rdna4r`, `goal/rdna4r-ship`)

Aim: less energy a WMMA on RDNA 4, to close the gap to SRBMiner 3.8.8 (about 95.5% on the RX 9070 XT and 96% on the
RX 9060 XT by shown rate). Boxes in llmjob2: `rdna4r-x1` (RX 9070 XT `93afc1b2`), `rdna4r-y1` (RX 9060 XT
`8aca3ffa`), `rdna4r-y2` (RX 9060 XT `7058e6f5`). `hashrate.js` 50 s after 15 s, builds taking turns, with a twin.
Every build 191 VGPRs (SOFF 192), 48 KB LDS, no spill.

**Two ideas that the layout rules out (no build):**
- Sharing fragment bytes across lanes (`v_permlane16`, DPP): in gfx12's 16x16x16 layout each fragment byte sits in
  exactly one lane (16 rows x 16 k = 256 bytes = 32 lanes x 8). A wave reads no byte twice.
- A wider wave tile: 64 x 64 already takes 128 accumulator VGPRs. 64 x 128 needs 256; 32 x 128 holds 128 but reads
  10 fragments a k-step, not 8. 64 x 64 reads the fewest LDS bytes a WMMA for its registers.

**The slab loads' cache hints** (`PEARL_AMD_LDAUXA`, timing only; gfx12's temporal hint in the buffer load's policy
word, checked in the ISA as `th:TH_LOAD_NT` and so on), 3 rounds:

| Hint | RX 9070 XT 93afc1b2 | RX 9060 XT 8aca3ffa |
|---|---|---|
| release (0) | 132.41 / 132.75 / 132.79 | 77.09 / 76.78 / 76.61 |
| twin | 132.65 / 132.72 / 132.80 | 76.04 / 76.08 / 76.76 |
| HT (2) | 132.71 / 132.69 / 132.71, level | 76.84 / 76.57 / 76.10, level |
| NT near, RT far (4) | 131.83 / 131.88 / 130.06, -0.7% | 76.30 / 76.28 / 76.27, -0.5% |
| RT near, NT far (5) | 129.6-129.7, -2.3% | 70.0-70.1, -9% |
| NT (1) | 106-108, -19% | 54.3-54.7, -29% |
| LU (3) | 107.7-107.9, -19% | 53.4-54.4, -30% |

The L2 and Infinity Cache reuse of A' and B' tiles is worth up to 30%, and keeping lines longer gains nothing. The
release's policy stays.

**Fewer instructions around the WMMAs** (`goal/rdna4r` 36b688cb, timing only), 2 rounds read:
- `PEARL_AMD_SOFF`: in a C4 chunk the slab's place as the load's immediate and the chunk base as its SGPR offset, so no
  `s_add` a slab (the loop's 7 `s_add_co_i32` went). RX 9070 XT 131.78 / 131.70 against release 132.85 / 132.85 and
  twin 132.80 / 132.79 (-0.8%); RX 9060 XT 76.19 / 76.05 against 77.10 / 76.77, twin 76.84 / 76.77 (-1.0%).
- `PEARL_AMD_WALL`: one LDS wait for all eight fragment reads, then 32 WMMAs unbroken (goal-rdna4f's WGRP=1 again):
  131.97 / 131.91 (-0.6%) and 76.30 / 76.25 (-0.8%). Both together: 130.5 (-1.7%) and 75.4 (-2%).

**What else was checked:** gfx1200 and gfx1201 take no `ds_load_tr_b64`, `v_bitop3_b32` or `v_xor_b64` (and, as
goal-rdna4i found, no load into LDS), so the readout's widest XOR stays `v_xor3`. Salad's `amd-smi metric` (WSL2)
reads GFX_ACTIVITY, the GFX voltage and clock only: UMC_ACTIVITY reads 0% under load and there is no power. So
SRBMiner's energy can't be split from outside beyond goal-rdna4m's clock and voltage.

**The hash in the fold for the REG2 readout** (`PEARL_AMD_FUSE`, shipped on gfx1200). The hash kernel reads the
batch's 128 MiB of transcripts back after each fold (0.24 ms of a 29-34 ms batch on these PCs). REG2's words are in
LDS when the slab loop ends (QLDS: word 2p + b of block region g at `sQ[p * T + 2g + b]`), so after one barrier
threads 0-255 each read one region's 16 words, apply the RFRAME rotation the store applies, and hash it with the
region number the store path gives it. Waves 8-15 have no work. The old FUSE form (keepLo/keepHi through the B'
staging) stays for the other folds. `amdFuse.test.js` checks the new gather's word places and numbers as JS.

| Round | 9070 XT fuse / release / twin | 9060 XT 8aca3ffa fuse / release / twin |
|---|---|---|
| 1 | 132.86 / (131.53, sd 2.2) / 132.82 | 77.24 / 77.00 / 76.82 |
| 2 | 132.87 / 132.79 / 132.85 | 77.20 / 76.74 / 76.79 |
| 3 | 132.87 / 132.77 / 132.82 | 77.15 / 76.74 / 76.74 |
| 4 | (132.12, sd 1.8) / (132.38, sd 1.3) / 132.88 | 77.16 / 76.78 / 76.75 |
| 5 | (132.24, sd 1.7) / 132.82 / 132.75 | 77.18 / 76.79 / 76.77 |

On the 9060 XT fuse led in every round, +0.5%, and the twin read within 0.1% of the release. The 9070 XT read level:
it is power-held, and its hash kernel is a smaller share of its batch. Trace (diag run, clock probe beside it):
9060 XT batch 28.58 ms with the hash in the fold against 29.73 (fold 29.45 + hash 0.244); 9070 XT 34.47 against
33.46. verify-hits 400 of 400 with fuse on both cards (9070 XT 1,511 hits, 193 salts, k 8192; 9060 XT 882 hits, 111
salts). So it is on by default in gfx1200 code only; gfx1201 keeps the hash kernel, and the Windows core leaves it
out as before.

**The shipped commit against the release on two RX 9060 XT PCs** (goal/rdna4r-ship 77c7a4b8, FUSE on by default in
gfx1200 code; the release is the same commit with `-DPEARL_AMD_FUSE=0`, and its twin), 5 rounds:

| Round | 8aca3ffa: ship / release / twin | 7058e6f5: ship / release / twin |
|---|---|---|
| 1 | 77.35 / 76.87 / 76.83 | 73.49 / 72.68 / 71.87 (the PC still warming) |
| 2 | (75.80, sd 1.9) / 76.78 / 76.79 | 72.67 / 71.67 / 71.55 |
| 3 | 77.15 / 76.77 / 76.78 | 72.39 / 71.43 / 71.47 |
| 4 | 77.17 / 76.76 / 76.59 | 72.31 / 71.36 / 71.36 |
| 5 | 77.16 / 76.75 / 76.75 | 72.23 / 71.31 / 71.31 |

+0.5% on 8aca3ffa and +1.2 to +1.3% on 7058e6f5 in every settled round; the release and its twin within 0.2%. The
trace shows where it comes from: 8aca3ffa batch 28.59 ms (fold 28.53) against 29.74 (fold 29.45 + hash 0.243) at
2724 against 2689 MHz; 7058e6f5 30.51 against 30.87 ms at 2571 against 2567 MHz. The fold that stores no transcripts
is itself no slower. verify-hits on the shipped default: 400 of 400 on 8aca3ffa (890 hits, 112 salts, "hash in the
fold") and on 7058e6f5 (847 hits, 108 salts). Earn suite on 77c7a4b8: 3544 passed, 68 skipped, coverage gate held.

**Not reached:** the goal. The RX 9070 XT is unchanged (about 95.5% of SRBMiner 3.8.8 by shown rate); the RX 9060 XT
gains 0.5 to 1.3% on these PCs, from about 96%. What is left on the 9070 XT is energy inside the slab loop
(fragment reads and staging about 3.5% of clock each, goal-rdna4m), and every cut tried there since has lost.

### Salad, 17 November: RDNA 4, B' straight from memory into registers on the RX 9060 XT (goal-rdna4u, branches `goal/rdna4u`, `goal/rdna4u-ship`)

Aim: try the one part of AMD's gfx1201 int8 kernels we had not tried, DirectToVgprB: B' goes from global memory
straight into each wave's registers, and only A' is staged through LDS. goal-rdna4t's instruction count priced it at
about 0.7% of the loop's clock cycles (the 2-chunk loop drops 32 `ds_load_b128` and 8 `ds_store_b128` and gains 24
`buffer_load_b128`: 2277 to 2261 WMMA-clock slots), plus the energy of the LDS traffic. Boxes in llmjob2:
`rdna4u-x1` (RX 9070 XT, `cbb31097`), `rdna4u-y1` (RX 9060 XT, `ec4f1585`, then `ac3f08f2`) and `rdna4u-y2` (RX 9060 XT,
`d684fc25`). `hashrate.js` 50 s after 15 s, builds taking turns, with a twin; clock from the clock probe in the diag
run. Every form was bit-exact in the fold check (the fused hash's hits too).

**The forms** (`PEARL_AMD_GB` on `goal/rdna4u`, 9bf7b6a0; VGPRs on gfx1201 and gfx1200):

| Form | What | VGPRs | Spill |
|---|---|---|---|
| base | B' through LDS (the release) | 191 | none |
| GB=1 | the next slab's B' a slab ahead, in a second register set | 192 | 36 / 60 bytes |
| GB=2 | refilled by halves: k-step 0's after the slab's k-step 0 WMMAs, k-step 1's at the end | 188 | none |
| GB=3 | refilled whole after the slab's last WMMA | 184 | none |
| GB=4 | tiles 0-1 a slab ahead (8 more VGPRs), 2-3 as GB=3, k-step 0 taking tiles 0-1 first | 192 | none |
| GB=5 | GB=3's loads in GB=4's WMMA order | 184 | none |

Two slabs ahead would need 16 VGPRs more than GB=1, which already spills: it can't run at 8 waves a SIMD.

**RX 9070 XT, `cbb31097`** (TH/s, rounds in turn):

| Build | u1 | u2 | Clock (diag) |
|---|---|---|---|
| base | 136.20 / 135.49 / 135.49 | 135.75 / 135.53 / 135.47 | 2343, 2339 MHz |
| twin | 135.80 / 135.54 / 135.48 | 135.59 / 135.49 / 135.47 | |
| GB=1 (spills) | 132.65 / 132.51 / 132.44 | | |
| GB=2 | 132.80 / 132.73 / 132.67 | | 2359 MHz |
| GB=3 | 135.22 / 135.21 / 135.16 | 135.20 / 135.15 / 135.15 | 2358, 2360 MHz |
| GB=4 | | 134.68 / 134.68 / 134.73 | 2364 MHz |

Every form loses on the 9070 XT. GB=3 runs 0.9% higher clock (the card is power-held, and the LDS work it drops was
energy) but does 1.2% less work a clock: -0.3%. The next slab's first WMMA waits on loads issued only at the end of
the last one, and on 64 CUs the four waves that share a B' tile each fetch it. GB=4 and GB=5 give those loads longer,
but break the release's row-by-row wait for A' and lose more.

**RX 9060 XT, `ac3f08f2`** (u2, u3):

| Build | u2 | u3 | Clock (diag) |
|---|---|---|---|
| base | 76.75 / 76.63 / 76.64 | 76.71 / 76.58 / 76.60 | 2681, 2687 MHz |
| twin | 76.71 / 76.65 / 76.62 | 76.68 / 76.59 / 76.58 | |
| GB=3 | 77.05 / 77.00 / 76.99 | 76.99 / 76.96 / 76.96 | 2685, 2691 MHz |
| GB=4 | 76.60 / 76.59 / 76.57 | 76.54 / 76.53 / 76.53 | 2691 MHz |
| GB=5 | | 76.62 / 76.61 / 76.60 | 2686 MHz |

GB=3 +0.45% in all 6 rounds, the twin within 0.05% of base; the clock is the same, so it is work a clock, as the count
said. `ec4f1585` was dropped: it ran 59-62 TH/s with base and twin 3% apart.

**The ship build** (`goal/rdna4u-ship`, 6a6089f6: GB=3's form as `PEARL_AMD_GB`, on by default in gfx1200 code only)
against `-DPEARL_AMD_GB=0` and its twin, 4 rounds:

| PC | Ship | GB=0 | GB=0 twin | Gain | Clock ship / GB=0 |
|---|---|---|---|---|---|
| 9060 XT `d684fc25` | 76.72 / 76.38 / 76.35 / 76.37 | 76.19 / 76.09 / 76.05 / 76.05 | 76.14 / 76.09 / 76.07 / 76.02 | +0.4% | 2675 / 2676 MHz |
| 9060 XT `ac3f08f2` | 77.01 / 76.96 / 76.95 / 76.96 | 76.63 / 76.60 / 76.58 / 76.59 | 76.61 / 76.60 / 76.58 / 76.60 | +0.48% | 2689 / 2689 MHz |

The twins sit within 0.07% (`d684fc25`) and 0.03% (`ac3f08f2`) of GB=0, so both gains are well past the spread.
verify-hits on the ship build: 400 of 400 on `d684fc25` (1289 hits, 0 failed) and on `ac3f08f2` (1293 hits, 0 failed).
gfx1201 code is the same with and without the change (the device assembly at 6a6089f6 matches line for line,
`__hip_cuid` aside), so the RX 9070 XT runs the release's fold.

**What is left.** With B' out of LDS on gfx1200, what the count leaves is the readout's 64 `v_xor3` a chunk (its floor)
and A's LDS path. GA (A' from memory, `goal-rdna4n`) lost 5%, and A' cannot also go direct: the registers are full.
On the 9070 XT the gap is clock (power), and every LDS-for-memory trade tried (GA, GB) costs it more work a clock
than it saves in energy. No further idea from AMD's kernels is left untried.

### Salad, 19 November: the RX 9060 XT against SRBMiner 3.8.9, 128x256 blocks with B' direct, and launch settings (branch `goal/rx9060a`)

Aim: 100% of SRBMiner 3.8.9's shown rate on the RX 9060 XT. Boxes in llmjob2: `g18-rx9060-y1` (PC `d684fc25`, after a
slow `ec4f1585`) and `g18-rx9060-y2` (PC `28ddb67e`), 16 GB, k 8192, batch 128. Ours is main (`0ab629ee`, v0.7.6, the
same AMD core as v0.7.5). rocprofv3 is in the image but can't run: Salad's PCs have no `/dev/kfd`. SRBMiner 3.8.9 exits
at start without `wget` (its dev fee fetch), so the box installs it.

**Ours against SRBMiner 3.8.9 on one PC** (`28ddb67e`; Kryptex turns of 300 s, ours, SRB, SRB, ours; then the stand-in
pool, `verify-pool.js` at 2^210, turns of 540 s, SRB, ours, ours, SRB, credit counted over 480 s from the first minute):

| | Ours | SRBMiner 3.8.9 | Ours % of SRB |
|---|---|---|---|
| Shown, Kryptex (means after 60 s) | 79.52, 78.00 (78.8) | 79.88, 80.77 (80.3) | 98.0% |
| Shown, Kryptex, SRB's own last figure | | 81.80, 82.47 (82.1) | 95.9% |
| Shown, stand-in pool | 76.76, 77.04 (76.9) | 79.81, 79.51 (79.7) | 96.5% |
| Credited, stand-in pool | 72.80, 77.92 (75.4; 1,029 shares) | 76.86, 76.92 (76.9; 1,054 shares) | 98.0% +-4.4% |

Every share of both was accepted (2,321, all k 8192). SRBMiner showed 3.6% more than it was credited. Ours showed 2%
more, but its two credit turns are 7% apart, inside the error. The PC was noisy (`hashrate.js` 77.9-80.0 between
rounds). On `d684fc25` ours read 76.4 shown on Kryptex, 76.4 shown and 78.4 +-3% credited on the stand-in pool
(1,071 shares); SRBMiner didn't run there (no `wget`).

**128x256 blocks of 8 waves, 4 a WGP, with B' straight to registers** (`-DPEARL_AMD_W12_WR=2 -DPEARL_AMD_RDLATE=0
-DPEARL_AMD_WAVES=8`). With GB each wave loads its own B', so this block moves as many bytes a MAC as the 256x256 one,
behind an 8-wave barrier. It has no REG2 or C4 (both need the 256x256 block). Every build bit-exact in the fold check.
`test/sweep.sh`, `hashrate.js` 50 s after 15 s, 4 rounds in turn, `28ddb67e`:

| Build | Rounds (TH/s) | Against base and twin |
|---|---|---|
| base | 78.87 / 80.03 / 77.87 / 78.27 | |
| twin | 78.50 / 79.26 / 79.59 / 78.32 | |
| 128x256, B' direct | 75.18 / 76.26 / 75.86 / 75.65 | about -3.8% |
| the same, `-DPEARL_AMD_FUSE=0` | 75.48 / 76.11 / 75.44 / 76.13 | about -3.7% |
| 128x256, B' through LDS (`-DPEARL_AMD_GB=0`) | 74.20 / 74.17 / 73.43 / 73.76 | about -6% |

Behind in every round. B' direct takes back about 2% of the 128x256 block's loss, but REG2 and C4 are worth more.

**Launch settings on the shipped core** (`d684fc25`, `hashrate.js` 50 s after 15 s, 4 rounds in turn): base 76.39 every
round, twin 76.26-76.33. Column batch 64: 76.24-76.36, level to -0.2%. `PEARL_AMD_K=4096` (batch 256, walk on):
75.89-76.06, -0.5% in every round; at batch 128: 75.87-76.01, -0.6%. A new job every 30 s: 76.16-76.30 at k 8192
(-0.2%), 75.89-76.04 at k 4096, so job switches give no reason to drop k. The shipped k 8192 at batch 128 stays.

**Cost:** two RX 9060 XT boxes, 02:22-03:31 and 03:32-04:58 UTC, about $0.45 in all.

### Salad, 18-19 November: RDNA 4 against SRBMiner 3.8.9, and five cheap tries that lost (goal-rdna4y, branch `goal/rdna4y`)

**Did we fall behind, and when?** Against SRBMiner 3.8.2 and 3.8.6 (9-11 November) the RX 9070 XT read 101-143%.
SRBMiner 3.8.8 (13 November) took its own rate on the 9070 XT from 91.5-105.9 to 136.5-154.8 TH/s while ours stayed
at 121-140, so the rows fell to 89-91% (the 14 November table above). Our rate never fell: the goal-rdna4 notes since
then took ours back to 94-96% (the official table of 15 November). So the drop was SRBMiner 3.8.8's AMD kernel, not
a change of ours. 3.8.9 (17 November) names only RTX 30, 40 and 50.

**Baseline on one PC** (box `g18-rdna4-x1`, RX 9070 XT `b9d88a2e`, llmjob2). Ours is v0.7.5 (main at 94e1e36c), the
CLI built on the box. Kryptex, 3 turns of 300 s each, taking turns, the mean of the 10 s samples after 60 s; then one
180 s turn each with the clock probe beside it.

| | Turn 1 | Turn 2 | Turn 3 | Clock (probe) | TH/s a GHz |
|---|---|---|---|---|---|
| Ours (v0.7.5) | 144.95 | 145.23 | 145.04 | 2527 MHz | 57.4 |
| SRBMiner 3.8.9 | 151.06 | 151.17 | 151.11 | 2562 MHz | 59.0 |

Ours is 96.0% of SRBMiner 3.8.9 by shown rate: it runs 1.4% more clock and does 2.8% more work a clock.

**Credit on the same PC** (the stand-in pool at 2^211, a job every 44 s, 4 rounds of 540 s in ABBA order, credit
window from 60 s in to the end of sampling; every share accepted):

| | Shares | Credited | Shown (same windows) | Credited / shown | Longest gap between shares |
|---|---|---|---|---|---|
| Ours | 8,138 | 149.0 +-1.1% | 145.04 | 1.027 | 1.5-2.2 s |
| SRBMiner 3.8.9 | 7,797 | 142.6 +-1.1% | 146.66 | 0.972 | 9.3-13.1 s |

By credit ours is 104.5% +-1.6% of SRBMiner 3.8.9 on this PC. SRBMiner's shown rate fell to 146.7 on this pool
(151.1 on Kryptex) and dips to 140 in each turn; it is still credited 2.8% less than it shows, about its 2% fee.

**Profile.** rocprofv3 can't run on Salad: there is no `/dev/kfd`, and with libdw1 installed it stops on "sysfs
nodes path '/sys/class/kfd/kfd/topology/nodes' does not exist" (SRBMiner) or a signal 6 (ours). So no kernel trace
or counters of either miner. Our own trace while mining on Kryptex (`PEARL_AMD_TRACE=1`, 300 s): 30.35 ms a batch,
the fold 30.07 ms (99.1%), the hash 0.229 ms (0.75%), idle 0.016 ms, a job's redraw 12-15 ms about every 20 s
(0.06%). So 2.8% more work a clock is all inside the fold's slab loop (or in SRBMiner's own way of counting), and the
1.4% clock is energy.

**Tried** (hashrate.js 50 s after 15 s, 3 rounds, builds taking turns, with a twin; same PC):

| Build | Round 1 | Round 2 | Round 3 | Against base |
|---|---|---|---|---|
| base | 145.50 | 145.35 | 145.22 | |
| twin | 145.22 | 145.33 | 145.32 | -0.1% |
| hash in the fold on gfx1201 (`-DPEARL_AMD_FUSE=1`) | 145.04 | 145.24 | 145.30 | -0.1%, level |
| B' straight to registers on gfx1201 (`-DPEARL_AMD_GB=1`) | 144.67 | 144.71 | 144.70 | -0.45% |
| k 16384 (`PEARL_AMD_K=16384`) | 144.36 | 144.55 | 144.41 | -0.65% |
| hash in the fold at k 16384 | 144.54 | 144.46 | 144.41 | -0.65% |

The fused hash moves the cost, not removes it: the fold went from 29.97 to 30.23 ms a batch as the 0.23 ms hash
kernel went away (30.25 against 30.29 ms a batch, 2526 against 2524 MHz). On the RX 9060 XT it wins because it also
drops the transcripts' store and read; on the 9070 XT that traffic is cheap.

The compiler's scheduler, for the same source (`-mllvm -amdgpu-sched-strategy=...`; each a different code object):

| Build | Round 1 | Round 2 | Round 3 | Against base |
|---|---|---|---|---|
| base | 145.20 | 145.32 | 145.24 | |
| twin | 145.06 | 145.39 | 145.22 | level |
| max-ilp | 145.18 | 145.41 | 145.34 | level |
| max-memory-clause | 145.29 | 145.31 | 145.28 | level |
| iterative-ilp | 143.18 | 143.16 | 143.15 | -1.4% |

**SRBMiner 3.8.8 on the same PC** (one 300 s turn after a turn of ours, which read 144.96): 151.15 TH/s, against 3.8.9's
151.06-151.17. 3.8.9 changed nothing on RDNA 4.

None of these is a win, and nothing ships from this branch.

**Is our shown rate under the work we prove?** The credit run above has ours credited 2.7% over its shown rate,
so this checked whether the display under-counts.
- What the rate counts (pearl_core.cc `collect`): each finished batch adds its regions x 256 x k MACs
  (`PEARL_DAF`, with k the one the card really mines), over steady-clock wall time, reported about every 0.5 s. The
  window has no breaks: the hash kernel, redraws and job switches all count as time. Work counts when its batch
  finishes. The CLI line is the time-weighted mean of those reports (`createRateMeter`). Work before a job switch
  is collected and counted, and its hits are still sent.
- What the pool credits (verify-pool.js): a share passes when its hash is at most target x tile x k, with the tile
  h x w = 256 sums. So each region is a ticket with odds target x 256 x k / 2^256, and a share is worth 2^256 / target
  MACs. Expected credit equals regions x 256 x k, the same count the display uses. The code has no constant
  missing and counts no idle time, so it cannot under-count.
- The stand-in pool's job headers are its sequence number and a fixed byte, and our salts start at 0, so fresh
  runs mine the same shares (goal-rdna4w saw two runs give identical counts). Here each run got its own header byte.

`ratecheck.js` (our core on the stand-in pool, every core tick through the CLI's meter), RX 9070 XT `38eab825`, main
at 0ab629ee, 2^212, two runs of 1800 s, the first 60 s left out (box `g18-rdna4-x2`):

| Run | Core work | CLI meter | Credited (shares) | Credited / shown |
|---|---|---|---|---|
| 1 (header byte 65) | 133.124 | 133.124 | 133.221 +-0.87% (13,170) | 1.0007 |
| 2 (header byte 59) | 133.589 | 133.578 | 133.677 +-0.87% (13,214) | 1.0007 |

Over 26,384 shares, ours is credited 1.0007 x what it shows, +-0.6%. The 2.7% above was share luck (2.4 sd on 8,138
shares). The RTX 2080 Ti's 83.6 and 89.8 credited against 82.6-82.9 shown (18 November) are 215 and 231 shares,
+-7% each, so they say nothing either way. The display is right, and nothing changes.

**Round 3: memory traffic, the tile walk, and what SRBMiner's shares declare** (box `g18-rdna4-x3`, RX 9070 XT
`1d0f121d`, a slow and noisy PC; main at 0ab629ee).
- **DRAM can't be read here.** Salad's `amd-smi metric` (WSL2) gives UMC_ACTIVITY 0% under both miners, no power,
  and no memory clock; there is no sysfs (no /dev/dri). It reads MM_ACTIVITY 13% under ours and 5% under SRBMiner,
  8% under ours on `b9d88a2e`, but that is the media engine's figure and neither miner uses it.
- **What memory past L2 can cost at most** (ablation 16: every block reads tile 0's operands, so they stay in L2;
  timing only), hashrate.js, 3 rounds: base 118.72 / 118.11 / 117.77, twin 117.99 / 118.80, ablation 16 120.73 /
  120.64 / 120.55 (+2.1%). goal-rdna4m read +0.5% on a fast PC. So cutting DRAM traffic is worth at most 0.5-2%.
- **A smaller tile walk** (4 x 4 supertiles in 2 x 2, fewer tiles in flight than the 8 x 8 default): 118.13 / 118.82
  / 117.94, level. goal-rdna4b's 4 x 16, 16 x 4, 8 x 16 and 16 x 8 were level too.
- **What each miner declares in its shares** (the stand-in pool decoding the shares it accepted, 2 minutes each,
  every share k 8192 for SRBMiner):

| Miner | m | n | k | rank | Region (h x w) | Region rows | Region columns |
|---|---|---|---|---|---|---|---|
| Ours | 131,072 | 262,144 | 8192 (k 16384 after its k trial) | 128 | 32 x 8 = 256 | 4 runs of 8, 16 apart | 4 pairs, 16 apart |
| SRBMiner 3.8.9 | 4,096 | 262,144 | 8192 | 128 | 8 x 16 = 128 | 8 single rows, 16 apart | 4 runs of 4, 16 apart |

SRBMiner's region is one row of each of 8 WMMA tiles stacked 16 apart, by 4 adjacent columns of each of 4 tiles
side by side: it spans 128 rows x 64 columns. In gfx12's wave64 accumulator layout that is one accumulator
register of 4 adjacent lanes across 8 x 4 tiles. So a 128 x 64 tile a wave in wave64 (128 accumulator VGPRs, 4 a
WMMA) is the natural reading, though the share can't prove it. That is the tile goal-rdna4j's W64 built (8 x 4
tiles a wave in wave64, its own 32 x 8 regions). W64 ran 1.7% more clock but did 4% less work a clock, -2.3% on the
9070 XT and level on the 9060 XT; goal-rdna4k's barrier variants of it also lost. SRBMiner's m of 4,096 keeps all of
its A' (32 MB) in the 64 MB Infinity Cache, where ours keeps a batch's B' (32 MB) and streams A'. For the same work
both read about the same bytes from DRAM, so m alone should not matter; it was not raced. The next lever these
point to is a wave64 128 x 64 fold whose barrier and readout cost less than W64's, with SRBMiner's 8 x 16 regions:
4 lanes a region, which a quad DPP step can combine.

**Round 4: the wave64 128 x 64 fold again** (box `g18-rdna4-x4`, RX 9070 XT `68cf68ff`). goal-rdna4j's
`PEARL_AMD_W64` (24901e82) carried onto this branch unchanged as 639cf859, off by default
(`-DPEARL_AMD_W64=1 -mwavefrontsize64`). Its fold check passed (all 16 words of 4096 regions), and verify-hits passed
400 of 400 (1,127 hits, 136 salts, k 8192). hashrate.js, 3 rounds:

| Build | Round 1 | Round 2 | Round 3 | Clock (diag) | Batch (trace) |
|---|---|---|---|---|---|
| base | 130.72 | 130.51 | 130.49 | 2266 MHz median (noisy run) | 33.70 ms |
| twin | 130.61 | 130.50 | 130.46 | | |
| W64 | 127.62 | 127.64 | 127.66 | 2298 MHz | 34.45 ms |

W64 is 2.2% behind on today's fold, as on the fold of 16 November (-2.1%): about 1% more clock and 3.5% less work a
clock. So SRBMiner's tile alone is not where its lead comes from.

SRBMiner's 8 x 16 regions would not make W64's readout cheaper either. A 128-sum region puts 4 regions in each lane
(one an accumulator register), each the XOR of 32 sums, and a quad step between 4 lanes: about 64 XORs and 6 more
VALU a chunk, against the 2-lane 256-sum regions' 62 `v_xor3` a chunk. It also doubles the number of regions, so
the hash does twice the work (0.24 ms a batch here, 0.7%). W64's loss is in work a clock, not in the readout, so the
8 x 16 readout was not built.

**Cost:** four RX 9070 XT boxes, high priority, llmjob2, 00:03-03:09, 03:14-04:17, 04:20-04:46 and 04:50-05:16 UTC on 19 November (about $0.99).

### Arc rig, 10 November (PR #290's SYCL core; IAminer 1.0.0, SRBMiner 3.8.6 and 3.8.2, WildRig 0.52.1 and 0.51.9)

The rig: six Arc A750 (DG2, 8 GB) in the x1 slots of a mining board (PCIe 2.5 GT/s x1 each), an i5-10400T, 7.8 GB
of RAM, HiveOS 0.6-231 on Ubuntu 22.04, kernel 6.12.0-hiveos, the cards on i915. Everything ran in two Ubuntu
24.04 containers given `/dev/dri`, each with Intel's compute runtime 26.40.40196.15: ours in one, the other miners
in the second. Downloads were checked against the SHA-256 digests GitHub lists for each release asset; IAminer's
also against its SHA256SUMS.txt and SRBMiner's against its published MD5.

One card (card 0), turns on Kryptex (prl-us:7048): ours 5 minutes, IAminer 6, alternating. TH/s is the mean after
the first minute: ours from its status line every second, IAminer from its `speed` line every 30 s. Power and clock
are i915's counters for the card (`hwmon/energy1_input`, `gt_act_freq_mhz`), read every 5 s and averaged from 60 s
to 285 s. Kryptex asked 9.01 PH MACs a share, so one card finds a share about every ten minutes and the share
counts here are small.

| Miner | Turn 1 | Turn 2 | Mean | Power | Clock | Shares (acc/rej) |
|---|---|---|---|---|---|---|
| Ours as first built (band 64) | 15.45 | 15.47 | 15.46 | 146 W | 2000 MHz | 1/0 |
| IAminer 1.0.0 (m = n = 131072, k 4096) | 13.64 | 13.64 | 13.64 | 173 W | 2018 MHz | 0/0 |
| Ours as pushed (band 256), one turn after the others | 16.88 | | 16.88 | 153 W | 2000 MHz | 2/0 |

Ours was 113% of IAminer as first built and is 124% with the band change found on this rig, at 12% less power:
about 40% more work a watt. IAminer's 30 s figures alternate between 11.72 and 14.07: a whole number of its
attempts (one attempt is the full m by n product at k 4096, 7.0e13 MACs) fits each window.

All six cards, 5 minutes for ours and 6 for IAminer, each card at the settings above (IAminer's shape copied to
the other five cards' entries in its config, as its README allows):

| Miner | TH/s, six cards | Range | Power, six cards | Shares (acc/rej) |
|---|---|---|---|---|
| Ours as first built (band 64) | 91.9 | 87.2–96.7 | 875 W | 5/0 |
| Ours as pushed (band 256) | 100.1 | 96.9–103.2 | 921 W | 4/0 |
| Ours as pushed, on the branch tip (with the core clock offset work merged in) | 100.1 | 96.7–104.0 | 919 W | 3/0 |
| IAminer 1.0.0 | 78.0 | 68.0–84.4 | 1,012 W | 0/2 |

- **Ours: 15 shares accepted, 0 rejected** over the six pool runs above, where about 11 were expected from its
  rate. Every hit was recomputed before it was sent, as the core's gate for untested cards makes the CLI do, and
  none was wrong.
- **IAminer: 0 accepted, 2 rejected.** Both came from its GPU4, 16 s apart, 85 s into the six-card run, and Kryptex
  answered "share rejected: Invalid share" to each. It found none in its 12 minutes on card 0. From its rate
  about 4 shares were expected over its three runs. Two shares cannot say whether the fault is that card, the
  shape copied from card 0's tuning, or the miner. It was not run on the pool again, so that no more rejected
  shares went in under the address. Until it has an accepted share here, its TH/s is a rate of work whose
  results are unproven.
- **IAminer's tuning** on the card (`iaminer tune`), TH/s: m 65536 with k 4096, 8192 and 16384: 13.32, 9.51 and
  7.71; m 131072 with k 4096 and 8192: 13.55 and 9.82. At m 262144, k 4096 it used one CPU core for five minutes
  with the card idle and was stopped; `iaminer tune --quick` (the two m 131072 shapes) gave the same best. Its
  README's 70.3 TH/s is an Arc Pro B70 at k 8192. On the A750 a larger k is slower for it. Its 1% dev fee (60 s
  every 100 minutes) did not fall in these runs.

Ours alone, before the comparison (`earn/native/sycl/README.md`, "Results on an Arc A750 rig", has it all):

- **Correct on all six cards.** The hardware DPAS fold and the plain fold match a scalar fold on the host on every
  region checked, at a small size and at the mainnet m and n. 3,000 hits were recomputed in JS as a pool's verifier
  does (1,500 a fold) and none failed.
- **One thing is wrong and mines nothing:** the reference code of the sub-group 8 fold (`xmx8-emu`, the
  specification's DPAS written out, which no card mines with) gives other transcripts on the A750, on all six
  cards. It is right on the CPU device. The cause is not found. `verify.sh` counted it and so failed a good card;
  it now reports it and counts the folds that mine.
- **Fold alone and miner loop** (`bench.sh`, card 0): 15.47 and 15.47 TH/s as first built; 16.87 and 16.88 as
  pushed. The plain int8 fold does 3.9 to 4.0 TH/s at 185 W.
- **Band.** `PEARL_SYCL_BAND` 4, 16, 64, 128, 256, 512, 1024, 4096: 14.1, 14.3, 15.2 to 15.5, 16.3, 16.6 to 16.9,
  13.8, 13.4, 13.5. The host took a quarter of the card's 16 MB cache, 64; it now takes all of it, 256, for the XMX
  folds. The plain fold stays at a quarter (3.92 at 64 against 3.75 at 256). The batch width makes no difference
  (64 to 1024: 15.44 to 15.48).
- **Power.** Under DPAS the card sits at 2,000 MHz and about 150 W, under its 190 W limit. An idle card draws 31
  to 36 W.

Other things this run found:

- SRBMiner prints nothing unless it has a terminal: `--list-devices` into a file or pipe wrote 0 bytes and exited
  0. Under `script` it listed the cards. SRBMiner 3.8.6 also quit at once with exit 73 until `wget` was installed
  ("Make sure you have 'wget' installed on your system!"). `earn/tuning/salad/bench.sh`'s note about 3.8.6 exiting
  73 on Salad may be the same two things.
- Not tried: BzMiner, which Kryptex's miner page lists for Pearl on Intel Arc, and mo-miner, whose engine IAminer
  uses. Neither was on the list for this run.

### Arc A750, 10 November: the XMX peak, and where the fold's time goes

What the card's matrix engines do with nothing in the way, as `pearl_amd_peak` asked of AMD's. On paper: 448 EUs x
128 int8 MACs a clock x 2.0 GHz, about 115 T-MAC/s. `earn/native/sycl/pearl_sycl_peak.cpp` runs the hardware DPAS
in the fold's own grid and thread shape and puts the fold's parts back one at a time. Card 0 of the rig, 4 s a
row; clock and watts are i915's counters; this kernel's i915 reports no temperature for Arc. The probe checks
nothing: only the last row is a fold.

| What runs | T-MAC/s | Clock, power |
|---|---|---|
| DPAS alone, operands in registers | **113.4** (126 MACs an EU a clock) | 2,012 MHz, 166 W |
| The same for 120 s | 113.4, 113.2 to 113.5 each second | 2,000 MHz, 177 W |
| On 1 of the 8 accumulators | 91.2 | 2,000 MHz, 165 W |
| With the fold's thread starts (16 times as many) | 112.6 | 1,998 MHz, 170 W |
| And the fold's readout, barrier, hash and test | 110.1 | 2,017 MHz, 177 W |
| A' loaded as the fold loads it, from one window that stays in the cache | 104.5 | 1,924 MHz, 184 W |
| B' the same way | 56.6 | 1,998 MHz, 185 W |
| Both, from one window | 39.8 | 1,995 MHz, 166 W |
| Both, from the fold's own addresses | 16.9 | 2,019 MHz, 148 W |
| The same loads with 7 of every 8 DPAS taken out | 16.9 (counted as if all ran) | 2,019 MHz, 138 W |
| The fold | **16.9** | 2,019 MHz, 151 W |

- **The ceiling is 113 T-MAC/s, and the fold is at 15% of it.** Intel's paper figure holds.
- **All of the gap is operand loads.** With 7 of every 8 DPAS taken out the rate is the same, so the matrix engines
  are idle 85% of the time. Thread starts, the accumulator chain, the readout and the hash cost about 3% together.
- **The loads cost twice.** The fold sends 3 load messages for each DPAS, which caps it at 39.8 even from memory
  that never leaves the cache. From its own addresses it falls to 16.9: A' and B' together do not stay in the
  card's 16 MB cache.

What a different fold could reach, timed with the readout, hash and test in place. The bits of these rows are not a
fold's, so they are estimates:

| Loads | 8 rows a sub-group, 128 registers (today's shape) | 16 rows a sub-group, 256 registers |
|---|---|---|
| The fold's | 16.9 | 21.6 |
| The fold's, at the addresses A' and B' would have in DPAS fragment order | 22.9 (31.4 at band 64) | 22.9 |
| One vector load a fragment, B' where it is | 18.6 | 33.8 |
| One vector load a fragment, both in fragment order | 22.8 | 38.0 (**42.7** at band 32 or 64) |

- **The target for the next fold on this card is about 40 TH/s**, 2.5 times today's: 16 rows a sub-group in the
  256-register mode (which costs nothing at the peak: 113.6), A' and B' stored in fragment order, one vector load a
  fragment, and the band tuned again.
- Past that the loads still limit: the best row leaves the matrix engines idle 62% of the time.
- Card 3 gave the same 128-register rows within 2%. Not timed: staging B through local memory, another card
  model, a run over two minutes.

### Arc rig, 10 November: the build a release would carry, on two Intel drivers (branch `goal/arc-ship`)

The same fold in the form CI stages for a release (`earn/native/sycl/README.md`, "The build a release carries"):
built with the open-source DPC++, its kernels compiled by the rig's own driver at first start, the hardware DPAS
called as an OpenCL built-in. Ours only; no other miner ran. One card, fold alone for 15 s, unless a pool is named.

| Where it ran | Intel driver | TH/s | |
|---|---|---|---|
| Ubuntu 24.04 container | Level Zero 26.40 | 16.81 | within 1% of the prebuilt kernels' 16.9 |
| The same | OpenCL 26.40 | 16.78 | |
| Ubuntu 22.04 container | OpenCL 24.35 | 10.13 | |
| HiveOS's own userland (Ubuntu 22.04), no container | OpenCL 24.35, no Level Zero | 10.2 | the plain fold: 2.4 |
| The same, six cards on Kryptex, 5 minutes | | 60.6 | 916 W; 2 shares accepted, 0 rejected |
| The same, the CLI given no core path and no card list | | 60.6 | it found the six cards; 2 accepted, 0 rejected |
| The same, the HiveOS package's own binary | | 60.6 | 1 accepted, 0 rejected |
| The same, the HiveOS package CI built, its own binary, 5 minutes | | 60.7 | 912 W; 2 accepted, 0 rejected |

- **The driver is worth 40%.** HiveOS has Intel's 2024 OpenCL driver (24.35.30872, compiler 1.0.17537). Its
  compiler sends about 8 load messages for each DPAS where the 2026 one sends 3, at the same power: six cards do
  60.6 TH/s at 916 W here against 100.1 at 921 W in the container.
- **No % of best for these rows.** IAminer's 13.6 TH/s was measured on the 2026 driver and it was not run on the
  2024 one. On a HiveOS rig as it comes, ours at 10.2 is below that figure.
- Nothing was installed on the host. The package's binary was run by hand from a folder under the user's home, not
  through a HiveOS flight sheet.

### Arc rig, 11 November: the fragment fold, 42 TH/s a card (branch `goal/arc-kernel`, not on the PR)

The founder's goal for Intel, 11 November: 40 TH/s a card on an A750, then 100. The first is met. The fold is
`fold_xmxf` (`xmx8f`): 16 rows a sub-group in the 256-register mode, A' and B' stored in DPAS fragment order, one
vector load a fragment, a band of 32 row windows. It is the build a release would carry (open-source DPC++ v7.2.0,
kernels compiled by the driver), run in the Ubuntu 24.04 container with Intel's 26.40 driver. Clock and watts are
i915's counters. IAminer was not run again; its figures are the 10 November ones (13.6 TH/s a card, 78.0 for six).

| What | First fold (`xmx8`) | Fragment fold | Fragment fold, twin |
|---|---|---|---|
| Card 0, the fold alone, 20 s, taking turns, round 1 | 16.87 TH/s, 2,000 MHz, 150 W | 42.22, 2,000 MHz, 166 W | 42.25, 2,011 MHz, 167 W |
| round 2 | 16.88, 2,011 MHz, 154 W | 42.22, 2,000 MHz, 169 W | 42.25, 2,011 MHz, 170 W |
| round 3 (card hot, clock falling) | 16.87, 2,011 MHz, 157 W | 40.76, 1,922 MHz, 160 W | 39.19, 1,850 MHz, 152 W |
| Six cards on Kryptex, 5 minutes, mean after the first | 100.08 (98.0 to 102.2), 924 W, 2,001 MHz, 1 accepted, 0 rejected | 251.42 (249.3 to 253.8), 1,002 W, 2,000 MHz, 9 accepted, 0 rejected | 245.14 (239.5 to 252.2), 977 W, 1,948 MHz, 6 accepted, 0 rejected |
| Five cards on Kryptex, 40 minutes | | 201.6, 40.3 a card, 52 accepted, 0 rejected | |

- Hits: `verify.sh --quick` passed on each of the six cards and the full `verify.sh` on card 0. No hit came out
  wrong and no share was rejected in any run.
- 41.9 TH/s a card with shares accepted, against 16.7 for the first fold beside it: 2.5 times, at 167 W against
  154 W. A twin on a cool card is within 0.1%.
- The rig does not cool six cards at 167 W for long. In the third six-card run in a row cards 0, 1 and 2 lost
  clock (card 0 to 1,750 MHz), and after 40 minutes with five cards, cards 1, 2 and 3 were at 1,800, 1,850 and
  1,950 MHz. The rate follows the clock. No temperature reading exists for Arc on this kernel.
- Not measured yet: the rig's own HiveOS userland with the 2024 driver, where the first fold did 10.2 TH/s.

Toward 100 (`pearl_sycl_peak --frag`; timing only, these compute wrong results on purpose and none is a mining
rate; T-MAC/s, and the same per EU and clock in brackets where the card dropped its clock at its 190 W limit):

| Row | T-MAC/s |
|---|---|
| DPAS alone, 16 accumulators | 113.8 |
| the fold's loads and DPAS, no readout | 42.8 |
| every window reading window 0; every load the same fragment | 43.3; 43.3 |
| A' loaded, B in registers; B' loaded, A in registers | 100.7; 50.1 |
| all ten loads of a k-step sent first, the 16 DPAS as one unbroken run | 43.4 |
| the ten loads kept by an XOR, the DPAS on register operands | 42.2 |
| no loads, every operand rewritten by an add each k-step | 106.9 |
| B' as two 128-byte halves a fragment, sent first; A in registers | 80.5 (96.1) |
| the same with A' loaded as one vector | 61.4 (71.6) |
| both operands as halves | 54.8 (61.2) |
| 32 rows by 32 columns a sub-group, one vector a fragment | 57.3 (63.9) |
| 32 by 32, halves, sent first | 54.3 (60.5); 77.6 (89.1) with every window reading window 0 |

- With 256-byte loads the limit is the load message itself: the card serves about 1.3e10 of them a second
  whatever the addresses, the order, or the use of the result. Designs that only reorder loads and DPAS, or only
  improve the cache, change nothing.
- A 128-byte load is served about four times as fast, so the next fold loads halves. Then two other limits show:
  a thread can have 16 loads in flight (the hardware's scoreboard tokens), and A', which no other sub-group
  shares, costs more than B'.
- Tried and dropped: one unbroken run of DPAS a k-step with 256-byte loads (43.4 against 42.8); a band's windows
  in one work-group (no change); each lane's bytes in a cache line of its own (half the rate); one vector of 4
  qwords a fragment (24 to 34).

### Arc rig, 11 November: the fragment fold with B' loaded in halves, 64 TH/s on a cool card (`goal/arc-kernel`)

The second version of the fold (commit `08e4ae8e`): a fragment of B' is loaded as two vectors of 4 dwords, a
k-step's 18 loads are sent before its 16 DPAS, and the band is 64. The stored order does not change. Same build
and driver as above. "First version" is the fold of the section above (`0308ff15`).

| What | First fold (`xmx8`) | Fragment fold, first version | Fragment fold now | Now, twin |
|---|---|---|---|---|
| Card 0 cool, the fold alone, 15 s, taking turns | 16.86 TH/s, 2,000 MHz, 149 W | 42.24, 2,000 MHz, 169 W | 63.91, 1,965 MHz, 189 W | 63.70, 1,954 MHz, 189 W |
| a minute later, the card hot | | 40.23, 1,900 MHz, 157 W | 61.01, 1,856 MHz, 178 W | 57.03, 1,741 MHz, 160 W |
| Six cards on Kryptex, 5 minutes each, one after the other on a hot rig, mean after the first minute | | 240.20 (236.2 to 247.7), 949 W, 1,908 MHz, 8 accepted, 0 rejected | 349.86 (329.0 to 371.9), 1,035 W, 1,810 MHz, 12 accepted, 0 rejected | 329.74 (322.9 to 343.6), 958 W, 1,704 MHz, 17 accepted, 0 rejected |
| The rig's own HiveOS userland (2024 driver), card 0 warm, 15 s | 9.14 | 39.41 | 54.89 | 54.88 |

- Hits: `verify.sh --quick` passed on each of the six cards (61.2 to 63.4 TH/s with every hit checked) and the
  full `verify.sh` on card 0. No hit wrong, no share rejected.
- 55 to 58 TH/s a card with shares accepted, against 40 for the first version in the same hour: 1.4 to 1.5 times.
  On a cool card 1.5 times, and 3.8 times the first fold.
- The fold now runs a card at its 190 W limit, and this rig cannot shed that: within three minutes cards 0 to 3
  were at 1,550 to 1,650 MHz and 138 to 152 W, card 4 at 1,800 MHz, and only card 5 kept 1,950 MHz. The 6% between
  the two runs is that. A rig with spaced cards should hold about 64 a card; this was not measured.
- The first version over an hour on five cards (11:09 to 12:08 UTC): 36.5, 37.6, 39.7, 42.0 and 41.9 TH/s, 39.6 a
  card, with the three inner cards at 1,700 to 1,900 MHz. 81 accepted, 0 rejected.
- Per EU and clock the fold is at 58% of the matrix engines' peak. What the probe says is left: A' as two
  256-byte loads is a third of the load time, and the card's power limit now binds as well as its loads.

### Arc rig, 11 November, afternoon: the fold written without loops on six cards, and what was timed toward 100 (`goal/arc-kernel`)

**The head, `900c154c`** (the fold of the section above with its k-step written out, so the prebuilt form is fast
too). Release build form: open-source DPC++ v7.2.0, kernels compiled by the driver (Intel 26.40, Level Zero).

| Check or run | Cards | Result |
|---|---|---|
| `verify.sh --quick` | all six | PASS on each; 63.7, 61.6, 61.6, 62.3, 62.3, 60.8 TH/s with every hit checked |
| full `verify.sh` | card 0 | PASS, 400 of 400 hits with the fold and 400 of 400 with the plain fold |
| Kryptex, 5 minutes, mean after the first minute (13:29 to 13:34 UTC) | six | 347.53 TH/s (326.8 to 372.2), 57.9 a card; 1,810 MHz, 171 W a card; 16 accepted, 0 rejected, no wrong hit |
| the version before it, `08e4ae8e`, straight after (13:34 to 13:40) | six | 325.89 TH/s (318.2 to 340.9), 54.3 a card; 1,684 MHz, 157 W a card; 14 accepted, 0 rejected, no wrong hit |

Per MHz the two runs are the same (0.192 and 0.194 TH/s); the second ran on a hotter rig. Five cards then mined
with the head for 41 minutes: 265 TH/s, 53 a card, 81 accepted, 0 rejected.

**Toward 100 TH/s a card.** 100 is 112 MACs an EU a clock at 2,000 MHz; the matrix engines alone do 125 and the
fold 73 to 77. Six designs were timed with `pearl_sycl_peak --frag` before building any. Timing only: these rows
compute wrong results on purpose, and none is a mining rate. Card 0, hot, 2 s a row, MACs an EU a clock.

| Design | From the fold's addresses | With every operand in one window (192 KB) |
|---|---|---|
| the fold's loads as built | 72.9 | 93.5; 98.6 in a later run |
| 1. every fragment as two sub-group block reads of 128 bytes (one address a message) | 70.7; 73.4 with A' as the fold loads it; 73.7 on a 32 by 32 tile | 107.1; 108.2 on the 32 by 32 tile |
| 2. every fragment as one block read of 256 bytes | 56.1 | 68.3 |
| 3. a work-group that fills an Xe-core and shares operands: in step; a k-step apart | 58.4; 66.7 to 69.6 | 91.3 in step |
| 4. a chunk's cache lines read first, a barrier, then its 4 k-steps | 55.1; 57.6 on 32 by 32 | 71.0; 77.1 (the barrier alone) |
| 5. the prefetch extension 2 k-steps ahead; one-byte loads a k-step ahead (their row without either: 64.7) | 58.5; 59.4 | |
| 6. operands made in registers: a loop of 16, 64, 128 adds a k-step beside the loads (without: 97.6) | | 84.7, 69.3, 54.3 |
| the compiler's cache control on every load: first level off; streaming; invalidate after read | 38.7; 51.8; 47.9 | |
| B' sent before A' | 75.0 | |

- No design moves the rate from real addresses, so none was built. The block read is the cheapest load when its
  lines are near (107 against 93.5 to 98.6) and no better when they are not.
- Each Xe-core (16 EUs, 64 threads) has a first cache level of about 256 KB. One window's operands, 192 KB, fit;
  two row windows of A' with one of B', 256 KB, give 88.9; with that level switched off the block reads do 38.7.
  One window in 16 reading other bytes is enough to drop 107 to 84.9, and from there the amount hardly matters:
  everything that misses it lands between 70 and 85, the fold included.
- A core's 64 threads hold 64 different tiles, which read at least 4 new cache lines a tile and k-step however
  they are dealt. So no deal of tiles stays inside that cache, and the two that come nearest (rows 3 and 4) lose
  more to running in step than they gain.
- The prefetch extension compiles to a load that names the last cache level only, whichever hint is given. A
  cache hint on A' alone (`annotated_ptr` with a `read_hint`) did not reach its loads, so it was not timed.
- A' cannot be made in registers: its noise table for a sub-group is 64 registers on top of 208 live of 256, and a
  k-step needs 128 byte selects. B''s table is 8 KB, all 256 registers.
- What is left is outside the kernel: the card holds 2,000 MHz under DPAS though its top is 2,400 (not touched),
  and a rig that sheds 190 W a card holds 64 TH/s where this one holds 54 to 58.

### Arc rig, 11 November, evening: heat, a load cycle, and the sampler as a second way in (`goal/arc-next`)

The rig froze at 16:38 UTC after three hours with five cards on the fold (267 TH/s), and its owner restarted it at
19:04. This round mined again with less load and a guard beside the miner. Cards are numbered in PCI order: card 0
(`03:00.0`, on the CPU's own port) is the one closest to the CPU, card 5 (`17:00.0`) the one furthest, with open
air in front of its fans. Development ran on card 5, mining on cards 4 inward, card 0 stayed idle.

**What the rig tells without root.** Each card's clock, the driver's throttle flags (thermal, PROCHOT, power
limits, regulator alert), fan rpm, an energy counter and the 190 W power limit; the CPU's temperature. No card
temperature in degrees: i915 on kernel 6.12 has no temp file for Arc, and HiveOS shows 0. The fans ran 1,600 to
2,100 rpm under load (2,223 once), so the thermal flag is the only sign that a card is at its limit.

**The fold at full load is more than an inner card can shed.** Kryptex, worker `arc750`, the verified head
`900c154c`. Rate, clock, watts and fan are a card's, one sample a minute.

| Cards mining (UTC) | Load | Rate | A card | What happened |
|---|---|---|---|---|
| 3 and 4 (19:14 to 19:21) | 100% | 127 TH/s falling to 122 | 63.2 and 63.3 at 2,000 MHz, 190 W, fans 2,100 and 2,040 | card 3 raised the thermal flag in minute 5 and fell to 1,880 MHz and 60.9; the guard stopped the miner |
| 2 and 4, not neighbours (19:23 to 19:27) | 100% | 125 falling to 114 | 62.3 and 63.4 at 190 W | card 2 went to PROCHOT at minute 4.7: 1,230 MHz, 153 W, 40.3 TH/s; the guard stopped the miner |
| 4 alone (19:30 to 19:33) | 100% | 63 | 63.5 at 1,970 to 2,000 MHz, 187 to 191 W, fan 2,080 | no flag in 3 minutes (10 with the two runs above) |
| 3 and 4 (19:37 to 20:00) | 60% | 74.2 to 75.6 | 37.5 at 126 to 131 W; fans 1,940 to 2,020 and 1,600 to 1,720 | no flag in 23 minutes; 14 accepted, 1 rejected (stale) |
| 2, 3 and 4 (20:00 to 20:21) | 60% | 111.9 to 113.3 | 37.5 at 127 to 133 W; fans 2,050, 1,900 to 1,990, 1,450 to 1,750 | no flag in 21 minutes; 12 accepted, 0 rejected |
| 1, 2, 3 and 4 (20:21 to 20:41) | 60% | 148.6 to 151.2 | 37.5 at 127 to 132 W; fans 2,050 to 2,120, 2,030 to 2,090, 1,920 to 2,000, 1,650 to 1,800 | no flag in 20 minutes; 17 accepted, 0 rejected |
| 1, 2, 3 and 4 (20:42 to 20:55) | 60%, card 4 at 75% | 159.3 to 160.8 | 37.5 at 129 to 133 W, card 4 47.3 at 163 to 167 W; fans 2,050 to 2,120, 2,030 to 2,060, 1,970 to 2,000, 1,870 to 2,030 | no flag in 13 minutes; 14 accepted, 0 rejected. The guard then stopped the miner over one failed read of idle card 0's fan or energy file; restarted 21:00 |

The 60% is a new switch in the host, `PEARL_SYCL_DUTY`: each batch waits for the one before it, then the card
rests for the remainder. The fold is the same one. It costs rate in proportion (37.5 TH/s is 60% of 63) and a
little more per watt, because a resting card still draws 30 W: 0.29 TH/s a watt against 0.33 at full load. A power
limit would do better, since a lower clock runs at a lower voltage: the cards that throttled by heat this morning
did about 51 TH/s at 138 to 152 W, 0.35 a watt. The limit (`power1_max` in the card's hwmon folder) can only be
written by root, which this login is not.

**The guard** (a shell loop in the container, one line a card a minute to a log; it stops the miner and never
starts one, and ends when the miner does): thermal flag in most of a minute's five samples for 2 minutes in a row;
PROCHOT or the regulator's alert for one; CPU over 85 C; any reading missing. It tripped rightly at 19:21 and
19:27, and wrongly twice: at 19:35 on a clock rule that the load cycle fools (now off under a load cycle), and at
20:55 on a single failed read of an idle card's file, which read again at once (a read is now tried three times).

**Card 5 copies wrong bytes back to the host when it is hot.** The fold check at the mainnet size (32,768
regions against the host's reference) came out with 4 to 80 regions mismatched on card 5 in 6 of 21 runs at first,
then in 0 of 10 with the card rested, then in 12 of 13 straight after 75 s at full load. The plain fold and the
fragment fold were wrong by the same count each time, with the verified build as well as the new one. A test that
fills 512 MB on the card, copies it to the host and compares, then compares again on the card, says where: rested,
0 words wrong in 8 rounds; hot, 16 to 608 words wrong in the host's copy in 6 of 6 rounds (whole 64-byte lines, one
of them all zeros) and none wrong on the card itself. So the card's memory and the folds are right, and it is the
copy back to the host that fails, which is what the check builds its reference from. The driver showed no thermal
flag on card 5 and its fan stayed at 900 to 1,600 rpm at 190 W, where the others run 2,100. Cards 1 to 4, tested
the same way straight after 20 minutes of mining at 60%: 0 words wrong in 2 rounds each. Cards 0 and 1 rested:
the check passed 18 of 18 and 8 of 8. For mining it matters little (a proof reads back a few KB a hit), but
`verify.sh` on a hot card 5 can fail with nothing wrong in the fold. Check new folds on another card.

**The sampler.** Each Xe-core has a sampler beside its data port. `pearl_sycl_peak --frag` got rows that read
part of an operand as texels of an image (16 bytes a texel, 128 bytes a message, like a half-fragment load).
Timing only, as above; card 5, 2 s a row; the image is a bindless one, filled from host memory and read back
right (134,217,728 words, none wrong).

| Row | Through the sampler | MACs an EU a clock | T-MAC/s, clock, power |
|---|---|---|---|
| `i-base` | nothing (the fold's loads) | 74.9 to 77.0 | 65.3 to 65.5, 1,900 to 1,945 MHz, 178 W |
| `j-b1` | 1 of a k-step's 8 fragments of B' | 76.0 to 78.1 | 66.1 |
| `j-b2` | 2 of the 8 | 80.8 to 82.5 | 67.4 to 67.7, 1,830 to 1,868 MHz, 176 to 192 W |
| `j-b3` | 3 of the 8 | 77.6 to 78.9 | 64.5 |
| `j-b4`, `j-b` | 4 of the 8; all 8 | 59.9; 21.5 | 51.7; 19.7 |
| `j-a` | A', from its own texels | 71.2 to 73.1 | 60.2 |
| `j-ab` | A' from texels every sub-group of a band shares (not a real A') | 97.7 to 100.1 | 80.7 to 80.9, 1,800 to 1,850 MHz, 176 to 178 W |

- This is the first design timed above the built fold from its real addresses. The sampler takes about 4 fetches
  a k-step for nearly nothing, but only of texels that many threads fetch at once: a column window's B', which the
  128 sub-groups of a band read one after another. A third fragment loses the gain. A', a sub-group's own, is
  slower through the sampler than through the data port, and small bands (where neighbours share A') are worse for
  every row.
- At the 190 W limit it is 3.5% more rate (67.6 T-MAC/s against 65.3 to 65.5 in each of four runs): the card does
  more a clock and then runs at a lower clock. A fold with it needs a bindless image of a quarter of B' (128 MB)
  refilled at every draw; the copy from host memory ran at 150 MB/s. Not built.
- Rows on the old SYCL image class (`i-`) came out at 89 to 97 a clock, on images that could not be filled (a
  kernel that writes one hangs the card). They are wrong for that reason and are not figures.
- Tried in the fold itself and dropped, 12 s runs taking turns with the built fold on card 5 (62.7 to 62.9 TH/s
  at 1,975 MHz, 189 W): every DPAS waiting for both halves of each B' fragment 60.0 to 60.2; for the second halves
  only 60.6 to 60.8; for the first four fragments only 61.6 to 61.7; B' loaded before A' 57.1 to 57.2 (its stand-in
  had said 75.0 against 72.9: a stand-in does not settle a difference that small).
- A k-step's parts, as the time it takes (16 is the DPAS alone): A' 3.4, B' 8.2, and a k-step of 20 load messages
  7.7 more than one of 18 (a thread has 16 in flight at most).

### Arc A750, 11 November, night: a price list, and what holds the card (`goal/arc-4`)

Timing only (stand-ins in `pearl_sycl_peak --frag`, wrong results on purpose), card 5 of the rented rig, the
coolest, 2 to 3 s a row. Skeleton `p-base`: the built fold's loads in one flat loop, 66.0 to 67.2 T-MAC/s at
1,850 to 1,936 MHz and 178 to 191 W (the power limit), 76 to 80 MACs an EU a clock. Cards 1 to 4 mined
throughout (160 TH/s, guard quiet); nothing new was built, so there is no new mining rate.

| Added to every k-step | MACs an EU a clock |
|---|---|
| 2, 4, 8 more loads that hit the first cache level (block reads, vectors of 4 or 8 dwords, sampler fetches) | 74.6 to 77.8 |
| 2, 4, 8 more 128-byte lines from L2 | 63.0, 52.0, 34.4 (178, 164, 141 W) |
| 2 or 8 local memory loads; stores (2 KB a group) | 64.2, 64.1; 61.6, 54.6 |
| 16, 32, 64 adds | 75.9, 70.4, 68.7 |
| 1, 2, 4 barriers | 66.6, 59.0, 53.4 |

- What holds the fold a clock: first-level cache misses, about 3.8 core clocks a 128-byte line from L2, about
  950 bytes of L2 a sub-group k-step. Not the data port: hot loads cost nothing.
- What holds it in rate on a cool card: the 190 W limit. Sampler 2 of 8 (`j-b2`): +4% a clock, +0.6% in
  T-MAC/s (67.6 against 67.2). Every operand hot, 32 by 32 tiles, block reads: 92.6 T-MAC/s at 182 W and
  1,850 MHz, so about 93 is this card's ceiling at its power limit. DPAS alone: 112.3 at 158 W.
- The built fold (`race.sh`, card 5): 62.7 TH/s at 1,972 MHz and 187 W against the skeleton's 67.0: its readout
  costs about 6%.
- Dropped: A' with its own cache control through IGC's LSC builtin (reaches the instruction; 66.6 to 67.6
  against 67.0); padded operand strides (no change); remapping windows to work-groups (best 76.8 against 77.0);
  a work-group that fills an Xe-core (67.9 a clock in step, 45 to 49 staggered); persistent work-groups (a loop
  around the k-loop, even run once, halves the rate on this driver); the fold as one flat loop with the readout
  (58.5 against 62.7).
- Records: `earn/native/sycl/README.md`, "A price list, and what holds the card".

### Arc rig, 11 to 12 November, night: the NVIDIA methods on the A750, and a ring fold (+6.9%, `goal/arc-from-nv`)

Which of the 4090's kernel methods carry over to the A750, priced on card 5 with new probe rows before anything was
built (the NVIDIA price-list method), then the best one built. Release build form (open-source DPC++ v7.2.0, Level
Zero, Intel 26.40). The full table of methods is in the agent's report; the kernel and the rows are described in
`earn/native/sycl/README.md`, "The ring fold".

**Built: the ring fold (`PEARL_SYCL_FOLD=xmx8r`, opt-in).** The NVIDIA fold copies an operand tile into shared
memory once a block and every warp reads it there. Here a work-group of 8 windows of one column window (16
sub-groups) copies each k-step's B' once into a three-slot ring in local memory, held by a split barrier, and every
sub-group reads its B' there by block reads; A' stays each sub-group's own. The copy is loaded a k-step early so its
store never waits for memory.

| Fold, card 5, `pearl_sycl_check --bench`, 20 s, taking turns | Round 1 | Round 2 | Round 3 | Clock | Power |
|---|---|---|---|---|---|
| fragment fold (`xmx8f`) | 62.79 | 62.64 | 62.65 | 1,970 MHz | 189 W |
| ring fold, 8 windows a work-group | 67.17 | 66.92 | 66.91 | 2,000 MHz | 189 W |
| fragment fold again (A/A twin) | 62.66 | 62.76 | 62.74 | 1,969 MHz | 189 W |
| ring fold, 16 windows | 63.46 | 63.45 | 63.46 | 2,000 MHz | 180 W |

+6.9% at the same 189 W; the twin's spread is 0.2%. Checks: 16,384 of 16,384 regions on card 5 at m = n = 2048;
`verify-hits.js` 400 of 400 at the mainnet size (62.3 TH/s with every hit checked); the CPU device, every region at 1
to 16 windows a work-group. At the mainnet size the fold check mismatched 4 of 32,768 regions for both folds alike:
card 5's known fault when hot, in the copy of the operands the host's reference reads.

On the pool (Kryptex, card 5 alone, worker `arc750-ci`, 5 minutes each, one after the other, mean after the first
minute): ring fold 66.55 TH/s (65.6 to 67.8), 3 accepted, 0 rejected, first hit right; fragment fold 62.18 (61.2 to
63.4), 3 accepted, 0 rejected. +7.0%, the same as the bench. Card 5 at about 1,790 and 1,770 MHz and its power limit
in both; cards 1 to 4 were mining beside it.

**What carried over and what did not:**
- Sharing an operand through on-chip memory carries over (above). Making operands on the chip from the noise
  tables (the gather, staged-draw and direct folds, the rotation fold's draw) does not: an A750 sub-group has no
  cheap byte gather, and a draw costs about 8 extra small messages a k-step a sub-group against a port budget of 16
  clocks. Copying wins.
- The split barrier is the A750's `bar.sync` for a group of sharers: 2.6% a k-step with nothing else in the loop
  (5% for a whole barrier), and about 11% less a clock than a whole barrier in the ring. With loads in the loop it
  still costs 10% against no barrier at all (wrong on purpose), because every sub-group waits for the slowest.
- The price list carries over, and the watts column matters most: DPAS alone on register operands draws 151 to 160
  W of the card's 190, and every load-bound row runs at the limit. Rows with one DPAS a k-step stay under it, so
  their watts are the loads': a last-level-cache byte about 22 pJ, a first-level hit about 7.5, local memory about 9
  (taking about 85 W as the card's own at 2,036 MHz; a fit). 100 TH/s at 190 W leaves about 6 pJ a byte at the
  fold's 2,560 bytes a k-step.
- Readout placement: moving each chunk's readout after the next k-step's arrival lost 16% in the ring fold (60 a
  clock against 72). Not kept.
- Not carried over, by reason: the clock offset and memory clock (need root here), the per-salt order search (the
  ring reads whole 128-byte blocks, nothing collides), the hash on the card (already there), k and cache residency
  (already k 2048 with band 64). Hand-written loops are possible on Intel (ESIMD, inline vISA, IGA on the native
  binary) but the port and the watts limit the A750, not the instruction order.
- Both operands through the ring, first stand-ins: 32 by 32 a sub-group (`s-g32`) 17.8 to 36.8 a clock, A' shared by
  two column windows (`s-ring2d`) 14.8 to 34.1, against the fold's 71. Lost as written; the cause is not found.
- Not to be trusted: the probe's `s-b` and `s-ab` rows and the first `s-ring` rows at 16 and 32 windows read local
  memory nothing wrote, so their watts are low (clock figures stand); the first `s-ring` rows at 16 and 32 windows
  also copied nothing. `p-slm` rows are held by a modulo in the address, not by local memory.

### Arc rig, 11 to 12 November, night: the AMD folds' methods on the A750, and a fold that shares A' (`goal/arc-from-amd`)

Which of the AMD core's methods carry over to the Arc A750, priced on card 5 with stand-ins in
`pearl_sycl_peak` (rows `s-`, `t-`, `u-`; timing only, they compute wrong results on purpose), then the best one
built. Card 5, 2 s a row, MACs an EU a clock and T-MAC/s, i915's clock and watts. The fold's stand-in (`f-v84-pf3`,
its loads with no readout) ran 71.4 to 74.5 a clock, 61.6 to 61.9 T-MAC/s at 1,850 to 1,927 MHz and 178 W, in
every run below.

**Operands read from shared local memory, as the AMD folds read LDS.** A work-group copies its operands into a ring
in local memory and its sub-groups read their fragments there. A 32 by 32 tile a sub-group (`t-`) beats the fold's
16 by 64 (`s-`): it reads 4 fragments of each operand a k-step, so a k-step sends 12 messages, not 18.

| Row | What | A clock | T-MAC/s, clock, watts |
|---|---|---|---|
| `t-ceil-v8` | both operands only read from local memory (filled once), 8-dword reads | 99.2 to 105.5 | 90.5 to 93.9, 1,982 to 2,036 MHz, 167 to 172 W |
| `t-ceil-a` | A' from local memory, B' from memory | 98.6 to 107 | 86 to 92, 178 W |
| `t-ceil` | both, B' read as two 4-dword halves | 71.1 to 72.5 | 64.7 to 64.9, 151 to 155 W |
| `s-ceil`, `s-ceil-v8` | 16 by 64, both from local memory: halves; 8-dword reads | 47.2; 83.4 | 43.0; 75.9 |
| `t-sbar-a` | A' from local memory, a split barrier every k-step, no copy | 93.0 to 95.6 | 82.6 to 85.3 |
| `s-bar-a`, `s-sbar-a` | 16 by 64, a plain barrier every k-step; a split one | 72.1; 78.9 | 64.3; 71.8 |
| `t-pinc-a` | A' copied a k-step ahead, the copy's loads waited for in their own k-step, no barrier | 78.5 to 87.0 | 69 to 75 |
| `t-ld-a`, `t-ldc-a`, `t-st-a` | only its loads; its loads from one cached place; only its stores | 88.7; 98.4; 101.1 | |
| `t-split-a` (4 by 8 sub-groups) | the whole pipeline: copy, split barrier | 73.2 to 75.2 | 66.4 to 66.9, 1,986 to 2,032 MHz, 172 to 174 W |
| `t-spr-a4` (4 by 8) | the same with the copy's loads spread over a 4-k-step stage | 75.4 | 67.7, 2,005 MHz, 177 W |
| `t-spr1` (4 by 8) | both operands through the ring | 64.7 | 57.8 |

- **Local memory is fast for 8-dword reads and slow for 4-dword ones.** B' read as two halves of 4 dwords runs at
  45 to 72; as one vector of 8, at 97 to 105.
- **The split barrier carries over from RDNA 4.** `SPV_INTEL_split_barrier` (arrive after the stores, wait a
  stage later) compiles and runs on DG2. A plain work-group barrier every k-step costs 16 to 22%: with 4 threads
  an EU in the 256-register mode there are no other threads to run while it waits. The split one costs 3 to 8%.
- **What the copy costs is its load from memory.** Stores to local memory cost nothing, and loads from a place
  in the cache cost nothing. A load of A' from L2 that its own k-step waits for costs 10%. Keeping it in
  registers for a stage, or pinning it further ahead, did no better (66 to 76).
- **At the card's power limit the pipeline gains more than it does a clock.** Per clock it is level to +5%. It
  draws less power for the same work, so the card keeps a higher clock: 66.4 to 67.9 T-MAC/s against 61.6.
- Producer sub-groups that only copy (`u-`) ran at 21 to 39 even when they copied nothing, at every shape. The
  cause is in code generation for a kernel with two roles; not found.

**Built: `xmx8s`, A' shared in local memory.** A sub-group holds 32 rows by 32 columns. A work-group is 4 by 8
sub-groups (16 windows); its A' goes into a ring of 3 stages of 2 k-steps, and B' is read as `xmx8f` reads it.
It uses the same operand layouts as `xmx8f`. Off unless asked for: built with `-DPEARL_SYCL_XMX8S`, chosen with
`PEARL_SYCL_FOLD=xmx8s`. The split barrier needs `-DPEARL_SYCL_SPLIT` and the translator's
`SPV_INTEL_split_barrier`, which a release does not ask for (a 2024 driver refuses extensions it lacks).

| Check, card 5 | Result |
|---|---|
| fold check, SYCL CPU device (the reference code), 8,192 regions | PASS: 0 mismatched, 23 of 23 hits |
| fold check, card 5, 262,144 regions (m 8,192, n 16,384) | PASS: 0 mismatched, 1,002 of 1,002 hits |
| `verify.sh` check step, 16,384 regions | PASS: 29 of 29 hits |
| `verify-hits.js`, mainnet, core opened on `xmx8s` | PASS: 400 of 400 verified, 0 failed, 7 salts |
| Kryptex, card 5, 5 minutes, worker `arc750-ci`, every hit checked before it was sent | 64.6 TH/s, 5 accepted, 0 rejected, no wrong hit; 2,000 MHz, 173 W |

| Race, card 5, 12 s turns | Round 1 | Round 2 |
|---|---|---|
| `xmx8f` (today's fold) | 62.79 TH/s, 1,976 MHz, 188.5 W | 62.64, 1,962 MHz, 188.6 W |
| `xmx8s`, split barrier | 64.96, 2,000 MHz, 171.7 W | 64.91, 2,019 MHz, 171.0 W |
| `xmx8s`, plain barrier | 53.74, 2,000 MHz, 153.8 W | 53.65, 2,000 MHz, 153.7 W |
| `xmx8f`, twin | 62.72, 1,974 MHz, 188.5 W | 62.68, 1,974 MHz, 188.7 W |

- **+3.5% rate at 17 W less: 0.38 TH/s a watt against 0.33.** The fold now sits at the 2,000 MHz the card holds
  under DPAS, not at its 190 W limit. On the rig's hot inner cards, which run a load cycle to stay cool, the
  watts are the gain.
- The plain-barrier build is 14% slower: the split barrier is what makes it pay.

**The rest of the AMD list, priced.**

- Operand fill, the real fold, card 5, 12 s turns, 3 rounds. Fill 48 did 62.66 to 62.86 TH/s at 1,974 to 1,979
  MHz and 188 W (twin 62.67 to 62.78). 63 and 56 were level. 32 did 62.31 to 62.44 at 1,955 to 1,962 MHz, 16 did
  61.72 to 61.87 at 1,936 to 1,943 MHz. Lower fills lose clock at the same watts, as on RDNA 4.
- Readout, hash and test: the real fold runs 71.5 a clock against 72.1 for `f-v84-pf3`, its loads alone, so
  under 1% by that stand-in; against `goal/arc-4`'s flat skeleton `p-base` (76 to 80 a clock) it is about 6%
  (entry above). AMD's late readout and k 4096 are worth pricing again against `p-base`. Thread starts cost
  1.4% (`f-regs` 113.7 against `f-regs-short` 112.1).
- The host loop (AMD's hash queue and ticker): the miner's loop runs 62.04 to 62.16 TH/s against the fold alone at
  62.63 to 62.76, so under 1%.
- Address math (AMD's buffer loads): each of the fold's 72 loads a chunk has a 64-bit address chain (72 `addc`,
  74 `add`, 147 `mov`). The scalar-address form was already timed level from real addresses (block reads, 73.4
  against 72.9).

### Arc rig, 12 November, night: the ring fold checked on a cool card and mining, and what was tried beside it (`goal/arc-5`)

What changed for the miner: the four mining cards (1 to 4, duty 60,60,60,60,75) run the ring fold `xmx8r` since
01:04 UTC, 12 November: 170.2 to 170.9 TH/s against 160 on the fragment fold, shares accepted and none rejected in
the first run (12 accepted, 0 rejected in the first 9 minutes). Before that it passed on card 1, paused for the check: verify.sh's fold checks
(16,384 of 16,384 regions at m = n = 2048, 32,768 of 32,768 on the last mainnet batch) and verify-hits.js 400 of 400
with `PEARL_SYCL_FOLD=xmx8r` (verify.sh's own hit step runs `auto`, so it was run by hand). The shared-A' fold
`xmx8s` passes the same on card 1 (400 of 400). The rig's HiveOS driver (OpenCL 24.35, no Level Zero) takes the
split barrier: the release form, staged in `arc-jammy`, passes `xmx8r` there and benches 63.2.

Races, `pearl_sycl_check --bench 8`, two rounds, taking turns, TH/s:

| Card | xmx8f | xmx8r | xmx8s | Clock, watts |
|---|---|---|---|---|
| 1 (paused mining card) | 63.2, 63.5 | 66.7, 67.2 | 65.6, 65.7 | f 1,958 to 1,965 MHz 188 W; r 1,992 to 2,000 MHz 188 W; s 2,000 MHz 174 to 176 W |
| 5 | 62.9 | 67.2, 67.4 | 65.3, 65.4 | f 1,973 MHz 187 W; r 2,000 MHz 187 to 189 W; s 2,000 MHz 170 to 172 W |

Both new folds sit at 2,000 MHz, the clock these cards hold under DPAS (the fragment fold drops below it at the
power limit). So from here only work a clock counts: xmx8r 75 MACs an EU a clock, xmx8s 73; 100 TH/s at 2,000 MHz
needs 112 of the 125 peak. In TH/s a watt xmx8s is best (0.38 against xmx8r's 0.36 and xmx8f's 0.33).

Findings:

- **The fold that mined had no loop around its k-loop.** In the form the rig's driver compiles, IGC writes a
  chunk's 4 k-steps out (71 DPAS, 72 loads in one block); only the prebuilt oneAPI form keeps a one-k-step loop.
  So goal-arc4's driver finding (a loop around the k-loop halves the rate) did not apply. Two chunks an iteration:
  level (62.8); all 16 written out, no loop: 40.2 (12,065 instructions, the instruction cache).
- **More than 32 KB of local memory on an Xe-core costs the fragment fold 10%.** 1,024 bytes a group (32 a core):
  62.8; 1,088, 2,048 or 4,096: 56.2 to 56.3. With 16 groups a core: 2,048 bytes level, 2,176 bytes 56.5. The ring
  fold asks for 56 KB a core and gains only 0.9% when cut to 26 KB (timing only), so it does not pay that step.
- **Sharers must read together.** Rotating the two sub-groups' k-step order inside a chunk, so they stop reading
  the same B' at once: 38.7 to 40.2 against 62.8. A line is gone from the first cache level within about a k-step.
- **Two windows a work-group** (16 by 64 tiles, B' read by 4 sub-groups together, 1 KB of local memory): 64.4
  against 62.8 on card 5 (+2.5%), 64.3 against 63.0 on card 1. Four windows 63.2, eight 54.9. Below the ring fold.

Tried and lost (correct unless marked timing only):

| Design | TH/s | Against |
|---|---|---|
| Fragment fold, readout in the next chunk's first k-step | 20.8 (spills 3.2 KB) | 62.8 |
| Fragment fold, no local memory, words in registers (timing only) | 61.8 | 62.8 |
| 32 by 32 tile a sub-group, A' and B' from global | 45.7; A' in halves 53.6 | 62.8 |
| Two windows a group of 32 by 32 tiles, local XOR atomics | 55.2 | 62.8 |
| Ring fold, words in registers (VAR 28) | 41.7 (spills 640 B) | 67.3 |
| Ring fold, 32 by 32 tiles, A' from global (VAR 140) | 59.3 | 67.2 |
| Ring fold, A' prefetched to L1 by IGC's LSC prefetch, 1 to 3 k-steps ahead | 47.8 to 48.3 | 67.2 |
| Ring fold, readout after the next arrival (VAR 13) | 57.9 | 67.2 |
| Ring fold, 4 row windows by 2 column windows a group (VAR 524) | 63.7 | 67.2 |
| Ring fold, two k-steps a slot, a barrier every second k-step (VAR 1032) | 67.2, level | 67.2 |

The variants are on `goal/arc-5` behind `PEARL_SYCL_EXTRA` (`-DPEARL_A5=N` for the fragment fold's, in
`pearl_sycl_a5.hpp`; `-DPEARL_RING_VAR=N` for the ring fold's); a release build has none of them.

### Arc rig, 12 November, night: a raised power limit, the ring fold as `auto`, and ways to hide its load waits (`goal/arc-6`)

**A raised power limit on one card is worth nothing to the ring fold today.** The rig's owner agreed to test one
cool, stable card above its 190 W limit, inside the card's rating, with the thermal safeguards on and the setting
restored afterwards. Card 4 (it had held 63 TH/s alone at full power for 10 minutes). It is Intel's own A750
Limited Edition (PCI subsystem 8086:1021). The driver's `power1_rated_max` reads 0, so the cap was Intel's
published board power of 225 W. The limit was written through a throwaway container that bind-mounts only that
card's hwmon folder (no `--privileged`), and a trap restored the original on any exit. Each step was three 60 s
`pearl_sycl_check --bench` runs of `xmx8r`, with the flags read every 2 s; it would have stopped at once on
PROCHOT, a regulator thermal alert, a thermal flag in most of 30 s, or a fan over 2,600 rpm.

| `power1_max` | TH/s (3 runs) | MHz | W | Thermal / PL1 flags | Fan max |
|---|---|---|---|---|---|
| 190 W (original) | 68.55, 68.51, 68.52 | 1,998 to 2,000 | 184.7 to 186.1 | 0 / 0 | 2,127 |
| 200 W | 68.43, 68.42, 68.39 | 1,992 to 1,996 | 186.0 to 186.2 | 0 / 0 | 2,400 |
| 210 W | 68.39, 68.39, 68.38 | 1,990 to 1,996 | 186.1 to 186.2 | 0 / 0 | 2,122 |
| 220 W | 68.39, 68.40, 68.39 | 1,992 to 1,998 | 186.1 | 0 / 0 | 2,122 |

- The card asks for 2,400 MHz (`punit_req_freq_mhz`) and holds 2,000 under DPAS with no throttle flag set, at
  186 W. So the ring fold is held by the DPAS clock, not by power. A higher limit would only pay for a fold that
  does more work a clock and then reaches 190 W.
- The limit was restored to 190,000,000 uW and read back. `verify-hits.js` on card 4 with `xmx8r` at the original
  limit: 400 of 400, 0 failed, 7 salts.

**`auto` is now the ring fold on Alchemist.** The four mining cards ran `xmx8r` for 60 min 45 s over three runs (36
min on cards 1 to 4, 18 min on 1 to 3 during the power test, 7 min on 1 to 4): 69 accepted, 0 rejected.
`PEARL_SYCL_FOLD=xmx8f` goes back. Found on the way, and older than this change: the reference folds run on the card
(`xmx8f-emu`, `xmx8r-emu`, so `auto-emu`) fail every region on card 5 in last night's builds too (4,096 of 4,096
mismatched, 15 of 20 hits); the CPU device's reference passes.

**Hiding the ring fold's load waits: everything built loses.** Card 5, `pearl_sycl_check --bench 8`, taking turns
with the shipped fold (`VAR 12`: 67.4 to 67.6 TH/s, 2,000 MHz, 183 to 188 W, no spill). Every build below passes the
fold check (4,096 of 4,096 regions, 20 of 20 hits). `PEARL_RING_VAR` bits are in `pearl_sycl_kernels.hpp`.

| Build | What | Spill (bytes) | TH/s | W |
|---|---|---|---|---|
| 2060 | A' a k-step ahead in registers | 3,168 | 22.4 | 146 to 151 |
| 4108 | counters in local memory in place of the barrier, 5 slots (4 and 8: the same) | 3,872 | 16.3 | 129 to 136 |
| 8204 | the k-step's A' loads sent before the barrier's wait | 4,320 | 24.5 | 156 to 160 |
| 16396 | B' in two halves of 4 fragments, each waited for, then its 8 DPAS | 0 | 49.4 | 164 to 169 |
| 98316 | software pipeline: each fragment register reloaded for the next k-step after its last DPAS, 4 slots | 8,768 | 8.5 | 145 |
| 557064 | the same with 256-byte block reads, no early copy | 4,096 | 39.6 | 173 to 178 |
| 360456 | the pipeline with B' in fragment order, read as 8 dwords a lane | 608 | 50.7 | 170 to 174 |
| 262156 | B' in fragment order, 8-dword reads, no pipeline | 0 | 45.6 | 149 to 153 |
| 524300 | a fragment as one 256-byte block read (8 dwords a lane) | 0 | 67.1 | 182 to 187 |
| 1048588 | DPAS wait only for A', B' reads left to the compiler | 0 | 67.1 | 181 to 187 |
| 2097164 | no forced wait at all | 0 | 67.3 to 67.5 | 184 to 188 |

- The fold sits at the edge of the 256 registers. Anything more that is live across the barrier, or a fragment
  carried across the loop, makes IGC spill kilobytes; the ISA shows the carried B' fragments going through scratch.
  IGC's scheduler also regroups the DPAS and sends every reload after the last one, so a rotation written in the
  source does not survive.
- 8-dword reads from local memory compile to a gather (`load.slm.d32x8`) and lose 32% here; the transposed block
  read (`load.slm.d32x32t`) is the fast form. The AMD agent's "8-dword reads are fast" compared them with 4-dword
  vector reads, not with block reads.
- A wait in the middle of a k-step costs 27% (two halves), but no forced wait at all is level. Neither the wait
  nor (last night) the number of barriers is what holds the fold now.
- What fits the numbers: each Xe-core's data port. A k-step sends 21 messages of 128 bytes (4 of A' from memory, 16
  of B' from local memory, 1 of copy) against 16 DPAS clocks, which caps the 16 by 64 tile near 76% of peak. That is
  where it runs (75 MACs an EU a clock). To pass it, a k-step must send 16 messages or fewer: a 32 by 32 tile
  (8 + 8) with both operands served near.

### Arc rig, 12 November, night: the split fold, 88 TH/s a card (`goal/arc-7`)

**A new fold, `xmx8w`: 88.0 TH/s on a cool card against the ring fold's 68.2, verified.** Card 4, `pearl_sycl_check
--bench 8`, three rounds taking turns: `xmx8w` 87.7 to 88.0 TH/s at 1,950 to 1,954 MHz and 188 W; `xmx8r` 68.2 to
68.3 at 2,000 MHz and 188 W (+28%). Fold check PASS (16,384 and 32,768 regions; its reference `xmx8w-emu` too),
`verify-hits.js` with `PEARL_SYCL_FOLD=xmx8w` 400 of 400, 0 failed. On the host's own 2024 driver (24.35, the
release form staged in `arc-jammy`, card 5 over OpenCL): check PASS. The four mining cards run it (223 TH/s at duty
60,60,60,60,75, was 170). What it does, in the order it was found:

- **The split DPAS.** Alchemist's DPASW (`intel_sub_group_i8_i8_split_matrix_mad_k32`, OpenCL's
  `cl_intel_subgroup_split_matrix_multiply_accumulate`) takes A from a pair of sub-groups (2i and 2i + 1), each
  giving half of the 8 rows, and gives each all 8 rows times its own B. So each window (32 rows by 64 columns) is
  split by columns: a sub-group holds 32 rows by 32 columns, reads half of each A' fragment (one 128-byte block read
  in the ring fold's layout) and 4 B' fragments from the ring. Same 512 bytes of A', half the B' reads, and it fits
  in 176 registers. Card 5: 71.7 to 71.8 against 67.5.
- **No branch in the copy.** The ring fold guarded its copy (`if (ks + 1 < 64)`, `if (i < mine)`), and IGC wrapped
  each guarded load and store in goto/join, which cut the k-step into blocks it would not schedule across. Copying
  one block a sub-group unconditionally (past the end it copies k-step 63 again into a slot no one reads) took the
  copy's cost from 23% to 7%: 82.1 to 82.3 on card 5. A program that only moved the copy's block through one more
  register ran 63.8: much rides on IGC's schedule.
- **The group meets once a chunk.** Ring slots of four k-steps (24 KB a work-group): card 4 87.7 to 88.3 against
  86.1 to 86.6 for two and 84.2 for one. Eight do not launch.
- **No forced load wait.** Dropping the gate that makes every DPAS of a k-step wait for all its loads: card 4 88.1
  to 88.4 at 1,950 to 1,977 MHz against 87.5 to 88.0, same 188 W (check PASS, verify-hits 400 of 400).

**`auto` is now `xmx8w` on Alchemist.** 63 minutes on the four mining cards, 92 shares accepted, none rejected (17
minutes in its first form under the `xmx8r` name, 46 as `xmx8w`). `PEARL_SYCL_FOLD=xmx8r` goes back.

**The message-count model does not hold.** Timing builds of `xmx8r` on card 5 (control 67.4 to 67.6 TH/s, 22
messages a k-step): B' as 64-byte block reads, the same bytes in 38 messages, 64.3 (the model says 39); as 32-byte
reads (70 messages) 50.3; A' as 64-byte reads 56.7. Reads the DPAS wait for cost by their latency, and more from
memory than from local memory. goal-arc6's "one 256-byte block read a fragment" compiled to the same two 128-byte
messages, so it did not test fewer messages.

**Where the ring fold's time went** (card 5, timing builds, wrong results): no barrier 71.3, no copy 82.9 (its load
74.2, its store 73.0), A' from local memory 73.4 at 140 W, all three 92.2 at 128 W. All three gone is 103 MACs an
EU a clock at 160 bytes a DPAS, 128 bytes a clock a core: a 16 by 64 tile tops out near 92. A raised power limit
was not tried for `xmx8w` (an agent may not change it); it now sits at 188 W.

**Toward 100** (card 5, timing builds of the split fold, wrong results): no copy and no barrier 90.2 at 2,000 MHz;
with A' a k-step ahead as well, **100.0 TH/s at 2,000 MHz and 179 W**. With every window reading one window's A'
(so A' stays in L1) the real kernel does 98.9 at 188 W. Lost or level: A' a k-step ahead in the real kernel (-1%
at the power limit), both k-steps' B' read at once (-3%), work-groups in a row sharing rows (level), B' straight
from memory with no ring (72), two column windows a work-group so A' is read once for both (4 by 2: 84.5; 8 by 2,
32 sub-groups: 83.1 at 2,000 MHz and 180 W, not power-bound but slower a clock).

**The reference folds on the card are fixed.** `select_from_group` with an index the same in every lane returned
the wrong lane's word on this runtime (26.40), from a vector or a scalar; `group_broadcast` the same. A test kernel
(hardware DPAS against the reference on random fragments) showed `permute_group_by_xor` matching the hardware bit
for bit. The reference DPAS now walks lanes by XOR: `xmx8-emu`, `xmx8f-emu`, `xmx8r-emu`, `xmx16-emu` pass on card 5
and on the CPU device.

### Arc rig, 12 November, early morning: `xmx8w` to 92.0 TH/s a card (`goal/arc-8`)

**91.7 TH/s on a cool card, verified (was 88.4).** Card 4, `pearl_sycl_check --bench 8`, three rounds taking
turns: the new build 91.6 to 91.8 TH/s at 1,938 to 1,946 MHz and 188 W; the previous mining build 89.2 to 89.4;
`xmx8r` 68.1 to 68.3. Fold check PASS (16,384 and 32,768 regions, and `xmx8w-emu`), `verify-hits.js` with
`PEARL_SYCL_FOLD=xmx8w` 400 of 400, 0 failed. The four mining cards run it. Two changes, both found by timing:

- **A 256-byte gap after each 16-row block of A'** (`PEARL_SYCL_APAD`, default 256). The blocks sat exactly 32 KB
  apart, so a work-group's 16 A' streams fell in the same cache sets. Card 5: 86.5 to 87.6-88.0 at 188 W (gaps of
  128 to 3,200 bytes all gained; 256 most; a gap after B' blocks gained nothing). Card 4: 89.2 to 89.4 against
  88.2 to 88.4.
- **Address sums in 32 bits.** IGC built each 64-bit global address from scratch with three to five scalar
  instructions between the DPAS groups, about 65 instructions a k-step. Those are not free: 32 extra scalar
  instructions a k-step cost 15% in a timing build. With each operand's high word fixed and offsets added to the low
  word, a k-step is about 50 instructions: 88.0 to 90.3 on card 5 at the same 188 W. It needs each operand inside one
  4 GB window, so the host allocates A' or B' again (up to five tries) when one crosses a boundary, and runs `xmx8r`
  if it still does.

**Then 92.0 TH/s, verified.** A window's two sub-groups now XOR their chunk words into one shared place in local
memory (4 KB, not 8) and the readout XORs whole accumulators before their dwords: same results (check PASS on card 5
and the CPU device). Card 4, three rounds: 92.0 at 1,954 to 1,962 MHz and 188 W against 91.6 for the build before;
verify.sh PASS, verify-hits 400 of 400. The mining cards run it (235 TH/s at duty 60,60,60,60,75).

**What holds it now.** At the 2,000 MHz clock cap the fold does about 97.5 MACs an EU a clock (A' zeroed, so not
power-bound: 87.3 TH/s at 169 to 173 W); 100 TH/s needs 112. With the address fix it sits at the 188 W limit near
1,940 MHz. Without the copy (timing, wrong results) it runs 95.8 TH/s at 2,000 MHz and 170 to 174 W: the copy still
costs about 6% and 15 W, mostly in the scalar instructions and register pressure it brings, not its bytes.

Timing builds on card 5 (wrong results unless "PASS"), all against a control of 86.5 to 88.0 at 188 W unless said:

| What | TH/s | MHz | W |
|---|---|---|---|
| A' data ANDed to zero (loads kept) / B' zero / both | 87.3 / 84.3 / 79.2 | 2,000 | 169-173 / 158-162 / 155-158 |
| every window reads window 0's A' / the same XORed per window (hot addresses, own data) | 98.1 / 94.8 | 1,981-2,031 | 184-188 |
| A' of 2 / 4 row windows only (256 KB in all) | 96.3 / 94.7 | 2,000 | 180-188 |
| 8 by 2 work-groups (A' shared by two column windows), KPS 4 / 2 / 1, PASS | 85.0 / 83.6 / 83.0 | 2,000 | 178-187 |
| 8 by 2 with column window 1's k-steps rotated by 1 or 2 inside the slot, PASS | 84.5 / 83.7 | 2,000 | 180-185 |
| A' addressed k-step major (same bytes) | 64.5 | 2,000 | 176-181 |
| `PEARL_SYCL_COL_BATCH` 32 / 64 / 128 / 512 / 1,024 (default 256) | 72.6 / 78.1 / 85.5 / 82.7 / 82.6 | | 188 |
| `PEARL_SYCL_BAND` 16 / 32 / 128 (default 64) | 79.6 / 86.2 / 82.4 | | 188 |
| no copy / no barrier / neither | 95.7 / 88.7 / 96.0 | 2,000 / 1,936 / 2,000 | 172-177 / 188 / 173-176 |
| the copy without its load / without its store | 86.9 / 87.1 | 1,946 / 2,000 | 188 / 171-176 |
| copy once a chunk, 512 bytes a sub-group, PASS | 85.4 to 87.0 | | 188 |
| copy loaded at the k-step's start and stored after its DPAS, PASS | 85.0 | | 188 |
| copy after the k-step's local reads, PASS | 87.3 | 1,904 | 188 |
| no ring: B' from memory, windows' k-steps rotated 0 / 1 / 3, PASS | 76.0 / 79.5 / 79.6 | | 188 |
| copier sub-groups beside the 16 DPAS ones (2 to 16) | FAIL: even idle copiers break DPASW | | |
| one work-group a core (local memory padded) | 59.1 | 2,000 | 156 |
| two-slot ring (arrive after the slot's reads), XOR-combined words, PASS | 87.2 to 87.3 (3 work-groups a core) | 1,912 | 188 |
| 16 row windows a work-group (32 sub-groups, half-block copies), PASS | 86.2 | 2,000 | 184-188 |
| work-groups in a row sharing rows, 2 / 4, PASS | 87.9 / 87.5 | | 188 |
| +32 / +96 scalar ALU instructions a k-step, PASS | 74.4 / 50.2 | | 188 |
| with 32-bit sums: A' a k-step ahead / KPS 2 / two slots / copy after reads, PASS | 86.4 / 87.7 / 88.9 / 88.2 (control 90.3) | | 188 |
| with 32-bit sums: pointers kept as (low, high) pairs stepped each k-step, PASS | 86.7 | | 188 |
| readout as a tree of whole accumulators, PASS | 90.2 (level; with the XOR-combined words 90.3 to 90.6, kept) | | 188 |
| copy once a chunk with 32-bit sums, load at k-step 0 / 1 / 2, PASS | 87.1 / 87.6 / 87.7 | | 188 |
| A' a k-step ahead with no copy (timing) / with no copy and no barrier | 93.6 / 92.7 (no copy alone 95.8) | 2,000 | 173-179 |
| chunk loop unrolled 1 / 4 / 16 times (IGC picks 2), PASS | 89.7 / 89.9 / 87.7 (2: 90.7) | | 188 |
| `IGC_LscLoadCacheControlOverride` 2 / 6 / 7 (L1 uncached, streaming, invalidate after read) | 89.7 / 89.7 / 89.6 (90.1) | | |

On HiveOS's own 2024 driver (24.35; the release form staged in `arc-jammy`, run on the host over OpenCL, card 5):
`xmx8w` and `xmx8w-emu` check PASS, `--bench 8` 84.3 TH/s (goal-arc7's build: 77.3 there).

### Arc rig, 12 November, morning: finalizer options, and `xmx8w` in ESIMD (`goal/arc-9`)

The host now sets `IGC_VISAOptions="-insertDPASRSDummyMov -DPASTokenReduction"` before the driver compiles
anything (`PEARL_SYCL_VISA=none` skips it; a user's own `IGC_VISAOptions` wins). Card 4, cool, 8 s, taking turns:
spare17 (cf88be3c) verify.sh PASS (`xmx8w` and `xmx8w-emu` at 16,384 regions, `xmx8w` at 32,768), verify-hits
400 of 400, 0 failed; **93.5 to 93.8 TH/s at 1,946-1,950 MHz, 188 W**, against spare15 (053b2a88) 92.2 at 1,958-1,973
MHz and `xmx8r` 68.1-68.2. Without the options the same build ran 92.6-92.8; spare15 with them 93.5-93.7.

Why the first option helps: Intel's finalizer puts a dummy DPAS (a `dpasw` with a null accumulator, a whole DPAS of
work) before a run of 8x8 DPAS whose first two share B, when B's registers are also written by ALU code somewhere in
the kernel or B is not loaded in the same block (Wa_18023229625; IGC's `visa/SWWA.cpp` and
`visa/LocalScheduler/SWSB_G4IR.cpp`). xmx8w's loop had 6 for 128 DPAS; with the option 4.

Timing on card 5 (8 s, 2-3 rounds; control 90.5-90.7 TH/s, 1,946-1,961 MHz):

| What | TH/s |
|---|---|
| `-insertDPASRSDummyMov` / `-DPASTokenReduction` / both (kept) | 91.4-91.6 / 91.0 / 91.8-92.0 |
| the same two on HiveOS's 24.35 driver (host, OpenCL) | 85.2 against 84.2 |
| on top of both: `-nopresched`, `-presched`, `-scheduleForReadSuppression`, `-SendTokenReduction`, `-schedWithSendSrcReadCycle`, `-multiplePipeSched`, `-groupScheduleForBC`, `-SWSBDepReduction`, `-sinkBarrierWait`, `-noremat`, `-nodpsendreorder`, `-SBIDDepLoc` | 91.4-92.1 (level) |
| `-sendQueueSched` | 89.9-90.2 |
| `-DPASBundleReduction`, `-enableBundleCR 3`, `-enableDpasFwd 0` alone | level |
| `-scheduleFor2xDpas` / `-enableDpasFwd 1` alone (3 dummies) | 90.1-90.3 / 90.2-90.3 |
| `-noDPASMacro` (24 dummies) | 83.2 |
| DPAS row-first in the SYCL fold (`PEARL_W_ROWFIRST`, 4 dummies) | 89.2 |
| operands as kernel arguments of their own, 32-bit surface loads (`PEARL_W_ARGP`; loop 598 to 517 instructions, 8 dummies) | 90.8 against 91.9 |
| two unused pointer members in the fold's functor (cd5b47de, card 4) | 89.6 against 92.2 |

`xmx8w` written in ESIMD (`FoldXmxE`, built with `-DPEARL_SYCL_ESIMD`, run with `PEARL_SYCL_ESIMD=1`): the same
work-group, ring, copy and split DPAS, each thread's program written out, words to a scratch buffer and a plain
epilogue kernel (2.4%). It passes the fold check but never beat the SYCL fold (card 5, control 90.7):

| What | TH/s | MHz |
|---|---|---|
| first form / 32-bit sums | 77.0 / 80.3 | 1,942-1,960 |
| `ch % 3` (a call to a division routine in VC's code) replaced by a stepped slot | 83.2 (85.5 without the epilogue) | 1,927-1,942 |
| + A' a k-step ahead, no epilogue | 86.1 | 1,888-1,896 |
| that with the finalizer's scheduler off (`IGC_VCApiOptions="-vc-codegen -Xfinalizer -noschedule"`) | 87.8 | 1,896 |
| DPAS row-first (1 dummy a chunk with the scheduler off), with the epilogue | 85.2 / 86.3 off | 1,920-1,933 |
| + an L1 prefetch of A' 2, 4 or 8 k-steps ahead | 84.3-84.5 | |
| B' bound to fixed registers (`ESIMD_REGISTER`) | FAIL | |

With the scheduler off ESIMD does the same work a clock as xmx8w (46.3 against 46.4 TH/s a GHz), but the card holds a
lower clock on it. The finalizer splits its DPAS into runs of 4 and puts a dummy before most (9 a chunk).

### Arc rig, 12 November, late morning: the hash at the end of a work-group (`goal/arc-10`)

`xmx8w` now runs two column windows a work-group, one after the other (c95de7d9; `PEARL_SYCL_WSEQ=1` runs one, and
so does a batch with an odd number of column windows). Card 4, cool, 8 s, taking turns: spare18 verify.sh PASS
(`xmx8w` and `xmx8w-emu` at 16,384 regions, `xmx8w` at 32,768), verify-hits 400 of 400, 0 failed; **94.4 to 94.6
TH/s at 1,938-1,950 MHz, 188 W**, against spare17 (cf88be3c) 93.8 and `xmx8r` 68.1-68.3.

Why: the BLAKE3 hash at the end of a work-group holds the Xe-core's local memory (two work-groups fit) while one
sub-group a window runs it, so the next work-group waits. With two windows in turn, sub-group `sgi` of each window
hashes set `sgi`'s words, both at once: the tail is paid once for twice the work. As a loop; unrolled, the second
set's set-up wrote registers the first set's B' uses and the finalizer added 16 dummy DPAS.

Timing on card 5 (8 s, 2-3 rounds, 188 W; control 91.8-92.2 TH/s, 1,935-1,946 MHz, 4 dummy DPAS a loop):

| What | TH/s | dummy DPAS |
|---|---|---|
| no hash at all (timing, `PEARL_W_NOEPI`) | 94.7-94.9 | 0 |
| hash in a second kernel, words out as plain stores (`PEARL_W_SEPEPI`), PASS | 88.6-88.9 | 2 |
| the same, words out as sub-group block writes (`SEPBLK`), PASS / without the hash kernel | 89.2 / 93.0-93.2 | 0 |
| one block write of the words' XOR, no hash (`SB1`) | 93.5 | 2 |
| hash reading its message words from local memory when needed (`PEARL_W_LEAN`), PASS | 85.5-85.7 | 3 |
| hash as a noinline function / with `IGC_FunctionControl` 2 or 3 | not raced / FAIL | 8 |
| DPAS in a snake (first two of each run differ in B'), PASS / with `-noschedule` | 90.2 / 88.1 | 5 / 2 |
| the hash twice (timing) | 85.3 | 18 |
| **two column windows in turn, as a loop, PASS (kept)** | **92.85-92.98** | 4 |
| the same unrolled | 90.6-90.8 | 20 |
| with it: band 32 / 128, column batch 512 | 92.6 / 90.1 / 90.4 | |
| with it: KPS 2 / two ring slots | 88.4-88.6 / 86.4 | |

What it taught: the dummy DPAS come from the hash's ALU code landing in registers the loop's B' uses (any ALU write
anywhere in the kernel counts), but removing them is worth less than it looked: a build with 0 dummies and a short
tail ran 93.1 where the one with no tail at all ran 94.9. The tail is the larger part. Small code changes also move
the rate by 1-2% through register allocation alone, so race every variant rather than reason from dummy counts.

### Arc rig, 12 November, 10:11-10:20 UTC: a raised power limit on the `xmx8w` build of 94.6 TH/s

Card 4 (PCI 13:00.0), the cool card, with the owner's agreement and the founder's go-ahead: the limit stepped 190, 200,
210 and 220 W (rated 225 W), two 60 s `pearl_sycl_check --bench` runs a step on the spare18 build (c95de7d9), watched
for PROCHOT, regulator alerts, thermal flags and fan speed, and put back to 190 W on exit (read back: 190 W).

| Limit | TH/s | MHz | Drawn |
|---|---|---|---|
| 190 W | 94.31, 94.31 | 1942, 1944 | 186.8 W |
| 200 W | 94.30, 93.76 | 1944, 1935 | 186.7 W |
| 210 W | 93.47, 93.75 | 1933, 1931 | 186.7 W |
| 220 W | 94.10, 93.69 | 1940, 1929 | 186.7 W |

No thermal flag, no power-limit flag, fans at most 2,163 rpm. The card draws 186.7 W whatever the limit, so the
limit does not hold this fold back: the clock stops near 1,940 MHz for another reason (the driver's status bit is set
in nearly every sample, but neither its thermal nor its PL1 reason). A higher limit is no lever for the Intel goal on
this card; work per clock is.

After the test the four mining cards moved from spare15 (92.0 TH/s a card on a cool card) to spare18, the founder's
decision.

### Arc rig, 12 November, late morning: dummy DPAS set the rate (`goal/arc-11`)

The host now adds `-noloopra` to the finalizer options it sets (3a398931). Card 4, cool, 8 s, taking turns:
spare19 verify.sh PASS (`xmx8w` and `xmx8w-emu` at 16,384 regions, `xmx8w` at 32,768), verify-hits 400 of 400, 0
failed; **95.0 to 95.5 TH/s at 1,935-1,950 MHz, 188 W**, against spare18 (c95de7d9) 94.4-94.8 and `xmx8r` 68.2-68.3.
HiveOS's 24.35 driver takes the option too (check PASS).

Why: the read-suppression workaround puts a dummy DPAS (a whole DPAS of work) before a run of DPAS whose B' registers
any ALU instruction anywhere in the kernel writes. `-noloopra` changes the register allocation so the loop has 3 where
it had 4. Each dummy costs about 0.5%, and that explains the earlier losses below: each design lost by the dummies it
added, not by its own cost.

Card 5, 8 s, 2-3 rounds, 188 W; control 92.9-93.0 TH/s with 4 dummy DPAS:

| What | TH/s | dummy DPAS |
|---|---|---|
| `-noloopra` (kept) | 93.4-93.5 | 3 |
| B' as one 256-byte local read / copy after the local reads / two ring slots / copy once a chunk | 91.6 / 90.0 / 86.5 / 86.4 | 7 / 10 / 16 / 19 |
| 16 row windows a work-group / no clamp in the copy / readout with no `if` (loop one block) | 91.2 / 91.9 / 92.6 | 8 / 8 / 7 |
| A' a k-step ahead / with `-linearScan` | 86.0 / 91.6 | 19 / 0 |
| `-linearScan` (20 more sync instructions in the loop) | 91.7 | 0 |
| `-enableBCR` / `-forceBCR` / `-twoSrcBundleBCR` (DPAS bank conflicts 88 to 16 / 16 / 0) | 92.2 / 90.1 / 85.4 | 9 / 13 / 22 |
| four column windows a work-group in turn (timing: upper bound for lead 1) | 93.2 | 4 |
| row block from the Xe-core, so a core's four work-groups share A' (timing, `PEARL_W_CORE`) | 94.0 | 4 |
| the same for real: a queue a core, claimed by a global atomic | 89.9 | 8 |
| no release fence at the arrive (unsafe, `PEARL_W_NOREL`) | 93.0 | 6 |
| hash with words read from local memory 8 at a time / four lanes a region (PASS) | not raced | 16 / 18 |

Also found: an Xe-core holds four `xmx8w` work-groups at once (128 KB of local memory, 64 threads), not two (a probe
reading each thread's core from `sr0`). About 40 other register-allocator and scheduler options left 3 or more
dummies. Raising card 4's power limit to 190-220 W (the coordinator, founder's approval) changed nothing: 186.7 W and
about 1,940 MHz every time.

### Arc rig, 12 November, midday: the 8x4 split DPAS ends the dummy DPAS (`goal/arc-12`)

`xmx8w` now issues each 8x8 split DPAS as two 8x4 ones (2e3d4887). The finalizer's dummy-DPAS workaround
(Wa_18023229625) applies only to 8x8 DPAS, so the loop has none, whatever registers the hash lands in. With that,
A' loaded a k-step ahead now wins, and so do two more finalizer options the host sets (`-noroundrobin
-keepDPASMacroInSchedule`). Card 5, 8 s, taking turns, three rounds: spare20 (2e3d4887) **96.3-96.5 TH/s at 1,888-1,896
MHz, 188 W, against spare19 93.2-93.4** at 1,923 MHz (+3.3%), `xmx8r` 67.1-67.2. verify.sh PASS (`xmx8w` at 16,384 and
32,768 regions), verify-hits 400 of 400, 0 failed. HiveOS's 24.35 driver: check PASS, 88.1 TH/s on card 5 (spare19
85.3 there at goal-arc11). Not yet confirmed on cool card 4: pausing its miner was refused by the permission check.
Card 4 has run about 1.7% above card 5, so about 98 is expected there.

Card 5, 8 s, 2-3 rounds (wrong-result timing builds marked):

| What | TH/s | dummy DPAS |
|---|---|---|
| control (spare19's code) | 93.5-93.6 | 3 |
| no copy, ring left as it was (zeros; timing) | 101.1 at 2,000 MHz, 177 W | 6 |
| no copy, every slot holding real B' (timing, new `PEARL_W_FILLALL`) | 97.3 at 1,960 MHz | 6 |
| no barrier / A' of 8 row windows only / no hash (timing) | 94.3 / 93.5 / 96.1 | 5 / 3 / 0 |
| operand fill 63 / 32 / -63 / 16 (shipped: 48) | 93.4 / 93.0 / 92.4 / 91.9 | 3 |
| the readout twice (`PEARL_W_RD2`, results kept) | 84.8 | 9 |
| extra int ALU a k-step, 8 / 16 (`PEARL_W_XALU`) | 86.5 / 83.0 | 6 / 8 |
| copy two slots ahead, release fence at the slot's start (`PEARL_W_A2`) | 90.5 | 8 |
| two column windows a work-group (RW 4, CW 2) / with KPS 2 | 64.5 / 86.9 | 3 / 6 |
| **two 8x4 split DPAS for each 8x8 (`PEARL_W_D4`, kept)** | **94.43** | **0** |
| D4 + A' a k-step ahead (kept) | 94.93 (D4 alone 94.37) | 0 |
| D4 + 256-byte B' reads / copy two slots ahead / 16 row windows / copy once a chunk | 94.2 / 93.1 / 94.1 / 92.1 | 0 |
| D4 + AH, options `-noroundrobin` / + `-keepDPASMacroInSchedule` (kept) | 95.6 / 96.4 | 0 |
| the same without `-insertDPASRSDummyMov` / without `-noloopra` | 95.4 / 96.2 | 0 |
| spare20 + readout as 16-lane 3-input XORs in inline vISA (`PEARL_W_ASMRD4`) | 93.8 | 0 |
| spare20 + no hash / no copy with real B' (timing) | 96.2 at 1,855 MHz / 100.5 at 1,936 MHz | 0 |
| spare20 + copy after the local reads / load and store in one k-step / two slots / KPS 2 | 93.5 / 92.1 / 94.6 / 91.5 | 0 |
| spare20 + 32-bit surface addresses (`PEARL_W_ARGP`) / B' straight from memory, no ring | 96.3 / 84.3 | 0 |
| spare20, band 16 / 24 / 32 / 48 / 128 rows (default 64: 96.1-96.5) | 96.1-96.4 / 96.1-96.2 / 96.4-96.9 / 96.5-96.7 / 90.0 | 0 |

What it shows: with no dummies left, the card sits at its power limit (188 W, about 1,900 MHz), so a cut in work a
clock (no hash) no longer raises the rate. The copy is what remains: about 4% of rate and of energy. `xmx8r` also
gains a little under the new options (69.0 against 68.5).

### Arc rig, 12 November, afternoon: 256-byte messages by inline vISA (`goal/arc-13`)

IGC splits every 256-byte block read into two 128-byte messages, so earlier "256-byte read" builds (B8, CPH) never
halved anything. Inline vISA sends one `lsc_load ... d32x64t` message, which fills registers in memory order, so a
`v8i` read this way holds the same dwords. `xmx8w` now reads A' as 2 messages a k-step instead of 4 (each sub-group
reads both halves of one 8-row block, so the split DPAS pairs rows of one class and the readout sorts by block) and
B' as 4 instead of 8 (63e23f71). The same commit fixes `xmx8w-emu`, which the 8x4 change had broken (its readout read
the 8x4 accumulators in the reference path too). eaca1720 drops `-keepDPASMacroInSchedule` from the host's default
options: with the new reads, HiveOS's 24.35 finalizer got 16,3xx of 16,384 regions wrong under it, a different count
each run.

- **Card 4 (cool, coordinator's check and mine): spare20 97.0-97.4 TH/s; spare21 (63e23f71) 99.2-99.5 against spare20
  97.1-97.4 at 1,908-1,915 MHz, 188 W. spare22 (eaca1720): 99.8-100.0 against spare20 97.6-97.7 at 1,919-1,923
  MHz, 188 W, `xmx8r` 67.9-68.2; verify.sh PASS (emu too), verify-hits 400 of 400 for `auto` and for `xmx8w`.**
- **spare23 (ac2edcc2: band 32 row windows for `xmx8w`, an eighth of the cache) on cool card 4: 100.11-100.14 TH/s
  in all three rounds, against spare22 99.86-99.88, at 1,915-1,919 MHz, 188 W.** verify.sh PASS (emu too),
  verify-hits 400 of 400. Card 5 98.4-98.6 against 97.9-98.1; 24.35 host driver PASS, 88.7 (spare20's 88.2).
- Card 5, three rounds: spare22's code 98.0-98.1 against spare20 96.2 (+1.9%); verify.sh PASS with emu at 16,384 and
  32,768 regions, verify-hits 400 of 400. 24.35 host driver: check PASS, 88.3 (spare20 88.2).
- Card 5 is held by PL2 (i915 throttle reason) at 1,900 MHz in 50 MHz steps; the bench's best batch, the first,
  runs at 2,000 MHz and 102 TH/s.

Card 5, 8 s, 2-3 rounds (wrong-result timing builds marked):

| What | TH/s | against |
|---|---|---|
| two chunks a loop iteration (`PEARL_W_CHUNROLL` 2) | 93.7 | 96.5 |
| `IGC_LscLoadCacheControlOverride` 2 / 6 / 7 (all loads) | 95.2 / 95.5 / 95.3 | 96.4 |
| readout of 4 / 1 of 8 groups a chunk, the rest once at the end (timing, `PEARL_W_RDH`) | 96.65 / 98.2 | 96.5 |
| 16 row windows, at 1,957 MHz | 93.9 | 96.4 |
| no copy, real B' (timing), 8 / 16 row windows | 100.5 / 97.5 | 96.35 / 93.9 |
| B' as 256-byte messages (`PEARL_W_BA`) | 96.3 at 1,880 MHz | 96.4 |
| copy as 256 bytes every other k-step (`PEARL_W_CA`), alone / with BA / with BA, address written another way | 95.3 / 97.6 / 95.9 | 96.35 |
| **A' as 256-byte messages (`PEARL_W_AA`) / AA + BA (kept)** | **97.4-97.9 / 97.9-98.0** | 96.1-96.5 |
| AA + BA + CA / AA + CA | 93.8-94.8 / 92.8 | 97.9 |
| AA + BA + copy once a chunk as 512 bytes (`PEARL_W_GA` 0 / 2) | 92.4 / 94.6 | 97.9 |
| AA + BA, without A' ahead / copy two slots ahead / forced load wait | 95.8 / 97.2 / 88.4 | 97.9 |
| AA + BA, options without `-keepDPASMacroInSchedule` (kept) / + `-SWSBDepReduction` / without `-noloopra` | 98.0-98.1 / 97.8-98.0 / 97.9-98.2 | 97.9-98.0 |
| spare22's code, band 16 / **32 (kept)** / 128 row windows (default was 64) | 98.0-98.3 / **98.5-98.6** / 95.0 | 98.0-98.1 |

Toward 105 (the founder's next goal), on spare23's code (card 5, 98.5; timing builds marked):

| What | TH/s |
|---|---|
| no copy (NOCOPY + FILLALL) / no hash (NOEPI) (timing) | 104.8 / 100.0 |
| no local store, loads kept (NOPUT + FILLALL) / no load, store kept (NOFETCH) (timing) | 103.3 at the 2,000 MHz cap, 172 W / 97.0 |
| half the copy, 128-byte messages at even k-steps (STE, timing) | **105.3** at 1,970 MHz |
| copy of two pieces at even k-steps (C2) / 256-byte store with 128-byte loads (CB) / CA at k-steps 1, 3 | 94.3 / 94.5 / 95.5 |
| 64-byte stores (STH, timing) / the store by inline vISA | 94.7 at 2,000 MHz / 98.9-99.2 |
| 8x8 DPAS with AA's layout (16 dummy DPAS) / B' straight from memory as 256-byte messages | 91.4 / 81.9 |
| 16 row windows a work-group (its old 64-byte copy) / + copy of 128 bytes at even k-steps (C16) | 99.3 / **99.6** |
| C16, its no-copy and no-hash ceilings (timing) | 102.3 / 101.2 |
| C16 without the inline B' read (shipped: 24.35 fails with it at 16 windows) | 98.5-98.9 |
| C16, one column window at a time / band 64 / band 16 | 93.0 / 98.6-98.8 / 98.1-98.2 |

The copy's local store is xmx8w's largest cost, and its bytes a MAC are what count: half the copy a thread
(STE) reaches 105.3, but any regrouping of the same bytes into fewer or bigger messages loses 4-5%. Only a work-group
of more rows (16 row windows, e7a3e10a / be43b135) cuts the bytes a MAC for real, and it gains 0.3-1.1%. On cool
card 4, spare25 (be43b135) ran 100.30-100.37 against spare23 100.24-100.30 (verify.sh PASS with emu, 400 of 400):
level, so the mining cards stay on spare23.
HiveOS's 24.35 finalizer mis-schedules the inline-vISA B' read at 16 windows (a B' register reloaded with no
wait on the DPAS reading it), so be43b135 turns that read off again.

What the first rows showed: for A' and B' reads, messages a k-step matter (halving them gave +1.7%). At the time,
halving the copy's bytes a MAC (16 row windows, old 64-byte copy) left its cost at 3.8%. Every copy reshaped into bigger messages lost, with the clock
rising (threads waiting): the copy still costs about 4%. `d64x64t` (512 bytes) does not assemble on DG2.

### Arc rig, 12 November, late afternoon: A' from L2 is the cost to cut (`goal/arc-14`)

Card 5 (PL2-throttled at 175 W), `pearl_sycl_check --bench 8`, taking turns, two rounds; base is 26fa1e41 (16 row
windows, C16 copy), 98.8-99.1 TH/s at about 1,940 MHz. Timing builds give wrong results.

| build | TH/s | MHz | W |
|---|---|---|---|
| 8 row windows (RW 8) | 97.6-98.0 | 1,900 | 176 |
| no copy (NOCOPY + FILLALL), RW 16 / RW 8 | 101.6 / 104.0 | 1,954 / 1,929 | 175 |
| no copy, no hash (NOEPI), RW 16 / RW 8 | 102.5 / 105.6 | 1,954 / 1,916 | 176 |
| no copy, no barrier, RW 16 / RW 8 | 100.3 / 103.5 | 1,971-1,996 / 1,939 | 175 |
| RW 8 padded to 36 KB of local memory (one work-group a core) | 78.5 | 2,000 | 165 |
| RW 16, two-slot ring (two work-groups, 64 threads a core) | 97.4 | 1,964 | 175 |
| RW 16, one column window a work-group (two work-groups) | 93.2 | 1,990 | 175 |
| four column windows a work-group in turn (`PEARL_SYCL_WSEQ=4`), PASS | 98.8-98.9 | 1,940 | 175 |
| band 16 / 64 / 128 | 98.4 / 98.8 / 96.7 | | 175 |
| `PEARL_SYCL_COL_BATCH` 512 / 128 | 95.3 / 96.5 | 1,860 / 1,940 | 175 |
| half the windows rotate each chunk's k-steps by 2 / by 1; odd windows by 2 (HROT), PASS | 93.7 / 97.5 / 97.2 | | 175 |
| the same, no copy | 97.6 (against 101.7) | 2,000 | 175 |
| copy spread: sub-groups 16-31 copy at odd k-steps (C16S), PASS | 96.6-97.0 | 1,940 | 175 |
| words by plain stores (NOPARTX) | 96.9, check FAIL | | 175 |
| **A' of only 2 row windows (AHOT 2: L1 hits)** | **104.3** | **2,000** | **163** |
| A' of only 16 row windows (AHOT 16: L2-resident) | 101.0 | 1,953 | 175 |
| **8 row by 2 column windows (RW 8, CW 2), PASS** | **99.4-99.5** | 1,968 | 175 |
| **RW 8, CW 2, B' as 256-byte messages (BA), PASS** | **100.1-100.3** | 1,957-1,993 | 175 |
| RW 8, CW 2 at band 64 | 98.8 | 1,980 | 175 |

- The RW 16 and RW 8 no-copy loops are the same ISA, instruction for instruction, yet RW 16 does 3.5-5% less a
  clock: one 32-thread work-group a core (40 KB of local memory) against two 16-thread ones. Not the hash, the
  barrier, the band or the work-group's start.
- A' from L2 is a large cost. goal-arc12's row "A' of 8 row windows only 93.5, A' costs nothing" timed the shipped
  code: its AHOT macro changed a pointer the 32-bit address path never reads. Wired correctly, A' read by 8 pairs
  at once (L1 hits) reaches the 2,000 MHz cap with 12 W to spare.
- So the route to 105 is fewer A' bytes from L2 a MAC: windows that share A' through L1, in step. CW 2 does it
  for two column windows; with BA it is the best build of the day. HiveOS 24.35 (staged /home/shawn/arc/ship-a14,
  card 5): xmx8w and xmx8w-emu PASS three times, bench 90.7 against ship-a13d 90.8 (level there).
- Keeping the last k-step's B' live to make BA safe at RW 16 on 24.35 does not reach the ISA (IGC drops asm
  operands the text does not name). BA at CW 2 passes on 24.35 as it is.

### Arc rig, 12 November, evening: 8 by 2 windows verified and made the default (`goal/arc-15`)

Cool card 4 (cards 1-3 mining beside it), `race5.sh 4 8 3`, three rounds, against spare23 (ac2edcc2, mining).

| build | TH/s | MHz | W |
|---|---|---|---|
| goal-arc14's opt-in (RW 8, CW 2, BA): verify.sh **FAIL** at the mainnet shape | 101.8-101.9 | 2,000 | 188 |
| **spare26: RW 8, CW 2, BA as the default, CW 1 where 2 does not divide the column windows; verify.sh PASS (emu too), 400/400** | **101.6-101.8** | 1,992-2,000 | 188 |
| spare23 beside it | 100.2-100.3 | 1,915-1,931 | 188 |

- verify.sh's mainnet check runs a 4-column batch: one column window. A work-group of two column windows then
  reads a window that is not there (256 regions wrong, 0 of 46 hits). verify-hits (2,048 columns) passed, so only
  the full verify.sh caught it. The host now runs CW 1 there (`fold_xmxw<.., RW, 1>`).
- HiveOS 24.35 (staged /home/shawn/arc/ship-a15, card 5): xmx8w, xmx8w-emu and the mainnet check PASS; bench 90.6
  against ship-a13d 90.7-90.8, level there.
- One card 5 verify.sh run had 8 regions wrong in both dot and xmx8w at the mainnet shape. Both folds agreeing
  points at the host reference. The same build passed three checks straight after and the full rerun. spare23's
  build does it too (1 run in 4): a rig flake, not a fold's.

**Under the mining duty CW 2 loses.** The cards mine at `PEARL_SYCL_DUTY` 60 and 75. Moved to spare26 for 8 minutes,
the rig mined 252 TH/s against about 260 on spare23 (card 4 75.3 against 77.5). The miner loop (bench.js, which
applies the duty) on card 5, 40 s runs, two rounds:

| build | duty 75 | duty 60 | full speed |
|---|---|---|---|
| spare23 (RW 8, CW 1) | 76.85-76.98 (2,097-2,109 MHz busy) | 60.70-60.73 | 98.15-98.17 (1,883 MHz, 190 W) |
| spare26 (RW 8, CW 2) | 74.49-74.55 (2,097-2,114) | 58.61-58.69 | 98.88-98.96 (1,970 MHz, 190 W) |
| CW 1 under duty, CW 2 at full speed (spare27) | 76.89-76.95 (2,082-2,119) | 60.57-60.71 | 98.73-98.83 (1,962 MHz) |

- CW 2 saves energy (A' from L1). That pays only on a card held at its power limit, where the saved watts become
  clock. Under duty the card rests between batches and runs above 2,050 MHz while busy, so the clock is free and
  the work a clock decides: CW 2 does about 3% less a clock. The host now runs CW 1 whenever the duty is under 100
  (`xmx8w_cw`; `PEARL_SYCL_CW=1` or `2` forces it).
- Price any new design both ways: the cool-card race (full speed) and bench.js at duty 75.

More card 5 prices on 8a01fa49 (def 100.2-100.5 TH/s at about 1,970 MHz, 50.9 a GHz; timing builds are wrong):

| build | TH/s | MHz | W |
|---|---|---|---|
| A' of 2 row windows (AHOT 2) | 102.3-102.4 | 2,000-2,029 | 166 |
| no copy (NOCOPY + FILLALL) | 104.2 | 2,027 | 172 |
| half the copy (STE + FILLALL) | 104.2-104.4 | 2,000-2,027 | 167 |
| no hash (NOEPI) | 101.4-101.5 | 1,961 | 177 |
| readout of 1 group of 8 (RDH 1), PASS | 100.8-101.0 | 1,968-1,979 | 175 |
| RW 8 x CW 4 (64 sub-groups), KPS 1, PASS | 89.9 | 2,000 | 163 |
| RW 8 x CW 4, KPS 2, SEQ 1 (SEQ 2 needs 80 KB: does not launch) | 82.2 | 2,000 | 155 |
| RW 8 x CW 2, KPS 2, PASS | 98.0-98.4 | 2,000 | 175 |
| RW 16 x CW 2 (64 sub-groups, C16 copy for CW 2), SEQ 1, PASS; with BA | 88.6; 92.6 | 2,000 | 163; 166 |
| def at SEQ 1 | 96.9 | 2,000 | 175 |

- Card 4 check of spare27 (eb1133b3): verify.sh PASS (emu too), verify-hits 400/400 at full speed and at duty 75;
  full speed 101.1-101.4 against spare23 99.7-99.9; duty 75 (bench.js, 40 s) 77.09-77.38 against 77.22-77.41.

Prices under duty 75 (bench.js on card 5, 30-40 s, two rounds, ~2,100 MHz busy, below the power limit; spare27 at CW 1
76.88-76.97). Under duty, work a clock is all that counts:

| build | TH/s | busy MHz | W |
|---|---|---|---|
| duty 100 / duty 99 (the drain each batch) | 98.15-98.29 / 96.76-97.14 | 1,881-1,904 | 190 |
| 4,096-column batches | 75.2 | 2,016 | 168 |
| no copy (NOCOPY + FILLALL) | 80.08-80.23 (+4.2%) | 2,102-2,116 | 173 |
| A' of 2 row windows (AHOT 2) | 79.67-79.77 (+3.6%) | 2,093-2,126 | 162 |
| no hash (NOEPI) | 77.69-77.74 (+1.0%) | 2,106-2,109 | 177 |
| RW 16 x CW 1 (spare25; current tree the same) | 74.81-74.94 | 2,096-2,126 | 171 |
| CW 2 squeezed to 32 KB (KPS 2, SEQ 1), PASS | 70.75-70.83 | 2,099-2,109 | 163 |
| two work-groups on the same rows (PAIR 2, CW 1), PASS | 73.69-73.74 | 2,106-2,112 | 180 |

- Work a clock falls as the clock rises: 52.1 TH/s a GHz at 1,885 MHz, 48.6 at 2,110 (same build).
- Under duty, A' from L2 costs work a clock (+3.6% if it were all L1 hits), but every way of sharing it tried so far
  (CW 2 at 64 or 32 KB, RW 16, PAIR) loses more than that. Two 16-thread work-groups a core at 32 KB, each on its own
  rows, is the shape to keep; the copy (4.2%) and A' (3.6%) must be cut inside it.
- A work-group of 64 sub-groups (a whole Xe-core) loses 8-12% however its windows are arranged.
- AHOT 2 does the same work a clock as def: at the clock cap, sharing A' more widely buys nothing more.

### Arc rig, 12 November, evening: where the last 10% a clock goes (`goal/arc-16`)

No new verified build; this section prices the loop. Card 5, `pearl_sycl_check --bench 8`, two or three rounds taking
turns, nearly all at the 2,000 MHz cap, so the rates compare as work a clock. Control: spare27's code (eb1133b3),
100.0-100.4 TH/s at 1,970-2,000 MHz, 175 W. The goal (105 on card 4, which sits at 2,000 MHz) needs 52.5 TH/s a GHz
against 50.5-50.7 now.

**The loop itself is at the DPAS peak.** New timing macro `PEARL_W_NOLD` holds A' (bit 1) or B' (bit 2) in registers,
loaded once with real bits. With both held, no copy and no barrier, the fold runs 111.5-112.0 TH/s at 2,000 MHz and
144 W: 55.8 a GHz against the probe's 56.4 for DPAS alone. The 8x4 split DPAS, the readout and the hash cost about
1%. All of the gap is getting operands in:

| Timing build (wrong results) | TH/s | W | against the skeleton |
|---|---|---|---|
| skeleton: no A' or B' loads, no copy, no barrier | 111.5-112.0 | 144 | - |
| copy and barrier only | 110.2-110.4 | 157 | -1.3% |
| B' local reads only | 106.0-106.2 | 150 | -5.0% |
| B' local reads, copy, barrier | 106.0-106.1 | 162 | -5.1% |
| A' loads only | 107.9-108.1 | 175 | -3.4% |
| A' loads only, all from 2 row windows (L1 hits) | 109.7-109.8 | 154 | -1.8% |
| A' loads, copy, barrier | 103.9-104.3 | 175 | -6.9% |
| B' local reads only, B' read a k-step ahead (32 more registers) | 109.0-109.2 | 157 | -2.4% |
| B' local reads only, each fragment as 128 bytes used twice (same messages, half the bytes) | 105.9 | 147 | -5.2% |
| skeleton + 2 / 4 unused 256-byte local reads a k-step | 109.0 / 107.8 | 155 / 152 | -2.5% / -3.6% |
| skeleton + 2 / 4 unused 32-byte local reads a k-step | 109.5 / 109.2-109.3 | 146 / 148 | -2.1% / -2.3% |
| skeleton + 2 unused 256-byte global reads of a hot line | 110.3-110.5 | 151 | -1.3% |
| 8x8 split DPAS skeleton (16 dummy DPAS) / with B' reads | 99.6 / 95.0 | 142 / 146 | |

- B' read from local memory is the largest item, then A' (half of it misses), then the copy and barrier.
- Bytes do not set the price: half-size B' reads cost the same. Messages do, but not linearly: the first one or two
  in a k-step cost about 2% and further ones little, so cutting a message buys less than the table's rows suggest.
- B' latency is real when nothing else loads (a k-step ahead recovers 2.8 of the 5%), but not in the full fold.

Full builds, all PASS unless noted (control 100.0-100.4):

| What | TH/s |
|---|---|
| B' a k-step ahead, all 4 fragments (`PEARL_W_BAH`) / 1 / 2 / 3 fragments | 68.8 / 95.1 / 96.5 / 85.7 |
| B' 4 ahead with A' not ahead / 2 ahead with A' not ahead | 95.9 / 95.7 |
| B' read into the same registers right after its DPAS (`PEARL_W_BROLL`) | 95.9 (in the B'-only build 105.2: the load waits on the DPAS) |
| the above with the finalizer's scheduler off (`-noschedule`; control 97.3 with it off) | 92.2-95.8 |
| `-keepDPASMacroInSchedule` on the control / on B' 4 ahead | 99.4-99.6 / 96.7-96.8 |
| B' reads before the copy and A''s next load (`PEARL_W_BFIRST`) / A''s next load after the DPAS (`PEARL_W_AHLATE`) | 99.8-100.0 / 98.6-98.7 |
| 1 / 2 B' fragments straight from global memory (`PEARL_W_BSPLIT`; same work a clock, more power) | 98.5-98.6 / 97.5-97.6 |
| all B' from global memory, no ring, copy or barrier / and the ring's local memory freed (`PEARL_W_NORING`) | 89.9 / 87.6 |
| copy as 256-byte messages every other k-step, now at CW 2 too (`PEARL_W_CA`) / at k-steps 1, 3 / unrolled twice | 94.3 / 94.9-95.0 / 97.1 |
| chunk loop unrolled twice (`PEARL_W_CHUNROLL=2`) | 100.5-100.6 (+0.2%) |
| no release fence on the arrive (unsafe) / no barrier at all | 99.7 / 97.3-97.4 (1,934 MHz, 50.3 a GHz) |
| band 16 / 32 / 128 rows (default 64) | 99.9-100.0 / 100.2 / 97.1 |
| 512-byte SIMD8 gather (`d64x8`) | does not assemble: one message brings at most 8 registers |

The barrier is worth keeping: without it the fold does less work a clock (the windows drift apart and stop sharing A'
lines in L1). Every way of hiding B''s latency tried here loses in the full fold, where registers and the finalizer's
placement of the loads decide; IGC put several of the new loads' addresses and results in the same registers, so
the loads wait on each other.

### Arc rig, 12 November, evening: the hash tail, meetings, threads and prefetch (`goal/arc-17`)

No new verified build. Card 5, `pearl_sycl_check --bench 8` taking turns (2 rounds) unless noted; control spare27's
code, 100.3-100.4 TH/s at 1,955-1,995 MHz, 175 W (50.9 TH/s a GHz). Under duty: `bench.js` with
`PEARL_SYCL_DUTY=75`, 20 s a run.

**The hash at the end of a work-group costs 1.6% a clock, and nothing tried can hide it.** At 8 by 2 windows only one
work-group fits an Xe-core (64 KB of local memory), so while it hashes the core's DPAS sit idle.

| What | TH/s |
|---|---|
| no hash (`PEARL_W_NOEPI`, timing) | 101.4 (51.7 a GHz) |
| one set a work-group (`PEARL_SYCL_WSEQ=1`) | 96.8-96.9 |
| hash in a second kernel after the fold, one set (`SEPEPI` + `SEPBLK`) / without the hash kernel | 94.7-94.9 / 99.3-99.6 |
| the hash kernel on the side queue, so the next fold need not wait (`PEARL_W_SEPQ`, bench.js full speed) | 93.1-93.6 against 98.9-99.1 |
| no hash, one shared part buffer (`PEARL_W_PART1`), 2 / 4 / 8 / 16 sets a work-group | 101.8-101.9 / 102.2 / 97.0 / 74.7 |
| 16 or 32 hasher sub-groups added to the work-group (not kept) | device lost, also with no hashing |

- The driver does not run the hash kernel beside the next fold, so a second kernel only adds its time.
- Extra sub-groups in an xmx8w work-group break the split DPAS (as the copier sub-groups did).
- Longer work-groups are not worth it without hiding the hash: 4 sets gain 0.3%, 8 or more lose.

**Meetings, gating, threads:**

| What | TH/s |
|---|---|
| the group meets every 2 / 4 chunks (`PEARL_W_BARN`, unsafe timing) | 97.2 / 96.9 |
| one more meeting mid-chunk (`PEARL_W_BARX`) | 100.2-100.3 |
| every DPAS of a k-step gated on all its loads (`PEARL_W_GATE`; runs of 24-32 DPAS) | 88.7 |
| CW 1 at KPS 2 (16 KB): 4 / 3 / 2 work-groups a core (`SLMPAD`) | 95.3-95.4 / 95.4-95.5 / 95.5-95.6 |
| CW 2 at KPS 2, one set (32 KB): 2 work-groups a core / padded to 1 | 92.0 / 93.2 |

More threads on a core hide nothing. The two runs of 16 DPAS a k-step are not the cost.

**Prefetch** (inline vISA load with a null destination, one message each):

| What | full speed | duty 75 |
|---|---|---|
| control | 100.3-100.4 | 76.9-77.0 |
| A' of the work-group 2 k-steps ahead, 256 bytes a sub-group (`PEARL_W_PFA=2`) | 96.5 | 74.1 |
| A' 4 ahead (`PFA=4`) | FAILS the check | |
| the copy's block 4 / 8 k-steps ahead (`PEARL_W_PFB`) | 99.7 / - | 76.8-76.9 / 76.9 |

Latency is not what costs, at full speed or under duty; the extra message does.

**A 64-thread work-group** (RW 16 x CW 2, half the copy a MAC), one set, all at the ~2,000-2,029 MHz cap, so these
compare as work a clock:

| What | TH/s | W |
|---|---|---|
| control | 96.8-96.9 | 174 |
| control, no copy (`NOCOPY` + `FILLALL`) | 97.7-97.8 | 166 |
| RW 16 x CW 2 (PASS) / no copy | 92.6 / 92.6 | 166 / 160 |
| no barriers: control / RW 16 x CW 2 | 95.8-95.9 / 89.2-89.3 | 176 / 171 |

The copy costs 8 W and no work a clock. The 64-thread work-group saves the energy but does 4% less work a clock,
and that loss is neither its copy nor its meetings.

### Arc rig, 12 November, night: why a 64-thread work-group loses, and the readout's atomics (`goal/arc-18`)

No new verified build. Card 5, `pearl_sycl_check --bench 8` taking turns, 2-3 rounds; control spare27's code,
100.2-100.4 TH/s at 1,960-1,995 MHz, 175 W (96.8 at one set a work-group, at the 2,000 MHz cap). Skeleton = A' and
B' held in registers, no copy, no barrier (`NOLD=3 NOCOPY NOBAR`), all at one set (`PEARL_SYCL_WSEQ=1`).

**A 64-thread work-group runs the bare DPAS loop 15% slower.** It is not its loads, its copy or its meetings:

| Threads a core, as | skeleton | B' reads only | A' loads only | full build |
|---|---|---|---|---|
| 1 work-group of 32 (RW 8 x CW 2, def) | 112.0-112.2 | 108.6-108.7 | 108.2-108.5 | 96.8 |
| 1 work-group of 64 (RW 16 x CW 2) | 95.8 | 94.3 | 92.4-92.5 | 92.5-92.7 (PASS) |
| 2 work-groups of 32 (local memory cut, `NORING`) | 112.1-112.2 | | | |
| 4 / 2 work-groups of 16 (CW 1) | 112.7 / 112.5-112.8 | | | |
| 1 work-group of 16 (padded) | 102.0-102.2 | | | |
| 1 work-group of 32 as RW 16 x CW 1 | 112.2 | | | |

- The same 64 threads a core from two or four work-groups run at full speed; only one work-group owning every slot
  loses. The loop's ISA is the same as def's but for register numbers. Thread placement (an `sr0` probe):
  sub-groups 0-15 take slot 0 of all 16 EUs, 16-31 slot 1, 32-47 slot 2, 48-63 slot 3; sub-groups 2i and 2i + 1
  are the two EUs of a pair. That is where two work-groups of 32 would sit too.
- Of the 15%: the hash 5.8% (skeleton without it 101.3; at 32 threads it costs nothing there), the readout's local
  atomic 2.4% (`PEARL_W_NOPXA` 103.8), the rest unexplained. Longer work-groups (SEQ 2-16 with `PART1`) do not
  help (98.3-98.9; without the hash 99.4-101.4), nor does starting half the threads late (`PEARL_W_PHASE`).
- So RW 16 x CW 2 cannot reach def's work a clock, and the 8 W it saves cannot be had that way.
- 64 KB of local memory a core, confirmed: a 16-thread work-group padded to 33 KB or to 60 KB runs the same.

**The readout's atomic and the hash:**

| What | TH/s |
|---|---|
| no local atomic a chunk (`PEARL_W_NOPXA`, timing) | 100.8-100.9 (+0.6%) |
| the atomic's words laid out without bank conflicts (`PEARL_W_PXT` 1 / 2, PASS) | 95.4 / 96.1 |
| two chunks a 64-bit atomic | OUT_OF_RESOURCES |
| two-slot ring (`PEARL_W_SLOTS=2`), PASS / + plain stores, no atomic (`NOPARTX`, fixed), PASS | 100.1-100.2 / 100.4-100.6 |
| the latter under duty 75 (bench.js, 30 s) | 76.2-76.3 against 76.9 |
| no hash at one set / at two sets (`PEARL_W_NOEPI`) | 100.5 (def 96.8) / 100.8-101.0 at a lower clock |
| two work-groups of 32 a core at 4 k-steps a slot (`SLOTS=2 PARTMIN`, 32 or 28 KB), one set | 98.9-99.2, the same padded to one |

- PXT keeps the same work but IGC rebuilds the chunk loop around the new addresses and loses 4-5%.
- A second work-group on the core hides neither the hash nor the loads.

**New default: the copy's load as one inline-vISA message** (`PEARL_W_CPCC`, CW 2 instance only; spare28). IGC's
block read sends the same 128-byte message; the gain is where the volatile load lands in the k-step.

| What (card 5 unless noted) | full speed | duty 75 |
|---|---|---|
| spare27's code | 100.0-100.5 | 76.9 |
| inline copy load, ca.ca / uc.ca / st.ca | 101.2-101.6 / 101.3-101.4 / 101.3-101.4 | 76.5-76.6 (on the CW 1 instance too) |
| inline copy load, uc.uc | 71.9 | |
| + inline store (STA) / + SLOTS 2 NOPARTX / + CPD2 / + BFIRST / not volatile | 100.6-100.8 / 100.2-100.4 / 94.4-94.7 / 98.7-99.0 / 101.2-101.5 | |
| **inline copy load on the CW 2 instance only (spare28)** | **101.2-101.6** | **76.9-77.0** |
| **card 4: spare28 / spare27** | **102.5-102.7 / 101.4-101.6** (1,986-2,000 MHz, 176 W) | **77.36-77.46 / 77.44-77.52** |

spare28 on card 4: verify.sh PASS (dot, xmx8w, xmx8w-emu; mainnet-shaped 46 of 46), verify-hits 400 of 400 at duty
100 and at duty 75. HiveOS 24.35 (staged ship-a18): xmx8w, xmx8w-emu and the mainnet-shaped check PASS. Under the
mining duty it is level with spare27, so moving the mining cards gains nothing yet.

Candidate for the duty path, not yet the default or checked on card 4: the same inline load on the CW 1 instance
with L1 uncached or streaming (`PEARL_W_CPCW=1`, `PEARL_W_CPCC="uc.ca"` / `"st.ca"`), card 5, duty 75, 30-40 s,
five rounds against spare28's code: 76.98-77.16 / 77.03-77.21 against 76.79-76.95 (+0.3%). With ca.ca it lost 0.4%.
Copy store after the k-step's local reads (`CLATE`) on top of spare28: 100.1-100.2 against 101.5-101.6.

### Arc rig, 12 November, night: the duty path's copy load, and where else the k-step's messages can go (`goal/arc-19`)

**New default: the CW 1 instance sends the copy's load inline too, streaming in L1** (`PEARL_W_CPCC1`, default
"st.ca"; `PEARL_W_NOCPC1` turns it off). The CW 2 instance, which runs at full speed, is unchanged (same ISA as
spare28). Frozen as spare29.

| What | full speed | duty 75 (bench.js, 40 s) |
|---|---|---|
| card 4: spare29 / spare28, 3 rounds | 102.60-102.64 / 102.61-102.71 (2,000 MHz, 176 W) | |
| card 4: spare29 / spare28, 8 rounds | | 77.27-77.50 / 77.08-77.42 (2,084-2,127 MHz, 175 W) |

spare29 on card 4: verify.sh PASS (dot, xmx8w, xmx8w-emu; mainnet-shaped 46 of 46), verify-hits 400 of 400 at duty
100 and at duty 75 (the CW 1 instance). HiveOS 24.35 (staged ship-a19): xmx8w and xmx8w-emu PASS at both column
widths (CW 1 forced too), mainnet-shaped dot and xmx8w PASS; bench 90.7 against ship-a18's 90.7-90.8. The gain under
duty is within noise on card 4 (+0.1%; +0.3% on card 5), so moving the mining cards is optional.

**Nothing else found toward 105.** Card 5, `pearl_sycl_check --bench 8`, 2-3 rounds, default 101.1-101.6 TH/s at
1,960-1,980 MHz, 175 W. Every variant below PASSes the check unless marked timing.

| What | TH/s |
|---|---|
| copy load after A''s next load / after the B' reads / between the DPAS halves / after the DPAS (`PEARL_W_CPLD` 1-4) | 100.3-100.6 / 98.0-98.1 / 98.0-98.1 / 98.7-98.9 |
| copy load before the group's wait (`CPLD=-1`) | 100.8-100.9 |
| copy store after the B' reads / between the DPAS halves / after the DPAS (`PEARL_W_CPST` 2-4, arrive with it) | 95.6-95.8 / 95.8-96.0 / 91.7-91.8 |
| copy load 1 + store 2 | 95.3-95.5 |
| A''s next load before the group's wait (`PEARL_W_AEARLY`) / + copy load there | 95.3-95.7 / 94.8-95.2 |
| SLOTS 2 + NOPARTX on the CW 2 instance only (`PEARL_W_S2CW=2`, emu PASS) | 100.4-100.5 |
| no release fence before the arrive (`NOREL`, timing) | 100.6-100.9 |
| no readout atomic (`NOPXA`, timing) | 101.4-101.7 (+0.15%) |
| two chunks a loop pass (`CHUNROLL=2`) | 100.8-101.1 |

- The finalizer moves the copy's messages across k-steps whatever the source order; the k-step's start is still the
  best place for both the load and the store.
- On this base the readout's atomic costs ~0.15% and the release fence nothing: neither is worth a redesign.
- Finalizer options on top of the default set, one at a time (2 rounds): `-nopresched`, `-presched`,
  `-scheduleForReadSuppression`, `-sinkBarrierWait`, `-SendTokenReduction` level (101.0-101.5);
  `-schedWithSendSrcReadCycle`, `-multiplePipeSched` 101.1-101.3; `-noSendSrcDstOverlap` 100.8-101.1;
  `-globalTokenAllocation`, `-distPropTokenAllocation`, `-quickTokenAllocation`, `-dontUseMultiThreadedLatencies`
  100.4-100.9; `-sendQueueSched` 99.7-99.9; `-nodefhoist` 99.5-99.9; `-enableDpasFwd` does not run.

### Arc rig, 12 November, late night: the part buffer's bank conflicts, and two structural ideas priced (`goal/arc-20`)

**New default: the part buffer laid out so a sub-group's 8 lanes touch 8 banks** (`PEARL_W_PXB`, `PEARL_W_PXA` on
goal-arc18's `PXT=2` layout; `PEARL_W_NOPXB` for the old one). In the old layout a lane's 16 words lay 64 bytes from
the next lane's, so the 16 stores that zero them at each set's start, the 16 reads before the hash, and every
chunk's atomic put all 8 lanes on one bank. Now word j of region q is at j * 8 + q: zeroing and reading are two
256-byte block messages each (a block read gives lane q, register j exactly word j of region q), and the atomic is
inline vISA with its address stepped by 32 bytes a chunk. goal-arc18's PXT alone lost 4-6% because its zeroing and
reads stayed 16 messages each and IGC rebuilt the chunk loop around them; with block messages it does not. Frozen
as spare30.

| What (card 5 unless noted) | full speed | duty 75 (bench.js, 30 s) |
|---|---|---|
| spare29's code / new, 6 rounds | 101.27-101.47 / 101.54-101.78 (new at the same or a lower clock) | 77.13-77.25 / 77.97-78.00 |
| PXT 2 / PXT 1 alone (the old zeroing and reads) | 95.4-95.8 | |
| block zeroing and reads, IGC's atomic (`PXB` only) | 101.4-101.6 | |
| timing: no atomic (`NOPXA`) / no atomic and a 64-word part (`PARTMIN`) | 101.5-101.7 / 102.1-102.4 | |
| HiveOS 24.35 (staged ship-a20) against ship-a19, `--bench 8` | 91.8 / 90.7 | |

The rest of `PARTMIN`'s lead (~0.5%) is the atomic itself; a form without it needs the two sub-groups' words
combined some other way, and their own copies of the part do not fit 64 KB at two sets.

Card 4 (cool), spare30 against spare29, taking turns: full speed 103.50-103.52 against 103.33-103.40 (1,986-2,000
MHz, 176-177 W, 3 rounds; +0.15%); duty 75 (bench.js, 40 s) 78.17-78.19 against 77.44-77.48 (+0.9%, 3 rounds).
verify.sh PASS (dot, xmx8w, xmx8w-emu; mainnet-shaped 46 of 46), verify-hits 400 of 400 at duty 100 and 75.
HiveOS 24.35: xmx8w and xmx8w-emu PASS at both column widths, mainnet-shaped dot and xmx8w PASS. The mining cards
run under duty, so this is the first build since spare27 worth moving them to.

**Priced and dropped** (card 5, `pearl_sycl_check --bench 8`, def 101.3-101.6 at 1,960-1,980 MHz):

| What | TH/s |
|---|---|
| transposed fold, timing (`PEARL_W_TR`): B' the split operand, 2 local reads a k-step; A' the full one, 4 global reads, in the k-step / a k-step ahead / half ahead | 91.3 / 73.1 (spills) / 85.2 |
| the same against def with A' in the k-step (`PEARL_W_AH=0`) | 97.0-97.1 |
| one / two of the 4 B' fragments through the sampler from a bindless image (`PEARL_W_SMP`, timing; 2 SIMD8 sample loads a fragment) / one fetched a k-step ahead | 95.8-96.0 / 72.4-72.6 / 95.1-95.3 |
| A' rolled into its own registers (`AROLL`), on this base | 98.6-98.7 |
| 4 row by 4 column windows (A' shared by 4), KPS 2 / KPS 1 | 88.9 / 87.2 |
| no hash (`NOEPI`) | 101.3-101.5 at a 10 MHz lower clock (+0.35% a clock) |
| two 32 KB work-groups a core (`PARTMIN SLOTS=2 NOPXA`) / padded to one; each without the hash | 102.4-102.5 / 102.2-102.3; 103.6 / 103.4-103.7 |
| the second work-group on each core in a launch held back first (`PEARL_W_STAG`, ALU spin of 0 / 2k / 20k / 40k / 80k steps) | 101.8 / 101.2-101.5 / 100.4-100.6 / 99.0-99.5 / 96.5 |

- The transposed fold sends the same 6 messages a k-step, but 4 global 256-byte reads cost far more than the 2
  local ones they replace (-6% against def with A' in the k-step), and A' ahead needs 32 more registers.
- The sampler has the bandwidth in a probe, but in the fold its latency stalls the k-step.
- A second work-group on the core does not take the hash tail off the DPAS, in step or held out of step; the
  per-work-group ticket alone costs 0.7%.

### Arc rig, 13 November, small hours: a work-group's two sets as one stream, opt-in (`goal/arc-21`)

**New, opt-in: the set stream** (`PEARL_SYCL_SEQP=1`; FoldXmxW's template switch `SP`, the CW 2 instance only).
Each of a work-group's two sets used to fill the ring's first slot from L2 and then meet the whole group at its
end, with every thread waiting. Priced on spare30's code: skipping one set's fill (timing, `PEARL_W_NOPRO`) is
+0.5%. Now the sets run as one stream of 128 k-steps: the ring's slots count on across sets, set s of column
window jc takes column block base + jc SEQ + s so each sub-group's copy runs from one set's B' straight into the
next set's (no clamp; the host leaves 16 KB after each operand for the reads past the end), only the first set
fills and arrives, only the last waits and meets. A' is loaded again at each set's start.

HiveOS's 24.35 driver gets most regions wrong in this instance (and in every other form tried: A' carried into the
next set, the copy by a select or by interleaved columns, the slot index as a rotation), while the newer driver
passes them all, so the host runs it only when asked. Off, the build's ISA is spare30's. Frozen as spare31.

| What (card 5, full speed, `pearl_sycl_check --bench 8`) | TH/s |
|---|---|
| spare31 off / on (`PEARL_SYCL_SEQP=1`), same build | 101.6-101.8 / 102.9-103.0 |
| on, with the slot index as a rotation added after the % 3 (passes 24.35 with the rest of the stream off) | 101.7-101.8 |
| on, A' carried into the next set in registers (made per-lane to IGC; as a uniform value it read wrong A') | 100.0-100.4 |
| on, at duty 75 (bench.js, 40 s; CW 1 instance with the stream) | 77.2-77.5 against 77.9 |

Card 4 (cool), spare31 with `PEARL_SYCL_SEQP=1` against spare30, taking turns: full speed 104.06-104.11 against
103.05-103.08 (1,971-1,989 MHz, 176-177 W, 3 rounds; +1.0%); duty 75 (bench.js, 40 s) 78.18-78.23 against 78.15-78.22 (level: under duty the CW 1 instance runs, without the stream). verify.sh PASS
(dot, xmx8w, xmx8w-emu; mainnet-shaped 46 of 46), verify-hits 400 of 400 at duty 100 and 75, all with the stream
on. HiveOS 24.35 (staged ship-a21), stream off: xmx8w and xmx8w-emu PASS at both column widths, mainnet-shaped dot
and xmx8w PASS, `--bench 8` 91.8 against ship-a20's 91.4-91.8; stream on: 16,384 of 16,384 regions wrong.
Card 4 ran ~0.5% slower tonight than at 23:15 (spare30 103.05 against 103.50 then, 1,975-1,989 MHz at 176 W).

**Where a clock goes now** (card 5, timing builds on spare30's code, def 101.6-101.8 at ~1,960 MHz; a clock = TH/s
a GHz, def 51.8):

| Taken out | TH/s, MHz, W | a clock |
|---|---|---|
| A' loads (`NOLD=1`, A' held in registers) | 107.2-107.3, 2,000-2,053, 160 | +3.4% |
| A' misses only (`AHOT=2`, A' of 2 row windows) | 105.3, 2,000, 169 | +1.6% |
| the copy (`NOCOPY FILLALL`) | 105.1-105.2, 2,000, 171 | +1.5% |
| B' local reads (`NOLD=2`) | 102.6, 2,000, 174 | -1% |
| the hash (`NOEPI`) | 101.8-102.0, 1,939-1,946, 177 | +1% |
| the meetings (`NOBAR`, unsafe) | 96.8, 1,961, 176 | -5% (windows drift apart, A' sharing lost) |
| A' loads and the copy / and the meetings | 108.1 / 107.2, 2,000-2,027, 149-153 | +4.3% / +3.3% |
| everything but the DPAS loop (`NOLD=3 NOCOPY NOBAR`) | 112.0, 2,000-2,029, 144-146 | +8% |

**Priced and dropped** (card 5, full speed):

| What | TH/s |
|---|---|
| def (spare30's code) | 101.6-101.8 |
| 8 / 16 extra 8-lane ALU ops a k-step (`XALU`) | 95.7-95.9 / 88.6-88.7 |
| every message's address made inside the asm, a chunk base plus an immediate (`PEARL_W_SA`; 356 loop instructions for 410) | 98.5-98.7 |
| the ring pointer per-lane to IGC, so a local read's address is one 8-lane add (`PEARL_W_VZB`; 391 for 410) | 101.6-101.9 |
| a plain store where the chunk's local atomic was (`PEARL_W_PXST`, timing) | 101.7-101.8 |
| B' fragments read just in time, 2 / 1 ahead (`PEARL_W_BJIT`) | 101.7-101.9 / 100.8-101.1 |
| A' two k-steps ahead with BJIT 2 / 1 (no spill now) | 94.4-94.6 / 95.7-95.8 |
| the second column window's k-steps rotated by 1 / 2 / 3 (`PEARL_W_HROTJ`) | 100.6 / 99.6 / 99.2-99.3 |
| the readout XORed in the last k-step, 2 / 1 fragments behind the DPAS (`PEARL_W_RDI`) | 101.9-102.1 / 91.2-91.4 |
| on the stream: A' L1 prefetch without PFA's loop, 2 / 3 k-steps ahead (`PEARL_W_PFS`) | 99.1-99.2 / 100.4-100.5 |
| on the stream: VZB / BJIT 2 / RDI 2 / VZB and RDI 2 | level with the stream (102.0-102.4) |
| on the stream: 4 sets a work-group, timing (`PART1`, one part for all sets) / 8 | +0.5% / -4.5% |
| 4 sets for real, a two-slot ring to make room (`SLOTS=2`), SEQ 2 / 4 | 101.0-101.2 / 101.3-101.5 |
| 4 sets for real, the words in global memory (each sub-group stores its chunk word, the hash reads them back), SEQ 4 / 2 | 87.7-88.1 / 90.1 |
| a persistent kernel (28 work-groups walking the batch) | every region wrong on the card, even at one pass each |

- Instruction count is not what the loop pays for: SA and VZB cut 54 and 19 instructions from the chunk loop and
  gained nothing; SA lost 3% because the finalizer stopped overlapping a k-step's messages with the last one's
  DPAS. XALU's cost is its dependent chains, not its count.
- A block read's result is a uniform value to IGC (its address is), and that breaks a value carried across an
  outer loop: A' carried from set to set read wrong in set 1, and wrapping the kernel in a persistent loop broke
  every region. Making the value per-lane fixes it but costs 2-6%.
- Four sets a work-group would be worth +0.5%, but their words do not fit 64 KB of local memory beside the
  ring, and both other homes cost more than that.
- Global words indexed by Xe-core (sr0) were wrong: two 48 KB work-groups share a core.

### Arc rig, 13 November, 02:00-03:00 UTC: priced and dropped, the stream on 24.35 bisected (`goal/arc-22`)

Nothing new ships. The build's ISA is spare31's on every path (check run, bench with the stream off and on;
diffed), so card 4's best stays 104.06-104.11 (spare31, `PEARL_SYCL_SEQP=1`).

**What limits the clock now.** At full speed with the stream on, card 5 has i915's PL2 throttle reason set in 37 of
43 samples, its clock stepping between 1,950 and 2,000 MHz at about 175 W (hwmon's PL1 is 190 W). Builds that do
less work a clock sit at a 2,000 MHz cap at about 173 W. So the default runs power-bound about 2% under the cap:
a build pays for its watts as well as for its work a clock.

| What (card 5, full speed, stream on, `pearl_sycl_check --bench 8`) | TH/s |
|---|---|
| def (spare31's code) | 102.9-103.0 |
| the chunk loop unrolled 2 / 4 (`CHUNROLL`; no A' copies, 188 / 183 ALU a chunk for 203) | 102.1-102.2 / 101.9-102.0 |
| finalizer `-forceBCR` / `-enableBCR` (the loop's DPAS bank conflicts 62 to 39 / 96) | 96.5 / 95.8-96.2 |
| 4 / 8 sets a work-group (`PART1` timing, wrong results; 2 sets 102.7) | 102.9-103.2 / 99.1-99.3 |
| an L1 prefetch of A' that touches each 64-byte line once (8-lane gather, a dword a lane, half a message a k-step; `PEARL_W_PFG` 2 / 3 / 5 k-steps ahead) | 100.4-100.5 / 100.4-100.5 / 101.0-101.2 |

- The chunk loop starts with 16 movs that copy the loop-carried A'. IGC scalarizes the vector phi, and the
  carried block lands on an odd register while DPAS reads A' as an even pair. Unrolling removes them and loses
  1%; `IGC_DisablePHIScalarization=1` moves the copies and adds more (219 loop ALU for 203); carrying A' as the two
  raw loads (`PEARL_W_AV`), loading it after the k-steps (`PEARL_W_AHEND`) or padding registers (`PEARL_W_PADR`)
  leave them where they are (ISA only, not timed). Unrolling 16 adds 32 dummy DPAS.
- A prefetch message costs as much as a load whatever its size: half a message a k-step costs 2-2.5%.

**The stream on HiveOS 24.35, bisected.** `PEARL_W_SQX` switches the stream instance's three parts on one at a
time (staged builds, host check at 2048 with `PEARL_SYCL_SEQP=1`):

| Parts on | 24.35 |
|---|---|
| none / interleaved column blocks (1) | PASS / PASS |
| slots counting on across sets (2), as `(k + 64 set) / 4 % 3` / without a division (`PEARL_W_SQM`) | 16,384 wrong / PASS |
| 1 and 2, with SQM | PASS |
| the stream (1, 2, 4) with SQM; also with every piece of 4 undone in turn (a meeting between sets, each set zeroing its own words, refilling its first slot, reloading the copy's hold, no barrier under a condition) | 16,384 wrong |
| 1 and 2 with SQM, no stream, the copy's loads past the set's last k-step unclamped / clamped at 71 (never fires) / wrapped to k-steps 0-4 (`PEARL_W_SQUC` 1 / 2 / 3) | 16,384 wrong each (all pass on the new driver) |

- 24.35 gets two things wrong here. The first is the slot's division by 3, which IGC writes as a mul and a mach
  through the accumulator. Since 16 = 1 (mod 3), the slot is (k / 4 + set) % 3, which a 32 by 16-bit multiply
  computes; that form passes. The second is the copy's load past a set's last k-step: whatever it loads there
  goes into a slot no one reads before it is filled again, yet on 24.35 every region is wrong unless the load
  repeats k-step 63. The stream needs that load, so it stays opt-in. Cause not found.
- `PEARL_CHECK_WORDS=1` makes `pearl_sycl_check` print, for the first batch, how many regions are wrong at each
  transcript word (word j is chunk j's readout). On 24.35 the stream is wrong from chunk 0 in both sets.
- Default build on 24.35 (staged ship-a22-base): xmx8w and xmx8w-emu PASS at both column widths, mainnet-shaped
  dot and xmx8w PASS, `--bench 8` 91.7-91.9 against ship-a21's 91.8-91.9.

### Arc rig, 13 November, from 03:00 UTC: the set stream on HiveOS, and where a batch's time goes (`goal/arc-23`)

**Ships: the set stream on HiveOS's 24.35 driver, by default there.** A new instance of the stream (FoldXmxW's `S24`,
`PEARL_SYCL_SEQP=2`) in forms 24.35 compiles right; the host picks it when the device's OpenCL driver version starts
with "24." and keeps the stream off elsewhere, as before. Staged default build on the HiveOS host (card 5): xmx8w and
xmx8w-emu PASS at 2048 at both column widths; `--bench 8` 94.8-94.9 TH/s against 91.8 for the same build with
`PEARL_SYCL_SEQP=0` and ship-a20 (+3.3%; an earlier pair of runs gave 94.5-94.7 against 93.6). The mainnet-shaped check
(dot and xmx8w alike) shows 0 to 40 wrong regions run to run on this host for every build, ship-a21 included: a
host flake, not this change. On the newer driver nothing changes: the default and `SEQP=1` ISA are the branch's
(diffed), and `S24` there runs 99.9 against 101.6 with the stream off and 102.9 with `SEQP=1`, so it stays a 24.x
choice. Card 4's best stays 104.06-104.11 (spare31, `PEARL_SYCL_SEQP=1`; the branch's stream runs level with
spare31's, 102.9-103.0 each on card 5). New measurement switches, off by default: `PEARL_W_PROF` (the
check bench times the fold by GPU timestamps, `PEARL_W_PROFGPU=1`) and `PEARL_W_TSC` (each work-group's first start,
last end and Xe-core for one batch, to `PEARL_W_TSCOUT`; read with `S/arc-k/tsc23.py` and `tsc23b.py`).

**A batch's timeline** (card 5, stream on, one mainnet-shaped batch: 8,192 work-groups, 21.0 M timer ticks at the
~1.98 GHz clock = the 10.6 ms the bench reports, so launch gaps are nothing):

- 55.5 of the 56 work-group slots (two a core) are busy throughout; the change-over between work-groups costs ~0.9%,
  the last wave 0.3%.
- A band's first column of work-groups runs 5.6% longer, and so does the work-group sharing its Xe-core (index +28:
  the same rows, 7 columns on), together ~0.75% of the batch. With A' always in L2 (`AHOT` 32, timing) both vanish
  and the batch is 0.9% shorter: they wait on A' rows coming from DRAM, which the band's first column reads first.
- Band 0's work-groups run 50-87% longer (~0.5% of the batch). Keeping B' to 8 blocks that are the same every batch
  does not help, so nothing survives in L2 from one launch to the next.
- B' stays in L2 for the whole batch (`BHOT` 8 / 32: 102.4-102.7 against 102.8-103.0). A' from DRAM is what costs:
  `AHOT` 32 runs 104.4-104.8 against 102.9 (+1.7%) at a higher clock and the same watts. A is 256 MB, read once a
  batch; a wider batch would read it less often, but B' then leaves L2 (column batch 1,024: 96.7; 128: 100.5).

**Prefetching the next band's A' into L2.** A load with no destination register does nothing on this card, whatever
its cache control: prefetches issued that way change neither the timeline nor the rate (so the PFA, PFB, PFG and PFS
prefetches of earlier goals never fetched). Real loads do fill L2, and an atomic load with no destination does too,
but each costs more than the 0.75% it saves:

| What (card 5, full speed, stream on, `pearl_sycl_check --bench 8`) | TH/s |
|---|---|
| def | 102.8-103.0 |
| null-destination loads of the next band's A', from the band's first 64 / 16 / 4 work-groups (`PEARL_W_PFN`) | 102.6 / 102.7-102.8 / 102.6-102.8 |
| real loads from the hash tail, 2 / 1 k-steps of every 16-row block a work-group (`PEARL_W_PFT`; each work-group then ends ~2.5 us later) | 101.1-101.2 / 101.7-101.8 |
| the same as atomic loads with no destination, 2 / 1 k-steps (`PEARL_W_PFTA`) | 101.2 / 101.5-101.8 |
| atomic loads from the band's last 16 / 32 work-groups as they start (`PEARL_W_PFS2` 4 / 2; every work-group ~4% slower) | 101.5-101.6 / 101.9-102.0 |

**Operand energy.** NVIDIA's tensor cores take 2% fewer watts with unsigned operands; Arc's DPAS does not care. With
fill 63 (B' 0..126, so u8 is exact for it): i8 x i8, i8 x u8, u8 x u8 (`PEARL_W_U8`) all 102.8-103.0 at 175 W.
Fill 40 / 48 / 56 / 63 level; 32 loses 0.6%.

| Also priced | TH/s |
|---|---|
| 4 sets a work-group (`PART1` timing, wrong results) against 2 | 102.8-103.2 against 102.7 (+0.2%) |
| a gap after each B' block (`PEARL_SYCL_BPAD` 256 / 4,096; timing, the stream needs none) | level |

**The stream on HiveOS 24.35.** Staged builds checked on the host (`hq23.sh`), ISA dumped there (`hd23.sh`):

- Without the stream, the copy's loads past k-step 63 break every region even when their bytes are never stored
  (`PEARL_W_SQNS`: 14,040 of 16,384 wrong, mostly as other regions' right words), and pass once an instruction reads
  the copy's hold after the chunk loop (`PEARL_W_SQWT`: 0 wrong, three runs). So 24.35 seems to let the set's last,
  dead copy load land after its register has been loaded again; re-reading k-step 63 (an L1 hit) lands first,
  which is why the clamp passed. The same addresses written another way fail too (`PEARL_W_SQRO`).
- The stream needs one more change on top: the run-on address written as the clamped address plus a separately
  selected offset (`SQRO` 1). With the plain form or a multiplied offset every region is still wrong, wait or not.
  All three together pass (0 of 16,384, deterministic over repeats): that is `S24`.
- Each costs on the newer driver: the slot without a division 0.35%, the hold read 1.9%, the selected offset 1.3%.
- Reading A' after the loop as well (`SQWT` 2) breaks even forms that pass (15,128 wrong): leave A' alone there.
- arc22's bisection never reloaded the copy's hold: the switch is `PEARL_W_SEQPRL`, not `SQPRL`.

### Arc rig, 13 November, from 04:45 UTC: why the mainnet-shaped check failed on card 5 (`goal/arc-24`)

**The folds were right; card 5's copies to the host lose data.** The check's host reference is made from the
operands read back from the card (768 MB). On card 5, now and then a 64-byte line of the host buffer is never
written by that copy: it keeps whatever was there before (zeros in a fresh buffer). The reference made from it is
wrong, so dot and xmx8w "fail" on the same regions: the 4 regions of a batch that use that A row, or the 16-row
group of a B column. That is the 0-72 wrong regions seen with every build, ship-a21 and spare31 included, in the
container and on the HiveOS host.

- Reading the operands back a second time (`PEARL_CHECK_DIAG`, build `diag`): in 11 of 15 runs on card 5 one of
  the two reads differed from the other by 64-319 bytes, always whole host cache lines, in random rows of A or B.
  The fold results matched the read that was not missing the line, and the reference made again from the same host
  copy came out the same: the host's arithmetic and memory are fine.
- Card 4 (paused from mining for it, then put back on spare30): 21 runs, 63 reads of 768 MB, not one byte
  different, every check PASS.
- A plain SYCL copy test (`S/arc-k/copytest24.cpp`: a device buffer of a known pattern, copied to the host and
  checked) loses lines on card 5 with Level Zero and OpenCL, into pageable and USM host memory, in one 768 MB copy or
  in 4 MB pieces. So it is not the runtime or the buffer kind; it follows the card (or its riser or slot), not the
  data. No AER or EDAC counters are readable without root on this host, and the link shows 2.5 GT/s x1 like the
  others.
- verify-hits stays 400/400: each hit is checked from a few kilobytes, so a lost line almost never touches one.
- Fix in the check: it reads the operands twice and, where they differ, a third read decides byte by byte. It
  prints a note naming the lost bytes and fails only if the reads cannot be settled. Card 5 with it: every
  mainnet-shaped check that ran to the end passed, 17 in the container and 10 on the HiveOS host, with 64-510
  bytes settled in most runs; verify.sh PASS.
- Five runs ended in a GPU hang (`12:2:ecffffe4`) while container and host runs overlapped on card 5; 8 runs alone
  after that, none. The copy test hit the same hang code on cards 4 and 5, so it looks like long copies, not card 5.

Also priced, card 5, full speed, `PEARL_SYCL_SEQP=1`, `pearl_sycl_check --bench 8`, two rounds: default 102.89-102.89 (two runs);
`PEARL_SYCL_COL_BATCH` 384 102.76-102.94 (the batch stays 2,097,152 regions), 512 96.9-97.0 (B' leaves L2);
`PEARL_SYCL_BAND` 64 102.4-102.7, 16 102.3-102.6. The defaults stay. The best single batch reads 105.3-105.6
against a mean of 102.9 in every one of these runs.

### Arc rig, 13 November, from 05:30 UTC: why batches vary, and wider batches under duty (`goal/arc-25`)

**Ships: under duty, xmx8w runs 1,024 column offsets a batch instead of 256**, walked as four chunks of 256 one after
the other (FoldXmxW's CW 1 instances, the duty path), so one chunk's B' (8 MB) is what sits in L2. A card at full speed
keeps 256 and its kernels are unchanged (ISA diffed against spare31 on the check and bench paths, stream on and off).
Card 4, `PEARL_SYCL_SEQP=1`, bench.js (the miner's own loop), 30 s runs:

| | old (spare31's code) | new |
|---|---|---|
| duty 75, three rounds | 78.20-78.21 TH/s, 176 W | 79.67 TH/s, 172 W (+1.9%) |
| duty 60, two rounds | 61.70-61.71, 139-140 W | 64.09, 139 W (+3.9%) |
| full speed, `pearl_sycl_check --bench 8`, three rounds | 104.62-104.78 | 104.61-104.72 (same kernel) |

Under duty the core waits for each batch, rests, and launches the next; after each rest the clock has to come back
up. That is a fixed cost a launch, and four times fewer launches pay it four times less often. 2,048 column offsets
did better still at duty 75 (80.4, 167 W) but was not taken: a batch is then 84 ms. verify.sh (emu included) PASS and
verify-hits 400/400 at duty 100 and 75 on card 4 (and on card 5 for an earlier form that walked chunks in
every instance at 1,024); HiveOS 24.35 host (card 5): the 2048 check, CW 1 forced,
the mainnet-shaped check and, with a build of 8-window chunks (`PEARL_W_CHUNKW=8`, so the 2048 check's batches walk
several chunks), every chunked mode PASS; `--bench 8` 94.75-94.85, as before. Not taken at full speed: bench.js at
duty 100 gained only 0.35% (103.3 / 102.9), and four times the batch means four times the work searched for an old
job after a job switch.

**Why batches vary** (new timing switches: `PEARL_W_PROF` with `PEARL_W_BLOG=1` prints each batch's wall time and
GPU span to stderr; `S/arc-k/a25blog.sh` samples clock, throttle reasons and energy every ~30 ms beside it):

- Card 5: the first ~0.6 s of a run runs at 2,000 MHz with no limit set, 10.43-10.50 ms a batch. Then PL2 sets in,
  the clock alternates 1,950/2,000 (mean ~1,972) at ~188 W, and every batch takes 10.60-10.67 ms. The "best batch"
  of 105.3-105.6 is that first half second, not a fast batch among slow ones. After it the spread is small
  (p10-p90 10.61-10.68 ms) and nothing in the batch sequence repeats (autocorrelation 0 at every lag). So the
  spread is the power limit, and only fewer joules a MAC close it.
- Card 4: 2,000 MHz nearly throughout (PL2 in 5% of samples), 190 W. GPU time per batch 10.46 ms mean, 10.44 median:
  105.1 TH/s on the GPU's own clock, steady over 50 s. The bench reads 104.2 because each batch also waits ~0.09 ms
  for the host (submit, wait, wake). The miner queues two batches and does not pay that, but bench.js at duty 100
  still reads only 103.0-103.4: with no gaps the card hits 190 W and runs at ~1,970 MHz.
- Redraws (`PEARL_W_RESEED=N` on the check bench redraws A' every N batches, as the miner does every salt): 3.36 ms
  a redraw (tree 1.16, A' noise 2.0, dense 0.3, perm 0.1), every 64 batches at mainnet: 0.5% of mining time.
  Ordering the A' writes so a sub-group fills whole 512-byte k-steps changed nothing (2.43 against 2.45 ms), so the
  write pattern is not what costs there.

### Arc rig, 13 November, from 06:30 UTC: a cheaper redraw, and what job switches cost under duty (`goal/arc-26`)

**Ships: a same-job redraw hashes A's tree path on the host.** A redraw only changes A's first 11 bytes (chunk 0),
so only node 0 of each tree level changes. After a job's full draw the host keeps chunk 0 and node 1 of each level,
checks that its own hashing gives the device's root, and from then on hashes each redraw's path itself; one small
kernel writes it into the tree. No wait for the root. If the check ever disagrees, redraws rebuild the tree on the
device as before. Kernels and the full-speed rate are unchanged.

| | before (spare32's code) | after |
|---|---|---|
| redraw, median (card 5, `pearl_sycl_check --bench 6`, `PEARL_W_RESEED=4`) | 3.78-3.85 ms | 2.68-2.69 ms |
| the same on the HiveOS 24.35 host (staged ship-a26-def vs ship-a25-def) | 3.20-3.29 ms | 2.43-2.45 ms |
| card 5, bench.js, duty 60, five rounds of 40 s | 63.86-63.92 | 63.65 (first round), then 63.96-64.02 (+0.15%) |
| card 4, bench.js, duty 75, three rounds of 30 s | 79.61-79.69 | 79.73-79.92 (+0.2%) |
| card 4, bench.js, duty 100 (190 W, power-bound), two rounds | 101.2-101.6 | 101.4-101.4 (level) |
| card 4, `pearl_sycl_check --bench 8` (card warm from mining), three rounds | 102.8-103.5 | 102.8-103.0 (same kernel) |

verify.sh PASS (emu included) and verify-hits 400/400 at duty 100 and 75 on card 4 and card 5; HiveOS 24.35 host:
2048 check (xmx8w, emu, CW 1) and the mainnet-shaped check PASS, plain bench level (94.6-94.8 each). One verify.sh run on
card 5 failed dot and xmx8w alike on 128 regions; two reruns passed, one noting 1,337 bytes settled by the third read:
card 5's lossy copies (goal-arc24), not this change.

**A' noise is not cheaper another way.** It is 2.0 ms of the 2.7 ms that is left (a plain memset of the same 268 MB:
0.69 ms). Rewritten four ways, all byte-identical to materialize16 at mainnet size: a sub-group of 16 holding the dense
row and picking bytes with shuffles 7.3 ms; the same with plain loads and the (p0, p1) pairs kept for 16 rows 2.0 ms
(64 rows 2.1, 4 rows 6.2); E_AL stored transposed so a sub-group's 16 rows read one cache line a byte, with 32-byte
vector stores filling whole 512-byte blocks 2.0 ms; the same at 4 rows a work-item 2.8 ms. With the dense reads
removed (wrong results, timing only) the kernel still took ~1.4 ms, so it is neither the reads nor the store pattern.

**Job switches under duty.** The Kryptex pool sent a new job every 45 s on average (38 jobs in 28 minutes, worker
`jobprobe`, no shares). On a job switch the core finishes what it queued under the old job, and those hits are dropped
(pearlMiner.js). Under duty that is two batches: the one running and the one launched after the rest. At 1,024
column offsets and duty 60 that is ~84 ms of card time every 45 s, ~0.3% of a mining card's rate. Resting before the
job check instead of inside `pearl_host_submit` would make it ~0.1%, but needs the search loop (shared with the CUDA
and HIP cores) and a new host call; not done. Batch width under duty, card 5, bench.js, duty 60, after this change:

| column offsets a batch | TH/s | waste at a job every 45 s |
|---|---|---|
| 512 (four rounds) | 62.87-62.98 | ~0.15% |
| 1,024, today's (four rounds) | 63.94-64.01 | ~0.3% |
| 2,048 (three rounds, before this change) | 63.89-63.91, level with 1,024 | ~0.6% |

So 1,024 stays: 512 loses 1.6%, 2,048 gains nothing at duty 60 and wastes twice as much at each job switch.

### Arc rig, 13 November, from 07:40 UTC: where xmx8w's watts go at full speed (`goal/arc-27`)

Nothing ships; the default build's ISA is spare33's (diffed on the check run and the bench, stream off and on). At full
speed both cool cards sit at the 190 W power limit (card 5 ~1,965 MHz, card 4 ~1,985 in the check bench, ~1,970 in
bench.js), so the rate there is 190 W divided by the joules a MAC costs. New: `S/arc-k/e27.sh CARD SECS ROUNDS
'label|BUILD|ENV'...` runs the check bench and reads clock, watts (hwmon energy) and throttle reasons beside each run;
the figure below is W / (TH/s), joules per peta-MAC. Each timing build removes or changes one part of the k-step; all
are wrong-result builds except where marked. Two rounds of 10 s each, taking turns; runs repeat within 0.5%.

| build (switches) | what it removes | card 5 J/PMAC | card 4 J/PMAC |
|---|---|---|---|
| def | nothing (stream off, the new driver's default) | 1.867 (101.7 TH/s, 1,965 MHz, 190 W) | 1.841 (103.2, 1,985, 190) |
| def, `PEARL_SYCL_SEQP=1` (real) | the set stream on | 1.844 (-1.2%) | 1.823 (-1.0%) |
| `NOLD=3 NOCOPY NOBAR` | every message in the loop (A' and B' in registers) | 1.396 (-25%), 112.0 at 2,005 MHz, 157 W | 1.403 (-24%) |
| `NOLD=1` | A' loads (A' in registers) | 1.621 (-13%) | 1.606 (-13%) |
| `AKM=0` (new) | A' from L2 and DRAM (every k-step reads k-step 0's A': L1 hits) | 1.725 (-7.6%) | 1.704 (-7.4%) |
| `AHOT=32` | A' from DRAM only (32 row windows' A' stays in L2) | 1.822 (-2.4%) | 1.806 (-1.9%) |
| `CPHOT=0` | the copy's loads from L2 (L1 hits instead) | 1.828 (-2.1%) | 1.794 (-2.6%) |
| `AMASK=0` | A' bits (the DPAS see zeros; loads kept) | 1.662 (-11%) | 1.635 (-11%) |
| `BMASK=0` | B' bits | 1.675 (-10%), with 11% less work a clock | 1.667 (-9%) |

From `AKM=0 CPHOT=0` ("hot": A' and the copy's loads all L1 hits, real bits; card 5 1.648, card 4 1.630), each message
kind taken out in turn, as a share of def's joules:

| removed from "hot" | messages a sub-group a k-step | card 5 | card 4 |
|---|---|---|---|
| B' local reads (`BREG`, new: B' held in registers, real bits) | 4 of 256 bytes | 7.7% | 7.5% |
| A' loads, L1 hits (`NOLD=1`) | 2 of 256 bytes | 5.0% | 4.9% |
| the copy, load and store (`NOCOPY FILLALL`) | 2 of 128 bytes | 2.5% | 2.8% |
| the copy's load only (`NOFETCHR`, new: stores without loading, real bits) | 1 | 1.4% | 1.8% |

So of def's joules: ~21% is the DPAS seeing real operand bits, ~25% is the k-step's messages and L2/DRAM traffic (B'
reads 7.7, A' L1 5.0, A' L2 5.2, A' DRAM 2.4, copy 2.5, copy L2 2.1), and the rest is the DPAS, readout, hash and the
card's clocked power. None of the message items is a whole message a sub-group that can go: the tile (32 rows by 32
columns a sub-group, 8 by 2 windows a work-group) already sends the fewest messages a MAC that fit 256 registers and
64 KB, the twin column windows' A' reads already share L1 (`AKJ=0`/`1`, new: one twin's A' made hot, 1.870 / 1.862,
level), and A' must come from DRAM once a batch (256 MB) because B' takes 8 MB of the 16 MB L2.

Correct forms priced for energy, card 5 (def 1.867 unless noted), all level or worse:

| change | J/PMAC | |
|---|---|---|
| operand fill 63 / 63 with unsigned B' (`PEARL_W_U8=2`) | 1.867 / 1.869 | level |
| fill 32 / -63 / 0 | 1.878 / 1.896 / 1.960 | +0.6 / +1.5 / +5.0%: negative values cost; 48 stays |
| 8x8 DPAS (`NOD4`, 13 dummies) | 1.985 | +6.3% |
| 8x2 DPAS (`D2T`, new, timing) | 2.196 | +18% (and spills) |
| no meeting (`NOBAR`, unsafe) | 1.964 | +5.2%: the windows drift apart and lose L1 sharing |
| 256-byte copy every other k-step (`CA`) | 2.003 | +7.3% |
| chunk loop unrolled twice (`CHUNROLL=2`) | 1.895 | +1.5% |
| copy load uncached / streaming in L1 (`CPCC` uc.ca / st.ca), stream on (def 1.844) | 1.856 / 1.856 | +0.6% |
| B' reads as d64x32t (`D64`), stream on | 1.845 | level |
| the same B' fragment for all 32 DPAS of a k-step (`BSEL=0`, new, timing) | 1.852 | -0.8%: which src1 the DPAS switch between barely matters |

Rest under duty, to keep the clock off the power limit: card 5, bench.js, stream on: duty 100 101.2-101.9; duty 99 /
98 / 96 / 93 on the duty path (CW 1, 1,024 columns) 98.7-98.9 / 98.0-98.3 / 97.0-97.4 / 95.1 (the clock falls, not
rises: CW 1 costs more joules a MAC); duty 99 / 98 / 97 with `PEARL_SYCL_CW=2` 99.9-100.0 / 99.2 / 98.7. Card 4, bench.js:
duty 100 103.0-103.2 (1,968 MHz, 190 W), duty 99 with CW 2 101.6-101.8. A rest costs more than the clock it buys.

So 105 on card 4 in the miner (103.0-103.2 now) needs ~2% fewer joules a MAC with no loss of work a clock, and no single
part of the k-step that can change offers that. The check bench on card 4 read 104.0-104.4 with the stream on today
(104.6-104.8 on goal-arc25's cooler card).

### Arc rig, 13 November, from 09:00 UTC: the set stream by default on newer drivers (`goal/arc-28`)

**Ships: xmx8w's set stream (`PEARL_SYCL_SEQP=1`) is the default on a Linux driver known newer than 24.x.** The host
now reads the driver version on Level Zero too (the rig's container: Level Zero "1.18.40196", OpenCL "26.40.40196.15";
before, a Level Zero device always got 0). New `seqp_default`: a 24.x OpenCL driver keeps 2 (HiveOS); an OpenCL major
of 25 to 30, or a Level Zero build of 32,500 or more (24.52 is 32224, 25.05 is 32567), gets 1; anything else gets 0,
including Windows, where no Intel card has run the core. `PEARL_SYCL_SEQP=0` still turns it off. Kernels unchanged; the
duty path (CW 1) has no stream and is untouched.

| | stream off (`SEQP=0`, or spare33) | new default |
|---|---|---|
| card 5, `pearl_sycl_check --bench 10`, three rounds | 101.60-101.76 TH/s, 1.867 J/PMAC | 102.90-102.95, 1.844 (+1.2%; `SEQP=1` 102.88-103.02) |
| card 4, the same | 102.75-103.04 (spare33 102.70-103.02), 1.845 | 103.81-104.00, 1.827 (+0.9%) |
| card 4, bench.js duty 100, three rounds of 30 s | spare33 101.61-101.80 | 102.61-102.98 (+1.0%) |
| card 4, bench.js duty 75, three rounds | spare33 79.69-79.79 | 79.73-79.79 (level, same kernel) |
| HiveOS 24.35 host, card 5, `--bench 8` (default must stay 2) | `SEQP=0` 91.86-91.89 | 94.56-94.69 (`SEQP=2` 94.42-94.65; ship-a26-def 94.53-94.60) |

verify.sh PASS (emu and the mainnet-shaped check included) and verify-hits 400/400 at duty 100 and 75 on card 4 and
card 5, with no `PEARL_SYCL_SEQP` set. HiveOS 24.35 host (staged /home/shawn/arc/ship-a28-def): 2048 check (xmx8w, emu,
CW 1) and the mainnet-shaped check PASS (card 5's lossy copy settled by the third read), redraw path agrees.

**Messages a MAC, priced.** Using goal-arc27's per-message shares (a B' local read ~1.9% of def's joules; an A' load
~6% with its L2 and DRAM part, ~2.5% as an L1 hit; the copy ~4.6% a k-step), the sub-group tile's other shapes that fit
the same 128 accumulator registers both lose: 16 rows by 64 columns halves A' loads (-6%) but doubles B' reads (+7.7%)
and, with half the rows a work-group, doubles the copy a MAC (+4.6%); 64 by 16 halves B' reads and the copy (-6%) but
doubles A' loads (+12%). With DPASW halving A', 32 by 32 is the cheapest shape, not just the fewest messages. Nothing
built.

### Arc rig, 13 November, from 17:10 UTC: the cache hint on A', pairing an Xe-core's work-groups, and the readout (`goal/arc-30`)

Nothing ships; the default build's bench ISA is goal-arc28's (diffed). All runs on card 0 (card 5 has its copy fault,
cards 1-4 mine), `pearl_sycl_check --bench 10`, xmx8w, stream on, each run under a temperature guard (`tg30.sh`).
Card 0 at full speed sits at 190 W with i915's PL2 flag set in 40-90% of samples, so the rate is 190 W divided by the
joules a MAC; its J/PMAC rises ~1% as the card warms over a few runs, so the decisive runs below take turns with 50-60
s of rest before each. Rested, the default reads 104.50-104.53 TH/s run after run (1,975-1,981 MHz, 1.816-1.817 J/PMAC,
52.8-52.9 TH/s a GHz); the mining build (spare33) 102.4-102.9. 105 needs ~0.4% fewer joules a MAC; the clock cap under
full DPAS (2,005 MHz, no throttle flag set) is ~1.2% above.

**A cache hint on A' (the last untried lever on the list): loses.** Same ISA but the LSC cache control of the two
256-byte A' loads (`PEARL_W_ACC`): st.ca 99.5-100.4, uc.ca 98.6, ri.ca 98.4, against ca.ca 103.5-104.7 in the same
runs. The copy's load uncached in L1 (`CPCC uc.ca`): 52.4 TH/s a GHz against 52.85.

**Sharing A' between the two work-groups of an Xe-core.** Prize (timing, `PEARL_W_AKH=2`: every other column group
reads k-step 0's A', an L1 hit): 105.8-105.9 at the 2,005 MHz cap, 187.5 W, 1.773 J/PMAC; all A' hot (`AKM=0`) 106.4.
One mainnet batch's timeline (`PEARL_W_TSC`, `S/arc-k/tsc30.py`): a core's two work-groups are g and g + 28 45% of the
time (g + 27 or g + 29 another 30%), and start 64 k-steps apart (p10-p90 62.9-67.1 of a work-group's 128): one is in
set 1 while the other is in set 0, at nearly the same k. The default walk gives them the same rows 26.7% of the time.

| What (card 0) | TH/s | J/PMAC | a GHz |
|---|---|---|---|
| def | 103.6-104.7 | 1.813-1.833 | 52.8-52.95 |
| walk in runs of 4 over bands of 7 row blocks (`window_of_runs`, runtime band `PEARL_W_WALKR=4`, `PEARL_SYCL_BANDX=56`): same rows 47-59% | 104.0 | 1.826 | 52.6 |
| the same build at band 4 (no sharing) | 103.6-104.0 | 1.825-1.833 | 52.7 |
| the same two with the copy's load uncached in L1 | 102.7-103.6 / 102.5-103.7 | 1.832-1.853 | 52.1-52.3 |
| the walk with a compile-time band and run (`WALKQ`, 4 forms incl. the default walk itself) | 95.8-97.0 | 1.95-1.98 | 48.0-48.5 |
| goal-arc29's pairing builds, re-timed: ticket only / global-counter pairs / + start wait / + soft meeting a chunk | 102.3 / 101.7 / 90.5 / 86.5 | 1.855 / 1.867 / 2.054 / 2.109 | |

Pairs 64 +- 3 k-steps apart do not meet in L1, and every way of pairing work-groups explicitly costs more than the A'
it shares. The compile-time walk's 8.6% loss is register allocation alone: same loop shape, but the loop's loads land
on registers the next loads need. Eight do-nothing prologue changes (`PEARL_W_PERT` 1-7) gave eight different loops,
all 52.74-52.95 a GHz, and finalizer option sets on top of the default (drop -noroundrobin, -noloopra,
-DPASTokenReduction or -insertDPASRSDummyMov; add -keepDPASMacroInSchedule, -scheduleForReadSuppression,
-SBIDBundle) 52.64-52.90: the default's allocation is not unlucky.

**What is left, priced** (timing builds, card 0; first block one run each in a warming series, def 1.815 first /
1.834 last; second block rested turns, two rounds):

| Taken out | TH/s | J/PMAC |
|---|---|---|
| the copy (`NOCOPY FILLALL`) | 106.4 at 2,005 MHz, 184 W | 1.730 |
| A' from DRAM (`AHOT=32`) | 105.0 | 1.808 |
| the hash (`NOEPI`), rested | 105.09 / 105.09 | 1.807 |
| three quarters of the readout's XORs (`PEARL_W_RDA`, new), rested | 105.89 / 105.89, 1,992-1,996 MHz | 1.795 |
| the readout's atomic (`NOPXA`) | 103.9 | level |
| 4 sets a work-group (`PART1`, `PEARL_SYCL_WSEQ=4`) | 104.5, +0.2% a clock | level |

The copy and A' from L2 are the work-group tile's two costs, and 256 rows by 128 columns already balances them (~0.037%
of the joules a column for the copy, ~0.020% a row for A' from L2, area fixed by 32 threads of 32 by 32). The readout
is the surprise: ~64 three-input XORs a chunk are ~1.5% of the joules.

**Real forms tried, all PASS, all lose** (rested turns against def 104.50-104.53, 1.816-1.817):

| What | TH/s | J/PMAC |
|---|---|---|
| readout in two halves: fragments 0-1 at the chunk's end, 2-3 in the next chunk's first k-step before their DPAS (`PEARL_W_RDS`) | 102.94 / 103.01 | 1.843 |
| readout as 16-lane three-input XORs in inline vISA, chain / tree (`PEARL_W_RD16` 1 / 2) | 104.15 / 104.15 | 1.823 |
| inline messages' uniform address passed with IGC's "rw.u" constraint plus one scalar mov in the asm, so no 8-lane broadcast movs (`PEARL_W_RWM` 11 / 2 / 1 / 8; 40 of the loop's 56 eight-lane movs gone at 11) | 102.8 / 103.1 / 103.3 / 103.0 | 1.84-1.85 |
| 1,024 column offsets a batch at full speed, the CW 2 instance walking chunks (`PEARL_W_CHUNK2`, `PEARL_SYCL_WIDE=1`) | 104.55-104.58 (+0.7% a clock, 0.7% lower clock) | 1.816 |
| a chunk's first k-step loading its own A' (`AHK0`) / next A' after the k-steps (`AHEND` 1) / A' carried as two v8i (`AV`) / `CHUNROLL=2` | 50.9 / 52.1 / 52.6 / 52.0 a GHz | |

- "rw.u" alone (`PEARL_W_RWU`) gets every region wrong: IGC passes the uniform value at its sub-register, and the
  message reads from the register's start. With the mov it passes but the finalizer schedules the loop worse.
- Fewer ALU instructions only pay when the schedule stays; moving the readout into a k-step costs more than the
  chunk end's wait. The readout's prize (1.2% of the joules for three quarters of it) stays open: a form that keeps
  it at the chunk's end with fewer register reads, or no form at all.

### Arc rig, 13 November, from 19:45 UTC: 384 column offsets a batch at full speed, 105.2-105.3 TH/s (`goal/arc-31`)

**Ships: xmx8w at full speed takes 384 column offsets a batch instead of 256, with half the band (16 row windows
instead of 32).** Host only; the kernels and their ISA are unchanged. A batch walks every row band across all of
its columns, so its B' (32 KB a column offset at mainnet) has to stay in the L2 while A' (256 MB) streams from
DRAM once a batch. At 256 columns B' is 8 MB of the A750's 16 MB; at 384 it is 12 MB, and with the band's A' cut
from 2 MB to 1 MB both still fit. A' then comes from DRAM a third fewer times a MAC. `PEARL_SYCL_W_COLS` sets
another width (256: the old one); `PEARL_SYCL_COL_BATCH` still forces a power of two. Under duty (the CW 1
instance, 1,024 columns walked in chunks of 256) nothing changes. A salt no longer splits into whole batches, so
pearl_core.cc's search loop now redraws when the next batch would pass the salt's last valid column (`nonce +
regions > span`; for a batch that divides the span, every other core's, this is the old `nonce >= span`); at
mainnet n that leaves 256 of a salt's 16,384 column offsets unsearched.

Card 0, `pearl_sycl_check --bench 10`, xmx8w, stream on, 45-50 s of rest before each run, every run under
`tg30.sh`; all at 189.8-190.2 W with the PL2 flag set in 36-53% of samples (14% for AHOT):

| What | TH/s | MHz | J/PMAC | TH/s a GHz |
|---|---|---|---|---|
| def (goal-arc30's build), 8 runs | 104.48-104.66 | 1,975-1,982 | 1.814-1.816 | 52.80-52.95 |
| 256 columns at band 16 / band 64 | 104.42 / 104.19-104.23 | 1,977-1,978 | 1.818 / 1.821-1.822 | 52.70-52.78 |
| 384 columns at band 32 (a timing build), 2 runs | 104.79-104.82 | 1,974-1,976 | 1.811 | 53.05-53.07 |
| 320 columns at band 32, 2 runs | 105.06-105.06 | 1,976-1,978 | 1.807 | 53.11-53.18 |
| 384 columns at band 16 (timing), 2 runs | 105.31-105.32 | 1,978 | 1.803 | 53.23-53.25 |
| A' held in the L2 (`AHOT=32`, wrong results: the ceiling for A' from DRAM) | 105.91 | 1,996 | 1.796 | 53.06 |
| **new default (384, band 16)**, 2 runs | **105.23-105.29** | 1,978-1,979 | 1.803-1.805 | 53.20-53.21 |
| final race, the commit's own build against def, 3 rounds each, taking turns | **105.28-105.32** against 104.60-104.66 (+0.65%) | 1,976-1,984 | 1.803 against 1.814-1.815 | 53.23-53.27 against 52.72-52.89 |
| the same build at 320 / 352 / 448 columns (band 16) | 104.96 / 105.17 / 104.53 | 1,975-1,979 | 1.809 / 1.805 / 1.817 | |

448 columns (14 MB of B' with the band's 1 MB) pushes B' out of the L2 and is level with def; 384 is the best of the
widths tried. Checks on the commit's own build (card 0): verify.sh's fold checks (m = n = 2048, and the mainnet one
at 4 columns), the emu fold and its 400 hits on the auto fold PASS; one whole 384-column batch at the end of a mainnet
salt (columns 15,744-16,127, 3,145,728 regions) checked region by region PASS for dot and xmx8w (0 mismatched, 6,078
hits each); verify-hits 400 of 400 at duty 100 (col_batch 384, band 16, 7 salts, so across the new salt-end redraw)
and at duty 75 (col_batch 1024, band 32: unchanged). verify.sh's last step, 400 hits on the dot fold (which this
change does not touch), was stopped twice by the temperature guard (card 0's thermal flag); run alone at duty 60:
400 of 400 (col_batch 256, 7 salts). Earn suites that read the changed files: 15 suites, 587 tests pass.

**Operand fill, modelled (no run).** A and B are the miner's own int7 values, so a separate fill for each was a
candidate. The expected number of bits that flip between two operand bytes (fill plus noise in [-63, 63]) is
3.49-3.50 from fill 44 to 56 (0: 4.00, 32: 3.59, 63: 3.50), which matches the measured 48 = 63 and 32 +0.6%: 48 sits
at the minimum already, so no fill can buy much.

## Intel Arc Battlemage (Arc Pro B50)

### Arc Pro B50, 14 November: the Battlemage fold `xmx16f`, 3.8 times `xmx16` (`goal/b50a`)

The founder's Arc Pro B50 (BMG G21, 16 Xe2 cores, 16 GB, 70 W board power with a 73 W cap, up to 2,600 MHz, PCIe
3.0 x1) in VM 600 on his Proxmox host: Debian 13, kernel 7.2.11, xe driver, compute runtime 26.40.40196.8, IGC
2.42.6, Level Zero 1.34.0. The core is the build a release carries (open-source DPC++ 7.2.0, `./build.sh --jit`).
Before this the card mined with `xmx16` at about 11.2 TH/s (worker devb50). Rates are `pearl_sycl_check --bench`
at the mainnet m and n, 16 to 20 s a run, taking turns; clock is xe's `act_freq` sampled 20 times a second, power
the card's hwmon energy counter, temperature its package sensor. Every row sits at the 73 W cap.

**The ceiling.** `pearl_sycl_peak --sg 16`, `regs` (DPAS alone, operands in registers): 64.87 T-MAC/s at 1,979 MHz,
73.0 W, 74 C: 256 MACs an EU a clock, the full rate, and already at the cap. `xmx16` itself in the probe: 11.19.

**The fold** (`fold_xmx16f`; README "The Battlemage fold"): one sub-group a window (32 rows by 64 columns, 16
accumulators, 256-register mode), operands in block-read order (`kFragA16`, `kFragB16`), 12 load messages of 256
bytes a k-step of 16 DPAS where `xmx16` sends 40 small ones, a k-step's loads before its DPAS, the readout kept in
registers. Bit-exact on the first build: 16,384 of 16,384 regions, 29 of 29 hits at m = n = 2048, hardware and
reference code.

| Run | Fill | TH/s | Clock, power, temp |
|---|---|---|---|
| `xmx16` | hashed | 11.64 | 2,148 MHz, 73 W, 61 C |
| `xmx16` | constant (what mines) | 11.55 | 1,826 MHz, 71 W |
| `xmx16f`, band 256 (the context's band when `xmx16` is its own fold) | hashed | 33.3 | 1,446 MHz |
| `xmx16f`, band 8 / 16 / 32 / 64 / 128 / 256, 1 window a work-group | hashed | 36.7 / 37.9 / 38.4 / 39.1 / 37.9 / 32.7 | 73 W |
| `xmx16f`, band 64, 1 window a work-group, twins | hashed | 38.29 to 39.63 | 1,347 to 1,369 MHz, 73 W |
| the same | constant | 42.03, 42.99 | 1,465, 1,493 MHz, 72 W |
| `xmx16f`, band 64, 4 column windows a work-group (shipped) | constant | 43.72, 43.73, 44.66, 44.91 | 1,503 to 1,570 MHz, 72 W |

- **4 column windows a work-group: +3.9%** over 1 (two rounds, each 1.6 to 1.7 TH/s above its own 1-window run).
  2 by 2: 44.41, as good. Hashed fill, one round: 1 x 1 38.9, 2 x 1 39.4, 4 x 1 39.5, 8 x 1 39.4, 16 x 1 39.7,
  1 x 4 40.5, 2 x 2 40.5, 4 x 2 40.9, 4 x 4 41.4 / 41.6, 1 x 8 37.5, 1 x 16 37.3, 2 x 8 40.7 (R row by C column
  windows). 4 x 4 was best hashed but no better than 1 x 4 with the constant fill; 1 x 4 is the default.
- **Batch width**: 256 column offsets stay. 128: 43.75 / 43.40 against 44.91 / 43.73; 512: 40.22 / 40.21.
- **Checks** (shipped build, card 0): `verify.sh` PASS: the fold and `dot` against the host at m = n = 2048 and at the
  mainnet size (32,768 of 32,768, 46 of 46 hits), the reference code too; verify-hits 400 of 400 for `auto`
  (`xmx16f`, 44.24 TH/s through the addon, hit ratio 0.95, 7 salts) and for `dot` (1.37 TH/s).
- **Pool**: the v0.7.2 CLI binary with this core (`PEARL_CORE_PATH`), worker devb50 on prl-us.kryptex.network, 15 minutes: 43.3 TH/s by the CLI's own count (43.1 to 46.3 over its samples), 4 shares accepted, 0 rejected, the first hit checked and right; 1,499 MHz, 71.8 W, 78 C mean. Its log named the card "unknown architecture": v0.7.2's `archOf` knew only xmx16, xmx8 and dot (fixed in the same commit)

**What holds it, timed** (timing builds, `PEARL_F16_VAR`, wrong results on purpose; hashed fill, 1 window a
work-group, band 64; base 38.3 TH/s at 1,347 MHz):

| Build | TH/s | Clock |
|---|---|---|
| no loads: fragments made by arithmetic each k-step | 52.0 | 1,747 MHz |
| every window reads window 0's operands (all loads hit L1) | 49.1 | 1,548 MHz |
| only A' from window 0 | 46.1 | 1,562 MHz |
| only B' from window 0 | 41.8 (+6%, an earlier round) | |

Per clock the fold runs 87% of the DPAS peak (28.4k of 32.8k MACs a clock); the card spends its 73 W and drops the
clock. About a fifth of the watts bring operands from L2, mostly A', which each window reads alone (B' is shared
by the windows of a column that run on a core together).

**Tried and dropped** (all at the cap; the twin spread of a pair of base runs was 1 to 3%):

- **A' through local memory** (`xmx16s`: a work-group of 4 column windows copies each k-step's A' into a 3-slot
  ring under a split barrier, as the ring fold does B'): constant fill 42.99 / 42.75 against 42.99 / 42.03 for 1
  window a work-group and 44.66 / 43.72 for 4 column windows with no copy. Check PASS. Hashed: 1,404 MHz against
  1,354, but 28.0k MACs a clock against 28.4k: the barrier and the local-memory traffic eat what the L2 saves.
- **A' prefetched into L1** by the work-group's sub-groups, 2 or 4 k-steps ahead (`lsc_load.ugm.ca.ca` to null): 1 x 4
  38.8 / 38.0 against 41.1, 1 x 16 35.5 / 36.1.
- **Cache controls** on A' or B' (inline vISA, the same 256-byte transposed loads): A' `ca.ca` 38.7, `uc.ca` 37.1,
  `st.ca` 38.0, `ri.ca` 36.6; B' `st.ca` 35.3; A' `uc.ca` with B' `ca.ca` 36.1; base 39.6 / 38.3.
- **An A' fragment's 4 DPAS in a row** instead of a B' fragment's: 41.30 / 40.84 against 44.91 / 43.73 (-8%).
- **No finalizer options** (`PEARL_SYCL_VISA=none`): 44.58 / 43.92 against 44.91 / 43.73, level; kept as for Alchemist.

Next: 2D block loads (`SPV_INTEL_2d_block_io`: A' 32 rows in one message, B' transposed), a way to share A' among
more windows without the copy's barrier, and a card with more watts (B580 at 190 W, B60 at 200 W), where the
same fold should run well above the B50's clock.

### Arc Pro B50, 15 November: 2D block loads and a pipelined k-step, +6.6% (`goal/b50b`)

Same card, VM and build as above (open-source DPC++ 7.2.0 `--jit`, driver 26.40.40196.8). The miner
(llmjob-earn-intel) was stopped for each session and started again after it. Rates are `pearl_sycl_check
--bench` at the mainnet m and n, constant fill (what mines), 16 or 20 s a run, builds taking turns; clock is xe's
`act_freq`, power the hwmon energy counter, temperature the package sensor, each the mean of the run after its
first 3 s, read every second (the script stopped a run at 90 C; none came near). Every row sits at the 73 W cap.

**Result** (final A/A round, 20 s runs, one session):

| Build | TH/s | Clock | Power | Temp (pkg, max) |
|---|---|---|---|---|
| shipped `xmx16f` (1 x 4 windows a work-group) | 43.57, 43.72, 43.69 | 1,499 to 1,519 MHz | 71.8 to 72.0 W | 79 C (79) |
| new `xmx16f` (2D loads, pipelined, 4 x 4) | 46.60, 46.54, 46.52 | 1,625 to 1,635 MHz | 71.8 to 72.2 W | 79 C (79) |

+6.6%, twin spreads 0.3% (old) and 0.2% (new). All of it is clock at the same watts: per clock both do about
0.0286 TH/s a MHz. Over the whole round the old fold read 43.6 to 43.9 TH/s in 14 runs and the new one 46.4 to
46.8 in 9.

**What changed** (README "The Battlemage fold"):
- B' stored [k][64 columns] a window and read by one 2 KB transform (VNNI) load a k-step (`SPV_INTEL_2d_block_io`),
  A' stored in 32-row blocks with line j holding rows j and j + 16 and read by one 1 KB load: 2 messages a k-step
  where the old fold sent 12 transposed block reads of 256 bytes.
- The k-step pipelined: the next A' into a second set of fragments, the next B' into this k-step's B' registers
  after its last DPAS.
- Each chunk's transcript word in local memory (the sub-group's own, no barrier), freeing 16 registers.
- 4 x 4 windows a work-group.

**The steps, in order** (each a build, its fold check at m = n = 2048 PASS unless marked; base = the shipped fold
in the same session, 43.6 to 43.85 throughout):

| Build | TH/s | Clock | Notes |
|---|---|---|---|
| 2D loads for both, as drop-in loads (`PEARL_F16_2D=3`) | 41.3, 41.5 | 1,552, 1,560 MHz | -5.5%: same clock, each k-step waits for its loads |
| 2D B' only | 43.84 to 43.97 (3 runs) | 1,652 to 1,655 MHz | level: clock +7%, rate per clock -6% |
| 2D A' only, row-major (32 rows of 32 bytes: 32 half-used cache lines a k-step) | 40.7 | 1,445 MHz | -7% |
| 2D B' a k-step ahead in a second set | 37.8, 38.0 | 1,607 to 1,620 MHz | not unrolled by the compiler |
| pipelined (A' in 2 sets, B' reloaded 1 / 2 / 4 fragments at a time) | 17.8 / 22.8 / 29.3 | | spills 3.4 / 2.5 / 1.4 KB |
| transcript words in local memory, alone | 44.37, 44.54, 44.49 | 1,581 to 1,589 MHz | +1.7% (16 registers and 15 moves a chunk freed) |
| + pipelined, 2D B' 4 fragments a load (no spill) | 44.66, 44.92, 44.92, 44.57, 44.95 | 1,574 to 1,591 MHz | +2.6% |
| the same with block-read B' (4 / 2 / 1 fragments a reload) | 44.35 / 42.10 / 36.6 | | the 2D B' load is worth 1.3% |
| + both operands double-buffered | 14.1 | | spills 7.8 KB |
| pipelined 2D B', work-group 4 x 4 | 46.4 to 46.8 (6 runs) | 1,648 to 1,664 MHz | +6.6%; old fold at 4 x 4: 45.04, 45.04 |
| the same, 8 x 4 / 1 x 8 / 4 x 8 / 2 x 4 / 2 x 8 / 1 x 16 | 46.6 / 46.2 to 46.3 / 45.9 / 45.7 / 45.7 / 43.1 | | 4 x 4 kept |
| 4 x 4, band 16 / 32 / 128 / 256 (64 kept) | 46.0 / 46.5 / 45.2 / 40.7 | | 8 x 4 band 32: 47.0, band 128: 44.1 (one run each) |
| 4 x 4, 128 / 512 column offsets a batch (256 kept) | 45.8 / 43.9 | | |
| + B' prefetched 1 / 2 / 4 k-steps ahead (one 2D prefetch a k-step) | 46.0 / 46.2, 46.2 / 45.9 | | -1% |
| + A' from kFragA16 by a 2D load 64 bytes wide | 38.8, 39.1 | | WRONG (fold check 16,384 mismatched) and spills: lanes get other shorts |
| + A' in 32-row blocks, one 2D load a k-step (shipped) | 46.84, 46.78, 46.81, 46.64, 46.64 | 1,632 to 1,669 MHz | +0.4% over the A' block reads, 5 of 5 runs |

Temperatures through every session: pkg 73 to 80 C, VRAM at most 82 C (one run), fan as the card sets it.
Nothing reached 85 C.

**Checks** (the shipped build, card 0): `verify.sh` PASS: the fold and `dot` against the host at m = n = 2048
(16,384 of 16,384 regions, 29 of 29 hits; the reference code too) and at the mainnet size (32,768 of 32,768, 46 of
46); verify-hits 400 of 400 for `auto` (`xmx16f`, 45.2 TH/s through the addon in a 10 s run, hit ratio 0.95, 7
salts) and for `dot` (1.37 TH/s). pkg 72 to 80 C during it.

**Pool**: the v0.7.2 CLI binary with this core (`PEARL_CORE_PATH`), worker devb50 on prl-us.kryptex.network, 15
minutes: 46.2 TH/s by the CLI's own count (46.0 to 46.3 over its last samples), 7 shares accepted, 0 rejected;
1,647 MHz, 72.9 W, pkg 79.9 C mean (80 max), VRAM 78 C max, read every second. goal-b50a's fold: 43.3 TH/s, 4
accepted, at 1,499 MHz, 71.8 W, 78 C.

**The compile gate** (`compile-gate.sh` on the VM with ocloc 26.40.1 unpacked in a home folder): every target
builds; `fold_xmx16f<hw>` has 64 DPAS, 256 registers and no spill on bmg_g21, bmg_g31, lnl_m, ptl_h, ptl_u and pvc.
The prebuilt kernels for those targets use the 2D loads (`PEARL_SYCL_2D`, from the per-target
`__SYCL_TARGET_INTEL_GPU_*__` macros, with the extension passed to the translator for those targets only:
Alchemist's ocloc refuses a module that names it). With plain loads instead the prebuilt Battlemage kernel spilled
5.6 KB. Alchemist's prebuilt `xmx16f` spills, as it is never run (the gate checks it only on Battlemage).

**What I learned**
- On this card fewer, larger messages for the same bytes are cheaper in watts. The 2D B' load alone raised the
  clock 7% at the same 73 W. But a 32-register load cannot be moved over the DPAS that read those registers, so
  it only pays with the k-step pipelined by hand, and that only fits once the transcript words leave registers.
- A' loads must stay whole cache lines: a 2D load of row-major A' (32 bytes a row) touched 32 half lines a k-step
  and lost 7%; stored so each 64-byte line holds two rows, the same single load gains.
- 4 x 4 windows a work-group pays with either fold today: the old fold 45.04 against 43.7 (+3%; goal-b50a found
  it no better with the constant fill), the new one 46.7 against 45.7 at 1 x 4 (+2%). Of the +6.6%, about half is
  the work-group and half the loads, the pipeline and the freed registers.
- Prefetching B' a few k-steps ahead with a 2D prefetch cost 1%. Double-buffering B' as well as A' does not fit in
  256 registers.

- After the push, B' in a ring of 3 slots of 2 fragments (48 registers; the next k-step's first pair loads before
  this one's DPAS, its second pair into the slot the first pair just freed; 12 k-steps unrolled for the rotation):
  fold check PASS but 28.9, 28.8 TH/s against 46.7, 46.6, spilling 4.4 KB. With one A' set instead of two: 16.9,
  spilling 9.5 KB. The compiler does not keep the rotating slots in registers.

  With the 3 slots as separate arrays passed by name: 29.0, 28.9, the same 4.4 KB spill. Registers are the wall:
  128 for the accumulators, 32 for two A' sets and 48 for the ring leave too little for addresses and temporaries
  (even reloading B' a pair at a time spilled 0.7 KB).

Next: the same fold on a card with more watts (B580, B60), where the clock is not held by the cap; on the B50,
anything that saves watts per k-step without registers (fewer operand bytes per MAC needs a bigger window, which
the 256 registers do not hold).

## Research notes

What future speed work needs: how to measure, the limits found, and what has already been tried.

**Measuring**
- Race mode (`earn/native/probes/race.cu`, builds taking 4 s turns in one process) is the comparison to trust: a copy
  of the release reads within ±0.05–0.10%. Separate `hashrate.js` runs favour the build that runs on the warm card.
  It also runs on Windows, each build a DLL: on rig01's RTX 4090 a copy of the release read −0.015% ±0.030 over 25
  rounds.
- Compare builds on the same PC only. Salad PCs of one card differ by up to 24% on the same code, and some run far
  slower for every miner because something else is using the GPU.
- Our CLI's 5-minute mean carries about ±1.5–2.4% noise on fast cards, so 1–2 points between PCs mean nothing.
- Some Salad PCs dip on their own: on 3060 PC `4e084f5e` (23 October) v0.6.8's core and the release core both read
  49.3 TH/s and 6–31 by turns, at 93 W of 170, so a low reading can be the PC, not the core.
- v0.6.8's Windows build on rig01's RTX 4090 read 316.2–317.0 TH/s for 5 minutes, 0 rejected, level with v0.6.7's
  316.2 (23 October). It's the founder's rig, not a tested host, so not a row.
- On AMD, use 10-minute hit checks: 2-minute checks read low hits a salt for no known reason. Salad's `amd-smi`
  reports no power or clocks.
- Time a short fold on a Salad AMD PC before a race. On 7 November a 7900 XTX PC (`ec58ea79`) ran `pearl_amd_peak`'s
  plain WMMA loop at 161 MAC a CU a clock (241 on a good PC) and its 4-VALU loop at 5.6 TMAC/s, and didn't finish
  600 launches of the fold in 15 minutes.
- Cards mine at their power limit, so a saving shows up as clock, and cuts don't add: the six RDNA 4 cuts summed to
  about 8.9% alone and gave +4.3–5.4% together.
- A slow Radeon PC may be one that stops a busy compute queue. On an RX 7900 XTX PC (`ae456453`, 10 November) the
  fold test's launches, back to back on one queue, were stopped 30 times a second for 3.6–4.0 ms: 10% low at an
  unchanged clock and cycle count, and no launch length helped. A miner with a second queue at work (the hash's) wasn't
  stopped. The fold test's stamps show it: blocks whose waves changed SIMD mid-run, and gaps with no block starting or
  ending (AMD Radeon, "what a slow PC is"). Read them on a new AMD PC before trusting a fold-alone number from it.
- `PEARL_AMD_TRACE` is not free on such a PC: with it (and a clock probe every 5 s) the default read 51.0 TH/s on
  `ae456453` against 55.0 without. Time a race there with the trace off.
- SRBMiner 3.8.6 needs `wget` in the image (3.8.2 doesn't). Without it 3.8.6 exits at once, and without a terminal it
  prints nothing: that is what "would not start on Salad" was on 9 November.
- Time the whole process beside the rate when a host change adds a queue (`time` on the `hashrate.js` run; `ps -L`
  shows the threads). Each HIP stream with work on it keeps one more thread busy in the ROCm 7.2 image: the default
  run takes four cores, the hash queue a fifth (285 s of CPU against 228 s in 57 s, RX 7800 XT, 11 November), and
  the ticker that ended an RX 9060 XT PC's queue stops 0.4 to 0.9 of a core. Its runs that took no CPU gained
  nothing.

**Limits found**
- RDNA 4's WMMA does 1,009 MAC a CU a clock (98.6% of AMD's 1,024). The fold ran at 70% of that in v0.6.11 and 77% after
  KPAIR and BUFLD (4 November). Each VALU instruction beside the WMMAs costs about a cycle of WMMA time. After the six
  cuts the slab loop's only VALU is the readout's (about 88 a chunk); run beside the next slab's fragment loads and in
  three chains it gave +1.4–1.7% (#287 item 4, 7 November). Its count can't fall much further without the readout
  forms already dropped (below).
- RDNA 3's slab loads one slab ahead are nearly hidden already. The stamps of the buffer-load fold put the wait for
  them at the release fence at 0.76% and 0.79% of the slab loop on two RX 7900 XTX PCs and 1.85% on an RX 7800 XT
  (7 November), while waves wait 27-30% of it at the slab barrier for each other. Two slabs ahead
  (`PEARL_AMD_PF2PIN`) gained +0.05 to +0.42% in the fold.
- RDNA 3's WMMA does 238-241 MAC a CU a clock (93-94% of AMD's 256; `pearl_amd_peak`, RX 7900 XTX and 7800 XT,
  7 November). A VALU beside the WMMAs costs about 1 clock of WMMA time alone and 3.5-5 in a dependent chain, read
  from MAC a CU a clock, since the clock moves between kernels.
- A stock RTX 5090 loses about 8% of its clock to moving operands from L2. Mid RTX 50 cards hit their top clock first,
  so they could gain at most 2–3.3%. A byte copied from another SM's shared memory costs as much energy as one read
  from L2 (33.0 against 34.1 pJ, #287 item 7), so sharing operands between SMs by a copy saves nothing.
- On RTX 30/40 the tile hash costs about 1.3% of a tile, almost all of it the hash running past the partner's window
  (step 0): each cycle past about 2,300 after the hand-off costs a tile 0.9–1.0 cycle, and the hash releases at about
  3,150. Any edit to the hash's code makes ptxas reallocate the chunk loop, which then costs 3,383 cycles a tile.
- On RTX 30/40 the staging ring's waits cost 475 cycles a tile on a 4090 (0.46%), and all of it is that hash seam.
  With the hash stubbed out, removing every ring wait made the tile 164 cycles slower (101,381 → 101,545; #287 item
  1's builds). So lighter ring synchronization has nothing to win outside the seam (suggestion list item 4, 6 November). RTX 50's TMA
  ring hasn't been ablated, but a 5070 Ti's cycles a tile (102,989) are within 0.2% of the 4090's, and a 5090 with
  no TMA bytes at all did +0.15% ±0.3 more work a clock.
- The hash on RTX 50 was never stubbed out (the stubs are cp.async-ring only), but it can cost little more than on
  Ada: a 5070 Ti at a locked 1346 MHz runs the fold at 102,989 cycles a tile, 211 (0.2%) behind the 4090's 102,778.
  Its TMA producer (warp 4) folds only 96 IMMAs after the hand-off before it waits for the hashers, against 128 on
  Ada. Issuing just that refill at the stage's end is bounded by the 0.2%, and `PEARL_TALL_TMA_FILL_PT` 16, which
  includes it, read +0.07% on a 5090; not built. At the power limit a cycle saving comes back at about half on RTX 30
  (the paired hash: −0.30% cycles gave +0.14 to +0.22%, and −0.15% on a 3080 Ti), so the whole hash is worth under 1%
  there. (Suggestion list item 8, 6 November, from records.)
- Everything outside the fold costs about 0.2% of a card's time, so there is little to win there (suggestion list item
  10, 6 November, from records, no GPU run). On the 4090 a salt is 223 ms. Between salts the miner rewrites A' (0.34 ms, DRAM-bound: 0.15%)
  and two batch tails show (~0.07 ms each). Launch gaps are ~2 µs a batch. Hit proofs are read while the next batch
  runs. Job switches every 35–60 s cost nothing measurable. Halving the redraws (n 131072 to 262144) measured +0.2%.
  On an RX 9060 XT, `PEARL_AMD_TRACE` shows 0.13% outside the fold and hash kernels. A larger k doesn't dilute the
  redraw, because A' and a salt's work both grow with k; only n does. The one stage above 0.5% is the transcript hash
  (1.36% in the 4090's fold; 0.7–1.4% as AMD's separate kernel).
- The B200 needs a tcgen05 fold; `mma.sync` can't pass about 41% of the competitors there.
- The RTX 4090's memory clock can't be tested on any 4090 we reach (#287 item 11, 6 November). rig01's account isn't
  an Administrator (nvidia-smi -lmc, -lgc and -pl exit 4), and Vast and Salad containers can't set clocks. At stock
  the core ran 315.9 TH/s over 8 legs (0.30% spread) at 2519 MHz and 449 of 450 W, power capped in every sample, with
  memory at 10251 MHz and 2-3% busy. TH/s follows the SM clock (125.1-125.8 TH/s per GHz), and the memory clock is
  worth at most 40 W, so a lower one might give +0.5-1.5%. That was a guess then. The sweep ran on 10 November with
  admin rights: a 5001 MHz lock gives +0.8% at 450 W and up to +1.7% under a lower power limit, and 810 and 405 MHz
  lose (see "rig01, 10 November: memory clock, core clock and power limit" under RTX 40-series).
- RDNA 2 (`dot4`, a fold for RDNA 2) is a projection, not a measurement: 14.0–41.0 TH/s, half that if its `v_dot4`
  runs at half rate. No RDNA 2 card has run it, and none can be rented: Salad has no RDNA 2 class and Vast had 0
  offers for every RX 6600 to 6950, V620 and W6800 (#287 item 10, 6 November). The only rental found is Azure's
  NGads V620, $0.47 an hour spot for a whole card.
- AMD's separate transcript hash (#287 item 6, 6 November, `PEARL_AMD_TRACE`): 0.41–0.45 ms of a 30–31 ms batch on an
  RX 9060 XT at k 4096 (1.4%; 128 MiB read at 322 GB/s), 0.23 ms of 33.6 ms on an RX 9070 XT at k 8192 (0.7%).
  Hashing inside the fold (`PEARL_AMD_FUSE`) made the fold 0.33 ms slower on the 9060 XT and 0.36 ms on the 9070 XT.
  It went in on gfx1200 first: `hashrate.js`, interleaved on one PC against two copies of the release, read +0.34% on a
  9060 XT (54b217a9: +0.30 to +0.38% a round over 5 scored rounds, copies within 0.08%), +0.36 to +0.44% in the quiet
  rounds of another (b9b8ec1e), with verify-hits 400 of 400 and 7 Kryptex shares in 30 minutes, none rejected (refverify
  7 of 7). The shipped build (default flags, gfx1200 only) passed again on a third 9060 XT (9454033f): verify-hits 400
  of 400, 3 Kryptex shares in 10 minutes, none rejected. On 7 November, on top of #287 items 4 and 5, it lost 0.47% on a
  9060 XT and read level on a second, too noisy to settle (the combined race above), so it is off by default.

**Credited against shown rate, all four miners (item 10, 10 November, stand-in pool)**

Each miner mined 10 minutes on the stand-in pool (`earn/native/probes/verify-pool.js`, run by `bench.sh` with
`VPOOL=BITS`). The stand-in checks every share as the spec's verifier does and refuses duplicates. Credited is
accepted shares × 2^(256−BITS) MACs over the run, less its first minute. Shown is the mean of the miner's own figure
after the first minute (PeakMiner: its API). Every share was accepted: 0 rejected, 0 duplicates, all miners.

| PC | Target | Miner | Shown | Credited | Shares | Credited ÷ shown |
|---|---|---|---|---|---|---|
| RTX 4090, 450 W (`1d80ae90`) | 2^210 | ours v0.6.13 | 331.4 | 346.5 | 2,945 | 1.05 ±0.02 |
| | 2^210 | PeakMiner 2.20.7 | ~355 | 334.1 | 2,836 | 0.94 |
| | 2^210 | SRBMiner 3.8.2 | 350.4 | 349.1 | 2,926 | 1.00 ±0.02 |
| | 2^210 | OneZeroMiner 1.8.5 | 351.9 | 259.6 | 2,157 | 0.74 |
| RTX 5090, 477 W (`ad3451c9`) | 2^210 | ours v0.6.13 | 346.7 | 334.7 | 2,857 | 0.97 ±0.02 |
| | 2^210 | PeakMiner | 405.2 | 379.9 | 3,251 | 0.94 |
| | 2^210 | SRBMiner | 407.1 | 399.1 | 3,343 | 0.98 ±0.02 |
| | 2^210 | OneZeroMiner | 400.7 | 272.5 | 2,295 | 0.68 |
| RTX 5090, 575 W (`756d361b`) | 2^207 | OneZeroMiner | 424.3 | 436.2 | 461 | 1.03 ±0.05 |
| | 2^207 | PeakMiner | 424.7 | 415.2 | 446 | 0.98 ±0.05 |
| | 2^207 | SRBMiner | 420.5 | 445.6 | 462 | 1.06 ±0.05 |
| RTX 5060 Ti (`a668f722`, driver 610) | 2^210 | ours v0.6.14, CUDA 13 core | 98.4 | 100.6 | 842 | 1.02 ±0.035 |

- The competitors' figures are honest. At a target near the pools' (2^207 is 64 times easier than Kryptex's 2^203)
  all three are credited what they show, within the share count's noise. SRBMiner matched at 2^210 too.
- The easy target cost OneZeroMiner a quarter of its shares and PeakMiner about 6%. At 2^210 a share comes about
  5 times a second. OneZeroMiner's own accepted count equalled the stand-in's, so it sent everything it found but
  found less. PeakMiner reported a submit latency of 1.3 s at 2^210 against 84 ms at 2^207, so its shares queue.
  That is a test artifact, not a pool loss: on Kryptex a 4090 finds a share every 26 s. Don't run this test above
  2^207 for them.
- PeakMiner, SRBMiner and OneZeroMiner all mine k = 65536, the most the spec allows at rank 128 (k ≤ 4r²). We mine
  k = 2048. A share pays the same per MAC at any k, so this costs us nothing in credit. It can matter for speed: work
  done once per region (finalize and the transcript hash) is spread over 32 times as many MACs.
- 342–345 TH/s on a 450 W 4090 is real work. The ~340 pure-IMMA ceiling was measured on rig01 at 2715 MHz in
  September. These Salad 4090s ran at 2664–2754 MHz, so their ceiling is about the same or higher. SRBMiner was
  credited 349 at 2750 MHz.
- Our own figure is honest too: 1.05, 0.97 and 1.02 on three cards, mean 1.01.
- RTX 5060 Ti (item 17's 0.73 on Kryptex): on v0.6.14 with the CUDA 13 core, the hit check passed on both cores
  (400 of 400 hits verified; cu13 1,298 hits and cu12.8 1,262 in 120 s), and the stand-in credited 1.02 of the shown
  rate. On this PC the core finds every hit and the shares carry the full rate. So the Kryptex shortfall is not in
  the core or the shares. It is somewhere between the box and the pool, or particular to the fleet's PCs.

**Pool credit against our hashrate, by cause (item 17, 10 November, pool data and logs only)**

The fleet's credit ratios, 0.97–0.98 on Kryptex and 0.93 on HeroMiners, are the pools' pay and luck, not shares we
lose. The credited work matches the hashrate we show to about 1%. A share at either pool is 2^53 MACs (difficulty
2,097,152 × 2^32), so a card showing R TH/s should send R × 10^12 / 2^53 shares a second.

| Cause | Kryptex | HeroMiners | How measured |
|---|---|---|---|
| Pool pay (fee, PPS+ terms) | 0.956 of share value | fee 0 (PROP) | Kryptex: credited ÷ ledger 0.982 (to 4 Nov, before other workers joined the wallet) ÷ work 1.027 below |
| Pool luck | none (PPS+) | 0.945 ±0.086 | HeroMiners' blocks 13–20 Oct, our mining there: 134 blocks, Σ difficulty ÷ Σ round shares |
| Orphaned blocks | none | −0.7% | 1 of 135 in that window |
| Work credited ÷ work counted | 1.023 (ledger), 1.000 ±0.020 (shown rate) | ~1.00 (0.937 ÷ 0.945 ÷ 0.993) | 125,435 Kryptex shares of 613 Salad groups against the watcher's ledger; 2,497 accepted against 2,497 expected from the boxes' own minute lines (80 box-hours, 9–10 Nov) |
| Stale shares | 0.14% fleet, 0.06% dev4090, ~0.3% Salad | penalty from 2% stale | Kryptex worker list: 13 stale of 9,213 in 24 h; dev4090 10 of 16,130 in 7 days |
| Rejected or invalid | 0 | — | 0 invalid of 9,213 in 24 h; 5 rejected of 2,502 in the Salad lines, 4 on one RX 9060 XT PC |
| Vardiff and target rounding | 0 | 0.0015% | both pools hold 2,097,152; HeroMiners' target is 0x7fff8… (1 − 2^−16 of 2^203) |
| Reconnects | ~0.03% | — | 3 reopened connections in 6.1 online worker-hours (two were bench workers), ~6 s each |
| Hits dropped at a job switch | 0.1–0.9% by card (estimate) | more: jobs every 35 s | from the code, below |

- **dev4090**, the one long run: Kryptex credited 9,049 shares on 5, 8 and 9 November, 314.5 TH/s ±1.1%, against
  316–317 shown. On Salad the ledger reads 2–3% under what Kryptex credits in every class (missed minute lines and
  each box's first minutes), so it can't see a 1% loss; the boxes' minute lines can (above).
- **The job switch.** Jobs came 81 an hour on Kryptex (33 in 25 min; 24 of them at the same height) and 122 an hour
  on HeroMiners (every 35 s; 23 of 30 at the same height), probed from rig01 on 10 November. When a job arrives the
  core finishes the batch it is waiting on and the one queued behind it, and the miner drops their hits
  (`hit.jobId !== job.jobId` in `_onHit`). The shown rate counts that work. A batch is col_batch × 8192 regions ×
  2^19 MACs, so about 2 batches a switch is 0.13% on a 4090 (28 ms batches), 0.17–0.21% on a 4060 or 3060 (37, 46 ms),
  0.10% on a 5060 Ti, and 0.4–0.9% on RTX 20 (Turing's col_batch 2048: 84–195 ms) at Kryptex's rate; half again at
  HeroMiners'. The batch times are from the code, not timed; the drop is now logged (below) so a box can count it.
- **RTX 5060 Ti on 10 November: 0.73 ±0.07.** Three boxes on v0.6.14 (cu13 core, drivers 610, 617, 621, four PCs)
  showed 95–100 TH/s and Kryptex credited 149 shares against 204 expected in 5.3 hours, steady through each run
  (368: 38 in 90 min against 57). Kryptex's counts match the miners'. The watcher already judged one at the pool's
  53 TH/s. No NVIDIA code changed between v0.6.13 and v0.6.14, and earlier cu13 5060 Ti boxes read about 0.95
  (329: 0.946 over 341 shares). Every other class was within its noise (RTX 2060 0.985 ±0.043, RX 9070 XT 1.014 ±0.041,
  4070 Ti 1.051 ±0.045). Not explained yet.

Fixes, by expected gain:
1. Find the 5060 Ti shortfall: on a 5060 Ti box, verify-hits plus a low-difficulty share count through verifypool
   (item 10's tool) against the shown rate. If a fold misses hits, it is up to 27% on that class; nothing else here
   is above 1%.
   Done on one PC (item 10, above): hits and shares were whole there (1.02 credited ÷ shown), so look between the
   box and Kryptex next.
2. Send same-height hits from the replaced job. HeroMiners runs a cryptonote-nodejs-pool fork (v1.3.439), which usually keeps recent jobs and
   rejects only a changed height; Kryptex is untested. 73–77% of switches are same-height, so it gets back most of
   the switch loss: about 0.1% on RTX 30/40/50 and 0.3–0.7% on RTX 20. Test first on one Kryptex box: HeroMiners
   penalises 50% of pay at 2–5% stale, so a pool that turns them down must not get them.
3. Shorter batches on Turing (col_batch 2048 → 512) to cut its switch loss 4×, if a race shows no launch cost.
4. A Pearl pool watchdog like Monero's ("no word from the pool in 5 minutes; reconnecting": 131 of them on Monero in
   six days on Salad). No Pearl silence was seen, so this is insurance, under 0.1%.

Changed: the miner now logs "dropping a hit for the job replaced N ms ago (same block | new block)" instead of
dropping it silently, so the switch loss can be counted on a box. Nothing it sends changed.

**What made PeakMiner faster after 2.18 (item 2, 10 November, Salad)**

Each release on one PC, against the stand-in pool (`verify-pool.js serve`, a fixed target, a new job every 30 s)
through `sharetap.py`, so every share was checked and decoded. 2.5 minutes a release, the first 45 s not counted; the
3060 ran two passes (forward, then reverse). Script: `pm.sh` in the item's scratchpad (not committed). Cost about $0.13.

| RTX 3060, `50f45594`, 170 W | Shown TH/s | SM MHz | Memory controller | VRAM | Shares ÷ shown rate | Share config |
|---|---|---|---|---|---|---|
| 2.17.6, 2.18.3, 2.19.1 | 50.2–50.6 | 1800–1837 | 39–65% | 1,334 MiB | 0.72–0.87 | m 16384, n 65536, k 8192, tile 8×32 |
| 2.19.2, 2.20.0 | 53.7–53.8 | 1890–1905 | 0% | 354 MiB | 0.96–1.04 | m 8000, n 14336, k 65536, tile 10×16 |
| 2.20.1 | 54.0 | 1897–1905 | 0% | 354 MiB | 0.93–0.94 | as 2.19.2 |
| 2.20.2 | 54.3–54.5 | 1905–1912 | 0% | 396 MiB | (its API read 0 in half the samples) | m 5248, n 261632, k 32768, tile 10×16, rows contiguous |
| 2.20.7 | 54.3 | 1905–1920 | 0% | 356 MiB | 0.94–0.96 | m 7680, n 14336, k 65536, tile 24×8 |

- **The step is 2.19.2 (19 October, "higher hashrate and better efficiency on RTX 30xx, 40xx and 50xx"): +6.6% shown
  on the 3060.** The memory controller went from 39–65% busy to 0% and VRAM from 1.3 GB to 354 MB, and the card spent
  the power it saved on clock: +4.3% SM clock at the same 170 W. The other +2.3% is work a clock (27.7 → 28.4 TH/s per
  GHz, 99% of the 3060's dense int8 peak), most likely from k 65536 (8× longer folds, so the per-attempt hash,
  noise and readout cost 8× less). 2.17.6 → 2.19.1 changed nothing on Pearl (within 0.4%). 2.20.1–2.20.7 add about 0.9%.
- **Every release sends constant operands:** every A and B byte in every decoded share is 64 (fill 64), rank 128,
  and no share was rejected. With constant A and B, A' and B' are 64 plus the low-rank noise, so a kernel can make them
  on chip from the small noise factors instead of reading them from DRAM. That is what 2.19.2 does: no DRAM traffic at
  all while mining. SRBMiner 3.8.2 and OneZeroMiner 1.8.5 show the same 0% at about 1,900 MHz on the 9–10 November 3060s.
- **Ours looks like PeakMiner before 2.19.2.** On the 9–10 November RTX 30 boxes our miner showed 32–35% memory
  controller, 1,400 MiB and 1,815–1,822 MHz against their ~1,900 MHz. On RTX 40 and 50 our memory controller is only
  4–7% busy (4060, 5060), but the clock still trails by 2–3% (2,685 against 2,760 MHz; 2,512 against 2,572). This is
  #290 item 7 (operands built on the GPU); on RTX 30 it should be worth about the 4% clock, plus whatever the longer k
  gives a clock.
- **Shares.** At a job every 30 s (Kryptex sends one every 44 s, HeroMiners every 35 s) the old releases' shares came
  to only 0.72–0.87 of their shown rate, falling over the run, while 2.19.2 and later held 0.93–1.04 (each run 190–335
  shares, so ±6–7%). So the credited gain was likely bigger than the shown 6.6%; a job switch costs the DRAM version
  much more (a 16384 × 8192 A and 65536 × 8192 B to rebuild), but a longer job interval would be needed to prove it.
- **RTX 5060 Ti (`922b3948`, 150 W, driver 621), one pass:** 2.19.1 exits at once on this driver (2.19.2 fixed "a
  crash on RTX 50xx with recent NVIDIA drivers"). 2.19.2 96.1, 2.20.0 97.1, 2.20.2 97.6, 2.20.4 98.0, 2.20.5 98.3,
  2.20.6 98.5, 2.20.7 98.5 TH/s: +2.5% in small steps, all at 0% memory controller, 492–524 MiB, 2,677–2,700 MHz.
  Configs: m 16000, n 15360, k 65536, tile 10×16 (2.19.2–2.20.0); 2.20.2 as on the 3060; m 15360, n 15360,
  k 65536, tile 24×8 from 2.20.4. Shares 0.91–1.00 of shown.
- **The memory clock lock isn't it.** 2.19.0 added `--gpu-lmem 810` and 2.20.7 tuned for a 405 or 810 MHz lock, but
  only with the flag. On Salad, 2.20.7 with `--gpu-lmem 810` logged "memory clock lock 810 MHz failed (native
  memory-control function is outside the selected NVML library)" and ran at 54.1 TH/s, the same as without it; the
  memory stayed at 7,301 MHz in every run. The flag pays where it can be set (item 1), and only because the kernel
  no longer needs DRAM.
- 2.20.7 offers zstd at login (`"encoding": ["zstd", "gzip"]`) and sends gzip when the pool says only `"type":"v2"`.

**Picking RTX 30's launch shape on each PC at startup (item 5, 10 November, Salad)**

What the miner can choose between at run time today: the CUDA 12.8 or CUDA 13 core (`PEARL_CORE_VARIANT`; picked by
itself only for RTX 50), and, on RTX 30 alone, the launch shape. RTX 30 has one core (12.8). Its shape is chosen on the
host from the card's L2 size: the flip (B' held in the L2 slice, a 128-offset batch; the 3060 Ti and the 6 MB cards) or
the band walk (A' held, band 8 or 4, a 512-offset batch). The kernel reads the band at run time and the rest is host
state, so a core can switch shapes between batches with no new build. Every other card has one fixed launch.

Race mode, the rule's shape (flip 128) against the band walk, on four RTX 3090 Ti PCs and an RTX 3080 Ti
(`nvtest`, 600–900 s):

| PC | Power, driver | Band 8 x 512 | Band 4 x 512 | Memory busy, flip / walk |
|---|---|---|---|---|
| efd04f2b | 450 W, 621.14 | −0.58% (0 of 36 rounds ahead) | −1.30% (0/36) | 11% / 18% |
| daa97471 | 450 W, 591.59 | −3.23% (0/34) | −3.92% (0/34) | 12% / 52% |
| b659e537 | 480 W, 617.42 | +0.02% (15/29; A/A sd 2.8%) | −0.76% (4/29) | 11% / 19% |
| c1a87da3 | 480 W, 621.14 | **+0.39% (25/28)** | −0.06% (10/28) | 12% / 23% |
| 594fad00 (RTX 3080 Ti) | 350 W, 617.14 | −0.53% (1/27) | −1.40% (1/27) | 9% / 17% |

On efd04f2b, batch 1024 read −0.43% and the walk without its L2 slice −0.68%; on daa97471 −3.79% and −3.39%. So the
same card's best shape moves by PC, from flip ahead by 3.2% to the walk ahead by 0.4%. The rule's pick lost on one PC
of four, by 0.4%. That is the size of what a per-PC pick can win on a 3090 Ti; it doesn't explain the 85% vs 95% PCs
in the 9–10 November table.

Built (commits on item/5): with `shapeTrialSecs` the core races its shapes while it mines, in 4 s turns ended on a
whole salt with the first second uncounted, the order reversing each round, as race.cu does. It keeps another shape
only when that beats the rule's by 0.3% on average and in two thirds of the rounds. The app runs a 150 s trial the
first time a card opens and caches the pick per card for a week (`launch-shapes.json`); `PEARL_SHAPE_TRIAL=0` turns it
off and `PEARL_SHAPE` forces one. Hits found during the trial are sent as usual.

The trial against the race, same PC: c1a87da3 picked band 8 x 512 in both runs (+0.76% and +0.43%, ahead in 10 of 10
rounds each); b659e537 kept flip 128 (band 8 −0.15% and −0.33%); the 3080 Ti kept flip 128 and read band 8 −0.47% and
band 4 −1.37%, against the race's −0.53% and −1.40%. verify-hits with the trial switching shapes: 400 of
400 on each of three builds (hits from flip, band 8 and band 4). A first version counted each turn from the switch,
while the clock was still settling, and read daa97471's −3.23% as −0.25%; the warm-up fixed it. The trial costs about
a third of 150 s at the worse shapes' rate, once a week.

**RTX 30's band-walk cards: does the flip ever win, and does the trial pick right? (item 5b, 10 November, Salad)**

On the RTX 3060, 3070, 3070 Ti and 3080 10 GB the rule picks the band walk. Each PC raced the rule against the flip
forced on (`-DPEARL_AMPERE_FLIP_SHARE=1000`, the width the trial times) and both band depths (`-DPEARL_AMPERE_BAND`;
the one equal to the rule's is a twin), plus a copy of the rule's build as a second twin (`nvtest` race mode, 900 s,
4 s turns). Then hashrate.js twice and verify-hits.js with the 150 s trial on the rule's build.

| PC | Card, power | Rule | Flip | Other band | Twins | Trial kept (flip, other band) |
|---|---|---|---|---|---|---|
| be2418c9 | 3060, 170 W | band 4 x 512 | flip 32 −1.74% (0/42) | band 8 −0.62% (1/42) | −0.02, −0.03% | band 4 twice (−1.76/−1.81, −0.66/−0.54) |
| 60cdc4aa | 3070, 347 W | band 8 x 512 | flip 64 −0.58% (0/39) | band 4 −0.17% (2/39) | −0.04, −0.03% | band 8 twice (−0.60/−0.49, −0.27/−0.05) |
| bb028be9 | 3070 Ti, 288 W | band 8 x 512 | flip 64 +0.04% (21/42) | band 4 −0.08% (11/42) | −0.01, −0.02% | band 8 twice (−0.07/+0.15, −0.09/−0.02) |
| 95be0c9b | 3070 Ti, 230 W | band 8 x 512 | flip 64 +0.01% (16/40) | band 4 −0.11% (11/40) | +0.03, −0.01% | band 8 twice (−0.03/−0.18, −0.33/−0.37) |
| 161a910f | 3080 10 GB, 320 W | band 8 x 512 | flip 64 −0.14% (0/40) | band 4 −0.88% (0/40) | −0.02, −0.03% | band 8 twice (−0.14/−0.15, −0.87/−0.88) |
| db9aa4de | 3080 10 GB, 287 W | band 8 x 512 | flip 64 −0.65% (15/40) | band 4 −0.97% (14/40) | +0.04, +0.65% | band 8 twice (−0.46/−0.33, +0.17/−0.48) |

The flip never won: it lost clearly on the 3060, 3070 and one 3080, and tied the walk on both 3070 Tis (within the
twins' spread). db9aa4de was noisy (per-round sd 2.7–4.7%, a twin read +0.65%), so its row says little, except that
the trial's +0.3% and two-thirds rule kept it from chasing the noise. On every PC the trial kept the race's winner (or
the rule, where the race tied), and on the quiet PCs its figures came within about 0.1 point of the race's.
verify-hits: 400 of 400 on every build on every PC, including the rule's build with the trial switching shapes.

The rule stays as it is: on these cards it is right or tied everywhere, so the trial buys nothing on them beyond its
cost (about a third of 150 s at up to 1.8% less, once a week). In the forced-flip build, where the flip is the
rule, the verify runs' trial moved to the walk on the 3060 (race: walk 1.74% ahead) and on db9aa4de (0.65%, noisy),
and kept the flip on the 3070 (walk 0.58% ahead), the 3070 Tis (tied) and 161a910f (0.14%). The default build never
runs the flip on these cards, but it shows the 0.3% bar sits about where the smaller gaps are.

**NVIDIA work on AMD (PR #290 item 23, 10 November, desk review, then two races)**

Every NVIDIA change that gained speed, against what the AMD core (`earn/native/amd`) already does. Ranked by what
is left to gain per hour of work. AMD cards mine at their power limit, so only energy a MAC (clock) or VALU beside
the WMMAs (work a clock) can pay. Item 7 put RDNA 3's whole operand staging at +5.4-6.9% and its off-chip share at
+3.3% (both ablations, not reachable).

| NVIDIA change (gain there) | AMD today | Left for RDNA 2 / 3 / 4 |
|---|---|---|
| Hash out of the fold's way: in the fold, 1.5 regions a lane, the paired hash (+0.14-0.22% RTX 30), "keep the GPU fed" (+0.7%) | separate hash kernel, 0.7-1.4% of a batch; in the fold (`FUSE`) lost | **raced** as `PEARL_AMD_HASHQ` (above): +3.0-3.7% on one XTX PC and +0.2-1.0% on a second, level or behind on two 7800 XTs and a 9060 XT. On for gfx1100 only |
| Readout off the matrix path: shared-XOR readout, readout before the chunk-0 guard | RDNA 4: `RDLATE` + `RDTREE` (+1.4-1.7%) | RDNA 3: **raced** `RDTREE11`, −0.2 to −0.5%. `RDLATE` needs the wrap readout and KPAIR, which RDNA 3 doesn't run; item 13 put k 4096 under +0.3% |
| Newer compiler: ptxas 13 (+3.2% on the 5090, nothing on sm_86 or sm_75) | ROCm 7.2.0 pinned (Linux CI image), HIP SDK 7.2 on Windows | ROCm 7.2.4, 7.14 and 10.1 exist (`rocm/dev-ubuntu-24.04` tags; 7.14 and 10.x images are 8-9 GB). No code; the fold's asm pins and sched_barriers were tuned on clang 22, so it can lose too. Guess −1 to +2%, 2-3 h of box plumbing. **Next to try** |
| Per-PC launch-shape trial at startup (item 5: up to +0.4-0.8% on some RTX 3090 Ti PCs) | fixed per arch: batch 128 (RDNA 3) or 256, `WALK` on gfx12, k 4096/8192 by CU count | RDNA 4's `WALK` read +0.38 to +1.85% by PC and k 8192 pays only on big gfx1201, so a trial could pick per PC. Guess +0.2-0.5%, 2-4 h (host shapes for `pearl_core.cc`'s trial). After the compiler |
| Taller tile 192x256 (+3.4% on a 5090, +4.6% on a 4090 over 128x256) | 128x256, 8 waves of 64x64 | 192 rows: 12 waves split 896 staging units unevenly, and m 131072 isn't a multiple of 192; 256x256 drops to 4 waves a SIMD in one block a WGP. Cuts staged bytes 22-33% (worth at most about a third of item 7's +5.4-6.9%). A new fold, days. Low per hour |
| Operands built on chip, no DRAM (PeakMiner 2.19.2, +6.6% on a 3060) | stored A' and B' | Ceiling +3.3% on an XTX (item 7); NVIDIA's draws cost more than that (E4 −3.6 to −5.6%), and on RDNA every VALU costs WMMA time (1 clock RDNA 4, 1-5 RDNA 3). Not worth it |
| L2 residency: RTX 30 flip and band walk, persisting L2 slice, Ada and RTX 50 batch narrowing | `WALK` (RDNA 4), batch 128 (RDNA 3); RDNA has no L2-persistence control | done where it paid; `WALK` level on RDNA 3 |
| Fill 48 (+3.6% on a 4090), k-blocked operands (+0.7%), restamp instead of redraw, hit list read only on hits | all in the AMD core (fill above 48 tried; `TILED`; portable kernels) | done |
| Longer k (RTX 50/40 lost at k 4096) | RDNA 4 k 4096/8192 shipped; RDNA 3 k 4096 under +0.3% (item 13) | done |
| Ring roles and shared bases in registers (+0.36% sm_120) | `LDS32_PIN` on RDNA 4 | RDNA 3: pinned offsets at 48-byte rows lost a little on RDNA 4; skip |
| Memory clock locked low on Blackwell | can't set clocks on Salad (WSL, no `amd-smi` power) | an owner's own Linux rig only (root, `rocm-smi --setmclk`); untested |

RDNA 2 (the `dot4` fold, which no RDNA 2 card has run): of these only host-side ones reach it (`PEARL_AMD_HASHQ`, the
compiler, a shape trial). The first thing to learn there is the tester's own rate, and whether `v_dot4_i32_i8` runs at
full rate on RDNA 2 (the projection is 14-41 TH/s).

**What a fold with free operands is worth, and what building them on the SM costs (the gate, 10 November)**

Three reviews of the NVIDIA gap (10 November) said the "+2.8% to +4.05% ceiling" for a kernel that stores no operands
priced the wrong thing: `PEARL_ABLATE_DRAM` keeps every copy and only moves its source from DRAM into the L2. This is
the experiment they asked for before any such kernel is built. It prices the copies themselves, then puts stand-in work
where a draw of the operands would have to go. Every arm computes wrong results on purpose and the host refuses any
target but zero. They are on branch `goal/gate` (not on the PR) and none can ship.

The arms, all on the tall fold:

- **DRAM ablation** (`PEARL_ABLATE_DRAM`, as before): every copy kept, its source held in the L2.
- **Copies gated off** (`PEARL_TALL_AB_SKIP`, now also on the cp.async ring of sm_86 and sm_89): the copy instructions
  stay in the stream behind a predicate that holds only in a block's first tile. On sm_120 it is the TMA ring's existing
  switch (the boxes are not issued). A' alone and B' alone too.
- **Copies compiled out, no operand redraw** (`PEARL_GATE_NOCOPY`, `PEARL_GATE_PHANTOM`): the chunk loop has no copy
  instruction, the host stores two tiles of A' and B', and a new salt redraws the noise tables but no operand. This is
  the fold a kernel with free operands would start from. On sm_120 it runs on the cp.async ring (`-DPEARL_TALL_TMA=0`).
- **Long k**: the same at k 65536 on the RTX 4090 and 5090, and at k 16384 on the RTX 3060, 3090 and 5070 Ti, where a
  salt at k 65536 outlasts a race turn. A launch holds the MACs a k 2048 launch does.
- **Sham draw** (`PEARL_GATE_SHAM`): each thread rebuilds its seven 16-byte units of the buffer the stage refills, a
  stage ahead of their use: two 16-byte shared loads, ALU steps, one 16-byte store, then the buffer's FULL arrival.
  The chunk loop grows by 98, 196 and 378 instructions: 0.51, 1.02 and 1.97 an IMMA. Placed three ways: in runs at the
  two k-step ends, where B's fragment registers are dead (`PEARL_GATE_SHAM_AT=0`); between the mma, each step its own
  asm (`=2`, and `=4`, which asks for between m16 tiles and which ptxas schedules the same way); and ALU steps alone
  after every mma (`=3`).
- **Stages as a loop** (`PEARL_GATE_S_ROLL`): the fold has no register to spare (253 to 255 of 255), and the sham
  between the mma spills 64 to 156 bytes inside the chunk loop on sm_89 and sm_86 unless a chunk's two stages are a
  loop. Rolled, the arms build with no spill on sm_86, sm_89 and sm_120 (one aside, below).
- **A warp of its own**: a ninth warp can't be built (288 threads leave 227 registers; ptxas 12.6 falls back to 40 and
  spills 6.4 KB). So it is a second kernel beside the fold (`PEARL_GATE_SPIN`): one block of one warp an SM, in the
  768 registers a 253-register fold leaves, doing the sham's units at a stage's pace. It shares the SM with the fold
  and nothing else, so it prices the instructions, not a design: a real producer would have to write the fold's
  shared memory.

Checked in the machine code (nvdisasm on the sm_89 cubins, ptxas 12.6): every arm's chunk loop holds 192 IMMA and 40
ldmatrix (96 and 20 in a rolled stage), and the code from the tile hand-off to the tile loop's branch is the release's
2,150 instructions give or take 14 of address arithmetic (the IADD3 count, 305, and every shuffle are the same). The
release's chunk loop is 540 instructions; with the copies compiled out, 417. The sm_86 cubins (ptxas 12.6) and the
sm_120 ones (ptxas 13.3) built here hold the same IMMA and ldmatrix counts; sm_86's hand-off is 2,135 to 2,162
instructions against 2,153, and sm_120's keeps its 54 warp shuffles, 50 global stores and 2 atomics in every arm. The
Salad builds (nvcc 12.8 and 13.3) were checked for registers and spill only, by a line `gate.sh` prints (the first
5090 pass and the 3060 ran before it did): one arm spills, `si0r` on sm_120, 28 bytes.

Where: rig01's RTX 4090 with `hashrate.js` (90 s after 20 s warm-up, a control every four arms, two sessions), and
Salad PCs in race mode with a twin, where the three competitors and our v0.6.15 CLI were measured on the same PC
first, on the stand-in pool, by the mean of each miner's own samples from 60 s in.

**RTX 4090, rig01, 450 W limit, `hashrate.js`.** The control ran 10 times, 316.81 to 317.64 TH/s, a range of 0.26%. Each
arm's change is against the controls on either side of it. The three competitors read 338.6 to 341.9 TH/s on this card
at 450 W and 2625 MHz (item 16).

| Arm at 450 W | Runs | TH/s | vs control (each run) | SM clock | Clock | W | °C | TH/s per GHz | Per clock |
|---|---|---|---|---|---|---|---|---|---|
| `base` Release fold (control) | 10 | 317.3 | range 0.26% | 2531 MHz | | 449 | 77 | 125.4 | |
| `dram` DRAM ablation (copies kept, source in L2) | 2 | 317.9 | +0.17% (+0.21, +0.14) | 2538 | +0.3% | 449 | 78 | 125.2 | -0.1% |
| `sk3` Copies gated off, A' and B' | 2 | 329.0 | +3.67% (+3.72, +3.61) | 2685 | +6.1% | 406 | 75 | 122.5 | -2.3% |
| `sk3p` Copies gated off, no operand redraw | 1 | 329.4 | +3.78% (+3.78) | 2685 | +6.1% | 406 | 76 | 122.7 | -2.2% |
| `ncp` Copies compiled out, no operand redraw | 2 | 334.2 | +5.40% (+5.38, +5.41) | 2685 | +6.1% | 400 | 75 | 124.5 | -0.7% |
| `nck` No copies, long k | 2 | 341.9 | +7.83% (+7.80, +7.85) | 2685 | +6.1% | 410 | 76 | 127.3 | +1.6% |
| `ncpr` No copies, stages as a loop | 2 | 325.8 | +2.76% (+2.73, +2.78) | 2685 | +6.1% | 391 | 74 | 121.3 | -3.2% |
| `ncprk` No copies, stages as a loop, long k | 2 | 339.8 | +7.13% (+7.06, +7.20) | 2685 | +6.1% | 404 | 76 | 126.5 | +1.0% |
| `si05r` Sham 0.5 an IMMA, between the mma | 2 | 323.6 | +1.97% (+1.95, +1.99) | 2685 | +6.1% | 435 | 77 | 120.5 | -3.8% |
| `si1r` Sham 1 an IMMA, between the mma | 2 | 318.6 | +0.41% (+0.19, +0.64) | 2636 | +4.2% | 448 | 77 | 120.9 | -3.5% |
| `si2r` Sham 2 an IMMA, between the mma | 1 | 306.3 | -3.50% (-3.50) | 2606 | +3.0% | 449 | 77 | 117.5 | -6.2% |
| `si1rk` Sham 1 an IMMA, long k | 2 | 322.5 | +1.65% (+1.72, +1.58) | 2628 | +3.9% | 449 | 77 | 122.7 | -2.1% |
| `sh1` Sham 1 an IMMA, runs at the k-step ends | 1 | 290.1 | -8.60% (-8.60) | 2685 | +6.1% | 428 | 76 | 108.0 | -13.8% |
| `si05rk` Sham 0.5 an IMMA, long k | 1 | 330.2 | +4.08% (+4.08) | 2670 | +5.5% | 439 | 76 | 123.7 | -1.3% |
| `sa0r` The sham's stage-end release alone (no ALU) | 1 | 332.5 | +4.78% (+4.78) | 2685 | +6.1% | 400 | 75 | 123.8 | -1.3% |
| `sa1rb` ALU only, 1 an IMMA, no release | 1 | 310.6 | -2.12% (-2.12) | 2685 | +6.1% | 425 | 77 | 115.7 | -7.8% |
| `si1rb` Sham 1 an IMMA, no release | 1 | 316.8 | -0.15% (-0.15) | 2630 | +3.9% | 449 | 78 | 120.5 | -4.0% |
| `spn0` Spin kernel beside the no-copy fold, idle | 1 | 269.0 | -15.21% (-15.21) | 2685 | +6.1% | 345 | 69 | 100.2 | -20.1% |
| `spn1` Spin kernel, 1 an IMMA's worth | 1 | 220.9 | -30.38% (-30.38) | 2691 | +6.3% | 305 | 67 | 82.1 | -34.6% |

**RTX 4090, rig01, 300 W limit, `hashrate.js`.** The control ran 8 times, 270.58 to 272.23 TH/s, a range of 0.61%. Each
arm's change is against the controls on either side of it. No competitor was run at 300 W.

| Arm at 300 W | Runs | TH/s | vs control (each run) | SM clock | Clock | W | °C | TH/s per GHz | Per clock |
|---|---|---|---|---|---|---|---|---|---|
| `base` Release fold (control) | 8 | 271.6 | range 0.61% | 2164 MHz | | 300 | 66 | 125.5 | |
| `dram` DRAM ablation (copies kept, source in L2) | 2 | 274.3 | +0.99% (+0.97, +1.01) | 2189 | +1.2% | 299 | 65 | 125.3 | -0.2% |
| `sk3` Copies gated off, A' and B' | 2 | 301.2 | +10.90% (+10.85, +10.95) | 2440 | +12.7% | 300 | 67 | 123.5 | -1.6% |
| `ncp` Copies compiled out, no operand redraw | 2 | 308.5 | +13.60% (+13.64, +13.56) | 2464 | +13.9% | 300 | 67 | 125.2 | -0.3% |
| `nck` No copies, long k | 2 | 311.8 | +14.89% (+14.90, +14.88) | 2435 | +12.5% | 300 | 68 | 128.0 | +2.0% |
| `ncpr` No copies, stages as a loop | 2 | 302.9 | +11.63% (+11.67, +11.60) | 2480 | +14.6% | 300 | 68 | 122.1 | -2.7% |
| `ncprk` No copies, stages as a loop, long k | 2 | 310.8 | +14.54% (+14.64, +14.43) | 2442 | +12.9% | 300 | 67 | 127.3 | +1.4% |
| `si05r` Sham 0.5 an IMMA, between the mma | 2 | 289.1 | +6.59% (+6.69, +6.49) | 2382 | +10.1% | 299 | 67 | 121.4 | -3.3% |
| `si1r` Sham 1 an IMMA, between the mma | 2 | 279.7 | +3.00% (+3.23, +2.77) | 2313 | +6.9% | 300 | 68 | 120.9 | -3.6% |
| `si2r` Sham 2 an IMMA, between the mma | 2 | 266.2 | -2.00% (-1.82, -2.17) | 2266 | +4.7% | 300 | 67 | 117.5 | -6.4% |
| `si1rk` Sham 1 an IMMA, long k | 2 | 282.4 | +3.97% (+4.18, +3.75) | 2304 | +6.5% | 300 | 67 | 122.6 | -2.3% |

**RTX 5090, Salad `08f8e389`, 575 W limit, race mode.**

The competitors, then the first race (5 s turns).

| Miner (its own figure) | TH/s | SM clock | W | °C | TH/s per GHz | Credited by the stand-in |
|---|---|---|---|---|---|---|
| PeakMiner 2.20.7 | 426.3 | 2506 MHz | 513 | 75 | 170.1 | 387.61 |
| SRBMiner 3.8.2 | 420.9 | 2472 MHz | 503 | 75 | 170.3 | 428.04 |
| OneZeroMiner 1.8.5 | 423.7 | 2513 MHz | 510 | 75 | 168.6 | 285.75 |

| Arm (16 rounds) | TH/s | vs control | ±95% | Ahead | SM clock | Clock | W | °C | TH/s per GHz | Per clock |
|---|---|---|---|---|---|---|---|---|---|---|
| `base` Release fold (control) | 409.1 | | | | 2488 MHz | | 575 | 77 | 164.4 | |
| `aa` Twin: a copy of the control | 409.4 | +0.08% | 0.07 | 11/16 | 2489 | +0.0% | 575 | 77 | 164.5 | +0.1% |
| `tdram` DRAM ablation (boxes kept, source in L2) | 412.5 | +0.84% | 0.10 | 16/16 | 2502 | +0.5% | 575 | 77 | 164.9 | +0.3% |
| `tsk3` Boxes skipped, A' and B' | 451.4 | +10.33% | 0.12 | 16/16 | 2722 | +9.4% | 575 | 79 | 165.8 | +0.9% |
| `tsk1` Boxes skipped, A' alone | 426.8 | +4.33% | 0.15 | 16/16 | 2581 | +3.7% | 575 | 78 | 165.3 | +0.6% |
| `tsk2` Boxes skipped, B' alone | 431.4 | +5.46% | 0.15 | 16/16 | 2610 | +4.9% | 575 | 78 | 165.3 | +0.6% |
| `tsk3p` Boxes skipped, no operand redraw | 451.8 | +10.45% | 0.12 | 16/16 | 2722 | +9.4% | 575 | 79 | 166.0 | +1.0% |
| `cbase` Release fold on the cp.async ring | 404.5 | -1.12% | 0.11 | 0/16 | 2466 | -0.9% | 575 | 77 | 164.1 | -0.2% |
| `cncp` cp.async ring, copies compiled out | 445.0 | +8.78% | 0.12 | 16/16 | 2719 | +9.3% | 575 | 79 | 163.7 | -0.5% |
| `cncpr` cp.async ring, no copies, stages as a loop | 452.2 | +10.53% | 0.12 | 16/16 | 2712 | +9.0% | 575 | 79 | 166.7 | +1.4% |
| `csh1` Sham 1 an IMMA, runs at the k-step ends | 384.4 | -6.03% | 0.10 | 0/16 | 2665 | +7.1% | 575 | 78 | 144.3 | -12.3% |
| `csi05r` Sham 0.5 an IMMA, between the mma | 412.9 | +0.94% | 0.11 | 16/16 | 2626 | +5.5% | 566 | 78 | 157.2 | -4.4% |
| `csi1r` Sham 1 an IMMA, between the mma | 408.3 | -0.19% | 0.11 | 1/16 | 2565 | +3.1% | 558 | 78 | 159.2 | -3.2% |
| `csi2r` Sham 2 an IMMA, between the mma | 393.5 | -3.82% | 0.11 | 0/16 | 2560 | +2.9% | 576 | 78 | 153.7 | -6.5% |
| `csi1` Sham 1 an IMMA, between the mma, stages unrolled | 400.4 | -2.14% | 0.11 | 0/16 | 2606 | +4.7% | 575 | 78 | 153.6 | -6.5% |
| `csa1r` ALU only, 1 an IMMA | 419.2 | +2.47% | 0.11 | 16/16 | 2696 | +8.3% | 575 | 79 | 155.5 | -5.4% |
| `tdramk` DRAM ablation, long k | 417.8 | +2.13% | 0.13 | 16/16 | 2479 | -0.4% | 575 | 77 | 168.5 | +2.5% |
| `tsk3k` Boxes skipped, long k | 455.6 | +11.37% | 0.12 | 16/16 | 2690 | +8.1% | 575 | 79 | 169.4 | +3.0% |
| `cnck` cp.async ring, no copies, long k | 453.5 | +10.86% | 0.12 | 16/16 | 2700 | +8.5% | 575 | 79 | 168.0 | +2.2% |
| `cncprk` cp.async ring, no copies, loop, long k | 456.6 | +11.61% | 0.12 | 16/16 | 2699 | +8.4% | 575 | 79 | 169.2 | +2.9% |
| `csi1rk` Sham 1 an IMMA, long k | 408.5 | -0.14% | 0.11 | 4/16 | 2528 | +1.6% | 543 | 76 | 161.6 | -1.7% |
| `csi05rk` Sham 0.5 an IMMA, long k | 416.7 | +1.85% | 0.11 | 16/16 | 2544 | +2.2% | 525 | 75 | 163.8 | -0.4% |
| `csi2rk` Sham 2 an IMMA, long k | 395.6 | -3.29% | 0.11 | 0/16 | 2545 | +2.3% | 569 | 77 | 155.5 | -5.4% |

Our CLI and a second race on the same PC (10 s turns). Its spin arms are the first version of the spin kernel, whose
waiting blocks slept 250 ns at a time.

| Miner (its own figure) | TH/s | SM clock | W | °C | TH/s per GHz | Credited by the stand-in |
|---|---|---|---|---|---|---|
| Ours, v0.6.15 (CLI) | 409.4 | 2491 MHz | 575 | 78 | 164.4 | 390.72 |

| Arm (9 rounds) | TH/s | vs control | ±95% | Ahead | SM clock | Clock | W | °C | TH/s per GHz | Per clock |
|---|---|---|---|---|---|---|---|---|---|---|
| `base` Release fold (control) | 409.7 | | | | 2489 MHz | | 575 | 77 | 164.6 | |
| `aa` Twin: a copy of the control | 409.5 | -0.05% | 0.14 | 6/9 | 2488 | -0.0% | 575 | 77 | 164.6 | -0.0% |
| `cncp` cp.async ring, copies compiled out | 445.0 | +8.63% | 0.30 | 9/9 | 2717 | +9.2% | 575 | 79 | 163.8 | -0.5% |
| `cncpr` cp.async ring, no copies, stages as a loop | 452.1 | +10.36% | 0.32 | 9/9 | 2710 | +8.9% | 575 | 79 | 166.8 | +1.4% |
| `csi05r` Sham 0.5 an IMMA, between the mma | 413.0 | +0.81% | 0.31 | 8/9 | 2626 | +5.5% | 566 | 78 | 157.3 | -4.5% |
| `csi1r` Sham 1 an IMMA, between the mma | 408.5 | -0.29% | 0.31 | 4/9 | 2565 | +3.1% | 558 | 78 | 159.3 | -3.3% |
| `csb1r` Sham 1 an IMMA, asked for between m16 tiles | 408.4 | -0.32% | 0.31 | 4/9 | 2577 | +3.5% | 562 | 78 | 158.5 | -3.7% |
| `csi05rk` Sham 0.5 an IMMA, long k | 416.7 | +1.72% | 0.31 | 9/9 | 2544 | +2.2% | 526 | 76 | 163.8 | -0.5% |
| `sa0r` The sham's stage-end release alone (no ALU) | 449.3 | +9.68% | 0.33 | 9/9 | 2701 | +8.5% | 575 | 78 | 166.4 | +1.1% |
| `sa1rb` ALU only, 1 an IMMA, no release | 420.7 | +2.69% | 0.32 | 9/9 | 2692 | +8.2% | 575 | 79 | 156.3 | -5.1% |
| `si0r` Sham's loads, stores and release, no ALU | 428.3 | +4.55% | 0.32 | 9/9 | 2640 | +6.1% | 575 | 78 | 162.2 | -1.4% |
| `si1rb` Sham 1 an IMMA, no release | 396.2 | -3.28% | 0.30 | 0/9 | 2615 | +5.1% | 575 | 78 | 151.5 | -8.0% |
| `si05rb` Sham 0.5 an IMMA, no release | 416.9 | +1.75% | 0.33 | 9/9 | 2614 | +5.0% | 562 | 77 | 159.5 | -3.1% |
| `spn0` Spin kernel beside the no-copy fold, idle | 265.4 | -35.23% | 0.22 | 0/9 | 2858 | +14.8% | 449 | 72 | 92.9 | -43.6% |
| `spn05` Spin kernel, 0.5 an IMMA's worth | 323.9 | -20.94% | 1.31 | 0/9 | 2830 | +13.7% | 538 | 76 | 114.5 | -30.5% |
| `spn1` Spin kernel, 1 an IMMA's worth | 310.6 | -24.18% | 2.43 | 0/9 | 2830 | +13.7% | 520 | 76 | 109.8 | -33.3% |
| `spn2` Spin kernel, 2 an IMMA's worth | 413.9 | +1.03% | 0.29 | 9/9 | 2692 | +8.2% | 575 | 78 | 153.7 | -6.6% |

**RTX 3060, Salad `16548f76`, 170 W, 66 °C, race mode.**

The competitors, then the race. The second race never printed on this PC.

| Miner (its own figure) | TH/s | SM clock | W | °C | TH/s per GHz | Credited by the stand-in |
|---|---|---|---|---|---|---|
| PeakMiner 2.20.7 | 53.1 | 1878 MHz | 170 | 66 | 28.3 | 52.48 |
| SRBMiner 3.8.2 | 52.4 | 1874 MHz | 170 | 66 | 28.0 | 57.47 |
| OneZeroMiner 1.8.5 | 53.3 | 1879 MHz | 170 | 66 | 28.4 | 52.10 |

| Arm (13 rounds) | TH/s | vs control | ±95% | Ahead | SM clock | Clock | W | °C | TH/s per GHz | Per clock |
|---|---|---|---|---|---|---|---|---|---|---|
| `base` Release fold (control) | 49.18 | | | | 1796 MHz | | 170 | 67 | 27.38 | |
| `aa` Twin: a copy of the control | 49.18 | +0.02% | 0.03 | 7/13 | 1796 | +0.0% | 170 | 67 | 27.38 | +0.0% |
| `dram` DRAM ablation (copies kept, source in L2) | 50.68 | +3.06% | 0.03 | 13/13 | 1851 | +3.1% | 170 | 66 | 27.38 | -0.0% |
| `sk3` Copies gated off, A' and B' | 51.71 | +5.14% | 0.05 | 13/13 | 1912 | +6.5% | 168 | 66 | 27.04 | -1.2% |
| `sk3p` Copies gated off, no operand redraw | 51.77 | +5.27% | 0.05 | 13/13 | 1912 | +6.5% | 168 | 66 | 27.07 | -1.1% |
| `ncp` Copies compiled out, no operand redraw | 51.84 | +5.43% | 0.05 | 13/13 | 1912 | +6.5% | 165 | 66 | 27.11 | -1.0% |
| `ncpr` No copies, stages as a loop | 51.01 | +3.72% | 0.05 | 13/13 | 1912 | +6.5% | 162 | 66 | 26.68 | -2.6% |
| `sh1` Sham 1 an IMMA, runs at the k-step ends | 45.52 | -7.43% | 0.04 | 0/13 | 1908 | +6.2% | 170 | 66 | 23.86 | -12.9% |
| `si05r` Sham 0.5 an IMMA, between the mma | 50.11 | +1.91% | 0.05 | 13/13 | 1905 | +6.1% | 170 | 66 | 26.31 | -3.9% |
| `si1r` Sham 1 an IMMA, between the mma | 49.57 | +0.80% | 0.04 | 13/13 | 1882 | +4.8% | 170 | 66 | 26.33 | -3.8% |
| `si2r` Sham 2 an IMMA, between the mma | 47.62 | -3.17% | 0.05 | 0/13 | 1873 | +4.3% | 170 | 66 | 25.42 | -7.2% |
| `sa1r` ALU only, 1 an IMMA | 49.06 | -0.23% | 0.05 | 0/13 | 1912 | +6.5% | 169 | 66 | 25.66 | -6.3% |
| `dramk` DRAM ablation, long k | 51.42 | +4.56% | 0.05 | 13/13 | 1846 | +2.8% | 170 | 66 | 27.85 | +1.7% |
| `nck` No copies, long k | 52.54 | +6.84% | 0.05 | 13/13 | 1912 | +6.5% | 166 | 66 | 27.48 | +0.4% |
| `ncprk` No copies, stages as a loop, long k | 52.18 | +6.11% | 0.05 | 13/13 | 1912 | +6.5% | 165 | 66 | 27.29 | -0.3% |
| `si1rk` Sham 1 an IMMA, long k | 50.34 | +2.38% | 0.04 | 13/13 | 1880 | +4.7% | 170 | 66 | 26.78 | -2.2% |

**RTX 3090, Salad `71128c8f`, 390 W limit, 88 °C, race mode.**

The competitors and our CLI, then the first race. The driver named heat as what held the clock in 352 of 359 samples
over both races (`clocks_event_reasons` 0x20).

| Miner (its own figure) | TH/s | SM clock | W | °C | TH/s per GHz | Credited by the stand-in |
|---|---|---|---|---|---|---|
| PeakMiner 2.20.7 | 130.2 | 1573 MHz | 345 | 88 | 82.8 | 134.13 |
| SRBMiner 3.8.2 | 130.2 | 1578 MHz | 347 | 88 | 82.5 | 132.18 |
| OneZeroMiner 1.8.5 | 131.3 | 1585 MHz | 350 | 88 | 82.8 | 127.13 |
| Ours, v0.6.15 (CLI) | 122.8 | 1538 MHz | 358 | 88 | 79.9 | 125.19 |

| Arm (14 rounds) | TH/s | vs control | ±95% | Ahead | SM clock | Clock | W | °C | TH/s per GHz | Per clock |
|---|---|---|---|---|---|---|---|---|---|---|
| `base` Release fold (control) | 124.2 | | | | 1550 MHz | | 362 | 88 | 80.2 | |
| `aa` Twin: a copy of the control | 124.3 | +0.03% | 0.07 | 8/14 | 1550 | -0.0% | 362 | 88 | 80.2 | +0.0% |
| `dram` DRAM ablation (copies kept, source in L2) | 124.9 | +0.56% | 0.15 | 14/14 | 1559 | +0.6% | 354 | 88 | 80.1 | -0.1% |
| `sk3` Copies gated off, A' and B' | 128.4 | +3.35% | 0.14 | 14/14 | 1619 | +4.5% | 339 | 88 | 79.3 | -1.1% |
| `sk1` Copies gated off, A' alone | 125.7 | +1.15% | 0.11 | 14/14 | 1587 | +2.4% | 349 | 88 | 79.2 | -1.2% |
| `sk2` Copies gated off, B' alone | 125.6 | +1.08% | 0.12 | 14/14 | 1588 | +2.4% | 355 | 88 | 79.1 | -1.3% |
| `br` Release fold, stages as a loop | 116.5 | -6.20% | 0.17 | 0/14 | 1575 | +1.6% | 358 | 88 | 74.0 | -7.7% |
| `sk3p` Copies gated off, no operand redraw | 128.8 | +3.67% | 0.19 | 14/14 | 1627 | +5.0% | 340 | 88 | 79.2 | -1.3% |
| `ncp` Copies compiled out, no operand redraw | 129.5 | +4.24% | 0.19 | 14/14 | 1633 | +5.3% | 339 | 88 | 79.3 | -1.1% |
| `ncpr` No copies, stages as a loop | 128.0 | +3.00% | 0.13 | 14/14 | 1640 | +5.8% | 338 | 87 | 78.0 | -2.7% |
| `sh1` Sham 1 an IMMA, runs at the k-step ends | 112.8 | -9.20% | 0.11 | 0/14 | 1615 | +4.2% | 342 | 88 | 69.8 | -12.9% |
| `si05r` Sham 0.5 an IMMA, between the mma | 123.9 | -0.26% | 0.13 | 3/14 | 1606 | +3.6% | 343 | 88 | 77.2 | -3.8% |
| `si1r` Sham 1 an IMMA, between the mma | 122.6 | -1.33% | 0.15 | 0/14 | 1590 | +2.6% | 348 | 88 | 77.1 | -3.8% |
| `si2r` Sham 2 an IMMA, between the mma | 117.6 | -5.37% | 0.13 | 0/14 | 1583 | +2.1% | 349 | 88 | 74.3 | -7.3% |
| `sa1r` ALU only, 1 an IMMA | 121.6 | -2.10% | 0.19 | 0/14 | 1621 | +4.6% | 341 | 88 | 75.0 | -6.4% |
| `dramk` DRAM ablation, long k | 127.0 | +2.19% | 0.16 | 14/14 | 1557 | +0.5% | 356 | 88 | 81.5 | +1.7% |
| `nck` No copies, long k | 131.1 | +5.49% | 0.18 | 14/14 | 1631 | +5.2% | 340 | 88 | 80.4 | +0.2% |
| `ncprk` No copies, stages as a loop, long k | 130.5 | +5.06% | 0.16 | 14/14 | 1634 | +5.4% | 340 | 88 | 79.9 | -0.3% |
| `si1rk` Sham 1 an IMMA, long k | 124.2 | -0.01% | 0.15 | 6/14 | 1585 | +2.3% | 349 | 88 | 78.4 | -2.3% |
| `si05rk` Sham 0.5 an IMMA, long k | 126.2 | +1.60% | 0.25 | 14/14 | 1606 | +3.7% | 345 | 88 | 78.6 | -2.0% |
| `si2rk` Sham 2 an IMMA, long k | 118.9 | -4.32% | 0.17 | 0/14 | 1572 | +1.5% | 349 | 88 | 75.6 | -5.7% |

The second race: the sham's parts and the spin kernel (its waiting blocks spin on a barrier test).

| Arm (15 rounds) | TH/s | vs control | ±95% | Ahead | SM clock | Clock | W | °C | TH/s per GHz | Per clock |
|---|---|---|---|---|---|---|---|---|---|---|
| `base` Release fold (control) | 123.9 | | | | 1564 MHz | | 365 | 88 | 79.2 | |
| `aa` Twin: a copy of the control | 124.0 | +0.10% | 0.27 | 9/15 | 1561 | -0.2% | 365 | 88 | 79.5 | +0.3% |
| `ncp` Copies compiled out, no operand redraw | 129.1 | +4.16% | 0.29 | 15/15 | 1641 | +4.9% | 342 | 87 | 78.7 | -0.7% |
| `ncpr` No copies, stages as a loop | 127.5 | +2.90% | 0.29 | 15/15 | 1653 | +5.7% | 341 | 87 | 77.1 | -2.6% |
| `si1r` Sham 1 an IMMA, between the mma | 122.3 | -1.34% | 0.44 | 2/15 | 1604 | +2.5% | 350 | 88 | 76.2 | -3.8% |
| `sb1r` Sham 1 an IMMA, asked for between m16 tiles | 121.7 | -1.82% | 0.36 | 0/15 | 1604 | +2.5% | 350 | 88 | 75.9 | -4.2% |
| `sa0r` The sham's stage-end release alone (no ALU) | 129.6 | +4.57% | 0.58 | 15/15 | 1642 | +5.0% | 343 | 87 | 78.9 | -0.4% |
| `sa1rb` ALU only, 1 an IMMA, no release | 120.9 | -2.46% | 0.61 | 1/15 | 1635 | +4.5% | 344 | 87 | 73.9 | -6.7% |
| `si0r` Sham's loads, stores and release, no ALU | 126.3 | +1.89% | 0.69 | 13/15 | 1621 | +3.6% | 347 | 88 | 77.9 | -1.7% |
| `si1rb` Sham 1 an IMMA, no release | 122.5 | -1.11% | 0.64 | 2/15 | 1608 | +2.8% | 351 | 88 | 76.2 | -3.8% |
| `si05rb` Sham 0.5 an IMMA, no release | 125.7 | +1.45% | 0.72 | 14/15 | 1624 | +3.8% | 349 | 88 | 77.4 | -2.3% |
| `spn0` Spin kernel beside the no-copy fold, idle | 120.5 | -2.74% | 0.63 | 1/15 | 1671 | +6.8% | 342 | 87 | 72.1 | -9.0% |
| `spn05` Spin kernel, 0.5 an IMMA's worth | 121.1 | -2.25% | 0.53 | 1/15 | 1664 | +6.3% | 344 | 87 | 72.8 | -8.1% |
| `spn1` Spin kernel, 1 an IMMA's worth | 120.9 | -2.46% | 0.62 | 1/15 | 1660 | +6.1% | 346 | 87 | 72.8 | -8.1% |
| `spn2` Spin kernel, 2 an IMMA's worth | 121.0 | -2.36% | 0.57 | 1/15 | 1662 | +6.2% | 346 | 87 | 72.8 | -8.1% |

**RTX 5070 Ti, Salad `d0a033a5`, 300 W limit, race mode.**

The competitors and our CLI, then the first race. The card ran 2805 MHz in nearly every arm, at 165 to 220 W of its 300
W limit and 58 to 61 °C, so nothing but its top clock held it.

| Miner (its own figure) | TH/s | SM clock | W | °C | TH/s per GHz | Credited by the stand-in |
|---|---|---|---|---|---|---|
| PeakMiner 2.20.7 | 198.1 | 2805 MHz | 184 | 60 | 70.6 | 177.67 |
| SRBMiner 3.8.2 | 199.3 | 2804 MHz | 186 | 60 | 71.1 | 202.16 |
| OneZeroMiner 1.8.5 | 197.3 | 2805 MHz | 183 | 60 | 70.3 | 168.34 |
| Ours, v0.6.15 (CLI) | 189.2 | 2812 MHz | 220 | 62 | 67.3 | 182.18 |

| Arm (19 rounds) | TH/s | vs control | ±95% | Ahead | SM clock | Clock | W | °C | TH/s per GHz | Per clock |
|---|---|---|---|---|---|---|---|---|---|---|
| `base` Release fold (control) | 191.0 | | | | 2807 MHz | | 218 | 61 | 68.0 | |
| `aa` Twin: a copy of the control | 188.8 | -1.18% | 2.01 | 2/19 | 2807 | +0.0% | 215 | 60 | 67.3 | -1.2% |
| `tdram` DRAM ablation (boxes kept, source in L2) | 191.2 | +0.07% | 0.22 | 10/19 | 2807 | +0.0% | 211 | 60 | 68.1 | +0.1% |
| `tsk3` Boxes skipped, A' and B' | 191.6 | +0.31% | 0.27 | 15/19 | 2805 | -0.1% | 165 | 58 | 68.3 | +0.4% |
| `tsk3p` Boxes skipped, no operand redraw | 191.8 | +0.43% | 0.32 | 19/19 | 2805 | -0.1% | 165 | 58 | 68.4 | +0.5% |
| `cbase` Release fold on the cp.async ring | 191.2 | +0.12% | 0.32 | 3/19 | 2807 | +0.0% | 220 | 60 | 68.1 | +0.1% |
| `cncp` cp.async ring, copies compiled out | 189.2 | -0.96% | 0.35 | 2/19 | 2805 | -0.1% | 166 | 58 | 67.4 | -0.9% |
| `cncpr` cp.async ring, no copies, stages as a loop | 192.6 | +0.86% | 0.25 | 19/19 | 2805 | -0.1% | 170 | 58 | 68.7 | +0.9% |
| `csh1` Sham 1 an IMMA, runs at the k-step ends | 167.3 | -12.42% | 0.33 | 0/19 | 2805 | -0.1% | 174 | 59 | 59.6 | -12.4% |
| `csi05r` Sham 0.5 an IMMA, between the mma | 181.7 | -4.87% | 0.36 | 0/19 | 2805 | -0.1% | 178 | 59 | 64.8 | -4.8% |
| `csi1r` Sham 1 an IMMA, between the mma | 182.7 | -4.36% | 0.36 | 0/19 | 2785 | -0.8% | 187 | 59 | 65.6 | -3.6% |
| `csi2r` Sham 2 an IMMA, between the mma | 175.8 | -7.97% | 0.35 | 0/19 | 2776 | -1.1% | 192 | 60 | 63.3 | -6.9% |
| `csi1` Sham 1 an IMMA, between the mma, stages unrolled | 177.7 | -6.97% | 0.35 | 0/19 | 2805 | -0.1% | 185 | 59 | 63.3 | -6.9% |
| `csa1r` ALU only, 1 an IMMA | 179.7 | -5.93% | 0.36 | 0/19 | 2805 | -0.1% | 171 | 58 | 64.1 | -5.9% |
| `tdramk` DRAM ablation, long k | 194.5 | +1.86% | 0.40 | 19/19 | 2803 | -0.1% | 218 | 60 | 69.4 | +2.0% |
| `tsk3k` Boxes skipped, long k | 195.3 | +2.26% | 0.39 | 19/19 | 2805 | -0.1% | 171 | 58 | 69.6 | +2.3% |
| `cnck` cp.async ring, no copies, long k | 187.7 | -1.72% | 0.37 | 2/19 | 2805 | -0.1% | 163 | 57 | 66.9 | -1.7% |
| `cncprk` cp.async ring, no copies, loop, long k | 195.1 | +2.15% | 0.39 | 19/19 | 2805 | -0.1% | 166 | 57 | 69.6 | +2.2% |
| `csi1rk` Sham 1 an IMMA, long k | 183.7 | -3.81% | 0.39 | 0/19 | 2777 | -1.1% | 182 | 58 | 66.2 | -2.8% |

The second race: the sham's parts and the spin kernel (its waiting blocks wait on a barrier, which suspends them on
sm_120).

| Arm (16 rounds) | TH/s | vs control | ±95% | Ahead | SM clock | Clock | W | °C | TH/s per GHz | Per clock |
|---|---|---|---|---|---|---|---|---|---|---|
| `base` Release fold (control) | 191.4 | | | | 2806 MHz | | 217 | 60 | 68.2 | |
| `aa` Twin: a copy of the control | 191.5 | +0.02% | 0.06 | 8/16 | 2807 | +0.0% | 216 | 60 | 68.2 | -0.0% |
| `cncp` cp.async ring, copies compiled out | 189.5 | -1.03% | 0.06 | 0/16 | 2805 | -0.0% | 165 | 58 | 67.5 | -1.0% |
| `cncpr` cp.async ring, no copies, stages as a loop | 192.9 | +0.77% | 0.06 | 16/16 | 2805 | -0.0% | 166 | 58 | 68.8 | +0.8% |
| `csi1r` Sham 1 an IMMA, between the mma | 182.8 | -4.53% | 0.26 | 0/16 | 2781 | -0.9% | 182 | 60 | 65.7 | -3.7% |
| `csb1r` Sham 1 an IMMA, asked for between m16 tiles | 182.8 | -4.50% | 1.04 | 1/16 | 2782 | -0.9% | 182 | 59 | 65.7 | -3.7% |
| `sa0r` The sham's stage-end release alone (no ALU) | 193.4 | +1.04% | 1.08 | 16/16 | 2805 | -0.0% | 166 | 58 | 69.0 | +1.1% |
| `sa1rb` ALU only, 1 an IMMA, no release | 181.7 | -5.11% | 1.02 | 1/16 | 2805 | -0.0% | 167 | 58 | 64.8 | -5.1% |
| `si0r` Sham's loads, stores and release, no ALU | 189.3 | -1.12% | 1.06 | 1/16 | 2805 | -0.0% | 174 | 58 | 67.5 | -1.1% |
| `si1rb` Sham 1 an IMMA, no release | 176.4 | -7.85% | 0.99 | 0/16 | 2805 | -0.0% | 178 | 58 | 62.9 | -7.8% |
| `si05rb` Sham 0.5 an IMMA, no release | 184.5 | -3.60% | 0.17 | 0/16 | 2805 | -0.0% | 174 | 58 | 65.8 | -3.6% |
| `spn0` Spin kernel beside the no-copy fold, idle | 164.5 | -14.08% | 1.00 | 0/16 | 2805 | -0.0% | 147 | 56 | 58.6 | -14.0% |
| `spn05` Spin kernel, 0.5 an IMMA's worth | 164.4 | -14.12% | 0.72 | 0/16 | 2805 | -0.0% | 153 | 56 | 58.6 | -14.1% |
| `spn1` Spin kernel, 1 an IMMA's worth | 164.0 | -14.33% | 0.24 | 0/16 | 2805 | -0.0% | 155 | 56 | 58.5 | -14.3% |
| `spn2` Spin kernel, 2 an IMMA's worth | 162.2 | -15.30% | 0.45 | 0/16 | 2802 | -0.1% | 153 | 56 | 57.9 | -15.2% |

What the tables say:

- **Free operands are the gap, and the old ceiling missed it.** "Free operands" is the no-copy arm (on RTX 50 the TMA
  ring with its boxes skipped); "long k" is k 65536 or 16384 as above.

  | PC | DRAM ablation | Free operands | Free operands, long k | Best competitor on that PC |
  |---|---|---|---|---|
  | RTX 4090 rig01, 450 W | +0.2% | +5.4% | +7.8% (341.9 TH/s) | 341.9 (item 16, the same card) |
  | RTX 4090 rig01, 300 W | +1.0% | +13.6% | +14.9% (311.8) | not run at 300 W |
  | RTX 5090 `08f8e389`, 575 W limit | +0.8% | +10.3% | +11.4% (455.6) | 426.3; our CLI 409.4 |
  | RTX 3060 `16548f76`, 170 W, 66 °C | +3.1% | +5.4% | +6.8% (52.54) | 53.30 |
  | RTX 3090 `71128c8f`, 390 W limit, 88 °C | +0.6% | +4.2% | +5.5% (131.1) | 131.3; our CLI 122.8 |
  | RTX 5070 Ti `d0a033a5`, 300 W limit | 0.0% | +0.1% | +2.0% (195.3) | 199.3; our CLI 189.2 |

  - The gain is clock wherever power or heat holds the clock. The 4090 goes from 2531 MHz to its 2685 MHz top with
    50 W unused at 450 W, and from 2164 to 2464 MHz at 300 W; the 5090 from 2488 to 2722 MHz; the 3060 from 1796 MHz
    to its 1912 MHz top; the 3090, which heat held all race (88 °C, 340 to 362 W of its 390 W limit), from 1550 to
    1633 MHz.
  - It is nothing where the clock is already at its top: the 5070 Ti ran 2805 MHz in every arm and only drew 54 W
    less (165 against 219 W).
  - Work a clock moves −1.1% to +0.9% at k 2048 and +0.2% to +3.0% at long k. Long k is the only part of the gain
    a clock-bound card sees.
  - One operand alone is worth half or less: on the 5090 A' alone +4.3% and B' alone +5.5%; on the 3090 +1.2% and
    +1.1% against +3.4% for both. Not redrawing A' on a new salt is worth 0.1 point.
- **A draw made of instructions in the MMA warps gives most of it back.** Net against the release fold with the sham
  between the mma, the cheapest placement found, at 0.5 / 1 / 2 extra instructions an IMMA:

  | PC | k 2048 | Long k |
  |---|---|---|
  | RTX 4090, 450 W | +2.0% / +0.4% / −3.5% | +4.1% / +1.6% / not run |
  | RTX 4090, 300 W | +6.6% / +3.0% / −2.0% | not run / +4.0% / not run |
  | RTX 5090, 575 W limit | +0.9% / −0.2% / −3.8% | +1.9% / −0.1% / −3.3% |
  | RTX 3060 | +1.9% / +0.8% / −3.2% | not run / +2.4% / not run |
  | RTX 3090 | −0.3% / −1.3% / −5.4% | +1.6% / 0.0% / −4.3% |
  | RTX 5070 Ti | −5.1% / −4.6% / −8.2% | not run / −4.1% / not run |

  - It costs work a clock. Against the best no-copy arm at the same k, an arm with the sham does 2.2 to 4.2% less a
    clock on RTX 30 and 40 at 0.5 and at 1 an IMMA alike, and 5.6 to 6.3% less at 2. On RTX 50: 3.2 to 5.7%, and 8%.
  - It costs watts. At 2685 MHz the 4090 draws 391 to 400 W with no copies, 435 W with the sham at 0.5, and at 1 it
    is back on its 450 W limit at 2636 MHz. So one instruction an IMMA costs about what the copies did. That is why
    the sham pays under a deep cap only at a low count, and why a clock-bound card loses the whole per-clock cost.
  - Placement decides the size. In runs at the k-step ends (a load, its use right behind it, no mma to issue
    meanwhile), one instruction an IMMA reads −8.6% (4090), −6.0% (5090), −7.4% (3060), −9.2% (3090) and −12.7%
    (5070 Ti). ALU steps alone, one after each mma, cost more a clock (5 to 7%) than the whole sham spread between
    them. ptxas spreads the steps between the mma whatever order the source asks for, so "between the mma" and
    "between m16 tiles" are one placement (5090: −0.29% and −0.32%).
  - The fold has no registers for it. Unrolled, the sham between the mma spills; with a chunk's two stages as a
    loop it fits. That loop alone moves the no-copy fold by −2.6% (4090), −1.7% (3060), −1.2% (3090) and +1.7%
    (5090), and the release fold by −6.2% on the 3090. An arm that differs from the looped no-copy fold only in
    where the stage's barrier is released reads 2.0% above it on the 4090 and 0.7% below it on the 5090. So arms
    here differ by two points for reasons that are not the draw, and a difference under two points is not a finding.
- **A warp of its own is no way out as built.** A ninth warp does not fit the fold (above), nor with B fragments
  loaded pair by pair (`PEARL_GATE_BSTREAM`: 168 registers and 768 bytes of spill). As a second kernel beside the
  fold (one block an SM, in the registers the fold leaves) it could not be priced, because its control fails. Blocks
  that only wait already take 7 points off the no-copy fold on the 3090, 13 on the 5070 Ti (where a waiting block
  is suspended), 21 on the 4090 and 44 on the 5090 (the first version, which slept). Giving them a draw's work adds
  at most a point to that on the 3090 and the 5070 Ti at 0.5, 1 or 2 instructions an IMMA's worth. On the 4090 at 1
  the fold is 36 points down and unsteady, in one run. Working blocks kept up about a quarter of a draw's pace on
  the 4090 and the 3090. So it is the blocks being there that costs. Most likely they do not sit one beside each
  fold block but crowd the SMs that are free at launch and keep the fold off them; that was not checked. The arm
  says nothing for or against a producer warp inside the fold's own block, which is what it stood in for.
- **Against the reviews' kill lines.**
  - RTX 40 (kill if copies skipped is under +4% on the 4090, or skipped plus sham under +2%). Copies skipped: the
    ported switch reads +3.7% at 450 W, 0.3 point under the line; with the copies compiled out, +5.4%; at 300 W
    +10.9% and +13.6%. Skipped plus sham at 450 W: +2.0% at 0.5 an IMMA, +0.4% at 1, −3.5% at 2 (+4.1% and +1.6% at
    k 65536). At 300 W: +6.6%, +3.0%, −2.0%. So it fails at 450 W unless the draw fits in half an instruction an
    IMMA, and passes at 300 W up to 1.
  - RTX 50 (kill under +4% at 1 an IMMA on a 5090 with a limit of 480 W or less). No such PC came up; on the one
    with a 575 W limit, which our fold sits on, it reads −0.2% (−0.1% at k 65536) and +0.9% at 0.5. It fails
    there. The 4090's cap from 450 to 300 W turned +0.4% into +3.0%, still under 4.
  - RTX 30 (kill under +6% for copies skipped at long k on a normal 3060). +6.8% at k 16384 on a 66 °C PC: not
    killed, and under the 7% the same review asked for before building. With the sham at 1 an IMMA it is +2.4%
    there and nothing on the 3090.
- **What the competitors' figures look like beside ours.** A clock, they do 97 to 99% of the tensor peak on every PC
  here; ours does 94 to 96%, and our no-copy fold at long k 96 to 97%. On the 5070 Ti, where every miner runs
  2805 MHz, that is the whole gap: 197 to 199 TH/s against 191.5, and free operands at long k recover 2 of those 3 to 4
  points. Their clock is not free either, and on every PC it is the clock and watts of our sham at 1 an IMMA:

  | PC | Competitors | Our sham at 1 an IMMA | Our no-copy fold |
  |---|---|---|---|
  | RTX 4090, 450 W | 2625 MHz at 450 W (item 16) | 2636 MHz at 448 W | 2685 MHz at 400 W |
  | RTX 3060 | 1874 to 1879 MHz at 170 W | 1882 MHz at 170 W | 1912 MHz at 165 W |
  | RTX 3090 | 1573 to 1585 MHz at 345 to 350 W | 1590 MHz at 348 W | 1633 MHz at 339 W |
  | RTX 5070 Ti | 2805 MHz at 183 to 186 W | 2785 MHz at 188 W | 2805 MHz at 165 W |
  | RTX 5090 | 2472 to 2513 MHz at 503 to 513 W | 2565 MHz at 558 W | 2722 MHz at 575 W |

  What they do not show is its loss of work a clock. So their draw costs about an instruction an IMMA in watts and
  no IMMA time, and nothing built here does that.
- **Also seen.**
  - The release fold on the cp.async ring is 1.1% behind TMA on this 5090 and level with it on the 5070 Ti
    (−0.1%). The +2.3% on record was measured on the card since sent back; TMA stays.
  - The 5090's sham arms ran at 525 to 565 W of 575 W and 2540 to 2630 MHz, steadily; the driver gave the power
    cap as the reason all race (`clocks_event_reasons` 0x4). The competitors showed the same there (503 to
    513 W, about 2500 MHz, on a card that reaches 2722 MHz). Why the cap bites below the limit is not known.
  - Salad's logs dropped every `build` line of compilers.sh; `gate.sh` prints its own. Our CLI's run on the 3060
    failed on a malformed test address and was not repeated, so that PC has no CLI figure.

**Verdict: the gate fails as the reviews wrote it, and the kernel they describe is not worth building.** The prize is
real. With free operands and long k our fold is level with the best competitor on the 4090 and the 3090, within 2% of
it on the 3060 and the 5070 Ti, and 7% ahead on the 5090. But a draw the MMA warps issue themselves, even written a
stage ahead and spread between the mma, keeps little of it at stock power. At one instruction an IMMA: +0.4 to +1.6%
on the 4090, +0.8 to +2.4% on the 3060, nothing on the 5090 or the 3090, a loss on a clock-bound card. At half of
one: +2.0 to +4.1% on the 4090. Under a deep power cap it keeps more (the 4090 at 300 W: +3.0% at one, +6.6% at
half). But the sham at 1 an IMMA has 77 ALU steps for a thread's 112 bytes a stage, under three a 32-bit word, and a
clamped add and subtract of four packed bytes takes more. So by our count a real draw sits nearer 2 an IMMA, where
every card lost at every limit tried.

Two things stand between this fold and the competitors, both work a clock, and neither has a known cause: the 1 to 3
points the fold still lacks with free operands and long k on every card but the 5090, and the 2 to 6 points the
draw's instructions cost beside the IMMAs. Until one of them is explained, no fold of ours with on-chip operands
can be specified. The arms, the tools (`gate.sh`, the spin kernel, the switches) and the raw runs and logs
(`earn/tuning/gate`, with the scripts that make these tables from them) stay on branch `goal/gate` (c3f6bf85).

**The gate's follow-up: where the loop's last points go, and a producer warp in the fold's own block (11 November, rig01)**

The gate above left two things unmeasured: where the no-copy fold loses its last points a clock, and what a draw costs
in a warp of its own inside the fold's block. Both were measured on rig01's RTX 4090, against a line set beforehand:
with the producer at the instruction count a real draw needs, the best arrangement must net +3% on the release at
450 W, or the on-chip-draw route is closed for stock-power cards. Every arm is timing-only, on branch `goal/gate`.

How it ran: `hashrate.js`, 60 s after 15 s warm-up, three holds of the card (40, 2 and 20 minutes). The release fold
(k 2048) is the control. The no-copy fold at k 65536, the gate's best arm, is the base of both measurements: 341.86,
341.86 and 341.87 TH/s in three runs. Work a clock is TH/s per GHz and its share of the tensor peak (131.072 on this
card); "cycles a chunk" is 6144 over that share, the SM clocks one chunk of a 192x256 tile takes.

The arms:

- **The loop gap.** One thing removed or moved at a time, on the no-copy k 65536 fold: the readout at the tile's last
  chunk only (`PEARL_ABLATE_READOUT`); the ring's waits and arrivals off (`PEARL_ABLATE_RING`, `_ARRIVE`); the tile
  hash off (`PEARL_ABLATE_TRANSCRIPT_HASH`); all three; B fragments reloaded for every m16 tile (`PEARL_GATE_BSTREAM`,
  80 more ldmatrix a chunk); and in that body the readout moved into the next chunk's first k-step
  (`PEARL_GATE_RO_LAZY`), where a lazy readout would sit. In the fold's own body the moved readout spills 108 bytes,
  so it was not run there.
- **Warps given up** (`PEARL_GATE_MMA_MASK`): the block keeps its eight warps and 253 registers, and only some fold.
  Four fold, one a scheduler, and four are free; or seven fold and one is free. The host still counts every region,
  so these rates are scaled by the share of warps that fold (a half, seven eighths).
- **A smaller fold with warps beside it** (`PEARL_TALL_ROW_TILES=4`, `PEARL_GATE_PROD_WARPS=4`): 64x64 warps and a
  128x256 tile need 165 to 170 registers, which leaves room for twelve warps: the eight that fold and four more, one a
  scheduler. Its tile holds two thirds of the release's MACs and the host counts them as they are.
- **Producers** (`PEARL_GATE_PROD`): the warps that do not fold. Each stage a producer waits for the folding warps
  to release the buffer the stage before read, writes its 16-byte units of it (two 16-byte loads from the buffer
  being read, ALU steps, one store) and arrives on its FULL barrier. The buffer is read two stages on. A waiting
  producer tests its barrier every 100 ns and sleeps between (`PEARL_GATE_PROD_NAP`), or spins. Its work is counted
  against the folding warps' IMMA: 1, 2 and 3 to 3.7 instructions an IMMA are 9 to 11, 17 to 21 and 32 instructions a
  16-byte unit, because a smaller tile needs more operand bytes an IMMA (58.7 at 96x256 and 48 at 128x256 against
  37.3).

Checked in the machine code (sm_89, ptxas 12.6): in every mask and producer arm the folding warps' chunk loop is the
no-copy fold's instruction for instruction (426 instructions, 192 IMMA, 40 ldmatrix, 253 registers, no spill). The
128x256 arms hold 128 IMMA and 32 ldmatrix a chunk in 321 to 323 instructions, 165 to 170 registers, no spill. The
producers' instructions a stage were counted in the same disassembly. Three things had to be found first. A hardware
barrier for the producer's wait made ptxas fence every ring wait of the chunk loop and spill 100 bytes there. So did
a plain branch to the producer's code, until the test was made warp-uniform by a vote. And a third barrier for the
folding warps' own arrival, one that nothing waits on, gave "unspecified launch failure" within a second whenever a
column slot had two folding warps; the arms that ran count that arrival on FULL beside the producers'. The
four-fold-four-produce arms were measured in the third-barrier form, the others in the final one.

**RTX 4090, rig01, 450 W limit, `hashrate.js`.** The release fold ran 5 times, 316.98 to 317.64 TH/s. The no-copy k
65536 fold ran 3 times, 341.86 to 341.87. Change against the release is against the controls on either side of a run;
against the no-copy fold, against the runs of it on either side, or their mean in a hold that had none.

| Arm at 450 W | Runs | TH/s | vs release | vs no-copy k 65536 | SM clock | W | °C | TH/s per GHz | Of the tensor peak | Cycles a chunk |
|---|---|---|---|---|---|---|---|---|---|---|
| `base` Release fold, k 2048 (control) | 5 | 317.3 |  |  | 2531 MHz | 449 | 77 | 125.3 | 95.6% | 6425 |
| `nck` No copies, k 65536 (the base of both measurements) | 3 | 341.9 | +7.70% (+7.70, +7.70, +7.71) |  | 2685 MHz | 410 | 76 | 127.3 | 97.1% | 6325 |
| `kro` Readout at the tile's last chunk only | 1 | 349.1 | +9.99% | +2.13% | 2685 MHz | 387 | 74 | 130.0 | 99.2% | 6193 |
| `krg` Ring waits and arrivals off | 1 | 326.8 | +2.96% | -4.40% | 2685 MHz | 388 | 74 | 121.7 | 92.9% | 6616 |
| `khs` Tile hash off | 1 | 342.4 | +7.86% | +0.15% | 2685 MHz | 412 | 76 | 127.5 | 97.3% | 6316 |
| `kall` Readout, ring and hash all off | 1 | 349.4 | +10.07% | +2.21% | 2685 MHz | 386 | 74 | 130.1 | 99.3% | 6188 |
| `kbs` B fragments reloaded a tile: 80 more ldmatrix a chunk | 1 | 313.4 | -1.26% | -8.32% | 2521 MHz | 400 | 75 | 124.3 | 94.8% | 6478 |
| `kbz1` The same, readout moved into the next chunk's first k-step | 1 | 312.9 | -1.42% | -8.47% | 2533 MHz | 402 | 75 | 123.5 | 94.2% | 6519 |
| `m4k` 4 warps fold (one a scheduler), 4 leave | 1 | 282.0 | -11.14% | -17.50% | 2685 MHz | 353 | 71 | 105.0 | 80.1% | 7666 |
| `m4p0k` 4 fold, 4 producers that spin on their barrier, no work | 1 | 278.0 | -12.43% | -18.69% | 2686 MHz | 385 | 73 | 103.5 | 79.0% | 7781 |
| `m4n0k` 4 fold, 4 producers that nap 100 ns between tests, no work | 1 | 280.5 | -11.63% | -17.95% | 2685 MHz | 359 | 71 | 104.5 | 79.7% | 7708 |
| `m4n1k` 4 fold, 4 producers, 1.03 instructions an IMMA (9 a unit) | 1 | 271.0 | -14.63% | -20.74% | 2685 MHz | 431 | 77 | 100.9 | 77.0% | 7980 |
| `m4n2k` 4 fold, 4 producers, 1.95 an IMMA (17 a unit) | 1 | 262.4 | -17.33% | -23.24% | 2685 MHz | 445 | 77 | 97.7 | 74.6% | 8240 |
| `m4n3k` 4 fold, 4 producers, 3.67 an IMMA (32 a unit) | 1 | 255.6 | -19.46% | -25.22% | 2597 MHz | 449 | 78 | 98.4 | 75.1% | 8181 |
| `m4p2k` 4 fold, 4 spinning producers, 1.95 an IMMA | 1 | 268.5 | -15.41% | -21.46% | 2640 MHz | 449 | 77 | 101.7 | 77.6% | 7919 |
| `m7k` 7 warps fold, 1 leaves | 1 | 299.5 | -5.64% | -12.39% | 2685 MHz | 369 | 71 | 111.5 | 85.1% | 7219 |
| `m7n1k` 7 fold, 1 producer for all seven, 1.00 an IMMA | 1 | 184.9 | -41.72% | -45.91% | 2700 MHz | 319 | 67 | 68.5 | 52.2% | 11759 |
| `t4k` 128x256 tile, 8 warps of 64x64, no producer | 2 | 328.5 | +3.53% (+3.62, +3.45) | -3.89% | 2685 MHz | 414 | 76 | 122.4 | 93.4% | 6581 |
| `t12n0k` 128x256, 8 fold, 4 napping producer warps, no work | 1 | 322.3 | +1.60% | -5.71% | 2685 MHz | 411 | 76 | 120.1 | 91.6% | 6708 |
| `t12n1k` 128x256, 8 fold, 4 producers, 1.03 an IMMA (11 a unit) | 1 | 312.7 | -1.44% | -8.52% | 2580 MHz | 450 | 78 | 121.2 | 92.5% | 6644 |
| `t12n2k` 128x256, 8 fold, 4 producers, 1.97 an IMMA (21 a unit) | 1 | 306.4 | -3.43% | -10.38% | 2541 MHz | 449 | 77 | 120.6 | 92.0% | 6679 |
| `t12n3k` 128x256, 8 fold, 4 producers, 3.00 an IMMA (32 a unit) | 1 | 298.4 | -5.96% | -12.73% | 2504 MHz | 449 | 77 | 119.2 | 90.9% | 6759 |
| `t12p2k` 128x256, 8 fold, 4 spinning producers, 1.97 an IMMA | 1 | 308.7 | -2.71% | -9.71% | 2525 MHz | 449 | 77 | 122.2 | 93.3% | 6588 |

**RTX 4090, rig01, 300 W limit, `hashrate.js`.** The release fold ran 3 times, 270.97 to 272.53 TH/s. The no-copy k
65536 fold ran once, 312.19. Change against the release is against the controls on either side of a run; against the
no-copy fold, against the runs of it on either side, or their mean in a hold that had none.

| Arm at 300 W | Runs | TH/s | vs release | vs no-copy k 65536 | SM clock | W | °C | TH/s per GHz | Of the tensor peak | Cycles a chunk |
|---|---|---|---|---|---|---|---|---|---|---|
| `base` Release fold, k 2048 (control) | 3 | 271.8 |  |  | 2166 MHz | 300 | 65 | 125.5 | 95.7% | 6419 |
| `nck` No copies, k 65536 (the base of both measurements) | 1 | 312.2 | +15.03% |  | 2438 MHz | 300 | 66 | 128.1 | 97.7% | 6289 |
| `m4k` 4 warps fold (one a scheduler), 4 leave | 1 | 270.2 | -0.43% | -13.44% | 2566 MHz | 300 | 67 | 105.3 | 80.3% | 7647 |
| `m4n1k` 4 fold, 4 producers, 1.03 instructions an IMMA (9 a unit) | 1 | 243.4 | -10.31% | -22.03% | 2387 MHz | 299 | 67 | 102.0 | 77.8% | 7897 |
| `m4n2k` 4 fold, 4 producers, 1.95 an IMMA (17 a unit) | 1 | 236.0 | -13.03% | -24.39% | 2336 MHz | 300 | 67 | 101.0 | 77.1% | 7970 |
| `t4k` 128x256 tile, 8 warps of 64x64, no producer | 2 | 298.8 | +9.86% (+10.06, +9.66) | -4.29% | 2426 MHz | 300 | 66 | 123.2 | 94.0% | 6539 |
| `t12n1k` 128x256, 8 fold, 4 producers, 1.03 an IMMA (11 a unit) | 1 | 271.7 | -0.32% | -12.98% | 2235 MHz | 300 | 66 | 121.6 | 92.7% | 6625 |
| `t12n2k` 128x256, 8 fold, 4 producers, 1.97 an IMMA (21 a unit) | 1 | 264.1 | -3.09% | -15.40% | 2190 MHz | 299 | 66 | 120.6 | 92.0% | 6678 |

What the tables say:

- **The loop gap is the readout.** The no-copy k 65536 fold does 97.1% of the tensor peak: 6325 cycles a chunk
  against 6144.

  | Piece | Arm | Work a clock | Cycles a chunk |
  |---|---|---|---|
  | The readout (96 XOR3 and 3 shared XOR a thread a chunk) | readout at the last chunk only | +2.13% | −132 |
  | The tile hash, at k 65536 | hash off | +0.15% | −9 |
  | The ring's waits and arrivals | off | −4.40% | +291 |
  | All three | off | +2.21%, 99.3% of the peak | −137 |
  | What is left: 40 ldmatrix a chunk and the loop's own steps | | 0.7% | 44 |
  | An ldmatrix more in the mma stream | 80 more a chunk | −2.4% for the 80 | +1.9 each |
  | The readout moved into the next chunk's first k-step | in the body with 80 more ldmatrix | −0.6% | +41 |

  - The readout can't be cut by count: one XOR3 for every two IMMA is the least that folds a lane's 192 accumulators
    into its three words, and the chunk is the spec's 128 of k. Moving it did not help: ptxas still schedules the
    XOR3 in runs, so the seam only moves.
  - The ring's waits cost nothing, and they do work: without them the fold is 4.4% slower, and with the readout
    off as well it is not. That fits the readout being a seam on each scheduler which the scheduler's other warp
    covers in part, with the ring keeping the two warps apart enough to cover it. It was not measured directly. A
    lone warp keeps 80% of its scheduler's tensor pipe (below), which bounds what a better stagger could recover.
    No stagger was built.
  - The hash is spread out by long k already. The fragment loads are inside the 0.7%.
  - The arm with 80 more ldmatrix also ran at 2521 MHz on 400 W of its 450 W limit, as the RTX 5090's sham arms did
    in the gate. Why the card leaves its top clock there is still not known.
- **A producer's instructions cost no IMMA time when its scheduler keeps two folding warps, and only then.**
  - 128x256 fold beside four producer warps, at 450 W: work a clock is 120.1 TH/s per GHz with the producers
    idle, 121.2 at 1.03 instructions an IMMA, 120.6 at 1.97 and 119.2 at 3.00. With producers that spin it is 122.2
    at 1.97, level with the same fold alone (122.4). At 300 W: 123.2 alone, 121.6 and 120.6. This is what the
    competitors' figures look like: a draw that costs watts and no IMMA time.
  - Four of eight warps folding, one a scheduler, with the other four producing: work a clock falls 3.4% at 1.03
    an IMMA and 6.5% at 1.95. A lone folding warp has no partner to keep the pipe fed, so the producer's instructions
    cost it what they cost inside it (the gate's sham: 3 to 4% at 1 an IMMA).
  - One producer for seven folding warps cannot keep up, and the fold waits for it: 184.9 TH/s against 299.5 with
    the eighth warp gone.
- **The watts are the cost.** At 450 W the 128x256 fold holds the 2685 MHz top clock on 414 W alone and on 411 W
  with idle producers. Producers at 1.03, 1.97 and 3.00 an IMMA bring it to the limit at 2580, 2541 and 2504 MHz. At
  300 W: 2426 MHz alone, 2235 and 2190 at 1.03 and 1.97. The release's copies cost 6.1% of clock at 450 W (2531
  against 2685 MHz); a producer at 1 an IMMA gives 3.9 of that back and at 2, 5.4.
- **Room is the other cost.** Only the 128x256 fold leaves registers for producer warps, and it is 3.8% a clock
  behind the 192x256 fold (122.4 against 127.3 TH/s per GHz): 328.5 TH/s against 341.9. Giving up warps of the
  eight costs more: seven folding is exactly seven eighths (299.5), and four, one a scheduler, 282.0, since a lone
  warp reaches 80.1% of the pipe.

Net against the release fold on the RTX 4090:

| Arrangement | No producer work | About 1 an IMMA | About 2 | About 3 |
|---|---|---|---|---|
| 450 W: sham inside the folding warps, 192x256, k 65536 (the gate) | +7.7% | +1.65% | −3.5% at k 2048 | |
| 450 W: 128x256 fold, four producer warps | +1.6% (+3.5% with no producer warps) | −1.4% | −3.4% (−2.7% spinning) | −6.0% |
| 450 W: four of eight warps fold, four produce | −11.6% | −14.6% | −17.3% | −19.5% (3.67) |
| 450 W: seven fold, one produces | −5.6% (the eighth warp gone) | −41.7% | | |
| 300 W: sham inside the folding warps (the gate) | +15.0% | +4.0% | −2.0% at k 2048 | |
| 300 W: 128x256 fold, four producer warps | +9.9% with no producer warps | −0.3% | −3.1% | |
| 300 W: four of eight warps fold, four produce | −0.4% with the four gone | −10.3% | −13.0% | |

**What a real draw needs.** The least we can write down is about 18 instructions a 16-byte unit. It takes the noise
table held column-wise, eight rows to a unit, so that ldmatrix's transposed form reads what a column draw writes:
one load of the pair offsets, four address adds, four 8-byte table loads, four subtracts, four byte selects to
interleave two k indexes, one store. That is 1.3 instructions an IMMA on the 192x256 tile and 1.7 on the 128x256
one. The one draw built so far (`pearl_e4_drawt`) is about 32 a unit: 2.3 and 3.0. Nothing has been built at 18.

**Later the same day: the verdict below does not stand on its evidence. Its sham draw was chained; see the two probes
after this entry.**

**Verdict: the on-chip-draw route is closed for stock-power cards.** The line was +3% on the release at 450 W with
the producer at a real draw's count. The best arrangement at 1.3 to 1.7 an IMMA is about 0% (the draw inside the
folding warps of the 192x256 fold) or about −3% (the 128x256 fold with producer warps); no arrangement reaches +3% at
1 an IMMA, and at 300 W only the draw inside the folding warps does (+4.0% at 1, less at a real draw's count). What
would have to change is known now: a fold that leaves room for producers and still does 97% of the peak (ours does
93.4%), and a draw under one instruction an IMMA in watts. Three competitors' figures show both. We have neither,
and nothing measured here says how to get them.

The arms, the raw runs and the table script are on branch `goal/gate` (4014c732, `earn/tuning/gate`).

**What fills the tensor pipe, and a draw that fits in it: two probes (11 November, rig01, RTX 4090)**

The three other miners could not be studied from inside: all three ship protected and none runs under Nsight Compute.
Their shares and the card's readings give the outline (two warps a scheduler, a proof tile that is one thread's sums,
k 65536, no memory traffic, 449 W at 2600 to 2625 MHz). The rest here is our own: two probes that run the fold's
instruction stream alone, one block an SM, each warp timed with the SM's cycle counter
(`earn/native/probes/pipeprobe.cu` and `pipeprobe2.cu` on branch `goal/nv-kernel`). Pipe use is 16 cycles an IMMA over
the cycles taken. Kernels of 0.05 s; clock and watts from 20 to 25 s of back-to-back launches at the 450 W limit.
Timing only: the streams compute nothing real.

| Stream (8 warps, two a scheduler, 48 IMMA a k-step, 4 k-steps a chunk) | Pipe use | Clock and watts, 450 W limit |
|---|---|---|
| IMMA only | 99.93% | |
| + 4 ldmatrix after every k-step (kept in every row below) | 100.00% | 2687 to 2700 MHz, 292 to 373 W by the operand bytes loaded |
| + readout as one run of 96 XOR3 at the chunk's end, word kept by a shared load and store | 98.72% | 2685 MHz, 399 W |
| the same with a shared XOR | 97.91% | |
| + readout as three runs of 32 XOR3, each after its tiles' last IMMA | 99.67% (99.71% with a shared XOR) | 2691 MHz, 318 W |
| + readout as six runs of 16 | 99.68% | |
| + 4 draw-shaped units a k-step (58 instructions), after the 24th IMMA | 99.61% | 2595 MHz, 449.5 W |
| the same at the k-step's end | 99.52% | |
| the same in two halves / one unit every 12 IMMA | 98.44% / 98.44% | |
| the same with two units' steps grouped by kind | 97.68% | |
| + 4 units that read their four table offsets first | 98.80% | 2670 MHz, 421 W |
| loads, readout in three runs, 4 such units, and one shared add and test a k-step | 94.71% (95.15% without the add and test) | 2660 MHz, 438 W |
| the same with 3 units a k-step | 96.97% | 2685 MHz, 418 W |
| the same with the 4 units at the k-step's end | 96.50% | |

| Stream (first probe, roles and counts set at run time) | Pipe use |
|---|---|
| One warp a scheduler, IMMA only | 91.0% (17.6 cycles between a warp's own IMMA) |
| Two a scheduler | 99.5 to 99.6% |
| Three a scheduler (12 warps, 32 IMMA a k-step) | 98.4 to 98.9% |
| Four a scheduler (16 warps, 24 IMMA a k-step) | 99.5% |
| Two folding, a third warp doing 1.1 to 1.7 instructions an IMMA beside them (ALU, loads, stores, draw units) | 95.0 to 96.6% |
| One folding, one working beside it | 78.6 to 82.4% |
| Two, one chained ALU step after every IMMA / two steps | 98.5% / 96.3% |
| Two, a run of 96 chained ALU steps every 192 IMMA | 94.1% (warps staggered: 94.8%) |
| Two, runs of 8 / 16 / 32 / 64 chained steps every 192 IMMA | 98.7 / 98.8 / 98.7 / 97.8% against 98.9% |
| Two, readout runs: tested between IMMA segments at run time / not | 91.6% / 98.0% |
| Three a scheduler, readout runs / four | 95.9% / 97.2% |

The first probe's own run-time tests cost it pipe time (its two-warp IMMA-only figure fell from 99.6% to 98.7% as
tests were added), so its rows compare with each other only.

- **Two warps taking turns fill the pipe, and nothing else does.** One warp alone gets 91%: it cannot follow its own
  IMMA in under 17.6 cycles. Three or four a scheduler add nothing. A third warp has no free issue slots: its
  instructions cost the folding warps 0.44 cycle each, the same as work inside them.
- **What costs is waiting and control, not the count of instructions.** Steps that each wait on the one before cost
  1.4 to 1.9 cycles each in a long run and 0.2 to 0.3 one at a time. A run-time test between two IMMA segments costs
  about 10 cycles (ptxas puts a fence before the next IMMA). Steps that depend on nothing recent, placed where ptxas
  can weave them between the IMMA, are close to free: the readout cut in three runs costs 0.3%, and four draw-shaped
  units a k-step (1.2 instructions an IMMA) cost 0.4%.
- **The pieces do not simply add.** Loads, readout and draw each alone hold 98.8 to 100%; all three together fell to
  94.7 to 97.0% as ptxas ordered them. So the order has to be pinned by hand and measured piece by piece.
- **Watts follow the operand bytes and the draw.** The same IMMA stream drew 263 W on constant registers, 292 W on
  mostly-zero loaded bytes and 373 W on random ones, all at the top clock. Four draw units a k-step added 77 to 130 W.
  At 450 W the draw-only stream ran at 2595 MHz and 99.6%: 338.8 TH/s by count. The table's watts come from two
  builds: random operand bytes (loads 373 W, one-run readout 399 W, the 58-instruction draw 449.5 W) and mostly-zero
  ones (loads 292 W, three-run readout 318 W, the offset-reading draw 421 W, all together 438 W).
- **The gate's verdict does not stand on its evidence.** Its sham draw chained 4 to 24 ALU steps on each word inside a
  fold with a ring, and read 2 to 4% of the pipe for one instruction an IMMA. A draw's units are four steps deep and
  independent of each other, and here they cost 0.4 to 1.2%. No real fold has been measured this way yet, so the route
  is open again, not proven.
- **Our fold's loss at long k is the readout's run.** The release fold has a predicated branch every 32 IMMA and holds
  99.2% with the readout off; the readout as it stands costs 2.13%, three runs cost 0.3% in the probe.

**Not verified.** Nothing here is a mining rate: 338.8 TH/s is pipe use times clock for a stream that computes
nothing. The draw unit's offsets and tables are stand-ins of the right size. One card. One run a row (the 20 s rows
repeat the short ones to 0.1 point). A real fold built this way has not been made.

**Tried and dropped (don't repeat without a new idea)**
"Item N" alone means issue #287 or an earlier numbered plan; "suggestion list item N" is the ten-item list worked on
6 November, which stopped on every item before any GPU time.

- RDNA 3 readout, hash and k (goal-rdna3c, 15 November, Salad; numbers under AMD Radeon). REG2's 32 x 8 tile in
  the RDNA 3 fold (`PEARL_AMD_REG2_11`, bit-exact, verify 400 of 400): +0.1% and level, of a 0.8% bound; its
  cross-lane steps a slab later: -1%. Two fused hashes a thread: -0.3%. The wrap readout at k 4096 and 8192: -0.5
  and -1.2%. Operand fills 32-63 against 48: level or worse. Switches on branch `goal/rdna3c`, off, not merged.
- RDNA 3 persistent fold and lagged slab barrier (goal-rdna3b, 15 November, Salad; the numbers are under AMD Radeon).
  One block a slot running tiles back to back (static walk, a tile counter, blocks started apart, the hash moved into
  the next tile) lost 2.6-3.2% on an RX 7900 XTX and 3% on an RX 7800 XT, at a higher clock: cycles, not watts. Groups
  of 2-4 tiles a block were level to -1.6%. Three LDS buffers with the fragment reads before the barrier: +0.25% on the
  7800 XT (7 of 7 rounds, verify 400 of 400) and level on the XTX with loads one step ahead; every other form spilled
  and lost. Loads straight into LDS don't exist on gfx11 (ROCm 7.2). Serpentine WMMA order: level. The switches
  (`PEARL_AMD_PERSIST`, `PDYN`, `PGRP`, `PSKEW`, `PHASH`, `LAG3`, `LAG3PF`) are on branch `goal/rdna3b`, off, not merged.
- RDNA 4 and the RX 7800 XT: the hash on its own queue, and a shorter launch, against a slow PC's queue stops (11
  November, Salad; the numbers are under AMD Radeon). On a slow RX 9060 XT (`3540c3e2`) the hash queue was level
  (−0.04 to −0.27%, twin ±0.3%) and 5 of 6 launches were still stopped: a launch there is 38 ms, longer than the
  33 ms between stops, and the hash queue gets work once a launch. 128 offsets a launch read −0.6 to −1.1%, with or
  without the hash queue. A ticker kernel on a third queue every 2 ms ends the stops there (+2.4 to +3.7%) but takes
  half a CPU core or more, so it is a switch on branch `goal/amd` (`PEARL_AMD_TICK_US`, `PEARL_AMD_TICK_WAIT`), off
  by default, not for merging as it is. On a slow RX 7800 XT (`559a11d6`) the hash queue read 0 to −0.5% and the
  ticker −4%: there a second queue's work stops the fold's queue itself, in every launch. And the hash queue takes
  one more CPU core.
- RX 7900: several batches a launch, and fewer commands between folds (the AMD goal, 10–11 November, Salad, two RX
  7900 XTX PCs and an XT; the numbers are under AMD Radeon). The review's lead was that each launch costs more on a slow
  PC, so fewer, longer launches should win there. They lose: 4 batches a launch, as strips so that B' in use stays one
  batch's, read −6.2 to −6.4% on the slow PC, and the fold alone gained nothing from 2 or 4 batches a launch on any of
  the three PCs. A slow PC stops a busy queue every 33 ms, and a 41 ms launch is always hit. 2 batches a launch (20 ms)
  read −0.2% there, batch 64 (5 ms) −1.1 to −1.4%; on the RX 7900 XT 2 and 4 batches a launch read −0.4% and −0.6%.
  With the hash on its own queue, zeroing the hit count there instead of before the fold, and dropping the fold queue's
  wait for the slot's last hash, were level (+0.09 to +0.29%, −0.11 to +0.27%, twin +0.05 to +0.44%); so was a ticker
  kernel on a third queue every 2 ms or 0.5 ms (+0.11 to +0.22%). The switches are on branch `goal/amd`
  (`PEARL_AMD_STRIP`, `PEARL_AMD_STRIPS`, `PEARL_AMD_HASHQ_ZERO`, `PEARL_AMD_HASHQ_WAIT`, `PEARL_AMD_TICK_US`), all off
  by default, not for merging.
- RTX 5090 B' drawn on the SM (research item 8, 10 November, desk study, no card time). The "+1.81%" from 2 November
  (b68b0399, 53ddd838, 87ba3d6e) was `PEARL_TALL_SHAM_DRAW`, a timing-only stand-in: it reads each B fragment from
  shared memory again and adds one IADD3 a register, but skips the real draw's table loads and byte gathers (each B'
  byte is 48 + T[col][p0] - T[col][p1], 8 one-byte lookups a register). So +1.81% is the ceiling (skipping B' entirely
  read +7.65%). A real draw needs about 2.5x the sham's extra instructions (about 100 a warp a stage against 40), and a
  tile's table is 32 KB with only 2.7 KB of shared memory free beside the ring. Rank space doesn't help: the chunk
  length equals the rank (128), so the dense work is the same and the sums leave int8. Item 7's on-chip draws on an
  RTX 3090 lost 3.6-5.6% for the same reason. Closed without the re-race; the sham rebased onto f7ccb357 is on branch
  item/8 (be35e42b) if anyone wants it. The 5090 lead to follow instead is long k (item 7: +2.80% on a 3090 at
  k 16384 with operands in L2). Item 21 (below) raced it: long k loses on the 5090 with stored operands.
- RTX 30 B' held at a width that fills the L2 slice (#287 item 3, 6 November, race mode, `PEARL_AMPERE_FLIP_FILL` on
  exp/287-3). On an RTX 3080 10 GB it lost to the band walk: −0.44% (±0.07, 0 of 46 rounds ahead) on a 320 W PC and
  −0.55% (±0.20, 2 of 45) on a 309 W PC. Memory busy fell from 21–23% to 13–14% and the clock rose 1.1%, but TH/s per
  MHz fell 1.5–1.6%. On the same card band 4 lost 0.6–0.8%, width 256 lost 0.27% and width 1024 read +0.11% (±0.12),
  so band 8 at 512 stays. The 3080 Ti's flip read −0.14% (±0.06) against the band walk at 400 W but +1.63% (±0.27) on
  a 350 W PC that slowed to about 1300 MHz as it warmed, so it keeps the flip.
- k 4096 on NVIDIA: RTX 20 (item 9) loses 1.2–2.8% to DRAM; RTX 50 and 40 (item 8) −0.78% and −0.29% at the power cap.
- Larger k after cutting operand traffic (suggestion list item 3, 6 November, from records, no GPU run). On a 4090 the
  fold takes 102,658, 101,764 and 101,220 cycles per 2048 of k at k 2048, 4096 and 8192: about 1,900 cycles a tile are
  fixed (mostly the hash), so depth can win at most +1.64% a clock at k 16384. A 3090 with operands held in L2 turned
  +1.30% and +1.56% a clock at k 8192 and 16384 into +0.6% and +0.8% at the power cap. With today's operands depth
  loses (above), and no on-chip operand supply pays (compact tables, below), so it stops. The cp.async fold's
  `tile_srcs` offsets are 32-bit and overflow past k 16384.
- CUDA 13's ptxas for RTX 30 (sm_86) and RTX 20 (sm_75) (PR #290 research item 14, 10 November, static, no GPU
  run). The shipped `pearl_core_cu13.node` holds only sm_89 and sm_120 (cuobjdump of v0.6.14's), so RTX 30 and 20
  mine on 12.8 code; the "CUDA 13 core" Vast rows are RTX 50 and RTX PRO only. Built `pearl_kernel.cu` on rig01 with
  ptxas 12.8.93 and 13.3.73 from the same PTX, and with cicc 13.3 too. sm_75: all 24 kernels byte-identical under
  both. sm_86: 20 of 24 identical; the tall fold differs only in register and predicate names, one NOP turned into a
  no-op uniform add and one branch's sense: 3,056 instructions in the same order, 254 registers, 0 spill, 7,775
  stall cycles in both, control bits different on 4 instructions. `pearl_blake3_chunk_cvs` and
  `pearl_restamp_operand` save 2 stall cycles of 3,100 and 4,600. The 12.8 build's fold hash (35efb7e36535) is the
  release's, so the comparison is against what ships. A race could only measure noise, so none was run. Ptxas 13's
  gain is Blackwell's (+3.2% on the 5090); it doesn't reach Ampere or Turing.
- Computing only the hash word the target test reads (PR #290 research item 19, 10 November). The kernel already
  does this. Every fold hashes a region with `pearl_transcript_msw`, `pearl_hp_msw` or `pearl_hp_msw_pair`, which
  return only the top word, and runs the full hash again only for the rare region that passes the first-word test.
  What was left to try was writing the dead work out by hand: in the last BLAKE3 round, diagonals 1 and 3 don't
  feed the word that is read. A test switch (`PEARL_HASH_MSW_TRIM`, on scratch branch `exp/290-19`, commit
  546551c9) left them out. ptxas was already dropping them: the PTX loses 20 lines and 4 shuffles a tall fold, the
  machine code loses one instruction. sm_75: all 25 kernels byte-identical. Tall fold on sm_86 3,042 → 3,041
  instructions (stall cycles 7,775 → 7,778), sm_89 3,030 → 3,029 (7,704 → 7,699, same under ptxas 12.8 and 13.3),
  sm_120 3,706 → 3,705 (9,679 → 9,645). Raced on Salad, 900 s, 4 s turns, with an A/A twin of the base:

  | Card | Rounds | Base TH/s | Trim vs base | Twin vs base |
  |---|---|---|---|---|
  | RTX 3090 (420 W cap, CUDA 12.8) | 69 | 112.44 | +0.017% ± 0.040, ahead 38/69 | +0.038% ± 0.028, ahead 45/69 |
  | RTX 5070 Ti (300 W, CUDA 13.3) | 74 | 190.44 | +0.021% ± 0.038, ahead 35/74 | +0.006% ± 0.027, ahead 36/74 |

  The trim is inside the twin's spread on both. verify-hits passed 400/400 on both cards with 0 failed, and the hit
  lists matched the base on all four completeness shapes. RTX 40 (sm_89) was not raced; its code changes by the
  same one instruction. Dropped, and the switch is not on the PR: there is nothing left in the hash for the target
  test to skip.
- RTX 20: a smaller transcript hash for the L0 instruction cache (item 11), at most +0.10%.
- RTX 30/40: copy groups and readout fences (item 10); hash control bits, warp turn-taking (#18), IMMAs during the hash;
  a wider window (g6: costs 1,380 cycles a tile to free at most 355); Z6 (no clearing at all saves only 20 cycles);
  f39 (`PEARL_TALL_HASH_FMA`, −3,517 cycles).
- RTX 4090 hash tail (#287 item 1, rig01, 450 W, 6 November). The hash costs 1,397 cycles a tile here (102,778 against
  101,381 with no hash). Every form lost in race mode, 25 rounds, 0 of 25 ahead:
  - chunk 0's stage-1 refill after the chunk, so the partner runs 192 IMMAs before it waits, not 128: −0.61%. The
    knee moved from about 2,600 to 3,300 cycles, but the hash got only 61 cycles cheaper and the moved refill cost 416.
    Below the knee each cycle of hash still costs about 0.4: the partner and then the hasher run alone.
  - the same refill moved into chunk 1, with a guard before chunk 0's readout: +3,150 cycles, ptxas rebuilt the loop.
  - the hash as a `__noinline__` call: the chunk loop stays the same whatever the hash's form, but the call itself
    costs 270–340 cycles: −0.76%.
  - in that call, the adds as IMAD: the hash releases 480 cycles sooner but the tile gains only 40, because the hash
    then takes issue slots from the partner's IMMAs (−0.62%; whole-region hash only, −0.56%). A shorter all-ALU hash
    does pay: dropping a round (timing only) released 280 cycles sooner and saved 316 a tile. The hash has no ALU
    instruction to spare, so nothing is left to try there.
- Tile shape on RTX 30, 40 and 50 (suggestion list item 5, 6 November, no GPU run). The tall fold's 192 accumulators a thread (192x256, eight
  96x64 warps) is the most 255 registers allow at 256 threads. 128 lost three ways: 128x256 on the same ring −4.6% on a
  4090 (probes/README.md), twelve 64x64 warps over 192x256 at 168 registers −2.1 to −5.3% a clock (branch
  exp/rtx30-layout), and on a 5090 the 192x256 tile was +3.4% over 128x256. 160 can't keep whole regions in a warp: a
  region spans 32 rows by 64 columns, so a warp holds 64, 128 or 192 words. A 160x256 tile that splits regions between
  warps moves 11.4% more bytes a MAC than 192x256 and frees registers nothing has used to gain; the model puts it near −2%.
- RTX 30/40 unrolling and prefetch distance (suggestion list item 7, 6 November; compile and records, no GPU run). Inside a
  chunk everything is unrolled, and ptxas already keeps A two m16 tiles ahead and reloads B under the last tile. Left
  to try were the chunk loop unrolled by 2 (`PEARL_TALL_CHUNK_UNROLL`: 470 → 455 instructions a chunk, all waits,
  branches and moves; by 4 it spills) and the A fragment after each copy group loaded before it (`PEARL_TALL_A_AHEAD`:
  same instructions). Both bound under 0.3%: with no hash, removing the ring's waits (33 wait and branch instructions
  a chunk, and half those post-copy-group stalls with them) made the tile 164 cycles slower, and 15 cheap instructions
  are at most 0.7% of a chunk's energy, about +0.2% of clock. A deeper ring doesn't fit (98,688 of 101,376 bytes of
  shared) and a 4 × 64 ring lost 2% on the probe; Blackwell's streamed fragments lost 3.5–5.3% a tile on this ring.
  The two switches stay on exp/gpu10-7, off.
- RTX 50 (item 16): drawing B' on the SM (+1.81% against a +2.5% bar); TMA multicast hangs because its transaction count
  never arrives on sm_120.
- RTX 5090: two CTAs sharing B' by a copy between their shared memories (#287 item 7, 7 November,
  `PEARL_TALL_CLUSTER_PUSH` on branch exp/287-7, CUDA 13.3, 254 registers, no spill). Each CTA loads half of the pair's
  B' box and copies it to the peer (`cp.async.bulk` shared::cta to shared::cluster), so the L2 serves one box a pair. It
  is correct (verify 400/400) but loses on both counts:
  - Energy: `multicast-probe.cu`'s push arms, all held to 7500 cycles a stage at 2425 MHz (PC `380ecfca`), put a
    copied byte at 33.0 pJ and a byte from L2 at 34.1 pJ. The copy saves 2% of an L2 box (fp 0.02, −0.04 to 0.04 over 6
    rounds), against a gate of 50%.
  - Time: in race mode on the same PC, −50.50% ±0.023 against the release (49 rounds, 0 ahead; 197.3 against 398.7
    TH/s). The half gets one stage to land and the copy one more, where the release gives a whole box two. A box takes
    longer than one stage under load, and the ring has no room for a fourth buffer.
  - That PC was clock-held (2467 MHz at 409 W of 540 W): skip B' read −0.26% ±0.032, the release's copy −0.005% ±0.021.
  - At the probe's default pace (3240) the loaded arms can't keep up (a2 4807, h2 3492, p 5962 cycles a stage, PC
    `64002f7c`), so its energies aren't comparable and fp read 3.02 (1.30 to 27.12). On exp/287-7 `multicast-read.js`
    now gives no verdict then and names a pace to rerun at. One high-priority box, about 50 minutes.
- RTX 40/50: making A' and B' on the SM from their compact tables (each row's 128 noise bytes plus one pair table, 16x
  smaller than A'). Built on 29 October (branch exp/newkernel, never merged): producer warps drawing into the ring ran
  80.8% of tensor peak a clock on the 4090 against 95.75%, at 1.68 pJ a MAC against 1.42. The one 255-register layout
  with producer warps (4+4) reached only 93.25% before any draw. Drawing in the fold's own warps costs 1.1–1.6 tensor
  cycles an instruction (E4 on a 3090, item 16's sham draw), so even at half a staged byte's energy a full draw nets
  −1.3 to −3.5%, and a tile's 57 KB of tables don't fit beside the ring (2.7 KB free). Rechecked on 6 November for
  suggestion list item 1, no GPU time.
- RTX 30: a cheaper in-fold A' draw, and a long k (item 7, 10 November, race mode, RTX 3090 `9b482f0a` at 350 W,
  memory 10% busy under the release). `-DPEARL_E4_DRAWT` (branch exp/item7-gen, bench only) holds E4's dense table
  transposed, so four rows of one k index are two loads and one add, and 8 byte selects make the ring's rows: 72
  instructions a chunk a warp against E4's 134, 252 registers, no spill. Against the release (35 rounds, copy of the
  release −0.002%):
  - DRAM ablation +1.75% (1592 → 1618 MHz); E4's data path with no draw +0.88%.
  - E4's draw −3.59% (1587 MHz); the transposed draw −5.57% (1613 MHz, so it saved energy, but did less work a
    clock). Fewer instructions didn't help: its table loads are long-latency and sit in the MMA warps' stream, and
    only 6 of 8 warps draw. There are 3 registers left to load them earlier.
  - Long k, second race on the same PC (24 rounds): k 8192 with stored operands −2.71% (memory 36% busy, 1530 MHz);
    k 16384 with operands in the L2 (DRAM ablation) +2.80%. So on this PC a DRAM-free, long-k kernel could gain at
    most 2.8%, and E4's draw alone costs 4.5-6.4% against its own data path. PeakMiner's +6.6% (item 2) was on a
    3060 whose memory was 39-65% busy.
    The cp.async fold's tile offsets were 32-bit and overflowed past k 16384; item 21 (below) widened them.
  - RTX 3060 ceiling (item 7b, same builds at 1c78804b, nvtest-rtx3060-72, 170 W cap, 16 rounds, twin −0.02%):
    DRAM ablation +2.82% (±0.12, 16 of 16 ahead), memory 28.7% → 0.2% busy, 1777 → 1828 MHz; the whole gain is
    clock, TH/s per MHz unchanged. k 16384 with operands in the L2 +4.05% (±0.17, 16 of 16), 1824 MHz (+1.4% a
    clock from the long k). So the 3060's ceiling is 4.05%, higher than the 3090's 2.80%, but a draw would have to
    cost under 4% to win, and E4's draw cost 4.5-6.4% against its own data path on the 3090. Dropped on the 3060 too.
    This PC's memory was 29% busy, not PeakMiner's 39-65%; a busier 3060 could sit higher.
- Long k on NVIDIA (PR #290 research item 21, 10 November, race mode at `9ac40341`, branch item/21). Every
  competitor mines k 65536; we mine 2048. The tall fold now builds for any k from 2048 to 65536 with
  `-DPEARL_FOLD_K=K` alone (64-bit cp.async offsets past k 16384, PREGUARD off past 2048, and the DRAM ablation on
  the TMA fold too). Operands at the mainnet m and n, so A' and B' grow with k (24 GB at 65536). Against the release:

  | Card | PC | Arm | vs release | Ahead | Memory busy | Clock |
  |---|---|---|---|---|---|---|
  | RTX 5090 (570 W) | a11172a9 | k 8192 stored | −0.46% ±0.10 | 1/24 | 2.3% → 7.0% | 2600 → 2543 MHz |
  | RTX 5090 | a11172a9 | k 16384 stored | −3.53% ±0.11 | 0/24 | 20.8% | 2457 MHz |
  | RTX 5090 | a11172a9 | k 65536 stored | −13.0% ±1.0 | 0/21 | 3.2% → 68.7% | 2623 → 2229 MHz |
  | RTX 5090 | a11172a9 | k 2048, operands in L2 | +0.73% ±0.08 | 24/24 | 0.0% | 2618 MHz |
  | RTX 5090 | a11172a9 | k 16384, operands in L2 | +1.87% ±0.09 | 24/24 | 0.2% | 2594 MHz |
  | RTX 5090 | a11172a9 | k 65536, operands in L2 | +1.33% ±0.05 | 24/24 | 0.4% | 2612 → 2593 MHz |
  | RTX 3060 (185 W) | b1785794 | k 16384 stored | −2.18% ±0.12 | 0/17 | 39.5% → 96.2% | 1904 → 1840 MHz |
  | RTX 4090 (~550 W) | bdcb5688 | k 32768 stored | −29.0% ±0.02 | 0/25 | 3.0% → 100% | 2670 MHz both |

  - **Dropped:** with stored operands long k loses on both cards, more the longer k gets. A launch's B' must stay
    in the L2, so the batch narrows as k grows (2048 → 64 column offsets on the 5090) and all of A' is read once a
    launch: DRAM traffic grows with k, and at the power cap it comes out of the clock.
  - Long k's own gain is about +2.1% a clock on the 5090 (k 16384 and 65536 alike, operands in L2) and +1.2% on the
    3060 (stored). So a kernel that made its operands on chip could gain at most about 1.9% on the 5090, against
    item 8's finding that drawing B' on the SM can't pay there. The 5090's gap to PeakMiner and SRBMiner is not k.
  - **Correctness:** verify-hits passed at k 8192 (400/400), 16384 (400/400) and 65536 (309/309) on the 5090's TMA
    fold, at k 16384 (167/167) on the 3060's cp.async fold, and at k 32768 (400/400) on the 4090's, the first run
    of the 64-bit offsets (B' is 8 GB there). The 4090 at k 32768 is DRAM-bound: its batch narrows to 16 offsets. The k ≤ 16384 cubins for sm_89 and sm_120 are
    byte-identical to before. The copy of the release read +0.01% (5090) and −0.01% (3060).
  - The first 3060 box (609fef99) had another job on its GPU (100% busy, 6 GB in use before we started) and read
    16 TH/s; discarded.
  - Cost: Salad boxes, about 2 hours in all, about $1. A fourth (4090) checked k 32768.
- Shared-memory layout and fragment loads on RTX 40 and 50 (suggestion list item 6, 6 November, no GPU run). The tall fold's 64-byte
  stage rows with unit q at q ^ ((r >> 1) & 3) (SWIZZLE_64B under TMA) give conflict-free `ldmatrix` and `cp.async`.
  It loads only `LDSM.x4`, 40 a chunk a warp, 106.7 shared bytes an IMMA: the least a 96x64 warp tile can read. Its
  address math is 2 IADD and 2 LOP3 a stage on sm_89 and sm_120. Holding k-step 1's lane bases in two more
  registers, with A's base centred, cut sm_89's chunk loop from 540 to 535 instructions (254 registers, no spill).
  That's about 0.23% of a tile's energy, so about +0.07% TH/s. On sm_120 it saves nothing, because the ring roles
  sit in plain registers there. The readout's `red.shared.xor` runs 16 serial passes (8 words on 2 banks, 4 lanes a
  word). Fixing that would save the hash seam at most 50–120 cycles a tile, and it adds about as many ALU instructions.
- Registers and occupancy (suggestion list item 9, 6 November, compiled code only, no GPU). No shipped fold spills anywhere: 0 bytes of stack on
  every NVIDIA kernel and arch, 0 local loads or stores in the SASS, 0 scratch on every AMD target. The tall fold
  (254-255 registers, 98.7-99.8 KB of shared) is held to one block an SM by both. ptxas
  `--register-usage-level` 8 and 10 leave the sm_89 and sm_86 chunk loop byte for byte the same (0 and 2 make it
  longer). RDNA 3 sits at 240 VGPRs, the 6-wave cap, and its LDS allows 3 blocks anyway. RDNA 4 used 201 VGPRs (213 since #287 item 4) and could
  use 240 at the 6 waves it runs, but with the compiler told so (`-DPEARL_AMD_WAVES=6,6`) the slab loop doesn't change
  and the readout gains a move. Nothing to race.
- RDNA 3's readout as three XOR chains a region (`PEARL_AMD_RDTREE11`, PR #290 item 23, 10 November): −0.18 to
  −0.52% in `hashrate.js` and −0.25 to −0.59% in the fold alone on an RX 7900 XTX (`44d3716f`), behind in every round.
  Removed from the code. The hash on its own queue (`PEARL_AMD_HASHQ`) on RDNA 4: −0.21 to −0.36% on an RX 9060 XT
  (6 of 6 rounds); on RX 7800 XTs level or −0.03 to −0.27%. It is on for gfx1100 only (see the AMD section).
- RDNA 3 (item 13): k 4096 gained under +0.3%. Batch 128 shipped on gfx1100 and gfx1101 (+0.1 to +1.8% a round, and 64
  was slower on both chips); RDNA 4 was level at 128 and keeps 256.
- RDNA 3 load scheduling (#287 item 8, 7 November): the loads pinned two slabs ahead (`PEARL_AMD_PF2PIN`, an empty
  asm with a memory clobber keeps them at the top of the step), alone +0.05%, +0.42% and +0.25% fold alone on three
  PCs and with CRANK and SCHEDB 0.2% behind the same without the pin; CRANK and SCHEDB without PFX, −1.22%, since
  three of a pair's six loads then wait at the barrier with no WMMA to hide them. The set that shipped is above.
- AMD fold: two-lane readout regions and readout v2 (items 2 and 4); start skew (item 14); the LDS store a k-step
  earlier (item 15); `PEARL_AMD_SADDR`; a 64-byte slab; a fold with no LDS; operand fill above 48; 8 waves a SIMD; a
  split slab barrier; the keeping lane by exec mask (below RDSEL alone); batch-width and hash-kernel changes.
- k 8192 pays only on gfx1201 with 56+ CUs (+0.70% on a 9070 XT, 4 November); the 9060 XT is about 6% slower at it.
  HIP counts WGPs on RDNA (32 on a 64-CU card), so the gate takes either count.
- Unsigned or biased int8 operands (suggestion list item 2, 6 November, desk study, no card time). Fill 63 already gives nonnegative A' and
  B' (the noise is triangular on [−63, 63]) with the same bits a u8 mma would see, and it measured level with 48 or
  just below it: −0.23% on the 4090, +0.13% fold alone on a 9070 XT. The four int8 `mma.sync` type pairs compile to
  the same IMMA, registers and schedule on sm_89 and sm_86; the types are two mode bits. Any bias the operand doesn't
  already carry has to come off every accumulator before the XOR readout: one IADD per IMMA, about +8.8% energy an
  IMMA (0.51 against 5.81 nJ), or about −2.6% TH/s. A bit-flip model fitted to the fill sweep puts every bias at or
  below fill 48.
- Folding fewer k-elements a chunk through the noise's pair graph (#290 item 11, 10 November, CPU study, no card
  time). B' column t is fill + E[c][p0(t)] − E[c][p1(t)], so a chunk's 128 pairs are edges of a graph on 128
  vertices, and with fill 0 every cycle is an exact linear dependency: drop one edge, add ±its A' column to the
  cycle's other columns, and the chunk's int32 sums don't change. On real pair tables (`generatePermutationMatrix`,
  24 jobs at k 2048 and 6 at k 8192, 768 chunks) the dependencies number 21.5 a chunk (14–31). But A' is ±63, so a
  kept column can take only one dropped column's A' and stay int8, which makes the usable count a maximum
  edge-disjoint cycle packing. Solved exactly (ILP on the contracted 2-core, with a proven upper bound), it is 8.7 a
  chunk (5–15), and 16 in none of 768 chunks; a job's worst chunk is 5–8. Checked on real noise: 288 chunks gave the
  same sums with A'' within ±123. So nothing on any WMMA or `mma` fold: RDNA 3/4 and RTX 20 need 16 dropped in a
  chunk, Ada and later 32. Only the dp4a fold (GTX 10 and 16) could use it: +6.0% fewer dp4a with a per-chunk trip
  count, +3.2–3.5% with one count a job. That needs A fill 0 (B' fixed, A'' redrawn per stamp), and fill 48 was
  worth +3.6% on the 4090 by power, so the net is near zero on cards nobody rents. With fill 48 kept on B only
  balanced cycles work (3.7 a chunk, +1.8% on dp4a at best), and directed cycles for A fill 48 give 1.4. Dropped.
- Operand fill 64 and higher (research item 3, 10 November, desk study). The spec's `verify_plain_proof`
  (pearl-research-labs/pearl, `zk-pow/src/api/verify.rs`) takes A and B values in [-64, 64], "matching the ZK
  circuit's IRANGE7P1 range check". So 64 is legal and 65 and up are refused by every pool ("Matrix value out of
  range"). The 127 that measured +5.9% (probes/README.md, "Operand values") is out of reach: its gain came from
  saturation, 127 plus any positive noise clamping to 127, and A + noise can't saturate once A is at most 64. The
  fill changes no share's credit: the target and the work a share stands for (tile * k) don't read it. At 64, A'
  sits in [1, 127], so like 63 it never flips the sign bit; the bit-flip model gives both 3.500 toggles a byte
  (48: 3.489) and 64 a popcount of 3.508 against 63's 3.492. 63 measured level with 48 (-0.23% on the 4090), so
  64 is expected level too. Raced on Salad, 10 November (`race` stage, base = fill 48, an A/A twin, turns of 4 s
  for 600 s; builds 3df81e61 and b120b72e on branch item/3, which open `PEARL_OPERAND_FILL` and checkHit.js to
  [-64, 64]):

  | PC | rounds | aa (twin) | fill 63 | fill 64 |
  |---|---|---|---|---|
  | RTX 3080, 104.1 TH/s | 36 | −0.00% ±0.48 | +0.04% ±0.55 | −0.39% ±0.81 |
  | RTX 4080 SUPER, 208.0 TH/s | 37 | −0.01% ±0.02 | −0.21% ±0.02, led 0/37 | −0.29% ±0.02, led 0/37 |
  | RTX 5080, 212.3 TH/s | 35 | −0.00% ±0.01 | −0.03% ±0.03 | −0.02% ±0.04 |

  (± is the 95% interval.) verify-hits passed 400 of 400 for fill 63 and 64 on all three. Fill 64 is level on the
  3080 and 5080 and a clear −0.29% on the 4080 SUPER; the 5080 drew 2 W more at 63 and 64. Fill 48 stays, the
  [-63, 63] limit stays in the code, and nothing above 64 is legal. Dropped.
- Our AMD work carried over to NVIDIA (PR #290 research item 22, 10 November). Every AMD change that gained speed
  (git log on `earn/native/amd`, the AMD sections above), checked against the NVIDIA folds (the tall fold: TMA on
  sm_120, cp.async on sm_89 and sm_86) and the sm_86 SASS of the release's chunk loop (192 IMMA, 40 LDSM, 117 LOP3,
  14 LDGSTS, ~100 other instructions a chunk a warp). The AMD gains came from two things NVIDIA doesn't share:
  on RDNA 4 a VALU next to the WMMAs costs about a WMMA cycle (3.5-5 chained on RDNA 3), while an NVIDIA scheduler
  issues ALU work beside an IMMA that holds the tensor pipe ~16 cycles, so there an instruction costs only energy;
  and AMD's fold started further from its peak (70% of WMMA on RDNA 4 in v0.6.11). By what each would give NVIDIA:
  - Already in the NVIDIA folds, most of them first: the per-tile operand layout (AMD took
    `pearl_materialize16_tiled` from the CUDA fold), loads kept in flight across the stage barrier (PF2, PFX: the
    3-stage cp.async/TMA ring; its waits cost 0.46% a tile, all at the hash seam), one fragment load for two
    k-steps (KPAIR: `ldmatrix.x4` is already 16 bytes a lane, the least a 96x64 warp tile can read), rank at compile
    time (CRANK: no division in the chunk loop), the readout as a tree (RDTREE: 3-input LOP3 XORs), DPP lane moves
    (shuffles and `red.shared.xor`), launch bounds (TID), the tile walk and batch width sized to the L2 (WALK and
    RDNA 3's batch 128: Ampere's band, Ada's and Blackwell's width rules), and the hash inside the fold.
  - Already tried on NVIDIA and dropped: the readout moved into the next chunk (RDLATE: −2.0% in September; tiles 4
    and 5 moved, +265-293 cycles a tile), pinned lane bases (LDS32_PIN: +0.07% by estimate), k 4096 (long k is
    item 21).
  - Not built: fewer address instructions for the cp.async copies (BUFLD). On sm_86 a stage spends about 9 on
    addresses, 6 of them for A's three copies (64-bit adds); about 8 a chunk could go, ~0.3% of a chunk's energy,
    so about +0.1%. It sits in `tile_srcs`, which item 21 is widening for long k, so it waits for that.
  - Dropped by estimate: the hash in its own kernel, AMD's default shape and what GA100 ships
    (`PEARL_TALL_UNFUSED`, +2.2-4.8% there, where the fused hash cost 14% of a tile). On RTX 30/40 the
    round trip is 64 bytes a region (2^19 MACs): on a 3090 about 2% of a batch to read it back, plus more DRAM
    traffic on a card already 18-30% busy, against about 0.65% of TH/s the fused hash costs at the power cap.
    Net about −1.5%.
  - Raced: RDNA 4's tile order (WALK, every other row of supertiles backwards) as the serpentine walk on RTX 50
    (`-DPEARL_FOLD_SERPENTINE=1`; flag only, no spill under ptxas 13.3, 8 more instructions in the fold). Measured
    before only on a 5060 (+2.3% at width 2048, nothing at the 512 it runs). RTX 5090 (`8d8f5bf3`, 480 W cap,
    driver 621.14, CUDA 13.3, width 2048, race mode, 74 rounds of 4 s): aa +0.262% (±0.092, a base-position offset,
    ahead 53/74), serpentine +0.287% (±0.108, 53/74), so +0.03% against the twin: level. Memory busy 4.7 → 4.0%,
    clock +3 MHz over the twin. verify-hits 400 of 400. Nothing to ship. Mid RTX 50 cards run width 1024 or 512,
    where B' fits their L2 like the 5060's 512, so it wasn't raced there. Cost: one 5090 box, 19 minutes.
  So nothing from the AMD side is left to give NVIDIA more than about 0.1%. The NVIDIA gap is the DRAM-free,
  long-k kernel (items 2, 7, 21), which the AMD core doesn't have either.
- The hash in the fold on gfx1201 (#287 item 6): −0.50% on an RX 9070 XT at k 8192 (a68b3382, both scored rounds, copies
  within 0.06%). RDNA 3 hasn't raced it; its hash kernel is a smaller share (0.8% on a 7900 XTX) and its DRAM faster.
- Stale shares (PR #290 item 9, 10 November, pool data only). HeroMiners' lifetime counts for our wallet: 121 stale and
  10 invalid of 83,125 shares (0.15% and 0.01%). Kryptex's worker counts the same day: our 35 workers 13 stale of
  8,387 (0.16%), 0 invalid; one RX 9060 XT box had 4 of 58, the rest 0–2. PeakMiner, SRBMiner and OneZeroMiner had
  0 stale on 255–304 shares each, too few to tell 0.15% from 0. So stale shares cost at most 0.15%. The miner also
  drops a hit for a job the pool has replaced (`pearlMiner.js` `_onHit`) without counting it; a 4090 batch is about
  56 ms against jobs 35–60 s apart, so with two batches queued that is at most about 0.25%, and only if the pool
  would still take the old job, which the 0.15% stale suggests it doesn't. Not worth a change.
- Power steadiness (PR #290 item 9, 10 November, from the 9–10 November bench samples, one every 15 s after the first
  minute, 17 a miner). Every miner holds the power limit: on an RTX 2060, 3060 and 5060 PC the power's standard
  deviation was 0.02–0.94 W for all four miners, ours included, and our clock's 4.7–14.5 MHz against 3.0–15.5 for
  the others. At the same mean power we ran 1822 MHz against 1899–1904 (3060) and 2510 against 2574–2575 (5060), so
  the gap is energy a MAC (items 1, 4, 6, 12 and 13), not a power draw that swings. The 3080 Ti PC (`c09c68c3`)
  swung for SRBMiner and OneZeroMiner (sd 21 and 41 W) and held for ours and PeakMiner's (6 and 0.4 W). Sampled
  every 100 ms for 2 minutes, release v0.6.14 in the app on rig01's RTX 4090 drew 449.3 W (sd 3.5, 420–461) at
  2520–2550 MHz, power-capped in all 1,122 samples. So it doesn't swing under a second either.
- RTX 20 tuning (PR #290 item 9, 10 November, from records). Seven rounds in October (`earn/native/probes/README.md`,
  "RTX 20-series") tried compilers, launch shapes, hand-scheduled SASS, readout and hash forms, k order, a 192x256
  tile and cubin patches; what paid shipped. The 9–10 November run has us at 97–100% (median 98%) of PeakMiner on
  RTX 20, the smallest gap of any NVIDIA series. Item 14 (sm_75 with CUDA 13's ptxas) is the one idea left, and
  October's round 1 found ptxas 12.4–13.4 build the sm_75 fold to the same bytes. Not worth more time now.
- CMP 90HX (PR #290 item 9, 10 November, from records). On Clore on 4 November it ran 72 TH/s at 200 W on 50 SMs at
  about 1480 MHz: about 975 MAC an SM a clock, the same as an RTX 3080 on Salad (113 TH/s, 68 SMs). So its IMMA isn't
  cut down and it runs the RTX 30 fold like any GA102; the RTX 30 items cover it. CMP 170HX is 99% of best (Vast,
  October). Clore also lists about 350 CMP 50HX (TU102) and 40 CMP 40HX (TU106), which we have never run; neither
  is on Vast or Salad, and the business Clore account is empty, so testing them needs funding.
- B200 (PR #290 item 9, 10 November). Still 33% of best (`mma.sync` tops out near 570 TH/s; competitors 1,380 on
  tcgen05). A tcgen05 fold is a new kernel, built and tested on a card that rents for $3 or more an hour, and at
  1,380 TH/s a B200 earns about $1.07 an hour, so nobody rents one to mine. Not worth it until someone mines on
  their own B200s.

- A fold that draws both operands on the SM: built as a body and a probe, correct, slow so far (11 November,
  RTX 4090 rig01, branch `goal/nv-kernel`). `earn/native/src/pearl_fold_draw.cuh` stores neither A' nor B': a tile
  keeps its dense noise factors in shared memory and each warp draws the next k-step's operands while it folds.
  `earn/native/probes/drawprobe.cu` runs it alone, times it in SM cycles and checks it against a CPU fold. It has no
  hash and no host side, so there is no TH/s figure: the pipe figures below are the probe's, at k 8192.

  | Form | CPU check | Pipe use, everything in | Clock, watts | TH/s by count |
  |---|---|---|---|---|
  | 192x256, six m16 tiles a warp | 2,304 of 2,304 regions | 65.3 to 71.0% | 2663 to 2687 MHz, 412 to 448 W | 228 to 251 |
  | 160x256, five tiles a warp | 1,920 of 1,920 | 72.05% | 2685 MHz, 428 W | 253 |
  | 160x256, table reads at least-conflict addresses (timing only, wrong sums) | | 89.63% (94.71% without the stage wait) | | 315 |

  The release fold is 317.3 TH/s on this card and the three competitors 338.6 to 341.9, so both forms lose as
  built. Most of the loss is one thing: a lane reads 8 bytes of the table at a random rank index, and a warp's 32
  reads fall into 8 of shared memory's 32 banks. The stage wait costs 14 points on top while warps differ, the
  readout 3 to 4, the offsets' global loads 2. Not built: tables laid out rank-index-major, so that a warp draws
  one k-pair for 32 row groups and every lane reads its own 8 bytes of one table row. Tables, switches and what
  ptxas had to be led around: `earn/native/probes/README.md`, "The draw fold".
- **The draw fold's third design: the table reads are fixed, and it still loses (11 November, rig01's RTX 4090,
  probe only).** Tables laid out by rank index, sixteen rows to a 16-byte word, so no table read has a bank
  conflict; every warp draws one unit of B and one of A a k-step, two k-steps ahead, so that no warp should wait
  for another. Probe figures, k 8192 (branch `goal/nv-kernel`; no hash or host side, so no mining rate):

  | Stream | CPU check | Pipe use | Clock, watts | TH/s by count |
  |---|---|---|---|---|
  | Draw alone, nothing holding the warps together | n/a | 96.15% (second design: about 79%) | | |
  | Everything, nothing holding the warps together | 0 of 2,560 regions | 92.38% | 2564 MHz, 448 W | 310 |
  | Everything, a barrier a chunk | 232 of 2,560 | 80.98% | 2625 MHz, 450 W | 279 |
  | Everything, a counter wait a chunk, two loop copies | 64 of 2,560 | 73.98% | 2601 MHz, 428 W | 252 |
  | Everything, a barrier a k-step (the form left on the branch) | 2,560 of 2,560 | 73.30% | 2685 MHz, 411 W | 258 |

  The release fold is 317.3 TH/s here, so every form loses, and only the last is correct. The conflict-free
  tables did what they were for. What is left is the hold: the four schedulers of an SM run this loop up to 3%
  apart, ptxas puts the draw's stores anywhere in a k-step (and moves the mma, so nothing in the source pins
  them), and a warp a few mma behind has its stage overwritten. A barrier makes it correct and starts both warps
  of a scheduler at the same place of the same code, where (as the figures read) they are busy at once: a barrier a
  k-step alone costs
  0.8 points, with the draw 18. Tables and the reasoning: `earn/native/probes/README.md`, "The third design".

- **The release fold's loads and stores: none to cut, and what holds its clock is where the copied bytes come from
  (11 November, rig01's RTX 4090, 450 W limit, branch `goal/copies`).** The sixth kernel round read the release
  fold as doing about twice the loads and stores its block needs (0.51 an IMMA), and priced a copy fold with half of
  them at 341 TH/s by count. So the shipped sm_89 loop was counted from its disassembly and its source changed to
  cut what could be cut. Nothing gains, and nothing ships from this.

  The count, for a warp over a k-step (48 IMMA), from nvdisasm of the release's sm_89 cubin (ptxas 12.6). The chunk
  loop is 540 instructions in the listing; a chunk that is not the tile's last runs 470 of them.

  | Step | A k-step | Where it comes from |
  |---|---|---|
  | IMMA | 48 | 6 m16 tiles by 8 n8 tiles |
  | LDSM, a fragment load of 16 bytes a lane | 10 | 6 of A and 4 of B: the least a 96 x 64 warp can load |
  | LDGSTS, a copy of 16 bytes from the L2 to shared memory through no register | 3.5 | 3 of A and 4 of B a stage of two k-steps |
  | LDS that run | 1 | the barrier tests: one FULL and one EMPTY a stage |
  | LDS that never run (`@!PT LDS RZ, [RZ]`) | 7 | pads ptxas adds: 3 before each LDGSTS group, 4 before each ARRIVES and each MEMBAR |
  | ARRIVES.LDGSTSBAR, ATOMS.ARRIVE, MEMBAR.ALL.CTA | 0.5 each | one of each a stage |
  | ATOMS.XOR | 0.75 | the readout's three shared XOR a chunk |
  | LOP3 of the readout | 24.75 | 99 a chunk, the least for 192 sums |
  | Other ALU, branches and NOP | about 22 | copy addresses, barrier parity, the k-step's address XOR, the ring's role moves, one test made four times |

  So the fold already copies the least a 192 x 256 block can: a stage is 192 rows and 256 columns of 64 bytes, 1,792
  units of 16 bytes over 256 threads and two k-steps, 3.5 a warp a k-step. The 0.51 counted the mnemonic LDS, and 28
  of a chunk's 32 are the pads. There is no second copy of an operand, no narrow load, and no noise added on the
  chip to remove.

  Method: `bench.exe` built with `-DPEARL_TALL_CYC`, 45 s a run (25 s for the forms judged by cycles alone), cycles
  a tile from the SM's own counter (ideal 98,304). Clock, watts and the power-cap flag are the last 20 one-second
  nvidia-smi samples of a run. The base ran nine times in six holds: 102,717 to 102,720 cycles a tile, 316.4 to
  317.5 TH/s by the bench's count, 448.8 to 450.0 W, flag on, and 2534 to 2539 MHz in the 45 s runs. "Timing only"
  rows compute wrong results on purpose and cannot ship.

  What each kind of step costs, in the release's own loop:

  | Arm | Kind | TH/s | Cycles a tile | Of the pipe | SM clock | Watts | Cap flag |
  |---|---|---|---|---|---|---|---|
  | The release | real | 316.4 to 317.5 | 102,717 to 102,720 | 95.70% | 2534 to 2539 MHz | 449 | on |
  | The 14 copies a chunk read 16 KB of fixed lines through the L1 (`PEARL_ABLATE_L1COPY`) | timing only | 332.8 | 102,779 | 95.65% | 2685 | 428.0 | off |
  | The same bytes by `ld.global` and `st.shared` through registers, the L1 serving the load | timing only | 330.7 | 103,399 | 95.07% | 2685 | 433.3 | 11 of 20 |
  | The same fixed lines by `cp.async.cg`, so all 128 SMs ask the L2 for the same 16 KB | timing only | 176.0 | 192,995 | 50.94% | 2685 | 331.2 | off |
  | A's copies off after a block's first tile (6 of the 14) | timing only | 321.8 | 104,502 | 94.07% | 2625 | 449.5 | on |
  | B's copies off (8 of the 14) | timing only | 329.5 | 102,535 | 95.87% | 2639 | 448.7 | on |
  | Copies kept, their sources held in the L2 (`PEARL_ABLATE_DRAM`) | timing only | 317.4 | 102,729 | 95.69% | 2544 | 449.0 | on |
  | Readout at the tile's last chunk only (`PEARL_ABLATE_READOUT`) | timing only | 324.4 | 101,107 | 97.23% | 2555 | 449.3 | on |
  | Tile hash off (`PEARL_ABLATE_TRANSCRIPT_HASH`) | timing only | 321.0 | 101,424 | 96.92% | 2544 | 449.3 | on |
  | Ring waits and EMPTY arrivals off (`PEARL_ABLATE_RING`, `_ARRIVE`) | timing only | 318.0 | 102,240 | 96.15% | 2540 | 448.7 | on |
  | Every fragment by a plain 16-byte shared load, not ldmatrix | timing only | 316.6 | 102,763 | 95.66% | 2535 | 449.1 | on |

  The copies' steps are nearly free and their bytes are not. With the same 14 LDGSTS a chunk at the same points,
  served by the SM's own L1, the card runs at the top of its clock range, 2685 MHz, 21 W under the limit with the
  flag off. Served by the L2 it runs about 2537 MHz at the limit. A's copies alone are worth 91 MHz and B's alone
  105. The sixth round's ladder (4 copy units a k-step 2626 to 2633 MHz, 8 units 2565) was read as a price a copy
  step, but its copies hit the SM's cache, as that round noted. With the bytes coming from the L2 a copy unit a
  k-step costs 50 to 60 MHz here. So a copy fold sits near the release's clock however its loop is written, and
  the 341 TH/s by count is not there to be had by copying less: this fold has no copy to drop. A plain step costs
  about 1 MHz: taking away the readout's 25.5 steps a k-step raised the clock 21 MHz while the pipe filled by 1.5
  points, and a point of pipe is worth about 6 MHz the other way (the slower forms below run 2580 MHz at 89%). So
  the 22 other plain steps are worth 20 to 25 MHz together if every one went. ldmatrix costs what a plain shared
  load of the same bytes costs.

  Forms written to compute the right result (ptxas 12.6, sm_89; loop length, registers of 255 and spill from the
  listing and `-Xptxas -v`). Only the first was hit-checked; the others lost before that mattered.

  | Form | Loop, registers, spill | Cycles a tile | Of the pipe | TH/s, clock |
  |---|---|---|---|---|
  | The release | 540, 255, none | 102,717 to 102,720 | 95.70% | 316.4 to 317.5, 2534 to 2539 MHz |
  | K-step 1's lane bases held in registers (`PEARL_TALL_LANE1`): 3 ALU steps fewer | 537, 254, none | 102,676 | 95.74% | 316.8, 2541 MHz |
  | The lane's transcript base held too: 8 fewer | 532, 254, none | 102,962 | 95.48% | 316.4 |
  | The transcript base held alone | 535, 255, none | 102,945 | 95.49% | 315.9 |
  | A's and B's copies through the L1 (`cp.async.ca`) | 540, 255, none | 102,742 | 95.68% | 313.3, 2509 MHz |
  | Copies by `ld.global` and `st.shared` through registers | 519, 255, none | 106,113 | 92.64% | 302.1, 2519 MHz |
  | The EMPTY wait one m16 tile ahead of A's copies | 544, 255, none; ptxas moves the copies up to the wait | 104,650 | 93.94% | 311.9 |
  | The EMPTY wait two tiles ahead | 544, 255, none; the same | 106,292 | 92.48% | 308.4 |
  | The copies' arrival one tile after B's copies | 546, 254, none; ptxas moves B's copies two tiles up | 105,010 | 93.61% | 310.6 |
  | The copies' arrival two tiles after | 545, 255, none | 104,970 | 93.65% | 311.4 |
  | The EMPTY arrival after the stage's last fragment load, before that tile's mma | 541, 253, none; 11 of the readout's LOP3 woven | 107,219 | 91.69% | 306.5 |
  | The EMPTY arrival made in the next stage, after its m16 tile 0 | 567, 255, none; loop rebuilt (below) | 110,060 | 89.32% | 300.9, 2580 MHz |
  | The same after its tile 1 | 567, 255, none; loop rebuilt | 110,403 | 89.04% | 299.9, 2580 MHz |
  | The same after its tile 2 | 567, 255, none; loop rebuilt | 117,495 | 83.67% | 285.9, 2618 MHz |
  | EMPTY as a counted word: `red.shared.add` and a plain load, no MEMBAR and no pads there | 553, 255, none; loop rebuilt | 108,081 | 90.95% | 304.0 |
  | The same with the bases held | 546, 254, none; loop rebuilt | 106,084 | 92.67% | 308.3 |
  | Copy groups issued with no `stage_next` test | 592, 255, spills 80 bytes | not run | | |
  | The chunk loop compiled twice, the tile's first 15 chunks and its last | 255, spills 84 bytes | not run | | |

  One form is ahead by cycles, by 0.04%: the lane bases held. By `hashrate.js` (60 s after 10 s, one run each in
  one hold) it is behind: 317.08 TH/s at 2529 MHz against the release's 317.67 and its twin's 317.67 at 2535 MHz,
  all at 449 W. It is not a gain. The others lose: 0.2 to 14% more cycles a tile, or, for the copies through the
  L1, 29 MHz of clock. What was learned about ptxas and this loop: a cut that takes a test out of the loop's
  control (no `stage_next` test, a chunk loop in two parts) makes it spill. Any form that takes the stage-end
  `MEMBAR.ALL.CTA` away from the chunk's end (the EMPTY arrival made earlier, made later, or made with a plain
  atomic) makes ptxas rebuild the loop on uniform registers with the readout woven into the last 20 IMMA, the form
  the readout round (branch `goal/readout`) found slower: the fence at the seam is part of what keeps the shipped
  schedule. A step moved to an emptier place does not stay there: ptxas hoists the copies to follow a moved wait.
  And a register held to save steps can cost more than the steps (the transcript base). More would need a loop
  whose order ptxas cannot change, which is the hand-written route; by the count above it has about 22 plain steps
  and 7 pads a k-step to win back, and the ring's waits and arrivals together are 0.45 points of the pipe.

  One idea was closed on paper. With A set to 0, a chunk's sum is the dense rows of A's noise against B' pushed
  through A's sparse factor, and A's operand is then 24.6 KB a tile, the same for all sixteen chunks: 40% fewer bytes
  from the L2, which the table prices at about 80 MHz. But the pushed B' sums the B' of every k that lands in one
  of the 128 bins, two on average, and an int8 holds two of them at most; about a third of bins get three or more.
  Splitting those bins takes about 160 to 190 products a chunk in place of 128, by arithmetic only: nothing was
  built.

  Checks. Two switches stay in the source on that branch, which is not merged since nothing gained:
  `PEARL_TALL_LANE1`, off, and the diagnostic `PEARL_ABLATE_L1COPY`. With
  them the default build's cubin is byte-identical to the release's on sm_80, sm_86 and sm_89. `verify-hits.js` on
  the lane-base form: 400 of 400 hits verified (1,395 hits over 177 salts). On sm_86 the same switch cuts the same
  four LOP3 (538 instructions a chunk against 542, 255 registers, no spill); no RTX 30 card ran it. sm_120 builds
  the TMA body, which the switch does not touch; no RTX 50 card ran anything here. Whether a newer ptxas pads the
  copy groups the same way was not checked: the CUDA 13.3 copy is no longer on this PC.

- **The draw fold, fourth round: 73% to 80.5% in the one correct form, and what the barrier costs (11 November,
  rig01's RTX 4090, probe only).** Probe figures, k 8192 (branch `goal/nv-kernel`; no hash or host side, so no
  mining rate). Timing-only rows compute wrong results on purpose.

  | Stream | CPU check | Pipe use | Clock, watts | TH/s by count |
  |---|---|---|---|---|
  | Third design as left (a barrier a k-step) | 2,560 of 2,560 | 72.93% | 2685 MHz, 411 W | 257 |
  | + draw tethered ahead of A-tile loads, readouts of the two copies 12 mma apart (the default now) | 2,560 of 2,560 (k 2048), 1,280 of 1,280 (k 8192) | 80.47% | 2665 MHz, 448 W | 281 |
  | 19 other correct variants (own barriers, lags, places) | all pass | 72.4 to 77.5% | | |
  | Four tiles a warp (128x256) | 2,048 of 2,048 | 68.01% | not measured on the checked build (2685 MHz, 408 W on a variant that fails the check) | about 239 |
  | Draw alone under the barrier / table words from registers / narrow stores / both / one copy drawing | timing only | 81.35% / 89.86% / 85.53% / 88.50% / 88.46% | | |
  | A draw with no stage, by instruction count (7 units a warp a k-step, nothing holding the warps) | timing only | 63.44% | 2436 MHz, 397 W | 203 |

  The release fold is 317.3 TH/s here, so every form still loses. What the round settled: a draw straight into
  the mma's fragment registers needs about 190 instructions a thread a k-step against 54 staged (the bytes have to
  be turned from "rows at one k" to "four k of one row", which `ldmatrix.trans` does for nothing), and its
  stand-in is the slowest stream measured. Under the barrier the draw costs 17.6 points (8.5 table reads, 4 wide
  stores, 10 that stay with registers only) where it costs 3.6 with nothing holding the warps; the cost is there
  with one warp of a scheduler drawing and the other not, and the idle one reaches the barrier no earlier, so it
  is not the two warps being busy at once. It is about fixed a k-step (four tiles a warp are worse than five).
  ptxas's order of one source is worth 4 to 8 points either way (72.4 to 80.5% over harmless variants that pass the CPU check; three that read 65 to 67% were wrong, and the 81.35% build is the unchecked timing one), its spills
  track the loss, and a tool that reorders the compiled loop safely (`probes/drawtools/spread.py`) can move
  almost nothing because ptxas reuses registers across the steps. Tables and reasoning:
  `earn/native/probes/README.md`, "What the barrier costs the draw".
- **Where the release fold's readout goes: every other place loses (11 November, rig01's RTX 4090, 450 W limit,
  branch `goal/readout`).** A probe of the bare stream had priced the readout's one run of 96 XOR3 at 1.3 points of
  the tensor pipe and the same XOR3 woven between the IMMA at 0.3, so the shipped fold (sm_89, the cp.async ring,
  k 2048) was rebuilt with the runs moved. No form gains, and nothing ships from this.

  Method: `bench.exe` built with `-DPEARL_TALL_CYC`, 12 s a run, cycles a tile from the SM's own counter (ideal
  98,304). The base ran ten times in five holds: 102,714 to 102,721. By `hashrate.js` (60 s after 10 s) the base is
  317.46 TH/s and its twin, the same build again, 317.33, both at 2531 MHz and 449 W (76 and 78 °C). Every form's
  clock was 2535 to 2595 MHz at 448 to 450 W (one or two readings a 12 s run). Positions are `PEARL_TALL_RO_AT0/1/2` for the three m16 tile pairs
  (`pearl_config.h`); "999" is where the shipped fold has the run.

  | Form | Cycles a tile | Of the pipe | As ptxas compiled it |
  |---|---|---|---|
  | Shipped: one run after the chunk's last IMMA | 102,714 to 102,721 | 95.70% | 99 LOP3 and 3 shared XOR in one gap; IMMA in source order; A reused on 168 of 192 IMMA |
  | Readout in the tile's last chunk only (timing only, wrong results on purpose) | 101,103 | 97.23% | so the readout costs 1.57 points |
  | Pair 2 after the next chunk's A copy group (999/999/403) | 103,386 | 95.08% | 40 LOP3 in the copy group's gap after IMMA 31 |
  | Pair 0 woven, the rest shipped (23/999/999) | 104,136 | 94.40% | pair 0 at IMMA 166 to 182, sums 14 to 30 IMMA old; tiles interleaved there; A reuse 150 |
  | All moved: 23/24/27 | 104,368 | 94.19% | pair 0 woven; pair 1 a run of 35 after IMMA 7; pair 2 woven at IMMA 15 to 26 |
  | All moved: 23/24/25 | 105,183 | 93.46% | pair 0 woven; pairs 1 and 2 one run of 66 after IMMA 7 |
  | Pair 2 woven, the rest shipped (999/999/27) | 105,447 | 93.23% | |
  | Pairs 1 and 2 deferred to the next chunk's top (999/200/200) | 105,497 | 93.18% | one run of 66 after IMMA 7 |
  | 23/23/27 | 105,497 | 93.18% | ptxas rebuilds the loop: uniform registers, 579 instructions for 540 |
  | 21/24/25 | 106,792 | 92.05% | pair 0 lifted to IMMA 149 to 159, sums 0 to 9 IMMA old |
  | Pair 2 as a run after IMMA 7, the rest shipped (999/999/24) | 107,066 | 91.82% | a run of 35 in mid-stream |
  | Each run after its tiles' last IMMA (19/21/23) | 108,879 | 90.29% | loop rebuilt, IMMA of two k-steps interleaved |
  | Hand-woven (form 2): one chain a word, 8 XOR3 before each of four tile passes | 110,116 | 89.27% | two LOP3 after every IMMA, sums 15 to 35 IMMA old; A reuse 132 |
  | 21/23/23 | 110,674 | 88.82% | loop rebuilt |
  | All at the chunk's end, before the EMPTY arrival (23/23/23) | 111,111 | 88.47% | loop rebuilt |

  The shared XOR, priced with three timing-only arms (wrong transcripts on purpose, not kept in the source): a plain
  store to a word of the thread's own 102,662; the shared XOR with no word and no bank shared 102,583; the shared
  XOR as shipped but on eight banks instead of two 102,854 and 102,858. So four lanes sharing a word costs at most 0.13 point,
  and a proof tile of one thread's own sums (the competitors' shape) would save that and no XOR3.

  What the compiled loops showed:
  - The chunk's end is the one place where every fragment register is dead (16 of B, up to 12 of A), so the run's
    temporaries are free there. Anywhere else they take the registers ptxas uses to load A fragments 8 to 24 IMMA
    ahead and to keep A in the operand reuse cache.
  - ptxas does not leave a run where the source puts it. It lifts it as early as its inputs allow, across a copy
    group's branch too, and weaves it two to four after an IMMA; in seven forms it also re-sorted the IMMA or
    moved loop values into uniform registers.
  - A weave that is clean by the probe's rules (two XOR3 an IMMA, sums at least 15 IMMA old, tile order kept) is the
    worst: 2.3 cycles of the pipe an XOR3, against 0.5 in the run at the chunk's end and 0.1 in the probe. The real
    fold's warps are tied by the ring and its stream has a branch every 32 IMMA; the probe's are neither.
  - The fence the TMA build uses to hold ptxas (`griddepcontrol.launch_dependents`) does not assemble for sm_89.

  Checked: the default build is byte for byte what it was on sm_80, sm_86 and sm_89 (sm_120 was not compiled:
  no CUDA 12.8 or 13 on this PC). `verify-hits.js` 400 of 400 on 23/24/27. `PEARL_HITLIST` over six batches: the
  shipped fold's hit list from 23/23/23, 23/24/27, form 2 and the five forms with a pair at 999. Forms 23/24/27
  and form 2 compile for sm_86 with no spill; no RTX 30 card ran them. Not verified: the other six forms' hits, any
  second card, and why an XOR3 in mid-stream costs ten times the probe's figure.

- **The draw fold, fifth round: its loop written by hand in machine code, 81.7% to 96% and correct (11 November,
  rig01's RTX 4090, probe only).** ptxas is out of the chunk loop: `probes/hand89/` assembles a loop with its own
  registers, stall counts and scoreboards and puts it into the cubin in place of the compiled loop function; the
  probe's CPU check judges it. Probe figures, k 8192, 64 tiles a block, clock and watts over 25 s at the 450 W
  limit; by count is pipe use x clock x 0.13107. No hash or host side yet, so none of this is a mining rate.
  Timing-only rows compute wrong results on purpose.

  | Stream | CPU check | Pipe use | Clock, watts | TH/s by count |
  |---|---|---|---|---|
  | The fold's loop as ptxas compiles it | 2,560 of 2,560 | 81.73% | 2669 MHz, 439 W | 286 |
  | By hand, first placement | 2,560 of 2,560 | 85.00% | | |
  | + unit B from gap 20, plain steps' yield flag off | 2,560 of 2,560 | 91.26% | 2587 MHz, 440 W | 309 |
  | + IMMA stall 5 when plain steps follow | 2,560 of 2,560 | 95.18 to 95.23% | 2550 to 2558 MHz, 430 to 437 W | 318 to 319 |
  | + table reads and one-register steps first, warps 4-7's readout 4 IMMA later, read scoreboards on the readout's atomics (`fold.py`'s default) | 20,480 of 20,480 (k 2048), 2,560 of 2,560 (k 8192) | 96.02% | 2548 MHz, 439 W | 321 |
  | The same without the barrier | timing only | 97.49% | | |
  | Stall counts alone changed in the compiled loop (`restall.py`) | 2,560 of 2,560 | 81.79 to 82.04% | 2653 MHz, 435 W | 285 |
  | Gather form's mix, 192 sums, no bank collisions (`gather.py`) | timing only | 97.74% | 2581 MHz, 438 W | 331 |
  | the same, one or two two-way collisions in every matrix | timing only | 96.77% | | |
  | the same, rank indices by chance | timing only | 91.15% | 2611 MHz, 449 W | 312 |

  The release fold is 317.3 TH/s here, so the hand loop is level with it by count and not ahead: no kernel yet.
  What the round found. (1) The rule "a scheduler stays on a warp until it stalls, so a clump costs its length
  minus 16" does not hold: one-register adds cost nothing clumped or spread (99.5 to 99.7%). (2) What costs is a
  plain step that reads two or more registers issued 1 to 4 cycles behind its own warp's IMMA: three such steps
  after every IMMA hold 88.7% with the IMMA at stall 1 and 99.7% at stall 5, and up to six are free there. ptxas
  writes stall 1. That one change took the real loop from 91.3% to 95.2%. (3) Changing only the stall counts of
  ptxas's own loop does nothing (82.0%): its loss is the order. (4) A load keeps reading its address register
  while the memory unit holds it: two of 45 placements computed wrong sums until the registers the chunk's end
  overwrites were held by read scoreboards, and one rare failure (8 of 10,240 regions, once) is not explained.
  (5) The maths agent's gather form (no stage, no barrier) holds 97.7% as a timing stream with the real register
  count, but only while a matrix's rows collide in the banks two at a time; by chance they cost 6.6 points.
  Tables: `earn/native/probes/README.md`, "The loop written by hand".

- **The draw fold, sixth round: the gather fold and a direct fold built for real, neither beats the release, and
  the reason is measured (11 November, rig01's RTX 4090, branch `goal/nv-kernel`).** No mining rate here: probe
  figures at k 8192, "by count" = pipe use x clock x 0.13107, the release mines 317.3 TH/s (2531 MHz, 449 W).
  (1) The gather fold (each warp builds its own fragments with `ldmatrix.trans` on static tables; no stage, no
  barrier) is correct as compiled and as a loop written by hand: 16,384 of 16,384 lane transcripts over 64 tiles
  against a CPU fold, in three orders of k. By hand it holds 88.9% of the pipe in the definition's order of k
  (2.75 rows of a matrix's 8 collide in the banks), 95.4% with k ordered for B alone, 96.7% ordered for both sides
  (1.0 collide), 97.4% with none (timing only). Over 60 s: 96.7% at 2520 MHz, 436 W, 319 by count; none collide
  321. (2) The fifth round's "11 to 20 W under the limit" is not spare power. The same loop at a 480 W limit runs
  2522 MHz and 432 W; with every operand byte constant it draws 318 W and still runs 2535 MHz. The clock has a
  ceiling that falls with the loop's memory and ALU steps: IMMA alone 2715 MHz, with the readout 2700, with the
  gathers' twenty LDSM a k-step 2638, the whole loop 2535 (all with constant operands, 215 to 319 W). (3) A fold
  with no draw, timing only: ten LDSM a k-step and the readout hold 99.4% at 2685 MHz and 425 W, 350 by count;
  four copy units a k-step more (a 16-byte global load and a 16-byte shared store each, about the least a
  192 x 256 block can copy) 99.2% at 2626 to 2633 MHz, 341 to 342; eight 2565 MHz, 333; the release's count of
  plain steps 2565 to 2571 MHz at 98.0%, 330. The four-copy row lands where the competitors measure (338 to 342
  TH/s at 2600 to 2625 MHz): that is an inference from our own streams, not something established about their
  programs. This round also said the release copies about twice that; that count was wrong (see the correction
  at the end of this entry). (4) A direct fold (A' and B'
  ready in global memory in the fragments' own layout, one 16-byte load a fragment, no stage and no barrier) is
  correct compiled and by hand (16,384 of 16,384, nine runs); by hand with each fragment of the next k-step
  loaded ahead it holds 99.0% of the pipe but runs 2476 MHz at 449 W, 321 by count, and 73.9% at 2387 MHz when
  the operands (k 65536, 370 MB) do not fit the card's cache: global loads cost the clock more than LDSM do.
  (5) So no draw on the chip clears +3%: gather 319, staged 319 (96.0% at 2536 MHz over 60 s), direct 321. What
  is left, this round concluded, is the release's own form with half its copies, written by hand. Nothing ships
  from this round. Tables: `earn/native/probes/README.md`, "The gather fold, the direct fold, and what sets the
  clock". **Correction, the same day (16:20 UTC, from the shipped loop's disassembly):** the release has no half
  to cut. A warp's k-step is 48 IMMA, 10 LDSM and 3.5 copy units (`LDGSTS.E.BYPASS.128`, 16 bytes from global to
  shared memory, no register), which is the least a 192 x 256 block can copy. The "0.51 loads and stores an IMMA"
  behind "twice" had counted 28 `@!PT LDS RZ, [RZ]` a chunk: pads ptxas places before each copy group and
  barrier, behind a predicate that is never true. So the release makes as many copies as the 341-by-count timing
  stream and still runs 2531 MHz against that stream's 2626. What differs is where the copied bytes come from: the
  stream's loads hit the SM's own cache, the release's come from L2 or DRAM. What a copy costs by its source is
  the next measurement.

- **The copy fold by hand, seventh round: correct, 98% of the pipe, and under the release, because a copied byte
  costs what it costs (11 November, rig01's RTX 4090, branch `goal/nv-kernel`).** No mining rate here: probe
  figures, "by count" = pipe use x clock x 0.13107, 450 W, 60 s with long launches; the release mines 317.3 TH/s.
  The sixth round's "release fold with half its copies, 341 by count" does not exist: the release already copies
  the least a 192 x 256 block can (14 LDGSTS a warp a chunk), and the 341 priced copies from the SM's own cache.
  (1) Built: the staged fold with real copies in place of the draw (`hand89/cfold_h.cuh`, `cprobe.cu`), A' and B'
  noised on the CPU in the stage's layout, and its chunk loop written by hand (`cfold.py`), at 160 x 256 and at the
  release's 192 x 256, with the copy as a global load and a shared store or as the release's LDGSTS. Correct in
  every form: 10,240 of 10,240 regions (ten hand forms) and 12,288 of 12,288 (192 x 256), and at k 8192 and
  k 65536. (2) What a copy costs is where its bytes come from (160 x 256, load plus store): no copies 98.7% of the
  pipe at 2700 MHz and 381 W; from the SM's cache 98.4% at 2640 MHz (340 by count); from the L2 (43 MB) 98.2% at
  2431 MHz (313); from DRAM (k 65536, 337 MB) 95.9% at 2325 MHz (292). (3) The release's own form by hand
  (192 x 256, LDGSTS): 98.1% of the pipe at k 65536 but 2321 MHz, 299 by count; at the release's working set
  (k 2048, 11 MB in the L2) 97.0% at 2432 MHz, 309. Under the release each time. (4) Why: the release mines at
  k 2048 (`PEARL_FOLD_K`), so its 95.7% includes a tile's hash and hand-off every 16 chunks and its operands stay
  in the L2; a copy fold at k 65536 reads DRAM. And at the power limit the rate is watts over joules a multiply:
  the hand loop's base draws what ptxas's does (410 W at 2685 MHz with no copies; the release with its copies
  off 406 W at 2685 MHz), so more pipe at the same joules only lowers the clock. (5) The definition allows no
  ready-made operand table an SM could hold: a byte is fill + U[r][p0] - U[r][p1], U is 128 bytes a row, a table
  by the pair is 127 times that. So a fold either brings 37.3 bytes an IMMA into the SM or makes them there.
  The copy route is closed. (6) What making them there would cost at the least (a timing-only stream with the
  staged draw's mix: a warp a k-step 7 shared loads, 3.5 shared stores, 14 IADD3, 14 PRMT, 7 adds, 2 offsets'
  loads; nothing from outside the SM): 96.7% of the pipe at 2596 MHz and 449 W at 192 x 256, 329 by count (332 with its best placement);
  326 at four stores a warp a k-step; 323 and 320 at 160 x 256 with 3.5 and 4, where the stream stands for a
  real fold and lands within 3 MHz and 0.3 points of it. So a draw fold at the least draw is the one form
  priced above the bar of 327; it is not built. The fifth round's real draw fold (160 x 256, four stores
  a warp a k-step) measures 96.0% at 2535 MHz, 319, in the same holds. (7) `fold.py` now keeps the sixth round's
  first rule (every reader of a loaded register waits on the load itself) at no cost, 96.07% either way; its one
  old failure did not come back in four more wide checks and could not have come from that breach (a draw fault
  spoils 32 regions or more, the failure was 8). Nothing ships from this round.
  Tables: `earn/native/probes/README.md`, "The copy fold by hand, and what a copied byte costs".
- **The draw fold, eighth round: a price list of steps, and a fold priced from it that beats the release by count
  (11 November, rig01's RTX 4090, branch `goal/nv-draw`).** No mining rate here: probe figures, "by count" =
  pipe use x clock x 0.13107; the release mines 317.7 TH/s (2535 MHz, 449 W). (1) The price of one more step a
  warp a k-step, all in one loop with ready fragments and operand-like bytes, 60 s at the 450 W limit (timing
  only): the clock under the card's cap is 419,840 / (152 + units), at most 2685 MHz, where a 16-byte shared
  load or an LDSM is one unit (16 MHz near 2600), a 16-byte shared store 0.8, a 4-byte load 0.36, a 16-byte
  global load 1.15 from the SM's cache and 2.9 from the L2, a PRMT or IADD3 about a tenth, and the first four
  units are free (2, 4, 8, 12, 16 loads: 2685, 2685, 2624, 2560, 2504 MHz). Watts fall as loads are added (443
  to 398): the cap counts steps, not joules. ALU steps cost pipe instead, 0.04 points each when few. Marking the
  IMMA's operand for reuse does nothing. (2) So a draw should make each fragment once. The "rotation fold" has
  every warp take a chunk's four k-steps in its own order; at the first turn the four warps of a row band fold
  four different k-steps, each makes that k-step's A fragments by gathers and stores them, and at the other
  three turns loads what another stored: 17 memory steps and 44 ALU steps a warp a k-step where the gather fold
  has 20 and 80. Priced at 333 before it was built. (3) Built by hand (`hand89/rfold.py`): with no barrier it
  holds 97.2% at 2600 MHz and is wrong (12 to 54 of 64 tiles fail: the two warps of a scheduler run 2.8% apart
  and nothing orders a load after its store). A barrier at every k-step's end over all eight warps makes it
  correct at 88.2%; one barrier for each row band 93.8%; B's gathers issued apart 96.0%; the subtract a gap
  later, the yield flag off and an order search that weighs B's rows 97.05% (k 8192). The cost was bank
  collisions among warps a barrier had put in step, not the barrier itself. (4) Over 60 s at k 65536: 97.48% of
  the pipe at 2594 MHz and 449.1 W, 331.4 by count, CPU check 512 of 512 (and 2,048 of 2,048 over 8 tiles; at
  k 2048 16,384 of 16,384 in three orders of k). The gather fold in the same hold: 96.68% at 2520 MHz, 319.3.
  (5) A job's own noise from the repo's port of the protocol, folded on the card, every hit through a port of
  the pool's check: 1,049 of 1,049, 1,004 of 1,004 and 1,068 of 1,068 at k 2048 (64 tiles each), 249 of 249 at
  k 8192. With the hash on the card (one keyed BLAKE3 a thread a tile, `-DGP_HASH=1`) the card's own list is
  the list made on the PC in the checked tiles, and 400 of 400 of its other hits pass, at k 2048 and at k 8192;
  the kernel with the hash runs 97.48% at 2593 MHz. (6) It is a probe: no share built or sent, no Merkle proof,
  no host side, and the loop is written by hand, so the product would have to launch a patched cubin. 331 by
  count is 4% over the release and 2 to 3% under the other miners (338.6 to 341.9). Nothing ships from this
  round. Tables: `earn/native/probes/README.md`, "The price of a step, and the rotation fold".

### rig01 RTX 4090, 11 November: the rotation fold as a miner, behind a switch, raced against the release fold (branch `goal/nv-mine`)

- **What ran.** The rotation fold of the note above, built into the CUDA core as a second sm_89 fold that is
  off unless `PEARL_FOLD_ROT=1` is set (`earn/native/README.md`). One addon for both arms, built on rig01 with
  CUDA 12.6; the release kernels' sources are unchanged. rig01's RTX 4090 at the 450 W limit. It mines k 65536
  with a tile of 12 rows by 16 columns.
- **Correct.** `verify-hits.js`: 400 of 400 hits verified, twice (21 and 13 salts). A stand-in pool that checks
  every share as the real one does: 52 of 52 accepted. Kryptex: 24 shares sent in two 330 s legs, 24 accepted,
  0 rejected (the release fold's two legs: 17 accepted, 0 rejected).
- **The race**, release and rotation alternating, 5 minutes counted a leg, twice each, two ways.

  | | Release fold | Rotation fold | Gain |
  |---|---|---|---|
  | On Kryptex, the miner's own count | 317.12 and 317.26 TH/s, 2531 MHz, 449 W | 330.56 and 330.36 TH/s, 2594 MHz, 449 W | +4.2% |
  | `hashrate.js` | 317.42 and 317.29 TH/s, 2531 MHz, 449 W | 330.95 and 330.96 TH/s, 2596 MHz, 449 W | +4.3% |
  | Kryptex shares accepted / rejected | 8 / 0 and 9 / 0 | 13 / 0 and 11 / 0 | |

  Twins of an arm are 0.06% apart at most. The gain is in the clock (+2.5%) and in the tensor pipe (95.6% to
  97.2%). Accepted shares cannot resolve it: at Kryptex's target a 5-minute leg holds about 10 shares, so a
  leg's rate by shares has a standard deviation of a third (release 210 and 270 TH/s, rotation 360 and 330).
- **Against the others.** 330.5 as a mining rate is 96.7 to 97.6% of the other miners' 338.6 to 341.9 on this
  card (measured earlier, not rerun here); the release is 92.8 to 93.7%.
- **Not shipped.** The switch is off by default and no build makes the fold's cubin. Its loop is written by
  hand and assembled by our own assembler, CI does not build it, only this one RTX 4090 has run it, and RTX 30
  and RTX 50 have no such fold. Table, checks and costs: `earn/native/probes/README.md`, "The rotation fold in
  the miner".

### rig01 RTX 4090, 11 to 12 November: the rotation fold as an opt-in a release carries, and a faster loop (branch `goal/nv-next`)

- **What changed.** The rotation fold of the note above is no longer a hand-made build. A release now carries
  it, off by default: `--rotation-fold` on the CLI, `LLMJOB_ROTATION_FOLD=1` in the desktop app's environment.
  CI builds its cubin with CUDA 12.8 and refuses one whose checks or hash differ; the installer, the AppImage,
  the CLI and the HiveOS package carry the file beside the core. And its loop is faster than the one raced
  above. rig01's RTX 4090 at the 450 W limit throughout.
- **The loop.** Three changes, each priced on the card first, each correct by the probe's CPU check (16,384 of
  16,384 at k 2048 in three orders of k, 2,048 of 2,048 at k 8192 and at k 65536) and by `verify-hits.js` (400
  of 400): a gather's subtract and regroup done as late as their readers allow, offsets records of 8 bytes in
  place of 16, and A's gathers sent out early with the scoreboards chosen so that no step waits for another
  gather's load, the sixth of a turn's B gathers given a quad of its own. Pipe 97.48% to 98.96%, clock 2594 to
  2596 MHz.
- **The race** (`hashrate.js`, the miner's own count, 120 s after 10 s, alternating, twins within 0.13%).

  | Loop | TH/s | Clock, watts |
  |---|---|---|
  | the ninth round's (the note above) | 330.89 and 330.94 | 2594 to 2595 MHz, 449 W |
  | subtract and regroup late | 332.87 and 332.87 | 2595 to 2596 MHz, 449 W |
  | and records of 8 bytes | 333.63 and 333.56 | 2598 to 2600 MHz, 449 W |
  | and A's gathers early, the scoreboards chosen (`sbo=5`) | 334.89 and 334.76; 334.59 and 334.71 | 2595 to 2597 MHz, 449 W |
  | and a sixth quad, `sbo=7`: the loop a release now builds | 336.18 and 336.00 | 2596 to 2597 MHz, 449 W |

  Three holds; each new loop was raced against the one before it. The release fold by the same probe: 317.4.
- **As CI built it, on a pool.** CI's own Windows core and cubin (native core run 43130613574, commit
  f58e1633), unpacked into one folder as the installer has them, nothing naming the cubin. `verify-hits.js`: 739
  hits, 400 of 400 verified, 19 salts. Then Kryptex (prl-us.kryptex.network:7048) through the CLI, three legs of
  300 s, the miner's own count from 40 s on:

  | Leg | TH/s | Clock, watts, temp | Shares accepted / rejected |
  |---|---|---|---|
  | `--rotation-fold` | 335.77 | 2599 MHz, 449.2 W, 76 C | 11 / 0 |
  | no flag (the release fold) | 317.44 | 2533 MHz, 449.1 W, 77 C | 10 / 0 |
  | `--rotation-fold` | 335.79 | 2598 MHz, 449.4 W, 77 C | 10 / 0 |

  **335.8 TH/s against 317.4: +5.8%, 21 shares accepted and none rejected.** Each fold leg began with the
  card's check of 24 hits, which passed.
- **The safety net.** A card that opens on the fold has 24 hits recomputed on the CPU before it mines the pool's
  job with it (8 s on this PC), and every share after. A cubin made wrong on purpose (the loop without its
  stores) failed that check within a second, sent nothing, and the card went on at 317.1 TH/s on the release
  fold with 25 shares accepted and none rejected. A cubin cut to 8 KB crashed the driver and the whole miner
  until the host checked the file itself; the host now holds the file beside the core to a list of hashes.
- **B shared, priced and not built.** Sharing B across the two row bands was the next design on the list at
  about 3 TH/s. As a timing-only stream of the real loop it holds 96.4 to 96.7% of the pipe at 2620 to 2625
  MHz, 331.5 to 332.1 by count against 331.4 for the loop it would replace; with the one barrier for all eight
  warps that it needs and the loads sent early, 326.8 to 328.4. The clock comes as priced and the pipe goes:
  the warp that makes B for two has four more stores a turn and is the slow one of its scheduler's pair.
- **Against the others.** 335.8 as a mining rate is 98.2 to 99.2% of the other miners' 338.6 to 341.9 on this
  card (measured earlier, not rerun here); the release fold is 92.8 to 93.7%. The gap that is left is clock:
  the fold runs at 2596 to 2599 MHz, and its pipe (98.96%) has a tenth of a point left to find.
- **Not verified.** One RTX 4090 on Windows is the only card that has run the fold. No other RTX 40 card, and no
  Linux rig: there the lookup of the cubin was run in CI with a stand-in addon (found in every case), not on a
  card. Tables: `earn/native/probes/README.md`, "Where the rotation fold's time goes".

## Community reports (not our measurements)

Figures people posted from their own rigs. We did not run them, the method and length of each run are unknown, and
none is checked against a pool. They change no row above. They are kept because they cover hardware we have no
figure for, and each says what would make it a real row.

| Date | Who | Hardware | What they reported | Notes |
|---|---|---|---|---|
| 11 Nov 2026 | jarvcat (Discord) | RX 9070 (not the XT), Windows or Linux not stated | Ours: 75 TH/s at 153 W with power turned down 30%; 95 TH/s at 220 W at stock | The first figure for the non-XT 9070. By our arithmetic 0.49 TH/s a watt turned down against 0.43 at stock. No other miner was run. A row needs SRBMiner on the same card at the same setting. |
| 11 Nov 2026 | 306Miner (Discord) | HiveOS rig: 4x RTX 3070, 2x CMP 50HX, RTX 3060, RTX 4070, on a 2-core Celeron G1840; every card at a locked core clock and a 100 W power limit setting | OneZeroMiner: 74.8 to 74.9 a 3070 at 101 W, 81.0 and 81.1 a CMP 50HX at 156 and 185 W, 45.4 the 3060 at 67 W, 117.2 the 4070 at 113 W. Ours (v0.6.15, with `--mode auto --llm-max-instances 1`): 31.8 a 3070 at 77 W, 87.7 a CMP 50HX at 206 and 221 W, 4.5 the 3060, 92.3 the 4070 at 115 W; load average 6.8, and HiveOS's watchdog rebooted the rig every four minutes | Not a fair race: ours also ran one LLM server, and its memory clock read 810 against 405 under OneZeroMiner. It is the report that led to the CPU work (`earn/README.md`, "How much CPU a card takes", and the CPU record at the end of the RTX 40-series section): v0.6.15 spins one CPU core a card. The first CMP 50HX figures we have. A row needs a mining-only run on a build with that fix. |

## Keeping this file current

- A change that re-measures a host here updates its row in the same PR. A number only in chat or a PR doesn't count.
- Only a run of our miner on the row's own machine replaces "Ours" and "% of best". Competitor figures stay until the
  next full rerun; a new host runs every miner.
- A run on another host or by another method doesn't change a row: record it as a new table or a research note.
- Before tagging a release, check that the intro and the Summary name the right release.
