# Benchmarks

Our miner against PeakMiner and SRBMiner, 5 minutes each on the same card and pool, and since 9 November against
OneZeroMiner too, a top-5 Pearl miner by downloads (NVIDIA only). WildRig Multi isn't measured: it mines NVIDIA
cards through OpenCL, which Salad's WSL2 containers don't have. "% of best" is our rate over the fastest competitor's. Vast rows ran v0.5.14 on 10–11 October; Salad rows ran v0.6.9 (October) and v0.6.13 (9–10 November) on home PCs, so compare "% of
best" between them, not TH/s. The old file with every run's notes:
[benchmark.md at 16192d68](https://github.com/super3/llmjob/blob/16192d68f40b9c7ba11048e1128e7ba2c5509dca/benchmark.md).

## Summary

| Cards | Latest Salad run | Note |
|---|---|---|
| RTX 20-series | 97–100%, median 98% (9–10 Nov, v0.6.13) | PeakMiner fastest on every PC |
| RTX 30-series | 85–95%, median 93% (9–10 Nov, v0.6.13) | OneZeroMiner fastest on most PCs |
| RTX 40-series | 87–96%, median 93% (9–10 Nov, v0.6.13) | Our gap widens at lower power limits |
| RTX 50-series | 86–97%, median 94% (9–10 Nov, v0.6.13) | 5090 trails most (86–87%) |
| GTX 16-series | 96–104% (17 Oct, v0.6.0) | 0.7–1.3 TH/s; not worth mining |
| AMD Radeon | 92–143%, median 103% against SRBMiner 3.8.2 (9–10 Nov, v0.6.13) | RDNA 4 111–143%; 7900 XTX 92–94% in that run, 102% on two PCs in the 10 Nov duel with the hash on its own queue |
| Intel Arc | Not Salad: one rented rig of six Arc A750s. 124% against IAminer 1.0.0 on one card, 128% on six (10 Nov, PR #290's SYCL core, in no release) | 16.9 TH/s a card, 15% of the 113 T-MAC/s its matrix engines do alone; 10.2 on HiveOS's own 2024 driver. SRBMiner and WildRig mine nothing on Arc; the pool rejected both shares IAminer found |

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

## A100-class (sm_80)

Vast, 10 October.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| A100 PCIe 40 GB | Japan (250 W) | 146405 | 211.1 | 213.7 | 215.3 | 98% |
| A100 SXM4 40 GB | New York (400 W) | 152837 | 245.5 | 251.8 | 253.4 | 97% |
| A100 SXM4 40 GB | Mississippi (400 W) | 117301 | 246.6 | 252.8 | 253.8 | 97% |
| A30 24 GB | Australia (144 W) | 145742 | 100.5 | 90.1 | 90.8 | 111% |
| CMP 170HX | Georgia (250 W) | 147823 | 163.4 | 165.5 | 165.3 | 99% |

The A30's host lowered the power limit during the competitors' runs; at equal power ours is about 107%.

## Hopper (sm_90a)

Vast, 10 October.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| H100 SXM 80 GB | Czechia (575 W) | 148671 | 682.8 | 697.7 | 688.1 | 98% |
| H100 NVL 94 GB | Japan (400 W) | 29785 | 536.5 | 548.6 | 539.1 | 98% |
| H100 PCIe 80 GB | Czechia (350 W) | 147981 | 484.9 | 448.3 | 488.0 | 99% |
| H200 141 GB | — | — | — | — | — | not run: over budget |
| H200 NVL 141 GB | — | — | — | — | — | not run: over budget |

Last H200 figures (8–9 October): 99.9% on the H200 141 GB and 101% on the H200 NVL.

## Ada workstation and data-center cards (sm_89)

Vast, 10 October.

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

## RTX PRO Blackwell (sm_120)

Vast, 10 October, on the CUDA 13 core.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| RTX PRO 6000 Workstation | Mississippi (600 W) | 148552 | 398.7 | 389.1 | 419.2 | 95% |
| RTX PRO 6000 Server | Australia (600 W) | 139338 | 435.1 | 432.4 | 435.9 | 99.8% |
| RTX PRO 6000 Max-Q | Australia (300 W) | 142021 | 308.0 | 302.9 | 306.7 | 100% |
| RTX PRO 5000 48 GB | California (250 W) | 150611 | 211.5 | 208.3 | 214.8 | 98% |
| RTX PRO 4500 | Arizona (200 W) | 148915 | 175.7 | 171.2 | 176.0 | 99.8% |
| RTX PRO 4000 | France (145 W) | 147490 | 127.9 | 126.5 | 128.8 | 99% |
| RTX 6000D | Czechia (550 W) | 149788 | 144.6 | 144.6 | 144.6 | 100% |

## Ampere workstation and data-center cards (sm_86)

Vast, 10 October.

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

SRBMiner holds a higher clock than ours at the same power on the A2000s and A4000s.

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
| B200 180 GB | Oregon (1000 W) | 96942 | 457.9 | 1380.6 | 1394.7 | 33% |

We mine it on `mma.sync`, which tops out near 568 TH/s here; the competitors use tcgen05.

## Tesla T4 (sm_75)

Vast, 10 October.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| Tesla T4 16 GB | Czechia (70 W) | 28909 | 30.1 | 32.5 | 33.1 | 91% |

## Tesla V100 (Volta, sm_70)

Vast, 10 November (PR #290 item 9). PeakMiner 2.20.7, SRBMiner 3.8.2 and OneZeroMiner 1.8.5 ran as bench.sh runs
them, on Kryptex. No release has V100 code yet, so "Ours" is this branch's core built for sm_70 alone (the dp4a fold
GTX 10 and GTX 16 cards mine on), run by `earn/native/probes/turing/pool.sh`. Not a release row.

| Card | Host | Machine ID | Ours (dp4a, this branch) | PeakMiner | SRBMiner | OneZeroMiner | % of best |
|---|---|---|---|---|---|---|---|
| Tesla V100 SXM2 32 GB | Washington (300 W) | 155858 | 29.2 | exits | 40.0 | 50.6 | 58% |

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

| Build | TH/s | vs dp4a | SM clock | Power | verify-hits |
|---|---|---|---|---|---|
| dp4a fold (item 9) | 29.71 | — | 1530 MHz | 291 W | 248/248 (item 9) |
| wmma 16x16x16 | 35.30 | +19.2% | 1523 MHz | 295 W | 314/314 |
| m8n8k4, k outermost | 40.45 | +36.0% | 1500 MHz | 299 W | 359/359 |
| + no bank conflicts on A' | 41.45 | +39.6% | 1485 MHz | 299 W | 232/232 |
| + accumulators start at 2^23 (shipped) | 43.84 | +47.9% | 1477 MHz | 298 W | 400 verified, 0 failed |

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

## GTX 10-series (Pascal, sm_61, on Salad)

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | % of best |
|---|---|---|---|---|---|---|
| GTX 1080 Ti | Salad | not run | not run | 0 | 0 | — |
| GTX 1080 | Salad (180 W) | c868132d | not run | 0 | 0 | — |
| GTX 1070 | Salad | not run | not run | 0 | 0 | — |
| GTX 1060 | Salad (6 GB) | not run | not run | 0 | 0 | — |
| GTX 1050 Ti | Salad (4 GB) | not run | not run | 0 | 0 | — |

The competitors mine nothing on Pascal; our v0.6.8 Pascal core mined 17.8 TH/s on a GTX 1080 Ti before release.

## GTX 16-series (Turing without tensor cores, sm_75, on Salad)

Salad, 17 October, v0.6.0 against PeakMiner 2.17.6 and SRBMiner 3.7.1.

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

No tensor cores, so every miner is equally slow and none is worth running on these cards.

Our miner now runs the GTX 10 cards' dp4a fold on these cards instead of the tensor fold (`PEARL_DP4A_TURING` in
pearl_config.h, on by default). Raced against the tensor fold on the same PC, Salad, 7 November. These are our own
builds on nvtest boxes, not the release files, so the rows above are unchanged:

| Card | Machine ID | Tensor fold | dp4a fold | Change | dp4a fold's clock and power |
|---|---|---|---|---|---|
| GTX 1660 Super | e4505b49 | 1.095 | 9.615 | +777% (ci95 3.3%, ahead 3 of 3 rounds) | 1895 MHz, 117 of 125 W |
| GTX 1650 | 13120675 | 0.692 | 5.654 | +717% (ci95 45%, ahead 2 of 2 rounds) | 1738 MHz, 67 of 75 W |

- Both cards run about 231 MAC an SM a clock. That is 90% of the 256 a half-rate dp4a allows, so there is little left
  to gain. The tensor fold drew only 47 W and 34 W.
- Both found the tensor fold's exact hits on three batch shapes and passed the pool's hit check.
- On Kryptex the 1660 Super mined 9.48 TH/s for 45 minutes, with 4 shares accepted and 0 rejected. The 1650 mined
  5.48 TH/s for 35 minutes on the default build, with 2 accepted and 0 rejected.
- Still not worth renting: Salad's 1660 Super costs $0.027/h at low priority, against $0.034/h for an RTX 2060 that
  mines 40–48 TH/s. The gain is for people who own these cards.

## AMD Radeon (RDNA 3 and 4, on Salad)

Salad, 30 October, v0.6.9 against SRBMiner 3.8.0. PeakMiner mines nothing on AMD.

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

### Salad, 9–10 November (v0.6.13 against SRBMiner 3.8.2)

PeakMiner and OneZeroMiner don't mine on AMD.

| Card | Host | Machine ID | Ours | PeakMiner | SRBMiner | OneZeroMiner | % of best |
|---|---|---|---|---|---|---|---|
| RX 9070 XT | Salad (16 GB) | 28f11ef2 | 131.0 | 0 | 91.5 | 0 | 143% |
| RX 9070 XT | Salad (16 GB) | c1940c4c | 120.8 | 0 | 105.9 | 0 | 114% |
| RX 9060 XT | Salad (16 GB) | d3ae7070 | 70.0 | 0 | 54.9 | 0 | 128% |
| RX 9060 XT | Salad (16 GB) | 3540c3e2 | 59.7 | 0 | 54.0 | 0 | 111% |
| RX 7900 XTX | Salad (24 GB) | 6f96bc89 | 49.8 | 0 | 54.0 | 0 | 92% |
| RX 7900 XTX | Salad (24 GB) | ae456453 | 50.8 | 0 | 54.1 | 0 | 94% |
| RX 7900 XT | Salad (20 GB) | dc8fcef5 | 46.3 | 0 | 45.4 | 0 | 102% |
| RX 7900 XT | Salad (20 GB) | 1d5cab04 | 48.4 | 0 | 47.7 | 0 | 101% |
| RX 7800 XT | Salad (16 GB) | 559a11d6 | 31.8 | 0 | 30.9 | 0 | 103% |
| RX 7800 XT | Salad (16 GB) | 16f1685c | 32.9 | 0 | 32.0 | 0 | 103% |

Every PC passed a 10-minute hit check: 400 hits verified, 0 failed, 7.94–8.05 hits a salt. The 7900 XTX is the one
AMD card where SRBMiner leads, on both PCs and in both run orders.

Rerun on the first of those PCs (`6f96bc89`), 10 November (item 7), with v0.6.13's AMD kernel (the same fold code,
built at 1c78804b): the duel (ours first, then SRBMiner, 5 minutes each on Kryptex, after a 30-minute sweep had
warmed the card) read ours 54.37 TH/s, mean after the first minute, and SRBMiner 54.40, its last figure (mean 53.89).
That is 99.9%, not the row's 92%. `hashrate.js` read 53.48 to 54.35 in three rounds, and the CLI's trace shows the
fold at 99% of each batch (10.07 of 10.17 ms; hash 0.8%, idle 0.2%). So the row's 49.8 was a low reading of this
PC, not a gap in the kernel. The row stays until a full rerun (see "Keeping this file current"). What off-chip
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

### Our own builds on Salad, 7 November: the RDNA 4 readout, late and in three chains (#287 item 4)

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

## Intel Arc (Alchemist, on a rented rig)

The first Intel hardware, 10 November: a rig of six Arc A750 cards. Ours is the SYCL core in `earn/native/sycl`
(the `xmx8` fold: the XMX matrix engines through DPAS at sub-group 8), built on the rig from PR #290 and run by
that branch's CLI through `PEARL_CORE_PATH`. It is in no release. The competitors are IAminer, the Pearl miner
for Arc that could be had, and the two we race elsewhere (SRBMiner, WildRig). TH/s is each miner's own figure.

| Card | Host | Rig | Ours | IAminer | SRBMiner | WildRig | % of best |
|---|---|---|---|---|---|---|---|
| Arc A750 | HiveOS rig (8 GB) | arc750, card 0 | 16.9 | 13.6 | 0 | 0 | 124% |
| Arc A750 x 6 | HiveOS rig (8 GB each) | arc750, all six | 100.1 | 78.0 | 0 | 0 | 128% |

- **IAminer 1.0.0** mines on the card. Its own tuning picked m = n = 131072, k 4096. The pool rejected the only two
  shares it found ("Invalid share"), so it has no accepted share on this rig.
- **SRBMiner 3.8.6 and 3.8.2** list the six cards but cannot mine Pearl on them: "No GPU devices available to mine
  with". Their algorithm list marks `pearlhash` for AMD and NVIDIA only.
- **WildRig 0.52.1 and 0.51.9** list the card, print "Start mining", then show "n/a TH/s" for four minutes with the
  card idle at 34 W.
- **ARCMiner** has no column: it could not be had. Kryptex's miner page lists it as Windows only, and its download
  page (github.com/jbman2025/ARC-miner) returned 404.

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

- **The draw fold, fourth round: 73% to 80.5% in the one correct form, and what the barrier costs (11 November,
  rig01's RTX 4090, probe only).** Probe figures, k 8192 (branch `goal/nv-kernel`; no hash or host side, so no
  mining rate). Timing-only rows compute wrong results on purpose.

  | Stream | CPU check | Pipe use | Clock, watts | TH/s by count |
  |---|---|---|---|---|
  | Third design as left (a barrier a k-step) | 2,560 of 2,560 | 72.93% | 2685 MHz, 411 W | 257 |
  | + draw tethered ahead of A-tile loads, readouts of the two copies 12 mma apart (the default now) | 2,560 of 2,560 (k 2048), 1,280 of 1,280 (k 8192) | 80.47% | 2665 MHz, 448 W | 281 |
  | 19 other correct variants (own barriers, lags, places) | all pass | 72.4 to 77.5% | | |
  | Four tiles a warp (128x256) | 2,048 of 2,048 | 68.01% | 2685 MHz, 408 W | 239 |
  | Draw alone under the barrier / table words from registers / narrow stores / both / one copy drawing | timing only | 81.35% / 89.86% / 85.53% / 88.50% / 88.46% | | |
  | A draw with no stage, by instruction count (7 units a warp a k-step, nothing holding the warps) | timing only | 63.44% | 2436 MHz, 397 W | 203 |

  The release fold is 317.3 TH/s here, so every form still loses. What the round settled: a draw straight into
  the mma's fragment registers needs about 190 instructions a thread a k-step against 54 staged (the bytes have to
  be turned from "rows at one k" to "four k of one row", which `ldmatrix.trans` does for nothing), and its
  stand-in is the slowest stream measured. Under the barrier the draw costs 17.6 points (8.5 table reads, 4 wide
  stores, 10 that stay with registers only) where it costs 3.6 with nothing holding the warps; the cost is there
  with one warp of a scheduler drawing and the other not, and the idle one reaches the barrier no earlier, so it
  is not the two warps being busy at once. It is about fixed a k-step (four tiles a warp are worse than five).
  ptxas's order of one source is worth 4 to 8 points either way (65 to 81% over harmless variants), its spills
  track the loss, and a tool that reorders the compiled loop safely (`probes/drawtools/spread.py`) can move
  almost nothing because ptxas reuses registers across the steps. Tables and reasoning:
  `earn/native/probes/README.md`, "What the barrier costs the draw".

## Keeping this file current

- A change that re-measures a host here updates its row in the same PR. A number only in chat or a PR doesn't count.
- Only a run of our miner on the row's own machine replaces "Ours" and "% of best". Competitor figures stay until the
  next full rerun; a new host runs every miner.
- A run on another host or by another method doesn't change a row: record it as a new table or a research note.
- Before tagging a release, check that the intro and the Summary name the right release.
