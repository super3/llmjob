# Kernel work: what we learned

About 120 agent runs between 10 October and 18 November 2026 raced our Pearl miner against PeakMiner, SRBMiner
and OneZeroMiner on every GPU series. Roughly four ideas in five lost. This file keeps what won, what lost and how
to measure, so nobody repeats a dead idea. Each idea's code is on its `goal/*` branch, and its numbers are in
[`earn/tuning/benchmark.md`](../tuning/benchmark.md).

Before 20 November this file held the first analysis of the CUDA kernel (8 October). That version is at
[821ddb4d](https://github.com/super3/llmjob/blob/821ddb4d9d611a390c41172b508469509d6e2e35/earn/native/OPTIMIZATION.md).

Pearl's FP8 fork (PIP-3) will change the mining math. After it, only RTX 50 and data-center cards keep mining
Pearl, unless an older card can reproduce Blackwell's FP8 arithmetic bit for bit. That work is
in [PR #317](https://github.com/super3/llmjob/pull/317). Everything below is about today's int8 math, but the measuring
rules apply to any kernel.

## How a win is proven and measured

A change counts only after it passes all four gates. Most "wins" that failed later broke one of the measuring
rules below them.

1. The fold's output is bit for bit the same as before.
2. `probes/verify-hits.js` passes 400 of 400 hits, checked the way the pool checks them.
3. It beats a twin: an identical copy of the current build, run beside it. A typical twin spread is 0.1 to 0.3%.
4. The full test suite passes on the exact commit that gets pushed. Then push it at once.

Measuring without fooling yourself:

- Alternate rounds of the new and old builds on the same PC, never across PCs. Salad PCs differ by 5 to 10%.
- Drop noisy, hot or broken PCs. Seen: an RTX 2080 throttling at 87 C, a card stuck at 877 MHz, a WSL host at
  0.4 TH/s, and one run drifting from 44 to 39.5 TH/s.
- Most cards are power-limited, so energy per hash decides. A change that saves power shows as a higher clock.
  A change that adds memory traffic loses, even when it saves instructions.
- Tests with constant or zeroed operands lie, because data that never changes uses less power. Zeroed operands
  alone gave +14% clock on RDNA 4, and constant B saved 45 to 53 W on RTX 40. The "+28% with everything removed"
  RDNA 4 ablation was this, not a real saving.
- Timing-only builds give a ceiling, not a result. One RTX 20 estimate of +1 to 2% became -12.9% once B was
  really loaded.
- At the register limit, trivial edits swing speed 5 to 7%. Always measure, never reason it out, and check for
  spills first.
- The test harness loads its own copy of the miner file. Three different builds once read exactly 53.5 TH/s
  because of it.
- Scheduling models are often wrong. On RTX 20 the model got the sign of every yield-bit change wrong.
- Rest Intel Arc cards 45 to 60 seconds between runs. Warm runs read about 1% low.

Scoring and the benchmark:

- The goal is scored on displayed hashrate (founder, 18 November). SRBMiner displays 2.5 to 7% more than the
  pool credits it, and ours displays about what is credited.
- Since commit 0ea0bbd5, every miner is scored the same way: the mean of all 15-second samples. Before that, a
  competitor's last sample was compared with our mean. Never compare rows scored before and after that change.
- After every run, check that the previous miner really stopped. A leftover PeakMiner once shared the card and
  wrecked two rows.

Running agents:

- One agent per target card, with a written brief, a check-in every $3 to $5 of Salad spend, a report file
  updated at each step, and boxes deleted and locks released at the end.
- Several agent pushes were refused by the permission check when the same command held a force push. Redo the
  push as a plain rebase and push.
- In CI, check each job's result, not the run's. Jobs with a matrix variant are allowed to fail, so a run can
  show green while a SYCL build failed.

## What shipped

The gains came in small steps. RDNA 4 took about ten separate wins of 0.2 to 1.9% each. RTX 20, 30 and 40 got no
kernel win in this round. Every change below passed the four gates.

| Series | Change | Gain | Commit |
|---|---|---|---|
| RDNA 2 | DOT4_ADDR + DOT4_SWZ (dot4 fold, A' words by DPP) | +2.0%, reaching 120% of SRBMiner | d5702de8 |
| RDNA 3 | FUSE on gfx1100/1101, FOLDQ2 on gfx1101 | about +0.4% (7900 XT 95.5 to 96.2%) | e8527660 |
| RDNA 4 | 256x256 blocks at 8 waves | +1.2 to 1.9% | goal/rdna4a |
| RDNA 4 | C4=2 + RFRAME + QLDS, then QLDS=2 with 5 readout chains | +0.7 to 1.2%, then +1.9% (9070 XT), +1.2 to 1.6% (9060 XT) | d2c8e93a, 00b15d75 |
| RDNA 4 | REG2, a cheaper readout | +0.2 to 1.0% | on main |
| RDNA 4 | WGRP=4, one unbroken run of the slab's WMMAs | +0.6 to 0.7% (9070 XT) | 43b20fa7, b619b92a |
| RDNA 4 | FUSE on gfx1200, the hash inside the kernel | +0.5 to 1.3% (9060 XT) | 77c7a4b8 |
| RDNA 4 | 9060 XT at k 8192 | +0.9% | 52bbad38 |
| RDNA 4 | DirectToVgprB on gfx1200 | +0.4 to 0.5% | 6a6089f6 |
| RDNA 4 | Serpentine order, fill 63, batch 128 on gfx1200 | +0.1 to 0.55% each | 6ccb0440, e3b1501e, 7c283621 |
| RDNA 4 | Per-PC k trial at mining start | picks the faster size per PC | e896f813 |
| RTX 50 | 5090 fold file (96x512 block, w8g=4), then trl=1 cq=0 | +1.5%, then +1.4 to 1.6%; 5090 at 100.1 to 100.4% | cd35909e, c580e560 |
| RTX 50 | Both fold files timed at start-up (600 s trial) | the best file per card | 881bbfb4, ef136b4a |
| RTX 50 | k-order refine for power-capped cards | +0.1 to 0.2% | v0.7.5 |
| RTX 30/40/50 | Rotation fold on by default | +2.7 to 9.8% per PC (5070 Ti +5.5%) | v0.6.16 |
| Arc B50 | xmx16f fold, then 2D loads and a pipelined k-step | 11.55 to 46.5 TH/s (4x) | b755ac1e, e447abad |
| Arc A750 | xmx8w set stream, wider batches | about 105 TH/s per card | v0.7.3 |
| Arc A750 | 384 columns a batch at half the band, at full speed | +0.65% (105.3 against 104.6 TH/s) | v0.7.2 |
| RTX 30/40/50 | Host fixes: one read per hit (PEARL_HIT_DEDUP), search threads (PEARL_ROT_SEARCH_THREADS), A's chunk trees taken from B's (PEARL_ROT_TREES) | +0.07 to 0.29%; job draw 39 to 30 ms | v0.7.2 |
| RDNA 4 | The 16 GB RX 9060 XT keeps the old tile order at batch 128 (PEARL_AMD_WALK_MINGX) | +0.4 to 0.45% | v0.7.5 |

One later RDNA 4 win came on 20 November: W64K, a wave64 fold with 64-byte slabs, is now the 9070 XT default.
Every earlier wave64 form lost (below).

## Tried and lost on AMD

These came out slower, level or broken. Don't retry them without a new reason.

RDNA 4 (RX 9070 XT, 9060 XT):

- Block shape: 512x256 -4.8%; 256x256 at 4 waves a SIMD -7.8%; W96 tile -5 to -5.8%; tile walks -0.1 to -0.25%;
  batch 512 -0.3%.
- No LDS at all (DIRECT=1/2) -5.7 to -7.9%; A' from memory (ADIR) -2.4%; GB on the 9070 XT -0.3 to -2.2%.
- Barriers and rings: split barriers -5 to -23%; lagged barrier LAGB=2 -4.5% (LAGB=1 level); RING4 -5.5 to
  -6.1%; RING4=3 spills; LAG -1.0 to -1.3%.
- Readout: late readout at 256x256 -3.6%; LDS-atomic readout -5 to -7%; RDK4 -5%; RDIL -5%; RDLS -4 to -5.5%;
  RDSPLIT -2.5 to -4.1%; RDBANK -1.4%; RDLSB -0.8%; RDRS -0.3%; RDINT and UNROLL4 spill; RDEXEC +0.2% (left off).
- Staging and prefetch: PFR -8 to -12%; K64 slabs -6%; AHEAD reads -5.7%; PF1+RDLATE=2 -1.4%; the ST3 family
  -0.15 to -7.3%; QG -3.3 to -4.3%; QS slot -2.2 to -2.7%; STFR -1.7%; GA -5%; STEARLY spills.
- Wave64 before W64K: W64=1 -8 to -11%; W64=2 -1.2%; W64 (128x64) -2.1 to -2.8%; W64K=2 -3.3 to -4.2%; W64P
  -3.4 to -7%.
- Scheduling: CU mode -2.8 to -4.5%; s_setprio -1.2%; expert scheduling mode -0.3 to -0.5%. Setting SCHED_MODE
  by hand hung GPUs.
- Memory hints: non-temporal loads -21%; LDAUXA hints level to -30%; SOFF -0.8 to -1.0%.
- Other: KORD/KORDP -2.7 to -8% per clock; run-time FOLDQ2 -1.7%; HASHQ -0.8 to -3.5%; k 16384 on the 9060 XT
  -3.5%; FUSE on the 9070 XT level; LDS padding to 48 or 64 KB level.
- AMD's own hipBLASLt int8 kernels are slower per clock than our fold.
- The Blackwell miner's column reorder doesn't apply to this fold.

RDNA 3 (RX 7000):

- 256x256 at 8 waves -0.3 to -1.5% (at 4 waves -8%); readout chains (RDTREE11) -0.3 to -0.9%; FOLDQ2 on the XTX
  -0.8%.
- Persistent blocks -2.6 to -6%; groups of 4 tiles -0.6 to -0.9%; a lagged barrier with two register sets -3 to
  -6%; batch 256 -0.65%.
- Larger k loses on RDNA 3 (KWRAP -0.4 to -1.2%); REG2_11 +0.1% at best; HASH2 -0.3%; serpentine order level.
  ROCm 7.2 refuses LDS DMA.

RDNA 2 (RX 6000):

- Readout after the barrier -2.6% (-0.9% with reads one k-pair ahead); 16-wave blocks -1%; conflict-free A' reads
  level.

- 20 Nov, RX 9070 XT, hand-written wave64 slab (all 8 A' tiles read before the readout): level, +0.1%. A sweep of
  where the slab barrier stands found the shipped place best; the 2.5% one test showed was the cost of a bad place.
  The same hand barrier edits on RDNA 3 (gfx1100) were level too. Branch goal/f-amdasm2.

## Tried and lost on NVIDIA

RTX 20 (Turing):

- Hand-edited scheduling: dropping 28 or 64 of 78 waits -0.03%; moving IMMA scoreboards -0.04%; every yield-bit
  change lost (-1.0 to -3.9%); reordered lists -1.5 to +0.25%, against a predicted +3 to +7%; 28 IMMA reuse
  flags -0.04%.
- Fewer PRMT gates spill at 255 registers, and with PEARL_BD_SB they lose 3.7 to 10.4%. Single-buffered operands
  -0.7 to -4.3%.
- Hand-placed readout: two forms fail verify (a race), and the third is level. Spread hash -4.5 to -6.0%; hash
  lag -1.9 to -2.8%.
- The rotation fold doesn't fit Turing: 55 to 65% of the pipe against the release's 92%. Hybrid fold -12.9%;
  full 8x16 fold -34.3%.
- Fewer memory reads: streaming loads -1.9 to -4.7%, last-use loads -21.7%, and the ceiling with A in L2 is only
  +1.1%. Narrow B loads -1.6 to -4.4%; B through a shared ring priced at -5 to -27 points of pipe.
- On-chip draws lose 1 to 6%. k 4096 was taken back out of the RTX 20 fold.

RTX 30 (Ampere):

- The 5090's 96x512 block only ties, even at its no-barrier ceiling. The RTX 40 loop on RTX 30 -0.5% (-2.6% with
  iy=0).
- Priced in a timing-only build on a 3080 Ti (ceilings, not results): nord +1.1%, noprm +0.7%, lgh +0.8%,
  rdx +0.5%.

RTX 40 (Ada):

- 96x512 block ties at best; iy=0 -2.5%; bt=13, ly=0 and py=1 no change; cq=0 -0.16 to -0.31%.
- Streamed B' -18.7 to -19.9%; B' in shared memory across row bands -4.2%; B grouped by rank (the G design)
  +2.6% before its overflow fix, -5% after.
- trl=2 fails one tile in six (a race). brr gives wrong results.
- No cheaper correct form of the readout, the PRMT or the transcript word exists (worked out on paper).
- ptxas -O1 -6.5%; -O2 builds byte-identical to the shipped -O3. TMA staging -0.24%.

RTX 50 (Blackwell):

- 128x48 tile -0.5 to -3.5%; SRBMiner's 128x384 block as a hand loop -0.17 to -0.53%; gathering A every turn
  -3.6 to -4.3%.
- Barrier variants: w8 with nvk22's barriers is an illegal instruction on sm_120; w8mu hung the card; w8m, w8mnt
  and w8msb level.
- Record ring in shared memory -0.3 to -0.7%; packed records no gain; trp -0.12 to -0.25%; trq -0.18%; eviction
  hints no gain.
- The 5090 file on smaller or capped cards: -0.17 to -1.18% on some PCs, +0.5% on others. That's why the start-up
  trial picks per card.
- A k order that re-reads B rows is within 0.3%. OneZeroMiner's own power-save mode lost 0.6% for them.

- 20 Nov, RTX 20 hand-written 96 x 64 warp tile: a timing-only build read +1.5 to 2% on two RTX 2060s, but the real
  build is level (40.35-40.58 against 40.37-40.82 TH/s on a 2060), because the bigger tile needs two hash passes
  where the release has one (4.3%). Turing's instruction cache is not the limit: the fold's loop is 10.2 KB and loss
  starts at 32 KB. Branch goal/f-tu-build4 (1991504c) has the build and a fixed readout race.

- 20 Nov, RTX 40, the RDNA 3 Infinity Cache idea (a smaller m so the operands stay in L2): no gain. On a Salad RTX 4090
  (the tall fold, hashrate.js, 3 rounds each on one PC) m 16,384 read 310.1 TH/s against 314.4-314.9 at the mainnet m,
  both at 449 W and about 2,514 MHz. The memory controller was only 3-5% busy at either m: the NVIDIA fold already
  keeps its operands on the chip, so there is no memory power to save. Probe switch PEARL_PROFILE_M on goal/nv-l2.

## Tried and lost on Intel Arc

A750 (Alchemist):

- Cache hints on A': 98.4 to 100.4 against 103.5 to 104.7 TH/s.
- Explicit work-group pairing to share A': -1 to -17%. A compile-time walk change: -8.6% per clock.
- Prologue changes, finaliser options, CHUNK2 + WIDE, and bands of 16 or 64 at 256 columns: all level or lower.
  320, 352 and 448 columns: no better than 384.

Arc Pro B50 (Battlemage):

- A' through local memory, A' prefetch, cache controls, another DPAS order, bigger work-groups: all dropped.
- 2D loads of A' and B' as a drop-in -5.5% (A' alone -7%); B' prefetch -1%; double-buffering both operands
  spills; a B' ring spills at 28.9 TH/s.
- The B50 is bound by its 73 W cap, and about a fifth of the watts go to L2 operand traffic.

## Where each card stands

The latest official run (15 November, build 1975683d) put 9 of 70 rows above 100% of the best competitor on
displayed hashrate. Later agent results are listed where they are newer.

| Series | Rows above 100% (15 Nov) | Median | Lowest rows and later results |
|---|---|---|---|
| AMD | 4 of 13 | 95.9% | RDNA 2 120 to 121%. RX 7800 XT 108%. RX 9070 XT 94.0 to 95.5% and RX 9060 XT 94.2 to 96%, but 102.4% ±0.9% on pool-credited work. RX 7900 XT 94.3%, RX 7900 XTX 94.9% (98.2 to 98.9% credited). |
| RTX 20 | 4 of 11 | 99.8% | 2080 Ti 101.2%; 2070 Super 96.5%; 2080 97.1 to 100.6% depending on the PC; 2060 98.2%. |
| RTX 30 | 1 of 16 | 98.4% | 3080 Ti 97.4%, 1.9% behind SRBMiner. |
| RTX 40 | 0 of 18 | 98.8% | 4070 Super 96.7%; 4090 1.1% behind OneZeroMiner at 450 W and 4.6% behind at 250 W. |
| RTX 50 | 0 of 12 | 99.3% | 5090 97.8 to 100.4% depending on the PC; 5080 98.0 to 99.95%; 5070 100.0 to 100.4%; 5060 99.8%. |
| Intel | not raced | | A750 105.3 TH/s per card at full speed; B50 46.5 TH/s at 73 W. |

Why we trail: on NVIDIA our work per clock equals OneZeroMiner's, but our loop draws 3.4 to 3.5% more power.
SRBMiner gets 1.5 to 2.8% more clock from the same watts on RTX 50. On the RX 7900 XTX, SRBMiner runs about 4%
higher clock at a higher voltage.

## Open leads

The biggest gaps are power, not instruction count, so leads that cut memory traffic or operand switching rank
first.

1. RDNA 4 readout cost, 8% a clock: free about 4 registers at the slab start and retry RDLS, or keep readout
   words in a small per-block L2 slot.
2. RTX 50: the transcript in shared memory is worth about +0.4% if room can be found. The rbr B side alone is
   +2.4 to 2.5%, but the A side currently takes it back. A rule "power limit below default uses the w8 file"
   would give about +0.5% on capped smaller cards.
3. RTX 20, 30 and 40: any correct source for B that costs under 20 to 25 W, and a gather scheme with fewer
   shared-memory reads per MAC.
4. B50: the no-load timing build reaches 52 TH/s against 46.5 shipped, so cutting L2 operand traffic is the lead.
5. Matching competitors' clock per watt on RTX 50 and the RX 7900 XTX, where most of the gap now is.

6. RDNA 3, a small matrix kept in the Infinity Cache (m 16,384 in 8 strips): +3.0 to +3.6% on the RX 7900 XTX
   (table PC 5ae23c47), all of it clock (2,810 against 2,698 MHz), verify-hits 400/400. Built as the gfx1100/gfx1101
   default on goal/f-amdasm2 (0ac5bf6f; `PEARL_AMD_M=131072` turns it off). Still to do on 20 Nov: the SRBMiner race,
   a second PC, the 7900 XT and 7800 XT, and the Windows core. Worth trying on RDNA 4's cache too.

## Quantus

The Quantus (QPoW) cores and their notes are on PR #317. They are not in this release: our miner
reaches about 70% of SRBMiner on NVIDIA.
