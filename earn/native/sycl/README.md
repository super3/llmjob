# Pearl core for Intel GPUs (SYCL)

A first version of the Pearl mining core for Intel GPUs, written in SYCL. It
builds into `pearl_core_sycl.node`. The N-API layer is `../src/pearl_core.cc`,
compiled unchanged; this folder replaces only what sits under it
(`pearl_host.cu` and `pearl_kernel.cu`). Nothing here changes the CUDA or AMD
builds or what the app does on those cards.

CI builds it for Linux and Windows and the packages carry it, in a folder of
its own ("The build a release carries"). The app and the CLI load it on a rig
of Intel Arc cards with no NVIDIA or AMD card, or with `--core intel`;
`PEARL_CORE_PATH` still names a local build. No release has shipped it yet.

It ran on Intel hardware for the first time on 10 November 2026: a rig of six
Arc A750 cards (Alchemist). On every card the hardware DPAS fold matches a
scalar fold on the host, every hit recomputed in JS is right, and the pool
accepted every share sent: 15 of 15. An A750 does 16.9 TH/s at 152 W and the
six together 100 TH/s ("Results on an Arc A750 rig"). No other Intel card has
run it. Every GPU still mines with each hit checked before it is sent
("Untested cards").

On 11 November the same rig got a new fold for these cards, the fragment fold
(`xmx8f`). A cool A750 does 64 TH/s with it, 3.8 times the first fold, at the
card's 190 W limit. The six together mined at 330 to 350 TH/s with every
share accepted, 55 to 58 a card: four of the rig's cards lose clock to heat
within minutes at that power ("The fragment fold"). It is what an Arc
A-series card mines with now.

"Running it on an Arc A750 rig" has the steps as they were run on that rig.
"On an Intel box" and "Test plan" are the same steps for any other Intel box.

It was written in October 2026, against the app as it was then, and brought up
to the app's core API as of v0.6.16 on 10 November ("Kept up with the CUDA
host").

## Status

| What | State |
|---|---|
| Draw: job_key, constant and hashed fills, salt stamp, commitment trees, cert-v3 seeds, noise, noised operands | written; checked end to end on the CPU device and on six Arc A750s (constant fill) |
| Plain fold (`dot`): int8 dot products in local memory, on any device the build has kernels for (see "Which build") | written; checked on the CPU device and on six A750s: 4.0 TH/s a card |
| XMX fold, sub-group 16 (`xmx16`): Arc B580/B570, Arc Pro B50/B60/B65/B70, Lunar Lake, Panther Lake (H and U), Data Center GPU Max | written; its reference code checked on the CPU device and on an A750; compiles to 32 DPAS, 128 GRF, no spill. The hardware instruction has not run: no such card yet |
| XMX fold, sub-group 8 (`xmx8`): Arc A770/A750/A580/A380/A310, Arc Pro A-series, Flex | written; **the hardware DPAS is right on six A750s**: 16.9 TH/s a card. Its reference code is right on the CPU device and wrong on the A750 (it mines nothing; see the results) |
| Fragment fold (`xmx8f`), sub-group 8: the same cards as `xmx8`, and what `auto` picks on them | written; **right on six A750s: 64 TH/s on a cool card, 55 to 58 a card with six running**. 16 rows a sub-group in the 256-register mode, operands stored in DPAS fragment order, B' loaded in halves and a k-step's loads sent first. Its reference code is checked on the CPU device |
| Two-batch pipeline, all hits a batch, share proofs carried with each hit | written; checked on the CPU device and on the A750s, where a pool accepted the shares |
| Device choice: Level Zero GPUs ranked by compute units x clock, `PEARL_GPU_INDEX` pins | written; six cards listed and each pinned in turn on the rig |
| Untested cards: a GPU opens only for an opener that checks every hit, and says so in `core.untested` | written; on the rig the core reported it and the CLI checked each card's first hit |
| Per-card k, launch shapes and their startup trial (the CUDA host's) | not here: the profile's k, no shapes ("Kept up with the CUDA host") |
| Runs on an Intel GPU | **yes, on Arc A750** (six cards, 10 November). Nothing else yet |
| Speed | **64 TH/s on a cool A750** at 189 W, its power limit, with the fragment fold; 330 to 350 TH/s for the six over 5 minutes at 960 to 1,035 W, with four cards losing clock to heat. The first fold: 16.9 and 100.1. The card's matrix engines do 113 T-MAC/s with nothing else in the loop, so the fold is at 56% on a cool card ("The fragment fold") |
| Windows build | written (`build-win.sh`), built in CI and checked on the SYCL CPU device. **No Intel card has run it** |
| In a release | CI builds and stages it for Linux and Windows, and the installers, the Linux CLI and the HiveOS package carry it ("The build a release carries"). Not released yet |
| HiveOS | **yes**: it mined on the rig's own HiveOS userland (Ubuntu 22.04, Intel OpenCL 24.35), six cards, 60.6 TH/s |

## Results on an Arc A750 rig (10 November 2026)

The first run on Intel hardware, on a rented rig: six Arc A750 cards (DG2,
`8086:56a1`, rev 08) in the x1 slots of a mining board (PCIe 2.5 GT/s x1 each),
an i5-10400T, 7.8 GB of RAM and no swap, HiveOS 0.6-231 on Ubuntu 22.04.5
(glibc 2.35), kernel 6.12.0-hiveos, the cards on `i915`. Everything ran in an
Ubuntu 24.04 container given `/dev/dri`: Intel's compute runtime
26.40.40196.15 (Level Zero and OpenCL, IGC 2.42.14), DPC++ 2026.1.1, Node
22.22.0, and the build for `intel_gpu_acm_g10`.

| Check | Cards | Result |
|---|---|---|
| `pearl_sycl_check --list` | | six cards through Level Zero, each `auto fold xmx8 (hardware DPAS)` |
| fold check, m = n = 2048: the plain fold and the hardware DPAS, each against the host | all six | 16,384 of 16,384 transcripts match for each fold on each card; 29 of 29 hits |
| the same at the mainnet m and n, the last column batch | all six | 32,768 of 32,768 match for each; 46 of 46 hits, the same 46 as on the CPU device |
| `verify.sh --quick`: 100 hits recomputed in JS with each fold | all six | PASS on each: 100 of 100 for `xmx8` and for `dot`, from 4 salts each. Hit ratio 0.88 on every card: the hits are the same on each, so that is this job's luck |
| `verify.sh`: 400 hits with each fold | card 0, then card 1 with the band as it is now | PASS on both: 400 of 400 for `xmx8` and for `dot`, from 7 salts each, hit ratio 0.95 on card 0 |
| five minutes on a pool (Kryptex) with the CLI, through `PEARL_CORE_PATH`: three times on all six cards, three times on card 0 | all six | 15 shares accepted, 0 rejected. The CLI checked each card's first hit before sending it, and each was right |
| the reference code of the sub-group 8 fold (`xmx8-emu`) | all six | **wrong on every card**: all 16,384 transcripts differ from the host. It mines nothing (below) |

Speed. The first column is the core as it arrived, walking 64 row windows
together (the band). The second is after the one change the rig led to: 256
for the XMX folds (below). Power and clock are i915's own counters for the
card (`hwmon/energy1_input`, `gt_act_freq_mhz`), read every 5 s.

| What | TH/s, band 64 | TH/s, band 256 (now) | Card power and clock, now |
|---|---|---|---|
| `xmx8` (hardware DPAS), the fold alone, 20 s | 15.47 | 16.87 | 152 W (145 W at band 64), 2,000 MHz |
| `xmx8`, the full miner loop (`bench.js`, 60 s) | 15.47 (13.5 to 17.4 over its half-second samples) | 16.88 (14.7 to 19.0) | 152 W, 2,000 MHz |
| one card on the pool, 5 minutes | 15.45 and 15.47 | 16.88 | 153 W (146 W at band 64) |
| six cards on the pool, 5 minutes | 91.9 (mean after the first minute; 87 to 97) | 100.1 (97 to 103) | 921 W for the six (875 W at band 64); 2,000 MHz each |
| `dot` (plain), the fold alone, 20 s; its band is 64 in both | 4.00 | 3.90 | 185 to 190 W, the card's limit; 2,250 MHz |
| a card idle | | | 31 to 36 W |

Card by card at band 64, with every hit checked (the quick verify's runs):
`xmx8` 15.44, 15.36, 15.17, 15.17, 15.17 and 15.16 TH/s; `dot` 3.98, 3.91,
3.91, 3.90, 3.92 and 3.92. Card 1 at band 256, the same way: 16.74.

Last, the branch as pushed (this port on that day's tip of PR #290, CLI
0.6.16) was built again on the rig. It passed `verify.sh --quick` on card 3
and on the CPU device, and mined a third five minutes on the six cards: 100.1
TH/s again, 919 W, 3 accepted, 0 rejected.

Sweeps on card 0 with `xmx8`, 20 s each (`bench.sh --sweep`, and the wider
bands by hand):

| `PEARL_SYCL_COL_BATCH` | 64 | 128 | 256 (default) | 512 | 1024 |
|---|---|---|---|---|---|
| TH/s | 15.48 | 15.44 | 15.47 | 15.47 | 15.46 |

| `PEARL_SYCL_BAND` | 4 | 16 | 64 (the default then) | 256 (the default now) | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|---|---|
| TH/s | 14.08 | 14.31 | 15.47 | 16.87 | 13.75 | 13.38 | 13.48 | 13.52 |

The same on card 5, twice each: 64 gave 15.23 and 15.24, 128 gave 16.30 and
16.30, 256 gave 16.58 and 16.59. The plain fold goes the other way: 3.90 at
16, 3.92 at 64, 3.75 at 256.

What it shows:

- **The hardware DPAS fold is right on Alchemist.** That was the open
  question. Every fold is compared with a scalar fold on the host, and the
  hardware fold matches it on every region checked, on six cards; 1,500 of its
  hits were recomputed in JS and a pool took its 15 shares.
- **The reference code is what failed.** At sub-group 8 it gives other
  transcripts on the A750: all of them, and the same wrong ones on every card.
  It does so whether the kernels are prebuilt or compiled by the driver
  (`--jit`), through Level Zero or OpenCL, and with `select_from_group` or
  `group_broadcast` as its read of the other lanes. The same source is right
  on the CPU device, and its sub-group 16 form is right on the A750. So it is
  either IGC 2.42.14's code for that kernel on DG2, or something in the kernel
  that the CPU device forgives. Not found. It mines nothing: `auto` never picks
  it. `verify.sh` used to count it and so failed a good card; it now reports
  it and counts the folds that mine.
- **The batch width does not matter and the band does.** For the DPAS fold,
  256 row windows is 9% faster than 64, and past 256 it falls off. The host
  sized the band so its A' filled a quarter of the cache the card reports (16
  MB, so 64). It now fills all of it for the XMX folds (256) and keeps the
  quarter for the plain fold, which is slower at 256. That is the one speed
  change from this run, and it rests on one card model.
- **The fold leaves the card idle in part.** Under DPAS the card runs at 2,000
  MHz and about 150 W, under its 190 W limit; the plain fold runs it at the
  limit. The fold loads its operands lane by lane ("Not done").
- **None of the failures the plan listed happened.** Docker was there, the
  container saw the cards, Level Zero worked, the prebuilt kernels loaded,
  nothing hung and no batch timed out. Setup took 2 min 43 s, the build 31 s
  with 300 MB at its peak, a card's quick verify 112 s, and the 7.8 GB of RAM
  was never short.

`earn/tuning/benchmark.md`, "Intel Arc", has the same card against the other
Pearl miners that were tried on it. Only IAminer mined: 13.6 TH/s a card.

## The A750's ceiling (10 November 2026)

On paper an A750 does 448 EUs x 128 int8 MACs a clock x 2.0 GHz, about 115
T-MAC/s (Intel's 229 TOPS, halved). The fold mines at 16.9, 15% of that.
`pearl_sycl_peak` is a probe that times the hardware DPAS in the fold's own
grid and thread shape with nothing else in the loop, then puts the fold's
parts back one at a time. It checks nothing: only `fold-copy` and `fold` are
real folds. Build it with `./build.sh --peak`; its header lists every row.

Card 0 of the rig, 4 s a row, the default band (256). Clock and watts are
i915's counters for the card, read while the row runs. i915 has no temperature
for Arc on kernel 6.12, so there is none; the fan went from 900 to 2,000 rpm
over a run.

| Row | What runs | T-MAC/s | Clock, power |
|---|---|---|---|
| `regs` | DPAS alone, operands in registers | **113.4** | 2,012 MHz, 166 W |
| `regs`, 120 s | the same, to see it hold | 113.4 (113.2 to 113.5 each second) | 2,000 MHz, 177 W |
| `regs-one` | 1 of the 8 accumulators | 91.2 | 2,000 MHz, 165 W |
| `regs-short` | one window a thread, as the fold has it (16 times the thread starts) | 112.6 | 1,998 MHz, 170 W |
| `regs-rd` | and the fold's readout, barrier, hash and test | 110.1 | 2,017 MHz, 177 W |
| `a-win0` | A' loaded as the fold loads it, from one window; B in registers | 104.5 | 1,924 MHz, 184 W |
| `b-win0` | B' loaded as the fold loads it, from one window; A in registers | 56.6 | 1,998 MHz, 185 W |
| `window0` | both, every window reading window 0 (192 KB that never leaves the cache) | 39.8 | 1,995 MHz, 166 W |
| `load-a` | A' from the fold's own addresses, B in registers | 79.9 | 1,871 MHz, 184 W |
| `own-a` | A' from the fold's own addresses, B' from window 0 | 22.7 | 2,019 MHz, 152 W |
| `own-b` | B' from the fold's own addresses, A' from window 0 | 25.0 | 2,019 MHz, 150 W |
| `loads` | both from the fold's own addresses, no readout | 16.9 | 2,019 MHz, 148 W |
| `loads-1dpas` | the same loads with 7 of every 8 DPAS taken out | 16.9, counted as if all 8 ran | 2,019 MHz, 138 W |
| `fold-copy` | the probe's copy of the fold | 16.9 | 2,038 MHz, 151 W |
| `fold` | the kernel the core mines with | 16.9 | 2,019 MHz, 151 W |

What it says:

- **The paper figure is real.** DPAS alone runs at 113.4 T-MAC/s and holds it
  for two minutes: 126 MACs an EU a clock, where Intel's figure is 128. The
  card's top clock is 2,400 MHz; under DPAS it sat at 2,000 at every load,
  with power to spare. Why is not known, and clocks were not touched.
- **The fold leaves the matrix engines idle 85% of the time, waiting for
  operands.** Take 7 of every 8 DPAS out and leave the loads in, and the rate
  does not move. One accumulator alone gets 80% of the peak, thread starts
  cost 1%, the readout, barrier and hash 2%.
- **The loads cost twice.** Even from memory that never leaves the cache, the
  fold's load instructions hold it to 39.8: it sends 3 load messages for each
  DPAS (8 for an A fragment, 2 for each of 8 B fragments, a k-step of 8 DPAS),
  and the card finishes 5.5 to 5.9e10 of them a second in every row that is
  not held by the DPAS itself. From the fold's own addresses it falls again,
  to 16.9. Either operand does most of that on its own (`own-a`, `own-b`),
  though each is fast when the other is in registers. The likely reason: the
  band fills the card's 16 MB cache with A', so the two together do not fit.

What a different fold could reach. These rows are the fold with its loads
changed, readout, barrier, hash and test included. Their bits are not a
fold's, so they are estimates:

| Loads | 8 rows a sub-group, 128 registers (as today) | 16 rows a sub-group, 256 registers (`--rb 2`) |
|---|---|---|
| the fold's (`fold-copy`) | 16.9 | 21.6 |
| the fold's, at the addresses the bytes would have with A' and B' stored in DPAS fragment order (`frag-fold`) | 22.9; 31.4 at band 64 | 22.9 |
| one vector load a fragment, B' where it is and A' in fragment order (`vec-fold`) | 18.6 | 33.8 |
| one vector load a fragment, both in fragment order (`vecfrag-fold`) | 22.8 | 38.0; **42.7 at band 32 or 64** |

- **16 rows a sub-group** needs the 256-register mode, which leaves an EU 4
  threads instead of 8. That costs nothing: `regs` is 113.6 there. A B
  fragment is then loaded once for two DPAS, and from memory that stays in the
  cache the ceiling rises from 39.8 to 60.5.
- **One vector load a fragment** is one message where the fold sends 2 (B) or
  8 (A). For A it needs each lane's 8 rows stored together.
- **Fragment order** puts the 8 lanes' 32 bytes of a B fragment side by side,
  in 4 cache lines. Today they are 8 rows of B' 2,048 bytes apart.
- The three together are 42.7 T-MAC/s, 2.5 times the fold and 38% of the peak.
  That is the realistic target for the next fold on this card. Card 3 gave the
  same 128-register rows within 2% (`fold` 16.6, `vecfrag-fold` 22.4, `regs`
  113.7).
- The band has to be tuned again with any of them: the best moved from 256 to
  32 or 64, and the rows do not change smoothly with it (`frag-fold` at 64,
  128, 256: 31.4, 18.2, 22.9).
- Past that it is still the loads that limit, not the matrix engines: the
  best row leaves them idle 62% of the time. No row here shows the way past
  about 43 on this card. A thread cannot hold a larger tile: 16 accumulators
  are half of the 256 registers, and the operands need most of the rest.

Not measured: staging B through local memory, any other card, and a run longer
than two minutes. The 2D block loads are Xe2 and Xe-HPC only.

## The fragment fold (11 November 2026)

The probe's stand-ins said what the next fold should be, and it was built:
`fold_xmxf` in `pearl_sycl_kernels.hpp`, `xmx8f` to `PEARL_SYCL_FOLD`.

- **16 rows a sub-group.** A window is 2 sub-groups, not 4, and each B'
  fragment loaded feeds two DPAS. The 16 accumulators need the 256-register
  mode. The kernel asks for it with a kernel property (`grf_size<256>`), so
  it works in the build the driver compiles and in the prebuilt one alike. An
  EU then runs 4 threads, not 8.
- **A' and B' in fragment order.** The draw's last kernel stores each noised
  operand the way DPAS takes it: a lane's 32 bytes of a fragment in a row, the
  8 lanes' 256 bytes in a row (`Layout` in the kernels' header). Nothing else
  reads the noised operands. The proofs come from the operands before the
  noise, which are stored as before.
- **Loads a lane's own bytes as vectors.** The first version (commit
  `0308ff15`) loaded every fragment as one vector of 8 dwords: 10 load
  messages for a k-step of 16 DPAS, where the first fold sends 24 for 8. It
  did 42.2 TH/s. The probe then showed that the card serves a 256-byte load
  message at a fixed rate whatever else changes, and a 128-byte one about
  four times as fast ("Where its time goes now"). So the fold as it is now
  (`08e4ae8e`) loads a fragment of B' as two vectors of 4 dwords, and A',
  a fifth of the bytes, still as one.
- **A k-step's loads first, then its 16 DPAS.** Left alone, the driver's
  compiler moves each load down to the DPAS that reads it and keeps two
  fragments in flight. To hold the order, a word of every fragment loaded is
  ANDed with a field that is always 0 (`FoldArgs::zero`), which the compiler
  cannot know, and ORed into A: no DPAS starts before its k-step's loads
  have landed.
- **Band 64.** A quarter of the cache the card reports, where the first fold
  wants all of it. Card 0, 6 s each: 55.8 TH/s at 16, 60.6 at 32, 64.2 at 64,
  44.1 at 128. A batch of 256 column offsets stays right: 60.8 at 64, 62.8
  at 128, 43.6 at 512.

- **Written without loops.** A k-step's loads, its wait and its DPAS, and
  the readout's XORs, are written out by a fold expression (`each<N>`). Kept
  as loops, oneAPI's compiler left the fragments and the accumulators as
  arrays in memory, and the prebuilt form ran at 3 TH/s. It compiled, spilled
  nothing and passed every check. The build a release carries was never
  slow, but only because its compiler unrolled the loops by itself. The
  prebuilt form now does 41.6 TH/s against 64.3 (it still keeps the loop over
  k-steps), and the compile gate wants a whole k-step's 16 DPAS unrolled.

`auto` picks it on Alchemist. The first fold is still there as
`PEARL_SYCL_FOLD=xmx8`, and is what the rows marked "first fold" ran.

All of it is the build a release carries: open-source DPC++ v7.2.0, kernels
compiled by the driver (Intel 26.40, Level Zero), Ubuntu 24.04 container.
Clock and watts are i915's counters for the card.

Both versions passed every check below; the figures are the second's.

| Check | Cards | Result |
|---|---|---|
| `verify.sh --quick`: the fold and the plain fold against the host at two sizes, 100 hits of each recomputed in JS | all six | PASS on each: 16,384 of 16,384 and 32,768 of 32,768 transcripts, 100 of 100 hits, at 61.2 to 63.4 TH/s with every hit checked |
| `verify.sh`: 400 hits of each | card 0 | PASS |
| the fold's reference code (`xmx8f-emu`) | all six | wrong on every card, as `xmx8-emu` is with this driver's compiler. It mines nothing |
| `compile-gate.sh` | the three Alchemist targets | the fold: SIMD8, 256 registers, no spill. The first fold: 128 registers, no spill, as before |
| the rig's own HiveOS userland, no container (Intel OpenCL 24.35 of 2024): `verify.sh --quick` with the first version, the fold check with the second | card 0 | PASS. There the reference code is right too |
| the SYCL CPU device, `verify.sh --quick` | | PASS: the fold's reference code, every region and 100 hits |

The fold as it is now, with the first fold and the fragment fold's first
version beside it:

| Rate | First fold (`xmx8`) | Fragment fold, first version | Fragment fold now | Now, twin |
|---|---|---|---|---|
| Card 0 cool, the fold alone, 15 s, taking turns | 16.86 TH/s, 2,000 MHz, 149 W | 42.24, 2,000 MHz, 169 W | 63.91, 1,965 MHz, 189 W | 63.70, 1,954 MHz, 189 W |
| the same a minute later, the card hot | | 40.23, 1,900 MHz, 157 W | 61.01, 1,856 MHz, 178 W | 57.03, 1,741 MHz, 160 W |
| Six cards on the pool (Kryptex), 5 minutes each, one after the other on a rig already hot, mean after the first minute | | 240.20 (236.2 to 247.7), 949 W, 1,908 MHz; 8 accepted, 0 rejected | 349.86 (329.0 to 371.9), 1,035 W, 1,810 MHz; 12 accepted, 0 rejected | 329.74 (322.9 to 343.6), 958 W, 1,704 MHz; 17 accepted, 0 rejected |
| The host's own userland, 2024 driver, card 0 warm, 15 s | 9.14 | 39.41 | 54.89 | 54.88 |

- **1.5 times the first version and 3.8 times the first fold on a cool
  card**, and 1.4 to 1.5 times the first version with six cards running.
- **It runs the card at its power limit.** A cool card draws 189 to 190 W
  and already gives up some clock for that (1,950 to 1,965 MHz). Per EU and
  clock the fold is at 58% of the matrix engines' peak.
- **And this rig cannot shed 190 W a card.** In the six-card runs every card
  started at 190 W. Within three minutes cards 0 to 3 were at 1,550 to 1,650
  MHz and 138 to 152 W, card 4 at 1,800 MHz, and only card 5, at the open
  end of the row, kept 1,950 MHz. That is where the 6% between the two runs
  comes from. The rate follows the clock.

The first version, on a cooler rig earlier the same day:

| Rate | First fold (`xmx8`) | Fragment fold | Fragment fold, twin |
|---|---|---|---|
| Card 0, the fold alone, 20 s, taking turns: round 1 | 16.87 TH/s, 2,000 MHz, 150 W | 42.22, 2,000 MHz, 166 W | 42.25, 2,011 MHz, 167 W |
| round 2 | 16.88, 2,011 MHz, 154 W | 42.22, 2,000 MHz, 169 W | 42.25, 2,011 MHz, 170 W |
| round 3, the card hot and losing clock | 16.87, 2,011 MHz, 157 W | 40.76, 1,922 MHz, 160 W | 39.19, 1,850 MHz, 152 W |
| Six cards on the pool (Kryptex), 5 minutes, mean after the first | 100.08 TH/s (98.0 to 102.2), 924 W, 2,001 MHz; 1 accepted, 0 rejected | 251.42 (249.3 to 253.8), 1,002 W, 2,000 MHz; 9 accepted, 0 rejected | 245.14 (239.5 to 252.2), 977 W, 1,948 MHz; 6 accepted, 0 rejected |
| Five cards on the pool, 40 minutes | | 201.6 TH/s, 40.3 a card; 52 accepted, 0 rejected | |

- The first version was 2.5 times the first fold at 9% more power, and a
  twin run on a cool card was within 0.1%.
- **The rig cannot cool even that for long.** The six cards sit in adjacent slots.
  The twin six-card run was the third in a row, and cards 0, 1 and 2 lost
  clock in it: card 0 fell from 2,000 to 1,750 MHz and from 168 to 140 W.
  After 40 minutes with five cards, cards 1, 2 and 3 were at 1,800, 1,850 and
  1,950 MHz and the outer two at 2,000. The rate follows the clock. The first
  fold, at 154 W, held 2,000 MHz in its runs. This kernel has no temperature
  reading for Arc, and the fans were near 2,100 rpm.
- The card's memory: `core.device` now carries `memTotalMb` and `memUsedMb`
  (8128 and 864 on an A750, with either fold).

### Where its time goes now

`pearl_sycl_peak --frag` has the fold's grid and loads with parts taken out.
Timing only: except for the fold's copy these rows compute wrong results on
purpose, and none is a mining rate. Card 0 or 1, 3 s a row, T-MAC/s, and the
same per EU and clock where the clock moved:

| Row | What runs | T-MAC/s |
|---|---|---|
| `f-regs` | DPAS alone, 16 accumulators | 113.8 |
| `f-fold` | the probe's copy of the fold | 42.2 |
| `f-loads` | its loads and DPAS, no readout | 42.8 |
| `f-win0`, `f-fixed` | every window reading window 0; every load the same fragment | 43.3, 43.3 |
| `f-a` | A' loaded, B in registers (2 loads a k-step) | 100.7 |
| `f-b` | B' loaded, A in registers (8 loads) | 50.1 |
| `f-pf3` | all ten loads sent first and the 16 DPAS run as one unbroken run | 43.4 |
| `f-keep` | the ten loads only kept by an XOR, the DPAS on register operands | 42.2 |
| `f-alu` | no loads: every operand rewritten by an add each k-step | 106.9 |

- **It is still the loads, and now it is the load message itself.** Memory
  and cache no longer matter (window 0 is no faster), nor does the order of
  loads and DPAS, nor whether a load's result goes into a DPAS at all. A
  k-step gets its ten 256-byte loads at one rate whatever else happens: about
  1.3e10 messages a second for the card, 4 clocks a message for each group
  of 16 EUs. Rewriting the same registers by arithmetic costs 6%.
- **A 128-byte load is served about four times as fast.** The first fold's
  B' loads, two halves a fragment, ran at 5.5e10 messages a second from
  cached memory. Loading each fragment as two vectors of 4 dwords:

| Row | What runs | T-MAC/s (per EU and clock) |
|---|---|---|
| `f-v4-b-pf3` | B' as two halves a fragment, all sent first; A in registers | 80.5 (96.1) at 1,869 MHz, 182 W |
| `f-v84-pf3` | the same with A' loaded as one vector | 61.4 (71.6) at 1,912 MHz, 182 W |
| `f-v4-pf3` | both operands as halves: 20 messages a k-step | 54.8 (61.2) |
| `g-loads` | 32 rows by 32 columns a sub-group, one vector a fragment: 8 messages a k-step | 57.3 (63.9) |
| `g-v4-pf3` | 32 by 32, halves, all sent first | 54.3 (60.5); 77.6 (89.1) with every window reading window 0 |

- One model fits every row: a 256-byte load costs the data port of a group
  of 16 EUs about 4.3 clocks and a 128-byte load about 1. The fold as built
  (`f-v84-pf3`, 2 and 16) is 24.6 a k-step, which alone holds it near 74
  T-MAC/s at 2,000 MHz.
- These rows run the card at its 190 W limit, and it drops its clock to stay
  there, so a faster fold is held by power as well as by loads.
- Tried in the fold itself and left out: A' loaded a k-step ahead as one
  vector (63.2 TH/s against 63.9), and the same in halves, which ran out of
  registers (3 TH/s).
- Larger work-groups (a band's windows in one, so every sub-group of it reads
  the same B') change nothing. Spacing a lane's bytes so each lane has a
  cache line of its own halves the rate. One vector of 4 qwords is no faster
  than one of 8 dwords.

## The build a release carries

A release carries this core as a folder, `sycl/`, beside its other cores: in
the Windows installer and the AppImage (`resources/native/sycl`), beside the
Linux CLI binary, in the HiveOS package, and on the release page as
`pearl_core_sycl-linux.tar.gz`. `native-core.yml` builds it for Linux and for
Windows; `miner-build.yml` stages it. It is the `--jit` form, built with the
open-source DPC++.

**Why that form.** The first hardware runs used kernels prebuilt by ocloc with
Intel's oneAPI compiler. Three things in that cannot ship:

- Prebuilt kernels are tied to the compiler of one driver release, there is no
  ocloc on a Windows runner, and Intel builds ocloc for Ubuntu 24.04 and newer.
  In the `--jit` form the kernels are SPIR-V, and the rig's own driver compiles
  each the first time it is used (a few seconds at the first start).
- The hardware DPAS was reached through a SPIR-V instruction that Intel's 24.35
  driver, from 2024, refuses ("unknown extension"). It is now the OpenCL built-in
  `intel_sub_group_i8_i8_matrix_mad_k32`, called by name, which Intel's
  compiler resolves in old and new drivers alike. A hardware fold's kernel has
  nothing else in it, so every kernel is a device image of its own and the
  host launches a hardware fold only on a card whose architecture has XMX at
  that sub-group size. The prebuilt form calls the same built-in and compiles
  to the same code: the compile gate's rows did not change.
- oneAPI's runtime libraries come under Intel's end user licence for developer
  tools, and are not ours to hand out. See "Licences" below.

**What is in the folder.**

| File | Linux | Windows | Licence |
|---|---|---|---|
| `pearl_core_sycl.node`, the addon | 85 KB | 229 KB | ours (MIT) |
| the host and kernels: `libpearl_sycl.so`, `pearl_sycl.dll` | 406 KB | 368 KB | ours |
| `pearl_sycl_check`, to list the cards and check a fold by hand | 35 KB | 54 KB | ours |
| the SYCL runtime: `libsycl.so.9`, `sycl9.dll` | 7.5 MB | 4.4 MB | Apache-2.0 with LLVM exceptions |
| its Level Zero adapters (two): `libur_adapter_level_zero*.so.0`, `ur_adapter_level_zero*.dll` | 6.4 MB | 2.7 MB | the same |
| `libumf.so.1` (Linux), `ur_win_proxy_loader.dll` (Windows) | 0.8 MB | 31 KB | the same |
| `LICENSE-DPCPP.txt`, `THIRD-PARTY-DPCPP.txt`, `NOTICE.txt` | 26 KB | 26 KB | |
| **Total** | **15 MB** (5.1 MB packed) | **7.6 MB** | |

The OpenCL adapter is built into the runtime. On Linux each file finds the
others beside itself (`$ORIGIN`); Windows looks beside a DLL by itself.
`stage.sh` makes the folder and refuses one that names the build machine in a
run path, needs a library it does not carry, or links oneAPI's libraries.

**Licences.** What `stage.sh` copies comes from the open-source release of the
DPC++ compiler, github.com/intel/llvm v7.2.0, fetched by SHA-256
(`install-toolchain.sh --open`): the SYCL runtime, the Unified Runtime adapters
and the Unified Memory Framework, all under the Apache License 2.0 with LLVM
exceptions. Its text travels in the folder. That release builds other
projects' code into those files: hwloc (BSD-3-Clause) in the memory framework,
Zstandard (BSD-3-Clause), emhash (MIT) and LLVM's copy of xxHash
(BSD-2-Clause) in the runtime, and the part of oneAPI Level Zero (MIT) an
adapter uses to find the driver's loader. All may be handed on with their
notices, which are in `THIRD-PARTY-DPCPP.txt`. The list was read from the
files' symbols and the release's build scripts; Intel's release has no such
list of its own. The Level Zero loader and the OpenCL loader are not carried:
the runtime opens them by name and the driver brings them. Not carried, and
not allowed to be: anything from Intel's oneAPI compiler (`libsvml`, `libimf`, `libirng`,
`libintlc`, and its builds of the same runtime), which is under Intel's end
user licence for developer tools. That compiler is still what the compile gate
runs, since ocloc needs it; nothing it makes is staged. CI also uses Intel's
OpenCL CPU runtime to check the folds, and does not stage that either. On
Windows the core needs Microsoft's Visual C++ runtime, which is not carried.

**What a rig needs.**

| | Needs | Checked |
|---|---|---|
| Linux | glibc 2.27 and GCC 7's libstdc++ or newer: Ubuntu 18.04 and later. CI holds every file to Ubuntu 20.04's | built in Ubuntu 20.04, run on 22.04 and 24.04 |
| Linux driver | Intel's compute runtime: Level Zero (`libze-intel-gpu1` with `libze1`) or OpenCL (`intel-opencl-icd`), and a user in the `render` group. The kernel's i915 or xe driver alone is not enough | Level Zero and OpenCL 26.40 (Ubuntu 24.04), and OpenCL 24.35 with no Level Zero, which is what the rig's Ubuntu 22.04 host has |
| Windows | Intel's Arc graphics driver, and the Visual C++ runtime | **no Intel card has run it**: the SYCL CPU device only |

**On the rig, 10 November.** Card 1 unless said; TH/s is the fold alone.

| Where it ran | Driver | Result |
|---|---|---|
| Ubuntu 24.04 container | Level Zero 26.40 | `verify.sh --quick` PASS; 16.81 TH/s |
| the same | OpenCL 26.40 | 16.78 TH/s |
| Ubuntu 22.04 container | OpenCL 24.35 | hardware fold right on every region; 10.13 TH/s |
| **the rig's own HiveOS userland, no container** | OpenCL 24.35, no Level Zero | `verify.sh --quick` PASS on card 0; 10.2 TH/s; plain fold 2.4 |
| the same, six cards on a pool, 5 minutes | | 60.6 TH/s, 916 W, 2 shares accepted, 0 rejected |
| the same, the CLI from source with no core path and no card list | | it named the rig "6x Intel Arc A750", started six cores: 60.6 TH/s, 2 accepted, 0 rejected |
| the same, the HiveOS package built from this branch, its own binary, `sycl/` beside it | | 60.6 TH/s, 1 accepted, 0 rejected |
| the same, **the HiveOS package CI built** (the installer workflow, run by hand on the branch), its own binary | | 60.7 TH/s, 912 W, 2 accepted, 0 rejected |

- **A HiveOS Arc rig can run it as it is**, with the Intel OpenCL packages
  HiveOS has. Nothing was installed on the host.
- **The 2024 driver is 40% slower**, at the same power. Its compiler does not
  merge the fold's loads: it sends 8 load messages for each DPAS where the
  2026 one sends 3 ("The A750's ceiling"). The next fold's vector loads would
  not depend on that. A rig that can run a 2026 driver should.
- **The reference code is right with the old compiler** and wrong with the
  new one (`xmx8-emu`, 16,384 of 16,384 with 24.35). So the mismatch in
  "Results on an Arc A750 rig" is the newer compiler's doing. It still mines
  nothing.
- CI checks both systems' staged folders on the SYCL CPU device: every region
  against the host, and 100 hits a fold recomputed in JS. Its HiveOS package
  job installs the package in Ubuntu 20.04, 22.04 and 24.04 containers and
  starts it with `--core intel`: the core loads from `sycl/` and says it found
  no Intel GPU, which is right on a runner.
- The Windows installer CI built has the folder in `resources/native/sycl`.
  Unpacked on a PC with an RTX card and no Intel GPU, the app's loader picked
  the CUDA build and never opened the folder; asked for by name, the Intel
  core loaded and said "no Intel GPU found through Level Zero or OpenCL".

In the app, the Intel core is tried only on a rig where nvidia-smi lists no
NVIDIA card and the system lists Arc cards and no AMD card the AMD core mines
on, or with `--core intel` (`PEARL_CORE_VARIANT=intel`). Every Intel card
still mines with each hit checked ("Untested cards"). The app shows no
temperature and no memory figure for an Arc card: the i915 and xe drivers
publish neither where it looks.

## Results from the container (2026-10-09)

No Intel GPU was available, so correctness was shown on the SYCL CPU device (the
same kernel source, run by Intel's OpenCL CPU runtime) and the GPU targets were
compiled but not run.

Correctness (`PEARL_SYCL_DEVICE=cpu ./verify.sh` and the runs below), re-run
after the review fixes. A hit ratio is hits found over hits expected from the
work the core reported (see verify-hits.js); the hits themselves are the same
on every run, since the job and salts are fixed.

| Check | Fold | Result |
|---|---|---|
| fold check, every region against a host fold, m = n = 1024 | dot, xmx16, xmx8 | 4,096 of 4,096 transcripts match each; 20 of 20 hits |
| fold check at the mainnet m and n, the last column batch (now part of `verify.sh`) | dot, xmx16 | 32,768 of 32,768 match each; 46 of 46 hits |
| fold check at the mainnet m and n, the first 640 batches of 4 column offsets (the regions the mainnet verify-hits run below searches) | dot | 20,971,520 of 20,971,520 match; 70 of 70 hits. 80 is the average for that many regions, so the 0.82 ratio below is this job's luck, not lost hits |
| verify-hits.js, m = n = 4096, constant fill | dot | 400 of 400 hits verified, 6 salts, hit ratio 1.02 |
| same | xmx16 | 400 of 400, 6 salts, 1.01 |
| same | xmx8 | 400 of 400, 6 salts, 1.01 |
| same, hashed fill | dot | 400 of 400, 7 salts, 0.95 |
| verify-hits.js at the mainnet m and n (col_batch 4) | dot | 60 of 60, 1 salt (a salt is 4,096 batches here), 0.82 |
| verify-hits.js, m = 131072, n = 64: a new salt every batch, mainnet-size A | dot | 100 of 100, 44 salts, 0.90 |
| the same fold check with a `--jit` build | dot, xmx16, xmx8 | 4,096 of 4,096 each |
| a build with no CPU kernels, run on the CPU device | | stops at start: "this build has no kernels for ..." |

1,760 hits were recomputed from scratch, and none failed.

On the CPU device the XMX folds run the DPAS specification's reference code,
so these show the fold's tiling, layout and readout are right. Whether Intel's
hardware DPAS matches that reference code is the first thing to check on a
GPU (`verify.sh` does).

Compile gate (`./compile-gate.sh`, DPC++ 2026.1.1, ocloc 26.31, IGC 2.40.13).
Hardware XMX fold of each target:

| Target | Platform | Fold | SIMD | GRF | DPAS | Spill |
|---|---|---|---|---|---|---|
| intel_gpu_bmg_g21 (B580, B570, Pro B50, B60) | Xe2 | xmx16 | 16 | 128 | 32 | 0 |
| intel_gpu_bmg_g31 (Pro B65, B70) | Xe2 | xmx16 | 16 | 128 | 32 | 0 |
| intel_gpu_lnl_m (Lunar Lake) | Xe2 | xmx16 | 16 | 128 | 32 | 0 |
| intel_gpu_ptl_h (Panther Lake) | Xe3 | xmx16 | 16 | 128 | 32 | 0 |
| intel_gpu_ptl_u (Panther Lake) | Xe3 | xmx16 | 16 | 128 | 32 | 0 |
| intel_gpu_pvc (Max 1550, 1100) | Xe-HPC | xmx16 | 16 | 128 | 32 | 0 |
| intel_gpu_acm_g10 (A770, A750, A580, Flex 170) | Xe-HPG | xmx8 | 8 | 128 | 17 (loop kept) | 0 |
| intel_gpu_acm_g11 (A380, A310, Flex 140) | Xe-HPG | xmx8 | 8 | 128 | 17 | 0 |
| intel_gpu_acm_g12 (Pro A60) | Xe-HPG | xmx8 | 8 | 128 | 17 | 0 |
| spir64 (`--jit`) | any | dot only | | | | |

The plain fold builds without a spill on every target too. The default build
(`./build.sh`) has kernels for ten targets, the nine GPU targets above and the
CPU device, and takes about a minute here. Not built: `intel_gpu_arl_h` (Arrow
Lake-H); this ocloc cannot target it.

## Results on the CPU device (10 November 2026)

Earlier the same day, before the rig. The port on the app as of v0.6.16: that
day's `pearl_core.cc` and `pearl_config.h`, unchanged, against the updated
host. No Intel GPU: an
Ubuntu 24.04 container on a rented PC (AMD Ryzen 5 2600, 12 threads), the SYCL
CPU device, DPC++ 2026.1.1, ocloc 26.40.40196.15, IGC 2.42.14.

| Check | Result |
|---|---|
| `./install-toolchain.sh --ocloc` | installs, in about 4 minutes; 2.5 GB. Before its fix it stopped with a 404 ("Compiler") |
| `./build.sh`: ten targets, the addon and the check tool | builds, 85 s. The addon linking is what shows the host has every function `pearl_core.cc` calls |
| `PEARL_SYCL_DEVICE=cpu ./verify.sh` | PASS, 105 s |
| its fold check, m = n = 1024: dot, xmx16, xmx8 | 4,096 of 4,096 transcripts match each; 20 of 20 hits |
| its fold check at the mainnet m and n, the last column batch: dot, xmx16 | 32,768 of 32,768 match each; 46 of 46 hits, as in October |
| its verify-hits.js runs, now with the app's own check (`checkHit.js`), m = n = 4096: dot, xmx16, xmx8 | 400 of 400 hits verified each, from 6 salts each; hit ratios 1.04, 1.07 and 1.03 |
| `PEARL_SYCL_DEVICE=cpu ./verify.sh --quick` | PASS, 62 s: 100 of 100 hits each, from 3 salts each; hit ratios 1.02, 1.02 and 1.06. It could not pass before (the end of "Kept up with the CUDA host") |
| the addon opened as the app opens it (`cardK`, `checkHits`, a shape trial asked for, a slice of the salts) | k 2048, no note, no shapes, no trial; the first hit is from the slice's first salt; `core.device.fold` names the fold |
| `pearl_sycl_check --list`, `PEARL_GPU_INDEX`, `ONEAPI_DEVICE_SELECTOR` | the list has the CPU device as index 0; index 5 is refused with the list's size; a selector that leaves no device leaves none |
| `bench.js`, m = n = 4096 | runs: about 0.013 TH/s on that CPU, which says nothing about a GPU |
| `./compile-gate.sh` | passes (table below) |
| `./build.sh --targets intel_gpu_acm_g10,spir64_x86_64`, the A750's build, and the fold check with it | builds in 45 s, its largest process at 360 MB; 4,096 of 4,096 each |
| `./build.sh --jit` and the fold check with it | builds in 12 s; 4,096 of 4,096 each |
| Intel's driver packages installed beside it (`libze-intel-gpu1`, `libze1`, `intel-opencl-icd`, all 26.40.40196.15) | they install in the container, and the fold check still passes |
| `pool-run.sh` with no address | gets as far as it can: Node check, the CLI's two packages, the CLI starting and refusing the address |

Not checked there, because each needs a GPU: Level Zero, the hardware DPAS,
`core.untested` being true, speed and a pool run. The rig checked those the
same evening. Still not run anywhere: the refusal of a GPU for an opener that
checks nothing. Its rule is run in JS by `earn/test/syclCore.test.js`.

The toolchain's `sycl-ls` printed nothing there, though the CPU device was
found by everything else. So the steps below list devices with `clinfo -l` and
`pearl_sycl_check --list`, not with `sycl-ls`.

Compile gate with ocloc 26.40.40196.15 and IGC 2.42.14. Hardware XMX fold of
each target. The DPAS counts are the same as in October, and again nothing
spills.

| Targets | Platform | Fold | SIMD | GRF | DPAS | Spill | Instructions |
|---|---|---|---|---|---|---|---|
| intel_gpu_bmg_g21, bmg_g31, lnl_m | Xe2 | xmx16 | 16 | 128 | 32 | 0 | 1,584 |
| intel_gpu_ptl_h, ptl_u | Xe3 | xmx16 | 16 | 128 | 32 | 0 | 1,549 |
| intel_gpu_pvc | Xe-HPC | xmx16 | 16 | 128 | 32 | 0 | 2,425 |
| intel_gpu_acm_g10, acm_g11, acm_g12 | Xe-HPG | xmx8 | 8 | 128 | 17 | 0 | 1,563 |

The plain fold has no spill on any target either.

## On an Intel box

Linux with an Intel GPU, Node 22 or later (what the CLI is built and tested
on), and this repository. None of the steps below has been run on a GPU yet.
The driver steps follow Intel's install guides at dgpu-docs.intel.com (read on
2026-10-09). The compiler and build steps were run in a container with no GPU,
using Ubuntu 24.04 packages.

Intel builds its driver packages and ocloc for Ubuntu 24.04 and 26.04, and they
need glibc 2.38 or newer. On an older system, such as HiveOS's Ubuntu 20.04
image, run everything in an Ubuntu 24.04 container that is given `/dev/dri`:
"Running it on an Arc A750 rig" has the commands.

All of this is for building the core here. The build a release carries needs
no compiler, no ocloc and no container, and runs with an older driver ("The
build a release carries").

### 1. Driver

First check that the kernel is new enough for the card. Without that, the card
is not driven, and `clinfo` lists nothing.

```
lspci -nn | grep -Ei 'VGA|DISPLAY'    # the card and its PCI id
uname -r                              # the running kernel
ls /dev/dri                           # needs a renderD* entry
```

| Card | Kernel driver | Kernel it needs (Intel's table) |
|---|---|---|
| Arc A-series, Arc Pro A60 | i915 | 6.2 (Ubuntu 24.04's stock 6.8 is fine) |
| Lunar Lake | xe | 6.11 |
| Arc B580, B570 | xe | 6.12 |
| Arc Pro B50 | xe | 6.14 |
| Arc Pro B60 | xe | 6.15 |
| Arc Pro B65, B70; Panther Lake | xe | 6.17 |
| Data Center GPU Max 1550, 1100; Flex 170, 140 | Intel's out-of-tree i915 (`intel-i915-dkms`) | what the LTS stack below supports: Ubuntu Server 22.04 or 24.04 |

Ubuntu 24.04 ships with kernel 6.8, which drives none of the xe cards above. On
24.04, install the hardware enablement kernel and check `uname -r` again, or use
Ubuntu 26.04, whose kernel drives them as installed:

```
sudo apt install -y --install-recommends linux-generic-hwe-24.04 && sudo reboot
```

Then the compute runtime with Level Zero. **Arc, Arc Pro, Lunar Lake and
Panther Lake** (Ubuntu 24.04 or 26.04, Intel's PPA):

```
sudo add-apt-repository -y ppa:kobuk-team/intel-graphics
sudo apt install -y libze-intel-gpu1 libze1 intel-opencl-icd intel-ocloc clinfo
sudo gpasswd -a $USER render     # then log out and back in
```

**Data Center GPU Max and Flex** need Intel's data-center stack instead, not the
PPA. On a cloud VM it is usually installed already: if `clinfo -l` lists the
card, skip this. Otherwise, on Ubuntu Server 22.04 or 24.04 (Intel's LTS 2523
release):

```
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key \
  | sudo gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg
. /etc/os-release
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu ${VERSION_CODENAME}/lts/2523 unified" \
  | sudo tee /etc/apt/sources.list.d/intel-gpu-${VERSION_CODENAME}.list
sudo apt update
sudo apt install -y linux-headers-$(uname -r) linux-modules-extra-$(uname -r) flex bison \
  intel-fw-gpu intel-i915-dkms xpu-smi
sudo reboot
sudo apt install -y intel-opencl-icd libze-intel-gpu1 libze1 clinfo
sudo gpasswd -a $USER render     # then log out and back in
```

Check with `clinfo -l`: it should list the card. If it does not, the usual
causes are a kernel older than the table says, a user not yet in the render
group (log out and back in), or, on Max and Flex, the i915 module from
`intel-i915-dkms` not loaded (`lsmod | grep i915`, `sudo dmesg | grep -i i915`).

### 2. Compiler

`install-toolchain.sh --ocloc` installs DPC++ 2026.1.1 from conda-forge, and
the ocloc and IGC that Intel's PPA has that day for the box's Ubuntu release
(24.04 or 26.04), without root and without touching the driver:

```
cd earn/native/sycl
./install-toolchain.sh --ocloc    # into ./toolchain, 2.5 GB on disk
. ./toolchain/envrc.sh
```

It prints the ocloc packages it took. On 10 November 2026 those were ocloc
26.40.40196.15 and IGC 2.42.14, the same versions the driver packages of step
1 had that day. October's builds were made with ocloc 26.31 and IGC 2.40.13,
which the script then asked for by file name. The PPA keeps only its newest
builds, so a month later those files were gone and the script stopped with a
404. It now reads the PPA's package index instead.

oneAPI works too (`source /opt/intel/oneapi/setvars.sh`), with ocloc from the
driver packages. But an older compiler or ocloc may not know `bmg_g31`, `ptl_h`
or `ptl_u`, and then the whole default build fails. Intel's GPU repository,
where the Max and Flex stack comes from, had no ocloc newer than 25.18 when
checked. If that happens, build for your card only, as in "Which build".

The build records the compiler's lib directory in the addon's run path, so the
addon finds the SYCL runtime without the environment script. Only building, and
the CPU device (`PEARL_SYCL_DEVICE=cpu`, which needs `OCL_ICD_VENDORS` from
`envrc.sh`), need it sourced. `envrc.sh` keeps the system's OpenCL drivers
listed, so `clinfo` in that shell still shows the card.

### Which build

`./build.sh` with no options compiles the kernels ahead of time for the GPUs
below plus the CPU device. A GPU not in the table has no kernels in that build:
the core then stops at start with "this build has no kernels for ...". Build
with `--jit` for those.

| Card | `--targets` entry | Fold it mines with |
|---|---|---|
| Arc B580, B570, Arc Pro B50, B60 | `intel_gpu_bmg_g21` | xmx16 |
| Arc Pro B65, B70 | `intel_gpu_bmg_g31` | xmx16 |
| Lunar Lake | `intel_gpu_lnl_m` | xmx16 |
| Panther Lake | `intel_gpu_ptl_h`, `intel_gpu_ptl_u` | xmx16 |
| Data Center GPU Max 1550, 1100 | `intel_gpu_pvc` | xmx16 |
| Arc A770, A750, A580, Flex 170 | `intel_gpu_acm_g10` | xmx8 |
| Arc A380, A310, Flex 140 | `intel_gpu_acm_g11` | xmx8 |
| Arc Pro A60 | `intel_gpu_acm_g12` | xmx8 |
| Anything else: Meteor Lake, Arrow Lake, Iris Xe, DG1 | `./build.sh --jit` | dot |

To build for one card, which is also faster:
`./build.sh --targets intel_gpu_bmg_g21,spir64_x86_64` (keep `spir64_x86_64`
for the CPU device). If unsure which entry is yours, build them all.

**Data Center GPU Max 1550:** in Level Zero's default mode it shows as two GPUs,
one per stack, and one core mines one stack. `verify.sh` and `bench.sh` measure
one stack; the card is about twice that. To mine both, give the CLI both:
`PEARL_GPU_INDEX=0,1 ./pool-run.sh <address> 300`. The Max 1100 is one stack.
The same goes for a box with several Intel cards: `./build/pearl_sycl_check
--list` numbers them, and `PEARL_GPU_INDEX=N ./verify.sh` checks card N.

### 3. The four commands

All run from `earn/native/sycl`.

| Step | Command | Time on a GPU (rough) |
|---|---|---|
| Build | `./build.sh` | about 1 minute |
| Correctness: fold checks, then 400 hits with the core's fold and 400 with the plain fold | `./verify.sh` | a few minutes |
| Speed: each fold alone, then the full miner loop | `./bench.sh`, or `./bench.sh --sweep` for batch widths and bands too | about 2 minutes, 5 with `--sweep` |
| 300 s on the pool with the CLI | `./pool-run.sh <prl1p...address> 300` | 5 minutes |

Run them in that order and stop at the first failure.

## Test plan, and what to send back

1. **Build.** `./build.sh 2>&1 | tail -20`. If it fails, send the output, then
   try your card's target alone (see "Which build").
2. **See the device.** `./build/pearl_sycl_check --list` and
   `PEARL_SYCL_VERBOSE=1 ./build/pearl_sycl_check --m 1024 --n 1024 --batches 1 --folds auto`.
   The first lists the cards the core may open, each with its index. The
   second prints the card, the fold it picked (`xmx16 ... (hardware)` on
   Battlemage and Max, `xmx8 ... (hardware)` on Alchemist), the batch width and
   the band. If the list is empty, go back to "Driver".
3. **Correctness.** `./verify.sh 2>&1 | tee verify.log`. What each part means:
   - fold check, `dot`: the plain fold on the GPU against the host. If it
     fails, the problem is in the port, not in XMX.
   - fold check, `auto`: the card's own fold against the host, which on an
     XMX card is the hardware DPAS. If it fails, do not mine with that card.
   - "for information, not part of PASS": the reference code of that fold
     (`auto-emu`) against the host. It mines nothing. On an Arc A750 it
     fails while the hardware fold passes ("Results on an Arc A750 rig").
   - the same fold check at the mainnet m and n, on the last column batch:
     every region of it, so a fold that only loses hits at full size fails.
   - verify-hits, twice: 400 hits through the addon, each recomputed in JS.
     Its `hitRatio` (hits found over hits expected from the core's own rate)
     should be near 1, within about 0.85 to 1.15. Well under that means hits
     are being lost.
4. **Speed.** `./bench.sh --sweep 2>&1 | tee bench.log`, with nothing else on
   the card. Also send `intel_gpu_top` or `xpu-smi stats -d 0` output taken
   while it runs, if either is installed (clock and power).
5. **Pool.** `./pool-run.sh <address> 300` (on a Max 1550,
   `PEARL_GPU_INDEX=0,1 ./pool-run.sh <address> 300`). It writes
   `pool-run.log` and `pool-run-stats.json`.

Send back: the card's name, `uname -r`, the driver package versions
(`dpkg -l | grep -E 'libze|intel-opencl|ocloc|libigc|i915'`), the `--list`
output, and `verify.log`, `bench.log`, `pool-run.log` and `pool-run-stats.json`. With those,
the next round can target the right bottleneck.

For comparison: ARC-miner's release notes give 36.6 TH/s on a B580. That figure
is self-reported, and its repository has returned 404 since October 2026, so it
could not be checked. The B580's int8 XMX ceiling is about 116 T-MAC/s
(233 TOPS / 2).

## Running it on an Arc A750 rig (HiveOS)

The steps as they were run on 10 November on the rig in "Results on an Arc
A750 rig": six A750s on HiveOS 0.6-231, which there is Ubuntu 22.04.5 (glibc
2.35) with kernel 6.12.0-hiveos and the cards on `i915`. Everything worked as
written here.

It runs in an Ubuntu 24.04 container. Intel's driver packages for Arc, ocloc
and this build all need glibc 2.38, and HiveOS's own userland is older: its
Intel OpenCL was 24.35, with no Level Zero. (The release's build has since
mined on that userland as it is, at 60% of the container's speed: "The build
a release carries".) The container gets the cards'
device nodes and one work folder, and nothing else of the host's. It is not
privileged. Removing it and the folder undoes everything. The login needs to
be in the `docker` group, or root; Docker was already on the rig (29.1.3).

**0. On the rig: free the cards and write down what the rig is.** The cards
must be idle, so stop whatever mines on them first (HiveOS: `miner stop`, and
`miner start` when done).

```sh
mkdir -p ~/arc/out && cd ~/arc
( uname -r; . /etc/os-release; echo "$PRETTY_NAME"; ldd --version | head -1
  lspci -nnk | grep -A3 -Ei 'vga|display'
  ls -l /dev/dri; df -h ~; free -m | head -2; nproc; grep -m1 'model name' /proc/cpuinfo
  grep -H . /sys/class/drm/card0/engine/*/preempt_timeout_ms
) 2>&1 | tee out/arc-facts.txt
```

It needs the A750s (`[8086:56a1]`) with `Kernel driver in use: i915`, a
`renderD` entry for each under `/dev/dri`, 8 GB of free disk and 4 GB of RAM.
The rig had 7.8 GB of RAM and no swap, and never ran short.

**1. The container.**

```sh
docker run -d --name arc --memory=6g --memory-swap=6g --device /dev/dri \
  -v ~/arc:/work -w /work ubuntu:24.04 sleep infinity
docker exec -it arc bash
```

Every later step runs in that shell. After a dropped SSH session, run the
last line again, then the three lines marked "each new shell". Files the
container writes under `~/arc` belong to root.

**2. In the container: Intel's driver, Node, this repository, the compiler.**
Under 3 minutes on the rig.

```sh
export DEBIAN_FRONTEND=noninteractive
echo 'Acquire::ForceIPv4 "true";' > /etc/apt/apt.conf.d/99force-ipv4   # HiveOS boots with IPv6 off
apt-get update && apt-get install -y --no-install-recommends software-properties-common gpg-agent \
  ca-certificates curl bzip2 xz-utils git g++ make pciutils
add-apt-repository -y ppa:kobuk-team/intel-graphics
apt-get install -y --no-install-recommends libze-intel-gpu1 libze1 intel-opencl-icd clinfo
clinfo -l                                   # lists every A750
mkdir -p /opt/node && curl -fsSL https://nodejs.org/dist/v22.22.0/node-v22.22.0-linux-x64.tar.gz \
  | tar xz -C /opt/node --strip-components=1
git clone --depth 1 https://github.com/super3/llmjob.git /work/llmjob
export PATH=/opt/node/bin:$PATH             # each new shell
cd /work/llmjob/earn/native/sycl            # each new shell
./install-toolchain.sh --ocloc 2>&1 | tail -8
. ./toolchain/envrc.sh                      # each new shell
```

**3. Build for the A750 and check every card.** The build took 31 s and 300
MB. A card's quick verify took 112 s.

```sh
./build.sh --targets intel_gpu_acm_g10,spir64_x86_64 2>&1 | tail -4
./build/pearl_sycl_check --list | tee /work/out/devices.txt
for i in 0 1 2 3 4 5; do PEARL_GPU_INDEX=$i ./verify.sh --quick 2>&1 | tee /work/out/verify-$i.log; done
grep -c '== PASS' /work/out/verify-?.log    # 1 for each card
PEARL_GPU_INDEX=0 ./verify.sh 2>&1 | tee /work/out/verify-full-0.log    # 400 hits a fold, about 3 minutes
```

`--list` prints the cards the core may open, each with the index the loop
uses. On the rig: six lines of `N: Intel(R) Arc(TM) A750 Graphics
[level_zero], auto fold xmx8f (hardware DPAS)` (`xmx8` on 10 November, before
the fragment fold). If there is another Intel GPU
in the list (a motherboard's own graphics), leave its index out of the loops.

Each log names the card and its fold, `xmx8 (DPAS 8x8x32, sub-group 8)
(hardware)`, then has one JSON line a check. The lines that count have
`"PASS":true`: the plain fold and the hardware DPAS, each against the host, at
a small size and at the mainnet size, then the hits recomputed in JS with each
of the two folds. One line is `"PASS":false` and expected on an A750: the
reference code, `xmx8-emu`, under "for information, not part of PASS".

**4. Speed, on one card.** About 5 minutes with the sweep.

```sh
PEARL_GPU_INDEX=0 ./bench.sh --sweep 2>&1 | tee /work/out/bench-0.log
```

To read a card's power and clock while it runs, on the host:
`cat /sys/class/drm/card0/gt_act_freq_mhz`, and the difference of two reads of
`/sys/class/drm/card0/device/hwmon/hwmon*/energy1_input` (microjoules).

**5. Five minutes on the pool with the cards that passed step 3.** List only
those. `PEARL_POOL` and `PEARL_WORKER` are optional: without them the CLI
mines on HeroMiners under its own worker name.

```sh
PEARL_GPU_INDEX=0,1,2,3,4,5 PEARL_POOL=prl-us.kryptex.network:7048 PEARL_WORKER=arc750 \
  ./pool-run.sh <prl1p... address> 300
cp pool-run.log pool-run-stats.json /work/out/
```

The log names each card (`mining on GPU 0 · Intel(R) Arc(TM) A750 Graphics`),
says once a card that its hits are checked before they are sent and that the
first was right, and counts the shares. A wrong hit is never sent: the CLI
turns that card off and logs why. On the rig: 5 accepted and 0 rejected in
the first run, 15 and 0 over the night's six runs.

**6. When done.** `docker rm -f arc` and remove `~/arc` (3 GB; its files are
root's, so remove them from a container: `docker run --rm -v ~/arc:/w
ubuntu:24.04 rm -rf /w/llmjob /w/out`). Then start the miner again.

What the plan expected to go wrong, for the next rig. None of it happened on
this one:

| What you see | What it probably is | What to do |
|---|---|---|
| `clinfo -l` lists no A750 | the container cannot reach the cards, or Intel's runtime does not accept this kernel's driver (`xe` on an A750 is not Intel's supported pairing) | `ls -l /dev/dri` in the container, and step 0's facts |
| `--list` says `[opencl]`, not `[level_zero]` | the Level Zero driver did not load | go on: the core takes the cards through OpenCL, where the folds were right too |
| `--list` says `auto fold dot` for an A750 | the runtime could not tell the card's architecture, so the core will not use DPAS | step 3 then checks the plain fold only |
| "this build has no kernels for ...", or a "SYCL error" at the first check | the index is another GPU, or the driver will not take the prebuilt kernels | check the index against `--list`. Else `./build.sh --jit --out build-jit`, then step 3's loop with `PEARL_SYCL_BUILD=$PWD/build-jit` in front of `./verify.sh`: the driver compiles the kernels itself, and the plain fold is checked |
| a fold check prints nothing for five minutes | Level Zero compute has hung on an A750 with some kernels from 6.6 on (drm/i915 issue 10955) | Ctrl-C, then the same command with `ONEAPI_DEVICE_SELECTOR=opencl:gpu` in front |
| "SYCL error", a lost device, or `GPU HANG` in dmesg | a batch ran longer than the driver's preempt timeout (7.5 s for compute on the rig; step 0 prints it). The rig's longest batch was 0.33 s | `PEARL_SYCL_COL_BATCH=32` in front of the command |
| `"fold":"xmx8"` or `"fold":"dot"` fails | the fold is wrong on that card or driver | do not run step 5. Send the log |
| `hitRatio` well under 0.85 over 400 hits | hits are being lost | do not run step 5. Send the log |
| step 5: "a hit came out wrong" | the fold is wrong on that card, and the CLI caught it | send `pool-run.log` |

## Switches

| Variable | Default | Meaning |
|---|---|---|
| `PEARL_CORE_PATH` | | names a local build to the app and the CLI: point it at `build/pearl_core_sycl.node`. Without it they look in the release's `sycl/` folder |
| `PEARL_CORE_VARIANT` | | `intel` asks the app for this core on any rig, with no fallback; the CLI's `--core intel` is the same. Without it, it is tried on a rig of Arc cards with no NVIDIA or AMD card |
| `PEARL_SYCL_DEVICE` | `gpu` | `cpu` takes the SYCL CPU device (the correctness gate without a GPU) |
| `PEARL_SYCL_FOLD` | `auto` | `dot`, `xmx16`, `xmx8`, `xmx8f`, `xmx16-emu`, `xmx8-emu`, `xmx8f-emu`. `auto` is the hardware DPAS where the build has it for the device (`xmx8f` on Alchemist, `xmx16` on Battlemage and newer), else `dot` |
| `PEARL_SYCL_COL_BATCH` | 256 | column offsets a batch (a power of two; at most 2^25 regions a batch, which is 4,096 at mainnet) |
| `PEARL_SYCL_BAND` | from the cache size | row windows walked together. By default their A' is a quarter of the cache the device reports for the fragment fold and the plain fold (64 on an A750) and all of it for `xmx8` and `xmx16` (256) |
| `PEARL_SYCL_VERBOSE` | | print the device, fold, batch width and band when a core starts |
| `PEARL_SYCL_UNTESTED` | | `1` opens a GPU for an opener that checks no hits and sends no shares (`bench.js`). It turns no check off ("Untested cards") |
| `PEARL_GPU_INDEX` | | the app's and the CLI's: the cards to mine on, by index in the core's device list (`pearl_sycl_check --list` prints it), such as `0,1,2,3,4,5`. As one number it also pins the card for `verify.sh`, `bench.sh` and the tools under them |
| `ONEAPI_DEVICE_SELECTOR` | | the SYCL runtime's own: `opencl:gpu` hides the Level Zero driver, so the core takes the cards through OpenCL |

`./build.sh --jit` builds SPIR-V that the driver compiles at first run, a
device image a kernel. It is the form a release carries. It runs on any Intel
GPU, with the hardware DPAS on the cards that have XMX and `dot` on the rest
(Iris Xe, Arc in Meteor Lake).

## How it works

Files:

| File | What |
|---|---|
| `pearl_sycl_host.cpp` | the `pearl_host_*` API on SYCL: device choice, memory, the draw, the pipeline, share proofs |
| `pearl_sycl_kernels.hpp` | the kernels: draw ports of `pearl_kernel.cu`, and both folds |
| `pearl_sycl_blake3.h` | BLAKE3, one copy for the host and the device |
| `pearl_sycl_check.cpp` | fold check, fold bench and device list (`--list`), no Node |
| `pearl_sycl_peak.cpp` | the XMX peak probe: DPAS with nothing in the way, then the fold's parts one at a time. Timing only (`./build.sh --peak`) |
| `verify-hits.js` | `probes/verify-hits.js` for any core, m and n |
| `bench.js` | the full miner loop's rate |
| `build.sh`, `verify.sh`, `bench.sh`, `pool-run.sh` | the four commands |
| `compile-gate.sh`, `isa-report.js` | per-target compile and IGC assembly report |
| `install-toolchain.sh` | DPC++ from conda-forge and, with `--ocloc`, ocloc from Intel's PPA; no root |

The draw is the CUDA core's: the same job_key, fills, stamp, trees, seeds and
noise, kernel for kernel. Two things are simpler than `pearl_host.cu`, on
purpose:

- A same-job redraw rebuilds A's whole tree on the device and waits for the
  root, where the CUDA host repairs one path and hashes the new a_seed itself.
  One wait a salt; a salt is 64 batches at mainnet.
- Before a redraw rewrites A, the batches still queued are finished and their
  hits read, so a proof always comes from its own salt's tree.

The folds work in windows of 32 rows by 64 columns, which hold exactly 8 whole
regions. A batch is 256 column offsets by every row offset (2^21 regions at
mainnet), walked in bands of row windows sized from the cache the device
reports (see `PEARL_SYCL_BAND` under "Switches").

- `dot`: a work-group of 256 a window, 32 work-items a region, each with 4 rows
  by 2 columns. Each 128-k chunk is staged in local memory. A work-item XORs
  its 8 sums into one word a chunk; XOR is linear, so a region's 32 words are
  combined once at the end.
- `xmx16` / `xmx8`: DPAS 8x16x32 or 8x8x32, in the layout the
  `cl_intel_subgroup_matrix_multiply_accumulate` extension fixes. A window is
  2 sub-groups of 16 rows (sub-group 16) or 4 of 8 rows (sub-group 8), each
  holding 8 DPAS accumulators. Per chunk a lane XORs its accumulators into one
  word for each of its two regions, two (or one) lane shuffles gather a
  region's columns, and the sub-groups' words are XORed at the end. The DPAS
  call is the hardware instruction, reached through the OpenCL built-in
  `intel_sub_group_i8_i8_matrix_mad_k32`, or the specification's reference
  code. A prebuilt kernel chooses inside itself, with `if_architecture_is`. In
  the `--jit` form a hardware fold is a kernel of its own, with no reference
  code in it, and the host launches it only on a card whose architecture has
  XMX at that sub-group size; every other device gets the reference kernel.
  That reference code is what the CPU device checks.
- `xmx8f`: the same DPAS and the same readout at sub-group 8, with 16 rows a
  sub-group (2 sub-groups a window) and the noised operands in fragment order,
  A' loaded one vector a fragment and B' two, a k-step's loads sent before
  its DPAS ("The fragment fold"). The host draws the noised
  operands in the order its fold reads. The check tool runs any fold on one
  context, so a batch of a fold that wants the other order draws them again.

## Untested cards

Every Intel GPU counts as an untested card: none has mined with these folds in
a release. They were written against the CPU device, where DPAS is reference
code, and one kind of card has run them since, the Arc A750, where the
hardware fold was right and that reference code was not. If a fold is wrong on
a card, every hash it reports is wrong, and every one still meets the target,
so every share would be sent and rejected, and a pool can ban the address for
that. So the host follows the AMD host's rule for its own untested cards
(`../amd/README.md`):

- A GPU opens only for an opener that checks every hit before sending it
  (createCore's `checkHits`). The app and the CLI pass it, and so do
  `verify-hits.js` and `pearl_sycl_check`, which check everything themselves.
- The core then reports `core.untested`, and the miner recomputes each of that
  card's hits with `../../src/shared/miner/checkHit.js` (4.6 ms a hit). At the
  first wrong one it sends nothing, turns that card off for the run and logs a
  line asking for a report. That line names the AMD README and an "unknown
  architecture", since the app reads a `gfx` name out of the fold's
  description; the card's name in it is right.
- `PEARL_SYCL_UNTESTED=1` does not turn the check off. It only lets an opener
  that checks nothing open a GPU, which is for `bench.js`: it sends no shares.
  `bench.sh` sets it for that one line.
- The CPU device is not gated. Nobody mines on it.

A kind of card leaves the gate by a change to `pearl_host_create_checked` in
`pearl_sycl_host.cpp`, once it has passed "Test plan". The A750 has: six
cards, 3,000 hits recomputed, 15 shares accepted and none rejected. It is
still behind the gate, which costs 4.6 ms a share; taking it out is a
decision, not a fix.

## Kept up with the CUDA host

`pearl_core.cc` calls six host functions that did not exist when this port was
written. The host defines them now, each with the answer that changes nothing,
because one kind of Intel card has been timed and there is nothing to choose
between yet:

| The CUDA host | Here |
|---|---|
| `pearl_host_card_k`: a larger k on cards whose fold is faster there (the AMD host: RDNA 4) | the profile's k. Both folds are built for k 2048, and create refuses any other |
| Launch shapes, and the startup trial `pearl_core.cc` races them in (RTX 30) | no shapes, so no trial runs and the app caches no pick. The batch width and band come from the profile, the device's cache size, or `PEARL_SYCL_COL_BATCH` and `PEARL_SYCL_BAND` |
| `pearl_host_create_checked`: no NVIDIA card is untested | every GPU is ("Untested cards") |
| `pearl_host_fold_name`, shown as `core.device.fold` | the fold, whether DPAS is the hardware instruction, the batch width, the band and k |

Not ported, and not needed for the API: the CUDA host's restamp records (a
same-job redraw here still rebuilds A's tree and waits, see "How it works"),
its per-card batch and band rules, and its persistent L2 settings.
`earn/test/syclCore.test.js` compares the host's functions with
`pearl_core.cc`'s declarations type for type, so the next API change fails
that test rather than the link.

Three faults in the port's own scripts turned up on the way, and are fixed:

- `install-toolchain.sh --ocloc` asked Intel's PPA for ocloc and IGC by
  version, and those builds were gone a month later ("Compiler").
- `verify.sh --quick` could not pass: its 100 hits came from 2 salts, and
  verify-hits.js wants 3. It now halves the bound, so they come from 3 or 4.
- `verify-hits.js` checked each hit inside its event. On the CPU device, with
  hits arriving as fast as they were checked, the rate events waited behind
  them, and a run ended with no rate and so no hit ratio. A hit is now checked
  in a turn of the event loop of its own.

## Not done

- Only the Arc A750 has run it. Battlemage and newer (`xmx16` with the
  hardware instruction) have not, nor has any card on Windows. The built-in
  the hardware folds call has a sub-group 16 form that compiles for those
  cards (the compile gate) and has never run.
- Speed work. The fragment fold is at 56% of what the A750's matrix engines
  do on a cool card, and the rest is still operand loads ("Where its time
  goes now"). What the load messages cost says where the next gain is: A' as
  two 256-byte loads is a third of the fold's load time, and in halves a
  k-step has 20 loads where a thread can have 16 in flight. A sub-group of 32
  rows by 32 columns needs fewer loads a DPAS and timed at 89 T-MAC/s a clock
  from cached memory, but 60 to 67 from its own addresses. Any gain also has
  to come with fewer watts: the card is at its limit. Not timed: staging B
  through local memory, uniform block loads, and 2D block loads with L1
  prefetch on Xe2 and Xe-HPC (`SPV_INTEL_2d_block_io`), which Alchemist does
  not have.
- The fragment fold is sub-group 8 only. Battlemage and newer still mine with
  `xmx16`, the first fold's form, which no card has run.
- The restamp is the slow form described above (fine for a first version).
- A run on Windows with an Intel card. The Windows core is built and checked
  on the CPU device only, and the app calls it experimental.
- The CLI's self-updater does not fetch the folder. A rig on the HiveOS
  package gets it with each package; a rig that downloaded the binary alone
  unpacks `pearl_core_sycl-linux.tar.gz` beside it, and again after an update.
- Temperature and memory readings for an Arc card. The app's come from
  nvidia-smi and, for AMD, from sysfs files the Intel drivers do not have, so
  it shows none. A Max 1550 (two stacks, two devices) is not counted by the
  app, and runs one core unless `PEARL_GPU_INDEX` lists both.
- `../src/pearl_core.cc` keeps its class in one global (`__pearl_exports`),
  set by each addon as it loads. A process that has loaded two cores builds
  the last one's core from either addon's `createCore`. The app always uses
  the core it loaded last, except when it retries the first after both
  failed to start (the AMD core's retries have the same fault). The loader
  works round the one case it matters in for an Intel rig; the fix is to keep
  the class with the addon.
- Per-card k, launch shapes, and the CUDA host's batch and band rules ("Kept
  up with the CUDA host"). The A750 gave this host one band rule; the rest
  wait for a second kind of card to time them on.
- Arrow Lake-H (XMX at sub-group 8) is not a build target; it needs `--jit`
  and mines with `dot`.

## Risks

- The hardware DPAS at sub-group 16 (Battlemage and newer) has not run. The
  sub-group 8 one matched the host on the A750, which says nothing about the
  other's register layout. `verify.sh` finds a mismatch in the first minute.
- The sub-group 8 reference code is wrong on the A750 with the 2026 driver's
  compiler and right with the 2024 one's, so the fault is in how the newer
  compiler builds it. Which part of it is not known. It mines nothing, but it
  means the reference code is no witness for the hardware fold on that card:
  the host's scalar fold is.
- Clock and power under Pearl are known for the A750 only, and over five
  minutes: 2,000 MHz and about 150 W a card.
- IGC versions differ in code generation, and `install-toolchain.sh --ocloc`
  takes the version Intel's PPA has that day. The compile gate passed with IGC
  2.40.13 in October and 2.42.14 in November; `./compile-gate.sh` says what a
  later one makes of the folds.
- A driver whose IGC is not the ocloc's may refuse the prebuilt kernels. A
  `--jit` build, which the driver compiles itself, is the way round, with the
  plain fold only.
- Older compilers and ocloc may not know every default target (see "Compiler").
- A batch longer than the driver's job timeout would be killed. On i915 the
  preempt timeout for compute is 7.5 s by default
  (`/sys/class/drm/card*/engine/*/preempt_timeout_ms`). The default batch is
  short (0.2 s at 5 TH/s at mainnet); `PEARL_SYCL_COL_BATCH` changes it.
- Level Zero compute has hung on an A750 with some kernels from 6.6 on
  (drm/i915 issue 10955, not checked here). The core falls back to OpenCL when
  Level Zero lists no GPU, and `ONEAPI_DEVICE_SELECTOR=opencl:gpu` makes it.
- Most Arc owners use Windows, where no Intel card has run this core.
- A driver compiles the kernels itself, so a driver's compiler decides the
  speed and could get a fold wrong. Every hit is checked before it is sent,
  which catches the second. The first was 40% between two drivers on one card.
