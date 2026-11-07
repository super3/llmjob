# Pearl mining core for AMD GPUs (MVP)

A HIP port of the Pearl core in `../src`, for AMD Instinct and Radeon cards. It builds
`pearl_core_hip.node`, an addon with exactly the CUDA core's interface: `pearl_core.cc`
is compiled unchanged against a HIP host. The app tries it first when nvidia-smi lists no
NVIDIA card, and loads it with no fallback when asked to (`--core amd` or
`PEARL_CORE_VARIANT=amd`, see below). Linux releases carry it from v0.6.8, built by
`native-core.yml`'s `ubuntu-latest-hip` job. The Windows installer carries one built with
AMD's HIP SDK by its `windows-2022-hip` job, which no Windows Radeon has run yet.

**Status: mines on RDNA 3 and RDNA 4 cards; not yet run on CDNA (Instinct) or RDNA 2.** On
those (gfx9 and gfx103x) the app and the CLI mine with every hit checked: a fold whose layout
differs on the card would have every share rejected, and a pool can ban the address for that, so
each hit is recomputed in JS as the pool's verifier does before it is sent, and the card stops at
the first wrong one ("Untested cards mine with every hit checked", below). No setting turns the
check off: a kind of card leaves it by a change to `resolve_fold`, once one has passed the steps
under "Testing on real hardware". On 22 October,
unchanged, it passed steps 1-5 under "Testing on real hardware" on Salad: an RX 9070 XT (gfx1201)
at 93.8 TH/s and an RX 7900 XTX (gfx1100) at 38-50 TH/s depending on the PC, every checked hit
right and every share accepted on Kryptex (`tuning/benchmark.md`, "AMD Radeon", notes). It was written
with no AMD card: correctness was first shown by running the same kernel source on the CPU
through HIP-CPU, with the matrix instructions emulated from AMD's documented lane layouts, and
checking hits against the JS reference the way the pool does. RDNA 2 runs the `dot4` fold, which
has run in RDNA 2's loop on an RX 7800 XT (a build with `-DPEARL_AMD_PF2=0 -DPEARL_AMD_DPP=0`, and
`PEARL_AMD_FOLD=dot4`) with every checked hit right, but on no RDNA 2 card.

This folder reads `../src` and writes only to `./build`. Building it changes nothing in the
NVIDIA build. `native-core.yml` builds both: the AMD core in its own `ubuntu-latest-hip` and
`windows-2022-hip` jobs.

## What is here

| File | What it is |
|---|---|
| `build-amd.sh` | builds the GPU addon, the CPU (HIP-CPU) addon, or runs the compile check |
| `src/pearl_amd_kernels.h` | every kernel the AMD core launches: the portable part of `pearl_kernel.cu`, then the folds |
| `src/pearl_amd_fold.h` | the AMD folds, their instruction layouts and CPU emulations, and the transcript hash kernel |
| `src/pearl_host_hip.cpp` | the host: `pearl_host.cu`'s C API on HIP |
| `compat/gpu`, `compat/cpu` | stand-ins for `<cuda.h>`, `<cuda_runtime.h>` and `<mma.h>` |
| `test/pearl_amd_foldtest.cpp` | runs each fold on random operands and checks every transcript word against the CPU; the layout check on hardware |
| `test/pearl_amd_peak.cpp` | times the RDNA matrix instruction with little or nothing else in the loop, and the clock it runs at; timing only |
| `test/verify-hits-profile.js` | `../probes/verify-hits.js` at a profile you choose (that one is fixed to mainnet) |
| `test/run-cpu-checks.sh` | every check that runs without a GPU |

## How it works

**The portable kernels are the CUDA core's own.** BLAKE3, the commitment trees, seeds,
noise, materialise and restamp are the first 1,188 lines of `pearl_kernel.cu`.
`build-amd.sh` copies the file up to the line `// A 16-byte global->shared copy that does
not pass through registers` (where the NVIDIA folds start) into
`build/gen/pearl_kernel_portable.cu` and compiles that. If the line moves, the build
stops and says so, so the two cannot drift silently.

**The folds are new.** Each computes C = A' x B'ᵀ in int32 over k and, at each of the 16
chunk ends, XORs every region's 256 running sums into that region's transcript word.
Each fold writes all 16 words of every region of the batch, and `pearl_amd_hash` hashes
them in a kernel of its own (as the CUDA core does on GA100 and Hopper).

| Fold | Targets | Instruction | Tile |
|---|---|---|---|
| `mfma16` | gfx942, gfx950 (MI300X/A, MI325X, MI350X, MI355X) | `v_mfma_i32_32x32x16_i8`, wave64 | 128x256, 4 waves of 64x128 |
| `mfma8` | gfx908, gfx90a (MI100, MI210, MI250, MI250X) | `v_mfma_i32_32x32x8_i8`, wave64 | 128x256, 4 waves of 64x128 |
| `wmma11` | gfx11 (RX 7000, Radeon PRO W7000, Ryzen AI 300 and Max) | `v_wmma_i32_16x16x16_iu8`, wave32 | 128x256, 8 waves of 64x64 |
| `wmma12` | gfx12 (RX 9060, RX 9070, Radeon AI PRO R9700) | `v_wmma_i32_16x16x16_iu8`, wave32, RDNA4 layout | 128x256, 8 waves of 64x64 |
| `dot4` | gfx1030-1032 (RX 6600 to 6950, Radeon PRO W6600 and W6800, V620); gfx11 and gfx12 only in a build with `-DPEARL_AMD_PF2=0` | `v_dot4_i32_i8` (`v_dot4_i32_iu8` on gfx11 and gfx12) from registers, wave32 | 128x256, 8 waves of 64x64 |
| `ref` | any (the baseline) | `v_dot4_i32_i8` where the card has it, else scalar | 32x64 |

The region pattern lines up with every one of these layouts. In a 32x64 block of C each
lane's accumulators belong to one region (two on RDNA, split by register), the eight
lanes that share a region differ in lane bits 0, 3 and 4, and three XOR shuffles finish
it. Lane r of the eight keeps words r and r + 8, so a region's transcript stays in
registers until the tile ends. The 16x16x16 WMMA on RDNA3 needs A and B duplicated in
both half-waves; the fold loads them that way.

RDNA 2 has no matrix instruction, but by AMD's figures its `v_dot4_i32_i8` does 4 int8 MACs a lane
a clock, 256 a CU a clock: RDNA 3's WMMA rate. The `dot4` fold is the matrix folds' code (tile,
operand layout, staging, readout, store) with dot4s from registers in place of the instruction.
Each lane holds 8 rows by 8 columns of each 32x64 block, laid out so the region rules above hold
(`PearlAmdDot4` in `pearl_amd_fold.h`), and a k-step of 4 bytes is 24 words read from LDS and 128
dot4s a lane. The reference fold reads both operands from LDS for every dot4, which is why it is
slow. The compiler first merged each row's reads for a slab into two 16-byte reads and waited on
each just before its dot4s, which spilled on gfx11 and ran at 10.2 TMAC/s on an RX 7800 XT; an
empty asm between k-steps keeps the reads apart and in flight (`PEARL_AMD_DOT4_SPLIT`), and the
fold runs at 18.0-18.1 there, in RDNA 2's loop. That card's `v_dot4_i32_iu8` did 18.5-19.2
TMAC/s in a kernel of nothing but dot4s, so the fold gets 94-98% of what the card's dot4 can do;
its own `wmma11` folds at 32.8. Per CU and clock, that dot4 rate is about half what AMD's figures
give RDNA 2's `v_dot4_i32_i8`; no RDNA 2 card has measured it. What an RDNA 2 card gets is
projected in `tuning/benchmark.md` until one runs it. RDNA 3 and RDNA 4 release builds have no `dot4`
fold: it is slower there than their WMMA folds, and in their PF2 loop it spilled (before the
split; not tried since).

The matrix folds and `dot4` stage 32 bytes of k at a time through LDS, and read A' and B' in per-tile
order: each 128-row tile of A' and 256-row tile of B' is stored as k / 32 slabs of [rows][32
bytes] (`pearl_materialize16_tiled`, the kernel the CUDA tall fold's draw uses), so a block's
slab is one contiguous run of whole cache lines. Row-major, it was 32 bytes of every row's
128-byte line, and the next three slabs fetched the same lines again. On an RX 9060 XT (gfx1200,
4 MB of L2) that took the fold alone from 15.4 to 52.1 TMAC/s at the mainnet batch, and
hashrate.js from 21.75 to 51.3 TH/s (PC `a777486d`, 23 October). `-DPEARL_AMD_TILED=0` builds
the row-major layout. The reference fold reads row-major operands.

**RDNA 4 spends as few VALU as it can around its WMMAs.** On RDNA 4 a VALU instruction doesn't overlap
the matrix work: each one costs about a cycle of WMMA time (`test/pearl_amd_peak.cpp`, `tuning/benchmark.md`,
4 November). So the RDNA 4 fold reads each lane's fragments for both k-steps of a slab with one 16-byte
LDS load (`PEARL_AMD_KPAIR`: a slab's k-steps may take its bytes in any order, as long as A' and B' take
them alike), and loads its slabs with buffer loads, whose base and slab offset are scalar
(`PEARL_AMD_BUFLD`). Before, the fold added the slab to each lane's 64-bit address for every global
load and made new LDS addresses for most fragments, about 9 VALU a slab of 32 WMMAs. On an RX 9070 XT
the two took `hashrate.js` from 115.8 to 125.6 TH/s on one PC (+8.5%).

**RDNA 3's gfx1100 and gfx1101 load their slabs with buffer loads too** (`PEARL_AMD_RDNA3_BUFLD`:
`PEARL_AMD_BUFLD` with `PEARL_AMD_TID`, 7 November, issue #287 item 2). Its fragments are 16 bytes
already, so KPAIR has nothing to give it, but its fold spent 12 VALU a slab on load addresses, mostly in
chains of 64-bit adds, and now spends none. On RDNA 3 a chained VALU beside the WMMAs costs about 4-5
clocks of WMMA time and a lone one about 1 (`test/pearl_amd_peak.cpp` on an RX 7900 XTX and an RX 7800
XT). Against the release on the same PC, the fold alone ran +1.02 to +1.25% a round and `hashrate.js`
+0.66 to +0.77% on an RX 7900 XTX (PC `57cc1801`), and +0.81 to +1.34% and +0.75 to +0.82% on an RX 7800
XT (`72dd708c`); TID alone gave +0.16% and +0.04% on the XTX, so the gain is the loads'. verify-hits
checked 400 of 400 hits on both. gfx1102, gfx1150 and gfx1151 weren't
measured and keep their global loads, and the Windows core leaves it out (`SET_OS` in `build-amd.sh`).
`-DPEARL_AMD_RDNA3_BUFLD=0` builds the fold without it. Only gfx11 and gfx12 code takes buffer loads,
whatever the setting.

**gfx1100 and gfx1101 also build three of RDNA 4's staging switches** (`PEARL_AMD_RDNA3_SET`: `PEARL_AMD_PFX`,
`PEARL_AMD_CRANK` and `PEARL_AMD_SCHEDB`, 7 November, issue #287 item 8). CRANK takes rank 128 at compile time,
which drops the per-slab division and the readout code an even slab never ran (the fold's code goes from 48 LDS
loads to 32) and makes a slab pair one block; SCHEDB keeps the WMMAs on their side of each barrier. Alone the two lost 1.2% on an RX 7800 XT: in the merged block three of the pair's six
slab loads sat just before a barrier whose fence waits for them, with no WMMA in between. PFX's LDS-only fence
lets those loads run on, so each gets 32 to 96 WMMAs. 226 VGPRs, 6 waves. Against the buffer-load fold on the
same PC, with a twin of it for the noise, the fold alone ran +0.60 to +0.88% a round and `hashrate.js` +0.89 to
+1.20% on an RX 7900 XTX (PC `28a12f15`), and +0.83 to +0.89% and +0.83 to +0.93% on an RX 7800 XT (`72dd708c`);
verify-hits checked 400 of 400 hits on the 7800 XT, and on Kryptex the XTX got 6 accepted shares and 0 rejected
in 10 minutes. The same with the loads pinned two slabs ahead (`PEARL_AMD_PF2PIN`, below) was about 0.2% slower
in the fold on both cards and in `hashrate.js` on the 7800 XT. It follows `PEARL_AMD_RDNA3_BUFLD`, so the other
gfx11 targets and the Windows core keep their code; `-DPEARL_AMD_RDNA3_SET=0` builds the fold without it.

**On top of that, RDNA 4 builds six more cuts** (`PEARL_AMD_RDNA4_SET`, 4 November). The wrap
readout picks the keep word's half with a scalar branch instead of selects (`PEARL_AMD_RDSEL`) and
skips a register copy before `v_permlanex16` (`PEARL_AMD_RDPERM=2`). The slab loop takes rank 128 at
compile time, so a chunk is a constant 4 slabs and the per-slab division goes (`PEARL_AMD_CRANK`),
with a `sched_barrier` before each slab's stores so the compiler can't move the WMMAs past the
barrier (`PEARL_AMD_SCHEDB`). And LDS rows are 32 bytes, their two halves swapped by row bits, where
they were 48 with padding, with each thread's store and read offsets pinned in registers
(`PEARL_AMD_LDS32`, `PEARL_AMD_LDS32_PIN`). Together `hashrate.js` read +4.3 to +4.6% on an RX 9070
XT and +4.6 to +5.4% on an RX 9060 XT, each interleaved on one PC, on three PCs a card, and
verify-hits passed 400 of 400 every time. CRANK makes the fold assume rank 128, so the card's caps
say so (`PEARL_AMD_CAP_RANK`) and the host refuses any other rank. `-DPEARL_AMD_RDNA4_SET=0` builds
the fold without them. The Windows core leaves them out, as it does the wrap readout: they were timed
only with ROCm's clang 22. Tried with them and left out: one array for both LDS tiles
(`PEARL_AMD_STOFF`, level), a split slab barrier (1.1-1.4% slower on top) and 8 waves a SIMD
(2.1-2.6% slower); see `tuning/benchmark.md`.

**The host** (`src/pearl_host_hip.cpp`) is `pearl_host.cu` without the NVIDIA folds. It
picks the fold from the card's `gcnArchName` and checks it against the folds the loaded
code object actually has (`pearl_amd_caps`), so a build without the card's target fails
with a message rather than searching nothing. `PEARL_AMD_FOLD=ref|mfma16|mfma8|wmma11|wmma12|dot4`
overrides the choice. A batch is at most 256 column offsets (`PEARL_AMD_COL_BATCH`), which
keeps each pipeline slot's transcript buffer to 128 MiB at the mainnet profile, and 128 on
gfx1100 and gfx1101 (RX 7900 and RX 7800/7700 cards), which were faster there in the
miner's own loop (+0.1 to +1.8% a round, `tuning/benchmark.md`). A build that sets `PEARL_AMD_COL_BATCH`
gets that cap on every card; the variable of that name in the environment can only lower it,
to measure widths on one build. All work
runs in order on one stream; proof reads use a second, and a hit's reads are waited for
once. `PEARL_AMD_TRACE=1` prints a line on stderr every 10 s with each batch's GPU time for
the fold and the hash, how long the GPU sat idle between batches, and the search thread's
waits.

**Untested cards mine with every hit checked.** A card whose fold has not run on its kind of card
in a release (gfx9, gfx103x, or one whose architecture HIP could not read) mines only for an opener
that checks every hit before sending it (createCore's `checkHits`, which the app and
`verify-hits.js` pass), and the core then reports `core.untested`. The miner recomputes each of
that card's hits with `../../src/shared/miner/checkHit.js`, the check `verify-hits.js` runs (4.6
ms a hit at k 2048, 8.6 at k 4096), and at the first wrong one sends nothing, stops that card for
the run and logs a line asking you to report it. `PEARL_AMD_UNTESTED=1` does not turn the check
off. It only lets an opener that checks nothing open such a card, which is for `hashrate.js`: it
sends no shares. In v0.6.8 and v0.6.9 it let such a card mine with no check, and the error there
told RX 6000 owners to set it for the reference fold those cards ran then. They now run `dot4`, so
a setting carried over must not vouch for it.

**RDNA 4 mines k 4096 or k 8192.** At k 4096 a hash covers twice the MACs, so there are half the hashes
and readouts per MAC, and credit per MAC is the same at rank 128. On an RX 9070 XT and an RX
9060 XT that was +1.0 to +1.2% (`tuning/benchmark.md`). Past 16 chunks the folds need the wrap readout:
chunk c goes into word c % 16 after a rotl 13 (`PEARL_AMD_KWRAP` in `pearl_amd_fold.h`). It costs
about 0.4% at k 2048, so only the gfx12 code objects have it; every other target keeps today's
readout and mines k 2048. The Windows core has it on no target, so RDNA 4 mines k 2048 there:
with it, the HIP SDK's clang 21 put 24 bytes of the gfx12 fold in scratch memory, where ROCm's
clang 22 on Linux uses none, and no Windows Radeon has run either readout (`KWRAP_OS` in
`build-amd.sh`). One build holds both, since only a gfx12 device pass defines
`__GFX12__`. The app opens each core with `cardK`, and the host reads the card's caps
(`PEARL_AMD_CAP_KWRAP`) and picks k 4096 or 8192 where it can (`pearl_host_card_k`). The core reports it as
`core.k`, and the miner builds that card's config52, job_key, share bound and proofs from it, so a
rig of RDNA 4 and RDNA 3 cards mines both k side by side without mixing them. A share says its own
k, so the pool needs nothing from us.

**The big gfx1201 cards mine k 8192.** On an RX 9070 XT (64 CUs), k 8192 read +0.5 to +0.9% over
k 4096, and HeroMiners and Kryptex both took its shares; on an RX 9060 XT (gfx1200, 32 CUs) it was
about 6% slower (`tuning/benchmark.md`). So the host picks k 8192 on gfx1201 with at least 56 CUs (RX
9070 XT and Radeon AI PRO R9700 at 64, RX 9070 at 56) and k 4096 on the rest, the RX 9060 XT and
the 48-CU RX 9070 GRE among them, since nothing between 32 and 64 CUs has been timed.
k 8192 needs about 3,633 MiB of VRAM. That doesn't crowd an LLM: the app reads VRAM through
nvidia-smi only, so on an AMD card it runs the smallest model (~4 GB), which fits beside k 8192 on
any of these 16 GB and 32 GB cards. `PEARL_AMD_K` (2048, 4096 or 8192) picks the k on any RDNA 4
card; another value is ignored, and the "mining on GPU" line says so. Without the free VRAM its k
needs, a card tries k 4096 (from 8192), then stays at k 2048 (k 4096 takes
about 1,969 MiB against 1,137), and the line says why. If a pool turns down 3 shares at a larger k and takes none, the
miner mines k 2048 on every card for the rest of the run (`K_REJECT_LIMIT` in `pearlMiner.js`).

`build-amd.sh check` fails a release build (no `PEARL_AMD_CFLAGS`) in which a gfx12 target lacks
the wrap readout or any other target has it, or, on Windows, in which any target has it. The fold test folds k
4096 on RDNA 4 by default, not k 8192 where the card mines that (verify-hits.js and hashrate.js open
the core with `cardK`, so they use the mined k), and names it on its device line; a 7th argument picks another.
`-DPEARL_AMD_KWRAP=0` builds today's release, k 2048 everywhere, to race against;
`-DPEARL_FOLD_K=4096` puts the wrap readout on every target. `PEARL_PROFILE_K=K` on
`hashrate.js` and `verify-hits.js` pins a core to k K; `test/sweep.sh` does that for a variant
whose flags set `PEARL_FOLD_K`, and times every variant's fold over the same MACs as a k 2048
build.

## Checked without a GPU

HIP-CPU (github.com/ROCm/HIP-CPU) runs HIP kernels as C++ on the CPU. The same
`pearl_host_hip.cpp` and kernels build against it into `pearl_core_hipcpu.node`; in that
build each matrix instruction is replaced by an emulation written from AMD's Matrix
Instruction Calculator layout. Everything else in the fold is the source the GPU runs. The `dot4`
fold needs no emulation: the CPU runs its code as it is, with four multiplies for each `v_dot4`.

`test/run-cpu-checks.sh`, on 2026-10-09 (4 cores, about 8 minutes):

| Check | Result |
|---|---|
| fold check, 256x512 and 512x1024, all five folds | every region's 16 words match the CPU computation |
| core, `ref` / `mfma16` / `mfma8` / `wmma11` / `wmma12`, constant fill, m=256 n=512 | 400 of 400 hits verified each, 95-176 salts |
| core, `mfma16` and `wmma12`, hashed fill | 400 of 400 each |
| core, `mfma16`, m=256 n=8192 (two batches a salt) | 400 of 400, 13 salts |

Since the RDNA 4 change (k 4096), it also builds `pearl_core_hipcpu_kwrap.node` and
`pearl_amd_foldtest_cpu_kwrap` with the wrap readout, and checks them at k 2048 and k 4096. The
kwrap core opened with `cardK` must pick k 4096 and pass every hit at that k; with
`PEARL_AMD_K=2048`, or the default core, it must stay at k 2048. With `SHARE_DIR` set, the k 4096
runs write their shares there for `probes/refverify.rs`, the spec's own verifier (amdtest.sh's
`refverify` task).

With the `dot4` fold it also checks `dot4` at its other k-steps (`_dot4k8`, `_dot4k16`) and runs
the whole core with it at k 2048 and, with the wrap readout, k 4096. All 23 checks passed on 2
November, on a Salad box's 4 CPUs in 17 minutes (amdtest.sh's `cpuchecks`): `dot4` 400 of 400 hits
at both k.

"Verified" is `verify-hits.js`'s check: Merkle proofs, seed chain, noise and fold
recomputed in JS, and the jackpot hash must match the core's exactly. The fold check was
also run with deliberate mistakes: a layout's lane-to-row map, its k offsets, a shuffle,
the column offset, the transcript word index, RDNA4's register split and RDNA3's
half-wave loads. Each one failed. (Swapping rows inside one region passes, and should:
it cannot change a transcript.)

What this cannot show: that the hardware matches the documented layouts. Step 2 below
checks that first.

One HIP-CPU problem had to be worked around: it can run work queued on the null stream
while earlier null-stream work is still running, which let a redraw rewrite A' under a
fold still reading it (about 1 hit in 6 failed in one run). The host now queues
everything on a stream of its own, which behaves the same on a GPU.

## Compile check

`./build-amd.sh check` with ROCm 7.2.4's hip-clang (LLVM 22). Every fold has no spills and
no scratch memory on every target:

| Target | Fold | VGPRs | AGPRs | LDS | Waves/SIMD |
|---|---|---|---|---|---|
| gfx908, gfx90a | mfma8 | 168 | 128 | 36 KB | 1 |
| gfx942, gfx950 | mfma16 | 168 | 128 | 36 KB | 1 |
| gfx1030, gfx1031, gfx1032 | dot4 | 201 | - | 36 KB | 4 |
| gfx1030, gfx1031, gfx1032 | ref | 42 | - | 13.5 KB | 16 |
| gfx1100, gfx1101 | wmma11 | 226 | - | 36 KB | 6 |
| gfx1102, gfx1150 | wmma11 | 240 | - | 36 KB | 4 |
| gfx1151 | wmma11 | 240 | - | 36 KB | 6 |
| gfx1200, gfx1201 | wmma12 | 201 | - | 24 KB | 7 |

The hash kernel takes 49-52 VGPRs and the reference fold 40-46 everywhere. The wmma12 row is the RDNA 4
set's (`PEARL_AMD_RDNA4_SET`, 4 November): 201 VGPRs and 24 KB of LDS, against 193 and 36 KB without it,
and 198 before `PEARL_AMD_KPAIR` and `PEARL_AMD_BUFLD`. Its 7 waves are what the registers allow; it
runs 6, since a block is 8 waves, 2 to a SIMD. The Windows core's is still 193 and 36 KB. The gfx1100 and
gfx1101 rows have RDNA 3's buffer loads and staging cuts (`PEARL_AMD_RDNA3_BUFLD`, `PEARL_AMD_RDNA3_SET`):
226 VGPRs, against 232 with the loads alone and 240 without either, as the Windows core and the other gfx11
targets build it. On 2 November, with the
`dot4` fold, every row was the same as before but gfx103x's new `dot4` row; RDNA 2's loop built for
gfx1101 (`-DPEARL_AMD_PF2=0 -DPEARL_AMD_DPP=0`) put `dot4` at 205 VGPRs, 6 waves. A release check
also fails a target without the fold the host runs on it (`folds_of` in `build-amd.sh`).
`./build-amd.sh gpu` builds all 14 targets into one addon (about 2 minutes) and the fold check
binary.

## Building

On Linux you need ROCm 7.x with hip-clang (`rocm-llvm`, `hip-dev`,
`rocm-device-libs`; for running, `rocm-hip-runtime`), Node 22 and node-addon-api:

```sh
cd earn/native/amd
npm install --no-save node-addon-api
./build-amd.sh gpu                  # all 14 targets
./build-amd.sh gpu gfx1100          # or just your card's, which is faster
./build-amd.sh check                # registers, scratch and LDS per fold and target
```

The CPU build needs HIP-CPU, TBB (`libtbb-dev`) and clang:

```sh
git clone https://github.com/ROCm/HIP-CPU /tmp/hip-cpu
HIP_CPU_PATH=/tmp/hip-cpu ./build-amd.sh cpu
./test/run-cpu-checks.sh
```

The addon links `libamdhip64.so.7`, so the machine it runs on needs the ROCm 7 runtime.

### Windows

You need AMD's HIP SDK 7.2 for Windows (its installer sets `HIP_PATH`), Visual Studio's C++
tools, Node 22 and Git Bash. Start Git Bash from the "x64 Native Tools Command Prompt", so
MSVC is on PATH, then:

```sh
cd earn/native
npm install --prefix . --no-save node-addon-api node-gyp
amd/build-amd.sh winlib               # the 10 Radeon targets
amd/build-amd.sh winlib gfx1100       # or just your card's, which is faster
PEARL_BACKEND=hip npx node-gyp rebuild
cp build/Release/pearl_core.node amd/build/pearl_core_hip.node
```

`winlib` compiles the kernels and host with the SDK's clang into `build/win/pearl_hip.lib`,
and node-gyp links it with `pearl_core.cc` (`binding.gyp`, `PEARL_BACKEND=hip`), as it links
the CUDA core. Both halves use the static C runtime. The addon links `amdhip64_7.dll`, which
AMD's graphics driver installs in System32, so a PC with the driver needs nothing else. There
are no Instinct targets: AMD's Windows runtime is for Radeon cards. `check` works on Windows
too, and `winlib` also builds `build/pearl_amd_foldtest.exe`, the layout check below; CI
uploads it as `pearl-amd-foldtest-windows-2022`.

## Testing on real hardware

On a Linux box with an AMD GPU, ROCm 7.x installed, and the user in the `render` and
`video` groups:

```sh
# 1. Build for the card.
rocminfo | grep -m1 -o 'gfx[0-9a-f]*'                  # the card's target, e.g. gfx1100
cd earn/native/amd
npm install --no-save node-addon-api
./build-amd.sh gpu gfx1100

# 2. Layout check: every fold the card has, against the CPU. Must end in PASS.
./build/pearl_amd_foldtest
./build/pearl_amd_foldtest 2048 2048 3                # bigger; the CPU side takes ~10 s
./build/pearl_amd_foldtest 16384 16384 1 20 0         # timed only: each fold's rate in TMAC/s

# 3. The whole core at the mainnet profile, checked as the pool checks it. On a gfx9 or
#    gfx103x card, which hasn't run the core yet, steps 3 and 5 check every hit themselves;
#    step 4 checks none and sends nothing, so give that one line PEARL_AMD_UNTESTED=1. It
#    turns no check off.
#    Must print PASS with 400 verified. The reference fold is slow, so give it an easier target.
node ../probes/verify-hits.js build/pearl_core_hip.node 120
PEARL_AMD_FOLD=ref node ../probes/verify-hits.js build/pearl_core_hip.node 300 236

# 4. Hashrate, the number the app shows.
node ../probes/hashrate.js build/pearl_core_hip.node 60 10

# 5. Mine on the pool through the CLI.
cd ../..
PEARL_CORE_VARIANT=amd node src/cli/earn-cli.js --address <prl1p...> --mode mining
```

On Windows, take `pearl_core_hip.node` and `pearl_amd_foldtest.exe` from a native-core run's
`pearl-core-hip-windows-2022` and `pearl-amd-foldtest-windows-2022` artifacts (or build them,
above), and run steps 2-4 with those paths. The card's target is in the foldtest's first
line. Step 5 is the installed app: with no NVIDIA card it picks the AMD core itself.

If step 2 fails for a fold, its layout differs from the documented one: send the output
line, which names the first wrong region and word. If step 3 fails, send its JSON line.
For a speed report, send the output of steps 2 (timed), 4, and `rocm-smi` while step 4
runs.

## Using it in the app

The app and `earn-cli` load `pearl_core_hip.node` from beside the executable, the packaged
`native/` folder, or this folder's `build/`. A Linux release's copy is beside the CLI binary,
in the HiveOS package and in the AppImage, and the CLI updates it with the binary on a rig
that uses it. The Windows installer's is in `resources\native`. A Linux rig needs the ROCm 7
HIP runtime (`libamdhip64.so.7`); when it is missing, the log names the library the core
could not load. A Windows PC needs AMD's graphics driver, for `amdhip64_7.dll`.

When nvidia-smi lists no NVIDIA card, the app tries it first (`shared/coreVariant.js`). If
it isn't installed, won't load, or can't start on its first card, the CUDA 12.8 core takes
over, and the log says why. That keeps an NVIDIA rig whose nvidia-smi failed mining.
`PEARL_CORE_VARIANT=amd` (the CLI's `--core amd` sets it) asks for the AMD core on any rig,
and then nothing falls back to a
CUDA build. `PEARL_CORE_PATH=/path/to/pearl_core_hip.node` works too. Give it an absolute
path: a relative one counts from the executable's folder. With `PEARL_CORE_VARIANT=amd`,
a `PEARL_CORE_PATH` that doesn't load is the end of it; the CUDA cores aren't tried.

The app finds cards with nvidia-smi, which lists no AMD card. So one core starts and picks
the AMD card with the most compute units times clock. If HIP lists a card the release has no
code for (an RX 5000 card, or an APU's graphics) before the ones it has, HIP loads no code for
any card after it; the error names that card, and `HIP_VISIBLE_DEVICES` hides it. On a rig with several, choose them
by index with `--gpu-index 0,1` (or `PEARL_GPU_INDEX`); the app passes indices
nvidia-smi does not list straight to the core, one core each. Indices are HIP's, which
follow `rocm-smi` unless `HIP_VISIBLE_DEVICES` says otherwise. Temperatures, VRAM rows on
the board and memory clocks are read with nvidia-smi and stay empty on AMD.

Without nvidia-smi the app also can't tell which card the local LLM loaded on. While the
LLM co-runs on a rig mining on one AMD card, each prompt pauses that card, whichever card
the model is on. On several mining AMD cards it pauses none of them, and prompts run
beside the miner. Pausing only the LLM's card needs AMD bus ids for the mining cards (from
`rocm-smi`, say).

After a GPU fault (an unstable overclock, say) ROCm refuses every later HIP call in the process,
on every card, so all the AMD cards stop and the error says to restart the miner. A CUDA fault,
by contrast, stops only its own card.

## Gaps and next steps

- On 22 October `build-amd.sh check` compiled all 14 targets on ROCm 7.2 with no spills and no
  scratch, and the 10 CPU checks passed on the current sources.
- The first hardware runs were on Salad (`tuning/salad/amdtest.py`, which builds this folder on a
  Salad AMD PC and runs the steps under "Testing on real hardware"). HIP reaches the GPU there
  through WSL2: `/dev/dxg` and Salad's bridge `/opt/rocm-host/lib/librocdxg.so`, with no `/dev/kfd`.
  On 23 October every Radeon class Salad rents ran: RX 9060 XT (gfx1200), RX 9070 XT (gfx1201),
  RX 7800 XT (gfx1101), RX 7900 XT and RX 7900 XTX (gfx1100). Each 300 s or 600 s hit check gave
  7.81-8.05 hits a salt with no failed hit, so the low counts of 22 October (6.92-7.55) came from
  short runs. `amdtest.sh`'s `duel` task runs SRBMiner on the same PC (tuning/benchmark.md's method);
  the rows are in `tuning/benchmark.md`.
- Pause only the LLM's card for a prompt on a rig of several AMD cards (see "Using it in
  the app"): read the mining cards' bus ids with `rocm-smi` or sysfs, and check HIP's
  card order against them on hardware.
- Run steps 1-5 above on an MI300X (CDNA 3: the mfma folds have only run on the CPU).
- Run steps 1-5 above on an RDNA 2 card (an RX 6600 to 6950, a Radeon PRO W6600 or W6800, or a
  V620; Salad's AMD classes have none): the `dot4` fold has run only in RDNA 2's loop on an RX 7800
  XT (`tuning/benchmark.md`, "a fold for RDNA 2"), so its speed on RDNA 2 is a projection. Then take gfx103x out of the
  untested cards in `resolve_fold`, so its hits are no longer checked, and time `PEARL_AMD_DOT4_K=8`
  and the DPP readout there. DPP needs its gate in `pearl_amd_group_xor` widened to gfx103x (its
  lane moves exist on gfx10 too); it made no difference to `dot4` on RDNA 3, where each dot4 took
  twice the clocks AMD's figures give it on RDNA 2.
- Speed: the operand layout is tuned (per-tile A' and B', above); tried on the RX 9060 XT and
  no faster on top of it: a group-of-4 tile walk, non-temporal A' loads, an LDS-only slab
  barrier with one slab of prefetch, batch widths 64 and 128, and 64-byte slabs (4% slower). On RDNA 3 the fold also
  loads two slabs ahead (`PEARL_AMD_PF2`) and does the readout's lane exchanges with DPP and
  `v_permlanex16` (`PEARL_AMD_DPP`): 31.10-31.16 TH/s to 32.17-32.28 on an RX 7800 XT (PC
  `16f1685c`), whose matrix instructions alone run the fold at 34.3 TMAC/s. That A/B against
  the old path is one PC per RDNA 3 chip (on gfx1100 each alone on an RX 7900 XTX: PF2 +2.4%, DPP
  +0.5%); a second PC of each is still to run. The compiler keeps the loads only one slab
  ahead, not two: on gfx1100 `__syncthreads()` waits for each slab's loads at the end of the step
  that issues them, and with PFX's barrier the compiler sinks them to that step's end instead.
  `PEARL_AMD_PF2PIN` keeps all of them two slabs ahead: PFX, TID,
  and the loads pinned at the top of the step by a memory clobber as well as a `sched_barrier`
  (`PEARL_AMD_LPIN=2`), since machine sinking moved half of them past the `sched_barrier` alone.
  235 VGPRs, 6 waves, no spills (see the comment in `pearl_amd_fold.h`). It stays off: one slab
  ahead already hides the loads (their wait at the fence is 0.8-1.9% of the slab loop, by the
  stamps), the pin alone ran the fold 0.05-0.42% faster, and with CRANK and SCHEDB it was 0.2%
  slower than `PEARL_AMD_RDNA3_SET`, which leaves three loads one slab ahead. RDNA 4 also has PF2
  and the DPP readout, and on top keeps the loads in flight across the slab barrier (`PEARL_AMD_PFX`: an
  LDS-only fence, since `__syncthreads()` waited for every outstanding global load there) and
  tells the compiler the thread index is under 256 (`PEARL_AMD_TID`): 52.88 TH/s to 59.99 on an
  RX 9060 XT (PC `ca4f64e4`). On RDNA 3 those two measured no faster alone; gfx1100 and gfx1101
  now take them with the buffer loads and `PEARL_AMD_RDNA3_SET`. A split slab
  barrier on gfx12 (`s_barrier_signal` after the LDS work, `s_barrier_wait` before the next slab)
  ran the same as this (59.39 against 59.99 on `ca4f64e4`, 33.20 against 32.85 on `54b217a9`)
  and was removed. Next: `v_mfma_i32_32x32x32_i8` on gfx950 (2x the rate; its
  layout is not in AMD's calculator yet, so gfx950 runs the gfx942 instruction); A' and B'
  materialised in fragment order so the folds skip LDS; `global_load_lds` on CDNA; the
  hash fused into the fold; persistent tiles; the batch and band sized from the L2 and
  Infinity Cache as `learnings.md` describes for NVIDIA.
- AMD cards in the app: rocm-smi or amd-smi for card lists, temperatures and VRAM.
- Run the Windows core on a Radeon: steps 2-5 under "Testing on real hardware", and its speed
  against the Linux core's on the same card. HIP SDK 7.2's compiler is clang 21; ROCm 7.2.0's
  on Linux is AMD clang 22, so the two cores' code differs.
- On Windows a Ryzen's built-in graphics may be listed before the Radeon card. The core then
  stops with an error naming it, which `HIP_VISIBLE_DEVICES` fixes but the app can't set.
