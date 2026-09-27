# Kernel probes

Standalone CUDA programs that establish a card's denominators. None needs the miner.
Run these before touching the kernel: every ratio in `OPTIMIZATION.md` and the 4090
tuning log is relative to that card's ceiling, and none of them transfer between
architectures.

```
nvcc -arch=sm_120 -O3 -o mmapeak mmapeak.cu && ./mmapeak 512
```

| Probe | Answers |
|---|---|
| `mmapeak.cu` | Pure `mma.m16n8k32.s8` from registers, no memory. Hardware ceiling + power. |

## Measured

| Card | Ceiling @512 thr | Power at ceiling | Miner | % of ceiling |
|---|---|---|---|---|
| RTX 4090 (sm_89) | 338 T-MAC/s | 156 W | 216.0 TH/s | 64% |
| RTX 5090 (sm_120) | **482.6 T-MAC/s** | **153 W @ 2400 MHz** | 138 TH/s | **28.6%** |

The 5090 reaches a 1.43x higher tensor ceiling on the same power as the 4090. The
shipped kernel meanwhile draws 575 W (its full cap) at 1170 MHz -- 38% of the 3090 MHz
max, with `SW Power Cap` active. Pure MMA does not do that, so the power is going to
the memory path, and the clock it costs is what puts a 5090 below a 4090.

## Traps (learned the hard way, see tuning log s6)

- **A store the compiler can prove dead deletes the whole workload.** The first version
  of `mmapeak` guarded its store with `if (threadIdx.x == 1024)`, which is never true;
  it reported 377,140 T-MAC/s in 0.00 ms. Store unconditionally.
- **Check `cudaGetLastError()` after the launch, not just after the sync.** A launch
  that fails on resources reports 0.00 ms rather than an error.
- **Check `nvidia-smi` compute processes first.** A miner or LLM sharing the card
  silently contaminates every reading.

## RTX 4090 at its stock 450 W: where the reported rate goes

Measured on the v0.5.2 core. The card is **power-capped** at stock (`SW Power Cap:
Active`, ~449 W, ~2470 MHz) -- the tuning log's "not power-limited" reading was taken
at a raised 480 W limit, which resets on reboot and which rigs do not run.

| what is timed | TH/s |
|---|---|
| the full miner loop -- what the app shows (`hashrate.js`) | 223-224 |
| fold + finalize, no operand redraws (`bench.cu`) | 232-233 |
| fold only (`bench.cu` with finalize ablated) | 238-240 |

So the operand redraw between salts costs **4.0%** of wall clock and finalize **2.6%**.
Neither is the fold, and every fold win is diluted by them.

Both are gone since. A new salt now restamps A instead of drawing both operands again
(0.72 ms against ~6 ms), and the transcript hash runs in the fold's own epilogue, so the
1 GiB of transcripts a batch is never written or read back. Finalize turned out to be
DRAM-bound, not hash-bound: its hash was already within a few instructions of ideal.
`-DPEARL_ABLATE_TRANSCRIPT_HASH` skips the fused hash to price it (1.8% of the fold).

Inside the fold, fold-only, all at the cap (these builds compute wrong answers):

| build | TH/s | SM clock |
|---|---|---|
| shipped | 239.9 | 2475 MHz |
| `-DPEARL_ABLATE_BARRIER` | 261.7 (+9%) | 2292 MHz |
| `-DPEARL_ABLATE_STAGING` | 285.6 (+19%) | 2641 MHz |
| `-DPEARL_ABLATE_TRANSCRIPT` | 244.5 (+2%) | 2441 MHz |
| 256 threads, 64x64 warp tile | 235.5 (-2%) | 2531 MHz |

The per-chunk `__syncthreads` is worth 18% per clock and still 9% after the clock
drops to pay for it. The square warp tile that won 2% on Blackwell loses 2% here.

### Where it ended: 223 -> 311 TH/s at the same 450 W

| step | full miner loop |
|---|---|
| v0.5.2 core | 223 |
| restamp A between salts instead of redrawing both operands | 229 |
| + hash in the fold's epilogue, no 1 GiB transcript round trip | 234 |
| + persistent fold, next tile's chunk 0 staged under the last chunk | 241 |
| + per-group staging, mid-k-step copies, one-barrier seam, serpentine bands | 264 |
| + ldmatrix lane bases held for the kernel, compile-time `active` | 270 |
| + tile coordinates by shift and mask, not divides | 273 |
| + a tile pattern the accumulators hold, eight 64x64 warps a block | 280 |
| + a 192x256 tile, three 64-deep stages on an mbarrier ring | 289 |
| + whole-line operand order, EMPTY arrivals from every thread, the ring's roles in registers | 300 |
| + a constant operand fill, A = B = 48 | **311** |

The 241 and 264 steps are gated to sm_89 (`PEARL_FOLD_PERSISTENT`, `PEARL_FOLD_GROUP_STAGE`,
`PEARL_FOLD_SERPENTINE`), because only a 4090 has run them. Ampere and Blackwell keep one
block per tile and the block-wide walk; run on this card with their settings forced, those
paths measure +1.8% and +1.9% over the previous core rather than a regression, and their
chunk loops compile to 370 and 484 instructions against 385 and 451 before.

The 270 and 273 rows are gated to sm_89 too (`PEARL_FOLD_LANE_BASES`,
`PEARL_FOLD_FAST_COORDS`). Measured interleaved against v0.5.5 in the full loop: +2.3%,
then +3.5% with both (262.6 / 262.8 -> 272.0 / 271.8; v0.5.5 measured ~1.5 TH/s under its
264 that day, so the rows carry the gains onto 264), 400/400 hits verified. Both shorten
the chunk and tile seams, where all sixteen warps wait at the barrier together and each
instruction costs about 0.12% of the rate. Both sit on the 128-register knife edge: holding
the fast-coordinate shift in a register instead of recomputing it cost ptxas the lane bases
and measured 265.0 against 273.0. Check the SASS after any fold edit: the first `LDSM` should
be a few instructions after `BAR.SYNC`.

The 280 row is gated to sm_89 as well (`PEARL_FOLD_WIDE_WARPS`; the tile pattern it needs is
every card's). It measured 271.6 -> 279.8 against the row before it, interleaved the same
day (+3.0%); see "Eight 64x64 warps a block" below.

The 289 row is its own kernel, `pearl_tile_fold_tall`, with a body only in the sm_89 and
sm_120 builds (`PEARL_FOLD_TALL`; sm_120's stages with TMA, see `PEARL_TALL_TMA`); the host
launches it when that is the binary it loaded. Against the 273 row, interleaved: 271.8 /
271.9 -> 288.6 / 289.4 in the full loop (+6.3%); see "A 192x256 tile on an mbarrier ring"
below.

The 300 row changes that kernel's sm_89 build, and the order the draws write its operands
in. Against v0.5.6, interleaved, both with the hashed fill: 289.6 / 290.6 -> 300.4 / 300.6
in the full loop (+3.6%); see "Whole-line staging and a leaner ring" below. The order is now
the k-blocked one sm_120's TMA reads, shared by both builds; see "Onto v0.5.7" below.

The last row changes only the bytes A and B hold, on every card: 48 plus the salt stamp,
instead of hashed int7. It is the mainnet default, and a pool has accepted shares made with
it. Against the 300 row's build, interleaved: 300.8 / 301.0 -> 311.6 / 311.7 in the full
loop (+3.6%). That session ran about 1 TH/s over the 300 row, so the row carries the +3.6%
onto 300. Against v0.5.6 in the same session: 290.8 / 290.7 -> 311.7 / 311.4 (+7.1%). See
"Operand values" below.

v0.5.7's sm_89 fold is v0.5.6's instruction for instruction (it added the sm_120 build), so
its row is 289 too. The last two rows on top of it, against it, interleaved: 291.6 / 290.2
-> 310.6 / 310.9 in the full loop (+6.8%); a second session, 290.2 / 291.1 -> 311.1 / 311.5
(+7.1%). See "Onto v0.5.7" below.

### Against the field: 264 is 15.8% behind

What a user compares is the number a miner DISPLAYS over a few minutes, so that is the
test: `compare-miners.sh`, 5 minutes each, back to back on this 4090 at stock 450 W, same
pool, same wallet, after a 60 s unmeasured warm-up (2026-09-25):

| miner | displayed TH/s (min 1-5) | shares ok/rej | SM clock | power | fee |
|---|---|---|---|---|---|
| PeakMiner 2.17.1 | **313.5** | 9/0 | 2456 MHz | 449 W | 2% |
| SRBMiner 3.6.9 | 313.0 | 15/0 | 2471 MHz | 449 W | 2% |
| ours, v0.5.5 | 264.1 | 6/0 | 2380 MHz | 449 W | 0% |

Two independent codebases land within 0.2% of each other, which reads as what a well-fed
fold reaches on this card rather than one vendor's trick. Note where the gap is: the same
449 W, and THEY hold the higher clock. They do ~16% less energy per multiply-accumulate
(0.70 TH/W against 0.59), and on a power-capped card that is the whole difference. The
pure-mma ceiling here is ~340 T-MAC/s, so they sit at ~92% of it and we sit at ~78%.

### A tile the accumulators already hold: +0.5%, all of it energy

The tile moved from contiguous 0..15 by 0..15 to rows {0,1,2,3}+8j by columns {0,1}+8i
(`pearl_config.h`). Across the 32x64 warp tile each lane's 64 m16n8k32 accumulators are
then a quarter of exactly one region, shared with lanes L^4, L^8 and L^12. So the per-chunk
readout is the lane's own XOR tree plus three shuffles, where it was a tree plus eight
whole-warp REDUX and their uniform-register moves. The pattern is self-describing, and
config52, the proofs and the oracle follow it. The pool takes it: 2 of 2 shares accepted in
49 s at us2.pearl.herominers.com (`earn-cli`, 2026-09-25). That also confirms the
two-dimension pattern encoding in config52 against the live verifier.

Bench, 4090 at 450 W, 30 s runs, interleaved (2026-09-25):

| build | TH/s | vs v0.5.5 |
|---|---|---|
| v0.5.5, REDUX readout | 263.4 / 264.1 / 263.9 | |
| lane readout, three independent shuffles (shipped) | 265.1 / 265.1 / 265.1 | +0.5% |
| lane readout, two-step butterfly | 264.6 / 264.7 / 264.8 | +0.3% |
| fold deferred into the next chunk's first k-step | 258.7 / 258.4 / 258.9 | -2.0% |
| diagnostic: no shuffles (`PEARL_ABLATE_TRANSCRIPT`) | 267.1 | +1.2% |
| diagnostic: no per-chunk readout at all (`PEARL_ABLATE_READOUT`) | 271.3 | +2.8% |

The full miner loop (`hashrate.js`) went 262.9 / 262.8 -> 264.0 / 264.9. The gain is
energy, not tensor-pipe time. Nsight Compute has the pipe 85.6% busy before and 85.5% after,
while instructions per mma fell from 4.43 to 4.12, and at the power cap that buys clock.

It is worth more once the chunk head is short. On a build that keeps the ldmatrix lane bases
live across chunks (the first ldmatrix two instructions after the barrier), the same readout
measured 269.2 / 269.7 / 269.4 -> 272.9 / 273.0 / 272.8 (bench) and 268.0 / 268.2 ->
271.4 / 271.3 (full loop), +1.2-1.3%, at the same clock. So there it is tensor-pipe time:
rate over clock times the 131072 MAC/clk peak goes from 0.867 to 0.880. With little else
between the barrier and the first mma, the readout's latency is on the critical path.
Add shift-and-mask tile coordinates to that build as well and most of it is gone again:
272.7 / 273.0 / 273.0 -> 274.0 / 273.5 / 273.8 (bench), 271.2 / 271.7 -> 272.5 / 272.6
(full loop), +0.3-0.4%. The two gains overlap rather than add.
The two diagnostics bound what any readout can still give: the 32-gate tree is the floor for
XORing 64 values and costs ~1.6%, and the shuffles ~0.7%. That is mostly their latency at the
chunk end, where every warp arrives at once. Moving them elsewhere did not work: ptxas sinks
them back to the end of the chunk, volatile asm or not, and holding their inputs across the
barrier made it re-read `threadIdx` in the chunk head.

### Eight 64x64 warps a block: +3.0%

On Ada the fold now runs eight 64x64 warp tiles (256 threads) over the same 128x256 CTA
tile, with the tile pattern above (`PEARL_FOLD_WIDE_WARPS`). A 512-thread block caps a
thread at 128 registers, half of them accumulators, and that is what held the warp tile
at 32x64. At 256 threads the cap is 255. A thread holds 128 accumulators, a fragment
ldmatrix feeds twice the mma (0.25 per mma, not 0.375), and each lane holds a quarter of
two regions of the pattern instead of one. The fold uses 235 registers and spills
nothing. The host reads the block size off the loaded binary, as it does for the
persistent grid, and refuses a fold whose launch bound says otherwise.

Against the sixteen-warp fold of the rows above, 4090 at 450 W, interleaved (2026-09-25):

| | sixteen 32x64 warps | eight 64x64 warps |
|---|---|---|
| bench, 30 s | 273.8 / 273.5 / 272.2 | 281.2 / 281.4 / 281.3 (+3.0%) |
| full miner loop, 60 s | 271.73 / 271.51 | 279.86 / 279.65 (+3.0%) |
| SM clock | ~2365 MHz | ~2425 MHz |
| rate / (clock x 131072) | 0.882 | 0.887 |
| instructions per mma (Nsight) | 3.75 | 2.89 |
| tensor pipe busy (Nsight) | 88.7% | 89.0% |

400 of 400 hits verified, across 303 operand draws. The pool takes it: 2 of 2 shares
accepted in 26 s at us2.pearl.herominers.com (`earn-cli`, 2026-09-25).

Every 256-thread fold before this lost: -2% at v0.5.2, -0.4% on the block-wide staging
walk. The probe pre-check (`feedprobe2c`, lane readout and hash in both) had it at +1.8%.
What it took on the real fold, each step measured against the one before (bench):

| step | gain |
|---|---|
| group staging, copies in the middle of k-steps 0-2, ldmatrix lane bases held | +1.1% |
| staging destinations held too | +0.7% |
| the next chunk's A slots first: A0-A3, B0-B3, B4-B7, one group a k-step | +1.1% |

Two things about eight warps explain all three.

- **A scheduler has two warps, and the barrier keeps them in step.** The tensor pipe takes
  one mma at a time (16 cycles; the issuing warp then waits 8 before its next instruction).
  It idles whenever both warps of a scheduler run something else at once, and with the
  barrier lining them up, they often do; sixteen warps had four a scheduler to cover each
  other. So every instruction between mma costs more. ptxas rebuilt the ldmatrix bases and
  the staging destinations from the lane id although registers were free (24 instructions
  after every barrier, 7 an operand in every copy group): it prices recomputing below
  holding. It cannot recompute a value behind an empty `asm volatile`.
- **The copy schedule only steers ptxas.** A small copy group gets predicated instead of
  branched around, and a predicated group is scheduled freely. Of the thirteen schedules
  in `pearl_slots_before`, every one that ended up with copies ahead of the chunk's first
  mma lost 2% or more, and so did the two whose first copies follow only 5-8 mma.
  Predicating every copy, which saves the branches, lost 1.2% for the same reason despite
  50 MHz more clock. The shipped one lands its copies after the chunk's 11th, 33rd and 80th
  mma.

Tried and dropped: offsetting the two warps' copies by a pair or two (0% to -2%);
computing the copy bases once a chunk (the groups then get predicated and land in the
seam); folding a region's transcript right after its last mma (ptxas sinks it back to the
chunk's end, SASS unchanged). Ablations, the first eight-warp build against sixteen warps
(wrong results, pricing only): without the fused hash 0% against -4.3% (sixteen warps need
their hashers' skew, eight do not); without the per-chunk barrier +1.9% against +4.0%.

Check after any fold edit: 0 spill; the first `LDSM` a few instructions after `BAR.SYNC`;
no `LDGSTS` ahead of the chunk loop's first `IMMA` (today they follow its 11th, 33rd and
80th).

### A 192x256 tile on an mbarrier ring: +3.2% over eight 64x64 warps

Staging is the fold's largest cuttable energy: 42 pJ a byte under load, 48 bytes an mma at
128x256 (see the next section). Bytes per MAC are 1/BM + 1/BN, so a 192-row tile moves 22%
fewer, 37.3 an mma, and a 96x64 warp tile feeds each ldmatrix to twelve mma, not eight. It
needs 192 accumulators a thread, which is as far as 255 registers go.

Two things had stopped it. Two 128-deep stages of 192x256 are 112 KB, over the 99 KB a
block may have, so the stages have to be 64 deep, three of them. And with a `__syncthreads`
a stage, as the fold synchronises, that is two lockstep seams a chunk: the old feed probe
had 64-deep stages losing 3.5-5.5% at either tile size.

So the tall fold drops the barrier for an mbarrier ring. FULL[b] completes when all 256
threads' copies into buffer b have landed (each thread arrives through
`cp.async.mbarrier.arrive.noinc` once it has issued them); EMPTY[b] when all eight warps have
read b. A warp waits on FULL before it reads a stage and on EMPTY before it refills a buffer,
and on nothing else, so the two warps of a scheduler can drift up to a stage apart and stop
lining their readout seams up. sm_89 has no `try_wait`, so a wait spins on `test_wait`; the
waits rarely spin, because what they wait for was issued a chunk earlier.

Priced first on a feed probe (`perf-scratch/r7-probe/feedprobe4.cu`) that models the
eight-warp fold -- group staging, lane readout with its shift queue, four hashers -- and
checks every variant's results against the barrier build's. 4090 at 450 W, 128 SMs, T-MAC/s,
interleaved, two sessions:

| CTA tile, warps | stages | sync | rate | clock |
|---|---|---|---|---|
| 128x256, eight 64x64 | 2 x 128 | `__syncthreads` (the eight-warp fold) | 275.8 - 276.8 | 2470 MHz |
| 128x256, eight 64x64 | 2 x 128 | ring | 278.1 - 279.4 | 2395 MHz |
| 128x256, eight 64x64 | 3 x 64 | `__syncthreads` | 267.3 | 2430 MHz |
| 128x256, eight 64x64 | 3 x 64 | ring | 278.2 - 278.8 | 2340 MHz |
| 128x256, eight 64x64 | 4 x 64 | ring | 272.2 - 273.9 | 2367 MHz |
| 192x256, eight 96x64 | 3 x 64 | `__syncthreads` | 277.5 - 277.6 | 2502 MHz |
| 192x256, eight 96x64 | 3 x 64 | ring | **291.0 - 292.6** | 2420 MHz |
| 128x256 / 192x256, no sync at all (wrong results) | | | 294.4 / 299.7 | |

The ring is worth little to the 128x256 tile and is what the 192x256 tile needs. On 64 SMs,
where nothing hits the cap (2685 MHz), the larger tile spends 4-7% less energy an mma at the
same synchronisation: 6.54 against 6.83 nJ with the barrier, 6.67 against 6.95 with the
ring (power above a 90 W baseline, corrected to 75 C). The rest of its gain is the tensor
pipe: rate / (clock x 131072) 0.921 against 0.853.

The fold (`PEARL_FOLD_TALL`, a kernel of its own on sm_89), interleaved on the same day:

| | eight 64x64 warps, 128x256 | eight 96x64 warps, 192x256, ring |
|---|---|---|
| bench, 30 s (sixteen-warp fold alongside: 273.5 / 272.8 / 273.4) | 281.4 / 281.5 / 281.4 | 290.7 / 290.8 / 290.1 (+3.2%) |
| full miner loop, 60 s | 280.9 / 279.7 | 289.3 / 289.4 (+3.2%) |
| full miner loop, 60 s, against the sixteen-warp fold (271.8 / 271.9) | | 288.6 / 289.4 (+6.3%) |
| SM clock | ~2420 MHz | ~2405 MHz |
| rate / (clock x 131072) | 0.887 | 0.921 |
| registers | 235 | 255, no spill |

400 of 400 hits verified, across 312 operand draws. The pool takes it: 2 of 2 shares
accepted in 62 s at us2.pearl.herominers.com (`earn-cli`, 2026-09-25), the tile pattern
unchanged.

What else it takes:

- **Transcripts in shared.** 192 accumulators leave no room for twelve transcript words a
  lane: the lane that keeps a chunk stores its region's word, one `STS` a region a chunk,
  into 12 KB after the stages and barriers (98368 bytes in all). A column slot's two warps
  meet once a tile on a 64-thread barrier, and the first hashes the slot's 48 regions, a
  pass and a half. Nothing else orders the hand-off: the second warp can overwrite those
  words only in its next chunk-0 readout, after its stage 1 waited on EMPTY for the buffer
  every warp's stage 0 read, which the hasher releases only after hashing.
- **A ragged last row group.** m is a power of two and 192 is not a factor of it, so 683
  row groups cover 131136 rows. The noised A is allocated that long (the 64 extra rows
  zeroed once and never generated), and the fold hashes no region past row m. The attempts
  it reports are the valid regions only, so the extra work (0.05%) never counts.
- **Where the copies go.** A's after the second B pair of a stage's first k-step, behind the
  EMPTY wait; B's after the first pair of its second, then the arrival. Bench, TH/s: that
  290.3 - 290.6; B one pair later 288.2 - 288.6; A a pair later 287.8; both at the start
  of their k-steps 283.0; B a pair earlier still spills.

Tried and dropped: releasing a stage right after its last ldmatrix instead of its last mma
(-0.6%); every warp hashing its own 24 regions, which drops the column barrier but stops both
warps of a scheduler to hash (-0.3%); a relaxed shared counter for EMPTY instead of an
mbarrier arrive (-0.6% in the probe); later copies in a chunk's second stage, whose EMPTY
wait spins the most (Nsight: 2.4% of warp samples), -1% to -4%. Band depth measured flat: 8,
16 and 32 row groups.

What is left to take, priced by ablation (bench, wrong results, the build at 290.9):
without the ring's waits (`PEARL_ABLATE_RING`) 294.2 / 294.6 (+1.2%), without the fused hash
293.5 / 293.8 (+1.0%). Neither is where the rest of the distance to 313 is. Nsight has the
fold at 3.22 instructions an mma against the eight-warp fold's 2.89: more copy groups and
their `@!PT LDS` pads (0.146 an mma), the ring's spins (0.05 `LDS` an mma), and a copy's
shared writes taking twice the wavefronts, because a 64-deep stage reads half of each
128-byte line of a 2 KB-strided row.

Check after any edit to the tall fold: 255 registers or fewer and 0 spill; SASS stays two
`LDGSTS` groups a stage, the A group behind the EMPTY spin and the B group ending in
`ARRIVES.LDGSTSBAR`. Since the next section, each stage ends in a `MEMBAR.ALL.CTA` and an
`ATOMS.ARRIVE` on EMPTY, from every thread, with no branch around them. On sm_120 (the TMA
build): 255 or fewer and 0 spill, no `LDGSTS`, four `UTMALDG.3D` in the chunk loop (the
producer's two boxes a stage), and 160 of its 192 `IMMA` with B `.reuse` (`cuobjdump -sass |
grep -c 'IMMA.*reuse'`; ptxas 13.3: 248 registers, 2.672 instructions an `IMMA`), which is
what `PEARL_TALL_MMA_FENCE_MASK` and `PEARL_TALL_STAGE_FENCE` are for.

### Whole-line staging and a leaner ring: +3.6%

Six changes to the tall fold, still sm_89 only. Against v0.5.6, interleaved in one session
(4090 at 450 W, 2026-09-26), both with the hashed operand fill. The constant fill is a
separate change; "Operand values" has the two together.

| | v0.5.6 | this |
|---|---|---|
| bench, 30 s | 294.8 / 291.4 | 302.7 / 302.3 (+3.2%) |
| full miner loop, 60 s | 289.6 / 290.6 | 300.4 / 300.6 (+3.6%) |
| SM clock (full loop) | 2414 - 2421 MHz | 2440 MHz |
| instructions per mma (Nsight) | 3.22 | 3.02 |
| tensor pipe busy (Nsight) | 92.7% | 95.0% |
| registers | 255 | 253, no spill |

400 of 400 hits verified (1271 hits across 488 operand draws in 60 s). What the pool sees
does not change: same tiles, same pattern, same transcripts.

Each step against the build before it, bench, two interleaved rounds. Sessions drift about
1% with the card's temperature, so every row is against its own control:

| step | before | after | |
|---|---|---|---|
| operands in staging order: whole 128-byte lines | 293.6 / 292.6 | 296.1 / 296.1 | +1.0% |
| the k-step holds B and streams A | 298.1 / 295.9 | 295.6 / 294.7 | -0.6% |
| + every thread arrives on EMPTY (measured as a `cp.async` arrival, see below) | 295.2 / 295.1 | 297.7 / 297.8 | +0.9% |
| + copy points moved (A after m16 tile 3, B after tile 8) | 297.9 / 297.8 | 300.7 / 300.8 | +1.0% |
| + the ring's buffer roles in registers | 300.5 / 300.9 | 303.4 / 303.3 | +0.9% |
| + 64-bit source pointers | 302.2 / 302.7 | 303.2 / 303.0 | +0.2% |

- **Whole lines.** A 64-deep stage of a row-major operand reads 64 bytes of each 2 KB row,
  half a 128-byte line, and a `cp.async` writes shared memory one wavefront per global line
  it touches: Nsight counted 1253M wavefronts against 627M ideal. The draws now write the
  noised operands so that a stage of a tile is one contiguous run, and eight threads copy
  one whole line, two rows. Wavefronts drop to the ideal and the clock rises 45 MHz. This
  table measured it with an order of its own -- A in blocks of 192 rows, B in blocks of
  256, each block k/64 slabs of [rows][64]. Since v0.5.7 it is the k-blocked order sm_120's
  TMA reads, [k/64][rows][64] (`pearl_materialize16_kblocked`), which gives the same whole
  lines and measured 0.15% behind; see "Onto v0.5.7". Only the fold reads the noised operands.
  The host settles the order when it makes the context, because the first draw comes
  before the first search.
- **The EMPTY arrival.** Lane 0 used to arrive for its warp after a `__syncwarp`, with
  EMPTY expecting 8: a branch and its `BSSY`/`BSYNC` a stage. Now every thread arrives
  itself and EMPTY expects 256. The arrival is an ordinary `mbarrier.arrive`, a release,
  and the refill's `test_wait` is an acquire. So PTX orders a stage's ldmatrix reads before
  any copy into its buffer, and the hasher's transcript reads before its partner's next
  writes. On sm_89 the release is a `MEMBAR.ALL.CTA` a stage per warp, and it costs
  nothing measurable. The step table's +0.9% was measured with the arrival made through
  `cp.async.mbarrier.arrive.noinc`, which has no fence. Against that build, interleaved,
  both with the constant fill: full loop 311.5 / 311.0 -> 311.5 / 311.7, bench 313.2 /
  312.8 -> 313.1 / 312.7. In an earlier session the lane-0 form measured 310.5 / 310.6 /
  310.5 against 313.4 / 313.2 / 313.3 (bench, -0.9%). So the gain was the branch, not the
  fence. The `cp.async` arrival never shipped: it is ordered only after the thread's own
  copies, not its ldmatrix reads. In that build ptxas happened to issue it after each read
  had landed, but nothing made it do so, and past that point only timing keeps a refill off
  data not yet read.
- **The k-step holds B.** Alone it lost 0.6%, but it frees eight registers: the warp's 64
  columns of B fragments are 16, one m16 tile of A is 4, where all 96 rows of A were 24. On
  the A-holding k-step the every-thread `cp.async` arrival spilled 52 - 252 bytes.
- **Copy points.** Now counted in m16 tiles, 12 a stage. A at 3 and B at 8 was the best of
  twenty placements, its neighbours 0.2 - 2.7% behind; see `PEARL_TALL_APT` for the table.
- **The ring's roles in registers.** Stage 0 of a chunk reads buffer A and refills C, stage
  1 reads B and refills A, and the three rotate once a chunk. Each buffer's FULL and EMPTY
  sit in 128 bytes behind its rows (98688 bytes of shared in all), so every ring address is
  a role register plus a constant. ptxas keeps the roles in uniform registers, and the
  per-stage modulo, the parity selects and the barrier-address arithmetic are gone.
- **64-bit source pointers**, held a tile, with B's centred on its four slots so all of them
  are immediate offsets of one register: a B copy group's address arithmetic goes from six
  instructions to three.

Tried and dropped, bench against the same session's control:

| tried | result |
|---|---|
| the partner signals the column hand-off with `bar.arrive`, only the hasher waits | -2.4%: any named-barrier arrive, even predicated or non-`.aligned`, costs ptxas the loop's warp uniformity -- the ring roles leave the uniform registers and every copy group gets `BSSY`/`BSYNC` |
| the same on an mbarrier, plus a 32-thread `bar.sync` a warp to keep uniformity | +0.05%, noise |
| the partner hashes 16 of its own regions itself | -0.3% |
| test the next FULL a tile early, wait only if it had not completed | -1.5% to +0.2% by position |
| `nanosleep` 20 or 100 ns between failed ring tests | -0.2% (clock +8 MHz) |
| fold a region pair right after its last mma of the chunk | -2.4% (ptxas interleaves, but renames accumulators) |
| later copy points, A at 4 - 9 and B at 8 - 10 | -0.7% to -2.7% |
| the A-holding k-step with the new ring (it fits once the roles are registers) | -0.5% to -0.9% |
| band depth 8 or 32 | flat |

What is left, priced by ablation on the build before the pointers (wrong results): without
the fused hash +1.1%, without the ring's waits +0.5%. Nsight still has the ring's spins at
~3.5% of warp samples, three quarters of them on EMPTY. The transcript stores run at 4x
their ideal wavefronts (8 lanes to one bank pair), but they are 0.16% of samples.

**A 256x192 tile was not built.** Bytes per MAC are the same (1/256 + 1/192), so it could
only win on fragment loads, and it cannot. Eight 32x192 warp tiles load 2 A and 12 B
fragments per 48 mma, 0.29 an mma against 0.21 now (+40% shared reads). 64x96 and 128x48
load the same or more, and a 96- or 48-column warp tile does not hold whole regions of the
tile pattern, whose columns span 64. A 192-column tile also does not divide the 8192 column
offsets, so a batch would wrap inside a tile. Pricing 32x192's extra reads in the real fold
(four more B ldmatrix a k-step) spilled at every placement.

### Onto v0.5.7: one operand order for both builds

v0.5.7 runs the tall fold on sm_120 too, staged by TMA from operands k-blocked
[k/64][rows][64] (`PEARL_TALL_TMA`). Its sm_89 SASS is v0.5.6's, instruction for instruction
(all 16 kernels the same; `pearl_materialize16_kblocked` is new), so the numbers above
carry over. The trims and the constant fill went onto it with sm_120 left as it was: its
preprocessed device code is main's token for token, ring, stage body and shared layout
included. Only the restamp kernel's stamp loop is spelled differently (`pearl_stamp_byte`,
the same bytes). The trims are sm_89's alone: its build takes the per-thread EMPTY
arrival, the ring roles in registers with the barriers behind each buffer (98688 bytes of
shared; sm_120 kept 98368), the k-step that holds B, and the new copy points.

Later, sm_120 took the ring roles too (`PEARL_TALL_TMA_ROLES`, 99840 bytes of shared,
buffers 512-byte aligned, which SWIZZLE_64B allows -- verify-hits 359/359). RTX 5090 at
600 W, memory at 7001 MHz, one session, 3 rounds: 123.34 -> 123.79 TH/s (+0.36%, ahead in
every round). The per-thread EMPTY arrival that paid on Ada measured flat there (123.41,
and 123.78 on top of the roles), and so did the other ring ideas (producer at the fill end
123.43, a second producer warp 123.11); they stay off, as switches.

The one open choice was sm_89's operand order. The k-blocked order stores each 192-row
stage slice of A' in one 12 KB run and each 256-column slice of B' in one 16 KB run, so
the `cp.async` copies get the same whole lines as from the trims' own per-tile blocks. The
difference is where a tile's next stage sits: one k-block on, m x 64 bytes of A' and n x 64
of B' later, instead of right behind. Both builds: 253 registers, no spill, 400 of 400 hits
verified. Their SASS differs only in the stride arithmetic, a stage stride of m x 64 or n x
64 from the constant bank instead of a constant: the stretch from the first `IMMA` to the
last is one instruction longer, and the fold 2640 instructions against 2632. The copy
points (A at 3, B at 8) were kept, not swept again. Interleaved, 4090 at 450 W, one
session:

| | per-tile blocks | k-blocked |
|---|---|---|
| full miner loop, 60 s, k-blocked run first | 311.69 / 311.33 | 311.16 / 310.89 |
| full miner loop, 60 s, per-tile run first | 311.26 / 311.27 | 310.84 / 310.74 |
| bench, 30 s | 313.2 / 313.0 | 312.9 / 312.7 |
| SM clock (full loop) | 2526 - 2534 MHz | 2527 - 2530 MHz |

The per-tile order was ahead in all six pairs, by 0.1 - 0.2%: 311.39 against 310.91 in the
full loop on average (+0.15%), +0.1% in the bench. That is small, and one order for both
builds is worth more, so the k-blocked order ships. Both builds now read one order, written
by one kernel. The per-tile order's `block_rows` argument, its per-context flag and its
column-offset check are gone: a k-block has no tile boundary, so any 256 consecutive
columns of it are one run. On sm_89 the last row group's rows past m read the next
k-block's first rows, or past the last k-block the zeroed rows `PEARL_TALL_A_ROWS` pads A'
with. The fold hashes no region on them. (TMA zero-fills them instead.)

The result, full miner loop, 60 s, interleaved, 4090 at 450 W, one session (449 - 450 W in
every run):

| | TH/s | SM clock |
|---|---|---|
| v0.5.7 (hashed fill) | 291.6 / 290.2 | 2428 / 2412 MHz |
| this | 310.6 / 310.9 (+6.8%) | 2525 / 2530 MHz |
| the same work on v0.5.6, per-tile order (`perf-scratch/r10-combo-bin`) | 311.2 / 311.6 | 2526 / 2534 MHz |
| this | 310.9 / 310.7 (-0.2%) | 2526 / 2525 MHz |

A second session against v0.5.7 alone, after the review fixes (comments only; the SASS is
the same): 290.2 / 291.1 -> 311.1 / 311.5 (+7.1%), 2413 / 2427 -> 2528 / 2531 MHz.

The -0.2% against the v0.5.6 build is the order: that build's tall fold is the per-tile
build's above, instruction for instruction. Tall fold: 253 registers, no spill, 2640
instructions (v0.5.7's: 255 and 2688). Every other kernel's sm_89 SASS is v0.5.7's, the
restamp kernel included. `verify-hits`: 400 of 400 hits verified (2625 hits across 1014
salts in 120 s). `r9-fill-distinct.js`: no a_seed shared between two cards (salt base 0 and
1, stride 2); a full draw and a restamp at the same salt gave the same a_seed and root_A at
all 17 salts the cards had in common; every A and B leaf byte a hit carried was 48. With
the hashed fill, cards still share no a_seed, and a full draw and a restamp differ, as
before.

Not measured: anything on sm_120, which this card cannot run and CUDA 12.6 cannot build.
Its fold, ring and operand order are v0.5.7's. What changes there is the bytes it reads:
the constant fill is every card's default, and it has not been on a 5090.

### What a staged byte costs, and why a standalone probe underprices it

A standalone copy loop priced staging (L2 to shared by `cp.async`) at 1.87 nJ per IMMA;
the fold's own power pointed to about 2.4. Measured directly, the fold's number is right,
and nothing in the copy pattern explains the gap. The chip's load does.

All on 64 SMs, so nothing hits the cap (2700 MHz), 30 s runs, corrected to 70 C. On and
off runs of one build share their SASS: the copies are switched at run time.

| measurement | cost |
|---|---|
| fold with staging minus fold without it, interleaved x2 | 2.38 nJ per IMMA |
| of which the copies themselves (same build, copies switched off) | ~2.0 nJ per IMMA, 42 pJ per byte |
| of which DRAM and L2 misses (all tiles re-read L2-resident data) | 0.11 nJ per IMMA |

A copy of the fold's chunk loop (same launch, addresses, swizzle and copy schedule, no
readout) costs the same 42 pJ/B. Nsight Compute counts the same traffic for it and for
v0.5.5: 48 B per IMMA from L2 at 99.5% hits, 12 bank writes and 48 bank reads per IMMA,
no bank conflicts. Varying that loop, pJ per staged byte:

| loop | pJ/B |
|---|---|
| as in the fold: ldmatrix + IMMA beside the copies | 41.6 - 43.3 |
| no ldmatrix (IMMA operands from registers) | 41.6 - 42.3 |
| one copy group a chunk instead of three | 42.1 |
| half the copies | 42.3 |
| contiguous source rows instead of 2 KB-strided | -1.5 |
| L2-resident source | -2.5 |
| the fold's operand values instead of uniform bytes | +0.5 |
| the same copies, rest of the chip idle | 20 - 24 |
| the same, with 64 other SMs running IMMA or ALU work | 37 - 41 |

The last two rows are the gap. A byte costs about 1.7x more when the chip is busy, and so
does a plain LOP3: 0.28 - 0.30 nJ per warp-instruction on an idle chip, 0.51 beside the
IMMA work, on and off alternating every 3 s so both share one die temperature. The
standalone probe ran with no tensor load; every other unit energy was measured under load.
There is no copy-pattern fix. What is left is fewer bytes per MAC: at 42 pJ/B a 192x256
tile saves about 0.45 nJ per IMMA, not 0.34.

### Operand values: a constant A and B, +3.6%

A and B are the miner's own int7 matrices. The protocol fixes only the low-rank noise,
seeded from their roots, so what they hold is our choice. It costs energy: A' = sat(A +
noise) and B' go through the staging copies, ldmatrix and the tensor datapath, and every
bit that flips between one mma's operands and the next draws power.

The research build set A and B to a constant with `cudaMemset` after drawing them
(`perf-scratch/r9-ideal`, v0.5.6's fold). Bench, capped. The rows come from three research
sessions, s3, s4 and s6, and each is against that session's own run of the shipped fold:
290.9 in s3, 292.0 in s4, 292.8 in s6. Sessions drift about 1%, so compare within one.

| A = B = | session | TH/s | vs that session's shipped |
|---|---|---|---|
| 0 | s3 | 292.7 / 293.1 | +0.6% / +0.8% |
| uniform [-4, 3] | s3 | 293.2 | +0.8% |
| 16 | s6 | 300.1 | +2.5% |
| 32 | s6 | 301.5 | +3.0% |
| 40 | s6 | 302.3 | +3.2% |
| **48** | s6 / s4 | **302.6 / 302.8** | **+3.3% / +3.7%** |
| 56 | s6 | 302.5 | +3.3% |
| 63 | s3 / s4 | 301.4 / 302.1 | +3.6% / +3.5% |
| -48 | s6 | 298.4 | +1.9% |
| A = 48, B = -48 | s6 | 300.3 | +2.6% |
| A = 63 only | s4 | 295.6 | +1.2% |
| B = 63 only | s4 | 297.7 | +2.0% |

- 48 and 63 are close. The one session that ran both had 48 ahead, 302.8 against 302.1.
- The noise is in [-63, 63], so at 48 A' stays in [-15, 111] and its sign bit rarely
  flips. Zero does not help: A' is then pure noise and flips sign half the time.
- 127 measured +5.9% (s4), but it is not int7, so it was not taken further.
- Marginal energy per mma (64 SMs minus 32, uncapped): 6.61 nJ shipped, 5.81 at 48.
- The same pure-mma build (operand values loaded once and held in registers, no staging,
  uncapped) drew 40 W less with A = B = 63 than with hashed values on 128 SMs (370 -> 330
  W, 2685 MHz), and 15 W less on 64 SMs (233 -> 218 W, 2700 MHz). The saving does not
  scale with the SM count, and these runs do not show why.

What ships (`PEARL_OPERAND_CONST`, the mainnet default): a full draw memsets A and B to 48,
then writes the salt stamp over A's first 11 bytes. These are the same bytes a restamp at
that salt writes (`pearl_stamp_byte`).

The stamp is not optional. Under the constant fill every card draws the same B, root_B,
b_seed and B' for a job, so the stamp in A is all that keeps two cards apart. The research
build had none, so every salt's full draw was the same A and B, and on a multi-GPU rig
every card would search the same space for a new job until its first restamp. Measured on
that build: cores with salt base 0 and 1 drew the same first a_seed. With the stamp,
a_seed depends only on the job and the salt (`perf-scratch/r9-fill-distinct.js`):

- base 0 and base 1 (stride 2) shared none of their a_seeds, first draws included;
- every reseed on one core gave a new a_seed (33 salts, 33 a_seeds);
- a full draw at salt s and a restamp at salt s gave the same a_seed and root_A at all 16
  salts two cores had in common;
- every A and B leaf byte a share carried was 48 (leaf 0, the stamp's, was never hit).

The hashed fill stays as `PEARL_OPERAND_HASHED` (`operandFill: 'hashed'`, code 0). The
frozen parity vectors were captured with it, and it is the fallback if a pool refuses the
constant.

First measured on v0.5.6's fold. The fill does not change the fold's SASS, only the bytes
it reads. Full miner loop, 60 s, interleaved against v0.5.6, 4090 at 450 W:

| | TH/s | SM clock |
|---|---|---|
| v0.5.6 (hashed) | 288.1 / 288.3 | 2401 / 2397 MHz |
| constant fill | 298.7 / 298.6 (+3.6%) | 2487 / 2487 MHz |

`verify-hits`: 400 of 400 hits verified across 325 salts.

Then with the trims ("Whole-line staging and a leaner ring"), all in one later session,
4090 at 450 W (449 - 450 W in every run). Full miner loop, 60 s, two interleaved A/Bs:

| | TH/s | SM clock |
|---|---|---|
| v0.5.6 (hashed) | 287.7 / 288.6 | 2399 / 2407 MHz |
| trims and constant fill | 308.8 / 308.8 (+7.2%) | 2509 / 2504 MHz |
| v0.5.6's fold, constant fill | 298.6 / 298.6 | 2487 / 2488 MHz |
| trims and constant fill | 308.7 / 308.6 (+3.4%) | 2508 / 2508 MHz |

Bench, 30 s, five builds interleaved, two rounds:

| | TH/s | SM clock |
|---|---|---|
| v0.5.6 (hashed) | 289.8 / 289.6 | 2400 / 2405 MHz |
| trims, hashed | 300.8 / 300.4 (+3.8%) | 2431 / 2426 MHz |
| trims, hashed, from the combined build (`PEARL_OPERAND_FILL_CODE=0`) | 300.1 / 300.4 (+3.6%) | 2423 / 2427 MHz |
| v0.5.6's fold, constant fill | 299.9 / 300.0 (+3.5%) | 2480 / 2486 MHz |
| trims and constant fill | 310.6 / 310.4 (+7.2%) | 2510 / 2505 MHz |

In the bench the two gains multiply: +3.6% and +3.5% alone, 1.036 x 1.035 = 1.072, and
+7.2% together. In the full loop the trims add a little less on top of the constant fill,
+3.4%, against +3.6% on the hashed fill. The combined build's
tall fold is the trims' SASS, instruction for instruction (253 registers, no spill), and
with the hashed fill it benches the same as the trims' own build. `verify-hits`: 400 of 400
hits verified (2614 hits across 1008 salts in 120 s). `r9-fill-distinct.js` on it: no
a_seed shared between the two cards, and a full draw and a restamp at the same salt gave
the same a_seed at all 17 salts the cards had in common.

Then the EMPTY arrival became a release (see "Whole-line staging and a leaner ring"), and a
third session measured the result. Full miner loop, 60 s, interleaved, 449 - 450 W:

| | TH/s | SM clock |
|---|---|---|
| v0.5.6 (hashed) | 290.8 / 290.7 | 2417 / 2419 MHz |
| trims and constant fill | 311.7 / 311.4 (+7.1%) | 2535 / 2532 MHz |
| trims, hashed (the 300 row's build) | 300.8 / 301.0 | 2445 / 2445 MHz |
| trims and constant fill | 311.6 / 311.7 (+3.6%) | 2534 / 2535 MHz |

So on top of the trims the fill adds the same +3.6% it adds alone. `verify-hits`: 400 of
400 hits verified (2636 hits across 1018 salts in 120 s).

**The pool takes it.** 3 of 3 shares accepted in 198 s at us2.pearl.herominers.com
(`earn-cli`, 2026-09-26), with the constant fill on v0.5.6's fold (`perf-scratch/r9-fill-bin`).
The build with the trims as well, on v0.5.7 (`perf-scratch/r12-port-bin`): 12 of 12 shares
accepted in 240 s at us2 (`earn-cli`, 2026-09-26). If a pool ever refuses these
shares, go back to the hashed fill:

- `earn/src/shared/miner/pearlhash.js`: in `PROFILE`, set `operandFill: 'hashed'` and
  `operandFillCode: 0`. This alone switches the miner, with no CUDA rebuild: the addon
  reads `operandFillCode`.
- Same file, `operandFillCode()`: return `p.operandFill === 'constant' ? 1 : 0`, so a
  profile that names no fill gets the hashed one, like the C default below.
- `earn/native/src/pearl_config.h`: make the last field of `PEARL_MAINNET_PROFILE`
  `PEARL_OPERAND_HASHED`. That is the default for bench and for any caller that passes no
  code. It takes effect at the next build.
- `earn/test/minerSeeds.test.js`, "the operand fill code agrees with the string and
  defaults to constant": expect `PROFILE.operandFill` to be `'hashed'`, and
  `operandFillCode()` and `operandFillCode({ k: 2048, rank: 128 })` to be 0. Change the
  `{ ...PROFILE, operandFill: 'hashed' }` line to `'constant'` and expect 1: otherwise no
  test takes the `? 1` branch and earn's 100% branch coverage fails. Rename the test to
  say hashed.
- `earn/test/nativeConfig.test.js`, "both sides default to the constant operand fill, and
  it is int7": expect `PROFILE.operandFillCode` to be `defineOf('PEARL_OPERAND_HASHED')`
  and the `PEARL_MAINNET_PROFILE` initialiser to contain `PEARL_OPERAND_HASHED`. Rename it
  too.

Then fix the comments that call the constant the default (beside `PROFILE.operandFill`
and `PEARL_OPERAND_CONST`), and run `npm test` in `earn/`.

### Two warp groups instead of the per-chunk barrier: measured, not shipped

The per-chunk `__syncthreads` holds all sixteen warps in the chunk seam together. The idea
was to split them into two groups of eight by row slot (row slots 0-1, which hold the
hashers, and 2-3), two warps of each on every scheduler, each group crossing chunk
boundaries on its own named barriers, so one group's seam runs while the other keeps the
tensor pipes fed. A is already private to a group. B is not: every warp reads all 64
columns of its column slot, so the groups have to hand B to each other.

What it could be worth, from builds that race (results wrong), bench, interleaved against
the same session's base:

| build | vs base |
|---|---|
| per-chunk barrier deleted | +2.8 to +3.1% |
| each group on its own 256-thread barrier, same code; the hash at the tile seam sets the offset | +1.7 to +2.2% |
| the same, with one group staging all of B | +2.8 to +3.2% |
| the same, B split by k between the groups | +0.8 to +1.2% |
| group 1's seam placed 2 k-steps into the chunk (the form priced at +1.3% before `PEARL_FOLD_FAST_COORDS`) | +0.25% |

What correct versions got. Each passed `verify-hits.js` 400/400. Under random warp and
group delays the three protocols (rows 1, 3 and 5) passed again, 400/400, while the same
protocols with the refill wait removed failed: 82, 89 and 160 of 400 hits wrong.

| how the groups share B | vs base |
|---|---|
| each stages half the columns, as now; the trailing group publishes its half mid-chunk (offset at most half a k-step) | 0.0% |
| the same, B copies bunched into k-step 1 (offset up to 1.5 k-steps) | -3.0% |
| split by k: the leading group stages quads 0-3, the trailing group 4-7, so the leading group needs the other half only from k-step 2 (offset up to 2) | -0.4% |
| the same, chunk 0 run like every other chunk | -4.7% |
| the leading group stages all of B in whole rows | -3.0% |
| the same, offset pinned at 1.5 k-steps | -0.9% |

In the full miner loop the best of them (row 1) measured 272.6 / 272.1 against 271.6 /
272.2 TH/s, +0.2%.

Two copy patterns cost on their own, under the plain barrier: the k split, -1.2%, because
every 128-byte source row is then fetched as two 64-byte halves by different warps; one
group staging all of B, -2.3%.

Why the correct versions lose what the racing ones show. With two stages -- 96 KB of the
99 KB -- a buffer can be refilled only once BOTH groups are done with it, so the refill
waits for the slower group and starts late in the chunk. The producer then waits for its
own copies at its next chunk top, and the other group waits for the producer. Nsight,
per-PC sampling, the pinned version against its racing twin: 8.6% of stall samples on that
copy wait against 0.9%, 11.4% against 9.1% at the chunk-top barrier, and 4.15 against 3.82
warp-instructions per IMMA. Every protocol adds three or four barrier instructions a warp
per chunk, and a barrier inside the k-loop can cost ptxas the A lane base at the register
cap: the k split and the pinned version rebuild it after every chunk-top barrier, about 8
instructions. And when one group stages all of B it is the slower group, so the other
catches up behind it and the seams line up again (row 5); pinning the offset (row 6)
recovers most of that, not all.

What would change the answer: a third stage (no room at 128x256), finer stages (k64 x 3
measured -5.5% with the barrier in the probe), or a tile where the shared operand is small.
The development code (all six protocols, the stress and negative-control switches) is kept
out of the fold.

### Measuring, and proving a build correct

- `node hashrate.js <pearl_core.node> 60` -- the app's own number: one core, a synthetic
  job at the hardest target, the core's hashrate samples averaged after a warm-up, with
  clock and power sampled alongside.
- `node verify-hits.js <pearl_core.node> 40` -- the correctness gate. It sets an easy
  target so the core hits about once a batch, then recomputes the first 400 hits in JS the
  way the pool verifies it: Merkle proofs, the seed chain, the noise, the cumulative fold
  and the transcript hash. The run crosses hundreds of operand redraws. The shipped core
  passes 400/400; a build with the barrier deleted fails 349 of 353. A faster build that
  does not pass is not faster, it is broken. A longer run counts more hits but checks no
  more of them, so it cannot catch an error rarer than about one hit in a few hundred.
- `bench.exe <secs>` (built from `bench.cu`) -- fold plus finalize, no redraws. It benches
  the mainnet fill, now the constant one; `PEARL_OPERAND_FILL_CODE=0` benches the hashed
  fill from the same binary.

Mind the operand fill when comparing builds. `hashrate.js` and `verify-hits.js` hand the
core the JS profile, so they run the constant fill, but a core from v0.5.7 or older has no
`operandFillCode` and always runs the hashed one. Its bench does too. An A/B against such
a build compares the fills as well as the code, which is about 3.5% on its own (see
"Operand values"). To compare code alone, bench the new build with
`PEARL_OPERAND_FILL_CODE=0`. `hashrate.js` and `verify-hits.js` take no profile option:
edit their `addon.createCore(PROFILE, {})` to `addon.createCore({ ...PROFILE,
operandFillCode: 0 }, {})`.

Run one at a time: two processes on the card corrupt each other's timing.

## Power is the binding constraint on sm_120

The 5090 is hard power-capped running this kernel: steady state sits at exactly the
cap with `SW Power Cap: Active` and only 47 C, so there is no thermal headroom to
recover. Hashrate is bought with watts, almost perfectly linearly:

| power limit | TH/s | TH/s per W |
|---|---|---|
| 450 W | 117.7 | 0.262 |
| 525 W | 135.6 | 0.258 |
| 575 W | 145.4 | 0.253 |
| 600 W | 148.7 | 0.248 |

At ~0.25 TH/W, 300 TH/s would need 1200 W. **The only route to the goal is roughly
doubling MACs per joule** -- there is no clock left to find.

Where the power goes, per launch (49.08 ms, from Nsight):

| | |
|---|---|
| tensor instructions | 2,147,483,648 |
| ALU instructions | 6,305,857,536 (2.9 per MMA) |
| total instructions | 13,310,255,104 |
| L2 traffic | 104 GB -> 2.12 TB/s |
| DRAM traffic | 3.18 GB -> 65 GB/s (3.5% of peak) |
| shared loads | 412 GB -> 8.4 TB/s |

The tensor pipe is 69.41% active, essentially identical to Ada's 69.9%, and the
arithmetic closes exactly: 0.694 x 482 x (1095/2400) = 152.6 against 152.5 measured.
Nsight's percentages are per ACTIVE CYCLE, so they hide the clock -- the kernel is
issuing MMA just as densely as on the 4090 and losing purely on frequency.

Ruled out as the power sink: DRAM (3.5%), shared-memory bank conflicts (64,620 of
3.2e9 wavefronts), operand data toggling (a variant with 8 distinct operand sets
measured 156 W against 153 W for constant operands), and codegen fallbacks (sm_120
emits native LDGSTS and LDSM, and its integer op counts match sm_89).

### The memory clock experiment

Locking the memory clock exposes how the budget is split:

| memory clock | SM clock | TH/s |
|---|---|---|
| 14001 MHz | 1322 MHz | 157.9 |
| 7001 MHz | 1353 MHz | 158.4 |
| 810 MHz | **2659 MHz** | 99.7 |

At 810 MHz the SM clock DOUBLES inside the same 600 W -- the memory domain is worth
roughly half the power budget -- but hashrate collapses, because L2 is in that clock
domain and the fold pulls 2.12 TB/s through it. At 2659 MHz with bandwidth intact the
kernel would be around 368 TH/s. So **L2 traffic per MAC is the thing to attack**;
DRAM traffic is already irrelevant.

## The ablation table does not transfer from Ada

Built with `-DPEARL_ABLATE_BARRIER`, `-DPEARL_ABLATE_STAGING`, `-DPEARL_ABLATE_TRANSCRIPT`.
These produce WRONG results by construction and exist only to price each layer.

| build | 5090 | clock | 4090 (tuning log) |
|---|---|---|---|
| shipped | 155.1 | 1281 MHz | 216.0 |
| barrier deleted | 155.3 (**0%**) | | 246.7 (14.2%) |
| transcript deleted | 155.5 (1.4%) | 1274 MHz | |
| staging deleted | 203.0 (**24%**) | 1514 MHz | 242.6 (~13%) |
| staging + transcript | 206.0 | 1492 MHz | |

**The two cards have opposite bottlenecks.** On Ada the block-wide barrier was the
biggest single item at 14.2% and staging cost ~13%. On Blackwell the barrier is free
-- consistent with its 1.11 barrier-stall ratio against Ada's 2.47 -- and staging
costs 24%, nearly double. So the tuning log's "biggest identified item: the block-wide
barrier" and its proposed half-chunk pipelining do not apply here; chasing them on
this card would have been wasted work.

Note the clock column: deleting staging lifts the SM clock from 1281 to 1514 MHz
inside the same 600 W. Staging is not costing time so much as costing *power*, and
power is costing clock.

### TMA is available and works

`cp.async.bulk.shared::cluster.global` compiles for sm_120 AND executes correctly on
the 5090 (verified byte-exact, 2048/2048). Ada has no such instruction, so this is a
genuinely Blackwell-only lever: one bulk copy can replace the 3072 separate 16-byte
`cp.async` requests a block issues per chunk, along with their address arithmetic --
which is where the 6.3e9 ALU instructions (2.9 per MMA) are going.

The kernel's manual XOR swizzle (16-byte unit q of row r stored at q XOR (r mod 8))
is exactly TMA's SWIZZLE_128B pattern, so the shared layout would not have to change.

## Tile geometry is already optimal on Blackwell

The register file is 65536 per SM on sm_120, the same as Ada, so the accumulator
bound the tuning log derives holds unchanged. Measured:

| tile | regs | spill | TH/s |
|---|---|---|---|
| 128x256 (shipped) | 128 | 0 | 154.7 |
| 256x128 | 128 | 0 | 154.3 |
| 256x256 | - | 5456 B | fails |
| 128x512 | - | 1416 B | fails |

Threads per block: 256 -> 138.6, 512 -> 153.6, 1024 -> exceeds registers. 512 stays.

## The ceiling at 600 W, and what it rules out

Ablating every layer in turn, all at the card's maximum 600 W power limit:

| build | TH/s | SM clock |
|---|---|---|
| shipped | 157.0 | 1176 MHz |
| - staging | 208.7 | 1444 MHz |
| - staging - ldmatrix | 292.9 | 2058 MHz |
| - staging - ldmatrix - transcript | **299.4** | 2062 MHz |

(Re-measured on v0.4.5, which includes the compile-time geometry work from #209.
That lifted both the floor and this ceiling -- it was 151.2 / 287.9 before -- without
changing the conclusion.)

The last row is a kernel that issues every `mma` but moves no operands at all. It is
not a miner -- it computes nothing usable -- and it still only reaches **287.9 TH/s**.

**That bounds the problem.** Any real fold must feed its tensor cores, so on this card,
at its hard 600 W maximum, the practical ceiling is well short of 300. Getting past that needs either more
power (600 W is `power.max_limit`) or an algorithm that moves less data per MAC -- not
tuning. Note the clock column: every gain here is bought by freeing power, not by
removing time.

## Measured and rejected on sm_120

| Attempt | Result | Why it lost |
|---|---|---|
| **TMA (`cp.async.bulk.tensor.2d`)** | rejected, **implemented and measured in the real kernel: -27%** | See below. |
| **All-at-barrier staging** | 142.6 vs 152.6 | Ada interleaves one slot per k-step to stop the memory pipeline backing up. That reasoning was about barrier cost and the barrier is free here, so the opposite schedule was worth pricing -- it still loses. Keep the interleave. |
| **Locking the SM clock** | no change | 1400 -> 155.5, 1600 -> 152.9, 1800 -> 150.5, 2100 -> 148.6, 2400 -> 147.3 TH/s. The power cap decides the operating point regardless; locking high only raises voltage and costs a little. |
| **Bigger tiles** | spills | 256x256 spills 5456 B, 128x512 spills 1416 B. The register file is 65536 per SM as on Ada, so the accumulator bound is identical and 128x256 stays optimal. |
| **Warp specialization to free registers** | premise false | Staging costs only 2 registers (REG:128 shipped vs 126 with staging ablated), so moving it to dedicated warps cannot buy tile size. Worth checking because Blackwell's free barrier voids one of the two reasons Ada rejected it -- but the register premise fails independently. |


## The trustworthy denominator: pure mma at the fold's launch geometry

Two earlier readings here were wrong and are worth recording as traps.

**A short kernel does not reach the power cap.** `mmapeak` first measured 482 T-MAC/s
at "153 W and 2400 MHz", which made the tensor pipe look nearly free and the memory
path look like the whole problem. It was sampling a 1.5 ms kernel: the card had not
settled. Run back to back until sustained, pure mma also sits at **600 W**, the same
cap as everything else.

**Block count is part of the denominator.** At constant total work:

| blocks | T-MAC/s |
|---|---|
| 170 | 422.9 |
| 1360 | 427.4 |
| 10880 | **429.4** |
| 43520 | 395.2 |
| 131072 (what the fold launches) | **347.2** |

So the fold's own launch geometry costs 19% before it does anything, and 347 T-MAC/s
-- not 482 -- is what a perfect kernel at this shape would approach.

### Persistent blocks: tried, regressed

Walking several tiles from a smaller resident grid should recover that 19%, paying
the per-block prologue once instead of per tile. Implemented as a grid-stride loop
over a virtual block index, it is monotonically **worse**:

| resident blocks | TH/s |
|---|---|
| 131072 (one per tile) | **158.9** |
| 43520 | 157.0 |
| 10880 | 154.7 |
| 2720 | 151.5 |
| 1360 | 151.1 |

The pure-mma probe has nothing to stage, so shrinking its grid costs it nothing. The
fold restages chunk 0 for every tile, and a smaller grid serialises those stages with
less work in flight to hide them. The scheduling win is real but the staging loss is
larger, which is another way of seeing that staging dominates this kernel on Blackwell.

## The honest denominator, and what it says about the target

Two of the numbers above were measured wrong, and the corrected ones change the
conclusion. Pure `mma` must be measured at the FOLD'S geometry -- its block count
and its occupancy -- or it is not a denominator at all.

| probe | T-MAC/s |
|---|---|
| `mmapeak`, 170 blocks, unconstrained | 422.9 |
| `mmablocks`, 10880 blocks | 429.4 |
| `mmablocks`, 131072 blocks (the fold's launch) | 392.4 |
| `mmaoccupancy`, 131072 blocks + 96 KB shared (1 block/SM, as the fold runs) | **393.3** |
| **the fold** | **147.0** |

Occupancy is not the excuse: constraining pure `mma` to one block per SM with the
fold's own 96 KB of shared memory costs it nothing (393.3 vs 392.4). The fold runs
at **37% of what this card will issue at the fold's own geometry**, and the whole of
that gap is the memory path -- staging and `ldmatrix` -- not the tensor pipe, not the
clock, and not the launch shape.

**So 300 TH/s is not out of reach for the silicon.** It is 76% of 393. SRBMiner
reaches 91% of Ada's ceiling on a 4090, so a fold that fed its tensor cores that well
would clear 300 here comfortably. What it is out of reach of is *tuning*: every knob
that exists has been swept, and the remaining 2.7x lives in how operands reach the
tensor cores.

### Everything swept, for the record

| knob | result |
|---|---|
| `PEARL_BLOCK_GROUP` | 1 is best on sm_120 (+4%); SHIPPED |
| threads/block | 256 -> 138.6, 512 -> 153.6, 1024 -> exceeds registers |
| tile geometry | 128x256 and 256x128 tie; anything larger spills |
| `col_batch` | 1024 -> 138.3, 2048 -> 139.3, 4096 -> 72.3 (clamped) |
| `m` = `n` | 65536 -> 140.6, 131072 -> 140.0, 262144 -> 139.5 (flat; Ada's knee is absent) |
| power limit | linear, ~0.25 TH/W, 600 W is `power.max_limit` |
| locked SM clock | no gain at any of 1400-2400 MHz |
| memory clock | 810 MHz doubles the SM clock but starves L2; net loss |
| staging schedule | Ada's interleave still wins (152.6 vs 142.6) |
| TMA staging | works, but slower AND more power for the same bytes |
| persistent blocks | monotonically worse (158.9 -> 151.1) |
| running staging cursors | +0.5%; ptxas already strength-reduces it |


## TMA staging, implemented and rejected

Worth writing down properly, because the first rejection was on weak evidence and
the second is not.

The first pass compared TMA against cp.async in a standalone benchmark of the
staging pattern: 4.14 vs 4.29 TB/s, and more power for the same bytes. That test
was **bandwidth-saturated**, so it could not show the thing TMA is supposed to win
-- one instruction in place of 3072 -- and the real kernel is not bandwidth-bound
(L2 sits at 32%). So it was re-done inside the fold itself.

It drops in cleanly. The staged regions really are contiguous rectangles: the
column indices expand to `base*16 + col0` for consecutive `col0`, because
`COLS_MASK` is `0xF` and `pearl_expand_offset` is then just a shift. `boxDim`'s
256 cap is fine -- A stages 128 rows and B 256. TMA's SWIZZLE_128B is exactly the
`q XOR (r mod 8)` the staging already does by hand, so the shared layout and every
`ldmatrix` below it are untouched. Descriptors are built once per allocation with
`cuTensorMapEncodeTiled`, fetched via `cudaGetDriverEntryPointByVersion` so the
addon keeps its cudart-only link line. Two gotchas: the shared destination must be
128-byte aligned (dynamic shared is 16 by default, and TMA answers "misaligned
address"), and each barrier's phase flips on every use, so barrier `c & 1` is
waited with parity `(c >> 1) & 1`.

Measured, alternating, on a clean card:

| | TH/s |
|---|---|
| cp.async (shipped) | 145.6 / 142.3 |
| TMA | 106.3 / 104.4 |
| | **-27%** |

And the reason is visible in the SASS: the TMA build is **bigger**, 760 instructions
against 520. The copies were never the instruction cost -- the cp.async ones sit
inside loops and amortise -- while the mbarrier init, expect_tx and parity waits are
all new, and one thread issuing both tiles serialises what 512 threads previously
did cooperatively. On a card that is power-bound rather than issue-bound, that trade
goes the wrong way.

The code is not kept: the kernel signature change and the per-allocation descriptors
are an always-on cost for a path that is off and slower.


## Where the 2.7x actually lives: operand traffic, and it costs ~450 W

**Corrected.** The section that followed originally claimed the fold was at 85% of
pure `mma` per cycle and lost only on clock. That rested on an Nsight profile of a
single 2.9 ms pure-`mma` launch, which is far too short to reach the power cap -- it
ran at a boost clock the fold never sees. Measured properly, running back to back
until sustained:

| | throughput | power | clock |
|---|---|---|---|
| pure `mma`, fold geometry | 387.7 T-MAC/s | **149 W** | 1744 MHz |
| the fold | ~145 TH/s | **600 W** | ~1230 MHz |

Pure `mma` does 2.7x the work on a QUARTER of the power, and is not power-limited at
all. So the fold's ~450 W of extra draw is bought entirely by moving operands:
104 GB through L2 and 412 GB through shared per launch, or 1.98 and 7.8 TB/s. At
plausible energies per byte those two land in the right order of magnitude to
account for the gap; nothing else in the kernel is close.

Which of the two dominates is answered by the warp-tile result below: a square warp
tile cut instructions per `mma` by 19% and shared reads per `mma` by a third, and
bought **2%**. Shared traffic is therefore not the expensive half. That leaves L2 --
and L2 traffic per MAC is `1/bM + 1/bN`, fixed by the 128x256 BLOCK tile, which is
pinned by the accumulator's claim on the register file. A 256x256 tile needs 65536
accumulator registers, the entire file, at any thread count.

**That is the wall, stated precisely.** Not instructions, not occupancy, not the
launch shape, not the instruction mix: operand bytes per MAC, bounded by registers.
Cutting it needs an accumulation scheme that does not hold the whole tile in
registers for the whole of k -- a different algorithm, not a different parameter.

## The earlier framing (kept for the record): instructions per mma

Profiling pure `mma` and the fold with the same metrics finally reconciles the
numbers, and the answer is not what the earlier notes assumed.

| | tensor pipe active | IMMA / SM / cycle | clock |
|---|---|---|---|
| pure `mma`, fold geometry | 94.0% | 0.228 | **2.38 GHz** |
| the fold | 78.1% | 0.195 (85% of pure) | **1.23 GHz** |

**Per cycle the fold is already at 85% of pure `mma`.** The whole of the 2.7x
throughput gap is CLOCK: both sit at the 600 W cap, and the fold draws roughly
double the power per cycle, so it runs at half the frequency. At pure `mma`'s clock
it would be ~322 T-MAC/s -- past the 300 target.

That reframes the problem. The fold is not short of issue slots or arithmetic; it
is short of *joules*. The thing to minimise is energy per MAC, and the proxy for
that is the 6.2 non-tensor instructions it issues per `mma` -- instruction issue and
register-file traffic, not the 104 GB of L2 or the 412 GB of shared, which at
plausible pJ/byte account for tens of watts, not hundreds.

Instructions per mma is set by the warp tile: the bigger and squarer it is, the more
`mma` each `ldmatrix` and each address computation serves. Which is bounded by
registers -- and that is where a latent bug was hiding.

### `__launch_bounds__(512)` was pinned, so no other thread count could be tried

The kernel hardcoded `__launch_bounds__(512)`, which caps ptxas at 65536/512 = 128
registers a thread **whatever the launch actually uses**. A 256-thread build is
entitled to 256 and got 128, so it spilled -- which is why earlier 256-thread sweeps
measured badly and looked like evidence against fewer, fatter warps. It was evidence
about the bounds. The bound now follows `PEARL_FOLD_THREADS`, and the default
512-thread build is SASS-identical on sm_120 and sm_89.

With that fixed, a square warp tile becomes measurable:

| geometry | warp tile | TH/s |
|---|---|---|
| 512 threads, WARP_ROWS 4, RT 2, CB 4 (shipped) | 32x64 | 143.4 / 139.8 / 138.6 |
| 256 threads, WARP_ROWS 2, RT 4, CB 4 | **64x64** | **145.4 / 143.4 / 141.9** |
| | | **+1.4 / +2.6 / +2.4%** |

Both cover the same 128x256 block tile, so L2 traffic is unchanged; the square warp
tile serves more `mma` per `ldmatrix`. Note this is the geometry the 4090 log
rejected outright -- "square 64x64 warp tile: 130, and it lost 40%" -- on the
grounds that warp count for latency hiding beats shared traffic. That reasoning was
about a card with time to spare; on a power-bound card the trade inverts.

**Not shipped here.** `PEARL_FOLD_THREADS` and the warp-grid constants are read by
the HOST as well as the device, so unlike `PEARL_BLOCK_GROUP` this cannot be gated
on `__CUDA_ARCH__` alone. The host now reads the block size off the loaded fold
(`PEARL_FOLD_WIDE_WARPS`), and the 4090 ships eight 64x64 warps. The 5090 still runs
sixteen: the Ada fold it would take has not been measured on this card.


## Why 300 TH/s is out of reach for this tiling, as arithmetic

L2 traffic per MAC is `1/bM + 1/bN`, and the accumulator claims `bM x bN` int32
registers for the whole k-loop. For a square tile of side b that is `2/b` bytes per
MAC against `b^2` registers -- so **halving the traffic costs four times the
registers**:

| tile | B/MAC | accumulator registers | share of the 65536-register file |
|---|---|---|---|
| 128x128 | 0.01562 | 16384 | 25% |
| 128x256 (shipped) | 0.01172 | 32768 | 50% |
| 181x181 | 0.01105 | 32761 | 50% |
| 256x256 | 0.00781 | 65536 | **100%** |

The shipped tile already spends half the register file. Spending ALL of it -- which
leaves nothing for fragments, addresses or transcripts, so it is not buildable -- cuts
traffic by only 33%. And a square 181 is both barely better than 128x256 and illegal:
`m` must be a power of two for the BLAKE3 commitment fold.

Now put that against the power budget. The fold moves 1.70 TB/s through L2 at
145 T-MAC/s, and ~450 W of its 600 W goes to operand movement (pure `mma` does 2.7x
the work on 149 W). 300 T-MAC/s needs **3.51 TB/s** through the same path -- roughly
double the bytes, so roughly double the memory power, against a cap that is already
binding. The tile cannot cut traffic enough to pay for it.

**So 300 TH/s is unreachable by tiling this algorithm on this card.** Not "no ideas
left": the traffic-versus-registers exchange rate is quadratic and the register file
is fixed.

### The one identified escape, and an honest estimate of it

Stop reading operands and regenerate them on-chip, trading L2 bandwidth for integer
ALU. The 4090 log rejected this on throughput grounds -- BLAKE3 costs ~21 int-ops a
byte, so it needs far more integer issue than an SM has. But that was a card with
power to spare, and this one is power-bound; the same inversion has already shown up
twice here (the grid band depth and the square warp tile both flipped sign from Ada).

Redoing the arithmetic for sm_120: at pure `mma`'s rate of ~1306 MACs/SM/cycle and
0.0117 operand bytes per MAC, regeneration needs ~321 int-ops/SM/cycle against the
~128 INT32 lanes a Blackwell SM has. So the kernel becomes integer-bound at ~40% of
the tensor rate -- but it would run at pure `mma`'s power and clock rather than
throttled. 388 x 0.40 is ~155 T-MAC/s: **about break-even with today, not a win.**

Regenerating only ONE operand halves both the traffic and the integer cost and is
the version worth pricing properly. It is a large piece of work and the estimate
above is not tight enough to promise anything.


## The power budget, decomposed

Measured sustained, same card, same session:

| build | TH/s | power | clock |
|---|---|---|---|
| shipped | 139.6 | **600 W** | 1046 MHz |
| - staging | 188.5 | **600 W** | 1303 MHz |
| - staging - ldmatrix | 264.7 | **600 W** | 1858 MHz |
| pure `mma` | 380.5 T-MAC/s | **145 W** | 1289 MHz |

Two things to read off this. **Pure `mma` is the only build that does not hit the
cap** -- it uses a quarter of the budget. And ablating work does NOT reduce power:
every fold variant sits at exactly 600 W and simply converts the freed power into
clock. So an ablation's power reading says nothing about what it removed; only its
throughput does.

Cross-referencing the standalone staging benchmark, which drew 551 W moving
4.27 TB/s, and the fold's own 1.98 TB/s of L2 traffic, the budget decomposes to
roughly: `mma` ~145 W, staging ~258 W, everything else ~200 W.

That is the whole argument against 300 in one line: **300 T-MAC/s needs ~2x the
operand bytes, so ~516 W of staging alone, before a single `mma` issues.** The cap
is 600 W and `power.max_limit` will not move.

It also explains why a 4090 can host a 309 T-MAC/s miner while this card cannot be
tuned to it. Ada is not power-bound at this workload -- the 4090 log records raising
its limit 450 -> 480 W and gaining 0.2%, because a voltage/boost ceiling bound it
instead. Blackwell hits a hard wall the 4090 never reaches, so a design tuned for
Ada's constraint does not transfer, and the reverse holds too: three separate
conclusions in that log (band depth, square warp tile, cache policy) flip sign here.


## Independent check: the best public miner on THIS card

Everything above infers the wall from our own kernel. The obvious way to test that
inference is to run somebody else's kernel on the same silicon, and SRBMiner 3.6.1
supports `pearlhash` -- its release notes even call out "improvements for 5000 series
GPUs". It is the miner the 4090 tuning log benchmarks against, at 309 T-MAC/s there.

Run on this 5090, same pool, same address, our miner stopped:

| 60 s sample | hashrate | power | efficiency |
|---|---|---|---|
| 1 | 166.82 TH/s | 600.0 W | 0.278 TH/W |
| 2 | 171.22 | 600.0 W | 0.285 |
| 3 | 171.27 | 600.0 W | 0.285 |
| 4 | 171.58 | 600.0 W | 0.286 |
| 5 | 172.35 | 600.0 W | 0.287 |

1 hr average 171.14 TH/s, 4 shares accepted, clock 1005 MHz.

**Three things follow, and they settle the question.**

1. **300 TH/s is not achievable on this card by any known miner.** The best public
   implementation reaches 171. The 309 figure is a 4090 number, and Ada is not
   power-bound at this workload -- the tuning log records raising its cap 450 -> 480 W
   for 0.2% because a voltage ceiling bound it instead. It does not transfer.
2. **SRBMiner hits the identical wall.** 600.0 W on every sample, at 1005 MHz -- the
   same cap, the same throttled clock as our fold. An independent codebase written by
   someone with every incentive to beat it lands in exactly the same place.
3. **Our gap to best-in-class is ~20%, not 2x.** Ours reports ~140-145 TH/s live
   against SRBMiner's 171 (which also takes a 2% dev fee off the top). That is worth
   chasing, and it is a different-sized problem from the one the 300 target implies.

This is what the earlier sections were missing: they were right about the mechanism
but had no way to know whether the bound was OUR kernel's or the CARD's. It is the
card's.
