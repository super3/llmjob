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

### Where it ended: 223 -> 280 TH/s at the same 450 W

| step | full miner loop |
|---|---|
| v0.5.2 core | 223 |
| restamp A between salts instead of redrawing both operands | 229 |
| + hash in the fold's epilogue, no 1 GiB transcript round trip | 234 |
| + persistent fold, next tile's chunk 0 staged under the last chunk | 241 |
| + per-group staging, mid-k-step copies, one-barrier seam, serpentine bands | 264 |
| + ldmatrix lane bases held for the kernel, compile-time `active` | 270 |
| + tile coordinates by shift and mask, not divides | 273 |
| + a tile pattern the accumulators hold, eight 64x64 warps a block | **280** |

The 241 and 264 steps are gated to sm_89 (`PEARL_FOLD_PERSISTENT`, `PEARL_FOLD_GROUP_STAGE`,
`PEARL_FOLD_SERPENTINE`), because only a 4090 has run them. Ampere and Blackwell keep one
block per tile and the block-wide walk; run on this card with their settings forced, those
paths measure +1.8% and +1.9% over the previous core rather than a regression, and their
chunk loops compile to 370 and 484 instructions against 385 and 451 before.

The last two rows are gated to sm_89 too (`PEARL_FOLD_LANE_BASES`,
`PEARL_FOLD_FAST_COORDS`). Measured interleaved against v0.5.5 in the full loop: +2.3%,
then +3.5% with both (262.6 / 262.8 -> 272.0 / 271.8; v0.5.5 measured ~1.5 TH/s under its
264 that day, so the rows carry the gains onto 264), 400/400 hits verified. Both shorten
the chunk and tile seams, where all sixteen warps wait at the barrier together and each
instruction costs about 0.12% of the rate. Both sit on the 128-register knife edge: holding
the fast-coordinate shift in a register instead of recomputing it cost ptxas the lane bases
and measured 265.0 against 273.0. Check the SASS after any fold edit: the first `LDSM` should
be a few instructions after `BAR.SYNC`.

The last row is gated to sm_89 as well (`PEARL_FOLD_WIDE_WARPS`; the tile pattern it needs is
every card's). It measured 271.6 -> 279.8 against the row before it, interleaved the same
day (+3.0%); see "Eight 64x64 warps a block" below.

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

### Measuring, and proving a build correct

- `node hashrate.js <pearl_core.node> 60` -- the app's own number: one core, a synthetic
  job at the hardest target, the core's hashrate samples averaged after a warm-up, with
  clock and power sampled alongside.
- `node verify-hits.js <pearl_core.node> 40` -- the correctness gate. It sets an easy
  target so the core hits about once a batch, then recomputes **every** hit in JS the way
  the pool verifies it: Merkle proofs, the seed chain, the noise, the cumulative fold and
  the transcript hash. The run crosses hundreds of operand redraws. The shipped core
  passes 400/400; a build with the barrier deleted fails 349 of 353. A faster build that
  does not pass is not faster, it is broken.

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
