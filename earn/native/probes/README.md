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
| **TMA (`cp.async.bulk.tensor.2d`)** | rejected | Works on this GPU (verified byte-exact) and the fold's manual `q XOR (r mod 8)` swizzle is exactly TMA's SWIZZLE_128B, so it drops in. But on the fold's own staging pattern it is not faster (4.14 vs 4.29 TB/s) and draws MORE power for the same bytes: 600 W / 2776 MHz against cp.async's 551 W / 2925 MHz. On a power-bound card that is strictly worse. Note `boxDim` is capped at 256, so 384 staged rows needs two tiles. |
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
