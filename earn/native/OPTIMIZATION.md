# Closing the performance gap

Measured on an RTX 4090, mainnet geometry (k=2048, rank=128, 4x8 tile), against
the frozen parity vectors after every change.

## Every number below this line was measured with a broken instrument

Read the table as a record of what was TRIED, not of what it achieved. Two
independent faults made the figures meaningless, and both are now fixed:

1. **The hashrate was never a rate.** `EmitHashrate` was passed
   `attempts * DAF / 1e12` with no division by elapsed time, so it reported
   work-per-batch. It came out identical to the last digit across four thousand
   samples no matter how fast the card ran.

2. **The miner stalled after about five seconds.** That same per-batch emission
   was a `BlockingCall` several hundred times a second, and the worker thread
   eventually blocked inside it and never returned -- GPU to 0%, no hits, no
   error. So most of every measurement window was spent doing nothing. The same
   binary read 0.07 TMAC/s over 40 seconds and 1.27 TMAC/s over 12.

The current figure, measured with a working instrument on a miner that does not
stall, is **5.35 TH/s** (81.6M regions/s) at the mainnet geometry -- **55x short**
of the 296 TH/s target.

The lesson is the one this project keeps relearning: an unvalidated instrument
is worse than no instrument, because it produces numbers confident enough to
optimise against. The parity vectors exist precisely because correctness had the
same problem, and they are what caught the noise construction being wrong.

## The unit, because it caused a five-order-of-magnitude misreading

Hashrate is **multiply-accumulates per second, not attempts per second**. The
protocol scales the jackpot bound by the work an attempt costs:

```rust
/// ...in proportion to the work one attempt costs: the hashed tile size
/// times the dot product length.
fn difficulty_adjustment_factor(config) -> usize {
    tile_size(config) * config.dot_product_length()   // 32 * 2048 = 65536
}
```

Read as attempts/sec, a competitor's 296 TH/s would need 3e14 BLAKE3 hashes a
second — impossible — and made the gap look like 6e9, i.e. hopeless. In the right
unit it is 4.5e9 attempts/sec, or 2.96e14 MACs, which is **45% of the card's int8
tensor-core peak**: an ordinary well-tuned GEMM.

## What worked

| Change | Result |
|---|---|
| Materialise the noised operands once per job, not per attempt | 0.001 → **0.53 TH/s** |
| One warp per region, `__shfl_xor_sync` instead of block reduction | 0.53 → **0.74 TH/s** |
| `int4` vectorised operand loads | 0.74 → **0.93 TH/s** |
| **Split into a partials GEMM + a gather** (removes a 4x redundancy) | 0.93 → **2.08 TH/s** |
| Reuse each A row across all eight columns | 2.08 → **2.92 TH/s** |
| Tune m to the cache/parallelism sweet spot (6144) | 2.92 → **3.61 TH/s** |

### The 4x redundancy, which none of the memory experiments could have found

A region folds 32 cells, each a dot product of
`A'[rows_pattern[i] + row_off]` against `B'[cols_pattern[j] + col_off]`. Across a
batch `row_off` sweeps every row, so **row R is reached by four different
regions** — once per `rows_pattern` element — and the fold recomputed its dot
product every time:

| | |
|---|---|
| MACs per batch, as written | 268,435,456 |
| distinct dot products | 67,108,864 |

`pearl_partials` now computes each distinct partial once into `D[chunk][r][c]`
and the fold gathers from it. D is 2 MiB, L2-resident. Beyond the arithmetic
saving this changes the *shape* of the work: D is a dense `[m x 8]` GEMM with a
k-reduction, which is what `mma.sync` wants — the per-warp tile fold never was,
which is why every attempt to speed that fold up kept failing.

### m is a tunable, and the curve is sharp

`m` and `n` are the miner's own choice, not protocol. Measured:

| m=n | operands | hashrate |
|---|---|---|
| 1024 | 16 MiB | 0.91 TH/s |
| 2048 | 32 MiB | 1.78 TH/s |
| 4096 | 64 MiB | 2.95 TH/s |
| **6144** | **96 MiB** | **3.61 TH/s** |
| 8192 | 128 MiB | 1.93 TH/s |
| 16384 | 256 MiB | 1.63 TH/s |

Rising while extra rows buy parallelism, collapsing once the working set stops
fitting near L2.

## What did not — and what each ruled out

| Attempt | Result | What it eliminated |
|---|---|---|
| int8 decomposition + `dp4a` | 0.93 → 0.37 | See below — structural |
| Shared-memory staging of the tile | 0.93 → 0.41 | Not bandwidth: L1 already serves these reads |
| Shared-memory staging of B in the partials pass | 2.92 → 2.85 | Same again: 4 KiB, warp-uniform, already broadcast by L1 |
| Batch/partials width 16384 to fill the GPU | — | Not occupancy: the larger working set costs more than the warps gain |
| Staging with padded stride (no bank conflicts) | 0.93 → 0.42 | Confirms the above; padding was not the issue |
| Four independent accumulators | 0.93 → 0.93 | Not the dependency chain — nvcc already split it |
| 16 warps/block instead of 8 | 0.93 → 0.94 | Not occupancy |

Batch timing rules out launch overhead too: 288 µs of compute against ~25 µs of
launch and copy.

### Why the int8 decomposition cannot work here

Expanding `(A+nA)(B+nB)` and exploiting the low-rank noise collapses three terms
into rank-length dot products against per-job precomputes:

```
acc(r,c) = Sum_t A[r,t]*B[c,t] + Sum_j E_BL[c,j]*(P+S)[r,j] + Sum_j E_AL[r,j]*Q[c,j]
```

It is an identity — parity held — but a losing trade, because **a chunk is
exactly `rank` elements long**. The chunk dot product it is meant to amortise is
itself only rank long, and so are both corrections:

| per chunk, per cell | loads | MACs |
|---|---|---|
| int32 materialised | 1024 B | 128 |
| int8 + corrections | 1536 B | 384 |

1.5x the traffic and 3x the arithmetic. The decomposition only pays when
contracting over the whole of k (2048) against rank-length (128) corrections — a
16x amortisation — and this fold never does, because every chunk must land in its
own transcript lane.

## Where that leaves it

**3.61 TH/s, ~82x short**, from 0.93 when the memory experiments ran out. The
structural change was worth more than every constant-factor attempt combined,
which is the lesson: the eliminations were correct, but they were eliminating
explanations for the wrong kernel.

The partials GEMM now runs at roughly 9e11 MACs/s against an int32 ceiling near
4e13 on this card — about 2%. So there is still a large factor available inside
the current int32 formulation, before tensor cores enter the picture at all.

### An unresolved correctness question, recorded rather than buried

The mining configuration names its arithmetic `Int7xInt7ToInt32`: int7 inputs,
int32 output. Our operands are the *noised* values, which reach ~5e5 and are
therefore int32 inputs, and the products overflow int32 and wrap. Our JS oracle
wraps identically, so parity holds — but parity only proves the two
implementations agree, not that either matches the network.

If the protocol really multiplies int7 by int7, the noise must be applied
somewhere other than straight onto the operand, and this core computes the wrong
function no matter how fast it gets. Settling that needs a share accepted by a
real pool, or the reference's own GEMM traced end to end.

## What would unblock further work

1. **A local CUDA toolchain with Nsight Compute.** Every measurement here is
   end-to-end wall-clock through a 6-minute CI round trip; five blind hypotheses
   cost more than one profile would have. The dev box has the GPU but no toolkit
   and no administrator rights.
2. **The pool's own work-to-candidate accounting**, to confirm the DAF reading
   against a live submit rate rather than inferring it from the reference source.

Whatever comes next, `test/minerDeviceParity.test.js` stays the gate: it pins the
exact transcript the device produces, and a faster core that changes it mines
nothing.
