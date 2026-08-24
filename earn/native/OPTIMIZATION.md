# Closing the performance gap

Measured on an RTX 4090, mainnet geometry (k=2048, rank=128, 4x8 tile), against
the frozen parity vectors after every change.

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

## What did not — and what each ruled out

| Attempt | Result | What it eliminated |
|---|---|---|
| int8 decomposition + `dp4a` | 0.93 → 0.37 | See below — structural |
| Shared-memory staging of the tile | 0.93 → 0.41 | Not bandwidth: L1 already serves these reads |
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

**0.93 TH/s, ~316x short.** Five hypotheses tested, five eliminated. The
remaining gap is not a tuning problem in this kernel: bandwidth, latency,
occupancy and launch overhead have all been measured and excluded, which points
at the loop *structure* rather than its constants.

A competitive miner almost certainly does not fold one tile per warp at all. It
computes large contiguous GEMM tiles with `mma.sync` and harvests many
transcripts from each result, so the operand loads amortise across hundreds of
attempts instead of tens. That is a different kernel, not a faster version of
this one, and it needs int8 operands — which the decomposition above cannot
supply for a chunk-wise fold.

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
