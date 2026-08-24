# Closing the performance gap

Measured on an RTX 4090, mainnet profile (k=2048, rank=128, 4x8 tile):

| Stage | Hashrate | Note |
|---|---|---|
| Block per region, operands rebuilt per attempt | ~0.001 TH/s | couldn't finish a batch at m=n=131072 |
| Operands materialised once per job | 0.53 TH/s | the O(rank²) work left the hot path |
| One warp per region, shuffle reduction | 0.74 TH/s | 1.4x — ruled out synchronisation |
| Vectorised int4 loads | **0.93 TH/s** | 1.26x — confirmed memory bound |
| PeakMiner, same card | 296 TH/s | ~45% of the card's int8 tensor-core peak |

**Gap: ~317x.**

## The unit, because it caused a five-order-of-magnitude misreading

Hashrate is **multiply-accumulates per second, not attempts per second**. The
protocol scales the jackpot bound by the work an attempt costs:

```rust
fn difficulty_adjustment_factor(config) -> usize {
    tile_size(config) * config.dot_product_length()   // 32 * 2048 = 65536
}
```

Read as attempts/sec, 296 TH/s would need 3e14 BLAKE3 hashes a second, which no
GPU can do — and made the gap look like 6e9, i.e. hopeless. In the right unit it
is 4.5e9 attempts/sec and 45% of the card's int8 peak: an ordinary well-tuned
GEMM.

## Why it is memory bound

The fold reads two **int32** operands per MAC. At 0.93 TH/s that is ~7.4 TB/s of
requests, essentially the 4090's L2 ceiling. Compute is not the constraint;
bytes-per-MAC is.

The operands are int32 because the noised value
`A'[r,t] = A[r,t] + Σ_j E_AL[r,j]·E_AR[j,t]` reaches ~5e5 and does not fit int8.
That is the thing to remove.

## The decomposition that fixes it

Never materialise `A'`. Expand the product and keep the raw int7 operands:

```
acc(r,c) = Σ_t (A[r,t] + nA[r,t])·(B[c,t] + nB[c,t])
         = Σ_t A[r,t]·B[c,t]        ← int8 × int8, dp4a, the dominant term
         + Σ_t A[r,t]·nB[c,t]
         + Σ_t nA[r,t]·B[c,t]
         + Σ_t nA[r,t]·nB[c,t]
```

With `nA = E_AL·E_AR` low-rank, the last three collapse to rank-length dot
products against per-job precomputes:

```
Σ_t A[r,t]·nB[c,t]  = Σ_j E_BL[c,j]·P[r,j]        P = A·E_BRᵀ        [m, rank]
Σ_t nA[r,t]·B[c,t]  = Σ_j E_AL[r,j]·Q[c,j]        Q = B·E_ARᵀ        [n, rank]
Σ_t nA[r,t]·nB[c,t] = Σ_j E_BL[c,j]·S[r,j]        S = E_AL·(E_AR·E_BRᵀ)  [m, rank]
```

so, folding the two that share `E_BL[c]`:

```
acc(r,c) = dp4a(A[r], B[c], k) + (P[r]+S[r])·E_BL[c] + E_AL[r]·Q[c]
```

`P`, `Q`, `S` cost `m·k·rank` once per job — about 1e9 ops, tens of microseconds.

**What it buys:** the dominant term drops from 4 bytes/element to 1 (4x less
traffic) and `dp4a` does four MACs per instruction. Per attempt the operand
traffic falls roughly 98 KiB → 30 KiB.

## Then: stage the shared columns

Regions in a batch already share a column offset (`col_off = (region/m) % n`, and
a batch is smaller than `m`), so every warp in the grid reads the *same 8* B
columns. As int8 those are 8 × 2048 = 16 KiB — they fit in shared memory, where
int32 (64 KiB) did not. Staging them per block divides B traffic by the warps per
block.

## Then: tensor cores

With int8 operands in shared memory, the inner product becomes
`mma.sync.aligned.m16n8k32.s32.s8.s8.s32`. This is the step that reaches the
~45% of peak the competing miners run at, and it is only reachable *after* the
operands are int8 — which is why the decomposition comes first.

## Order of work

1. int8 decomposition + `dp4a` — removes the int32 operands (est. 4-8x)
2. Shared-memory staging of the batch's B columns (est. 2-3x)
3. `mma.sync` int8 tensor-core mainloop (the remainder)

Every step must keep `test/minerDeviceParity.test.js` green: it pins the exact
transcript the device produces, and a faster core that changes it mines nothing.
