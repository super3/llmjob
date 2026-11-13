# Quantus (QPoW) benchmarks

Every Quantus measurement goes here. Pearl's stay in `benchmark.md`. Started 13 November 2026 on PR #297
(branch `quantus/support`). Speeds are in MH/s unless a row says otherwise. Every rate has its clock,
power and temperature beside it, and every comparison runs both miners on the same card.

## Where we stand

| | Status (13 Nov) |
|---|---|
| CPU reference (`earn/src/shared/quantus/`) | Matches every published vector: 2 permutations, 18 `hash_bytes` inputs, 5 nonce hashes with their midstates |
| CPU core (`earn/src/main/quantusCore.js`) | Works end to end against a fake pool. About 4,000 nonces/s on one thread, for tests only |
| NVIDIA core (`quantus_cuda.node`) | Being written (agent quantus-cuda) |
| AMD core (`quantus_hip.node`) | Being written (agent quantus-amd) |
| Intel core (`quantus_sycl.node`) | Passes `verify-core.js` on an Arc A750: 93 MH/s against SRBMiner's 116 on the same card. Tuning goes on (agent quantus-sycl) |
| Live pool test | Not done: needs a real Quantus address. Only a fake login was tried at Kryptex and HeroMiners; both refused it, as they should |

## The work per nonce

`hash = Poseidon2.hash_squeeze_twice(header || nonce)` over Goldilocks (p = 2^64 - 2^32 + 1): width 12,
4 + 4 full rounds, 22 partial rounds, S-box x^7. A share is a hash below `(2^512 - 1) / difficulty`.

The header and the first half of the nonce are fixed for a job, so a miner starts from a midstate.
That leaves **2 permutations a nonce**: block 3 (nonce bytes 32-63) and block 4. The hash's top 64 bits
are `bswap64(state[0])` after block 4, so a fifth permutation runs only for a near-hit.

Each permutation is about 470 field multiplies in the S-boxes and 264 in the partial-round diagonal.
Each multiply is 64 x 64 -> 128 bits plus a Goldilocks reduction, and there's no memory traffic. So
speed comes down to integer multiply throughput per watt.

Sources: chain `qpow-math`, qp-poseidon 3.1.0, QIP-0013, quantus-miner `pow-core`. Details in the scope
notes (`S/quantus/scope.md`).

## Published speeds: what to beat

No competitor binary was run for these; they are published figures. SRBMiner improved `quantus` in
almost every release from 3.6.4 (9 Sep) to 3.8.6 (8 Nov), so they move.

| Card | Speed (MH/s) | Power | Source |
|---|---|---|---|
| RTX 4090 | 1,130 | 300 W | whattomine GPU page, 13 Nov |
| RTX 4090 | 880-900 | 420-450 W cap | quantus-miner `docs/vast-gpu-benches.md` (official open CUDA kernel, September) |
| RTX 3080 | 440 | 170 W | whattomine |
| RTX 3080 Ti | 380 | 330 W cap | quantus-miner Vast benches |
| RTX 2080 Super | 285 | | Kryptex device page |
| RX 7900 XTX | 400 | 220 W | whattomine |
| RX 6950 XT | 340 | 210 W | whattomine |
| RX 6800 XT | 300 | 180 W | whattomine |
| RX 6800 | 230 | 160 W | whattomine |
| RX 6600 | 120 | 90 W | whattomine |
| GTX 1660 Super | 130 | 60 W | whattomine |

Miners: SRBMiner-MULTI (closed source, 2% fee on `quantus`, AMD, NVIDIA and Intel) and the official
quantus-miner (open source, Apache-2.0, CPU, WGSL and CUDA). Kryptex lists no other.

## Our results

One row per card and core build. A row counts only once `verify-core.js` passes all five checks on that
card (vectors, exact range, top of range, overflow, and a timed 2^30-nonce run with every hit checked).

| Card | Where | Core (commit) | Ours (MH/s) | Clock | Power | Temp | SRBMiner 3.8.6, same card | % of SRBMiner | Official kernel, same card | verify-core | Date |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CPU, 1 thread | rig01 | JS reference (605c4698) | ~0.004 | | | | | | | vectors only | 13 Nov |
| RTX 4090 | rig01 | `quantus_cuda` build g (COMMIT) | 943-946 | 2,575-2,585 MHz | 449 W (450 W limit) | 76-77 °C | 1,414-1,422 (2,475-2,490 MHz, 450 W, 77-78 °C) | 67% | 930-932 (2,569-2,576 MHz, 450 W, 76-78 °C) | all five PASS | 13 Nov |
| RTX 3080 | Salad, driver 617.42 | `quantus_cuda` build g (COMMIT) | 357 | 1,830 MHz | 366 W (370 W limit) | 82 °C | 538-541 (1,770-1,785 MHz, 368 W, 82-83 °C) | 66% | 348-349 (1,826-1,830 MHz, 365-368 W, 83 °C) | all five PASS | 13 Nov |
| RTX 50 | Salad | | not run yet | | | | | | | | |
| RTX 20 | Salad | | not run yet | | | | | | | | |
| RX 6600 | Chedr's rig (1400 MHz cap) | | not run yet | | | | | | | | |
| RX 6800 | Chedr's rig (1400 MHz cap) | | not run yet | | | | | | | | |
| RX 7900 XTX | Salad | | not run yet | | | | | | | | |
| RX 9070 | Salad | | not run yet | | | | | | | | |
| Arc A750 | rented rig, card 0 | `quantus_sycl` (8c8149c4) | 92.7-93.4 | 2,100-2,150 MHz | 190 W (power limit) | not exposed (fans 1,500-2,200 rpm, no thermal flag) | 115-117 (2,150-2,200 MHz, 190 W; thermal flag after 70 s) | 80% | not run | all five PASS | 13 Nov |

Chedr's cards run with their core capped at 1,400 MHz by their owner, so compare them by "% of
SRBMiner", not by MH/s.

## Ceilings: how much room each card has

Before tuning, each agent counts the integer instructions per nonce (theirs, and quantus-miner's open
kernel) and divides by the card's measured instruction rates. That gives the most any miner could reach on
the card, and tells us where a large gain is possible.

| Card | Instructions per nonce (by pipe) | Measured rates | Ceiling (MH/s) | SRBMiner / ceiling | Official / ceiling | Notes |
|---|---|---|---|---|---|---|
| RTX 4090 | Ours (build g, sm_89): 25,057 SASS a nonce: ALU 13,500, IMAD 4,335 (IMAD.X, IMAD.MOV, IMAD.IADD), IMAD.WIDE 6,895, other 327. Official (`mining.cu` at 7d6af15): 26,400: ALU 13,856, IMAD 4,736, IMAD.WIDE 5,421, IMAD.HI 1,714, other 673. Nsight Compute's dynamic counts match both to the instruction | Per SM a clock (`cuda/tools/ubench.cu`): IMAD 2.0; IADD3, LOP3, SHF 1.97; IMAD and IADD3 together 3.9; IMAD.WIDE 1.0 and IMAD.HI 1.0 (two multiplier slots each); IMAD.WIDE with IADD3 at 1:1 1.92 and at 1:2 1.95, so a WIDE also takes an ALU slot; FFMA 3.8 | 1,880 at 2,485 MHz (1,950 at 2,580) if only the 5,416 32x32 products a nonce cost anything (1,000 multiplies x 4 and 472 squares x 3, each IMAD.WIDE two multiplier slots). Each kernel's own bound (ALU slots, a WIDE counted as one): ours 1,037 at 2,580 MHz, official 1,004 at 2,573 | 75% (1,416 at 2,485 MHz) | 48% (931 at 2,573 MHz); ours 48%. Both reach 91-93% of their own bound | All three miners sit at the 450 W limit. SRBMiner takes 7,190 SM-cycles a 32 nonces, so it needs at most 14,400 multiplier and 14,400 ALU slots a nonce; we use 18,100 and 20,400. Moving our IMAD.X/IMAD.MOV to the ALU in the cubin made us 8% slower (`cuda/tools/rebalance.py`) |
| RTX 3080 | Same SASS as sm_89 (ours 25,057, official 26,400) | Same per-SM rates as the 4090, to the third digit (68 SMs) | 717 at 1,785 MHz, 735 at 1,830, by the same product count. Own bounds at 1,830 MHz: ours 390, official 378 | 75% (539 at 1,785 MHz) | 47% (348.5 at 1,826 MHz); ours 49%. Both at 92% of their own bound | At the 370 W limit |
| RTX 50 | | | | | | NVIDIA describes Blackwell's cores as unified INT32/FP32: check whether INT32 issue per SM doubles over Ada |
| RX 6600 / 6800 | | | | | | Full 32-bit multiplies (v_mul_lo/hi_u32, v_mad_u64_u32) believed quarter rate, 24-bit ones (v_mul_u32_u24, v_mad_u32_u24) full rate: confirm with a micro-benchmark |
| Arc A750 | Ours: 8,676 int-pipe cycles an EU a nonce (69,400 SIMD8-lane instructions; 997 more on the float pipe, all `mov`). Of that, the 5,416 32x32 products a nonce are 2,031 (3 instructions each: `mul`, `mach`, `mov` from acc); the rest is carries and reduction | add, add3, mul, mach, mov: 1 SIMD8 instruction a clock an EU on the int pipe (`qtc_rates.cpp`, card 0, 2,100-2,300 MHz) | 111 at 2,150 MHz for our instruction count; 474 if only the products cost anything | SRBMiner 116 at 2,150 MHz is 104% of our code's ceiling, so it runs fewer instructions than we do (at most 8,300 a nonce) | not counted | Power-bound at 190 W for every miner. A 64 x 64 product is 12 instructions, the field multiply with reduction 37-39. We reach 83% of our own ceiling |

## Revenue per card (whattomine, 13 Nov)

| Card | Quantus | Pearl on whattomine | Pearl on our miner |
|---|---|---|---|
| RTX 4090 | $2.84/day (1.13 GH/s) | $5.28/day (330 TH/s) | about the same |
| RTX 3080 | $1.10 | $1.60 (100 TH/s) | |
| RX 7900 XTX | $1.00 | $0.77 (48 TH/s) | about $0.88 (55 TH/s) |
| RX 6800 | about $0.59 (230 MH/s) | | about $0.30 (19 TH/s at a 1400 MHz cap) |

The founder's whattomine link (900 MH/s at 230 W, $0.10/kWh): about 0.049 QTC a day, $2.32 revenue,
$0.55 power, $1.77 profit. On RTX cards Pearl pays about twice what Quantus does. On RDNA 3 the two are
close, and on RDNA 2 Quantus pays about twice Pearl.

Market: one exchange (SafeTrade, QUANTUS-USDT), 0.00068682 BTC a coin (down 12% in a week), 24-hour volume
1.195 BTC against a daily emission of 1.245 BTC.

## Pools

| Pool | Share of network (38.7 TH/s) | Fee | Server | Share difficulty |
|---|---|---|---|---|
| Kryptex | 18.6 TH/s (48%) | 3% by its API (its article says 1%) | `qtc[-us,-eu,...].kryptex.network:7049`, TLS 8049 | fixed 18,253,611,008 |
| HeroMiners | 2.29 TH/s (6%) | 1% | `<region>.quantus.herominers.com:1155` | fixed 1e10 |
| LuckyPool, Suprnova, Himpool, 1Miner | not measured | 1% | `eu.lproute.com:5660`, `quantus.suprnova.cc:7072` | |

All of them use the same JSON-RPC stratum (`login`, `job`, `submit`, `keepalived`). Not seen yet: a job
from Kryptex or HeroMiners, an accepted share, and how Kryptex names a worker for QTC.

## How to measure

- `node earn/native/quantus/verify-core.js --addon <file> --device N` for correctness and the timed run.
- Compare builds on the same card, taking turns, with an A/A twin of the control. Report the twin's
  spread beside every gain.
- Competitor miners are observed only: run them and read their hashrate, clock and power. Never unpack,
  disassemble, debug or profile them.
- Record clock (MHz), power (W) and temperature (°C) beside every rate.

## Log

Newest last. One entry per run: the card, what was compared, the numbers with the twin's spread, and what
it means.

- **13 Nov, scope (PR #297).** CPU reference and CPU core built; vectors all pass.
  Published speeds and revenue collected above. 139 new tests; full earn suite 3,571 passed. No GPU
  run yet.
- **13 Nov, Arc A750, first Intel core (quantus-sycl, 8c8149c4).** Card 0 of the rented rig, under the
  card-0 lock and a temperature guard; cards 1-4 kept mining Pearl. The rig exposes no GPU temperature;
  fans and the i915 throttle flags stand in for it. Each bench is 8-10 s of back-to-back 16.8M-nonce
  scans (`earn/native/quantus/sycl/bench.js`); the twin is the same build run again.
  - `verify-core.js`: all five checks PASS on every build below. First build 80.4 MH/s, final 93.4.
  - Builds, in turn: first arithmetic (Plonky2's reduce) 79.9; one 32-bit fold for the reduction
    85.0-85.3 (twin 85.13/85.29); one counter a work-item instead of 4: 88.2-88.8 (16 a work-item 79.7);
    linear layers on sums of 32-bit halves (8c8149c4) 92.7-92.8 (+4.8%). All at 190 W (PL2 flag on),
    2,100-2,200 MHz; the faster build runs about 50 MHz lower, as the power limit binds.
  - Lost: 256-register mode 80.7 (no spills, 4 threads an EU); sub-groups of 8 (35.5) and 32;
    one copy of the rounds for both permutations 11.7 (EUs idle at 99 W); S-box outputs carried
    unfolded through the linear layers: 9% fewer instructions but 4.9 KB of spills (not timed).
  - SRBMiner 3.8.6 (`--algorithm quantus --gpu-id 0`, observation only) against a local stub pool
    that checks each share on our reference: 115.3-117.5 MH/s on its own counter, 6 shares, all good,
    2,150-2,200 MHz, 190 W. The temperature guard stopped it after 70 s on card 0's thermal flag,
    which none of our runs raised.
  - Ceiling: every integer instruction measured (add, add3, mul, mach, mov) issues at one SIMD8
    instruction a clock an EU (`qtc_rates.cpp`). Our build needs 8,676 int-pipe cycles a nonce an
    EU, so 111 MH/s at 2,150 MHz; we reach 83% of it. SRBMiner's 116 means it needs at most 8,300.
    The products are only 2,031 of our 8,676; the rest is carries and reduction.
  - Not done: a pool run with a real address (there is none yet); quantus-miner's open kernel on Arc.
- **13 Nov, RTX 4090 (rig01), first NVIDIA core (quantus-cuda, build g, COMMIT).** Under the rig01-gpu
  lock with the app stopped; 450 W limit, no clock or memory locks. Timing is `cuda/tools/qcbench.cu`:
  back-to-back scans for 25-30 s, ours and quantus-miner's `mining.cu` (pinned at 7d6af15, unchanged,
  its own geometry: 256 threads, 4,096 blocks, 32M nonces a launch) built the same way, taking turns.
  - `verify-core.js`: all five checks PASS on builds d and g (big check 947 and 951 MH/s).
  - Builds, in turn: d (mul.lo/hi.u64 products, the 128-bit fold as one IMAD.WIDE with carry) 935 and
    927 (A/A twin); g (squares from three partial products, Q96 folds and the 12-term sums on the ALU)
    945.5 and 942.6 (twin 0.3%); official 930-932. Ours and the official run at the same work per
    clock; all at 450 W, 2,570-2,600 MHz, 71-78 °C.
  - Lost: the official-style fold through IMAD.HI, -5% (884); 3 or 2 blocks an SM (80 and 104
    registers), -1% and -2%; rounds unrolled, -2%; 5 blocks of 192, -6%; every fold exact (no 2^-33
    slip a multiply), +30% instructions, not timed. Moving IMAD.X/IMAD.MOV/IMAD.IADD to the ALU in the
    cubin (`cuda/tools/rebalance.py`, checked exact with `qcdrv.cu`): -8% (870); half of them: -4%.
    Rewrites next to an IMAD gave wrong hits, so a value crossing pipes needs more cycles than ptxas
    leaves on one pipe.
  - SRBMiner 3.8.6 (`--algorithm quantus`, observation only) against `cuda/tools/mockpool.js`, a local
    stub pool that checks every share on our reference: 1,414-1,422 MH/s on its API for 150 s, 210 of
    210 shares valid at difficulty 1e9 (about 1.4-1.5 GH/s by count), 2,475-2,490 MHz, 450 W, 77 °C.
  - Nsight Compute (our kernel and the official one only): multiplier pipe 81-86% busy, ALU 51-61% by its
    count, issue 52-59%. ubench shows why the ALU is still the limit: an IMAD.WIDE takes an ALU slot
    that Nsight files under the multiplier pipe.
- **13 Nov, RTX 3080 (Salad qcuda-3080-1, driver 617.42, 370 W limit), build g.** `cuda/tools/qbox.sh`
  at babc5dca (same kernel as g): CUDA 12.8.1 and Node 22 installed on the box, the core built there.
  - `verify-core.js`: all five checks PASS (big check 367 MH/s).
  - 60 s legs, in turn: ours 336.3 (1,688 MHz, 332 W, first leg, clocks still rising) and 357.3
    (1,830 MHz, 366 W, 82 °C); official 349.0 and 348.0 (1,826-1,830 MHz, 365-368 W, 83 °C, twin 0.3%).
    We are 2.4% ahead of the official kernel at the same clock.
  - SRBMiner 3.8.6 on the same box and stub pool: 538-541 MH/s for 240 s, 1,770-1,785 MHz, 368 W,
    82-83 °C; 28 of 28 shares valid at difficulty 4e9.
  - ubench: the same per-SM rates as the 4090.
