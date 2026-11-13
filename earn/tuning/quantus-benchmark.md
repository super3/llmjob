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
| Intel core (`quantus_sycl.node`) | Being written (agent quantus-sycl) |
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
| RTX 4090 | rig01 | | not run yet | | | | | | | | |
| RTX 30 | Salad | | not run yet | | | | | | | | |
| RTX 50 | Salad | | not run yet | | | | | | | | |
| RTX 20 | Salad | | not run yet | | | | | | | | |
| RX 6600 | Chedr's rig (1400 MHz cap) | | not run yet | | | | | | | | |
| RX 6800 | Chedr's rig (1400 MHz cap) | | not run yet | | | | | | | | |
| RX 7900 XTX | Salad | | not run yet | | | | | | | | |
| RX 9070 | Salad | | not run yet | | | | | | | | |
| Arc A750 | rented rig, card 0 | | not run yet | | | | | | | | |

Chedr's cards run with their core capped at 1,400 MHz by their owner, so compare them by "% of
SRBMiner", not by MH/s.

## Ceilings: how much room each card has

Before tuning, each agent counts the integer instructions per nonce (theirs, and quantus-miner's open
kernel) and divides by the card's measured instruction rates. That gives the most any miner could reach on
the card, and tells us where a large gain is possible.

| Card | Instructions per nonce (by pipe) | Measured rates | Ceiling (MH/s) | SRBMiner / ceiling | Official / ceiling | Notes |
|---|---|---|---|---|---|---|
| RTX 4090 | | | | | | IMAD on the FMA pipe, IADD3/LOP3/shifts on the ALU pipe |
| RTX 50 | | | | | | NVIDIA describes Blackwell's cores as unified INT32/FP32: check whether INT32 issue per SM doubles over Ada |
| RX 6600 / 6800 | | | | | | Full 32-bit multiplies (v_mul_lo/hi_u32, v_mad_u64_u32) believed quarter rate, 24-bit ones (v_mul_u32_u24, v_mad_u32_u24) full rate: confirm with a micro-benchmark |
| Arc A750 | | | | | | No native 64-bit integer multiply: count what a 64 x 64 multiply compiles to |

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
