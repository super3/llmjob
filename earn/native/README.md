# Pearl mining core (CUDA)

Our own PearlHash core. This is the compute half of the miner; the protocol and
lifecycle half is JavaScript, under `earn/src/shared/miner/` and
`earn/src/main/pearlMiner.js`.

## Why this exists

Every third-party Pearl miner we evaluated is blocked for a product like ours:

| Miner | Blocker |
|---|---|
| PeakMiner, alpha-miner, WildRig, SRBMiner, BzMiner | proprietary; Windows Defender deletes the binaries; licences forbid redistribution |
| Muskwak/Open-Pearl-Miner | open source, but its licence **mandates a 2% dev fee on any distribution** — a fork was DMCA'd in July 2026 for lowering it |
| pearl-research-labs (official) | ISC, but **sm90-only** (H100/H200) and a vLLM plugin, not a pool miner |

Owning the core removes all three problems at once: no dev fee, no
redistribution restriction, and a binary **we compile and code-sign**, which is
what stops Defender flagging it as `Trojan:Wacatac.H!ml`.

## Provenance

Written against the algorithm specification in the ISC-licensed
[`pearl-research-labs/pearl`](https://github.com/pearl-research-labs/pearl)
(zk-pow crate) plus protocol behaviour captured from the live pool. It is **not**
derived from any dev-fee-licensed miner — deliberately, so there is no fee to
retain and no licence to comply with beyond ISC attribution.

## Status

| Piece | State |
|---|---|
| Stratum protocol, job/target math, MDL, share assembly, lifecycle (JS) | **done, 70 tests, 100% gate** |
| BLAKE3 device compression, keyed hashing | written |
| Low-rank noise generation | written |
| GEMM + jackpot transcript fold | written — **scalar/dp4a reference path** |
| `pearl_host.cu` (device memory + pipeline driver) | written |
| JS reference oracle + known-answer vectors | **done** — BLAKE3 validated against the official vectors |
| CI compile + link gate (Windows + Linux) | **done** — `.github/workflows/native-core.yml` |
| Tensor-core (`mma.sync` int8) mainloop | **not written** — this is the performance work |
| Benchmarked on a GPU | **no** — no GPU on any runner; needs a real box |

The kernels are written to be *bit-exact with the reference first, fast second*.
A fast core that disagrees with the spec mines nothing, so correctness is
cross-checked against the JS reference before any tensor-core work begins.

**Nothing here has been compiled on the development box** — it has no CUDA
toolkit and no MSVC, and the account is not an administrator, so there is no way
to install one. CI covers the gap: `native-core.yml` compiles AND links the
addon on Windows and Linux runners, which is what catches a drift between
`pearl_core.cc`'s `extern "C"` declarations and `pearl_host.cu`'s definitions —
that failure is a link error, invisible to code review.

What CI cannot do is run it: the runners have no GPU. Device semantics are
pinned instead by the JS reference oracle (`src/shared/miner/reference.js`) and
its frozen known-answer vectors, whose BLAKE3 foundation is validated against
the official published BLAKE3 test vectors — so the arithmetic is checked
against evidence outside this repo, not merely against itself.

Those vectors pin OUR semantics and make the JS and CUDA sides provably agree.
They do not by themselves prove agreement with the Pearl network; that needs a
share accepted by a real pool. What they buy is that when a rejection happens,
it points at the protocol rather than at arithmetic.

## Building

Requires the CUDA Toolkit 12.x and a host compiler (MSVC on Windows, gcc on
Linux). On Windows `nvcc` **hard-requires** MSVC's `cl.exe` as its host
compiler — there is no way around that, which is why this cannot be built on a
box without Visual Studio.

```bash
# 1. Compile the CUDA half into a static library
mkdir -p cuda-build   # NOT build/ — node-gyp rebuild cleans that directory
# On Linux add -Xcompiler -fPIC: the addon is a shared object.
nvcc -O3 -std=c++17 -arch=sm_89 -c src/pearl_kernel.cu -o cuda-build/pearl_kernel.o
nvcc -O3 -std=c++17 -arch=sm_89 -c src/pearl_host.cu   -o cuda-build/pearl_host.o
nvcc -lib cuda-build/pearl_kernel.o cuda-build/pearl_host.o -o cuda-build/pearl_cuda.lib  # .a on Linux

# 2. Build the N-API addon around it
npx node-gyp rebuild
```

`.github/workflows/native-core.yml` does exactly this for sm_86/89/120 on both
platforms, so a green run means it compiles and links even though nothing there
can execute it.

`CUDA_PATH` is picked up automatically; override the arch for other cards
(`sm_86` Ampere, `sm_89` Ada, `sm_120` Blackwell).

The addon lands at `build/Release/pearl_core.node`, which is exactly where
`src/main/pearlCore.js` looks for it. When it is absent — as on any machine
without a CUDA build — the host reports "Pearl core is not built" and stops
cleanly rather than crashing.

## Interface

```js
const core = addon.createCore(profile);   // profile = PROFILE from pearlhash.js
core.setJob({ header, target, jobId });   // Buffer(76), BigInt, string
core.on('hit', (hit) => {});              // { jobId, nonce, jackpotHash, aSeed, bSeed, proof }
core.on('hashrate', (thPerSec) => {});
core.on('error', (err) => {});
core.stop();
```

Every hit is re-verified against the current job's target in JS before it is
submitted (`pearlMiner._onHit`), so a core bug can waste work but can never push
a bad share to the pool and earn a ban.
