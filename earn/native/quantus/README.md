# Quantus (QPoW) cores

Quantus mining is experimental and off by default (`--coin quantus` on the CLI). The JavaScript side is
done: the reference hash, the pool client, the miner engine and a CPU core
(`src/shared/quantus/`, `src/main/quantusEngine.js`, `src/main/quantusCore.js`). What is missing is a GPU
core for each vendor. This file is the contract those cores build to.

## What a core computes

QPoW hashes `header (32 bytes) || nonce (64 bytes)` with Poseidon2 over the Goldilocks field
(p = 2^64 - 2^32 + 1; width 12, rate 8, 8 full rounds, 22 partial rounds, S-box x^7) and squeezes 512
bits. A share is a hash below `(2^512 - 1) / difficulty`. Sources: the chain's `qpow-math/src/lib.rs`,
`qp-poseidon` 3.1.0 (`src/poseidon2.rs`, `src/lib.rs`), QIP-0013. Our port is
`src/shared/quantus/poseidon2.js`; the constants are there.

The host keeps the sponge state after the header and nonce bytes 0..31 (the midstate), so a nonce costs two
permutations: block 3 (nonce bytes 32..63, eight little-endian u32 words) and block 4 (the terminator and the pad).

A core does not count through block 3. The host hands it a **line** of nonces, `w[i] = base[i] + STEP[i] * t` for
t = 0, 1, 2, ..., with `STEP = (-4, 17, -11, 3, 4, -17, 11, -3)`. That step is chosen so that after the permutation's
first linear layer only elements 0 and 4 change with t (`x0 = X0 + 35 t`, `x4 = X4 - 35 t`). The other ten S-boxes
of the first round, and the first round's linear layer of them, are the same along the whole line, so the host does
them once. `src/shared/quantus/qpow.js` has the algebra and the layout of a nonce; a line has 252,645,136 points
(every word must stay a u32). Per point a core runs:

    // line = C[0..12], X0, X4: 14 field elements from the host (qpow.lineWork)
    a = (X0 + 35 t)^7                     // mod p
    b = (X4 - 35 t)^7
    for g in 0..3, k in 0..4:             // M = (2, 1, 1, 3)
      s[4g + k] = C[4g + k] + M[k] * (a * (g == 0 ? 2 : 1) + b * (g == 1 ? 2 : 1))
    the first permutation from its second round's S-boxes (s already holds that round's constants)
    s[0] = s[0] + 1; s[1] = s[1] + 1      // the encoding's terminator and the sponge's pad
    permute(s)
    top64 = bswap64(canonical(s[0]))      // the hash's first 8 bytes, read big-endian
    hit if top64 <= targetHigh

`qpow.indexTop64` in `src/shared/quantus/qpow.js` is this loop in JavaScript. That is 226 S-boxes and a little over
16 linear layers a nonce, where starting at the first layer takes 236 and 17. The fifth permutation (the second
squeeze) and the rest of the hash are the host's job, for hits only. It is all 64-bit modular multiply and add on
twelve registers: compute-bound, no memory traffic.

## The interface (version 2)

Each core is an N-API addon. The loader (`quantusCoreFactory`) looks for these files beside the app, in
`resources/native` in the desktop app, and in `earn/native/quantus/build/`. The Intel one is looked for
first in a `sycl/` folder in each of those places: that is the Pearl Intel core's folder, which holds the
SYCL runtime, and on Windows a DLL is found only beside the `.node` that loads it.

| Vendor | File | Source folder |
|---|---|---|
| NVIDIA (CUDA) | `quantus_cuda.node` | `earn/native/quantus/cuda/` |
| AMD (HIP) | `quantus_hip.node` | `earn/native/quantus/amd/` |
| Intel (SYCL) | `quantus_sycl.node` | `earn/native/quantus/sycl/` |

`QUANTUS_CORE_PATH=<file>` loads one file and nothing else. `QUANTUS_CORE=cpu` loads the CPU core.

The addon exports:

    QUANTUS_API             2
    devices()               [{ index, name }]   the cards it can open, in its own numbering; [] when none
    createCore({ device })  a core on that card; throws an Error saying why not

A core has:

    device                  { index, name, vendor }    vendor: 'nvidia', 'amd' or 'intel'
    scan(work)              Promise<{ hits: Buffer, overflow: boolean }>
    close()                 frees the card; a scan after it rejects

    work = {
      line:       Buffer(112) 14 field elements, each a u64 little-endian, each below p: C[0..12], X0, X4
      start:      Buffer(8)   the first point t on the line, u64 big-endian
      count:      number      points start .. start + count - 1; 1 to 2^32; never past 2^32 - 1
      targetHigh: Buffer(8)   u64 big-endian
      maxHits:    number      at most this many hits back
    }

`hits` is 8 bytes a hit, each a point t as a u64 big-endian, in any order. `overflow` is true when the scan
found more than `maxHits`. A scan should run on a thread of its own (an N-API AsyncWorker or a
ThreadSafeFunction) so the event loop keeps running. The host runs one scan at a time a core, each a new line from
its point 0, and sizes `count` so a scan takes about 100 ms, never more than a line's 252,645,136 points.

Version 1 gave a core the midstate, nonce bytes 32..55 (`tail`) and a counter in bytes 56..63. The two versions
cannot be mixed, and neither side tries: the loader refuses a file whose `QUANTUS_API` is not 2 and says which
version it speaks, an older app refuses these cores the same way, and a version 2 core rejects work with no
`line`. The host and all three cores change together.

The host recomputes every hit's full hash on the CPU before it submits anything, and takes a card off
mining after three hits that are not hits. A tie with `targetHigh` is the host's to settle, so `<=` is
right.

## Proving a core right

    node earn/native/quantus/verify-core.js --addon build/quantus_cuda.node --device 0

It runs `verifyCore` from `src/main/quantusCore.js`:

- **vectors**: the five official nonce hashes from quantus-miner's `pow-core` (`NONCE_HASH_KVS`). Each
  must hit with `targetHigh` at its own top 64 bits and miss one below.
- **vectors** take each official nonce as point 0 of the line that starts at it.
- **exact**: 4096 points of one of the miner's own lines, across point 2^26 (where a core starts a new launch), at
  a target one nonce in 16 meets. The hits must be exactly the reference's, and each hit's nonce, built as the miner
  builds it, must be a hit on the full hash.
- **top**: the same at a line's last 4096 points.
- **overflow**: a target every nonce meets, `maxHits` 16: 16 hits back and `overflow` set.
- **big**: 2^30 nonces, as eight lines of 2^27, at a target about 32 nonces meet. Every hit is checked on the reference and
  the count must be 8 to 80. Its time is the core's hashrate.

A core is right when every check passes. Then run it on a pool with a real address (the founder's, when
there is one) and check shares are accepted. Record hashrate with clock, power and temperature, as for
Pearl (`earn/tuning/benchmark.md`).

## Building and packaging

CI builds each core in `.github/workflows/native-core.yml`, with the vendor folder's own build script,
and uploads it for the installers:

| Job | Builds | With | Artifact |
|---|---|---|---|
| `ubuntu-latest-quantus-sycl` | `sycl/build.sh --jit --stage` | DPC++ 7.2.0, Ubuntu 20.04 | `quantus-core-sycl-ubuntu-latest` (a `sycl/` folder) |
| `windows-2022-quantus-sycl` | `build-sycl-win.sh --stage` | DPC++ 7.2.0, MSVC | `quantus-core-sycl-windows-2022` (a `sycl/` folder) |
| `ubuntu-latest-quantus-hip` | `amd/build.sh check`, `gpu` | ROCm 7.2.0 | `quantus-core-hip-ubuntu-latest` |
| `ubuntu-latest-quantus-cuda` | `cuda/build.sh` | CUDA 12.8.1, Rocky Linux 8 | `quantus-core-cuda-ubuntu-latest` |
| `windows-2022-quantus-cuda` | `cuda/build.ps1` | CUDA 12.8.1, MSVC | `quantus-core-cuda-windows-2022` |

A vendor's jobs are skipped until its build script is on the branch (the `quantus-cores` job looks).

The CUDA jobs first run `cuda/sm86/build.py`: the RTX 30 kernel built with `QC_RED 3`, whose reduction tails
`cuda/sm86/carryfix.py` makes right by inverting one carry-in each in the cubin (its header says why that is exact;
`test/quantusCarryfix.test.js` pins it). It is 2% faster on an RTX 3070. The result is embedded in the addon only when
its hash is listed in `cuda/sm86/qc_sm86.ref` for that nvcc, after a run of `verify-core.js` on an RTX 30 card. On a
compute 8.6 card the core checks it against the compiled kernel when the card opens and keeps the compiled kernel on
any failure (`cuda/qc_cubin.inc`). `QUANTUS_CUDA_CUBIN=0` turns it off, `QUANTUS_CUDA_CUBIN_FILE=<cubin>` tries
another file, and `QUANTUS_CUDA_LOG=1` says which kernel each card runs.
Each job is non-blocking and checks what a runner without a card can: the addon loads, speaks API 2,
and answers `devices()`; it loads on Ubuntu 20.04's glibc (Linux) or needs no DLL a rig lacks and
delay-loads `node.exe` (Windows); the Intel core passes `verify-core.js` on the SYCL CPU device.

`miner-build.yml` takes the cores from the same native-core run as the Pearl cores. The CUDA and HIP
cores go beside the Pearl cores; the Intel one goes into the Pearl Intel core's `sycl/` folder
(`earn/scripts/stage-quantus-sycl.mjs`, which refuses a runtime that differs from the one there). From
there they reach `resources/native` in the Windows installer and the AppImage, the CLI's `dist/`
(`scripts/build-cli.mjs`) and the HiveOS package (`scripts/build-hiveos.mjs`), whose smoke test runs
`--coin quantus` on Ubuntu 20.04, 22.04 and 24.04 and needs each core to load. The CLI mines
Quantus with `--coin quantus`, and the desktop app when Quantus is picked in Settings → GPU Coin. The
app tries the core for the rig's own cards first: CUDA on NVIDIA, HIP on AMD, SYCL on Intel Arc.

## Rules

- Each vendor works only in its own folder, plus its addon's build entry. Nothing in `src/` should need
  to change; if it does, say so and change it on the PR, not in a copy.
- The reference is ours (`src/shared/quantus/`) and Quantus's own Rust and CUDA/WGSL sources
  (quantus-miner is Apache-2.0, qp-poseidon MIT). Competing closed miners (SRBMiner and others) are for
  ordinary observation only: run them and read their hashrate, never unpack or disassemble them.
