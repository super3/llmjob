# Pearl mining core for AMD GPUs (MVP)

A HIP port of the Pearl core in `../src`, for AMD Instinct and Radeon cards. It builds
`pearl_core_hip.node`, an addon with exactly the CUDA core's interface: `pearl_core.cc`
is compiled unchanged against a HIP host. The app loads it only when asked to
(`PEARL_CORE_VARIANT=amd`, see below).

**Status: correct on the CPU, compiled for 14 AMD targets, not yet run on an AMD GPU.**
No AMD card was available to write it. Correctness was shown by running the same kernel
source on the CPU through HIP-CPU, with the matrix instructions emulated from AMD's
documented lane layouts, and checking hits against the JS reference the way the pool
does. The first run on hardware has to confirm the layouts (step 2 under "Testing on
real hardware"). Nothing has been tuned for speed.

Nothing here changes the NVIDIA build. The CUDA sources and `native-core.yml` are
untouched; this folder reads `../src` and writes only to `./build`.

## What is here

| File | What it is |
|---|---|
| `build-amd.sh` | builds the GPU addon, the CPU (HIP-CPU) addon, or runs the compile check |
| `src/pearl_amd_kernels.h` | every kernel the AMD core launches: the portable part of `pearl_kernel.cu`, then the folds |
| `src/pearl_amd_fold.h` | the AMD folds, their instruction layouts and CPU emulations, and the transcript hash kernel |
| `src/pearl_host_hip.cpp` | the host: `pearl_host.cu`'s C API on HIP |
| `compat/gpu`, `compat/cpu` | stand-ins for `<cuda.h>`, `<cuda_runtime.h>` and `<mma.h>` |
| `test/pearl_amd_foldtest.cpp` | runs each fold on random operands and checks every transcript word against the CPU; the layout check on hardware |
| `test/verify-hits-profile.js` | `../probes/verify-hits.js` at a profile you choose (that one is fixed to mainnet) |
| `test/run-cpu-checks.sh` | every check that runs without a GPU |

## How it works

**The portable kernels are the CUDA core's own.** BLAKE3, the commitment trees, seeds,
noise, materialise and restamp are the first 1,188 lines of `pearl_kernel.cu`.
`build-amd.sh` copies the file up to the line `// A 16-byte global->shared copy that does
not pass through registers` (where the NVIDIA folds start) into
`build/gen/pearl_kernel_portable.cu` and compiles that. If the line moves, the build
stops and says so, so the two cannot drift silently.

**The folds are new.** Each computes C = A' x B'ᵀ in int32 over k and, at each of the 16
chunk ends, XORs every region's 256 running sums into that region's transcript word.
Each fold writes all 16 words of every region of the batch, and `pearl_amd_hash` hashes
them in a kernel of its own (as the CUDA core does on GA100 and Hopper).

| Fold | Targets | Instruction | Tile |
|---|---|---|---|
| `mfma16` | gfx942, gfx950 (MI300X/A, MI325X, MI350X, MI355X) | `v_mfma_i32_32x32x16_i8`, wave64 | 128x256, 4 waves of 64x128 |
| `mfma8` | gfx908, gfx90a (MI100, MI210, MI250, MI250X) | `v_mfma_i32_32x32x8_i8`, wave64 | 128x256, 4 waves of 64x128 |
| `wmma11` | gfx11 (RX 7000, Radeon PRO W7000, Ryzen AI 300 and Max) | `v_wmma_i32_16x16x16_iu8`, wave32 | 128x256, 8 waves of 64x64 |
| `wmma12` | gfx12 (RX 9060, RX 9070, Radeon AI PRO R9700) | `v_wmma_i32_16x16x16_iu8`, wave32, RDNA4 layout | 128x256, 8 waves of 64x64 |
| `ref` | any (the fold RDNA2, gfx103x, runs) | `v_dot4_i32_i8` where the card has it, else scalar | 32x64 |

The region pattern lines up with every one of these layouts. In a 32x64 block of C each
lane's accumulators belong to one region (two on RDNA, split by register), the eight
lanes that share a region differ in lane bits 0, 3 and 4, and three XOR shuffles finish
it. Lane r of the eight keeps words r and r + 8, so a region's transcript stays in
registers until the tile ends. The 16x16x16 WMMA on RDNA3 needs A and B duplicated in
both half-waves; the fold loads them that way.

**The host** (`src/pearl_host_hip.cpp`) is `pearl_host.cu` without the NVIDIA folds. It
picks the fold from the card's `gcnArchName` and checks it against the folds the loaded
code object actually has (`pearl_amd_caps`), so a build without the card's target fails
with a message rather than searching nothing. `PEARL_AMD_FOLD=ref|mfma16|mfma8|wmma11|wmma12`
overrides the choice. A batch is at most 256 column offsets (`PEARL_AMD_COL_BATCH`), which
keeps each pipeline slot's transcript buffer to 128 MiB at the mainnet profile. All work
runs in order on one stream; proof reads use a second.

## Checked without a GPU

HIP-CPU (github.com/ROCm/HIP-CPU) runs HIP kernels as C++ on the CPU. The same
`pearl_host_hip.cpp` and kernels build against it into `pearl_core_hipcpu.node`; in that
build each matrix instruction is replaced by an emulation written from AMD's Matrix
Instruction Calculator layout. Everything else in the fold is the source the GPU runs.

`test/run-cpu-checks.sh`, on 2026-10-09 (4 cores, about 8 minutes):

| Check | Result |
|---|---|
| fold check, 256x512 and 512x1024, all five folds | every region's 16 words match the CPU computation |
| core, `ref` / `mfma16` / `mfma8` / `wmma11` / `wmma12`, constant fill, m=256 n=512 | 400 of 400 hits verified each, 95-176 salts |
| core, `mfma16` and `wmma12`, hashed fill | 400 of 400 each |
| core, `mfma16`, m=256 n=8192 (two batches a salt) | 400 of 400, 13 salts |

"Verified" is `verify-hits.js`'s check: Merkle proofs, seed chain, noise and fold
recomputed in JS, and the jackpot hash must match the core's exactly. The fold check was
also run with deliberate mistakes: a layout's lane-to-row map, its k offsets, a shuffle,
the column offset, the transcript word index, RDNA4's register split and RDNA3's
half-wave loads. Each one failed. (Swapping rows inside one region passes, and should:
it cannot change a transcript.)

What this cannot show: that the hardware matches the documented layouts. Step 2 below
checks that first.

One HIP-CPU problem had to be worked around: it can run work queued on the null stream
while earlier null-stream work is still running, which let a redraw rewrite A' under a
fold still reading it (about 1 hit in 6 failed in one run). The host now queues
everything on a stream of its own, which behaves the same on a GPU.

## Compile check

`./build-amd.sh check` with ROCm 7.2.4's hip-clang (LLVM 22). Every fold has no spills and
no scratch memory on every target:

| Target | Fold | VGPRs | AGPRs | LDS | Waves/SIMD |
|---|---|---|---|---|---|
| gfx908 | mfma8 | 170 | 128 | 36 KB | 1 |
| gfx90a | mfma8 | 172 | 128 | 36 KB | 1 |
| gfx942 | mfma16 | 172 | 128 | 36 KB | 1 |
| gfx950 | mfma16 | 172 | 128 | 36 KB | 1 |
| gfx1030, gfx1031, gfx1032 | ref | 42 | - | 13.5 KB | 16 |
| gfx1100, gfx1101 | wmma11 | 228 | - | 36 KB | 6 |
| gfx1102, gfx1150 | wmma11 | 228 | - | 36 KB | 4 |
| gfx1151 | wmma11 | 228 | - | 36 KB | 6 |
| gfx1200, gfx1201 | wmma12 | 198 | - | 36 KB | 6 |

The hash kernel takes 49-52 VGPRs and the reference fold 40-46 everywhere. `./build-amd.sh
gpu` builds all 14 targets into one addon (about 2 minutes) and the fold check binary.

## Building

Linux only for now. You need ROCm 7.x with hip-clang (`rocm-llvm`, `hip-dev`,
`rocm-device-libs`; for running, `rocm-hip-runtime`), Node 22 and node-addon-api:

```sh
cd earn/native/amd
npm install --no-save node-addon-api
./build-amd.sh gpu                  # all 14 targets
./build-amd.sh gpu gfx1100          # or just your card's, which is faster
./build-amd.sh check                # registers, scratch and LDS per fold and target
```

The CPU build needs HIP-CPU, TBB (`libtbb-dev`) and clang:

```sh
git clone https://github.com/ROCm/HIP-CPU /tmp/hip-cpu
HIP_CPU_PATH=/tmp/hip-cpu ./build-amd.sh cpu
./test/run-cpu-checks.sh
```

The addon links `libamdhip64.so.7`, so the machine it runs on needs the ROCm 7 runtime.

## Testing on real hardware

On a Linux box with an AMD GPU, ROCm 7.x installed, and the user in the `render` and
`video` groups:

```sh
# 1. Build for the card.
rocminfo | grep -m1 -o 'gfx[0-9a-f]*'                  # the card's target, e.g. gfx1100
cd earn/native/amd
npm install --no-save node-addon-api
./build-amd.sh gpu gfx1100

# 2. Layout check: every fold the card has, against the CPU. Must end in PASS.
./build/pearl_amd_foldtest
./build/pearl_amd_foldtest 2048 2048 3                # bigger; the CPU side takes ~10 s
./build/pearl_amd_foldtest 16384 16384 1 20 0         # timed only: each fold's rate in TMAC/s

# 3. The whole core at the mainnet profile, checked as the pool checks it.
#    Must print PASS with 400 verified. The reference fold is slow, so give it an easier target.
node ../probes/verify-hits.js build/pearl_core_hip.node 120
PEARL_AMD_FOLD=ref node ../probes/verify-hits.js build/pearl_core_hip.node 300 236

# 4. Hashrate, the number the app shows.
node ../probes/hashrate.js build/pearl_core_hip.node 60 10

# 5. Mine on the pool through the CLI.
cd ../..
PEARL_CORE_VARIANT=amd node src/cli/earn-cli.js --address <prl1p...> --mode mining
```

If step 2 fails for a fold, its layout differs from the documented one: send the output
line, which names the first wrong region and word. If step 3 fails, send its JSON line.
For a speed report, send the output of steps 2 (timed), 4, and `rocm-smi` while step 4
runs.

## Using it in the app

`PEARL_CORE_VARIANT=amd` makes the app and `earn-cli` load `pearl_core_hip.node`: beside
the executable, in the packaged `native/` folder, or in this folder's `build/`. It is
never chosen on its own, and when it is asked for and missing, nothing falls back to a
CUDA build. `PEARL_CORE_PATH=/path/to/pearl_core_hip.node` works too.

The app finds cards with nvidia-smi, which lists no AMD card. So one core starts and picks
the AMD card with the most compute units times clock. On a rig with several, choose them
by index with `--gpu-index 0,1` (or `PEARL_GPU_INDEX`); the app passes indices
nvidia-smi does not list straight to the core, one core each. Indices are HIP's, which
follow `rocm-smi` unless `HIP_VISIBLE_DEVICES` says otherwise. Temperatures, VRAM rows on
the board and memory clocks are read with nvidia-smi and stay empty on AMD.

## Gaps and next steps

- Run steps 1-4 above on an MI300X, an RX 7900 and an RX 9070.
- Speed: nothing is tuned. Obvious first steps: DPP or `ds_swizzle` instead of
  `__shfl_xor` in the readout; `v_mfma_i32_32x32x32_i8` on gfx950 (2x the rate; its
  layout is not in AMD's calculator yet, so gfx950 runs the gfx942 instruction); A' and B'
  materialised in fragment order so the folds skip LDS; `global_load_lds` on CDNA; the
  hash fused into the fold; persistent tiles; the batch and band sized from the L2 and
  Infinity Cache as `learnings.md` describes for NVIDIA.
- AMD cards in the app: rocm-smi or amd-smi for card lists, temperatures and VRAM.
- Windows (the HIP SDK) and a release workflow that builds `pearl_core_hip.node`.
