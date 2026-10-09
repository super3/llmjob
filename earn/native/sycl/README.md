# Pearl core for Intel GPUs (SYCL)

A first version of the Pearl mining core for Intel GPUs, written in SYCL. It
builds into `pearl_core_sycl.node`, which the app and the CLI load through the
existing `PEARL_CORE_PATH` opt-in. The N-API layer is `../src/pearl_core.cc`,
compiled unchanged; this folder replaces only what sits under it
(`pearl_host.cu` and `pearl_kernel.cu`). Nothing here changes the CUDA build or
what the app does on NVIDIA.

It has not run on an Intel GPU yet. It has run on the SYCL CPU device, where
every hit it found was checked against the JS reference, and it compiles for
every Intel GPU target listed below. See "Test plan" for the first hardware run.

## Status

| What | State |
|---|---|
| Draw: job_key, constant and hashed fills, salt stamp, commitment trees, cert-v3 seeds, noise, noised operands | written; checked end to end on the CPU device |
| Plain fold (`dot`): int8 dot products in local memory, any SYCL device | written; checked on the CPU device |
| XMX fold, sub-group 16 (`xmx16`): Arc B580/B570, Arc Pro B50/B60/B65/B70, Lunar Lake, Panther Lake, Data Center GPU Max | written; its reference code checked on the CPU device; compiles to 32 DPAS, 128 GRF, no spill |
| XMX fold, sub-group 8 (`xmx8`): Arc A770/A750/A580/A380, Arc Pro A-series, Flex | written; same checks; compiles to DPAS, 128 GRF, no spill |
| Two-batch pipeline, all hits a batch, share proofs carried with each hit | written; checked on the CPU device |
| Device choice: Level Zero GPUs ranked by compute units x clock, `PEARL_GPU_INDEX` pins | written; untested on a GPU |
| Runs on an Intel GPU | **not yet** |
| Speed | **unknown** until it runs on a GPU |
| Windows build | not written (`build.sh` is Linux only) |

## Results from the container (2026-10-09)

No Intel GPU was available, so correctness was shown on the SYCL CPU device (the
same kernel source, run by Intel's OpenCL CPU runtime) and the GPU targets were
compiled but not run.

Correctness (`PEARL_SYCL_DEVICE=cpu ./verify.sh` and the runs below):

| Check | Fold | Result |
|---|---|---|
| fold check, every region against a host fold, m = n = 1024 | dot, xmx16, xmx8 | 4,096 of 4,096 transcripts match each; hit counts match |
| fold check at the mainnet m and n, the last column batch | dot, xmx16 | 32,768 of 32,768 match each |
| verify-hits.js, m = n = 4096, constant fill | dot | 400 of 400 hits verified, 6 salts |
| same | xmx16 | 400 of 400, 6 salts |
| same | xmx8 | 400 of 400, 6 salts |
| same, hashed fill | dot | 400 of 400, 7 salts |
| verify-hits.js at the mainnet m and n (col_batch 4) | dot | 60 of 60, 1 salt (a salt is 4,096 batches here) |
| verify-hits.js, m = 131072, n = 64: a new salt every batch, mainnet-size A | dot | 100 of 100, 44 salts |

On the CPU device the XMX folds run the DPAS specification's reference code,
so these show the fold's tiling, layout and readout are right. Whether Intel's
hardware DPAS matches that reference code is the first thing to check on a
GPU (`verify.sh` does).

Compile gate (`./compile-gate.sh`, DPC++ 2026.1.1, ocloc 26.31, IGC 2.40.13).
Hardware XMX fold of each target:

| Target | Platform | Fold | SIMD | GRF | DPAS | Spill |
|---|---|---|---|---|---|---|
| intel_gpu_bmg_g21 (B580, B570, Pro B50, B60) | Xe2 | xmx16 | 16 | 128 | 32 | 0 |
| intel_gpu_bmg_g31 (Pro B65, B70) | Xe2 | xmx16 | 16 | 128 | 32 | 0 |
| intel_gpu_lnl_m (Lunar Lake) | Xe2 | xmx16 | 16 | 128 | 32 | 0 |
| intel_gpu_ptl_h (Panther Lake) | Xe3 | xmx16 | 16 | 128 | 32 | 0 |
| intel_gpu_pvc (Max 1550, 1100) | Xe-HPC | xmx16 | 16 | 128 | 32 | 0 |
| intel_gpu_acm_g10 (A770, A750, A580, Flex 170) | Xe-HPG | xmx8 | 8 | 128 | 17 (loop kept) | 0 |
| intel_gpu_acm_g11 (A380, Flex 140) | Xe-HPG | xmx8 | 8 | 128 | 17 | 0 |
| intel_gpu_acm_g12 (Pro A60) | Xe-HPG | xmx8 | 8 | 128 | 17 | 0 |
| spir64 (`--jit`) | any | dot only | | | | |

The plain fold builds without a spill on every target too. Not built:
`intel_gpu_arl_h` (Arrow Lake-H); this ocloc cannot target it.

## On an Intel box

Linux with an Intel GPU (Ubuntu 24.04 is what was tried here), Node 18 or later,
and this repository.

### 1. Driver

The GPU needs the compute runtime with Level Zero. On Ubuntu 24.04:

```
sudo add-apt-repository -y ppa:kobuk-team/intel-graphics
sudo apt install -y libze-intel-gpu1 libze1 intel-opencl-icd intel-ocloc clinfo
sudo usermod -aG render $USER     # then log out and back in
```

`clinfo -l` should list the card. (Other distributions: Intel's "Installing
client GPUs" guide.)

### 2. Compiler

Either Intel oneAPI (`source /opt/intel/oneapi/setvars.sh`), or without root:

```
cd earn/native/sycl
./install-toolchain.sh            # into ./toolchain, ~2.2 GB on disk
. ./toolchain/envrc.sh
sycl-ls                           # should show [level_zero:gpu] for the card
```

The build records the compiler's lib directory in the addon's run path, so the
addon finds the SYCL runtime without the environment script. Only building, and
the CPU device (`PEARL_SYCL_DEVICE=cpu`, which needs `OCL_ICD_VENDORS` from
`envrc.sh`), need it sourced.

### 3. The four commands

All run from `earn/native/sycl`.

| Step | Command | Time on a GPU (rough) |
|---|---|---|
| Build | `./build.sh` | about 1 minute |
| Correctness: fold check, then 400 hits with the core's fold and 400 with the plain fold | `./verify.sh` | a few minutes |
| Speed: each fold alone, then the full miner loop | `./bench.sh` (add `--sweep` for batch widths and bands) | about 2 minutes |
| 300 s on the pool with the CLI | `./pool-run.sh <prl1p...address> 300` | 5 minutes |

Run them in that order and stop at the first failure.

## Test plan, and what to send back

1. **Build.** `./build.sh 2>&1 | tail -20`. If it fails, send the output.
2. **See the device.** `sycl-ls` and
   `PEARL_SYCL_VERBOSE=1 ./build/pearl_sycl_check --m 1024 --n 1024 --batches 1 --folds auto`.
   The second prints the card, the fold it picked (`xmx16 ... (hardware)` on
   Battlemage, `xmx8 ... (hardware)` on Alchemist), the batch width and the band.
3. **Correctness.** `./verify.sh 2>&1 | tee verify.log`. What each part means:
   - fold check, `dot`: the plain fold on the GPU against the host. If it
     fails, the problem is in the port, not in XMX.
   - fold check, `auto` against `auto-emu`: the hardware DPAS against the
     specification's reference code, on the GPU. If `auto` fails and
     `auto-emu` passes, the DPAS register layout is wrong.
   - verify-hits, twice: 400 hits through the addon, each recomputed in JS.
4. **Speed.** `./bench.sh --sweep 2>&1 | tee bench.log`, with nothing else on
   the card. Also send `intel_gpu_top` or `xpu-smi stats -d 0` output taken
   while it runs, if either is installed (clock and power).
5. **Pool.** `./pool-run.sh <address> 300`. It writes `pool-run.log` and
   `pool-run-stats.json`.

Send back: the card's name, `uname -r`, the driver package versions
(`dpkg -l | grep -E 'libze|intel-opencl|ocloc|libigc'`), `sycl-ls`, and
`verify.log`, `bench.log`, `pool-run.log` and `pool-run-stats.json`. With those,
the next round can target the right bottleneck.

For comparison: the best public Intel Pearl miner reports 36.6 TH/s on a B580,
and the B580's int8 XMX ceiling is about 116 T-MAC/s (233 TOPS / 2).

## Switches

| Variable | Default | Meaning |
|---|---|---|
| `PEARL_CORE_PATH` | | the app's and the CLI's opt-in: point it at `build/pearl_core_sycl.node` |
| `PEARL_SYCL_DEVICE` | `gpu` | `cpu` takes the SYCL CPU device (the correctness gate without a GPU) |
| `PEARL_SYCL_FOLD` | `auto` | `dot`, `xmx16`, `xmx8`, `xmx16-emu`, `xmx8-emu`. `auto` is the hardware DPAS where the build has it for the device, else `dot` |
| `PEARL_SYCL_COL_BATCH` | 256 | column offsets a batch (a power of two) |
| `PEARL_SYCL_BAND` | from the L2 | row windows walked together, so their A' stays in the L2 |
| `PEARL_SYCL_VERBOSE` | | print the device, fold, batch width and band when a core starts |
| `PEARL_GPU_INDEX` | | pin a card by its index in SYCL's GPU list (Level Zero order) |

`./build.sh --jit` builds one SPIR-V image that the driver compiles at first
run. It runs on any Intel GPU, including ones without XMX (Iris Xe, Arc in
Meteor Lake), but its XMX folds are the reference code, so it mines with `dot`.

## How it works

Files:

| File | What |
|---|---|
| `pearl_sycl_host.cpp` | the `pearl_host_*` API on SYCL: device choice, memory, the draw, the pipeline, share proofs |
| `pearl_sycl_kernels.hpp` | the kernels: draw ports of `pearl_kernel.cu`, and both folds |
| `pearl_sycl_blake3.h` | BLAKE3, one copy for the host and the device |
| `pearl_sycl_check.cpp` | fold check and fold bench, no Node |
| `verify-hits.js` | `probes/verify-hits.js` for any core, m and n |
| `bench.js` | the full miner loop's rate |
| `build.sh`, `verify.sh`, `bench.sh`, `pool-run.sh` | the four commands |
| `compile-gate.sh`, `isa-report.js` | per-target compile and IGC assembly report |
| `install-toolchain.sh` | DPC++ from conda-forge, no root |

The draw is the CUDA core's: the same job_key, fills, stamp, trees, seeds and
noise, kernel for kernel. Two things are simpler than `pearl_host.cu`, on
purpose:

- A same-job redraw rebuilds A's whole tree on the device and waits for the
  root, where the CUDA host repairs one path and hashes the new a_seed itself.
  One wait a salt; a salt is 64 batches at mainnet.
- Before a redraw rewrites A, the batches still queued are finished and their
  hits read, so a proof always comes from its own salt's tree.

The folds work in windows of 32 rows by 64 columns, which hold exactly 8 whole
regions. A batch is 256 column offsets by every row offset (2^21 regions at
mainnet), walked in bands of row windows so the band's A' stays in the L2.

- `dot`: a work-group of 256 a window, 32 work-items a region, each with 4 rows
  by 2 columns. Each 128-k chunk is staged in local memory. A work-item XORs
  its 8 sums into one word a chunk; XOR is linear, so a region's 32 words are
  combined once at the end.
- `xmx16` / `xmx8`: DPAS 8x16x32 or 8x8x32, in the layout the
  `cl_intel_subgroup_matrix_multiply_accumulate` extension fixes. A window is
  2 sub-groups of 16 rows (sub-group 16) or 4 of 8 rows (sub-group 8), each
  holding 8 DPAS accumulators. Per chunk a lane XORs its accumulators into one
  word for each of its two regions, two (or one) lane shuffles gather a
  region's columns, and the sub-groups' words are XORed at the end. The DPAS
  call is the hardware instruction on the targets that have it, selected inside
  the kernel with `if_architecture_is`, and the specification's reference code
  everywhere else. That reference code is what the CPU device checks.

## Not done

- Nothing has run on an Intel GPU.
- Speed work. The XMX fold loads its operands with plain per-lane loads. The
  likely next steps, in order: B' stored in DPAS fragment order (a fragment is
  then 512 contiguous bytes instead of 16 half-used cache lines), 2D block
  loads with L1 prefetch on Xe2 and Xe-HPC (`SPV_INTEL_2d_block_io`), larger
  sub-group tiles in 256-GRF mode, and, for Alchemist, staging B through local
  memory.
- The restamp is the slow form described above (fine for a first version).
- Windows build, packaging the SYCL runtime beside the addon, and a CI job.
- The app's GPU list, VRAM and temperature readings use nvidia-smi, so on an
  Intel rig they are empty. A rig with several Intel cards runs one core unless
  `PEARL_GPU_INDEX` is set.
- Arrow Lake-H (XMX at sub-group 8) is not a build target.

## Risks

- The hardware DPAS might not match the reference code in some detail of the
  register layout. `verify.sh`'s `auto` against `auto-emu` finds that in the
  first minute.
- Sustained XMX clock and power under Pearl are unknown.
- IGC versions differ in code generation. The numbers above are IGC 2.40.13.
- A batch longer than the driver's job timeout would be killed. The default
  batch is short (0.2 s at 5 TH/s at mainnet); `PEARL_SYCL_COL_BATCH` changes it.
- Most Arc owners use Windows, which this build does not cover yet.
