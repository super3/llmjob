# Pearl core for Intel GPUs (SYCL)

A first version of the Pearl mining core for Intel GPUs, written in SYCL. It
builds into `pearl_core_sycl.node`, which the app and the CLI load through the
existing `PEARL_CORE_PATH` opt-in. The N-API layer is `../src/pearl_core.cc`,
compiled unchanged; this folder replaces only what sits under it
(`pearl_host.cu` and `pearl_kernel.cu`). Nothing here changes the CUDA build or
what the app does on NVIDIA.

It has not run on an Intel GPU yet. It has run on the SYCL CPU device, where
every hit it found was checked against the JS reference, and it compiles for
every Intel GPU target listed below. See "On an Intel box" and "Test plan" for
the first hardware run.

## Status

| What | State |
|---|---|
| Draw: job_key, constant and hashed fills, salt stamp, commitment trees, cert-v3 seeds, noise, noised operands | written; checked end to end on the CPU device |
| Plain fold (`dot`): int8 dot products in local memory, on any device the build has kernels for (see "Which build") | written; checked on the CPU device |
| XMX fold, sub-group 16 (`xmx16`): Arc B580/B570, Arc Pro B50/B60/B65/B70, Lunar Lake, Panther Lake (H and U), Data Center GPU Max | written; its reference code checked on the CPU device; compiles to 32 DPAS, 128 GRF, no spill |
| XMX fold, sub-group 8 (`xmx8`): Arc A770/A750/A580/A380/A310, Arc Pro A-series, Flex | written; same checks; compiles to DPAS, 128 GRF, no spill |
| Two-batch pipeline, all hits a batch, share proofs carried with each hit | written; checked on the CPU device |
| Device choice: Level Zero GPUs ranked by compute units x clock, `PEARL_GPU_INDEX` pins | written; untested on a GPU |
| Runs on an Intel GPU | **not yet** |
| Speed | **unknown** until it runs on a GPU |
| Windows build | not written (`build.sh` is Linux only) |

## Results from the container (2026-10-09)

No Intel GPU was available, so correctness was shown on the SYCL CPU device (the
same kernel source, run by Intel's OpenCL CPU runtime) and the GPU targets were
compiled but not run.

Correctness (`PEARL_SYCL_DEVICE=cpu ./verify.sh` and the runs below), re-run
after the review fixes. A hit ratio is hits found over hits expected from the
work the core reported (see verify-hits.js); the hits themselves are the same
on every run, since the job and salts are fixed.

| Check | Fold | Result |
|---|---|---|
| fold check, every region against a host fold, m = n = 1024 | dot, xmx16, xmx8 | 4,096 of 4,096 transcripts match each; 20 of 20 hits |
| fold check at the mainnet m and n, the last column batch (now part of `verify.sh`) | dot, xmx16 | 32,768 of 32,768 match each; 46 of 46 hits |
| fold check at the mainnet m and n, the first 640 batches of 4 column offsets (the regions the mainnet verify-hits run below searches) | dot | 20,971,520 of 20,971,520 match; 70 of 70 hits. 80 is the average for that many regions, so the 0.82 ratio below is this job's luck, not lost hits |
| verify-hits.js, m = n = 4096, constant fill | dot | 400 of 400 hits verified, 6 salts, hit ratio 1.02 |
| same | xmx16 | 400 of 400, 6 salts, 1.01 |
| same | xmx8 | 400 of 400, 6 salts, 1.01 |
| same, hashed fill | dot | 400 of 400, 7 salts, 0.95 |
| verify-hits.js at the mainnet m and n (col_batch 4) | dot | 60 of 60, 1 salt (a salt is 4,096 batches here), 0.82 |
| verify-hits.js, m = 131072, n = 64: a new salt every batch, mainnet-size A | dot | 100 of 100, 44 salts, 0.90 |
| the same fold check with a `--jit` build | dot, xmx16, xmx8 | 4,096 of 4,096 each |
| a build with no CPU kernels, run on the CPU device | | stops at start: "this build has no kernels for ..." |

1,760 hits were recomputed from scratch, and none failed.

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
| intel_gpu_ptl_u (Panther Lake) | Xe3 | xmx16 | 16 | 128 | 32 | 0 |
| intel_gpu_pvc (Max 1550, 1100) | Xe-HPC | xmx16 | 16 | 128 | 32 | 0 |
| intel_gpu_acm_g10 (A770, A750, A580, Flex 170) | Xe-HPG | xmx8 | 8 | 128 | 17 (loop kept) | 0 |
| intel_gpu_acm_g11 (A380, A310, Flex 140) | Xe-HPG | xmx8 | 8 | 128 | 17 | 0 |
| intel_gpu_acm_g12 (Pro A60) | Xe-HPG | xmx8 | 8 | 128 | 17 | 0 |
| spir64 (`--jit`) | any | dot only | | | | |

The plain fold builds without a spill on every target too. The default build
(`./build.sh`) has kernels for ten targets, the nine GPU targets above and the
CPU device, and takes about a minute here. Not built: `intel_gpu_arl_h` (Arrow
Lake-H); this ocloc cannot target it.

## On an Intel box

Linux with an Intel GPU, Node 18 or later, and this repository. None of the
steps below has been run on a GPU yet. The driver steps follow Intel's install
guides at dgpu-docs.intel.com (read on 2026-10-09). The compiler and build steps
were run in a container with no GPU, using Ubuntu 24.04 packages.

### 1. Driver

First check that the kernel is new enough for the card. Without that, the card
is not driven, and `clinfo` and `sycl-ls` list nothing.

```
lspci -nn | grep -Ei 'VGA|DISPLAY'    # the card and its PCI id
uname -r                              # the running kernel
ls /dev/dri                           # needs a renderD* entry
```

| Card | Kernel driver | Kernel it needs (Intel's table) |
|---|---|---|
| Arc A-series, Arc Pro A60 | i915 | 6.2 (Ubuntu 24.04's stock 6.8 is fine) |
| Lunar Lake | xe | 6.11 |
| Arc B580, B570 | xe | 6.12 |
| Arc Pro B50 | xe | 6.14 |
| Arc Pro B60 | xe | 6.15 |
| Arc Pro B65, B70; Panther Lake | xe | 6.17 |
| Data Center GPU Max 1550, 1100; Flex 170, 140 | Intel's out-of-tree i915 (`intel-i915-dkms`) | what the LTS stack below supports: Ubuntu Server 22.04 or 24.04 |

Ubuntu 24.04 ships with kernel 6.8, which drives none of the xe cards above. On
24.04, install the hardware enablement kernel and check `uname -r` again, or use
Ubuntu 26.04, whose kernel drives them as installed:

```
sudo apt install -y --install-recommends linux-generic-hwe-24.04 && sudo reboot
```

Then the compute runtime with Level Zero. **Arc, Arc Pro, Lunar Lake and
Panther Lake** (Ubuntu 24.04 or 26.04, Intel's PPA):

```
sudo add-apt-repository -y ppa:kobuk-team/intel-graphics
sudo apt install -y libze-intel-gpu1 libze1 intel-opencl-icd intel-ocloc clinfo
sudo gpasswd -a $USER render     # then log out and back in
```

**Data Center GPU Max and Flex** need Intel's data-center stack instead, not the
PPA. On a cloud VM it is usually installed already: if `clinfo -l` lists the
card, skip this. Otherwise, on Ubuntu Server 22.04 or 24.04 (Intel's LTS 2523
release):

```
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key \
  | sudo gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg
. /etc/os-release
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu ${VERSION_CODENAME}/lts/2523 unified" \
  | sudo tee /etc/apt/sources.list.d/intel-gpu-${VERSION_CODENAME}.list
sudo apt update
sudo apt install -y linux-headers-$(uname -r) linux-modules-extra-$(uname -r) flex bison \
  intel-fw-gpu intel-i915-dkms xpu-smi
sudo reboot
sudo apt install -y intel-opencl-icd libze-intel-gpu1 libze1 clinfo
sudo gpasswd -a $USER render     # then log out and back in
```

Check with `clinfo -l`: it should list the card. If it does not, the usual
causes are a kernel older than the table says, a user not yet in the render
group (log out and back in), or, on Max and Flex, the i915 module from
`intel-i915-dkms` not loaded (`lsmod | grep i915`, `sudo dmesg | grep -i i915`).

### 2. Compiler

The default build was compiled with DPC++ 2026.1.1 and ocloc 26.31 (IGC
2.40.13). `install-toolchain.sh --ocloc` installs exactly those, without root
and without touching the driver, so it is the safe choice on any box:

```
cd earn/native/sycl
./install-toolchain.sh --ocloc    # into ./toolchain, ~2.4 GB on disk
. ./toolchain/envrc.sh
sycl-ls                           # should show [level_zero:gpu] for the card
```

oneAPI works too (`source /opt/intel/oneapi/setvars.sh`), with ocloc from the
driver packages. But an older compiler or ocloc may not know `bmg_g31`, `ptl_h`
or `ptl_u`, and then the whole default build fails. Intel's GPU repository,
where the Max and Flex stack comes from, had no ocloc newer than 25.18 when
checked. If that happens, build for your card only, as in "Which build".

The build records the compiler's lib directory in the addon's run path, so the
addon finds the SYCL runtime without the environment script. Only building, and
the CPU device (`PEARL_SYCL_DEVICE=cpu`, which needs `OCL_ICD_VENDORS` from
`envrc.sh`), need it sourced. `envrc.sh` keeps the system's OpenCL drivers
listed, so `clinfo` in that shell still shows the card.

### Which build

`./build.sh` with no options compiles the kernels ahead of time for the GPUs
below plus the CPU device. A GPU not in the table has no kernels in that build:
the core then stops at start with "this build has no kernels for ...". Build
with `--jit` for those.

| Card | `--targets` entry | Fold it mines with |
|---|---|---|
| Arc B580, B570, Arc Pro B50, B60 | `intel_gpu_bmg_g21` | xmx16 |
| Arc Pro B65, B70 | `intel_gpu_bmg_g31` | xmx16 |
| Lunar Lake | `intel_gpu_lnl_m` | xmx16 |
| Panther Lake | `intel_gpu_ptl_h`, `intel_gpu_ptl_u` | xmx16 |
| Data Center GPU Max 1550, 1100 | `intel_gpu_pvc` | xmx16 |
| Arc A770, A750, A580, Flex 170 | `intel_gpu_acm_g10` | xmx8 |
| Arc A380, A310, Flex 140 | `intel_gpu_acm_g11` | xmx8 |
| Arc Pro A60 | `intel_gpu_acm_g12` | xmx8 |
| Anything else: Meteor Lake, Arrow Lake, Iris Xe, DG1 | `./build.sh --jit` | dot |

To build for one card, which is also faster:
`./build.sh --targets intel_gpu_bmg_g21,spir64_x86_64` (keep `spir64_x86_64`
for the CPU device). If unsure which entry is yours, build them all.

**Data Center GPU Max 1550:** in Level Zero's default mode it shows as two GPUs,
one per stack, and one core mines one stack. `verify.sh` and `bench.sh` measure
one stack; the card is about twice that. To mine both, give the CLI both:
`PEARL_GPU_INDEX=0,1 ./pool-run.sh <address> 300`. The Max 1100 is one stack.
The same goes for a box with several Intel cards.

### 3. The four commands

All run from `earn/native/sycl`.

| Step | Command | Time on a GPU (rough) |
|---|---|---|
| Build | `./build.sh` | about 1 minute |
| Correctness: fold checks, then 400 hits with the core's fold and 400 with the plain fold | `./verify.sh` | a few minutes |
| Speed: each fold alone, then the full miner loop | `./bench.sh`, or `./bench.sh --sweep` for batch widths and bands too | about 2 minutes, 5 with `--sweep` |
| 300 s on the pool with the CLI | `./pool-run.sh <prl1p...address> 300` | 5 minutes |

Run them in that order and stop at the first failure.

## Test plan, and what to send back

1. **Build.** `./build.sh 2>&1 | tail -20`. If it fails, send the output, then
   try your card's target alone (see "Which build").
2. **See the device.** `sycl-ls` and
   `PEARL_SYCL_VERBOSE=1 ./build/pearl_sycl_check --m 1024 --n 1024 --batches 1 --folds auto`.
   The second prints the card, the fold it picked (`xmx16 ... (hardware)` on
   Battlemage and Max, `xmx8 ... (hardware)` on Alchemist), the batch width and
   the band. If `sycl-ls` shows no `level_zero:gpu`, go back to "Driver".
3. **Correctness.** `./verify.sh 2>&1 | tee verify.log`. What each part means:
   - fold check, `dot`: the plain fold on the GPU against the host. If it
     fails, the problem is in the port, not in XMX.
   - fold check, `auto` against `auto-emu`: the hardware DPAS against the
     specification's reference code, on the GPU. If `auto` fails and
     `auto-emu` passes, the DPAS register layout is wrong.
   - the same fold check at the mainnet m and n, on the last column batch:
     every region of it, so a fold that only loses hits at full size fails.
   - verify-hits, twice: 400 hits through the addon, each recomputed in JS.
     Its `hitRatio` (hits found over hits expected from the core's own rate)
     should be near 1, within about 0.85 to 1.15. Well under that means hits
     are being lost.
4. **Speed.** `./bench.sh --sweep 2>&1 | tee bench.log`, with nothing else on
   the card. Also send `intel_gpu_top` or `xpu-smi stats -d 0` output taken
   while it runs, if either is installed (clock and power).
5. **Pool.** `./pool-run.sh <address> 300` (on a Max 1550,
   `PEARL_GPU_INDEX=0,1 ./pool-run.sh <address> 300`). It writes
   `pool-run.log` and `pool-run-stats.json`.

Send back: the card's name, `uname -r`, the driver package versions
(`dpkg -l | grep -E 'libze|intel-opencl|ocloc|libigc|i915'`), `sycl-ls`, and
`verify.log`, `bench.log`, `pool-run.log` and `pool-run-stats.json`. With those,
the next round can target the right bottleneck.

For comparison: ARC-miner's release notes give 36.6 TH/s on a B580. That figure
is self-reported, and its repository has returned 404 since October 2026, so it
could not be checked. The B580's int8 XMX ceiling is about 116 T-MAC/s
(233 TOPS / 2).

## Switches

| Variable | Default | Meaning |
|---|---|---|
| `PEARL_CORE_PATH` | | the app's and the CLI's opt-in: point it at `build/pearl_core_sycl.node` |
| `PEARL_SYCL_DEVICE` | `gpu` | `cpu` takes the SYCL CPU device (the correctness gate without a GPU) |
| `PEARL_SYCL_FOLD` | `auto` | `dot`, `xmx16`, `xmx8`, `xmx16-emu`, `xmx8-emu`. `auto` is the hardware DPAS where the build has it for the device, else `dot` |
| `PEARL_SYCL_COL_BATCH` | 256 | column offsets a batch (a power of two; at most 2^25 regions a batch, which is 4,096 at mainnet) |
| `PEARL_SYCL_BAND` | from the L2 | row windows walked together, so their A' stays in the L2 |
| `PEARL_SYCL_VERBOSE` | | print the device, fold, batch width and band when a core starts |
| `PEARL_GPU_INDEX` | | pin a card by its index in SYCL's GPU list (Level Zero order) |

`./build.sh --jit` builds one SPIR-V image that the driver compiles at first
run. It runs on any Intel GPU, including ones without XMX (Iris Xe, Arc in
Meteor Lake) and ones the default build has no kernels for, but its XMX folds
are the reference code, so it mines with `dot`.

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
  Intel rig they are empty. A rig with several Intel cards, or a Max 1550
  (two stacks, two devices), runs one core unless `PEARL_GPU_INDEX` is set.
- Arrow Lake-H (XMX at sub-group 8) is not a build target; it needs `--jit`
  and mines with `dot`.

## Risks

- The hardware DPAS might not match the reference code in some detail of the
  register layout. `verify.sh`'s `auto` against `auto-emu` finds that in the
  first minute.
- Sustained XMX clock and power under Pearl are unknown.
- IGC versions differ in code generation. The numbers above are IGC 2.40.13.
- Older compilers and ocloc may not know every default target (see "Compiler").
- A batch longer than the driver's job timeout would be killed. The default
  batch is short (0.2 s at 5 TH/s at mainnet); `PEARL_SYCL_COL_BATCH` changes it.
- Most Arc owners use Windows, which this build does not cover yet.
