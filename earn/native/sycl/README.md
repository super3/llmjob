# Pearl core for Intel GPUs (SYCL)

A first version of the Pearl mining core for Intel GPUs, written in SYCL. It
builds into `pearl_core_sycl.node`, which the app and the CLI load through the
existing `PEARL_CORE_PATH` opt-in. The N-API layer is `../src/pearl_core.cc`,
compiled unchanged; this folder replaces only what sits under it
(`pearl_host.cu` and `pearl_kernel.cu`). Nothing here changes the CUDA or AMD
builds or what the app does on those cards, and no installer includes this
folder or its core.

It has not run on an Intel GPU yet. It has run on the SYCL CPU device, where
every hit it found was checked against the JS reference, and it compiles for
every Intel GPU target listed below. Until a kind of card has passed the steps
below, every GPU mines with each hit checked before it is sent ("Untested
cards"). The first hardware is a rig of six Arc A750 cards on HiveOS: see
"First run on an Arc A750 rig". "On an Intel box" and "Test plan" are the same
steps for any other Intel box.

It was written in October 2026, against the app as it was then, and brought up
to the app's core API as of v0.6.16 on 10 November ("Kept up with the CUDA
host").

## Status

| What | State |
|---|---|
| Draw: job_key, constant and hashed fills, salt stamp, commitment trees, cert-v3 seeds, noise, noised operands | written; checked end to end on the CPU device |
| Plain fold (`dot`): int8 dot products in local memory, on any device the build has kernels for (see "Which build") | written; checked on the CPU device |
| XMX fold, sub-group 16 (`xmx16`): Arc B580/B570, Arc Pro B50/B60/B65/B70, Lunar Lake, Panther Lake (H and U), Data Center GPU Max | written; its reference code checked on the CPU device; compiles to 32 DPAS, 128 GRF, no spill |
| XMX fold, sub-group 8 (`xmx8`): Arc A770/A750/A580/A380/A310, Arc Pro A-series, Flex | written; same checks; compiles to DPAS, 128 GRF, no spill |
| Two-batch pipeline, all hits a batch, share proofs carried with each hit | written; checked on the CPU device |
| Device choice: Level Zero GPUs ranked by compute units x clock, `PEARL_GPU_INDEX` pins | written; untested on a GPU |
| Untested cards: a GPU opens only for an opener that checks every hit, and says so in `core.untested` | written; the rule is run in JS by `earn/test/syclCore.test.js`; untested on a GPU |
| Per-card k, launch shapes and their startup trial (the CUDA host's) | not here: the profile's k, no shapes ("Kept up with the CUDA host") |
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

## Results on 10 November 2026

The port on the app as of v0.6.16: that day's `pearl_core.cc` and
`pearl_config.h`, unchanged, against the updated host. Again no Intel GPU: an
Ubuntu 24.04 container on a rented PC (AMD Ryzen 5 2600, 12 threads), the SYCL
CPU device, DPC++ 2026.1.1, ocloc 26.40.40196.15, IGC 2.42.14.

| Check | Result |
|---|---|
| `./install-toolchain.sh --ocloc` | installs, in about 4 minutes; 2.5 GB. Before its fix it stopped with a 404 ("Compiler") |
| `./build.sh`: ten targets, the addon and the check tool | builds, 85 s. The addon linking is what shows the host has every function `pearl_core.cc` calls |
| `PEARL_SYCL_DEVICE=cpu ./verify.sh` | PASS, 105 s |
| its fold check, m = n = 1024: dot, xmx16, xmx8 | 4,096 of 4,096 transcripts match each; 20 of 20 hits |
| its fold check at the mainnet m and n, the last column batch: dot, xmx16 | 32,768 of 32,768 match each; 46 of 46 hits, as in October |
| its verify-hits.js runs, now with the app's own check (`checkHit.js`), m = n = 4096: dot, xmx16, xmx8 | 400 of 400 hits verified each, from 6 salts each; hit ratios 1.04, 1.07 and 1.03 |
| `PEARL_SYCL_DEVICE=cpu ./verify.sh --quick` | PASS, 62 s: 100 of 100 hits each, from 3 salts each; hit ratios 1.02, 1.02 and 1.06. It could not pass before (the end of "Kept up with the CUDA host") |
| the addon opened as the app opens it (`cardK`, `checkHits`, a shape trial asked for, a slice of the salts) | k 2048, no note, no shapes, no trial; the first hit is from the slice's first salt; `core.device.fold` names the fold |
| `pearl_sycl_check --list`, `PEARL_GPU_INDEX`, `ONEAPI_DEVICE_SELECTOR` | the list has the CPU device as index 0; index 5 is refused with the list's size; a selector that leaves no device leaves none |
| `bench.js`, m = n = 4096 | runs: about 0.013 TH/s on that CPU, which says nothing about a GPU |
| `./compile-gate.sh` | passes (table below) |
| `./build.sh --targets intel_gpu_acm_g10,spir64_x86_64`, the A750's build, and the fold check with it | builds in 45 s, its largest process at 360 MB; 4,096 of 4,096 each |
| `./build.sh --jit` and the fold check with it | builds in 12 s; 4,096 of 4,096 each |
| Intel's driver packages installed beside it (`libze-intel-gpu1`, `libze1`, `intel-opencl-icd`, all 26.40.40196.15) | they install in the container, and the fold check still passes |
| `pool-run.sh` with no address | gets as far as it can: Node check, the CLI's two packages, the CLI starting and refusing the address |

Not checked, because each needs a GPU: Level Zero, the hardware DPAS, the
refusal of a GPU for an opener that checks nothing (its rule is run in JS by
`earn/test/syclCore.test.js`), `core.untested` being true, and speed. Not
checked either: a run on a pool.

The toolchain's `sycl-ls` printed nothing there, though the CPU device was
found by everything else. So the steps below list devices with `clinfo -l` and
`pearl_sycl_check --list`, not with `sycl-ls`.

Compile gate with ocloc 26.40.40196.15 and IGC 2.42.14. Hardware XMX fold of
each target. The DPAS counts are the same as in October, and again nothing
spills.

| Targets | Platform | Fold | SIMD | GRF | DPAS | Spill | Instructions |
|---|---|---|---|---|---|---|---|
| intel_gpu_bmg_g21, bmg_g31, lnl_m | Xe2 | xmx16 | 16 | 128 | 32 | 0 | 1,584 |
| intel_gpu_ptl_h, ptl_u | Xe3 | xmx16 | 16 | 128 | 32 | 0 | 1,549 |
| intel_gpu_pvc | Xe-HPC | xmx16 | 16 | 128 | 32 | 0 | 2,425 |
| intel_gpu_acm_g10, acm_g11, acm_g12 | Xe-HPG | xmx8 | 8 | 128 | 17 | 0 | 1,563 |

The plain fold has no spill on any target either.

## On an Intel box

Linux with an Intel GPU, Node 22 or later (what the CLI is built and tested
on), and this repository. None of the steps below has been run on a GPU yet.
The driver steps follow Intel's install guides at dgpu-docs.intel.com (read on
2026-10-09). The compiler and build steps were run in a container with no GPU,
using Ubuntu 24.04 packages.

Intel builds its driver packages and ocloc for Ubuntu 24.04 and 26.04, and they
need glibc 2.38 or newer. On an older system, such as HiveOS's Ubuntu 20.04
image, run everything in an Ubuntu 24.04 container that is given `/dev/dri`:
"First run on an Arc A750 rig" has the commands.

### 1. Driver

First check that the kernel is new enough for the card. Without that, the card
is not driven, and `clinfo` lists nothing.

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

`install-toolchain.sh --ocloc` installs DPC++ 2026.1.1 from conda-forge, and
the ocloc and IGC that Intel's PPA has that day for the box's Ubuntu release
(24.04 or 26.04), without root and without touching the driver:

```
cd earn/native/sycl
./install-toolchain.sh --ocloc    # into ./toolchain, 2.5 GB on disk
. ./toolchain/envrc.sh
```

It prints the ocloc packages it took. On 10 November 2026 those were ocloc
26.40.40196.15 and IGC 2.42.14, the same versions the driver packages of step
1 had that day. October's builds were made with ocloc 26.31 and IGC 2.40.13,
which the script then asked for by file name. The PPA keeps only its newest
builds, so a month later those files were gone and the script stopped with a
404. It now reads the PPA's package index instead.

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
The same goes for a box with several Intel cards: `./build/pearl_sycl_check
--list` numbers them, and `PEARL_GPU_INDEX=N ./verify.sh` checks card N.

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
2. **See the device.** `./build/pearl_sycl_check --list` and
   `PEARL_SYCL_VERBOSE=1 ./build/pearl_sycl_check --m 1024 --n 1024 --batches 1 --folds auto`.
   The first lists the cards the core may open, each with its index. The
   second prints the card, the fold it picked (`xmx16 ... (hardware)` on
   Battlemage and Max, `xmx8 ... (hardware)` on Alchemist), the batch width and
   the band. If the list is empty, go back to "Driver".
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
(`dpkg -l | grep -E 'libze|intel-opencl|ocloc|libigc|i915'`), the `--list`
output, and `verify.log`, `bench.log`, `pool-run.log` and `pool-run-stats.json`. With those,
the next round can target the right bottleneck.

For comparison: ARC-miner's release notes give 36.6 TH/s on a B580. That figure
is self-reported, and its repository has returned 404 since October 2026, so it
could not be checked. The B580's int8 XMX ceiling is about 116 T-MAC/s
(233 TOPS / 2).

## First run on an Arc A750 rig (HiveOS)

For the first hardware: six Arc A750 cards (Alchemist: target
`intel_gpu_acm_g10`, fold `xmx8`) on HiveOS with kernel 6.12. Nobody has run
these steps on a GPU. The same packages, compiler, build and checks ran on 10
November in an Ubuntu 24.04 container with no GPU, on the CPU device ("Results
on 10 November 2026"). The Docker commands, and everything that touches a
card, have not been run by anyone.

HiveOS's image is Ubuntu 20.04 as far as we know (step 0 reads it), and Intel's
driver packages, ocloc and this build all need Ubuntu 24.04's glibc. So
everything runs in an Ubuntu 24.04 container that is given the cards' device
nodes. It uses the host's kernel driver and nothing else of the host's, and
removing the container undoes it.

**0. On the rig, as root: stop the miner and write down what the rig is.**

```sh
miner stop          # HiveOS: the cards must be idle. `miner start` when done
( uname -r; . /etc/os-release; echo "$PRETTY_NAME"; ldd --version | head -1
  lspci -nnk | grep -A3 -Ei 'vga|display'
  ls -l /dev/dri; df -h /; free -g | head -2; nproc; grep -m1 'model name' /proc/cpuinfo
  grep -H . /sys/class/drm/card*/engine/*/preempt_timeout_ms
) 2>&1 | tee /root/arc-facts.txt
```

It needs six `[8086:56a1]` cards with `Kernel driver in use: i915`, a
`renderD` entry for each under `/dev/dri`, 8 GB free on `/` and 4 GB of RAM.
If HiveOS's hashrate watchdog is on for this worker, turn it off for the test,
so it does not restart the miner. If something is missing, send
`arc-facts.txt` and stop here.

**1. The container.**

```sh
apt-get update && apt-get install -y docker.io
mkdir -p /root/arc/out
docker run -d --name arc --network host --device /dev/dri -v /root/arc:/work ubuntu:24.04 sleep infinity
docker exec -it arc bash
```

Every later step runs in that shell. After a dropped SSH session, run the
last line again, then the three lines marked "each new shell".

**2. In the container: Intel's driver, Node, this repository, the compiler.**

```sh
export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y --no-install-recommends software-properties-common gpg-agent \
  ca-certificates curl bzip2 xz-utils git g++ make pciutils
add-apt-repository -y ppa:kobuk-team/intel-graphics
apt-get install -y --no-install-recommends libze-intel-gpu1 libze1 intel-opencl-icd clinfo
clinfo -l                                   # must list six Intel(R) Arc(TM) A750 Graphics
mkdir -p /opt/node && curl -fsSL https://nodejs.org/dist/v22.22.0/node-v22.22.0-linux-x64.tar.gz \
  | tar xz -C /opt/node --strip-components=1
cd /work && git clone --depth 1 https://github.com/super3/llmjob.git
export PATH=/opt/node/bin:$PATH             # each new shell
cd /work/llmjob/earn/native/sycl            # each new shell
./install-toolchain.sh --ocloc 2>&1 | tail -8
. ./toolchain/envrc.sh                      # each new shell
```

**3. Build for the A750 and check every card.**

```sh
./build.sh --targets intel_gpu_acm_g10,spir64_x86_64 2>&1 | tail -4
./build/pearl_sycl_check --list | tee /work/out/devices.txt
for i in 0 1 2 3 4 5; do PEARL_GPU_INDEX=$i ./verify.sh --quick 2>&1 | tee /work/out/verify-$i.log; done
grep -c '== PASS' /work/out/verify-?.log    # 1 for each card
```

`--list` prints the cards the core may open, each with the index the loop
uses: six lines like `0: Intel(R) Arc(TM) A750 Graphics [level_zero], auto fold
xmx8 (hardware DPAS)`. If there is another Intel GPU in the list (a
motherboard's own graphics), leave its index out of the loops.

Each log starts by naming the card and its fold, which must read
`xmx8 (DPAS 8x8x32, sub-group 8) (hardware)`. Then comes one JSON line a
check, each with `"PASS":true`: the plain fold, the hardware DPAS and the
reference code of it, each against the host, at a small size and at the
mainnet size; then 100 hits recomputed in JS, with each of two folds.

How long a card takes is not known, because the folds' speed on an A750 is
not. After a line starting `== fold check`, five minutes with no output is a
hang: stop it with Ctrl-C and see the table below. After a line starting
`== 100 hits` there may be no output for up to 30 minutes, and the run ends by
itself. If card 0 takes that long with the plain fold, say so and run the
other cards later: its JSON lines are what we need first.

**4. Speed, on one card.**

```sh
PEARL_GPU_INDEX=0 ./bench.sh 2>&1 | tee /work/out/bench-0.log
```

**5. Five minutes on the pool, with the cards that passed step 3.** List only
those.

```sh
PEARL_GPU_INDEX=0,1,2,3,4,5 ./pool-run.sh <prl1p... address> 300
cp pool-run.log pool-run-stats.json /work/out/
```

The log should name each card (`mining on GPU 0 · Intel(R) Arc(TM) A750
Graphics`), say once a card that its hits are checked before they are sent and
that the first was right, and count accepted shares. A wrong hit is never
sent: the CLI turns that card off and logs why.

**6. Send back, from the rig:**

```sh
dmesg | grep -iE 'i915|drm| xe ' | tail -80 > /root/arc/out/dmesg.txt
tar czf /root/arc-first-run.tgz -C /root arc-facts.txt arc/out
```

Then `docker rm -f arc` and `miner start`. `/root/arc` (3 GB) can go too.

What is most likely to go wrong, most likely first:

| What you see | What it probably is | What to do |
|---|---|---|
| Step 0 shows another Ubuntu, driver `xe`, or little disk | not the rig these steps assume | send `arc-facts.txt` before going on |
| `docker` will not install or start | HiveOS's kernel or image lacks something Docker needs | send the error; the steps need only an Ubuntu 24.04 userland with `/dev/dri`, however it is made |
| `clinfo -l` lists no A750 | the container cannot reach the cards, or Intel's runtime does not accept this kernel's driver | `ls -l /dev/dri` in the container; send it with the facts |
| `--list` says `[opencl]`, not `[level_zero]` | the Level Zero driver did not load | go on: the core takes the cards through OpenCL. Say so in the report |
| `--list` says `auto fold dot` for an A750 | the runtime could not tell the card's architecture, so the core would not use DPAS | send `devices.txt`. Step 3 then checks the plain fold only |
| "this build has no kernels for ...", or a "SYCL error" at the first check | the index is another GPU, or the driver will not take the prebuilt kernels | check the index against `--list`. Else `./build.sh --jit --out build-jit`, then step 3's loop with `PEARL_SYCL_BUILD=$PWD/build-jit` in front of `./verify.sh`: the driver compiles the kernels itself, and the plain fold is checked, which still tells us whether the port runs on the card |
| a fold check prints nothing for five minutes | Level Zero compute has hung on an A750 with some kernels from 6.6 on (drm/i915 issue 10955) | Ctrl-C, then the same command with `ONEAPI_DEVICE_SELECTOR=opencl:gpu` in front |
| "SYCL error", a lost device, or `GPU HANG` in dmesg, during the plain fold or `bench.sh` | a batch ran longer than the driver's preempt timeout (7.5 s on i915 by default; step 0 printed it) | `PEARL_SYCL_COL_BATCH=32` in front of the command |
| `"fold":"xmx8"` fails while `xmx8-emu` and `dot` pass | the hardware DPAS does not lay its registers out as the reference code does | the failure this run exists to find. Send the log. Do not run step 5 |
| `dot` fails | the port itself is wrong on this driver | send the log. Do not run step 5 |
| `hitRatio` well under 0.85 | hits are being lost | send the log. Do not run step 5 |
| step 5: "a hit came out wrong" | the fold is wrong on that card, and the app caught it | send `pool-run.log` |
| step 5: rejected shares | not the fold, since every hit sent was checked | send `pool-run.log` and `pool-run-stats.json` |

## Switches

| Variable | Default | Meaning |
|---|---|---|
| `PEARL_CORE_PATH` | | the app's and the CLI's opt-in: point it at `build/pearl_core_sycl.node` |
| `PEARL_SYCL_DEVICE` | `gpu` | `cpu` takes the SYCL CPU device (the correctness gate without a GPU) |
| `PEARL_SYCL_FOLD` | `auto` | `dot`, `xmx16`, `xmx8`, `xmx16-emu`, `xmx8-emu`. `auto` is the hardware DPAS where the build has it for the device, else `dot` |
| `PEARL_SYCL_COL_BATCH` | 256 | column offsets a batch (a power of two; at most 2^25 regions a batch, which is 4,096 at mainnet) |
| `PEARL_SYCL_BAND` | from the L2 | row windows walked together, so their A' stays in the L2 |
| `PEARL_SYCL_VERBOSE` | | print the device, fold, batch width and band when a core starts |
| `PEARL_SYCL_UNTESTED` | | `1` opens a GPU for an opener that checks no hits and sends no shares (`bench.js`). It turns no check off ("Untested cards") |
| `PEARL_GPU_INDEX` | | the app's and the CLI's: the cards to mine on, by index in the core's device list (`pearl_sycl_check --list` prints it), such as `0,1,2,3,4,5`. As one number it also pins the card for `verify.sh`, `bench.sh` and the tools under them |
| `ONEAPI_DEVICE_SELECTOR` | | the SYCL runtime's own: `opencl:gpu` hides the Level Zero driver, so the core takes the cards through OpenCL |

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
| `pearl_sycl_check.cpp` | fold check, fold bench and device list (`--list`), no Node |
| `verify-hits.js` | `probes/verify-hits.js` for any core, m and n |
| `bench.js` | the full miner loop's rate |
| `build.sh`, `verify.sh`, `bench.sh`, `pool-run.sh` | the four commands |
| `compile-gate.sh`, `isa-report.js` | per-target compile and IGC assembly report |
| `install-toolchain.sh` | DPC++ from conda-forge and, with `--ocloc`, ocloc from Intel's PPA; no root |

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

## Untested cards

Every Intel GPU is an untested card: no fold here has run on one. The folds
were checked on the CPU device, where DPAS is reference code. If a real card
differs, every hash it reports is wrong, and every one still meets the target,
so every share would be sent and rejected, and a pool can ban the address for
that. So the host follows the AMD host's rule for its own untested cards
(`../amd/README.md`):

- A GPU opens only for an opener that checks every hit before sending it
  (createCore's `checkHits`). The app and the CLI pass it, and so do
  `verify-hits.js` and `pearl_sycl_check`, which check everything themselves.
- The core then reports `core.untested`, and the miner recomputes each of that
  card's hits with `../../src/shared/miner/checkHit.js` (4.6 ms a hit). At the
  first wrong one it sends nothing, turns that card off for the run and logs a
  line asking for a report. That line names the AMD README and an "unknown
  architecture", since the app reads a `gfx` name out of the fold's
  description; the card's name in it is right.
- `PEARL_SYCL_UNTESTED=1` does not turn the check off. It only lets an opener
  that checks nothing open a GPU, which is for `bench.js`: it sends no shares.
  `bench.sh` sets it for that one line.
- The CPU device is not gated. Nobody mines on it.

A kind of card leaves the gate by a change to `pearl_host_create_checked` in
`pearl_sycl_host.cpp`, once one has passed "First run on an Arc A750 rig".

## Kept up with the CUDA host

`pearl_core.cc` calls six host functions that did not exist when this port was
written. The host defines them now, each with the answer that changes nothing,
because no Intel GPU has been timed and there is nothing to choose from:

| The CUDA host | Here |
|---|---|
| `pearl_host_card_k`: a larger k on cards whose fold is faster there (the AMD host: RDNA 4) | the profile's k. Both folds are built for k 2048, and create refuses any other |
| Launch shapes, and the startup trial `pearl_core.cc` races them in (RTX 30) | no shapes, so no trial runs and the app caches no pick. The batch width and band come from the profile, the device's cache size, or `PEARL_SYCL_COL_BATCH` and `PEARL_SYCL_BAND` |
| `pearl_host_create_checked`: no NVIDIA card is untested | every GPU is ("Untested cards") |
| `pearl_host_fold_name`, shown as `core.device.fold` | the fold, whether DPAS is the hardware instruction, the batch width, the band and k |

Not ported, and not needed for the API: the CUDA host's restamp records (a
same-job redraw here still rebuilds A's tree and waits, see "How it works"),
its per-card batch and band rules, and its persistent L2 settings.
`earn/test/syclCore.test.js` compares the host's functions with
`pearl_core.cc`'s declarations type for type, so the next API change fails
that test rather than the link.

Three faults in the port's own scripts turned up on the way, and are fixed:

- `install-toolchain.sh --ocloc` asked Intel's PPA for ocloc and IGC by
  version, and those builds were gone a month later ("Compiler").
- `verify.sh --quick` could not pass: its 100 hits came from 2 salts, and
  verify-hits.js wants 3. It now halves the bound, so they come from 3 or 4.
- `verify-hits.js` checked each hit inside its event. On the CPU device, with
  hits arriving as fast as they were checked, the rate events waited behind
  them, and a run ended with no rate and so no hit ratio. A hit is now checked
  in a turn of the event loop of its own.

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
  The addon needs glibc 2.38 as built here, and Intel's driver for Arc needs
  Ubuntu 24.04 or newer, so on HiveOS's own userland neither loads: the
  HiveOS package (`earn/hiveos`) cannot carry this core as it is.
- The app does not look for this core. `PEARL_CORE_PATH` is the only way in;
  without it, on an Intel rig, the app tries its AMD and CUDA cores and
  reports that neither found a card.
- The app's GPU list, VRAM and temperature readings use nvidia-smi, so on an
  Intel rig they are empty. A rig with several Intel cards, or a Max 1550
  (two stacks, two devices), runs one core unless `PEARL_GPU_INDEX` is set.
- Per-card k, launch shapes, and the CUDA host's batch and band rules ("Kept
  up with the CUDA host"). Each waits for a card to time it on.
- Arrow Lake-H (XMX at sub-group 8) is not a build target; it needs `--jit`
  and mines with `dot`.

## Risks

- The hardware DPAS might not match the reference code in some detail of the
  register layout. `verify.sh`'s `auto` against `auto-emu` finds that in the
  first minute.
- Sustained XMX clock and power under Pearl are unknown.
- IGC versions differ in code generation, and `install-toolchain.sh --ocloc`
  takes the version Intel's PPA has that day. The compile gate passed with IGC
  2.40.13 in October and 2.42.14 in November; `./compile-gate.sh` says what a
  later one makes of the folds.
- A driver whose IGC is not the ocloc's may refuse the prebuilt kernels. A
  `--jit` build, which the driver compiles itself, is the way round, with the
  plain fold only.
- Older compilers and ocloc may not know every default target (see "Compiler").
- A batch longer than the driver's job timeout would be killed. On i915 the
  preempt timeout for compute is 7.5 s by default
  (`/sys/class/drm/card*/engine/*/preempt_timeout_ms`). The default batch is
  short (0.2 s at 5 TH/s at mainnet); `PEARL_SYCL_COL_BATCH` changes it.
- Level Zero compute has hung on an A750 with some kernels from 6.6 on
  (drm/i915 issue 10955, not checked here). The core falls back to OpenCL when
  Level Zero lists no GPU, and `ONEAPI_DEVICE_SELECTOR=opencl:gpu` makes it.
- Most Arc owners use Windows, which this build does not cover yet.
