# LLMJob Earn (`earn`)

A desktop GUI that turns the excess compute on GPUs you already own into crypto.
It mines Pearl (**PRL**) on the HeroMiners pool with its own CUDA mining core.
Paste a payout address, hit **Start**, and earn — no command line. Built with Electron and shipped for **Windows**, **Linux** and
**[macOS](#macos-llm-only)** (LLM only — see below); headless rigs can use the
[command-line miner](#headless-cli-linux) instead of the GUI.

> The LLM "co-mining" side of LLMJob is landing: both clients (GUI and headless
> CLI) can now run a local llama.cpp `llama-server` alongside — or instead of —
> mining, exposing an OpenAI-compatible endpoint at `127.0.0.1:8080/v1`. Pick the
> **Compute Mode** in the GUI's Settings, or `--mode` on the CLI.

**Highlights**

- **Live pool balance** — your pending payout plus lifetime paid for the address.
- **Public board** — while mining, the app publishes live status to the network
  page (your Pearl address only — nothing else is reported).
- **Zero-config** — mines on every GPU, picks the lowest-latency pool region,
  and updates itself.
- **Runs in the background** — closes to the tray, can start with the computer,
  and can mine only while nobody is using it. See
  [Running in the background](#running-in-the-background).

## How it works

- **`src/shared/`** — pure, fully unit-tested logic (no Electron/DOM):
  - `config.js` — pool regions (and the old AlphaPool ids they replace), the llama-server and model downloads, economics.
  - `address.js` — `prl1p…` / `mdl1p…` validation, shortening, and the merge-mining combined address.
  - `cliArgs.js` — parses/validates the headless CLI flags into the same settings shape the GUI uses.
  - `llmMode.js` — the compute-mode policy (mining / llm / auto → which engines run; the old `both` is read as `auto`), shared by the GUI and the CLI.
  - `platform.js` — what each OS can do: whether it can mine (not on macOS) and whether electron-updater can install there.
  - `llama.js` / `vram.js` — build the local `llama-server` command line + parse its output, and size the GPU offload (`--n-gpu-layers`) from free VRAM.
  - `selfUpdate.js` — decides, from the running version + GitHub's latest release, whether the CLI binary should self-update.
  - `miner/` — the Pearl protocol: Stratum messages, PearlHash, and the share proofs the pool checks.
  - `xmr/` — Monero on the CPU ([below](#monero-on-the-cpu-opt-in)): its Stratum, address check (Keccak-256), thread plan, and the worker's line protocol.
  - `coreVariant.js` — which build of the mining core a rig loads (CUDA 12.8, CUDA 13, AMD or Intel).
  - `memClock.js` — which mining cards get their memory clock locked, and at what.
  - `coreOffset.js` / `coreOffsetTrial.js` — which mining cards try a core clock offset, and the trial that decides what each keeps.
  - `miningStats.js` — accumulates the miner's events into the live stats snapshot.
  - `earnings.js` — PRL/USD per-day estimates.
  - `balance.js` — builds the pool balance lookup and parses the pending + paid response.
  - `minerReport.js` — the payload published to the public network board while mining.
  - `statsFile.js` — the stats JSON the CLI writes for HiveOS's `h-stats.sh`.
  - `gpu.js` — reads `nvidia-smi`'s card list, names AMD and Intel Arc cards, and decides which cards mine (`--gpu-index`, `PEARL_GPU_INDEX`).
  - `region.js` — picks the lowest-latency pool region.
  - `updateStatus.js` — formats the in-app auto-update banner.
  - `format.js` — uptime / hashrate / number formatting.
- **`src/main/`** — Electron main process:
  - `pearlMiner.js` — the miner: one core per card, the pool connection, and share submission.
  - `pearlEngine.js` — wraps `pearlMiner.js` in the start/stop and event interface both shells drive.
  - `pearlCore.js` — loads `pearl_core.node` (built from [`native/`](native)), picking the CUDA 12.8 or CUDA 13 build, or the AMD or Intel core on a rig of those cards.
  - `xmrEngine.js` — the Monero CPU miner: the pool connection and the RandomX worker process (built from [`xmr-worker/`](xmr-worker)); `xmrWorker.js` finds the worker, `cpuFacts.js` reads the CPUs, cgroup limit, memory and L3 the thread plan needs.
  - `probe.js` — asks `nvidia-smi` about the cards (names, VRAM, temperatures, PCI bus ids), finds AMD and Intel Arc cards from the system's own list, pings the pool regions, and posts the board report.
  - `gpuClocks.js` — locks and releases a card's memory clock while it mines.
  - `gpuClockOffset.js` — reads and sets a card's core clock offset, through the mining core.
  - `llmManager.js` / `llmEngineManager.js` — spawn/supervise the local `llama-server` and download its binary + GGUF model on demand (injected IO, unit-tested).
  - `lanShare.js` — shares the local LLM on the network when Settings asks (see [below](#sharing-the-local-llm-on-your-network)).
  - `main.js` / `preload.js` — window, settings persistence, IPC bridge (thin shells).
- **`src/renderer/`** — the GUI (Setup → Running → Settings → Logs), pure display + IPC.
- **`src/cli/`** — headless Linux miner (no Electron); thin IO shells that reuse
  the same `shared/*` logic and the same miner as the GUI:
  - `earn-cli.js` — the CLI entry (arg handling, which cards mine, run loop, self-update check).
  - `selfUpdater.js` — the IO side of self-update (GitHub fetch, download, atomic self-replace, re-exec).
  - `sea-entry.js` — entry shim for the packaged single-file binary (`scripts/build-cli.mjs`).

## The mining engine

The miner is our own. Its GPU work is a CUDA core, `pearl_core.node`, an N-API
addon built from [`native/`](native), that runs inside the app's own process.
`src/main/pearlMiner.js` drives it and speaks Stratum to the pool. There is no
separate engine to download or start. The installer ships the cores in
`resources/native/` (electron-builder `extraResources`, from `vendor/native/`),
and the CLI looks for them beside its binary. If no core loads, the app says so
and doesn't mine.

By default it mines on **HeroMiners** (`<region>.pearl.herominers.com:1200`),
in the region with the lowest latency unless one is pinned: `us`, `us2`, `ca`,
`br`, `de`, `fi`, `fr`, `tr`, `sg`, `hk`, `kr` or `au`. The desktop app's Pool
menu (Settings → Connection) can pick **Kryptex** instead, on one of its
servers: `prl-<server>.kryptex.network:7048`, where the server is `us`, `eu`,
`br`, `sg` or `hk`. Kryptex charges 3% and has no MDL merge mining. The CLI
mines there with `--pool`. Either pool sends a block header and a target, and
the core chooses how to search.

Earlier versions wrapped AlphaPool's `alpha-miner`. That engine and
that pool are gone. An old AlphaPool region id (`us1`, `eu1`, `eu2`, `ru1`,
`sg1`, `hk1`, `in1`), in saved settings or passed to the CLI, maps to the
nearest HeroMiners region.

### When the pool's name won't resolve

Both shells look up the pool's name with the system resolver first. If that
fails, they try the computer's DNS servers directly (skipping the system's
resolver and its cache), then public DNS (1.1.1.1, 8.8.8.8), then the last
address that worked, saved in `pool-addresses.json` beside the node identity.
An address is saved once the pool has accepted the rig's login there. A filter
that blocks with a real-looking address (a hosts-file `127.0.0.1`, Pi-hole's
own address, a block page) never gets that far, so it can't replace the saved
one.
An answer of `0.0.0.0` or `::` counts as no answer at every step and is never
saved: it is how DNS filters that block crypto-mining names (Cloudflare's
1.1.1.2, Pi-hole, AdGuard) answer. The log names the fallback used:

```
could not look up us2.pearl.herominers.com (ENOENT); using 203.0.113.7 from the last address that worked
```

Once the pool connection has been down for 30 seconds, the rig reports 0 TH/s
to the network board, so the board agrees with the pool about whether the rig
is mining. A shorter drop doesn't show: pools close connections every few
minutes, the miner is back within seconds, and the cards mine throughout.

### Which GPUs it mines on

**Every card the rig has**, one mining core each, and each names itself in the
log as it starts:

```
mining on GPU 0 · NVIDIA RTX PRO 4500 Blackwell
mining on GPU 1 · NVIDIA GeForce RTX 4070
```

Each core gets its own slice of the search space (a starting salt and a stride),
so two cards never draw the same operands and never find the same share. Each
reports its own hashrate, shares and temperature, so the UI, the stats file and
the network board show one row per card.

A card that can't start is skipped, not fatal — the usual reason is the local LLM
holding most of that card's VRAM. The rest of the rig keeps mining.

A card that stops mid-run (a CUDA error such as `misaligned address`) is opened
again 30 seconds later, on salts it hasn't searched, so it can't resend shares the
pool already has. If it stops again within 10 minutes of each of 3 restarts in a
row, it's left off until mining is restarted. That usually means an unstable
overclock or a card that is failing.

```
GPU 1 (NVIDIA GeForce RTX 4070) stopped: CUDA error during search: an illegal instruction was encountered
restarting GPU 1 (NVIDIA GeForce RTX 4070) in 30 s
mining on GPU 1 · NVIDIA GeForce RTX 4070
```

Both shells set `CUDA_DEVICE_ORDER=PCI_BUS_ID` at startup, so "GPU 1" means the
same card to the miner as it does to `nvidia-smi`. Left to itself the CUDA
runtime numbers cards by its own "fastest first" heuristic, which on a mixed rig
is a different card than the one `nvidia-smi` lists first.

To mine on some cards only, list their `nvidia-smi` indices with `--gpu-index`
on the CLI, or with `PEARL_GPU_INDEX` in either app. The flag wins over the
variable:

```bash
llmjob-earn-cli --address prl1p… --gpu-index 0,2
PEARL_GPU_INDEX=1 llmjob-earn-cli --address prl1p…
```

A listed card that `nvidia-smi` doesn't show is skipped, and the log says so.
If it shows none of them, they go to the core, which fails and names the cards
that exist. Each card names itself in the log as its core starts, and the CLI
also logs the whole list and what chose it.

`none` mines on no card. With `--mode mining` the CLI then waits until it is
stopped instead of exiting. The CLI takes the same values from the flag and the
variable, and stops with an error on one it can't read. The desktop app can't
run with no mining card (set Compute Mode to LLM for that). So it ignores
`PEARL_GPU_INDEX=none`, and any value it can't read, says so in the log, and
mines on every card.

These choose the mining cards only: the local LLM picks its own.

A `CUDA_VISIBLE_DEVICES` left in the environment is removed at start, and the
log says so. It hides cards from CUDA but not from `nvidia-smi`, so the two
disagreed about which card was which, and once kept a rig's second card idle.

The local LLM works the same way — an instance on every card with room for the
model, mining cards included. Its `llama-server` is a Vulkan build, which numbers
cards its own way (on Windows the card driving the desktop comes first), so the
CUDA ordering above does not apply to it. Before each start the app runs
`llama-server --list-devices`, matches its cards to nvidia-smi's, and starts
each instance with `--device VulkanN`. On a rig with more than one card it
also reads each card's PCI bus id, with a short `llama-server` run that loads
only the model's header and is killed after about a second. Cards are matched
by bus id where it has one, else by name. The GPU numbers in the log are
nvidia-smi's, and the start line adds llama-server's name when it differs:
`local LLM starting on 1 GPU [0] (llama-server calls it Vulkan1)`. A card it
can't match is started with `--main-gpu <index>`, as before, and the log warns
that it may load on a different card. Cards with the same name and no bus id
are paired in order, and the log warns when that may pick the wrong one. A
vision model's projector goes on the same card as the model
(`MTMD_BACKEND_DEVICE`, unless you set it yourself).

### How much CPU a card takes (`--gpu-wait`)

Each NVIDIA card has one thread that queues its launches and waits for them. Up
to v0.6.15 that thread busy-waited: a whole CPU core a card, all the time. On a
rig with more cards than CPU cores the threads took turns, a finished launch
waited for its thread's turn, and the cards sat idle. One rig of 8 cards on a
2-core Celeron made 31.8 TH/s on each RTX 3070 where another miner made 74.9.

Now the thread sleeps until the driver wakes it, and raises its own priority so
it runs as soon as it is woken (on Linux that needs root, which HiveOS has;
without it the priority stays as it was). Measured on an RTX 4090, 60 seconds
each:

| Wait | CPU cores used | TH/s |
|---|---|---|
| `spin` (v0.6.15 and before) | 0.995 and 0.999 | 317.44 and 317.41 |
| `block` (default) | 0.003 | 317.37 |
| `poll` | 0.006 | 317.11 |

And with the miner held to one CPU core beside three busy programs, which is
what 8 spinning cards on 2 cores do to each other: `spin` 173.8 TH/s, `block`
317.5, `poll` 317.3.

`--gpu-wait block|poll|spin` on the CLI, or `PEARL_WAIT` in either app, picks the
wait. `poll` sleeps half a millisecond at a time and looks again, and does not
depend on the driver waking the thread. `spin` is the old behaviour. These
numbers are from Windows on one card. Linux and rigs of many cards have not been
measured yet: if a rig mines slower on this release, try `--gpu-wait poll`, then
`--gpu-wait spin`, and tell us. The AMD and Intel cores are unchanged.

### A second kernel for RTX 40 cards, to try (`--rotation-fold`)

Experimental, and off by default. `--rotation-fold` on the CLI, or
`LLMJOB_ROTATION_FOLD=1` in the desktop app's environment, makes each RTX 40
card mine on a second kernel, the rotation fold. The app has no setting for it
in its window: set the variable the way `LLMJOB_MINE_MEM_CLOCK` is set, in the
shell that launches the app on Linux or in Windows' system environment
variables. It is read on every start, and a value other than 1 or 0 is ignored
with a warning. (Both become `PEARL_FOLD_ROT=1`, the variable the mining core
reads.) On the one card it has been measured on, an
RTX 4090 at 450 W, it mined 330.5 TH/s against 317.2 for the normal kernel,
about 4% more, with every share accepted. It uses about 1.3 GB more card memory
(2.7 GB against 1.3 GB).

Its inner loop is written by hand, so the miner does not take it on trust:

- When a card opens on it, the miner recomputes 24 of the fold's hits on the CPU
  before it mines the pool's job with it. That takes about 10 seconds.
- After that it recomputes every share the card finds before sending it (about
  a quarter of a second of CPU a share).
- One wrong result and the card goes back to the normal kernel for the rest of
  the run, and the log asks you to report the line.

If the fold cannot be used, the card mines on the normal kernel and one log line
says why: the card is not an RTX 40, the fold's file (`pearl_rot_sm89.cubin`,
beside the core) is missing, the card has too little free memory, or the driver
would not load it. RTX 20, 30 and 50 cards, AMD and Intel cards are not affected
by the switch.

### Two builds of the core (CUDA 12.8 and CUDA 13)

A release ships two builds of the mining core side by side, in the installer's
`resources/native/` and beside the CLI binary:

| File | Toolkit | Cards |
|---|---|---|
| `pearl_core.node` | CUDA 12.8 | GTX 10, V100, RTX 20, 30, 40 and 50, A100 / A800 / A30 / CMP 170HX, H100 / H200 and B200 (sm_61/70/75/80/86/89/90a/100/120) — every rig |
| `pearl_core_cu13.node` | CUDA 13.3 | sm_89 and sm_120; picked automatically for RTX 50 / Blackwell only (the sm_89 half measured no faster on a 4090) |

**GTX 10 (Pascal) note.** These cards have no tensor cores, so the sm_61 half of
`pearl_core.node` mines on a fold of its own built on the int8 `dp4a`
instruction. Its arithmetic is exact, so its hits are the ones any other card
would find. It has not been run on a GTX 10 card yet; from the instruction
rate it should be four to five times slower than an RTX 2080 Ti. CUDA 13 cannot
build Pascal, so these cards always load `pearl_core.node`.

**RTX 30 (Ampere) note.** The sm_86 half of `pearl_core.node` now runs the same
fold path as an RTX 40 (the persistent, eight-warp, tall fold). Its hits check
out (the sm_86 code, run on a 4090), but it has not been run on an Ampere card:
there is no hashrate figure for it, and the previous Ampere path is no longer
in this build. If a 3090 mines slower on this release,
`PEARL_CORE_PATH=<path to the previous release's pearl_core.node>` loads the old
core unchanged, and a report with both numbers is what settles it.

**B200 note.** A B200 (compute 10.0, in an HGX or DGX system with an x86-64
host) runs the sm_100 half of `pearl_core.node`: the CUDA 12.8 build of the tall
fold source that RTX 50 cards use, built for sm_100. (An RTX 50 card on driver
580 or newer mines on the CUDA 13 build of it.) It does not use the B200's own
tensor instructions (tcgen05) yet, so it is slow: a 1000 W B200 mined 460 TH/s
on it, a third of PeakMiner's and SRBMiner's 1,380 (`earn/tuning/benchmark.md`, "B200").
Before this build the core had no sm_100 code, so a B200 could not mine at all. A GB200 can't run the miner: its host CPU is Arm
(Grace), and the core is built for x86-64 Linux and Windows only.

The CUDA 13 compiler produces faster code for Blackwell: on an RTX 5090 at
600 W the same source ran 107.27 TH/s built with CUDA 13.3 against 103.97 with
12.8 (+3.2%). But a CUDA 13 build needs NVIDIA driver 580 or newer, so it is used
only when **the driver is 580+ and every card that will mine is compute 12.x**
(read from `nvidia-smi --query-gpu=index,compute_cap,driver_version`). Everything
else — a 2080 Ti, 3090, 4090 or B200, a mixed 4090 + 5090 rig, an older driver, or a rig where
`nvidia-smi` can't say — loads `pearl_core.node`, as before. If the CUDA 13 core
is missing, won't load, or fails its first start with a driver/runtime error, the
app falls back to `pearl_core.node` and logs why. The choice is logged once per
start:

```
Pearl core: CUDA 13 build · driver 610, mining card is compute 12.0
Pearl core: CUDA 12.8 build · GPU 1 is compute 8.9 (the CUDA 13 build has code for it but measured no faster on a 4090)
Pearl core: CUDA 12.8 build · GPU 0 is compute 8.6 (the CUDA 13 build has no code for it)
```

To override it, set `PEARL_CORE_VARIANT=cu12` or `cu13`, which forces that
build (a forced `cu13` still falls back if it can't start). `PEARL_CORE_PATH=/path/to/core.node`
beats both and loads exactly that file. On the CLI, `--core cu12` or `--core cu13`
does the same as the variable and wins over it.

### The AMD core (Radeon and Instinct)

Releases also carry `pearl_core_hip.node`, a core for AMD cards (`native/amd`):

- **Linux**, from v0.6.8: built with ROCm 7.2, in the AppImage, beside the CLI
  binary and in the HiveOS package. It needs ROCm 7's HIP runtime
  (`libamdhip64.so.7`), which AMD's normal `amdgpu` driver doesn't include. The
  HiveOS package carries it, in `rocm/` beside the core: AMD's ROCm 7.2.0
  packages for RHEL 8, so it loads on HiveOS's Ubuntu 20.04 image too
  (`scripts/fetch-hip-runtime.sh`). Elsewhere the rig installs it
  (`rocm-hip-runtime` 7.x, in `/opt/rocm/lib` or on `LD_LIBRARY_PATH`). The core
  looks in `/opt/rocm/lib` first, then in `rocm/`. Without either the log says
  the core would not load and names the missing library.
- **Windows**: built with AMD's HIP SDK 7.2 for the Radeon cards, in the
  installer. Its runtime, `amdhip64_7.dll`, comes with a current Adrenalin
  driver, so nothing else needs installing. A driver too old to have it can't
  load the core, and the log says to update the driver. No Windows Radeon has
  run it yet, and the SDK's compiler (clang 21) is not the Linux one's (clang
  22), so its speed there is unmeasured.

When `nvidia-smi` lists no NVIDIA card, the app and the CLI try the AMD core
first. If it isn't installed, won't load, or can't start on its card, they load
`pearl_core.node` instead and log why:

```
Pearl core: AMD build (ROCm) · nvidia-smi lists no NVIDIA card
Pearl core: CUDA 12.8 build · the AMD build could not start (this core has no code for gfx1036 …)
```

On a rig with NVIDIA cards, ask for it with `--core amd` on the CLI (or Extra
config on HiveOS), or `PEARL_CORE_VARIANT=amd` for either app. Asked for, it
has no fallback:

```bash
./llmjob-earn-cli --address prl1p… --core amd
```

It has code for RDNA 2, 3 and 4 Radeon cards and, on Linux, CDNA Instinct cards
(gfx908, gfx90a, gfx942, gfx950, gfx1030-1032, gfx1100-1102, gfx1150, gfx1151,
gfx1200, gfx1201). On Salad it beat SRBMiner on every Radeon class rented there,
the RX 7800 XT to the RX 9070 XT (`earn/tuning/benchmark.md`, "AMD Radeon"). Its folds for
Instinct (gfx9) and RX 6000 (gfx103x) cards have not run on such a card (the
RX 6000 one, which uses `v_dot4` since RDNA 2 has no matrix instructions, has
run only on an RX 7800 XT and in emulation). A fold that is wrong on the card
makes every share wrong, and a pool can ban the address for that, so on those
cards the miner checks every hit as the pool would before sending it (about 5
ms a share), and stops the card at the first wrong one without sending it.
`PEARL_AMD_UNTESTED=1`, which v0.6.8 and v0.6.9 asked RX 6000 owners to set
to mine, does not turn the check off. Card temperatures stay empty on AMD. On Linux the
board names the card and shows its VRAM from the `amdgpu` driver's sysfs files, since ROCm
calls an RX 9070 XT just "AMD Radeon Graphics". Every AMD card the system lists mines, one core
each; `--gpu-index` picks cards instead (`native/amd/README.md`, "Using it in the app").

#### Setting up a Linux AMD rig

What a Debian 13 VM with an RX 9070 XT needed (6 November):

- **ROCm's HIP runtime.** ROCm 7.2 is published for Ubuntu 22.04 and 24.04 only; on Debian
  13 the Ubuntu 24.04 (`noble`) repository works (`repo.radeon.com/rocm/apt/7.2.4 noble`,
  pinned at 600). Only `hip-runtime-amd` is needed; it pulls in the rest. Then
  `echo /opt/rocm/lib > /etc/ld.so.conf.d/rocm.conf && ldconfig`, and add the miner's user
  to the `render` and `video` groups.
- **Firmware.** `firmware-amd-graphics` is in Debian's non-free-firmware and isn't
  installed by default. Without it `/lib/firmware/amdgpu` is empty and the driver stops
  before the card works.
- **A recent kernel for RDNA 4.** Debian 13's stock 6.12 kernel crashed in `amdgpu` while
  starting an RX 9070 XT. The trixie-backports kernel (7.2) with the backports firmware
  worked. As a rule, RDNA 4 wants kernel 6.14 or later and a recent linux-firmware.
- **For the local LLM**, the CLI's Vulkan llama-server needs Mesa's Vulkan driver
  (`mesa-vulkan-drivers`). Its "radv is not a conformant Vulkan implementation" warning
  is harmless.
- **Huge pages** speed up the Monero dataset build: `sysctl -w vm.nr_hugepages=1280` took
  it to 3.1 s.

#### In a virtual machine (Proxmox passthrough)

An RX 9070 XT sits behind its own PCIe switch, and a bus reset kills it: the card reads
all ones until the host reboots. QEMU resets the bus at VM start when it owns every
function on it. What worked: pass through only the GPU function (`1002:7550`), leave its
HDMI audio (`1002:ab40`) on the host so no bus reset can happen, clear the GPU's
`reset_method`, and set `vfio-pci disable_idle_d3=1`. If it is the host's only GPU, also
boot the host with `initcall_blacklist=sysfb_init` so its framebuffer doesn't hold the
card. A VM's L3 cache size is the hypervisor's, not the host's, so the Monero miner
doesn't cap its threads by it there; `--cpu-threads` sets the count.

On Linux, RDNA 4 cards (RX 9000, gfx1200 and gfx1201) mine at k 4096, the matrix depth a
share declares, where every other card mines k 2048. On Windows every card mines k 2048 for
now: the Windows core leaves the k 4096 readout out, since its compiler builds it worse and
no Windows Radeon has tested it. Their fold measured about 1% faster per MAC
there (`earn/tuning/benchmark.md`), and a MAC earns the same at either k. The log names it:
`mining on GPU 0 · AMD Radeon RX 9070 XT · k 4096`. Each card's shares carry its own k, so
a rig with RDNA 4 and older cards mines both at once. A card stays at k 2048 when it has too
little free VRAM for k 4096 (about 1,969 MiB against 1,137) and the log says why, and
`PEARL_AMD_K=2048` keeps every card there. If a pool turns down 3 shares at k 4096 and takes
none, the miner mines k 2048 for the rest of the run.

### The Intel core (Arc)

A third core, for Intel Arc cards, is built from `native/sycl` and packed with
the others. No release has carried it yet. It is a folder, `sycl/`, not one
file: `pearl_core_sycl.node` with the open-source SYCL runtime it needs beside
it (Apache-2.0 with LLVM exceptions; its licence text is in the folder). The
folder is 15 MB on Linux and 7.6 MB on Windows.

- **Linux**: in the AppImage, beside the CLI binary, in the HiveOS package, and
  on the release page as `pearl_core_sycl-linux.tar.gz`. Six Arc A750 cards
  have mined with it, on Ubuntu 24.04 and on HiveOS's own Ubuntu 22.04
  userland. It needs glibc 2.27 (Ubuntu 18.04 or newer) and Intel's compute
  runtime, which the kernel's graphics driver does not include: OpenCL
  (`intel-opencl-icd`) or Level Zero (`libze-intel-gpu1` and `libze1`), with
  the miner's user in the `render` group. A newer runtime is faster: an A750
  does 16.8 TH/s with Intel's 2026 release and 10.2 with the 2024 one HiveOS
  has.
- **Windows**: in the installer, and **experimental**. CI builds it and checks
  its folds on a CPU. No Intel card has run it. It needs Intel's Arc driver and
  Microsoft's Visual C++ runtime, which the installer does not carry.

It is tried only on a rig where `nvidia-smi` lists no NVIDIA card and the
system lists Arc cards and no AMD card the AMD core mines on. Every other rig
loads what it loaded before and never opens the folder, so a missing or broken
one says nothing there. `--core intel` (or `PEARL_CORE_VARIANT=intel`) asks for
it by name, with no fallback. On a rig of Arc cards the log reads:

```
gpu:        6× Intel Arc A750  (auto)
mining on 6 Intel cards (GPU 0, 1, 2, 3, 4, 5), one core each; use --gpu-index to choose cards
Pearl core: Intel build (SYCL) · nvidia-smi lists no NVIDIA card, and the system has 6 Intel Arc cards
mining on GPU 0 · Intel(R) Arc(TM) A750 Graphics
```

- **Which cards.** Discrete Arc cards. On Linux the app knows the A770, A750,
  A580, A380, A310, B580 and B570 by PCI id; on Windows it reads the adapter's
  name. A processor's built-in graphics is not counted, and neither is an Arc
  card it has no name for: `--core intel` mines on those. Every card the core
  finds mines, one core each, and `--gpu-index` picks among them by the core's
  own numbers.
- **Every hit is checked.** No Intel card has mined in a release, so each one
  mines the way an untested AMD card does: the miner checks every hit as the
  pool would before sending it, and stops the card at the first wrong one. The
  log says so once a card: `Intel xmx8 fold: this kind of card has not mined in
  a release yet`.
- **No driver.** When the system lists Arc cards and the core finds none, the
  log says once which packages to install (on Windows, to update the Arc
  driver). The CUDA build is tried after it, as on an AMD rig.
- **VRAM on the network board.** Nothing outside the miner reports an Arc
  card's memory, so its row shows what the Intel core reports: the card's
  size, and what the miner itself has allocated on it. Memory another program
  holds on the card (a local LLM, a game) is not counted, where an NVIDIA or
  AMD row counts it. A core that reports no memory leaves the row's VRAM
  empty, as before.
- **What is not shown.** Temperature for an Arc card: the app has no source
  for it yet, so that field stays empty. The memory clock lock and the core
  clock offset are for NVIDIA cards and are left off. The local LLM has not
  been tried on an Arc rig.
- **Updates.** The CLI's self-updater does not fetch the folder yet. The
  HiveOS package and the installers replace it with each release; a rig that
  runs the bare binary unpacks the archive again after an update.

`native/sycl/README.md` has the measurements, the licences and what is not
done.

## Monero on the CPU (opt-in)

The app can also mine Monero (**XMR**) on the CPU while the GPU mines Pearl.
It is **off by default** and does nothing until you turn it on.

**Turning it on**

- Desktop app: paste a Monero address in **Settings → CPU Mining**, then
  START. Leaving the box blank keeps it off, as merge mining's address did. It runs in any mode, with or without the GPU miner, and STOP
  stops it. A Monero address is enough to START in Mining mode, with no Pearl
  address. With "Only when idle" on, it waits for an idle
  computer like the GPU miner does. The Mine tab then shows Monero's
  hashrate between Pearl's and the LLM's, tagged CPU, and its shares beside
  Pearl's. Its tooltip has the XMR balance. The settings apply at the next
  START.
- CLI: `--xmr-address 4…` (or `8…`). `--cpu-threads <n>` sets the thread count;
  `0` turns it off. `--cpu-threads` or `--cpu-priority` without an address
  only logs a warning, so Pearl still mines.
- HiveOS: put `--xmr-address 4…` in **Extra config arguments**. The dashboard
  gets the CPU's hashrate as the rig's second algorithm, `randomx`.

**The address** is a standard Monero address (95 characters, starts with `4`)
or a subaddress (starts with `8`) from your own wallet, such as the Monero GUI,
Feather or Cake Wallet. The app checks its checksum, so a typo is caught before
anything is mined. Integrated addresses, and exchange deposits that need a
payment ID, are not taken.

**The pool** is HeroMiners' Monero pool, port 1111, in the same region as the
Pearl pool: a rig on `us2` mines Monero on `us2.monero.herominers.com`. Today
every regional name points at one server in Frankfurt. It charges 0.9%, and pays every
10 minutes once the balance reaches 0.01 XMR. In the desktop app, picking
Kryptex in the Pool menu moves Monero there too, to the same server
(`xmr-<server>.kryptex.network:7029`), at 1% and a 0.01 XMR minimum payout. The
CLI mines Monero on HeroMiners unless `--xmr-pool` names another endpoint.
There is no dev fee.

**How many threads.** It counts the CPUs this process may use, its affinity
and any container CPU limit (cgroup), not just the machine's cores, and always
leaves one free for the GPU miner (two on 6 CPUs or fewer). It also stops at
2 MB of L3 cache per thread: past that, threads add power, not hashrate. With 4
CPUs or fewer, or under 8 GB of memory, the default is off; pick a thread count
to mine anyway. The log and Settings say what it chose and why.

**Priority.** The free CPU is what keeps the GPU miner whole, so on top of it
the CPU miner runs at low priority on Linux (nice 19) and at idle priority on
Windows, and steps aside for the programs you use. On a server whose CPUs are
busy with other work, low priority gets less out of it; the CLI's
`--cpu-priority normal` takes its full share (measured below).

**What it costs.** About 2.4 GB of memory (RandomX's 2080 MiB dataset and its
cache), and on a typical desktop CPU 45-65 W more power, more on a big one. At
$0.15/kWh a Ryzen 5 5600X earns about what that power costs, and most Intel
desktop CPUs earn less. It pays when power is cheap or the computer is on
anyway for the GPU.

**Huge pages** make RandomX up to 20-30% faster. The worker uses them when it
can and falls back when it can't, and the log says which it got
(`dataset ready in 5.2 s · transparent huge pages`). On Linux, root can reserve
them with `sudo sysctl -w vm.nr_hugepages=1280`; without that it asks for
transparent huge pages. On Windows they need the "Lock pages in memory" right
(Local Security Policy, then sign out and back in).

**How it works.** The hashing is `llmjob-xmr-worker`, our own small program
around [tevador/RandomX](https://github.com/tevador/RandomX) (BSD-3-Clause),
the library Monero itself uses ([`xmr-worker/`](xmr-worker)). The app runs it
as a separate process, so if it crashes, or an antivirus removes it, CPU mining
stops and Pearl mining carries on. A worker that stops is started again after
30 seconds, up to three times; if it stops within 10 minutes of each of those
restarts, it is left off. It checks itself
against RandomX's test vectors when it starts and checks every share again
before it is sent. It ships beside the Pearl cores: in the
installer's `resources/native/` and beside the CLI binary. There is none for
macOS, and Settings says so on a build without it.

**Antivirus.** Any program that mines Monero may be flagged as a coin miner,
on Windows especially. If only the worker is flagged, CPU mining stops and the
app goes on. If the Windows installer, which contains the worker, is flagged,
the whole download could be blocked. That has not been tested.

**Monero's next fork** (v17, not scheduled yet) changes the proof of work to
RandomX v2. The worker already has it; the app will need a release that turns it
on when the fork is scheduled.

**Measured** on 2026-10-11 on rented hosts, with the CLI built from source the
way CI builds it, mining Pearl on the GPU and Monero for one address on
HeroMiners. Each host was a container with a CPU limit, and neither had huge
pages, so both miners ran without them (ours used transparent huge pages).

| Host | CPU threads (of CPUs it may use) | Pearl alone → with Monero | Monero beside Pearl | Monero alone: ours vs SRBMiner 3.7.5, same threads |
|---|---|---|---|---|
| Core i5-11400, RTX 3060 | 6 (of 11; L3 caps it) | 44.23 → 44.03 TH/s | 2.48 kH/s | 2.46 vs 1.88 kH/s |
| Ryzen 9 5950X, RTX 2060 | 14 (of 15) | 42.21 → 41.90 TH/s | 1.74 kH/s at idle priority | 6.44 vs 4.53 kH/s |

The pool accepted every share our miner sent (34 across these runs, 0 rejected,
0 stale), and its stats for the address count the same. The Pearl miner used one CPU
throughout. The 5950X host's other tenants kept all 32 of its CPUs busy, which
is what held the CPU miner back beside Pearl. Run there again with each
priority, Pearl was the same every time (40.91-40.92 TH/s, against 40.92-41.14
alone) and the CPU miner made 1.89 kH/s at idle, 3.97 at low and 6.47 at normal.
That is why Linux now defaults to low.

## Sharing the local LLM on your network

The local LLM listens on `127.0.0.1:8080`, so only this computer can use it.
**Share the local LLM on my network**, on the API tab, lets phones and
other computers on the same network use it too.

- **How:** the app runs the gate the CLI serves on (`autoGate.createServeGate`)
  on port **8000**, on every network interface. It passes requests to
  `llama-server`, which stays on `127.0.0.1`, so the in-app chat and cluster jobs
  don't change. The address to use, `http://<this computer>:8000/v1`, shows in
  the switch's row and on the API tab.
- **Off by default, and no key.** While it's on, anyone on the network can use
  the model.
- **When it runs:** only while the local LLM runs (Auto or LLM mode). The switch
  applies at once, with no restart. It's saved in `settings.json` as `shareLlm`.
- **Firewall:** the first time, Windows may ask whether to let LLMJob Earn through
  the firewall. Allow it on private networks.
- **Port 8000 taken** (for example by the CLI on the same computer): the switch
  says so, and the app carries on without it.

## Running in the background

Three switches in **Settings → Background**, saved in `preferences.json` next to
`settings.json` in the app's data folder:

- **Keep in the tray** (on by default). Closing
  the window hides it and mining carries on. The first time, a notification says
  so. The tray icon opens the window again, and its menu has **Quit**. Opening
  the app a second time shows the running window instead of starting a second
  copy. On a desktop with no tray, closing quits as before. That includes GNOME
  without an AppIndicator extension, which draws no tray icon: the app asks the
  session bus whether anything will. On Windows, a close from another program
  (the installer, the uninstaller, Task Manager's End task) quits the app, so
  the miner stops cleanly before the installer replaces the files.
- **Start at login** (off by default). Windows gets a
  login item, a Linux AppImage an entry in `~/.config/autostart`, and macOS a
  LaunchAgent in `~/Library/LaunchAgents`. Each starts the app with `--hidden`,
  so it stays in the tray and starts mining with the saved payout address. Only
  the installed app can do this; a dev run turns the switch back off. Each
  start points the entry at the app again if an update moved it, and otherwise
  leaves it alone, so turning it off in Task Manager or the desktop's startup
  settings sticks.
- **Only when idle** (off by default). Idle means 5 minutes
  without keyboard or mouse input, or a locked screen. While someone uses the
  computer, mining stops and the window says what it is waiting for. The local
  LLM keeps running, so chat, cluster jobs and the network share go on, and it
  stays sized to leave mining its share of the card. **STOP** still shows,
  because the START stands. Mining resumes on its own once the computer goes
  idle again, without waiting for a model that is still downloading. With no
  mining to pause (LLM mode, no payout address, or macOS), the switch does
  nothing. A desktop that cannot report idle time (some
  Linux setups) counts as idle.

Updates download in the background. With the window open, the update bar's
**Update & restart** installs one. With the window in the tray, the app installs
it as soon as it has downloaded, restarts in the tray, and picks its START back
up. It waits for a START that is still downloading the model, and leaves an
install for all users (in Program Files) to the button, since Windows would ask
for an administrator.

## macOS (LLM only)

The Mac build runs **the local LLM and nothing else**. The mining core is
CUDA, and a Mac has no NVIDIA GPU to run it on, so the app turns the miner off
up front (`src/shared/platform.js`). Concretely:

- **Settings → Compute Mode** offers only **Auto** and **LLM**; the two mining
  modes are removed rather than left to arm a **START** that runs nothing.
- **Auto** is the default and does the right thing — the model comes up, the
  mining half is skipped, and the Logs tab says so.
- Everything downstream of the model is unchanged: the Chat tab, the
  OpenAI-compatible endpoint at `127.0.0.1:8080/v1`, and
  [serving cluster jobs](#serve-cluster-jobs-proxy-llm-through-llmjob) all work
  exactly as they do on a Windows or Linux box.

The Mac build is **Apple silicon only**. Intel Macs are not shipped: the app's
one job there is running the model, and an Intel Mac has no Metal GPU worth
running it on. `llama-server` comes from llama.cpp's macOS arm64 release, so
Metal acceleration is built in and there is no separate GPU runtime to install.
(`shared/config.js` still maps `darwin-x64` to the Intel llama-server build, for
running from source on an Intel Mac with `npm start` — there is just no
installer for it.)

The GPU shows up by name — "Apple M3 Max" — read from `system_profiler`, since
macOS has neither `nvidia-smi` nor WMI to ask.

**First launch.** The DMG is **ad-hoc signed, not notarized** — there is no Apple
Developer ID behind this project, and CI signs the bundle itself
(`scripts/mac-adhoc-sign.mjs`) only because Apple silicon refuses to execute a
binary with no signature at all. macOS will therefore block the first open:

> "LLMJob Earn" can't be opened because Apple cannot check it for malicious software.

Allow it once in **System Settings → Privacy & Security → Open Anyway**, or from
a terminal:

```bash
xattr -dr com.apple.quarantine "/Applications/LLMJob Earn.app"
```

For the same reason the app **does not auto-update on macOS**: Squirrel.Mac only
installs an update whose signature matches the running app's, which an ad-hoc
signature cannot satisfy. "Check for updates" opens the
[Releases page](https://github.com/super3/llmjob/releases/latest) instead, and
you install the new DMG over the old app.

## HiveOS (flight sheet)

Each release ships a HiveOS custom-miner package, `llmjob-earn-<version>.tar.gz`:
the headless CLI binary, its mining cores, and the hook scripts in `hiveos/`,
packed by `scripts/build-hiveos.mjs` (`npm run dist:hiveos`). CI packs it from
the same binary and cores it publishes on the release. Releases v0.5.0 to
v0.5.12 have no package; use a later one.

CI then installs each build's package the way HiveOS does and starts it, in
Ubuntu 22.04, 24.04 and 20.04 containers (`.github/scripts/hiveos-smoke.sh`).
They have no GPU, so this checks everything short of mining: the files, that
each core loads, the hook scripts, and the stats HiveOS reads.

**It needs a HiveOS image on Ubuntu 20.04 or newer.** The CLI is built on Node
22, which needs glibc 2.28. The CUDA cores need less: they are built on Rocky
Linux 8 (`.github/workflows/native-core.yml`) and need glibc 2.17 and
`GLIBCXX_3.4.22`. The AMD core also needs ROCm 7, which supports Ubuntu 22.04
and newer. The Intel core needs glibc 2.27 and Intel's OpenCL or Level Zero
driver. RTX 50-series and RTX PRO Blackwell cards need
a newer NVIDIA driver than the stable image's 550: run `nvidia-driver-update`
from Hive Shell.

Flight sheet: choose **Custom** as the miner, then Setup Miner Config. Four
fields, and the rest stays blank:

- **Miner name** → `llmjob-earn`.
- **Installation URL** →
  `https://github.com/super3/llmjob/releases/latest/download/llmjob-earn-hiveos.tar.gz`.
  The same URL for every version: the miner updates itself (below).
- **Wallet and worker template** → your `prl1p…` address. The worker is then
  the rig's HiveOS name. `prl1p….rig1` names it yourself: everything after the
  first `.` is the worker name. `%WAL%` and `%WAL%.%WORKER_NAME%` work too, with
  the address as the flight sheet's wallet. An old `prl1p…+mdl1p…` wallet still
  mines, but merge mining is retired.
- **Pool URL** → `us.pearl.herominers.com:1200`. HiveOS won't save the flight
  sheet without one (`The url field is required`). With a HeroMiners address
  there, whichever region it names, the miner mines on HeroMiners and picks
  the region with the lowest latency when it starts, so a rig in Germany is
  not held on the US server. To mine on another pool, put its `host:port`
  there, such as `prl-us.kryptex.network:7048` for Kryptex. The pool must
  speak the same protocol as HeroMiners, and Kryptex does. A `stratum+tcp://`
  prefix is fine. The miner log's `pool:` line says which pool it is on. The
  rig's page in HiveOS gets a message when Pool URL moved the rig to another
  pool, and when it can't be used (no port, for one); the rig then stays on
  HeroMiners. If the field holds several addresses, the first is read.

If HiveOS asks for a **Hash algorithm**, use `pearlhash`. **Pass** stays blank.
**Extra config arguments** are optional CLI flags; leave it empty for the
defaults. The ones that make sense on a rig:

- `--pool <host:port>` picks the pool too, and beats Pool URL.
- `--region <id>` pins the pool region: `us`, `us2`, `ca`, `br`, `de`, `fi`,
  `fr`, `tr`, `sg`, `hk`, `kr` or `au`. An old AlphaPool id from an older
  flight sheet (`us1`, `eu1`, `eu2`, `ru1`, `sg1`, `hk1`, `in1`) maps to the
  nearest of these, and the miner log says which (`region eu1 is now de`).
- `--no-update` keeps the rig on the version it has.
- `--mine-mem-clock <MHz>` locks the memory clock while mining; `0` turns off
  the RTX 5090's default, which tries a lock and keeps it only if the card
  mines faster (see below).
- `--mine-core-offset auto` tries a core clock offset on each RTX card and
  keeps it only if the card still computes right and mines faster. Off unless
  you ask; read [what it risks](#try-a-core-clock-offset-while-mining---mine-core-offset-auto) first.
- `--mode auto` also serves a local LLM (below).
- `--no-report` keeps the rig off the public network board.
- `--xmr-address 4…` also mines Monero on the CPU (see
  [Monero on the CPU](#monero-on-the-cpu-opt-in)).
- On an AMD rig the package's `pearl_core_hip.node` mines once nvidia-smi
  lists no NVIDIA card, and `--core amd` asks for it on a rig that has NVIDIA
  cards too (see [The AMD core](#the-amd-core-radeon-and-instinct)). The
  package carries ROCm 7's HIP runtime in `rocm/`, so the rig needn't install
  ROCm. That runtime is about 60 MB of the package. No AMD rig has mined on
  HiveOS yet.
- On a rig of Intel Arc cards the package's `sycl/` folder mines, with nothing
  in Extra config, and `--core intel` asks for it on a mixed rig (see
  [The Intel core](#the-intel-core-arc)). It is about 5 MB of the package. It
  uses the Intel OpenCL driver HiveOS already has (`intel-opencl-icd` 24.35 on
  the rig tried). On 10 November the package's binary mined on a HiveOS rig
  of six Arc A750 cards, run by hand from a folder: 60.6 TH/s, about 10 a
  card, shares accepted. It has not been installed through a flight sheet on
  an Arc rig, so the hook scripts and HiveOS's stats are unproven there.

`llmjob-earn-cli-linux --help` lists every flag. A flag the CLI doesn't have
makes it exit at once, HiveOS restarts it every few seconds, and the rig's page
in HiveOS names the flag. Old examples such as `--difficulty` no longer exist.

**The miner updates itself.** HiveOS installs a custom miner only when its
Installation URL changes, and then reuses any archive of the same file name it
already downloaded. So the miner does it: `h-run.sh` tells the CLI it runs under
HiveOS (`LLMJOB_HIVEOS=1`), and the CLI checks for a new release when it starts
and every 6 hours. When there is one, it downloads that release's
`llmjob-earn-<version>.tar.gz`, unpacks it beside
`/hive/miners/custom/llmjob-earn`, checks the new CLI runs, swaps the two
folders and exits. HiveOS's miner loop starts the new `h-run.sh` 3 s later. Its
installer leaves the new folder alone: the downloaded archive and the folder are
both there, so it says "Already installed" (`src/cli/hiveos.js`).

The new version is on trial until it hashes, and the old one waits in
`llmjob-earn.prev`. If the new one exits with an error before then, or hasn't
hashed in 10 minutes, `h-run.sh` puts the old folder back and the old version
skips the new one for a day. A run that mines nothing on purpose (every GPU off,
or `--mode llm`) passes its trial after 2 minutes. The HiveOS dashboard says
when a rig updates or goes back.

A rig installed from a versioned URL (`llmjob-earn-0.6.8.tar.gz` and earlier)
doesn't update itself: its `h-config.sh` turned updates off. Change its
Installation URL once to the one above. A rig that already had that URL keeps
its archive: run
`/hive/miners/custom/custom-get https://github.com/super3/llmjob/releases/latest/download/llmjob-earn-hiveos.tar.gz -f`
from Hive Shell, then `miner restart`.

**Problems show on the rig's page in HiveOS**, through HiveOS's own `message`
command, so nobody has to open the miner log to find them: an address that
isn't a Pearl address, a flag the CLI doesn't know, a core that won't load (an
image too old for it, say), and RTX 50 or other Blackwell cards on a driver
older than 570. The same message goes out at most once an hour.

**The rig mines only.** The package adds `--mode mining` unless Extra config
has a `--mode`. The CLI's own default, `auto`, also serves a local LLM. It
downloads a 5 GB model to the HiveOS drive, or about 18 GB on a card with 30 GB
or more free. On such a card it also stops mining while it serves, which
HiveOS's watchdog can take for a hung miner. To serve the LLM anyway, put
`--mode auto` in Extra config (`--mode llm` for the LLM only).

**A card turned off in HiveOS doesn't mine.** HiveOS turns a card off with a
power limit of 1 in its NVIDIA overclock, or with `GPU_DISABLE`, and tells the
miner which cards are still on through `CUDA_VISIBLE_DEVICES`. `h-config.sh`
passes that on as `--gpu-index`, so the CLI mines on the cards still on, and
each one's hashrate lands on its own row. With every card off it passes
`--gpu-index none`: the miner logs that and waits, instead of exiting and being
restarted every few seconds. HiveOS reads the setting when the miner starts, so
restart the miner after turning a card on or off. A `--gpu-index` in Extra
config replaces HiveOS's choice. The local LLM (`--mode auto`) doesn't follow
it. With `--core amd` HiveOS's choice isn't passed on: its numbers are NVIDIA
cards, and the AMD core would take them for AMD ones (`--core intel` is the
same). To leave AMD cards out,
list the ones to mine on with `--gpu-index` in Extra config.

`h-stats.sh` sends HiveOS each card's own hashrate with its PCI bus number,
plus the rig total, shares, uptime and version. HiveOS uses the bus numbers to
put each hashrate on its GPU's row, so an iGPU or a server's BMC display in the
GPU list doesn't shift them. It reads all this from the file the CLI writes
every 10 s (`--stats-file`), and a file older than 2 minutes reports 0.

## Headless CLI (Linux)

For rigs and servers with no desktop, `src/cli/earn-cli.js` mines from the
command line with the same mining core as the GUI, with no Electron and no
window. It shares the GUI's logic (address checks, which cards mine, the stats,
the public-board report, the local LLM), so the two behave the same.

Like the GUI, it **auto-detects** what you don't pin: the lowest-latency
HeroMiners region (it TCP-pings every endpoint on start, and falls back to `us`
if none answers) and the GPUs (`nvidia-smi`), and mines on every card.
`--region`, `--gpu` (the card name shown on the board) and `--gpu-index`
(which cards mine) override what it detects.

```bash
# from this earn/ directory
node src/cli/earn-cli.js --address prl1pYOUR_ADDRESS
# or, once installed (npm i -g / npx): llmjob-earn-cli --address prl1p…
npm run start:cli -- --address prl1pYOUR_ADDRESS   # via the package script
```

The mining core is a CUDA addon (`pearl_core.node`) that runs inside the CLI's
own process, not a separate engine to download. The CLI loads it from beside
its binary (see [Standalone binary](#standalone-binary--self-update)), from
`native/build/Release/` when run from source, or from the file
`PEARL_CORE_PATH` names. It prints a hashrate and shares line about once a
second, and stops cleanly on Ctrl-C.

### Local LLM (`--mode`)

The CLI runs the same local LLM as the GUI. `--mode` picks how the GPU is used:

- `auto` (default) — mine **and** serve a local LLM (the VRAM budgeter
  keeps a mining reserve free and offloads only the model layers that fit).
  `both`, its old name, still works.
- `mining` — mine only; never touches the LLM.
- `llm` — serve the LLM only, no mining (so no `--address` is required).

Because `auto` is the default, a rig started with no `--mode` downloads the
model (~5 GB, once) and serves inference alongside mining. Pass `--mode mining`
to opt out and keep the card purely on shares.

When the LLM runs it spawns llama.cpp's `llama-server` and exposes an
OpenAI-compatible endpoint at `http://127.0.0.1:8080/v1`. The small default model
(`Gemma-4-E4B-it-Q4_K_M`, ~5 GB — ~4.5B *effective* params, so a low VRAM
footprint) is a plain download cached under `~/.local/share/llmjob-earn/llm/`;
point `--llm-model /path/to/model.gguf` at your own to skip it.

**VRAM preflight** — before starting (and before downloading the model), the app
checks free GPU VRAM via `nvidia-smi` and **won't start the LLM unless at least
~6 GB is free** (`model.minVramMb`), so it never spawns `llama-server` into an
out-of-memory crash; it logs a clear "not enough free VRAM" line and skips the
LLM (mining, if enabled, carries on). If VRAM can't be read (non-NVIDIA / no
driver) it proceeds and lets llama.cpp decide.

**Mining pauses while a prompt runs** — when the LLM co-runs with the miner,
a card stops mining the moment its `llama-server` starts answering a request,
and mines again 10 s after the last one ends. Other cards keep mining, and the
pool connection stays open. The board and HiveOS show that card at 0 TH/s
while it is paused. The signal is `llama-server`'s own log, so it covers every
caller: the API endpoint, its web UI, cluster jobs and the network share. A card
busy for 10 minutes without a break mines again regardless. On a rig with
several cards it pauses the card the model really loaded on, by the PCI bus id
of the card it was matched to (see the GPU section), and pauses nothing if it
can't tell, as with two same-name cards whose bus ids could not be read. If
something else takes the card's memory during a pause, the card stays paused
and tries again every 30 s. Demand-driven
`auto` is unchanged: there the miner already stops for as long as the model
serves.

**The LLM restarts when the GPU drops out** — if the driver resets or loses
the card, `llama-server` logs `ErrorDeviceLost` and keeps running, but fails
every request after that. When the app sees that line during a request, it
stops `llama-server` and starts a fresh one 10 s later, which gets the card
back. The request that hit the error fails, even one already streaming its
answer, so a cluster job is reported as failed, not done. If that happens 3 times within an hour, it stops the LLM and says so in
the log. This covers the Vulkan build the app uses on Windows and Linux.

The app also remembers that card and model, in `gpu-lost.json` in its data
folder. Later starts, and launches at login, don't load that model on that
card again: Qwen3.8-27B falls back to Gemma, and a card that also lost its GPU
under Gemma gets no LLM. START LLM on the Chat or API tab clears the record,
and so does an app update. The CLI still tries again on every start.

A `llama-server` that exits on its own after it was serving also restarts,
after 2 s, then 4 s, and so on. If it keeps dying within 5 minutes of coming
up, the app stops after 5 restarts.

`llama-server` comes from a llama.cpp release, a `.tar.gz` on Linux. The CLI
extracts it with `tar` into the same `llm/` dir, so the binary sits next to its
shared libraries. To skip the download, or pin your own build, pass
`--llm-binary /path/to/llama-server`.

```bash
# mine and co-run the local LLM
llmjob-earn-cli --address prl1p… --mode auto --llm-binary /opt/llama/llama-server

# LLM only — no mining, no payout address
llmjob-earn-cli --mode llm --llm-binary /opt/llama/llama-server
```

### Connect to your LLMJob account (`connect`)

Link a headless box to your account so it shows online in your cluster — the
command-line counterpart to the desktop app's **API → Connect** tab (and the
replacement for the old `install.sh` agent). Copy your pairing token from the
dashboard, then:

```bash
llmjob-earn-cli connect --token <pairing-token> [--name my-rig]
```

It creates an Ed25519 key under `~/.local/share/llmjob-earn/node.json` (only the
**public** key ever leaves the machine), self-registers with `POST /api/nodes/join`,
then pings `POST /api/nodes/ping` every 5 minutes with a signed heartbeat + basic
telemetry (GPU / VRAM) so the node stays online. It runs in the foreground (like
the miner) — wrap it in systemd for an unattended rig. Once linked you can re-run
`llmjob-earn-cli connect` with no token to resume pinging; point `--server` at a
self-hosted backend if needed.

### Serve cluster jobs (proxy LLM through LLMJob)

Once a box is **linked** and running the local LLM (`--mode llm` or `auto`, or the
desktop app with the LLM started), it automatically **serves inference relayed
through LLMJob** — no inbound networking required. A caller submits a request to
the server with an API key; the server hands it to an online node; the node polls
`POST /api/jobs/poll`, runs it against its local `llama-server`, and streams the
result back in chunks (`POST /api/jobs/:id/chunks` → `…/complete`). Every call is
**outbound** and signed with the node key, so a GPU behind NAT or a provider
network is reachable through the shared API without opening a port or exposing
`127.0.0.1:8080`. Stop serving by stopping the LLM or disconnecting.

The server also tests each node's speed, so it knows which jobs the node can
finish in time. A speed test is an ordinary cluster job pinned to that node,
so the log shows it as `cluster job … running locally`, and on a card that
also mines it pauses mining like any other prompt. A node gets one when it
first serves, then at most one a day, whether the last one passed or failed.
Real jobs keep a busy node's speed current without any test.

### Standalone binary + self-update

CI packages the CLI into a **standalone single-file Linux executable**
(`llmjob-earn-cli-linux`, built with [Node SEA](https://nodejs.org/api/single-executable-applications.html))
and attaches it to each GitHub Release, so a headless box can run it with **no
Node install**. The mining cores and the CPU worker (`llmjob-xmr-worker`, used
only with `--xmr-address`) are separate files on the same release, and the
binary looks for them beside itself (`pearl_core_cu13.node` is used only on
Blackwell cards, such as the RTX 50 series and RTX PRO Blackwell, with driver
580 or newer; on an AMD rig, add `-O $base/pearl_core_hip.node`, which loads
when nvidia-smi lists no NVIDIA card; on a rig of Intel Arc cards, unpack
`pearl_core_sycl-linux.tar.gz` there too, which makes the `sycl/` folder):

```bash
base=https://github.com/super3/llmjob/releases/latest/download
curl -L -o llmjob-earn-cli $base/llmjob-earn-cli-linux
curl -L -O $base/pearl_core.node -O $base/pearl_core_cu13.node -O $base/llmjob-xmr-worker
chmod +x llmjob-earn-cli llmjob-xmr-worker
./llmjob-earn-cli --address prl1pYOUR_ADDRESS
```

That binary **auto-updates itself**. On start it checks the GitHub "latest
release", and if a newer version is out it downloads the new binary, replaces
itself in place, and re-launches with the same arguments before mining — so a
long-running rig stays current hands-off. Both mining cores beside it
(`pearl_core.node` and `pearl_core_cu13.node`) update with it. The binary and
the 12.8 core are both downloaded before either is replaced, so a failed
download leaves the old pair as it was. The old CUDA 13 core is deleted before
anything is replaced, so a release without one (or a failed download of it)
leaves the 12.8 core in charge rather than last release's CUDA 13 core paired
with a newer binary. The AMD core, `pearl_core_hip.node`, updates with them
on a rig that runs `--core amd` or already has one, and is paired like the
12.8 core: a failed download of it leaves the old install as it was. A rig that
runs `--core amd` without one fetches it on its next start. A release without
an AMD core leaves the old one in place. The Intel core's folder, `sycl/`, is
not updated yet: unpack the new release's archive over it. The CPU worker, `llmjob-xmr-worker`, updates with
it on a rig that mines Monero (`--xmr-address`) or already has one; a
Pearl-only rig never gets it. On a Monero rig, a start that finds the worker
missing, not executable or from another release also fetches this release's
copy, and `update` repairs one that is already there. Opt out per-run with
`--no-update`, or update on demand without (re)starting a mine:

```bash
./llmjob-earn-cli update      # check + self-replace if a newer release exists
```

Run from source (`node src/cli/earn-cli.js`) it doesn't replace anything — it
just prints a notice when a newer release is available (update via git/npm).
`npm run dist:cli` builds the binary locally into `dist/llmjob-earn-cli-linux`.

```
Usage: llmjob-earn-cli --address <prl1p…> [options]
       llmjob-earn-cli connect --token <pairing-token>   Link this box to your LLMJob account
       llmjob-earn-cli update                            Update the CLI to the latest release

  -a, --address <prl1p…>   Your Pearl payout address (not needed with --mode llm)
      --mode <mode>        auto (default) mines and serves a local LLM; mining
                           only mines; llm only serves. "both" means auto.
  -r, --region <id>        Pool region: us/us2/ca/br/de/fi/fr/tr/sg/hk/kr/au
                           (default: the fastest). An old AlphaPool id such as
                           eu1 maps to the nearest.
      --pool <host:port>   Mine Pearl on this stratum endpoint instead of a
                           HeroMiners region, such as prl-us.kryptex.network:7048.
                           It must use the same protocol as HeroMiners.
                           --xmr-pool moves Monero.
      --hive-pool-url <url>  The flight sheet's Pool URL, which the HiveOS package
                           passes on. A HeroMiners address leaves the region
                           to the miner. Another host:port is mined on, as
                           with --pool, which beats it.
  -w, --worker <name>      Worker/rig name (default: this machine's hostname)
  -g, --gpu <card>         GPU name to report on the board (default: from nvidia-smi)
      --gpu-index <list>   Mine only on these GPUs: nvidia-smi indices such as
                           0,2, or "none" (default: every GPU)
      --core <build>       amd for AMD cards (needs the ROCm 7 HIP runtime),
                           intel for Intel Arc cards (needs Intel's Level Zero
                           or OpenCL driver), or cu12/cu13 to force a CUDA
                           build (default: the CUDA build that fits the cards;
                           when nvidia-smi lists none, the Intel core on a rig
                           of Arc cards, else the AMD core first; see above)
      --gpu-wait <how>     How the miner waits for an NVIDIA card between
                           launches: block (default, almost no CPU), poll, or
                           spin (a whole CPU core a card; see above)
      --rotation-fold      Experimental, RTX 40 only: a second kernel, about 4%
                           faster on an RTX 4090, 1.3 GB more card memory; checked
                           on each card, and the normal kernel if anything is
                           missing (default: off; see above)
      --mine-mem-clock <MHz>  Lock each mining GPU's memory clock while it mines
                           (default: an RTX 5090 tries 7001 and keeps it only
                           if faster; 0 turns that off; auto tries a lock on
                           RTX 40 and other RTX 50 cards too; see below)
      --mine-core-offset <auto|0>  auto tries a core clock offset on each RTX
                           card, up to +150 MHz, and keeps the fastest step that
                           computes right (default: 0, off; it
                           overclocks the card; see below)
      --xmr-address <4…|8…>  Also mine Monero (XMR) on the CPU, paid here (see below)
      --xmr-pool <host:port>  Mine Monero on this stratum endpoint instead of
                           HeroMiners, such as xmr-us.kryptex.network:7029
      --cpu-threads <n>    CPU threads for Monero (default: all CPUs but 1, or but 2
                           on 6 CPUs or fewer, at most 1 per 2 MB of L3 cache;
                           off on 4 or fewer or under 8 GB)
      --cpu-priority <p>   Monero thread priority: idle, low or normal (default:
                           low on Linux, idle on Windows)
      --stats-file <path>  Write live stats JSON here every 10 s (HiveOS's h-stats.sh reads it)
      --llm-binary <path>  Use this llama-server instead of downloading one
      --llm-model <path>   Use this GGUF model instead of downloading the default
      --llm-max-instances <n>  Cap how many llama-servers run (default: one per GPU that fits)
      --gate-port <port>   Port the LLM gate serves on (default: 8000), in llm mode
                           and in auto mode unless the LLM co-runs with the miner
      --gate-host <addr>   Address the gate binds (default: 0.0.0.0)
      --gate-quiet <secs>  Seconds with no requests before the GPU goes back to mining (default: 60)
      --no-report          Do not publish live status to the public network board
      --no-serve           Do not serve inference jobs for the LLMJob network
      --no-update          Do not auto-update the CLI to a newer release on start
  -h, --help               Show the full help, with what each option is for
  -v, --version            Print the version and exit
```

### Running on a server (pinned, no surprises)

For unattended rigs, pin what the CLI would otherwise decide or fetch when it
starts:

```bash
llmjob-earn-cli --address prl1p… --mode mining --region de --no-update --no-report
```

- `--mode mining` mines only, so no model is downloaded.
- `--region de` skips the latency probe and always mines in one region.
- `--no-update` keeps this version: the CLI never replaces itself, or the
  mining cores and CPU worker beside it.
- `--no-report` (optional) keeps the rig off the public network board.

The mining cores (`pearl_core.node`, and `pearl_core_cu13.node` for Blackwell
cards) sit beside the binary, so nothing is downloaded to mine. To run a core
you built or vetted yourself, set `PEARL_CORE_PATH=/path/to/pearl_core.node`.

Log lines are **journald-friendly**: the `[HH:MM:SS]` prefix is only added when
stdout is a TTY, so under systemd / `docker logs` (where the collector adds its
own timestamp) the CLI prints unprefixed lines — no double timestamps. A minimal
unit:

```ini
[Service]
ExecStart=/opt/llmjob/llmjob-earn-cli --address prl1p… --mode mining --no-update
Restart=always
```

### Lock the memory clock while mining (`--mine-mem-clock`)

On a power-capped card the fold is limited by watts, not by memory bandwidth.
It moves about 5 GB/s of DRAM traffic (418 MB per launch at a 99.6% L2 hit rate,
measured with Nsight Compute), so GDDR7 at its default 13801 MHz spends power
the SMs could use. Locking the memory clock lower hands that power to the SM
clock. Measured on an RTX 5090 capped at 600 W, same core, runs interleaved in
one session:

| Memory clock | SM clock | TH/s |
|---|---|---|
| 13801 MHz (default) | 682 MHz | 95.58 / 95.90 |
| 7001 MHz (`--mine-mem-clock 7001`) | 742 MHz | 103.18 / 103.55 |

That is +8.0%, with the same work done per SM clock. Going lower (810 or
405 MHz) raises the SM clock further, but the work per clock drops (the
L2/crossbar appears to slow down with the memory P-state), so 7001 is the clock
to try on a 5090.

How much it buys depends on how much of the cap the memory is taking. On the
same card in early September, before its power draw rose, 7001 MHz measured a
tie (native/probes/README.md). That one card is all that has been measured, so
by default a 5090 tries the lock on its own rig and keeps it only if it pays.

```bash
llmjob-earn-cli --address prl1p… --mode mining                        # RTX 5090: try 7001, keep it if faster
llmjob-earn-cli --address prl1p… --mode mining --mine-mem-clock 7001  # any card: lock at 7001, no trial
llmjob-earn-cli --address prl1p… --mode mining --mine-mem-clock 0     # leave the driver's clock
llmjob-earn-cli --address prl1p… --mode mining --mine-mem-clock auto  # other cards too: try a lock, keep it if faster
```

- **Tried by default on the RTX 5090.** A mining RTX 5090 tries a lock at
  7001 MHz, in the CLI and in the desktop app, and keeps it only if that card
  mines faster with it. The card mines 75 seconds at the driver's clock and
  75 seconds locked, twice, and keeps the lock for the rest of the run only if
  it was at least 0.3% faster and ahead in every turn. It mines the whole time,
  and the log says what it measured and what it kept. The lines read like
  this; the rates are the table's above, since no 5090 has run the trial yet:

  ```
  trying a memory clock lock of 7001 MHz on GPU 0 by default (RTX 5090), kept only if the card mines faster; --mine-mem-clock 7001 locks without a trial, --mine-mem-clock 0 leaves the driver's clock
  memory clock trial on GPU 0: mining 75 s at the driver's clock and 75 s locked at 7001 MHz, twice; the lock is kept only if it is faster
  memory clock kept at 7001 MHz on GPU 0 while mining: 103.4 TH/s locked against 95.7 (+8.0%); remembered until 2026-11-18
  ```

  A card that is not faster ends at the driver's clock (`memory clock left to
  the driver on GPU 0`, with the two rates). So does one whose trial cannot
  finish: the card stopped, too few hashrate samples came in, or the app may
  not set clocks. Earlier releases locked a 5090 without a trial. Other
  Blackwell cards (the rest of the RTX 50 line, RTX PRO Blackwell) may gain the
  same way, since they share the hard power cap and the power-bound fold, but
  none has been measured, so they are left alone unless you ask.
- **The answer is remembered for a week.** A card is tried once, not on every
  start: for the next seven days a card that kept its lock is locked at once
  when mining starts, and a card that did not is left alone, with no trial.
  The log says it is the remembered answer and when the card is tried again:

  ```
  memory clock on GPU 0: its trial of 2026-11-11 kept the 7001 MHz lock, 103.4 TH/s locked against 95.7 (+8.0%). Remembered: no trial until 2026-11-18.
  memory clock locked at 7001 MHz on GPU 0 while mining
  ```

  After a week the card is tried again, since drivers, power limits and cooling
  change. Only a finished comparison is remembered: a trial that could not run
  or was cut short leaves nothing, and the next start tries again. The answer
  is kept under the card's index and name, so a different model put in the same
  slot is tried afresh; a second card of the same model in that slot is not
  told apart. It is the lifetime and the key the RTX 30 launch shapes use. The
  answers are in `mem-clock-trials.json`, beside `node.json` for the CLI
  (`~/.local/share/llmjob-earn/`) and in the desktop app's data folder.
- **To try a card again now,** after changing its power limit or its cooling,
  start mining once with `LLMJOB_MINE_MEM_CLOCK_RETRY=1` in the environment, in
  the CLI or the app, or delete `mem-clock-trials.json`. The new answer
  replaces the old one.
- **The trial first hands the card's clock back to the driver.** A miner that
  was killed could not release its lock, and a trial that started on a card
  still locked would compare the lock with itself.
- **`--mine-mem-clock <MHz>` locks at once, with no trial,** on every mining
  card, a 5090 included, whatever is remembered for it. Use it on a card you
  have measured, or on a rig whose mining runs are all shorter than the five
  minutes a trial takes (demand-driven `auto` that serves often): a run that
  ends before its trial does leaves no answer, so such a rig would start a
  trial every time and never finish one.
- **`--mine-mem-clock auto` runs the same trial on more cards.** It covers RTX
  40 cards (tried at 5001 MHz) and every RTX 50 card (tried at 7001 MHz). On an
  RTX 4090, 5001 MHz measured +0.8% at a 450 W limit and +1.3 to +1.7% at 300
  to 400 W; the next memory speed down (810 MHz) lost 11%, which is why cards
  are tried rather than locked outright (`tuning/benchmark.md`, "rig01,
  10 November"). Off unless you ask for it, except on a 5090. Older cards are
  left alone: no lock has been measured on them.
- **`--mine-mem-clock 0` turns it off** in the CLI: no lock and no trial. The
  desktop app has no setting for it in its window yet; its switch is the
  environment variable `LLMJOB_MINE_MEM_CLOCK`, which takes the same values
  (`0` leaves the driver's clock, a number locks every mining card at it with
  no trial, `auto` runs the trial on more cards) and is read on every start.
  Set it the way you set any variable for the app: in the shell that launches
  it on Linux, or in Windows' system environment variables. A bad value is
  ignored with a warning in the log and the default stands. Without it an RTX
  5090 mining alone is tried whenever the app has the rights to set clocks, and
  the log says so.
- **Skipped while an LLM co-runs.** `llama-server` is memory-bandwidth-bound, so
  a card that serves a model alongside mining keeps the driver's clock: no lock
  and no trial. Demand mode releases the lock, and ends a trial, before the
  model starts, so it keeps the default.
- **Needs root.** Setting clocks does, so either run the CLI as root or give its
  user a sudoers rule for `nvidia-smi`. The CLI calls it through `sudo -n`, which
  never waits for a password. The desktop app needs the same: root or that rule
  on Linux, and on Windows it must run as administrator. Without them, it mines
  at the driver's clock: the default's trial says so in one info line per card,
  a requested lock or trial in a warning.

  ```
  # visudo -f /etc/sudoers.d/llmjob-earn   (path from `command -v nvidia-smi`)
  miner ALL=(root) NOPASSWD: /usr/bin/nvidia-smi
  ```

- **Released whenever the miner stops**, before anything else happens: on
  shutdown, and in demand-driven `auto` before the LLM starts. The desktop app
  also releases it the moment START LLM joins a running miner, before the model
  loads. LLM decode is memory-bandwidth-bound, so a model is never served from a
  locked card. When `auto` co-runs the LLM with the miner, neither the default
  nor the flag is applied, and the log says so.
- **A hard kill can't release it.** After a SIGKILL or a crash, in either shell,
  run `sudo nvidia-smi -i <index> -rmc` (on Windows `nvidia-smi -i <index> -rmc`
  as administrator; a reboot clears it too).

### Try a core clock offset while mining (`--mine-core-offset auto`)

**This overclocks the card.** A core clock offset moves a card's whole
clock-to-voltage curve: at +100 MHz it runs each voltage 100 MHz faster than its
maker set it. Read the risks below before you turn it on. It is off unless you
ask for it.

Why a miner would want it: on a card at its power limit the fold is limited by
watts, and an offset buys more clock for the same watts. Measured on one RTX
4090 at its 450 W limit, with a run at offset 0 between every setting
(`tuning/benchmark.md`, "rig01, 10 November: core clock offset on the RTX 4090"):

| Core clock offset | SM clock | TH/s | Against 0 |
|---|---|---|---|
| 0 | 2530–2533 MHz | 317.3–317.5 | — |
| +50 MHz | 2559 MHz | 320.9 | +1.1% |
| +100 MHz | 2593 MHz | 325.3 | +2.5% |
| +150 MHz | 2640 MHz | 331.2 | +4.3% |
| +150 MHz and `--mine-mem-clock 5001` | 2660 MHz | 333.3 | +5.0% |

That is the only card measured. How far a card goes before it computes wrong
differs from card to card and with heat, and a card past that point does not
stop: it finds wrong hashes that still meet the target, and every share it sent
would be rejected. So there is no offset to ask for by number. `auto` makes each
card show what it can do:

```bash
llmjob-earn-cli --address prl1p… --mode mining --mine-core-offset auto
llmjob-earn-cli --address prl1p… --mode mining --mine-core-offset auto --mine-mem-clock auto
```

- **Which cards.** NVIDIA RTX 20, 30, 40 and 50 cards (told by compute
  capability, so the other cards of those generations are tried too). Not AMD
  cards, and not when the AMD core mines.
- **The trial.** Each card mines 75 seconds at offset 0, then at +50, +100 and
  +150 MHz, then at 0 again, with the first 15 seconds of each turn left out.
  +150 MHz is as far as it goes: nothing above it has been measured. The card
  mines the whole time, and the trial takes about six and a half minutes.
- **Every step is checked before it counts.** A share is too rare to show that
  a step computes right, so at the start of each step the miner gives the card
  a job of its own with an easy target and recomputes 100 of the hits it finds,
  the way a pool's verifier would. That takes three or four seconds; those hits
  are never sent anywhere. One wrong hit fails the step, and the card climbs no
  further.
- **What it keeps.** The fastest step that passed its check, was at least 0.3%
  faster than the step before it, and at least 0.3% faster than offset 0 (the
  mean of the two turns at 0). A higher step takes a lower one's place only
  when it is at least 0.3% faster than it. If no step does, the card stays at 0. The log
  says what was measured at each step and what was kept. On the RTX 4090 above
  the trial measured 317.4 TH/s at 0 and 320.7, 325.2 and 331.2 at the three
  steps, and kept +150 MHz (+4.4%). With `--mine-mem-clock auto` as well it
  ended at 333.6 TH/s, +5.0%. The two runs had 42 shares accepted and none
  rejected (`tuning/benchmark.md`, "rig01, 10 November: `--mine-core-offset
  auto`").
- **After the trial**, every share a card finds while it has an offset is
  recomputed before it is sent. If one comes out wrong it is not sent, the card
  goes back to offset 0 at once and stays there for the rest of the run, and
  the log says so in an error line. A card whose core fails and restarts also
  comes back at offset 0 for the rest of the run.
- **With `--mine-mem-clock auto`** the memory clock trial runs first and the
  offset trial starts when it has its answer, so the two never overlap. A lock
  the first one kept stays in place.
- **Never while an LLM shares the card.** When `auto` co-runs the LLM with the
  miner the request is dropped, and the log says so. Demand mode puts the card
  back at offset 0 before the model starts, and the desktop app does the same
  the moment START LLM joins a running miner. Nothing checks what a model
  computes, so a model is never served from a card with an offset.
- **Needs root, or administrator rights on Windows.** The call goes to NVML
  from the miner itself, not through `nvidia-smi`, so a sudoers rule for
  `nvidia-smi` does not cover it: the CLI or the app has to run as root (as
  administrator on Windows). Without the rights it mines at the card's own
  clock and says so in one warning per card. It also needs the CUDA mining core
  of this release or later, and an NVIDIA driver recent enough to have NVML's
  clock offset calls.
- **Set back to 0 whenever the miner stops**, for the cards it changed and only
  those, before anything else happens. The trial starts from offset 0: an
  offset already on the card when mining starts, from an earlier run or from
  another tool such as MSI Afterburner, is replaced and not put back. The log
  says so in a warning, with what it was, so you can set it again afterwards.
- **A hard kill can't put it back.** After a SIGKILL, a crash or a power cut
  while a card has an offset, it stays until the next start with
  `--mine-core-offset auto` (which starts from 0), a driver reload or a reboot.
  `nvidia-smi` has no switch for it.
- **The desktop app** has no setting for it in its window. Its switch is the
  environment variable `LLMJOB_MINE_CORE_OFFSET=auto`, read on every start, set
  the way `LLMJOB_MINE_MEM_CLOCK` is.

**The risks, plainly.** It runs the card faster than its maker set it, and only
one RTX 4090 has been measured. The checks catch a card that computes wrong;
they cannot catch a step so unstable that it crashes the graphics driver or
the PC before any check runs, and a crash can lose unsaved work in other
programs. Nothing longer than a few minutes at an offset has been measured, so
stability over hours is not known: a card that passes at start can fail later
as it warms up, which the check on every share is there for. Running a card
outside its maker's settings may also affect its warranty. If a rig becomes
unstable with this on, turn it off.

## Develop

```bash
npm install        # from this earn/ directory
npm start          # launch the Electron app
npm run start:cli -- --address prl1p…   # run the headless Linux miner
npm test           # jest — 100% coverage gate (see jest.config.js)
```

## Build (Windows + Linux + macOS)

```bash
npm run dist:win     # electron-builder --win    → dist/LLMJob-Earn-Setup-<version>.exe (NSIS)
npm run dist:linux   # electron-builder --linux  → dist/LLMJob-Earn-<version>.AppImage
npm run dist:mac     # electron-builder --mac    → dist/LLMJob-Earn-<version>-arm64.dmg
```

Producing the Windows **installer** must happen on Windows (or Linux + Wine);
the Linux **AppImage** builds on Linux; the macOS **DMG** builds on macOS
(Apple silicon only). CI builds all three — `windows-latest`,
`ubuntu-latest` and `macos-latest` — see
[`.github/workflows/miner-build.yml`](../.github/workflows/miner-build.yml); each
build is uploaded as an artifact and, on a `v*` tag, published to the GitHub
Release.

The mining cores, the Intel core's `sycl/` folder among them, are bundled into the Windows and Linux builds only
(`build.win.extraResources` / `build.linux.extraResources`); the Mac build ships
neither them nor the Windows VC++ runtime DLLs, since it cannot mine. The macOS app
is ad-hoc signed in an `afterPack` hook — see
[macOS (LLM only)](#macos-llm-only) for why, and what it means on first launch.

`src/assets/icon.png` (1024×1024, the source electron-builder converts into the
macOS `.icns` and the Linux icon set) is generated by
`node scripts/build-icon.mjs`; Windows keeps its own `icon.ico`.

---

Not affiliated with Pearl Research Labs or HeroMiners — this is a third-party GUI.
