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

## How it works

- **`src/shared/`** — pure, fully unit-tested logic (no Electron/DOM):
  - `config.js` — pool regions (and the old AlphaPool ids they replace), the llama-server and model downloads, economics.
  - `address.js` — `prl1p…` / `mdl1p…` validation, shortening, and the merge-mining combined address.
  - `cliArgs.js` — parses/validates the headless CLI flags into the same settings shape the GUI uses.
  - `llmMode.js` — the compute-mode policy (mining / both / llm / auto → which engines run), shared by the GUI and the CLI.
  - `platform.js` — what each OS can do: whether it can mine (not on macOS) and whether electron-updater can install there.
  - `llama.js` / `vram.js` — build the local `llama-server` command line + parse its output, and size the GPU offload (`--n-gpu-layers`) from free VRAM.
  - `selfUpdate.js` — decides, from the running version + GitHub's latest release, whether the CLI binary should self-update.
  - `miner/` — the Pearl protocol: Stratum messages, PearlHash, and the share proofs the pool checks.
  - `coreVariant.js` — which build of the mining core a rig loads (CUDA 12.8 or CUDA 13).
  - `memClock.js` — which mining cards get their memory clock locked, and at what.
  - `miningStats.js` — accumulates the miner's events into the live stats snapshot.
  - `earnings.js` — PRL/USD per-day estimates.
  - `balance.js` — builds the pool balance lookup and parses the pending + paid response.
  - `minerReport.js` — the payload published to the public network board while mining.
  - `statsFile.js` — the stats JSON the CLI writes for HiveOS's `h-stats.sh`.
  - `gpu.js` — reads `nvidia-smi`'s card list and decides which cards mine (`--gpu-index`, `PEARL_GPU_INDEX`).
  - `region.js` — picks the lowest-latency pool region.
  - `updateStatus.js` — formats the in-app auto-update banner.
  - `format.js` — uptime / hashrate / number formatting.
- **`src/main/`** — Electron main process:
  - `pearlMiner.js` — the miner: one core per card, the pool connection, and share submission.
  - `pearlEngine.js` — wraps `pearlMiner.js` in the start/stop and event interface both shells drive.
  - `pearlCore.js` — loads `pearl_core.node` (built from [`native/`](native)), picking the CUDA 12.8 or CUDA 13 build.
  - `probe.js` — asks `nvidia-smi` about the cards (names, VRAM, temperatures, PCI bus ids), pings the pool regions, and posts the board report.
  - `gpuClocks.js` — locks and releases a card's memory clock while it mines.
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

It mines on **HeroMiners** (`<region>.pearl.herominers.com:1200`), in the
region with the lowest latency unless one is pinned: `us`, `us2`, `ca`, `br`,
`de`, `fi`, `fr`, `tr`, `sg`, `hk`, `kr` or `au`. The pool sends a block header
and a target, and the core chooses how to search.

Earlier versions wrapped AlphaPool's `alpha-miner`. That engine and
that pool are gone. An old AlphaPool region id (`us1`, `eu1`, `eu2`, `ru1`,
`sg1`, `hk1`, `in1`), in saved settings or passed to the CLI, maps to the
nearest HeroMiners region.

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
model (`--main-gpu <index>` each), mining cards included. Its `llama-server` is a
Vulkan build, so those indices are Vulkan's own and the CUDA ordering above does
not apply to them.

### Two builds of the core (CUDA 12.8 and CUDA 13)

A release ships two builds of the mining core side by side, in the installer's
`resources/native/` and beside the CLI binary:

| File | Toolkit | Cards |
|---|---|---|
| `pearl_core.node` | CUDA 12.8 | RTX 20, 30, 40 and 50, A100 / A800 / A30 / CMP 170HX, and H100 / H200 (sm_75/80/86/89/90a/120) — every rig |
| `pearl_core_cu13.node` | CUDA 13.3 | sm_89 and sm_120; picked automatically for RTX 50 / Blackwell only (the sm_89 half measured no faster on a 4090) |

**RTX 30 (Ampere) note.** The sm_86 half of `pearl_core.node` now runs the same
fold path as an RTX 40 (the persistent, eight-warp, tall fold). Its hits check
out (the sm_86 code, run on a 4090), but it has not been run on an Ampere card:
there is no hashrate figure for it, and the previous Ampere path is no longer
in this build. If a 3090 mines slower on this release,
`PEARL_CORE_PATH=<path to the previous release's pearl_core.node>` loads the old
core unchanged, and a report with both numbers is what settles it.

The CUDA 13 compiler produces faster code for Blackwell: on an RTX 5090 at
600 W the same source ran 107.27 TH/s built with CUDA 13.3 against 103.97 with
12.8 (+3.2%). But a CUDA 13 build needs NVIDIA driver 580 or newer, so it is used
only when **the driver is 580+ and every card that will mine is compute 12.x**
(read from `nvidia-smi --query-gpu=index,compute_cap,driver_version`). Everything
else — a 2080 Ti, 3090 or 4090, a mixed 4090 + 5090 rig, an older driver, or a rig where
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
beats both and loads exactly that file.

## Sharing the local LLM on your network

The local LLM listens on `127.0.0.1:8080`, so only this computer can use it.
**Settings → Local network → Share the local LLM on my network** lets phones and
other computers on the same network use it too.

- **How:** the app runs the gate the CLI serves on (`autoGate.createServeGate`)
  on port **8000**, on every network interface. It passes requests to
  `llama-server`, which stays on `127.0.0.1`, so the in-app chat and cluster jobs
  don't change. The address to use, `http://<this computer>:8000/v1`, shows under
  the switch and on the API tab.
- **Off by default, and no key.** While it's on, anyone on the network can use
  the model.
- **When it runs:** only while the local LLM runs (Auto or LLM mode). The switch
  applies at once, with no restart. It's saved in `settings.json` as `shareLlm`.
- **Firewall:** the first time, Windows may ask whether to let LLMJob Earn through
  the firewall. Allow it on private networks.
- **Port 8000 taken** (for example by the CLI on the same computer): the switch
  says so, and the app carries on without it.

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

**It needs the HiveOS 0.6 jammy image (Ubuntu 22.04).** The mining cores need
glibc 2.34 and libstdc++ with `GLIBCXX_3.4.29` (read with `objdump -T` from the
v0.5.12 release files). Jammy has glibc 2.35. The older focal image (Ubuntu
20.04) has glibc 2.31, so the cores don't load there and the rig mines nothing.
RTX 50-series and RTX PRO Blackwell cards also need a newer NVIDIA driver than
the stable image's 550: run `nvidia-driver-update` from Hive Shell.

Flight sheet:

- **Miner** → Custom. **Miner name** → `llmjob-earn`.
- **Installation URL** →
  `https://github.com/super3/llmjob/releases/download/v<version>/llmjob-earn-<version>.tar.gz`,
  with `<version>` from the [latest release](https://github.com/super3/llmjob/releases/latest).
  To update a rig, change the URL to the new version. HiveOS reinstalls only
  when the URL changes, and then reuses any archive of the same file name it
  already downloaded. So an unchanging file name never updates: the
  unversioned `llmjob-earn-hiveos.tar.gz` that each release also carries (at
  `releases/latest/download/`) is good for a first install only. The part of
  the file name before the version must stay `llmjob-earn`. HiveOS takes the
  miner name from it and refuses to install when it differs from Miner name.
  A PR build from CI has the same file name as its version's release, so to
  install a different build of a version a rig already has, delete
  `/hive/miners/custom/downloads/llmjob-earn-<version>.tar.gz` before changing
  the URL. Or, after changing it, run
  `/hive/miners/custom/custom-get <url> -f` from Hive Shell, then
  `miner restart`.
- **Hash algorithm** → `pearlhash`.
- **Wallet and worker template** → `%WAL%`, with your `prl1p…` address as the
  wallet. `%WAL%.%WORKER_NAME%` also works: everything after the first `.` is
  the worker name. Otherwise the worker is the rig's HiveOS name. An old
  `prl1p…+mdl1p…` wallet still mines, but merge mining is retired.
- **Pool URL** → any `host:port`, e.g. `us.pearl.herominers.com:1200`. HiveOS
  won't save the flight sheet without one (`The url field is required`), but
  the miner doesn't read it: it mines on HeroMiners and picks the region with
  the lowest latency when it starts. **Pass** → leave blank.
- **Extra config arguments** → optional CLI flags; leave it empty for the
  defaults. The ones that make sense on a rig:
  - `--region <id>` pins the pool region: `us`, `us2`, `ca`, `br`, `de`, `fi`,
    `fr`, `tr`, `sg`, `hk`, `kr` or `au`. An old AlphaPool id from an older
    flight sheet (`us1`, `eu1`, `eu2`, `ru1`, `sg1`, `hk1`, `in1`) maps to the
    nearest of these, and the miner log says which (`region eu1 is now de`).
  - `--mine-mem-clock <MHz>` locks the memory clock while mining; `0` turns off
    the RTX 5090 default (see below).
  - `--mode auto` also serves a local LLM (next paragraph).
  - `--no-report` keeps the rig off the public network board.

  `llmjob-earn-cli-linux --help` lists every flag. A flag the CLI doesn't have
  makes it exit at once. HiveOS then restarts it every few seconds and shows
  "Miner starting error", and `miner log` names the flag. Old examples such as
  `--difficulty` no longer exist. A region the CLI doesn't know stops it the
  same way.

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
it.

`h-stats.sh` sends HiveOS each card's own hashrate with its PCI bus number,
plus the rig total, shares, uptime and version. HiveOS uses the bus numbers to
put each hashrate on its GPU's row, so an iGPU or a server's BMC display in the
GPU list doesn't shift them. It reads all this from the file the CLI writes
every 10 s (`--stats-file`), and a file older than 2 minutes reports 0.

Self-update is off under HiveOS (`--no-update`): the agent owns the miner, and
a new version arrives with a new Installation URL. To move a single rig without
touching its flight sheet, run the CLI's update from Hive Shell. It replaces the
binary in place but leaves the hook scripts and the `CUSTOM_VERSION` the
dashboard shows as they were:

```
miner stop
/hive/miners/custom/llmjob-earn/llmjob-earn-cli-linux update
miner start
```

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

- `both` / `auto` (default) — mine **and** serve a local LLM (the VRAM budgeter
  keeps a mining reserve free and offloads only the model layers that fit).
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

`llama-server` comes from a llama.cpp release, a `.tar.gz` on Linux. The CLI
extracts it with `tar` into the same `llm/` dir, so the binary sits next to its
shared libraries. To skip the download, or pin your own build, pass
`--llm-binary /path/to/llama-server`.

```bash
# mine and co-run the local LLM
llmjob-earn-cli --address prl1p… --mode both --llm-binary /opt/llama/llama-server

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

Once a box is **linked** and running the local LLM (`--mode llm`/`both`, or the
desktop app with the LLM started), it automatically **serves inference relayed
through LLMJob** — no inbound networking required. A caller submits a request to
the server with an API key; the server hands it to an online node; the node polls
`POST /api/jobs/poll`, runs it against its local `llama-server`, and streams the
result back in chunks (`POST /api/jobs/:id/chunks` → `…/complete`). Every call is
**outbound** and signed with the node key, so a GPU behind NAT or a provider
network is reachable through the shared API without opening a port or exposing
`127.0.0.1:8080`. Stop serving by stopping the LLM or disconnecting.

### Standalone binary + self-update

CI packages the CLI into a **standalone single-file Linux executable**
(`llmjob-earn-cli-linux`, built with [Node SEA](https://nodejs.org/api/single-executable-applications.html))
and attaches it to each GitHub Release, so a headless box can run it with **no
Node install**. The mining cores are separate files on the same release, and
the binary looks for them beside itself (`pearl_core_cu13.node` is used only on
Blackwell cards, such as the RTX 50 series and RTX PRO Blackwell, with driver
580 or newer):

```bash
base=https://github.com/super3/llmjob/releases/latest/download
curl -L -o llmjob-earn-cli $base/llmjob-earn-cli-linux
curl -L -O $base/pearl_core.node -O $base/pearl_core_cu13.node
chmod +x llmjob-earn-cli
./llmjob-earn-cli --address prl1pYOUR_ADDRESS
```

That binary **auto-updates itself**. On start it checks the GitHub "latest
release", and if a newer version is out it downloads the new binary, atomically
replaces itself in place, and re-launches with the same arguments before mining
— so a long-running rig stays current hands-off. Both mining cores beside it
(`pearl_core.node` and `pearl_core_cu13.node`) update with it; the old CUDA 13
core is always deleted first, so a release without one (or a failed download of
it) leaves the 12.8 core in charge rather than last release's CUDA 13 core
paired with a newer binary. Opt out per-run with
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
  -w, --worker <name>      Worker/rig name (default: this machine's hostname)
  -g, --gpu <card>         GPU name to report on the board (default: from nvidia-smi)
      --gpu-index <list>   Mine only on these GPUs: nvidia-smi indices such as
                           0,2, or "none" (default: every GPU)
      --mine-mem-clock <MHz>  Lock each mining GPU's memory clock while it mines
                           (default: 7001 on the RTX 5090; 0 turns it off; see below)
      --stats-file <path>  Write live stats JSON here every 10 s (HiveOS's h-stats.sh reads it)
      --llm-binary <path>  Use this llama-server instead of downloading one
      --llm-model <path>   Use this GGUF model instead of downloading the default
      --llm-max-instances <n>  Cap how many llama-servers run (default: one per GPU that fits)
      --gate-port <port>   Port the auto-mode gate serves on (default: 8000)
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
  mining cores beside it.
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
L2/crossbar appears to slow down with the memory P-state), so use 7001 on a 5090.

How much it buys depends on how much of the cap the memory is taking. On the
same card in early September, before its power draw rose, 7001 MHz measured a
tie (native/probes/README.md). Measure your own card before and after.

```bash
llmjob-earn-cli --address prl1p… --mode mining                        # RTX 5090: 7001 by default
llmjob-earn-cli --address prl1p… --mode mining --mine-mem-clock 7001  # any card: lock at 7001
llmjob-earn-cli --address prl1p… --mode mining --mine-mem-clock 0     # leave the driver's clock
```

- **On by default on the RTX 5090.** A mining RTX 5090 is locked at 7001 MHz,
  in the CLI and in the desktop app. It is the only card measured. Other
  Blackwell cards (the rest of the RTX 50 line, RTX PRO Blackwell) may gain the
  same way, since they share the hard power cap and the power-bound fold, but
  they stay at the driver's clock until one has been measured. Pass the flag to
  lock any card.
- **`--mine-mem-clock 0` turns it off** in the CLI. Any other value replaces the
  default on every mining card. The desktop app has no setting for it in its
  window yet; its switch is the environment variable `LLMJOB_MINE_MEM_CLOCK`,
  which takes the same values (`0` leaves the driver's clock, any other value
  locks every mining card at it) and is read on every start. Set it the way you
  set any variable for the app: in the shell that launches it on Linux, or in
  Windows' system environment variables. A bad value is ignored with a warning
  in the log and the default stands. Without it an RTX 5090 mining alone is
  locked whenever the app has the rights to set clocks, and the log says so.
- **Skipped while an LLM co-runs.** `llama-server` is memory-bandwidth-bound, so
  a card that serves a model alongside mining keeps the driver's clock. Demand
  mode releases the lock before the model starts, so it keeps the default.
- **Needs root.** Setting clocks does, so either run the CLI as root or give its
  user a sudoers rule for `nvidia-smi`. The CLI calls it through `sudo -n`, which
  never waits for a password. The desktop app needs the same: root or that rule
  on Linux, and on Windows it must run as administrator. Without them, it mines
  at the driver's clock: the default lock says so in one info line per card, a
  requested one in a warning.

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

The mining cores are bundled into the Windows and Linux builds only
(`build.win.extraResources` / `build.linux.extraResources`); the Mac build ships
neither them nor the Windows VC++ runtime DLLs, since it cannot mine. The macOS app
is ad-hoc signed in an `afterPack` hook — see
[macOS (LLM only)](#macos-llm-only) for why, and what it means on first launch.

`src/assets/icon.png` (1024×1024, the source electron-builder converts into the
macOS `.icns` and the Linux icon set) is generated by
`node scripts/build-icon.mjs`; Windows keeps its own `icon.ico`.

---

Not affiliated with Pearl Research Labs or HeroMiners — this is a third-party GUI.
