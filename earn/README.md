# LLMJob Earn (`earn`)

A free desktop GUI that mines Pearl (**PRL**) on the GPUs you already own, with
our own CUDA miner built in. Paste a payout address, hit **Start**, and earn —
no command line, no account, no fee. Built with Electron and shipped for
**Windows** and **Linux**; headless rigs can use the
[command-line miner](#headless-cli-linux) instead of the GUI.

**Highlights**

- **Our own miner** — the PearlHash core in [`native/`](native) is compiled into
  a Node addon (`pearl_core.node`) and runs inside the app. Nothing third-party
  to download, and nothing for antivirus to quarantine.
- **No fees** — no dev fee, and the pool ([HeroMiners](https://pearl.herominers.com))
  takes no pool fee. Payouts go hourly, straight to your address.
- **Live pool balance** — your pending payout plus lifetime paid for the address.
- **Public board** — while mining, the app publishes live status to the
  [network page](https://llmjob.com/network), one row per card.
- **Zero-config** — detects every NVIDIA GPU, picks the lowest-latency pool
  region, and updates itself.

## How it works

- **`src/shared/`** — pure, fully unit-tested logic (no Electron/DOM):
  - `config.js` — pool endpoints and regions, engine metadata, economics.
  - `address.js` — `prl1p…` / `mdl1p…` validation and shortening.
  - `cliArgs.js` — parses/validates the headless CLI flags into the same settings shape the GUI uses.
  - `platform.js` — what each OS can do: whether it can mine (not on macOS) and whether electron-updater can install there.
  - `miner/` — the protocol half of the miner in JS: stratum, PearlHash, BLAKE3, Merkle proofs, share proofs.
  - `miningStats.js` — accumulates engine events into the live stats snapshot.
  - `earnings.js` / `economics.js` — PRL/USD per-day estimates from live prlscan data.
  - `balance.js` — builds the pool balance lookup and parses the pending + paid response.
  - `minerReport.js` — the payload published to the public network board while mining.
  - `statsFile.js` — the JSON the CLI writes with `--stats-file` (read by HiveOS's `h-stats.sh`).
  - `gpu.js` / `region.js` — GPU detection parsing and the lowest-latency pool region.
  - `selfUpdate.js` — decides, from the running version + GitHub's latest release, whether the CLI binary should self-update.
  - `updateStatus.js` — formats the in-app auto-update banner.
  - `format.js`, `settingsStore.js`, `worker.js` — formatting, settings persistence, the default worker name.
- **`src/main/`** — Electron main process and the shared engine:
  - `pearlCore.js` / `pearlMiner.js` / `pearlEngine.js` — load the native core, drive one core per card, and translate its events for the UI.
  - `probe.js` — GPU, VRAM, temperature and region detection.
  - `io.js` — JSON requests and hardened file downloads (used by the CLI self-updater).
  - `main.js` / `preload.js` — window, settings persistence, IPC bridge (thin shells).
- **`src/renderer/`** — the GUI (Mine → Settings → Logs), pure display + IPC.
- **`src/cli/`** — headless Linux miner (no Electron); thin IO shells that reuse
  the same `shared/*` logic and engine as the GUI:
  - `earn-cli.js` — the CLI entry (arg handling, detection, run loop, self-update check).
  - `selfUpdater.js` — the IO side of self-update (GitHub fetch, download, atomic self-replace, re-exec).
  - `sea-entry.js` — entry shim for the packaged single-file binary (`scripts/build-cli.mjs`).

## The mining engine

The miner is this process. The GPU work is a native addon, `pearl_core.node`,
built from [`native/`](native) and loaded in-process, so there is no binary to
download, no version to select and nothing to spawn. The installer ships it
under `<resources>/native/`; the CLI looks for it beside its executable. Set
`PEARL_CORE_PATH` to point either at a specific build.

A build with no loadable core can't mine, and says so: the GUI shows the error
under **Start**, and the CLI exits non-zero so a supervisor like systemd sees
the failure instead of a silent restart loop.

The pool is HeroMiners (`<region>.pearl.herominers.com:1200`). Its terms, read
from its own stats API: no fee, hourly payouts, a 1 PRL minimum, proportional
rewards.

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

A card that can't start is skipped, not fatal — the rest of the rig keeps
mining.

Both shells set `CUDA_DEVICE_ORDER=PCI_BUS_ID` at startup, so "GPU 1" means the
same card to the miner as it does to `nvidia-smi`. Left to itself the CUDA
runtime numbers cards by its own "fastest first" heuristic, which on a mixed rig
is a different card than the one `nvidia-smi` lists first.

To mine on one specific card, set `PEARL_GPU_INDEX` to its `nvidia-smi` index:

```bash
PEARL_GPU_INDEX=1 llmjob-earn-cli --address prl1p…
```

## macOS

The Pearl core is CUDA and Macs have no NVIDIA GPU, so the app cannot mine on
macOS. `src/shared/platform.js` refuses the miner up front: the GUI keeps
**Start** disabled and says why, and the CLI exits with the same explanation.
No macOS build is published.

## HiveOS (flight sheet)

> **Not currently published.** As of v0.5.0 the release no longer builds or
> attaches the HiveOS package: it had gone weeks without being exercised, and a
> release should not carry an artifact nobody has verified. The packaging script
> and hook scripts below are still in the tree, so restoring it is re-adding two
> steps to `.github/workflows/miner-build.yml`. Rigs on a flight sheet pinned to
> an older release keep working — they install by URL from the release they
> already name. The rest of this section describes how it works when enabled.

The release ships a HiveOS custom-miner package wrapping the headless CLI
(`hiveos/` + `scripts/build-hiveos.mjs` → `llmjob-earn-<version>.tar.gz`,
versioned because HiveOS caches the download by filename and can leave rigs
stuck on an old build when the name never changes; an unversioned
`llmjob-earn-hiveos.tar.gz` copy is still published for flight sheets that
predate the rename):

- **Miner** → Custom · **Miner name** → `llmjob-earn`
- **Installation URL** → the versioned tarball from the [latest release](https://github.com/super3/llmjob/releases/latest),
  i.e. `https://github.com/super3/llmjob/releases/download/v<version>/llmjob-earn-<version>.tar.gz` —
  update the URL (and rigs re-download automatically) when a new version ships

The versioned filename's stem must stay exactly `llmjob-earn`, matching
`CUSTOM_NAME`. HiveOS splits a `<name>-<version>.tar.gz` install URL to derive
the miner name and then looks for `<name>/h-manifest.conf` inside the archive,
so a stem of `llmjob-earn-hiveos` made every install fail with
`No llmjob-earn-hiveos/h-manifest.conf` while leaving the rig on its previous
build. Releases v0.1.17–v0.3.0 shipped that broken name; rigs pointed at one of
those URLs need the flight sheet moved to a `llmjob-earn-<version>.tar.gz` URL.
- **Wallet** → your `prl1p…` address only (HiveOS caps the wallet field at 90
  characters, so the combined `prl1p…+mdl1p…` form doesn't fit)
- **Pool URL** → any non-empty placeholder (e.g. `alphapool.tech:5566`) — HiveOS
  refuses to save the flight sheet without it (`The url field is required`), but
  the miner ignores it and auto-picks the fastest region (override with
  `--region` in extra config); leave **Pass** blank
- **Extra config arguments** → `--mdl mdl1p…` to merge-mine MDL, plus any other
  CLI flags, e.g. `--region eu1` (optional)

The worker name comes from the rig's HiveOS name, and the dashboard gets live
hashrate/shares via `h-stats.sh`, which reads the JSON the CLI writes with
`--stats-file` (10s cadence; a stale file reports zeros rather than lying).
Auto-update on start is disabled under HiveOS (`--no-update`) — the agent owns
the lifecycle, so updates normally arrive by reinstalling the package URL. To
move a single rig without touching its flight sheet, run the CLI's explicit
update from Hive Shell; it replaces the binary in place but leaves the wrapper
scripts (and the `CUSTOM_VERSION` the dashboard displays) as they were:

```
miner stop
/hive/miners/custom/llmjob-earn/llmjob-earn-cli-linux update
miner start
```

## Headless CLI (Linux)

For rigs and servers with no desktop, `src/cli/earn-cli.js` runs the exact same
miner from the command line — no Electron, no window. It shares all the logic
with the GUI (addresses, region detection, the engine, the public-board report),
so behaviour matches the app.

Like the GUI, it **auto-detects** the bits you don't pin: the lowest-latency
pool region (it TCP-pings every endpoint on start), a worker name from the
hostname, and every GPU (`nvidia-smi`). An explicit `--region`, `--worker` or
`--gpu` always wins.

```bash
# from this earn/ directory
node src/cli/earn-cli.js --address prl1pYOUR_ADDRESS
npm run start:cli -- --address prl1pYOUR_ADDRESS   # via the package script
```

It prints a hashrate/share summary once a second and shuts the miner down
cleanly on Ctrl-C.

```
Usage: llmjob-earn-cli --address <prl1p…> [options]
       llmjob-earn-cli update                            Update the CLI to the latest release

Required:
  -a, --address <prl1p…>   Your Pearl payout address

Options:
  -r, --region <id>        Pool region (default: auto-detect fastest)
  -w, --worker <name>      Worker/rig name (default: this machine's hostname)
  -g, --gpu <card>         GPU name to report on the board (default: auto-detect via nvidia-smi)
      --stats-file <path>  Write live stats JSON here every 10s (for HiveOS h-stats etc.)
      --no-report          Do not publish live status to the public network board
      --no-update          Do not auto-update the CLI to a newer release on start
  -h, --help               Show this help and exit
  -v, --version            Print the version and exit
```

**Retired flags.** The local-LLM options (`--mode`, `--no-serve`, `--llm-*`,
`--gate-*`) were removed with the LLM. They are still accepted and ignored, with
one line saying so at startup, so a unit or flight sheet written for an older
build keeps mining after the CLI updates itself. The `connect` subcommand is
retired too.

### Standalone binary + self-update

CI packages the CLI into a **standalone single-file Linux executable**
(`llmjob-earn-cli-linux`, built with [Node SEA](https://nodejs.org/api/single-executable-applications.html))
and attaches it to each GitHub Release, next to `pearl_core.node`, so a
headless box can run it with **no Node install**:

```bash
curl -L -o llmjob-earn-cli https://github.com/super3/llmjob/releases/latest/download/llmjob-earn-cli-linux
curl -L -o pearl_core.node https://github.com/super3/llmjob/releases/latest/download/pearl_core.node
chmod +x llmjob-earn-cli
./llmjob-earn-cli --address prl1pYOUR_ADDRESS
```

That binary **auto-updates itself**. On start it checks the GitHub "latest
release", and if a newer version is out it downloads the new binary (and core),
atomically replaces itself in place, and re-launches with the same arguments
before mining — so a long-running rig stays current hands-off. Opt out per-run
with `--no-update`, or update on demand without (re)starting a mine:

```bash
./llmjob-earn-cli update      # check + self-replace if a newer release exists
```

Run from source (`node src/cli/earn-cli.js`) it doesn't replace anything — it
just prints a notice when a newer release is available (update via git/npm).
`npm run dist:cli` builds the binary locally into `dist/llmjob-earn-cli-linux`.

### Running on a server

For unattended rigs, pin the version with `--no-update` and update on your own
schedule with `llmjob-earn-cli update`. Log lines are **journald-friendly**: the
`[HH:MM:SS]` prefix is only added when stdout is a TTY, so under systemd /
`docker logs` (where the collector adds its own timestamp) the CLI prints
unprefixed lines — no double timestamps. A minimal unit:

```ini
[Service]
ExecStart=/opt/llmjob/llmjob-earn-cli --address prl1p… --no-update
Restart=always
```

## Develop

```bash
npm install        # from this earn/ directory
npm start          # launch the Electron app
npm run start:cli -- --address prl1p…   # run the headless Linux miner
npm test           # jest — 100% coverage gate (see jest.config.js)
```

## Build (Windows + Linux)

```bash
npm run dist:win     # electron-builder --win    → dist/LLMJob-Earn-Setup-<version>.exe (NSIS)
npm run dist:linux   # electron-builder --linux  → dist/LLMJob-Earn-<version>.AppImage
```

Producing the Windows **installer** must happen on Windows (or Linux + Wine);
the Linux **AppImage** builds on Linux. CI builds both — `windows-latest` and
`ubuntu-latest` — see
[`.github/workflows/miner-build.yml`](../.github/workflows/miner-build.yml); each
build is uploaded as an artifact and, on a `v*` tag, published to the GitHub
Release.

`src/assets/icon.png` (1024×1024, the source electron-builder converts into the
Linux icon set) is generated by `node scripts/build-icon.mjs`; Windows keeps its
own `icon.ico`.

---

Not affiliated with Pearl Research Labs or HeroMiners — this is a third-party miner.
