# LLMJob Earn (`earn`)

A desktop GUI that turns the excess compute on GPUs you already own into crypto,
wrapping the AlphaPool [`alpha-miner`](https://pearl.alphapool.tech/#setup) engine
for Pearl (**PRL**). Paste a payout address, hit **Start**, and earn — no command
line. Built with Electron and shipped for **Windows** and **Linux**; headless
rigs can use the [command-line miner](#headless-cli-linux) instead of the GUI.

> The LLM "co-mining" side of LLMJob comes later — this app is the easy on-ramp
> that gets GPUs onto the network and earning today.

**Highlights**

- **Live pool balance** — your pending payout plus lifetime paid for the address.
- **Merge mining** — add an `mdl1p…` address in Settings to also earn ModelOS
  (MDL) on the very same shares, no extra power or hardware.
- **Public board** — while mining, the app publishes live status to the network
  page (your Pearl address only — nothing else is reported).
- **Zero-config** — auto-detects the discrete GPU and its recommended static
  difficulty, picks the lowest-latency pool region, and updates itself.

## How it works

- **`src/shared/`** — pure, fully unit-tested logic (no Electron/DOM):
  - `config.js` — pool endpoints, per-card static difficulty, engine metadata, economics.
  - `address.js` — `prl1p…` / `mdl1p…` validation, shortening, and the merge-mining combined address.
  - `cliArgs.js` — parses/validates the headless CLI flags into the same settings shape the GUI uses.
  - `minerArgs.js` — builds the engine argument vector / launcher env (`--address`, `--worker`, `--password "x;d=N"`, `--force-backend`).
  - `parser.js` — turns `alpha-miner` stdout into structured events (shares, hashrate, connect).
  - `miningStats.js` — accumulates those events into the live stats snapshot.
  - `earnings.js` — PRL/USD per-day estimates.
  - `balance.js` — builds the pool balance lookup and parses the pending + paid response.
  - `minerReport.js` — the payload published to the public network board while mining.
  - `gpu.js` / `region.js` — pick the discrete GPU and the lowest-latency pool region from what's detected.
  - `engine.js` — engine download URLs, binary names, and progress math.
  - `engineError.js` — plain-language guidance for launch failures (incl. antivirus quarantine).
  - `updateStatus.js` — formats the in-app auto-update banner.
  - `format.js` — uptime / hashrate / number formatting.
- **`src/main/`** — Electron main process:
  - `minerManager.js` — spawns and supervises the engine (injectable `spawn`, unit-tested).
  - `engineManager.js` — downloads + installs the engine on first run (injected IO, unit-tested).
  - `main.js` / `preload.js` — window, settings persistence, IPC bridge (thin shells).
- **`src/renderer/`** — the GUI (Setup → Running → Settings → Logs), pure display + IPC.
- **`src/cli/earn-cli.js`** — headless Linux miner (no Electron); a thin IO shell that
  reuses the same `shared/*` logic and process supervisor as the GUI.

## The mining engine

The installer **bundles the engine** — electron-builder `extraResources` ships
`vendor/engine/` to `<resources>/engine/`, so a normal install runs offline with
no unsigned download at runtime. If no bundled binary is present (a dev run, or a
build where antivirus stripped it), the app **downloads it on first Start** and
caches it under the user-data folder (`…/LLMJob Earn/engine/`): on Windows it
fetches `AlphaMiner-Pearl-Windows.zip` from the pool's `/downloads/` path and
extracts `alpha-miner-windows.exe` (via PowerShell `Expand-Archive`, no extra
dependency; base URL overridable). If that also fails it surfaces a plain-language
engine error (with antivirus-quarantine guidance) — the stats shown are always the
engine's real output, never simulated. Point `binaryPath` at your own build to
skip the download entirely.

The app drives `alpha-miner` with its documented CLI: `--address prl1…` (or
`prl1…+mdl1…` when merge mining), `--worker`, static difficulty via
`--password "x;d=N"`, an optional `--force-backend` for cards that need it, and
the regional endpoint (`us1/us2/eu1/eu2/ru1/sg1/in1.alphapool.tech:5566`).

## Headless CLI (Linux)

For rigs and servers with no desktop, `src/cli/earn-cli.js` runs the exact same
engine from the command line — no Electron, no window. It shares all the logic
with the GUI (engine download, `prl1…`/`mdl1…` addresses, static difficulty,
merge mining, the public-board report), so behaviour matches the app.

```bash
# from this earn/ directory
node src/cli/earn-cli.js --address prl1pYOUR_ADDRESS
# or, once installed (npm i -g / npx): llmjob-earn-cli --address prl1p…
npm run start:cli -- --address prl1pYOUR_ADDRESS   # via the package script
```

On first run it downloads the Linux `alpha-miner` binary from the pool and
caches it under `~/.local/share/llmjob-earn/engine/` (override with
`--engine-dir`, or skip the download entirely with `--binary /path/to/alpha-miner`).
It streams the engine's real output, prints a periodic hashrate/share summary,
and shuts the engine down cleanly on Ctrl-C.

```
Usage: llmjob-earn-cli --address <prl1p…> [options]

  -a, --address <prl1p…>   Your Pearl payout address (required)
  -m, --mdl <mdl1p…>       Also merge-mine ModelOS (MDL) on the same shares
  -r, --region <id>        Pool region: us1/us2/eu1/eu2/ru1/sg1/in1 (default: us2)
  -w, --worker <name>      Worker/rig name (default: rig01)
  -d, --difficulty <n>     Static share difficulty (default: from --gpu, else 524288)
  -g, --gpu <card>         GPU name, used to auto-pick a static difficulty
      --backend <name>     Force an engine backend (e.g. ampere)
  -b, --binary <path>      Use this alpha-miner binary instead of downloading one
      --engine-dir <path>  Where to cache the downloaded engine
      --no-report          Do not publish live status to the public network board
  -h, --help / -v, --version
```

## Develop

```bash
npm install        # from this earn/ directory
npm start          # launch the Electron app
npm run start:cli -- --address prl1p…   # run the headless Linux miner
npm test           # jest — 100% coverage gate on shared/* + miner/engineManager
```

## Build (Windows + Linux)

```bash
npm run dist:win     # electron-builder --win    → dist/LLMJob-Earn-Setup-<version>.exe (NSIS)
npm run dist:linux   # electron-builder --linux  → dist/LLMJob-Earn-<version>.AppImage
```

Producing the Windows **installer** must happen on Windows (or Linux + Wine);
the Linux **AppImage** builds on Linux. CI builds both — Windows on
`windows-latest` and Linux on `ubuntu-latest` — see
[`.github/workflows/miner-build.yml`](../.github/workflows/miner-build.yml); each
build is uploaded as an artifact and, on a `v*` tag, published to the GitHub
Release.

---

Not affiliated with Pearl Research Labs or AlphaPool — this is a third-party GUI.
