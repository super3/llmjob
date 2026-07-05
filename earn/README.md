# LLMJob Earn (`earn`)

A desktop GUI that turns the excess compute on GPUs you already own into crypto,
wrapping the AlphaPool [`alpha-miner`](https://pearl.alphapool.tech/#setup) engine
for Pearl (**PRL**). Paste a payout address, hit **Start**, and earn — no command
line. Built with Electron; **Windows** is the shipped target for now.

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

## Develop

```bash
npm install        # from this earn/ directory
npm start          # launch the Electron app
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
