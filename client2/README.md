# LLMJob Miner (`client2`)

A desktop GUI that wraps the AlphaPool [`alpha-miner`](https://pearl.alphapool.tech/#setup)
engine for Pearl (**PRL**). Paste a payout address, hit **Start**, and mine — no
command line. Built with Electron; **Windows** is the shipped target for now.

> The LLM "co-mining" side of LLMJob comes later — this app is the easy on-ramp
> that gets GPUs onto the network via Pearl mining.

## How it works

- **`src/shared/`** — pure, fully unit-tested logic (no Electron/DOM):
  - `config.js` — pool endpoints, per-card static difficulty, engine metadata, economics.
  - `address.js` — `prl1p…` address validation / shortening.
  - `earnings.js` — PRL/USD per-day estimates.
  - `format.js` — uptime / hashrate / number formatting.
  - `parser.js` — turns `alpha-miner` stdout into structured events (shares, hashrate, connect).
  - `minerArgs.js` — builds the engine argument vector / launcher env (`--password "x;d=N"`, etc.).
  - `simulator.js` — a believable live-stats feed for the in-app preview.
- **`src/main/`** — Electron main process:
  - `minerManager.js` — spawns and supervises the engine (injectable `spawn`, unit-tested).
  - `main.js` / `preload.js` — window, settings persistence, IPC bridge (thin shells).
- **`src/renderer/`** — the GUI (Setup → Running → Settings → Logs), pure display + IPC.

## The mining engine

The pool ships the engine separately (it's statically linked, ~36 MB). Download the
Windows build from the [AlphaPool setup page](https://pearl.alphapool.tech/#setup)
— `AlphaMiner-Pearl-Windows.zip` contains `alpha-miner-windows.exe`. Place it on the
`PATH` or point `binaryPath` at it. Until the engine is present the app shows
**simulated** stats so the UI is usable for preview/demo.

Settings map to the documented knobs: Stratum user `‹address›.‹worker›`, static
difficulty via the password (`x;d=N`), and the regional endpoint
(`us1/us2/eu1/eu2/ru1/sg1/in1.alphapool.tech:5566`).

## Develop

```bash
npm install        # from this client2/ directory
npm start          # launch the Electron app
npm test           # jest — 100% coverage gate on shared/* + minerManager
```

## Build (Windows)

```bash
npm run dist:win   # electron-builder --win  → dist/LLMJob-Miner-<version>-win.exe (NSIS)
```

Producing the Windows **installer** must happen on Windows (or Linux + Wine).
CI builds it on `windows-latest` — see
[`.github/workflows/miner-build.yml`](../.github/workflows/miner-build.yml); the
installer is uploaded as a build artifact.

---

Not affiliated with Pearl Research Labs or AlphaPool — this is a third-party GUI.
