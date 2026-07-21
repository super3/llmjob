# LLMJob Earn (`earn`)

A desktop GUI that turns the excess compute on GPUs you already own into crypto,
wrapping the AlphaPool [`alpha-miner`](https://pearl.alphapool.tech/#setup) engine
for Pearl (**PRL**). Paste a payout address, hit **Start**, and earn — no command
line. Built with Electron and shipped for **Windows** and **Linux**; headless
rigs can use the [command-line miner](#headless-cli-linux) instead of the GUI.

> The LLM "co-mining" side of LLMJob is landing: both clients (GUI and headless
> CLI) can now run a local llama.cpp `llama-server` alongside — or instead of —
> mining, exposing an OpenAI-compatible endpoint at `127.0.0.1:8080/v1`. Pick the
> **Compute Mode** in the GUI's Settings, or `--mode` on the CLI.

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
  - `llmMode.js` — the compute-mode policy (mining / both / llm / auto → which engines run), shared by the GUI and the CLI.
  - `llama.js` / `vram.js` — build the local `llama-server` command line + parse its output, and size the GPU offload (`--n-gpu-layers`) from free VRAM.
  - `selfUpdate.js` — decides, from the running version + GitHub's latest release, whether the CLI binary should self-update.
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
  - `llmManager.js` / `llmEngineManager.js` — spawn/supervise the local `llama-server` and download its binary + GGUF model on demand (same injectable pattern as the miner pair, unit-tested).
  - `main.js` / `preload.js` — window, settings persistence, IPC bridge (thin shells).
- **`src/renderer/`** — the GUI (Setup → Running → Settings → Logs), pure display + IPC.
- **`src/cli/`** — headless Linux miner (no Electron); thin IO shells that reuse
  the same `shared/*` logic and process supervisor as the GUI:
  - `earn-cli.js` — the CLI entry (arg handling, engine resolution, run loop, self-update check).
  - `selfUpdater.js` — the IO side of self-update (GitHub fetch, download, atomic self-replace, re-exec).
  - `sea-entry.js` — entry shim for the packaged single-file binary (`scripts/build-cli.mjs`).

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

The app drives `alpha-miner` with its documented CLI: `--address prl1…`,
`--worker`, static difficulty via `--password "x;d=N"`, an optional
`--force-backend` for cards that need it, and the regional endpoint
(`us1/us2/eu1/eu2/ru1/sg1/hk1/in1.alphapool.tech:5566`). Merge mining differs by
platform: Windows appends the MDL address to `--address` as `prl1…+mdl1…`, while
Linux passes it in the password's `mdl=` field (`x;d=N;mdl=mdl1…`) because the
Linux engine validates `--address` as a single bech32m address and rejects the
combined form.

On Linux the engine version is picked per rig (`shared/engine.js`): driver
≥ 580 gets the faster CUDA 13 build (`alpha-miner-1.8.8`, 3–8% more hashrate on
40/50-series), older drivers stay on the CUDA 12 stable (`alpha-miner-1.8.3`).
The version is part of the cached filename, so bumping it forces a fresh
download instead of trusting a stale cache.

## HiveOS (flight sheet)

The release ships a HiveOS custom-miner package wrapping the headless CLI
(`hiveos/` + `scripts/build-hiveos.mjs` → `llmjob-earn-hiveos-<version>.tar.gz`,
versioned because HiveOS caches the download by filename and can leave rigs
stuck on an old build when the name never changes; an unversioned
`llmjob-earn-hiveos.tar.gz` copy is still published for flight sheets that
predate the rename):

- **Miner** → Custom · **Miner name** → `llmjob-earn`
- **Installation URL** → the versioned tarball from the [latest release](https://github.com/super3/llmjob/releases/latest),
  e.g. `https://github.com/super3/llmjob/releases/download/v0.1.17/llmjob-earn-hiveos-0.1.17.tar.gz` —
  update the URL (and rigs re-download automatically) when a new version ships
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
Self-update is disabled under HiveOS — the agent owns the lifecycle, so updates
arrive by reinstalling the package URL.

## Headless CLI (Linux)

For rigs and servers with no desktop, `src/cli/earn-cli.js` runs the exact same
engine from the command line — no Electron, no window. It shares all the logic
with the GUI (engine download, `prl1…`/`mdl1…` addresses, static difficulty,
merge mining, the public-board report), so behaviour matches the app.

Like the GUI, it **auto-detects** the bits you don't pin: the lowest-latency
pool region (it TCP-pings every endpoint on start) and the GPU (`nvidia-smi`),
picking that card's recommended static difficulty from the table. Both are
best-effort — if the pings all fail it falls back to `us2`, and if there's no
`nvidia-smi` it falls back to the default difficulty — and an explicit
`--region`, `--gpu`, or `--difficulty` always overrides the detected value.

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

### Local LLM (`--mode`)

The CLI runs the same local LLM as the GUI. `--mode` picks how the GPU is used:

- `mining` (default) — mine only; unchanged from before.
- `both` / `auto` — mine **and** serve a local LLM (the VRAM budgeter keeps a
  mining reserve free and offloads only the model layers that fit).
- `llm` — serve the LLM only, no mining (so no `--address` is required).

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

The pool ships `llama-server` as a release **zip**; the CLI downloads and
extracts it with `unzip` (flattening the archive so the binary sits next to its
shared libraries), caching it under the same `llm/` dir. If `unzip` isn't
installed — or you'd rather pin your own build — pass a prebuilt binary with
`--llm-binary /path/to/llama-server` to skip the download entirely.

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
Node install**:

```bash
curl -L -o llmjob-earn-cli https://github.com/super3/llmjob/releases/latest/download/llmjob-earn-cli-linux
chmod +x llmjob-earn-cli
./llmjob-earn-cli --address prl1pYOUR_ADDRESS
```

That binary **auto-updates itself**. On start it checks the GitHub "latest
release", and if a newer version is out it downloads the new binary, atomically
replaces itself in place, and re-launches with the same arguments before mining
— so a long-running rig stays current hands-off. Opt out per-run with
`--no-update`, or update on demand without (re)starting a mine:

```bash
./llmjob-earn-cli update      # check + self-replace if a newer release exists
```

Run from source (`node src/cli/earn-cli.js`) it doesn't replace anything — it
just prints a notice when a newer release is available (update via git/npm).
`npm run dist:cli` builds the binary locally into `dist/llmjob-earn-cli-linux`.

```
Usage: llmjob-earn-cli --address <prl1p…> [options]

  -a, --address <prl1p…>   Your Pearl payout address (required unless --mode llm)
  -m, --mdl <mdl1p…>       Also merge-mine ModelOS (MDL) on the same shares
      --mode <mode>        Compute mode: mining/both/llm/auto (default: mining)
      --llm-binary <path>  Path to a llama-server binary (to run the local LLM)
      --llm-model <path>   Path to a GGUF model file (default: download the small model)
  -r, --region <id>        Pool region: us1/us2/eu1/eu2/ru1/sg1/hk1/in1 (default: auto-detect fastest)
  -w, --worker <name>      Worker/rig name (default: this machine's hostname)
  -d, --difficulty <n>     Static share difficulty (default: from detected/--gpu card, else 524288)
  -g, --gpu <card>         GPU name for the difficulty table (default: auto-detect via nvidia-smi)
      --backend <name>     Force an engine backend (e.g. ampere)
  -b, --binary <path>      Use this alpha-miner binary instead of downloading one
      --engine-dir <path>  Where to cache the downloaded engine
      --no-report          Do not publish live status to the public network board
      --no-update          Do not auto-update the CLI to a newer release on start
  -h, --help / -v, --version
```

### Running on a server (pinned, no surprises)

For unattended / production rigs, prefer a fully-pinned setup — a vetted engine
you control, no background self-updates, and no outbound fetches at start:

```bash
llmjob-earn-cli --address prl1p… \
  --binary /opt/llmjob/alpha-miner \   # vetted engine you placed — no download, no engine drift
  --no-update \                        # don't self-replace the CLI binary
  --no-report                          # optional: don't publish to the public board
```

`--binary` skips the on-demand engine download entirely and pins a known-good
`alpha-miner` (download + audit it once, then point every host at it), so an
engine bump never lands on a box without you choosing it. `--no-update` does the
same for the CLI itself.

Log lines are **journald-friendly**: the `[HH:MM:SS]` prefix is only added when
stdout is a TTY, so under systemd / `docker logs` (where the collector adds its
own timestamp) the CLI prints unprefixed lines — no double timestamps. A minimal
unit:

```ini
[Service]
ExecStart=/opt/llmjob/llmjob-earn-cli --address prl1p… --binary /opt/llmjob/alpha-miner --no-update
Restart=always
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
