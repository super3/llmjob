# Bundled engine

electron-builder copies the contents of this directory into the packaged app at
`<resources>/engine` (see `build.extraResources` in `package.json`). At runtime
`main.js` prefers a binary found here (`shared/engine.bundledEnginePath`) and
only falls back to the on-demand download when it's absent.

The miner binaries themselves are **not** committed (antivirus-flagged and large
— see `.gitignore`, which keeps only this README). They are fetched into this
folder at build time by the `Earn build (Windows + Linux)` workflow before
packaging, so release installers ship with them while the repo stays clean. A
local `npm run dist:win` / `dist:linux` without them present simply builds an
installer that downloads the engine on first run (the previous behaviour).

- **Windows** gets the single pool build, staged under the legacy unversioned
  name `alpha-miner-windows.exe`.
- **Linux** gets *both* versioned builds (`alpha-miner-<preferred>` and
  `alpha-miner-<fallback>`, from `ENGINE` in `shared/engine.js`). The version is
  chosen per rig at run time from the NVIDIA driver
  (`shared/engine.pickEngineVersion` — driver >= 580 gets the faster CUDA 13
  build, older drivers the CUDA 12 stable), and `bundledEnginePath` is
  version-aware, so shipping both keeps that choice intact. A rig whose driver
  selects a build that isn't bundled — an AppImage older than a version bump —
  falls back to the on-demand download.

Bundled Linux binaries must keep their execute bit through packaging (the
workflow chmods them; `main.js` re-asserts it best-effort at startup, which is a
no-op inside the read-only AppImage mount).
