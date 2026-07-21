# Bundled engine

electron-builder copies the contents of this directory into the packaged app at
`<resources>/engine` (see `build.extraResources` in `package.json`). At runtime
`main.js` prefers a binary found here (`shared/engine.bundledEnginePath`) and
only falls back to the on-demand download when it's absent.

The miner binary itself is **not** committed (it's antivirus-flagged and large —
see `.gitignore`'s `*.exe`). On Windows it is fetched into this folder at build
time by the `Earn build (Windows)` workflow before packaging, so release
installers ship with it while the repo stays clean. A local `npm run dist:win`
without the binary present simply builds an installer that downloads the engine
on first run (the previous behaviour).

No Linux engine is bundled: the engine version is chosen per rig at run time
from the NVIDIA driver version (`shared/engine.pickEngineVersion` — driver
>= 580 gets the faster CUDA 13 build, older drivers the CUDA 12 stable), so the
app downloads the matching versioned binary on first Start instead.
