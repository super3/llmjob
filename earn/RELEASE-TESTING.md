# Earn — pre-release GPU test plan

CI proves the pure logic: 40-odd suites with `spawn`, `fs`, the GPU probe and
every HTTP call injected. What it structurally cannot reach is the boundary
where those injections stop — a real NVIDIA driver, a real engine process, a
real `llama-server` holding real VRAM, and a real round trip to the deployed
board. Every field bug this app has had lives on that boundary.

Run this on the Windows GPU box against the build that is about to ship.

**Order matters.** Phases are arranged so a failure invalidates as little
downstream work as possible. Stop at the first ✗ in phases 1–4; a failure there
means later results are meaningless.

## What CI already covers — do not re-test by hand

`buildMinerReports` shapes, `planLlmInstances` VRAM maths, `LlmFleet`
supervision, mode resolution, arg building, parser output, settings
persistence. If one of these is wrong, the unit suite is the place to prove it,
not the GUI. Hand-testing them wastes the GPU box's time.

---

## Phase 0 — baseline

```bash
cd earn && npm test
```

```bash
npm test && npm run lint && npm run build:site
```

Both suites green before anything else. A release never ships on a red suite.

Record the version under test: `earn/package.json` → `version`. Every check
below that reports a version must show *this* one — a stale build serving old
code is the single easiest way to get a false pass.

---

## Phase 1 — GPU detection and the VRAM gate

The planner's inputs come from `nvidia-smi`. If the parse is wrong everything
downstream is wrong, so establish ground truth first.

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
```

Launch the app (`npm start` in `earn/`), then check the GPU Activity header
names the same card `nvidia-smi` reports, and that the app's VRAM figure tracks
the driver's within a few hundred MB.

**Gate check — the branch CI can only fake.** With the model needing ~6 GB free
(`LLM.model.minVramMb`), occupy the card until less than that remains (another
`llama-server`, a game, `nvidia-smi -q` to confirm), then start in `auto` mode.

- ✓ log reads `not enough free VRAM for the local LLM: <N> MB free, need ~5800 MB`
- ✓ the app keeps **mining** rather than dying
- ✓ the LLM indicator shows the `Needs ~6 GB free VRAM` error, not a spinner

Free the VRAM, restart, confirm the LLM now starts. This gate has to fail
*gracefully* — a hard failure here strands every low-VRAM user on the network.

---

## Phase 2 — process spawn and GPU pinning

The most valuable single check in this document, and the one that caught the
`--main-gpu` wiring this session.

```powershell
Get-Process | Where-Object { $_.ProcessName -match 'llama|alpha' } | Select-Object Id, ProcessName, StartTime
```

For each PID, read back the command line the app actually built:

```powershell
(Get-CimInstance Win32_Process -Filter "ProcessId=<PID>").CommandLine
```

- ✓ `alpha-miner` carries the address, worker and difficulty from `settings.json`
- ✓ `llama-server` carries `--main-gpu <index>` with a **real index**, not absent
- ✓ `--n-gpu-layers` is > 0 and consistent with the free VRAM
- ✓ co-running: `--n-gpu-layers` is *lower* than in LLM-only mode (the 2048 MB
  mining reserve is being applied, not ignored)

A `llama-server` with no `--main-gpu` means the planner fell through to the
unknown-placement path — that card will never appear in `servingIndices()` and
will silently report no model to the board.

```bash
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/v1/models
```

- ✓ `{"status":"ok"}`
- ✓ the model path in `/v1/models` is the expected GGUF under
  `%APPDATA%\llmjob-earn\llm\`

---

## Phase 3 — the co-run, under load

Mining and inference contending for one card is the app's whole premise and the
hardest thing to simulate.

Start in `auto`, let it settle 5 minutes, then:

- ✓ hashrate is within ~10% of mining-only for the same card (record the
  mining-only number first — this is the number users will complain about)
- ✓ `nvidia-smi` shows both processes resident, total VRAM below the card's
  capacity with headroom
- ✓ tok/s in the header is non-zero and the LLM answers in the Chat tab
- ✓ no engine restarts in the log over the 5 minutes
- ✓ the miner reached a non-zero hashrate *before* the LLM started — `applyPlan`
  waits on `waitForMinerUp` so the layer budget is sized against real
  post-mining free VRAM. If the LLM comes up first, the budgeter read the wrong
  number and the co-run will be over-committed.

---

## Phase 4 — the board round trip

**The tier CI cannot have, and the one that would have caught #136.** Every
layer in isolation was correct there; only the end-to-end assertion failed.

Wait one report interval (60s), then read the deployed board back:

```bash
node -e "fetch('https://llmjob-production.up.railway.app/api/miners').then(r=>r.json()).then(d=>{const me=d.miners.find(h=>h.worker==='<your-worker>');console.log(JSON.stringify(me,null,1));})"
```

Assert against what the box is *actually* doing, field by field:

- ✓ `gpu` matches `nvidia-smi`
- ✓ `hash` is within a few percent of the app's display
- ✓ `vramUsedMb` / `vramTotalMb` match the driver
- ✓ `version` is **the version under test**
- ✓ `llmModel` names the served model when serving, `null` when mining-only
- ✓ multi-GPU: one `cards[]` entry per physical card, `worker/gpuN` suffixed,
  and no phantom bare-worker row inflating the host's VRAM

Then stop the app and confirm the row ages out of the online list rather than
lingering as a stale phantom.

Any field that is `null` here but populated locally is a whitelist or plumbing
break somewhere between the client and the column — check the controller
destructure first.

---

## Phase 5 — lifecycle and the adopt path

Windows-specific and impossible to exercise in CI, because it depends on the OS
holding a port after process death.

1. **Stop / start** from the UI. ✓ both processes die; ✓ restart brings both
   back; ✓ no orphaned `llama-server` or `alpha-miner` left behind
   (`Get-Process` after stop).
2. **Adopt.** Stop the app but leave a healthy `llama-server` on 8080, then
   start the app. ✓ log reads `local LLM already running on … — reusing it`;
   ✓ no second `llama-server` spawns (double-loading the model is an OOM).
   ⚠ Known consequence: an adopted instance has no GPU index, so
   `servingIndices()` drops it and the board shows **no model** for that run.
   Confirm that is still the intended trade-off before shipping.
3. **Port contention.** Occupy 8080 with something that is *not* llama-server
   (`python -m http.server 8080`). ✓ the app walks to 8081+ rather than adopting
   the impostor — `probeLlmHealth` requires llama-server's own health body.
4. **Kill the engine externally** (`Stop-Process`). ✓ the UI reflects the stop
   rather than showing a live hashrate for a dead process.
5. **Settings persistence.** Change worker/region/mode, restart, ✓ they survive
   in `%APPDATA%\llmjob-earn\settings.json` and repopulate the UI.

---

## Phase 6 — mode matrix

Four modes, each with mining on/off and the LLM on/off. CI covers
`resolvePlan`; what it cannot cover is whether the *processes* match the plan.

| Mode | Expect |
|---|---|
| `mining` | engine only; no `llama-server`; board row with `llmModel: null` |
| `llm` | `llama-server` only; **no board row at all** (reporting lives in `startMining`) |
| `both` / `auto` | both processes; board row with the model named |

The `llm`-only row is worth confirming explicitly — a user serving the network
but not mining is invisible to the board by design. Decide whether that is
acceptable for this release rather than discovering it from a user.

Also: `llm` mode with the VRAM gate refusing → ✓ the UI returns to a stopped
state rather than showing STOP for a session running nothing.

---

## Phase 7 — node linking and cluster serving

Needs a real account and a real pairing token; CI stubs the whole exchange.

1. Dashboard → Add node → copy token → Connect in the app.
   ✓ the node appears online in the dashboard; ✓ only the public key left the box.
2. With the LLM ready and the node linked: ✓ log reads
   `serving cluster jobs for the LLMJob network`.
3. Send a job through the network model from the web chat.
   ✓ it routes to this box; ✓ a reply streams back; ✓ `activeJobs()` moves.
4. Disconnect. ✓ workers stop; ✓ the LLM keeps serving locally (chat still works).

---

## Phase 8 — the packaged artifact

Everything above tested `npm start` against source. Users get an installer, and
the two differ in ways that have bitten this app before.

```bash
npm run dist:win
```

Install the NSIS output on a box **without** the dev toolchain — this is the
whole point of the phase, and a machine with Visual Studio installed will hide
the failure.

- ✓ installs and launches
- ✓ the bundled engine under `resources/engine` is used — no download on first
  run (watch the log)
- ✓ the VC++ runtime DLLs land next to `llama-server.exe`; a missing
  redistributable kills it with `STATUS_DLL_NOT_FOUND` before it logs anything
- ✓ re-run phases 2 and 4 against the installed build — packaged path
  resolution (`process.resourcesPath`) is a genuinely different code path
- ✓ `app:version` matches the release version, and the board's `version` field
  agrees
- ✓ uninstall leaves no running processes

**Auto-update.** Install the *previous* release, publish this one, ✓ the update
is detected, ✓ "Update & restart" relaunches into the new version, ✓ the
adopt path (phase 5.2) behaves during the relaunch — that is exactly the window
it was written for.

---

## Phase 9 — the other entry points

Both ship from this repo and neither is exercised by launching the GUI.

```bash
npm run start:cli -- --help
npm run dist:cli
npm run dist:hiveos
```

- ✓ the CLI mines, serves, and reports to the board with the same worker
  semantics as the GUI (it has its own copy of the report wiring in
  `earn-cli.js` — the two have drifted before)
- ✓ the SEA binary runs on a box without Node installed
- ✓ the HiveOS package produces stats the rig UI can read

---

## Known limitation of this box

**This machine has one GPU.** The headline feature of the current cycle —
`LlmFleet` running one `llama-server` per eligible card — cannot be validated
end-to-end here. A single 4090 exercises exactly one instance, which is the old
behaviour with new code around it.

Phases 2, 4 and 6 will pass on a single card and still tell you nothing about:

- per-instance port walking (`findFreePort` stepping 8080 → 8081 → …)
- `worker/gpuN` row splitting and the bare-row drop on the board
- partial readiness — one card up, one still loading
- one instance dying while others keep serving
- summed `activeJobs()` across several workers

Before shipping a release that changes fleet behaviour, run phases 2, 4 and 6
on a genuine multi-GPU rig — the CLI is enough, no GUI needed. The 6-card and
13-card hosts already on the board are the realistic targets. Treat a
single-GPU pass as *necessary but not sufficient* for any fleet change.

---

## Sign-off

A release is good to ship when phases 0–8 pass on this box, phase 9 passes for
whichever entry points changed, and — for any fleet change — phases 2/4/6 pass
on a multi-GPU rig. Record the version, the date, and any phase skipped with
its reason; a skipped phase is a decision, not an omission.
