'use strict';

// ── Local LLM (llama.cpp llama-server) and the in-app chat ──────────────────
// The desktop app's local LLM: planning it on the cards, fetching and starting
// llama-server, its status row, sharing it on the network, and the Chat tab's
// stream. Moved out of main.js unchanged. The parts of main.js it reads and
// calls come through appContext.

const { app } = require('electron');
const { execFile } = require('child_process');
const path = require('path');
const fs = require('fs');

const { ctx, share, late } = require('./appContext');
const { getJson, streamChatCompletion, extractLlamaZip } = require('./io');
const { detectGpusVram } = require('./probe');
const probe = require('./probe');
const { llmEngine, makeLlmFleet, llmRunOptions, probeLlmHealth } = require('./llmLaunch');
const { createLanShare } = require('./lanShare');
const { progressThrottle } = require('../shared/progressThrottle');
const { LLM } = require('../shared/config');
const { pickLlmGpu, waitForFreeVram } = require('../shared/vram');
const { planLlmInstances, vramShortfall } = require('../shared/llmPlan');
const { allModels, pickModel, ctxLadder } = require('../shared/models');
const gpuLoss = require('../shared/gpuLoss');
const { cardForPrompt } = require('../shared/promptPause');
const { buildChatBody } = require('../shared/llmChat');
const { placeLlmPlan, describeLlmStart, parseServedModel } = require('../shared/llama');
const { runtimeCopyPlan } = require('../shared/llmRuntime');
const { readGpuIndexEnv, planMinerGpus, planAmdGpus, planIntelGpus } = require('../shared/gpu');

const send = late('send');
const loadSettings = late('loadSettings');
const loadGpuLost = late('loadGpuLost');
const saveGpuLost = late('saveGpuLost');
const readAmdCards = late('readAmdCards');
const endIfNothingRuns = late('endIfNothingRuns');
const syncWorker = late('syncWorker');
const makeJobWorker = late('makeJobWorker');

let fleet = null;               // LlmFleet (one llama-server per eligible GPU) while up

// nvidia-smi's PCI bus id per card, read when the LLM starts, to find the card
// each llama-server really loaded on (see promptPause.cardForPrompt).
let llmBusIds = {};
// And nvidia-smi's cards, read at the same time, to name a card that gives up
// on its GPU (see rememberGpuLoss).
let llmCards = [];
// How many cards mine, for the same function: a rig with no nvidia-smi (an AMD
// one) has no bus ids, and several cards are left mining. On an AMD rig that is
// the cards PEARL_GPU_INDEX names, else the AMD cards the OS lists
// (planAmdGpus) that the miner started; one AMD card is 0 or 1 here, so its
// prompts still pause it. Read with the bus ids when the LLM starts and kept
// for its life, so a prompt's start and its end map to the same card even when
// the miner starts in between.
let llmMiningCards = 0;
let llmEverReady = false;       // did any instance reach "ready" this run (vs. dying first)
let serveLogged = false;        // "serving cluster jobs" logged once per serving run
// `note` is a transient "what is it doing right now" line — "Downloading model…
// 42%", "Starting…" — shown in place of the model name until the LLM is ready.
// Without it a first run logs "preparing local LLM…" and then goes silent for
// however long a ~5 GB model takes, which reads as a hang: a user on a 2-GPU rig
// hit START seventeen times over ninety seconds while the download was in fact
// progressing normally.
// `vramMb` and `ctxSize` describe the model being served, for the API tab's
// "~N GB VRAM · NK ctx" line, which was hard-coded to Gemma's old numbers.
// `held` says, in a sentence, which model a card that kept losing its GPU is
// not running, and how to try it again (see startLlm).
let llmStatus = { ready: false, endpoint: null, webUrl: null, tokensPerSec: 0, model: LLM.model.name,
  vramMb: LLM.model.vramFullMb, ctxSize: ctxLadder(LLM.model)[0], error: null, note: null, held: null };
// Which model this machine actually loaded, once a run has chosen one. Every
// place that reports a model name reads this rather than LLM.model, which stopped
// being the answer when selection became per-node: a 5090 serving Qwen3.8 would
// otherwise tell the dashboard, the telemetry and — via metrics.model — the
// `model` field of every gateway completion that it was serving Gemma. Seeded
// with the default so a report before the first start is still truthful.
let servingModel = LLM.model;
// What the running fleet's model was chosen on: the best card, its free VRAM,
// and the models it kept losing its GPU under (see llmReplan).
let llmPlannedOn = { index: null, freeMb: null, lost: [] };
// Sharing the local LLM on the network (the API tab, #257): the
// CLI's gate on port 8000, in front of llama-server. syncLanShare() decides
// when it runs.
const lanShare = createLanShare({
  upstreamUrl: () => fleet && fleet.webUrl(),
  track: (port, low) => (fleet ? fleet.trackRequest(port, low) : () => {}),
  isLlmReady: () => !!(fleet && fleet.isReady()),
  log: (line) => send('miner:log', { level: 'info', line }),
  onChange: () => {
    llmStatus = Object.assign({}, llmStatus, { lan: lanShare.status() });
    sendLlmStatus();
  },
});

// Quote a value as a single PowerShell single-quoted literal, doubling any
// embedded single quote. Interpolating a raw path into a '...' PS string lets a
// path containing a quote break out of the quoting and inject commands; routing
// every interpolated value through this closes that.
function psQuote(value) {
  return "'" + String(value).replace(/'/g, "''") + "'";
}

// Extract a llama.cpp Windows build zip and flatten EVERY file into dest's
// directory, so llama-server.exe ends up beside all ~30 of its DLLs (a lone
// .exe can't launch — Windows resolves sibling DLLs from the exe's folder). This
// mirrors what `unzip -j` does for the Linux/macOS archives. llama.cpp's zip is
// already flat; the recursive copy also handles a build that nests the binaries.
//
// Unzips via .NET's ZipFile rather than PowerShell's Expand-Archive, which
// validates by FILE EXTENSION and accepts only `.zip`. The download lands at a
// deliberately format-neutral name (llmEngineManager's ARCHIVE_TMP,
// `llama-download.archive`, sniffed by magic bytes on POSIX), so Expand-Archive
// refused every Windows install with "'.archive' is not a supported archive file
// format" and the local LLM could never start. ZipFile reads the file, not its
// name. Add-Type is needed on Windows PowerShell 5.1 and absent-but-harmless on
// 7, hence the swallow.
function extractLlamaZipWin(zipPath, dest) {
  return new Promise((resolve, reject) => {
    const dir = path.dirname(dest);
    const tmp = path.join(dir, '_llama_unzip');
    const ps = "$ErrorActionPreference='Stop';"
      + 'try{Add-Type -AssemblyName System.IO.Compression.FileSystem}catch{};'
      + 'if(Test-Path -LiteralPath ' + psQuote(tmp) + '){Remove-Item -LiteralPath ' + psQuote(tmp) + ' -Recurse -Force};'
      + '[System.IO.Compression.ZipFile]::ExtractToDirectory(' + psQuote(zipPath) + ',' + psQuote(tmp) + ');'
      + 'Get-ChildItem -Path ' + psQuote(tmp) + ' -Recurse -File | ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination ' + psQuote(dir) + ' -Force };'
      + 'Remove-Item -LiteralPath ' + psQuote(tmp) + ' -Recurse -Force';
    execFile('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', ps], { maxBuffer: 8 * 1024 * 1024 }, (err) => {
      if (err) return reject(err);
      if (!fs.existsSync(dest)) return reject(new Error('llama-server was not found in the downloaded archive'));
      resolve(dest);
    });
  });
}

// ── Local LLM (llama.cpp llama-server), run alongside the miner ─────────────

function sendLlmStatus() { send('llm:status', llmStatus); }

// The LLM row when a card gives up on its GPU and nothing else serves, and when
// no card is left to start on (see startLlm). START LLM clears it ('llm:retry').
const GPU_LOST_ERROR = 'GPU kept failing — START LLM to retry';

// An instance gave up on its card after losing the GPU again and again
// (LlmManager 'gpu-lost'). Remember the card and the model, so the next START
// or launch at login does not load that model there again.
function rememberGpuLoss(e) {
  const keys = gpuLoss.saveKeys(e.index, llmCards, llmBusIds, e.busId);
  saveGpuLost(gpuLoss.remember(loadGpuLost(), keys, servingModel.name));
  send('miner:log', { level: 'error', line: servingModel.name + ' kept losing ' + (e.index == null ? 'the GPU' : 'GPU ' + e.index)
    + '. It won\'t be started there again until the app updates or you press START LLM on the Chat tab.' });
}

// What fetches llama-server, the model and its projector into `dir`. None of
// them is bundled: each is downloaded on first use.
function llmFiles(dir) {
  // The server ships as a folder of shared libraries + the binary, so extraction
  // must keep them together: Windows flattens the zip with PowerShell; elsewhere
  // extractLlamaZip sniffs the archive (tar.gz on Linux/macOS, so `tar
  // --strip-components=1`) and flattens it the same way. macOS needs exactly
  // that: llama-server's only LC_RPATH is `@loader_path`, so it finds its
  // dylibs — including libggml-metal — beside itself and nowhere else.
  const extract = process.platform === 'win32' ? extractLlamaZipWin : (zip, dest) => extractLlamaZip(zip, dest);
  return llmEngine(dir, extract);
}

// Reset the live-status side of llmStatus and broadcast — the single cleanup
// used by both user-initiated stops and the process's own exit.
function resetLlmStatus() {
  llmStatus = Object.assign({}, llmStatus, { ready: false, tokensPerSec: 0, note: null });
  sendLlmStatus();
}

// Set (or clear) the transient stage line and push it to the renderer.
function setLlmNote(note) {
  llmStatus = Object.assign({}, llmStatus, { note: note || null });
  sendLlmStatus();
}

// A download-progress callback that reports to both the hero line and the log.
// downloadFile fires onProgress for every chunk, which at a few hundred chunks a
// second would flood the IPC and bury the log — so emit only on a 5-point move,
// or every 2s for a slow link, plus the final 100%.
function progressReporter(label, now) {
  const show = progressThrottle({ step: 5, everyMs: 2000, now: now || Date.now });
  return (pct) => {
    // A download goes on after STOP (it can't be cancelled), and its progress
    // would bring back a note the stop cleared.
    if (!ctx.session) return;
    const p = show(pct);
    if (p === null) return;
    setLlmNote(label + ' ' + p + '%');
    send('miner:log', { level: 'info', line: label + ' ' + p + '%' });
  };
}

// The model for a card with `freeMb` free and `reserveMb` set aside for mining,
// without the models it kept losing its GPU under (`lost`, see rememberGpuLoss).
// The default is never left out, so it is still the fallback.
function modelFor(freeMb, reserveMb, lost) {
  const models = allModels().filter((m) => m === LLM.model || !lost.includes(m.name));
  return pickModel(freeMb, reserveMb, models);
}

// Start the local LLM: ensure binary + model, size the GPU offload to leave
// `reserveMb` free for mining, spawn llama-server, and surface its OpenAI
// endpoint. Best-effort — failures are logged, never thrown to the UI. Returns
// whether the server was actually started (callers use it to reset the UI when
// an LLM-only session ends up running nothing).
//
// `epoch` is the run's miningEpoch. A STOP or an idle pause during the awaits
// below (a first run downloads several GB) moves it, and the run then stops
// short instead of starting a model for a START that is gone, or leaving the
// LLM row on "Starting…" after a STOP. A pause queues the next run, which picks
// up from the files this one downloaded.
async function startLlm(reserveMb, epoch) {
  const stale = () => epoch !== ctx.miningEpoch;
  // A fleet already up, spawned or adopted. An adopted server has no manager, so
  // hasSpawned() alone missed it, and every idle pause and resume adopted the
  // same server again, with another cluster worker each time.
  if (fleet && (fleet.hasSpawned() || fleet.isReady())) return true;

  // One fleet drives both paths: adopt a lingering server, or spawn one instance
  // per eligible GPU. Build it once with the process/GPU factories injected.
  if (!fleet) fleet = buildFleet();

  // If a healthy llama-server is already on our port (e.g. one lingering from a
  // just-restarted session), adopt it instead of spawning a second process that
  // would double-load the model and risk an OOM. Reusing it is safe — it serves
  // the same model on the same OpenAI endpoint.
  const targetBase = 'http://' + LLM.host + ':' + LLM.port;
  const healthy = await probeLlmHealth(targetBase);
  if (stale()) return false;
  if (healthy) {
    // Ask it what it serves. It is often the 27B a 5090 was running before an
    // update, and this run has planned nothing, so servingModel still held
    // the default: the hero, the board, telemetry and every cluster job named
    // the wrong model. Its name is our alias for a catalogue model; a name we
    // don't know keeps the old label. Its window is its own, not a ladder's.
    const served = parseServedModel(await getJson(targetBase + '/v1/models', { timeout: 2000 }));
    if (stale()) return false;
    const known = served && allModels().find((m) => m.name === served.name);
    if (known) servingModel = known;
    const ctxSize = (served && served.ctxSize) || ctxLadder(servingModel)[0];
    fleet.adopt(targetBase, { ctxSize });
    llmEverReady = true;
    // `held` too: a server this run did not plan was not held to any model, and
    // an old "kept losing GPU" sentence stayed in the LLM tooltip.
    llmStatus = Object.assign({}, llmStatus, {
      ready: true, error: null, note: null, held: null, endpoint: targetBase + '/v1', webUrl: targetBase,
      model: servingModel.name, vramMb: servingModel.vramFullMb, ctxSize,
    });
    send('miner:log', { level: 'info', line: 'local LLM already running on ' + targetBase + ' — reusing it' });
    sendLlmStatus();
    syncLanShare();
    syncWorker();
    warmUpLlm(targetBase);
    return true;
  }

  // Plan one llama-server per eligible GPU: the model is small enough to hold a
  // copy on every card with room, so a multi-GPU rig serves from all of them
  // (each pinned to its card) instead of only the best card. An empty plan
  // means VRAM was measured but nothing fits — refuse, as the single-card path
  // did; when VRAM can't be read the planner returns one unknown-placement
  // instance and lets llama.cpp decide.
  const [cards, busIds] = await Promise.all([detectGpusVram(), probe.detectPciBusIds()]);
  // On an AMD rig the miner's cards come from the OS's count (planAmdGpus), as
  // in startMining. Asked only when nvidia-smi lists nothing and no card was chosen.
  const chosen = readGpuIndexEnv(process.env).indices;
  const amdCards = await readAmdCards(cards.length > 0 || chosen != null);
  if (stale()) return false;
  llmBusIds = busIds;
  llmCards = cards;
  // The AMD plan can name a card HIP doesn't have, such as an old Radeon it
  // has no code for. That card fails on its own (PearlMiner._startCores), so a
  // miner already running counts the cards that started. A rig left mining on
  // one card still pauses it for each prompt.
  const amdPlanned = planAmdGpus({ env: process.env, det: amdCards, chosen }).length
    // An Intel rig's cards are counted the same way, and fail the same way.
    || planIntelGpus({ env: process.env, det: amdCards, chosen }).length;
  const miner = ctx.miner;
  llmMiningCards = planMinerGpus(cards, chosen).length
    || (amdPlanned && miner && miner.isRunning() ? Math.min(amdPlanned, miner.cardCount()) : amdPlanned);
  // A model that kept losing the GPU on a card is not loaded there again until
  // the app updates or the user presses START LLM (see rememberGpuLoss). A card
  // that lost it even under the smallest model is left out altogether. Without
  // nvidia-smi there are no cards, and the one llama.cpp picks is index null.
  const gpuLost = loadGpuLost();
  const lostOn = (index) => gpuLoss.lostModels(gpuLost, gpuLoss.lookupKeys(index, cards, busIds));
  const usable = cards.filter((c) => !lostOn(c.index).includes(LLM.model.name));
  if (cards.length ? !usable.length : lostOn(null).includes(LLM.model.name)) {
    const why = 'not starting the local LLM: it kept losing the GPU even with ' + LLM.model.name
      + '. Press START LLM on the Chat tab to try again.';
    send('miner:log', { level: 'error', line: why });
    llmStatus = Object.assign({}, llmStatus, { ready: false, note: null, error: GPU_LOST_ERROR, held: why });
    sendLlmStatus();
    return false;
  }
  // Which model this run serves. Chosen from the best card's free VRAM, because
  // the fleet loads ONE model across every instance — planLlmInstances then drops
  // any card that cannot hold it. On a mixed rig that trades breadth for
  // capability: a box with one 32 GB card and three small ones serves the large
  // model from the one card rather than the small model from four. That is the
  // intended reading of "for cards that can support it", but it is a trade, and
  // the small-card case is unaffected because those rigs resolve to the default.
  const bestCard = pickLlmGpu(usable);
  const bestIndex = bestCard ? bestCard.index : null;
  const bestFree = bestCard ? bestCard.freeMb : null;
  const lost = lostOn(bestIndex);
  const wanted = modelFor(bestFree, reserveMb || 0, []);
  // The same choice without the models this card kept losing the GPU under:
  // the next one down, or `wanted` itself when it isn't one of them.
  const model = modelFor(bestFree, reserveMb || 0, lost);
  const held = model.name === wanted.name ? null : wanted.name + ' kept losing GPU ' + bestIndex + ', so '
    + model.name + ' runs there instead. To try it again, press STOP, then START LLM on the Chat tab.';
  if (held) send('miner:log', { level: 'warn', line: held });
  servingModel = model;
  llmPlannedOn = { index: bestIndex, freeMb: bestFree, lost };
  // The hero shows this name, and it was seeded once at startup and never
  // updated — fine while every node ran the same model, wrong the moment one
  // doesn't. Set before the first failure path below so even "Needs ~N GB free
  // VRAM" names the model it was actually short for.
  llmStatus = Object.assign({}, llmStatus, { model: model.name, vramMb: model.vramFullMb, ctxSize: ctxLadder(model)[0], held });
  // Any other card that lost the GPU under this model is left out too.
  const plan = planLlmInstances(usable.filter((c) => !lostOn(c.index).includes(model.name)), model, reserveMb || 0);
  if (!plan.length) {
    // An empty plan means VRAM was measured but no card had room.
    const { freeMb, needMb } = vramShortfall(cards, model, reserveMb);
    const needGb = Math.round(needMb / 1024);
    send('miner:log', { level: 'error', line: 'not enough free VRAM for the local LLM: ' + freeMb
      + ' MB free, need ~' + needMb + ' MB for ' + model.name + ' — skipping the LLM.' });
    llmStatus = Object.assign({}, llmStatus, { ready: false, note: null, error: 'Needs ~' + needGb + ' GB free VRAM' });
    sendLlmStatus();
    return false;
  }

  const dir = path.join(app.getPath('userData'), 'llm');
  send('miner:log', { level: 'info', line: 'preparing local LLM (' + model.name + ')…' });

  // Both of these are no-ops once cached, but on a first run they fetch ~100 MB
  // of llama-server and a ~5 GB model. Report progress so the wait is legible
  // instead of looking like a stall.
  let binaryPath, modelPath, mmprojPath = null;
  try {
    setLlmNote('Preparing…');
    const engine = llmFiles(dir);
    binaryPath = await engine.ensureServer(progressReporter('downloading llama-server…'));
    modelPath = await engine.ensureModel(progressReporter('downloading model ' + model.name + '…'), model);
    // A vision model needs its projector too. Fetched after the weights and
    // reported separately: it is ~1 GB against ~18, so folding it into the same
    // progress line would make the last 5% look like a stall.
    mmprojPath = await engine.ensureMmproj(progressReporter('downloading vision projector…'), model);
  } catch (e) {
    if (stale()) return false;
    // Surface the failure on the hero, not only in the log. Clearing the note
    // without setting an error dropped the row straight back to a grey dot and
    // the model name — identical to "not started yet" — so a download that died
    // at 57% after twenty minutes looked exactly like nothing having happened,
    // with the reason buried in the Logs tab. Same reasoning as the note itself:
    // the failing path deserves the legibility the working path just got.
    llmStatus = Object.assign({}, llmStatus, { ready: false, note: null, error: 'Setup failed — see Logs' });
    sendLlmStatus();
    send('miner:log', { level: 'error', line: 'LLM setup failed: ' + e.message });
    return false;
  }
  if (stale()) {
    // A pause's queued run takes over and sets its own note. After a STOP, the
    // download's last progress note goes.
    if (!ctx.session) setLlmNote(null);
    return false;
  }

  // Make sure the VC++ runtime DLLs sit next to llama-server.exe. llama.cpp's
  // zips don't include them; on a machine without the redistributable — or one
  // where it's only reachable via the *user* PATH, which the app relaunched by
  // the elevated updater doesn't inherit — llama-server dies instantly with
  // STATUS_DLL_NOT_FOUND before it ever logs a line. The installer bundles the
  // three DLLs (build.extraResources → <resources>/llm-runtime); co-locating
  // them beats any PATH because the exe's own dir is searched first.
  try {
    const dllPlan = runtimeCopyPlan({
      platform: process.platform, binDir: path.dirname(binaryPath),
      resourcesPath: process.resourcesPath, existsFn: fs.existsSync, joinFn: path.join,
    });
    dllPlan.forEach((c) => {
      fs.copyFileSync(c.from, c.to);
      send('miner:log', { level: 'info', line: 'installed LLM runtime DLL: ' + path.basename(c.to) });
    });
  } catch (e) {
    send('miner:log', { level: 'error', line: 'could not install the LLM runtime DLLs: ' + e.message });
  }

  // llama-server counts cards its own way, so ask it for its list and put each
  // instance on the card it was planned for (shared/llama.placeLlmPlan). After
  // the DLL copy: without them the listing can't run on Windows. With the
  // model, it reads each card's bus id too, which prompt pause needs.
  const placed = placeLlmPlan(plan, cards, await probe.detectLlmDevices(binaryPath, modelPath), busIds);
  if (stale()) {
    if (!ctx.session) setLlmNote(null);
    return false;
  }
  for (const w of placed.warnings) send('miner:log', { level: 'warn', line: w });
  send('miner:log', { level: 'info', line: describeLlmStart(placed.plan) });
  // Loading a multi-GB model off disk takes seconds even when nothing downloads,
  // and until 'ready' fires the hero is otherwise indistinguishable from idle.
  llmStatus = Object.assign({}, llmStatus, { ready: false, error: null, note: 'Starting…' });
  sendLlmStatus();
  await fleet.start(placed.plan, llmRunOptions(model, { binaryPath, modelPath, mmprojPath }));
  return true;
}

// Construct an LlmFleet wired to the GUI (llmLaunch.makeLlmFleet), building a
// cluster job-worker per ready instance. Aggregate fleet events map onto the
// same IPC the single-instance flow used.
function buildFleet() {
  llmEverReady = false;
  serveLogged = false;
  const f = makeLlmFleet({
    makeWorker: (baseUrl, index, track) => makeJobWorker(baseUrl, track),
    // An adopted server has no process of ours to watch, so it is asked.
    probeHealth: probeLlmHealth,
  });
  f.on('log', (l) => send('miner:log', l));
  f.on('first-ready', ({ baseUrl }) => warmUpLlm(baseUrl)); // one warm-up per run
  f.on('ready', ({ baseUrl }) => {
    llmEverReady = true;
    // webUrl() is the first ready instance's base URL — non-null here, since this
    // instance was just marked ready before the event fired.
    const web = f.webUrl();
    // `error: null` alongside ready, matching the adopt path: a model that is up
    // and answering must not still be wearing an error. The renderer ranks error
    // above ready in both the hero dot and its label, so a stale one outranks the
    // truth — and nothing else clears it once the fleet is running, since
    // startLlm's pre-spawn reset only runs on a fresh start.
    llmStatus = Object.assign({}, llmStatus, {
      ready: true, error: null, endpoint: web + '/v1', webUrl: web, note: null, ctxSize: f.ctxSize(),
    });
    send('miner:log', { level: 'info', line: 'local LLM ready — OpenAI endpoint ' + baseUrl + '/v1' });
    sendLlmStatus();
    syncLanShare();
    syncWorker(); // serve cluster jobs once a model is up, if we're linked
  });
  f.on('stats', ({ tokensPerSec }) => { llmStatus = Object.assign({}, llmStatus, { tokensPerSec }); sendLlmStatus(); });
  // A card's server died and is restarting. With no other card serving, the
  // row said ready all through it while chat and cluster jobs failed. It says
  // "Restarting…" until 'ready' comes back.
  f.on('crashed', () => {
    if (f.isReady()) return;
    llmStatus = Object.assign({}, llmStatus, { ready: false, note: 'Restarting…' });
    sendLlmStatus();
  });
  // A card would not load at its context size and is retrying at the next one
  // down. Show the window it will really have, not the ladder's top.
  f.on('ctx-downgrade', () => {
    llmStatus = Object.assign({}, llmStatus, { ctxSize: f.ctxSize() });
    sendLlmStatus();
  });
  f.on('busy', (e) => {
    // The app's own warm-up prompt (warmUpLlm) mines again as soon as it ends.
    const warmUp = Date.now() < warmUpUntil;
    if (warmUp && !e.busy) warmUpUntil = 0;
    const card = cardForPrompt(e, llmBusIds, llmMiningCards);
    if (card !== undefined) ctx.promptPause.setBusy(card, e.busy, warmUp ? 0 : undefined, e.low);
  });
  // An instance gave up on its card (see rememberGpuLoss), and its 'stopped'
  // follows. The row says why when no other card is still serving.
  f.on('gpu-lost', (e) => {
    rememberGpuLoss(e);
    if (!f.isReady()) {
      llmStatus = Object.assign({}, llmStatus, { error: GPU_LOST_ERROR });
      sendLlmStatus();
    }
  });
  // Logged, not swallowed: a binary that cannot be run says why only here.
  // The fleet keeps any other cards up.
  f.on('error', (e) => send('miner:log', { level: 'error', line: 'local LLM error: ' + e.message }));
  f.on('stopped', () => {
    const { miner, session } = ctx;
    // A fleet that stopped on its own is not used again: the next run builds a
    // new one, which starts its own "ready" and "serving" flags. A dead one
    // kept llmEverReady from its last run, which hid the error below, and it
    // never says 'stopped' a second time.
    // Always true today: stopLlm stops a fleet before it lets go of it, and a
    // fleet asked to stop says no 'stopped'. Kept so a late event from an old
    // fleet can never drop the new one.
    /* istanbul ignore else */
    if (fleet === f) fleet = null;
    cancelChat('the local LLM stopped');
    resetLlmStatus();
    syncLanShare();
    // Every instance died before any became ready while mining keeps running, or
    // while idle-only has it paused — the silent-failure case (typically a
    // port-bind/OOM), not a user stop.
    if (!llmEverReady && ((miner && miner.isRunning()) || (session && session.paused))) {
      llmStatus = Object.assign({}, llmStatus, { ready: false, error: 'The local LLM stopped before it was ready. See Logs.' });
      sendLlmStatus();
    }
    // An LLM-only session ends when the fleet exits, and so does one whose miner
    // never started — or the UI keeps showing a running session with nothing
    // running, and the idle watcher can bring things back behind it (see
    // runPlan).
    endIfNothingRuns(miner && miner.isRunning());
  });
  return f;
}

// Fire one tiny generation as soon as the model is ready, so the Mine tab's
// tokens/sec figure populates on its own — the user shouldn't have to open Chat
// to see the LLM is alive. Best-effort and discarded; a real request just
// overwrites the warm-up's tok/s.
//
// Mining pauses for it as for any prompt (promptPause), so the figure is the
// card's own speed, what a real prompt gets: beside the miner it read 7 tok/s
// on an RTX 4090 that answers at 140. But it starts mining again as soon as
// the warm-up ends, not 10s after like a prompt: that wait showed Pearl, and
// $/day, at 0 for ten seconds on every START. Until the warm-up ends, or
// WARM_UP_MS at most, the LLM's busy signal is taken as its own.
const WARM_UP_MS = 30000;
let warmUpUntil = 0;
function warmUpLlm(baseUrl) {
  try {
    const body = buildChatBody([{ role: 'user', content: 'Say hello.' }], { stream: true });
    body.max_tokens = 24;
    const req = streamChatCompletion(baseUrl, body, () => {});
    warmUpUntil = Date.now() + WARM_UP_MS;
    req.done.catch(() => {});
  } catch (e) { /* best effort — never blocks startup */ }
}

// The model a run that mines should serve in place of the running one, or null
// to keep it. A model planned for the card alone (Compute Mode LLM) can be
// joined by the miner with no STOP between: Auto picked in Settings, then START
// LLM. With the mining reserve the same card may get another model: the 27B
// plus the miner does not fit a 32 GB card, which serves Gemma. Judged on what the fleet was
// planned on, because a reading now counts the fleet itself, and with the same
// models left out, so a card held to Gemma (see rememberGpuLoss) keeps it.
//
// Only a fleet we spawned. An adopted server has no manager, so stopping the
// fleet would not stop it, and the next plan would adopt it again.
function llmReplan() {
  if (!fleet || !fleet.hasSpawned()) return null;
  const next = modelFor(llmPlannedOn.freeMb, LLM.miningReserveMb, llmPlannedOn.lost);
  const same = next.name === servingModel.name && ctxLadder(next)[0] === ctxLadder(servingModel)[0];
  return same ? null : next;
}

// Stop the running fleet for a fresh plan, and wait until the card is back.
// stopAndWait stops the fleet itself; stopLlm then does the rest of a STOP. A
// fleet says no 'stopped' for a stop it was asked for, so its handler neither
// ends the session nor shows an error.
async function stopLlmForReplan(needMb, epoch) {
  const exited = fleet.stopAndWait();
  stopLlm();
  await exited;
  // An exited server can still hold its VRAM for a moment (shared/vram
  // waitForFreeVram). A miner started then fails to build its core, and a plan
  // made then reads the card as full. About 8s at most, as the CLI waits; a
  // card that is really short is left for the plan to refuse. The card the
  // model was planned on, not the emptiest: on a rig with a second card, that
  // one has room at once.
  await waitForFreeVram({
    needMb,
    read: detectGpusVram,
    pick: (cards) => pickLlmGpu(cards.filter((c) => c.index === llmPlannedOn.index)),
    tries: 32,
    stillWanted: () => epoch === ctx.miningEpoch,
  });
}

function stopLlm() {
  cancelChat('the local LLM was stopped');
  // Nothing is left to answer a prompt, so every card mines now rather than
  // after the usual wait.
  ctx.promptPause.reset();
  if (fleet) { fleet.stop(); fleet = null; }
  serveLogged = false;
  resetLlmStatus();
  syncLanShare();
}

// Run the network gate while the switch is on and a model is answering, and
// only then. Called whenever either changes.
function syncLanShare() {
  const want = loadSettings().shareLlm === true && !!fleet && fleet.isReady();
  lanShare.sync(want, { name: servingModel.name, ctxSize: llmStatus.ctxSize });
}

// ── In-app chat: stream the local llama-server's OpenAI chat completions ──────
// The renderer can't hit http://127.0.0.1 under its CSP, so main proxies the
// request over the shared streamChatCompletion and relays batched deltas via
// IPC (delta → done / error). Only one turn runs at a time. Cancellation always
// settles the stream, so the renderer is guaranteed a done/error outcome — no
// stuck "streaming" state when the LLM is stopped mid-reply.
let chatStream = null;

function cancelChat(reason) {
  if (!chatStream) return;
  const s = chatStream;
  chatStream = null;
  s.cancel(reason);
}

// Grounding for the in-app chat: a small local model has no idea what "LLMJob"
// or "PPLNS" mean, so prompts like "What is LLMJob?" produce generic guesses.
// This system message gives it the facts to answer from. In-app chat only — jobs
// relayed from the cluster (jobWorker) are arbitrary API requests and get none.
const CHAT_SYSTEM_PROMPT = [
  "You are the assistant built into LLMJob Earn, running entirely on the user's own GPU via a local server — nothing they type leaves their machine. Use this context when relevant:",
  '- LLMJob turns the spare power of a GPU the user already owns into money: while the app runs, their GPU mines Pearl (PRL, a cryptocurrency the user is paid in) and can also run you, this local AI model.',
  '- The mining pool pays with PPLNS ("Pay Per Last N Shares"): when it finds a block, the reward is split across the last N shares miners submitted, so payout reflects sustained contribution rather than luck. Payouts settle about every 4 hours with a 1 PRL minimum.',
  '- This chat is private — it runs on the user\'s machine, so prompts never leave the computer.',
  'Answer conversationally and concisely. For anything unrelated to LLMJob, just answer normally.',
].join('\n');

function llmChat(messages) {
  cancelChat('superseded by a new message');
  const base = (fleet && fleet.chatUrl()) || ('http://' + LLM.host + ':' + LLM.port);
  if (!fleet || !fleet.isReady()) { send('llm:chat:error', { message: 'the local LLM is not running' }); return; }

  const grounded = [{ role: 'system', content: CHAT_SYSTEM_PROMPT }].concat(Array.isArray(messages) ? messages : []);
  const s = streamChatCompletion(base, buildChatBody(grounded, { stream: true }),
    (text) => send('llm:chat:delta', { text }));
  chatStream = s;
  s.done.then(
    () => { if (chatStream === s) chatStream = null; send('llm:chat:done', {}); },
    (e) => { if (chatStream === s) chatStream = null; send('llm:chat:error', { message: e.message }); },
  );
}

share({
  fleet: { get: () => fleet },
  llmStatus: { get: () => llmStatus },
  servingModel: { get: () => servingModel },
  serveLogged: { get: () => serveLogged, set: (v) => { serveLogged = v; } },
}, { startLlm, stopLlm, llmReplan, stopLlmForReplan });

module.exports = { syncLanShare, llmChat };
