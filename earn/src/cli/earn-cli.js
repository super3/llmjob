#!/usr/bin/env node
'use strict';

// Headless LLMJob Earn miner for Linux — the command-line counterpart to the
// Electron GUI. It shares all the real logic with the desktop app (config,
// address handling, engine download, argument building, the process supervisor
// and stats accumulator); this file is only the thin IO shell that wires the
// real filesystem / network / child_process around them, exactly like main.js
// does for the GUI. No window, no DOM — just stdout.

const { spawn } = require('child_process');
const path = require('path');
const os = require('os');
const fs = require('fs');

const { parseCliArgs, USAGE } = require('../shared/cliArgs');
const selfUpdater = require('./selfUpdater');
const { planUpdate } = require('../shared/selfUpdate');
const hiveos = require('./hiveos');
const {
  TRIAL_FAILED_EXIT, TRIAL_FAIL_MS, TRIAL_CHECK_MS, trialVerdict, startupMessage, coreLoadMessage, poolUrlNote,
  KRYPTEX_ACCOUNT_MESSAGE,
} = require('../shared/hiveos');
const { NO_JOB_MS } = require('../main/pearlMiner');
const { findWorker, makeSpawnWorker } = require('../main/xmrWorker');
const { readCpuFacts } = require('../main/cpuFacts');
const { planCpuThreads } = require('../shared/xmr/cpuPlan');
const { shortenXmrAddress } = require('../shared/xmr/address');
const { coreFactory } = require('../main/pearlCore');
const { withShapePick } = require('../shared/shapePick');
const { writeFileAtomic } = require('../shared/settingsStore');
const { extractLlamaZip } = require('../main/io');
const {
  detectRegion, detectVram, detectGpusVram, detectMinerGpus, postMinerReport,
} = require('../main/probe');
const probe = require('../main/probe');
const nodeStore = require('../main/nodeStore');
const { createNodeClient, serverOf } = require('../main/nodeClient');
const { llmEngine, makeLlmFleet, llmRunOptions } = require('../main/llmLaunch');
const { initStats, applyEvent, snapshot, offPool, createRateMeter, meterSample, meterRead } = require('../shared/miningStats');
const { minerDeps } = require('../shared/minerDeps');
const { progressThrottle } = require('../shared/progressThrottle');
const {
  NETWORK, LLM, NODE, REGIONS, resolveEndpoint, regionLabel, xmrEndpointFor, splitEndpoint,
} = require('../shared/config');
const { isKryptexHost } = require('../shared/miner/stratum');
const { defaultWorker } = require('../shared/worker');
const { hostStats, reportToBoard } = require('../shared/minerReport');
const { statsFilePayload } = require('../shared/statsFile');
const { shortenAddress, payoutProblem, isKryptexAccount } = require('../shared/address');
const { requiredFreeMb, pickLlmGpu, waitForFreeVram } = require('../shared/vram');
const { planLlmInstances, vramShortfall } = require('../shared/llmPlan');
const { allModels, pickModel, ctxLadder, planAutoMode } = require('../shared/models');
const { createPromptPause, cardForPrompt } = require('../shared/promptPause');
const {
  AMD: CORE_AMD, INTEL: CORE_INTEL, INTEL_DIR: CORE_INTEL_DIR, FILES: CORE_FILES, pickCoreVariant,
} = require('../shared/coreVariant');
const { createAutoGate, createServeGate } = require('../main/autoGate');
const { lanAddresses } = require('../main/lanShare');
const { amdWait } = require('../shared/amdWait');
const { resolvePlan, normalizeMode } = require('../shared/llmMode');
const { minerSupported, minerUnsupportedNote } = require('../shared/platform');
const { resolveServerUrl, placeLlmPlan, describeLlmStart } = require('../shared/llama');
const {
  alignCudaDeviceOrder, clearCudaVisibleDevices, describeClearedCuda, parseGpuIndexList,
  describeSkippedGpus, planMinerGpus, describeAmdPlan,
  describeIntelPlan, intelCount, planMiningCards, llmCards, pickGpu,
} = require('../shared/gpu');
const { planClocks, noClocks } = require('../shared/clockPlan');
const { CORE_ROTATION_FOLD_ENV, resolveRotationFold } = require('../shared/rotationFold');
const { describeAmdDrivers } = require('../shared/amdDriver');
const format = require('../shared/format');
const pkg = require('../../package.json');

// Number the GPUs the way nvidia-smi does, before anything opens a CUDA device.
// Everything here — the device label, per-card VRAM, temperatures, the board's
// rows — speaks nvidia-smi's indices, and the CUDA runtime does not unless told
// to. Set at load, because the mining core initialises CUDA inside THIS process
// and reads it then. For the same reason CUDA must see every card nvidia-smi
// lists; the removed value is logged when a mining run starts. Cards are chosen
// with --gpu-index instead, which HiveOS's h-config.sh fills in from this
// variable when cards are turned off there.
alignCudaDeviceOrder(process.env);
const clearedCuda = clearCudaVisibleDevices(process.env);

// The shortest gap between two mining status lines. See the miner event handler.
const MINE_LOG_MS = 1000;
// And between two CPU miner lines. Its worker reports every 2 s; a Monero
// share takes about a minute on a desktop CPU, so this is plenty.
const CPU_LOG_MS = 30000;

// Write a log line. When attached to a TTY we prefix a wall-clock time; when
// piped (systemd/journald, `docker logs`, a file) we drop it, since the log
// collector adds its own timestamp and two would just be noise.
function log(line, stream) {
  const out = stream || process.stdout;
  const prefix = out.isTTY ? '[' + format.formatLogTime(new Date()) + '] ' : '';
  out.write(prefix + line + '\n');
}

// A line with a level, from the engines, the core loader and the planners:
// 'error' and 'warn' go to stderr, anything else to stdout. The call sites used
// to decide this one by one, and a warning went to stdout from some and stderr
// from others.
function logAt(level, line) {
  log(line, level === 'error' || level === 'warn' ? process.stderr : process.stdout);
}

// The quiet window this run uses: the operator's --gate-quiet if given,
// otherwise the product default. Infinity means "never hand the card back",
// which is what --gate-quiet 0 asks for.
function gateQuietMs(settings) {
  return settings.gateQuietMs == null ? LLM.gate.quietMs : settings.gateQuietMs;
}

// The port the gate listens on: --gate-port, or the documented default.
function gatePort(settings) {
  return settings.gatePort == null ? LLM.gate.port : settings.gatePort;
}

// The URL another computer on the network uses to reach the gate, or null when
// there is none: a gate bound to loopback, or a machine with no LAN address.
// Issue #308's user typed http://<rig's IP>:8000/v1 and could not tell which
// address or port the rig wanted, so the log names it.
function lanEndpoint(settings) {
  const host = settings.gateHost;
  if (host && /^(127\.|localhost$|::1$)/.test(host)) return null;
  const addr = host && host !== '0.0.0.0' ? host : lanAddresses(os.networkInterfaces())[0];
  return addr ? 'http://' + addr + ':' + gatePort(settings) + '/v1' : null;
}

// The cards this run may mine on, and what chose them: --gpu-index first, then
// PEARL_GPU_INDEX, which takes the same list, none included. Null means every
// card; an empty list means none. A PEARL_GPU_INDEX that doesn't parse is an
// error, as the flag is: mining on a card someone meant to leave out is worse
// than not starting. (The desktop app can't refuse to start, so it logs and
// ignores such a value: shared/gpu readGpuIndexEnv.)
function gpuChoice(settings, env) {
  if (settings.gpuIndices != null) return { indices: settings.gpuIndices, from: '--gpu-index' };
  const raw = env.PEARL_GPU_INDEX;
  const parsed = parseGpuIndexList(raw);
  if (!parsed) return null;
  if (parsed.error) return { error: 'invalid PEARL_GPU_INDEX: ' + raw + ' (' + parsed.error + ')' };
  return { indices: parsed.indices, from: 'PEARL_GPU_INDEX' };
}

// nvidia-smi's cards the local LLM may use: those --gpu-index or
// PEARL_GPU_INDEX chose, else every card (shared/gpu llmCards). Sizing, the
// fleet and demand mode's waits all read the cards through here, so a second
// instance on one box keeps off the first one's card.
async function detectLlmCards(settings) {
  return llmCards(await detectGpusVram(), settings.llmGpuIndices);
}

// "2 GPUs [0, 2]"
function cardList(indices) {
  return indices.length + (indices.length === 1 ? ' GPU' : ' GPUs') + ' [' + indices.join(', ') + ']';
}

function quietLabel(settings) {
  const ms = gateQuietMs(settings);
  return ms === Infinity ? 'no amount of quiet' : Math.round(ms / 1000) + 's';
}

// Explicit `llmjob-earn-cli update` — check the latest release and, if this is
// the packaged binary, replace it in place.
async function runExplicitUpdate() {
  log('checking for updates (current v' + pkg.version + ')…');
  const release = await selfUpdater.fetchLatestRelease();
  if (!release) { log('could not reach the update server', process.stderr); return 1; }

  const plan = planUpdate({ currentVersion: pkg.version, release, platform: process.platform });
  if (!plan.updateAvailable) {
    if (plan.reason === 'up-to-date') {
      log('already up to date (v' + pkg.version + ')');
      // `update` has no flags to say whether this rig mines Monero, so it only
      // repairs a worker that is already there.
      if (plan.xmrWorkerUrl && selfUpdater.isPackaged() && selfUpdater.hasXmrWorker()) await repairXmrWorker(plan.xmrWorkerUrl);
    } else if (plan.reason === 'asset-missing') log('v' + plan.latestVersion + ' is out but has no Linux CLI binary yet', process.stderr);
    else if (plan.reason === 'unsupported-platform') log('self-update is only available for the Linux binary', process.stderr);
    else log('no newer release found', process.stderr);
    return 0;
  }

  if (!selfUpdater.isPackaged()) {
    log('v' + plan.latestVersion + ' is available (you have v' + plan.currentVersion + ').');
    log('this is running from source — update via git/npm, or download: ' + plan.downloadUrl);
    return 0;
  }

  log('updating ' + plan.currentVersion + ' → ' + plan.latestVersion + ' …');
  try {
    const exe = await selfUpdater.applyUpdate(plan, undefined, { amd: wantsAmdCore() });
    log('updated to v' + plan.latestVersion + ' (' + exe + '). Re-run to use it.');
    return 0;
  } catch (e) {
    log('update failed: ' + e.message, process.stderr);
    return 1;
  }
}

// The CPU worker beside the binary, made this release's when it is missing,
// won't run, or is from another release. Without this, a worker download
// that failed, or an install that never had one, waits for the next release.
async function repairXmrWorker(url) {
  try {
    if (await selfUpdater.syncXmrWorker(url, pkg.version)) {
      log('fetched llmjob-xmr-worker v' + pkg.version + ' (it was missing, would not run, or was from another release)');
    }
  } catch (e) {
    log('could not fetch llmjob-xmr-worker: ' + e.message, process.stderr);
  }
}

// Whether this run asked for the AMD core: --core amd, which run() copies into
// PEARL_CORE_VARIANT, or that variable itself.
function wantsAmdCore() {
  return pickCoreVariant({ env: process.env }).variant === CORE_AMD;
}

// The same for the Intel core: --core intel or PEARL_CORE_VARIANT=intel. A rig
// of Arc cards that asked for nothing gets it too, but that is found out later,
// from what the system lists (shared/gpu planIntelGpus).
function wantsIntelCore() {
  return pickCoreVariant({ env: process.env }).variant === CORE_INTEL;
}

// The AMD core beside the binary, fetched when this run asked for it and there
// is none, as from an install of the bare binary. Without this the rig would
// wait for the next release to get it.
async function repairAmdCore(url) {
  if (selfUpdater.hasAmdCore()) return;
  try {
    log('fetched ' + await selfUpdater.fetchAmdCore(url) + ' (it was missing)');
  } catch (e) {
    log('could not fetch ' + CORE_FILES[CORE_AMD] + ': ' + e.message, process.stderr);
  }
}

// Best-effort auto-update on start. Returns an exit code when it replaced and
// re-ran the binary (caller should return it), or null to keep mining.
async function maybeAutoUpdate(argv, settings) {
  if (process.env[selfUpdater.UPDATED_ENV]) return null; // already the updated child
  const release = await selfUpdater.fetchLatestRelease();
  if (!release) return null; // offline — never block mining

  const plan = planUpdate({ currentVersion: pkg.version, release, platform: process.platform });
  if (!plan.updateAvailable) {
    // Only for a rig that mines Monero: a Pearl-only one gets no coin miner
    // it didn't ask for.
    if (plan.xmrWorkerUrl && settings.xmrAddress && selfUpdater.isPackaged()) await repairXmrWorker(plan.xmrWorkerUrl);
    if (plan.coreHipUrl && wantsAmdCore() && selfUpdater.isPackaged()) await repairAmdCore(plan.coreHipUrl);
    return null;
  }

  if (!selfUpdater.isPackaged()) {
    log('a newer release is available: v' + plan.latestVersion + ' (run "llmjob-earn-cli update")');
    return null;
  }

  // Exit 0 after the swap: HiveOS's miner loop starts the new h-run.sh, and the
  // new CLI with it, three seconds later.
  if (hiveos.isHiveos(process.env)) return (await updateHivePackage(plan)) ? 0 : null;

  log('updating ' + plan.currentVersion + ' → ' + plan.latestVersion + ' before starting…');
  try {
    await selfUpdater.applyUpdate(plan, undefined, { xmr: !!settings.xmrAddress, amd: wantsAmdCore() });
    log('updated to v' + plan.latestVersion + '; restarting');
    return selfUpdater.reexec(argv);
  } catch (e) {
    log('auto-update failed (' + e.message + '); continuing on v' + pkg.version, process.stderr);
    return null;
  }
}

// Under HiveOS the CLI replaces its whole package folder -- binary, cores and
// hook scripts -- because HiveOS installs from the flight sheet's URL only once.
// True when the new package is in place and this CLI should exit. See
// shared/hiveos for the trial the new version then runs on.
async function updateHivePackage(plan) {
  const dir = hiveos.packageDir();
  if (hiveos.isSkipped(dir, plan.latestVersion, Date.now())) {
    log('v' + plan.latestVersion + ' is out, but it failed on this rig in the last day; staying on v' + pkg.version);
    return false;
  }
  if (!plan.packageUrl) {
    log('v' + plan.latestVersion + ' is out, but its HiveOS package is not up yet; staying on v' + pkg.version);
    return false;
  }
  log('updating the HiveOS package ' + plan.currentVersion + ' → ' + plan.latestVersion + '…');
  try {
    await hiveos.updatePackage(plan);
  } catch (e) {
    log('update failed: ' + e.message + '; staying on v' + pkg.version, process.stderr);
    return false;
  }
  log('updated to v' + plan.latestVersion + '; HiveOS starts it in a few seconds');
  return true;
}

// The check a HiveOS rig makes every few hours while it runs: nothing else
// restarts the CLI for weeks at a time. True when it swapped in a new package.
async function checkHiveUpdate() {
  const release = await selfUpdater.fetchLatestRelease();
  if (!release) return false;
  const plan = planUpdate({ currentVersion: pkg.version, release, platform: process.platform });
  if (!plan.updateAvailable || !selfUpdater.isPackaged()) return false;
  return updateHivePackage(plan);
}

// Where the CLI caches the local-LLM binary + model (mirrors the engine dir).
function llmDir() {
  return path.join(os.homedir(), '.local', 'share', 'llmjob-earn', 'llm');
}

// Download progress. In the foreground it redraws one line. Beside a running
// miner, whose own line goes out every second, it logs a line per 10% instead,
// so the two don't write over each other.
function downloadProgress(label, background) {
  if (!background) {
    return {
      onPct: (pct) => { if (pct != null) process.stdout.write('\r  ' + label + ' ' + pct + '%   '); },
      done: () => process.stdout.write('\n'),
    };
  }
  const show = progressThrottle({ step: 10, round: true });
  return {
    onPct: (pct) => {
      const step = show(pct);
      if (step !== null) log(label + ' ' + step + '%');
    },
    done: () => {},
  };
}

// Resolve the llama-server binary for the local LLM. An explicit --llm-binary
// wins; otherwise fall back to a previously installed one in the cache dir, and
// only then download the llama.cpp release and unpack it (io.extractLlamaZip:
// tar for Linux's .tar.gz). If that fails, the error names the tool that failed
// and points at --llm-binary as the escape hatch.
async function resolveLlmBinary(settings, dir, background) {
  if (settings.llmBinary) {
    if (!fs.existsSync(settings.llmBinary)) {
      throw new Error('llama-server binary not found: ' + settings.llmBinary);
    }
    return settings.llmBinary;
  }
  const engine = llmEngine(dir, (zip, dest) => extractLlamaZip(zip, dest));
  if (engine.isServerInstalled()) {
    log('LLM server found: ' + engine.serverBinaryPath());
    // Through ensureServer, not straight to the path: it puts the execute bit
    // back on a cached binary that lost it. Returning the path skipped that,
    // and such a binary failed with EACCES on every start.
    return engine.ensureServer();
  }
  log('downloading llama-server from ' + resolveServerUrl(process.platform, process.arch) + ' …');
  const progress = downloadProgress('downloading…', background);
  try {
    return await engine.ensureServer(progress.onPct);
  } catch (e) {
    throw new Error(e.message + ' — pass --llm-binary </path/to/llama-server> instead');
  } finally {
    progress.done();
  }
}

// Resolve the GGUF model path. An explicit --llm-model wins; otherwise reuse a
// cached download or fetch the small default model (a plain file, so this works
// on the CLI without zip extraction).
// `model` is the tier the caller chose (shared/models.pickModel); it is always
// passed, so there is deliberately no `|| LLM.model` default here — a silent
// fallback would be indistinguishable from the selection failing.
async function resolveLlmModel(settings, dir, model, background) {
  const m = model;
  // An explicit --llm-model is the operator's own file: we cannot know whether
  // it ships a projector, so it is served exactly as given, text-only.
  if (settings.llmModel) {
    if (!fs.existsSync(settings.llmModel)) {
      throw new Error('LLM model not found: ' + settings.llmModel);
    }
    return { modelPath: settings.llmModel, mmprojPath: null };
  }
  const engine = llmEngine(dir);
  let modelPath;
  if (engine.isModelInstalled(m)) {
    modelPath = engine.modelPath(m);
    log('LLM model found: ' + modelPath);
  } else {
    log('downloading LLM model (' + m.name + ') …');
    const progress = downloadProgress('downloading model…', background);
    modelPath = await engine.ensureModel(progress.onPct, m);
    progress.done();
  }
  // The vision projector, for a model that ships one. Separate from the weights
  // so a node that already has 17 GB on disk can pick up a ~1 GB projector
  // without re-downloading them; null for a text-only model.
  let mmprojPath = null;
  if (!engine.isMmprojInstalled(m)) {
    log('downloading vision projector …');
    const progress = downloadProgress('downloading projector…', background);
    mmprojPath = await engine.ensureMmproj(progress.onPct, m);
    progress.done();
  } else {
    mmprojPath = engine.mmprojPath(m);
  }
  return { modelPath, mmprojPath };
}

// ── Cluster serving state (workers + keep-alive pings while the LLM is up) ───
// Serving and pinging must live in the SAME process: only /api/nodes/ping
// updates last_seen server-side, so a worker that polls jobs without pinging
// gets marked offline (15 min) and eventually deleted — silently starving it.
// The fleet owns one worker per serving GPU; we track it here for telemetry and
// teardown, plus the single keep-alive ping shared across the whole fleet.
let serveFleet = null;
let servePinger = null;
// `model` is which model this run actually loaded, not the fleet default: with
// per-node selection those stopped being the same thing, and telemetry read the
// default — so a 5090 serving Qwen3.8 reported Gemma to the network board, and
// metrics.model put the same wrong name in the `model` field of every gateway
// completion it served. Seeded with the default, replaced once startLlm picks.
let serveLlmState = { ready: false, tps: 0, model: LLM.model };
// No model is up: nothing ready, no rate. `model` stays, because it names what
// this node serves, and the last telemetry ping should still say so.
function resetServeLlmState() {
  serveLlmState = { ready: false, tps: 0, model: serveLlmState.model };
}
// This machine's node id while it is armed to serve cluster jobs — reported on the
// miner ping so the network board can tell "running the model" from "serving the
// cluster". Null when not serving (mining only, or --no-serve).
let serveNodeId = null;

// Register, ping, join, telemetry and the job workers: nodeClient, shared with
// the GUI.
const nodeClient = createNodeClient({
  log: logAt,
  linkHint: 'run "connect" to attach it to your account',
});

// One verbose ping, for the standalone `connect` loop.
async function pingServer(node, telemetry) {
  const res = await nodeClient.ping(node, telemetry);
  if (res.error) log('✗ ping error: ' + res.error, process.stderr);
  else if (res.status === 200) log('✓ ping');
  else log('✗ ping failed (HTTP ' + res.status + ')', process.stderr);
}

// Sparse telemetry for the standalone `connect` loop: just device + VRAM (+ the
// node's name so renames propagate). model/capabilities/tps are omitted —
// sending nulls would wipe what a serving process last reported.
async function sparseTelemetry(node) {
  const t = {};
  try {
    const vram = await detectVram();
    if (vram) { t.vramTotal = vram.totalMb; t.vramUsed = vram.usedMb; }
  } catch (e) { /* ignore */ }
  const device = await nodeClient.deviceName();
  if (device) t.device = device;
  if (node.name) t.name = node.name;
  return t;
}

// Full telemetry for the serving run: live model/readiness/tok-s/active jobs.
// `demandJobs` are the jobs running on demand mode's own worker, which no fleet
// owns: a demand wake starts its fleet without workers, so counting the fleet
// alone reported 0 active jobs all through a demand node's work.
async function fullTelemetry(node, demandJobs = 0) {
  return nodeClient.telemetry({
    model: serveLlmState.model, ready: serveLlmState.ready, tokensPerSec: serveLlmState.tps,
    activeJobs: (serveFleet ? serveFleet.activeJobs() : 0) + demandJobs,
    name: node.name,
  });
}

function stopServe() {
  if (serveFleet) { serveFleet.syncWorkers(false); serveFleet = null; }
  if (servePinger) { clearInterval(servePinger); servePinger = null; }
  serveNodeId = null;
}

// Node identity and its server registration, resolved ONCE per process.
//
// This used to live inside startLlm, which was fine while a fleet was the only
// thing that ever served. Demand mode needs an identity before any model is
// loaded -- it polls for cluster work while mining -- and resolving it twice
// would register the node twice and log it twice.
//
// Never throws. Minting an identity writes node.json, and a disk that refuses
// it used to end demand and llm runs with 'fatal'. The LLM runs locally then,
// for the rest of the run.
let serveIdentity = null;
async function resolveServeIdentity(settings) {
  if (serveIdentity) return serveIdentity;
  // Reuse the stored identity when there is one (it carries the node's name and
  // server URL); only mint a keypair when serving and none exists yet.
  let nodeCfg;
  try {
    nodeCfg = loadNodeConfig() || (settings.serve ? getOrCreateNodeConfig() : null);
  } catch (e) {
    log('could not set up this node\'s identity (' + e.message + ') — running the LLM locally only', process.stderr);
    serveIdentity = { nodeCfg: null, canServe: false };
    return serveIdentity;
  }
  const canServe = !!(settings.serve && nodeCfg);
  // Publish the node id on the miner ping only while actually armed to serve.
  serveNodeId = canServe ? nodeCfg.nodeId : null;

  // An unclaimed node needs a row server-side before /jobs/poll will answer it.
  // Best-effort: a failure here just means no jobs arrive until a keep-alive
  // ping's retry lands, and the local model still runs.
  if (canServe) await nodeClient.enrol(nodeCfg);
  serveIdentity = { nodeCfg, canServe };
  return serveIdentity;
}

// A cluster job-worker for the llama-server at `baseUrl`: one per serving GPU,
// built by the fleet, or demand mode's own. `baseUrl` may be a thunk: in demand
// mode no server exists when the worker is built, and the port is not known
// until one is spawned. `gate`, when given, makes this worker WAKE the model
// rather than assume one is loaded.
function makeCliJobWorker(nodeCfg, baseUrl, gate, track) {
  return nodeClient.makeJobWorker(nodeCfg, { baseUrl, gate, servingModel: () => serveLlmState.model, track });
}

// Start the local LLM (llama.cpp llama-server) alongside — or instead of — the
// miner. Plans one server per eligible GPU (the model is small enough to hold a
// copy on every card with room), sizes each card's offload from its free VRAM
// (keeping `reserveMb` free for mining), spawns the fleet, and logs its
// OpenAI-compatible endpoint. Returns the LlmFleet, or null if setup failed
// (best-effort — a failing LLM never takes the miner down).
// modelOverride pins the tier. Demand mode has already chosen one, and choosing
// again here would re-run that decision against whatever VRAM happens to be free
// at the moment of the switch -- see waitForCardRoom for why that is not the same
// number a moment later.
// armServe=false leaves cluster polling AND the keep-alive ping to the caller.
// Demand mode owns both: it has to poll and stay online while no fleet exists,
// and a fleet arming its own on every wake would double-poll, race for the same
// jobs, and leak a ping timer per cycle.
async function startLlm(settings, reserveMb, modelOverride, armServe = true) {
  const planned = await planLlm(settings, reserveMb, modelOverride);
  return planned ? launchLlm(settings, planned, armServe) : null;
}

// The models this node may pick from: null (the whole catalogue) unless the
// operator passed their own file with --llm-model. That file is served with the
// settings of the catalogue entry it IS, matched by file name, or else the
// default model's. Picking by VRAM alone handed a 32 GB card the 27B tier, so
// an operator's own Gemma copy started with the 27B's context, its MTP
// speculation flags and its name, which the server routes jobs on.
function candidateModels(settings) {
  if (!settings.llmModel) return null;
  const file = path.basename(settings.llmModel);
  return allModels().filter((m) => m === LLM.model || m.file === file);
}

// The VRAM half of startLlm: which model, on which cards. Reads the cards and
// downloads nothing, so co-running auto can make this call before the miner
// claims its share of the card and leave the slow half for later.
async function planLlm(settings, reserveMb, modelOverride) {
  // Plan one llama-server per eligible GPU before doing anything expensive
  // (downloading a ~5 GB model). Each instance runs --split-mode none and is
  // pinned to its card (--device, or --main-gpu), so the per-card free VRAM is
  // what sizes and gates it — never the rig's summed total (the model can't
  // span cards, and sizing against the sum would cram it onto device 0 and
  // OOM). An empty plan means VRAM was measured but no card had room — refuse.
  // When VRAM can't be read (non-NVIDIA / no driver) the planner returns one
  // unknown-placement instance and lets llama.cpp decide.
  // Only the cards --gpu-index chose, when it chose some: the model is sized
  // for those and planned on those. Every card still goes to the placement
  // (launchLlm), which matches nvidia-smi's cards to llama-server's and pairs
  // cards of the same name in order, so it needs to see them all.
  const allCards = await detectGpusVram();
  const cards = llmCards(allCards, settings.llmGpuIndices);
  // Which model this run serves, from the best card's free VRAM. The headless
  // shell has to make the same choice the GUI does, or a large card running
  // under systemd silently keeps serving the small default.
  const bestCard = pickLlmGpu(cards);
  let model = modelOverride
    || pickModel(bestCard ? bestCard.freeMb : null, reserveMb || 0, candidateModels(settings));
  // The operator's own copy of a larger model that does not fit here. pickModel
  // falls back to the default, and the file would start under the default's
  // name and settings: the 27B's weights taking Gemma's jobs, without the 27B's
  // flags. Skipped instead, like any model that does not fit. With no VRAM
  // reading there is nothing to check, so the file is served as what it is.
  const asked = settings.llmModel
    ? allModels().find((m) => m.file === path.basename(settings.llmModel)) : null;
  if (asked && model.name !== asked.name) {
    if (bestCard) {
      log('not enough free VRAM on any single GPU for --llm-model: ' + bestCard.freeMb + ' MB free on GPU '
        + bestCard.index + ', need ~' + requiredFreeMb(asked, reserveMb) + ' MB for ' + asked.name
        + ' — skipping the LLM.', process.stderr);
      return null;
    }
    model = asked;
  }
  serveLlmState.model = model;
  const plan = planLlmInstances(cards, model, reserveMb || 0, {
    maxInstances: settings.llmMaxInstances,
  });
  // Say so when the operator's cap bit — a silently smaller fleet is
  // indistinguishable from "this rig only had one card with room".
  if (plan.length && plan.length < cards.length && settings.llmMaxInstances != null) {
    log('serving on ' + plan.length + ' of ' + cards.length
      + ' GPUs — capped by --llm-max-instances ' + settings.llmMaxInstances);
  }
  if (!plan.length) {
    // An empty plan means at least one card parsed but none fit.
    const gpu = vramShortfall(cards, model, reserveMb);
    log('not enough free VRAM on any single GPU for the local LLM: ' + gpu.freeMb
      + ' MB free on GPU ' + gpu.index + ', need ~' + gpu.needMb
      + ' MB for ' + model.name + ' — skipping the LLM.', process.stderr);
    return null;
  }
  return { model, plan, cards: allCards };
}

// The slow half: fetch llama-server and the model if they aren't cached yet,
// then spawn one server per planned card. `background` is set when the miner is
// already running beside it (co-running auto), which only changes how download
// progress is printed. `onBusy` hears each card start and finish a prompt
// ({ index, busy }); it is attached before any card can take one.
async function launchLlm(settings, planned, armServe, background, onBusy) {
  const dir = llmDir();
  const { model, cards } = planned;
  log('preparing local LLM (' + model.name + ') …');

  let binaryPath, modelPath, mmprojPath = null;
  try {
    binaryPath = await resolveLlmBinary(settings, dir, background);
    ({ modelPath, mmprojPath } = await resolveLlmModel(settings, dir, model, background));
  } catch (e) {
    log('LLM setup failed: ' + e.message, process.stderr);
    return null;
  }

  // llama-server counts cards its own way, so ask it for its list and put each
  // instance on the card it was planned for (shared/llama.placeLlmPlan). With
  // the model, it reads each card's bus id too, which prompt pause needs.
  const [devices, busIds] = await Promise.all([
    probe.detectLlmDevices(binaryPath, modelPath), probe.detectPciBusIds(),
  ]);
  // A --llm-binary built for another GPU vendor runs but sees no card: a rig
  // swapped from an RTX 5090 to an RX 9070 XT kept its CUDA build (6 Nov).
  if (settings.llmBinary && !devices.length) {
    log('LLM: --llm-binary ' + settings.llmBinary + ' lists no GPU (llama-server --list-devices), or would not run. '
      + 'It may be built for another GPU vendor; without --llm-binary the CLI downloads a Vulkan build that runs on '
      + 'NVIDIA, AMD and Intel', process.stderr);
  }
  const placed = placeLlmPlan(planned.plan, cards, devices, busIds);
  for (const w of placed.warnings) log(w);
  const plan = placed.plan;

  // Serve cluster jobs once a model is up — by DEFAULT, account or not. A rig
  // that can run the model is useful to the network whether or not anyone has
  // linked it, so an unlinked box self-registers (signature only) and takes
  // public work. The server hands an unclaimed node non-private jobs only, so
  // "unlinked" costs it access to private queues, nothing else. --no-serve opts
  // out entirely and keeps the model purely local.
  // Reuse the stored identity when there is one (it carries the node's name and
  // server URL); only mint a keypair when serving and none exists yet.
  const { nodeCfg, canServe } = await resolveServeIdentity(settings);

  // One fleet spawns a llama-server per plan entry (llmLaunch.makeLlmFleet, as
  // the GUI does) and builds a cluster worker per ready card.
  const fleet = makeLlmFleet({ makeWorker: (baseUrl, index, track) => makeCliJobWorker(nodeCfg, baseUrl, undefined, track) });
  serveFleet = fleet;
  fleet.on('log', (l) => logAt(l.level, l.line));
  fleet.on('ready', ({ baseUrl }) => {
    serveLlmState.ready = true;
    log('🧠 local LLM ready — OpenAI endpoint ' + baseUrl + '/v1');
  });
  // A card's server died and is restarting. Telemetry and the stats file said
  // ready all through it; now they say so only while another card serves, and
  // 'ready' above puts it back when the restart lands.
  fleet.on('crashed', () => { serveLlmState.ready = fleet.isReady(); });
  // One keep-alive ping loop for the whole fleet, started on the first ready
  // card: pings ride along with serving so the node stays online on the
  // dashboard (and never gets pruned) while it works.
  fleet.on('first-ready', () => {
    if (!canServe || !armServe) return;
    const pingFull = async () => nodeClient.ping(nodeCfg, await fullTelemetry(nodeCfg));
    pingFull();
    servePinger = setInterval(pingFull, NODE.pingIntervalMs);
    servePinger.unref();
    log('serving cluster jobs for the LLMJob network');
  });
  fleet.on('stats', ({ tokensPerSec, promptTokensPerSec }) => {
    serveLlmState.tps = Number(tokensPerSec) || 0;
    serveLlmState.promptTps = Number(promptTokensPerSec) || 0;
    // Only the generation rate is logged. Prefill is an order of magnitude
    // higher, so printing both under one label read like the rig had sped up.
    log('🧠 ' + Number(serveLlmState.tps).toFixed(1) + ' tok/s');
  });
  fleet.on('error', (err) => log('LLM error: ' + err.message, process.stderr));
  if (onBusy) fleet.on('busy', onBusy);

  fleet.syncWorkers(canServe && armServe); // arm serving before instances come up
  // The same run options the GUI passes (llmLaunch.llmRunOptions).
  await fleet.start(plan, llmRunOptions(model, { binaryPath, modelPath, mmprojPath }));
  log(describeLlmStart(plan));
  return fleet;
}

// Demand mode's model, fetched while the miner runs. Without this the first
// request stopped the miner and then downloaded the model (about 30 GB for the
// 5090 tier) with the card idle. Never throws: a failed download is logged and
// the wake tries again, as it always has.
async function prefetchLlm(settings, model) {
  const dir = llmDir();
  try {
    await resolveLlmBinary(settings, dir, true);
    await resolveLlmModel(settings, dir, model, true);
    log('auto:       ' + model.name + ' is on disk; a request now waits only for it to load');
  } catch (e) {
    log('LLM download failed: ' + e.message + ' — it will be tried again when a request arrives', process.stderr);
  }
}

// Wait for the emptiest GPU this run may use (detectLlmCards) to actually
// report `needMb` free, after the miner stopped (shared/vram waitForFreeVram
// has why). 8s is far more than the release needs, and short enough that a
// card which genuinely has no room fails fast instead of stalling the first
// request behind a doomed wait.
function waitForCardRoom(settings, needMb, timeoutMs = 8000) {
  return waitForFreeVram({ needMb, read: () => detectLlmCards(settings), pick: pickLlmGpu, timeoutMs });
}

// ── Connect with LLMJob (node pairing + ping) ────────────────────────────────
// Replaces the old install.sh agent: link this headless box to an account with a
// pairing token, then ping so it shows online. The identity lives in the shared
// nodeStore — the SAME node.json the GUI uses, so one machine keeps one nodeId
// across shells. Only the public key ever leaves the machine.

const { loadNode: loadNodeConfig, saveNode: saveNodeConfig, getOrCreateNode: getOrCreateNodeConfig } = nodeStore;

// Flag parse for the `connect` subcommand (--token/-t, --name/-n, --server).
// Kept local to the shell, but strict like shared/cliArgs: unknown flags and
// missing values are reported instead of silently ignored (a typo'd --token
// must not fall through to a misleading "no pairing token yet").
function parseConnectArgs(argv) {
  const opts = { token: null, name: null, server: null, errors: [] };
  for (let i = 0; i < argv.length; i++) {
    const tok = String(argv[i]);
    const eq = tok.indexOf('=');
    const flag = eq !== -1 ? tok.slice(0, eq) : tok;
    const inline = eq !== -1 ? tok.slice(eq + 1) : null;
    const value = () => {
      if (inline != null) return inline;
      const next = i + 1 < argv.length ? String(argv[i + 1]) : null;
      if (next == null || next.startsWith('-')) { opts.errors.push('missing value for ' + flag); return null; }
      i++;
      return next;
    };
    if (flag === '--token' || flag === '-t') opts.token = value();
    else if (flag === '--name' || flag === '-n') opts.name = value();
    else if (flag === '--server') opts.server = value();
    else opts.errors.push('unknown option: ' + tok);
  }
  return opts;
}

async function runConnect(argv) {
  const opts = parseConnectArgs(argv);
  if (opts.errors.length) {
    for (const e of opts.errors) log('error: ' + e, process.stderr);
    log('usage: llmjob-earn-cli connect --token <pairing-token> [--name <rig>] [--server <url>]', process.stderr);
    return 1;
  }
  const node = getOrCreateNodeConfig();
  if (opts.server && node.serverUrl !== opts.server) { node.serverUrl = opts.server; saveNodeConfig(node); }
  const base = serverOf(node);
  const name = (opts.name && opts.name.trim()) || node.name || defaultWorker();

  log('LLMJob node ' + node.nodeId + ' → ' + base);

  if (opts.token) {
    const res = await nodeClient.join(node, String(opts.token).trim(), name);
    if (res.unreachable) {
      log('could not reach ' + base + ': ' + res.error, process.stderr);
      return 1;
    }
    if (!res.ok) {
      log('join failed (HTTP ' + res.status + '): ' + (res.message || res.raw || ''), process.stderr);
      return 1;
    }
    // The server decides the enrolled id — see nodeProto.adoptedNodeId.
    node.nodeId = res.nodeId;
    node.name = name; node.connected = true; node.user = res.user; saveNodeConfig(node);
    log('✓ linked' + (res.user ? ' to ' + res.user + '’s account' : ' to your account') + ' as ' + name);
  } else if (!node.connected) {
    log('no pairing token yet — run:  llmjob-earn-cli connect --token <token>', process.stderr);
    log('copy your token from the dashboard: ' + NODE.dashboardUrl, process.stderr);
    return 1;
  } else {
    log('resuming pings for ' + (node.name || node.nodeId));
  }

  // Foreground keep-alive loop. Telemetry is SPARSE (device + VRAM + name only):
  // this process isn't the one serving inference, so sending model/capabilities/
  // tps here would overwrite what a serving run last reported with nulls.
  const pingOnce = async () => pingServer(node, await sparseTelemetry(node));

  await pingOnce();
  const timer = setInterval(pingOnce, NODE.pingIntervalMs);
  return new Promise((resolve) => {
    const shutdown = () => { clearInterval(timer); log('stopped pinging'); resolve(0); };
    process.on('SIGINT', shutdown);
    process.on('SIGTERM', shutdown);
  });
}

// Monero on the CPU, when --xmr-address asks for it: the thread plan (said in
// the log) and the engine, or null when it won't run here. It mines beside the
// GPU miner whatever the mode, and stops with it.
function setUpCpuMining(settings, parts) {
  // --xmr-pool, else the HeroMiners host for the Pearl region.
  const endpoint = settings.xmrEndpoint || xmrEndpointFor(settings.region);
  log('cpu:        Monero to ' + shortenXmrAddress(settings.xmrAddress) + ' · ' + endpoint
    + (settings.xmrEndpoint ? '  (--xmr-pool)' : ''));
  if (settings.cpuThreads === 0) {
    log('cpu:        off (--cpu-threads 0)');
    return null;
  }
  const plan = planCpuThreads(readCpuFacts({ fs, os }), settings.cpuThreads);
  if (!plan.threads) {
    log('cpu:        off: ' + plan.reason);
    return null;
  }
  const workerPath = findWorker({ execPath: process.execPath, env: process.env, exists: fs.existsSync });
  if (!workerPath) {
    log('cpu:        off: llmjob-xmr-worker is not beside the executable. Keep it from the release next to it,'
      + ' or set LLMJOB_XMR_WORKER=/path/to/llmjob-xmr-worker. From a source checkout, build it with'
      + ' earn/scripts/build-xmr-worker.sh.', process.stderr);
    return null;
  }
  const auto = settings.cpuThreads ? '' : 'auto';
  const why = plan.reason ? (auto && auto + ': ') + plan.reason : auto;
  log('cpu:        ' + plan.threads + ' of ' + plan.available + ' CPUs' + (why ? '  (' + why + ')' : ''));
  const engine = parts.xmrEngine({ spawnWorker: makeSpawnWorker(spawn), workerPath });
  return { engine, threads: plan.threads, endpoint };
}

// What a mining run's rig is, worked out and logged before anything starts:
// the knobs the user didn't pin (worker, region, card name), which cards get a
// core (shared/gpu planMiningCards), the pool, and the lines that say so.
// Returns { endpoint, amdGpus, intelGpus, intelCards, amdListed }:
//   amdGpus, intelGpus  planAmdGpus' and planIntelGpus' cards
//   intelCards  the Intel Arc cards the system lists on a rig with no NVIDIA or AMD card
//   amdListed   the AMD cards the OS lists, for the core's error when HIP lists none
// Sets settings.worker, region, gpu, gpus and gpuCount as it goes.
async function prepareMiningRig(settings, gpuPick, hive) {
  // Auto-detect the knobs the user didn't pin. Best-effort: any failure falls
  // back to the defaults already in `settings` and never blocks mining. Explicit
  // --region / --gpu always win.
  if (!settings.workerProvided) settings.worker = defaultWorker();
  // A region is needed for Pearl without --pool, and for Monero without --xmr-pool.
  const moneroNeedsRegion = settings.xmrAddress && !settings.xmrEndpoint;
  if (!settings.regionProvided && (!settings.endpoint || moneroNeedsRegion)) settings.region = await detectRegion();
  // The card's name and how many there are, from the probe the GUI uses too.
  // Probe regardless of --gpu: naming the card and counting the cards are two
  // different questions, and the count is still wanted for the 'gpu:' line
  // below when --gpu names the card.
  const det = await probe.detectGpuInfo();
  const amdListed = det && det.amd;
  if (!settings.gpuProvided && det && det.name) settings.gpu = det.name;
  // Every card mines, one core each, unless --gpu-index or PEARL_GPU_INDEX
  // chose some. Read here, with the rest of the startup probing, because the
  // start site below is inside a callback.
  const chosen = gpuPick ? gpuPick.indices : null;
  let nvidiaGpus = await detectMinerGpus(process.env, chosen);
  // The AMD core's cards are HIP's, which nvidia-smi doesn't list: on a rig
  // with an NVIDIA card beside the AMD ones, its list would drop the AMD
  // cards the indices name. The chosen indices go to the core as they are.
  // The same for the Intel core, whose cards are SYCL's.
  if (wantsAmdCore() || wantsIntelCore()) nvidiaGpus = planMinerGpus([], chosen);
  // With none chosen, an AMD or Intel rig's list is empty and one core would
  // mine one card. The OS counts them instead (det.amd, det.intel), as in the
  // GUI (shared/gpu planMiningCards).
  const { gpus, amdGpus, intelGpus } = planMiningCards({ env: process.env, det, chosen, nvidiaGpus });
  settings.gpus = gpus;
  // The Intel count also tells the core loader that this is an Intel rig
  // (shared/coreVariant).
  const intelCards = intelCount(det);
  // Always set, so downstream reads don't need a fallback: 1 when detection
  // found nothing or found a single card.
  settings.gpuCount = det && det.count > 1 ? det.count : 1;
  // With cards chosen, the 'gpu:' line describes those, not the box: the
  // second of two instances on one box, mining on GPU 1, called itself
  // "2× RTX A2000 12GB" when it had one A2000 beside an RTX PRO 6000. Its
  // name comes from the chosen cards too, unless --gpu names it. An AMD or
  // Intel card chosen has no name here, so the OS's name stays.
  if (gpuPick) {
    settings.gpuCount = settings.gpus.length;
    const picked = pickGpu(settings.gpus.map((g) => g.name));
    if (!settings.gpuProvided && picked) settings.gpu = picked;
  }
  const endpoint = resolveEndpoint(settings);
  // Kryptex has no merge mining, so the miner leaves the MDL address out there.
  const mdl = settings.mdlAddress && !isKryptexHost(splitEndpoint(endpoint).host);
  log('address:    ' + shortenAddress(settings.address) + (mdl ? '  (+MDL ' + shortenAddress(settings.mdlAddress) + ')' : ''));
  // An old AlphaPool id (--region eu1) mines in the region it maps to.
  if (settings.legacyRegion) {
    log('region ' + settings.legacyRegion + ' is now ' + settings.region
      + ' (' + REGIONS[settings.region].name + ')');
  }
  if (settings.endpointFrom === 'pool-url') log('pool:       ' + endpoint + '  (the flight sheet\'s Pool URL)');
  else if (settings.endpoint) log('pool:       ' + endpoint + '  (--pool)');
  else log('pool:       ' + endpoint + '  ' + regionLabel(settings.region) + (settings.regionProvided ? '' : '  (auto)'));
  // What the flight sheet's Pool URL did, said in the log and, when the owner
  // would not expect it, on the HiveOS dashboard (at most once an hour).
  const poolNote = poolUrlNote(settings.poolUrl,
    { poolFlag: settings.endpointFrom === '--pool', regionProvided: settings.regionProvided });
  if (poolNote.log) log(poolNote.log);
  if (hive && poolNote.post) hiveos.postMessage(poolNote.post.type, poolNote.post.text);
  // The miner's own warning is in the log; the dashboard is where a HiveOS
  // owner looks (at most once an hour, as every message).
  if (hive && isKryptexAccount(settings.address)) hiveos.postMessage('warning', KRYPTEX_ACCOUNT_MESSAGE);
  log('worker:     ' + settings.worker + (settings.workerProvided ? '' : '  (auto)'));
  if (settings.gpu) {
    log('gpu:        ' + (settings.gpuCount > 1 ? settings.gpuCount + '× ' : '') + settings.gpu
      + (settings.gpuProvided ? '' : '  (auto)'));
  }
  // The AMD driver's version, and a warning for a build reported to move idle
  // VRAM to system RAM (shared/amdDriver). Only Windows reports a version.
  for (const l of describeAmdDrivers(det)) logAt(l.level, l.line);
  // What will actually mine, which is not always what the line above names: a
  // mixed rig has one name there and several cards here, and --gpu-index or
  // PEARL_GPU_INDEX narrows it. Each card names itself again as its core
  // starts. A chosen card nvidia-smi doesn't list is dropped (shared/gpu
  // planMinerGpus), and said so here.
  const mining = settings.gpus.map((g) => g.index);
  if (gpuPick) {
    for (const line of describeSkippedGpus(gpuPick.indices, settings.gpus, gpuPick.from)) {
      log(line, process.stderr);
    }
    log('mining on:  ' + cardList(mining) + '  (' + gpuPick.from + ')');
    if (clearedCuda != null) {
      log('ignoring CUDA_VISIBLE_DEVICES=' + clearedCuda + ': ' + gpuPick.from + ' picks the cards');
    }
  } else {
    const cudaLine = describeClearedCuda(clearedCuda, 'use --gpu-index to choose the cards');
    if (cudaLine) log(cudaLine);
    if (amdGpus.length) log(describeAmdPlan(amdGpus, 'use --gpu-index'));
    else if (intelGpus.length) log(describeIntelPlan(intelGpus, 'use --gpu-index'));
    else if (mining.length > 1) log('mining on:  ' + cardList(mining));
  }
  // An AMD rig short of CPUs for its cards has their threads sleep instead of
  // spin (issue #308). Set before the core opens a card, which is when it reads it.
  if (!settings.gpuWait) {
    const facts = readCpuFacts({ fs, os });
    const cpus = facts.quotaCpus ? Math.min(facts.cpus, facts.quotaCpus) : facts.cpus;
    const wait = amdWait({ cards: amdGpus.length, cpus, env: process.env });
    if (wait) {
      process.env.PEARL_WAIT = wait.mode;
      log(wait.line);
    }
  }
  return { endpoint, amdGpus, intelGpus, intelCards, amdListed };
}

// No mining run: nothing planned.
function noRig() {
  return { endpoint: null, amdGpus: [], intelGpus: [], intelCards: 0, amdListed: null };
}

// The mining core this rig loads (pearlCore coreFactory, wrapped with the
// launch-shape cache), and nvidia-smi's compute capabilities it was picked from:
// { createCore, cudaCards }. createCore is null when no core loads; the loader
// has said why, and this adds where the file goes. `rig` is prepareMiningRig's.
async function loadMiningCore(settings, rig, parts, hive) {
  const { intelCards, amdListed } = rig;
  const cudaCards = await probe.detectCudaCards();
  // Set when the loader found a core that would not load. It has said why
  // (a missing library, too old a glibc), and "not found" would contradict it.
  let loadFailed = null;
  const coreLog = (level, line) => {
    if (line.includes(' is there but would not load: ')) loadFailed = line;
    logAt(level, line);
  };
  // Each RTX 30 card's launch shape, timed once on this PC and cached beside node.json
  // (shared/shapePick).
  const createCore = withShapePick(coreFactory({
    resourcesPath: process.resourcesPath,
    gpus: settings.gpus,
    cards: cudaCards,
    intel: intelCards,
    amd: amdListed,
    coreFlag: !!settings.core,
    log: coreLog,
  }), {
    store: parts.shapeStore(),
    log: coreLog,
    gpus: settings.gpus,
  });
  // A rig whose Blackwell cards the driver is too old for mines on the rest,
  // or on none, and the dashboard should say why. Not awaited.
  if (hive) {
    hiveos.checkDriver().then((msg) => {
      if (!msg) return;
      log(msg, process.stderr);
      hiveos.postMessage('warning', msg);
    });
  }
  if (!createCore) {
    if (hive && loadFailed) hiveos.postMessage('danger', coreLoadMessage(loadFailed));
    // --core amd (or PEARL_CORE_VARIANT=amd) asked for the AMD core, a file
    // of its own. Pointing at pearl_core.node there would load a CUDA core on
    // an AMD rig. The core loader has already said what failed; when the file
    // is missing, this says where it goes. With PEARL_CORE_PATH set, only that
    // file is tried, so its own line is the whole answer.
    if (loadFailed) {
      // Nothing to add.
    } else if (wantsAmdCore()) {
      if (!process.env.PEARL_CORE_PATH) {
        log('Fix: put ' + CORE_FILES[CORE_AMD] + ' from the release next to the executable (a run from the source tree also finds earn/native/amd/build), or set PEARL_CORE_PATH to its absolute path.', process.stderr);
      }
    } else if (wantsIntelCore()) {
      // The Intel core is a folder: the addon needs the libraries beside it.
      if (!process.env.PEARL_CORE_PATH) {
        log('Fix: put the ' + CORE_INTEL_DIR + ' folder from the release (' + CORE_FILES[CORE_INTEL] + ' and the libraries beside it) next to the executable (a run from the source tree also finds earn/native/sycl/build), or set PEARL_CORE_PATH to the core\'s absolute path.', process.stderr);
      }
    } else {
      const where = 'searched: PEARL_CORE_PATH, beside the executable, and the dev tree';
      log('pearl_core.node not found -- this build cannot mine (' + where + ').', process.stderr);
      log('Fix: keep pearl_core.node from the release next to the executable, or set PEARL_CORE_PATH=/path/to/pearl_core.node.', process.stderr);
    }
  }
  return { createCore, cudaCards };
}

// The first step of a run: read the command line, and set what the cores read
// from the environment. Returns { code } when the run ends here (help,
// version, a bad flag), else { settings, hive, gpuPick, noGpus }.
function readRunSettings(argv) {
  const parsed = parseCliArgs(argv);

  if (parsed.help) { process.stdout.write(USAGE + '\n'); return { code: 0 }; }
  if (parsed.version) { process.stdout.write(pkg.version + '\n'); return { code: 0 }; }

  // Under HiveOS (h-run.sh sets LLMJOB_HIVEOS=1) problems also go to the
  // dashboard, where they are seen without opening the miner log.
  const hive = hiveos.isHiveos(process.env);

  if (parsed.errors.length) {
    for (const e of parsed.errors) log('error: ' + e, process.stderr);
    log('run with --help for usage', process.stderr);
    if (hive) hiveos.postMessage('danger', startupMessage(parsed.errors));
    return { code: 1 };
  }

  const settings = parsed.settings;
  for (const w of settings.warnings) log('warning: ' + w);
  // --core beats PEARL_CORE_VARIANT. Set here, before the update, so the
  // updater and the core loader read the same choice.
  if (settings.core) process.env.PEARL_CORE_VARIANT = settings.core;
  // --gpu-wait beats PEARL_WAIT, which the CUDA and AMD cores read when they open a card.
  if (settings.gpuWait) process.env.PEARL_WAIT = settings.gpuWait;
  // The rotation fold is on unless --no-rotation-fold (or the rig's own
  // PEARL_FOLD_ROT=0) says otherwise; PEARL_FOLD_ROT is what the CUDA core reads.
  // From here on settings.rotationFold is the answer, true or false.
  settings.rotationFold = resolveRotationFold(settings.rotationFold, process.env);
  process.env[CORE_ROTATION_FOLD_ENV] = settings.rotationFold ? '1' : '0';
  const gpuPick = gpuChoice(settings, process.env);
  if (gpuPick && gpuPick.error) {
    log('error: ' + gpuPick.error, process.stderr);
    return { code: 1 };
  }
  // --gpu-index none (or PEARL_GPU_INDEX=none): no card mines. The LLM, which then picks its own cards,
  // still follows the mode. With nothing else to run the CLI waits rather than exits:
  // HiveOS passes none when every GPU is turned off there, and an exit would
  // only be restarted every few seconds.
  const noGpus = !!(gpuPick && !gpuPick.indices.length);
  // The cards the local LLM may use: the same ones, or every card when none
  // was chosen (shared/gpu llmCards).
  settings.llmGpuIndices = gpuPick ? gpuPick.indices : null;
  return { settings, hive, gpuPick, noGpus };
}

// What the phases of a run share: the engines it started, the timers that
// keep it going, and how far it is from stopping. They were locals of run(),
// which every phase read and set through closures. Each phase is now a
// function of its own, handed this object.
function createRunState() {
  const r = {
    stats: initStats(Date.now()),
    miner: null,
    llm: null,
    reporter: null,
    statsWriter: null,
    // Set once the auto plan is known, for demand mode only: the node id this
    // machine serves under. Its presence is what tells the reporter below that a
    // model is available on request even though none is loaded.
    demandServing: null,
    // The reporter's own function, hoisted so the demand plan can fire one extra
    // board row the moment it knows what this node will serve.
    reportNow: null,
    stopping: false,
    // The exit code a deliberate shutdown ends with: 0, unless it is a failed
    // trial (shared/hiveos), which HiveOS's h-run.sh rolls back on.
    stopCode: 0,
    // Whether the miner has reported a hashrate yet, for the trial.
    hashed: false,
    // Set once the run has ended (finish).
    settled: false,
    // The gate in front of the LLM (demand mode's, or the plain one). A
    // deliberate switch must not look like an engine dying: the handlers below
    // tear the process down, and the gate stops engines on purpose.
    auto: null,
    // Demand mode's own cluster poller and keep-alive. Not owned by the fleet,
    // so they outlive every wake/sleep cycle and have to be stopped explicitly.
    demandWorker: null,
    demandPinger: null,
    // A run with nothing to do on purpose has nothing else holding the process
    // open, so it would exit at once. This timer keeps it waiting for a signal.
    idle: null,
    // HiveOS only: the update check every few hours, and a new version's trial.
    updateTimer: null,
    trialTimer: null,
    // Demand mode's model, fetched while mining (prefetchLlm). The gate waits on
    // it BEFORE stopping the miner, so a request that arrives mid-download keeps
    // the card mining until the files are there.
    prefetch: null,
    // The end of the run (makeFinish) and a deliberate stop (makeShutdown).
    finish: null,
    shutdown: null,
  };
  // The numbers the board and the stats file publish: no hashrate while the
  // pool connection is down or the miner is stopped (demand mode serving the
  // LLM), since the pool is getting none. The stats file used to read the raw
  // snapshot, so HiveOS showed the last hashrate while the GPU served.
  r.poolSnapshot = () => {
    const live = snapshot(r.stats, Date.now());
    return r.miner && !r.miner.poolConnected() ? offPool(live) : live;
  };
  // Co-running only: mining stops on a card while the LLM beside it answers a
  // prompt, and starts again a little after (shared/promptPause). Demand mode
  // has the whole card stop mining for as long as it serves, and LLM-only has no
  // miner, so neither feeds it: there is always a miner when this is called. One
  // that has stopped pauses and resumes nothing (PearlMiner).
  r.promptPause = createPromptPause({
    pause: (indices) => r.miner.pauseCards(indices).length > 0,
    resume: (indices) => r.miner.resumeCards(indices).length > 0,
    log: logAt,
    resumeAfterMs: LLM.promptPause.resumeAfterMs,
    maxPauseMs: LLM.promptPause.maxPauseMs,
  });
  return r;
}

// The GPU miner: load the core, build the engine, and start its log line and
// board reports. Returns the compute capabilities the core was picked from,
// or null when a 'mining' run has no core and ends.
async function setUpMiner(r, settings, rig, parts, gpuPick, hive, xmr) {
  // No engine to resolve: the GPU work is a linked N-API addon, so there is
  // nothing to download, no version to pick and no driver gate to clear
  // before we know whether this rig can mine. coreFactory returns null when
  // pearl_core.node is absent -- and that is decided HERE, before anything
  // starts, because what happens next depends on what was asked for. A
  // 'mining' run with no core has nothing to do: exiting non-zero says so to
  // systemd, where the old exit 0 read as success and produced a silent
  // ten-second restart loop that mined nothing. An 'auto' run still has its
  // LLM half, so it says loudly what is missing and serves inference.
  //
  // Which of the two builds loads (CUDA 12.8, or CUDA 13 on an all-Blackwell
  // rig with driver 580+) is decided from one more nvidia-smi query and the
  // card list above; see shared/coreVariant. The factory logs its choice.
  const { createCore, cudaCards } = await loadMiningCore(settings, rig, parts, hive);
  if (!createCore) {
    if (settings.mode === 'mining') return null;
    log('continuing with the local LLM only.', process.stderr);
  }
  if (createCore) r.miner = parts.pearlEngine({ createCore });
  if (r.miner) watchMining(r);
  if (settings.report) startBoardReport(r, settings, gpuPick, xmr);
  return cudaCards;
}

// The mining loop as the log sees it: the rig's line once a second, and the
// engine's own log lines and errors.
function watchMining(r) {
  r.miner.on('log', (l) => logAt(l.level, l.line));
  // The line reports the RIG, and every card reports itself: each core ticks
  // its hashrate about twice a second, so a 13-card rig would write 26 copies
  // of the same totals every second into the journal. One a second is plenty
  // — no card's numbers are lost, they are all in the total.
  // The line prints the rig's rate averaged over the second since the last line,
  // not the latest card window: printing every other window could lock onto one
  // phase of an alternating pattern (see createRateMeter).
  let lastMineLog = 0;
  const meter = createRateMeter();
  r.miner.on('event', (evt) => {
    applyEvent(r.stats, evt, Date.now());
    if (evt.type === 'status' && evt.hashrate > 0) r.hashed = true;
    if (evt.type === 'status') {
      const now = Date.now();
      meterSample(meter, snapshot(r.stats, now).total, now);
      if (now - lastMineLog < MINE_LOG_MS) return;
      lastMineLog = now;
      const snap = snapshot(r.stats, now);
      // A rig with no job from the pool says so on this line too, as the last
      // line of a log is often the only one a supervisor shows (the Salad
      // boxes' minute lines are). Added at the end, after "up", so readers of
      // the line's start keep working.
      const wait = r.miner.poolWait();
      const noJob = wait && wait.ms >= NO_JOB_MS
        ? ' · no pool job for ' + Math.floor(wait.ms / 60000) + ' min (' + String(wait.why).slice(0, 120) + ')'
        : '';
      log('⛏  ' + format.formatHashrate(meterRead(meter, now)) + ' TH/s · '
        + format.formatInt(snap.accepted) + ' accepted · ' + snap.rejected + ' rejected · up '
        + format.formatUptime(snap.uptimeSec) + noJob);
    }
  });
  r.miner.on('error', (err) => log('engine error: ' + err.message, process.stderr));
}

// The network page's board: one row per card, now and every report interval.
function startBoardReport(r, settings, gpuPick, xmr) {
  // Sample per-card live VRAM (nvidia-smi, or amdgpu on an AMD rig) and post one board row per GPU,
  // just like the GUI — otherwise the board shows 0 GB for a CLI-driven rig.
  //
  // nvidia-smi lists every card, mining or not. When --gpu-index or
  // PEARL_GPU_INDEX chose the cards, only those get a row, so a card turned
  // off in HiveOS is not on the board at all.
  const report = () => reportToBoard({
    settings,
    version: pkg.version,
    snapshot: r.poolSnapshot,
    boardGpus: () => probe.detectBoardGpus(),
    indices: gpuPick ? settings.gpus.map((g) => g.index) : null,
    // Tag the cards serving the local LLM so the board shows which model each
    // GPU runs. `nodeId` rides along only while this machine is armed to serve
    // cluster jobs — running the model and serving the cluster are different
    // things, and the board should be able to tell them apart.
    //
    // In demand mode the model is deliberately NOT loaded between requests, so
    // keying this off a live fleet left the board's model column blank for a
    // node whose whole purpose is to serve that model — it appeared only for
    // the seconds either side of an actual request. "Not resident" is not the
    // same as "not available": the card answers for it within seconds, and the
    // tier was pinned at startup, so the honest thing to report is the model
    // this node serves. `indices` is the card the wake would land on, which is
    // the same card pickLlmGpu will choose then, from the cards this run may use.
    serving: (allVram) => {
      if (serveFleet) return { model: serveLlmState.model.name, indices: serveFleet.servingIndices(), nodeId: serveNodeId };
      if (!r.demandServing) return null;
      const best = pickLlmGpu(llmCards(allVram, settings.llmGpuIndices));
      return { model: serveLlmState.model.name, indices: best ? [best.index] : [], nodeId: r.demandServing.nodeId };
    },
    // The machine's RAM, and the CPU miner's hashrate (null when it isn't
    // running), for every row. The CPU miner starts after this reporter and
    // can give up on its own, so it is read at each report.
    host: () => hostStats(os, xmr && xmr.engine),
    post: postMinerReport,
  });
  r.reportNow = report;
  report();
  r.reporter = setInterval(report, NETWORK.reportIntervalMs);
  r.reporter.unref();
}

// What the local LLM does in this run, decided before anything starts: co-run
// beside the miner, wait for requests (demand), or serve alone, and the clock
// plan that follows from it. An LLM with no miner beside it starts here.
// Returns { autoPlan, demand, demandServe, coRun }.
async function planLlmRun(r, settings, plan, rig, cudaCards) {
  const { amdGpus, intelGpus } = rig;
  // Auto on a card that could serve a BIGGER model with the GPU to itself than it
  // can while mining does not co-run: it mines until something asks for tokens.
  // planAutoMode makes that call and returns 'corun' whenever the two choices
  // agree, so a small card behaves exactly as before.
  let autoPlan = null;
  if (plan.llm && plan.miner && normalizeMode(settings.mode) === 'auto') {
    const best = pickLlmGpu(await detectLlmCards(settings));
    autoPlan = planAutoMode(best ? best.freeMb : null, LLM.miningReserveMb, candidateModels(settings));
  }
  // `miner` is null when the mining core failed to load. Without that guard the
  // node believed it was in demand mode, skipped the up-front LLM start, never
  // built the gate (which needs a miner), and fell through to "nothing to run" --
  // exiting 1 on a card that could have served perfectly well.
  const demand = !!(r.miner && autoPlan && autoPlan.strategy === 'demand');
  // Demand mode polls for cluster work WHILE MINING, so it needs its identity
  // before any model exists -- not on the first wake, which may never come.
  const demandServe = demand ? await resolveServeIdentity(settings) : null;
  if (demand) {
    r.demandServing = { nodeId: demandServe && demandServe.canServe ? demandServe.nodeCfg.nodeId : null };
    // The first report already went out, before the plan was known, and would
    // otherwise leave the board blank until the next one a minute later.
    if (r.reportNow) r.reportNow();
  }
  // And it has to ADVERTISE the tier it will serve, not the one it has loaded --
  // which is none. serveLlmState.model is seeded with the small default and only
  // replaced inside startLlm, which demand mode reaches only on a wake, so
  // everything reading it while mining saw a model this node never serves: the
  // board and dashboard named the wrong one, and the ping is what the server
  // routes on, so a job asking for the tier was never offered to the one node
  // running it while a job asking for the default would have been answered by it.
  if (demand) serveLlmState.model = autoPlan.model;
  if (demand) {
    log('auto:       ' + autoPlan.model.name + ' needs the GPU to itself — mining until a request arrives');
  }

  // Which cards lock their memory clock while mining, and at what: the
  // RTX 5090 default, which tries the lock and keeps it only where the card
  // mines faster, or --mine-mem-clock (shared/memClock). Decided here,
  // where the LLM plan is known, because the lock is only safe while nothing
  // else wants the memory. LLM decode is memory-bandwidth-bound -- the opposite
  // of the fold -- so a model served from a locked card slows down with the
  // memory clock. Demand mode never overlaps the two: the miner's stop()
  // releases the lock, and ends a trial, before the gate starts llama-server.
  // Each of its mining runs takes the answer remembered for the card (the
  // engine's memClockStore, above), or starts the trial when there is none: a
  // run that ends before its trial does leaves no answer. A co-running LLM
  // shares the card for the whole run, so that drops the plan, and the miner
  // never has to ask which mode it is in. Set on `settings` so the gate's
  // restarts (startMinerArgs) carry the same plan.
  // Not on an AMD core: the plan is nvidia-smi's, and on a rig with an NVIDIA
  // card beside the AMD ones it would lock that idle card's memory clock.
  // An AMD core the CLI picked itself (HiveOS passes no --core) mines the AMD
  // plan, which is no NVIDIA card: an empty list says "NVIDIA-only".
  //
  // The core clock offset trial (--mine-core-offset auto, shared/coreOffset) is
  // planned beside it and by the same rules: NVIDIA cards only, so never on an
  // AMD core, and never while an LLM co-runs. In demand mode the miner's stop()
  // puts every card back at offset 0 before the gate starts llama-server.
  // An Intel core is the same on both counts: none of its cards is nvidia-smi's.
  if (r.miner && (wantsAmdCore() || wantsIntelCore())) {
    Object.assign(settings, noClocks());
  } else if (r.miner) {
    // On the rotation fold an RTX 40 card also tries a memory clock lock by
    // default (shared/memClock). The cards that have to earn their lock: each
    // RTX 5090 by default, more cards with --mine-mem-clock auto, none when a
    // clock was asked for.
    const clocks = planClocks({
      memMhz: settings.mineMemClockMhz, offset: settings.mineCoreOffset,
      cards: cudaCards, gpus: amdGpus.length || intelGpus.length ? [] : settings.gpus,
      llmCoRuns: !!(plan.llm && !demand), rotationFold: settings.rotationFold,
    });
    Object.assign(settings, clocks.start);
    for (const l of clocks.lines) logAt(l.level, l.line);
  }

  // Keep a mining reserve free only when co-running with the miner. In demand
  // mode nothing is co-resident, so the model is sized against the whole card.
  //
  // Co-running with a miner, only the VRAM plan is made here, while the card is
  // still empty -- the same reading and the same reserve as before. Fetching the
  // model and starting llama-server wait until the miner is running (startCoRun,
  // below), so a first run's multi-GB download no longer costs any mining time.
  // With no miner the LLM is all there is, and it starts now as it always has.
  // That includes a run meant to mine whose core failed: nothing else is on the
  // card, so it keeps no reserve.
  let coRun = null;
  if (plan.llm && !demand) {
    if (r.miner) coRun = await planLlm(settings, LLM.miningReserveMb);
    else r.llm = await startLlm(settings, 0);
  }
  return { autoPlan, demand, demandServe, coRun };
}

// The end of a run, whichever way it comes: stop what still runs and resolve
// the run with `code`. Once only.
function makeFinish(r, resolve, xmr) {
  return (code) => {
    if (r.settled) return;
    r.settled = true;
    // Every exit path runs through here, so the gate's listening socket is
    // released whether we stopped on a signal or an engine died underneath us.
    if (r.auto) { r.auto.stop(); r.auto = null; }
    if (r.demandWorker) { r.demandWorker.stop(); r.demandWorker = null; }
    if (r.demandPinger) { clearInterval(r.demandPinger); r.demandPinger = null; }
    if (r.idle) { clearInterval(r.idle); r.idle = null; }
    if (r.updateTimer) { clearInterval(r.updateTimer); r.updateTimer = null; }
    if (r.trialTimer) { clearInterval(r.trialTimer); r.trialTimer = null; }
    if (xmr) xmr.engine.stop();
    resolve(code);
  };
}

// A deliberate stop: a signal, a self-update, a failed trial.
function makeShutdown(r) {
  // `code` is the exit code once everything has stopped. A signal handler is
  // passed the signal's name, which is not one.
  return (code) => {
    if (r.stopping) return;
    r.stopping = true;
    if (Number.isInteger(code)) r.stopCode = code;
    log('shutting down…');
    r.promptPause.stop();
    if (r.reporter) clearInterval(r.reporter);
    if (r.statsWriter) clearInterval(r.statsWriter);
    stopServe();
    if (r.llm) r.llm.stop();
    // Stop the gate FIRST, so nothing can start a new transition -- or restart
    // the miner -- while we are tearing down.
    if (r.auto) { r.auto.stop(); r.auto = null; }
    // isRunning(), not truthiness. In demand mode the gate has usually already
    // stopped the miner, and PearlEngine.stop() on a stopped miner emits
    // nothing: `miner` was still truthy, so the else arm never ran, finish()
    // was never called, and the run never resolved. The process then sat until
    // TimeoutStopSec while its release timer restarted the miner underneath the
    // shutdown and its worker kept claiming jobs it would never finish. A
    // self-update is a restart, so the rollout mechanism hit this every time.
    if (r.miner && r.miner.isRunning()) r.miner.stop();
    else r.finish(r.stopCode); // nothing running that will resolve us with a 'stopped'
  };
}

// The LLM/gate startup: watch an LLM already running, and stand up the gate
// in front of it (demand mode's, or the plain one for LLM-only). Returns
// startCoRun, which starts a co-running LLM once the miner is up.
function startLlmAndGates(r, settings, plan, rig, llmRun) {
  const { amdGpus, intelGpus } = rig;
  const { autoPlan, demand, demandServe, coRun } = llmRun;
  // The context window the gate reports while the model is down, from the
  // fleet once a card has loaded or dropped down its ladder. Before, the gate
  // kept the ladder's top rung, so after one step down the 27B's ladder
  // clients were told twice the real window. Kept after the fleet stops:
  // demand mode answers from it all the time it mines.
  const advertiseCtx = (fleet) => {
    const sync = () => { if (r.auto) r.auto.setCtxSize(fleet.ctxSize()); };
    fleet.on('ready', sync);
    fleet.on('ctx-downgrade', sync);
  };

  // llama-server dying must never be a silent clean exit: without this
  // listener an LLM-only run drains the event loop and exits 0, so a systemd
  // Restart=on-failure supervisor never restarts the node. A deliberate stop
  // goes through fleet.stop(), which suppresses the fleet's 'stopped' event —
  // so reaching here always means the whole fleet died on its own.
  const watchLlm = (fleet) => {
    advertiseCtx(fleet);
    fleet.on('stopped', (code) => {
      // No demand-mode guard needed here: this handler is only registered for
      // an LLM started up front or beside the miner, which demand mode never
      // does.
      resetServeLlmState();
      // Nothing is left to answer a prompt: every card mines now.
      r.promptPause.reset();
      stopServe();
      log('local LLM exited (code ' + code + ')', process.stderr);
      if (!r.miner) r.finish(code || 1);
    });
  };
  if (r.llm) watchLlm(r.llm);

  // Co-running auto: the miner is already running; fetch and start the LLM
  // beside it. A setup failure is logged inside launchLlm and mining carries
  // on, as it did when the LLM started first. A run that ended while the model
  // was downloading must not leave a server behind.
  const startCoRun = async () => {
    // How many cards mine. The AMD or Intel plan can name a card its core
    // doesn't have; it fails on its own, so count the cards that started, as
    // the GUI does. A rig left mining on one card still pauses it for each prompt.
    const planned = amdGpus.length || intelGpus.length;
    const miningCards = planned ? Math.min(planned, r.miner.cardCount()) : settings.gpus.length;
    // Each card's bus id, to tell which card a prompt runs on
    // (promptPause.cardForPrompt).
    const busIds = await probe.detectPciBusIds();
    const fleet = await launchLlm(settings, coRun, true, true, (e) => {
      const card = cardForPrompt(e, busIds, miningCards);
      if (card !== undefined) r.promptPause.setBusy(card, e.busy, undefined, e.low);
    });
    if (!fleet) return;
    if (r.settled || r.stopping) { fleet.stop(); stopServe(); return; }
    r.llm = fleet;
    watchLlm(fleet);
  };

  // What both gates ask about the fleet `llm`. Whether a card is up to answer;
  // and the port to proxy to. NOT LLM.port: llmFleet probes upward from it
  // when it is busy, so the server can land on 8081+ while the gate still
  // proxies to 8080 and every request 502s. Ask the fleet where it actually
  // bound.
  const isLlmReady = () => !!(r.llm && r.llm.readyCount && r.llm.readyCount() > 0);
  const upstreamPort = () => {
    const url = r.llm && r.llm.webUrl && r.llm.webUrl();
    const p = url ? Number(new URL(url).port) : NaN;
    return Number.isFinite(p) && p > 0 ? p : LLM.port;
  };

  // Stood up BEFORE the miner starts, not after: a fatal start calls finish(),
  // and finish() closes the gate. Created afterwards it would bind the public
  // port on a run that had already resolved, leaving a listening socket behind.
  // ── Demand-driven auto ───────────────────────────────────────────────────
  // Stood up only when planAutoMode asked for it, so every other mode -- and
  // every card where the model co-runs -- is untouched.
  if (demand && r.miner) {
    r.auto = createAutoGate({
      miner: r.miner,
      prepareLlm: () => r.prefetch,
      startMinerArgs: () => Object.assign({}, settings, { endpoint: resolveEndpoint(settings) }),
      isLlmReady,
      // Returns the fleet; the gate owns waiting for it to answer.
      startLlm: async () => {
        // A fleet that is still up is restarting after a crash: the gate is
        // SERVING with no card ready. Hand it back and the gate waits for its
        // 'ready'. Planning another put a second server on the card and lost
        // the handle to the first, which kept restarting with nothing able
        // to stop it.
        if (r.llm && r.llm.hasSpawned()) return r.llm;
        // The miner has exited but the driver may not have reclaimed its VRAM
        // yet; plan against the card as it will be, not as it momentarily reads.
        // Ask for exactly what the planner will demand a moment later, using the
        // planner's own function so the two cannot drift apart. Waiting on
        // vramFullMb was waiting for LESS: the planner refuses below minVramMb
        // (the measured cost plus a margin so we never spawn at the edge), so
        // the wait could hand control back the instant there was ALMOST enough
        // and the refusal that followed read as "not enough VRAM" on a card that
        // would have been ready a second later. Reserve is 0: nothing co-resides.
        await waitForCardRoom(settings, requiredFreeMb(autoPlan.model, 0));
        // Sized against the whole card, and pinned to the tier already chosen:
        // nothing is co-resident in demand mode.
        r.llm = await startLlm(settings, 0, autoPlan.model, false);
        if (r.llm) {
          advertiseCtx(r.llm);
          r.llm.on('stopped', () => {
            if (r.auto && r.auto.isSwitching()) return;
            resetServeLlmState();
            stopServe();
          });
        }
        return r.llm;
      },
      stopLlm: async () => {
        // Wait for llama-server to EXIT, not merely to be signalled. Polling
        // free VRAM instead was not enough: on a real handback the 30s poll
        // expired with 1,341 MiB free of the ~2,081 the miner needs, the miner
        // failed to start, and the node sat with no gate, no miner and a 30 GB
        // model still resident until someone restarted it by hand.
        // stopAndWait says so on the fleet's own log channel if it gives up.
        if (r.llm) { try { await r.llm.stopAndWait(45000); } catch { /* already gone */ } }
        r.llm = null;
        // llm.stop() is a kill, not a join: LlmFleet.stop() signals the process
        // and returns, so llama-server still holds its ~30 GB for a moment. The
        // wake path already waits for the MINER's VRAM before spawning the
        // model; without the mirror image here the miner is restarted into a
        // card that is still full and its core fails to construct -- which,
        // now that a failed restart is fatal, takes the whole node down.
        // Observed on a 5090: 1,469 MiB free at the instant of the restart
        // against the ~2,081 MiB the rank-128 profile needs.
        // Backstop only, now that the exit is joined above: the driver can
        // still lag the process by a moment.
        await waitForCardRoom(settings, LLM.miningReserveMb, 15000);
        // The 'stopped' handler above also calls this, but it early-returns
        // while the gate is switching -- which is exactly this path. Without
        // it the ping timer survived every sleep, so a node that woke and
        // slept N times posted N duplicate telemetry pings per interval, each
        // spawning its own nvidia-smi.
        stopServe();
        // Handing the card back does not change WHICH model this node serves,
        // so the reported model survives the flip back to mining.
        resetServeLlmState();
      },
      // A failed restart is fatal for the same reason it is at first start:
      // the node would otherwise mine nothing while looking healthy.
      onMinerFailed: () => {
        log('engine failed to restart after serving — see the error above', process.stderr);
        if (r.llm) r.llm.stop();
        r.finish(1);
      },
      port: gatePort(settings),
      host: settings.gateHost || undefined,
      upstreamPort,
      modelName: autoPlan.model.name, ctxSize: ctxLadder(autoPlan.model)[0],
      quietMs: gateQuietMs(settings), log,
    }).start();
    log('auto:       serving on :' + gatePort(settings) + ' — '
      + quietLabel(settings) + ' with no requests hands the GPU back to mining');

    // Cluster work is PULLED, outbound, so a node behind NAT can serve with no
    // inbound networking at all. But a JobWorker is only built per READY
    // llama-server instance, and demand mode has no instance while mining --
    // so a demand node stopped pulling cluster jobs entirely and served only
    // whatever reached :8000 directly. On a NAT'd rig that is nothing: the
    // most capable cards in the fleet went silently idle.
    //
    // Give demand mode a worker of its own that polls while mining and wakes
    // the model the way an inbound request does.
    if (demandServe && demandServe.canServe) {
      const worker = makeCliJobWorker(demandServe.nodeCfg,
        () => (r.llm && r.llm.webUrl && r.llm.webUrl()) || null, r.auto.gate);
      r.demandWorker = worker;
      worker.start();

      // Keep-alive. The fleet's own ping loop is armed on its first ready card,
      // which in demand mode means the node only appeared online during the
      // seconds it happened to be serving and went stale on the board the rest
      // of the time. This one runs on the mining side of the flip too.
      //
      // serveNodeId is re-asserted every tick because handing the card back
      // runs stopServe(), which clears it -- correct for a node that has
      // stopped serving, wrong for one that is merely between jobs.
      const demandPing = async () => {
        serveNodeId = demandServe.nodeCfg.nodeId;
        return nodeClient.ping(demandServe.nodeCfg, await fullTelemetry(demandServe.nodeCfg, worker.activeJobs()));
      };
      demandPing();
      r.demandPinger = setInterval(demandPing, NODE.pingIntervalMs);
      r.demandPinger.unref();
      log('auto:       polling for cluster jobs while mining');
    }
  } else if ((plan.llm && !r.miner) || coRun) {
    // Serving with the card to ourselves, or beside the miner. There is nothing
    // to switch, but the endpoint must not move: the gate port is THE
    // documented endpoint, and it used to exist only in auto -- so choosing llm
    // mode relocated callers to llama-server's own port without telling them.
    // Co-running auto had no gate at all until issue #308: llama-server listens
    // on 127.0.0.1 only, so another computer's http://<rig>:8000/v1 found
    // nothing there. Until the co-running model is up, the gate answers probes
    // from its own state and a chat request gets a 502, as in llm mode.
    r.auto = createServeGate({
      port: gatePort(settings),
      host: settings.gateHost || undefined,
      upstreamPort,
      modelName: serveLlmState.model.name,
      ctxSize: ctxLadder(serveLlmState.model)[0],
      isLlmReady,
      // A request with "low_priority": true keeps the card mining while it runs
      // (LlmFleet.trackRequest). The gate exists only once the fleet does; a
      // request after the fleet is gone throws here, and the gate answers 503.
      track: (port, low) => serveFleet.trackRequest(port, low),
      log,
    }).start();
    log((r.miner ? 'auto:' : 'llm:') + '        serving on :' + gatePort(settings));
  }
  if (r.auto) {
    const lan = lanEndpoint(settings);
    if (lan) log('            other computers on your network: ' + lan + ' (no password: keep it off the internet)');
  }
  return startCoRun;
}

// Write live stats JSON for external consumers (HiveOS h-stats.sh reads this
// to feed the dashboard). Atomic write (tmp + rename) so readers never see a
// torn file; best-effort — a failed write must never affect mining.
function startStatsFile(r, settings, autoPlan, xmr) {
  // Each card's PCI bus id, read once. HiveOS needs it to put each card's
  // hashrate on the right dashboard row. Not awaited: mining shouldn't wait
  // on nvidia-smi, and the first writes simply go out without it.
  let pciBusIds = {};
  probe.detectPciBusIds().then((ids) => { pciBusIds = ids; });
  const writeStats = () => {
    try {
      const payload = statsFilePayload(r.poolSnapshot(), {
        version: pkg.version,
        nowMs: Date.now(),
        mode: normalizeMode(settings.mode),
        strategy: autoPlan ? autoPlan.strategy : null,
        gate: r.auto ? r.auto.gate.state : null,
        mining: !!(r.miner && r.miner.isRunning()),
        llm: serveLlmState,
        pciBusIds,
        xmr: xmr ? xmr.engine.snapshot() : null,
      });
      writeFileAtomic(settings.statsFile, JSON.stringify(payload), { fs });
    } catch (e) { /* best effort */ }
  };
  writeStats();
  r.statsWriter = setInterval(writeStats, 10000);
  r.statsWriter.unref();
}

// HiveOS only: the update check, and the trial of a version an update swapped in.
function startHiveTimers(r, settings, hive, trial) {
  // HiveOS: check for a new release every few hours and restart into it.
  // Nothing else restarts a rig's CLI for weeks at a time.
  if (hive && settings.update && selfUpdater.isPackaged()) {
    let checking = false;
    r.updateTimer = setInterval(() => {
      if (checking || r.stopping) return;
      checking = true;
      checkHiveUpdate().then((swapped) => {
        checking = false;
        if (swapped) r.shutdown(0);
      }, (e) => {
        checking = false;
        log('update check failed: ' + e.message, process.stderr);
      });
    }, NETWORK.updateCheckIntervalMs);
    r.updateTimer.unref();
  }

  // A version an update swapped in is on trial until it hashes. If a run that
  // should mine hasn't within ten minutes, it exits with an error, and
  // h-run.sh puts the old version back (shared/hiveos).
  if (trial) {
    const trialStart = Date.now();
    const from = trial.from ? ' from v' + trial.from : '';
    r.trialTimer = setInterval(() => {
      if (r.stopping) return;
      const verdict = trialVerdict({ elapsedMs: Date.now() - trialStart, hashing: r.hashed, mines: !!r.miner });
      if (verdict === 'wait') return;
      clearInterval(r.trialTimer);
      r.trialTimer = null;
      if (verdict === 'pass') {
        try {
          hiveos.confirmTrial(hiveos.packageDir());
        } catch (e) { /* the trial runs again on the next start */ }
        log('the update' + from + ' to v' + pkg.version + ' is done');
        hiveos.postMessage('success', 'LLMJob Earn updated to v' + pkg.version + from + '.');
        return;
      }
      log('v' + pkg.version + ' has not hashed in ' + (TRIAL_FAIL_MS / 60000) + ' minutes since the update'
        + from + '; exiting so HiveOS puts the old version back', process.stderr);
      r.shutdown(TRIAL_FAILED_EXIT);
    }, TRIAL_CHECK_MS);
    r.trialTimer.unref();
  }
}

// The CPU miner runs beside whatever else runs. It ends the run only when it
// is all there is and it gave up.
function startCpuMiner(r, settings, xmr) {
  const cpu = xmr.engine;
  cpu.on('log', (l) => logAt(l.level, l.line));
  // Already in the log; this listener keeps it from being thrown.
  cpu.on('error', () => {});
  let lastCpuLog = 0;
  // Rated lines only: the first status after the dataset is built comes
  // before the worker's first tick, at 0 H/s.
  cpu.on('event', (e) => {
    if (e.type !== 'status' || e.state !== 'mining' || !(e.hashrate > 0)) return;
    const now = Date.now();
    if (now - lastCpuLog < CPU_LOG_MS) return;
    lastCpuLog = now;
    log('cpu: ' + format.formatKhs(e.hashrate) + ' kH/s · ' + format.formatInt(e.accepted) + ' accepted · '
      + e.rejected + ' rejected · ' + e.threads + ' threads');
  });
  // Outside a shutdown, a stop is the engine giving up on its own. With
  // every GPU turned off (idle) the CPU miner was the run, so it ends too.
  cpu.on('stopped', () => {
    if (r.settled || r.stopping) return;
    if (!r.miner && !r.llm) {
      log('the CPU miner gave up and nothing else runs; exiting', process.stderr);
      r.finish(1);
    }
  });
  cpu.start({
    address: settings.xmrAddress,
    worker: settings.workerProvided ? settings.worker : defaultWorker(),
    endpoint: xmr.endpoint,
    threads: xmr.threads,
    priority: settings.cpuPriority,
  });
}

// Start the GPU miner, and end the run when it stops. Once it is up, the
// co-running LLM starts beside it, or demand mode fetches its model.
function runMiner(r, settings, llmRun, startCoRun) {
  const { autoPlan, demand, coRun } = llmRun;
  r.miner.on('stopped', (code) => {
    if (r.auto && r.auto.isSwitching()) return;   // handed to the LLM on purpose
    if (r.reporter) clearInterval(r.reporter);
    if (r.statsWriter) clearInterval(r.statsWriter);
    stopServe();
    if (r.llm) r.llm.stop();
    log('engine exited (code ' + code + ')');
    r.finish(r.stopping ? r.stopCode : (code || 0));
  });
  try {
    // A false return is a fatal start failure, not a hiccup: the core did not
    // construct, so there is no socket, no job, and no 'stopped' event coming.
    // Left unchecked the process simply ran out of work and exited 0 -- which
    // under Restart=always is a ten-second restart loop that mines nothing and
    // looks healthy to systemd. Exit non-zero so a supervisor can see it.
    // No LLM to stop on either failure: a co-running one starts only once the
    // miner has, and demand mode loads one only on request.
    if (r.miner.start(Object.assign({}, settings, { endpoint: resolveEndpoint(settings) })) === false) {
      log('engine failed to start — see the error above', process.stderr);
      r.finish(1);
    } else if (coRun) {
      // Not awaited, so nothing may escape it: an unhandled rejection would
      // end the process, miner and all.
      startCoRun().catch((e) => log('LLM start failed: ' + e.message + ' — mining carries on', process.stderr));
    } else if (demand) {
      r.prefetch = prefetchLlm(settings, autoPlan.model);
    }
  } catch (e) {
    log('failed to launch engine: ' + e.message, process.stderr);
    r.finish(1);
  }
}

async function run(argv) {
  if (argv[0] === 'update') return runExplicitUpdate();
  if (argv[0] === 'connect') return runConnect(argv.slice(1));

  const start = readRunSettings(argv);
  if (start.code != null) return start.code;
  const { settings, hive, gpuPick, noGpus } = start;

  // An update killed half way leaves its staging folder; and a package swapped
  // in by one is on trial until it hashes (shared/hiveos).
  let trial = null;
  if (hive) {
    hiveos.cleanStaging(hiveos.packageDir());
    trial = hiveos.readTrial(hiveos.packageDir());
  }

  if (settings.update) {
    const code = await maybeAutoUpdate(argv, settings);
    if (code != null) return code;
  }

  // Decide which engines run from the compute mode (mirrors the GUI): mine,
  // run a local LLM, or both. `canLlm` is always true on the CLI — the LLM's
  // own setup (binary/model) fails soft below if it can't start. `canMine` also
  // asks the platform: macOS has no mining core, and a Mac left ungated would go
  // looking for a Linux one (see shared/platform).
  const plan = resolvePlan(settings.mode, {
    canMine: !payoutProblem(settings.address, splitEndpoint(resolveEndpoint(settings)).host)
      && minerSupported(process.platform) && !noGpus,
    canLlm: true,
  });
  const idleByChoice = noGpus && !plan.llm;

  log('LLMJob Earn CLI v' + pkg.version);
  log('mode:       ' + settings.mode + (settings.modeProvided ? '' : '  (default)'));
  const platformNote = minerUnsupportedNote(process.platform, settings.mode);
  if (platformNote) log(platformNote, process.stderr);
  if (noGpus && normalizeMode(settings.mode) !== 'llm') {
    log('mining on:  no GPU  (' + (gpuPick.from === '--gpu-index' ? '--gpu-index none' : 'PEARL_GPU_INDEX=none') + ')');
  }

  // What both miners are built from (shared/minerDeps). Made once for both: they
  // share the pool lookup, whose last good addresses are kept beside the node
  // identity, shared with the desktop app. The launch shapes and memory clock
  // trials are kept beside node.json too, where the CLI has always kept them
  // (the desktop app keeps its own in userData).
  const nodeDir = () => path.dirname(nodeStore.nodePath());
  const parts = minerDeps({
    version: pkg.version, nodeDir, storeDir: nodeDir, fs, settingsLog: log, lookupLog: (line) => log(line),
  });

  // The rig: which cards mine, where, and the lines that say so.
  const rig = plan.miner ? await prepareMiningRig(settings, gpuPick, hive) : noRig();

  // ── CPU miner (Monero) ─────────────────────────────────────────────────────
  // Without the GPU miner the region is still detected, for the Monero host,
  // unless --xmr-pool names it.
  if (settings.xmrAddress && !settings.xmrEndpoint && !plan.miner && !settings.regionProvided) settings.region = await detectRegion();
  const xmr = settings.xmrAddress ? setUpCpuMining(settings, parts) : null;

  const r = createRunState();
  // nvidia-smi's compute capability per card, read with the miner below. The
  // memory clock plan needs it too, further down, once the LLM plan is known.
  let cudaCards = [];

  // ── Miner ────────────────────────────────────────────────────────────────
  if (plan.miner) {
    cudaCards = await setUpMiner(r, settings, rig, parts, gpuPick, hive, xmr);
    if (!cudaCards) return 1;
  }

  // ── Local LLM ──────────────────────────────────────────────────────────────
  const llmRun = await planLlmRun(r, settings, plan, rig, cudaCards);

  // Nothing to run (e.g. the LLM failed to set up and there's no miner): exit
  // with an error rather than hanging on an idle process. Unless nothing was
  // asked to run: no GPU in mining mode waits (see noGpus above).
  // The CPU miner does not cover for a GPU that was meant to serve the LLM and
  // serves nothing: that exits 1 too, so the failure is seen and retried. With
  // every GPU turned off, the CPU is all that was meant to run, so it runs.
  const llmFailed = plan.llm && !noGpus;
  if (!r.miner && !r.llm && !idleByChoice && (!xmr || llmFailed)) {
    log('nothing to run — no miner and the LLM did not start', process.stderr);
    return 1;
  }
  if (idleByChoice) {
    log(xmr ? 'no GPU mines (every one is turned off); mining Monero on the CPU'
      : 'nothing to run with every GPU turned off; waiting until stopped');
  }

  return new Promise((resolve) => {
    r.finish = makeFinish(r, resolve, xmr);
    // A run with nothing to do on purpose has nothing else holding the process
    // open, so it would exit at once. This timer keeps it waiting for a signal.
    r.idle = idleByChoice ? setInterval(() => {}, 60 * 60 * 1000) : null;
    r.shutdown = makeShutdown(r);
    process.on('SIGINT', r.shutdown);
    process.on('SIGTERM', r.shutdown);
    const startCoRun = startLlmAndGates(r, settings, plan, rig, llmRun);
    // Placed HERE, after the gate exists, rather than beside the miner report:
    // it used to live inside `if (plan.miner)`, so `--mode llm` wrote nothing at
    // all, and it could not see the model or the gate even when they were up.
    if (settings.statsFile) startStatsFile(r, settings, llmRun.autoPlan, xmr);
    startHiveTimers(r, settings, hive, trial);
    if (xmr) startCpuMiner(r, settings, xmr);
    if (r.miner) runMiner(r, settings, llmRun, startCoRun);
  });
}

// Force the exit if the loop will not drain on its own.
//
// Setting exitCode asks node to leave once nothing is left to do, which assumes
// everything we started has been torn down. A llama-server child that outlived
// its stop kept the loop alive after a fatal handback: the gate socket was
// closed, the miner was not running, and the process sat there -- so systemd saw
// `active`, Restart=on-failure never fired, and the node was neither mining nor
// serving until someone noticed. A non-zero exit that does not exit is not an
// exit; give the teardown a few seconds and then insist.
function exitWith(code) {
  process.exitCode = code;
  const t = setTimeout(() => process.exit(code), 5000);
  t.unref();
}

// Run the CLI and exit with its code. Both entry points call this: the guard
// below for `node earn-cli.js`, and sea-entry.js for the packaged binary that
// HiveOS and systemd run. That binary used to call run() itself and only set
// exitCode, so it never got the forced exit above.
function main(argv) {
  return run(argv).then(exitWith, (e) => {
    log('fatal: ' + (e && e.message ? e.message : e), process.stderr);
    exitWith(1);
  });
}

/* istanbul ignore next */
if (require.main === module) main(process.argv.slice(2));

module.exports = { run, main };
