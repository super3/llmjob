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
const net = require('net');
const { PearlEngine } = require('../main/pearlEngine');
const { coreFactory } = require('../main/pearlCore');
const { LlmManager } = require('../main/llmManager');
const { LlmEngineManager } = require('../main/llmEngineManager');
const { postJson, downloadFile, streamChatCompletion, extractLlamaZip } = require('../main/io');
const {
  detectRegion, detectVram, detectGpusVram, postMinerReport, findFreePort,
} = require('../main/probe');
const probe = require('../main/probe');
const nodeStore = require('../main/nodeStore');
const { initStats, applyEvent, snapshot } = require('../shared/miningStats');
const { NETWORK, LLM, NODE, resolveEndpoint, regionLabel } = require('../shared/config');
const { defaultWorker } = require('../shared/worker');
const nodeProto = require('../shared/node');
const { buildMinerReports } = require('../shared/minerReport');
const { statsFilePayload } = require('../shared/statsFile');
const { shortenAddress, isValidAddress } = require('../shared/address');
const { requiredFreeMb, pickLlmGpu } = require('../shared/vram');
const { planLlmInstances } = require('../shared/llmPlan');
const { pickModel, ctxLadder, planAutoMode } = require('../shared/models');
const { LlmFleet } = require('../main/llmFleet');
const { createAutoGate, createServeGate } = require('../main/autoGate');
const { JobWorker } = require('../main/jobWorker');
const { resolvePlan, normalizeMode } = require('../shared/llmMode');
const { minerSupported, minerUnsupportedNote } = require('../shared/platform');
const { resolveServerUrl } = require('../shared/llama');
const format = require('../shared/format');
const pkg = require('../../package.json');

// Write a log line. When attached to a TTY we prefix a wall-clock time; when
// piped (systemd/journald, `docker logs`, a file) we drop it, since the log
// collector adds its own timestamp and two would just be noise.
function log(line, stream) {
  const out = stream || process.stdout;
  const prefix = out.isTTY ? '[' + format.formatLogTime(new Date()) + '] ' : '';
  out.write(prefix + line + '\n');
}

// Detect the discrete GPU name via nvidia-smi (Linux/NVIDIA). Resolves the card
// name or null (no nvidia-smi / non-NVIDIA / parse failure). Never rejects — the
// engine still auto-detects the real device to mine; this is only for the
// status label.
// Resolve { name, count } — the representative card plus how many discrete
// GPUs the rig actually mines with.
// Delegates to the shared probe so the GUI and the CLI detect the same way —
// they had drifted into two different methods, and the GUI's was Windows-only.
function detectGpu() {
  return probe.detectGpuInfo();
}

// llama-server zips extract via the shared io helper; point failures at the
// CLI's escape hatch.
const LLM_UNZIP_HINT = 'install unzip, or pass --llm-binary </path/to/llama-server>';

// Explicit `llmjob-earn-cli update` — check the latest release and, if this is
// the packaged binary, replace it in place.
async function runExplicitUpdate() {
  log('checking for updates (current v' + pkg.version + ')…');
  const release = await selfUpdater.fetchLatestRelease();
  if (!release) { log('could not reach the update server', process.stderr); return 1; }

  const plan = planUpdate({ currentVersion: pkg.version, release, platform: process.platform });
  if (!plan.updateAvailable) {
    if (plan.reason === 'up-to-date') log('already up to date (v' + pkg.version + ')');
    else if (plan.reason === 'asset-missing') log('v' + plan.latestVersion + ' is out but has no Linux CLI binary yet', process.stderr);
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
    const exe = await selfUpdater.applyUpdate(plan);
    log('updated to v' + plan.latestVersion + ' (' + exe + '). Re-run to use it.');
    return 0;
  } catch (e) {
    log('update failed: ' + e.message, process.stderr);
    return 1;
  }
}

// Best-effort auto-update on start. Returns an exit code when it replaced and
// re-ran the binary (caller should return it), or null to keep mining.
async function maybeAutoUpdate(argv) {
  if (process.env[selfUpdater.UPDATED_ENV]) return null; // already the updated child
  const release = await selfUpdater.fetchLatestRelease();
  if (!release) return null; // offline — never block mining

  const plan = planUpdate({ currentVersion: pkg.version, release, platform: process.platform });
  if (!plan.updateAvailable) return null;

  if (!selfUpdater.isPackaged()) {
    log('a newer release is available: v' + plan.latestVersion + ' (run "llmjob-earn-cli update")');
    return null;
  }

  log('updating ' + plan.currentVersion + ' → ' + plan.latestVersion + ' before starting…');
  try {
    await selfUpdater.applyUpdate(plan);
    log('updated to v' + plan.latestVersion + '; restarting');
    return selfUpdater.reexec(argv);
  } catch (e) {
    log('auto-update failed (' + e.message + '); continuing on v' + pkg.version, process.stderr);
    return null;
  }
}

// Where the CLI caches the local-LLM binary + model (mirrors the engine dir).
function llmDir(settings) {
  return settings.llmDir || path.join(os.homedir(), '.local', 'share', 'llmjob-earn', 'llm');
}

// Resolve the llama-server binary for the local LLM. An explicit --llm-binary
// wins; otherwise fall back to a previously installed one in the cache dir, and
// only then download the llama.cpp release zip and extract it (via unzip). If
// extraction isn't possible (no `unzip`), we surface a clear error pointing at
// --llm-binary as the escape hatch.
async function resolveLlmBinary(settings, dir) {
  if (settings.llmBinary) {
    if (!fs.existsSync(settings.llmBinary)) {
      throw new Error('llama-server binary not found: ' + settings.llmBinary);
    }
    return settings.llmBinary;
  }
  const serverUrl = resolveServerUrl(process.platform, process.arch);
  const engine = new LlmEngineManager({
    dir, platform: process.platform, serverUrl,
    fs, download: downloadFile, extract: (zip, dest) => extractLlamaZip(zip, dest, LLM_UNZIP_HINT), chmod: fs.chmodSync,
  });
  if (engine.isServerInstalled()) {
    log('LLM server found: ' + engine.serverBinaryPath());
    return engine.serverBinaryPath();
  }
  log('downloading llama-server from ' + serverUrl + ' …');
  try {
    return await engine.ensureServer((pct) => {
      if (pct != null) process.stdout.write('\r  downloading… ' + pct + '%   ');
    });
  } catch (e) {
    throw new Error(e.message + ' — pass --llm-binary </path/to/llama-server> instead');
  } finally {
    process.stdout.write('\n');
  }
}

// Resolve the GGUF model path. An explicit --llm-model wins; otherwise reuse a
// cached download or fetch the small default model (a plain file, so this works
// on the CLI without zip extraction).
// `model` is the tier the caller chose (shared/models.pickModel); it is always
// passed, so there is deliberately no `|| LLM.model` default here — a silent
// fallback would be indistinguishable from the selection failing.
async function resolveLlmModel(settings, dir, model) {
  const m = model;
  // An explicit --llm-model is the operator's own file: we cannot know whether
  // it ships a projector, so it is served exactly as given, text-only.
  if (settings.llmModel) {
    if (!fs.existsSync(settings.llmModel)) {
      throw new Error('LLM model not found: ' + settings.llmModel);
    }
    return { modelPath: settings.llmModel, mmprojPath: null };
  }
  const engine = new LlmEngineManager({ dir, platform: process.platform, fs, download: downloadFile });
  let modelPath;
  if (engine.isModelInstalled(m)) {
    modelPath = engine.modelPath(m);
    log('LLM model found: ' + modelPath);
  } else {
    log('downloading LLM model (' + m.name + ') …');
    modelPath = await engine.ensureModel((pct) => {
      if (pct != null) process.stdout.write('\r  downloading model… ' + pct + '%   ');
    }, m);
    process.stdout.write('\n');
  }
  // The vision projector, for a model that ships one. Separate from the weights
  // so a node that already has 17 GB on disk can pick up a ~1 GB projector
  // without re-downloading them; null for a text-only model.
  let mmprojPath = null;
  if (!engine.isMmprojInstalled(m)) {
    log('downloading vision projector …');
    mmprojPath = await engine.ensureMmproj((pct) => {
      if (pct != null) process.stdout.write('\r  downloading projector… ' + pct + '%   ');
    }, m);
    process.stdout.write('\n');
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
// This machine's node id while it is armed to serve cluster jobs — reported on the
// miner ping so the network board can tell "running the model" from "serving the
// cluster". Null when not serving (mining only, or --no-serve).
let serveNodeId = null;

// The GPU name is static — probe it once and reuse, instead of spawning
// nvidia-smi twice per ping forever.
let cliGpuProbed = false;
let cliGpuName = null;
async function cachedDeviceName() {
  if (!cliGpuProbed) {
    cliGpuProbed = true;
    try { const det = await detectGpu(); cliGpuName = det && det.name ? det.name : null; } catch (e) { cliGpuName = null; }
  }
  return cliGpuName;
}

// Register this box as an unclaimed node so the server will answer its job
// polls. Signed like a ping (the server derives the nodeId from the public key
// and never trusts the body's). Resolves true on success; never throws — an
// unreachable server must not stop the local model from running.
async function registerNode(node, base) {
  const body = nodeProto.buildPingBody({
    nodeId: node.nodeId, publicKey: node.publicKey, secretKey: node.secretKey,
    timestamp: Date.now(), telemetry: { name: node.name || undefined },
  });
  try {
    const res = await postJson(base + '/api/nodes/register', body, 15000);
    return res.status === 200;
  } catch (e) {
    return false;
  }
}

// One signed ping. `telemetry` may be sparse — fields left undefined keep the
// server's stored values (its pick() only overwrites defined fields).
async function pingServer(node, base, telemetry, verbose) {
  const body = nodeProto.buildPingBody({
    nodeId: node.nodeId, publicKey: node.publicKey, secretKey: node.secretKey,
    timestamp: Date.now(), telemetry,
  });
  try {
    const res = await postJson(base + '/api/nodes/ping', body, 15000);
    if (verbose) {
      log(res.status === 200 ? '✓ ping' : ('✗ ping failed (HTTP ' + res.status + ')'),
        res.status === 200 ? process.stdout : process.stderr);
    }
  } catch (e) {
    if (verbose) log('✗ ping error: ' + e.message, process.stderr);
  }
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
  const device = await cachedDeviceName();
  if (device) t.device = device;
  if (node.name) t.name = node.name;
  return t;
}

// Full telemetry for the serving run: live model/readiness/tok-s/active jobs.
async function fullTelemetry(node) {
  let vram = null;
  try { vram = await detectVram(); } catch (e) { /* ignore */ }
  return nodeProto.buildTelemetry({
    model: serveLlmState.model.name, quant: serveLlmState.model.quant,
    device: await cachedDeviceName(), vram,
    tokensPerSec: serveLlmState.tps, ready: serveLlmState.ready,
    activeJobs: serveFleet ? serveFleet.activeJobs() : 0,
    name: node.name,
  });
}

function stopServe() {
  if (serveFleet) { serveFleet.syncWorkers(false); serveFleet = null; }
  if (servePinger) { clearInterval(servePinger); servePinger = null; }
  serveNodeId = null;
}

// Build a cluster job-worker for the ready LLM instance at `baseUrl` — one per
// serving GPU, each running jobs against its own llama-server. The fleet only
// calls this once serving is armed (syncWorkers(canServe)), so `nodeCfg` is
// always a linked node here; the fleet starts the returned worker.
// Node identity and its server registration, resolved ONCE per process.
//
// This used to live inside startLlm, which was fine while a fleet was the only
// thing that ever served. Demand mode needs an identity before any model is
// loaded -- it polls for cluster work while mining -- and resolving it twice
// would register the node twice and log it twice.
let serveIdentity = null;
async function resolveServeIdentity(settings) {
  if (serveIdentity) return serveIdentity;
  // Reuse the stored identity when there is one (it carries the node's name and
  // server URL); only mint a keypair when serving and none exists yet.
  const nodeCfg = loadNodeConfig() || (settings.serve ? getOrCreateNodeConfig() : null);
  const canServe = !!(settings.serve && nodeCfg);
  // Publish the node id on the miner ping only while actually armed to serve.
  serveNodeId = canServe ? nodeCfg.nodeId : null;
  const base = (nodeCfg && nodeCfg.serverUrl) || NODE.serverUrl;

  // An unclaimed node needs a row server-side before /jobs/poll will answer it.
  // Best-effort: a failure here just means no jobs arrive, and the local model
  // still runs.
  if (canServe && !nodeCfg.connected) {
    const registered = await registerNode(nodeCfg, base);
    log(registered
      ? 'serving public jobs as an unlinked node (' + nodeCfg.nodeId + ') — run "connect" to attach it to your account'
      : 'could not register with the network — running the LLM locally only', registered ? process.stdout : process.stderr);
  }
  serveIdentity = { nodeCfg, canServe, base };
  return serveIdentity;
}

// `baseUrl` may be a thunk: in demand mode no server exists when the worker is
// built, and the port is not known until one is spawned. `gate`, when given,
// makes this worker WAKE the model rather than assume one is loaded.
function makeCliJobWorker(nodeCfg, base, baseUrl, gate) {
  const urlOf = typeof baseUrl === 'function' ? baseUrl : () => baseUrl;
  const stream = (chatBody, h) => streamChatCompletion(urlOf(), chatBody, h.onDelta, h.onReasoning).done;
  const w = new JobWorker({
    identity: { nodeId: nodeCfg.nodeId, publicKey: nodeCfg.publicKey, secretKey: nodeCfg.secretKey },
    serverUrl: base,
    post: (url, body) => postJson(url, body, 30000),
    runJob: gate
      ? async (chatBody, h) => {
        // begin()/end() bracket the WHOLE job, not just the wake. shouldRelease()
        // requires inFlight === 0, so without this the quiet timer would hand the
        // card back to mining part-way through a long generation -- the cluster
        // path bypasses the HTTP handler that normally does this accounting.
        gate.begin();
        try {
          await gate.ensureServing();
          return await stream(chatBody, h);
        } finally { gate.end(); }
      }
      : stream,
    servingModel: () => serveLlmState.model,
  });
  // The 'error' listener is mandatory: a listener-less EventEmitter 'error'
  // throws, and one transient poll failure would crash the whole CLI.
  w.on('error', (e) => log('job poll failed: ' + e.message + ' (retrying)', process.stderr));
  w.on('job', ({ id }) => log('cluster job ' + id + ' — running locally'));
  w.on('failed', ({ id, error }) => log('cluster job ' + id + ' failed: ' + error, process.stderr));
  return w;
}

// Start the local LLM (llama.cpp llama-server) alongside — or instead of — the
// miner. Plans one server per eligible GPU (the model is small enough to hold a
// copy on every card with room), sizes each card's offload from its free VRAM
// (keeping `reserveMb` free for mining), spawns the fleet, and logs its
// OpenAI-compatible endpoint. Returns the LlmFleet, or null if setup failed
// (best-effort — a failing LLM never takes the miner down).
// modelOverride pins the tier. Demand mode has already chosen one, and choosing
// again here would re-run that decision against whatever VRAM happens to be free
// at the moment of the switch -- see waitForFreeVram for why that is not the same
// number a moment later.
// armServe=false leaves cluster polling AND the keep-alive ping to the caller.
// Demand mode owns both: it has to poll and stay online while no fleet exists,
// and a fleet arming its own on every wake would double-poll, race for the same
// jobs, and leak a ping timer per cycle.
async function startLlm(settings, reserveMb, modelOverride, armServe = true) {
  const dir = llmDir(settings);

  // Plan one llama-server per eligible GPU before doing anything expensive
  // (downloading a ~5 GB model). Each instance runs --split-mode none and is
  // pinned to its card (--main-gpu), so the per-card free VRAM is what sizes and
  // gates it — never the rig's summed total (the model can't span cards, and
  // sizing against the sum would cram it onto device 0 and OOM). An empty plan
  // means VRAM was measured but no card had room — refuse. When VRAM can't be
  // read (non-NVIDIA / no driver) the planner returns one unknown-placement
  // instance and lets llama.cpp decide.
  const cards = await detectGpusVram();
  // Which model this run serves, from the best card's free VRAM. The headless
  // shell has to make the same choice the GUI does, or a large card running
  // under systemd silently keeps serving the small default.
  const bestCard = pickLlmGpu(cards);
  const model = modelOverride || pickModel(bestCard ? bestCard.freeMb : null, reserveMb || 0);
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
    // An empty plan means at least one card parsed but none fit, so pickLlmGpu
    // (same parse rules) returns that card for the error message.
    const gpu = pickLlmGpu(cards);
    // Quote the binding constraint (model + mining reserve when co-running), not
    // just the preflight floor — the floor understates what was actually enforced.
    log('not enough free VRAM on any single GPU for the local LLM: ' + gpu.freeMb
      + ' MB free on GPU ' + gpu.index + ', need ~' + requiredFreeMb(model, reserveMb || 0)
      + ' MB for ' + model.name + ' — skipping the LLM.', process.stderr);
    return null;
  }

  log('preparing local LLM (' + model.name + ') …');

  let binaryPath, modelPath, mmprojPath = null;
  try {
    binaryPath = await resolveLlmBinary(settings, dir);
    ({ modelPath, mmprojPath } = await resolveLlmModel(settings, dir, model));
  } catch (e) {
    log('LLM setup failed: ' + e.message, process.stderr);
    return null;
  }

  // Serve cluster jobs once a model is up — by DEFAULT, account or not. A rig
  // that can run the model is useful to the network whether or not anyone has
  // linked it, so an unlinked box self-registers (signature only) and takes
  // public work. The server hands an unclaimed node non-private jobs only, so
  // "unlinked" costs it access to private queues, nothing else. --no-serve opts
  // out entirely and keeps the model purely local.
  // Reuse the stored identity when there is one (it carries the node's name and
  // server URL); only mint a keypair when serving and none exists yet.
  const { nodeCfg, canServe, base } = await resolveServeIdentity(settings);

  // One fleet spawns a llama-server per plan entry, walking to a free port per
  // instance (the same busy-port self-heal the GUI has) and building a cluster
  // worker per ready card. Retries an early exit a few times per instance
  // (startAttempts/startRetryMs) for a port-bind clash that clears on retry.
  const fleet = new LlmFleet({
    host: LLM.host,
    basePort: LLM.port,
    makeManager: () => new LlmManager({ spawn, startAttempts: LLM.startAttempts, retryDelayMs: LLM.startRetryMs }),
    findFreePort,
    makeWorker: (baseUrl) => makeCliJobWorker(nodeCfg, base, baseUrl),
  });
  serveFleet = fleet;
  fleet.on('log', (l) => log(l.line, l.level === 'error' ? process.stderr : process.stdout));
  fleet.on('ready', ({ baseUrl }) => {
    serveLlmState.ready = true;
    log('🧠 local LLM ready — OpenAI endpoint ' + baseUrl + '/v1');
  });
  // One keep-alive ping loop for the whole fleet, started on the first ready
  // card: pings ride along with serving so the node stays online on the
  // dashboard (and never gets pruned) while it works.
  fleet.on('first-ready', () => {
    if (!canServe || !armServe) return;
    const pingFull = async () => pingServer(nodeCfg, base, await fullTelemetry(nodeCfg), false);
    pingFull();
    servePinger = setInterval(pingFull, NODE.pingIntervalMs);
    if (servePinger.unref) servePinger.unref();
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

  fleet.syncWorkers(canServe && armServe); // arm serving before instances come up
  // Same run bits the GUI passes: llmFleet merges these into every
  // llmManager.start(), so the model's context ladder, projector and tuned flags
  // reach llama-server here too.
  const ladder = ctxLadder(model);
  await fleet.start(plan, {
    platform: process.platform, binaryPath, modelPath, mmprojPath,
    ctxSize: ladder[0], ctxLadder: ladder, extraArgs: model.extraArgs,
    alias: model.name,
  });
  const gpus = plan.map((p) => (p.index == null ? 'auto' : p.index)).join(', ');
  log('local LLM starting on ' + plan.length + ' GPU' + (plan.length === 1 ? '' : 's') + ' [' + gpus + ']');
  return fleet;
}

// Wait for the GPU to actually report `needMb` free.
//
// A process exiting does not reclaim its VRAM synchronously: the miner's ~2.6 GB
// is still counted for a moment after its 'stopped' event. The LLM planner sizes
// against a live reading, so starting it too early sees a card that is short by
// exactly that much and quietly plans a SMALLER model -- observed on a 5090,
// which fell back to Gemma because 29,959 MiB was read where 30,720 was needed.
// 8s is far more than the release needs -- it is sub-second in practice -- and
// short enough that a card which genuinely has no room fails fast instead of
// stalling the first request behind a doomed wait.
async function waitForFreeVram(needMb, timeoutMs = 8000, stepMs = 250) {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const best = pickLlmGpu(await detectGpusVram());
    if (!best || !Number.isFinite(best.freeMb) || best.freeMb >= needMb) return best;
    if (Date.now() >= deadline) return best;   // let the planner refuse, with a real reading
    await new Promise((r) => setTimeout(r, stepMs));
  }
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
  const base = node.serverUrl || NODE.serverUrl;
  const name = (opts.name && opts.name.trim()) || node.name || defaultWorker();

  log('LLMJob node ' + node.nodeId + ' → ' + base);

  if (opts.token) {
    const joinBody = nodeProto.buildJoinBody({
      token: String(opts.token).trim(), nodeId: node.nodeId, publicKey: node.publicKey, name,
    });
    let res;
    try {
      res = await postJson(base + '/api/nodes/join', joinBody, 20000);
    } catch (e) {
      log('could not reach ' + base + ': ' + e.message, process.stderr);
      return 1;
    }
    if (res.status !== 200 && res.status !== 201) {
      log('join failed (HTTP ' + res.status + '): ' + ((res.data && res.data.error) || res.raw || ''), process.stderr);
      return 1;
    }
    const user = (res.data && res.data.user) || null;
    node.name = name; node.connected = true; node.user = user; saveNodeConfig(node);
    log('✓ linked' + (user ? ' to ' + user + '’s account' : ' to your account') + ' as ' + name);
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
  const pingOnce = async () => pingServer(node, base, await sparseTelemetry(node), true);

  await pingOnce();
  const timer = setInterval(pingOnce, NODE.pingIntervalMs);
  return new Promise((resolve) => {
    const shutdown = () => { clearInterval(timer); log('stopped pinging'); resolve(0); };
    process.on('SIGINT', shutdown);
    process.on('SIGTERM', shutdown);
  });
}

async function run(argv) {
  if (argv[0] === 'update') return runExplicitUpdate();
  if (argv[0] === 'connect') return runConnect(argv.slice(1));

  const parsed = parseCliArgs(argv);

  if (parsed.help) { process.stdout.write(USAGE + '\n'); return 0; }
  if (parsed.version) { process.stdout.write(pkg.version + '\n'); return 0; }

  if (parsed.errors.length) {
    for (const e of parsed.errors) log('error: ' + e, process.stderr);
    log('run with --help for usage', process.stderr);
    return 1;
  }

  const settings = parsed.settings;

  if (settings.update) {
    const code = await maybeAutoUpdate(argv);
    if (code != null) return code;
  }

  // Decide which engines run from the compute mode (mirrors the GUI): mine,
  // run a local LLM, or both. `canLlm` is always true on the CLI — the LLM's
  // own setup (binary/model) fails soft below if it can't start. `canMine` also
  // asks the platform: macOS has no alpha-miner build, and without the gate the
  // engine resolver would download the Linux binary and spawn something the
  // kernel refuses to exec (see shared/platform).
  const plan = resolvePlan(settings.mode, {
    canMine: isValidAddress(settings.address) && minerSupported(process.platform),
    canLlm: true,
  });

  log('LLMJob Earn CLI v' + pkg.version);
  log('mode:       ' + settings.mode + (settings.modeProvided ? '' : '  (default)'));
  const platformNote = minerUnsupportedNote(process.platform, settings.mode);
  if (platformNote) log(platformNote, process.stderr);

  let endpoint = null;
  if (plan.miner) {
    // Auto-detect the knobs the user didn't pin. Best-effort: any failure falls
    // back to the defaults already in `settings` and never blocks mining. Explicit
    // --region / --gpu always win.
    if (!settings.workerProvided) settings.worker = defaultWorker();
    if (!settings.regionProvided) settings.region = await detectRegion();
    // Probe regardless of --gpu. Naming the card and counting the cards are two
    // different questions, and folding them into one `if` meant that passing
    // --gpu (to name it) also skipped the COUNT — so an N-card rig fell back to
    // gpuCount 1 and reported one card on a board row for N.
    const det = await detectGpu();
    if (!settings.gpuProvided && det && det.name) settings.gpu = det.name;
    // Always set, so downstream reads don't need a fallback: 1 when detection
    // found nothing or found a single card.
    settings.gpuCount = det && det.count > 1 ? det.count : 1;
    endpoint = resolveEndpoint(settings);
    log('address:    ' + shortenAddress(settings.address) + (settings.mdlAddress ? '  (+MDL ' + shortenAddress(settings.mdlAddress) + ')' : ''));
    log('pool:       ' + endpoint + '  ' + regionLabel(settings.region) + (settings.regionProvided ? '' : '  (auto)'));
    log('worker:     ' + settings.worker + (settings.workerProvided ? '' : '  (auto)'));
    if (settings.gpu) {
      log('gpu:        ' + (settings.gpuCount > 1 ? settings.gpuCount + '× ' : '') + settings.gpu
        + (settings.gpuProvided ? '' : '  (auto)'));
    }
  }

  const stats = initStats(Date.now());
  let miner = null;
  let reporter = null;
  let statsWriter = null;
  let llm = null;
  let stopping = false;

  // ── Miner ────────────────────────────────────────────────────────────────
  if (plan.miner) {
    // No engine to resolve: the GPU work is a linked N-API addon, so there is
    // nothing to download, no version to pick and no driver gate to clear
    // before we know whether this rig can mine. coreFactory returns null when
    // pearl_core.node is absent -- and that is decided HERE, before anything
    // starts, because what happens next depends on what was asked for. A
    // 'mining' run with no core has nothing to do: exiting non-zero says so to
    // systemd, where the old exit 0 read as success and produced a silent
    // ten-second restart loop that mined nothing. An 'auto' run still has its
    // LLM half, so it says loudly what is missing and serves inference.
    const createCore = coreFactory({ resourcesPath: process.resourcesPath });
    if (!createCore) {
      const where = 'searched: PEARL_CORE_PATH, beside the executable, and the dev tree';
      log('pearl_core.node not found -- this build cannot mine (' + where + ').', process.stderr);
      log('Fix: keep pearl_core.node from the release next to the executable, or set PEARL_CORE_PATH=/path/to/pearl_core.node.', process.stderr);
      if (settings.mode === 'mining') return 1;
      log('continuing with the local LLM only.', process.stderr);
    }
    if (createCore) {
      miner = new PearlEngine({
        connect: (host, port) => net.connect(port, host),
        createCore,
        // Without this the engine never polls for a card temperature, so every
        // headless rig reported temp 0 -- to the stats file, to the miner report,
        // and to the network board. The GUI has always passed it (main.js);
        // omitting it here was an oversight, not a decision.
        readTemps: () => probe.detectGpuTemps(),
      });
    }
    if (miner) {
    miner.on('log', (l) => log(l.line, l.level === 'error' ? process.stderr : process.stdout));
    miner.on('event', (evt) => {
      applyEvent(stats, evt, Date.now());
      if (evt.type === 'status') {
        const snap = snapshot(stats, Date.now());
        log('⛏  ' + format.formatHashrate(snap.total) + ' TH/s · '
          + format.formatInt(snap.accepted) + ' accepted · ' + snap.rejected + ' rejected · up '
          + format.formatUptime(snap.uptimeSec));
      }
    });
    miner.on('error', (err) => log('engine error: ' + err.message, process.stderr));
    }

    if (settings.report) {
      // Sample per-card live VRAM (nvidia-smi) and post one board row per GPU,
      // just like the GUI — otherwise the board shows 0 GB for a CLI-driven rig.
      const report = async () => {
        const snap = snapshot(stats, Date.now());
        const gpuVram = await detectGpusVram();
        // Tag the cards serving the local LLM so the board shows which model each
        // GPU runs; null when the fleet isn't up (mining only) → blank on the board.
        // `nodeId` rides along only while this machine is armed to serve cluster
        // jobs — running the model and serving the cluster are different things,
        // and the board should be able to tell them apart.
        const serving = serveFleet
          ? { model: serveLlmState.model.name, indices: serveFleet.servingIndices(), nodeId: serveNodeId }
          : null;
        return Promise.all(buildMinerReports(settings, snap, gpuVram, pkg.version, serving).map(postMinerReport));
      };
      report();
      reporter = setInterval(report, NETWORK.reportIntervalMs);
      if (reporter.unref) reporter.unref();
    }

  }

  // ── Local LLM ──────────────────────────────────────────────────────────────
  // Auto on a card that could serve a BIGGER model with the GPU to itself than it
  // can while mining does not co-run: it mines until something asks for tokens.
  // planAutoMode makes that call and returns 'corun' whenever the two choices
  // agree, so a small card behaves exactly as before.
  let autoPlan = null;
  if (plan.llm && plan.miner && normalizeMode(settings.mode) === 'auto') {
    const cards = await detectGpusVram();
    const best = pickLlmGpu(cards);
    autoPlan = planAutoMode(best ? best.freeMb : null, LLM.miningReserveMb);
  }
  const demand = !!(autoPlan && autoPlan.strategy === 'demand');
  // Demand mode polls for cluster work WHILE MINING, so it needs its identity
  // before any model exists -- not on the first wake, which may never come.
  const demandServe = demand ? await resolveServeIdentity(settings) : null;
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

  // Keep a mining reserve free only when co-running with the miner. In demand
  // mode nothing is co-resident, so the model is sized against the whole card.
  if (plan.llm && !demand) {
    llm = await startLlm(settings, plan.miner ? LLM.miningReserveMb : 0);
  }

  // Nothing to run (e.g. the LLM failed to set up and there's no miner): exit
  // with an error rather than hanging on an idle process.
  if (!miner && !llm) {
    log('nothing to run — no miner and the LLM did not start', process.stderr);
    return 1;
  }

  return new Promise((resolve) => {
    let settled = false;
    const finish = (code) => {
      if (settled) return;
      settled = true;
      // Every exit path runs through here, so the gate's listening socket is
      // released whether we stopped on a signal or an engine died underneath us.
      if (auto) { auto.stop(); auto = null; }
      if (demandWorker) { demandWorker.stop(); demandWorker = null; }
      if (demandPinger) { clearInterval(demandPinger); demandPinger = null; }
      resolve(code);
    };
    // A deliberate switch must not look like an engine dying: the handlers below
    // tear the process down, and the gate stops engines on purpose.
    let auto = null;
    // Demand mode's own cluster poller and keep-alive. Not owned by the fleet,
    // so they outlive every wake/sleep cycle and have to be stopped explicitly.
    let demandWorker = null;
    let demandPinger = null;

    const shutdown = () => {
      if (stopping) return;
      stopping = true;
      log('shutting down…');
      if (reporter) clearInterval(reporter);
      if (statsWriter) clearInterval(statsWriter);
      stopServe();
      if (llm) llm.stop();
      if (miner) miner.stop();
      else finish(0); // LLM-only: no miner 'stopped' will resolve us
    };
    process.on('SIGINT', shutdown);
    process.on('SIGTERM', shutdown);

    // llama-server dying must never be a silent clean exit: without this
    // listener an LLM-only run drains the event loop and exits 0, so a systemd
    // Restart=on-failure supervisor never restarts the node. A deliberate stop
    // goes through fleet.stop(), which suppresses the fleet's 'stopped' event —
    // so reaching here always means the whole fleet died on its own.
    if (llm) {
      llm.on('stopped', (code) => {
        // Keep `model`: the run is over but the last telemetry ping should still
        // name what this node was serving, and fullTelemetry dereferences it.
        // No demand-mode guard needed here either: this handler is only
        // registered when the LLM was started up front, which demand mode
        // never does.
        serveLlmState = { ready: false, tps: 0, model: serveLlmState.model };
        stopServe();
        log('local LLM exited (code ' + code + ')', process.stderr);
        if (!miner) finish(code || 1);
      });
    }

    // Stood up BEFORE the miner starts, not after: a fatal start calls finish(),
    // and finish() closes the gate. Created afterwards it would bind the public
    // port on a run that had already resolved, leaving a listening socket behind.
    // ── Demand-driven auto ───────────────────────────────────────────────────
    // Stood up only when planAutoMode asked for it, so every other mode -- and
    // every card where the model co-runs -- is untouched.
    if (demand && miner) {
      auto = createAutoGate({
        miner,
        startMinerArgs: () => Object.assign({}, settings, { endpoint: resolveEndpoint(settings) }),
        isLlmReady: () => !!(llm && llm.readyCount && llm.readyCount() > 0),
        // Returns the fleet; the gate owns waiting for it to answer.
        startLlm: async () => {
          // The miner has exited but the driver may not have reclaimed its VRAM
          // yet; plan against the card as it will be, not as it momentarily reads.
          await waitForFreeVram(autoPlan.model.vramFullMb);
          // Sized against the whole card, and pinned to the tier already chosen:
          // nothing is co-resident in demand mode.
          llm = await startLlm(settings, 0, autoPlan.model, false);
          if (llm) {
            llm.on('stopped', () => {
              if (auto && auto.isSwitching()) return;
              // Keep `model` for the same reason the up-front handler does: the
              // last telemetry ping should still name what this node served.
              serveLlmState = { ready: false, tps: 0, model: serveLlmState.model };
              stopServe();
            });
          }
          return llm;
        },
        stopLlm: async () => {
          if (llm) { try { llm.stop(); } catch { /* already gone */ } }
          llm = null;
          // llm.stop() is a kill, not a join: LlmFleet.stop() signals the process
          // and returns, so llama-server still holds its ~30 GB for a moment. The
          // wake path already waits for the MINER's VRAM before spawning the
          // model; without the mirror image here the miner is restarted into a
          // card that is still full and its core fails to construct -- which,
          // now that a failed restart is fatal, takes the whole node down.
          // Observed on a 5090: 1,469 MiB free at the instant of the restart
          // against the ~2,081 MiB the rank-128 profile needs.
          await waitForFreeVram(LLM.miningReserveMb, 30000, 500);
          // The 'stopped' handler above also calls this, but it early-returns
          // while the gate is switching -- which is exactly this path. Without
          // it the ping timer survived every sleep, so a node that woke and
          // slept N times posted N duplicate telemetry pings per interval, each
          // spawning its own nvidia-smi.
          stopServe();
          // Handing the card back does not change WHICH model this node serves,
          // so the reported model survives the flip back to mining.
          serveLlmState = { ready: false, tps: 0, model: serveLlmState.model };
        },
        // A failed restart is fatal for the same reason it is at first start:
        // the node would otherwise mine nothing while looking healthy.
        onMinerFailed: () => {
          log('engine failed to restart after serving — see the error above', process.stderr);
          if (llm) llm.stop();
          finish(1);
        },
        port: settings.gatePort == null ? LLM.gate.port : settings.gatePort,
        // NOT LLM.port: llmFleet probes upward from it when it is busy, so the
        // server can land on 8081+ while the gate still proxies to 8080 and
        // every request 502s. Ask the fleet where it actually bound.
        upstreamPort: () => {
          const url = llm && llm.webUrl && llm.webUrl();
          const p = url ? Number(new URL(url).port) : NaN;
          return Number.isFinite(p) && p > 0 ? p : LLM.port;
        },
        modelName: autoPlan.model.name, quietMs: LLM.gate.quietMs, log,
      }).start();
      log('auto:       serving on :' + (settings.gatePort == null ? LLM.gate.port : settings.gatePort) + ' — '
        + Math.round(LLM.gate.quietMs / 1000) + 's with no requests hands the GPU back to mining');

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
        demandWorker = makeCliJobWorker(demandServe.nodeCfg, demandServe.base,
          () => (llm && llm.webUrl && llm.webUrl()) || null, auto.gate);
        demandWorker.start();

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
          return pingServer(demandServe.nodeCfg, demandServe.base,
            await fullTelemetry(demandServe.nodeCfg), false);
        };
        demandPing();
        demandPinger = setInterval(demandPing, NODE.pingIntervalMs);
        if (demandPinger.unref) demandPinger.unref();
        log('auto:       polling for cluster jobs while mining');
      }
    } else if (plan.llm && !miner) {
      // Serving with the card to ourselves. There is nothing to switch, but the
      // endpoint must not move: the gate port is THE documented endpoint, and it
      // used to exist only in auto -- so choosing llm mode relocated callers to
      // llama-server's own port without telling them.
      const gp = settings.gatePort == null ? LLM.gate.port : settings.gatePort;
      auto = createServeGate({
        port: gp,
        upstreamPort: () => {
          const url = llm && llm.webUrl && llm.webUrl();
          const p = url ? Number(new URL(url).port) : NaN;
          return Number.isFinite(p) && p > 0 ? p : LLM.port;
        },
        modelName: serveLlmState.model.name,
        isLlmReady: () => !!(llm && llm.readyCount && llm.readyCount() > 0),
        log,
      }).start();
      log('llm:        serving on :' + gp);
    }

    // Write live stats JSON for external consumers (HiveOS h-stats.sh reads this
    // to feed the dashboard). Atomic write (tmp + rename) so readers never see a
    // torn file; best-effort — a failed write must never affect mining.
    //
    // Placed HERE, after the gate exists, rather than beside the miner report:
    // it used to live inside `if (plan.miner)`, so `--mode llm` wrote nothing at
    // all, and it could not see the model or the gate even when they were up.
    if (settings.statsFile) {
      const writeStats = () => {
        try {
          const payload = statsFilePayload(snapshot(stats, Date.now()), {
            version: pkg.version,
            nowMs: Date.now(),
            mode: normalizeMode(settings.mode),
            strategy: autoPlan ? autoPlan.strategy : null,
            gate: auto ? auto.gate.state : null,
            mining: !!(miner && miner.isRunning()),
            llm: serveLlmState,
          });
          const tmp = settings.statsFile + '.tmp';
          fs.writeFileSync(tmp, JSON.stringify(payload));
          fs.renameSync(tmp, settings.statsFile);
        } catch (e) { /* best effort */ }
      };
      writeStats();
      statsWriter = setInterval(writeStats, 10000);
      if (statsWriter.unref) statsWriter.unref();
    }

    if (miner) {
      miner.on('stopped', (code) => {
        if (auto && auto.isSwitching()) return;   // handed to the LLM on purpose
        if (reporter) clearInterval(reporter);
        if (statsWriter) clearInterval(statsWriter);
        stopServe();
        if (llm) llm.stop();
        log('engine exited (code ' + code + ')');
        finish(stopping ? 0 : (code || 0));
      });
      try {
        // A false return is a fatal start failure, not a hiccup: the core did not
        // construct, so there is no socket, no job, and no 'stopped' event coming.
        // Left unchecked the process simply ran out of work and exited 0 -- which
        // under Restart=always is a ten-second restart loop that mines nothing and
        // looks healthy to systemd. Exit non-zero so a supervisor can see it.
        if (miner.start(Object.assign({}, settings, { endpoint: resolveEndpoint(settings) })) === false) {
          log('engine failed to start — see the error above', process.stderr);
          if (llm) llm.stop();
          finish(1);
        }
      } catch (e) {
        log('failed to launch engine: ' + e.message, process.stderr);
        if (llm) llm.stop();
        finish(1);
      }
    }

  });
}

/* istanbul ignore next */
if (require.main === module) {
  run(process.argv.slice(2)).then((code) => { process.exitCode = code; }).catch((e) => {
    log('fatal: ' + (e && e.message ? e.message : e), process.stderr);
    process.exitCode = 1;
  });
}

module.exports = { run };
