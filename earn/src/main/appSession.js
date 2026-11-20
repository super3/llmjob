'use strict';

// ── The START and idle-only ─────────────────────────────────────────────────
// What a START runs (the compute mode's plan, one run at a time), the START
// itself until STOP, and pausing its mining while someone uses the computer.
// Moved out of main.js unchanged. The parts of main.js it reads and calls come
// through appContext.

const { app, powerMonitor } = require('electron');

const { ctx, share, late } = require('./appContext');
const { initStats, snapshot } = require('../shared/miningStats');
const { LLM } = require('../shared/config');
const { requiredFreeMb } = require('../shared/vram');
const { resolvePlan, DEFAULT_MODE } = require('../shared/llmMode');
const { minerSupported, minerUnsupportedNote } = require('../shared/platform');
const { isValidAddress } = require('../shared/address');
const { isValidQuantusAddress } = require('../shared/quantus/address');
const { fp8On, canMine, quantusOnly } = require('../shared/forkPlan');
const appPrefs = require('../shared/appPrefs');

const send = late('send');
const persistSettings = late('persistSettings');
const getPrefs = late('getPrefs');
const savePrefs = late('savePrefs');
const applyLoginItem = late('applyLoginItem');
const statsView = late('statsView');
const startMiningOnce = late('startMiningOnce');
const haltMiner = late('haltMiner');
const stopMining = late('stopMining');
const startLlm = late('startLlm');
const stopLlm = late('stopLlm');
const llmReplan = late('llmReplan');
const stopLlmForReplan = late('stopLlmForReplan');
const syncNodeName = late('syncNodeName');
const installWhenHidden = late('installWhenHidden');

// Resolve once the running miner is actually mining — i.e. it has reported a
// non-zero hashrate, which proves the GPU is doing real work (not just connected
// to the pool). Also settles on a miner failure (exit/error) so a broken miner
// never blocks the LLM, and on a hard cap so a miner that connects but never
// produces a share doesn't hold it forever.
function waitForMinerUp(capMs) {
  return new Promise((resolve) => {
    // Capture the manager we subscribed to: `miner` is a module global that
    // stopMining() nulls, and the manager's 'stopped' event arrives async after
    // the kill — so cleanup through the global crashes the main process with
    // "Cannot read properties of null (reading 'removeListener')" whenever the
    // miner is stopped during this wait (seen in the field as the Electron
    // error dialog). Unsubscribe from the captured reference instead. The one
    // caller has just checked there is a running miner.
    const m = ctx.miner;
    let done = false;
    const settle = () => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      m.removeListener('event', onEvent);
      m.removeListener('stopped', settle);
      m.removeListener('error', settle);
      resolve();
    };
    const onEvent = (e) => { if (e && e.type === 'status' && Number(e.hashrate) > 0) settle(); };
    const timer = setTimeout(settle, capMs || 60000);
    m.on('event', onEvent);
    m.once('stopped', settle);
    m.once('error', settle);
  });
}

// Which engines a START asks for. With Quantus picked (Settings → GPU Coin) the
// GPUs mine Quantus to its own address, and only that, as the CLI's --coin
// quantus does: Auto mines without the local LLM, and LLM mode runs the model
// alone, as it does for Pearl.
//
// A card can be picked for the other coin (settings.cardCoins), and Pearl's
// FP8 fork moves the cards that cannot mine it to Quantus (shared/forkPlan).
// So a START mines when some card may want a coin there is an address for,
// and after the fork a rig with no FP8 build is a Quantus rig.
function planFor(settings) {
  const fork = {
    coin: settings.coin, forkActive: ctx.forkActive, fp8: fp8On(process.env),
    quantusOk: isValidQuantusAddress(settings.quantusAddress),
  };
  const quantus = quantusOnly(fork);
  const mode = settings.mode || DEFAULT_MODE;
  const mines = canMine(Object.assign({ cardCoins: settings.cardCoins, pearlOk: isValidAddress(settings.address) }, fork));
  return resolvePlan(quantus && mode !== 'llm' ? 'mining' : mode, {
    canMine: mines && minerSupported(process.platform),
    canLlm: true,
  });
}

// Plan the START again, as it stands: the fork changed what its cards mine.
function replan() {
  applyPlan(session.settings, true);
}

// Apply the compute mode: run the miner and/or the LLM per the plan. When the
// plan ends up running nothing (LLM-only mode with the VRAM gate refusing, or
// no engine at all), tell the renderer — otherwise its optimistic "running"
// state shows STOP for a session in which nothing runs.
async function runPlan() {
  // The session's settings as they are now, not as they were when this run was
  // queued: the network-share switch is saved live, and a run queued before it
  // was flipped would save the old value back. A run only starts for a START,
  // so there is a session here; one that ends later is checked after each await.
  const settings = session.settings;
  const epoch = ctx.miningEpoch;
  const mode = settings.mode || DEFAULT_MODE;
  // macOS can serve the local LLM but has no mining engine to run, so the miner
  // is refused up front rather than left to fail on a Linux binary it would have
  // downloaded first (see shared/platform). The note explains the gap; without
  // it an 'auto' Mac start silently serves inference only, and a 'mining' one
  // runs nothing at all with no reason on screen.
  const note = minerUnsupportedNote(process.platform, mode);
  if (note) send('miner:log', { level: 'warn', line: note });
  const plan = planFor(settings);
  // A model planned without the miner, that the miner must not join (see
  // llmReplan). It goes before the miner starts, so the two never share the
  // card, and startLlm below plans again beside the miner. Paused or not: an
  // idle-only pause sizes the model for the miner too.
  const replan = plan.llm && plan.miner ? llmReplan() : null;
  if (replan) {
    send('miner:log', { level: 'info', line: 'restarting the local LLM: ' + ctx.servingModel.name
      + ' was sized for the card alone, and the miner is joining it' });
    await stopLlmForReplan(requiredFreeMb(replan, LLM.miningReserveMb), epoch);
    if (epoch !== ctx.miningEpoch) return;
  }
  // Idle-only, while someone uses the computer: the miner waits, the LLM runs.
  // The model is still sized with the mining reserve, so mining can start
  // beside it later without reloading it.
  const mineNow = plan.miner && !(session && session.paused);
  if (mineNow) {
    // Start mining FIRST, then — when co-running — wait until the miner reports a
    // non-zero hashrate before starting the LLM. Spawning the process (or even
    // connecting to the pool) isn't enough proof mining works: the LLM loads its
    // model and warms up in a few seconds, so without this wait it goes live
    // first, and its GPU-layer budgeter reads free VRAM before mining has claimed
    // its share. Waiting for real TH/s confirms the GPU is mining and its VRAM is
    // allocated, so the LLM then sizes its offload to what's actually left.
    try {
      await startMiningOnce(settings, plan.llm);
    } catch (e) {
      send('miner:log', { level: 'error', line: 'start failed: ' + e.message });
    }
    if (plan.llm && ctx.miner && ctx.miner.isRunning()) await waitForMinerUp();
    // No card had anything to mine (a coin with no address, or the fork with no
    // Quantus address): the start said why, and the miner is not part of what
    // keeps this START going.
    if (ctx.noCardMines) plan.miner = false;
  } else {
    // A miner from an earlier plan can still be running: LLM-only picked while
    // mining, or idle-only holding mining back. Stop it. For LLM-only, mining is
    // no longer asked for, and the model is about to be served from these cards
    // with no mining reserve, under the miner's memory clock lock if it holds
    // one (on an RTX 5090 that slows the model).
    // Not stopMining(): the session goes on, so the epoch stays (this run still
    // starts the model) and the renderer is not told it stopped. Its mining
    // figures are zeroed instead, as an LLM-only start shows them.
    if (ctx.miner) {
      haltMiner();
      send('miner:stats', statsView(snapshot(initStats(Date.now()), Date.now())));
    }
    persistSettings(settings); // startMining persists; do it here when the miner is off
  }
  // STOP arrived during miner setup or the hashrate wait: don't bring the LLM up
  // (or re-touch node state) for a session the user has already stopped.
  if (epoch !== ctx.miningEpoch) return;
  // The CPU miner runs for any START that asks for it, whatever the GPU does,
  // and waits out an idle-only pause like the GPU miner.
  if (settings.cpuMining && !(session && session.paused)) ctx.cpu.start(settings);
  else ctx.cpu.stop();
  syncNodeName(settings);
  if (plan.llm) {
    const started = await startLlm(plan.miner ? LLM.miningReserveMb : 0, epoch).catch(() => false);
    // A run cut short by STOP or a pause ends nothing: the START it belonged to
    // is already gone, or the next run carries it on.
    if (epoch !== ctx.miningEpoch) return;
    // Nothing runs for this START: the card had no room for an LLM-only start.
    // End the START itself, not just the window's STOP button: a session left
    // behind kept the idle watcher going, which could bring the model back while
    // the window showed START.
    if (!started) endIfNothingRuns(plan.miner);
  } else {
    stopLlm();
    endIfNothingRuns(plan.miner);
  }
}

// End the START when nothing runs for it: `others` is what the caller knows
// still runs, besides the CPU miner. A START that idle-only has paused still
// stands, though its miners are off: mining comes back on the next resume. A
// Monero-only START with idle-only on used to end itself on its first pause.
function endIfNothingRuns(others) {
  if (others || ctx.cpu.running() || (session && session.paused)) return;
  endSession();
}

// Whether the local LLM runs, or a run is still bringing it up. A miner that
// gives up on its own leaves the START to these.
function llmRunsOrComing() {
  const fleet = ctx.fleet;
  return !!planRun || !!(fleet && (fleet.hasSpawned() || fleet.isReady()));
}

// Single-flight wrapper around runPlan. `miner:start` is a plain IPC event with
// no re-entry guard, and the Chat tab's START LLM button fires it on every click
// while the LLM isn't ready — so a user clicking through a failing LLM setup
// launched a fresh download+extract each time, all racing on the same scratch
// files. Concurrent calls now collapse into the run already in flight.
//
// Settings that arrived mid-run are replayed once afterwards ONLY if they differ
// from the run in flight: a click that flips mining → both (START LLM while the
// miner runs) genuinely changes the plan and swallowing it would leave the LLM
// off, while a repeated click with the same settings has nothing left to do.
let planRun = null;
let planActive = null;
let planQueued = null;

// `force` replays even the same settings once the run in flight ends. An idle
// pause and resume need it: the pause cancelled that run (miningEpoch), and the
// same settings now mean a different run, with or without the miner.
function applyPlan(settings, force) {
  if (planRun) {
    if (force || JSON.stringify(settings) !== planActive) planQueued = settings;
    return planRun;
  }
  planActive = JSON.stringify(settings);
  planRun = runPlan().finally(() => {
    planRun = null;
    planActive = null;
    const next = planQueued;
    planQueued = null;
    if (next) applyPlan(next);
    // An update that downloaded during the run may install now.
    installWhenHidden();
  });
  return planRun;
}

// The START the user pressed, kept until they press STOP: { settings, paused }.
// With "only mine when idle" on, the idle watcher pauses its mining while
// someone uses the computer and resumes it when they leave. The local LLM is
// not paused: it keeps serving chat, cluster jobs and the network share. The
// renderer keeps showing STOP throughout, because the user's START still stands.
let session = null;
let idleTimer = null;

function startSession(settings) {
  session = { settings, paused: false };
  ctx.appLog.markMining({ version: app.getVersion() });
  syncIdleWatcher();
  // Someone just pressed START, so with idle-only on the computer is in use:
  // mining waits for it to go idle instead of starting now.
  if (getPrefs().mineWhenIdle && sessionMines() && appPrefs.idleAction(idleState(), false) === 'pause') {
    pauseSession();
    return;
  }
  applyPlan(settings);
}

function endSession() {
  session = null;
  ctx.appLog.clearMining();
  // A pause or resume queued behind a slow start (a first model download can
  // take minutes) must not replay after STOP and start mining nobody asked for.
  planQueued = null;
  // And the cancelled run still in flight is no longer what's running, so a
  // START with the same settings must not be dropped as a duplicate of it.
  planActive = null;
  syncIdleWatcher();
  stopMining();
  stopLlm();
}

// Idle-only only ever holds back mining, GPU or CPU, so a START with neither
// in it (LLM mode with CPU mining off, say) has nothing to pause.
function sessionMines() {
  return planFor(session.settings).miner || !!session.settings.cpuMining;
}

function idleState() {
  return powerMonitor.getSystemIdleState(appPrefs.IDLE_AFTER_SEC);
}

// Stop mining for the user, keeping their START and the local LLM. Not
// stopMining(): that tells the renderer mining stopped. The epoch still moves,
// so a miner start in flight does not finish after the pause; that also cancels
// the LLM half of a run in flight, so the plan runs again without the miner,
// which brings the LLM up if it isn't already (and leaves it alone if it is).
function pauseSession() {
  session.paused = true;
  ctx.miningEpoch++;
  haltMiner();
  ctx.cpu.stop();
  applyPlan(session.settings, true);
  send('miner:stats', statsView(snapshot(initStats(Date.now()), Date.now())));
  send('miner:idle', { paused: true, idleAfterSec: appPrefs.IDLE_AFTER_SEC });
  send('miner:log', {
    level: 'info',
    line: 'paused while the computer is in use; mining starts again after '
      + appPrefs.IDLE_AFTER_SEC / 60 + ' minutes without keyboard or mouse input',
  });
}

function resumeSession() {
  session.paused = false;
  send('miner:idle', { paused: false, idleAfterSec: appPrefs.IDLE_AFTER_SEC });
  send('miner:log', { level: 'info', line: 'the computer is idle; mining' });
  if (!planRun) {
    applyPlan(session.settings, true);
    return;
  }
  // A run is still in flight: usually a paused START bringing up the model,
  // which on a first run downloads several GB. Mining doesn't wait for it. That
  // model was planned with the mining reserve, so the miner starts beside it
  // now, and the run goes on to finish the model.
  const plan = planFor(session.settings);
  if (plan.miner) {
    startMiningOnce(session.settings, plan.llm)
      .catch((e) => send('miner:log', { level: 'error', line: 'start failed: ' + e.message }));
  }
  ctx.cpu.start(session.settings);
}

function checkIdle() {
  if (!session.paused && !sessionMines()) return;
  const action = appPrefs.idleAction(idleState(), session.paused);
  if (action === 'pause') pauseSession();
  else if (action === 'resume') resumeSession();
}

// The watcher runs only while there is a START to pause and idle-only is on.
function syncIdleWatcher() {
  const wanted = !!session && getPrefs().mineWhenIdle;
  if (wanted && !idleTimer) idleTimer = setInterval(checkIdle, appPrefs.IDLE_POLL_MS);
  else if (!wanted && idleTimer) {
    clearInterval(idleTimer);
    idleTimer = null;
  }
}

// Settings changed one or more of the app's switches. Returns what now holds.
function setPrefs(change) {
  const next = Object.assign({}, change);
  // Starting with the computer is reported as it ended up: a dev run, or a
  // Linux build that is not an AppImage, cannot do it.
  if (typeof next.runAtStartup === 'boolean') next.runAtStartup = applyLoginItem(next.runAtStartup);
  const saved = savePrefs(next);
  // Idle-only switched off while paused: there is nothing left to wait for.
  if (!saved.mineWhenIdle && session && session.paused) resumeSession();
  syncIdleWatcher();
  return saved;
}

share({
  session: { get: () => session },
  planRun: { get: () => planRun },
}, { endSession, endIfNothingRuns, replan });

module.exports = { startSession, endSession, setPrefs, endIfNothingRuns, llmRunsOrComing };
