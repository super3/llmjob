'use strict';

// Safety checks for the local LLM on a rig that runs it on "auto", and the
// state the network board shows for it.
//
// HiveOS rigs run auto by default. Many of them have a weak CPU (a 2-thread
// Celeron for 6 or 8 cards), little RAM and a small USB drive, and a model that
// fights the miner for those costs hashrate. So before the LLM starts, the rig
// has to have room for it (preflight), and once it runs, a watch turns it off
// for the rest of the run when it fails or gets in mining's way (createLlmWatch,
// driven by createLlmGuard). Mining never waits on any of this: every check
// fails safe by leaving the LLM off, and none of them touches the miner.
//
// The reasons below go to the log and the HiveOS stats. The board gets one
// short code for them (llmState).

const { MODES, DEFAULT_MODE, normalizeMode } = require('./llmMode');

const LIMITS = {
  // CPU threads llama-server needs on top of mining's: one per mining card
  // (each card's search thread) and the Monero miner's threads. With full GPU
  // offload it still samples, tokenizes and answers HTTP on the CPU.
  llmThreads: 2,
  // Free RAM before the model loads. The weights go to VRAM, but llama-server
  // reads them through RAM and keeps its own buffers there.
  minFreeRamMb: 2048,
  // Free disk left over after a model download, so the HiveOS drive never fills.
  diskMarginMb: 2048,
  // A start that has not answered /health in this long has failed.
  readyTimeoutMs: 15 * 60 * 1000,
  // Crashes within one start before it counts as crashing again and again.
  crashLimit: 3,
  // Health probes in a row that fail or time out.
  healthMisses: 3,
  healthTimeoutMs: 10000,
  healthEveryMs: 60000,
  // Generation rates in a row below this many tokens a second.
  minTokensPerSec: 2,
  slowSamples: 3,
  // A card shared with the LLM hashing below this share of its rate before the
  // LLM started, for this many checks in a row (one a minute).
  hashDropRatio: 0.75,
  hashDropChecks: 5,
  hashEveryMs: 60000,
  // Rejected shares since the LLM started: at least this many, and more than
  // this share of the new shares (or twice the rig's share before, if higher).
  minRejects: 5,
  rejectShare: 0.1,
  // The process's own timer running this late, this many ticks in a row: the
  // CPU is so busy that the miner's reports and watchdogs lag too.
  lagMs: 2000,
  lagChecks: 3,
  tickMs: 5000,
  // How long after the LLM stops the one retry starts, so its VRAM is free.
  retryDelayMs: 60000,
};

// Why the LLM is off, as the board stores it, and the words the log uses.
const REASONS = {
  vram: 'not enough free VRAM',
  'weak-cpu': 'not enough CPU threads',
  'low-ram': 'not enough free RAM',
  'low-disk': 'not enough free disk space',
  'start-failed': 'it failed to start',
  crashed: 'it kept crashing',
  health: 'it stopped answering health checks',
  'too-slow': 'it was too slow',
  'hashrate-drop': 'mining slowed down beside it',
  rejects: 'the pool rejected shares beside it',
  'cpu-lag': 'the CPU was too busy',
};

const plural = (n, word) => n + ' ' + word + (n === 1 ? '' : 's');

// Whether this rig has room to start the LLM beside mining. Returns null when
// it does, or { reason, message } for the first check that fails. A figure that
// could not be read (null) passes its check: the watch still catches trouble.
//   cpus             CPU threads this process may use
//   miningCards      GPUs mining, one search thread each
//   cpuMinerThreads  Monero threads, 0 when it is off
//   freeRamMb        available RAM
//   freeDiskMb       free space where the model is stored
//   needDiskMb       what the download still needs, 0 when it is on disk
function preflight(facts, limits) {
  const L = Object.assign({}, LIMITS, limits);
  const f = facts || {};
  const cards = Math.max(0, Number(f.miningCards) || 0);
  const xmr = Math.max(0, Number(f.cpuMinerThreads) || 0);
  const cpus = Number(f.cpus);
  if (Number.isFinite(cpus) && cpus > 0 && cpus < cards + xmr + L.llmThreads) {
    return {
      reason: 'weak-cpu',
      message: 'only ' + plural(cpus, 'CPU thread') + ' for ' + plural(cards, 'GPU')
        + (xmr ? ' and ' + plural(xmr, 'Monero thread') : '') + '; the LLM needs ' + L.llmThreads + ' more',
    };
  }
  const ram = f.freeRamMb == null ? NaN : Number(f.freeRamMb);
  if (Number.isFinite(ram) && ram < L.minFreeRamMb) {
    return { reason: 'low-ram', message: 'only ' + Math.round(ram) + ' MB of free RAM; the LLM needs ' + L.minFreeRamMb };
  }
  const disk = f.freeDiskMb == null ? NaN : Number(f.freeDiskMb);
  const need = Number(f.needDiskMb) || 0;
  if (need > 0 && Number.isFinite(disk) && disk < need + L.diskMarginMb) {
    return {
      reason: 'low-disk',
      message: 'only ' + Math.round(disk) + ' MB of free disk; the model needs ' + Math.round(need)
        + ' MB and ' + L.diskMarginMb + ' MB must stay free',
    };
  }
  return null;
}

// Free space in MB on the filesystem holding `dir`, or null when it can't be
// read. `dir` may not exist yet (no model downloaded), so the nearest folder
// above it that does is measured.
function freeDiskMb(fs, path, dir) {
  try {
    let d = dir;
    while (!fs.existsSync(d) && path.dirname(d) !== d) d = path.dirname(d);
    const s = fs.statfsSync(d);
    const mb = (Number(s.bavail) * Number(s.bsize)) / (1024 * 1024);
    return Number.isFinite(mb) ? Math.floor(mb) : null;
  } catch (e) {
    return null;
  }
}

// What one start of the LLM is watched for. `onTrip({ reason, message, retry })`
// is called at most once: `retry` is true when a second start might work (a
// crash, a failed start, a hung server) and false when it would only hit the
// same wall (too slow, mining slowed down, a busy CPU).
function createLlmWatch({ limits, now, onTrip }) {
  const L = Object.assign({}, LIMITS, limits);
  const clock = now || Date.now;
  const startedAt = clock();
  let ready = false;
  let tripped = false;
  let crashes = 0;
  let misses = 0;
  let slow = 0;
  let drops = 0;
  let lags = 0;
  let shareBase = null;
  const trip = (reason, message, retry) => {
    if (tripped) return true;
    tripped = true;
    onTrip({ reason, message, retry });
    return true;
  };
  return {
    isTripped: () => tripped,
    isReady: () => ready,
    ready() { ready = true; misses = 0; },
    startFailed(why) { return trip('start-failed', 'it failed to start' + (why ? ' (' + why + ')' : ''), true); },
    // The whole fleet died on its own.
    died(code) { return trip('crashed', 'llama-server exited (code ' + code + ')', true); },
    crashed() {
      crashes += 1;
      if (crashes < L.crashLimit) return false;
      return trip('crashed', 'llama-server crashed ' + crashes + ' times', true);
    },
    health(ok) {
      if (!ready || ok) { misses = 0; return false; }
      misses += 1;
      if (misses < L.healthMisses) return false;
      return trip('health', 'it missed ' + misses + ' health checks in a row', true);
    },
    tokens(tps) {
      const r = Number(tps);
      if (!(r > 0)) return false;
      slow = r < L.minTokensPerSec ? slow + 1 : 0;
      if (slow < L.slowSamples) return false;
      return trip('too-slow', 'only ' + r.toFixed(1) + ' tokens/s; the floor is ' + L.minTokensPerSec, false);
    },
    // The shared cards' hashrate now and before the LLM started. `paused` when
    // a prompt has paused or just resumed a card, which says nothing about
    // whether the two can share it.
    hashrate({ rate, baseline, paused }) {
      const base = Number(baseline);
      if (paused || !(base > 0)) return false;
      drops = Number(rate) < base * L.hashDropRatio ? drops + 1 : 0;
      if (drops < L.hashDropChecks) return false;
      return trip('hashrate-drop', 'the shared cards hash at ' + Math.round((Number(rate) / base) * 100)
        + '% of their rate before the LLM', false);
    },
    // The rig's accepted and rejected share totals. The first call sets the
    // baseline.
    shares({ accepted, rejected }) {
      const a = Number(accepted) || 0;
      const r = Number(rejected) || 0;
      if (!shareBase) { shareBase = { a, r }; return false; }
      const da = a - shareBase.a;
      const dr = r - shareBase.r;
      const before = shareBase.a + shareBase.r > 0 ? shareBase.r / (shareBase.a + shareBase.r) : 0;
      if (dr < L.minRejects || dr / (da + dr) <= Math.max(L.rejectShare, 2 * before)) return false;
      return trip('rejects', dr + ' of ' + (da + dr) + ' shares rejected since the LLM started', false);
    },
    lag(ms) {
      lags = Number(ms) >= L.lagMs ? lags + 1 : 0;
      if (lags < L.lagChecks) return false;
      return trip('cpu-lag', 'the timers ran ' + (Number(ms) / 1000).toFixed(1) + ' s late ' + lags + ' times in a row', false);
    },
    tick() {
      if (ready || clock() - startedAt < L.readyTimeoutMs) return false;
      return trip('start-failed', 'it did not answer within ' + Math.round(L.readyTimeoutMs / 60000) + ' minutes', true);
    },
  };
}

// The LLM's guard for one run: the preflight, a watch per start, the one
// retry, and why it is off. Off is for the rest of the run.
//   enabled  whether the checks apply (auto on HiveOS). Off, the guard only
//            records why the LLM is off, for the board.
//   stop()   stop the LLM now; must never throw into the miner
//   retry()  start it once more
//   schedule(fn, ms)  a timer for the retry
function createLlmGuard({ enabled, limits, now, log, stop, retry, schedule }) {
  const L = Object.assign({}, LIMITS, limits);
  let off = null;
  let retried = false;
  let watch = null;
  const turnOff = (reason, message) => {
    off = { reason, message };
    log('LLM off: ' + message + '. Mining carries on.');
  };
  const onTrip = (t) => {
    try { stop(); } catch (e) { /* already gone */ }
    if (t.retry && !retried) {
      retried = true;
      log('LLM: ' + t.message + '; trying once more in ' + Math.round(L.retryDelayMs / 1000) + ' s');
      schedule(() => { if (!off) retry(); }, L.retryDelayMs);
      return;
    }
    turnOff(t.reason, t.message);
  };
  return {
    enabled: !!enabled,
    limits: L,
    // Run the checks. True when the LLM may start.
    preflight(facts) {
      if (!enabled) return true;
      const p = preflight(facts, L);
      if (p) turnOff(p.reason, p.message);
      return !p;
    },
    // A start is under way: a new watch, or null when the checks don't apply.
    begin() {
      watch = enabled && !off ? createLlmWatch({ limits: L, now, onTrip }) : null;
      return watch;
    },
    watch: () => watch,
    // The LLM is off for a reason found outside the watch (no VRAM, a failed
    // download, a death on a run without checks). With checks on, a failure of
    // a start in progress goes to its watch, so it gets its retry.
    note(reason, message) {
      if (watch && !watch.isTripped() && (reason === 'start-failed' || reason === 'crashed')) {
        if (reason === 'start-failed') watch.startFailed(message); else watch.died(message);
        return;
      }
      if (!off) off = { reason, message: REASONS[reason] + (message == null || message === '' ? '' : ' (' + message + ')') };
    },
    // It came up: nothing to report as off any more.
    clear() { if (!enabled) off = null; },
    off: () => off,
  };
}

// The board's short code for each reason the LLM is off: vram (the card has
// too little), cpu (the rig's CPU, RAM or disk held it back) or error (it
// should run, but failed or hurt mining). The detail stays in the log.
const CODES = {
  vram: 'vram',
  'weak-cpu': 'cpu',
  'low-ram': 'cpu',
  'low-disk': 'cpu',
};

// The board's two fields, { mode, llm }:
//   mode  the owner's setting: auto, llm or mine (null for anything else)
//   llm   on (running, starting, or waiting for a request in demand mode),
//         off (not wanted: mine), vram, cpu or error (see CODES)
// `off` is the reason code the LLM is off, or null.
function llmState({ mode, off }) {
  const m = normalizeMode(mode);
  const shown = m === 'mining' ? 'mine' : (MODES.includes(m) ? m : null);
  if (m === 'mining') return { mode: shown, llm: 'off' };
  if (off) return { mode: shown, llm: CODES[off] || 'error' };
  return { mode: shown, llm: 'on' };
}

// The desktop app's, from its mode and its LLM status row (appLlm's
// llmStatus, whose `error` says why the LLM is not up). Settings saved before
// the mode existed have none, and run the default.
function desktopLlmState({ mode, status }) {
  const e = String((status && status.error) || '');
  let off = null;
  if (/^Needs .*VRAM/.test(e)) off = 'vram';
  else if (e) off = 'start-failed';
  return llmState({ mode: mode || DEFAULT_MODE, off });
}

module.exports = { LIMITS, REASONS, CODES, preflight, freeDiskMb, createLlmWatch, createLlmGuard, llmState, desktopLlmState };
