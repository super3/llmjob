'use strict';

const path = require('path');
const { createRequire } = require('module');
const {
  CU12, CU13, FILES, LABELS, pickCoreVariant, isRuntimeError,
} = require('../shared/coreVariant');

// Loads the native PearlHash core (earn/native → a compiled `pearl_core.node`
// N-API addon) and adapts it to the small event interface the host drives:
//
//   core.setJob({ header, target, jobId })   // switch what the GPU searches
//   core.stop()                              // release the GPU, end the search
//   core.on('hit', hit)                      // a candidate at/under target
//   core.on('hashrate', thPerSec)            // periodic throughput
//   core.on('error', err)
//
// Returns null when the addon is not present. That is the EXPECTED state on any
// machine without a CUDA build of the core (including this dev box and, until the
// build lands, CI): the JS host is complete and tested, but there is nothing to
// mine with until native/ is compiled. The host treats a null core as a first-
// class "engine not built" condition and says so, rather than crashing on a
// missing require.
//
// The require is injected so the whole thing is unit-testable without a real
// .node file, and so the several candidate paths (a dev build tree vs. the
// packaged app's resources) can be probed in one place.
//
// There are two builds of the addon -- see shared/coreVariant for which one a
// rig gets and why. loadCore finds `pearl_core.node` (CUDA 12.8); coreFactory
// decides whether to try `pearl_core_cu13.node` first.

// What every probe shares: the require to load with, and the directories a
// packaged install keeps its addons in.
function loaderContext(opts) {
  // Not this module's own require: inside a packaged (SEA) binary the
  // bundler's require can only see the snapshot, and a native addon cannot be
  // dlopen'd out of a snapshot at all. createRequire anchored to the real
  // executable can load from the real filesystem in every packaging.
  const req = opts.require || createRequire(opts.execPath || process.execPath);
  const env = opts.env || process.env;
  const execDir = path.dirname(opts.execPath || process.execPath);
  const dirs = [];
  if (opts.resourcesPath) dirs.push(path.join(opts.resourcesPath, 'native'));
  // Packaged CLI: the release ships the cores BESIDE the executable (and the
  // HiveOS tarball unpacks them there), because process.resourcesPath is
  // Electron-only and a snapshot path cannot host a .node file.
  dirs.push(execDir);
  dirs.push(path.join(execDir, 'native'));
  return { req, env, dirs };
}

// The first candidate that loads and looks like our addon, as { addon, file },
// or null.
function firstLoadable(req, candidates) {
  for (const c of candidates) {
    try {
      const addon = req(c);
      if (addon && typeof addon.createCore === 'function') return { addon, file: c };
    } catch (e) {
      // Not at this path — try the next. A genuinely broken addon (present but
      // throwing on load) also lands here and falls through to null, which is
      // the right outcome: the host reports "not built" and nobody mines on a
      // core that would not load.
    }
  }
  return null;
}

// The CUDA 12.8 core: PEARL_CORE_PATH, then the packaged locations, then the
// dev build tree.
function findCore(opts) {
  const { req, env, dirs } = loaderContext(opts);
  const candidates = [];
  // An operator override first: it makes field diagnosis a one-liner, and it
  // lets a rig run a locally built core without touching the install.
  if (env.PEARL_CORE_PATH) candidates.push(env.PEARL_CORE_PATH);
  for (const d of dirs) candidates.push(path.join(d, FILES[CU12]));
  const dev = [
    path.join(__dirname, '..', '..', 'native', 'build', 'Release', FILES[CU12]),
    path.join(__dirname, '..', '..', 'native', 'build', 'Debug', FILES[CU12]),
  ];
  const found = firstLoadable(req, candidates.concat(dev));
  // A dev-tree build is whatever toolkit that box has -- the 5090 rig builds
  // its own with CUDA 13.3 -- so the log must not call it the 12.8 release.
  if (found && dev.includes(found.file)) found.dev = true;
  return found;
}

// The CUDA 13 core, from the packaged locations only. Nothing builds it into the
// dev tree (a local build is one pearl_core.node, for whatever toolkit that box
// has), and a rig running its own build points PEARL_CORE_PATH at it.
function findCu13Core(opts) {
  const { req, dirs } = loaderContext(opts);
  return firstLoadable(req, dirs.map((d) => path.join(d, FILES[CU13])));
}

function loadCore(opts = {}) {
  const found = findCore(opts);
  return found ? found.addon : null;
}

function errText(e) {
  return (e && e.message) || String(e);
}

// A factory the host calls to get a running core, or null when the addon is
// unavailable. Kept separate from loadCore so the host depends on a tiny surface
// (`createCore(profile, opts) -> core | null`) that a test can fake with a bare
// EventEmitter.
//
// `opts` is per-core: which card it opens, and which slice of the search space
// it owns. The host builds one core per card and fills those in — see
// shared/gpu.planMinerGpus. An addon built before those options existed ignores
// the second argument and behaves exactly as it did, which is what a rig running
// an older pearl_core.node needs.
//
// Which build: `opts.cards` is what nvidia-smi said about the rig
// (probe.detectCudaCards) and `opts.gpus` the cards that will mine; from those
// shared/coreVariant.pickCoreVariant chooses. A caller that passes neither gets
// the 12.8 build, as every rig did before there were two. PEARL_CORE_PATH still
// beats everything: an operator who names a file gets that file.
// `opts.log(level, line)` hears the choice, once, and any fallback.
function coreFactory(opts = {}) {
  const env = opts.env || process.env;
  const log = typeof opts.log === 'function' ? opts.log : () => {};
  const say = (level, variant, why) => log(level, 'Pearl core: ' + LABELS[variant] + ' · ' + why);
  const wrap = (addon) => (profile, coreOpts) => addon.createCore(profile, coreOpts || {});

  if (env.PEARL_CORE_PATH) {
    const found = findCore(opts);
    if (!found) return null;
    log('info', 'Pearl core: ' + found.file
      + (found.file === env.PEARL_CORE_PATH ? ' (PEARL_CORE_PATH)' : ' (PEARL_CORE_PATH did not load)'));
    return wrap(found.addon);
  }

  const pick = pickCoreVariant({ env, cards: opts.cards, gpus: opts.gpus });
  const cu13 = pick.variant === CU13 ? findCu13Core(opts) : null;

  if (!cu13) {
    const found = findCore(opts);
    if (!found) return null;
    if (found.dev) log('info', 'Pearl core: local build ' + found.file);
    else say('info', CU12, pick.variant === CU13 ? FILES[CU13] + ' is not installed' : pick.reason);
    return wrap(found.addon);
  }

  say('info', CU13, pick.reason);
  // The fallback. A static CUDA 13 runtime loads on any driver and fails at its
  // first CUDA call, which is inside createCore -- so that is where an old
  // driver the gate above did not catch shows up, and where the 12.8 build
  // takes over. Only until a CUDA 13 core has started: once one has, the
  // runtime demonstrably works here, and a later card's failure is that card's
  // own (usually no VRAM), which the 12.8 build would hit too.
  //
  // The 12.8 addon is loaded only then, so a rig that passed the gate never
  // has two CUDA runtimes in one process.
  let addon = cu13.addon;
  let settled = false;
  return (profile, coreOpts) => {
    const o = coreOpts || {};
    if (settled) return addon.createCore(profile, o);
    try {
      const core = addon.createCore(profile, o);
      settled = true;
      return core;
    } catch (e) {
      const cu12 = isRuntimeError(e) ? findCore(opts) : null;
      if (!cu12) throw e;
      say('warn', CU12, 'the CUDA 13 build could not start (' + errText(e) + ')');
      addon = cu12.addon;
      settled = true;
      return addon.createCore(profile, o);
    }
  };
}

module.exports = { loadCore, coreFactory };
