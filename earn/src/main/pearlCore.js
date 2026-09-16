'use strict';

const path = require('path');
const { createRequire } = require('module');

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
function loadCore(opts = {}) {
  // Not this module's own require: inside a packaged (SEA) binary the
  // bundler's require can only see the snapshot, and a native addon cannot be
  // dlopen'd out of a snapshot at all. createRequire anchored to the real
  // executable can load from the real filesystem in every packaging.
  const req = opts.require || createRequire(opts.execPath || process.execPath);
  const resourcesPath = opts.resourcesPath || null;
  const env = opts.env || process.env;
  const execDir = path.dirname(opts.execPath || process.execPath);

  const candidates = [];
  // An operator override first: it makes field diagnosis a one-liner, and it
  // lets a rig run a locally built core without touching the install.
  if (env.PEARL_CORE_PATH) candidates.push(env.PEARL_CORE_PATH);
  if (resourcesPath) candidates.push(path.join(resourcesPath, 'native', 'pearl_core.node'));
  // Packaged CLI: the release ships pearl_core.node BESIDE the executable
  // (and the HiveOS tarball unpacks it there), because process.resourcesPath
  // is Electron-only and a snapshot path cannot host a .node file.
  candidates.push(path.join(execDir, 'pearl_core.node'));
  candidates.push(path.join(execDir, 'native', 'pearl_core.node'));
  candidates.push(path.join(__dirname, '..', '..', 'native', 'build', 'Release', 'pearl_core.node'));
  candidates.push(path.join(__dirname, '..', '..', 'native', 'build', 'Debug', 'pearl_core.node'));

  for (const c of candidates) {
    try {
      const addon = req(c);
      if (addon && typeof addon.createCore === 'function') return addon;
    } catch (e) {
      // Not at this path — try the next. A genuinely broken addon (present but
      // throwing on load) also lands here and falls through to null, which is
      // the right outcome: the host reports "not built" and nobody mines on a
      // core that would not load.
    }
  }
  return null;
}

// An operator's explicit choice of mining card, from PEARL_GPU_INDEX, or null
// when they haven't made one (the core then ranks the cards itself).
//
// It is an escape hatch, not the mechanism: the core's own ranking is what
// decides on every ordinary rig. But "the app picked the wrong card" is exactly
// the report we cannot reproduce from here, and a rig that can pin the card in
// one env var can answer it in one run. Same reasoning as PEARL_CORE_PATH above,
// and the same place to look for it.
//
// Anything that isn't a non-negative integer is ignored rather than passed on:
// the core reads a negative index as "choose for me", so a typo must not read as
// an instruction.
function parseDeviceIndex(env) {
  const raw = (env || process.env).PEARL_GPU_INDEX;
  if (raw == null || String(raw).trim() === '') return null;
  const n = Number(raw);
  if (!Number.isInteger(n) || n < 0) return null;
  return n;
}

// A factory the host calls to get a running core for one profile, or null when
// the addon is unavailable. Kept separate from loadCore so the host depends on a
// tiny surface (`createCore(profile) -> core | null`) that a test can fake with a
// bare EventEmitter.
//
// The card choice rides along here rather than through the host: it is a
// property of this machine, like the addon's path, and every caller would
// otherwise have to remember to pass it. An addon built before `device` existed
// ignores the second argument and behaves exactly as it did.
function coreFactory(opts = {}) {
  const addon = loadCore(opts);
  if (!addon) return null;
  const deviceIndex = parseDeviceIndex(opts.env || process.env);
  return (profile) => addon.createCore(profile, { deviceIndex });
}

module.exports = { loadCore, coreFactory, parseDeviceIndex };
