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

// A factory the host calls to get a running core for one profile, or null when
// the addon is unavailable. Kept separate from loadCore so the host depends on a
// tiny surface (`createCore(profile) -> core | null`) that a test can fake with a
// bare EventEmitter.
function coreFactory(opts = {}) {
  const addon = loadCore(opts);
  if (!addon) return null;
  return (profile) => addon.createCore(profile);
}

module.exports = { loadCore, coreFactory };
