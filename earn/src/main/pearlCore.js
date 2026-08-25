'use strict';

const path = require('path');

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
  const req = opts.require || require;
  const resourcesPath = opts.resourcesPath || null;

  const candidates = [];
  if (resourcesPath) candidates.push(path.join(resourcesPath, 'native', 'pearl_core.node'));
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
