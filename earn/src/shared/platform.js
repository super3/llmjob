'use strict';

// What the OS this copy is running on can actually do. macOS is why this module
// exists: the Mac build runs the local LLM — llama.cpp publishes a Metal build
// and shared/config.js points at it — but it cannot mine. The Pearl core is
// CUDA, Macs have no NVIDIA GPU, and no macOS core is built.
//
// The gate has to be explicit rather than implicit, because code that picks a
// file per platform tends to treat "not Windows" as "Linux". A Mac left ungated
// would go looking for a Linux core and fail with an opaque error.
//
// Pure predicates over a platform string so both shells (Electron main and the
// headless CLI) ask the same question, and so it is testable without faking
// process.platform.

// Platforms with no mining engine to download.
const NO_MINER = ['darwin'];

function minerSupported(platform) {
  return NO_MINER.indexOf(platform) === -1;
}

// The one-line explanation for a user whose compute mode asked for mining on a
// platform that has no engine, or '' when there is nothing to explain (mining
// works here, or the mode never wanted it).
//
// The branch lives here rather than at the call sites so both shells say the
// same thing and main.js — which carries a coverage ratchet for its unreachable
// defensive paths — doesn't grow another one. Without the line the plan simply
// drops the miner and the run looks broken: an 'auto' Mac start would serve
// inference and never mention that the mining half was skipped, and a 'mining'
// Mac start would run nothing at all with no reason given.
function minerUnsupportedNote(platform, mode) {
  if (minerSupported(platform)) return '';
  if (mode === 'llm') return ''; // this mode never asked to mine
  return 'note: mining is not available on macOS — the Pearl core is CUDA, and'
    + ' Macs have no NVIDIA GPU to run it on.'
    + (mode === 'mining'
      ? ' Switch the compute mode to LLM to serve the local model instead.'
      : ' The local LLM runs as usual — this Mac serves inference, not Pearl.');
}

// Can electron-updater actually install an update here?
//
// No on macOS: Squirrel.Mac verifies that the downloaded bundle's code signature
// matches the running app's, and the Mac build is only ad-hoc signed (there is
// no Apple Developer ID — see scripts/mac-adhoc-sign.mjs). Every check would end
// in a download the updater then refuses to apply, so the app doesn't wire the
// updater at all there and points at the Releases page instead. Windows (NSIS)
// and Linux (AppImage) are unaffected.
function autoUpdateSupported(platform) {
  return platform !== 'darwin';
}

module.exports = { NO_MINER, minerSupported, minerUnsupportedNote, autoUpdateSupported };
