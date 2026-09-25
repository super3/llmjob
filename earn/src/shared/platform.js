'use strict';

// What the OS this copy is running on can actually do. macOS is why this module
// exists: the Pearl core is CUDA and Macs have no NVIDIA GPU, so a Mac cannot
// mine at all.
//
// The gate has to be explicit rather than implicit, because everything
// downstream treats "not Windows" as "Linux": engine.enginePackage falls back to
// the Linux table, engineBinaryName returns the bare `alpha-miner-<version>`
// name, and engineDownloadUrl builds a pool URL that resolves. So a Mac left
// ungated would happily download a Linux ELF, chmod +x it, and spawn something
// the kernel refuses to exec — a download, a wait, and an opaque failure, for a
// binary that was never going to run.
//
// Pure predicates over a platform string so both shells (Electron main and the
// headless CLI) ask the same question, and so it is testable without faking
// process.platform.

// Platforms with no mining engine to download.
const NO_MINER = ['darwin'];

function minerSupported(platform) {
  return NO_MINER.indexOf(platform) === -1;
}

// The one-line explanation for a user who pressed Start on a platform that has
// no engine, or '' when mining works here.
//
// The branch lives here rather than at the call sites so both shells say the
// same thing and main.js — which carries a coverage ratchet for its unreachable
// defensive paths — doesn't grow another one.
function minerUnsupportedNote(platform) {
  if (minerSupported(platform)) return '';
  return 'note: mining is not available on macOS — the Pearl core is CUDA, and'
    + ' Macs have no NVIDIA GPU to run it on.';
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
