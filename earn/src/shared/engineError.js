'use strict';

// Turns a miner-engine launch failure into honest, user-facing messaging.
//
// The dominant real-world cause on Windows is antivirus: Windows Defender (and
// others) classify crypto miners as PUA and quarantine the binary in the
// moment between "engine ready" and spawn, so the launch fails with the file
// gone (missing) or a spawn error like `spawn UNKNOWN` / `ENOENT`. Rather than
// surface a cryptic error, we detect that shape and explain it.

// Spawn error codes Windows returns for a quarantined / blocked / vanished exe.
const AV_CODES = /ENOENT|EACCES|EPERM|UNKNOWN/i;

// True when a Windows launch failure most likely means antivirus removed or
// blocked the engine: either it vanished from disk after we staged it
// (`missing`), or the spawn error code/message matches the shapes above.
function isLikelyAntivirusBlock({ platform, missing, err } = {}) {
  if (platform !== 'win32') return false;
  if (missing) return true;
  const code = String((err && (err.code || err.errno)) || '');
  const msg = String((err && err.message) || '');
  return AV_CODES.test(code) || AV_CODES.test(msg);
}

// Returns { antivirus, ui, log } — a short line for the in-app status area and a
// fuller line for the log terminal.
function describeLaunchError(opts = {}) {
  if (isLikelyAntivirusBlock(opts)) {
    return {
      antivirus: true,
      ui: 'Antivirus blocked the mining engine. Allow it (or add an exclusion for the engine folder), then press Start again.',
      log: 'the mining engine was blocked or removed by antivirus — crypto miners are commonly flagged by Windows Defender. Allow the engine (Virus & threat protection → Protection history / Allowed threats) or add an exclusion for the engine folder, then press Start again.',
    };
  }
  const detail = (opts.err && opts.err.message) ? opts.err.message : String(opts.err || 'unknown error');
  return {
    antivirus: false,
    ui: 'The mining engine failed to start — see Logs.',
    log: 'failed to launch engine: ' + detail,
  };
}

module.exports = { AV_CODES, isLikelyAntivirusBlock, describeLaunchError };
