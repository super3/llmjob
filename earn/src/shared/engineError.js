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

// ── Engine setup (download/install) failures ────────────────────────────────
//
// The other failure users report is the engine never getting installed at all,
// and the dominant cause there is HTTPS trust: Node/Electron ships its own
// compiled-in root list and, unlike a browser, does no AIA chasing, so a rig
// dies with "unable to verify the first certificate" on a download the same
// machine fetches fine in Chrome. Two setups produce it — a server that sends
// its leaf without the intermediate, and a proxy/VPN/antivirus re-signing TLS
// with a private root that lives in the OS store Node never reads.
const TLS_TRUST_CODES = /UNABLE_TO_VERIFY_LEAF_SIGNATURE|UNABLE_TO_GET_ISSUER_CERT|SELF_SIGNED_CERT_IN_CHAIN|DEPTH_ZERO_SELF_SIGNED_CERT|CERT_UNTRUSTED/i;
// Some layers hand the failure on as a plain Error with the OpenSSL reason in
// the text and no code, so match the wording too.
const TLS_TRUST_TEXT = /unable to verify the first certificate|self[- ]signed certificate|unable to get (local )?issuer certificate/i;

// True when a download failed because the certificate chain could not be
// verified — i.e. a missing trust anchor, which retrying with more anchors can
// genuinely fix. Deliberately excludes expiry and hostname mismatches: those
// are real rejections that no extra CA should paper over.
function isTlsTrustError(err) {
  if (!err) return false;
  const code = String(err.code || err.errno || '');
  return TLS_TRUST_CODES.test(code) || TLS_TRUST_TEXT.test(String(err.message || ''));
}

// Returns { tls, ui, log } for a failed engine install.
//
// `downloadUrl` is the exact artifact THIS rig needs (driver-picked build, not
// the generic link) and `manualPath` the exact file it has to be saved as. Both
// matter: the first user to hit this downloaded the engine by hand, left it in
// ~/Downloads under the pool's unversioned name, and got the identical failure
// on the next start — the old hint printed a bare URL and never said where the
// file had to end up.
function describeSetupError({ err, downloadUrl, manualPath } = {}) {
  const detail = (err && err.message) ? err.message : String(err || 'unknown error');
  const manual = downloadUrl
    ? ' Manual install: download ' + downloadUrl
      + (manualPath ? ' and save it as ' + manualPath : '') + ', then start again.'
    : '';
  if (isTlsTrustError(err)) {
    return {
      tls: true,
      ui: 'The mining engine download failed an HTTPS certificate check — see Logs.',
      log: 'engine setup failed: ' + detail
        + ' — the pool download could not be verified over HTTPS, which usually means a proxy, VPN or'
        + ' antivirus is intercepting TLS, or the system CA store is out of date'
        + ' (Ubuntu/Debian: sudo apt install --reinstall ca-certificates).'
        + manual,
    };
  }
  return {
    tls: false,
    ui: 'Could not download or set up the mining engine — see Logs.',
    log: 'engine setup failed: ' + detail + '.' + manual,
  };
}

module.exports = {
  AV_CODES,
  isLikelyAntivirusBlock,
  describeLaunchError,
  isTlsTrustError,
  describeSetupError,
};
