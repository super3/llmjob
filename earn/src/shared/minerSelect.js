'use strict';

// Which mining engine to run, decided from flags, env and what is actually on
// disk. Pure: every filesystem question arrives as an injected `exists`, so the
// whole decision table is unit-testable without a binary on the machine.
//
// PeakMiner is PREFERRED when present because it is materially faster on this
// hardware -- measured 143.6 TH/s against our own core's 111 TH/s on an RTX
// 5090, and still ~140.7 after its 2% dev fee, so it earns more even net of the
// fee. See earn/native/OPTIMIZATION.md for why our core is where it is.
//
// It is NOT bundled, and cannot be: PeakMiner ships "proprietary, all rights
// reserved; no reverse engineering / redistribution". Shipping it in a release,
// mirroring it, or downloading it on the operator's behalf are all things that
// licence does not give us. The operator installs it; we find it and use it.
// That is why 'auto' is a PREFERENCE rather than a requirement -- a rig with
// nothing installed still mines, on our own zero-fee core.

const CHOICES = ['auto', 'peak', 'native'];
const DEFAULT_MINER = 'auto';
const BIN_NAME = 'peakminer';
const DEV_FEE_PCT = 2.0;

function isValidMiner(v) {
  return CHOICES.includes(v);
}

function minerChoices() {
  return CHOICES.join(', ');
}

// Resolve the PeakMiner binary. Explicit flag, then env, then PATH -- so a rig
// can pin one exact build without touching PATH, and an explicit path that is
// absent resolves to nothing rather than silently falling through to a
// different binary than the one that was named.
function resolvePeakBin({
  minerBin = null, env = {}, pathDirs = [], exists = () => false,
} = {}) {
  const explicit = minerBin || env.PEAK_MINER_BIN || null;
  if (explicit) return exists(explicit) ? explicit : null;
  for (const dir of pathDirs) {
    if (!dir) continue;
    const candidate = dir.replace(/\/+$/, '') + '/' + BIN_NAME;
    if (exists(candidate)) return candidate;
  }
  return null;
}

// Split a PATH value into directories. Separate from resolvePeakBin so callers
// can pass an explicit list in tests and process.env.PATH in production.
function pathDirsFrom(pathValue, sep = ':') {
  return String(pathValue || '').split(sep).filter(Boolean);
}

// -> { engine: 'peak'|'native'|null, bin, reason }
//
// engine null means the operator asked for something this rig cannot do
// (--miner peak with no binary). The caller decides whether that is fatal,
// exactly as it already does for a missing pearl_core.node.
function selectMiner(opts = {}) {
  const choice = opts.miner || DEFAULT_MINER;

  if (choice === 'native') {
    return { engine: 'native', bin: null, reason: 'forced by --miner native' };
  }

  const bin = resolvePeakBin(opts);
  if (bin) {
    return { engine: 'peak', bin, reason: 'found PeakMiner at ' + bin };
  }

  if (choice === 'peak') {
    return {
      engine: null,
      bin: null,
      reason: 'PeakMiner was requested but no binary was found '
        + '(searched: --miner-bin, PEAK_MINER_BIN, PATH). '
        + 'Install it from https://peakminer.org -- it is proprietary and cannot be bundled.',
    };
  }

  return {
    engine: 'native',
    bin: null,
    reason: 'no PeakMiner binary found; using the built-in zero-fee core',
  };
}

module.exports = {
  CHOICES,
  DEFAULT_MINER,
  BIN_NAME,
  DEV_FEE_PCT,
  isValidMiner,
  minerChoices,
  resolvePeakBin,
  pathDirsFrom,
  selectMiner,
};
