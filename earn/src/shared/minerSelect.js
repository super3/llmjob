'use strict';

// Which mining engine to run, decided from flags, env and what is actually on
// disk. Pure: every filesystem question arrives as an injected `exists`, so the
// whole decision table is unit-testable without a binary on the machine.
//
// SRBMiner-Multi is PREFERRED when present because it is materially faster on
// this hardware -- measured 144.2 TH/s against our own core's 111 TH/s on an
// RTX 5090, same pool and session, and still ~141 after its 2% dev fee, so it
// earns more even net of the fee. See earn/native/OPTIMIZATION.md for why our
// core is where it is.
//
// It is NOT bundled, and cannot be: SRBMiner-Multi is closed source and ships
// with no licence granting redistribution (the licence text in its ReadMe
// covers the third-party libraries it embeds -- WinIO, WinRing0, RandomX -- not
// SRBMiner itself). With no grant there is no right to ship or mirror it, so the
// operator installs it and we find it. That is why 'auto' is a PREFERENCE rather
// than a requirement -- a rig with nothing installed still mines, on our own
// zero-fee core.

const CHOICES = ['auto', 'srb', 'native'];
const DEFAULT_MINER = 'auto';
// The release tarball ships the binary as `SRBMiner-MULTI`. The lowercase form
// is accepted too because a rig that symlinks it onto PATH usually lowercases.
const BIN_NAMES = ['SRBMiner-MULTI', 'srbminer-multi'];
const DEV_FEE_PCT = 2.0;

function isValidMiner(v) {
  return CHOICES.includes(v);
}

function minerChoices() {
  return CHOICES.join(', ');
}

// Resolve the SRBMiner binary. Explicit flag, then env, then PATH -- so a rig
// can pin one exact build without touching PATH, and an explicit path that is
// absent resolves to nothing rather than silently falling through to a
// different binary than the one that was named.
function resolveSrbBin({
  minerBin = null, env = {}, pathDirs = [], exists = () => false,
} = {}) {
  const explicit = minerBin || env.SRBMINER_BIN || null;
  if (explicit) return exists(explicit) ? explicit : null;
  for (const dir of pathDirs) {
    if (!dir) continue;
    const base = dir.replace(/\/+$/, '');
    for (const name of BIN_NAMES) {
      const candidate = base + '/' + name;
      if (exists(candidate)) return candidate;
    }
  }
  return null;
}

// Split a PATH value into directories. Separate from resolveSrbBin so callers
// can pass an explicit list in tests and process.env.PATH in production.
function pathDirsFrom(pathValue, sep = ':') {
  return String(pathValue || '').split(sep).filter(Boolean);
}

// -> { engine: 'srb'|'native'|null, bin, reason }
//
// engine null means the operator asked for something this rig cannot do
// (--miner srb with no binary). The caller decides whether that is fatal,
// exactly as it already does for a missing pearl_core.node.
function selectMiner(opts = {}) {
  const choice = opts.miner || DEFAULT_MINER;

  if (choice === 'native') {
    return { engine: 'native', bin: null, reason: 'forced by --miner native' };
  }

  const bin = resolveSrbBin(opts);
  if (bin) {
    return { engine: 'srb', bin, reason: 'found SRBMiner-Multi at ' + bin };
  }

  if (choice === 'srb') {
    return {
      engine: null,
      bin: null,
      reason: 'SRBMiner-Multi was requested but no binary was found '
        + '(searched: --miner-bin, SRBMINER_BIN, PATH). '
        + 'Install it from https://github.com/doktor83/SRBMiner-Multi/releases '
        + '-- it is closed source and cannot be bundled.',
    };
  }

  return {
    engine: 'native',
    bin: null,
    reason: 'no SRBMiner-Multi binary found; using the built-in zero-fee core',
  };
}

module.exports = {
  CHOICES,
  DEFAULT_MINER,
  BIN_NAMES,
  DEV_FEE_PCT,
  isValidMiner,
  minerChoices,
  resolveSrbBin,
  pathDirsFrom,
  selectMiner,
};
