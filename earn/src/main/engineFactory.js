'use strict';

const http = require('http');
const { PearlEngine } = require('./pearlEngine');
const { SrbEngine, httpStatsFetcher } = require('./srbEngine');
const { selectMiner, pathDirsFrom } = require('../shared/minerSelect');

// Build whichever mining engine the settings ask for.
//
// Both entry points (earn-cli.js and main.js) construct a miner, and without
// this they would each grow the same three-way branch -- SRBMiner, our core, or
// nothing minable. Keeping the decision here means the callers keep exactly one
// branch ("did I get an engine?") and the interesting part is unit-testable
// without Electron, a GPU or a binary.
//
// Returns { miner, notes, coreMissing }:
//   miner        an engine, or null when this rig cannot mine what was asked for
//   notes        lines for the caller to log, in order. Always at least the
//                reason the selection came out the way it did, because "why is
//                it using that engine" is the first question anyone asks of a
//                rig that changed speed, and it should be answerable from the
//                log alone.
//   coreMissing  the native engine was chosen but pearl_core.node is absent.
//
// `requireCore` picks between the two callers' long-standing behaviours, which
// genuinely differ. The GUI (false) constructs the engine anyway and lets it
// announce the problem in the miner log, where the user is already looking. The
// CLI (true) refuses up front and builds nothing, so a 'mining' run can exit
// non-zero for systemd -- exit 0 there read as success and produced a silent
// ten-second restart loop that mined nothing. Choosing one here would break the
// other, so it is a parameter rather than a decision.
//
// A null miner is NOT automatically fatal: an 'auto'-mode run still has its LLM
// half. The caller decides, exactly as it already did for a missing core.
function createMinerEngine({
  settings = {}, createCore = null, env = {}, exists = () => false,
  connect, spawn, readTemps, requireCore = false,
} = {}) {
  const sel = selectMiner({
    miner: settings.miner,
    minerBin: settings.minerBin,
    env,
    pathDirs: pathDirsFrom(env.PATH),
    exists,
  });
  const notes = [sel.reason];

  if (sel.engine === 'srb') {
    return {
      miner: new SrbEngine({
        binPath: sel.bin,
        spawn,
        fetchStats: httpStatsFetcher(http),
      }),
      notes,
      coreMissing: false,
    };
  }

  // --miner srb on a rig with no binary. Nothing to fall back to: asking for a
  // specific engine and silently getting a slower one would misreport what the
  // rig is doing.
  if (sel.engine === null) return { miner: null, notes, coreMissing: false };

  const coreMissing = !createCore;
  if (coreMissing && requireCore) return { miner: null, notes, coreMissing };

  return {
    miner: new PearlEngine({ connect, createCore, readTemps }),
    notes,
    coreMissing,
  };
}

module.exports = { createMinerEngine };
