'use strict';

// The GPU can mine and run inference at the same time (slower), so "how to use
// the GPU" is a mode the user picks. This is the pure policy: given the mode and
// what's currently possible, decide which engines should run. main.js applies the
// plan (start/stop miner + llama-server). Auto's demand-driven smarts land in
// Phase 4; for now Auto co-runs when the LLM is available.

const MODES = ['mining', 'both', 'llm', 'auto'];
const DEFAULT_MODE = 'mining';

function isValidMode(mode) {
  return MODES.indexOf(mode) !== -1;
}

// ctx: { canMine: bool (valid payout address), canLlm: bool (LLM enabled/ready) }
// Returns { miner: bool, llm: bool }.
function resolvePlan(mode, ctx = {}) {
  const canMine = !!ctx.canMine;
  const canLlm = !!ctx.canLlm;
  switch (mode) {
    case 'llm':
      return { miner: false, llm: canLlm };
    case 'both':
    case 'auto': // Phase 2: co-run whenever the LLM is available (Phase 4 makes this demand-driven)
      return { miner: canMine, llm: canLlm };
    case 'mining':
    default:
      return { miner: canMine, llm: false };
  }
}

module.exports = { MODES, DEFAULT_MODE, isValidMode, resolvePlan };
