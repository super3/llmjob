'use strict';

// The GPU can mine and run inference at the same time (slower), so "how to use
// the GPU" is a mode the user picks. This is the pure policy: given the mode and
// what's currently possible, decide which engines should run. main.js applies the
// plan (start/stop miner + llama-server). Auto's demand-driven smarts land in
// Phase 4; for now Auto co-runs when the LLM is available.

// Three modes (v2 mocks). There used to be a fourth, 'both' / "Mining+LLM",
// which resolved to EXACTLY what 'auto' resolves to — the same plan under a
// second name, offered as if it were a different choice. See normalizeMode for
// what happens to a rig that still has it stored.
const MODES = ['mining', 'llm', 'auto'];
// Shared fallback for BOTH clients: co-run mining and the LLM. Serving
// inference is the point of the network, and a mining-only default meant every
// headless rig — HiveOS flight sheets especially, which pass no --mode — mined
// silently and never served, with nothing in the log to say why. Auto still
// fails soft: the VRAM preflight skips the LLM on a card with no room, and a
// failed binary/model setup never takes the miner down.
const DEFAULT_MODE = 'auto';

// Legacy modes → the mode that means the same thing today. 'both' behaved
// identically to 'auto', so this migration cannot change what a rig does — but
// leaving it out would: an unrecognised mode falls through resolvePlan's default
// to mining-only, silently switching the LLM off for everyone who had picked
// Mining+LLM, with nothing in the log to explain it.
function normalizeMode(mode) {
  return mode === 'both' ? 'auto' : mode;
}

function isValidMode(mode) {
  return MODES.indexOf(normalizeMode(mode)) !== -1;
}

// ctx: { canMine: bool (valid payout address), canLlm: bool (LLM enabled/ready) }
// Returns { miner: bool, llm: bool }.
function resolvePlan(mode, ctx = {}) {
  const canMine = !!ctx.canMine;
  const canLlm = !!ctx.canLlm;
  switch (normalizeMode(mode)) {
    case 'llm':
      return { miner: false, llm: canLlm };
    case 'auto': // Phase 2: co-run whenever the LLM is available (Phase 4 makes this demand-driven)
      return { miner: canMine, llm: canLlm };
    case 'mining':
    default:
      return { miner: canMine, llm: false };
  }
}

module.exports = { MODES, DEFAULT_MODE, isValidMode, normalizeMode, resolvePlan };
