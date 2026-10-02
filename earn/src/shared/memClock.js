'use strict';

// Which mining cards get their memory clock locked, and at what. Pure: the
// shells run nvidia-smi and pass what it said in here, and PearlMiner takes the
// locks (main/gpuClocks has the mechanism and the measurements).
//
// The lock is the DEFAULT on Blackwell (compute 12.x). The fold is power-bound
// on a capped card and barely touches DRAM, so memory clock is watts the SMs
// could use: on an RTX 5090 at its 600 W cap, 7001 MHz measured +8.0%. The
// runs, and why 7001 and not lower, are in main/gpuClocks. Only the 5090 has
// been measured; the default still covers every compute 12.x card because what
// it relies on -- a hard power cap and a power-bound fold -- is shared across
// the line, not a 5090 trait. The GUI has no setting for the lock, so the
// default is the only way a GUI rig gets it; the CLI turns it off with
// --mine-mem-clock 0.
//
// An explicit request still wins everywhere, and an LLM co-running on the card
// still cancels either: llama-server is memory-bandwidth-bound, the opposite of
// the fold, so no card may serve a model with its memory clock cut.

const BLACKWELL_MINE_MEM_CLOCK_MHZ = 7001;
// RTX 50 and RTX PRO Blackwell cards are compute 12.x (see shared/coreVariant).
const BLACKWELL_COMPUTE_MAJOR = 12;

// The CLI's line for a request dropped because the LLM co-runs. Kept word for
// word: it is what the docs and an operator's grep know.
const CORUN_IGNORED = '--mine-mem-clock ignored: the LLM co-runs with the miner and needs full memory bandwidth';

function listGpus(indices) {
  return 'GPU ' + indices.join(', ');
}

// Decide the per-card memory clock plan.
//   requestedMhz  --mine-mem-clock: null when not given, 0 for "leave the
//                 driver's clock", else the MHz to lock on every mining card.
//   cards         parseCudaCards' list, [{ index, major, minor, driverMajor }].
//   gpus          the mining list (shared/gpu.planMinerGpus). Empty means every
//                 card nvidia-smi listed may mine.
//   llmCoRuns     true when a local LLM shares the cards for the whole run.
// Returns { byIndex, isDefault, dropped, reason }:
//   byIndex    { [cardIndex]: mhz } for the cards to lock; {} for none.
//   isDefault  true when byIndex came from the Blackwell default, not a request.
//   dropped    true when an explicit request was refused (the LLM co-runs).
//   reason     one line for the log, or null when nothing was asked for and
//              nothing applies.
function planMemClocks({ requestedMhz, cards, gpus, llmCoRuns } = {}) {
  const known = Array.isArray(cards) ? cards : [];
  const mining = Array.isArray(gpus) && gpus.length
    ? gpus.map((g) => g.index)
    : known.map((c) => c.index);
  const requested = Number.isInteger(requestedMhz) && requestedMhz > 0 ? requestedMhz : null;
  const none = (reason, dropped) => ({ byIndex: {}, isDefault: false, dropped: !!dropped, reason });

  if (requestedMhz === 0) return none('--mine-mem-clock 0: memory clocks left to the driver');

  // A card whose compute capability nvidia-smi did not report gets no default:
  // "unknown" is not "Blackwell".
  const blackwell = mining.filter((index) => {
    const card = known.find((c) => c.index === index);
    return !!card && card.major === BLACKWELL_COMPUTE_MAJOR;
  });

  if (llmCoRuns) {
    if (requested) return none(CORUN_IGNORED, true);
    if (!blackwell.length) return none(null);
    return none('memory clock left to the driver on ' + listGpus(blackwell)
      + ': the LLM co-runs with the miner and needs full memory bandwidth');
  }

  if (requested) {
    if (!mining.length) return none('--mine-mem-clock ' + requested + ': nvidia-smi listed no GPU, nothing to lock');
    const byIndex = {};
    for (const index of mining) byIndex[index] = requested;
    return {
      byIndex, isDefault: false, dropped: false,
      reason: '--mine-mem-clock ' + requested + ' MHz on ' + listGpus(mining),
    };
  }

  if (!blackwell.length) return none(null);
  const byIndex = {};
  for (const index of blackwell) byIndex[index] = BLACKWELL_MINE_MEM_CLOCK_MHZ;
  // "on the CLI", because the GUI prints this line too and has no such switch.
  return {
    byIndex, isDefault: true, dropped: false,
    reason: 'memory clock ' + BLACKWELL_MINE_MEM_CLOCK_MHZ + ' MHz by default on ' + listGpus(blackwell)
      + ' (Blackwell; --mine-mem-clock 0 on the CLI leaves the driver\'s clock)',
  };
}

module.exports = {
  BLACKWELL_MINE_MEM_CLOCK_MHZ, BLACKWELL_COMPUTE_MAJOR, CORUN_IGNORED, planMemClocks,
};
