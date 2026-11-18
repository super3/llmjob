'use strict';

// What the clock plans (shared/memClock, shared/coreOffset), the core picker
// (shared/coreVariant) and the rotation fold (shared/rotationFold) all ask about
// a card and a switch. Each kept its own copy before.

// RTX 50 and RTX PRO Blackwell cards are compute 12.x. Data-center Blackwell
// (B200) is 10.0: also Blackwell, but a different major, and it must never be
// matched as compute 12.x is (see shared/coreVariant).
const BLACKWELL_COMPUTE_MAJOR = 12;
// RTX 40 and RTX Ada cards are compute 8.9 exactly; 8.6 is Ampere (RTX 30).
const ADA_COMPUTE = { major: 8, minor: 9 };

// `card` is one of parseCudaCards' entries, { index, major, minor, driverMajor },
// or undefined for a card nvidia-smi did not report.
function isBlackwell(card) {
  return !!card && card.major === BLACKWELL_COMPUTE_MAJOR;
}

function isAda(card) {
  return !!card && card.major === ADA_COMPUTE.major && card.minor === ADA_COMPUTE.minor;
}

// The indices of the cards that mine: the mining list (shared/gpu.planMinerGpus,
// [{ index, name }]) when there is one, else every card nvidia-smi listed.
function miningIndices(gpus, cards) {
  if (Array.isArray(gpus) && gpus.length) return gpus.map((g) => g.index);
  return (Array.isArray(cards) ? cards : []).map((c) => c.index);
}

// "GPU 0, 1", for a log line.
function listGpus(indices) {
  return 'GPU ' + indices.join(', ');
}

// A switch with its value, as its operator would type it: "--mine-mem-clock 0"
// for a CLI flag, "LLMJOB_MINE_MEM_CLOCK=0" for an environment variable.
function switchName(name, value) {
  return name + (name.startsWith('--') ? ' ' : '=') + value;
}

// One environment variable, from an env that may be missing.
function readEnvSwitch(env, name) {
  return env ? env[name] : undefined;
}

module.exports = {
  BLACKWELL_COMPUTE_MAJOR, ADA_COMPUTE, isBlackwell, isAda, miningIndices, listGpus, switchName, readEnvSwitch,
};
