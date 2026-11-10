'use strict';

// Which mining cards try a core clock offset. Pure: the shells run nvidia-smi
// and pass what it said in here, and PearlMiner runs the trial
// (shared/coreOffsetTrial has the method, main/gpuClockOffset the mechanism).
//
// An offset moves a card's whole clock-to-voltage curve: at +100 MHz it runs
// each voltage 100 MHz faster than its maker set it. The fold is power-bound on
// a card at its power limit, so the same watts then buy more clock. On an
// RTX 4090 at 450 W: +50 MHz measured +1.1%, +100 MHz +2.5%, +150 MHz +4.3%,
// every step with 400 hits of 400 recomputed right (tuning/benchmark.md,
// "rig01, 10 November: core clock offset on the RTX 4090").
//
// That is one card. How far a card goes before it computes wrong differs from
// card to card, and a wrong fold still finds hashes that meet the target, so
// every share it sent would be rejected. So there is no fixed offset to ask
// for: the one value is `auto`, a trial in which every step has to show on that
// card that it computes right and is faster before it is kept. It is off unless
// asked for: --mine-core-offset auto on the CLI, LLMJOB_MINE_CORE_OFFSET=auto
// in the desktop app's environment (readCoreOffsetEnv, read by main.js on every
// start). 0 is "off", the same as not asking.
//
// The cards: NVIDIA's RTX 20, 30, 40 and 50 lines, told by compute capability.
// Not AMD cards, which have no NVML. And never a card an LLM shares: a model
// must not be served from a card running faster than its maker set it, with
// nothing checking what it computes.

const CORE_OFFSET_AUTO = 'auto';
// The compute capabilities a trial covers: 7.5 is RTX 20 (and the other Turing
// cards), 8.6 RTX 30, 8.9 RTX 40, and 12.x RTX 50 (see shared/coreVariant).
const TRIAL_COMPUTE = [{ major: 7, minor: 5 }, { major: 8, minor: 6 }, { major: 8, minor: 9 }];
const BLACKWELL_COMPUTE_MAJOR = 12;

// The two switches, by the name each shell's log lines call them.
const CLI_CORE_OFFSET_FLAG = '--mine-core-offset';
const GUI_CORE_OFFSET_ENV = 'LLMJOB_MINE_CORE_OFFSET';

// Read one request, from either switch. Returns { offset } -- 'auto' for the
// trial, 0 for off -- or { error } with the reason in the words the CLI's error
// line uses. A number of MHz is refused, not rounded to a step: no offset is set
// that the trial has not checked on that card. An empty value is refused too.
function parseCoreOffset(raw) {
  const text = String(raw).trim().toLowerCase();
  if (text === CORE_OFFSET_AUTO) return { offset: CORE_OFFSET_AUTO };
  if (text === '0') return { offset: 0 };
  return { error: 'must be auto or 0' };
}

// The GUI's switch, from the app's environment. Returns { offset, warning }:
// offset is null when the variable is unset or unusable, and warning is one
// line for the log when it is set but unusable. A typo leaves the trial off.
function readCoreOffsetEnv(env) {
  const raw = env ? env[GUI_CORE_OFFSET_ENV] : undefined;
  if (raw == null) return { offset: null, warning: null };
  const parsed = parseCoreOffset(raw);
  if (parsed.error) {
    return {
      offset: null,
      warning: GUI_CORE_OFFSET_ENV + '=' + raw + ' ignored (' + parsed.error + '); no core clock offset is tried',
    };
  }
  return { offset: parsed.offset, warning: null };
}

function listGpus(indices) {
  return 'GPU ' + indices.join(', ');
}

function covered(card) {
  return !!card && (card.major === BLACKWELL_COMPUTE_MAJOR
    || TRIAL_COMPUTE.some((c) => c.major === card.major && c.minor === card.minor));
}

// Decide which cards try an offset.
//   requested    --mine-core-offset: null when not given, 0 for off, 'auto' for
//                the trial.
//   cards        parseCudaCards' list, [{ index, major, minor, driverMajor }].
//   gpus         the mining list (shared/gpu.planMinerGpus), [{ index, name }].
//                Empty means every card nvidia-smi listed may mine.
//   llmCoRuns    true when a local LLM shares the cards for the whole run.
//   requestName  the switch the log lines name: the CLI's flag (the default) or
//                the GUI's environment variable.
// Returns { trial, dropped, reason }:
//   trial    the card indices that run the trial (PearlMiner runs it); [] for
//            none.
//   dropped  true when the request was refused (the LLM co-runs).
//   reason   one line for the log, or null when nothing was asked for.
function planCoreOffsets({ requested, cards, gpus, llmCoRuns, requestName = CLI_CORE_OFFSET_FLAG } = {}) {
  const none = (reason, dropped) => ({ trial: [], dropped: !!dropped, reason });
  if (requested !== CORE_OFFSET_AUTO) return none(null);

  const known = Array.isArray(cards) ? cards : [];
  const mining = Array.isArray(gpus) && gpus.length
    ? gpus.map((g) => g.index)
    : known.map((c) => c.index);
  // "--mine-core-offset auto" on the CLI, "LLMJOB_MINE_CORE_OFFSET=auto" in
  // the app: each as its operator would type it.
  const named = requestName + (requestName.startsWith('--') ? ' ' : '=') + CORE_OFFSET_AUTO;

  // Said first, as the memory clock's plan does: on an AMD rig the LLM is not
  // the reason.
  if (!mining.length) {
    return none(named + ': nvidia-smi lists no NVIDIA card, and the core clock offset is NVIDIA-only');
  }
  if (llmCoRuns) {
    return none(requestName + ' ignored: the LLM co-runs with the miner, and a model is never served '
      + 'from a card running above its own clock', true);
  }

  // A card whose compute capability nvidia-smi did not report gets no trial:
  // "unknown" is not one of the lines covered.
  const trial = mining.filter((index) => covered(known.find((c) => c.index === index)));
  if (!trial.length) {
    return none(named + ': no mining card is one a core clock offset is tried on '
      + '(RTX 20, 30, 40 and 50 cards are); core clocks left alone');
  }
  // The line says how little has been measured, so nobody reads the switch as
  // a tested setting for their card.
  return {
    trial, dropped: false,
    reason: named + ': trying core clock offsets on ' + listGpus(trial) + ', each step kept only if the card '
      + 'still computes right and mines faster (measured on one RTX 4090 only; it runs a card above the '
      + 'clock its maker set)',
  };
}

module.exports = {
  CORE_OFFSET_AUTO, CLI_CORE_OFFSET_FLAG, GUI_CORE_OFFSET_ENV,
  parseCoreOffset, readCoreOffsetEnv, planCoreOffsets,
};
