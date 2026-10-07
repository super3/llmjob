'use strict';

// Which mining cards get their memory clock locked, and at what. Pure: the
// shells run nvidia-smi and pass what it said in here, and PearlMiner takes the
// locks (main/gpuClocks has the mechanism and the measurements).
//
// The lock is the DEFAULT on the RTX 5090. The fold is power-bound on a capped
// card and barely touches DRAM, so memory clock is watts the SMs could use: on
// an RTX 5090 at its 600 W cap, 7001 MHz measured +8.0%. The runs, and why 7001
// and not lower, are in main/gpuClocks. The other Blackwell cards (compute
// 12.x) may gain the same way, since a hard power cap and a power-bound fold are
// shared across the line, but none has been measured, and the rented hosts the
// kernel work is tested on don't allow clock changes. So they stay at the
// driver's clock unless someone asks for a lock. A 5090 is told by its name in
// the mining list and its compute capability; both must agree. The CLI turns
// the default off with --mine-mem-clock 0; the GUI, which has no setting for it
// in the renderer, with LLMJOB_MINE_MEM_CLOCK=0 in the app's environment
// (readMemClockEnv, read by main.js on every start). Both take the same values,
// parsed by parseMemClockMhz below.
//
// An explicit request still wins everywhere, and an LLM co-running on the card
// still cancels either: llama-server is memory-bandwidth-bound, the opposite of
// the fold, so no card may serve a model with its memory clock cut.

const BLACKWELL_MINE_MEM_CLOCK_MHZ = 7001;
// RTX 50 and RTX PRO Blackwell cards are compute 12.x (see shared/coreVariant).
const BLACKWELL_COMPUTE_MAJOR = 12;
// The cards the default covers, by the name nvidia-smi gives them: "NVIDIA
// GeForce RTX 5090" and "... RTX 5090 D", not the RTX PRO 6000 or the rest of
// the RTX 50 line.
const DEFAULT_LOCK_NAME = /\bRTX 5090\b/i;

// The two switches, by the name each shell's log lines call them.
const CLI_MEM_CLOCK_FLAG = '--mine-mem-clock';
const GUI_MEM_CLOCK_ENV = 'LLMJOB_MINE_MEM_CLOCK';

// Bounds on a non-zero request, in MHz. The range is a typo guard, not a
// hardware table -- nvidia-smi and the driver decide what a card accepts. Below
// 100 is a GHz figure (`7` for 7001), above 30000 a kHz one.
const MEM_CLOCK_MIN_MHZ = 100;
const MEM_CLOCK_MAX_MHZ = 30000;

// The CLI's line for a request dropped because the LLM co-runs. Kept word for
// word: it is what the docs and an operator's grep know.
const CORUN_IGNORED_TAIL = ' ignored: the LLM co-runs with the miner and needs full memory bandwidth';
const CORUN_IGNORED = CLI_MEM_CLOCK_FLAG + CORUN_IGNORED_TAIL;

// Read one request, from either switch. Returns { mhz } -- 0 for "leave the
// driver's clock", else the MHz to lock -- or { error } with the reason in the
// words the CLI's error line uses. An empty value is refused rather than read
// as 0: Number('') is 0, and a cleared setting must not silently switch the
// default off.
function parseMemClockMhz(raw) {
  const text = String(raw).trim();
  const mhz = text === '' ? NaN : Number(text);
  if (!Number.isInteger(mhz) || (mhz !== 0 && (mhz < MEM_CLOCK_MIN_MHZ || mhz > MEM_CLOCK_MAX_MHZ))) {
    return { error: 'must be 0, or a whole number of MHz, ' + MEM_CLOCK_MIN_MHZ + '-' + MEM_CLOCK_MAX_MHZ };
  }
  return { mhz };
}

// The GUI's switch, from the app's environment. Returns { mhz, warning }: mhz
// is null when the variable is unset or unusable, and warning is one line for
// the log when it is set but unusable. A typo keeps the RTX 5090 default
// rather than silently turning it off, and the warning says so.
function readMemClockEnv(env) {
  const raw = env ? env[GUI_MEM_CLOCK_ENV] : undefined;
  if (raw == null) return { mhz: null, warning: null };
  const parsed = parseMemClockMhz(raw);
  if (parsed.error) {
    return {
      mhz: null,
      warning: GUI_MEM_CLOCK_ENV + '=' + raw + ' ignored (' + parsed.error + '); the RTX 5090 default stands',
    };
  }
  return { mhz: parsed.mhz, warning: null };
}

function listGpus(indices) {
  return 'GPU ' + indices.join(', ');
}

// Decide the per-card memory clock plan.
//   requestedMhz  --mine-mem-clock: null when not given, 0 for "leave the
//                 driver's clock", else the MHz to lock on every mining card.
//   cards         parseCudaCards' list, [{ index, major, minor, driverMajor }].
//   gpus          the mining list (shared/gpu.planMinerGpus), [{ index, name }].
//                 Empty means every card nvidia-smi listed may mine; their
//                 names are then unknown, so none gets the default.
//   llmCoRuns     true when a local LLM shares the cards for the whole run.
//   requestName   the switch the log lines name: the CLI's flag (the default)
//                 or the GUI's environment variable, so each shell's lines
//                 name the switch its operator can reach.
// Returns { byIndex, isDefault, dropped, reason }:
//   byIndex    { [cardIndex]: mhz } for the cards to lock; {} for none.
//   isDefault  true when byIndex came from the RTX 5090 default, not a request.
//   dropped    true when an explicit request was refused (the LLM co-runs).
//   reason     one line for the log, or null when nothing was asked for and
//              nothing applies.
function planMemClocks({ requestedMhz, cards, gpus, llmCoRuns, requestName = CLI_MEM_CLOCK_FLAG } = {}) {
  const known = Array.isArray(cards) ? cards : [];
  const mining = Array.isArray(gpus) && gpus.length
    ? gpus.map((g) => g.index)
    : known.map((c) => c.index);
  const requested = Number.isInteger(requestedMhz) && requestedMhz > 0 ? requestedMhz : null;
  const none = (reason, dropped) => ({ byIndex: {}, isDefault: false, dropped: !!dropped, reason });
  // "--mine-mem-clock 0" on the CLI, "LLMJOB_MINE_MEM_CLOCK=0" in the app: each
  // as its operator would type it.
  const named = (value) => requestName + (requestName.startsWith('--') ? ' ' : '=') + value;

  if (requestedMhz === 0) return none(named(0) + ': memory clocks left to the driver');

  // The default covers a 5090 only. A card whose compute capability or name
  // nvidia-smi did not report gets none: "unknown" is not "RTX 5090".
  const listed = Array.isArray(gpus) ? gpus : [];
  const nameOf = (index) => {
    const g = listed.find((x) => x && x.index === index);
    return (g && g.name) || '';
  };
  const byDefault = mining.filter((index) => {
    const card = known.find((c) => c.index === index);
    return !!card && card.major === BLACKWELL_COMPUTE_MAJOR && DEFAULT_LOCK_NAME.test(nameOf(index));
  });

  if (llmCoRuns) {
    if (requested) return none(requestName + CORUN_IGNORED_TAIL, true);
    if (!byDefault.length) return none(null);
    return none('memory clock left to the driver on ' + listGpus(byDefault)
      + ': the LLM co-runs with the miner and needs full memory bandwidth');
  }

  if (requested) {
    if (!mining.length) return none(named(requested) + ': nvidia-smi listed no GPU, nothing to lock');
    const byIndex = {};
    for (const index of mining) byIndex[index] = requested;
    return {
      byIndex, isDefault: false, dropped: false,
      reason: named(requested) + ' MHz on ' + listGpus(mining),
    };
  }

  if (!byDefault.length) return none(null);
  const byIndex = {};
  for (const index of byDefault) byIndex[index] = BLACKWELL_MINE_MEM_CLOCK_MHZ;
  // The line names the off switch, so a rig that regresses knows the way back.
  return {
    byIndex, isDefault: true, dropped: false,
    reason: 'memory clock ' + BLACKWELL_MINE_MEM_CLOCK_MHZ + ' MHz by default on ' + listGpus(byDefault)
      + ' (RTX 5090; ' + named(0) + ' leaves the driver\'s clock)',
  };
}

module.exports = {
  BLACKWELL_MINE_MEM_CLOCK_MHZ, BLACKWELL_COMPUTE_MAJOR, CLI_MEM_CLOCK_FLAG, GUI_MEM_CLOCK_ENV,
  MEM_CLOCK_MIN_MHZ, MEM_CLOCK_MAX_MHZ, CORUN_IGNORED,
  parseMemClockMhz, readMemClockEnv, planMemClocks,
};
