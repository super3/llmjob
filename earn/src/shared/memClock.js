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
//
// `auto` asks for a trial instead of a fixed clock (shared/memClockTrial). The
// RTX 40 cards and the rest of the Blackwell line then mine a few minutes with
// and without a lock and keep it only when it is faster. The lock tried is the
// one measured on the family's one tested card: 5001 MHz from an RTX 4090
// (+0.8% at 450 W, up to +1.7% under a lower power limit; the next speed down
// lost 11%), and the 5090's 7001 MHz for Blackwell. An RTX 5090 keeps its
// default under `auto`, with no trial: it is the measured card. Older cards
// get nothing, since no lock has been measured on any of them.

const BLACKWELL_MINE_MEM_CLOCK_MHZ = 7001;
// RTX 50 and RTX PRO Blackwell cards are compute 12.x (see shared/coreVariant).
const BLACKWELL_COMPUTE_MAJOR = 12;
// The cards the default covers, by the name nvidia-smi gives them: "NVIDIA
// GeForce RTX 5090" and "... RTX 5090 D", not the RTX PRO 6000 or the rest of
// the RTX 50 line.
const DEFAULT_LOCK_NAME = /\bRTX 5090\b/i;

// `auto`, and the lock it tries on an RTX 40 or RTX Ada card (compute 8.9).
const MEM_CLOCK_AUTO = 'auto';
const ADA_TRIAL_MEM_CLOCK_MHZ = 5001;
const ADA_COMPUTE = { major: 8, minor: 9 };

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

// Read one request, from either switch. Returns { mhz } -- 0 for "leave the
// driver's clock", 'auto' for the trial, else the MHz to lock -- or { error }
// with the reason in the words the CLI's error line uses. An empty value is refused rather than read
// as 0: Number('') is 0, and a cleared setting must not silently switch the
// default off.
function parseMemClockMhz(raw) {
  const text = String(raw).trim();
  if (text.toLowerCase() === MEM_CLOCK_AUTO) return { mhz: MEM_CLOCK_AUTO };
  const mhz = text === '' ? NaN : Number(text);
  if (!Number.isInteger(mhz) || (mhz !== 0 && (mhz < MEM_CLOCK_MIN_MHZ || mhz > MEM_CLOCK_MAX_MHZ))) {
    return { error: 'must be 0, auto, or a whole number of MHz, ' + MEM_CLOCK_MIN_MHZ + '-' + MEM_CLOCK_MAX_MHZ };
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
//                 driver's clock", 'auto' for a trial on the cards that have a
//                 lock to try, else the MHz to lock on every mining card.
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
//   trial      only on an `auto` plan: the card indices in byIndex that keep
//              their lock only if a trial shows it is faster (PearlMiner runs
//              it). The other cards in byIndex are locked outright.
function planMemClocks({ requestedMhz, cards, gpus, llmCoRuns, requestName = CLI_MEM_CLOCK_FLAG } = {}) {
  const known = Array.isArray(cards) ? cards : [];
  const mining = Array.isArray(gpus) && gpus.length
    ? gpus.map((g) => g.index)
    : known.map((c) => c.index);
  const auto = requestedMhz === MEM_CLOCK_AUTO;
  const requested = auto ? MEM_CLOCK_AUTO
    : (Number.isInteger(requestedMhz) && requestedMhz > 0 ? requestedMhz : null);
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

  // Said first: an AMD rig with the 5090's setting left over was told the LLM
  // was the reason (6 Nov), when the lock is nvidia-smi's and has no card here.
  if (requested && !mining.length) {
    return none(named(requested) + ': nvidia-smi lists no NVIDIA card, and the lock is NVIDIA-only');
  }

  if (llmCoRuns) {
    if (requested) return none(requestName + CORUN_IGNORED_TAIL, true);
    if (!byDefault.length) return none(null);
    return none('memory clock left to the driver on ' + listGpus(byDefault)
      + ': the LLM co-runs with the miner and needs full memory bandwidth');
  }

  if (auto) {
    // A 5090 keeps its measured default. The other cards with a lock to try:
    // compute 8.9 at 5001 MHz, the rest of compute 12.x at the 5090's 7001.
    const byIndex = {};
    const trial = [];
    for (const index of mining) {
      const card = known.find((c) => c.index === index);
      if (byDefault.includes(index)) {
        byIndex[index] = BLACKWELL_MINE_MEM_CLOCK_MHZ;
      } else if (card && card.major === BLACKWELL_COMPUTE_MAJOR) {
        byIndex[index] = BLACKWELL_MINE_MEM_CLOCK_MHZ;
        trial.push(index);
      } else if (card && card.major === ADA_COMPUTE.major && card.minor === ADA_COMPUTE.minor) {
        byIndex[index] = ADA_TRIAL_MEM_CLOCK_MHZ;
        trial.push(index);
      }
    }
    const parts = [];
    if (trial.length) {
      parts.push('trying a memory clock lock on ' + listGpus(trial) + ', kept only if the card mines faster');
    }
    if (byDefault.length) {
      parts.push(BLACKWELL_MINE_MEM_CLOCK_MHZ + ' MHz on ' + listGpus(byDefault) + ' (RTX 5090, as by default)');
    }
    if (!parts.length) {
      return none(named(MEM_CLOCK_AUTO) + ': no mining card has a memory clock lock to try '
        + '(RTX 40 and RTX 50 cards do); memory clocks left to the driver');
    }
    return {
      byIndex, trial, isDefault: false, dropped: false,
      reason: named(MEM_CLOCK_AUTO) + ': ' + parts.join('; '),
    };
  }

  if (requested) {
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
  MEM_CLOCK_MIN_MHZ, MEM_CLOCK_MAX_MHZ, MEM_CLOCK_AUTO, ADA_TRIAL_MEM_CLOCK_MHZ,
  parseMemClockMhz, readMemClockEnv, planMemClocks,
};
