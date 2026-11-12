'use strict';

const {
  BLACKWELL_MINE_MEM_CLOCK_MHZ, BLACKWELL_COMPUTE_MAJOR, CLI_MEM_CLOCK_FLAG, GUI_MEM_CLOCK_ENV,
  MEM_CLOCK_MIN_MHZ, MEM_CLOCK_MAX_MHZ, parseMemClockMhz, readMemClockEnv, planMemClocks,
} = require('../src/shared/memClock');

// Cards as parseCudaCards returns them.
const card = (index, cap, driverMajor) => {
  const [major, minor] = cap.split('.').map(Number);
  return { index, major, minor, driverMajor: driverMajor == null ? 610 : driverMajor };
};
const RTX5090 = (index) => card(index, '12.0');
const RTX4090 = (index) => card(index, '8.9');
// The mining list, as shared/gpu.planMinerGpus returns it. The default goes by
// the name, so cards are named the way nvidia-smi names them.
const gpu = (index, name = 'NVIDIA GeForce RTX 5090') => ({ index, name });

const IGNORED = '--mine-mem-clock ignored: the LLM co-runs with the miner and needs full memory bandwidth';

// With no request, each RTX 5090 tries a lock at 7001 MHz and keeps it only if
// that card mines faster with it (the trial is shared/memClockTrial's, and
// PearlMiner runs it). The 5090 used to be locked outright, on one run on one
// card.
describe('the RTX 5090 default', () => {
  // 7001 is the one value measured: 682 -> 742 MHz SM clock, +8.0% on a 600 W
  // 5090 (main/gpuClocks). The constant is where both shells and the docs
  // get it from, so it is pinned here.
  test('is 7001 MHz on compute 12.x', () => {
    expect(BLACKWELL_MINE_MEM_CLOCK_MHZ).toBe(7001);
    expect(BLACKWELL_COMPUTE_MAJOR).toBe(12);
  });

  // The line names both ways out of the trial, the way its shell's operator
  // types them: the CLI's flag by default, the GUI's environment variable when
  // main.js asks.
  test('puts every mining 5090 on trial at 7001 MHz and says so, naming both switches', () => {
    const plan = planMemClocks({ requestedMhz: null, cards: [RTX5090(0), RTX5090(1)], gpus: [gpu(0), gpu(1)] });
    expect(plan).toEqual({
      byIndex: { 0: 7001, 1: 7001 },
      trial: [0, 1],
      isDefault: true,
      dropped: false,
      reason: 'trying a memory clock lock of 7001 MHz on GPU 0, 1 by default (RTX 5090), kept only if the card '
        + 'mines faster; --mine-mem-clock 7001 locks without a trial, --mine-mem-clock 0 leaves the driver\'s clock',
    });
    expect(planMemClocks({ cards: [RTX5090(0)], gpus: [gpu(0)], requestName: GUI_MEM_CLOCK_ENV }).reason)
      .toBe('trying a memory clock lock of 7001 MHz on GPU 0 by default (RTX 5090), kept only if the card mines '
        + 'faster; LLMJOB_MINE_MEM_CLOCK=7001 locks without a trial, LLMJOB_MINE_MEM_CLOCK=0 leaves the driver\'s clock');
  });

  // No card of the default is locked outright: the miner locks a planned card
  // at once unless the plan lists it for a trial.
  test('lists every card it names for a trial', () => {
    const rigs = [
      { cards: [RTX5090(0)], gpus: [gpu(0)] },
      { cards: [RTX5090(0), RTX5090(1), RTX5090(2)], gpus: [gpu(0), gpu(2)] },
      { cards: [RTX4090(0), RTX5090(1)], gpus: [gpu(0, 'NVIDIA GeForce RTX 4090'), gpu(1)] },
    ];
    for (const rig of rigs) {
      const plan = planMemClocks(rig);
      expect(plan.isDefault).toBe(true);
      expect(plan.trial).toEqual(Object.keys(plan.byIndex).map(Number));
      expect(plan.trial.length).toBeGreaterThan(0);
    }
  });

  // A mixed rig: the 4090 is not power-bound the same way and was never
  // measured with this lock, so it is left alone.
  test('leaves every other card at the driver\'s clock', () => {
    const plan = planMemClocks({ cards: [RTX4090(0), RTX5090(1)], gpus: [gpu(0), gpu(1)] });
    expect(plan.byIndex).toEqual({ 1: 7001 });
    expect(plan.trial).toEqual([1]);
    expect(plan.isDefault).toBe(true);
    expect(plan.reason).toContain('on GPU 1 by default');
  });

  test('is nothing on a rig with no Blackwell card, with nothing to say', () => {
    expect(planMemClocks({ cards: [RTX4090(0)], gpus: [gpu(0)] }))
      .toEqual({ byIndex: {}, isDefault: false, dropped: false, reason: null });
  });

  // nvidia-smi said nothing about the cards (too old a driver, or not there):
  // "unknown" is not "Blackwell".
  test('is nothing when the compute capability is unknown', () => {
    expect(planMemClocks({ cards: [], gpus: [gpu(0)] }).byIndex).toEqual({});
    expect(planMemClocks({ cards: [RTX5090(1)], gpus: [gpu(0)] }).byIndex).toEqual({});
    expect(planMemClocks({}).byIndex).toEqual({});
    expect(planMemClocks().byIndex).toEqual({});
    expect(planMemClocks({ cards: null, gpus: null }).reason).toBeNull();
  });

  // An empty mining list means the shell starts one core and lets it choose.
  // Without the list there are no names, so no card can be told to be a 5090.
  test('with no mining list, tries nothing: no names, so no 5090', () => {
    const plan = planMemClocks({ cards: [RTX4090(0), RTX5090(1), RTX5090(2)], gpus: [] });
    expect(plan).toEqual({ byIndex: {}, isDefault: false, dropped: false, reason: null });
  });

  // Only the 5090 has been measured. The rest of the Blackwell line is compute
  // 12.x too, and stays at the driver's clock unless `auto` asks.
  test('covers the RTX 5090 and 5090 D, not the rest of Blackwell', () => {
    const blackwell = [RTX5090(0), RTX5090(1), RTX5090(2), RTX5090(3), RTX5090(4)];
    const plan = planMemClocks({
      cards: blackwell,
      gpus: [
        gpu(0, 'NVIDIA GeForce RTX 5090'),
        gpu(1, 'NVIDIA GeForce RTX 5090 D'),
        gpu(2, 'NVIDIA GeForce RTX 5080'),
        gpu(3, 'NVIDIA GeForce RTX 5070 Ti'),
        gpu(4, 'NVIDIA RTX PRO 6000 Blackwell Workstation Edition'),
      ],
    });
    expect(plan.byIndex).toEqual({ 0: 7001, 1: 7001 });
    expect(plan.trial).toEqual([0, 1]);
    expect(plan.reason).toContain('on GPU 0, 1 by default (RTX 5090)');
  });

  test('is nothing on a rig of other Blackwell cards, with nothing to say', () => {
    expect(planMemClocks({ cards: [RTX5090(0)], gpus: [gpu(0, 'NVIDIA GeForce RTX 5060')] }))
      .toEqual({ byIndex: {}, isDefault: false, dropped: false, reason: null });
    expect(planMemClocks({ cards: [RTX5090(0)], gpus: [gpu(0, null)] }).byIndex).toEqual({});
  });

  // A name that says 5090 on a card nvidia-smi reports as something else is
  // not trusted: both have to agree.
  test('needs the compute capability to agree with the name', () => {
    expect(planMemClocks({ cards: [RTX4090(0)], gpus: [gpu(0, 'NVIDIA GeForce RTX 5090')] }).byIndex).toEqual({});
  });

  // PEARL_GPU_INDEX narrows the fleet to one card; the plan follows it.
  test('follows the mining list, not nvidia-smi\'s', () => {
    const plan = planMemClocks({ cards: [RTX5090(0), RTX5090(1)], gpus: [gpu(1)] });
    expect(plan.byIndex).toEqual({ 1: 7001 });
    expect(plan.trial).toEqual([1]);
  });
});

describe('an explicit --mine-mem-clock', () => {
  test('wins on every mining card, Blackwell or not', () => {
    const plan = planMemClocks({ requestedMhz: 810, cards: [RTX4090(0), RTX5090(1)], gpus: [gpu(0), gpu(1)] });
    expect(plan).toEqual({
      byIndex: { 0: 810, 1: 810 },
      isDefault: false,
      dropped: false,
      reason: '--mine-mem-clock 810 MHz on GPU 0, 1',
    });
  });

  // Asking for the default's own clock is still asking: the 5090 is locked at
  // once, from either switch. This is the way past the trial, for a rig that
  // knows the lock pays or whose mining runs are too short to finish one.
  test('7001 on an RTX 5090 locks it with no trial', () => {
    const rig = { cards: [RTX5090(0), RTX5090(1)], gpus: [gpu(0), gpu(1)] };
    const cli = planMemClocks({ ...rig, requestedMhz: 7001 });
    expect(cli).toEqual({
      byIndex: { 0: 7001, 1: 7001 }, isDefault: false, dropped: false,
      reason: '--mine-mem-clock 7001 MHz on GPU 0, 1',
    });
    expect(cli).not.toHaveProperty('trial');
    const app = planMemClocks({ ...rig, requestedMhz: 7001, requestName: GUI_MEM_CLOCK_ENV });
    expect(app).toEqual({
      byIndex: { 0: 7001, 1: 7001 }, isDefault: false, dropped: false,
      reason: 'LLMJOB_MINE_MEM_CLOCK=7001 MHz on GPU 0, 1',
    });
    expect(app).not.toHaveProperty('trial');
  });

  // The request does not need nvidia-smi's compute capability, and it follows
  // the mining list even to a card nvidia-smi did not describe.
  test('applies to the mining list even when the cards are unknown', () => {
    expect(planMemClocks({ requestedMhz: 7001, cards: [], gpus: [gpu(3)] }).byIndex).toEqual({ 3: 7001 });
    expect(planMemClocks({ requestedMhz: 7001, cards: [RTX4090(0)], gpus: [] }).byIndex).toEqual({ 0: 7001 });
  });

  test('with no card listed anywhere has nothing to lock, and says so', () => {
    expect(planMemClocks({ requestedMhz: 7001, cards: [], gpus: [] })).toEqual({
      byIndex: {}, isDefault: false, dropped: false,
      reason: '--mine-mem-clock 7001: nvidia-smi lists no NVIDIA card, and the lock is NVIDIA-only',
    });
    // An AMD rig with a co-running LLM: no NVIDIA card is the reason, not the LLM.
    expect(planMemClocks({ requestedMhz: 7001, cards: [], gpus: [], llmCoRuns: true }).reason)
      .toBe('--mine-mem-clock 7001: nvidia-smi lists no NVIDIA card, and the lock is NVIDIA-only');
  });

  // 0 is the off switch for the default: nothing is locked and nothing is
  // tried, even on a 5090.
  test('0 leaves the driver\'s clock everywhere', () => {
    const plan = planMemClocks({ requestedMhz: 0, cards: [RTX5090(0)], gpus: [gpu(0)] });
    expect(plan).toEqual({
      byIndex: {}, isDefault: false, dropped: false,
      reason: '--mine-mem-clock 0: memory clocks left to the driver',
    });
    expect(plan).not.toHaveProperty('trial');
    // Also when the LLM co-runs: 0 is not a request to drop.
    expect(planMemClocks({ requestedMhz: 0, cards: [RTX5090(0)], gpus: [gpu(0)], llmCoRuns: true }).dropped).toBe(false);
  });

  // Anything that is not a positive whole number is "not given", so a shell
  // that passes an unparsed value through cannot lock a card at it. A 5090
  // then gets its default, which is the trial.
  test('treats a non-numeric request as none', () => {
    expect(planMemClocks({ requestedMhz: '7001', cards: [RTX4090(0)], gpus: [gpu(0)] }).byIndex).toEqual({});
    expect(planMemClocks({ requestedMhz: -5, cards: [RTX5090(0)], gpus: [gpu(0)] }))
      .toMatchObject({ byIndex: { 0: 7001 }, trial: [0], isDefault: true });
    expect(planMemClocks({ requestedMhz: '7001', cards: [RTX5090(0)], gpus: [gpu(0)] }))
      .toMatchObject({ byIndex: { 0: 7001 }, trial: [0], isDefault: true });
  });

  // The GUI's lines name its switch as an assignment, the way an operator sets
  // an environment variable; the plan itself is the same.
  test('names the GUI\'s environment variable when asked to', () => {
    const gui = { requestName: GUI_MEM_CLOCK_ENV, cards: [RTX4090(0)], gpus: [gpu(0)] };
    expect(planMemClocks({ ...gui, requestedMhz: 0 }))
      .toEqual({ byIndex: {}, isDefault: false, dropped: false, reason: 'LLMJOB_MINE_MEM_CLOCK=0: memory clocks left to the driver' });
    expect(planMemClocks({ ...gui, requestedMhz: 7001 }))
      .toEqual({ byIndex: { 0: 7001 }, isDefault: false, dropped: false, reason: 'LLMJOB_MINE_MEM_CLOCK=7001 MHz on GPU 0' });
    expect(planMemClocks({ ...gui, requestedMhz: 7001, cards: [], gpus: [] }).reason)
      .toBe('LLMJOB_MINE_MEM_CLOCK=7001: nvidia-smi lists no NVIDIA card, and the lock is NVIDIA-only');
    expect(planMemClocks({ ...gui, requestedMhz: 7001, llmCoRuns: true }))
      .toEqual({ byIndex: {}, isDefault: false, dropped: true, reason: 'LLMJOB_MINE_MEM_CLOCK ignored: the LLM co-runs with the miner and needs full memory bandwidth' });
  });
});

// One parser for both switches: the CLI's flag (cliArgs.test.js has the flag's
// own cases) and the GUI's environment variable.
describe('parseMemClockMhz', () => {
  test('reads 0 and a whole number of MHz within the bounds', () => {
    expect(CLI_MEM_CLOCK_FLAG).toBe('--mine-mem-clock');
    expect(parseMemClockMhz('0')).toEqual({ mhz: 0 });
    expect(parseMemClockMhz(' 7001 ')).toEqual({ mhz: 7001 });
    expect(parseMemClockMhz(7001)).toEqual({ mhz: 7001 });
    expect(parseMemClockMhz(String(MEM_CLOCK_MIN_MHZ))).toEqual({ mhz: 100 });
    expect(parseMemClockMhz(String(MEM_CLOCK_MAX_MHZ))).toEqual({ mhz: 30000 });
  });

  // An empty value is refused, not read as 0: Number('') is 0, and a cleared
  // setting must not switch the default off by accident.
  test('refuses anything else, with the reason in the CLI\'s words', () => {
    for (const bad of ['', ' ', 'abc', '7', '99', '30001', '7001.5', '-7001', '7e3x']) {
      expect(parseMemClockMhz(bad)).toEqual({ error: 'must be 0, auto, or a whole number of MHz, 100-30000' });
    }
  });
});

// The GUI's switch, as main.js reads it on every start.
describe('readMemClockEnv', () => {
  test('is no request when the variable is unset', () => {
    expect(GUI_MEM_CLOCK_ENV).toBe('LLMJOB_MINE_MEM_CLOCK');
    expect(readMemClockEnv({})).toEqual({ mhz: null, warning: null });
    expect(readMemClockEnv(undefined)).toEqual({ mhz: null, warning: null });
  });

  test('reads 0 and a clock, with nothing to warn about', () => {
    expect(readMemClockEnv({ LLMJOB_MINE_MEM_CLOCK: '0' })).toEqual({ mhz: 0, warning: null });
    expect(readMemClockEnv({ LLMJOB_MINE_MEM_CLOCK: '7001' })).toEqual({ mhz: 7001, warning: null });
  });

  // A typo keeps the default rather than silently turning it off, and the
  // warning says both what was ignored and what stands.
  test('ignores a bad value and warns, keeping the default', () => {
    expect(readMemClockEnv({ LLMJOB_MINE_MEM_CLOCK: 'off' })).toEqual({
      mhz: null,
      warning: 'LLMJOB_MINE_MEM_CLOCK=off ignored (must be 0, auto, or a whole number of MHz, 100-30000); the RTX 5090 default stands',
    });
    expect(readMemClockEnv({ LLMJOB_MINE_MEM_CLOCK: '' }).mhz).toBeNull();
  });
});

// LLM decode is memory-bandwidth-bound, the opposite of the fold. A card that
// serves a model for the whole run keeps its memory clock, whoever asked.
describe('an LLM co-running with the miner', () => {
  test('drops an explicit request with the CLI\'s line, word for word', () => {
    const plan = planMemClocks({ requestedMhz: 7001, cards: [RTX5090(0)], gpus: [gpu(0)], llmCoRuns: true });
    expect(plan).toEqual({ byIndex: {}, isDefault: false, dropped: true, reason: IGNORED });
  });

  // Nobody asked for the default, so losing it is information, not a drop.
  // The 5090 is not locked and not tried: a trial would cut the memory clock
  // under the model for half its turns.
  test('skips the 5090 default, trial and all, and says why in one line', () => {
    const plan = planMemClocks({ cards: [RTX4090(0), RTX5090(1)], gpus: [gpu(0), gpu(1)], llmCoRuns: true });
    expect(plan).toEqual({
      byIndex: {}, isDefault: false, dropped: false,
      reason: 'memory clock left to the driver on GPU 1: the LLM co-runs with the miner and needs full memory bandwidth',
    });
    expect(plan).not.toHaveProperty('trial');
  });

  test('has nothing to say on a rig with no Blackwell card and no request', () => {
    expect(planMemClocks({ cards: [RTX4090(0)], gpus: [gpu(0)], llmCoRuns: true }))
      .toEqual({ byIndex: {}, isDefault: false, dropped: false, reason: null });
  });
});

// `auto`: the cards with a measured lock in their family try it, and keep it
// only where it is faster (shared/memClockTrial; PearlMiner runs the trial).
describe('--mine-mem-clock auto', () => {
  const { MEM_CLOCK_AUTO, ADA_TRIAL_MEM_CLOCK_MHZ } = require('../src/shared/memClock');
  const RTX3080 = (index) => card(index, '8.6');

  // 5001 MHz is the one lock measured on an RTX 4090: +0.8% at 450 W, and the
  // next memory speed down lost 11% (tuning/benchmark.md, "rig01, 10 November").
  test('is spelled auto and tries 5001 MHz on compute 8.9', () => {
    expect(MEM_CLOCK_AUTO).toBe('auto');
    expect(ADA_TRIAL_MEM_CLOCK_MHZ).toBe(5001);
  });

  test('is read from either switch, in any case', () => {
    expect(parseMemClockMhz('auto')).toEqual({ mhz: 'auto' });
    expect(parseMemClockMhz(' AUTO ')).toEqual({ mhz: 'auto' });
    expect(readMemClockEnv({ LLMJOB_MINE_MEM_CLOCK: 'Auto' })).toEqual({ mhz: 'auto', warning: null });
  });

  test('puts an RTX 40 card on trial at 5001 MHz', () => {
    expect(planMemClocks({
      requestedMhz: 'auto', cards: [RTX4090(0)], gpus: [gpu(0, 'NVIDIA GeForce RTX 4090')],
    })).toEqual({
      byIndex: { 0: 5001 },
      trial: [0],
      isDefault: false,
      dropped: false,
      reason: '--mine-mem-clock auto: trying a memory clock lock on GPU 0, kept only if the card mines faster',
    });
  });

  // Every RTX 40 and RTX 50 card is tried, a 5090 like the rest of the
  // Blackwell line at the 5090's clock. It used to keep its lock under `auto`
  // with no trial, as the measured card. Older cards get nothing.
  test('on a mixed rig: trials for RTX 40 and RTX 50 cards, a 5090 among them, nothing older', () => {
    const plan = planMemClocks({
      requestedMhz: 'auto',
      requestName: GUI_MEM_CLOCK_ENV,
      cards: [RTX4090(0), RTX5090(1), RTX5090(2), RTX3080(3)],
      gpus: [
        gpu(0, 'NVIDIA GeForce RTX 4090'), gpu(1), gpu(2, 'NVIDIA GeForce RTX 5080'),
        gpu(3, 'NVIDIA GeForce RTX 3080'),
      ],
    });
    expect(plan).toEqual({
      byIndex: { 0: 5001, 1: 7001, 2: 7001 },
      trial: [0, 1, 2],
      isDefault: false,
      dropped: false,
      reason: 'LLMJOB_MINE_MEM_CLOCK=auto: trying a memory clock lock on GPU 0, 1, 2, kept only if the card '
        + 'mines faster',
    });
  });

  // The same trial the default gives it. Asked for, so not `isDefault`: what
  // goes wrong with it is a warning.
  test('with only a 5090, puts it on trial like any other card', () => {
    expect(planMemClocks({ requestedMhz: 'auto', cards: [RTX5090(0)], gpus: [gpu(0)] })).toEqual({
      byIndex: { 0: 7001 },
      trial: [0],
      isDefault: false,
      dropped: false,
      reason: '--mine-mem-clock auto: trying a memory clock lock on GPU 0, kept only if the card mines faster',
    });
  });

  // `auto` goes by what nvidia-smi says the card computes with, not its name:
  // with no mining list there are no names, and the cards are still tried.
  test('needs no card names', () => {
    expect(planMemClocks({ requestedMhz: 'auto', cards: [RTX4090(0), RTX5090(1)], gpus: [] }))
      .toMatchObject({ byIndex: { 0: 5001, 1: 7001 }, trial: [0, 1] });
  });

  // No lock has been measured on RTX 30 or older, and a card whose compute
  // capability nvidia-smi did not report is not known to be anything.
  test('with no card that has a lock to try, leaves the clocks alone and says so', () => {
    const NONE = {
      byIndex: {},
      isDefault: false,
      dropped: false,
      reason: '--mine-mem-clock auto: no mining card has a memory clock lock to try '
        + '(RTX 40 and RTX 50 cards do); memory clocks left to the driver',
    };
    expect(planMemClocks({
      requestedMhz: 'auto', cards: [RTX3080(0)], gpus: [gpu(0, 'NVIDIA GeForce RTX 3080')],
    })).toEqual(NONE);
    expect(planMemClocks({
      requestedMhz: 'auto', cards: [RTX4090(0)], gpus: [gpu(1, 'NVIDIA GeForce RTX 4090')],
    })).toEqual(NONE);
  });

  test('is dropped when the LLM co-runs, like any request', () => {
    expect(planMemClocks({
      requestedMhz: 'auto', cards: [RTX4090(0)], gpus: [gpu(0, 'NVIDIA GeForce RTX 4090')], llmCoRuns: true,
    })).toEqual({ byIndex: {}, isDefault: false, dropped: true, reason: IGNORED });
    // On a 5090 too: no trial runs under a model.
    const plan = planMemClocks({ requestedMhz: 'auto', cards: [RTX5090(0)], gpus: [gpu(0)], llmCoRuns: true });
    expect(plan).toEqual({ byIndex: {}, isDefault: false, dropped: true, reason: IGNORED });
    expect(plan).not.toHaveProperty('trial');
  });

  test('says so when there is no NVIDIA card', () => {
    expect(planMemClocks({ requestedMhz: 'auto', cards: [], gpus: [] }).reason)
      .toBe('--mine-mem-clock auto: nvidia-smi lists no NVIDIA card, and the lock is NVIDIA-only');
  });
});

// On the rotation fold (--rotation-fold, LLMJOB_ROTATION_FOLD=1) an RTX 40 card
// tries a lock at 810 MHz by default and under `auto`: that fold reads almost no
// DRAM, and on rig01's RTX 4090 810 MHz mined +1.05% against the driver's clock
// and 5001 +0.7% (12 November). Without the fold nothing about RTX 40 changes.
describe('the rotation fold on RTX 40', () => {
  const { ROTATION_ADA_MEM_CLOCK_MHZ } = require('../src/shared/memClock');
  const RTX3080 = (index) => card(index, '8.6');
  const g4090 = (index) => gpu(index, 'NVIDIA GeForce RTX 4090');

  test('is 810 MHz', () => {
    expect(ROTATION_ADA_MEM_CLOCK_MHZ).toBe(810);
  });

  test('puts each RTX 40 card on trial at 810 MHz by default, whatever its name, and says so', () => {
    expect(planMemClocks({
      requestedMhz: null, rotationFold: true, cards: [RTX4090(0), card(1, '8.9')], gpus: [g4090(0), gpu(1, 'NVIDIA RTX 4000 Ada Generation')],
    })).toEqual({
      byIndex: { 0: 810, 1: 810 },
      trial: [0, 1],
      isDefault: true,
      dropped: false,
      reason: 'trying a memory clock lock of 810 MHz on GPU 0, 1 (RTX 40 on the rotation fold) by default, kept only '
        + 'if the card mines faster; --mine-mem-clock <MHz> locks without a trial, --mine-mem-clock 0 leaves the '
        + 'driver\'s clock',
    });
  });

  test('names the GUI switch when main.js asks', () => {
    expect(planMemClocks({
      rotationFold: true, cards: [RTX4090(0)], gpus: [g4090(0)], requestName: GUI_MEM_CLOCK_ENV,
    }).reason).toBe('trying a memory clock lock of 810 MHz on GPU 0 (RTX 40 on the rotation fold) by default, kept '
      + 'only if the card mines faster; LLMJOB_MINE_MEM_CLOCK=<MHz> locks without a trial, LLMJOB_MINE_MEM_CLOCK=0 '
      + 'leaves the driver\'s clock');
  });

  test('leaves an RTX 40 card alone without the fold', () => {
    expect(planMemClocks({ requestedMhz: null, cards: [RTX4090(0)], gpus: [g4090(0)] }).byIndex).toEqual({});
    expect(planMemClocks({ requestedMhz: null, rotationFold: false, cards: [RTX4090(0)], gpus: [g4090(0)] }).reason)
      .toBeNull();
  });

  test('keeps the 5090 default beside it, and gives older cards nothing', () => {
    expect(planMemClocks({
      rotationFold: true, cards: [RTX4090(0), RTX5090(1), RTX3080(2)], gpus: [g4090(0), gpu(1), gpu(2, 'NVIDIA GeForce RTX 3080')],
    })).toEqual({
      byIndex: { 0: 810, 1: 7001 },
      trial: [1, 0],
      isDefault: true,
      dropped: false,
      reason: 'trying a memory clock lock of 7001 MHz on GPU 1 (RTX 5090) and 810 MHz on GPU 0 (RTX 40 on the '
        + 'rotation fold) by default, kept only if the card mines faster; --mine-mem-clock <MHz> locks without a '
        + 'trial, --mine-mem-clock 0 leaves the driver\'s clock',
    });
    expect(planMemClocks({ rotationFold: true, cards: [RTX3080(0)], gpus: [gpu(0, 'NVIDIA GeForce RTX 3080')] }))
      .toEqual({ byIndex: {}, isDefault: false, dropped: false, reason: null });
  });

  test('tries 810 in place of 5001 under auto', () => {
    expect(planMemClocks({ requestedMhz: 'auto', rotationFold: true, cards: [RTX4090(0), RTX5090(1)], gpus: [] }))
      .toEqual({
        byIndex: { 0: 810, 1: 7001 },
        trial: [0, 1],
        isDefault: false,
        dropped: false,
        reason: '--mine-mem-clock auto: trying a memory clock lock on GPU 0, 1, kept only if the card mines faster',
      });
  });

  test('gives way to an explicit request and to the LLM', () => {
    expect(planMemClocks({ requestedMhz: 0, rotationFold: true, cards: [RTX4090(0)], gpus: [g4090(0)] }).byIndex)
      .toEqual({});
    expect(planMemClocks({ requestedMhz: 5001, rotationFold: true, cards: [RTX4090(0)], gpus: [g4090(0)] }))
      .toEqual({ byIndex: { 0: 5001 }, isDefault: false, dropped: false, reason: '--mine-mem-clock 5001 MHz on GPU 0' });
    expect(planMemClocks({ rotationFold: true, llmCoRuns: true, cards: [RTX4090(0)], gpus: [g4090(0)] })).toEqual({
      byIndex: {}, isDefault: false, dropped: false,
      reason: 'memory clock left to the driver on GPU 0: the LLM co-runs with the miner and needs full memory bandwidth',
    });
  });
});
