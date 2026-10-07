'use strict';

const {
  BLACKWELL_MINE_MEM_CLOCK_MHZ, BLACKWELL_COMPUTE_MAJOR, CLI_MEM_CLOCK_FLAG, GUI_MEM_CLOCK_ENV,
  MEM_CLOCK_MIN_MHZ, MEM_CLOCK_MAX_MHZ, CORUN_IGNORED, parseMemClockMhz, readMemClockEnv, planMemClocks,
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

describe('the RTX 5090 default', () => {
  // 7001 is the one value measured: 682 -> 742 MHz SM clock, +8.0% on a 600 W
  // 5090 (main/gpuClocks). The constant is where both shells and the docs
  // get it from, so it is pinned here.
  test('is 7001 MHz on compute 12.x', () => {
    expect(BLACKWELL_MINE_MEM_CLOCK_MHZ).toBe(7001);
    expect(BLACKWELL_COMPUTE_MAJOR).toBe(12);
  });

  // The line names the off switch the way its shell's operator types it: the
  // CLI's flag by default, the GUI's environment variable when main.js asks.
  test('locks every mining 5090 and says so, naming the off switch', () => {
    const plan = planMemClocks({ requestedMhz: null, cards: [RTX5090(0), RTX5090(1)], gpus: [gpu(0), gpu(1)] });
    expect(plan).toEqual({
      byIndex: { 0: 7001, 1: 7001 },
      isDefault: true,
      dropped: false,
      reason: 'memory clock 7001 MHz by default on GPU 0, 1 (RTX 5090; --mine-mem-clock 0 leaves the driver\'s clock)',
    });
    expect(planMemClocks({ cards: [RTX5090(0)], gpus: [gpu(0)], requestName: GUI_MEM_CLOCK_ENV }).reason)
      .toBe('memory clock 7001 MHz by default on GPU 0 (RTX 5090; LLMJOB_MINE_MEM_CLOCK=0 leaves the driver\'s clock)');
  });

  // A mixed rig: the 4090 is not power-bound the same way and was never
  // measured with the lock, so it is left alone.
  test('leaves every other card at the driver\'s clock', () => {
    const plan = planMemClocks({ cards: [RTX4090(0), RTX5090(1)], gpus: [gpu(0), gpu(1)] });
    expect(plan.byIndex).toEqual({ 1: 7001 });
    expect(plan.isDefault).toBe(true);
    expect(plan.reason).toContain('on GPU 1 ');
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
  test('with no mining list, locks nothing: no names, so no 5090', () => {
    const plan = planMemClocks({ cards: [RTX4090(0), RTX5090(1), RTX5090(2)], gpus: [] });
    expect(plan).toEqual({ byIndex: {}, isDefault: false, dropped: false, reason: null });
  });

  // Only the 5090 has been measured. The rest of the Blackwell line is compute
  // 12.x too, and stays at the driver's clock until a card has been run.
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
    expect(plan.reason).toContain('on GPU 0, 1 (RTX 5090;');
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

  // The request does not need nvidia-smi's compute capability, and it follows
  // the mining list even to a card nvidia-smi did not describe.
  test('applies to the mining list even when the cards are unknown', () => {
    expect(planMemClocks({ requestedMhz: 7001, cards: [], gpus: [gpu(3)] }).byIndex).toEqual({ 3: 7001 });
    expect(planMemClocks({ requestedMhz: 7001, cards: [RTX4090(0)], gpus: [] }).byIndex).toEqual({ 0: 7001 });
  });

  test('with no card listed anywhere has nothing to lock, and says so', () => {
    expect(planMemClocks({ requestedMhz: 7001, cards: [], gpus: [] })).toEqual({
      byIndex: {}, isDefault: false, dropped: false,
      reason: '--mine-mem-clock 7001: nvidia-smi listed no GPU, nothing to lock',
    });
  });

  // 0 is the off switch for the default: nothing is locked, even on a 5090.
  test('0 leaves the driver\'s clock everywhere', () => {
    const plan = planMemClocks({ requestedMhz: 0, cards: [RTX5090(0)], gpus: [gpu(0)] });
    expect(plan).toEqual({
      byIndex: {}, isDefault: false, dropped: false,
      reason: '--mine-mem-clock 0: memory clocks left to the driver',
    });
    // Also when the LLM co-runs: 0 is not a request to drop.
    expect(planMemClocks({ requestedMhz: 0, cards: [RTX5090(0)], gpus: [gpu(0)], llmCoRuns: true }).dropped).toBe(false);
  });

  // Anything that is not a positive whole number is "not given", so a shell
  // that passes an unparsed value through cannot lock a card at it.
  test('treats a non-numeric request as none', () => {
    expect(planMemClocks({ requestedMhz: '7001', cards: [RTX4090(0)], gpus: [gpu(0)] }).byIndex).toEqual({});
    expect(planMemClocks({ requestedMhz: -5, cards: [RTX5090(0)], gpus: [gpu(0)] }).byIndex).toEqual({ 0: 7001 });
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
      .toBe('LLMJOB_MINE_MEM_CLOCK=7001: nvidia-smi listed no GPU, nothing to lock');
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
      expect(parseMemClockMhz(bad)).toEqual({ error: 'must be 0, or a whole number of MHz, 100-30000' });
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
      warning: 'LLMJOB_MINE_MEM_CLOCK=off ignored (must be 0, or a whole number of MHz, 100-30000); the RTX 5090 default stands',
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
    expect(CORUN_IGNORED).toBe(IGNORED);
  });

  // Nobody asked for the default, so losing it is information, not a drop.
  test('skips the 5090 default and says why in one line', () => {
    const plan = planMemClocks({ cards: [RTX4090(0), RTX5090(1)], gpus: [gpu(0), gpu(1)], llmCoRuns: true });
    expect(plan).toEqual({
      byIndex: {}, isDefault: false, dropped: false,
      reason: 'memory clock left to the driver on GPU 1: the LLM co-runs with the miner and needs full memory bandwidth',
    });
  });

  test('has nothing to say on a rig with no Blackwell card and no request', () => {
    expect(planMemClocks({ cards: [RTX4090(0)], gpus: [gpu(0)], llmCoRuns: true }))
      .toEqual({ byIndex: {}, isDefault: false, dropped: false, reason: null });
  });
});
