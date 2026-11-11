'use strict';

const {
  CORE_OFFSET_AUTO, CLI_CORE_OFFSET_FLAG, GUI_CORE_OFFSET_ENV,
  parseCoreOffset, readCoreOffsetEnv, planCoreOffsets,
} = require('../src/shared/coreOffset');

// Cards as parseCudaCards returns them.
const card = (index, cap) => {
  const [major, minor] = cap.split('.').map(Number);
  return { index, major, minor, driverMajor: 610 };
};
// The mining list, as shared/gpu.planMinerGpus returns it.
const gpu = (index, name = 'NVIDIA GeForce RTX 4090') => ({ index, name });

const LINE = (name, gpus) => name + ': trying core clock offsets on ' + gpus + ', each step kept only if the card '
  + 'still computes right and mines faster (measured on one RTX 4090 only; it runs a card above the clock its maker set)';
const IGNORED = '--mine-core-offset ignored: the LLM co-runs with the miner, and a model is never served '
  + 'from a card running above its own clock';
const OFF = { trial: [], dropped: false, reason: null };

describe('the core clock offset switch', () => {
  test('is --mine-core-offset on the CLI and LLMJOB_MINE_CORE_OFFSET in the app, and its one value is auto', () => {
    expect(CLI_CORE_OFFSET_FLAG).toBe('--mine-core-offset');
    expect(GUI_CORE_OFFSET_ENV).toBe('LLMJOB_MINE_CORE_OFFSET');
    expect(CORE_OFFSET_AUTO).toBe('auto');
  });

  test('reads auto in any case, and 0 for off', () => {
    expect(parseCoreOffset('auto')).toEqual({ offset: 'auto' });
    expect(parseCoreOffset(' AUTO ')).toEqual({ offset: 'auto' });
    expect(parseCoreOffset('0')).toEqual({ offset: 0 });
    expect(parseCoreOffset(0)).toEqual({ offset: 0 });
  });

  // No offset is set that the trial has not checked on that card, so there is
  // no MHz to ask for. An empty value is not read as 0 or as auto.
  test('refuses a number of MHz, and anything else', () => {
    for (const bad of ['150', '+150', '50', '-50', '', ' ', 'on', 'true', '0.5', 'auto 150']) {
      expect([bad, parseCoreOffset(bad)]).toEqual([bad, { error: 'must be auto or 0' }]);
    }
  });

  describe('from the app\'s environment', () => {
    test('is off when the variable is unset, or there is no environment', () => {
      expect(readCoreOffsetEnv({})).toEqual({ offset: null, warning: null });
      expect(readCoreOffsetEnv(undefined)).toEqual({ offset: null, warning: null });
    });

    test('takes the values the flag takes', () => {
      expect(readCoreOffsetEnv({ LLMJOB_MINE_CORE_OFFSET: 'Auto' })).toEqual({ offset: 'auto', warning: null });
      expect(readCoreOffsetEnv({ LLMJOB_MINE_CORE_OFFSET: '0' })).toEqual({ offset: 0, warning: null });
    });

    test('ignores a value it cannot use, with a line that says the trial is off', () => {
      expect(readCoreOffsetEnv({ LLMJOB_MINE_CORE_OFFSET: '150' })).toEqual({
        offset: null,
        warning: 'LLMJOB_MINE_CORE_OFFSET=150 ignored (must be auto or 0); no core clock offset is tried',
      });
    });
  });
});

describe('which cards try a core clock offset', () => {
  // Off by default everywhere: with nothing asked for, or 0, no card is
  // planned and nothing is said, whatever the rig.
  test('none unless auto is asked for', () => {
    const rig = { cards: [card(0, '8.9')], gpus: [gpu(0)] };
    expect(planCoreOffsets(rig)).toEqual(OFF);
    expect(planCoreOffsets({ ...rig, requested: null })).toEqual(OFF);
    expect(planCoreOffsets({ ...rig, requested: 0 })).toEqual(OFF);
    expect(planCoreOffsets({ ...rig, requested: 0, llmCoRuns: true })).toEqual(OFF);
    expect(planCoreOffsets()).toEqual(OFF);
  });

  // The line has to say how little has been measured.
  test('an RTX 4090 under auto, with a line that says one RTX 4090 is all that was measured', () => {
    expect(planCoreOffsets({ requested: 'auto', cards: [card(0, '8.9')], gpus: [gpu(0)] })).toEqual({
      trial: [0], dropped: false, reason: LINE('--mine-core-offset auto', 'GPU 0'),
    });
  });

  test('names the switch its shell\'s operator can reach', () => {
    expect(planCoreOffsets({
      requested: 'auto', cards: [card(0, '8.9')], gpus: [gpu(0)], requestName: GUI_CORE_OFFSET_ENV,
    }).reason).toBe(LINE('LLMJOB_MINE_CORE_OFFSET=auto', 'GPU 0'));
  });

  // Compute 7.5 is RTX 20, 8.6 RTX 30, 8.9 RTX 40 and 12.x RTX 50. Not GTX 10
  // (6.1), Volta (7.0), the A100 (8.0), Hopper (9.0), or a card nvidia-smi gave
  // no compute capability for.
  test('RTX 20, 30, 40 and 50 cards, and no others', () => {
    const cards = [
      card(0, '7.5'), card(1, '8.6'), card(2, '8.9'), card(3, '12.0'), card(4, '12.1'),
      card(5, '6.1'), card(6, '7.0'), card(7, '8.0'), card(8, '9.0'),
    ];
    const gpus = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((i) => gpu(i, 'card ' + i));
    const plan = planCoreOffsets({ requested: 'auto', cards, gpus });
    expect(plan.trial).toEqual([0, 1, 2, 3, 4]);
    expect(plan.reason).toBe(LINE('--mine-core-offset auto', 'GPU 0, 1, 2, 3, 4'));
  });

  test('only the cards that mine', () => {
    const cards = [card(0, '8.9'), card(1, '8.9'), card(2, '8.6')];
    expect(planCoreOffsets({ requested: 'auto', cards, gpus: [gpu(1), gpu(2)] }).trial).toEqual([1, 2]);
    // No mining list: every card nvidia-smi listed may mine.
    expect(planCoreOffsets({ requested: 'auto', cards, gpus: [] }).trial).toEqual([0, 1, 2]);
    expect(planCoreOffsets({ requested: 'auto', cards }).trial).toEqual([0, 1, 2]);
  });

  test('with no card of those lines, leaves the clocks alone and says so', () => {
    expect(planCoreOffsets({ requested: 'auto', cards: [card(0, '6.1')], gpus: [gpu(0, 'NVIDIA GeForce GTX 1080 Ti')] }))
      .toEqual({
        trial: [],
        dropped: false,
        reason: '--mine-core-offset auto: no mining card is one a core clock offset is tried on '
          + '(RTX 20, 30, 40 and 50 cards are); core clocks left alone',
      });
  });

  // An overclock nothing checks must not sit under a served model.
  test('is dropped when the LLM co-runs, with a line that says why', () => {
    expect(planCoreOffsets({ requested: 'auto', cards: [card(0, '8.9')], gpus: [gpu(0)], llmCoRuns: true }))
      .toEqual({ trial: [], dropped: true, reason: IGNORED });
    expect(planCoreOffsets({
      requested: 'auto', cards: [card(0, '8.9')], gpus: [gpu(0)], llmCoRuns: true, requestName: GUI_CORE_OFFSET_ENV,
    }).reason).toBe('LLMJOB_MINE_CORE_OFFSET ignored: the LLM co-runs with the miner, and a model is never served '
      + 'from a card running above its own clock');
  });

  // Said before the LLM's line, as the memory clock's plan does: on an AMD rig
  // the LLM is not the reason.
  test('says so when there is no NVIDIA card, whether or not an LLM co-runs', () => {
    const NONE = {
      trial: [],
      dropped: false,
      reason: '--mine-core-offset auto: nvidia-smi lists no NVIDIA card, and the core clock offset is NVIDIA-only',
    };
    expect(planCoreOffsets({ requested: 'auto', cards: [], gpus: [] })).toEqual(NONE);
    expect(planCoreOffsets({ requested: 'auto', cards: null, gpus: null, llmCoRuns: true })).toEqual(NONE);
  });
});
