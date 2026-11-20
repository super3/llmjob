'use strict';

const {
  BLACKWELL_COMPUTE_MAJOR, ADA_COMPUTE, isBlackwell, fp8Family, isAda, miningIndices, listGpus, switchName, readEnvSwitch,
} = require('../src/shared/cardFamily');

describe('cardFamily', () => {
  test('Blackwell is compute 12.x; data-center Blackwell (10.0) is not', () => {
    expect(BLACKWELL_COMPUTE_MAJOR).toBe(12);
    expect(isBlackwell({ major: 12, minor: 0 })).toBe(true);
    expect(isBlackwell({ major: 10, minor: 0 })).toBe(false);
    expect(isBlackwell(undefined)).toBe(false);
  });

  test('the FP8 families are Hopper 9.0 and Blackwell 10.x and 12.x', () => {
    expect(fp8Family({ major: 12, minor: 0 })).toBe(true);
    expect(fp8Family({ major: 10, minor: 0 })).toBe(true);
    expect(fp8Family({ major: 9, minor: 0 })).toBe(true);
    expect(fp8Family({ major: 8, minor: 9 })).toBe(false);
    expect(fp8Family({ major: 9, minor: 1 })).toBe(false);
    expect(fp8Family(undefined)).toBe(false);
  });

  test('Ada is compute 8.9 exactly', () => {
    expect(ADA_COMPUTE).toEqual({ major: 8, minor: 9 });
    expect(isAda({ major: 8, minor: 9 })).toBe(true);
    expect(isAda({ major: 8, minor: 6 })).toBe(false);
    expect(isAda(null)).toBe(false);
  });

  test('miningIndices: the mining list when there is one, else every listed card', () => {
    const cards = [{ index: 0 }, { index: 1 }];
    expect(miningIndices([{ index: 1, name: 'x' }], cards)).toEqual([1]);
    expect(miningIndices([], cards)).toEqual([0, 1]);
    expect(miningIndices(null, null)).toEqual([]);
  });

  test('log wording', () => {
    expect(listGpus([0, 2])).toBe('GPU 0, 2');
    expect(switchName('--mine-mem-clock', 0)).toBe('--mine-mem-clock 0');
    expect(switchName('LLMJOB_MINE_MEM_CLOCK', 'auto')).toBe('LLMJOB_MINE_MEM_CLOCK=auto');
  });

  test('readEnvSwitch reads a variable from an env that may be missing', () => {
    expect(readEnvSwitch({ A: '1' }, 'A')).toBe('1');
    expect(readEnvSwitch(undefined, 'A')).toBeUndefined();
  });
});
