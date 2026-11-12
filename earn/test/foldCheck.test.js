'use strict';

// The rotation fold's check (shared/foldCheck): a card that opens on the fold
// has 24 of its hits recomputed before the miner trusts it.
// pearlMinerFoldCheck.test.js has what the miner does with the answers.

const {
  FOLD_CHECK_JOB, CHECK_HITS, CHECK_HITS_PER_S, CHECK_MS, createCheck, begin, hit, tick,
} = require('../src/shared/foldCheck');
const { checkBound, CHECK_HITS_PER_S: OFFSET_HITS_PER_S } = require('../src/shared/coreOffsetTrial');

describe('the rotation fold\'s check', () => {
  test('its numbers: 24 hits at 3 a second, a minute at most', () => {
    expect([CHECK_HITS, CHECK_HITS_PER_S, CHECK_MS, FOLD_CHECK_JOB]).toEqual([24, 3, 60000, 'fold-check-']);
  });

  test('it waits for the card\'s first hashrate sample', () => {
    const c = createCheck();
    expect(c).toEqual({ state: 'waiting', jobId: null, hits: 0, since: 0 });
    for (const th of [0, undefined, NaN, -1]) expect(tick(c, th, 1000)).toBeNull();
    expect(tick(c, 330.5, 1500)).toBe('begin');
    // The miner starts the job; until it has, every sample says begin.
    expect(c.state).toBe('waiting');
    begin(c, 'fold-check-1', 1500);
    expect(c).toEqual({ state: 'checking', jobId: 'fold-check-1', hits: 0, since: 1500 });
    expect(tick(c, 330.5, 2000)).toBeNull();
  });

  test('it passes at 24 right hits of its own job', () => {
    const c = createCheck();
    begin(c, 'fold-check-1', 0);
    for (let n = 1; n < CHECK_HITS; n += 1) expect(hit(c, 'fold-check-1', null)).toBeNull();
    // Another job's hit, right or wrong, changes nothing.
    expect(hit(c, 'fold-check-0', 'jackpot hash mismatch')).toBeNull();
    expect(c.hits).toBe(CHECK_HITS - 1);
    expect(hit(c, 'fold-check-1', null)).toBe('pass');
    expect(c.state).toBe('passed');
    // Nor does a hit that arrives after, or a sample.
    expect(hit(c, 'fold-check-1', 'jackpot hash mismatch')).toBeNull();
    expect(tick(c, 330, CHECK_MS * 2)).toBeNull();
    expect(c.state).toBe('passed');
  });

  test('one wrong hit fails it', () => {
    const c = createCheck();
    begin(c, 'fold-check-1', 0);
    expect(hit(c, 'fold-check-1', null)).toBeNull();
    expect(hit(c, 'fold-check-1', 'jackpot hash mismatch (fold/transcript)')).toBe('fail');
    expect([c.state, c.why]).toEqual(['failed', 'jackpot hash mismatch (fold/transcript)']);
    expect(hit(c, 'fold-check-1', null)).toBeNull();
    expect(tick(c, 330, CHECK_MS * 2)).toBeNull();
  });

  test('a hit before the check has started is no answer', () => {
    const c = createCheck();
    expect(hit(c, null, null)).toBeNull();
    expect(c.hits).toBe(0);
  });

  test('too few hits when the time is up fails it, and says how many', () => {
    const c = createCheck();
    begin(c, 'fold-check-1', 1000);
    for (let n = 0; n < 5; n += 1) hit(c, 'fold-check-1', null);
    expect(tick(c, 330, 1000 + CHECK_MS - 1)).toBeNull();
    expect(tick(c, 330, 1000 + CHECK_MS)).toBe('fail');
    expect([c.state, c.why]).toEqual(['failed', 'only 5 of the 24 hits it needs in 60 s']);
  });

  // A hit at k 65536 takes about 0.13 s to recompute, so the
  // check's job is set for 3 hits a second where an offset step's is set for 30.
  test('its job is ten times harder to hit than an offset step\'s', () => {
    const macs = 192 * 65536;
    const mine = checkBound(330.5, macs, CHECK_HITS_PER_S);
    expect(checkBound(330.5, macs) / mine).toBe(BigInt(OFFSET_HITS_PER_S / CHECK_HITS_PER_S));
    // 330.5 TH/s is 26.3 million attempts a second of 192 x 65536 MACs.
    const perSec = (330.5e12 / macs) * (Number(mine) / 2 ** 256);
    expect(perSec).toBeCloseTo(CHECK_HITS_PER_S, 3);
  });
});
