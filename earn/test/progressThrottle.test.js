'use strict';

const { progressThrottle } = require('../src/shared/progressThrottle');

describe('progressThrottle', () => {
  test('the desktop app: 5 points apart, or every 2 s, and always 100%', () => {
    let t = 10000;
    const show = progressThrottle({ step: 5, everyMs: 2000, now: () => t });
    expect([0, 1, 4, 5, 7].map(show)).toEqual([0, null, null, 5, null]);
    t += 2000;
    expect(show(7)).toBe(7);          // a slow link still gets a line
    expect(show(7)).toBeNull();       // the same figure twice never does
    expect(show(100)).toBe(100);
    expect(show(100)).toBeNull();
  });

  test('the CLI beside a miner: one line per 10%, rounded down', () => {
    const show = progressThrottle({ step: 10, round: true });
    expect([0, 3, 9, 10, 19.5, 25, 20, 100].map(show)).toEqual([0, null, null, 10, null, 20, null, 100]);
  });

  test('an unknown, unreadable or negative percent shows nothing', () => {
    const show = progressThrottle({ step: 5 });
    for (const v of [null, undefined, 'x', -1]) expect(show(v)).toBeNull();
    expect(show('50')).toBe(50);
  });
});
