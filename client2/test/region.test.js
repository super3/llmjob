'use strict';

const { pickFastestRegion } = require('../src/shared/region');

describe('pickFastestRegion', () => {
  test('picks the lowest-latency region', () => {
    expect(pickFastestRegion([
      { region: 'us1', ms: 80 },
      { region: 'eu1', ms: 25 },
      { region: 'sg1', ms: 140 },
    ])).toBe('eu1');
  });

  test('ignores unreachable regions (non-numeric ms)', () => {
    expect(pickFastestRegion([
      { region: 'us1', ms: null },
      { region: 'eu1', ms: undefined },
      { region: 'sg1', ms: 90 },
    ])).toBe('sg1');
  });

  test('keeps the first on a tie', () => {
    expect(pickFastestRegion([
      { region: 'us1', ms: 30 },
      { region: 'us2', ms: 30 },
    ])).toBe('us1');
  });

  test('falls back when nothing is reachable', () => {
    expect(pickFastestRegion([{ region: 'us1', ms: null }], 'us2')).toBe('us2');
    expect(pickFastestRegion([], 'us2')).toBe('us2');
  });

  test('returns null with no fallback / bad input', () => {
    expect(pickFastestRegion([])).toBeNull();
    expect(pickFastestRegion(null)).toBeNull();
  });
});
