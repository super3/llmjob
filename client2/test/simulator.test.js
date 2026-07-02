'use strict';

const { clamp, Simulator } = require('../src/shared/simulator');

describe('clamp', () => {
  test('clamps below, within and above the range', () => {
    expect(clamp(-1, 0, 10)).toBe(0);
    expect(clamp(5, 0, 10)).toBe(5);
    expect(clamp(20, 0, 10)).toBe(10);
  });
});

describe('Simulator', () => {
  test('constructs with defaults (no opts)', () => {
    const sim = new Simulator();
    expect(sim.total).toBe(354.1);
    expect(sim.accepted).toBe(14820);
    expect(sim.rejected).toBe(3);
    expect(sim.points).toHaveLength(60);
    expect(sim.maxPoints).toBe(60);
  });

  test('honors provided options', () => {
    const sim = new Simulator({
      rng: () => 0.5, total: 100, accepted: 1, rejected: 0,
      load: 80, power: 200, uptimeSec: 10, maxPoints: 4, seedLength: 2,
    });
    expect(sim.total).toBe(100);
    expect(sim.uptimeSec).toBe(10);
    expect(sim.points).toHaveLength(2);
    expect(sim.maxPoints).toBe(4);
  });

  test('step advances state and increments counters when rng is low', () => {
    const sim = new Simulator({ rng: () => 0, accepted: 0, rejected: 0, uptimeSec: 0 });
    const snap = sim.step();
    expect(snap.accepted).toBe(1); // 0 < 0.82
    expect(snap.rejected).toBe(1); // 0 < 0.012
    expect(snap.uptimeSec).toBe(1);
    expect(snap.points.length).toBeLessThanOrEqual(60);
    expect(snap.total).toBeGreaterThanOrEqual(334);
    expect(snap.total).toBeLessThanOrEqual(368);
  });

  test('step does not increment counters when rng is high', () => {
    const sim = new Simulator({ rng: () => 0.9, accepted: 0, rejected: 0 });
    const snap = sim.step();
    expect(snap.accepted).toBe(0); // 0.9 !< 0.82
    expect(snap.rejected).toBe(0); // 0.9 !< 0.012
  });

  test('trims points to maxPoints', () => {
    const sim = new Simulator({ rng: () => 0, maxPoints: 3, seedLength: 5 });
    const snap = sim.step();
    expect(snap.points).toHaveLength(3);
  });

  test('snapshot returns a copy of points', () => {
    const sim = new Simulator({ rng: () => 0.5, seedLength: 3, maxPoints: 3 });
    const a = sim.snapshot();
    a.points.push(999);
    expect(sim.points).toHaveLength(3);
  });
});
