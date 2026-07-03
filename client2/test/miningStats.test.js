'use strict';

const { MAX_POINTS, initStats, applyEvent, snapshot } = require('../src/shared/miningStats');

describe('initStats', () => {
  test('anchors the uptime clock and zeroes counters', () => {
    expect(initStats(1000)).toEqual({ startMs: 1000, hashrate: 0, accepted: 0, rejected: 0, load: 0, power: 0, points: [] });
  });
  test('defaults startMs to 0 when omitted', () => {
    expect(initStats().startMs).toBe(0);
  });
});

describe('applyEvent', () => {
  test('returns the accumulator untouched for null stats', () => {
    expect(applyEvent(null, { type: 'share', status: 'accepted' })).toBeNull();
  });

  test('ignores a null event', () => {
    const s = initStats(0);
    expect(applyEvent(s, null)).toBe(s);
    expect(s.accepted).toBe(0);
  });

  test('records hashrate with load and power and appends a chart point', () => {
    const s = initStats(0);
    applyEvent(s, { type: 'hashrate', hashrate: 12.5, load: 90, power: 300 });
    expect(s.hashrate).toBe(12.5);
    expect(s.load).toBe(90);
    expect(s.power).toBe(300);
    expect(s.points).toEqual([12.5]);
  });

  test('hashrate without load/power leaves them and coerces a bad value to 0', () => {
    const s = initStats(0);
    s.load = 5; s.power = 7;
    applyEvent(s, { type: 'hashrate', hashrate: 'nope' });
    expect(s.hashrate).toBe(0);
    expect(s.load).toBe(5);
    expect(s.power).toBe(7);
    expect(s.points).toEqual([0]);
  });

  test('caps the chart point buffer at MAX_POINTS', () => {
    const s = initStats(0);
    for (let i = 0; i < MAX_POINTS + 5; i++) applyEvent(s, { type: 'hashrate', hashrate: i });
    expect(s.points.length).toBe(MAX_POINTS);
    expect(s.points[0]).toBe(5); // first five shifted out
    expect(s.points[s.points.length - 1]).toBe(MAX_POINTS + 4);
  });

  test('counts accepted and rejected shares', () => {
    const s = initStats(0);
    applyEvent(s, { type: 'share', status: 'accepted' });
    applyEvent(s, { type: 'share', status: 'accepted' });
    applyEvent(s, { type: 'share', status: 'rejected' });
    expect(s.accepted).toBe(2);
    expect(s.rejected).toBe(1);
  });

  test('ignores a share with an unknown status', () => {
    const s = initStats(0);
    applyEvent(s, { type: 'share', status: 'stale' });
    expect(s.accepted).toBe(0);
    expect(s.rejected).toBe(0);
  });

  test('ignores unrecognised event types', () => {
    const s = initStats(0);
    applyEvent(s, { type: 'connected', endpoint: 'x' });
    expect(s).toEqual(initStats(0));
  });
});

describe('snapshot', () => {
  test('projects the accumulator into the renderer shape', () => {
    const s = initStats(1000);
    applyEvent(s, { type: 'hashrate', hashrate: 3.2, load: 80, power: 250 });
    applyEvent(s, { type: 'share', status: 'accepted' });
    expect(snapshot(s, 6000)).toEqual({
      total: 3.2, points: [3.2], accepted: 1, rejected: 0, load: 80, power: 250, uptimeSec: 5,
    });
  });

  test('returns a copy of points, not the live array', () => {
    const s = initStats(0);
    applyEvent(s, { type: 'hashrate', hashrate: 1 });
    const snap = snapshot(s, 0);
    snap.points.push(999);
    expect(s.points).toEqual([1]);
  });

  test('clamps uptime to 0 and treats a missing clock as 0', () => {
    const s = initStats(5000);
    expect(snapshot(s).uptimeSec).toBe(0); // nowMs undefined -> 0, then clamped
  });
});
