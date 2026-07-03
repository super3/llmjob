'use strict';

const { MAX_POINTS, initStats, applyEvent, snapshot } = require('../src/shared/miningStats');

describe('initStats', () => {
  test('anchors the uptime clock and zeroes everything', () => {
    expect(initStats(1000)).toEqual({ startMs: 1000, hashrate: 0, accepted: 0, rejected: 0, load: 0, power: 0, gpu: null, points: [] });
  });
  test('defaults startMs to 0 when omitted', () => {
    expect(initStats().startMs).toBe(0);
  });
});

describe('applyEvent', () => {
  test('returns the accumulator untouched for null stats', () => {
    expect(applyEvent(null, { type: 'status' })).toBeNull();
  });

  test('ignores a null event', () => {
    const s = initStats(0);
    expect(applyEvent(s, null)).toBe(s);
    expect(s).toEqual(initStats(0));
  });

  test('a status event sets hashrate, cumulative counts, power and gpu, and appends a point', () => {
    const s = initStats(0);
    applyEvent(s, { type: 'status', hashrate: 286.86, accepted: 5, rejected: 1, power: 449, gpu: 'NVIDIA GeForce RTX 4090' });
    expect(s.hashrate).toBe(286.86);
    expect(s.accepted).toBe(5);
    expect(s.rejected).toBe(1);
    expect(s.power).toBe(449);
    expect(s.gpu).toBe('NVIDIA GeForce RTX 4090');
    expect(s.points).toEqual([286.86]);
  });

  test('share counts are SET from the cumulative counter, not incremented', () => {
    const s = initStats(0);
    applyEvent(s, { type: 'status', hashrate: 1, accepted: 5, rejected: 0 });
    applyEvent(s, { type: 'status', hashrate: 1, accepted: 8, rejected: 1 });
    expect(s.accepted).toBe(8);
    expect(s.rejected).toBe(1);
    expect(s.points).toEqual([1, 1]);
  });

  test('a sparse status event changes nothing it does not carry', () => {
    const s = initStats(0);
    s.hashrate = 9; s.accepted = 3; s.rejected = 2; s.power = 100; s.gpu = 'keep';
    applyEvent(s, { type: 'status' });
    expect(s).toEqual({ startMs: 0, hashrate: 9, accepted: 3, rejected: 2, load: 0, power: 100, gpu: 'keep', points: [] });
  });

  test('caps the chart point buffer at MAX_POINTS', () => {
    const s = initStats(0);
    for (let i = 0; i < MAX_POINTS + 5; i++) applyEvent(s, { type: 'status', hashrate: i });
    expect(s.points.length).toBe(MAX_POINTS);
    expect(s.points[0]).toBe(5);
    expect(s.points[s.points.length - 1]).toBe(MAX_POINTS + 4);
  });

  test('a connected event records the gpu', () => {
    const s = initStats(0);
    applyEvent(s, { type: 'connected', endpoint: 'us2.alphapool.tech:5566', gpu: 'NVIDIA GeForce RTX 4090' });
    expect(s.gpu).toBe('NVIDIA GeForce RTX 4090');
  });

  test('a connected event without a gpu leaves it unchanged', () => {
    const s = initStats(0);
    s.gpu = 'prev';
    applyEvent(s, { type: 'connected', endpoint: 'h:1' });
    expect(s.gpu).toBe('prev');
  });

  test('ignores unrecognised event types', () => {
    const s = initStats(0);
    applyEvent(s, { type: 'whatever' });
    expect(s).toEqual(initStats(0));
  });
});

describe('snapshot', () => {
  test('projects the accumulator into the renderer shape', () => {
    const s = initStats(1000);
    applyEvent(s, { type: 'status', hashrate: 3.2, accepted: 4, rejected: 0, power: 300, gpu: 'RTX 4090' });
    expect(snapshot(s, 6000)).toEqual({
      total: 3.2, points: [3.2], accepted: 4, rejected: 0, load: 0, power: 300, gpu: 'RTX 4090', uptimeSec: 5,
    });
  });

  test('returns a copy of points, not the live array', () => {
    const s = initStats(0);
    applyEvent(s, { type: 'status', hashrate: 1 });
    const snap = snapshot(s, 0);
    snap.points.push(999);
    expect(s.points).toEqual([1]);
  });

  test('clamps uptime to 0 and treats a missing clock as 0', () => {
    const s = initStats(5000);
    expect(snapshot(s).uptimeSec).toBe(0);
  });
});
