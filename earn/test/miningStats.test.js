'use strict';

const { MAX_POINTS, initStats, applyEvent, snapshot } = require('../src/shared/miningStats');

describe('initStats', () => {
  test('anchors the uptime clock and starts with no cards', () => {
    expect(initStats(1000)).toEqual({ startMs: 1000, gpus: {}, load: 0, points: [] });
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

  test('a status event fills the card bucket and appends the rig total as a point', () => {
    const s = initStats(0);
    applyEvent(s, { type: 'status', gpuIndex: 0, hashrate: 286.86, accepted: 5, rejected: 1, power: 449, gpu: 'NVIDIA GeForce RTX 4090' });
    expect(s.gpus[0]).toEqual({ index: 0, hashrate: 286.86, accepted: 5, rejected: 1, power: 449, gpu: 'NVIDIA GeForce RTX 4090' });
    expect(s.points).toEqual([286.86]);
  });

  test('a card with no index folds into card 0', () => {
    const s = initStats(0);
    applyEvent(s, { type: 'status', hashrate: 4 });
    expect(s.gpus[0].hashrate).toBe(4);
  });

  test('share counts are SET per card from the cumulative counter, not incremented', () => {
    const s = initStats(0);
    applyEvent(s, { type: 'status', gpuIndex: 0, hashrate: 1, accepted: 5, rejected: 0 });
    applyEvent(s, { type: 'status', gpuIndex: 0, hashrate: 1, accepted: 8, rejected: 1 });
    expect(s.gpus[0].accepted).toBe(8);
    expect(s.gpus[0].rejected).toBe(1);
    expect(s.points).toEqual([1, 1]);
  });

  test('two cards accumulate independently; each point is the rig total', () => {
    const s = initStats(0);
    applyEvent(s, { type: 'status', gpuIndex: 0, hashrate: 10, accepted: 2, gpu: 'RTX 4090' });
    applyEvent(s, { type: 'status', gpuIndex: 1, hashrate: 4, accepted: 1, gpu: 'RTX 4060 Ti' });
    expect(s.gpus[0].hashrate).toBe(10);
    expect(s.gpus[1].hashrate).toBe(4);
    expect(s.points).toEqual([10, 14]);   // total after each card's update
  });

  test('a sparse status event changes nothing it does not carry', () => {
    const s = initStats(0);
    applyEvent(s, { type: 'status', gpuIndex: 0, hashrate: 9, accepted: 3, rejected: 2, power: 100, gpu: 'keep' });
    applyEvent(s, { type: 'status', gpuIndex: 0 });
    expect(s.gpus[0]).toEqual({ index: 0, hashrate: 9, accepted: 3, rejected: 2, power: 100, gpu: 'keep' });
  });

  test('caps the chart point buffer at MAX_POINTS', () => {
    const s = initStats(0);
    for (let i = 0; i < MAX_POINTS + 5; i++) applyEvent(s, { type: 'status', gpuIndex: 0, hashrate: i });
    expect(s.points.length).toBe(MAX_POINTS);
    expect(s.points[0]).toBe(5);
    expect(s.points[s.points.length - 1]).toBe(MAX_POINTS + 4);
  });

  test('a connected event records the gpu name on its card without a point', () => {
    const s = initStats(0);
    applyEvent(s, { type: 'connected', gpuIndex: 0, endpoint: 'us2.alphapool.tech:5566', gpu: 'NVIDIA GeForce RTX 4090' });
    expect(s.gpus[0].gpu).toBe('NVIDIA GeForce RTX 4090');
    expect(s.points).toEqual([]);
  });

  test('a connected event without a gpu creates no card', () => {
    const s = initStats(0);
    applyEvent(s, { type: 'connected', endpoint: 'h:1' });
    expect(s.gpus).toEqual({});
  });

  test('ignores unrecognised event types', () => {
    const s = initStats(0);
    applyEvent(s, { type: 'whatever' });
    expect(s).toEqual(initStats(0));
  });
});

describe('snapshot', () => {
  test('projects a single card into the renderer shape with a one-card breakdown', () => {
    const s = initStats(1000);
    applyEvent(s, { type: 'status', gpuIndex: 0, hashrate: 3.2, accepted: 4, rejected: 0, power: 300, gpu: 'RTX 4090' });
    expect(snapshot(s, 6000)).toEqual({
      total: 3.2, points: [3.2], accepted: 4, rejected: 0, load: 0, power: 300, gpu: 'RTX 4090',
      gpus: [{ index: 0, gpu: 'RTX 4090', hashrate: 3.2, accepted: 4, rejected: 0, power: 300 }],
      uptimeSec: 5,
    });
  });

  test('aggregates rig-level figures across cards and lists them by index', () => {
    const s = initStats(0);
    applyEvent(s, { type: 'status', gpuIndex: 1, hashrate: 4, accepted: 1, rejected: 0, power: 160, gpu: 'RTX 4060 Ti' });
    applyEvent(s, { type: 'status', gpuIndex: 0, hashrate: 10, accepted: 3, rejected: 1, power: 300, gpu: 'RTX 4090' });
    const snap = snapshot(s, 0);
    expect(snap.total).toBe(14);
    expect(snap.accepted).toBe(4);
    expect(snap.rejected).toBe(1);
    expect(snap.power).toBe(460);
    expect(snap.gpu).toBe('RTX 4090');   // representative = lowest-index card
    expect(snap.gpus.map((g) => g.index)).toEqual([0, 1]);
    expect(snap.gpus[1]).toEqual({ index: 1, gpu: 'RTX 4060 Ti', hashrate: 4, accepted: 1, rejected: 0, power: 160 });
  });

  test('reads zeros and a null gpu before any card reports', () => {
    expect(snapshot(initStats(0), 0)).toEqual({
      total: 0, points: [], accepted: 0, rejected: 0, load: 0, power: 0, gpu: null, gpus: [], uptimeSec: 0,
    });
  });

  test('a card that reported only zeros yields a null representative name and zero totals', () => {
    const s = initStats(0);
    applyEvent(s, { type: 'status', gpuIndex: 0, hashrate: 0 });
    const snap = snapshot(s, 0);
    expect(snap.gpu).toBeNull();
    expect(snap.total).toBe(0);
    expect(snap.gpus).toEqual([{ index: 0, gpu: null, hashrate: 0, accepted: 0, rejected: 0, power: 0 }]);
  });

  test('returns a copy of points, not the live array', () => {
    const s = initStats(0);
    applyEvent(s, { type: 'status', gpuIndex: 0, hashrate: 1 });
    const snap = snapshot(s, 0);
    snap.points.push(999);
    expect(s.points).toEqual([1]);
  });

  test('clamps uptime to 0 and treats a missing clock as 0', () => {
    const s = initStats(5000);
    expect(snapshot(s).uptimeSec).toBe(0);
  });
});
