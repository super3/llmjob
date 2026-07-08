'use strict';

const { statsFilePayload } = require('../src/shared/statsFile');

describe('statsFilePayload', () => {
  test('maps a live snapshot to the stats-file shape (ths stays in TH/s)', () => {
    const snap = { total: 354.1, accepted: 14820, rejected: 3, uptimeSec: 15128, gpu: 'RTX 5090' };
    expect(statsFilePayload(snap, { version: '0.1.14', nowMs: 1783300000000 })).toEqual({
      ver: '0.1.14', algo: 'pearlhash', ths: 354.1, accepted: 14820, rejected: 3,
      uptimeSec: 15128, gpu: 'RTX 5090', updatedMs: 1783300000000,
    });
  });

  test('zeros and nulls a cold snapshot; tolerates missing args entirely', () => {
    expect(statsFilePayload()).toEqual({
      ver: '', algo: 'pearlhash', ths: 0, accepted: 0, rejected: 0,
      uptimeSec: 0, gpu: null, updatedMs: 0,
    });
  });

  test('non-finite or negative hashrate reads as 0, never NaN', () => {
    expect(statsFilePayload({ total: NaN }).ths).toBe(0);
    expect(statsFilePayload({ total: -5 }).ths).toBe(0);
    expect(statsFilePayload({ total: 'x' }).ths).toBe(0);
  });
});
