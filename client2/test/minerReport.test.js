'use strict';

const { buildMinerReport } = require('../src/shared/minerReport');

describe('buildMinerReport', () => {
  test('maps settings + snapshot into the ping payload', () => {
    expect(buildMinerReport(
      { address: '  prl1pabc ', worker: 'rig9', region: 'eu1' },
      { gpu: 'NVIDIA GeForce RTX 4090', total: 285.8, accepted: 5 }
    )).toEqual({
      address: 'prl1pabc', worker: 'rig9', region: 'eu1',
      gpu: 'NVIDIA GeForce RTX 4090', hashrate: 285.8, accepted: 5,
    });
  });

  test('applies defaults when called with nothing', () => {
    expect(buildMinerReport()).toEqual({
      address: '', worker: 'rig01', region: 'us2', gpu: null, hashrate: 0, accepted: 0,
    });
  });

  test('falls back to rig01 for a blank worker and zeroes bad numbers', () => {
    expect(buildMinerReport({ worker: '   ' }, { total: 'x', accepted: null })).toMatchObject({
      worker: 'rig01', hashrate: 0, accepted: 0,
    });
  });
});
