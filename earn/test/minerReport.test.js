'use strict';

const { buildMinerReports } = require('../src/shared/minerReport');

describe('buildMinerReports', () => {
  test('one card → one rig row with that card\'s own VRAM and hashrate', () => {
    const snap = { gpu: 'NVIDIA GeForce RTX 4090', total: 285.8, accepted: 5,
      gpus: [{ index: 0, gpu: 'NVIDIA GeForce RTX 4090', hashrate: 285.8, accepted: 5 }] };
    const vram = [{ index: 0, name: 'NVIDIA GeForce RTX 4090', usedMb: 4096, totalMb: 24564 }];
    expect(buildMinerReports({ address: '  prl1pabc ', worker: 'rig9', region: 'eu1' }, snap, vram)).toEqual([
      { address: 'prl1pabc', worker: 'rig9', region: 'eu1',
        gpu: 'NVIDIA GeForce RTX 4090', hashrate: 285.8, accepted: 5, vramUsedMb: 4096, vramTotalMb: 24564 },
    ]);
  });

  test('a multi-GPU rig posts one row per card, each with its own VRAM and a distinct worker', () => {
    const snap = { gpu: 'RTX 4090', total: 300,
      gpus: [
        { index: 0, gpu: 'RTX 4090', hashrate: 200, accepted: 10 },
        { index: 1, gpu: 'RTX 4060 Ti', hashrate: 100, accepted: 4 },
      ] };
    const vram = [
      { index: 1, name: 'RTX 4060 Ti', usedMb: 2000, totalMb: 16380 },
      { index: 0, name: 'RTX 4090', usedMb: 4096, totalMb: 24564 },
    ];
    expect(buildMinerReports({ address: 'prl1pabc', worker: 'rig01', region: 'us2' }, snap, vram)).toEqual([
      { address: 'prl1pabc', worker: 'rig01/gpu0', region: 'us2', gpu: 'RTX 4090', hashrate: 200, accepted: 10, vramUsedMb: 4096, vramTotalMb: 24564 },
      { address: 'prl1pabc', worker: 'rig01/gpu1', region: 'us2', gpu: 'RTX 4060 Ti', hashrate: 100, accepted: 4, vramUsedMb: 2000, vramTotalMb: 16380 },
    ]);
  });

  test('a card with no VRAM match reports 0 VRAM and falls back to the probe name', () => {
    const snap = { gpus: [{ index: 0, gpu: null, hashrate: 5, accepted: 1 }] };
    const vram = [{ index: 0, name: 'RTX 4070', usedMb: 1000, totalMb: 12282 }];
    const [row] = buildMinerReports({ address: 'prl1pabc', worker: 'rig01' }, snap, vram);
    expect(row).toMatchObject({ gpu: 'RTX 4070', vramUsedMb: 1000, vramTotalMb: 12282 });

    const [none] = buildMinerReports({ address: 'prl1pabc', worker: 'rig01' }, snap, []);
    expect(none).toMatchObject({ gpu: null, vramUsedMb: 0, vramTotalMb: 0 });
  });

  test('a just-connected card (no hashrate/shares yet, zero VRAM) still reports as a row', () => {
    const snap = { gpus: [{ index: 0, gpu: 'RTX 4090' }] };   // no hashrate/accepted yet
    const [row] = buildMinerReports({ address: 'prl1pabc', worker: 'rig01' }, snap,
      [{ index: 0, name: 'RTX 4090', usedMb: 0, totalMb: 0 }]);
    expect(row).toMatchObject({ gpu: 'RTX 4090', hashrate: 0, accepted: 0, vramUsedMb: 0, vramTotalMb: 0 });
  });

  test('no per-card data and no snapshot gpu → names the row from the first probe entry, summing VRAM incl. zeros', () => {
    const snap = { total: 0, accepted: 0, gpus: [] };   // no gpu name
    const vram = [
      { index: 0, name: 'RTX 4070', usedMb: 0, totalMb: 0 },
      { index: 1, name: 'RTX 4070', usedMb: 500, totalMb: 12282 },
    ];
    const [row] = buildMinerReports({ address: 'prl1pabc' }, snap, vram);
    expect(row).toMatchObject({ gpu: 'RTX 4070', vramUsedMb: 500, vramTotalMb: 12282 });
  });

  test('no per-card data yet → a single rig-level row with summed VRAM (back-compat)', () => {
    const snap = { gpu: 'RTX 4090', total: 100, accepted: 7, gpus: [] };
    const vram = [
      { index: 0, name: 'RTX 4090', usedMb: 4096, totalMb: 24564 },
      { index: 1, name: 'RTX 4090', usedMb: 2048, totalMb: 24564 },
    ];
    expect(buildMinerReports({ address: 'prl1pabc', worker: 'rig01', region: 'us2' }, snap, vram)).toEqual([
      { address: 'prl1pabc', worker: 'rig01', region: 'us2',
        gpu: 'RTX 4090', hashrate: 100, accepted: 7, vramUsedMb: 6144, vramTotalMb: 49128 },
    ]);
  });

  test('applies defaults when called with nothing', () => {
    expect(buildMinerReports()).toEqual([
      { address: '', worker: 'rig01', region: 'us2', gpu: null, hashrate: 0, accepted: 0, vramUsedMb: 0, vramTotalMb: 0 },
    ]);
  });

  test('falls back to rig01 for a blank worker and zeroes bad numbers', () => {
    const [row] = buildMinerReports({ worker: '   ' }, { total: 'x', accepted: null, gpus: [] });
    expect(row).toMatchObject({ worker: 'rig01', hashrate: 0, accepted: 0 });
  });
});
