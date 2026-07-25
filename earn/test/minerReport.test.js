'use strict';

const { buildMinerReports } = require('../src/shared/minerReport');

describe('buildMinerReports', () => {
  test('one card → one rig row with that card\'s own VRAM and hashrate', () => {
    const snap = { gpu: 'NVIDIA GeForce RTX 4090', total: 285.8, accepted: 5,
      gpus: [{ index: 0, gpu: 'NVIDIA GeForce RTX 4090', hashrate: 285.8, accepted: 5 }] };
    const vram = [{ index: 0, name: 'NVIDIA GeForce RTX 4090', usedMb: 4096, totalMb: 24564 }];
    expect(buildMinerReports({ address: '  prl1pabc ', worker: 'rig9', region: 'eu1' }, snap, vram, '0.1.16')).toEqual([
      { address: 'prl1pabc', worker: 'rig9', region: 'eu1', version: '0.1.16',
        gpu: 'NVIDIA GeForce RTX 4090', hashrate: 285.8, accepted: 5, vramUsedMb: 4096, vramTotalMb: 24564, llmModel: null },
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
    expect(buildMinerReports({ address: 'prl1pabc', worker: 'rig01', region: 'us2' }, snap, vram, '0.1.16')).toEqual([
      { address: 'prl1pabc', worker: 'rig01/gpu0', region: 'us2', version: '0.1.16', gpu: 'RTX 4090', hashrate: 200, accepted: 10, vramUsedMb: 4096, vramTotalMb: 24564, llmModel: null },
      { address: 'prl1pabc', worker: 'rig01/gpu1', region: 'us2', version: '0.1.16', gpu: 'RTX 4060 Ti', hashrate: 100, accepted: 4, vramUsedMb: 2000, vramTotalMb: 16380, llmModel: null },
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

  test('safety net: engine under-enumerates GPUs → one row per physical card, VRAM per card, hashrate split evenly', () => {
    // The engine logged a single aggregate line (one card, whole-rig hashrate),
    // but nvidia-smi sees three cards. Don't collapse the rig into one 24 GB row.
    const snap = { gpu: 'RTX 3070 Laptop GPU', total: 172.8,
      gpus: [{ index: 0, gpu: 'RTX 3070 Laptop GPU', hashrate: 172.8, accepted: 30 }] };
    const vram = [
      { index: 0, name: 'RTX 3070 Laptop GPU', usedMb: 2000, totalMb: 8192 },
      { index: 1, name: 'RTX 3070 Laptop GPU', usedMb: 2000, totalMb: 8192 },
      { index: 2, name: 'RTX 3070 Laptop GPU', usedMb: 2000, totalMb: 8192 },
    ];
    const rows = buildMinerReports({ address: 'prl1pabc', worker: 'IPTU', region: 'us2' }, snap, vram);
    expect(rows).toHaveLength(3);
    expect(rows.map((r) => r.worker)).toEqual(['IPTU/gpu0', 'IPTU/gpu1', 'IPTU/gpu2']);
    expect(rows.every((r) => r.gpu === 'RTX 3070 Laptop GPU')).toBe(true);
    expect(rows.every((r) => r.vramTotalMb === 8192 && r.vramUsedMb === 2000)).toBe(true);
    expect(rows.every((r) => r.hashrate === 57.6)).toBe(true);       // 172.8 / 3
    expect(rows.reduce((a, r) => a + r.hashrate, 0)).toBeCloseTo(172.8); // total preserved
    expect(rows.every((r) => r.accepted === 10)).toBe(true);         // 30 / 3
  });

  test('safety net with no engine hashrate/shares and a nameless zero-VRAM card falls back cleanly', () => {
    const snap = { gpus: [{ index: 0, gpu: 'RTX 3070' }] };   // engine name, but no hashrate/accepted
    const vram = [
      { index: 0, name: null, usedMb: 0, totalMb: 0 },        // no probe name → engine name; zero VRAM
      { index: 1, name: 'RTX 3070', usedMb: 100, totalMb: 8192 },
    ];
    const rows = buildMinerReports({ address: 'prl1pabc', worker: 'rig01' }, snap, vram);
    expect(rows).toHaveLength(2);
    expect(rows[0]).toMatchObject({ gpu: 'RTX 3070', hashrate: 0, accepted: 0, vramUsedMb: 0, vramTotalMb: 0 });
    expect(rows[1]).toMatchObject({ gpu: 'RTX 3070', vramTotalMb: 8192 });
  });

  test('safety net with no name anywhere reports a null gpu', () => {
    const snap = { gpus: [{ index: 0 }] };                    // no engine name
    const vram = [
      { index: 0, name: null, usedMb: 1, totalMb: 2 },
      { index: 1, name: null, usedMb: 3, totalMb: 4 },
    ];
    const rows = buildMinerReports({ address: 'prl1pabc', worker: 'rig01' }, snap, vram);
    expect(rows.every((r) => r.gpu === null)).toBe(true);
  });

  test('safety net does NOT trigger when the engine already reports every card (per-card hashrate kept)', () => {
    const snap = { gpus: [
      { index: 0, gpu: 'RTX 4090', hashrate: 200, accepted: 10 },
      { index: 1, gpu: 'RTX 4060 Ti', hashrate: 100, accepted: 4 },
    ] };
    const vram = [
      { index: 0, name: 'RTX 4090', usedMb: 4096, totalMb: 24564 },
      { index: 1, name: 'RTX 4060 Ti', usedMb: 2000, totalMb: 16380 },
    ];
    const rows = buildMinerReports({ address: 'prl1pabc', worker: 'rig01' }, snap, vram);
    expect(rows.map((r) => r.hashrate)).toEqual([200, 100]);   // measured, not split
  });

  test('safety net needs nvidia-smi: one aggregate line with no probe stays a single row', () => {
    const snap = { gpu: 'RTX 3070', total: 57.6, gpus: [{ index: 0, gpu: 'RTX 3070', hashrate: 57.6 }] };
    const rows = buildMinerReports({ address: 'prl1pabc', worker: 'rig01' }, snap, []);
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({ worker: 'rig01', hashrate: 57.6, vramTotalMb: 0 });
  });

  test('a just-connected card (no hashrate/shares yet, zero VRAM) still reports as a row', () => {
    const snap = { gpus: [{ index: 0, gpu: 'RTX 4090' }] };   // no hashrate/accepted yet
    const [row] = buildMinerReports({ address: 'prl1pabc', worker: 'rig01' }, snap,
      [{ index: 0, name: 'RTX 4090', usedMb: 0, totalMb: 0 }]);
    expect(row).toMatchObject({ gpu: 'RTX 4090', hashrate: 0, accepted: 0, vramUsedMb: 0, vramTotalMb: 0 });
  });

  test('multi-GPU rig with no per-card engine data → one row per physical card, never a bare aggregate', () => {
    // nvidia-smi sees two cards but the engine hasn't broken out per-card stats
    // yet. Post per-card /gpuN rows (not one bare-worker summed row that the
    // board would treat as a phantom card and double-count into the host VRAM).
    const snap = { total: 0, accepted: 0, gpus: [] };   // no per-card data, no gpu name
    const vram = [
      { index: 0, name: 'RTX 4070', usedMb: 0, totalMb: 0 },
      { index: 1, name: 'RTX 4070', usedMb: 500, totalMb: 12282 },
    ];
    const rows = buildMinerReports({ address: 'prl1pabc' }, snap, vram);
    expect(rows.map((r) => r.worker)).toEqual(['rig01/gpu0', 'rig01/gpu1']);
    expect(rows.every((r) => r.gpu === 'RTX 4070' && r.hashrate === 0)).toBe(true);
    expect(rows[1]).toMatchObject({ vramUsedMb: 500, vramTotalMb: 12282 }); // each card's own VRAM, not summed
  });

  test('multi-GPU rig, no per-card data, with a rig-level total → split evenly across the cards', () => {
    const snap = { gpu: 'RTX 4090', total: 100, accepted: 7, gpus: [] };
    const vram = [
      { index: 0, name: 'RTX 4090', usedMb: 4096, totalMb: 24564 },
      { index: 1, name: 'RTX 4090', usedMb: 2048, totalMb: 24564 },
    ];
    expect(buildMinerReports({ address: 'prl1pabc', worker: 'rig01', region: 'us2' }, snap, vram)).toEqual([
      { address: 'prl1pabc', worker: 'rig01/gpu0', region: 'us2', version: null, gpu: 'RTX 4090', hashrate: 50, accepted: 4, vramUsedMb: 4096, vramTotalMb: 24564, llmModel: null },
      { address: 'prl1pabc', worker: 'rig01/gpu1', region: 'us2', version: null, gpu: 'RTX 4090', hashrate: 50, accepted: 4, vramUsedMb: 2048, vramTotalMb: 24564, llmModel: null },
    ]);
  });

  test('single GPU, no per-card data → one bare-worker row named from the snapshot', () => {
    const snap = { gpu: 'RTX 4090', total: 120, accepted: 3, gpus: [] };
    const vram = [{ index: 0, name: 'RTX 4090', usedMb: 4096, totalMb: 24564 }];
    expect(buildMinerReports({ address: 'prl1pabc', worker: 'rig01', region: 'us2' }, snap, vram)).toEqual([
      { address: 'prl1pabc', worker: 'rig01', region: 'us2', version: null,
        gpu: 'RTX 4090', hashrate: 120, accepted: 3, vramUsedMb: 4096, vramTotalMb: 24564, llmModel: null },
    ]);
  });

  test('single GPU, no snapshot gpu name and zero VRAM → named from the probe, zero VRAM', () => {
    const snap = { total: 0, accepted: 0, gpus: [] };   // no gpu name, card not warmed up
    const [row] = buildMinerReports({ address: 'prl1pabc', worker: 'rig01' }, snap,
      [{ index: 0, name: 'RTX 4070', usedMb: 0, totalMb: 0 }]);
    expect(row).toMatchObject({ worker: 'rig01', gpu: 'RTX 4070', vramUsedMb: 0, vramTotalMb: 0 });
  });

  test('applies defaults when called with nothing', () => {
    expect(buildMinerReports()).toEqual([
      { address: '', worker: 'rig01', region: 'us2', version: null, gpu: null, hashrate: 0, accepted: 0, vramUsedMb: 0, vramTotalMb: 0, llmModel: null },
    ]);
  });

  test('tags cards serving the local LLM (per index), leaving the rest blank', () => {
    // A two-card rig serving the model only on GPU 0 (GPU 1 lacks the VRAM): the
    // serving card carries the model, the other reports null → blank on the board.
    const snap = { gpus: [
      { index: 0, gpu: 'RTX 4090', hashrate: 200, accepted: 10 },
      { index: 1, gpu: 'RTX 4060', hashrate: 100, accepted: 4 },
    ] };
    const vram = [
      { index: 0, name: 'RTX 4090', usedMb: 4096, totalMb: 24564 },
      { index: 1, name: 'RTX 4060', usedMb: 2000, totalMb: 8192 },
    ];
    const serving = { model: 'Gemma-4-E4B-it-Q4_K_M', indices: [0] };
    const rows = buildMinerReports({ address: 'prl1pabc', worker: 'rig01' }, snap, vram, '0.2.0', serving);
    expect(rows.map((r) => r.llmModel)).toEqual(['Gemma-4-E4B-it-Q4_K_M', null]);
  });

  test('serving tags flow through the split-rows and single-row paths', () => {
    // Split-rows path: engine under-enumerates, both physical cards serve.
    const splitSnap = { gpu: 'RTX 4090', total: 172.8, gpus: [{ index: 0, gpu: 'RTX 4090', hashrate: 172.8 }] };
    const splitVram = [
      { index: 0, name: 'RTX 4090', usedMb: 2000, totalMb: 24564 },
      { index: 1, name: 'RTX 4090', usedMb: 2000, totalMb: 24564 },
    ];
    const splitRows = buildMinerReports({ address: 'prl1pabc', worker: 'rig01' }, splitSnap, splitVram, '0.2.0',
      { model: 'Gemma-4-E4B-it-Q4_K_M', indices: [0, 1] });
    expect(splitRows.map((r) => r.llmModel)).toEqual(['Gemma-4-E4B-it-Q4_K_M', 'Gemma-4-E4B-it-Q4_K_M']);

    // Single bare-worker row: a lone card serving the model.
    const [single] = buildMinerReports({ address: 'prl1pabc', worker: 'rig01' },
      { gpu: 'RTX 4090', total: 120, gpus: [] },
      [{ index: 0, name: 'RTX 4090', usedMb: 4096, totalMb: 24564 }], '0.2.0',
      { model: 'Gemma-4-E4B-it-Q4_K_M', indices: [0] });
    expect(single.llmModel).toBe('Gemma-4-E4B-it-Q4_K_M');
  });

  test('falls back to rig01 for a blank worker and zeroes bad numbers', () => {
    const [row] = buildMinerReports({ worker: '   ' }, { total: 'x', accepted: null, gpus: [] });
    expect(row).toMatchObject({ worker: 'rig01', hashrate: 0, accepted: 0 });
  });
});
