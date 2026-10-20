'use strict';

const { buildMinerReports, hostStats, readRamMb } = require('../src/shared/minerReport');

// What a row says about the machine when the caller passes no `host`: unknown,
// which is also what the server stores for a client too old to send it.
const NO_HOST = { ramUsedMb: null, ramTotalMb: null, xmrHashrate: null };

describe('buildMinerReports', () => {
  test('one card → one rig row with that card\'s own VRAM and hashrate', () => {
    const snap = { gpu: 'NVIDIA GeForce RTX 4090', total: 285.8, accepted: 5,
      gpus: [{ index: 0, gpu: 'NVIDIA GeForce RTX 4090', hashrate: 285.8, accepted: 5 }] };
    const vram = [{ index: 0, name: 'NVIDIA GeForce RTX 4090', usedMb: 4096, totalMb: 24564 }];
    expect(buildMinerReports({ address: '  prl1pabc ', worker: 'rig9', region: 'eu1' }, snap, vram, '0.1.16')).toEqual([
      { address: 'prl1pabc', worker: 'rig9', region: 'eu1', version: '0.1.16',
        gpu: 'NVIDIA GeForce RTX 4090', hashrate: 285.8, accepted: 5, vramUsedMb: 4096, vramTotalMb: 24564, llmModel: null, nodeId: null, ...NO_HOST },
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
      { address: 'prl1pabc', worker: 'rig01/gpu0', region: 'us2', version: '0.1.16', gpu: 'RTX 4090', hashrate: 200, accepted: 10, vramUsedMb: 4096, vramTotalMb: 24564, llmModel: null, nodeId: null, ...NO_HOST },
      { address: 'prl1pabc', worker: 'rig01/gpu1', region: 'us2', version: '0.1.16', gpu: 'RTX 4060 Ti', hashrate: 100, accepted: 4, vramUsedMb: 2000, vramTotalMb: 16380, llmModel: null, nodeId: null, ...NO_HOST },
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

  // The engine mines on card 0 only: card 1's core refused because the local
  // LLM holds its VRAM. Card 1 still shows, with its VRAM and its model, at 0.
  // The rig's hashrate used to be split evenly over both, which put an idle
  // card on the board at half the real card's rate.
  test('a card the engine does not report reads 0, and each card keeps its own numbers', () => {
    const snap = { gpu: 'RTX 4090', total: 172.8,
      gpus: [{ index: 0, gpu: 'RTX 4090', hashrate: 172.8, accepted: 30 }] };
    const vram = [
      { index: 0, name: 'RTX 4090', usedMb: 2000, totalMb: 24564 },
      { index: 1, name: 'RTX 3090', usedMb: 20000, totalMb: 24576 },
    ];
    const rows = buildMinerReports({ address: 'prl1pabc', worker: 'IPTU', region: 'us2' }, snap, vram, '0.6.6',
      { model: 'Gemma-4-E4B-it-Q4_K_M', indices: [1] });
    expect(rows).toEqual([
      { address: 'prl1pabc', worker: 'IPTU/gpu0', region: 'us2', version: '0.6.6', gpu: 'RTX 4090', hashrate: 172.8, accepted: 30, vramUsedMb: 2000, vramTotalMb: 24564, llmModel: null, nodeId: null, ...NO_HOST },
      { address: 'prl1pabc', worker: 'IPTU/gpu1', region: 'us2', version: '0.6.6', gpu: 'RTX 3090', hashrate: 0, accepted: 0, vramUsedMb: 20000, vramTotalMb: 24576, llmModel: 'Gemma-4-E4B-it-Q4_K_M', nodeId: null, ...NO_HOST },
    ]);
  });

  test('a card the engine reports that nvidia-smi does not list still gets a row', () => {
    const snap = { gpus: [
      { index: 0, gpu: 'RTX 4090', hashrate: 200, accepted: 10 },
      { index: 1, gpu: 'RTX 4060', hashrate: 100, accepted: 4 },
    ] };
    const rows = buildMinerReports({ address: 'prl1pabc', worker: 'rig01' }, snap,
      [{ index: 0, name: 'RTX 4090', usedMb: 4096, totalMb: 24564 }]);
    expect(rows.map((r) => [r.worker, r.gpu, r.hashrate, r.vramTotalMb])).toEqual([
      ['rig01/gpu0', 'RTX 4090', 200, 24564],
      ['rig01/gpu1', 'RTX 4060', 100, 0],
    ]);
  });

  test('no engine hashrate/shares and a nameless zero-VRAM card falls back cleanly', () => {
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

  test('no name anywhere reports a null gpu', () => {
    const snap = { gpus: [{ index: 0 }] };                    // no engine name
    const vram = [
      { index: 0, name: null, usedMb: 1, totalMb: 2 },
      { index: 1, name: null, usedMb: 3, totalMb: 4 },
    ];
    const rows = buildMinerReports({ address: 'prl1pabc', worker: 'rig01' }, snap, vram);
    expect(rows.every((r) => r.gpu === null)).toBe(true);
  });

  test('the engine reports every card: per-card hashrate kept', () => {
    const snap = { gpus: [
      { index: 0, gpu: 'RTX 4090', hashrate: 200, accepted: 10 },
      { index: 1, gpu: 'RTX 4060 Ti', hashrate: 100, accepted: 4 },
    ] };
    const vram = [
      { index: 0, name: 'RTX 4090', usedMb: 4096, totalMb: 24564 },
      { index: 1, name: 'RTX 4060 Ti', usedMb: 2000, totalMb: 16380 },
    ];
    const rows = buildMinerReports({ address: 'prl1pabc', worker: 'rig01' }, snap, vram);
    expect(rows.map((r) => r.hashrate)).toEqual([200, 100]);
  });

  test('one engine card with no nvidia-smi stays a single bare row', () => {
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

  test('single GPU, no per-card data → one bare-worker row named from the snapshot', () => {
    const snap = { gpu: 'RTX 4090', total: 120, accepted: 3, gpus: [] };
    const vram = [{ index: 0, name: 'RTX 4090', usedMb: 4096, totalMb: 24564 }];
    expect(buildMinerReports({ address: 'prl1pabc', worker: 'rig01', region: 'us2' }, snap, vram)).toEqual([
      { address: 'prl1pabc', worker: 'rig01', region: 'us2', version: null,
        gpu: 'RTX 4090', hashrate: 120, accepted: 3, vramUsedMb: 4096, vramTotalMb: 24564, llmModel: null, nodeId: null, ...NO_HOST },
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
      { address: '', worker: 'rig01', region: 'us2', version: null, gpu: null, hashrate: 0, accepted: 0, vramUsedMb: 0, vramTotalMb: 0, llmModel: null, nodeId: null, ...NO_HOST },
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

  test('reports the node id on every row while serving the cluster, null when only running the model', () => {
    // Running a model and serving the cluster are different things: the board can
    // only tell them apart because an armed node sends its id. It's machine-level,
    // so every card row of the rig carries the same one.
    const snap = { gpus: [
      { index: 0, gpu: 'RTX 4090', hashrate: 200, accepted: 10 },
      { index: 1, gpu: 'RTX 4060', hashrate: 100, accepted: 4 },
    ] };
    const vram = [
      { index: 0, name: 'RTX 4090', usedMb: 4096, totalMb: 24564 },
      { index: 1, name: 'RTX 4060', usedMb: 2000, totalMb: 8192 },
    ];
    const model = 'Gemma-4-E4B-it-Q4_K_M';
    const settings = { address: 'prl1pabc', worker: 'rig01' };

    // Armed to serve → the id rides on both rows, even the card not running the model.
    const serving = buildMinerReports(settings, snap, vram, '0.2.0', { model, indices: [0], nodeId: '5840fc' });
    expect(serving.map((r) => r.nodeId)).toEqual(['5840fc', '5840fc']);

    // Model loaded but not linked/armed → no id, so the board shows it as advertising only.
    const local = buildMinerReports(settings, snap, vram, '0.2.0', { model, indices: [0], nodeId: null });
    expect(local.map((r) => r.nodeId)).toEqual([null, null]);
  });

  test('serving tags flow through the per-card and single-row paths', () => {
    // Per-card path: the engine reports one card, both physical cards serve.
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

  // RAM and the Monero hashrate are the machine's, like the node id: every card
  // row of a rig carries the same values, on the per-card and single-row paths.
  test('puts the machine\'s RAM and Monero hashrate on every row', () => {
    const host = { ramUsedMb: 12288, ramTotalMb: 32768, xmrHashrate: 1845.2 };
    const snap = { gpus: [
      { index: 0, gpu: 'RTX 4090', hashrate: 200, accepted: 10 },
      { index: 1, gpu: 'RTX 4060', hashrate: 100, accepted: 4 },
    ] };
    const vram = [
      { index: 0, name: 'RTX 4090', usedMb: 4096, totalMb: 24564 },
      { index: 1, name: 'RTX 4060', usedMb: 2000, totalMb: 8192 },
    ];
    const rows = buildMinerReports({ address: 'prl1pabc', worker: 'rig01' }, snap, vram, '0.6.6', null, host);
    expect(rows.map((r) => [r.worker, r.ramUsedMb, r.ramTotalMb, r.xmrHashrate])).toEqual([
      ['rig01/gpu0', 12288, 32768, 1845.2],
      ['rig01/gpu1', 12288, 32768, 1845.2],
    ]);

    const [single] = buildMinerReports({ address: 'prl1pabc' }, { total: 120, gpus: [] }, [], '0.6.6', null, host);
    expect(single).toMatchObject(host);
  });

  // Not CPU mining is null, which the board must not confuse with 0 H/s; a
  // value that isn't a number is unknown, not 0.
  test('a host field left out, null, or not a number reports null; 0 stays 0', () => {
    const [row] = buildMinerReports({}, {}, [], null, null, { ramUsedMb: 'lots', ramTotalMb: 0, xmrHashrate: null });
    expect(row).toMatchObject({ ramUsedMb: null, ramTotalMb: 0, xmrHashrate: null });
    const [partial] = buildMinerReports({}, {}, [], null, null, { xmrHashrate: 0 });
    expect(partial).toMatchObject({ ramUsedMb: null, ramTotalMb: null, xmrHashrate: 0 });
  });

  // alpha-miner 1.9.4 renders a stats table whose name column is abbreviated
  // ("RTX 5090"), where 1.8.x logged nvidia-smi's exact string. nvidia-smi is
  // the canonical device name, so it wins in EVERY row-building path — this rig
  // must not be labelled one way as a single card and another way as two, which
  // is exactly what happened when the paths disagreed.
  describe('the card name always prefers nvidia-smi over the engine label', () => {
    const SHORT = 'RTX 5090';
    const FULL = 'NVIDIA GeForce RTX 5090';
    const settings = { address: 'prl1pabc', worker: 'rig01', region: 'us1' };
    const vram = (n) => Array.from({ length: n }, (_, i) => (
      { index: i, name: FULL, usedMb: 1000, totalMb: 32607 }));

    test('aggregate-only snapshot on a single card', () => {
      const snap = { gpu: SHORT, total: 310, accepted: 3, gpus: [] };
      expect(buildMinerReports(settings, snap, vram(1)).map((r) => r.gpu)).toEqual([FULL]);
    });

    test('per-card snapshot', () => {
      const snap = { gpu: SHORT, total: 310, accepted: 3,
        gpus: [{ index: 0, gpu: SHORT, hashrate: 310, accepted: 3 }] };
      expect(buildMinerReports(settings, snap, vram(1)).map((r) => r.gpu)).toEqual([FULL]);
    });

    test('a multi-GPU rig, with and without engine data for every card', () => {
      // multi-GPU rig with no per-card engine data
      const aggregate = { gpu: SHORT, total: 620, accepted: 6, gpus: [] };
      expect(buildMinerReports(settings, aggregate, vram(2)).map((r) => r.gpu)).toEqual([FULL, FULL]);
      // the engine reports one of the two cards
      const under = { gpu: SHORT, total: 620, accepted: 6,
        gpus: [{ index: 0, gpu: SHORT, hashrate: 620, accepted: 6 }] };
      expect(buildMinerReports(settings, under, vram(2)).map((r) => r.gpu)).toEqual([FULL, FULL]);
    });

    // The engine's label is still worth keeping: a slim container without
    // nvidia-smi has nothing else to name the card with.
    test('falls back to the engine label when nvidia-smi is absent', () => {
      const snap = { gpu: SHORT, total: 310, accepted: 3,
        gpus: [{ index: 0, gpu: SHORT, hashrate: 310, accepted: 3 }] };
      expect(buildMinerReports(settings, snap, []).map((r) => r.gpu)).toEqual([SHORT]);
      expect(buildMinerReports(settings, { gpu: SHORT, total: 310, gpus: [] }, []).map((r) => r.gpu)).toEqual([SHORT]);
    });
  });
});

describe('readRamMb', () => {
  const GB = 2 ** 30;
  const fakeOs = (total, free) => ({ totalmem: () => total, freemem: () => free });
  // A process with no memory limit, as on Windows, macOS or bare-metal Linux.
  const unlimited = {};

  test('whole MB, used being total minus free', () => {
    expect(readRamMb(fakeOs(32 * GB, 20 * GB), unlimited)).toEqual({ usedMb: 12288, totalMb: 32768 });
    // A size that isn't a whole number of MB rounds to one.
    expect(readRamMb(fakeOs(16 * GB - 300 * 1024, 0), unlimited)).toEqual({ usedMb: 16384, totalMb: 16384 });
  });

  // Nonsense from the OS is not passed on as a number the board would believe.
  test('a reading that makes no sense is null', () => {
    expect(readRamMb(fakeOs(0, 0), unlimited)).toEqual({ usedMb: null, totalMb: null });
    expect(readRamMb(fakeOs(NaN, 0), unlimited)).toEqual({ usedMb: null, totalMb: null });
    // Free above total, below zero, or not a number: the total still stands.
    expect(readRamMb(fakeOs(8 * GB, 9 * GB), unlimited)).toEqual({ usedMb: null, totalMb: 8192 });
    expect(readRamMb(fakeOs(8 * GB, -1), unlimited)).toEqual({ usedMb: null, totalMb: 8192 });
    expect(readRamMb(fakeOs(8 * GB, undefined), unlimited)).toEqual({ usedMb: null, totalMb: 8192 });
  });

  // A container limited to 32 GB on a 512 GB host whose other tenants use 300:
  // the OS figures are the host's, and the board would show 300 / 512.
  test('in a container with a memory limit, the limit is the total and used is the container\'s', () => {
    const host = fakeOs(512 * GB, 212 * GB);
    const container = (available) => ({ constrainedMemory: () => 32 * GB, availableMemory: () => available });
    expect(readRamMb(host, container(20 * GB))).toEqual({ usedMb: 12288, totalMb: 32768 });
    // Use past the limit for a moment: availableMemory() says 0, so all of it.
    expect(readRamMb(host, container(0))).toEqual({ usedMb: 32768, totalMb: 32768 });
    // More available than the limit makes no sense; it counts as none used.
    expect(readRamMb(host, container(40 * GB))).toEqual({ usedMb: 0, totalMb: 32768 });
    // No availableMemory() to ask: the host's free figure says nothing about
    // the container, so used is unknown.
    expect(readRamMb(host, { constrainedMemory: () => 32 * GB })).toEqual({ usedMb: null, totalMb: 32768 });
  });

  // No limit reads as 0, or as a number far past the RAM (cgroup "max"); a
  // limit above the RAM limits nothing. The OS figures stand in each case.
  test('a limit of 0, or one at or above the RAM, is no limit', () => {
    const os = fakeOs(32 * GB, 20 * GB);
    const limited = (limit) => ({ constrainedMemory: () => limit, availableMemory: () => 1 * GB });
    for (const limit of [0, 2 ** 64, 32 * GB, 64 * GB]) {
      expect(readRamMb(os, limited(limit))).toEqual({ usedMb: 12288, totalMb: 32768 });
    }
  });

  // The shells pass no process and get the real one.
  test('reads the real process when given none', () => {
    const spy = jest.spyOn(process, 'constrainedMemory').mockReturnValue(16 * GB);
    const avail = jest.spyOn(process, 'availableMemory').mockReturnValue(4 * GB);
    try {
      expect(readRamMb(fakeOs(32 * GB, 20 * GB))).toEqual({ usedMb: 12288, totalMb: 16384 });
    } finally {
      spy.mockRestore();
      avail.mockRestore();
    }
  });
});

describe('hostStats', () => {
  const os = { totalmem: () => 32 * 2 ** 30, freemem: () => 20 * 2 ** 30 };
  const proc = {};
  const engine = (running, hashrate) => ({ isRunning: () => running, snapshot: () => ({ hashrate }) });

  test('the RAM, and the CPU miner\'s H/s while it runs', () => {
    expect(hostStats(os, engine(true, 1845.2), proc)).toEqual({ ramUsedMb: 12288, ramTotalMb: 32768, xmrHashrate: 1845.2 });
    // Running but off its pool: the engine reads 0, and so does the board.
    expect(hostStats(os, engine(true, 0), proc).xmrHashrate).toBe(0);
    expect(hostStats(os, engine(true, undefined), proc).xmrHashrate).toBe(0);
  });

  // Null, not 0: "this machine is not CPU mining" is a different answer from
  // "it is, at 0 H/s".
  test('no CPU miner, or one that stopped, is null', () => {
    expect(hostStats(os, null, proc).xmrHashrate).toBeNull();
    expect(hostStats(os, undefined, proc).xmrHashrate).toBeNull();
    expect(hostStats(os, engine(false, 1845.2), proc).xmrHashrate).toBeNull();
  });

  // The container's limit, not the host's RAM, as readRamMb reads it.
  test('the RAM is a container\'s own when it has a limit', () => {
    const container = { constrainedMemory: () => 16 * 2 ** 30, availableMemory: () => 4 * 2 ** 30 };
    expect(hostStats(os, null, container)).toEqual({ ramUsedMb: 12288, ramTotalMb: 16384, xmrHashrate: null });
  });
});
