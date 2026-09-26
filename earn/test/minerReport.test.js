'use strict';

const { buildMinerReports } = require('../src/shared/minerReport');

// What a row carries beyond its hashrate when the report has no telemetry and no
// identity to add — i.e. what every older client effectively sent.
const NO_HEALTH = { tempC: null, powerW: null, powerLimitW: null, coreClockMhz: null, memClockMhz: null, fanPct: null };
const NO_META = { client: null, os: null, driver: null, uptimeSec: 0, lastShareSec: null };

describe('buildMinerReports', () => {
  test('one card → one rig row with that card\'s own VRAM and hashrate', () => {
    const snap = { gpu: 'NVIDIA GeForce RTX 4090', total: 285.8, accepted: 5,
      gpus: [{ index: 0, gpu: 'NVIDIA GeForce RTX 4090', hashrate: 285.8, accepted: 5 }] };
    const vram = [{ index: 0, name: 'NVIDIA GeForce RTX 4090', usedMb: 4096, totalMb: 24564 }];
    expect(buildMinerReports({ address: '  prl1pabc ', worker: 'rig9', region: 'eu1' }, snap, vram, '0.1.16')).toEqual([
      { address: 'prl1pabc', worker: 'rig9', region: 'eu1', version: '0.1.16',
        gpu: 'NVIDIA GeForce RTX 4090', hashrate: 285.8, accepted: 5, vramUsedMb: 4096, vramTotalMb: 24564, rejected: 0, ...NO_HEALTH, ...NO_META },
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
      { address: 'prl1pabc', worker: 'rig01/gpu0', region: 'us2', version: '0.1.16', gpu: 'RTX 4090', hashrate: 200, accepted: 10, vramUsedMb: 4096, vramTotalMb: 24564, rejected: 0, ...NO_HEALTH, ...NO_META },
      { address: 'prl1pabc', worker: 'rig01/gpu1', region: 'us2', version: '0.1.16', gpu: 'RTX 4060 Ti', hashrate: 100, accepted: 4, vramUsedMb: 2000, vramTotalMb: 16380, rejected: 0, ...NO_HEALTH, ...NO_META },
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
      { address: 'prl1pabc', worker: 'rig01/gpu0', region: 'us2', version: null, gpu: 'RTX 4090', hashrate: 50, accepted: 4, vramUsedMb: 4096, vramTotalMb: 24564, rejected: 0, ...NO_HEALTH, ...NO_META },
      { address: 'prl1pabc', worker: 'rig01/gpu1', region: 'us2', version: null, gpu: 'RTX 4090', hashrate: 50, accepted: 4, vramUsedMb: 2048, vramTotalMb: 24564, rejected: 0, ...NO_HEALTH, ...NO_META },
    ]);
  });

  test('single GPU, no per-card data → one bare-worker row named from the snapshot', () => {
    const snap = { gpu: 'RTX 4090', total: 120, accepted: 3, gpus: [] };
    const vram = [{ index: 0, name: 'RTX 4090', usedMb: 4096, totalMb: 24564 }];
    expect(buildMinerReports({ address: 'prl1pabc', worker: 'rig01', region: 'us2' }, snap, vram)).toEqual([
      { address: 'prl1pabc', worker: 'rig01', region: 'us2', version: null,
        gpu: 'RTX 4090', hashrate: 120, accepted: 3, vramUsedMb: 4096, vramTotalMb: 24564, rejected: 0, ...NO_HEALTH, ...NO_META },
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
      { address: '', worker: 'rig01', region: 'us2', version: null, gpu: null, hashrate: 0, accepted: 0, vramUsedMb: 0, vramTotalMb: 0, rejected: 0, ...NO_HEALTH, ...NO_META },
    ]);
  });

  test('falls back to rig01 for a blank worker and zeroes bad numbers', () => {
    const [row] = buildMinerReports({ worker: '   ' }, { total: 'x', accepted: null, gpus: [] });
    expect(row).toMatchObject({ worker: 'rig01', hashrate: 0, accepted: 0 });
  });

  // Per-card health rides on each card's own row, matched by index — a rig-level
  // sum is exactly what hides the one card that is throttling.
  test('each card carries its own health from nvidia-smi, matched by index', () => {
    const snap = { total: 300, accepted: 14, rejected: 2,
      gpus: [
        { index: 0, gpu: 'RTX 4090', hashrate: 200, accepted: 10, rejected: 1, temp: 70 },
        { index: 1, gpu: 'RTX 4070', hashrate: 100, accepted: 4, rejected: 1, temp: 66 },
      ] };
    const vram = [
      { index: 0, name: 'RTX 4090', usedMb: 3000, totalMb: 24564 },
      { index: 1, name: 'RTX 4070', usedMb: 2000, totalMb: 12282 },
    ];
    const telemetry = [
      { index: 1, tempC: 61, powerW: 180.5, powerLimitW: 200, coreClockMhz: 2610, memClockMhz: 10501, fanPct: 48, driver: '580.82' },
      { index: 0, tempC: 64, powerW: 312.4, powerLimitW: 450, coreClockMhz: 2520, memClockMhz: 10501, fanPct: 55, driver: '580.82' },
    ];
    const rows = buildMinerReports({ address: 'prl1pabc', worker: 'rig' }, snap, vram, '0.6.0', { telemetry });
    expect(rows[0]).toMatchObject({ worker: 'rig/gpu0', rejected: 1, tempC: 64, powerW: 312.4, powerLimitW: 450, fanPct: 55, driver: '580.82' });
    expect(rows[1]).toMatchObject({ worker: 'rig/gpu1', rejected: 1, tempC: 61, powerW: 180.5, coreClockMhz: 2610, memClockMhz: 10501 });
  });

  // No nvidia-smi: the engine's own core temperature is the only reading there
  // is, and it is still worth reporting. Everything else stays null.
  test('falls back to the engine temperature, and leaves unknown sensors null', () => {
    const snap = { total: 200, accepted: 3, temp: 71,
      gpus: [{ index: 0, gpu: 'RTX 4090', hashrate: 200, accepted: 3, temp: 69 }] };
    const [row] = buildMinerReports({ address: 'prl1pabc' }, snap, [], '0.6.0', { telemetry: [{ index: 0, powerW: 300 }] });
    expect(row).toMatchObject({ tempC: 69, powerW: 300, powerLimitW: null, coreClockMhz: null, fanPct: null, driver: null });
    // …and a single aggregate row uses the rig-level reading the same way.
    const [agg] = buildMinerReports({ address: 'prl1pabc' }, { total: 1, temp: 71, gpus: [] }, []);
    expect(agg.tempC).toBe(71);
  });

  test('splits rejected shares evenly when the engine gives no per-card figures', () => {
    const rows = buildMinerReports({ address: 'prl1pabc', worker: 'rig' },
      { total: 100, accepted: 10, rejected: 4, gpus: [] },
      [{ index: 0, name: 'A', usedMb: 1, totalMb: 2 }, { index: 1, name: 'B', usedMb: 1, totalMb: 2 }]);
    expect(rows.map((r) => r.rejected)).toEqual([2, 2]);
    const under = buildMinerReports({ address: 'prl1pabc', worker: 'rig' },
      { total: 100, gpus: [{ index: 0, hashrate: 100, accepted: 10, rejected: 6 }] },
      [{ index: 0, name: 'A', usedMb: 1, totalMb: 2 }, { index: 1, name: 'B', usedMb: 1, totalMb: 2 }]);
    expect(under.map((r) => r.rejected)).toEqual([3, 3]);
  });

  test('carries the rig identity, shell, OS, uptime and seconds since the last share on every row', () => {
    const identity = { rigId: 'a1b2c3d4e5f60789', publicKey: 'pk', timestamp: 1000, signature: 'sig' };
    const snap = { total: 1, uptimeSec: 3600, lastShareMs: 50_000,
      gpus: [{ index: 0, hashrate: 1 }, { index: 1, hashrate: 1 }] };
    const rows = buildMinerReports({ address: 'prl1pabc' }, snap, [],
      '0.6.0', { identity, client: 'cli', os: 'linux', nowMs: 62_400 });
    for (const row of rows) {
      expect(row).toMatchObject({
        ...identity, client: 'cli', os: 'linux', uptimeSec: 3600, lastShareSec: 12,
      });
    }
  });

  test('no share yet, or no clock, reads as null rather than a huge age', () => {
    const noShare = buildMinerReports({}, { total: 0, gpus: [] }, [], null, { nowMs: 5000 })[0];
    expect(noShare.lastShareSec).toBeNull();
    const noClock = buildMinerReports({}, { total: 0, lastShareMs: 1000, gpus: [] }, [])[0];
    expect(noClock.lastShareSec).toBeNull();
    // A share stamped a moment ahead of the report's clock is "just now", not negative.
    expect(buildMinerReports({}, { lastShareMs: 9000, gpus: [] }, [], null, { nowMs: 8000 })[0].lastShareSec).toBe(0);
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

    test('the split-rows path, both ways in', () => {
      // multi-GPU rig with no per-card engine data
      const aggregate = { gpu: SHORT, total: 620, accepted: 6, gpus: [] };
      expect(buildMinerReports(settings, aggregate, vram(2)).map((r) => r.gpu)).toEqual([FULL, FULL]);
      // engine under-enumerated the cards
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
