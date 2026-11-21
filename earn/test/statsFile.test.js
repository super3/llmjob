'use strict';

const { statsFilePayload } = require('../src/shared/statsFile');

describe('statsFilePayload', () => {
  test('maps a live snapshot to the stats-file shape (ths stays in TH/s)', () => {
    const snap = { total: 354.1, accepted: 14820, rejected: 3, uptimeSec: 15128, gpu: 'RTX 5090' };
    expect(statsFilePayload(snap, { version: '0.1.14', nowMs: 1783300000000 })).toEqual({
      ver: '0.1.14', algo: 'pearlhash', ths: 354.1, accepted: 14820, rejected: 3,
      uptimeSec: 15128, gpu: 'RTX 5090', updatedMs: 1783300000000,
      schema: 1, mode: null, strategy: null, gate: null, mining: false,
      lastShareMs: null, model: null, tps: { gen: 0, prefill: 0 }, gpus: [], xmr: null,
    });
  });

  test('zeros and nulls a cold snapshot; tolerates missing args entirely', () => {
    expect(statsFilePayload()).toEqual({
      ver: '', algo: 'pearlhash', ths: 0, accepted: 0, rejected: 0,
      uptimeSec: 0, gpu: null, updatedMs: 0,
      schema: 1, mode: null, strategy: null, gate: null, mining: false,
      lastShareMs: null, model: null, tps: { gen: 0, prefill: 0 }, gpus: [], xmr: null,
    });
  });

  test('non-finite or negative hashrate reads as 0, never NaN', () => {
    expect(statsFilePayload({ total: NaN }).ths).toBe(0);
    expect(statsFilePayload({ total: -5 }).ths).toBe(0);
    expect(statsFilePayload({ total: 'x' }).ths).toBe(0);
  });
});

// The CPU miner, for HiveOS's second algorithm. Its hashrate is H/s.
describe('statsFilePayload: the CPU miner', () => {
  test('carries the Monero engine\'s snapshot', () => {
    const xmr = { state: 'mining', hashrate: 1845.2, accepted: 3, rejected: 1, threads: 7, pages: 'thp' };
    expect(statsFilePayload({}, { xmr }).xmr).toEqual({
      algo: 'randomx', hs: 1845.2, accepted: 3, rejected: 1, threads: 7, state: 'mining',
    });
  });

  test('zeros what is missing, and is null with no CPU miner', () => {
    expect(statsFilePayload({}, { xmr: { hashrate: -3 } }).xmr).toEqual({
      algo: 'randomx', hs: 0, accepted: 0, rejected: 0, threads: 0, state: null,
    });
    expect(statsFilePayload({}, { xmr: {} }).xmr.hs).toBe(0);
    expect(statsFilePayload({}, { xmr: null }).xmr).toBeNull();
  });
});

describe('statsFilePayload: what the node is doing', () => {
  // The counters alone cannot express this. A rig in demand mode reading 0 TH/s
  // is not a broken miner, it is one that is busy serving.
  const model = { name: 'Qwen3.8-27B', quant: 'Q4_K_XL', ctxSize: 262144, vision: true };

  test('carries mode, strategy, gate and the mining flag', () => {
    const p = statsFilePayload({}, {
      mode: 'auto', strategy: 'demand', gate: 'SERVING', mining: false,
    });
    expect(p).toMatchObject({ mode: 'auto', strategy: 'demand', gate: 'SERVING', mining: false, schema: 1 });
  });

  test('flattens the served model and both throughput figures', () => {
    const p = statsFilePayload({}, { llm: { model, ready: true, tps: 61.7, promptTps: 1840 } });
    expect(p.model).toEqual({ name: 'Qwen3.8-27B', quant: 'Q4_K_XL', ctxSize: 262144, vision: true, ready: true });
    expect(p.tps).toEqual({ gen: 61.7, prefill: 1840 });
  });

  test('model is null when nothing is loaded, not an object of nulls', () => {
    expect(statsFilePayload({}, { llm: { model: null, ready: false } }).model).toBeNull();
    expect(statsFilePayload({}, { llm: null }).model).toBeNull();
    expect(statsFilePayload({}, {}).model).toBeNull();
  });

  test('a model missing optional fields reads null rather than undefined', () => {
    const p = statsFilePayload({}, { llm: { model: { name: 'x' } } });
    expect(p.model).toEqual({ name: 'x', quant: null, ctxSize: null, vision: false, ready: false });
  });

  test('a nameless model reads null, not undefined', () => {
    expect(statsFilePayload({}, { llm: { model: { ctxSize: 4096 } } }).model)
      .toEqual({ name: null, quant: null, ctxSize: 4096, vision: false, ready: false });
  });

  test('lastShareMs passes through, and null stays null', () => {
    expect(statsFilePayload({ lastShareMs: 1783300000000 }, {}).lastShareMs).toBe(1783300000000);
    expect(statsFilePayload({ lastShareMs: null }, {}).lastShareMs).toBeNull();
    expect(statsFilePayload({}, {}).lastShareMs).toBeNull();
  });

  test('per-card rows pass through, defaulting to an empty array', () => {
    const card = { index: 0, gpu: 'RTX 5090', hashrate: 130, accepted: 9, rejected: 0, power: 600, temp: 61 };
    expect(statsFilePayload({ gpus: [card] }, {}).gpus).toEqual([{ ...card, pciBusId: null }]);
    expect(statsFilePayload({ gpus: 'nope' }, {}).gpus).toEqual([]);
  });

  // HiveOS binds each card's hashrate to a dashboard row by PCI bus. The bus is
  // looked up by the card's own index, so a rig mining on card 1 only gets card
  // 1's bus, not card 0's.
  test('each card carries its PCI bus id, looked up by its index', () => {
    const gpus = [{ index: 1, hashrate: 90 }, { index: 3, hashrate: 80 }];
    const pciBusIds = { 0: '00000000:01:00.0', 1: '00000000:02:00.0', 3: '00000000:81:00.0' };
    expect(statsFilePayload({ gpus }, { pciBusIds }).gpus).toEqual([
      { index: 1, hashrate: 90, pciBusId: '00000000:02:00.0' },
      { index: 3, hashrate: 80, pciBusId: '00000000:81:00.0' },
    ]);
    // The snapshot's own rows are not modified.
    expect(gpus[0]).toEqual({ index: 1, hashrate: 90 });
  });

  test('a card with no known bus id gets null', () => {
    expect(statsFilePayload({ gpus: [{ index: 2 }] }, { pciBusIds: { 0: '00000000:01:00.0' } }).gpus)
      .toEqual([{ index: 2, pciBusId: null }]);
    expect(statsFilePayload({ gpus: [null] }, { pciBusIds: {} }).gpus).toEqual([{ pciBusId: null }]);
  });
});

// A Quantus run (--coin quantus): its snapshot is in H/s. HiveOS's h-stats.sh
// reads algo "quantus" and the rate in MH/s; ths and each card's hashrate keep
// their TH/s for any reader that knows only those.
describe('statsFilePayload: Quantus', () => {
  test('algo quantus, the rate in MH/s, and TH/s where the file always had it', () => {
    const snap = {
      total: 344.5e6, accepted: 30, rejected: 1, uptimeSec: 600, gpu: 'AMD Radeon RX 7800 XT',
      gpus: [{ index: 0, gpu: 'AMD Radeon RX 7800 XT', hashrate: 172.25e6, accepted: 15, rejected: 1, temp: 0 },
        { index: 2, gpu: 'AMD Radeon RX 7800 XT', hashrate: 172.25e6, accepted: 15, rejected: 0, temp: 0 }],
    };
    const p = statsFilePayload(snap, { coin: 'quantus', version: '0.7.7', mining: true, pciBusIds: { 0: '0000:03:00.0' } });
    expect(p).toMatchObject({ algo: 'quantus', mhs: 344.5, accepted: 30, rejected: 1, mining: true });
    expect(p.ths).toBeCloseTo(344.5e-6, 15);
    expect(p.gpus).toEqual([
      { index: 0, gpu: 'AMD Radeon RX 7800 XT', hashrate: 172.25e-6, mhs: 172.25, accepted: 15, rejected: 1, temp: 0,
        pciBusId: '0000:03:00.0' },
      { index: 2, gpu: 'AMD Radeon RX 7800 XT', hashrate: 172.25e-6, mhs: 172.25, accepted: 15, rejected: 0, temp: 0,
        pciBusId: null },
    ]);
    // The snapshot's own rows are not modified.
    expect(snap.gpus[0].hashrate).toBe(172.25e6);
  });

  test('a cold or broken Quantus snapshot reads 0, never NaN', () => {
    expect(statsFilePayload({}, { coin: 'quantus' })).toMatchObject({ algo: 'quantus', ths: 0, mhs: 0, gpus: [] });
    expect(statsFilePayload({ total: -1, gpus: [{ index: 0 }, null] }, { coin: 'quantus' })).toMatchObject({
      ths: 0, mhs: 0, gpus: [{ index: 0, hashrate: 0, mhs: 0, pciBusId: null }, { hashrate: 0, mhs: 0, pciBusId: null }],
    });
  });

  test('Pearl\'s file has no mhs', () => {
    expect(statsFilePayload({ total: 1, gpus: [{ index: 0, hashrate: 1 }] }, {})).not.toHaveProperty('mhs');
    expect(statsFilePayload({ total: 1, gpus: [{ index: 0, hashrate: 1 }] }, { coin: 'pearl' }).gpus[0]).not.toHaveProperty('mhs');
  });
});

// A rig mining Pearl on some cards and Quantus on the others (shared/forkPlan).
describe('statsFilePayload: a rig mining both coins', () => {
  test('a Pearl file, with each Quantus card in MH/s and their total beside it', () => {
    const snap = {
      total: 230, quantusTotal: 9e8, accepted: 5,
      gpus: [{ index: 0, hashrate: 230 }, { index: 1, hashrate: 9e8, coin: 'quantus' }],
    };
    const p = statsFilePayload(snap, { version: '0.7.8', mining: true });
    expect(p).toMatchObject({ algo: 'pearlhash', ths: 230, quantusMhs: 900 });
    expect(p).not.toHaveProperty('mhs');
    expect(p.gpus[0]).toEqual({ index: 0, hashrate: 230, pciBusId: null });
    expect(p.gpus[1]).toEqual({ index: 1, hashrate: 9e-4, mhs: 900, coin: 'quantus', pciBusId: null });
    // A total that is not a number reads 0.
    expect(statsFilePayload({ total: 1, quantusTotal: 'x', gpus: [] }, {}).quantusMhs).toBe(0);
  });
});
