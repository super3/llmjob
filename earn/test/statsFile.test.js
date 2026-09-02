'use strict';

const { statsFilePayload } = require('../src/shared/statsFile');

describe('statsFilePayload', () => {
  test('maps a live snapshot to the stats-file shape (ths stays in TH/s)', () => {
    const snap = { total: 354.1, accepted: 14820, rejected: 3, uptimeSec: 15128, gpu: 'RTX 5090' };
    expect(statsFilePayload(snap, { version: '0.1.14', nowMs: 1783300000000 })).toEqual({
      ver: '0.1.14', algo: 'pearlhash', ths: 354.1, accepted: 14820, rejected: 3,
      uptimeSec: 15128, gpu: 'RTX 5090', updatedMs: 1783300000000,
      schema: 1, mode: null, strategy: null, gate: null, mining: false,
      lastShareMs: null, model: null, tps: { gen: 0, prefill: 0 }, gpus: [],
    });
  });

  test('zeros and nulls a cold snapshot; tolerates missing args entirely', () => {
    expect(statsFilePayload()).toEqual({
      ver: '', algo: 'pearlhash', ths: 0, accepted: 0, rejected: 0,
      uptimeSec: 0, gpu: null, updatedMs: 0,
      schema: 1, mode: null, strategy: null, gate: null, mining: false,
      lastShareMs: null, model: null, tps: { gen: 0, prefill: 0 }, gpus: [],
    });
  });

  test('non-finite or negative hashrate reads as 0, never NaN', () => {
    expect(statsFilePayload({ total: NaN }).ths).toBe(0);
    expect(statsFilePayload({ total: -5 }).ths).toBe(0);
    expect(statsFilePayload({ total: 'x' }).ths).toBe(0);
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
    const gpus = [{ index: 0, gpu: 'RTX 5090', hashrate: 130, accepted: 9, rejected: 0, power: 600, temp: 61 }];
    expect(statsFilePayload({ gpus }, {}).gpus).toBe(gpus);
    expect(statsFilePayload({ gpus: 'nope' }, {}).gpus).toEqual([]);
  });
});
