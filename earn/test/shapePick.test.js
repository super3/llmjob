'use strict';

const { EventEmitter } = require('events');
const {
  TRIAL_SECS, MAX_AGE_MS, shapeKey, readTrial, trialLine, withShapePick, shapeFileStore,
} = require('../src/shared/shapePick');

// A core as the addon makes one: shapes for an RTX 30 card, the one it runs, and events.
function fakeCore({ shapes = ['flip 128', 'band 8 x 512', 'band 4 x 512'], shape, device = { index: 0 } } = {}) {
  const core = new EventEmitter();
  core.shapes = shapes;
  core.shape = shape == null ? shapes[0] || '' : shape;
  core.device = device;
  return core;
}

function memStore(init = {}) {
  const data = { ...init };
  return {
    data,
    get: (k) => data[k] || null,
    set: (k, v) => { data[k] = v; },
    remove: (k) => { delete data[k]; },
  };
}

const NOW = 1_800_000_000_000;

describe('shapeKey', () => {
  test('names the card by index and, when listed, its name', () => {
    expect(shapeKey({ deviceIndex: 1 }, [{ index: 1, name: 'RTX 3090' }])).toBe('gpu1 RTX 3090');
    expect(shapeKey({ deviceIndex: 2 }, [{ index: 1, name: 'RTX 3090' }, null])).toBe('gpu2');
    expect(shapeKey({ deviceIndex: 0 }, [{ index: 0 }])).toBe('gpu0');
    expect(shapeKey({ deviceIndex: 0 }, 'nope')).toBe('gpu0');
  });
  test('a core that picks its own card is "auto"', () => {
    expect(shapeKey({}, [])).toBe('auto');
    expect(shapeKey(undefined, [])).toBe('auto');
  });
});

describe('readTrial and trialLine', () => {
  test('parses the core report and refuses junk', () => {
    expect(readTrial('{"shape":"flip 128","rule":"band 8 x 512","trial":[]}').shape).toBe('flip 128');
    expect(readTrial('not json')).toBeNull();
    expect(readTrial('{"shape":""}')).toBeNull();
    expect(readTrial('null')).toBeNull();
  });
  test('says what was kept and how each shape did', () => {
    expect(trialLine({ shape: 'flip 128', rule: 'band 8 x 512',
      trial: [{ shape: 'flip 128', pct: 1.234, ahead: 9, rounds: 10 }, { shape: 'band 4 x 512', pct: -0.5, ahead: 1, rounds: 10 }] }))
      .toBe('flip 128 over the default band 8 x 512 · against the default: flip 128 +1.23% (ahead 9 of 10), '
        + 'band 4 x 512 -0.50% (ahead 1 of 10)');
    expect(trialLine({ shape: 'band 8 x 512', rule: 'band 8 x 512' })).toBe('kept the default band 8 x 512');
  });
});

describe('withShapePick', () => {
  test('passes a null factory through', () => {
    expect(withShapePick(null, {})).toBeNull();
    expect(withShapePick(undefined)).toBeNull();
  });

  // main/pearlCore hangs the addon's core clock offset calls on the factory it
  // returns, and the miner reads them off the factory it is given.
  test('keeps the factory\'s core clock offset calls on the factory it returns', () => {
    const create = () => null;
    create.clockOffsets = { readCoreOffset: () => ({ ok: true }), setCoreOffset: () => ({ ok: true }) };
    expect(withShapePick(create, { store: memStore(), env: {} }).clockOffsets).toBe(create.clockOffsets);
    expect(withShapePick(() => null, { store: memStore(), env: {} }).clockOffsets).toBeUndefined();
  });

  test('first start: asks for a trial, caches the pick and logs it', () => {
    const store = memStore();
    const lines = [];
    const core = fakeCore();
    const create = jest.fn(() => core);
    const make = withShapePick(create, { store, log: (l, s) => lines.push([l, s]), gpus: [{ index: 0, name: 'RTX 3090' }], now: () => NOW, env: {} });
    expect(make({ k: 2048 }, { deviceIndex: 0, saltBase: 0 })).toBe(core);
    expect(create).toHaveBeenCalledWith({ k: 2048 }, { deviceIndex: 0, saltBase: 0, shapeTrialSecs: TRIAL_SECS });
    expect(lines[0][1]).toMatch(/^GPU 0: timing its launch shapes .*flip 128, band 8 x 512/);
    core.emit('shape', JSON.stringify({ shape: 'band 8 x 512', rule: 'flip 128', trial: [{ shape: 'band 8 x 512', pct: 0.8, ahead: 12, rounds: 12 }] }));
    expect(store.data['gpu0 RTX 3090']).toEqual({ shape: 'band 8 x 512', at: NOW });
    expect(lines[1]).toEqual(['info', 'GPU 0: launch shape trial band 8 x 512 over the default flip 128 · against the default: band 8 x 512 +0.80% (ahead 12 of 12)']);
    core.emit('shape', 'garbage');
    expect(lines[2]).toEqual(['warn', 'GPU 0: launch shape trial sent a report that does not parse']);
  });

  test('later starts: pass the cached shape and run no trial', () => {
    const store = memStore({ gpu0: { shape: 'band 8 x 512', at: NOW - 1000 } });
    const lines = [];
    const core = fakeCore({ shape: 'band 8 x 512' });
    const create = jest.fn(() => core);
    withShapePick(create, { store, log: (l, s) => lines.push(s), now: () => NOW, env: {} })({}, { deviceIndex: 0 });
    expect(create.mock.calls[0][1]).toEqual({ deviceIndex: 0, shape: 'band 8 x 512' });
    expect(lines).toEqual(['GPU 0: launch shape band 8 x 512 (timed on this PC)']);
  });

  test('a cached shape the core lacks is dropped, so the next start times again', () => {
    const store = memStore({ auto: { shape: 'flip 64', at: NOW } });
    const lines = [];
    const core = fakeCore({ device: null });
    withShapePick(() => core, { store, log: (l, s) => lines.push(s), now: () => NOW, env: {} })({}, undefined);
    expect(store.data.auto).toBeUndefined();
    expect(lines[0]).toBe('the GPU: launch shape flip 64 is not one this core has; mining with flip 128, '
      + 'and timing the shapes at the next start');
  });

  test('a week-old pick is timed again', () => {
    const store = memStore({ gpu0: { shape: 'flip 128', at: NOW - MAX_AGE_MS - 1 } });
    const create = jest.fn(() => fakeCore());
    withShapePick(create, { store, now: () => NOW, env: {} })({}, { deviceIndex: 0 });
    expect(create.mock.calls[0][1]).toEqual({ deviceIndex: 0, shapeTrialSecs: TRIAL_SECS });
  });

  test('PEARL_SHAPE forces a shape and leaves the cache alone', () => {
    const store = memStore({ gpu0: { shape: 'flip 128', at: NOW } });
    const lines = [];
    const ok = fakeCore({ shape: 'band 4 x 512' });
    const make = withShapePick(() => ok, { store, log: (l, s) => lines.push(s), now: () => NOW, env: { PEARL_SHAPE: 'band 4 x 512' } });
    make({}, { deviceIndex: 0 });
    expect(lines[0]).toBe('GPU 0: launch shape band 4 x 512 (PEARL_SHAPE)');
    const other = fakeCore({ shape: 'flip 128' });
    withShapePick(() => other, { store, log: (l, s) => lines.push(s), now: () => NOW, env: { PEARL_SHAPE: 'band 16 x 512' } })({}, { deviceIndex: 0 });
    expect(lines[1]).toBe('GPU 0: launch shape band 16 x 512 is not one this core has; mining with flip 128');
    expect(store.data.gpu0).toEqual({ shape: 'flip 128', at: NOW });
  });

  test('PEARL_SHAPE_TRIAL=0 opens the card as the rule says', () => {
    const create = jest.fn(() => fakeCore());
    withShapePick(create, { store: memStore(), now: () => NOW, env: { PEARL_SHAPE_TRIAL: '0' } })({}, {});
    expect(create.mock.calls[0][1]).toEqual({});
  });

  test('a core with nothing to choose, or none at all, opens as before', () => {
    const lines = [];
    const plain = fakeCore({ shapes: [] });
    expect(withShapePick(() => plain, { store: memStore(), log: (l, s) => lines.push(s), env: {} })({}, {})).toBe(plain);
    const old = new EventEmitter();
    expect(withShapePick(() => old, { store: memStore(), env: {} })({}, {})).toBe(old);
    expect(withShapePick(() => null, { store: memStore(), env: {} })({}, {})).toBeNull();
    expect(lines).toEqual([]);
  });

  // The miner holds a card's core clock offset trial back while this says true
  // (main/pearlMiner, _feedCoreOffsetTrial).
  describe('timingShapes says whether a core\'s trial is under way', () => {
    const opts = (over = {}) => ({ store: memStore(), now: () => NOW, env: {}, ...over });

    test('from the open that asked for the trial until the core reports, parsed or not', () => {
      for (const report of ['{"shape":"flip 128","rule":"flip 128"}', 'garbage']) {
        const core = fakeCore();
        const make = withShapePick(() => core, opts());
        expect(make.timingShapes(core)).toBe(false);
        make({}, { deviceIndex: 0 });
        expect(make.timingShapes(core)).toBe(true);
        core.emit('shape', report);
        expect(make.timingShapes(core)).toBe(false);
      }
    });

    test('never for a core that times nothing', () => {
      const none = (core, over) => {
        const make = withShapePick(() => core, opts(over));
        make({}, { deviceIndex: 0 });
        return make.timingShapes(core);
      };
      // A cached pick, a forced shape, trials turned off.
      expect(none(fakeCore(), { store: memStore({ gpu0: { shape: 'flip 128', at: NOW } }) })).toBe(false);
      expect(none(fakeCore(), { env: { PEARL_SHAPE: 'flip 128' } })).toBe(false);
      expect(none(fakeCore(), { env: { PEARL_SHAPE_TRIAL: '0' } })).toBe(false);
      // One shape or none: the core runs no trial, so no report would end the wait.
      expect(none(fakeCore({ shapes: ['flip 128'] }))).toBe(false);
      expect(none(fakeCore({ shapes: [] }))).toBe(false);
      // A core this factory did not open.
      expect(withShapePick(() => null, opts()).timingShapes(fakeCore())).toBe(false);
    });
  });

  test('defaults: Date.now and process.env, and a silent log', () => {
    const create = jest.fn(() => fakeCore());
    const before = process.env.PEARL_SHAPE;
    delete process.env.PEARL_SHAPE;
    const trial = process.env.PEARL_SHAPE_TRIAL;
    delete process.env.PEARL_SHAPE_TRIAL;
    try {
      const store = memStore();
      const core = fakeCore();
      withShapePick(() => core, { store })({}, { deviceIndex: 3 });
      core.emit('shape', '{"shape":"flip 128","rule":"flip 128"}');
      expect(store.data.gpu3.shape).toBe('flip 128');
      expect(store.data.gpu3.at).toBeGreaterThan(0);
    } finally {
      if (before !== undefined) process.env.PEARL_SHAPE = before;
      if (trial !== undefined) process.env.PEARL_SHAPE_TRIAL = trial;
    }
    expect(create).not.toHaveBeenCalled();
  });
});

describe('shapeFileStore', () => {
  test('reads once, writes whole on each change, and removes', () => {
    const files = { '/d/launch-shapes.json': { gpu0: { shape: 'flip 128', at: 1 } } };
    const readSettings = jest.fn((f) => ({ ...files[f] }));
    const writeSettings = jest.fn((f, d) => { files[f] = { ...d }; });
    const s = shapeFileStore('/d/launch-shapes.json', { readSettings, writeSettings, fs: {}, log: () => {} });
    expect(s.get('gpu0')).toEqual({ shape: 'flip 128', at: 1 });
    expect(s.get('gpu1')).toBeNull();
    s.set('gpu1', { shape: 'band 8 x 512', at: 2 });
    expect(files['/d/launch-shapes.json'].gpu1).toEqual({ shape: 'band 8 x 512', at: 2 });
    s.remove('gpu9');
    expect(writeSettings).toHaveBeenCalledTimes(1);
    s.remove('gpu0');
    expect(files['/d/launch-shapes.json'].gpu0).toBeUndefined();
    expect(readSettings).toHaveBeenCalledTimes(1);
  });

  // The CLI keeps the file beside node.json, and a mining-only or --no-serve
  // run never writes node.json, so the folder may not be there yet. Against
  // the real settings store and a real disk: without the folder every save
  // failed with ENOENT, and the trial ran again on every start.
  test('makes the folder the file goes in, so a first run\'s answer is kept', () => {
    const fs = require('fs');
    const os = require('os');
    const path = require('path');
    const { readSettings, writeSettings } = require('../src/shared/settingsStore');
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'shape-store-'));
    try {
      const file = path.join(root, 'share', 'llmjob-earn', 'launch-shapes.json');
      const log = jest.fn();
      const open = () => shapeFileStore(file, { readSettings, writeSettings, fs, log });
      const first = open();
      expect(first.get('gpu0')).toBeNull();
      first.set('gpu0', { shape: 'flip 128', at: 1 });
      first.set('gpu1', { shape: 'band 8 x 512', at: 2 });
      expect(log).not.toHaveBeenCalled();
      // The next start reads what this one wrote.
      expect(open().get('gpu0')).toEqual({ shape: 'flip 128', at: 1 });
      const next = open();
      next.remove('gpu0');
      expect(open().get('gpu0')).toBeNull();
      expect(open().get('gpu1')).toEqual({ shape: 'band 8 x 512', at: 2 });
    } finally {
      fs.rmSync(root, { recursive: true, force: true });
    }
  });

  // A folder that cannot be made is the write's failure to report, not a throw.
  test('a folder that cannot be made leaves the save to fail in the settings store\'s words', () => {
    const writeSettings = jest.fn();
    const fs = { mkdirSync: () => { throw new Error('EACCES'); } };
    const s = shapeFileStore('/d/launch-shapes.json', { readSettings: () => ({}), writeSettings, fs, log: () => {} });
    expect(() => s.set('gpu0', { shape: 'flip 128', at: 1 })).not.toThrow();
    expect(writeSettings).toHaveBeenCalledTimes(1);
  });
});
