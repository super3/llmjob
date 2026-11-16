'use strict';

// The k trial in the miner: an RX 9070 XT that opens at k 8192 is opened again
// at k 16384 and back in 20 s turns, keeps the faster k, and remembers it for a
// week (shared/kTrial has the method; kTrial.test.js tests it).

jest.mock('../src/main/gpuClocks', () => ({
  lockMemoryClock: jest.fn(() => ({ ok: true, error: null })),
  resetMemoryClock: jest.fn(() => ({ ok: true, error: null })),
}));

const { EventEmitter } = require('events');
const { PearlMiner } = require('../src/main/pearlMiner');
const { encode } = require('../src/shared/miner/stratum');

const ADDR = 'prl1px5ervx6ftaegmdhqa5ajemh20j2uw7l9jt5j5s97rljp72yt3s8qncrxud';
const HEADER = '000000203a49fea8b6d42c60c543fe0f029749787679372495e5d2d1007e29e3'
  + '25e1c08065a0cedc057ba091aabd10476017b3d3f4e38eafb59707ded57f71e1feb7b5d155918b6a0dea0018';
const TARGET = '00000000000007fff80000000000000000000000000000000000000000000000';
const NAME = 'AMD Radeon RX 9070 XT';
const GPUS = [{ index: 0, name: NAME }];
const KEY = 'gpu0 ' + NAME;
const CARD = 'GPU 0 (' + NAME + ')';
const DAY = 24 * 3600 * 1000;
const T0 = Date.UTC(2026, 10, 16, 21);
const settings = { address: ADDR, worker: 'rig01', endpoint: 'us.pearl.herominers.com:1200', gpus: GPUS };

function memStore(data = {}) {
  return {
    data,
    get: jest.fn((key) => data[key] || null),
    set: jest.fn((key, v) => { data[key] = v; }),
    remove: jest.fn((key) => { delete data[key]; }),
  };
}

// A rig of one card. Its core, like the AMD core: k 8192 on a gfx1201 card when
// the miner lets it choose (cardK), kPin when asked and the VRAM allows (`over.max`),
// and the card's arch at the end of its fold's description.
function rig(over = {}) {
  const t = { now: T0 };
  const made = [];
  const arch = over.arch || 'gfx1201';
  const createCore = jest.fn((profile, opts) => {
    if (over.throwAt && over.throwAt(createCore.mock.calls.length - 1)) throw new Error('out of memory');
    const c = new EventEmitter();
    c.setJob = jest.fn();
    c.stop = jest.fn();
    c.opts = opts;
    // As the real core does, it names the card it opened, chosen or not.
    c.device = { index: opts.deviceIndex == null ? 0 : opts.deviceIndex, name: NAME, fold: 'wmma12 ' + arch };
    const max = over.max || 16384;
    c.k = opts.cardK ? Math.min(opts.kPin || over.k || 8192, max) : 2048;
    if (opts.cardK && opts.kPin > max) c.kNote = 'k ' + max + ': k ' + opts.kPin + ' needs ~6961 MiB and 6000 MiB is free';
    made.push(c);
    return c;
  });
  let sock = null;
  const connect = () => {
    sock = new EventEmitter();
    sock.write = jest.fn(() => true);
    sock.destroy = jest.fn();
    return sock;
  };
  const store = over.store === null ? null : (over.store || memStore());
  const m = new PearlMiner({
    connect, createCore, reconnectMs: 0, restartMs: 10, now: () => t.now, kStore: store, env: over.env || {},
  });
  const logs = [];
  m.on('log', (l) => logs.push(l.line));
  m.on('error', () => {});
  const last = () => made[made.length - 1];
  // `secs` of samples, two a second, at `rate(k)` TH/s from the card's current core.
  const mine = (secs, rate) => {
    for (let i = 0; i < secs * 2; i++) {
      t.now += 500;
      last().emit('hashrate', rate(last().k));
    }
  };
  const begin = () => {
    m.start(settings);
    sock.emit('connect');
    sock.emit('data', Buffer.from('{"id":1,"result":true,"error":null}\n'
      + encode({ id: null, method: 'mining.notify', params: { job_id: 'j1', header: HEADER, target: TARGET, height: 1 } })));
  };
  return { m, t, made, createCore, store, logs, mine, begin, last };
}

const TIMING = CARD + ': timing k 8192 against k 16384 while it mines, about 150 s';

describe('the k trial', () => {
  test('keeps k 16384 on a PC where it is faster, and remembers it', () => {
    const r = rig();
    r.begin();
    expect(r.made).toHaveLength(1);
    expect(r.made[0].opts).not.toHaveProperty('kPin');
    expect(r.logs).toContain(TIMING);
    r.mine(61, (k) => (k === 16384 ? 137.2 : 136.3));
    // Opened again at k 16384, on fresh salts, and given the job.
    expect(r.made).toHaveLength(2);
    expect(r.made[0].stop).toHaveBeenCalled();
    expect(r.made[1].opts).toMatchObject({ kPin: 16384, cardK: true });
    expect(r.made[1].opts.saltBase).not.toBe(r.made[0].opts.saltBase);
    expect(r.made[1].setJob).toHaveBeenCalled();
    expect(r.logs).toContain('mining on GPU 0 · ' + NAME + ' · k 16384');
    r.mine(90, (k) => (k === 16384 ? 137.2 : 136.3));
    expect(r.made).toHaveLength(4);
    expect(r.made.map((c) => c.k)).toEqual([8192, 16384, 8192, 16384]);
    // The last turn's core mines on.
    expect(r.last().stop).not.toHaveBeenCalled();
    expect(r.store.data[KEY]).toEqual({ k: 16384, at: expect.any(Number), baseTh: 136.3, wideTh: 137.2 });
    expect(r.logs).toContain(CARD + ': k trial kept k 16384: 137.2 TH/s at k 16384, 136.3 at k 8192 (+0.66%; turns 136.30, 137.20, 137.20, 136.30). '
      + 'Timed again after 2026-11-23');
    // Nothing more after the trial.
    r.mine(60, () => 137);
    expect(r.made).toHaveLength(4);
    // A later open in this run (a restart) keeps k 16384.
    r.last().emit('error', new Error('GPU 0: lost'));
    return new Promise((done) => setTimeout(done, 30)).then(() => {
      expect(r.made).toHaveLength(5);
      expect(r.last().opts.kPin).toBe(16384);
      r.m.stop();
    });
  });

  test('keeps k 8192 where they are level or k 16384 is slower, opening the card once more', () => {
    for (const wide of [136.4, 134.2]) {
      const r = rig();
      r.begin();
      r.mine(151, (k) => (k === 16384 ? wide : 136.3));
      expect(r.made).toHaveLength(3);
      expect(r.last().k).toBe(8192);
      expect(r.last().opts).not.toHaveProperty('kPin');
      expect(r.store.data[KEY]).toMatchObject({ k: 8192, baseTh: 136.3, wideTh: wide });
      expect(r.logs.filter((l) => l.startsWith(CARD + ': k trial kept k 8192: ')).length).toBe(1);
      r.mine(60, () => 136);
      expect(r.made).toHaveLength(3);
      r.m.stop();
    }
    const level = rig();
    level.begin();
    level.mine(151, (k) => (k === 16384 ? 136.4 : 136.3));
    expect(level.logs).toContain(CARD + ': k trial kept k 8192: 136.4 TH/s at k 16384, 136.3 at k 8192 (+0.07%; turns 136.30, 136.40, 136.40, 136.30); '
      + 'k 16384 was not ahead by 0.3%. Timed again after 2026-11-23');
    level.m.stop();
  });

  test('a trial with too few samples keeps k 8192 and is not remembered', () => {
    const r = rig();
    r.begin();
    for (let i = 0; i < 16; i++) {
      r.t.now += 10000;
      r.last().emit('hashrate', 136);
    }
    expect(r.store.data).toEqual({});
    expect(r.logs).toContain(CARD + ': k trial kept k 8192; too few hashrate samples to compare');
    expect(r.last().opts).not.toHaveProperty('kPin');
    r.m.stop();
  });

  test('a remembered answer is used with no trial', () => {
    const wide = rig({ store: memStore({ [KEY]: { k: 16384, at: T0 - DAY, baseTh: 136.3, wideTh: 137.2 } }) });
    wide.begin();
    expect(wide.made).toHaveLength(1);
    expect(wide.made[0].opts.kPin).toBe(16384);
    expect(wide.logs).toContain(CARD + ': k 16384, from its k trial of 2026-11-15 (137.2 TH/s at k 16384, 136.3 at '
      + 'k 8192). Timed again after 2026-11-22');
    wide.mine(100, () => 137);
    expect(wide.made).toHaveLength(1);
    // Not written back with anything of the miner's.
    expect(wide.store.set).not.toHaveBeenCalled();
    wide.m.stop();

    const base = rig({ store: memStore({ [KEY]: { k: 8192, at: T0 - DAY, baseTh: 136.3, wideTh: 134.2 } }) });
    base.begin();
    expect(base.made[0].opts).not.toHaveProperty('kPin');
    expect(base.logs.some((l) => l.startsWith(CARD + ': k 8192, from its k trial of 2026-11-15'))).toBe(true);
    base.mine(100, () => 136);
    expect(base.made).toHaveLength(1);
    base.m.stop();

    // A week old, or LLMJOB_K_TRIAL_RETRY=1: timed again.
    const old = rig({ store: memStore({ [KEY]: { k: 16384, at: T0 - 8 * DAY, baseTh: 1, wideTh: 2 } }) });
    old.begin();
    expect(old.made[0].opts).not.toHaveProperty('kPin');
    expect(old.logs).toContain(TIMING);
    old.m.stop();
    const retry = rig({
      store: memStore({ [KEY]: { k: 16384, at: T0 - DAY, baseTh: 1, wideTh: 2 } }), env: { LLMJOB_K_TRIAL_RETRY: '1' },
    });
    retry.begin();
    expect(retry.logs).toContain(TIMING);
    retry.m.stop();
  });

  test('says nothing of a remembered k the card did not open at', () => {
    const r = rig({ max: 8192, store: memStore({ [KEY]: { k: 16384, at: T0 - DAY, baseTh: 1, wideTh: 2 } }) });
    r.begin();
    expect(r.made[0].k).toBe(8192);
    expect(r.logs.some((l) => l.includes('from its k trial'))).toBe(false);
    expect(r.logs).not.toContain(TIMING);
    r.m.stop();
  });

  test('forgets a k 16384 answer another kind of card opened with, and opens it again at its own k', () => {
    const r = rig({
      arch: 'gfx1200', store: memStore({ auto: { k: 16384, at: T0 - DAY, baseTh: 1, wideTh: 2 } }),
    });
    r.m.start({ ...settings, gpus: [] });
    expect(r.made[0].opts.kPin).toBe(16384);
    expect(r.store.data).toEqual({});
    return new Promise((done) => setImmediate(done)).then(() => {
      expect(r.made).toHaveLength(2);
      expect(r.made[0].stop).toHaveBeenCalled();
      expect(r.made[1].opts).not.toHaveProperty('kPin');
      r.m.stop();
    });
  });

  test('times nothing on other cards, or when told not to', () => {
    const cases = [
      { arch: 'gfx1200' },
      { k: 4096 },
      { store: null },
      { env: { LLMJOB_K_TRIAL: '0' } },
      { env: { PEARL_AMD_K: '8192' } },
    ];
    for (const c of cases) {
      const r = rig(c);
      r.begin();
      r.mine(100, () => 136);
      expect([c, r.made.length]).toEqual([c, 1]);
      expect(r.logs).not.toContain(TIMING);
      r.m.stop();
    }
  });

  test('ends when the card cannot open at k 16384', () => {
    const r = rig({ max: 8192 });
    r.begin();
    r.mine(61, () => 136);
    expect(r.made).toHaveLength(2);
    expect(r.logs).toContain(CARD + ': k trial ended: asked for k 16384 and the core opened k 8192 (k 8192: k 16384 '
      + 'needs ~6961 MiB and 6000 MiB is free). Timing it again at the next start');
    r.mine(100, () => 136);
    expect(r.made).toHaveLength(2);
    expect(r.store.data).toEqual({});
    r.m.stop();
  });

  test('ends, unanswered, when something else opens the card', () => {
    const r = rig();
    r.begin();
    r.mine(61, () => 136);
    expect(r.last().k).toBe(16384);
    r.last().emit('error', new Error('GPU 0: lost'));
    return new Promise((done) => setTimeout(done, 30)).then(() => {
      expect(r.made).toHaveLength(3);
      expect(r.last().opts).not.toHaveProperty('kPin');
      expect(r.last().k).toBe(8192);
      r.mine(100, () => 136);
      expect(r.made).toHaveLength(3);
      expect(r.store.data).toEqual({});
      r.m.stop();
    });
  });

  test('a card that will not open again for its next turn goes down as any card does', () => {
    const r = rig({ throwAt: (n) => n === 1 });
    r.begin();
    const zero = [];
    r.m.on('hashrate', (th, device) => { if (th === 0) zero.push(device.index); });
    r.mine(61, () => 136);
    expect(r.logs).toContain('could not open ' + CARD + ' again for its k trial: out of memory');
    expect(zero).toEqual([0]);
    return new Promise((done) => setTimeout(done, 30)).then(() => {
      // Restarted at the core's own k, with no trial.
      expect(r.made).toHaveLength(2);
      expect(r.last().opts).not.toHaveProperty('kPin');
      r.m.stop();
    });
  });

  test('a sample from a closed core is not counted, and an old core cannot reopen the card', () => {
    const r = rig();
    r.begin();
    const first = r.made[0];
    r.mine(61, () => 136);
    first.emit('hashrate', 500);
    expect(r.made).toHaveLength(2);
    r.m._reopenForK({ core: first, slot: {} });
    expect(r.made).toHaveLength(2);
    r.m.stop();
  });
});
