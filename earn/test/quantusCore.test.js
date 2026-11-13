'use strict';

const path = require('path');
const qc = require('../src/main/quantusCore');
const q = require('../src/shared/quantus/qpow');

const HEADER = Buffer.from('000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f', 'hex');
const MID = q.midstate(HEADER, Buffer.alloc(32));
const TAIL = Buffer.alloc(24, 7);

function work(over = {}) {
  return {
    midstate: q.midstateBytes(MID), tail: TAIL, start: qc.u64Bytes(1000n), count: 40,
    targetHigh: qc.u64Bytes(1n << 61n), maxHits: 64, ...over,
  };
}

describe('the CPU core', () => {
  test('its hits are the reference\'s, across chunks', async () => {
    const core = qc.cpuAddon.createCore({ device: 3 });
    expect(core.device).toEqual({ index: 3, name: 'CPU (reference)', vendor: 'cpu' });
    const count = qc.CPU_CHUNK + 30;
    const res = await core.scan(work({ count }));
    const want = [];
    qc.scanRange({ mid: MID, tail: TAIL, targetHigh: 1n << 61n }, 1000n, 1000n + BigInt(count), want);
    expect(want.length).toBeGreaterThan(3);
    expect(qc.hitList(res)).toEqual(want);
    expect(res.overflow).toBe(false);
    // Each hit's top 64 bits really are at or under the target.
    for (const c of want) expect(q.nonceTop64(MID, TAIL, c) <= 1n << 61n).toBe(true);
  });

  test('maxHits caps the hits, and overflow says so', async () => {
    const core = new qc.CpuCore();
    expect(core.device.index).toBe(0);
    const res = await core.scan(work({ count: 10, targetHigh: qc.u64Bytes(q.U64_MAX), maxHits: 4 }));
    expect(qc.hitList(res)).toEqual([1000n, 1001n, 1002n, 1003n]);
    expect(res.overflow).toBe(true);
    // No maxHits: no hits back.
    expect((await core.scan(work({ count: 2, targetHigh: qc.u64Bytes(q.U64_MAX), maxHits: undefined }))).hits).toHaveLength(0);
  });

  test('a closed core refuses, before and during a scan', async () => {
    const core = new qc.CpuCore();
    const p = core.scan(work({ count: qc.CPU_CHUNK * 2 }));
    core.close();
    await expect(p).rejects.toThrow('the core is closed');
    await expect(core.scan(work())).rejects.toThrow('the core is closed');
  });

  test.each([
    [{ midstate: Buffer.alloc(8) }, 'midstate must be 96 bytes'],
    [{ tail: Buffer.alloc(8) }, 'tail must be 24 bytes'],
    [{ tail: 'x' }, 'tail must be 24 bytes'],
    [{ start: Buffer.alloc(4) }, 'start must be 8 bytes'],
    [{ targetHigh: null }, 'targetHigh must be 8 bytes'],
    [{ count: 0 }, 'count must be 1 to 2^32'],
    [{ count: 2 ** 32 + 1 }, 'count must be 1 to 2^32'],
    [{ count: 1.5 }, 'count must be 1 to 2^32'],
    [{ start: qc.u64Bytes(q.U64_MAX), count: 2 }, 'start + count passes 2^64'],
  ])('work refused: %j', async (over, message) => {
    await expect(new qc.CpuCore().scan(work(over))).rejects.toThrow(message);
  });

  test('no work at all is refused', () => {
    expect(() => qc.readWork(undefined)).toThrow('midstate must be 96 bytes');
  });

  test('hits come back in any order, and are read sorted', () => {
    const hits = Buffer.concat([2n, 1n, 2n].map(qc.u64Bytes));
    expect(qc.hitList({ hits })).toEqual([1n, 2n, 2n]);
  });

  test('the last counter, 2^64 - 1, is scanned', async () => {
    const res = await new qc.CpuCore().scan(work({ start: qc.u64Bytes(q.U64_MAX), count: 1, targetHigh: qc.u64Bytes(q.U64_MAX) }));
    expect(qc.hitList(res)).toEqual([q.U64_MAX]);
  });
});

describe('finding a core', () => {
  const fakeAddon = (over = {}) => ({
    QUANTUS_API: 1, devices: () => [{ index: 0, name: 'Card A' }, { index: 1, name: 'Card B' }],
    createCore: jest.fn((o) => ({ device: { index: o.device, name: 'core' } })), ...over,
  });
  const notFound = () => Object.assign(new Error('nope'), { code: 'MODULE_NOT_FOUND' });

  test('QUANTUS_CORE=cpu: the CPU core, said to be for tests', () => {
    const log = jest.fn();
    const f = qc.quantusCoreFactory({ env: { QUANTUS_CORE: ' CPU ' }, log });
    expect(f).toMatchObject({ label: 'cpu', file: null, devices: [{ index: 0, name: 'CPU (reference)' }] });
    expect(f.createCore(0)).toBeInstanceOf(qc.CpuCore);
    expect(log).toHaveBeenCalledWith('info', expect.stringContaining('for tests, not for mining'));
  });

  test('QUANTUS_CORE_PATH: that file only', () => {
    const addon = fakeAddon();
    const req = jest.fn(() => addon);
    const log = jest.fn();
    const f = qc.quantusCoreFactory({ env: { QUANTUS_CORE_PATH: '/x/my.node' }, require: req, log, execPath: '/opt/earn/earn' });
    expect(req).toHaveBeenCalledTimes(1);
    expect(req).toHaveBeenCalledWith('/x/my.node');
    expect(f).toMatchObject({ label: 'my', file: '/x/my.node' });
    expect(f.createCore(1)).toEqual({ device: { index: 1, name: 'core' } });
    expect(addon.createCore).toHaveBeenCalledWith({ device: 1 });
    expect(log).toHaveBeenCalledWith('info', 'Quantus core: /x/my.node · 2 cards');
  });

  test('the vendor files in order, beside the app and then the dev tree; the first that sees a card wins', () => {
    const tried = [];
    const hip = path.join('/res', 'native', 'quantus_hip.node');
    const req = (file) => {
      tried.push(file);
      if (file.endsWith(path.join('/opt/earn', 'quantus_cuda.node'))) throw new Error('libcuda.so.1: cannot open shared object file');
      if (file.endsWith(path.join('/opt/earn', 'native', 'quantus_cuda.node'))) return { nothing: true };
      if (file.includes(path.join('native', 'quantus', 'build', 'quantus_cuda.node'))) return fakeAddon({ devices: () => [] });
      if (file === hip) return fakeAddon({ devices: () => [{ index: 0, name: 'RX 7900 XTX' }] });
      throw notFound();
    };
    const log = jest.fn();
    const f = qc.quantusCoreFactory({ env: {}, require: req, log, execPath: '/opt/earn/earn', resourcesPath: '/res' });
    expect(f.file).toBe(hip);
    expect(f.label).toBe('quantus_hip');
    expect(f.devices).toEqual([{ index: 0, name: 'RX 7900 XTX' }]);
    // Every CUDA place first, then the HIP core in resources.
    expect(tried.slice(0, 5)).toEqual([
      path.join('/res', 'native', 'quantus_cuda.node'), path.join('/opt/earn', 'quantus_cuda.node'),
      path.join('/opt/earn', 'native', 'quantus_cuda.node'), expect.stringContaining(path.join('native', 'quantus', 'build')), hip,
    ]);
    expect(log).toHaveBeenCalledWith('info', 'Quantus core: ' + hip + ' · 1 card');
  });

  test('nothing usable: each reason, then none found', () => {
    const files = [];
    const req = (file) => {
      files.push(file);
      const i = files.length;
      if (i === 1) throw new Error('bad ELF');
      if (i === 2) return fakeAddon({ QUANTUS_API: 2 });
      if (i === 3) return fakeAddon({ devices: () => { throw new Error('driver too old'); } });
      if (i === 4) return fakeAddon({ devices: () => { throw 'no'; } });
      if (i === 5) return fakeAddon({ devices: () => null });
      if (i === 6) return null;
      if (i === 7) throw null;
      throw notFound();
    };
    const log = jest.fn();
    expect(qc.quantusCoreFactory({ env: {}, require: req, log, execPath: '/opt/earn/earn' })).toBe(null);
    const lines = log.mock.calls.map((c) => c[0] + ' ' + c[1]);
    expect(lines).toEqual([
      'warn Quantus core: ' + files[0] + ' would not load: bad ELF',
      'warn Quantus core: ' + files[1] + ' speaks Quantus core API 2, this miner 1',
      'warn Quantus core: ' + files[2] + ' could not list its cards: driver too old',
      'warn Quantus core: ' + files[3] + ' could not list its cards: no',
      'warn Quantus core: ' + files[4] + ' found no card',
      'warn Quantus core: ' + files[5] + ' is not a Quantus core',
      'warn Quantus core: ' + files[6] + ' would not load: null',
      'error Quantus core: none found (looked for quantus_cuda.node, quantus_hip.node, quantus_sycl.node beside the app and in earn/native/quantus/build)',
    ]);
    // 3 files x (2 beside the executable + the dev tree).
    expect(files).toHaveLength(9);
  });

  test('a QUANTUS_CORE_PATH that is not there', () => {
    const log = jest.fn();
    expect(qc.quantusCoreFactory({ env: { QUANTUS_CORE_PATH: '/nope.node' }, require: () => { throw notFound(); }, log })).toBe(null);
    expect(log).toHaveBeenCalledWith('error', 'Quantus core: none found (QUANTUS_CORE_PATH=/nope.node)');
  });

  test('with no options: the process\'s environment and require, and no log', () => {
    const saved = process.env.QUANTUS_CORE;
    process.env.QUANTUS_CORE = 'cpu';
    try {
      expect(qc.quantusCoreFactory().label).toBe('cpu');
    } finally {
      if (saved === undefined) delete process.env.QUANTUS_CORE;
      else process.env.QUANTUS_CORE = saved;
    }
    // A real require of a file that is not there.
    expect(qc.quantusCoreFactory({ env: { QUANTUS_CORE_PATH: path.join(__dirname, 'nothing-here.node') } })).toBe(null);
  });
});

describe('checking a core against the reference', () => {
  const SMALL = { exactCount: 64, bigCount: 256, random: (n) => Buffer.alloc(n, 0x5a) };

  test('the CPU core passes every check', async () => {
    let t = 0;
    const r = await qc.verifyCore(new qc.CpuCore(), { ...SMALL, now: () => (t += 500) });
    expect(r.checks.map((c) => [c.name, c.ok])).toEqual([
      ['vectors', true], ['exact', true], ['top', true], ['overflow', true], ['big', true],
    ]);
    expect(r.ok).toBe(true);
    expect(r.checks[0].detail).toBe('5 of 5');
    expect(r.checks[4].detail).toMatch(/hits \(about 32 expected\), 0 wrong · 256 nonces in 0\.50 s/);
    expect(r.hashrate).toBe(512);
  });

  test('a core that gets the hash wrong fails, and says where', async () => {
    // Off by one: it reports the counter after each real hit.
    const cpu = new qc.CpuCore();
    const wrong = {
      async scan(w) {
        const res = await cpu.scan(w);
        const shifted = qc.hitList(res).map((c) => (c < q.U64_MAX ? c + 1n : c - 1n));
        return { hits: Buffer.concat(shifted.map(qc.u64Bytes)), overflow: false };
      },
    };
    const r = await qc.verifyCore(wrong, SMALL);
    expect(r.ok).toBe(false);
    const by = Object.fromEntries(r.checks.map((c) => [c.name, c]));
    expect(by.vectors).toMatchObject({ ok: false, detail: 'wrong on vector 0, 1, 2, 3, 4' });
    expect(by.exact.ok).toBe(false);
    expect(by.exact.detail).toMatch(/: they differ$/);
    expect(by.overflow.detail).toBe('16 hits back, overflow false');
    expect(by.big.ok).toBe(false);
  });

  test('a core that finds nothing fails the big check on its count', async () => {
    const none = { scan: async () => ({ hits: Buffer.alloc(0), overflow: false }) };
    const r = await qc.verifyCore(none, { ...SMALL, now: Date.now });
    const big = r.checks.find((c) => c.name === 'big');
    expect(big.ok).toBe(false);
    expect(big.detail).toMatch(/^0 hits/);
  });

  test('the defaults: 4096 for the exact ranges, 2^30 for the big one, random bytes, the clock', async () => {
    const seen = [];
    const fake = { scan: async (w) => { seen.push(w.count); return { hits: Buffer.alloc(0), overflow: false }; } };
    const r = await qc.verifyCore(fake);
    expect(seen).toEqual([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4096, 4096, 64, 2 ** 30]);
    expect(r.ok).toBe(false);
  }, 30000);
});
