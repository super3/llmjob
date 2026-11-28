'use strict';

const fs = require('fs');
const path = require('path');
const nc = require('../src/main/nebCore');
const nh = require('../src/shared/neb/nebhash2');
const hdr = require('../src/shared/neb/header');

const VEC_DIR = path.join(__dirname, '..', 'native', 'neb', 'vectors');
const VECTORS = fs.readFileSync(path.join(VEC_DIR, 'nebhash2.txt'), 'utf8');
const PACKED = Buffer.from(fs.readFileSync(path.join(VEC_DIR, 'syn2-trained.hex'), 'utf8').trim(), 'hex');
const DS = nc.checkDataset();
const WEIGHTS = nh.modelFromPacked(nh.KEP1, PACKED).w;
const HEADER = Buffer.alloc(80, 3);

function work(over = {}) {
  return { slot: 0, header: HEADER, target: nc.ALL_ONES, start: 0, count: 64, maxHits: 64, ...over };
}

function cpuCore() {
  const core = new nc.CpuCore({ device: 2 });
  core.setDataset(0, DS.arch, DS.values, DS.labels);
  return core;
}

describe('the check dataset', () => {
  test('5 samples in 8 are dead on the trained model', () => {
    const live = nh.liveSamples({ arch: nh.KEP1, w: WEIGHTS }, DS);
    expect(live.reduce((a, b) => a + b, 0)).toBe(768);
    expect(Array.from(live.subarray(0, 8))).toEqual([0, 0, 0, 0, 0, 1, 1, 1]);
  });
});

describe('readWork', () => {
  test('takes well-formed work', () => {
    expect(nc.readWork(work())).toEqual(work());
    expect(nc.readWork(work({ maxHits: -1 })).maxHits).toBe(0);
    expect(nc.readWork(work({ start: 2 ** 32 - 1, count: 1 })).start).toBe(2 ** 32 - 1);
  });

  test('refuses the rest', () => {
    expect(() => nc.readWork()).toThrow('header must be 80 bytes');
    expect(() => nc.readWork(work({ header: Buffer.alloc(79) }))).toThrow('header must be 80 bytes');
    expect(() => nc.readWork(work({ target: 'ff' }))).toThrow('target must be 32 bytes');
    expect(() => nc.readWork(work({ target: Buffer.alloc(31) }))).toThrow('target must be 32 bytes');
    expect(() => nc.readWork(work({ start: -1 }))).toThrow('start must be a 32-bit nonce');
    expect(() => nc.readWork(work({ start: 2 ** 32 }))).toThrow('start must be a 32-bit nonce');
    expect(() => nc.readWork(work({ start: 1.5 }))).toThrow('start must be a 32-bit nonce');
    expect(() => nc.readWork(work({ count: 0 }))).toThrow('start + count passes 2^32');
    expect(() => nc.readWork(work({ count: 1.5 }))).toThrow('start + count passes 2^32');
    expect(() => nc.readWork(work({ start: 2 ** 32 - 1, count: 2 }))).toThrow('start + count passes 2^32');
  });
});

describe('hits', () => {
  test('packHits keeps maxHits nonces, little-endian, and says when there were more', () => {
    const r = nc.packHits([7, 0xdeadbeef, 9], 2, 11);
    expect(r).toEqual({ hits: Buffer.from('07000000efbeadde', 'hex'), overflow: true, evaluated: 11 });
    expect(nc.packHits([7], 2, 1).overflow).toBe(false);
    expect(nc.hitList({ hits: Buffer.from('0900000007000000ff', 'hex') })).toEqual([7, 9]);
  });
});

describe('the CPU core', () => {
  test('says what it is', () => {
    expect(new nc.CpuCore().device).toEqual({ index: 0, name: 'CPU (reference)', vendor: 'cpu' });
    expect(cpuCore().device.index).toBe(2);
    expect(nc.cpuAddon.devices()).toEqual([{ index: 0, name: 'CPU (reference)' }]);
    expect(nc.cpuAddon.createCore({ device: 1 }).device.index).toBe(1);
    expect(nc.cpuAddon.NEB_API).toBe(nc.API);
  });

  test('setModel reports the live samples; a scan evaluates only the attempts with at most 2', async () => {
    const core = cpuCore();
    expect(await core.setModel(0, Buffer.from(WEIGHTS.buffer))).toEqual({
      samples: 2048, live: 768, maxLive: 2, entries: 0, tableBytes: 0, cpuMs: 0, gpuMs: 0, freeBytes: 0, totalBytes: 0, tripleBytes: 0,
    });
    const live = nh.liveSamples({ arch: nh.KEP1, w: WEIGHTS }, DS);
    const few = [];
    for (let n = 0; n < 300; n++) if (nh.liveCount(hdr.attemptKey(HEADER, n), live) <= 2) few.push(n);
    expect(few.length).toBeGreaterThan(2);
    const all = await core.scan(work({ count: 300, maxHits: 300 }));
    expect(nc.hitList(all)).toEqual(few);
    expect(all.evaluated).toBe(few.length);
    expect(all.overflow).toBe(false);
    // A target nothing meets: the same attempts evaluated, no hit.
    const none = await core.scan(work({ count: 300, target: Buffer.alloc(32) }));
    expect(none).toEqual({ hits: Buffer.alloc(0), overflow: false, evaluated: few.length });
    // maxHits cuts the list and sets overflow.
    const cut = await core.scan(work({ count: 300, maxHits: 1 }));
    expect(nc.hitList(cut)).toEqual([few[0]]);
    expect(cut.overflow).toBe(true);
  }, 60000);

  test('buildTriples adds the attempts with 3 live samples among the rows it covers', async () => {
    const core = cpuCore();
    await core.setModel(0, WEIGHTS);
    const live = nh.liveSamples({ arch: nh.KEP1, w: WEIGHTS }, DS);
    const before = nc.hitList(await core.scan(work({ count: 200, maxHits: 200 })));
    expect(await core.buildTriples(0, 5000, 0)).toEqual({ rows: 768, entries: 0, tableBytes: 0, ms: 0 });
    const after = nc.hitList(await core.scan(work({ count: 200, maxHits: 200 })));
    const want = [];
    for (let n = 0; n < 200; n++) if (nh.liveCount(hdr.attemptKey(HEADER, n), live) <= 3) want.push(n);
    expect(after).toEqual(want);
    expect(after.length).toBeGreaterThan(before.length);
    // Half the rows: only the triples whose three rows are all among them.
    expect((await core.buildTriples(0, 384, 0)).rows).toBe(384);
    const some = nc.hitList(await core.scan(work({ count: 200, maxHits: 200 })));
    expect(some.length).toBeGreaterThanOrEqual(before.length);
    expect(some.length).toBeLessThan(after.length);
    // A new model drops the table.
    await core.setModel(0, WEIGHTS);
    expect(nc.hitList(await core.scan(work({ count: 200, maxHits: 200 })))).toEqual(before);
    await expect(new nc.CpuCore().buildTriples(0, 1, 0)).rejects.toThrow('the slot has no dataset');
  }, 120000);

  test('evalKeys gives the reference\'s digest and ticket', async () => {
    const core = cpuCore();
    await core.setModel(0, WEIGHTS);
    const keys = Buffer.concat([Buffer.alloc(32, 1), Buffer.alloc(32, 2), Buffer.alloc(5)]);
    const out = await core.evalKeys(0, keys);
    expect(out.length).toBe(128);
    const r = nh.shareEval({ arch: nh.KEP1, w: WEIGHTS }, DS, Buffer.alloc(32, 2));
    expect(out.subarray(64).equals(Buffer.concat([r.digest, r.ticket]))).toBe(true);
  });

  test('refuses a slot with no dataset or no model, and anything after close', async () => {
    const core = new nc.CpuCore();
    await expect(core.setModel(0, WEIGHTS)).rejects.toThrow('the slot has no dataset');
    core.setDataset(0, DS.arch, DS.values, DS.labels);
    await expect(core.scan(work())).rejects.toThrow('the slot has no model');
    await expect(core.evalKeys(0, Buffer.alloc(32))).rejects.toThrow('the slot has no model');
    await expect(core.scan({})).rejects.toThrow('header must be 80 bytes');
    await core.setModel(0, WEIGHTS);
    const running = core.scan(work({ count: 1000 }));
    core.close();
    await expect(running).rejects.toThrow('the core is closed');
    await expect(core.scan(work())).rejects.toThrow('the core is closed');
    expect(() => core.setDataset(0, DS.arch, DS.values, DS.labels)).toThrow('the core is closed');
  });
});

describe('coverage', () => {
  const live = Uint8Array.from([0, 1, 1, 0, 1, 0, 0, 0]);

  test('numbers the live samples as rows, from 1, in dataset order', () => {
    const c = nc.coverage(live, 2, 0);
    expect(Array.from(c.rowOf)).toEqual([0, 1, 2, 0, 3, 0, 0, 0]);
    expect(c).toMatchObject({ rows: 3, maxLive: 2, tripleRows: 0 });
  });

  test('the share evaluated is the pairs, plus the triples whose rows the table covers', () => {
    const p = 3 / 8;
    expect(nc.coveredShare(nc.coverage(live, 2, 0))).toBeCloseTo(nh.candidateShare(p, 2), 12);
    expect(nc.coveredShare(nc.coverage(live, 1, 0))).toBeCloseTo(nh.candidateShare(p, 1), 12);
    expect(nc.coveredShare(nc.coverage(live, 2, 3))).toBeCloseTo(nh.candidateShare(p, 3), 12);
    expect(nc.coveredShare(nc.coverage(live, 2, 1))).toBeCloseTo(nh.candidateShare(p, 2) + nh.exactShare(p, 3) / 27, 12);
    expect(nc.coveredShare(nc.coverage(new Uint8Array(8), 2, 0))).toBe(1);
  });

  test('evaluates: at most maxLive live samples, or exactly 3 that the triple table covers', () => {
    const all = nc.coverage(nh.liveSamples({ arch: nh.KEP1, w: WEIGHTS }, DS), 2, 0);
    const counts = {};
    for (let n = 0; n < 400; n++) {
      const key = hdr.attemptKey(HEADER, n);
      const rows = nh.shareIndices(key, DS.n).map((i) => all.rowOf[i]).filter((r) => r);
      const kind = rows.length <= 2 ? 'few' : rows.length === 3 ? (Math.max(...rows) <= 384 ? 'low3' : 'high3') : 'many';
      counts[kind] = (counts[kind] || 0) + 1;
      expect(nc.evaluates(all, key)).toBe(kind === 'few');
      expect(nc.evaluates(Object.assign({}, all, { tripleRows: 384 }), key)).toBe(kind === 'few' || kind === 'low3');
      expect(nc.evaluates(Object.assign({}, all, { tripleRows: 768 }), key)).toBe(kind !== 'many');
    }
    expect(Object.keys(counts).sort()).toEqual(['few', 'high3', 'low3', 'many']);
  });

  test('targetFor: a 256-bit little-endian target a given fraction of tickets meet', () => {
    expect(nc.targetFor(1).equals(nc.ALL_ONES)).toBe(true);
    expect(nc.targetFor(7).equals(nc.ALL_ONES)).toBe(true);
    expect(nc.targetFor(0.5).toString('hex')).toBe('ff'.repeat(31) + '7f');
    expect(nc.targetFor(2 ** -32).toString('hex')).toBe('ff'.repeat(28) + '00000000');
  });
});

describe('nebCoreFactory', () => {
  const logs = () => {
    const lines = [];
    return { lines, log: (level, line) => lines.push(level + ' ' + line) };
  };
  const good = { NEB_API: 1, devices: () => [{ index: 0, name: 'Card' }], createCore: jest.fn(({ device }) => ({ device })) };
  const missing = () => { throw Object.assign(new Error('not there'), { code: 'MODULE_NOT_FOUND' }); };

  test('NEB_CORE=cpu is the CPU core', () => {
    const l = logs();
    const f = nc.nebCoreFactory({ env: { NEB_CORE: ' CPU ' }, log: l.log });
    expect(f.label).toBe('cpu');
    expect(f.file).toBeNull();
    expect(f.createCore(3).device.index).toBe(3);
    expect(l.lines).toEqual(['info NEB core: the CPU reference (NEB_CORE=cpu). For tests, not for mining.']);
  });

  test('looks beside the app, then in the dev tree; the first core that sees a card wins', () => {
    const l = logs();
    const tried = [];
    const f = nc.nebCoreFactory({
      env: {}, log: l.log, execPath: path.join('/app', 'earn'), resourcesPath: path.join('/app', 'resources'),
      require: (file) => { tried.push(file); if (file.endsWith(path.join('build', 'neb_cuda.node'))) return good; return missing(); },
    });
    expect(tried.slice(0, 3)).toEqual([
      path.join('/app', 'resources', 'native', 'neb_cuda.node'), path.join('/app', 'neb_cuda.node'), path.join('/app', 'native', 'neb_cuda.node'),
    ]);
    expect(f.label).toBe('neb_cuda');
    expect(f.devices).toEqual([{ index: 0, name: 'Card' }]);
    expect(f.createCore(4)).toEqual({ device: 4 });
    expect(l.lines).toEqual(['info NEB core: ' + f.file + ' · 1 card']);
  });

  test('the rig\'s own maker is tried first, and "cards" is plural', () => {
    const tried = [];
    const two = Object.assign({}, good, { devices: () => [{ index: 0 }, { index: 1 }] });
    const l = logs();
    const f = nc.nebCoreFactory({ env: {}, vendor: 'amd', log: l.log, require: (file) => { tried.push(path.basename(file)); return two; } });
    expect(tried).toEqual(['neb_hip.node']);
    expect(f.label).toBe('neb_hip');
    expect(l.lines[0]).toMatch(/ · 2 cards$/);
    nc.nebCoreFactory({ env: {}, vendor: 'intel', require: (file) => { tried.push(path.basename(file)); return good; } });
    expect(tried[1]).toBe('neb_cuda.node');
  });

  test('NEB_CORE_PATH tries that file alone', () => {
    const tried = [];
    const l = logs();
    expect(nc.nebCoreFactory({ env: { NEB_CORE_PATH: '/x/core.node' }, log: l.log, require: (file) => { tried.push(file); return missing(); } })).toBeNull();
    expect(tried).toEqual(['/x/core.node']);
    expect(l.lines).toEqual(['error NEB core: none found (NEB_CORE_PATH=/x/core.node)']);
  });

  test('says why each file that loaded was refused', () => {
    const l = logs();
    const files = {
      'neb_cuda.node': [() => { throw new Error('libcuda missing'); }, () => ({}), () => ({ NEB_API: 2, devices: () => [], createCore: () => {} })],
      'neb_hip.node': [() => ({ NEB_API: 1, devices: () => { throw new Error('no driver'); }, createCore: () => {} }),
        () => ({ NEB_API: 1, devices: () => [], createCore: () => {} }), () => ({ NEB_API: 1, devices: () => 5, createCore: () => {} }),
        () => null],
    };
    const f = nc.nebCoreFactory({
      env: {}, log: l.log, execPath: path.join('/app', 'earn'), resourcesPath: path.join('/app', 'res'),
      require: (file) => { const q = files[path.basename(file)]; return q.length ? q.shift()() : missing(); },
    });
    expect(f).toBeNull();
    expect(l.lines.map((s) => s.replace(/NEB core: \S+node /, 'NEB core: FILE '))).toEqual([
      'warn NEB core: FILE would not load: libcuda missing', 'warn NEB core: FILE is not a NEB core',
      'warn NEB core: FILE speaks NEB core API 2, this miner 1', 'warn NEB core: FILE could not list its cards: no driver',
      'warn NEB core: FILE found no card', 'warn NEB core: FILE found no card', 'warn NEB core: FILE is not a NEB core',
      'error NEB core: none found (looked for neb_cuda.node, neb_hip.node beside the app and in earn/native/neb/build)',
    ]);
  });

  test('with nothing given it reads the process\'s own environment and says nothing', () => {
    const before = process.env.NEB_CORE;
    process.env.NEB_CORE = 'cpu';
    try {
      expect(nc.nebCoreFactory().label).toBe('cpu');
    } finally {
      if (before === undefined) delete process.env.NEB_CORE; else process.env.NEB_CORE = before;
    }
    expect(nc.nebCoreFactory({ env: {}, require: missing })).toBeNull();
    // And Node's own loader when none is given: a file that is not there is passed over.
    expect(nc.nebCoreFactory({ env: { NEB_CORE_PATH: path.join(__dirname, 'no-such-core.node') } })).toBeNull();
  });
});

describe('verifyCore', () => {
  test('parseVectors reads one architecture\'s shares, by step', () => {
    const steps = nc.parseVectors(VECTORS, nh.KEP1);
    expect(steps.map((s) => s.shares.length)).toEqual([32, 1, 3, 32, 17, 2]);
    expect(steps[0].shares[0].digest).toBe('c904bc03b52554035df36f03cd196903fced9e03111b9803c3a3980380a9a603');
    expect(nc.parseVectors('', nh.KEP1)).toEqual([]);
  });

  test('the CPU core passes every check', async () => {
    let clock = 0;
    const r = await nc.verifyCore(cpuCore(), { ds: DS, weights: WEIGHTS, vectors: VECTORS, exactCount: 64, bigCount: 2048, scanCount: 1500, tripleRows: 768, tripleBytes: 2 ** 30, now: () => (clock += 500) });
    expect(r.checks.map((c) => c.name)).toEqual(['vectors', 'full', 'exact', 'top', 'overflow', 'big', 'triples', 'big3']);
    expect(r.checks.filter((c) => !c.ok)).toEqual([]);
    expect(r.ok).toBe(true);
    expect(r.checks[0].detail).toBe('87 of 87 shares');
    expect(r.info.live).toBe(768);
    expect(r.screenRate).toBe(2048 / 0.5);
    expect(r.hashrate).toBeGreaterThan(0);
    expect(r.triples.rows).toBe(768);
    expect(r.tripleScreenRate).toBe(2048 / 0.5);
    expect(r.tripleHashrate).toBeGreaterThan(r.hashrate);
    expect(r.checks[7].detail).toMatch(/9\.474% expected/);
  }, 240000);

  // A core that answers as the CPU core does, with changes.
  function bent(change) {
    const core = new nc.CpuCore();
    const real = { scan: core.scan.bind(core), evalKeys: core.evalKeys.bind(core), setModel: core.setModel.bind(core) };
    change(core, real);
    return core;
  }
  // The first 256 samples of the check dataset: the same mix of live and dead, and quicker to go through.
  const SMALL = { arch: DS.arch, n: 256, values: DS.values.subarray(0, 256 * 784), labels: DS.labels.subarray(0, 256) };
  const quick = { ds: SMALL, weights: WEIGHTS, exactCount: 128, bigCount: 512, scanCount: 512, tripleRows: 0 };

  test('wrong digests fail the vectors and the full check; no vectors given, no vectors check', async () => {
    const core = bent((c, real) => { c.evalKeys = async (slot, keys) => { const out = await real.evalKeys(slot, keys); out[0] ^= 1; return out; }; });
    const r = await nc.verifyCore(core, Object.assign({}, quick, { vectors: VECTORS.split('\n').slice(0, 4).join('\n') }));
    expect(r.ok).toBe(false);
    expect(r.checks.find((c) => c.name === 'vectors')).toEqual({ name: 'vectors', ok: false, detail: '1 of 2 shares' });
    expect(r.checks.find((c) => c.name === 'full')).toMatchObject({ ok: false, detail: expect.stringMatching(/^63 of 64 keys/) });
    const none = await nc.verifyCore(cpuCore(), Object.assign({}, quick, { vectors: 'neb-nebhash2-vectors 1\n' }));
    expect(none.checks[0]).toEqual({ name: 'vectors', ok: false, detail: '0 of 0 shares' });
    const without = await nc.verifyCore(new nc.CpuCore(), quick);
    expect(without.checks.map((c) => c.name)).toEqual(['full', 'exact', 'top', 'overflow', 'big']);
  }, 240000);

  test('a wrong live count fails the full check', async () => {
    const core = bent((c, real) => { c.setModel = async (slot, w) => Object.assign(await real.setModel(slot, w), { live: 700 }); });
    const r = await nc.verifyCore(core, quick);
    expect(r.checks.find((c) => c.name === 'full').ok).toBe(false);
  }, 240000);

  test('a missed hit, an extra hit, a wrong evaluated count and no overflow each fail', async () => {
    const missed = bent((c, real) => { c.scan = async (w) => { const r = await real.scan(w); return Object.assign(r, { hits: r.hits.subarray(4) }); }; });
    let r = await nc.verifyCore(missed, quick);
    expect(r.checks.filter((c) => !c.ok).map((c) => c.name)).toEqual(expect.arrayContaining(['exact', 'top', 'overflow']));

    const moved = bent((c, real) => {
      c.scan = async (w) => {
        const res = await real.scan(w);
        if (res.hits.length) res.hits.writeUInt32LE((res.hits.readUInt32LE(0) + 1) >>> 0, 0);
        return res;
      };
    });
    r = await nc.verifyCore(moved, Object.assign({}, quick, { bigCount: 4096 }));
    expect(r.checks.find((c) => c.name === 'exact').ok).toBe(false);
    expect(r.checks.find((c) => c.name === 'big')).toMatchObject({ ok: false, detail: expect.stringMatching(/ [1-9]\d* wrong /) });

    // Hits that have few live rows but do not meet the target they were asked for.
    const easy = bent((c, real) => { c.scan = (w) => real.scan(Object.assign({}, w, { target: nc.ALL_ONES })); });
    r = await nc.verifyCore(easy, Object.assign({}, quick, { bigCount: 4096, scanCount: 4096 }));
    expect(r.checks.find((c) => c.name === 'big')).toMatchObject({ ok: false, detail: expect.stringMatching(/ [1-9]\d* wrong /) });

    const miscount = bent((c, real) => { c.scan = async (w) => { const res = await real.scan(w); res.evaluated += 1; return res; }; });
    r = await nc.verifyCore(miscount, quick);
    expect(r.checks.find((c) => c.name === 'exact').ok).toBe(false);
  }, 240000);

  test('a model with every sample dead evaluates every attempt: the big target is still about 32 hits', async () => {
    const ds = { arch: DS.arch, n: DS.n, values: DS.values, labels: Buffer.alloc(DS.n) };
    const core = new nc.CpuCore();
    const r = await nc.verifyCore(core, { ds, weights: WEIGHTS, exactCount: 8, bigCount: 16, scanCount: 16, tripleRows: 0, random: (n) => Buffer.alloc(n, 5) });
    expect(r.info.live).toBe(0);
    expect(r.checks.find((c) => c.name === 'big').detail).toMatch(/100\.000% of attempts evaluated, 100\.000% expected/);
    expect(r.checks.find((c) => c.name === 'overflow').ok).toBe(true);
  }, 240000);
});
