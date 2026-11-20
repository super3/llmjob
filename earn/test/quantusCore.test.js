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

  test('a known card maker tries its own file first: AMD before CUDA, Intel before both; an unknown one keeps the order', () => {
    const order = (vendor) => {
      const tried = [];
      qc.quantusCoreFactory({
        env: {}, vendor, log: () => {}, execPath: '/opt/earn/earn',
        require: (file) => { tried.push(path.basename(file)); throw notFound(); },
      });
      return [...new Set(tried)];
    };
    expect(order('amd')).toEqual(['quantus_hip.node', 'quantus_cuda.node', 'quantus_sycl.node']);
    expect(order('intel')).toEqual(['quantus_sycl.node', 'quantus_cuda.node', 'quantus_hip.node']);
    expect(order('nvidia')).toEqual(qc.FILES);
    expect(order('matrox')).toEqual(qc.FILES);
    expect(qc.VENDOR_FILES).toEqual({ nvidia: 'quantus_cuda.node', amd: 'quantus_hip.node', intel: 'quantus_sycl.node' });
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
      'error Quantus core: none found (looked for quantus_cuda.node, quantus_hip.node, quantus_sycl.node beside the app, '
        + 'quantus_sycl.node in sycl/ there too, and in earn/native/quantus/build)',
    ]);
    // 3 files x (2 beside the executable + the dev tree), and the Intel one in sycl/ in both places too.
    expect(files).toHaveLength(11);
  });

  // A package as CI builds it (miner-build.yml, scripts/build-cli.mjs, build-hiveos.mjs): the CUDA
  // and HIP cores beside the app, the Intel one in the Pearl Intel core's sycl/ folder with the
  // SYCL runtime. The search does not depend on the system it runs on.
  describe('the places a package keeps the cores, on Linux and Windows', () => {
    const fs = require('fs');
    const os = require('os');
    let tmp;
    const realPlatform = process.platform;
    beforeEach(() => { tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'quantus-layout-')); });
    afterEach(() => {
      fs.rmSync(tmp, { recursive: true, force: true });
      Object.defineProperty(process, 'platform', { value: realPlatform });
    });
    const put = (...p) => {
      fs.mkdirSync(path.join(tmp, ...p.slice(0, -1)), { recursive: true });
      fs.writeFileSync(path.join(tmp, ...p), '');
    };
    // A require that loads what is on disk, as a card-less core that only the Intel file has a card for.
    const diskRequire = (tried) => (file) => {
      tried.push(file);
      if (!fs.existsSync(file)) throw notFound();
      const name = path.basename(file);
      return fakeAddon({ devices: () => (name === 'quantus_sycl.node' ? [{ index: 0, name: 'Arc A750' }] : []) });
    };

    for (const platform of ['linux', 'win32']) {
      test(platform + ': sycl/quantus_sycl.node beside the CLI and in resources/native', () => {
        Object.defineProperty(process, 'platform', { value: platform });
        // The CLI binary or HiveOS package: the cores beside the executable.
        put('cli', 'quantus_cuda.node');
        put('cli', 'sycl', 'quantus_sycl.node');
        put('cli', 'sycl', platform === 'win32' ? 'sycl9.dll' : 'libsycl.so.9');
        let tried = [];
        let f = qc.quantusCoreFactory({ env: {}, require: diskRequire(tried), log: () => {}, execPath: path.join(tmp, 'cli', 'earn') });
        expect(f.file).toBe(path.join(tmp, 'cli', 'sycl', 'quantus_sycl.node'));
        expect(f.label).toBe('quantus_sycl');
        // The CUDA core loaded and had no card, so the Intel one was looked for: sycl/ first.
        expect(tried).toContain(path.join(tmp, 'cli', 'quantus_cuda.node'));
        expect(tried.filter((t) => t.endsWith('quantus_sycl.node'))[0]).toBe(f.file);

        // The desktop app: resources/native/sycl, tried before anything beside the executable.
        put('app', 'resources', 'native', 'sycl', 'quantus_sycl.node');
        tried = [];
        f = qc.quantusCoreFactory({
          env: {}, require: diskRequire(tried), log: () => {},
          execPath: path.join(tmp, 'app', 'earn.exe'), resourcesPath: path.join(tmp, 'app', 'resources'),
        });
        expect(f.file).toBe(path.join(tmp, 'app', 'resources', 'native', 'sycl', 'quantus_sycl.node'));
        expect(tried.filter((t) => t.endsWith('quantus_sycl.node'))[0]).toBe(f.file);
      });
    }

    test('a Linux build beside the sycl/ folder still loads from there', () => {
      put('cli', 'quantus_sycl.node');
      const tried = [];
      const f = qc.quantusCoreFactory({ env: {}, require: diskRequire(tried), log: () => {}, execPath: path.join(tmp, 'cli', 'earn') });
      expect(f.file).toBe(path.join(tmp, 'cli', 'quantus_sycl.node'));
      // Its sycl/ place came first and was not there.
      const sycl = tried.filter((t) => t.endsWith('quantus_sycl.node'));
      expect(sycl.slice(0, 2)).toEqual([path.join(tmp, 'cli', 'sycl', 'quantus_sycl.node'), f.file]);
    });

    test('only the Intel core has a sub-folder, and it is the Pearl Intel core\'s', () => {
      const { INTEL_DIR } = require('../src/shared/coreVariant');
      expect(qc.SUBDIRS).toEqual({ 'quantus_sycl.node': [INTEL_DIR, ''] });
      expect(INTEL_DIR).toBe('sycl');
    });
  });

  // RTX 50 rigs on driver 580+ try quantus_cuda_cu13.node first, by the Pearl core's rule
  // (shared/coreVariant.pickCoreVariant); everything else keeps quantus_cuda.node.
  describe('the CUDA 13 build', () => {
    const rtx50 = [{ index: 0, major: 12, minor: 0, driverMajor: 610 }];
    const cu13 = path.join('/opt/earn', 'quantus_cuda_cu13.node');
    const cu12 = path.join('/opt/earn', 'quantus_cuda.node');
    // Loads the files `have` names (basename -> addon) beside the executable, and none other.
    const fromDisk = (have, tried = []) => (file) => {
      tried.push(file);
      if (path.dirname(file) === path.dirname(cu12) && have[path.basename(file)]) return have[path.basename(file)];
      throw notFound();
    };
    const factory = (over) => qc.quantusCoreFactory({ env: {}, execPath: '/opt/earn/earn', log: () => {}, ...over });

    test('an RTX 50 on driver 580 or newer: the CUDA 13 build, tried before every other file', () => {
      const tried = [];
      const log = jest.fn();
      const f = factory({ cards: rtx50, gpus: [{ index: 0 }], log, require: fromDisk({ 'quantus_cuda_cu13.node': fakeAddon(), 'quantus_cuda.node': fakeAddon() }, tried) });
      expect(f).toMatchObject({ file: cu13, label: 'quantus_cuda_cu13' });
      expect(path.basename(tried[0])).toBe('quantus_cuda_cu13.node');
      expect(log).toHaveBeenCalledWith('info', 'Quantus core: ' + cu13 + ' · 2 cards · CUDA 13 build (driver 610, mining card is compute 12.0)');
    });

    test('an RTX 40, or an RTX 50 on a driver below 580: the CUDA 12.8 build, and the CUDA 13 one is never tried', () => {
      for (const [cards, reason] of [
        [[{ index: 0, major: 8, minor: 9, driverMajor: 610 }], 'GPU 0 is compute 8.9 (the CUDA 13 build has code for it but measured no faster on a 4090)'],
        [[{ index: 0, major: 12, minor: 0, driverMajor: 575 }], 'driver 575 (the CUDA 13 build needs 580+)'],
      ]) {
        const tried = [];
        const log = jest.fn();
        const f = factory({ cards, log, require: fromDisk({ 'quantus_cuda_cu13.node': fakeAddon(), 'quantus_cuda.node': fakeAddon() }, tried) });
        expect(f.file).toBe(cu12);
        expect(tried.some((t) => t.endsWith('quantus_cuda_cu13.node'))).toBe(false);
        expect(log).toHaveBeenCalledWith('info', 'Quantus core: ' + cu12 + ' · 2 cards · CUDA 12.8 build (' + reason + ')');
      }
    });

    test('wanted but not installed, or seeing no card (a runtime the driver cannot run): the CUDA 12.8 build, with a warning', () => {
      let log = jest.fn();
      let f = factory({ cards: rtx50, log, require: fromDisk({ 'quantus_cuda.node': fakeAddon() }) });
      expect(f.file).toBe(cu12);
      expect(log).toHaveBeenCalledWith('warn', 'Quantus core: ' + cu12 + ' · 2 cards · CUDA 12.8 build: the CUDA 13 one was wanted '
        + '(driver 610, mining card is compute 12.0), but quantus_cuda_cu13.node is not installed');
      log = jest.fn();
      f = factory({ cards: rtx50, log, require: fromDisk({ 'quantus_cuda_cu13.node': fakeAddon({ devices: () => [] }), 'quantus_cuda.node': fakeAddon() }) });
      expect(f.file).toBe(cu12);
      expect(log).toHaveBeenCalledWith('warn', expect.stringContaining('but ' + cu13 + ' found no card'));
    });

    test('QUANTUS_CORE_VARIANT=cu13 or cu12 overrides the rule; an AMD or Intel rig never tries a CUDA build first', () => {
      const have = { 'quantus_cuda_cu13.node': fakeAddon(), 'quantus_cuda.node': fakeAddon() };
      expect(factory({ env: { QUANTUS_CORE_VARIANT: ' CU13 ' }, require: fromDisk(have) }).file).toBe(cu13);
      const log = jest.fn();
      expect(factory({ env: { QUANTUS_CORE_VARIANT: 'cu12' }, cards: rtx50, log, require: fromDisk(have) }).file).toBe(cu12);
      expect(log).toHaveBeenCalledWith('info', 'Quantus core: ' + cu12 + ' · 2 cards · CUDA 12.8 build (QUANTUS_CORE_VARIANT=cu12)');
      // No list and no override (a caller that does not pass nvidia-smi's): the old order.
      expect(factory({ cards: [], require: fromDisk(have) }).file).toBe(cu12);
      const tried = [];
      const hip = fakeAddon({ devices: () => [{ index: 0, name: 'RX 7900 XTX' }] });
      factory({ vendor: 'amd', cards: rtx50, require: fromDisk({ ...have, 'quantus_hip.node': hip }, tried) });
      expect(tried.map((t) => path.basename(t))).toEqual(['quantus_hip.node']);
    });

    test('a file other than the CUDA ones, or QUANTUS_CORE_PATH, logs as before', () => {
      const log = jest.fn();
      const hip = fakeAddon({ devices: () => [{ index: 0, name: 'RX 7900 XTX' }] });
      expect(factory({ cards: rtx50, log, require: fromDisk({ 'quantus_hip.node': hip }) }).label).toBe('quantus_hip');
      expect(log).toHaveBeenCalledWith('info', 'Quantus core: ' + path.join('/opt/earn', 'quantus_hip.node') + ' · 1 card');
      const named = jest.fn();
      factory({ env: { QUANTUS_CORE_PATH: '/x/quantus_cuda_cu13.node' }, cards: rtx50, log: named, require: () => fakeAddon() });
      expect(named).toHaveBeenCalledWith('info', 'Quantus core: /x/quantus_cuda_cu13.node · 2 cards');
    });

    test('a CUDA 13 core that cannot start on this driver: the CUDA 12.8 build takes over, for that card and every later one', () => {
      const bad = fakeAddon({ createCore: jest.fn(() => { throw new Error('cudaSetDevice: CUDA driver version is insufficient for CUDA runtime version'); }) });
      const good = fakeAddon();
      const log = jest.fn();
      const f = factory({ cards: rtx50, log, require: fromDisk({ 'quantus_cuda_cu13.node': bad, 'quantus_cuda.node': good }) });
      expect(f.createCore(1)).toEqual({ device: { index: 1, name: 'core' } });
      expect(good.createCore).toHaveBeenCalledWith({ device: 1 });
      expect(f).toMatchObject({ file: cu12, label: 'quantus_cuda' });
      expect(log).toHaveBeenCalledWith('warn', 'Quantus core: the CUDA 13 build could not start (cudaSetDevice: CUDA driver version is '
        + 'insufficient for CUDA runtime version); ' + cu12 + ' (CUDA 12.8) takes over');
      f.createCore(0);
      expect(bad.createCore).toHaveBeenCalledTimes(1);
      expect(good.createCore).toHaveBeenCalledTimes(2);
    });

    test('a CUDA 13 core that started keeps the card errors after it; other errors, or no 12.8 build, are thrown', () => {
      const out = new Error('cudaMalloc: out of memory');
      let n = 0;
      const ok = fakeAddon({ createCore: jest.fn((o) => { if (n++) throw new Error('no CUDA device found'); return { device: o }; }) });
      const f = factory({ cards: rtx50, require: fromDisk({ 'quantus_cuda_cu13.node': ok, 'quantus_cuda.node': fakeAddon() }) });
      f.createCore(0);
      expect(() => f.createCore(1)).toThrow('no CUDA device found');
      expect(f.file).toBe(cu13);
      const oom = fakeAddon({ createCore: () => { throw out; } });
      expect(() => factory({ cards: rtx50, require: fromDisk({ 'quantus_cuda_cu13.node': oom, 'quantus_cuda.node': fakeAddon() }) }).createCore(0)).toThrow(out);
      const old = fakeAddon({ createCore: () => { throw new Error('no CUDA device found'); } });
      expect(() => factory({ cards: rtx50, require: fromDisk({ 'quantus_cuda_cu13.node': old }) }).createCore(0)).toThrow('no CUDA device found');
    });
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
