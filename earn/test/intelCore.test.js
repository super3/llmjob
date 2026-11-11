'use strict';

// The Intel (SYCL) core in the app: when it is picked, where it is looked for,
// what is said when it is missing or cannot start, and how a rig's Arc cards
// are found and counted. The rule that matters most is the one for every other
// rig: nothing here may change what an NVIDIA or an AMD rig loads or logs.

jest.mock('child_process');

const path = require('path');
const { execFile } = require('child_process');
const {
  CU12, CU13, AMD, INTEL, INTEL_DIR, FILES, LABELS, pickCoreVariant,
} = require('../src/shared/coreVariant');
const gpu = require('../src/shared/gpu');
const { parseCliArgs, USAGE } = require('../src/shared/cliArgs');
const { coreFactory } = require('../src/main/pearlCore');
const probe = require('../src/main/probe');

const withPlatform = (value, fn) => {
  const original = Object.getOwnPropertyDescriptor(process, 'platform');
  Object.defineProperty(process, 'platform', { value, configurable: true });
  return Promise.resolve(fn()).finally(() => Object.defineProperty(process, 'platform', original));
};

describe('which rig gets the Intel core (shared/coreVariant)', () => {
  test('it is a file of its own, in a folder of its own', () => {
    expect([INTEL, FILES[INTEL], LABELS[INTEL], INTEL_DIR]).toEqual(['intel', 'pearl_core_sycl.node', 'Intel build (SYCL)', 'sycl']);
  });

  test('a rig with no NVIDIA card whose system lists Arc cards tries it first', () => {
    expect(pickCoreVariant({ env: {}, cards: [], intel: 6 })).toEqual({
      variant: INTEL, auto: true, reason: 'nvidia-smi lists no NVIDIA card, and the system has 6 Intel Arc cards',
    });
    expect(pickCoreVariant({ env: {}, cards: [], intel: 1 }).reason)
      .toBe('nvidia-smi lists no NVIDIA card, and the system has 1 Intel Arc card');
  });

  test('every other rig is picked for exactly as before', () => {
    const amdFirst = { variant: AMD, auto: true, reason: 'nvidia-smi lists no NVIDIA card' };
    // No Arc card listed, or nobody counted: the AMD core first, as since v0.6.8.
    for (const intel of [0, undefined, null, NaN, '6', -1, 1.5]) {
      expect([intel, pickCoreVariant({ env: {}, cards: [], intel })]).toEqual([intel, amdFirst]);
    }
    // nvidia-smi listed cards: an Arc card beside them changes nothing.
    const blackwell = [{ index: 0, major: 12, minor: 0, driverMajor: 610 }];
    const ada = [{ index: 0, major: 8, minor: 9, driverMajor: 570 }];
    for (const cards of [blackwell, ada]) {
      expect(pickCoreVariant({ env: {}, cards, intel: 2 })).toEqual(pickCoreVariant({ env: {}, cards }));
    }
    expect(pickCoreVariant({ env: {}, cards: blackwell, intel: 2 }).variant).toBe(CU13);
    // And a caller that did not ask nvidia-smi keeps the 12.8 build.
    expect(pickCoreVariant({ env: {}, intel: 6 }).variant).toBe(CU12);
  });

  test('PEARL_CORE_VARIANT=intel asks for it by name, and another value still wins over the count', () => {
    expect(pickCoreVariant({ env: { PEARL_CORE_VARIANT: ' Intel ' }, cards: [{ index: 0, major: 12, minor: 0, driverMajor: 610 }] }))
      .toEqual({ variant: INTEL, reason: 'PEARL_CORE_VARIANT=intel' });
    expect(pickCoreVariant({ env: { PEARL_CORE_VARIANT: 'cu12' }, cards: [], intel: 6 }).variant).toBe(CU12);
    expect(pickCoreVariant({ env: { PEARL_CORE_VARIANT: 'amd' }, cards: [], intel: 6 }).variant).toBe(AMD);
    expect(pickCoreVariant({ env: { PEARL_CORE_VARIANT: 'arc' }, cards: [], intel: 2 }).reason)
      .toBe('PEARL_CORE_VARIANT=arc ignored (use cu12, cu13, amd or intel); nvidia-smi lists no NVIDIA card, and the system has 2 Intel Arc cards');
  });

  test('--core takes intel', () => {
    const r = parseCliArgs(['-a', 'prl1pql8r6m4z9x7v2k0t3whu8e2snd4p6c', '--core', 'INTEL']);
    expect([r.errors, r.settings.core]).toEqual([[], 'intel']);
    expect(USAGE).toContain('intel for Intel Arc cards');
  });
});

describe('Intel Arc cards by name (shared/gpu)', () => {
  test('a discrete Arc card is told from a processor\'s own graphics by its model', () => {
    for (const name of ['Intel(R) Arc(TM) A750 Graphics', 'Intel(R) Arc(TM) B580 Graphics', 'Intel(R) Arc(TM) A770M Graphics',
      'Intel(R) Arc(TM) Pro B60 Graphics', 'Intel(R) Arc(TM) Pro A60 Graphics', 'Intel Arc A380']) {
      expect([name, gpu.isDiscreteIntel(name)]).toEqual([name, true]);
    }
    for (const name of ['Intel(R) Arc(TM) Graphics', 'Intel(R) Arc(TM) 140V GPU', 'Intel(R) UHD Graphics 630',
      'Intel(R) Iris(R) Xe Graphics', 'AMD Radeon RX 9070 XT', 'NVIDIA GeForce RTX 4090', 'Parsec Virtual Arc A750', '', null, undefined]) {
      expect([name, gpu.isDiscreteIntel(name)]).toEqual([name, false]);
    }
  });

  // Every Intel name used to count as integrated, so a rig of six Arc cards
  // was one card on the board.
  test('Arc cards count as the rig\'s cards, and one is picked over the processor\'s graphics', () => {
    const arc = 'Intel(R) Arc(TM) A750 Graphics';
    expect(gpu.countGpus(['Intel(R) UHD Graphics 630', arc, arc, arc])).toBe(3);
    expect(gpu.pickGpu(['Intel(R) UHD Graphics 630', arc])).toBe(arc);
    // And nothing changes for the names it already knew.
    expect(gpu.countGpus(['Intel(R) UHD Graphics 630', 'NVIDIA GeForce RTX 3070'])).toBe(1);
    expect(gpu.countGpus(['Intel(R) UHD Graphics 630'])).toBe(1);
    expect(gpu.pickGpu(['Intel(R) Arc(TM) Graphics'])).toBe('Intel(R) Arc(TM) Graphics');
  });

  test('a card is named from its PCI id, or from pci.ids when that calls it an Arc model', () => {
    expect(Object.keys(gpu.INTEL_DEVICES).sort()).toEqual(['56a0', '56a1', '56a2', '56a5', '56a6', 'e20b', 'e20c']);
    expect(gpu.intelCardName('56a1', { deviceName: null })).toBe('Intel Arc A750');
    expect(gpu.intelCardName('e20b', {})).toBe('Intel Arc B580');
    expect(gpu.intelCardName('5690', { deviceName: 'DG2 [Arc A770M]' })).toBe('Intel Arc A770M');
    expect(gpu.intelCardName('e211', { deviceName: 'Battlemage G21 [Intel Arc(TM)  Pro B60 Graphics]' })).toBe('Intel Arc Pro B60');
    expect(gpu.intelCardName('56c0', { deviceName: 'Arc A999X something' })).toBe('Intel Arc A999X');
    // A processor's graphics, and anything pci.ids does not call an Arc card.
    for (const deviceName of ['Meteor Lake-P [Intel Arc Graphics]', 'Lunar Lake [Intel Arc Graphics 140V]', 'UHD Graphics 630', '', null]) {
      expect([deviceName, gpu.intelCardName('7d55', { deviceName })]).toEqual([deviceName, null]);
    }
  });

  test('a rig of Arc cards mines on every one of them, one core each', () => {
    const det = (n) => ({ name: 'Intel Arc A750', count: n, amd: [], intel: Array.from({ length: n }, (_, index) => ({ index })) });
    expect([gpu.intelCount(det(6)), gpu.intelCount(null), gpu.intelCount({ name: 'x', count: 1 }), gpu.intelCount({ intel: 'six' })])
      .toEqual([6, 0, 0, 0]);
    const six = gpu.planIntelGpus({ env: {}, det: det(6), chosen: null });
    expect(six).toEqual([0, 1, 2, 3, 4, 5].map((index) => ({ index, name: null })));
    expect(gpu.describeIntelPlan(six, 'use --gpu-index'))
      .toBe('mining on 6 Intel cards (GPU 0, 1, 2, 3, 4, 5), one core each; use --gpu-index to choose cards');
    // One card, cards the operator chose, nothing listed, or another core asked for: the old path.
    expect(gpu.planIntelGpus({ env: {}, det: det(1) })).toEqual([]);
    expect(gpu.planIntelGpus({ env: {}, det: det(6), chosen: [0, 2] })).toEqual([]);
    expect(gpu.planIntelGpus({ env: {}, det: null })).toEqual([]);
    expect(gpu.planIntelGpus()).toEqual([]);
    for (const v of ['cu12', 'cu13', 'amd']) {
      expect([v, gpu.planIntelGpus({ env: { PEARL_CORE_VARIANT: v }, det: det(6) })]).toEqual([v, []]);
    }
    expect(gpu.planIntelGpus({ env: { PEARL_CORE_VARIANT: 'intel' }, det: det(2) })).toHaveLength(2);
    // An AMD rig's answer has no `intel`, so its plan is the AMD one alone.
    const amdRig = { name: 'AMD Radeon RX 9070 XT', count: 2, amd: [{}, {}] };
    expect([gpu.planAmdGpus({ env: {}, det: amdRig }).length, gpu.planIntelGpus({ env: {}, det: amdRig }).length]).toEqual([2, 0]);
    expect(gpu.planAmdGpus({ env: {}, det: det(6) })).toEqual([]);
  });
});

describe('Intel Arc cards from the system (main/probe)', () => {
  beforeEach(() => jest.clearAllMocks());
  const execCb = (err, stdout) => execFile.mockImplementation((_bin, _args, _opts, cb) => cb(err, stdout));
  const fakeFs = (files, real, entries) => ({
    readdirSync: jest.fn(() => {
      if (!entries) throw new Error('ENOENT');
      return entries;
    }),
    readFileSync: jest.fn((p) => {
      if (!(p in files)) throw new Error('ENOENT ' + p);
      return files[p];
    }),
    realpathSync: jest.fn((p) => {
      if (!(p in real)) throw new Error('ENOENT ' + p);
      return real[p];
    }),
  });
  const IDS = '1002  AMD\n\t7550  Navi 48 [Radeon RX 9070/9070 XT/9070 GRE]\n8086  Intel Corporation\n\t5690  DG2 [Arc A770M]\n'
    + '\t9bc8  CometLake-S GT2 [UHD Graphics 630]\n';
  const card = (n, vendor, device) => ({
    ['/d/card' + n + '/device/vendor']: '0x' + vendor + '\n',
    ['/d/card' + n + '/device/device']: '0x' + device + '\n',
  });
  // The rig: its processor's graphics, two A750s, an A770M named by pci.ids only.
  const files = { '/ids': IDS, ...card(0, '8086', '9bc8'), ...card(1, '8086', '56a1'), ...card(2, '8086', '56a1'), ...card(3, '8086', '5690') };
  const real = {
    '/d/card1/device': '/sys/devices/pci0000:00/0000:04:00.0', '/d/card2/device': '/sys/devices/pci0000:00/0000:03:00.0',
    '/d/card3/device': '/sys/devices/pci0000:00/0000:0b:00.0',
  };
  const entries = ['card0', 'card1', 'card1-DP-1', 'card2', 'card3', 'renderD128'];
  const opts = { fs: fakeFs(files, real, entries), root: '/d', pciIds: ['/ids'] };
  const CARDS = [
    { index: 0, name: 'Intel Arc A750', usedMb: 0, totalMb: 0 },
    { index: 1, name: 'Intel Arc A750', usedMb: 0, totalMb: 0 },
    { index: 2, name: 'Intel Arc A770M', usedMb: 0, totalMb: 0 },
  ];
  // The same rig with an AMD card in it.
  const mixed = { fs: fakeFs({ ...files, ...card(4, '1002', '7550') }, { ...real, '/d/card4/device': '/sys/devices/pci0000:00/0000:01:00.0' },
    entries.concat(['card4'])), root: '/d', pciIds: ['/ids'] };

  test('each Arc card is named, in PCI bus order, with no memory figure, and the processor\'s graphics is left out', () => {
    expect(probe.detectIntelGpus(opts)).toEqual(CARDS);
    expect(probe.detectIntelBusIds(opts)).toEqual({ 0: '0000:03:00.0', 1: '0000:04:00.0', 2: '0000:0b:00.0' });
    expect(probe.detectIntelGpus({ fs: fakeFs({}, {}, null) })).toEqual([]);
    expect(Array.isArray(probe.detectIntelGpus())).toBe(true);
    expect(probe.detectIntelBusIds()).toEqual(expect.any(Object));
    // The AMD cards of the same folder are still the AMD cards alone.
    expect(probe.detectAmdGpus(opts)).toEqual([]);
    expect(probe.detectAmdGpus(mixed).map((c) => c.name)).toEqual(['AMD Radeon RX 9070/9070 XT/9070 GRE']);
  });

  test('Linux with no nvidia-smi and no AMD card: the rig is named from its Arc cards, and lists them', async () => {
    await withPlatform('linux', async () => {
      execCb(new Error('ENOENT'));
      expect(await probe.detectGpuInfo(opts)).toEqual({ name: 'Intel Arc A750', count: 3, amd: [], intel: CARDS });
      // With an AMD card the answer is the AMD rig's, as before: no `intel` at all.
      const amd = await probe.detectGpuInfo(mixed);
      expect(amd).toEqual({ name: 'AMD Radeon RX 9070/9070 XT/9070 GRE', count: 1, amd: probe.detectAmdGpus(mixed) });
      expect('intel' in amd).toBe(false);
      expect(await probe.detectBoardGpus(opts)).toEqual(CARDS);
      expect((await probe.detectBoardGpus(mixed)).map((c) => c.name)).toEqual(['AMD Radeon RX 9070/9070 XT/9070 GRE']);
      expect(await probe.detectPciBusIds(opts)).toEqual({ 0: '0000:03:00.0', 1: '0000:04:00.0', 2: '0000:0b:00.0' });
      expect(await probe.detectPciBusIds(mixed)).toEqual({ 0: '0000:01:00.0' });
    });
    // nvidia-smi's answer still comes first, on any system.
    execCb(null, 'NVIDIA GeForce RTX 4090\n');
    await withPlatform('linux', async () => {
      expect(await probe.detectGpuInfo(opts)).toEqual({ name: 'NVIDIA GeForce RTX 4090', count: 1 });
    });
  });

  test('Windows: the Arc cards WMI lists, unless it lists an AMD card too', async () => {
    const wmi = (out) => execFile.mockImplementation((bin, _args, _opts, cb) => (bin === 'nvidia-smi' ? cb(new Error('none')) : cb(null, out)));
    await withPlatform('win32', async () => {
      const A750 = { name: 'Intel(R) Arc(TM) A750 Graphics', driverVersion: '32.0.101.6913' };
      wmi('Intel(R) UHD Graphics 630|31.0.101.2130\r\nIntel(R) Arc(TM) A750 Graphics|32.0.101.6913\r\nIntel(R) Arc(TM) A750 Graphics|32.0.101.6913\r\n');
      expect(await probe.detectGpuInfo()).toEqual({ name: A750.name, count: 2, amd: [], intel: [A750, A750] });
      wmi('AMD Radeon RX 9070 XT|32.0.1\r\nIntel(R) Arc(TM) A750 Graphics|32.0.101.6913\r\n');
      const amd = await probe.detectGpuInfo();
      expect(amd).toEqual({ name: 'AMD Radeon RX 9070 XT', count: 2, amd: [{ name: 'AMD Radeon RX 9070 XT', driverVersion: '32.0.1' }] });
      // Only a processor's graphics: named, and no card of either kind to plan from.
      wmi('Intel(R) UHD Graphics 630|31.0.101.2130\r\n');
      expect(await probe.detectGpuInfo()).toEqual({ name: 'Intel(R) UHD Graphics 630', count: 1, amd: [] });
    });
  });
});

// Where the loader looks, what it says, and the fallbacks. As in
// pearlCore.test.js: a rig is a set of files a fake require knows, each addon a
// spy, and the log is read back line by line.
describe('loading the Intel core (main/pearlCore)', () => {
  const EXEC = '/opt/rig/llmjob-earn-cli-linux';
  const CU12_BESIDE = path.join('/opt/rig', 'pearl_core.node');
  const AMD_BESIDE = path.join('/opt/rig', 'pearl_core_hip.node');
  const INTEL_BESIDE = path.join('/opt/rig', 'sycl', 'pearl_core_sycl.node');
  const INTEL_DEV = path.join(__dirname, '..', 'src', 'main', '..', '..', 'native', 'sycl', 'build', 'pearl_core_sycl.node');
  const NO_GPU = 'no Intel GPU found through Level Zero or OpenCL';
  const ARC_RIG = { cards: [], gpus: [], intel: 6 };
  const REASON = 'nvidia-smi lists no NVIDIA card, and the system has 6 Intel Arc cards';

  const addon = (build, fail) => ({ createCore: jest.fn(() => { if (fail) throw fail(); return { build }; }) });
  function rig(files) {
    const map = new Map(Object.entries(files));
    const req = jest.fn((p) => {
      if (!map.has(p)) throw Object.assign(new Error('Cannot find module ' + p), { code: 'MODULE_NOT_FOUND' });
      const v = map.get(p);
      if (v instanceof Error) throw v;
      return v;
    });
    const lines = [];
    return { req, lines, log: (level, line) => lines.push([level, line]) };
  }
  // Linux unless a test says Windows: the lines must not depend on where the tests run.
  const factory = (r, extra) => coreFactory(Object.assign({ require: r.req, execPath: EXEC, env: {}, log: r.log, platform: 'linux' }, extra));

  test('an Arc rig mines on it, from its folder beside the other cores, and loads no other core', () => {
    const intel = addon('intel');
    const cu12 = addon('cu12');
    const r = rig({ [INTEL_BESIDE]: intel, [CU12_BESIDE]: cu12 });
    const f = factory(r, ARC_RIG);
    expect(f({ rank: 128 }, { deviceIndex: 3, checkHits: true })).toEqual({ build: 'intel' });
    expect(f({ rank: 128 })).toEqual({ build: 'intel' });
    expect(intel.createCore).toHaveBeenCalledWith({ rank: 128 }, { deviceIndex: 3, checkHits: true });
    expect(r.req).not.toHaveBeenCalledWith(CU12_BESIDE);
    expect(r.req).not.toHaveBeenCalledWith(AMD_BESIDE);
    expect(r.lines).toEqual([['info', 'Pearl core: Intel build (SYCL) · ' + REASON]]);
    // The Intel core has no clock offset calls: those are NVIDIA's.
    expect(f.clockOffsets.readCoreOffset(0)).toMatchObject({ ok: false });
  });

  test('the desktop app finds it in resources/native/sycl, and a source tree in its build folder', () => {
    const packed = rig({ [path.join('/app/resources', 'native', 'sycl', 'pearl_core_sycl.node')]: addon('packed') });
    expect(factory(packed, { ...ARC_RIG, resourcesPath: '/app/resources' })()).toEqual({ build: 'packed' });
    const dev = rig({ [INTEL_DEV]: addon('dev') });
    expect(factory(dev, ARC_RIG)()).toEqual({ build: 'dev' });
    expect(dev.lines).toEqual([['info', 'Pearl core: Intel build (SYCL) · local build ' + INTEL_DEV + ', ' + REASON]]);
  });

  test('on Windows it says the core is experimental', () => {
    const r = rig({ [INTEL_BESIDE]: addon('intel') });
    factory(r, { ...ARC_RIG, platform: 'win32' });
    expect(r.lines).toEqual([['info', 'Pearl core: Intel build (SYCL) · ' + REASON
      + '; experimental on Windows, where no Intel card has run it yet']]);
    // Asked for by name it says so too.
    const named = rig({ [INTEL_BESIDE]: addon('intel') });
    factory(named, { env: { PEARL_CORE_VARIANT: 'intel' }, platform: 'win32', coreFlag: true });
    expect(named.lines).toEqual([['info', 'Pearl core: Intel build (SYCL) · --core intel'
      + '; experimental on Windows, where no Intel card has run it yet']]);
  });

  // The rule for every other rig. Nothing looks for the Intel core unless the
  // system listed Arc cards and no NVIDIA or AMD one, so a release without it,
  // or one whose folder is broken, reads exactly as before.
  describe('an NVIDIA or an AMD rig never hears of it', () => {
    const broken = new Error('libsycl.so.9: cannot open shared object file: No such file or directory');
    const NVIDIA = { cards: [{ index: 0, major: 8, minor: 9, driverMajor: 580 }], gpus: [{ index: 0 }] };

    test.each([['missing', {}], ['there but broken', { [INTEL_BESIDE]: broken }], ['there and fine', { [INTEL_BESIDE]: addon('intel') }]])(
      'NVIDIA rig, Intel core %s', (_what, extra) => {
        const r = rig({ [CU12_BESIDE]: addon('cu12'), ...extra });
        // Even with an Arc card in the box beside the NVIDIA ones.
        for (const intel of [undefined, 0, 2]) {
          r.lines.length = 0;
          expect(factory(r, { ...NVIDIA, intel })()).toEqual({ build: 'cu12' });
          expect(r.lines).toEqual([['info', 'Pearl core: CUDA 12.8 build · GPU 0 is compute 8.9 (the CUDA 13 build has code for it but measured no faster on a 4090)']]);
        }
        expect(r.req.mock.calls.filter((c) => /sycl/.test(c[0]))).toEqual([]);
      });

    test.each([['missing', {}], ['there but broken', { [INTEL_BESIDE]: broken }]])(
      'AMD rig, Intel core %s', (_what, extra) => {
        const r = rig({ [AMD_BESIDE]: addon('amd'), [CU12_BESIDE]: addon('cu12'), ...extra });
        // probe.detectGpuInfo gives an AMD rig no `intel`, whatever else is in it.
        expect(factory(r, { cards: [], gpus: [] })()).toEqual({ build: 'amd' });
        expect(r.lines).toEqual([['info', 'Pearl core: AMD build (ROCm) · nvidia-smi lists no NVIDIA card']]);
        expect(r.req.mock.calls.filter((c) => /sycl/.test(c[0]))).toEqual([]);
      });
  });

  describe('an Arc rig whose Intel core cannot be used', () => {
    test('not installed: the 12.8 build, and the line says why', () => {
      const r = rig({ [CU12_BESIDE]: addon('cu12') });
      expect(factory(r, ARC_RIG)()).toEqual({ build: 'cu12' });
      expect(r.lines).toEqual([['info', 'Pearl core: CUDA 12.8 build · ' + REASON + ', and pearl_core_sycl.node is not installed']]);
    });

    test('there but it will not load: a warning with the reason, and that reason again when no card is found', () => {
      const gone = new Error('libsycl.so.9: cannot open shared object file: No such file or directory');
      const cu12 = addon('cu12', () => new Error('no CUDA device found (is an NVIDIA driver installed?)'));
      const r = rig({ [INTEL_BESIDE]: gone, [CU12_BESIDE]: cu12 });
      const f = factory(r, ARC_RIG);
      const why = INTEL_BESIDE + ' is there but would not load: ' + gone.message + '. Its folder must hold every file the release put '
        + 'there (the SYCL runtime sits beside the core): install the release again';
      expect(r.lines).toEqual([['warn', 'Pearl core: CUDA 12.8 build · ' + REASON + ', and ' + why]]);
      expect(() => f({})).toThrow(new Error('no CUDA device found, and the Intel core would not load: ' + why));
    });

    test('the loader\'s words get a hint where they do not say what to do', () => {
      const said = (error, platform) => {
        const r = rig({ [INTEL_BESIDE]: error });
        expect(factory(r, { env: { PEARL_CORE_VARIANT: 'intel' }, platform })).toBeNull();
        return r.lines[0][1].replace('Pearl core: Intel build (SYCL) · ' + INTEL_BESIDE + ' is there but would not load: ', '');
      };
      expect(said(new Error("/lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.27' not found (required by libsycl.so.9)")))
        .toMatch(/not found \(required by libsycl\.so\.9\)\. It needs glibc 2\.27 and GCC 7's libstdc\+\+ or newer \(Ubuntu 18\.04 and later have both\)$/);
      expect(said(new Error("libstdc++.so.6: version `GLIBCXX_3.4.22' not found"))).toMatch(/It needs glibc 2\.27/);
      expect(said(new Error('libumf.so.1: cannot open shared object file'))).toMatch(/Its folder must hold every file the release put there \(the SYCL runtime sits beside the core\): install the release again$/);
      expect(said(new Error('The specified module could not be found.\r\n\\\\?\\C:\\app\\sycl\\pearl_core_sycl.node'), 'win32'))
        .toBe('The specified module could not be found. \\\\?\\C:\\app\\sycl\\pearl_core_sycl.node. Its folder must hold every file the '
          + 'release put there (the SYCL runtime sits beside the core), and Windows needs Microsoft\'s Visual C++ runtime installed: '
          + 'install the release again');
      // Anything else is the loader's own words alone.
      expect(said(new Error('invalid ELF header'))).toBe('invalid ELF header');
    });

    test('it cannot start on its first card for a reason of its own: the 12.8 build takes over, with a warning', () => {
      const intel = addon('intel', () => new Error('not enough device memory for this profile: need ~900 MiB, the device has 512 MiB'));
      const cu12 = addon('cu12');
      const r = rig({ [INTEL_BESIDE]: intel, [CU12_BESIDE]: cu12 });
      const f = factory(r, ARC_RIG);
      expect(f({}, { deviceIndex: 0 })).toEqual({ build: 'cu12' });
      expect(r.lines[1]).toEqual(['warn', 'Pearl core: CUDA 12.8 build · the Intel build could not start (not enough device memory '
        + 'for this profile: need ~900 MiB, the device has 512 MiB)']);
      // Settled: the next card goes straight to the 12.8 build.
      expect(f({}, { deviceIndex: 1 })).toEqual({ build: 'cu12' });
      expect(intel.createCore).toHaveBeenCalledTimes(1);
    });

    // The system lists the cards and the core finds none: the cards are there
    // and Intel's compute runtime is not. Said with what to install, once.
    test('the driver is missing: its error says what to install, the first time only', () => {
      const intel = addon('intel', () => new Error(NO_GPU));
      const cu12 = addon('cu12', () => new Error('no CUDA device found (is an NVIDIA driver installed?)'));
      const r = rig({ [INTEL_BESIDE]: intel, [CU12_BESIDE]: cu12 });
      const f = factory(r, ARC_RIG);
      expect(() => f({})).toThrow(new Error(NO_GPU + ': the system lists an Arc card, so Intel\'s compute runtime is missing or this '
        + 'user may not open the card. Install intel-opencl-icd (OpenCL) or libze-intel-gpu1 and libze1 (Level Zero), and add the '
        + 'user to the render group'));
      // Each restart tries both again, and says only the short reason.
      expect(() => f({})).toThrow(new Error(NO_GPU));
      expect(() => f({})).toThrow(new Error(NO_GPU));
      expect(intel.createCore).toHaveBeenCalledTimes(3);
      const w = rig({ [INTEL_BESIDE]: addon('intel', () => new Error(NO_GPU)), [CU12_BESIDE]: cu12 });
      expect(() => factory(w, { ...ARC_RIG, platform: 'win32' })({})).toThrow(new Error(NO_GPU + ': the system lists an Arc card, so '
        + 'Intel\'s graphics driver is missing or too old. Install the current Arc driver from Intel'));
    });

    test('neither can start, for a reason of the Intel core\'s own: that reason is thrown', () => {
      const intel = addon('intel', () => new Error('GPU 7 was asked for but SYCL lists 6: valid indices are 0..5'));
      const cu12 = addon('cu12', () => new Error('no CUDA device found'));
      const r = rig({ [INTEL_BESIDE]: intel, [CU12_BESIDE]: cu12 });
      expect(() => factory(r, ARC_RIG)({})).toThrow(new Error('GPU 7 was asked for but SYCL lists 6: valid indices are 0..5'));
    });

    test('the 12.8 build fails for a reason that is not about the driver: that one is thrown', () => {
      const intel = addon('intel', () => new Error(NO_GPU));
      const cu12 = addon('cu12', () => new Error('out of memory'));
      const r = rig({ [INTEL_BESIDE]: intel, [CU12_BESIDE]: cu12 });
      expect(() => factory(r, ARC_RIG)({})).toThrow(new Error('out of memory'));
    });

    test('with no 12.8 build behind it, the Intel core\'s error is thrown as it is', () => {
      const r = rig({ [INTEL_BESIDE]: addon('intel', () => new Error(NO_GPU)) });
      expect(() => factory(r, ARC_RIG)({})).toThrow(new Error(NO_GPU));
    });
  });

  describe('asked for by name (--core intel, PEARL_CORE_VARIANT=intel)', () => {
    const ASKED = { env: { PEARL_CORE_VARIANT: 'intel' } };

    test('it is the only core tried, on any rig', () => {
      const intel = addon('intel', () => new Error(NO_GPU));
      const cu12 = addon('cu12');
      const r = rig({ [INTEL_BESIDE]: intel, [CU12_BESIDE]: cu12 });
      const f = factory(r, { ...ASKED, cards: [{ index: 0, major: 12, minor: 0, driverMajor: 610 }] });
      expect(r.lines).toEqual([['info', 'Pearl core: Intel build (SYCL) · PEARL_CORE_VARIANT=intel']]);
      expect(() => f({})).toThrow(new Error(NO_GPU));
      expect(cu12.createCore).not.toHaveBeenCalled();
    });

    test('missing: an error that says where a release keeps it', () => {
      const linux = rig({ [CU12_BESIDE]: addon('cu12') });
      expect(factory(linux, { ...ASKED, coreFlag: true, platform: 'linux' })).toBeNull();
      expect(linux.lines).toEqual([['error', 'Pearl core: Intel build (SYCL) · pearl_core_sycl.node not found (--core intel; the Linux '
        + 'release has it in the sycl folder beside the CLI binary and in the AppImage, or build it with earn/native/sycl/build.sh --jit, '
        + 'or set PEARL_CORE_PATH)']]);
      const win = rig({});
      expect(factory(win, { ...ASKED, platform: 'win32' })).toBeNull();
      expect(win.lines).toEqual([['error', 'Pearl core: Intel build (SYCL) · pearl_core_sycl.node not found (PEARL_CORE_VARIANT=intel; '
        + 'the Windows installer has it in resources\\native\\sycl, or set PEARL_CORE_PATH)']]);
    });

    test('PEARL_CORE_PATH names the file, and nothing else is tried', () => {
      const env = { PEARL_CORE_VARIANT: 'intel', PEARL_CORE_PATH: '/builds/sycl/pearl_core_sycl.node' };
      const here = rig({ '/builds/sycl/pearl_core_sycl.node': addon('named'), [CU12_BESIDE]: addon('cu12') });
      expect(factory(here, { env })()).toEqual({ build: 'named' });
      expect(here.lines).toEqual([['info', 'Pearl core: /builds/sycl/pearl_core_sycl.node (PEARL_CORE_PATH)']]);
      const gone = rig({ [CU12_BESIDE]: addon('cu12') });
      expect(factory(gone, { env })).toBeNull();
      expect(gone.lines).toEqual([['error', 'Pearl core: Intel build (SYCL) · PEARL_CORE_PATH=/builds/sycl/pearl_core_sycl.node not found '
        + '(PEARL_CORE_VARIANT=intel, so no CUDA build is tried)']]);
      const rel = rig({});
      expect(factory(rel, { env: { ...env, PEARL_CORE_PATH: 'sycl/pearl_core_sycl.node' } })).toBeNull();
      expect(rel.lines[0][1]).toContain('not found (a relative path counts from the executable\'s folder; PEARL_CORE_VARIANT=intel, so no CUDA build is tried)');
      const bad = rig({ '/builds/sycl/pearl_core_sycl.node': new Error('libumf.so.1: cannot open shared object file') });
      expect(factory(bad, { env })).toBeNull();
      expect(bad.lines[0][1]).toMatch(/^Pearl core: Intel build \(SYCL\) · \/builds\/sycl\/pearl_core_sycl\.node is there but would not load: .*install the release again \(PEARL_CORE_PATH; PEARL_CORE_VARIANT=intel, so no CUDA build is tried\)$/);
    });
  });
});
