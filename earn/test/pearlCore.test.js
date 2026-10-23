'use strict';

const path = require('path');
const { coreFactory } = require('../src/main/pearlCore');

const REL = path.join('native', 'build', 'Release', 'pearl_core.node');
const DBG = path.join('native', 'build', 'Debug', 'pearl_core.node');

function fakeRequire(map) {
  return jest.fn((p) => {
    for (const [suffix, val] of map) {
      if (p.endsWith(suffix)) {
        if (val instanceof Error) throw val;
        return val;
      }
    }
    const e = new Error('Cannot find module ' + p);
    e.code = 'MODULE_NOT_FOUND';
    throw e;
  });
}

// An addon whose cores are the addon itself, so a test can see which addon the
// factory loaded by building one core.
function fakeAddon(tag) {
  const a = { tag };
  a.createCore = () => a;
  return a;
}
const ADDON = fakeAddon('release');

// The addon coreFactory found, or null when it found none. No PEARL_CORE_PATH
// and no cards, so this is the 12.8 search.
function loaded(opts) {
  const f = coreFactory(Object.assign({ env: {} }, opts));
  return f ? f() : null;
}

describe('where the core is looked for', () => {
  const probes = () => {
    const seen = [];
    const req = (p) => { seen.push(p); throw new Error('not here'); };
    return { seen, req };
  };

  test('PEARL_CORE_PATH is probed first, before every packaged location', () => {
    const { seen, req } = probes();
    coreFactory({
      require: req,
      env: { PEARL_CORE_PATH: '/rig/custom/pearl_core.node' },
      resourcesPath: '/res',
      execPath: '/opt/rig/llmjob-earn-cli-linux',
    });
    expect(seen[0]).toBe('/rig/custom/pearl_core.node');
  });

  test('a packaged CLI probes beside its executable — resourcesPath is Electron-only', () => {
    const { seen, req } = probes();
    coreFactory({ require: req, env: {}, execPath: '/opt/rig/llmjob-earn-cli-linux' });
    expect(seen).toContain(path.join('/opt/rig', 'pearl_core.node'));
    expect(seen).toContain(path.join('/opt/rig', 'native', 'pearl_core.node'));
    // and the exe-adjacent probe comes before the dev tree's build directories
    expect(seen.indexOf(path.join('/opt/rig', 'pearl_core.node')))
      .toBeLessThan(seen.findIndex((c) => c.includes('build')));
  });

  test('without an injected require it builds a real one and still degrades to null', () => {
    // The default require is created against the real executable so a packaged
    // (SEA) binary can dlopen from the real filesystem. Steer every candidate
    // somewhere nonexistent; on a box with a dev-tree build the addon may
    // genuinely load, so the only portable assertion is "a factory or null,
    // without throwing".
    const f = coreFactory({
      env: { PEARL_CORE_PATH: '/nonexistent/pearl_core.node' },
      execPath: process.execPath,
    });
    expect(f === null || typeof f === 'function').toBe(true);
  });

  test('finds the addon in the dev Release build tree', () => {
    expect(loaded({ require: fakeRequire([[REL, ADDON]]) })).toBe(ADDON);
  });

  test('falls through to the Debug build when Release is absent', () => {
    expect(loaded({ require: fakeRequire([[DBG, ADDON]]) })).toBe(ADDON);
  });

  // In the packaged app the addon ships under resources/native, which must be
  // preferred over any stale dev build tree that happens to be alongside.
  test('prefers the packaged resources path when one is given', () => {
    const packaged = fakeAddon('packaged');
    const req = fakeRequire([
      [path.join('native', 'pearl_core.node'), packaged],
      [REL, ADDON],
    ]);
    expect(loaded({ require: req, resourcesPath: '/res' })).toBe(packaged);
    expect(req.mock.calls[0][0]).toBe(path.join('/res', 'native', 'pearl_core.node'));
  });

  // The expected state on any machine without a CUDA build — including this dev
  // box and CI. It must be a clean null, not a throw, because the host turns it
  // into an "engine not built" message rather than a crash.
  test('is null when the addon is nowhere to be found', () => {
    expect(loaded({ require: fakeRequire([]) })).toBeNull();
  });

  // A present-but-broken addon (wrong ABI, missing CUDA runtime) throws on load.
  // Falling through to null is right: nobody should mine on a core that would
  // not load, and the host already explains that state.
  test('a throwing addon is treated as absent', () => {
    const req = fakeRequire([[REL, new Error('The specified module could not be found.')]]);
    expect(loaded({ require: req })).toBeNull();
  });

  test('an addon without createCore is rejected', () => {
    expect(loaded({ require: fakeRequire([[REL, {}]]) })).toBeNull();
    expect(loaded({ require: fakeRequire([[REL, null]]) })).toBeNull();
  });
});

describe('coreFactory', () => {
  test('with no options it is null, or a usable factory when built', () => {
    const f = coreFactory();
    if (f === null) return;
    expect(typeof f).toBe('function');
  });

  test('returns a factory that builds cores from the addon', () => {
    const made = { id: 'core' };
    const addon = { createCore: jest.fn(() => made) };
    const factory = coreFactory({ require: fakeRequire([[REL, addon]]) });
    expect(factory({ rank: 128 })).toBe(made);
    expect(addon.createCore).toHaveBeenCalledWith({ rank: 128 }, {});
  });

  // The host builds one core per card and says which card and which slice of the
  // search space each one gets. The factory's job is to not lose that.
  test('passes the per-core options through untouched', () => {
    const addon = { createCore: jest.fn(() => ({})) };
    const factory = coreFactory({ require: fakeRequire([[REL, addon]]) });
    factory({ rank: 128 }, { deviceIndex: 1, saltBase: 1, saltStride: 2 });
    expect(addon.createCore)
      .toHaveBeenCalledWith({ rank: 128 }, { deviceIndex: 1, saltBase: 1, saltStride: 2 });
  });

  test('is null when there is no addon, so the host can say "not built"', () => {
    expect(coreFactory({ require: fakeRequire([]) })).toBeNull();
  });
});

// Two builds of the core: pearl_core.node (CUDA 12.8, every card) and
// pearl_core_cu13.node (CUDA 13, sm_89 and sm_120). Which one loads is decided from
// what nvidia-smi said (shared/coreVariant) -- these tests cover the loading
// half: where each is looked for, what is logged, and the fallback.
describe('coreFactory — choosing a build', () => {
  const EXEC = '/opt/rig/llmjob-earn-cli-linux';
  const CU12_BESIDE = path.join('/opt/rig', 'pearl_core.node');
  const CU13_BESIDE = path.join('/opt/rig', 'pearl_core_cu13.node');
  const BLACKWELL = { cards: [{ index: 0, major: 12, minor: 0, driverMajor: 610 }], gpus: [{ index: 0 }] };
  const ADA = { cards: [{ index: 0, major: 8, minor: 9, driverMajor: 610 }], gpus: [{ index: 0 }] };
  const TURING = { cards: [{ index: 0, major: 7, minor: 5, driverMajor: 610 }], gpus: [{ index: 0 }] };
  const PASCAL = { cards: [{ index: 0, major: 6, minor: 1, driverMajor: 580 }], gpus: [{ index: 0 }] };

  // Both builds installed beside the executable, each with a spy createCore.
  function rig(overrides = {}) {
    const cu12 = { createCore: jest.fn(() => ({ build: 'cu12' })) };
    const cu13 = { createCore: jest.fn(() => ({ build: 'cu13' })) };
    const files = new Map([[CU12_BESIDE, cu12], [CU13_BESIDE, cu13]]);
    for (const [k, v] of Object.entries(overrides)) {
      if (v === undefined) files.delete(k); else files.set(k, v);
    }
    const req = jest.fn((p) => {
      if (!files.has(p)) throw Object.assign(new Error('Cannot find module ' + p), { code: 'MODULE_NOT_FOUND' });
      const v = files.get(p);
      if (v instanceof Error) throw v;
      return v;
    });
    const lines = [];
    const log = (level, line) => lines.push([level, line]);
    return { cu12, cu13, req, lines, log };
  }
  const factory = (r, extra) => coreFactory(Object.assign(
    { require: r.req, execPath: EXEC, env: {}, log: r.log }, extra));

  test('an all-Blackwell rig on driver 580+ loads the CUDA 13 build, and says so once', () => {
    const r = rig();
    const f = factory(r, BLACKWELL);
    expect(f({ rank: 128 }, { deviceIndex: 0 })).toEqual({ build: 'cu13' });
    expect(f({ rank: 128 })).toEqual({ build: 'cu13' });
    expect(r.cu13.createCore).toHaveBeenCalledWith({ rank: 128 }, { deviceIndex: 0 });
    expect(r.cu13.createCore).toHaveBeenLastCalledWith({ rank: 128 }, {});
    expect(r.cu12.createCore).not.toHaveBeenCalled();
    // The 12.8 addon is never even loaded: two CUDA runtimes in one process is
    // what the fallback has to risk, not what a rig that passed the gate pays.
    expect(r.req).not.toHaveBeenCalledWith(CU12_BESIDE);
    expect(r.lines).toEqual([
      ['info', 'Pearl core: CUDA 13 build · driver 610, mining card is compute 12.0'],
    ]);
  });

  test('a 4090 rig loads the 12.8 build and never looks for the CUDA 13 one', () => {
    const r = rig();
    const f = factory(r, ADA);
    expect(f({ rank: 128 })).toEqual({ build: 'cu12' });
    expect(r.req).not.toHaveBeenCalledWith(CU13_BESIDE);
    expect(r.lines).toEqual([['info',
      'Pearl core: CUDA 12.8 build · GPU 0 is compute 8.9 (the CUDA 13 build has code for it but measured no faster on a 4090)']]);
  });

  test('a 2080 Ti rig loads the 12.8 build and never looks for the CUDA 13 one', () => {
    const r = rig();
    const f = factory(r, TURING);
    expect(f({ rank: 128 })).toEqual({ build: 'cu12' });
    expect(r.req).not.toHaveBeenCalledWith(CU13_BESIDE);
    expect(r.lines).toEqual([['info',
      'Pearl core: CUDA 12.8 build · GPU 0 is compute 7.5 (the CUDA 13 build has no code for it)']]);
  });

  // A GTX 10 card mines on the 12.8 build's sm_61 dp4a fold; CUDA 13 has no Pascal.
  test('a GTX 1080 Ti rig loads the 12.8 build and never looks for the CUDA 13 one', () => {
    const r = rig();
    const f = factory(r, PASCAL);
    expect(f({ rank: 128 }, { deviceIndex: 0 })).toEqual({ build: 'cu12' });
    expect(r.cu12.createCore).toHaveBeenCalledWith({ rank: 128 }, { deviceIndex: 0 });
    expect(r.req).not.toHaveBeenCalledWith(CU13_BESIDE);
    expect(r.lines).toEqual([['info',
      'Pearl core: CUDA 12.8 build · GPU 0 is compute 6.1 (the CUDA 13 build has no code for it)']]);
  });

  // An install from before there were two, or a release whose CUDA 13 job failed.
  test('a qualifying rig without the CUDA 13 file loads the 12.8 build and says why', () => {
    const r = rig({ [CU13_BESIDE]: undefined });
    expect(factory(r, BLACKWELL)({ rank: 128 })).toEqual({ build: 'cu12' });
    expect(r.lines).toEqual([['info', 'Pearl core: CUDA 12.8 build · pearl_core_cu13.node is not installed']]);
  });

  test('a CUDA 13 file that will not load hands over to the 12.8 build and says why', () => {
    const r = rig({ [CU13_BESIDE]: new Error('invalid ELF header') });
    expect(factory(r, BLACKWELL)({ rank: 128 })).toEqual({ build: 'cu12' });
    expect(r.lines[0]).toEqual(['warn', 'Pearl core: CUDA 12.8 build · ' + CU13_BESIDE
      + ' is there but would not load: invalid ELF header']);
  });

  // A source checkout mines on its own build, whose toolkit is whatever that box
  // has (the 5090 rig builds with CUDA 13.3), so it is named by path, never
  // called the 12.8 release build.
  test('a dev-tree build is logged by its path, not as the 12.8 build', () => {
    const devRel = path.join(__dirname, '..', 'native', 'build', 'Release', 'pearl_core.node');
    const local = { createCore: jest.fn(() => ({ build: 'local' })) };
    const r = rig({ [CU12_BESIDE]: undefined, [CU13_BESIDE]: undefined, [devRel]: local });
    expect(factory(r, BLACKWELL)({})).toEqual({ build: 'local' });
    expect(r.lines).toEqual([['info', 'Pearl core: local build ' + devRel]]);
  });

  test('is null when neither build is there, whatever the rig qualifies for', () => {
    const r = rig({ [CU12_BESIDE]: undefined, [CU13_BESIDE]: undefined });
    expect(factory(r, BLACKWELL)).toBeNull();
    expect(factory(r, ADA)).toBeNull();
    expect(r.lines).toEqual([]);
  });

  // v0.6.7's Linux cores needed glibc 2.38, and on Ubuntu 22.04 that read as a
  // missing core. A core that is there but won't load says why.
  test('a 12.8 core that is there but will not load says why', () => {
    const r = rig({ [CU12_BESIDE]: new Error("/lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.38' not found"),
      [CU13_BESIDE]: undefined });
    expect(factory(r, ADA)).toBeNull();
    expect(r.lines).toEqual([['error', 'Pearl core: ' + CU12_BESIDE + " is there but would not load: "
      + "/lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.38' not found"]]);
  });

  // The GUI keeps both in resources/native, the packaged CLI beside itself.
  test('finds the CUDA 13 build in the packaged app\'s resources first', () => {
    const inRes = { createCore: jest.fn(() => ({ build: 'res-cu13' })) };
    const r = rig({ [path.join('/res', 'native', 'pearl_core_cu13.node')]: inRes });
    expect(factory(r, Object.assign({ resourcesPath: '/res' }, BLACKWELL))({})).toEqual({ build: 'res-cu13' });
  });

  describe('falling back to the 12.8 build', () => {
    const OLD_DRIVER = new Error('no CUDA device found — is an NVIDIA driver installed?');

    // A static CUDA 13 runtime on a pre-580 driver loads fine and fails at its
    // first CUDA call, which is inside createCore.
    test('when the first CUDA 13 core fails with a runtime error', () => {
      const r = rig();
      r.cu13.createCore.mockImplementation(() => { throw OLD_DRIVER; });
      const f = factory(r, BLACKWELL);
      expect(f({ rank: 128 }, { deviceIndex: 0 })).toEqual({ build: 'cu12' });
      expect(r.cu12.createCore).toHaveBeenCalledWith({ rank: 128 }, { deviceIndex: 0 });
      expect(r.lines[1]).toEqual(['warn', 'Pearl core: CUDA 12.8 build · the CUDA 13 build could not '
        + 'start (no CUDA device found — is an NVIDIA driver installed?)']);
      // And it stays there for every later card: one fallback, said once.
      expect(f({ rank: 128 }, { deviceIndex: 1 })).toEqual({ build: 'cu12' });
      expect(r.cu13.createCore).toHaveBeenCalledTimes(1);
      expect(r.lines).toHaveLength(2);
    });

    // Forced onto the CUDA 13 build, a GTX 10 card is refused by that core (it has
    // no sm_61 code), and the 12.8 build, which has, takes over.
    test('when PEARL_CORE_VARIANT=cu13 meets a card the CUDA 13 build has no code for', () => {
      const r = rig();
      r.cu13.createCore.mockImplementation(() => {
        throw new Error('this Pearl core has no GPU code for compute 6.1 (GPU 0, NVIDIA GeForce '
          + 'GTX 1080 Ti): no kernel image is available for execution on the device');
      });
      const f = factory(r, Object.assign({ env: { PEARL_CORE_VARIANT: 'cu13' } }, PASCAL));
      expect(f({ rank: 128 }, { deviceIndex: 0 })).toEqual({ build: 'cu12' });
      expect(r.cu12.createCore).toHaveBeenCalledWith({ rank: 128 }, { deviceIndex: 0 });
      expect(r.lines[1][0]).toBe('warn');
      expect(r.lines[1][1]).toMatch(/could not start \(this Pearl core has no GPU code for compute 6\.1/);
    });

    test('a thrown non-Error is described too', () => {
      const r = rig();
      r.cu13.createCore.mockImplementation(() => { throw 'no kernel image is available'; });
      expect(factory(r, BLACKWELL)({})).toEqual({ build: 'cu12' });
      expect(r.lines[1][1]).toMatch(/could not start \(no kernel image is available\)$/);
    });

    // When the 12.8 build fails too, its own error is the answer.
    test('the 12.8 build\'s own failure is what the host sees', () => {
      const r = rig();
      r.cu13.createCore.mockImplementation(() => { throw OLD_DRIVER; });
      r.cu12.createCore.mockImplementation(() => { throw new Error('no CUDA device found (cu12)'); });
      expect(() => factory(r, BLACKWELL)({})).toThrow('no CUDA device found (cu12)');
    });

    test('not when there is no 12.8 build to fall back to', () => {
      const r = rig({ [CU12_BESIDE]: undefined });
      r.cu13.createCore.mockImplementation(() => { throw OLD_DRIVER; });
      expect(() => factory(r, BLACKWELL)({})).toThrow(OLD_DRIVER);
    });

    // A full card is the card's problem; the 12.8 build would refuse it too,
    // and switching would cost a 5090 its CUDA 13 speed for nothing.
    test('not for an ordinary card failure, which is rethrown as is', () => {
      const r = rig();
      const full = new Error('GPU 0 has 1.2 GiB free, the profile needs 3.1 GiB');
      r.cu13.createCore.mockImplementationOnce(() => { throw full; });
      const f = factory(r, BLACKWELL);
      expect(() => f({})).toThrow(full);
      expect(r.req).not.toHaveBeenCalledWith(CU12_BESIDE);
      // Nothing has started yet, so the next card may still fall back...
      expect(f({})).toEqual({ build: 'cu13' });
    });

    // ...but once a CUDA 13 core runs, the runtime demonstrably works here.
    test('not once a CUDA 13 core has started', () => {
      const r = rig();
      const f = factory(r, BLACKWELL);
      expect(f({})).toEqual({ build: 'cu13' });
      r.cu13.createCore.mockImplementation(() => { throw OLD_DRIVER; });
      expect(() => f({})).toThrow(OLD_DRIVER);
      expect(r.cu12.createCore).not.toHaveBeenCalled();
    });
  });

  describe('PEARL_CORE_VARIANT', () => {
    test('cu13 loads the CUDA 13 build on a rig that would not get it', () => {
      const r = rig();
      expect(factory(r, Object.assign({ env: { PEARL_CORE_VARIANT: 'cu13' } }, ADA))({})).toEqual({ build: 'cu13' });
      expect(r.lines).toEqual([['info', 'Pearl core: CUDA 13 build · PEARL_CORE_VARIANT=cu13']]);
    });

    test('cu12 keeps a qualifying rig on the 12.8 build', () => {
      const r = rig();
      expect(factory(r, Object.assign({ env: { PEARL_CORE_VARIANT: 'cu12' } }, BLACKWELL))({})).toEqual({ build: 'cu12' });
      expect(r.req).not.toHaveBeenCalledWith(CU13_BESIDE);
      expect(r.lines).toEqual([['info', 'Pearl core: CUDA 12.8 build · PEARL_CORE_VARIANT=cu12']]);
    });
  });

  // PEARL_CORE_VARIANT=amd: the AMD core, from the packaged locations or the dev
  // tree, and never a CUDA build in its place.
  describe('the AMD build', () => {
    const AMD_BESIDE = path.join('/opt/rig', 'pearl_core_hip.node');
    const AMD_DEV = path.join(__dirname, '..', 'native', 'amd', 'build', 'pearl_core_hip.node');
    const AMD_ENV = { env: { PEARL_CORE_VARIANT: 'amd' } };

    test('loads pearl_core_hip.node beside the executable, and says so', () => {
      const amd = { createCore: jest.fn(() => ({ build: 'amd' })) };
      const r = rig({ [AMD_BESIDE]: amd });
      const f = factory(r, AMD_ENV);
      expect(f({ rank: 128 }, { deviceIndex: 1 })).toEqual({ build: 'amd' });
      expect(amd.createCore).toHaveBeenCalledWith({ rank: 128 }, { deviceIndex: 1 });
      expect(r.cu12.createCore).not.toHaveBeenCalled();
      expect(r.req).not.toHaveBeenCalledWith(CU12_BESIDE);
      expect(r.lines).toEqual([['info', 'Pearl core: AMD build (ROCm) · PEARL_CORE_VARIANT=amd']]);
    });

    test('finds a dev-tree build and logs its path', () => {
      const amd = { createCore: jest.fn(() => ({ build: 'amd-dev' })) };
      const r = rig({ [AMD_DEV]: amd });
      expect(factory(r, AMD_ENV)({})).toEqual({ build: 'amd-dev' });
      expect(r.lines).toEqual([['info', 'Pearl core: AMD build (ROCm) · local build ' + AMD_DEV
        + ', PEARL_CORE_VARIANT=amd']]);
    });

    test('is null, with the reason, when it is not built; the CUDA build does not stand in', () => {
      const r = rig();
      expect(factory(r, Object.assign({}, AMD_ENV, BLACKWELL))).toBeNull();
      expect(r.req).not.toHaveBeenCalledWith(CU12_BESIDE);
      expect(r.req).not.toHaveBeenCalledWith(CU13_BESIDE);
      expect(r.lines).toEqual([['error', 'Pearl core: AMD build (ROCm) · pearl_core_hip.node not found '
        + '(PEARL_CORE_VARIANT=amd; the Linux release has it beside the CLI binary and in the AppImage, or build it '
        + 'with earn/native/amd/build-amd.sh gpu, or set PEARL_CORE_PATH)']]);
    });

    // The CLI's --core sets PEARL_CORE_VARIANT, so the log names the flag; and
    // Windows has no AMD core at all.
    test('names --core when the CLI asked with it, and says the AMD core is Linux only on Windows', () => {
      const r = rig();
      expect(factory(r, Object.assign({ coreFlag: true, platform: 'win32' }, AMD_ENV))).toBeNull();
      expect(r.lines).toEqual([['error', 'Pearl core: AMD build (ROCm) · pearl_core_hip.node not found '
        + '(--core amd; the AMD core is built for Linux only)']]);
      const amd = { createCore: jest.fn(() => ({ build: 'amd' })) };
      const r2 = rig({ [AMD_BESIDE]: amd });
      factory(r2, Object.assign({ coreFlag: true }, AMD_ENV));
      expect(r2.lines).toEqual([['info', 'Pearl core: AMD build (ROCm) · --core amd']]);
    });

    // The release carries the core but not ROCm. Without the HIP runtime the file is
    // there and dlopen fails, which must not read as "not found".
    test('a core that is there but cannot find the HIP runtime says so, and what to install', () => {
      const dl = Object.assign(new Error('libamdhip64.so.7: cannot open shared object file: No such file or directory'),
        { code: 'ERR_DLOPEN_FAILED' });
      const r = rig({ [AMD_BESIDE]: dl });
      expect(factory(r, AMD_ENV)).toBeNull();
      expect(r.lines).toEqual([['error', 'Pearl core: AMD build (ROCm) · ' + AMD_BESIDE + ' is there but would not '
        + 'load: libamdhip64.so.7: cannot open shared object file: No such file or directory. It needs the ROCm 7 '
        + 'HIP runtime (libamdhip64.so.7): install rocm-hip-runtime 7.x, or put ROCm\'s lib folder on LD_LIBRARY_PATH']]);
    });

    test('another load failure is named as it is, with no ROCm hint', () => {
      const r = rig({ [AMD_BESIDE]: new Error('invalid ELF header') });
      expect(factory(r, AMD_ENV)).toBeNull();
      expect(r.lines).toEqual([['error', 'Pearl core: AMD build (ROCm) · ' + AMD_BESIDE
        + ' is there but would not load: invalid ELF header']]);
    });

    test('is never looked for unless asked', () => {
      const amd = { createCore: jest.fn(() => ({ build: 'amd' })) };
      const r = rig({ [AMD_BESIDE]: amd });
      expect(factory(r, TURING)({})).toEqual({ build: 'cu12' });
      expect(r.req).not.toHaveBeenCalledWith(AMD_BESIDE);
      expect(r.req).not.toHaveBeenCalledWith(AMD_DEV);
    });

    // The fallbacks behind PEARL_CORE_PATH are CUDA builds. On an AMD rig one
    // would load, then fail on every start.
    test('a PEARL_CORE_PATH that does not load is the end of it: no CUDA build is tried', () => {
      const r = rig();
      const env = { PEARL_CORE_VARIANT: 'amd', PEARL_CORE_PATH: '/opt/amd/pearl_core_hip.nod' };
      expect(factory(r, Object.assign({ env }, BLACKWELL))).toBeNull();
      expect(r.req).toHaveBeenCalledWith('/opt/amd/pearl_core_hip.nod');
      expect(r.req).not.toHaveBeenCalledWith(CU12_BESIDE);
      expect(r.req).not.toHaveBeenCalledWith(CU13_BESIDE);
      expect(r.lines).toEqual([['error', 'Pearl core: AMD build (ROCm) · PEARL_CORE_PATH=/opt/amd/pearl_core_hip.nod '
        + 'not found (PEARL_CORE_VARIANT=amd, so no CUDA build is tried)']]);
      // A relative path counts from the executable's folder, which the line says.
      const r2 = rig();
      expect(factory(r2, { env: { PEARL_CORE_VARIANT: 'amd', PEARL_CORE_PATH: 'build/pearl_core_hip.node' } })).toBeNull();
      expect(r2.lines[0][1]).toBe('Pearl core: AMD build (ROCm) · PEARL_CORE_PATH=build/pearl_core_hip.node not found '
        + "(a relative path counts from the executable's folder; PEARL_CORE_VARIANT=amd, so no CUDA build is tried)");
    });

    // Named without PEARL_CORE_VARIANT=amd, the AMD core is tried as the CUDA one
    // would be, and a CUDA build that loads after it must not hide why it failed.
    test('a PEARL_CORE_PATH that is there but will not load, with a CUDA build behind it, says why', () => {
      const r = rig({ '/opt/amd/pearl_core_hip.node': new Error('libamdhip64.so.7: cannot open shared object file') });
      expect(factory(r, { env: { PEARL_CORE_PATH: '/opt/amd/pearl_core_hip.node' } })({})).toEqual({ build: 'cu12' });
      expect(r.lines[0][0]).toBe('warn');
      expect(r.lines[0][1]).toMatch(/^Pearl core: \/opt\/amd\/pearl_core_hip\.node is there but would not load: libamdhip64/);
      expect(r.lines[1]).toEqual(['info', 'Pearl core: ' + CU12_BESIDE + ' (PEARL_CORE_PATH did not load)']);
    });

    test('a PEARL_CORE_PATH that is there but will not load says why', () => {
      const r = rig({ '/opt/amd/pearl_core_hip.node': new Error('librocprofiler-register.so.0: cannot open shared object file') });
      const env = { PEARL_CORE_VARIANT: 'amd', PEARL_CORE_PATH: '/opt/amd/pearl_core_hip.node' };
      expect(factory(r, { env })).toBeNull();
      expect(r.lines[0][1]).toBe('Pearl core: AMD build (ROCm) · /opt/amd/pearl_core_hip.node is there but would not load: '
        + 'librocprofiler-register.so.0: cannot open shared object file. It needs the ROCm 7 HIP runtime (libamdhip64.so.7): '
        + 'install rocm-hip-runtime 7.x, or put ROCm\'s lib folder on LD_LIBRARY_PATH (PEARL_CORE_PATH; '
        + 'PEARL_CORE_VARIANT=amd, so no CUDA build is tried)');
    });

    test('PEARL_CORE_PATH still names the file', () => {
      const mine = { createCore: jest.fn(() => ({ build: 'mine' })) };
      const r = rig({ '/src/pearl_core_hip.node': mine });
      const env = { PEARL_CORE_VARIANT: 'amd', PEARL_CORE_PATH: '/src/pearl_core_hip.node' };
      expect(factory(r, { env })({})).toEqual({ build: 'mine' });
      expect(r.lines).toEqual([['info', 'Pearl core: /src/pearl_core_hip.node (PEARL_CORE_PATH)']]);
    });
  });

  describe('PEARL_CORE_PATH', () => {
    // An operator who names a file gets that file -- no choosing, no fallback.
    test('beats the choice entirely, and says which file loaded', () => {
      const mine = { createCore: jest.fn(() => ({ build: 'mine' })) };
      const r = rig({ '/src/pearl_core.node': mine });
      const f = factory(r, Object.assign({ env: { PEARL_CORE_PATH: '/src/pearl_core.node' } }, BLACKWELL));
      expect(f({})).toEqual({ build: 'mine' });
      expect(r.req).not.toHaveBeenCalledWith(CU13_BESIDE);
      expect(r.lines).toEqual([['info', 'Pearl core: /src/pearl_core.node (PEARL_CORE_PATH)']]);
    });

    test('says so when the named file did not load and the installed 12.8 build did', () => {
      const r = rig();
      const f = factory(r, Object.assign({ env: { PEARL_CORE_PATH: '/nope.node' } }, BLACKWELL));
      expect(f({})).toEqual({ build: 'cu12' });
      expect(r.lines).toEqual([['info', 'Pearl core: ' + CU12_BESIDE + ' (PEARL_CORE_PATH did not load)']]);
    });

    test('is null when nothing loads at all', () => {
      const r = rig({ [CU12_BESIDE]: undefined });
      expect(factory(r, { env: { PEARL_CORE_PATH: '/nope.node' } })).toBeNull();
      expect(r.lines).toEqual([]);
    });

    test('says why when the named file is there but will not load', () => {
      const r = rig({ [CU12_BESIDE]: undefined, '/src/pearl_core.node': new Error("version `GLIBC_2.38' not found") });
      expect(factory(r, { env: { PEARL_CORE_PATH: '/src/pearl_core.node' } })).toBeNull();
      expect(r.lines).toEqual([['error', "Pearl core: /src/pearl_core.node is there but would not load: version `GLIBC_2.38' not found"]]);
    });
  });

  // The GUI and CLI always pass a logger; nothing else has to.
  test('works without a logger', () => {
    const r = rig();
    expect(coreFactory({ require: r.req, execPath: EXEC, env: {}, cards: BLACKWELL.cards })({}))
      .toEqual({ build: 'cu13' });
  });
});
