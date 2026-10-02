'use strict';

const path = require('path');
const { loadCore, coreFactory } = require('../src/main/pearlCore');

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

const ADDON = { createCore: () => ({}) };

describe('loadCore candidate paths', () => {
  const probes = () => {
    const seen = [];
    const req = (p) => { seen.push(p); throw new Error('not here'); };
    return { seen, req };
  };

  test('PEARL_CORE_PATH is probed first, before every packaged location', () => {
    const { seen, req } = probes();
    loadCore({
      require: req,
      env: { PEARL_CORE_PATH: '/rig/custom/pearl_core.node' },
      resourcesPath: '/res',
      execPath: '/opt/rig/llmjob-earn-cli-linux',
    });
    expect(seen[0]).toBe('/rig/custom/pearl_core.node');
  });

  test('a packaged CLI probes beside its executable — resourcesPath is Electron-only', () => {
    const { seen, req } = probes();
    loadCore({ require: req, env: {}, execPath: '/opt/rig/llmjob-earn-cli-linux' });
    const path = require('path');
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
    // genuinely load, so the only portable assertion is "returns the addon or
    // null, without throwing".
    const out = loadCore({
      env: { PEARL_CORE_PATH: '/nonexistent/pearl_core.node' },
      execPath: process.execPath,
    });
    expect(out === null || typeof out.createCore === 'function').toBe(true);
  });
});

describe('loadCore', () => {
  test('finds the addon in the dev Release build tree', () => {
    const req = fakeRequire([[REL, ADDON]]);
    expect(loadCore({ require: req })).toBe(ADDON);
  });

  test('falls through to the Debug build when Release is absent', () => {
    const req = fakeRequire([[DBG, ADDON]]);
    expect(loadCore({ require: req })).toBe(ADDON);
  });

  // In the packaged app the addon ships under resources/native, which must be
  // preferred over any stale dev build tree that happens to be alongside.
  test('prefers the packaged resources path when one is given', () => {
    const packaged = { createCore: () => ({ packaged: true }) };
    const req = fakeRequire([
      [path.join('native', 'pearl_core.node'), packaged],
      [REL, ADDON],
    ]);
    expect(loadCore({ require: req, resourcesPath: '/res' })).toBe(packaged);
    expect(req.mock.calls[0][0]).toBe(path.join('/res', 'native', 'pearl_core.node'));
  });

  // The expected state on any machine without a CUDA build — including this dev
  // box and CI. It must be a clean null, not a throw, because the host turns it
  // into an "engine not built" message rather than a crash.
  test('returns null when the addon is nowhere to be found', () => {
    expect(loadCore({ require: fakeRequire([]) })).toBeNull();
  });

  // A present-but-broken addon (wrong ABI, missing CUDA runtime) throws on load.
  // Falling through to null is right: nobody should mine on a core that would
  // not load, and the host already explains that state.
  test('a throwing addon is treated as absent', () => {
    const req = fakeRequire([[REL, new Error('The specified module could not be found.')]]);
    expect(loadCore({ require: req })).toBeNull();
  });

  test('an addon without createCore is rejected', () => {
    expect(loadCore({ require: fakeRequire([[REL, {}]]) })).toBeNull();
    expect(loadCore({ require: fakeRequire([[REL, null]]) })).toBeNull();
  });
});

describe('loadCore — real defaults', () => {
  // Called with no options at all it uses the real require and the real build
  // paths, so the answer depends on whether this machine happens to have a
  // compiled core. It used to assert a flat null, which was true of CI and of
  // the dev box until the dev box got a local toolchain -- at which point the
  // test failed for a reason that had nothing to do with the code.
  //
  // What actually matters is the contract: probing never throws, and either
  // finds nothing or finds something usable.
  test('probes the real paths without throwing', () => {
    const core = loadCore();
    if (core === null) return;                    // unbuilt: the common case
    expect(typeof core).toBe('object');
    expect(core.createCore || core.PearlCore).toBeTruthy();
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

  // Both builds installed beside the executable, each with a spy createCore.
  function rig(overrides = {}) {
    const cu12 = { createCore: jest.fn(() => ({ build: 'cu12' })) };
    const cu13 = { createCore: jest.fn(() => ({ build: 'cu13' })) };
    const files = new Map([[CU12_BESIDE, cu12], [CU13_BESIDE, cu13]]);
    for (const [k, v] of Object.entries(overrides)) {
      if (v === undefined) files.delete(k); else files.set(k, v);
    }
    const req = jest.fn((p) => {
      if (!files.has(p)) throw new Error('Cannot find module ' + p);
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
      'Pearl core: CUDA 12.8 build · GPU 0 is compute 8.9 (the CUDA 13 build has code for it but is not yet selected automatically)']]);
  });

  // An install from before there were two, or a release whose CUDA 13 job failed.
  test('a qualifying rig without the CUDA 13 file loads the 12.8 build and says why', () => {
    const r = rig({ [CU13_BESIDE]: undefined });
    expect(factory(r, BLACKWELL)({ rank: 128 })).toEqual({ build: 'cu12' });
    expect(r.lines).toEqual([['info', 'Pearl core: CUDA 12.8 build · pearl_core_cu13.node is not installed']]);
  });

  test('a CUDA 13 file that will not load counts as not installed', () => {
    const r = rig({ [CU13_BESIDE]: new Error('invalid ELF header') });
    expect(factory(r, BLACKWELL)({ rank: 128 })).toEqual({ build: 'cu12' });
    expect(r.lines[0][1]).toBe('Pearl core: CUDA 12.8 build · pearl_core_cu13.node is not installed');
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
    });
  });

  // The GUI and CLI always pass a logger; nothing else has to.
  test('works without a logger', () => {
    const r = rig();
    expect(coreFactory({ require: r.req, execPath: EXEC, env: {}, cards: BLACKWELL.cards })({}))
      .toEqual({ build: 'cu13' });
  });
});
