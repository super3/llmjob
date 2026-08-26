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
    expect(addon.createCore).toHaveBeenCalledWith({ rank: 128 });
  });

  test('is null when there is no addon, so the host can say "not built"', () => {
    expect(coreFactory({ require: fakeRequire([]) })).toBeNull();
  });
});
