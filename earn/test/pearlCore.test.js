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
  // paths. On any machine without a compiled core — this dev box, and CI — that
  // is a clean null, which is exactly the state the host explains to the user.
  test('probes the real paths and finds nothing when unbuilt', () => {
    expect(loadCore()).toBeNull();
  });
});

describe('coreFactory', () => {
  test('is null with no options when the core is unbuilt', () => {
    expect(coreFactory()).toBeNull();
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
