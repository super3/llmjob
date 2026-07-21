'use strict';

const path = require('path');
const {
  VC_RUNTIME_DLLS, bundledRuntimeDir, missingRuntimeDlls, runtimeCopyPlan,
} = require('../src/shared/llmRuntime');

const join = (...p) => p.join('/');

describe('VC_RUNTIME_DLLS', () => {
  test('names the three VC++ runtime DLLs llama-server links against', () => {
    expect(VC_RUNTIME_DLLS).toEqual(['msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll']);
  });
});

describe('bundledRuntimeDir', () => {
  test('resolves under resourcesPath, null in a dev run', () => {
    expect(bundledRuntimeDir('/res', join)).toBe('/res/llm-runtime');
    expect(bundledRuntimeDir(null, join)).toBeNull();
    expect(bundledRuntimeDir(undefined, join)).toBeNull();
  });
});

describe('missingRuntimeDlls', () => {
  test('lists the DLLs absent from the binary dir on Windows', () => {
    const have = new Set(['/llm/msvcp140.dll']);
    const missing = missingRuntimeDlls('win32', '/llm', (p) => have.has(p), join);
    expect(missing).toEqual(['vcruntime140.dll', 'vcruntime140_1.dll']);
  });
  test('empty when all present, and always empty off Windows', () => {
    expect(missingRuntimeDlls('win32', '/llm', () => true, join)).toEqual([]);
    expect(missingRuntimeDlls('linux', '/llm', () => false, join)).toEqual([]);
  });
});

describe('runtimeCopyPlan', () => {
  const resPath = '/app/resources';

  test('plans copies only for missing DLLs the bundle has', () => {
    // llm dir has msvcp140 already; bundle carries msvcp140 + vcruntime140 (but
    // not vcruntime140_1) → plan copies just vcruntime140.
    const exists = (p) => p === '/llm/msvcp140.dll'
      || p === '/app/resources/llm-runtime/msvcp140.dll'
      || p === '/app/resources/llm-runtime/vcruntime140.dll';
    const plan = runtimeCopyPlan({ platform: 'win32', binDir: '/llm', resourcesPath: resPath, existsFn: exists, joinFn: join });
    expect(plan).toEqual([
      { from: '/app/resources/llm-runtime/vcruntime140.dll', to: '/llm/vcruntime140.dll' },
    ]);
  });

  test('empty in a dev run (no resourcesPath), when nothing is missing, or off Windows', () => {
    expect(runtimeCopyPlan({ platform: 'win32', binDir: '/llm', resourcesPath: null, existsFn: () => false, joinFn: join })).toEqual([]);
    expect(runtimeCopyPlan({ platform: 'win32', binDir: '/llm', resourcesPath: resPath, existsFn: () => true, joinFn: join })).toEqual([]);
    expect(runtimeCopyPlan({ platform: 'linux', binDir: '/llm', resourcesPath: resPath, existsFn: () => false, joinFn: join })).toEqual([]);
  });

  test('works with the real path.join', () => {
    const plan = runtimeCopyPlan({
      platform: 'win32', binDir: 'C:\\llm', resourcesPath: 'C:\\res',
      existsFn: (p) => p.startsWith('C:\\res'), joinFn: path.join,
    });
    expect(plan).toHaveLength(3);
    expect(plan[0].to).toBe(path.join('C:\\llm', 'msvcp140.dll'));
  });
});
