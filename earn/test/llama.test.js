'use strict';

const { ALL_LAYERS } = require('../src/shared/vram');

const { LLM } = require('../src/shared/config');

const {
  resolveServerBinary, resolveServerUrl, serverBaseUrl, buildServerArgs, isServerReady, parseTokensPerSec,
} = require('../src/shared/llama');

describe('resolveServerBinary', () => {
  test('prefers a configured path', () => {
    expect(resolveServerBinary('/opt/llama-server', 'linux')).toBe('/opt/llama-server');
  });
  test('per-platform name, with a linux fallback for unknown platforms', () => {
    expect(resolveServerBinary(null, 'win32')).toBe('llama-server.exe');
    expect(resolveServerBinary(null, 'linux')).toBe('llama-server');
    expect(resolveServerBinary(undefined, 'sunos')).toBe('llama-server');
  });
});

describe('resolveServerUrl', () => {
  test('platforms with one build ignore the arch', () => {
    expect(resolveServerUrl('win32', 'x64')).toBe(LLM.serverUrl.win32);
    expect(resolveServerUrl('win32', 'arm64')).toBe(LLM.serverUrl.win32);
    expect(resolveServerUrl('linux', 'x64')).toBe(LLM.serverUrl.linux);
  });

  // llama.cpp ships macOS as two archives. Handing an Intel Mac the arm64 build
  // installs a binary the kernel refuses to exec — and on macOS the LLM is the
  // only thing the app can run at all.
  test('macOS picks per architecture', () => {
    expect(resolveServerUrl('darwin', 'arm64')).toBe(LLM.serverUrl.darwin);
    expect(resolveServerUrl('darwin', 'x64')).toBe(LLM.serverUrl['darwin-x64']);
    expect(LLM.serverUrl['darwin-x64']).not.toBe(LLM.serverUrl.darwin);
    expect(LLM.serverUrl.darwin).toMatch(/macos-arm64\.tar\.gz$/);
    expect(LLM.serverUrl['darwin-x64']).toMatch(/macos-x64\.tar\.gz$/);
  });

  test('an unrecognised arch falls back to the platform build', () => {
    expect(resolveServerUrl('darwin', 'ppc')).toBe(LLM.serverUrl.darwin);
  });

  test('an unknown platform falls back to linux, like resolveServerBinary', () => {
    expect(resolveServerUrl('sunos', 'x64')).toBe(LLM.serverUrl.linux);
  });
});

describe('serverBaseUrl', () => {
  test('defaults from config and honors overrides', () => {
    expect(serverBaseUrl()).toBe('http://127.0.0.1:8080');
    expect(serverBaseUrl({ host: '0.0.0.0', port: 9090 })).toBe('http://0.0.0.0:9090');
  });
});

describe('buildServerArgs', () => {
  test('defaults --n-gpu-layers to ALL_LAYERS (llama.cpp clamps) and host/port/ctx from config', () => {
    const a = buildServerArgs({ modelPath: '/m.gguf' });
    expect(a).toEqual([
      '--model', '/m.gguf', '--host', '127.0.0.1', '--port', '8080',
      // From config, not a literal: the test says "ctx from config" and pinning
      // the number here just means editing it in two places whenever it moves.
      '--ctx-size', String(LLM.ctxSize), '--n-gpu-layers', String(ALL_LAYERS), '--parallel', '1',
      '--split-mode', 'none',
    ]);
    expect(a).not.toContain('--flash-attn');
  });

  test('always pins the model to one GPU (multi-GPU Vulkan split crashes llama-server)', () => {
    expect(buildServerArgs()).toEqual(expect.arrayContaining(['--split-mode', 'none']));
    expect(buildServerArgs({ nGpuLayers: 0 })).toEqual(expect.arrayContaining(['--split-mode', 'none']));
  });

  test('honors overrides and appends --flash-attn', () => {
    const a = buildServerArgs({ modelPath: '/m.gguf', nGpuLayers: 8, host: '0.0.0.0', port: 9090, ctxSize: 2048, parallel: 2, flashAttn: true });
    expect(a).toEqual(expect.arrayContaining([
      '--n-gpu-layers', '8', '--host', '0.0.0.0', '--port', '9090', '--ctx-size', '2048', '--parallel', '2', '--flash-attn',
    ]));
  });

  test('n-gpu-layers 0 (CPU-only) and an empty model path when omitted', () => {
    const a = buildServerArgs({ nGpuLayers: 0 });
    expect(a).toEqual(expect.arrayContaining(['--n-gpu-layers', '0', '--model', '']));
  });

  test('works with no opts (all defaults from config)', () => {
    const a = buildServerArgs();
    expect(a).toEqual(expect.arrayContaining(['--host', '127.0.0.1', '--n-gpu-layers', String(ALL_LAYERS), '--model', '']));
  });

  test('pins --main-gpu to a non-negative integer index (incl. device 0)', () => {
    expect(buildServerArgs({ mainGpu: 0 })).toEqual(expect.arrayContaining(['--main-gpu', '0', '--split-mode', 'none']));
    expect(buildServerArgs({ mainGpu: 3 })).toEqual(expect.arrayContaining(['--main-gpu', '3']));
  });

  test('omits --main-gpu for a missing or invalid index (llama falls back to device 0)', () => {
    expect(buildServerArgs({}).includes('--main-gpu')).toBe(false);
    expect(buildServerArgs({ mainGpu: -1 }).includes('--main-gpu')).toBe(false);   // negative
    expect(buildServerArgs({ mainGpu: 1.5 }).includes('--main-gpu')).toBe(false);  // not an integer
    expect(buildServerArgs({ mainGpu: '2' }).includes('--main-gpu')).toBe(false);  // wrong type
    expect(buildServerArgs({ mainGpu: null }).includes('--main-gpu')).toBe(false);
  });
});

describe('isServerReady', () => {
  test('matches only post-model-load lines', () => {
    expect(isServerReady('main: server is listening on http://127.0.0.1:8080 - starting the main loop')).toBe(true);
    expect(isServerReady('main: model loaded')).toBe(true);
    expect(isServerReady('srv  update_slots: all slots are idle')).toBe(true);
  });
  test('rejects the pre-load listening line and loading noise', () => {
    // printed BEFORE the model loads, while /v1/chat/completions still 503s
    expect(isServerReady('main: HTTP server is listening, hostname: 127.0.0.1, port: 8080, http threads: 15')).toBe(false);
    expect(isServerReady('loading model from /m.gguf')).toBe(false);
    expect(isServerReady(null)).toBe(false);
  });
});

describe('parseTokensPerSec', () => {
  test('extracts tokens/sec from a timing line, else null', () => {
    expect(parseTokensPerSec('eval time = 1234 ms / 200 tokens ( ... 162.02 tokens per second)')).toBeCloseTo(162.02);
    expect(parseTokensPerSec('nothing here')).toBeNull();
    expect(parseTokensPerSec(null)).toBeNull();
  });
});

describe('buildServerArgs — vision projector', () => {
  test('passes --mmproj when the model ships a projector', () => {
    const a = buildServerArgs({ modelPath: '/m.gguf', mmprojPath: '/mmproj.gguf' });
    expect(a).toEqual(expect.arrayContaining(['--mmproj', '/mmproj.gguf']));
  });

  test('omits it entirely when there is none', () => {
    // A missing projector is not an error: the weights serve fine as a text-only
    // model. Passing an empty --mmproj would be, so the flag must not appear.
    expect(buildServerArgs({ modelPath: '/m.gguf' })).not.toContain('--mmproj');
    expect(buildServerArgs({ modelPath: '/m.gguf', mmprojPath: '' })).not.toContain('--mmproj');
    expect(buildServerArgs({ modelPath: '/m.gguf', mmprojPath: null })).not.toContain('--mmproj');
  });
});

describe('buildServerArgs — per-model extra flags', () => {
  test('appends a model\'s own flags verbatim, in order, last', () => {
    const a = buildServerArgs({ modelPath: '/m.gguf', extraArgs: ['-fa', '1', '--cache-type-k', 'q8_0'] });
    expect(a.slice(-4)).toEqual(['-fa', '1', '--cache-type-k', 'q8_0']);
  });

  test('no extra flags is the existing behaviour, unchanged', () => {
    const base = buildServerArgs({ modelPath: '/m.gguf' });
    expect(buildServerArgs({ modelPath: '/m.gguf', extraArgs: [] })).toEqual(base);
    expect(buildServerArgs({ modelPath: '/m.gguf', extraArgs: null })).toEqual(base);
    expect(buildServerArgs({ modelPath: '/m.gguf', extraArgs: 'not an array' })).toEqual(base);
  });

  test('drops empty entries rather than passing a blank argv slot to llama-server', () => {
    const a = buildServerArgs({ modelPath: '/m.gguf', extraArgs: ['--jinja', '', null, undefined, '--kv-unified'] });
    expect(a.slice(-2)).toEqual(['--jinja', '--kv-unified']);
  });

  test('numbers survive as strings, since argv is strings', () => {
    const a = buildServerArgs({ modelPath: '/m.gguf', extraArgs: ['--spec-draft-n-max', 3] });
    expect(a.slice(-2)).toEqual(['--spec-draft-n-max', '3']);
  });
});
