'use strict';

const { ALL_LAYERS } = require('../src/shared/vram');

const { LLM } = require('../src/shared/config');

const {
  resolveServerBinary, resolveServerUrl, serverBaseUrl, buildServerArgs, buildServerEnv, isServerReady,
  parseTokensPerSec,
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

  // --main-gpu counts cards in llama-server's order. A name says which card it
  // is, and with one device named, --main-gpu 1 would be out of range.
  test('loads on the named device instead of --main-gpu when it has one', () => {
    const a = buildServerArgs({ mainGpu: 1, device: 'Vulkan0' });
    expect(a).toEqual(expect.arrayContaining(['--device', 'Vulkan0', '--split-mode', 'none']));
    expect(a.includes('--main-gpu')).toBe(false);
    expect(buildServerArgs({ mainGpu: 1, device: '' })).toEqual(expect.arrayContaining(['--main-gpu', '1']));
    expect(buildServerArgs({ mainGpu: 1 }).includes('--device')).toBe(false);
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

// mtmd ignores --device and loads the projector on the first GPU unless told.
describe('buildServerEnv', () => {
  test('puts the projector on the model\'s card', () => {
    expect(buildServerEnv({ device: 'Vulkan1', mmprojPath: '/p.gguf' }, { PATH: '/bin' }))
      .toEqual({ PATH: '/bin', MTMD_BACKEND_DEVICE: 'Vulkan1' });
  });

  test('null without a device or a projector, or when the user set it', () => {
    expect(buildServerEnv({ mmprojPath: '/p.gguf' }, {})).toBeNull();
    expect(buildServerEnv({ device: 'Vulkan1' }, {})).toBeNull();
    expect(buildServerEnv({ device: 'Vulkan1', mmprojPath: '/p.gguf' }, { MTMD_BACKEND_DEVICE: 'Vulkan0' })).toBeNull();
    expect(buildServerEnv()).toBeNull();
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

describe('parseTiming', () => {
  const { parseTiming } = require('../src/shared/llama');

  test('tags a generation line', () => {
    expect(parseTiming('eval time = 10 ms / 200 tokens ... 162.02 tokens per second'))
      .toEqual({ kind: 'gen', tokensPerSec: 162.02, tokens: 200 });
  });

  test('tags a prefill line', () => {
    expect(parseTiming('prompt eval time = 5 ms / 900 tokens ... 1840.00 tokens per second'))
      .toEqual({ kind: 'prompt', tokensPerSec: 1840, tokens: 900 });
  });

  // The real format, with llama-server's padding, from a two-card rig's log.
  test('reads the token count from a real generation line', () => {
    expect(parseTiming('0.42.314.567 I slot print_timing: id 0 | task 0 |        eval time =    3292.87 ms /   270 tokens (   12.20 ms per token,    82.00 tokens per second)'))
      .toEqual({ kind: 'gen', tokensPerSec: 82, tokens: 270 });
  });

  // Printed mid-prefill. It carries a rate but no "/ N tokens", so it is no
  // measure of a card's generation speed.
  test('has no token count on a prefill progress line', () => {
    expect(parseTiming('slot print_timing: id 0 | task 0 | prompt processing, n_tokens = 8, progress = 0.42, t = 9.93 s / 0.81 tokens per second'))
      .toEqual({ kind: 'gen', tokensPerSec: 0.81, tokens: null });
  });

  test('is null for a line with no timing', () => {
    expect(parseTiming('model loaded')).toBeNull();
    expect(parseTiming(null)).toBeNull();
    expect(parseTiming()).toBeNull();
  });

  test('parseTokensPerSec still returns the bare number', () => {
    const { parseTokensPerSec } = require('../src/shared/llama');
    expect(parseTokensPerSec('eval time = 1 ms / 2 tokens ... 9.5 tokens per second')).toBe(9.5);
    expect(parseTokensPerSec('nothing here')).toBeNull();
  });
});

describe('parseSlotEvent', () => {
  const { parseSlotEvent } = require('../src/shared/llama');

  // The exact lines the pinned build (b9902) prints. If an upgrade rewords them,
  // these fail, rather than a node pausing mining for a prompt it never sees end.
  test('reads the start and the end of a request from the pinned build', () => {
    expect(parseSlotEvent('slot launch_slot_: id  0 | task 12 | processing task, is_child = 0'))
      .toEqual({ slot: 0, busy: true });
    expect(parseSlotEvent('slot      release: id  0 | task 12 | stop processing: n_tokens = 250, truncated = 0'))
      .toEqual({ slot: 0, busy: false });
  });

  test('keeps the slot number, and accepts a slot with no task', () => {
    expect(parseSlotEvent('slot launch_slot_: id  3 | task -1 | processing task, is_child = 1'))
      .toEqual({ slot: 3, busy: true });
  });

  test('ignores every other line', () => {
    expect(parseSlotEvent('srv  update_slots: all slots are idle')).toBeNull();
    expect(parseSlotEvent('slot print_timing: id  0 | task 12 | prompt eval time = 10 ms')).toBeNull();
    expect(parseSlotEvent('que    start_loop: processing task, id = 7')).toBeNull();
    expect(parseSlotEvent('')).toBeNull();
    expect(parseSlotEvent(null)).toBeNull();
  });
});

describe('isDeviceLost', () => {
  const { isDeviceLost } = require('../src/shared/llama');

  // As the pinned build (b9902) prints them, with the log's timestamp, from a
  // node whose card dropped out from under it. If an upgrade rewords them, these
  // fail, rather than the node going back to failing every job.
  test('reads both shapes of a lost GPU from the pinned build', () => {
    expect(isDeviceLost('4.55.314.438 E srv  update_slots: decode() failed: vk::Device::waitForFences: ErrorDeviceLost')).toBe(true);
    expect(isDeviceLost('9.52.225.242 E slot      iterate: id  0 | task 14 | got exception: vk::Queue::submit: ErrorDeviceLost')).toBe(true);
    // Without the timestamp, with a trailing carriage return, and from the
    // other calls the same error can come out of.
    expect(isDeviceLost('E srv  update_slots: decode() failed: vk::Device::waitForFences: ErrorDeviceLost\r')).toBe(true);
    expect(isDeviceLost('srv  update_slots: post_decode() failed: vk::Device::getFenceStatus: ErrorDeviceLost')).toBe(true);
    expect(isDeviceLost('121.03.000.001 E srv  update_slots: pre_decode() failed: vk::Device::waitSemaphores: ErrorDeviceLost')).toBe(true);
  });

  test('ignores a card that ran out of memory, and lines that only repeat or quote the error', () => {
    // Out of memory is a sizing problem with its own handling, not a lost card.
    expect(isDeviceLost('ggml_vulkan: vk::Device::allocateMemory: ErrorOutOfDeviceMemory')).toBe(false);
    expect(isDeviceLost('E srv  update_slots: decode() failed: vk::Device::allocateMemory: ErrorOutOfDeviceMemory')).toBe(false);
    expect(isDeviceLost('0.10.213.478 E alloc_tensor_range: failed to allocate Vulkan0 buffer of size 1060214272')).toBe(false);
    // The line after the real one repeats it; one is enough.
    expect(isDeviceLost('4.55.314.461 E srv    send_error: task id = 0, error: decode() failed: vk::Device::waitForFences: ErrorDeviceLost')).toBe(false);
    // A request echoed into the log can carry any text.
    expect(isDeviceLost('W srv    operator(): got exception: {"error":"decode() failed: vk::Device::waitForFences: ErrorDeviceLost"}')).toBe(false);
    expect(isDeviceLost('srv  update_slots: decode() failed: vk::Device::waitForFences: ErrorDeviceLost and more')).toBe(false);
    expect(isDeviceLost('')).toBe(false);
    expect(isDeviceLost(null)).toBe(false);
  });
});

describe('parseDeviceList', () => {
  const { parseDeviceList } = require('../src/shared/llama');

  test('reads llama-server --list-devices, in its order', () => {
    // As b9902 prints it on a one-card Windows rig.
    expect(parseDeviceList('Available devices:\r\n  Vulkan0: NVIDIA GeForce RTX 4090 (24138 MiB, 23370 MiB free)\r\n'))
      .toEqual([{ device: 'Vulkan0', name: 'NVIDIA GeForce RTX 4090' }]);
    expect(parseDeviceList([
      'Available devices:',
      '  Vulkan0: NVIDIA GeForce RTX 4070 (11874 MiB, 11200 MiB free)',
      '  Vulkan1: AMD Radeon RX 7900 XTX (RADV NAVI31) (24560 MiB, 24000 MiB free)',
      '  Vulkan2: NVIDIA RTX PRO 4500 Blackwell (32111 MiB, 31500 MiB free)',
    ].join('\n'))).toEqual([
      { device: 'Vulkan0', name: 'NVIDIA GeForce RTX 4070' },
      { device: 'Vulkan1', name: 'AMD Radeon RX 7900 XTX (RADV NAVI31)' },
      { device: 'Vulkan2', name: 'NVIDIA RTX PRO 4500 Blackwell' },
    ]);
  });

  test('empty for anything else', () => {
    expect(parseDeviceList('Available devices:\n')).toEqual([]);
    expect(parseDeviceList('ggml_vulkan: Found 1 Vulkan devices:')).toEqual([]);
    expect(parseDeviceList(null)).toEqual([]);
  });
});

describe('matchLlmDevices', () => {
  const { matchLlmDevices } = require('../src/shared/llama');
  const card = (index, name) => ({ index, name, usedMb: 0, totalMb: 24000 });
  const dev = (device, name) => ({ device, name });

  // The report: nvidia-smi's GPU 0 is a 32 GB RTX PRO 4500, but Vulkan lists the
  // 12 GB RTX 4070 that drives the desktop first. --main-gpu 0 put the 27B model
  // on the 4070, and it never started.
  test('finds each card under llama-server\'s own number', () => {
    expect(matchLlmDevices(
      [card(0, 'NVIDIA RTX PRO 4500 Blackwell'), card(1, 'NVIDIA GeForce RTX 4070')],
      [dev('Vulkan0', 'NVIDIA GeForce RTX 4070'), dev('Vulkan1', 'NVIDIA RTX PRO 4500 Blackwell')],
    )).toEqual({ 0: 'Vulkan1', 1: 'Vulkan0' });
  });

  test('ignores case and punctuation in names, and cards llama-server has that nvidia-smi does not', () => {
    expect(matchLlmDevices(
      [card(0, 'NVIDIA RTX PRO 4500 Blackwell')],
      [dev('Vulkan0', 'AMD Radeon RX 7900 XTX'), dev('Vulkan1', 'nvidia rtx pro 4500 (Blackwell)')],
    )).toEqual({ 0: 'Vulkan1' });
  });

  // Nothing tells two cards of one name apart, and either holds what the other does.
  test('pairs cards of the same name in order', () => {
    expect(matchLlmDevices(
      [card(2, 'RTX 3090'), card(0, 'RTX 3090'), card(1, 'RTX 4070')],
      [dev('Vulkan0', 'RTX 4070'), dev('Vulkan1', 'RTX 3090'), dev('Vulkan2', 'RTX 3090')],
    )).toEqual({ 0: 'Vulkan1', 1: 'Vulkan0', 2: 'Vulkan2' });
  });

  test('pairs the one NVIDIA card left on each side when both list as many', () => {
    expect(matchLlmDevices(
      [card(0, 'NVIDIA RTX PRO 4500 Blackwell'), card(1, 'NVIDIA GeForce RTX 4070')],
      [dev('Vulkan0', 'NVIDIA GeForce RTX 4070'), dev('Vulkan1', 'NVIDIA RTX PRO Blackwell 4500')],
    )).toEqual({ 0: 'Vulkan1', 1: 'Vulkan0' });
    // Other makers' cards in llama-server's list don't count against that.
    expect(matchLlmDevices(
      [card(0, 'NVIDIA RTX PRO 4500 Blackwell'), card(1, 'NVIDIA GeForce RTX 4070')],
      [dev('Vulkan0', 'NVIDIA GeForce RTX 4070'), dev('Vulkan1', 'Intel(R) UHD Graphics 770'),
        dev('Vulkan2', 'NVIDIA RTX PRO Blackwell 4500')],
    )).toEqual({ 0: 'Vulkan2', 1: 'Vulkan0' });
    // Not when the counts differ: the one left over could be any card.
    expect(matchLlmDevices(
      [card(0, 'NVIDIA RTX A'), card(1, 'NVIDIA RTX B')],
      [dev('Vulkan0', 'NVIDIA RTX B'), dev('Vulkan1', 'NVIDIA RTX C'), dev('Vulkan2', 'NVIDIA RTX D')],
    )).toEqual({ 1: 'Vulkan0' });
  });

  // An A6000 in TCC mode is in nvidia-smi's list but not Vulkan's. Pairing it
  // with the iGPU, the other one left over, would put the model in shared
  // system memory, where --main-gpu never could: llama.cpp skips iGPUs when a
  // discrete card is there.
  test('never pairs a card with an integrated GPU or another maker\'s card', () => {
    expect(matchLlmDevices(
      [card(0, 'NVIDIA RTX A6000'), card(1, 'NVIDIA GeForce RTX 4090')],
      [dev('Vulkan0', 'NVIDIA GeForce RTX 4090'), dev('Vulkan1', 'Intel(R) UHD Graphics 770')],
    )).toEqual({ 1: 'Vulkan0' });
    expect(matchLlmDevices(
      [card(0, 'Tesla P40'), card(1, 'NVIDIA GeForce RTX 3060')],
      [dev('Vulkan0', 'NVIDIA GeForce RTX 3060'), dev('Vulkan1', 'AMD Radeon(TM) Graphics')],
    )).toEqual({ 1: 'Vulkan0' });
  });

  test('leaves out a name whose counts differ, or with no name at all', () => {
    expect(matchLlmDevices(
      [card(0, 'RTX 3090'), card(1, 'RTX 3090'), card(2, 'RTX 4070')],
      [dev('Vulkan0', 'RTX 3090'), dev('Vulkan1', 'RTX 4070')],
    )).toEqual({ 2: 'Vulkan1' });
    expect(matchLlmDevices(
      [card(0, ''), card(1, 'RTX 4070'), card(2, 'RTX 4070'), { index: null, name: 'RTX 4070' }, null],
      [dev('Vulkan0', ''), dev('Vulkan1', 'RTX 4070'), { name: 'RTX 4070' }],
    )).toEqual({});
  });

  test('empty without both lists', () => {
    expect(matchLlmDevices([card(0, 'RTX 4090')], [])).toEqual({});
    expect(matchLlmDevices(null, null)).toEqual({});
  });

  // Two 3090s, the desktop on nvidia-smi's GPU 1, which Vulkan lists first.
  // By name they pair in order, the wrong way round. Bus ids say which is which.
  test('pairs cards by bus id first, same-name cards included', () => {
    const name = 'NVIDIA GeForce RTX 3090';
    const busIds = { 0: '00000000:01:00.0', 1: '00000000:02:00.0' };
    const vk = [{ device: 'Vulkan0', name, busId: '0000:02:00.0' }, { device: 'Vulkan1', name, busId: '0000:01:00.0' }];
    expect(matchLlmDevices([card(0, name), card(1, name)], vk, busIds)).toEqual({ 0: 'Vulkan1', 1: 'Vulkan0' });
    // Without them, the old guess.
    expect(matchLlmDevices([card(0, name), card(1, name)], vk)).toEqual({ 0: 'Vulkan0', 1: 'Vulkan1' });
  });

  test('matches by name the cards bus ids leave over, but never two that differ', () => {
    const vk = [
      { device: 'Vulkan0', name: 'NVIDIA GeForce RTX 4070', busId: '0000:02:00.0' },
      { device: 'Vulkan1', name: 'NVIDIA RTX PRO 4500 Blackwell' },
    ];
    // GPU 0 has no bus id, so its name decides. GPU 1's matched by bus id.
    expect(matchLlmDevices([card(0, 'NVIDIA RTX PRO 4500 Blackwell'), card(1, 'NVIDIA GeForce RTX 4070')], vk,
      { 1: '00000000:02:00.0' })).toEqual({ 0: 'Vulkan1', 1: 'Vulkan0' });
    // Same name, but the bus ids say they are different cards.
    expect(matchLlmDevices([card(0, 'NVIDIA GeForce RTX 4070')], vk.slice(0, 1), { 0: '00000000:03:00.0' }))
      .toEqual({});
    // Nor as the one card left on each side.
    expect(matchLlmDevices([card(0, 'NVIDIA RTX A')], [{ device: 'Vulkan0', name: 'NVIDIA RTX B', busId: '0000:02:00.0' }],
      { 0: '00000000:03:00.0' })).toEqual({});
  });
});

describe('placeLlmPlan', () => {
  const { placeLlmPlan } = require('../src/shared/llama');
  const cards = [
    { index: 0, name: 'NVIDIA RTX PRO 4500 Blackwell' },
    { index: 1, name: 'NVIDIA GeForce RTX 4070' },
  ];
  const devices = [
    { device: 'Vulkan0', name: 'NVIDIA GeForce RTX 4070' },
    { device: 'Vulkan1', name: 'NVIDIA RTX PRO 4500 Blackwell' },
  ];

  test('gives each planned card its device, and leaves the plan itself alone', () => {
    const plan = [{ index: 0, freeMb: 28000, nGpuLayers: 999 }];
    expect(placeLlmPlan(plan, cards, devices)).toEqual({
      plan: [{ index: 0, freeMb: 28000, nGpuLayers: 999, device: 'Vulkan1' }],
      warnings: [],
    });
    expect(plan[0].device).toBeUndefined();
  });

  const amd = [{ device: 'Vulkan0', name: 'AMD Radeon RX 7900 XTX' }];

  test('warns for a card it could not match on a rig of several, which keeps --main-gpu', () => {
    const plan = [{ index: 0, nGpuLayers: 999 }, { index: 1, nGpuLayers: 999 }];
    expect(placeLlmPlan(plan, cards, amd)).toEqual({
      plan,
      warnings: [
        "could not tell which of llama-server's cards is GPU 0 (NVIDIA RTX PRO 4500 Blackwell), so the local LLM may load on a different card",
        "could not tell which of llama-server's cards is GPU 1 (NVIDIA GeForce RTX 4070), so the local LLM may load on a different card",
      ],
    });
    // A card nvidia-smi gave no name, or that is not in its list.
    expect(placeLlmPlan([{ index: 0 }, { index: 3 }], [{ index: 0 }, null, { index: 1, name: 'x' }], amd).warnings).toEqual([
      "could not tell which of llama-server's cards is GPU 0, so the local LLM may load on a different card",
      "could not tell which of llama-server's cards is GPU 3, so the local LLM may load on a different card",
    ]);
  });

  // Once, and on a one-card rig too: a failed listing should not pass unseen.
  test('says once when llama-server listed no cards', () => {
    const line = 'llama-server did not list its cards, so the local LLM is placed by GPU number (--main-gpu), which may be a different card';
    expect(placeLlmPlan([{ index: 0 }, { index: 1 }], cards, []).warnings).toEqual([line]);
    expect(placeLlmPlan([{ index: 0 }], [cards[0]], null).warnings).toEqual([line]);
  });

  // One NVIDIA card Vulkan can't see (TCC mode, say) and an AMD card or iGPU
  // it can: --main-gpu 0 then loads on that, the model sized for the NVIDIA card.
  test('warns on a one-card rig too when the card is not in llama-server\'s list', () => {
    expect(placeLlmPlan([{ index: 0 }], [{ index: 0, name: 'NVIDIA RTX A6000' }], amd)).toEqual({
      plan: [{ index: 0 }],
      warnings: ["could not tell which of llama-server's cards is GPU 0 (NVIDIA RTX A6000), so the local LLM may load on a different card"],
    });
  });

  test('quiet for a matched card and an unmeasured one, and an empty plan stays empty', () => {
    expect(placeLlmPlan([{ index: 0 }], [cards[0]], devices).warnings).toEqual([]);
    expect(placeLlmPlan([{ index: null }], cards, [])).toEqual({ plan: [{ index: null }], warnings: [] });
    expect(placeLlmPlan(null, null, null)).toEqual({ plan: [], warnings: [] });
  });

  // Two 5090s with the desktop on GPU 1: the plan takes only GPU 0, but Vulkan
  // may list GPU 1 first, and the names can't tell them apart.
  test('warns when a planned card shares its name with one the plan left out for less room', () => {
    const name = 'NVIDIA GeForce RTX 5090';
    const twins = [{ index: 0, name, usedMb: 400, totalMb: 32607 }, { index: 1, name, usedMb: 2900, totalMb: 32607 }];
    const vk = [{ device: 'Vulkan0', name }, { device: 'Vulkan1', name }];
    expect(placeLlmPlan([{ index: 0, freeMb: 32207 }], twins, vk)).toEqual({
      plan: [{ index: 0, freeMb: 32207, device: 'Vulkan0' }],
      warnings: ['GPU 0 (NVIDIA GeForce RTX 5090) has the same name as another card here, so llama-server may load the local LLM on that one instead'],
    });
    // Unmeasured counts as less room.
    expect(placeLlmPlan([{ index: 0 }], [{ index: 0, name }, { index: 1, name }], vk).warnings).toHaveLength(1);
    // Every twin planned: whichever card each gets, each has room.
    expect(placeLlmPlan([{ index: 0, freeMb: 32207 }, { index: 1, freeMb: 29707 }], twins, vk).warnings).toEqual([]);
  });

  // The CLI's --llm-max-instances leaves out cards that have room. Landing on
  // one of those is no worse than landing on the planned card.
  test('quiet when the twin left out has as much room', () => {
    const name = 'NVIDIA GeForce RTX 3090';
    const rig = [0, 1, 2].map((index) => ({ index, name, usedMb: 1000, totalMb: 24576 }));
    const vk = [0, 1, 2].map((i) => ({ device: 'Vulkan' + i, name }));
    expect(placeLlmPlan([{ index: 0, freeMb: 23576 }, { index: 1, freeMb: 23576 }], rig, vk).warnings).toEqual([]);
  });

  // Prompt pause reads the instance's bus id to know which card to stop.
  test('gives a card its bus id when the match is certain', () => {
    const busIds = { 0: '00000000:01:00.0', 1: '00000000:02:00.0' };
    // By a name no other card has.
    expect(placeLlmPlan([{ index: 0 }], cards, devices, busIds).plan)
      .toEqual([{ index: 0, device: 'Vulkan1', busId: '00000000:01:00.0' }]);
    // Not when nvidia-smi gave none.
    expect(placeLlmPlan([{ index: 0 }], cards, devices, {}).plan).toEqual([{ index: 0, device: 'Vulkan1' }]);
  });

  test('matched by bus id, a twin gets its bus id and no warning', () => {
    const name = 'NVIDIA GeForce RTX 5090';
    const twins = [{ index: 0, name, usedMb: 400, totalMb: 32607 }, { index: 1, name, usedMb: 2900, totalMb: 32607 }];
    const busIds = { 0: '00000000:01:00.0', 1: '00000000:02:00.0' };
    const vk = [{ device: 'Vulkan0', name, busId: '0000:02:00.0' }, { device: 'Vulkan1', name, busId: '0000:01:00.0' }];
    expect(placeLlmPlan([{ index: 0, freeMb: 32207 }], twins, vk, busIds)).toEqual({
      plan: [{ index: 0, freeMb: 32207, device: 'Vulkan1', busId: '00000000:01:00.0' }],
      warnings: [],
    });
    // Paired by name, it is only a guess: a warning, and no bus id.
    const byName = vk.map((d) => ({ device: d.device, name }));
    const guess = placeLlmPlan([{ index: 0, freeMb: 32207 }, { index: 1, freeMb: 29707 }], twins, byName, busIds);
    expect(guess.plan).toEqual([
      { index: 0, freeMb: 32207, device: 'Vulkan0' },
      { index: 1, freeMb: 29707, device: 'Vulkan1' },
    ]);
  });
});

describe('describeLlmStart', () => {
  const { describeLlmStart } = require('../src/shared/llama');

  test('names the cards by nvidia-smi index', () => {
    expect(describeLlmStart([{ index: 0 }])).toBe('local LLM starting on 1 GPU [0]');
    expect(describeLlmStart([{ index: 0, device: 'Vulkan0' }, { index: 1, device: 'Vulkan1' }]))
      .toBe('local LLM starting on 2 GPUs [0, 1]');
    expect(describeLlmStart([{ index: null }])).toBe('local LLM starting on 1 GPU [auto]');
    expect(describeLlmStart(null)).toBe('local LLM starting on 0 GPUs []');
  });

  // So a log shows which card the model really went to.
  test('adds llama-server\'s names when it numbers the cards differently', () => {
    expect(describeLlmStart([{ index: 0, device: 'Vulkan1' }]))
      .toBe('local LLM starting on 1 GPU [0] (llama-server calls it Vulkan1)');
    expect(describeLlmStart([{ index: 0, device: 'Vulkan1' }, { index: 1 }]))
      .toBe('local LLM starting on 2 GPUs [0, 1] (llama-server calls them Vulkan1, ?)');
  });
});

describe('parseUsingDeviceLine', () => {
  const { parseUsingDeviceLine } = require('../src/shared/llama');

  test('reads the device and its bus id from the line b9902 prints', () => {
    expect(parseUsingDeviceLine('0.00.513.847 I llama_prepare_model_devices: using device Vulkan0 (NVIDIA GeForce RTX 4090) (0000:01:00.0) - 23370 MiB free'))
      .toEqual({ device: 'Vulkan0', busId: '0000:01:00.0' });
  });

  test('null without a bus id, and for every other line', () => {
    expect(parseUsingDeviceLine('x: using device Vulkan0 (Some GPU) (unknown id) - 8000 MiB free')).toBeNull();
    expect(parseUsingDeviceLine('load: 0 unused tokens')).toBeNull();
    expect(parseUsingDeviceLine(undefined)).toBeNull();
  });
});

describe('parseUsingDevice', () => {
  const { parseUsingDevice } = require('../src/shared/llama');

  test('reads the bus id of the card the pinned build loaded on', () => {
    expect(parseUsingDevice('llama_model_load_from_file_impl: using device Vulkan0 (NVIDIA GeForce RTX 3090) (0000:02:00.0) - 23893 MiB free'))
      .toBe('0000:02:00.0');
    // A name with brackets of its own.
    expect(parseUsingDevice('x: using device Vulkan1 (NVIDIA RTX PRO 4500 (Blackwell)) (0000:0a:00.0) - 31000 MiB free'))
      .toBe('0000:0a:00.0');
  });

  test('null when the backend does not know the bus id, and for every other line', () => {
    expect(parseUsingDevice('x: using device Vulkan0 (Some GPU) (unknown id) - 8000 MiB free')).toBeNull();
    expect(parseUsingDevice('srv  update_slots: all slots are idle')).toBeNull();
    expect(parseUsingDevice(null)).toBeNull();
  });
});
