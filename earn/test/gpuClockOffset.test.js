'use strict';

const fs = require('fs');
const path = require('path');
const {
  clockOffsets, NO_OFFSETS, NOT_SUPPORTED, NVML_NO_PERMISSION,
} = require('../src/main/gpuClockOffset');

// An addon as the CUDA builds export it: getClockOffset and setClockOffset
// beside createCore.
function addon(over = {}) {
  return {
    createCore: jest.fn(),
    getClockOffset: jest.fn(() => ({ ok: true, mhz: 0, min: -1000, max: 1000 })),
    setClockOffset: jest.fn(() => ({ ok: true })),
    ...over,
  };
}
const of = (a) => clockOffsets(() => a);

describe('reading a card\'s core clock offset', () => {
  // What rig01's RTX 4090 answers: offset 0, allowed -1000 to 1000.
  test('asks the addon for that card and passes on the offset and its range', () => {
    const a = addon();
    expect(of(a).readCoreOffset(2)).toEqual({ ok: true, error: null, mhz: 0, min: -1000, max: 1000 });
    expect(a.getClockOffset).toHaveBeenCalledWith(2);
  });

  test('passes on what NVML said when it refuses', () => {
    const a = addon({ getClockOffset: () => ({ ok: false, error: 'reading the clock offset: Not Supported', code: 3 }) });
    expect(of(a).readCoreOffset(0)).toEqual({
      ok: false, error: 'reading the clock offset: Not Supported', noRights: false,
    });
  });

  test('an answer without whole numbers is no answer', () => {
    for (const bad of [{ ok: true }, { ok: true, mhz: 0, min: -1000 }, { ok: true, mhz: 0.5, min: 0, max: 100 }]) {
      expect(of(addon({ getClockOffset: () => bad })).readCoreOffset(0))
        .toEqual({ ok: false, error: 'the core gave no usable answer', noRights: false });
    }
  });
});

describe('setting a card\'s core clock offset', () => {
  test('asks the addon for that card and offset', () => {
    const a = addon();
    expect(of(a).setCoreOffset(1, 150)).toEqual({ ok: true, error: null });
    expect(a.setClockOffset).toHaveBeenCalledWith(1, 150);
    expect(of(a).setCoreOffset(1, 0)).toEqual({ ok: true, error: null });
    expect(a.setClockOffset).toHaveBeenLastCalledWith(1, 0);
  });

  // NVML_ERROR_NO_PERMISSION: the process is not root, or not an administrator.
  test('says when NVML refused for want of rights', () => {
    expect(NVML_NO_PERMISSION).toBe(4);
    const a = addon({
      setClockOffset: () => ({ ok: false, error: 'setting the clock offset: Insufficient Permissions', code: 4 }),
    });
    expect(of(a).setCoreOffset(0, 50)).toEqual({
      ok: false, error: 'setting the clock offset: Insufficient Permissions', noRights: true,
    });
  });

  test('any other refusal is not a matter of rights', () => {
    const a = addon({ setClockOffset: () => ({ ok: false, error: 'setting the clock offset: Invalid Argument', code: 2 }) });
    expect(of(a).setCoreOffset(0, 5000)).toEqual({
      ok: false, error: 'setting the clock offset: Invalid Argument', noRights: false,
    });
  });
});

describe('a core without the offset calls', () => {
  // The AMD core, and a pearl_core.node built before them.
  test('answers "not supported by this core"', () => {
    expect(NOT_SUPPORTED).toBe('not supported by this core');
    const none = { ok: false, error: 'not supported by this core', noRights: false };
    for (const a of [{ createCore: jest.fn() }, null, undefined, { getClockOffset: 'yes', setClockOffset: 1 }]) {
      expect(of(a).readCoreOffset(0)).toEqual(none);
      expect(of(a).setCoreOffset(0, 50)).toEqual(none);
    }
    expect(NO_OFFSETS.readCoreOffset(0)).toEqual(none);
    expect(NO_OFFSETS.setCoreOffset(0, 50)).toEqual(none);
  });
});

describe('nothing throws', () => {
  test('an addon that throws answers with what it threw', () => {
    const a = addon({
      getClockOffset: () => { throw new Error('boom'); },
      setClockOffset: () => { throw 'a string'; },
    });
    expect(of(a).readCoreOffset(0)).toEqual({ ok: false, error: 'boom', noRights: false });
    expect(of(a).setCoreOffset(0, 50)).toEqual({ ok: false, error: 'the core gave no answer', noRights: false });
  });

  test('an addon that answers nothing, or not ok without a reason, is a failure', () => {
    for (const answer of [undefined, null, {}, { ok: 'yes' }, { ok: false }]) {
      const a = addon({ getClockOffset: () => answer, setClockOffset: () => answer });
      expect(of(a).readCoreOffset(0)).toEqual({ ok: false, error: 'the core gave no answer', noRights: false });
      expect(of(a).setCoreOffset(0, 50)).toEqual({ ok: false, error: 'the core gave no answer', noRights: false });
    }
  });
});

// pearlCore's fallbacks change which addon a rig's cores come from after its
// first card opens, so the addon is asked for on every call.
test('calls whichever addon is current at the time', () => {
  const first = addon();
  const second = addon({ getClockOffset: jest.fn(() => ({ ok: true, mhz: 50, min: -500, max: 500 })) });
  let current = first;
  const offsets = clockOffsets(() => current);
  expect(offsets.readCoreOffset(0).mhz).toBe(0);
  current = second;
  expect(offsets.readCoreOffset(0).mhz).toBe(50);
  offsets.setCoreOffset(0, 100);
  expect(first.setClockOffset).not.toHaveBeenCalled();
  expect(second.setClockOffset).toHaveBeenCalledWith(0, 100);
});

// The native half cannot run here. What can be checked is what the brief for it
// rests on: the struct NVML is handed, that NVML is found at run time and not
// linked, and that only the CUDA builds have the code at all.
describe('the addon\'s side (native/src/pearl_core.cc)', () => {
  const NATIVE = path.join(__dirname, '..', 'native');
  const read = (...p) => fs.readFileSync(path.join(NATIVE, ...p), 'utf8').replace(/\r\n/g, '\n');
  const CORE = read('src', 'pearl_core.cc');
  const GYP = read('binding.gyp');
  const start = CORE.indexOf('#ifdef PEARL_NVML_OFFSETS\n// ---- The core clock offset');
  const NVML = CORE.slice(start, CORE.indexOf('#endif  // PEARL_NVML_OFFSETS'));

  // The file's code lines that are under no `#ifdef PEARL_NVML_OFFSETS`.
  function outsideTheGuard(src) {
    const out = [];
    let depth = 0;
    for (const line of src.split('\n')) {
      const l = line.trim();
      if (depth) {
        if (/^#\s*if/.test(l)) depth += 1;
        else if (/^#\s*endif/.test(l)) depth -= 1;
      } else if (l === '#ifdef PEARL_NVML_OFFSETS') {
        depth = 1;
      } else if (!l.startsWith('//')) {
        out.push(line);
      }
    }
    return out.join('\n');
  }

  test('is all inside PEARL_NVML_OFFSETS', () => {
    expect(start).toBeGreaterThan(0);
    expect(NVML).toContain('namespace nvml {');
    expect(NVML).toContain('}  // namespace nvml');
    // Outside the guards the file names neither the library nor the exports,
    // and still has the rest of the binding.
    const outside = outsideTheGuard(CORE);
    for (const word of ['nvml', 'ClockOffset', 'dlopen', 'LoadLibrary', 'windows.h', 'dlfcn.h']) {
      expect([word, outside.includes(word)]).toEqual([word, false]);
    }
    expect(outside).toContain('exports.Set("createCore", Napi::Function::New(env, CreateCore));');
    expect(outside).toContain('NODE_API_MODULE(pearl_core, InitAll)');
  });

  test('exports getClockOffset and setClockOffset beside createCore', () => {
    expect(CORE).toContain([
      '  exports.Set("createCore", Napi::Function::New(env, CreateCore));',
      '#ifdef PEARL_NVML_OFFSETS',
      '  exports.Set("getClockOffset", Napi::Function::New(env, nvml::GetClockOffset));',
      '  exports.Set("setClockOffset", Napi::Function::New(env, nvml::SetClockOffset));',
      '#endif',
    ].join('\n'));
  });

  // nvmlClockOffset_v1_t in NVIDIA's nvml.h: version, type, pstate, then the
  // offset and its range, six 4-byte fields. Its version constant is the size
  // with the struct version in the top byte, 0x1000018.
  test('hands NVML nvmlClockOffset_v1_t, for the graphics clock on P-state 0', () => {
    expect(NVML).toContain([
      'struct ClockOffset {',
      '  unsigned int version;',
      '  int type;',
      '  int pstate;',
      '  int clockOffsetMHz;',
      '  int minClockOffsetMHz;',
      '  int maxClockOffsetMHz;',
      '};',
    ].join('\n'));
    expect(NVML).toContain('const unsigned int kClockOffsetV1 = (unsigned int)(sizeof(ClockOffset) | (1U << 24U));');
    expect(6 * 4 | (1 << 24)).toBe(0x1000018);
    expect(NVML).toContain('const int kClockGraphics = 0;');
    expect(NVML).toContain('const int kPstate0 = 0;');
    expect(NVML).toContain([
      '  info.version = kClockOffsetV1;',
      '  info.type = kClockGraphics;',
      '  info.pstate = kPstate0;',
    ].join('\n'));
  });

  test('opens NVML at run time and finds its functions by name', () => {
    expect(NVML).toContain('LoadLibraryExA("nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32)');
    expect(NVML).toContain('dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL)');
    for (const name of ['nvmlInit_v2', 'nvmlShutdown', 'nvmlDeviceGetHandleByIndex_v2',
      'nvmlDeviceGetClockOffsets', 'nvmlDeviceSetClockOffsets', 'nvmlErrorString']) {
      expect([name, NVML.includes('find("' + name + '")')]).toEqual([name, true]);
    }
    // Nothing links it: no build names an NVML library.
    for (const f of [GYP, read('build-local.ps1'), read('amd', 'build-amd.sh')]) {
      expect(/nvml\.lib|nvidia-ml/i.test(f)).toBe(false);
    }
  });

  test('never throws into JS: every argument is checked before it is read', () => {
    expect(NVML).toContain('if (info.Length() < 1 || !info[0].IsNumber()) return -1;');
    expect(NVML).toContain('const bool given = info.Length() > 1 && info[1].IsNumber();');
    expect(NVML).not.toContain('ThrowAsJavaScriptException');
  });

  // binding.gyp builds the CUDA core on Windows and Linux, and the AMD and
  // Intel cores on Windows. The AMD core on Linux is compiled by build-amd.sh,
  // and the Intel one by sycl/build.sh; all compile this same pearl_core.cc.
  test('is compiled into the CUDA builds only', () => {
    const conditions = GYP.split('\n').filter((l) => /^\s*\["OS==/.test(l)).map((l) => l.trim());
    expect(conditions).toEqual([
      '["OS==\'win\' and pearl_backend==\'cuda\'", {',
      '["OS==\'win\' and pearl_backend==\'hip\'", {',
      '["OS==\'win\' and pearl_backend==\'sycl\'", {',
      '["OS==\'linux\'", {',
    ]);
    const blocks = GYP.split(/\n\s*\["OS==/).slice(1);
    expect(blocks.map((b) => b.includes('"defines": [ "PEARL_NVML_OFFSETS", "PEARL_HOST_ROT" ]'))).toEqual([true, false, false, true]);
    expect(blocks[3]).toContain('-lpearl_cuda');
    expect(read('amd', 'build-amd.sh')).not.toContain('PEARL_NVML_OFFSETS');
    expect(read('sycl', 'build.sh')).toContain('-c "$SRC/pearl_core.cc"');
    expect(read('sycl', 'build.sh')).not.toContain('PEARL_NVML_OFFSETS');
  });
});
