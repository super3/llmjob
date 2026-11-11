'use strict';

// PEARL_AMD_SCHED (earn/native/amd/src/pearl_host_hip.cpp): an environment switch for measuring how HIP's threads
// wait for the card (hipSetDeviceFlags). No HIP compiler runs here, so the C is read as text: that nothing is called
// unless the variable names a schedule, that it is never part of the CPU build, and that it comes before the first
// allocation, copy or stream.

const fs = require('fs');
const path = require('path');

const HOST = fs.readFileSync(path.join(__dirname, '..', 'native', 'amd', 'src', 'pearl_host_hip.cpp'), 'utf8')
  .replace(/\r\n/g, '\n');

function slice(s, from, to) {
  const a = s.indexOf(from);
  expect(a).toBeGreaterThanOrEqual(0);
  const b = s.indexOf(to, a + from.length);
  expect(b).toBeGreaterThan(a);
  return s.slice(a, b + to.length);
}

describe('the wait schedule switch (PEARL_AMD_SCHED)', () => {
  const create = slice(HOST, 'extern "C" void *pearl_host_create_checked(', '\n}\n');
  const block = slice(create, 'const char *sc = getenv("PEARL_AMD_SCHED");', '(void)hipGetLastError();\n    }\n  }\n');

  it('calls hipSetDeviceFlags only for a schedule it knows, and leaves the default alone otherwise', () => {
    expect(block).toContain('if (sc && !strcmp(sc, "block")) flag = hipDeviceScheduleBlockingSync;');
    expect(block).toContain('else if (sc && !strcmp(sc, "spin")) flag = hipDeviceScheduleSpin;');
    expect(block).toContain('else if (sc && !strcmp(sc, "yield")) flag = hipDeviceScheduleYield;');
    expect(block).toContain('else if (sc && !strcmp(sc, "auto")) flag = hipDeviceScheduleAuto;');
    expect(block).toContain('else set = false;');
    expect(block).toContain('if (set) {\n      const hipError_t e = hipSetDeviceFlags(flag);');
    // The one call in the file, and no build-time default: the environment is the only way in.
    expect(HOST.match(/hipSetDeviceFlags\(/g)).toHaveLength(1);
    expect(HOST).not.toMatch(/#define PEARL_AMD_SCHED/);
    // The schedules the C accepts, read as it reads them.
    const pick = (env) => ({ block: 'BlockingSync', spin: 'Spin', yield: 'Yield', auto: 'Auto' })[env];
    expect([undefined, '', '1', 'Block', 'block', 'spin', 'yield', 'auto'].map((e) => pick(e) || null))
      .toEqual([null, null, null, null, 'BlockingSync', 'Spin', 'Yield', 'Auto']);
  });

  it('says on stderr what the runtime answered, and clears the error', () => {
    expect(block).toContain('fprintf(stderr, "[pearl-amd] PEARL_AMD_SCHED=%s: hipSetDeviceFlags %s\\n", sc,');
    expect(block).toContain('e == hipSuccess ? "ok" : hipGetErrorString(e));');
    expect(block.endsWith('(void)hipGetLastError();\n    }\n  }\n')).toBe(true);
  });

  it('runs on a GPU only, once the device is known and before anything is allocated, copied or queued', () => {
    const at = create.indexOf('const char *sc = getenv("PEARL_AMD_SCHED");');
    const guard = create.lastIndexOf('#if !defined(__HIP_CPU_RT__)', at);
    expect(guard).toBeGreaterThan(0);
    expect(create.slice(guard, at)).not.toContain('#endif');
    expect(at).toBeGreaterThan(create.indexOf('if (hipGetDevice(&ctx->device) != hipSuccess) ctx->device = 0;'));
    for (const first of ['hipMalloc(', 'hipMemcpy(', 'hipStreamCreateWithFlags(', 'hipEventCreateWithFlags(']) {
      expect(create.indexOf(first)).toBeGreaterThan(at);
    }
  });
});
