'use strict';

// PEARL_AMD_TICK_US (earn/native/amd/src/pearl_host_hip.cpp): a thread that launches a one-thread kernel on a queue
// of its own every so many microseconds. No HIP compiler runs here, so the C is read as text: that it is off unless
// asked for, never part of the CPU build, touches nothing the search uses, and is stopped before anything it uses is
// freed.

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

// The build's default: 0 is no ticker.
const DEFAULT_US = 0;

describe('the ticker (PEARL_AMD_TICK_US)', () => {
  const create = slice(HOST, 'extern "C" void *pearl_host_create_checked(', '\n}\n');
  const start = create.slice(create.lastIndexOf('#if !defined(__HIP_CPU_RT__)   // a GPU only'));

  it('is a build default the environment overrides, and only 100 to 100000 microseconds starts one', () => {
    expect(HOST).toContain('#ifndef PEARL_AMD_TICK_US\n#define PEARL_AMD_TICK_US ' + DEFAULT_US + '\n#endif\n');
    expect(start).toContain('uint32_t us = PEARL_AMD_TICK_US;');
    expect(start).toContain('getenv("PEARL_AMD_TICK_US")');
    expect(start).toContain('if (us >= 100u && us <= 100000u) {');
    // The environment's number, read as the C reads it.
    const read = (env, build = DEFAULT_US) => {
      let us = build;
      if (env && env[0] >= '0' && env[0] <= '9') {
        us = 0;
        for (const ch of env) {
          if (!(ch >= '0' && ch <= '9' && us <= 100000)) break;
          us = us * 10 + Number(ch);
        }
      }
      return us >= 100 && us <= 100000 ? us : 0;
    };
    expect(start).toContain("if (env && env[0] >= '0' && env[0] <= '9') {\n      us = 0;\n"
      + "      for (const char *p = env; *p >= '0' && *p <= '9' && us <= 100000u; p++) us = us * 10u + (uint32_t)(*p - '0');");
    expect([read(undefined), read(''), read('0'), read('50'), read('100'), read('2000'), read('100000'), read('100001'),
      read('9999999999'), read('fast'), read('0', 2000), read(undefined, 2000), read('x', 2000)])
      .toEqual([0, 0, 0, 0, 100, 2000, 100000, 0, 0, 0, 0, 2000, 2000]);
  });

  it('runs on a GPU only, on a queue and two words of its own, once the card\'s fold is settled', () => {
    expect(start.startsWith('#if !defined(__HIP_CPU_RT__)')).toBe(true);
    expect(create.indexOf(start)).toBeGreaterThan(create.indexOf('  resolve_fold(ctx);'));
    expect(start).toContain('HIP_OK(hipStreamCreateWithFlags(&ctx->tickq, hipStreamDefault), "creating the ticker\'s stream");');
    expect(start).toContain('HIP_OK(hipMalloc(&ctx->dTick, 2 * sizeof(uint32_t)), "allocating the ticker\'s words");');
    const loop = slice(start, 'ctx->ticker = std::thread([c] {', '});');
    expect(loop).toContain('hipSetDevice(c->device);');
    expect(loop).toContain('while (!c->tickStop.load(std::memory_order_relaxed)) {');
    expect(loop).toContain('hipLaunchKernelGGL(pearl_amd_caps, dim3(1), dim3(1), 0, c->tickq, c->dTick);');
    expect(loop).toContain('(void)hipStreamSynchronize(c->tickq);');
    expect(loop).toContain('std::this_thread::sleep_for(std::chrono::microseconds(c->tickUs));');
    // Nothing of the search's: not its queues, not the caps words resolve_fold reads.
    expect(loop).not.toMatch(/c->(work|hashq|side|dCaps|dTr|dHit)/);
  });

  it('can run without waiting for its kernel, and then never has two ticks on the queue (PEARL_AMD_TICK_WAIT)', () => {
    expect(HOST).toContain('#ifndef PEARL_AMD_TICK_WAIT\n#define PEARL_AMD_TICK_WAIT 1\n#endif\n');
    expect(start).toContain('ctx->tickWait = PEARL_AMD_TICK_WAIT != 0;');
    expect(start).toContain('const char *tw = getenv("PEARL_AMD_TICK_WAIT");\n'
      + '      if (tw && (!strcmp(tw, "0") || !strcmp(tw, "1"))) ctx->tickWait = tw[0] == \'1\';');
    // The event is made before the thread that records it starts.
    const made = start.indexOf('HIP_OK(hipEventCreateWithFlags(&ctx->tickDone, hipEventDisableTiming), "creating the ticker\'s event");');
    expect(made).toBeGreaterThan(0);
    expect(made).toBeLessThan(start.indexOf('ctx->ticker = std::thread([c] {'));
    const loop = slice(start, 'ctx->ticker = std::thread([c] {', '});');
    // With the wait: the launch, then the wait. Without: a launch only once the last tick's event is done.
    const wait = slice(loop, 'if (c->tickWait) {', '} else {');
    expect(wait).toContain('hipLaunchKernelGGL(pearl_amd_caps,');
    expect(wait).toContain('(void)hipStreamSynchronize(c->tickq);');
    const free = loop.slice(loop.indexOf('} else {'));
    const seen = free.indexOf('if (out && hipEventQuery(c->tickDone) == hipSuccess) out = false;');
    const gate = free.indexOf('if (!out) {');
    const launch = free.indexOf('hipLaunchKernelGGL(pearl_amd_caps,');
    const record = free.indexOf('if (hipEventRecord(c->tickDone, c->tickq) == hipSuccess) out = true;');
    expect(seen).toBeGreaterThan(0);
    expect(gate).toBeGreaterThan(seen);
    expect(launch).toBeGreaterThan(gate);
    expect(record).toBeGreaterThan(launch);
    // A tick not seen done after 50 ms is waited for, so an event that never reads done can't end the ticking.
    const stuck = free.indexOf('if (out && ++late * c->tickUs >= 50000u) {\n              (void)hipStreamSynchronize(c->tickq);\n'
      + '              out = false;');
    expect(stuck).toBeGreaterThan(seen);
    expect(stuck).toBeLessThan(gate);
    expect(free.slice(gate, launch)).toContain('late = 0;');
    // An event that would not record: wait for the kernel instead, so launches can't pile up.
    expect(free.slice(record)).toContain('else (void)hipStreamSynchronize(c->tickq);');
    expect(loop.match(/hipLaunchKernelGGL\(/g)).toHaveLength(2);
    // The event goes after the thread has stopped.
    const destroy = slice(HOST, 'extern "C" void pearl_host_destroy(void *handle) {', '\n}\n');
    expect(destroy.indexOf('if (ctx->tickDone) hipEventDestroy(ctx->tickDone);'))
      .toBeGreaterThan(destroy.indexOf('if (ctx->ticker.joinable()) ctx->ticker.join();'));
    expect(HOST).toContain('  bool tickWait = true;');
    expect(HOST).toContain('  hipEvent_t tickDone = nullptr;');
  });

  it('is stopped and joined before the device is waited for and anything is freed', () => {
    const destroy = slice(HOST, 'extern "C" void pearl_host_destroy(void *handle) {', '\n}\n');
    const stop = destroy.indexOf('ctx->tickStop.store(true);\n  if (ctx->ticker.joinable()) ctx->ticker.join();');
    expect(stop).toBeGreaterThan(0);
    expect(stop).toBeLessThan(destroy.indexOf('hipDeviceSynchronize();'));
    expect(destroy.indexOf('if (ctx->tickq) hipStreamDestroy(ctx->tickq);')).toBeGreaterThan(stop);
    expect(destroy.indexOf('hipFree(ctx->dTick);')).toBeGreaterThan(stop);
    // A context that never started one has nothing to join.
    expect(HOST).toContain('  std::thread ticker;\n  std::atomic<bool> tickStop{false};\n  hipStream_t tickq = nullptr;\n'
      + '  uint32_t *dTick = nullptr;\n  uint32_t tickUs = 0;\n');
  });
});
