'use strict';

// PEARL_AMD_FOLDQ2 (earn/native/amd/src/pearl_host_hip.cpp, goal-rdna3a, 15 November): pipeline slot 1's fold on a
// second queue, so one batch's fold starts while the last blocks of the one before still run (2, the default, is
// gfx1101 only). These check the source: where it is on, that slot 1's fold waits for the draws, and that the
// draws wait for slot 1's fold, so no fold reads operands a redraw is rewriting.
const fs = require('fs');
const path = require('path');

const HOST = fs.readFileSync(path.join(__dirname, '../native/amd/src/pearl_host_hip.cpp'), 'utf8').replace(/\r\n/g, '\n');

function slice(s, from, to) {
  const a = s.indexOf(from);
  expect(a).toBeGreaterThanOrEqual(0);
  const b = s.indexOf(to, a + from.length);
  expect(b).toBeGreaterThan(a);
  return s.slice(a, b);
}

describe('slot 1\'s fold on a second queue (PEARL_AMD_FOLDQ2)', () => {
  it('is on for gfx1101 only by default (2), and 0 or 1 in the environment overrides the build', () => {
    expect(HOST).toContain('#ifndef PEARL_AMD_FOLDQ2\n#define PEARL_AMD_FOLDQ2 2\n#endif\n');
    const create = slice(HOST, '{   // PEARL_AMD_FOLDQ2 (above)', '\n  }\n');
    expect(create).toContain('PEARL_AMD_FOLDQ2 == 1 || (PEARL_AMD_FOLDQ2 == 2 && !strncmp(ctx->arch, "gfx1101", 7))');
    expect(create).toContain('getenv("PEARL_AMD_FOLDQ2")');
    expect(create).toContain('hipStreamCreateWithFlags(&ctx->work2, hipStreamDefault)');
    // A GPU only: HIP-CPU's streams stay as they are.
    const gpu = slice(HOST, '#if !defined(__HIP_CPU_RT__)   // a GPU only', '#endif');
    expect(gpu).toContain('{   // PEARL_AMD_FOLDQ2 (above)');
  });

  it('runs slot 1\'s fold on work2, after the last draws, and marks it done there', () => {
    const submit = slice(HOST, 'extern "C" bool pearl_host_submit(', '\n}\n');
    const pick = submit.indexOf('if (ctx->work2 && slot == 1) {');
    expect(pick).toBeGreaterThan(0);
    expect(submit).toContain('st = ctx->work2;\n    hipStreamWaitEvent(st, ctx->workMark, 0);');
    expect(pick).toBeLessThan(submit.indexOf('hipMemsetAsync(hitList.count, 0, sizeof(uint32_t), st);'));
    expect(pick).toBeLessThan(submit.indexOf('switch (f.cap)'));
    const done = submit.indexOf('hipEventRecord(ctx->work2Done, foldSt);');
    expect(done).toBeGreaterThan(submit.indexOf('switch (f.cap)'));
  });

  it('makes reseed\'s draws wait for work2, and marks them for work2\'s next fold', () => {
    const reseed = slice(HOST, 'extern "C" void pearl_host_reseed(void *handle, uint64_t salt) {', '\n}\n');
    const wait = reseed.indexOf('if (ctx->work2Busy) hipStreamWaitEvent(ctx->work, ctx->work2Done, 0);');
    expect(wait).toBeGreaterThan(0);
    expect(wait).toBeLessThan(reseed.indexOf('ctx->salt = salt;'));
    const mark = reseed.indexOf('if (ctx->work2) hipEventRecord(ctx->workMark, ctx->work);');
    expect(mark).toBeGreaterThan(reseed.indexOf('ctx->baseDrawn = true;'));
  });

  it('frees what it made', () => {
    const destroy = slice(HOST, 'extern "C" void pearl_host_destroy(void *handle) {', '\n}\n');
    expect(destroy).toContain('if (ctx->work2) hipStreamDestroy(ctx->work2);');
    expect(destroy).toContain('if (ctx->workMark) hipEventDestroy(ctx->workMark);');
    expect(destroy).toContain('if (ctx->work2Done) hipEventDestroy(ctx->work2Done);');
    expect(destroy.indexOf('hipDeviceSynchronize();')).toBeLessThan(destroy.indexOf('hipStreamDestroy(ctx->work2)'));
  });
});
