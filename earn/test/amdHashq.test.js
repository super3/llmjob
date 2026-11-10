// PEARL_AMD_HASHQ (earn/native/amd/src/pearl_host_hip.cpp, PR #290 research item 23): the hash kernel on a queue of
// its own (off by default; 2 is RX 7900 only), so the next batch's fold follows this one with no hash between them.
// These check the source: where it is on, and that a slot's fold can't rewrite transcripts or a hit list its last
// hash still reads.
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

describe('the hash on its own queue (PEARL_AMD_HASHQ)', () => {
  it('is off by default, 2 turns it on for gfx1100 only, and 0 or 1 in the environment overrides the build', () => {
    expect(HOST).toContain('#ifndef PEARL_AMD_HASHQ\n#define PEARL_AMD_HASHQ 0\n#endif\n');
    const create = slice(HOST, '#if !defined(__HIP_CPU_RT__)   // a GPU only', '#endif');
    expect(create).toContain('PEARL_AMD_HASHQ == 1 || (PEARL_AMD_HASHQ == 2 && !strncmp(ctx->arch, "gfx1100", 7))');
    expect(create).toContain('getenv("PEARL_AMD_HASHQ")');
    expect(create).toContain('hipStreamCreateWithFlags(&ctx->hashq, hipStreamDefault)');
    // The card's arch is known by then.
    expect(HOST.indexOf('snprintf(ctx->arch, sizeof ctx->arch, "%s", prop.gcnArchName);'))
      .toBeLessThan(HOST.indexOf('hipStreamCreateWithFlags(&ctx->hashq'));
  });

  it('waits for the slot\'s last hash before the fold rewrites its transcripts and hit list', () => {
    const submit = slice(HOST, 'extern "C" bool pearl_host_submit(', '\n}\n');
    const wait = submit.indexOf('if (hashq) hipStreamWaitEvent(st, ctx->slotDone[slot], 0);');
    expect(wait).toBeGreaterThan(0);
    expect(wait).toBeLessThan(submit.indexOf('hipMemsetAsync(hitList.count, 0, sizeof(uint32_t), st);'));
    expect(wait).toBeLessThan(submit.indexOf('switch (f.cap)'));
  });

  it('runs the hash, the count copy and the slot event on the hash queue, after the fold', () => {
    const submit = slice(HOST, 'extern "C" bool pearl_host_submit(', '\n}\n');
    const move = submit.indexOf('hipEventRecord(ctx->foldDone[slot], st);\n    st = ctx->hashq;\n'
      + '    hipStreamWaitEvent(st, ctx->foldDone[slot], 0);');
    expect(move).toBeGreaterThan(submit.indexOf('switch (f.cap)'));
    expect(move).toBeLessThan(submit.indexOf('hipLaunchKernelGGL(pearl_amd_hash,'));
    expect(move).toBeLessThan(submit.indexOf('hipEventRecord(ctx->slotDone[slot], st);'));
    // Not when the fold hashes its own transcripts.
    expect(submit).toContain('const bool hashq = ctx->hashq && !fusedQ;');
  });

  it('frees what it made', () => {
    const destroy = slice(HOST, 'extern "C" void pearl_host_destroy(void *handle) {', '\n}\n');
    expect(destroy).toContain('if (ctx->foldDone[i]) hipEventDestroy(ctx->foldDone[i]);');
    expect(destroy).toContain('if (ctx->hashq) hipStreamDestroy(ctx->hashq);');
    expect(destroy.indexOf('hipDeviceSynchronize();')).toBeLessThan(destroy.indexOf('hipStreamDestroy(ctx->hashq)'));
  });
});
