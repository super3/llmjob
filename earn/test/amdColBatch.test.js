'use strict';

// The AMD host's batch width (amd_col_batch in earn/native/amd/src/pearl_host_hip.cpp): 128 column offsets on
// gfx1100 and gfx1101, where 128 was faster in the miner's own loop (earn/tuning/benchmark.md, AMD, item 13), and 256
// elsewhere. No HIP compiler runs here, so the function is read out of the source and run as JS: the same
// statements, with C's strncmp and getenv standing in. A build that sets PEARL_AMD_COL_BATCH keeps exactly its
// own cap, so a sweep can still race 256 against 128.

const fs = require('fs');
const path = require('path');

const HOST = fs.readFileSync(path.join(__dirname, '..', 'native', 'amd', 'src', 'pearl_host_hip.cpp'), 'utf8')
  .replace(/\r\n/g, '\n');
const README = fs.readFileSync(path.join(__dirname, '..', 'native', 'amd', 'README.md'), 'utf8').replace(/\r\n/g, '\n');

const body = (() => {
  const start = HOST.indexOf('uint32_t amd_col_batch(const PearlProfile *profile, const char *arch) {');
  expect(start).toBeGreaterThan(0);
  return HOST.slice(start, HOST.indexOf('\n}\n', start) + 3);
})();

// The function as JS, with PEARL_AMD_COL_BATCH and the by-arch switch as a build would set them.
function colBatch(profile, arch, { cap = 256, byArch = true, env } = {}) {
  let js = body
    .replace(/^uint32_t amd_col_batch\(.*\{\n/, '')
    .replace(/\n\}\n$/, '')
    .replace(/#if PEARL_AMD_COL_BATCH_BY_ARCH\n([\s\S]*?)#endif\n/, (m, inner) => (byArch ? inner : ''))
    .replace(/\/\/.*$/gm, '')
    .replace(/\(uint32_t\)/g, '')
    .replace(/const char \*/g, 'let ')
    .replace(/const uint32_t /g, 'const ')
    .replace(/uint32_t /g, 'let ')
    .replace(/(\d+)u\b/g, '$1')
    .replace(/PEARL_COLS_COUNT/g, '16')
    .replace(/PEARL_AMD_COL_BATCH(?!")/g, String(cap))
    .replace(/profile->/g, 'profile.')
    .replace(/for \(let p = env; p && \*p >= '0' && \*p <= '9' && want < limit; p\+\+\) want = want \* 10 \+ \(\*p - '0'\);/,
      "for (const ch of (env || '').match(/^\\d*/)[0]) { if (want >= limit) break; want = want * 10 + Number(ch); }");
  expect(js).not.toMatch(/\*p|->|uint32_t/);
  const strncmp = (a, b, n) => (a.slice(0, n) === b.slice(0, n) ? 0 : 1);
  // A C string ends in a NUL, which arch[7] reads on a 7-character name.
  const cArch = typeof arch === 'string' ? `${arch}\0` : arch;
  return new Function('profile', 'arch', 'getenv', 'strncmp', js)(profile, cArch, () => env, strncmp);
}

const MAINNET = { n: 262144, col_batch: 2048 };

describe('amd_col_batch', () => {
  it('caps gfx1100 and gfx1101 at 128 and every other card at 256', () => {
    for (const [arch, want] of [
      ['gfx1100', 128], ['gfx1100:sramecc-:xnack-', 128], ['gfx1101', 128],
      ['gfx1102', 256], ['gfx1150', 256], ['gfx1151', 256], ['gfx1200', 256], ['gfx1201', 256],
      ['gfx942:sramecc+:xnack-', 256], ['gfx1030', 256], ['gfx11000', 256], ['', 256], [null, 256],
    ]) expect([arch, colBatch(MAINNET, arch)]).toEqual([arch, want]);
  });

  it('keeps a build\'s own PEARL_AMD_COL_BATCH on every card, so 256 can still be raced on gfx1100', () => {
    expect(colBatch(MAINNET, 'gfx1100', { cap: 256, byArch: false })).toBe(256);
    expect(colBatch(MAINNET, 'gfx1100', { cap: 64, byArch: false })).toBe(64);
    expect(HOST).toContain('#ifndef PEARL_AMD_COL_BATCH\n#define PEARL_AMD_COL_BATCH 256u\n#define PEARL_AMD_COL_BATCH_BY_ARCH 1\n#else\n'
      + '#define PEARL_AMD_COL_BATCH_BY_ARCH 0\n#endif\n');
  });

  it('lets the environment lower the cap and never raise it', () => {
    expect(colBatch(MAINNET, 'gfx1100', { env: '64' })).toBe(64);
    expect(colBatch(MAINNET, 'gfx1100', { env: '256' })).toBe(128);
    expect(colBatch(MAINNET, 'gfx1201', { env: '128' })).toBe(128);
    expect(colBatch(MAINNET, 'gfx1201', { env: '4096' })).toBe(256);
  });

  it('still takes the profile\'s smaller width and halves to divide the valid offsets', () => {
    expect(colBatch({ n: 262144, col_batch: 64 }, 'gfx1100')).toBe(64);
    expect(colBatch({ n: 16 * 96, col_batch: 2048 }, 'gfx1100')).toBe(96);
    expect(colBatch({ n: 16 * 160, col_batch: 2048 }, 'gfx1100')).toBe(32);
  });

  it('gives each card its own width, and sizes the VRAM check for the widest', () => {
    expect(HOST).toContain('ctx->colBatch = amd_col_batch(profile, ctx->arch);');
    // The card's name is written before the width is picked.
    const named = HOST.indexOf('snprintf(ctx->arch, sizeof ctx->arch, "%s", prop.gcnArchName);');
    expect(named).toBeGreaterThan(0);
    expect(named).toBeLessThan(HOST.indexOf('ctx->colBatch = amd_col_batch(profile, ctx->arch);'));
    expect(HOST).toContain('(size_t)Ctx::kSlots * amd_col_batch(profile, nullptr)');
    expect(HOST.match(/amd_col_batch\(/g)).toHaveLength(3);
  });

  it('is documented where the AMD core is', () => {
    expect(README).toContain('128 on\ngfx1100 and gfx1101');
  });
});
