'use strict';

// PEARL_AMD_STRIP (earn/native/amd/src/pearl_amd_fold.h) and PEARL_AMD_STRIPS (pearl_host_hip.cpp): one launch folds
// several batches of the card's width side by side, as strips of the grid's z, so there are fewer launches a second
// and the B' in use at any time is still one batch's. No HIP compiler runs here, so the C is read as text: where it
// is on, that a block's tile follows its strip and nothing else does, and, run as JS, that every tile of every launch
// is folded once and that the host picks a count the batch can take.

const fs = require('fs');
const path = require('path');
const { preprocess } = require('./helpers/amdSwitches');

const NATIVE = path.join(__dirname, '..', 'native');
const read = (...f) => fs.readFileSync(path.join(NATIVE, ...f), 'utf8').replace(/\r\n/g, '\n');
const FOLD = read('amd', 'src', 'pearl_amd_fold.h');
const HOST = read('amd', 'src', 'pearl_host_hip.cpp');

function slice(s, from, to) {
  const a = s.indexOf(from);
  expect(a).toBeGreaterThanOrEqual(0);
  const b = s.indexOf(to, a + from.length);
  expect(b).toBeGreaterThan(a);
  return s.slice(a, b + to.length);
}

// The C without its comments.
const code = (s) => s.replace(/\/\/.*$/gm, '');

describe('the fold in a launch of several strips (PEARL_AMD_STRIP)', () => {
  it('is a build switch, off unless given, and never on with the walk', () => {
    expect(FOLD).toContain('#ifndef PEARL_AMD_STRIP\n#define PEARL_AMD_STRIP 0\n#endif\n');
    expect(FOLD).toContain('#if PEARL_AMD_STRIP != 0 && PEARL_AMD_STRIP != 1\n#error "PEARL_AMD_STRIP is 0 or 1"\n#endif\n');
    expect(FOLD).toContain('#define PEARL_AMD_STRIP_ON (PEARL_AMD_STRIP != 0 && !PEARL_AMD_WALK_ON)\n');
    // The walk is settled first: the strips' switch reads it.
    expect(FOLD.indexOf('#define PEARL_AMD_WALK_ON (')).toBeLessThan(FOLD.indexOf('#define PEARL_AMD_STRIP_ON ('));
  });

  it('moves a block\'s columns on by its strip, and leaves a build without it as it was', () => {
    const on = preprocess(FOLD, { PEARL_AMD_WALK_ON: 0, PEARL_AMD_STRIP_ON: 1 });
    expect(on).toContain('  const uint32_t tileRow = blockIdx.y * (uint32_t)BM;\n'
      + '  const uint32_t tileCol = (blockIdx.z * gridDim.x + blockIdx.x) * (uint32_t)BN;\n');
    const off = preprocess(FOLD, { PEARL_AMD_WALK_ON: 0, PEARL_AMD_STRIP_ON: 0 });
    expect(off).toContain('  const uint32_t tileRow = blockIdx.y * (uint32_t)BM;   // first A\' row of the tile\n'
      + '  const uint32_t tileCol = blockIdx.x * (uint32_t)BN;   // first column, within the batch\n');
    expect(code(off)).not.toContain('blockIdx.z');
    // With the walk, the walk's tile: no strips there.
    const walk = preprocess(FOLD, { PEARL_AMD_WALK_ON: 1, PEARL_AMD_STRIP_ON: 0 });
    expect(code(walk)).not.toContain('blockIdx.z');
    // blockIdx.z is read once, for the tile; the operands and the transcript's place follow the tile.
    expect(code(on).match(/blockIdx\.z/g)).toHaveLength(1);
    expect(on).toContain('  const int8_t *gB = Bp + (size_t)(col0 + tileCol) * k;\n');
    expect(on).toContain('const uint32_t colIdx = 4u * ((tileCol + (uint32_t)(wc * WN + cb * 64)) / 64u) + coIdx;');
  });

  it('folds every tile of a launch once, strip after strip', () => {
    // The tile of block (x, y, z) of a gx x gy x gz launch, as the fold computes it, in launch order (x fastest, z
    // slowest): every tile of the gx * gz column tiles by gy row tiles once, and a strip's tiles before the next's.
    const line = /const uint32_t tileCol = \((blockIdx\.z \* gridDim\.x \+ blockIdx\.x)\) \* \(uint32_t\)BN;/.exec(FOLD);
    expect(line).not.toBeNull();
    const col = new Function('blockIdx', 'gridDim', `return ${line[1]};`);
    for (const [gx, gy, gz] of [[8, 1024, 4], [8, 1024, 1], [8, 3, 2], [1, 5, 16], [4, 7, 8]]) {
      const seen = new Set();
      let last = -1;
      for (let z = 0; z < gz; z++) {
        for (let y = 0; y < gy; y++) {
          for (let x = 0; x < gx; x++) {
            const c = col({ x, y, z }, { x: gx, y: gy, z: gz });
            expect(c >= 0 && c < gx * gz).toBe(true);
            seen.add(y * gx * gz + c);
            const strip = Math.floor(c / gx);
            expect(strip >= last).toBe(true);
            last = strip;
          }
        }
      }
      expect([gx, gy, gz, seen.size]).toEqual([gx, gy, gz, gx * gy * gz]);
    }
  });

  it('says so in the card\'s caps, which is how the host knows', () => {
    expect(FOLD).toContain('#define PEARL_AMD_CAP_STRIP 0x8000u');
    expect(slice(FOLD, 'extern "C" __global__ void pearl_amd_caps(uint32_t *out) {', '\n}\n'))
      .toContain('  if (PEARL_AMD_STRIP_ON) out[0] |= PEARL_AMD_CAP_STRIP;\n');
    // No other cap has the bit.
    const bits = [...FOLD.matchAll(/^#define PEARL_AMD_CAP_\w+ (0x[0-9a-f]+|\d+)u/gm)].map((m) => Number(m[1]));
    expect(new Set(bits).size).toBe(bits.length);
    expect(slice(HOST, 'void resolve_fold(Ctx *ctx) {', '\n}\n'))
      .toContain('  ctx->stripCap = (caps[0] & PEARL_AMD_CAP_STRIP) != 0;\n');
  });
});

// amd_strips as JS: the same statements, with C's getenv standing in.
function strips(ctx, { build = 1, env, folds } = {}) {
  const body = slice(HOST, 'uint32_t amd_strips(const Ctx *ctx) {', '\n}\n');
  const js = body
    .replace(/^uint32_t amd_strips\(.*\{\n/, '')
    .replace(/\n\}\n$/, '')
    .replace(/\/\/.*$/gm, '')
    .replace(/\(uint32_t\)/g, '')
    .replace(/const char \*/g, 'let ')
    .replace(/uint32_t /g, 'let ')
    .replace(/(\d+)u\b/g, '$1')
    .replace(/PEARL_COLS_COUNT/g, '16')
    .replace(/PEARL_AMD_CAP_REF/g, '1')
    .replace(/PEARL_AMD_STRIPS(?!")/g, String(build))
    .replace(/ctx->/g, 'ctx.')
    .replace(/for \(let p = env; p && \*p >= '0' && \*p <= '9' && want <= 16; p\+\+\) want = want \* 10 \+ \(\*p - '0'\);/,
      "for (const ch of (env || '').match(/^\\d*/)[0]) { if (want > 16) break; want = want * 10 + Number(ch); }");
  expect(js).not.toMatch(/\*p|->|uint32_t/);
  return new Function('ctx', 'kFolds', 'getenv', js)(ctx, folds || [{ cap: 1, bn: 64 }, { cap: 8, bn: 256 }], () => env);
}
// An RX 7900 at the mainnet profile: 16,384 valid column offsets, a batch of 128, the wmma11 fold.
const XTX = { stripCap: true, fold: 1, stripBatch: 128, colsValid: 16384 };

describe('the host\'s strips (PEARL_AMD_STRIPS)', () => {
  it('launches one strip unless the build or the environment asks for more', () => {
    expect(HOST).toContain('#ifndef PEARL_AMD_STRIPS\n#define PEARL_AMD_STRIPS 1\n#endif\n');
    expect(HOST).toContain('#if PEARL_AMD_STRIPS < 1 || PEARL_AMD_STRIPS > 16\n#error "PEARL_AMD_STRIPS is 1 to 16"\n#endif\n');
    expect(strips(XTX)).toBe(1);
    expect(strips(XTX, { build: 4 })).toBe(4);
    expect(strips(XTX, { env: '4' })).toBe(4);
    expect(strips(XTX, { build: 4, env: '1' })).toBe(1);
    expect(strips(XTX, { env: '16' })).toBe(16);
  });

  it('only on a card whose code takes strips, and never for the reference fold', () => {
    expect(strips({ ...XTX, stripCap: false }, { env: '4' })).toBe(1);
    expect(strips({ ...XTX, fold: 0 }, { env: '4' })).toBe(1);
    expect(strips({ ...XTX, fold: -1 }, { env: '4' })).toBe(1);
    // A strip narrower than a tile of the fold (256 columns, 16 offsets) can't be a strip.
    expect(strips({ ...XTX, stripBatch: 8 }, { env: '4' })).toBe(1);
    expect(strips({ ...XTX, stripBatch: 16 }, { env: '4' })).toBe(4);
  });

  it('takes a power of two that divides the valid offsets, and ignores anything else in the environment', () => {
    expect(strips(XTX, { env: '3' })).toBe(2);
    expect(strips(XTX, { env: '7' })).toBe(4);
    expect(strips(XTX, { env: '0' })).toBe(1);
    expect(strips(XTX, { env: '17' })).toBe(1);
    expect(strips(XTX, { env: '160' })).toBe(1);
    expect(strips(XTX, { env: 'four' })).toBe(1);
    expect(strips(XTX, { build: 8, env: '99' })).toBe(8);
    // 96 valid offsets at a batch of 32: 3 batches, so no pair of them is a launch.
    expect(strips({ ...XTX, stripBatch: 32, colsValid: 96 }, { env: '2' })).toBe(1);
    expect(strips({ ...XTX, stripBatch: 128, colsValid: 1024 }, { env: '16' })).toBe(8);
  });

  it('sizes the launch, the batch and the transcript buffers by the strips, once the caps are read', () => {
    const create = slice(HOST, 'extern "C" void *pearl_host_create_checked(', '\n}\n');
    const width = create.indexOf('  ctx->colBatch = amd_col_batch(profile, ctx->arch);\n  ctx->stripBatch = ctx->colBatch;\n');
    const caps = create.indexOf('  resolve_fold(ctx);');
    const wide = create.indexOf('  ctx->strips = amd_strips(ctx);\n  ctx->colBatch = ctx->stripBatch * ctx->strips;\n'
      + '  ctx->batch = ctx->colBatch * ctx->rowsValid;\n');
    const room = create.indexOf('HIP_OK(hipMalloc(&ctx->dTr[i], (size_t)ctx->batch * PEARL_JACKPOT_BUCKETS * sizeof(uint32_t)),');
    expect(width).toBeGreaterThan(0);
    expect([width < caps, caps < wide, wide < room]).toEqual([true, true, true]);
    expect(create.match(/hipMalloc\(&ctx->dTr\[i\]/g)).toHaveLength(1);
    // A stamp build's room is the launch's too.
    expect(wide).toBeLessThan(create.indexOf('  if (ctx->stamp) {'));
    const submit = slice(HOST, 'extern "C" bool pearl_host_submit(', '\n}\n');
    expect(submit).toContain('  const dim3 grid(batchCols / f.bn / ctx->strips, ctx->profile.m / f.bm, ctx->strips), block(PEARL_AMD_THREADS);\n');
    expect(submit).toContain('  const uint32_t regions = ctx->batch;\n');
  });

  it('names the strips in the fold\'s description', () => {
    const name = slice(HOST, 'extern "C" const char *pearl_host_fold_name(void *handle) {', '\n}\n');
    expect(name).toContain('if (ctx->strips > 1u) snprintf(strips, sizeof strips, " in %u strips of %u", ctx->strips, ctx->stripBatch);');
    expect(name).toContain('"%s, %s, batch %u column offsets%s, k %u%s%s%s%s"');
  });
});
