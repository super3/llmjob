'use strict';

const fs = require('fs');
const path = require('path');

// Volta's fp16 tensor-core fold (pearl_tile_fold_hmma, PEARL_VOLTA_HMMA): a V100 runs it
// in place of the dp4a fold. Nothing here runs device code. The checks read the sources,
// and run the header's index functions (src/pearl_fold_hmma.h), translated to JS, on the
// CPU:
// - the switch, and that only the sm_70 pass builds the body;
// - the staging permutation is the dp4a fold's row and column groups;
// - every region is written by exactly one lane of one warp, from tiles of its own groups;
// - the 16-byte loads of A' and B' a warp makes for the mma are free of bank conflicts;
// - the int8-to-fp16 bits and the biased readout are exact, and where the window ends;
// - the host takes a V100 for it only from the body, and launches it with the fill's bias;
// - CI fails an sm_70 build whose fp16 fold has no body;
// - the exactness margins pearl_config.h states.
// probes/hmma-check.cc runs the whole fold on the CPU against a plain fold; verify-hits.js
// on a V100 checks the hardware's operand layout, which nothing here can.

const NL = String.fromCharCode(10);
const CR = String.fromCharCode(13);
const NATIVE = path.join(__dirname, '..', 'native');
const ROOT = path.join(NATIVE, '..', '..');
const rawRead = (file) => fs.readFileSync(file, 'utf8').split(CR + NL).join(NL);
const CONFIG = rawRead(path.join(NATIVE, 'src', 'pearl_config.h'));
const HOST = rawRead(path.join(NATIVE, 'src', 'pearl_host.cu'));
const KERNEL = rawRead(path.join(NATIVE, 'src', 'pearl_kernel.cu'));
const HEADER = rawRead(path.join(NATIVE, 'src', 'pearl_fold_hmma.h'));
const CI = rawRead(path.join(ROOT, '.github', 'workflows', 'native-core.yml'));
const CHECK = rawRead(path.join(NATIVE, 'probes', 'hmma-check.cc'));

const define = (name) => {
  const m = new RegExp('^#define ' + name + ' (\\w+)', 'm').exec(CONFIG);
  expect([name, m !== null]).toEqual([name, true]);
  return Number(m[1].replace(/u$/, ''));
};
const LDM = define('PEARL_HMMA_LDM');
const ASWZ = define('PEARL_HMMA_ASWZ');

// The header's PEARL_HD functions on 32-bit values, as JS: types and u suffixes dropped.
function headerFns(macros) {
  const names = ['pearl_hmma_row_of', 'pearl_hmma_col_of', 'pearl_hmma_r8', 'pearl_hmma_qp',
    'pearl_hmma_a_row', 'pearl_hmma_b_col', 'pearl_hmma_region'];
  let js = '';
  for (const name of names) {
    const at = HEADER.indexOf('PEARL_HD inline uint32_t ' + name + '(');
    expect([name, at >= 0]).toEqual([name, true]);
    const open = HEADER.indexOf('{', at);
    let depth = 0;
    let end = open;
    for (; end < HEADER.length; end++) {
      if (HEADER[end] === '{') depth++;
      if (HEADER[end] === '}' && --depth === 0) break;
    }
    const params = HEADER.slice(HEADER.indexOf('(', at) + 1, HEADER.indexOf(')', at)).replace(/uint32_t /g, '');
    const body = HEADER.slice(open + 1, end)
      .replace(/\bconst uint32_t\b/g, 'const')
      .replace(/\b(0x[0-9a-fA-F]+|\d+)u\b/g, '$1');
    js += `function ${name}(${params}) {${body}}` + NL;
  }
  for (const [k, v] of Object.entries(macros)) js = js.replace(new RegExp('\\b' + k + '\\b', 'g'), String(v));
  expect(/uint|int32|PEARL_/.test(js)).toBe(false);
  return new Function(js + 'return {' + names.join(', ') + '};')();
}
const F = headerFns({ PEARL_HMMA_ASWZ: ASWZ });

const ROWS_PATTERN = [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27];
const COLS_PATTERN = [0, 1, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 49, 56, 57];

describe('the switch and where the body is built (pearl_config.h)', () => {
  test('on by default, 0 or 1, a body in the sm_70 pass only', () => {
    const lines = CONFIG.split(NL).map((l) => l.trim());
    const at = lines.indexOf('#ifndef PEARL_VOLTA_HMMA');
    expect(lines.slice(at, at + 11)).toEqual([
      '#ifndef PEARL_VOLTA_HMMA', '#define PEARL_VOLTA_HMMA 1', '#endif',
      '#if PEARL_VOLTA_HMMA != 0 && PEARL_VOLTA_HMMA != 1', '#error "PEARL_VOLTA_HMMA is 0 or 1"', '#endif',
      '#if defined(__CUDA_ARCH__) && PEARL_VOLTA_HMMA && __CUDA_ARCH__ == 700',
      '#define PEARL_HMMA_BODY 1', '#else', '#define PEARL_HMMA_BODY 0', '#endif']);
    // The kernel's body follows PEARL_HMMA_BODY, once, and nothing else reads it.
    expect(KERNEL.match(/^#if PEARL_HMMA_BODY$/gm)).toHaveLength(1);
    expect(KERNEL.indexOf('#if PEARL_HMMA_BODY')).toBeGreaterThan(KERNEL.indexOf('void pearl_tile_fold_hmma('));
    for (const f of ['pearl_host.cu', 'pearl_fold_bd.cuh', 'pearl_fold_tall.cuh', 'pearl_core.cc', 'pearl_fold_dp4a.h']) {
      const text = rawRead(path.join(NATIVE, 'src', f));
      expect([f, /PEARL_HMMA_BODY\b/.test(text)]).toEqual([f, false]);
    }
  });

  test('the kernel: 128 threads, two blocks an SM, m8n8k4 with fp32 accumulators', () => {
    expect(KERNEL).toContain('extern "C" __global__ __launch_bounds__(PEARL_HMMA_THREADS, 2) void pearl_tile_fold_hmma(');
    expect(define('PEARL_HMMA_THREADS')).toBe(128);
    expect(KERNEL).toContain('"mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "');
    // 44 KB of static shared: inside the 48 KB a block gets, and two fit a V100's 96 KB.
    const smem = 2 * 2 * 128 * LDM * 2 + 64 * 17 * 4;
    expect([smem, smem <= 48 * 1024, 2 * smem <= 96 * 1024]).toEqual([45312, true, true]);
    expect(KERNEL).toContain('__shared__ __align__(32) __half sA[2][128u * LDM];');
    expect(KERNEL).toContain('__shared__ uint32_t sT[PEARL_DP4A_REGIONS * PEARL_DP4A_TR_STRIDE];');
  });
});

describe('the layout (pearl_fold_hmma.h)', () => {
  test('shared row (column) group g holds exactly the dp4a fold\'s row (column) group g', () => {
    for (let g = 0; g < 8; g++) {
      const rows = [];
      const cols = [];
      for (let e = 0; e < 16; e++) {
        rows.push(F.pearl_hmma_row_of(16 * g + e));
        cols.push(F.pearl_hmma_col_of(16 * g + e));
      }
      const wantRows = ROWS_PATTERN.map((r) => 32 * (g >> 1) + 4 * (g & 1) + r);
      const wantCols = COLS_PATTERN.map((c) => 64 * (g >> 2) + 2 * (g & 3) + c);
      expect([g, rows.sort((a, b) => a - b)]).toEqual([g, wantRows]);
      expect([g, cols.sort((a, b) => a - b)]).toEqual([g, wantCols.sort((a, b) => a - b)]);
    }
  });

  test('every region is written once a chunk, from four tiles of its own groups', () => {
    const writer = new Map();
    for (let warp = 0; warp < 4; warp++) {
      for (let lane = 0; lane < 32; lane += 2) {
        const region = F.pearl_hmma_region(warp, lane);
        expect(writer.has(region)).toBe(false);
        writer.set(region, [warp, lane]);
        const q = F.pearl_hmma_qp(lane);
        const a = (lane >> 4) & 1;
        const b = (lane >> 1) & 1;
        // Every lane of the quad pair, for the region's 2x2 of 8x8 tiles.
        for (let pl = 0; pl < 32; pl++) {
          if (F.pearl_hmma_qp(pl) !== q) continue;
          for (const i of [2 * a, 2 * a + 1]) expect(F.pearl_hmma_a_row(warp, pl, i) >> 4).toBe(region >> 3);
          for (const j of [2 * b, 2 * b + 1]) expect(F.pearl_hmma_b_col(warp, pl, j) >> 4).toBe(region & 7);
        }
      }
    }
    expect([...writer.keys()].sort((x, y) => x - y)).toEqual([...Array(64).keys()]);
  });

  test('a quad pair\'s 8 lanes bring 8 different rows and 8 different columns to each tile', () => {
    for (let warp = 0; warp < 4; warp++) {
      for (let q = 0; q < 4; q++) {
        const lanes = [...Array(32).keys()].filter((l) => F.pearl_hmma_qp(l) === q);
        expect(lanes).toHaveLength(8);
        for (let i = 0; i < 4; i++) {
          expect(new Set(lanes.map((l) => F.pearl_hmma_a_row(warp, l, i))).size).toBe(8);
          expect(new Set(lanes.map((l) => F.pearl_hmma_b_col(warp, l, i))).size).toBe(8);
        }
      }
    }
  });

  // A warp's 16-byte shared load runs 8 lanes at a time; within those 8, two lanes conflict
  // when they read different 16-byte units in the same group of 4 banks. Without the A
  // swizzle every load of A' is a 2-way conflict.
  test('the mma\'s 16-byte loads have no bank conflicts, and do without PEARL_HMMA_ASWZ', () => {
    const conflicts = (fns) => {
      let worst = 1;
      for (let warp = 0; warp < 4; warp++) {
        for (let i = 0; i < 4; i++) {
          for (const kk of [0, 8, 16, 24]) {
            for (const fn of [fns.pearl_hmma_a_row, fns.pearl_hmma_b_col]) {
              for (let phase = 0; phase < 4; phase++) {
                const byGroup = new Map();
                for (let lane = 8 * phase; lane < 8 * phase + 8; lane++) {
                  const unit = (fn(warp, lane, i) * LDM + kk) * 2 / 16;
                  const group = unit % 8;
                  if (!byGroup.has(group)) byGroup.set(group, new Set());
                  byGroup.get(group).add(unit);
                }
                for (const units of byGroup.values()) worst = Math.max(worst, units.size);
              }
            }
          }
        }
      }
      return worst;
    };
    expect(conflicts(F)).toBe(1);
    expect(conflicts(headerFns({ PEARL_HMMA_ASWZ: 0 }))).toBe(2);
    // The staging stores: thread t writes row t, 8 lanes at a time.
    for (let phase = 0; phase < 16; phase++) {
      const groups = new Set();
      for (let t = 8 * phase; t < 8 * phase + 8; t++) groups.add((t * LDM * 2 / 16) % 8);
      expect(groups.size).toBe(8);
    }
  });
});

describe('the numbers', () => {
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  const bitsOf = (x) => {
    f32[0] = x;
    return u32[0];
  };

  test('int8 to fp16: the biased byte under 0x64, less 1152, is the value', () => {
    // fp16 bits of 1152: exponent 10 (+15 = 25), mantissa 128.
    expect((25 << 10) | 128).toBe(0x6480);
    expect(HEADER).toContain('#define PEARL_HMMA_BIAS2 0x64806480u   // 1152 in both halves');
    expect(HEADER).toContain('return __byte_perm(u, 0x64646464u, hi ? 0x5342u : 0x5140u);');
    for (let x = -128; x < 128; x++) {
      const h = 0x6400 | ((x & 0xff) ^ 0x80);
      // A normal half in [1024, 2048): 1024 plus the mantissa.
      expect((h >> 10) === 25 && 1024 + (h & 0x3ff) - 1152 === x).toBe(true);
    }
  });

  // The accumulators start at the bias, so one holding n is bias + n, an exact fp32 while
  // under 2^24. Its bits less U are n; when the bias is 2^23 the bits alone XOR to the
  // same word, pairwise.
  test('the biased readout reads every accumulator in its window exactly', () => {
    expect(HEADER).toContain('  if (center >= 2048 && center <= 3072) return 8388608u;' + NL
      + '  return 12582912u - (uint32_t)center * 1088u;');
    expect(HEADER).toContain('PEARL_HD inline uint32_t pearl_hmma_unbias(uint32_t bias) { return 0x4B000000u + bias - 8388608u; }');
    expect(KERNEL).toContain('for (int e = 0; e < 8; e++) acc[i][j][e] = (float)bias;');
    expect(KERNEL).toContain('if (bias == 8388608u) {');
    const biasOf = (center) => (center >= 2048 && center <= 3072 ? 8388608 : 12582912 - center * 1088);
    const unbias = (bias) => (0x4b000000 + bias - 8388608) >>> 0;
    // What the accumulator holds after summing n, as the tensor core keeps it: exact.
    const held = (n, bias) => bitsOf(bias + n);
    for (const fill of [48, 0, 46, 55, 63, 16]) {
      const center = fill * fill;
      const bias = biasOf(center);
      const U = unbias(bias);
      // The window: [2^23 - bias, 2^24 - bias).
      const lo = 2 ** 23 - bias;
      const hi = 2 ** 24 - bias;
      // Every chunk's mean product sum is inside it.
      for (let chunk = 0; chunk < 16; chunk++) {
        const E = center * 128 * (chunk + 1);
        expect([fill, chunk, E >= lo && E < hi]).toEqual([fill, chunk, true]);
      }
      for (const n of [lo, lo + 1, hi - 1, Math.floor((lo + hi) / 2), lo + 12345]) {
        expect([fill, n, (held(n, bias) - U) | 0]).toEqual([fill, n, n]);
      }
      // One past the top is where an fp32's last place is 2: no longer n.
      expect(((held(hi + 1, bias) - U) | 0) === hi + 1).toBe(false);
      if (bias === 8388608) {
        // The bits are n | 0x4B000000, and 32 of them XOR to the XOR of the n.
        let x = 0;
        let want = 0;
        for (let t = 0; t < 32; t++) {
          const n = 300000 + t * 98765;
          x ^= held(n, bias);
          want ^= n;
        }
        expect(x >>> 0).toBe(want >>> 0);
      }
    }
    expect(biasOf(48 * 48)).toBe(2 ** 23);
    expect(biasOf(0)).toBe(1.5 * 2 ** 23);
  });

  test('the margins pearl_config.h states for the constant and hashed fills', () => {
    // The noise is a difference of two uniform [-32, 32) entries.
    const vu = (64 * 64 - 1) / 12;
    const vn = 2 * vu;
    const mean = 48 * 48;
    const sdProd = Math.sqrt((mean + vn) ** 2 - mean ** 2);
    const sdSum = sdProd * Math.sqrt(2048);
    expect(Math.round(Math.sqrt(vn))).toBe(26);
    expect(Math.round(sdProd / 100) * 100).toBe(1900);
    expect(Math.round(sdSum / 1000)).toBe(86);
    expect(mean * 2048).toBe(4718592);
    expect(Math.round((2 ** 23 - mean * 2048) / sdSum)).toBe(43);
    const sd0 = sdProd * Math.sqrt(128);
    expect(Math.round(sd0 / 100) * 100).toBe(21500);
    expect(Math.round(mean * 128 / 1000)).toBe(295);
    expect(Math.round(mean * 128 / sd0)).toBe(14);
    // Hashed: A' int7 uniform [-64, 64) plus the noise.
    const va = (128 * 128 - 1) / 12 + vn;
    expect(Math.round(Math.sqrt(va))).toBe(45);
    expect(Math.round(va / 100) * 100).toBe(2000);
    expect(Math.round(va * Math.sqrt(2048) / 1000)).toBe(93);
    expect(Math.round(2 ** 22 / (va * Math.sqrt(2048)))).toBe(45);
    expect(CONFIG).toContain('// accumulator is about 295,000, sd about 21,500: 14 sd above 0. After chunk 15 (2048)');
    expect(CONFIG).toContain('// it is about 4.72 million, sd about 86,000: 43 sd below 2^23. Under the hashed fill');
    expect(CONFIG).toContain('// side of 0; a product\'s sd is about 2,000 and the sum\'s about 93,000: 45 sd to either');
  });
});

describe('the host', () => {
  test('pearl_hmma_for: a 7.0 card, the sm_70 body, unless PEARL_VOLTA_DP4A=1', () => {
    const at = HOST.indexOf('bool pearl_hmma_for(const Ctx *ctx) {');
    expect(at).toBeGreaterThan(0);
    const body = HOST.slice(at, HOST.indexOf(NL + '}' + NL, at)).split(NL).map((l) => l.trim());
    expect(body).toEqual([
      'bool pearl_hmma_for(const Ctx *ctx) {',
      'if (!PEARL_VOLTA_HMMA || ctx->ccMajor != 7 || ctx->ccMinor != 0) return false;',
      'uint32_t v = 0;',
      'if (pearl_bd_sweep_env("PEARL_VOLTA_DP4A", &v) && v == 1u) return false;',
      'cudaFuncAttributes fh;',
      'if (cudaFuncGetAttributes(&fh, reinterpret_cast<const void *>(pearl_tile_fold_hmma))',
      '!= cudaSuccess) {',
      '(void)cudaGetLastError();',
      'return false;',
      '}',
      'return fh.binaryVersion == 70 && fh.numRegs >= 64',
      '&& (uint32_t)fh.maxThreadsPerBlock >= PEARL_HMMA_THREADS;']);
    expect(at).toBeLessThan(HOST.indexOf('void resolve_fold(Ctx *ctx) {'));
    // Only a card the dp4a fold took, so everything else about it is the dp4a fold's.
    expect(HOST).toContain('        ctx->foldDp4a = true;' + NL + '        ctx->foldKnown = true;' + NL
      + '        ctx->foldHmma = pearl_hmma_for(ctx);' + NL + '        return;');
    expect(HOST.match(/foldHmma = .*/g)).toEqual(['foldHmma = false;', 'foldHmma = pearl_hmma_for(ctx);']);
    expect(HOST).toContain('  bool foldHmma = false;' + NL);
  });

  test('the launch: the fp16 fold with the fill\'s bias, else the dp4a fold', () => {
    expect(HOST).toContain('  const uint32_t bias = pearl_hmma_bias(' + NL
      + '      ctx->profile.operand_fill == PEARL_OPERAND_CONST' + NL
      + '          ? (int32_t)PEARL_OPERAND_FILL * (int32_t)PEARL_OPERAND_FILL' + NL
      + '          : 0);' + NL
      + '  if (ctx->foldHmma)' + NL
      + '    pearl_tile_fold_hmma<<<tiles, PEARL_HMMA_THREADS, 0, st>>>(' + NL
      + '        ctx->dAp, ctx->dBp, ctx->profile.m, ctx->profile.n, k, rank, chunks, col_off,' + NL
      + '        ctx->rowsValid, col_groups, tiles, PEARL_DP4A_BAND, bias, test, hitList);' + NL
      + '  else' + NL
      + '    pearl_tile_fold_dp4a<<<tiles, PEARL_DP4A_THREADS, 0, st>>>(');
    // The kernel and the host agree on the arguments.
    expect(KERNEL).toContain('uint32_t rows_valid, uint32_t col_groups, uint32_t tiles, uint32_t band, uint32_t bias,'
      + NL + '    const PearlTranscriptTest test, const PearlHitList hits) {');
    expect(HOST).toContain('    uint32_t tiles, uint32_t band, uint32_t bias, const PearlTranscriptTest test,');
  });

  test('the fold name fits its buffer', () => {
    const size = Number(/char foldName\[(\d+)\] = \{0\};/.exec(HOST)[1]);
    expect(HOST).toContain('"fp16 mma 128x128, 128 threads of 64x64 m8n8k4, a block a tile, band %u, "' + NL
      + '             "col_batch %u (Volta)",');
    const name = 'fp16 mma 128x128, 128 threads of 64x64 m8n8k4, a block a tile, band 4294967295, col_batch 4294967295 (Volta)';
    expect(name.length).toBeLessThan(size);
  });
});

describe('builds and checks', () => {
  test('CI fails an sm_70 build whose fp16 fold has no body', () => {
    const at = CI.indexOf('- name: Check the fold kernels for register spills');
    const spill = CI.slice(at, CI.indexOf('          exit $bad', at));
    const order = ['hmma_body_70=0', 'pearl_tile_fold_hmma) seen_hmma=1 ;;',
      'if [ "$name" = pearl_tile_fold_hmma ] && [ "$arch" = sm_70 ] \\',
      'hmma_body_70=1', '[ "$seen_hmma" != 1 ]', '*"code=sm_70 "*)', 'if [ "$hmma_body_70" != 1 ]; then',
      '::error::pearl_tile_fold_hmma on sm_70 has no body'];
    let pos = 0;
    for (const p of order) {
      const i = spill.indexOf(p, pos);
      expect([p, i >= 0]).toEqual([p, true]);
      pos = i + p.length;
    }
  });

  test('the CPU check runs the header\'s own functions, on all four operand kinds', () => {
    expect(CHECK).toContain('#include "pearl_fold_hmma.h"');
    for (const fn of ['pearl_hmma_a_src', 'pearl_hmma_b_src', 'pearl_hmma_biased', 'pearl_hmma_bias',
      'pearl_hmma_unbias', 'pearl_hmma_a_row', 'pearl_hmma_b_col', 'pearl_hmma_region', 'pearl_dp4a_tile',
      'pearl_dp4a_region_index']) {
      expect([fn, CHECK.includes(fn + '(')]).toEqual([fn, true]);
    }
    expect(CHECK).toContain('{"hashed fill", -1, 0,');
    expect(CHECK).toContain('{"uniform bytes", -2, 0,');
  });
});
