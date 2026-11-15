// The Intel core's Battlemage fold (xmx16f, native/sycl/pearl_sycl_kernels.hpp): what an Arc B-series or Arc
// Pro B50 to B70 card mines with since 14 November 2026 (goal-b50a; 2D block loads since goal-b50b, 15
// November). No test here runs a GPU. These run the fold's index arithmetic as JS: every byte its loads hand
// a lane must be the byte DPAS 8x16x32 wants there, and every window of a batch must be taken by exactly one
// sub-group. The B50 checked the same on hardware (verify.sh); CI runs the fold's reference code on the SYCL
// CPU device, which reads the same bytes as the 2D loads.
const fs = require('fs');
const path = require('path');
const { PROFILE } = require('../src/shared/miner/pearlhash');

const SYCL = path.join(__dirname, '..', 'native', 'sycl');
const read = (...p) => fs.readFileSync(path.join(...p), 'utf8').replace(/\r\n/g, '\n');
const KERNELS = read(SYCL, 'pearl_sycl_kernels.hpp');
const HOST = read(SYCL, 'pearl_sycl_host.cpp');
const VERIFY = read(SYCL, 'verify.sh');

function slice(text, from, to) {
  const i = text.indexOf(from);
  if (i < 0) throw new Error('not found: ' + from);
  const j = text.indexOf(to, i + from.length);
  if (j < 0) throw new Error('not found after it: ' + to);
  return text.slice(i, j + to.length);
}
function cExpr(src, re, vars) {
  const m = re.exec(src);
  if (!m) throw new Error('pattern not found: ' + re);
  const js = m[1].replace(/\((?:size_t|uint64_t|uint32_t)\)/g, '').replace(/\b(\d+)u\b/g, '$1');
  return new Function(...vars, 'return ' + js + ';');
}

const FOLD = slice(KERNELS, 'template <bool HW, int V = 0> struct FoldXmx16F {', '\n};\n');
const fragA2d = cExpr(slice(KERNELS, 'inline uint64_t frag_a2d_word(', '\n}\n'), /return ([^;]+);/, ['stride', 'r', 'kw']);
const fragB2d = cExpr(slice(KERNELS, 'inline uint64_t frag_b2d_byte(', '\n}\n'), /return ([^;]+);/, ['stride', 'r', 'kk']);
const LOAD_A = slice(KERNELS, 'template <bool HW> inline void f16_load_a(', '\n}\n');
const LOAD_B = slice(KERNELS, 'template <bool HW> inline void f16_load_b(', '\n}\n');
// window_of, as the kernels' header has it (a test below holds the two together).
const windowOf = (g, rowWins, colWins, band) => {
  const per = band * colWins, b = Math.floor(g / per), r0 = b * band;
  const br = rowWins - r0 < band ? rowWins - r0 : band;
  const idx = g - b * per;
  return [r0 + (idx % br), Math.floor(idx / br)];
};

const K = PROFILE.k;
const KW = K / 4;

describe('xmx16f reads every operand byte where DPAS 8x16x32 wants it', () => {
  test('each layout stores every byte of a block once, inside the block', () => {
    // A': a 32-row block is 2 of the host's 16-row strides (dwords); B': a 64-column block, a byte each.
    const aStride = 16 * KW, bStride = 64 * KW;
    const seenA = new Uint8Array(2 * aStride);
    for (let r = 32; r < 64; r++) {
      for (let kw = 0; kw < KW; kw++) {
        const at = fragA2d(aStride, r, kw) - 2 * aStride;
        expect(at >= 0 && at < 2 * aStride).toBe(true);
        seenA[at]++;
      }
    }
    expect(seenA.every((x) => x === 1)).toBe(true);
    const seenB = new Uint8Array(4 * bStride);
    for (let r = 64; r < 128; r++) {
      for (let kk = 0; kk < K; kk++) {
        const at = fragB2d(bStride, r, kk) - 4 * bStride;
        expect(at >= 0 && at < 4 * bStride).toBe(true);
        seenB[at]++;
      }
    }
    expect(seenB.every((x) => x === 1)).toBe(true);
  });

  test('a lane gets its short of each A row and its column of B, as the fold addresses them', () => {
    // The fold's addresses and loads, as the source has them.
    expect(FOLD).toContain('const int8_t *Aw = A.Ap + (size_t)(2u * awr) * A.aStride;');
    expect(FOLD).toContain('const int8_t *Bb = A.Bp + (size_t)awc * A.bStride;');
    expect(FOLD).toContain('f16_load_a<HW>(Aw, A.k, ks + 1u < 64u ? ks + 1u : ks, lane, nxt);');
    expect(FOLD).toContain('f16_load_b<HW>(Bb, A.k, ks + 1u, lane, bf);');
    // The 2D loads: A' 2 blocks of 16 shorts by 16 lines of a 64-byte surface from line 16 ks, B' 4 blocks of
    // 16 bytes by 32 lines, transformed, from line 32 ks. A load writes the lines in order, so element i of
    // fragment rb of lane l is short l of line (8 rb + i) mod 16 of block (8 rb + i) / 16, and the transform
    // gives lane l of block cb the 4 lines of dword c of its column: what the reference code reads.
    expect(LOAD_A).toContain('__spirv_Subgroup2DBlockLoadINTEL(2, 16, 16, 2, (const __attribute__((opencl_global)) void *)Aw, 64, (int)(k / 2u),');
    expect(LOAD_A).toContain('64, v2coord{0, (int)(ks * 16u)}, (__attribute__((opencl_private)) void *)af);');
    expect(LOAD_A).toContain('af[rb][i] = (short)w[(r & 15u) * 32u + (r >> 4) * 16u + lane];');
    expect(LOAD_B).toContain('__spirv_Subgroup2DBlockLoadTransformINTEL(1, 16, 32, 4, (const __attribute__((opencl_global)) void *)Bb, 64,');
    expect(LOAD_B).toContain('(int)k, 64, v2coord{0, (int)(ks * 32u)},');
    expect(LOAD_B).toContain('const uint8_t *p = reinterpret_cast<const uint8_t *>(Bb) + (size_t)kk * 64u + 16u * (uint32_t)cb + lane;');
    expect(LOAD_B).toContain('w = (uint32_t)p[0] | ((uint32_t)p[64] << 8) | ((uint32_t)p[128] << 16) | ((uint32_t)p[192] << 24);');
    const aStrideB = 16 * K + 256, bStrideB = 64 * K;  // bytes, with the host's 256-byte gap after A' strides
    for (const [wr, wc] of [[0, 0], [5, 3], [4095, 63]]) {
      for (const ks of [0, 1, 37, 63]) {
        for (let rb = 0; rb < 4; rb++) {
          for (let lane = 0; lane < 16; lane++) {
            for (let i = 0; i < 8; i++) {
              // DPAS A: row i of the 8-row block, k 2 lane and 2 lane + 1 of the k-step.
              const r = 8 * rb + i;
              const byteA = 2 * wr * aStrideB + ks * 1024 + 2 * ((r & 15) * 32 + (r >> 4) * 16 + lane);
              const row = 32 * wr + r, k = 32 * ks + 2 * lane;
              expect(byteA).toBe(4 * fragA2d(aStrideB / 4, row, k >> 2) + (k & 3));
            }
          }
        }
        for (let cb = 0; cb < 4; cb++) {
          for (let lane = 0; lane < 16; lane++) {
            for (let c = 0; c < 8; c++) {
              for (let j = 0; j < 4; j++) {
                // DPAS B: the lane's column, byte j of dword c (k 4 c + j) of the k-step.
                const kk = 32 * ks + 4 * c;
                const byteB = wc * bStrideB + kk * 64 + 16 * cb + lane + 64 * j;
                expect(byteB).toBe(fragB2d(bStrideB / 4, 64 * wc + 16 * cb + lane, kk + j));
              }
            }
          }
        }
      }
    }
  });

  test('the host stores both operands so for xmx16f', () => {
    expect(HOST).toContain('if (ctx->frag == 4) lay = isA ? Layout{kA2D, ctx->aStride / 4u} : Layout{kB2D, ctx->bStride / 4u};');
    const mat = slice(KERNELS, 'inline void materialize16(', '\n}\n');
    expect(mat).toContain('ob[frag_b2d_byte(lay.stride, r, kk + (uint32_t)i)] = (uint8_t)(ow[i >> 2] >> ((i & 3) * 8));');
    expect(slice(KERNELS, 'inline uint64_t layout_word(', '\n}\n')).toContain('if (l.kind == kA2D) return frag_a2d_word(l.stride, r, kw);');
  });

  test('the readout gives lane q < 8 region q, as fold_xmx does', () => {
    // Rows 0-3 of each 8-row block are row offset 0, 4-7 row offset 4; lane j's column offset is j & 6, and
    // lanes j, j ^ 1, j ^ 8 share it.
    expect(FOLD).toContain('x0 ^= (uint32_t)(v[0] ^ v[1] ^ v[2] ^ v[3]);');
    expect(FOLD).toContain('x0 ^= sycl::permute_group_by_xor(sg, x0, 8u);');
    expect(FOLD).toContain('tw[ch * 16u + lane] = (lane & 1u) ? x1 : x0;');
    expect(FOLD).toContain('each<16>([&](int j) { t[j] = tw[(uint32_t)j * 16u + lane]; });');
    expect(FOLD).toContain('const uint32_t local = (2u * wr + (lane & 1u)) + A.rowsValid * (4u * wc + (lane >> 1));');
    for (let q = 0; q < 8; q++) expect([(q & 6), 2 * (q >> 1)]).toEqual([q & 6, q & 6]);
  });
});

describe('xmx16f takes every window of a batch once', () => {
  const mapped = (groups, rowWins, colWins, band, R, C) => {
    const seen = new Map();
    for (let g = 0; g < groups / (R * C); g++) {
      const bandB = Math.floor(band / R) > 0 ? Math.floor(band / R) : 1;
      const [br, bc] = windowOf(g, rowWins / R, colWins / C, bandB);
      for (let sgi = 0; sgi < R * C; sgi++) {
        const key = (br * R + (sgi % R)) + ',' + (bc * C + Math.floor(sgi / R));
        seen.set(key, (seen.get(key) || 0) + 1);
      }
    }
    return seen;
  };

  test('the source maps a sub-group to its window this way', () => {
    const wo = slice(KERNELS, 'inline void window_of(', '\n}\n');
    expect(wo).toContain('const uint32_t br = rowWins - r0 < band ? rowWins - r0 : band;');
    expect(wo).toContain('wc = idx / br;\n  wr = r0 + idx % br;');
    expect(FOLD).toContain('window_of((uint32_t)it.get_group(0), rowWins / R, colWins / C, bandB, br, bc);');
    expect(FOLD).toContain('wr = br * R + sgi % R;');
    expect(FOLD).toContain('wc = bc * C + sgi / R;');
    expect(KERNELS).toContain('h.parallel_for(sycl::nd_range<1>((size_t)groups * 16u, wg), FoldXmx16F<HW, V>{a, tl});');
    // A sub-group's 256 transcript words in local memory: one slice of the work-group's each.
    expect(KERNELS).toContain('sycl::local_accessor<uint32_t, 1> tl(256u * a.gRows * a.gCols, h);');
    expect(FOLD).toContain('uint32_t *tw = &tl[0] + (size_t)sgi * 256u;');
  });

  test('at every shape the host allows', () => {
    for (const [rowWins, colWins, band] of [[64, 8, 64], [64, 64, 16], [48, 4, 32], [32, 1, 8]]) {
      for (const [R, C] of [[1, 1], [1, 4], [2, 2], [4, 4], [1, 16]]) {
        if (rowWins % R || band % R || colWins % C) continue;
        const seen = mapped(rowWins * colWins, rowWins, colWins, band, R, C);
        expect([rowWins, colWins, R, C, seen.size]).toEqual([rowWins, colWins, R, C, rowWins * colWins]);
        expect([...seen.values()].every((n) => n === 1)).toBe(true);
      }
    }
  });

  test('4 by 4 windows a work-group by default, 1 where 4 does not divide the batch or the band', () => {
    const launch = slice(HOST, 'void launch_fold(Ctx *ctx, int fold, const FoldArgs &a, uint32_t groups) {', '\n}\n');
    expect(launch).toContain('uint32_t R = (uint32_t)strtoul(env_str("PEARL_SYCL_F16R", "4").c_str(), nullptr, 10);');
    expect(launch).toContain('uint32_t C = (uint32_t)strtoul(env_str("PEARL_SYCL_F16C", "4").c_str(), nullptr, 10);');
    expect(launch).toContain('if (R < 1 || R > 32 || (a.rowsValid / 2u) % R || a.band % R) R = 1;');
    expect(launch).toContain('if (C < 1 || R * C > 32 || (a.colBatch / 4u) % C) C = 1;');
    // The other folds get one window a sub-group's worth of fields they ignore.
    expect(slice(HOST, 'FoldArgs fold_args(', '\n}\n')).toContain('a.gRows = a.gCols = 1;');
  });
});

describe('where xmx16f runs', () => {
  test('auto on Battlemage, xmx16 on the other sub-group 16 cards, by name anywhere', () => {
    expect(HOST).toContain('int auto_fold(int sg, bool bmg) { return sg == 16 ? (bmg ? kXmx16F : kXmx16) : sg == 8 ? kXmx8W : kDot; }');
    // Its operands in its own order, and a band of a quarter of the cache, as the other fragment folds.
    expect(HOST).toContain('if (ctx->frag == 4) lay = isA ? Layout{kA2D, ctx->aStride / 4u} : Layout{kB2D, ctx->bStride / 4u};');
    expect(HOST).toContain('|| f == kXmx16FEmu;\n}');
    // The SPIR-V extension its 2D loads need, in both builds the driver compiles.
    expect(read(SYCL, 'build.sh')).toContain('-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io');
    expect(read(SYCL, 'build-win.sh')).toContain('-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io');
    // The CPU device checks its reference code in CI.
    expect(VERIFY).toContain('--folds dot,xmx16,xmx16f,xmx8,xmx8f)');
    expect(VERIFY).toContain('FOLDS=(dot xmx16 xmx16f xmx8 xmx8f)');
    // The hardware kernel asks for the 256-register mode: its 16 accumulators are 128 registers.
    expect(FOLD).toContain('return syclex::properties{syclex::sub_group_size<16>, sycl::ext::intel::experimental::grf_size<256>};');
  });
});
