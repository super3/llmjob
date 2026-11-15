// The Intel core's Battlemage fold (xmx16f, native/sycl/pearl_sycl_kernels.hpp): what an Arc B-series or Arc
// Pro B50 to B70 card mines with since 14 November 2026 (goal-b50a). No test here runs a GPU. These run the
// fold's index arithmetic as JS: every byte its block reads hand a lane must be the byte DPAS 8x16x32 wants
// there, and every window of a batch must be taken by exactly one sub-group. The B50 checked the same on
// hardware (verify.sh); CI runs the fold's reference code on the SYCL CPU device.
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
const fragA16 = cExpr(slice(KERNELS, 'inline uint64_t frag_a16_word(', '\n}\n'), /return ([^;]+);/, ['stride', 'r', 'kw']);
const fragB16 = cExpr(slice(KERNELS, 'inline uint64_t frag_b16_word(', '\n}\n'), /return ([^;]+);/, ['stride', 'r', 'kw']);
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
  test('each layout stores every dword of a block once, inside the block', () => {
    for (const [fn, rowsABlock] of [[fragA16, 16], [fragB16, 64]]) {
      const stride = rowsABlock * KW;
      const seen = new Uint8Array(stride);
      for (let r = rowsABlock; r < 2 * rowsABlock; r++) {
        for (let kw = 0; kw < KW; kw++) {
          const at = fn(stride, r, kw) - stride;
          expect(at >= 0 && at < stride).toBe(true);
          seen[at]++;
        }
      }
      expect(seen.every((x) => x === 1)).toBe(true);
    }
  });

  test('a lane gets its short of each A row and its column of B, as the fold addresses them', () => {
    // The fold's addresses, as the source has them.
    expect(FOLD).toContain('const int8_t *A0 = A.Ap + (size_t)(2u * awr) * A.aStride;');
    expect(FOLD).toContain('const int8_t *A1 = A0 + A.aStride;');
    expect(FOLD).toContain('const int8_t *Bb = A.Bp + (size_t)awc * A.bStride;');
    expect(FOLD).toContain('af[rb] = b16_read_a<HW>((rb >> 1 ? A1 : A0) + (size_t)ks * 512u + (size_t)(rb & 1) * 256u, lane);');
    expect(FOLD).toContain('bf[cb] = b16_read_b<HW>(Bb + (size_t)ks * 2048u + (size_t)cb * 512u, lane);');
    // The reads: lane l, element c is short (A) or dword (B) c * 16 + l at the address, as a block read hands
    // them out and as the reference code reads them.
    expect(slice(KERNELS, 'template <bool HW> inline v8s b16_read_a(', '\n}\n')).toContain('r[i] = (short)w[16u * (uint32_t)i + lane];');
    expect(slice(KERNELS, 'template <bool HW> inline v8i b16_read_b(', '\n}\n')).toContain('r[i] = (int)w[16u * (uint32_t)i + lane];');
    const aStrideB = 16 * K + 256, bStrideB = 64 * K;  // bytes, with the host's 256-byte gap after A' blocks
    for (const [wr, wc] of [[0, 0], [5, 3], [4095, 63]]) {
      for (const ks of [0, 1, 37, 63]) {
        for (let rb = 0; rb < 4; rb++) {
          for (let lane = 0; lane < 16; lane++) {
            for (let c = 0; c < 8; c++) {
              // DPAS A: row c of the 8-row block, k 2 lane and 2 lane + 1 of the k-step.
              const byteA = (2 * wr + (rb >> 1)) * aStrideB + ks * 512 + (rb & 1) * 256 + 2 * (c * 16 + lane);
              const row = 32 * wr + 8 * rb + c, k = 32 * ks + 2 * lane;
              expect(byteA).toBe(4 * fragA16(aStrideB / 4, row, k >> 2) + (k & 3));
            }
          }
        }
        for (let cb = 0; cb < 4; cb++) {
          for (let lane = 0; lane < 16; lane++) {
            for (let c = 0; c < 8; c++) {
              // DPAS B: the lane's column, dword c (k 4 c .. 4 c + 3) of the k-step.
              const byteB = wc * bStrideB + ks * 2048 + cb * 512 + 4 * (c * 16 + lane);
              expect(byteB).toBe(4 * fragB16(bStrideB / 4, 64 * wc + 16 * cb + lane, 8 * ks + c));
            }
          }
        }
      }
    }
  });

  test('the readout gives lane q < 8 region q, as fold_xmx does', () => {
    // Rows 0-3 of each 8-row block are row offset 0, 4-7 row offset 4; lane j's column offset is j & 6, and
    // lanes j, j ^ 1, j ^ 8 share it.
    expect(FOLD).toContain('x0 ^= (uint32_t)(v[0] ^ v[1] ^ v[2] ^ v[3]);');
    expect(FOLD).toContain('x0 ^= sycl::permute_group_by_xor(sg, x0, 8u);');
    expect(FOLD).toContain('t[15] = (lane & 1u) ? x1 : x0;');
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
    expect(KERNELS).toContain('h.parallel_for(sycl::nd_range<1>((size_t)groups * 16u, wg), FoldXmx16F<HW, V>{a});');
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

  test('4 column windows a work-group by default, 1 where 4 does not divide the batch', () => {
    const launch = slice(HOST, 'void launch_fold(Ctx *ctx, int fold, const FoldArgs &a, uint32_t groups) {', '\n}\n');
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
    expect(HOST).toContain('if (ctx->frag == 4) lay = isA ? Layout{kFragA16, ctx->aStride / 4u} : Layout{kFragB16, ctx->bStride / 4u};');
    expect(HOST).toContain('|| f == kXmx16FEmu;\n}');
    // The CPU device checks its reference code in CI.
    expect(VERIFY).toContain('--folds dot,xmx16,xmx16f,xmx8,xmx8f)');
    expect(VERIFY).toContain('FOLDS=(dot xmx16 xmx16f xmx8 xmx8f)');
    // The hardware kernel asks for the 256-register mode: its 16 accumulators are 128 registers.
    expect(FOLD).toContain('return syclex::properties{syclex::sub_group_size<16>, sycl::ext::intel::experimental::grf_size<256>};');
  });
});
