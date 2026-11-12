// The Intel core's fragment fold (xmx8f, native/sycl/pearl_sycl_kernels.hpp):
// the fold an Arc A-series card mines with since November 2026. It reads the
// noised operands in an order of its own, which the draw's last kernel writes.
// No test here runs a GPU. These take the index arithmetic out of the C++ and
// run it: every byte the draw stores must be the byte the fold's DPAS takes
// there, and every sum must land in the region regionToTile gives it. The rig
// checks the same on hardware (verify.sh); CI runs the fold's reference code
// on the SYCL CPU device.
const fs = require('fs');
const path = require('path');
const { PROFILE, regionToTile } = require('../src/shared/miner/pearlhash');

const SYCL = path.join(__dirname, '..', 'native', 'sycl');
const read = (...p) => fs.readFileSync(path.join(...p), 'utf8').replace(/\r\n/g, '\n');
const KERNELS = read(SYCL, 'pearl_sycl_kernels.hpp');
const HOST = read(SYCL, 'pearl_sycl_host.cpp');
const CHECK = read(SYCL, 'pearl_sycl_check.cpp');
const GATE = read(SYCL, 'compile-gate.sh');

function slice(text, from, to) {
  const i = text.indexOf(from);
  if (i < 0) throw new Error('not found: ' + from);
  const j = text.indexOf(to, i + from.length);
  if (j < 0) throw new Error('not found after it: ' + to);
  return text.slice(i, j + to.length);
}
const stripComments = (src) => src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/.*$/gm, '');

// A C index expression as a JS function of `vars`: casts and the unsigned
// suffix dropped. Every value here is far below 2^31, so JS's 32-bit shifts
// and masks give what the C gives.
function cExpr(src, re, vars) {
  const m = re.exec(src);
  if (!m) throw new Error('pattern not found: ' + re);
  const js = m[1].replace(/\((?:size_t|uint64_t|uint32_t)\)/g, '').replace(/\b(\d+)u\b/g, '$1').replace(/\bA\./g, '');
  return new Function(...vars, 'return ' + js + ';');
}

const FOLD = stripComments(slice(KERNELS, 'template <int SG, int RB, bool HW> struct FoldXmxF {', '\n};\n'));
const fragA = cExpr(slice(KERNELS, 'inline uint64_t frag_a_word(', '\n}\n'), /return ([^;]+);/, ['stride', 'r', 'kw']);
const fragB = cExpr(slice(KERNELS, 'inline uint64_t frag_b_word(', '\n}\n'), /return ([^;]+);/, ['stride', 'r', 'kw']);

const K = PROFILE.k;
const KW = K / 4;        // dwords of k a row
const KSTEPS = K / 32;   // DPAS k-steps
const SG = 8, RB = 2, S = 2, NCB = 8;

describe('the fragment order of the noised operands', () => {
  test('the mainnet k is 64 k-steps of 32', () => {
    expect([K, KW, KSTEPS]).toEqual([2048, 512, 64]);
  });

  test('each layout stores every dword of a block once, inside the block', () => {
    // A': blocks of 16 rows. B': blocks of 64. With the stride a block's own
    // size there is no gap, so the operand is the size it was row-major.
    for (const [fn, rowsABlock] of [[fragA, 16], [fragB, 64]]) {
      const stride = rowsABlock * KW;
      for (const block of [0, 3]) {
        const seen = new Uint8Array(stride);
        for (let r = block * rowsABlock; r < (block + 1) * rowsABlock; r++) {
          for (let kw = 0; kw < KW; kw++) {
            const at = fn(stride, r, kw) - block * stride;
            expect(at >= 0 && at < stride).toBe(true);
            seen[at]++;
          }
        }
        expect(seen.every((x) => x === 1)).toBe(true);
      }
    }
  });

  test('a wider stride leaves a gap after each block and moves nothing inside it', () => {
    for (const [fn, rowsABlock] of [[fragA, 16], [fragB, 64]]) {
      const own = rowsABlock * KW, wide = own + 48;
      for (const r of [0, 5, rowsABlock, 3 * rowsABlock + 9]) {
        for (const kw of [0, 7, 8, 511]) {
          const block = Math.floor(r / rowsABlock);
          expect(fn(wide, r, kw) - block * wide).toBe(fn(own, r, kw) - block * own);
        }
      }
    }
  });

  test('materialize16 stores each dword where the layout says, row-major by default', () => {
    const mat = slice(KERNELS, 'inline void materialize16(', '\n}\n');
    expect(mat).toContain('o[layout_word(lay, r, (kk >> 2) + (uint32_t)w, 1u << k_log2)] = ow[w];');
    const lw = slice(KERNELS, 'inline uint64_t layout_word(', '\n}\n');
    expect(lw).toContain('if (l.kind == kFragA) return frag_a_word(l.stride, r, kw);');
    expect(lw).toContain('if (l.kind == kFragB) return frag_b_word(l.stride, r, kw);');
    expect(lw).toContain('return (uint64_t)r * (k >> 2) + kw;');
    expect(KERNELS).toContain('enum : uint32_t { kRowMajor = 0, kFragA = 1, kFragB = 2, kFragAT = 3, kFragBT = 4 };');
    expect(lw).toContain('if (l.kind == kFragAT) return frag_at_word(l.stride, r, kw);');
    expect(lw).toContain('if (l.kind == kFragBT) return frag_bt_word(l.stride, r, kw);');
  });
});

describe('fold_xmxf reads what the draw stored', () => {
  // The kernel's addresses, as byte offsets into A' and B'.
  const row0 = cExpr(FOLD, /const uint32_t row0 = ([^;]+);/, ['wr', 'sgi', 'RB']);
  const aBase = cExpr(FOLD, /\*Ab = A\.Ap \+ ([^;]+);/, ['row0', 'aStride', 'lane']);
  const bBase = cExpr(FOLD, /\*Bb = A\.Bp \+ ([^;]+);/, ['colStart', 'wc', 'bStride', 'lane']);
  const aLoad = cExpr(FOLD, /af\[rb\] = \*reinterpret_cast<const v8iu \*>\(Ab \+ ([^;]+)\);/, ['ks', 'rb']);
  const bLoad = cExpr(FOLD, /bf\[cb\] = load_halves\(Bb \+ ([^;]+)\);/, ['ks', 'cb']);

  test('the fold is sub-group 8 with 16 rows a sub-group', () => {
    expect(FOLD).toContain('static_assert(SG == 8, "the fragment layouts are sub-group 8\'s");');
    expect(FOLD).toContain('static constexpr int S = 4 / RB;');
    expect(FOLD).toContain('static constexpr int NCB = 64 / SG;');
    const launch = slice(HOST, 'void launch_fold(Ctx *ctx, int fold, const FoldArgs &a, uint32_t groups) {', '\n}\n');
    expect(launch).toContain('case kXmx8F: fold_xmxf<8, 2, true>(q, a, groups); break;');
    expect(launch).toContain('case kXmx8FEmu: fold_xmxf<8, 2, false>(q, a, groups); break;');
    // The launch: 16 work-items a window, 2 sub-groups of 8.
    const submit = slice(KERNELS, 'inline void fold_xmxf(sycl::queue &q, const FoldArgs &a, uint32_t groups) {', '\n}\n');
    expect(submit).toContain('h.parallel_for(sycl::nd_range<1>((size_t)groups * WG, WG), FoldXmxF<SG, RB, HW>{a, part});');
  });

  // The DPAS operand layout at sub-group 8 (the extension's, as the header of
  // the XMX fold says): A is a dword a row, lane l holding k = 4l .. 4l+3 of a
  // k-step; B is the lane's column, dword d holding k = 4d .. 4d+3.
  for (const [aStride, bStride] of [[16 * K, 64 * K], [16 * K + 192, 64 * K + 4096]]) {
    test('A\': dword r of a lane\'s load is row r of its 8-row block, at the lane\'s 4 k (stride ' + aStride + ')', () => {
      for (const wr of [0, 1, 3]) {
        for (let sgi = 0; sgi < S; sgi++) {
          const first = row0(wr, sgi, RB);
          expect(first).toBe(32 * wr + 16 * sgi);
          for (let lane = 0; lane < SG; lane++) {
            for (const ks of [0, 1, 37, KSTEPS - 1]) {
              for (let rb = 0; rb < RB; rb++) {
                const at = aBase(first, aStride, lane) + aLoad(ks, rb);
                for (let r = 0; r < 8; r++) {
                  expect(at + 4 * r).toBe(4 * fragA(aStride / 4, first + 8 * rb + r, ks * 8 + lane));
                }
              }
            }
          }
        }
      }
    });

    test('B\': dword d of a lane\'s load is its column\'s k = 4d .. 4d+3 of the k-step (stride ' + bStride + ')', () => {
      for (const colStart of [0, 8, 24]) {
        for (const wc of [0, 1]) {
          for (let lane = 0; lane < SG; lane++) {
            for (const ks of [0, 1, 37, KSTEPS - 1]) {
              for (let cb = 0; cb < NCB; cb++) {
                const at = bBase(colStart, wc, bStride, lane) + bLoad(ks, cb);
                const col = 16 * colStart + 64 * wc + SG * cb + lane;
                for (let d = 0; d < 8; d++) {
                  expect(at + 4 * d).toBe(4 * fragB(bStride / 4, col, ks * 8 + d));
                }
              }
            }
          }
        }
      }
    });
  }

  test('every k-step of every chunk is folded once, in order', () => {
    expect(FOLD).toContain('for (uint32_t ch = 0; ch < 16u; ch++) {');
    expect(FOLD).toContain('for (uint32_t s = 0; s < 4u; s++) {');
    expect(FOLD).toContain('const uint32_t ks = ch * 4u + s;');
    expect(FOLD).toContain('each<RB>([&](int rb) { c[rb][cb] = dpas<SG, HW>(sg, af[rb], bf[cb], c[rb][cb]); });');
  });

  test('a k-step is written out, with no loop for a compiler to keep', () => {
    // each<N>(f) is f(0) .. f(N - 1) by a fold expression. Kept as loops, the
    // fragments and accumulators stayed arrays in memory under oneAPI's
    // compiler and the fold ran at 3 TH/s where it does 64.
    expect(KERNELS).toContain('template <typename F, size_t... I> inline void each_of(F &f, std::index_sequence<I...>) { (f((int)I), ...); }');
    expect(KERNELS).toContain('template <int N, typename F> inline void each(F f) { each_of(f, std::make_index_sequence<N>{}); }');
    const kstep = slice(FOLD, 'const uint32_t ks = ch * 4u + s;', 'x0 ^= sycl::permute_group_by_xor(sg, x0, 1u);');
    expect(kstep).not.toMatch(/\bfor \(/);
    expect(kstep.match(/each<(RB|NCB)>\(/g)).toHaveLength(9);
  });

  test('a fragment of B\' is loaded as its two halves, in order', () => {
    const halves = stripComments(slice(KERNELS, 'inline v8i load_halves(const int8_t *p) {', '\n}\n'));
    expect(halves).toContain('const v4iu lo = *reinterpret_cast<const v4iu *>(p);');
    expect(halves).toContain('const v4iu hi = *reinterpret_cast<const v4iu *>(p + 16);');
    expect(halves).toContain('return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);');
    expect(KERNELS).toContain('typedef int v4iu __attribute__((ext_vector_type(4), aligned(4)));');
  });

  test('the word that orders the loads is always zero', () => {
    // The fold ORs (a word of every loaded fragment & zero) into A, so that no
    // DPAS of a k-step starts before its loads have landed. Any other value
    // would change A: the host passes 0 and nothing else.
    const body = FOLD.split('\n').map((l) => l.trim());
    const at = body.indexOf('int wait = 0;');
    expect(at).toBeGreaterThan(0);
    expect(body.slice(at, at + 7)).toEqual([
      'int wait = 0;',
      'each<RB>([&](int rb) { wait ^= af[rb][0]; });',
      'each<NCB>([&](int cb) { wait ^= bf[cb][0]; });',
      'wait &= (int)A.zero;',
      'each<RB>([&](int rb) { af[rb][0] |= wait; });',
      'each<NCB>([&](int cb) {',
      'each<RB>([&](int rb) { c[rb][cb] = dpas<SG, HW>(sg, af[rb], bf[cb], c[rb][cb]); });',
    ]);
    expect(HOST.match(/a\.zero = /g)).toHaveLength(1);
    expect(slice(HOST, 'FoldArgs fold_args(Ctx *ctx, uint64_t nonce_base, int slot, uint32_t *allTr) {', '\n}\n'))
      .toContain('a.zero = 0;');
    for (const v of [0x12345678, -1, 0]) expect(v & 0).toBe(0);
  });
});

describe('fold_xmxf places every region where regionToTile does', () => {
  const P = Object.assign({}, PROFILE, { m: 128, n: 512 });
  const ROWS_VALID = P.m / 16;
  const COL_BATCH = 8;
  const row0 = cExpr(FOLD, /const uint32_t row0 = ([^;]+);/, ['wr', 'sgi', 'RB']);
  const local = cExpr(FOLD, /const uint32_t local = ([^;]+);/, ['wr', 'wc', 'qr', 'rowsValid']);

  test('each lane\'s accumulators, the one shuffle and the region index', () => {
    // What the model takes as given, pinned to the code: lane = column of a
    // column block, element r = row r of an 8-row block, rows 0-3 into x0 and
    // 4-7 into x1, lane ^ 1 gathered, lane q writing region q's word.
    expect(FOLD).toContain('const v8i v = c[rb][cb];');
    expect(FOLD).toContain('x0 ^= (uint32_t)(v[0] ^ v[1] ^ v[2] ^ v[3]);');
    expect(FOLD).toContain('x1 ^= (uint32_t)(v[4] ^ v[5] ^ v[6] ^ v[7]);');
    expect([...FOLD.matchAll(/permute_group_by_xor\(sg, x0, (\d+)u\)/g)].map((m) => Number(m[1]))).toEqual([1]);
    expect([...FOLD.matchAll(/permute_group_by_xor\(sg, x1, (\d+)u\)/g)].map((m) => Number(m[1]))).toEqual([1]);
    expect(FOLD).toContain('if (lane < 8u) part[(sgi * 8u + lane) * 16u + ch] = (lane & 1u) ? x1 : x0;');
    expect(FOLD).toContain('for (int s = 0; s < S; s++) x ^= part[((uint32_t)s * 8u + qr) * 16u + (uint32_t)j];');
    for (const colStart of [0, 8, 24]) {
      const cells = new Map();
      for (let wr = 0; wr < ROWS_VALID / 2; wr++) {
        for (let wc = 0; wc < COL_BATCH / 4; wc++) {
          for (let qr = 0; qr < 8; qr++) {
            const at = local(wr, wc, qr, ROWS_VALID);
            if (!cells.has(at)) cells.set(at, new Set());
            for (let sgi = 0; sgi < S; sgi++) {
              for (const lane of [qr, qr ^ 1]) {
                for (let rb = 0; rb < RB; rb++) {
                  for (let r = (qr & 1) * 4; r < (qr & 1) * 4 + 4; r++) {
                    for (let cb = 0; cb < NCB; cb++) {
                      cells.get(at).add((row0(wr, sgi, RB) + 8 * rb + r) + ',' + (16 * colStart + 64 * wc + SG * cb + lane));
                    }
                  }
                }
              }
            }
          }
        }
      }
      expect(cells.size).toBe(COL_BATCH * ROWS_VALID);
      for (const [at, got] of cells) {
        const { rows, cols } = regionToTile(colStart * ROWS_VALID + at, P);
        const want = [];
        for (const r of rows) for (const c of cols) want.push(r + ',' + c);
        expect([...got].sort()).toEqual(want.sort());
      }
    }
  });
});

describe('the host keeps the operands in the order the fold reads', () => {
  test('a context draws for its own fold, and a batch of another fold draws again', () => {
    expect(HOST).toContain('bool fold_is_frag(int f) { return f == kXmx8F || f == kXmx8FEmu || f == kXmx8R || f == kXmx8REmu; }');
    expect(HOST).toContain('int fold_layout(int f) { return f == kXmx8R || f == kXmx8REmu ? 2 : fold_is_frag(f) ? 1 : 0; }');
    const create = slice(HOST, 'extern "C" void *pearl_host_create_checked(', '\n}\n');
    expect(create).toContain('ctx->frag = fold_layout(ctx->fold);');
    expect(create).toContain('ctx->aStride = 16u * k;');
    expect(create).toContain('ctx->bStride = 64u * k;');
    const noise = slice(HOST, 'void draw_noise(Ctx *ctx, bool isA) {', '\n}\n');
    expect(noise).toContain('if (ctx->frag == 1) lay = isA ? Layout{kFragA, ctx->aStride / 4u} : Layout{kFragB, ctx->bStride / 4u};');
    expect(noise).toContain('if (ctx->frag == 2) lay = isA ? Layout{kFragAT, ctx->aStride / 4u} : Layout{kFragBT, ctx->bStride / 4u};');
    const set = slice(HOST, 'void set_layout(Ctx *ctx, int frag) {', '\n}\n');
    expect(set).toContain('if (ctx->frag == frag) return;');
    expect(set).toContain('draw_noise(ctx, true);');
    expect(set).toContain('draw_noise(ctx, false);');
    // Mining: the context's own. The check tool: whatever fold it asks for.
    expect(slice(HOST, 'extern "C" bool pearl_host_submit(', '\n}\n'))
      .toContain('set_layout(ctx, fold_layout(ctx->fold));');
    expect(slice(HOST, 'extern "C" bool pearl_sycl_run_batch(', '\n}\n'))
      .toContain('set_layout(ctx, fold_layout(fold < 0 ? ctx->fold : fold));');
    // The host's reference fold reads rows: it gets them row-major.
    expect(slice(HOST, 'extern "C" bool pearl_sycl_read_operands(', '\n}\n')).toContain('set_layout(ctx, 0);');
  });

  test('auto picks the fragment fold on Alchemist, and the old fold keeps its name', () => {
    const pick = slice(HOST, 'bool pick_fold(Ctx *ctx, char *err, size_t err_len) {', '\n}\n');
    expect(pick).toContain('ctx->fold = sg == 16 ? kXmx16 : sg == 8 ? kXmx8F : kDot;');
    expect(pick).toContain('} else if (want == "xmx8") {\n    ctx->fold = kXmx8;');
    expect(pick).toContain('} else if (want == "xmx8f") {\n    ctx->fold = kXmx8F;');
    expect(pick).toContain('} else if (want == "xmx8r") {\n    ctx->fold = kXmx8R;');
    expect(pick).toContain('ctx->xmxHw = (ctx->fold == kXmx16 && sg == 16) || ((ctx->fold == kXmx8 || ctx->fold == kXmx8F || ctx->fold == kXmx8R) && sg == 8);');
    // The check tool's names are the host's codes, in order.
    expect(CHECK).toContain('const char *kFoldNames[] = {"dot", "xmx16", "xmx8", "xmx16-emu", "xmx8-emu", "xmx8f", "xmx8f-emu", "xmx8r",\n                            "xmx8r-emu"};');
    expect(CHECK).toContain('if (s == "auto-emu") return own == 1 ? 3 : own == 2 ? 4 : own == 5 ? 6 : own == 7 ? 8 : -3;');
  });

  test('the hardware kernel asks for 256 registers by a kernel property, the reference kernel does not', () => {
    const get = slice(FOLD, 'auto get(sycl::ext::oneapi::experimental::properties_tag) const {', '\n  }\n');
    expect(get).toContain('if constexpr (HW && RB > 1)');
    expect(get).toContain('return syclex::properties{syclex::sub_group_size<SG>, sycl::ext::intel::experimental::grf_size<256>};');
    expect(get).toContain('return syclex::properties{syclex::sub_group_size<SG>};');
    // The gate holds a build to it, and still gates the fold PEARL_SYCL_FOLD=xmx8 picks.
    expect(GATE).toContain("intel_gpu_acm_*) hws=('fold_xmxf<sg8,rb2,hw>' 'fold_xmx<sg8,rb1,hw>') ;;");
    expect(GATE).toContain('$2 == "fold_xmxf<sg8,rb2,hw>" && ($4 != 256 || $5 < 16) { bad = 1 }');
  });
});
