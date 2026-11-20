'use strict';

// PEARL_AMD_FUSE (earn/native/amd/src/pearl_amd_fold.h): the matrix folds and dot4 hash their own transcripts. A
// block puts its tile's 128 regions in LDS, and 128 of its threads hash one each into the hit list, so no
// transcript is stored and no hash kernel runs. With RDNA 4's REG2 readout the words are in LDS already, and half the
// block's threads read one region's each. It is on by default in gfx1100, gfx1101 (15 November) and gfx1200 (17
// November) code only, and
// the host and the fold test learn it from the card's caps, and the Windows core leaves it out. No HIP compiler
// runs here, so the C is read as text. These check that selection, that a build without it is the release's, and,
// run as JS for every fold's shape, the gather's index arithmetic: every word of every region lands in LDS once,
// and the thread that hashes a region numbers it as the store path does. Also sweep.sh's trace line, which a race
// reads.

const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');
const { preprocess } = require('./helpers/amdSwitches');

const NATIVE = path.join(__dirname, '..', 'native');
const read = (...f) => fs.readFileSync(path.join(NATIVE, ...f), 'utf8').replace(/\r\n/g, '\n');
const FOLD = read('amd', 'src', 'pearl_amd_fold.h');
const HOST = read('amd', 'src', 'pearl_host_hip.cpp');
const FOLDTEST = read('amd', 'test', 'pearl_amd_foldtest.cpp');
const SWEEP = read('amd', 'test', 'sweep.sh');
const BUILD = read('amd', 'build-amd.sh');
const WORKFLOW = fs.readFileSync(path.join(__dirname, '..', '..', '.github', 'workflows', 'native-core.yml'), 'utf8')
  .replace(/\r\n/g, '\n');

const hasBash = spawnSync('bash', ['--version'], { stdio: 'ignore' }).status === 0;
const bashIt = hasBash ? it : it.skip;

function slice(text, from, to) {
  const at = text.indexOf(from);
  if (at < 0) throw new Error('not found: ' + from);
  return text.slice(at, text.indexOf(to, at + from.length) + to.length);
}
// The C as a build with PEARL_AMD_FUSE at v leaves it, the timing switches off as in a release.
const at = (text, v) => preprocess(text, { PEARL_AMD_FUSE: v, PEARL_AMD_ABLATE: 0, PEARL_AMD_STAMP: 0 });
const fused = (t) => slice(t, '  if constexpr (!kReg2) {   // PEARL_AMD_FUSE (above)', '\n  }\n');
// The REG2 form (RDNA 4's default readout), after the general one.
const fusedReg2 = (t) => slice(t, '#if PEARL_AMD_REG2_ON && PEARL_AMD_QLDS\n  if constexpr (kReg2) {',
  'pearl_amd_hash_region(tm, colIdx * rows_valid + rowIdx, test, hits);\n    }\n  }\n#endif');
const submit = (t) => slice(t, 'extern "C" bool pearl_host_submit(', '\n}\n');

describe('the fused-hash switch (PEARL_AMD_FUSE)', () => {
  // Code without its comments: what the compiler reads.
  const code = (t) => t.split('\n').map((l) => l.replace(/\/\/.*$/, '')).join('\n');

  it('is on by default in gfx1100, gfx1101 and gfx1200 code only, off elsewhere, and takes 0 or 1', () => {
    expect(FOLD).toContain('#ifndef PEARL_AMD_FUSE\n'
      + '#if !PEARL_AMD_EMU && (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1200__))\n'
      + '#define PEARL_AMD_FUSE 1\n'
      + '#else\n#define PEARL_AMD_FUSE 0\n#endif\n#endif\n'
      + '#if PEARL_AMD_FUSE != 0 && PEARL_AMD_FUSE != 1\n#error "PEARL_AMD_FUSE is 0 or 1"\n#endif\n');
    // Only that one place picks it on by default.
    expect(code(FOLD).match(/#define PEARL_AMD_FUSE 1/g)).toHaveLength(1);
  });

  it('says so in the caps, and the host and the fold test read the caps, never the macro', () => {
    const bits = [...FOLD.matchAll(/#define PEARL_AMD_CAP_\w+ (0x[0-9a-f]+|\d+)u/g)].map((m) => Number(m[1]));
    expect(new Set(bits).size).toBe(bits.length);
    expect(FOLD).toContain('#define PEARL_AMD_CAP_FUSE 0x4000u');
    expect(slice(FOLD, 'void pearl_amd_caps(uint32_t *out) {', '\n}\n'))
      .toContain('  if (PEARL_AMD_FUSE) out[0] |= PEARL_AMD_CAP_FUSE;\n');
    // The host pass never sees the gfx12 default, so neither the host nor the fold test may read the macro.
    expect(code(HOST)).not.toContain('PEARL_AMD_FUSE');
    expect(code(FOLDTEST)).not.toContain('PEARL_AMD_FUSE');
    expect(HOST).toContain('  ctx->fuse = (caps[0] & PEARL_AMD_CAP_FUSE) != 0;\n');
    expect(FOLDTEST).toContain('  const bool fuse = (caps[0] & PEARL_AMD_CAP_FUSE) != 0;');
  });

  it('passes every fold the test and hit list, and only a fused card\'s skip the buffer and the hash kernel', () => {
    // Unconditional, so one host launch fits every target's code object.
    expect(FOLD).toContain('\n#define PEARL_AMD_FUSE_PARAMS , [[maybe_unused]] const PearlTranscriptTest test, '
      + '[[maybe_unused]] const PearlHitList hits\n#define PEARL_AMD_FUSE_ARGS , test, hits\n');
    expect(at(FOLD, 0)).toContain('#define PEARL_AMD_FUSE_PARAMS , [[maybe_unused]]');
    expect(FOLD).toContain('uint32_t col0, uint32_t rows_valid, uint32_t *__restrict__ tr PEARL_AMD_FUSE_PARAMS\n');
    for (const p of ['Mfma16', 'Mfma8', 'Wmma11', 'Wmma12', 'Dot4']) {
      expect(FOLD).toContain(`pearl_amd_fold_body<PearlAmd${p}>(Ap, Bp, k, rank, col0, rows_valid, tr PEARL_AMD_FUSE_ARGS);`);
    }
    const s = submit(HOST);
    expect(s).toContain('  const bool fused = ctx->fuse && f.cap != PEARL_AMD_CAP_REF;\n  if (fused) tr = nullptr;\n'
      + '  switch (f.cap) {\n');
    const launches = s.match(/hipLaunchKernelGGL\(pearl_amd_fold_\w+, [^;]*;/g);
    expect(launches.map((l) => l.match(/pearl_amd_fold_(\w+)/)[1]))
      .toEqual(['mfma16', 'mfma8', 'wmma11', 'wmma12', 'dot4', 'ref']);
    for (const l of launches) expect(l.endsWith('ap, bp, k, rank, col0, rv, tr, test, hitList);')).toBe(true);
    expect(s).toContain('  if (!fused)\n    hipLaunchKernelGGL(pearl_amd_hash,');
    // The hit count is cleared and the test and list made before the fold, which may write them.
    expect(s.indexOf('hipMemsetAsync(hitList.count, 0, sizeof(uint32_t), st);')).toBeLessThan(s.indexOf('switch (f.cap)'));
    expect(s.indexOf('PearlHitList hitList;')).toBeLessThan(s.indexOf('switch (f.cap)'));
  });

  it('says where the card hashes in the fold\'s description, which the "mining on GPU" line shows', () => {
    const name = slice(HOST, 'extern "C" const char *pearl_host_fold_name(void *handle) {', '\n}\n');
    expect(name).toContain('const bool fused = ctx->fuse && kFolds[ctx->fold].cap != PEARL_AMD_CAP_REF;');
    expect(name).toContain('kFolds[ctx->fold].text, fused ? "hash in the fold" : "hash in its own kernel"');
    // The table's texts no longer say it, so a card without it reads as it did.
    expect(slice(HOST, 'const FoldInfo kFolds[] = {', '\n};\n')).not.toContain('hash in');
  });

  it('leaves a build without it as the release was', () => {
    const rel = at(FOLD, 0);
    expect(rel).not.toContain('PEARL_AMD_FUSE (above)');
    expect(rel).not.toContain('if (tr == nullptr)');
    // The words are stored as before, and the hash kernel tests them with the code it had (now a function).
    expect(rel).toContain('        w[r] = keepLo[rb][cb][q];\n        w[r + 8] = keepHi[rb][cb][q];\n      }\n');
    expect(slice(rel, 'void pearl_amd_hash(', '\n}\n')).toContain(
      '    tm[4 * q] = w.x; tm[4 * q + 1] = w.y; tm[4 * q + 2] = w.z; tm[4 * q + 3] = w.w;\n  }\n'
      + '  pearl_amd_hash_region(tm, i, test, hits);\n}\n');
    expect(slice(rel, 'void pearl_amd_hash_region(', '\n}\n')).toContain([
      '  if (pearl_transcript_msw(test.key, tm, test.hash_big_endian) > test.target_w[0]) return;',
      '  uint32_t h[8];',
      '  pearl_amd_hash_again(test.key, tm, h);',
      '  if (!pearl_hash_meets_words(h, test.target_w, test.hash_big_endian)) return;',
      '  const uint32_t slot = atomicAdd(hits.count, 1u);',
      '  if (slot >= PEARL_MAX_HITS) return;',
      '  hits.index[slot] = i;',
      '#pragma unroll',
      '  for (int j = 0; j < 8; j++) hits.hash[slot * 8u + j] = h[j];',
      '#pragma unroll',
      '  for (int j = 0; j < 16; j++) hits.transcript[slot * 16u + j] = tm[j];',
    ].join('\n'));
  });

  it('hashes after the slab loop, and returns before the stores only when given no buffer', () => {
    const f = at(FOLD, 1);
    const block = fused(f);
    const body = slice(f, 'void pearl_amd_fold_body(', '\n}\n');
    expect(body.indexOf(block)).toBeGreaterThan(body.indexOf('  // The end of chunk c: its transcript word'));
    expect(body).toContain(block + fusedReg2(f) + '\n  if (tr == nullptr) return;');
    expect(body.indexOf('if (tr == nullptr) return;')).toBeLessThan(body.indexOf('// Each lane writes words r and r + 8'));
    expect(body).not.toContain('if (tr == nullptr) break;');
    // Two barriers: the last slab's reads are done before the words go in the staging, and the words are all
    // there before any thread reads its region.
    expect(block.match(/__syncthreads\(\);/g)).toHaveLength(2);
    expect(block).toContain('uint32_t *const sT = reinterpret_cast<uint32_t *>(&sB[0][0]);');
    expect(block).toContain('static_assert(2 * BN * S >= RR * RC * (int)PEARL_JACKPOT_BUCKETS * 4,');
    expect(block).toContain('pearl_amd_hash_region(tm, colIdx * rows_valid + rowIdx, test, hits);');
  });

  it('in a stamps build skips only the stores, so the stamps are still written', () => {
    const f = preprocess(FOLD, { PEARL_AMD_FUSE: 1, PEARL_AMD_ABLATE: 0, PEARL_AMD_STAMP: 1 });
    const body = slice(f, 'void pearl_amd_fold_body(', '\n}\n');
    expect(body).toContain(fused(f));
    expect(body).not.toContain('if (tr == nullptr) return;');
    const stores = body.indexOf('// Each lane writes words r and r + 8');
    const skip = body.indexOf('        if (tr == nullptr) break;   // no words to store, but the stamps below still are\n'
      + '        const uint32_t roIdx');
    const stamps = body.indexOf('PEARL_AMD_STAMP: the end, then lane 0 of each wave stores its words');
    expect([stores > 0, stores < skip, skip < stamps]).toEqual([true, true, true]);
  });

  it('stays out of the Windows core, before any tuning flags, in its build and its check', () => {
    expect(BUILD).toContain('FUSE_OS=\nif windows; then FUSE_OS=-DPEARL_AMD_FUSE=0; fi\n');
    expect(BUILD).toContain('-Wno-unused-value $KWRAP_OS $SET_OS $FUSE_OS ${PEARL_AMD_CFLAGS:-}"');
    // The check's device pass also takes the target's own flags (arch_flags: gfx1201's wave size), but not on Windows.
    expect(BUILD).toContain(
      '-std=c++17 -O3 $KWRAP_OS $SET_OS $FUSE_OS ${PEARL_AMD_CFLAGS:-} $(windows || arch_flags "$a") $inc --cuda-device-only');
    // The gpu build (Linux, the release's) has no OS flag: the default holds there.
    const gpu = BUILD.slice(BUILD.indexOf('\n  gpu)\n'), BUILD.indexOf('\n  winlib)\n'));
    expect(gpu).not.toContain('FUSE_OS');
  });
});

describe('the fused fold\'s gather, run as JS', () => {
  // Each fold's shape, from its policy.
  function policy(name) {
    const s = slice(FOLD, `struct ${name} {`, 'kRoByLane = ');
    // wmma12's rows of waves are PEARL_AMD_W12_WR, 4 by default (256x256 tiles of 16 waves).
    const n = (k) => {
      const v = new RegExp(`\\b${k} = (\\d+|PEARL_AMD_W12_WR)`).exec(s)[1];
      return v === 'PEARL_AMD_W12_WR' ? 4 : Number(v);
    };
    const ro = /kRoByLane = (true|false);/.exec(FOLD.slice(FOLD.indexOf(`struct ${name} {`)))[1] === 'true';
    return { W: n('kWave'), TM: n('kTM'), TN: n('kTN'), WR: n('kWR'), WC: n('kWC'), MT: n('kMT'), NT: n('kNT'), ro };
  }
  const block = fused(at(FOLD, 1));

  it('is the arithmetic simulated here', () => {
    expect(block).toContain('constexpr int RR = 2 * BM / 32, RC = 4 * BN / 64;');
    expect(block).toContain('const uint32_t co = (uint32_t)((lane >> 1) & 3);');
    expect(block).toContain('const uint32_t lr = 2u * (uint32_t)((wr * WM + rb * 32) / 32) + '
      + '(P::kRoByLane ? (uint32_t)(lane >> 5) : (uint32_t)q);');
    expect(block).toContain('const uint32_t lc = 4u * (uint32_t)((wc * WN + cb * 64) / 64) + co;');
    expect(block).toContain('uint32_t *const o = sT + (lc * RR + lr) * PEARL_JACKPOT_BUCKETS;\n'
      + '          o[r] = keepLo[rb][cb][q];\n          o[r + 8] = keepHi[rb][cb][q];');
    expect(block).toContain('const uint32_t j = (uint32_t)t ^ (((blockIdx.x ^ blockIdx.y) & 1u) << 7);\n'
      + '    if (j < (uint32_t)(RR * RC)) {');
    expect(block).toContain('const uint32_t rowIdx = 2u * (tileRow / 32u) + j % (uint32_t)RR;\n'
      + '      const uint32_t colIdx = 4u * (tileCol / 64u) + j / (uint32_t)RR;');
    // The store path's numbers, which the hashing thread must match.
    const store = slice(FOLD, '  // Each lane writes words r and r + 8 of its regions.', 'PEARL_JACKPOT_BUCKETS;');
    expect(store).toContain('const uint32_t roIdx = P::kRoByLane ? (uint32_t)(lane >> 5) : (uint32_t)q;');
    expect(store).toContain('const uint32_t rowIdx = 2u * ((tileRow + (uint32_t)(wr * WM + rb * 32)) / 32u) + roIdx;');
    expect(store).toContain('const uint32_t colIdx = 4u * ((tileCol + (uint32_t)(wc * WN + cb * 64)) / 64u) + coIdx;');
    expect(FOLD).toContain('const uint32_t r = (uint32_t)((lane & 1) | ((lane >> 2) & 2) | ((lane >> 2) & 4));');
  });

  for (const name of ['PearlAmdMfma16', 'PearlAmdMfma8', 'PearlAmdWmma11', 'PearlAmdWmma12', 'PearlAmdDot4']) {
    it(`puts every word of every region in LDS once, numbered as the store path numbers it: ${name}`, () => {
      const p = policy(name);
      const WM = p.MT * p.TM, WN = p.NT * p.TN, BM = p.WR * WM, BN = p.WC * WN, T = p.W * p.WR * p.WC;
      expect([BM, BN, T]).toEqual(name === 'PearlAmdWmma12' ? [256, 256, 512] : [128, 256, 256]);
      const RB = WM / 32, CB = WN / 64, NRO = p.ro ? 1 : 2, RR = 2 * BM / 32, RC = 4 * BN / 64;
      expect([RR, RC, 2 * RR * RC]).toEqual([BM / 16, 16, T]);
      const rowsValid = 4096;
      for (const [bx, by] of [[0, 0], [3, 5], [15, 1023]]) {
        const tileRow = by * BM, tileCol = bx * BN;
        const slots = new Array(RR * RC * 16).fill(0);
        const number = new Array(RR * RC).fill(null);   // the store path's region number for each LDS region
        for (let t = 0; t < T; t++) {
          const wave = Math.floor(t / p.W), lane = t % p.W, wr = Math.floor(wave / p.WC), wc = wave % p.WC;
          const r = (lane & 1) | ((lane >> 2) & 2) | ((lane >> 2) & 4);
          for (let rb = 0; rb < RB; rb++) {
            for (let cb = 0; cb < CB; cb++) {
              for (let q = 0; q < NRO; q++) {
                const ro = p.ro ? lane >> 5 : q, co = (lane >> 1) & 3;
                const lr = 2 * Math.floor((wr * WM + rb * 32) / 32) + ro;
                const lc = 4 * Math.floor((wc * WN + cb * 64) / 64) + co;
                const j = lc * RR + lr;
                slots[j * 16 + r]++;
                slots[j * 16 + r + 8]++;
                const rowIdx = 2 * Math.floor((tileRow + wr * WM + rb * 32) / 32) + ro;
                const colIdx = 4 * Math.floor((tileCol + wc * WN + cb * 64) / 64) + co;
                const n = colIdx * rowsValid + rowIdx;
                if (number[j] !== null && number[j] !== n) throw new Error(`region ${j}: ${number[j]} and ${n}`);
                number[j] = n;
              }
            }
          }
        }
        expect(slots.every((c) => c === 1)).toBe(true);
        // Each block's hashing threads: one a region, and the number they give it is the store path's.
        const hashed = new Array(RR * RC).fill(0);
        for (let t = 0; t < T; t++) {
          const j = t ^ (((bx ^ by) & 1) << 7);
          if (j >= RR * RC) continue;
          hashed[j]++;
          const rowIdx = 2 * (tileRow / 32) + (j % RR), colIdx = 4 * (tileCol / 64) + Math.floor(j / RR);
          expect(colIdx * rowsValid + rowIdx).toBe(number[j]);
        }
        expect(hashed.every((c) => c === 1)).toBe(true);
      }
    });
  }

  it('fits the transcripts in the B\' staging at every row width the fold builds', () => {
    // S is the slab (32) or the slab plus 16; the B' staging is two buffers of 256 rows.
    // 256 regions in wmma12's 256x256 tile (PEARL_AMD_W12_WR=4), 128 in the others' 128x256.
    for (const S of [32, 48]) expect(2 * 256 * S).toBeGreaterThanOrEqual(256 * 16 * 4);
  });
});

describe('the fused fold\'s REG2 gather (RDNA 4), run as JS', () => {
  const f = at(FOLD, 1);
  const block = fusedReg2(f);
  const store = slice(FOLD, '    // PEARL_AMD_REG2 (above): the lane\'s region is rows', '    return;\n  }\n#endif');

  it('is the arithmetic simulated here, after a barrier, in threads 0 to T / 2 - 1', () => {
    expect(FOLD).toContain('static_assert(!PEARL_AMD_RDLATE && (!PEARL_AMD_FUSE || PEARL_AMD_QLDS), '
      + '"PEARL_AMD_REG2 has the plain readout only (FUSE with QLDS)");');
    expect(block.match(/__syncthreads\(\);/g)).toHaveLength(1);
    expect(block.indexOf('__syncthreads();')).toBeLessThan(block.indexOf('sQ[(uint32_t)p'));
    expect(block).toContain('const uint32_t g = (uint32_t)t;\n    if (g < (uint32_t)(T / 2)) {');
    expect(block).toContain('uint32_t x = sQ[(uint32_t)p * (uint32_t)T + 2u * g + (uint32_t)b];');
    expect(block).toContain('tm[2 * p + b] = x;');
    // A wave holds W / 2 regions: 16 in wave32, 32 in the wave64 fold (PEARL_AMD_W64), whose rows are its own.
    expect(block).toContain(
      'const int gw = (int)(g / (uint32_t)(W / 2)), gl = 2 * (int)(g % (uint32_t)(W / 2));');
    expect(block).toContain('const int gwr = gw / P::kWC, gwc = gw % P::kWC;');
    expect(block).toContain(
      '      const uint32_t rowIdx = kW64 ? 4u * ((tileRow + (uint32_t)(gwr * WM)) / 128u) + 2u * (uint32_t)((gl >> 4) & 1)\n'
      + '                                         + (uint32_t)(gl >> 5)\n'
      + '                                   : 2u * ((tileRow + (uint32_t)(gwr * WM)) / 64u) + (uint32_t)(gl >> 4);\n');
    expect(block).toContain('const uint32_t colIdx = 8u * ((tileCol + (uint32_t)(gwc * WN)) / 64u) + '
      + '(uint32_t)((gl >> 1) & 7);');
    // The rotation is the store path's.
    const rot = 'const uint32_t rl = (13u * ((chunksAll - 1u) >> 4)) & 31u;';
    expect(block).toContain(rot);
    expect(store).toContain(rot);
    expect(block).toContain('x = __builtin_amdgcn_alignbit(x, x, (32u - rl) & 31u);');
    expect(store).toContain('q2[j] = __builtin_amdgcn_alignbit(q2[j], q2[j], (32u - rl) & 31u);');
    // The store path's numbers and word places, which the hashing thread must match.
    expect(store).toContain(
      '    const uint32_t rowIdx = kW64 ? 4u * ((tileRow + (uint32_t)(wr * WM)) / 128u) + 2u * (uint32_t)((lane >> 4) & 1)\n'
      + '                                       + (uint32_t)(lane >> 5)\n'
      + '                                 : 2u * ((tileRow + (uint32_t)(wr * WM)) / 64u) + (uint32_t)(lane >> 4);\n');
    // The shapes the simulation below runs: the wave32 policy's and the wave64 policy's.
    expect(FOLD).toContain('constexpr int W = P::kWave, T = W * P::kWR * P::kWC;\n'
      + '  constexpr int WM = P::kMT * P::kTM, WN = P::kNT * P::kTN;');
    expect(slice(FOLD, 'struct PearlAmdWmma12W64 {', '\n};\n')).toContain(
      '  static constexpr int kWave = 64, kTM = 16, kTN = 16, kTK = 16;\n'
      + '  static constexpr int kWR = PEARL_AMD_W12_WR / 2, kWC = 4, kMT = 8, kNT = 4;');
    expect(slice(FOLD, 'struct PearlAmdWmma12W64 {', '\n};\n')).toContain(
      'static __device__ __forceinline__ int c_row(int lane, int g) { return g + 8 * ((lane / 16) & 1) + 4 * (lane / 32); }');
    expect(FOLD).toContain('#define PEARL_AMD_R4_ROWS_MASK 0x73u');
    expect(store).toContain('const uint32_t colIdx = 8u * ((tileCol + (uint32_t)(wc * WN)) / 64u) + '
      + '(uint32_t)((lane >> 1) & 7);');
    expect(store).toContain('for (int j = 0; j < 8; j++) q2[j] = sQ[((j + turn) & 7u) * (uint32_t)T + (uint32_t)t];');
    expect(store).toContain('w[2u * ((j + turn) & 7u) + (uint32_t)(lane & 1)] = q2[j];');
  });

  it('gives each region\'s hashing thread every word once, in the store path\'s place and number', () => {
    // The wave32 fold (16 waves of 64 x 64), and the wave64 fold (PEARL_AMD_W64: waves of 128 x 64) at its 256x256
    // block of 8 waves and its 128x256 block of 4 (PEARL_AMD_W12_WR 4 and 2).
    const rowsValid = 4096;
    for (const [W, WR, WM, kW64] of [[32, 4, 64, false], [64, 2, 128, true], [64, 1, 128, true]]) {
      const WC = 4, WN = 64, T = W * WR * WC;
      // rowIdx as the fold has it, in the store path (wr, lane) and in the gather (gwr, gl).
      const rowOf = (tileRow, wr, lane) => (kW64
        ? 4 * Math.floor((tileRow + wr * WM) / 128) + 2 * ((lane >> 4) & 1) + (lane >> 5)
        : 2 * Math.floor((tileRow + wr * WM) / 64) + (lane >> 4));
      for (const [tileRow, tileCol] of [[0, 0], [256, 768], [8192 - 256, 2048]]) {
        for (const turn of [0, 3, 7]) {
          // Where the store path puts LDS word sQ[p * T + t]: region number and word index.
          const owner = new Map();
          for (let t = 0; t < T; t++) {
            const wave = Math.floor(t / W), lane = t % W, wr = Math.floor(wave / WC), wc = wave % WC;
            const rowIdx = rowOf(tileRow, wr, lane);
            const colIdx = 8 * Math.floor((tileCol + wc * WN) / 64) + ((lane >> 1) & 7);
            if (kW64) {
              // The wave64 lane's first row (c_row at g 0), and its place among the offsets mask 0x73 leaves free:
              // row bits 2, 3 and 7 up.
              const row = tileRow + wr * WM + 8 * ((lane >> 4) & 1) + 4 * (lane >> 5);
              expect(row & 0x73).toBe(0);
              expect(rowIdx).toBe(((row >> 2) & 3) | ((row >> 7) << 2));
            }
            for (let j = 0; j < 8; j++) {
              const p = (j + turn) & 7;
              owner.set(p * T + t, [colIdx * rowsValid + rowIdx, 2 * p + (lane & 1)]);
            }
          }
          const seen = new Map();
          for (let g = 0; g < T / 2; g++) {
            const gw = Math.floor(g / (W / 2)), gl = 2 * (g % (W / 2)), gwr = Math.floor(gw / WC), gwc = gw % WC;
            const rowIdx = rowOf(tileRow, gwr, gl);
            const colIdx = 8 * Math.floor((tileCol + gwc * WN) / 64) + ((gl >> 1) & 7);
            const n = colIdx * rowsValid + rowIdx;
            expect(seen.has(n)).toBe(false);
            seen.set(n, g);
            for (let p = 0; p < 8; p++) {
              for (let b = 0; b < 2; b++) expect(owner.get(p * T + 2 * g + b)).toEqual([n, 2 * p + b]);
            }
          }
          // Every region the store path writes is hashed.
          expect(new Set([...owner.values()].map((v) => v[0]))).toEqual(new Set(seen.keys()));
        }
      }
    }
  });
});

describe('the fold test and PEARL_AMD_FUSE', () => {
  it('checks a fused card\'s hits against pearl_amd_hash\'s, then times launches that store nothing', () => {
    expect(FOLDTEST.match(/rowsValid, tr, g_test, g_hits\);/g)).toHaveLength(6);
    const main = FOLDTEST.slice(FOLDTEST.indexOf('int main('));
    const check = main.indexOf('launch(f, grid, fa, fb, k, rank, col0, rv, dTr);');
    const gate = main.indexOf('    if (fuse && f.tall) {\n      // The fold\'s hits against pearl_amd_hash\'s');
    const kernel = main.indexOf('hipLaunchKernelGGL(pearl_amd_hash,');
    const timed = main.indexOf('for (int i = 0; i < n; i++) launch(f, grid, fa, fb, k, rank, col0, rv, timedTr);');
    expect([check > 0, check < gate, gate < kernel, kernel < timed]).toEqual([true, true, true, true]);
    expect(main).toContain('(const uint32_t *)dTr, regions, checkTest, kernelHits);');
    expect(main).toContain('if (nf != nk || nk > PEARL_MAX_HITS || hf != hk) {');
    // Only a fused card's timed launches store nothing; the rest store the words as before.
    expect(main).toContain('    uint32_t *timedTr = dTr;\n    if (fuse && f.tall) {');
    expect(main).toContain('      timedTr = nullptr;   // the miner\'s launch: no words stored\n    }\n    g_test = timedTest;');
  });

  it('sets a target about 16 regions meet, at the sizes the fold test runs', () => {
    expect(FOLDTEST).toContain('const uint64_t top = (16ull << 32) / regions;');
    for (const regions of [32 * 16, 64 * 64, 128 * 128, 8192 * 256]) {
      const top = Math.min(0xFFFFFFFF, Math.floor(16 * 2 ** 32 / regions));
      const expected = (top + 1) / 2 ** 32 * regions;
      expect(expected).toBeGreaterThan(15);
      expect(expected).toBeLessThan(17);
    }
  });
});

describe('CI and PEARL_AMD_FUSE', () => {
  it('checks every target with it on, after the release check and without failing the run', () => {
    const release = WORKFLOW.indexOf('        run: ./build-amd.sh check\n');
    const fuseCheck = WORKFLOW.indexOf('run: PEARL_AMD_CFLAGS=-DPEARL_AMD_FUSE=1 ./build-amd.sh check\n');
    const addon = WORKFLOW.indexOf('- name: Build the addon (all 14 targets)');
    expect([release > 0, release < fuseCheck, fuseCheck < addon]).toEqual([true, true, true]);
    expect(WORKFLOW.slice(WORKFLOW.lastIndexOf('- name:', fuseCheck), fuseCheck)).toContain('continue-on-error: true');
    // The release build has it on gfx12 already; no separate fused build.
    expect(WORKFLOW).not.toContain('-DPEARL_AMD_FUSE=1 ./build-amd.sh gpu');
  });
});

describe('sweep.sh with PEARL_AMD_TRACE', () => {
  // A stand-in build-amd.sh whose fold test passes and times anything, and a stand-in hashrate.js that prints
  // the core's timing line on stderr when PEARL_AMD_TRACE is set.
  const STUB = [
    '#!/bin/bash',
    'mkdir -p build',
    'cat > build/pearl_amd_foldtest <<\'FT\'',
    '#!/bin/bash',
    'if [ "$5" = 0 ]; then echo "$6  timed only: $1 x $2, $4 launches, 1.0000 s, 100.000 TMAC/s"; exit 0; fi',
    'echo "$6  ok: $1 x $2, all 16 words match; 1 launches, 0.0100 s, 1.000 TMAC/s"; echo PASS',
    'FT',
    'chmod +x build/pearl_amd_foldtest',
    'echo "${PEARL_AMD_CFLAGS:-}" > build/pearl_core_hip.node',
    '',
  ].join('\n');
  const HASHRATE = [
    'if (process.env.PEARL_AMD_TRACE === \'1\') {',
    '  console.error(\'[pearl-amd] 2.0 s: 9 batches, 18.00 ms each; GPU fold 17.70 ms, hash 0.240 ms\');',
    '  console.error(\'[pearl-amd] 12.0 s: 50 batches, 18.00 ms each; GPU fold 17.71 ms, hash 0.235 ms\');',
    '}',
    'console.log(JSON.stringify({ th: 100 }));',
    '',
  ].join('\n');

  let tmp;
  beforeEach(() => { tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'amd-sweep-fuse-')); });
  afterEach(() => { fs.rmSync(tmp, { recursive: true, force: true }); });

  function sweep(env) {
    const amd = path.join(tmp, 'amd');
    fs.mkdirSync(path.join(amd, 'test'), { recursive: true });
    fs.mkdirSync(path.join(tmp, 'probes'), { recursive: true });
    fs.mkdirSync(path.join(tmp, 'root'), { recursive: true });
    fs.writeFileSync(path.join(amd, 'build-amd.sh'), STUB, { mode: 0o755 });
    fs.writeFileSync(path.join(tmp, 'probes', 'hashrate.js'), HASHRATE);
    fs.writeFileSync(path.join(amd, 'test', 'sweep.sh'), SWEEP.replace('/proc/1/fd/1', '/dev/null'));
    const e = { ...process.env, ARCH: 'gfx1201', SWEEP_ROOT: path.join(tmp, 'root').replace(/\\/g, '/'),
      SWEEP_ROUNDS: '1', SWEEP_FOLD: '', ...env };
    for (const k of ['PEARL_PROFILE_K', 'SWEEP_OPERANDS', 'PEARL_AMD_TRACE']) if (!(k in env)) delete e[k];
    const r = spawnSync('bash', ['test/sweep.sh'], { cwd: amd, encoding: 'utf8', env: e });
    return { status: r.status, lines: r.stdout.split('\n').filter(Boolean).map((l) => l.replace(/^\[amdtest\] \S+ sweep /, '')) };
  }

  bashIt('prints each run\'s last timing line after its hashrate', () => {
    const { status, lines } = sweep({ SWEEP: 'base=;fuse=-DPEARL_AMD_FUSE=1', PEARL_AMD_TRACE: '1' });
    expect(status).toBe(0);
    expect(lines.filter((l) => / (hashrate|trace): /.test(l))).toEqual([
      'base hashrate: {"th":100}',
      'base trace: [pearl-amd] 12.0 s: 50 batches, 18.00 ms each; GPU fold 17.71 ms, hash 0.235 ms',
      'fuse hashrate: {"th":100}',
      'fuse trace: [pearl-amd] 12.0 s: 50 batches, 18.00 ms each; GPU fold 17.71 ms, hash 0.235 ms',
    ]);
  }, 30000);

  bashIt('prints no trace line without it', () => {
    const { status, lines } = sweep({ SWEEP: 'base=' });
    expect(status).toBe(0);
    expect(lines.filter((l) => / (hashrate|trace): /.test(l))).toEqual(['base hashrate: {"th":100}']);
  }, 30000);
});
