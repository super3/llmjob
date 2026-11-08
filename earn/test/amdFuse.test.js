'use strict';

// PEARL_AMD_FUSE (earn/native/amd/src/pearl_amd_fold.h): the matrix folds and dot4 hash their own transcripts. A
// block puts its tile's 128 regions in LDS, and 128 of its threads hash one each into the hit list, so no
// transcript is stored and no hash kernel runs. It is off by default (it lost on top of #287 items 4 and 5), and
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
const fused = (t) => slice(t, '  {   // PEARL_AMD_FUSE (above)', '\n  }\n');
const submit = (t) => slice(t, 'extern "C" bool pearl_host_submit(', '\n}\n');

describe('the fused-hash switch (PEARL_AMD_FUSE)', () => {
  // Code without its comments: what the compiler reads.
  const code = (t) => t.split('\n').map((l) => l.replace(/\/\/.*$/, '')).join('\n');

  it('is off by default on every target (it lost on top of items 4 and 5), and takes 0 or 1', () => {
    expect(FOLD).toContain('#ifndef PEARL_AMD_FUSE\n#define PEARL_AMD_FUSE 0\n#endif\n'
      + '#if PEARL_AMD_FUSE != 0 && PEARL_AMD_FUSE != 1\n#error "PEARL_AMD_FUSE is 0 or 1"\n#endif\n');
    // No target picks it on by default.
    expect(code(FOLD)).not.toMatch(/#define PEARL_AMD_FUSE 1/);
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
    expect(body).toContain(block + '  if (tr == nullptr) return;');
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
    expect(BUILD).toContain('-std=c++17 -O3 $KWRAP_OS $SET_OS $FUSE_OS ${PEARL_AMD_CFLAGS:-} $inc --cuda-device-only');
    // The gpu build (Linux, the release's) has no OS flag: the default holds there.
    const gpu = BUILD.slice(BUILD.indexOf('\n  gpu)\n'), BUILD.indexOf('\n  winlib)\n'));
    expect(gpu).not.toContain('FUSE_OS');
  });
});

describe('the fused fold\'s gather, run as JS', () => {
  // Each fold's shape, from its policy.
  function policy(name) {
    const s = slice(FOLD, `struct ${name} {`, 'kRoByLane = ');
    const n = (k) => Number(new RegExp(`\\b${k} = (\\d+)`).exec(s)[1]);
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
      expect([BM, BN, T]).toEqual([128, 256, 256]);
      const RB = WM / 32, CB = WN / 64, NRO = p.ro ? 1 : 2, RR = 2 * BM / 32, RC = 4 * BN / 64;
      expect([RR, RC, 2 * RR * RC]).toEqual([8, 16, T]);
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
    for (const S of [32, 48]) expect(2 * 256 * S).toBeGreaterThanOrEqual(128 * 16 * 4);
  });
});

describe('the fold test and PEARL_AMD_FUSE', () => {
  it('checks a fused card\'s hits against pearl_amd_hash\'s, then times launches that store nothing', () => {
    expect(FOLDTEST.match(/rowsValid, tr, g_test, g_hits\);/g)).toHaveLength(6);
    const main = FOLDTEST.slice(FOLDTEST.indexOf('int main('));
    const check = main.indexOf('launch(f, grid, fa, fb, k, rank, col0, rowsValid, dTr);');
    const gate = main.indexOf('    if (fuse && f.tall) {\n      // The fold\'s hits against pearl_amd_hash\'s');
    const kernel = main.indexOf('hipLaunchKernelGGL(pearl_amd_hash,');
    const timed = main.indexOf('for (int i = 0; i < n; i++) launch(f, grid, fa, fb, k, rank, col0, rowsValid, timedTr);');
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
