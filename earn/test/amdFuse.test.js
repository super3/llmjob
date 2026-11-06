'use strict';

// PEARL_AMD_FUSE (earn/native/amd/src/pearl_amd_fold.h): the matrix folds and dot4 hash their own transcripts. A
// block puts its tile's 128 regions in LDS, and 128 of its threads hash one each into the hit list, so no
// transcript is stored and no hash kernel runs. No HIP compiler runs here, so the C is read as text. These check
// that a release build is unchanged, how the host and the fold test use the switch, and, run as JS for every
// fold's shape, the gather's index arithmetic: every word of every region lands in LDS once, and the thread that
// hashes a region numbers it as the store path does. Also sweep.sh's trace line, which a race reads.

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
  it('is 0 unless a build sets it, and takes 0 or 1', () => {
    expect(FOLD).toContain('#ifndef PEARL_AMD_FUSE\n#define PEARL_AMD_FUSE 0\n#endif\n'
      + '#if PEARL_AMD_FUSE != 0 && PEARL_AMD_FUSE != 1\n#error "PEARL_AMD_FUSE is 0 or 1"\n#endif\n');
  });

  it('leaves a release build\'s folds, hash kernel and host as they were', () => {
    const rel = at(FOLD, 0);
    expect(rel).toContain('#define PEARL_AMD_FUSE_PARAMS\n#define PEARL_AMD_FUSE_ARGS\n\n');
    expect(rel).not.toContain('PEARL_AMD_FUSE (above)');
    expect(rel).not.toContain('if (tr == nullptr) return;');
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
    const s = submit(at(HOST, 0));
    expect(s).toContain('  const bool fused = false;\n#define PEARL_AMD_FUSE_LAUNCH\n  switch (f.cap) {\n');
    expect(s).not.toContain('tr = nullptr');
    expect(s).toContain('  if (!fused)\n    hipLaunchKernelGGL(pearl_amd_hash,');
    expect(at(HOST, 0)).toContain('#define PEARL_AMD_HASH_TEXT "hash in its own kernel"\n\n// The folds');
  });

  it('passes every fold the test and hit list, and only the reference fold a buffer and a hash kernel', () => {
    expect(FOLD).toContain('#define PEARL_AMD_FUSE_PARAMS , [[maybe_unused]] const PearlTranscriptTest test, '
      + '[[maybe_unused]] const PearlHitList hits\n#define PEARL_AMD_FUSE_ARGS , test, hits\n');
    expect(FOLD).toContain('uint32_t col0, uint32_t rows_valid, uint32_t *__restrict__ tr PEARL_AMD_FUSE_PARAMS\n');
    for (const p of ['Mfma16', 'Mfma8', 'Wmma11', 'Wmma12', 'Dot4']) {
      expect(FOLD).toContain(`pearl_amd_fold_body<PearlAmd${p}>(Ap, Bp, k, rank, col0, rows_valid, tr PEARL_AMD_FUSE_ARGS);`);
    }
    const s = submit(at(HOST, 1));
    expect(s).toContain('  const bool fused = f.cap != PEARL_AMD_CAP_REF;\n  if (fused) tr = nullptr;\n'
      + '#define PEARL_AMD_FUSE_LAUNCH , test, hitList\n');
    const launches = s.match(/hipLaunchKernelGGL\(pearl_amd_fold_\w+, [^;]*;/g);
    expect(launches.map((l) => l.match(/pearl_amd_fold_(\w+)/)[1]))
      .toEqual(['mfma16', 'mfma8', 'wmma11', 'wmma12', 'dot4', 'ref']);
    for (const l of launches) expect(l.endsWith('ap, bp, k, rank, col0, rv, tr PEARL_AMD_FUSE_LAUNCH);')).toBe(true);
    // The hit count is cleared and the test and list made before the fold, which now writes them.
    expect(s.indexOf('hipMemsetAsync(hitList.count, 0, sizeof(uint32_t), st);')).toBeLessThan(s.indexOf('switch (f.cap)'));
    expect(s.indexOf('PearlHitList hitList;')).toBeLessThan(s.indexOf('switch (f.cap)'));
    expect(at(HOST, 1)).toContain('#define PEARL_AMD_HASH_TEXT "hash in the fold"\n');
  });

  it('hashes after the slab loop, and returns before the stores only when given no buffer', () => {
    const f = at(FOLD, 1);
    const block = fused(f);
    const body = slice(f, 'void pearl_amd_fold_body(', '\n}\n');
    expect(body.indexOf(block)).toBeGreaterThan(body.indexOf('  // The end of chunk c: its transcript word'));
    expect(body).toContain(block + '  if (tr == nullptr) return;');
    expect(body.indexOf('if (tr == nullptr) return;')).toBeLessThan(body.indexOf('// Each lane writes words r and r + 8'));
    // Two barriers: the last slab's reads are done before the words go in the staging, and the words are all
    // there before any thread reads its region.
    expect(block.match(/__syncthreads\(\);/g)).toHaveLength(2);
    expect(block).toContain('uint32_t *const sT = reinterpret_cast<uint32_t *>(&sB[0][0]);');
    expect(block).toContain('static_assert(2 * BN * S >= RR * RC * (int)PEARL_JACKPOT_BUCKETS * 4,');
    expect(block).toContain('pearl_amd_hash_region(tm, colIdx * rows_valid + rowIdx, test, hits);');
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

describe('the fold test with PEARL_AMD_FUSE', () => {
  const ft = at(FOLDTEST, 1);

  it('checks the fold\'s hits against pearl_amd_hash\'s, then times launches that store nothing', () => {
    expect(ft).toContain('#define PEARL_AMD_FT_FUSE , g_test, g_hits\n');
    expect(ft.match(/rowsValid, tr PEARL_AMD_FT_FUSE\);/g)).toHaveLength(6);
    const main = ft.slice(ft.indexOf('int main('));
    const check = main.indexOf('launch(f, grid, fa, fb, k, rank, col0, rowsValid, dTr);');
    const kernel = main.indexOf('hipLaunchKernelGGL(pearl_amd_hash,');
    const timed = main.indexOf('for (int i = 0; i < n; i++) launch(f, grid, fa, fb, k, rank, col0, rowsValid, timedTr);');
    expect([check > 0, check < kernel, kernel < timed]).toEqual([true, true, true]);
    expect(main).toContain('(const uint32_t *)dTr, regions, checkTest, kernelHits);');
    expect(main).toContain('if (nf != nk || nk > PEARL_MAX_HITS || hf != hk) {');
    expect(main).toContain('      timedTr = nullptr;   // the miner\'s launch: no words stored\n    }\n    g_test = timedTest;');
  });

  it('sets a target about 16 regions meet, at the sizes the fold test runs', () => {
    expect(ft).toContain('const uint64_t top = (16ull << 32) / regions;');
    for (const regions of [32 * 16, 64 * 64, 128 * 128, 8192 * 256]) {
      const top = Math.min(0xFFFFFFFF, Math.floor(16 * 2 ** 32 / regions));
      const expected = (top + 1) / 2 ** 32 * regions;
      expect(expected).toBeGreaterThan(15);
      expect(expected).toBeLessThan(17);
    }
  });

  it('is the release\'s fold test without it', () => {
    const rel = at(FOLDTEST, 0);
    expect(rel).toContain('#define PEARL_AMD_FT_FUSE\n\nvoid launch(');
    expect(rel).not.toContain('pearl_amd_hash,');
    expect(rel).not.toContain('timedTr = nullptr');
  });
});

describe('CI builds the fused variant without shipping it', () => {
  it('checks and builds it before the release build overwrites build/', () => {
    const fuseCheck = WORKFLOW.indexOf('run: PEARL_AMD_CFLAGS=-DPEARL_AMD_FUSE=1 ./build-amd.sh check\n');
    const fuseBuild = WORKFLOW.indexOf('run: PEARL_AMD_CFLAGS=-DPEARL_AMD_FUSE=1 ./build-amd.sh gpu gfx1100 gfx1201\n');
    const release = WORKFLOW.indexOf('- name: Build the addon (all 14 targets)');
    expect([fuseCheck > 0, fuseCheck < fuseBuild, fuseBuild < release]).toEqual([true, true, true]);
    for (const at of [fuseCheck, fuseBuild]) {
      expect(WORKFLOW.slice(WORKFLOW.lastIndexOf('- name:', at), at)).toContain('continue-on-error: true');
    }
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
