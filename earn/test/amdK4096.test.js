'use strict';

// k 4096 on AMD (round 4 item 5), step 1: measurement only. A build with -DPEARL_FOLD_K=4096 turns on
// PEARL_AMD_KWRAP (earn/native/amd/src/pearl_amd_fold.h): the folds put chunk c into transcript word
// c % 16 after a rotl 13, and the host takes 32 chunks. hashrate.js and verify-hits.js take
// PEARL_PROFILE_K, and sweep.sh times a k 4096 build at its own k. No HIP compiler runs here, so the
// fold's rules are read out of the source and run against the JS reference; the scripts run with
// stand-ins for the core.

const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');
const { PROFILE, penalizedAdjustmentFactor, difficultyAdjustmentFactor } = require('../src/shared/miner/pearlhash');
const { foldTranscript, rotl13 } = require('../src/shared/miner/reference');

const NATIVE = path.join(__dirname, '..', 'native');
const read = (...f) => fs.readFileSync(path.join(NATIVE, ...f), 'utf8').replace(/\r\n/g, '\n');
const FOLD = read('amd', 'src', 'pearl_amd_fold.h');
const HOST = read('amd', 'src', 'pearl_host_hip.cpp');
const SWEEP = read('amd', 'test', 'sweep.sh');
const HASHRATE = path.join(NATIVE, 'probes', 'hashrate.js');
const VERIFY = path.join(NATIVE, 'probes', 'verify-hits.js');

const hasBash = spawnSync('bash', ['--version'], { stdio: 'ignore' }).status === 0;
const bashIt = hasBash ? it : it.skip;

function grab(text, re) {
  const m = re.exec(text);
  if (!m) throw new Error('not found: ' + re);
  return m[1];
}
// A C condition or expression of these sources as JS: no u suffixes, and pearl_rotl13 is rotl13.
const toJs = (c) => c.replace(/\b(\d+)u\b/g, '$1').replace(/pearl_rotl13/g, 'rotl13');

let seed = 12345;
function rand() {
  seed ^= seed << 13; seed >>>= 0;
  seed ^= seed >>> 17;
  seed ^= seed << 5; seed >>>= 0;
  return seed;
}
const int = (lo, hi) => lo + (rand() % (hi - lo + 1));

// The protocol's rule, written out on its own: C accumulates over all of k so far, and after chunk c
// the XOR of the tile's 256 sums goes into word c % 16 after that word's rotl 13.
function chunkXors(A, Bt, noiseA, noiseB, k, rank) {
  const out = [];
  const sums = new Int32Array(256);
  for (let c = 0; c < k / rank; c++) {
    let x = 0;
    for (let i = 0; i < 16; i++)
      for (let j = 0; j < 16; j++) {
        let s = sums[i * 16 + j];
        for (let kk = c * rank; kk < (c + 1) * rank; kk++)
          s = (s + (A[i * k + kk] + noiseA[i][kk]) * (Bt[j * k + kk] + noiseB[j][kk])) | 0;
        sums[i * 16 + j] = s;
        x = (x ^ s) >>> 0;
      }
    out.push(x);
  }
  return out;
}
function byRule(xors) {
  const w = new Array(16).fill(0);
  xors.forEach((x, c) => { w[c % 16] = (rotl13(w[c % 16]) ^ x) >>> 0; });
  return w;
}
// Operands and noise whose sums never saturate: |A + noise| <= 127.
function operands(k) {
  const A = Int8Array.from({ length: 16 * k }, () => int(-63, 63));
  const Bt = Int8Array.from({ length: 16 * k }, () => int(-63, 63));
  const noise = () => Array.from({ length: 16 }, () => Int8Array.from({ length: k }, () => int(-64, 64)));
  return { A, Bt, noiseA: noise(), noiseB: noise() };
}
const idx = Array.from({ length: 16 }, (_, i) => i);
const xorsOf = (n) => Array.from({ length: n }, () => rand());

describe('the JS reference at k 4096', () => {
  it('puts chunk c into word c % 16 after a rotl 13, as the protocol says', () => {
    const rank = 128;
    for (const k of [2048, 4096, 8192]) {
      const o = operands(k);
      const xors = chunkXors(o.A, o.Bt, o.noiseA, o.noiseB, k, rank);
      const jp = Array.from(foldTranscript({ ...o, rows: idx, cols: idx, k, rank }));
      expect([k, jp]).toEqual([k, byRule(xors)]);
      if (k === 2048) expect(jp).toEqual(xors);   // 16 chunks: each word once, onto zero
      if (k === 4096) expect(jp).toEqual(idx.map((w) => (rotl13(xors[w]) ^ xors[w + 16]) >>> 0));
    }
  });

  it('matches k 2048 on the first 2048 bytes only before the fold-in', () => {
    const k = 4096, rank = 128;
    const o = operands(k);
    const half = (m) => Int8Array.from({ length: 16 * 2048 }, (_, i) => m[Math.floor(i / 2048) * k + (i % 2048)]);
    const halfN = (n) => n.map((r) => r.subarray(0, 2048));
    const first = Array.from(foldTranscript({ A: half(o.A), Bt: half(o.Bt), noiseA: halfN(o.noiseA),
      noiseB: halfN(o.noiseB), rows: idx, cols: idx, k: 2048, rank }));
    const xors = chunkXors(o.A, o.Bt, o.noiseA, o.noiseB, k, rank);
    expect(first).toEqual(xors.slice(0, 16));
    const jp = Array.from(foldTranscript({ ...o, rows: idx, cols: idx, k, rank }));
    expect(jp).not.toEqual(first);
  });

  it('credits a MAC the same at k 4096 as at k 2048, so TH/s is the measure to compare', () => {
    // A hash costs tile x k MACs, and a pool scales its bound by tile x (k / rank) x 128: the same at rank 128.
    for (const k of [2048, 4096]) {
      const p = { ...PROFILE, k };
      expect([k, penalizedAdjustmentFactor(p)]).toEqual([k, 256 * k]);
      expect([k, difficultyAdjustmentFactor(p)]).toEqual([k, 256 * k]);
    }
    // The core's TH/s counts tile x k MACs a hash at the profile's k, not at a built-in one.
    expect(read('src', 'pearl_config.h')).toContain(
      '#define PEARL_DAF(profile) ((double)(PEARL_ROWS_COUNT * PEARL_COLS_COUNT) * (double)(profile).k)');
  });
});

describe('the AMD folds with PEARL_AMD_KWRAP', () => {
  const wrapAt = FOLD.indexOf('#if PEARL_AMD_KWRAP\n  // PEARL_AMD_KWRAP: chunk c goes into word c % 16');
  const wrap = FOLD.slice(wrapAt, FOLD.indexOf('\n#else\n', wrapAt));

  it('is on for a k past 16 chunks and off by default, and refuses what it can\'t build', () => {
    expect(FOLD).toContain('#ifndef PEARL_AMD_KWRAP\n#define PEARL_AMD_KWRAP (PEARL_FOLD_CHUNKS > PEARL_JACKPOT_BUCKETS)\n#endif');
    expect(FOLD).toContain('#if PEARL_AMD_KWRAP != 0 && PEARL_AMD_KWRAP != 1\n#error');
    expect(FOLD).toContain('#if !PEARL_AMD_KWRAP && PEARL_FOLD_CHUNKS > PEARL_JACKPOT_BUCKETS\n#error');
    expect(FOLD).toContain('#if PEARL_AMD_KWRAP && PEARL_AMD_READOUT_DROP\n#error');
    // A correct build: no cap bit, so the host mines with it.
    expect(FOLD).not.toMatch(/PEARL_AMD_CAP_\w*WRAP/);
  });

  it('keeps today\'s readout, end of tile and reference fold, word for word, in the default build', () => {
    expect(wrapAt).toBeGreaterThan(0);
    const dflt = FOLD.slice(FOLD.indexOf('\n#else\n', wrapAt), FOLD.indexOf('\n#endif\n', wrapAt));
    expect(dflt).toContain('  auto readout = [&](uint32_t c) {\n    const bool mine = (c & 7u) == r, hi = c >= 8u;');
    expect(dflt).toContain('            const uint32_t v = pearl_amd_group_xor(x[q]);\n'
      + '            keepLo[rb][cb][q] = (mine && !hi) ? v : keepLo[rb][cb][q];\n'
      + '            keepHi[rb][cb][q] = (mine && hi) ? v : keepHi[rb][cb][q];\n');
    expect(FOLD).toContain('        w[r] = keepLo[rb][cb][q];\n        w[r + 8] = keepHi[rb][cb][q];');
    expect(FOLD).toContain('#if PEARL_AMD_KWRAP\n    keep = p == (c & 15u) ? pearl_rotl13(keep) ^ x : keep;'
      + '   // chunk c into word c % 16 (PEARL_AMD_KWRAP)\n#else\n    keep = p == c ? x : keep;\n#endif');
    // Two readouts, one for each build, and nothing else in the fold names the switch.
    expect(FOLD.match(/^#if PEARL_AMD_KWRAP$/gm)).toHaveLength(2);
  });

  it('folds chunk c into word c % 16 after a rotl 13, in the lane that keeps that word', () => {
    // The rules, out of the source.
    const rotate = toJs(grab(wrap, /\n    if \((\(c & 15u\) == 0u)\) \{\n/));
    expect(wrap).toContain('            keepLo[rb][cb][q] = pearl_rotl13(keepLo[rb][cb][q]);\n'
      + '            keepHi[rb][cb][q] = pearl_rotl13(keepHi[rb][cb][q]);\n');
    const mine = toJs(grab(wrap, /\n    const bool mine = (.*);\n/));
    const hi = toJs(grab(wrap, /\n    if \((c & 8u)\) \{\n/));
    expect(wrap).toContain('for (int q = 0; q < NRO; q++) keepHi[rb][cb][q] ^= mine ? v[rb][cb][q] : 0u;\n    } else {');
    expect(wrap).toContain('for (int q = 0; q < NRO; q++) keepLo[rb][cb][q] ^= mine ? v[rb][cb][q] : 0u;\n    }\n  };');
    expect(wrap).toContain('for (int q = 0; q < NRO; q++) v[rb][cb][q] = pearl_amd_group_xor(x[q]);');
    // Lane r of a region's eight keeps words r and r + 8 (the end of tile above).
    const lanes = grab(FOLD, /const uint32_t r = \(uint32_t\)\((.*)\);\n/);
    expect(lanes).toBe('(lane & 1) | ((lane >> 2) & 2) | ((lane >> 2) & 4)');
    const run = new Function('rotl13', 'xors',
      `const w = new Array(16).fill(0);
       for (let r = 0; r < 8; r++) {
         let lo = 0, hi = 0;
         xors.forEach((v, c) => {
           if (${rotate}) { lo = rotl13(lo); hi = rotl13(hi); }
           const mine = ${mine};
           if (${hi}) hi = (hi ^ (mine ? v : 0)) >>> 0; else lo = (lo ^ (mine ? v : 0)) >>> 0;
         });
         w[r] = lo; w[r + 8] = hi;
       }
       return w;`).bind(null, rotl13);
    for (const n of [1, 5, 8, 15, 16, 32, 48, 64]) {
      const xors = xorsOf(n);
      expect([n, run(xors)]).toEqual([n, byRule(xors)]);
    }
    // Past 16, only whole passes: the host refuses the rest (below).
    for (const n of [17, 20, 31]) {
      const xors = xorsOf(n);
      expect(run(xors)).not.toEqual(byRule(xors));
    }
  });

  it('folds chunks past 16 the same way in the reference fold', () => {
    const keep = toJs(grab(FOLD, /#if PEARL_AMD_KWRAP\n    keep = (.*);   \/\/ chunk c into word/));
    const run = new Function('rotl13', 'xors',
      `const out = [];
       for (let p = 0; p < 32; p++) { let keep = 0; xors.forEach((x, c) => { keep = (${keep}) >>> 0; }); out.push(keep); }
       return out.slice(0, 16);`).bind(null, rotl13);
    for (const n of [16, 20, 32, 64]) {
      const xors = xorsOf(n);
      expect([n, run(xors)]).toEqual([n, byRule(xors)]);
    }
  });

  it('makes the host take up to the build\'s chunks, a multiple of 16 past 16, and only then', () => {
    const max = grab(HOST, /const uint32_t maxChunks = (.*);\n/);
    const ok = grab(HOST, /const bool chunksOk = (.*);\n/);
    expect(max).toBe('PEARL_AMD_KWRAP ? PEARL_FOLD_CHUNKS : PEARL_JACKPOT_BUCKETS');
    expect(HOST).toMatch(/\|\| !chunksOk\n/);
    expect(HOST).toContain('PEARL_AMD_KWRAP ? " (a multiple of 16 past 16)" : ""');
    const accepts = (kwrap, foldChunks) => new Function('chunks',
      `const PEARL_AMD_KWRAP = ${kwrap}, PEARL_FOLD_CHUNKS = ${foldChunks}, PEARL_JACKPOT_BUCKETS = 16;
       const maxChunks = ${max};
       return ${toJs(ok)};`);
    const range = (f) => Array.from({ length: 80 }, (_, i) => i + 1).filter(f);
    // Today's build: at most 16, as before.
    expect(range(accepts(0, 16))).toEqual(range((c) => c <= 16));
    // The readout alone at k 2048: the same.
    expect(range(accepts(1, 16))).toEqual(range((c) => c <= 16));
    // A k 4096 build: k 2048 and k 4096 at rank 128, and nothing in between.
    expect(range(accepts(1, 32))).toEqual([...range((c) => c <= 16), 32]);
    expect(range(accepts(1, 64))).toEqual([...range((c) => c <= 16), 32, 48, 64]);
  });
});

describe('PEARL_PROFILE_K in hashrate.js and verify-hits.js', () => {
  // A stand-in core: it prints the profile it was made with, then reports 100 TH/s and no hits.
  const STUB = `module.exports = { createCore(p) {
    console.log('PROFILE ' + JSON.stringify(p));
    const on = {}; let t = null;
    return { device: 'stub', on(ev, cb) { on[ev] = cb; },
      setJob() { t = setInterval(() => on.hashrate && on.hashrate(100), 10); },
      stop() { clearInterval(t); } };
  } };\n`;
  let tmp, stub;
  beforeAll(() => {
    tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'amd-k4096-'));
    stub = path.join(tmp, 'stub.js');
    fs.writeFileSync(stub, STUB);
  });
  afterAll(() => fs.rmSync(tmp, { recursive: true, force: true }));

  function run(script, args, k) {
    const env = { ...process.env };
    delete env.PEARL_PROFILE_K;
    if (k !== undefined) env.PEARL_PROFILE_K = k;
    const r = spawnSync(process.execPath, [script, stub, ...args], { env, encoding: 'utf8' });
    const line = r.stdout.split('\n').find((l) => l.startsWith('PROFILE '));
    const last = r.stdout.trim().split('\n').pop();
    return { status: r.status, profile: line ? JSON.parse(line.slice(8)) : null, out: last, err: r.stderr };
  }

  for (const [script, args, done] of [[HASHRATE, ['0.3', '0'], 0], [VERIFY, ['0.3'], 1]]) {
    const name = path.basename(script);

    it(`${name}: the mainnet profile, unchanged, when it isn't set`, () => {
      for (const k of [undefined, '']) {
        const r = run(script, args, k);
        expect(r.status).toBe(done);
        expect(JSON.stringify(r.profile)).toBe(JSON.stringify(PROFILE));
        expect(JSON.parse(r.out)).not.toHaveProperty('k');
      }
    }, 20000);

    it(`${name}: the profile at that k when it is set`, () => {
      for (const k of [4096, 8192]) {
        const r = run(script, args, String(k));
        expect(r.status).toBe(done);
        expect(r.profile).toEqual({ ...PROFILE, k });
        expect(JSON.parse(r.out).k).toBe(k);
      }
    }, 20000);

    it(`${name}: refuses a k the protocol or the operand trees can't take, before making a core`, () => {
      for (const k of ['1024', '3072', '6144', '4096.5', 'abc', '-4096', '0']) {
        const r = run(script, args, k);
        expect([k, r.status, r.profile]).toEqual([k, 2, null]);
        expect(r.out + r.err).toContain(`PEARL_PROFILE_K must be a power of two of at least 2048, not ${k}`);
      }
    }, 20000);
  }

  it('verify-hits.js checks every hit at the profile it made the core with', () => {
    const src = fs.readFileSync(VERIFY, 'utf8');
    // MAINNET is named only where the profile is chosen; the job key, proofs, noise and fold use PROFILE.
    expect(src.match(/\bMAINNET\b/g)).toHaveLength(3);
    for (const use of ['addon.createCore(PROFILE, {})', 'buildConfig52(PROFILE)', 'const { k, rank, m, n } = PROFILE;',
      'buildShareProof(hit, jobKey, PROFILE)', 'regionToTile(hit.nonce, PROFILE)'])
      expect(src).toContain(use);
  });
});

describe('sweep.sh with a k 4096 variant', () => {
  // A stand-in build-amd.sh whose fold test answers by its flags (base 100 TMAC/s, k 4096 101) and
  // echoes its launches, and a stand-in hashrate.js that shows the PEARL_PROFILE_K it was given.
  const STUB = [
    '#!/bin/bash',
    'mkdir -p build',
    'printf \'#!/bin/bash\\nf="%s"\\n\' "${PEARL_AMD_CFLAGS:-}" > build/pearl_amd_foldtest',
    'cat >> build/pearl_amd_foldtest <<\'FT\'',
    'case "$f" in *PEARL_FOLD_K=4096*) t=101.000 ;; *) t=100.000 ;; esac',
    'if [ "$5" = 0 ]; then echo "$6  timed only: $1 x $2, $4 launches, 1.0000 s, $t TMAC/s"; exit 0; fi',
    'echo "$6  ok: $1 x $2, all 16 words match; 1 launches, 0.0100 s, 1.000 TMAC/s"; echo PASS',
    'FT',
    'chmod +x build/pearl_amd_foldtest',
    'echo "${PEARL_AMD_CFLAGS:-}" > build/pearl_core_hip.node',
    'echo "build ${PEARL_AMD_CFLAGS:-}" >> builds.log',
    '',
  ].join('\n');
  const HASH = [
    "const f = require('fs').readFileSync(process.argv[2], 'utf8');",
    "console.log(JSON.stringify({ th: /FOLD_K=4096/.test(f) ? 101 : 100, k: process.env.PEARL_PROFILE_K === undefined ? null : process.env.PEARL_PROFILE_K }));",
    '',
  ].join('\n');

  let tmp;
  beforeEach(() => { tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'amd-sweep-k-')); });
  afterEach(() => { fs.rmSync(tmp, { recursive: true, force: true }); });

  // Runs a copy of sweep.sh with the stand-ins, writing nothing outside tmp: the container-log copy of
  // each line goes to /dev/null.
  function sweep(env) {
    const amd = path.join(tmp, 'amd');
    fs.mkdirSync(path.join(amd, 'test'), { recursive: true });
    fs.mkdirSync(path.join(tmp, 'probes'));
    fs.mkdirSync(path.join(tmp, 'root'));
    fs.writeFileSync(path.join(amd, 'build-amd.sh'), STUB, { mode: 0o755 });
    fs.writeFileSync(path.join(tmp, 'probes', 'hashrate.js'), HASH);
    fs.writeFileSync(path.join(amd, 'test', 'sweep.sh'), SWEEP.replace('/proc/1/fd/1', '/dev/null'));
    const e = { ...process.env, ARCH: 'gfx1201', SWEEP_ROOT: path.join(tmp, 'root').replace(/\\/g, '/'), ...env };
    delete e.PEARL_PROFILE_K;
    const r = spawnSync('bash', ['test/sweep.sh'], { cwd: amd, encoding: 'utf8', env: e });
    const log = path.join(amd, 'builds.log');
    return {
      status: r.status,
      lines: r.stdout.split('\n').filter(Boolean).map((l) => l.replace(/^\[amdtest\] \S+ sweep /, '')),
      builds: fs.existsSync(log) ? fs.readFileSync(log, 'utf8').split('\n').filter(Boolean).map((l) => l.trimEnd()) : [],
    };
  }

  bashIt('checks and times each build at its own k: half the launches, and hashrate.js at k 4096', () => {
    const { status, lines, builds } = sweep({
      SWEEP: 'base=;k4096=-DPEARL_FOLD_K=4096', SWEEP_FOLD: '131072:4096:100', SWEEP_FOLD_ROUNDS: '4',
      SWEEP_ROUNDS: '4', SWEEP_HASH: '60:20',
    });
    expect(status).toBe(0);
    expect(builds).toEqual(['build', 'build -DPEARL_FOLD_K=4096', 'build']);
    expect(lines[0]).toBe('start gfx1201 fold wmma12; variants: base=;k4096=-DPEARL_FOLD_K=4096; '
      + 'fold timing 131072:4096:100 x 4 rounds in turn; hashrate 60s+20s x 4');
    expect(lines.find((l) => l.startsWith('base built'))).toMatch(/\(\); check: wmma12 {2}ok: 1024 x 1024, all 16 words match/);
    expect(lines.find((l) => l.startsWith('k4096 built'))).toMatch(
      /\(-DPEARL_FOLD_K=4096\), k 4096; check: wmma12 {2}ok: 1024 x 1024, all 16 words match/);
    expect(lines).toContain('base fold 131072 x 4096, round 2 of 4: wmma12 timed only: 131072 x 4096, 100 launches, 1.0000 s, 100.000 TMAC/s');
    expect(lines).toContain('k4096 fold 131072 x 4096, round 2 of 4: wmma12 timed only: 131072 x 4096, 50 launches, 1.0000 s, 101.000 TMAC/s');
    expect(lines.filter((l) => l.startsWith('fold 131072 x 4096, round'))).toEqual([1, 2, 3, 4].map((i) =>
      `fold 131072 x 4096, round ${i} of 4, TMAC/s: base 100.000, k4096 101.000 (+1.00%)`));
    const hash = lines.filter((l) => / hashrate: /.test(l));
    expect(hash).toHaveLength(8);
    expect(hash.slice(0, 2)).toEqual(['base hashrate: {"th":100,"k":null}', 'k4096 k=4096 hashrate: {"th":101,"k":"4096"}']);
    expect(lines[lines.length - 1]).toBe('done');
  }, 30000);

  bashIt('does what it did before for builds that don\'t set PEARL_FOLD_K', () => {
    const { status, lines } = sweep({ SWEEP: 'base=;tiled0=-DPEARL_AMD_TILED=0', SWEEP_ROUNDS: '1' });
    expect(status).toBe(0);
    expect(lines[0]).toBe('start gfx1201 fold wmma12; variants: base=;tiled0=-DPEARL_AMD_TILED=0; fold timing 131072:4096:10; '
      + 'hashrate 40s+10s x 1');
    // Each fold timed once, right after its build, at the launches given.
    expect(lines[1]).toMatch(/^base built in \d+s \(\); check: /);
    expect(lines[2]).toBe('base fold 131072 x 4096: wmma12 timed only: 131072 x 4096, 10 launches, 1.0000 s, 100.000 TMAC/s');
    expect(lines[4]).toBe('tiled0 fold 131072 x 4096: wmma12 timed only: 131072 x 4096, 10 launches, 1.0000 s, 100.000 TMAC/s');
    expect(lines.filter((l) => / hashrate: /.test(l))).toEqual(['base hashrate: {"th":100,"k":null}',
      'tiled0 hashrate: {"th":100,"k":null}']);
  }, 30000);

  bashIt('keeps a k build to at least one launch, and reads a k with a u suffix', () => {
    const { status, lines } = sweep({ SWEEP: 'k8192=-O3 -DPEARL_FOLD_K=8192u', SWEEP_FOLD: '131072:4096:3', SWEEP_ROUNDS: '1' });
    expect(status).toBe(0);
    expect(lines.find((l) => l.startsWith('k8192 built'))).toContain('(-O3 -DPEARL_FOLD_K=8192u), k 8192; check:');
    expect(lines).toContain('k8192 fold 131072 x 4096: wmma12 timed only: 131072 x 4096, 1 launches, 1.0000 s, 100.000 TMAC/s');
    expect(lines).toContain('k8192 k=8192 hashrate: {"th":100,"k":"8192"}');
  }, 30000);
});
