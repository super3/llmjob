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

  // Step 2 turned it on in every gfx12 device pass (RDNA 4 mines k 4096); amdCardK.test.js has the rule.
  it('is on for a k past 16 chunks, and refuses what it can\'t build', () => {
    expect(FOLD).toContain('#ifndef PEARL_AMD_KWRAP\n#if PEARL_FOLD_CHUNKS > PEARL_JACKPOT_BUCKETS\n#define PEARL_AMD_KWRAP 1\n');
    expect(FOLD).toContain('#if PEARL_AMD_KWRAP != 0 && PEARL_AMD_KWRAP != 1\n#error');
    expect(FOLD).toContain('#if !PEARL_AMD_KWRAP && PEARL_FOLD_CHUNKS > PEARL_JACKPOT_BUCKETS\n#error');
    expect(FOLD).toContain('#if PEARL_AMD_KWRAP && PEARL_AMD_READOUT_DROP\n#error');
    // A correct build, so the host mines with it; its caps say it has the wrap readout.
    expect(FOLD).toContain('if (PEARL_AMD_KWRAP) out[0] |= PEARL_AMD_CAP_KWRAP;');
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
    const rotate = toJs(grab(wrap, /\n {4}if \((\(c & 15u\) == 0u)\) \{\n/));
    expect(wrap).toContain('            keepLo[rb][cb][q] = pearl_rotl13(keepLo[rb][cb][q]);\n'
      + '            keepHi[rb][cb][q] = pearl_rotl13(keepHi[rb][cb][q]);\n');
    const mine = toJs(grab(wrap, /\n {4}const bool mine = (.*);\n/));
    const hi = toJs(grab(wrap, /\n {4}if \((c & 8u)\) \{\n/));
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
    const keep = toJs(grab(FOLD, /#if PEARL_AMD_KWRAP\n {4}keep = (.*); {3}\/\/ chunk c into word/));
    const run = new Function('rotl13', 'xors',
      `const out = [];
       for (let p = 0; p < 32; p++) { let keep = 0; xors.forEach((x, c) => { keep = (${keep}) >>> 0; }); out.push(keep); }
       return out.slice(0, 16);`).bind(null, rotl13);
    for (const n of [16, 20, 32, 64]) {
      const xors = xorsOf(n);
      expect([n, run(xors)]).toEqual([n, byRule(xors)]);
    }
  });

  // Step 2: the card's caps say whether it has the wrap readout (ctx->kwrap), and a card with it takes up to
  // 64 chunks (k 8192, the most PEARL_AMD_K asks for; RDNA 4 mines k 4096), or the build's PEARL_FOLD_CHUNKS
  // where that is more.
  it('makes the host take more than 16 chunks only from a card with the wrap readout, a multiple of 16', () => {
    const max = grab(HOST, /const uint32_t maxChunks = (.*);\n/);
    const ok = grab(HOST, /const bool chunksOk = (.*);\n/);
    const most = grab(HOST, /#define PEARL_AMD_KWRAP_CHUNKS (.*)\n/);
    expect(max).toBe('ctx->kwrap ? PEARL_AMD_KWRAP_CHUNKS : PEARL_JACKPOT_BUCKETS');
    expect(most).toBe('(PEARL_FOLD_CHUNKS > 64u ? PEARL_FOLD_CHUNKS : 64u)');
    expect(HOST).toMatch(/\|\| !chunksOk\n/);
    expect(HOST).toContain('ctx->kwrap ? " (a multiple of 16 past 16)" : ""');
    const accepts = (kwrap, foldChunks) => new Function('chunks',
      `const ctx = { kwrap: ${kwrap} }, PEARL_FOLD_CHUNKS = ${foldChunks}, PEARL_JACKPOT_BUCKETS = 16;
       const PEARL_AMD_KWRAP_CHUNKS = ${toJs(most)};
       const maxChunks = ${toJs(max).replace('ctx->kwrap', 'ctx.kwrap')};
       return ${toJs(ok)};`);
    const range = (f) => Array.from({ length: 80 }, (_, i) => i + 1).filter(f);
    // A card without the wrap readout: at most 16, as before, whatever the build's k.
    expect(range(accepts(false, 16))).toEqual(range((c) => c <= 16));
    // With it: up to 16, then whole multiples of 16 up to 64 (k 8192 at rank 128), whatever the build's k
    // below that. (The tiled operands also need k a power of two, checked apart.)
    expect(range(accepts(true, 16))).toEqual([...range((c) => c <= 16), 32, 48, 64]);
    expect(range(accepts(true, 32))).toEqual([...range((c) => c <= 16), 32, 48, 64]);
    expect(range(accepts(true, 64))).toEqual([...range((c) => c <= 16), 32, 48, 64]);
    expect(range(accepts(true, 80)).slice(-1)).toEqual([80]);
  });
});

describe('PEARL_PROFILE_K in hashrate.js and verify-hits.js', () => {
  // A stand-in core: it prints the profile and options it was made with, then reports 100 TH/s and no hits.
  const STUB = `module.exports = { createCore(p, o) {
    console.log('PROFILE ' + JSON.stringify(p));
    console.log('OPTS ' + JSON.stringify(o));
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
    const opts = r.stdout.split('\n').find((l) => l.startsWith('OPTS '));
    const last = r.stdout.trim().split('\n').pop();
    return {
      status: r.status, profile: line ? JSON.parse(line.slice(8)) : null, opts: opts ? JSON.parse(opts.slice(5)) : null,
      out: last, err: r.stderr,
    };
  }

  for (const [script, args, done] of [[HASHRATE, ['0.3', '0'], 0], [VERIFY, ['0.3'], 1]]) {
    const name = path.basename(script);

    // Step 2: unset, the core is opened as the app opens it (cardK), and the k is the core's. This stand-in
    // reports none, so it is the profile's. verify-hits.js checks every hit as the app checks an untested
    // card's, so it opens the core with checkHits too; hashrate.js finds no hit and checks none.
    const checks = script === VERIFY ? { checkHits: true } : {};
    it(`${name}: the mainnet profile, opened as the app opens it, when it isn't set`, () => {
      for (const k of [undefined, '']) {
        const r = run(script, args, k);
        expect(r.status).toBe(done);
        expect(JSON.stringify(r.profile)).toBe(JSON.stringify(PROFILE));
        expect(r.opts).toEqual({ cardK: true, ...checks });
        expect(JSON.parse(r.out).k).toBe(2048);
      }
    }, 20000);

    it(`${name}: the profile at that k, without cardK, when it is set`, () => {
      for (const k of [4096, 8192]) {
        const r = run(script, args, String(k));
        expect(r.status).toBe(done);
        expect(r.profile).toEqual({ ...PROFILE, k });
        expect(r.opts).toEqual(checks);
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

  it('verify-hits.js checks every hit at the profile the core mines', () => {
    const src = fs.readFileSync(VERIFY, 'utf8');
    // MAINNET is named only where the profile asked for is chosen; the core's k makes PROFILE, and the job
    // key, proofs, noise and fold use PROFILE.
    expect(src.match(/\bMAINNET\b/g)).toHaveLength(3);
    for (const use of ['addon.createCore(ASKED, kEnv ? { checkHits: true } : { cardK: true, checkHits: true })',
      'const PROFILE = core.k === undefined ? ASKED : profileForK(ASKED, core.k);', 'buildConfig52(PROFILE)',
      'const { k } = PROFILE;', 'buildShareProof(hit, jobKey, PROFILE)', 'checkHit(hit, jobKey, PROFILE)'])
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
  function sweep(env, hash = HASH) {
    const amd = path.join(tmp, 'amd');
    fs.mkdirSync(path.join(amd, 'test'), { recursive: true });
    fs.mkdirSync(path.join(tmp, 'probes'), { recursive: true });
    fs.mkdirSync(path.join(tmp, 'root'), { recursive: true });
    fs.writeFileSync(path.join(amd, 'build-amd.sh'), STUB, { mode: 0o755 });
    fs.writeFileSync(path.join(tmp, 'probes', 'hashrate.js'), hash);
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

  // The fold the host runs on each card (native_fold in pearl_host_hip.cpp): RDNA 2's is dot4, not the reference
  // fold. hashrate.js opens an untested card (gfx9, gfx103x) only with PEARL_AMD_UNTESTED=1, which it gets.
  bashIt('checks and times the fold the host runs, dot4 on RDNA 2, and lets hashrate.js open an untested card', () => {
    const shows = "console.log(JSON.stringify({ th: 100, untested: process.env.PEARL_AMD_UNTESTED || null }));\n";
    for (const [arch, fold] of [['gfx1030', 'dot4'], ['gfx1032', 'dot4'], ['gfx1100', 'wmma11'], ['gfx1201', 'wmma12'],
      ['gfx90a', 'mfma8'], ['gfx942', 'mfma16'], ['gfx1010', 'ref']]) {
      const { status, lines } = sweep({ ARCH: arch, SWEEP_ROUNDS: '1' }, shows);
      expect([arch, status]).toEqual([arch, 0]);
      expect(lines[0]).toMatch(new RegExp(`^start ${arch} fold ${fold}; `));
      expect(lines[1]).toMatch(new RegExp(`^base built in \\d+s \\(\\); check: ${fold} {2}ok: 1024 x 1024`));
      expect(lines[2]).toBe(`base fold 131072 x 4096: ${fold} timed only: 131072 x 4096, 10 launches, 1.0000 s, 100.000 TMAC/s`);
      expect(lines.filter((l) => / hashrate: /.test(l))).toEqual(['base hashrate: {"th":100,"untested":"1"}']);
    }
  }, 60000);

  bashIt('keeps a k build to at least one launch, and reads a k with a u suffix', () => {
    const { status, lines } = sweep({ SWEEP: 'k8192=-O3 -DPEARL_FOLD_K=8192u', SWEEP_FOLD: '131072:4096:3', SWEEP_ROUNDS: '1' });
    expect(status).toBe(0);
    expect(lines.find((l) => l.startsWith('k8192 built'))).toContain('(-O3 -DPEARL_FOLD_K=8192u), k 8192; check:');
    expect(lines).toContain('k8192 fold 131072 x 4096: wmma12 timed only: 131072 x 4096, 1 launches, 1.0000 s, 100.000 TMAC/s');
    expect(lines).toContain('k8192 k=8192 hashrate: {"th":100,"k":"8192"}');
  }, 30000);
});
