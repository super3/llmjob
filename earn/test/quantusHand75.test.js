// The hand-written kernel body for RTX 20 and RTX 30 (native/quantus/cuda/sm75): every step of both permutations
// after the first round, written as SASS by full75.py and put into the cubin sm86/build.py builds. The card checks it
// against the compiled kernel when it opens. These tests model the arithmetic as shipped (build.py's HAND options:
// the products and the layer's fold of imx dmx=1 sq2=1 anch=6, and the constants folded by sum2 kf kt, at the
// end of this file) in BigInt, and pin what the tools rely on.
const fs = require('fs');
const path = require('path');

const cuda = path.join(__dirname, '..', 'native', 'quantus', 'cuda');
const read = (...p) => fs.readFileSync(path.join(cuda, ...p), 'utf8');
const P = (1n << 64n) - (1n << 32n) + 1n;
const F = 1n << 32n;
const M32 = F - 1n;
const M64 = (1n << 64n) - 1n;
const kernel = read('qc_kernel.cuh');
const D = [...Array(12).keys()].map((i) => BigInt(new RegExp(`#define QC_D${i} (0x[0-9a-f]+)ull`).exec(kernel)[1]));
const DP = D.map((d) => (d * F) % P);

const EDGE = [0n, 1n, M32, F, F + 1n, P - 1n, P, M64, M64 - M32, 0xffffffff00000000n];
// A deterministic 64-bit generator, so a failure repeats.
function random(n) {
  const out = [];
  let s = 0x9e3779b97f4a7c15n;
  for (let i = 0; i < n; i++) {
    s = (s * 6364136223846793005n + 1442695040888963407n) & M64;
    out.push(s ^ (s >> 29n));
  }
  return out;
}

// gen75.py's fold, as its four instructions compute it: [L0 - u - T - pa - pb] + [L1 + M0 + u] f.
// slip: the last two instructions wrapped (the documented case, about once in 2^30 products).
function fold(L0, L1, M0, u, T, pa, pb) {
  const s2 = L0 + (M32 - u) + (M32 - T) + (1n - pa) + (1n - pb);
  const lo1 = s2 & M32, k = s2 >> 32n;
  const s3 = L1 + M0 + u + k;
  const hi1 = s3 & M32, c = s3 >> 32n;
  const hi = hi1 - 2n + c, lo = lo1 - c;
  return { v: (lo & M32) + (hi & M32) * F, slip: hi < 0n || hi > M32 || lo < 0n };
}

// The multiply with an anchor (anch): the middle sum's high word is the high product's addend.
function mul(a, b) {
  const [a0, a1, b0, b1] = [a & M32, a >> 32n, b & M32, b >> 32n];
  const m = a1 * b0 + a0 * b1;
  const M2 = m & M64, pm = m >> 64n;
  const L = a0 * b0, Hq = a1 * b1 + (M2 >> 32n);
  expect(Hq <= M64).toBe(true);
  return fold(L & M32, L >> 32n, M2 & M32, Hq & M32, Hq >> 32n, pm, 0n);
}

// The square (sq2 with an anchor): 2 x0 x1 = (2 x0 mod f) x1 + pt x1 f, and the high product is x1 (x1 + pt) + m1.
// wrap: x1 + pt passed 2^32 (x1 = 2^32 - 1 with the doubled-out bit set, odds 2^-32).
function sqr(x) {
  const [x0, x1] = [x & M32, x >> 32n];
  const b = (2n * x0) & M32, pt = (2n * x0) >> 32n;
  const xp = (x1 + pt) & M32;
  const M = b * x1, L = x0 * x0, Hq = x1 * xp + (M >> 32n);
  expect(Hq <= M64).toBe(true);
  return { ...fold(L & M32, L >> 32n, M & M32, Hq & M32, Hq >> 32n, 0n, 0n), wrap: x1 + pt > M32 };
}

// The constant multiply plus a sum (dmx): X = a0 d0 + a1 e0 + L, Y = a0 d1 + a1 e1 + H + cx, with D' = D f = e0 + e1 f
// and cx X's carry out of 2^64; the fold takes - cx and - cy.
function dmul(a, i, L, H) {
  const [a0, a1] = [a & M32, a >> 32n];
  const [d0, d1, e0, e1] = [D[i] & M32, D[i] >> 32n, DP[i] & M32, DP[i] >> 32n];
  const X = a0 * d0 + a1 * e0 + L;
  const cx = X >> 64n;
  const Y = a0 * d1 + a1 * e1 + H + cx;
  const cy = Y >> 64n;
  expect(cx <= 1n && cy <= 1n).toBe(true);
  const [x, y] = [X & M64, Y & M64];
  return fold(x & M32, x >> 32n, y & M32, y >> 32n, 0n, cy, cx);
}

// The external layer's fold of a three-word sum w0 + w1 f + w2 f^2 (imx 4): the count is kept as w2 - 1.
function layerFold(w0, w1, w2) {
  const w2m = (w2 - 1n) & M32;
  const s = w0 + (M32 - w2m);
  const lo = s & M32, pb = s >> 32n;
  const hi = w2m + w1 + pb;
  return { v: lo + (hi & M32) * F };
}

describe('hand body: the arithmetic as shipped', () => {
  const rnd = random(70);

  test('multiply: the product mod p, with no slip on random values', () => {
    for (const a of rnd) {
      for (const b of rnd) {
        const r = mul(a, b);
        expect(r.slip).toBe(false);
        expect(r.v % P).toBe((a * b) % P);
      }
    }
    for (const a of EDGE) {
      for (const b of EDGE.concat(rnd.slice(0, 10))) {
        const r = mul(a, b);
        if (!r.slip) expect(r.v % P).toBe((a * b) % P);
      }
    }
  });

  test('square: the square mod p, with no slip on random values', () => {
    for (const x of random(3000)) {
      const r = sqr(x);
      expect(r.slip || r.wrap).toBe(false);
      expect(r.v % P).toBe((x * x) % P);
    }
    for (const x of EDGE) {
      const r = sqr(x);
      if (!r.slip && !r.wrap) expect(r.v % P).toBe((x * x) % P);
    }
  });

  test('constant multiply plus a sum, for all twelve constants', () => {
    const sums = [0n, 1n, M32, 5n * F + 7n, 12n * F + M32];  // a round's half sums are below 13 * 2^32
    for (let i = 0; i < 12; i++) {
      for (const a of rnd.concat(EDGE)) {
        for (const L of sums) {
          const H = sums[(i + Number(L % 5n)) % 5];
          const r = dmul(a, i, L, H);
          if (EDGE.includes(a) && r.slip) continue;
          expect(r.slip).toBe(false);
          expect(r.v % P).toBe((a * D[i] + L + H * F) % P);
        }
      }
    }
  });

  test('x^7 through two squares and two multiplies', () => {
    for (const x of random(300)) {
      const x2 = sqr(x).v, x3 = mul(x2, x).v, x4 = sqr(x2).v;
      expect(mul(x3, x4).v % P).toBe(x ** 7n % P);
    }
  });

  test('the external layer\'s fold is exact for a count of one or more', () => {
    for (const x of random(300)) {
      const w0 = x & M32, w1 = x >> 32n;
      for (const w2 of [1n, 2n, 5n, 28n]) {
        const r = layerFold(w0, w1, w2);
        if (w1 + w2 - 1n + (w0 >= w2 ? 1n : 0n) > M32) continue;  // the high word passes 2^32: the documented slip
        expect(r.v % P).toBe((w0 + w1 * F + w2 * F * F) % P);
      }
    }
  });

  // full75.py says why this cannot matter: the count is 0 only when 28 elements' worth of terms sum to under 2^64.
  test('a count of zero is the fold\'s one wrong case, and the tool says so', () => {
    const r = layerFold(5n, 9n, 0n);
    expect(r.v).toBe(5n + 8n * F);
    expect(read('sm75/full75.py')).toContain('w2 = 0 is the one case this gets wrong');
  });
});

describe('hand body: the build pins', () => {
  const files = ['sm86/inside.py', 'sm75/gen75.py', 'sm75/full75.py', 'sm75/patch75.py', 'sm75/build75.py', 'sm89/sass.py', 'sm89/enc.py',
    'sm89/gen.py', 'sm89/hand.py', 'sm89/iface.py', 'sm89/patch.py', 'sm89/ref.py'];

  test('build.py ships the options this file models', () => {
    const opts = 'imx=47 dmx=1 sq2=1 sum2=1 kf=1 ylast=1 kt=1 dl=1 ybit=3 bank=1 nsq=1';
    const sm89 = 'imx=63 dmx=1 sq2=1 sum2=1 kf=1 kt=1 sq3=1 sm=1 sbatch=6 bank=1 nsq=1 nmul=1';
    expect(read('sm86', 'build.py')).toContain(`HAND = {"sm_75": "${opts}", "sm_86": "${opts}", "sm_89": "${sm89}"}`);
  });

  // After a merge, build.py embedded the kernel as it was before full75.py rewrote the file, under the hand body's
  // hash: the compiled kernel, right but no faster. The image must be read from the file that is hashed.
  test('build.py embeds the file it hashes, read after every step that rewrites it', () => {
    const py = read('sm86', 'build.py');
    expect(py).toMatch(/\n {4}b = bytearray\(open\(out \+ "\.unverified", "rb"\)\.read\(\)\)\r?\n {4}h = kernel_hash\(out \+ "\.unverified", kernel\)/);
    expect(py.indexOf('h = kernel_hash(out + ".unverified", kernel)')).toBeGreaterThan(py.indexOf('full75.run('));
  });

  // CI used to look for the hash string in the addon, which the bug above passed. sm86/inside.py compares bytes
  // instead, in each of the four Quantus CUDA jobs, and its "MISSING patched sm_NN kernel" line is what to grep for.
  test('CI checks that each addon holds the patched kernels themselves', () => {
    const yml = fs.readFileSync(path.join(__dirname, '..', '..', '.github', 'workflows', 'native-core.yml'), 'utf8');
    const all = 'sm75:sm_75 sm86:sm_86 sm89:sm_89 sm120:sm_120';
    expect(yml).toContain(`python3 ../cuda/sm86/inside.py quantus_cuda.node ${all} || rc=$?`);
    expect(yml).toContain(`python -X utf8 ../cuda/sm86/inside.py quantus_cuda.node ${all} || rc=$?`);
    expect(yml).toContain('python3 ../cuda/sm86/inside.py quantus_cuda_cu13.node sm120:sm_120 || rc=$?');
    expect(yml).toContain('python -X utf8 ../cuda/sm86/inside.py quantus_cuda_cu13.node sm120:sm_120 || rc=$?');
    expect(yml.split('msg="MISSING patched kernel in ').length).toBe(5);
    expect(yml).not.toContain('grep -aqF "$h"');
    const py = read('sm86', 'inside.py');
    expect(py).toContain('if image != open(patched, "rb").read():');
    expect(py).toContain('if build.kernel_hash(patched, kernel) != named:');
    expect(py).toContain('if code == compiled:');
    expect(py).toContain('if image not in addon:');
    expect(py).toContain('print("MISSING patched %s kernel in %s: %s" % (arch, os.path.basename(argv[1]), why))');
    expect(py).toContain('return 1 if missing else 0');
  });

  // CI's Linux job runs sm86/build.py in Rocky Linux 8, whose python3 is 3.6.
  test('the tools keep to what Python 3.6 has', () => {
    for (const f of files) {
      const py = read(f).replace(/"""[\s\S]*?"""/g, '');
      expect([f, py]).not.toEqual([f, expect.stringMatching(/capture_output|\btext=True|\.removeprefix\(|:=|(^|[\s(,=+[])f['"]/m)]);
    }
  });

  test('the kernel has the function full75.py replaces, with its markers', () => {
    expect(kernel).toContain('__device__ __noinline__ QcState qc_hand_loopall(QcState st, u32 z) {');
    expect(kernel).toContain('QC_MARK12(0x1001u)');
    expect(kernel).toContain('QC_MARK12(0x2001u)');
    expect(kernel).toMatch(/#define QC_MARK12\(B\) \\\r?\n/);
    expect(kernel).toMatch(/#ifndef QC_HAND75\r?\n#define QC_HAND75 0\r?\n#endif/);
  });

  test('a uniform register waits 13 cycles after a uniform op (6 read the old pointer on an RTX 2060 SUPER)', () => {
    expect(read('sm75/full75.py')).toContain("lat = (30 if text.startswith('ULDC') else 13) * OPT['ulat']");
  });

  test('the core loads the sm_75 image on compute 7.5 cards only', () => {
    const inc = read('qc_cubin.inc');
    expect(inc).toMatch(/#if defined\(QC_SM75_CUBIN\)\r?\n#include "qc_sm75_cubin.h"\r?\n#endif/);
    expect(inc).toMatch(/if \(major == 7 && minor == 5\) \{\r?\n\s+img\.assign\(\(const char \*\)qc_sm75_image/);
    expect(read('build.sh')).toContain('SM75=(-DQC_SM75_CUBIN=1 -I "$QC_SM75_DIR")');
    expect(read('build.ps1')).toContain("$d += @('-DQC_SM75_CUBIN=1', '-I', (Resolve-Path $Sm75Dir).Path)");
  });

  test('qc_sm75.ref lists toolkits and hashes, each verified on an RTX 20 card', () => {
    const lines = read('sm86', 'qc_sm75.ref').split('\n').map((l) => l.split('#')[0].trim()).filter(Boolean);
    expect(lines.length).toBeGreaterThan(0);
    for (const l of lines) expect(l).toMatch(/^V\d+\.\d+\.\d+ [0-9a-f]{64} RTX 20\S*/);
  });
});

// sum2=1 (RTX 40): the partial rounds' sum of twelve values in 18 instructions. The low words are added three, then
// two at a time; each add's carries are the carry-ins of the matching add of high words, whose own carries are
// counted into H's high word. The low sum's high word is never written.
describe('hand body: the 18-op sum (sum2)', () => {
  function sum2(v) {
    const lo = v.map((x) => x & M32), hi = v.map((x) => x >> 32n);
    const groups = [[0, 1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11]];
    let L0 = 0n, H0 = 0n, H1 = 0n, ops = 0;
    for (const g of groups) {
      const sl = L0 + g.reduce((a, i) => a + lo[i], 0n);
      L0 = sl & M32;
      const sh = H0 + g.reduce((a, i) => a + hi[i], 0n) + (sl >> 32n);
      expect(sl >> 32n).toBeLessThanOrEqual(2n);   // two carry-outs at most
      expect(sh >> 32n).toBeLessThanOrEqual(2n);
      H0 = sh & M32;
      H1 += sh >> 32n;
      ops += 3;
    }
    return { L0, H: H0 + (H1 << 32n), ops };
  }

  test('L.0 + 2^32 H is the sum, in 18 instructions, with H inside the constant multiplies\' bound', () => {
    const all = [...EDGE, ...random(600)];
    for (let k = 0; k + 12 <= all.length; k += 12) {
      const v = all.slice(k, k + 12);
      const r = sum2(v);
      expect(r.ops).toBe(18);
      expect(r.L0 + (r.H << 32n)).toBe(v.reduce((a, x) => a + x, 0n));
      expect(r.H).toBeLessThan(13n * F + 13n);
    }
    const top = sum2(new Array(12).fill((1n << 64n) - 1n));
    expect(top.L0 + (top.H << 32n)).toBe(12n * ((1n << 64n) - 1n));
  });

  test('full75.py writes it that way and keeps it off unless asked', () => {
    const py = read('sm75', 'full75.py');
    expect(py).toMatch(/sum2=0[,)]/);
    expect(py).toContain("idx = [[0, 1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11]]");
    expect(py).toContain("g.emit('IADD3 %L.1, RZ, RZ, RZ')");
    // no chain of a constant multiply is within 13 of its limit: the same carries with 14 * 2^32 as with 13
    const gen = read('sm89', 'gen.py');
    expect(gen).toContain('13 * (1 << 32) >= 1 << 64');
  });
});

// sq3=1 (RTX 40): a full round's S-box hands the layer its output as three words. The fold's two sums and a count
// of the high sum's carries stand for the product plus 2 f; the layer adds 28 elements' worth into an output, so
// its constants are 56 f lower.
describe('hand body: S-box outputs as three words (sq3)', () => {
  test('lo1 + hi1 f + c f^2 is the folded value plus 2 f', () => {
    const all = random(400);
    for (let k = 0; k + 4 <= all.length; k += 4) {
      // any two sums the fold could leave: lo1, hi1 below f, c carries, and the corrections it would have made
      const lo1 = all[k] & M32, hi1 = all[k + 1] & M32, c = all[k + 2] % 3n;
      const folded = (lo1 - c) + (hi1 - 2n + c) * F;
      const three = lo1 + hi1 * F + c * F * F;
      expect(((three - folded - 2n * F) % P + P) % P).toBe(0n);
    }
  });

  test('an output of the layer holds 28 elements, so 56 f comes off its constant', () => {
    const m4 = [[2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]];
    for (const row of m4) expect(row.reduce((a, x) => a + x, 0) * 4).toBe(28);
    const py = read('sm75', 'full75.py');
    expect(py).toMatch(/sq3=0\)/);
    expect(py).toContain('(other.get(o, RC_SEQ[j][i]) - (56 << 32)) % P');
    expect(py).toContain('for j in range(1, 9):');
    expect(py).toContain("assert not C or OPT['imx'] & 4, 'sq3=1 needs imx bit 4'");
  });
});

// kf=1 and kt=1 (full75.py's kf_consts): the partial rounds add their constant to the sum, so every element gets
// it, and the elements enter with offsets that leave the terminal constants behind after the last round. The
// tables in constant bank 3 are rewritten to match. This is that algebra, against the rounds as the spec has them.
describe('hand body: the constants folded into the sum and the layer (kf, kt)', () => {
  const arr = (name) => new RegExp(`__constant__ u64 ${name}\\[[^=]*= \\{([\\s\\S]*?)\\};`).exec(kernel)[1]
    .replace(/ull/g, '').match(/0x[0-9a-f]+|\d+/g).map(BigInt);
  const RCP = arr('QC_RC_P22');
  const RCT0 = arr('QC_RC_T').slice(0, 12);
  const pow = (b, e) => {
    let r = 1n;
    for (let x = b % P; e > 0n; e >>= 1n, x = (x * x) % P) if (e & 1n) r = (r * x) % P;
    return r;
  };
  const sbox = (x) => pow(x, 7n);
  const sum = (v) => v.reduce((a, b) => a + b, 0n);

  // 22 partial rounds as the kernel's reference has them, then the terminal constants on elements 1..11.
  function plain(s0) {
    const s = s0.map((v) => v % P);
    let x = s[0];
    for (let r = 0; r < 22; r++) {
      const y = sbox(x);
      const tot = (y + sum(s.slice(1))) % P;
      x = (D[0] * y + tot + RCP[r]) % P;
      for (let i = 1; i < 12; i++) s[i] = (D[i] * s[i] + tot) % P;
    }
    s[0] = x;
    for (let i = 1; i < 12; i++) s[i] = (s[i] + RCT0[i]) % P;
    return s;
  }

  // The tables kf_consts writes: e the offsets the layer before adds, K the rounds' constants.
  function tables() {
    const d = Array(12).fill(0n);
    const e = Array(12).fill(0n);
    for (let i = 1; i < 12; i++) {
      let acc = 0n;
      for (let r = 0; r < 22; r++) acc = (D[i] * acc + RCP[r]) % P;
      e[i] = ((RCT0[i] % P - acc + P) * pow(pow(D[i], 22n), P - 2n)) % P;
      d[i] = e[i];
    }
    const K = [];
    for (let r = 0; r < 22; r++) {
      K.push((RCP[r] % P - sum(d.slice(1)) % P + P) % P);
      for (let i = 1; i < 12; i++) d[i] = (D[i] * d[i] + RCP[r]) % P;
    }
    return { e, K, last: d };
  }

  function folded(s0, t) {
    const s = s0.map((v, i) => (v + t.e[i]) % P);
    let x = s[0];
    for (let r = 0; r < 22; r++) {
      const y = sbox(x);
      const tot = (y + sum(s.slice(1)) + t.K[r]) % P;
      x = (D[0] * y + tot) % P;
      for (let i = 1; i < 12; i++) s[i] = (D[i] * s[i] + tot) % P;
    }
    s[0] = x;
    return s;
  }

  test('the tables have 22 round constants and 12 terminal ones', () => {
    expect(RCP.length).toBe(22);
    expect(RCT0.length).toBe(12);
  });

  test('the offsets left after the last round are the terminal constants', () => {
    const t = tables();
    for (let i = 1; i < 12; i++) expect(t.last[i]).toBe(RCT0[i] % P);
  });

  test('the folded rounds give the spec\'s state on random and edge states', () => {
    const t = tables();
    const r = random(12 * 6);
    for (let k = 0; k < 6; k++) {
      const s = r.slice(12 * k, 12 * k + 12);
      expect(folded(s, t)).toEqual(plain(s));
    }
    const edge = EDGE.concat([0n, 1n]);
    expect(folded(edge, t)).toEqual(plain(edge));
  });

  test('full75.py writes those tables: offsets into row 4 of the layer table, constants over QC_RC_P22', () => {
    const py = read('sm75/full75.py');
    expect(py).toContain("out[c['seq'] + 8 * (4 * 12 + i)] = (RC_SEQ[4][i] + d[i]) % P");
    expect(py).toContain("out[c['p22'] + 8 * r] = (RC_P22[r] - sum(d[1:])) % P");
    expect(py).toContain('assert all(d[i] == ref.RC_T0[i] for i in range(1, 12))');
  });
});
