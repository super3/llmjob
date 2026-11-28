// The hand-written kernel body for RTX 20 and RTX 30 (native/quantus/cuda/sm75): every step of both permutations
// after the first round, written as SASS by full75.py and put into the cubin sm86/build.py builds. The card checks it
// against the compiled kernel when it opens; these tests pin what the tools rely on and the arithmetic's identities.
const fs = require('fs');
const path = require('path');

const cuda = path.join(__dirname, '..', 'native', 'quantus', 'cuda');
const read = (...p) => fs.readFileSync(path.join(cuda, ...p), 'utf8');
const P = (1n << 64n) - (1n << 32n) + 1n;
const F = 1n << 32n;
const M32 = F - 1n;

// A deterministic 64-bit generator, so a failure repeats.
function* values(n) {
  let s = 0x9e3779b97f4a7c15n;
  const edge = [0n, 1n, M32, F, F + 1n, P - 1n, P, (1n << 64n) - 1n, (1n << 64n) - F, 0xffffffff00000000n];
  for (const e of edge) yield e;
  for (let i = 0; i < n; i++) {
    s = (s * 6364136223846793005n + 1442695040888963407n) & ((1n << 64n) - 1n);
    yield s ^ (s >> 29n);
  }
}

// gen75.py's fold, as its four instructions compute it: [L0 - u - T - pa - pb] + [L1 + M0 + u] f.
function fold(L0, L1, M0, u, T, pa, pb) {
  const s2 = L0 + (M32 - u) + (M32 - T) + (1n - pa) + (1n - pb);
  const lo1 = s2 & M32, k = s2 >> 32n;
  const s3 = L1 + M0 + u + k;
  const hi1 = s3 & M32, c = s3 >> 32n;
  const hi = hi1 - 2n + c, lo = lo1 - c;
  return { lo, hi, slip: hi < 0n || hi > M32 || lo < 0n };
}

function mul(a, b) {
  const [a0, a1, b0, b1] = [a & M32, a >> 32n, b & M32, b >> 32n];
  const m = a1 * b0 + a0 * b1;
  const M2 = m & ((1n << 64n) - 1n), pm = m >> 64n;
  const L = a0 * b0, H = a1 * b1;
  const U = (H & M32) + (M2 >> 32n);
  return fold(L & M32, L >> 32n, M2 & M32, U & M32, H >> 32n, pm, U >> 32n);
}

// sq2: 2 x0 x1 = (2 x0 mod f) x1 + pt x1 f, and the high product is x1 (x1 + pt).
function sqr(x) {
  const [x0, x1] = [x & M32, x >> 32n];
  const b = (2n * x0) & M32, pt = (2n * x0) >> 32n;
  const xp = (x1 + pt) & M32;
  const M = b * x1, L = x0 * x0, H = x1 * xp;
  const U = (H & M32) + (M >> 32n);
  return { ...fold(L & M32, L >> 32n, M & M32, U & M32, H >> 32n, U >> 32n, 0n), wrap: x1 + pt > M32 };
}

describe('hand body: the arithmetic', () => {
  test('the fold of a product is the product mod p, but for a slip it reports', () => {
    let slips = 0, n = 0;
    const v = [...values(60)];
    for (const a of v) {
      for (const b of v) {
        const r = mul(a, b);
        n++;
        if (r.slip) slips++;
        else expect((r.lo + r.hi * F) % P).toBe((a * b) % P);
      }
    }
    expect(slips).toBeLessThan(n / 50);  // the edge values hit it; random ones about once in 2^30
  });

  test('the three-product square is the square mod p', () => {
    for (const x of values(400)) {
      const r = sqr(x);
      if (!r.slip && !r.wrap) expect((r.lo + r.hi * F) % P).toBe((x * x) % P);
    }
  });

  test('the external layer\'s fold: (w0 - w2) + (w1 + w2) f for a three-word sum', () => {
    for (const x of values(200)) {
      for (const w2 of [0n, 1n, 5n, 28n]) {
        const w0 = x & M32, w1 = x >> 32n;
        const pb = w0 >= w2 ? 1n : 0n;
        const lo = (w0 - w2 + F) & M32, hi = w1 + w2 - 1n + pb;
        if (hi <= M32 && hi >= 0n) expect((lo + hi * F) % P).toBe((w0 + w1 * F + w2 * F * F) % P);
      }
    }
  });
});

describe('hand body: the build pins', () => {
  const files = ['sm75/gen75.py', 'sm75/full75.py', 'sm75/patch75.py', 'sm75/build75.py', 'sm89/sass.py', 'sm89/enc.py',
    'sm89/gen.py', 'sm89/hand.py', 'sm89/iface.py', 'sm89/patch.py', 'sm89/ref.py'];

  // CI's Linux job runs sm86/build.py in Rocky Linux 8, whose python3 is 3.6.
  test('the tools keep to what Python 3.6 has', () => {
    for (const f of files) {
      const py = read(f).replace(/"""[\s\S]*?"""/g, '');
      expect([f, py]).not.toEqual([f, expect.stringMatching(/capture_output|\btext=True|\.removeprefix\(|:=|(^|[\s(,=+[])f['"]/m)]);
    }
  });

  test('the kernel has the function full75.py replaces, with its markers', () => {
    const k = read('qc_kernel.cuh');
    expect(k).toContain('__device__ __noinline__ QcState qc_hand_loopall(QcState st, u32 z) {');
    expect(k).toContain('QC_MARK12(0x1001u)');
    expect(k).toContain('QC_MARK12(0x2001u)');
    expect(k).toMatch(/#ifndef QC_HAND75\r?\n#define QC_HAND75 0\r?\n#endif/);
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
