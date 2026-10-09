'use strict';

// The SYCL core for Intel GPUs (native/sycl) cannot be built or run here, so,
// as nativeConfig.test.js does for the CUDA core, these tests read its sources
// and pin what must agree with the rest of the repo: the C API pearl_core.cc
// calls, signature for signature; the structs the two files share; and the
// index arithmetic the folds use to place each region, evaluated against the
// protocol's regionToTile. They also run the harness scripts that need no
// device. Whether the folds compute the right transcripts is checked by
// running them (pearl_sycl_check and verify-hits.js on the SYCL CPU device).

const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');
const { PROFILE, regionToTile } = require('../src/shared/miner/pearlhash');

const NATIVE = path.join(__dirname, '..', 'native');
const SYCL = path.join(NATIVE, 'sycl');
// CRLF to LF: a Windows checkout gives the C++ sources CRLF endings, and the
// line-based helpers below read a trailing \r as part of the line.
const read = (...p) => fs.readFileSync(path.join(...p), 'utf8').replace(/\r\n/g, '\n');
const CORE = read(NATIVE, 'src', 'pearl_core.cc');
const HOST = read(SYCL, 'pearl_sycl_host.cpp');
const KERNELS = read(SYCL, 'pearl_sycl_kernels.hpp');

const stripComments = (src) => src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/.*$/gm, '');

// The body of `struct Name { ... };` with comments and whitespace removed.
function structBody(src, name) {
  const at = src.indexOf('struct ' + name + ' {');
  if (at < 0) return null;
  const end = src.indexOf('};', at);
  return src.slice(at, end)
    .split('\n').map((l) => l.replace(/\/\/.*$/, '').trim()).filter(Boolean).join(' ');
}

// One C declaration or definition as "ret name(type,type)": whitespace and
// parameter names dropped, so a declaration and a definition compare equal
// exactly when their types do.
function signature(ret, name, params) {
  const squash = (t) => t.replace(/\s+/g, ' ').replace(/\s*\*\s*/g, '*').trim();
  const types = params.split(',').map(squash).filter(Boolean)
    .map((p) => p.replace(/^(.*?[\w*])(?: |(?<=\*))[A-Za-z_]\w*$/, '$1'));
  return squash(ret) + ' ' + name + '(' + types.join(',') + ')';
}

// pearl_host_* name -> signature, from pearl_core.cc's extern "C" block.
function declaredApi() {
  const block = stripComments(CORE.slice(CORE.indexOf('extern "C" {'), CORE.indexOf('namespace {')))
    .replace('extern "C" {', '');
  const out = new Map();
  for (const decl of block.split(';')) {
    const m = /^\s*([\s\S]*?)\b(pearl_host_\w+)\s*\(([\s\S]*)\)\s*$/.exec(decl);
    if (m) out.set(m[2], signature(m[1], m[2], m[3]));
  }
  return out;
}

// pearl_host_* name -> signature, from the SYCL host's extern "C" definitions.
function definedApi() {
  const out = new Map();
  const re = /extern "C"\s+([^;{}()]*?)\b(pearl_host_\w+)\s*\(([^)]*)\)\s*\{/g;
  let m;
  while ((m = re.exec(stripComments(HOST)))) out.set(m[2], signature(m[1], m[2], m[3]));
  return out;
}

describe('SYCL core sources', () => {
  test('define every pearl_host_* function pearl_core.cc declares, with the same types', () => {
    const declared = declaredApi();
    const defined = definedApi();
    expect(declared.size).toBeGreaterThanOrEqual(10);
    for (const [name, sig] of declared) expect(defined.get(name)).toBe(sig);
  });

  test('read signatures by type, not by parameter name', () => {
    expect(signature('void *', 'f', 'const PearlProfile *profile, char *err, size_t err_len'))
      .toBe('void* f(const PearlProfile*,char*,size_t)');
    expect(signature('void', 'f', 'void *')).toBe('void f(void*)');
    expect(signature('bool', 'f', 'uint64_t nonce_base, uint32_t batch'))
      .not.toBe(signature('bool', 'f', 'uint64_t nonce_base, uint64_t batch'));
  });

  test('share the result structs with pearl_core.cc member for member', () => {
    for (const name of ['PearlProofSide', 'PearlSearchResult']) {
      const want = structBody(CORE, name);
      expect(want).not.toBeNull();
      expect(structBody(HOST, name)).toBe(want);
    }
  });
});

// The folds' index arithmetic, taken from pearl_sycl_kernels.hpp and run here.
// A fold works on windows of 32 rows by 64 columns, 8 regions each. Each work-
// item (dot) or lane (xmx) accumulates some rows by some columns and its sums
// end up in one region's transcript word, which the fold reports as batch-local
// region index `local`. For every one of them, the rows and columns must be
// exactly the tile regionToTile gives that region.

// The text of a fold, from its signature to the closing brace.
function foldSource(head) {
  const at = KERNELS.indexOf(head);
  if (at < 0) throw new Error('no ' + head + ' in pearl_sycl_kernels.hpp');
  return stripComments(KERNELS.slice(at, KERNELS.indexOf('\n}\n', at)));
}

// A C index expression from the fold, as a JS function of `vars`.
function cExpr(src, re, vars) {
  const m = re.exec(src);
  if (!m) throw new Error('pattern not found in the fold: ' + re);
  const js = m[1].replace(/\(size_t\)/g, '').replace(/\b(\d+)u\b/g, '$1').replace(/\bA\./g, '');
  return new Function(...vars, 'return ' + js + ';');
}

// A small profile with several row and column windows a batch.
const P = Object.assign({}, PROFILE, { m: 128, n: 512 });
const ROWS_VALID = P.m / 16;
const COL_BATCH = 8;

// Does each region's set of (row, column) cells equal its tile? `cells` maps
// the region's batch-local index to a Set of "r,c" keys.
function expectTiles(cells, nonceBase) {
  expect(cells.size).toBe(COL_BATCH * ROWS_VALID);
  for (const [local, got] of cells) {
    const { rows, cols } = regionToTile(nonceBase + local, P);
    const want = new Set();
    for (const r of rows) for (const c of cols) want.add(r + ',' + c);
    expect([...got].sort()).toEqual([...want].sort());
  }
}

function addCell(cells, local, r, c) {
  if (!cells.has(local)) cells.set(local, new Set());
  cells.get(local).add(r + ',' + c);
}

describe('the folds place every region where regionToTile does', () => {
  const colStarts = [0, 8, 24];

  test('fold_dot: each work-item\'s 4 rows by 2 columns, and its region index', () => {
    const src = foldSource('inline void fold_dot(');
    const threads = Number(/constexpr uint32_t kThreads = (\d+);/.exec(KERNELS)[1]);
    const [, accRows, accCols] = /int32_t acc\[(\d+)\]\[(\d+)\]/.exec(src).map(Number);
    const qOf = cExpr(src, /const uint32_t q = ([^,]+), e = /, ['w']);
    const eOf = cExpr(src, /, e = ([^;]+);/, ['w']);
    const rbase = cExpr(src, /const uint32_t rbase = ([^;]+);/, ['e', 'q']);
    const cbase = cExpr(src, /const uint32_t cbase = ([^;]+);/, ['e', 'q']);
    const rowBase = cExpr(src, /\*Ab = A\.Ap \+ \(size_t\)\(([^)]+)\) \* k;/, ['wr']);
    const colBase = cExpr(src, /\*Bb = A\.Bp \+ \(size_t\)\(([^;]+)\) \* k;/, ['colStart', 'wc']);
    const local = cExpr(src, /const uint32_t local = ([^;]+);/, ['wr', 'wc', 'q', 'rowsValid']);
    expect([threads, accRows, accCols]).toEqual([256, 4, 2]);
    for (const colStart of colStarts) {
      const cells = new Map();
      for (let wr = 0; wr < ROWS_VALID / 2; wr++) {
        for (let wc = 0; wc < COL_BATCH / 4; wc++) {
          for (let w = 0; w < threads; w++) {
            const q = qOf(w), e = eOf(w);
            const at = local(wr, wc, q, ROWS_VALID);
            for (let i = 0; i < accRows; i++) {
              for (let j = 0; j < accCols; j++) {
                // Rows of A' from rowBase, columns (rows of B') from colBase.
                addCell(cells, at, rowBase(wr) + rbase(e, q) + i, colBase(colStart, wc) + cbase(e, q) + j);
              }
            }
          }
        }
      }
      expectTiles(cells, colStart * ROWS_VALID);
    }
  });

  test('fold_xmx at sub-group 16 and 8: each lane\'s accumulators, the shuffles and the region index', () => {
    const src = foldSource('inline void fold_xmx(');
    const rowBase = cExpr(src, /\*Ab = A\.Ap \+ \(size_t\)\(([^)]+)\) \* k;/, ['wr', 'sgi', 'RB']);
    const colBase = cExpr(src, /\*Bb = A\.Bp \+ \(size_t\)\(([^;]+)\) \* k;/, ['colStart', 'wc', 'lane']);
    const local = cExpr(src, /const uint32_t local = ([^;]+);/, ['wr', 'wc', 'qr', 'rowsValid']);
    const masks = [...src.matchAll(/permute_group_by_xor\(sg, x0, (\d+)u\)/g)].map((m) => Number(m[1]));
    expect(masks).toEqual([1, 8]);
    // What the model below takes as given, pinned to the code: the DPAS C
    // layout (lane = column, element r = row r of the 8-row block), rows 0-3
    // into x0 and 4-7 into x1, column block cb at SG columns apart, and lane
    // q < 8 writing region q's word from x1 when q is odd.
    expect(src).toContain('for (int r = 0; r < 4; r++) x0 ^= (uint32_t)c[rb][cb][r];');
    expect(src).toContain('for (int r = 4; r < 8; r++) x1 ^= (uint32_t)c[rb][cb][r];');
    expect(src).toContain('Bb + (size_t)(SG * (BG * g + i)) * k + k0');
    expect(src).toContain('const int8_t *p = Ab + (size_t)(8 * rb + r) * k + k0;');
    expect(src).toContain('if (lane < 8u) part[(sgi * 8u + lane) * 16u + ch] = (lane & 1u) ? x1 : x0;');
    expect(src).toContain('x ^= part[((uint32_t)s * 8u + qr) * 16u + (uint32_t)j];');
    for (const [SG, RB] of [[16, 2], [8, 1]]) {
      const S = 4 / RB, NCB = 64 / SG;
      // The lanes a lane's word gathers: itself and every XOR of the masks that
      // exist at this sub-group size (lane ^ 8 has no partner at 8 lanes).
      const gather = (q) => {
        let set = [q];
        for (const m of masks.filter((x) => x < SG)) set = set.concat(set.map((l) => l ^ m));
        return set;
      };
      for (const colStart of colStarts) {
        const cells = new Map();
        for (let wr = 0; wr < ROWS_VALID / 2; wr++) {
          for (let wc = 0; wc < COL_BATCH / 4; wc++) {
            for (let qr = 0; qr < 8; qr++) {
              const at = local(wr, wc, qr, ROWS_VALID);
              for (let sgi = 0; sgi < S; sgi++) {
                for (const lane of gather(qr)) {
                  for (let rb = 0; rb < RB; rb++) {
                    for (let r = (qr & 1) * 4; r < (qr & 1) * 4 + 4; r++) {
                      for (let cb = 0; cb < NCB; cb++) {
                        addCell(cells, at, rowBase(wr, sgi, RB) + 8 * rb + r,
                          colBase(colStart, wc, lane) + SG * cb);
                      }
                    }
                  }
                }
              }
            }
          }
        }
        expectTiles(cells, colStart * ROWS_VALID);
      }
    }
  });
});

describe('SYCL harness scripts', () => {
  const node = (args) => spawnSync(process.execPath, args, { encoding: 'utf8' });

  test('isa-report summarises IGC assembly by kernel', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'isa-'));
    try {
      const hw = '_ZTSZZN10pearl_sycl8fold_xmxILi16ELi2ELb1EEEvRN4sycl3_V15queueE';
      fs.writeFileSync(path.join(dir, 'OCL_asm1_simd16_entry_0001.asm'), [
        '//.kernel ' + hw, '//.platform XE2', '//.thread_config numGRF=128, numAcc=4',
        '//.instCount 1618', '        dpas.8x8 (16|M0)  r1:d r2:d r3:d r4:b',
        '(W)     dpas.8x8 (16|M0)  r5:d r6:d r7:d r8:b', ''].join('\n'));
      fs.writeFileSync(path.join(dir, 'OCL_asm2_simd8_entry_0002.asm'), [
        '//.kernel _ZTSZN10pearl_sycl8fold_dotERN4sycl', '//.platform DG2',
        '//.thread_config numGRF=128, numAcc=4', '//.spill size 192', '//.instCount 1983', ''].join('\n'));
      fs.writeFileSync(path.join(dir, 'notes.txt'), 'not assembly');
      const r = node([path.join(SYCL, 'isa-report.js'), dir]);
      expect(r.status).toBe(0);
      const rows = r.stdout.trim().split('\n').map((l) => l.trim().split(/\s+/));
      expect(rows[0]).toEqual(['platform', 'kernel', 'simd', 'grf', 'dpas', 'spill', 'insts']);
      expect(rows[1]).toEqual(['XE2', 'fold_xmx<sg16,rb2,hw>', '16', '128', '2', '0', '1618']);
      expect(rows[2]).toEqual(['DG2', 'fold_dot', '8', '128', '0', '192', '1983']);
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });

  test('isa-report, verify-hits and bench refuse to run without their arguments', () => {
    for (const script of ['isa-report.js', 'verify-hits.js', 'bench.js']) {
      const r = node([path.join(SYCL, script)]);
      expect(r.status).not.toBe(0);
      expect(r.stderr).toMatch(/usage/);
    }
  });
});
