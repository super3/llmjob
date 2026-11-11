'use strict';

// The SYCL core for Intel GPUs (native/sycl) cannot be built or run here, so,
// as nativeConfig.test.js does for the CUDA core, these tests read its sources
// and pin what must agree with the rest of the repo: the C API pearl_core.cc
// calls, signature for signature; the structs the two files share; and the
// index arithmetic the folds use to place each region, evaluated against the
// protocol's regionToTile. They also run the harness scripts that need no
// device. Whether the folds compute the right transcripts is checked by
// running them (pearl_sycl_check and verify-hits.js on the SYCL CPU device).
//
// pearl_core.cc's API has grown since the port was written (per-card k, the
// checked create, the fold's name, launch shapes). The port answers each
// plainly, and the second half of this file pins those answers: above all that
// a GPU, which no fold here has run on yet, mines only with every hit checked.

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
const CHECK = read(SYCL, 'pearl_sycl_check.cpp');
const HELPER = path.join(__dirname, 'helpers', 'pearlHit.js');

// The text from `from` to the first `to` after it, both included.
function slice(text, from, to) {
  const i = text.indexOf(from);
  if (i < 0) throw new Error('not found: ' + from);
  const j = text.indexOf(to, i + from.length);
  if (j < 0) throw new Error('not found after it: ' + to);
  return text.slice(i, j + to.length);
}
// A body's statements, trimmed, without comments or blank lines.
const statements = (body) => body.split('\n').map((l) => l.trim()).filter((l) => l && !l.startsWith('//'));

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

// pearl_host.cu chooses a k per card, has launch shapes for RTX 30 that the
// binding races at startup, and names its fold. No Intel GPU has run this port,
// so there is nothing measured to choose from: the host gives the answer that
// changes nothing.
describe('what the host answers where pearl_host.cu has more', () => {
  test('per-card k: the profile\'s, with no note', () => {
    const body = statements(slice(HOST, 'extern "C" uint32_t pearl_host_card_k(', '\n}\n'));
    expect(body.slice(1)).toEqual([
      "if (note && note_len) note[0] = '\\0';", 'return profile ? profile->k : 0;', '}',
    ]);
    // Create still refuses any other k, so a profile cannot ask for one either.
    expect(HOST).toContain('if (k != PEARL_FOLD_K || rank != PEARL_FOLD_RANK) {');
  });

  test('launch shapes: none, so pearl_core.cc runs no trial and names no shape', () => {
    expect(HOST).toContain('extern "C" int pearl_host_shape_count(void *) { return 0; }');
    expect(HOST).toContain('extern "C" bool pearl_host_set_shape(void *, int) { return false; }');
    expect(statements(slice(HOST, 'extern "C" void pearl_host_shape_name(', '\n}\n')).slice(1))
      .toEqual(['if (buf && len) snprintf(buf, len, "none");', '}']);
    // What makes "0 shapes" mean "no trial" in the binding.
    expect(CORE).toContain('const int nShapes = shape_trial_secs_ > 0 ? pearl_host_shape_count(ctx_) : 0;');
    expect(CORE).toContain('bool trial = nShapes > 1;');
  });

  test('the fold name: written once at create, and "unresolved" without a context', () => {
    const body = statements(slice(HOST, 'extern "C" const char *pearl_host_fold_name(', '\n}\n'));
    expect(body.slice(1)).toEqual([
      'const Ctx *ctx = static_cast<const Ctx *>(handle);',
      'return ctx && !ctx->foldName.empty() ? ctx->foldName.c_str() : "unresolved";', '}',
    ]);
    expect(HOST.match(/ctx->foldName = /g)).toHaveLength(1);
    // pearl_core.cc asks on every read of core.device, after stop() too.
    expect(CORE).toContain('out.Set("fold", Napi::String::New(env, pearl_host_fold_name(ctx_)));');
  });
});

// One setting comes from the first run on hardware (an Arc A750 rig, 10
// November 2026) and not from a guess. It is pinned so that it is changed only
// against a new measurement (native/sycl/README.md, "Results on an Arc A750 rig").
describe('what the first run on hardware set', () => {
  test('a band is the device cache\'s size for the XMX folds and a quarter of it for the plain fold', () => {
    // On an A750 (16 MiB): xmx8 16.6 to 16.9 TH/s at 256 row windows against
    // 15.2 to 15.5 at 64, the quarter that was the rule for both; dot 3.74 at
    // 256 against 3.91 at 64.
    const rule = slice(HOST, 'const uint64_t share =', 'ctx->band = floor_pow2(band);');
    expect(statements(rule).slice(0, 2)).toEqual([
      'const uint64_t share = ctx->fold == kDot ? 4 : 1;',
      'uint64_t band = cache ? cache / share / (32ull * k) : 16;',
    ]);
    // The fold is settled before the band is: pick_fold comes first in create.
    const create = slice(HOST, 'extern "C" void *pearl_host_create_checked(', '\n}\n');
    const picked = create.indexOf('if (!pick_fold(ctx, err, err_len)) {');
    expect(picked).toBeGreaterThan(0);
    expect(picked).toBeLessThan(create.indexOf('const uint64_t share ='));
    // 16 MiB at the mainnet k: 32 rows of 2048 bytes a row window.
    const windows = (share) => Math.floor((16 * 1024 * 1024) / share / (32 * PROFILE.k));
    expect([windows(1), windows(4)]).toEqual([256, 64]);
    // PEARL_SYCL_BAND still wins, so a rig can time another.
    expect(rule).toContain('if (!bEnv.empty()) band = strtoull(bEnv.c_str(), nullptr, 10);');
    // bench.sh times each fold in a core that would mine with it, so each
    // fold is timed at its own band.
    const bench = read(SYCL, 'bench.sh');
    expect(bench).toContain('PEARL_SYCL_FOLD=dot "$B/pearl_sycl_check" --bench "$SECS" --folds dot');
    expect(bench).toContain('"$B/pearl_sycl_check" --bench "$SECS" --folds auto\n');
  });
});

// The folds were checked on the SYCL CPU device, where DPAS is reference code.
// A fold that is wrong on a real card makes every hash wrong, and every one
// still meets the target, so every share would be sent and rejected. So a GPU
// opens only for an opener that checks each hit first (createCore's checkHits,
// which the app passes), and the core says so in core.untested. The same rule
// as the AMD host's for its untested cards (amdCheckHits.test.js).
describe('every Intel GPU mines with every hit checked', () => {
  const create = slice(HOST, 'extern "C" void *pearl_host_create_checked(', '\n}\n');
  const lines = statements(create);

  test('the checked create is the one pearl_core.cc calls, and it tells the opener what it got', () => {
    expect(CORE).toContain('ctx_ = pearl_host_create_checked(&profile, check_hits, &untested_, err, sizeof(err));');
    const at = (s) => {
      const i = lines.indexOf(s);
      expect([s, i >= 0]).toEqual([s, true]);
      return i;
    };
    expect(at('if (untested) *untested = false;')).toBe(2);
    const made = at('Ctx *ctx = new Ctx();');
    const kind = at('ctx->isGpu = ctx->dev.is_gpu();');
    const gate = at('const char *unchecked = getenv("PEARL_SYCL_UNTESTED");');
    const told = at('if (untested) *untested = ctx->untested;');
    expect([made < kind, kind < gate, gate < told]).toEqual([true, true, true]);
    expect(lines.slice(-2)).toEqual(['return ctx;', '}']);
    expect(HOST.match(/ctx->untested = /g)).toHaveLength(1);
  });

  test('pearl_host_create is the checked create for an opener that checks nothing', () => {
    const body = statements(slice(HOST, 'extern "C" void *pearl_host_create(', '\n}\n'));
    expect(body.slice(1)).toEqual(['return pearl_host_create_checked(profile, false, nullptr, err, err_len);', '}']);
  });

  // The gate, from the read of PEARL_SYCL_UNTESTED to the first line after it,
  // run in JS: what an opener gets.
  const gate = slice(create, '  const char *unchecked = getenv("PEARL_SYCL_UNTESTED");', '  sycl::queue &q = *ctx->q;');
  const js = gate
    .replace('const char *unchecked = getenv("PEARL_SYCL_UNTESTED");', 'const unchecked = env;')
    .replace(/ctx->/g, 'ctx.')
    .replace(/fail\(err, err_len,[\s\S]*?\);\n\s*pearl_host_destroy\(ctx\);\n\s*return nullptr;/, "return 'refused';")
    .replace('sycl::queue &q = *ctx.q;', "return ctx.untested ? 'checked' : 'opens';");
  const strcmp = (a, b) => (a === b ? 0 : 1);
  const run = new Function('ctx', 'check_hits', 'env', 'strcmp', js);
  const open = ({ gpu = true, checkHits = false, env } = {}) => run({ isGpu: gpu, untested: false }, checkHits, env, strcmp);

  test('the gate reads as JS', () => {
    expect(js).not.toMatch(/->|fail\(|const char|nullptr|sycl::/);
    expect(js).toContain("return 'refused';");
  });

  test('a GPU is refused unless the opener checks hits, and checked when it does', () => {
    expect([open(), open({ checkHits: true })]).toEqual(['refused', 'checked']);
    for (const env of ['0', 'yes', '', undefined]) {
      expect([env, open({ env }), open({ env, checkHits: true })]).toEqual([env, 'refused', 'checked']);
    }
    // PEARL_SYCL_UNTESTED=1, and only that, opens a GPU for an opener that
    // checks nothing (bench.js). It never turns the check off.
    expect([open({ env: '1' }), open({ env: '1', checkHits: true })]).toEqual(['opens', 'checked']);
  });

  test('the SYCL CPU device, where the folds are checked, is never gated', () => {
    for (const checkHits of [false, true]) {
      for (const env of [undefined, '0', '1']) {
        expect([checkHits, env, open({ gpu: false, checkHits, env })]).toEqual([checkHits, env, 'opens']);
      }
    }
  });

  test('the refusal fits the buffer pearl_core.cc gives it, and says what to do', () => {
    const literals = /fail\(err, err_len,\s*((?:"[^"]*"\s*)+)\);/.exec(gate)[1];
    const msg = literals.split('\n').map((l) => l.trim().replace(/^"|"$/g, '')).join('');
    expect(CORE).toContain('char err[256] = {0};');
    expect(msg.length).toBeLessThan(256);
    expect(msg).toMatch(/^no Intel GPU has mined with this core in a release yet/);
    expect(msg).toContain('PEARL_SYCL_UNTESTED=1');
    expect(msg).toContain('earn/native/sycl/README.md');
  });

  test('the tools that check hits open the core that way, and the one that checks none is told to', () => {
    expect(CHECK).toContain('void *ctx = pearl_host_create_checked(&p, true, &untested, err, sizeof err);');
    expect(CHECK).not.toMatch(/pearl_host_create\(/);
    const verify = read(SYCL, 'verify-hits.js');
    expect(verify).toContain('const coreOpts = { checkHits: true };');
    expect(verify).toContain("const { checkHit } = require(R + 'checkHit');");
    // bench.js sends no shares. Only the line that runs it gets the override.
    const uses = (f) => read(SYCL, f).split('\n').filter((l) => !l.startsWith('#') && l.includes('PEARL_SYCL_UNTESTED'));
    expect(uses('bench.sh')).toEqual([
      'PEARL_SYCL_UNTESTED=1 node "$HERE/bench.js" "$B/pearl_core_sycl.node" --seconds 60 --warmup 10',
    ]);
    for (const f of ['verify.sh', 'pool-run.sh', 'build.sh', 'compile-gate.sh']) expect([f, uses(f)]).toEqual([f, []]);
  });

  // A rig has several cards, and PEARL_GPU_INDEX names one by its place in the
  // core's own list. The list the check tool prints must be that same list.
  test('pearl_sycl_check --list numbers the cards as the core does', () => {
    const own = 'const std::vector<sycl::device> devs = list_devices();';
    for (const fn of ['extern "C" int pearl_host_select_device(', 'extern "C" void *pearl_host_create_checked(',
      'extern "C" int pearl_sycl_list_devices(']) {
      expect([fn, slice(HOST, fn, '\n}\n').includes(own)]).toEqual([fn, true]);
    }
    expect(CHECK).toContain('const int n = pearl_sycl_list_devices(list, sizeof list);');
    expect(CHECK).toContain('if (pearl_host_select_device(&p, requested, name, sizeof name, err, sizeof err) < 0) {');
  });
});

// A stand-in addon: it prints the options it was opened with, and on setJob
// reports 20 genuine hits built in JS, each with one transcript word changed
// when STUB_WRONG is set, as a card whose fold is wrong would report them.
const STUB = `
const { EventEmitter } = require('events');
const { hitFor } = require(${JSON.stringify(HELPER)});
module.exports = { createCore(profile, opts = {}) {
  console.log('OPTS ' + JSON.stringify(opts));
  if (process.env.STUB_REFUSE) throw new Error(process.env.STUB_REFUSE);
  const c = new EventEmitter();
  c.k = profile.k;
  c.untested = true;
  c.device = { index: opts.deviceIndex || 0, name: 'stand-in', fold: 'js' };
  const tamper = process.env.STUB_WRONG ? (jp) => { jp[9] ^= 1; } : undefined;
  c.setJob = (job) => setImmediate(() => {
    const h = hitFor(profile, job.header, { bound: job.target, jobId: job.jobId, tamper });
    for (let s = 0; s < 20; s++) c.emit('hit', { ...h, salt: s });
    c.emit('hashrate', 0.25);
  });
  c.stop = () => {};
  return c;
} };
`;

describe('verify-hits.js, with the check the app runs', () => {
  let tmp;
  beforeAll(() => {
    tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'sycl-verify-'));
    fs.writeFileSync(path.join(tmp, 'stub.js'), STUB);
  });
  afterAll(() => fs.rmSync(tmp, { recursive: true, force: true }));
  const run = (env) => {
    const e = { ...process.env, ...env };
    for (const k of ['PEARL_GPU_INDEX', 'PEARL_SYCL_FOLD', 'STUB_WRONG', 'STUB_REFUSE']) if (!(k in env)) delete e[k];
    const r = spawnSync(process.execPath, [path.join(SYCL, 'verify-hits.js'), path.join(tmp, 'stub.js'),
      '--hits', '20', '--seconds', '30', '--bits', '256', '--m', '64', '--n', '64'], { env: e, encoding: 'utf8' });
    const out = r.stdout.trim().split('\n');
    return { status: r.status, opts: JSON.parse(out.find((l) => l.startsWith('OPTS ')).slice(5)), json: JSON.parse(out.pop()) };
  };

  test('opens the core with checkHits, passes right hits and says what the core answered', () => {
    const ok = run({});
    expect(ok.opts).toEqual({ checkHits: true });
    expect(ok.json).toMatchObject({
      hits: 20, verified: 20, failed: 0, salts: 20, PASS: true, untested: true, m: 64, n: 64,
      device: { index: 0, name: 'stand-in', fold: 'js' },
      // The rate event that came behind the 20 hits was heard: a hit is checked
      // in a turn of its own, so the events behind it do not wait for the checks.
      meanThs: 0.25,
    });
    expect(ok.status).toBe(0);
  }, 60000);

  test('says why in one JSON line when the core will not open', () => {
    const why = 'GPU 5 was asked for but SYCL lists 1: valid indices are 0..0';
    const r = run({ PEARL_GPU_INDEX: '5', STUB_REFUSE: why });
    expect(r.opts).toEqual({ checkHits: true, deviceIndex: 5 });
    expect(r.json).toEqual({ error: why });
    expect(r.status).toBe(2);
  }, 60000);

  test('fails wrong hits with the check\'s reason', () => {
    const bad = run({ STUB_WRONG: '1' });
    expect(bad.json).toMatchObject({ verified: 0, failed: 20, PASS: false });
    expect(bad.json.fails[0].why).toBe('jackpot hash mismatch (fold/transcript)');
    expect(bad.status).toBe(1);
  }, 60000);

  test('PEARL_GPU_INDEX pins the card when it is one number, and not when it is the CLI\'s list', () => {
    expect(run({ PEARL_GPU_INDEX: '3' }).opts).toEqual({ checkHits: true, deviceIndex: 3 });
    expect(run({ PEARL_GPU_INDEX: '0,1' }).opts).toEqual({ checkHits: true });
  }, 60000);
});

// The port is not shipped: the app loads it only when PEARL_CORE_PATH names a
// build of it. These pin that, so adding the folder cannot change an installer.
describe('the installer leaves the port out', () => {
  const EARN = path.join(__dirname, '..');
  const pkg = JSON.parse(read(EARN, 'package.json'));

  test('electron-builder packs only src, package.json and what CI stages in vendor', () => {
    expect(pkg.build.files.length).toBeGreaterThan(0);
    for (const pattern of pkg.build.files) {
      expect([pattern, pattern === 'package.json' || pattern.startsWith('src/')]).toEqual([pattern, true]);
    }
    const extra = Object.values(pkg.build).flatMap((v) => (v && Array.isArray(v.extraResources) ? v.extraResources : []));
    expect(extra.length).toBeGreaterThan(0);
    for (const r of extra) expect([r.from, r.from.startsWith('vendor/')]).toEqual([r.from, true]);
    // From vendor/native, only addons and the Monero worker: no script, source or library.
    for (const r of extra.filter((x) => x.to === 'native')) {
      expect(r.filter).toEqual(['**/*.node', 'llmjob-xmr-worker', 'llmjob-xmr-worker.exe']);
    }
  });

  test('nothing that builds or loads a release names the SYCL core', () => {
    for (const f of [['scripts', 'build-cli.mjs'], ['scripts', 'build-hiveos.mjs'],
      ['src', 'shared', 'coreVariant.js'], ['src', 'main', 'pearlCore.js']]) {
      expect([f.join('/'), /sycl/i.test(read(EARN, ...f))]).toEqual([f.join('/'), false]);
    }
    const workflows = path.join(EARN, '..', '.github', 'workflows');
    for (const f of fs.readdirSync(workflows)) expect([f, /sycl/i.test(read(workflows, f))]).toEqual([f, false]);
  });
});

// verify.sh and pool-run.sh, run for real under bash against stand-ins for the
// two programs they call: a pearl_sycl_check that prints its arguments and
// fails for the folds named in FAIL_FOLDS, and a `node` that prints its
// arguments. Skipped where there is no bash.
const hasBash = spawnSync('bash', ['--version'], { stdio: 'ignore' }).status === 0;
(hasBash ? describe : describe.skip)('verify.sh and pool-run.sh under bash', () => {
  const ADDR = 'prl1pql8r6m4z9x7v2k0t3whu8e2snd4p6c';
  let tmp;
  beforeAll(() => {
    tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'sycl-sh-'));
    fs.mkdirSync(path.join(tmp, 'bin'));
    fs.writeFileSync(path.join(tmp, 'pearl_sycl_check'), [
      '#!/usr/bin/env bash',
      'echo "CHECK $*"',
      'folds=""; while [ $# -gt 0 ]; do if [ "$1" = "--folds" ]; then folds=$2; fi; shift; done',
      'for f in ${FAIL_FOLDS//,/ }; do case ",$folds," in *",$f,"*) exit 1 ;; esac; done',
      'exit 0', ''].join('\n'), { mode: 0o755 });
    fs.writeFileSync(path.join(tmp, 'bin', 'node'), ['#!/usr/bin/env bash', 'echo "NODE $*"', 'exit 0', ''].join('\n'),
      { mode: 0o755 });
  });
  afterAll(() => {
    fs.rmSync(tmp, { recursive: true, force: true });
    // pool-run.sh writes its log beside itself.
    for (const f of ['pool-run.log', 'pool-run-stats.json']) fs.rmSync(path.join(SYCL, f), { force: true });
  });
  const sh = (script, args, env = {}) => {
    const e = { ...process.env, PATH: path.join(tmp, 'bin') + path.delimiter + process.env.PATH, PEARL_SYCL_BUILD: tmp, ...env };
    for (const k of ['PEARL_SYCL_DEVICE', 'FAIL_FOLDS', 'PEARL_ADDRESS', 'PEARL_POOL', 'PEARL_WORKER', 'PEARL_GPU_INDEX']) {
      if (!(k in env)) delete e[k];
    }
    const r = spawnSync('bash', [path.join(SYCL, script), ...args], { encoding: 'utf8', env: e });
    const lines = r.stdout.split('\n');
    return {
      status: r.status, out: r.stdout, err: r.stderr,
      checks: lines.filter((l) => l.startsWith('CHECK ')), nodes: lines.filter((l) => l.startsWith('NODE ')),
    };
  };
  const folds = (r) => r.checks.map((l) => /--folds (\S+)/.exec(l)[1]);

  // On an Arc A750 the hardware DPAS fold matches the host and the sub-group 8
  // reference code does not. The reference code mines nothing, so it must not
  // fail a card whose own folds are right.
  test('on a GPU, verify.sh passes when only the reference-code fold is wrong, and says so', () => {
    const r = sh('verify.sh', [], { FAIL_FOLDS: 'auto-emu' });
    expect(r.status).toBe(0);
    expect(folds(r)).toEqual(['dot,auto', 'auto-emu', 'dot,auto']);
    expect(r.out).toContain('== for information, not part of PASS: the reference code of the core\'s fold');
    expect(r.out).toContain('note: the reference code differs from the host on this device');
    expect(r.out.trim().split('\n').pop()).toBe('== PASS');
    // The hits: the core's own fold, then the plain one, 400 each at 2^235.
    expect(r.nodes).toHaveLength(2);
    for (const l of r.nodes) expect(l).toMatch(/verify-hits\.js \S*pearl_core_sycl\.node --hits 400 --seconds 1800 --bits 235$/);
  });

  test('on a GPU, verify.sh fails when a fold that mines is wrong', () => {
    for (const bad of ['auto', 'dot']) {
      const r = sh('verify.sh', [], { FAIL_FOLDS: bad });
      expect([bad, r.status === 0, r.out.includes('== PASS')]).toEqual([bad, false, false]);
      expect(r.nodes).toHaveLength(0);
    }
  });

  test('--quick asks for 100 hits at half the bound, so they come from 3 salts or more', () => {
    const gpu = sh('verify.sh', ['--quick']);
    expect(gpu.status).toBe(0);
    for (const l of gpu.nodes) expect(l).toMatch(/--hits 100 --seconds 1800 --bits 234$/);
    const cpu = sh('verify.sh', ['--quick'], { PEARL_SYCL_DEVICE: 'cpu' });
    expect(cpu.status).toBe(0);
    expect(folds(cpu)).toEqual(['dot,xmx16,xmx8', 'dot,xmx16']);
    expect(cpu.nodes).toHaveLength(3);
    for (const l of cpu.nodes) expect(l).toMatch(/--hits 100 --seconds 1800 --m 4096 --n 4096 --col-batch 64 --bits 245$/);
    expect(cpu.out).not.toContain('not part of PASS');
  });

  test('pool-run.sh passes the pool and worker on, and takes the address from the environment for "-"', () => {
    const r = sh('pool-run.sh', ['-', '1'], { PEARL_ADDRESS: ADDR, PEARL_POOL: 'prl-us.kryptex.network:7048', PEARL_WORKER: 'arc750' });
    expect(r.status).toBe(0);
    const cli = r.nodes.filter((l) => l.includes('earn-cli.js'));
    expect(cli).toHaveLength(1);
    expect(cli[0]).toMatch(new RegExp('earn-cli\\.js --address ' + ADDR + ' --mode mining --no-report --no-serve --no-update'
      + ' --stats-file \\S+pool-run-stats\\.json --pool prl-us\\.kryptex\\.network:7048 --worker arc750$'));
    const plain = sh('pool-run.sh', [ADDR, '1']);
    expect(plain.nodes.filter((l) => l.includes('earn-cli.js'))[0]).toMatch(/--stats-file \S+pool-run-stats\.json$/);
    const none = sh('pool-run.sh', ['-', '1']);
    expect(none.status).not.toBe(0);
    expect(none.err).toContain('PEARL_ADDRESS is not set');
    expect(none.nodes.filter((l) => l.includes('earn-cli.js'))).toHaveLength(0);
  });
});
