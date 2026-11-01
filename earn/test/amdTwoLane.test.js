'use strict';

// Two-lane 32x8 regions in the RDNA 4 fold (PEARL_AMD_TWOLANE; earn/native/amd/src/
// pearl_amd_pattern.h), Stage A: the fold and its fold test only. No HIP compiler runs here, so
// these check the pattern against the JS protocol code, run the fold's own region arithmetic
// (read out of the source) over a whole batch, pin the guards that keep the build from mining,
// and run sweep.sh's fold rounds and the instruction census on stand-ins.

const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');
const {
  PROFILE, patternFromList, patternToBytes, expandOffset, buildConfig52,
} = require('../src/shared/miner/pearlhash');
const census = require('../native/amd/test/readout-census');

const ROOT = path.join(__dirname, '..', '..');
const AMD = path.join(__dirname, '..', 'native', 'amd');
const read = (f) => fs.readFileSync(path.join(AMD, f), 'utf8').replace(/\r\n/g, '\n');
const PATTERN = read('src/pearl_amd_pattern.h');
const FOLD = read('src/pearl_amd_fold.h');
const HOST = read('src/pearl_host_hip.cpp');
const FOLDTEST = read('test/pearl_amd_foldtest.cpp');
const BUILD = read('build-amd.sh');
const CPUCHECKS = read('test/run-cpu-checks.sh');
const SWEEP = read('test/sweep.sh');
const CENSUS_SH = read('test/readout-census.sh');
const CI = fs.readFileSync(path.join(ROOT, '.github', 'workflows', 'native-core.yml'), 'utf8')
  .replace(/\r\n/g, '\n');

const hasBash = spawnSync('bash', ['--version'], { stdio: 'ignore' }).status === 0;
const bashIt = hasBash ? it : it.skip;

// The numbers in a C initialiser `NAME[...] = { ... };`.
function initialiser(text, name) {
  const at = text.indexOf(name + '[');
  if (at < 0) throw new Error('no ' + name);
  const open = text.indexOf('{', at), close = text.indexOf('}', open);
  return text.slice(open + 1, close).split(',').map((s) => Number(s.trim()));
}

// The text from the line starting with `first` to the next line that is exactly `last`.
function block(text, first, last) {
  const lines = text.split('\n');
  const a = lines.findIndex((l) => l.startsWith(first));
  const b = lines.findIndex((l, i) => i > a && l === last);
  if (a < 0 || b < 0) throw new Error('block not found: ' + first);
  return lines.slice(a, b + 1).join('\n') + '\n';
}

const ROWS = initialiser(PATTERN, 'PEARL_AMD_32X8_ROWS');
const COLS = initialiser(PATTERN, 'PEARL_AMD_32X8_COLS');
const orAll = (l) => l.reduce((a, b) => a | b, 0);

describe('the 32x8 pattern', () => {
  it('is the decision\'s rows and columns', () => {
    expect(ROWS).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23,
      32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55]);
    expect(COLS).toEqual([0, 1, 16, 17, 32, 33, 48, 49]);
    expect([orAll(ROWS), orAll(COLS)]).toEqual([0x37, 0x31]);
    expect(ROWS.length * COLS.length).toBe(256);
  });

  it('has the bytes the JS derives, and so the config52 the decision pins', () => {
    expect(Buffer.from(initialiser(PATTERN, 'PEARL_AMD_32X8_ROWS_BYTES')))
      .toEqual(patternToBytes(patternFromList(ROWS)));
    expect(Buffer.from(initialiser(PATTERN, 'PEARL_AMD_32X8_COLS_BYTES')))
      .toEqual(patternToBytes(patternFromList(COLS)));
    const c52 = buildConfig52({ ...PROFILE, rows: ROWS, cols: COLS });
    expect(c52.subarray(0, 20).toString('hex')).toBe('0008000080000000000701030000000107030000');
    expect(buildConfig52().subarray(8, 20).toString('hex')).toBe('000301030000000103070000');
  });

  it('is every subset of its mask, so valid offsets tile the operands with no gap or overlap', () => {
    for (const list of [ROWS, COLS]) {
      const mask = orAll(list);
      const subsets = [];
      for (let s = 0; s <= mask; s++) if ((s & mask) === s) subsets.push(s);
      expect(list).toEqual(subsets);
    }
  });

  it('is off by default, and the store form needs the switch', () => {
    expect(PATTERN).toContain('#ifndef PEARL_AMD_TWOLANE\n#define PEARL_AMD_TWOLANE 0\n#endif');
    expect(PATTERN).toContain('#ifndef PEARL_AMD_TWOLANE_STORE\n#define PEARL_AMD_TWOLANE_STORE 0\n#endif');
    expect(PATTERN).toContain('#if PEARL_AMD_TWOLANE_STORE && !PEARL_AMD_TWOLANE\n#error');
    expect(FOLD).toContain('#if PEARL_AMD_READOUT_DROP && PEARL_AMD_TWOLANE\n#error');
  });
});

describe('the two-lane fold', () => {
  it('is wmma12 only, and its code object says so', () => {
    const policy = (name) => FOLD.slice(FOLD.indexOf('struct ' + name + ' {'),
      FOLD.indexOf('\n};', FOLD.indexOf('struct ' + name + ' {')));
    expect(policy('PearlAmdWmma12')).toContain('static constexpr bool kTwoLane = PEARL_AMD_TWOLANE != 0;');
    for (const p of ['PearlAmdMfma16', 'PearlAmdMfma8', 'PearlAmdWmma11'])
      expect(policy(p)).toContain('static constexpr bool kTwoLane = false;');
    expect(FOLD).toContain('#define PEARL_AMD_CAP_TWOLANE 0x400u');
    expect(FOLD).toContain('if (PEARL_AMD_TWOLANE && (PEARL_AMD_EMU || PEARL_AMD_WMMA12)) out[0] |= PEARL_AMD_CAP_TWOLANE;');
    // The pair step is one quad permute on the card and a shuffle on the CPU.
    expect(FOLD).toContain('__builtin_amdgcn_update_dpp(0, (int)x, 0xB1, 0xF, 0xF, false)');
    expect(FOLD).toContain('return x ^ (uint32_t)__shfl_xor((int)x, 1);');
  });

  // The fold's own arithmetic, read out of the source and run here over a whole batch with
  // gfx12's WMMA layout: every cell of C must land in exactly one region, and each region must
  // hold exactly the cells the fold test's reference (and a pool) gives that region number.
  it('puts every cell of a batch in exactly the region the reference names', () => {
    const wmma12 = FOLD.slice(FOLD.indexOf('struct PearlAmdWmma12 {'));
    expect(wmma12).toContain('int c_row(int lane, int g) { return 8 * (lane / 16) + g; }');
    expect(wmma12).toContain('int c_col(int lane, int) { return lane % 16; }');
    const src = block(FOLD, '  if constexpr (kTwo) {\n'.trimEnd(), '  }')
      .split('\n').slice(1, -2).join('\n');
    expect(src).toContain('region2 = colIdx * rows_valid + rowIdx;');
    // C to JS: drop the types, the casts and the u suffixes. Every division here is exact.
    const js = src.replace(/const uint32_t /g, 'const ').replace(/\(uint32_t\)/g, '')
      .replace(/(\d+)u\b/g, '$1');
    const regionOf = new Function('lane', 'tileRow', 'tileCol', 'wr', 'wc', 'WM', 'WN', 'rows_valid',
      'let region2;\n' + js + '\nreturn region2;');

    const m = 256, cols = 512, rowsValid = m / 32, colsValid = cols / 8;
    const cells = new Map();   // region -> Set of "row,col"
    let total = 0;
    for (let by = 0; by < m / 128; by++)
      for (let bx = 0; bx < cols / 256; bx++)
        for (let wave = 0; wave < 8; wave++)
          for (let lane = 0; lane < 32; lane++) {
            const wr = Math.floor(wave / 4), wc = wave % 4;
            const region = regionOf(lane, by * 128, bx * 256, wr, wc, 64, 64, rowsValid);
            expect(Number.isInteger(region)).toBe(true);
            // Lanes l and l ^ 1 share it; the even lane's words go first.
            expect(regionOf(lane ^ 1, by * 128, bx * 256, wr, wc, 64, 64, rowsValid)).toBe(region);
            if (!cells.has(region)) cells.set(region, new Set());
            for (let mt = 0; mt < 4; mt++)
              for (let nt = 0; nt < 4; nt++)
                for (let g = 0; g < 8; g++) {
                  const row = by * 128 + wr * 64 + 16 * mt + 8 * Math.floor(lane / 16) + g;
                  const col = bx * 256 + wc * 64 + 16 * nt + (lane % 16);
                  cells.get(region).add(row + ',' + col);
                  total++;
                }
          }
    expect(total).toBe(m * cols);
    expect(cells.size).toBe(rowsValid * colsValid);
    for (let region = 0; region < rowsValid * colsValid; region++) {
      const ro = expandOffset(region % rowsValid, 0x37), co = expandOffset(Math.floor(region / rowsValid), 0x31);
      const want = new Set();
      for (const r of ROWS) for (const c of COLS) want.add((ro | r) + ',' + (co | c));
      expect(cells.get(region)).toEqual(want);
    }
  });

  it('keeps words 0-7 on the even lane and 8-15 on the odd one, and stores them there', () => {
    expect(FOLD).toContain('if (odd == (c >= 8u)) keep[c & 7u] = (int32_t)x;');
    expect(FOLD).toContain('tr + (size_t)region2 * PEARL_JACKPOT_BUCKETS + 8u * (uint32_t)(lane & 1));');
    expect(FOLD).toContain('w[0] = keep.lo;\n    w[1] = keep.hi;');
    // The store form: the even lane, a scalar base that moves a word a chunk, a per-lane offset.
    expect(FOLD).toContain('if (!odd) *reinterpret_cast<uint32_t *>(reinterpret_cast<char *>(tr + c) + trByte) = x;');
    // The offset goes through an empty asm each chunk, before the store, so the compiler can't
    // move tr + trByte out of the slab loop as a per-lane pointer and lose the scalar base.
    expect(FOLD).toContain('[[maybe_unused]] uint32_t trByte = region2 *');
    const storeForm = FOLD.slice(FOLD.indexOf('      if constexpr (kTwoStore) {'));
    const hide = storeForm.indexOf('        asm("" : "+v"(trByte));');
    expect(hide).toBeGreaterThan(0);
    expect(hide).toBeLessThan(storeForm.indexOf('if (!odd) *reinterpret_cast'));
    expect(storeForm.slice(0, hide)).toMatch(/#if !PEARL_AMD_EMU && defined\(__HIP_DEVICE_COMPILE__\)\n$/);
  });

  it('keeps today\'s readout and end of tile, unchanged, for every other build', () => {
    // The 16x16 path sits in the else of each two-lane branch, word for word as before.
    expect(FOLD).toContain('    } else {\n      const bool mine = (c & 7u) == r, hi = c >= 8u;');
    expect(FOLD).toContain('              keepLo[rb][cb][q] = (mine && !hi) ? v : keepLo[rb][cb][q];\n'
      + '              keepHi[rb][cb][q] = (mine && hi) ? v : keepHi[rb][cb][q];');
    expect(FOLD).toContain('          w[r] = keepLo[rb][cb][q];\n          w[r + 8] = keepHi[rb][cb][q];');
    expect(FOLD).toContain('constexpr bool kTwo = P::kTwoLane, kTwoStore = kTwo && PEARL_AMD_TWOLANE_STORE;');
  });
});

describe('Stage A can\'t mine', () => {
  it('has the host refuse the two-lane wmma12 fold before it picks it', () => {
    const resolve = HOST.slice(HOST.indexOf('void resolve_fold(Ctx *ctx) {'));
    const refuse = resolve.indexOf('two-lane folds are fold-test only in this build');
    expect(refuse).toBeGreaterThan(0);
    expect(refuse).toBeLessThan(resolve.indexOf('ctx->fold = want;'));
    expect(resolve.slice(0, refuse)).toContain(
      'if (kFolds[want].cap == PEARL_AMD_CAP_WMMA12 && (PEARL_AMD_TWOLANE || (caps[0] & PEARL_AMD_CAP_TWOLANE))) {');
  });

  it('checks each fold in the fold test against its own pattern', () => {
    expect(FOLDTEST).toContain('PEARL_AMD_TWOLANE ? &kAmdPattern32x8 : &kAmdPattern16x16');
    expect(FOLDTEST).toContain('if (got[i] != want->words[i] && bad++ == 0) first = i;');
    expect(FOLDTEST).toContain('x ^= (uint32_t)run[(size_t)(ro | p.rows[a]) * cols + (co | p.cols[b])];');
  });
});

describe('the CPU checks', () => {
  it('pass PEARL_AMD_CFLAGS to the cpu build, and run every two-lane fold test they build', () => {
    expect(BUILD).toContain('flags="-std=c++17 -O2 -fPIC -DNDEBUG ${PEARL_AMD_CFLAGS:-}"');
    const built = [];
    for (const form of ['twolane', 'twolane_store'])
      for (const loop of ['', '_pf2', '_pfx']) built.push('pearl_amd_foldtest_cpu_' + form + loop);
    expect(BUILD).toContain('for form in twolane twolane_store; do');
    for (const b of built) expect(CPUCHECKS).toContain('"$B/' + b + '"');
  });
});

describe('sweep.sh SWEEP_FOLD_ROUNDS', () => {
  // A stand-in build-amd.sh whose fold test answers by its flags: base 50.000 TMAC/s, the
  // register form 50.750, the store form 50.500.
  const STUB = [
    '#!/bin/bash',
    'mkdir -p build',
    'printf \'#!/bin/bash\\nf="%s"\\n\' "${PEARL_AMD_CFLAGS:-}" > build/pearl_amd_foldtest',
    'cat >> build/pearl_amd_foldtest <<\'FT\'',
    'case "$f" in *TWOLANE_STORE=1*) p=32x8; t=50.500 ;; *TWOLANE=1*) p=32x8; t=50.750 ;; *) p=16x16; t=50.000 ;; esac',
    'if [ "$5" = 0 ]; then echo "$6  timed only: 2097152 regions of $p, $4 launches, 20.0 s, $t TMAC/s"; exit 0; fi',
    'echo "$6  ok: 4096 regions of $p, all 16 words match; 1 launches, 0.0100 s, 1.000 TMAC/s"; echo PASS',
    'FT',
    'chmod +x build/pearl_amd_foldtest',
    'echo "${PEARL_AMD_CFLAGS:-}" > build/pearl_core_hip.node',
    'echo "build ${PEARL_AMD_CFLAGS:-}" >> builds.log',
    '',
  ].join('\n');
  const SWEEP_LINE = 'base=;reg=-DPEARL_AMD_TWOLANE=1;store=-DPEARL_AMD_TWOLANE=1 -DPEARL_AMD_TWOLANE_STORE=1';

  let tmp;
  beforeEach(() => { tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'amd-sweep-')); });
  afterEach(() => { fs.rmSync(tmp, { recursive: true, force: true }); });

  // Runs a copy of sweep.sh with the stand-in, writing nothing outside tmp: the container-log
  // copy of each line goes to /dev/null.
  function sweep(env) {
    const amd = path.join(tmp, 'amd');
    fs.mkdirSync(path.join(amd, 'test'), { recursive: true });
    fs.mkdirSync(path.join(tmp, 'root'));
    fs.writeFileSync(path.join(amd, 'build-amd.sh'), STUB, { mode: 0o755 });
    fs.writeFileSync(path.join(amd, 'test', 'sweep.sh'), SWEEP.replace('/proc/1/fd/1', '/dev/null'));
    const r = spawnSync('bash', ['test/sweep.sh'], {
      cwd: amd, encoding: 'utf8',
      env: Object.assign({}, process.env, { ARCH: 'gfx1201', SWEEP_ROOT: path.join(tmp, 'root').replace(/\\/g, '/') }, env),
    });
    return {
      status: r.status,
      lines: r.stdout.split('\n').filter(Boolean).map((l) => l.replace(/^\[amdtest\] \S+ sweep /, '')),
      builds: fs.readFileSync(path.join(amd, 'builds.log'), 'utf8').split('\n').filter(Boolean)
        .map((l) => l.trimEnd()),
    };
  }

  bashIt('runs Stage A\'s line: three builds checked, then every round times each in turn', () => {
    const { status, lines, builds } = sweep({
      SWEEP: SWEEP_LINE, SWEEP_FOLD: '131072:4096:20', SWEEP_FOLD_ROUNDS: '4', SWEEP_ROUNDS: '0',
    });
    expect(status).toBe(0);
    // base, reg, store, then the default again.
    expect(builds).toEqual(['build', 'build -DPEARL_AMD_TWOLANE=1',
      'build -DPEARL_AMD_TWOLANE=1 -DPEARL_AMD_TWOLANE_STORE=1', 'build']);
    expect(lines[0]).toContain('fold timing 131072:4096:20 x 4 rounds in turn; hashrate 40s+10s x 0');
    expect(lines.filter((l) => / check: wmma12 +ok: 4096 regions of .* PASS/.test(l))).toHaveLength(3);
    expect(lines.find((l) => l.startsWith('reg built'))).toContain('regions of 32x8');
    // No variant is timed on its own right after its build.
    expect(lines.filter((l) => /^\w+ fold 131072 x 4096: /.test(l))).toEqual([]);
    const rounds = lines.filter((l) => l.startsWith('fold 131072 x 4096, round'));
    expect(rounds).toEqual([1, 2, 3, 4].map((i) => `fold 131072 x 4096, round ${i} of 4, TMAC/s: `
      + 'base 50.000, reg 50.750 (+1.50%), store 50.500 (+1.00%)'));
    // In turn: base, reg, store, base, reg, store, ...
    const order = lines.filter((l) => / fold 131072 x 4096, round /.test(l)).map((l) => l.split(' ')[0]);
    expect(order).toEqual([].concat(...[1, 2, 3, 4].map(() => ['base', 'reg', 'store'])));
    expect(lines).toContain('reg fold 131072 x 4096, round 2 of 4: wmma12 timed only: 2097152 regions of 32x8, '
      + '20 launches, 20.0 s, 50.750 TMAC/s');
    expect(lines[lines.length - 1]).toBe('done');
  });

  bashIt('times each variant once after its build when it isn\'t set, as before', () => {
    const { status, lines } = sweep({ SWEEP: SWEEP_LINE, SWEEP_FOLD: '131072:4096:20', SWEEP_ROUNDS: '0' });
    expect(status).toBe(0);
    expect(lines.filter((l) => /^\w+ fold 131072 x 4096: /.test(l)).map((l) => l.split(' ')[0]))
      .toEqual(['base', 'reg', 'store']);
    expect(lines.filter((l) => l.includes(', round '))).toEqual([]);
  });

  bashIt('refuses a SWEEP_FOLD_ROUNDS that isn\'t a whole number', () => {
    const check = SWEEP.split('\n').find((l) => l.startsWith('case "$FOLD_ROUNDS" in'));
    for (const [v, ok] of [['x', false], ['-1', false], ['', false], ['1.5', false], ['0', true], ['4', true]]) {
      const r = spawnSync('bash', ['-c', 'out() { echo "$*"; }\n' + check + '\necho went-on'],
        { env: { PATH: process.env.PATH, FOLD_ROUNDS: v }, encoding: 'utf8' });
      expect([v, r.status, r.stdout.includes('went-on')]).toEqual([v, ok ? 0 : 2, ok]);
    }
  });
});

describe('the readout census', () => {
  // A stand-in for hip-clang's assembly of pearl_amd_fold_wmma12: `copies` readouts, all but the
  // last inside the slab loop, then whatever `after` holds (the store form's zero fill).
  const acc = (i) => `v${i % 128}`;
  function chain(dst, len, from) {
    const out = [`v_xor3_b32 ${dst}, ${acc(from)}, ${acc(from + 1)}, ${acc(from + 2)}`];
    for (let i = 1; i < len; i++) out.push(`v_xor3_b32 ${dst}, ${dst}, ${acc(from + 2 * i + 1)}, ${acc(from + 2 * i + 2)}`);
    return out;
  }
  const READOUT = {
    // v_permlanex16 also reads its destination, and the word is still needed after it, so the
    // compiler copies it there first: 89 listed VALU a copy and 4 of those copies.
    base: () => [].concat(...[0, 1, 2, 3].map((w) => [
      ...chain(`v${140 + w}`, 16, 32 * w),
      `v_xor_b32_dpp v${140 + w}, v${140 + w}, v${140 + w} quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf`,
      `v_xor_b32_dpp v${140 + w}, v${140 + w}, v${140 + w} row_ror:8 row_mask:0xf bank_mask:0xf`,
      `v_mov_b32_e32 v${150 + w}, v${140 + w}`,
      `v_permlanex16_b32 v${150 + w}, v${140 + w}, s4, s5`,
      `v_xor_b32_e32 v${140 + w}, v${140 + w}, v${150 + w}`,
    ]), 'v_cmp_eq_u32_e32 vcc_lo, s6, v160',
    ...[0, 1, 2, 3, 4, 5, 6, 7].map((i) => `v_cndmask_b32_e32 v${170 + i}, v${170 + i}, v${140 + (i % 4)}, vcc_lo`)),
    reg: ({ oneChain = false, keep = ['s_mov_b32 m0, s7', 'v_movreld_b32_e32 v180, v140'], extra = [] } = {}) => [
      ...(oneChain ? chain('v140', 64, 0) : [].concat(...[0, 1, 2, 3].map((p) => chain(`v${141 + p}`, 16, 32 * p)))),
      ...(oneChain ? [] : ['v_xor3_b32 v140, v141, v142, v143', 'v_xor_b32_e32 v140, v140, v144']),
      'v_xor_b32_dpp v140, v140, v140 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf',
      ...keep,
      ...extra,
    ],
    store: ({ addr = 'v190, v140, s[8:9]', math = [], wait = false } = {}) => [
      ...READOUT.reg({ keep: [] }),
      ...math,
      ...(wait ? ['s_wait_storecnt 0x0'] : []),
      `global_store_b32 ${addr}`,
    ],
  };
  // The zero fill as the compiler may build it when it moves tr + trByte out of the loop: a
  // per-lane pointer stepped with v_add_co, and the "off" store form.
  const ZERO_FILL = [
    '.LBB4_3:',
    '\tv_add_co_u32 v230, vcc_lo, v230, 4',
    '\tv_add_co_ci_u32_e32 v231, vcc_lo, 0, v231, vcc_lo',
    '\tglobal_store_b32 v[230:231], v0, off',
    '\ts_add_co_i32 s20, s20, 1',
    '\ts_cmp_lg_u32 s20, 16',
    '\ts_cbranch_scc1 .LBB4_3',
  ];
  function asm(readout, { copies = 3, after = [], wmma = true } = {}) {
    const body = [
      'pearl_amd_fold_wmma12:                  ; @pearl_amd_fold_wmma12',
      '; %bb.0:',
      '\ts_load_b128 s[0:3], s[0:1], 0x0',
      '\t.p2align\t6',
      '.LBB4_1:                                ; =>This Inner Loop Header: Depth=1',
      wmma ? '\tv_wmma_i32_16x16x16_iu8 v[0:7], v[200:201], v[202:203], v[0:7]' : '\ts_nop 0',
      ...[].concat(...Array.from({ length: copies - 1 }, () => readout.map((l) => '\t' + l))),
      '\tv_add_co_u32 v210, vcc_lo, s10, v211',
      '\ts_barrier_signal -1',
      '\ts_barrier_wait -1',
      '\ts_cbranch_scc1 .LBB4_1',
      '; %bb.2:',
      ...readout.map((l) => '\t' + l),
      '\tglobal_store_b32 v[220:221], v170, off',
      ...after,
      '\ts_endpgm',
      '.Lfunc_end4:',
      'pearl_amd_fold_wmma11:',
      '\tv_permlanex16_b32 v1, v1, s0, s1',
      '.Lfunc_end5:',
    ];
    return body.join('\n');
  }
  const counted = (text) => census.count(census.parse(census.cutFunction(text, 'pearl_amd_fold_wmma12')));
  const RES = { vgprs: 198, scratch: 0, vgprSpill: 0, sgprSpill: 0, lds: 36864, occupancy: 6 };
  const res = { base: RES, reg: { ...RES, vgprs: 190 }, store: { ...RES, vgprs: 188 } };
  const misses = (c, r = res) => census.gates(c, r).filter((x) => x[4] === false).map((x) => x[0]);
  const VALU_ROW = 'slab-loop VALU (two-lane: <= base - 20 a copy)';
  const KEEP_ROW = 'keep step a copy, reg less store (<= 3)';
  const base = counted(asm(READOUT.base()));
  const reg = counted(asm(READOUT.reg()));
  const store = counted(asm(READOUT.store()));

  it('counts today\'s readout: 93 VALU a copy, with the copies before the lane swaps', () => {
    expect(base.permlane).toBe(12);
    expect(base.slabPermlane).toBe(8);
    expect(base.oldMov).toBe(12);
    expect(base.readout).toBe(3 * 93);
    // The slab loop: its v_wmma and v_add_co, and two readouts.
    expect(base.slabValu).toBe(2 + 2 * 93);
    expect(base.dpp).toBe(24);
    expect(base.chain).toBe(16);
    expect(base.storecntSlab).toBe(0);
    expect(base.addCoSlab).toBe(1);
  });

  it('passes the two forms as the decision wants them', () => {
    expect(reg.chain).toBe(17);
    expect(reg.readout).toBe(3 * 68);
    expect(reg.slabValu).toBe(2 + 2 * 68);
    // Only the slab loop's stores: the tail's are outside it.
    expect(store.store32).toBe(2);
    expect(store.store32Saddr).toBe(2);
    expect(misses({ base, reg, store })).toEqual([]);
  });

  it('passes a register form at the keep step\'s limit, which today\'s full count leaves room for', () => {
    // Read the word, select, write it back: 3 VALU, so 70 a copy against base's 93. Counting
    // only the listed opcodes put base at 89, and this at 19 saved, under the bar.
    const atLimit = counted(asm(READOUT.reg({
      keep: ['v_movrels_b32_e32 v181, v180', 'v_cndmask_b32_e32 v181, v181, v140, vcc_lo', 'v_movreld_b32_e32 v180, v181'],
    })));
    expect(misses({ base, reg: atLimit, store })).toEqual([]);
    const rows = census.gates({ base, reg: atLimit, store }, res);
    expect(rows.find((x) => x[0] === '  saved a copy')[2]).toBe('23.0');
  });

  it('misses a single 64-deep chain and a costly keep step, copies included', () => {
    expect(misses({ base, reg: counted(asm(READOUT.reg({ oneChain: true }))), store }))
      .toEqual(['longest dependent v_xor3 chain (two-lane: <= 17)']);
    const selects = Array.from({ length: 8 }, (_, i) => `v_cndmask_b32_e32 v${180 + i}, v${180 + i}, v140, vcc_lo`);
    expect(misses({ base, reg: counted(asm(READOUT.reg({ keep: selects }))), store })).toEqual([VALU_ROW, KEEP_ROW]);
    // Plain copies of the keep registers around the indexed write count too.
    const copies = ['v_mov_b32_e32 v190, v180', 'v_mov_b32_e32 v191, v181', 'v_mov_b32_e32 v192, v182',
      's_mov_b32 m0, s7', 'v_movreld_b32_e32 v180, v140'];
    expect(misses({ base, reg: counted(asm(READOUT.reg({ keep: copies }))), store })).toEqual([KEEP_ROW]);
  });

  it('misses VALU the compiler adds that no opcode list names', () => {
    const extra = Array.from({ length: 12 }, (_, i) => `v_add_nc_u32_e32 v${200 + i}, s9, v${200 + i}`);
    expect(misses({ base, reg: counted(asm(READOUT.reg({ extra }))), store })).toEqual([VALU_ROW]);
  });

  it('misses an addressed store and a store wait in the slab loop', () => {
    const addressed = counted(asm(READOUT.store({
      addr: 'v[190:191], v140, off',
      math: ['v_add_co_u32 v190, vcc_lo, s8, v192', 'v_add_co_ci_u32_e32 v191, vcc_lo, s9, v193, vcc_lo'],
    })));
    expect(misses({ base, reg, store: addressed }))
      .toEqual(['slab-loop global_store_b32: all / SGPR base (store: all)', 'slab-loop v_add_co* (store: <= base)']);
    expect(misses({ base, reg, store: counted(asm(READOUT.store({ wait: true }))) }))
      .toEqual(['slab-loop s_wait_storecnt* (store: 0)']);
    expect(misses({ base, reg, store: counted(asm(READOUT.base())) }))
      .toEqual(expect.arrayContaining(['v_permlanex16 (two-lane: 0)', 'DPP ops (two-lane: N)']));
    expect(misses({ base, reg, store }, { ...res, reg: { ...RES, vgprs: 217 } })).toEqual(['VGPRs (<= 216; target 192)']);
    expect(misses({ base, reg, store }, { ...res, store: { ...RES, scratch: 4 } })).toEqual(['scratch bytes / spills (0)']);
  });

  it('leaves out loops after the slab loop, like the store form\'s zero fill', () => {
    const filled = counted(asm(READOUT.store(), { after: ZERO_FILL }));
    expect(filled.store32).toBe(2);
    expect(filled.addCoSlab).toBe(1);
    expect(misses({ base, reg, store: filled })).toEqual([]);
  });

  it('misses a build whose slab loop it can\'t find', () => {
    const lost = counted(asm(READOUT.store(), { wmma: false }));
    expect(lost.slabInsts).toBe(0);
    expect(misses({ base, reg, store: lost })).toEqual(expect.arrayContaining(['slab loop (the loop with the v_wmma) found']));
    expect(misses({ base: lost, reg, store })).toEqual(expect.arrayContaining(
      ['copies, from base: function (N) / slab loop', VALU_ROW]));
  });

  it('reads the resource remarks build-amd.sh check reads', () => {
    const t = '-Rpass-analysis=kernel-resource-usage';
    const remarks = ['pearl_amd_fold_wmma11', 'pearl_amd_fold_wmma12'].map((f, i) => [
      `x.cpp:1:1: remark: Function Name: ${f} [${t}]`,
      `x.cpp:1:1: remark:     VGPRs: ${190 + i} [${t}]`,
      `x.cpp:1:1: remark:     AGPRs: 0 [${t}]`,
      `x.cpp:1:1: remark:     ScratchSize [bytes/lane]: ${i ? 0 : 8} [${t}]`,
      `x.cpp:1:1: remark:     Occupancy [waves/SIMD]: 6 [${t}]`,
      `x.cpp:1:1: remark:     SGPRs Spill: 0 [${t}]`,
      `x.cpp:1:1: remark:     VGPRs Spill: 0 [${t}]`,
      `x.cpp:1:1: remark:     LDS Size [bytes/block]: 36864 [${t}]`,
    ].join('\n')).join('\n');
    expect(census.resources(remarks, 'pearl_amd_fold_wmma12'))
      .toEqual({ vgprs: 191, scratch: 0, occupancy: 6, sgprSpill: 0, vgprSpill: 0, lds: 36864 });
  });

  it('compiles the decision\'s three builds, and CI runs it with the switched-on check', () => {
    expect(CENSUS_SH).toContain('reg) flags="-DPEARL_AMD_TWOLANE=1" ;;');
    expect(CENSUS_SH).toContain('store) flags="-DPEARL_AMD_TWOLANE=1 -DPEARL_AMD_TWOLANE_STORE=1" ;;');
    expect(CENSUS_SH).toContain('--cuda-device-only -S');
    const step = CI.slice(CI.indexOf('- name: Two-lane regions on RDNA 4'));
    expect(step).toContain('continue-on-error: true');
    expect(step).toContain('PEARL_AMD_CFLAGS=-DPEARL_AMD_TWOLANE=1 ./build-amd.sh check gfx1200 gfx1201 || rc=1');
    expect(step).toContain('bash test/readout-census.sh gfx1200 gfx1201 || rc=1');
  });
});
