'use strict';

// The SYCL core for Intel GPUs (native/sycl) cannot be built or run here, so,
// as nativeConfig.test.js does for the CUDA core, these tests read its sources
// and pin what must agree with the rest of the repo: the C API pearl_core.cc
// calls, the structs the two files share, and the region numbering its folds
// assume. They also run the harness scripts that need no device.

const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');
const { PROFILE, ROWS_MASK, COLS_MASK, expandOffset, regionToTile } = require('../src/shared/miner/pearlhash');

const NATIVE = path.join(__dirname, '..', 'native');
const SYCL = path.join(NATIVE, 'sycl');
const read = (...p) => fs.readFileSync(path.join(...p), 'utf8');
const CORE = read(NATIVE, 'src', 'pearl_core.cc');
const HOST = read(SYCL, 'pearl_sycl_host.cpp');

// The body of `struct Name { ... };` with comments and whitespace removed.
function structBody(src, name) {
  const at = src.indexOf('struct ' + name + ' {');
  if (at < 0) return null;
  const end = src.indexOf('};', at);
  return src.slice(at, end)
    .split('\n').map((l) => l.replace(/\/\/.*$/, '').trim()).filter(Boolean).join(' ');
}

describe('SYCL core sources', () => {
  test('define every pearl_host_* function pearl_core.cc declares', () => {
    const block = CORE.slice(CORE.indexOf('extern "C" {'), CORE.indexOf('namespace {'));
    const declared = [...new Set(block.match(/pearl_host_[a-z_]+(?=\()/g))];
    expect(declared.length).toBeGreaterThanOrEqual(10);
    for (const fn of declared) {
      expect(HOST).toMatch(new RegExp('extern "C" [^;{]*\\b' + fn + '\\('));
    }
  });

  test('share the result structs with pearl_core.cc member for member', () => {
    for (const name of ['PearlProofSide', 'PearlSearchResult']) {
      const want = structBody(CORE, name);
      expect(want).not.toBeNull();
      expect(structBody(HOST, name)).toBe(want);
    }
  });
});

describe('the fold windows', () => {
  // The SYCL folds work in 32 x 64 windows of 8 regions: region q of window
  // (wr, wc) is row index 2wr + (q & 1) and column index 4wc + (q >> 1), at row
  // offset 32wr + 4(q & 1) and column offset 64wc + 2(q >> 1). Checked here
  // against the protocol's own enumeration of valid offsets.
  test('match the valid offsets regionToTile enumerates', () => {
    for (let rowIdx = 0; rowIdx < 512; rowIdx++) {
      const wr = rowIdx >> 1, q0 = rowIdx & 1;
      expect(expandOffset(rowIdx, ROWS_MASK)).toBe(32 * wr + 4 * q0);
    }
    for (let colIdx = 0; colIdx < 512; colIdx++) {
      const wc = colIdx >> 2, q1 = colIdx & 3;
      expect(expandOffset(colIdx, COLS_MASK)).toBe(64 * wc + 2 * q1);
    }
  });

  test('hold whole regions that tile the window exactly', () => {
    const p = Object.assign({}, PROFILE, { m: 64, n: 128 });
    const rowsValid = p.m / 16;
    const seen = new Set();
    for (let region = 0; region < rowsValid * (p.n / 16); region++) {
      const { rows, cols } = regionToTile(region, p);
      const rowIdx = region % rowsValid, colIdx = Math.floor(region / rowsValid);
      const wr = rowIdx >> 1, wc = colIdx >> 2;
      for (const r of rows) for (const c of cols) {
        expect(r >> 5).toBe(wr);
        expect(c >> 6).toBe(wc);
        const key = r + ',' + c;
        expect(seen.has(key)).toBe(false);
        seen.add(key);
      }
    }
    expect(seen.size).toBe(p.m * p.n);
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
