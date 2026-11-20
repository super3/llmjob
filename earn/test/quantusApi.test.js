'use strict';

// Core API 2 (the nonce layout of src/shared/quantus/qpow.js) across the host
// and the NVIDIA and AMD cores. They change together: a core of another version
// computes nothing useful from this host's work, so each side must say its
// version and refuse the other's. The Intel core stays on version 1 until
// version 2 has run on an Arc card, and the host speaks both. These tests read the native sources, which
// a runner without a card cannot run, and pin what has to agree.

const fs = require('fs');
const path = require('path');
const q = require('../src/shared/quantus/qpow');
const qc = require('../src/main/quantusCore');
const p2 = require('../src/shared/quantus/poseidon2');

const NATIVE = path.join(__dirname, '../native/quantus');
const read = (...p) => fs.readFileSync(path.join(NATIVE, ...p), 'utf8').replace(/\r\n/g, '\n');
const number = (text, re) => {
  const m = re.exec(text);
  if (!m) throw new Error('not found: ' + re);
  return Number(m[1].replace(/u$/, ''));
};

describe('core API 2 on NVIDIA and AMD, 1 on Intel', () => {
  const napi = { cuda: read('cuda', 'qc_napi.c'), amd: read('amd', 'qh_napi.cc') };
  const sycl = read('sycl', 'quantus_sycl.cpp');

  test('the host and each core export the same version', () => {
    expect(qc.API).toBe(2);
    expect(qc.API_V1).toBe(1);
    expect(qc.V1_FILES).toEqual(['quantus_sycl.node']);
    expect(number(napi.cuda, /#define QC_API (\d+)/)).toBe(qc.API);
    expect(number(napi.amd, /exports\.Set\("QUANTUS_API", Napi::Number::New\(env, (\d+)\)\)/)).toBe(qc.API);
    expect(number(sycl, /constexpr int API = (\d+);/)).toBe(qc.API_V1);
  });

  test('the Intel core takes version 1\'s work', () => {
    expect(sycl).toContain('"midstate must be 96 bytes"');
    expect(sycl).toContain('"tail must be 24 bytes"');
    expect(sycl).not.toContain('"line must be 112 bytes"');
  });

  test('each core takes a line of 112 bytes, and nothing of version 1\'s work', () => {
    expect(q.LINE_BYTES).toBe(112);
    for (const src of Object.values(napi)) {
      expect(src).toContain('"line must be 112 bytes"');
      expect(src).toMatch(/line element .* is not below the field prime/);
      expect(src).toContain('start + count passes 2^32');
      expect(src).not.toMatch(/"midstate"|"tail"/);
    }
    expect(number(read('cuda', 'qc_host.h'), /#define QC_LINE (\d+)/)).toBe(14);
    expect(number(read('amd', 'qh_host.h'), /#define QH_LINE (\d+)/)).toBe(14);
  });

  test('CI\'s check of each built core asks for this version', () => {
    const yml = fs.readFileSync(path.join(__dirname, '../../.github/workflows/native-core.yml'), 'utf8');
    const asked = [...yml.matchAll(/a\.QUANTUS_API !== (\d+)/g)].map((m) => Number(m[1]));
    expect(asked.filter((v) => v === qc.API).length).toBeGreaterThanOrEqual(6);
    // The two version 1 checks are the Intel core's, Linux and Windows.
    expect(asked.filter((v) => v !== qc.API)).toEqual([qc.API_V1, qc.API_V1]);
    const v1 = yml.split('\n').filter((l) => l.includes('a.QUANTUS_API !== 1'));
    expect(v1[0]).toContain("require('./quantus_sycl.node')");
    const win = yml.indexOf('addon=$(cygpath -w "$PWD/quantus_sycl.node")');
    expect(yml.lastIndexOf('a.QUANTUS_API !== 1', win)).toBeGreaterThan(yml.lastIndexOf('a.QUANTUS_API !== 2', win));
  });

  test('the loader says which version a refused file speaks', () => {
    const log = jest.fn();
    const old = { QUANTUS_API: 1, devices: () => [{ index: 0, name: 'x' }], createCore: () => ({}) };
    expect(qc.quantusCoreFactory({ env: { QUANTUS_CORE_PATH: '/x/old.node' }, require: () => old, log })).toBe(null);
    expect(log).toHaveBeenCalledWith('warn', 'Quantus core: /x/old.node speaks Quantus core API 1, this miner 2');
  });

  test('the loader takes the Intel core at version 1 and says so in `api`', () => {
    const old = { QUANTUS_API: 1, devices: () => [{ index: 0, name: 'Arc' }], createCore: () => ({}) };
    const f = qc.quantusCoreFactory({ env: { QUANTUS_CORE_PATH: '/x/quantus_sycl.node' }, require: () => old });
    expect(f.api).toBe(1);
    expect(f.label).toBe('quantus_sycl');
    const now = Object.assign({}, old, { QUANTUS_API: 2 });
    expect(qc.quantusCoreFactory({ env: { QUANTUS_CORE_PATH: '/x/quantus_sycl.node' }, require: () => now }).api).toBe(2);
  });
});

describe('the line each kernel walks', () => {
  const cuda = read('cuda', 'qc_kernel.cuh');
  const amd = read('amd', 'qh_kernel.h');

  test('one step is 35 on element 0 in every kernel, as in the reference', () => {
    expect(number(cuda, /#define QC_SLOPE (\d+u)/)).toBe(Number(q.SLOPE));
    expect(number(amd, /#define QH_SLOPE (\d+u)/)).toBe(Number(q.SLOPE));
  });

  // The CUDA and AMD kernels work out 35 i for a launch's point i in a u32.
  test('35 times a launch\'s last point fits a u32 on CUDA and AMD', () => {
    const chunk = number(read('cuda', 'qc_host.cu'), /#define QC_CHUNK \(1u << (\d+)\)/);
    const launch = number(read('amd', 'qh_host.cpp'), /#define QH_LAUNCH_N \(1u << (\d+)\)/);
    expect(35 * 2 ** chunk).toBeLessThanOrEqual(0xffffffff);
    expect(35 * 2 ** launch).toBeLessThanOrEqual(0xffffffff);
    expect(cuda).toContain('const u32 d = QC_SLOPE * i;');
    expect(amd).toContain('const u64 d = (u64)(QH_SLOPE * i);');
  });

  // qc_add_small is exact only for an addend of at most p (quantusCudaConstants.test.js).
  test('the CUDA kernel\'s two adds are inside qc_add_small\'s bound', () => {
    expect(cuda).toContain('qc_add_small(L.x0, (u64)d)');
    expect(cuda).toContain('qc_add_small(L.x4, QC_P - (u64)d)');
    expect(BigInt(35 * 2 ** 26) <= p2.P).toBe(true);
  });

  test('each kernel starts its rounds at the first permutation\'s second round', () => {
    // CUDA: the two-pass loop skips round 0 of pass 0; the compact loop starts at step 1 and skips its layer.
    expect(cuda).toContain('for (int r = pass ? 0 : 1; r < 4; r++) qc_full(s, QC_RC_F1[r], two);');
    expect(cuda).toContain('for (int k = 1; k < 17; k++) {');
    expect(cuda).toContain('if (k > 1) qc_ext(s, QC_RC_SEQ[j], two);');
    expect(cuda.match(/qc_first_round\(L, i( \+ n)?, /g)).toHaveLength(2);
    // AMD: row 0 is not in the loop.
    expect(amd).toContain('int row = 1;');
    expect(amd).toContain('for (int r = seg ? 0 : 1; r < rows; r++, row++) {');
    expect(amd).toContain('qh_first_round(a, i0 + n, s[n]);');
  });

  // A model of what the kernels compute from a line, in their own form: sums of
  // a = x0^7 and b = x4^7 times (2, 1, 1, 3), against the reference state.
  test('the twelve sums the kernels make are the reference\'s state after the first round', () => {
    const mid = q.midstate(Buffer.alloc(32, 9), Buffer.alloc(32, 3));
    const work = q.lineWork(mid, q.lineBase(42n));
    for (const t of [0, 5, 2 ** 26 - 1, q.SPAN - 1]) {
      const d = (35n * BigInt(t)) % p2.P;
      const a = p2.exp7((work[12] + d) % p2.P);
      const b = p2.exp7((work[13] + p2.P - d) % p2.P);
      const v = [2n * a + b, a + 2n * b, a + b];
      const s = [];
      for (let g = 0; g < 3; g++) {
        s.push((2n * v[g] + work[4 * g]) % p2.P, (v[g] + work[4 * g + 1]) % p2.P, (v[g] + work[4 * g + 2]) % p2.P,
          (3n * v[g] + work[4 * g + 3]) % p2.P);
      }
      expect(s).toEqual(q.indexState(work, t));
    }
  });
});
