'use strict';

// PEARL_AMD_W11K (earn/native/amd/src/pearl_amd_fold_w11k.h): RDNA 3's fold in wave64. It read level with the wave32
// fold on an RX 7900 XTX (earn/tuning/benchmark.md, 20 November), so no build turns it on: it needs a gfx1100 or
// gfx1101 device pass given -mwavefrontsize64 -DPEARL_AMD_WAVE64=1 by hand. These pin that a release still builds
// the wave32 fold on RDNA 3, and that the fold's own writes of exec come in pairs. No HIP compiler runs here, so the
// C++ is read as text.

const fs = require('fs');
const path = require('path');

const NATIVE = path.join(__dirname, '..', 'native');
const read = (...f) => fs.readFileSync(path.join(NATIVE, ...f), 'utf8').replace(/\r\n/g, '\n');
const FOLD = read('amd', 'src', 'pearl_amd_fold.h');
const W11K = read('amd', 'src', 'pearl_amd_fold_w11k.h');
const HOST = read('amd', 'src', 'pearl_host_hip.cpp');
const BUILD = read('amd', 'build-amd.sh');
// Code without its comments: what the compiler reads.
const code = (t) => t.split('\n').map((l) => l.replace(/\/\/.*$/, '')).join('\n');

describe('PEARL_AMD_W11K, RDNA 3\'s wave64 fold', () => {
  it('is on only in a gfx1100 or gfx1101 pass the build says is wave64', () => {
    expect(W11K).toContain('#ifndef PEARL_AMD_W11K\n'
      + '#if PEARL_AMD_WMMA11 && PEARL_AMD_GFX1100_1101 && !PEARL_AMD_EMU && defined(PEARL_AMD_WAVE64) && PEARL_AMD_WAVE64 && \\\n'
      + '    PEARL_AMD_RDNA3_SET && PEARL_AMD_GB11 && PEARL_AMD_BUFLD && PEARL_AMD_TILED && !PEARL_AMD_STAMP && !PEARL_AMD_ABLATE && \\\n'
      + '    !PEARL_AMD_KWRAP\n'
      + '#define PEARL_AMD_W11K 1\n#else\n#define PEARL_AMD_W11K 0\n#endif\n#endif\n');
    expect(code(W11K).match(/#define PEARL_AMD_W11K 1/g)).toHaveLength(1);
    // A wave64 gfx11 pass that would build the wave32 fold stops the compiler.
    expect(W11K).toContain('#if !PEARL_AMD_W11K && PEARL_AMD_WMMA11 && !PEARL_AMD_EMU && defined(PEARL_AMD_WAVE64) && PEARL_AMD_WAVE64\n#error ');
  });

  it('is built by no release: build-amd.sh gives RDNA 3 no wave size, and nothing sets PEARL_AMD_W11K', () => {
    const arch = BUILD.slice(BUILD.indexOf('\narch_flags() {\n'), BUILD.indexOf('\n}\n', BUILD.indexOf('\narch_flags() {\n')));
    expect(arch).toContain('gfx120[01]:*) echo "-mwavefrontsize64 -DPEARL_AMD_WAVE64=1" ;;');
    expect(arch).not.toMatch(/gfx11/);
    expect(BUILD).not.toMatch(/PEARL_AMD_W11K/);
    expect(code(HOST)).not.toMatch(/W11K/);
    expect(code(FOLD.replace('#include "pearl_amd_fold_w11k.h"', ''))).not.toMatch(/#define PEARL_AMD_W11K\b/);
  });

  it('reaches the wmma11 kernel through one include and one call, and leaves the wave32 call as it was', () => {
    expect(FOLD.match(/#include "pearl_amd_fold_w11k\.h"/g)).toHaveLength(1);
    // After the folds' namespace closes and before the kernels.
    const inc = FOLD.indexOf('#include "pearl_amd_fold_w11k.h"');
    expect(inc).toBeGreaterThan(FOLD.indexOf('\n}  // namespace\n'));
    expect(inc).toBeLessThan(FOLD.indexOf('#define PEARL_AMD_FOLD_PARAMS'));
    expect(FOLD).toContain('#if PEARL_AMD_W11K\n'
      + '  pearl_amd_fold_w11k(Ap, Bp, k, rank, col0, rows_valid, tr PEARL_AMD_FUSE_ARGS);   // PEARL_AMD_W11K (above)\n'
      + '#elif PEARL_AMD_EMU || PEARL_AMD_WMMA11\n'
      + '  pearl_amd_fold_body<PearlAmdWmma11>(Ap, Bp, k, rank, col0, rows_valid, tr PEARL_AMD_FUSE_ARGS);\n#else\n');
    // The occupancy attribute is there only with the wave64 fold.
    expect(FOLD).toContain('#if PEARL_AMD_W11K\n#define PEARL_AMD_WMMA11_OCC __attribute__((amdgpu_waves_per_eu(4)))');
    expect(FOLD).toContain('#else\n#define PEARL_AMD_WMMA11_OCC\n#endif\n');
  });

  it('keeps the host\'s tile, threads and slab: one row of 4 waves64 over 128 x 256', () => {
    const c = code(W11K);
    expect(c).toContain('constexpr int W = 64, MT = 8, NT = 4, WM = 16 * MT, WN = 16 * NT;');
    expect(c).toContain('constexpr int BM = (int)PEARL_AMD_BM, BN = (int)PEARL_AMD_BN, T = (int)PEARL_AMD_THREADS;');
    expect(c).toContain('static_assert(BM == WM && BN == 4 * WN && T == 4 * W,');
    expect(c).toContain('constexpr int KS = (int)PEARL_AMD_SLAB,');
  });

  it('restores exec after every write of it', () => {
    const c = code(W11K);
    const writes = c.match(/asm volatile\("s_mov_b(?:32|64) exec[^"]*"/g);
    expect(writes).not.toBeNull();
    const kinds = {};
    for (const w of writes) {
      const k = /"(.*)"/.exec(w)[1];
      kinds[k] = (kinds[k] || 0) + 1;
    }
    // Three masks (lanes 0-31, A's 16 lanes, the four keeper lanes) and one restore, which sets all 64 lanes.
    expect(Object.keys(kinds).sort()).toEqual(['s_mov_b32 exec_hi, 0', 's_mov_b32 exec_lo, 0xAAAA5555', 's_mov_b64 exec, -1',
      's_mov_b64 exec, 0x55']);
    // The burst macros: each opening is closed by PEARL_AMD_W11K_ALL in the same block of the slab step.
    const opens = (c.match(/PEARL_AMD_W11K_A?LO\(\);/g) || []).length;
    const closes = (c.match(/PEARL_AMD_W11K_ALL\(\);/g) || []).length;
    expect(opens).toBeGreaterThan(0);
    expect(closes).toBeGreaterThanOrEqual(opens - 2);   // two places pick one of two openings, then close once
    // The keeper stores: opened and closed in the readout, eight stores between.
    const ro = c.slice(c.indexOf('asm volatile("s_mov_b64 exec, 0x55"'), c.indexOf('asm volatile("s_mov_b64 exec, -1"', c.indexOf('asm volatile("s_mov_b64 exec, 0x55"')));
    expect(ro).toContain('w[(2 * rb + ro) * (int)PEARL_JACKPOT_BUCKETS] = v[rb][ro];');
    expect(ro).not.toMatch(/__builtin_amdgcn_(update_dpp|permlane)/);
  });
});
