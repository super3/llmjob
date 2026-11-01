'use strict';

const fs = require('fs');
const path = require('path');

// Ampere's flip (PEARL_AMPERE_FLIP): on cards whose largest persisting L2 slice holds the B'
// of a narrow batch, the host runs pearl_tile_fold_tall_pinb, which holds B' instead of a
// band of A'. Which cards take it is pearl_ampere_flip_for's byte model. The native side
// cannot run here, so its source is checked for the model, and a JS transcription of the
// model is checked against what Salad measured on 31 October (race mode): the 3060 Ti +1.92%
// at 64 offsets and the 3090 +0.66% at 128 take it; the 3060 (-0.63% at 32) and the 3080
// 10 GB (+0.03% at 64) don't, nor the 3070, whose slice allows the 3080's width.

const SRC = path.join(__dirname, '..', 'native', 'src');
const HOST = fs.readFileSync(path.join(SRC, 'pearl_host.cu'), 'utf8');
const CONFIG = fs.readFileSync(path.join(SRC, 'pearl_config.h'), 'utf8');
const KERNEL = fs.readFileSync(path.join(SRC, 'pearl_kernel.cu'), 'utf8');
const FOLD_TALL = fs.readFileSync(path.join(SRC, 'pearl_fold_tall.cuh'), 'utf8');
const NL = String.fromCharCode(10);

function body(src, signature) {
  const at = src.indexOf(signature);
  expect(at).toBeGreaterThanOrEqual(0);
  return src.slice(at, src.indexOf(NL + '}', at));
}

// pearl_ampere_flip_for, line for line.
const K = 2048, COLS = 16, TALL_BM = 192, TALL_COL_OFFSETS = 16, SHARE = 80;
function flipFor(rowGroups, band, colBatch, colsValid, maxSlice) {
  const colBytes = COLS * K;
  let width = 0;
  for (let c = TALL_COL_OFFSETS; c < colBatch && c * colBytes <= maxSlice; c *= 2)
    if (colsValid % c === 0) width = c;
  if (width === 0) return 0;
  const aBytes = rowGroups * TALL_BM * K;
  const bandWalk = Math.ceil(rowGroups / band) * colBytes + Math.floor(aBytes / colBatch);
  const flip = Math.floor(aBytes / width);
  return flip * 100 <= bandWalk * SHARE ? width : 0;
}
// pearl_ampere_band_for at PEARL_AMPERE_BAND_L2_SHARE 80: 16, halved while the band's A'
// is more than 80% of the L2, never below 4.
function bandFor(l2) {
  let band = 16;
  while (band > 4 && band * TALL_BM * K * 100 > l2 * 80) band /= 2;
  return band;
}

describe('Ampere flip', () => {
  test('the host model is the one transcribed here', () => {
    const fn = body(HOST, 'uint32_t pearl_ampere_flip_for(');
    expect(fn).toContain('if (!PEARL_AMPERE_FLIP || band == 0u || maxSlice == 0u) return 0u;');
    expect(fn).toContain('const uint64_t colBytes = (uint64_t)PEARL_COLS_COUNT * k;');
    expect(fn).toContain('for (uint32_t c = PEARL_TALL_COL_OFFSETS; c < colBatch && (uint64_t)c * colBytes <= maxSlice; c *= 2u)');
    expect(fn).toContain('if (colsValid % c == 0u) width = c;');
    expect(fn).toContain('const uint64_t aBytes = (uint64_t)rowGroups * PEARL_TALL_BM * k;');
    expect(fn).toContain('const uint64_t bandWalk = (uint64_t)((rowGroups + band - 1u) / band) * colBytes + aBytes / colBatch;');
    expect(fn).toContain('const uint64_t flip = aBytes / width;');
    expect(fn).toContain('return flip * 100u <= bandWalk * (uint64_t)PEARL_AMPERE_FLIP_SHARE ? width : 0u;');
    expect(CONFIG).toContain('#define PEARL_AMPERE_FLIP_SHARE 80u');
    expect(CONFIG).toContain('#define PEARL_AMPERE_FLIP 1');
  });

  // L2 and largest slice as the driver reports them (race mode's AUDIT lines, 31 October;
  // the 3070's slice is 11/16 of its 4 MB, as on the others). m = 131072: 8192 valid row
  // offsets, 683 row groups of 12; n = 262144: 16384 valid column offsets; batch 512.
  test.each([
    ['RTX 3060', 2359296, 1622016, 0],
    ['RTX 3060 Ti', 3145728, 2162688, 64],
    ['RTX 3070', 4194304, 2883584, 0],
    ['RTX 3080 10 GB', 5242880, 3604480, 0],
    ['RTX 3090', 6291456, 4325376, 128],
  ])('%s: flip width %#', (card, l2, maxSlice, want) => {
    expect(flipFor(683, bandFor(l2), 512, 16384, maxSlice)).toBe(want);
  });

  test('the flip kernel holds B, the other A, and only sm_86 builds its body', () => {
    const first = KERNEL.indexOf('#define PEARL_TALL_FN pearl_tile_fold_tall');
    const second = KERNEL.indexOf('#define PEARL_TALL_FN pearl_tile_fold_tall_pinb');
    expect(first).toBeGreaterThan(0);
    expect(KERNEL.indexOf('#define PEARL_TALL_PIN_B 0', first)).toBeLessThan(second);
    expect(KERNEL.indexOf('#define PEARL_TALL_PIN_B 1', second)).toBeGreaterThan(second);
    expect(KERNEL.split('#include "pearl_fold_tall.cuh"').length - 1).toBe(2);
    expect(FOLD_TALL).toContain('#if PEARL_AMPERE_PERSIST_A && !PEARL_TALL_PIN_B && defined(__CUDA_ARCH__)');
    expect(FOLD_TALL).toContain('#if PEARL_TALL_PIN_B && defined(__CUDA_ARCH__) && PEARL_TALL_PIN_B_ARCH(__CUDA_ARCH__)');
    expect(FOLD_TALL).toContain('"l"(bP + (sg) * bKB + (int)((p) * bStep * SK) - (int)BMID), "l"(polB));');
    expect(CONFIG).toContain('#define PEARL_TALL_PIN_B_ARCH(a) ((a) == 860)');
    // The host runs it only where sm_86's build is loaded, and launches it by its own name.
    expect(HOST).toContain('if (ctx->foldPersistA && ctx->tallArch == 86 && !ctx->foldUnfused) {');
    expect(HOST).toContain('if (ctx->foldPinB) foldFn = reinterpret_cast<const void *>(pearl_tile_fold_tall_pinb);');
    expect(HOST).toContain('pearl_tile_fold_tall_pinb<<<blocks, threads, smem, st>>>(');
  });
});
