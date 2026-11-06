'use strict';

const fs = require('fs');
const path = require('path');

// Ampere's flip (PEARL_AMPERE_FLIP): on cards whose largest persisting L2 slice holds the B'
// of a narrow batch, the host runs pearl_tile_fold_tall_pinb, which holds B' instead of a
// band of A'. Which cards take it is pearl_ampere_flip_for's byte model. The native side
// cannot run here, so its source is checked for the model, and a JS transcription of the
// model is checked against what Salad measured on 31 October (race mode): the 3060 Ti +1.92%
// at 64 offsets and the 3090 +0.66% at 128 take it; the 3060 and the 3080 10 GB don't (their
// widths, 32 and 64, read -0.63% and +0.03% as strips of a 512 batch), nor the 3070, whose
// slice allows the 3080's width.

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

// pearl_ampere_flip_for, line for line. `fill` is PEARL_AMPERE_FLIP_FILL (issue #287, item 3):
// any whole number of 16-offset tiles whose B' fits the slice, not only powers of two that
// divide the valid offsets.
const K = 2048, COLS = 16, TALL_BM = 192, TALL_COL_OFFSETS = 16, SHARE = 80;
function flipFor(rowGroups, band, colBatch, colsValid, maxSlice, { fill = false, share = SHARE } = {}) {
  const colBytes = COLS * K;
  let width = 0;
  if (fill) {
    if (colsValid % TALL_COL_OFFSETS === 0)
      for (let c = TALL_COL_OFFSETS; c < colBatch && c <= colsValid && c * colBytes <= maxSlice;
        c += TALL_COL_OFFSETS)
        width = c;
  } else {
    for (let c = TALL_COL_OFFSETS; c < colBatch && c * colBytes <= maxSlice; c *= 2)
      if (colsValid % c === 0) width = c;
  }
  if (width === 0) return 0;
  const aBytes = rowGroups * TALL_BM * K;
  const bandWalk = Math.ceil(rowGroups / band) * colBytes + Math.floor(aBytes / colBatch);
  const flip = Math.floor(aBytes / width);
  return flip * 100 <= bandWalk * share ? width : 0;
}
// The model's flip bytes as a percentage of the band walk's, at a given width.
function flipShare(rowGroups, band, colBatch, width) {
  const colBytes = COLS * K;
  const aBytes = rowGroups * TALL_BM * K;
  const bandWalk = Math.ceil(rowGroups / band) * colBytes + Math.floor(aBytes / colBatch);
  return (100 * Math.floor(aBytes / width)) / bandWalk;
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

  test('the fill rule and the forced width are in the host, and off by default', () => {
    const fn = body(HOST, 'uint32_t pearl_ampere_flip_for(');
    expect(fn).toContain('#if PEARL_AMPERE_FLIP_FILL');
    expect(fn).toContain('if (colsValid % PEARL_TALL_COL_OFFSETS == 0u)');
    expect(fn).toContain('c < colBatch && c <= colsValid && (uint64_t)c * colBytes <= maxSlice;');
    expect(fn).toContain('c += PEARL_TALL_COL_OFFSETS)');
    expect(fn).toContain('#ifdef PEARL_AMPERE_FLIP_WIDTH');
    expect(CONFIG).toContain('#define PEARL_AMPERE_FLIP_FILL 0');
    expect(CONFIG).not.toMatch(/#define PEARL_AMPERE_FLIP_WIDTH/);
  });

  // With the fill rule a card's width is the widest whole number of tiles its slice holds.
  // The 3060 Ti and the 6 MB cards keep the width they mine with now. The others get one that
  // fills more of the slice, and at the shipped share none of them takes it: what the rule
  // picks is unchanged until a race raises the share.
  test.each([
    ['RTX 3060', 2359296, 1622016, 48, 91.3],
    ['RTX 3060 Ti', 3145728, 2162688, 64, 68.5],
    ['RTX 3070', 4194304, 2883584, 80, 100.4],
    ['RTX 3080 10 GB', 5242880, 3604480, 96, 83.7],
    ['RTX 3090', 6291456, 4325376, 128, 62.8],
  ])('%s: fill width %i, %d%% of the band walk\'s bytes', (card, l2, maxSlice, width, pct) => {
    const band = bandFor(l2);
    expect(flipFor(683, band, 512, 16384, maxSlice, { fill: true, share: 1000 })).toBe(width);
    expect(flipShare(683, band, 512, width)).toBeCloseTo(pct, 1);
    expect(flipFor(683, band, 512, 16384, maxSlice, { fill: true }))
      .toBe(flipFor(683, band, 512, 16384, maxSlice));
  });

  test('the share a card needs to take its fill width', () => {
    // 3080 10 GB: 96 offsets from a share of 84; 3060: 48 from 92.
    expect(flipFor(683, bandFor(5242880), 512, 16384, 3604480, { fill: true, share: 83 })).toBe(0);
    expect(flipFor(683, bandFor(5242880), 512, 16384, 3604480, { fill: true, share: 84 })).toBe(96);
    expect(flipFor(683, bandFor(2359296), 512, 16384, 1622016, { fill: true, share: 91 })).toBe(0);
    expect(flipFor(683, bandFor(2359296), 512, 16384, 1622016, { fill: true, share: 92 })).toBe(48);
  });

  test('a width that does not divide the salt cuts its last batch short', () => {
    // pearl_host_submit's cap, transcribed: batches of 96 from col_off 0 cover the 16384
    // valid offsets in 170 batches of 96 and one of 64, and end exactly on the salt.
    expect(HOST).toContain('if (ctx->foldPinB && col_off + col_groups > ctx->colsValid) {');
    expect(HOST).toContain('col_groups = ctx->colsValid - col_off;');
    expect(HOST).toContain('regions = col_groups * ctx->rowsValid;');
    const colsValid = 16384, width = 96;
    const batches = [];
    for (let off = 0; off < colsValid;) {
      const groups = off + width > colsValid ? colsValid - off : width;
      expect(groups % TALL_COL_OFFSETS).toBe(0);
      batches.push(groups);
      off += groups;
    }
    expect(batches.length).toBe(171);
    expect(batches[170]).toBe(64);
    expect(batches.reduce((a, b) => a + b, 0)).toBe(colsValid);
  });

  test('without its slice the flip goes back to the band walk', () => {
    expect(CONFIG).toContain('#define PEARL_AMPERE_FLIP_GUARD 1');
    expect(HOST).toContain('const uint32_t walkColBatch = ctx->colBatch, walkBand = ctx->tallBand;');
    expect(HOST).toContain('if ((uint64_t)ctx->l2Granted * 4u < (uint64_t)asked * 3u) {');
    const guard = HOST.slice(HOST.indexOf('#if PEARL_AMPERE_FLIP_GUARD'));
    expect(guard).toContain('ctx->foldPinB = false;');
    expect(guard).toContain('ctx->colBatch = walkColBatch;');
    expect(guard).toContain('ctx->tallBand = walkBand;');
    // The fold's band word is written after the guard, so it reads the depth that runs.
    expect(guard.indexOf('ctx->tallBand = walkBand;'))
      .toBeLessThan(guard.indexOf('cudaMemcpy(ctx->dHitCount + PEARL_BD_BAND_WORD'));
  });

  test('the flip kernel holds B, the other A, and only sm_86 builds its body', () => {
    const lf = KERNEL.replace(/\r\n/g, NL);
    expect(lf).toContain(['#define PEARL_TALL_FN pearl_tile_fold_tall', '#define PEARL_TALL_PIN_B 0',
      '#include "pearl_fold_tall.cuh"'].join(NL));
    expect(lf).toContain(['#define PEARL_TALL_FN pearl_tile_fold_tall_pinb', '#define PEARL_TALL_PIN_B 1',
      '#include "pearl_fold_tall.cuh"'].join(NL));
    expect(FOLD_TALL).toContain('&& (!PEARL_TALL_PIN_B || PEARL_TALL_PIN_B_ARCH(__CUDA_ARCH__))');
    expect(KERNEL.split('#include "pearl_fold_tall.cuh"').length - 1).toBe(2);
    expect(FOLD_TALL).toContain('#if PEARL_AMPERE_PERSIST_A && !PEARL_TALL_PIN_B && defined(__CUDA_ARCH__)');
    expect(FOLD_TALL).toContain('#if PEARL_TALL_PIN_B && defined(__CUDA_ARCH__) && PEARL_TALL_PIN_B_ARCH(__CUDA_ARCH__)');
    expect(FOLD_TALL).toContain('"l"(bP + (sg) * bKB + (int)((p) * bStep * SK) - (int)BMID), "l"(polB));');
    expect(CONFIG).toContain('#define PEARL_TALL_PIN_B_ARCH(a) ((a) == 860)');
    // The host runs it only where sm_86's build is loaded, and launches it by its own name.
    expect(HOST).toContain('if (ctx->foldPersistA && PEARL_TALL_PIN_B_ARCH(ctx->tallArch * 10) && !ctx->foldUnfused) {');
    // A forced band (a tuning build of the band walk) never flips.
    expect(body(HOST, 'uint32_t pearl_ampere_flip_for(')).toContain('#ifdef PEARL_AMPERE_BAND');
    expect(HOST).toContain('if (ctx->foldPinB) foldFn = reinterpret_cast<const void *>(pearl_tile_fold_tall_pinb);');
    expect(HOST).toContain('pearl_tile_fold_tall_pinb<<<blocks, threads, smem, st>>>(');
  });
});
