'use strict';

const fs = require('fs');
const path = require('path');

// Two A/B switches on the tall fold's cp.async ring, both off, for race mode (round 4's
// item 10): PEARL_TALL_COPY_RUN (copy groups from running source pointers, with the next
// tile's A fragment loaded before the group) and PEARL_TALL_RO_FENCE (chunk c's readout
// in chunk c + 1's first k-step, between pmevent fences). The native side cannot run
// here, so the source is checked for the switches and their guards, and JS walks of the
// running pointers and of the moved readout are checked against what the fold does today.

const SRC = path.join(__dirname, '..', 'native', 'src');
const read = (name) => fs.readFileSync(path.join(SRC, name), 'utf8').replace(/\r\n/g, '\n');
const CONFIG = read('pearl_config.h');
const KERNEL = read('pearl_kernel.cu');
const FOLD_TALL = read('pearl_fold_tall.cuh');

// The fold's staging constants (pearl_fold_tall.cuh, pearl_config.h).
const SK = 64, BM = 192, BN = 256, SPC = 2, CHUNKS = 16;
const ASLOTS = 3, BSLOTS = 4, A_STEP = 32, B_STEP = 16;
const BMID = (BSLOTS / 2) * B_STEP * SK;
const AMID = A_STEP * SK;
// An LDGSTS source offset: 12 bits, signed.
const OFF_MIN = -2048, OFF_MAX = 2047;

describe('tall fold copy-group and readout-fence switches', () => {
  test('both switches are off by default, with their guards', () => {
    expect(CONFIG).toContain('#ifndef PEARL_TALL_COPY_RUN\n#define PEARL_TALL_COPY_RUN 0\n#endif');
    expect(CONFIG).toContain('#ifndef PEARL_TALL_RO_FENCE\n#define PEARL_TALL_RO_FENCE 0\n#endif');
    expect(FOLD_TALL).toContain('#error "PEARL_TALL_COPY_RUN and PEARL_TALL_RO_FENCE: not with E4\'s builds"');
    expect(FOLD_TALL).toContain('#if PEARL_TALL_COPY_RUN && defined(PEARL_ABLATE_DRAM) && PEARL_FOLD_CHUNKS > PEARL_JACKPOT_BUCKETS');
    expect(FOLD_TALL).toContain('#if PEARL_TALL_RO_FENCE && (defined(PEARL_ABLATE_READOUT) || !PEARL_TALL_RED_ON');
    // Today's copies are still there for the default build, in the prologue and the body.
    expect(FOLD_TALL.split('for (uint32_t p = 0; p < ASLOTS; p++) PEARL_TALL_ISSUE_A(ib, sg, p)').length - 1).toBe(1);
    expect(FOLD_TALL.split('for (uint32_t p = 0; p < ASLOTS; p++) PEARL_TALL_ISSUE_A(ib, s, p)').length - 1).toBe(1);
    expect(FOLD_TALL).toContain('uint32_t sg0 = (last ? 0u : chunk + 1u) * SPC;');
    // The fence and the XOR group are volatile, so NVVM keeps their order.
    expect(KERNEL).toContain('__device__ __forceinline__ void pearl_pm_fence() { asm volatile("pmevent.mask 0;"); }');
    expect(KERNEL).toContain('__device__ __forceinline__ void pearl_xor16_into(uint32_t &x, const int32_t (*a)[4]) {\n  asm volatile(');
    expect((KERNEL.match(/lop3\.b32 [^"]*0x96;\\n\\t/g) || []).length).toBe(8);
  });

  test('the running sources are the macros walked below', () => {
    for (const line of [
      'constexpr uint32_t AMID = aStep * SK;',
      'PEARL_TALL_ISSUE_A_AT(ib, 0u, aRun - AMID)',
      'PEARL_TALL_ISSUE_A_AT(ib, 1u, aRun)',
      'PEARL_TALL_ISSUE_A_AT(ib, 2u, aRun2)',
      'PEARL_TALL_ISSUE_B_AT(ib, 0u, bRun - (int)BMID)',
      'PEARL_TALL_ISSUE_B_AT(ib, 1u, bRun + (int)(bStep * SK) - (int)BMID)',
      'PEARL_TALL_ISSUE_B_AT(ib, 2u, bRun + (int)(2u * bStep * SK) - (int)BMID)',
      'PEARL_TALL_ISSUE_B_AT(ib, 3u, bRun + (int)(3u * bStep * SK) - (int)BMID)',
      'aRun += aKB;',
      'aRun2 += aKB;',
      'asm volatile("" : "+l"(aRun), "+l"(aRun2));',
      'bRun += bKB;',
      'asm volatile("" : "+l"(bRun));',
      'const int8_t *aRun = PEARL_E4_ABASE + asrc0 + AMID, *bRun = Bprime + bsrc0 + BMID;',
      'const int8_t *aRun2 = aRun + AMID;',
      'aRun = PEARL_E4_ABASE + asrc0 + AMID;',
      'aRun2 = aRun + AMID;',
      'bRun = Bprime + bsrc0 + BMID;',
    ]) expect(FOLD_TALL).toContain(line);
    // The prologue steps after each group; the chunk loop at the group's tile, copies or not.
    expect(FOLD_TALL).toContain(['    PEARL_TALL_GROUP_A(ib)', '    PEARL_TALL_STEP_A',
      '    PEARL_TALL_GROUP_B(ib)', '    PEARL_TALL_STEP_B'].join('\n'));
    expect(FOLD_TALL).toContain('if (pt == PEARL_TALL_APT) {\n              PEARL_TALL_STEP_A\n            }');
    expect(FOLD_TALL).toContain('if (pt == PEARL_TALL_BPT) {\n              PEARL_TALL_STEP_B\n            }');
  });

  // Two tiles a block (the grid stride), then a block with one; per-tile slabs (aKB, bKB
  // a stage of the tile) and the k-blocked order (a k-block of m and n rows).
  test.each([
    ['per-tile order', BM * SK, BN * SK, 2],
    ['per-tile order, one tile', BM * SK, BN * SK, 1],
    ['k-blocked order', 131072 * SK, 262144 * SK, 2],
  ])('the running pointers copy what sg0 does, at every copy (%s)', (name, aKB, bKB, tiles) => {
    const aBase = (v) => 0x4000000 + v * 0x1230000 + 0x40;
    const bBase = (v) => 0x9000000 + v * 0x3450000 + 0x80;
    // Today: aP + sg * aKB + slot, bP + sg * bKB + slot - BMID, bP = the base + BMID.
    const today = [];
    const issueToday = (v, sg) => {
      for (let p = 0; p < ASLOTS; p++) today.push(['A', p, aBase(v) + sg * aKB + p * A_STEP * SK]);
      for (let p = 0; p < BSLOTS; p++) today.push(['B', p, bBase(v) + BMID + sg * bKB + p * B_STEP * SK - BMID]);
    };
    for (let s = 0; s < SPC; s++) issueToday(0, s);
    for (let v = 0; v < tiles; v++) {
      const hasNext = v + 1 < tiles;
      for (let chunk = 0; chunk < CHUNKS; chunk++) {
        const last = chunk + 1 === CHUNKS;
        if (last && !hasNext) continue;   // stage_next false: no copies
        const sg0 = (last ? 0 : chunk + 1) * SPC;
        for (let s = 0; s < SPC; s++) issueToday(last ? v + 1 : v, sg0 + s);
      }
    }
    // The switch: the macros above, register and offset kept apart.
    const run = [];
    let aRun, aRun2, bRun;
    const at = (op, p, reg, off) => {
      expect(off).toBeGreaterThanOrEqual(OFF_MIN);
      expect(off).toBeLessThanOrEqual(OFF_MAX);
      run.push([op, p, reg + off]);
    };
    const groupA = () => {
      at('A', 0, aRun, -AMID);
      at('A', 1, aRun, 0);
      at('A', 2, aRun2, 0);
    };
    const groupB = () => {
      for (let p = 0; p < BSLOTS; p++) at('B', p, bRun, p * B_STEP * SK - BMID);
    };
    const stepA = () => { aRun += aKB; aRun2 += aKB; };
    const stepB = () => { bRun += bKB; };
    aRun = aBase(0) + AMID;
    aRun2 = aRun + AMID;
    bRun = bBase(0) + BMID;
    for (let s = 0; s < SPC; s++) { groupA(); stepA(); groupB(); stepB(); }
    for (let v = 0; v < tiles; v++) {
      const hasNext = v + 1 < tiles;
      for (let chunk = 0; chunk < CHUNKS; chunk++) {
        const last = chunk + 1 === CHUNKS;
        const stageNext = !last || hasNext;
        if (last && hasNext) {
          aRun = aBase(v + 1) + AMID;
          aRun2 = aRun + AMID;
          bRun = bBase(v + 1) + BMID;
        }
        for (let s = 0; s < SPC; s++) {
          if (stageNext) groupA();
          stepA();
          if (stageNext) groupB();
          stepB();
        }
      }
    }
    // The prologue's stage pair, and one a chunk but the last tile's last.
    expect(run.length).toBe(tiles * CHUNKS * SPC * (ASLOTS + BSLOTS));
    expect(run).toEqual(today);
  });

  test('the moved readout is in the fold as modelled', () => {
    for (const line of [
      'const bool fenced = PEARL_TALL_RO_FENCE && s == 0u && t == 0u;',
      'pearl_xor16_into(xw, acc[mb]);',
      'pearl_xor16_into(xw, acc[mb] + 4);',
      'if (mb & 1u) {',
      'setp.ne.u32 p, %2, 0;\\n @p red.shared.xor.b32 [%0], %1;',
      '::"r"(trBase + (mb / 2u) * 128u + chunk * 4u - 4u), "r"(xw), "r"(chunk)',
      '#elif PEARL_TALL_RO_FENCE && !PEARL_TALL_TMA_BODY',
    ]) expect(FOLD_TALL).toContain(line);
    // Each group opens and closes with a fence, and its XORs come before the tile's mma.
    const g = FOLD_TALL.indexOf('if (fenced) {');
    const mma = FOLD_TALL.indexOf('pearl_mma_m16n8k32(acc[mb][nb][0]', g);
    const group = FOLD_TALL.slice(g, mma);
    expect(group.split('pearl_pm_fence();').length - 1).toBe(2);
    expect(group.indexOf('pearl_pm_fence();')).toBeLessThan(group.indexOf('pearl_xor16_into'));
    expect(group.lastIndexOf('pearl_pm_fence();')).toBeGreaterThan(group.indexOf('red.shared.xor'));
  });

  // A tile of 16 chunks, 2 stages of 2 k-steps, 6 m16 tiles: each tile's accumulators move
  // one step a k-step. A readout write records the steps its two tiles had, and when.
  function walk(moved) {
    const MT = 6, KS = 2, RPL = 3;
    const writes = [];
    const ver = new Array(MT).fill(0);
    let time = 0;
    for (let chunk = 0; chunk < CHUNKS; chunk++) {
      const last = chunk + 1 === CHUNKS;
      for (let s = 0; s < SPC; s++) {
        for (let t = 0; t < KS; t++) {
          let xw = [];
          for (let mb = 0; mb < MT; mb++) {
            if (moved && s === 0 && t === 0) {
              xw.push(ver[mb]);
              if (mb & 1) {
                if (chunk !== 0) writes.push({ rl: mb >> 1, word: chunk - 1, vers: xw, time });
                xw = [];
              }
            }
            ver[mb]++;
            time++;
          }
        }
      }
      if (!moved || last)
        for (let rl = 0; rl < RPL; rl++)
          writes.push({ rl, word: chunk, vers: [ver[2 * rl], ver[2 * rl + 1]], time });
    }
    return { writes, handoff: time };
  }

  test('each word is written once, with its chunk\'s sums, before the hand-off, never in chunk 0', () => {
    const today = walk(false), moved = walk(true);
    const key = (w) => `${w.rl}/${w.word}`;
    expect(moved.writes.length).toBe(3 * CHUNKS);
    expect(new Set(moved.writes.map(key)).size).toBe(3 * CHUNKS);
    const was = new Map(today.writes.map((w) => [key(w), w]));
    for (const w of moved.writes) {
      const old = was.get(key(w));
      // Chunk c's sums: 4 k-steps a chunk, so 4 (c + 1) steps on both tiles.
      expect(w.vers).toEqual(old.vers);
      expect(w.vers).toEqual([4 * (w.word + 1), 4 * (w.word + 1)]);
      // Only ever later than today (the chunk-0 hand-off guard), and before the hand-off.
      expect(w.time).toBeGreaterThanOrEqual(old.time);
      expect(w.time).toBeLessThanOrEqual(moved.handoff);
      // Nothing in chunk 0: its first k-step is the first 6 tile-steps of the tile.
      expect(w.time).toBeGreaterThanOrEqual(4 * 6);
    }
  });
});
