'use strict';

const fs = require('fs');
const path = require('path');

// Turing's A-direct fold (PEARL_BD_A_DIRECT): the host writes A' fragment-ordered
// (pearl_materialize16_afrag) and each warp loads its A fragments with LDG.128
// instead of staging A through shared and reading it with ldmatrix. Nothing here
// can run the device code, so this checks, on the CPU and byte for byte, that the
// layout the host writes, read back the way the fold reads it, hands every lane
// the same four words of A as today's staged path (the staging stores, the
// swizzle and ldmatrix.x4), for every tile, warp, 16-row block, chunk and k-step
// of a small A'. Both sides are transcribed from the source, and the source lines
// they come from are pinned below, so a change to either one fails here.

const SRC = path.join(__dirname, '..', 'native', 'src');
const NL = String.fromCharCode(10);
// Line endings normalised: a Windows checkout has CRLF.
const read = (name) => fs.readFileSync(path.join(SRC, name), 'utf8')
  .split(String.fromCharCode(13) + NL).join(NL);
const KERNEL = read('pearl_kernel.cu');
const FOLD = read('pearl_fold_bd.cuh');
const HOST = read('pearl_host.cu');
const HEADER = read('pearl_config.h');

// A function's source, from its signature to the brace that closes it at the
// start of a line.
function body(src, signature) {
  const at = src.indexOf(signature);
  expect(at).toBeGreaterThanOrEqual(0);
  const end = src.indexOf(NL + '}', at);
  expect(end).toBeGreaterThan(at);
  return src.slice(at, end);
}

// The fold's compile-time geometry (pearl_config.h: PEARL_FOLD_K, _RANK).
const K = 2048;
const K_LOG2 = 11;
const RANK = 128;
const CHUNKS = K / RANK;
const KSTEPS = RANK / 32;
const MB = 4;
const ROWS = 128;
const FRAG_BLOCK = (K / 32) * 512;
const ABUF = ROWS * 128;

// Deterministic bytes, so a failure reproduces.
function rowMajorA(m) {
  const a = new Uint8Array(m * K);
  let x = 0x9e3779b9;
  for (let i = 0; i < a.length; i++) {
    x ^= x << 13; x >>>= 0;
    x ^= x >>> 17;
    x ^= x << 5; x >>>= 0;
    a[i] = x & 0xff;
  }
  return a;
}

function word(bytes, at) {
  return (bytes[at] | (bytes[at + 1] << 8) | (bytes[at + 2] << 16) | (bytes[at + 3] << 24)) >>> 0;
}

// pearl_materialize16_afrag's store, thread by thread: thread v holds the
// sixteen row-major bytes at v * 16 and stores word w at o[w * 4].
function materializeAfrag(a, rows) {
  const out = new Uint8Array(rows * K);
  const written = new Uint8Array(rows * K);
  for (let v = 0; v < (rows * K) / 16; v++) {
    const idx = v * 16;
    const r = Math.floor(idx / K);
    const kk = idx & (K - 1);
    const blk = (r >> 4) * 2 ** (K_LOG2 - 5) + (kk >> 5);
    const reg = ((kk >> 4) & 1) * 2 + ((r >> 3) & 1);
    const o = blk * 512 + (r & 7) * 64 + reg * 4;  // bytes: uint32 pointer + reg
    for (let w = 0; w < 4; w++) {
      for (let j = 0; j < 4; j++) {
        out[o + w * 16 + j] = a[idx + 4 * w + j];
        written[o + w * 16 + j]++;
      }
    }
  }
  return { out, written };
}

// The A-direct fold's read: lane's sixteen bytes of 16-row block mb of warp row
// slot wr, k-step s, tile row group rbg (a_off, then + s * 512 + mb * FRAG_BLOCK).
function adirectWords(frag, rbg, wr, lane, mb, s) {
  const aoff = (rbg * 8 + wr * 4) * FRAG_BLOCK + lane * 16;
  const at = aoff + s * 512 + mb * FRAG_BLOCK;
  return [word(frag, at), word(frag, at + 4), word(frag, at + 8), word(frag, at + 12)];
}

// Today's staged path for one tile and chunk: the staging threads' stores into
// the chunk's stage (groups 2: each row slot stages its own 64 rows; groups 1:
// the block stages all 128), then ldmatrix.x4 at each lane's address. sA is 128:
// the dynamic shared window starts after the fold's 128 bytes of static shared,
// and the fold relies on it being 128-aligned.
const SA = 128;
function stageChunk(a, rbg, c, groups) {
  const smem = new Uint8Array(SA + 2 * ABUF);
  const buf = (c & 1) * ABUF;
  for (let tid = 0; tid < 256; tid++) {
    const wr = tid >> 7;
    const sid = groups === 2 ? tid & 127 : tid;
    const srow0 = groups === 2 ? wr * 64 + (sid >> 3) : sid >> 3;
    const sstep = (groups === 2 ? 128 : 256) / 8;
    const sq = sid & 7;
    const adst0 = SA + srow0 * 128 + ((sq ^ (srow0 & 7)) << 4);
    // a_src(rbg) + chunk c's offset, slot p rows SSTEP apart.
    const src = (rbg * ROWS + srow0) * K + sq * 16 + c * RANK;
    for (let p = 0; p < 4; p++) {
      for (let b = 0; b < 16; b++) smem[adst0 + buf + p * sstep * 128 + b] = a[src + p * sstep * K + b];
    }
  }
  return smem;
}

function ldmatrixWords(smem, wr, lane, mb, c, t) {
  const buf = (c & 1) * ABUF;
  const laneAddr = (l) => {
    const alrow = (l & 7) + ((l >> 3) & 1) * 8;
    const albyte = ((l >> 4) & 1) * 16;
    const swz = (l & 7) << 4;
    const aLane0 = SA + (wr * 64 + alrow) * 128 + ((albyte ^ swz) & 0x10) + (swz & 0x60);
    return ((aLane0 + buf) ^ (t * 32)) + mb * 16 * 128;
  };
  // Matrix i takes its eight row addresses from lanes 8i..8i+7; lane L gets
  // bytes 4 (L % 4)..+3 of row L / 4 of each matrix, in register i.
  const out = [];
  for (let i = 0; i < 4; i++) out.push(word(smem, laneAddr(8 * i + (lane >> 2)) + 4 * (lane & 3)));
  return out;
}

// What the m16n8k32 A operand is, from the row-major bytes: a0 rows 0-7 at k
// 0-15, a1 rows 8-15 at k 0-15, a2 rows 0-7 at k 16-31, a3 rows 8-15 at k 16-31;
// lane L holds row L / 4, k bytes 4 (L % 4)..+3 of each.
function specWords(a, rbg, wr, lane, mb, s) {
  const row0 = rbg * ROWS + wr * 64 + mb * 16 + (lane >> 2);
  const k0 = s * 32 + 4 * (lane & 3);
  return [word(a, row0 * K + k0), word(a, (row0 + 8) * K + k0), word(a, row0 * K + k0 + 16),
    word(a, (row0 + 8) * K + k0 + 16)];
}

describe('Turing A-direct: A fragment-ordered', () => {
  test('is off by default, so the shipped fold and draw are unchanged', () => {
    const at = HEADER.indexOf('#ifndef PEARL_BD_A_DIRECT');
    expect(at).toBeGreaterThanOrEqual(0);
    expect(HEADER.slice(at).split(NL)[1].trim()).toBe('#define PEARL_BD_A_DIRECT 0');
    // The draw takes the fragment order for A only under the switch, the same
    // macro the fold is built with.
    expect(HOST).toContain('#if PEARL_TURING_BDIRECT && PEARL_BD_A_DIRECT' + NL
      + '  if (ctx->foldBDirect && isA && kPow2 && rows % 16u == 0u && k >= 32u) {');
  });

  // A fold built with the switch and a host built without it would read a
  // row-major A' as fragments and find nothing, with no error. The host tags
  // the band word it already passes, and the A-direct fold traps without the
  // tag. Band depths are small, so the tag bit never collides with one.
  test('a fold and host built differently fail loudly', () => {
    expect(HEADER).toContain('#define PEARL_BD_A_DIRECT_TAG 0x80000000u');
    expect(HOST).toContain('const uint32_t tag = PEARL_BD_A_DIRECT ? PEARL_BD_A_DIRECT_TAG : 0u;');
    expect(HOST).toContain('const uint32_t bands[Ctx::kSlots] = {ctx->bdBand | tag, ctx->bdBand | tag};');
    expect(FOLD).toContain('if (!(band_word & PEARL_BD_A_DIRECT_TAG)) __trap();');
    expect(FOLD).toContain('const uint32_t band_depth = band_word & ~PEARL_BD_A_DIRECT_TAG;');
    expect(HOST).toContain('if (v >= 1u && v <= 64u) s->band = v;');
  });

  // The transcriptions above, pinned to the source they copy.
  test('the transcriptions match the source', () => {
    const mat = body(KERNEL, 'void pearl_materialize16_afrag(');
    expect(mat).toContain('const uint64_t blk = ((uint64_t)(r >> 4) << (k_log2 - 5)) + (kk >> 5);');
    expect(mat).toContain('const uint32_t reg = ((kk >> 4) & 1u) * 2u + ((r >> 3) & 1u);');
    expect(mat).toContain(
      'uint32_t *o = reinterpret_cast<uint32_t *>(out + blk * 512u + (r & 7u) * 64u) + reg;');
    expect(mat).toContain('o[w * 4] = packed;');
    // B's kernel keeps its own word order (rows before k).
    expect(body(KERNEL, 'void pearl_materialize16_frag('))
      .toContain('const uint32_t reg = ((r >> 3) & 1u) * 2u + ((kk >> 4) & 1u);');

    expect(FOLD).toContain('constexpr uint32_t FRAG_BLOCK = (k / 32u) * 512u;');
    expect(FOLD).toContain('return (rbg_ * 8u + wr * 4u) * FRAG_BLOCK + lane * 16u;');
    expect(FOLD).toContain('af[0][mb] = pearl_bd_ldg(Aprime + aoff + mb * FRAG_BLOCK);');
    expect(FOLD).toContain('const uint32_t ao = last ? aoffn : aoff + (s + 1u) * 512u;');
    expect(FOLD).toContain('af[(t + 1u) & 1u][mb] = pearl_bd_ldg(Aprime + ao + mb * FRAG_BLOCK);');
    // a0-a3 are the loaded words x, y, z, w, in the slots the staged path gives
    // ldmatrix's r0-r3.
    expect(FOLD).toContain('acc[mb][2u * j][3], (uint32_t)a.x, (uint32_t)a.y, (uint32_t)a.z,'
      + NL + '                           (uint32_t)a.w, b0, b1);');
    expect(FOLD).toContain(
      'acc[mb][2u * j][3], af[mb][0], af[mb][1], af[mb][2], af[mb][3], b0, b1);');
    expect(FOLD).toContain('pearl_ldmatrix_x4(af[mb][0], af[mb][1], af[mb][2], af[mb][3], ap);');

    // The staged path.
    expect(FOLD).toContain('const uint32_t srow0 = wr * 64u + sid / 8u;');
    expect(FOLD).toContain('uint32_t adst0 = sA + srow0 * 128u + ((sq ^ (srow0 & 7u)) << 4);');
    expect(FOLD).toContain('return Aprime + (size_t)(rbg_ * ROWS + srow0) * k + sq * 16u;');
    expect(FOLD).toContain('ra[p] = pearl_bd_lda(an + p * SSTEP * k);');
    expect(FOLD).toContain('pearl_st_shared_v4((adst0 + (ABUF - buf)) + p * SSTEP * 128u, ra[p]);');
    expect(FOLD).toContain('const uint32_t alrow = (lane & 7u) + ((lane >> 3) & 1u) * 8u;');
    expect(FOLD).toContain('const uint32_t albyte = ((lane >> 4) & 1u) * 16u;');
    expect(FOLD).toContain('const uint32_t swz = (lane & 7u) << 4;');
    expect(FOLD).toContain(
      'uint32_t aLane0 = sA + (wr * 64u + alrow) * 128u + ((albyte ^ swz) & 0x10u) + (swz & 0x60u);');
    expect(FOLD).toContain('const uint32_t ap = ((aLane0 + buf) ^ (t * 32u)) + mb * 16u * 128u;');
    expect(FOLD).toContain('const uint32_t buf = (ch & 1u) * ABUF;');
    expect(FOLD).toContain('extern __shared__ __align__(128) uint32_t smem_u32[];');
  });

  test('the host writes every byte of A\' exactly once', () => {
    const m = 256;
    const { written } = materializeAfrag(rowMajorA(m), m);
    expect(written.every((n) => n === 1)).toBe(true);
  });

  test('each lane loads the same four words as ldmatrix hands it from the staged rows', () => {
    // Three row groups, so tiles away from row 0 are covered too.
    const m = 3 * ROWS;
    const a = rowMajorA(m);
    const { out: frag } = materializeAfrag(a, m);
    const same = (x, y) => x.length === 4 && x.every((w, i) => w === y[i]);
    const bad = [];
    let checked = 0;
    for (let rbg = 0; rbg < m / ROWS; rbg++) {
      for (let c = 0; c < CHUNKS; c++) {
        const staged = [stageChunk(a, rbg, c, 2), stageChunk(a, rbg, c, 1)];
        for (let wr = 0; wr < 2; wr++) {
          for (let mb = 0; mb < MB; mb++) {
            for (let t = 0; t < KSTEPS; t++) {
              const s = c * KSTEPS + t;
              for (let lane = 0; lane < 32; lane++) {
                const want = specWords(a, rbg, wr, lane, mb, s);
                const got = [adirectWords(frag, rbg, wr, lane, mb, s),
                  ldmatrixWords(staged[0], wr, lane, mb, c, t),
                  ldmatrixWords(staged[1], wr, lane, mb, c, t)];
                if (!got.every((g) => same(g, want)) && bad.length < 5) {
                  bad.push(`rbg ${rbg} chunk ${c} k-step ${t} wr ${wr} mb ${mb} lane ${lane}: `
                    + `want ${want}, A-direct ${got[0]}, ldmatrix ${got[1]} / ${got[2]}`);
                }
                checked++;
              }
            }
          }
        }
      }
    }
    expect(bad).toEqual([]);
    expect(checked).toBe(3 * CHUNKS * 2 * MB * KSTEPS * 32);
  });
});

// The fold's tile walk, transcribed (pearl_fold_bd.cuh, tile_coords).
function tileCoords(v, walk, bandDepth, rowBlockGroups, colBlockGroups) {
  if (walk === 1) {
    const bandTiles = bandDepth * rowBlockGroups;
    const band = Math.floor(v / bandTiles);
    const inBand = v % bandTiles;
    const bandFirst = band * bandDepth;
    const bandCols = Math.min(colBlockGroups - bandFirst, bandDepth);
    return { rbg: Math.floor(inBand / bandCols), cbg: bandFirst + (inBand % bandCols) };
  }
  const bandBlocks = bandDepth * colBlockGroups;
  const band = Math.floor(v / bandBlocks);
  const inBand = v % bandBlocks;
  const bandFirst = band * bandDepth;
  const bandRows = Math.min(rowBlockGroups - bandFirst, bandDepth);
  return { rbg: bandFirst + (inBand % bandRows), cbg: Math.floor(inBand / bandRows) };
}

describe('Turing A-direct: every load stays inside A\' and B\'', () => {
  test('the walk is the fold\'s', () => {
    expect(FOLD).toContain('cbg_ = band_first + in_band % band_cols;');
    expect(FOLD).toContain('rbg_ = in_band / band_cols;');
    expect(FOLD).toContain('const uint32_t row_block_groups = row_blocks / 2u;');
    expect(FOLD).toContain('const uint32_t col_block_groups = tiles / row_block_groups;');
    expect(FOLD).toContain('return (col_off + cbg_ * 16u + wc * 4u) * FRAG_BLOCK + lane * 16u;');
    expect(FOLD).toContain('const uint32_t bo = last ? boffn : boff + (s + 1u) * 512u;');
    // The last k-step's loads: the next tile's, or this tile's again.
    expect(FOLD).toContain('tile_coords(vnext < tiles ? vnext : v, rbgn, cbgn);');
    expect(FOLD).toContain('aoffn = aoff;');
    // The host refuses what the draw does not write fragment-ordered, and
    // offsets that do not fit 32 bits.
    expect(HOST).toContain('&& (ctx->profile.m % 16u != 0u || (uint64_t)ctx->profile.m * k > 0xFFFFFFFFull');
    expect(HOST).toContain('|| (uint64_t)ctx->profile.n * k > 0xFFFFFFFFull)) {');
  });

  // Every tile of a launch, every k-step's loads including the last one's (the
  // next tile's k-step 0), for A' of m rows and B' of n columns: the highest
  // byte read is below m * k and n * k, and every offset fits 32 bits.
  function check(m, n, colBatch, colOff, walk, bandDepth) {
    const rowsValid = m / 16;
    const rowBlocks = Math.floor(rowsValid / 4);
    const rowBlockGroups = Math.floor(rowBlocks / 2);
    const colBlockGroups = colBatch / 16;
    const tiles = rowBlockGroups * colBlockGroups;
    let aMax = 0;
    let bMax = 0;
    let outside = 0;
    for (let v = 0; v < tiles; v++) {
      const { rbg, cbg } = tileCoords(v, walk, bandDepth, rowBlockGroups, colBlockGroups);
      if (rbg >= rowBlockGroups || cbg >= colBlockGroups) outside++;
      // A warp's highest load: wr 1, mb 3, lane 31, k-step 63, sixteen bytes.
      const aTop = (rbg * 8 + 4) * FRAG_BLOCK + 31 * 16 + (CHUNKS * KSTEPS - 1) * 512 + 3 * FRAG_BLOCK + 16;
      const bTop = (colOff + cbg * 16 + 12) * FRAG_BLOCK + 31 * 16 + (CHUNKS * KSTEPS - 1) * 512
        + 3 * FRAG_BLOCK + 16;
      aMax = Math.max(aMax, aTop);
      bMax = Math.max(bMax, bTop);
    }
    expect(outside).toBe(0);
    expect(aMax).toBeLessThanOrEqual(m * K);
    expect(bMax).toBeLessThanOrEqual(n * K);
    expect(aMax).toBeLessThanOrEqual(2 ** 32);
    expect(bMax).toBeLessThanOrEqual(2 ** 32);
    return { aMax, bMax };
  }

  test('at the mainnet profile, and at the edge of smaller ones', () => {
    // m 131072, n 262144, col_batch 2048: the last batch of column offsets.
    const colsValid = 262144 / 16;
    for (const walk of [0, 1]) {
      for (const band of [2, 4, 5, 8, 16]) {
        check(131072, 262144, 2048, colsValid - 2048, walk, band);
        check(131072, 262144, 2048, 0, walk, band);
      }
    }
    // m with rows past the last whole row group: they are never read.
    const { aMax } = check(400, 512, 32, 0, 1, 5);
    expect(aMax).toBeLessThanOrEqual(384 * K);
    check(272, 512, 32, 0, 1, 2);
  });
});

describe('Turing A-direct: shared memory', () => {
  // The host's formula (pearl_host.cu) with A-direct's names: no A stages, one
  // transcript buffer, or two without the realign. Both fit the 32 KB of the
  // smaller split it asks for.
  test('8 KB, or 16 with two transcript buffers, inside Turing\'s 32 KB split', () => {
    expect(HOST).toContain('? (size_t)PEARL_BD_SMEM_A_STAGES * warpRows * regionsPerWarp * PEARL_ROWS_COUNT');
    expect(HOST).toContain('+ (size_t)PEARL_BD_T_BUFS * warpsPerBlock * regionsPerWarp * PEARL_WMMA_COL_BLK');
    expect(HEADER).toContain('#define PEARL_BD_SMEM_A_STAGES (PEARL_BD_A_DIRECT ? 0 : PEARL_BD_A_STAGES)');
    expect(HEADER).toContain('#define PEARL_BD_T_BUFS (PEARL_BD_A_DIRECT && !PEARL_BD_A_REALIGN ? 2 : 1)');
    expect(HEADER).toContain('#define PEARL_BD_A_DIRECT_CARVEOUT 50');
    const transcripts = 8 * 4 * 4 * 16 * 4;  // warps x regions x column blocks x buckets x 4 bytes
    for (const bufs of [1, 2]) expect(bufs * transcripts).toBeLessThanOrEqual(65536 * 50 / 100);
    expect(FOLD).toContain('PEARL_BD_T_BUFS * TWORDS * 4u == (PEARL_BD_A_REALIGN ? 8192u : 16384u)');
    expect(HOST).toContain('cudaFuncAttributePreferredSharedMemoryCarveout');
  });
});
