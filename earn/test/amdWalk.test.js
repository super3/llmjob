'use strict';

// PEARL_AMD_WALK (earn/native/amd/src/pearl_amd_fold.h): the order the AMD matrix folds take their tiles in. A map
// that folded one tile twice would leave another unfolded, and that tile's transcripts would be the last batch's,
// so every hit in it would be wrong. No HIP compiler runs here, so pearl_amd_walk is read out of the source and run
// as JS on BigInts, whose division truncates as C's unsigned division does: the same statements, for every grid the
// host can launch and every shape a target gets.

const fs = require('fs');
const path = require('path');
const { preprocess } = require('./helpers/amdSwitches');

const FOLD = fs.readFileSync(path.join(__dirname, '..', 'native', 'amd', 'src', 'pearl_amd_fold.h'), 'utf8')
  .replace(/\r\n/g, '\n');

function between(text, from, to) {
  const at = text.indexOf(from);
  if (at < 0) throw new Error('not found: ' + from);
  const end = text.indexOf(to, at + from.length);
  if (end < 0) throw new Error('not found after ' + from + ': ' + to);
  return text.slice(at + from.length, end);
}

const WALK = '__device__ __forceinline__ void pearl_amd_walk(';
const BODY = between(FOLD, WALK, '\n}\n');
const ON = between(BODY, '#if PEARL_AMD_WALK_ON\n', '#else\n');

// pearl_amd_walk as JS for walk `mode` (1 or 2) and shape { W, H, SW, SH }: (bx, by, gx, gy) -> [x, y], BigInts.
function walkFn(mode, { W, H, SW = W, SH = H }) {
  const js = ON
    .replace(/ {2}static_assert\([^;]*;\n/, '')
    .replace(/\bconstexpr uint32_t |\bconst uint32_t /g, 'const ')
    .replace(/\bPEARL_AMD_WALK_W\b/g, `${W}n`)
    .replace(/\bPEARL_AMD_WALK_H\b/g, `${H}n`)
    .replace(/\bPEARL_AMD_WALK_SW\b/g, `${SW}n`)
    .replace(/\bPEARL_AMD_WALK_SH\b/g, `${SH}n`)
    .replace(/\bPEARL_AMD_WALK\b/g, String(mode))
    .replace(/\b(\d+)u\b/g, '$1n')
    .replace(/return;/g, 'return [x, y];');
  if (/uint32_t|PEARL_|static_assert/.test(js)) throw new Error('left untranslated:\n' + js);
  const f = new Function('bx', 'by', 'gx', 'gy', `let x = bx, y = by;\n${js}\nreturn [x, y];`);
  return (bx, by, gx, gy) => f(BigInt(bx), BigInt(by), BigInt(gx), BigInt(gy)).map(Number);
}

// The per-target shapes, as the preprocessor picks them in each device pass.
const SHAPES = {};
for (const m of between(FOLD, '#if PEARL_AMD_WALK && !defined(PEARL_AMD_WALK_W)\n', '#endif\n#endif\n')
  .matchAll(/#(?:el)?if defined\(__(gfx\d+)__\)\n#define PEARL_AMD_WALK_W (\d+)\n#define PEARL_AMD_WALK_H (\d+)\n#define PEARL_AMD_WALK_SW (\d+)\n#define PEARL_AMD_WALK_SH (\d+)\n/g)) {
  SHAPES[m[1]] = { W: Number(m[2]), H: Number(m[3]), SW: Number(m[4]), SH: Number(m[5]) };
}

// The cards each shape is for: WGPs (3 blocks of the fold each), shader arrays, and the batch's column tiles at the
// card's default width (amd_col_batch: 128 column offsets on gfx1100 and gfx1101, 256 elsewhere; 16 a tile).
const CARDS = {
  gfx1201: { card: 'RX 9070 XT', wgps: 32, arrays: 8, cols: 16 },
  gfx1200: { card: 'RX 9060 XT', wgps: 16, arrays: 4, cols: 16 },
  gfx1100: { card: 'RX 7900 XTX', wgps: 48, arrays: 12, cols: 8 },
  gfx1101: { card: 'RX 7800 XT', wgps: 30, arrays: 6, cols: 8 },
};
const ROWS = 1024;   // mainnet m / 128

// Every block of a gx x gy grid, in launch order, and the tile each folds.
function tiles(f, gx, gy) {
  const out = [];
  for (let by = 0; by < gy; by++) for (let bx = 0; bx < gx; bx++) out.push(f(bx, by, gx, gy));
  return out;
}
function oneToOne(f, gx, gy) {
  const seen = new Set();
  for (const [x, y] of tiles(f, gx, gy)) {
    if (!(x >= 0 && x < gx && y >= 0 && y < gy)) return `tile (${x}, ${y}) is off the ${gx} x ${gy} grid`;
    seen.add(y * gx + x);
  }
  return seen.size === gx * gy ? null : `${gx * gy - seen.size} tiles of ${gx} x ${gy} folded twice`;
}
// The A' tiles (rows) and B' tiles (columns) a set of tiles reads.
const spread = (ts) => ({ b: new Set(ts.map(([x]) => x)).size, a: new Set(ts.map(([, y]) => y)).size });

describe('the supertile walk (PEARL_AMD_WALK)', () => {
  it('is on by default only on the RDNA 4 chips it was raced on, and -DPEARL_AMD_WALK=0 turns it off', () => {
    // The default as each device pass (and the host pass, which defines no __gfx...__) sees it.
    const block = between(FOLD, '#ifndef PEARL_AMD_WALK\n', '\n#endif\n#endif\n') + '\n#endif\n';
    const m = /^#if (.*)\n#define PEARL_AMD_WALK 1\n#else\n#define PEARL_AMD_WALK 0\n#endif\n$/.exec(block);
    expect(m).not.toBeNull();
    const cond = m[1];
    expect(cond.replace(/defined\(__gfx[0-9a-f]+__\)/g, 'D').replace(/ \|\| /g, '')).toMatch(/^D+$/);
    const defaultFor = (target) => (new Function('defined', `return ${cond.replace(/defined\((__\w+__)\)/g, 'defined("$1")')};`)(
      (n) => target !== null && n === `__${target}__`) ? 1 : 0);
    const targets = fs.readFileSync(path.join(__dirname, '..', 'native', 'amd', 'build-amd.sh'), 'utf8')
      .match(/^DEFAULT_ARCHS="(.*)"$/m)[1].split(' ');
    expect(targets).toHaveLength(14);
    const on = targets.filter((t) => defaultFor(t));
    expect(on).toEqual(['gfx1200', 'gfx1201']);
    expect(defaultFor(null)).toBe(0);
    // Each has a shape, so the default does remap there.
    for (const t of on) expect(SHAPES[t]).toBeDefined();
    // A -D given to the compiler wins: the default sits under #ifndef.
    expect(FOLD).toContain('#ifndef PEARL_AMD_WALK\n#if ' + cond + '\n');
  });

  it('stays out of the Windows core, which mines k 2048, and the Linux build keeps the default', () => {
    const BUILD = fs.readFileSync(path.join(__dirname, '..', 'native', 'amd', 'build-amd.sh'), 'utf8')
      .replace(/\r\n/g, '\n');
    expect(BUILD).toContain(
      'SET_OS=\nif windows; then SET_OS="-DPEARL_AMD_RDNA4_SET=0 -DPEARL_AMD_RDNA3_BUFLD=0 -DPEARL_AMD_RDNA3_SET=0"; fi\n');
    expect(BUILD).toContain('\nif windows; then SET_OS="$SET_OS -DPEARL_AMD_WALK=0"; fi\n');
    // Both winlib compiles take SET_OS before PEARL_AMD_CFLAGS, so a flag given there can still turn it on.
    expect(BUILD.match(/\$KWRAP_OS \$SET_OS \$FUSE_OS \$\{PEARL_AMD_CFLAGS:-\}/g)).toHaveLength(2);
    const gpu = BUILD.slice(BUILD.indexOf('\n  gpu)\n'), BUILD.indexOf('\n  winlib)\n'));
    expect(gpu).not.toContain('SET_OS');
    expect(gpu).not.toContain('PEARL_AMD_WALK');
  });

  it('a build with the walk off keeps the release order\'s lines', () => {
    expect(FOLD).toContain('#if PEARL_AMD_WALK != 0 && PEARL_AMD_WALK != 1 && PEARL_AMD_WALK != 2\n#error');
    expect(FOLD).toContain('#define PEARL_AMD_WALK_ON (PEARL_AMD_WALK != 0 && PEARL_AMD_WALK_W > 0)\n');
    // Without a shape (every target but the four below, and the host pass) W is 0, so it is off there.
    expect(FOLD).toContain('#ifndef PEARL_AMD_WALK_W\n#define PEARL_AMD_WALK_W 0\n#endif\n');
    expect(FOLD).toContain('#ifndef PEARL_AMD_WALK_SW\n#define PEARL_AMD_WALK_SW PEARL_AMD_WALK_W\n#endif\n');
    expect(FOLD).toContain('#ifndef PEARL_AMD_WALK_SH\n#define PEARL_AMD_WALK_SH PEARL_AMD_WALK_H\n#endif\n');
    const off = preprocess(FOLD, { PEARL_AMD_WALK_ON: 0 });
    expect(off).toContain('  const int wr = wave / P::kWC, wc = wave % P::kWC;\n'
      + '  const uint32_t tileRow = blockIdx.y * (uint32_t)BM;   // first A\' row of the tile\n'
      + '  const uint32_t tileCol = blockIdx.x * (uint32_t)BN;   // first column, within the batch\n');
    expect(off).not.toContain('pearl_amd_walk(blockIdx');
    expect(between(off, WALK, '\n}\n')).toContain('  x = bx;\n  y = by;\n  (void)gx;\n  (void)gy;');
  });

  it('changes only which tile a block folds', () => {
    const on = preprocess(FOLD, { PEARL_AMD_WALK_ON: 1 });
    expect(on).toContain('  pearl_amd_walk(blockIdx.x, blockIdx.y, gridDim.x, gridDim.y, tileX, tileY);\n'
      + '  const uint32_t tileRow = tileY * (uint32_t)BM;\n  const uint32_t tileCol = tileX * (uint32_t)BN;\n');
    expect(on.match(/pearl_amd_walk\(/g)).toHaveLength(2);   // its definition and the one call
    // The operands and the transcript stores follow the tile; a stamp record keeps the launch's own block number.
    expect(on).toContain('  const int8_t *gA = Ap + (size_t)tileRow * k;\n  const int8_t *gB = Bp + (size_t)(col0 + tileCol) * k;\n');
    expect(on).toContain('const uint32_t rowIdx = 2u * ((tileRow + (uint32_t)(wr * WM + rb * 32)) / 32u) + roIdx;');
    expect(on).toContain('const uint32_t colIdx = 4u * ((tileCol + (uint32_t)(wc * WN + cb * 64)) / 64u) + coIdx;');
    expect(on).toContain('const uint32_t block = blockIdx.y * gridDim.x + blockIdx.x;');
    // gridDim is read nowhere else in the matrix fold.
    const body = between(on, 'void pearl_amd_fold_body(', '\n}\n');
    expect(body.match(/gridDim/g)).toHaveLength(3);
  });

  it('gives the four RDNA 3 and RDNA 4 chips Salad rents a shape sized to the card, and no other target one', () => {
    expect(SHAPES).toEqual({
      gfx1201: { W: 8, H: 12, SW: 2, SH: 6 },
      gfx1200: { W: 4, H: 12, SW: 2, SH: 6 },
      gfx1100: { W: 8, H: 18, SW: 2, SH: 6 },
      gfx1101: { W: 8, H: 12, SW: 4, SH: 4 },
    });
    for (const [arch, { wgps, arrays, cols }] of Object.entries(CARDS)) {
      const { W, H, SW, SH } = SHAPES[arch];
      const held = 3 * wgps;
      // One supertile is about the blocks the card holds at once: the same, or under one more row of tiles.
      expect([arch, W * H >= held && W * H < held + W]).toEqual([arch, true]);
      expect([arch, cols % W, W % SW, H % SH]).toEqual([arch, 0, 0, 0]);
      // A sub-tile a shader array.
      expect([arch, (W / SW) * (H / SH)]).toEqual([arch, arrays]);
    }
  });

  it('folds every tile of every grid exactly once', () => {
    const shapes = [
      ...Object.values(SHAPES),
      { W: 1, H: 1 }, { W: 2, H: 3, SW: 1, SH: 3 }, { W: 3, H: 2, SW: 3, SH: 1 }, { W: 16, H: 1 },
      { W: 4, H: 5, SW: 2, SH: 5 }, { W: 8, H: 12, SW: 8, SH: 1 }, { W: 2, H: 64, SW: 1, SH: 4 },
    ];
    for (const shape of shapes) {
      for (const mode of [1, 2]) {
        const f = walkFn(mode, shape);
        const bad = [];
        // The host's grids are 1 to 16 column tiles (batches of 16 to 256 column offsets) by m / 128 row tiles.
        for (let gx = 1; gx <= 17; gx++) {
          for (let gy = 1; gy <= 40; gy++) {
            const why = oneToOne(f, gx, gy);
            if (why) bad.push(why);
          }
        }
        for (const gx of [8, 16]) {
          const why = oneToOne(f, gx, ROWS);
          if (why) bad.push(why);
        }
        expect([mode, shape, bad.slice(0, 3)]).toEqual([mode, shape, []]);
      }
    }
  });

  it('keeps the release order where a shape doesn\'t fit, and past the last whole supertile row', () => {
    const f = walkFn(2, SHAPES.gfx1201);
    // 8 doesn't divide 12 column tiles, or 4.
    for (const gx of [4, 12]) for (const [i, t] of tiles(f, gx, 30).entries()) expect(t).toEqual([i % gx, Math.floor(i / gx)]);
    // 1024 = 85 x 12 + 4: the last 4 rows.
    for (let by = 1020; by < ROWS; by++) for (let bx = 0; bx < 16; bx++) expect(f(bx, by, 16, ROWS)).toEqual([bx, by]);
    expect(f(0, 1019, 16, ROWS)).not.toEqual([0, 1019]);
  });

  it('is the release order on RDNA 3 in mode 1, where the batch is the supertile\'s 8 column tiles', () => {
    for (const arch of ['gfx1100', 'gfx1101']) {
      const f = walkFn(1, SHAPES[arch]);
      for (const [i, t] of tiles(f, 8, ROWS).entries()) {
        if (t[0] !== i % 8 || t[1] !== Math.floor(i / 8)) throw new Error(`${arch}: block ${i} folds (${t})`);
      }
      // Mode 2 isn't.
      expect(tiles(walkFn(2, SHAPES[arch]), 8, ROWS).slice(0, 8)).not.toEqual(tiles(f, 8, ROWS).slice(0, 8));
    }
  });

  it('puts the blocks a card holds at once on fewer bytes past the L2 than the release, on RDNA 4', () => {
    // Each group of as many blocks as the card holds, in launch order, reads A' tiles of 128 x k bytes and B' tiles
    // of 256 x k: what the L2 fetches for the group when the blocks run in step. Counted for the first supertile
    // row's groups.
    const release = (bx, by) => [bx, by];
    const got = {};
    for (const arch of ['gfx1201', 'gfx1200']) {
      const { wgps, cols } = CARDS[arch];
      const held = 3 * wgps;
      const all = { release: tiles(release, cols, ROWS) };
      for (const mode of [1, 2]) all['walk' + mode] = tiles(walkFn(mode, SHAPES[arch]), cols, ROWS);
      got[arch] = {};
      for (const [name, ts] of Object.entries(all)) {
        const groups = [0, 1].map((g) => spread(ts.slice(g * held, (g + 1) * held)));
        got[arch][name] = groups.map(({ a, b }) => `${b} B' + ${a} A' tiles, ${(a * 128 + b * 256) / 128} A' tiles' bytes`);
      }
    }
    expect(got).toEqual({
      gfx1201: {
        release: ["16 B' + 6 A' tiles, 38 A' tiles' bytes", "16 B' + 6 A' tiles, 38 A' tiles' bytes"],
        walk1: ["8 B' + 12 A' tiles, 28 A' tiles' bytes", "8 B' + 12 A' tiles, 28 A' tiles' bytes"],
        walk2: ["8 B' + 12 A' tiles, 28 A' tiles' bytes", "8 B' + 12 A' tiles, 28 A' tiles' bytes"],
      },
      gfx1200: {
        release: ["16 B' + 3 A' tiles, 35 A' tiles' bytes", "16 B' + 3 A' tiles, 35 A' tiles' bytes"],
        walk1: ["4 B' + 12 A' tiles, 20 A' tiles' bytes", "4 B' + 12 A' tiles, 20 A' tiles' bytes"],
        walk2: ["4 B' + 12 A' tiles, 20 A' tiles' bytes", "4 B' + 12 A' tiles, 20 A' tiles' bytes"],
      },
    });
    // A supertile row reads each A' tile in turn in its supertiles, then goes on: every A' tile is first read in
    // one supertile row only, so A' comes from memory once a batch, as in the release.
    const ts = tiles(walkFn(1, SHAPES.gfx1201), 16, ROWS);
    for (let i = 0; i < 85 * 12 * 16; i++) expect(Math.floor(ts[i][1] / 12)).toBe(Math.floor(i / (12 * 16)));
    // Every other supertile row runs backwards, so the turn starts on the B' tiles just read.
    expect(spread(ts.slice(96, 192)).b).toBe(8);
    expect(ts[192][0] >= 8 && ts[191][0] >= 8).toBe(true);
  });

  it('in mode 2 deals each supertile to its sub-tiles in turn, one compact sub-tile a shader array', () => {
    // If the dispatcher hands block i to shader array i % arrays (the guess mode 2 tests), these are an array's
    // tiles in a launch's first group.
    for (const [arch, { arrays, cols }] of Object.entries(CARDS)) {
      const { W, H, SW, SH } = SHAPES[arch];
      const ts = tiles(walkFn(2, SHAPES[arch]), cols, ROWS).slice(0, W * H);
      for (let s = 0; s < arrays; s++) {
        const mine = ts.filter((t, j) => j % arrays === s);
        expect([arch, s, spread(mine)]).toEqual([arch, s, { b: SW, a: SH }]);
      }
    }
  });
});
