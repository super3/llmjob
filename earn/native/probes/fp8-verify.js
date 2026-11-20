// node fp8-verify.js <pearl_core_fp8.node> [device=0]
//
// The correctness gate of the FP8 core (earn/native/fp8), on a real RTX 50 or RTX PRO
// Blackwell card. Everything the card computes is compared, bit for bit, with the CPU
// reference (earn/src/shared/miner/fp8.js), which is itself checked against Pearl's
// pinned vectors and a real testnet2 block:
//
//   matmul    the FP32 cells of a 64 x 128 block, for random E4M3 codes of every kind
//             (zeros, subnormals, saturated values, both signs) and for a k that is not a
//             whole number of staged chunks
//   noise     A' and B' from committed planes: the rank-32 noise product, the mix, the
//             cast and the downscale rewrite, with unit scales and with random scales
//   tickets   every tile's fold and ticket, for the 4 x 64, 8 x 32 and 16 x 16 tiles
//   hits      the tiles the card reports under a bound are exactly the ones whose
//             ticket meets it
//   block     the proof of testnet2 block 150,688, mined on an RTX 5080 by Pearl's own
//             miner: from its opened rows the card must reach the same codes, the same
//             cells and the ticket dff32409...
//
// Prints one line a check and a final JSON line with "pass". Exit code 0 only if all pass.
'use strict';
const path = require('path');
const R = path.join(__dirname, '..', '..', 'src', 'shared', 'miner') + path.sep;
const fp8 = require(R + 'fp8');
const host = require(R + 'fp8Host');
const { keyedHash } = require(R + 'blake3');

const core = require(path.resolve(process.argv[2]));
const card = core.open(Number(process.argv[3] || 0));
const B200 = fp8.DEVICE_B200;
const results = [];
const MAX = Buffer.alloc(32, 0xFF);

function check(name, bad, detail) {
  results.push({ name, pass: bad === 0, bad, ...detail });
  console.log(`${bad === 0 ? 'PASS' : 'FAIL'} ${name}${bad ? ` (${bad} differ)` : ''}${detail ? ' ' + JSON.stringify(detail) : ''}`);
}

function diffBits(got, want) {
  let bad = 0;
  for (let i = 0; i < want.length; i++) if (fp8.f32Bits(got[i]) !== fp8.f32Bits(want[i])) bad += 1;
  return bad;
}

function diffBytes(got, want) {
  let bad = got.length === want.length ? 0 : 1;
  for (let i = 0; i < want.length; i++) if (got[i] !== want[i]) bad += 1;
  return bad;
}

// Random E4M3 codes. `mode` picks the mix: 0 any code, 1 mostly small, 2 mostly zero.
function randomCodes(next, count, mode) {
  const out = new Uint8Array(count);
  for (let i = 0; i < count; i++) {
    const x = next();
    let c = x & 0xFF;
    if (mode === 1) c = (c & 0x80) | ((x >>> 8) % 40);
    if (mode === 2 && (x >>> 16) % 4 !== 0) c = (x >>> 8) & 0x80;
    if ((c & 0x7F) === 0x7F) c ^= 1;
    out[i] = c;
  }
  return out;
}

// The cells of block (0, 0) on the card against the reference.
function matmulCheck(name, a, b, k, h = 16) {
  core.configure(64, 128, k, h, 64);
  core.setCodes(0, a);
  core.setCodes(1, b);
  const got = core.cells(0, 0);
  const want = fp8.matmulB200(a, b, null, 64, 128, k, false);
  check(name, diffBits(got, want), { k });
  return want;
}

// --- matmul ---
{
  const next = host.rng(1);
  for (const [mode, k] of [[0, 1024], [1, 1056], [2, 1024], [0, 2048]]) {
    matmulCheck(`matmul mode ${mode}`, randomCodes(next, 64 * k, mode), randomCodes(next, 128 * k, mode), k);
  }
}

// One operand through the card's noise path and through the reference.
// E lines hang off eSeed and F lines off fSeed (for A those are seedA and seedB).
function noiseSide(side, rows, k, ints, scales, eSeed, fSeed, sideId, lineOf = (i) => i) {
  const params = host.rowParams(B200, ints, scales, rows, k);
  const key = fp8.noiseLineKey(eSeed);
  const e = new Uint8Array(rows * 32);
  for (let i = 0; i < rows; i++) e.set(fp8.sampleLine(key, sideId, fp8.FACTOR_E, lineOf(i), 32), i * 32);
  const f = host.noiseLines(fSeed, sideId, fp8.FACTOR_F, 0, k);
  core.noise(side, ints, e, f, params.signal, params.beta, params.rewrite);
  const norms = fp8.exactNorms(ints, scales, rows, k);
  const want = fp8.noisyQuantize(B200, ints, scales, { e, f }, norms).noised;
  return { got: core.getCodes(side, 0, rows), want };
}

// --- noise ---
{
  const k = 1024;
  const next = host.rng(7);
  for (const variant of ['unit scales', 'random scales']) {
    core.configure(64, 128, k, 16, 64);
    const scalesFor = (rows) => {
      const s = host.unitScales(rows, k);
      // BF16 between 0.25 and 4, so rows keep an RMS the policy accepts.
      if (variant !== 'unit scales') for (let i = 0; i < s.length; i++) s[i] = 0x3E80 + (next() % 0x200);
      return s;
    };
    const a = noiseSide(0, 64, k, host.fillValues(11, 64, k), scalesFor(64), Buffer.alloc(32, 3), Buffer.alloc(32, 4), fp8.SIDE_A);
    const b = noiseSide(1, 128, k, host.fillValues(12, 128, k), scalesFor(128), Buffer.alloc(32, 4), Buffer.alloc(32, 4), fp8.SIDE_B);
    let rewritten = 0;
    for (const c of a.want) if ((c & 1) && ((c >> 3) & 15) < 4) rewritten += 1;
    check(`noise A, ${variant}`, diffBytes(a.got, a.want), { smallOddCodes: rewritten });
    check(`noise B, ${variant}`, diffBytes(b.got, b.want));
    // And the product of the two noised operands, which is what gets mined.
    check(`matmul of noised operands, ${variant}`,
      diffBits(core.cells(0, 0), fp8.matmulB200(a.want, b.want, null, 64, 128, k, false)), { k });
  }
}

// --- tickets and hits ---
{
  const k = 1024;
  const next = host.rng(21);
  const a = randomCodes(next, 64 * k, 1);
  const b = randomCodes(next, 128 * k, 1);
  const key = Buffer.alloc(32, 9);
  for (const h of [4, 8, 16]) {
    const w = 256 / h;
    const cells = matmulCheck(`matmul for the ${h} x ${w} tiles`, a, b, k, h);
    const all = core.search(key, MAX, true);
    const tilesA = 64 / h;
    const tilesB = 128 / w;
    let bad = 0;
    const tickets = [];
    for (let ta = 0; ta < tilesA; ta++) {
      for (let tb = 0; tb < tilesB; tb++) {
        const tile = new Float32Array(256);
        for (let i = 0; i < h; i++) for (let j = 0; j < w; j++) tile[i * w + j] = cells[(ta * h + i) * 128 + tb * w + j];
        const want = keyedHash(key, fp8.foldTile(tile));
        tickets.push(want);
        const at = (ta * tilesB + tb) * 32;
        if (!want.equals(Buffer.from(all.tickets.subarray(at, at + 32)))) bad += 1;
      }
    }
    check(`tickets, ${h} x ${w} tiles`, bad + (all.count === tickets.length ? 0 : 1), { tiles: tickets.length });

    // A bound about a quarter of the tiles meet: the card's hits must be exactly those.
    const bound = Buffer.alloc(32, 0xFF);
    bound[31] = 0x3F;
    const le = (t) => BigInt('0x' + Buffer.from(t).reverse().toString('hex'));
    const want = new Set();
    tickets.forEach((t, i) => { if (le(t) <= le(bound)) want.add(`${Math.floor(i / tilesB)},${i % tilesB}`); });
    const some = core.search(key, bound, false);
    const got = new Set();
    for (let i = 0; i < some.hits.length; i += 2) got.add(`${some.hits[i]},${some.hits[i + 1]}`);
    let wrong = some.count === want.size ? 0 : 1;
    for (const t of want) if (!got.has(t)) wrong += 1;
    for (const t of got) if (!want.has(t)) wrong += 1;
    check(`hits, ${h} x ${w} tiles`, wrong, { winning: want.size, of: tickets.length });
  }
}

// --- the RTX 5080's testnet2 block ---
{
  const v = require(path.join(__dirname, '..', '..', 'test', 'fixtures', 'fp8Rtx5080Testnet2.json'));
  const hex = (s) => Buffer.from(s, 'hex');
  const k = v.k;
  const job = { device: v.device, k, a: v.a, b: v.b };
  const proposedHeader = hex(v.proposedHeader);
  const ancestorHeader = hex(v.ancestorHeader);
  const kA = fp8.keyA(proposedHeader);
  const kB = fp8.keyB(ancestorHeader);
  const seeds = fp8.noiseSeeds({
    proposedHeader, ancestorHeader, job,
    hashA: fp8.operandDigest(hex(v.roots.aValues), hex(v.roots.aScales), kA),
    hashB: fp8.operandDigest(hex(v.roots.bValues), hex(v.roots.bScales), kB),
  });
  const u16 = (b64) => {
    const b = Buffer.from(b64, 'base64');
    return Uint16Array.from({ length: b.length / 2 }, (_, i) => b.readUInt16LE(i * 2));
  };
  const i8 = (b64) => new Int8Array(Buffer.from(b64, 'base64'));
  const h = v.aRows.length;
  const w = v.bRows.length;
  const want = fp8.evaluateTile({
    device: v.device, k, aRows: v.aRows, bRows: v.bRows, seedA: seeds.seedA, seedB: seeds.seedB,
    aInts: i8(v.aInts), aScales: u16(v.aScales), bInts: i8(v.bInts), bScales: u16(v.bScales),
  });
  // The opened rows sit at the top of a 64 x 128 block; the rest is filler the tile never
  // touches. The noise lines of the opened rows are addressed by their rows in the block.
  core.configure(64, 128, k, h, 64);
  const place = (rows, ints, scales) => {
    const allInts = host.fillValues(31, rows, k);
    const allScales = host.unitScales(rows, k);
    allInts.set(ints);
    allScales.set(scales);
    return [allInts, allScales];
  };
  const a = noiseSide(0, 64, k, ...place(64, i8(v.aInts), u16(v.aScales)), seeds.seedA, seeds.seedB, fp8.SIDE_A,
    (i) => (i < h ? v.aRows[i] : i));
  const b = noiseSide(1, 128, k, ...place(128, i8(v.bInts), u16(v.bScales)), seeds.seedB, seeds.seedB, fp8.SIDE_B,
    (i) => (i < w ? v.bRows[i] : i));
  check('block 150,688: A\' codes', diffBytes(a.got.subarray(0, h * k), want.a.noised));
  check('block 150,688: B\' codes', diffBytes(b.got.subarray(0, w * k), want.b.noised));
  const cells = core.cells(0, 0);
  const tile = new Float32Array(h * w);
  for (let i = 0; i < h; i++) for (let j = 0; j < w; j++) tile[i * w + j] = cells[i * 128 + j];
  check('block 150,688: the tile\'s FP32 cells', diffBits(tile, want.tile));
  const all = core.search(fp8.subkey(fp8.LABELS.jackpot, seeds.seedA), MAX, true);
  const ticket = Buffer.from(all.tickets.subarray(0, 32));
  check('block 150,688: the ticket', ticket.equals(want.ticket) ? 0 : 1, { ticket: ticket.toString('hex') });
  check('block 150,688: the ticket wins the block', fp8.meetsDifficulty(ticket, v.nbits, h, w, k) ? 0 : 1);
}

core.close();
const pass = results.every((r) => r.pass);
console.log(JSON.stringify({ pass, card: card.name, compute: `${card.major}.${card.minor}`, checks: results.length, failed: results.filter((r) => !r.pass).map((r) => r.name) }));
process.exit(pass ? 0 : 1);
