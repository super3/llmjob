'use strict';

const fp8 = require('./fp8');
const merkle = require('./merkle');
const { u64le, serializeMatrixMerkleProof, BINCODE_OPTION_NONE } = require('./plainProof');

// The host side of Pearl's PIP-3 (FP8) proof of work: everything a miner does
// on the CPU around the GPU core (earn/native/fp8).
//
//   - operand data that passes the jackpot policy (a constant fill does not),
//   - the two committed planes of each operand and their Merkle trees,
//   - the per-row numbers the core needs to noise an operand,
//   - one attempt's seeds, noise lines and winning bound,
//   - the PlainProofV4 a winning tile is submitted as.
//
// Nothing in the app requires this file yet; the FP8 tools do, behind
// PEARL_FP8=1 (fp8.enabled). The arithmetic is all in ./fp8, the bit-exact
// reference. See earn/tuning/pip3-fp8.md.

const HASH_ID_CHUNK_1024 = 3;  // HashId::Blake3Chunk1024, the tree ./merkle builds
const UNIT_SCALE = 0x3F80;     // BF16 1.0
const HEADER_LEN = 76;         // the proposed header, without its proof commitment
const ANCESTOR_LEN = 108;      // a complete header
const SCALE_ROW_BYTES = (k) => 2 * (k / fp8.BLOCK_SIZE);

// ---------------------------------------------------------------------------
// Operand data
// ---------------------------------------------------------------------------

// xorshift32. Any stream works; this one is cheap and has no short cycle.
function rng(seed) {
  let x = (seed >>> 0) || 0x9E3779B9;
  return () => {
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    return x >>> 0;
  };
}

// One centred triangular int8, in -127..127: the spread Pearl's own CPU miner
// uses (zk-pow v4 mine.rs) because it passes the noise-coverage check.
function triangular(x) {
  return (x & 127) + ((x >>> 16) & 127) - 127;
}

// rows x k values, row-major.
function fillValues(seed, rows, k) {
  const next = rng(seed);
  const out = new Int8Array(rows * k);
  for (let i = 0; i < out.length; i++) out[i] = triangular(next());
  return out;
}

// The scales plane for unit scales: one BF16 1.0 per 8 values.
function unitScales(rows, k) {
  return new Uint16Array(rows * (k / fp8.BLOCK_SIZE)).fill(UNIT_SCALE);
}

// A plane as the bytes that are committed: int8 as is, BF16 little endian.
function planeBytes(plane) {
  if (plane instanceof Int8Array) return Buffer.from(plane.buffer, plane.byteOffset, plane.byteLength);
  const out = Buffer.alloc(plane.length * 2);
  plane.forEach((v, i) => out.writeUInt16LE(v, i * 2));
  return out;
}

// ---------------------------------------------------------------------------
// Commitment: two keyed Merkle trees an operand, and their combined digest
// ---------------------------------------------------------------------------

// The planes here are a power-of-two number of 1024-byte chunks, at least two,
// so the tree has no padded or promoted node.
function checkPlaneSize(bytes) {
  const chunks = bytes.length / merkle.CHUNK_LEN;
  if (!Number.isInteger(chunks) || chunks < 2 || (chunks & (chunks - 1)) !== 0) {
    throw new Error(`a plane must be a power-of-two number of 1024-byte chunks, at least 2 || bytes=${bytes.length}`);
  }
}

// `tree(key, bytes)` returns ./merkle layers; the native core has a faster one.
function commitOperand(key, ints, scales, tree = merkle.buildLayers) {
  const values = planeBytes(ints);
  const scaleBytes = planeBytes(scales);
  checkPlaneSize(values);
  checkPlaneSize(scaleBytes);
  const valuesLayers = tree(key, values);
  const scalesLayers = tree(key, scaleBytes);
  return { key, values, scaleBytes, valuesLayers, scalesLayers, digest: digestOf(key, valuesLayers, scalesLayers) };
}

// The FP8 core's tree (every node, leaves first, the root last, 32 bytes
// each) as ./merkle layers. Pass coreTree(core) to commitOperand.
function layersFromNodes(nodes) {
  const layers = [];
  let at = 0;
  for (let count = (nodes.length / merkle.OUT_LEN + 1) / 2; count >= 1; count /= 2) {
    const layer = new Array(count);
    for (let i = 0; i < count; i++, at += merkle.OUT_LEN) layer[i] = Buffer.from(nodes.subarray(at, at + merkle.OUT_LEN));
    layers.push(layer);
  }
  return layers;
}

function coreTree(core) {
  return (key, bytes) => layersFromNodes(core.tree(key, bytes));
}

function digestOf(key, valuesLayers, scalesLayers) {
  return fp8.operandDigest(merkle.rootOf(valuesLayers), merkle.rootOf(scalesLayers), key);
}

// Recompute the tree after `bytes` changed inside one chunk: that leaf and its
// path to the root. This is what makes a new attempt cheap.
function updateLeaf(key, layers, bytes, leaf) {
  layers[0][leaf] = merkle.chunkCV(key, bytes.subarray(leaf * merkle.CHUNK_LEN, (leaf + 1) * merkle.CHUNK_LEN), leaf);
  let i = leaf;
  for (let level = 1; level < layers.length; level++) {
    i >>= 1;
    const below = layers[level - 1];
    layers[level][i] = merkle.parentCV(key, below[2 * i], below[2 * i + 1], level === layers.length - 1);
  }
}

// ---------------------------------------------------------------------------
// The job tuple
// ---------------------------------------------------------------------------

// A dense job: A is m x k with tiles h rows tall, B is n x k with tiles w rows
// wide. The shape of fp8.encodePB / encodePA.
function makeJob({ device = fp8.DEVICE_B200, m, n, k, h, w }) {
  fp8.checkLotteryLayout(h, w);
  if (k < fp8.MIN_K || k > fp8.MAX_K || k % 32 !== 0) throw new Error(`k must be a multiple of 32 in 1024..=65536 || k=${k}`);
  if (m % h !== 0 || n % w !== 0) throw new Error(`m and n must be whole tiles || m=${m} n=${n} h=${h} w=${w}`);
  return {
    device,
    k,
    a: { numRows: m, hashId: HASH_ID_CHUNK_1024, tileSize: h },
    b: { numRows: n, hashId: HASH_ID_CHUNK_1024, tileSize: w },
  };
}

// JobParams::to_wire_bytes: ancestor header (108) || pB || pA.
function jobWire(ancestorHeader, job) {
  if (ancestorHeader.length !== ANCESTOR_LEN) throw new Error(`the ancestor header must be ${ANCESTOR_LEN} bytes`);
  return Buffer.concat([ancestorHeader, fp8.encodePB(job), fp8.encodePA(job)]);
}

// ---------------------------------------------------------------------------
// What the core needs to noise an operand
// ---------------------------------------------------------------------------

// Per row: beta as FP32 and the downscale rewrite's two constants (the
// threshold code, and the safe exponent already shifted into place). Per
// 8-value block: `signal`, fl(alpha * scale), which the card multiplies by the
// int8 to get the clean term.
// Rows the policy would refuse (alpha * l2 outside 4..256) throw here, before
// any GPU time is spent on them.
function rowParams(device, ints, scales, rows, k) {
  const { g, gSmall } = fp8.deviceParams(device);
  const norms = fp8.exactNorms(ints, scales, rows, k);
  const blocks = k / fp8.BLOCK_SIZE;
  const signal = new Float32Array(rows * blocks);
  const beta = new Float32Array(rows);
  const rewrite = new Uint32Array(rows);
  norms.forEach(([l2Raw, linfRaw], i) => {
    const l2 = fp8.bf16Max(l2Raw, fp8.NORM_FLOOR_BF16);
    const [a, b] = fp8.deriveRowScales(device, l2, fp8.bf16Max(linfRaw, fp8.NORM_FLOOR_BF16), fp8.RANK);
    const alpha = fp8.bf16ToF32(a);
    beta[i] = fp8.bf16ToF32(b);
    for (let j = 0; j < blocks; j++) signal[i * blocks + j] = alpha * fp8.bf16ToF32(scales[i * blocks + j]);
    const rms = Math.fround(alpha * fp8.bf16ToF32(l2));
    if (!(rms >= 4 && rms < 256)) throw new Error(`row ${i} has alpha*l2=${rms}, outside the policy's 4..256`);
    const safeExp = ((fp8.f32Bits(rms) >>> 23) & 0xFF) - 127 - 9 + 7;
    const threshold = fp8.f32ToE4m3(Math.fround((rms >= 8 ? g : gSmall) * rms));
    rewrite[i] = threshold | (safeExp << 11);
  });
  return { signal, beta, rewrite };
}

// ---------------------------------------------------------------------------
// One attempt
// ---------------------------------------------------------------------------

// ticket <= target * h * w * k as 32 little-endian bytes, the bound the core
// compares tickets with.
function ticketBound(nbits, h, w, k) {
  const max = (1n << 256n) - 1n;
  let bound = fp8.nbitsToTarget(nbits) * BigInt(h * w * k);
  if (bound > max) bound = max;
  return Buffer.from(bound.toString(16).padStart(64, '0'), 'hex').reverse();
}

// The seeds of one (A, B, header) triple, and what follows from them alone.
function attemptSeeds({ proposedHeader, ancestorHeader, job, hashA, hashB }) {
  if (proposedHeader.length !== HEADER_LEN) throw new Error(`the proposed header must be ${HEADER_LEN} bytes`);
  const seeds = fp8.noiseSeeds({ proposedHeader, ancestorHeader, job, hashA, hashB });
  return {
    ...seeds,
    jackpotKey: fp8.subkey(fp8.LABELS.jackpot, seeds.seedA),
    bound: ticketBound(proposedHeader.readUInt32LE(72), job.a.tileSize, job.b.tileSize, job.k),
  };
}

// Noise lines `first .. first + count` of one side and factor, count x 32
// E4M3 codes. A's E lines hang off seedA; everything else off seedB.
function noiseLines(seed, side, factor, first, count) {
  const key = fp8.noiseLineKey(seed);
  const out = new Uint8Array(count * fp8.RANK);
  for (let i = 0; i < count; i++) out.set(fp8.sampleLine(key, side, factor, first + i, fp8.RANK), i * fp8.RANK);
  return out;
}

// ---------------------------------------------------------------------------
// PlainProofV4
// ---------------------------------------------------------------------------

// The opening of `rows` (consecutive, ascending) of one plane: the 1024-byte
// chunks that hold them and the siblings that tie those to the root. This is
// the unique minimal set the verifier insists on.
function openRows(key, bytes, layers, rows, rowBytes) {
  const leaves = merkle.leafIndicesFromRows(rows, rowBytes);
  return { proof: merkle.multiLeafProof(key, bytes, layers, leaves), rowIndices: rows };
}

function tileRows(tile, size) {
  return Array.from({ length: size }, (_, i) => tile * size + i);
}

// The proof of tile (tileA, tileB) of a job whose ancestor is the parent, so
// the chain between them is empty. `a` and `b` are commitOperand results.
function buildProof({ ancestorHeader, job, a, b, tileA, tileB }) {
  const aRows = tileRows(tileA, job.a.tileSize);
  const bRows = tileRows(tileB, job.b.tileSize);
  const scaleRow = SCALE_ROW_BYTES(job.k);
  return {
    job: jobWire(ancestorHeader, job),
    chain: Buffer.alloc(0),
    values: {
      a: openRows(a.key, a.values, a.valuesLayers, aRows, job.k),
      b: openRows(b.key, b.values, b.valuesLayers, bRows, job.k),
    },
    scales: {
      a: openRows(a.key, a.scaleBytes, a.scalesLayers, aRows, scaleRow),
      b: openRows(b.key, b.scaleBytes, b.scalesLayers, bRows, scaleRow),
    },
  };
}

function byteVec(bytes) {
  return Buffer.concat([u64le(bytes.length), bytes]);
}

// PlainProofV4::to_bytes: strict fixint bincode of
//   job bytes, ancestor chain bytes, values {a, b}, scales {a, b}, no MoE.
function serializeProof(proof) {
  return Buffer.concat([
    byteVec(proof.job),
    byteVec(proof.chain),
    serializeMatrixMerkleProof(proof.values.a),
    serializeMatrixMerkleProof(proof.values.b),
    serializeMatrixMerkleProof(proof.scales.a),
    serializeMatrixMerkleProof(proof.scales.b),
    Buffer.from([BINCODE_OPTION_NONE]),
  ]);
}

// The inverse, for the tools that check a proof before it is sent. Throws on
// anything that is not one whole dense proof.
function parseProof(bytes) {
  let o = 0;
  const take = (n) => {
    if (o + n > bytes.length) throw new Error('not a valid PlainProofV4: it ends early');
    o += n;
    return bytes.subarray(o - n, o);
  };
  const u64 = () => Number(take(8).readBigUInt64LE(0));
  const count = () => {
    const n = u64();
    if (n > bytes.length - o) throw new Error('not a valid PlainProofV4: it ends early');
    return n;
  };
  const many = (one) => Array.from({ length: count() }, one);
  const matrix = () => {
    const leafData = many(() => Buffer.from(take(count())));
    const leafIndices = many(u64);
    const totalLeaves = u64();
    const root = Buffer.from(take(32));
    const siblings = many(() => Buffer.from(take(32)));
    return { proof: { leafData, leafIndices, totalLeaves, root, siblings }, rowIndices: many(u64) };
  };
  const job = Buffer.from(take(count()));
  const chain = Buffer.from(take(count()));
  const proof = { job, chain, values: { a: matrix(), b: matrix() }, scales: { a: matrix(), b: matrix() } };
  if (take(1)[0] !== BINCODE_OPTION_NONE) throw new Error('not a dense PlainProofV4: it carries MoE fields');
  if (o !== bytes.length) throw new Error(`not a valid PlainProofV4: ${bytes.length - o} bytes left over`);
  return proof;
}

// The job tuple back out of a proof's job bytes (dense jobs only).
function parseJobWire(wire) {
  if (wire.length !== ANCESTOR_LEN + 19 + 9) throw new Error('not a dense FP8 job tuple');
  const pB = wire.subarray(ANCESTOR_LEN);
  const pA = pB.subarray(19);
  const operand = (b, o) => ({ numRows: b.readUInt32LE(o), hashId: b[o + 4], tileSize: b.readUInt32LE(o + 5) });
  return {
    ancestorHeader: Buffer.from(wire.subarray(0, ANCESTOR_LEN)),
    job: { device: pB[7], k: pB.readUInt32LE(0), a: operand(pA, 0), b: operand(pB, 8) },
  };
}

// The opened rows of one plane as contiguous bytes, cut from its leaves.
function rowsOf(matrix, rowBytes) {
  const leaf = new Map(matrix.proof.leafIndices.map((idx, i) => [idx, matrix.proof.leafData[i]]));
  const out = Buffer.alloc(matrix.rowIndices.length * rowBytes);
  matrix.rowIndices.forEach((row, r) => {
    for (let j = 0; j < rowBytes; j++) {
      const at = row * rowBytes + j;
      out[r * rowBytes + j] = leaf.get(Math.floor(at / merkle.CHUNK_LEN))[at % merkle.CHUNK_LEN];
    }
  });
  return out;
}

// Check a proof the way a pool or node does, short of the ancestry link:
// the four openings against their roots, then ./fp8 from the opened rows to
// the ticket. Returns { ticket, wins, tile } or throws with the reason.
function verifyProof(proposedHeader, proof, nbits = proposedHeader.readUInt32LE(72)) {
  const { ancestorHeader, job } = parseJobWire(proof.job);
  const kA = fp8.keyA(proposedHeader);
  const kB = fp8.keyB(ancestorHeader);
  const k = job.k;
  const aRows = proof.values.a.rowIndices;
  const bRows = proof.values.b.rowIndices;
  if (aRows.join() !== proof.scales.a.rowIndices.join() || bRows.join() !== proof.scales.b.rowIndices.join()) {
    throw new Error('values and scales must open the same rows');
  }
  for (const [name, matrix, key, rowBytes] of [
    ['A values', proof.values.a, kA, k], ['A scales', proof.scales.a, kA, SCALE_ROW_BYTES(k)],
    ['B values', proof.values.b, kB, k], ['B scales', proof.scales.b, kB, SCALE_ROW_BYTES(k)],
  ]) {
    const minimal = merkle.leafIndicesFromRows(matrix.rowIndices, rowBytes);
    if (minimal.join() !== matrix.proof.leafIndices.join()) throw new Error(`${name}: not the minimal leaf set`);
    if (!merkle.verifyProof(key, matrix.proof)) throw new Error(`${name}: the opening does not reach its root`);
  }
  const seeds = fp8.noiseSeeds({
    proposedHeader, ancestorHeader, job,
    hashA: fp8.operandDigest(proof.values.a.proof.root, proof.scales.a.proof.root, kA),
    hashB: fp8.operandDigest(proof.values.b.proof.root, proof.scales.b.proof.root, kB),
  });
  const scalesOf = (matrix) => {
    const b = rowsOf(matrix, SCALE_ROW_BYTES(k));
    return Uint16Array.from({ length: b.length / 2 }, (_, i) => b.readUInt16LE(i * 2));
  };
  const result = fp8.evaluateTile({
    device: job.device, k, aRows, bRows, seedA: seeds.seedA, seedB: seeds.seedB,
    aInts: new Int8Array(rowsOf(proof.values.a, k)), aScales: scalesOf(proof.scales.a),
    bInts: new Int8Array(rowsOf(proof.values.b, k)), bScales: scalesOf(proof.scales.b),
  });
  if (!result) throw new Error('the jackpot policy refuses this tile');
  return {
    ticket: result.ticket,
    tile: result.tile,
    wins: fp8.meetsDifficulty(result.ticket, nbits, aRows.length, bRows.length, k),
  };
}

module.exports = {
  HASH_ID_CHUNK_1024, UNIT_SCALE, HEADER_LEN, ANCESTOR_LEN,
  rng, triangular, fillValues, unitScales, planeBytes,
  commitOperand, layersFromNodes, coreTree, digestOf, updateLeaf,
  makeJob, jobWire, rowParams,
  ticketBound, attemptSeeds, noiseLines,
  openRows, tileRows, buildProof, serializeProof, parseProof, parseJobWire, rowsOf, verifyProof,
};
