'use strict';

const fp8 = require('../src/shared/miner/fp8');
const host = require('../src/shared/miner/fp8Host');
const merkle = require('../src/shared/miner/merkle');
const realProof = require('./fixtures/fp8Rtx5080Testnet2Proof.json');
const rtx5080 = require('./fixtures/fp8Rtx5080Testnet2.json');

// The host side of the FP8 miner. The strongest check here is the last block:
// the PlainProofV4 that Pearl's own miner made on an RTX 5080 and testnet2
// accepted as block 150,688. Our parser reads it, our serializer writes the
// same bytes back, our Merkle code reaches its four roots, and the reference
// gets the winning ticket from its opened rows.

const B200 = fp8.DEVICE_B200;
const EASY_NBITS = 0x207FFFFF;

function header(nbits = EASY_NBITS, fill = 5) {
  const h = Buffer.alloc(host.HEADER_LEN, fill);
  h.writeUInt32LE(nbits, 72);
  return h;
}

// A whole small job: 16 x 16 x 1024, one tile.
function smallJob(seed = 1) {
  const job = host.makeJob({ m: 16, n: 16, k: 1024, h: 16, w: 16 });
  const proposedHeader = header();
  const ancestorHeader = Buffer.alloc(host.ANCESTOR_LEN, 6);
  const a = host.commitOperand(fp8.keyA(proposedHeader), host.fillValues(seed, 16, 1024), host.unitScales(16, 1024));
  const b = host.commitOperand(fp8.keyB(ancestorHeader), host.fillValues(seed + 100, 16, 1024), host.unitScales(16, 1024));
  const proof = host.buildProof({ ancestorHeader, job, a, b, tileA: 0, tileB: 0 });
  return { job, proposedHeader, ancestorHeader, a, b, proof };
}

describe('operand data', () => {
  test('the generator is deterministic and a zero seed still runs', () => {
    const a = host.rng(7);
    const b = host.rng(7);
    expect([a(), a(), a()]).toEqual([b(), b(), b()]);
    expect(host.rng(0)()).toBe(host.rng(0x9E3779B9)());
  });

  test('values are centred triangular int8', () => {
    expect(host.triangular(0)).toBe(-127);
    expect(host.triangular(0x007F007F)).toBe(127);
    const v = host.fillValues(3, 4, 1024);
    expect(v).toBeInstanceOf(Int8Array);
    expect(v.length).toBe(4096);
    const mean = v.reduce((s, x) => s + x, 0) / v.length;
    expect(Math.abs(mean)).toBeLessThan(5);
    expect(Math.min(...v)).toBeGreaterThanOrEqual(-127);
  });

  test('unit scales are BF16 1.0, one per 8 values', () => {
    const s = host.unitScales(2, 1024);
    expect(s.length).toBe(256);
    expect(fp8.bf16ToF32(s[5])).toBe(1);
  });

  test('planes are committed as int8 bytes and little-endian BF16', () => {
    expect(host.planeBytes(Int8Array.of(-1, 2))).toEqual(Buffer.from([0xFF, 2]));
    expect(host.planeBytes(Uint16Array.of(0x3F80))).toEqual(Buffer.from([0x80, 0x3F]));
  });
});

describe('commitment', () => {
  const key = Buffer.alloc(32, 1);

  test('an operand is two trees and their combined digest', () => {
    const c = host.commitOperand(key, host.fillValues(1, 16, 1024), host.unitScales(16, 1024));
    expect(c.valuesLayers[0].length).toBe(16);
    expect(c.scalesLayers[0].length).toBe(4);
    expect(c.digest).toEqual(fp8.operandDigest(merkle.rootOf(c.valuesLayers), merkle.rootOf(c.scalesLayers), key));
  });

  test('another tree builder can be passed in', () => {
    const calls = [];
    const tree = (k, bytes) => {
      calls.push(bytes.length);
      return merkle.buildLayers(k, bytes);
    };
    const ints = host.fillValues(1, 16, 1024);
    const c = host.commitOperand(key, ints, host.unitScales(16, 1024), tree);
    expect(calls).toEqual([16384, 4096]);
    expect(c.digest).toEqual(host.commitOperand(key, ints, host.unitScales(16, 1024)).digest);
  });

  test('the core\'s flat tree becomes the same layers', () => {
    const ints = host.fillValues(3, 16, 1024);
    const bytes = host.planeBytes(ints);
    const layers = merkle.buildLayers(key, bytes);
    const core = { tree: jest.fn((k, plane) => new Uint8Array(Buffer.concat(merkle.buildLayers(k, plane).flat()))) };
    expect(host.layersFromNodes(core.tree(key, bytes))).toEqual(layers);
    const c = host.commitOperand(key, ints, host.unitScales(16, 1024), host.coreTree(core));
    expect(core.tree).toHaveBeenCalledWith(key, bytes);
    expect(c.digest).toEqual(host.commitOperand(key, ints, host.unitScales(16, 1024)).digest);
  });

  test('a plane that is not a power-of-two number of chunks is refused', () => {
    expect(() => host.commitOperand(key, new Int8Array(3 * 1024), host.unitScales(16, 1024))).toThrow(/power-of-two/);
    expect(() => host.commitOperand(key, new Int8Array(1024), host.unitScales(16, 1024))).toThrow(/power-of-two/);
    expect(() => host.commitOperand(key, new Int8Array(2048), new Uint16Array(100))).toThrow(/power-of-two/);
  });

  test('updating one leaf gives the tree a full rebuild gives', () => {
    const ints = host.fillValues(1, 16, 1024);
    const c = host.commitOperand(key, ints, host.unitScales(16, 1024));
    for (const leaf of [0, 5, 15]) {
      c.values[leaf * 1024 + 3] ^= 0x55;
      host.updateLeaf(key, c.valuesLayers, c.values, leaf);
      expect(c.valuesLayers).toEqual(merkle.buildLayers(key, c.values));
    }
    expect(host.digestOf(key, c.valuesLayers, c.scalesLayers)).not.toEqual(c.digest);
  });
});

describe('the job tuple', () => {
  test('makeJob checks the shape', () => {
    const job = host.makeJob({ m: 64, n: 128, k: 2048, h: 4, w: 64 });
    expect(job).toEqual({
      device: B200, k: 2048,
      a: { numRows: 64, hashId: 3, tileSize: 4 },
      b: { numRows: 128, hashId: 3, tileSize: 64 },
    });
    expect(host.makeJob({ device: fp8.DEVICE_H100, m: 16, n: 16, k: 1024, h: 16, w: 16 }).device).toBe(fp8.DEVICE_H100);
    expect(() => host.makeJob({ m: 64, n: 128, k: 2048, h: 4, w: 8 })).toThrow(/tile/);
    for (const k of [512, 131072, 1040]) {
      expect(() => host.makeJob({ m: 64, n: 128, k, h: 16, w: 16 })).toThrow(/k must be/);
    }
    expect(() => host.makeJob({ m: 60, n: 128, k: 1024, h: 16, w: 16 })).toThrow(/whole tiles/);
    expect(() => host.makeJob({ m: 64, n: 120, k: 1024, h: 16, w: 16 })).toThrow(/whole tiles/);
  });

  test('the wire form is ancestor, pB, pA and reads back', () => {
    const job = host.makeJob({ m: 64, n: 128, k: 2048, h: 4, w: 64 });
    const ancestor = Buffer.alloc(108, 9);
    const wire = host.jobWire(ancestor, job);
    expect(wire.length).toBe(108 + 19 + 9);
    expect(host.parseJobWire(wire)).toEqual({ ancestorHeader: ancestor, job });
    expect(() => host.jobWire(Buffer.alloc(76), job)).toThrow(/108 bytes/);
    expect(() => host.parseJobWire(wire.subarray(1))).toThrow(/dense FP8 job/);
  });
});

describe('what the core is handed', () => {
  test('row parameters agree with the reference quantizer', () => {
    const k = 1024;
    const ints = host.fillValues(4, 2, k);
    const scales = host.unitScales(2, k);
    scales[3] = 0x4000; // 2.0 on one block
    const p = host.rowParams(B200, ints, scales, 2, k);
    const norms = fp8.exactNorms(ints, scales, 2, k);
    const [alpha, beta] = fp8.deriveRowScales(B200, norms[0][0], norms[0][1], 32);
    expect(p.beta[0]).toBe(fp8.bf16ToF32(beta));
    expect(p.signal[0]).toBe(fp8.bf16ToF32(alpha));
    expect(p.signal[3]).toBe(2 * fp8.bf16ToF32(alpha));
    const rms = fp8.bf16ToF32(alpha) * fp8.bf16ToF32(norms[0][0]);
    expect(p.rewrite[0] & 0xFF).toBe(fp8.f32ToE4m3(Math.fround(Math.fround(0.08) * rms)));
    expect(p.rewrite[0] >>> 11).toBe(Math.floor(Math.log2(rms)) - 2);
  });

  test('a low-RMS row uses the wider strip, and a row under the floor is refused', () => {
    const spike = (k) => {
      const row = new Int8Array(k);
      row[0] = 127;
      return host.rowParams(B200, row, host.unitScales(1, k), 1, k);
    };
    // One spike in 4096: alpha * l2 is about 6.7, so g is 0.1.
    const p = spike(4096);
    expect(p.rewrite[0] >>> 11).toBe(0);
    expect(p.rewrite[0] & 0xFF).toBe(fp8.f32ToE4m3(Math.fround(Math.fround(0.1) * 448 / (64 + Math.sqrt(8)))));
    expect(() => spike(16384)).toThrow(/outside the policy/);
  });

  test('the bound is target * h * w * k, capped at 2^256 - 1', () => {
    expect(host.ticketBound(EASY_NBITS, 16, 16, 1024)).toEqual(Buffer.alloc(32, 0xFF));
    const nbits = 0x1D00FFFF;
    const bound = host.ticketBound(nbits, 16, 16, 1024);
    const want = fp8.nbitsToTarget(nbits) * 262144n;
    expect(BigInt('0x' + Buffer.from(bound).reverse().toString('hex'))).toBe(want);
  });

  test('an attempt has its seeds, jackpot key and bound', () => {
    const { job, proposedHeader, ancestorHeader, a, b } = smallJob();
    const s = host.attemptSeeds({ proposedHeader, ancestorHeader, job, hashA: a.digest, hashB: b.digest });
    const want = fp8.noiseSeeds({ proposedHeader, ancestorHeader, job, hashA: a.digest, hashB: b.digest });
    expect(s.seedA).toEqual(want.seedA);
    expect(s.seedB).toEqual(want.seedB);
    expect(s.jackpotKey).toEqual(fp8.subkey(fp8.LABELS.jackpot, want.seedA));
    expect(s.bound).toEqual(Buffer.alloc(32, 0xFF));
    expect(() => host.attemptSeeds({ proposedHeader: Buffer.alloc(80), ancestorHeader, job, hashA: a.digest, hashB: b.digest }))
      .toThrow(/76 bytes/);
  });

  test('noise lines are the reference lines', () => {
    const seed = Buffer.alloc(32, 2);
    const lines = host.noiseLines(seed, fp8.SIDE_B, fp8.FACTOR_E, 5, 3);
    const want = fp8.sampleNoise({ k: 1024, seedA: seed, seedB: seed, aRows: [0], bRows: [5, 6, 7] });
    expect(lines).toEqual(want.b.e);
  });
});

describe('PlainProofV4', () => {
  test('a mined tile serializes, parses and verifies', () => {
    const { proposedHeader, proof, a, job } = smallJob();
    expect(host.tileRows(3, 4)).toEqual([12, 13, 14, 15]);
    expect(proof.values.a.rowIndices).toEqual(host.tileRows(0, 16));
    expect(proof.values.a.proof.leafData.length).toBe(16);
    expect(proof.scales.a.proof.leafData.length).toBe(4);
    const bytes = host.serializeProof(proof);
    const parsed = host.parseProof(bytes);
    expect(host.serializeProof(parsed)).toEqual(bytes);
    expect(parsed.job).toEqual(proof.job);
    expect(host.rowsOf(parsed.values.a, job.k)).toEqual(a.values);
    const result = host.verifyProof(proposedHeader, parsed);
    expect(result.wins).toBe(true);
    expect(result.tile.length).toBe(256);
    // A share target can be passed in place of the header's.
    expect(host.verifyProof(proposedHeader, parsed, 0x03000001).wins).toBe(false);
  });

  test('a proof that is cut, padded or carries MoE fields is refused', () => {
    const bytes = host.serializeProof(smallJob().proof);
    expect(() => host.parseProof(bytes.subarray(0, 4))).toThrow(/ends early/);
    expect(() => host.parseProof(bytes.subarray(0, 100))).toThrow(/ends early/);
    expect(() => host.parseProof(Buffer.concat([bytes, Buffer.from([0])]))).toThrow(/1 bytes left over/);
    const moe = Buffer.from(bytes);
    moe[moe.length - 1] = 1;
    expect(() => host.parseProof(moe)).toThrow(/MoE/);
  });

  test('verifyProof names what is wrong with a bad proof', () => {
    const { proposedHeader, proof } = smallJob();
    const copy = () => host.parseProof(host.serializeProof(proof));

    const rows = copy();
    rows.scales.a.rowIndices = rows.scales.a.rowIndices.map((r) => r + 1);
    expect(() => host.verifyProof(proposedHeader, rows)).toThrow(/same rows/);
    const rowsB = copy();
    rowsB.scales.b.rowIndices = [];
    expect(() => host.verifyProof(proposedHeader, rowsB)).toThrow(/same rows/);

    const leaves = copy();
    leaves.values.b.proof.leafIndices[0] = 99;
    expect(() => host.verifyProof(proposedHeader, leaves)).toThrow(/B values: not the minimal leaf set/);

    const data = copy();
    data.scales.a.proof.leafData[0][0] ^= 1;
    expect(() => host.verifyProof(proposedHeader, data)).toThrow(/A scales: the opening does not reach its root/);
  });

  test('a tile the jackpot policy refuses does not verify', () => {
    const job = host.makeJob({ m: 16, n: 16, k: 1024, h: 16, w: 16 });
    const proposedHeader = header();
    const ancestorHeader = Buffer.alloc(host.ANCESTOR_LEN, 6);
    // A constant fill: what the INT7 miner used, and what the policy rejects.
    const flat = new Int8Array(16 * 1024).fill(40);
    const a = host.commitOperand(fp8.keyA(proposedHeader), flat, host.unitScales(16, 1024));
    const b = host.commitOperand(fp8.keyB(ancestorHeader), flat, host.unitScales(16, 1024));
    const proof = host.buildProof({ ancestorHeader, job, a, b, tileA: 0, tileB: 0 });
    expect(() => host.verifyProof(proposedHeader, proof)).toThrow(/policy refuses/);
  });
});

describe('the proof of testnet2 block 150,688 (RTX 5080, Pearl\'s miner)', () => {
  const bytes = Buffer.from(realProof.plain_proof, 'base64');
  const proposedHeader = Buffer.from(realProof.mining_job.incomplete_header_bytes, 'base64');

  test('parses, and serializes back to the same bytes', () => {
    const proof = host.parseProof(bytes);
    expect(host.serializeProof(proof).equals(bytes)).toBe(true);
    expect(proof.chain.length).toBe(0);
    const { ancestorHeader, job } = host.parseJobWire(proof.job);
    expect(ancestorHeader).toEqual(Buffer.from(realProof.mining_job.ancestor_headers[0], 'base64'));
    expect(job).toEqual({ device: B200, k: rtx5080.k, a: rtx5080.a, b: rtx5080.b });
    expect(proof.values.a.rowIndices).toEqual(rtx5080.aRows);
  });

  test('its openings reach their roots and its tile wins the block', () => {
    const result = host.verifyProof(proposedHeader, host.parseProof(bytes));
    expect(result.ticket.toString('hex')).toBe('dff324097fee6a158eedee360147e58f2244da047ecfe18d6566fd0100000000');
    expect(result.wins).toBe(true);
  });
});
