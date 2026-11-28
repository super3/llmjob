'use strict';

const fs = require('fs');
const path = require('path');
const blake3 = require('../src/shared/miner/blake3');
const chacha8 = require('../src/shared/neb/chacha8');
const nh = require('../src/shared/neb/nebhash2');

// The chain's own vectors (vectors/nebhash2.txt of github.com/SilasEigen/neb, Apache-2.0), made by its C reference.
const VECTORS = fs.readFileSync(path.join(__dirname, '..', 'native', 'neb', 'vectors', 'nebhash2.txt'), 'utf8').trim().split('\n');

function shareKey(arch, step, j) {
  const le = Buffer.alloc(8);
  le.writeUInt32LE(step, 0);
  le.writeUInt32LE(j, 4);
  return blake3.hash(Buffer.concat([Buffer.from('neb-test-share2'), Buffer.from([arch]), le]));
}

// Replays the vectors of one architecture up to and including step `lastStep`.
function replay(arch, lastStep) {
  let m = null;
  let ds = null;
  let sum = null;
  let shares = 0;
  for (const line of VECTORS.slice(1)) {
    const p = line.split(' ');
    if (Number(p[1]) !== arch) continue;
    if (p[0] === 'init') {
      m = nh.modelInit(arch, Buffer.alloc(32));
      ds = nh.syn2Dataset(arch);
      sum = new Int32Array(m.w.length);
      expect(nh.weightsHash(nh.packWeights(m.w)).toString('hex')).toBe(p[2]);
      expect(nh.residualHash(m.r).toString('hex')).toBe(p[3]);
    } else if (Number(p[2]) > lastStep) {
      break;
    } else if (p[0] === 'share') {
      const r = nh.shareEval(m, ds, shareKey(arch, Number(p[2]), Number(p[3])));
      expect(r.digest.toString('hex')).toBe(p[4]);
      expect(r.ticket.toString('hex')).toBe(p[5]);
      expect(r.idx.join(' ')).toBe(p.slice(6).join(' '));
      for (let f = 0; f < sum.length; f++) sum[f] += r.g[f];
      shares++;
    } else {
      nh.modelApply(m, sum);
      sum.fill(0);
      expect(nh.weightsHash(nh.packWeights(m.w)).toString('hex')).toBe(p[4]);
      expect(nh.residualHash(m.r).toString('hex')).toBe(p[5]);
    }
  }
  return shares;
}

describe('nebhash v2 against the chain\'s vectors', () => {
  test('KEP1: every share and every model state', () => {
    expect(replay(nh.KEP1, 5)).toBe(87);
  }, 120000);

  test('RET2: the first two steps', () => {
    expect(replay(nh.RET2, 1)).toBe(33);
  }, 120000);
});

describe('ChaCha8', () => {
  test('the keystream is the blocks in order, cut to length', () => {
    const key = Buffer.alloc(32, 7);
    const long = chacha8.stream(key, 130);
    expect(long.length).toBe(130);
    expect(chacha8.stream(key, 64).equals(long.subarray(0, 64))).toBe(true);
    expect(chacha8.stream(key, 3).equals(long.subarray(0, 3))).toBe(true);
    const block1 = chacha8.blockWords(chacha8.keyWords(key), 1);
    expect(long.readUInt32LE(64)).toBe(block1[0]);
  });

  test('the block counter is 64 bits', () => {
    const k = chacha8.keyWords(Buffer.alloc(32, 1));
    expect(Array.from(chacha8.blockWords(k, 2 ** 32))).not.toEqual(Array.from(chacha8.blockWords(k, 0)));
  });
});

describe('weights', () => {
  test('pack and unpack are inverses, low nibble first', () => {
    const w = Int8Array.from([-8, 7, 0, -1, 3]);
    const packed = nh.packWeights(w);
    expect(packed.toString('hex')).toBe('78f003');
    expect(Array.from(nh.unpackWeights(packed, 5))).toEqual(Array.from(w));
  });

  test('a model must be the network\'s size and a known network', () => {
    expect(() => nh.modelFromPacked(9, Buffer.alloc(13000))).toThrow('unknown NEB architecture 9');
    expect(() => nh.modelFromPacked(nh.KEP1, Buffer.alloc(12))).toThrow('this network takes 13000');
    expect(nh.modelFromPacked(nh.KEP1, Buffer.alloc(13000)).w.length).toBe(26000);
    expect(nh.archOf(3)).toBeNull();
  });

  test('rshift rounds half up and leaves a zero shift alone', () => {
    expect(nh.rshift(16, 5)).toBe(1);
    expect(nh.rshift(-16, 5)).toBe(0);
    expect(nh.rshift(-17, 5)).toBe(-1);
    expect(nh.rshift(5, 0)).toBe(5);
  });
});

describe('live samples', () => {
  const ds = nh.syn2Dataset(nh.KEP1);
  const trained = nh.modelFromPacked(nh.KEP1, Buffer.from(
    fs.readFileSync(path.join(__dirname, '..', 'native', 'neb', 'vectors', 'syn2-trained.hex'), 'utf8').trim(), 'hex'));
  const live = nh.liveSamples(trained, ds);

  test('a sample whose output equals its target adds nothing to the gradient', () => {
    const A = nh.archOf(nh.KEP1);
    const W = nh.makeWork(A);
    const dead = live.indexOf(0);
    const g = new Int32Array(A.nweights);
    expect(nh.sampleGrad(A, trained, W, ds.values, dead * A.nin, ds.labels[dead], g)).toBe(false);
    // The backward pass run anyway, as the chain's reference runs it, leaves the gradient zero.
    for (let l = A.layers.length - 1; l >= 0; l--) nh.backwardLayer(A, trained.w, g, W, l);
    expect(g.every((v) => v === 0)).toBe(true);
    const alive = live.indexOf(1);
    expect(nh.sampleGrad(A, trained, W, ds.values, alive * A.nin, ds.labels[alive], g)).toBe(true);
    expect(g.some((v) => v !== 0)).toBe(true);
  });

  test('this model answers class 0 for every sample: half the test dataset is live', () => {
    expect(live.reduce((a, b) => a + b, 0)).toBe(1024);
    expect(Array.from(live.subarray(0, 4))).toEqual([0, 1, 0, 1]);
  });

  test('liveCount is the live ones among an attempt\'s 16 samples, and shareEval agrees', () => {
    const key = Buffer.alloc(32, 9);
    const idx = nh.shareIndices(key, ds.n);
    expect(nh.liveCount(key, live)).toBe(idx.filter((i) => live[i]).length);
    expect(nh.shareEval(trained, ds, key).live).toBe(nh.liveCount(key, live));
  });

  test('candidateShare is the binomial tail', () => {
    expect(nh.candidateShare(0, 2)).toBe(1);
    expect(nh.candidateShare(1, 2)).toBe(0);
    expect(nh.candidateShare(0.5, 0)).toBeCloseTo(2 ** -16, 12);
    expect(nh.candidateShare(0.5, 2)).toBeCloseTo((1 + 16 + 120) / 65536, 12);
  });
});

describe('tickets and targets', () => {
  test('both are little-endian integers', () => {
    const t = Buffer.alloc(32);
    t[31] = 1;
    const below = Buffer.alloc(32, 0xff);
    below[31] = 0;
    const above = Buffer.alloc(32);
    above[31] = 2;
    expect(nh.meetsTarget(below, t)).toBe(true);
    expect(nh.meetsTarget(t, t)).toBe(true);
    expect(nh.meetsTarget(above, t)).toBe(false);
  });

  test('the ticket is BLAKE3 of the digest and the key', () => {
    const d = Buffer.alloc(32, 1);
    const k = Buffer.alloc(32, 2);
    expect(nh.ticketOf(d, k).equals(blake3.hash(Buffer.concat([d, k])))).toBe(true);
  });
});

describe('the dataset file', () => {
  function file(sets, tail = Buffer.alloc(0)) {
    const head = Buffer.alloc(8);
    head.write('NMDS');
    head.writeUInt32LE(sets.length, 4);
    return Buffer.concat([head, ...sets, tail]);
  }
  function set(magic, arch, n, records) {
    const h = Buffer.alloc(12);
    h.write(magic);
    h.writeUInt32LE(arch, 4);
    h.writeUInt32LE(n, 8);
    return Buffer.concat([h, records]);
  }
  const rec = Buffer.alloc(785 * 2, 3);
  rec[784] = 1;
  rec[785 + 784] = 0;

  test('reads each dataset\'s values and labels', () => {
    const sets = nh.parseDatasets(file([set('NMD2', 1, 2, rec), set('NMD2', 1, 1, rec.subarray(0, 785))]));
    expect(sets.map((s) => [s.arch, s.n])).toEqual([[1, 2], [1, 1]]);
    expect(Array.from(sets[0].labels)).toEqual([1, 0]);
    expect(sets[0].values.length).toBe(2 * 784);
    expect(sets[0].values[0]).toBe(3);
  });

  test('refuses what it cannot read', () => {
    expect(() => nh.parseDatasets(Buffer.from('nope'))).toThrow('not a NEB dataset file');
    expect(() => nh.parseDatasets(Buffer.from('XXXXXXXXXXXX'))).toThrow('not a NEB dataset file');
    expect(() => nh.parseDatasets(file([Buffer.alloc(4)]))).toThrow('cut short');
    expect(() => nh.parseDatasets(file([set('NMD1', 1, 2, rec)]))).toThrow('reads NMD2 only');
    expect(() => nh.parseDatasets(file([set('NMD2', 7, 2, rec)]))).toThrow('unknown network (7)');
    expect(() => nh.parseDatasets(file([set('NMD2', 1, 3, rec)]))).toThrow('cut short');
    expect(() => nh.parseDatasets(file([set('NMD2', 1, 0, rec)]))).toThrow('cut short');
    expect(() => nh.parseDatasets(file([set('NMD2', 1, 2, rec)], Buffer.alloc(1)))).toThrow('bytes after its last dataset');
  });
});
