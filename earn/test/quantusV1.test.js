'use strict';

// Core API 1, which the Intel Arc core keeps until nonce layout v2 has run on
// an Arc card: the old layout's helpers in qpow.js, verifyCore's version 1
// checks, and the engine's version 1 work.

const { EventEmitter } = require('events');
const qc = require('../src/main/quantusCore');
const q = require('../src/shared/quantus/qpow');
const { QuantusEngine } = require('../src/main/quantusEngine');

const HASH = '415c416d00ae68342369dbc22db3b8f805a1b1a0ea0848cf08116ab894e46753';
const LANE = Buffer.from('abababababab0000', 'hex');
const EXTRANONCE = Buffer.from('0004ee45', 'hex');

// A version 1 core in JavaScript, as the SYCL core computes: counters under a
// midstate and tail.
function cpuV1() {
  return {
    async scan(w) {
      const mid = q.midstateFromBytes(w.midstate);
      const start = w.start.readBigUInt64BE(0);
      const target = w.targetHigh.readBigUInt64BE(0);
      const found = [];
      for (let c = start; c < start + BigInt(w.count); c++) if (q.nonceTop64(mid, w.tail, c) <= target) found.push(c);
      return { hits: Buffer.concat(found.slice(0, w.maxHits).map(qc.u64Bytes)), overflow: found.length > w.maxHits };
    },
  };
}

describe('the version 1 layout', () => {
  test('a nonce: the extranonce, zeros, the lane in bytes 48..55, the counter in 56..63', () => {
    const n = q.buildNonceV1(EXTRANONCE, LANE, 0x0102n);
    expect(n.subarray(0, 4)).toEqual(EXTRANONCE);
    expect(n.subarray(4, 48)).toEqual(Buffer.alloc(44));
    expect(n.subarray(48, 56)).toEqual(LANE);
    expect(n.readBigUInt64BE(56)).toBe(0x0102n);
    expect(q.buildNonceV1(null, LANE, 0n).subarray(0, 48)).toEqual(Buffer.alloc(48));
    expect(() => q.buildNonceV1(Buffer.alloc(33), LANE, 0n)).toThrow('an extranonce of more than 32 bytes');
    expect(() => q.buildNonceV1(EXTRANONCE, Buffer.alloc(7), 0n)).toThrow('lane must be 8 bytes');
    const { high, tail, counter } = q.splitNonceV1(n);
    expect(Buffer.concat([high, tail])).toEqual(n.subarray(0, 56));
    expect(counter).toBe(0x0102n);
    expect(() => q.splitNonceV1(Buffer.alloc(63))).toThrow('nonce must be 64 bytes');
  });

  test('nonceTop64 is the top of the full hash', () => {
    const header = Buffer.from(HASH, 'hex');
    for (const c of [0n, 77n, q.U64_MAX]) {
      const n = q.buildNonceV1(EXTRANONCE, LANE, c);
      const { high, tail } = q.splitNonceV1(n);
      expect(q.nonceTop64(q.midstate(header, high), tail, c)).toBe(q.toBig(q.nonceHash(header, n)) >> 448n);
    }
  });

  test('the midstate as 96 bytes, and back; a short buffer or an element not below p is refused', () => {
    const mid = q.midstate(Buffer.from(HASH, 'hex'), Buffer.alloc(32, 1));
    expect(q.midstateFromBytes(q.midstateBytes(mid))).toEqual(mid);
    expect(() => q.midstateFromBytes(Buffer.alloc(95))).toThrow('midstate must be 96 bytes');
    const bad = q.midstateBytes(mid);
    bad.writeBigUInt64LE(0xffffffffffffffffn, 8);
    expect(() => q.midstateFromBytes(bad)).toThrow('midstate element 1 is not below the field prime');
  });
});

describe('verifyCore at version 1', () => {
  const SMALL = { api: 1, exactCount: 64, bigCount: 256, random: (n) => Buffer.alloc(n, 0x5a) };

  test('a right version 1 core passes every check', async () => {
    let t = 0;
    const r = await qc.verifyCore(cpuV1(), { ...SMALL, now: () => (t += 500) });
    expect(r.checks.map((c) => [c.name, c.ok])).toEqual([
      ['vectors', true], ['exact', true], ['top', true], ['overflow', true], ['big', true],
    ]);
    expect(r.checks[0].detail).toBe('5 of 5');
    expect(r.ok).toBe(true);
    expect(r.hashrate).toBe(512);
  });

  test('one that reports the counter after each hit fails, and says where', async () => {
    const cpu = cpuV1();
    const wrong = {
      async scan(w) {
        const shifted = qc.hitList(await cpu.scan(w)).map((c) => (c + 1n) & q.U64_MAX);
        return { hits: Buffer.concat(shifted.map(qc.u64Bytes)), overflow: false };
      },
    };
    const r = await qc.verifyCore(wrong, SMALL);
    const by = Object.fromEntries(r.checks.map((c) => [c.name, c]));
    expect(by.vectors.detail).toBe('wrong on vector 0, 1, 2, 3, 4');
    expect(by.exact.detail).toMatch(/: they differ$/);
    expect(by.overflow.ok).toBe(false);
    expect(by.big.ok).toBe(false);
    expect(r.ok).toBe(false);
  });

  test('the defaults: 4096 for the exact ranges, 2^30 in one scan, version 1 work', async () => {
    const seen = [];
    const fake = { scan: async (w) => { seen.push(w); return { hits: Buffer.alloc(0), overflow: false }; } };
    const r = await qc.verifyCore(fake, { api: 1 });
    expect(seen.map((w) => w.count)).toEqual([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4096, 4096, 64, 2 ** 30]);
    expect(seen.every((w) => w.midstate.length === 96 && w.tail.length === 24 && !w.line)).toBe(true);
    expect(seen[11].start.readBigUInt64BE(0)).toBe(q.U64_MAX - 4095n);
    expect(r.ok).toBe(false);
  });
});

describe('the engine with a version 1 core', () => {
  beforeEach(() => jest.useFakeTimers({ doNotFake: ['setImmediate', 'nextTick', 'queueMicrotask'] }));
  afterEach(() => jest.useRealTimers());
  const flush = async (n = 3) => { for (let i = 0; i < n; i++) await new Promise((r) => setImmediate(r)); };

  test('counts through the counter under the job\'s midstate, and sends the version 1 nonce', async () => {
    const scans = [];
    const core = { device: { name: 'Arc' }, scan: jest.fn((w) => new Promise((resolve) => scans.push({ w, resolve }))), close: jest.fn() };
    const sock = new EventEmitter();
    sock.lines = [];
    sock.write = (l) => sock.lines.push(JSON.parse(l));
    sock.destroy = () => sock.emit('close');
    let clock = 1000000;
    const engine = new QuantusEngine({
      connect: () => sock,
      factory: { api: 1, devices: [{ index: 0 }], createCore: () => core },
      agent: 'llmjob-earn/9.9.9',
      now: () => clock,
      random: (n) => Buffer.alloc(n, 0xab),
    });
    engine.on('error', () => {});
    engine.start({ address: 'qzjTEdq3Wd1d98kaeE1RTDPBBYBfY6XpfCtaLq97wmfcACqCd', worker: 'w', endpoint: 'us.quantus.herominers.com:1155' });
    sock.emit('connect');
    sock.emit('data', JSON.stringify({ id: 1, result: { id: 's', status: 'OK', job: {
      job_id: 'j1', mining_hash: HASH, target: 'ff'.repeat(64), difficulty: 1, extranonce: '0004ee45',
    } } }) + '\n');
    await flush();
    const high = Buffer.alloc(32);
    EXTRANONCE.copy(high);
    const mid = q.midstateBytes(q.midstate(Buffer.from(HASH, 'hex'), high));
    expect(scans[0].w).toMatchObject({ midstate: mid, tail: Buffer.concat([Buffer.alloc(16), LANE]), start: qc.u64Bytes(0n), count: 256 });
    expect(scans[0].w.line).toBeUndefined();
    clock += 50;
    scans.shift().resolve({ hits: qc.u64Bytes(3n), overflow: false });
    await flush();
    expect(sock.lines[1].params.nonce).toBe(q.buildNonceV1(EXTRANONCE, LANE, 3n).toString('hex'));
    // The next batch follows the last counter, under the same midstate.
    expect(scans[0].w.start).toEqual(qc.u64Bytes(256n));
    expect(scans[0].w.midstate).toEqual(mid);
    engine.stop();
  });
});
