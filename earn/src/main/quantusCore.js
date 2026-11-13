'use strict';

const path = require('path');
const crypto = require('crypto');
const { createRequire } = require('module');
const q = require('../shared/quantus/qpow');

// The Quantus core interface, version 1. Every vendor core (NVIDIA, AMD,
// Intel) is an N-API addon with exactly this surface, and the CPU core below is
// the same thing in JavaScript: slow, but right, so the whole miner runs end to
// end without a GPU. earn/native/quantus/README.md says the same for the
// people writing kernels.
//
//   addon.QUANTUS_API             1
//   addon.devices()               [{ index, name }] the cards it can open, in its own numbering
//   addon.createCore({ device })  a core on that card; throws an Error that says why not
//
//   core.device                   { index, name, vendor }   vendor: 'nvidia' | 'amd' | 'intel' | 'cpu'
//   core.scan(work)               Promise<{ hits: Buffer, overflow: boolean }>
//   core.close()                  frees the card; a scan after it rejects
//
//   work = {
//     midstate:   Buffer(96)  12 field elements, u64 little-endian, each below p:
//                             the sponge after the header and nonce bytes 0..31
//     tail:       Buffer(24)  nonce bytes 32..55
//     start:      Buffer(8)   the first counter (nonce bytes 56..63), u64 big-endian
//     count:      number      counters start .. start + count - 1; 1 to 2^32, never past 2^64 - 1
//     targetHigh: Buffer(8)   u64 big-endian
//     maxHits:    number      at most this many hits back
//   }
//
// A hit is a counter whose hash's top 64 bits (qpow.nonceTop64) are at or below
// targetHigh. `hits` is 8 bytes a hit, each a u64 big-endian counter, in any
// order; `overflow` is true when more were found than maxHits. The host
// recomputes every hit's full hash on the CPU before it sends anything, so a
// tie with targetHigh is the host's to settle, and a core that miscounts is
// caught. The host runs one scan at a time on a core.

const API = 1;
const CPU_VENDOR = 'cpu';

// The vendor addons, tried in this order. The file names are fixed here so the
// three can be built and shipped separately.
const FILES = ['quantus_cuda.node', 'quantus_hip.node', 'quantus_sycl.node'];
const DEV_DIR = path.join(__dirname, '..', '..', 'native', 'quantus', 'build');

// Nonces the CPU core hashes before it lets the event loop run: about 15 ms.
const CPU_CHUNK = 64;

function u64(buf, what) {
  if (!Buffer.isBuffer(buf) || buf.length !== 8) throw new Error(what + ' must be 8 bytes');
  return buf.readBigUInt64BE(0);
}

function u64Bytes(n) {
  const b = Buffer.alloc(8);
  b.writeBigUInt64BE(BigInt(n));
  return b;
}

// What a scan's work means, checked: the reference for every core.
function readWork(work) {
  const w = work || {};
  const mid = q.midstateFromBytes(w.midstate);
  if (!Buffer.isBuffer(w.tail) || w.tail.length !== q.TAIL_BYTES) throw new Error('tail must be 24 bytes');
  const start = u64(w.start, 'start');
  const targetHigh = u64(w.targetHigh, 'targetHigh');
  const count = w.count;
  if (!Number.isInteger(count) || count < 1 || count > 2 ** 32) throw new Error('count must be 1 to 2^32');
  if (start + BigInt(count) - 1n > q.U64_MAX) throw new Error('start + count passes 2^64');
  const maxHits = Number.isInteger(w.maxHits) && w.maxHits >= 0 ? w.maxHits : 0;
  return { mid, tail: w.tail, start, count, targetHigh, maxHits };
}

// The reference scan, synchronous: the counters in [from, to) that hit.
function scanRange(w, from, to, found) {
  for (let c = from; c < to; c++) {
    if (q.nonceTop64(w.mid, w.tail, c) <= w.targetHigh) found.push(c);
  }
}

function packHits(found, maxHits) {
  const kept = found.slice(0, maxHits);
  return { hits: Buffer.concat(kept.map(u64Bytes)), overflow: found.length > maxHits };
}

// The CPU core: the interface above, in JavaScript, on the reference hash.
// About 4,000 nonces a second on one thread, so it is for tests and for
// running the miner end to end, not for mining: a GPU does about a GH/s.
class CpuCore {
  constructor({ device } = {}) {
    this.device = { index: Number.isInteger(device) ? device : 0, name: 'CPU (reference)', vendor: CPU_VENDOR };
    this.closed = false;
  }

  async scan(work) {
    if (this.closed) throw new Error('the core is closed');
    const w = readWork(work);
    const found = [];
    const end = w.start + BigInt(w.count);
    for (let c = w.start; c < end; c += BigInt(CPU_CHUNK)) {
      const to = c + BigInt(CPU_CHUNK) < end ? c + BigInt(CPU_CHUNK) : end;
      scanRange(w, c, to, found);
      // Let the pool socket and timers run between chunks.
      await new Promise((resolve) => setImmediate(resolve));
      if (this.closed) throw new Error('the core is closed');
    }
    return packHits(found, w.maxHits);
  }

  close() {
    this.closed = true;
  }
}

const cpuAddon = {
  QUANTUS_API: API,
  devices: () => [{ index: 0, name: 'CPU (reference)' }],
  createCore: (opts) => new CpuCore(opts),
};

// Whether a loaded file is a Quantus core of this version that sees a card.
// `why` collects what was wrong with each file that loaded but was refused.
function usable(addon, file, why) {
  if (!addon || typeof addon.createCore !== 'function' || typeof addon.devices !== 'function') {
    why.push(file + ' is not a Quantus core');
    return null;
  }
  if (addon.QUANTUS_API !== API) {
    why.push(file + ' speaks Quantus core API ' + addon.QUANTUS_API + ', this miner ' + API);
    return null;
  }
  let devices;
  try {
    devices = addon.devices();
  } catch (e) {
    why.push(file + ' could not list its cards: ' + ((e && e.message) || e));
    return null;
  }
  if (!Array.isArray(devices) || !devices.length) {
    why.push(file + ' found no card');
    return null;
  }
  return devices;
}

// The core this rig mines Quantus with, as { label, file, devices, createCore(index) },
// or null when there is none. QUANTUS_CORE=cpu picks the CPU core;
// QUANTUS_CORE_PATH names one file and only it is tried. Otherwise each vendor
// file is tried beside the app, then in the dev tree (earn/native/quantus/build),
// and the first that loads and sees a card wins.
function quantusCoreFactory(opts = {}) {
  const env = opts.env || process.env;
  const log = typeof opts.log === 'function' ? opts.log : () => {};
  const make = (addon, file, devices, label) => ({
    label, file, devices, createCore: (index) => addon.createCore({ device: index }),
  });

  if (String(env.QUANTUS_CORE || '').trim().toLowerCase() === 'cpu') {
    log('info', 'Quantus core: the CPU reference (QUANTUS_CORE=cpu). It hashes about 4,000 nonces a second: for tests, not for mining.');
    return make(cpuAddon, null, cpuAddon.devices(), 'cpu');
  }

  const execPath = opts.execPath || process.execPath;
  const req = opts.require || createRequire(execPath);
  const execDir = path.dirname(execPath);
  const dirs = [];
  if (opts.resourcesPath) dirs.push(path.join(opts.resourcesPath, 'native'));
  dirs.push(execDir, path.join(execDir, 'native'));

  const candidates = env.QUANTUS_CORE_PATH
    ? [env.QUANTUS_CORE_PATH]
    : FILES.flatMap((f) => dirs.map((d) => path.join(d, f)).concat([path.join(DEV_DIR, f)]));
  const why = [];
  for (const file of candidates) {
    let addon;
    try {
      addon = req(file);
    } catch (e) {
      if (!(e && e.code === 'MODULE_NOT_FOUND')) why.push(file + ' would not load: ' + ((e && e.message) || e));
      continue;
    }
    const devices = usable(addon, file, why);
    if (!devices) continue;
    log('info', 'Quantus core: ' + file + ' · ' + devices.length + ' card' + (devices.length === 1 ? '' : 's'));
    return make(addon, file, devices, path.basename(file, '.node'));
  }
  for (const w of why) log('warn', 'Quantus core: ' + w);
  log('error', 'Quantus core: none found (' + (env.QUANTUS_CORE_PATH ? 'QUANTUS_CORE_PATH=' + env.QUANTUS_CORE_PATH
    : 'looked for ' + FILES.join(', ') + ' beside the app and in earn/native/quantus/build') + ')');
  return null;
}

// ── Checking a core against the reference ───────────────────────────────────
//
// What a vendor core has to pass before it mines (earn/native/quantus/
// verify-core.js runs it on a card). Each check is { name, ok, detail }.
//
//   vectors   the five official nonce hashes (quantus-miner pow-core
//             NONCE_HASH_KVS): each must hit at its own top 64 bits, and miss
//             one below.
//   exact     a range across a 32-bit carry of the counter, at a target 1 in
//             16 nonces meet: the hits must be exactly the reference's.
//   top       the same at the very end of the counter's range.
//   overflow  a target every nonce meets: maxHits hits back and overflow set.
//   big       a large range at a target about 32 nonces meet: every hit must
//             check out on the reference, and the count must be near 32. Its
//             speed is the core's hashrate.
const VECTORS = [
  ['00'.repeat(32), '00'.repeat(64), '8e64e3d8e0f38f88'],
  ['01'.repeat(32), '00'.repeat(63) + '01', 'e4ef79db80b642a0'],
  ['000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f', '00'.repeat(56) + '1234567890abcdef', '7f6c410ce62e7fc5'],
  ['ff'.repeat(32), '00'.repeat(56) + 'ffffffffffffffff', '822f70999ee517f6'],
  ['07'.repeat(32), '00000000000000000000000000000000000000000deadbeefcafe000000000000000000000000000000000000000000000000000000000001234567890abcdef', 'eab81cdb33db293a'],
];

function workFor(header, nonceHigh, tail, start, count, targetHigh, maxHits) {
  return {
    midstate: q.midstateBytes(q.midstate(header, nonceHigh)),
    tail, start: u64Bytes(start), count, targetHigh: u64Bytes(targetHigh), maxHits,
  };
}

function hitList(res) {
  const out = [];
  for (let i = 0; i + 8 <= res.hits.length; i += 8) out.push(res.hits.readBigUInt64BE(i));
  return out.sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
}

async function verifyCore(core, opts = {}) {
  const exactCount = opts.exactCount || 4096;
  const bigCount = opts.bigCount || 2 ** 30;
  const random = opts.random || crypto.randomBytes;
  const now = opts.now || Date.now;
  const checks = [];
  const record = (name, ok, detail) => checks.push({ name, ok, detail });

  // vectors
  let bad = [];
  for (const [i, [h, n, top]] of VECTORS.entries()) {
    const { high, tail, counter } = q.splitNonce(Buffer.from(n, 'hex'));
    const t = BigInt('0x' + top);
    const at = hitList(await core.scan(workFor(Buffer.from(h, 'hex'), high, tail, counter, 1, t, 4)));
    const below = hitList(await core.scan(workFor(Buffer.from(h, 'hex'), high, tail, counter, 1, t - 1n, 4)));
    if (at.length !== 1 || at[0] !== counter || below.length !== 0) bad.push(i);
  }
  record('vectors', !bad.length, bad.length ? 'wrong on vector ' + bad.join(', ') : '5 of 5');

  // exact, top
  const header = random(32);
  const high = random(32);
  const tail = random(24);
  const target = 1n << 60n;
  for (const [name, start] of [['exact', (1n << 32n) - BigInt(exactCount >> 1)], ['top', q.U64_MAX - BigInt(exactCount) + 1n]]) {
    const want = [];
    const mid = q.midstate(header, high);
    scanRange({ mid, tail, targetHigh: target }, start, start + BigInt(exactCount), want);
    const got = hitList(await core.scan(workFor(header, high, tail, start, exactCount, target, exactCount)));
    const same = got.length === want.length && got.every((c, i) => c === want[i]);
    record(name, same, got.length + ' hits, the reference has ' + want.length + (same ? '' : ': they differ'));
  }

  // overflow
  const over = await core.scan(workFor(header, high, tail, 0n, 64, q.U64_MAX, 16));
  record('overflow', over.overflow === true && over.hits.length === 16 * 8,
    (over.hits.length / 8) + ' hits back, overflow ' + over.overflow);

  // big
  const bigTarget = (32n << 64n) / BigInt(bigCount);
  const mid = q.midstate(header, high);
  const t0 = now();
  const res = await core.scan(workFor(header, high, tail, 0n, bigCount, bigTarget, 1024));
  const secs = Math.max(now() - t0, 1) / 1000;
  const hits = hitList(res);
  const wrong = hits.filter((c) => q.nonceTop64(mid, tail, c) > bigTarget);
  // 32 expected: fewer than 8 or more than 80 is about a 1 in 10^5 chance.
  const ok = !wrong.length && !res.overflow && hits.length >= 8 && hits.length <= 80;
  record('big', ok, hits.length + ' hits (about 32 expected), ' + wrong.length + ' wrong · '
    + bigCount + ' nonces in ' + secs.toFixed(2) + ' s');

  return { ok: checks.every((c) => c.ok), checks, hashrate: bigCount / secs };
}

module.exports = {
  API, FILES, CPU_CHUNK, CpuCore, cpuAddon, readWork, scanRange, packHits, quantusCoreFactory, verifyCore, u64Bytes, hitList,
};
