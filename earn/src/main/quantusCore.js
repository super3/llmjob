'use strict';

const path = require('path');
const crypto = require('crypto');
const { createRequire } = require('module');
const q = require('../shared/quantus/qpow');
const { INTEL_DIR, CU13, pickCoreVariant, isRuntimeError } = require('../shared/coreVariant');

// The Quantus core interface, version 2. Every vendor core (NVIDIA, AMD,
// Intel) is an N-API addon with exactly this surface, and the CPU core below is
// the same thing in JavaScript: slow, but right, so the whole miner runs end to
// end without a GPU. earn/native/quantus/README.md says the same for the
// people writing kernels.
//
//   addon.QUANTUS_API             2
//   addon.devices()               [{ index, name }] the cards it can open, in its own numbering
//   addon.createCore({ device })  a core on that card; throws an Error that says why not
//
//   core.device                   { index, name, vendor }   vendor: 'nvidia' | 'amd' | 'intel' | 'cpu'
//   core.scan(work)               Promise<{ hits: Buffer, overflow: boolean }>
//   core.close()                  frees the card; a scan after it rejects
//
//   work = {
//     line:       Buffer(112) 14 field elements, u64 little-endian, each below p: one line of
//                             nonces (qpow.lineWork): C[0..12], then X0 and X4
//     start:      Buffer(8)   the first point t on the line, u64 big-endian
//     count:      number      points start .. start + count - 1; 1 to 2^32, never past 2^32 - 1
//     targetHigh: Buffer(8)   u64 big-endian
//     maxHits:    number      at most this many hits back
//   }
//
// A hit is a point t whose hash's top 64 bits (qpow.indexTop64) are at or below
// targetHigh. `hits` is 8 bytes a hit, each t as a u64 big-endian, in any
// order; `overflow` is true when more were found than maxHits. The host
// recomputes every hit's full hash on the CPU before it sends anything, so a
// tie with targetHigh is the host's to settle, and a core that miscounts is
// caught. The host runs one scan at a time on a core.
//
// Version 1 gave a core the midstate, nonce bytes 32..55 and a counter in
// bytes 56..63. Version 2 hands it a line instead (qpow.js says why: most of
// the first round is then the same for every nonce). The two cannot be mixed:
// a core of one version computes nothing useful from the other's work. So the
// loader below refuses any file whose QUANTUS_API is not this number, and a
// version 2 core rejects work that has no `line`.

const API = 2;
const CPU_VENDOR = 'cpu';

// The vendor addons, tried in this order. The file names are fixed here so the
// three can be built and shipped separately.
const FILES = ['quantus_cuda.node', 'quantus_hip.node', 'quantus_sycl.node'];
// Each card maker's file. A rig whose cards are known tries its own first.
const VENDOR_FILES = { nvidia: 'quantus_cuda.node', amd: 'quantus_hip.node', intel: 'quantus_sycl.node' };
const DEV_DIR = path.join(__dirname, '..', '..', 'native', 'quantus', 'build');
// The Intel core needs the SYCL runtime, and on Windows a DLL is found only
// beside the .node that loads it. So packages put quantus_sycl.node in the
// Pearl Intel core's folder (sycl/), with that runtime, on Linux and Windows
// alike. A Linux build also loads from the folder above (its run path names
// $ORIGIN/sycl), so that place is tried next.
const SUBDIRS = { 'quantus_sycl.node': [INTEL_DIR, ''] };
// The NVIDIA core built with CUDA 13, for RTX 50 (sm_120) only. The same source
// runs 3-8% more a clock built with ptxas 13.3 than with 12.8 on an RTX 5070
// (earn/tuning/quantus-benchmark.md). Its runtime is linked in and needs driver
// 580 or newer, so it is tried first only where the Pearl core's rule
// (shared/coreVariant.pickCoreVariant) would load pearl_core_cu13.node: driver
// 580+ and every mining card compute 12.x. Anywhere else, and whenever it does
// not load, sees no card or cannot start, quantus_cuda.node (CUDA 12.8) mines.
// QUANTUS_CORE_VARIANT=cu12 or cu13 overrides the rule.
const CU13_FILE = 'quantus_cuda_cu13.node';

// Whether to try the CUDA 13 build first, and why: { cu13, reason }, or null
// when nothing says (no nvidia-smi list and no override), which keeps the
// search this file always had.
function cudaBuild(env, opts) {
  const raw = String(env.QUANTUS_CORE_VARIANT || '').trim().toLowerCase();
  if (raw === 'cu12' || raw === CU13) return { cu13: raw === CU13, reason: 'QUANTUS_CORE_VARIANT=' + raw };
  if (!Array.isArray(opts.cards) || !opts.cards.length) return null;
  const pick = pickCoreVariant({ env: {}, cards: opts.cards, gpus: opts.gpus });
  return { cu13: pick.variant === CU13, reason: pick.reason };
}

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
  const line = q.lineFromBytes(w.line);
  const start = u64(w.start, 'start');
  const targetHigh = u64(w.targetHigh, 'targetHigh');
  const count = w.count;
  if (!Number.isInteger(count) || count < 1 || count > 2 ** 32) throw new Error('count must be 1 to 2^32');
  if (start + BigInt(count) > 1n << 32n) throw new Error('start + count passes 2^32');
  const maxHits = Number.isInteger(w.maxHits) && w.maxHits >= 0 ? w.maxHits : 0;
  return { line, start: Number(start), count, targetHigh, maxHits };
}

// The reference scan, synchronous: the points in [from, to) that hit.
function scanRange(w, from, to, found) {
  for (let t = from; t < to; t++) {
    if (q.indexTop64(w.line, t) <= w.targetHigh) found.push(BigInt(t));
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
    const end = w.start + w.count;
    for (let t = w.start; t < end; t += CPU_CHUNK) {
      scanRange(w, t, Math.min(t + CPU_CHUNK, end), found);
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

// The CUDA 13 core, with the 12.8 one behind it. A static CUDA 13 runtime can
// load and still fail at its first real CUDA call on a driver it cannot use,
// and that shows up in createCore. So until a core has started, an error that
// says the runtime cannot run here (shared/coreVariant.isRuntimeError, the
// Pearl loader's test) moves this factory to quantus_cuda.node and tries the
// card again. Once a CUDA 13 core has started, the runtime works here, and a
// later card's error is that card's own.
function cu13Fallback(got, { make, log, load, places }) {
  const f = make(got.addon, got.file, got.devices, path.basename(got.file, '.node'));
  let addon = got.addon;
  let settled = false;
  f.createCore = (index) => {
    if (settled) return addon.createCore({ device: index });
    try {
      const core = addon.createCore({ device: index });
      settled = true;
      return core;
    } catch (e) {
      if (!isRuntimeError(e)) throw e;
      let cu12 = null;
      places(VENDOR_FILES.nvidia).some((file) => (cu12 = load(file, [])));
      if (!cu12) throw e;
      log('warn', 'Quantus core: the CUDA 13 build could not start (' + e.message + '); ' + cu12.file + ' (CUDA 12.8) takes over');
      addon = cu12.addon;
      settled = true;
      Object.assign(f, { file: cu12.file, label: path.basename(cu12.file, '.node'), devices: cu12.devices });
      return addon.createCore({ device: index });
    }
  };
  return f;
}

// The core this rig mines Quantus with, as { label, file, devices, createCore(index) },
// or null when there is none. QUANTUS_CORE=cpu picks the CPU core;
// QUANTUS_CORE_PATH names one file and only it is tried. Otherwise each vendor
// file is tried beside the app (the Intel one in sycl/ there first), then in
// the dev tree (earn/native/quantus/build), and the first that loads and sees
// a card wins. `opts.vendor` ('nvidia', 'amd' or 'intel'), when the shell
// knows whose cards the rig has, puts that maker's file first: the app mines
// an AMD rig with quantus_hip even if a CUDA runtime is installed beside it.
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

  const first = VENDOR_FILES[opts.vendor];
  const files = first ? [first].concat(FILES.filter((f) => f !== first)) : FILES;
  // `opts.cards` is nvidia-smi's list (probe.detectCudaCards) and `opts.gpus`
  // the cards that will mine. An AMD or Intel rig never tries a CUDA build first.
  const build = !first || first === VENDOR_FILES.nvidia ? cudaBuild(env, opts) : null;
  const cu13 = Boolean(build && build.cu13);
  const places = (f) => dirs.flatMap((d) => (SUBDIRS[f] || ['']).map((s) => path.join(d, s, f)))
    .concat([path.join(DEV_DIR, f)]);
  const candidates = env.QUANTUS_CORE_PATH
    ? [env.QUANTUS_CORE_PATH]
    : (cu13 ? [CU13_FILE] : []).concat(files).flatMap(places);
  const load = (file, why) => {
    let addon;
    try {
      addon = req(file);
    } catch (e) {
      if (!(e && e.code === 'MODULE_NOT_FOUND')) why.push(file + ' would not load: ' + ((e && e.message) || e));
      return null;
    }
    const devices = usable(addon, file, why);
    return devices ? { addon, file, devices } : null;
  };
  const why = [];
  for (const file of candidates) {
    const got = load(file, why);
    if (!got) continue;
    const name = path.basename(file);
    const cards = got.devices.length + ' card' + (got.devices.length === 1 ? '' : 's');
    if (!build || env.QUANTUS_CORE_PATH || (name !== CU13_FILE && name !== VENDOR_FILES.nvidia)) {
      log('info', 'Quantus core: ' + file + ' · ' + cards);
      return make(got.addon, file, got.devices, path.basename(file, '.node'));
    }
    if (name === VENDOR_FILES.nvidia) {
      const missed = why.filter((w) => w.includes(CU13_FILE));
      if (cu13) {
        log('warn', 'Quantus core: ' + file + ' · ' + cards + ' · CUDA 12.8 build: the CUDA 13 one was wanted ('
          + build.reason + '), but ' + (missed.length ? missed.join('; ') : CU13_FILE + ' is not installed'));
      } else log('info', 'Quantus core: ' + file + ' · ' + cards + ' · CUDA 12.8 build (' + build.reason + ')');
      return make(got.addon, file, got.devices, path.basename(file, '.node'));
    }
    log('info', 'Quantus core: ' + file + ' · ' + cards + ' · CUDA 13 build (' + build.reason + ')');
    return cu13Fallback(got, { make, log, load, places });
  }
  for (const w of why) log('warn', 'Quantus core: ' + w);
  log('error', 'Quantus core: none found (' + (env.QUANTUS_CORE_PATH ? 'QUANTUS_CORE_PATH=' + env.QUANTUS_CORE_PATH
    : 'looked for ' + FILES.join(', ') + ' beside the app, quantus_sycl.node in ' + INTEL_DIR
      + '/ there too, and in earn/native/quantus/build') + ')');
  return null;
}

// ── Checking a core against the reference ───────────────────────────────────
//
// What a vendor core has to pass before it mines (earn/native/quantus/
// verify-core.js runs it on a card). Each check is { name, ok, detail }.
//
//   vectors   the five official nonce hashes (quantus-miner pow-core
//             NONCE_HASH_KVS), each as point 0 of the line that starts at it:
//             each must hit at its own top 64 bits, and miss one below.
//   exact     a stretch of one of the miner's own lines, across the point
//             where a core starts a new launch (2^26), at a target 1 in 16
//             nonces meet: the hits must be exactly the reference's, and each
//             hit's nonce, built as the miner builds it, must be a hit on the
//             full hash.
//   top       the same at the very end of a line (the last of its SPAN points).
//   overflow  a target every nonce meets: maxHits hits back and overflow set.
//   big       a large number of nonces over several lines at a target about
//             32 nonces meet: every hit must check out on the reference, and
//             the count must be near 32. Its speed is the core's hashrate.
const VECTORS = [
  ['00'.repeat(32), '00'.repeat(64), '8e64e3d8e0f38f88'],
  ['01'.repeat(32), '00'.repeat(63) + '01', 'e4ef79db80b642a0'],
  ['000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f', '00'.repeat(56) + '1234567890abcdef', '7f6c410ce62e7fc5'],
  ['ff'.repeat(32), '00'.repeat(56) + 'ffffffffffffffff', '822f70999ee517f6'],
  ['07'.repeat(32), '00000000000000000000000000000000000000000deadbeefcafe000000000000000000000000000000000000000000000000000000000001234567890abcdef', 'eab81cdb33db293a'],
];

function workFor(line, start, count, targetHigh, maxHits) {
  return { line: q.lineBytes(line), start: u64Bytes(start), count, targetHigh: u64Bytes(targetHigh), maxHits };
}

function hitList(res) {
  const out = [];
  for (let i = 0; i + 8 <= res.hits.length; i += 8) out.push(res.hits.readBigUInt64BE(i));
  return out.sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
}

// Points one scan of the big check takes: a power of two below a line's SPAN.
const BIG_SCAN = 2 ** 27;

async function verifyCore(core, opts = {}) {
  const exactCount = opts.exactCount || 4096;
  const bigCount = opts.bigCount || 2 ** 30;
  const random = opts.random || crypto.randomBytes;
  const now = opts.now || Date.now;
  const checks = [];
  const record = (name, ok, detail) => checks.push({ name, ok, detail });

  // vectors
  const bad = [];
  for (const [i, [h, n, top]] of VECTORS.entries()) {
    const { high, base } = q.splitNonce(Buffer.from(n, 'hex'));
    const line = q.lineWork(q.midstate(Buffer.from(h, 'hex'), high), base);
    const t = BigInt('0x' + top);
    const at = hitList(await core.scan(workFor(line, 0, 1, t, 4)));
    const below = hitList(await core.scan(workFor(line, 0, 1, t - 1n, 4)));
    if (at.length !== 1 || at[0] !== 0n || below.length !== 0) bad.push(i);
  }
  record('vectors', !bad.length, bad.length ? 'wrong on vector ' + bad.join(', ') : '5 of 5');

  // exact, top: on one of the miner's own lines.
  const header = random(32);
  const extranonce = random(4);
  const lane = random(q.LANE_BYTES);
  const first = BigInt('0x' + random(8).toString('hex')) % (q.LINES - BigInt(Math.ceil(bigCount / BIG_SCAN)));
  const mid = q.midstate(header, q.splitNonce(q.buildNonce(extranonce, lane, first, 0)).high);
  const lineOf = (n) => q.lineWork(mid, q.lineBase(n));
  const target = 1n << 60n;
  const line = lineOf(first);
  for (const [name, start] of [['exact', 2 ** 26 - (exactCount >> 1)], ['top', q.SPAN - exactCount]]) {
    const want = [];
    scanRange({ line, targetHigh: target }, start, start + exactCount, want);
    const got = hitList(await core.scan(workFor(line, start, exactCount, target, exactCount)));
    // Each hit also as the miner sends it: point t's nonce, hashed in full.
    const full = (t) => q.toBig(q.nonceHash(header, q.buildNonce(extranonce, lane, first, Number(t)))) >> 448n <= target;
    const same = got.length === want.length && got.every((t, i) => t === want[i] && full(t));
    record(name, same, got.length + ' hits, the reference has ' + want.length + (same ? '' : ': they differ'));
  }

  // overflow
  const over = await core.scan(workFor(line, 0, 64, q.U64_MAX, 16));
  record('overflow', over.overflow === true && over.hits.length === 16 * 8,
    (over.hits.length / 8) + ' hits back, overflow ' + over.overflow);

  // big
  const bigTarget = (32n << 64n) / BigInt(bigCount);
  const t0 = now();
  let hits = 0;
  let wrong = 0;
  let overflow = false;
  for (let done = 0, n = first; done < bigCount; done += BIG_SCAN, n++) {
    const l = lineOf(n);
    const res = await core.scan(workFor(l, 0, Math.min(BIG_SCAN, bigCount - done), bigTarget, 1024));
    const found = hitList(res);
    hits += found.length;
    wrong += found.filter((t) => q.indexTop64(l, Number(t)) > bigTarget).length;
    overflow = overflow || res.overflow;
  }
  const secs = Math.max(now() - t0, 1) / 1000;
  // 32 expected: fewer than 8 or more than 80 is about a 1 in 10^5 chance.
  const ok = !wrong && !overflow && hits >= 8 && hits <= 80;
  record('big', ok, hits + ' hits (about 32 expected), ' + wrong + ' wrong · '
    + bigCount + ' nonces in ' + secs.toFixed(2) + ' s');

  return { ok: checks.every((c) => c.ok), checks, hashrate: bigCount / secs };
}

module.exports = {
  API, FILES, VENDOR_FILES, SUBDIRS, CPU_CHUNK, BIG_SCAN, CpuCore, cpuAddon, readWork, scanRange, packHits, quantusCoreFactory, verifyCore, u64Bytes, hitList,
};
