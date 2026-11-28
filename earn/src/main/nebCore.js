'use strict';

const path = require('path');
const crypto = require('crypto');
const { createRequire } = require('module');
const blake3 = require('../shared/miner/blake3');
const nh = require('../shared/neb/nebhash2');
const hdr = require('../shared/neb/header');

// The NEB core interface, version 1. Each vendor core is an N-API addon with
// this surface (earn/native/neb), and the CPU core below is the same thing in
// JavaScript on the reference: slow, but right.
//
//   addon.NEB_API                 1
//   addon.devices()               [{ index, name }]
//   addon.createCore({ device })  a core on that card; throws an Error that says why not
//
//   core.device                   { index, name, vendor }
//   core.setDataset(slot, arch, values, labels)
//   core.setModel(slot, weights)  Promise<{ samples, live, maxLive, entries, tableBytes, cpuMs, gpuMs,
//                                 freeBytes, totalBytes, tripleBytes }>
//   core.buildTriples(slot, rows, maxBytes)   Promise<{ rows, entries, tableBytes, ms }>
//   core.scan(work)               Promise<{ hits: Buffer, overflow, evaluated }>
//   core.evalKeys(slot, keys)     Promise<Buffer>: digest and ticket (64 bytes) for each 32-byte key
//   core.close()
//
//   work = { slot, header: Buffer(80), target: Buffer(32), start, count, maxHits }
//
// A core does not evaluate every nonce. A sample whose output equals its
// target has a zero gradient, and on a trained model most samples are like
// that. The core looks at each nonce's 16 samples and evaluates the attempt
// only when at most `maxLive` of them are live (2, from a table of row pairs
// it builds for each model; 1 when that table does not fit the card).
// buildTriples adds a table of row triples over the model's first `rows` live
// samples, as many as fit maxBytes and the card: an attempt with exactly 3
// live samples that are all among them is then evaluated too. The table is
// large (rows^3 / 6 entries of 32 bytes) and takes seconds to build, so the
// host decides whether a model is worth one. setModel drops it. The
// rest are skipped, which costs no luck: a nonce not tried is not a lost
// ticket. `evaluated` counts the attempts evaluated, and that is the rate that
// earns. A hit is a nonce (4 bytes, little-endian) whose attempt was evaluated
// and whose ticket is at or under the target. The host evaluates every hit
// again on the reference before it sends it.

const API = 1;
const FILES = ['neb_cuda.node', 'neb_hip.node'];
const VENDOR_FILES = { nvidia: 'neb_cuda.node', amd: 'neb_hip.node' };
const DEV_DIR = path.join(__dirname, '..', '..', 'native', 'neb', 'build');
const CPU_MAX_LIVE = 2;
// Nonces the CPU core tries before it lets the event loop run.
const CPU_CHUNK = 256;
const ALL_ONES = Buffer.alloc(32, 0xff);

// Weights as signed bytes, from an Int8Array or a Buffer of them.
function int8(weights) {
  return Int8Array.from(new Int8Array(weights.buffer, weights.byteOffset, weights.length));
}

function readWork(work) {
  const w = work || {};
  if (!Buffer.isBuffer(w.header) || w.header.length !== hdr.HEADER_BYTES) throw new Error('header must be 80 bytes');
  if (!Buffer.isBuffer(w.target) || w.target.length !== 32) throw new Error('target must be 32 bytes');
  if (!Number.isInteger(w.start) || w.start < 0 || w.start > 0xffffffff) throw new Error('start must be a 32-bit nonce');
  if (!Number.isInteger(w.count) || w.count < 1 || w.start + w.count > 2 ** 32) throw new Error('start + count passes 2^32');
  const maxHits = Number.isInteger(w.maxHits) && w.maxHits >= 0 ? w.maxHits : 0;
  return { slot: w.slot, header: w.header, target: w.target, start: w.start, count: w.count, maxHits };
}

// What a core's tables cover on a model, for the reference: `rowOf` is each
// sample's row (0 when dead, else 1 and up in dataset order), maxLive the live
// samples an attempt may have, tripleRows the rows the triple table covers.
function coverage(live, maxLive, tripleRows) {
  const rowOf = new Uint16Array(live.length);
  let rows = 0;
  for (let i = 0; i < live.length; i++) if (live[i]) rowOf[i] = ++rows;
  return { rowOf, rows, maxLive, tripleRows };
}

// Whether a core with this coverage evaluates the attempt of `key`.
function evaluates(cover, key) {
  const rows = nh.shareIndices(key, cover.rowOf.length).map((i) => cover.rowOf[i]).filter((r) => r > 0);
  if (rows.length <= cover.maxLive) return true;
  return rows.length === 3 && rows.every((r) => r <= cover.tripleRows);
}

// The share of attempts a core with this coverage evaluates.
function coveredShare(cover) {
  const p = cover.rows / cover.rowOf.length;
  const triples = cover.rows ? nh.exactShare(p, 3) * (cover.tripleRows / cover.rows) ** 3 : 0;
  return nh.candidateShare(p, cover.maxLive) + triples;
}

// The reference scan: of the nonces in [from, to), the ones a core would
// evaluate, and of those the hits.
function scanRange(w, model, from, to, out) {
  for (let nonce = from; nonce < to; nonce++) {
    const key = hdr.attemptKey(w.header, nonce);
    if (!evaluates(model.cover, key)) continue;
    out.evaluated++;
    if (nh.meetsTarget(nh.shareEval(model.m, model.ds, key).ticket, w.target)) out.hits.push(nonce);
  }
}

function packHits(found, maxHits, evaluated) {
  const kept = found.slice(0, maxHits);
  const hits = Buffer.alloc(kept.length * 4);
  kept.forEach((n, i) => hits.writeUInt32LE(n, 4 * i));
  return { hits, overflow: found.length > maxHits, evaluated };
}

function hitList(res) {
  const out = [];
  for (let i = 0; i + 4 <= res.hits.length; i += 4) out.push(res.hits.readUInt32LE(i));
  return out.sort((a, b) => a - b);
}

// The CPU core. A few hundred nonces a second: for tests, not for mining.
class CpuCore {
  constructor({ device } = {}) {
    this.device = { index: Number.isInteger(device) ? device : 0, name: 'CPU (reference)', vendor: 'cpu' };
    this.closed = false;
    this.slots = [];
  }

  _slot(slot, needModel) {
    if (this.closed) throw new Error('the core is closed');
    const s = this.slots[slot];
    if (!s) throw new Error('the slot has no dataset');
    if (needModel && !s.m) throw new Error('the slot has no model');
    return s;
  }

  setDataset(slot, arch, values, labels) {
    if (this.closed) throw new Error('the core is closed');
    this.slots[slot] = { ds: { arch, n: labels.length, values, labels }, m: null, cover: null };
  }

  async setModel(slot, weights) {
    const s = this._slot(slot, false);
    s.m = { arch: s.ds.arch, w: int8(weights) };
    s.cover = coverage(nh.liveSamples(s.m, s.ds), CPU_MAX_LIVE, 0);
    return {
      samples: s.ds.n, live: s.cover.rows, maxLive: CPU_MAX_LIVE, entries: 0, tableBytes: 0, cpuMs: 0, gpuMs: 0,
      freeBytes: 0, totalBytes: 0, tripleBytes: 0,
    };
  }

  // No table to build: the CPU core evaluates in full. It covers the rows asked for.
  async buildTriples(slot, rows) {
    const s = this._slot(slot, true);
    s.cover.tripleRows = Math.min(rows, s.cover.rows);
    return { rows: s.cover.tripleRows, entries: 0, tableBytes: 0, ms: 0 };
  }

  async scan(work) {
    const w = readWork(work);
    const s = this._slot(w.slot, true);
    const out = { hits: [], evaluated: 0 };
    const end = w.start + w.count;
    for (let t = w.start; t < end; t += CPU_CHUNK) {
      scanRange(w, s, t, Math.min(t + CPU_CHUNK, end), out);
      await new Promise((resolve) => setImmediate(resolve));
      if (this.closed) throw new Error('the core is closed');
    }
    return packHits(out.hits, w.maxHits, out.evaluated);
  }

  async evalKeys(slot, keys) {
    const s = this._slot(slot, true);
    const out = [];
    for (let i = 0; i + 32 <= keys.length; i += 32) {
      const r = nh.shareEval(s.m, s.ds, keys.subarray(i, i + 32));
      out.push(r.digest, r.ticket);
    }
    return Buffer.concat(out);
  }

  close() {
    this.closed = true;
  }
}

const cpuAddon = {
  NEB_API: API,
  devices: () => [{ index: 0, name: 'CPU (reference)' }],
  createCore: (opts) => new CpuCore(opts),
};

// The core this rig mines NEB with, as { label, file, devices, createCore(index) },
// or null when there is none. NEB_CORE=cpu picks the CPU core; NEB_CORE_PATH
// names one file and only it is tried. Otherwise each vendor file is tried
// beside the app and then in the dev tree (earn/native/neb/build), the rig's
// own maker first when `opts.vendor` says who that is.
function nebCoreFactory(opts = {}) {
  const env = opts.env || process.env;
  const log = typeof opts.log === 'function' ? opts.log : () => {};
  const make = (addon, file, devices, label) => ({
    label, file, devices, createCore: (index) => addon.createCore({ device: index }),
  });
  if (String(env.NEB_CORE || '').trim().toLowerCase() === 'cpu') {
    log('info', 'NEB core: the CPU reference (NEB_CORE=cpu). For tests, not for mining.');
    return make(cpuAddon, null, cpuAddon.devices(), 'cpu');
  }
  const execPath = opts.execPath || process.execPath;
  const req = opts.require || createRequire(execPath);
  const execDir = path.dirname(execPath);
  const dirs = [];
  if (opts.resourcesPath) dirs.push(path.join(opts.resourcesPath, 'native'));
  dirs.push(execDir, path.join(execDir, 'native'), DEV_DIR);
  const first = VENDOR_FILES[opts.vendor];
  const files = first ? [first].concat(FILES.filter((f) => f !== first)) : FILES;
  const candidates = env.NEB_CORE_PATH ? [env.NEB_CORE_PATH] : files.flatMap((f) => dirs.map((d) => path.join(d, f)));
  const why = [];
  for (const file of candidates) {
    let addon;
    try {
      addon = req(file);
    } catch (e) {
      if (e.code !== 'MODULE_NOT_FOUND') why.push(file + ' would not load: ' + e.message);
      continue;
    }
    if (!addon || typeof addon.createCore !== 'function' || typeof addon.devices !== 'function') {
      why.push(file + ' is not a NEB core');
      continue;
    }
    if (addon.NEB_API !== API) {
      why.push(file + ' speaks NEB core API ' + addon.NEB_API + ', this miner ' + API);
      continue;
    }
    let devices;
    try {
      devices = addon.devices();
    } catch (e) {
      why.push(file + ' could not list its cards: ' + e.message);
      continue;
    }
    if (!Array.isArray(devices) || !devices.length) {
      why.push(file + ' found no card');
      continue;
    }
    log('info', 'NEB core: ' + file + ' · ' + devices.length + ' card' + (devices.length === 1 ? '' : 's'));
    return make(addon, file, devices, path.basename(file, '.node'));
  }
  for (const w of why) log('warn', 'NEB core: ' + w);
  log('error', 'NEB core: none found (' + (env.NEB_CORE_PATH ? 'NEB_CORE_PATH=' + env.NEB_CORE_PATH
    : 'looked for ' + FILES.join(', ') + ' beside the app and in earn/native/neb/build') + ')');
  return null;
}

// ── Checking a core against the reference ───────────────────────────────────
//
// What a core has to pass before it mines (earn/native/neb/verify-core.js runs
// it on a card). opts.ds and opts.weights are a dataset and a model to check
// on: the network's own and a model its chain has reached, or the test dataset
// with a trained model. Each check is { name, ok, detail }.
//
//   vectors   the chain's own vectors for the KEP1 network (vectors/nebhash2.txt):
//             every share's digest and ticket from evalKeys, on each model
//             state the vectors step through. Skipped without opts.vectors.
//   full      64 random keys on the given model: digest and ticket from
//             evalKeys against the reference, whatever their live rows.
//   exact     a stretch of nonces across the point where a core starts a new
//             launch (2^24): at a target every ticket meets, the hits must be
//             exactly the nonces with at most maxLive live samples; at a
//             target half of them meet, exactly the reference's hits.
//   top       the same at the last nonces of the 32-bit range.
//   overflow  a target every ticket meets and maxHits 4: 4 hits, overflow set.
//   big       2^30 nonces at a target about 32 evaluated attempts meet: every
//             hit must check out on the reference, the count must be near 32,
//             and the share of attempts evaluated must be what the model's
//             live samples predict. Its speed is the core's hashrate.
//   triples   with opts.tripleRows: after buildTriples, an exact stretch again.
//             The hits now include the attempts with 3 live samples among the
//             rows the table covers.
//   big3      the long run again with the triple table.
const VERIFY_SLOT = 3;

function shareKey(arch, step, j) {
  const le = Buffer.alloc(8);
  le.writeUInt32LE(step, 0);
  le.writeUInt32LE(j, 4);
  return blake3.hash(Buffer.concat([Buffer.from('neb-test-share2'), Buffer.from([arch]), le]));
}

// The vectors of one architecture, as steps of { shares: [{ key, digest, ticket }] }.
function parseVectors(text, arch) {
  const steps = [];
  for (const line of String(text).split('\n')) {
    const p = line.trim().split(' ');
    if (p[0] !== 'share' || Number(p[1]) !== arch) continue;
    const step = Number(p[2]);
    if (!steps[step]) steps[step] = { shares: [] };
    steps[step].shares.push({ key: shareKey(arch, step, Number(p[3])), digest: p[4], ticket: p[5] });
  }
  return steps;
}

async function checkVectors(core, text) {
  const arch = nh.KEP1;
  const ds = nh.syn2Dataset(arch);
  const m = nh.modelInit(arch, Buffer.alloc(32));
  core.setDataset(VERIFY_SLOT, arch, ds.values, ds.labels);
  let shares = 0;
  let wrong = 0;
  for (const step of parseVectors(text, arch)) {
    await core.setModel(VERIFY_SLOT, Buffer.from(m.w.buffer, m.w.byteOffset, m.w.length));
    const got = await core.evalKeys(VERIFY_SLOT, Buffer.concat(step.shares.map((s) => s.key)));
    const sum = new Int32Array(m.w.length);
    step.shares.forEach((s, i) => {
      shares++;
      if (got.toString('hex', 64 * i, 64 * i + 64) !== s.digest + s.ticket) wrong++;
      const g = nh.shareEval(m, ds, s.key).g;
      for (let f = 0; f < sum.length; f++) sum[f] += g[f];
    });
    nh.modelApply(m, sum);
  }
  return { ok: shares > 0 && !wrong, detail: (shares - wrong) + ' of ' + shares + ' shares' };
}

// A dataset to check a core on when the network's file is not at hand: the
// test dataset's samples, with 5 labels in 8 set to class 0. On the model in
// earn/native/neb/vectors/syn2-trained.hex, which answers class 0 for every
// sample, those 5 in 8 are dead and the rest live, about a trained chain
// model's mix.
function checkDataset() {
  const ds = nh.syn2Dataset(nh.KEP1);
  for (let i = 0; i < ds.n; i++) ds.labels[i] = i % 8 < 5 ? 0 : 1;
  return ds;
}

// A target that a fraction `frac` of tickets meet, as 32 little-endian bytes.
function targetFor(frac) {
  const t = BigInt(Math.floor(Math.min(1, frac) * 2 ** 52)) << 204n;
  return Buffer.from((t - 1n).toString(16).padStart(64, '0'), 'hex').reverse();
}

async function verifyCore(core, opts) {
  const random = opts.random || crypto.randomBytes;
  const now = opts.now || Date.now;
  // exactCount: nonces in each exact range (2,048 on a card). bigCount: nonces in each big run (2^30), in scans of
  // scanCount (2^28). tripleRows: rows to ask the triple table for (0: none), within tripleBytes.
  // fullCount: keys in the full check (64).
  const { exactCount, bigCount, scanCount, tripleRows, fullCount } = opts;
  const slot = 0;
  const checks = [];
  const record = (name, ok, detail) => checks.push({ name, ok, detail });

  if (opts.vectors) {
    const v = await checkVectors(core, opts.vectors);
    record('vectors', v.ok, v.detail);
  }

  const ds = opts.ds;
  const m = { arch: ds.arch, w: int8(opts.weights) };
  core.setDataset(slot, ds.arch, ds.values, ds.labels);
  const info = await core.setModel(slot, Buffer.from(m.w.buffer, m.w.byteOffset, m.w.length));
  const model = { m, ds, cover: coverage(nh.liveSamples(m, ds), info.maxLive, 0) };

  // full
  const keys = random(32 * fullCount);
  const got = await core.evalKeys(slot, keys);
  let wrong = 0;
  for (let i = 0; i < fullCount; i++) {
    const r = nh.shareEval(m, ds, keys.subarray(32 * i, 32 * i + 32));
    if (!got.subarray(64 * i, 64 * i + 64).equals(Buffer.concat([r.digest, r.ticket]))) wrong++;
  }
  record('full', !wrong && info.live === model.cover.rows, (fullCount - wrong) + ' of ' + fullCount + ' keys · ' + info.live + ' of '
    + ds.n + ' samples live, the reference has ' + model.cover.rows);

  const header = random(hdr.HEADER_BYTES);
  // A target about half the tickets meet: the top byte 0x7f.
  const half = Buffer.concat([Buffer.alloc(31, 0xff), Buffer.from([0x7f])]);
  // A stretch of nonces at both targets: the core's hits and its count against the reference's.
  const exact = async (name, start) => {
    const parts = [];
    for (const target of [ALL_ONES, half]) {
      const want = { hits: [], evaluated: 0 };
      scanRange({ header, target }, model, start, start + exactCount, want);
      const res = await core.scan({ slot, header, target, start, count: exactCount, maxHits: exactCount });
      const mine = hitList(res);
      const same = mine.length === want.hits.length && mine.every((n, i) => n === want.hits[i]) && res.evaluated === want.evaluated;
      parts.push({ same, text: mine.length + ' hits, the reference has ' + want.hits.length });
    }
    record(name, parts.every((p) => p.same), parts.map((p) => p.text).join(' · '));
  };
  // A long run at a target about 32 evaluated attempts meet. Returns its rates.
  const big = async (name) => {
    const expected = coveredShare(model.cover);
    const target = targetFor(32 / (bigCount * expected));
    const t0 = now();
    let hits = 0;
    let badHits = 0;
    let evaluated = 0;
    for (let done = 0; done < bigCount; done += scanCount) {
      const count = Math.min(scanCount, bigCount - done);
      const res = await core.scan({ slot, header, target, start: done, count, maxHits: 1024 });
      evaluated += res.evaluated;
      for (const nonce of hitList(res)) {
        hits++;
        const key = hdr.attemptKey(header, nonce);
        if (!evaluates(model.cover, key) || !nh.meetsTarget(nh.shareEval(m, ds, key).ticket, target)) badHits++;
      }
    }
    const secs = Math.max(now() - t0, 1) / 1000;
    const share = evaluated / bigCount;
    // 32 expected: fewer than 8 or more than 80 is about a 1 in 10^5 chance. The share evaluated: within 5% of
    // what the live samples predict, or within 5 standard deviations when the run is too short for that.
    const slack = Math.max(expected * 0.05, 5 * Math.sqrt(expected * (1 - expected) / bigCount));
    const ok = !badHits && hits >= 8 && hits <= 80 && Math.abs(share - expected) <= slack;
    record(name, ok, hits + ' hits (about 32 expected), ' + badHits + ' wrong · ' + (share * 100).toFixed(3)
      + '% of attempts evaluated, ' + (expected * 100).toFixed(3) + '% expected · ' + bigCount + ' nonces in ' + secs.toFixed(2) + ' s');
    return { screenRate: bigCount / secs, hashrate: evaluated / secs };
  };

  await exact('exact', 2 ** 24 - (exactCount >> 1));
  await exact('top', 2 ** 32 - exactCount);

  // overflow: enough nonces for more than 4 attempts to be evaluated.
  const overCount = Math.min(2 ** 20, Math.max(64, Math.ceil(64 / coveredShare(model.cover))));
  const over = await core.scan({ slot, header, target: ALL_ONES, start: 0, count: overCount, maxHits: 4 });
  record('overflow', over.overflow === true && over.hits.length === 16, (over.hits.length / 4) + ' hits back, overflow ' + over.overflow);

  const rates = await big('big');
  const out = { info, screenRate: rates.screenRate, hashrate: rates.hashrate };

  // The triple table: the same exact stretch and long run, with the attempts it adds.
  if (tripleRows) {
    const triples = await core.buildTriples(slot, tripleRows, opts.tripleBytes);
    model.cover.tripleRows = triples.rows;
    await exact('triples', 2 ** 25 - (exactCount >> 1));
    const withTriples = await big('big3');
    Object.assign(out, { triples, tripleScreenRate: withTriples.screenRate, tripleHashrate: withTriples.hashrate });
  }

  return Object.assign({ ok: checks.every((c) => c.ok), checks }, out);
}

module.exports = {
  API, FILES, VENDOR_FILES, CPU_CHUNK, ALL_ONES, CpuCore, cpuAddon, readWork, coverage, evaluates, coveredShare, scanRange,
  packHits, hitList, nebCoreFactory, parseVectors, checkDataset, targetFor, verifyCore,
};
