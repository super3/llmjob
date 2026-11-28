'use strict';

const fs = require('fs');
const path = require('path');
const { EventEmitter } = require('events');
const E = require('../src/main/nebEngine');
const nc = require('../src/main/nebCore');
const nh = require('../src/shared/neb/nebhash2');
const hdr = require('../src/shared/neb/header');
const stratum = require('../src/shared/neb/stratum');

const { NebEngine } = E;

// Real setImmediate, so promises can be flushed; fake intervals and timeouts.
beforeEach(() => jest.useFakeTimers({ doNotFake: ['setImmediate', 'nextTick', 'queueMicrotask'] }));
afterEach(() => jest.useRealTimers());

const flush = async (n = 4) => { for (let i = 0; i < n; i++) await new Promise((r) => setImmediate(r)); };

const ADDRESS = 'neb1kwgy6lnm774h2cckdag40sj6sw25cpustla0gu5e37n0mn30wnzsjvwr4n';
const DS = nc.checkDataset();
const PACKED = Buffer.from(fs.readFileSync(path.join(__dirname, '..', 'native', 'neb', 'vectors', 'syn2-trained.hex'), 'utf8').trim(), 'hex');
const WH = nh.weightsHash(PACKED).toString('hex');
const DATASET_HASH = 'ab'.repeat(32);
const EASY = 'ff'.repeat(32);
const NEVER = '00'.repeat(32);

function fakeSocket() {
  const s = new EventEmitter();
  s.lines = [];
  s.write = (l) => s.lines.push(JSON.parse(l));
  s.destroy = jest.fn(() => { s.emit('close'); });
  s.remoteAddress = '203.0.113.5';
  s.remoteFamily = 'IPv4';
  s.say = (obj) => s.emit('data', JSON.stringify(obj) + '\n');
  return s;
}

// A core whose calls wait until the test answers them.
function heldCore(index) {
  const core = {
    device: { index, name: 'Card ' + index, vendor: 'fake' },
    datasets: [],
    models: [],
    scans: [],
    setDataset: jest.fn((slot, arch, values, labels) => { core.datasets[slot] = { arch, n: labels.length }; }),
    setModel: jest.fn((slot, weights) => new Promise((resolve, reject) => core.models.push({ slot, weights, resolve, reject }))),
    scan: jest.fn((w) => new Promise((resolve, reject) => core.scans.push({ w, resolve, reject }))),
    close: jest.fn(),
  };
  core.loaded = (info = {}) => core.models.shift().resolve({ samples: 2048, live: 768, maxLive: 2, entries: 10, tableBytes: 3 << 20, cpuMs: 5, gpuMs: 7, ...info });
  core.answer = (nonces, extra = {}) => core.scans.shift().resolve(Object.assign(nc.packHits(nonces, 256, 1000), extra));
  return core;
}

function setup({ cards = [0], timing = {}, factory, datasets = [DS, DS] } = {}) {
  const cores = cards.map(heldCore);
  const socks = [];
  let clock = 1000000;
  const engine = new NebEngine({
    connect: jest.fn(() => { const s = fakeSocket(); socks.push(s); return s; }),
    factory: factory === undefined ? { devices: cards.map((index) => ({ index })), createCore: (i) => cores[cards.indexOf(i)] } : factory,
    agent: 'llmjob-earn/9.9.9',
    now: () => clock,
    random: (n) => Buffer.alloc(n, 0xab),
    timing,
  });
  const logs = [];
  const events = [];
  engine.on('log', (l) => logs.push(l.level + ' ' + l.line));
  engine.on('event', (e) => events.push(e));
  engine.on('error', () => {});
  const sock = () => socks[socks.length - 1];
  return {
    engine, cores, socks, sock, logs, events, tick: (ms) => { clock += ms; },
    settings: { address: ADDRESS, worker: 'rig01', endpoint: 'neb.rabidmining.com:3470', datasets, datasetHash: DATASET_HASH },
  };
}

function job(over = {}) {
  return {
    job_id: 'j1', clean: true, version: 1, parent: '11'.repeat(32), time_start: 1700000000, time_window: 3600, bits: 5,
    target: '00'.repeat(32), share_target: '00'.repeat(32), coinbase_prefix: 'aa', coinbase_suffix: 'bb', branch: [],
    tx_count: 1, shares_root: '22'.repeat(32), height: 7, slot: 0, weights_hash: WH, expiry: 2000, ...over,
  };
}

const SUBSCRIBED = { id: 1, result: { session: 's1', extranonce1: '0a0b0c0d', extranonce2_size: 12, chain_id: 'cc', dataset_hash: DATASET_HASH }, error: null };

async function loggedIn(t, { target = EASY, jobOver, model = true } = {}) {
  t.engine.start(t.settings);
  t.sock().emit('connect');
  t.sock().say(SUBSCRIBED);
  t.sock().say({ id: 2, result: true, error: null });
  t.sock().say({ id: null, method: 'mining.set_target', params: [target] });
  if (model) t.sock().say({ id: null, method: 'mining.set_model', params: [0, WH, PACKED.toString('hex')] });
  t.sock().say({ id: null, method: 'mining.notify', params: [job(jobOver)] });
  await flush();
}

// Logged in, the job's model on the card, the first scan waiting.
async function mining(t, opts) {
  await loggedIn(t, opts);
  t.cores[0].loaded();
  await flush();
}

describe('starting', () => {
  test('with no core: false, and says why', () => {
    const t = setup({ factory: null });
    const errors = [];
    t.engine.on('error', (e) => errors.push(e.message));
    expect(t.engine.start(t.settings)).toBe(false);
    expect(errors).toEqual(['there is no NEB core in this build']);
    expect(t.engine.isRunning()).toBe(false);
    expect(t.engine.stop()).toBe(false);
  });

  test('without the dataset file: false', () => {
    const t = setup();
    expect(t.engine.start({ address: ADDRESS })).toBe(false);
    expect(t.engine.start(Object.assign({}, t.settings, { datasets: [] }))).toBe(false);
    expect(t.logs).toEqual(['error neb: the NEB dataset file is not loaded', 'error neb: the NEB dataset file is not loaded']);
    const bare = new NebEngine();
    bare.on('error', () => {});
    expect(bare.start()).toBe(false);
  });

  test('when no card opens: false', () => {
    const t = setup({ factory: { devices: [{ index: 0 }], createCore: () => { throw new Error('out of memory'); } } });
    expect(t.engine.start(t.settings)).toBe(false);
    expect(t.logs).toEqual(['error neb: card 0 would not open: out of memory', 'error neb: no card would open a NEB core']);
  });

  test('a card that will not open is skipped; the rest mine; a core with no name gets one; each gets every dataset', () => {
    const made = [];
    const t = setup({
      factory: {
        devices: [{ index: 0 }, { index: 1 }],
        createCore: (i) => {
          if (i === 0) throw 'busy';
          const c = { setDataset: jest.fn(), setModel: () => new Promise(() => {}), scan: () => new Promise(() => {}), close: jest.fn() };
          made.push(c);
          return c;
        },
      },
    });
    expect(t.engine.start(t.settings)).toBe(true);
    expect(t.engine.start(t.settings)).toBe(false);
    expect(t.logs.slice(0, 3)).toEqual([
      'error neb: card 0 would not open: busy', 'info neb: card 1: card 1',
      'info neb: mining NEB for neb1kwgy…vwr4n on 1 card · neb.rabidmining.com:3470',
    ]);
    expect(made[0].setDataset.mock.calls.map((c) => c.slice(0, 2))).toEqual([[0, 1], [1, 1]]);
    expect(t.engine.snapshot()).toEqual({ state: 'mining', hashrate: 0, accepted: 0, rejected: 0, cards: 1, poolError: null });
    t.engine.stop();
  });

  test('the cards asked for, and "cards" in the log', () => {
    const t = setup({ cards: [0, 1] });
    t.engine.start(Object.assign({}, t.settings, { cards: [1, 0] }));
    expect(t.logs).toContain('info neb: mining NEB for neb1kwgy…vwr4n on 2 cards · neb.rabidmining.com:3470');
    expect(t.engine.connect).toHaveBeenCalledWith('neb.rabidmining.com', 3470, false);
    t.engine.stop();
  });
});

describe('the pool', () => {
  test('subscribes, checks the dataset hash, authorizes, and announces the cards at the first job', async () => {
    const t = setup();
    const seen = [];
    t.engine.on('authorized', (a) => seen.push(a));
    await loggedIn(t);
    expect(t.sock().lines).toEqual([
      { id: 1, method: 'mining.subscribe', params: ['llmjob-earn/9.9.9', 'neb-pool/1'] },
      { id: 2, method: 'mining.authorize', params: [ADDRESS + '.rig01', 'x'] },
    ]);
    expect(seen).toEqual([{ host: 'neb.rabidmining.com', address: '203.0.113.5', family: 'IPv4' }]);
    expect(t.events.filter((e) => e.type === 'connected')).toEqual([{ type: 'connected', gpuIndex: 0, endpoint: 'neb.rabidmining.com:3470', gpu: 'Card 0' }]);
    expect(t.engine.poolConnected()).toBe(true);
    // The same announcement is not made again for the next job.
    t.sock().say({ id: null, method: 'mining.notify', params: [job({ job_id: 'j2' })] });
    expect(t.events.filter((e) => e.type === 'connected').length).toBe(1);
    t.engine.stop();
  });

  test('a pool on another dataset file is refused, and tried again later', async () => {
    const t = setup();
    t.engine.start(t.settings);
    t.sock().emit('connect');
    t.sock().say({ id: 1, result: Object.assign({}, SUBSCRIBED.result, { dataset_hash: 'cd'.repeat(32) }), error: null });
    expect(t.logs).toContain('error neb: the NEB pool mines another dataset file (cdcdcdcdcdcdcdcd…); this miner has abababababababab…');
    expect(t.sock().destroy).toHaveBeenCalled();
    expect(t.logs).toContain('info neb: pool connection closed; reconnecting in 60 s');
    expect(t.engine.snapshot().poolError).toMatch(/another dataset file/);
    jest.advanceTimersByTime(E.LOGIN_RETRY_MS);
    expect(t.socks.length).toBe(2);
    t.engine.stop();
  });

  test('with no dataset hash of its own it takes the pool\'s word', async () => {
    const t = setup();
    t.settings.datasetHash = null;
    await loggedIn(t);
    expect(t.engine.poolConnected()).toBe(true);
    t.engine.stop();
  });

  test('a refused subscribe and a refused login', async () => {
    const t = setup();
    t.engine.start(t.settings);
    t.sock().emit('connect');
    t.sock().say({ id: 1, result: null, error: [20, 'full'] });
    expect(t.logs).toContain('error neb: the NEB pool refused the connection: [20] full');
    jest.advanceTimersByTime(E.LOGIN_RETRY_MS);
    t.sock().emit('connect');
    t.sock().say(SUBSCRIBED);
    t.sock().say({ id: 2, result: null, error: [24, 'not a valid neb1 address'] });
    expect(t.logs).toContain('error neb: the NEB pool refused the login: [24] not a valid neb1 address');
    // The second failure in a row says no more about the close.
    expect(t.logs.filter((l) => /pool connection closed/.test(l)).length).toBe(1);
    t.engine.stop();
  });

  test('no login in time: closes and reconnects; the timer of a socket already replaced or logged in does nothing', async () => {
    const t = setup();
    t.engine.start(t.settings);
    jest.advanceTimersByTime(E.LOGIN_TIMEOUT_MS);
    expect(t.logs).toContain('warn neb: no login at the pool in 20 s; reconnecting');
    expect(t.engine.snapshot().poolError).toBe('the NEB pool did not answer the login');
    jest.advanceTimersByTime(E.RECONNECT_MS);
    expect(t.socks.length).toBe(2);
    // The second time it stays quiet.
    jest.advanceTimersByTime(E.LOGIN_TIMEOUT_MS);
    expect(t.logs.filter((l) => /no login at the pool/.test(l)).length).toBe(1);
    jest.advanceTimersByTime(E.RECONNECT_MS);
    t.sock().emit('connect');
    t.sock().say(SUBSCRIBED);
    t.sock().say({ id: 2, result: true, error: null });
    const before = t.sock().destroy.mock.calls.length;
    jest.advanceTimersByTime(E.LOGIN_TIMEOUT_MS);
    expect(t.sock().destroy.mock.calls.length).toBe(before);
    t.engine.stop();
  });

  test('a connect that throws is tried again', () => {
    const t = setup();
    let n = 0;
    const real = t.engine.connect;
    t.engine.connect = (...a) => { if (n++ === 0) throw new Error('no network'); return real(...a); };
    t.engine.start(t.settings);
    expect(t.logs).toContain('error neb: pool connection failed: no network');
    jest.advanceTimersByTime(E.RECONNECT_MS);
    expect(t.socks.length).toBe(1);
    t.engine.stop();
  });

  test('a socket error is reported once, as an event each time, and a close before the login counts', () => {
    const t = setup();
    t.engine.start(t.settings);
    t.sock().emit('error', new Error('getaddrinfo ENOTFOUND neb.rabidmining.com'));
    t.sock().emit('close');
    expect(t.logs).toContain('error neb: pool socket error: getaddrinfo ENOTFOUND neb.rabidmining.com');
    expect(t.events).toContainEqual({ type: 'connect-failed', reason: 'getaddrinfo ENOTFOUND neb.rabidmining.com', dns: true });
    expect(t.engine.snapshot().poolError).toBe('can\'t reach the NEB pool: getaddrinfo ENOTFOUND neb.rabidmining.com');
    jest.advanceTimersByTime(E.RECONNECT_MS);
    t.sock().emit('error', new Error('ECONNREFUSED'));
    expect(t.logs.filter((l) => /pool socket error/.test(l)).length).toBe(1);
    expect(t.events).toContainEqual({ type: 'connect-failed', reason: 'ECONNREFUSED', dns: false });
    t.engine.stop();
  });

  test('a close with no word before the login has its own reason', () => {
    const t = setup();
    t.engine.start(t.settings);
    t.sock().emit('close');
    expect(t.engine.snapshot().poolError).toBe('the NEB pool closed the connection before the login');
    t.engine.stop();
  });

  test('a lost connection keeps the rig "connected" for a grace period, then not', async () => {
    const t = setup();
    await loggedIn(t);
    const old = t.sock();
    old.emit('close');
    expect(t.engine.poolConnected()).toBe(true);
    t.tick(E.POOL_GRACE_MS);
    expect(t.engine.poolConnected()).toBe(false);
    expect(t.engine.snapshot().hashrate).toBe(0);
    // What the old socket says or does after it was replaced is ignored.
    jest.advanceTimersByTime(E.RECONNECT_MS);
    old.say({ id: 2, result: true, error: null });
    old.emit('close');
    expect(t.socks.length).toBe(2);
    t.engine.stop();
  });

  test('silence for too long: reconnects', async () => {
    const t = setup();
    await loggedIn(t);
    jest.advanceTimersByTime(E.WATCHDOG_MS);
    expect(t.sock().destroy).not.toHaveBeenCalled();
    t.tick(E.SILENCE_MS);
    jest.advanceTimersByTime(E.WATCHDOG_MS);
    expect(t.logs).toContain('warn neb: no word from the pool in 5 minutes; reconnecting');
    // Between sockets the watchdog has nothing to check.
    jest.advanceTimersByTime(E.WATCHDOG_MS);
    t.engine.stop();
  });

  test('lines it cannot use are logged or ignored', async () => {
    const t = setup();
    await loggedIn(t);
    t.sock().emit('data', '\n{nope\n');
    t.sock().say({ id: null, method: 'mining.ping' });
    t.sock().say({ id: null, result: null, error: [20, 'slow down'] });
    t.sock().say({ id: null, method: 'mining.set_model', params: [0, 'zz', ''] });
    t.sock().say({ id: 99, result: true, error: null });
    expect(t.logs).toEqual(expect.arrayContaining([
      'info neb: pool: {nope', 'info neb: pool: {"id":null,"method":"mining.ping"}', 'error neb: pool: [20] slow down',
      'error neb: ignoring a model that is not [slot, hash, weights]',
    ]));
    t.engine.stop();
  });
});

describe('models and jobs', () => {
  test('a model for a dataset the file lacks, of the wrong size, or with the wrong hash is ignored', async () => {
    const t = setup();
    await loggedIn(t, { model: false });
    expect(t.logs).toContain('error neb: ignoring job j1: the pool has not sent its model');
    t.sock().say({ id: null, method: 'mining.set_model', params: [5, WH, PACKED.toString('hex')] });
    t.sock().say({ id: null, method: 'mining.set_model', params: [0, WH, 'abcd'] });
    t.sock().say({ id: null, method: 'mining.set_model', params: [0, 'ab'.repeat(32), PACKED.toString('hex')] });
    expect(t.logs).toEqual(expect.arrayContaining([
      'error neb: ignoring a model for dataset 5, which the dataset file does not have',
      'error neb: ignoring a model of 2 bytes',
      'error neb: ignoring a model whose hash is not the one the pool names',
    ]));
    expect(t.cores[0].setModel).not.toHaveBeenCalled();
    t.engine.stop();
  });

  test('a job before any target, for another slot than its model\'s, or one it cannot read is not mined', async () => {
    const t = setup();
    t.engine.start(t.settings);
    t.sock().emit('connect');
    t.sock().say(SUBSCRIBED);
    t.sock().say({ id: 2, result: true, error: null });
    t.sock().say({ id: null, method: 'mining.set_model', params: [0, WH, PACKED.toString('hex')] });
    t.sock().say({ id: null, method: 'mining.notify', params: [job()] });
    expect(t.logs).toContain('error neb: ignoring job j1: the pool has set no target');
    t.sock().say({ id: null, method: 'mining.set_target', params: [EASY] });
    t.sock().say({ id: null, method: 'mining.notify', params: [job({ slot: 1 })] });
    expect(t.logs).toContain('error neb: ignoring job j1: the pool has not sent its model');
    t.sock().say({ id: null, method: 'mining.notify', params: [job({ bits: -1 })] });
    t.sock().say({ id: null, method: 'mining.notify', params: [{}] });
    expect(t.logs).toEqual(expect.arrayContaining([
      'error neb: ignoring a job this miner cannot mine (j1): bits is not a 32-bit number',
      'error neb: ignoring a job this miner cannot mine: no job_id',
    ]));
    await flush();
    expect(t.cores[0].setModel).not.toHaveBeenCalled();
    t.engine.stop();
  });

  test('only the newest models are kept', async () => {
    const t = setup();
    await loggedIn(t);
    for (let i = 0; i < E.MODELS_KEPT; i++) {
      const packed = Buffer.from(PACKED);
      packed[0] = i + 1;
      t.sock().say({ id: null, method: 'mining.set_model', params: [1, nh.weightsHash(packed).toString('hex'), packed.toString('hex')] });
    }
    expect(t.engine.models.size).toBe(E.MODELS_KEPT);
    expect(t.engine.models.has(WH)).toBe(false);
    t.engine.stop();
  });

  test('the card gets the job\'s model once, and says what it holds', async () => {
    const t = setup();
    await loggedIn(t);
    expect(t.cores[0].setModel).toHaveBeenCalledTimes(1);
    const sent = t.cores[0].models[0];
    expect(sent.slot).toBe(0);
    expect(Array.from(new Int8Array(sent.weights.buffer, sent.weights.byteOffset, 8))).toEqual(Array.from(nh.unpackWeights(PACKED, 8)));
    t.cores[0].loaded();
    await flush();
    expect(t.logs).toContain('info neb: card 0 model ' + WH.slice(0, 8) + ' (slot 0): 768 of 2048 samples live · 2.92% of attempts evaluated · table 3 MB in 12 ms');
    expect(t.cores[0].scan).toHaveBeenCalledTimes(1);
    // The next job on the same model scans without loading it again.
    t.sock().say({ id: null, method: 'mining.notify', params: [job({ job_id: 'j2' })] });
    t.cores[0].answer([]);
    await flush();
    expect(t.cores[0].setModel).toHaveBeenCalledTimes(1);
    expect(t.cores[0].scan).toHaveBeenCalledTimes(2);
    t.engine.stop();
  });

  test('a job past its expiry is not mined; one with no expiry is', async () => {
    const t = setup();
    await loggedIn(t, { jobOver: { expiry: 1000 } });
    expect(t.cores[0].setModel).not.toHaveBeenCalled();
    t.sock().say({ id: null, method: 'mining.notify', params: [job({ job_id: 'j2', expiry: undefined })] });
    await flush();
    expect(t.cores[0].setModel).toHaveBeenCalledTimes(1);
    t.engine.stop();
  });
});

describe('scanning', () => {
  test('a scan is the job\'s header under a fresh extranonce2, from nonce 0, at the pool target', async () => {
    const t = setup();
    await mining(t);
    const w = t.cores[0].scans[0].w;
    const en2 = Buffer.from('ababababab00' + 'abababab'.padStart(12, '0'), 'hex');
    const parsed = stratum.parseJob(job());
    expect(w).toEqual({
      slot: 0, header: hdr.buildHeader(parsed, Buffer.from('0a0b0c0d', 'hex'), en2, 1700000000), target: Buffer.alloc(32, 0xff),
      start: 0, count: E.FIRST_BATCH, maxHits: E.MAX_HITS,
    });
    t.cores[0].answer([]);
    await flush();
    // The next batch has the next extranonce2.
    const en2b = Buffer.from(en2);
    en2b.writeUIntBE(0xabababab + 1, 6, 6);
    expect(t.cores[0].scans[0].w.header.equals(hdr.buildHeader(parsed, Buffer.from('0a0b0c0d', 'hex'), en2b, 1700000000))).toBe(true);
    t.engine.stop();
  });

  test('a short extranonce2 is all counter', async () => {
    const t = setup();
    t.engine.start(t.settings);
    t.sock().emit('connect');
    t.sock().say({ id: 1, result: Object.assign({}, SUBSCRIBED.result, { extranonce2_size: 4 }), error: null });
    t.sock().say({ id: 2, result: true, error: null });
    t.sock().say({ id: null, method: 'mining.set_target', params: [EASY] });
    t.sock().say({ id: null, method: 'mining.set_model', params: [0, WH, PACKED.toString('hex')] });
    t.sock().say({ id: null, method: 'mining.notify', params: [job()] });
    await flush();
    t.cores[0].loaded();
    await flush();
    const parsed = stratum.parseJob(job());
    expect(t.cores[0].scans[0].w.header.equals(hdr.buildHeader(parsed, Buffer.from('0a0b0c0d', 'hex'), Buffer.from('abababab', 'hex'), 1700000000))).toBe(true);
    t.engine.stop();
  });

  test('the rate is attempts evaluated a second, smoothed; the batch grows toward the time it should take', async () => {
    const t = setup();
    await mining(t);
    t.tick(100);
    t.cores[0].answer([], { evaluated: 50000 });
    await flush();
    expect(t.engine.snapshot().hashrate).toBe(500000);
    expect(t.cores[0].scans[0].w.count).toBe(Math.round(E.FIRST_BATCH * E.BATCH_MS / 100));
    t.tick(1000);
    t.cores[0].answer([], { evaluated: 0 });
    await flush();
    expect(t.engine.snapshot().hashrate).toBe(400000);
    jest.advanceTimersByTime(E.STATUS_MS);
    const status = t.events.filter((e) => e.type === 'status').pop();
    expect(status).toMatchObject({ gpuIndex: 0, hashrate: 400000, accepted: 0, rejected: 0, gpu: 'Card 0', coin: 'neb' });
    expect(status.screened).toBeGreaterThan(0);
    t.engine.stop();
  });

  test('a batch never passes 2^32 nonces or drops under 2^16, and grows at most 8 times a step', async () => {
    const t = setup({ timing: { firstBatch: 2 ** 31 } });
    await mining(t, { jobOver: { expiry: undefined } });
    t.cores[0].answer([]);
    await flush();
    expect(t.cores[0].scans[0].w.count).toBe(2 ** 32);
    t.tick(10 ** 9);
    t.cores[0].answer([]);
    await flush();
    expect(t.cores[0].scans[0].w.count).toBe(2 ** 16);
    t.cores[0].answer([]);
    await flush();
    expect(t.cores[0].scans[0].w.count).toBe(2 ** 19);
    t.engine.stop();
  });

  test('a hit is evaluated on the reference and submitted; the answers are counted', async () => {
    const t = setup();
    await mining(t);
    const w = t.cores[0].scans[0].w;
    t.cores[0].answer([5, 9], { overflow: true });
    await flush();
    expect(t.logs).toContain('warn neb: card 0 found more than 256 hits in one batch');
    const en2 = 'ababababab00' + '0000abababab';
    expect(t.sock().lines.slice(2)).toEqual([
      { id: 10, method: 'mining.submit', params: [ADDRESS + '.rig01', 'j1', en2, 1700000000, 5] },
      { id: 11, method: 'mining.submit', params: [ADDRESS + '.rig01', 'j1', en2, 1700000000, 9] },
    ]);
    expect(w.header.length).toBe(80);
    t.sock().say({ id: 10, result: true, error: null });
    t.sock().say({ id: 11, result: null, error: [22, 'duplicate attempt'] });
    expect(t.logs).toEqual(expect.arrayContaining(['info neb: NEB share accepted (card 0)', 'error neb: NEB share rejected (card 0): [22] duplicate attempt']));
    expect(t.engine.snapshot()).toMatchObject({ accepted: 1, rejected: 1 });
    t.engine.stop();
  });

  test('the same hit is not sent twice', async () => {
    const t = setup();
    await mining(t);
    t.cores[0].answer([5, 5]);
    await flush();
    expect(t.sock().lines.filter((l) => l.method === 'mining.submit').length).toBe(1);
    t.engine.stop();
  });

  test('hits of a batch are kept when the job was replaced on the same parent, dropped on a new block', async () => {
    const t = setup();
    await mining(t);
    t.sock().say({ id: null, method: 'mining.notify', params: [job({ job_id: 'j2', clean: false })] });
    t.cores[0].answer([5]);
    await flush();
    expect(t.sock().lines.filter((l) => l.method === 'mining.submit').map((l) => l.params[1])).toEqual(['j1']);
    t.sock().say({ id: null, method: 'mining.notify', params: [job({ job_id: 'j3', parent: '33'.repeat(32) })] });
    t.cores[0].answer([6]);
    await flush();
    expect(t.sock().lines.filter((l) => l.method === 'mining.submit').length).toBe(1);
    // And with the connection gone there is no job to hold them against.
    t.sock().emit('close');
    t.cores[0].answer([7]);
    await flush();
    expect(t.sock().lines.filter((l) => l.method === 'mining.submit').length).toBe(1);
    t.engine.stop();
  });

  test('a core that keeps reporting nonces that are not hits is taken off mining, and the miner gives up', async () => {
    const t = setup();
    const stopped = [];
    t.engine.on('stopped', (c) => stopped.push(c));
    await mining(t, { target: NEVER });
    t.cores[0].answer([1, 2, 3, 4]);
    await flush();
    expect(t.logs).toEqual(expect.arrayContaining([
      'warn neb: card 0 reported a nonce that is not a hit (1 of 3)', 'warn neb: card 0 reported a nonce that is not a hit (3 of 3)',
      'error neb: card 0 stopped: its core keeps reporting nonces that are not hits', 'error neb: every card stopped; the NEB miner gives up',
    ]));
    expect(t.logs.filter((l) => /not a hit/.test(l)).length).toBe(3);
    expect(t.cores[0].close).toHaveBeenCalled();
    expect(stopped).toEqual([1]);
    expect(t.engine.isRunning()).toBe(false);
  });

  test('a scan that fails takes its card off; the other card carries on', async () => {
    const t = setup({ cards: [0, 1] });
    await loggedIn(t);
    t.cores[0].loaded();
    t.cores[1].loaded();
    await flush();
    t.cores[0].close.mockImplementation(() => { throw new Error('gone'); });
    t.cores[0].scans.shift().reject(new Error('Xid 79'));
    await flush();
    expect(t.logs).toContain('error neb: card 0 stopped: its core failed: Xid 79');
    expect(t.engine.isRunning()).toBe(true);
    expect(t.events.filter((e) => e.type === 'status' && e.gpuIndex === 0).pop().hashrate).toBe(0);
    t.engine.stop();
    expect(t.cores[1].close).toHaveBeenCalled();
  });

  test('a model that fails to load takes its card off', async () => {
    const t = setup();
    await loggedIn(t);
    t.cores[0].models.shift().reject('no memory for the table');
    await flush();
    expect(t.logs).toContain('error neb: card 0 stopped: its core failed: no memory for the table');
  });

  test('a core whose answer is not a result takes its card off', async () => {
    const t = setup();
    await mining(t);
    t.cores[0].scans.shift().resolve({ evaluated: 1 });
    await flush();
    expect(t.logs.some((l) => /card 0 stopped: its core failed: /.test(l))).toBe(true);
  });

  test('a card loop that throws outside a core call is caught too', async () => {
    const t = setup();
    t.engine._liveJob = () => { throw new Error('boom'); };
    t.engine.start(t.settings);
    await flush();
    expect(t.logs).toContain('error neb: card 0 stopped: it answered with boom');
  });
});

describe('stopping', () => {
  test('stop closes the cores and the socket, and what a core answers afterwards is dropped', async () => {
    const t = setup({ cards: [0, 1] });
    const stopped = [];
    t.engine.on('stopped', (c) => stopped.push(c));
    await loggedIn(t);
    t.cores[0].loaded();
    await flush();
    t.sock().destroy.mockImplementation(() => { throw new Error('already closed'); });
    expect(t.engine.stop()).toBe(true);
    expect(stopped).toEqual([0]);
    expect(t.cores.every((c) => c.close.mock.calls.length === 1)).toBe(true);
    // Card 0 was scanning and card 1 loading its model: both answers arrive after the stop.
    t.cores[0].answer([5]);
    t.cores[1].loaded();
    await flush();
    expect(t.cores[1].scan).not.toHaveBeenCalled();
    expect(t.engine.snapshot()).toMatchObject({ state: 'stopped', hashrate: 0 });
    expect(t.engine.poolConnected()).toBe(false);
  });

  test('a scan that fails after the stop is not an error', async () => {
    const t = setup();
    await mining(t);
    t.engine.stop();
    t.cores[0].scans.shift().reject(new Error('the core is closed'));
    await flush();
    expect(t.logs.some((l) => /stopped: its core failed/.test(l))).toBe(false);
  });

  test('stopping between sockets', () => {
    const t = setup();
    t.engine.start(t.settings);
    t.sock().emit('close');
    expect(t.engine.stop()).toBe(true);
  });
});
