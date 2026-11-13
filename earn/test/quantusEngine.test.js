'use strict';

const { EventEmitter } = require('events');
const E = require('../src/main/quantusEngine');
const qc = require('../src/main/quantusCore');
const q = require('../src/shared/quantus/qpow');

const { QuantusEngine } = E;

// Real setImmediate, so promises can be flushed; fake intervals and timeouts,
// so reconnects and keepalives run when a test says.
beforeEach(() => jest.useFakeTimers({ doNotFake: ['setImmediate', 'nextTick', 'queueMicrotask'] }));
afterEach(() => jest.useRealTimers());

const flush = async (n = 3) => { for (let i = 0; i < n; i++) await new Promise((r) => setImmediate(r)); };

const ADDRESS = 'qzjTEdq3Wd1d98kaeE1RTDPBBYBfY6XpfCtaLq97wmfcACqCd';
const HASH = '415c416d00ae68342369dbc22db3b8f805a1b1a0ea0848cf08116ab894e46753';
const EASY = 'ff'.repeat(64); // every hash but the very top is a share

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

// A core whose scans wait until the test answers them.
function heldCore(index) {
  const core = {
    device: { index, name: 'Card ' + index, vendor: 'fake' },
    scans: [],
    scan: jest.fn((w) => new Promise((resolve, reject) => core.scans.push({ w, resolve, reject }))),
    close: jest.fn(),
  };
  core.answer = (counters, extra = {}) => core.scans.shift().resolve({ hits: Buffer.concat(counters.map(qc.u64Bytes)), overflow: false, ...extra });
  return core;
}

function setup({ cards = [0], timing = {}, host = 'us2.quantus.herominers.com', factory } = {}) {
  const cores = cards.map(heldCore);
  const socks = [];
  let clock = 1000000;
  const engine = new QuantusEngine({
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
  return { engine, cores, socks, sock, logs, events, tick: (ms) => { clock += ms; }, settings: { address: ADDRESS, worker: 'rig01', endpoint: host + ':1155' } };
}

function job(over = {}) {
  return { job_id: 'j1', mining_hash: HASH, target: EASY, difficulty: 1, extranonce: '0004ee45', ...over };
}

async function loggedIn(t, jobOver) {
  t.engine.start(t.settings);
  t.sock().emit('connect');
  t.sock().say({ id: 1, result: { id: 'sess-1', status: 'OK', job: job(jobOver) } });
  await flush();
}

describe('starting', () => {
  test('with no core: false, and says why', () => {
    const t = setup({ factory: null });
    const errors = [];
    t.engine.on('error', (e) => errors.push(e.message));
    expect(t.engine.start(t.settings)).toBe(false);
    expect(errors).toEqual(['there is no Quantus core in this build']);
    expect(t.engine.isRunning()).toBe(false);
  });

  test('when no card opens: false', () => {
    const t = setup({ factory: { devices: [{ index: 0 }], createCore: () => { throw new Error('out of memory'); } } });
    expect(t.engine.start()).toBe(false);
    expect(t.logs).toEqual(['error quantus: card 0 would not open: out of memory', 'error quantus: no card would open a Quantus core']);
  });

  test('a card that will not open is skipped; the rest mine; a core with no name gets one', async () => {
    const t = setup({
      factory: {
        devices: [{ index: 0 }, { index: 1 }],
        createCore: (i) => { if (i === 0) throw 'busy'; return { scan: () => new Promise(() => {}), close: jest.fn() }; },
      },
    });
    expect(t.engine.start(t.settings)).toBe(true);
    expect(t.logs).toContain('error quantus: card 0 would not open: busy');
    expect(t.logs).toContain('info quantus: card 1: card 1');
    expect(t.engine.start(t.settings)).toBe(false);   // already running
    t.engine.stop();
  });

  test('the login: address, worker and agent; on Kryptex, address/worker', () => {
    const t = setup({ cards: [0] });
    t.engine.start({ ...t.settings, cards: [0] });
    expect(t.engine.connect).toHaveBeenCalledWith('us2.quantus.herominers.com', 1155);
    t.sock().emit('connect');
    expect(t.sock().lines).toEqual([{
      id: 1, jsonrpc: '2.0', method: 'login',
      params: { login: ADDRESS, pass: 'rig01', agent: 'llmjob-earn/9.9.9', rigid: 'rig01' },
    }]);
    expect(t.logs).toContain('info quantus: mining Quantus for qzjTEdq3…cACqCd on 1 card · us2.quantus.herominers.com:1155');
    t.engine.stop();

    const k = setup({ host: 'qtc-us.kryptex.network' });
    k.engine.start({ address: ADDRESS, endpoint: 'qtc-us.kryptex.network:7049' });
    k.sock().emit('connect');
    expect(k.sock().lines[0].params.login).toBe(ADDRESS + '/x');
    expect(k.logs).toContain('info quantus: connecting to qtc-us.kryptex.network:7049 · worker x');
    k.engine.stop();
  });

  test('no endpoint: it still runs, and the connect fails on its own', () => {
    const t = setup();
    expect(t.engine.start({ address: ADDRESS })).toBe(true);
    expect(t.engine.connect).toHaveBeenCalledWith('', NaN);
    t.engine.stop();
  });

  test('defaults: the clock, random lanes, the agent, the timings', () => {
    const e = new QuantusEngine();
    expect(e.agent).toBe('llmjob-earn');
    expect(e.t).toEqual({
      reconnect: E.RECONNECT_MS, loginRetry: E.LOGIN_RETRY_MS, keepalive: E.KEEPALIVE_MS, silence: E.SILENCE_MS,
      jobSilence: E.JOB_SILENCE_MS, watchdog: E.WATCHDOG_MS, status: E.STATUS_MS, batchMs: E.BATCH_MS, firstBatch: E.FIRST_BATCH,
    });
    expect(e.random(6)).toHaveLength(6);
    expect(typeof e.now()).toBe('number');
    expect(e.poolConnected()).toBe(false);
    expect(e.snapshot()).toEqual({ state: 'stopped', hashrate: 0, accepted: 0, rejected: 0, cards: 0, poolError: null });
    expect(e.stop()).toBe(false);
  });
});

describe('mining a job', () => {
  test('each card scans the job: the midstate of header and extranonce, its own lane, from counter 0', async () => {
    const t = setup({ cards: [0, 2] });
    const authorized = [];
    t.engine.on('authorized', (a) => authorized.push(a));
    await loggedIn(t);
    expect(authorized).toEqual([{ host: 'us2.quantus.herominers.com', address: '203.0.113.5', family: 'IPv4' }]);
    expect(t.events.filter((e) => e.type === 'connected')).toEqual([
      { type: 'connected', gpuIndex: 0, endpoint: 'us2.quantus.herominers.com:1155', gpu: 'Card 0' },
      { type: 'connected', gpuIndex: 2, endpoint: 'us2.quantus.herominers.com:1155', gpu: 'Card 2' },
    ]);
    const high = Buffer.alloc(32);
    Buffer.from('0004ee45', 'hex').copy(high);
    const w = t.cores[1].scans[0].w;
    expect(w.midstate).toEqual(q.midstateBytes(q.midstate(Buffer.from(HASH, 'hex'), high)));
    expect(w.tail.toString('hex')).toBe('00'.repeat(16) + 'abababababab0002');
    expect(w.start).toEqual(qc.u64Bytes(0n));
    expect(w.count).toBe(E.FIRST_BATCH);
    expect(w.targetHigh).toEqual(qc.u64Bytes(q.U64_MAX));
    expect(w.maxHits).toBe(E.MAX_HITS);
    expect(t.logs).toContain('info quantus: logged in to the pool');
    expect(t.logs).toContain('info quantus: pool difficulty 1');
    expect(t.engine.poolConnected()).toBe(true);
    t.engine.stop();
  });

  test('a hit is checked on the full hash, submitted once, and counted when the pool takes it', async () => {
    const t = setup();
    await loggedIn(t);
    const core = t.cores[0];
    // Counter 3 twice: the second is dropped.
    t.tick(50);
    core.answer([3n, 3n]);
    await flush();
    const nonce = q.buildNonce(Buffer.from('0004ee45', 'hex'), Buffer.from('abababababab0000', 'hex'), 3n).toString('hex');
    expect(t.sock().lines[1]).toEqual({ id: 10, jsonrpc: '2.0', method: 'submit', params: { id: 'sess-1', job_id: 'j1', nonce } });
    expect(t.sock().lines).toHaveLength(2);
    expect(t.logs).toContain('warn quantus: dropping a share already sent (job j1)');
    // The next batch starts where this one ended, sized for the batch time.
    expect(core.scans[0].w.start).toEqual(qc.u64Bytes(256n));
    expect(core.scans[0].w.count).toBe(512);   // 256 nonces in 50 ms; 100 ms wanted
    t.sock().say({ id: 10, result: { status: 'OK' }, error: null });
    expect(t.logs).toContain('info quantus: QTC share accepted (card 0)');
    expect(t.events.filter((e) => e.type === 'status').pop()).toEqual({
      type: 'status', gpuIndex: 0, hashrate: 5120, accepted: 1, rejected: 0, gpu: 'Card 0', coin: 'quantus',
    });
    expect(t.engine.snapshot()).toEqual({ state: 'mining', hashrate: 5120, accepted: 1, rejected: 0, cards: 1, poolError: null });
    // A reply to nothing we sent changes nothing.
    t.sock().say({ id: 99, result: { status: 'OK' } });
    t.sock().say({ id: 98, error: { code: -1, message: 'x' } });
    expect(t.engine.snapshot().accepted).toBe(1);
    t.engine.stop();
  });

  test('a rejected share is counted and logged', async () => {
    const t = setup();
    await loggedIn(t);
    t.cores[0].answer([1n]);
    await flush();
    t.sock().say({ id: 10, result: null, error: { code: -1, message: 'Low difficulty share' } });
    expect(t.logs).toContain('error quantus: QTC share rejected (card 0): [-1] Low difficulty share');
    expect(t.engine.snapshot().rejected).toBe(1);
    t.engine.stop();
  });

  test('the batch grows at most 16x a step, and the rate is smoothed', async () => {
    const t = setup();
    await loggedIn(t);
    t.tick(1);
    t.cores[0].answer([]);
    await flush();
    expect(t.cores[0].scans[0].w.count).toBe(256 * 16);
    t.tick(4096 * 1000 / 1e6);    // a second batch at 1 MH/s
    t.cores[0].answer([]);
    await flush();
    expect(t.engine.cards[0].rate).toBeCloseTo(256000 * 0.7 + 1e6 * 0.3);
    t.engine.stop();
  });

  test('hits of a batch whose job was replaced are dropped', async () => {
    const t = setup();
    await loggedIn(t);
    t.sock().say({ jsonrpc: '2.0', method: 'job', params: { clean_jobs: true, job: job({ job_id: 'j2' }) } });
    t.cores[0].answer([1n]);
    await flush();
    expect(t.sock().lines).toHaveLength(1);
    // The next scan is the new job's.
    expect(t.cores[0].scans).toHaveLength(1);
    t.engine.stop();
  });

  test('the same job again changes nothing; a new difficulty is logged', async () => {
    const t = setup();
    await loggedIn(t);
    const first = t.engine.job;
    t.sock().say({ jsonrpc: '2.0', method: 'job', params: { job: job() } });
    expect(t.engine.job).toBe(first);
    // The same job id with a new target keeps the sent shares, and says the difficulty.
    t.sock().say({ jsonrpc: '2.0', method: 'job', params: { job: job({ target: undefined, difficulty: 5 }) } });
    expect(t.engine.job).not.toBe(first);
    expect(t.logs).toContain('info quantus: pool difficulty 5');
    const n = t.logs.length;
    t.sock().say({ jsonrpc: '2.0', method: 'job', params: { job: job({ job_id: 'j3', target: undefined, difficulty: 5 }) } });
    expect(t.logs).toHaveLength(n);
    t.engine.stop();
  });

  test('a job it cannot mine: the cards wait for the next', async () => {
    const t = setup();
    await loggedIn(t);
    t.sock().say({ jsonrpc: '2.0', method: 'job', params: { job: job({ algo: 'kawpow' }) } });
    expect(t.logs).toContain('error quantus: ignoring a job this miner cannot mine (j1): algorithm kawpow is not one this miner mines');
    t.cores[0].answer([1n]);
    await flush();
    expect(t.cores[0].scans).toHaveLength(0);
    t.sock().say({ jsonrpc: '2.0', method: 'job', params: { job: { job_id: '' } } });
    expect(t.logs).toContain('error quantus: ignoring a job this miner cannot mine: no job_id');
    t.sock().say({ jsonrpc: '2.0', method: 'job', params: { job: job({ job_id: 'j9' }) } });
    await flush();
    expect(t.cores[0].scans).toHaveLength(1);
    t.engine.stop();
  });

  test('a tie on the top 64 bits that loses on the rest is not a bad hit', async () => {
    const t = setup();
    // The target is the hash of counter 5, with its low 448 bits cleared.
    const nonce = q.buildNonce(Buffer.from('0004ee45', 'hex'), Buffer.from('abababababab0000', 'hex'), 5n);
    const top = q.toBig(q.nonceHash(Buffer.from(HASH, 'hex'), nonce)) >> 448n;
    await loggedIn(t, { target: (top << 448n).toString(16).padStart(128, '0'), difficulty: undefined });
    t.cores[0].answer([5n]);
    await flush();
    expect(t.sock().lines).toHaveLength(1);
    expect(t.engine.cards[0].bad).toBe(0);
    t.engine.stop();
  });

  test('a core that keeps reporting nonces that are not hits loses its card; the last card ends the miner', async () => {
    const t = setup({ cards: [0, 1] });
    const stopped = [];
    const errors = [];
    t.engine.on('stopped', (c) => stopped.push(c));
    t.engine.on('error', (e) => errors.push(e.message));
    // Difficulty 2^64: the target's top 64 bits are 0.
    await loggedIn(t, { target: undefined, difficulty: '18446744073709551616' });
    t.cores[0].answer([1n, 2n, 3n, 4n]);
    await flush();
    expect(t.logs).toContain('warn quantus: card 0 reported a nonce that is not a hit (3 of 3)');
    expect(t.logs).toContain('error quantus: card 0 stopped: its core keeps reporting nonces that are not hits');
    expect(t.cores[0].close).toHaveBeenCalled();
    expect(t.cores[0].scans).toHaveLength(0);
    expect(t.engine.isRunning()).toBe(true);
    // The other card's core throws: that is the last card.
    t.cores[1].scans.shift().reject(new Error('device lost'));
    await flush();
    expect(t.logs).toContain('error quantus: card 1 stopped: its core failed: device lost');
    expect(errors).toEqual(['every card stopped; the Quantus miner gives up']);
    expect(stopped).toEqual([1]);
    expect(t.engine.isRunning()).toBe(false);
  });

  test('a core that answers nonsense loses its card; one that throws on close is still let go', async () => {
    const t = setup({ cards: [0, 1] });
    t.cores[0].close.mockImplementation(() => { throw new Error('gone'); });
    await loggedIn(t);
    t.cores[0].scans.shift().resolve(null);
    await flush();
    expect(t.logs.find((l) => l.startsWith('error quantus: card 0 stopped: it answered a scan with'))).toBeTruthy();
    // Stopping skips the dead card.
    t.engine.stop();
    expect(t.cores[0].close).toHaveBeenCalledTimes(1);
    expect(t.cores[1].close).toHaveBeenCalledTimes(1);
  });

  test('more hits than maxHits is said', async () => {
    const t = setup();
    await loggedIn(t);
    t.cores[0].answer([], { overflow: true });
    await flush();
    expect(t.logs).toContain('warn quantus: card 0 found more than ' + E.MAX_HITS + ' hits in one batch');
    t.engine.stop();
  });

  test('a scan that fails after the stop is not a card failure', async () => {
    const t = setup();
    await loggedIn(t);
    const pending = t.cores[0].scans.shift();
    t.engine.stop();
    pending.reject(new Error('closed'));
    await flush();
    expect(t.logs.some((l) => l.includes('stopped:'))).toBe(false);
    // And one that answers after the stop is ignored.
    const u = setup();
    await loggedIn(u);
    const p2 = u.cores[0].scans.shift();
    u.engine.stop();
    p2.resolve({ hits: qc.u64Bytes(1n), overflow: false });
    await flush();
    expect(u.sock()).toBeTruthy();
    expect(u.engine.cards[0].counter).toBe(0n);
  });
});

describe('the pool connection', () => {
  test('a refused login: retried after the long wait', async () => {
    const t = setup({ timing: { loginRetry: 60000, reconnect: 5000 } });
    t.engine.start(t.settings);
    t.sock().emit('connect');
    t.sock().say({ id: 1, result: null, error: { code: -20, message: 'Invalid Quantus address' } });
    expect(t.logs).toContain('error quantus: the pool refused the login: [-20] Invalid Quantus address');
    expect(t.engine.snapshot().poolError).toBe('the Quantus pool refused the login: [-20] Invalid Quantus address');
    expect(t.logs).toContain('info quantus: pool connection closed; reconnecting in 60 s');
    jest.advanceTimersByTime(59999);
    expect(t.engine.connect).toHaveBeenCalledTimes(1);
    jest.advanceTimersByTime(1);
    expect(t.engine.connect).toHaveBeenCalledTimes(2);
    t.engine.stop();
  });

  test('closed before a login: retried soon, and said once', () => {
    const t = setup({ timing: { reconnect: 5000 } });
    t.engine.start(t.settings);
    t.sock().emit('close');
    expect(t.engine.snapshot().poolError).toBe('the Quantus pool closed the connection before the login');
    jest.advanceTimersByTime(5000);
    t.sock().emit('close');
    expect(t.logs.filter((l) => l.includes('reconnecting in 5 s'))).toHaveLength(1);
    t.engine.stop();
  });

  test('a connect that throws is retried', () => {
    const t = setup({ timing: { reconnect: 5000 } });
    let first = true;
    const real = t.engine.connect;
    t.engine.connect = jest.fn((...a) => { if (first) { first = false; throw new Error('bad host'); } return real(...a); });
    t.engine.start(t.settings);
    expect(t.logs).toContain('error quantus: pool connection failed: bad host');
    jest.advanceTimersByTime(5000);
    expect(t.engine.connect).toHaveBeenCalledTimes(2);
    t.engine.stop();
  });

  test('socket errors: connect-failed, with dns; the same error logged once', () => {
    const t = setup();
    t.engine.start(t.settings);
    t.sock().emit('error', new Error('getaddrinfo ENOTFOUND x'));
    t.engine.poolFails = 1;
    t.sock().emit('error', new Error('getaddrinfo ENOTFOUND x'));
    t.sock().emit('error', new Error('ECONNREFUSED'));
    expect(t.logs.filter((l) => l.startsWith('error quantus: pool socket error'))).toEqual([
      'error quantus: pool socket error: getaddrinfo ENOTFOUND x', 'error quantus: pool socket error: ECONNREFUSED',
    ]);
    expect(t.events.filter((e) => e.type === 'connect-failed').map((e) => e.dns)).toEqual([true, true, false]);
    t.engine.stop();
  });

  test('a dropped connection: the cards stop on the job, the board keeps the rate for the grace, then logs in again', async () => {
    const t = setup({ timing: { reconnect: 5000 } });
    await loggedIn(t);
    const first = t.sock();
    first.emit('close');
    expect(t.engine.job).toBe(null);
    expect(t.engine.poolConnected()).toBe(true);
    t.tick(E.POOL_GRACE_MS);
    expect(t.engine.poolConnected()).toBe(false);
    // The batch in flight ends with nothing to mine.
    t.cores[0].answer([1n]);
    await flush();
    expect(first.lines).toHaveLength(1);
    // Data from the old socket is ignored.
    first.emit('data', '{"id":1}\n');
    jest.advanceTimersByTime(5000);
    t.sock().emit('connect');
    t.sock().say({ id: 1, result: { id: 'sess-2', job: job({ job_id: 'k1' }) } });
    await flush();
    expect(t.cores[0].scans).toHaveLength(1);
    t.engine.stop();
  });

  test('keepalives: sent while logged in, answered, capped at 16 waiting', async () => {
    const t = setup({ timing: { keepalive: 1000 } });
    t.engine.start(t.settings);
    jest.advanceTimersByTime(1000);
    expect(t.sock().lines).toHaveLength(0);   // not logged in yet
    t.sock().emit('connect');
    t.sock().say({ id: 1, result: { id: 's' } });
    jest.advanceTimersByTime(1000);
    expect(t.sock().lines[1]).toEqual({ id: 10, jsonrpc: '2.0', method: 'keepalived', params: { id: 's' } });
    t.sock().say({ id: 10, result: { status: 'KEEPALIVED' } });
    expect(t.engine.keepaliveIds.size).toBe(0);
    jest.advanceTimersByTime(17000);
    expect(t.engine.keepaliveIds.size).toBe(16);
    t.engine.stop();
  });

  test('errors that mean the session is gone log in again; a ban waits the long wait', async () => {
    const t = setup({ timing: { keepalive: 1000, reconnect: 5000, loginRetry: 60000 } });
    await loggedIn(t);
    jest.advanceTimersByTime(1000);
    t.sock().say({ id: 10, error: { code: -1, message: 'Unauthenticated' } });
    expect(t.logs).toContain('warn quantus: the pool dropped this session ([-1] Unauthenticated); logging in again');
    jest.advanceTimersByTime(5000);
    expect(t.engine.connect).toHaveBeenCalledTimes(2);
    t.sock().emit('connect');
    t.sock().say({ id: 1, result: { id: 's2' } });
    t.sock().say({ id: null, error: { code: -1, message: 'Your IP is banned' } });
    expect(t.logs).toContain('error quantus: pool: [-1] Your IP is banned');
    jest.advanceTimersByTime(59999);
    expect(t.engine.connect).toHaveBeenCalledTimes(2);
    jest.advanceTimersByTime(1);
    expect(t.engine.connect).toHaveBeenCalledTimes(3);
    // A rejected share for a lost session, and other errors that are not.
    t.sock().emit('connect');
    t.sock().say({ id: 1, result: { id: 's3' } });
    t.sock().say({ id: null, error: { code: 1, message: 'slow down' } });
    t.sock().say({ id: 55, error: { code: 1, message: 'invalid job id' } });
    jest.advanceTimersByTime(5000);
    expect(t.engine.connect).toHaveBeenCalledTimes(4);
    t.engine.stop();
  });

  test('lines it does not know are logged; lines may arrive in pieces', async () => {
    const t = setup();
    t.engine.start(t.settings);
    t.sock().emit('data', 'not json\n{"method":"mining.set_dif');
    t.sock().emit('data', 'ficulty"}\n\n');
    expect(t.logs).toContain('info quantus: pool: not json');
    expect(t.logs).toContain('info quantus: pool: {"method":"mining.set_difficulty"}');
    t.engine.stop();
  });

  test('a silent pool, or one with no new job, is left and joined again', async () => {
    const t = setup({ timing: { watchdog: 1000, silence: 300000, jobSilence: 180000, reconnect: 1000 } });
    await loggedIn(t);
    t.tick(179000);
    jest.advanceTimersByTime(1000);
    expect(t.engine.connect).toHaveBeenCalledTimes(1);
    t.tick(1000);
    jest.advanceTimersByTime(1000);
    expect(t.logs).toContain('warn quantus: no new job from the pool in 3 minutes; reconnecting');
    jest.advanceTimersByTime(1000);
    expect(t.engine.connect).toHaveBeenCalledTimes(2);
    t.tick(300000);
    jest.advanceTimersByTime(1000);
    expect(t.logs).toContain('warn quantus: no word from the pool in 5 minutes; reconnecting');
    // With no socket the watchdog has nothing to do.
    t.engine.sock = null;
    jest.advanceTimersByTime(1000);
    t.engine.stop();
  });

  test('a status for every card each second; 0 while off the pool', async () => {
    const t = setup({ cards: [0, 1], timing: { status: 1000 } });
    t.engine.start(t.settings);
    jest.advanceTimersByTime(1000);
    expect(t.events.filter((e) => e.type === 'status').map((e) => [e.gpuIndex, e.hashrate])).toEqual([[0, 0], [1, 0]]);
    t.engine.stop();
  });

  test('stop: cores closed, timers gone, stopped 0', async () => {
    const t = setup();
    const stopped = [];
    t.engine.on('stopped', (c) => stopped.push(c));
    await loggedIn(t);
    const s = t.sock();
    s.destroy.mockImplementation(() => { throw new Error('already closed'); });
    expect(t.engine.stop()).toBe(true);
    expect(stopped).toEqual([0]);
    expect(t.cores[0].close).toHaveBeenCalled();
    expect(jest.getTimerCount()).toBe(0);
    expect(t.engine.snapshot().state).toBe('stopped');
  });
});

describe('end to end on the CPU core', () => {
  test('a job, the reference core, and a share whose hash meets the target', async () => {
    jest.useRealTimers();
    const factory = qc.quantusCoreFactory({ env: { QUANTUS_CORE: 'cpu' } });
    const socks = [];
    const engine = new QuantusEngine({
      connect: () => { const s = fakeSocket(); socks.push(s); return s; },
      factory,
      timing: { firstBatch: 64, batchMs: 20, keepalive: 1e8, watchdog: 1e8, status: 1e8 },
    });
    engine.start({ address: ADDRESS, worker: 'e2e', endpoint: 'us2.quantus.herominers.com:1155', cards: [0] });
    const sock = socks[0];
    sock.emit('connect');
    // Difficulty 16: about one nonce in 16 is a share.
    sock.say({ id: 1, result: { id: 'sess', job: job({ target: undefined, difficulty: 16 }) } });
    const deadline = Date.now() + 20000;
    while (sock.lines.length < 2 && Date.now() < deadline) await new Promise((r) => setTimeout(r, 5));
    engine.stop();
    const submit = sock.lines[1];
    expect(submit.method).toBe('submit');
    const nonce = Buffer.from(submit.params.nonce, 'hex');
    expect(nonce.subarray(0, 4).toString('hex')).toBe('0004ee45');
    // Checked here on the plain reference, not the midstate path the core used.
    expect(q.meetsTarget(q.nonceHash(Buffer.from(HASH, 'hex'), nonce), q.targetFromDifficulty(16n))).toBe(true);
  }, 30000);
});
