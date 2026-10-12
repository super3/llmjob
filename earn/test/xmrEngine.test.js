'use strict';

const { EventEmitter } = require('events');
const {
  XmrEngine, RECONNECT_MS, LOGIN_RETRY_MS, KEEPALIVE_MS, SILENCE_MS, JOB_SILENCE_MS, RESTART_MS, RESTART_LIMIT, STABLE_MS,
  KILL_MS, POOL_GRACE_MS,
} = require('../src/main/xmrEngine');

const ADDRESS = '87wTMj88rEGV3BFiYXJc2UHeTGmHikMRWAvAskmPgaF3PZxuxonkKaJZsprNmuhM1Aca7P6UbyatGFHkQfUTYgoNN1JxND9';
const SEED = 'ab'.repeat(32);
const BLOB = 'cd'.repeat(76);
const RESULT = 'ef'.repeat(32);
const SETTINGS = { address: ADDRESS, worker: 'rig01', endpoint: 'de.monero.herominers.com:1111', threads: 6 };

// A socket that records what is written to it.
function fakeSocket() {
  const s = new EventEmitter();
  s.written = [];
  s.write = (line) => { s.written.push(JSON.parse(line)); };
  s.destroy = jest.fn(() => { s.destroyed = true; s.emit('close'); });
  s.remoteAddress = '141.95.126.31';
  s.remoteFamily = 'IPv4';
  s.recv = (o) => s.emit('data', Buffer.from(JSON.stringify(o) + '\n'));
  return s;
}

// A worker process: stdout/stderr to push lines on, stdin that records.
function fakeChild() {
  const c = new EventEmitter();
  c.stdout = new EventEmitter();
  c.stderr = new EventEmitter();
  c.stdin = new EventEmitter();
  c.stdin.lines = [];
  c.stdin.write = (line) => { c.stdin.lines.push(JSON.parse(line)); };
  c.stdin.end = jest.fn();
  c.kill = jest.fn();
  c.say = (o) => c.stdout.emit('data', Buffer.from(JSON.stringify(o) + '\n'));
  return c;
}

const HELLO = { type: 'hello', protocol: 1, version: '0.5.14', randomx: '7607fb2faed2', jit: true, hardAes: true };
const job = (id, extra) => Object.assign({ blob: BLOB, job_id: id, target: 'b88d0600', algo: 'rx/0', height: 1, seed_hash: SEED }, extra);

// An engine with fakes for everything, and what it said.
function setup(opts = {}) {
  const sockets = [];
  const children = [];
  let t = 1000000;
  const engine = new XmrEngine(Object.assign({
    connect: jest.fn(() => { const s = fakeSocket(); sockets.push(s); return s; }),
    spawnWorker: jest.fn(() => { const c = fakeChild(); children.push(c); return c; }),
    workerPath: '/opt/llmjob-xmr-worker',
    agent: 'llmjob-earn/0.5.14',
    now: () => t,
    platform: 'linux',
  }, opts));
  const logs = [];
  const events = [];
  const errors = [];
  const stopped = [];
  engine.on('log', (l) => logs.push(l));
  engine.on('event', (e) => events.push(e));
  engine.on('error', (e) => errors.push(e.message));
  engine.on('stopped', (c) => stopped.push(c));
  const lines = () => logs.map((l) => l.line);
  return {
    engine, sockets, children, logs, lines, events, errors, stopped,
    advance: (ms) => { t += ms; jest.advanceTimersByTime(ms); },
    setNow: (v) => { t = v; },
    sock: () => sockets[sockets.length - 1],
    child: () => children[children.length - 1],
  };
}

// Started, logged in with a job, and the worker through its dataset.
function mining(opts) {
  const h = setup(opts);
  h.engine.start(SETTINGS);
  h.sock().emit('connect');
  h.sock().recv({ id: 1, jsonrpc: '2.0', error: null, result: { id: 'sess', job: job('j1'), status: 'OK' } });
  h.child().say(HELLO);
  h.child().say({ type: 'selftest', ok: true, ms: 1800 });
  h.child().say({ type: 'dataset', state: 'building', threads: 6, pages: 'thp' });
  h.child().say({ type: 'dataset', state: 'ready', ms: 42268, pages: 'thp' });
  return h;
}

beforeEach(() => jest.useFakeTimers());
afterEach(() => jest.useRealTimers());

describe('start', () => {
  test('spawns the worker with the plan\'s threads and logs in to the pool', () => {
    const h = setup();
    expect(h.engine.start(SETTINGS)).toBe(true);
    expect(h.engine.isRunning()).toBe(true);
    expect(h.engine.spawnWorker).toHaveBeenCalledWith('/opt/llmjob-xmr-worker', ['--threads', '6', '--priority', 'low']);
    expect(h.engine.connect).toHaveBeenCalledWith('de.monero.herominers.com', 1111);
    h.sock().emit('connect');
    expect(h.sock().written[0]).toEqual({
      id: 1, jsonrpc: '2.0', method: 'login',
      params: { login: ADDRESS, pass: 'rig01', agent: 'llmjob-earn/0.5.14', rigid: 'rig01', algo: ['rx/0'] },
    });
    expect(h.lines()).toEqual([
      'cpu: mining Monero for 87wTMj88…1JxND9 on 6 CPU threads at low priority · de.monero.herominers.com:1111',
      'cpu: connecting to de.monero.herominers.com:1111 · worker rig01',
    ]);
    expect(h.engine.snapshot().state).toBe('starting');
    expect(h.engine.start(SETTINGS)).toBe(false);   // already running
  });

  test('one thread, and no worker name', () => {
    const h = setup();
    h.engine.start(Object.assign({}, SETTINGS, { threads: 1, worker: '' }));
    h.sock().emit('connect');
    expect(h.lines()[0]).toMatch(/ on 1 CPU thread at low priority · /);
    expect(h.lines()[1]).toBe('cpu: connecting to de.monero.herominers.com:1111 · worker x');
  });

  test('the priority: the one asked for, or the platform\'s default', () => {
    const h = setup({ platform: 'win32' });
    h.engine.start(SETTINGS);
    expect(h.engine.spawnWorker).toHaveBeenLastCalledWith('/opt/llmjob-xmr-worker', ['--threads', '6', '--priority', 'idle']);
    h.engine.stop();
    h.engine.start(Object.assign({}, SETTINGS, { priority: 'normal' }));
    expect(h.engine.spawnWorker).toHaveBeenLastCalledWith('/opt/llmjob-xmr-worker', ['--threads', '6', '--priority', 'normal']);
    expect(h.lines()).toContain('cpu: mining Monero for 87wTMj88…1JxND9 on 6 CPU threads at normal priority · de.monero.herominers.com:1111');
  });

  test('without a worker file there is nothing to start', () => {
    const h = setup({ workerPath: null });
    expect(h.engine.start(SETTINGS)).toBe(false);
    expect(h.errors[0]).toBe('there is no CPU miner (llmjob-xmr-worker) beside the app: this build has none,'
      + ' or an antivirus removed it.');
    expect(h.lines()).toEqual(['cpu: ' + h.errors[0]]);
    expect(h.engine.isRunning()).toBe(false);
    expect(h.engine.snapshot()).toMatchObject({ state: 'stopped', hashrate: 0, threads: 6 });
    const bare = new XmrEngine({ workerPath: '/w' });
    bare.on('error', () => {});
    expect(bare.start()).toBe(false);
  });

  test('a worker that cannot be spawned stops the CPU miner', () => {
    const h = setup({ spawnWorker: () => { throw new Error('EACCES'); } });
    expect(h.engine.start(SETTINGS)).toBe(false);
    expect(h.errors).toEqual(['could not start /opt/llmjob-xmr-worker: EACCES']);
    expect(h.stopped).toEqual([1]);
    expect(h.engine.snapshot().state).toBe('failed');
    const h2 = setup({ spawnWorker: () => { throw 'nope'; } });
    h2.engine.start(SETTINGS);
    expect(h2.errors).toEqual(['could not start /opt/llmjob-xmr-worker: nope']);
  });

  test('defaults: timings, agent and clock', () => {
    const e = new XmrEngine();
    expect(e.t).toEqual({
      reconnect: RECONNECT_MS, loginRetry: LOGIN_RETRY_MS, keepalive: KEEPALIVE_MS, silence: SILENCE_MS,
      jobSilence: JOB_SILENCE_MS, watchdog: 30000, restart: RESTART_MS, kill: KILL_MS,
    });
    const t = { reconnect: 1, loginRetry: 2, keepalive: 3, silence: 4, jobSilence: 8, watchdog: 5, restart: 6, kill: 7 };
    expect(new XmrEngine({ timing: t }).t).toEqual(t);
    expect(e.agent).toBe('llmjob-earn');
    expect(e.now).toBe(Date.now);
    expect(e.platform).toBe(process.platform);
    expect(e.snapshot()).toMatchObject({ threads: 0, state: 'stopped' });
    expect([RECONNECT_MS, RESTART_LIMIT, STABLE_MS, POOL_GRACE_MS]).toEqual([5000, 3, 600000, 30000]);
  });
});

describe('the worker', () => {
  test('logs its hello, self-test and dataset, then mines', () => {
    const h = mining();
    expect(h.lines()).toEqual(expect.arrayContaining([
      'cpu: RandomX worker 0.5.14 (RandomX 7607fb2faed2) · JIT · hardware AES',
      'cpu: RandomX self-test passed (1800 ms)',
      'cpu: building the 2080 MiB RandomX dataset on 6 threads (about a minute; again when Monero changes its seed, every 2048 blocks)',
      'cpu: dataset ready in 42.3 s · transparent huge pages',
    ]));
    expect(h.engine.snapshot()).toMatchObject({ state: 'mining', pages: 'thp', difficulty: 10000 });
    expect(h.events.map((e) => e.state)).toEqual(expect.arrayContaining(['starting', 'dataset', 'mining']));
  });

  test('other hellos, one dataset thread, and every kind of pages', () => {
    const h = setup();
    h.engine.start(SETTINGS);
    h.child().say(Object.assign({}, HELLO, { jit: false, hardAes: false }));
    h.child().say({ type: 'dataset', state: 'building', threads: 1 });
    for (const pages of ['hugetlb', 'normal', 'odd']) h.child().say({ type: 'dataset', state: 'ready', ms: 1000, pages });
    expect(h.lines()).toEqual(expect.arrayContaining([
      'cpu: RandomX worker 0.5.14 (RandomX 7607fb2faed2) · no JIT · software AES',
      expect.stringMatching(/dataset on 1 thread \(/),
      'cpu: dataset ready in 1 s · huge pages',
      'cpu: dataset ready in 1 s · no huge pages (hashrate is up to 30% lower without them)',
      'cpu: dataset ready in 1 s · odd',
    ]));
  });

  test('the job goes to the worker as soon as the pool sends it', () => {
    const h = mining();
    expect(h.child().stdin.lines[0]).toEqual({
      cmd: 'job', id: 'j1', blob: BLOB, target: '00068db8bac710cb', seed: SEED, variant: 'v1', commit: false,
      nonce: expect.any(Number),
    });
    h.sock().recv({ jsonrpc: '2.0', method: 'job', params: job('j2') });
    expect(h.child().stdin.lines[1].id).toBe('j2');
    // Same difficulty: said once.
    expect(h.lines().filter((l) => l.startsWith('cpu: pool difficulty'))).toEqual(['cpu: pool difficulty 10000']);
    h.sock().recv({ method: 'job', params: job('j3', { target: '0000000000000001' }) });
    expect(h.lines()).toContain('cpu: pool difficulty 255');
  });

  // Sent again, it would restart the worker's search for nothing.
  test('a job the worker already has is not sent again', () => {
    const h = mining();
    h.sock().recv({ method: 'job', params: job('j1') });
    expect(h.child().stdin.lines).toHaveLength(1);
    // The same id with a new target is a new job.
    h.sock().recv({ method: 'job', params: job('j1', { target: '0000000000000001' }) });
    h.sock().recv({ method: 'job', params: job('j1', { target: '0000000000000001', blob: 'ce'.repeat(76) }) });
    expect(h.child().stdin.lines.map((l) => l.id)).toEqual(['j1', 'j1', 'j1']);
  });

  test('hashrate ticks become status events', () => {
    const h = mining();
    h.child().say({ type: 'hashrate', hs: 1845.2, total: 10, threads: 6 });
    expect(h.events[h.events.length - 1]).toEqual({
      type: 'status', hashrate: 1845.2, accepted: 0, rejected: 0, threads: 6, state: 'mining',
    });
    expect(h.engine.snapshot().hashrate).toBe(1845.2);
  });

  // No share of it can reach a pool, so HiveOS and the Mine tab say 0.
  test('the hashrate reads 0 while the pool is out of reach', () => {
    const h = mining();
    h.child().say({ type: 'hashrate', hs: 4100, total: 10, threads: 6 });
    h.sock().emit('close');
    expect(h.engine.snapshot().hashrate).toBe(4100);   // a routine reconnect
    h.setNow(1000000 + POOL_GRACE_MS);
    expect(h.engine.snapshot().hashrate).toBe(0);
    h.child().say({ type: 'hashrate', hs: 4100, total: 20, threads: 6 });
    expect(h.events[h.events.length - 1]).toMatchObject({ type: 'status', hashrate: 0 });
    h.advance(RECONNECT_MS);
    h.sock().recv({ id: 1, result: { id: 'sess2', job: job('j2') } });
    expect(h.engine.snapshot().hashrate).toBe(4100);
  });

  test('its log lines, errors, stderr and junk reach the log; bench and blank lines don\'t', () => {
    const h = mining();
    h.child().say({ type: 'log', level: 'warn', msg: 'searched every nonce' });
    h.child().say({ type: 'error', fatal: false, msg: 'dropped a share' });
    h.child().say({ type: 'bench', hs: 1 });
    h.child().stdout.emit('data', 'Segmentation fault\n\n');
    h.child().say({ type: 'mystery' });
    h.child().stderr.emit('data', 'missing value for --threads\n\n');
    expect(h.logs.slice(-5)).toEqual([
      { level: 'warn', line: 'cpu: searched every nonce' },
      { level: 'error', line: 'cpu: dropped a share' },
      { level: 'info', line: 'cpu: worker: Segmentation fault' },
      { level: 'info', line: 'cpu: worker: {"type":"mystery"}' },
      { level: 'warn', line: 'cpu: missing value for --threads' },
    ]);
  });

  test('a line split across chunks is read whole', () => {
    const h = setup();
    h.engine.start(SETTINGS);
    const line = JSON.stringify({ type: 'selftest', ok: true, ms: 5 }) + '\n';
    h.child().stdout.emit('data', line.slice(0, 10));
    h.child().stdout.emit('data', line.slice(10));
    expect(h.lines()).toContain('cpu: RandomX self-test passed (5 ms)');
  });

  // A worker from another release is refused, not run: it is not worth a
  // restart, since it would be the same file again.
  test('a worker that speaks another protocol is killed and the CPU miner stops', () => {
    const h = setup();
    h.engine.start(SETTINGS);
    const c = h.child();
    c.stdout.emit('data', JSON.stringify(Object.assign({}, HELLO, { protocol: 2 })) + '\n'
      + JSON.stringify({ type: 'log', msg: 'too late' }) + '\n');
    expect(c.kill).toHaveBeenCalledWith('SIGKILL');
    expect(h.errors[0]).toMatch(/^this llmjob-xmr-worker speaks protocol 2, the app speaks 1/);
    expect(h.lines()).not.toContain('cpu: too late');
    expect(h.stopped).toEqual([1]);
    expect(h.engine.isRunning()).toBe(false);
    c.stdout.emit('data', JSON.stringify({ type: 'log', msg: 'later still' }) + '\n');
    c.emit('exit', null, 'SIGKILL');
    expect(h.lines()).not.toContain('cpu: later still');
  });

  test('a failed self-test stops the CPU miner without a restart', () => {
    const h = setup();
    h.engine.start(SETTINGS);
    h.child().say({ type: 'selftest', ok: false, msg: 'wrong hash for "This is a test"' });
    expect(h.errors).toEqual(['RandomX self-test failed, so nothing was mined: wrong hash for "This is a test"']);
    expect(h.stopped).toEqual([1]);
    expect(h.engine.snapshot().state).toBe('failed');
    h.advance(RESTART_MS * 2);
    expect(h.children).toHaveLength(1);
  });

  // Ctrl+C or a service stop can reach the worker before the app's shutdown
  // runs. The stop that follows must not leave a restart in the log.
  test('a stop soon after the worker dies says nothing about restarting it', () => {
    const h = mining();
    h.child().emit('exit', 130, null);
    h.engine.stop();
    h.advance(RESTART_MS * 2);
    expect(h.lines().filter((l) => l.includes('restarting'))).toEqual([]);
    expect(h.children).toHaveLength(1);
  });

  test('a worker that dies is started again, given the job, and restarts are capped', () => {
    const h = mining();
    h.child().emit('exit', 139, null);
    expect(h.lines()).toContain('cpu: the CPU miner stopped (exit code 139)');
    expect(h.engine.snapshot()).toMatchObject({ state: 'restarting', hashrate: 0, pages: null });
    // Said when it happens, not before.
    expect(h.lines()).not.toContain('cpu: restarting the CPU miner');
    h.advance(RESTART_MS);
    expect(h.lines()).toContain('cpu: restarting the CPU miner');
    expect(h.children).toHaveLength(2);
    // The pool won't resend it. A new start nonce keeps the new worker from
    // finding the old one's shares again.
    expect(h.child().stdin.lines[0].id).toBe('j1');
    expect(h.child().stdin.lines[0].nonce).not.toBe(h.children[0].stdin.lines[0].nonce);

    // Killed: out of memory, or an antivirus.
    h.child().emit('exit', null, 'SIGKILL');
    expect(h.lines()).toContain('cpu: the CPU miner stopped (signal SIGKILL) (killed: out of memory, or an antivirus)');
    h.advance(RESTART_MS);
    h.child().emit('exit', null, 'SIGSEGV');
    h.advance(RESTART_MS);
    expect(h.children).toHaveLength(4);
    h.child().emit('exit', 1, null);
    expect(h.lines()).toContain('cpu: leaving the CPU miner off: it stopped within 10 minutes of each of 3 restarts');
    expect(h.errors).toEqual(['the CPU miner stopped (exit code 1)']);
    expect(h.stopped).toEqual([1]);
    expect(h.engine.snapshot().state).toBe('failed');
  });

  test('a worker that ran a good while starts the restart count over', () => {
    const h = mining();
    for (let i = 0; i < RESTART_LIMIT; i++) {
      h.child().emit('exit', 1, null);
      h.advance(RESTART_MS);
    }
    h.setNow(10 ** 9);   // mined long past STABLE_MS
    h.child().emit('exit', 1, null);
    expect(h.stopped).toEqual([]);
    expect(h.engine.snapshot().state).toBe('restarting');
  });

  test('a fatal worker error ends that worker and restarts it', () => {
    const h = mining();
    const c = h.child();
    c.say({ type: 'error', fatal: true, msg: 'could not allocate the 2080 MiB RandomX dataset: not enough free memory' });
    expect(c.kill).toHaveBeenCalledWith('SIGKILL');
    expect(h.lines()).toContain('cpu: could not allocate the 2080 MiB RandomX dataset: not enough free memory');
    h.advance(RESTART_MS);
    expect(h.children).toHaveLength(2);
  });

  test('a worker file that went missing, or would not run', () => {
    const h = setup();
    h.engine.start(SETTINGS);
    h.child().emit('error', Object.assign(new Error('spawn ENOENT'), { code: 'ENOENT' }));
    expect(h.errors[0]).toBe('the CPU miner is missing (/opt/llmjob-xmr-worker). An antivirus may have removed it; reinstalling puts it back.');
    expect(h.stopped).toEqual([1]);

    // Not executable: a hand download without chmod +x, or on Windows an
    // antivirus. Neither is worth a restart.
    const h3 = setup();
    h3.engine.start(SETTINGS);
    h3.child().emit('error', Object.assign(new Error('spawn EACCES'), { code: 'EACCES' }));
    expect(h3.errors).toEqual(['the CPU miner is not allowed to run (/opt/llmjob-xmr-worker): make it executable with chmod +x.']);
    expect(h3.stopped).toEqual([1]);
    const h4 = setup({ platform: 'win32' });
    h4.engine.start(SETTINGS);
    h4.child().emit('error', Object.assign(new Error('spawn EACCES'), { code: 'EACCES' }));
    expect(h4.errors).toEqual(['the CPU miner is not allowed to run (/opt/llmjob-xmr-worker): an antivirus may be blocking it.']);

    const h2 = setup();
    h2.engine.start(SETTINGS);
    h2.child().emit('error', new Error('spawn EMFILE'));
    expect(h2.lines()).toContain('cpu: the CPU miner could not run: spawn EMFILE');
    expect(h2.engine.snapshot().state).toBe('restarting');
    h2.children[0].emit('exit', 1, null);   // the same child's exit is already handled
    expect(h2.children).toHaveLength(1);
    h2.advance(RESTART_MS);
    h2.children[0].emit('error', new Error('late'));   // nor is an old child's error
    expect(h2.lines()).not.toContain('cpu: the CPU miner could not run: late');
  });

  test('a kill that throws is fine', () => {
    const h = setup();
    h.engine.start(SETTINGS);
    h.child().kill = () => { throw new Error('ESRCH'); };
    h.child().say({ type: 'selftest', ok: false, msg: 'x' });
    expect(h.stopped).toEqual([1]);
  });

  test('a job for a worker that is between restarts waits for the next one', () => {
    const h = mining();
    h.child().emit('exit', 1, null);
    h.sock().recv({ method: 'job', params: job('j9') });
    h.advance(RESTART_MS);
    expect(h.child().stdin.lines[0].id).toBe('j9');
  });

  test('writes to a worker whose stdin is gone are dropped', () => {
    const h = mining();
    h.child().stdin.destroyed = true;
    h.sock().recv({ method: 'job', params: job('j2') });
    expect(h.child().stdin.lines).toHaveLength(1);
    h.child().stdin.destroyed = false;
    h.child().stdin.write = () => { throw new Error('EPIPE'); };
    h.sock().recv({ method: 'job', params: job('j3') });
    h.child().stdin.emit('error', new Error('EPIPE'));
    expect(h.engine.isRunning()).toBe(true);
  });
});

describe('shares', () => {
  test('a share of the current job is submitted under the session and counted when accepted', () => {
    const h = mining();
    h.child().say({ type: 'share', job: 'j1', nonce: '3a000000', result: RESULT });
    const submit = h.sock().written[1];
    expect(submit).toEqual({
      id: 10, jsonrpc: '2.0', method: 'submit', params: { id: 'sess', job_id: 'j1', nonce: '3a000000', result: RESULT },
    });
    h.sock().recv({ id: 10, jsonrpc: '2.0', error: null, result: { status: 'OK' } });
    expect(h.lines()).toContain('cpu: XMR share accepted');
    expect(h.engine.snapshot()).toMatchObject({ accepted: 1, rejected: 0, lastShareMs: 1000000 });
    // A reply to nothing we sent counts for nothing.
    h.sock().recv({ id: 10, error: null, result: { status: 'OK' } });
    expect(h.engine.snapshot().accepted).toBe(1);
  });

  test('a rejected share is counted and its reason logged', () => {
    const h = mining();
    h.child().say({ type: 'share', job: 'j1', nonce: '3a000000', result: RESULT });
    h.sock().recv({ id: 10, error: { code: -1, message: 'Low difficulty share' }, result: null });
    expect(h.lines()).toContain('cpu: XMR share rejected: [-1] Low difficulty share');
    expect(h.engine.snapshot().rejected).toBe(1);
    h.sock().recv({ id: 99, error: { code: -1, message: 'x' } });
    expect(h.engine.snapshot().rejected).toBe(1);
  });

  // HeroMiners holds back pay from a miner with too many stale shares.
  test('a share for an older job, or with no pool to send it to, is dropped', () => {
    const h = mining();
    h.sock().recv({ method: 'job', params: job('j2') });
    h.child().say({ type: 'share', job: 'j1', nonce: '3a000000', result: RESULT });
    h.sock().destroy();
    h.child().say({ type: 'share', job: 'j2', nonce: '3b000000', result: RESULT });
    expect(h.lines().filter((l) => l === 'cpu: dropping a share for a job the pool has moved on from')).toHaveLength(2);
    expect(h.sockets[0].written).toHaveLength(1);   // the login only
  });

  // cryptonote-nodejs-pool can ban the IP for a duplicate share.
  test('a share already sent for this job is not sent again', () => {
    const h = mining();
    const share = { type: 'share', job: 'j1', nonce: '3a000000', result: RESULT };
    h.child().say(share);
    h.child().say(share);
    expect(h.lines()).toContain('cpu: dropping a share already sent (job j1, nonce 3a000000)');
    // The same job with a new target still remembers it.
    h.sock().recv({ method: 'job', params: job('j1', { target: '0000000000000001' }) });
    h.child().say(share);
    // A worker restarted on the same job too.
    h.child().emit('exit', 1, null);
    h.advance(RESTART_MS);
    h.child().say(share);
    const submits = () => h.sock().written.filter((m) => m.method === 'submit').map((m) => m.params.job_id + ':' + m.params.nonce);
    expect(submits()).toEqual(['j1:3a000000']);
    // On the next job the same nonce is a new share.
    h.sock().recv({ method: 'job', params: job('j2') });
    h.child().say(Object.assign({}, share, { job: 'j2' }));
    expect(submits()).toEqual(['j1:3a000000', 'j2:3a000000']);
  });

  test('a share before any job is dropped', () => {
    const h = setup();
    h.engine.start(SETTINGS);
    h.sock().recv({ id: 1, result: { id: 'sess' } });
    h.child().say({ type: 'share', job: 'j1', nonce: '3a000000', result: RESULT });
    expect(h.lines()).toContain('cpu: dropping a share for a job the pool has moved on from');
  });
});

describe('the pool', () => {
  test('a login takes the session and the address that worked', () => {
    const h = setup();
    const authorized = [];
    h.engine.on('authorized', (a) => authorized.push(a));
    h.engine.start(SETTINGS);
    h.sock().recv({ id: 1, result: { id: 'sess' } });
    expect(authorized).toEqual([{ host: 'de.monero.herominers.com', address: '141.95.126.31', family: 'IPv4' }]);
    expect(h.lines()).toContain('cpu: logged in to the pool');
    expect(h.engine.poolConnected()).toBe(true);
  });

  test('a refused login closes the socket and waits a minute', () => {
    const h = setup();
    h.engine.start(SETTINGS);
    h.sock().recv({ id: 1, error: { code: -1, message: 'Invalid address used for login' }, result: null });
    expect(h.lines()).toContain('cpu: the pool refused the login: [-1] Invalid address used for login');
    expect(h.lines()).toContain('cpu: pool connection closed; reconnecting in 60 s');
    h.advance(RECONNECT_MS);
    expect(h.sockets).toHaveLength(1);
    h.advance(LOGIN_RETRY_MS);
    expect(h.sockets).toHaveLength(2);
    expect(h.engine.snapshot().poolError).toBe('the Monero pool refused the login: [-1] Invalid address used for login');
    // The next close is a normal one again.
    h.sock().emit('close');
    h.advance(RECONNECT_MS);
    expect(h.sockets).toHaveLength(3);
  });

  // DNS filters, antivirus web protection and firewalls often block Monero
  // pools. The Mine tab says so, and the log isn't flooded every 5 s.
  test('a pool it can\'t reach: said once in the log, and kept for the UI until a login', () => {
    const h = setup();
    h.engine.start(SETTINGS);
    for (let i = 0; i < 3; i++) {
      h.sock().emit('error', new Error('connect ECONNREFUSED 141.95.126.31:1111'));
      h.sock().emit('close');
      h.advance(RECONNECT_MS);
    }
    expect(h.sockets).toHaveLength(4);
    expect(h.lines().filter((l) => /pool socket error|pool connection closed/.test(l))).toEqual([
      'cpu: pool socket error: connect ECONNREFUSED 141.95.126.31:1111',
      'cpu: pool connection closed; reconnecting in 5 s (until it connects, the retries are not logged)',
    ]);
    expect(h.events.filter((e) => e.type === 'connect-failed')).toHaveLength(3);
    expect(h.engine.snapshot().poolError).toBe('can\'t reach the Monero pool: connect ECONNREFUSED 141.95.126.31:1111');
    // A new reason is news.
    h.sock().emit('error', new Error('getaddrinfo ENOTFOUND de.monero.herominers.com'));
    expect(h.lines()).toContain('cpu: pool socket error: getaddrinfo ENOTFOUND de.monero.herominers.com');
    h.sock().emit('close');
    h.advance(RECONNECT_MS);

    h.sock().recv({ id: 1, result: { id: 'sess', job: job('j1') } });
    expect(h.engine.snapshot().poolError).toBeNull();
    // After a session, a lost connection is logged again.
    h.sock().emit('error', new Error('read ECONNRESET'));
    h.sock().emit('close');
    expect(h.lines().slice(-2)).toEqual([
      'cpu: pool socket error: read ECONNRESET',
      'cpu: pool connection closed; reconnecting in 5 s',
    ]);
    // Within the grace for a routine reconnect, the UI doesn't hear of it.
    expect(h.engine.snapshot().poolError).toBeNull();
    h.advance(POOL_GRACE_MS);
    expect(h.engine.snapshot().poolError).toBe('can\'t reach the Monero pool: read ECONNRESET');
  });

  test('a pool that closes before the login, with no error', () => {
    const h = setup();
    h.engine.start(SETTINGS);
    h.sock().emit('close');
    expect(h.engine.snapshot().poolError).toBe('the Monero pool closed the connection before the login');
  });

  test('a dropped connection reconnects, and counts as up for the grace period', () => {
    const h = mining();
    h.sock().emit('close');
    expect(h.engine.poolConnected()).toBe(true);
    h.setNow(1000000 + POOL_GRACE_MS);
    expect(h.engine.poolConnected()).toBe(false);
    h.advance(RECONNECT_MS);
    expect(h.sockets).toHaveLength(2);
    // Data from the old socket is ignored, and so is its late close.
    h.sockets[0].recv({ method: 'job', params: job('old') });
    h.sockets[0].emit('close');
    expect(h.sockets).toHaveLength(2);
    expect(h.child().stdin.lines.map((l) => l.id)).toEqual(['j1']);
  });

  test('a connection that never logged in has no grace', () => {
    const h = setup();
    h.engine.start(SETTINGS);
    expect(h.engine.poolConnected()).toBe(false);
    h.sock().emit('close');
    expect(h.engine.poolConnected()).toBe(false);
  });

  test('a socket error is logged, and a DNS failure is said to be one', () => {
    const h = setup();
    h.engine.start(SETTINGS);
    h.sock().emit('error', new Error('getaddrinfo ENOTFOUND de.monero.herominers.com'));
    h.sock().emit('error', new Error('connect ECONNREFUSED'));
    expect(h.events.filter((e) => e.type === 'connect-failed')).toEqual([
      { type: 'connect-failed', reason: 'getaddrinfo ENOTFOUND de.monero.herominers.com', dns: true },
      { type: 'connect-failed', reason: 'connect ECONNREFUSED', dns: false },
    ]);
    expect(h.lines()).toContain('cpu: pool socket error: connect ECONNREFUSED');
  });

  test('no endpoint is no host and no port', () => {
    const h = setup();
    h.engine.start({ address: ADDRESS, threads: 2 });
    expect(h.engine.connect).toHaveBeenCalledWith('', NaN);
  });

  test('a connect that throws tries again later', () => {
    let n = 0;
    const h = setup({ connect: () => { n++; if (n === 1) throw new Error('EMFILE'); return fakeSocket(); } });
    h.engine.start(SETTINGS);
    expect(h.lines()).toContain('cpu: pool connection failed: EMFILE');
    h.advance(RECONNECT_MS);
    expect(n).toBe(2);
  });

  test('keepalives go out while logged in, and their replies are not shares', () => {
    const h = mining();
    h.advance(KEEPALIVE_MS);
    const ka = h.sock().written[1];
    expect(ka).toEqual({ id: 10, jsonrpc: '2.0', method: 'keepalived', params: { id: 'sess' } });
    h.sock().recv({ id: 10, error: null, result: { status: 'KEEPALIVED' } });
    expect(h.engine.snapshot().accepted).toBe(0);
    h.sock().emit('close');
    h.advance(KEEPALIVE_MS);
    expect(h.sockets[0].written).toHaveLength(2);
  });

  test('a pool that never answers keepalives does not grow the list forever', () => {
    const h = mining();
    for (let i = 0; i < 20; i++) h.engine._sendKeepalive();
    expect(h.engine.keepaliveIds.size).toBe(16);
  });

  test('five quiet minutes means a dead connection', () => {
    const h = mining({ timing: { reconnect: 120000 } });
    h.advance(SILENCE_MS - 30000);
    expect(h.sockets).toHaveLength(1);
    h.advance(30000);
    expect(h.lines()).toContain('cpu: no word from the pool in 5 minutes; reconnecting');
    expect(h.sockets[0].destroy).toHaveBeenCalledTimes(1);
    // Between sockets there is nothing to watch.
    h.advance(60000);
    expect(h.sockets).toHaveLength(1);
    h.advance(60000);
    expect(h.sockets).toHaveLength(2);
  });

  // Keepalive replies kept the five-minute watch happy, so a pool that answered
  // them and sent no jobs was never left.
  test('ten minutes with no new job means a pool that stopped, even if it answers keepalives', () => {
    const h = mining({ timing: { reconnect: 120000 } });
    const answer = () => {
      const w = h.sock().written;
      h.sock().recv({ id: w[w.length - 1].id, error: null, result: { status: 'KEEPALIVED' } });
    };
    for (let m = 1; m <= 5; m++) { h.advance(KEEPALIVE_MS); answer(); }
    // A job resets the clock.
    h.sock().recv({ method: 'job', params: job('j2') });
    for (let m = 1; m < 10; m++) { h.advance(KEEPALIVE_MS); answer(); }
    expect(h.sockets[0].destroy).not.toHaveBeenCalled();
    h.advance(KEEPALIVE_MS);
    expect(h.lines()).toContain('cpu: no new job from the pool in 10 minutes; reconnecting');
    expect(h.lines()).not.toContain('cpu: no word from the pool in 5 minutes; reconnecting');
    expect(h.sockets[0].destroy).toHaveBeenCalledTimes(1);
  });

  // cryptonote-nodejs-pool bans an IP by forgetting its session and leaving
  // the socket open. Nothing comes back on that socket but errors.
  test('a pool that drops the session but keeps the socket open gets a new login', () => {
    const h = mining();
    h.advance(KEEPALIVE_MS);
    h.sock().recv({ id: 10, jsonrpc: '2.0', error: { code: -1, message: 'Unauthenticated' }, result: null });
    expect(h.lines()).toContain('cpu: the pool dropped this session ([-1] Unauthenticated); logging in again');
    expect(h.lines()).toContain('cpu: pool connection closed; reconnecting in 5 s');
    expect(h.sockets[0].destroy).toHaveBeenCalledTimes(1);
    h.advance(RECONNECT_MS);
    expect(h.sockets).toHaveLength(2);

    // A banned IP: the rejected share counts, and the next login waits a minute.
    h.sock().recv({ id: 1, result: { id: 'sess2', job: job('j2') } });
    h.child().say({ type: 'share', job: 'j2', nonce: '3a000000', result: RESULT });
    const id = h.sock().written.filter((m) => m.method === 'submit')[0].id;
    h.sock().recv({ id, error: { code: -1, message: 'Your IP is banned' } });
    expect(h.engine.snapshot().rejected).toBe(1);
    expect(h.lines()).toContain('cpu: pool connection closed; reconnecting in 60 s');
    h.advance(LOGIN_RETRY_MS);
    expect(h.sockets).toHaveLength(3);

    // A pool error with no id, and the rest of its chunk is not read.
    h.sock().recv({ id: 1, result: { id: 'sess3', job: job('j3') } });
    h.sock().emit('data', JSON.stringify({ id: null, error: { code: -1, message: 'Invalid job id' } }) + '\n'
      + JSON.stringify({ method: 'job', params: job('j4') }) + '\n');
    expect(h.sockets[2].destroy).toHaveBeenCalledTimes(1);
    expect(h.child().stdin.lines.map((l) => l.id)).not.toContain('j4');
  });

  test('other errors leave the session alone', () => {
    const h = mining();
    h.advance(KEEPALIVE_MS);
    h.sock().recv({ id: 10, error: { code: -1, message: 'Server busy' } });
    h.child().say({ type: 'share', job: 'j1', nonce: '3a000000', result: RESULT });
    h.sock().recv({ id: 11, error: { code: -1, message: 'Low difficulty share' } });
    expect(h.sock().destroy).not.toHaveBeenCalled();
    expect(h.engine.snapshot().rejected).toBe(1);
  });

  // Monero's next fork changes the algorithm, for one. Every share of the
  // last job would be rejected from then on.
  test('a job it cannot mine pauses the worker until one it can', () => {
    const h = mining();
    const c = h.child();
    h.sock().recv({ method: 'job', params: job('j2', { algo: 'rx/2' }) });
    h.sock().recv({ method: 'job', params: job('j3', { algo: 'rx/2' }) });
    expect(c.stdin.lines.slice(1)).toEqual([{ cmd: 'pause' }]);
    c.say({ type: 'share', job: 'j1', nonce: '3a000000', result: RESULT });
    expect(h.sock().written.filter((m) => m.method === 'submit')).toEqual([]);
    // The new job goes first, so resume starts on it.
    h.sock().recv({ method: 'job', params: job('j4') });
    expect(c.stdin.lines.slice(2).map((l) => l.cmd + (l.id ? ' ' + l.id : ''))).toEqual(['job j4', 'resume']);
    h.sock().recv({ method: 'job', params: job('j5') });
    expect(c.stdin.lines.slice(4).map((l) => l.cmd)).toEqual(['job']);
  });

  test('a bad job in the login after a reconnect leaves no old job live', () => {
    const h = mining();
    h.sock().emit('close');
    h.advance(RECONNECT_MS);
    h.sock().recv({ id: 1, result: { id: 'sess2', job: job('j2', { seed_hash: undefined }) } });
    h.child().say({ type: 'share', job: 'j1', nonce: '3a000000', result: RESULT });
    expect(h.sock().written.filter((m) => m.method === 'submit')).toEqual([]);
    expect(h.child().stdin.lines.slice(-1)).toEqual([{ cmd: 'pause' }]);
  });

  test('a bad job between worker restarts: the new worker just waits', () => {
    const h = mining();
    h.child().emit('exit', 1, null);
    h.sock().recv({ method: 'job', params: job('j2', { algo: 'rx/2' }) });
    h.advance(RESTART_MS);
    expect(h.child().stdin.lines).toEqual([]);
    h.sock().recv({ method: 'job', params: job('j3') });
    expect(h.child().stdin.lines.map((l) => l.cmd)).toEqual(['job']);
  });

  test('bad jobs, pool errors and unknown lines are logged', () => {
    const h = mining();
    h.sock().recv({ method: 'job', params: job('j2', { algo: 'rx/arq' }) });
    h.sock().recv({ method: 'job', params: {} });
    h.sock().recv({ id: null, error: { code: -1, message: 'Server busy' } });
    h.sock().emit('data', 'hello\n\n');
    h.sock().recv({ method: 'mining.notify' });
    expect(h.lines().slice(-5, -1)).toEqual([
      'cpu: ignoring a job the CPU miner cannot mine (j2): algorithm rx/arq is not one this miner mines',
      'cpu: ignoring a job the CPU miner cannot mine: no job_id',
      'cpu: pool: [-1] Server busy',
      'cpu: pool: hello',
    ]);
  });
});

describe('stop', () => {
  test('asks the worker to quit, closes the pool, and kills a worker that lingers', () => {
    const h = mining();
    const c = h.child();
    const s = h.sock();
    expect(h.engine.stop()).toBe(true);
    expect(c.stdin.lines[c.stdin.lines.length - 1]).toEqual({ cmd: 'quit' });
    expect(c.stdin.end).toHaveBeenCalled();
    expect(s.destroy).toHaveBeenCalled();
    expect(h.stopped).toEqual([0]);
    expect(h.events[h.events.length - 1]).toMatchObject({ type: 'status', state: 'stopped', hashrate: 0 });
    expect(h.engine.isRunning()).toBe(false);
    expect(h.engine.poolConnected()).toBe(false);
    h.advance(KILL_MS);
    expect(c.kill).toHaveBeenCalledWith('SIGKILL');
    expect(h.engine.stop()).toBe(false);
  });

  test('a worker that exits in time is not killed, and its exit is not a failure', () => {
    const h = mining();
    const c = h.child();
    h.engine.stop();
    c.emit('exit', 0, null);
    h.advance(KILL_MS);
    expect(c.kill).not.toHaveBeenCalled();
    expect(h.errors).toEqual([]);
  });

  test('a worker already gone, a quit that throws, a kill that throws', () => {
    const h = mining();
    const c = h.child();
    c.stdin.write = () => { throw new Error('EPIPE'); };
    c.kill = () => { throw new Error('ESRCH'); };
    h.engine.stop();
    h.advance(KILL_MS);
    expect(h.stopped).toEqual([0]);

    const h2 = mining();
    h2.child().emit('exit', 1, null);   // between restarts: no worker, no socket after a close
    h2.sock().emit('close');
    h2.engine.stop();
    expect(h2.stopped).toEqual([0]);
    h2.advance(RESTART_MS + RECONNECT_MS);
    expect(h2.children).toHaveLength(1);
    expect(h2.sockets).toHaveLength(1);
  });

  test('the engine can start again after a stop, with fresh counters', () => {
    const h = mining();
    h.child().say({ type: 'share', job: 'j1', nonce: '3a000000', result: RESULT });
    h.sock().recv({ id: 10, result: { status: 'OK' } });
    h.engine.stop();
    h.engine.start(SETTINGS);
    expect(h.engine.snapshot()).toMatchObject({ accepted: 0, state: 'starting' });
    expect(h.children).toHaveLength(2);
  });
});
