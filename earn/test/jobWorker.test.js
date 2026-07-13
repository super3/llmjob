'use strict';

const { JobWorker } = require('../src/main/jobWorker');
const nacl = require('tweetnacl');
const naclUtil = require('tweetnacl-util');

const flush = () => new Promise((r) => setImmediate(r));
const IDENT = (() => {
  const kp = nacl.sign.keyPair();
  return { nodeId: 'nnn111', publicKey: naclUtil.encodeBase64(kp.publicKey), secretKey: naclUtil.encodeBase64(kp.secretKey) };
})();

// A `post` spy that records calls and returns a queued/looked-up response.
function makePost(responder) {
  const calls = [];
  const post = (url, body) => { calls.push({ url, body }); return Promise.resolve(responder(url, body)); };
  return { post, calls };
}

describe('constructor defaults', () => {
  test('sensible defaults and a real scheduler/clock', () => {
    const w = new JobWorker({ identity: IDENT, serverUrl: 's', post: () => {}, runJob: () => {} });
    expect(w.activeJobs()).toBe(0);
    expect(typeof w.now()).toBe('number');
    const t = w.schedule(() => {}, 0); // default setTimeout
    w.cancel(t);                       // default clearTimeout
    expect(w.running).toBe(false);
  });
});

describe('pollOnce', () => {
  test('returns 0 and processes nothing across all empty shapes', async () => {
    for (const resp of [undefined, {}, { data: {} }, { data: { jobs: [] } }]) {
      const { post, calls } = makePost(() => resp);
      const w = new JobWorker({ identity: IDENT, serverUrl: 's', post, runJob: () => Promise.resolve(), now: () => 1 });
      expect(await w.pollOnce()).toBe(0);
      expect(calls).toHaveLength(1);
      expect(calls[0].url).toBe('s/api/jobs/poll');
      // the poll body is signed and carries maxJobs
      expect(calls[0].body).toMatchObject({ nodeId: 'nnn111', timestamp: 1, maxJobs: 1 });
      const ok = nacl.sign.detached.verify(
        naclUtil.decodeUTF8('nnn111:1'), naclUtil.decodeBase64(calls[0].body.signature), naclUtil.decodeBase64(IDENT.publicKey));
      expect(ok).toBe(true);
    }
  });
});

describe('processJob — success streaming', () => {
  test('streams ordered chunks (mid + final remainder) then completes', async () => {
    const { post, calls } = makePost((url) => (url.endsWith('/poll') ? { data: { jobs: [{ id: 'J1', prompt: 'hi' }] } } : { status: 200 }));
    const runJob = (chatBody, { onDelta }) => {
      expect(chatBody.messages[0].content).toBe('hi');
      onDelta('abcd'); onDelta('efgh'); onDelta('XY'); // 10 chars, chunkChars 4 → flush at 'abcd'(0), 'efgh'(1), remainder 'XY' final
      return Promise.resolve();
    };
    const events = [];
    const w = new JobWorker({ identity: IDENT, serverUrl: 's', post, runJob, now: () => 2, chunkChars: 4 });
    w.on('job', (e) => events.push('job:' + e.active)).on('done', (e) => events.push('done:' + e.id));
    await w.pollOnce();
    const chunkCalls = calls.filter((c) => c.url === 's/api/jobs/J1/chunks').map((c) => c.body);
    expect(chunkCalls.map((c) => [c.chunkIndex, c.content, c.isFinal]))
      .toEqual([[0, 'abcd', false], [1, 'efgh', false], [2, 'XY', true]]);
    expect(calls.some((c) => c.url === 's/api/jobs/J1/complete')).toBe(true);
    expect(events).toEqual(['job:1', 'done:J1']);
    expect(w.activeJobs()).toBe(0);
  });

  test('exact-multiple stream sends no empty final chunk, still completes', async () => {
    const { post, calls } = makePost((url) => (url.endsWith('/poll') ? { data: { jobs: [{ id: 'J2', prompt: 'x' }] } } : {}));
    const runJob = (_b, { onDelta }) => { onDelta('abcd'); return Promise.resolve(); }; // exactly chunkChars
    const w = new JobWorker({ identity: IDENT, serverUrl: 's', post, runJob, now: () => 1, chunkChars: 4 });
    await w.pollOnce();
    const chunks = calls.filter((c) => c.url === 's/api/jobs/J2/chunks');
    expect(chunks).toHaveLength(1);
    expect(chunks[0].body).toMatchObject({ chunkIndex: 0, content: 'abcd', isFinal: false });
    expect(calls.some((c) => c.url === 's/api/jobs/J2/complete')).toBe(true);
  });

  test('empty result → no chunks, just complete', async () => {
    const { post, calls } = makePost((url) => (url.endsWith('/poll') ? { data: { jobs: [{ id: 'J3', prompt: '' }] } } : {}));
    const w = new JobWorker({ identity: IDENT, serverUrl: 's', post, runJob: () => Promise.resolve(), now: () => 1 });
    await w.pollOnce();
    expect(calls.some((c) => c.url === 's/api/jobs/J3/chunks')).toBe(false);
    expect(calls.some((c) => c.url === 's/api/jobs/J3/complete')).toBe(true);
  });
});

describe('processJob — failure', () => {
  test('flushes buffered content then reports failure', async () => {
    const { post, calls } = makePost((url) => (url.endsWith('/poll') ? { data: { jobs: [{ id: 'J4', prompt: 'p' }] } } : {}));
    const runJob = (_b, { onDelta }) => { onDelta('partial-'); return Promise.reject(new Error('boom')); };
    const failed = [];
    const w = new JobWorker({ identity: IDENT, serverUrl: 's', post, runJob, now: () => 1, chunkChars: 4 });
    w.on('failed', (e) => failed.push(e));
    await w.pollOnce();
    const fail = calls.find((c) => c.url === 's/api/jobs/J4/fail');
    expect(fail.body).toMatchObject({ error: 'boom' });
    expect(failed).toEqual([{ id: 'J4', error: 'boom' }]);
    expect(calls.some((c) => c.url === 's/api/jobs/J4/complete')).toBe(false);
    expect(w.activeJobs()).toBe(0);
  });

  test('a rejected chunk POST is swallowed and the job still fails cleanly', async () => {
    const seen = [];
    const post = (url) => {
      seen.push(url);
      if (url.endsWith('/chunks')) return Promise.reject(new Error('chunk fail'));
      if (url.endsWith('/poll')) return Promise.resolve({ data: { jobs: [{ id: 'J5', prompt: 'p' }] } });
      return Promise.resolve({});
    };
    const runJob = (_b, { onDelta }) => { onDelta('abcd'); return Promise.reject(new Error('boom')); };
    const failed = [];
    const w = new JobWorker({ identity: IDENT, serverUrl: 's', post, runJob, now: () => 1, chunkChars: 4 });
    w.on('failed', (e) => failed.push(e));
    await w.pollOnce();
    expect(failed).toEqual([{ id: 'J5', error: 'boom' }]);
    expect(seen.some((u) => u.endsWith('/fail'))).toBe(true);
  });
});

describe('start / stop loop', () => {
  test('polls, schedules the next tick, and stop() cancels it', async () => {
    let pending = null;
    const schedule = (fn) => { pending = fn; return 'TIMER'; };
    const cancel = jest.fn();
    const { post, calls } = makePost(() => ({ data: { jobs: [] } }));
    const w = new JobWorker({ identity: IDENT, serverUrl: 's', post, runJob: () => Promise.resolve(), now: () => 1, schedule, cancel });

    w.start();
    w.start(); // already running → no-op
    await flush();
    expect(calls).toHaveLength(1);          // one poll happened
    expect(typeof pending).toBe('function'); // next tick scheduled

    // drive one more tick manually
    pending();
    await flush();
    expect(calls).toHaveLength(2);

    w.stop();
    expect(cancel).toHaveBeenCalledWith('TIMER');
    expect(w.running).toBe(false);
  });

  test('a poll error is emitted and the loop keeps scheduling', async () => {
    let pending = null;
    const schedule = (fn) => { pending = fn; return 1; };
    const w = new JobWorker({
      identity: IDENT, serverUrl: 's', now: () => 1, schedule, cancel: () => {},
      post: () => Promise.reject(new Error('offline')), runJob: () => Promise.resolve(),
    });
    const errors = [];
    w.on('error', (e) => errors.push(e.message));
    w.start();
    await flush();
    expect(errors).toEqual(['offline']);
    expect(typeof pending).toBe('function'); // still scheduled despite the error
    w.stop();
  });

  test('_tick after stop is a no-op', () => {
    const w = new JobWorker({ identity: IDENT, serverUrl: 's', post: () => {}, runJob: () => {}, schedule: () => {} });
    w.running = false;
    expect(() => w._tick()).not.toThrow();
  });

  test('stop() on a never-started worker is safe; no-arg construction works', () => {
    expect(() => new JobWorker()).not.toThrow();         // default opts = {}
    const w = new JobWorker({ identity: IDENT, serverUrl: 's', post: () => {}, runJob: () => {} });
    w.stop();                                            // _timer is null → nothing to cancel
    expect(w.running).toBe(false);
  });

  test('a stop mid-poll does not schedule the next tick', async () => {
    let resolvePost;
    const post = () => new Promise((r) => { resolvePost = () => r({ data: { jobs: [] } }); });
    const schedule = jest.fn();
    const w = new JobWorker({ identity: IDENT, serverUrl: 's', post, runJob: () => Promise.resolve(), now: () => 1, schedule, cancel: () => {} });
    w.start();
    await flush();            // poll in flight
    w.stop();                 // running → false before the poll resolves
    resolvePost();
    await flush();
    expect(schedule).not.toHaveBeenCalled();
  });
});
