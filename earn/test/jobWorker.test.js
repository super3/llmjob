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

// A fake scheduler that records (fn, ms) pairs and returns inspectable handles.
function makeScheduler() {
  const scheduled = [];
  const cancelled = [];
  return {
    scheduled,
    cancelled,
    schedule: (fn, ms) => { const t = { fn, ms }; scheduled.push(t); return t; },
    cancel: (t) => cancelled.push(t),
  };
}

const okFor = (jobs) => (url) => (url.endsWith('/poll') ? { status: 200, data: { jobs } } : { status: 200 });

describe('constructor defaults', () => {
  test('sensible defaults and a real scheduler/clock', () => {
    const w = new JobWorker({ identity: IDENT, serverUrl: 's', post: () => {}, runJob: () => {} });
    expect(w.activeJobs()).toBe(0);
    expect(typeof w.now()).toBe('number');
    expect(w.idleMs).toBe(5000);
    expect(w.maxIdleMs).toBe(60000);
    expect(w.heartbeatMs).toBe(30000);
    expect(w.flushMs).toBe(1000);
    const t = w.schedule(() => {}, 0); // default setTimeout (unref'd)
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
      expect(calls[0].body).toMatchObject({ nodeId: 'nnn111', timestamp: 1, maxJobs: 1 });
      const ok = nacl.sign.detached.verify(
        naclUtil.decodeUTF8('nnn111:1'), naclUtil.decodeBase64(calls[0].body.signature), naclUtil.decodeBase64(IDENT.publicKey));
      expect(ok).toBe(true);
    }
  });
});

describe('processJob — success streaming', () => {
  test('heartbeats immediately + on cadence, streams ordered chunks, final chunk carries metrics, completes', async () => {
    const sch = makeScheduler();
    let t = 1000;
    const { post, calls } = makePost(okFor([{ id: 'J1', prompt: 'hi', model: 'M' }]));
    const runJob = (chatBody, { onDelta }) => {
      expect(chatBody.messages[0].content).toBe('hi');
      onDelta('abcd'); onDelta('efgh', 3); onDelta('XY'); // chunkChars 4 → flush 'abcd', 'efgh'; 'XY' rides the final
      t += 2000; // 2s elapsed
      return Promise.resolve();
    };
    const events = [];
    const w = new JobWorker({
      identity: IDENT, serverUrl: 's', post, runJob, now: () => t,
      chunkChars: 4, schedule: sch.schedule, cancel: sch.cancel,
    });
    w.on('job', (e) => events.push('job:' + e.active)).on('done', (e) => events.push('done:' + e.id));
    await w.pollOnce();

    // heartbeat: one immediate POST + the next beat scheduled at heartbeatMs, cancelled in finally
    const hb = calls.filter((c) => c.url === 's/api/jobs/J1/heartbeat');
    expect(hb).toHaveLength(1);
    const hbTimer = sch.scheduled.find((s) => s.ms === 30000);
    expect(hbTimer).toBeTruthy();
    expect(sch.cancelled).toContain(hbTimer);

    const chunks = calls.filter((c) => c.url === 's/api/jobs/J1/chunks').map((c) => c.body);
    expect(chunks.map((c) => [c.chunkIndex, c.content, c.isFinal])).toEqual([
      [0, 'abcd', false], [1, 'efgh', false], [2, 'XY', true],
    ]);
    expect(chunks[0].metrics).toBeUndefined();
    expect(chunks[2].metrics).toEqual({ totalTokens: 5, tokensPerSecond: 2.5, elapsedSeconds: 2, model: 'M' });
    expect(calls.some((c) => c.url === 's/api/jobs/J1/complete')).toBe(true);
    expect(events).toEqual(['job:1', 'done:J1']);
    expect(w.activeJobs()).toBe(0);
  });

  test('empty result still sends one final metrics chunk, then completes', async () => {
    const { post, calls } = makePost(okFor([{ id: 'J3', prompt: '' }]));
    const w = new JobWorker({ identity: IDENT, serverUrl: 's', post, runJob: () => Promise.resolve(), now: () => 1 });
    await w.pollOnce();
    const chunks = calls.filter((c) => c.url === 's/api/jobs/J3/chunks').map((c) => c.body);
    expect(chunks).toHaveLength(1);
    expect(chunks[0]).toMatchObject({ chunkIndex: 0, content: '', isFinal: true });
    expect(chunks[0].metrics.totalTokens).toBe(0);
    expect(chunks[0].metrics.elapsedSeconds).toBe(0.001); // clock frozen → floor applies
    expect(calls.some((c) => c.url === 's/api/jobs/J3/complete')).toBe(true);
  });

  test('time floor flushes a small buffer once flushMs has elapsed', async () => {
    let t = 0;
    const { post, calls } = makePost(okFor([{ id: 'J6', prompt: 'p' }]));
    const runJob = (_b, { onDelta }) => {
      onDelta('ab');        // below chunkChars, below flushMs → buffered
      t += 150;
      onDelta('cd');        // 150ms ≥ flushMs(100) → flush 'abcd'
      onDelta('Z');         // rides the final
      return Promise.resolve();
    };
    const w = new JobWorker({ identity: IDENT, serverUrl: 's', post, runJob, now: () => t, chunkChars: 1000, flushMs: 100 });
    await w.pollOnce();
    const chunks = calls.filter((c) => c.url === 's/api/jobs/J6/chunks').map((c) => c.body);
    expect(chunks.map((c) => [c.content, c.isFinal])).toEqual([['abcd', false], ['Z', true]]);
  });

  test('a heartbeat schedule returning undefined skips the cancel; a rejected heartbeat POST is swallowed', async () => {
    const post = (url) => {
      if (url.endsWith('/heartbeat')) return Promise.reject(new Error('offline'));
      if (url.endsWith('/poll')) return Promise.resolve({ status: 200, data: { jobs: [{ id: 'J7', prompt: 'p' }] } });
      return Promise.resolve({ status: 200 });
    };
    const cancel = jest.fn();
    const done = [];
    const w = new JobWorker({
      identity: IDENT, serverUrl: 's', post, runJob: () => Promise.resolve(), now: () => 1,
      schedule: () => undefined, cancel,
    });
    w.on('done', (e) => done.push(e.id));
    await w.pollOnce();
    await flush();
    expect(cancel).not.toHaveBeenCalled();
    expect(done).toEqual(['J7']); // heartbeat failure never affects the job
  });

  test('an empty delta at the time floor does not emit an empty mid-chunk', async () => {
    let t = 0;
    const { post, calls } = makePost(okFor([{ id: 'JE', prompt: 'p' }]));
    const runJob = (_b, { onDelta }) => {
      t += 500;
      onDelta('');          // time floor exceeded but nothing buffered → no chunk
      onDelta('hi');
      return Promise.resolve();
    };
    const w = new JobWorker({ identity: IDENT, serverUrl: 's', post, runJob, now: () => t, chunkChars: 1000, flushMs: 100 });
    await w.pollOnce();
    const chunks = calls.filter((c) => c.url === 's/api/jobs/JE/chunks').map((c) => c.body);
    // no ['', false] entry from the empty delta; 'hi' flushes at the floor and
    // the always-sent final metrics chunk is empty by design
    expect(chunks.map((c) => [c.content, c.isFinal])).toEqual([['hi', false], ['', true]]);
  });
});

describe('processJob — failure paths', () => {
  test('a rejected (non-2xx) chunk fails the job instead of completing', async () => {
    const { post, calls } = makePost((url) => {
      if (url.endsWith('/poll')) return { status: 200, data: { jobs: [{ id: 'J8', prompt: 'p' }] } };
      if (url.endsWith('/chunks')) return { status: 400, data: { error: 'stale lock' } };
      return { status: 200 };
    });
    const runJob = (_b, { onDelta }) => { onDelta('abcd'); onDelta('efgh'); return Promise.resolve(); }; // two failing chunks
    const failed = [];
    const w = new JobWorker({ identity: IDENT, serverUrl: 's', post, runJob, now: () => 1, chunkChars: 4 });
    w.on('failed', (e) => failed.push(e));
    await w.pollOnce();
    expect(failed).toEqual([{ id: 'J8', error: 'chunk 0 rejected (HTTP 400)' }]); // first failure wins
    expect(calls.some((c) => c.url === 's/api/jobs/J8/fail')).toBe(true);
    expect(calls.some((c) => c.url === 's/api/jobs/J8/complete')).toBe(false);
  });

  test('a rejected /complete emits failed (no false done, no /fail) — dead and 4xx responses', async () => {
    for (const [completeRes, expectedMsg] of [[undefined, 'HTTP 0'], [{ status: 400 }, 'HTTP 400']]) {
      const { post, calls } = makePost((url) => {
        if (url.endsWith('/poll')) return { status: 200, data: { jobs: [{ id: 'J9', prompt: 'p' }] } };
        if (url.endsWith('/complete')) return completeRes;
        return { status: 200 };
      });
      const events = [];
      const w = new JobWorker({ identity: IDENT, serverUrl: 's', post, runJob: () => Promise.resolve(), now: () => 1 });
      w.on('done', () => events.push('done')).on('failed', (e) => events.push('failed:' + e.error));
      await w.pollOnce();
      expect(events).toEqual(['failed:complete rejected (' + expectedMsg + ')']);
      expect(calls.some((c) => c.url === 's/api/jobs/J9/fail')).toBe(false);
    }
  });

  test('a chunk answered with a dead response reports HTTP 0', async () => {
    const { post } = makePost((url) => {
      if (url.endsWith('/poll')) return { status: 200, data: { jobs: [{ id: 'J0', prompt: 'p' }] } };
      if (url.endsWith('/chunks')) return undefined; // dead response → HTTP 0
      return { status: 200 };
    });
    const failed = [];
    const w = new JobWorker({ identity: IDENT, serverUrl: 's', post, runJob: (_b, { onDelta }) => { onDelta('abcd'); return Promise.resolve(); }, now: () => 1, chunkChars: 4 });
    w.on('failed', (e) => failed.push(e.error));
    await w.pollOnce();
    expect(failed).toEqual(['chunk 0 rejected (HTTP 0)']);
  });

  test('runJob rejection flushes buffered content then reports failure', async () => {
    const { post, calls } = makePost(okFor([{ id: 'J4', prompt: 'p' }]));
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

  test('transport-rejected chunk and /fail POSTs are swallowed, job still fails cleanly', async () => {
    const seen = [];
    const post = (url) => {
      seen.push(url);
      if (url.endsWith('/chunks') || url.endsWith('/fail')) return Promise.reject(new Error('offline'));
      if (url.endsWith('/poll')) return Promise.resolve({ status: 200, data: { jobs: [{ id: 'J5', prompt: 'p' }] } });
      return Promise.resolve({ status: 200 });
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

describe('start / stop loop + backoff', () => {
  test('polls, schedules the next tick, and stop() cancels it', async () => {
    const sch = makeScheduler();
    const { post, calls } = makePost(() => ({ status: 200, data: { jobs: [] } }));
    const w = new JobWorker({ identity: IDENT, serverUrl: 's', post, runJob: () => Promise.resolve(), now: () => 1, schedule: sch.schedule, cancel: sch.cancel });

    w.start();
    w.start(); // already running → no-op
    await flush();
    expect(calls).toHaveLength(1);
    expect(sch.scheduled).toHaveLength(1);

    sch.scheduled[0].fn(); // drive one more tick
    await flush();
    expect(calls).toHaveLength(2);

    w.stop();
    expect(sch.cancelled).toContain(sch.scheduled[1]);
    expect(w.running).toBe(false);
  });

  test('empty polls back off exponentially to maxIdleMs and a job resets the cadence', async () => {
    const sch = makeScheduler();
    let jobs = [];
    const { post } = makePost(() => ({ status: 200, data: { jobs } }));
    const w = new JobWorker({
      identity: IDENT, serverUrl: 's', post, runJob: () => Promise.resolve(), now: () => 1,
      schedule: sch.schedule, cancel: sch.cancel, idleMs: 1000, maxIdleMs: 4000,
    });
    w.start();
    await flush();
    expect(sch.scheduled.map((s) => s.ms)).toEqual([2000]); // 1000*2 after 1st empty poll
    sch.scheduled[0].fn(); await flush();
    sch.scheduled[1].fn(); await flush();
    expect(sch.scheduled.map((s) => s.ms)).toEqual([2000, 4000, 4000]); // capped at maxIdleMs

    jobs = [{ id: 'JB', prompt: 'p' }];
    sch.scheduled[2].fn(); await flush();
    jobs = [];
    // a job schedules its own heartbeat timer (30000); the poll tick is the last entry
    const pollTicks = sch.scheduled.filter((s) => s.ms !== 30000);
    expect(pollTicks[pollTicks.length - 1].ms).toBe(1000); // work → reset to idleMs
  });

  test('a poll error backs off, is emitted when listened, and never kills the loop when not', async () => {
    // With a listener: emitted normally.
    const sch1 = makeScheduler();
    const errors = [];
    const w1 = new JobWorker({
      identity: IDENT, serverUrl: 's', now: () => 1, schedule: sch1.schedule, cancel: sch1.cancel, idleMs: 1000,
      post: () => Promise.reject(new Error('offline')), runJob: () => Promise.resolve(),
    });
    w1.on('error', (e) => errors.push(e.message));
    w1.start();
    await flush();
    expect(errors).toEqual(['offline']);
    expect(sch1.scheduled.map((s) => s.ms)).toEqual([2000]); // error also backs off
    w1.stop();

    // Without a listener: the EventEmitter 'error' throw is contained and the loop keeps going.
    const sch2 = makeScheduler();
    const w2 = new JobWorker({
      identity: IDENT, serverUrl: 's', now: () => 1, schedule: sch2.schedule, cancel: sch2.cancel,
      post: () => Promise.reject(new Error('offline')), runJob: () => Promise.resolve(),
    });
    w2.start();
    await flush();
    expect(sch2.scheduled).toHaveLength(1); // still rescheduled despite no listener
    w2.stop();
  });

  test('_tick after stop is a no-op; stop() on a never-started worker is safe; no-arg construction works', () => {
    expect(() => new JobWorker()).not.toThrow();
    const w = new JobWorker({ identity: IDENT, serverUrl: 's', post: () => {}, runJob: () => {}, schedule: () => {} });
    w.running = false;
    expect(() => w._tick()).not.toThrow();
    w.stop(); // _timer null → nothing to cancel
    expect(w.running).toBe(false);
  });

  test('a stop mid-poll does not schedule the next tick', async () => {
    let resolvePost;
    const post = () => new Promise((r) => { resolvePost = () => r({ status: 200, data: { jobs: [] } }); });
    const sch = makeScheduler();
    const w = new JobWorker({ identity: IDENT, serverUrl: 's', post, runJob: () => Promise.resolve(), now: () => 1, schedule: sch.schedule, cancel: sch.cancel });
    w.start();
    await flush();
    w.stop();
    resolvePost();
    await flush();
    expect(sch.scheduled).toHaveLength(0);
  });
});
