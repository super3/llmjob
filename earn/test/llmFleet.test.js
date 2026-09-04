'use strict';

const { EventEmitter } = require('events');
const { LlmFleet } = require('../src/main/llmFleet');

// A stand-in LlmManager: records its start opts, exposes baseUrl, and lets the
// test drive ready/stats/stopped/error like the real process would.
class FakeMgr extends EventEmitter {
  constructor() { super(); this.baseUrl = null; this.startOpts = null; this.stopped = false; }
  start(opts) { this.startOpts = opts; this.baseUrl = 'http://' + opts.host + ':' + opts.port; }
  stop() { this.stopped = true; }
}
class FakeWorker {
  constructor(url, index) { this.url = url; this.index = index; this.started = false; this.stopped = false; this._active = 0; }
  start() { this.started = true; }
  stop() { this.stopped = true; }
  activeJobs() { return this._active; }
}

// Build a fleet with capturing factories. findFreePort is identity by default
// (ports = basePort, basePort+1, …).
function makeFleet(over = {}) {
  const mgrs = [];
  const workers = [];
  const opts = Object.assign({
    host: '127.0.0.1',
    basePort: 8080,
    makeManager: () => { const m = new FakeMgr(); mgrs.push(m); return m; },
    findFreePort: async (h, p) => p,
    // Instances start one at a time, each waiting for its predecessor to settle
    // (see LlmFleet.start). Tests that don't drive 'ready' rely on this short
    // settle timeout to let the rest of the fleet come up.
    startSettleMs: 1,
    makeWorker: (url, index) => { const w = new FakeWorker(url, index); workers.push(w); return w; },
  }, over);
  const fleet = new LlmFleet(opts);
  fleet.on('error', () => {}); // keep EventEmitter from throwing in tests
  return { fleet, mgrs, workers };
}

// Let the background drain spawn the remaining instances.
const drain = () => new Promise((r) => setTimeout(r, 25));

describe('LlmFleet', () => {
  test('start() launches one manager per plan entry on distinct ports', async () => {
    const { fleet, mgrs } = makeFleet();
    const n = await fleet.start(
      [{ index: 0, nGpuLayers: 42 }, { index: 1, nGpuLayers: 40 }],
      { binaryPath: 'b', modelPath: 'm', platform: 'linux' }
    );
    expect(n).toBe(2);
    await drain();
    expect(mgrs).toHaveLength(2);
    expect(mgrs[0].startOpts).toMatchObject({ host: '127.0.0.1', port: 8080, nGpuLayers: 42, mainGpu: 0, binaryPath: 'b', modelPath: 'm', platform: 'linux' });
    expect(mgrs[1].startOpts).toMatchObject({ port: 8081, nGpuLayers: 40, mainGpu: 1 });
    expect(fleet.instances[0].baseUrl).toBe('http://127.0.0.1:8080');
  });

  // The whole point of serialised startup: N servers each streaming a multi-GB
  // model at once thrash the page cache and can livelock (a 13-GPU rig with
  // 8 GB RAM never got a byte into VRAM). Each instance waits for the one
  // before it, so the first load warms the cache for the rest.
  test('start() spawns one instance at a time, waiting for each to become ready', async () => {
    const { fleet, mgrs } = makeFleet({ startSettleMs: 60000 }); // no timeout escape
    const n = await fleet.start(
      [{ index: 0, nGpuLayers: 42 }, { index: 1, nGpuLayers: 42 }, { index: 2, nGpuLayers: 42 }], {}
    );

    // Returns the PLANNED count immediately, but only card 0 is loading.
    expect(n).toBe(3);
    await drain();
    expect(mgrs).toHaveLength(1);

    // Card 0 ready → card 1 starts, and still no card 2.
    mgrs[0].emit('ready', { baseUrl: 'http://127.0.0.1:8080' });
    await drain();
    expect(mgrs).toHaveLength(2);

    // A dead instance also releases the queue — one bad card can't strand the rest.
    mgrs[1].emit('stopped', 1);
    await drain();
    expect(mgrs).toHaveLength(3);
    expect(mgrs[2].startOpts).toMatchObject({ port: 8082, mainGpu: 2 });
  });

  test('stop() mid-startup cancels the instances that have not spawned yet', async () => {
    const { fleet, mgrs } = makeFleet({ startSettleMs: 60000 });
    await fleet.start([{ index: 0 }, { index: 1 }, { index: 2 }], {});
    await drain();
    expect(mgrs).toHaveLength(1);

    fleet.stop();
    await drain();
    expect(mgrs).toHaveLength(1); // nothing further spawned
    expect(mgrs[0].stopped).toBe(true);
  });

  test('start() with a non-array or empty plan launches nothing', async () => {
    const { fleet, mgrs } = makeFleet();
    expect(await fleet.start(undefined, {})).toBe(0);
    expect(await fleet.start([], {})).toBe(0);
    expect(mgrs).toHaveLength(0);
    expect(fleet.isReady()).toBe(false);
    expect(fleet.webUrl()).toBeNull();
  });

  test('an unmeasured (null-index) instance omits --main-gpu', async () => {
    const { fleet, mgrs } = makeFleet();
    await fleet.start([{ index: null, nGpuLayers: 42 }], {});
    expect(mgrs[0].startOpts.mainGpu).toBeUndefined();
  });

  test('emits ready per instance and first-ready exactly once', async () => {
    const { fleet, mgrs } = makeFleet();
    await fleet.start([{ index: 0, nGpuLayers: 42 }, { index: 1, nGpuLayers: 42 }], {});
    await drain();
    const ready = []; const firsts = [];
    fleet.on('ready', (e) => ready.push(e));
    fleet.on('first-ready', (e) => firsts.push(e));
    mgrs[0].emit('ready', { baseUrl: 'http://127.0.0.1:8080' });
    mgrs[1].emit('ready', { baseUrl: 'http://127.0.0.1:8081' });
    expect(ready).toEqual([{ baseUrl: 'http://127.0.0.1:8080', index: 0 }, { baseUrl: 'http://127.0.0.1:8081', index: 1 }]);
    expect(firsts).toEqual([{ baseUrl: 'http://127.0.0.1:8080' }]);
    expect(fleet.isReady()).toBe(true);
    expect(fleet.readyCount()).toBe(2);
    expect(fleet.webUrl()).toBe('http://127.0.0.1:8080');
    expect(fleet.servingIndices()).toEqual([0, 1]);
  });

  test('serving starts one worker per ready instance; activeJobs sums them', async () => {
    const { fleet, mgrs, workers } = makeFleet();
    await fleet.start([{ index: 0, nGpuLayers: 42 }, { index: 1, nGpuLayers: 42 }], {});
    await drain();
    fleet.syncWorkers(true);             // enabled, but nothing ready yet
    expect(workers).toHaveLength(0);
    mgrs[0].emit('ready', { baseUrl: 'http://127.0.0.1:8080' }); // → worker for gpu0
    mgrs[1].emit('ready', { baseUrl: 'http://127.0.0.1:8081' }); // → worker for gpu1
    expect(workers).toHaveLength(2);
    expect(workers[0]).toMatchObject({ url: 'http://127.0.0.1:8080', index: 0, started: true });
    workers[0]._active = 1; workers[1]._active = 2;
    expect(fleet.activeJobs()).toBe(3);
  });

  test('enabling serving after instances are ready spins up their workers', async () => {
    const { fleet, mgrs, workers } = makeFleet();
    await fleet.start([{ index: 0, nGpuLayers: 42 }], {});
    mgrs[0].emit('ready', { baseUrl: 'http://127.0.0.1:8080' }); // ready, not serving
    expect(workers).toHaveLength(0);
    fleet.syncWorkers(true);
    expect(workers).toHaveLength(1);
    // idempotent — a second ready or sync doesn't double up
    fleet.syncWorkers(true);
    expect(workers).toHaveLength(1);
  });

  test('disabling serving stops and clears the workers', async () => {
    const { fleet, mgrs, workers } = makeFleet();
    await fleet.start([{ index: 0, nGpuLayers: 42 }], {});
    fleet.syncWorkers(true);
    mgrs[0].emit('ready', { baseUrl: 'http://127.0.0.1:8080' });
    expect(workers[0].started).toBe(true);
    fleet.syncWorkers(false);
    expect(workers[0].stopped).toBe(true);
    expect(fleet.activeJobs()).toBe(0);
  });

  test('makeWorker returning null serves nothing (and is the default)', async () => {
    const { fleet, mgrs } = makeFleet({ makeWorker: () => null });
    await fleet.start([{ index: 0, nGpuLayers: 42 }], {});
    fleet.syncWorkers(true);
    mgrs[0].emit('ready', { baseUrl: 'http://127.0.0.1:8080' });
    expect(fleet.activeJobs()).toBe(0);

    // no makeWorker at all → the built-in default (also null)
    const bare = new LlmFleet({ host: 'h', basePort: 9000, makeManager: () => new FakeMgr(), findFreePort: async (h, p) => p });
    bare.on('error', () => {});
    await bare.start([{ index: 0, nGpuLayers: 1 }], {});
    bare.syncWorkers(true);
    bare.instances[0].mgr.emit('ready', { baseUrl: 'http://h:9000' });
    expect(bare.activeJobs()).toBe(0);
  });

  test('tracks tok/s from stats, coercing junk to 0', async () => {
    const { fleet, mgrs } = makeFleet();
    await fleet.start([{ index: 0, nGpuLayers: 42 }], {});
    const stats = [];
    fleet.on('stats', (e) => stats.push(e));
    mgrs[0].emit('stats', { tokensPerSec: 42.5 });
    expect(fleet.tokensPerSec()).toBe(42.5);
    expect(stats).toEqual([{ tokensPerSec: 42.5, promptTokensPerSec: 0 }]);
    mgrs[0].emit('stats', { tokensPerSec: 'oops' });
    expect(fleet.tokensPerSec()).toBe(0);
    // Prefill is held separately and is sticky: a generation line must not wipe
    // the last prefill figure, and vice versa.
    mgrs[0].emit('stats', { tokensPerSec: null, promptTokensPerSec: 1840 });
    expect(fleet.promptTokensPerSec()).toBe(1840);
    expect(fleet.tokensPerSec()).toBe(0);
    mgrs[0].emit('stats', { tokensPerSec: 55, promptTokensPerSec: null });
    expect(fleet.tokensPerSec()).toBe(55);
    expect(fleet.promptTokensPerSec()).toBe(1840);
    mgrs[0].emit('stats', { promptTokensPerSec: 'junk' });
    expect(fleet.promptTokensPerSec()).toBe(0);
    mgrs[0].emit('stats', undefined);
    expect(fleet.tokensPerSec()).toBe(55);
  });

  test('re-emits log and error from instances', async () => {
    const { fleet, mgrs } = makeFleet();
    await fleet.start([{ index: 0, nGpuLayers: 42 }], {});
    const logs = []; const errs = [];
    fleet.on('log', (l) => logs.push(l));
    fleet.removeAllListeners('error');
    fleet.on('error', (e) => errs.push(e));
    mgrs[0].emit('log', { level: 'info', line: 'hi' });
    const err = new Error('boom');
    mgrs[0].emit('error', err);
    expect(logs).toEqual([{ level: 'info', line: 'hi' }]);
    expect(errs).toEqual([err]);
  });

  test('emits fleet "stopped" only once all instances have stopped', async () => {
    const { fleet, mgrs, workers } = makeFleet();
    await fleet.start([{ index: 0, nGpuLayers: 42 }, { index: 1, nGpuLayers: 42 }], {});
    await drain();
    fleet.syncWorkers(true);
    mgrs[0].emit('ready', { baseUrl: 'http://127.0.0.1:8080' });
    mgrs[1].emit('ready', { baseUrl: 'http://127.0.0.1:8081' });
    const stops = [];
    fleet.on('stopped', (code) => stops.push(code));
    mgrs[0].emit('stopped', 0);     // one card down, the other still serving
    expect(stops).toHaveLength(0);
    expect(workers[0].stopped).toBe(true); // its worker was torn down
    expect(fleet.readyCount()).toBe(1);
    mgrs[1].emit('stopped', 7);     // last card down → one fleet 'stopped', its code
    expect(stops).toEqual([7]);
    // never fires twice
    mgrs[1].emit('stopped');
    expect(stops).toHaveLength(1);
  });

  // Startup is serialised, so `instances` holds only the cards spawned so far.
  // Judging "is the fleet down?" on those alone declared death while the rest of
  // the plan was still queued — the GUI ended the session (and the CLI exited)
  // moments before the next card came up and started serving.
  test('does not emit "stopped" while a planned card is still queued', async () => {
    const { fleet, mgrs } = makeFleet();
    await fleet.start([{ index: 0, nGpuLayers: 42 }, { index: 1, nGpuLayers: 42 }], {});
    const stops = [];
    fleet.on('stopped', (code) => stops.push(code));

    // Only GPU0 has spawned; GPU1 is still waiting its turn.
    expect(mgrs).toHaveLength(1);
    mgrs[0].emit('stopped', 1);          // GPU0 dies before it ever loads
    expect(stops).toHaveLength(0);       // …the fleet has NOT had its chance yet

    await drain();
    expect(mgrs).toHaveLength(2);        // GPU1 spawned as planned
    mgrs[1].emit('ready', { baseUrl: 'http://127.0.0.1:8081' });
    expect(fleet.isReady()).toBe(true);  // and is serving
    expect(fleet.servingIndices()).toEqual([1]);
    expect(stops).toHaveLength(0);

    mgrs[1].emit('stopped', 3);          // now the last card is gone → really down
    expect(stops).toEqual([3]);
  });

  // The queue is shifted BEFORE the async port probe, so there is a window where
  // the entry is off `_pending` but its instance does not exist yet. Judging the
  // fleet on `_pending` alone reopened the same premature-'stopped' bug a few
  // milliseconds later.
  test('does not emit "stopped" while a successor is still probing for a port', async () => {
    let release;
    const { fleet, mgrs } = makeFleet({
      findFreePort: (h, p) => (mgrs.length === 1 ? new Promise((r) => { release = () => r(p); }) : Promise.resolve(p)),
    });
    await fleet.start([{ index: 0, nGpuLayers: 42 }, { index: 1, nGpuLayers: 42 }], {});
    const stops = [];
    fleet.on('stopped', (code) => stops.push(code));

    mgrs[0].emit('ready', { baseUrl: 'http://127.0.0.1:8080' }); // frees the drain
    await new Promise((r) => setImmediate(r));
    expect(fleet._pending).toEqual([]);        // shifted…
    expect(fleet.instances).toHaveLength(1);   // …but GPU1 is not an instance yet

    mgrs[0].emit('stopped', 1);                // GPU0 dies inside that window
    expect(stops).toHaveLength(0);

    release();
    await drain();
    expect(mgrs).toHaveLength(2);              // GPU1 still spawns
  });

  // The first spawn is awaited by start(), so its failure rejects — and would
  // strand the rest of the plan in _pending, gagging 'stopped' forever.
  test('a first spawn that throws abandons the queue before rethrowing', async () => {
    const { fleet } = makeFleet({ findFreePort: () => Promise.reject(new Error('probe failed')) });
    await expect(fleet.start([{ index: 0 }, { index: 1 }, { index: 2 }], {})).rejects.toThrow('probe failed');
    expect(fleet._pending).toEqual([]);
    expect(fleet._spawning).toBe(false);
  });

  test('a single-GPU fleet still reports down immediately (nothing was queued)', async () => {
    const { fleet, mgrs } = makeFleet();
    await fleet.start([{ index: 0, nGpuLayers: 42 }], {});
    const stops = [];
    fleet.on('stopped', (code) => stops.push(code));
    mgrs[0].emit('stopped', 1);
    expect(stops).toEqual([1]);
  });

  // The queue guard must not become a way to never report down: a drain that
  // throws would otherwise strand entries in _pending and gag 'stopped' forever.
  test('a failed drain abandons the queue so the fleet can still report down', async () => {
    let calls = 0;
    const { fleet, mgrs } = makeFleet({
      findFreePort: async (h, p) => { if (++calls > 1) throw new Error('no free port'); return p; },
    });
    await fleet.start([{ index: 0, nGpuLayers: 42 }, { index: 1, nGpuLayers: 42 }], {});
    const stops = [];
    fleet.on('stopped', (code) => stops.push(code));
    mgrs[0].emit('stopped', 1);   // the only spawned card dies
    await drain();                // GPU1's spawn throws; the queue is abandoned
    expect(fleet._pending).toEqual([]);
    expect(stops).toEqual([1]);   // and the fleet reports down rather than hanging
  });

  test('stop() tears down every worker and manager and goes quiet', async () => {
    const { fleet, mgrs, workers } = makeFleet();
    await fleet.start([{ index: 0, nGpuLayers: 42 }, { index: 1, nGpuLayers: 42 }], {});
    await drain();
    fleet.syncWorkers(true);
    mgrs[0].emit('ready', { baseUrl: 'http://127.0.0.1:8080' });
    mgrs[1].emit('ready', { baseUrl: 'http://127.0.0.1:8081' });
    const stops = [];
    fleet.on('stopped', () => stops.push(1));
    fleet.stop();
    expect(mgrs.every((m) => m.stopped)).toBe(true);
    expect(workers.every((w) => w.stopped)).toBe(true);
    expect(fleet.instances).toEqual([]);
    // a late 'stopped' from a manager after fleet.stop() stays silent
    mgrs[0].emit('stopped');
    expect(stops).toHaveLength(0);
  });

  test('adopt() adds a ready instance without spawning, and serves it', async () => {
    const { fleet, mgrs, workers } = makeFleet();
    fleet.syncWorkers(true);
    fleet.adopt('http://127.0.0.1:8080');
    expect(mgrs).toHaveLength(0);            // nothing spawned
    expect(fleet.isReady()).toBe(true);
    expect(fleet.webUrl()).toBe('http://127.0.0.1:8080');
    expect(fleet.servingIndices()).toEqual([]); // unknown GPU → not tagged
    expect(workers).toHaveLength(1);         // serving → a worker on the adopted URL
    expect(workers[0].url).toBe('http://127.0.0.1:8080');
    fleet.stop();                            // stops cleanly despite no manager
    expect(workers[0].stopped).toBe(true);
    expect(fleet.instances).toEqual([]);
  });

  test('adopt() without serving starts no worker', () => {
    const { fleet, workers } = makeFleet();
    fleet.adopt('http://h:1');
    expect(workers).toHaveLength(0);
    expect(fleet.isReady()).toBe(true);
  });

  test('hasSpawned() is true only for a live spawned instance, not an adopted one', async () => {
    const { fleet, mgrs } = makeFleet();
    expect(fleet.hasSpawned()).toBe(false);
    fleet.adopt('http://h:1');
    expect(fleet.hasSpawned()).toBe(false);      // adopted has no manager
    await fleet.start([{ index: 0, nGpuLayers: 42 }], {});
    expect(fleet.hasSpawned()).toBe(true);       // spawned & running
    mgrs[0].emit('stopped');
    expect(fleet.hasSpawned()).toBe(false);      // stopped
  });

  test('walks past busy ports using findFreePort', async () => {
    // findFreePort bumps every probe by +5, so instances land on non-adjacent ports.
    const { fleet, mgrs } = makeFleet({ findFreePort: async (h, p) => p + 5 });
    await fleet.start([{ index: 0, nGpuLayers: 42 }, { index: 1, nGpuLayers: 42 }], {});
    await drain();
    expect(mgrs[0].startOpts.port).toBe(8085);
    expect(mgrs[1].startOpts.port).toBe(8091); // 8086 probed → +5
  });
  // ── lifecycle across start/stop cycles ─────────────────────────────────────
  // A fleet object outlives a single run: main.js builds one and reuses it. The
  // run-scoped flags therefore have to be reset by start(), and two of them were
  // not — each producing a distinct user-visible failure.

  test('a fleet restarted after stop() brings up EVERY instance again', async () => {
    // `_stopping` used to survive the stop, so _drain() returned immediately on
    // the next run and only the first GPU of a multi-GPU rig ever came up.
    const { fleet, mgrs } = makeFleet();
    await fleet.start([{ index: 0, nGpuLayers: 42 }, { index: 1, nGpuLayers: 42 }], {});
    await drain();
    expect(mgrs).toHaveLength(2);

    fleet.stop();
    await fleet.start([{ index: 0, nGpuLayers: 42 }, { index: 1, nGpuLayers: 42 }], {});
    await drain();
    expect(mgrs).toHaveLength(4); // two more, not one
  });

  test("a fleet that died on its own can emit 'stopped' again on the next run", async () => {
    // `_downEmitted` used to survive, so the second run could never report going
    // down — the UI kept showing "LLM ready" with no llama-server running.
    const { fleet, mgrs } = makeFleet();
    const downs = [];
    fleet.on('stopped', () => downs.push(1));

    await fleet.start([{ index: 0, nGpuLayers: 42 }], {});
    mgrs[0].emit('stopped', 1);
    expect(downs).toHaveLength(1);

    await fleet.start([{ index: 0, nGpuLayers: 42 }], {});
    mgrs[1].emit('stopped', 1);
    expect(downs).toHaveLength(2);
  });

  test('a stop() landing during the port probe never spawns an untracked server', async () => {
    // findFreePort is where _startNext yields. Stopping inside that window used
    // to let the continuation spawn a llama-server AFTER stop() had already
    // drained the instance list — so nothing held a handle to it, it survived
    // the stop holding VRAM, and it kept the port the next start needed.
    let release;
    const gate = new Promise((r) => { release = r; });
    const { fleet, mgrs } = makeFleet({
      findFreePort: async (h, p) => { await gate; return p; },
    });

    const starting = fleet.start([{ index: 0, nGpuLayers: 42 }], {});
    fleet.stop();      // lands while the probe is in flight
    release();
    await starting;
    await drain();

    expect(mgrs).toHaveLength(0);          // no manager was ever constructed
    expect(fleet.instances).toHaveLength(0);
  });

  test('start() with an empty or non-array plan is a no-op', async () => {
    const { fleet, mgrs } = makeFleet();
    expect(await fleet.start([])).toBe(0);          // no run opts at all
    expect(await fleet.start(undefined, {})).toBe(0);
    expect(mgrs).toHaveLength(0);
  });

  test('_startNext() with nothing pending returns null instead of spawning', async () => {
    const { fleet, mgrs } = makeFleet();
    expect(await fleet._startNext()).toBeNull();
    expect(mgrs).toHaveLength(0);
  });

  test('constructs with no options at all', () => {
    const fleet = new LlmFleet();
    expect(fleet.instances).toEqual([]);
    expect(fleet.makeWorker('http://h:1', 0)).toBeNull(); // default factory
  });

  test('activeJobs() sums live workers and ignores instances without one', async () => {
    const { fleet, mgrs, workers } = makeFleet();
    await fleet.start([{ index: 0, nGpuLayers: 42 }, { index: 1, nGpuLayers: 42 }], {});
    await drain();
    fleet.syncWorkers(true);
    mgrs[0].emit('ready', { baseUrl: 'http://127.0.0.1:8080' });
    mgrs[1].emit('ready', { baseUrl: 'http://127.0.0.1:8081' });

    workers[0]._active = 2;
    workers[1]._active = 3;
    expect(fleet.activeJobs()).toBe(5);

    workers[1]._active = NaN; // a worker reporting junk counts as zero
    expect(fleet.activeJobs()).toBe(2);
  });

  test('_waitSettled resolves immediately for an instance that is already settled', async () => {
    const { fleet } = makeFleet();
    await expect(fleet._waitSettled(null)).resolves.toBeUndefined();
    await expect(fleet._waitSettled({ ready: true })).resolves.toBeUndefined();
    await expect(fleet._waitSettled({ stopped: true })).resolves.toBeUndefined();
  });

  test('_waitSettled settles once, whether by the instance or by the timeout', async () => {
    const { fleet, mgrs } = makeFleet({ startSettleMs: 5 });
    await fleet.start([{ index: 0, nGpuLayers: 42 }], {});
    const inst = fleet.instances[0];

    // Settled by the instance: the explicit settle wins and clears the timer,
    // and a second call is a no-op (the `done` guard).
    const waited = fleet._waitSettled(inst);
    const settle = inst._settle;
    settle();
    settle();
    await expect(waited).resolves.toBeUndefined();

    // Settled by the timeout: nothing calls _settle, so the timer fires. This is
    // the path that keeps one slow card from blocking the rest of the rig.
    const inst2 = { ready: false, stopped: false };
    await expect(fleet._waitSettled(inst2)).resolves.toBeUndefined();
    expect(inst2._settle).toBeNull();

    mgrs[0].emit('stopped');
  });
  test("the settle timer is unref'd when the runtime provides it, and tolerated when not", async () => {
    // Node's timers carry unref(); a stubbed/edge runtime may not. Neither shape
    // may keep the CLI process alive, so both must be accepted.
    const real = global.setTimeout;
    // A handle with no unref() at all — `delete t.unref` wouldn't do it, since
    // unref lives on Timeout.prototype.
    global.setTimeout = (fn, ms) => { real(fn, ms); return {}; };
    try {
      const { fleet } = makeFleet({ startSettleMs: 1 });
      await expect(fleet._waitSettled({ ready: false, stopped: false })).resolves.toBeUndefined();
    } finally {
      global.setTimeout = real;
    }
  });

  test('a background drain that blows up is swallowed, not left unhandled', async () => {
    // start() fires _drain() without awaiting it; an unhandled rejection there
    // would be fatal in Node, so the .catch is load-bearing.
    let calls = 0;
    const { fleet } = makeFleet({
      findFreePort: async (h, p) => {
        calls += 1;
        if (calls > 1) throw new Error('no free port');
        return p;
      },
    });
    await fleet.start([{ index: 0, nGpuLayers: 42 }, { index: 1, nGpuLayers: 42 }], {});
    await expect(fleet._draining).resolves.toBeUndefined();
  });

  test('takes a crashed instance out of service, then puts it back when it recovers', async () => {
    // A crash is not a fleet stop: the manager is restarting it. But while it is
    // down the instance must stop polling for cluster jobs and drop off
    // servingIndices(), or the node advertises a model that isn't listening and
    // fails every job it claims — which is exactly what the field showed.
    const { fleet, mgrs, workers } = makeFleet();
    const down = jest.fn();
    fleet.on('crashed', down);

    await fleet.start([{ index: 0 }], { modelPath: '/m.gguf' });
    fleet.syncWorkers(true);
    await drain();
    mgrs[0].emit('ready', { baseUrl: 'http://127.0.0.1:8080' });
    expect(fleet.servingIndices()).toEqual([0]);
    expect(fleet.isReady()).toBe(true);
    expect(workers.length).toBe(1);

    mgrs[0].emit('crashed', { code: 139, restartInMs: 3000, attempt: 1 });
    expect(fleet.servingIndices()).toEqual([]);   // not offered to the cluster
    expect(fleet.isReady()).toBe(false);
    expect(workers[0].stopped).toBe(true);
    expect(down).toHaveBeenCalledWith(expect.objectContaining({ code: 139, index: 0 }));

    mgrs[0].emit('ready', { baseUrl: 'http://127.0.0.1:8080' }); // restart landed
    expect(fleet.servingIndices()).toEqual([0]);
    expect(fleet.isReady()).toBe(true);
  });

  test('a crash with no worker attached is still taken out of service', () => {
    // Not serving cluster jobs (or crashed between ready and the worker being
    // wired): there is nothing to stop, and the instance must still stop counting
    // as ready.
    const { fleet, mgrs } = makeFleet();
    return fleet.start([{ index: 0 }], { modelPath: '/m.gguf' }).then(() => {
      mgrs[0].emit('ready', { baseUrl: 'http://127.0.0.1:8080' });
      expect(fleet.isReady()).toBe(true);
      mgrs[0].emit('crashed', { code: 1, restartInMs: 3000, attempt: 1 });
      expect(fleet.isReady()).toBe(false);
      expect(fleet.servingIndices()).toEqual([]);
    });
  });
});

describe('stopAndWait', () => {
  // stop() signals and returns; a 30 GB model is still tearing down when it does.
  // The caller restarting the miner needs the card back, not a promise that
  // someone has been told to give it back.
  //
  // One live instance, because startup is SERIALISED: start() reports the planned
  // count while the rest queue behind the first card's settle.
  const oneUp = async () => {
    const { fleet, mgrs } = makeFleet();
    await fleet.start([{ index: 0 }], {});
    mgrs.forEach((m) => { m.running = true; m.proc = {}; });
    return { fleet, mgrs };
  };
  const tick = () => new Promise((r) => setImmediate(r));

  test('does not resolve until the process has actually exited', async () => {
    const { fleet, mgrs } = await oneUp();
    let done = false;
    const p = fleet.stopAndWait(5000).then((t) => { done = true; return t; });
    await tick();
    expect(mgrs[0].stopped).toBe(true);        // signalled…
    expect(done).toBe(false);                  // …but not yet gone
    mgrs[0].emit('stopped', 0);
    await expect(p).resolves.toBe(false);      // false = exited, not timed out
  });

  test('does not wait on a manager whose process is already gone', async () => {
    const { fleet, mgrs } = await oneUp();
    mgrs.forEach((m) => { m.running = false; m.proc = null; });
    await expect(fleet.stopAndWait(5000)).resolves.toBe(false);
  });

  test('gives up rather than wedging the handback forever, and says so', async () => {
    // A stuck child is better reported by the miner failing to start than by
    // hanging here; the VRAM check downstream is the backstop.
    const { fleet } = await oneUp();
    const lines = [];
    fleet.on('log', (l) => lines.push(l.line));
    await expect(fleet.stopAndWait(10)).resolves.toBe(true);
    expect(lines.join(' ')).toContain('did not exit');
  });

  test('defaults its budget, and tolerates a timer with no unref', async () => {
    const realSetTimeout = global.setTimeout;
    global.setTimeout = (fn, ms) => { const h = realSetTimeout(fn, ms); return { _h: h }; };
    try {
      const { fleet, mgrs } = await oneUp();
      const p = fleet.stopAndWait();          // no argument → default budget
      mgrs[0].emit('stopped', 0);
      await expect(p).resolves.toBe(false);
    } finally {
      global.setTimeout = realSetTimeout;
    }
  });

  test('a fleet with nothing running resolves at once', async () => {
    const { fleet } = makeFleet();
    await expect(fleet.stopAndWait(5000)).resolves.toBe(false);
  });
});
