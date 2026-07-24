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
    makeWorker: (url, index) => { const w = new FakeWorker(url, index); workers.push(w); return w; },
  }, over);
  const fleet = new LlmFleet(opts);
  fleet.on('error', () => {}); // keep EventEmitter from throwing in tests
  return { fleet, mgrs, workers };
}

describe('LlmFleet', () => {
  test('start() launches one manager per plan entry on distinct ports', async () => {
    const { fleet, mgrs } = makeFleet();
    const n = await fleet.start(
      [{ index: 0, nGpuLayers: 42 }, { index: 1, nGpuLayers: 40 }],
      { binaryPath: 'b', modelPath: 'm', platform: 'linux' }
    );
    expect(n).toBe(2);
    expect(mgrs).toHaveLength(2);
    expect(mgrs[0].startOpts).toMatchObject({ host: '127.0.0.1', port: 8080, nGpuLayers: 42, mainGpu: 0, binaryPath: 'b', modelPath: 'm', platform: 'linux' });
    expect(mgrs[1].startOpts).toMatchObject({ port: 8081, nGpuLayers: 40, mainGpu: 1 });
    expect(fleet.instances[0].baseUrl).toBe('http://127.0.0.1:8080');
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
    expect(stats).toEqual([{ tokensPerSec: 42.5 }]);
    mgrs[0].emit('stats', { tokensPerSec: 'oops' });
    expect(fleet.tokensPerSec()).toBe(0);
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
    fleet.syncWorkers(true);
    mgrs[0].emit('ready', { baseUrl: 'http://127.0.0.1:8080' });
    mgrs[1].emit('ready', { baseUrl: 'http://127.0.0.1:8081' });
    const stops = [];
    fleet.on('stopped', () => stops.push(1));
    mgrs[0].emit('stopped');        // one card down, the other still serving
    expect(stops).toHaveLength(0);
    expect(workers[0].stopped).toBe(true); // its worker was torn down
    expect(fleet.readyCount()).toBe(1);
    mgrs[1].emit('stopped');        // last card down → one fleet 'stopped'
    expect(stops).toHaveLength(1);
    // never fires twice
    mgrs[1].emit('stopped');
    expect(stops).toHaveLength(1);
  });

  test('stop() tears down every worker and manager and goes quiet', async () => {
    const { fleet, mgrs, workers } = makeFleet();
    await fleet.start([{ index: 0, nGpuLayers: 42 }, { index: 1, nGpuLayers: 42 }], {});
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
    expect(mgrs[0].startOpts.port).toBe(8085);
    expect(mgrs[1].startOpts.port).toBe(8091); // 8086 probed → +5
  });
});
