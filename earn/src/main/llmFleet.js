'use strict';

const { EventEmitter } = require('events');

// Supervises a FLEET of local llama-server instances — one per eligible GPU (see
// shared/llmPlan.planLlmInstances) — plus one cluster job-worker per ready
// instance, so a multi-GPU rig serves the model from every card that has room
// instead of only the single best card. Both the GUI (main.js) and the headless
// CLI drive this same fleet; process/GPU specifics are injected so it's fully
// unit-testable without real processes or a GPU.
//
// Injected factories/options:
//   makeManager()                    -> an LlmManager-like { start(opts), stop(),
//                                       on(ev,fn), baseUrl } (one llama-server)
//   findFreePort(host, port, tries)  -> Promise<number>  (a distinct free port
//                                       per instance; the fleet walks upward)
//   makeWorker(baseUrl, index)       -> a JobWorker-like { start(), stop(),
//                                       activeJobs() } fully wired by the caller,
//                                       or null to not serve this instance
//   host, basePort                   -> where the first instance binds
//
// Aggregate events:
//   log      { level, line }   re-emitted from every instance
//   ready    { baseUrl, index } each time an instance first becomes ready
//   first-ready { baseUrl }    once, on the first instance to become ready
//   stats    { tokensPerSec }  latest tok/s from any instance
//   stopped                    once, when the last running instance stops
//   error    Error             re-emitted (a listener is required, as ever)
class LlmFleet extends EventEmitter {
  constructor(opts = {}) {
    super();
    this.makeManager = opts.makeManager;
    this.findFreePort = opts.findFreePort;
    this.makeWorker = opts.makeWorker || (() => null);
    this.host = opts.host;
    this.basePort = opts.basePort;
    // How long one instance gets to load before the next is started anyway.
    this.startSettleMs = opts.startSettleMs || 600000;
    this.instances = []; // { index, port, mgr, ready, stopped, baseUrl, worker }
    this._serve = false;
    this._lastTps = 0;
    this._sawFirstReady = false;
    this._stopping = false;
    this._downEmitted = false;
    this._pending = [];   // plan entries not yet spawned (serialised startup)
    this._run = null;
    this._nextPort = null;
  }

  // Spawn one llama-server per plan entry ([{ index, nGpuLayers }, …]). `run`
  // carries the shared spawn bits (binaryPath, modelPath, platform). Returns the
  // number of instances launched.
  //
  // Instances start ONE AT A TIME: each is given until it reports ready (or
  // dies) before the next is spawned. Loading is the expensive part — every
  // server streams the whole multi-GB GGUF off disk to reach its card — and
  // starting N of them at once makes N processes fight over the page cache,
  // evicting each other's pages until throughput collapses. A 13-GPU rig with
  // 8 GB of RAM sat there with all 13 alive, answering 503, and not a byte in
  // VRAM. Serialised, the first load warms the cache and the rest read from it,
  // so the fleet comes up steadily instead of livelocking. Once loaded the
  // weights live in VRAM, so the instances coexist happily — it was only ever
  // the simultaneous loads that hurt.
  // Resolves once the FIRST instance is spawned, returning how many are planned.
  // The rest follow in the background, so the caller isn't blocked for minutes
  // (the headless CLI starts mining right after this returns — a blocking fleet
  // would hold the miner hostage while the models load).
  async start(plan, run = {}) {
    const entries = Array.isArray(plan) ? plan : [];
    if (!entries.length) return 0;
    this._pending = entries.slice();
    this._run = run;
    this._nextPort = this.basePort;
    await this._startNext();
    // Background: each subsequent instance waits for its predecessor to settle.
    // Errors here are already surfaced per-instance via 'error'/'log'.
    this._draining = this._drain().catch(() => {});
    return entries.length;
  }

  async _drain() {
    while (this._pending.length && !this._stopping) {
      await this._waitSettled(this.instances[this.instances.length - 1]);
      if (this._stopping) return;
      await this._startNext();
    }
  }

  async _startNext() {
    const e = this._pending.shift();
    if (!e) return null;
    const port = await this.findFreePort(this.host, this._nextPort, 10);
    this._nextPort = port + 1; // the next instance probes from the following port
    const mgr = this.makeManager();
    const inst = { index: e.index, port, mgr, ready: false, stopped: false, baseUrl: null, worker: null };
    this.instances.push(inst);
    mgr.on('log', (l) => this.emit('log', l));
    mgr.on('ready', ({ baseUrl }) => this._onReady(inst, baseUrl));
    mgr.on('stats', ({ tokensPerSec }) => this._onStats(tokensPerSec));
    mgr.on('stopped', (code) => this._onStopped(inst, code));
    mgr.on('error', (err) => this.emit('error', err));
    mgr.start(Object.assign({}, this._run, {
      host: this.host,
      port,
      nGpuLayers: e.nGpuLayers,
      mainGpu: e.index == null ? undefined : e.index,
    }));
    inst.baseUrl = mgr.baseUrl;
    return inst;
  }

  // Adopt an already-healthy llama-server (e.g. one lingering on our port right
  // after an "Update & restart") as a ready instance, without spawning a process
  // — reusing it beats spawning a second server that would double-load the model
  // and risk an OOM. It has no GPU index (unknown placement), so it never appears
  // in servingIndices(). Returns the adopted instance.
  adopt(baseUrl) {
    const inst = { index: null, port: null, mgr: null, ready: true, stopped: false, baseUrl, worker: null };
    this.instances.push(inst);
    this._sawFirstReady = true;
    if (this._serve) this._ensureWorker(inst);
    return inst;
  }

  // Resolve when `inst` has either become ready or stopped for good, or when
  // startSettleMs elapses. The timeout matters: a server that neither loads nor
  // exits must not strand the rest of the fleet forever, so a slow card delays
  // its successors rather than blocking them.
  _waitSettled(inst) {
    if (!inst || inst.ready || inst.stopped) return Promise.resolve();
    return new Promise((resolve) => {
      let done = false;
      const finish = () => {
        if (done) return;
        done = true;
        inst._settle = null;
        clearTimeout(timer);
        resolve();
      };
      const timer = setTimeout(finish, this.startSettleMs);
      if (timer.unref) timer.unref();
      inst._settle = finish;
    });
  }

  _onReady(inst, baseUrl) {
    inst.ready = true;
    inst.baseUrl = baseUrl;
    if (inst._settle) inst._settle();
    if (this._serve) this._ensureWorker(inst);
    this.emit('ready', { baseUrl, index: inst.index });
    if (!this._sawFirstReady) {
      this._sawFirstReady = true;
      this.emit('first-ready', { baseUrl });
    }
  }

  _onStats(tokensPerSec) {
    const tps = Number(tokensPerSec) || 0;
    this._lastTps = tps;
    this.emit('stats', { tokensPerSec: tps });
  }

  _onStopped(inst, code) {
    inst.ready = false;
    inst.stopped = true;
    if (inst._settle) inst._settle();
    if (inst.worker) { inst.worker.stop(); inst.worker = null; }
    // Surface a single fleet-level 'stopped' once every instance has stopped —
    // not on each individual card (others may still be serving). Forward the
    // last card's exit code so the headless CLI can exit with it.
    if (this._stopping || this._downEmitted) return;
    if (this.instances.length && this.instances.every((i) => i.stopped)) {
      this._downEmitted = true;
      this.emit('stopped', code);
    }
  }

  // Turn cluster-serving on/off across every ready instance. Idempotent — called
  // when the node link changes and whenever an instance becomes ready.
  syncWorkers(enabled) {
    this._serve = !!enabled;
    for (const inst of this.instances) {
      if (this._serve && inst.ready) this._ensureWorker(inst);
      else if (!this._serve && inst.worker) { inst.worker.stop(); inst.worker = null; }
    }
  }

  _ensureWorker(inst) {
    if (inst.worker) return;
    const w = this.makeWorker(inst.baseUrl, inst.index);
    if (!w) return;
    inst.worker = w;
    w.start();
  }

  // The endpoint the in-app chat talks to: the first ready instance's base URL.
  webUrl() {
    const inst = this.instances.find((i) => i.ready);
    return inst ? inst.baseUrl : null;
  }

  // True once at least one *spawned* instance is still running (adopted instances
  // have no manager). The caller uses this to avoid re-spawning a live fleet
  // while still allowing the adopt path to re-run after a restart.
  hasSpawned() { return this.instances.some((i) => i.mgr && !i.stopped); }

  isReady() { return this.instances.some((i) => i.ready); }
  readyCount() { return this.instances.filter((i) => i.ready).length; }
  tokensPerSec() { return this._lastTps; }

  // Total jobs in flight across every instance's worker (for node telemetry).
  activeJobs() {
    return this.instances.reduce((a, i) => a + (i.worker ? Number(i.worker.activeJobs()) || 0 : 0), 0);
  }

  // GPU indices currently serving the model (ready instances). Drives the miner
  // board's "serving LLM" column. A null index (unmeasured single instance) is
  // dropped — we only tag cards we can actually attribute.
  servingIndices() {
    return this.instances.filter((i) => i.ready && i.index != null).map((i) => i.index);
  }

  stop() {
    this._stopping = true;
    this._pending = []; // nothing further should spawn
    for (const inst of this.instances) {
      if (inst._settle) inst._settle(); // unblock a serialised start mid-flight
      if (inst.worker) { inst.worker.stop(); inst.worker = null; }
      if (inst.mgr) inst.mgr.stop();
    }
    this.instances = [];
    this._serve = false;
  }
}

module.exports = { LlmFleet };
