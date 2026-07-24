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
    this.instances = []; // { index, port, mgr, ready, stopped, baseUrl, worker }
    this._serve = false;
    this._lastTps = 0;
    this._sawFirstReady = false;
    this._stopping = false;
    this._downEmitted = false;
  }

  // Spawn one llama-server per plan entry ([{ index, nGpuLayers }, …]). `run`
  // carries the shared spawn bits (binaryPath, modelPath, platform). Returns the
  // number of instances launched.
  async start(plan, run = {}) {
    const entries = Array.isArray(plan) ? plan : [];
    let port = this.basePort;
    for (const e of entries) {
      port = await this.findFreePort(this.host, port, 10);
      const mgr = this.makeManager();
      const inst = { index: e.index, port, mgr, ready: false, stopped: false, baseUrl: null, worker: null };
      this.instances.push(inst);
      mgr.on('log', (l) => this.emit('log', l));
      mgr.on('ready', ({ baseUrl }) => this._onReady(inst, baseUrl));
      mgr.on('stats', ({ tokensPerSec }) => this._onStats(tokensPerSec));
      mgr.on('stopped', () => this._onStopped(inst));
      mgr.on('error', (err) => this.emit('error', err));
      mgr.start(Object.assign({}, run, {
        host: this.host,
        port,
        nGpuLayers: e.nGpuLayers,
        mainGpu: e.index == null ? undefined : e.index,
      }));
      inst.baseUrl = mgr.baseUrl;
      port += 1; // next instance probes from the following port
    }
    return this.instances.length;
  }

  _onReady(inst, baseUrl) {
    inst.ready = true;
    inst.baseUrl = baseUrl;
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

  _onStopped(inst) {
    inst.ready = false;
    inst.stopped = true;
    if (inst.worker) { inst.worker.stop(); inst.worker = null; }
    // Surface a single fleet-level 'stopped' once every instance has stopped —
    // not on each individual card (others may still be serving).
    if (this._stopping || this._downEmitted) return;
    if (this.instances.length && this.instances.every((i) => i.stopped)) {
      this._downEmitted = true;
      this.emit('stopped');
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
    for (const inst of this.instances) {
      if (inst.worker) { inst.worker.stop(); inst.worker = null; }
      if (inst.mgr) inst.mgr.stop();
    }
    this.instances = [];
    this._serve = false;
  }
}

module.exports = { LlmFleet };
