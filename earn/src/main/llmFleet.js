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
//   stats    { tokensPerSec, promptTokensPerSec }  latest generation and prefill
//                                                  rates from any instance
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
    this._lastPromptTps = 0;
    this._sawFirstReady = false;
    this._stopping = false;
    this._downEmitted = false;
    this._pending = [];   // plan entries not yet spawned (serialised startup)
    this._spawning = false; // an entry is off the queue but not yet an instance
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
    // Reset the RUN-scoped flags. They are not constructor-scoped: a fleet is
    // reused across start/stop cycles, and leaving them set broke the second run
    // two ways. `_stopping` stayed true after a stop(), so _drain() returned
    // immediately and only the first GPU of a multi-GPU rig ever came up. And
    // `_downEmitted` stayed true after the fleet had died on its own, so the
    // next run could never emit 'stopped' again — the UI kept showing
    // "LLM ready" with no llama-server running.
    this._stopping = false;
    this._downEmitted = false;
    this._sawFirstReady = false;
    this._pending = entries.slice();
    this._run = run;
    this._nextPort = this.basePort;
    // The FIRST spawn is awaited, so its failure rejects start() — and would
    // leave the rest of the plan stranded in `_pending`, which the down-check
    // then treats as "cards still coming" forever. Abandon the queue before
    // rethrowing so the caller still gets the error but the fleet can report
    // down; the drain below does the same for every later spawn.
    try {
      await this._startNext();
    } catch (err) {
      this._abandonPending();
      throw err;
    }
    // Background: each subsequent instance waits for its predecessor to settle.
    // Errors here are already surfaced per-instance via 'error'/'log'.
    this._draining = this._drain().catch(() => this._abandonPending());
    return entries.length;
  }

  // A drain that threw (findFreePort exhausting its tries, say) leaves entries
  // stranded in `_pending`. Since the down-check refuses to declare the fleet
  // dead while cards are queued, those stranded entries would suppress
  // 'stopped' forever and leave the UI on "Starting…" with nothing running.
  // Drop them and re-run the check so a fleet whose spawned instances have all
  // died can still report it.
  _abandonPending() {
    this._pending = [];
    this._maybeEmitDown(1);
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
    // The entry is off the queue but its instance does not exist yet, and the
    // port probe below is async. Without this flag that window reads as "queue
    // empty, every instance stopped" to the down-check, so a card dying while
    // its successor probes for a port declared the whole fleet dead — the very
    // premature 'stopped' the queue guard exists to prevent, just moved a few
    // milliseconds later. `finally` clears it, so a probe that throws cannot
    // leave the fleet permanently unable to report down either.
    this._spawning = true;
    try {
      return await this._spawn(e);
    } finally {
      this._spawning = false;
    }
  }

  async _spawn(e) {
    const port = await this.findFreePort(this.host, this._nextPort, 10);
    // stop() may have landed while we were probing for a port. If it did, the
    // instance list has already been drained and every manager stopped, so
    // spawning now would put a llama-server on the machine that NOTHING holds a
    // handle to — it survives the stop, keeps the model in VRAM and keeps the
    // port bound, and the next start can't bind.
    if (this._stopping) return null;
    this._nextPort = port + 1; // the next instance probes from the following port
    const mgr = this.makeManager();
    const inst = { index: e.index, port, mgr, ready: false, stopped: false, baseUrl: null, worker: null };
    this.instances.push(inst);
    mgr.on('log', (l) => this.emit('log', l));
    mgr.on('ready', ({ baseUrl }) => this._onReady(inst, baseUrl));
    mgr.on('stats', (s) => this._onStats(s));
    mgr.on('crashed', (info) => this._onCrashed(inst, info));
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

  // llama-server died but the manager is restarting it. Take the instance out of
  // service without declaring the fleet down: its worker must stop polling for
  // cluster jobs (it would fail every one it claimed), and servingIndices() must
  // stop listing the card, so the board shows what is actually being served.
  // _onReady puts both back when the restart lands.
  _onCrashed(inst, info) {
    inst.ready = false;
    if (inst.worker) { inst.worker.stop(); inst.worker = null; }
    this.emit('crashed', Object.assign({ index: inst.index }, info));
  }

  // Each line carries ONE phase, so hold both and re-emit both every time. That
  // keeps the fleet's `tokensPerSec` a plain number for existing consumers while
  // making the prefill figure available to anyone who wants it.
  _onStats(s) {
    const gen = s && s.tokensPerSec;
    const pre = s && s.promptTokensPerSec;
    if (gen != null) this._lastTps = Number(gen) || 0;
    if (pre != null) this._lastPromptTps = Number(pre) || 0;
    this.emit('stats', { tokensPerSec: this._lastTps, promptTokensPerSec: this._lastPromptTps });
  }

  _onStopped(inst, code) {
    inst.ready = false;
    inst.stopped = true;
    if (inst._settle) inst._settle();
    if (inst.worker) { inst.worker.stop(); inst.worker = null; }
    // Surface a single fleet-level 'stopped' once every instance has stopped —
    // not on each individual card (others may still be serving). Forward the
    // last card's exit code so the headless CLI can exit with it.
    //
    // `_pending` is part of the test, not just `instances`: startup is
    // SERIALISED, so on a multi-GPU rig `instances` holds only the cards spawned
    // so far while the rest wait their turn. Without it, the first card dying
    // before it loads made every() trivially true and declared the whole fleet
    // down — the GUI ended the session and the CLI exited — moments before
    // _drain spawned the next card, which then loaded fine and served cluster
    // jobs with the UI insisting nothing was running. A card still queued means
    // the fleet has not had its chance yet.
    this._maybeEmitDown(code);
  }

  // Emit the one fleet-level 'stopped', if the fleet really is down.
  _maybeEmitDown(code) {
    if (this._stopping || this._downEmitted || this._pending.length || this._spawning) return;
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

  // Prefill rate, i.e. how fast the prompt was read. Separate from tokensPerSec
  // because they are different measurements, not two samples of one.
  promptTokensPerSec() { return this._lastPromptTps; }

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

  // stop(), but resolving only once every llama-server has actually EXITED.
  //
  // stop() signals and returns; the processes are still tearing down when it
  // does, and a 30 GB model does not release its VRAM the instant it is asked
  // to. The caller that restarts the miner needs the card back, not a promise
  // that someone has been told to give it back -- so join the exits the same way
  // autoGate joins the miner's, for the same reason.
  //
  // Listeners are attached BEFORE stop() or the event races us. A manager with
  // no live process resolves at once rather than waiting for an event that will
  // never come.
  stopAndWait(timeoutMs = 60000) {
    const waits = this.instances
      .map((inst) => inst.mgr)
      .filter(Boolean)
      .map((mgr) => new Promise((resolve) => {
        if (!mgr.running && !mgr.proc) return resolve();
        mgr.once('stopped', resolve);
      }));
    this.stop();
    if (!waits.length) return Promise.resolve(false);
    // Never wedge the handback on a server that will not die: the VRAM check
    // that follows is the backstop, and a stuck child is better reported by the
    // miner failing to start than by hanging here forever.
    let timer = null;
    const expiry = new Promise((resolve) => {
      timer = setTimeout(() => resolve(true), timeoutMs);
      if (timer.unref) timer.unref();
    });
    return Promise.race([Promise.all(waits).then(() => false), expiry])
      .then((timedOut) => {
        clearTimeout(timer);
        // Said here rather than by the caller: the fleet knows its children did
        // not die, and whoever is about to reuse the card needs it in the log.
        if (timedOut) {
          this.emit('log', {
            level: 'error',
            line: 'local LLM did not exit within ' + Math.round(timeoutMs / 1000)
              + 's — the card may still be held',
          });
        }
        return timedOut;
      });
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
