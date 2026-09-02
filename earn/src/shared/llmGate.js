'use strict';

// Demand-driven auto mode: mine while nobody is asking for tokens, serve the LLM
// when they are. This is the "Phase 4" resolvePlan refers to.
//
// Why it cannot simply co-run. A node co-runs the LLM and the miner whenever both
// fit, and for the default model on a big card they do. They do not for a large
// tier: Qwen3.8 at 262144 takes 30,150 MiB of a 5090's 32,607, leaving ~2,457
// against the ~2,581 the miner's rank-128 profile plus its CUDA context need. The
// card is ~124 MiB short, so one of the two has to be stopped -- and stopped, not
// paused, because a suspended process keeps every byte of its VRAM.
//
// Measured on a 5090, that switch is cheap enough to do on demand:
//   mining -> serving   4.2 s   (miner stop 0.2 s, llama healthy 3.9 s)
//   serving -> mining   2.8 s   (llama stop 0.2 s, hashing 2.6 s)
// The 3.9 s load depends on the GGUF being in page cache; a cold first load after
// boot is disk-bound and much slower.

const EventEmitter = require('events');

// Paths that need the model loaded. Anything here wakes it.
const WAKE_PREFIXES = [
  '/completion', '/completions', '/v1/completions',
  '/chat/completions', '/v1/chat/completions',
  '/infill', '/embedding', '/embeddings', '/v1/embeddings',
  '/v1/responses', '/apply-template', '/tokenize', '/detokenize',
  '/rerank', '/v1/rerank',
];

// Paths answered from the gate's own state when the model is down.
//
// THIS IS LOAD-BEARING, not tidiness. Monitoring polls /health continuously -- our
// own dashboard does it every second. If a probe counted as demand the model would
// wake on the first poll and never go back to mining, and auto mode would quietly
// become llm mode. Probes must be answerable without the model.
const PASSIVE_PREFIXES = ['/health', '/v1/models', '/models', '/props', '/metrics', '/slots'];

function classifyPath(pathname) {
  const p = String(pathname == null ? '' : pathname).split('?')[0];
  if (WAKE_PREFIXES.some((w) => p === w || p.startsWith(w + '/'))) return 'wake';
  if (PASSIVE_PREFIXES.some((w) => p === w || p.startsWith(w + '/'))) return 'passive';
  // Unknown paths are treated as demand. A new llama-server endpoint we have not
  // listed should serve correctly-but-slowly rather than 404 while the model sits
  // stopped; the failure mode of guessing wrong the other way is worse.
  return 'wake';
}

const MINING = 'MINING';
const SERVING = 'SERVING';
const TO_SERVING = 'SWITCHING->SERVING';
const TO_MINING = 'SWITCHING->MINING';

class LlmGate extends EventEmitter {
  // Deps are injected so the state machine can be tested without spawning
  // anything: startLlm/stopLlm/startMiner/stopMiner return promises, isLlmReady
  // reports whether the server currently answers.
  constructor(opts = {}) {
    super();
    this.startLlm = opts.startLlm;
    this.stopLlm = opts.stopLlm;
    this.startMiner = opts.startMiner;
    this.stopMiner = opts.stopMiner;
    this.isLlmReady = opts.isLlmReady || (() => false);
    this.quietMs = opts.quietMs == null ? 60000 : opts.quietMs;
    this.now = opts.now || (() => Date.now());
    // A gate in front of an already-loaded model starts SERVING: there is no
    // miner holding the card, so calling the initial state MINING would be a lie
    // that /health reports until the first request happens to correct it.
    this.state = opts.state || MINING;
    this.inFlight = 0;
    this.lastRequestAt = this.now();
    this._transition = null;   // single-flight: one switch at a time
  }

  _setState(s) {
    if (this.state === s) return;
    this.state = s;
    this.emit('state', s);
  }

  // Bring the LLM up, stopping the miner first to free its VRAM. Concurrent
  // callers share one transition rather than racing to start two servers.
  ensureServing() {
    if (this.state === SERVING && this.isLlmReady()) return Promise.resolve(true);
    // Only join a transition heading the SAME way. _transition is one slot shared
    // with ensureMining, so a request arriving during a release used to await the
    // RELEASE, get `true`, and forward into a model that had just been stopped --
    // a 502 on the HTTP path, and on the cluster path a claimed job reported
    // FAILED. The window is the whole release: stopLlm waits for the model's VRAM
    // to come back, up to 30s.
    if (this._transition && this.state === TO_SERVING) return this._transition;
    // Heading the other way: let the release finish, then wake from a settled
    // state rather than racing it.
    if (this._transition) return this._transition.then(() => this.ensureServing(), () => this.ensureServing());
    this._transition = (async () => {
      this._setState(TO_SERVING);
      if (this.stopMiner) await this.stopMiner();
      if (this.startLlm) await this.startLlm();
      this._setState(SERVING);
      return true;
    })().catch(async (e) => {
      // stopMiner() ran before startLlm(), so the miner is provably DOWN here.
      // Relabelling the state MINING without restarting it leaves the node
      // running neither engine, and nothing recovers on its own: the release
      // timer only fires from SERVING, and ensureMining() short-circuits on
      // MINING. The node would sit at zero, alive and answering /health.
      if (this.startMiner) { try { await this.startMiner(); } catch { /* nothing left to try */ } }
      this._setState(MINING);
      throw e;
    })
      .finally(() => { this._transition = null; });
    return this._transition;
  }

  // Hand the card back. Never while a request is in flight.
  ensureMining() {
    if (this.state === MINING) return Promise.resolve(true);
    if (this.inFlight > 0) return Promise.resolve(false);
    if (this._transition) return this._transition;
    this._transition = (async () => {
      this._setState(TO_MINING);
      if (this.stopLlm) await this.stopLlm();
      if (this.startMiner) await this.startMiner();
      this._setState(MINING);
      return true;
    })().finally(() => { this._transition = null; });
    return this._transition;
  }

  begin() { this.inFlight += 1; this.lastRequestAt = this.now(); }
  end() { this.inFlight = Math.max(0, this.inFlight - 1); this.lastRequestAt = this.now(); }

  // Time since the last request STARTED or FINISHED -- not since the GPU last
  // had work. While mining this climbs with the card at full load.
  quietFor() { return this.now() - this.lastRequestAt; }

  // Called on a timer. Flips back only when serving, nothing is in flight, and the
  // window has actually elapsed -- so a long generation can never be interrupted
  // by the clock.
  shouldRelease() {
    return this.state === SERVING && this.inFlight === 0 && this.quietFor() >= this.quietMs;
  }
}

module.exports = {
  LlmGate, classifyPath, WAKE_PREFIXES, PASSIVE_PREFIXES,
  MINING, SERVING, TO_SERVING, TO_MINING,
};
