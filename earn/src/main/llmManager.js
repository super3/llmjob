'use strict';

const { EventEmitter } = require('events');
const {
  resolveServerBinary, buildServerArgs, serverBaseUrl, isServerReady, parseTiming,
} = require('../shared/llama');

// Spawns and supervises the llama.cpp `llama-server` child process. Like
// MinerManager, `spawn` is injected so this is testable without a real binary or
// GPU. Readiness is detected from the server's own "listening" log line (main.js
// can additionally poll /health), and tokens/sec is scraped from timing lines.
//
// Self-heal: llama-server exits immediately when it can't bind its fixed port
// ("couldn't bind HTTP server socket"), which happens when a previous server is
// still releasing port 8080 — e.g. right after an "Update & restart", when the
// outgoing app's server overlaps the resumed one. An early exit *before* the
// server ever became ready is therefore retried a few times (`startAttempts`,
// spaced by `retryDelayMs`) instead of being reported as a dead LLM, so it comes
// up on its own once the port frees.
//
// An exit *after* it was ready is either the user stopping us or a crash, and
// those need opposite responses. `_stopping` tells them apart: it is set only by
// stop(). A crash therefore respawns too — llama-server dying mid-generation was
// silently permanent, because this branch treated every post-ready exit as
// intentional. In the field that took nodes out of the serving pool for hours at
// a time while the miner kept running, so the machine still looked healthy on the
// board with no model loaded and nothing ever restarting it.
//
// Crash restarts back off and are capped (`crashRestarts`), so a genuinely broken
// install surfaces as a stopped LLM instead of spinning forever; staying up for
// `crashSettleMs` counts as recovered and clears the tally.
//
// Events:
//   started  { bin, args, baseUrl }
//   ready    { baseUrl }
//   log      { level: 'info'|'error', line }
//   stats    { tokensPerSec }
//   crashed  { code, restartInMs, attempt }   — down, restarting shortly
//   stopped  exitCode
//   error    Error
class LlmManager extends EventEmitter {
  constructor({ spawn, sleep, startAttempts, retryDelayMs, crashRestarts, crashSettleMs, now } = {}) {
    super();
    this.spawn = spawn;
    this.sleep = sleep || ((ms) => new Promise((r) => setTimeout(r, ms)));
    this.now = now || Date.now;
    this.startAttempts = startAttempts || 1; // total spawn attempts (1 = no retry)
    this.retryDelayMs = retryDelayMs || 3000;
    // Restarts after a crash, i.e. an exit once the server had been serving.
    this.crashRestarts = crashRestarts == null ? 5 : crashRestarts;
    // Serving this long before dying counts as a recovery, not a crash loop.
    this.crashSettleMs = crashSettleMs == null ? 5 * 60 * 1000 : crashSettleMs;
    this.proc = null;
    this.running = false;
    this.ready = false;
    this.baseUrl = null;
    this._opts = null;
    this._attempt = 0;
    this._stopping = false;
    this._crashes = 0;
    this._readyAt = 0;
    // Context sizes to try, largest first, and where we are in that list. A
    // large-context model is the one case where "it won't start" is a sizing
    // problem we can actually fix ourselves — see _onExit.
    this._ladder = [];
    this._ctxIdx = 0;
  }

  isRunning() { return this.running; }
  isReady() { return this.ready; }

  start(opts = {}) {
    if (this.running) return false;
    this._opts = opts;
    this._attempt = 0;
    this._stopping = false;
    this._crashes = 0;
    this._readyAt = 0;
    // A model may offer several context sizes to try. Anything else keeps the
    // single size the caller asked for, so behaviour is unchanged without one.
    this._ladder = Array.isArray(opts.ctxLadder) ? opts.ctxLadder.filter((n) => Number(n) > 0) : [];
    this._ctxIdx = 0;
    this.running = true;
    this.baseUrl = serverBaseUrl(opts);
    this._spawn();
    return true;
  }

  // The context size this spawn should ask for: the current rung of the ladder,
  // or whatever the caller configured when there is no ladder.
  _ctxSize() {
    return this._ladder.length ? this._ladder[this._ctxIdx] : this._opts.ctxSize;
  }

  _spawn() {
    this._attempt++;
    const bin = resolveServerBinary(this._opts.binaryPath, this._opts.platform);
    const args = buildServerArgs({ ...this._opts, ctxSize: this._ctxSize() });
    const proc = this.spawn(bin, args);

    this.proc = proc;
    this.ready = false;

    if (proc.stdout) proc.stdout.on('data', (c) => this._onData(c));
    if (proc.stderr) proc.stderr.on('data', (c) => this._onData(c)); // llama logs to stderr
    proc.on('exit', (code) => this._onExit(code));
    proc.on('error', (err) => this.emit('error', err));

    this.emit('started', { bin, args, baseUrl: this.baseUrl });
  }

  _onExit(code) {
    const wasReady = this.ready;
    const servedFor = wasReady ? this.now() - this._readyAt : 0;
    this.proc = null;
    this.ready = false;

    // Exited before ever becoming ready, not a user stop, and attempts remain:
    // most likely a port-bind clash that clears once the previous server dies —
    // wait and re-spawn instead of declaring the LLM dead.
    if (!wasReady && !this._stopping && this._attempt < this.startAttempts) {
      this.emit('log', { level: 'info', line: 'local LLM exited before it was ready — retrying (attempt ' + this._attempt + '/' + this.startAttempts + ')' });
      this._retryIn(this.retryDelayMs);
      return;
    }

    // Still never became ready, and the retries above are spent. If the model
    // offered smaller context sizes, the size is the likeliest thing we got
    // wrong: the KV cache for a very large window is the one cost we cannot read
    // off the weights ahead of time, so a card that is short by a little would
    // otherwise restart forever. Drop to the next rung and give it a full budget
    // of attempts there.
    //
    // Deliberately AFTER the retry branch, not instead of it: a port-bind clash
    // also exits before ready, and shrinking the context because the previous
    // server had not finished releasing 8080 would be the wrong fix applied
    // permanently.
    if (!wasReady && !this._stopping && this._ctxIdx + 1 < this._ladder.length) {
      const from = this._ladder[this._ctxIdx];
      this._ctxIdx++;
      this._attempt = 0;
      const to = this._ladder[this._ctxIdx];
      this.emit('log', { level: 'warn', line: 'local LLM would not start at a ' + from + '-token context — retrying at ' + to });
      this.emit('ctx-downgrade', { from, to });
      this._retryIn(this.retryDelayMs);
      return;
    }

    // It was serving and we didn't stop it: llama-server crashed. Bring it back.
    if (wasReady && !this._stopping) {
      // A server that stayed up a good while and then died is an isolated crash,
      // not a loop — forgive the earlier ones so a long-lived node keeps healing.
      if (servedFor >= this.crashSettleMs) this._crashes = 0;
      if (this._crashes < this.crashRestarts) {
        this._crashes++;
        // Back off per consecutive crash so a model that dies on every request
        // doesn't spin the GPU, while a one-off comes back in seconds.
        const wait = this.retryDelayMs * this._crashes;
        this.emit('log', { level: 'error', line: 'local LLM crashed (code ' + code + ') after serving — restarting in ' + Math.round(wait / 1000) + 's (' + this._crashes + '/' + this.crashRestarts + ')' });
        // Distinct from 'stopped', which means the LLM is gone for good. This says
        // "down, coming back": the fleet must stop offering this instance to the
        // cluster now, or it keeps advertising a model that isn't listening and
        // fails every job it is handed until the restart lands.
        this.emit('crashed', { code, restartInMs: wait, attempt: this._crashes });
        this._attempt = 0; // a crash restart gets the full startup-retry budget
        this._retryIn(wait);
        return;
      }
      this.emit('log', { level: 'error', line: 'local LLM crashed ' + this._crashes + ' times without staying up — giving up. See Logs.' });
    }

    this.running = false;
    this.emit('stopped', code);
  }

  // Re-spawn after `ms`, unless we were stopped while waiting.
  _retryIn(ms) {
    this.sleep(ms).then(() => {
      if (this._stopping || !this.running) return;
      this._spawn();
    });
  }

  _onData(chunk) {
    const lines = String(chunk).split(/\r?\n/).filter(Boolean);
    for (const line of lines) {
      this.emit('log', { level: 'info', line });
      if (!this.ready && isServerReady(line)) {
        this.ready = true;
        this._readyAt = this.now();
        this.emit('ready', { baseUrl: this.baseUrl });
      }
      // Exactly one of the two is non-null per line, so a consumer can tell which
      // phase this measurement describes instead of guessing from magnitude.
      const t = parseTiming(line);
      if (t) {
        this.emit('stats', t.kind === 'prompt'
          ? { tokensPerSec: null, promptTokensPerSec: t.tokensPerSec }
          : { tokensPerSec: t.tokensPerSec, promptTokensPerSec: null });
      }
    }
  }

  stop() {
    if (!this.running && !this.proc) return false;
    this._stopping = true; // cancels any pending retry
    this.running = false;
    if (this.proc) this.proc.kill();
    return true;
  }
}

module.exports = { LlmManager };
