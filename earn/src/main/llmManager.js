'use strict';

const { EventEmitter } = require('events');
const {
  resolveServerBinary, buildServerArgs, serverBaseUrl, isServerReady, parseTokensPerSec,
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
// up on its own once the port frees. An exit *after* it was ready is a real stop.
//
// Events:
//   started  { bin, args, baseUrl }
//   ready    { baseUrl }
//   log      { level: 'info'|'error', line }
//   stats    { tokensPerSec }
//   stopped  exitCode
//   error    Error
class LlmManager extends EventEmitter {
  constructor({ spawn, sleep, startAttempts, retryDelayMs } = {}) {
    super();
    this.spawn = spawn;
    this.sleep = sleep || ((ms) => new Promise((r) => setTimeout(r, ms)));
    this.startAttempts = startAttempts || 1; // total spawn attempts (1 = no retry)
    this.retryDelayMs = retryDelayMs || 3000;
    this.proc = null;
    this.running = false;
    this.ready = false;
    this.baseUrl = null;
    this._opts = null;
    this._attempt = 0;
    this._stopping = false;
  }

  isRunning() { return this.running; }
  isReady() { return this.ready; }

  start(opts = {}) {
    if (this.running) return false;
    this._opts = opts;
    this._attempt = 0;
    this._stopping = false;
    this.running = true;
    this.baseUrl = serverBaseUrl(opts);
    this._spawn();
    return true;
  }

  _spawn() {
    this._attempt++;
    const bin = resolveServerBinary(this._opts.binaryPath, this._opts.platform);
    const args = buildServerArgs(this._opts);
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
    this.proc = null;
    // Exited before ever becoming ready, not a user stop, and attempts remain:
    // most likely a port-bind clash that clears once the previous server dies —
    // wait and re-spawn instead of declaring the LLM dead.
    if (!this.ready && !this._stopping && this._attempt < this.startAttempts) {
      this.emit('log', { level: 'info', line: 'local LLM exited before it was ready — retrying (attempt ' + this._attempt + '/' + this.startAttempts + ')' });
      this.sleep(this.retryDelayMs).then(() => {
        if (this._stopping || !this.running) return; // stopped while we waited
        this._spawn();
      });
      return;
    }
    this.running = false;
    this.ready = false;
    this.emit('stopped', code);
  }

  _onData(chunk) {
    const lines = String(chunk).split(/\r?\n/).filter(Boolean);
    for (const line of lines) {
      this.emit('log', { level: 'info', line });
      if (!this.ready && isServerReady(line)) {
        this.ready = true;
        this.emit('ready', { baseUrl: this.baseUrl });
      }
      const tps = parseTokensPerSec(line);
      if (tps != null) this.emit('stats', { tokensPerSec: tps });
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
