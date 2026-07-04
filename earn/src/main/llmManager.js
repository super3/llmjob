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
// Events:
//   started  { bin, args, baseUrl }
//   ready    { baseUrl }
//   log      { level: 'info'|'error', line }
//   stats    { tokensPerSec }
//   stopped  exitCode
//   error    Error
class LlmManager extends EventEmitter {
  constructor({ spawn } = {}) {
    super();
    this.spawn = spawn;
    this.proc = null;
    this.running = false;
    this.ready = false;
    this.baseUrl = null;
  }

  isRunning() { return this.running; }
  isReady() { return this.ready; }

  start(opts = {}) {
    if (this.running) return false;

    const bin = resolveServerBinary(opts.binaryPath, opts.platform);
    const args = buildServerArgs(opts);
    const proc = this.spawn(bin, args);

    this.proc = proc;
    this.running = true;
    this.ready = false;
    this.baseUrl = serverBaseUrl(opts);

    if (proc.stdout) proc.stdout.on('data', (c) => this._onData(c));
    if (proc.stderr) proc.stderr.on('data', (c) => this._onData(c)); // llama logs to stderr
    proc.on('exit', (code) => {
      this.running = false;
      this.ready = false;
      this.proc = null;
      this.emit('stopped', code);
    });
    proc.on('error', (err) => this.emit('error', err));

    this.emit('started', { bin, args, baseUrl: this.baseUrl });
    return true;
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
    if (!this.proc) return false;
    this.proc.kill();
    return true;
  }
}

module.exports = { LlmManager };
