'use strict';

const { EventEmitter } = require('events');
const { parseLine } = require('../shared/parser');
const { resolveBinary, buildArgs } = require('../shared/minerArgs');

// Spawns and supervises the `alpha-miner` child process, turning its stdout
// into structured events. `spawn` is injected (defaults wired up by the
// Electron main process to child_process.spawn) so the manager is testable
// without launching a real binary.
//
// Events:
//   started  { bin, args }
//   log      { level: 'info'|'error', line }
//   event    <parsed miner event>   (share / hashrate / connected)
//   stopped  exitCode
//   error    Error
class MinerManager extends EventEmitter {
  constructor({ spawn } = {}) {
    super();
    this.spawn = spawn;
    this.proc = null;
    this.running = false;
  }

  isRunning() {
    return this.running;
  }

  start(settings = {}) {
    if (this.running) return false;

    const bin = resolveBinary(settings.binaryPath, settings.platform);
    const args = buildArgs(settings);
    const proc = this.spawn(bin, args);

    this.proc = proc;
    this.running = true;

    proc.stdout.on('data', (chunk) => this._onData(chunk));
    proc.stderr.on('data', (chunk) => {
      this.emit('log', { level: 'error', line: String(chunk).trim() });
    });
    proc.on('exit', (code) => {
      this.running = false;
      this.proc = null;
      this.emit('stopped', code);
    });
    proc.on('error', (err) => {
      this.emit('error', err);
    });

    this.emit('started', { bin, args });
    return true;
  }

  _onData(chunk) {
    const lines = String(chunk).split(/\r?\n/).filter(Boolean);
    for (const line of lines) {
      this.emit('log', { level: 'info', line });
      const evt = parseLine(line);
      if (evt) this.emit('event', evt);
    }
  }

  stop() {
    if (!this.proc) return false;
    this.proc.kill();
    return true;
  }
}

module.exports = { MinerManager };
