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
    this.paused = false;
    this.platform = null;
  }

  isRunning() {
    return this.running;
  }

  isPaused() {
    return this.paused;
  }

  start(settings = {}) {
    if (this.running) return false;

    const bin = resolveBinary(settings.binaryPath, settings.platform);
    const args = buildArgs(settings);
    const proc = this.spawn(bin, args);

    this.proc = proc;
    this.running = true;
    this.paused = false;
    this.platform = settings.platform || null;

    proc.stdout.on('data', (chunk) => this._onData(chunk));
    proc.stderr.on('data', (chunk) => {
      this.emit('log', { level: 'error', line: String(chunk).trim() });
    });
    proc.on('exit', (code) => {
      this.running = false;
      this.paused = false;
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

  // Pause mining without unloading anything — freeze the engine process so the
  // GPU's compute is free for LLM inference (2–4× faster tok/s while a request
  // runs). SIGSTOP is instant and in-place; SIGCONT (resume) picks up exactly
  // where it left off, so there's no pool reconnect. POSIX-only: Windows has no
  // SIGSTOP, so pause is a no-op there and the rig simply co-runs as before.
  // Returns whether the state actually changed.
  pause() {
    if (!this.proc || this.paused || this.platform === 'win32') return false;
    try {
      this.proc.kill('SIGSTOP');
    } catch (e) {
      return false; // couldn't signal — leave it mining rather than lie
    }
    this.paused = true;
    return true;
  }

  resume() {
    if (!this.proc || !this.paused) return false;
    try {
      this.proc.kill('SIGCONT');
    } catch (e) {
      /* clear the flag anyway: a failed SIGCONT must not strand us "paused" */
    }
    this.paused = false;
    return true;
  }

  stop() {
    if (!this.proc) return false;
    this.proc.kill();
    return true;
  }
}

module.exports = { MinerManager };
