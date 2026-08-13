'use strict';

const { EventEmitter } = require('events');
const { parseLine } = require('../shared/parser');
const { resolveBinary, buildArgs } = require('../shared/minerArgs');

// Cap on the partial line held between stdout chunks — see _onData. Comfortably
// above any real engine line (the longest, a status, is a few hundred chars).
const MAX_LINE_BYTES = 64 * 1024;

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
  // `planFor` selects a non-alpha-miner engine: given the settings it returns
  // { bin, args } and the manager spawns that instead. SRBMiner needs a
  // different executable and a different argument vector (see srbArgs), and
  // routing it through here keeps process supervision — restart, stderr,
  // exit codes, the partial-line buffer — in one place rather than forked per
  // engine. Without one the alpha-miner resolution below is used unchanged.
  constructor({ spawn, planFor } = {}) {
    super();
    this.spawn = spawn;
    this.planFor = planFor || null;
    this.proc = null;
    this.running = false;
    this.stdoutBuf = '';
  }

  isRunning() {
    return this.running;
  }

  start(settings = {}) {
    if (this.running) return false;

    const plan = this.planFor
      ? this.planFor(settings)
      : { bin: resolveBinary(settings.binaryPath, settings.platform), args: buildArgs(settings) };
    const bin = plan.bin;
    const args = plan.args;
    const proc = this.spawn(bin, args);

    this.proc = proc;
    this.running = true;
    this.stdoutBuf = '';

    proc.stdout.on('data', (chunk) => this._onData(chunk));
    proc.stderr.on('data', (chunk) => {
      this.emit('log', { level: 'error', line: String(chunk).trim() });
    });
    proc.on('exit', (code) => {
      this.running = false;
      this.proc = null;
      this._flush(); // a last line with no trailing newline still counts
      this.emit('stopped', code);
    });
    proc.on('error', (err) => {
      this.emit('error', err);
    });

    this.emit('started', { bin, args });
    return true;
  }

  // A stdout 'data' chunk is a slice of the pipe, not a line: the engine's
  // status lines routinely straddle a chunk boundary. Splitting each chunk on
  // its own emitted the halves as two entries — the log filled with fragments
  // like `power=44` followed by `9W`, and parseLine saw neither as a status. Hold
  // the trailing partial back until the rest of it arrives.
  _onData(chunk) {
    this.stdoutBuf += String(chunk);
    const parts = this.stdoutBuf.split(/\r?\n/);
    this.stdoutBuf = parts.pop();
    for (const line of parts) this._line(line);
    // An engine that emits a huge line with no newline (or binary garbage) must
    // not grow this buffer without bound — past the cap, flush what's held and
    // treat the next chunk as a fresh line. After the complete lines above, so
    // the forced flush can't jump ahead of them.
    if (this.stdoutBuf.length > MAX_LINE_BYTES) this._flush();
  }

  // Emit whatever partial line is held, if any.
  _flush() {
    const rest = this.stdoutBuf;
    this.stdoutBuf = '';
    if (rest) this._line(rest);
  }

  _line(line) {
    if (!line) return;
    this.emit('log', { level: 'info', line });
    const evt = parseLine(line);
    if (evt) this.emit('event', evt);
  }

  stop() {
    if (!this.proc) return false;
    this.proc.kill();
    return true;
  }
}

module.exports = { MinerManager };
