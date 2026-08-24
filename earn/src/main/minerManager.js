'use strict';

const { EventEmitter } = require('events');
const { parseLine } = require('../shared/parser');
const { resolveBinary, buildArgs } = require('../shared/minerArgs');

// Cap on the partial line held between chunks — see _onData. Comfortably above
// any real engine line (the longest, a status row, is a few hundred chars).
const MAX_LINE_BYTES = 64 * 1024;

// Spawns and supervises the PeakMiner child process, turning its output into
// structured events. `spawn` is injected (defaults wired up by the Electron main
// process to child_process.spawn) so the manager is testable without launching a
// real binary.
//
// BOTH streams are parsed, and that is not belt-and-braces. PeakMiner writes
// every line — banner, connect, accepted shares, the status table, errors — to
// STDERR, leaving stdout completely empty (measured: 6704 bytes of stderr to 0
// of stdout on a 50-second run). alpha-miner used stdout. Reading only stdout
// would give a dashboard that sits at zero for ever while the rig mines fine,
// and reading stderr as pure error text would paint the entire log red. So the
// two streams share one line buffer and the level comes from the line itself.
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
    this.lineBuf = '';
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
    this.lineBuf = '';

    proc.stdout.on('data', (chunk) => this._onData(chunk));
    proc.stderr.on('data', (chunk) => this._onData(chunk));
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

  // A 'data' chunk is a slice of the pipe, not a line: the engine's status
  // rows routinely straddle a chunk boundary. Splitting each chunk on
  // its own emitted the halves as two entries — the log filled with fragments
  // like `power=44` followed by `9W`, and parseLine saw neither as a status. Hold
  // the trailing partial back until the rest of it arrives.
  _onData(chunk) {
    this.lineBuf += String(chunk);
    const parts = this.lineBuf.split(/\r?\n/);
    this.lineBuf = parts.pop();
    for (const line of parts) this._line(line);
    // An engine that emits a huge line with no newline (or binary garbage) must
    // not grow this buffer without bound — past the cap, flush what's held and
    // treat the next chunk as a fresh line. After the complete lines above, so
    // the forced flush can't jump ahead of them.
    if (this.lineBuf.length > MAX_LINE_BYTES) this._flush();
  }

  // Emit whatever partial line is held, if any.
  _flush() {
    const rest = this.lineBuf;
    this.lineBuf = '';
    if (rest) this._line(rest);
  }

  // Level comes from the line, not from which pipe it arrived on — see the class
  // comment. PeakMiner stamps its own severity (`ERROR failed to connect to …`),
  // which is the only signal that actually distinguishes a problem from routine
  // progress now that everything shares one stream.
  _line(line) {
    if (!line) return;
    const level = /\bERROR\b/.test(line) ? 'error' : 'info';
    this.emit('log', { level, line });
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
