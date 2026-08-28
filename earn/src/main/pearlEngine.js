'use strict';

const { EventEmitter } = require('events');
const { PearlMiner } = require('./pearlMiner');
const { PROFILE } = require('../shared/miner/pearlhash');

// Our own Pearl miner, wearing MinerManager's clothes.
//
// The app's mining UI is driven by two parsed events — a periodic `status`
// carrying hashrate and cumulative share counters, and a `connected` naming the
// endpoint and card. alpha-miner produces those by having its stdout scraped;
// PearlMiner produces richer, differently-shaped events because it IS the
// miner rather than a process being watched.
//
// This adapter is the whole difference. It exposes exactly the surface
// main.js already drives (start/stop/isRunning, and log/event/error/stopped),
// so choosing an engine is a choice of constructor and nothing else. Keeping
// the translation here rather than in main.js is what makes it testable without
// Electron, a socket, or a GPU.
//
// Temperature comes from an injected `readTemps` (nvidia-smi, via probe.js),
// sampled on a slow timer. alpha-miner read it from NVML and printed it on
// every status line; our miner IS the core and has no such reading to forward,
// so the engine polls for it instead. Injected rather than required so this
// stays testable without a GPU, and absent when nothing answers -- the UI
// renders the bare card name for a rig that reports no temperature, which is
// still better than inventing a number.
//
// Deliberately still NOT reported: power. Nothing in the UI shows it, and the
// per-card figure would go to the network board, which is a behaviour change
// rather than a display one.
const TEMP_POLL_MS = 5000;

class PearlEngine extends EventEmitter {
  constructor({ connect, createCore, profile, readTemps, tempPollMs } = {}) {
    super();
    this.connect = connect;
    this.createCore = createCore;
    this.profile = profile || PROFILE;
    this.readTemps = readTemps || null;
    this.tempPollMs = tempPollMs || TEMP_POLL_MS;
    this.temp = null;
    this.tempTimer = null;
    this.miner = null;

    // Cumulative for the session, because that is what the UI's counters mean.
    // PearlMiner reports each verdict as it lands, one event per share.
    this.accepted = 0;
    this.rejected = 0;
    this.hashrate = 0;
    this.gpu = null;
    this.endpoint = null;
  }

  isRunning() {
    return !!(this.miner && this.miner.isRunning());
  }

  start(settings = {}) {
    this.accepted = 0;
    this.rejected = 0;
    this.hashrate = 0;
    this.gpu = settings.gpu || null;
    this.endpoint = settings.endpoint || null;

    const m = new PearlMiner({ connect: this.connect, createCore: this.createCore });
    this.miner = m;

    m.on('log', (l) => this.emit('log', l));
    m.on('error', (err) => this.emit('error', err));
    m.on('stopped', () => this.emit('stopped', 0));

    // The pool accepted the wallet and we have work: the card is mining. This is
    // the moment alpha-miner prints its connection banner, and the UI wants the
    // endpoint and card name from it.
    m.on('job', () => {
      if (this._announced) return;
      this._announced = true;
      this.emit('event', {
        type: 'connected',
        gpuIndex: 0,
        endpoint: this.endpoint,
        gpu: this.gpu,
      });
    });

    // Forwarded as a parsed event because that is the shape main.js's DNS hint
    // reads, and a name that does not resolve is worth saying plainly.
    m.on('connect-failed', (e) => this.emit('event', Object.assign({ type: 'connect-failed' }, e)));

    m.on('share', () => { this.accepted++; this._status(); });
    m.on('rejected', () => { this.rejected++; this._status(); });

    // The core's own throughput tick drives the sparkline. Both sides count
    // multiply-accumulates per second in TH/s, which is the unit the UI's
    // hashrate field already carries.
    m.on('hashrate', (th) => { this.hashrate = th; this._status(); });

    this._announced = false;
    this._startTemps();
    m.start(Object.assign({}, settings, { profile: this.profile }));
  }

  stop() {
    this._stopTemps();
    if (this.miner) this.miner.stop();
  }

  // Poll the card temperature while mining. Sampled once immediately so the
  // reading appears with the first status rather than five seconds into the
  // run, then on a timer -- a status event fires on every share and every
  // hashrate tick, and spawning nvidia-smi that often would cost more than the
  // number is worth. unref'd so it can never hold the process open.
  _startTemps() {
    if (!this.readTemps || this.tempTimer) return;
    const sample = () => {
      let p;
      // A reader that throws synchronously must not take the miner down with
      // it; a temperature is the least important thing on the screen.
      try {
        p = Promise.resolve(this.readTemps());
      } catch (e) {
        this.temp = null;
        return;
      }
      p.then((temps) => {
        const t = temps ? Number(temps[this.gpuIndex()]) : NaN;
        this.temp = Number.isFinite(t) && t > 0 ? t : null;
      }).catch(() => { this.temp = null; });
    };
    sample();
    this.tempTimer = setInterval(sample, this.tempPollMs);
    if (this.tempTimer.unref) this.tempTimer.unref();
  }

  _stopTemps() {
    if (this.tempTimer) clearInterval(this.tempTimer);
    this.tempTimer = null;
    this.temp = null;
  }

  // The card this engine mines on. One core, one GPU, so index 0 -- kept as a
  // method so the reading follows if that ever stops being true.
  gpuIndex() {
    return 0;
  }

  _status() {
    this.emit('event', {
      type: 'status',
      gpuIndex: this.gpuIndex(),
      hashrate: this.hashrate,
      accepted: this.accepted,
      rejected: this.rejected,
      power: null,
      temp: this.temp,
      gpu: this.gpu,
    });
  }
}

module.exports = { PearlEngine };
