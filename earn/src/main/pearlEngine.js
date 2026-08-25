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
// Deliberately NOT reported: power and temperature. alpha-miner reads them from
// NVML; we do not, and inventing a number would be worse than the blank the UI
// already renders for a card that does not report them.
class PearlEngine extends EventEmitter {
  constructor({ connect, createCore, profile } = {}) {
    super();
    this.connect = connect;
    this.createCore = createCore;
    this.profile = profile || PROFILE;
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

    m.on('share', () => { this.accepted++; this._status(); });
    m.on('rejected', () => { this.rejected++; this._status(); });

    // The core's own throughput tick drives the sparkline. Both sides count
    // multiply-accumulates per second in TH/s, which is the unit the UI's
    // hashrate field already carries.
    m.on('hashrate', (th) => { this.hashrate = th; this._status(); });

    this._announced = false;
    m.start(Object.assign({}, settings, { profile: this.profile }));
  }

  stop() {
    if (this.miner) this.miner.stop();
  }

  _status() {
    this.emit('event', {
      type: 'status',
      gpuIndex: 0,
      hashrate: this.hashrate,
      accepted: this.accepted,
      rejected: this.rejected,
      power: null,
      temp: null,
      gpu: this.gpu,
    });
  }
}

module.exports = { PearlEngine };
