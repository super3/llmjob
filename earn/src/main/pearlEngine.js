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
    this.tempTimer = null;
    this.miner = null;
    // One entry per mining card, keyed by card index:
    //   { name, hashrate, accepted, rejected, temp }
    // The UI, the stats accumulator and the network board are all per card
    // (miningStats buckets on the index the event carries), so this is the shape
    // they want. A rig mining on one card has one entry, which is what every
    // consumer saw before.
    this.cards = new Map();
    this.gpu = null;
    this.endpoint = null;
  }

  // Get-or-create a card's counters. Cumulative for the session, because that is
  // what the UI's counters mean.
  _card(index) {
    const i = Number.isInteger(index) ? index : 0;
    if (!this.cards.has(i)) {
      this.cards.set(i, { name: null, hashrate: 0, accepted: 0, rejected: 0, temp: null });
    }
    return this.cards.get(i);
  }

  isRunning() {
    return !!(this.miner && this.miner.isRunning());
  }

  // One card's current numbers, or null when that card isn't mining. Read-only:
  // unlike _card it never creates a bucket, so asking about a card the rig does
  // not mine on cannot invent one.
  card(index) {
    return this.cards.get(Number.isInteger(index) ? index : 0) || null;
  }

  start(settings = {}) {
    this.cards.clear();
    this.gpu = settings.gpu || null;
    this.endpoint = settings.endpoint || null;

    const m = new PearlMiner({ connect: this.connect, createCore: this.createCore });
    this.miner = m;

    m.on('log', (l) => this.emit('log', l));
    m.on('error', (err) => this.emit('error', err));
    m.on('stopped', () => this.emit('stopped', 0));

    // The pool accepted the wallet and we have work: the cards are mining. This
    // is the moment alpha-miner printed its connection banner, and the UI wants
    // the endpoint and card name from it. One per card, because that is how the
    // board learns which GPU is which.
    m.on('job', () => {
      if (this._announced) return;
      this._announced = true;
      for (const [index, card] of this.cards) {
        this.emit('event', {
          type: 'connected',
          gpuIndex: index,
          endpoint: this.endpoint,
          gpu: card.name || this.gpu,
        });
      }
    });

    // Forwarded as a parsed event because that is the shape main.js's DNS hint
    // reads, and a name that does not resolve is worth saying plainly.
    m.on('connect-failed', (e) => this.emit('event', Object.assign({ type: 'connect-failed' }, e)));

    // Credited to the card that found it — the submit carries its index back.
    m.on('share', (e) => { this._card(e && e.index).accepted++; this._status(e && e.index); });
    m.on('rejected', (e) => { this._card(e && e.index).rejected++; this._status(e && e.index); });

    // Each core's own throughput tick drives the sparkline. Both sides count
    // multiply-accumulates per second in TH/s, which is the unit the UI's
    // hashrate field already carries.
    m.on('hashrate', (th, device) => {
      const index = device ? device.index : 0;
      this._card(index).hashrate = th;
      this._status(index);
    });

    this._announced = false;
    // Propagate it. PearlMiner.start() returns false when the core will not
    // construct -- no VRAM for the rank-128 profile, no pearl_core.node -- and it
    // emits 'error' but NOT 'stopped', because nothing ever started. Discarding
    // this return dropped the only signal that the engine is dead rather than
    // merely quiet.
    const ok = m.start(Object.assign({}, settings, { profile: this.profile }));

    // The cores have now chosen their cards, so stop guessing at them. The GPU
    // name used to be detected separately (nvidia-smi's first card) and the index
    // was hardcoded to 0 -- both true only while CUDA and nvidia-smi happen to
    // number the cards the same way, which on a multi-GPU rig they need not
    // (issue #226). What the cores report is what is mining.
    for (const d of m.devices()) this._card(d.index).name = d.name;
    // Nothing named a card: an older core that won't say. Keep the one bucket
    // and the detected name, which is what this did before.
    if (!this.cards.size) this._card(0).name = this.gpu;

    // Started after m.start() so the first sample already knows the cards: it
    // reads a temperature per mining card, and before the cores exist there are
    // none to read.
    this._startTemps();
    return ok;
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
    const clear = () => { for (const card of this.cards.values()) card.temp = null; };
    const sample = () => {
      let p;
      // A reader that throws synchronously must not take the miner down with
      // it; a temperature is the least important thing on the screen.
      try {
        p = Promise.resolve(this.readTemps());
      } catch (e) {
        clear();
        return;
      }
      p.then((temps) => {
        // One reading per mining card. nvidia-smi answers for every card on the
        // rig; each of ours takes its own.
        for (const [index, card] of this.cards) {
          const t = temps ? Number(temps[index]) : NaN;
          card.temp = Number.isFinite(t) && t > 0 ? t : null;
        }
      }).catch(clear);
    };
    sample();
    this.tempTimer = setInterval(sample, this.tempPollMs);
    if (this.tempTimer.unref) this.tempTimer.unref();
  }

  _stopTemps() {
    if (this.tempTimer) clearInterval(this.tempTimer);
    this.tempTimer = null;
    for (const card of this.cards.values()) card.temp = null;
  }

  // One status event for one card, carrying that card's own numbers. The index
  // is the core's own, which is nvidia-smi's too now that the shells pin
  // CUDA_DEVICE_ORDER (see shared/gpu.alignCudaDeviceOrder). It used to be
  // hardcoded to 0, so on a rig where the core landed elsewhere the temperature
  // shown belonged to a card that wasn't mining and the board row was filed
  // under the wrong GPU.
  _status(index) {
    const i = Number.isInteger(index) ? index : 0;
    const card = this._card(i);
    this.emit('event', {
      type: 'status',
      gpuIndex: i,
      hashrate: card.hashrate,
      accepted: card.accepted,
      rejected: card.rejected,
      power: null,
      temp: card.temp,
      gpu: card.name || this.gpu,
    });
  }
}

module.exports = { PearlEngine };
