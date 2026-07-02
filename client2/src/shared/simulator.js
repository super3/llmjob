'use strict';

// Deterministic-friendly demo feed. When no real miner is producing output
// (e.g. the in-app live preview, or a machine without the engine yet), the
// simulator advances a believable hashrate/shares stream — the same model the
// design mock uses. `rng` is injectable so behaviour is fully testable.

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

class Simulator {
  constructor(opts = {}) {
    this.rng = opts.rng || Math.random;
    this.total = opts.total != null ? opts.total : 354.1;
    this.accepted = opts.accepted != null ? opts.accepted : 14820;
    this.rejected = opts.rejected != null ? opts.rejected : 3;
    this.load = opts.load != null ? opts.load : 92;
    this.power = opts.power != null ? opts.power : 318;
    this.uptimeSec = opts.uptimeSec != null ? opts.uptimeSec : 0;
    this.maxPoints = opts.maxPoints != null ? opts.maxPoints : 60;

    const seedLength = opts.seedLength != null ? opts.seedLength : this.maxPoints;
    this.points = [];
    for (let i = 0; i < seedLength; i++) {
      this.points.push(346 + Math.sin(i / 6) * 7 + this.rng() * 5);
    }
  }

  step() {
    const r = this.rng;
    this.total = clamp(this.total + (r() - 0.5) * 8, 334, 368);
    this.points = this.points.concat(this.total).slice(-this.maxPoints);
    this.load = clamp(this.load + (r() - 0.5) * 4, 86, 97);
    this.power = Math.round(clamp(this.power + (r() - 0.5) * 8, 300, 342));
    this.accepted += r() < 0.82 ? 1 : 0;
    this.rejected += r() < 0.012 ? 1 : 0;
    this.uptimeSec += 1;
    return this.snapshot();
  }

  snapshot() {
    return {
      total: this.total,
      points: this.points.slice(),
      accepted: this.accepted,
      rejected: this.rejected,
      load: this.load,
      power: this.power,
      uptimeSec: this.uptimeSec,
    };
  }
}

module.exports = { clamp, Simulator };
