'use strict';

// Accumulates *real* telemetry parsed from the alpha-miner engine (see
// shared/parser) into the snapshot shape the renderer's stat panel expects.
// Pure and clock-injected (uptime is derived from an injected nowMs) so it is
// fully unit-tested; main.js feeds it live miner events and Date.now().
//
// There is deliberately no fabricated/demo data here: before the engine
// reports anything the snapshot reads 0 TH/s, 0 shares, $0/day.

const MAX_POINTS = 60;

// A fresh accumulator; startMs anchors the uptime clock.
function initStats(startMs) {
  return { startMs: startMs || 0, hashrate: 0, accepted: 0, rejected: 0, load: 0, power: 0, gpu: null, points: [] };
}

// Fold one parsed miner event into the accumulator (mutates and returns it).
// The engine's periodic `status` event carries the live hashrate and the
// *cumulative* share counters, so counts are SET (not incremented). `connected`
// carries the GPU name. Anything else is ignored.
function applyEvent(stats, evt) {
  if (!stats || !evt) return stats;
  if (evt.type === 'status') {
    if (evt.hashrate != null) {
      stats.hashrate = evt.hashrate;
      stats.points.push(evt.hashrate);
      if (stats.points.length > MAX_POINTS) stats.points.shift();
    }
    if (evt.accepted != null) stats.accepted = evt.accepted;
    if (evt.rejected != null) stats.rejected = evt.rejected;
    if (evt.power != null) stats.power = evt.power;
    if (evt.gpu) stats.gpu = evt.gpu;
  } else if (evt.type === 'connected') {
    if (evt.gpu) stats.gpu = evt.gpu;
  }
  return stats;
}

// Project the accumulator into the renderer snapshot shape at time nowMs.
function snapshot(stats, nowMs) {
  return {
    total: stats.hashrate,
    points: stats.points.slice(),
    accepted: stats.accepted,
    rejected: stats.rejected,
    load: stats.load,
    power: stats.power,
    gpu: stats.gpu,
    uptimeSec: Math.max(0, Math.floor(((nowMs || 0) - stats.startMs) / 1000)),
  };
}

module.exports = { MAX_POINTS, initStats, applyEvent, snapshot };
