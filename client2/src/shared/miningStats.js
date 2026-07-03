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
  return { startMs: startMs || 0, hashrate: 0, accepted: 0, rejected: 0, load: 0, power: 0, points: [] };
}

// Fold one parsed miner event into the accumulator (mutates and returns it).
// Recognised events: { type:'hashrate', hashrate, load?, power? } and
// { type:'share', status:'accepted'|'rejected' }. Anything else is ignored.
function applyEvent(stats, evt) {
  if (!stats || !evt) return stats;
  if (evt.type === 'hashrate') {
    stats.hashrate = Number(evt.hashrate) || 0;
    if (evt.load != null) stats.load = evt.load;
    if (evt.power != null) stats.power = evt.power;
    stats.points.push(stats.hashrate);
    if (stats.points.length > MAX_POINTS) stats.points.shift();
  } else if (evt.type === 'share') {
    if (evt.status === 'accepted') stats.accepted += 1;
    else if (evt.status === 'rejected') stats.rejected += 1;
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
    uptimeSec: Math.max(0, Math.floor(((nowMs || 0) - stats.startMs) / 1000)),
  };
}

module.exports = { MAX_POINTS, initStats, applyEvent, snapshot };
