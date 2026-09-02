'use strict';

// Accumulates *real* telemetry parsed from the alpha-miner engine (see
// shared/parser) into the snapshot shape the renderer's stat panel expects.
// Pure and clock-injected (uptime is derived from an injected nowMs) so it is
// fully unit-tested; main.js feeds it live miner events and Date.now().
//
// The engine tags every status line with the card it came from (`gpu=<index>`),
// so state is kept *per card* — a multi-GPU rig folds each card's hashrate and
// share counts into its own bucket. The rig-level figures the renderer shows
// (total hashrate, total shares) are the sum across cards; the per-card buckets
// also let the network board report each GPU as its own row (see minerReport).
//
// There is deliberately no fabricated/demo data here: before the engine
// reports anything the snapshot reads 0 TH/s, 0 shares, $0/day.

const MAX_POINTS = 60;

// A fresh accumulator; startMs anchors the uptime clock. `gpus` is a map of
// card index → per-card bucket, populated lazily as the engine reports.
function initStats(startMs) {
  // lastShareMs is rig-level, not per-card: "when did this rig last get paid"
  // is the question it answers, and the CLI only ever fills bucket 0 anyway
  // (PearlEngine.gpuIndex() is hardcoded to it).
  return { startMs: startMs || 0, gpus: {}, load: 0, points: [], lastShareMs: null };
}

// Get-or-create the bucket for a card index (a line with no index — e.g. a
// single-GPU engine that omits it — folds into card 0).
function bucketFor(stats, index) {
  const idx = Number.isFinite(index) ? index : 0;
  if (!stats.gpus[idx]) {
    stats.gpus[idx] = {
      index: idx, hashrate: 0, accepted: 0, rejected: 0, power: 0, temp: 0, gpu: null,
      // `raw*` is the last value the ENGINE reported; `carry*` is everything it
      // reported in earlier engine sessions. `accepted`/`rejected` above stay the
      // effective totals, so snapshot() and every consumer are unchanged.
      rawAccepted: 0, rawRejected: 0, carryAccepted: 0, carryRejected: 0,
    };
  }
  return stats.gpus[idx];
}

// The engine's share counters are cumulative PER ENGINE SESSION, not for the life
// of the rig: PearlEngine.start() zeroes them. That was invisible while the miner
// ran once and ran forever, but demand mode stops and restarts it on every flip
// back to mining, so the counts fell to 0 several times an hour while uptime kept
// climbing -- the rig looked like it had stopped earning.
//
// A counter moving BACKWARDS is therefore a new session, not a correction: bank
// what the old session finished with and keep counting from there.
function foldCounter(g, rawKey, carryKey, raw) {
  if (raw < g[rawKey]) g[carryKey] += g[rawKey];
  g[rawKey] = raw;
  return g[carryKey] + raw;
}

// Sum a numeric field across every card bucket.
function sumField(stats, field) {
  let total = 0;
  for (const k of Object.keys(stats.gpus)) total += Number(stats.gpus[k][field]) || 0;
  return total;
}

// Fold one parsed miner event into the accumulator (mutates and returns it).
// The engine's periodic `status` event carries a card's live hashrate and its
// *cumulative* share counters, so a card's counts are SET (not incremented).
// `connected` carries the GPU name. Anything else is ignored.
// `nowMs` is optional and only stamps lastShareMs; omitting it keeps every other
// behaviour identical, which is why the two existing call sites need no change to
// stay correct.
function applyEvent(stats, evt, nowMs) {
  if (!stats || !evt) return stats;
  if (evt.type === 'status') {
    const g = bucketFor(stats, evt.gpuIndex);
    if (evt.hashrate != null) g.hashrate = evt.hashrate;
    if (evt.accepted != null) {
      const total = foldCounter(g, 'rawAccepted', 'carryAccepted', evt.accepted);
      // Only a RISE is a share landing. A restart that replays the same total, or
      // the backwards step itself, must not look like fresh work.
      if (total > g.accepted && Number.isFinite(nowMs)) stats.lastShareMs = nowMs;
      g.accepted = total;
    }
    if (evt.rejected != null) g.rejected = foldCounter(g, 'rawRejected', 'carryRejected', evt.rejected);
    if (evt.power != null) g.power = evt.power;
    if (evt.temp != null) g.temp = evt.temp;
    if (evt.gpu) g.gpu = evt.gpu;
    // The sparkline charts the rig's *total* hashrate, so push the sum across
    // cards after folding this update — not the single card's value.
    stats.points.push(sumField(stats, 'hashrate'));
    if (stats.points.length > MAX_POINTS) stats.points.shift();
  } else if (evt.type === 'connected') {
    if (evt.gpu) bucketFor(stats, evt.gpuIndex).gpu = evt.gpu;
  }
  return stats;
}

// Project the accumulator into the renderer snapshot shape at time nowMs. The
// top-level figures are rig-level aggregates (sum across cards); `gpus` is the
// per-card breakdown the network board reports (one row per GPU).
function snapshot(stats, nowMs) {
  const cards = Object.keys(stats.gpus)
    .map((k) => stats.gpus[k])
    .sort((a, b) => a.index - b.index);
  const named = cards.find((g) => g.gpu);
  return {
    total: cards.reduce((a, g) => a + (Number(g.hashrate) || 0), 0),
    points: stats.points.slice(),
    accepted: cards.reduce((a, g) => a + (Number(g.accepted) || 0), 0),
    rejected: cards.reduce((a, g) => a + (Number(g.rejected) || 0), 0),
    load: stats.load,
    power: cards.reduce((a, g) => a + (Number(g.power) || 0), 0),
    // The hottest card, not a sum or an average: this exists to answer "is
    // anything running too hot", and a mean would hide one cooking card behind
    // several cool ones. 0 when the engine has not reported a temperature yet.
    temp: cards.reduce((a, g) => Math.max(a, Number(g.temp) || 0), 0),
    gpu: named ? named.gpu : null,   // representative name (lowest-index card)
    gpus: cards.map((g) => ({
      index: g.index,
      gpu: g.gpu,
      hashrate: Number(g.hashrate) || 0,
      accepted: Number(g.accepted) || 0,
      rejected: Number(g.rejected) || 0,
      power: Number(g.power) || 0,
      temp: Number(g.temp) || 0,
    })),
    uptimeSec: Math.max(0, Math.floor(((nowMs || 0) - stats.startMs) / 1000)),
    // null, not 0: "no share yet" and "a share at the epoch" are different, and a
    // renderer showing "56 years ago" would be worse than showing nothing.
    lastShareMs: stats.lastShareMs == null ? null : stats.lastShareMs,
  };
}

module.exports = { MAX_POINTS, initStats, applyEvent, snapshot };
