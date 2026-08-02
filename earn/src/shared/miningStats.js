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

// A card that hasn't been mentioned by the engine for this long is treated as
// gone, and stops being counted or reported.
//
// Buckets are never otherwise removed, and a bucket holds the card's LAST known
// hashrate — so a card that dies mid-session (riser, driver, thermal cutout)
// left its final reading frozen in place while the surviving cards kept the
// process alive. The once-a-minute network report re-posted that dead card every
// cycle with a fresh timestamp, so the server's 5-minute offline sweep never
// aged the row out: the phantom card inflated the rig's hashrate, the payout
// address's total, and the network-wide total indefinitely. The GUI's own
// headline number was overstated the same way.
//
// Three minutes is three report cycles — far longer than any legitimate gap
// between a live card's status lines, and comfortably inside the server's
// 5-minute offline window, so once we stop posting a dead card its board row
// ages off promptly instead of lingering.
const CARD_STALE_MS = 3 * 60 * 1000;

// A fresh accumulator; startMs anchors the uptime clock. `gpus` is a map of
// card index → per-card bucket, populated lazily as the engine reports.
function initStats(startMs) {
  return { startMs: startMs || 0, gpus: {}, load: 0, points: [] };
}

// Get-or-create the bucket for a card index (a line with no index — e.g. a
// single-GPU engine that omits it — folds into card 0).
function bucketFor(stats, index) {
  const idx = Number.isFinite(index) ? index : 0;
  if (!stats.gpus[idx]) {
    stats.gpus[idx] = { index: idx, hashrate: 0, accepted: 0, rejected: 0, power: 0, gpu: null, at: null };
  }
  return stats.gpus[idx];
}

// Is this card still reporting as of nowMs? A bucket with no timestamp (nothing
// has stamped it — an older caller that folds events without a clock) is treated
// as live: silence we cannot date is not evidence the card is gone.
function isLive(bucket, nowMs) {
  if (!Number.isFinite(bucket.at) || !Number.isFinite(nowMs)) return true;
  return nowMs - bucket.at <= CARD_STALE_MS;
}

// Every still-reporting card, ordered by index.
function liveCards(stats, nowMs) {
  return Object.keys(stats.gpus)
    .map((k) => stats.gpus[k])
    .filter((g) => isLive(g, nowMs))
    .sort((a, b) => a.index - b.index);
}

// Sum a numeric field across every card still reporting at nowMs.
function sumField(stats, field, nowMs) {
  let total = 0;
  for (const g of liveCards(stats, nowMs)) total += Number(g[field]) || 0;
  return total;
}

// Fold one parsed miner event into the accumulator (mutates and returns it).
// The engine's periodic `status` event carries a card's live hashrate and its
// *cumulative* share counters, so a card's counts are SET (not incremented).
// `connected` carries the GPU name. Anything else is ignored.
// `nowMs` stamps the card as heard-from, which is what lets a card that goes
// silent be dropped later (see CARD_STALE_MS). Optional: omit it and no card
// ever ages out, which is the pre-existing behaviour.
function applyEvent(stats, evt, nowMs) {
  if (!stats || !evt) return stats;
  if (evt.type === 'status') {
    const g = bucketFor(stats, evt.gpuIndex);
    if (evt.hashrate != null) g.hashrate = evt.hashrate;
    if (evt.accepted != null) g.accepted = evt.accepted;
    if (evt.rejected != null) g.rejected = evt.rejected;
    if (evt.power != null) g.power = evt.power;
    if (evt.gpu) g.gpu = evt.gpu;
    if (Number.isFinite(nowMs)) g.at = nowMs;
    // The sparkline charts the rig's *total* hashrate, so push the sum across
    // cards after folding this update — not the single card's value. Summed over
    // LIVE cards only, so a dead card stops propping the line up.
    stats.points.push(sumField(stats, 'hashrate', nowMs));
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
  // Live cards only: a card the engine has stopped mentioning is gone, and must
  // not keep contributing its last-known hashrate to the rig total or keep being
  // posted to the board as an online GPU.
  const cards = liveCards(stats, nowMs);
  const named = cards.find((g) => g.gpu);
  return {
    total: cards.reduce((a, g) => a + (Number(g.hashrate) || 0), 0),
    points: stats.points.slice(),
    accepted: cards.reduce((a, g) => a + (Number(g.accepted) || 0), 0),
    rejected: cards.reduce((a, g) => a + (Number(g.rejected) || 0), 0),
    load: stats.load,
    power: cards.reduce((a, g) => a + (Number(g.power) || 0), 0),
    gpu: named ? named.gpu : null,   // representative name (lowest-index card)
    gpus: cards.map((g) => ({
      index: g.index,
      gpu: g.gpu,
      hashrate: Number(g.hashrate) || 0,
      accepted: Number(g.accepted) || 0,
      rejected: Number(g.rejected) || 0,
      power: Number(g.power) || 0,
    })),
    uptimeSec: Math.max(0, Math.floor(((nowMs || 0) - stats.startMs) / 1000)),
  };
}

module.exports = { MAX_POINTS, initStats, applyEvent, snapshot };
