'use strict';

// Accumulates the miner's own status events (see main/pearlEngine) into the
// snapshot shape the renderer's stat panel expects.
// Pure and clock-injected (uptime is derived from an injected nowMs) so it is
// fully unit-tested; main.js feeds it live miner events and Date.now().
//
// The engine tags every status event with the card it came from (`gpuIndex`),
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
  // is the question it answers.
  return { startMs: startMs || 0, gpus: {}, points: [], lastShareMs: null };
}

// Get-or-create the bucket for a card index (a line with no index — e.g. a
// single-GPU engine that omits it — folds into card 0).
function bucketFor(stats, index) {
  const idx = Number.isFinite(index) ? index : 0;
  if (!stats.gpus[idx]) {
    stats.gpus[idx] = {
      index: idx, hashrate: 0, accepted: 0, rejected: 0, temp: 0, gpu: null,
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
// From a core that reports it, the event also carries the card's memory.
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
    if (evt.temp != null) g.temp = evt.temp;
    if (evt.gpu) g.gpu = evt.gpu;
    // The card's memory, from a core that reports it (the Intel core; see
    // PearlEngine._memory). The two always come as a pair, and a card whose
    // events never carry them has neither on its bucket.
    if (evt.memTotalMb != null) {
      g.memTotalMb = evt.memTotalMb;
      g.memUsedMb = evt.memUsedMb;
    }
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
    // The hottest card, not a sum or an average: this exists to answer "is
    // anything running too hot", and a mean would hide one cooking card behind
    // several cool ones. 0 when the engine has not reported a temperature yet.
    temp: cards.reduce((a, g) => Math.max(a, Number(g.temp) || 0), 0),
    gpu: named ? named.gpu : null,   // representative name (lowest-index card)
    // A card whose core reports its memory also has memUsedMb and memTotalMb
    // (MB), which the board falls back on when nothing else knows that card's
    // memory (see minerReport). The other cards' entries are as they always
    // were, so what is built from them (the HiveOS stats file, for one) does
    // not change for a rig with no such card.
    gpus: cards.map((g) => Object.assign({
      index: g.index,
      gpu: g.gpu,
      hashrate: Number(g.hashrate) || 0,
      accepted: Number(g.accepted) || 0,
      rejected: Number(g.rejected) || 0,
      temp: Number(g.temp) || 0,
    }, g.memTotalMb == null ? null : { memUsedMb: g.memUsedMb, memTotalMb: g.memTotalMb })),
    uptimeSec: Math.max(0, Math.floor(((nowMs || 0) - stats.startMs) / 1000)),
    // null, not 0: "no share yet" and "a share at the epoch" are different, and a
    // renderer showing "56 years ago" would be worse than showing nothing.
    lastShareMs: stats.lastShareMs == null ? null : stats.lastShareMs,
  };
}

// The rig's hashrate averaged over time between two log lines.
//
// Each core reports a rate about every half second, worked out over the batches that
// finished in that window. A log line once a second used to print only the latest of
// those, so it printed every other window. When the windows alternate (two batches
// finishing together put one batch's work in the next window: 3 and 5 batches around
// a true 4 on an RTX 4060), every printed value came from the same phase, and the
// printed rate was off by up to a quarter for as long as the phase held. Averaging
// the latest rate over the time it was showing uses every window.
//
// With windowMs the read is a moving average instead: the same time-weighted mean, over
// the last windowMs rather than since the last read. On RTX 20 cards a second holds 8 to
// 10 launches, so a line a second stepped by about 10% (65, 73 and 81 TH/s on an RTX 2080
// that averaged 72.7), and a reader that takes the latest line every 15 s, as our
// benchmark does, was off by up to 2% over 5 minutes. A moving average counts the same
// hashes over the same time, so its long-run mean is unchanged.
function createRateMeter(windowMs = 0) {
  return { t: null, v: 0, area: 0, start: null, windowMs, spans: [] };
}

// The wall clock stepped back (an NTP step, a VM resume): the spans kept so far are stamped
// in the old time and would never age out, so start over from nowMs with the value showing.
function meterRewind(meter, nowMs) {
  if (meter.t != null && nowMs < meter.t) {
    meter.spans = [];
    meter.area = 0;
    meter.start = nowMs;
    meter.t = nowMs;
  }
}

// Close the value now showing at nowMs: add its area, and with a window keep its span.
function meterClose(meter, nowMs) {
  meter.area += meter.v * (nowMs - meter.t);
  if (meter.windowMs > 0 && nowMs > meter.t) meter.spans.push([meter.t, nowMs, meter.v]);
}

// The rig total changed (or was re-reported) at nowMs.
function meterSample(meter, value, nowMs) {
  meterRewind(meter, nowMs);
  if (meter.t == null) meter.start = nowMs;
  else meterClose(meter, nowMs);
  meter.t = nowMs;
  meter.v = value;
  return meter;
}

// The time-weighted mean since the last read (or the first sample), and start the next
// interval. With no time elapsed, the value as it stands. With a window: the mean over
// the last windowMs, or since the first sample when that is more recent.
function meterRead(meter, nowMs) {
  if (meter.t == null) return 0;
  meterRewind(meter, nowMs);
  meterClose(meter, nowMs);
  meter.t = nowMs;
  if (meter.windowMs > 0) {
    const from = Math.max(nowMs - meter.windowMs, meter.start);
    while (meter.spans.length && meter.spans[0][1] <= from) meter.spans.shift();
    let area = 0;
    for (const [t0, t1, v] of meter.spans) area += v * (t1 - Math.max(t0, from));
    return nowMs > from ? area / (nowMs - from) : meter.v;
  }
  const span = nowMs - meter.start;
  const mean = span > 0 ? meter.area / span : meter.v;
  meter.area = 0;
  meter.start = nowMs;
  return mean;
}

// A snapshot as the pool sees it while the connection to the pool is down: no
// hashrate, because no share can reach it. The GPU may still be working on its
// last job, but the network board must not show a rig as mining that the pool
// shows as offline. Shares and everything else stay as they are.
function offPool(snap) {
  return Object.assign({}, snap, {
    total: 0,
    gpus: snap.gpus.map((g) => Object.assign({}, g, { hashrate: 0 })),
  });
}

module.exports = { MAX_POINTS, initStats, applyEvent, snapshot, offPool, createRateMeter, meterSample, meterRead };
