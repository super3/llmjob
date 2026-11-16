'use strict';

// Which k an RX 9070 XT mines on this PC: k 8192 or k 16384. Pure: PearlMiner
// feeds it the card's hashrate samples and the time, and opens the card's core
// again at the k it is told (main/pearlMiner, _feedKTrial).
//
// Why a trial and not a rule. k 16384 does the same work per MAC with fewer
// commitment-tree passes, and on one RX 9070 XT PC it read +0.65% over k 8192.
// On two others it was level, and on a fourth it lost 1.6% (benchmark.md,
// goal-rdna4p, 16 November). The same card model, so nothing the app can read
// tells those PCs apart in advance. A launch shape on RTX 30 is chosen the same
// way, by timing it on the PC (shared/shapePick).
//
// The card mines through four turns: k 8192, k 16384, k 8192, k 16384. The
// first is the core the card opened with; each later one opens the card again
// at that k, which takes a few seconds. The start of every turn is left out:
// the first has the core's warm-up in it, and the others the new core's. k
// 16384 is kept only when it is ahead on the means by MIN_GAIN and each of its
// turns beat each k 8192 turn. Anything less keeps k 8192, the default, so a PC
// where the two are level mines as it did before. A turn with too few samples
// is no answer, and the card stays at k 8192.
//
// The answer is remembered for a week a card, under the launch shape's key and
// freshness rule (shared/shapePick), so the trial runs once a week, not at
// every start. One that could not finish has no answer, and the next start
// tries again.

const { shapeKey, MAX_AGE_MS, stillFresh } = require('./shapePick');

const BASE_K = 8192;
const WIDE_K = 16384;
// One turn, and how much of its start is left out. Four turns are about 80 s,
// of which about 40 s mine at the k that turns out slower: at the -1.6% PC's
// loss that costs under a second of mining. The reopenings cost a few seconds
// more.
const TURN_MS = 20 * 1000;
const SETTLE_MS = 6 * 1000;
// The k each turn mines.
const TURNS = [BASE_K, WIDE_K, BASE_K, WIDE_K];
// What k 16384 has to be ahead by: the memory clock trial's bar
// (shared/memClockTrial). hashrate.js twins on one RX 9070 XT differ by 0.1 to
// 0.25%.
const MIN_GAIN = 0.003;
// The fewest samples a turn needs. A core samples about twice a second, so a
// full turn has about 28 after its settling time.
const MIN_SAMPLES = 10;

// The card (`arch`, the gcnArchName the AMD core reports) mines k 8192 by
// default and could mine k 16384: an RX 9070 XT, RX 9070 or Radeon AI PRO
// R9700 (gfx1201), whose core chose k 8192 on its own (`k`). A core opened
// with any other k (too little VRAM, PEARL_AMD_K, the pool turning k 8192 down)
// has nothing to time. The RX 9060 XT (gfx1200) lost 3.5% at k 16384 and is
// never tried.
function eligible(arch, k) {
  return arch === 'gfx1201' && k === BASE_K;
}

// A new trial, in its first turn from `now` (ms).
function createTrial(now) {
  return { turn: 0, since: now, sums: TURNS.map(() => 0), counts: TURNS.map(() => 0) };
}

const mean = (trial, turn) => trial.sums[turn] / trial.counts[turn];

// What the four turns say: { k, baseTh, wideTh, gain, reason }. `k` is the k to
// keep, the TH/s are each k's mean, and `gain` is k 16384's over k 8192 as a
// fraction. The rates and gain are null when a turn had too few samples;
// `reason` says why k 8192 stays.
function verdict(trial) {
  if (trial.counts.some((n) => n < MIN_SAMPLES)) {
    return { k: BASE_K, baseTh: null, wideTh: null, gain: null, reason: 'too few hashrate samples to compare' };
  }
  const of = (k) => TURNS.map((tk, t) => (tk === k ? mean(trial, t) : null)).filter((v) => v != null);
  const base = of(BASE_K);
  const wide = of(WIDE_K);
  const baseTh = (base[0] + base[1]) / 2;
  const wideTh = (wide[0] + wide[1]) / 2;
  const gain = wideTh / baseTh - 1;
  const out = { k: BASE_K, baseTh, wideTh, gain, reason: null };
  if (gain < MIN_GAIN) out.reason = 'k ' + WIDE_K + ' was not ahead by ' + (MIN_GAIN * 100).toFixed(1) + '%';
  else if (Math.min(...wide) <= Math.max(...base)) out.reason = 'k ' + WIDE_K + ' was not ahead in every turn';
  else out.k = WIDE_K;
  return out;
}

// One hashrate sample `th` (TH/s) at `now` (ms). Returns null while a turn
// runs, { k } when the next turn starts (open the card at that k), or
// { done: true, ...verdict } after the last turn. The last turn mines k 16384,
// so a kept k 16384 is already in place and a refused one is for the caller to
// undo. The sample that ends a turn is not counted: it straddles the change.
function feed(trial, th, now) {
  if (now - trial.since >= TURN_MS) {
    trial.turn += 1;
    trial.since = now;
    if (trial.turn >= TURNS.length) return { done: true, ...verdict(trial) };
    return { k: TURNS[trial.turn] };
  }
  if (now - trial.since >= SETTLE_MS && th > 0) {
    trial.sums[trial.turn] += th;
    trial.counts[trial.turn] += 1;
  }
  return null;
}

// Set to 0 in either shell's environment, no card is timed and every card
// mines the core's own choice. Set to 1 in RETRY_ENV, this start forgets what
// is remembered and times every such card again.
const OFF_ENV = 'LLMJOB_K_TRIAL';
const RETRY_ENV = 'LLMJOB_K_TRIAL_RETRY';
const REMEMBER_MS = MAX_AGE_MS;

function trialsOff(env) {
  return !!env && env[OFF_ENV] === '0';
}

// The key an answer is kept under: the launch shape pick's (card index and name).
function cardKey(index, gpus) {
  return shapeKey({ deviceIndex: index }, gpus);
}

// What to keep of a finished trial, or null when it compared nothing (no
// answer: the next start tries again). The rates to a hundredth of a TH/s, for
// a later start's log line.
function remember(v, now) {
  if (v.gain == null) return null;
  const rate = (th) => Math.round(th * 100) / 100;
  return { k: v.k, at: now, baseTh: rate(v.baseTh), wideTh: rate(v.wideTh) };
}

// A kept answer that still stands at `now`, or null: none, one that is not an
// answer, one over a week old or dated after `now`, or `retry` asked.
function recall(saved, now, retry) {
  if (retry || !saved || (saved.k !== BASE_K && saved.k !== WIDE_K)) return null;
  return stillFresh(saved.at, now, REMEMBER_MS) ? saved : null;
}

function retryAsked(env) {
  return !!env && env[RETRY_ENV] === '1';
}

// A gain as the log says it: +0.65%.
function pct(gain) {
  return (gain >= 0 ? '+' : '') + (gain * 100).toFixed(2) + '%';
}

module.exports = {
  BASE_K, WIDE_K, TURN_MS, SETTLE_MS, TURNS, MIN_GAIN, MIN_SAMPLES, OFF_ENV, RETRY_ENV, REMEMBER_MS,
  eligible, createTrial, feed, verdict, trialsOff, cardKey, remember, recall, retryAsked, pct,
};
