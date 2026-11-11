'use strict';

// The trial behind the RTX 5090's default and `--mine-mem-clock auto`
// (shared/memClock): does this card mine faster with its memory clock locked?
// Pure: PearlMiner feeds it the card's hashrate samples and the time, and takes
// the lock or releases it when told to (main/pearlMiner, _feedMemClockTrial).
//
// Why a trial and not a table. A lower memory clock frees watts for the SM
// clock on a power-capped card, but the fold still reads DRAM, and a card only
// has a few memory speeds. On an RTX 4090 a lock at 5001 MHz measured +0.8% at
// 450 W and up to +1.7% under a lower power limit, while the next speed down
// (810 MHz) lost 11% and 405 MHz lost 75% (tuning/benchmark.md, "rig01,
// 10 November"). The rented hosts the kernels are tested on can't set clocks,
// so one RTX 4090 and one RTX 5090 are all that has been measured, and the
// 5090's two runs did not agree (+8.0% in one, level in the other). So every
// card, a 5090 too, has to show on its own rig that the lock pays before it
// keeps it.
//
// The card mines through four turns: driver's clock, locked, driver's clock,
// locked. Two turns of each, alternating, so a card that is still warming up or
// a pool that changes difficulty does not decide it. The start of every turn is
// left out: the first turn has the core's own warm-up in it, and the others the
// clock change. The lock is kept only when it is ahead on the means by MIN_GAIN
// and each locked turn beat each unlocked one. A card that sent too few samples
// in any turn keeps the driver's clock: no answer is not a yes.
//
// The answer is remembered for a week a card (remember, recall), as the RTX 30
// launch shape is (shared/shapePick): a trial on every start cost a card where
// the lock pays half of its first five minutes each time, and a rig whose
// mining runs are short (demand mode, a service that restarts) never got to
// keep the lock at all. Only a real comparison is remembered. A trial that
// could not run or finish has no answer, and the next start tries again.

const { shapeKey, MAX_AGE_MS } = require('./shapePick');

// One turn, and how much of its start is left out.
const TURN_MS = 75 * 1000;
const SETTLE_MS = 15 * 1000;
// Whether each turn mines with the lock held.
const TURNS = [false, true, false, true];
// What the lock has to be ahead by. The startup trial of RTX 30's launch shapes
// uses the same bar (shared/shapePick): twin runs on one card differ by less.
const MIN_GAIN = 0.003;
// The fewest samples a turn needs. A core samples about twice a second, so a
// full turn has over a hundred; a turn with under twenty spent most of its
// time not mining.
const MIN_SAMPLES = 20;

// A new trial, in its first turn (the driver's clock) from `now` (ms).
function createTrial(now) {
  return { turn: 0, since: now, sums: TURNS.map(() => 0), counts: TURNS.map(() => 0) };
}

const mean = (trial, turn) => trial.sums[turn] / trial.counts[turn];

// What the four turns say. { keep, offTh, onTh, gain, reason }: the mean TH/s
// without and with the lock, the gain as a fraction, and why not when keep is
// false. offTh, onTh and gain are null when a turn had too few samples.
function verdict(trial) {
  if (trial.counts.some((n) => n < MIN_SAMPLES)) {
    return { keep: false, offTh: null, onTh: null, gain: null, reason: 'too few hashrate samples to compare' };
  }
  const off = TURNS.map((locked, t) => (locked ? null : mean(trial, t))).filter((v) => v != null);
  const on = TURNS.map((locked, t) => (locked ? mean(trial, t) : null)).filter((v) => v != null);
  const offTh = (off[0] + off[1]) / 2;
  const onTh = (on[0] + on[1]) / 2;
  const gain = onTh / offTh - 1;
  if (gain < MIN_GAIN) {
    return { keep: false, offTh, onTh, gain, reason: 'the lock was not ahead by ' + (MIN_GAIN * 100).toFixed(1) + '%' };
  }
  if (Math.min(on[0], on[1]) <= Math.max(off[0], off[1])) {
    return { keep: false, offTh, onTh, gain, reason: 'the lock was not ahead in every turn' };
  }
  return { keep: true, offTh, onTh, gain, reason: null };
}

// One hashrate sample `th` (TH/s) at `now` (ms). Returns null while a turn
// runs, { lock } when the next turn starts (true: take the lock, false: give it
// back), or { done: true, ...verdict } after the last turn. The last turn is a
// locked one, so a kept lock is already in place and a refused one is still to
// be released by the caller. The sample that ends a turn is not counted: it
// straddles the change.
function feed(trial, th, now) {
  if (now - trial.since >= TURN_MS) {
    trial.turn += 1;
    trial.since = now;
    if (trial.turn >= TURNS.length) return { done: true, ...verdict(trial) };
    return { lock: TURNS[trial.turn] };
  }
  if (now - trial.since >= SETTLE_MS && th > 0) {
    trial.sums[trial.turn] += th;
    trial.counts[trial.turn] += 1;
  }
  return null;
}

// How long an answer stands before the card is tried again: the week a launch
// shape pick stands, and for its reason. Drivers, power limits and cooling
// change, and each can move what the lock is worth.
const REMEMBER_MS = MAX_AGE_MS;
// Set to 1 in the environment of either shell, this start forgets what is
// remembered and tries every planned card again. The new answer replaces the
// old one. (Deleting the file the answers are kept in does the same.)
const RETRY_ENV = 'LLMJOB_MINE_MEM_CLOCK_RETRY';

// The key an answer is kept under: the card's index and its name in the mining
// list, which is the launch shape pick's key (shapeKey). A different model in
// the same slot is therefore tried afresh. Another card of the same model in
// that slot is not told apart: nothing the app reads names one card of a model
// from another.
function cardKey(index, gpus) {
  return shapeKey({ deviceIndex: index }, gpus);
}

// What to keep of a finished trial (`verdict` is feed()'s last answer), or null
// when it compared nothing: too few samples is no answer, so nothing is kept
// and the next start tries again. The two rates are kept for the log line of
// a later start, to a hundredth of a TH/s.
function remember(verdict, mhz, now) {
  if (verdict.gain == null) return null;
  const rate = (th) => Math.round(th * 100) / 100;
  return { keep: verdict.keep, mhz, at: now, onTh: rate(verdict.onTh), offTh: rate(verdict.offTh) };
}

// A kept answer that still stands at `now` for a lock at `mhz`, or null: none
// kept, one for another clock, one that is not an answer, one over a week old,
// or one dated after `now` (the PC's clock has been set back, so its age is
// not known). `retry` (RETRY_ENV is set) forgets it whatever it says.
function recall(saved, mhz, now, retry) {
  if (retry || !saved || typeof saved.keep !== 'boolean' || saved.mhz !== mhz) return null;
  const at = Number(saved.at);
  return at > now - REMEMBER_MS && at <= now ? saved : null;
}

// Whether RETRY_ENV asks for fresh trials.
function retryAsked(env) {
  return !!env && env[RETRY_ENV] === '1';
}

// A time as the log says it: the UTC date, 2026-11-18.
function day(ms) {
  return new Date(ms).toISOString().slice(0, 10);
}

module.exports = {
  TURN_MS, SETTLE_MS, TURNS, MIN_GAIN, MIN_SAMPLES, REMEMBER_MS, RETRY_ENV,
  createTrial, feed, cardKey, remember, recall, retryAsked, day,
};
