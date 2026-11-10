'use strict';

// The trial behind `--mine-core-offset auto` (shared/coreOffset): how far can
// this card's core clock offset go and still compute right, and is it faster
// there? Pure: PearlMiner feeds it the card's hashrate samples, the results of
// the hits it checks and the time, and sets the offset it is told to
// (main/pearlMiner, _feedCoreOffsetTrial).
//
// Why a trial and not a table. On rig01's RTX 4090 at 450 W, +50, +100 and
// +150 MHz measured +1.1%, +2.5% and +4.3% (tuning/benchmark.md, "rig01,
// 10 November: core clock offset on the RTX 4090"). That is one card. An offset
// takes away the margin the card's maker left between its clock and the clock
// its silicon fails at, and that margin differs from one card to the next and
// with heat. Past it the card does not stop: it computes wrong. A wrong fold
// still finds hashes that meet the target, so every share would be sent and
// rejected. So each step has to prove itself on the card it runs on.
//
// The card mines through turns at offset 0, then each step up, then 0 again.
// The two turns at 0 sit either side of the steps, so a card still warming up
// does not read as a gain. The start of every turn is left out: the first has
// the core's own warm-up in it, and the others the clock change.
//
// A step's turn only starts once the step has passed a hit check. A share is
// far too rare to show in a turn that a step computes right, so the miner
// hands the card a job with a target easy enough to hit many times a second,
// and recomputes each hit the way a pool would (shared/miner/checkHit). The
// step passes at CHECK_HITS right hits. One wrong hit fails it, and so does a
// check that ends with too few hits to mean anything. A failed step ends the
// climb: no higher step is tried.
//
// At the end the card keeps the highest step that passed its check, beat the
// step before it by MIN_GAIN and beat offset 0 (the mean of its two turns) by
// MIN_GAIN. A turn with too few samples is no answer, and no answer never
// keeps the higher setting. 150 MHz is as far as it goes: nothing above that
// has been measured on any card.

// One turn, and how much of its start is left out. As shared/memClockTrial.
const TURN_MS = 75 * 1000;
const SETTLE_MS = 15 * 1000;
// The steps of the climb, in MHz, lowest first.
const STEPS_MHZ = [50, 100, 150];
// What a step has to be ahead by, of the step before it and of offset 0. The
// memory clock trial uses the same bar: twin runs on one card differ by less.
const MIN_GAIN = 0.003;
// The fewest samples a turn needs. A core samples about twice a second, so a
// full turn has over a hundred.
const MIN_SAMPLES = 20;
// A step's hit check: it passes at CHECK_HITS right hits. The check's job is
// set to hit about CHECK_HITS_PER_S times a second at the card's own rate
// (checkBound), so it takes three or four seconds on any card. After CHECK_MS
// it ends anyway, and passes only with CHECK_MIN_HITS or more. Recomputing a
// hit costs the miner's event loop 4.6 ms (shared/miner/checkHit), so a whole
// check is about half a second of it.
const CHECK_HITS = 100;
const CHECK_HITS_PER_S = 30;
const CHECK_MS = 10 * 1000;
const CHECK_MIN_HITS = 20;

// The steps a card can try: those its driver allows (`maxMhz` is the top of the
// range NVML reports for it).
function stepsFor(maxMhz) {
  return STEPS_MHZ.filter((mhz) => mhz <= maxMhz);
}

// The bound for a check's job: a jackpot hash at or under it is a hit. Set so a
// card mining `th` TH/s hits about CHECK_HITS_PER_S times a second, whatever
// its speed. One attempt costs `macsPerAttempt` multiply-accumulates
// (pearlhash.difficultyAdjustmentFactor), so the card makes th * 1e12 /
// macsPerAttempt attempts a second, and each is a hit with probability
// bound / 2^256. A card with no rate yet is taken as 1 TH/s.
function checkBound(th, macsPerAttempt) {
  const macsPerSec = BigInt(Math.round(Math.max(Number(th) || 0, 1) * 1e12));
  return ((1n << 256n) * BigInt(CHECK_HITS_PER_S * macsPerAttempt)) / macsPerSec;
}

// A new trial, in its first turn (offset 0) from `now` (ms). `steps` is the
// climb, from stepsFor.
function createTrial(now, steps = STEPS_MHZ) {
  const turns = [0, ...steps, 0];
  return {
    turns,                // the offset of each turn, in order
    turn: 0,              // the turn under way
    reached: 0,           // the last step turn whose hit check passed
    checking: false,      // whether the turn's step is still in its hit check
    hits: 0,              // right hits in the check under way
    failed: null,         // the step that failed its check, as { mhz, why }
    since: now,
    sums: turns.map(() => 0),
    counts: turns.map(() => 0),
  };
}

// The step under check passed: its turn starts now.
function pass(trial, now) {
  trial.checking = false;
  trial.reached = trial.turn;
  trial.since = now;
  return { offset: trial.turns[trial.turn], check: false };
}

// The step under check failed. The climb is over: straight to the last turn,
// at offset 0.
function fail(trial, why, now) {
  trial.failed = { mhz: trial.turns[trial.turn], why };
  trial.checking = false;
  trial.turn = trial.turns.length - 1;
  trial.since = now;
  return { offset: 0, check: false };
}

// What the turns say: { offset, baseTh, keptTh, gain, rates, failed, reason }.
// `offset` is the step to keep, 0 for none. baseTh is the mean TH/s of the two
// turns at offset 0, keptTh the kept step's, and gain the one over the other
// as a fraction; null where there is nothing to quote. `rates` lists the steps
// that had a turn, as { mhz, th }, th null for a turn with too few samples.
// `reason` says why nothing was kept.
function verdict(trial) {
  const last = trial.turns.length - 1;
  const rate = (t) => (trial.counts[t] >= MIN_SAMPLES ? trial.sums[t] / trial.counts[t] : null);
  const rates = [];
  for (let t = 1; t <= trial.reached; t += 1) rates.push({ mhz: trial.turns[t], th: rate(t) });
  const none = { offset: 0, baseTh: null, keptTh: null, gain: null, rates, failed: trial.failed };
  if (rate(0) == null || rate(last) == null) {
    return { ...none, reason: 'too few hashrate samples to compare' };
  }
  const baseTh = (rate(0) + rate(last)) / 2;
  let offset = 0;
  let keptTh = null;
  for (let t = 1; t <= trial.reached; t += 1) {
    const th = rate(t);
    const before = rate(t - 1);
    if (th != null && before != null && th >= before * (1 + MIN_GAIN) && th >= baseTh * (1 + MIN_GAIN)) {
      offset = trial.turns[t];
      keptTh = th;
    }
  }
  if (!offset) {
    const reason = trial.reached ? 'no step was faster by ' + (MIN_GAIN * 100).toFixed(1) + '%'
      : 'no step passed its hit check';
    return { ...none, baseTh, reason };
  }
  return { offset, baseTh, keptTh, gain: keptTh / baseTh - 1, rates, failed: trial.failed, reason: null };
}

// One hashrate sample `th` (TH/s) at `now` (ms). Returns null while nothing
// changes, or what the card should do next:
//   { offset, check: true }   a step starts: set this offset, then check its hits
//   { offset, check: false }  mine the pool's job at this offset (after a check
//                             that passed, the offset is the one already set)
//   { done: true, offset, check: false, ...verdict }  the trial is over: set the
//                             kept offset (the last turn was at 0)
// The sample that ends a turn is not counted: it straddles the change.
function feed(trial, th, now) {
  const waited = now - trial.since;
  if (trial.checking) {
    if (waited < CHECK_MS) return null;
    if (trial.hits >= CHECK_MIN_HITS) return pass(trial, now);
    return fail(trial, 'only ' + trial.hits + ' hits in ' + CHECK_MS / 1000 + ' s, too few to check', now);
  }
  if (waited >= TURN_MS) {
    const last = trial.turns.length - 1;
    if (trial.turn === last) return { done: true, check: false, ...verdict(trial) };
    trial.turn += 1;
    trial.since = now;
    if (trial.turn === last) return { offset: 0, check: false };
    trial.checking = true;
    trial.hits = 0;
    return { offset: trial.turns[trial.turn], check: true };
  }
  if (waited >= SETTLE_MS && th > 0) {
    trial.sums[trial.turn] += th;
    trial.counts[trial.turn] += 1;
  }
  return null;
}

// One hit of the check's job, recomputed: `why` is null when it was right, or
// what differed. Returns as feed does. A hit that arrives with no check under
// way changes nothing.
function hit(trial, why, now) {
  if (!trial.checking) return null;
  if (why) return fail(trial, why, now);
  trial.hits += 1;
  return trial.hits >= CHECK_HITS ? pass(trial, now) : null;
}

module.exports = {
  TURN_MS, SETTLE_MS, STEPS_MHZ, MIN_GAIN, MIN_SAMPLES, CHECK_HITS, CHECK_HITS_PER_S, CHECK_MS, CHECK_MIN_HITS,
  stepsFor, checkBound, createTrial, feed, hit,
};
