'use strict';

// The check a card's rotation fold has to pass before the miner trusts it
// (--rotation-fold on the CLI, LLMJOB_ROTATION_FOLD=1 in the desktop app's
// environment; off unless asked for). Pure: PearlMiner feeds it the hits and
// the time (main/pearlMiner, _feedFoldCheck).
//
// Why a check. The rotation fold is a second kernel for RTX 30 and 40 cards,
// about 6% faster than the normal one on an RTX 4090 and an RTX 3090
// (tuning/benchmark.md). Its inner loop is written by hand, instruction by
// instruction, and it has mined on few cards. A fold that is wrong on a card
// still finds hashes that meet the target, so every share would be sent and
// rejected, and a pool can ban an address for that.
//
// So a card that opens on the rotation fold first searches a job of the miner's
// own, with a target easy enough to hit a few times a second, and each hit is
// recomputed on the CPU the way a pool would (shared/miner/checkHit). At
// CHECK_HITS right hits the card goes on to the pool's job. One wrong hit, or
// too few hits when the time is up, and the card is opened again on the normal
// kernel for the rest of the run. After the check every share the card finds
// is still recomputed before it is sent (main/pearlMiner, _foldHitRight): a
// loop that is wrong once in a thousand tiles passes a check of 24.
//
// One card is checked at a time (main/pearlMiner, _feedFoldCheck), so a rig
// of many cards spends about 8 s a card before all of them are trusted, and
// each mines the pool's job, its shares recomputed, while it waits its turn.
//
// The numbers. A hit at the fold's k 65536 takes about 0.13 s of the miner's
// event loop to recompute (120 to 147 ms on rig01's CPU, six runs), against
// 4.6 ms at k 2048, because the fold and the noise are 32 times the work. So
// the check asks for 3 hits a second, not the offset trial's 30, and 24 hits:
// about 8 s on any card. After CHECK_MS it ends anyway, and fails: a card that
// hits at a seventh of the rate its job was set for is not computing what it
// should.

// The check's jobs. Their header is no block's (coreOffsetTrial's check uses
// the same), so nothing found for one could be a share.
const FOLD_CHECK_JOB = 'fold-check-';
const CHECK_HITS = 24;
const CHECK_HITS_PER_S = 3;
const CHECK_MS = 60 * 1000;

// A new check, waiting for the card's first hashrate sample: the check's
// target is set from the card's own rate.
function createCheck() {
  return { state: 'waiting', jobId: null, hits: 0, since: 0 };
}

// The check's job starts at `now` (ms).
function begin(check, jobId, now) {
  check.state = 'checking';
  check.jobId = jobId;
  check.hits = 0;
  check.since = now;
}

// One hit of the check's job, recomputed: `why` is null when it was right, or
// what differed. Returns 'pass', 'fail' or null (keep going). A hit of another
// job, or one that arrives when no check is under way, changes nothing.
function hit(check, jobId, why) {
  if (check.state !== 'checking' || check.jobId !== jobId) return null;
  if (why) {
    check.state = 'failed';
    check.why = why;
    return 'fail';
  }
  check.hits += 1;
  if (check.hits < CHECK_HITS) return null;
  check.state = 'passed';
  return 'pass';
}

// A hashrate sample at `now`: 'begin' when the check should start (the card's
// first sample above 0), 'fail' when a check has run out of time, else null.
function tick(check, th, now) {
  if (check.state === 'waiting') return th > 0 ? 'begin' : null;
  if (check.state !== 'checking' || now - check.since < CHECK_MS) return null;
  check.state = 'failed';
  check.why = 'only ' + check.hits + ' of the ' + CHECK_HITS + ' hits it needs in ' + CHECK_MS / 1000 + ' s';
  return 'fail';
}

module.exports = {
  FOLD_CHECK_JOB, CHECK_HITS, CHECK_HITS_PER_S, CHECK_MS, createCheck, begin, hit, tick,
};
