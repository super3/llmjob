'use strict';

// A hit check, as the rotation fold's check (shared/foldCheck) and each core
// clock offset step (shared/coreOffsetTrial) run it: the card searches a job of
// the miner's own, each hit is recomputed on the CPU, and the check passes at
// `need` right hits. One wrong hit fails it, and so does running out of time.

// One recomputed hit. `state.hits` counts the right ones; `why` is null when
// this one was right, or what differed. Returns 'fail', 'pass' or null (keep
// going).
function hitCheck(state, why, need) {
  if (why) return 'fail';
  state.hits += 1;
  return state.hits >= need ? 'pass' : null;
}

// Why a check that ran out of time failed: it had `hits` of the `need` it
// needs in `ms`.
function tooFewHits(hits, need, ms) {
  return 'only ' + hits + ' of the ' + need + ' hits it needs in ' + ms / 1000 + ' s';
}

module.exports = { hitCheck, tooFewHits };
