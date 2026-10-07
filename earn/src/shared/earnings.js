'use strict';

const { ECON } = require('./config');

// Earnings estimates from a hashrate in TH/s, at the network difficulty, block
// reward, fee and PRL price. Probabilistic in reality — these are expectations.
// An `econ` override may be passed so the app can feed live values from the
// prlscan API (see shared/economics.js); it defaults to the static ECON
// constants, which are only a fallback and drift over time.

// The hashes a block takes on average, per unit of difficulty, in the unit
// hashrates are reported in. prlscan's own network hashrate is difficulty x 2^48
// / block time, and HeroMiners' calculator uses the same scale.
const HASHES_PER_DIFFICULTY = 2 ** 48;

// Expected PRL/day = the blocks your hashes find on average x the reward. The
// odds of a block per hash are set by difficulty alone, the way the pool works
// it out.
//
// This used to be your share of the network hashrate x the PRL the network makes
// a day, from two prlscan figures taken over different spans. The PRL/day used
// the last 100 blocks' actual times, while the hashrate estimate divides
// difficulty by a slower moving average of them. When blocks came faster than
// that average, the estimate counted the extra blocks but not the extra hashrate
// finding them: on 2026-10-06 that read 8% above HeroMiners' figure for the
// same card (6.70 PRL/day against 6.19). Faster blocks mean the network grew
// and difficulty has not caught up yet; they do not pay more per hash.
function estDailyPrl(ths, econ = ECON) {
  const t = Number(ths) || 0;
  const blocksPerDay = (t * 1e12 * 86400) / (econ.DIFFICULTY * HASHES_PER_DIFFICULTY);
  return blocksPerDay * econ.BLOCK_REWARD * econ.FEE;
}

function estDailyUsd(ths, econ = ECON) {
  return estDailyPrl(ths, econ) * econ.PRL_USD;
}

function estDailyUsdLabel(ths, econ = ECON) {
  return '$' + estDailyUsd(ths, econ).toFixed(2);
}

function prlToUsd(prl, econ = ECON) {
  return (Number(prl) || 0) * econ.PRL_USD;
}

function prlToUsdLabel(prl, econ = ECON) {
  return '$' + prlToUsd(prl, econ).toFixed(2);
}

module.exports = {
  estDailyPrl, estDailyUsd, estDailyUsdLabel, prlToUsd, prlToUsdLabel, HASHES_PER_DIFFICULTY,
};
