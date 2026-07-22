'use strict';

const { ECON } = require('./config');

// Earnings estimates from a hashrate in TH/s, at the network hashrate, daily
// emission, pool fee and PRL price. Probabilistic in reality — these are
// expectations. An `econ` override may be passed so the app can feed live
// values from the prlscan API (see shared/economics.js); it defaults to the
// static ECON constants, which are only a fallback and drift over time.

function estDailyPrl(ths, econ = ECON) {
  const t = Number(ths) || 0;
  return (t / econ.NET_TH) * econ.DAILY_NET_PRL * econ.FEE;
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

module.exports = { estDailyPrl, estDailyUsd, estDailyUsdLabel, prlToUsd, prlToUsdLabel };
