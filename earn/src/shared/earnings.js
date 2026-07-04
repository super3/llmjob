'use strict';

const { ECON } = require('./config');

// Earnings estimates from a hashrate in TH/s, at the live network hashrate,
// daily emission, pool fee and PRL price. Probabilistic in reality — these are
// expectations, matching the figures shown in the design mock.

function estDailyPrl(ths) {
  const t = Number(ths) || 0;
  return (t / ECON.NET_TH) * ECON.DAILY_NET_PRL * ECON.FEE;
}

function estDailyUsd(ths) {
  return estDailyPrl(ths) * ECON.PRL_USD;
}

function estDailyUsdLabel(ths) {
  return '$' + estDailyUsd(ths).toFixed(2);
}

function prlToUsd(prl) {
  return (Number(prl) || 0) * ECON.PRL_USD;
}

function prlToUsdLabel(prl) {
  return '$' + prlToUsd(prl).toFixed(2);
}

module.exports = { estDailyPrl, estDailyUsd, estDailyUsdLabel, prlToUsd, prlToUsdLabel };
