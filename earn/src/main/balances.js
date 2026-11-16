'use strict';

// The pool balances the desktop app's Settings shows: Pearl's for the payout
// address, Monero's for the CPU miner's. Fetched here in the main process so
// they aren't subject to the renderer's CSP and cross-origin rules. Every
// fetch is best-effort and never rejects: null on any failure. Moved out of
// main.js, unchanged.

const { getJson } = require('./io');
const { KRYPTEX, XMR, poolId } = require('../shared/config');
const {
  buildBalanceUrl, parseBalance, buildPoolStatsUrl, parsePoolPriceUsd, buildKryptexBalanceUrl, parseKryptexBalance,
} = require('../shared/balance');
const { isValidAddress } = require('../shared/address');
const { checkXmrAddress } = require('../shared/xmr/address');

// A Kryptex balance for either coin: the same fields as the HeroMiners ones,
// or null on any failure.
async function fetchKryptexBalance(coin, address, priceUsd) {
  const json = await getJson(buildKryptexBalanceUrl(KRYPTEX.apiBase, coin, address)).catch(() => null);
  return parseKryptexBalance(json, priceUsd);
}

// Fetch a pool balance for a payout address. Resolves the parsed
// { pending, paid, earned, usd } or null (unknown address, offline, non-200,
// bad JSON). `priceUsd` (optional) adds a USD figure. `pool` is the one
// Settings has picked for Pearl.
async function fetchBalance(address, priceUsd, pool) {
  if (!isValidAddress(address)) return null;
  if (poolId(pool) === 'kryptex') return fetchKryptexBalance('prl', address, priceUsd);
  let json;
  try {
    json = await getJson(buildBalanceUrl(String(address).trim()));
  } catch (e) {
    return null; // buildUrl threw
  }
  if (json == null) return null;
  try {
    return parseBalance(json, priceUsd);
  } catch (e) {
    return null; // parse threw
  }
}

// HeroMiners' XMR price, read at most every 10 minutes, as often as the PRL
// price. A failed read keeps the last price.
const XMR_PRICE_MAX_AGE_MS = 10 * 60 * 1000;
let xmrPrice = null;
let xmrPriceAt = 0;
async function xmrPriceUsd() {
  if (xmrPrice != null && Date.now() - xmrPriceAt < XMR_PRICE_MAX_AGE_MS) return xmrPrice;
  const usd = parsePoolPriceUsd(await getJson(buildPoolStatsUrl(XMR.apiBase)).catch(() => null));
  if (usd != null) {
    xmrPrice = usd;
    xmrPriceAt = Date.now();
  }
  return xmrPrice;
}

// The Monero balance on the CPU miner's pool. HeroMiners' Monero pool runs the
// same software and payload as the Pearl one, in 1e12 units, and with no
// `balance` field until there is one. Null on any failure. Its dollar value
// uses HeroMiners' XMR price, on either pool; without one, usd is null and
// Settings shows the balance without dollars.
async function fetchXmrBalance(address, pool) {
  if (!checkXmrAddress(address).ok) return null;
  if (poolId(pool) === 'kryptex') {
    const price = await xmrPriceUsd();
    return fetchKryptexBalance('xmr', address, price == null ? undefined : price);
  }
  const json = await getJson(buildBalanceUrl(String(address).trim(), XMR.apiBase)).catch(() => null);
  if (json == null) return null;
  const price = await xmrPriceUsd();
  return parseBalance(json, price == null ? undefined : price, { units: XMR.coinUnits, absentIsZero: true });
}

module.exports = { fetchBalance, fetchXmrBalance };
