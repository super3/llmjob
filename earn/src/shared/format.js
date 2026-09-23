'use strict';

// Display formatting helpers shared by the main process and tests.

function pad2(n) {
  return String(n).padStart(2, '0');
}

// Seconds -> `2h 14m 08s` (hours omitted when zero), e.g. for uptime.
function formatUptime(totalSeconds) {
  const s = Math.max(0, Math.floor(Number(totalSeconds) || 0));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  return (h > 0 ? h + 'h ' : '') + pad2(m) + 'm ' + pad2(sec) + 's';
}

function formatHashrate(ths) {
  return (Number(ths) || 0).toFixed(1);
}

function formatInt(n) {
  return (Number(n) || 0).toLocaleString('en-US');
}

function formatLogTime(d) {
  const date = d instanceof Date ? d : new Date(d);
  return date.toLocaleTimeString('en-GB');
}

// The device label: which cards are mining, from their names.
//
//   ['RTX 4090']                    -> 'RTX 4090'
//   ['RTX 4090', 'RTX 4090']        -> '2x RTX 4090'
//   ['RTX PRO 4500', 'RTX 4070']    -> 'RTX PRO 4500 + RTX 4070'
//
// A rig mines on every card it has, so one name is the honest answer only when
// there is one card. Identical cards count rather than repeat, because a 13-card
// rig would otherwise print the same string thirteen times. Mixed cards are
// listed: the names are the point, and dropping them is how a rig ends up
// labelled with a card that isn't doing the work.
//
// Returns null when there is nothing to name, so the caller can fall back.
function formatDeviceLabel(names) {
  const list = (Array.isArray(names) ? names : [])
    .map((n) => (n == null ? '' : String(n).trim()))
    .filter(Boolean);
  if (!list.length) return null;
  const unique = [...new Set(list)];
  if (unique.length === 1) return list.length > 1 ? list.length + 'x ' + unique[0] : unique[0];
  return unique.join(' + ');
}

module.exports = {
  pad2, formatUptime, formatHashrate, formatInt, formatLogTime, formatDeviceLabel,
};
