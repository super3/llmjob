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

module.exports = { pad2, formatUptime, formatHashrate, formatInt, formatLogTime };
