'use strict';

// Parse a line of `alpha-miner` stdout into a structured event.
//
// The exact engine output format is configurable; these patterns cover the
// conventional Stratum-miner log shapes (and the lines shown in the design
// mock): connection lines, accepted/rejected shares, and periodic hashrate
// reports. Unrecognized lines return null so callers can pass them through as
// raw log text.

const UNIT_TO_TH = {
  'th/s': 1,
  'gh/s': 1e-3,
  'mh/s': 1e-6,
  'kh/s': 1e-9,
  'h/s': 1e-12,
};

// Convert a "<number> <unit>" hashrate to TH/s. Unknown units are assumed
// to already be TH/s.
function parseHashrateValue(num, unit) {
  const n = Number(num) || 0;
  const factor = UNIT_TO_TH[String(unit).toLowerCase()];
  return n * (factor == null ? 1 : factor);
}

function parseLine(line) {
  const s = String(line == null ? '' : line).trim();
  if (!s) return null;

  const conn = s.match(/^connected to (\S+?)(?:[\s·]+worker\s+(\S+))?$/i);
  if (conn) {
    return { type: 'connected', endpoint: conn[1], worker: conn[2] || null };
  }

  const share = s.match(/\b(accepted|rejected)\b[^#]*share(?:\s*#?\s*([\d,]+))?/i);
  if (share) {
    return {
      type: 'share',
      status: share[1].toLowerCase(),
      index: share[2] ? Number(share[2].replace(/,/g, '')) : null,
    };
  }

  const hr = s.match(/([\d.]+)\s*(TH\/s|GH\/s|MH\/s|kH\/s|H\/s)/i);
  if (hr) {
    const load = s.match(/load\s+(\d+)\s*%/i);
    const power = s.match(/(\d+)\s*W\b/i);
    return {
      type: 'hashrate',
      hashrate: parseHashrateValue(hr[1], hr[2]),
      load: load ? Number(load[1]) : null,
      power: power ? Number(power[1]) : null,
    };
  }

  return null;
}

module.exports = { UNIT_TO_TH, parseHashrateValue, parseLine };
