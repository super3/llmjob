'use strict';

// Compare two dotted numeric versions: -1, 0 or 1, part by part, a missing part
// counting as 0.
//
// Two readings, for two kinds of input:
//   strict false  a release tag ("v0.7.3"): a leading v is dropped, and a part
//                 that is not a number counts as 0, so odd input never throws
//                 (shared/selfUpdate).
//   strict true   a driver version ("32.0.31007.5012"): anything that is not
//                 numbers between dots is no version, and the answer is null
//                 (shared/amdDriver).
function parseVersion(v, strict) {
  const text = String(v == null ? '' : v).trim();
  if (strict) return /^\d+(\.\d+)*$/.test(text) ? text.split('.').map(Number) : null;
  return text.replace(/^v/i, '').split('.').map((p) => parseInt(p, 10) || 0);
}

function compareVersions(a, b, { strict = false } = {}) {
  const x = parseVersion(a, strict);
  const y = parseVersion(b, strict);
  if (!x || !y) return null;
  for (let i = 0; i < Math.max(x.length, y.length); i++) {
    const d = (x[i] || 0) - (y[i] || 0);
    if (d) return d < 0 ? -1 : 1;
  }
  return 0;
}

module.exports = { parseVersion, compareVersions };
