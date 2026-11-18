// A caller-supplied number, as an integer in [min, max].
//
// Missing (null/undefined) or non-finite values give `fallback`. So does a
// value under `min`, unless `clampLow` is set, in which case it becomes `min`.
// A value over `max` becomes `max`. Fractions round down.
//
// The range is checked before rounding down. For a whole-number `min` that
// gives the same answer as rounding first. It also lets a caller pass
// `min: Number.MIN_VALUE` to mean "any value above 0", which is how the
// max_tokens resolvers treat 0 as unset while still rounding 0.5 down to 0.
//
// Each caller keeps its own rules through these options, and they differ on
// purpose: jobService keeps max_tokens 0, the gateways treat it as unset.
function clampInt(v, { min = -Infinity, max = Infinity, fallback, clampLow = false } = {}) {
  if (v == null) return fallback;
  const n = Number(v);
  if (!Number.isFinite(n)) return fallback;
  if (n < min) return clampLow ? min : fallback;
  return Math.min(Math.floor(n), max);
}

module.exports = { clampInt };
