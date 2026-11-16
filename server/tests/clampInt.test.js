const { clampInt } = require('../src/clampInt');
const { resolveMaxTokens } = require('../src/controllers/gatewayShared');
const OpenRouterService = require('../src/services/openRouterService');
const { clampMaxTokens, clampPriority, DEFAULT_MAX_TOKENS, MAX_TOKENS_CEILING } = require('../src/services/jobService');
const { clampMaxJobs, MAX_JOBS_PER_POLL } = require('../src/controllers/jobController');
const { countField, logLimit, MAX_REPORTED_TOKENS, MAX_LOG_LIMIT } = require('../src/controllers/logController');

describe('clampInt', () => {
  it('rounds down and caps at max', () => {
    expect(clampInt(7.9, { min: 0, max: 10, fallback: 3 })).toBe(7);
    expect(clampInt(99, { min: 0, max: 10, fallback: 3 })).toBe(10);
    expect(clampInt('4', { min: 0, max: 10, fallback: 3 })).toBe(4);
  });

  it('falls back for missing, non-numeric and non-finite values', () => {
    for (const v of [undefined, null, 'abc', NaN, Infinity, -Infinity, {}]) {
      expect(clampInt(v, { min: 0, max: 10, fallback: 3 })).toBe(3);
    }
  });

  it('falls back below min, or clamps there when asked', () => {
    expect(clampInt(-2, { min: 0, max: 10, fallback: 3 })).toBe(3);
    expect(clampInt(-2, { min: -1, max: 10, fallback: 0, clampLow: true })).toBe(-1);
  });

  it('has no bounds when none are given', () => {
    expect(clampInt(-5.5)).toBe(-6);
    expect(clampInt(undefined)).toBeUndefined();
  });
});

// The clamps were each hand-written before clampInt replaced them. These are
// the originals, copied verbatim, checked against the new versions over a wide
// spread of inputs so the move cannot have changed a rule.
describe('the clamps behave exactly as before', () => {
  const inputs = [
    undefined, null, '', ' ', 'abc', '5', ' 7 ', '1e3', '0x10', true, false, [], [5], {},
    NaN, Infinity, -Infinity, 0, -0, 0.4, 0.5, 0.99, 1, 1.5, 2, 9, 10, 10.5, 11, 100, 499, 500, 501,
    6400, 32767, 32768, 32769, 1e6, 1e7, 1e7 + 1, 1e18, -0.5, -1, -1.5, -2, -100, Number.MIN_VALUE,
  ];
  const old = {
    gateway(v, ceiling) {
      if (v == null) return undefined;
      const n = Number(v);
      if (Number.isFinite(n) && n > 0) return Math.min(Math.floor(n), ceiling);
      return ceiling;
    },
    openRouter(v, ceiling) {
      const n = Number(v);
      if (Number.isFinite(n) && n > 0) return Math.min(Math.floor(n), ceiling);
      return ceiling;
    },
    maxTokens(v) {
      if (v == null) return DEFAULT_MAX_TOKENS;
      const n = Math.floor(Number(v));
      if (!Number.isFinite(n) || n < 0) return DEFAULT_MAX_TOKENS;
      return Math.min(n, MAX_TOKENS_CEILING);
    },
    priority(v) {
      const n = Math.floor(Number(v));
      if (!Number.isFinite(n)) return 0;
      return Math.min(Math.max(n, -1), 10);
    },
    maxJobs(v) {
      const n = Math.floor(Number(v));
      if (!Number.isFinite(n) || n < 1) return 1;
      return Math.min(n, MAX_JOBS_PER_POLL);
    },
    count(v) {
      const n = Math.floor(Number(v));
      if (!Number.isFinite(n) || n < 0) return 0;
      return Math.min(n, MAX_REPORTED_TOKENS);
    },
    limit(v) {
      const n = Math.floor(Number(v));
      if (!Number.isFinite(n) || n < 1) return 50;
      return Math.min(n, MAX_LOG_LIMIT);
    },
  };
  const openRouter = new OpenRouterService({ maxTokens: 2048 });

  it.each(inputs.map((v) => [v]))('for %p', (v) => {
    expect(resolveMaxTokens(v, 32768)).toBe(old.gateway(v, 32768));
    expect(openRouter.resolveMaxTokens(v)).toBe(old.openRouter(v, openRouter.maxTokens));
    expect(clampMaxTokens(v)).toBe(old.maxTokens(v));
    expect(clampPriority(v)).toBe(old.priority(v));
    expect(clampMaxJobs(v)).toBe(old.maxJobs(v));
    expect(countField(v)).toBe(old.count(v));
    expect(logLimit(v)).toBe(old.limit(v));
  });
});
