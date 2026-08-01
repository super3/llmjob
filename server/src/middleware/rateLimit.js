'use strict';

// Fixed-window per-client rate limiting.
//
// Deliberately dependency-free and in-process. The endpoints that most need a
// limit are the ones that cost real money — POST /api/chat/completions is
// unauthenticated and proxies to OpenRouter on our key, and the network-model
// branch of it enqueues fleet jobs with no budget gate at all. CORS does not
// help: corsOptions allows a request with no Origin (correctly — CORS is a
// browser control), so any script or curl loop reaches them unthrottled.
//
// In-process means the counter is per server instance, so N instances allow N×
// the limit. That is fine for the threat this closes (a single client hammering
// the free tier); it is not a billing control. Swap in a shared store if the
// deployment ever runs more than one instance and the limit needs to be exact.
//
// Fixed window rather than sliding: one integer and one timestamp per client,
// no per-request array to grow. A caller can burst up to 2×max across a window
// boundary, which is an acceptable trade for the memory profile.

// Identify the client. Behind Railway/nginx, Express populates req.ip from
// X-Forwarded-For only when the app sets `trust proxy` (index.js does). Falls
// back to the raw socket address, then to a single shared bucket — a missing
// identity must not mean "unlimited".
function clientKey(req) {
  return (req && req.ip)
    || (req && req.socket && req.socket.remoteAddress)
    || 'unknown';
}

// Build a rate-limit middleware.
//   windowMs  length of the fixed window
//   max       requests allowed per client per window
//   now       injectable clock (tests)
//   keyFn     injectable client identity (tests)
function rateLimit(opts = {}) {
  const windowMs = opts.windowMs || 60000;
  const max = opts.max || 60;
  const now = opts.now || (() => Date.now());
  const keyFn = opts.keyFn || clientKey;

  // key -> { count, resetAt }
  const hits = new Map();

  // Drop windows that have already expired. Called on every request, which is
  // enough to keep the map proportional to *active* clients rather than to every
  // client ever seen — without it this is an unbounded memory leak on a public
  // endpoint, i.e. a slower version of the DoS it exists to prevent.
  function sweep(t) {
    for (const [k, v] of hits) {
      if (v.resetAt <= t) hits.delete(k);
    }
  }

  function middleware(req, res, next) {
    const t = now();
    sweep(t);

    const key = keyFn(req);
    const entry = hits.get(key);

    if (!entry) {
      hits.set(key, { count: 1, resetAt: t + windowMs });
      return next();
    }

    entry.count += 1;
    if (entry.count > max) {
      const retryAfter = Math.max(1, Math.ceil((entry.resetAt - t) / 1000));
      if (res.setHeader) res.setHeader('Retry-After', String(retryAfter));
      return res.status(429).json({
        error: {
          message: 'Too many requests. Please slow down and try again shortly.',
          type: 'rate_limit_error',
          code: null,
        },
      });
    }
    return next();
  }

  // Exposed for tests and for a future shared-store swap.
  middleware.hits = hits;
  return middleware;
}

module.exports = { rateLimit, clientKey };
