'use strict';

// Browser-origin allowlist for CORS. The site is served both from llmjob.com
// (GitHub Pages) and from the Railway app itself, and the pages call the API
// cross-origin (API_BASE points at the Railway prod host), so those origins
// must be allowed. Any other website trying to call the API from a browser —
// e.g. embedding the free chat proxy on its own page — is refused.
//
// Non-browser callers (curl, the CLI, cluster nodes, server-side API-key use)
// send no Origin header, so they are always allowed: CORS is a browser-only
// control and never gates them. This is deliberately NOT auth or rate limiting;
// it only stops third-party *browser* origins from using our endpoints.

const ALLOWED_ORIGINS = new Set([
  'https://llmjob.com',
  'https://www.llmjob.com',
  'https://llmjob-production.up.railway.app',
]);

const ALLOWED_PATTERNS = [
  /^http:\/\/localhost(:\d+)?$/, // local dev
  /^http:\/\/127\.0\.0\.1(:\d+)?$/, // local dev
  /^https:\/\/llmjob-llmjob-pr-\d+\.up\.railway\.app$/, // Railway PR previews
];

// True when a request's Origin is allowed to read cross-origin responses. A
// missing origin (non-browser request) is always allowed.
function isAllowedOrigin(origin) {
  if (!origin) return true;
  if (ALLOWED_ORIGINS.has(origin)) return true;
  return ALLOWED_PATTERNS.some((re) => re.test(origin));
}

// The callback form the `cors` package expects for its `origin` option. Never
// errors — a disallowed origin just resolves to `false`, so the request still
// completes server-side but the browser withholds the response from the caller.
function corsOrigin(origin, cb) {
  cb(null, isAllowedOrigin(origin));
}

module.exports = { isAllowedOrigin, corsOrigin, ALLOWED_ORIGINS, ALLOWED_PATTERNS };
