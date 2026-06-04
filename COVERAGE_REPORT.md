# Test Coverage Report

## Summary
The server test suite now reports **100% coverage** across all metrics.

| Metric | Coverage |
|--------|----------|
| **Statements** | 100% |
| **Branches** | 100% |
| **Functions** | 100% |
| **Lines** | 100% |

`npm test` runs Jest with `--coverage` and the `coverageThreshold` in
`jest.config.js` is set to 100% for statements, branches, functions and lines,
so the build fails if coverage regresses.

## Coverage by Module

| Module | Coverage |
|--------|----------|
| Routes | 100% |
| Controllers | 100% |
| Middleware | 100% |
| Repositories | 100% |
| Services | 100% |
| Utils | 100% |

> `server/src/index.js` is excluded from coverage in `jest.config.js` because it
> is the process entry point (binds the port, wires up timers and signal
> handlers).

## How the remaining gaps were closed

The previous report stopped at ~85% because the repository layer, the V2
services and the Redis compatibility layers were only partially exercised. The
following work brought everything to 100%:

1. **Both Redis compatibility paths are tested.** The compat layers branch on
   whether the client exposes camelCase (Redis v5) or lowercase (redis-mock,
   callback-based) methods. A small camelCase adapter
   (`server/tests/helpers/camelRedis.js`) delegates to redis-mock for correct
   semantics while exposing the v5-style API, so the same suites run against
   both code paths.
2. **Repositories and V2 services** (`BaseRepository`, `JobRepository`,
   `NodeRepository`, `JobServiceV2`, `NodeServiceV2`) are covered end-to-end,
   including error conditions, ownership checks, queue transitions, lock
   handling, timeouts, cleanup and statistics.
3. **Job routes** are exercised through the Express router with the auth and
   signature middleware mocked to pass through, so every route handler runs.
4. **Defensive fallbacks** (e.g. `result || []`, missing records, error
   callbacks) are triggered with purpose-built mock clients.
5. **Genuinely unreachable defensive code** — callback fallbacks for Redis
   operations whose method name is identical in both the v5 and redis-mock APIs
   (so the `typeof` guard is always satisfied) — is annotated with
   `/* istanbul ignore else */` and a comment explaining why it cannot be hit.

## Key test files

- `server/tests/coverage-100.test.js` — repository/service/compat coverage.
- `server/tests/routesJobs.test.js` — job route handlers end-to-end.
- `server/tests/helpers/camelRedis.js` — camelCase Redis client used by the tests.
