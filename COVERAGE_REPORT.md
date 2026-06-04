# Test Coverage Report

## Summary
The server test suite reports **100% coverage** across all metrics.

| Metric | Coverage |
|--------|----------|
| **Statements** | 100% |
| **Branches** | 100% |
| **Functions** | 100% |
| **Lines** | 100% |

`npm test` runs Jest with `--coverage` and the `coverageThreshold` in the root
`jest.config.js` is set to 100% for statements, branches, functions and lines,
so the build fails if coverage regresses.

## Coverage by Module

| Module | Coverage |
|--------|----------|
| Routes | 100% |
| Controllers | 100% |
| Middleware | 100% |
| Services | 100% |

> `server/src/index.js` is excluded from coverage in `jest.config.js` because it
> is the process entry point (binds the port, wires up timers and signal
> handlers).

## How the suite stays at 100%

The server code talks to Postgres through the real `pg` `Pool` (parameterized
queries) with no compatibility shim. Tests run against an in-memory Postgres,
`server/tests/helpers/pgmem.js`, which builds a `pg`-compatible pool with
[pg-mem](https://github.com/oguimbal/pg-mem) and applies the same `SCHEMA` from
`server/src/db.js`. Because production and tests exercise real SQL against the
same schema, there are no dual code paths and no `istanbul ignore` annotations.

Coverage is driven entirely by behaviour-focused suites — one per unit — rather
than a dedicated "hit every line" file:

- `jobService.test.js` / `nodeService.test.js` — service lifecycles plus the
  edge branches (lock contention, heartbeat/timeout transitions, cleanup).
- `jobController.test.js` / `nodeController.test.js` — controller responses and
  error handling, including the node-verification and auth-fallback branches.
- `dashboardRoutes.test.js` — API keys, request logs, usage, and join tokens.
- `routes.test.js` / `routesJobs.test.js` — route wiring and handlers
  end-to-end through the Express router.
- `auth.test.js` / `signature.test.js` — middleware.
- `db.test.js` — the pool factory and shared schema.
- `server/tests/helpers/pgmem.js` — the in-memory Postgres used by the tests.
