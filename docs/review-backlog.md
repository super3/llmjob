# Review backlog

What is left of the codebase review after the fixes landed. The top 10 went in
#167 and #168; the rest were verified against `main` at `ca91112` and then fixed
in this PR, except the items listed under **Still open** below.

Keep this file honest: when something here is fixed, move it out rather than
leaving it to rot. Numbers refer to the original review's ordering and are stable.

---

## Still open

### 31. `jobs` keeps status in two places — **large**

`server/src/services/jobService.js`

The promoted `status` column and `data->>'status'` inside the jsonb blob are
updated non-atomically and read by *different* code paths:
`assignJobsToNode` / `checkTimeouts` / `expireStalePending` filter on the column,
while `getJobResult` branches on the blob. Any divergence means a job invisible to
the poller but reported as running to the caller, or vice versa. Several other
fields are duplicated the same way.

Not fixed here because the two sensible resolutions — make the column
authoritative and derive the blob on read, or drop the promoted columns and index
the jsonb — are both schema-migration-shaped and want their own PR.

### 32. `earn/src/main/main.js` is a god object — **large**

~1200 lines, 19 mutable module globals, eight independent subsystems (window,
settings, miner, LLM fleet, node link, chat, updater, economics). This is the
direct cause of the lifecycle races fixed in #167 (`miningEpoch`) and #168
(`LlmFleet` run-scoped flags) — both were module globals read after a long
`await`. Worth extracting the serving stack and the miner stack into owned
objects, as its own PR.

### Rate limiting

There is no throttle anywhere on the server, and `POST /api/chat/completions` is
unauthenticated while spending OpenRouter credit. Implemented in #168 and removed
as out of scope for that PR, so this is a known deferral rather than an
undiscovered gap.

### Node signatures bind neither the request nor a nonce — **partly fixed**

`server/src/middleware/signature.js`

The NaN-timestamp bypass is fixed (a non-numeric timestamp is now rejected rather
than skipping the freshness window). The *scope* problem remains: the signed bytes
are still only `nodeId:timestamp`, so one captured signed body authenticates any
signature-guarded route for five minutes, and there is no nonce.

Left open deliberately — widening the signed payload is a protocol change that
breaks every deployed earn client until it updates, so it needs a rollout plan
(accept both forms, then drop the old one) rather than a one-line edit.

### `/api/miners/ping` can overwrite another miner's board row — **partly fixed**

`server/src/services/minerService.js`

The address is now normalized, so case can no longer fork one miner into two rows.
The underlying issue stands: the route is unauthenticated and upserts on
`sha256(address|worker)`, and `GET /api/miners` publishes both inputs — so anyone
can recompute a victim's row id and overwrite it. Fixing it properly means
requiring a node signature for reports claiming an address already on the board,
which is the same protocol-change problem as above.

### Features with no client

Not bugs, and not obviously deletable — each is working server code that simply
has nothing calling it. Listed so the next person doesn't rediscover them:

- **`chat_requests` is write-only.** A row per request (latency, TTFT, speed,
  model, token counts) is inserted and trimmed to a cap; the only reader,
  `chatUsageService.getRecent()`, is called by no route. Either surface it or drop
  the table — both are product calls.
- **The user-facing job API has no client.** `POST /api/jobs`,
  `GET /api/jobs/stats` and `GET /api/jobs/:jobId` are unused: both gateways call
  `jobService.createJob` directly. Removing public endpoints is a product call.
- **Node public/private visibility is unwired end to end.** Nothing calls
  `PUT /api/nodes/:id/visibility` (the dashboard wires visibility for API *keys*
  only) and nothing calls `GET /api/nodes/public`, so `is_public` never becomes
  true and the public-nodes list is always empty. The server side is correct and
  ready; it needs a UI.

---

## Fixed in this PR

Bugs and logic errors — 1 `earn.html`'s two drifted earnings figures · 2 the fee
copy contradicting the config and the calculator · 3 claimed nodes hard-deleted
after a week offline · 4 `--gpu` disabling multi-GPU difficulty scaling ·
5 auto-detect clobbering a saved pool region · 6 downloads renamed into place
without a size check, and a transient HTTP failure aborting setup instead of
retrying · 7 client-controlled job priority starving the global queue · 9 the
NaN-timestamp freshness bypass.

Duplication — 10 payout-address validation normalizing in one copy but not the
other · 11 the node's server URL resolved inconsistently across ping/join vs
register/poll · 12 two `esc()` helpers escaping different characters ·
13 token formatting drifted between two pages · 15 the gateway timeout restated
in three places with a comment asserting an invariant that no longer held.

Performance — 17 unbounded `SELECT *` behind unauthenticated endpoints ·
18 no index supporting the hot poll query or the cleanup scans.

Dead code — 19 `minerArgs.buildEnv` and the never-passed `gpu` parameter ·
23 the `miner:event` IPC channel and its preload bridge · 24 the orphaned
`server/tests/loadTest.js` · 25 two unreferenced `MinerService` static exports ·
26 `earnings.prlToUsd` / `prlToUsdLabel` · 27 `LLM.model.layers` and its
misleading comment · 28 `dashboard.html`'s unused colour/initials helpers ·
29 `chat.html`'s no-op `refreshServed()` call · 30 the write-only
`miners.first_seen` column.

Items 14 (the Clerk bootstrap copy-pasted across three pages) and 16
(`chat.html` re-rendering per streamed token) are not in this PR — both are
refactors of working code rather than fixes, and are better done deliberately than
folded into a batch of bug fixes.

---

## Closed before this PR

- **Requeued jobs keeping the previous attempt's chunks** — purge in #167, the
  heartbeat margin that triggered it in #168.
- **`expireStalePending` racing a node's claim** — #167 added the
  `AND status = 'pending'` guard.
- **The two gateways duplicating helpers / `/v1` missing `X-Accel-Buffering`** —
  #167 (`gatewayShared.js`), clamps in #168.
- **Unbounded `maxJobs`, `/nodes/claim` proof-of-possession, unbounded
  self-reported tokens, the pool `'error'` listener, `temperature: 0`, the
  `before-quit` leak, Linux GPU detection, the two `LlmFleet` races and the
  coverage-gate hole** — #168.

Two findings the review raised turned out **not to be true**, recorded so they
don't get re-raised:

- *"`nodes.active_jobs` / `max_concurrent_jobs` are write-only."* They are read
  back in `nodeService.getNode`.
- *"The miner-report loop is duplicated between the GUI and the CLI."* Both shells
  already call the shared `buildMinerReports`; only the call site differs.
