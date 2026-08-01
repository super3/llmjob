# Review backlog

Findings from the codebase review that are **not yet fixed**. The top-10 from that
review landed in #167 and #168; this is what was left over, plus the dead code and
duplication the review turned up along the way.

Every item below was **re-verified against `main` at `ca91112`** — after #165,
#167, #168, #169, #171 and #172 — so nothing here is stale by construction. Each
entry cites the code as it exists at that commit. Items the review raised that
have since been closed, or that turned out not to be true, are listed at the
bottom rather than silently dropped.

Ordered roughly by priority. Numbering is stable; if an item is fixed, mark it
done rather than renumbering the rest.

Legend: **S**/**M**/**L** = rough effort.

---

## Bugs and logic errors

### 1. `earn.html` shows two different earnings figures for the same GPU — **M**

`site/pages/earn.html:627` (mockup) vs `:707-709` (calculator)

The page states PRL economics twice and the copies have drifted ~2.3×:

```js
// :627  the animated app mockup
elEst.textContent = '$' + (((s.total / 30.79e6) * 1.2e6 * 0.99) * 0.47).toFixed(2);

// :707-709  the earnings calculator
let PRICE = 0.30;  let NET_TH = 60.6e6;  let DAILY_NET_PRL = 1.62e6;
```

The calculator also refreshes those three from the prlscan API (`:816`, `:825`,
`:835`) while the mockup's stay frozen forever, so the gap widens over time.
`:624` uses the same stale `0.47` for the balance's USD figure.

**Fix:** hoist one set of constants (ideally the live-refreshed ones) and have the
mockup read them.

### 2. The page's fee copy contradicts both the config and the calculator — **S**

`site/pages/earn.html:405` vs `earn/src/shared/config.js:39-40` and `earn.html:710`

```
:405   "A flat 1% dev fee, 1% pool. You keep 98% of everything you mine"
config devFeePct: 0, poolFeePct: 1        →  1% total
:710   const FEE = 0.99;                  →  1% total
:468   "...and the 1% pool fee"           →  1% total
```

Three places say 1%; the headline copy says 2% and "98%". The copy *understates*
what the calculator directly below it shows.

**Fix:** correct the copy to 1%/99%, and template the number from the config
constants so it can't drift again.

### 3. Nodes offline for a week are hard-deleted, losing ownership — **M**

`server/src/services/nodeService.js:225`, `NODE_TTL_MS` at `:3`

```js
await this.db.query('DELETE FROM nodes WHERE last_seen < $1', [now - NODE_TTL_MS]);
```

`NODE_TTL_MS` is 7 days and the delete does not exclude *claimed* nodes. A rig
offline for a week loses its name, its `user_id` and its `is_public` setting. When
it comes back it self-registers as unclaimed, so a user who relied on `private`
routing silently drops to public-only until they notice and re-claim.

**Fix:** only prune rows with `user_id IS NULL`, or soft-delete and keep claimed
identities indefinitely.

### 4. `--gpu` silently disables multi-GPU difficulty scaling — **S**

`earn/src/cli/earn-cli.js:613-621`

```js
if (!settings.gpuProvided) {          // ← the whole block, including the scaling
  ...
  settings.gpuCount = det.count > 1 ? det.count : 1;
  if (!settings.difficultyProvided) settings.difficulty = difficultyForCard(det.name) * settings.gpuCount;
}
```

Passing `--gpu` to name the card skips auto-detect *and* the `* gpuCount` scaling,
so an N-card rig submits shares at 1/N the intended difficulty.

**Fix:** separate "which card" from "how many cards" — detect the count even when
the name is supplied.

### 5. Auto-detection overwrites the user's saved pool region on every launch — **S**

`earn/src/renderer/renderer.js:664-667`

```js
if (api.detectRegion && !state.mining) {
  const region = await api.detectRegion();
  if (region) el.setRegion.value = region;   // clobbers the persisted choice
}
```

A user who deliberately picks `eu1` because `us2` is flaky for them gets silently
reverted next launch.

**Fix:** only auto-detect when no region has been explicitly saved.

### 6. Downloaded executables are never integrity-checked — **M**

`earn/src/main/io.js:284-294`

The alpha-miner engine, the `llama-server` binary and the multi-GB GGUF model are
downloaded, renamed into place and later executed with no checksum or signature
check. `received` is tracked (`:285`, `:290`) but never compared against
`content-length` before the rename:

```js
const total = parseInt(res.headers['content-length'] || '0', 10);
let received = 0;
...
try { fs.renameSync(part, dest); } catch (e) { return reject(e); }
```

So a truncated transfer can land a corrupt binary at the final path, which every
later start accepts (installation is checked with `existsSync`). The long comment
above `downloadFile` explains that resume was removed to avoid exactly this class
of corruption — the same gap remains on the non-resume path.

Separately, a non-200, non-redirect response rejects immediately instead of going
through `retryOrFail`, so a transient CDN 502/503 aborts setup while a socket drop
retries four times.

**Fix:** pin a SHA-256 next to each pinned URL and verify before the rename (the
URLs already point at immutable release assets); compare `received` to `total` as
a cheap first guard; route non-200 through the retry path.

### 7. Client-controlled job priority can starve the global queue — **S**

`server/src/controllers/jobController.js:50,64` → `server/src/services/jobService.js:55`

`priority` comes straight from `req.body` into `createJob`, and
`assignJobsToNode` orders the *global* pending set by `priority DESC`. Every other
producer writes `0` (both gateways), so any signed-in user can submit at
`2147483647` and be served ahead of all paid API and public-chat traffic.

**Fix:** clamp to a small range, or ignore caller-supplied priority entirely and
derive it server-side.

### 8. `/api/miners/ping` lets anyone overwrite any miner's public row — **M**

`server/src/services/minerService.js:16-17,131`

The route is unauthenticated and upserts on `id = sha256(address + '|' + worker)`,
while `GET /api/miners` publishes both `address` and `worker`. Anyone can scrape
the board, recompute a victim's row id, and POST to blank or falsify their entry —
or flood the table with unique workers, since nothing bounds row count.

**Fix:** require the node signature for reports that claim an address already on
the board, or accept reports only from registered nodes.

### 9. Node signatures bind neither the request nor a nonce — **M**

`server/src/middleware/signature.js:14,20`

```js
const timeDiff = Math.abs(now - timestamp);   // NaN for a non-numeric timestamp
if (timeDiff > 5 * 60 * 1000) { ... }         // NaN > x is false → check skipped
...
const message = `${nodeId}:${timestamp}`;     // no method, no path, no body
```

Two issues. The signed bytes cover only `nodeId:timestamp`, so one captured signed
body authenticates *any* signature-guarded route for five minutes. And a
non-numeric `timestamp` makes the freshness comparison `NaN`, which is never
`>` the window, so the check is skipped entirely — a node can mint a signature
that never expires.

Note this is defence-in-depth: `jobController._requireNode` still checks the
presented key against the registered one, so an attacker needs a genuine captured
signature, not just a forged body.

**Fix:** sign method + path + a body hash + a nonce, and coerce/validate
`timestamp` as a number before comparing.

---

## Duplication that has already drifted

### 10. Payout-address validation exists three times, and the server doesn't normalize — **M**

`earn/src/shared/address.js:7,14,18` · `server/src/services/minerService.js:11,103` · `earn/src/renderer/renderer.js:65`

```js
// shared/address.js — no /i flag, but normalizes (lowercases) before testing
const ADDRESS_RE = /^prl1p[0-9a-z]{20,80}$/;
isValidAddress = (a) => ADDRESS_RE.test(normalizeAddress(a));   // trim + toLowerCase

// server/minerService.js — /i flag, but only trims. Never lowercases.
const ADDRESS_RE = /^prl1p[0-9a-z]{20,80}$/i;
isValidAddress = (a) => ADDRESS_RE.test(String(a ?? '').trim());
```

The server therefore accepts `PRL1P…` and stores it un-normalized, and
`minerFingerprint` keys the row on that raw string — so `PRL1P…` and `prl1p…` are
two different board rows for one miner, splitting its hashrate and worker count.

**Fix:** one shared validator that normalizes at the boundary; normalize on write
in `reportMiner`.

### 11. The node's server URL is resolved inconsistently — **S**

`earn/src/main/main.js:894,932` vs `:974,993`

```js
:894  postJson(NODE.serverUrl + '/api/nodes/ping', …)              // ignores node.serverUrl
:932  postJson(NODE.serverUrl + '/api/nodes/join', …)              // ignores it
:974  postJson((node.serverUrl || NODE.serverUrl) + '/api/nodes/register', …)   // honours it
:993  serverUrl: node.serverUrl || NODE.serverUrl                  // honours it
```

`llmjob-earn-cli connect --server https://staging…` persists `serverUrl` into the
shared `node.json`; the GUI then registers and polls against staging but pings and
joins against production.

**Fix:** resolve it once into a module-level accessor and use that everywhere.

### 12. `esc()` is defined twice and the copies escape different characters — **S**

`site/pages/chat.html:235` · `site/pages/network.html:256`

```js
// chat.html   — escapes & < > " '
// network.html — escapes & < >  "     (no single quote)
```

Both currently interpolate only into double-quoted attributes, so neither is
exploitable today — but two escapers with different character sets is exactly the
shape that becomes an XSS the first time someone uses the weaker one in a
single-quoted attribute.

**Fix:** hoist one escaper into a shared partial.

### 13. Token formatting drifted between two pages — **S**

`site/pages/network.html:397-401` · `site/pages/dashboard.html:201`

network.html has a billions tier and an uppercase `K`; dashboard.html has neither
(`'k'`, and nothing above millions). The same token total renders differently
depending on which page you're on.

### 14. The Clerk bootstrap is copy-pasted across three pages — **M**

`site/pages/{dashboard,earn,index}.html`

Three implementations of "wait for the async Clerk script, then `Clerk.load()`,
then route on signed-in state", with different timeouts and different failure
behaviour — including one that leaves the page spinning with no error state if
`Clerk.load()` rejects.

**Fix:** one `site/partials/clerk-boot.html`.

### 15. The gateway timeout lives in three places and the invariant comment is now false — **S**

`server/src/services/jobService.js:12-15` · `openaiController.js:47` · `chatController.js:85`

`jobService`'s comment asserts "Both gateways give up waiting after 120s" and
sizes `PENDING_TTL_MS` on that. The OpenAI gateway was since raised to 280s while
the chat gateway stayed at 120s. The 5-minute TTL still clears 280s, so nothing is
broken today — but the stated reasoning is wrong and the margin is now 20s rather
than the 180s the comment implies.

**Fix:** export one constant, derive `PENDING_TTL_MS` from it, and fix the comment.

---

## Performance

### 16. `chat.html` re-renders the whole conversation on every streamed token — **M**

`site/pages/chat.html:445`

```js
if (obj.delta) { state.streamText += obj.delta; render(); }
```

`render()` rebuilds the entire message list via `innerHTML`, re-escaping and
re-running `renderMarkdown` over every prior message, then re-queries and re-wires
listeners. At 30–60 tok/s in a long conversation this is O(n) DOM work per token.

**Fix:** append into the streaming node only, and do the full render once on
finish.

### 17. Unbounded `SELECT *` behind unauthenticated endpoints — **S**

`server/src/services/minerService.js:183` · `server/src/services/nodeService.js:180`

`GET /api/miners` and `GET /api/nodes/public` both read whole tables and filter in
JS. Combined with #8 (anyone can insert miner rows), the miners query is a cheap
amplification target.

**Fix:** filter and bound in SQL.

### 18. No index supports the hot job-poll query — **S**

`server/src/db.js:131`

The only jobs index is `idx_jobs_status`. The poll query filters on `status` +
`visibility` + `target_node` and sorts by `priority DESC, created_at ASC`, so the
sort is unsupported; the hourly cleanup scan is unsupported too.

**Fix:** a composite index on `(status, priority DESC, created_at)` and one on
`(status, updated_at)` for cleanup.

---

## Dead code

Each verified by grepping the whole repo — `server/`, `earn/`, `site/` including
inline `<script>`, `.github/`, migrations, both test dirs and package scripts.

### 19. `minerArgs.buildEnv` is entirely dead — **S**
`earn/src/shared/minerArgs.js:56` — imported only by `earn/test/minerArgs.test.js`.
It builds environment variables for a Windows `.bat` launcher that does not exist
in the repo; the live path is `MinerManager`, which spawns with argv.
`resolveBinary`'s third `gpu` parameter is likewise never passed by any caller,
making its `WIN_BINARIES[gpu]` branch unreachable outside its test.

### 20. The `chat_requests` table is write-only — **S**
`server/src/services/chatUsageService.js:103` — a row is inserted per request
(latency, TTFT, speed, model, per-request token counts), trimmed to a cap, and the
only reader is `getRecent()`, which no route or client calls.

### 21. The user-facing job API has no client — **S**
`POST /api/jobs`, `GET /api/jobs/stats`, `GET /api/jobs/:jobId` — both in-repo
gateways call `jobService.createJob` directly, and nothing in `site/` or `earn/`
calls these routes. `submitJob`'s `'anonymous'` userId fallback is also
unreachable behind `requireAuth`.

### 22. Node public/private visibility is unwired end to end — **S**
Nothing calls `PUT /api/nodes/:id/visibility` (the dashboard wires visibility for
API *keys* only) and nothing calls `GET /api/nodes/public`. The server code works
and is correct — it just has no client, so `is_public` never becomes true in
practice and the public-nodes list is always empty.

### 23. The `miner:event` IPC bridge has no consumer — **S**
`earn/src/main/preload.js:32` exposes `onEvent` for the `miner:event` channel that
`main.js:358` emits; `renderer.js` never subscribes. The data already reaches the
UI via the stats path.

### 24. `server/tests/loadTest.js` is an unreachable orphan — **S**
~390 lines referenced by nothing — no npm script, no CI workflow, no other file —
and its dependencies are not installed, so it cannot run as written.

### 25. Two `MinerService` static exports are unused — **S**
`server/src/services/minerService.js:218-219` — `dropHostAggregates` and
`groupHosts` are attached as statics but nothing outside the module reads them,
not even the tests. (The functions themselves are live, used internally at `:188`
and `:203` — it's only the exports that are dead.)

### 26. `earnings.prlToUsd` / `prlToUsdLabel` are unreachable — **S**
`earn/src/shared/earnings.js` — nothing outside the module calls either; the
renderer formats USD inline.

### 27. `LLM.model.layers` is read by nothing — **S**
`earn/src/shared/config.js` — and its comment claims it feeds `--n-gpu-layers`,
which is actively misleading: the shipped command line uses `ALL_LAYERS` (999) by
design, precisely so a wrong layer count can't leave part of the model on the CPU.

### 28. `dashboard.html` ships helpers referenced by nothing — **S**
`colorFor`, `initials`, `nodeColor`, `PALETTE` — each appears exactly once, at its
own definition.

### 29. `chat.html`'s post-reply `refreshServed()` is a guaranteed no-op — **S**
`site/pages/chat.html:479` — by the time it runs, `render()` on the line above has
replaced the container's `innerHTML`, and `#served` (`:328`) exists only inside
`emptyStateHtml()`. With a reply on screen the element is gone, so the function
returns immediately. The call at `:499` (which clears messages first) does work.

### 30. `miners.first_seen` is written and never read — **S**
`server/src/db.js:24` — set on every ping; the row mapper in `getPublicMiners`
doesn't project it and nothing else selects it.

---

## Structural

### 31. `jobs` keeps status in two places — **L**

The promoted `status` column and `data->>'status'` inside the jsonb blob are
updated non-atomically and read by *different* code paths:
`assignJobsToNode`/`checkTimeouts`/`expireStalePending` filter on the column
(`jobService.js:136,281,313`), while `getJobResult` branches on the blob
(`:364,368,380`). Any divergence means a job that is invisible to the poller but
reported as running to the caller, or vice versa. Several other fields are
duplicated the same way.

**Fix:** make the column authoritative and derive the blob's copy on read, or drop
the promoted columns and index the jsonb.

### 32. `earn/src/main/main.js` is a god object — **L**

~1200 lines, 19 mutable module globals, eight independent subsystems (window,
settings, miner, LLM fleet, node link, chat, updater, economics). This is the
direct cause of the lifecycle races fixed in #167 (`miningEpoch`) and #168
(`LlmFleet` run-scoped flags) — both were module globals read after a long
`await`. Worth extracting the serving stack and the miner stack into owned objects,
as its own PR.

---

## Closed since the review

Recorded so they don't get re-raised:

- **Requeued jobs keeping the previous attempt's chunks** — purge landed in #167,
  the heartbeat margin that triggered it in #168.
- **`expireStalePending` racing a node's claim** — #167 added the
  `AND status = 'pending'` guard on the UPDATE.
- **The two gateways duplicating helpers / the `/v1` SSE stream missing
  `X-Accel-Buffering`** — #167 (`gatewayShared.js`), clamps added in #168.
- **Unbounded `maxJobs`, `/nodes/claim` proof-of-possession, unbounded
  self-reported tokens, the pool `'error'` listener, `temperature: 0`, the Electron
  `before-quit` leak, Linux GPU detection, the two `LlmFleet` races, and the
  `llmFleet.js` coverage-gate hole** — all in #168.
- **`add-node.html` as an orphaned page** — its claim flow was removed in #168.
- **`server/tests/loadTest.js`** is still present (see #24) but the *page* it was
  paired with is gone.

Two items the review raised turned out **not to be true** and are deliberately
absent above:

- *"`nodes.active_jobs` / `max_concurrent_jobs` are write-only."* They are read
  back in `nodeService.getNode` (`:253-254`).
- *"The miner-report loop is duplicated between the GUI and the CLI."* Both shells
  already call the shared `buildMinerReports` (`main.js:268`,
  `earn-cli.js:683`); only the surrounding call site differs, which is expected.

---

## Not addressed, tracked elsewhere

**Rate limiting.** There is no throttle anywhere on the server, and
`POST /api/chat/completions` is unauthenticated while spending OpenRouter credit.
This was implemented in #168 and then removed as out of scope for that PR. It is
not in the numbered list above because it is a known, deliberate deferral rather
than an undiscovered gap.
