const { Pool } = require('pg');

// All persistent state lives in Postgres. Jobs keep their full payload in a
// `data` jsonb column (with a few promoted columns for querying); everything
// else is columnar. Expirations/TTLs are modeled with explicit *_at columns
// that callers compare against the current time.

// Live status of LLMJob Earn mining clients (public network board). Kept as its
// own constant so the add-miners migration can apply it to existing databases.
const MINERS_SCHEMA = `
CREATE TABLE IF NOT EXISTS miners (
  id text PRIMARY KEY,
  address text,
  worker text,
  gpu text,
  region text,
  hashrate double precision,
  accepted bigint,
  vram_used double precision,
  vram_total double precision,
  version text,
  llm_model text,
  node_id text,
  first_seen bigint,
  last_seen bigint
);
CREATE INDEX IF NOT EXISTS idx_miners_last_seen ON miners (last_seen);
`;

// Free public web chat, served through the OpenRouter proxy. We deliberately do
// NOT store prompts or replies — only per-request performance and token counts,
// plus a single running-totals row so free usage can be summed and capped. Kept
// as its own constant so the add-chat-usage migration can apply it to existing
// databases.
const CHAT_SCHEMA = `
CREATE TABLE IF NOT EXISTS chat_requests (
  id text PRIMARY KEY,
  ts bigint,
  model text,
  in_tokens integer,
  out_tokens integer,
  total_tokens integer,
  speed double precision,
  latency_ms integer,
  ttft_ms integer,
  finish text
);
CREATE INDEX IF NOT EXISTS idx_chat_requests_ts ON chat_requests (ts);

CREATE TABLE IF NOT EXISTS chat_usage_totals (
  id text PRIMARY KEY,
  requests bigint DEFAULT 0,
  in_tokens bigint DEFAULT 0,
  out_tokens bigint DEFAULT 0,
  total_tokens bigint DEFAULT 0,
  api_total_tokens bigint DEFAULT 0
);
`;

const SCHEMA = `
CREATE TABLE IF NOT EXISTS nodes (
  node_id text PRIMARY KEY,
  seq serial,
  public_key text,
  name text,
  user_id text,
  status text,
  is_public boolean DEFAULT false,
  last_seen bigint,
  claimed_at bigint,
  capabilities jsonb,
  active_jobs integer,
  max_concurrent_jobs integer,
  device text,
  vram_total double precision,
  vram_used double precision,
  model text,
  quant text,
  tps double precision,
  -- Server-measured generation speed, deliberately separate from the tps column
  -- above: that one is whatever the node reports on its ping, and a number that
  -- decides which jobs a node is offered (and eventually what it earns) can't be
  -- self-reported. These are computed at the server from the tokens a job
  -- produced over the wall time the server itself observed.
  measured_tps double precision,
  speed_samples integer DEFAULT 0,
  speed_at bigint
);
CREATE INDEX IF NOT EXISTS idx_nodes_user ON nodes (user_id);
-- "Which nodes are live right now" is the most-asked question in the system:
-- every job creation resolves its model pin against it, /v1/models answers from
-- it, and the public network page reads it on a 15s timer. Without this each of
-- those was a full scan of the nodes table.
CREATE INDEX IF NOT EXISTS idx_nodes_last_seen ON nodes (last_seen);

CREATE TABLE IF NOT EXISTS api_keys (
  hash text PRIMARY KEY,
  id text,
  user_id text,
  name text,
  masked text,
  created_at bigint,
  last_used bigint,
  usage bigint DEFAULT 0,
  visibility text DEFAULT 'public'
);
CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys (user_id);

CREATE TABLE IF NOT EXISTS request_logs (
  id text PRIMARY KEY,
  user_id text,
  ts bigint,
  model text,
  node text,
  app text,
  in_tokens integer,
  out_tokens integer,
  speed double precision,
  finish text,
  key text
);
CREATE INDEX IF NOT EXISTS idx_request_logs_user_ts ON request_logs (user_id, ts);

CREATE TABLE IF NOT EXISTS node_join_tokens (
  user_id text PRIMARY KEY,
  token text UNIQUE,
  created_at bigint
);

CREATE TABLE IF NOT EXISTS jobs (
  id text PRIMARY KEY,
  data jsonb,
  status text,
  priority integer,
  created_at bigint,
  updated_at bigint,
  user_id text,
  visibility text,
  target_node text,
  -- Promoted out of the data column so assignment can filter on it: a node is only offered
  -- work it can finish inside the gateway's budget at its measured speed.
  max_tokens integer,
  -- Promoted for the same reason. NULL means any eligible node; set only when the
  -- caller named a model the fleet actually advertises, in which case only a node
  -- running that model is offered the job.
  model text,
  assigned_to text,
  lock_node text,
  lock_token text,
  lock_expires_at bigint,
  heartbeat_at bigint
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs (status);
-- The assignment query, which every serving node runs on every poll — by far the
-- hottest statement here. A bare status index still left Postgres sorting the
-- whole pending set to satisfy "ORDER BY priority DESC, created_at ASC"; a
-- partial index in exactly that order returns the queue head as an index scan.
-- Partial on the pending status so it stays small: it holds only the live queue,
-- not the day of completed jobs the table keeps for the cleanup sweep.
CREATE INDEX IF NOT EXISTS idx_jobs_pending_queue
  ON jobs (priority DESC, created_at ASC) WHERE status = 'pending';
-- The timeout and cleanup sweeps ("assigned/running whose lock or heartbeat has
-- gone stale", "completed/failed older than a day").
CREATE INDEX IF NOT EXISTS idx_jobs_status_updated ON jobs (status, updated_at);

CREATE TABLE IF NOT EXISTS job_chunks (
  job_id text,
  idx integer,
  chunk jsonb,
  PRIMARY KEY (job_id, idx)
);

${MINERS_SCHEMA}
${CHAT_SCHEMA}
`;

async function initSchema(db) {
  await db.query(SCHEMA);
}

// Pool sizing, stated rather than inherited. pg's defaults are `max: 10` and no
// connection timeout, and both were wrong here:
//
//   • 10 is small for a process whose job dispatch, both chat gateways and the
//     dashboard all check out connections concurrently. Every node holding a
//     long poll takes one for the length of each claim transaction, so a modest
//     fleet plus a handful of in-flight requests could sit at the ceiling.
//   • With no connectionTimeoutMillis a request that arrives at an exhausted
//     pool waits FOREVER rather than failing. That turns a brief spike into a
//     pile of sockets that never answer — the caller's own timeout fires first
//     and the work is done anyway, for nobody.
//
// Both are env-overridable so the founder can retune against whatever the
// hosted database actually allows without a code change.
const DEFAULT_POOL_MAX = 20;
const DEFAULT_CONNECT_TIMEOUT_MS = 10000;

function numberEnv(v, fallback) {
  const n = Number(v);
  return Number.isFinite(n) && n > 0 ? n : fallback;
}

function createPool() {
  const pool = new Pool({
    connectionString: process.env.DATABASE_URL || 'postgres://localhost:5432/llmjob',
    max: numberEnv(process.env.PGPOOL_MAX, DEFAULT_POOL_MAX),
    connectionTimeoutMillis: numberEnv(process.env.PGPOOL_CONNECT_TIMEOUT_MS, DEFAULT_CONNECT_TIMEOUT_MS)
  });
  // REQUIRED, not optional hygiene: pg emits 'error' on the Pool when a client
  // sitting IDLE in it drops — a Postgres restart, a failover, an idle-timeout
  // kill by the provider. That is routine on a hosted database. An 'error' event
  // with no listener is rethrown by EventEmitter as an uncaught exception, so a
  // one-second database blip took the entire API process down. Logging it is
  // enough: pg discards the dead client and the next query checks out a fresh one.
  pool.on('error', (err) => {
    console.error('Postgres pool error (idle client):', err.message);
  });
  return pool;
}

module.exports = { createPool, initSchema, SCHEMA, MINERS_SCHEMA, CHAT_SCHEMA };
