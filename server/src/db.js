const { Pool } = require('pg');

// All persistent state lives in Postgres. Jobs keep their full payload in a
// `data` jsonb column (with a few promoted columns for querying); everything
// else is columnar. Expirations/TTLs are modeled with explicit *_at columns
// that callers compare against the current time.
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
  tps double precision
);
CREATE INDEX IF NOT EXISTS idx_nodes_user ON nodes (user_id);

CREATE TABLE IF NOT EXISTS api_keys (
  hash text PRIMARY KEY,
  id text,
  user_id text,
  name text,
  masked text,
  created_at bigint,
  last_used bigint,
  usage bigint DEFAULT 0
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
  assigned_to text,
  lock_node text,
  lock_expires_at bigint,
  heartbeat_at bigint
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs (status);

CREATE TABLE IF NOT EXISTS job_chunks (
  job_id text,
  idx integer,
  chunk jsonb,
  PRIMARY KEY (job_id, idx)
);

CREATE TABLE IF NOT EXISTS miners (
  id text PRIMARY KEY,
  address text,
  worker text,
  gpu text,
  region text,
  hashrate double precision,
  accepted bigint,
  first_seen bigint,
  last_seen bigint
);
CREATE INDEX IF NOT EXISTS idx_miners_last_seen ON miners (last_seen);
`;

async function initSchema(db) {
  await db.query(SCHEMA);
}

function createPool() {
  return new Pool({
    connectionString: process.env.DATABASE_URL || 'postgres://localhost:5432/llmjob'
  });
}

module.exports = { createPool, initSchema, SCHEMA };
