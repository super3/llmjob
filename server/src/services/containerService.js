const crypto = require('crypto');

// Rented containers on our own rigs (issue #318). An admin creates a job — an
// image pinned by digest, its environment, which rigs may take it, how long it
// runs and on how many rigs — the way we create a Salad container group today.
// Rigs ask for work on their regular poll, claim a run on one card, send
// heartbeats and log lines while it runs, and report how it ended. A run whose
// rig goes quiet is released, so another rig can take the slot.
//
// Every run's job time goes into container_minutes, the ledger owners are paid
// on per hour (the rate may be $0 while we test; the minutes count from the
// start). There is no daily cap.

// Only images from our own registry, pinned by digest. A tag can be moved to a
// different image after the job is created; a digest cannot. The rig checks the
// same rule again before it pulls (earn/src/shared/rental.js).
const DEFAULT_REGISTRY = 'ghcr.io/super3/';
const DIGEST_RE = /^[a-z0-9]+(?:[._/-][a-z0-9]+)*@sha256:[0-9a-f]{64}$/;
const ENV_KEY_RE = /^[A-Za-z_][A-Za-z0-9_]{0,127}$/;
const NODE_ID_RE = /^([0-9a-f]{6}|[0-9a-f]{16})$/;

const MAX_ENV_KEYS = 64;
const MAX_ENV_VALUE = 8192;
const MAX_DURATION_SEC = 7 * 24 * 3600;    // a week
const MAX_REPLICAS = 100;
const MAX_PAY_RATE = 100;                  // USD an hour, a sanity clamp
const MAX_CARDS = 32;

// A run that has sent nothing for this long is released. Rigs send a heartbeat
// every 15 s while a run is up, pulling included, so this is about ten missed.
const RUN_TIMEOUT_MS = 3 * 60 * 1000;
// A run still going this long after its job time ran out is told to stop.
const OVERRUN_GRACE_MS = 60 * 1000;

const MAX_LOG_LINES = 20000;               // per run
const MAX_LOG_BATCH = 500;                 // per request
const MAX_LINE_CHARS = 2000;
const MAX_LOG_PAGE = 1000;

// How long a rig waits between polls when there is nothing for it.
const POLL_MS = 60 * 1000;

const ACTIVE = ['claimed', 'running'];
// Runs that did what the job asked, and so fill a slot for good.
const COMPLETED = ['exited', 'expired'];
// How a rig may say a run ended. Anything else is 'failed'.
const END_REASONS = ['exited', 'expired', 'stopped', 'taken_back', 'temperature', 'refused', 'failed'];

function registryPrefix() {
  return process.env.CONTAINER_REGISTRY || DEFAULT_REGISTRY;
}

// Whether `image` is ours and pinned by digest. Returns an error string or null.
function imageProblem(image) {
  const s = typeof image === 'string' ? image.trim() : '';
  if (!s) return 'image is required';
  const prefix = registryPrefix();
  if (!s.startsWith(prefix)) return 'image must come from ' + prefix;
  if (!DIGEST_RE.test(s.slice(prefix.length))) return 'image must be pinned by digest (name@sha256:<64 hex>)';
  return null;
}

// The environment as { KEY: 'value' }, or an error string.
function cleanEnv(env) {
  if (env == null) return {};
  if (typeof env !== 'object' || Array.isArray(env)) return 'env must be an object';
  const keys = Object.keys(env);
  if (keys.length > MAX_ENV_KEYS) return 'env has more than ' + MAX_ENV_KEYS + ' keys';
  const out = {};
  for (const k of keys) {
    if (!ENV_KEY_RE.test(k)) return 'bad env key: ' + k.slice(0, 40);
    const v = env[k] == null ? '' : String(env[k]);
    if (v.length > MAX_ENV_VALUE) return 'env value too long: ' + k;
    out[k] = v;
  }
  return out;
}

// The target as stored, or an error string. Either one exact rig and card, or
// a filter on the card's name, VRAM, driver and vendor. Empty takes any rig.
function cleanTarget(t) {
  if (t == null) return {};
  if (typeof t !== 'object' || Array.isArray(t)) return 'target must be an object';
  if (t.nodeId != null) {
    const nodeId = String(t.nodeId).trim();
    if (!NODE_ID_RE.test(nodeId)) return 'target.nodeId is not a node id';
    const card = Number(t.card);
    if (!Number.isInteger(card) || card < 0 || card >= MAX_CARDS) return 'target.card must be a card index';
    return { nodeId, card };
  }
  const out = {};
  if (t.gpu != null && String(t.gpu).trim()) out.gpu = String(t.gpu).trim().slice(0, 80);
  if (t.minVramMb != null) {
    const n = Number(t.minVramMb);
    if (!Number.isFinite(n) || n < 0) return 'target.minVramMb must be a number';
    out.minVramMb = n;
  }
  if (t.minDriver != null) {
    const n = parseFloat(t.minDriver);
    if (!Number.isFinite(n) || n < 0) return 'target.minDriver must be a number';
    out.minDriver = n;
  }
  if (t.vendor != null) {
    const v = String(t.vendor).toLowerCase();
    if (v !== 'nvidia' && v !== 'amd') return 'target.vendor must be nvidia or amd';
    out.vendor = v;
  }
  return out;
}

function newId(prefix) {
  return prefix + crypto.randomBytes(8).toString('hex');
}

function num(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

// The minutes a run is paid for: from the container starting to `untilMs`,
// never more than the job's duration. A run that never started is paid nothing.
function paidMinutes(startedAt, untilMs, durationSec) {
  if (startedAt == null) return 0;
  const start = Number(startedAt);
  const ms = Math.max(0, Math.min(untilMs - start, durationSec * 1000));
  return +(ms / 60000).toFixed(3);
}

// Whether `card` (as a rig reports it) on a rig that reported `rig` fits `target`.
function cardFits(target, card, rig) {
  const t = target || {};
  if (t.nodeId != null) return t.nodeId === rig.nodeId && t.card === card.index;
  if (t.gpu && !String(card.gpu || '').toLowerCase().includes(t.gpu.toLowerCase())) return false;
  if (t.minVramMb != null && !(Number(card.vramMb) >= t.minVramMb)) return false;
  if (t.minDriver != null && !(parseFloat(rig.driver) >= t.minDriver)) return false;
  if (t.vendor && String(card.vendor || '').toLowerCase() !== t.vendor) return false;
  return true;
}

// The cards a rig reports, cleaned: [{ index, gpu, vramMb, vendor, free }].
function cleanCards(cards) {
  if (!Array.isArray(cards)) return [];
  const out = [];
  for (const c of cards.slice(0, MAX_CARDS)) {
    if (!c || typeof c !== 'object') continue;
    const index = Number(c.index);
    if (!Number.isInteger(index) || index < 0 || index >= MAX_CARDS) continue;
    out.push({
      index,
      gpu: c.gpu ? String(c.gpu).slice(0, 80) : null,
      vramMb: num(c.vramMb),
      vendor: c.vendor ? String(c.vendor).toLowerCase().slice(0, 16) : null,
      free: c.free === true,
    });
  }
  return out;
}

function cleanLines(lines) {
  if (!Array.isArray(lines)) return [];
  return lines.slice(0, MAX_LOG_BATCH).map((l) => String(l == null ? '' : l).slice(0, MAX_LINE_CHARS));
}

function jobView(row) {
  return {
    id: row.id,
    name: row.name,
    image: row.image,
    env: row.env,
    target: row.target,
    durationSec: Number(row.duration_sec),
    replicas: Number(row.replicas),
    payRateUsdHour: Number(row.pay_rate_usd_hour),
    status: row.status,
    createdBy: row.created_by || null,
    createdAt: num(row.created_at),
    updatedAt: num(row.updated_at),
  };
}

function runView(row, minutes) {
  return {
    id: row.id,
    jobId: row.job_id,
    nodeId: row.node_id,
    card: Number(row.card),
    gpu: row.gpu || null,
    worker: row.worker || null,
    status: row.status,
    reason: row.reason || null,
    exitCode: row.exit_code == null ? null : Number(row.exit_code),
    claimedAt: num(row.claimed_at),
    startedAt: num(row.started_at),
    heartbeatAt: num(row.heartbeat_at),
    endedAt: num(row.ended_at),
    logLines: Number(row.log_lines) || 0,
    minutes: Number(minutes),
  };
}

class ContainerService {
  constructor(db, opts = {}) {
    this.db = db;
    this.now = opts.now || Date.now;
  }

  // ── Admin side ────────────────────────────────────────────────────────────

  async createJob(input, createdBy = null) {
    const imgErr = imageProblem(input.image);
    if (imgErr) return { error: imgErr, status: 400 };
    const env = cleanEnv(input.env);
    if (typeof env === 'string') return { error: env, status: 400 };
    const target = cleanTarget(input.target);
    if (typeof target === 'string') return { error: target, status: 400 };
    const durationSec = Number(input.durationSec);
    if (!Number.isInteger(durationSec) || durationSec < 60 || durationSec > MAX_DURATION_SEC) {
      return { error: 'durationSec must be a whole number from 60 to ' + MAX_DURATION_SEC, status: 400 };
    }
    const replicas = input.replicas == null ? 1 : Number(input.replicas);
    if (!Number.isInteger(replicas) || replicas < 1 || replicas > MAX_REPLICAS) {
      return { error: 'replicas must be a whole number from 1 to ' + MAX_REPLICAS, status: 400 };
    }
    // An exact rig and card can only ever hold one run at a time.
    if (target.nodeId != null && replicas !== 1) return { error: 'an exact rig and card takes replicas: 1', status: 400 };
    const rate = input.payRateUsdHour == null ? 0 : Number(input.payRateUsdHour);
    if (!Number.isFinite(rate) || rate < 0 || rate > MAX_PAY_RATE) {
      return { error: 'payRateUsdHour must be from 0 to ' + MAX_PAY_RATE, status: 400 };
    }
    const id = newId('cj_');
    const now = this.now();
    const name = input.name ? String(input.name).slice(0, 80) : id;
    await this.db.query(
      `INSERT INTO container_jobs (id, name, image, env, target, duration_sec, replicas, pay_rate_usd_hour,
                                   status, created_by, created_at, updated_at)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'pending', $9, $10, $10)`,
      [id, name, input.image.trim(), JSON.stringify(env), JSON.stringify(target), durationSec, replicas, rate,
        createdBy, now]
    );
    return { job: await this._jobWithRuns(id) };
  }

  async listJobs(limit = 50) {
    const n = Math.max(1, Math.min(200, Math.floor(Number(limit)) || 50));
    const r = await this.db.query('SELECT * FROM container_jobs ORDER BY created_at DESC LIMIT $1', [n]);
    return { jobs: r.rows.map(jobView) };
  }

  async getJob(id) {
    const job = await this._jobWithRuns(id);
    if (!job) return { error: 'Job not found', status: 404 };
    return { job };
  }

  // Log lines in order, after `since` (a line's id), for one run or every run
  // of the job. `next` is the id to pass as `since` for the following page.
  async getLogs(id, { since, limit, runId } = {}) {
    const job = await this.db.query('SELECT id FROM container_jobs WHERE id = $1', [id]);
    if (!job.rows.length) return { error: 'Job not found', status: 404 };
    const after = Math.max(0, Math.floor(Number(since)) || 0);
    const n = Math.max(1, Math.min(MAX_LOG_PAGE, Math.floor(Number(limit)) || MAX_LOG_PAGE));
    const params = [id, after, n];
    let runFilter = '';
    if (runId) { runFilter = ' AND l.run_id = $4'; params.push(String(runId)); }
    const r = await this.db.query(
      `SELECT l.id, l.run_id, l.seq, l.ts, l.line, r.node_id, r.card
         FROM container_logs l JOIN container_runs r ON r.id = l.run_id
        WHERE r.job_id = $1 AND l.id > $2${runFilter}
        ORDER BY l.id ASC LIMIT $3`,
      params
    );
    const lines = r.rows.map((x) => ({
      id: Number(x.id), runId: x.run_id, nodeId: x.node_id, card: Number(x.card), seq: Number(x.seq),
      ts: num(x.ts), line: x.line,
    }));
    return { lines, next: lines.length ? lines[lines.length - 1].id : after };
  }

  // Stop a job. Runs still going are told to stop on their next heartbeat, and
  // the job reads 'stopping' until they have; with none going it is 'stopped'.
  async deleteJob(id) {
    const r = await this.db.query('SELECT status FROM container_jobs WHERE id = $1', [id]);
    if (!r.rows.length) return { error: 'Job not found', status: 404 };
    const active = await this._activeCount(this.db, id);
    const status = active > 0 ? 'stopping' : 'stopped';
    if (r.rows[0].status !== 'stopped') {
      await this.db.query('UPDATE container_jobs SET status = $2, updated_at = $3 WHERE id = $1', [id, status, this.now()]);
    }
    return { job: await this._jobWithRuns(id) };
  }

  // Job time per rig and card, for paying owners: one row per (node, card)
  // with its minutes, hours and what it is owed at each job's rate.
  async ledger({ nodeId } = {}) {
    const params = [];
    let where = '';
    if (nodeId) { where = 'WHERE node_id = $1'; params.push(String(nodeId)); }
    const r = await this.db.query(
      `SELECT node_id, card, SUM(minutes) AS minutes, SUM(amount_usd) AS amount, COUNT(*) AS runs
         FROM container_minutes ${where} GROUP BY node_id, card ORDER BY node_id, card`,
      params
    );
    return {
      rows: r.rows.map((x) => ({
        nodeId: x.node_id,
        card: Number(x.card),
        runs: Number(x.runs),
        minutes: +Number(x.minutes).toFixed(3),
        hours: +(Number(x.minutes) / 60).toFixed(4),
        amountUsd: +Number(x.amount).toFixed(4),
      })),
    };
  }

  // Rigs that asked for work in the last ten minutes, with their cards, so a
  // job can name one exact rig and card.
  async listRigs() {
    const r = await this.db.query(
      'SELECT * FROM container_rigs WHERE last_poll >= $1 ORDER BY node_id', [this.now() - 10 * 60 * 1000]
    );
    return {
      rigs: r.rows.map((x) => ({
        nodeId: x.node_id,
        worker: x.worker || null,
        platform: x.platform || null,
        docker: !!x.docker,
        driver: x.driver || null,
        cards: x.cards,
        lastPoll: num(x.last_poll),
      })),
    };
  }

  // ── Rig side ──────────────────────────────────────────────────────────────

  // A rig asks for work. It reports its cards, marking `free` the ones it would
  // run a job on now: mining or idle, not running a job, and not while its LLM
  // answers a prompt (the rig decides that and offers no card then). Claims at
  // most one run, on the first free card some job fits.
  async poll(nodeId, body = {}) {
    const now = this.now();
    const cards = cleanCards(body.cards);
    const rig = {
      nodeId,
      worker: body.worker ? String(body.worker).slice(0, 64) : null,
      platform: body.platform ? String(body.platform).slice(0, 16) : null,
      docker: body.docker === true,
      driver: body.driver ? String(body.driver).slice(0, 32) : null,
    };
    await this.db.query(
      `INSERT INTO container_rigs (node_id, worker, platform, docker, driver, cards, last_poll)
       VALUES ($1, $2, $3, $4, $5, $6, $7)
       ON CONFLICT (node_id) DO UPDATE SET worker = EXCLUDED.worker, platform = EXCLUDED.platform,
         docker = EXCLUDED.docker, driver = EXCLUDED.driver, cards = EXCLUDED.cards, last_poll = EXCLUDED.last_poll`,
      [nodeId, rig.worker, rig.platform, rig.docker, rig.driver, JSON.stringify(cards), now]
    );
    await this.releaseStale();
    // Only a rig that can run a container is handed one: Linux with docker.
    // Windows has no runner yet.
    const free = cards.filter((c) => c.free);
    if (!rig.docker || rig.platform !== 'linux' || !free.length) return { job: null, pollMs: POLL_MS };
    const run = await this._claim(rig, free, now);
    return { job: run, pollMs: POLL_MS };
  }

  async _claim(rig, free, now) {
    const client = await this.db.connect();
    try {
      await client.query('BEGIN');
      // Cards of this rig already running something are not free, whatever it said.
      const busy = await client.query(
        `SELECT card FROM container_runs WHERE node_id = $1 AND status IN ('claimed', 'running')`, [rig.nodeId]
      );
      const busyCards = new Set(busy.rows.map((x) => Number(x.card)));
      const cards = free.filter((c) => !busyCards.has(c.index));
      const jobs = await client.query(
        `SELECT * FROM container_jobs WHERE status IN ('pending', 'running')
          ORDER BY created_at ASC LIMIT 50 FOR UPDATE SKIP LOCKED`
      );
      for (const row of jobs.rows) {
        const target = row.target;
        const runs = await client.query('SELECT node_id, status FROM container_runs WHERE job_id = $1', [row.id]);
        const taken = runs.rows.filter((x) => ACTIVE.includes(x.status) || COMPLETED.includes(x.status)).length;
        if (taken >= Number(row.replicas)) continue;
        // A filter job runs once per rig: a rig that already had a go at it
        // (finished, failed, gave it back) is not handed it again. An exact
        // target can only go to its one rig, so it may try again.
        if (target.nodeId == null && runs.rows.some((x) => x.node_id === rig.nodeId)) continue;
        const card = cards.find((c) => cardFits(target, c, rig));
        if (!card) continue;
        const runId = newId('cr_');
        await client.query(
          `INSERT INTO container_runs (id, job_id, node_id, card, gpu, worker, status, claimed_at, heartbeat_at)
           VALUES ($1, $2, $3, $4, $5, $6, 'claimed', $7, $7)`,
          [runId, row.id, rig.nodeId, card.index, card.gpu, rig.worker, now]
        );
        await client.query(
          `INSERT INTO container_minutes (run_id, job_id, node_id, card, minutes, pay_rate_usd_hour, amount_usd, updated_at)
           VALUES ($1, $2, $3, $4, 0, $5, 0, $6)`,
          [runId, row.id, rig.nodeId, card.index, Number(row.pay_rate_usd_hour), now]
        );
        await client.query("UPDATE container_jobs SET status = 'running', updated_at = $2 WHERE id = $1", [row.id, now]);
        await client.query('COMMIT');
        return {
          runId,
          jobId: row.id,
          name: row.name,
          image: row.image,
          env: row.env,
          card: card.index,
          durationSec: Number(row.duration_sec),
          payRateUsdHour: Number(row.pay_rate_usd_hour),
        };
      }
      await client.query('COMMIT');
      return null;
    } catch (error) {
      await client.query('ROLLBACK');
      throw error;
    } finally {
      client.release();
    }
  }

  // The rig's run, or an error result when it isn't this rig's.
  async _ownRun(nodeId, runId) {
    const r = await this.db.query(
      `SELECT r.*, j.status AS job_status, j.duration_sec, j.pay_rate_usd_hour
         FROM container_runs r JOIN container_jobs j ON j.id = r.job_id WHERE r.id = $1`,
      [String(runId)]
    );
    if (!r.rows.length || r.rows[0].node_id !== nodeId) return { error: 'Run not found', status: 404 };
    return { run: r.rows[0] };
  }

  // A heartbeat while a run is up: `state` is 'pulling' or 'running', and
  // `logs` the lines since the last one. Answers 'stop' when the job was
  // deleted, its time is up, or the run was already released.
  async heartbeat(nodeId, runId, body) {
    const own = await this._ownRun(nodeId, runId);
    if (own.error) return own;
    const run = own.run;
    const now = this.now();
    if (!ACTIVE.includes(run.status)) return { action: 'stop', reason: 'run is ' + run.status };
    const running = body.state === 'running';
    const startedAt = run.started_at != null ? Number(run.started_at) : (running ? now : null);
    await this.db.query(
      `UPDATE container_runs SET heartbeat_at = $2, started_at = $3, status = $4 WHERE id = $1`,
      [run.id, now, startedAt, startedAt != null ? 'running' : 'claimed']
    );
    await this._appendLogs(run, body.logs, now);
    await this._recordMinutes(run, startedAt, now, false);
    const durationMs = Number(run.duration_sec) * 1000;
    if (run.job_status === 'stopping' || run.job_status === 'stopped') return { action: 'stop', reason: 'job deleted' };
    if (startedAt != null && now > startedAt + durationMs + OVERRUN_GRACE_MS) return { action: 'stop', reason: 'job time is up' };
    return { action: 'continue', endsAt: startedAt != null ? startedAt + durationMs : null };
  }

  // How a run ended, from the rig: `reason` (see END_REASONS), the container's
  // exit code if it exited, and its last log lines.
  async endRun(nodeId, runId, body) {
    const own = await this._ownRun(nodeId, runId);
    if (own.error) return own;
    const run = own.run;
    const now = this.now();
    if (!ACTIVE.includes(run.status)) return { success: true, status: run.status };
    const reason = END_REASONS.includes(body.reason) ? body.reason : 'failed';
    const exitCode = Number.isInteger(body.exitCode) ? body.exitCode : null;
    const detail = body.detail ? String(body.detail).slice(0, 500) : null;
    await this._appendLogs(run, body.logs, now);
    await this.db.query(
      `UPDATE container_runs SET status = $2, reason = $3, exit_code = $4, ended_at = $5, heartbeat_at = $5 WHERE id = $1`,
      [run.id, reason, detail || reason, exitCode, now]
    );
    await this._recordMinutes(run, run.started_at, now, true);
    await this._refreshJob(run.job_id);
    return { success: true, status: reason };
  }

  // Release runs whose rig has gone quiet. Their minutes stop at the last
  // heartbeat, and the slot opens for another rig. Called on every poll and
  // from the server's own timer. Returns the released run ids.
  async releaseStale() {
    const now = this.now();
    const r = await this.db.query(
      `SELECT r.*, j.duration_sec FROM container_runs r JOIN container_jobs j ON j.id = r.job_id
        WHERE r.status IN ('claimed', 'running') AND r.heartbeat_at < $1`,
      [now - RUN_TIMEOUT_MS]
    );
    const jobs = new Set();
    for (const run of r.rows) {
      await this.db.query(
        `UPDATE container_runs SET status = 'lost', reason = 'no heartbeat', ended_at = $2 WHERE id = $1`,
        [run.id, now]
      );
      await this._recordMinutes(run, run.started_at, Number(run.heartbeat_at), true);
      jobs.add(run.job_id);
    }
    for (const id of jobs) await this._refreshJob(id);
    return r.rows.map((x) => x.id);
  }

  // ── Internals ─────────────────────────────────────────────────────────────

  async _appendLogs(run, logs, now) {
    const lines = cleanLines(logs);
    const have = Number(run.log_lines) || 0;
    const room = Math.max(0, MAX_LOG_LINES - have);
    const keep = lines.slice(0, room);
    for (let i = 0; i < keep.length; i++) {
      await this.db.query(
        'INSERT INTO container_logs (run_id, seq, ts, line) VALUES ($1, $2, $3, $4)',
        [run.id, have + i, now, keep[i]]
      );
    }
    if (keep.length) {
      await this.db.query('UPDATE container_runs SET log_lines = $2 WHERE id = $1', [run.id, have + keep.length]);
      run.log_lines = have + keep.length;
    }
  }

  async _recordMinutes(run, startedAt, untilMs, final) {
    const minutes = paidMinutes(startedAt, untilMs, Number(run.duration_sec));
    const rate = Number(run.pay_rate_usd_hour) || 0;
    await this.db.query(
      `UPDATE container_minutes SET minutes = $2, amount_usd = $3, final = $4, updated_at = $5 WHERE run_id = $1`,
      [run.id, minutes, +(minutes / 60 * rate).toFixed(6), final, this.now()]
    );
  }

  async _activeCount(db, jobId) {
    const r = await db.query(
      `SELECT COUNT(*) AS n FROM container_runs WHERE job_id = $1 AND status IN ('claimed', 'running')`, [jobId]
    );
    return Number(r.rows[0].n);
  }

  // Bring a job's status in line with its runs once one has ended.
  async _refreshJob(jobId) {
    const r = await this.db.query('SELECT status, replicas FROM container_jobs WHERE id = $1', [jobId]);
    const job = r.rows[0];
    const runs = await this.db.query('SELECT status FROM container_runs WHERE job_id = $1', [jobId]);
    const active = runs.rows.filter((x) => ACTIVE.includes(x.status)).length;
    const done = runs.rows.filter((x) => COMPLETED.includes(x.status)).length;
    let status;
    if (job.status === 'stopping' || job.status === 'stopped') status = active ? 'stopping' : 'stopped';
    else if (done >= Number(job.replicas)) status = 'done';
    else status = active ? 'running' : 'pending';
    if (status !== job.status) {
      await this.db.query('UPDATE container_jobs SET status = $2, updated_at = $3 WHERE id = $1', [jobId, status, this.now()]);
    }
  }

  async _jobWithRuns(id) {
    const r = await this.db.query('SELECT * FROM container_jobs WHERE id = $1', [String(id)]);
    if (!r.rows.length) return null;
    const runs = await this.db.query(
      `SELECT r.*, m.minutes FROM container_runs r LEFT JOIN container_minutes m ON m.run_id = r.id
        WHERE r.job_id = $1 ORDER BY r.claimed_at ASC, r.id ASC`,
      [String(id)]
    );
    const job = jobView(r.rows[0]);
    job.runs = runs.rows.map((x) => runView(x, x.minutes));
    job.minutes = +job.runs.reduce((a, x) => a + x.minutes, 0).toFixed(3);
    return job;
  }
}

// Whether `nodeId` is the id this public key mints: sha256(publicKey) cut to
// the current 16 hex characters, or the 6 a rig enrolled under before ids were
// widened. verifySignature proves the caller holds the key; this ties the
// claimed node id to it, so nobody can take another rig's runs.
function nodeIdMatchesKey(nodeId, publicKey) {
  if (typeof nodeId !== 'string' || !NODE_ID_RE.test(nodeId) || !publicKey) return false;
  const full = crypto.createHash('sha256').update(String(publicKey)).digest('hex');
  return full.slice(0, nodeId.length) === nodeId;
}

ContainerService.imageProblem = imageProblem;
ContainerService.cleanEnv = cleanEnv;
ContainerService.cleanTarget = cleanTarget;
ContainerService.cleanCards = cleanCards;
ContainerService.cardFits = cardFits;
ContainerService.paidMinutes = paidMinutes;
ContainerService.nodeIdMatchesKey = nodeIdMatchesKey;
ContainerService.RUN_TIMEOUT_MS = RUN_TIMEOUT_MS;
ContainerService.OVERRUN_GRACE_MS = OVERRUN_GRACE_MS;
ContainerService.MAX_LOG_LINES = MAX_LOG_LINES;
ContainerService.POLL_MS = POLL_MS;

module.exports = ContainerService;
