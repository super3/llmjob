const JobService = require('./jobService');
const NodeService = require('./nodeService');

// Measure how fast each node generates, so assignment can stop handing a slow
// card work it can't finish (see jobService.assignJobsToNode).
//
// Most nodes never need this: every completed job is already a measurement, taken
// under the conditions the node really serves under. The benchmark exists for the
// two cases passive measurement can't cover — a node we have never seen produce
// anything, and a node whose last sample is old enough that it may describe
// hardware that isn't there any more. Both would otherwise sit unmeasured
// forever, since an unmeasured node is served permissively and a gated one would
// never earn a fresh sample on its own.
//
// It rides entirely on machinery that already exists: a benchmark is an ordinary
// job pinned to one node with `target_node`, which is why none of this needs a
// client release. The prompt and the budget below can be changed in a deploy.

// Fixed work, so two nodes' numbers are comparable and a change in the fleet's
// scores means the fleet changed. Asks for prose rather than a fact, so the
// answer's length doesn't depend on what the model happens to know.
const BENCH_PROMPT =
  'Write a single paragraph of about 120 words explaining what a GPU does, in plain language.';
// Enough generation for a stable rate (a very short reply is mostly prefill and
// measures the wrong thing), small enough that even a slow node finishes well
// inside the gateway budget and the benchmark never becomes the timeout it exists
// to prevent.
const BENCH_MAX_TOKENS = 256;
// Benchmarks run at negative priority so they always queue behind real traffic —
// measuring the fleet must never delay serving it.
const BENCH_PRIORITY = -1;
// Give a benchmark this long to come back before we conclude it isn't going to and
// allow another. Comfortably over the gateway budget, so a slow-but-working node
// isn't re-benchmarked while its first one is still running.
const BENCH_TTL_MS = 6 * 60 * 1000;
// Cap per sweep so a fleet that all goes stale at once (a redeploy, a long quiet
// weekend) doesn't flood the queue with benchmarks ahead of real work.
const MAX_PER_SWEEP = 5;

class BenchmarkService {
  constructor(db, opts = {}) {
    this.db = db;
    this.jobs = opts.jobs || new JobService(db);
    this.nodes = opts.nodes || new NodeService(db);
    this.now = opts.now || Date.now;
    this.maxPerSweep = opts.maxPerSweep || MAX_PER_SWEEP;
  }

  // Nodes online right now whose speed we don't know: never measured, or measured
  // too long ago to trust. Ordered oldest-first so the least-known node is served
  // by the sweep cap rather than whichever row Postgres returned first.
  async _needBenchmark() {
    const now = this.now();
    const r = await this.db.query(
      `SELECT node_id, speed_at, speed_samples FROM nodes
       WHERE last_seen >= $1 AND (speed_at IS NULL OR speed_at < $2)
       ORDER BY speed_at ASC NULLS FIRST`,
      [now - 15 * 60 * 1000, now - NodeService.SPEED_STALE_MS]
    );
    return r.rows;
  }

  // Benchmarks already in flight for these nodes, so a sweep doesn't queue a
  // second one behind the first — the sweeper runs far more often than a
  // benchmark takes to come back.
  async _inFlight() {
    const r = await this.db.query(
      `SELECT target_node FROM jobs
       WHERE status IN ('pending', 'assigned', 'running') AND target_node IS NOT NULL AND created_at >= $1`,
      [this.now() - BENCH_TTL_MS]
    );
    return new Set(r.rows.map((row) => row.target_node));
  }

  // One pass: queue a benchmark for each unmeasured online node. Returns the node
  // ids benchmarked, so the caller can log what it did.
  async sweep() {
    const due = await this._needBenchmark();
    if (!due.length) return [];
    const busy = await this._inFlight();

    const queued = [];
    for (const row of due) {
      if (queued.length >= this.maxPerSweep) break;
      if (busy.has(row.node_id)) continue;
      await this.jobs.createJob({
        prompt: BENCH_PROMPT,
        messages: [{ role: 'user', content: BENCH_PROMPT }],
        maxTokens: BENCH_MAX_TOKENS,
        priority: BENCH_PRIORITY,
        targetNode: row.node_id,
        visibility: 'public',
        benchmark: true,
        // First measurement of this node: its result replaces rather than blends,
        // because a cold llama-server's first generation includes model load and
        // KV warm-up and reads far slower than the node's steady state.
        benchmarkWarmup: !Number(row.speed_samples),
      });
      queued.push(row.node_id);
    }
    return queued;
  }
}

BenchmarkService.BENCH_PROMPT = BENCH_PROMPT;
BenchmarkService.BENCH_MAX_TOKENS = BENCH_MAX_TOKENS;
BenchmarkService.BENCH_TTL_MS = BENCH_TTL_MS;

module.exports = BenchmarkService;
