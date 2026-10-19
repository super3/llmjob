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
// A benchmark is not free for the node: on a card that also mines, the app
// pauses mining while any prompt runs, this one included. So a node gets one
// when it first shows up, then at most one a day (BENCH_INTERVAL_MS), whatever
// became of the last one. That last part matters. A node whose benchmark failed
// (its GPU had dropped out from under llama-server) recorded no speed, so every
// 5-minute sweep sent it another, and its miner paused every 5 minutes for a
// test that could never pass.
//
// It rides entirely on machinery that already exists: a benchmark is an ordinary
// job pinned to one node with `target_node`, which is why none of this needs a
// client release. The prompt and the budget below can be changed in a deploy.

// Fixed work, so two nodes' numbers are comparable and a change in the fleet's
// scores means the fleet changed. Asks for prose rather than a fact, so the
// answer's length doesn't depend on what the model happens to know.
const BENCH_PROMPT =
  'Write a single paragraph of about 120 words explaining what a GPU does, in plain language.';
// Enough generation for a stable rate, and sized against the work we actually
// measure against. End-to-end tok/s divides by wall time that includes prefill
// and pickup, so a short reply amortizes that fixed cost over few tokens and
// reads slower than the same node serving a real request — which biases nodes
// toward the "too slow for a full job" side of the line for no reason. At 256
// that bias was large; 1024 puts it within a few percent of a typical 2.5k-token
// request. Still small enough that the slowest node in the fleet (~18 tok/s)
// finishes in under a minute, so the benchmark never becomes the timeout it
// exists to prevent.
const BENCH_MAX_TOKENS = 1024;
// Benchmarks run at negative priority so they always queue behind real traffic —
// measuring the fleet must never delay serving it.
const BENCH_PRIORITY = -1;
// The most often one node is benchmarked: once a day. The same span as a
// measurement stays trusted, so a node is re-measured as its speed goes stale.
// It is also the window in which an earlier benchmark, passed or not, counts —
// which relies on jobService.cleanupOldJobs keeping finished jobs at least this
// long (it keeps them a day).
const BENCH_INTERVAL_MS = NodeService.SPEED_STALE_MS;
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

  // Nodes online right now whose speed we can't vouch for: never measured, or
  // measured longer ago than BENCH_INTERVAL_MS.
  //
  // A node held back by admission control waits for the same daily re-measure
  // as any other. It used to be re-measured every 30 minutes, since a gated node
  // gets no jobs and so never earns a sample on its own. A day is a long time
  // for a card to sit out, but the network carries little traffic, and each
  // re-measure paused that node's miner.
  //
  // Ordered oldest-first so the least-known node is served by the sweep cap
  // rather than whichever row Postgres returned first.
  async _needBenchmark() {
    const now = this.now();
    const r = await this.db.query(
      `SELECT node_id, speed_at, speed_samples FROM nodes
       WHERE last_seen >= $1
         AND (speed_at IS NULL OR speed_at < $2)
       ORDER BY speed_at ASC NULLS FIRST`,
      [now - 15 * 60 * 1000, now - BENCH_INTERVAL_MS]
    );
    return r.rows;
  }

  // Nodes that had a benchmark inside the interval, whatever became of it:
  // still queued or running, passed, failed, or expired with nobody picking it
  // up. A pass already shows in speed_at. The others leave speed_at as it was,
  // which is why this looks at the jobs themselves: without it a node whose
  // benchmark failed was due again on the very next sweep.
  //
  // By the benchmark flag, not just target_node, since a caller can pin its own
  // request to a node (X-LLMJob-Node) and that is not a benchmark.
  async _recentlyBenchmarked() {
    const r = await this.db.query(
      `SELECT DISTINCT target_node FROM jobs
       WHERE target_node IS NOT NULL AND created_at >= $1 AND data->>'benchmark' = 'true'`,
      [this.now() - BENCH_INTERVAL_MS]
    );
    return new Set(r.rows.map((row) => row.target_node));
  }

  // One pass: queue a benchmark for each unmeasured online node. Returns the node
  // ids benchmarked, so the caller can log what it did.
  async sweep() {
    const due = await this._needBenchmark();
    if (!due.length) return [];
    const recent = await this._recentlyBenchmarked();

    const queued = [];
    for (const row of due) {
      if (queued.length >= this.maxPerSweep) break;
      if (recent.has(row.node_id)) continue;
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
BenchmarkService.BENCH_INTERVAL_MS = BENCH_INTERVAL_MS;

module.exports = BenchmarkService;
