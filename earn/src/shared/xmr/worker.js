'use strict';

// The line protocol between the app and the RandomX worker
// (earn/xmr-worker/src/xmr_worker.cpp): the commands the app writes to the
// worker's stdin, and the events it reads from its stdout, one JSON object per
// line each way. Pure; main/xmrEngine.js owns the process.

const { randomInt } = require('crypto');
const { workerTarget } = require('./stratum');

// The protocol version the worker states in its hello. A worker that states
// another one is from a different release and is not used.
const PROTOCOL = 1;

const PRIORITIES = ['idle', 'low', 'normal'];

// The worker's priority when nobody picks one. Either way at least one CPU is
// left free (see cpuPlan), and that is what keeps the GPU miner whole: on a
// rented Ryzen 9 5950X its Pearl rate was the same at all three priorities.
// What the priority decides is how the CPU miner fares against everything else
// on the computer.
//
// Linux gets "low" (nice 19). "idle" (SCHED_IDLE) gave up nearly everything
// there: on that host, with other tenants keeping all 32 CPUs busy, 14 threads
// made 1.74 kH/s at idle, 3.97 at low and 6.47 at normal. A SCHED_IDLE thread
// weighs almost nothing in its cgroup's share of a CPU once the GPU miner's
// normal threads are in the same cgroup. Nice 19 still lets any normal program
// have about 98% of a CPU it wants.
//
// Windows keeps "idle" (the idle priority class): its scheduler runs strictly
// by priority, so idle threads take every CPU nothing else wants, and step
// aside at once for anything that does.
function defaultPriority(platform) {
  return platform === 'win32' ? 'idle' : 'low';
}

function encode(msg) {
  return JSON.stringify(msg) + '\n';
}

// The worker's command line.
function workerArgs({ threads, priority, hugePages } = {}) {
  const args = ['--threads', String(Math.max(1, Math.floor(Number(threads) || 1)))];
  if (priority && PRIORITIES.includes(priority)) args.push('--priority', priority);
  if (hugePages === false) args.push('--no-huge-pages');
  return args;
}

// Where the worker's threads start on the job's 32-bit nonce. The worker
// searches from there in a fixed order, so a job sent twice from the same
// start (to a restarted worker, say) finds the same shares twice. A pool counts
// a repeated share as invalid, and cryptonote-nodejs-pool, which HeroMiners
// runs, can ban the IP for one. A random start makes a repeat unlikely; the
// engine also drops any nonce it already sent.
function randomNonce() {
  return randomInt(0, 2 ** 32);
}

// A parsed pool job (stratum.parseJob) as a worker command.
function jobCommand(job, nonce = randomNonce()) {
  return encode({
    cmd: 'job',
    id: job.jobId,
    blob: job.blob,
    target: workerTarget(job.target),
    seed: job.seedHash,
    variant: job.variant,
    commit: !!job.commit,
    nonce,
  });
}

function quitCommand() {
  return encode({ cmd: 'quit' });
}

const HEX = (s, chars) => typeof s === 'string' && s.length === chars && /^[0-9a-f]+$/.test(s);
const num = (v) => (Number.isFinite(v) ? v : null);

// One line from the worker, as { kind, ... }. Never throws.
function parseWorkerLine(line) {
  const s = String(line == null ? '' : line).trim();
  if (!s) return { kind: 'empty' };
  let m;
  try {
    m = JSON.parse(s);
  } catch (e) {
    return { kind: 'unparseable', raw: s };
  }
  if (!m || typeof m !== 'object' || Array.isArray(m)) return { kind: 'unparseable', raw: s };
  switch (m.type) {
    case 'hello':
      return {
        kind: 'hello',
        protocol: num(m.protocol),
        version: String(m.version || ''),
        randomx: String(m.randomx || ''),
        cpus: num(m.cpus),
        threads: num(m.threads),
        jit: !!m.jit,
        hardAes: !!m.hardAes,
      };
    case 'selftest':
      return { kind: 'selftest', ok: m.ok === true, ms: num(m.ms), message: m.msg ? String(m.msg) : null };
    case 'dataset':
      if (m.state !== 'building' && m.state !== 'ready') break;
      return {
        kind: 'dataset',
        state: m.state,
        ms: num(m.ms),
        threads: num(m.threads),
        pages: m.pages ? String(m.pages) : null,
      };
    case 'share':
      if (typeof m.job !== 'string' || !HEX(m.nonce, 8) || !HEX(m.result, 64)) break;
      return {
        kind: 'share', jobId: m.job, nonce: m.nonce, result: m.result,
        commitment: HEX(m.commitment, 64) ? m.commitment : null,
      };
    case 'hashrate':
      return { kind: 'hashrate', hs: Math.max(0, num(m.hs) || 0), total: num(m.total), threads: num(m.threads) };
    case 'log':
      return { kind: 'log', level: m.level === 'warn' || m.level === 'error' ? m.level : 'info', message: String(m.msg || '') };
    case 'error':
      return { kind: 'error', fatal: m.fatal === true, message: String(m.msg || 'unknown error') };
    case 'bench':
      return { kind: 'bench', hs: num(m.hs), seconds: num(m.seconds) };
  }
  return { kind: 'unknown', raw: s };
}

module.exports = {
  PROTOCOL, PRIORITIES, defaultPriority, workerArgs, randomNonce, jobCommand, quitCommand, parseWorkerLine,
};
