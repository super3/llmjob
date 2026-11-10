// Usage: node hashrate.js <path-to-pearl_core.node> [seconds=60] [warmup=10]
// Loads the addon, runs one core on a synthetic job with the HARDEST target
// (no hits), and averages the core's own hashrate samples after warmup. This is
// exactly the number the app shows: SearchLoop times fold + finalize + reseed.
//
// The core is opened as the app opens it (cardK): it may mine a larger k on its card than the
// mainnet profile's 2048 (an RDNA 4 card mines k 4096), and the JSON names the k it mined (core.k).
// PEARL_PROFILE_K=4096 pins the core to that k instead, without cardK, for a bench build made for it
// (the AMD core with -DPEARL_FOLD_K=4096). The rate is MACs a second at any k.
//
// PEARL_JOB_EVERY=secs gives the core a new job (another header) every secs seconds, as a pool does,
// so the rate includes what a job switch costs: the queued batches drained and the operands drawn
// again, both larger at a larger k. The JSON counts the jobs. Unset, the core keeps one job.
'use strict';
const path = require('path');
const { execSync } = require('child_process');
const file = path.resolve(process.argv[2]);
const secs = Number(process.argv[3] || 60), warm = Number(process.argv[4] || 10);
const { PROFILE: MAINNET } = require(path.join(__dirname, '..', '..', 'src', 'shared', 'miner', 'pearlhash'));
const kEnv = process.env.PEARL_PROFILE_K;
const PROFILE = kEnv ? { ...MAINNET, k: Number(kEnv) } : MAINNET;
// The sanity checks' least k is 16 x rank, and the operand trees need a power of two.
if (kEnv && !(Number.isInteger(PROFILE.k) && PROFILE.k >= 16 * PROFILE.rank && (PROFILE.k & (PROFILE.k - 1)) === 0)) {
  console.error(`PEARL_PROFILE_K must be a power of two of at least ${16 * PROFILE.rank}, not ${kEnv}`);
  process.exit(2);
}
const every = Number(process.env.PEARL_JOB_EVERY || 0);
if (!(every >= 0)) {
  console.error(`PEARL_JOB_EVERY must be a number of seconds, not ${process.env.PEARL_JOB_EVERY}`);
  process.exit(2);
}
const addon = require(file);
// PEARL_SHAPE_TRIAL=secs runs the core's startup trial of RTX 30's launch shapes, and
// PEARL_SHAPE=<name> ("flip 128", "band 8 x 512") opens the core in that shape. The trial's
// pick is printed as a "shape" line and the JSON names the shape the core ended in.
const shapeOpts = {};
if (process.env.PEARL_SHAPE_TRIAL) shapeOpts.shapeTrialSecs = Number(process.env.PEARL_SHAPE_TRIAL);
if (process.env.PEARL_SHAPE) shapeOpts.shape = process.env.PEARL_SHAPE;
const core = addon.createCore(PROFILE, kEnv ? { ...shapeOpts } : { cardK: true, ...shapeOpts });
core.on('shape', (s) => console.log('shape ' + s));
// The k the core mines: its own report, or the profile's from a core that has none.
const k = core.k === undefined ? PROFILE.k : core.k;
const t0 = Date.now(); const samples = []; const gpu = [];
core.on('hashrate', (th) => { if (Date.now() - t0 > warm * 1000) samples.push(th); });
core.on('error', (e) => { console.error('core error', e); process.exit(2); });
core.on('hit', () => {});
let jobs = 0;
const job = () => {
  const header = Buffer.alloc(76, 0xA5);
  header.writeUInt32LE(jobs, 0);   // each job a different header, so a different job_key
  core.setJob({ header, target: 0n, jobId: 'bench-' + jobs++ });
};
job();
const switcher = every > 0 ? setInterval(job, every * 1000) : null;
const poll = setInterval(() => {
  if (Date.now() - t0 < warm * 1000) return;
  try {
    // The memory controller's use, the memory in use and the memory clock too: on Ampere
    // the fold's DRAM traffic costs SM clock at the power limit (probes/turing/compilers.sh).
    const o = execSync('nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,utilization.memory,memory.used,clocks.mem --format=csv,noheader,nounits').toString().trim().split('\n')[0].split(',').map(Number);
    gpu.push(o);
  } catch (e) {}
}, 2000);
setTimeout(() => {
  clearInterval(poll); if (switcher) clearInterval(switcher); core.stop();
  const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
  const sd = Math.sqrt(samples.reduce((a, b) => a + (b - mean) ** 2, 0) / samples.length);
  const avg = (i) => {
    const xs = gpu.map((g) => g[i]).filter(Number.isFinite);   // a field the card reads as [N/A] is NaN
    return xs.length ? (xs.reduce((a, x) => a + x, 0) / xs.length).toFixed(0) : '?';
  };
  console.log(JSON.stringify({ file: path.basename(path.dirname(file)) + '/' + path.basename(file), k, ...(every > 0 ? { jobEvery: every, jobs } : {}), device: core.device, shape: core.shape, shapes: core.shapes, th: +mean.toFixed(2), sd: +sd.toFixed(2), n: samples.length, mhz: +avg(0), watts: +avg(1), tempC: +avg(2), memUtil: +avg(3), memUsedMb: +avg(4), memMhz: +avg(5) }));
  process.exit(0);
}, (secs + warm) * 1000);
