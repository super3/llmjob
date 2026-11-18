// Usage: node hashrate.js <path-to-pearl_core.node> [seconds=60] [warmup=10] [--device N]
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
//
// --device N (or PEARL_DEVICE=N) opens card N. Without it the core picks a card. Either way the
// clock, power and memory figures are nvidia-smi's for the card the core opened (core.device.index),
// and CUDA numbers the cards as nvidia-smi does (CUDA_DEVICE_ORDER=PCI_BUS_ID, as the app sets it), so
// on a box with several cards the watts are the mining card's.
'use strict';
const path = require('path');
const { execSync } = require('child_process');
const { askedProfile, coreOptions, deviceOption } = require('./lib/openCore');
const dev = deviceOption(process.argv.slice(2));
if (dev.error) { console.error(dev.error); process.exit(2); }
const file = path.resolve(dev.args[0]);
const secs = Number(dev.args[1] || 60), warm = Number(dev.args[2] || 10);
const { PROFILE: MAINNET } = require(path.join(__dirname, '..', '..', 'src', 'shared', 'miner', 'pearlhash'));
const asked = askedProfile(MAINNET);
if (asked.error) { console.error(asked.error); process.exit(2); }
const { kEnv, profile: PROFILE } = asked;
const every = Number(process.env.PEARL_JOB_EVERY || 0);
if (!(every >= 0)) {
  console.error(`PEARL_JOB_EVERY must be a number of seconds, not ${process.env.PEARL_JOB_EVERY}`);
  process.exit(2);
}
if (!process.env.CUDA_DEVICE_ORDER) process.env.CUDA_DEVICE_ORDER = 'PCI_BUS_ID';
const addon = require(file);
// PEARL_SHAPE_TRIAL and PEARL_SHAPE (lib/openCore.js) pick the launch shape. The trial's pick is
// printed as a "shape" line and the JSON names the shape the core ended in.
const core = addon.createCore(PROFILE, coreOptions(kEnv, dev.device === null ? {} : { deviceIndex: dev.device }));
// The card nvidia-smi reads: the one the core opened, or the one asked for.
const smiIndex = core.device && Number.isInteger(core.device.index) ? core.device.index : dev.device;
const smiCard = smiIndex === null ? '' : ' -i ' + smiIndex;
let trial = null;
core.on('shape', (s) => { trial = s; console.log('shape ' + s); });
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
    const o = execSync('nvidia-smi' + smiCard + ' --query-gpu=clocks.sm,power.draw,temperature.gpu,utilization.memory,memory.used,clocks.mem --format=csv,noheader,nounits').toString().trim().split('\n')[0].split(',').map(Number);
    gpu.push(o);
  } catch (e) {}
}, 2000);
setTimeout(() => {
  clearInterval(poll); if (switcher) clearInterval(switcher); const shape = core.shape; core.stop();
  const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
  const sd = Math.sqrt(samples.reduce((a, b) => a + (b - mean) ** 2, 0) / samples.length);
  const avg = (i) => {
    const xs = gpu.map((g) => g[i]).filter(Number.isFinite);   // a field the card reads as [N/A] is NaN
    return xs.length ? (xs.reduce((a, x) => a + x, 0) / xs.length).toFixed(0) : '?';
  };
  console.log(JSON.stringify({ file: path.basename(path.dirname(file)) + '/' + path.basename(file), k, ...(every > 0 ? { jobEvery: every, jobs } : {}), device: core.device, shape, ...(trial ? { trial: JSON.parse(trial) } : {}), th: +mean.toFixed(2), sd: +sd.toFixed(2), n: samples.length, mhz: +avg(0), watts: +avg(1), tempC: +avg(2), memUtil: +avg(3), memUsedMb: +avg(4), memMhz: +avg(5) }));
  process.exit(0);
}, (secs + warm) * 1000);
