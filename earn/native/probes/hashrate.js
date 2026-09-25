// Usage: node hashrate.js <path-to-pearl_core.node> [seconds=60] [warmup=10]
// Loads the addon, runs one core on a synthetic job with the HARDEST target
// (no hits), and averages the core's own hashrate samples after warmup. This is
// exactly the number the app shows: SearchLoop times fold + finalize + reseed.
'use strict';
const path = require('path');
const { execSync } = require('child_process');
const file = path.resolve(process.argv[2]);
const secs = Number(process.argv[3] || 60), warm = Number(process.argv[4] || 10);
const { PROFILE } = require(path.join(__dirname, '..', '..', 'src', 'shared', 'miner', 'pearlhash'));
const addon = require(file);
const core = addon.createCore(PROFILE, {});
const t0 = Date.now(); const samples = []; const gpu = [];
core.on('hashrate', (th) => { if (Date.now() - t0 > warm * 1000) samples.push(th); });
core.on('error', (e) => { console.error('core error', e); process.exit(2); });
core.on('hit', () => {});
core.setJob({ header: Buffer.alloc(76, 0xA5), target: 0n, jobId: 'bench' });
const poll = setInterval(() => {
  if (Date.now() - t0 < warm * 1000) return;
  try {
    const o = execSync('nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu --format=csv,noheader,nounits').toString().trim().split(',').map(Number);
    gpu.push(o);
  } catch (e) {}
}, 2000);
setTimeout(() => {
  clearInterval(poll); core.stop();
  const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
  const sd = Math.sqrt(samples.reduce((a, b) => a + (b - mean) ** 2, 0) / samples.length);
  const avg = (i) => gpu.length ? (gpu.reduce((a, g) => a + g[i], 0) / gpu.length).toFixed(0) : '?';
  console.log(JSON.stringify({ file: path.basename(path.dirname(file)) + '/' + path.basename(file), device: core.device, th: +mean.toFixed(2), sd: +sd.toFixed(2), n: samples.length, mhz: +avg(0), watts: +avg(1), tempC: +avg(2) }));
  process.exit(0);
}, (secs + warm) * 1000);
