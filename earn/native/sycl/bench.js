// node bench.js <core.node> [--seconds 60] [--warmup 10] [--m M] [--n N] [--col-batch C]
//
// The full miner loop's rate: the addon's own search thread with its two-batch
// pipeline and operand redraws, read from the 'hashrate' events the app shows
// (TH/s = MACs a second / 1e12). The target is zero, so nothing hits and only
// the search is timed. Default profile: mainnet (PROFILE). Prints one JSON line.
'use strict';
const path = require('path');
const { PROFILE } = require(path.join(__dirname, '..', '..', 'src', 'shared', 'miner', 'pearlhash'));

const o = { file: null, seconds: 60, warmup: 10, m: PROFILE.m, n: PROFILE.n, colBatch: PROFILE.colBatch };
const argv = process.argv.slice(2);
for (let i = 0; i < argv.length; i++) {
  const a = argv[i];
  if (a === '--seconds') o.seconds = Number(argv[++i]);
  else if (a === '--warmup') o.warmup = Number(argv[++i]);
  else if (a === '--m') o.m = Number(argv[++i]);
  else if (a === '--n') o.n = Number(argv[++i]);
  else if (a === '--col-batch') o.colBatch = Number(argv[++i]);
  else if (!o.file) o.file = a;
  else { console.error('unknown argument ' + a); process.exit(2); }
}
if (!o.file) { console.error('usage: node bench.js <core.node> [--seconds S] [--warmup W] [--m M] [--n N] [--col-batch C]'); process.exit(2); }

const addon = require(path.resolve(o.file));
const core = addon.createCore(Object.assign({}, PROFILE, { m: o.m, n: o.n, colBatch: o.colBatch }), {});
const t0 = Date.now();
const samples = [];
core.on('hashrate', (r) => { if (Date.now() - t0 >= o.warmup * 1000) samples.push(r); });
core.on('hit', () => {});
core.on('error', (e) => { console.log(JSON.stringify({ error: String(e) })); process.exit(2); });
core.setJob({ header: Buffer.alloc(76, 0x5A), target: 0n, jobId: 'bench' });
setTimeout(() => {
  core.stop();
  const s = samples.slice().sort((a, b) => a - b);
  const mean = s.length ? s.reduce((x, y) => x + y, 0) / s.length : 0;
  const round = (x) => Math.round(x * 1000) / 1000;
  console.log(JSON.stringify({
    device: core.device && core.device.name, fold: process.env.PEARL_SYCL_FOLD || 'auto',
    m: o.m, n: o.n, colBatch: o.colBatch, seconds: o.seconds, warmup: o.warmup, samples: s.length,
    meanThs: round(mean), medianThs: round(s.length ? s[s.length >> 1] : 0),
    minThs: round(s.length ? s[0] : 0), maxThs: round(s.length ? s[s.length - 1] : 0),
  }));
  process.exit(s.length ? 0 : 1);
}, (o.warmup + o.seconds) * 1000);
