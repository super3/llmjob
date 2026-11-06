// Usage, as Administrator on Windows or root on Linux:
//   node memclock-sweep.js <pearl_core.node> [clocks=0,5001] [rounds=3] [secs=90] [warm=20] [gpu=0]
//
// Times the miner at each memory clock in `clocks`, interleaved: every round runs
// each clock once, in an order that rotates from round to round, so drift in
// heat and power lands on all of them alike. 0 means the driver's own clock
// (nvidia-smi -rmc); any other value is locked with nvidia-smi -lmc, as the
// app's --mine-mem-clock does. Each leg is hashrate.js, the app's own number,
// with the clock, power and temperature it sampled. A leg whose memory clock
// did not read what was locked is marked, since a lock that does not take
// looks like a tie. The card's clock goes back to the driver at the end, on
// an error, and on Ctrl-C.
//
// Why: on a card mining at its power limit, memory clock is watts the SMs could
// use, and the fold barely touches DRAM. An RTX 5090 gained 8% at 7001 MHz
// (main/gpuClocks). Ada has not been measured: rig01's account cannot set
// clocks and rented containers cannot either (README.md here, "The memory
// clock on Ada"). An RTX 4090 offers 10501, 10251, 5001, 810 and 405 MHz
// (nvidia-smi -q -d SUPPORTED_CLOCKS); CUDA runs it at 10251.
//
// PEARL_SWEEP_VERIFY=1 also runs verify-hits.js for 40 s under the first
// non-zero clock before the clock is given back.
'use strict';
const path = require('path');
const { execFileSync, spawnSync } = require('child_process');

const core = process.argv[2];
if (!core) {
  console.error('usage: node memclock-sweep.js <pearl_core.node> [clocks=0,5001] [rounds=3] [secs=90] [warm=20] [gpu=0]');
  process.exit(2);
}
const clocks = String(process.argv[3] || '0,5001').split(',').map(Number);
const rounds = Number(process.argv[4] || 3);
const secs = Number(process.argv[5] || 90);
const warm = Number(process.argv[6] || 20);
const gpu = String(process.argv[7] || '0');
if (!clocks.every((c) => Number.isInteger(c) && c >= 0) || !(rounds >= 1)) {
  console.error('clocks must be whole MHz (0 for the driver\'s), rounds at least 1');
  process.exit(2);
}

function smi(args) {
  try {
    return { ok: true, out: execFileSync('nvidia-smi', args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }) };
  } catch (e) {
    return { ok: false, out: String(e.stdout || e.stderr || e.message).trim().split(/\r?\n/)[0] };
  }
}
function setClock(mhz) {
  return mhz === 0 ? smi(['-i', gpu, '-rmc']) : smi(['-i', gpu, '-lmc', mhz + ',' + mhz]);
}
function release() {
  const r = smi(['-i', gpu, '-rmc']);
  console.error(r.ok ? 'memory clock back to the driver' : 'could not reset the memory clock: ' + r.out);
}

// The rights check: a reset changes nothing and needs the same rights as a lock.
const probe = setClock(0);
if (!probe.ok) {
  if (clocks.some((c) => c > 0)) {
    console.error('cannot set clocks on GPU ' + gpu + ': ' + probe.out);
    console.error('run this as Administrator (Windows) or root (Linux)');
    process.exit(3);
  }
  console.error('no rights to set clocks; every clock here is 0, so this is an A/A run');
}
process.on('SIGINT', () => { release(); process.exit(130); });

const legs = [];
try {
  for (let r = 0; r < rounds; r++) {
    for (let i = 0; i < clocks.length; i++) {
      const mhz = clocks[(i + r) % clocks.length];
      const set = setClock(mhz);
      if (!set.ok && mhz > 0) throw new Error('lock at ' + mhz + ' MHz failed: ' + set.out);
      const run = spawnSync(process.execPath, [path.join(__dirname, 'hashrate.js'), core, String(secs), String(warm)],
        { encoding: 'utf8' });
      const line = (run.stdout || '').trim().split(/\r?\n/).pop();
      let res;
      try { res = JSON.parse(line); } catch (e) { throw new Error('hashrate.js said: ' + (line || run.stderr)); }
      // CUDA runs a 4090 at 10251 on the driver's clock, not the 10501 maximum,
      // so only a locked leg is checked against what it asked for.
      const took = mhz === 0 || Math.abs(res.memMhz - mhz) <= 10;
      const leg = { round: r + 1, memLock: mhz, th: res.th, sd: res.sd, mhz: res.mhz, watts: res.watts, tempC: res.tempC,
        memMhz: res.memMhz, memUtil: res.memUtil, ...(took ? {} : { lockDidNotTake: true }) };
      legs.push(leg);
      console.log(JSON.stringify(leg));
    }
  }
  if (process.env.PEARL_SWEEP_VERIFY === '1' && clocks.some((c) => c > 0)) {
    const mhz = clocks.find((c) => c > 0);
    const set = setClock(mhz);
    if (!set.ok) throw new Error('lock at ' + mhz + ' MHz failed: ' + set.out);
    const v = spawnSync(process.execPath, [path.join(__dirname, 'verify-hits.js'), core, '40'], { encoding: 'utf8' });
    console.log('verify-hits at ' + mhz + ' MHz: ' + (v.stdout || '').trim().split(/\r?\n/).pop());
  }
} catch (e) {
  console.error(String(e.message || e));
  process.exitCode = 1;
} finally {
  release();
}

// Per clock: the mean of its legs, and the gain against the driver's clock (0)
// when it was in the list. The spread of the 0 legs is this session's noise.
const mean = (xs) => xs.reduce((a, x) => a + x, 0) / xs.length;
const base = legs.filter((l) => l.memLock === 0 && !l.lockDidNotTake);
const baseTh = base.length ? mean(base.map((l) => l.th)) : null;
console.log('mem lock   legs   TH/s     SM MHz   W      C    mem MHz   vs driver');
for (const mhz of [...new Set(clocks)]) {
  const ls = legs.filter((l) => l.memLock === mhz && !l.lockDidNotTake);
  if (!ls.length) { console.log(String(mhz).padEnd(10) + ' none that took'); continue; }
  const th = mean(ls.map((l) => l.th));
  console.log([String(mhz || 'driver').padEnd(10), String(ls.length).padEnd(6), th.toFixed(2).padEnd(8),
    mean(ls.map((l) => l.mhz)).toFixed(0).padEnd(8), mean(ls.map((l) => l.watts)).toFixed(0).padEnd(6),
    mean(ls.map((l) => l.tempC)).toFixed(0).padEnd(4), mean(ls.map((l) => l.memMhz)).toFixed(0).padEnd(9),
    baseTh && mhz ? ((th / baseTh - 1) * 100).toFixed(2) + '%' : ''].join(' '));
}
if (base.length > 1) {
  const ths = base.map((l) => l.th);
  console.log('driver legs spread: ' + ((Math.max(...ths) / Math.min(...ths) - 1) * 100).toFixed(2) + '% (the noise to beat)');
}
