#!/usr/bin/env node
'use strict';

// Time a Quantus core the way the miner drives it: back-to-back scans of
// --count nonces at a target nothing meets, for --secs seconds. Prints each
// second's rate and the mean of all scans after the first (which may include
// the driver's compile of the kernel).
//
//   node earn/native/quantus/sycl/bench.js --addon build/quantus_sycl.node [--device 0] [--secs 20] [--count 16777216]

const path = require('path');
const crypto = require('crypto');
const q = require('../../../src/shared/quantus/qpow');

function arg(name, fallback) {
  const i = process.argv.indexOf(name);
  return i > 0 && i + 1 < process.argv.length ? process.argv[i + 1] : fallback;
}

async function main() {
  const addon = require(path.resolve(arg('--addon', path.join(__dirname, '..', 'build', 'quantus_sycl.node'))));
  const device = Number(arg('--device', 0));
  const secs = Number(arg('--secs', 20));
  const count = Number(arg('--count', 2 ** 24));
  const core = addon.createCore({ device });
  console.log('core: ' + JSON.stringify(core.device) + ' · ' + count + ' nonces a scan');
  const work = {
    midstate: q.midstateBytes(q.midstate(crypto.randomBytes(32), crypto.randomBytes(32))),
    tail: crypto.randomBytes(24), start: Buffer.alloc(8), count, targetHigh: Buffer.alloc(8), maxHits: 16,
  };
  let counter = 0n;
  const t0 = Date.now();
  let n = 0, done = 0, first = true, tick = t0, tickDone = 0, allMs = 0;
  try {
    while (Date.now() - t0 < secs * 1000) {
      work.start.writeBigUInt64BE(counter);
      const s = Date.now();
      await core.scan(work);
      const ms = Date.now() - s;
      counter += BigInt(count);
      if (first) {
        console.log('first scan ' + ms + ' ms');
        first = false;
      } else {
        n += 1;
        done += count;
        allMs += ms;
        tickDone += count;
      }
      if (Date.now() - tick >= 1000) {
        const r = tickDone / ((Date.now() - tick) / 1000);
        console.log('t=' + ((Date.now() - t0) / 1000).toFixed(0) + 's ' + (r / 1e6).toFixed(2) + ' MH/s');
        tick = Date.now();
        tickDone = 0;
      }
    }
  } finally {
    core.close();
  }
  const mean = done / (allMs / 1000);
  console.log('mean ' + (mean / 1e6).toFixed(2) + ' MH/s over ' + n + ' scans (' + (allMs / n).toFixed(1) + ' ms a scan)');
}

main().catch((e) => {
  console.error('bench: ' + ((e && e.stack) || e));
  process.exitCode = 1;
});
