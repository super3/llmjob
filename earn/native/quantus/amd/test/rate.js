#!/usr/bin/env node
'use strict';

// Time a Quantus core on one card: back-to-back scans at a target nothing meets, as the miner runs it.
//
//   node rate.js <addon.node> [--device 0] [--secs 30] [--warm 5] [--count 268435456]
//
// Prints one JSON line a scan ({ t, mhs }) and, at the end, the mean and spread of the scans after the
// warm-up: { mhs, sd, scans }. Timing only: verify-core.js is what says a core is right.

const path = require('path');
const crypto = require('crypto');

function arg(name, fallback) {
  const i = process.argv.indexOf(name);
  return i > 0 && i + 1 < process.argv.length ? process.argv[i + 1] : fallback;
}

async function main() {
  const addon = require(path.resolve(process.argv[2]));
  const device = Number(arg('--device', 0));
  const secs = Number(arg('--secs', 30));
  const warm = Number(arg('--warm', 5));
  const count = Number(arg('--count', 2 ** 28));
  const core = addon.createCore({ device });
  const mid = Buffer.alloc(96);
  for (let i = 0; i < 12; i++) mid.writeBigUInt64LE(crypto.randomBytes(8).readBigUInt64LE(0) % 0xffffffff00000001n, i * 8);
  const work = { midstate: mid, tail: crypto.randomBytes(24), start: Buffer.alloc(8), count,
    targetHigh: Buffer.alloc(8), maxHits: 16 };
  const t0 = Date.now();
  const rates = [];
  let start = 0n;
  try {
    while ((Date.now() - t0) / 1000 < warm + secs) {
      work.start = Buffer.alloc(8);
      work.start.writeBigUInt64BE(start);
      start += BigInt(count);
      const a = process.hrtime.bigint();
      await core.scan(work);
      const s = Number(process.hrtime.bigint() - a) / 1e9;
      const t = (Date.now() - t0) / 1000;
      const mhs = count / s / 1e6;
      console.log(JSON.stringify({ t: +t.toFixed(1), mhs: +mhs.toFixed(3) }));
      if (t > warm) rates.push(mhs);
    }
  } finally {
    core.close();
  }
  const mean = rates.reduce((a, b) => a + b, 0) / Math.max(rates.length, 1);
  const sd = Math.sqrt(rates.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(rates.length - 1, 1));
  console.log(JSON.stringify({ device: core.device, mhs: +mean.toFixed(3), sd: +sd.toFixed(3), scans: rates.length }));
}

main().catch((e) => { console.error(e && e.stack || e); process.exitCode = 1; });
