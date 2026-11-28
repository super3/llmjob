#!/usr/bin/env node
'use strict';

// Mine NEB on a pool with the engine alone, for a set time, and print what happened. For test boxes.
//
//   node earn/native/neb/tools/nebmine.js --address neb1… --dataset neb-datasets.bin [--pool neb.rabidmining.com:3470]
//       [--worker n1] [--secs 300] [--addon <file.node>] [--cards 0,1]
//
// A status line every 10 s: the rate (attempts evaluated a second), nonces looked at a second, shares.
// The last line is "end ..." with the run's means. Exit 0 when at least one share was accepted and none was bad.

const net = require('net');
const path = require('path');
const { NebEngine } = require('../../../src/main/nebEngine');
const { nebCoreFactory } = require('../../../src/main/nebCore');
const { formatHs } = require('../../../src/shared/format');
const { loadDatasets } = require('./pool');

function arg(name, fallback) {
  const i = process.argv.indexOf(name);
  return i > 0 && i + 1 < process.argv.length ? process.argv[i + 1] : fallback;
}

function main() {
  const env = Object.assign({}, process.env);
  const addon = arg('--addon', null);
  if (addon) env.NEB_CORE_PATH = path.resolve(addon);
  const say = (line) => console.log(new Date().toISOString().slice(11, 19) + ' ' + line);
  const factory = nebCoreFactory({ env, log: (level, line) => say(level + ': ' + line) });
  if (!factory) return Promise.resolve(1);
  const { datasets, hash } = loadDatasets(arg('--dataset', 'neb-datasets.bin'));
  say('dataset ' + hash.slice(0, 16) + ' · ' + datasets.map((d) => d.n).join(' and ') + ' samples');
  const engine = new NebEngine({ factory, connect: (host, port) => net.connect({ host, port }), agent: 'llmjob-earn/dev' });
  const cards = arg('--cards', null);
  const secs = Number(arg('--secs', 300));
  const last = new Map();
  let bad = 0;
  const samples = [];
  engine.on('log', (l) => {
    if (/not a hit/.test(l.line)) bad++;
    say(l.level + ': ' + l.line);
  });
  engine.on('event', (e) => { if (e.type === 'status') last.set(e.gpuIndex, e); });
  engine.on('error', () => {});
  return new Promise((resolve) => {
    const report = setInterval(() => {
      const all = [...last.values()];
      const sum = (k) => all.reduce((a, e) => a + (e[k] || 0), 0);
      if (sum('hashrate') > 0) samples.push([sum('hashrate'), sum('screened')]);
      say('status ' + formatHs(sum('hashrate')) + ' · ' + formatHs(sum('screened')) + ' screened · ' + sum('accepted')
        + ' accepted · ' + sum('rejected') + ' rejected');
    }, 10000);
    const finish = () => {
      clearInterval(report);
      const s = engine.snapshot();
      engine.stop();
      const mean = (i) => (samples.length ? samples.reduce((a, x) => a + x[i], 0) / samples.length : 0);
      say('end ' + formatHs(mean(0)) + ' mean · ' + formatHs(mean(1)) + ' screened · ' + s.accepted + ' accepted · '
        + s.rejected + ' rejected · ' + bad + ' bad hits · ' + secs + ' s');
      resolve(s.accepted > 0 && bad === 0 ? 0 : 1);
    };
    engine.on('stopped', (code) => { if (code) finish(); });
    const ok = engine.start({
      address: arg('--address', ''), worker: arg('--worker', 'n1'), endpoint: arg('--pool', 'neb.rabidmining.com:3470'),
      cards: cards ? cards.split(',').map(Number) : null, datasets, datasetHash: hash,
    });
    if (!ok) {
      clearInterval(report);
      resolve(1);
      return;
    }
    setTimeout(finish, secs * 1000);
  });
}

main().then((code) => { process.exit(code); }, (e) => {
  console.error('nebmine: ' + ((e && e.stack) || e));
  process.exit(1);
});
