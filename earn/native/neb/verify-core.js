#!/usr/bin/env node
'use strict';

// Check a NEB core against the reference, on a card, and time it.
//
//   node earn/native/neb/verify-core.js --addon <file.node> [--device 0] [--big 1073741824] [--exact 2048]
//       [--triples 768 (rows for the triple table; 0 for none)] [--triple-gb 4]
//       [--dataset neb-datasets.bin (--model <file of packed weights, hex> | --pool host:port --address neb1…) [--slot 0]]
//   NEB_CORE=cpu node earn/native/neb/verify-core.js --big 4096 --exact 64
//
// It runs src/main/nebCore.js's verifyCore: the chain's vectors for the KEP1
// network, random keys in full, two exact ranges, the overflow path, and a
// large timed run whose every hit is checked. With --dataset the model checks
// run on the network's own dataset, on a model from --model or the first one
// the pool sends for --slot; without it, on nebCore.checkDataset and the
// model in vectors/syn2-trained.hex. Exit 0 when every check passes.

const fs = require('fs');
const path = require('path');
const { nebCoreFactory, verifyCore, checkDataset } = require('../../src/main/nebCore');
const nh = require('../../src/shared/neb/nebhash2');
const { formatHs } = require('../../src/shared/format');
const { loadDatasets, fetchModels } = require('./tools/pool');

function arg(name, fallback) {
  const i = process.argv.indexOf(name);
  return i > 0 && i + 1 < process.argv.length ? process.argv[i + 1] : fallback;
}

async function modelFor(slot) {
  const file = arg('--model', null);
  if (file) return Buffer.from(fs.readFileSync(file, 'utf8').trim(), 'hex');
  const pool = arg('--pool', null);
  if (!pool) throw new Error('--dataset needs --model or --pool and --address');
  const got = await fetchModels(pool, arg('--address', ''), [slot], 240000);
  return got[slot];
}

async function main() {
  const addon = arg('--addon', null);
  const env = Object.assign({}, process.env);
  if (addon) env.NEB_CORE_PATH = path.resolve(addon);
  const factory = nebCoreFactory({ env, log: (level, line) => console.log(level + ': ' + line) });
  if (!factory) return 1;
  const device = Number(arg('--device', factory.devices[0].index));
  const slot = Number(arg('--slot', 0));
  let ds;
  let packed;
  const dataset = arg('--dataset', null);
  if (dataset) {
    ds = loadDatasets(dataset).datasets[slot];
    packed = await modelFor(slot);
  } else {
    ds = checkDataset();
    packed = Buffer.from(fs.readFileSync(path.join(__dirname, 'vectors', 'syn2-trained.hex'), 'utf8').trim(), 'hex');
  }
  const weights = nh.modelFromPacked(ds.arch, packed).w;
  console.log('model: ' + nh.weightsHash(packed).toString('hex').slice(0, 16) + ' · ' + ds.n + ' samples');
  const core = factory.createCore(device);
  console.log('core: ' + factory.label + ' · card ' + device + ' · ' + JSON.stringify(core.device));
  try {
    const r = await verifyCore(core, {
      ds, weights, vectors: fs.readFileSync(path.join(__dirname, 'vectors', 'nebhash2.txt'), 'utf8'),
      bigCount: Number(arg('--big', 2 ** 30)), exactCount: Number(arg('--exact', 2048)), scanCount: 2 ** 28, fullCount: 64,
      tripleRows: Number(arg('--triples', 768)), tripleBytes: Number(arg('--triple-gb', 4)) * 2 ** 30,
    });
    for (const c of r.checks) console.log((c.ok ? 'PASS ' : 'FAIL ') + c.name.padEnd(9) + c.detail);
    console.log('table: ' + JSON.stringify(r.info));
    console.log((r.ok ? 'PASS' : 'FAIL') + ' · ' + formatHs(r.hashrate) + ' evaluated · ' + formatHs(r.screenRate) + ' screened');
    if (r.triples) {
      console.log('triples: ' + JSON.stringify(r.triples) + ' · ' + formatHs(r.tripleHashrate) + ' evaluated · '
        + formatHs(r.tripleScreenRate) + ' screened');
    }
    return r.ok ? 0 : 1;
  } finally {
    core.close();
  }
}

main().then((code) => { process.exitCode = code; }, (e) => {
  console.error('verify-core: ' + ((e && e.stack) || e));
  process.exitCode = 1;
});
