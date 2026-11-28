#!/usr/bin/env node
'use strict';

// Time a NEB core on two recorded pool models, with no pool: nonces screened and attempts evaluated a second,
// and how long the tables take to build. For comparing builds of the core on one card.
//
//   node earn/native/neb/tools/nebbench.js --addon <file.node> --dataset neb-datasets.bin [--label name]
//       [--secs 4] [--triples 2000 (rows asked of the triple table; 0 for none)] [--triple-gb 20] [--device 0]
//
// The models are vectors/model-kepler-14038.hex and model-tess-14037.hex: what neb.rabidmining.com sent for
// blocks 14038 (the Kepler dataset) and 14037 (TESS) on 28 Nov 2026. One line each:
//   bench <label> kepler: 18.90 G/s screened · 796.1 MH/s evaluated (4.14%) · model 1368 ms
// and with --triples a second line each, after the triple table:
//   bench <label> tess+3: ... · triples 1602 rows, 21968 MB in 2100 ms (326 M entries/s)

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { nebCoreFactory } = require('../../../src/main/nebCore');
const nh = require('../../../src/shared/neb/nebhash2');
const { loadDatasets } = require('./pool');

function arg(name, fallback) {
  const i = process.argv.indexOf(name);
  return i > 0 && i + 1 < process.argv.length ? process.argv[i + 1] : fallback;
}

const MODELS = [['kepler', 0, 'model-kepler-14038.hex'], ['tess', 1, 'model-tess-14037.hex']];
const COUNT = 2 ** 30;
const NEVER = Buffer.alloc(32);

async function rate(core, slot, secs) {
  const header = crypto.randomBytes(80);
  // One scan to warm up, then timed ones.
  await core.scan({ slot, header, target: NEVER, start: 0, count: 2 ** 26, maxHits: 1 });
  const t0 = process.hrtime.bigint();
  let screened = 0;
  let evaluated = 0;
  let took = 0;
  for (let i = 0; took < secs; i++) {
    const res = await core.scan({ slot, header, target: NEVER, start: (i * 2 ** 30) % 2 ** 32, count: COUNT, maxHits: 1 });
    screened += COUNT;
    evaluated += res.evaluated;
    took = Number(process.hrtime.bigint() - t0) / 1e9;
  }
  return { screened: screened / took, evaluated: evaluated / took, share: evaluated / screened };
}

const show = (r) => (r.screened / 1e9).toFixed(2) + ' G/s screened · ' + (r.evaluated / 1e6).toFixed(1) + ' MH/s evaluated ('
  + (r.share * 100).toFixed(2) + '%)';

async function main() {
  const env = Object.assign({}, process.env);
  const addon = arg('--addon', null);
  if (addon) env.NEB_CORE_PATH = path.resolve(addon);
  const factory = nebCoreFactory({ env, log: (level, line) => { if (level !== 'info') console.log(level + ': ' + line); } });
  if (!factory) return 1;
  const label = arg('--label', factory.label);
  const secs = Number(arg('--secs', 4));
  const rows = Number(arg('--triples', 0));
  const { datasets } = loadDatasets(arg('--dataset', 'neb-datasets.bin'));
  const core = factory.createCore(Number(arg('--device', factory.devices[0].index)));
  try {
    datasets.forEach((ds, slot) => core.setDataset(slot, ds.arch, ds.values, ds.labels));
    for (const [name, slot, file] of MODELS) {
      const packed = Buffer.from(fs.readFileSync(path.join(__dirname, '..', 'vectors', file), 'utf8').trim(), 'hex');
      const w = nh.modelFromPacked(datasets[slot].arch, packed).w;
      const weights = Buffer.from(w.buffer, w.byteOffset, w.length);
      const info = await core.setModel(slot, weights);
      // A second load is what a miner sees from the second block on: buffers and code are warm.
      const again = await core.setModel(slot, weights);
      console.log('bench ' + label + ' ' + name + ': ' + show(await rate(core, slot, secs)) + ' · model ' + Math.round(info.cpuMs + info.gpuMs)
        + ' ms, then ' + Math.round(again.cpuMs) + ' + ' + Math.round(again.gpuMs) + ' ms'
        + (info.sparse ? ' · ' + info.sparse + ' sparse rows of at most ' + info.sparseMax + ' entries' : ''));
      if (!rows) continue;
      const t = await core.buildTriples(slot, rows, Number(arg('--triple-gb', 20)) * 2 ** 30);
      const t2 = await core.buildTriples(slot, rows, Number(arg('--triple-gb', 20)) * 2 ** 30);
      console.log('bench ' + label + ' ' + name + '+3: ' + show(await rate(core, slot, secs)) + ' · triples ' + t.rows + ' rows, '
        + Math.round(t.tableBytes / 1048576) + ' MB in ' + Math.round(t.ms) + ' ms, then ' + Math.round(t2.ms) + ' ms ('
        + Math.round(t2.entries / t2.ms / 1000) + ' M entries/s)');
      // Leave the card's memory to the next model.
      await core.buildTriples(slot, 0, 0);
    }
  } finally {
    core.close();
  }
  return 0;
}

main().then((code) => { process.exit(code); }, (e) => {
  console.error('nebbench: ' + ((e && e.stack) || e));
  process.exit(1);
});
