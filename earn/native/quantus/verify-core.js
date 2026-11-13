#!/usr/bin/env node
'use strict';

// Check a Quantus core against the CPU reference, on a card, and time it.
//
//   node earn/native/quantus/verify-core.js --addon <file.node> [--device 0] [--big 1073741824] [--exact 4096]
//   QUANTUS_CORE=cpu node earn/native/quantus/verify-core.js --big 4096 --exact 256
//
// It runs src/main/quantusCore.js's verifyCore: the five official vectors, two
// exact ranges against the reference, the overflow path, and a large timed run
// whose every hit is checked. Exit 0 when every check passes. See README.md.

const path = require('path');
const { quantusCoreFactory, verifyCore } = require('../../src/main/quantusCore');
const { formatHs } = require('../../src/shared/format');

function arg(name, fallback) {
  const i = process.argv.indexOf(name);
  return i > 0 && i + 1 < process.argv.length ? process.argv[i + 1] : fallback;
}

async function main() {
  const addon = arg('--addon', null);
  const env = Object.assign({}, process.env);
  if (addon) env.QUANTUS_CORE_PATH = path.resolve(addon);
  const factory = quantusCoreFactory({ env, log: (level, line) => console.log(level + ': ' + line) });
  if (!factory) return 1;
  const device = Number(arg('--device', factory.devices[0].index));
  const core = factory.createCore(device);
  console.log('core: ' + factory.label + ' · card ' + device + ' · ' + JSON.stringify(core.device));
  try {
    const r = await verifyCore(core, { bigCount: Number(arg('--big', 2 ** 30)), exactCount: Number(arg('--exact', 4096)) });
    for (const c of r.checks) console.log((c.ok ? 'PASS ' : 'FAIL ') + c.name.padEnd(9) + c.detail);
    console.log((r.ok ? 'PASS' : 'FAIL') + ' · ' + formatHs(r.hashrate));
    return r.ok ? 0 : 1;
  } finally {
    core.close();
  }
}

main().then((code) => { process.exitCode = code; }, (e) => {
  console.error('verify-core: ' + ((e && e.stack) || e));
  process.exitCode = 1;
});
