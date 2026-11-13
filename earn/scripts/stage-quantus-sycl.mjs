#!/usr/bin/env node
'use strict';

// Put the Intel Quantus core (quantus_sycl.node, for --coin quantus) into the
// Pearl Intel core's sycl/ folder, where every package keeps it and
// src/main/quantusCore.js looks for it:
//
//   node scripts/stage-quantus-sycl.mjs <from> <into>
//
// <from> is the Quantus core's staged folder (native-core.yml's
// quantus-core-sycl-* artifact): the addon, the SYCL runtime it loads, the
// runtime's licence and a NOTICE.txt. <into> is the package's sycl/ folder
// (vendor/native/sycl), made when it is not there. CI builds the two Intel
// cores in separate jobs with the same open-source DPC++ release, and each
// stages that runtime beside its addon. On Windows a DLL is found only beside
// the .node that loads it, so the two cores share one folder and one copy of
// the runtime:
//
// - every file of <from> that <into> does not have goes in;
// - a file <into> has must be the same bytes. The two cores would load one
//   runtime, so different bytes mean the jobs built with different compilers,
//   and nothing is staged (exit 1);
// - the NOTICE.txt goes in as NOTICE-quantus.txt, beside the Pearl one;
// - the addon goes in last, so a folder never has it without its runtime.
//
// Nothing to stage (no <from>, or no quantus_sycl.node in it) is not an error:
// the core is optional, and the script says so and exits 0.

import { existsSync, readdirSync, readFileSync, copyFileSync, mkdirSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';

const ADDON = 'quantus_sycl.node';
const NOTICE = 'NOTICE.txt';
const NOTICE_AS = 'NOTICE-quantus.txt';

const [fromArg, intoArg] = process.argv.slice(2);
if (!fromArg || !intoArg) {
  console.error('usage: node scripts/stage-quantus-sycl.mjs <from> <into>');
  process.exit(2);
}
const from = resolve(fromArg);
const into = resolve(intoArg);

if (!existsSync(join(from, ADDON))) {
  console.log('no ' + join(from, ADDON) + ': no Intel Quantus core to stage');
  process.exit(0);
}

const files = readdirSync(from).filter((f) => statSync(join(from, f)).isFile());
const target = (f) => (f === NOTICE ? NOTICE_AS : f);
const plan = [];
const clash = [];
for (const f of files) {
  const dest = join(into, target(f));
  if (!existsSync(dest)) plan.push(f);
  else if (!readFileSync(dest).equals(readFileSync(join(from, f)))) clash.push(f);
}
if (clash.length) {
  console.error('the Intel Quantus core was not staged: ' + clash.join(', ') + ' in ' + from
    + ' differ from the copies in ' + into + '. The two Intel cores must be built with the same DPC++'
    + ' release (native-core.yml), since they load one SYCL runtime.');
  process.exit(1);
}

mkdirSync(into, { recursive: true });
// The addon last: until it is in, the folder is what it was plus runtime files it already needed.
plan.sort((a, b) => (a === ADDON) - (b === ADDON));
for (const f of plan) copyFileSync(join(from, f), join(into, target(f)));
const kept = files.length - plan.length;
console.log('staged ' + ADDON + ' in ' + into + ' (' + plan.length + ' file' + (plan.length === 1 ? '' : 's') + ' copied'
  + (kept ? ', ' + kept + ' already there and the same' : '') + ')');
