#!/usr/bin/env node
'use strict';
// handfold.js IN.s OUT.s OPS [FUNCTION]
//
// Edits the device assembly of one target (clang -S of pearl_host_hip.cpp's device pass) before build-amd.sh
// assembles it: the hand-written route for the AMD folds. It works on the fold's chunk loop, the stretch from a
// label to the branch back to it that holds the most WMMAs. OPS is a "+"-separated list, applied in order.
//
// Ops that keep the fold's results (a build with only these must pass the fold check and verify-hits):
//   rt        nothing: the assembly goes back as it came (the round trip itself)
//   nodly     the loop's s_delay_alu taken out (they are hints; the hardware waits on its own)
//   dly1      every s_delay_alu in the loop that sits between two WMMAs taken out, the others kept
//   uns       the WMMAs unsigned (no neg_lo). The same sums only while every operand byte is 0..127, as on
//             gfx12 with fill 63; the fold check's int8 operands fail it, so it is timing-only for now
//
// Timing-only ops (wrong transcripts, to price a part of the loop at the clock the card really runs):
//   noro      no readout XORs (v_xor3, v_xor)
//   nob       no B' loads (buffer loads through s[8:11]); B' keeps the first slab's bytes
//   noa       no A' reads from LDS (ds_load_b128); A' keeps the first group's bytes
//   nost      no staging (buffer loads through s[4:7], ds_store_b128)
//   nobar     no slab barrier
//
// Exit 0: OUT.s written. Exit 3: the fold or its loop wasn't found as expected; nothing written, and the caller
// builds the compiler's assembly.

const fs = require('fs');

function fail(msg) {
  process.stderr.write(`handfold: ${msg}\n`);
  process.exit(3);
}

const opOf = (line) => {
  const m = /^\s+([a-z_0-9]+)/.exec(line);
  return m ? m[1] : '';
};

// The function's lines [start, end) in the file: from its label to its .Lfunc_end.
function findFunction(lines, name) {
  // clang ends the label's line with a comment
  const start = lines.findIndex((l) => l === `${name}:` || l.startsWith(`${name}: `));
  if (start < 0) return null;
  for (let i = start + 1; i < lines.length; i++) if (/^\.Lfunc_end/.test(lines[i])) return [start, i];
  return null;
}

// Every backward branch in [start, end): { head, tail, wmma }, head the label's line and tail the branch's.
function findLoops(lines, start, end) {
  const label = new Map();
  for (let i = start; i < end; i++) {
    const m = /^(\.LBB[0-9_]+):/.exec(lines[i]);
    if (m) label.set(m[1], i);
  }
  const loops = [];
  for (let i = start; i < end; i++) {
    const m = /^\s+s_c?branch\S*\s+(\.LBB[0-9_]+)/.exec(lines[i]);
    if (!m || !label.has(m[1]) || label.get(m[1]) > i) continue;
    const head = label.get(m[1]);
    let wmma = 0;
    for (let j = head; j <= i; j++) if (opOf(lines[j]).startsWith('v_wmma')) wmma++;
    loops.push({ head, tail: i, wmma });
  }
  return loops;
}

// Deletes the loop lines `drop` picks, and any s_clause whose instructions lose one of their number.
function dropLines(body, drop) {
  const gone = body.map((l) => drop(l));
  for (let i = 0; i < body.length; i++) {
    if (opOf(body[i]) !== 's_clause') continue;
    const n = Number(/s_clause\s+(\S+)/.exec(body[i])[1]) + 1;
    let seen = 0;
    for (let j = i + 1; j < body.length && seen < n; j++) {
      if (!opOf(body[j])) continue;
      seen++;
      if (gone[j]) gone[i] = true;
    }
  }
  return body.filter((_, i) => !gone[i]);
}

const OPS = {
  rt: (body) => body,
  nodly: (body) => dropLines(body, (l) => opOf(l) === 's_delay_alu'),
  dly1: (body) => {
    const isIns = body.map((l) => opOf(l) !== '');
    const next = (i) => { for (let j = i + 1; j < body.length; j++) if (isIns[j]) return opOf(body[j]); return ''; };
    const prev = (i) => { for (let j = i - 1; j >= 0; j--) if (isIns[j]) return opOf(body[j]); return ''; };
    const gone = body.map((l, i) => opOf(l) === 's_delay_alu' && next(i).startsWith('v_wmma') && prev(i).startsWith('v_wmma'));
    return body.filter((_, i) => !gone[i]);
  },
  uns: (body) => body.map((l) => (opOf(l).startsWith('v_wmma') ? l.replace(/\s+neg_lo:\[1,1,0\]/, '') : l)),
  noro: (body) => dropLines(body, (l) => /^v_xor3?_b32/.test(opOf(l))),
  nob: (body) => dropLines(body, (l) => opOf(l) === 'buffer_load_b128' && /s\[8:11\]/.test(l)),
  noa: (body) => dropLines(body, (l) => opOf(l) === 'ds_load_b128'),
  nost: (body) => dropLines(body, (l) => (opOf(l) === 'buffer_load_b128' && /s\[4:7\]/.test(l)) || opOf(l) === 'ds_store_b128'),
  nobar: (body) => dropLines(body, (l) => /^s_barrier_(signal|wait)$/.test(opOf(l))),
};

function main(argv) {
  const [inPath, outPath, opList, fn = 'pearl_amd_fold_wmma12'] = argv;
  if (!inPath || !outPath || !opList) fail('usage: handfold.js IN.s OUT.s OPS [FUNCTION]');
  const lines = fs.readFileSync(inPath, 'utf8').split('\n');
  const f = findFunction(lines, fn);
  if (!f) fail(`no function ${fn} in ${inPath}`);
  const loops = findLoops(lines, f[0], f[1]);
  if (!loops.length) fail(`no loop in ${fn}`);
  const loop = loops.reduce((a, b) => (b.wmma > a.wmma ? b : a));
  if (loop.wmma < 64) fail(`the largest loop of ${fn} has only ${loop.wmma} WMMAs`);
  // The loop's body: after its label, up to and including its closing branch.
  let body = lines.slice(loop.head + 1, loop.tail + 1);
  const before = body.filter((l) => opOf(l)).length;
  for (const op of opList.split('+')) {
    if (!OPS[op]) fail(`unknown op ${op}`);
    body = OPS[op](body);
  }
  const after = body.filter((l) => opOf(l)).length;
  const out = lines.slice(0, loop.head + 1).concat(body, lines.slice(loop.tail + 1));
  fs.writeFileSync(outPath, out.join('\n'));
  process.stdout.write(`handfold: ${fn} loop of ${loop.wmma} WMMAs, ${before} instructions -> ${after} (${opList})\n`);
}

if (require.main === module) main(process.argv.slice(2));
module.exports = { findFunction, findLoops, dropLines, opOf, OPS };
