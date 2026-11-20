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

// The loop's slabs: the body cut after each s_barrier_wait. What follows the last barrier (the closing branch) is
// returned as the tail.
function slabsOf(body) {
  const slabs = [];
  let from = 0;
  for (let i = 0; i < body.length; i++) {
    if (opOf(body[i]) !== 's_barrier_wait') continue;
    slabs.push(body.slice(from, i + 1));
    from = i + 1;
  }
  return { slabs, tail: body.slice(from) };
}

// One W64K slab in pieces. The compiler's slab is: 4 A' reads (tiles 0-3), each tile's 16 WMMAs after a wait for
// its own read, 4 more reads (tiles 4-7) and their tiles, then the next slab's B' loads, the readout on a chunk's
// last slab, the staged A' stored to LDS, and the barrier.
//   pre     up to the first group's reads
//   g1, g2  the two groups' ds_load_b128 lines, 4 each
//   t[0..7] each tile's lines; t[0] starts after g1 and t[i] after tile i's s_wait_dscnt, which is not kept;
//           t[7] runs to the stores
//   stores  the two ds_store_b128
//   post    from the wait for the stores on (the barrier, and the next slab's staging loads when they follow)
function slabPieces(slab) {
  const idx = (pred) => slab.map((l, i) => (pred(l) ? i : -1)).filter((i) => i >= 0);
  const loads = idx((l) => opOf(l) === 'ds_load_b128');
  const stores = idx((l) => opOf(l) === 'ds_store_b128');
  if (loads.length !== 8 || stores.length !== 2) fail(`a slab has ${loads.length} LDS reads and ${stores.length} stores, not 8 and 2`);
  const run = (a) => a.every((v, i) => i === 0 || slab.slice(a[i - 1] + 1, v).every((l) => !opOf(l)));
  if (!run(loads.slice(0, 4)) || !run(loads.slice(4)) || !run(stores)) fail('a slab\'s LDS reads or stores are not in runs of 4, 4 and 2');
  const waits = idx((l) => opOf(l) === 's_wait_dscnt').filter((i) => i > loads[3] && i < stores[0]);
  const want = ['0x2', '0x1', '0x0', '0x3', '0x2', '0x1', '0x0'];
  const got = waits.map((i) => /s_wait_dscnt\s+(\S+)/.exec(slab[i])[1]);
  if (got.join() !== want.join()) fail(`a slab's LDS waits are ${got.join(' ')}, not ${want.join(' ')}`);
  if (!(waits[2] < loads[4] && loads[7] < waits[3])) fail('the second group of LDS reads is not between tiles 3 and 4');
  if (opOf(slab[stores[1] + 1]) !== 's_wait_dscnt') fail('no wait after the LDS stores');
  const t = [];
  t.push(slab.slice(loads[3] + 1, waits[0]));
  t.push(slab.slice(waits[0] + 1, waits[1]));
  t.push(slab.slice(waits[1] + 1, waits[2]));
  t.push(slab.slice(waits[2] + 1, loads[4]));
  t.push(slab.slice(waits[3] + 1, waits[4]));
  t.push(slab.slice(waits[4] + 1, waits[5]));
  t.push(slab.slice(waits[5] + 1, waits[6]));
  t.push(slab.slice(waits[6] + 1, stores[0]));
  for (let i = 0; i < 8; i++) {
    const n = t[i].filter((l) => opOf(l).startsWith('v_wmma')).length;
    if (n !== 16) fail(`tile ${i} of a slab has ${n} WMMAs, not 16`);
  }
  return {
    pre: slab.slice(0, loads[0]),
    g1: loads.slice(0, 4).map((i) => slab[i]),
    g2: loads.slice(4).map((i) => slab[i]),
    t,
    stores: stores.map((i) => slab[i]),
    post: slab.slice(stores[1] + 1),
  };
}

const dsWait = (n) => `\ts_wait_dscnt 0x${n.toString(16)}`;

// The slab written again from its pieces, in the order `opt` asks for:
//   roll  tile i + 4's A' read is issued as soon as tile i's WMMAs are done (its registers are free then), so no
//         tile waits for a burst of reads: each read has three tiles' WMMAs to arrive
//   se    the staged A' goes to LDS in mid-slab (it arrived a slab ago), not at the slab's end, so the barrier
//         doesn't wait for the stores
// The waits are recounted: LDS operations return in order, and each tile waits for its own read.
function writeSlab(p, opt) {
  const out = [...p.pre, ...p.g1, ...p.t[0]];
  if (opt.roll) {
    for (let i = 1; i <= 3; i++) out.push(p.g2[i - 1], dsWait(3), ...p.t[i]);
    out.push(p.g2[3]);
  } else {
    for (let i = 1; i <= 3; i++) out.push(dsWait(3 - i), ...p.t[i]);
    if (opt.se) out.push(...p.stores);   // before the reads, so the reads' waits stay as they were
    out.push(...p.g2);
  }
  if (opt.roll && opt.se) out.push(...p.stores);
  const behind = opt.roll && opt.se ? 2 : 0;   // the stores issued after tile 7's read
  for (let i = 4; i <= 7; i++) out.push(dsWait(7 - i + behind), ...p.t[i]);
  if (!opt.se) out.push(...p.stores);
  out.push(...p.post);
  return out;
}

// The compiler's readout XORs into the A' fragment registers, which are dead at a slab's end in its loop. In xb
// they hold the next slab's first tiles by then, so the readout's working registers are renamed to the staging
// registers this slab stored in mid-slab (free until the next slab's staging loads, which come after the readout).
function readoutInto(lines, stores) {
  const first = lines.findIndex((l) => /^v_xor/.test(opOf(l)));
  if (first < 0) return lines;   // not a chunk's last slab
  const end = lines.findIndex((l) => opOf(l) === 'ds_xor_b32');
  if (end < first) fail('the readout does not end with ds_xor_b32');
  const free = stores.flatMap((l) => {
    const m = /ds_store_b128\s+v\d+,\s*v\[(\d+):(\d+)\]/.exec(l);
    if (!m) fail(`not a staged store: ${l.trim()}`);
    const r = [];
    for (let v = Number(m[1]); v <= Number(m[2]); v++) r.push(v);
    return r;
  });
  const map = new Map();
  for (let i = first; i <= end; i++) {
    const m = /^\s+v_(?:xor3?_b32\w*|alignbit_b32)\s+v(\d+),/.exec(lines[i]);
    if (!m) continue;
    const dst = Number(m[1]);
    if (dst >= 1 && dst <= 128) fail(`the readout writes v${dst}, a sum`);
    if (!map.has(dst)) {
      if (map.size >= free.length) fail('the readout uses more working registers than the staging set holds');
      map.set(dst, free[map.size]);
    }
  }
  return lines.map((l, i) => (i < first || i > end || !opOf(l)
    ? l
    : l.replace(/\bv(\d+)\b/g, (t, n) => (map.has(Number(n)) ? `v${map.get(Number(n))}` : t))));
}

// xb: the pass's four slabs with every A' read rolling, across the slab ends inside the pass too. Slab s stages
// slab s + 1's A' in mid-slab and has its barrier after tile 4 (after a wait for all its own LDS work, so no wave
// still reads the buffer the next stores go to). From there on each tile that finishes issues the read of the next
// slab's tile 0-3 into its own registers. So a slab end inside the pass has no stores, no barrier and no burst of
// reads: the next slab's first tile only waits for its B'. The pass's last slab ends as `rollse` does, so the loop
// meets the code around it in the state the compiler's loop does.
function writePassXb(ps, wrap) {
  const out = [];
  const last = ps.length - 1;
  ps.forEach((p, s) => {
    // tile 7's lines up to its last WMMA, and what follows them (the next slab's B' loads, the readout)
    const lastW = p.t[7].map((l) => opOf(l).startsWith('v_wmma')).lastIndexOf(true);
    const t7w = p.t[7].slice(0, lastW + 1);
    const t7rest = p.t[7].slice(lastW + 1);
    out.push(...p.pre);
    if (s === 0 && !wrap) out.push(...p.g1);   // every other slab's tiles 0-3 were read during the slab before
    out.push(...p.t[0]);
    for (let i = 1; i <= 3; i++) out.push(p.g2[i - 1], dsWait(3), ...p.t[i]);
    out.push(p.g2[3], ...p.stores, dsWait(5), ...p.t[4]);
    if (s === last && !wrap) {
      for (let i = 5; i <= 7; i++) out.push(dsWait(9 - i), ...p.t[i]);
      out.push(...p.post);
      return;
    }
    const next = ps[(s + 1) % ps.length].g1;
    out.push(...p.post, next[0], ...p.t[5], next[1], ...p.t[6], next[2], ...t7w, next[3], ...readoutInto(t7rest, p.stores));
  });
  return out;
}

function xbPieces(body) {
  const { slabs, tail } = slabsOf(body);
  if (slabs.length !== 4) fail(`the loop has ${slabs.length} slabs, not 4`);
  const ps = slabs.map(slabPieces);
  for (const p of ps) {
    const ops = p.post.map(opOf).filter(Boolean);
    if (ops.join() !== 's_wait_dscnt,s_barrier_signal,s_barrier_wait') fail(`a slab ends with ${ops.join(' ')}, not the wait and the barrier`);
  }
  return { ps, tail };
}

const xb = (body) => {
  const { ps, tail } = xbPieces(body);
  return writePassXb(ps, false).concat(tail);
};

// xbp: xb at the pass's own end too. The first slab's first four reads move out in front of the loop, and the
// pass's last slab issues them for the next pass as its tiles 4-7 finish. On the way out of the loop those four
// reads are waited for, since the code after the loop reuses their registers.
const xbp = (body) => {
  const { ps, tail } = xbPieces(body);
  return { before: ps[0].g1, body: writePassXb(ps, true).concat(tail), after: [dsWait(0)] };
};

const reslab = (opt) => (body) => {
  const { slabs, tail } = slabsOf(body);
  if (slabs.length !== 4) fail(`the loop has ${slabs.length} slabs, not 4`);
  return slabs.flatMap((s) => writeSlab(slabPieces(s), opt)).concat(tail);
};

const OPS = {
  rt: (body) => body,
  rs: reslab({}),
  roll: reslab({ roll: true }),
  se: reslab({ se: true }),
  rollse: reslab({ roll: true, se: true }),
  xb,
  xbp,
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
  // An op gives the body back, or { before, body, after }: lines to go in front of the loop's label and after its
  // closing branch as well. The label must then be reached only by falling into it and by that branch.
  const front = [];
  const back = [];
  for (const op of opList.split('+')) {
    if (!OPS[op]) fail(`unknown op ${op}`);
    const r = OPS[op](body);
    if (Array.isArray(r)) {
      body = r;
      continue;
    }
    body = r.body;
    front.push(...r.before);
    back.unshift(...r.after);
  }
  if (front.length || back.length) {
    const name = /^(\.LBB[0-9_]+):/.exec(lines[loop.head])[1];
    const uses = lines.slice(f[0], f[1]).filter((l) => new RegExp(`\\s${name.replace('.', '\\.')}(\\s|$)`).test(l)).length;
    if (uses !== 1) fail(`the loop's label ${name} is branched to from ${uses} places, not one`);
    if (!opOf(lines[loop.head - 1]) || /^s_c?branch/.test(opOf(lines[loop.head - 1]))) fail('the loop is not entered by falling into it');
  }
  const after = body.filter((l) => opOf(l)).length;
  const out = lines.slice(0, loop.head).concat(front, [lines[loop.head]], body, back, lines.slice(loop.tail + 1));
  fs.writeFileSync(outPath, out.join('\n'));
  process.stdout.write(`handfold: ${fn} loop of ${loop.wmma} WMMAs, ${before} instructions -> ${after} (${opList})\n`);
}

if (require.main === module) main(process.argv.slice(2));
module.exports = { findFunction, findLoops, dropLines, opOf, OPS };
