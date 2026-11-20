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
  for (let i = 4; i <= 7; i++) {
    out.push(dsWait(7 - i + behind), ...p.t[i]);
    // bar: the barrier after tile `bar`, not at the slab's end. The stores are before it (se), and the wait in front
    // of it is for all the wave's LDS work, so no wave reads this slab's buffer once the barrier has let go.
    if (opt.bar === i) out.push(...p.post);
  }
  if (!opt.se) out.push(...p.stores);
  if (!opt.bar) out.push(...p.post);
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

// xbp2: xbp with the reads issued two at a time, every other tile, so the WMMA stream is broken half as often
// (an LDS instruction alone between WMMAs costs about four cycles, in a burst about one). Tiles 4 and 5's reads go
// after tile 1, tiles 6 and 7's with the stores after tile 3, the next slab's tiles 0 and 1's after tile 5 and its
// tiles 2 and 3's after tile 7. Each wait is right for the pass that falls into the loop as well, whose first four
// reads were issued together in front of it.
const xbp2 = (body) => {
  const { ps, tail } = xbPieces(body);
  const out = [];
  ps.forEach((p, s) => {
    const lastW = p.t[7].map((l) => opOf(l).startsWith('v_wmma')).lastIndexOf(true);
    const next = ps[(s + 1) % ps.length].g1;
    out.push(...p.pre, ...p.t[0], dsWait(2), ...p.t[1], p.g2[0], p.g2[1], dsWait(3), ...p.t[2], dsWait(2), ...p.t[3]);
    out.push(p.g2[2], p.g2[3], ...p.stores, dsWait(5), ...p.t[4], ...p.post, ...p.t[5], next[0], next[1], ...p.t[6]);
    out.push(...p.t[7].slice(0, lastW + 1), next[2], next[3], ...readoutInto(p.t[7].slice(lastW + 1), p.stores));
  });
  return { before: ps[0].g1, body: out.concat(tail), after: [dsWait(0)] };
};

// ---- g8: one break a slab ----------------------------------------------------------------------------------
// Any instruction that is not a WMMA costs about four cycles when it stands alone between WMMAs and about one in
// a burst (roll lost 1.5% to four lone reads a slab), and the compiler's slab stalls twice for a burst of A'
// reads. g8 gives the slab one break, at its end, and no read is waited for on a chunk's last slab:
//   - all 8 tiles' A' are read at once. The 16 more registers come from the loop itself: the second staging set
//     (one set is enough when it is stored before it is loaded again), the three B' address registers (the loads
//     take the tile's offset as an immediate instead), and four values the loop never uses, which are made again
//     from the thread index on the way out;
//   - the next slab's reads for tiles 0-5 are issued right after the slab's last WMMA, before the B' loads and
//     the readout, so the readout's XORs cover their latency; tiles 6 and 7's follow the readout, which works in
//     those two tiles' registers;
//   - the staged A' is stored at the slab's start, and the barrier stands after tile 4, where only waits are, with
//     a wait for all the wave's LDS work in front of it. The next slab's A' is then in LDS before the slab's end,
//     and no wave still reads the buffer that the stores after the next slab end go to.
// g8m issues the next slab's tiles 0-3 earlier still, right after the barrier (their registers are free from tile 4
// on), so a slab in mid-chunk, which has no readout to cover the reads, doesn't wait for them either.
// g8e keeps the barrier at the slab's end instead, just before the reads, so nothing at all stands between a
// slab's 128 WMMAs but waits.
// The loop meets the code around it in the compiler's state: staged A' for the slab after next in the first
// staging set, the next slab's A' in LDS behind a barrier. Its first reads go in front of the loop, and on the way
// out the reads issued for a slab the loop won't run are waited for before their registers are given back.
const regsOf = (line) => {
  const out = new Set();
  const t = line.split(';')[0];
  for (const m of t.matchAll(/v\[(\d+):(\d+)\]/g)) for (let v = Number(m[1]); v <= Number(m[2]); v++) out.add(v);
  for (const m of t.replace(/v\[\d+:\d+\]/g, '').matchAll(/\bv(\d+)\b/g)) out.add(Number(m[1]));
  return out;
};
const quadOf = (line, re) => {
  const m = re.exec(line);
  if (!m || Number(m[2]) !== Number(m[1]) + 3) fail(`not a four-register operand: ${line.trim()}`);
  return Number(m[1]);
};
// Does the instruction write VGPR r? Its first operand is the destination, except for the stores and LDS atomics,
// whose first operand is an address.
const writesReg = (line, r) => {
  const op = opOf(line);
  if (!/^(v_|ds_load|buffer_load)/.test(op)) return false;
  const m = /^\s+\S+\s+v(?:\[(\d+):(\d+)\]|(\d+))/.exec(line);
  if (!m) return false;
  const lo = Number(m[1] !== undefined ? m[1] : m[3]);
  const hi = Number(m[2] !== undefined ? m[2] : m[3]);
  return r >= lo && r <= hi;
};
const renumber = (line, map) => line
  .replace(/v\[(\d+):(\d+)\]/g, (t, lo, hi) => (map.has(Number(lo)) ? `v[${map.get(Number(lo))}:${map.get(Number(lo)) + Number(hi) - Number(lo)}]` : t))
  .replace(/\bv(\d+)\b/g, (t, n) => (map.has(Number(n)) ? `v${map.get(Number(n))}` : t));

const g8of = (mode, barAfter = 4) => (body, ctx) => {
  const mid = mode === 'm';
  const atEnd = mode === 'e';
  const { ps, tail } = xbPieces(body);
  const used = new Set();
  for (const l of body) if (opOf(l)) for (const r of regsOf(l)) used.add(r);

  // The staging sets: P is the one the pass's first slab stores (the code before the loop loaded it), Q the other.
  const setOf = (slab) => slab.stores.map((l) => quadOf(l, /ds_store_b128\s+v\d+,\s*v\[(\d+):(\d+)\]/));
  const P = setOf(ps[0]);
  const Q = setOf(ps[1]);
  if (P.join() !== setOf(ps[2]).join() || Q.join() !== setOf(ps[3]).join() || P.some((v) => Q.includes(v))) fail('the staging sets do not alternate');
  const toP = new Map();
  Q.forEach((q, i) => { for (let j = 0; j < 4; j++) toP.set(q + j, P[i] + j); });

  // The B' loads: one address register and three more, each the first OR a constant (the tile's offset).
  const before = ctx.lines.slice(ctx.fn[0], ctx.head);
  const defsOf = (r) => before.filter((l) => writesReg(l, r));
  const bLoads = body.filter((l) => opOf(l) === 'buffer_load_b128' && /s\[8:11\]/.test(l));
  const bAddr = [...new Set(bLoads.map((l) => Number(/buffer_load_b128\s+v\[\d+:\d+\],\s*v(\d+),/.exec(l)[1])))].sort((x, y) => x - y);
  if (bAddr.length !== 4) fail(`the B' loads use ${bAddr.length} address registers, not 4`);
  const bOff = new Map([[bAddr[0], 0]]);
  for (const r of bAddr.slice(1)) {
    const d = defsOf(r);
    const m = d.length === 1 && new RegExp(`^\\s+v_or_b32_e32\\s+v${r},\\s*(0x[0-9a-f]+|\\d+),\\s*v${bAddr[0]}\\s*$`).exec(d[0].split(';')[0]);
    if (!m) fail(`v${r} is not v${bAddr[0]} OR a constant, defined once before the loop`);
    bOff.set(r, Number(m[1]));
  }
  if (bLoads.some((l) => /offset:/.test(l))) fail("a B' load in the loop already has an offset");
  const bLoad = (l) => {
    if (opOf(l) !== 'buffer_load_b128' || !/s\[8:11\]/.test(l)) return l;
    const m = /^(\s+buffer_load_b128\s+v\[\d+:\d+\],\s*)v(\d+)(,.*?offen)\s*$/.exec(l.split(';')[0]);
    if (!m) fail(`a B' load of another form: ${l.trim()}`);
    const off = bOff.get(Number(m[2]));
    return `${m[1]}v${bAddr[0]}${m[3]}${off ? ` offset:${off}` : ''}`;
  };

  // The registers the loop gives up for tiles 4-7's A': Q's two quads, and two quads that hold the three B' address
  // registers and values the loop doesn't touch. Each of those is made again after the loop from registers the
  // loop leaves alone.
  const spare = [];
  for (let lo = 161; lo + 3 <= 191 && spare.length < 2; lo++) {
    if (spare.length && lo < spare[0] + 4) continue;
    if ([lo, lo + 1, lo + 2, lo + 3].every((r) => !used.has(r) || bAddr.slice(1).includes(r))) spare.push(lo);
  }
  if (spare.length !== 2) fail('no two free register quads beside the staging set');
  const freed = spare.flatMap((lo) => [lo, lo + 1, lo + 2, lo + 3]);
  // The way back. A register the loop only reads, or doesn't touch, still holds its value after the loop. A freed
  // register defined once before the loop from such registers (or from freed ones already made again) is defined
  // again the same way. One whose sources are gone, but which a kept sum x + y holds with y made again, is taken
  // back out of the sum.
  const bodyWrites = (r) => body.some((l) => writesReg(l, r));
  const redo = [];
  const todo = new Set(freed);
  const defLine = (r) => {
    const d = defsOf(r);
    if (d.length !== 1) fail(`v${r} is written ${d.length} times before the loop, not once`);
    return d[0];
  };
  for (let pass = 0; pass < freed.length && todo.size; pass++) {
    for (const r of [...todo]) {
      const d = defLine(r);
      const src = [...regsOf(d)].filter((v) => v !== r);
      const at = before.indexOf(d);
      const good = (v) => !todo.has(v) && !bodyWrites(v) && (freed.includes(v) || !before.slice(at + 1).some((l) => writesReg(l, v)));
      if (!opOf(d).startsWith('v_') || !src.every(good)) continue;
      redo.push(d.split(';')[0].replace(/\s+$/, ''));
      todo.delete(r);
    }
  }
  for (const r of [...todo]) {
    const sum = before.map((l) => new RegExp(`^\\s+v_add_nc_u32_e32\\s+v(\\d+),\\s*v${r},\\s*v(\\d+)\\s*$`).exec(l.split(';')[0])).filter(Boolean).pop();
    if (!sum) fail(`v${r} can't be made again after the loop`);
    const [x, y] = [Number(sum[1]), Number(sum[2])];
    const at = before.findIndex((l) => l.split(';')[0] === sum[0]);
    const kept = (v) => !bodyWrites(v) && !freed.includes(v) && !before.slice(at + 1).some((l) => writesReg(l, v));
    if (!kept(x) || todo.has(y) || !(freed.includes(y) || kept(y))) fail(`v${r} can't be made again after the loop`);
    redo.push(`\tv_sub_nc_u32_e32 v${r}, v${x}, v${y}`);
    todo.delete(r);
  }

  const quads = [Q[0], Q[1], spare[0], spare[1]];   // tiles 4-7
  const tmp = [spare[0], spare[0] + 1, spare[0] + 2, spare[0] + 3, spare[1]];   // the readout's, in tiles 6 and 7's
  const out = [];
  const reads = ps.map((p) => {
    const g2 = p.g2.map((l, i) => {
      const lo = quadOf(l, /ds_load_b128\s+v\[(\d+):(\d+)\]/);
      return { line: l.replace(/v\[\d+:\d+\]/, `v[${quads[i]}:${quads[i] + 3}]`), from: lo, to: quads[i] };
    });
    return { all: [...p.g1, ...g2.map((g) => g.line)], g2 };
  });
  ps.forEach((p, s) => {
    const next = reads[(s + 1) % ps.length].all;
    const stage = p.pre.filter((l) => opOf(l) === 'buffer_load_b128' || opOf(l) === 's_clause');
    const nStage = stage.filter((l) => opOf(l) === 'buffer_load_b128').length;
    if (nStage !== 2 || stage.length !== 3) fail(`a slab starts with ${nStage} staging loads in ${stage.length} lines, not 2 under one s_clause`);
    out.push(...p.pre.filter((l) => !stage.includes(l)));
    out.push('\ts_wait_loadcnt 0x4', ...p.stores.map((l) => renumber(l, toP)), ...stage.map((l) => renumber(l, toP)));
    // tile 0: its first wait asked for the first of four reads; now ten LDS operations may be out
    const t0 = p.t[0].slice();
    const w = t0.findIndex((l) => opOf(l) === 's_wait_loadcnt_dscnt');
    if (w < 0 || !/0x503\s*$/.test(t0[w])) fail('tile 0 does not start with s_wait_loadcnt_dscnt 0x503');
    t0[w] = t0[w].replace('0x503', '0x509');
    out.push(...t0);
    if (barAfter === 0 && !atEnd) out.push(...p.post);
    for (let i = 1; i <= 7; i++) {
      if (i <= barAfter || atEnd) out.push(dsWait(9 - i));   // after the barrier's own wait everything has arrived
      let lines = p.t[i];
      if (i === 7) lines = lines.slice(0, lines.map((l) => opOf(l).startsWith('v_wmma')).lastIndexOf(true) + 1);
      if (i >= 4) {
        const g = reads[s].g2[i - 4];
        const map = new Map([0, 1, 2, 3].map((j) => [g.from + j, g.to + j]));
        // only the A operand: the second register of a WMMA
        lines = lines.map((l) => (opOf(l).startsWith('v_wmma')
          ? l.replace(/^(\s+\S+\s+v\[\d+:\d+\],\s*)v(\d+)/, (t, head, n) => `${head}v${map.has(Number(n)) ? map.get(Number(n)) : fail(`tile ${i} reads v${n}`)}`)
          : l));
      }
      out.push(...lines);
      if (i === barAfter && !atEnd) out.push(...p.post, ...(mid ? next.slice(0, 4) : []));
    }
    const rest = p.t[7].slice(p.t[7].map((l) => opOf(l).startsWith('v_wmma')).lastIndexOf(true) + 1);
    const x0 = rest.findIndex((l) => /^v_xor/.test(opOf(l)));
    const x1 = rest.findIndex((l) => opOf(l) === 'ds_xor_b32');
    let ro = rest.map(bLoad);
    if (x0 >= 0) {
      const map = new Map();
      for (let i = x0; i <= x1; i++) {
        const m = /^\s+v_(?:xor3?_b32\w*|alignbit_b32)\s+v(\d+),/.exec(ro[i]);
        if (!m || map.has(Number(m[1]))) continue;
        if (map.size >= tmp.length) fail('the readout uses more than five working registers');
        map.set(Number(m[1]), tmp[map.size]);
      }
      ro = ro.map((l, i) => (i < x0 || i > x1 || !opOf(l) ? l : l.replace(/\bv(\d+)\b/g, (t, n) => (map.has(Number(n)) ? `v${map.get(Number(n))}` : t))));
    }
    if (atEnd) out.push(...p.post);
    out.push(...next.slice(mid ? 4 : 0, 6), ...ro, ...next.slice(6));
  });
  return { before: reads[0].all, body: out.concat(tail), after: [dsWait(0), ...redo] };
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
  xbp2,
  g8: g8of(''),
  g8m: g8of('m'),
  g8e: g8of('e'),
  // the barrier's place: after tile N of the compiler's slab (stores in mid-slab), or of g8's
  bm4: reslab({ se: true, bar: 4 }),
  bm5: reslab({ se: true, bar: 5 }),
  bm6: reslab({ se: true, bar: 6 }),
  g8b0: g8of('', 0),
  g8b1: g8of('', 1),
  g8b2: g8of('', 2),
  g8b3: g8of('', 3),
  g8b5: g8of('', 5),
  g8b6: g8of('', 6),
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
    const r = OPS[op](body, { lines, fn: f, head: loop.head });
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
