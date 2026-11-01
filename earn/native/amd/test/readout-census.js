'use strict';

// The counting half of readout-census.sh: reads the assembly and resource remarks that script
// wrote for each target and build (base, reg, store), cuts out pearl_amd_fold_wmma12, counts
// its instructions, and checks them against the static gates of the two-lane decision (round 4
// item 2, Stage A). Prints one table a target and exits 1 on any miss.
//
//   node readout-census.js DIR gfx1200 [gfx1201 ...]
//
// DIR holds <target>-<build>.s (hip-clang --cuda-device-only -S) and <target>-<build>.txt
// (its -Rpass-analysis=kernel-resource-usage remarks). The counts come from the text of one
// function, so they are static counts: an instruction in a loop counts once.
//
// Terms:
//   slab loop    the loop that holds the v_wmma, from its label to the branch back to it: the
//                hot path. The gates on VALU, the keep step and the store look only here, so
//                code that runs once a tile (the tail's readout, the store form's zero fill
//                after the loop) can't sink or save a build.
//   copies (N)   how many copies of the readout the compiler made: today's build's
//                v_permlanex16 count / 4 (four words a copy, one lane swap each). Expect 3 in
//                the function (the two PF2 parities and the tail) and 2 in the slab loop.
//   slab VALU    every v_ instruction in the slab loop. Everything there but the readout (WMMA,
//                staging, addresses) is the same in the three builds, so a two-lane build's
//                difference from base is what its readout saves, whatever instructions the
//                compiler picked.
//   readout VALU a record only: the instructions only a readout uses (v_xor3, v_xor in any
//                form, v_mov_b32_dpp, v_permlane*, v_cndmask, v_cmp, v_movrel), plus the
//                v_mov_b32 that fills a lane move's destination first. v_permlanex16 and
//                v_mov_b32_dpp read their destination too, so a word still needed is copied
//                there before each one.
//   keep step    the register form's v_movrel, v_cndmask, v_cmp and plain v_mov_b32 in the slab
//                loop, less the store form's: what keeping word c in registers costs over
//                storing it.
//   xor3 chain   the longest run of v_xor3 that each read the one before, in program order.

const fs = require('fs');
const path = require('path');

const FOLD = 'pearl_amd_fold_wmma12';
const BUILDS = ['base', 'reg', 'store'];

// The decision's gates (decision.md, "Static checks").
const GATE = {
  vgprsHard: 216,     // allocates 216 at 6 waves; more and the fold loses a wave
  vgprsTarget: 192,   // what round 2's 8-wave arm would need; recorded, not gated
  savePerCopy: 20,    // readout VALU at most base - 20 a copy
  xor3Chain: 17,      // four chains of 16 and the join
  keepPerCopy: 3,     // the register form's keep step
  baseUnfolded: 95,   // today's readout at 95 or more a copy means DPP isn't folded
};

// Lines from `name:` to the function's end label.
function cutFunction(asm, name) {
  const lines = asm.replace(/\r\n/g, '\n').split('\n');
  const a = lines.findIndex((l) => l.startsWith(name + ':'));
  if (a < 0) return null;
  let b = lines.findIndex((l, i) => i > a && /^\.Lfunc_end\d+:/.test(l));
  if (b < 0) b = lines.length;
  return lines.slice(a + 1, b);
}

// Split operands on commas outside brackets.
function splitOperands(s) {
  const out = [];
  let depth = 0, cur = '';
  for (const ch of s) {
    if (ch === '[') depth++;
    if (ch === ']') depth--;
    if (ch === ',' && depth === 0) { out.push(cur.trim()); cur = ''; } else cur += ch;
  }
  if (cur.trim()) out.push(cur.trim());
  return out;
}

// One instruction or label a line, comments and directives dropped.
function parse(lines) {
  const insts = [], labels = new Map();
  for (const raw of lines) {
    const line = raw.replace(/;.*$/, '').trim();
    if (!line) continue;
    const lab = /^([.\w$]+):$/.exec(line);
    if (lab) { labels.set(lab[1], insts.length); continue; }
    if (line.startsWith('.')) continue;
    const m = /^([a-z][a-z0-9_.]*)\s*(.*)$/.exec(line);
    if (!m) continue;
    const ops = splitOperands(m[2]);
    // The last operand may carry modifiers after a space (quad_perm:[...] row_mask:0xf, offset:16).
    const regs = ops.map((o) => o.split(/\s+/)[0]);
    insts.push({ op: m[1], ops: regs, text: m[2] });
  }
  // Loops: from a label to a branch back to it. The slab loop is any of them that holds a
  // v_wmma.
  const loops = [];
  insts.forEach((ins, i) => {
    if (!/^s_(cbranch_\w+|branch)$/.test(ins.op)) return;
    const at = labels.get(ins.ops[0]);
    if (at !== undefined && at <= i)
      loops.push({ a: at, b: i, slab: insts.slice(at, i + 1).some((x) => /^v_wmma/.test(x.op)) });
  });
  insts.forEach((ins, i) => { ins.slab = loops.some((l) => l.slab && l.a <= i && i <= l.b); });
  return insts;
}

// The VGPR numbers an operand names: v7, v[4:7].
function vgprs(op) {
  let m = /^v(\d+)$/.exec(op || '');
  if (m) return [Number(m[1])];
  m = /^v\[(\d+):(\d+)\]$/.exec(op || '');
  if (!m) return [];
  const out = [];
  for (let r = Number(m[1]); r <= Number(m[2]); r++) out.push(r);
  return out;
}

const isStore = (op) => /^(global|flat|buffer|scratch)_store|^ds_(store|write)/.test(op);
const isDpp = (ins) => /_dpp$|_dpp_/.test(ins.op) || /\b(quad_perm|row_\w+|dpp8):/.test(ins.text);
const isMov = (ins) => /^v_mov_b32(_e32|_e64)?$/.test(ins.op) && !isDpp(ins);
const isReadout = (op) =>
  /^v_(xor3_b32|xor_b32|mov_b32_dpp|permlane|cndmask_b32|cmp_|cmpx_|movrel)/.test(op);

// The plain v_mov_b32 that last wrote a lane move's destination before it: a v_permlane* or
// v_mov_b32_dpp keeps its destination's old value in some lanes, so the compiler fills it first.
function oldMovs(insts) {
  let n = 0;
  insts.forEach((ins, i) => {
    if (!/^v_(permlane|mov_b32_dpp)/.test(ins.op)) return;
    const dst = vgprs(ins.ops[0]);
    for (let j = i - 1; j >= 0 && dst.length; j--) {
      const w = insts[j];
      if (isStore(w.op) || !vgprs(w.ops[0]).some((r) => dst.includes(r))) continue;
      if (isMov(w)) n++;
      break;
    }
  });
  return n;
}

// The longest chain of v_xor3, each reading the one before. A v_xor passes a chain on without
// adding to it; anything else that writes a register ends the chain there.
function xor3Chain(insts) {
  const depth = new Map();
  let longest = 0;
  for (const ins of insts) {
    const xor3 = ins.op === 'v_xor3_b32', xor = /^v_xor_b32/.test(ins.op);
    if (xor3 || xor) {
      let d = 0;
      for (const src of ins.ops.slice(1)) for (const r of vgprs(src)) d = Math.max(d, depth.get(r) || 0);
      if (xor3) d++;
      for (const r of vgprs(ins.ops[0])) depth.set(r, d);
      longest = Math.max(longest, d);
    } else if (/^v_/.test(ins.op) || /_load/.test(ins.op)) {
      if (!isStore(ins.op)) for (const r of vgprs(ins.ops[0])) depth.delete(r);
    }
  }
  return longest;
}

// Counts over the whole function, and over the slab loop (slab*, the keep-step opcodes and the
// stores).
function count(insts) {
  const slab = insts.filter((i) => i.slab);
  const n = (re, list = insts) => list.filter((i) => re.test(i.op)).length;
  const stores32 = slab.filter((i) => /^global_store_b32$/.test(i.op));
  const oldMov = oldMovs(insts);
  return {
    valu: n(/^v_/),
    readout: insts.filter((i) => isReadout(i.op)).length + oldMov,
    oldMov,
    xor3: n(/^v_xor3_b32$/),
    xor: n(/^v_xor_b32/),
    movDpp: n(/^v_mov_b32_dpp/),
    dpp: insts.filter(isDpp).length,
    permlane: n(/^v_permlanex16/),
    chain: xor3Chain(insts),
    store128: n(/^global_store_b128$/),
    slabInsts: slab.length,
    slabValu: n(/^v_/, slab),
    slabPermlane: n(/^v_permlanex16/, slab),
    cndmask: n(/^v_cndmask_b32/, slab),
    cmp: n(/^v_cmpx?_/, slab),
    movrel: n(/^v_movrel/, slab),
    mov: slab.filter(isMov).length,
    store32: stores32.length,
    // The SGPR-base form names an SGPR pair as its third operand; the other form says "off".
    store32Saddr: stores32.filter((i) => /^s\[\d+:\d+\]$/.test(i.ops[2] || '')).length,
    addCoSlab: n(/^v_add_co/, slab),
    storecntSlab: n(/^s_wait_storecnt/, slab),
  };
}

// The fold's line of -Rpass-analysis=kernel-resource-usage remarks.
function resources(remarks, name) {
  const out = {};
  let cur = null;
  for (const line of remarks.replace(/\r\n/g, '\n').split('\n')) {
    const f = /remark: Function Name: (\S+)/.exec(line);
    if (f) { cur = f[1]; continue; }
    if (cur !== name) continue;
    const v = (re) => { const m = re.exec(line); return m ? Number(m[1]) : undefined; };
    const set = (k, x) => { if (x !== undefined) out[k] = x; };
    set('vgprs', v(/remark:\s+VGPRs: (\d+)/));
    set('scratch', v(/remark:\s+ScratchSize \[bytes\/lane\]: (\d+)/));
    set('vgprSpill', v(/remark:\s+VGPRs Spill: (\d+)/));
    set('sgprSpill', v(/remark:\s+SGPRs Spill: (\d+)/));
    set('lds', v(/remark:\s+LDS Size \[bytes\/block\]: (\d+)/));
    set('occupancy', v(/remark:\s+Occupancy \[waves\/SIMD\]: (\d+)/));
  }
  return out;
}

// Rows of [check, base, reg, store, ok]; ok is true, false, or null for a record.
function gates(c, r) {
  const rows = [];
  const add = (name, f, ok) => rows.push([name, f('base'), f('reg'), f('store'), ok]);
  // Readout copies in the function and in the slab loop.
  const N = Math.round(c.base.permlane / 4), NL = Math.round(c.base.slabPermlane / 4);
  const per = (x, n) => (n ? (x / n).toFixed(1) : '?');
  const both = (pred) => pred('reg') && pred('store');
  add('slab loop (the loop with the v_wmma) found', (b) => (c[b].slabInsts ? 'yes' : 'no'),
    BUILDS.every((b) => c[b].slabInsts > 0));
  add('copies, from base: function (N) / slab loop', (b) => (b === 'base' ? `${N} / ${NL}` : ''),
    N > 0 && NL > 0 ? null : false);
  add('scratch bytes / spills (0)', (b) => `${r[b].scratch} / ${(r[b].vgprSpill || 0) + (r[b].sgprSpill || 0)}`,
    BUILDS.every((b) => r[b].scratch === 0 && !r[b].vgprSpill && !r[b].sgprSpill));
  add(`VGPRs (<= ${GATE.vgprsHard}; target ${GATE.vgprsTarget})`, (b) => r[b].vgprs,
    BUILDS.every((b) => r[b].vgprs !== undefined && r[b].vgprs <= GATE.vgprsHard));
  add('LDS bytes (as base)', (b) => r[b].lds, both((b) => r[b].lds === r.base.lds));
  add('occupancy, waves a SIMD', (b) => r[b].occupancy, null);
  add(`readout VALU a copy (base: record; DPP unfolded at ${GATE.baseUnfolded}+)`, (b) => per(c[b].readout, N), null);
  add('  of them, v_mov_b32 filling a lane move', (b) => c[b].oldMov, null);
  add('v_mov_b32_dpp', (b) => c[b].movDpp, null);
  add('DPP ops (two-lane: N)', (b) => c[b].dpp, both((b) => c[b].dpp === N));
  add('v_permlanex16 (two-lane: 0)', (b) => c[b].permlane, both((b) => c[b].permlane === 0));
  add(`slab-loop VALU (two-lane: <= base - ${GATE.savePerCopy} a copy)`, (b) => c[b].slabValu,
    NL > 0 && both((b) => c[b].slabValu <= c.base.slabValu - GATE.savePerCopy * NL));
  add('  saved a copy', (b) => (b === 'base' ? '' : per(c.base.slabValu - c[b].slabValu, NL)), null);
  add(`longest dependent v_xor3 chain (two-lane: <= ${GATE.xor3Chain})`, (b) => c[b].chain,
    both((b) => c[b].chain <= GATE.xor3Chain));
  const keep = (b) => c[b].movrel + c[b].cndmask + c[b].cmp + c[b].mov;
  add(`keep step a copy, reg less store (<= ${GATE.keepPerCopy})`,
    (b) => (b === 'reg' ? per(keep('reg') - keep('store'), NL) : ''),
    NL > 0 && keep('reg') - keep('store') <= GATE.keepPerCopy * NL);
  add('  slab loop: v_movrel / v_cndmask / v_cmp / v_mov', (b) =>
    `${c[b].movrel} / ${c[b].cndmask} / ${c[b].cmp} / ${c[b].mov}`, null);
  add('slab-loop global_store_b32: all / SGPR base (store: all)', (b) => `${c[b].store32} / ${c[b].store32Saddr}`,
    c.store.store32 > 0 && c.store.store32Saddr === c.store.store32);
  add('global_store_b128', (b) => c[b].store128, null);
  add('slab-loop v_add_co* (store: <= base)', (b) => c[b].addCoSlab, c.store.addCoSlab <= c.base.addCoSlab);
  add('slab-loop s_wait_storecnt* (store: 0)', (b) => c[b].storecntSlab, c.store.storecntSlab === 0);
  add('all VALU (record)', (b) => c[b].valu, null);
  return rows;
}

function table(target, rows) {
  const w = Math.max(...rows.map((x) => x[0].length));
  const text = (v) => String(v === undefined ? '?' : v);
  const cw = Math.max(11, ...rows.map((x) => Math.max(...x.slice(1, 4).map((v) => text(v).length + 1))));
  const cell = (v) => text(v).padStart(cw);
  const lines = [`${target}: ${FOLD}`, `  ${''.padEnd(w)} ${cell('base')} ${cell('reg')} ${cell('store')}`];
  for (const [name, b, g, s, ok] of rows) {
    const mark = ok === null ? '' : ok ? '  ok' : '  MISS';
    lines.push(`  ${name.padEnd(w)} ${cell(b)} ${cell(g)} ${cell(s)}${mark}`);
  }
  return lines.join('\n');
}

function censusTarget(dir, target) {
  const c = {}, r = {};
  for (const b of BUILDS) {
    const asm = fs.readFileSync(path.join(dir, `${target}-${b}.s`), 'utf8');
    const fn = cutFunction(asm, FOLD);
    if (!fn) throw new Error(`${target}-${b}.s has no ${FOLD}`);
    c[b] = count(parse(fn));
    r[b] = resources(fs.readFileSync(path.join(dir, `${target}-${b}.txt`), 'utf8'), FOLD);
  }
  return gates(c, r);
}

function main(argv) {
  const [dir, ...targets] = argv;
  if (!dir || !targets.length) {
    console.error('usage: node readout-census.js DIR gfx1200 [gfx1201 ...]');
    return 2;
  }
  let misses = 0;
  for (const t of targets) {
    const rows = censusTarget(dir, t);
    console.log(table(t, rows));
    misses += rows.filter((x) => x[4] === false).length;
  }
  console.log(misses ? `census: ${misses} miss${misses > 1 ? 'es' : ''}` : 'census: every gate passed');
  return misses ? 1 : 0;
}

module.exports = { cutFunction, parse, count, resources, gates, table, xor3Chain, oldMovs, GATE };

if (require.main === module) process.exitCode = main(process.argv.slice(2));
