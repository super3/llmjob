'use strict';

// The counting half of readout-census.sh: reads the assembly and resource remarks that script
// wrote for each target and build (base, v2, v2w, and drop on gfx12), cuts out the target's RDNA
// fold, counts its instructions, and checks them against the static gates of the readout v2
// plan (round 4 item 4). Prints one table a target and exits 1 on any miss.
//
//   node readout-census.js DIR gfx1100 [gfx1200 ...]
//
// DIR holds <target>-<build>.s (hip-clang --cuda-device-only -S) and <target>-<build>.txt (its
// -Rpass-analysis=kernel-resource-usage remarks). The counts come from the text of one function,
// so they are static counts: an instruction in a loop counts once.
//
// Terms:
//   slab loop    the loop that holds the v_wmma, from its label to the branch back to it: the hot
//                path. Code that runs once a tile (the zeroing, the read-back, the tile's last
//                readout after the loop) can't sink or save a build there.
//   copies (N)   how many copies of the readout the compiler made: base's v_permlanex16 count / 4
//                (four words a copy, one half swap each). Expect 3 in the function (the two PF2
//                parities and the tail) and 2 in the slab loop (NL).
//   slab VALU    every v_ instruction in the slab loop. Everything there but the readout is the
//                same in every build, so a v2 build's difference from base is what its readout
//                saves, whatever instructions the compiler picked.
//   selects      v_cndmask, v_cmp and v_movrel in the slab loop. Base's readout has 8 v_cndmask a
//                copy (four words, low and high); v2 has no select at all.
//   LDS XORs     ds_xor_b64 (v2: 2 a copy, one a block) or ds_xor_b32 (v2w: 4 a copy). A returning
//                form (ds_xor_rtn_*), a compare-and-swap loop (ds_cmp*) or a flat or global atomic
//                means the compiler didn't see a plain LDS XOR whose result is unused.
//   DS waits     s_wait_dscnt in any form (gfx12) or s_waitcnt with lgkmcnt (gfx11) in the slab
//                loop: v2 must add none. The slab barrier's own fence already waits for the XORs.
//   data movs    plain v_mov_b32 that last wrote an LDS XOR's data registers: more than about one a
//                copy means the 8-byte pair isn't formed where the tree leaves it.

const fs = require('fs');
const path = require('path');

// The plan's gates (readout-v2/plan.md, section 5).
const GATE = {
  vgprsGfx12: 216,   // allocates 216 at 6 waves a SIMD; more and the fold loses a wave
  vgprsGfx11: 240,   // gfx11 sits at 240 today and loses a wave above it
  vgprsTarget: 192,  // what 8 waves would need on gfx12; recorded, not gated
  savePerCopy: 20,   // slab-loop VALU at most base - 20 a copy
  selectsPerCopy: 8, // base's v_cndmask a copy, all of which v2 drops
};

const foldOf = (target) => (/^gfx12/.test(target) ? 'pearl_amd_fold_wmma12' : 'pearl_amd_fold_wmma11');
const buildsOf = (target) => (/^gfx12/.test(target) ? ['base', 'v2', 'v2w', 'drop'] : ['base', 'v2', 'v2w']);

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
  // Loops: from a label to a branch back to it. The slab loop is any of them that holds a v_wmma.
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

const isStore = (op) => /^(global|flat|buffer|scratch)_store|^ds_(store|write)|^ds_xor_b(32|64)$/.test(op);
const isDpp = (ins) => /_dpp$|_dpp_/.test(ins.op) || /\b(quad_perm|row_\w+|dpp8):/.test(ins.text);
const isMov = (ins) => /^v_mov_b32(_e32|_e64)?$/.test(ins.op) && !isDpp(ins);
const isLdsXor = (op) => /^ds_xor_b(32|64)$/.test(op);
const isDsWait = (ins) => /^s_wait_\w*dscnt|^s_waitcnt_lgkmcnt/.test(ins.op)
  || (ins.op === 's_waitcnt' && /lgkmcnt/.test(ins.text));
const isReadout = (op) => /^v_(xor3_b32|xor_b32|mov_b32_dpp|permlane|cndmask_b32|cmp_|cmpx_|movrel)/.test(op);

// The plain v_mov_b32 that last wrote each data register of an LDS XOR (a ds_xor's operands are
// address, data). A lane move or select before it isn't counted: only a copy is.
function dataMovs(insts) {
  let n = 0;
  insts.forEach((ins, i) => {
    if (!isLdsXor(ins.op)) return;
    for (const reg of vgprs(ins.ops[1])) {
      for (let j = i - 1; j >= 0; j--) {
        const w = insts[j];
        if (isStore(w.op) || !vgprs(w.ops[0]).includes(reg)) continue;
        if (isMov(w)) n++;
        break;
      }
    }
  });
  return n;
}

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

// Counts over the whole function, and over the slab loop (slab*).
function count(insts) {
  const slab = insts.filter((i) => i.slab);
  const n = (re, list = insts) => list.filter((i) => re.test(i.op)).length;
  return {
    valu: n(/^v_/),
    readout: insts.filter((i) => isReadout(i.op)).length + oldMovs(insts),
    dpp: insts.filter(isDpp).length,
    permlane: n(/^v_permlane/),
    xor64: n(/^ds_xor_b64$/),
    xor32: n(/^ds_xor_b32$/),
    xorRtn: n(/^ds_xor_rtn/),
    dsCmp: n(/^ds_cmp/),
    memAtomic: n(/^(flat|global|buffer)_atomic/),
    dataMov: dataMovs(insts),
    store128: n(/^global_store_b128$/),
    store32: n(/^global_store_b32$/),
    slabInsts: slab.length,
    slabValu: n(/^v_/, slab),
    slabPermlane: n(/^v_permlanex16/, slab),
    slabXor: n(/^ds_xor_b(32|64)$/, slab),
    slabSelects: n(/^v_(cndmask_b32|cmpx?_|movrel)/, slab),
    slabDsWait: slab.filter(isDsWait).length,
  };
}

// The fold's lines of -Rpass-analysis=kernel-resource-usage remarks.
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

// Rows of [check, [value a build], ok]; ok is true, false, or null for a record.
function gates(c, r, target) {
  const builds = Object.keys(c);
  const v2s = ['v2', 'v2w'];
  const rows = [];
  const add = (name, f, ok) => rows.push([name, builds.map(f), ok]);
  const all = (pred, list = builds) => list.every(pred);
  // Readout copies in the function and in the slab loop, from base.
  const N = Math.round(c.base.permlane / 4), NL = Math.round(c.base.slabPermlane / 4);
  const per = (x, n) => (n ? (x / n).toFixed(1) : '?');
  const vmax = /^gfx12/.test(target) ? GATE.vgprsGfx12 : GATE.vgprsGfx11;
  add('slab loop (the loop with the v_wmma) found', (b) => (c[b].slabInsts ? 'yes' : 'no'),
    all((b) => c[b].slabInsts > 0));
  add('copies, from base: function (N) / slab loop', (b) => (b === 'base' ? `${N} / ${NL}` : ''),
    N > 0 && NL > 0 ? null : false);
  add('scratch bytes / spills (0)', (b) => `${r[b].scratch} / ${(r[b].vgprSpill || 0) + (r[b].sgprSpill || 0)}`,
    all((b) => r[b].scratch === 0 && !r[b].vgprSpill && !r[b].sgprSpill));
  add(`VGPRs (<= ${vmax}${vmax === GATE.vgprsGfx12 ? `; target ${GATE.vgprsTarget}` : ''})`, (b) => r[b].vgprs,
    all((b) => r[b].vgprs !== undefined && r[b].vgprs <= vmax));
  add('VGPRs, v2 against base (no more)', (b) => (b === 'base' ? '' : r[b].vgprs - r.base.vgprs),
    all((b) => r[b].vgprs <= r.base.vgprs, v2s));
  add('LDS bytes (as base)', (b) => r[b].lds, all((b) => r[b].lds === r.base.lds));
  add('occupancy, waves a SIMD (v2: no fewer)', (b) => r[b].occupancy,
    all((b) => r[b].occupancy >= r.base.occupancy, v2s));
  add('DPP ops (v2: 0)', (b) => c[b].dpp, all((b) => c[b].dpp === 0, v2s));
  add('v_permlane* (v2: 0)', (b) => c[b].permlane, all((b) => c[b].permlane === 0, v2s));
  add(`slab-loop selects: v_cndmask, v_cmp, v_movrel (v2: <= base - ${GATE.selectsPerCopy} a copy)`,
    (b) => c[b].slabSelects,
    NL > 0 && all((b) => c[b].slabSelects <= c.base.slabSelects - GATE.selectsPerCopy * NL, v2s));
  add('ds_xor_b64 (v2: 2N; others 0)', (b) => c[b].xor64,
    c.v2.xor64 === 2 * N && all((b) => c[b].xor64 === 0, builds.filter((b) => b !== 'v2')));
  add('ds_xor_b32 (v2w: 4N; others 0)', (b) => c[b].xor32,
    c.v2w.xor32 === 4 * N && all((b) => c[b].xor32 === 0, builds.filter((b) => b !== 'v2w')));
  add('  of them in the slab loop (v2: 2NL, v2w: 4NL)', (b) => c[b].slabXor,
    c.v2.slabXor === 2 * NL && c.v2w.slabXor === 4 * NL);
  add('ds_xor_rtn / ds_cmp* / flat or global atomics (0)',
    (b) => `${c[b].xorRtn} / ${c[b].dsCmp} / ${c[b].memAtomic}`,
    all((b) => c[b].xorRtn === 0 && c[b].dsCmp === 0 && c[b].memAtomic === 0));
  add('  plain v_mov_b32 feeding their data, a copy (record)', (b) => (v2s.includes(b) ? per(c[b].dataMov, N) : ''), null);
  add(`slab-loop VALU (v2: <= base - ${GATE.savePerCopy} a copy)`, (b) => c[b].slabValu,
    NL > 0 && all((b) => c[b].slabValu <= c.base.slabValu - GATE.savePerCopy * NL, v2s));
  add('  saved a copy', (b) => (b === 'base' ? '' : per(c.base.slabValu - c[b].slabValu, NL)), null);
  add('slab-loop DS waits (v2: no more than base)', (b) => c[b].slabDsWait,
    all((b) => c[b].slabDsWait <= c.base.slabDsWait, v2s));
  add('readout VALU a copy (record)', (b) => per(c[b].readout, N), null);
  add('tile-end stores: global_store_b128 / _b32 (record)', (b) => `${c[b].store128} / ${c[b].store32}`, null);
  add('all VALU (record)', (b) => c[b].valu, null);
  return rows;
}

function table(target, builds, rows) {
  const w = Math.max(...rows.map((x) => x[0].length));
  const text = (v) => String(v === undefined || Number.isNaN(v) ? '?' : v);
  const cw = Math.max(9, ...rows.map((x) => Math.max(...x[1].map((v) => text(v).length + 1))));
  const cell = (v) => text(v).padStart(cw);
  const lines = [`${target}: ${foldOf(target)}`, `  ${''.padEnd(w)} ${builds.map(cell).join(' ')}`];
  for (const [name, vals, ok] of rows) {
    const mark = ok === null ? '' : ok ? '  ok' : '  MISS';
    lines.push(`  ${name.padEnd(w)} ${vals.map(cell).join(' ')}${mark}`);
  }
  return lines.join('\n');
}

function censusTarget(dir, target) {
  const fold = foldOf(target), c = {}, r = {};
  for (const b of buildsOf(target)) {
    const asm = fs.readFileSync(path.join(dir, `${target}-${b}.s`), 'utf8');
    const fn = cutFunction(asm, fold);
    if (!fn) throw new Error(`${target}-${b}.s has no ${fold}`);
    c[b] = count(parse(fn));
    r[b] = resources(fs.readFileSync(path.join(dir, `${target}-${b}.txt`), 'utf8'), fold);
  }
  return gates(c, r, target);
}

function main(argv) {
  const [dir, ...targets] = argv;
  if (!dir || !targets.length) {
    console.error('usage: node readout-census.js DIR gfx1100 [gfx1200 ...]');
    return 2;
  }
  let misses = 0;
  for (const t of targets) {
    const rows = censusTarget(dir, t);
    console.log(table(t, buildsOf(t), rows));
    misses += rows.filter((x) => x[2] === false).length;
  }
  console.log(misses ? `census: ${misses} miss${misses > 1 ? 'es' : ''}` : 'census: every gate passed');
  return misses ? 1 : 0;
}

module.exports = { cutFunction, parse, count, resources, gates, table, dataMovs, oldMovs, foldOf, buildsOf, GATE };

if (require.main === module) process.exitCode = main(process.argv.slice(2));
