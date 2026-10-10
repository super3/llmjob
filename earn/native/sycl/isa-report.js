// node isa-report.js <dump-dir>
//
// Summarises the GPU assembly IGC wrote during `./build.sh --dump <dump-dir>`:
// one row per kernel and platform with its SIMD width, register mode, DPAS
// count, spill size and instruction count. The fold rows are the ones that
// matter: a hardware XMX fold must show DPAS instructions (32 a chunk) and
// should show no spill. The same check CI makes on ptxas output for CUDA.
'use strict';
const fs = require('fs');
const path = require('path');

const dir = process.argv[2];
if (!dir) { console.error('usage: node isa-report.js <dump-dir>'); process.exit(2); }

// The kernel, from its mangled lambda name.
function shortName(k) {
  const x = /fold_xmxILi(\d+)ELi(\d+)ELb(\d)E/.exec(k);
  if (x) return 'fold_xmx<sg' + x[1] + ',rb' + x[2] + (x[3] === '1' ? ',hw>' : ',emu>');
  for (const n of ['fold_dot', 'materialize16', 'chunk_cvs', 'parent_layer', 'gen_dense', 'gen_perm', 'gen_operand', 'stamp_a']) {
    if (k.includes(n.length + n)) return n;
  }
  return k.length > 40 ? k.slice(0, 40) + '...' : k;
}

const rows = [];
for (const f of fs.readdirSync(dir).filter((x) => x.endsWith('.asm')).sort()) {
  const text = fs.readFileSync(path.join(dir, f), 'utf8');
  const head = (re) => { const m = re.exec(text); return m ? m[1] : ''; };
  const kernel = head(/^\/\/\.kernel (\S+)/m);
  if (!kernel) continue;
  rows.push({
    platform: head(/^\/\/\.platform (\S+)/m),
    kernel: shortName(kernel),
    simd: (/_simd(\d+)_/.exec(f) || [])[1] || '?',
    grf: head(/numGRF=(\d+)/),
    dpas: (text.match(/^\s*(?:\(\S+\)\s*)?dpas/gm) || []).length,
    spill: head(/^\/\/\.spill size (\d+)/m) || '0',
    insts: head(/^\/\/\.instCount (\d+)/m),
  });
}
const cols = ['platform', 'kernel', 'simd', 'grf', 'dpas', 'spill', 'insts'];
const width = cols.map((c) => Math.max(c.length, ...rows.map((r) => String(r[c]).length)));
const line = (vals) => vals.map((v, i) => String(v).padEnd(width[i])).join('  ');
console.log(line(cols));
for (const r of rows) console.log(line(cols.map((c) => r[c])));
