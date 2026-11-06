'use strict';
// Temporary, for issue #287 item 2 (exp/287-2 only; removed before merge). Reads one fold's device assembly (as
// test/variant-check.sh extracts it: comments and directives dropped, labels as .LBB_N) and prints what the slab
// loop issues: WMMAs, VALU, SALU, waits, global or buffer loads and LDS ops, per block and for the loop, and the
// VALU mnemonics in the loop's WMMA blocks. The slab loop is the backward branch whose range holds the most WMMAs.
const fs = require('fs');

const [file, tag] = process.argv.slice(2);
const lines = fs.readFileSync(file, 'utf8').split('\n').map((l) => l.trim()).filter(Boolean);
const blocks = [{ label: 'entry', ins: [] }];
for (const l of lines) {
  if (/^\.LBB_\d+:$/.test(l)) blocks.push({ label: l.slice(0, -1), ins: [] });
  else blocks[blocks.length - 1].ins.push(l);
}
const kind = (m) => {
  if (/^v_wmma/.test(m)) return 'W';
  if (/^v_/.test(m)) return 'V';
  if (/^(global|buffer)_load/.test(m)) return 'G';
  if (/^(global|buffer)_store/.test(m)) return 'GS';
  if (/^ds_/.test(m)) return 'D';
  if (/^s_(waitcnt|wait_|delay_alu|nop|clause)/.test(m)) return 'WT';
  if (/^s_(cbranch|branch|barrier|endpgm|setpc)/.test(m)) return 'BR';
  if (/^s_/.test(m)) return 'S';
  return 'O';
};
const K = ['W', 'V', 'S', 'WT', 'BR', 'G', 'GS', 'D', 'O'];
const count = (bs) => {
  const c = Object.fromEntries(K.map((k) => [k, 0]));
  for (const b of bs) for (const i of b.ins) c[kind(i.split(/\s+/)[0])]++;
  return c;
};
const fmt = (c) => K.map((k) => `${k} ${c[k]}`).join(', ');
const index = new Map(blocks.map((b, i) => [b.label, i]));
let loop = null;
blocks.forEach((b, i) => {
  for (const ins of b.ins) {
    const m = /^s_(cbranch_\w+|branch)\s+(\.LBB_\d+)/.exec(ins);
    if (!m || !index.has(m[2]) || index.get(m[2]) > i) continue;
    const range = blocks.slice(index.get(m[2]), i + 1);
    const w = count(range).W;
    if (!loop || w > loop.w) loop = { from: index.get(m[2]), to: i, w, range };
  }
});
console.log(`${tag}: whole fold: ${fmt(count(blocks))}`);
if (!loop) { console.log(`${tag}: no backward branch found`); process.exit(0); }
const wblocks = loop.range.filter((b) => count([b]).W > 0);
const c = count(loop.range), cw = count(wblocks), slabs = c.W / 32;
console.log(`${tag}: slab loop ${blocks[loop.from].label}..${blocks[loop.to].label}, ${loop.range.length} blocks, ` +
  `${slabs} slabs of 32 WMMAs: ${fmt(c)}`);
console.log(`${tag}: its WMMA blocks only (no readout): ${fmt(cw)}; VALU a slab ${(cw.V / slabs).toFixed(1)}, ` +
  `SALU a slab ${(cw.S / slabs).toFixed(1)}, loads a slab ${(cw.G / slabs).toFixed(1)}`);
for (const b of loop.range) console.log(`${tag}:   ${b.label}: ${fmt(count([b]))}`);
const hist = {};
for (const b of wblocks) for (const i of b.ins) { const m = i.split(/\s+/)[0]; if (kind(m) === 'V') hist[m] = (hist[m] || 0) + 1; }
console.log(`${tag}: VALU in the WMMA blocks: ${Object.entries(hist).sort((a, b) => b[1] - a[1]).map(([m, n]) => `${m} ${n}`).join(', ') || 'none'}`);
const loads = [];
for (const b of wblocks) for (const i of b.ins) if (kind(i.split(/\s+/)[0]) === 'G') loads.push(i);
console.log(`${tag}: loads in the WMMA blocks: ${loads.join(' | ')}`);
