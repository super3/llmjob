'use strict';
// handsim.js: a model of what the wave64 fold's chunk loop computes, for checking handfold.js's edits without a GPU.
// It walks a loop's instructions in order and follows:
//   - what each A' register holds: the LDS read that filled it (the staging buffer, how many times that buffer had
//     been stored to by then, the offset) and which word of it;
//   - whether that read is known to have arrived when a WMMA uses it. LDS operations return in the order they
//     were issued, and s_wait_dscnt N lets the wave on once at most N are still out, so after a wait every
//     operation but the last N has arrived;
//   - what each B' and staging register holds: the buffer load that filled it (its address, and the value its
//     offset register had, counted in writes to that register).
// The result is, per pass of the loop, the WMMAs as "sum += A' word * B' word" and the stores as "LDS offset <-
// staged load". Two loops that give the same lists compute the same sums from the same bytes and stage the same
// bytes in the same places at the same points between the reads. Register names of operands don't appear in the
// lists, so a loop that renames registers compares equal; a loop that uses a register before its read arrived, or
// after something else wrote it, throws.
//
// It does not model the barrier: that every wave has stored before any wave reads is argued in handfold.js and
// checked on a card by the fold test.

const opOf = (line) => {
  const m = /^\s+([a-z_0-9]+)/.exec(line);
  return m ? m[1] : '';
};

function range(text) {
  const m = /^v\[(\d+):(\d+)\]$/.exec(text) || /^v(\d+)$/.exec(text);
  if (!m) throw new Error(`not a register operand: ${text}`);
  const lo = Number(m[1]);
  const hi = Number(m[2] === undefined ? m[1] : m[2]);
  return Array.from({ length: hi - lo + 1 }, (_, i) => lo + i);
}

const BUFFER = 16384;   // an A' staging buffer in LDS: 256 rows of 64 bytes
const SALU_WRITES = /^s_(add|addk|sub|mov|movk|and|or|xor|lshl|lshr|ashr|mul|cselect|min|max)/;

// before: lines in front of the loop (reads handfold.js moved there). body: the loop. alias: address registers that
// are another register plus a constant, { 166: [164, 1024] }, so a load through v166 and one through v164 with
// offset:1024 name the same bytes.
function simulate({ before = [], body, passes = 2, alias = {} }) {
  const reg = new Map();        // VGPR -> { what, lds }: lds is the index of the LDS read that fills it
  const sver = new Map();       // SGPR -> writes so far
  const stored = new Map();     // staging buffer -> stores so far
  let issued = 0;               // LDS operations issued so far
  let arrived = 0;              // how many of them are known to have arrived
  let pass = { wmma: [], stores: [] };
  const addr = (r, off) => (alias[r] ? `v${alias[r][0]}+${alias[r][1] + off}` : `v${r}+${off}`);
  const held = (r) => (reg.has(r) ? reg.get(r).what : `entry(v${r})`);
  const step = (line) => {
    const op = opOf(line);
    if (!op) return;
    const args = line.split(';')[0].trim().slice(op.length).trim();
    const off = Number((/offset:(\S+)/.exec(args) || [0, 0])[1]);
    const first = args.split(',')[0].trim();
    if (op === 'ds_load_b128') {
      const a = args.split(',')[1].trim().split(' ')[0];
      const buf = Math.floor(off / BUFFER);
      const what = `lds[${a} buffer ${buf} after ${stored.get(buf) || 0} stores, +${off % BUFFER}]`;
      range(first).forEach((r, j) => reg.set(r, { what: `${what}.${j}`, lds: issued }));
      issued++;
    } else if (op === 'ds_store_b128') {
      const src = range(args.split(',')[1].trim().split(' ')[0]).map(held);
      if (src.some((h) => !h)) throw new Error(`a store of a register something else wrote: ${line.trim()}`);
      const buf = Math.floor(off / BUFFER);
      stored.set(buf, (stored.get(buf) || 0) + 1);
      pass.stores.push(`${first}+${off} <- ${src.join(' ')}`);
      issued++;
    } else if (op === 'ds_xor_b32') {
      issued++;
    } else if (op === 'buffer_load_b128') {
      const m = /^(v\[\d+:\d+\]),\s*v(\d+),\s*(s\[\d+:\d+\]),\s*(\S+)\s+offen/.exec(args);
      if (!m) throw new Error(`a buffer load of another form: ${line.trim()}`);
      const what = `mem[${m[3]} ${addr(Number(m[2]), off)} @${m[4]} write ${sver.get(m[4]) || 0}]`;
      range(m[1]).forEach((r, j) => reg.set(r, { what: `${what}.${j}` }));
    } else if (op === 's_wait_dscnt' || op === 's_wait_loadcnt_dscnt') {
      arrived = Math.max(arrived, issued - (Number(args) & 0x3f));
    } else if (op.startsWith('v_wmma')) {
      const [a, b] = args.split(',').slice(1, 3).map((t) => t.trim());
      const A = reg.get(range(a)[0]);
      if (!A || !A.what || !A.what.startsWith('lds[')) throw new Error(`a WMMA whose A' register holds no LDS read: ${line.trim()}`);
      if (A.lds >= arrived) throw new Error(`a WMMA uses an A' read that may not have arrived: ${line.trim()}`);
      const B = held(range(b)[0]);
      if (!B) throw new Error(`a WMMA whose B' register something else wrote: ${line.trim()}`);
      pass.wmma.push(`${first} += ${A.what} * ${B}`);
    } else if (/^v_/.test(op)) {
      range(first).forEach((r) => reg.set(r, { what: null }));   // any other VALU: no operand bytes there now
    } else if (SALU_WRITES.test(op)) {
      sver.set(first, (sver.get(first) || 0) + 1);
    }
  };
  before.forEach(step);
  const out = [];
  for (let n = 0; n < passes; n++) {
    pass = { wmma: [], stores: [] };
    body.forEach(step);
    // the order of a pass's WMMAs may differ between two loops; what each adds to which sum may not
    out.push({ wmma: pass.wmma.slice().sort(), stores: pass.stores });
  }
  return out;
}

module.exports = { simulate, opOf };
