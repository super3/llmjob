#!/usr/bin/env node
'use strict';
// The clock stamps of a PEARL_AMD_STAMP fold (pearl_amd_fold.h), summed up in a few lines short enough for
// Salad's log. pearl_amd_foldtest writes them (its stamps argument); sweep.sh's SWEEP_STAMP runs both.
//
//   node stamp-summary.js summary <stamps file>
//   node stamp-summary.js gates <stamp build's file> <file of a stamp build with PEARL_AMD_ABLATE=8>
//
// summary prints:
//   run      what was read, and what was dropped.
//   first    the blocks that started before any block ended (the launch's first wave), then
//   rest     the others: the clock in MHz (a wave's cycles over its REALTIME ticks), the slab's cycles (each
//            block's mean), the share of the slab loop spent at the barrier and that wait a slab, the same for
//            the memory wait at the release fence just before it, and the shares of a block spent before the
//            slab loop and after it. A block is read from its wave 0.
//   counter  how far apart the waves of one block read the counter as they leave their last barrier, which
//            they leave together: waves on one SIMD, and on different SIMDs. The ISA says the counter isn't
//            synchronised across SIMDs, so the phases below compare waves on one SIMD only.
//   counter across blocks  the premise those phases rest on, checked between blocks: for each wave that
//            started on a SIMD after another wave there ended, the counter's gap less REALTIME's gap in
//            cycles. One free-running counter a SIMD gives tens of cycles; a counter of each wave, or one reset
//            for each block, gives noise.
//   pairs    blocks that shared a WGP at the same time, and how far apart they started.
//   phase    for those pairs, at the first barrier exit after the counter's bit 16 (then bit 19) changed:
//            how far apart the two blocks' barriers are, from waves on one SIMD that crossed the same
//            boundary, folded into one slab (0 to half a slab). Bit 16's crossing is near a block's start;
//            bit 19's falls anywhere in it. Two crossings of one boundary are under a slab apart in time, so
//            a pair whose crossings are more than 3 slabs apart on REALTIME is dropped, and counted unless
//            they are a whole number of gfx11's 2^20-cycle wraps apart (another boundary, the same value).
// gates prints item 14's and 15's readings (benchmark.md, item 13) from two builds on one PC:
//   D        the stall a start skew could hide: the stamp build's median barrier wait a slab, plus what its
//            LDS fragment loads cost a slab. That is its median work a slab (the loop less barrier and memory
//            wait) less the abl8 stamp build's, so a change in barrier wait isn't counted twice. A second D
//            adds the memory wait. All from "rest".
//   item 14  kill if the median phase (bit 19's, or bit 16's with too few pairs) is larger than D. No reading if
//            the counter across blocks is off REALTIME by more than half a slab.
//   item 15  kill if abl8 saves under 3% of the stamp build's slab, whole slabs (barrier wait and all).
const fs = require('fs');

const FILE_MAGIC = 0x31545350;   // "PST1"
const WAVE_MAGIC = 0x53540002;   // "ST", version 2
const HEAD = 16;
const MIN_PAIRS = 30;
// A crossing pair more than this many slabs apart on REALTIME is not one boundary.
const MAX_SLABS_APART = 3;
// Successions on a SIMD are compared only when the gap is under this many cycles: a quarter of gfx11's wrap.
const MAX_GAP = 2 ** 18;

// The file: its header and every wave record written. A record left zero is missing.
function load(buf) {
  const u = (i) => buf.readUInt32LE(4 * i);
  if (buf.length < 4 * HEAD || u(0) !== FILE_MAGIC) throw new Error('not a stamps file');
  const head = {
    words: u(1), waves: u(2), blocks: u(3), gridX: u(4), wallKhz: u(5), k: u(6), launches: u(7), m: u(8),
    cols: u(9),
  };
  if (head.words !== 16) throw new Error('stamps file has ' + head.words + ' words a wave, not 16');
  if (buf.length < 4 * (HEAD + head.blocks * head.waves * head.words)) throw new Error('stamps file is short');
  const waves = [];
  let missing = 0, misplaced = 0;
  for (let b = 0; b < head.blocks; b++) {
    for (let w = 0; w < head.waves; w++) {
      const o = HEAD + (b * head.waves + w) * head.words;
      if (u(o) !== WAVE_MAGIC) { missing++; continue; }
      // Word 1: the wave (bits 7:0), the counter's bits (13:8), the crossings seen (15:14), the block's low 16.
      const w1 = u(o + 1);
      const r = {
        wave: w1 & 0xff, bits: (w1 >>> 8) & 63, seen: (w1 >>> 14) & 3, block: b,
        hw0: u(o + 2), hw1: u(o + 3), cyc0: u(o + 4), rt0: u(o + 5), cyc1: u(o + 6), rt1: u(o + 7),
        pro: u(o + 8), loop: u(o + 9), epi: u(o + 10), bar: u(o + 11), x16: u(o + 12), x19: u(o + 13),
        slabs: u(o + 14), mem: u(o + 15),
      };
      if (w1 >>> 16 !== (b & 0xffff) || r.wave !== w) { misplaced++; continue; }
      waves.push(r);
    }
  }
  return { head, waves, missing, misplaced };
}

// b - a on a counter of `bits` bits, as the fold's pearl_amd_cycles_since: right while they are less than a
// wrap apart. And the distance between two readings, whichever is first.
function since(a, b, bits) {
  return bits >= 32 ? (b - a) >>> 0 : (b - a) & (2 ** bits - 1);
}
function apart(a, b, bits) {
  const d = since(a, b, bits);
  return Math.min(d, 2 ** bits - d);
}

// HW_ID1: SIMD_ID 9:8, WGP_ID 13:10, SA_ID 16, SE_ID 20:18. A WGP is its engine, array and number.
const wgpOf = (h) => (((h >>> 18) & 7) << 5) | (((h >>> 16) & 1) << 4) | ((h >>> 10) & 15);
const simdOf = (h) => (wgpOf(h) << 2) | ((h >>> 8) & 3);

function quantile(sorted, q) {
  if (!sorted.length) return NaN;
  return sorted[Math.min(sorted.length - 1, Math.round(q * (sorted.length - 1)))];
}
function spread(values) {
  const s = values.filter(Number.isFinite).sort((a, b) => a - b);
  return { n: s.length, p10: quantile(s, 0.1), med: quantile(s, 0.5), p90: quantile(s, 0.9) };
}

// One position's blocks, from their wave 0.
function position(blocks) {
  const w = blocks.map((b) => b.lead);
  const sum = (f) => w.reduce((s, x) => s + f(x), 0);
  const cycles = sum((x) => x.cycles), loop = sum((x) => x.loop);
  return {
    n: w.length,
    mhz: spread(w.map((x) => x.mhz)),
    slab: spread(w.map((x) => x.slab)),
    work: spread(w.map((x) => x.work)),
    barSlab: spread(w.map((x) => x.barSlab)),
    memSlab: spread(w.map((x) => x.memSlab)),
    bar: loop ? sum((x) => x.bar) / loop : NaN,
    mem: loop ? sum((x) => x.mem) / loop : NaN,
    pro: cycles ? sum((x) => x.pro) / cycles : NaN,
    epi: cycles ? sum((x) => x.epi) / cycles : NaN,
  };
}

// Where a wave's counter stood at `x`, put on REALTIME from its start (ticks).
const tickOf = (w, x) => w.start + since(w.cyc0, x, w.bits) / w.cpt;

// A pair of blocks on one WGP at the same time: for a boundary of bit `bit`, the first two waves on one SIMD
// that crossed it, their counters agreeing above the bit and their crossings at most MAX_SLABS_APART slabs
// apart on REALTIME. Then how far apart their barriers are within one slab, and how far apart REALTIME puts
// the two crossings. gfx11's 20-bit counter repeats a boundary every 2^20 cycles: crossings a whole number of
// wraps apart on REALTIME are two boundaries, and are skipped. `rejected` if waves agreed above the bit but
// REALTIME put them neither together nor whole wraps apart.
function phaseOf(A, B, bit) {
  const flag = bit === 16 ? 1 : 2, key = 'x' + bit;
  const slab = (A.lead.slab + B.lead.slab) / 2, near = MAX_SLABS_APART * slab;
  let rejected = false;
  for (const a of A.waves) {
    if (!(a.seen & flag)) continue;
    for (const b of B.waves) {
      if (!(b.seen & flag) || simdOf(b.hw0) !== simdOf(a.hw0)) continue;
      if (Math.floor(a[key] / 2 ** bit) !== Math.floor(b[key] / 2 ** bit)) continue;
      const gap = Math.abs(tickOf(a, a[key]) - tickOf(b, b[key])) * (a.cpt + b.cpt) / 2;
      if (!(gap <= near)) {
        const wrap = 2 ** a.bits, laps = Math.round(gap / wrap);
        if (!(laps >= 1 && Math.abs(gap - laps * wrap) <= near)) rejected = true;
        continue;
      }
      const d = apart(a[key], b[key], a.bits) % slab;
      return { phase: Math.min(d, slab - d), gap };
    }
  }
  return rejected ? { rejected } : null;
}

// For each wave that started on a SIMD after another wave there ended, the counter's gap between the two
// less REALTIME's, in cycles, when the gap is short enough to read on gfx11's 20 bits.
function across(waves) {
  const bySimd = new Map();
  for (const w of waves) {
    const k = simdOf(w.hw0);
    if (!bySimd.has(k)) bySimd.set(k, []);
    bySimd.get(k).push(w);
  }
  const off = [];
  for (const list of bySimd.values()) {
    list.sort((x, y) => x.start - y.start);
    for (const A of list) {
      // The first wave to start once A has ended.
      let lo = 0, hi = list.length;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (list[mid].start < A.end) lo = mid + 1; else hi = mid;
      }
      if (lo === list.length) continue;
      const B = list[lo];
      const gap = (B.start - A.end) * (A.cpt + B.cpt) / 2;
      if (!(gap < MAX_GAP)) continue;
      let d = since(A.cyc1, B.cyc0, A.bits);
      if (d >= 2 ** (A.bits - 1)) d -= 2 ** A.bits;
      off.push(Math.abs(d - gap));
    }
  }
  return spread(off);
}

// Why a summary has no blocks to read.
function why(s, stamped, leadless) {
  if (!s.read) {
    return 'no stamps: no wave stored a record in its place (' + s.missing + ' slots empty, ' + s.misplaced
      + ' misplaced)';
  }
  if (!stamped) return 'no stamps: this fold has none on this card (only gfx11 and gfx12 do)';
  return 'no usable blocks: ' + s.moved + ' waves dropped (HW_ID1 changed), ' + leadless
    + ' blocks without their wave 0';
}

function summarize(data) {
  const tickMhz = (data.head.wallKhz || 100000) / 1000;
  const stamped = data.waves.filter((w) => w.bits > 0);
  const waves = stamped.filter((w) => w.hw0 === w.hw1);
  const s = {
    head: data.head, tickMhz, read: data.waves.length, missing: data.missing, misplaced: data.misplaced,
    unstamped: data.waves.length - stamped.length, moved: stamped.length - waves.length,
    bits: stamped.length ? stamped[0].bits : 0, blocks: 0,
  };
  // REALTIME from the first start, in ticks: a launch is far shorter than 2^31 ticks (21 s at 100 MHz).
  const ref = waves.length ? waves[0].rt0 : 0;
  const base = waves.reduce((m, w) => Math.min(m, (w.rt0 - ref) | 0), 0);
  for (const w of waves) {
    w.start = ((w.rt0 - ref) | 0) - base;
    w.ticks = (w.rt1 - w.rt0) >>> 0;
    w.end = w.start + w.ticks;
    w.cycles = w.pro + w.loop + w.epi;
    w.cpt = w.ticks ? w.cycles / w.ticks : NaN;
    w.mhz = w.cpt * tickMhz;
    w.slab = w.slabs ? w.loop / w.slabs : NaN;
    w.barSlab = w.slabs ? w.bar / w.slabs : NaN;
    w.memSlab = w.slabs ? w.mem / w.slabs : NaN;
    w.work = w.slabs ? (w.loop - w.bar - w.mem) / w.slabs : NaN;
  }
  const byBlock = new Map();
  for (const w of waves) {
    if (!byBlock.has(w.block)) byBlock.set(w.block, { waves: [] });
    const b = byBlock.get(w.block);
    b.waves.push(w);
    if (w.wave === 0) b.lead = w;
  }
  const blocks = [...byBlock.values()].filter((b) => b.lead);
  s.blocks = blocks.length;
  if (!s.blocks) {
    s.why = why(s, stamped.length, byBlock.size);
    return s;
  }
  const firstEnd = blocks.reduce((m, b) => Math.min(m, b.lead.end), Infinity);
  s.first = position(blocks.filter((b) => b.lead.start < firstEnd));
  s.rest = position(blocks.filter((b) => b.lead.start >= firstEnd));

  // The waves of a block leave their last barrier together: where each one's counter stood then.
  const one = [], other = [];
  for (const b of blocks) {
    const last = b.waves.map((w) => [simdOf(w.hw0), since(0, w.cyc0 + w.pro + w.loop, w.bits), w.bits]);
    for (let i = 0; i < last.length; i++) {
      for (let j = i + 1; j < last.length; j++) {
        (last[i][0] === last[j][0] ? one : other).push(apart(last[i][1], last[j][1], last[i][2]));
      }
    }
  }
  s.counter = { one: spread(one), other: spread(other), across: across(waves) };

  // Blocks that shared a WGP at the same time.
  const byWgp = new Map();
  for (const b of blocks) {
    const k = wgpOf(b.lead.hw0);
    if (!byWgp.has(k)) byWgp.set(k, []);
    byWgp.get(k).push(b);
  }
  const gaps = [], phases = { 16: [], 19: [] }, apartRt = { 16: [], 19: [] }, rejected = { 16: 0, 19: 0 };
  let pairs = 0, simdPairs = 0;
  for (const g of byWgp.values()) {
    g.sort((x, y) => x.lead.start - y.lead.start);
    for (let i = 0; i < g.length; i++) {
      for (let j = i + 1; j < g.length && g[j].lead.start < g[i].lead.end; j++) {
        const A = g[i], B = g[j];
        pairs++;
        gaps.push((B.lead.start - A.lead.start) * (A.lead.cpt + B.lead.cpt) / 2);
        const simds = new Set(A.waves.map((w) => simdOf(w.hw0)));
        if (B.waves.some((w) => simds.has(simdOf(w.hw0)))) simdPairs++;
        for (const bit of [16, 19]) {
          const p = phaseOf(A, B, bit);
          if (p && p.rejected) rejected[bit]++;
          else if (p) {
            phases[bit].push(p.phase);
            apartRt[bit].push(p.gap);
          }
        }
      }
    }
  }
  s.pairs = { n: pairs, simd: simdPairs, gap: spread(gaps) };
  s.phase = {};
  for (const bit of [16, 19]) {
    s.phase[bit] = Object.assign(spread(phases[bit]), { rejected: rejected[bit], rt: spread(apartRt[bit]) });
  }
  return s;
}

const r0 = (x) => (Number.isFinite(x) ? String(Math.round(x)) : '-');
const pct = (x) => (Number.isFinite(x) ? (100 * x).toFixed(2) + '%' : '-');

function format(s) {
  const h = s.head;
  const lines = [
    'run: k ' + h.k + ', ' + h.m + ' x ' + h.cols + ', the last of ' + h.launches + ' launches; ' + s.blocks
      + ' blocks, ' + s.read + ' waves read, ' + s.moved + ' dropped (HW_ID1 changed), ' + s.unstamped
      + ' unstamped, ' + s.missing + ' slots empty, ' + s.misplaced + ' misplaced; REALTIME ' + s.tickMhz
      + ' MHz; counter ' + s.bits + ' bits',
  ];
  if (!s.blocks) return lines.concat(s.why);
  for (const name of ['first', 'rest']) {
    const p = s[name];
    lines.push(name + ': ' + p.n + ' blocks, ' + r0(p.mhz.med) + ' MHz (p10 ' + r0(p.mhz.p10) + ', p90 '
      + r0(p.mhz.p90) + '); slab ' + r0(p.slab.med) + ' cycles (p90 ' + r0(p.slab.p90) + '); barrier '
      + pct(p.bar) + ' of the loop, ' + r0(p.barSlab.med) + ' a slab; memory wait ' + pct(p.mem) + ', '
      + r0(p.memSlab.med) + ' a slab; prologue ' + pct(p.pro) + ', epilogue ' + pct(p.epi));
  }
  const c = s.counter;
  lines.push('counter: waves of a block at their last barrier exit, one SIMD ' + r0(c.one.med) + ' cycles apart (p90 '
    + r0(c.one.p90) + ', ' + c.one.n + ' pairs), different SIMDs ' + r0(c.other.med) + ' (p90 ' + r0(c.other.p90)
    + ', ' + c.other.n + ')');
  lines.push('counter across blocks: ' + c.across.n + ' waves that started on a SIMD after another wave there ended;'
    + ' counter gap less REALTIME gap median ' + r0(c.across.med) + ' cycles (p90 ' + r0(c.across.p90) + ')');
  lines.push('pairs: ' + s.pairs.n + ' pairs of blocks on one WGP at once, ' + s.pairs.simd
    + ' with a SIMD in common; start gap median ' + r0(s.pairs.gap.med) + ' cycles');
  for (const bit of [16, 19]) {
    const p = s.phase[bit];
    lines.push('phase bit ' + bit + ': ' + p.n + ' pairs on one boundary, median ' + r0(p.med) + ', p10 '
      + r0(p.p10) + ', p90 ' + r0(p.p90) + ' cycles (rest slab ' + r0(s.rest.slab.med) + '); the crossings '
      + r0(p.rt.med) + ' cycles apart on REALTIME (p90 ' + r0(p.rt.p90) + '); ' + p.rejected + ' dropped: more than '
      + MAX_SLABS_APART + ' slabs apart there');
  }
  return lines;
}

// Items 14 and 15, from the stamp build's summary and the abl8 stamp build's, on one PC.
function gates(st, ab) {
  const none = (s, name) => (!s.blocks ? 'the ' + name + ': ' + s.why
    : !s.rest.n ? 'the ' + name + ' has no blocks after the first round (each started before the first one'
      + ' ended): run more blocks' : null);
  const reason = none(st, 'stamp build') || none(ab, 'abl8 build');
  if (reason) return { why: reason };
  const barSlab = st.rest.barSlab.med, memSlab = st.rest.memSlab.med;
  const slab = st.rest.slab.med, slab8 = ab.rest.slab.med, work = st.rest.work.med, work8 = ab.rest.work.med;
  const lds = work - work8;
  const D = barSlab + lds, Dm = D + memSlab;
  const save = slab - slab8, share = save / slab;
  const bit = st.phase[19].n >= MIN_PAIRS ? 19 : st.phase[16].n >= MIN_PAIRS ? 16 : 0;
  const phase = bit ? st.phase[bit] : null;
  // The phases rest on one counter a SIMD: across blocks it must follow REALTIME.
  const ac = st.counter.across;
  const counter = ac.n < MIN_PAIRS ? 'unchecked' : ac.med <= slab / 2 ? 'ok' : 'off';
  const verdict = (d) => (!phase || counter === 'off' || !Number.isFinite(d) ? 'no reading'
    : phase.med > d ? 'kill' : 'pass');
  return {
    barSlab, memSlab, slab, slab8, work, work8, lds, D, Dm, bit, phase, counter, across: ac,
    item14: verdict(D), item14m: verdict(Dm),
    save, share, item15: Number.isFinite(share) ? (share >= 0.03 ? 'pass' : 'kill') : 'no reading',
  };
}

function formatGates(g) {
  if (g.why) return ['gates: no reading: ' + g.why];
  const note = g.counter === 'off' ? '; the counter is ' + r0(g.across.med) + ' cycles off REALTIME across blocks'
      + ' on one SIMD (median), so it isn\'t one counter a SIMD'
    : g.counter === 'unchecked' ? '; the counter across blocks is unchecked (' + g.across.n + ' waves)' : '';
  return [
    'D: ' + r0(g.D) + ' cycles a slab = barrier wait ' + r0(g.barSlab) + ' + LDS fragment loads ' + r0(g.lds)
      + ' (work a slab, the loop less barrier and memory wait: ' + r0(g.work) + ' less abl8\'s ' + r0(g.work8)
      + '); with memory wait ' + r0(g.memSlab) + ': ' + r0(g.Dm) + '; medians over the rest of the launch',
    'item 14: ' + (g.phase ? 'phase ' + r0(g.phase.med) + ' cycles (bit ' + g.bit + ', ' + g.phase.n + ' pairs)'
      : 'under ' + MIN_PAIRS + ' pairs') + ' against D ' + r0(g.D) + ': ' + g.item14 + '; against D with memory'
      + ' wait ' + r0(g.Dm) + ': ' + g.item14m + ' (kill if the phase is larger)' + note,
    'item 15: abl8 saves ' + r0(g.save) + ' of ' + r0(g.slab) + ' cycles a slab, ' + pct(g.share) + ': ' + g.item15
      + ' (kill under 3%)',
  ];
}

function main(argv, out = console.log, err = console.error) {
  const [cmd, ...files] = argv;
  try {
    if (cmd === 'summary' && files.length === 1) {
      format(summarize(load(fs.readFileSync(files[0])))).forEach((l) => out(l));
      return 0;
    }
    if (cmd === 'gates' && files.length === 2) {
      const [st, ab] = files.map((f) => summarize(load(fs.readFileSync(f))));
      formatGates(gates(st, ab)).forEach((l) => out(l));
      return 0;
    }
  } catch (e) {
    err('stamps: ' + e.message);
    return 1;
  }
  err('usage: stamp-summary.js summary <stamps file> | gates <stamp build\'s file> <abl8 stamp build\'s file>');
  return 2;
}

if (require.main === module) process.exitCode = main(process.argv.slice(2));

module.exports = { load, since, apart, summarize, format, gates, formatGates, main, MIN_PAIRS, MAX_SLABS_APART };
