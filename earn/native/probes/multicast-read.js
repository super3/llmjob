'use strict';

// node multicast-read.js <log> -- multicast-probe.cu's lines read for item 16's step 2: f, the
// share of a B' box's cost that TMA multicast saves, and which barrier pattern hangs, if one
// does. And for issue #287's item 7: fp, the share a copy between the pair's shared memories saves
// (the push arms a2, h2 and p), and whether that copy lands at all. compilers.sh's "multicast" stage runs it on the probe's own log; a box log from
// `amdtest.py logs` reads the same, since each probe line keeps its tag after the prefix.
// Every line it prints starts with "read:", so its output is never read back as the probe's.

const fs = require('fs');

// The probe's boxes, and what each arm moves a pair-stage (multicast-probe.cu's ARMS): bytes TMA
// lands in shared in both CTAs, and bytes read from the L2, a multicast counted once.
const BOXB = 256 * 64;
const BOXA = 192 * 64;
const ARMS = {
  e: { fill: 0, l2: 0 },
  a: { fill: 2 * BOXB, l2: 2 * BOXB },
  b: { fill: 2 * BOXB, l2: BOXB },
  d: { fill: BOXB, l2: BOXB },
  c: { fill: 2 * BOXB, l2: BOXB },
  cA: { fill: 2 * (BOXB + BOXA), l2: BOXB + 2 * BOXA },
  cAH: { fill: 2 * (BOXB + BOXA), l2: BOXB + 2 * BOXA },
  a2: { fill: 2 * BOXB, l2: 2 * BOXB },
  h2: { fill: BOXB, l2: BOXB },
  p: { fill: 2 * BOXB, l2: BOXB },
};
const PUSH_ARMS = ['a2', 'h2', 'p'];
const MC_ARMS = ['a', 'd', 'b', 'c', 'cA', 'cAH'];
// cAH's stages carry the tile seam's hash sleep, so its energy and pace are not compared.
const UNTIMED = new Set(['cAH']);
// The step-2 plan's gate for item 16's #1 (B' multicast) and #3 (A' multicast).
const PASS_F = 0.5;
// Item 7's gate for building the fold that shares B' by the copy: fp at least this. What a B' box a
// CTA-stage costs a stock RTX 5090 at its 600 W limit: skip B' in item 16's step 1 (+5.20%, 1
// November). A pair that shares B' reads half of it from the L2, so the fold could gain about
// SKIP_B / 2 x fp, before its own costs.
const PASS_FP = 0.5;
const SKIP_B = 0.052;
// A stage more than 1% over the pace did not hold it.
const PACE_SLACK = 1.01;
// Arms compared in one round must take the same time a stage at the same clock. A box is a small
// part of a pair-stage's energy and the card's base power is most of it, so 1% more time in one
// arm moves f by 1% of E_e over (E_a - E_d), several hundredths (the reader prints how many). So
// flag 0.5% between arms of a round, and take f from energies brought to the reference arm's
// stage time at e's power.
const SAME_SLACK = 0.005;
const TAGS = /\b(CARD|NVML|CLUSTERS|LAUNCH|SMOKE|ROUND|TIMEOUT|FAULT|HANG|DONE|SKIP)\b(.*)$/;

// key=value pairs of a line, decimal numbers as numbers (a barrier word stays hex text).
function fields(rest) {
  const out = {};
  for (const m of rest.matchAll(/(\w+)=(\S+)/g)) out[m[1]] = /^-?\d+(\.\d+)?$/.test(m[2]) ? Number(m[2]) : m[2];
  return out;
}

function parse(text) {
  const out = { card: null, nvml: null, clusters: {}, smoke: {}, rounds: [], timeouts: [], hangs: [],
    faults: [], skip: false, done: false, early: false };
  for (const raw of text.split('\n')) {
    if (/\bread:/.test(raw)) continue;
    const m = raw.match(TAGS);
    if (!m) continue;
    const [, tag, rest] = m;
    const f = fields(rest);
    if (tag === 'CARD') {
      const name = rest.match(/name=(.*?) cc=/);
      out.card = { ...f, name: name ? name[1] : f.name };
    } else if (tag === 'NVML') {
      out.nvml = { on: /^\s*yes/.test(rest), energy: f.energy === 'yes' };
    } else if (tag === 'CLUSTERS') {
      out.clusters[f.arm] = { rc: f.rc, clusters: f.clusters, fallback: f.fallback };
    } else if (tag === 'SMOKE') {
      out.smoke[f.arm] = { ...f, timeout: /\btimeout\b/.test(rest) };
    } else if (tag === 'ROUND') {
      out.rounds.push(f);
    } else if (tag === 'TIMEOUT') {
      out.timeouts.push(f);
    } else if (tag === 'HANG') {
      out.hangs.push({ ...f, noWait: /no wait gave up/.test(rest) });
    } else if (tag === 'FAULT') {
      // A launch the runtime refused, a kernel that crashed once launched, or the setup.
      const kind = /\blaunch rc=/.test(rest) ? 'launch' : /\bsync rc=/.test(rest) ? 'sync' : 'setup';
      out.faults.push({ ...f, kind, text: rest.trim() });
    } else if (tag === 'SKIP') {
      out.skip = true;
    } else if (tag === 'DONE') {
      // "DONE early": the probe stopped on a fault or a hang.
      if (/^\s*early\b/.test(rest)) out.early = true;
      else out.done = true;
    }
  }
  return out;
}

// Joules a pair-stage: a window's energy over its CTA-stages, two CTAs a pair.
function energyPerPairStage(row) {
  return row.cta_stages > 0 ? row.j / (row.cta_stages / 2) : NaN;
}

// Seconds a stage: every CTA runs its stages one after another, all CTAs at once.
function stageTime(row) {
  return row.cta_stages > 0 ? row.secs * row.ctas / row.cta_stages : NaN;
}

// A row's energy a pair-stage as if its stages took ref's time, the difference at `watts`, the
// card's power with no loads (e's): time is no box's cost.
function evenTime(row, ref, watts) {
  return energyPerPairStage(row) - watts * (stageTime(row) - stageTime(ref)) * 2 / row.ctas;
}

function median(xs) {
  const s = xs.filter(Number.isFinite).sort((x, y) => x - y);
  if (!s.length) return NaN;
  const h = Math.floor(s.length / 2);
  return s.length % 2 ? s[h] : (s[h - 1] + s[h]) / 2;
}

// Each round's rows by arm: { r: { a: row, ... } }.
function byRound(rounds) {
  const out = {};
  for (const row of rounds) {
    if (!Number.isFinite(row.r) || !ARMS[row.arm]) continue;
    (out[row.r] = out[row.r] || {})[row.arm] = row;
  }
  return out;
}

// f = (E_a - E_b) / (E_a - E_d): what multicast saves of a whole box's cost, at equal bytes
// filled (a and b both fill two boxes). NaN when a box costs nothing measurable (E_a <= E_d).
function fShare(E) {
  const box = E.a - E.d;
  return box > 0 ? (E.a - E.b) / box : NaN;
}

// fp = (E_a2 - E_p) / (E_a2 - E_h2): what the copy saves of an L2 box's cost. a2 and p fill the same
// bytes; h2 reads the same bytes from the L2 as p and fills half. NaN when an L2 box costs nothing
// measurable (E_a2 <= E_h2).
function fPush(E) {
  const box = E.a2 - E.h2;
  return box > 0 ? (E.a2 - E.p) / box : NaN;
}

// Whether the copy lands: p against a2 and h2, which share its barriers and loads but copy nothing.
// Returns the verdict line, or null when no push arm ran.
function pushVerdict(p, stuck) {
  const ran = (arm) => (p.smoke[arm] && !p.smoke[arm].timeout) || p.rounds.some((r) => r.arm === arm);
  if (!PUSH_ARMS.some((a) => ran(a) || stuck.has(a))) return null;
  const first = (arm) => {
    const t = p.timeouts.find((x) => x.arm === arm);
    return t ? ` (first to give up: a ${t.role} wait on rank ${t.rank}, stage ${t.stage})` : '';
  };
  const base = ['a2', 'h2'].filter((a) => stuck.has(a));
  if (base.length) {
    return `${base.join(' and ')} stuck${first(base[0])}: the fold's cluster barriers with plain loads stick on this `
      + 'card, so p cannot be read';
  }
  if (stuck.has('p')) {
    return `p sticks and a2 and h2 do not${first('p')}: the copy between the pair's shared memories, or its `
      + 'complete_tx on the peer\'s FULL, never lands on this card. With multicast gone too, two CTAs cannot share '
      + 'B\' here: drop item 7';
  }
  if (PUSH_ARMS.every(ran)) return 'a2, h2 and p ran clean: the copy between the pair\'s shared memories lands on this card';
  const missing = PUSH_ARMS.filter((a) => !ran(a));
  return `${missing.join(', ')} never ran${p.done ? '' : ' (the probe stopped early)'}: no verdict on the copy`;
}

function pct(x, digits = 1) {
  return (x >= 0 ? '+' : '') + (100 * x).toFixed(digits) + '%';
}
function nj(x) {
  return (x * 1e9).toFixed(1);
}
const range = (xs) => `${xs.length} rounds, ${Math.min(...xs).toFixed(2)} to ${Math.max(...xs).toFixed(2)}`;

// Which pattern hangs: the ladder b, c, cA, cAH, each the one before plus one thing, under e, a
// and d, which share the probe's coupling. Returns the verdict line, or null.
function hangVerdict(p, stuck) {
  const ran = (arm) => (p.smoke[arm] && !p.smoke[arm].timeout) || p.rounds.some((r) => r.arm === arm);
  const clean = (arm) => ran(arm) && !stuck.has(arm);
  const first = (arm) => {
    const t = p.timeouts.find((x) => x.arm === arm);
    return t ? ` (first to give up: a ${t.role} wait on rank ${t.rank}, stage ${t.stage})` : '';
  };
  const base = ['e', 'a', 'd'].filter((a) => stuck.has(a));
  if (base.length) {
    return `${base.join(', ')} stuck: the probe's own coupling or its plain loads stick, so f below cannot be `
      + 'trusted and the ladder above them says nothing; read the timeout lines';
  }
  if (stuck.has('b')) {
    if (clean('c')) {
      return 'b sticks and c, the same multicast with the fold\'s barriers, does not: the probe\'s coupling '
        + 'beside the multicast sticks (rank 0\'s remote arrive on rank 1\'s FULL with the multicast\'s '
        + `complete_tx on it)${first('b')}, not the multicast. f cannot be read; the fold's barriers did not hang`;
    }
    return 'b sticks and a, d and e do not: b differs from a only in its load, so the multicast\'s bytes or its '
      + `complete_tx never arrive on this card${first('b')}. A FULL wait on rank 1 is the multicast's `
      + 'receiver. That fits the cluster build\'s hang (it multicasts B\' the same way): drop #1 to #3, all multicast';
  }
  if (stuck.has('c')) {
    return `c sticks and b does not${first('c')}: the fold's barrier protocol hangs on this card (EMPTY counting `
      + 'both CTAs, with mapa remote arrives and cluster-scope acquires, and FULL with no remote arrival; c changes '
      + 'both, so this does not say which). That is the cluster build\'s hang';
  }
  if (stuck.has('cA')) {
    return `cA sticks and c does not${first('cA')}: multicast beside each CTA's own A' box in one FULL phase hangs `
      + 'on this card. That is the cluster build\'s hang';
  }
  if (stuck.has('cAH')) {
    return `cAH sticks and cA does not${first('cAH')}: the fold's tile seam hangs on this card (the hand-off's `
      + 'CTA-scope wait on an EMPTY that counts the peer\'s arrivals, with the hashers a hash behind). That is the '
      + 'cluster build\'s hang';
  }
  if (['b', 'c', 'cA', 'cAH'].every(clean)) {
    return 'nothing stuck, up to the fold\'s whole barrier pattern with its tile seam (cAH): the cluster build\'s '
      + 'hang is in what the probe leaves out (the mma, the real tile walk, its spill, the hash\'s code)';
  }
  const missing = ['b', 'c', 'cA', 'cAH'].filter((a) => !ran(a));
  return `${missing.join(', ')} never ran${p.done ? '' : ' (the probe stopped early)'}: no verdict on what hangs`;
}

function summarize(p) {
  const lines = [];
  const say = (s) => lines.push('read: ' + s);
  if (p.skip) {
    say('skipped: the card has no clusters or TMA (sm_90 or newer needed)');
    return lines;
  }
  // Was the cluster launch granted? A refused query or launch says the card is out before any
  // pattern is tried; a fault after a launch is a crash, not a refusal.
  const cl = Object.entries(p.clusters);
  const refused = cl.filter(([, c]) => c.rc !== 0 || !(c.clusters > 0));
  if (!cl.length) say('no CLUSTERS lines: the probe did not get as far as the occupancy query');
  else if (refused.length)
    say(`occupancy query refused or zero for ${refused.map(([a, c]) => `${a} (rc ${c.rc}, ${c.clusters} clusters)`).join(', ')}`);
  else say(`clusters granted: ${median(cl.map(([, c]) => c.clusters))} of two CTAs (query rc 0 for every arm)`);
  if (p.card && p.card.own_dst) say(`own boxes in the ${p.card.own_dst === 'cta' ? 'fold\'s .shared::cta' : '.shared::cluster'} form`);

  const launchFaults = p.faults.filter((f) => f.kind === 'launch');
  const syncFaults = p.faults.filter((f) => f.kind === 'sync');
  for (const f of p.faults.filter((x) => x.kind === 'setup')) say(`fault in the setup: ${f.text}`);
  for (const f of launchFaults) say(`launch refused for ${f.arm}: ${f.text}`);
  for (const f of syncFaults) {
    say(`${f.arm} crashed once launched (${f.text}): a fault, not a refusal and not a hang; the probe stopped there`);
  }
  const stuck = new Set([...p.timeouts.map((t) => t.arm), ...p.hangs.map((h) => h.arm)]);
  for (const t of p.timeouts)
    say(`timeout in ${t.arm}: ${t.role} wait, block ${t.block} (rank ${t.rank}), stage ${t.stage}, barrier word ${t.state}`);
  for (const h of p.hangs) {
    say(`hang in ${h.arm}: the launch never finished, ${h.noWait
      ? 'and no wait gave up, so a wait the watchdog cannot bound stuck (a cluster barrier, or cAH\'s named barrier)'
      : 'after a wait gave up (the timeout above): a warp was left at a barrier that cannot give up'}`);
  }
  for (const [arm, s] of Object.entries(p.smoke)) {
    if (s.bad > 0) say(`smoke ${arm}: ${s.bad} CTAs stopped short or read bytes that were not the boxes loaded`);
  }

  // The verdict: a refusal first, then a crash, then the ladder.
  if (launchFaults.some((f) => f.arm === 'e')) {
    say('verdict: the card refused the cluster launch itself (e, the first arm, loads nothing): no cluster work '
      + 'runs here at this footprint; drop #1 to #5');
  } else if (syncFaults.length) {
    const f = syncFaults[0];
    say(['c', 'cA', 'cAH'].includes(f.arm)
      ? `verdict: ${f.arm} crashed on this card: the fold's cluster pattern faults rather than hangs here, and the `
        + 'arms after it did not run'
      : `verdict: ${f.arm} crashed: f cannot be read, and the arms after it did not run`);
  } else {
    const ranAny = (arm) => p.smoke[arm] || p.rounds.some((r) => r.arm === arm) || stuck.has(arm);
    // A run of the push phase alone (MC_ARMS=push) has no multicast verdict to give.
    if (MC_ARMS.some(ranAny) || !PUSH_ARMS.some(ranAny)) say(`verdict: ${hangVerdict(p, stuck)}`);
    const pv = pushVerdict(p, stuck);
    if (pv) say(`push verdict: ${pv}`);
  }

  // f, round by round, from energies at e's stage time (and as measured).
  const R = byRound(p.rounds);
  const fEven = [];
  const fRaw = [];
  const lin = [];
  const perB = [];
  const sens = [];
  const baseW = [];
  const fpEven = [];
  const fpRaw = [];
  const fpSens = [];
  const l2Box = [];
  const copyBox = [];
  const drift = {};
  const note = (arm, ref, row, refRow) => {
    const d = drift[arm] = drift[arm] || { ref, time: [], mhz: [] };
    d.time.push(stageTime(row) / stageTime(refRow) - 1);
    d.mhz.push(row.mhz / refRow.mhz - 1);
  };
  for (const rows of Object.values(R)) {
    if (!rows.e) continue;
    const w = rows.e.j / rows.e.secs;
    baseW.push(w);
    const E = {};
    const raw = {};
    for (const [arm, row] of Object.entries(rows)) {
      raw[arm] = energyPerPairStage(row);
      E[arm] = evenTime(row, rows.e, w);
      if (arm !== 'e' && !UNTIMED.has(arm)) note(arm, 'e', row, rows.e);
    }
    if (['a', 'b', 'd'].every((k) => Number.isFinite(E[k]))) {
      fEven.push(fShare(E));
      fRaw.push(fShare(raw));
      if (E.a > E.d) sens.push(0.01 * E.e / (E.a - E.d));
    }
    if (['a', 'd'].every((k) => Number.isFinite(E[k])) && E.d > E.e) {
      lin.push((E.a - E.e) / (E.d - E.e));
      perB.push((E.d - E.e) / BOXB);
    }
    if (PUSH_ARMS.every((k) => Number.isFinite(E[k]))) {
      fpEven.push(fPush(E));
      fpRaw.push(fPush(raw));
      if (E.a2 > E.h2) fpSens.push(0.01 * E.e / (E.a2 - E.h2));
      l2Box.push(E.a2 - E.h2);
      copyBox.push(E.p - E.h2);
    }
  }
  const fOk = fEven.filter(Number.isFinite);
  const rawOk = fRaw.filter(Number.isFinite);
  const mcRounds = p.rounds.some((r) => MC_ARMS.includes(r.arm));
  const pushTried = PUSH_ARMS.some((a) => p.smoke[a] || p.rounds.some((r) => r.arm === a));
  if (!fEven.length && !mcRounds && pushTried) {
    // The push phase alone (MC_ARMS=push): no multicast f to give.
  } else if (!fEven.length) {
    say('f: no round has a, b, d and e');
  } else if (!fOk.length) {
    say('f: a box costs nothing measurable here (E_a <= E_d in every round), so there is no f');
  } else {
    const f = median(fOk);
    const fr = median(rawOk);
    say(`f = ${f.toFixed(2)} (${range(fOk)}; as measured ${Number.isFinite(fr) ? fr.toFixed(2) : 'none'}): `
      + `multicast saves ${Math.round(100 * f)}% of a whole B' box's cost`);
    say(f >= PASS_F
      ? `gate f >= ${PASS_F}: pass, multicast stays alive (A' multicast, #3, before B', #1, while #6 lives)`
      : `gate f >= ${PASS_F}: fail, drop #1 and #3`);
    if (Number.isFinite(fr) && (fr >= PASS_F) !== (f >= PASS_F)) {
      say('f as measured falls on the other side of the gate: the arms\' stage times differ (lines below), so '
        + 'repeat with longer rounds before deciding');
    }
    if (sens.length) say(`1% more stage time in one arm moves f by ${median(sens).toFixed(2)} here`);
  }
  if (perB.length)
    say(`one box: ${(median(perB) * 1e12).toFixed(1)} pJ a byte filled (d - e); two cost ${median(lin).toFixed(2)} `
      + 'times one (a - e over d - e; 2 is linear)');

  // fp, round by round, from energies at e's stage time (and as measured).
  const fpOk = fpEven.filter(Number.isFinite);
  const fpRawOk = fpRaw.filter(Number.isFinite);
  if (fpEven.length && !fpOk.length) {
    say('fp: an L2 box costs nothing measurable here (E_a2 <= E_h2 in every round), so there is no fp');
  } else if (fpOk.length) {
    const f = median(fpOk);
    const fr = median(fpRawOk);
    say(`fp = ${f.toFixed(2)} (${range(fpOk)}; as measured ${Number.isFinite(fr) ? fr.toFixed(2) : 'none'}): `
      + `the copy between the pair saves ${Math.round(100 * f)}% of an L2 box's cost`);
    say(`an L2 box (a2 - h2): ${nj(median(l2Box))} nJ a pair-stage, ${(median(l2Box) / BOXB * 1e12).toFixed(1)} pJ a `
      + `byte; the copy of one (p - h2): ${nj(median(copyBox))} nJ, ${(median(copyBox) / BOXB * 1e12).toFixed(1)} pJ a byte`);
    say(f >= PASS_FP
      ? `push gate fp >= ${PASS_FP}: pass. A stock 5090 pays about ${pct(SKIP_B)} for a B' box a CTA-stage, so a fold `
        + `sharing B' this way could gain about ${pct(SKIP_B / 2 * f)} there before its own costs: race it`
      : `push gate fp >= ${PASS_FP}: fail (about ${pct(SKIP_B / 2 * f)} at most on a stock 5090): drop item 7`);
    if (Number.isFinite(fr) && (fr >= PASS_FP) !== (f >= PASS_FP)) {
      say('fp as measured falls on the other side of the gate: the arms\' stage times differ (lines below), so '
        + 'repeat with longer rounds before deciding');
    }
    if (fpSens.length) say(`1% more stage time in one push arm moves fp by ${median(fpSens).toFixed(2)} here`);
  } else if (PUSH_ARMS.some((a) => p.rounds.some((r) => r.arm === a))) {
    say('fp: no round has e, a2, h2 and p');
  }

  // The fold's barriers against the probe's, and the A' boxes on top, in the rounds without e:
  // b is the reference there, at e's power from the rounds before.
  const W = median(baseW);
  const dc = [];
  const dA = [];
  for (const rows of Object.values(R)) {
    if (rows.e || !rows.b) continue;
    for (const arm of ['c', 'cA']) if (rows[arm]) note(arm, 'b', rows[arm], rows.b);
    const E = (arm) => (rows[arm] ? (Number.isFinite(W) ? evenTime(rows[arm], rows.b, W) : energyPerPairStage(rows[arm])) : NaN);
    if (Number.isFinite(E('c'))) dc.push([E('c') - E('b'), (E('c') - E('b')) / E('b')]);
    if (Number.isFinite(E('cA')) && Number.isFinite(E('c'))) dA.push(E('cA') - E('c'));
  }
  if (dc.length)
    say(`c's barrier protocol (cluster-wide EMPTY, FULL without the remote arrival): c - b = `
      + `${nj(median(dc.map((x) => x[0])))} nJ a pair-stage (${pct(median(dc.map((x) => x[1])))} of b)`);
  if (dA.length) say(`the A' boxes: cA - c = ${nj(median(dA))} nJ a pair-stage`);
  if ((dc.length || dA.length) && !Number.isFinite(W)) say('no e rounds: c - b and cA - c as measured, not at b\'s stage time');

  // Arms that did not run alike in their rounds: the time correction then carries the reading.
  for (const [arm, d] of Object.entries(drift)) {
    const worst = (xs) => xs.reduce((m, x) => (Math.abs(x) > Math.abs(m) ? x : m), 0);
    const t = worst(d.time);
    const c = worst(d.mhz);
    if (Math.abs(t) > SAME_SLACK || Math.abs(c) > SAME_SLACK) {
      say(`${arm} against ${d.ref} in one round: stage time ${pct(t, 2)}, clock ${pct(c, 2)}, past `
        + `${pct(SAME_SLACK, 1)}: the arms did not run alike, so the reading leans on the time correction`);
    }
  }

  // Per arm, and what would spoil a reading.
  const arms = Object.keys(ARMS).filter((a) => p.rounds.some((r) => r.arm === a));
  for (const a of arms) {
    const rows = p.rounds.filter((r) => r.arm === a);
    const E = median(rows.map(energyPerPairStage));
    const cyc = median(rows.map((r) => r.cyc_stage));
    const pace = median(rows.map((r) => r.pace));
    const bytes = rows.reduce((s, r) => s + r.cta_stages / 2 * ARMS[a].fill, 0);
    const gb = rows.reduce((s, r) => s + r.fill_gb, 0);
    const slow = !UNTIMED.has(a) && pace > 0 && cyc > pace * PACE_SLACK;
    say(`${a}: ${nj(E)} nJ a pair-stage, ${median(rows.map((r) => r.w)).toFixed(1)} W, ${median(rows.map((r) => r.mhz))} MHz, `
      + `${cyc} cycles a stage, ${rows.length} rounds${slow ? `, pace ${pace} not held` : ''}`
      + `${UNTIMED.has(a) ? ' (the seam\'s hash sleep is in its stages: not compared)' : ''}`);
    if (Math.abs(bytes / 1e9 - gb) > 0.002 * Math.max(gb, 1)) say(`${a}: fill_gb ${gb.toFixed(3)} is not the table's ${(bytes / 1e9).toFixed(3)}`);
    const bad = rows.reduce((s, r) => s + (r.bad || 0), 0);
    if (bad) say(`${a}: ${bad} CTA checksums wrong: the bytes read were not the boxes loaded`);
    if (rows.some((r) => r.j_src !== 'energy')) say(`${a}: energy from mean power, not NVML's energy counter`);
  }
  return lines;
}

module.exports = { BOXB, BOXA, ARMS, PUSH_ARMS, PASS_F, PASS_FP, SKIP_B, PACE_SLACK, SAME_SLACK, parse,
  energyPerPairStage, stageTime, evenTime, fShare, fPush, summarize };

if (require.main === module) {
  const file = process.argv[2];
  if (!file) {
    console.error('usage: node multicast-read.js <multicast.log>');
    process.exit(2);
  }
  for (const l of summarize(parse(fs.readFileSync(file, 'utf8')))) console.log(l);
}
