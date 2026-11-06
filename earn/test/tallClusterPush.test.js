'use strict';

const fs = require('fs');
const path = require('path');
const { preprocess: resolve } = require('./helpers/amdSwitches');

// PEARL_TALL_CLUSTER_PUSH (pearl_config.h, 0 by default; issue #287, item 7): the two-CTA cluster
// build of the TMA tall fold shares B' by a copy between the CTAs' shared memories instead of
// multicast, which doesn't complete on sm_120. Each CTA loads half of the pair's B' box on a HALF
// mbarrier; once it has landed the producer copies it into the peer's buffer, completing on the
// peer's FULL, and makes its own second FULL arrival. Raced only if multicast-probe.cu's push
// arms pass (fp >= 0.5).
//
// Nothing here runs device code. These check:
// - off by default, and refused without the cluster build or the ring's roles;
// - off, the fold is the shipped code: nothing of the switch is left;
// - on, the fill, the barriers, the prologue and the stage-top copy are what the model runs;
// - the ring, modelled step by step over several tiles, never reads a half that has not landed,
//   never lands a half on a buffer being read, never copies a half before it landed or after it
//   was refilled, and never deadlocks; and the model finds each of five broken variants;
// - the host loads B' in 128-column boxes and names the build; CI compiles it.

const SRC = path.join(__dirname, '..', 'native', 'src');
const NL = String.fromCharCode(10);
const rawRead = (file) => fs.readFileSync(file, 'utf8').split(String.fromCharCode(13) + NL).join(NL);
const read = (name) => rawRead(path.join(SRC, name));
const HEADER = read('pearl_config.h');
const KERNEL = read('pearl_kernel.cu');
const FOLD = read('pearl_fold_tall.cuh');
const HOST = read('pearl_host.cu');
const CI = rawRead(path.join(SRC, '..', '..', '..', '.github', 'workflows', 'native-core.yml'));
const NAME = 'PEARL_TALL_CLUSTER_PUSH';
const lines = (...l) => l.join(NL);

// Source without comments: comment lines dropped, trailing ones cut.
function code(text) {
  return text.split(NL).filter((l) => !l.trim().startsWith('//')).map((l) => {
    const at = l.indexOf('//');
    return at > 0 && l.slice(0, at).trim() ? l.slice(0, at).trimEnd() : l;
  }).join(NL);
}

// The switch's block in pearl_config.h, from its #ifndef to the body define's #endif.
function block() {
  const from = HEADER.indexOf(`#ifndef ${NAME}${NL}`);
  const end = '#define PEARL_TALL_PUSH_BODY 0' + NL + '#endif';
  const to = HEADER.indexOf(end, from);
  expect([from > 0, to > from]).toEqual([true, true]);
  return HEADER.slice(from, to + end.length);
}

// ---------------------------------------------------------------------------
// A model of the push build's ring, two CTAs of W warps (warp P = W - 1 has the producer), three
// buffers with FULL (two arrivals), HALF (one) and a cluster-wide EMPTY (both CTAs' warps), over
// T tiles of CH chunks of two stages. Stage g reads buffer g % 3 (the roles soA, soB, soC rotate
// so that it does). It follows the fold line for line (pinned below): the prologue loads stage 0
// whole and stage 1 as a half; at the top of each stage the producer copies stage g + 1's half to
// the peer (stage 0 always, stage 1 if stage_next), then refills stage g + 2 behind its EMPTY wait
// (if stage_next); the warps wait on FULL, read, and arrive on both CTAs' EMPTY.
// bug: null, or one of 'nohalf' (copy without waiting for the half to land), 'always' (copy at
// stage 1 without stage_next), 'full1' (FULL counts one arrival), 'localempty' (EMPTY counts this
// CTA's warps only), 'nohalf0' (the prologue makes no HALF arrival).
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function simulate(seed, { W = 2, T = 3, CH = 3, bug = null } = {}) {
  const HALF = 8192;
  const BOXA = 12288;
  const STAGE = 2 * HALF + BOXA;
  const G = T * CH * 2;
  const P = W - 1;
  const rand = rng(seed);
  const errors = [];
  const bar = (count) => ({ count, pending: count, phase: 0, tx: 0 });
  const ctas = [0, 1].map(() => ({
    full: [0, 1, 2].map(() => bar(bug === 'full1' ? 1 : 2)),
    half: [0, 1, 2].map(() => bar(1)),
    empty: [0, 1, 2].map(() => bar(bug === 'localempty' ? W : 2 * W)),
    content: [[-1, -1], [-1, -1], [-1, -1]],   // the stage each half of B' in the buffer holds
    reading: [0, 0, 0],
  }));
  const complete = (b) => {
    if (b.pending === 0 && b.tx === 0) {
      b.phase++;
      b.pending = b.count;
    }
  };
  const arrive = (b, what) => {
    if (b.pending === 0) errors.push('over-arrival ' + what);
    b.pending--;
    complete(b);
  };
  const expectTx = (b, bytes, what) => {
    b.tx += bytes;
    arrive(b, what);
  };
  const ready = (b, use, what) => {
    if (b.phase > use + 1) errors.push(`${what}: barrier two phases past use ${use}`);
    return (b.phase & 1) !== (use & 1);
  };
  const lands = [];
  // The fill of stage n as a half (tma_fill): this CTA's half of B' on HALF, its A on FULL, and
  // FULL expecting A and the peer's half.
  const fillHalf = (r, n) => {
    const b = n % 3;
    const me = ctas[r];
    expectTx(me.full[b], STAGE - HALF, 'full');
    expectTx(me.half[b], HALF, 'half');
    lands.push({ rank: r, b, g: n, bytes: HALF, halves: [r], on: 'half' });
    lands.push({ rank: r, b, g: n, bytes: BOXA, halves: [], on: 'full' });
  };
  // The prologue: stage 0 whole on FULL with a plain second arrival and HALF's first phase, and
  // stage 1 a half.
  for (const r of [0, 1]) {
    const me = ctas[r];
    expectTx(me.full[0], STAGE, 'full');
    lands.push({ rank: r, b: 0, g: 0, bytes: 2 * HALF, halves: [0, 1], on: 'full' });
    lands.push({ rank: r, b: 0, g: 0, bytes: BOXA, halves: [], on: 'full' });
    arrive(me.full[0], 'prologue full');
    if (bug !== 'nohalf0') arrive(me.half[0], 'prologue half');
    fillHalf(r, 1);
  }
  // Whether stage g's chunk has a stage after it in this CTA's walk: not its tile's last chunk,
  // or not the last tile (stage_next).
  const stageNext = (g) => {
    const chunk = Math.floor(g / 2) % CH;
    const tile = Math.floor(g / (2 * CH));
    return chunk + 1 < CH || tile + 1 < T;
  };
  const copies = (g) => g % 2 === 0 || stageNext(g) || bug === 'always';
  // The copy of stage n's half of rank r into the peer, on the peer's FULL, then r's own arrival.
  const push = (r, n) => {
    const b = n % 3;
    if (ctas[r].content[b][r] !== n) errors.push(`rank ${r} copied stage ${n}'s half from buffer ${b} holding ${ctas[r].content[b][r]}`);
    lands.push({ rank: 1 - r, b, g: n, bytes: HALF, halves: [r], on: 'full', src: { rank: r, b, h: r, g: n } });
    arrive(ctas[r].full[b], 'full after copy');
  };
  const warps = [];
  for (const rank of [0, 1]) for (let w = 0; w < W; w++) warps.push({ rank, w, g: 0, step: w === P ? 'push' : 'full' });
  const bias = warps.map(() => 0.2 + rand());
  for (let steps = 0; steps < 200000; steps++) {
    const can = [];
    warps.forEach((x, i) => {
      if (x.g >= G) return;
      const me = ctas[x.rank];
      if (x.step === 'push') {
        const n = x.g + 1;
        if (copies(x.g) && n < G && bug !== 'nohalf' && !ready(me.half[n % 3], Math.floor(n / 3), 'half')) return;
      } else if (x.step === 'top') {
        const n = x.g + 2;
        if (stageNext(x.g) && n >= 3 && !ready(me.empty[n % 3], Math.floor(n / 3) - 1, 'empty')) return;
      } else if (x.step === 'full') {
        if (!ready(me.full[x.g % 3], Math.floor(x.g / 3), 'full')) return;
      }
      can.push({ i, weight: bias[i] });
    });
    lands.forEach((l, k) => can.push({ land: k, weight: 1 }));
    if (!can.length) break;
    let pick = rand() * can.reduce((s, c) => s + c.weight, 0);
    const c = can.find((x) => (pick -= x.weight) < 0) || can[can.length - 1];
    if (c.land !== undefined) {
      const l = lands.splice(c.land, 1)[0];
      const t = ctas[l.rank];
      if (l.halves.length && t.reading[l.b] > 0) errors.push(`stage ${l.g}'s B' landed on rank ${l.rank} buffer ${l.b} while it was read`);
      for (const h of l.halves) t.content[l.b][h] = l.g;
      if (l.src && ctas[l.src.rank].content[l.src.b][l.src.h] !== l.src.g)
        errors.push(`copy of stage ${l.src.g} from rank ${l.src.rank} read a refilled source`);
      const on = l.on === 'half' ? t.half[l.b] : t.full[l.b];
      on.tx -= l.bytes;
      complete(on);
      continue;
    }
    const x = warps[c.i];
    const me = ctas[x.rank];
    const b = x.g % 3;
    if (x.step === 'push') {
      if (copies(x.g) && x.g + 1 < G) push(x.rank, x.g + 1);
      else if (copies(x.g)) errors.push(`rank ${x.rank} copied stage ${x.g + 1}, past the walk`);
      x.step = 'top';
    } else if (x.step === 'top') {
      if (stageNext(x.g)) fillHalf(x.rank, x.g + 2);
      x.step = 'full';
    } else if (x.step === 'full') {
      x.step = 'read';
    } else if (x.step === 'read') {
      for (const h of [0, 1]) {
        if (me.content[b][h] !== x.g) errors.push(`rank ${x.rank} read stage ${x.g}'s half ${h} holding ${me.content[b][h]}`);
      }
      me.reading[b]++;
      x.step = 'end';
    } else if (x.step === 'end') {
      me.reading[b]--;
      if (bug !== 'localempty') arrive(ctas[1 - x.rank].empty[b], 'remote empty');
      arrive(me.empty[b], 'empty');
      x.g++;
      x.step = x.w === P ? 'push' : 'full';
    }
  }
  const stuck = warps.filter((x) => x.g < G).length;
  return { errors, stuck, landsLeft: lands.length };
}

// ---------------------------------------------------------------------------
describe('TMA tall fold, B\' shared by a copy between the cluster\'s CTAs (PEARL_TALL_CLUSTER_PUSH)', () => {
  test('is off by default, and refused without the cluster build or the ring\'s roles', () => {
    const B = block();
    expect(B).toContain(`#ifndef ${NAME}${NL}#define ${NAME} 0${NL}#endif`);
    const errs = (v) => resolve(B, v).split(NL).filter((l) => l.startsWith('#error'));
    const body = (v) => resolve(B, v).split(NL).filter((l) => l.startsWith('#define PEARL_TALL_PUSH_BODY'));
    const base = { [NAME]: 1, PEARL_TALL_CLUSTER: 1, PEARL_TALL_TMA_ROLES: 1, PEARL_TALL_CLUSTER_BODY: 1 };
    expect(errs(base)).toEqual([]);
    expect(body(base)).toEqual(['#define PEARL_TALL_PUSH_BODY 1']);
    // The host pass (no cluster body) builds it with the body off.
    expect(body({ ...base, PEARL_TALL_CLUSTER_BODY: 0 })).toEqual(['#define PEARL_TALL_PUSH_BODY 0']);
    expect(errs({ ...base, PEARL_TALL_CLUSTER: 0 })).toHaveLength(1);
    expect(errs({ ...base, PEARL_TALL_CLUSTER: 0 })[0]).toContain(NAME);
    expect(errs({ ...base, PEARL_TALL_TMA_ROLES: 0 })).toHaveLength(1);
    expect(errs({ ...base, PEARL_TALL_TMA_ROLES: 0 })[0]).toContain('PEARL_TALL_TMA_ROLES');
    // Off, nothing is refused and the body is off.
    const off = { [NAME]: 0, PEARL_TALL_CLUSTER: 0, PEARL_TALL_TMA_ROLES: 0, PEARL_TALL_CLUSTER_BODY: 0 };
    expect(errs(off)).toEqual([]);
    expect(body(off)).toEqual(['#define PEARL_TALL_PUSH_BODY 0']);
    // It reads the cluster switches, so it comes after them.
    expect(HEADER.indexOf(`#ifndef ${NAME}`)).toBeGreaterThan(HEADER.indexOf('#define PEARL_TALL_CLUSTER_BODY 0'));
    expect(HEADER.indexOf(`#ifndef ${NAME}`)).toBeGreaterThan(HEADER.indexOf('#define PEARL_TALL_TMA_ROLES 1'));
  });

  test('off, the fold and the kernel are the shipped code: nothing of the switch is left', () => {
    for (const [name, text] of [['fold', FOLD], ['kernel', KERNEL]]) {
      const OFF = resolve(text, { PEARL_TALL_PUSH_BODY: 0 });
      expect([name, OFF.match(/PUSH_BODY|pearl_tall_push|pearl_bulk_push|full \+ 16u|STAGE \+ 16u, 1u\)|BN \* SK \/ 2u/g)]).toEqual([name, null]);
    }
    // Everything it adds to the fold sits in #if PEARL_TALL_PUSH_BODY blocks: five of them.
    expect(FOLD.match(/^#if PEARL_TALL_PUSH_BODY$/gm)).toHaveLength(5);
    // With it off the fill, the barriers and the prologue read as before.
    const F = code(resolve(FOLD, { PEARL_TALL_PUSH_BODY: 0 }));
    expect(F).toContain(lines(
      '    pearl_mbar_expect_tx(full, STAGE);',
      '#if PEARL_TALL_CLUSTER_BODY',
      '    if (pearl_cluster_ctarank() == 0u)'));
    expect(F).toContain(lines(
      '#if PEARL_TALL_TMA_ROLES',
      '    pearl_mbar_init(sbase + threadIdx.x * STRIDE + STAGE, 1u);',
      '    pearl_mbar_init(sbase + threadIdx.x * STRIDE + STAGE + 8u, PEARL_TALL_TMA_EMPTY_ARRIVALS);'));
    expect(F).toContain(lines(
      '#if PEARL_TALL_TMA_ROLES',
      '#pragma unroll',
      '    for (uint32_t s = 0; s < SPC; s++) tma_fill(sbase + s * STRIDE, s * SK);',
      '#else'));
    expect(F).toContain(lines(
      '#endif',
      '#if PEARL_TALL_TMA_FILL_PT == 0',
      '        PEARL_TALL_REFILL(PEARL_TALL_PRODUCER_S)'));
  });

  test('on, the fill, barriers, prologue and stage-top copy are the model\'s', () => {
    const ON = code(resolve(FOLD, { PEARL_TALL_PUSH_BODY: 1 }));
    // The fill: this CTA's half of B' on HALF (FULL + 16), its A on FULL, which expects A and the
    // peer's half; the multicast and the whole-box fill are gone.
    expect(ON).toContain(lines(
      '    {',
      '      const uint32_t hr = pearl_cluster_ctarank();',
      '      pearl_mbar_expect_tx(full, STAGE - BN * SK / 2u);',
      '      pearl_mbar_expect_tx(full + 16u, BN * SK / 2u);',
      '      pearl_tma_3d(dst + hr * (BN * SK / 2u), &tmB, 0u, boxB + hr * (BN / 2u), kofs / SK, full + 16u);',
      '      pearl_tma_3d(dst + BN * SK, &tmA, 0u, boxA, kofs / SK, full);',
      '    }',
      '  };'));
    expect(ON).not.toContain('pearl_tma_3d_multicast(');
    expect(ON).not.toContain('pearl_mbar_expect_tx(full, STAGE);');
    // FULL two arrivals, HALF one, behind EMPTY (+8) in the buffer's pad.
    expect(ON).toContain(lines(
      '    pearl_mbar_init(sbase + threadIdx.x * STRIDE + STAGE, 2u);',
      '    pearl_mbar_init(sbase + threadIdx.x * STRIDE + STAGE + 16u, 1u);',
      '    pearl_mbar_init(sbase + threadIdx.x * STRIDE + STAGE + 8u, PEARL_TALL_TMA_EMPTY_ARRIVALS);'));
    // The copy: wait for the half at FULL's parity, copy it to the same offset in the peer, on
    // the peer's FULL, then this CTA's own second arrival.
    expect(ON).toContain(lines(
      '  auto pearl_tall_push = [&](uint32_t bi, uint32_t par) {',
      '    pearl_mbar_wait(bi + STAGE + 16u, par);',
      '    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");',
      '    const uint32_t hr = pearl_cluster_ctarank();',
      '    const uint32_t src = bi + hr * (BN * SK / 2u);',
      '    pearl_bulk_push(pearl_mapa(src, hr ^ 1u), src, BN * SK / 2u, pearl_mapa(bi + STAGE, hr ^ 1u));',
      '    pearl_mbar_arrive(bi + STAGE);',
      '  };'));
    // The prologue: stage 0 whole (both halves and A on FULL), FULL's and HALF's plain arrivals,
    // stage 1 a half. No wait.
    expect(ON).toContain(lines(
      '    pearl_mbar_expect_tx(sbase + STAGE, STAGE);',
      '#pragma unroll',
      '    for (uint32_t h = 0; h < 2u; h++)',
      '      pearl_tma_3d(sbase + h * (BN * SK / 2u), &tmB, 0u, boxB + h * (BN / 2u), 0u, sbase + STAGE);',
      '    pearl_tma_3d(sbase + BN * SK, &tmA, 0u, boxA, 0u, sbase + STAGE);',
      '    pearl_mbar_arrive(sbase + STAGE);',
      '    pearl_mbar_arrive(sbase + STAGE + 16u);',
      '    tma_fill(sbase + STRIDE, SK);'));
    // The stage top: stage g + 1's half (soB at stage 0, soC at stage 1 if stage_next) before
    // the refill of g + 2.
    expect(ON).toContain(lines(
      '        if (threadIdx.x == PRODUCER && (s == 0u || stage_next))',
      '          pearl_tall_push(s == 0u ? soB : soC, s == 0u ? pB : pC);',
      '#if PEARL_TALL_TMA_FILL_PT == 0',
      '        PEARL_TALL_REFILL(PEARL_TALL_PRODUCER_S)'));
    // The roles the copy names: stage 0 reads A, stage 1 reads B, C is the next chunk's stage 0
    // (it becomes A), and each role's parity is its next FULL phase's.
    expect(ON).toContain(lines(
      '        const uint32_t so = s == 0u ? soA : soB;',
      '        const uint32_t bi = s == 0u ? soC : soA;',
      '        const uint32_t rpar = s == 0u ? pA : pB;'));
    expect(ON).toContain(lines('        soA = tso;', '        pA = tpar;'));
    // EMPTY counts both CTAs' warps, every arrival made on both, and the refill acquires at
    // cluster scope: the copy's write into the peer is behind the same wait.
    expect(HEADER).toContain('  (PEARL_TALL_TMA_EMPTY_COUNT * (PEARL_TALL_CLUSTER_BODY ? PEARL_TALL_CLUSTER_SIZE : 1u))');
    expect(FOLD).toContain('#define PEARL_TALL_REFILL_WAIT pearl_mbar_wait_cluster(emptyBi, epar);');
    expect(FOLD).toContain('  pearl_mbar_arrive_remote(pearl_mapa((bar), pearl_cluster_ctarank() ^ 1u)); ');
    // The copy's PTX: distributed shared memory, this CTA's to the peer's, on the peer's barrier.
    expect(KERNEL).toContain(lines(
      '#if PEARL_TALL_PUSH_BODY',
      '// The push build\'s copy (PEARL_TALL_CLUSTER_PUSH): `bytes` of this CTA\'s shared memory at'));
    expect(KERNEL).toContain('      "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"');
    // HALF sits at +16 in the pad, clear of the hasher's stamps at its end (sTr - 112 on).
    expect(HEADER).toContain('#define PEARL_TALL_STAGE_STRIDE_TMA (PEARL_TALL_STAGE_BYTES + 512u)');
    expect(FOLD).toContain('      pearl_tall_hcyc_mark(sTr - 112u + 8u * wc);');
  });

  test('the ring, modelled over several tiles, reads only landed halves and never deadlocks', () => {
    for (let seed = 1; seed <= 200; seed++) {
      const W = 2 + (seed % 3);
      const T = 1 + (seed % 3);
      const r = simulate(seed, { W, T, CH: 2 + (seed % 2) });
      expect([seed, r.errors.slice(0, 2), r.stuck, r.landsLeft]).toEqual([seed, [], 0, 0]);
    }
    // Each broken variant is found in some run: a copy before its half landed, a copy past the
    // walk, a read before the CTA's own half landed, a copy landing on a buffer the peer still
    // reads, and HALF's parity a phase off.
    for (const bug of ['nohalf', 'always', 'full1', 'localempty', 'nohalf0']) {
      let bad = 0;
      for (let seed = 1; seed <= 200; seed++) {
        const r = simulate(seed, { W: 2 + (seed % 2), T: 2, bug });
        if (r.errors.length || r.stuck || r.landsLeft) bad++;
      }
      expect([bug, bad > 0]).toEqual([bug, true]);
    }
  });

  test('the host loads B\' in 128-column boxes for it and names it; CI compiles it', () => {
    expect(HOST).toContain(lines(
      '                                   ctx->foldCluster && PEARL_TALL_CLUSTER_PUSH ? PEARL_TALL_BN / 2u :',
      '                                   PEARL_TALL_BN))) {'));
    expect(HOST).toContain('"2-CTA cluster sharing B by a copy between the CTAs"');
    // The launch is the cluster build's.
    expect(HOST).toContain('    ctx->foldCluster = ctx->foldTma && PEARL_TALL_CLUSTER != 0;');
    expect(CI).toContain('            -DPEARL_TALL_CLUSTER=1 -DPEARL_TALL_CLUSTER_PUSH=1 \\');
    expect(CI).toContain('-c probes/multicast-probe.cu -o cuda-build/push/multicast-probe.o');
    expect(CI).toContain('      - name: Compile the push build of the tall fold and the multicast probe (sm_120, not shipped)');
  });
});
