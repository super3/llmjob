'use strict';

const fs = require('fs');
const path = require('path');
const reader = require('../native/probes/multicast-read');

// earn/native/probes/multicast-probe.cu (item 16, step 2): the TMA fold's B' feed in clusters of
// two CTAs without the mma, which prices what multicast saves (f) and shows which cluster barrier
// pattern hangs on sm_120. multicast-read.js reads its lines; compilers.sh's "multicast" stage
// runs both. Nothing here runs device code. These check:
// - the probe's geometry, PTX, barrier protocol and tile seam are the fold's where they claim to be;
// - every mbarrier wait it makes is bounded, so a hang is reported, and the host leaves cleanly;
// - its ring, modelled step by step for every arm, never refills a buffer a CTA is reading,
//   never lets a barrier run two phases ahead of a waiter, and never deadlocks;
// - the reader's bytes, f (with the stage-time correction) and verdicts;
// - the compilers.sh stage.

const NL = String.fromCharCode(10);
const BS = String.fromCharCode(92);
const rawRead = (file) => fs.readFileSync(file, 'utf8').split(String.fromCharCode(13) + NL).join(NL);
const NATIVE = path.join(__dirname, '..', 'native');
const PROBE = rawRead(path.join(NATIVE, 'probes', 'multicast-probe.cu'));
const COMPILERS = rawRead(path.join(NATIVE, 'probes', 'turing', 'compilers.sh'));
const HEADER = rawRead(path.join(NATIVE, 'src', 'pearl_config.h'));
const KERNEL = rawRead(path.join(NATIVE, 'src', 'pearl_kernel.cu'));
const FOLD = rawRead(path.join(NATIVE, 'src', 'pearl_fold_tall.cuh'));
const HOSTSRC = rawRead(path.join(NATIVE, 'src', 'pearl_host.cu'));
const lines = (...l) => l.join(NL);

// The value of `constexpr uint32_t NAME = expr;` in the probe, with the names before it.
function probeConsts() {
  const env = {};
  for (const m of PROBE.matchAll(/^constexpr uint32_t (.*?);/gm)) {
    for (const part of m[1].split(/,\s*(?=[A-Z_]+ = )/)) {
      const [name, expr] = part.split(' = ');
      const js = expr.replace(/\b(\d+)u\b/g, '$1').replace(/\b[A-Z_]+\b/g, (n) => {
        expect([n, n in env]).toEqual([n, true]);
        return String(env[n]);
      });
      env[name.trim()] = Function(`return (${js});`)();
    }
  }
  return env;
}

// ---------------------------------------------------------------------------
// A model of the probe's ring: two CTAs of W warps (warps below H = W / 2 hash at cAH's seam; warp
// H has the producer), three buffers with a FULL and an EMPTY mbarrier each, TMA boxes that land
// later, and a random scheduler. It follows the kernel's protocol line for line (pinned below):
// who inits what count, who arrives where, which phase each wait waits for, what each arm's fill
// issues, and cAH's seam (a named barrier for warps w and w + H every KB stages) and hand-off.
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

// protocol: 'probe' (the kernel's), or 'oneway' (rank 0's producer does not arrive on rank 1's
// FULL: the hole the kernel closes, kept to show the model finds it).
function simulate(arm, seed, { W = 2, S = 30, KB = 6, protocol = 'probe' } = {}) {
  const BOXB = 16384;
  const BOXA = 12288;
  const SYM = arm === 'c' || arm === 'cA' || arm === 'cAH';
  const OWNA = arm === 'cA' || arm === 'cAH';
  const SEAM = arm === 'cAH';
  const H = Math.floor(W / 2);
  const rand = rng(seed);
  const bar = (count) => ({ count, pending: count, phase: 0, tx: 0 });
  const errors = [];
  const ctas = [0, 1].map((rank) => ({
    full: [0, 1, 2].map(() => bar(!SYM && rank === 1 && protocol === 'probe' ? 2 : 1)),
    empty: [0, 1, 2].map(() => bar(SYM || rank === 0 ? 2 * W : W)),
    content: [-1, -1, -1],   // the use of the B' box the buffer holds
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
  // Waiting for use k's phase to complete, by parity, as try_wait.parity does.
  const ready = (b, k, what) => {
    if (b.phase > k + 1) errors.push(`${what}: barrier two phases past use ${k}`);
    return (b.phase & 1) !== (k & 1);
  };
  const lands = [];
  const loads = (rank) => !(arm === 'e' || (arm === 'd' && rank === 1));
  const fill = (rank, n) => {
    const b = n % 3;
    const u = Math.floor(n / 3);
    const me = ctas[rank];
    if (!SYM && rank === 0 && protocol === 'probe') arrive(ctas[1].full[b], 'remote full');
    if (!loads(rank)) {
      arrive(me.full[b], 'full');
      return;
    }
    expectTx(me.full[b], OWNA ? BOXB + BOXA : BOXB, 'full');
    if (arm === 'a' || arm === 'd') lands.push({ rank, b, u, bytes: BOXB, isB: true });
    else if (rank === 0) for (const r of [0, 1]) lands.push({ rank: r, b, u, bytes: BOXB, isB: true });
    if (OWNA) lands.push({ rank, b, u, bytes: BOXA, isB: false });
  };
  // Each warp: the seam's named barrier (cAH, at a tile's first stage but the first), then at the
  // top of stage g the producer warp refills g + 2 behind its EMPTY wait; then FULL wait, read
  // start, read end and the EMPTY arrivals; then cAH's hand-off (warps H and up, after a tile's
  // second stage).
  const seams = {};
  const seamKey = (x) => `${x.rank}:${x.w % H}:${x.g}`;
  const begin = (x) => (x.w === H ? 'top' : 'full');
  const next = (x) => {
    x.g++;
    x.step = SEAM && x.g % KB === 0 && x.g < S ? 'seam' : begin(x);
  };
  const warps = [];
  for (const rank of [0, 1]) for (let w = 0; w < W; w++) warps.push({ rank, w, g: 0, step: w === H ? 'top' : 'full' });
  for (const rank of [0, 1]) {
    fill(rank, 0);
    fill(rank, 1);
  }
  const bias = warps.map(() => 0.2 + rand());
  for (let steps = 0; steps < 100000; steps++) {
    const can = [];
    warps.forEach((x, i) => {
      if (x.g >= S) return;
      const me = ctas[x.rank];
      const b = x.g % 3;
      if (x.step === 'top') {
        const n = x.g + 2;
        if (n < S && n >= 3 && !ready(me.empty[n % 3], Math.floor(n / 3) - 1, 'empty')) return;
      } else if (x.step === 'full') {
        if (!ready(me.full[b], Math.floor(x.g / 3), 'full')) return;
      } else if (x.step === 'seamwait') {
        if (seams[seamKey(x)] !== 2) return;
      } else if (x.step === 'handoff') {
        const h = x.g - 1;
        if (!ready(me.empty[h % 3], Math.floor(h / 3), 'handoff')) return;
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
      if (l.isB) {
        if (t.reading[l.b] > 0) errors.push(`B' box of use ${l.u} landed on rank ${l.rank} buffer ${l.b} while it was read`);
        t.content[l.b] = l.u;
      }
      t.full[l.b].tx -= l.bytes;
      complete(t.full[l.b]);
      continue;
    }
    const x = warps[c.i];
    const me = ctas[x.rank];
    const b = x.g % 3;
    const u = Math.floor(x.g / 3);
    if (x.step === 'seam') {
      seams[seamKey(x)] = (seams[seamKey(x)] || 0) + 1;
      x.step = 'seamwait';
    } else if (x.step === 'seamwait') {
      x.step = begin(x);
    } else if (x.step === 'top') {
      if (x.g + 2 < S) fill(x.rank, x.g + 2);
      x.step = 'full';
    } else if (x.step === 'full') {
      x.step = 'read';
    } else if (x.step === 'read') {
      if (loads(0) && (arm !== 'd' || x.rank === 0) && me.content[b] !== u)
        errors.push(`rank ${x.rank} read buffer ${b} for use ${u} holding ${me.content[b]}`);
      me.reading[b]++;
      x.step = 'end';
    } else if (x.step === 'end') {
      me.reading[b]--;
      if (SYM) {
        arrive(ctas[1 - x.rank].empty[b], 'remote empty');
        arrive(me.empty[b], 'empty');
      } else {
        if (x.rank === 1) arrive(ctas[0].empty[b], 'remote empty');
        arrive(me.empty[b], 'empty');
      }
      if (SEAM && x.g % KB === 1 && x.w >= H) x.step = 'handoff';
      else next(x);
    } else {
      next(x);
    }
  }
  const stuck = warps.filter((x) => x.g < S).length;
  return { errors, stuck, seams: Object.keys(seams).length };
}

// A probe log: ROUND lines for energy a pair-stage E (joules) by arm, and whatever else is given.
// time[arm] stretches that arm's window (the same stages in more seconds), mhz[arm] its clock.
function log(E, { rounds = 3, extra = [], phase2 = { b: 1, c: 1, cA: 1 }, cah = true, pace = 3240, cyc = 3250,
  time = {}, mhz = {}, smoke = ['c', 'cA', 'cAH'], end = 'DONE' } = {}) {
  const out = ['CARD name=NVIDIA GeForce RTX 5090 cc=12.0 sms=170 l2_mb=96 cluster_launch=1 smem_optin=101376 pace=3240 '
    + 'secs=6.0 rounds=3 settle=1.5 wait_ms=1000 hash=3000 own_dst=cta', 'NVML yes energy=yes'];
  for (const a of Object.keys(reader.ARMS)) out.push(`CLUSTERS arm=${a} rc=0 (cudaSuccess) smem_attr=0 clusters=85 ctas=170`);
  out.push('SMOKE arm=e launch=0 sync=0 tiles=4 ctas=170 stages=21760 cyc_stage=3250 bad=0',
    'SMOKE arm=e launch=0 sync=0 tiles=512 ctas=170 stages=2785280 cyc_stage=3241 bad=0',
    'LAUNCH tiles=7 stages=224 cycles_per_ms=2900000 stage_cycles=3241');
  const stages = 170 * 4000 * 32;
  const row = (r, a, e) => {
    const secs = 6 * (1 + (time[a] || 0));
    const j = e * stages / 2;
    return `ROUND r=${r} arm=${a} secs=${secs.toFixed(6)} launches=20 ctas=170 cta_stages=${stages} `
      + `fill_gb=${(stages / 2 * reader.ARMS[a].fill / 1e9).toFixed(3)} l2_gb=${(stages / 2 * reader.ARMS[a].l2 / 1e9).toFixed(3)} `
      + `j=${j.toFixed(9)} j_src=energy w=${(j / secs).toFixed(4)} mhz=${(2900 * (1 + (mhz[a] || 0))).toFixed(1)} c=60 `
      + `cyc_stage=${cyc} pace=${pace} bad=0`;
  };
  for (let r = 1; r <= rounds; r++) for (const a of ['e', 'a', 'd', 'b']) if (E[a] !== undefined) out.push(row(r, a, E[a] * (1 + 0.01 * (r - 2))));
  for (const a of smoke) out.push(`SMOKE arm=${a} launch=0 sync=0 tiles=8 ctas=170 stages=43520 cyc_stage=3250 bad=0`);
  for (let r = rounds + 1; r <= rounds + 2; r++) for (const a of ['b', 'c', 'cA']) if (phase2[a] && E[a] !== undefined) out.push(row(r, a, E[a]));
  if (cah && E.cAH !== undefined) out.push(row(rounds + 3, 'cAH', E.cAH));
  return [...out, ...extra, end].join(NL);
}
const E0 = { e: 100e-9, a: 200e-9, d: 150e-9, b: 165e-9, c: 170e-9, cA: 230e-9, cAH: 240e-9 };
// Phase 2 with b alone: c and cA stopped before their rounds.
const NO_C = { b: 1, c: 0, cA: 0 };
const summary = (text) => reader.summarize(reader.parse(text));
const verdict = (text) => summary(text).filter((l) => l.startsWith('read: verdict:'));

// ---------------------------------------------------------------------------
describe('RTX 50 multicast probe (multicast-probe.cu, multicast-read.js)', () => {
  test('the probe\'s geometry is the fold\'s: boxes, ring, shared footprint, producer and lane bases', () => {
    const P = probeConsts();
    expect([P.THREADS, P.SK, P.BN, P.BM, P.BOXB, P.BOXA, P.STAGE, P.STRIDE, P.NST, P.SMEM, P.PRODUCER, P.KB])
      .toEqual([256, 64, 256, 192, 16384, 12288, 28672, 29184, 3, 99840, 128, 32]);
    expect(HEADER).toContain('#define PEARL_TALL_THREADS 256u');
    expect(HEADER).toContain('#define PEARL_TALL_BM (2u * PEARL_TALL_ROW_TILES * 16u)          // 192 rows of A');
    expect(HEADER).toContain('#define PEARL_TALL_BN (4u * 64u)                                 // 256 columns of B');
    expect(HEADER).toContain('#define PEARL_TALL_STAGE_K 64u');
    expect(HEADER).toContain('#define PEARL_TALL_STAGES 3u');
    expect(HEADER).toContain('#define PEARL_TALL_STAGE_STRIDE_TMA (PEARL_TALL_STAGE_BYTES + 512u)');
    expect(HEADER).toContain(lines('#define PEARL_TALL_SMEM_TMA ' + BS,
      '  (PEARL_TALL_STAGES * PEARL_TALL_STAGE_STRIDE_TMA + PEARL_TALL_BM * PEARL_TALL_BN / 4u)'));
    expect(3 * (P.STAGE + 512) + 192 * 256 / 4).toBe(P.SMEM);
    expect(FOLD).toContain('  constexpr uint32_t PRODUCER = 4u * 32u;');
    // The fold's lane bases (the roles build), with the column slot as warp & 3.
    expect(FOLD).toContain('  const uint32_t f = (lane >> 1) & 3u;');
    expect(FOLD).toContain('  const uint32_t blcol = (lane & 7u) + ((lane >> 4) & 1u) * 8u;');
    expect(FOLD).toContain('  uint32_t bLane0 = (wc * 64u + blcol) * SK + 16u * (((lane >> 3) & 1u) ^ f);');
    expect(FOLD).toContain('  const uint32_t wc = warp & 3u;');
    expect(PROBE).toContain('  const uint32_t bLane = ((warp & 3u) * 64u + blcol) * SK + 16u * (((lane >> 3) & 1u) ^ f);');
    expect(FOLD).toContain('                                  (bS ^ kt) + (p + 1u) * 16u * SK);');
    expect(PROBE).toContain('        ldmatrix_x4(r0, r1, r2, r3, ((buf + bLane) ^ (t * 32u)) + q * 16u * SK);');
    // The boxes as the host encodes the fold's: 64 bytes by the box's rows by one k-block, SWIZZLE_64B.
    expect(HOSTSRC).toContain('  const cuuint32_t box[3] = {kBlock, boxRows, 1u};');
    expect(PROBE).toContain('  const cuuint32_t box[3] = {SK, boxRows, 1u};');
    for (const t of ['CU_TENSOR_MAP_SWIZZLE_64B', 'CU_TENSOR_MAP_L2_PROMOTION_L2_256B', 'CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE']) {
      expect(HOSTSRC).toContain(t);
      expect(PROBE).toContain(t);
    }
    expect(PROBE).toContain('  const cuuint64_t dims[3] = {(cuuint64_t)SK, (cuuint64_t)rows, (cuuint64_t)KB};');
  });

  test('c, cA and cAH use the fold\'s cluster PTX and barrier protocol; e, a, b and d share another', () => {
    // The same instructions as the fold's cluster helpers in pearl_kernel.cu.
    for (const ptx of [
      'mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];',
      'mbarrier.arrive.release.cluster.shared::cta.b64 st, [%0];',
      'mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64',
      'mbarrier.try_wait.parity.shared::cta.b64',
      'mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;',
      'mapa.shared::cluster.u32 %0, %1, %2;',
      'barrier.cluster.arrive.release.aligned;' + BS + 'n' + BS + 'tbarrier.cluster.wait.acquire.aligned;',
      'fence.mbarrier_init.release.cluster;',
      // The multicast (pearl_tma_3d_multicast), and a CTA's own box in the fold's form (pearl_tma_3d).
      '".multicast::cluster [%0], [%1, {%2, %3, %4}], [%5], %6;"',
      '"cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"',
      '"cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes"',
      'asm volatile("bar.sync %0, %1;" ::"r"(',
    ]) {
      expect([ptx, KERNEL.includes(ptx), PROBE.includes(ptx)]).toEqual([ptx, true, true]);
    }
    // The own box takes the fold's .shared::cta form wherever the toolkit has it (12.8 on: the
    // boxes build with 13.3), and CARD says which form ran.
    expect(PROBE).toContain(lines(
      '  asm volatile(',
      '#if MC_SHARED_CTA',
      '      "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes"',
      '#else',
      '      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"',
      '#endif',
      '      " [%0], [%1, {%2, %3, %4}], [%5];"'));
    expect(PROBE).toContain(lines('#ifndef MC_SHARED_CTA',
      '#if defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ * 100 + __CUDACC_VER_MINOR__ >= 1208'));
    expect(PROBE).toContain('MC_SHARED_CTA ? "cta" : "cluster");');
    expect(PROBE.match(/tma_3d\(/g)).toHaveLength(3);   // the helper and the two own boxes
    // The fold's arrival, made twice: on the peer's EMPTY through mapa, then its own.
    expect(FOLD).toContain(lines(
      '#define PEARL_TALL_EMPTY_ARRIVE(bar)                       ' + BS,
      '  pearl_mbar_arrive_remote(pearl_mapa((bar), pearl_cluster_ctarank() ^ 1u)); ' + BS,
      '  pearl_mbar_arrive_cluster(bar);'));
    expect(PROBE).toContain(lines(
      '      if (SYM) {',
      '        mbar_arrive_remote(mapa(empty, rank ^ 1u));',
      '        mbar_arrive_cluster(empty);',
      '      } else {',
      '        if (rank == 1u) mbar_arrive_remote(mapa(empty, 0u));',
      '        mbar_arrive(empty);',
      '      }'));
    expect(PROBE).toContain(lines(
      '  constexpr bool SYM = ARM == ARM_C || ARM == ARM_CA || ARM == ARM_CAH;',
      '  constexpr bool OWNA = ARM == ARM_CA || ARM == ARM_CAH;',
      '  constexpr bool SEAM = ARM == ARM_CAH;'));
    // EMPTY counts both CTAs' eight warps in the fold's pattern, as PEARL_TALL_TMA_EMPTY_ARRIVALS does;
    // FULL is the producer's alone there, as in the fold.
    expect(HEADER).toContain('  (PEARL_TALL_TMA_EMPTY_COUNT * (PEARL_TALL_CLUSTER_BODY ? PEARL_TALL_CLUSTER_SIZE : 1u))');
    expect(FOLD).toContain('    pearl_mbar_init(sbase + threadIdx.x * STRIDE + STAGE, 1u);');
    expect(PROBE).toContain('    mbar_init(sbase + threadIdx.x * STRIDE + STAGE, !SYM && rank == 1u ? 2u : 1u);');
    expect(PROBE).toContain('    mbar_init(sbase + threadIdx.x * STRIDE + STAGE + 8u, SYM || rank == 0u ? 16u : 8u);');
    // The fills, arm by arm, as the model has them.
    expect(PROBE).toContain(lines(
      '    if (!SYM && rank == 0u) mbar_arrive_remote(mapa(full, 1u));',
      '    if (ARM == ARM_E || (ARM == ARM_D && rank != 0u)) {',
      '      mbar_arrive(full);',
      '      return;',
      '    }',
      '    expect_tx(full, OWNA ? STAGE : BOXB);',
      '    if (ARM == ARM_A || ARM == ARM_D)',
      '      tma_3d(buf, &tmB.map, 0u, col, kb, full);',
      '    else if (rank == 0u)',
      '      tma_3d_mc(buf, &tmB.map, 0u, col, kb, full, (uint16_t)3u);',
      '    if (OWNA) tma_3d(buf + BOXB, &tmA.map, 0u, row, kb, full);'));
    // The producer refills two ahead behind its EMPTY wait for the buffer's last use; the warps
    // wait on FULL for theirs. Cluster-scope acquire wherever the peer's warps arrive.
    expect(PROBE).toContain('      const uint32_t par = (n / NST - 1u) & 1u;');
    expect(PROBE).toContain(lines(
      '        ok = SYM || rank == 0u ? wait_bounded<true>(empty, par, waitNs, abortFlag, st, 1u, n)',
      '                               : wait_bounded<false>(empty, par, waitNs, abortFlag, st, 1u, n);'));
    expect(PROBE).toContain('wait_bounded<false>(buf + STAGE, (g / NST) & 1u, waitNs, abortFlag, st, 0u, g);');
  });

  test('cAH has the fold\'s tile seam: the named barrier, the hasher behind, and the hand-off', () => {
    // The fold: at each tile's end warps w and w + 4 meet, w (wr 0) hashes, and at chunk 0's end
    // every thread of wr 1 waits at CTA scope on the tile's stage 0 EMPTY, which in the cluster
    // build counts the peer's arrivals too.
    expect(FOLD).toContain('    pearl_bar_sync(1u + wc, 64u);');
    expect(FOLD).toContain('      if (chunk == 0u && wr != 0u) pearl_mbar_wait(soA + STAGE + 8u, pA);');
    expect(FOLD).toContain('  const uint32_t wr = warp >> 2;');
    expect(KERNEL).toContain(lines(
      '__device__ __forceinline__ void pearl_bar_sync(uint32_t id, uint32_t count) {',
      '  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(count) : "memory");'));
    // The probe: the same barrier, the hasher's sleep, and the same wait, at the same stages of a
    // tile (its first stage but the first tile's; its second), at CTA scope, bounded.
    expect(PROBE).toContain(lines(
      '    if (SEAM && g > 0u && g % KB == 0u) {'));
    expect(PROBE).toContain(lines(
      '      const unsigned quit = __shfl_sync(0xFFFFFFFFu, lane == 0u ? *(volatile unsigned *)abortFlag : 0u, 0);',
      '      if (quit) break;',
      '      asm volatile("bar.sync %0, %1;" ::"r"(1u + (warp & 3u)), "r"(64u) : "memory");',
      '      if (warp < 4u) {',
      '        const long long end = clock64() + hashCyc;',
      '        while (clock64() < end) __nanosleep(32);',
      '      }',
      '      due = clock64();',
      '    }',
      '    // The producer refills two stages ahead, once both CTAs (or this one) released the buffer.'));
    expect(PROBE).toContain(lines(
      '    if (SEAM && g % KB == 1u && warp >= 4u) {',
      '      // The fold\'s hand-off (cAH): at the end of the tile\'s first chunk every thread of warps'));
    expect(PROBE).toContain(lines(
      '      const uint32_t h = g - 1u;',
      '      const uint32_t empty = sbase + (h % NST) * STRIDE + STAGE + 8u;',
      '      const bool held = wait_bounded<false>(empty, (h / NST) & 1u, waitNs, abortFlag, st, 2u, h);',
      '      if (__any_sync(0xFFFFFFFFu, !held)) break;'));
    // The hand-off comes after the stage's own EMPTY arrival, as the fold's after its chunk's.
    expect(PROBE.indexOf('    if (SEAM && g % KB == 1u && warp >= 4u) {'))
      .toBeGreaterThan(PROBE.indexOf('        mbar_arrive_cluster(empty);'));
    // The hash's default is the 4090's measured phase, and the host passes it to every arm.
    expect(HEADER).toContain('// 48 regions, about 3,000 cycles, while its partner runs on alone for about 1,700 before it');
    expect(PROBE).toContain('  p.hashCyc = argc > 7 ? (uint32_t)atoi(argv[7]) : 3000u;');
    expect(PROBE).toContain('kernel_for(a.arm), p.tmB, p.tmA, tiles, p.pace, p.hashCyc,');
    expect(PROBE).toContain('s->role == 2u ? "handoff-empty" : s->role ? "producer-empty" : "consumer-full"');
  });

  test('every wait gives up and says so, no CTA leaves while its peer may arrive on it, and the host exits cleanly', () => {
    // try_wait only inside mbar_try, mbar_try only inside wait_bounded, which gives up on time or abort.
    expect(PROBE.match(/mbarrier\.try_wait/g)).toHaveLength(2);
    expect(PROBE.match(/mbar_try<CL>\(bar, parity\)/g)).toHaveLength(2);
    expect(PROBE.match(/mbar_try</g)).toHaveLength(2);
    expect(PROBE).toContain(lines(
      '    if (*(volatile unsigned *)abortFlag) return false;',
      '    const uint64_t dt = gtime() - t0;',
      '    if (dt > waitNs) {',
      '      if (atomicCAS(abortFlag, 0u, 1u) == 0u) {'));
    // The producer's EMPTY (two scopes), the warps' FULL, and cAH's hand-off.
    expect(PROBE.match(/wait_bounded<(true|false)>\(/g)).toHaveLength(4);
    // A warp leaves the stage loop together (a vote), so every ldmatrix and the cluster barrier
    // stay converged; the cluster barrier is the last thing before the kernel returns.
    expect(PROBE).toContain('    if (__any_sync(0xFFFFFFFFu, !got)) break;');
    expect(PROBE).toContain(lines('  __syncwarp();', '  // No CTA leaves while its peer may still arrive on it or multicast into it.',
      '  cluster_sync();', '#else'));
    // The host: a deadline on every launch, and the status in host memory it can read regardless.
    expect(PROBE).toContain('      printf("HANG arm=%s launch did not finish in %.1f s (expected %.2f s)%s' + BS + 'n", a.name, now_s() - t0, expect,');
    expect(PROBE).toContain('  cudaHostAlloc(&p.stHost, sizeof(Status), cudaHostAllocMapped);');
    // The sampler thread starts after every early return, so none leaves it running (which would
    // call std::terminate); after it, the process ends at the join or at launch()'s _Exit.
    const main = PROBE.slice(PROBE.indexOf('int main('));
    const start = main.indexOf('  std::thread samp(sampler);');
    expect(start).toBeGreaterThan(main.lastIndexOf('return 2;'));
    expect(main.slice(start).match(/\breturn\b/g)).toEqual(['return']);
    expect(main.slice(start)).toContain(lines('  g_quit = true;', '  samp.join();', '  printf("DONE' + BS + 'n");', '  return 0;'));
    expect(PROBE.match(/std::_Exit\(/g)).toHaveLength(2);
    // The arms that can hang run after the rounds that give f, and cAH, whose named barrier
    // cannot give up, last of all.
    const order = ['rounds_over({ARM_E, ARM_A, ARM_D, ARM_B}, 1, rounds);', 'for (int arm : {ARM_C, ARM_CA}) launch(p, ARMS[arm], 8u, true);',
      'rounds_over({ARM_B, ARM_C, ARM_CA}, rounds + 1, phase2);', 'launch(p, ARMS[ARM_CAH], 8u, true);',
      'rounds_over({ARM_CAH}, rounds + phase2 + 1, 1);', 'g_quit = true;'];
    for (const o of order) expect([o, PROBE.includes(o)]).toEqual([o, true]);
    for (let i = 1; i < order.length; i++) expect(PROBE.indexOf(order[i])).toBeGreaterThan(PROBE.indexOf(order[i - 1]));
    expect(PROBE.indexOf(order[0])).toBeGreaterThan(start);
  });

  test('the ring, modelled for every arm, never refills a buffer being read and never deadlocks', () => {
    for (const arm of Object.keys(reader.ARMS)) {
      for (let seed = 1; seed <= 150; seed++) {
        const W = arm === 'cAH' ? 2 + 2 * (seed % 2) : 2 + (seed % 2);
        const r = simulate(arm, seed, { W, S: 24 });
        expect([arm, seed, r.errors.slice(0, 2), r.stuck]).toEqual([arm, seed, [], 0]);
        // cAH met at every seam: 3 seams in 24 stages of tiles of 6, a pair of warps each, both CTAs.
        if (arm === 'cAH') expect(r.seams).toBe(3 * (W / 2) * 2);
      }
    }
    // Without rank 0's arrival on rank 1's FULL, rank 1 can run a phase ahead in the arms where it
    // loads its own box (or nothing): the model finds it, so it can find such holes.
    for (const arm of ['a', 'e']) {
      let bad = 0;
      for (let seed = 1; seed <= 150; seed++) {
        const r = simulate(arm, seed, { protocol: 'oneway', S: 24 });
        if (r.errors.length || r.stuck) bad++;
      }
      expect([arm, bad > 0]).toEqual([arm, true]);
    }
  });

  test('the reader\'s table is the probe\'s, and bytes add up', () => {
    expect(reader.BOXB).toBe(256 * 64);
    expect(reader.BOXA).toBe(192 * 64);
    const P = probeConsts();
    const table = {};
    for (const m of PROBE.matchAll(/\{"(\w+)", ARM_\w+, ([^,]+), ([^,]+), (true|false), (true|false)\}/g)) {
      const ev = (e) => Function(`return (${e.replace(/\b(\d+)u\b/g, '$1').replace(/BOXB/g, P.BOXB).replace(/BOXA/g, P.BOXA)});`)();
      table[m[1]] = { fill: ev(m[2]), l2: ev(m[3]) };
    }
    expect(table).toEqual(reader.ARMS);
    expect(PROBE).toContain('enum Arm { ARM_E, ARM_A, ARM_B, ARM_D, ARM_C, ARM_CA, ARM_CAH, NARMS };');
    // Multicast fills what a fills from half the L2 reads; d fills one box from one read.
    expect(reader.ARMS.b.fill).toBe(reader.ARMS.a.fill);
    expect(reader.ARMS.b.l2 * 2).toBe(reader.ARMS.a.l2);
    expect(reader.ARMS.d).toEqual({ fill: reader.BOXB, l2: reader.BOXB });
    expect(reader.energyPerPairStage({ j: 3, cta_stages: 6 })).toBe(1);
    expect(reader.energyPerPairStage({ j: 3, cta_stages: 0 })).toBeNaN();
    // A stage's time: a CTA's stages one after another, all CTAs at once.
    expect(reader.stageTime({ secs: 6, ctas: 170, cta_stages: 170 * 1000 })).toBeCloseTo(0.006, 12);
    expect(reader.stageTime({ secs: 6, ctas: 170, cta_stages: 0 })).toBeNaN();
    // Brought to another stage time at e's power: 1 W over 6 us more a stage (6 ms over 1000
    // stages), shared by 170 CTAs, two a pair.
    const ref = { secs: 6, ctas: 170, cta_stages: 170 * 1000, j: 0 };
    const slow = { secs: 6.006, ctas: 170, cta_stages: 170 * 1000, j: 85000 };
    expect(reader.evenTime(slow, ref, 1)).toBeCloseTo(1 - 6e-6 * 2 / 170, 12);
    expect(reader.evenTime(ref, ref, 1)).toBe(0);
    // The probe prints fill_gb and l2_gb from the same table.
    expect(PROBE).toContain('stages / 2.0 * a.fillPair / 1e9,');
    expect(PROBE).toContain('stages / 2.0 * a.l2Pair / 1e9, j,');
    expect(PROBE).toContain('"j_src=%s w=%.2f mhz=%.0f c=%.0f cyc_stage=%.0f pace=%u bad=%d' + BS + 'n",');
    expect(PROBE).toContain('printf("ROUND r=%d arm=%s secs=%.3f launches=%d ctas=%d cta_stages=%llu fill_gb=%.3f l2_gb=%.3f j=%.3f "');
  });

  test('f, the box\'s cost, the pattern costs and the gate', () => {
    expect(reader.fShare({ a: 200, b: 165, d: 150 })).toBeCloseTo(0.7, 12);
    expect(reader.fShare({ a: 150, b: 140, d: 150 })).toBeNaN();
    const out = summary(log(E0)).join(NL);
    expect(out).toContain('read: clusters granted: 85 of two CTAs (query rc 0 for every arm)');
    expect(out).toContain('read: own boxes in the fold\'s .shared::cta form');
    expect(out).toMatch(/read: f = 0\.70 \(3 rounds, 0\.7\d to 0\.7\d; as measured 0\.70\): multicast saves 70% of a whole B' box's cost/);
    expect(out).toContain('read: gate f >= 0.5: pass');
    expect(out).toContain('read: 1% more stage time in one arm moves f by 0.02 here');
    // One box: 50 nJ over 16384 bytes, 3.05 pJ a byte; two boxes cost twice one.
    expect(out).toContain('read: one box: 3.1 pJ a byte filled (d - e); two cost 2.00 times one');
    expect(out).toContain('read: c\'s barrier protocol (cluster-wide EMPTY, FULL without the remote arrival): c - b = 5.0 nJ '
      + 'a pair-stage (+3.0% of b)');
    expect(out).toContain('read: the A\' boxes: cA - c = 60.0 nJ a pair-stage');
    expect(out).toContain('read: cAH: 240.0 nJ a pair-stage');
    expect(out).toContain('(the seam\'s hash sleep is in its stages: not compared)');
    expect(out).toContain('read: verdict: nothing stuck, up to the fold\'s whole barrier pattern with its tile seam (cAH)');
    expect(out).not.toMatch(/not held|checksums wrong|mean power|did not run alike|other side/);
    const fail = summary(log({ ...E0, b: 190e-9 })).join(NL);
    expect(fail).toMatch(/read: f = 0\.2\d/);
    expect(fail).toContain('read: gate f >= 0.5: fail, drop #1 and #3');
    const free = summary(log({ ...E0, a: 150e-9, d: 150e-9 })).join(NL);
    expect(free).toContain('read: f: a box costs nothing measurable here');
    expect(summary(log({ e: 1e-7 })).join(NL)).toContain('read: f: no round has a, b, d and e');
  });

  test('arms that ran slower or at another clock are flagged, and f is taken at e\'s stage time', () => {
    // The true costs give f = 0.48, a fail. Arm a ran 3% longer a stage, at e's power that much
    // more energy: as measured f reads 0.51, a pass. Brought to e's stage time it is 0.48 again.
    const E = { e: 100e-9, a: 200e-9, d: 150e-9, b: 176e-9 };
    const out = summary(log({ ...E, a: 203e-9 }, { time: { a: 0.03 }, rounds: 1 })).join(NL);
    expect(out).toContain('read: f = 0.48 (1 rounds, 0.48 to 0.48; as measured 0.51)');
    expect(out).toContain('read: gate f >= 0.5: fail, drop #1 and #3');
    expect(out).toContain('read: f as measured falls on the other side of the gate');
    expect(out).toContain('read: a against e in one round: stage time +3.00%, clock +0.00%, past +0.5%: the arms did not run alike');
    // A clock 1% lower is flagged too, and within 0.5% nothing is.
    expect(summary(log(E, { mhz: { d: -0.01 } })).join(NL)).toContain('read: d against e in one round: stage time +0.00%, clock -1.00%');
    expect(summary(log(E, { mhz: { d: -0.004 }, time: { b: 0.004 } })).join(NL)).not.toContain('did not run alike');
    // In the rounds without e, c and cA are brought to b's stage time at e's power from before.
    const c = summary(log({ ...E0, c: 175e-9 }, { time: { c: 0.05 } })).join(NL);
    expect(c).toContain('c - b = 5.0 nJ a pair-stage (+3.0% of b)');
    expect(c).toContain('read: c against b in one round: stage time +5.00%');
    const noE = summary(log({ ...E0, e: undefined })).join(NL);
    expect(noE).toContain('read: no e rounds: c - b and cA - c as measured');
    // A pace not held: a stage over 1% past it.
    expect(summary(log(E0, { cyc: 3280 })).join(NL)).toContain('pace 3240 not held');
    expect(summary(log(E0, { cyc: 3270 })).join(NL)).not.toContain('not held');
  });

  test('the verdict on what hangs climbs the ladder: b, then c, cA and cAH', () => {
    const t = (arm, role, rank = 1) => `TIMEOUT arm=${arm} role=${role} block=17 rank=${rank} thread=128 stage=5 parity=0 `
      + 'bar=0x6d10 state=0x0000000800000000 waited_ms=1000';
    // e, a or d: the probe itself, so nothing above them is read.
    expect(verdict(log(E0, { extra: [t('a', 'consumer-full')] }))[0])
      .toContain('read: verdict: a stuck: the probe\'s own coupling or its plain loads stick, so f below cannot be trusted');
    // b and not a: the multicast never completes, unless c (the same multicast) ran clean.
    const bOnly = verdict(log(E0, { phase2: { b: 0, c: 0, cA: 0 }, cah: false, smoke: [],
      extra: [t('b', 'consumer-full'), t('c', 'consumer-full')], end: 'DONE' }))[0];
    expect(bOnly).toContain('b sticks and a, d and e do not: b differs from a only in its load, so the multicast\'s bytes or '
      + 'its complete_tx never arrive on this card (first to give up: a consumer-full wait on rank 1, stage 5)');
    expect(bOnly).toContain('drop #1 to #3, all multicast');
    expect(verdict(log(E0, { phase2: { b: 0, c: 1, cA: 1 }, extra: [t('b', 'consumer-full')] }))[0])
      .toContain('b sticks and c, the same multicast with the fold\'s barriers, does not: the probe\'s coupling beside the multicast sticks');
    // c and not b; cA and not c; cAH and not cA.
    expect(verdict(log(E0, { phase2: NO_C, cah: false, extra: [t('c', 'producer-empty')] }))[0])
      .toContain('c sticks and b does not (first to give up: a producer-empty wait on rank 1, stage 5): the fold\'s barrier '
        + 'protocol hangs on this card (EMPTY counting both CTAs, with mapa remote arrives and cluster-scope acquires, and '
        + 'FULL with no remote arrival; c changes both, so this does not say which)');
    expect(verdict(log(E0, { phase2: { b: 1, c: 1, cA: 0 }, cah: false, extra: [t('cA', 'consumer-full')] }))[0])
      .toContain('cA sticks and c does not');
    expect(verdict(log(E0, { cah: false, extra: [t('cAH', 'handoff-empty', 0)] }))[0])
      .toContain('cAH sticks and cA does not (first to give up: a handoff-empty wait on rank 0, stage 5): the fold\'s tile seam hangs');
    // A HANG counts as stuck, and says whether a wait gave up before it.
    const hang = summary(log(E0, { phase2: NO_C, cah: false, smoke: [], end: 'DONE early',
      extra: ['HANG arm=c launch did not finish in 12.0 s (expected 0.30 s); no wait gave up'] })).join(NL);
    expect(hang).toContain('read: verdict: c sticks and b does not: the fold\'s barrier protocol hangs');
    expect(hang).toContain('read: hang in c: the launch never finished, and no wait gave up');
    const after = summary(log(E0, { cah: false, end: 'DONE early',
      extra: [t('cAH', 'handoff-empty'), 'HANG arm=cAH launch did not finish in 12.0 s (expected 0.30 s)'] })).join(NL);
    expect(after).toContain('read: hang in cAH: the launch never finished, after a wait gave up');
    expect(after).toContain('read: timeout in cAH: handoff-empty wait, block 17 (rank 1), stage 5');
    // Arms that never ran give no verdict, and say whether the probe stopped early.
    expect(verdict(log(E0, { phase2: NO_C, cah: false, smoke: [], end: 'DONE early' }))[0])
      .toBe('read: verdict: c, cA, cAH never ran (the probe stopped early): no verdict on what hangs');
  });

  test('a refused launch, a crash and a stop are told apart, and a box log reads the same', () => {
    const p = reader.parse(log(E0, { end: 'DONE early' }));
    expect([p.done, p.early]).toEqual([false, true]);
    expect([reader.parse(log(E0)).done, reader.parse(log(E0)).early]).toEqual([true, false]);
    // The runtime refused the cluster launch at the first arm: no cluster work runs.
    const refused = summary(['CARD name=NVIDIA GeForce RTX 5090 cc=12.0 sms=170', 'NVML yes energy=yes',
      'CLUSTERS arm=e rc=0 (cudaSuccess) smem_attr=0 clusters=85 ctas=170',
      'FAULT arm=e launch rc=801 (cudaErrorNotSupported) last=801 (cudaErrorNotSupported)',
      'FAULT calibration', 'DONE early'].join(NL)).join(NL);
    expect(refused).toContain('read: launch refused for e: arm=e launch rc=801 (cudaErrorNotSupported)');
    expect(refused).toContain('read: fault in the setup: calibration');
    expect(refused).toContain('read: verdict: the card refused the cluster launch itself (e, the first arm, loads nothing): '
      + 'no cluster work runs here at this footprint; drop #1 to #5');
    // A crash once launched is not a refusal; in c it is the fold's pattern, before a, f is lost.
    const crash = summary(log(E0, { phase2: NO_C, cah: false, smoke: [], end: 'DONE early',
      extra: ['FAULT arm=c sync rc=715 (cudaErrorIllegalInstruction)'] })).join(NL);
    expect(crash).toContain('read: c crashed once launched (arm=c sync rc=715 (cudaErrorIllegalInstruction)): a fault, not a '
      + 'refusal and not a hang; the probe stopped there');
    expect(crash).toContain('read: verdict: c crashed on this card: the fold\'s cluster pattern faults rather than hangs here');
    expect(summary(['CARD name=x cc=12.0', 'FAULT arm=a sync rc=719 (cudaErrorLaunchFailure)', 'DONE early'].join(NL)).join(NL))
      .toContain('read: verdict: a crashed: f cannot be read, and the arms after it did not run');
    // Wrong checksums, energy from power, a refused query, a smoke that stopped short.
    const spoiled = log(E0).replace('j_src=energy', 'j_src=power').replace(/^(ROUND .*)bad=0$/m, '$1bad=2')
      .replace('CLUSTERS arm=a rc=0 (cudaSuccess) smem_attr=0 clusters=85 ctas=170',
        'CLUSTERS arm=a rc=801 (cudaErrorNotSupported) smem_attr=0 clusters=0 fallback=85')
      .replace('SMOKE arm=cA launch=0 sync=0 tiles=8 ctas=170 stages=43520 cyc_stage=3250 bad=0',
        'SMOKE arm=cA launch=0 sync=0 tiles=8 ctas=170 stages=43520 cyc_stage=3250 bad=3');
    const s = summary(spoiled).join(NL);
    expect(s).toContain('read: e: energy from mean power');
    expect(s).toMatch(/read: \w+: 2 CTA checksums wrong/);
    expect(s).toContain('read: occupancy query refused or zero for a (rc 801, 0 clusters)');
    expect(s).toContain('read: smoke cA: 3 CTAs stopped short or read bytes that were not the boxes loaded');
    expect(summary('CARD name=x cc=8.9\nSKIP clusters and TMA need sm_90 or newer\nDONE'))
      .toEqual(['read: skipped: the card has no clusters or TMA (sm_90 or newer needed)']);
    const card = reader.parse(log(E0)).card;
    expect([card.name, card.sms, card.own_dst, card.hash]).toEqual(['NVIDIA GeForce RTX 5090', 170, 'cta', 3000]);
    // The probe's prefix in a box log changes nothing, and the reader's own lines are never read back.
    const boxed = log(E0).split(NL).map((l) => `[nvtest] 22:00:00 [compilers] multicast ${l}`).join(NL);
    const plain = summary(log(E0));
    expect(summary(boxed + NL + plain.map((l) => `[nvtest] 22:10:00 [compilers] multicast ${l}`).join(NL))).toEqual(plain);
  });

  test('compilers.sh runs it as the "multicast" stage, last, and only when asked', () => {
    expect(COMPILERS).toContain('STAGES=" ${STAGES:-setup build bench hashrate verify complete probe} "');
    const at = COMPILERS.indexOf('if has multicast; then');
    expect(at).toBeGreaterThan(COMPILERS.indexOf('if has race && [ -f "$OUT/v/base/libpearl.so" ]; then'));
    expect(at).toBeGreaterThan(COMPILERS.indexOf('if has probe && [ -f "$OUT/v/base/ok" ] && [ "$SM" = 75 ]; then'));
    expect(COMPILERS.indexOf('say "done in $(( ($(date +%s) - T0) / 60 )) min"')).toBeGreaterThan(at);
    expect(COMPILERS.slice(at)).toContain(lines(
      'if has multicast; then',
      '  if [ "${SM:-0}" -lt 90 ]; then',
      '    say "multicast: skipped, sm_$SM has no clusters"',
      '  elif ! nvcc -O3 -std=c++17 $GC "$NATIVE/probes/multicast-probe.cu" -o "$OUT/multicast-probe" -ldl -lpthread ' + BS));
    expect(COMPILERS).toContain('    timeout -k 10 "${MC_TIMEOUT:-900}" "$OUT/multicast-probe" "${MC_SECS:-6}" "${MC_ROUNDS:-3}" "${MC_PACE:-3240}" 2>&1 |');
    expect(COMPILERS).toContain('    node "$NATIVE/probes/multicast-read.js" "$OUT/multicast.log" 2>&1 | while IFS= read -r l; do say "multicast $l"; done');
    // The stage's defaults are the probe's own.
    expect(PROBE).toContain('  const double secs = argc > 1 ? atof(argv[1]) : 6.0;');
    expect(PROBE).toContain('  const int rounds = argc > 2 ? atoi(argv[2]) : 3;');
    expect(PROBE).toContain('  p.pace = argc > 3 ? (uint32_t)atoi(argv[3]) : 3240u;');
  });
});
