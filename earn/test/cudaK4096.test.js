'use strict';

// k 4096 on NVIDIA (round 4 items 8 and 9), step 0: the harness only, nothing the app runs.
// - race.cu takes a k for each build (name=lib.so:K) and gives it its own profile and DAF,
//   so every rate stays MACs a second and builds at different k race fairly.
// - The CUDA host lets the TMA tall fold (sm_120) past 16 chunks only in a bench build with
//   PEARL_TALL_PREGUARD=0, whose fold reads out with the wrap. Shipped builds have 16 chunks
//   and PREGUARD 1, so for them the gate refuses exactly what it refused before.
// No CUDA compiler runs here, so both rules are read out of the source and run in JS.

const fs = require('fs');
const path = require('path');

const NATIVE = path.join(__dirname, '..', 'native');
const ROOT = path.join(NATIVE, '..', '..');
const NL = String.fromCharCode(10);
// Line endings normalised: a Windows checkout has CRLF.
const rawRead = (file) => fs.readFileSync(file, 'utf8').split(String.fromCharCode(13) + NL).join(NL);
const read = (...f) => rawRead(path.join(NATIVE, ...f));
const RACE = read('probes', 'race.cu');
const HOST = read('src', 'pearl_host.cu');
const CONFIG = read('src', 'pearl_config.h');
const KERNEL = read('src', 'pearl_kernel.cu');
const FOLD = read('src', 'pearl_fold_tall.cuh');
const SHIPPED = [
  '.github/workflows/native-core.yml',
  '.github/workflows/miner-build.yml',
  'earn/native/build-local.ps1',
];

const lines = (...l) => l.join(NL);
function slice(text, from, to) {
  const at = text.indexOf(from);
  expect([from, at >= 0]).toEqual([from, true]);
  return text.slice(at, text.indexOf(to, at) + to.length);
}
function grab(text, re) {
  const m = re.exec(text);
  if (!m) throw new Error('not found: ' + re);
  return m[1];
}
// A C condition of these sources as JS: no u or ul suffixes, and ctx-> as ctx.
const toJs = (c) => c.replace(/\b(\d+)ul?\b/g, '$1').replace(/ctx->/g, 'ctx.');
const range = (n, keep) => Array.from({ length: n }, (_, i) => i + 1).filter(keep);

describe('race.cu: each build at its own k', () => {
  const parse = slice(RACE, 'static void parse_build(const char *arg, Build &b) {', NL + '}' + NL);

  it('reads name=lib.so:K, and without :K keeps the mainnet profile as it is', () => {
    expect(RACE).toContain(lines(
      '  for (int i = 4; i < argc; i++) {',
      '    Build b;',
      '    parse_build(argv[i], b);',
      '    builds.push_back(b);',
      '  }'));
    expect(parse).toContain(lines(
      '  b.name.assign(arg, eq - arg);',
      '  b.path = eq + 1;',
      '  b.prof = PEARL_MAINNET_PROFILE;',
      "  const size_t colon = b.path.rfind(':');",
      '  if (colon != std::string::npos) {'));
    expect(parse).toContain(lines(
      '    b.prof.k = (uint32_t)k;',
      '    b.path.erase(colon);',
      '  }'));
    expect(parse).toContain('  b.daf = PEARL_DAF(b.prof);');
    // k is the only field of the profile a build changes, so k 2048 is the mainnet profile.
    expect(parse.match(/b\.prof\.\w+ =/g)).toEqual(['b.prof.k =']);
    expect(CONFIG).toContain('static const PearlProfile PEARL_MAINNET_PROFILE = {2048u, 128u, 0u,');
  });

  it('takes as k exactly the powers of two from 16 x rank (2048), and nothing but digits', () => {
    const cond = grab(parse, /if \((k < 16ul \* b\.prof\.rank \|\| \(k & \(k - 1\)\))\)\n/);
    const refuses = new Function('k', 'b', 'return !!(' + toJs(cond) + ');');
    const b = { prof: { rank: 128 } };
    expect(range(70000, (k) => !refuses(k, b))).toEqual([2048, 4096, 8192, 16384, 32768, 65536]);
    expect(refuses(0, b)).toBe(true);
    // "", "abc", "-4096" and "4k" are not all digits, so k is 0 and refused.
    expect(parse).toContain('const bool digits = !ks.empty() && ks.size() <= 9 && '
      + 'ks.find_first_not_of("0123456789") == std::string::npos;');
    expect(parse).toContain('const unsigned long k = digits ? strtoul(ks.c_str(), nullptr, 10) : 0;');
    expect(parse).toContain('fail(b.name + ": k must be a power of two of at least " + std::to_string(16 * b.prof.rank) + ", not " + ks);');
  });

  it('creates each build with its own profile and scores it with its own DAF', () => {
    expect(RACE).toContain('    b.ctx = b.create(&b.prof, err, sizeof err);');
    expect(RACE).toContain('  leg.th = (double)workAtEnd * b.daf / leg.secs / 1e12;');
    expect(RACE).not.toContain('b.create(&prof,');
    expect(RACE).not.toMatch(/const double daf\b/);
    // DAF is 256 x k, so the rate is MACs a second at any k.
    expect(CONFIG).toContain('#define PEARL_DAF(profile) ((double)(PEARL_ROWS_COUNT * PEARL_COLS_COUNT) * (double)(profile).k)');
    // A salt's regions do not depend on k, so every build shares them.
    expect(RACE).toContain(lines(
      '  const PearlProfile prof = PEARL_MAINNET_PROFILE;',
      '  const uint64_t span = (uint64_t)(prof.m / PEARL_ROWS_COUNT) * (prof.n / PEARL_COLS_COUNT);'));
  });

  it('ends the AUDIT and RACE lines with the k, after every field they had', () => {
    expect(RACE).toContain('printf("AUDIT %s fold=[%s] %s k=%u%s\\n", b.name.c_str(), b.foldName(b.ctx), a, b.prof.k,');
    expect(RACE).toContain('printf("RACE %s reference rounds=%zu th=%.3f mhz=%.0f w=%.1f mem=%.1f k=%u\\n", b.name.c_str(),');
    expect(RACE).toContain(lines(
      '      printf("RACE %s vs %s rounds=%zu d=%+.3f%% sd=%.3f%% ci95=%.3f%% ahead=%zu/%zu th=%.3f mhz=%.0f "',
      '             "w=%.1f mem=%.1f k=%u\\n",'));
    expect(RACE).toContain('ratio.size(), median(th), median(mhz), median(w), median(mem), b.prof.k);');
  });
});

describe('the CUDA host: the tall fold past 16 chunks', () => {
  const GATE = lines(
    '  if (chunks > PEARL_JACKPOT_BUCKETS',
    '      && !(ctx->foldTall && (!ctx->foldTma || PEARL_TALL_PREGUARD == 0)',
    '           && PEARL_FOLD_CHUNKS > PEARL_JACKPOT_BUCKETS && chunks % PEARL_JACKPOT_BUCKETS == 0u)) {');
  // The gate before this change, as it was in pearl_host.cu.
  const BEFORE = 'chunks > PEARL_JACKPOT_BUCKETS && !(ctx->foldTall && !ctx->foldTma '
    + '&& PEARL_FOLD_CHUNKS > PEARL_JACKPOT_BUCKETS && chunks % PEARL_JACKPOT_BUCKETS == 0u)';
  const rule = (cond) => new Function('ctx', 'chunks', 'PEARL_FOLD_CHUNKS', 'PEARL_TALL_PREGUARD',
    'const PEARL_JACKPOT_BUCKETS = 16; return !!(' + toJs(cond) + ');');
  const now = rule(GATE.replace(/^\s*if \(/, '').replace(/\) \{$/, '').replace(/\s+/g, ' '));
  const before = rule(BEFORE);
  const FOLDS = {
    cpAsync: { foldTall: true, foldTma: false },   // RTX 30 and 40
    tma: { foldTall: true, foldTma: true },        // RTX 50
    other: { foldTall: false, foldTma: false },    // RTX 20's B-direct fold, and the rest
  };
  const accepted = (fold, foldChunks, preguard) =>
    range(80, (c) => !now(FOLDS[fold], c, foldChunks, preguard));

  it('is the submit refusal, once, ahead of its error', () => {
    expect(HOST.split(GATE).length).toBe(2);
    expect(slice(HOST, 'extern "C" bool pearl_host_submit(', NL + '}' + NL)).toContain(lines(
      GATE,
      '    if (err && err_len)',
      '      snprintf(err, err_len, "%u chunks exceed %u transcript buckets", chunks,'));
  });

  it('refuses exactly what it refused before in any build with PREGUARD on, shipped ones (16 chunks) included', () => {
    for (const fold of Object.keys(FOLDS))
      for (const foldChunks of [16, 32, 64])
        for (let c = 1; c <= 80; c++)
          expect([fold, foldChunks, c, now(FOLDS[fold], c, foldChunks, 1)])
            .toEqual([fold, foldChunks, c, before(FOLDS[fold], c, foldChunks)]);
    // What ships: k 2048, 16 chunks, and no fold past 16.
    expect(CONFIG).toContain(lines(
      '#ifndef PEARL_FOLD_K',
      '#define PEARL_FOLD_K 2048u',
      '#endif',
      '#define PEARL_FOLD_CHUNKS (PEARL_FOLD_K / PEARL_FOLD_RANK)'));
    expect(CONFIG).toContain('#define PEARL_FOLD_RANK 128u');
    for (const fold of Object.keys(FOLDS)) expect(accepted(fold, 16, 1)).toEqual(range(16, () => true));
  });

  it('lets the TMA fold past 16 chunks only in a PREGUARD 0 build past k 2048, at a multiple of 16', () => {
    const upTo16 = range(16, () => true);
    // This gate takes any multiple of 16 in such a build; the k check ahead of it holds
    // the build to its own k (32 chunks in a k 4096 build).
    const past16 = [...upTo16, 32, 48, 64, 80];
    expect(slice(HOST, 'extern "C" bool pearl_host_submit(', GATE)).toContain(lines(
      '  if (rank != PEARL_FOLD_RANK || k != PEARL_FOLD_K) {',
      '    if (err && err_len)',
      '      snprintf(err, err_len, "fold kernel is built for rank %u, k %u (got rank %u, k %u)",'));
    expect(accepted('tma', 32, 0)).toEqual(past16);
    expect(accepted('tma', 64, 0)).toEqual(past16);
    expect(accepted('tma', 32, 1)).toEqual(upTo16);
    // A PREGUARD 0 build at k 2048 changes nothing.
    for (const fold of Object.keys(FOLDS)) expect(accepted(fold, 16, 0)).toEqual(upTo16);
    // The cp.async fold past k 2048 either way, as before, and never any other fold.
    expect(accepted('cpAsync', 32, 1)).toEqual(past16);
    expect(accepted('cpAsync', 32, 0)).toEqual(past16);
    expect(accepted('other', 32, 0)).toEqual(upTo16);
  });

  it('reads PEARL_TALL_PREGUARD, which the host sees, and the fold agrees', () => {
    expect(HOST).toContain('#include "pearl_config.h"');
    // Defined inside nothing but the header's include guard: 1 by default, and 0 by default
    // in a build past k 2048, whose readout must wrap.
    const at = CONFIG.indexOf(lines('#ifndef PEARL_TALL_PREGUARD', '#if PEARL_FOLD_K > 2048',
      '#define PEARL_TALL_PREGUARD 0', '#else', '#define PEARL_TALL_PREGUARD 1', '#endif', '#endif'));
    expect(at).toBeGreaterThan(0);
    const open = [];
    for (const l of CONFIG.slice(0, at).split(NL)) {
      if (/^#\s*if/.test(l)) open.push(l);
      else if (/^#\s*endif/.test(l)) open.pop();
    }
    expect(open).toEqual(['#ifndef PEARL_CONFIG_H']);
    // The fold's own switch, PEARL_TALL_PREGUARD_ON, is device-only (it needs __CUDA_ARCH__
    // through PEARL_TALL_RED_ON), so the host reads the switch it is made from.
    expect(KERNEL).toContain(lines(
      '#if defined(__CUDA_ARCH__) && PEARL_TALL_BODY_ARCH(__CUDA_ARCH__) && PEARL_TALL_RED_READOUT',
      '#define PEARL_TALL_RED_ON 1'));
    expect(KERNEL).toContain('#if PEARL_TALL_RED_ON && PEARL_TALL_TMA_BODY && PEARL_TALL_TMA_ROLES && PEARL_TALL_PREGUARD');
    // Past 16 chunks the fold compiles only with the shared-XOR readout and PREGUARD off.
    expect(FOLD).toContain(lines(
      '  static_assert((chunks <= PEARL_JACKPOT_BUCKETS',
      '                 || (PEARL_TALL_RED_ON && !PEARL_TALL_PREGUARD_ON',
      '                     && chunks % PEARL_JACKPOT_BUCKETS == 0u))'));
  });

  it('no build that ships sets PEARL_TALL_PREGUARD or PEARL_FOLD_K', () => {
    for (const file of SHIPPED) {
      const text = rawRead(path.join(ROOT, file));
      expect(text.length).toBeGreaterThan(0);
      for (const name of ['PEARL_TALL_PREGUARD', 'PEARL_FOLD_K'])
        expect([file, name, text.includes(name)]).toEqual([file, name, false]);
    }
  });
});

// Long k (PR #290 research item 21): builds up to k 65536, the largest the spec allows at rank 128.
describe('the tall fold at long k', () => {
  // The bytes an operand takes at the mainnet m and n.
  const M = 131072, N = 262144;

  it('takes PEARL_FOLD_K only as a power of two from 2048 to 65536', () => {
    const rule = grab(CONFIG, /\n#if (PEARL_FOLD_K < 2048 \|\| PEARL_FOLD_K > 65536 \|\| \(PEARL_FOLD_K & \(PEARL_FOLD_K - 1\)\))\n#error/);
    // eslint-disable-next-line no-new-func
    const refused = new Function('PEARL_FOLD_K', 'return ' + rule + ';');
    const ok = [];
    for (let k = 1024; k <= 131072; k += 1024) if (!refused(k)) ok.push(k);
    expect(ok).toEqual([2048, 4096, 8192, 16384, 32768, 65536]);
  });

  it('widens the cp.async offsets exactly where an operand passes 4 GB', () => {
    expect(CONFIG).toContain(lines(
      '#if PEARL_FOLD_K > 16384',
      '#define PEARL_TALL_OFF uint64_t',
      '#else',
      '#define PEARL_TALL_OFF uint32_t',
      '#endif'));
    for (const k of [2048, 4096, 8192, 16384, 32768, 65536]) {
      const wide = k > 16384;
      // B' is the larger; at k 16384 its last byte is still a 32-bit offset.
      expect([k, N * k - 1 > 0xFFFFFFFF]).toEqual([k, wide]);
      // A' (with its padding rows) stays under 4 GB to k 16384 too.
      if (!wide) expect(M * k).toBeLessThan(2 ** 32);
    }
    expect(FOLD).toContain('  auto tile_srcs = [&](uint32_t v, PEARL_TALL_OFF &bsrc_, PEARL_TALL_OFF &asrc_) {');
    expect(FOLD).toContain('  PEARL_TALL_OFF bsrc0, asrc0;');
    expect(FOLD).toContain('    asrc_ = (PEARL_TALL_OFF)(rbg_ * BM) * k + (arow * SK + aq * 16u);');
  });

  it('has the host refuse over 4 GB only where the offsets are 32-bit, and never on the TMA fold', () => {
    expect(HOST).toContain(lines(
      '  if (ctx->foldTall && !ctx->foldTma && sizeof(PEARL_TALL_OFF) < 8',
      "      && ((uint64_t)PEARL_TALL_A_ROWS(ctx->profile.m) * k > (uint64_t)0x100000000ull",
      '          || (uint64_t)ctx->profile.n * k > (uint64_t)0x100000000ull)) {'));
  });

  it('wraps the TMA fold\'s DRAM ablation every 16 chunks, from the first two row and column groups', () => {
    expect(FOLD).toContain(lines(
      '    rbg_ &= 1u;',
      '    cbg_ &= 1u;',
      '    bcol_ = cbg_ * PEARL_TALL_COL_OFFSETS * PEARL_COLS_COUNT;',
      '#else',
      '    bcol_ = (col_off + cbg_ * PEARL_TALL_COL_OFFSETS) * PEARL_COLS_COUNT;',
      '#endif'));
    // Comment lines dropped.
    const code = FOLD.split(NL).filter((l) => !/^\s*\/\//.test(l)).join(NL);
    expect(code).toContain(lines(
      '#ifdef PEARL_ABLATE_DRAM',
      '      const uint32_t cn = (last ? 0u : chunk + 1u) & 15u;',
      '#else',
      '      const uint32_t cn = last ? 0u : chunk + 1u;',
      '#endif'));
  });
});
