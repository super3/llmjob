'use strict';

const fs = require('fs');
const path = require('path');

// GTX 16 cards (sm_75 without tensor cores) on Pascal's dp4a fold: PEARL_DP4A_TURING,
// on by default. The switch builds pearl_tile_fold_dp4a's body into the sm_75 binary too,
// and the host runs it on an sm_75 card with fewer than 30 SMs. Nothing here runs device
// code: the checks read the sources, and run the host's card rule on the CPU.
// - where each fold has a body, with and without the switch, and that the RTX 20 fold
//   keeps its body in the sm_75 pass when the dp4a fold gets one;
// - the host's card rule: every GTX 16 and Quadro T card takes it, no RTX 20 card or T4;
// - resolve_fold picks it before any tensor fold is set up, and only from a body;
// - a Turing card asks for the 64 KB shared part, so two blocks fit an SM;
// - the names and log lines say which cards it runs on, as before on Pascal;
// - CI holds the shipped sm_75 kernels, all but the dp4a fold, to a build with the switch
//   off, and fails a build whose sm_75 dp4a fold has no body;
// - pool.sh can build it.

const NL = String.fromCharCode(10);
const CR = String.fromCharCode(13);
const NATIVE = path.join(__dirname, '..', 'native');
const ROOT = path.join(NATIVE, '..', '..');
// Line endings normalised: a Windows checkout has CRLF.
const rawRead = (file) => fs.readFileSync(file, 'utf8').split(CR + NL).join(NL);
const CONFIG = rawRead(path.join(NATIVE, 'src', 'pearl_config.h'));
const HOST = rawRead(path.join(NATIVE, 'src', 'pearl_host.cu'));
const KERNEL = rawRead(path.join(NATIVE, 'src', 'pearl_kernel.cu'));
const FOLD = rawRead(path.join(NATIVE, 'src', 'pearl_fold_dp4a.h'));
const CI = rawRead(path.join(ROOT, '.github', 'workflows', 'native-core.yml'));
const POOL = rawRead(path.join(NATIVE, 'probes', 'turing', 'pool.sh'));
const NVTEST = rawRead(path.join(ROOT, 'tuning', 'salad', 'nvtest.sh'));

const LINES = CONFIG.split(NL).map((l) => l.trim());

// The text from the line holding `from` up to and including the line holding `to`.
function slice(text, from, to) {
  const at = text.indexOf(from);
  expect([from, at >= 0]).toEqual([from, true]);
  const end = text.indexOf(to, at);
  expect([to, end >= at]).toEqual([to, true]);
  const eol = text.indexOf(NL, end);
  return text.slice(text.lastIndexOf(NL, at) + 1, eol < 0 ? text.length : eol);
}

// Each of `parts` in `text`, in this order.
function inOrder(text, parts) {
  let at = 0;
  for (const p of parts) {
    const i = text.indexOf(p, at);
    expect([p, i >= 0]).toEqual([p, true]);
    at = i + p.length;
  }
}

// A preprocessor condition of the header's, run for one pass: `arch` is __CUDA_ARCH__
// (null in the host pass), `vals` the macros it reads. PEARL_DP4A_ARCH is expanded from
// its own #define.
const ARCH_MACRO = /^#define PEARL_DP4A_ARCH\(a\) (.*)$/m.exec(CONFIG)[1];
function condition(cond, arch, vals) {
  let js = cond
    .replace(/PEARL_DP4A_ARCH\(__CUDA_ARCH__\)/g, '(' + ARCH_MACRO.replace(/\(a\)/g, '(__CUDA_ARCH__)') + ')')
    .replace(/defined\(__CUDA_ARCH__\)/g, String(arch !== null));
  js = js.replace(/__CUDA_ARCH__/g, String(arch === null ? 0 : arch));
  for (const [k, v] of Object.entries(vals)) js = js.replace(new RegExp('\\b' + k + '\\b', 'g'), String(v));
  expect([cond, /[A-Za-z_]{3,}/.test(js.replace(/\b(true|false)\b/g, ''))]).toEqual([cond, false]);
  return Number(new Function('return (' + js + ');')()) ? 1 : 0;
}

const ONLY_IF = '#if defined(__CUDA_ARCH__) && PEARL_DP4A_ARCH(__CUDA_ARCH__)';
const BODY_IF = '#if defined(__CUDA_ARCH__) && (PEARL_DP4A_ONLY || (PEARL_DP4A_TURING && __CUDA_ARCH__ == 750))';
const ARCHS = [null, 600, 610, 620, 700, 750, 800, 860, 890, 900, 1000, 1200];

// Every sm_75 card, its SM count (cudaDevAttrMultiProcessorCount) and whether it has tensor
// cores. TU116 and TU117 have none; the TU106 GTX 1650 has them fused off.
const CARDS = [
  ['GTX 1650 (TU117)', 14, false], ['GTX 1650 (TU106)', 14, false], ['GTX 1650 Super', 20, false],
  ['GTX 1660', 22, false], ['GTX 1660 Super', 22, false], ['GTX 1660 Ti', 24, false],
  ['GTX 1650 Ti laptop', 16, false], ['CMP 30HX', 22, false], ['MX450', 14, false], ['MX550', 16, false],
  ['Quadro T400', 6, false], ['T600', 10, false], ['Quadro T1000', 14, false], ['T1200', 16, false],
  ['Quadro T2000', 16, false],
  ['RTX 2060', 30, true], ['RTX 2060 laptop', 30, true], ['Quadro RTX 3000 laptop', 30, true],
  ['RTX 2060 12 GB', 34, true], ['RTX 2060 Super', 34, true], ['RTX 2070', 36, true],
  ['Quadro RTX 4000', 36, true], ['RTX 2070 Super', 40, true], ['Tesla T4', 40, true], ['RTX 2080', 46, true],
  ['RTX 2080 Super', 48, true], ['Quadro RTX 5000', 48, true], ['RTX 2080 Ti', 68, true],
  ['Titan RTX', 72, true], ['Quadro RTX 8000', 72, true],
];

// pearl_dp4a_turing_for, read out of the host and run with the switch's values.
function turingFor(sw) {
  const head = 'bool pearl_dp4a_turing_for(int sms, int force) {' + NL;
  const at = HOST.indexOf(head);
  expect(at).toBeGreaterThan(0);
  const body = HOST.slice(at + head.length, HOST.indexOf(NL + '}' + NL, at));
  expect(body.trim().split(NL).map((l) => l.trim())).toEqual([
    'if (!PEARL_DP4A_TURING) return false;',
    'if (force == 0 || force == 1) return force == 1;',
    'return sms > 0 && sms < PEARL_DP4A_TURING_SMS;']);
  const f = new Function('sms', 'force', 'PEARL_DP4A_TURING', 'PEARL_DP4A_TURING_SMS', body);
  const sms = Number(/^#define PEARL_DP4A_TURING_SMS (\d+)$/m.exec(CONFIG)[1]);
  return (cardSms, force = -1) => f(cardSms, force, sw, sms);
}

describe('the switch and where each fold has a body (pearl_config.h)', () => {
  test('is on by default, 0 or 1, with 30 SMs as the line', () => {
    const at = LINES.indexOf('#ifndef PEARL_DP4A_TURING');
    expect(LINES.slice(at, at + 7)).toEqual([
      '#ifndef PEARL_DP4A_TURING', '#define PEARL_DP4A_TURING 1', '#endif',
      '#if PEARL_DP4A_TURING != 0 && PEARL_DP4A_TURING != 1', '#error "PEARL_DP4A_TURING is 0 or 1"', '#endif',
      '#define PEARL_DP4A_TURING_SMS 30']);
    const refused = (v) => condition('PEARL_DP4A_TURING != 0 && PEARL_DP4A_TURING != 1', null, { PEARL_DP4A_TURING: v });
    expect([-1, 0, 1, 2].map(refused)).toEqual([1, 0, 0, 1]);
    // Each macro is defined once a pass: one #if, two branches.
    for (const name of ['PEARL_DP4A_ONLY', 'PEARL_DP4A_BODY']) {
      expect(CONFIG.match(new RegExp('^#define ' + name + ' [01]$', 'gm'))).toEqual(
        ['#define ' + name + ' 1', '#define ' + name + ' 0']);
    }
    expect(slice(CONFIG, ONLY_IF, '#endif').split(NL).map((l) => l.trim())).toEqual(
      [ONLY_IF, '#define PEARL_DP4A_ONLY 1', '#else', '#define PEARL_DP4A_ONLY 0', '#endif']);
    expect(slice(CONFIG, BODY_IF, '#endif').split(NL).map((l) => l.trim())).toEqual(
      [BODY_IF, '#define PEARL_DP4A_BODY 1', '#else', '#define PEARL_DP4A_BODY 0', '#endif']);
    // ONLY is settled before BODY reads it, and the switch before either.
    expect(CONFIG.indexOf(ONLY_IF)).toBeLessThan(CONFIG.indexOf('#ifndef PEARL_DP4A_TURING'));
    expect(CONFIG.indexOf('#ifndef PEARL_DP4A_TURING')).toBeLessThan(CONFIG.indexOf(BODY_IF));
  });

  test('without the switch, every pass is as before; with it, sm_75 alone gains the dp4a body', () => {
    const passes = (sw) => ARCHS.map((arch) => {
      const only = condition(ONLY_IF.slice(4), arch, {});
      const body = condition(BODY_IF.slice(4), arch, { PEARL_DP4A_ONLY: only, PEARL_DP4A_TURING: sw });
      return [arch, only, body];
    });
    // [__CUDA_ARCH__, PEARL_DP4A_ONLY, PEARL_DP4A_BODY]: Pascal only, as PEARL_DP4A_ARCH was.
    expect(passes(0)).toEqual([
      [null, 0, 0], [600, 0, 0], [610, 1, 1], [620, 1, 1], [700, 1, 1], [750, 0, 0], [800, 0, 0], [860, 0, 0],
      [890, 0, 0], [900, 0, 0], [1000, 0, 0], [1200, 0, 0]]);
    expect(passes(1)).toEqual([
      [null, 0, 0], [600, 0, 0], [610, 1, 1], [620, 1, 1], [700, 1, 1], [750, 0, 1], [800, 0, 0], [860, 0, 0],
      [890, 0, 0], [900, 0, 0], [1000, 0, 0], [1200, 0, 0]]);
  });

  // The wmma fold is empty where PEARL_DP4A_ONLY says there is no int8 mma. Keyed on
  // PEARL_DP4A_BODY, the switch would empty the RTX 20 fold in the sm_75 pass.
  test('the RTX 20 fold keeps its body in an sm_75 pass with the dp4a body', () => {
    const wmma = slice(KERNEL, 'void pearl_tile_fold_wmma(', '#endif  // PEARL_DP4A_ONLY');
    const firstIf = wmma.split(NL).find((l) => l.startsWith('#if'));
    expect(firstIf).toBe('#if PEARL_DP4A_ONLY');
    expect(KERNEL.match(/#if PEARL_DP4A_ONLY\b/g)).toHaveLength(1);
    // The dp4a fold's own body, and its inner loop's __dp4a, follow PEARL_DP4A_BODY.
    expect(KERNEL.match(/^#if PEARL_DP4A_BODY$/gm)).toHaveLength(1);
    expect(KERNEL.indexOf('#if PEARL_DP4A_BODY')).toBeGreaterThan(KERNEL.indexOf('void pearl_tile_fold_dp4a('));
    expect(FOLD).toContain('#if defined(__CUDA_ARCH__) && PEARL_DP4A_BODY' + NL + '  return __dp4a((int)a, (int)b, c);');
    // Nothing else in the sources reads either macro.
    for (const f of ['pearl_host.cu', 'pearl_fold_bd.cuh', 'pearl_fold_tall.cuh', 'pearl_core.cc']) {
      const text = rawRead(path.join(NATIVE, 'src', f));
      expect([f, /PEARL_DP4A_(ONLY|BODY)\b/.test(text)]).toEqual([f, false]);
    }
  });
});

describe('the host\'s card rule (pearl_dp4a_turing_for)', () => {
  test('with the switch, every sm_75 card without tensor cores and none with them', () => {
    const on = turingFor(1);
    for (const [name, sms, tensor] of CARDS) expect([name, on(sms)]).toEqual([name, !tensor]);
    // An SM count the driver did not give keeps the tensor fold.
    expect(on(0)).toBe(false);
    // The line sits between the largest card without and the smallest with.
    const without = CARDS.filter((c) => !c[2]).map((c) => c[1]);
    const withTc = CARDS.filter((c) => c[2]).map((c) => c[1]);
    expect(Math.max(...without)).toBeLessThan(30);
    expect(Math.min(...withTc)).toBe(30);
  });

  test('PEARL_DP4A_TURING_FORCE: 1 every sm_75 card, 0 none; without the switch never', () => {
    const on = turingFor(1);
    const off = turingFor(0);
    for (const [name, sms] of CARDS) {
      expect([name, on(sms, 1), on(sms, 0), off(sms), off(sms, 1), off(sms, 0)])
        .toEqual([name, true, false, false, false, false]);
    }
    expect(on(0, 1)).toBe(true);
  });
});

describe('resolve_fold and what follows a card it takes', () => {
  const resolve = slice(HOST, 'void resolve_fold(Ctx *ctx) {', '// Which binary loaded, read off the default kernel.');

  // Only an sm_75 binary whose dp4a kernel has a body (the empty one uses a few registers),
  // the SM count from the driver, the run-time override as the launch sweep reads its own
  // (0 or 1, else not set), then the same test and return as Pascal's, before any tensor
  // fold is looked at.
  test('takes an sm_75 card from the body, the SM count and the override, before the tensor folds', () => {
    inOrder(resolve, [
      'cudaFuncGetAttributes(&fd, reinterpret_cast<const void *>(pearl_tile_fold_dp4a))',
      'bool turing = false;',
      'if (PEARL_DP4A_TURING && fd.binaryVersion == 75 && fd.numRegs >= 64) {',
      'if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, ctx->device)',
      'pearl_bd_sweep_env("PEARL_DP4A_TURING_FORCE", &v) && v <= 1u ? (int)v : -1;',
      'turing = pearl_dp4a_turing_for(sms, force);',
      'if ((PEARL_DP4A_ARCH(fd.binaryVersion * 10) || turing)',
      '&& (uint32_t)fd.maxThreadsPerBlock == PEARL_DP4A_THREADS) {',
      'ctx->foldDp4a = true;', 'ctx->foldKnown = true;', 'return;']);
    // Nothing about the tensor folds is set before it returns.
    const head = resolve.slice(0, resolve.indexOf('ctx->foldDp4a = true;'));
    expect(head).not.toMatch(/ctx->fold(Turing|BDirect|Tall|Wide|Persistent)\b/);
    // The override is read with the launch sweep's parser, declared before resolve_fold.
    expect(HOST.indexOf('bool pearl_bd_sweep_env(const char *name, uint32_t *out);'))
      .toBeLessThan(HOST.indexOf('void resolve_fold(Ctx *ctx) {'));
    expect(HOST.indexOf('bool pearl_dp4a_turing_for(int sms, int force) {'))
      .toBeLessThan(HOST.indexOf('void resolve_fold(Ctx *ctx) {'));
    // The 64-register line is CI's for the sm_61 body too.
    expect(CI).toContain('[ "$regs" -ge 64 ] 2>/dev/null; then');
  });

  // A Turing card asks for the larger shared part, so two blocks (42 KB) fit an SM. Pascal's
  // shared is fixed, and its setup is as before.
  test('a Turing card asks for 64 KB of shared, and the log says how many blocks an SM holds', () => {
    const setup = slice(HOST, '    const uint32_t cb = pearl_dp4a_col_batch(ctx->colBatch, ctx->colsValid);',
      'ctx->device, ctx->ccMajor, ctx->ccMinor, pearl_dp4a_arch(ctx), ctx->colBatch, perSm);');
    inOrder(setup, [
      'if (ctx->ccMajor == 7',
      '&& cudaFuncSetAttribute(reinterpret_cast<const void *>(pearl_tile_fold_dp4a),',
      'cudaFuncAttributePreferredSharedMemoryCarveout,',
      '(int)cudaSharedmemCarveoutMaxShared) != cudaSuccess)',
      '(void)cudaGetLastError();',
      'const cudaError_t oe = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSm,',
      'reinterpret_cast<const void *>(pearl_tile_fold_dp4a), (int)PEARL_DP4A_THREADS, 0);',
      'if (oe != cudaSuccess) {']);
    // 21 KB of static shared a block: two fit Turing's 64 KB part, not its 32 KB one.
    const smem = 2 * 2 * 8 * 132 * 4 + 64 * 17 * 4;
    expect([smem, 2 * smem > 32 * 1024, 2 * smem <= 64 * 1024]).toEqual([21248, true, true]);
  });

  // A dp4a card gets the dp4a batch width and launch, whichever cards it is on.
  test('the batch width, the draw and the launch are the dp4a fold\'s, as on Pascal', () => {
    expect(HOST).toContain('  if (ctx->foldDp4a) {' + NL
      + '    const uint32_t cb = pearl_dp4a_col_batch(ctx->colBatch, ctx->colsValid);');
    expect(HOST).toContain('  if (ctx->foldDp4a && kPow2 && k >= PEARL_DP4A_STAGE_K) {');
    expect(HOST).toContain('  if (ctx->foldDp4a)' + NL + '    return pearl_submit_dp4a(');
  });

  // Pascal's lines read as they did; a Turing card's say Turing.
  test('the log line, the error tag and the fold name say which cards it runs on', () => {
    expect(HOST).toContain(
      'const char *pearl_dp4a_arch(const Ctx *ctx) { return ctx->ccMajor == 7 ? "Turing" : "Pascal"; }');
    const arch = (major) => (major === 7 ? 'Turing' : 'Pascal');
    expect(HOST).toContain('"[pearl] GPU %d: compute %d.%d, %s dp4a fold, %u column offsets a batch, %d "' + NL
      + '            "blocks an SM\\n",' + NL
      + '            ctx->device, ctx->ccMajor, ctx->ccMinor, pearl_dp4a_arch(ctx), ctx->colBatch, perSm);');
    expect(HOST).toContain('  if (!ctx->foldDp4a) return "";' + NL
      + '  return ctx->ccMajor == 7 ? " (Turing dp4a fold)" : " (Pascal dp4a fold)";');
    expect(HOST).toContain('"%s dp4a fold launch failed on compute %d.%d (%u blocks of %u threads): %s",' + NL
      + '               pearl_dp4a_arch(ctx), ctx->ccMajor, ctx->ccMinor, tiles,');
    expect(HOST).toContain('"dp4a 128x128, 256 threads of 8x8, a block a tile, band %u, col_batch %u (%s)",' + NL
      + '             (unsigned)PEARL_DP4A_BAND, ctx->colBatch, pearl_dp4a_arch(ctx));');
    // As before on Pascal, and within the name's buffer on both.
    expect(`dp4a 128x128, 256 threads of 8x8, a block a tile, band 8, col_batch 256 (${arch(6)})`)
      .toBe('dp4a 128x128, 256 threads of 8x8, a block a tile, band 8, col_batch 256 (Pascal)');
    const size = Number(/char foldName\[(\d+)\] = \{0\};/.exec(HOST)[1]);
    for (const major of [6, 7]) {
      const name = `dp4a 128x128, 256 threads of 8x8, a block a tile, band 4294967295, col_batch 4294967295 (${arch(major)})`;
      expect(name.length).toBeLessThan(size);
    }
  });
});

describe('builds', () => {
  // The shipped compile has the switch on (the default). CI compiles sm_75 again with it
  // off and holds every other sm_75 kernel to that build's machine code, so the RTX 20
  // cards and the T4 run what they ran. Its spill check fails a build whose sm_75 dp4a
  // fold has no body, as it does for sm_61.
  test('CI compares the shipped sm_75 kernels with the switch off', () => {
    const step = slice(CI, '- name: Check that the GTX 16 dp4a fold changes no other sm_75 kernel', '          EOF');
    expect(step).toContain("if: runner.os == 'Linux' && matrix.variant == ''");
    // Blocking: the switch ships, so a change it makes to the RTX 20 folds fails the job.
    expect(step).not.toContain('continue-on-error');
    expect(step.match(/-gencode arch=compute_75,code=sm_75 \\/g)).toHaveLength(1);
    expect(step.match(/-DPEARL_DP4A_TURING=0 \\/g)).toHaveLength(1);
    expect(step).toContain('-c src/pearl_kernel.cu -o $G/pearl_kernel.o');
    expect(step).toContain('(cd $G/shipped && cuobjdump -xelf all ../../pearl_kernel.o > /dev/null)');
    expect(step).toContain('(cd $G/off && cuobjdump -xelf all ../pearl_kernel.o > /dev/null)');
    expect(step).toContain("python3 - $G/off/*sm_75*.cubin $G/shipped/*sm_75*.cubin <<'EOF'");
    expect(step).toContain('if n == "pearl_tile_fold_dp4a":');
    expect(step).toContain('changed an sm_75 kernel other than the dp4a fold');
    expect(step).toContain('sys.exit(bad)');
    // After the shipped compile it reads, which takes the default.
    expect(CI.indexOf('- name: Compile the CUDA kernels')).toBeLessThan(CI.indexOf(step));
    const shipped = slice(CI, '- name: Compile the CUDA kernels', '-c src/pearl_host.cu -o cuda-build/pearl_host.o');
    expect(shipped).not.toContain('PEARL_DP4A_TURING');
    // That step is the only place a workflow or build script sets it.
    expect(CI.replace(step, '')).not.toContain('PEARL_DP4A_TURING=');
    for (const f of ['.github/workflows/miner-build.yml', 'earn/native/build-local.ps1', 'earn/native/binding.gyp']) {
      expect([f, fs.readFileSync(path.join(ROOT, f), 'utf8').includes('PEARL_DP4A_TURING')]).toEqual([f, false]);
    }
    // The spill check: an sm_75 dp4a fold under 64 registers is the empty kernel, and fails
    // any build that compiles sm_75.
    const spill = slice(CI, '- name: Check the fold kernels for register spills', '          exit $bad');
    inOrder(spill, [
      'dp4a_body_75=0',
      'if [ "$name" = pearl_tile_fold_dp4a ] && [ "$arch" = sm_75 ] \\',
      '&& [ "$regs" -ge 64 ] 2>/dev/null; then',
      'dp4a_body_75=1',
      '*"code=sm_75 "*)',
      'if [ "$dp4a_body_75" != 1 ]; then',
      '::error::pearl_tile_fold_dp4a on sm_75 has no body',
      'bad=1']);
  });

  // pool.sh builds the core it mines with from POOL_FLAGS, as nvtest.sh documents.
  test('pool.sh can mine with the switch', () => {
    expect(POOL).toContain('STAGES="setup build" VARIANTS_SPEC="base|${POOL_FLAGS:-}" SRC="$SRC" OUT="$OUT/cs" \\');
    expect(POOL).toContain('worker $WORKER flags [${POOL_FLAGS:-}]"');
    expect(NVTEST).toContain('#   POOL_FLAGS      nvcc flags for the core pool.sh builds');
  });
});
