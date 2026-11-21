'use strict';

// PEARL_AMD_W64K (earn/native/amd/src/pearl_amd_fold.h): the RX 9070 XT's fold in wave64 with a 64-byte slab. It is
// on only in a gfx1200 or gfx1201 device pass that is compiled wave64, and build-amd.sh compiles those targets wave64 on their
// own (arch_flags, hipc). The code object then says PEARL_AMD_CAP_SLAB64, and the host and the fold test tile that
// card's operands by 64 bytes. Every other pass, the Windows core, and the stamp and ablation builds keep the wave32
// fold and the 32-byte slab. No HIP compiler runs here, so the C++ is read as text and its conditions run as JS; the
// shell functions run under bash where there is one, against stand-in compilers.

const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');

const NATIVE = path.join(__dirname, '..', 'native');
const read = (...f) => fs.readFileSync(path.join(NATIVE, ...f), 'utf8').replace(/\r\n/g, '\n');
const FOLD = read('amd', 'src', 'pearl_amd_fold.h');
const HOST = read('amd', 'src', 'pearl_host_hip.cpp');
const FOLDTEST = read('amd', 'test', 'pearl_amd_foldtest.cpp');
const BUILD = read('amd', 'build-amd.sh');
const CONFIG = read('src', 'pearl_config.h');

const hasBash = spawnSync('bash', ['--version'], { stdio: 'ignore' }).status === 0;
const bashIt = hasBash ? it : it.skip;

function slice(text, from, to) {
  const at = text.indexOf(from);
  if (at < 0) throw new Error('not found: ' + from);
  return text.slice(at, text.indexOf(to, at + from.length) + to.length);
}
function grab(text, re) {
  const m = re.exec(text);
  if (!m) throw new Error('not found: ' + re);
  return m[1];
}
// A shell function of build-amd.sh, from its first line to its closing brace.
const shFn = (name) => slice(BUILD, '\n' + name + '() {\n', '\n}\n').slice(1);
// Code without its comments: what the compiler reads.
const code = (t) => t.split('\n').map((l) => l.replace(/\/\/.*$/, '')).join('\n');
const list = (name) => grab(BUILD, new RegExp('^' + name + '="([^"]*)"$', 'm')).split(' ');

describe('build-amd.sh arch_flags', () => {
  it('is this text', () => {
    expect(shFn('arch_flags')).toBe('arch_flags() {\n'
      + '  case "$1:${PEARL_AMD_CFLAGS:-}" in\n'
      + '    gfx120[01]:*PEARL_AMD_W64K=0* | gfx120[01]:*mwavefrontsize* | gfx120[01]:*PEARL_AMD_STAMP* | gfx120[01]:*PEARL_AMD_ABLATE* | \\\n'
      + '    gfx120[01]:*PEARL_AMD_RDNA4_SET=0* | gfx120[01]:*PEARL_AMD_W12_WR* | gfx120[01]:*PEARL_AMD_W64=*) ;;\n'
      + '    gfx120[01]:*) echo "-mwavefrontsize64 -DPEARL_AMD_WAVE64=1" ;;\n'
      + '  esac\n}\n');
    // And the header stops a wave64 gfx12 pass that would build the wave32 fold, should the two ever disagree.
    expect(FOLD).toContain('#else\n#define PEARL_AMD_W64_ON 0\n'
      + '#if PEARL_AMD_WMMA12 && !PEARL_AMD_EMU && defined(PEARL_AMD_WAVE64) && PEARL_AMD_WAVE64\n'
      + '#error "a wave64 gfx12 pass builds the wave64 fold only: PEARL_AMD_W64K is off here (build-amd.sh\'s arch_flags)"\n'
      + '#endif\n#endif\n');
  });

  bashIt('gives -mwavefrontsize64 to gfx1200 and gfx1201 only, and nothing when the flags turn W64K off or give a wave size', () => {
    const targets = [...new Set([...list('DEFAULT_ARCHS'), ...list('WINDOWS_ARCHS')])];
    expect(targets).toContain('gfx1201');
    expect(targets).toContain('gfx1200');
    expect(targets).toContain('gfx1100');
    const run = (flags) => {
      const env = { PATH: process.env.PATH, TARGETS: targets.join(' ') };
      if (flags !== null) env.PEARL_AMD_CFLAGS = flags;
      const r = spawnSync('bash', ['-c', 'set -euo pipefail\n' + shFn('arch_flags')
        + 'for a in $TARGETS; do echo "$a:$(arch_flags "$a")"; done\n'], { env, encoding: 'utf8' });
      expect([flags, r.status, r.stderr]).toEqual([flags, 0, '']);
      return r.stdout.split('\n').filter(Boolean).filter((l) => !l.endsWith(':'));
    };
    // The release build has no PEARL_AMD_CFLAGS; a tuning build's other flags change nothing.
    for (const flags of [null, '', '-DPEARL_AMD_TILED=0', '-DPEARL_AMD_W64K=1', '-DPEARL_AMD_W64K_GRP=2 -DPEARL_AMD_W64KD=0']) {
      expect([flags, run(flags)]).toEqual([flags, ['gfx1200:-mwavefrontsize64 -DPEARL_AMD_WAVE64=1', 'gfx1201:-mwavefrontsize64 -DPEARL_AMD_WAVE64=1']]);
      // RDNA 3 and the other RDNA 4 families' neighbours get nothing, by name.
      expect(run(flags).filter((l) => /^gfx1100:|^gfx1101:|^gfx1150:/.test(l))).toEqual([]);
    }
    for (const flags of ['-DPEARL_AMD_W64K=0', '-O3 -DPEARL_AMD_W64K=0 -DPEARL_AMD_TILED=0', '-mwavefrontsize32',
      '-mwavefrontsize64', '-DPEARL_AMD_W64=1 -DPEARL_AMD_W64GB=1 -mwavefrontsize64',
      // The builds PEARL_AMD_W64K's default is off in keep the wave32 fold, so they must stay wave32 passes.
      '-DPEARL_AMD_STAMP=1', '-DPEARL_AMD_ABLATE=2', '-O3 -DPEARL_AMD_ABLATE=8 -DPEARL_AMD_STAMP=1',
      '-DPEARL_AMD_RDNA4_SET=0', '-DPEARL_AMD_W12_WR=2', '-DPEARL_AMD_W64=0', '-DPEARL_AMD_W64=1']) {
      expect([flags, run(flags)]).toEqual([flags, []]);
    }
  });

  it('reaches the gpu build through hipc and the check on Linux only, and never winlib', () => {
    const gpu = BUILD.slice(BUILD.indexOf('\n  gpu)\n'), BUILD.indexOf('\n  winlib)\n'));
    const winlib = BUILD.slice(BUILD.indexOf('\n  winlib)\n'), BUILD.indexOf('\n  check)\n'));
    const check = BUILD.slice(BUILD.indexOf('\n  check)\n'));
    // Both objects of the gpu build that hold device code go through hipc; the addon and the links are plain calls.
    expect(gpu.match(/^ *hipc .*$/gm)).toEqual([
      '    hipc "$HERE/src/pearl_host_hip.cpp" "$OUT/gpu/pearl_host_hip.o" $flags $targets $inc',
      '    hipc "$HERE/test/pearl_amd_foldtest.cpp" "$OUT/gpu/pearl_amd_foldtest.o" $flags $inc',
    ]);
    expect(gpu).not.toContain('-x hip');
    // The Windows core is compiled in one call, wave32, with the RDNA 4 set off before any tuning flags.
    expect(winlib).not.toMatch(/hipc|arch_flags|wavefrontsize|offload-bundler/);
    expect(winlib.match(/"\$cc" -x hip \$\(offload_flags\) --rocm-path="\$rocm" \$flags /g)).toHaveLength(2);
    expect(BUILD).toContain('if windows; then SET_OS="-DPEARL_AMD_RDNA4_SET=0 -DPEARL_AMD_RDNA3_BUFLD=0 -DPEARL_AMD_RDNA3_SET=0"; fi');
    // The check's one use is behind `windows ||`, so on Windows the target gets no flag of its own.
    expect(check.match(/arch_flags/g)).toHaveLength(1);
    expect(check).toContain('${PEARL_AMD_CFLAGS:-} $(windows || arch_flags "$a") $inc --cuda-device-only');
    expect(BUILD.match(/arch_flags "\$a"/g)).toHaveLength(3);   // hipc's two and the check's
  });
});

describe('build-amd.sh hipc', () => {
  let tmp;
  beforeEach(() => { tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'amd-hipc-')); });
  afterEach(() => { fs.rmSync(tmp, { recursive: true, force: true }); });

  // Runs hipc from a folder with stand-ins for clang++ and clang-offload-bundler, which log their arguments.
  function run(archs, flags) {
    fs.mkdirSync(path.join(tmp, 'bin'), { recursive: true });
    for (const tool of ['clang++', 'clang-offload-bundler']) {
      fs.writeFileSync(path.join(tmp, 'bin', tool), '#!/bin/bash\necho "' + tool + ' $*" >> calls.txt\n', { mode: 0o755 });
    }
    fs.rmSync(path.join(tmp, 'calls.txt'), { force: true });
    const env = { PATH: process.env.PATH };
    if (flags !== null) env.PEARL_AMD_CFLAGS = flags;
    const r = spawnSync('bash', ['-c', 'set -euo pipefail\narchs="' + archs + '"\ncc=bin/clang++\nrocm=/opt/rocm\n'
      + shFn('offload_flags') + shFn('arch_flags') + shFn('hipc')
      + 'hipc src/a.cpp out/a.o -O3 ${PEARL_AMD_CFLAGS:-} -Iinc\necho done\n'], { cwd: tmp, env, encoding: 'utf8' });
    expect([archs, flags, r.status, r.stderr, r.stdout]).toEqual([archs, flags, 0, '', 'done\n']);
    return fs.readFileSync(path.join(tmp, 'calls.txt'), 'utf8').split('\n').filter(Boolean);
  }

  bashIt('compiles in one call, as before, when no target has flags of its own', () => {
    expect(run('gfx1100 gfx1030', null)).toEqual([
      'clang++ -x hip --offload-arch=gfx1100 --offload-arch=gfx1030 --rocm-path=/opt/rocm -O3 -Iinc -c src/a.cpp -o out/a.o',
    ]);
    // gfx1200 and gfx1201 too, in a build that turns W64K off or gives the wave size itself.
    expect(run('gfx1200 gfx1201', '-DPEARL_AMD_W64K=0')).toEqual([
      'clang++ -x hip --offload-arch=gfx1200 --offload-arch=gfx1201 --rocm-path=/opt/rocm -O3 -DPEARL_AMD_W64K=0 -Iinc '
      + '-c src/a.cpp -o out/a.o',
    ]);
    expect(run('gfx1201', '-mwavefrontsize64 -DPEARL_AMD_W64=1')).toEqual([
      'clang++ -x hip --offload-arch=gfx1201 --rocm-path=/opt/rocm -O3 -mwavefrontsize64 -DPEARL_AMD_W64=1 -Iinc '
      + '-c src/a.cpp -o out/a.o',
    ]);
    expect(fs.existsSync(path.join(tmp, 'out'))).toBe(false);
    // That call is the one the gpu build made before hipc.
    expect(shFn('hipc')).toContain('  if [ $split = 0 ]; then\n'
      + '    $cc -x hip $(offload_flags) --rocm-path="$rocm" "$@" -c "$src" -o "$obj"\n    return\n  fi\n');
  });

  bashIt('builds each target\'s code object alone when one has flags, bundles them, and gives the host pass the pack', () => {
    const dev = (a, own) => 'clang++ -x hip --offload-arch=' + a + ' --rocm-path=/opt/rocm -O3 -Iinc ' + own
      + '--cuda-device-only --no-gpu-bundle-output -c src/a.cpp -o out/a.o.targets/' + a + '.co';
    expect(run('gfx1100 gfx1201 gfx1200', null)).toEqual([
      // Only gfx1200's and gfx1201's passes are wave64; the others get the flags the one call gave them.
      dev('gfx1100', ''),
      dev('gfx1201', '-mwavefrontsize64 -DPEARL_AMD_WAVE64=1 '),
      dev('gfx1200', '-mwavefrontsize64 -DPEARL_AMD_WAVE64=1 '),
      'clang-offload-bundler -type=o -bundle-align=4096 -targets=host-x86_64-unknown-linux-gnu,'
      + 'hipv4-amdgcn-amd-amdhsa--gfx1100,hipv4-amdgcn-amd-amdhsa--gfx1201,hipv4-amdgcn-amd-amdhsa--gfx1200 '
      + '-input=/dev/null -input=out/a.o.targets/gfx1100.co -input=out/a.o.targets/gfx1201.co '
      + '-input=out/a.o.targets/gfx1200.co -output=out/a.o.targets/fat.hipfb',
      'clang++ -x hip --offload-arch=gfx1100 --offload-arch=gfx1201 --offload-arch=gfx1200 --rocm-path=/opt/rocm -O3 -Iinc '
      + '--cuda-host-only -Xclang -fcuda-include-gpubinary -Xclang out/a.o.targets/fat.hipfb -c src/a.cpp -o out/a.o',
    ]);
    expect(fs.existsSync(path.join(tmp, 'out', 'a.o.targets'))).toBe(true);
    // The bundler is the one beside the compiler.
    expect(shFn('hipc')).toContain('  "$(dirname "$cc")/clang-offload-bundler" -type=o -bundle-align=4096 -targets="$tg" $ins '
      + '-output="$d/fat.hipfb"\n');
  });
});

describe('PEARL_AMD_W64K\'s default', () => {
  const block = slice(FOLD, '#ifndef PEARL_AMD_W64K\n', '\n#endif\n#endif\n');
  const cond = grab(block, /^#ifndef PEARL_AMD_W64K\n#if ((?:.*\\\n)*.*)\n#define PEARL_AMD_W64K 1\n#else\n#define PEARL_AMD_W64K 0\n#endif\n#endif\n$/)
    .replace(/\\\n/g, ' ');
  // The condition as the preprocessor reads it: defined(X) asks whether X is set, and a name that is not set is 0.
  const on = (macros) => Number(Boolean(new Function('D', 'V', 'return ' + cond
    .replace(/defined\((\w+)\)/g, 'D["$1"]')
    .replace(/(^|[^"\w])([A-Za-z_]\w*)\b(?!\[)/g, (m, pre, n) => (n === 'D' ? m : pre + 'V("' + n + '")')) + ';')(
    Object.fromEntries(Object.keys(macros).map((n) => [n, true])), (n) => macros[n] || 0)));
  // A gfx1201 device pass of the Linux core, compiled wave64 or wave32, with nothing set by the build but the wave.
  const W64 = { PEARL_AMD_WMMA12: 1, PEARL_AMD_EMU: 0, __gfx1201__: 1, PEARL_AMD_WAVE64: 1 };
  const W32 = { ...W64, PEARL_AMD_WAVE64: 0 };

  it('is this text, and reads only names that are set before it or tested with defined()', () => {
    expect(block).toBe('#ifndef PEARL_AMD_W64K\n'
      + '#if PEARL_AMD_WMMA12 && (defined(__gfx1201__) || defined(__gfx1200__)) && !PEARL_AMD_EMU && !defined(PEARL_AMD_W64) && \\\n'
      + '    defined(PEARL_AMD_WAVE64) && PEARL_AMD_WAVE64 && \\\n'
      + '    !(defined(PEARL_AMD_RDNA4_SET) && !PEARL_AMD_RDNA4_SET) && !(defined(PEARL_AMD_STAMP) && PEARL_AMD_STAMP) && \\\n'
      + '    !(defined(PEARL_AMD_ABLATE) && PEARL_AMD_ABLATE) && !(defined(PEARL_AMD_W12_WR) && PEARL_AMD_W12_WR != 4)\n'
      + '#define PEARL_AMD_W64K 1\n#else\n#define PEARL_AMD_W64K 0\n#endif\n#endif\n');
    const at = FOLD.indexOf(block);
    expect(code(FOLD.slice(0, at))).not.toMatch(/PEARL_AMD_W64K\b/);
    for (const n of ['PEARL_AMD_WMMA12', 'PEARL_AMD_EMU']) {
      const def = FOLD.indexOf(`#define ${n} `);
      expect([n, def > 0 && def < at]).toEqual([n, true]);
    }
    // No other place turns it on.
    expect(code(FOLD).match(/#define PEARL_AMD_W64K 1/g)).toHaveLength(1);
    expect(code(HOST)).not.toMatch(/#define PEARL_AMD_W64/);
    expect(BUILD).not.toMatch(/-DPEARL_AMD_W64/);
  });

  it('needs __gfx1200__ or __gfx1201__ and a wave64 pass', () => {
    expect(on(W64)).toBe(1);
    expect(on(W32)).toBe(0);
    // A pass that says no wave size, another gfx12 target, the host pass, and HIP-CPU.
    const { PEARL_AMD_WAVE64: w, ...noSize } = W64;
    expect(w).toBe(1);
    expect(on(noSize)).toBe(0);
    // The build says so itself (arch_flags' -DPEARL_AMD_WAVE64=1): ROCm 7.2's clang defines no wave-size macro,
    // and one from a later clang does not turn it on alone.
    expect(on({ ...noSize, __AMDGCN_WAVEFRONT_SIZE: 64 })).toBe(0);
    const { __gfx1201__: g, ...other } = W64;
    expect(g).toBe(1);
    // gfx1200 (RX 9060 XT) too, wave64 only. Its pass also has PEARL_AMD_GB and PEARL_AMD_FUSE on, which the
    // default does not read.
    const gfx1200 = { ...other, __gfx1200__: 1 };
    expect(on(gfx1200)).toBe(1);
    expect(on({ ...gfx1200, PEARL_AMD_WAVE64: 0 })).toBe(0);
    expect(on({ ...gfx1200, PEARL_AMD_GB: 1, PEARL_AMD_FUSE: 1 })).toBe(1);
    expect(block).not.toMatch(/PEARL_AMD_GB\b|PEARL_AMD_FUSE/);
    expect(FOLD).toContain('#if !PEARL_AMD_EMU && (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1200__))\n'
      + '#define PEARL_AMD_FUSE 1\n');
    // No other target: a gfx12 pass that names neither, and RDNA 3.
    expect(on(other)).toBe(0);
    expect(on({ ...other, __gfx1100__: 1, PEARL_AMD_WMMA12: 0 })).toBe(0);
    expect(on({ ...other, __gfx1100__: 1 })).toBe(0);
    expect(on({ ...W64, PEARL_AMD_WMMA12: 0 })).toBe(0);
    expect(on({ ...W64, PEARL_AMD_EMU: 1 })).toBe(0);
  });

  it('stays off for the Windows core, stamp and ablation builds, the 128x256 block, and a build that sets W64 itself', () => {
    // Each also off in a wave64 pass, so the wave size alone never turns it on there.
    for (const [n, v] of [['PEARL_AMD_RDNA4_SET', 0], ['PEARL_AMD_STAMP', 1], ['PEARL_AMD_ABLATE', 2],
      ['PEARL_AMD_ABLATE', 8], ['PEARL_AMD_ABLATE', 64], ['PEARL_AMD_W12_WR', 2], ['PEARL_AMD_W12_WR', 8],
      ['PEARL_AMD_W64', 0], ['PEARL_AMD_W64', 1]]) {
      expect([n, v, on({ ...W64, [n]: v })]).toEqual([n, v, 0]);
      expect([n, v, on({ ...W32, [n]: v })]).toEqual([n, v, 0]);
    }
    // The same flags at the values a release has change nothing.
    for (const [n, v] of [['PEARL_AMD_RDNA4_SET', 1], ['PEARL_AMD_STAMP', 0], ['PEARL_AMD_ABLATE', 0],
      ['PEARL_AMD_W12_WR', 4]]) {
      expect([n, v, on({ ...W64, [n]: v })]).toEqual([n, v, 1]);
    }
    // The Windows core's flags, as build-amd.sh sets them, and its passes are wave32 (winlib has no arch_flags).
    expect(grab(BUILD, /^if windows; then SET_OS="(-DPEARL_AMD_RDNA4_SET=0) /m)).toBe('-DPEARL_AMD_RDNA4_SET=0');
    expect(on({ ...W32, PEARL_AMD_RDNA4_SET: 0 })).toBe(0);
  });

  it('turns on the wave64 fold with B\' straight to registers, unless the build set those itself', () => {
    expect(FOLD).toContain(block + '#if PEARL_AMD_W64K\n#ifndef PEARL_AMD_W64\n#define PEARL_AMD_W64 1\n#endif\n'
      + '#ifndef PEARL_AMD_W64GB\n#define PEARL_AMD_W64GB 1\n#endif\n#endif\n'
      + '#ifndef PEARL_AMD_W64\n#define PEARL_AMD_W64 0\n#endif\n'
      + '#if PEARL_AMD_W64 && PEARL_AMD_WMMA12\n#define PEARL_AMD_W64_ON 1\n'
      + '#if defined(__AMDGCN_WAVEFRONT_SIZE) && __AMDGCN_WAVEFRONT_SIZE != 64\n'
      + '#error "PEARL_AMD_W64 needs -mwavefrontsize64"\n#endif\n');
    // Without W64K, B' goes through LDS, and the opt-in parts are off.
    expect(FOLD).toContain('#ifndef PEARL_AMD_W64GB\n#define PEARL_AMD_W64GB 0\n#endif\n');
    expect(FOLD).toContain('#ifndef PEARL_AMD_W64KD\n#define PEARL_AMD_W64KD 0\n#endif\n');
    expect(FOLD).toContain('#ifndef PEARL_AMD_W64K_GRP\n#define PEARL_AMD_W64K_GRP 4\n#endif\n');
    expect(FOLD).toContain('#ifndef PEARL_AMD_W64K_ORD\n#define PEARL_AMD_W64K_ORD 0   // the WMMA order within a group of A\' tiles (slab_step)\n#endif\n');
    expect(FOLD).toContain('#if PEARL_AMD_W64KD < 0 || PEARL_AMD_W64KD > 3\n#error "PEARL_AMD_W64KD is 0, 1, 2 or 3"\n#endif\n'
      + '#if PEARL_AMD_W64K_GRP != 1 && PEARL_AMD_W64K_GRP != 2 && PEARL_AMD_W64K_GRP != 4 && PEARL_AMD_W64K_GRP != 8\n'
      + '#error "PEARL_AMD_W64K_GRP is 1, 2, 4 or 8"\n#endif\n');
    // The wmma12 kernel runs the wave64 policy only then; every other pass keeps the wave32 one.
    expect(FOLD).toContain('#if PEARL_AMD_EMU || PEARL_AMD_WMMA12\n#if PEARL_AMD_W64_ON\n'
      + '  pearl_amd_fold_body<PearlAmdWmma12W64>(Ap, Bp, k, rank, col0, rows_valid, tr PEARL_AMD_FUSE_ARGS);   // PEARL_AMD_W64\n'
      + '#else\n  pearl_amd_fold_body<PearlAmdWmma12>(Ap, Bp, k, rank, col0, rows_valid, tr PEARL_AMD_FUSE_ARGS);\n#endif\n');
    expect(FOLD.match(/pearl_amd_fold_body<PearlAmdWmma12W64>/g)).toHaveLength(1);
    // PEARL_AMD_W12_WR may be 8 only for W64KD's 512x256 block, which the fold body refuses without it.
    expect(FOLD).toContain('#if PEARL_AMD_W12_WR != 2 && PEARL_AMD_W12_WR != 4 && PEARL_AMD_W12_WR != 8\n'
      + '#error "PEARL_AMD_W12_WR is 2 or 4 (8 with PEARL_AMD_W64KD: a 512x256 block of 16 waves64)"\n#endif\n');
    expect(FOLD).toContain('  constexpr bool kKD = kW64 && KS == 64 && PEARL_AMD_W64KD != 0;   // PEARL_AMD_W64KD (above): no staging at all\n'
      + '  static_assert(BM <= 256 || kKD, "a 512-row block needs PEARL_AMD_W64KD: its staging would not fit in LDS");\n');
  });
});

describe('the caps bit PEARL_AMD_CAP_SLAB64', () => {
  const caps = slice(FOLD, 'extern "C" __global__ void pearl_amd_caps(uint32_t *out) {', '\n}\n');

  it('is a bit of its own, set only by a code object whose wmma12 fold is W64K\'s', () => {
    const bits = [...FOLD.matchAll(/#define PEARL_AMD_CAP_(\w+) (0x[0-9a-f]+|\d+)u/g)].map((m) => [m[1], Number(m[2])]);
    const v = Object.fromEntries(bits);
    expect([v.W64, v.SLAB64]).toEqual([0x10000, 0x20000]);
    expect(new Set(bits.map((b) => b[1])).size).toBe(bits.length);
    for (const [, x] of bits) expect(x & (x - 1)).toBe(0);
    expect(caps).toContain('  if (PEARL_AMD_REG2_ON) out[0] |= PEARL_AMD_CAP_REG2;\n'
      + '  if (PEARL_AMD_REG2_ON && PEARL_AMD_W64_ON) out[0] |= PEARL_AMD_CAP_W64;\n'
      + '  if (PEARL_AMD_W64_ON && PEARL_AMD_W64K) out[0] |= PEARL_AMD_CAP_SLAB64;\n'
      + '  if (PEARL_AMD_KWRAP) out[0] |= PEARL_AMD_CAP_KWRAP;\n');
    expect(code(FOLD).match(/PEARL_AMD_CAP_SLAB64/g)).toHaveLength(2);   // its #define and that line
    // As JS: W64K alone is not enough (a pass with no wave64 fold), and a wave64 fold with the 32-byte slab has no bit.
    const set = new Function('PEARL_AMD_W64_ON', 'PEARL_AMD_W64K',
      'return Boolean(' + grab(caps, / {2}if \((.*)\) out\[0\] \|= PEARL_AMD_CAP_SLAB64;/) + ');');
    expect([set(1, 1), set(1, 0), set(0, 1), set(0, 0)]).toEqual([true, false, false, false]);
  });

  it('stands for the fold\'s own slab: 64 bytes in the wave64 fold with W64K, else PEARL_AMD_SLAB', () => {
    expect(FOLD).toContain('#define PEARL_AMD_SLAB 32u\n');
    expect(FOLD).toContain('  constexpr int KS = std::is_same<P, PearlAmdDot4>::value ? (int)PEARL_AMD_DOT4_SLAB\n'
      + '                     : std::is_same<P, PearlAmdWmma12W64>::value && PEARL_AMD_W64K ? 64 : (int)PEARL_AMD_SLAB;\n');
    expect(FOLD).toContain('  constexpr bool kW64K = kW64 && KS == 64;   // W64K (above): 16 bytes a lane, four k-steps\n');
    expect(FOLD).toContain('  static_assert(!kW64K || (kGb && PEARL_AMD_W64G), "W64K needs PEARL_AMD_W64GB");\n');
    // A chunk is then two slabs, and the C4 loop and its readout count them so.
    const RANK = Number(grab(CONFIG, /#define PEARL_FOLD_RANK (\d+)u/));
    expect([RANK / 64, RANK / 32]).toEqual([2, 4]);
    expect(FOLD).toContain('  static_assert(PEARL_FOLD_RANK / (uint32_t)KS == 4u || (kW64K && PEARL_FOLD_RANK / (uint32_t)KS == 2u),\n'
      + '                "PEARL_AMD_C4 is for chunks of four slabs (two in W64K)");\n');
    expect(FOLD).toContain('      if constexpr (J == 3) readout(s / (PEARL_FOLD_RANK / (uint32_t)KS));\n');
  });

  it('W64K\'s chunk loop reads out every chunk but the last once, in order, at its second slab', () => {
    const loop = slice(FOLD, '  if constexpr (kW64K) {\n    // W64K: a chunk is two slabs.', '\n  } else {\n');
    expect(loop).toBe('  if constexpr (kW64K) {\n'
      + '    // W64K: a chunk is two slabs. Two chunks at a time (s a multiple of 4), then one; place 3 ends a chunk.\n'
      + '#if PEARL_AMD_C4 == 2\n'
      + '    for (; s + 4 < slabs; s += 4) {\n'
      + '      __builtin_assume(s + 6u <= slabs);   // whole chunks: slab s + 5, the last loaded here, is there\n'
      + '#pragma unroll\n'
      + '      for (int h = 0; h < 2; h++) {\n'
      + '        const uint32_t s2 = s + 2u * (uint32_t)h;\n'
      + '        slab_step(s2, std::integral_constant<int, 0>(), std::integral_constant<int, 2>());\n'
      + '        slab_step(s2 + 1, std::integral_constant<int, 1>(), std::integral_constant<int, 3>());\n'
      + '      }\n'
      + '    }\n'
      + '#endif\n'
      + '    for (; s + 2 < slabs; s += 2) {\n'
      + '      __builtin_assume(s + 4u <= slabs);\n'
      + '      slab_step(s, std::integral_constant<int, 0>(), std::integral_constant<int, 2>());\n'
      + '      slab_step(s + 1, std::integral_constant<int, 1>(), std::integral_constant<int, 3>());\n'
      + '    }\n'
      + '  } else {\n');
    // That loop as JS, for C4 = 2 and 1: the slabs it steps, their parity, and the chunk place 3 reads out.
    const RANK = Number(grab(CONFIG, /#define PEARL_FOLD_RANK (\d+)u/)), KS = 64;
    for (const c4 of [1, 2]) {
      for (const k of [128, 256, 1024, 2048, 4096, 8192]) {
        const slabs = k / KS, steps = [], chunks = [];
        const step = (s, parity, place) => {
          steps.push(s);
          expect(parity).toBe(s % 2);
          if (place === 3) chunks.push(Math.floor(s / (RANK / KS)));
        };
        let s = 0;
        if (c4 === 2) {
          for (; s + 4 < slabs; s += 4) {
            expect(s + 6).toBeLessThanOrEqual(slabs);
            for (let h = 0; h < 2; h++) { const s2 = s + 2 * h; step(s2, 0, 2); step(s2 + 1, 1, 3); }
          }
        }
        for (; s + 2 < slabs; s += 2) {
          expect(s + 4).toBeLessThanOrEqual(slabs);
          step(s, 0, 2); step(s + 1, 1, 3);
        }
        // The last chunk's two slabs are left for the loop after it.
        expect([c4, k, s]).toEqual([c4, k, slabs - 2]);
        expect(steps).toEqual([...Array(slabs - 2).keys()]);
        expect(chunks).toEqual([...Array(k / RANK - 1).keys()]);
      }
    }
  });
});

describe('W64K\'s 64-byte LDS rows, run as JS', () => {
  // The wave64 fold's block at PEARL_AMD_W12_WR=4: 2 x 4 waves64 of 128 x 64, 512 threads, 256 rows of A'. B' goes
  // straight to registers (W64GB), so only A' is staged: 16-byte units, 4 a row.
  const W = 64, WR = 2, WC = 4, MT = 8, TM = 16, WM = MT * TM, S = 64, QR = S / 16, T = W * WR * WC, BM = WR * WM;
  const UA = BM * QR;

  it('is the arithmetic in the fold', () => {
    const pol = slice(FOLD, 'struct PearlAmdWmma12W64 {', '\n};\n');
    expect(pol).toContain('  static constexpr int kWave = 64, kTM = 16, kTN = 16, kTK = 16;\n'
      + '  static constexpr int kWR = PEARL_AMD_W12_WR / 2, kWC = 4, kMT = 8, kNT = 4;');
    expect(FOLD).toContain('  constexpr int QR = KS / 16;');
    expect(FOLD).toContain('  constexpr int UA = BM * QR, UB = BN * QR, U = (UA + UB) / T;');
    expect(FOLD).toContain('  constexpr int S = kL32 ? KS : KS + 16;\n');
    // The store: the thread's quarter of its row, turned by the row's bits 2 and 3.
    expect(FOLD).toContain('  const int l32sw = kL32 ? (KS == 64 ? ((t / QR) >> 2) & 3 : pearl_amd_l32_swz(t / QR)) : 0;   // W64K: quarters\n');
    expect(FOLD).toContain('  int l32st = (t / QR) * S + 16 * ((t % QR) ^ l32sw);\n');
    expect(FOLD).toContain('    int8_t *dst = (u * T < UA ? sA[buf] + ((u * T) / QR) * S : sB[buf] + ((u * T - UA) / QR) * S) + l32st; \\\n');
    expect(FOLD).toContain('if constexpr (kKD) continue; if constexpr (kGbAny) { if (u * T >= UA) continue; } \\\n');
    // The read: lane group q (lane / 16) takes quarter q of its row, where the store put it.
    expect(FOLD).toContain('  int l32rd = kW64 && KS == 64 ? (lane & 15) * S + 16 * ((lane >> 4) ^ (((lane & 15) >> 2) & 3))   // W64K (above)\n');
    expect(FOLD).toContain('    const int8_t *aBase = sA[B] + wr * WM * S;\n');
    expect(FOLD).toContain('                  qa[g0 + i] = *reinterpret_cast<const pearl_amd_v4i *>(aBase + (g0 + i) * P::kTM * S + l32rd);\n');
    // Word j of a read is k-step j's fragment, for A' from LDS and B' from its buffer load alike.
    expect(FOLD).toContain('                  for (int nt = 0; nt < P::kNT; nt++) P::mma(acc[g0 + i][nt], qa[i][j], bq[nt][j]);\n');
    expect(FOLD).toContain('  const int gbOff = (wc * WN + (lane & 15)) * KS + (kW64 && !kW64K ? 8 : 16) * (lane >> 4);\n');
  });

  it('reads every quarter of every A\' row where the staging stored it', () => {
    const stored = new Map();
    for (let t = 0; t < T; t++) {
      const sw = ((t / QR | 0) >> 2) & 3;
      for (let u = 0; u * T < UA; u++) {
        const idx = t + u * T, row = idx / QR | 0;
        // One swap a thread: every row a thread stores has its first row's bits 2 and 3.
        expect((row >> 2) & 3).toBe(sw);
        const at = ((u * T) / QR) * S + (t / QR | 0) * S + 16 * ((t % QR) ^ sw);
        expect(stored.has(at)).toBe(false);
        stored.set(at, `${row} ${idx % QR}`);
      }
    }
    expect(stored.size).toBe(BM * QR);
    expect(Math.max(...stored.keys())).toBe(BM * S - 16);
    for (let wr = 0; wr < WR; wr++) {
      for (let mt = 0; mt < MT; mt++) {
        for (let lane = 0; lane < W; lane++) {
          const rd = (lane & 15) * S + 16 * ((lane >> 4) ^ (((lane & 15) >> 2) & 3));
          expect(stored.get(wr * WM * S + mt * TM * S + rd)).toBe(`${wr * WM + mt * TM + (lane & 15)} ${lane >> 4}`);
        }
      }
    }
  });

  it('keeps a lane group\'s 16 reads off each other\'s banks', () => {
    // A 16-byte read is 4 of the 64 banks a lane; the 16 lanes of a group cover each bank once. With no turn they
    // would share 16 banks, four reads a bank.
    const banks = (q, turn) => [...Array(16).keys()].flatMap((r) => {
      const dw = (r * S + 16 * (q ^ (turn ? (r >> 2) & 3 : 0))) / 4;
      return [0, 1, 2, 3].map((i) => (dw + i) % 64);
    });
    for (let q = 0; q < 4; q++) {
      expect(new Set(banks(q, true)).size).toBe(64);
      expect(new Set(banks(q, false)).size).toBe(16);
    }
  });
});

describe('the host and PEARL_AMD_CAP_SLAB64', () => {
  it('learns the slab and the rows from the card\'s caps, never from a macro', () => {
    expect(HOST).toContain('  bool slab64 = false;      // PEARL_AMD_CAP_SLAB64: this card\'s wmma12 fold stages 64 bytes of k (PEARL_AMD_W64K)\n'
      + '  bool w64Cap = false;      // and its rows are PEARL_AMD_W64\'s (PEARL_AMD_CAP_W64: PEARL_AMD_R4_ROWS_*)\n');
    expect(HOST).toContain('  ctx->reg2Cap = (caps[0] & PEARL_AMD_CAP_REG2) != 0;\n'
      + '  ctx->w64Cap = (caps[0] & PEARL_AMD_CAP_W64) != 0;\n'
      + '  ctx->slab64 = (caps[0] & PEARL_AMD_CAP_SLAB64) != 0;\n');
    // The host pass is not a gfx12 device pass, so the macros say nothing about the card there.
    expect(code(HOST)).not.toMatch(/PEARL_AMD_W64/);
    expect(code(FOLDTEST)).not.toMatch(/PEARL_AMD_W64/);
    expect(code(HOST).match(/slab64/g)).toHaveLength(4);   // the field, the caps read, and the two uses below
    // The fold table keeps one slab a fold: the wmma12 row's is PEARL_AMD_SLAB.
    expect(HOST).toContain('    {"wmma12", PEARL_AMD_CAP_WMMA12, PEARL_AMD_W12_BM, PEARL_AMD_BN, PEARL_AMD_SLAB,\n');
  });

  it('tiles the operands by 64 bytes for wmma12 only when the cap is set', () => {
    const draw = slice(HOST, 'void draw_noise(Ctx *ctx, bool isA) {', '\n}\n');
    expect(draw).toContain('    uint32_t kbLog2 = 0;\n'
      + '    // The card\'s code object may stage a slab of its own (PEARL_AMD_CAP_SLAB64): this pass can\'t know from a macro.\n'
      + '    const uint32_t slab = ctx->slab64 && tf->cap == PEARL_AMD_CAP_WMMA12 ? 64u : tf->slab;\n'
      + '    while ((1u << kbLog2) < slab) kbLog2++;\n'
      + '    hipLaunchKernelGGL(pearl_materialize16_tiled,');
    expect(draw).not.toContain('< tf->slab)');
    // That choice as JS, over every fold of the table.
    const expr = grab(draw, /const uint32_t slab = (.*);/).replace(/(\d+)u\b/g, '$1');
    const pick = new Function('ctx', 'tf', 'PEARL_AMD_CAP_WMMA12', 'return ' + expr.replace(/->/g, '.') + ';');
    const capOf = (n) => Number(grab(FOLD, new RegExp('#define PEARL_AMD_CAP_' + n + ' (\\d+)u')));
    const SLAB = Number(grab(FOLD, /#define PEARL_AMD_SLAB (\d+)u/)), DOT4 = Number(grab(FOLD, /#define PEARL_AMD_DOT4_SLAB (\d+)u?\n/));
    expect([SLAB, DOT4]).toEqual([32, 64]);
    const folds = [...slice(HOST, 'const FoldInfo kFolds[] = {', '\n};\n').matchAll(/\{"(\w+)", PEARL_AMD_CAP_(\w+), \w+, \w+, (\w+),/g)]
      .map((m) => ({ name: m[1], cap: capOf(m[2]), slab: m[3] === 'PEARL_AMD_DOT4_SLAB' ? DOT4 : SLAB, macro: m[3] }));
    expect(folds.map((f) => f.name)).toEqual(['ref', 'mfma16', 'mfma8', 'wmma11', 'wmma12', 'dot4']);
    for (const f of folds) {
      expect([f.name, pick({ slab64: false }, f, capOf('WMMA12'))]).toEqual([f.name, f.slab]);
      expect([f.name, pick({ slab64: true }, f, capOf('WMMA12'))]).toEqual([f.name, f.name === 'wmma12' ? 64 : f.slab]);
    }
    const log2 = (slab) => { let b = 0; while ((1 << b) < slab) b++; return b; };
    expect([log2(32), log2(64)]).toEqual([5, 6]);
  });

  it('holds the rank to that slab in submit, and names it in the error', () => {
    const submit = slice(HOST, 'extern "C" bool pearl_host_submit(', '\n}\n');
    expect(submit).toContain('  const uint32_t fslab = ctx->slab64 && f.cap == PEARL_AMD_CAP_WMMA12 ? 64u : f.slab;\n'
      + '  if (rank == 0 || k % rank != 0 || rank % fslab != 0 || !chunksOk\n');
    expect(submit).toContain('               (unsigned)fslab, ctx->rank128 ? ", exactly 128 for this card\'s matrix fold" : "",\n');
    expect(code(submit).match(/\bf\.slab\b/g)).toHaveLength(1);
  });
});

describe('the fold test and PEARL_AMD_CAP_SLAB64', () => {
  it('tiles the wmma12 fold\'s A\' and B\' by the slab the caps name, and frees the copies', () => {
    expect(FOLDTEST).toContain('  const uint32_t slab12 = (caps[0] & PEARL_AMD_CAP_SLAB64) ? 64u : (uint32_t)PEARL_AMD_SLAB;\n');
    expect(FOLDTEST).toContain('    if (PEARL_AMD_W12_BM != PEARL_AMD_BM || slab12 != PEARL_AMD_SLAB) {\n'
      + '      if (!ok(hipMalloc(&dAt12, A.size()), "hipMalloc A tiled (wmma12)")) return 2;\n'
      + '      const std::vector<int8_t> At12 = tiled(A, m, PEARL_AMD_W12_BM, slab12);\n');
    expect(FOLDTEST).toContain('    if (slab12 != PEARL_AMD_SLAB) {\n'
      + '      if (!ok(hipMalloc(&dBt12, B.size()), "hipMalloc B tiled (wmma12)")) return 2;\n'
      + '      const std::vector<int8_t> Bt12 = tiled(B, n, PEARL_AMD_BN, slab12);\n');
    // Only the wmma12 fold reads them; every other fold keeps its operands.
    expect(FOLDTEST).toContain('const int8_t *fa = f.tall ? (dot4 ? dAt4 : f.cap == PEARL_AMD_CAP_WMMA12 ? dAt12 : dAt) : dA, '
      + '*fb = f.tall ? (dot4 ? dBt4 : f.cap == PEARL_AMD_CAP_WMMA12 ? dBt12 : dBt) : dB;');
    expect(FOLDTEST).toContain('  if (dBt12 != dBt && dBt12 != dB) hipFree(dBt12);\n');
    // A wave64 card's rows are checked against PEARL_AMD_R4_ROWS_PATTERN's, from the caps too.
    expect(FOLDTEST).toContain('  const bool w64 = (caps[0] & PEARL_AMD_CAP_W64) != 0;\n'
      + '  const uint32_t *const r2Rows = w64 ? PEARL_AMD_R4_ROWS_PATTERN : PEARL_AMD_R2_ROWS_PATTERN;\n'
      + '  const uint32_t r2Mask = w64 ? PEARL_AMD_R4_ROWS_MASK : PEARL_AMD_R2_ROWS_MASK;\n');
  });
});
