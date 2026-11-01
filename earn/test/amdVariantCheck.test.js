'use strict';

// earn/native/amd/test/variant-check.sh: what each build variant compiles to on a card's target, and which a timed
// run keeps, with nothing timed. Run here on stand-ins: a build-amd.sh whose check prints a row (and its remarks)
// from a table, and a compiler whose device assembly is made up per variant. No HIP compiler runs here; the shell
// parts run under bash where there is one.

const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');

const AMD = path.join(__dirname, '..', 'native', 'amd');
const SCRIPT = fs.readFileSync(path.join(AMD, 'test', 'variant-check.sh'), 'utf8').replace(/\r\n/g, '\n');
const BUILD = fs.readFileSync(path.join(AMD, 'build-amd.sh'), 'utf8').replace(/\r\n/g, '\n');

const hasBash = spawnSync('bash', ['--version'], { stdio: 'ignore' }).status === 0;
const bashIt = hasBash ? it : it.skip;

// A stand-in build-amd.sh: `check ARCH` prints build-amd.sh's header and rows for the flags in PEARL_AMD_CFLAGS,
// from STUB_ROWS ("flags<TAB>VGPRs scratch spills LDS waves SGPRs"), and writes the remarks the real one leaves.
const STUB_BUILD = [
  '#!/bin/bash',
  'a=$2',
  'mkdir -p build/check',
  'row=$(awk -F\'\\t\' -v f="${PEARL_AMD_CFLAGS:-}" \'$1 == f { print $2; exit }\' "$STUB_ROWS")',
  'if [ -z "$row" ]; then echo "$a: compile failed, see build/check/$a.txt" >&2; : > "build/check/$a.txt"; exit 1; fi',
  'read -r v sc sp l w sg <<< "$row"',
  'case "$a" in gfx11*) k=pearl_amd_fold_wmma11 kw=0 ;; *) k=pearl_amd_fold_wmma12 kw=1 ;; esac',
  '{ echo "x.cpp:1:1: remark: Function Name: pearl_amd_hash [-Rpass-analysis=kernel-resource-usage]"',
  '  echo "x.cpp:1:1: remark:     SGPRs: 99 [-Rpass-analysis=kernel-resource-usage]"',
  '  echo "x.cpp:1:1: remark: Function Name: $k [-Rpass-analysis=kernel-resource-usage]"',
  '  echo "x.cpp:1:1: remark:     SGPRs: $sg [-Rpass-analysis=kernel-resource-usage]"',
  '  echo "x.cpp:1:1: remark:     VGPRs: $v [-Rpass-analysis=kernel-resource-usage]"; } > "build/check/$a.txt"',
  'r() { printf \'%-9s %-22s %6s %6s %8s %7s %8s %10s %5s\\n\' "$@"; }',
  'r target kernel VGPRs AGPRs scratch spills LDS "waves/SIMD" wrap',
  'r "$a" pearl_amd_hash 40 - 0 0 0 16 "$kw"',
  'r "$a" "$k" "$v" - "$sc" "$sp" "$l" "$w" "$kw"',
  '[ "$sc" = 0 ] && [ "$sp" = 0 ] || { echo "check: a fold failed to compile, spills, or uses scratch memory" >&2; exit 1; }',
  '',
].join('\n');
// A stand-in compiler: the -D flags pick the assembly from STUB_ASM ("flags<TAB>file"); it logs its arguments to
// STUB_ARGS.
const STUB_CC = [
  '#!/bin/bash',
  'echo "$*" >> "$STUB_ARGS"',
  'flags=\'\' out=\'\'',
  'while [ $# -gt 0 ]; do',
  '  case "$1" in -o) out=$2; shift ;; -D*) flags="${flags:+$flags }$1" ;; esac',
  '  shift',
  'done',
  'asm=$(awk -F\'\\t\' -v f="$flags" \'$1 == f { print $2; exit }\' "$STUB_ASM")',
  '[ -n "$asm" ] || { echo "error: no assembly for \'$flags\'" >&2; exit 1; }',
  'cp "$asm" "$out"',
  '',
].join('\n');

// Device assembly with the fold made of these ops: Gn global loads, Sn LDS stores, Dn LDS loads, Wn WMMAs, B a
// barrier, w a wait on LDS, L a block label. Another kernel comes first, with loads of its own that don't count.
function asm(gfx, ops) {
  const one = {
    G: '\tglobal_load_b128 v[0:3], v4, s[0:1]',
    S: '\tds_store_b128 v5, v[0:3]',
    D: '\tds_load_2addr_b64 v[6:9], v10 offset1:2',
    W: '\tv_wmma_i32_16x16x16_iu8 v[20:27], v[0:1], v[2:3], v[20:27]   ; a comment',
  };
  const out = ['\t.text', '\t.p2align 8', 'pearl_amd_hash:                         ; @pearl_amd_hash',
    '\tglobal_load_b128 v[0:3], v4, s[0:1]', '\tds_load_b64 v[0:1], v2', '\ts_endpgm', '.Lfunc_end0:',
    '\t.p2align 8', `pearl_amd_fold_${gfx === 11 ? 'wmma11' : 'wmma12'}:      ; @fold`, '; %bb.0:'];
  let label = 0;
  for (const op of ops) {
    const n = Number(op.slice(1)) || 1;
    if (op[0] === 'B') out.push(...(gfx === 11 ? ['\ts_barrier'] : ['\ts_barrier_signal -1', '\ts_barrier_wait -1']));
    else if (op[0] === 'w') out.push(gfx === 11 ? '\ts_waitcnt vmcnt(3) lgkmcnt(0)' : '\ts_wait_dscnt 0x0');
    else if (op[0] === 'L') out.push(`.LBB2_${++label}:                                ; %bb.${label}`, '\t.p2align 6');
    else for (let i = 0; i < n; i++) out.push(one[op[0]]);
  }
  out.push('\ts_endpgm', '.Lfunc_end2:', '\t.section\t.AMDGPU.csdata');
  return out.join('\n') + '\n';
}
// The release's fold, as the gfx1201 one counts: 9 LDS stores, 12 global loads, 24 LDS loads, and each slab barrier
// followed by the next slab's fragment loads and a wait before any WMMA.
const RELEASE = ['G3', 'S3', 'B', 'G3', 'L', 'G3', 'D8', 'w', 'W32', 'S3', 'B', 'G3', 'D8', 'w', 'W32', 'S3', 'B', 'L',
  'D8', 'w', 'W32', 'B'];
// A made-up schedule: the same stores and global loads, 8 more LDS loads, and 16 WMMAs after each slab barrier before
// the wait.
const MOVED = ['G3', 'S3', 'B', 'G3', 'D8', 'L', 'G3', 'w', 'W16', 'S3', 'B', 'D8', 'W16', 'G3', 'w', 'W16', 'S3', 'B',
  'D8', 'W16', 'L', 'w', 'W16', 'B', 'D8', 'W16'];

describe('variant-check.sh', () => {
  let tmp;
  beforeEach(() => { tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'amd-variant-check-')); });
  afterEach(() => { fs.rmSync(tmp, { recursive: true, force: true }); });

  // Runs a copy of the script from a stand-in earn/native/amd. variants: [name, flags, row, assembly or null].
  function run(variants, env, gfx = 12) {
    const amd = path.join(tmp, 'amd');
    fs.mkdirSync(path.join(amd, 'test'), { recursive: true });
    fs.writeFileSync(path.join(amd, 'build-amd.sh'), STUB_BUILD, { mode: 0o755 });
    fs.writeFileSync(path.join(tmp, 'cc'), STUB_CC, { mode: 0o755 });
    fs.writeFileSync(path.join(amd, 'test', 'variant-check.sh'), SCRIPT.replace('/proc/1/fd/1', '/dev/null'));
    const rows = [], asms = [];
    variants.forEach(([name, flags, row, ops], i) => {
      if (row) rows.push(flags + '\t' + row);
      if (ops) {
        const f = path.join(tmp, `v${i}.s`);
        fs.writeFileSync(f, asm(gfx, ops));
        asms.push(flags + '\t' + f.replace(/\\/g, '/'));
      }
    });
    fs.writeFileSync(path.join(tmp, 'rows.txt'), rows.join('\n') + '\n');
    fs.writeFileSync(path.join(tmp, 'asm.txt'), asms.join('\n') + '\n');
    const fwd = (f) => path.join(tmp, f).replace(/\\/g, '/');
    // CHECK_CC names the compiler; a CC in the environment (often gcc) must not be used.
    const e = { ...process.env, ARCH: gfx === 11 ? 'gfx1100' : 'gfx1201', CC: fwd('no-such-cc'), CHECK_CC: fwd('cc'),
      CHECK_ROOT: fwd('out'),
      STUB_ROWS: fwd('rows.txt'), STUB_ASM: fwd('asm.txt'), STUB_ARGS: fwd('args.txt'), KEEP_TO: fwd('keep.txt'),
      VARIANTS: variants.map(([n, f]) => n + '=' + f).join(';'), ...env };
    delete e.TWINS;
    if (env && env.TWINS) e.TWINS = env.TWINS;
    const r = spawnSync('bash', ['test/variant-check.sh'], { cwd: amd, encoding: 'utf8', env: e });
    return {
      status: r.status,
      err: r.stderr,
      lines: r.stdout.split('\n').filter(Boolean).map((l) => l.replace(/^\[amdtest\] \S+ check /, '')),
      keep: fs.existsSync(path.join(tmp, 'keep.txt')) ? fs.readFileSync(path.join(tmp, 'keep.txt'), 'utf8') : null,
      args: fs.existsSync(path.join(tmp, 'args.txt')) ? fs.readFileSync(path.join(tmp, 'args.txt'), 'utf8') : '',
    };
  }

  bashIt('parses', () => {
    expect(spawnSync('bash', ['-n', path.join(AMD, 'test', 'variant-check.sh')]).status).toBe(0);
  });

  bashIt('prints each variant\'s row, SGPRs, staging counts and barrier cover, and keeps the ones that fit', () => {
    const { status, lines, keep, args } = run([
      ['base', '', '198 0 0 36864 6 70', RELEASE],
      ['stamp', '-DPEARL_AMD_STAMP=1', '198 0 0 36864 6 81', RELEASE.concat(['W1'])],
      ['abl8', '-DPEARL_AMD_ABLATE=8', '188 0 0 36864 6 70', RELEASE],
      ['abl8s', '-DPEARL_AMD_ABLATE=8 -DPEARL_AMD_STAMP=1', '205 0 0 36864 6 81', RELEASE],
      ['abl2s', '-DPEARL_AMD_ABLATE=2 -DPEARL_AMD_STAMP=1', '218 0 0 36864 6 81', RELEASE],
      ['moved', '-DMOVED', '204 0 0 36864 6 72', MOVED],
      ['wave5', '-DWAVE5', '245 0 0 36864 5 72', MOVED],
      ['spills', '-DSPILLS', '240 0 3 36864 6 72', MOVED],
      ['lds0', '-DLDS0', '170 0 0 0 8 60', RELEASE],
      ['stores', '-DSTORES', '198 0 0 36864 6 70', RELEASE.slice(0, -4).concat(['S2', 'B'])],
      ['noasm', '-DNOASM', '198 0 0 36864 6 70', null],
      ['norow', '-DNOROW', null, RELEASE],
    ], { TWINS: 'stamp=base abl8s=abl8 abl2s~stamp' });
    expect(status).toBe(0);
    expect(lines[0]).toBe('start gfx1201 fold wmma12; variants: base=;stamp=-DPEARL_AMD_STAMP=1;abl8=-DPEARL_AMD_ABLATE=8;'
      + 'abl8s=-DPEARL_AMD_ABLATE=8 -DPEARL_AMD_STAMP=1;abl2s=-DPEARL_AMD_ABLATE=2 -DPEARL_AMD_STAMP=1;'
      + 'moved=-DMOVED;wave5=-DWAVE5;spills=-DSPILLS;lds0=-DLDS0;stores=-DSTORES;'
      + 'noasm=-DNOASM;norow=-DNOROW; twins: stamp=base abl8s=abl8 abl2s~stamp');
    const row = (v, l, w) => `gfx1201   pearl_amd_fold_wmma12     ${v}      -        0       0    ${l}          ${w}     1`;
    expect(lines).toContain(`base (release): ${row(198, 36864, 6)}; SGPRs 70; fold code 9 12 24 (LDS stores, global loads, `
      + 'LDS loads); WMMAs after each barrier before an LDS wait: 0 0 0 0+ '
      + '(after the loop\'s first: GGGDDDDDDDD); LDS loads between the last WMMA and a barrier: 0');
    expect(lines).toContain(`stamp (-DPEARL_AMD_STAMP=1): ${row(198, 36864, 6)}; SGPRs 81; fold code 9 12 24 (LDS stores, `
      + 'global loads, LDS loads); WMMAs after each barrier before an LDS wait: 0 0 0 1+ '
      + '(after the loop\'s first: GGGDDDDDDDD); LDS loads between the last WMMA and a barrier: 0; code differs '
      + 'from base');
    expect(lines).toContain(`abl8 (-DPEARL_AMD_ABLATE=8): ${row(188, 36864, 6)}; SGPRs 70; fold code 9 12 24 (LDS stores, `
      + 'global loads, LDS loads); WMMAs after each barrier before an LDS wait: 0 0 0 0+ '
      + '(after the loop\'s first: GGGDDDDDDDD); LDS loads between the last WMMA and a barrier: 0; the same code '
      + 'as base');
    expect(lines).toContain(`moved (-DMOVED): ${row(204, 36864, 6)}; SGPRs 72; fold code 9 12 32 (LDS stores, `
      + 'global loads, LDS loads); WMMAs after each barrier before an LDS wait: 0 16 16 16+ '
      + '(after the loop\'s first: DDDDDDDDWWWWWWWWWWWWWWWWGGG); LDS loads between the last WMMA and a barrier: 0; '
      + 'code differs from base');
    // Kept: the rows that fit, and twins with their twin's VGPRs; a ~ pair is only a note.
    expect(lines.filter((l) => / dropped: |VGPRs, stamp|^kept: /.test(l))).toEqual([
      'abl8s dropped: 205 VGPRs, not abl8\'s 188',
      'abl2s has 218 VGPRs, stamp 198: comparing them compares two schedules',
      'wave5 dropped: not LDS 36864 and 6 waves with no scratch and no spills',
      'spills dropped: not LDS 36864 and 6 waves with no scratch and no spills',
      'lds0 dropped: not LDS 36864 and 6 waves with no scratch and no spills',
      'stores dropped: LDS stores and global loads 11 12, not base\'s 9 12',
      'noasm dropped: its code couldn\'t be read, so its LDS stores and global loads can\'t be held to base\'s',
      'norow dropped: not LDS 36864 and 6 waves with no scratch and no spills',
      'kept: base stamp abl8 abl2s moved',
    ]);
    expect(lines).toContain('norow (-DNOROW): no row; SGPRs ?; fold code 9 12 24 (LDS stores, global loads, LDS loads); '
      + 'WMMAs after each barrier before an LDS wait: 0 0 0 0+ '
      + '(after the loop\'s first: GGGDDDDDDDD); LDS loads between the last WMMA and a barrier: 0; the same code '
      + 'as base');
    expect(keep).toBe('base=\nstamp=-DPEARL_AMD_STAMP=1\nabl8=-DPEARL_AMD_ABLATE=8\n'
      + 'abl2s=-DPEARL_AMD_ABLATE=2 -DPEARL_AMD_STAMP=1\nmoved=-DMOVED\n');
    // The assembly is compiled as build-amd.sh check compiles the device code, with -S.
    const first = args.split('\n')[0];
    expect(first).toBe('-x hip --offload-arch=gfx1201 --rocm-path=/opt/rocm -std=c++17 -O3 -Icompat/gpu -Isrc -I../src '
      + '-Ibuild/gen --cuda-device-only -S src/pearl_host_hip.cpp -o ' + path.join(tmp, 'out').replace(/\\/g, '/')
      + '/base.s');
    expect(BUILD).toContain('"$cc" -x hip --offload-arch="$a" --rocm-path="$rocm" -std=c++17 -O3 $KWRAP_OS ${PEARL_AMD_CFLAGS:-} $inc --cuda-device-only');
    expect(BUILD).toContain('inc="-I$HERE/compat/gpu -I$HERE/src -I$SRC -I$GEN"');
  }, 60000);

  bashIt('reads gfx11\'s waits and barriers, and finds a variant whose code is the base\'s', () => {
    const { status, lines, keep } = run([
      ['base', '', '240 0 0 36864 6 90', RELEASE],
      ['same', '-DSAME', '240 0 0 36864 6 90', RELEASE],
      ['stamp', '-DPEARL_AMD_STAMP=1', '245 0 0 36864 5 101', RELEASE],
    ], { TWINS: 'stamp=base' }, 11);
    expect(status).toBe(0);
    expect(lines[0]).toMatch(/^start gfx1100 fold wmma11; /);
    expect(lines).toContain('same (-DSAME): gfx1100   pearl_amd_fold_wmma11     240      -        0       0    '
      + '36864          6     0; SGPRs 90; fold code 9 12 24 (LDS stores, global loads, LDS loads); WMMAs after each barrier '
      + 'before an LDS wait: 0 0 0 0+ (after the loop\'s first: GGGDDDDDDDD); LDS loads between the last WMMA and a '
      + 'barrier: 0; the same code as base');
    expect(lines).toContain('stamp dropped: not LDS 36864 and 6 waves with no scratch and no spills');
    expect(keep).toBe('base=\nsame=-DSAME\n');
  }, 60000);

  bashIt('keeps variants on their row alone when the base\'s code can\'t be read', () => {
    const { lines } = run([
      ['base', '', '198 0 0 36864 6 70', null],
      ['moved', '-DMOVED', '204 0 0 36864 6 72', MOVED.concat(['S5'])],
    ]);
    expect(lines).toContain('base: the fold\'s code couldn\'t be read, so the variants are kept on their row alone');
    expect(lines[lines.length - 1]).toBe('kept: base moved');
    expect(lines.find((l) => l.startsWith('moved ('))).toMatch(/fold code 14 12 32 .*: 0 16 16 16\+ \(after the loop's first: D{8}W{16}G{3}\); .*: 0$/);
  }, 60000);

  bashIt('counts the LDS loads between the last WMMA and a barrier, and keeps the build for them', () => {
    // The step's top loads sunk below its first k-step's WMMAs: 4 before the second barrier, 4 before the third.
    const late = MOVED.slice();
    late.splice(late.indexOf('S3'), 0, 'D4');
    late.splice(late.lastIndexOf('S3'), 0, 'D4');
    const vs = [
      ['base', '', '198 0 0 36864 6 70', RELEASE],
      ['late', '-DLATE', '198 0 0 36864 6 72', late],
      ['abl8', '-DPEARL_AMD_ABLATE=8', '188 0 0 36864 6 70', late],
    ];
    const r = run(vs);
    expect(r.lines.find((l) => l.startsWith('late ('))).toMatch(/; LDS loads between the last WMMA and a barrier: 8; /);
    expect(r.lines.filter((l) => / dropped: |^kept: /.test(l))).toEqual(['kept: base late abl8']);
    // The compiler was CHECK_CC's: the run sets CC to a file that doesn't exist.
    expect(r.args.split('\n').filter(Boolean)).toHaveLength(3);
    // The same on gfx11.
    const r11 = run(vs, {}, 11);
    expect(r11.lines.find((l) => l.startsWith('late ('))).toMatch(/; LDS loads between the last WMMA and a barrier: 8; /);
    expect(r11.lines.filter((l) => / dropped: |^kept: /.test(l))).toEqual(['kept: base late abl8']);
  }, 60000);

  bashIt('keeps a < variant with its twin\'s VGPRs or fewer, and drops one with more', () => {
    // The start skew's builds must fit in the release's registers: fewer is fine, as the waves are the same.
    const { status, lines, keep } = run([
      ['base', '', '198 0 0 36864 6 70', RELEASE],
      ['fewer', '-DFEWER', '190 0 0 36864 6 74', RELEASE],
      ['same', '-DSAME', '198 0 0 36864 6 74', RELEASE],
      ['more', '-DMORE', '206 0 0 36864 6 74', RELEASE],
      ['orphan', '-DORPHAN', '190 0 0 36864 6 74', RELEASE],
    ], { TWINS: 'fewer<base same<base more<base orphan<gone' });
    expect(status).toBe(0);
    expect(lines.filter((l) => / dropped: |^kept: /.test(l))).toEqual([
      'more dropped: 206 VGPRs, more than base\'s 198',
      'orphan dropped: its VGPRs can\'t be held to gone\'s, which weren\'t read',
      'kept: base fewer same',
    ]);
    expect(keep).toBe('base=\nfewer=-DFEWER\nsame=-DSAME\n');
    // On gfx11 the release's 240 is the most for 6 waves; one more VGPR is a wave less, which the row check drops first.
    const r11 = run([
      ['base', '', '240 0 0 36864 6 90', RELEASE],
      ['skew1', '-DPEARL_AMD_SKEW=1', '240 0 0 36864 6 94', RELEASE],
      ['skew2', '-DPEARL_AMD_SKEW=2', '245 0 0 36864 5 94', RELEASE],
    ], { TWINS: 'skew1<base skew2<base' }, 11);
    expect(r11.lines.filter((l) => / dropped: |^kept: /.test(l))).toEqual([
      'skew2 dropped: not LDS 36864 and 6 waves with no scratch and no spills',
      'kept: base skew1',
    ]);
  }, 60000);

  bashIt('refuses a card it has no fold for, and a twin it can\'t read', () => {
    const bad = run([['base', '', '198 0 0 36864 6 70', RELEASE]], { ARCH: 'gfx942' });
    expect([bad.status, bad.lines]).toEqual([2, ['no RDNA 3 or RDNA 4 target here (gfx942): nothing to check']]);
    const twins = run([['base', '', '198 0 0 36864 6 70', RELEASE]], { TWINS: 'stamp' });
    expect([twins.status, twins.lines]).toEqual([2, ['TWINS takes name=twin, name<twin or name~twin, not stamp']]);
  }, 60000);
});
