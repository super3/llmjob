'use strict';

// earn/native/amd/hand/handfold.js edits the gfx1201 wave64 fold's chunk loop in the device assembly, and
// build-amd.sh runs it when PEARL_AMD_CFLAGS carries -DPEARL_AMD_HAND=OPS (no release build does). No HIP compiler
// runs here: the edits are run on a stand-in for the compiler's loop, shaped as ROCm 7.2's clang writes it, and
// checked with hand/handsim.js, which follows what every WMMA adds to which sum, whether its A' read had arrived,
// and what is staged where.

const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');

const HAND = path.join(__dirname, '..', 'native', 'amd', 'hand');
const { OPS, findFunction, findLoops, opOf } = require(path.join(HAND, 'handfold.js'));
const { simulate } = require(path.join(HAND, 'handsim.js'));
const BUILD = fs.readFileSync(path.join(HAND, '..', 'build-amd.sh'), 'utf8').replace(/\r\n/g, '\n');

const hasBash = spawnSync('bash', ['--version'], { stdio: 'ignore' }).status === 0;
const bashIt = hasBash ? it : it.skip;

// The loop as the compiler writes it: four slabs of 64 bytes of k, two chunks. A slab reads tiles 0-3's A' from LDS
// (v176-191), runs each tile's 16 WMMAs after a wait for its own read, reads tiles 4-7 into the same registers,
// then loads the next slab's B', XORs the sums on a chunk's last slab, stores the staged A' and meets the barrier.
function fixture() {
  const L = [];
  const add = (...l) => L.push(...l.map((x) => (x.startsWith('.') || x.startsWith('pearl') || x.startsWith(';') ? x : '\t' + x)));
  add('\t.text', 'pearl_amd_fold_wmma12:                  ; @pearl_amd_fold_wmma12', '; %bb.0:',
    'v_lshlrev_b32_e32 v161, 4, v0', 'v_lshlrev_b32_e32 v172, 2, v0', 'v_and_or_b32 v162, v2, 48, v1',
    'v_and_or_b32 v169, v3, 48, v10', 'v_or_b32_e32 v163, 0x2000, v161', 'v_and_or_b32 v164, 0x3030, v9, v10',
    'v_lshrrev_b32_e32 v165, 8, v0', 'v_or_b32_e32 v166, 0x400, v164', 'v_or_b32_e32 v167, 0x800, v164',
    'v_or_b32_e32 v168, 0xc00, v164', 'v_and_b32_e32 v171, 0x1fe, v0', 'v_lshlrev_b32_e32 v170, 13, v165',
    'v_lshl_or_b32 v173, v171, 2, 0x8000', 'v_add_nc_u32_e32 v174, v169, v170', 's_mov_b32 s13, 0x14000',
    '.LBB21_7:                               ; =>This Inner Loop Header: Depth=1');
  const sreg = ['s20', 's19', 's20', 's13'];
  const next = ['s_add_co_i32 s19, s13, 0xffff8000', 's_add_co_i32 s20, s13, 0xffffc000', 's_nop 0', 's_add_co_i32 s13, s13, 0x10000'];
  for (let s = 0; s < 4; s++) {
    const stage = s % 2 ? 129 : 153;      // the set loaded at this slab's start
    const store = s % 2 ? 153 : 129;      // and the one stored at its end
    const buf = (s % 2) * 16384;
    if (s === 0) add('s_add_co_i32 s20, s13, 0xffff4000', 'v_lshl_add_u32 v175, s15, 2, v173');
    add('s_clause 0x1', `buffer_load_b128 v[${stage}:${stage + 3}], v161, s[4:7], ${sreg[s]} offen`,
      `buffer_load_b128 v[${stage + 4}:${stage + 7}], v163, s[4:7], ${sreg[s]} offen`, '; sched_barrier mask(0x00000000)');
    for (let t = 0; t < 8; t++) {
      if (t % 4 === 0) {
        for (let i = 0; i < 4; i++) add(`ds_load_b128 v[${176 + 4 * i}:${179 + 4 * i}], v174 offset:${buf + 1024 * (t + i)}`);
        add('; sched_barrier mask(0x00000000)');
      }
      add(t === 0 ? 's_wait_loadcnt_dscnt 0x503' : `s_wait_dscnt 0x${3 - (t % 4)}`);
      for (let j = 0; j < 4; j++) {
        for (let n = 0; n < 4; n++) {
          const acc = 1 + 16 * t + 4 * n;
          if (j && n === 3) add('s_delay_alu instid0(VALU_DEP_4)');
          add(`v_wmma_i32_16x16x16_iu8 v[${acc}:${acc + 3}], v${176 + 4 * (t % 4) + j}, v${137 + 4 * n + j}, v[${acc}:${acc + 3}] neg_lo:[1,1,0]`);
          if (t === 0 && j === 0 && n < 3) add(`s_wait_loadcnt 0x${4 - n}`);
        }
      }
      add('; sched_barrier mask(0x00000000)');
    }
    add('s_clause 0x3');
    [164, 166, 167, 168].forEach((a, n) => add(`buffer_load_b128 v[${137 + 4 * n}:${140 + 4 * n}], v${a}, s[8:11], ${sreg[s]} offen`));
    if (s % 2) {
      add('v_xor_b32_e32 v176, v126, v113', 'v_xor_b32_e32 v177, v127, v114', 'v_xor_b32_e32 v178, v128, v115',
        'v_xor3_b32 v179, v121, v116, v118', 'v_xor3_b32 v180, v122, v125, v119', 'v_xor3_b32 v176, v176, v123, v120',
        's_delay_alu instid0(VALU_DEP_1)', 'v_xor3_b32 v176, v177, v176, v178', 'v_xor3_b32 v176, v176, v179, v180',
        'v_alignbit_b32 v176, v176, v176, s15', 'ds_xor_b32 v175, v176');
    }
    add(next[s]);
    if (s === 3) add('s_cmp_ge_u32 s16, s3');
    add(`ds_store_b128 v162, v[${store}:${store + 3}] offset:${16384 - buf}`, `ds_store_b128 v162, v[${store + 4}:${store + 7}] offset:${24576 - buf}`,
      's_wait_dscnt 0x0', 's_barrier_signal -1', 's_barrier_wait -1');
  }
  add('s_cbranch_scc0 .LBB21_7', '; %bb.8:', 'v_or_b32_e32 v172, 0x8000, v172', 's_endpgm', '.Lfunc_end21:', '');
  return L.join('\n').replace(/ offset:0$/gm, '');
}

const ALIAS = { 166: [164, 0x400], 167: [164, 0x800], 168: [164, 0xc00], 163: [161, 0x2000] };
// What a loop computes over three passes: the reads handfold put in front of it, then its body.
function meaning(text) {
  const lines = text.split('\n');
  const f = findFunction(lines, 'pearl_amd_fold_wmma12');
  const loop = findLoops(lines, f[0], f[1]).reduce((a, b) => (b.wmma > a.wmma ? b : a));
  const before = [];
  for (let i = loop.head - 1; opOf(lines[i]) === 'ds_load_b128'; i--) before.unshift(lines[i]);
  return { lines, loop, said: simulate({ before, body: lines.slice(loop.head + 1, loop.tail + 1), passes: 3, alias: ALIAS }) };
}

describe('handfold.js', () => {
  let tmp;
  let src;
  beforeAll(() => {
    tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'amd-hand-'));
    src = path.join(tmp, 'in.s');
    fs.writeFileSync(src, fixture());
  });
  afterAll(() => { fs.rmSync(tmp, { recursive: true, force: true }); });
  const edit = (ops, input = src) => {
    const out = path.join(tmp, `out-${ops.replace(/\W/g, '_')}.s`);
    fs.rmSync(out, { force: true });
    const r = spawnSync(process.execPath, [path.join(HAND, 'handfold.js'), input, out, ops], { encoding: 'utf8' });
    return { status: r.status, stdout: r.stdout, stderr: r.stderr, text: fs.existsSync(out) ? fs.readFileSync(out, 'utf8') : null };
  };

  it('finds the chunk loop and gives the assembly back unchanged for rt', () => {
    const ref = meaning(fixture());
    expect(ref.loop.wmma).toBe(512);
    expect(ref.said.map((p) => [p.wmma.length, p.stores.length])).toEqual([[512, 8], [512, 8], [512, 8]]);
    const r = edit('rt');
    expect([r.status, r.stdout]).toEqual([0, 'handfold: pearl_amd_fold_wmma12 loop of 512 WMMAs, 766 instructions -> 766 (rt)\n']);
    expect(r.text).toBe(fixture());
  });

  it.each(['rs', 'nodly', 'dly1', 'roll', 'se', 'rollse', 'xb', 'xbp', 'xbp2', 'g8', 'g8m', 'g8e', 'g8+nodly'])(
    '%s adds the same bytes to the same sums, after their reads arrived, and stages the same bytes', (ops) => {
      const r = edit(ops);
      expect([ops, r.status, r.stderr]).toEqual([ops, 0, '']);
      expect(meaning(r.text).said).toEqual(meaning(fixture()).said);
      expect(r.text.split('\n').filter((l) => opOf(l).startsWith('v_wmma')).length).toBe(512);
    });

  it.each(['uns', 'noro', 'nob', 'noa', 'nost', 'nobar'])('%s is timing-only: it changes the loop', (ops) => {
    const r = edit(ops);
    expect(r.status).toBe(0);
    expect(r.text).not.toBe(fixture());
    let same = true;
    try { same = JSON.stringify(meaning(r.text).said) === JSON.stringify(meaning(fixture()).said); } catch (e) { same = false; }
    // the model follows operand bytes, so it can't see a missing XOR, barrier or sign: those three compare equal
    expect(same).toBe(['uns', 'noro', 'nobar'].includes(ops));
  });

  it('g8 leaves one break a slab, takes its registers from the loop and gives them back after it', () => {
    const { lines, loop } = meaning(edit('g8e').text);
    const body = lines.slice(loop.head + 1, loop.tail + 1);
    // between a slab's first and last WMMA nothing but WMMAs, waits and delay hints
    const isW = body.map((l) => opOf(l).startsWith('v_wmma'));
    let run = 0;
    const runs = [];
    body.forEach((l, i) => {
      if (isW[i]) run++;
      else if (opOf(l) && !/^s_(wait|delay)/.test(opOf(l)) && run) { runs.push(run); run = 0; }
    });
    expect(runs).toEqual([128, 128, 128, 128]);
    // all 8 tiles' reads in front of the loop for the first pass, tiles 4-7 in the freed registers
    expect(lines.slice(loop.head - 8, loop.head).map((l) => /v\[(\d+):/.exec(l)[1])).toEqual(['176', '180', '184', '188', '153', '157', '165', '169']);
    // one staging set, and B' through one address register
    expect(body.filter((l) => /ds_store_b128/.test(l)).every((l) => /v\[(129|133):/.test(l))).toBe(true);
    expect(body.filter((l) => /s\[8:11\]/.test(l)).map((l) => l.trim().replace(/^.*, v164, s\[8:11\], \S+ offen ?/, '')).slice(0, 4))
      .toEqual(['', 'offset:1024', 'offset:2048', 'offset:3072']);
    expect(body.some((l) => /\bv16[678]\b/.test(l.replace(/v\[\d+:\d+\]/g, '')) && !opOf(l).startsWith('v_'))).toBe(false);
    // after the loop: the reads issued for a slab it won't run are waited for, then the registers are made again
    expect(lines.slice(loop.tail + 1, loop.tail + 10).map((l) => l.trim())).toEqual([
      's_wait_dscnt 0x0', 'v_lshrrev_b32_e32 v165, 8, v0', 'v_or_b32_e32 v166, 0x400, v164', 'v_or_b32_e32 v167, 0x800, v164',
      'v_or_b32_e32 v168, 0xc00, v164', 'v_lshlrev_b32_e32 v170, 13, v165', 'v_and_b32_e32 v171, 0x1fe, v0',
      'v_lshlrev_b32_e32 v172, 2, v0', 'v_sub_nc_u32_e32 v169, v174, v170',
    ]);
  });

  it('writes nothing and exits 3 when the loop is not the one it knows', () => {
    const bad = (name, change) => {
      const p = path.join(tmp, name + '.s');
      fs.writeFileSync(p, change(fixture()));
      return p;
    };
    const cases = [
      ['rt', bad('nofn', (t) => t.replace('pearl_amd_fold_wmma12:', 'other:')), /no function/],
      ['roll', bad('noread', (t) => t.replace(/\tds_load_b128 v\[188:191\], v174 offset:3072\n/, '')), /7 LDS reads/],
      ['g8', bad('addr', (t) => t.replace('v_or_b32_e32 v166, 0x400, v164', 'v_add_nc_u32_e32 v166, v164, v9')), /is not v164 OR a constant/],
      ['g8', bad('used', (t) => t.replace('\ts_cmp_ge_u32 s16, s3\n', '\ts_cmp_ge_u32 s16, s3\n\tv_mov_b32_e32 v170, 0\n')), /no two free register quads/],
      ['bogus', src, /unknown op/],
    ];
    for (const [ops, input, why] of cases) {
      const r = edit(ops, input);
      expect([ops, r.status, r.text, r.stdout]).toEqual([ops, 3, null, '']);
      expect(r.stderr).toMatch(why);
    }
    expect(Object.keys(OPS)).toEqual(expect.arrayContaining(['rt', 'g8', 'g8m', 'g8e', 'xbp', 'noro']));
  });
});

describe('build-amd.sh and the hand route', () => {
  let tmp;
  beforeEach(() => { tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'amd-handco-')); });
  afterEach(() => { fs.rmSync(tmp, { recursive: true, force: true }); });
  const shFn = (name) => {
    const at = BUILD.indexOf('\n' + name + '() {\n');
    return BUILD.slice(at + 1, BUILD.indexOf('\n}\n', at) + 3);
  };
  // hipc with stand-ins: clang++ logs its arguments and, asked for assembly, writes the stand-in loop; lld logs.
  function run(archs, flags) {
    fs.mkdirSync(path.join(tmp, 'bin'));
    fs.writeFileSync(path.join(tmp, 'fold.s'), fixture());
    fs.writeFileSync(path.join(tmp, 'bin', 'clang++'), '#!/bin/bash\necho "clang++ $*" >> calls.txt\n'
      + 'case " $* " in *" -S "*) cp fold.s "${@: -1}" ;; esac\n', { mode: 0o755 });
    for (const tool of ['clang-offload-bundler', 'lld']) {
      fs.writeFileSync(path.join(tmp, 'bin', tool), '#!/bin/bash\necho "' + tool + ' $*" >> calls.txt\n', { mode: 0o755 });
    }
    const here = HAND.replace(/\\/g, '/').replace(/\/hand$/, '');
    const r = spawnSync('bash', ['-c', 'set -euo pipefail\narchs="' + archs + '"\ncc=bin/clang++\nrocm=/opt/rocm\nHERE="' + here + '"\n'
      + shFn('offload_flags') + shFn('arch_flags') + shFn('wave64_of') + shFn('hand_of') + shFn('hand_co') + shFn('hipc')
      + 'hipc src/a.cpp out/a.o -O3 ${PEARL_AMD_CFLAGS:-}\n'], { cwd: tmp, env: { PATH: process.env.PATH, PEARL_AMD_CFLAGS: flags }, encoding: 'utf8' });
    const calls = fs.readFileSync(path.join(tmp, 'calls.txt'), 'utf8').split('\n').filter(Boolean);
    return { status: r.status, stdout: r.stdout, stderr: r.stderr, calls };
  }

  bashIt('builds gfx1201 by way of its edited assembly when PEARL_AMD_HAND names the edits, and only then', () => {
    const r = run('gfx1100 gfx1201', '-DPEARL_AMD_HAND=g8e');
    expect([r.status, r.stderr]).toEqual([0, '']);
    expect(r.stdout).toBe('handfold: pearl_amd_fold_wmma12 loop of 512 WMMAs, 766 instructions -> 770 (g8e)\n');
    const co = 'out/a.o.targets/gfx1201.co';
    expect(r.calls.slice(0, 4)).toEqual([
      'clang++ -x hip --offload-arch=gfx1100 --rocm-path=/opt/rocm -O3 -DPEARL_AMD_HAND=g8e --cuda-device-only --no-gpu-bundle-output '
      + '-c src/a.cpp -o out/a.o.targets/gfx1100.co',
      'clang++ -x hip --offload-arch=gfx1201 --rocm-path=/opt/rocm -O3 -DPEARL_AMD_HAND=g8e -mwavefrontsize64 -DPEARL_AMD_WAVE64=1 '
      + `--cuda-device-only -S -c src/a.cpp -o ${co}.s`,
      `clang++ -x assembler --target=amdgcn-amd-amdhsa -mcpu=gfx1201 -mwavefrontsize64 -c ${co}.hand.s -o ${co}.o`,
      `lld -flavor gnu -m elf64_amdgpu --no-undefined -shared -o ${co} ${co}.o`,
    ]);
    expect(meaning(fs.readFileSync(path.join(tmp, co + '.hand.s'), 'utf8')).said).toEqual(meaning(fixture()).said);
    expect(r.calls).toHaveLength(6);   // then the bundler and the host pass, as without the edit
  });

  bashIt('builds the compiler\'s own assembly when the edit does not apply', () => {
    const r = run('gfx1201', '-DPEARL_AMD_HAND=bogus');
    expect(r.status).toBe(0);
    expect(r.stdout).toBe('hand: gfx1201 keeps the compiler\'s loop\n');
    expect(r.stderr).toBe('handfold: unknown op bogus\n');
    expect(fs.readFileSync(path.join(tmp, 'out/a.o.targets/gfx1201.co.hand.s'), 'utf8')).toBe(fixture());
  });

  bashIt('takes no edit for another target, a wave32 pass or a build without PEARL_AMD_HAND', () => {
    const hand = (arch, flags) => spawnSync('bash', ['-c', shFn('arch_flags') + shFn('wave64_of') + shFn('hand_of') + 'hand_of ' + arch],
      { env: { PATH: process.env.PATH, PEARL_AMD_CFLAGS: flags }, encoding: 'utf8' }).stdout;
    expect(hand('gfx1201', '-DPEARL_AMD_HAND=g8')).toBe('g8\n');
    expect(hand('gfx1201', '-O1 -DPEARL_AMD_HAND=g8+nodly -DX=1')).toBe('g8+nodly\n');
    // a wave64 fold built by flags (the 512x256 block) takes edits too
    expect(hand('gfx1201', '-mwavefrontsize64 -DPEARL_AMD_W64=1 -DPEARL_AMD_W12_WR=8 -DPEARL_AMD_HAND=nob')).toBe('nob\n');
    expect(hand('gfx1201', '')).toBe('');
    expect(hand('gfx1200', '-DPEARL_AMD_HAND=g8')).toBe('');
    expect(hand('gfx1100', '-DPEARL_AMD_HAND=g8')).toBe('');
    expect(hand('gfx1201', '-DPEARL_AMD_W64K=0 -DPEARL_AMD_HAND=g8')).toBe('');
    // the release builds with no PEARL_AMD_CFLAGS, so it never takes the route; winlib has no hipc at all
    expect(BUILD.slice(BUILD.indexOf('\n  winlib)\n'), BUILD.indexOf('\n  check)\n'))).not.toMatch(/hand_|hipc/);
  });
});
