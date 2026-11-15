'use strict';

// The rotation fold as something a release carries (PEARL_FOLD_ROT=1, on by default since 0.7.0): how its
// cubin is built and refused, where it rides in each package, where the host looks for it, and what the host
// says when a rig asks for the fold and cannot have it. These read the sources: the runners have no GPU, and
// the loop in that file is written by hand, so the checks around it are pinned here.
// pearlMinerFoldCheck.test.js has the miner's own check of the fold; cliArgs.test.js the flag.

const fs = require('fs');
const path = require('path');

const EARN = path.join(__dirname, '..');
const read = (...p) => fs.readFileSync(path.join(EARN, ...p), 'utf8').replace(/\r\n/g, '\n');
const NATIVE_CORE = read('..', '.github', 'workflows', 'native-core.yml');
const MINER_BUILD = read('..', '.github', 'workflows', 'miner-build.yml');
const BUILD_ROT = read('native', 'probes', 'hand89', 'build_rot.py');
const RFOLD = read('native', 'probes', 'hand89', 'rfold.py');
const ASM = read('native', 'probes', 'hand89', 'asm89.py');
const GFOLD = read('native', 'probes', 'hand89', 'gfold.py');
const REF = read('native', 'src', 'pearl_rot_sm89.ref');
const REF86 = read('native', 'src', 'pearl_rot_sm86.ref');
const REF120 = read('native', 'src', 'pearl_rot_sm120.ref');
const REF120W = read('native', 'src', 'pearl_rot_sm120w.ref');
const refLines = (text) => text.split('\n').filter((l) => l.trim() && !l.startsWith('#'));
const HOST = read('native', 'src', 'pearl_host.cu');
const ROT = read('native', 'src', 'pearl_host_rot.inc');
const ROT_PATH = read('native', 'src', 'pearl_rot_path.h');
const PATH_CHECK = read('native', 'probes', 'rot-path-check.cc');

// A workflow step, from its "- name:" line to the next step.
function step(workflow, name) {
  const at = workflow.indexOf('      - name: ' + name);
  expect([name, at >= 0]).toEqual([name, true]);
  const next = workflow.indexOf('\n      - ', at + 1);
  return workflow.slice(at, next < 0 ? undefined : next);
}

describe('the cubin is only written when every check passes (build_rot.py)', () => {
  test('the steps run in order, and the file gets its name last', () => {
    const order = [
      '"-arch=" + arch.rstrip("w"), "-std=c++17", "-O3", "-cubin"',
      'if asm89.check(base, sass):',
      'os.path.join(HERE, "rfold.py"), base, sass, nvdisasm, tmp]',
      'have = asm89.fold_hash(tmp)',
      'if have not in known:',
      'os.replace(tmp, unverified)',
      'os.replace(tmp, final)',
    ].map((text) => [text, BUILD_ROT.indexOf(text)]);
    for (const [text, at] of order) expect([text, at > 0]).toEqual([text, true]);
    expect(order.map((o) => o[1])).toEqual(order.map((o) => o[1]).slice().sort((a, b) => a - b));
    // Nothing else writes the final name, and a build starts by removing the last one.
    expect(BUILD_ROT.split('os.replace(tmp, final)')).toHaveLength(2);
    expect(BUILD_ROT).toContain('for f in (final, tmp, unverified):\n        if os.path.exists(f):\n            os.remove(f)');
    // Every failure is exit 1.
    expect(BUILD_ROT).toContain('def fail(msg):\n    print("build_rot: FAILED: " + msg)\n    return 1');
  });

  test('--arch sm_86 builds the RTX 30 file against its own reference', () => {
    expect(BUILD_ROT).toContain('ARCHS = ("sm_89", "sm_86", "sm_120", "sm_120w")');
    expect(BUILD_ROT).toContain('NAME = "pearl_rot_%s.cubin" % arch.replace("_", "")');
    // Both take rpr=2 and u8=1 (measured on a 4090 and a 3090); RTX 30's also iy=0 (measured on a 3090); RTX 40's
    // and RTX 30's also rgf=1 rb0=3 rb1=0 (measured on a 4090 and a 3090); RTX 40's also sr=1, records the eight
    // warps share and B drawn once with its exchange keys and readout window (measured on a 4090); RTX 30's also
    // sr=1 (measured on a 3060 and a 3090).
    expect(BUILD_ROT).toContain('ARCH_KEYS = {"sm_89": ["rpr=2", "u8=1", "rgf=1", "rb0=6", "rb1=3", "bsym=4", "bsg=8", "bsg3=24", "bss=2", "bord=2", "bx=1",\n'
      + '                       "barv=1", "bxt=1", "sr=1"], "sm_86": ["iy=0", "rpr=2", "u8=1", "rgf=1", "rb0=3", "rb1=0",\n'
      + '                                                                                  "sr=1"],');
    // RTX 50's: sr=1 with srp=1 (the prologue's pointer step behind its stall; without it sm_120 folds one band
    // wrong). Measured on a 5090 and a 5080; sr without srp must never ship there.
    expect(BUILD_ROT).toContain('"dpb=7", "dpm=1", "dpg=3", "sr=1", "srp=1"],\n');
    // The RTX 5090's file: the same keys on the 96 x 512 block, its stage ordered by the two groups' barriers, in a
    // kernel built with eight barriers (ROT_C 49155) and the 96 x 512 tile. Measured on 575 W 5090s.
    expect(BUILD_ROT).toContain('"sm_120w": ["rpr=2", "u8=1", "rgf=1", "rb0=3", "rb1=0", "ist=7", "iy=0", "b8=1", "dpb=7", "dpm=1",\n'
      + '                         "dpg=3", "sr=1", "srp=1", "sr_ub=34", "w8=1", "w8g=4", "trl=1", "cq=0"]}');
    expect(BUILD_ROT).toContain('"-DROT_C=%d" % (49155 if "w8g=4" in ARCH_KEYS[arch] else 16387)');
    expect(BUILD_ROT).toContain('if "w8=1" in ARCH_KEYS[arch]:\n        defs.append("-DROT_W8=1")');
    // sr's loop reads the shared stream and bx's keeps the transcript in the frame, so the kernel is built for both.
    expect(BUILD_ROT).toContain('if "sr=1" in ARCH_KEYS[arch]:\n        defs.append("-DROT_SR=1")');
    expect(BUILD_ROT).toContain('if "bx=1" in ARCH_KEYS[arch]:\n        defs.append("-DROT_BX=1")');
    expect(BUILD_ROT).toContain('+ ARCH_KEYS[arch])');
    expect(BUILD_ROT).toContain('ref = os.path.join(SRC, "pearl_rot_%s.ref" % arch.replace("_", ""))');
    // The reference's --ref still wins over the one --arch picks.
    expect(BUILD_ROT.indexOf('if "--ref" in args:')).toBeGreaterThan(BUILD_ROT.indexOf('if "--arch" in args:'));
  });

  test('the hand loop is compared with what the compiler made, and not written when they differ', () => {
    const expectFn = RFOLD.slice(RFOLD.indexOf('def expect(text, cfg, code):'), RFOLD.indexOf('def main():'));
    for (const said of [
      'the loop writes registers the compiled function does not',
      'predicates and the loop needs two',
      'the loop loads the address base into UR%d and the compiled function into %s',
      'the loop reads kernel parameters the compiled function does not',
      'the kernel does not zero its transcripts at shared address 0x%x',
      'the loop is %d instructions and the compiled function has room for %d',
    ]) expect(expectFn).toContain(said);
    const main = RFOLD.slice(RFOLD.indexOf('def main():'));
    // Latency first, then the comparison, each ending the build before asm89.patch writes anything.
    const at = (text) => main.indexOf(text);
    expect(at('print("%d latency problems: not written" % bad)')).toBeGreaterThan(0);
    expect(at('wrong = expect(text, cfg, code)')).toBeGreaterThan(at('latency problems: not written'));
    expect(at('print("the compiled kernel is not what the hand loop expects: not written")')).toBeGreaterThan(at('wrong = expect('));
    expect(at('asm89.patch(cubin, text, out, code, nvd)')).toBeGreaterThan(at('hand loop expects: not written'));
    // The barrier count is read back from the patched file.
    expect(main).toContain('have = asm89.barriers(asm89.disassemble(out, nvd))\n        if have != 3:\n            os.remove(out)');
    // The read-back of every instruction is asm89.patch's own.
    expect(ASM).toContain('raise RuntimeError("%d instructions did not come back as asked" % bad)');
  });

  test('the reference lists a hash for each toolkit that has mined, and nothing else', () => {
    const lines = refLines(REF);
    // RTX 30's: the toolkit CI builds with, run on an RTX 30 card.
    expect(refLines(REF86).map((l) => l.split(' ')[0])).toContain('12.8');
    for (const l of refLines(REF86)) expect(l).toMatch(/^\d+\.\d+ [0-9a-f]{64}$/);
    // RTX 50's: CUDA 13.3, the toolkit of the core RTX 50 rigs load, run on an RTX 5090.
    expect(refLines(REF120).map((l) => l.split(' ')[0])).toEqual(['13.3']);
    for (const l of refLines(REF120)) expect(l).toMatch(/^\d+\.\d+ [0-9a-f]{64}$/);
    // The RTX 5090's: the same toolkit, run on an RTX 5090.
    expect(refLines(REF120W).map((l) => l.split(' ')[0])).toEqual(['13.3']);
    for (const l of refLines(REF120W)) expect(l).toMatch(/^\d+\.\d+ [0-9a-f]{64}$/);
    expect(lines.length).toBeGreaterThan(0);
    for (const l of lines) expect(l).toMatch(/^\d+\.\d+ [0-9a-f]{64}$/);
    // CUDA 12.8 is what CI builds with; 12.6 is the toolkit the fold was first built and raced with.
    const kits = lines.map((l) => l.split(' ')[0]);
    expect(kits).toEqual(expect.arrayContaining(['12.6', '12.8']));
    // The hash leaves out the string table, which holds a hash of the source file's path.
    expect(ASM).toContain('off, size = sections(b)[".strtab"]\n    b[off:off + size] = bytes(size)\n    return hashlib.sha256(bytes(b)).hexdigest()');
  });
});

// The offsets records are a contract between three places: the kernel that writes them (rot_aux_offs, in the
// cubin), the hand loop that reads them (rfold.py's co, in the same cubin) and the host that sizes the table
// (in the addon). The hash list is what keeps a cubin of one form out of a core of the other.
describe('the offsets records are 8 bytes in the kernel, the loop and the host alike', () => {
  const KERNEL = read('native', 'src', 'pearl_rot.cu');

  test('rot_aux_offs writes two words a lane a k-step', () => {
    const offs = KERNEL.slice(KERNEL.indexOf('extern "C" __global__ void rot_aux_offs('));
    expect(offs).toContain('const size_t warpWords = (size_t)(ksteps + 1u) * 32u * 2u;');
    expect(offs).toContain('uint32_t *o = offs + w * warpWords + ((size_t)(ch * 4u + turn) * 32u + l) * 2u;');
    // A first turn's record is A's two offsets and B's two; a later turn's has the stage address whole.
    expect(offs).toContain('o[0] = turn && ks < ksteps ? ROT_STG + band * (2048u * ROT_MT) + s * (512u * ROT_MT) + 16u * l\n'
      + '                               : a0 | (a1 << 16);\n    o[1] = b0 | (b1 << 16);');
    // Two offsets fit a word only while every table is under 65536 in shared memory: A's and B's are.
    const num = (name) => Number(new RegExp('#define ' + name + ' \\(?(\\d+)u').exec(KERNEL)[1]);
    expect(num('ROT_MT')).toBe(6);
    expect(2048 * (2 * num('ROT_MT') + 16) + 2048).toBeLessThan(65536);
  });

  test('with ROT_SR, rot_aux_offs writes one stream the eight warps share, inside the same table', () => {
    const offs = KERNEL.slice(KERNEL.indexOf('extern "C" __global__ void rot_aux_offs('));
    const sr = offs.slice(offs.indexOf('#if ROT_SR'), offs.indexOf('#endif'));
    // A k-step's record is lane l's two A offsets and two B offsets, with no warp's base in them.
    expect(sr).toContain('offs[((size_t)ks * 32u + l) * 2u] = 16u * pa[0] | (16u * pa[1]) << 16;');
    expect(sr).toContain('offs[((size_t)ks * 32u + l) * 2u + 1u] = 16u * pb[0] | (16u * pb[1]) << 16;');
    // The loop reads up to three k-steps past the last chunk: the extra k-step's thread writes four of zeros.
    expect(sr).toContain('for (uint32_t e = 0; e < 4u; e++) {');
    // (ksteps + 4) records of 64 words fit the per-warp table of 8 x (ksteps + 1) x 64 words the host allocates.
    expect(sr).toContain('  return;');
    // The loop keeps the bases in uniform registers and moves its pointer by a uniform step a turn.
    expect(RFOLD).toContain('sr=0, sr_ua=16, sr_ub=17,');
    expect(RFOLD).toContain('put(b + cfg["lgg"], ldg(nxt, R["poff"], 0, size=64, wsb=SB_OFF, rsb=hold, wait=hw))');
    expect(GFOLD).toContain('ur = "+UR%d" % rfold_sr_u(cfg, 0 if d1 < BF else 1) if cfg.get("sr") else ""');
    expect(GFOLD).toContain('return cfg["sr_ua"] if i == 0 else cfg["sr_ub"]');
  });

  test('with ROT_BX the transcript lives in the frame, and the loop\'s store of it holds its read scoreboard', () => {
    // B drawn once: the 16 KB of transcripts become the exchange, the transcript moves to the kernel's frame.
    expect(KERNEL).toContain('__align__(16) uint32_t trl[16];');
    expect(KERNEL).toContain('pearl_rot_fold_loop(offs, chunks, trl);');
    expect(KERNEL).toContain('for (uint32_t q = 0; q < 16u; q++) m[q] = trl[q];');
    // Without the read scoreboard the store could read R3 after the function's return wrote it (local address 0).
    expect(RFOLD).toContain('put(end + 6, Ins("STL [R%d], R%d" % (x3, w), rsb=hold)');
  });

  test('the hand loop loads 8 bytes a k-step by default and parts them', () => {
    // The schedule keys the reference hashes were built with (probes/README.md has what each is worth).
    expect(RFOLD).toMatch(/\n {11}al=1, bl=1, bp=1, cq=35, ae=1, sbo=7, bt=15,\n/);
    expect(RFOLD).toMatch(/\n {11}co=1,\n/);
    expect(RFOLD).toContain('rec = 0x100 if cfg["co"] else 0x200');
    expect(RFOLD).toContain('size=64 if cfg["co"] else 128');
    expect(ASM).toContain('(r"LDG\\.E\\.64 " + R + ", " + MEM64, lambda m: (_mem(0x7981 | _reg(m[1]) << 16 | 6 << 32, m, 2), 0x0c1e1b00)),');
  });

  test('the host allocates a table of that size', () => {
    expect(ROT).toContain('r->offsWords = (size_t)8 * (r->ksteps + 1u) * 64u;');
    expect(ROT).toContain('const size_t offs = (size_t)3 * 8 * (p->k / 32 + 1) * 256;');
  });
});

describe('CI builds it with the release toolkit (native-core.yml)', () => {
  const build = step(NATIVE_CORE, 'Build the rotation fold\'s cubin (RTX 40 and 30, opt-in, not blocking)');

  test('in the two CUDA 12.8 jobs, after the addon, without blocking them', () => {
    expect(build).toContain("if: matrix.variant == ''");
    expect(build).toContain('continue-on-error: true');
    expect(build).toContain('"$py" -X utf8 probes/hand89/build_rot.py nvcc nvdisasm rot-build || rc=$?');
    expect(build).toContain('if [ "$rc" != 0 ] || [ ! -f rot-build/pearl_rot_sm89.cubin ]; then');
    expect(build).toContain('echo "::error::$msg"');
    expect(build).toContain('cp rot-build/pearl_rot_sm89.cubin build/Release/pearl_rot_sm89.cubin');
    // RTX 30's file, from its own folder; either one failing fails the step, and the other still ships.
    expect(build).toContain('"$py" -X utf8 probes/hand89/build_rot.py nvcc nvdisasm rot-build/sm86 --arch sm_86 || rc=$?');
    expect(build).toContain('cp rot-build/sm86/pearl_rot_sm86.cubin build/Release/pearl_rot_sm86.cubin');
    expect(build).toContain('exit "$bad"');
    // node-gyp rebuild empties build/, so the cubin is built after it.
    expect(NATIVE_CORE.indexOf('- name: Build the rotation fold\'s cubin')).toBeGreaterThan(NATIVE_CORE.indexOf('- name: Confirm the addon was produced'));
    // Windows needs nvdisasm installed; NVIDIA's Linux image has it, or the step installs it.
    expect(NATIVE_CORE).toContain("subpackages: '[\"nvcc\", \"cudart\", \"nvrtc\", \"cuobjdump\", \"nvdisasm\"]'");
    expect(build).toContain('dnf install -y --setopt=install_weak_deps=False cuda-nvdisasm-12-8');
  });

  test('the cubin rides in the core\'s artifact, and what the build made is kept either way', () => {
    expect(NATIVE_CORE).toContain('          name: pearl-core${{ matrix.variant }}-${{ matrix.os }}\n'
      + '          path: |\n'
      + '            earn/native/build/Release/${{ matrix.file }}\n'
      + '            earn/native/build/Release/pearl_rot_sm89.cubin\n'
      + '            earn/native/build/Release/pearl_rot_sm86.cubin\n'
      + '            earn/native/build/Release/pearl_rot_sm120.cubin\n'
      + '            earn/native/build/Release/pearl_rot_sm120w.cubin\n'
      + '          if-no-files-found: error\n');
    // The RTX 50 file is built by the CUDA 13 job, with the same refusal to write an unlisted file.
    expect(NATIVE_CORE).toContain("- name: Build the rotation fold's cubin for RTX 50 (opt-in, not blocking)\n"
      + '        id: rot120\n'
      + "        if: matrix.variant == '-cu13'\n"
      + '        continue-on-error: true\n');
    expect(NATIVE_CORE).toContain('"$py" -X utf8 probes/hand89/build_rot.py nvcc nvdisasm rot-build/sm120 --arch sm_120 || rc=$?');
    expect(NATIVE_CORE).toContain('cp rot-build/sm120/pearl_rot_sm120.cubin build/Release/pearl_rot_sm120.cubin');
    // The RTX 5090's file, by the same job and the same refusal; a 5090 without it mines on the RTX 50 one.
    const w = step(NATIVE_CORE, 'Build the rotation fold\'s cubin for RTX 5090 (not blocking)');
    expect(w).toContain("if: matrix.variant == '-cu13'\n        continue-on-error: true\n");
    expect(w).toContain('"$py" -X utf8 probes/hand89/build_rot.py nvcc nvdisasm rot-build/sm120w --arch sm_120w || rc=$?');
    expect(w).toContain('cp rot-build/sm120w/pearl_rot_sm120w.cubin build/Release/pearl_rot_sm120w.cubin');
    expect(NATIVE_CORE).toContain("        if: always() && matrix.variant == ''\n        with:\n"
      + '          name: pearl-rot-build-${{ matrix.os }}\n          path: earn/native/rot-build\n');
  });

  test('the Linux lookup of the cubin is run for real, with a stand-in addon', () => {
    const check = step(NATIVE_CORE, 'Check where the rotation fold\'s cubin is looked for (Linux, not blocking)');
    expect(check).toContain("if: runner.os == 'Linux' && matrix.variant == ''");
    expect(check).toContain('continue-on-error: true');
    expect(check).toContain('-fPIC -shared -DROT_PATH_ADDON -I src probes/rot-path-check.cc -o "$d/pearl_core.node"');
    expect(check).toContain('if ! "$d/rot-path-check" "$d"; then');
    // The cases it runs.
    for (const said of [
      'full path: found beside the addon',
      'relative path, folder changed: found',
      'symbolic link, cubin beside the real file: found',
      'symbolic link, cubin beside the link: that one',
      'PEARL_ROT_CUBIN: the named file',
      'PEARL_ROT_CUBIN names no file: not found, not the one beside the addon',
    ]) expect(PATH_CHECK).toContain('"' + said + '"');
  });
});

describe('every package carries it beside the core', () => {
  test('the installer and the AppImage, in resources/native', () => {
    const pkg = JSON.parse(read('package.json'));
    const native = Object.values(pkg.build)
      .flatMap((v) => (v && Array.isArray(v.extraResources) ? v.extraResources : []))
      .filter((x) => x.to === 'native');
    expect(native).toHaveLength(2);
    for (const r of native) {
      expect(r.filter).toEqual(expect.arrayContaining(['pearl_rot_sm89.cubin', 'pearl_rot_sm86.cubin', 'pearl_rot_sm120.cubin',
        'pearl_rot_sm120w.cubin']));
    }
  });

  test('the Linux CLI and the HiveOS package, beside the binary', () => {
    const cli = read('scripts', 'build-cli.mjs');
    expect(cli).toContain("const rot = join(root, 'vendor', 'native', 'pearl_rot_sm89.cubin');");
    // A cubin left in dist/ by an earlier build must not ship beside a core it was not built with.
    expect(cli).toContain("rmSync(join(dist, 'pearl_rot_sm89.cubin'), { force: true });\nif (existsSync(rot)) {\n"
      + "  copyFileSync(rot, join(dist, 'pearl_rot_sm89.cubin'));");
    expect(cli).toContain('warning: no vendor/native/pearl_rot_sm89.cubin');
    const hive = read('scripts', 'build-hiveos.mjs');
    expect(hive).toContain("const rot = join(dist, 'pearl_rot_sm89.cubin');\nif (existsSync(rot)) {\n"
      + "  copyFileSync(rot, join(pkgDir, 'pearl_rot_sm89.cubin'));\n"
      + "  chmodSync(join(pkgDir, 'pearl_rot_sm89.cubin'), 0o644);");
    expect(hive).toContain('warning: no pearl_rot_sm89.cubin');
    expect(cli).toContain("copyFileSync(rot86, join(dist, 'pearl_rot_sm86.cubin'));");
    expect(hive).toContain("chmodSync(join(pkgDir, 'pearl_rot_sm86.cubin'), 0o644);");
    expect(cli).toContain("copyFileSync(rot120, join(dist, 'pearl_rot_sm120.cubin'));");
    expect(hive).toContain("chmodSync(join(pkgDir, 'pearl_rot_sm120.cubin'), 0o644);");
    expect(cli).toContain("rmSync(join(dist, 'pearl_rot_sm120w.cubin'), { force: true });\nif (existsSync(rot120w)) {\n"
      + "  copyFileSync(rot120w, join(dist, 'pearl_rot_sm120w.cubin'));");
    expect(hive).toContain("chmodSync(join(pkgDir, 'pearl_rot_sm120w.cubin'), 0o644);");
    const smoke = read('..', '.github', 'scripts', 'hiveos-smoke.sh');
    expect(smoke).toMatch(/for f in h-manifest\.conf pearl_core\.node pearl_core_cu13\.node pearl_rot_sm89\.cubin [^\n]*; do mode_is "\$f" 644; done/);
  });

  test('the installer build says when it has none, and a release publishes it with the cores', () => {
    expect(MINER_BUILD).toContain('if [ -f vendor/native/pearl_rot_sm89.cubin ]; then');
    expect(MINER_BUILD).toContain('msg="no pearl_rot_sm89.cubin staged');
    const publish = step(MINER_BUILD, 'Publish CLI + HiveOS package to release');
    const up = publish.indexOf('gh release upload "v$VERSION" dist/pearl_rot_sm89.cubin --clobber');
    expect(up).toBeGreaterThan(0);
    // Before the binary, as the cores are: a rig that updates mid-upload must not pair a new binary with an old fold.
    expect(up).toBeLessThan(publish.indexOf('gh release upload "v$VERSION" dist/llmjob-earn-cli-linux'));
    // A release without one drops a stale copy, since the self-updater removes a rig's copy then.
    expect(publish).toContain('gh release delete-asset "v$VERSION" pearl_rot_sm89.cubin --yes');
    const up86 = publish.indexOf('gh release upload "v$VERSION" dist/pearl_rot_sm86.cubin --clobber');
    expect(up86).toBeGreaterThan(0);
    expect(up86).toBeLessThan(publish.indexOf('gh release upload "v$VERSION" dist/llmjob-earn-cli-linux'));
    expect(publish).toContain('gh release delete-asset "v$VERSION" pearl_rot_sm86.cubin --yes');
    const up120 = publish.indexOf('gh release upload "v$VERSION" dist/pearl_rot_sm120.cubin --clobber');
    expect(up120).toBeGreaterThan(0);
    expect(up120).toBeLessThan(publish.indexOf('gh release upload "v$VERSION" dist/llmjob-earn-cli-linux'));
    expect(publish).toContain('gh release delete-asset "v$VERSION" pearl_rot_sm120.cubin --yes');
    expect(MINER_BUILD).toContain('if [ -f vendor/native/pearl_rot_sm120.cubin ]; then');
    const up120w = publish.indexOf('gh release upload "v$VERSION" dist/pearl_rot_sm120w.cubin --clobber');
    expect(up120w).toBeGreaterThan(0);
    expect(up120w).toBeLessThan(publish.indexOf('gh release upload "v$VERSION" dist/llmjob-earn-cli-linux'));
    expect(publish).toContain('gh release delete-asset "v$VERSION" pearl_rot_sm120w.cubin --yes');
    expect(MINER_BUILD).toContain('if [ -f vendor/native/pearl_rot_sm120w.cubin ]; then');
  });
});

describe('the host: only with PEARL_FOLD_ROT=1, and one plain reason when it cannot run', () => {
  test('only PEARL_FOLD_ROT=1 turns it on', () => {
    expect(ROT).toContain('bool rot_env() {\n  const char *v = getenv("PEARL_FOLD_ROT");\n  return v && v[0] == \'1\' && v[1] == 0;\n}');
    // Every use of the fold in the host is behind it, or behind a context it made.
    expect(HOST).toContain('if (profile && rot_env() && rot_arch_ok(device)) {');
    expect(ROT).toContain('return rot_env() && p->k == PEARL_ROT_K && rot_card_ok(p, device, nullptr, 0);');
    expect(HOST).toContain('if (ctx->rot) return rot_submit(ctx, slot, st, nonce_base, regions_out, err, err_len);');
  });

  // On by default, so an RTX 20 or GTX card, which the fold was never for, gets no note about it.
  test('a card the fold is not for says nothing about it', () => {
    const archOk = ROT.slice(ROT.indexOf('bool rot_arch_ok(int device) {'), ROT.indexOf('bool rot_wanted('));
    expect(archOk).toContain('return (major == 8 && (minor == 9 || minor == 6)) || (major == 12 && minor == 0);');
    // The same cards rot_card_ok takes.
    expect(ROT).toContain('if (!(major == 8 && (minor == 9 || minor == 6)) && !(major == 12 && minor == 0))');
  });

  test('the card\'s line says why the fold is not used', () => {
    const cardK = HOST.slice(HOST.indexOf('extern "C" uint32_t pearl_host_card_k('), HOST.indexOf('extern "C" void *pearl_host_create('));
    expect(cardK).toContain('if (rot_card_ok(profile, device, why, sizeof why)) {\n'
      + '      if (note && note_len) snprintf(note, note_len, "rotation fold (experimental)");\n'
      + '      return 65536u;\n    }');
    expect(cardK).toContain('"rotation fold asked for but not used: %s. Mining on the normal kernel", why);');
    // A note has to fit the binding's buffer (pearl_core.cc, char note[192]) with the longest reason in it.
    const cardOk = ROT.slice(ROT.indexOf('bool rot_card_ok('), ROT.indexOf('bool rot_wanted('));
    const reasons = [
      '"the card\'s compute capability could not be read"',
      '"it is for RTX 30, 40 and 50 cards, and this card is compute %lld.%lld"',
      '"it does not mine this profile"',
      '"the file PEARL_ROT_CUBIN names cannot be read"',
      '"its file, %s, is not beside the Pearl core"',
      '"its file, %s, is not a kernel file"',
      '"its file, %s, is not the one this core was built with"',
      '"it needs %lld MiB of card memory and %lld MiB are free"',
    ];
    for (const r of reasons) expect(cardOk).toContain(r);
    // A note has to fit the binding's buffer with the longest reason in it, numbers and all.
    expect(read('native', 'src', 'pearl_core.cc')).toContain('char note[192] = {0};');
    const longest = Math.max(...reasons.map((r) => r.replace('%s', 'pearl_rot_sm89.cubin')
      .replace(/%lld/g, '4294967296').length));
    expect('rotation fold asked for but not used: . Mining on the normal kernel'.length + longest).toBeLessThan(192);
  });

  // A cubin cut to 8 KB crashed the driver, and the whole miner, on rig01 (11 November).
  test('a file that is not a whole cubin, or not one this core was built with, is never handed to the driver', () => {
    expect(ROT).toContain('return rot_elf_ok(reinterpret_cast<const uint8_t *>(out->data()), out->size() - 1, nullptr, nullptr);');
    // Every section and segment has to lie inside the file.
    expect(ROT_PATH).toContain('if (shentsize != 64 || shnum == 0 || shoff > n || shnum * 64 > n - shoff || strndx >= shnum) return false;');
    expect(ROT_PATH).toContain('if (type != 8 && (off > n || size > n - off)) return false;');
    expect(ROT_PATH).toContain('if (off > n || filesz > n - off) return false;');
    const create = ROT.slice(ROT.indexOf('bool rot_create('));
    const guard = create.indexOf('!rot_read_file(path, &image) || !rot_listed(image)');
    expect(guard).toBeGreaterThan(0);
    expect(guard).toBeLessThan(create.indexOf('load(&r->mod, image.data())'));
    // The file beside the core is held to the reference's hashes; one PEARL_ROT_CUBIN names is a developer's own.
    const listed = [...ROT.slice(ROT.indexOf('const char *const kRotCubins[] = {'), ROT.indexOf('bool rot_named()'))
      .matchAll(/"([0-9a-f]{64})"/g)].map((m) => m[1]).sort();
    const ref = [...refLines(REF), ...refLines(REF86), ...refLines(REF120), ...refLines(REF120W)]
      .map((l) => l.split(' ')[1]).sort();
    expect(listed).toEqual(ref);
    expect(ROT).toContain('bool rot_listed(const std::vector<char> &image) {\n  if (rot_named()) return true;');
    // The host's hash is the build's: sha256 with the string table zeroed.
    expect(ROT_PATH).toContain('memset(b + off, 0, size);\n  uint8_t d[32];\n  rot_sha256(b, n, d);');
    // What the miner retries on (pearlMiner.js, ROT_FAIL_RE).
    expect(create).toContain('snprintf(err, err_len, "rotation fold: %s (%d)", msg, code);');
  });

  test('the cubin is looked for beside the module, by the loader\'s path and then the kernel\'s', () => {
    expect(ROT).toContain('return rot_cubin_path_for(reinterpret_cast<const void *>(&rot_env), buf, len, name);');
    // The card's own file: sm_86's on an RTX 30, sm_89's on an RTX 40, sm_120's on an RTX 50.
    expect(ROT).toContain('const char *rot_file_for(int major, int minor) {\n'
      + '  return major == 12 ? PEARL_ROT_FILE_SM120 : minor == 6 ? PEARL_ROT_FILE_SM86 : PEARL_ROT_FILE;\n}');
    expect(ROT_PATH).toContain('#define PEARL_ROT_FILE_SM86 "pearl_rot_sm86.cubin"');
    expect(ROT_PATH).toContain('#define PEARL_ROT_FILE_SM120 "pearl_rot_sm120.cubin"');
    // An RTX 5090 (compute 12.0, 128 SMs or more) takes its own file when it is there, listed, of the 96 x 512 tile,
    // and the profile's n allows that tile; any other card, or a 5090 without it, the card's own file.
    expect(ROT_PATH).toContain('#define PEARL_ROT_FILE_SM120W "pearl_rot_sm120w.cubin"');
    const pick = ROT.slice(ROT.indexOf('const char *rot_file_for_device('), ROT.indexOf('bool rot_card_ok('));
    expect(pick).toContain('if (major != 12 || rot_named() || rot_w8_off() || n % 512u != 0u) return rot_file_for(major, minor);');
    // PEARL_ROT_W8=0 keeps a 5090 on the RTX 50 file (clocks locked below the cap).
    expect(ROT).toContain('const char *v = getenv("PEARL_ROT_W8");\n  return v && v[0] == \'0\' && v[1] == 0;');
    expect(pick).toContain('if (sms >= 128 && rot_cubin_path(path, sizeof path, PEARL_ROT_FILE_SM120W) && rot_read_file(path, &image)\n'
      + '      && rot_listed(image) && rot_image_w8(image))\n    return PEARL_ROT_FILE_SM120W;\n  return PEARL_ROT_FILE_SM120;');
    expect(ROT).toContain('const char *file = rot_file_for_device(device, major, minor, p->n);');
    expect(ROT).toContain('const char *file = rot_file_for_device(ctx->device, major, minor, p.n);\n'
      + '  if (!rot_cubin_path(path, sizeof path, file) || !rot_read_file(path, &image) || !rot_listed(image))');
    const linux = ROT_PATH.slice(ROT_PATH.indexOf('#else\n  Dl_info info;'));
    const order = [
      'if (info.dli_fname[0] == \'/\' && rot_readable(cand)) {',
      'fopen("/proc/self/maps", "r")',
      'if (realpath(info.dli_fname, real) && rot_beside(real, cand, sizeof cand, name) && rot_readable(cand)) {',
      'return first[0] && rot_readable(first);',
    ].map((text) => linux.indexOf(text));
    expect(order.every((at) => at > 0)).toBe(true);
    expect(order).toEqual(order.slice().sort((a, b) => a - b));
    // PEARL_ROT_CUBIN wins on both systems, and a named file that is missing is not replaced by another.
    expect(ROT_PATH).toContain('if (e && *e) {\n    snprintf(buf, len, "%s", e);\n    return rot_readable(buf);\n  }');
  });
});
