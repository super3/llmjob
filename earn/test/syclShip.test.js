'use strict';

// The Intel (SYCL) core as a release carries it: how its kernels reach the
// hardware DPAS when the rig's own driver compiles them, what CI builds and
// stages on Linux and Windows, and that nothing staged is under a licence a
// release may not carry. None of it can be built here, so, as syclCore.test.js
// does for the port's sources, these read the files and pin what must hold,
// and run the staging script under bash against stand-ins for binutils.

const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');

const EARN = path.join(__dirname, '..');
const NATIVE = path.join(EARN, 'native');
const SYCL = path.join(NATIVE, 'sycl');
const read = (...p) => fs.readFileSync(path.join(...p), 'utf8').replace(/\r\n/g, '\n');
const KERNELS = read(SYCL, 'pearl_sycl_kernels.hpp');
const HOST = read(SYCL, 'pearl_sycl_host.cpp');
const BUILD = read(SYCL, 'build.sh');
const BUILD_WIN = read(SYCL, 'build-win.sh');
const STAGE = read(SYCL, 'stage.sh');
const TOOLCHAIN = read(SYCL, 'install-toolchain.sh');
const WORKFLOW = read(EARN, '..', '.github', 'workflows', 'native-core.yml');

function slice(text, from, to) {
  const i = text.indexOf(from);
  if (i < 0) throw new Error('not found: ' + from);
  const j = text.indexOf(to, i + from.length);
  if (j < 0) throw new Error('not found after it: ' + to);
  return text.slice(i, j + to.length);
}
// One job of the workflow: from its key to the next job's, or the end.
function job(name) {
  const at = WORKFLOW.indexOf('\n  ' + name + ':\n');
  if (at < 0) throw new Error('no job ' + name);
  const next = /\n {2}[a-z][a-z-]*:\n/.exec(WORKFLOW.slice(at + 1));
  return WORKFLOW.slice(at, next ? at + 1 + next.index : undefined);
}

// The open-source DPC++ release both systems build with, by name and digest.
const DPCPP_TAG = 'v7.2.0';
const SHA_LINUX = '385586a25e75ac727883f0be5e442dec0ee81b3c20f9a29037a79921ad2f1728';
const SHA_WINDOWS = '9d8964878a68e6d4ca3edced4fc1093fa6f4f51fd44698808e244870caa77bad';

// A driver compiles a kernel only when the kernel is launched on its card. So a
// fold whose kernel is nothing but the DPAS built-in is safe exactly when it
// is an image of its own and the host launches it only where the card has the
// instruction. On an Arc A750 this build mined through Intel's 26.40 Level
// Zero driver and through its 24.35 OpenCL driver, which is from August 2024.
describe('kernels the driver compiles reach the hardware DPAS by the built-in\'s name', () => {
  const NAME = 'intel_sub_group_i8_i8_matrix_mad_k32';

  test('the built-in is declared at global scope, for a word or a dword a lane of A', () => {
    const decl = slice(KERNELS, '#if defined(__SYCL_DEVICE_ONLY__) && (defined(PEARL_SYCL_XMX_HW) || defined(PEARL_SYCL_XMX_JIT))', '#endif');
    expect(decl.split('\n').slice(1, -1)).toEqual([
      'extern SYCL_EXTERNAL v8i ' + NAME + '(v8s, v8i, v8i);',
      'extern SYCL_EXTERNAL v8i ' + NAME + '(v8i, v8i, v8i);',
    ]);
    // Before the namespace opens: inside it the name would be mangled as ours,
    // and the compiler would not know it.
    expect(KERNELS.indexOf(decl)).toBeLessThan(KERNELS.indexOf('namespace pearl_sycl {'));
    expect(KERNELS).toContain('typedef int v8i __attribute__((ext_vector_type(8)));');
    expect(KERNELS).toContain('typedef short v8s __attribute__((ext_vector_type(8)));');
    // The linker's warning names it by its mangled form, 36 characters long.
    expect(NAME).toHaveLength(36);
  });

  test('the SPIR-V instruction, which a 2024 driver refuses, is gone from the build', () => {
    for (const [f, text] of [['kernels', KERNELS], ['host', HOST], ['build.sh', BUILD], ['build-win.sh', BUILD_WIN]]) {
      expect([f, /__spirv_SubgroupMatrixMultiplyAccumulateINTEL\(|-spirv-ext/.test(text)]).toEqual([f, false]);
    }
  });

  test('a hardware fold of a driver-compiled build is the built-in and nothing else', () => {
    const dpas = slice(KERNELS, 'template <int SG, bool HW>\ninline v8i dpas(', '\n}\n');
    const lines = dpas.split('\n').map((l) => l.trim()).filter((l) => l && !l.startsWith('//'));
    expect(lines.slice(2, 5)).toEqual([
      '#if defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)',
      'if constexpr (HW) return ::' + NAME + '(a, b, c);',
      '#endif',
    ]);
    // The prebuilt form picks inside the kernel, per target, and calls the same built-in.
    expect(dpas.match(new RegExp('r = ::' + NAME + '\\(a, b, c\\);', 'g'))).toHaveLength(2);
    expect(dpas.match(/\.otherwise\(\[&\]\(\) \{ r = dpas_emu<(16|8)>\(sg, a, b, c\); \}\);/g)).toHaveLength(2);
    expect(lines[lines.length - 2]).toBe('return dpas_emu<SG>(sg, a, b, c);');
    // One build is one or the other.
    expect(KERNELS).toContain('#if defined(PEARL_SYCL_XMX_HW) && defined(PEARL_SYCL_XMX_JIT)\n#error ');
  });

  test('the host launches a hardware fold only on a card that has it at that sub-group size', () => {
    const launch = slice(HOST, 'void launch_fold(Ctx *ctx, int fold, const FoldArgs &a, uint32_t groups) {', '\n}\n');
    const body = launch.split('\n').map((l) => l.trim());
    expect(body.slice(2, 4)).toEqual([
      'if (fold == kXmx16 && ctx->hwSg != 16) fold = kXmx16Emu;',
      'if (fold == kXmx8 && ctx->hwSg != 8) fold = kXmx8Emu;',
    ]);
    // The two lines in JS, over every fold and every kind of device.
    const [kDot, kXmx16, kXmx8, kXmx16Emu, kXmx8Emu] = [0, 1, 2, 3, 4];
    expect(HOST).toContain('enum Fold { kDot = 0, kXmx16 = 1, kXmx8 = 2, kXmx16Emu = 3, kXmx8Emu = 4 };');
    const launched = (fold, hwSg) => {
      if (fold === kXmx16 && hwSg !== 16) fold = kXmx16Emu;
      if (fold === kXmx8 && hwSg !== 8) fold = kXmx8Emu;
      return fold;
    };
    const hw = (fold) => fold === kXmx16 || fold === kXmx8;
    for (const hwSg of [0, 8, 16]) {
      for (const fold of [kDot, kXmx16, kXmx8, kXmx16Emu, kXmx8Emu]) {
        const got = launched(fold, hwSg);
        // A hardware kernel only where the device has that instruction.
        if (hw(got)) expect([hwSg, got === kXmx16 ? 16 : 8]).toEqual([hwSg, hwSg]);
        // And the same fold by name: its reference code otherwise.
        expect([fold, hwSg, got === fold || got === fold + 2]).toEqual([fold, hwSg, true]);
      }
    }
    // hwSg is the device's own answer, asked at run time in both builds.
    expect(HOST).toContain('#if defined(PEARL_SYCL_XMX_HW) || defined(PEARL_SYCL_XMX_JIT)\n// Which XMX DPAS the device\'s architecture has (16, 8), or 0 for none.');
    expect(slice(HOST, 'bool pick_fold(Ctx *ctx, char *err, size_t err_len) {', '\n}\n')).toContain('ctx->hwSg = sg;');
    expect(HOST.match(/ctx->hwSg = /g)).toHaveLength(1);
  });

  test('build.sh --jit makes every kernel a device image of its own', () => {
    expect(BUILD).toContain('SYCL_FLAGS+=(-fsycl-targets=spir64 -fsycl-device-code-split=per_kernel -DPEARL_SYCL_XMX_JIT)');
    expect(BUILD).toContain('SYCL_FLAGS+=(-fsycl-targets="$TARGETS" -DPEARL_SYCL_XMX_HW)');
    // --dump and --large-grf read or set what ocloc does, and --jit runs none.
    expect(BUILD).toContain('if [ "$JIT" = 1 ] && { [ -n "$DUMP" ] || [ "$LARGE_GRF" = 1 ]; }; then');
    expect(BUILD_WIN).toContain('-fsycl-targets=spir64 -fsycl-device-code-split=per_kernel -DPEARL_SYCL_XMX_JIT)');
  });
});

describe('the compiler a release is built with', () => {
  test('install-toolchain.sh --open fetches one release of the open-source DPC++, by digest', () => {
    const open = slice(TOOLCHAIN, 'if [ "$OPEN" = 1 ]; then', '\n  exit 0\nfi\n');
    expect(open).toContain('DPCPP_TAG=' + DPCPP_TAG + '\n');
    expect(open).toContain('DPCPP_SHA256=' + SHA_LINUX + '\n');
    expect(open).toContain('"https://github.com/intel/llvm/releases/download/$DPCPP_TAG/sycl_linux.tar.gz"');
    expect(open).toContain('echo "$DPCPP_SHA256  dl/sycl_linux.tar.gz" | sha256sum -c - >/dev/null');
    // It has no ocloc, and says so rather than failing a target in build.sh.
    expect(open).toContain('[ "$OCLOC" = 0 ] || { echo "--open has no ocloc');
  });

  test('the CPU runtime for the checks is Intel\'s, kept apart, and never on the compiler\'s path alone', () => {
    const open = slice(TOOLCHAIN, 'if [ "$OPEN" = 1 ]; then', '\n  exit 0\nfi\n');
    expect(open).toContain('intel-opencl-rt=2026.1.1 ocl-icd');
    expect(open).toContain('cat > cpu-rt.sh <<ENV');
    expect(open).toContain('echo "if [ -f \\"$DIR/cpu-rt.sh\\" ]; then . \\"$DIR/cpu-rt.sh\\"; fi"');
    // Its libraries go on the library path, since it needs a newer libstdc++
    // than Ubuntu 20.04 has. That set must hold no SYCL runtime, or a check
    // of a staged core would run the compiler's.
    expect(open).toContain('export LD_LIBRARY_PATH="$DIR/cpu-rt/lib\\${LD_LIBRARY_PATH:+:\\$LD_LIBRARY_PATH}"');
    expect(slice(open, 'micromamba create', 'rm -rf mroot dl mm')).not.toMatch(/dpcpp|sycl/);
  });

  test('build.sh takes icpx or the open-source clang++, and the addon links no SYCL of its own', () => {
    expect(BUILD).toContain('if command -v icpx >/dev/null; then CXX=icpx');
    expect(BUILD).toContain('elif command -v clang++ >/dev/null && [ -f "$(clang++ -fsycl -print-file-name=libsycl.so 2>/dev/null)" ]; then CXX=clang++');
    const addon = slice(BUILD, 'echo "[2/3] pearl_core_sycl.node"', 'echo "[3/3] pearl_sycl_check"');
    expect(addon).toContain('-c "$SRC/pearl_core.cc" -o "$OUT/pearl_core.o"');
    expect(addon).not.toContain('-fsycl');
    expect(addon).toContain('-shared "$OUT/pearl_core.o" -L"$OUT" -lpearl_sycl');
  });
});

describe('staging: the folder a release carries', () => {
  test('build.sh --stage needs --jit, and links ours to look beside themselves only', () => {
    expect(BUILD).toContain('if [ -n "$STAGE" ] && [ "$JIT" != 1 ]; then');
    expect(BUILD).toContain('if [ -n "$STAGE" ]; then RPATH+=(-Wl,-rpath-link,"$SYCL_LIB"); else RPATH+=(-Wl,-rpath,"$SYCL_LIB"); fi');
    expect(BUILD).toContain("RPATH=(-Wl,-rpath,'$ORIGIN')");
    expect(BUILD.trim().split('\n').slice(-3)).toEqual([
      'if [ -n "$STAGE" ]; then', '  bash "$HERE/stage.sh" "$OUT" "$SYCL_LIB" "$STAGE"', 'fi']);
  });

  test('the licence that travels with the runtime is the open-source DPC++\'s', () => {
    const lic = read(SYCL, 'licenses', 'LICENSE-DPCPP.txt');
    expect(lic).toContain('The LLVM Project is under the Apache License v2.0 with LLVM Exceptions:');
    expect(lic).toContain('---- LLVM Exceptions to the Apache 2.0 License ----');
    expect(STAGE).toContain('cp "$HERE/licenses/LICENSE-DPCPP.txt" "$STAGE/LICENSE-DPCPP.txt"');
    expect(BUILD_WIN).toContain('cp "$HERE/licenses/LICENSE-DPCPP.txt" "$STAGE/LICENSE-DPCPP.txt"');
  });

  // stage.sh for real, under bash. A stand-in "ELF" is a text file holding what
  // `readelf -d` would print for it, and the stand-in readelf prints the file.
  const hasBash = spawnSync('bash', ['--version'], { stdio: 'ignore' }).status === 0;
  (hasBash ? describe : describe.skip)('stage.sh under bash', () => {
    let tmp;
    const dyn = ({ needed = [], runpath } = {}) => needed.map((n) => ' 0x0001 (NEEDED)   Shared library: [' + n + ']')
      .concat(runpath == null ? [] : [' 0x001d (RUNPATH)  Library runpath: [' + runpath + ']']).join('\n') + '\n';
    const SYSTEM = ['libstdc++.so.6', 'libm.so.6', 'libgcc_s.so.1', 'libc.so.6'];
    const layout = (over = {}) => {
      const dir = fs.mkdtempSync(path.join(tmp, 'case-'));
      const build = path.join(dir, 'build');
      const lib = path.join(dir, 'dpcpp', 'lib');
      fs.mkdirSync(build);
      fs.mkdirSync(lib, { recursive: true });
      const files = {
        'build/pearl_core_sycl.node': dyn({ needed: ['libpearl_sycl.so', ...SYSTEM], runpath: '$ORIGIN' }),
        'build/libpearl_sycl.so': dyn({ needed: ['libsycl.so.9', ...SYSTEM], runpath: '$ORIGIN' }),
        'build/pearl_sycl_check': dyn({ needed: ['libpearl_sycl.so', ...SYSTEM, 'libpthread.so.0'], runpath: '$ORIGIN' }),
        'dpcpp/lib/libsycl.so.9': dyn({ needed: ['libumf.so.1', 'libdl.so.2', ...SYSTEM], runpath: '$ORIGIN' }),
        'dpcpp/lib/libumf.so.1': dyn({ needed: ['libdl.so.2', 'librt.so.1', 'libc.so.6'] }),
        'dpcpp/lib/libur_adapter_level_zero.so.0': dyn({ needed: ['libumf.so.1', ...SYSTEM], runpath: '$ORIGIN' }),
        'dpcpp/lib/libur_adapter_level_zero_v2.so.0': dyn({ needed: ['libumf.so.1', ...SYSTEM], runpath: '$ORIGIN' }),
        // In the compiler's folder, and no business of a release.
        'dpcpp/lib/libur_adapter_cuda.so.0': dyn({ needed: ['libcuda.so.1'] }),
        'dpcpp/lib/libsycl-preview.so.9': dyn({ needed: SYSTEM }),
        ...over,
      };
      for (const [f, text] of Object.entries(files)) if (text != null) fs.writeFileSync(path.join(dir, f), text);
      return { dir, build, lib, stage: path.join(dir, 'stage', 'sycl') };
    };
    const run = (l) => {
      const env = { ...process.env, PATH: path.join(tmp, 'bin') + path.delimiter + process.env.PATH };
      const r = spawnSync('bash', [path.join(SYCL, 'stage.sh'), l.build, l.lib, l.stage], { encoding: 'utf8', env });
      return { status: r.status, out: r.stdout, err: r.stderr, staged: fs.existsSync(l.stage) ? fs.readdirSync(l.stage).sort() : null };
    };
    beforeAll(() => {
      tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'sycl-stage-'));
      fs.mkdirSync(path.join(tmp, 'bin'));
      const tool = (name, lines) => fs.writeFileSync(path.join(tmp, 'bin', name), ['#!/usr/bin/env bash', ...lines, ''].join('\n'), { mode: 0o755 });
      tool('readelf', ['cat "$2"']);
      tool('strip', ['exit 0']);
    });
    afterAll(() => fs.rmSync(tmp, { recursive: true, force: true }));

    test('stages ours, the runtime ours load and the licence, and nothing else of the compiler', () => {
      const l = layout();
      const r = run(l);
      expect([r.status, r.err]).toEqual([0, '']);
      expect(r.staged).toEqual([
        'LICENSE-DPCPP.txt', 'NOTICE.txt', 'libpearl_sycl.so', 'libsycl.so.9', 'libumf.so.1',
        'libur_adapter_level_zero.so.0', 'libur_adapter_level_zero_v2.so.0', 'pearl_core_sycl.node', 'pearl_sycl_check',
      ]);
      const notice = read(l.stage, 'NOTICE.txt');
      for (const f of ['libsycl.so.9', 'libumf.so.1', 'libur_adapter_level_zero.so.0']) expect(notice).toContain('  ' + f + '\n');
      expect(notice).toContain('Apache License 2.0 with LLVM exceptions: LICENSE-DPCPP.txt');
      expect(notice).toContain('Not here, and needed on the rig: Intel\'s graphics driver for compute (Level Zero or OpenCL).');
      expect(r.out).toContain('staged in ');
    });

    test('takes the OpenCL adapter too when the compiler has it as a file of its own', () => {
      const l = layout({ 'dpcpp/lib/libur_adapter_opencl.so.0': dyn({ needed: ['libumf.so.1', ...SYSTEM], runpath: '$ORIGIN' }) });
      expect(run(l).staged).toContain('libur_adapter_opencl.so.0');
    });

    test('refuses a build that links Intel\'s oneAPI runtime, which a release may not carry', () => {
      for (const lib of ['libsvml.so', 'libimf.so', 'libirng.so', 'libintlc.so.5']) {
        const l = layout({ 'build/libpearl_sycl.so': dyn({ needed: ['libsycl.so.9', lib, ...SYSTEM], runpath: '$ORIGIN' }) });
        const r = run(l);
        expect([lib, r.status === 0, r.staged]).toEqual([lib, false, null]);
        expect(r.err).toContain('libpearl_sycl.so needs ' + lib + ', Intel\'s oneAPI runtime, which a release may not carry.');
        expect(r.err).toContain('./install-toolchain.sh --open');
      }
    });

    test('refuses a file whose run path names the build machine', () => {
      const l = layout({ 'build/libpearl_sycl.so': dyn({ needed: ['libsycl.so.9', ...SYSTEM], runpath: '$ORIGIN:/work/tools/dpcpp/lib' }) });
      const r = run(l);
      expect(r.status).not.toBe(0);
      expect(r.err).toContain("libpearl_sycl.so has run path '$ORIGIN:/work/tools/dpcpp/lib', not $ORIGIN");
    });

    test('refuses a staged file that needs a library the folder does not have', () => {
      const l = layout({ 'dpcpp/lib/libsycl.so.9': dyn({ needed: ['libumf.so.1', 'libhwloc.so.15', ...SYSTEM], runpath: '$ORIGIN' }) });
      const r = run(l);
      expect(r.status).not.toBe(0);
      expect(r.err).toContain('libsycl.so.9 needs libhwloc.so.15, which is not staged');
    });

    test('says which file is missing from the build', () => {
      const l = layout({ 'build/pearl_sycl_check': null });
      const r = run(l);
      expect(r.status).not.toBe(0);
      expect(r.err).toContain('pearl_sycl_check');
    });
  });
});

describe('the Windows build', () => {
  test('pearl_sycl.def exports every function the host defines, and no other', () => {
    const defined = (HOST.match(/^extern "C" [^\n(]*?\b(pearl_(?:host|sycl)_\w+)\(/gm) || [])
      .map((l) => /(pearl_(?:host|sycl)_\w+)\($/.exec(l)[1]);
    expect(defined.length).toBeGreaterThan(20);
    const def = read(SYCL, 'pearl_sycl.def').split('\n').filter((l) => l && !l.startsWith(';'));
    expect(def.slice(0, 2)).toEqual(['LIBRARY pearl_sycl', 'EXPORTS']);
    expect(def.slice(2).map((l) => l.trim()).sort()).toEqual(defined.slice().sort());
    expect(new Set(defined).size).toBe(defined.length);
  });

  test('build-win.sh makes a DLL with those exports and an import library for node-gyp', () => {
    expect(BUILD_WIN).toContain('"-Wl,/DEF:$(cygpath -m "$HERE/pearl_sycl.def")" "-Wl,/IMPLIB:$(cygpath -m "$OUT/pearl_sycl.lib")"');
    // The dynamic C++ runtime, as sycl9.dll has.
    expect(BUILD_WIN).toContain('CRT=(-fms-runtime-lib=dll)');
    const gyp = read(NATIVE, 'binding.gyp');
    expect(slice(gyp, '["OS==\'win\' and pearl_backend==\'sycl\'", {', '}],'))
      .toContain('"libraries": [ "<(module_root_dir)/sycl/build-win/pearl_sycl.lib" ]');
  });

  test('it stages the addon under its release name with the runtime DLLs beside it', () => {
    expect(BUILD_WIN).toContain('cp "$ADDON" "$STAGE/pearl_core_sycl.node"');
    expect(BUILD_WIN).toContain('RUNTIME=(sycl9.dll ur_adapter_level_zero.dll ur_adapter_level_zero_v2.dll ur_win_proxy_loader.dll)');
    expect(BUILD_WIN).toContain('[ -f "$ROOT/bin/$f" ] || { echo "stage: the DPC++ release has no bin/$f" >&2; exit 1; }');
  });
});

describe('CI builds the core for Linux and Windows (native-core.yml)', () => {
  const linux = job('intel');
  const win = job('intel-windows');
  const gate = job('intel-gate');

  test('each job is named for its artifact, cannot fail the run, and ends before the installer stops waiting', () => {
    for (const [j, name] of [[linux, 'ubuntu-latest-sycl'], [win, 'windows-2022-sycl'], [gate, 'ubuntu-24.04-sycl-gate']]) {
      expect(j).toContain('\n    name: ' + name + '\n');
      expect(j).toContain('\n    continue-on-error: true\n');
    }
    // pick-core-run.sh waits 25 minutes for the whole run. The gate ships nothing.
    const minutes = (j) => parseInt(/\n {4}timeout-minutes: (\d+)\n/.exec(j)[1], 10);
    expect(minutes(linux)).toBeLessThan(25);
    expect(minutes(win)).toBeLessThanOrEqual(25);
    expect(gate).not.toContain('upload-artifact');
  });

  test('Linux: built in Ubuntu 20.04, staged, and uploaded as a folder', () => {
    expect(linux).toContain('\n    container: ubuntu:20.04\n');
    expect(linux).toContain('run: ./install-toolchain.sh --open --cpu-rt');
    expect(linux).toContain('./build.sh --jit --stage stage/sycl');
    expect(linux).toContain('*"DPC++ compiler 7.2.0"*) ;;');
    expect(slice(linux, '- uses: actions/upload-artifact@v4', 'if-no-files-found: error'))
      .toMatch(/name: pearl-core-sycl-ubuntu-latest\n\s+path: earn\/native\/sycl\/stage\n/);
    // The cache key changes with either version install-toolchain.sh names.
    expect(linux).toContain('key: dpcpp-open-' + DPCPP_TAG + "-cpurt-2026.1.1-ubuntu-20.04-${{ hashFiles('earn/native/sycl/install-toolchain.sh') }}");
  });

  test('Linux: what is checked is the staged folder, on Ubuntu 20.04\'s limits and on the CPU device', () => {
    const check = slice(linux, '- name: Check the staged core', '- name: Check the folds on the SYCL CPU device');
    expect(check).toContain('check-glibc-floor.sh --ubuntu 20.04 pearl_core_sycl.node pearl_sycl_check *.so*');
    expect(check).toContain("node -e \"const a = require('./pearl_core_sycl.node');");
    const cpu = slice(linux, '- name: Check the folds on the SYCL CPU device', '- uses: actions/upload-artifact@v4');
    // Only the CPU runtime's environment: the staged core finds its own SYCL runtime.
    expect(cpu).toContain('. toolchain-open/cpu-rt.sh\n');
    expect(cpu).not.toContain('envrc.sh');
    expect(cpu).toContain('PEARL_SYCL_BUILD=$PWD/stage/sycl PEARL_SYCL_DEVICE=cpu ./verify.sh --quick');
  });

  test('Windows: the same release of the compiler by digest, the DLL, the addon by node-gyp, the same checks', () => {
    expect(win).toContain("DPCPP_TAG: " + DPCPP_TAG + '\n');
    expect(win).toContain('DPCPP_SHA256: ' + SHA_WINDOWS + '\n');
    expect(win).toContain('echo "$DPCPP_SHA256  sycl_windows.tar.gz" | sha256sum -c -');
    const order = ['run: ./build-win.sh\n', 'PEARL_BACKEND: sycl\n', 'run: npx node-gyp rebuild\n', 'run: ./build-win.sh --stage stage/sycl\n',
      'PEARL_SYCL_BUILD=$PWD/stage/sycl PEARL_SYCL_DEVICE=cpu ./verify.sh --quick', 'name: pearl-core-sycl-windows-2022\n'];
    let at = -1;
    for (const s of order) {
      const i = win.indexOf(s, at + 1);
      expect([s, i > at]).toEqual([s, true]);
      at = i;
    }
    // Every DLL a staged file needs is staged, Windows' own, Microsoft's C++
    // runtime, or a loader the driver installs.
    expect(win).toContain('*) [ -f "$d" ] || { echo "::error::$f needs $d, which is not staged and not part of Windows"; exit 1; } ;;');
  });

  test('the compile gate runs on every target with Intel\'s own compiler, and ships nothing', () => {
    expect(gate).toContain('run: ./install-toolchain.sh --ocloc');
    expect(gate).toMatch(/\n\s+\.\/compile-gate\.sh\n/);
  });
});

// What each package carries. The core is a folder, sycl/, beside the other
// cores: the addon cannot load without the libraries next to it.
describe('the packages carry the folder', () => {
  const pkg = JSON.parse(read(EARN, 'package.json'));
  const MINER_BUILD = read(EARN, '..', '.github', 'workflows', 'miner-build.yml');
  const BUILD_CLI = read(EARN, 'scripts', 'build-cli.mjs');
  const BUILD_HIVE = read(EARN, 'scripts', 'build-hiveos.mjs');
  const SMOKE = read(EARN, '..', '.github', 'scripts', 'hiveos-smoke.sh');

  test('the installers pack src, package.json and what CI stages in vendor, the sycl folder with it', () => {
    expect(pkg.build.files.length).toBeGreaterThan(0);
    for (const pattern of pkg.build.files) {
      expect([pattern, pattern === 'package.json' || pattern.startsWith('src/')]).toEqual([pattern, true]);
    }
    const extra = Object.values(pkg.build).flatMap((v) => (v && Array.isArray(v.extraResources) ? v.extraResources : []));
    for (const r of extra) expect([r.from, r.from.startsWith('vendor/')]).toEqual([r.from, true]);
    // From vendor/native: addons, the Monero worker, and the Intel core's folder
    // whole (its libraries and their licence). Windows and Linux alike.
    const native = extra.filter((x) => x.to === 'native');
    expect(native).toHaveLength(2);
    for (const r of native) {
      expect(r.filter).toEqual(['**/*.node', 'llmjob-xmr-worker', 'llmjob-xmr-worker.exe', 'sycl/**']);
    }
    // That folder is where the loader looks (src/main/pearlCore.js).
    const { INTEL_DIR, FILES, INTEL } = require('../src/shared/coreVariant');
    expect([INTEL_DIR, FILES[INTEL]]).toEqual(['sycl', 'pearl_core_sycl.node']);
    expect(read(EARN, 'src', 'main', 'pearlCore.js')).toContain('dirs.map((d) => path.join(d, INTEL_DIR, FILES[INTEL]))');
  });

  test('the installer build stages it from the same native-core run as the other cores, and says when it has none', () => {
    const step = slice(MINER_BUILD, '- name: Stage the Intel Pearl core', 'if_no_artifact_found: warn');
    expect(step).toContain("name: pearl-core-sycl-${{ runner.os == 'Windows' && 'windows-2022' || 'ubuntu-latest' }}");
    expect(step).toContain('run_id: ${{ steps.pick.outputs.run_id }}');
    expect(step).toContain('path: earn/vendor/native');
    expect(step).toContain('continue-on-error: true');
    expect(MINER_BUILD).toContain('if [ -f vendor/native/sycl/pearl_core_sycl.node ]; then');
    expect(MINER_BUILD).toContain('this $RUNNER_OS build cannot mine on Intel Arc cards');
    // The artifact is the staged folder's parent, so it unpacks as sycl/.
    expect(job('intel')).toMatch(/name: pearl-core-sycl-ubuntu-latest\n\s+path: earn\/native\/sycl\/stage\n/);
    expect(job('intel')).toContain('./build.sh --jit --stage stage/sycl');
  });

  test('the Linux CLI keeps it beside the binary and as one archive for the release', () => {
    expect(BUILD_CLI).toContain("const sycl = join(root, 'vendor', 'native', 'sycl');");
    expect(BUILD_CLI).toContain("const syclOut = join(dist, 'sycl');");
    expect(BUILD_CLI).toContain("chmodSync(join(syclOut, f), f === 'pearl_sycl_check' ? 0o755 : 0o644);");
    expect(BUILD_CLI).toContain("execFileSync('tar', ['-czf', join(dist, 'pearl_core_sycl-linux.tar.gz'), '-C', dist, 'sycl']);");
    // A stale folder from an earlier build is not left beside a new binary.
    expect(BUILD_CLI.indexOf('rmSync(syclOut, { recursive: true, force: true });')).toBeLessThan(BUILD_CLI.indexOf('mkdirSync(syclOut);'));
    const publish = slice(MINER_BUILD, '- name: Publish CLI + HiveOS package to release', 'gh release upload "v$VERSION" dist/llmjob-earn-cli-linux');
    expect(publish).toContain('gh release upload "v$VERSION" dist/pearl_core_sycl-linux.tar.gz --clobber');
    expect(publish).toContain('gh release delete-asset "v$VERSION" pearl_core_sycl-linux.tar.gz --yes');
  });

  test('the HiveOS package has it, and its smoke test loads it on every image', () => {
    expect(BUILD_HIVE).toContain("const sycl = join(dist, 'sycl');");
    expect(BUILD_HIVE).toContain("chmodSync(join(syclDir, f), f === 'pearl_sycl_check' ? 0o755 : 0o644);");
    expect(SMOKE).toContain('configure "--region us --core intel"');
    expect(SMOKE).toContain('wait_for "Pearl core: Intel build (SYCL)" 60');
    // The core's own words when it loaded and found no card (pearl_sycl_host.cpp).
    const NO_CARD = 'no Intel GPU found through Level Zero or OpenCL';
    expect(SMOKE).toContain('wait_for "' + NO_CARD + '" 60');
    expect(HOST).toContain('"' + NO_CARD + '"');
    expect(read(EARN, 'src', 'main', 'pearlCore.js')).toContain('noCard: /no Intel GPU found/i');
  });
});
