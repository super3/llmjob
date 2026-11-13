'use strict';

// The Quantus cores as CI builds and packages them: which jobs build which
// core, where each lands in each package, and that those are the places
// src/main/quantusCore.js looks. Nothing here can build a core, so, as
// syclShip.test.js does for the Pearl Intel core, these read the workflows
// and scripts and pin what must hold, and run the scripts that can run here
// (build-sycl-win.sh with a stand-in compiler, stage-quantus-sycl.mjs, the
// job that finds the vendor build scripts) for real.

const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');
const qc = require('../src/main/quantusCore');
const { encodeQuantusAddress, isValidQuantusAddress } = require('../src/shared/quantus/address');

const EARN = path.join(__dirname, '..');
const REPO = path.join(EARN, '..');
const QUANTUS = path.join(EARN, 'native', 'quantus');
const read = (...p) => fs.readFileSync(path.join(...p), 'utf8').replace(/\r\n/g, '\n');
const NATIVE_CORE = read(REPO, '.github', 'workflows', 'native-core.yml');
const MINER_BUILD = read(REPO, '.github', 'workflows', 'miner-build.yml');
const SMOKE = read(REPO, '.github', 'scripts', 'hiveos-smoke.sh');
const BUILD_WIN = read(QUANTUS, 'build-sycl-win.sh');
const HOOK = read(QUANTUS, 'win_delay_load_hook.cc');
const PEARL_BUILD_WIN = read(EARN, 'native', 'sycl', 'build-win.sh');
const BUILD_CLI = read(EARN, 'scripts', 'build-cli.mjs');
const BUILD_HIVE = read(EARN, 'scripts', 'build-hiveos.mjs');
const STAGE_SCRIPT = path.join(EARN, 'scripts', 'stage-quantus-sycl.mjs');

function slice(text, from, to) {
  const i = text.indexOf(from);
  if (i < 0) throw new Error('not found: ' + from);
  const j = text.indexOf(to, i + from.length);
  if (j < 0) throw new Error('not found after it: ' + to);
  return text.slice(i, j + to.length);
}
// One job of native-core.yml: from its key to the next job's, or the end.
function job(name) {
  const at = NATIVE_CORE.indexOf('\n  ' + name + ':\n');
  if (at < 0) throw new Error('no job ' + name);
  const next = /\n {2}[a-z][a-z-]*:\n/.exec(NATIVE_CORE.slice(at + 1));
  return NATIVE_CORE.slice(at, next ? at + 1 + next.index : undefined);
}
// A job's comment header and body without its comments, so a step's text is matched, not a remark.
const code = (text) => text.split('\n').filter((l) => !/^\s*#/.test(l)).join('\n');
const hasBash = spawnSync('bash', ['--version'], { stdio: 'ignore' }).status === 0;

// What each job builds, the artifact it uploads, and the folder that artifact holds.
const JOBS = {
  'quantus-intel': { name: 'ubuntu-latest-quantus-sycl', artifact: 'quantus-core-sycl-ubuntu-latest', path: 'earn/native/quantus/build/stage' },
  'quantus-intel-windows': { name: 'windows-2022-quantus-sycl', artifact: 'quantus-core-sycl-windows-2022', path: 'earn/native/quantus/build/stage' },
  'quantus-amd': { name: 'ubuntu-latest-quantus-hip', artifact: 'quantus-core-hip-ubuntu-latest', path: 'earn/native/quantus/build/quantus_hip.node', gate: 'hip' },
  'quantus-cuda': { name: 'ubuntu-latest-quantus-cuda', artifact: 'quantus-core-cuda-ubuntu-latest', path: 'earn/native/quantus/build/quantus_cuda.node', gate: 'cuda' },
  'quantus-cuda-windows': { name: 'windows-2022-quantus-cuda', artifact: 'quantus-core-cuda-windows-2022', path: 'earn/native/quantus/build/quantus_cuda.node', gate: 'cuda_windows' },
};

describe('native-core.yml builds each Quantus core in a job of its own', () => {
  test('each job is named for its core, cannot fail the run, and ends before the installer stops waiting', () => {
    for (const [key, j] of Object.entries(JOBS)) {
      const text = job(key);
      expect([key, text.includes('\n    name: ' + j.name + '\n')]).toEqual([key, true]);
      expect([key, text.includes('\n    continue-on-error: true\n')]).toEqual([key, true]);
      // pick-core-run.sh waits 25 minutes for the whole run.
      const minutes = parseInt(/\n {4}timeout-minutes: (\d+)\n/.exec(text)[1], 10);
      expect([key, minutes <= 25]).toEqual([key, true]);
    }
    expect(job('quantus-cores')).toContain('\n    continue-on-error: true\n');
  });

  test('each uploads its core under the name miner-build.yml downloads, from where it was built', () => {
    for (const [key, j] of Object.entries(JOBS)) {
      const upload = slice(job(key), '- uses: actions/upload-artifact@v4', 'if-no-files-found: error');
      expect([key, upload]).toEqual([key, expect.stringContaining('name: ' + j.artifact + '\n')]);
      expect([key, upload]).toEqual([key, expect.stringContaining('path: ' + j.path + '\n')]);
    }
  });

  test('the NVIDIA and AMD jobs run only once their build script is on the branch; the Intel ones always', () => {
    const find = job('quantus-cores');
    for (const out of ['cuda', 'cuda_windows', 'hip']) {
      expect(find).toContain('\n      ' + out + ': ${{ steps.find.outputs.' + out + ' }}\n');
    }
    expect(find).toContain('have cuda cuda/build.sh\n');
    expect(find).toContain('have cuda_windows cuda/build.ps1\n');
    expect(find).toContain('have hip amd/build.sh\n');
    for (const [key, j] of Object.entries(JOBS)) {
      const text = job(key);
      if (j.gate) {
        expect([key, text]).toEqual([key, expect.stringContaining('\n    needs: quantus-cores\n    if: needs.quantus-cores.outputs.' + j.gate + " == 'true'\n")]);
      } else {
        expect([key, /\n {4}(needs|if):/.test(text)]).toEqual([key, false]);
      }
    }
  });

  // The find step's shell, run for real on a checkout with and without the scripts.
  (hasBash ? test : test.skip)('the find step says which scripts are there, and a notice for each that is not', () => {
    const block = slice(job('quantus-cores'), 'have() {', 'have hip amd/build.sh');
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'quantus-find-'));
    try {
      const run = (files) => {
        const root = fs.mkdtempSync(path.join(tmp, 'co-'));
        for (const f of files) {
          fs.mkdirSync(path.join(root, 'earn', 'native', 'quantus', path.dirname(f)), { recursive: true });
          fs.writeFileSync(path.join(root, 'earn', 'native', 'quantus', f), '');
        }
        const out = path.join(root, 'out.txt');
        fs.writeFileSync(path.join(root, 'find.sh'), block + '\n');
        const r = spawnSync('bash', ['find.sh'], { cwd: root, encoding: 'utf8', env: { ...process.env, GITHUB_OUTPUT: out } });
        expect([r.status, r.stderr]).toEqual([0, '']);
        return { outputs: fs.readFileSync(out, 'utf8').trim().split('\n'), log: r.stdout };
      };
      const none = run(['sycl/build.sh']);
      expect(none.outputs).toEqual(['cuda=false', 'cuda_windows=false', 'hip=false']);
      expect(none.log).toContain('::notice::no earn/native/quantus/amd/build.sh on this commit, so its job is skipped');
      const two = run(['cuda/build.sh', 'amd/build.sh']);
      expect(two.outputs).toEqual(['cuda=true', 'cuda_windows=false', 'hip=true']);
      expect(two.log).toContain('earn/native/quantus/cuda/build.sh: there, so its job runs');
    } finally {
      fs.rmSync(tmp, { recursive: true, force: true });
    }
  });

  test('the Intel jobs build with the Pearl Intel jobs\' compiler, from their caches', () => {
    const cacheKey = (text) => /\n {10}key: (.+)\n/.exec(text.slice(text.indexOf('- name: Restore the DPC++ toolchain')))[1];
    expect(cacheKey(job('quantus-intel'))).toBe(cacheKey(job('intel')));
    expect(cacheKey(job('quantus-intel-windows'))).toBe(cacheKey(job('intel-windows')));
    // The Windows compiler by the same tag and digest, installed by the same commands.
    const env = (text) => slice(text, '\n    env:\n', '\n    steps:\n');
    expect(env(job('quantus-intel-windows'))).toBe(env(job('intel-windows')));
    const install = (text) => slice(text, '- name: Install the DPC++ toolchain', 'rm -rf mm mm.tar.bz2 mroot\n')
      .split('\n').filter((l) => !/^\s*#/.test(l)).join('\n');
    expect(install(job('quantus-intel-windows'))).toBe(install(job('intel-windows')));
    expect(job('quantus-intel')).toContain('run: ./install-toolchain.sh --open --cpu-rt\n');
    expect(job('quantus-intel')).toContain('\n    container: ubuntu:20.04\n');
    for (const key of ['quantus-intel', 'quantus-intel-windows']) {
      expect([key, job(key)]).toEqual([key, expect.stringContaining('*"DPC++ compiler 7.2.0"*) ;;')]);
    }
  });

  test('Linux Intel: built and staged by the core\'s own build.sh, checked on 20.04\'s limits and on the CPU device', () => {
    const j = code(job('quantus-intel'));
    expect(j).toContain('working-directory: earn/native/quantus/sycl\n');
    expect(j).toContain('./build.sh --jit --stage ../build/stage/sycl\n');
    const check = slice(j, '- name: Check the staged core', '- name: Check the core on the SYCL CPU device');
    expect(check).toContain('check-glibc-floor.sh" --ubuntu 20.04 quantus_sycl.node *.so*');
    expect(check).toContain("a.QUANTUS_API !== 1");
    const cpu = slice(j, '- name: Check the core on the SYCL CPU device', '- uses: actions/upload-artifact@v4');
    // Only the CPU runtime's environment: the staged core must find its own SYCL runtime.
    expect(cpu).toContain('. earn/native/sycl/toolchain-open/cpu-rt.sh\n');
    expect(cpu).not.toContain('envrc.sh');
    expect(cpu).toContain('QUANTUS_SYCL_DEVICE=cpu node earn/native/quantus/verify-core.js --addon earn/native/quantus/build/stage/sycl/quantus_sycl.node --big 16777216');
  });

  test('Windows Intel: build-sycl-win.sh, the Pearl core\'s DLL rules, node.exe delay-loaded, and the CPU device', () => {
    const j = code(job('quantus-intel-windows'));
    const order = ['uses: ilammy/msvc-dev-cmd@v1', 'npx node-gyp install\n', 'run: bash build-sycl-win.sh --stage build/stage/sycl\n',
      '- name: Check the staged core', '- name: Check the core on the SYCL CPU device', 'name: quantus-core-sycl-windows-2022\n'];
    let at = -1;
    for (const s of order) {
      const i = j.indexOf(s, at + 1);
      expect([s, i > at]).toEqual([s, true]);
      at = i;
    }
    // Every DLL a staged file needs is staged, Windows' own, Microsoft's C++ runtime, or a loader
    // the driver installs: the Pearl Intel core's list, word for word.
    const allowed = (text) => slice(text, 'case "$d" in\n', '*) [ -f "$d" ]').split('\n').map((l) => l.trim());
    expect(allowed(j)).toEqual(allowed(job('intel-windows')));
    expect(j).toContain("sed -n '/delay load dependencies/,$p'");
    expect(j).toContain('*node.exe*) echo "node.exe is delay-loaded" ;;');
    expect(j).toContain('PATH="/c/Windows/System32:/c/Windows" "$d/other-host.exe" -e "$check" "$addon"');
    expect(j).toContain('QUANTUS_SYCL_DEVICE=cpu node verify-core.js --addon build/stage/sycl/quantus_sycl.node --big 16777216');
    expect(j).toContain("MSYS_NO_PATHCONV=1 reg add 'HKLM\\SOFTWARE\\Khronos\\OpenCL\\Vendors'");
  });

  test('AMD and NVIDIA: each vendor\'s build script, in the Pearl core\'s image, held to Ubuntu 20.04', () => {
    const image = (text) => /\n {4}container: (\S+)\n/.exec(text)[1];
    expect(image(job('quantus-amd'))).toBe(image(job('amd')));
    expect(job('quantus-cuda')).toContain('\n    container: nvidia/cuda:12.8.1-devel-rockylinux8\n');
    expect(job('build')).toContain('container: nvidia/cuda:12.8.1-devel-rockylinux8\n');
    const amd = code(job('quantus-amd'));
    expect(amd).toContain('run: bash build.sh check\n');
    expect(amd).toContain('run: bash build.sh gpu\n');
    expect(amd).toContain('check-glibc-floor.sh" --ubuntu 20.04 "$f"');
    const cuda = code(job('quantus-cuda'));
    expect(cuda).toContain('run: CUDA_HOME=/usr/local/cuda bash earn/native/quantus/cuda/build.sh\n');
    expect(cuda).toContain('check-glibc-floor.sh" --ubuntu 20.04 "$f"');
    expect(code(job('quantus-cuda-windows'))).toContain('run: .\\earn\\native\\quantus\\cuda\\build.ps1 -Cuda "$env:CUDA_PATH"\n');
    // With no card, each is asked for its cards, as the loader asks every core it finds.
    for (const key of ['quantus-amd', 'quantus-cuda', 'quantus-cuda-windows']) {
      expect([key, job(key)]).toEqual([key, expect.stringContaining('try { d = a.devices(); }')]);
    }
  });
});

describe('build-sycl-win.sh: the Intel core for Windows', () => {
  test('the same runtime the Pearl Intel core stages, the dynamic C++ runtime, and SPIR-V the driver compiles', () => {
    const runtime = (text) => /\n\s*RUNTIME=\((.+)\)\n/.exec(text)[1];
    expect(runtime(BUILD_WIN)).toBe(runtime(PEARL_BUILD_WIN));
    expect(BUILD_WIN).toContain('CRT=(-fms-runtime-lib=dll)');
    expect(BUILD_WIN).toContain('-fsycl-targets=spir64 -fsycl-device-code-split=per_kernel)');
    expect(BUILD_WIN).toContain('cp "$HERE/../sycl/licenses/LICENSE-DPCPP.txt" "$STAGE/LICENSE-DPCPP.txt"');
    expect(BUILD_WIN).toContain('cp "$HERE/../sycl/licenses/THIRD-PARTY-DPCPP.txt" "$STAGE/THIRD-PARTY-DPCPP.txt"');
  });

  test('the hook answers the node.exe load with the running program, as node-gyp\'s does', () => {
    expect(HOOK).toContain('if (event != dliNotePreLoadLibrary) return NULL;');
    expect(HOOK).toContain('if (_stricmp(info->szDll, "node.exe") != 0) return NULL;');
    expect(HOOK).toContain('if (m == NULL) m = GetModuleHandleA(NULL);');
    expect(HOOK).toContain('decltype(__pfnDliNotifyHook2) __pfnDliNotifyHook2 = quantus_load_exe_hook;');
    expect(BUILD_WIN).toContain('-Wl,delayimp.lib -Wl,/DELAYLOAD:node.exe');
  });

  // The script for real, under bash, with stand-ins for the compiler, cygpath and node. The
  // stand-in compiler writes down its arguments and makes the files it is asked for.
  (hasBash ? describe : describe.skip)('under bash, with a stand-in compiler', () => {
    let tmp;
    const VER = '22.99.0';
    const tool = (dir, name, lines) => fs.writeFileSync(path.join(dir, name), ['#!/usr/bin/env bash', ...lines, ''].join('\n'), { mode: 0o755 });
    const RUNTIME = ['sycl9.dll', 'ur_adapter_level_zero.dll', 'ur_adapter_level_zero_v2.dll', 'ur_win_proxy_loader.dll'];
    beforeEach(() => {
      tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'quantus-win-'));
      fs.mkdirSync(path.join(tmp, 'bin'));
      fs.mkdirSync(path.join(tmp, 'dpcpp', 'bin'), { recursive: true });
      fs.mkdirSync(path.join(tmp, 'gyp', VER, 'include', 'node'), { recursive: true });
      fs.mkdirSync(path.join(tmp, 'gyp', VER, 'x64'), { recursive: true });
      fs.writeFileSync(path.join(tmp, 'gyp', VER, 'include', 'node', 'node_api.h'), '');
      fs.writeFileSync(path.join(tmp, 'gyp', VER, 'x64', 'node.lib'), '');
      tool(path.join(tmp, 'bin'), 'cygpath', ['echo "$2"']);
      tool(path.join(tmp, 'bin'), 'node', ['echo ' + VER]);
      tool(path.join(tmp, 'dpcpp', 'bin'), 'clang++.exe', [
        'if [ "$1" = --version ]; then echo "clang version 21.0.0git (DPC++ compiler 7.2.0, a stand-in)"; exit 0; fi',
        'printf "%s\\n" "$@" "----" >> "$QTC_FAKE_LOG"',
        'out=""; shared=0',
        'while [ $# -gt 0 ]; do case "$1" in -o) out="$2"; shift 2 ;; -shared) shared=1; shift ;; *) shift ;; esac; done',
        'echo stand-in > "$out"',
        'if [ "$shared" = 1 ]; then echo lib > "${out%.*}.lib"; echo exp > "${out%.*}.exp"; fi',
      ]);
      for (const f of RUNTIME) fs.writeFileSync(path.join(tmp, 'dpcpp', 'bin', f), 'stand-in ' + f + '\n');
    });
    afterEach(() => fs.rmSync(tmp, { recursive: true, force: true }));

    const run = (args, over = {}) => {
      const env = {
        ...process.env, PATH: path.join(tmp, 'bin') + path.delimiter + process.env.PATH,
        DPCPP_ROOT: path.join(tmp, 'dpcpp'), NODE_GYP_DIR: path.join(tmp, 'gyp'), QTC_FAKE_LOG: path.join(tmp, 'calls.txt'), ...over,
      };
      for (const [k, v] of Object.entries(env)) if (v === undefined) delete env[k];
      const r = spawnSync('bash', [path.join(QUANTUS, 'build-sycl-win.sh'), '--out', path.join(tmp, 'out'), ...args], { encoding: 'utf8', env });
      const log = fs.existsSync(env.QTC_FAKE_LOG || '') ? fs.readFileSync(env.QTC_FAKE_LOG, 'utf8') : '';
      const calls = log.split('----\n').filter(Boolean).map((c) => c.trim().split('\n'));
      return { ...r, calls };
    };
    const ends = (list, tail) => list.some((a) => a.replace(/\\/g, '/').endsWith(tail));

    test('compiles the hook, then the addon against node.lib with node.exe delay-loaded, and stages the folder', () => {
      const stage = path.join(tmp, 'stage', 'sycl');
      const r = run(['--stage', stage], { QTC_SYCL_EXTRA: '-DQTC_SHARE=1' });
      expect([r.status, r.stderr]).toEqual([0, '']);
      expect(r.calls).toHaveLength(2);
      const [hook, addon] = r.calls;
      expect(hook.slice(0, 4)).toEqual(['-c', '-O2', '-std=c++17', '-fms-runtime-lib=dll']);
      expect(ends(hook, 'native/quantus/win_delay_load_hook.cc')).toBe(true);
      expect(addon.slice(0, 7)).toEqual(['-fsycl', '-O3', '-std=c++17', '-fms-runtime-lib=dll', '-Wno-psabi',
        '-fsycl-targets=spir64', '-fsycl-device-code-split=per_kernel']);
      expect(addon).toContain('-DQTC_SHARE=1');
      expect(addon).toContain('-shared');
      expect(ends(addon, 'native/quantus/sycl/quantus_sycl.cpp')).toBe(true);
      expect(ends(addon, '/win_delay_load_hook.o')).toBe(true);
      expect(addon.some((a) => a.startsWith('-I') && a.replace(/\\/g, '/').endsWith('gyp/' + VER + '/include/node'))).toBe(true);
      expect(addon.some((a) => a.startsWith('-Wl,') && a.replace(/\\/g, '/').endsWith('gyp/' + VER + '/x64/node.lib'))).toBe(true);
      expect(addon.slice(-2)).toEqual(['-Wl,delayimp.lib', '-Wl,/DELAYLOAD:node.exe']);
      // The addon, and not the import library or exports file the link leaves beside it.
      expect(fs.readdirSync(path.join(tmp, 'out')).sort()).toEqual(['quantus_sycl.node']);
      expect(fs.readdirSync(stage).sort()).toEqual(['LICENSE-DPCPP.txt', 'NOTICE.txt', 'THIRD-PARTY-DPCPP.txt', 'quantus_sycl.node', ...RUNTIME].sort());
      for (const f of RUNTIME) expect(fs.readFileSync(path.join(stage, f), 'utf8')).toBe('stand-in ' + f + '\n');
      const notice = fs.readFileSync(path.join(stage, 'NOTICE.txt'), 'utf8');
      expect(notice).toContain('Ours (MIT, as the rest of LLMJob Earn): quantus_sycl.node.');
      expect(notice).toContain('(https://github.com/intel/llvm, clang version 21.0.0git (DPC++ compiler 7.2.0, a stand-in)):');
      for (const f of RUNTIME) expect(notice).toContain('  ' + f + '\n');
      expect(fs.readFileSync(path.join(stage, 'LICENSE-DPCPP.txt'))).toEqual(fs.readFileSync(path.join(EARN, 'native', 'sycl', 'licenses', 'LICENSE-DPCPP.txt')));
      expect(r.stdout).toContain('staged in ');
    });

    test('without --stage it builds the addon only', () => {
      const r = run([]);
      expect(r.status).toBe(0);
      expect(r.stdout).toContain('done: ');
      expect(r.stdout).not.toContain('staged in');
    });

    test('a DPC++ release without one of the runtime DLLs stages nothing', () => {
      fs.rmSync(path.join(tmp, 'dpcpp', 'bin', 'ur_win_proxy_loader.dll'));
      const r = run(['--stage', path.join(tmp, 'stage', 'sycl')]);
      expect(r.status).not.toBe(0);
      expect(r.stderr).toContain('stage: the DPC++ release has no bin/ur_win_proxy_loader.dll');
    });

    test('says what is missing: the compiler\'s folder, Node\'s headers, an option', () => {
      let r = run([], { DPCPP_ROOT: undefined });
      expect(r.status).not.toBe(0);
      expect(r.stderr).toContain('set DPCPP_ROOT to the unpacked sycl_windows release');
      r = run([], { DPCPP_ROOT: path.join(tmp, 'nowhere') });
      expect(r.status).not.toBe(0);
      expect(r.stderr).toContain('nowhere/bin/clang++.exe');
      // No headers in the cache, and npx (a stand-in here) does not fetch them.
      tool(path.join(tmp, 'bin'), 'npx', ['exit 0']);
      fs.rmSync(path.join(tmp, 'gyp', VER, 'x64', 'node.lib'));
      r = run([]);
      expect(r.status).not.toBe(0);
      expect(r.stderr).toContain('no Node ' + VER + ' headers and node.lib in ');
      r = run(['--fast']);
      expect(r.status).toBe(2);
      expect(r.stderr).toContain('unknown option --fast');
    });
  });
});

describe('stage-quantus-sycl.mjs: the Intel Quantus core into the Pearl Intel core\'s folder', () => {
  let tmp;
  const put = (dir, files) => {
    fs.mkdirSync(dir, { recursive: true });
    for (const [f, text] of Object.entries(files)) fs.writeFileSync(path.join(dir, f), text);
  };
  const RUNTIME = { 'libsycl.so.9': 'sycl 7.2.0', 'libumf.so.1': 'umf', 'LICENSE-DPCPP.txt': 'apache', 'THIRD-PARTY-DPCPP.txt': 'notices' };
  const quantus = () => ({ ...RUNTIME, 'quantus_sycl.node': 'quantus', 'NOTICE.txt': 'the Quantus notice' });
  const pearl = () => ({ ...RUNTIME, 'pearl_core_sycl.node': 'pearl', 'libpearl_sycl.so': 'host', 'pearl_sycl_check': 'tool', 'NOTICE.txt': 'the Pearl notice' });
  const run = (...args) => spawnSync(process.execPath, [STAGE_SCRIPT, ...args], { encoding: 'utf8' });
  const list = (dir) => fs.readdirSync(dir).sort();
  beforeEach(() => { tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'quantus-stage-')); });
  afterEach(() => fs.rmSync(tmp, { recursive: true, force: true }));

  test('beside the Pearl Intel core: the addon and its notice go in, the runtime is the one already there', () => {
    const from = path.join(tmp, 'q', 'sycl');
    const into = path.join(tmp, 'native', 'sycl');
    put(from, quantus());
    put(into, pearl());
    const r = run(from, into);
    expect([r.status, r.stderr]).toEqual([0, '']);
    expect(r.stdout).toBe('staged quantus_sycl.node in ' + into + ' (2 files copied, 4 already there and the same)\n');
    expect(list(into)).toEqual(['LICENSE-DPCPP.txt', 'NOTICE-quantus.txt', 'NOTICE.txt', 'THIRD-PARTY-DPCPP.txt', 'libpearl_sycl.so',
      'libsycl.so.9', 'libumf.so.1', 'pearl_core_sycl.node', 'pearl_sycl_check', 'quantus_sycl.node']);
    expect(fs.readFileSync(path.join(into, 'NOTICE.txt'), 'utf8')).toBe('the Pearl notice');
    expect(fs.readFileSync(path.join(into, 'NOTICE-quantus.txt'), 'utf8')).toBe('the Quantus notice');
  });

  test('without the Pearl Intel core: the folder is made, with the runtime the addon needs', () => {
    const from = path.join(tmp, 'q', 'sycl');
    const into = path.join(tmp, 'native', 'sycl');
    put(from, quantus());
    fs.mkdirSync(path.join(from, 'a-folder'));
    const r = run(from, into);
    expect(r.status).toBe(0);
    expect(r.stdout).toBe('staged quantus_sycl.node in ' + into + ' (6 files copied)\n');
    expect(list(into)).toEqual(['LICENSE-DPCPP.txt', 'NOTICE-quantus.txt', 'THIRD-PARTY-DPCPP.txt', 'libsycl.so.9', 'libumf.so.1', 'quantus_sycl.node']);
  });

  test('a different runtime in the folder: nothing of the Quantus core is staged', () => {
    const from = path.join(tmp, 'q', 'sycl');
    const into = path.join(tmp, 'native', 'sycl');
    put(from, { ...quantus(), 'libsycl.so.9': 'sycl 7.3.0' });
    put(into, pearl());
    const before = list(into);
    const r = run(from, into);
    expect(r.status).toBe(1);
    expect(r.stderr).toContain('the Intel Quantus core was not staged: libsycl.so.9 in ' + from + ' differ from the copies in ' + into);
    expect(r.stderr).toContain('must be built with the same DPC++ release');
    expect(list(into)).toEqual(before);
  });

  test('nothing to stage is not an error: no folder, or no addon in it', () => {
    let r = run(path.join(tmp, 'none'), path.join(tmp, 'native', 'sycl'));
    expect(r.status).toBe(0);
    expect(r.stdout).toContain('no Intel Quantus core to stage');
    expect(fs.existsSync(path.join(tmp, 'native'))).toBe(false);
    put(path.join(tmp, 'q'), RUNTIME);
    r = run(path.join(tmp, 'q'), path.join(tmp, 'native', 'sycl'));
    expect(r.status).toBe(0);
    expect(fs.existsSync(path.join(tmp, 'native'))).toBe(false);
  });

  test('both folders are needed', () => {
    const r = run(path.join(tmp, 'q'));
    expect(r.status).toBe(2);
    expect(r.stderr).toContain('usage: node scripts/stage-quantus-sycl.mjs <from> <into>');
  });

  test('the folder it fills is the one the loader looks in', () => {
    expect(qc.SUBDIRS['quantus_sycl.node'][0]).toBe('sycl');
    expect(MINER_BUILD).toContain('node scripts/stage-quantus-sycl.mjs vendor/quantus-sycl/sycl vendor/native/sycl');
  });
});

describe('miner-build.yml packs the cores the native-core run built', () => {
  const STEPS = [
    ['Stage the NVIDIA Quantus core', 'quantus-core-cuda-', 'earn/vendor/native'],
    ['Stage the AMD Quantus core', 'quantus-core-hip-', 'earn/vendor/native'],
    ['Download the Intel Quantus core', 'quantus-core-sycl-', 'earn/vendor/quantus-sycl'],
  ];
  const step = (name) => slice(MINER_BUILD, '- name: ' + name + '\n', 'if_no_artifact_found: ignore');

  test('from the same native-core run as the Pearl cores, by the names native-core.yml uploads, optional and quiet', () => {
    for (const [name, artifact, into] of STEPS) {
      const s = step(name);
      expect([name, s]).toEqual([name, expect.stringContaining("name: " + artifact + "${{ runner.os == 'Windows' && 'windows-2022' || 'ubuntu-latest' }}\n")]);
      expect([name, s]).toEqual([name, expect.stringContaining('run_id: ${{ steps.pick.outputs.run_id }}\n')]);
      expect([name, s]).toEqual([name, expect.stringContaining('continue-on-error: true\n')]);
      expect([name, s]).toEqual([name, expect.stringContaining('path: ' + into + '\n')]);
      // Every artifact name it can ask for is one a job uploads.
      for (const os of ['ubuntu-latest', 'windows-2022']) {
        const uploaded = Object.values(JOBS).some((j) => j.artifact === artifact + os);
        // AMD has no Windows Quantus core yet: that download finds nothing, and is ignored.
        if (artifact + os !== 'quantus-core-hip-windows-2022') expect([artifact + os, uploaded]).toEqual([artifact + os, true]);
      }
    }
  });

  test('after the Pearl Intel core is staged, and before the installer, the CLI and the HiveOS package are built', () => {
    const at = (s) => {
      const i = MINER_BUILD.indexOf(s);
      expect([s, i >= 0]).toEqual([s, true]);
      return i;
    };
    const order = ['- name: Stage the Intel Pearl core', '- name: Stage the NVIDIA Quantus core', '- name: Stage the AMD Quantus core',
      '- name: Download the Intel Quantus core', '- name: Stage the Intel Quantus core', '- name: Verify the core is really there',
      '- name: Build installer', '- name: Package headless CLI binary (Linux)', '- name: Package HiveOS custom miner (Linux)'];
    for (let i = 1; i < order.length; i++) expect([order[i], at(order[i - 1]) < at(order[i])]).toEqual([order[i], true]);
    const merge = slice(MINER_BUILD, '- name: Stage the Intel Quantus core', 'rm -rf vendor/quantus-sycl');
    expect(merge).toContain('echo "::warning::$msg"');
  });

  test('the verify step lists each Quantus core, and a missing one is not a warning', () => {
    const verify = slice(MINER_BUILD, '- name: Verify the core is really there', '- name: Build the RandomX CPU worker');
    expect(verify).toContain('for f in quantus_cuda.node quantus_hip.node sycl/quantus_sycl.node; do');
    const quantus = slice(verify, '# The Quantus cores', '\n          done\n');
    expect(quantus).not.toContain('::warning::');
  });

  // electron-builder takes resources/native from vendor/native, through these filters. Both
  // installers and the AppImage, the same way.
  test('the installers and the AppImage carry them through the filters they already have', () => {
    const pkg = JSON.parse(read(EARN, 'package.json'));
    const native = [pkg.build.win, pkg.build.linux].map((p) => p.extraResources.find((r) => r.to === 'native'));
    const matches = (filter, file) => filter.some((f) => (f === '**/*.node' ? file.endsWith('.node')
      : f.endsWith('/**') ? file.startsWith(f.slice(0, -2)) : f === file));
    for (const r of native) {
      expect(r.from).toBe('vendor/native');
      for (const f of ['quantus_cuda.node', 'quantus_hip.node', 'sycl/quantus_sycl.node', 'sycl/NOTICE-quantus.txt']) {
        expect([f, matches(r.filter, f)]).toEqual([f, true]);
      }
    }
  });
});

describe('the CLI binary and the HiveOS package', () => {
  test('dist:cli puts the NVIDIA and AMD cores beside the binary, and sycl/ goes when either Intel core is in it', () => {
    expect(BUILD_CLI).toContain("for (const f of ['quantus_cuda.node', 'quantus_hip.node']) {");
    expect(BUILD_CLI).toContain("chmodSync(join(dist, f), 0o644);");
    // A stale copy from an earlier build is not left beside a new binary.
    expect(BUILD_CLI).toContain('rmSync(join(dist, f), { force: true });');
    expect(BUILD_CLI).toContain("if (pearlSycl || existsSync(join(sycl, 'quantus_sycl.node'))) {");
    // The release archive is the Pearl Intel core's, as before.
    expect(slice(BUILD_CLI, 'if (pearlSycl) {', '}')).toContain("execFileSync('tar', ['-czf', join(dist, 'pearl_core_sycl-linux.tar.gz')");
  });

  test('dist:hiveos packs them, with rocm/ for either AMD core and sycl/ for either Intel one', () => {
    expect(BUILD_HIVE).toContain("for (const f of ['quantus_cuda.node', 'quantus_hip.node']) {");
    expect(BUILD_HIVE).toContain('chmodSync(join(pkgDir, f), 0o644);');
    expect(BUILD_HIVE).toContain('const anyHip = existsSync(hip) || existsSync(quantusHip);');
    expect(BUILD_HIVE).toContain('if (pearlSycl || quantusSycl) {');
  });

  // Every place a package puts a core is one the loader tries, in each kind of package.
  test('every place they land is one src/main/quantusCore.js tries', () => {
    for (const [execPath, resourcesPath, dir] of [
      [path.join('/hive', 'llmjob-earn', 'llmjob-earn-cli-linux'), undefined, path.join('/hive', 'llmjob-earn')],
      [path.join('/opt', 'LLMJob Earn', 'earn'), path.join('/opt', 'LLMJob Earn', 'resources'), path.join('/opt', 'LLMJob Earn', 'resources', 'native')],
    ]) {
      const tried = [];
      qc.quantusCoreFactory({
        env: {}, execPath, resourcesPath, log: () => {},
        require: (f) => { tried.push(f); throw Object.assign(new Error('no'), { code: 'MODULE_NOT_FOUND' }); },
      });
      for (const f of [path.join(dir, 'quantus_cuda.node'), path.join(dir, 'quantus_hip.node'), path.join(dir, 'sycl', 'quantus_sycl.node')]) {
        expect([f, tried.includes(f)]).toEqual([f, true]);
      }
    }
  });

  test('the HiveOS smoke test loads each Quantus core on every image, with an address no one holds', () => {
    const qz = /\nqz=(\S+)\n/.exec(SMOKE)[1];
    expect(qz).toBe(encodeQuantusAddress(Buffer.alloc(32)));
    expect(isValidQuantusAddress(qz)).toBe(true);
    expect(SMOKE).toContain('configure "--region us --coin quantus --quantus-address $qz"');
    expect(SMOKE).toContain('for f in quantus_cuda.node quantus_hip.node sycl/quantus_sycl.node; do [ -f "$f" ] && quantus+=("$f"); done');
    // The loader's own words for a core that loaded and has no card, or would not load.
    expect(SMOKE).toContain('grep -qF "Quantus core: $d/$f found no card"');
    expect(SMOKE).toContain('grep -qF "Quantus core: $d/$f could not list its cards"');
    expect(SMOKE).toContain('if grep -qF "Quantus core: $d/$f would not load" <<< "$(run_log)"; then');
    const src = read(EARN, 'src', 'main', 'quantusCore.js');
    for (const words of ["' found no card'", "' could not list its cards: '", "' would not load: '", "'Quantus core: ' + w"]) {
      expect([words, src.includes(words)]).toEqual([words, true]);
    }
    expect(read(EARN, 'src', 'cli', 'earn-cli.js')).toContain("log('coin:       Quantus (QTC), experimental');");
    expect(SMOKE).toContain("grep -qF 'coin:       Quantus (QTC), experimental'");
    // Their libraries are checked like the Pearl cores', the AMD one with the same rocm/ rule.
    expect(SMOKE).toContain('for core in pearl_core.node pearl_core_cu13.node pearl_core_hip.node quantus_cuda.node quantus_hip.node; do');
    expect(SMOKE).toContain('case "$core" in *_hip.node) hip_core=yes ;; esac');
    expect(SMOKE).toContain('  for f in sycl/*.node sycl/*.so*; do');
  });
});
