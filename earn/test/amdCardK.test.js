'use strict';

// k 4096 on AMD (round 4 item 5), step 2: RDNA 4 mines k 4096 in the app, every other card k 2048. The
// gfx12 code objects have the wrap readout (PEARL_AMD_KWRAP) and say so in their caps; the core picks
// k 4096 for such a card when the miner lets it (cardK) and reports it as core.k; the miner gives each
// card its own profile (pearlMiner.test.js has that half). No HIP compiler runs here, so the native half
// is read out of the sources, as amdK4096.test.js does, and the probes run against stand-in cores whose
// hits are built in JS (helpers/pearlHit.js) and checked by the stand-in pool (probes/verify-pool.js).

const fs = require('fs');
const os = require('os');
const net = require('net');
const path = require('path');
const { spawnSync } = require('child_process');
const { PROFILE, profileForK, shareBound, penalizedAdjustmentFactor } = require('../src/shared/miner/pearlhash');
const { buildShareProof } = require('../src/shared/miner/shareProof');
const { serializePlainProof, decodePlainProof } = require('../src/shared/miner/plainProof');
const { hash } = require('../src/shared/miner/blake3');
const { buildConfig52 } = require('../src/shared/miner/pearlhash');
const { PearlMiner } = require('../src/main/pearlMiner');
const { hitFor } = require('./helpers/pearlHit');

const NATIVE = path.join(__dirname, '..', 'native');
const read = (...f) => fs.readFileSync(path.join(NATIVE, ...f), 'utf8').replace(/\r\n/g, '\n');
const CORE = read('src', 'pearl_core.cc');
const CUDA = read('src', 'pearl_host.cu');
const HOST = read('amd', 'src', 'pearl_host_hip.cpp');
const FOLD = read('amd', 'src', 'pearl_amd_fold.h');
const BUILD = read('amd', 'build-amd.sh');
const PROBES = path.join(NATIVE, 'probes');
const VERIFY_POOL = require(path.join(PROBES, 'verify-pool.js'));
const HELPER = path.join(__dirname, 'helpers', 'pearlHit.js');
const PEARLHASH = path.join(__dirname, '..', 'src', 'shared', 'miner', 'pearlhash.js');

const hasBash = spawnSync('bash', ['--version'], { stdio: 'ignore' }).status === 0;
const bashIt = hasBash ? it : it.skip;

// m and n small enough to build whole operands in JS.
const SMALL = { ...PROFILE, m: 64, n: 64 };
const MAX = (1n << 256n) - 1n;

function slice(text, from, to) {
  const i = text.indexOf(from);
  if (i < 0) throw new Error('not found: ' + from);
  const j = text.indexOf(to, i + from.length);
  return text.slice(i, j < 0 ? undefined : j + to.length);
}

describe('the host interface', () => {
  it('every host entry point the binding declares is defined by both hosts', () => {
    // Linux links the AMD addon with undefined symbols allowed, so one missing would fail only when called.
    const decl = slice(CORE, 'extern "C" {', '\n}\n');
    const names = [...decl.matchAll(/\b(pearl_host_\w+)\(/g)].map((m) => m[1]);
    expect(names).toContain('pearl_host_card_k');
    for (const n of new Set(names)) {
      expect([n, new RegExp('extern "C" [^;{(]*\\b' + n + '\\(').test(CUDA)]).toEqual([n, true]);
      expect([n, new RegExp('extern "C" [^;{(]*\\b' + n + '\\(').test(HOST)]).toEqual([n, true]);
    }
  });

  it('the CUDA host keeps every card at the profile\'s k', () => {
    const body = slice(CUDA, 'extern "C" uint32_t pearl_host_card_k(', '\n}\n');
    expect(body).toContain('return profile ? profile->k : 0;');
    expect(body).toContain("note[0] = '\\0';");
  });
});

describe('the binding (pearl_core.cc)', () => {
  const ctor = slice(CORE, 'PearlCore::PearlCore(const Napi::CallbackInfo &info)', '\n}\n');

  it('reads cardK, and asks the host for the card\'s k after choosing the card and before allocating', () => {
    expect(ctor).toContain('Napi::Value ck = o.Get("cardK");');
    expect(ctor).toContain('card_k = ck.IsBoolean() && ck.As<Napi::Boolean>().Value();');
    const at = (s) => {
      const i = ctor.indexOf(s);
      expect([s, i >= 0]).toEqual([s, true]);
      return i;
    };
    const select = at('pearl_host_select_device(&profile');
    const cardK = at('pearl_host_card_k(&profile, device_index_, note, sizeof(note))');
    const set = at('profile_ = profile;');
    const create = at('pearl_host_create_checked(&profile');
    // profile_ after the k is known: the hashrate counts PEARL_DAF(profile_) MACs an attempt.
    expect(select < cardK && cardK < set && set < create).toBe(true);
    expect(ctor.match(/profile_ = profile;/g)).toHaveLength(1);
    expect(ctor).toContain('if (card_k) {');
    expect(ctor).toContain('if (k) profile.k = k;');
  });

  it('reports the k it mines as core.k, and why not more as core.kNote', () => {
    expect(CORE).toContain('InstanceAccessor("k", &PearlCore::GetK, nullptr)');
    expect(CORE).toContain('InstanceAccessor("kNote", &PearlCore::GetKNote, nullptr)');
    expect(slice(CORE, 'Napi::Value PearlCore::GetK(', '\n}\n')).toContain('Napi::Number::New(info.Env(), profile_.k)');
    expect(slice(CORE, 'Napi::Value PearlCore::GetKNote(', '\n}\n')).toContain('Napi::String::New(info.Env(), k_note_)');
    // The rate is MACs at the core's own k.
    expect(CORE).toContain('win_work_ += (double)attempts * PEARL_DAF(profile_);');
  });
});

describe('the wrap readout in the AMD folds', () => {
  // The #ifndef PEARL_AMD_KWRAP block, evaluated as the preprocessor would for a compile pass.
  const block = slice(FOLD, '#ifndef PEARL_AMD_KWRAP\n#if ', '#endif\n#endif\n');
  function kwrap(pass) {
    const lines = block.split('\n').slice(1, -3);
    const cond = (c) => new Function('d', 'v', 'return ' + c
      .replace(/defined\((\w+)\)/g, 'd("$1")')
      .replace(/\b(PEARL_\w+)\b/g, 'v("$1")') + ';')((n) => !!pass[n], (n) => pass[n] || 0);
    let value = null;
    let taken = false;
    let on = false;
    for (const l of lines) {
      let m;
      if ((m = /^#if (.*)$/.exec(l))) { on = !!cond(m[1]); taken = on; }
      else if ((m = /^#elif (.*)$/.exec(l))) { on = !taken && !!cond(m[1]); taken = taken || on; }
      else if (l === '#else') { on = !taken; taken = true; }
      else if ((m = /^#define PEARL_AMD_KWRAP (\d)$/.exec(l))) { if (on) value = Number(m[1]); }
      else throw new Error('unexpected line: ' + l);
    }
    return value;
  }
  const device = (over) => ({ __HIP_DEVICE_COMPILE__: 1, PEARL_FOLD_CHUNKS: 16, PEARL_JACKPOT_BUCKETS: 16, ...over });

  it('is on in every gfx12 device pass, and in no other pass of a release build', () => {
    expect(kwrap(device({ __GFX12__: 1 }))).toBe(1);
    for (const other of [{ __GFX11__: 1 }, { __gfx942__: 1 }, { __gfx1030__: 1 }, {}]) expect(kwrap(device(other))).toBe(0);
    // A host pass never sees __GFX12__, and nor does HIP-CPU.
    expect(kwrap({ PEARL_FOLD_CHUNKS: 16, PEARL_JACKPOT_BUCKETS: 16 })).toBe(0);
    expect(kwrap(device({ __GFX12__: 1, PEARL_AMD_EMU: 1 }))).toBe(0);
  });

  it('is off in a readout-drop build, so its #errors never fire there, and on past 16 chunks everywhere', () => {
    expect(kwrap(device({ __GFX12__: 1, PEARL_AMD_READOUT_DROP: 1 }))).toBe(0);
    expect(kwrap(device({ __GFX11__: 1, PEARL_FOLD_CHUNKS: 32 }))).toBe(1);
    expect(kwrap({ PEARL_FOLD_CHUNKS: 64, PEARL_JACKPOT_BUCKETS: 16 })).toBe(1);
    // -DPEARL_AMD_KWRAP=0 or 1 sets it for every target: the block is skipped.
    expect(FOLD).toContain('#ifndef PEARL_AMD_KWRAP\n#if PEARL_FOLD_CHUNKS > PEARL_JACKPOT_BUCKETS');
  });

  it('the caps say a code object has it, on a bit of its own', () => {
    const bits = [...FOLD.matchAll(/#define PEARL_AMD_CAP_(\w+) (0x[0-9a-f]+|\d+)u/g)].map((m) => [m[1], Number(m[2])]);
    const kw = bits.find(([n]) => n === 'KWRAP')[1];
    expect(kw).toBe(0x400);
    for (const [n, v] of bits) if (n !== 'KWRAP') expect([n, v & kw]).toEqual([n, 0]);
    const caps = slice(FOLD, 'extern "C" __global__ void pearl_amd_caps(uint32_t *out) {', '\n}\n');
    expect(caps).toContain('if (PEARL_AMD_KWRAP) out[0] |= PEARL_AMD_CAP_KWRAP;');
  });
});

describe('the AMD host (pearl_host_hip.cpp)', () => {
  const resolve = slice(HOST, 'void resolve_fold(Ctx *ctx) {', '\n}\n');
  const cardK = slice(HOST, 'extern "C" uint32_t pearl_host_card_k(', '\n}\n');

  it('reads the card\'s wrap readout off its caps, and refuses past 16 chunks without it, at create', () => {
    expect(resolve).toContain('ctx->kwrap = (caps[0] & PEARL_AMD_CAP_KWRAP) != 0;');
    expect(resolve).toContain('if (!ctx->kwrap && rank && ctx->profile.k / rank > PEARL_JACKPOT_BUCKETS) {');
    expect(resolve).toContain('k %u needs the wrap readout');
    // The host pass never sees __GFX12__, so nothing in the host decides by the macro.
    const code = HOST.split('\n').filter((l) => !/^\s*\/\//.test(l)).join('\n');
    expect(code).not.toMatch(/\bPEARL_AMD_KWRAP\b(?!_CHUNKS)/);
  });

  it('moves only the mainnet k, at rank 128, and only on a card whose code has the wrap readout', () => {
    const lines = cardK.split('\n').map((l) => l.trim());
    const at = (s) => {
      const i = lines.indexOf(s);
      expect([s, i >= 0]).toEqual([s, true]);
      return i;
    };
    const mainnet = at('if (k != 2048u || profile->rank != 128u) return k;');
    const caps = at('if (e == hipSuccess) e = read_caps(nullptr, dCaps, caps);');
    const cap = at('if (e != hipSuccess || !(caps[0] & PEARL_AMD_CAP_KWRAP) || (caps[0] & PEARL_AMD_CAP_DROP)) return k;');
    const pin = at('if (pin && !strcmp(pin, "2048")) {');
    const vram = at('if (known && freeMem < need) {');
    const yes = at('return 4096u;');
    expect(mainnet < caps && caps < cap && cap < pin && pin < vram && vram < yes).toBe(true);
    expect(lines).toContain('const char *pin = getenv("PEARL_AMD_K");');
    expect(lines).toContain('wide.k = 4096u;');
    expect(lines).toContain('const size_t need = needed_bytes(&wide);');
    // Every way out but the last keeps the profile's k.
    expect(lines.filter((l) => /return (4096u|k|0);/.test(l))).toHaveLength(6);
    // Why it stayed, for the log.
    expect(cardK).toContain('"k 2048 (PEARL_AMD_K=2048)"');
    expect(cardK).toContain('"k 2048: k 4096 needs ~%zu MiB and %zu MiB is free"');
  });

  it('names the k and the readout in the fold\'s description', () => {
    const name = slice(HOST, 'extern "C" const char *pearl_host_fold_name(void *handle) {', '\n}\n');
    expect(name).toContain('"%s, batch %u column offsets, k %u%s%s%s%s"');
    expect(name).toContain('ctx->kwrap ? ", wrap readout" : ""');
  });

  it('the README\'s and config\'s VRAM figures are needed_bytes at k 2048 and k 4096', () => {
    // needed_bytes, for the mainnet profile with the constant fill: operands primed, noise, the batch hit
    // lists, the trees, the two transcript buffers (256 column offsets), and 1 MiB.
    const need = (k) => {
      const { m, n, rank } = PROFILE;
      const a = m * k, b = n * k;
      const noise = m * rank + n * rank + 2 * k * 2 * 4 + 64;
      const batch = 2 * 64 * (32 + 8 + 16 * 4);
      const tree = 2 * (a / 1024 + b / 1024) * 32 + Math.max(a, b) / 1024 * 32;
      const tr = 2 * 256 * (m / 16) * 16 * 4;
      return Math.floor((2048 + 2048 + a + b + noise + batch + tree + tr + (1 << 20)) / (1 << 20));
    };
    expect([need(2048), need(4096)]).toEqual([1137, 1969]);
    const readme = read('amd', 'README.md');
    expect(readme).toContain('about 1,969 MiB against 1,137');
  });
});

describe('build-amd.sh', () => {
  it('builds kwrap twins of the CPU core and fold check', () => {
    expect(BUILD).toContain('$cxx $flags -DPEARL_AMD_KWRAP=1 $inc -c "$HERE/src/pearl_host_hip.cpp" -o "$OUT/cpu/pearl_host_hip_kwrap.o"');
    expect(BUILD).toContain('-o "$OUT/pearl_core_hipcpu_kwrap.node"');
    expect(BUILD).toContain('-o "$OUT/pearl_amd_foldtest_cpu_kwrap"');
  });

  it('builds the Windows library without the wrap readout, before any tuning flags', () => {
    expect(BUILD).toContain('if windows; then KWRAP_OS=-DPEARL_AMD_KWRAP=0; fi');
    expect(BUILD).toContain('-D_CRT_SECURE_NO_WARNINGS -Wno-unused-result -Wno-unused-value $KWRAP_OS ${PEARL_AMD_CFLAGS:-}"');
  });

  // A stand-in for hip-clang: -E -dM gives the device pass's PEARL_AMD_KWRAP (1 on gfx12, unless FLIP names the
  // target; the last -DPEARL_AMD_KWRAP=N sets it, as in clang), with CR LF line ends when CRLF is set (as clang on
  // Windows may write them), and -c gives the resource remarks of the folds a target has, clean: the reference
  // fold and the one the host runs there (dot4 on gfx103x), less the one DROP names (target:fold), which shows 0
  // VGPRs as a fold without a body does.
  const CC = [
    '#!/bin/bash',
    'arch=; e=0; set=',
    'for a in "$@"; do case $a in --offload-arch=*) arch=${a#*=} ;; -E) e=1 ;; -DPEARL_AMD_KWRAP=*) set=${a#*=} ;; esac; done',
    'if [ $e = 1 ]; then',
    '  case $arch in gfx12*) kw=1 ;; *) kw=0 ;; esac',
    '  [ -n "$set" ] && kw=$set',
    '  [ "${FLIP:-}" = "$arch" ] && kw=$((1 - kw))',
    '  cr=; [ -n "${CRLF:-}" ] && cr=$\'\\r\'',
    '  printf "#define PEARL_AMD_TARGETS x%s\\n#define PEARL_AMD_KWRAP %s%s\\n" "$cr" "$kw" "$cr"; exit 0',
    'fi',
    'r() { echo "x.cpp:1:1: remark: $1 [-Rpass-analysis=kernel-resource-usage]"; }',
    'k() { v=64; [ "${DROP:-}" = "$arch:$1" ] && v=0',
    '  r "Function Name: pearl_amd_fold_$1"; r "    VGPRs: $v"; r "    AGPRs: 0"; r "    ScratchSize [bytes/lane]: 0"',
    '  r "    VGPRs Spill: 0"; r "    SGPRs Spill: 0"; r "    Occupancy [waves/SIMD]: 8"; r "    LDS Size [bytes/block]: 1024"; }',
    'k ref',
    'case $arch in gfx103*) k dot4 ;; gfx11*) k wmma11 ;; gfx12*) k wmma12 ;; esac',
    '',
  ].join('\n');
  let tmp;
  beforeEach(() => {
    tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'amd-check-'));
    fs.mkdirSync(path.join(tmp, 'amd', 'src'), { recursive: true });
    fs.mkdirSync(path.join(tmp, 'src'));
    fs.writeFileSync(path.join(tmp, 'amd', 'build-amd.sh'), BUILD);
    fs.writeFileSync(path.join(tmp, 'amd', 'src', 'pearl_host_hip.cpp'), '');
    fs.writeFileSync(path.join(tmp, 'src', 'pearl_kernel.cu'), 'a\n// A 16-byte global->shared copy that does not pass through registers\n');
    for (const p of [['lib', 'llvm', 'bin', 'clang++'], ['bin', 'clang++.exe']]) {
      fs.mkdirSync(path.join(tmp, 'rocm', ...p.slice(0, -1)), { recursive: true });
      fs.writeFileSync(path.join(tmp, 'rocm', ...p), CC, { mode: 0o755 });
    }
    // uname says which system build-amd.sh is on (Linux, or Git Bash on Windows), so both run on any host;
    // cygpath -m only has to hand the path back.
    fs.mkdirSync(path.join(tmp, 'bin'));
    fs.writeFileSync(path.join(tmp, 'bin', 'uname'), '#!/bin/bash\necho "$FAKE_UNAME"\n', { mode: 0o755 });
    fs.writeFileSync(path.join(tmp, 'bin', 'cygpath'), '#!/bin/bash\nfor a; do p=$a; done; echo "$p"\n', { mode: 0o755 });
  });
  afterEach(() => fs.rmSync(tmp, { recursive: true, force: true }));

  const WINDOWS = 'MINGW64_NT-10.0-22631';
  function check(env = {}, uname = 'Linux', archs = 'gfx1100 gfx1201') {
    const rocm = path.join(tmp, 'rocm').replace(/\\/g, '/');
    const e = { ...process.env, ROCM_PATH: rocm, HIP_PATH: rocm, FAKE_BIN: path.join(tmp, 'bin'), FAKE_UNAME: uname, ...env };
    if (!('PEARL_AMD_CFLAGS' in env)) delete e.PEARL_AMD_CFLAGS;
    // The real cygpath (Git Bash) turns the folder into a PATH entry; Linux has none and needs none.
    const run = 'PATH="$(cygpath -u "$FAKE_BIN" 2>/dev/null || echo "$FAKE_BIN"):$PATH" exec bash build-amd.sh check ' + archs;
    const r = spawnSync('bash', ['-c', run], { cwd: path.join(tmp, 'amd'), encoding: 'utf8', env: e });
    return { status: r.status, out: r.stdout, err: r.stderr };
  }

  bashIt('passes a release build whose gfx12 targets, and only those, have the wrap readout', () => {
    // CRLF: the Windows job's compiler may end its -E lines with CR LF, which must still read as 0 and 1.
    for (const env of [{}, { CRLF: '1' }]) {
      const r = check(env);
      expect([env, r.status, r.err]).toEqual([env, 0, '']);
      expect(r.out).toMatch(/gfx1100 +pearl_amd_fold_ref +64 +0 +0 +0 +1024 +8 +0\n/);
      expect(r.out).toMatch(/gfx1201 +pearl_amd_fold_ref +64 +0 +0 +0 +1024 +8 +1\n/);
      expect(r.out).toContain('and the wrap readout on exactly the gfx12 targets');
    }
  }, 30000);

  bashIt('fails one where a gfx12 target lacks it, or another target has it', () => {
    for (const [flip, line] of [
      ['gfx1201', 'gfx1201: the wrap readout (PEARL_AMD_KWRAP) is 0, not 1: RDNA 4 (gfx12) has it'],
      ['gfx1100', 'gfx1100: the wrap readout (PEARL_AMD_KWRAP) is 1, not 0: RDNA 4 (gfx12) has it'],
    ]) {
      const r = check({ FLIP: flip });
      expect(r.status).toBe(1);
      expect(r.err).toContain(line);
      expect(r.err).toContain('check: the wrap readout should be on exactly the gfx12 targets');
      expect(r.err).not.toContain('spills');
    }
  }, 30000);

  bashIt('passes a Windows build only with the wrap readout on no target, gfx12 included', () => {
    // The Windows core mines k 2048 on every card: build-amd.sh sets -DPEARL_AMD_KWRAP=0 there, which the
    // stand-in honours, so gfx1201 shows 0.
    for (const env of [{}, { CRLF: '1' }]) {
      const r = check(env, WINDOWS);
      expect([env, r.status, r.err]).toEqual([env, 0, '']);
      expect(r.out).toMatch(/gfx1100 +pearl_amd_fold_ref +64 +0 +0 +0 +1024 +8 +0\n/);
      expect(r.out).toMatch(/gfx1201 +pearl_amd_fold_ref +64 +0 +0 +0 +1024 +8 +0\n/);
      expect(r.out).toContain('and the wrap readout on no target (the Windows core mines k 2048)');
    }
    // A Windows build where a target still has it fails.
    const r = check({ FLIP: 'gfx1201' }, WINDOWS);
    expect(r.status).toBe(1);
    expect(r.err).toContain('gfx1201: the wrap readout (PEARL_AMD_KWRAP) is 1, not 0: the Windows core mines k 2048 on every card');
    expect(r.err).toContain('check: the wrap readout should be on no target (the Windows core mines k 2048)');
  }, 30000);

  bashIt('only shows it for a tuning build, which may set it either way', () => {
    for (const [uname, cflags, kw] of [['Linux', '-DPEARL_AMD_KWRAP=0', 0], [WINDOWS, '-DPEARL_AMD_KWRAP=1', 1]]) {
      const r = check({ PEARL_AMD_CFLAGS: cflags }, uname);
      expect([uname, r.status, r.err]).toEqual([uname, 0, '']);
      expect(r.out).toMatch(new RegExp(`gfx1201 +pearl_amd_fold_ref +64 +0 +0 +0 +1024 +8 +${kw}\n`));
      expect(r.out).not.toContain('and the wrap readout');
    }
  }, 30000);

  bashIt('fails a release build where a target lacks the fold the host runs on it (RDNA 2: dot4)', () => {
    const ARCHS = 'gfx1030 gfx1100 gfx1201';
    const r0 = check({}, 'Linux', ARCHS);
    expect([r0.status, r0.err]).toEqual([0, '']);
    for (const row of ['gfx1030 +pearl_amd_fold_dot4 ', 'gfx1100 +pearl_amd_fold_wmma11 ', 'gfx1201 +pearl_amd_fold_wmma12 '])
      expect(r0.out).toMatch(new RegExp(row + '+64 '));
    expect(r0.out).toContain('every target has its fold');
    for (const drop of ['gfx1030:dot4', 'gfx1201:wmma12']) {
      const [arch, fold] = drop.split(':');
      const r = check({ DROP: drop }, 'Linux', ARCHS);
      expect([drop, r.status]).toEqual([drop, 1]);
      expect(r.err).toContain(`${arch}: no body for pearl_amd_fold_${fold}\n`);
      expect(r.err).toContain('check: a target lacks the fold the host runs on it (folds_of in build-amd.sh)');
      expect(r.err).not.toContain('spills');
      // A fold without a body has no row.
      expect(r.out.split('\n').filter((l) => l.startsWith(arch) && l.includes(`_${fold} `))).toEqual([]);
      // A tuning build (an ablation build has no dot4) only shows the rows.
      const t = check({ DROP: drop, PEARL_AMD_CFLAGS: '-DPEARL_AMD_TILED=0' }, 'Linux', ARCHS);
      expect([drop, t.status, t.err]).toEqual([drop, 0, '']);
    }
  }, 60000);

  bashIt('every changed script parses', () => {
    for (const f of [['amd', 'build-amd.sh'], ['amd', 'test', 'run-cpu-checks.sh'], ['amd', 'test', 'sweep.sh']]) {
      const r = spawnSync('bash', ['-n', path.join(NATIVE, ...f)], { encoding: 'utf8' });
      expect([f.join('/'), r.status, r.stderr]).toEqual([f.join('/'), 0, '']);
    }
  });
});

// A stand-in addon whose cores build their hits in JS: a core opened with cardK reports k 4096 (STUB_K), any
// other the profile's k. On setJob it finds the first region that meets the bound it was given and reports it,
// STUB_SALTS times as if from that many draws. At target 0 (hashrate.js) it only reports a rate.
const STUB_ADDON = `
const { EventEmitter } = require('events');
const { hitFor } = require(${JSON.stringify(HELPER)});
const { profileForK } = require(${JSON.stringify(PEARLHASH)});
module.exports = { createCore(profile, opts = {}) {
  const c = new EventEmitter();
  c.k = opts.cardK ? (Number(process.env.STUB_K) || 4096) : profile.k;
  c.device = { index: opts.deviceIndex || 0, name: 'stand-in', fold: 'js' };
  const p = profileForK(profile, c.k) || profile;
  let t = null;
  c.setJob = (job) => setImmediate(() => {
    clearInterval(t);
    t = setInterval(() => c.emit('hashrate', 1), 20);
    if (job.target === 0n) return;
    const h = hitFor(p, job.header, { bound: job.target, jobId: job.jobId });
    for (let s = 0; h && s < Number(process.env.STUB_SALTS || 1); s++) c.emit('hit', { ...h, salt: (opts.saltBase || 0) + s });
  });
  c.stop = () => clearInterval(t);
  return c;
} };
`;

function stubDir() {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'amd-cardk-'));
  fs.writeFileSync(path.join(tmp, 'stub.js'), STUB_ADDON);
  return tmp;
}

describe('the probes, opened as the app opens a core', () => {
  let tmp;
  beforeAll(() => { tmp = stubDir(); });
  afterAll(() => fs.rmSync(tmp, { recursive: true, force: true }));
  const run = (script, args, env = {}) => {
    const e = { ...process.env, ...env };
    for (const k of ['PEARL_PROFILE_K', 'PEARL_CARD_K', 'PEARL_SHARE_DUMP', 'PEARL_JOB_EVERY']) if (!(k in env)) delete e[k];
    const r = spawnSync(process.execPath, [script, path.join(tmp, 'stub.js'), ...args], { env: e, encoding: 'utf8' });
    return { status: r.status, json: JSON.parse(r.stdout.trim().split('\n').pop()), err: r.stderr };
  };

  it('hashrate.js reports the core\'s k, and with PEARL_JOB_EVERY gives it a new job that often', () => {
    const r = run(path.join(PROBES, 'hashrate.js'), ['0.5', '0'], { PEARL_JOB_EVERY: '0.1' });
    expect(r.status).toBe(0);
    expect(r.json.k).toBe(4096);
    expect(r.json.jobEvery).toBe(0.1);
    expect(r.json.jobs).toBeGreaterThanOrEqual(3);
    const pinned = run(path.join(PROBES, 'hashrate.js'), ['0.3', '0'], { PEARL_PROFILE_K: '2048' });
    expect([pinned.json.k, pinned.json.jobs]).toEqual([2048, undefined]);
    const bad = spawnSync(process.execPath, [path.join(PROBES, 'hashrate.js'), path.join(tmp, 'stub.js'), '0.1', '0'],
      { env: { ...process.env, PEARL_JOB_EVERY: 'soon' }, encoding: 'utf8' });
    expect(bad.status).toBe(2);
    expect(bad.stderr).toContain('PEARL_JOB_EVERY must be a number of seconds, not soon');
  }, 30000);

  it('verify-hits.js refuses a k the core chose that a share can\'t carry', () => {
    const r = run(path.join(PROBES, 'verify-hits.js'), ['0.3'], { STUB_K: '3072' });
    expect(r.status).toBe(2);
    expect(r.json.error).toBe("the core chose k 3072, which a share can't carry");
  }, 30000);

  it('verify-hits-profile.js with PEARL_CARD_K checks every hit at the core\'s k 4096, and dumps shares a pool takes', () => {
    const dump = path.join(tmp, 'shares.jsonl');
    const r = run(path.join(NATIVE, 'amd', 'test', 'verify-hits-profile.js'), ['1', '256', '64', '64', '1'],
      { PEARL_CARD_K: '1', PEARL_SHARE_DUMP: dump, STUB_SALTS: '20' });
    expect(r.json).toMatchObject({ k: 4096, m: 64, n: 64, verified: 20, failed: 0, salts: 20, PASS: true });
    const shares = fs.readFileSync(dump, 'utf8').trim().split('\n').map((l) => JSON.parse(l));
    expect(shares).toHaveLength(20);
    const s = shares[0];
    expect(s.k).toBe(4096);
    // The pool's target: the bound (2^256) over the k 4096 scaling.
    expect(BigInt('0x' + s.target)).toBe((1n << 256n) / BigInt(penalizedAdjustmentFactor(profileForK({ ...PROFILE, m: 64, n: 64 }, 4096))));
    expect(VERIFY_POOL.checkShare(Buffer.from(s.header, 'hex'), BigInt('0x' + s.target), s.plain_proof)).toEqual({ k: 4096, why: null });
    // Without PEARL_CARD_K the core is asked for nothing, and stays at the profile's k.
    const plain = run(path.join(NATIVE, 'amd', 'test', 'verify-hits-profile.js'), ['1', '256', '64', '64', '1'], { STUB_SALTS: '20' });
    expect(plain.json).toMatchObject({ k: 2048, verified: 20, PASS: true });
  }, 60000);
});

describe('the stand-in pool (verify-pool.js)', () => {
  const header = Buffer.alloc(76, 0x5A);
  const K4 = profileForK(SMALL, 4096);
  const target = MAX / BigInt(penalizedAdjustmentFactor(K4));
  const field = (p, hit, gzip) => buildShareProof(hit, hash(Buffer.concat([header, buildConfig52(p)])), p, { gzip });

  it('takes a genuine share at k 2048 and at k 4096, plain or compressed', () => {
    for (const p of [SMALL, K4]) {
      const hit = hitFor(p, header);
      for (const gzip of [false, true]) {
        expect([p.k, gzip, VERIFY_POOL.checkShare(header, target, field(p, hit, gzip), { gzip })]).toEqual([p.k, gzip, { k: p.k, why: null }]);
      }
    }
  });

  it('refuses the same proof with its k changed, under another header, or over the target', () => {
    const hit = hitFor(K4, header);
    const pp = decodePlainProof(field(K4, hit, false));
    const as = (o) => serializePlainProof({ ...pp, ...o }).toString('base64');
    expect(VERIFY_POOL.checkShare(header, target, as({ k: 2048 })).why)
      .toBe('A Merkle tree declares 256 leaves but m=64 k=2048 imply 128');
    expect(VERIFY_POOL.checkShare(Buffer.alloc(76, 1), target, as({})).why).toMatch(/^Hash A mismatch/);
    expect(VERIFY_POOL.checkShare(header, 1n, as({})).why).toMatch(/^Jackpot condition not satisfied/);
    expect(VERIFY_POOL.checkShare(header, target, as({ noiseRank: 96 })).why).toMatch(/^Rank must be/);
    expect(VERIFY_POOL.checkShare(header, target, as({ k: 1024 })).why).toMatch(/^k must be >= 16r/);
    expect(VERIFY_POOL.checkShare(header, target, 'AAAA').why).toMatch(/ends early/);
    // Rows that are no pattern at a valid offset.
    expect(VERIFY_POOL.checkShare(header, target, as({ a: { ...pp.a, rowIndices: pp.a.rowIndices.slice().reverse() } })).why)
      .toBe('pattern parsing error: Indices not strictly increasing');
    expect(VERIFY_POOL.checkShare(header, target, as({ a: { ...pp.a, rowIndices: pp.a.rowIndices.map((r) => r + 1) } })).why)
      .toBe('pattern parsing error: offset 1 is not valid for pattern');
  });

  // PearlMiner over a real socket: a card at k 4096 and a card at k 2048 on one connection, each share checked
  // in full by the stand-in.
  for (const plain of [false, true]) {
    it(`a mixed rig's shares at both k are accepted on one ${plain ? 'plain' : 'compressed'} connection`, async () => {
      const pool = VERIFY_POOL.servePool({ port: 0, target, plain, jobEvery: 0 });
      await new Promise((res) => pool.server.on('listening', res));
      const tmp = stubDir();
      const addon = require(path.join(tmp, 'stub.js'));
      const made = [];
      const m = new PearlMiner({
        connect: (host, port) => net.connect({ host, port }),
        createCore: (p, o) => { const c = addon.createCore(p, o.deviceIndex === 1 ? { ...o, cardK: false } : o); made.push(c); return c; },
      });
      const verdicts = [];
      m.on('share', (e) => verdicts.push(['ok', e.index]));
      m.on('rejected', (e) => verdicts.push(['rej', e.index, e.reason]));
      m.start({
        endpoint: '127.0.0.1:' + pool.server.address().port, address: 'prl1pstandin', worker: 'w',
        gpus: [{ index: 0, name: 'a' }, { index: 1, name: 'b' }], profile: SMALL,
      });
      for (let i = 0; i < 200 && verdicts.length < 2; i++) await new Promise((res) => setTimeout(res, 50));
      m.stop();
      pool.close();
      fs.rmSync(tmp, { recursive: true, force: true });
      expect(made.map((c) => c.k)).toEqual([4096, 2048]);
      expect(verdicts.sort()).toEqual([['ok', 0], ['ok', 1]]);
      expect(pool.stats.byK).toEqual({ 2048: { shares: 1, accepted: 1, rejected: 0 }, 4096: { shares: 1, accepted: 1, rejected: 0 } });
    }, 30000);
  }

  it('mixed mode runs one card at both k through PearlMiner, and passes when the pool takes both', () => {
    const tmp = stubDir();
    const r = spawnSync(process.execPath, [path.join(PROBES, 'verify-pool.js'), 'mixed', path.join(tmp, 'stub.js'), '3', '235'],
      { env: { ...process.env, PEARL_PROFILE_MN: '64x64', PEARL_JOB_EVERY: '0' }, encoding: 'utf8' });
    fs.rmSync(tmp, { recursive: true, force: true });
    const res = JSON.parse(r.stdout.trim().split('\n').pop());
    expect(res.cores.map((c) => c.k)).toEqual([4096, 2048]);
    expect(res.byK).toEqual({
      2048: { sent: 1, accepted: 1, rejected: 0, reasons: {} }, 4096: { sent: 1, accepted: 1, rejected: 0, reasons: {} },
    });
    expect([res.pool, res.unanswered, res.PASS, r.status]).toEqual(['stand-in', 0, true, 0]);
    // The wallet is never printed.
    expect(r.stdout + r.stderr).not.toContain('prl1pstandin');
  }, 60000);

  it('serve mode sums up what it saw when its time is up, and fails a run that sent nothing', () => {
    // Port 0 is no port a CLI could be given: this only shows the end of a run.
    const r = spawnSync(process.execPath, [path.join(PROBES, 'verify-pool.js'), 'serve', '0', '1', '235', 'plain'],
      { encoding: 'utf8' });
    expect(JSON.parse(r.stdout.trim())).toMatchObject({ mode: 'serve', bits: 235, plain: true, connections: 0, PASS: false });
    expect(r.stderr).toContain('raw proofs only, target 2^235');
    expect(r.status).toBe(1);
  }, 30000);
});

describe('shareBound at each k', () => {
  it('scales the same pool target twice as far for a k 4096 card', () => {
    const t = 1n << 200n;
    expect(shareBound(t, profileForK(SMALL, 4096))).toBe(2n * shareBound(t, SMALL));
  });
});
