'use strict';

// RX 7000 cards (gfx1100, gfx1101) mine a smaller m than mainnet's, so A' stays in the card's Infinity Cache
// (goal/f-amdasm2, 20 November). Their code objects fold strips and say so in their caps; the host picks the
// m when the miner lets the core choose (cardK) and folds as many batches a launch as m is smaller; the core
// reports the m as core.m, and the miner builds that card's proofs from it. No HIP compiler runs here, so the
// native half is read out of the sources, as amdCardK.test.js does.

const fs = require('fs');
const path = require('path');
const { PROFILE, profileForCore, buildConfig52 } = require('../src/shared/miner/pearlhash');

const NATIVE = path.join(__dirname, '..', 'native');
const read = (...f) => fs.readFileSync(path.join(NATIVE, ...f), 'utf8').replace(/\r\n/g, '\n');
const CORE = read('src', 'pearl_core.cc');
const CUDA = read('src', 'pearl_host.cu');
const HOST = read('amd', 'src', 'pearl_host_hip.cpp');
const FOLD = read('amd', 'src', 'pearl_amd_fold.h');

function slice(text, from, to) {
  const i = text.indexOf(from);
  if (i < 0) throw new Error('not found: ' + from);
  const j = text.indexOf(to, i + from.length);
  if (j < 0) throw new Error('not found after ' + from + ': ' + to);
  return text.slice(i, j);
}

describe('strips in the fold (PEARL_AMD_STRIP)', () => {
  it('are on for gfx1100, gfx1101, gfx1200 and gfx1201 alone, so every other target keeps its code', () => {
    // PEARL_AMD_GFX1100_1101 is always defined (0 or 1), so the gate tests its value, not defined().
    expect(FOLD).toContain('#ifndef PEARL_AMD_STRIP\n#if PEARL_AMD_GFX1100_1101 || (defined(__HIP_DEVICE_COMPILE__) && '
      + '(defined(__gfx1200__) || defined(__gfx1201__)))\n#define PEARL_AMD_STRIP 1\n'
      + '#else\n#define PEARL_AMD_STRIP 0\n#endif\n#endif\n');
    expect(FOLD).toContain('#if defined(__gfx1100__) || defined(__gfx1101__)\n#define PEARL_AMD_GFX1100_1101 1\n#endif');
    expect(FOLD).toContain('#ifndef PEARL_AMD_GFX1100_1101\n#define PEARL_AMD_GFX1100_1101 0\n#endif');
    expect(FOLD).not.toContain('#if defined(PEARL_AMD_GFX1100_1101)');
    expect(FOLD).toContain('#define PEARL_AMD_STRIP_ON (PEARL_AMD_STRIP != 0)');
  });

  it('start each strip where the one before it ended, and leave a launch of one strip as it was', () => {
    expect(FOLD).toContain('#elif PEARL_AMD_STRIP_ON\n');
    expect(FOLD).toContain('  const uint32_t tileCol = (blockIdx.z * gridDim.x + blockIdx.x) * (uint32_t)BN;\n');
    expect(FOLD).toContain('  const uint32_t tileCol = blockIdx.x * (uint32_t)BN;');
  });

  it('take the walk inside each strip on RDNA 4, offset as the plain strips are', () => {
    const walk = slice(FOLD, '#if PEARL_AMD_WALK_ON\n  uint32_t tileX, tileY;', '#elif PEARL_AMD_STRIP_ON\n');
    expect(walk).toContain('pearl_amd_walk(blockIdx.x, blockIdx.y, gridDim.x, gridDim.y, tileX, tileY);');
    expect(walk).toContain('#if PEARL_AMD_STRIP_ON\n');
    expect(walk).toContain('  const uint32_t tileCol = (blockIdx.z * gridDim.x + tileX) * (uint32_t)BN;\n#else\n'
      + '  const uint32_t tileCol = tileX * (uint32_t)BN;\n#endif\n');
  });

  it('are named in the code object\'s caps, with a bit no other cap uses', () => {
    const bits = [...FOLD.matchAll(/#define PEARL_AMD_CAP_(\w+) (0x[0-9a-f]+|\d+)u/g)].map((m) => [m[1], Number(m[2])]);
    const v = Object.fromEntries(bits);
    expect(v.STRIP).toBe(0x40000);
    for (const [n, x] of bits) if (n !== 'STRIP') expect([n, x & v.STRIP]).toEqual([n, 0]);
    expect(slice(FOLD, 'extern "C" __global__ void pearl_amd_caps(uint32_t *out) {', '\n}\n'))
      .toContain('  if (PEARL_AMD_STRIP_ON) out[0] |= PEARL_AMD_CAP_STRIP;\n');
  });
});

describe('the m a card mines (pearl_host_card_m)', () => {
  const cardM = slice(HOST, 'extern "C" uint32_t pearl_host_card_m(const PearlProfile *profile, int device) {', '\n}\n');
  const strips = slice(HOST, 'uint32_t amd_card_strips(const PearlProfile *profile, const char *arch) {', '\n}\n');

  it('moves only the mainnet shape, on RDNA 3 and 4, where the code object folds strips', () => {
    expect(HOST).toContain('#ifndef PEARL_AMD_CARD_M\n#define PEARL_AMD_CARD_M 16384u\n#endif\n');
    expect(cardM).toContain('if (m != PEARL_MAINNET_PROFILE.m || profile->rank != 128u) return m;');
    expect(cardM).toContain('const bool rdna3 = arch_is(arch, "gfx1100") || arch_is(arch, "gfx1101");');
    expect(cardM).toContain('if (rdna3 && profile->k == 2048u) want = PEARL_AMD_CARD_M;\n'
      + '  else if (!rdna3) want = amd_rdna4_m(arch, profile->k);\n  if (!want) return m;');
    expect(cardM).toContain('if (e != hipSuccess || !(caps[0] & PEARL_AMD_CAP_STRIP) || (caps[0] & PEARL_AMD_CAP_DROP)) return m;');
    expect(cardM.trimEnd().endsWith('return want;')).toBe(true);
  });

  it('takes PEARL_AMD_M, and only an m whose commitment tree is a power of two of chunks', () => {
    expect(cardM).toContain('pearl_env_u32("PEARL_AMD_M", &asked)');
    const ms = [...cardM.matchAll(/asked == (\d+)u/g)].map((m) => Number(m[1]));
    expect(ms).toEqual([131072, 65536, 32768, 16384, 8192, 4096, 2048]);
    for (const m of ms) {
      for (const k of [2048, 4096, 8192, 16384]) {
        const chunks = (m * k) / 1024;
        expect(Number.isInteger(chunks) && (chunks & (chunks - 1)) === 0).toBe(true);
      }
      expect(m % 256).toBe(0);   // whole 256 x 256 tiles, and whole 16-row regions
    }
    // Mainnet's m asked for is no change.
    expect(cardM).toContain('if (want >= m) return m;');
  });

  it('on RDNA 4, picks the largest m whose A\' fits half the Infinity Cache', () => {
    const rdna4 = slice(HOST, 'static uint32_t amd_rdna4_m(const char *arch, uint32_t k) {', '\n}\n');
    const bytes = (a) => Number(new RegExp('#define PEARL_AMD_RDNA4_A_BYTES_' + a + ' \\((\\d+)u << 20\\)').exec(HOST)[1]) * 2 ** 20;
    expect(bytes(1201)).toBe(32 * 2 ** 20);   // RX 9070 XT: 64 MB
    expect(bytes(1200)).toBe(16 * 2 ** 20);   // RX 9060 XT: 32 MB
    expect(rdna4).toContain('if (!budget || k < 4096u || (k & (k - 1u)) != 0u) return 0u;');
    expect(rdna4).toContain('while (m > 2048u && (uint64_t)m * k > budget) m /= 2u;');
    expect(rdna4).toContain('return (uint64_t)m * k <= budget ? m : 0u;');
    // The same walk in JS: m 4,096 at k 8192 on the 9070 XT (SRBMiner's m there), 2,048 on the 9060 XT.
    const pick = (budget, k) => { let m = 65536; while (m > 2048 && m * k > budget) m /= 2; return m * k <= budget ? m : 0; };
    expect([4096, 8192, 16384].map((k) => pick(bytes(1201), k))).toEqual([8192, 4096, 2048]);
    expect([4096, 8192, 16384].map((k) => pick(bytes(1200), k))).toEqual([4096, 2048, 0]);
  });

  it('folds as many batches a launch as m is smaller, up to 16, and one on every other card', () => {
    expect(strips).toContain('if (!profile || !profile->m || profile->m >= mainM || mainM % profile->m != 0u) return 1u;');
    expect(strips).toContain('if (!arch_strips(arch)) return 1u;');
    expect(slice(HOST, 'static bool arch_strips(const char *arch) {', '\n}\n')).toContain('return arch_is(arch, "gfx1100") || '
      + 'arch_is(arch, "gfx1101") || arch_is(arch, "gfx1200") || arch_is(arch, "gfx1201");');
    expect(strips).toContain('return s > 16u ? 16u : s;');
    const create = slice(HOST, 'extern "C" void *pearl_host_create_checked(', '\n}\n');
    expect(create).toContain('ctx->stripAsked = pearl_env_u32("PEARL_AMD_STRIPS", &want);\n'
      + '    if (!ctx->stripAsked) want = amd_card_strips(profile, ctx->arch);\n'
      + '    if (want > 1u && want <= 16u && (want & (want - 1u)) == 0u && ctx->colsValid % (ctx->colBatch * want) == 0u) {\n'
      + '      ctx->strips = want;\n      ctx->colBatch *= want;\n');
    // The strips are part of the batch before anything is sized by it.
    expect(create.indexOf('ctx->colBatch *= want;')).toBeLessThan(create.indexOf('ctx->batch = ctx->colBatch * ctx->rowsValid;'));
  });

  it('refuses strips on a code object that folds none', () => {
    const resolve = slice(HOST, 'void resolve_fold(Ctx *ctx) {', '\n}\n');
    expect(resolve).toContain('if (ctx->strips > 1u && !(caps[0] & PEARL_AMD_CAP_STRIP)) {');
  });
});

describe('the core reports its m (core.m)', () => {
  it('asks the AMD host alone, when the miner lets it choose, before profile_ is set', () => {
    const ctor = slice(CORE, 'PearlCore::PearlCore(const Napi::CallbackInfo &info)', '\n}\n');
    const at = (s) => { const i = ctor.indexOf(s); expect([s, i >= 0]).toEqual([s, true]); return i; };
    const cardM = at('const uint32_t m = pearl_host_card_m(&profile, device_index_);');
    expect(ctor).toContain('#ifdef PEARL_HOST_K_PIN\n    // And a smaller m');
    expect(ctor).toContain('if (m) profile.m = m;');
    expect(at('if (card_k) {') < cardM && cardM < at('profile_ = profile;')).toBe(true);
    expect(slice(CORE, 'extern "C" {', '\n}\n')).not.toContain('pearl_host_card_m');
    expect(CUDA).not.toContain('pearl_host_card_m');
    expect(CORE).toContain('InstanceAccessor("m", &PearlCore::GetM, nullptr),');
    expect(slice(CORE, 'Napi::Value PearlCore::GetM(', '\n}\n')).toContain('profile_.m');
  });

  it('and the miner mines the profile at that m: the same config52, other proofs', () => {
    const p = profileForCore(PROFILE, { k: 2048, m: 16384 });
    expect(p).toEqual({ ...PROFILE, m: 16384 });
    expect(buildConfig52(p).equals(buildConfig52(PROFILE))).toBe(true);
    // A core that says mainnet's m, or none, mines the profile untouched.
    expect(profileForCore(PROFILE, { k: 2048, m: PROFILE.m })).toBe(PROFILE);
    expect(profileForCore(PROFILE, { k: 2048 })).toBe(PROFILE);
    expect(profileForCore(PROFILE, { m: 16384 })).toEqual({ ...PROFILE, m: 16384 });
    // With a larger k too, and with a k no share can carry.
    expect(profileForCore(PROFILE, { k: 4096, m: 16384 })).toEqual({ ...PROFILE, k: 4096, m: 16384 });
    expect(profileForCore(PROFILE, { k: 3000, m: 16384 })).toBeNull();
  });
});
