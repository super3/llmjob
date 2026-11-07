'use strict';

// PEARL_AMD_BUFLD (earn/native/amd/src/pearl_amd_fold.h): on RDNA 4, and on RDNA 3's gfx1100 and gfx1101, the fold's
// slab loads are buffer loads, from a buffer resource for the block's tile of A' and of B'. A buffer load past its
// resource's size returns zeros instead of the operands, so these check, by running the loads' arithmetic as JS, that
// every load the fold makes lies inside its tile and is the byte the plain load read. No HIP compiler runs here, so
// the C++ is read as text.

const fs = require('fs');
const path = require('path');

const FOLD = fs.readFileSync(path.join(__dirname, '..', 'native', 'amd', 'src', 'pearl_amd_fold.h'), 'utf8')
  .replace(/\r\n/g, '\n');

function slice(text, from, to) {
  const at = text.indexOf(from);
  if (at < 0) throw new Error('not found: ' + from);
  return text.slice(at, text.indexOf(to, at + from.length) + to.length);
}

const buf = slice(FOLD, '#if PEARL_AMD_BUFLD && PEARL_AMD_TILED', '\n#else\n');
const plain = slice(FOLD, '#else\n#define PEARL_AMD_LOADR(slab, R)', '\n#endif\n');

const buildScript = fs.readFileSync(path.join(__dirname, '..', 'native', 'amd', 'build-amd.sh'), 'utf8')
  .replace(/\r\n/g, '\n');

// The default an #ifndef block gives a macro, as written, and its value for given macros.
function defaultOf(name) {
  const m = new RegExp(`#ifndef ${name}\\n#define ${name} (.*)\\n#endif`).exec(FOLD);
  if (!m) throw new Error('no default for ' + name);
  return m[1];
}
const evalWith = (expr, vars) => Number(new Function(...Object.keys(vars), `return (${expr});`)(...Object.values(vars)));

const devicePass = slice(FOLD, '#if !PEARL_AMD_EMU && defined(__HIP_DEVICE_COMPILE__)\n#if defined(__gfx942__)',
  '\n#endif\n#endif\n');

describe('slab loads as buffer loads (PEARL_AMD_BUFLD)', () => {
  it('is on by default in RDNA 4 code and, with PEARL_AMD_RDNA3_BUFLD, in RDNA 3 code, for the per-tile layout only', () => {
    expect(defaultOf('PEARL_AMD_BUFLD')).toBe('(PEARL_AMD_WMMA12 || (PEARL_AMD_WMMA11 && PEARL_AMD_RDNA3_BUFLD))');
    // TID goes with it: without TID the first staging unit's A'-or-B' test stays a run-time branch.
    expect(defaultOf('PEARL_AMD_TID')).toBe('(PEARL_AMD_WMMA12 || (PEARL_AMD_WMMA11 && PEARL_AMD_RDNA3_BUFLD))');
    expect(buf.split('\n')[0]).toBe(
      '#if PEARL_AMD_BUFLD && PEARL_AMD_TILED && (PEARL_AMD_WMMA11 || PEARL_AMD_WMMA12)');
  });

  // Word 3 of the resource (0x31004000) is gfx11's and gfx12's. -DPEARL_AMD_BUFLD=1 must not reach CDNA or RDNA 2
  // code, nor the HIP-CPU build.
  it('reaches only gfx11 and gfx12 device passes, whatever the setting', () => {
    const gate = /#if defined\(__GFX11__\)\n#define PEARL_AMD_WMMA11 1\n#endif\n#if defined\(__GFX12__\)\n#define PEARL_AMD_WMMA12 1\n#endif/;
    expect(devicePass).toMatch(gate);
    // Nothing else sets them to 1.
    expect(FOLD.match(/#define PEARL_AMD_WMMA11 1/g)).toHaveLength(1);
    expect(FOLD.match(/#define PEARL_AMD_WMMA12 1/g)).toHaveLength(1);
    // Every buffer resource in the fold is behind that gate.
    const rsrcAt = [...FOLD.matchAll(/__builtin_amdgcn_make_buffer_rsrc/g)].map((m) => m.index);
    const start = FOLD.indexOf(buf), end = start + buf.length;
    expect(rsrcAt.length).toBeGreaterThan(0);
    for (const at of rsrcAt) expect(at > start && at < end).toBe(true);
  });

  // RDNA 3's buffer loads (with TID) won on an RX 7900 XTX (gfx1100) and an RX 7800 XT (gfx1101): issue #287, item 2.
  // They are the default on those two chips, in the Linux core only. Every other target keeps the code it had.
  it('is on in gfx1100, gfx1101 and gfx12 code, and off on every other target and in the Windows core', () => {
    const chips = /#if ((?:defined\(__gfx\w+__\)(?: \|\| )?)+)\n#define PEARL_AMD_GFX1100_1101 1\n#endif/.exec(devicePass);
    expect(chips).not.toBeNull();
    expect([...chips[1].matchAll(/__(gfx\w+)__/g)].map((m) => m[1])).toEqual(['gfx1100', 'gfx1101']);
    expect(FOLD.match(/#define PEARL_AMD_GFX1100_1101 1/g)).toHaveLength(1);
    expect(defaultOf('PEARL_AMD_GFX1100_1101')).toBe('0');
    const sw = defaultOf('PEARL_AMD_RDNA3_BUFLD');
    expect(sw).toBe('PEARL_AMD_GFX1100_1101');
    expect(FOLD).toContain('#if PEARL_AMD_RDNA3_BUFLD != 0 && PEARL_AMD_RDNA3_BUFLD != 1\n#error');
    // The switch is set after the chip test's default and before the two defaults that read it.
    const at = FOLD.indexOf('#define PEARL_AMD_RDNA3_BUFLD ');
    expect(at).toBeGreaterThan(FOLD.indexOf('#define PEARL_AMD_GFX1100_1101 0'));
    expect(FOLD.indexOf('#define PEARL_AMD_TID ')).toBeGreaterThan(at);
    expect(FOLD.indexOf('#define PEARL_AMD_BUFLD ')).toBeGreaterThan(at);

    const archs = (name) => new RegExp(`^${name}="([^"]+)"$`, 'm').exec(buildScript)[1].split(' ');
    // [BUFLD, TID] in a target's device pass, as the Linux core (no flags) or the Windows core (SET_OS) builds it.
    const loads = (arch, windows) => {
      const v = { PEARL_AMD_WMMA11: +/^gfx11/.test(arch), PEARL_AMD_WMMA12: +/^gfx12/.test(arch),
        PEARL_AMD_GFX1100_1101: +['gfx1100', 'gfx1101'].includes(arch) };
      v.PEARL_AMD_RDNA3_BUFLD = windows ? 0 : evalWith(sw, v);
      return [evalWith(defaultOf('PEARL_AMD_BUFLD'), v), evalWith(defaultOf('PEARL_AMD_TID'), v)];
    };
    const linux = archs('DEFAULT_ARCHS');
    expect(linux).toEqual(expect.arrayContaining(['gfx1100', 'gfx1101', 'gfx1102', 'gfx1150', 'gfx1151', 'gfx1030']));
    for (const a of linux) {
      expect([a, loads(a, false)]).toEqual([a, ['gfx1100', 'gfx1101', 'gfx1200', 'gfx1201'].includes(a) ? [1, 1] : [0, 0]]);
    }
    for (const a of archs('WINDOWS_ARCHS')) {
      expect([a, loads(a, true)]).toEqual([a, /^gfx12/.test(a) ? [1, 1] : [0, 0]]);
    }
  });

  it('stays out of the Windows core, and build-amd.sh never turns the loads on itself', () => {
    expect(buildScript).toContain('SET_OS=\nif windows; then SET_OS="-DPEARL_AMD_RDNA4_SET=0 -DPEARL_AMD_RDNA3_BUFLD=0"; fi\n');
    expect(buildScript).not.toMatch(/-DPEARL_AMD_BUFLD/);
    expect(buildScript).not.toMatch(/-DPEARL_AMD_RDNA3_BUFLD=1/);
  });

  it('makes each tile\'s resource exactly its tile, from the plain loads\' own start', () => {
    expect(buf).toContain('__builtin_amdgcn_make_buffer_rsrc(const_cast<int8_t *>(gA), (short)0, (int)((uint32_t)BM * k), 0x31004000)');
    expect(buf).toContain('__builtin_amdgcn_make_buffer_rsrc(const_cast<int8_t *>(gB), (short)0, (int)((uint32_t)BN * k), 0x31004000)');
    // The plain loads read the same tiles from the same starts.
    expect(plain).toContain('gA + (size_t)(slab) * (BM * KS) + 16 * idx');
    expect(plain).toContain('gB + (size_t)(slab) * (BN * KS) + 16 * (idx - UA)');
  });

  it('reads, for every slab and staging unit, the byte the plain load read, inside the tile', () => {
    expect(buf).toContain('__builtin_amdgcn_raw_buffer_load_b128(rsA, 16 * idx, (int)((slab) * (uint32_t)(BM * KS)), 0)');
    expect(buf).toContain('__builtin_amdgcn_raw_buffer_load_b128(rsB, 16 * (idx - UA), (int)((slab) * (uint32_t)(BN * KS)), 0)');
    const KS = Number(/#define PEARL_AMD_SLAB (\d+)u/.exec(FOLD)[1]);
    const BM = Number(/#define PEARL_AMD_BM (\d+)u/.exec(FOLD)[1]);
    const BN = Number(/#define PEARL_AMD_BN (\d+)u/.exec(FOLD)[1]);
    const T = Number(/#define PEARL_AMD_THREADS (\d+)u/.exec(FOLD)[1]);
    const QR = KS / 16, UA = BM * QR, UB = BN * QR, U = (UA + UB) / T;
    for (const k of [2048, 4096]) {
      const slabs = k / KS;
      for (let slab = 0; slab < slabs; slab++) {
        for (let t = 0; t < T; t++) {
          for (let u = 0; u < U; u++) {
            const idx = t + u * T;
            const a = idx < UA;
            // The buffer load's offset from its tile's start (voffset + soffset), and the plain load's.
            const off = a ? 16 * idx + slab * BM * KS : 16 * (idx - UA) + slab * BN * KS;
            const want = a ? slab * (BM * KS) + 16 * idx : slab * (BN * KS) + 16 * (idx - UA);
            if (off !== want || off < 0 || off + 16 > (a ? BM : BN) * k) {
              throw new Error(`k ${k} slab ${slab} thread ${t} unit ${u}: offset ${off}, want ${want}`);
            }
          }
        }
      }
    }
  });
});
