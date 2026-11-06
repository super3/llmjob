'use strict';

// PEARL_AMD_BUFLD (earn/native/amd/src/pearl_amd_fold.h): on RDNA 4 the fold's slab loads are buffer loads, from a
// buffer resource for the block's tile of A' and of B'. A buffer load past its resource's size returns zeros
// instead of the operands, so these check, by running the loads' arithmetic as JS, that every load the fold makes
// lies inside its tile and is the byte the plain load read. No HIP compiler runs here, so the C++ is read as text.

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

const buildScript = fs.readFileSync(path.join(__dirname, '..', 'native', 'amd', 'build-amd.sh'), 'utf8');

describe('slab loads as buffer loads (PEARL_AMD_BUFLD)', () => {
  it('is on by default in RDNA 4 code only, and only for the per-tile layout', () => {
    expect(FOLD).toContain('#ifndef PEARL_AMD_BUFLD\n#define PEARL_AMD_BUFLD PEARL_AMD_WMMA12\n#endif');
    expect(buf.split('\n')[0]).toBe(
      '#if PEARL_AMD_BUFLD && PEARL_AMD_TILED && (PEARL_AMD_WMMA11 || PEARL_AMD_WMMA12)');
  });

  // Word 3 of the resource (0x31004000) is gfx11's and gfx12's. -DPEARL_AMD_BUFLD=1 builds RDNA 3's buffer loads to
  // race (issue #287, item 2); it must not reach CDNA or RDNA 2 code, nor the HIP-CPU build.
  it('reaches only gfx11 and gfx12 device passes, whatever the setting', () => {
    const gate = /#if defined\(__GFX11__\)\n#define PEARL_AMD_WMMA11 1\n#endif\n#if defined\(__GFX12__\)\n#define PEARL_AMD_WMMA12 1\n#endif/;
    const devicePass = slice(FOLD, '#if !PEARL_AMD_EMU && defined(__HIP_DEVICE_COMPILE__)\n#if defined(__gfx942__)',
      '\n#endif\n#endif\n');
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

  it('stays off for RDNA 3 in a release build', () => {
    // The release builds with no PEARL_AMD_CFLAGS: nothing in build-amd.sh turns it on.
    expect(buildScript).not.toMatch(/PEARL_AMD_BUFLD/);
    expect(FOLD).not.toMatch(/#define PEARL_AMD_BUFLD \(?PEARL_AMD_WMMA11/);
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
