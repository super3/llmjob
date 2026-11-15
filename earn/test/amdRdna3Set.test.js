'use strict';

// PEARL_AMD_RDNA3_SET (earn/native/amd/src/pearl_amd_fold.h, issue #287 item 8): RDNA 3's staging cuts, PFX, CRANK
// and SCHEDB, on top of item 2's buffer loads. Raced on 7 November on gfx1100 (RX 7900 XTX) and gfx1101 (RX 7800 XT)
// in the Linux core only, so it follows PEARL_AMD_RDNA3_BUFLD: on in those two device passes, off on every other
// target and in the Windows core. CRANK also makes the matrix fold assume rank 128, which the host enforces from the
// card's caps (amdRdna4Set.test.js). No HIP compiler runs here, so the C++ is read as text.

const fs = require('fs');
const path = require('path');

const AMD = path.join(__dirname, '..', 'native', 'amd');
const read = (...p) => fs.readFileSync(path.join(AMD, ...p), 'utf8').replace(/\r\n/g, '\n');
const FOLD = read('src', 'pearl_amd_fold.h');
const BUILD = read('build-amd.sh');

// The default an #ifndef block gives a macro, as written.
function defaultOf(name) {
  const m = new RegExp(`#ifndef ${name}\\n#define ${name} (.*)\\n#endif`).exec(FOLD);
  if (!m) throw new Error('no default for ' + name);
  return m[1];
}
function defineOf(name) {
  const m = new RegExp(`\\n#define ${name} (.*)\\n`).exec(FOLD);
  if (!m) throw new Error('no #define of ' + name);
  return m[1];
}
const evalWith = (expr, vars) => Number(new Function(...Object.keys(vars), `return (${expr});`)(...Object.values(vars)));

// What a target's device pass gets, as the Linux core (no flags) or the Windows core (SET_OS) builds it.
function target(arch, windows) {
  const v = { PEARL_AMD_WMMA11: +/^gfx11/.test(arch), PEARL_AMD_WMMA12: +/^gfx12/.test(arch),
    PEARL_AMD_GFX1100_1101: +['gfx1100', 'gfx1101'].includes(arch) };
  v.PEARL_AMD_RDNA4_SET = windows ? 0 : evalWith(defaultOf('PEARL_AMD_RDNA4_SET'), v);
  v.PEARL_AMD_RDNA3_BUFLD = windows ? 0 : evalWith(defaultOf('PEARL_AMD_RDNA3_BUFLD'), v);
  v.PEARL_AMD_RDNA3_SET = windows ? 0 : evalWith(defaultOf('PEARL_AMD_RDNA3_SET'), v);
  v.PEARL_AMD_RDNA3_SET_ON = evalWith(defineOf('PEARL_AMD_RDNA3_SET_ON'), v);
  v.PEARL_AMD_PF2PIN = evalWith(defaultOf('PEARL_AMD_PF2PIN'), v);
  v.PEARL_AMD_PF2PIN_ON = evalWith(defineOf('PEARL_AMD_PF2PIN_ON'), v);
  // RDNA 2's slab loop (PEARL_AMD_DOT4_ADDR, amdDot4Addr.test.js) brings buffer loads and TID to gfx103x, Linux only.
  v.PEARL_AMD_DOT4_ADDR_ON = /^gfx103/.test(arch) && !windows ? 1 : 0;
  const out = {};
  for (const n of ['PFX', 'TID', 'LPIN', 'CRANK', 'SCHEDB', 'BUFLD']) out[n] = evalWith(defaultOf('PEARL_AMD_' + n), v);
  return out;
}

describe('RDNA 3\'s staging cuts (PEARL_AMD_RDNA3_SET)', () => {
  it('follows RDNA 3\'s buffer loads, acts in gfx11 code only, and is 0 or 1', () => {
    expect(defaultOf('PEARL_AMD_RDNA3_SET')).toBe('PEARL_AMD_RDNA3_BUFLD');
    expect(defineOf('PEARL_AMD_RDNA3_SET_ON')).toBe('(PEARL_AMD_RDNA3_SET && PEARL_AMD_WMMA11)');
    expect(FOLD).toContain('#if PEARL_AMD_RDNA3_SET != 0 && PEARL_AMD_RDNA3_SET != 1\n#error');
    expect(defaultOf('PEARL_AMD_PFX')).toBe('(PEARL_AMD_WMMA12 || PEARL_AMD_PF2PIN_ON || PEARL_AMD_RDNA3_SET_ON)');
    expect(defaultOf('PEARL_AMD_TID')).toBe(
      '(PEARL_AMD_WMMA12 || (PEARL_AMD_WMMA11 && PEARL_AMD_RDNA3_BUFLD) || PEARL_AMD_PF2PIN_ON || PEARL_AMD_RDNA3_SET_ON || PEARL_AMD_DOT4_ADDR_ON)');
    expect(defaultOf('PEARL_AMD_CRANK')).toBe('(PEARL_AMD_RDNA4_SET || PEARL_AMD_RDNA3_SET_ON)');
    expect(defaultOf('PEARL_AMD_SCHEDB')).toBe('(PEARL_AMD_RDNA4_SET || PEARL_AMD_RDNA3_SET_ON)');
    // The load pin stays off: the set without it was faster on both chips.
    expect(defaultOf('PEARL_AMD_PF2PIN')).toBe('0');
  });

  it('is defined after the buffer-load switch and before every switch that reads it', () => {
    const at = FOLD.indexOf('#define PEARL_AMD_RDNA3_SET ');
    expect(at).toBeGreaterThan(FOLD.indexOf('#define PEARL_AMD_RDNA3_BUFLD '));
    const on = FOLD.indexOf('#define PEARL_AMD_RDNA3_SET_ON ');
    expect(on).toBeGreaterThan(at);
    for (const name of ['PEARL_AMD_PFX', 'PEARL_AMD_TID', 'PEARL_AMD_CRANK', 'PEARL_AMD_SCHEDB']) {
      expect([name, FOLD.indexOf(`#define ${name} `) > on]).toEqual([name, true]);
    }
  });

  it('is on in gfx1100 and gfx1101 code of the Linux core only; every other target keeps what it had', () => {
    const archs = (name) => new RegExp(`^${name}="([^"]+)"$`, 'm').exec(BUILD)[1].split(' ');
    const set = { PFX: 1, TID: 1, LPIN: 0, CRANK: 1, SCHEDB: 1, BUFLD: 1 };
    const rdna4 = { PFX: 1, TID: 1, LPIN: 0, CRANK: 1, SCHEDB: 1, BUFLD: 1 };
    const plain = { PFX: 0, TID: 0, LPIN: 0, CRANK: 0, SCHEDB: 0, BUFLD: 0 };
    const linux = archs('DEFAULT_ARCHS');
    expect(linux).toEqual(expect.arrayContaining(['gfx1100', 'gfx1101', 'gfx1102', 'gfx1150', 'gfx1151', 'gfx1201']));
    for (const a of linux) {
      const want = ['gfx1100', 'gfx1101'].includes(a) ? set : /^gfx12/.test(a) ? rdna4
        : /^gfx103/.test(a) ? { ...plain, TID: 1, BUFLD: 1 } : plain;
      expect([a, target(a, false)]).toEqual([a, want]);
    }
    // The Windows core: RDNA 4 without its set, RDNA 3 as the release had it.
    for (const a of archs('WINDOWS_ARCHS')) {
      const want = /^gfx12/.test(a) ? { ...plain, PFX: 1, TID: 1, BUFLD: 1 } : plain;
      expect([a, target(a, true)]).toEqual([a, want]);
    }
  });

  it('stays out of the Windows core by name, and build-amd.sh never turns it on', () => {
    expect(BUILD).toContain('SET_OS="-DPEARL_AMD_RDNA4_SET=0 -DPEARL_AMD_RDNA3_BUFLD=0 -DPEARL_AMD_RDNA3_SET=0"');
    expect(BUILD).not.toMatch(/-DPEARL_AMD_RDNA3_SET=1|-DPEARL_AMD_PF2PIN/);
  });
});
