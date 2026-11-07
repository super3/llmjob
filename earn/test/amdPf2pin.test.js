'use strict';

// PEARL_AMD_PF2PIN (earn/native/amd/src/pearl_amd_fold.h, issue #287 item 8): RDNA 3's PF2 loop with every slab load
// two slabs ahead. With it, a gfx11 device pass gets PFX, TID and LPIN=2, and no other pass changes, so RDNA 4's code
// and the host stay as they are. LPIN=2 is an empty asm with a memory clobber after the slab loads, then LPIN's
// sched_barrier: the gfx1100 assembly (read in CI) kept all six loads at the top of their step only with the clobber.
// PEARL_AMD_RDNA3_PIN turns it on, with CRANK and SCHEDB, where RDNA 3's buffer loads are: gfx1100 and gfx1101 in
// the Linux core, the code it was raced on. No HIP compiler runs here, so the C++ is read as text.

const fs = require('fs');
const path = require('path');
const { preprocess } = require('./helpers/amdSwitches');

const FOLD = fs.readFileSync(path.join(__dirname, '..', 'native', 'amd', 'src', 'pearl_amd_fold.h'), 'utf8')
  .replace(/\r\n/g, '\n');

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

// PFX, TID and LPIN as a pass's defaults give them: wmma11 and wmma12 say which device pass (both 0: the host's, or
// another target's), pf2pin is -DPEARL_AMD_PF2PIN, bufld is PEARL_AMD_RDNA3_BUFLD (item 2's buffer loads, which bring TID).
function pass({ wmma11 = 0, wmma12 = 0, pf2pin = 0, bufld = 0 }) {
  const on = evalWith(defineOf('PEARL_AMD_PF2PIN_ON'), { PEARL_AMD_PF2PIN: pf2pin, PEARL_AMD_WMMA11: wmma11 });
  const v = { PEARL_AMD_WMMA11: wmma11, PEARL_AMD_WMMA12: wmma12, PEARL_AMD_PF2PIN_ON: on, PEARL_AMD_RDNA3_BUFLD: bufld };
  return {
    PFX: evalWith(defaultOf('PEARL_AMD_PFX'), v),
    TID: evalWith(defaultOf('PEARL_AMD_TID'), v),
    LPIN: evalWith(defaultOf('PEARL_AMD_LPIN'), v),
  };
}

const BUILD = fs.readFileSync(path.join(__dirname, '..', 'native', 'amd', 'build-amd.sh'), 'utf8')
  .replace(/\r\n/g, '\n');

// What a target's device pass gets, as the Linux core (no flags) or the Windows core (SET_OS) builds it.
function target(arch, windows) {
  const v = { PEARL_AMD_WMMA11: +/^gfx11/.test(arch), PEARL_AMD_WMMA12: +/^gfx12/.test(arch),
    PEARL_AMD_GFX1100_1101: +['gfx1100', 'gfx1101'].includes(arch) };
  v.PEARL_AMD_RDNA4_SET = windows ? 0 : evalWith(defaultOf('PEARL_AMD_RDNA4_SET'), v);
  v.PEARL_AMD_RDNA3_BUFLD = windows ? 0 : evalWith(defaultOf('PEARL_AMD_RDNA3_BUFLD'), v);
  v.PEARL_AMD_RDNA3_PIN = windows ? 0 : evalWith(defaultOf('PEARL_AMD_RDNA3_PIN'), v);
  v.PEARL_AMD_RDNA3_PIN_ON = evalWith(defineOf('PEARL_AMD_RDNA3_PIN_ON'), v);
  v.PEARL_AMD_PF2PIN = evalWith(defaultOf('PEARL_AMD_PF2PIN'), v);
  v.PEARL_AMD_PF2PIN_ON = evalWith(defineOf('PEARL_AMD_PF2PIN_ON'), v);
  const out = {};
  for (const n of ['PF2PIN_ON', 'PFX', 'TID', 'LPIN', 'CRANK', 'SCHEDB', 'BUFLD']) {
    out[n] = n === 'PF2PIN_ON' ? v.PEARL_AMD_PF2PIN_ON : evalWith(defaultOf('PEARL_AMD_' + n), v);
  }
  return out;
}

describe('PEARL_AMD_RDNA3_PIN', () => {
  it('follows RDNA 3\'s buffer loads, acts in gfx11 code only, and is 0 or 1', () => {
    expect(defaultOf('PEARL_AMD_RDNA3_PIN')).toBe('PEARL_AMD_RDNA3_BUFLD');
    expect(defineOf('PEARL_AMD_RDNA3_PIN_ON')).toBe('(PEARL_AMD_RDNA3_PIN && PEARL_AMD_WMMA11)');
    expect(FOLD).toContain('#if PEARL_AMD_RDNA3_PIN != 0 && PEARL_AMD_RDNA3_PIN != 1\n#error');
    expect(defaultOf('PEARL_AMD_PF2PIN')).toBe('PEARL_AMD_RDNA3_PIN');
    expect(defaultOf('PEARL_AMD_CRANK')).toBe('(PEARL_AMD_RDNA4_SET || PEARL_AMD_RDNA3_PIN_ON)');
    expect(defaultOf('PEARL_AMD_SCHEDB')).toBe('(PEARL_AMD_RDNA4_SET || PEARL_AMD_RDNA3_PIN_ON)');
  });

  it('is defined after the buffer-load switch and before every switch that reads it', () => {
    const at = FOLD.indexOf('#define PEARL_AMD_RDNA3_PIN ');
    expect(at).toBeGreaterThan(FOLD.indexOf('#define PEARL_AMD_RDNA3_BUFLD '));
    expect(FOLD.indexOf('#define PEARL_AMD_RDNA3_PIN_ON ')).toBeGreaterThan(at);
    for (const name of ['PEARL_AMD_PF2PIN', 'PEARL_AMD_PF2PIN_ON', 'PEARL_AMD_PFX', 'PEARL_AMD_TID', 'PEARL_AMD_CRANK',
      'PEARL_AMD_SCHEDB', 'PEARL_AMD_LPIN']) {
      expect([name, FOLD.indexOf(`#define ${name} `) > FOLD.indexOf('#define PEARL_AMD_RDNA3_PIN_ON ')]).toEqual([name, true]);
    }
  });

  // pinc on 7 November: PF2PIN + CRANK + SCHEDB on top of the buffer loads, raced on gfx1100 and gfx1101 only.
  it('is on in gfx1100 and gfx1101 code of the Linux core only; every other target keeps what it had', () => {
    const archs = (name) => new RegExp(`^${name}="([^"]+)"$`, 'm').exec(BUILD)[1].split(' ');
    const pinned = { PF2PIN_ON: 1, PFX: 1, TID: 1, LPIN: 2, CRANK: 1, SCHEDB: 1, BUFLD: 1 };
    const rdna4 = { PF2PIN_ON: 0, PFX: 1, TID: 1, LPIN: 0, CRANK: 1, SCHEDB: 1, BUFLD: 1 };
    const plain = { PF2PIN_ON: 0, PFX: 0, TID: 0, LPIN: 0, CRANK: 0, SCHEDB: 0, BUFLD: 0 };
    const linux = archs('DEFAULT_ARCHS');
    expect(linux).toEqual(expect.arrayContaining(['gfx1100', 'gfx1101', 'gfx1102', 'gfx1150', 'gfx1151', 'gfx1201']));
    for (const a of linux) {
      const want = ['gfx1100', 'gfx1101'].includes(a) ? pinned : /^gfx12/.test(a) ? rdna4 : plain;
      expect([a, target(a, false)]).toEqual([a, want]);
    }
    // The Windows core: RDNA 4 without its set, RDNA 3 as the release had it.
    for (const a of archs('WINDOWS_ARCHS')) {
      const want = /^gfx12/.test(a) ? { ...plain, PFX: 1, TID: 1, BUFLD: 1 } : plain;
      expect([a, target(a, true)]).toEqual([a, want]);
    }
    expect(BUILD).toContain('SET_OS="-DPEARL_AMD_RDNA4_SET=0 -DPEARL_AMD_RDNA3_BUFLD=0 -DPEARL_AMD_RDNA3_PIN=0"');
    expect(BUILD).not.toMatch(/-DPEARL_AMD_RDNA3_PIN=1|-DPEARL_AMD_PF2PIN/);
  });
});

describe('PEARL_AMD_PF2PIN', () => {
  it('acts in gfx11 code only', () => {
    expect(defineOf('PEARL_AMD_PF2PIN_ON')).toBe('(PEARL_AMD_PF2PIN && PEARL_AMD_WMMA11)');
  });

  it('turns on PFX, TID and LPIN=2 in a gfx11 device pass, and changes no other pass', () => {
    const none = { PFX: 0, TID: 0, LPIN: 0 };
    const rdna4 = { PFX: 1, TID: 1, LPIN: 0 };
    expect(pass({})).toEqual(none);
    expect(pass({ pf2pin: 1 })).toEqual(none);
    expect(pass({ wmma11: 1 })).toEqual(none);
    expect(pass({ wmma11: 1, pf2pin: 1 })).toEqual({ PFX: 1, TID: 1, LPIN: 2 });
    expect(pass({ wmma12: 1 })).toEqual(rdna4);
    expect(pass({ wmma12: 1, pf2pin: 1 })).toEqual(rdna4);
    // On top of RDNA 3's buffer loads (gfx1100 and gfx1101 by default), which already bring TID.
    expect(pass({ wmma11: 1, bufld: 1 })).toEqual({ PFX: 0, TID: 1, LPIN: 0 });
    expect(pass({ wmma11: 1, bufld: 1, pf2pin: 1 })).toEqual({ PFX: 1, TID: 1, LPIN: 2 });
  });

  it('is defined after the instruction gates and before the switches that read it', () => {
    const at = FOLD.indexOf('#define PEARL_AMD_PF2PIN_ON ');
    expect(at).toBeGreaterThan(FOLD.indexOf('#define PEARL_AMD_WMMA11 0'));
    for (const name of ['PEARL_AMD_PFX', 'PEARL_AMD_TID', 'PEARL_AMD_LPIN']) {
      expect([name, FOLD.indexOf(`#define ${name} `) > at]).toEqual([name, true]);
    }
    // The sched pin is defined for LPIN after LPIN's default.
    expect(FOLD.indexOf('#if (PEARL_AMD_SCHEDB || PEARL_AMD_LPIN)')).toBeGreaterThan(FOLD.indexOf('#define PEARL_AMD_LPIN '));
    expect(FOLD).toContain('#if PEARL_AMD_LPIN < 0 || PEARL_AMD_LPIN > 2\n#error');
  });

  it('pins the slab loads with a memory clobber, then the sched_barrier, at the top of each step', () => {
    const at = (v) => preprocess(FOLD, { PEARL_AMD_ABLATE: 0, PEARL_AMD_PFX: 1, PEARL_AMD_LPIN: v });
    const loads = '      PEARL_AMD_LOADR(n, pre2[B])\n    }\n';
    const clobber = '#if PEARL_AMD_LPIN == 2 && !PEARL_AMD_EMU && defined(__HIP_DEVICE_COMPILE__)\n'
      + '    asm volatile("" ::: "memory");   // PEARL_AMD_LPIN=2 (above)\n#endif\n';
    const pin = '    PEARL_AMD_SCHED_PIN();   // PEARL_AMD_LPIN (above)\n';
    const frags = '    const int8_t *aBase = sA[B] + wr * WM * S;\n';
    // The clobber's #if names more than the switch, so it stays as written; only a GPU's device pass has the asm.
    expect(at(2)).toContain(loads + clobber + pin + frags);
    expect(at(1)).toContain(loads + clobber + pin + frags);
    expect(at(0)).toContain(loads + clobber + frags);
    // One clobber, in the PF2 loop's step only: an empty asm with no operands, which adds no instruction.
    expect(FOLD.split('asm volatile("" ::: "memory");   // PEARL_AMD_LPIN=2 (above)')).toHaveLength(2);
  });
});
