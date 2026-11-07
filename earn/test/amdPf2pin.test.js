'use strict';

// PEARL_AMD_PF2PIN (earn/native/amd/src/pearl_amd_fold.h, issue #287 item 8): RDNA 3's PF2 loop with every slab load
// two slabs ahead. It is off by default. With it, a gfx11 device pass gets PFX, TID and LPIN=2, and no other pass
// changes, so RDNA 4's code and the host stay as they are. LPIN=2 is an empty asm with a memory clobber after the
// slab loads, then LPIN's sched_barrier: the gfx1100 assembly (read in CI) kept all six loads at the top of their
// step only with the clobber. No HIP compiler runs here, so the C++ is read as text.

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

describe('PEARL_AMD_PF2PIN', () => {
  it('is off by default', () => {
    expect(defaultOf('PEARL_AMD_PF2PIN')).toBe('0');
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
