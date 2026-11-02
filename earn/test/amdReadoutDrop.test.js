'use strict';

// PEARL_AMD_READOUT_DROP (earn/native/amd/src/pearl_amd_fold.h) builds an AMD core whose
// transcripts are wrong, to time the RDNA 4 readout. These pin the guards that keep it from
// mining or shipping, and sweep.sh's rule for timing it. The shell parts run under bash where
// there is one; the C++ is checked as text, since no HIP compiler runs here.

const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

const AMD = path.join(__dirname, '..', 'native', 'amd');
const read = (f) => fs.readFileSync(path.join(AMD, f), 'utf8').replace(/\r\n/g, '\n');
const FOLD = read('src/pearl_amd_fold.h');
const HOST = read('src/pearl_host_hip.cpp');
const BUILD = read('build-amd.sh');
const SWEEP = read('test/sweep.sh');

const hasBash = spawnSync('bash', ['--version'], { stdio: 'ignore' }).status === 0;
const bashIt = hasBash ? it : it.skip;
const bash = (script, env) => spawnSync('bash', ['-c', script],
  { env: Object.assign({ PATH: process.env.PATH }, env), encoding: 'utf8' });

// The text from the line starting with `first` to the next line that is exactly `last`.
function block(text, first, last) {
  const lines = text.split('\n');
  const a = lines.findIndex((l) => l.startsWith(first));
  const b = lines.findIndex((l, i) => i > a && l === last);
  if (a < 0 || b < 0) throw new Error('block not found: ' + first);
  return lines.slice(a, b + 1).join('\n') + '\n';
}

describe('the readout-drop build', () => {
  it('is off by default, drops only the RDNA 4 readout, and marks every code object it is in', () => {
    expect(FOLD).toContain('#ifndef PEARL_AMD_READOUT_DROP\n#define PEARL_AMD_READOUT_DROP 0\n#endif');
    expect(FOLD).toContain('constexpr bool kDrop = PEARL_AMD_READOUT_DROP && std::is_same<P, PearlAmdWmma12>::value;');
    expect(FOLD).toContain('if (PEARL_AMD_READOUT_DROP) out[0] |= PEARL_AMD_CAP_DROP;');
    // Its own bit, clear of the fold bits and the emulation bit.
    expect(FOLD).toContain('#define PEARL_AMD_CAP_DROP 0x200u');
  });

  it('makes the host refuse every target but zero, before any fold is launched', () => {
    expect(HOST).toContain('ctx->readoutDrop = (caps[0] & PEARL_AMD_CAP_DROP) != 0;');
    const submit = HOST.slice(HOST.indexOf('extern "C" bool pearl_host_submit('));
    const guard = submit.indexOf('readout-drop build is timing-only');
    expect(guard).toBeGreaterThan(0);
    expect(guard).toBeLessThan(submit.indexOf('hipLaunchKernelGGL(pearl_amd_fold_'));
    expect(submit.slice(0, guard)).toContain('zero = zero && ctx->target[i] == 0;');
  });

  const guard = block(BUILD, 'case "${PEARL_AMD_CFLAGS:-}" in', 'esac');
  const tryGuard = (mode, flags, ci) => bash('mode=' + mode + '\n' + guard + 'echo allowed',
    Object.assign({ PEARL_AMD_CFLAGS: flags }, ci || {}));

  bashIt('is refused by build-amd.sh in the builds that ship, and allowed on a test box', () => {
    const drop = '-O3 -DPEARL_AMD_READOUT_DROP=1';
    for (const [mode, flags, ci, ok] of [
      ['winlib', drop, null, false],
      ['winlib', '-DPEARL_AMD_READOUT_DROP=0', null, false],   // a release has no reason to name it
      ['gpu', drop, { CI: 'true' }, false],
      ['gpu', drop, { GITHUB_ACTIONS: 'true' }, false],
      ['gpu', drop, null, true],                              // sweep.sh on a Salad box
      ['check', drop, { CI: 'true' }, true],                  // compiles only; nothing ships from it
      ['cpu', drop, null, true],
      ['winlib', '-DPEARL_AMD_TILED=0', null, true],
      ['gpu', '', { CI: 'true' }, true],
    ]) {
      const r = tryGuard(mode, flags, ci);
      expect([mode, flags, ci, r.stdout.trim(), r.status]).toEqual([mode, flags, ci, ok ? 'allowed' : '', ok ? 0 : 1]);
      if (!ok) expect(r.stderr).toContain('readout-drop build is timing-only');
    }
  });

  bashIt('stops build-amd.sh before it builds anything', () => {
    const r = spawnSync('bash', [path.join(AMD, 'build-amd.sh'), 'winlib'], {
      env: Object.assign({}, process.env, { PEARL_AMD_CFLAGS: '-DPEARL_AMD_READOUT_DROP=1' }), encoding: 'utf8',
    });
    expect(r.status).toBe(1);
    expect(r.stderr).toContain('readout-drop build is timing-only');
    expect(r.stdout).toBe('');
  });
});

describe('sweep.sh SWEEP_TIMEONLY', () => {
  const rule = block(SWEEP, '  case " ${SWEEP_TIMEONLY:-} " in', '  esac');
  const OK = 'wmma12  ok: 4096 regions, all 16 words match; 1 launches, 0.0100 s, 1.000 TMAC/s PASS ';
  const BAD = 'wmma12  FAIL: 61440 of 65536 words differ; first: region 0 word 0, got 1 want 2 FAIL ';
  const FAULT = 'wmma12: hipErrorLaunchFailure FAIL ';

  bashIt('times a timing-only variant whose check finds wrong words, and nothing else that fails', () => {
    const run = (name, timeonly, chk) => {
      const script = 'out() { echo "$*"; }\nnames=()\nfor name in "$NAME"; do\n' + rule
        + 'echo timed\ndone\necho "names=${names[*]}"';
      const r = bash(script, { NAME: name, SWEEP_TIMEONLY: timeonly, chk });
      return r.stdout.trim().split('\n');
    };
    expect(run('base', 'drop', OK)).toEqual(['timed', 'names=base']);
    expect(run('base', 'drop', BAD)).toEqual(['base not timed: the check failed', 'names=']);
    expect(run('drop', 'drop', BAD)).toEqual(['drop check FAIL (expected: timing only)', 'timed', 'names=drop']);
    expect(run('drop', 'x drop y', BAD)).toEqual(['drop check FAIL (expected: timing only)', 'timed', 'names=drop']);
    // A drop build that matches the CPU did not get its flag; a fold that faulted can't be timed.
    expect(run('drop', 'drop', OK)[0]).toMatch(/^drop not timed: it is timing-only, yet the check passed/);
    expect(run('drop', 'drop', OK)[1]).toBe('names=');
    expect(run('drop', 'drop', FAULT)).toEqual(['drop not timed: the check failed', 'names=']);
    expect(run('drop', 'drop', 'no fold ran ')).toEqual(['drop not timed: the check failed', 'names=']);
    // Only whole names count.
    expect(run('drop', 'drop2', BAD)).toEqual(['drop not timed: the check failed', 'names=']);
    expect(run('drop', '', BAD)).toEqual(['drop not timed: the check failed', 'names=']);
  });
});
