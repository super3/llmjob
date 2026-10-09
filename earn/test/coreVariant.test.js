'use strict';

const fs = require('fs');
const path = require('path');
const {
  CU12, CU13, AMD, FILES, LABELS, MIN_DRIVER_CU13, cu13HasCodeFor, cu13AutoSelectsFor,
  parseCudaCards, pickCoreVariant, isRuntimeError,
} = require('../src/shared/coreVariant');

// Cards as parseCudaCards returns them.
const card = (index, cap, driverMajor) => {
  const [major, minor] = cap.split('.').map(Number);
  return { index, major, minor, driverMajor };
};
const RTX5090 = (index, drv) => card(index, '12.0', drv == null ? 610 : drv);
const RTX4090 = (index, drv) => card(index, '8.9', drv == null ? 610 : drv);
const RTX2080TI = (index, drv) => card(index, '7.5', drv == null ? 610 : drv);
const RTX3090 = (index, drv) => card(index, '8.6', drv == null ? 610 : drv);
const A100 = (index, drv) => card(index, '8.0', drv == null ? 610 : drv);
const H100 = (index, drv) => card(index, '9.0', drv == null ? 610 : drv);
const B200 = (index, drv) => card(index, '10.0', drv == null ? 610 : drv);

describe('core file names', () => {
  // These are the release asset names too: the self-updater and the workflows
  // look them up by these strings.
  test('are the two names the release ships', () => {
    expect(FILES[CU12]).toBe('pearl_core.node');
    expect(FILES[CU13]).toBe('pearl_core_cu13.node');
    expect(MIN_DRIVER_CU13).toBe(580);
  });

  // build-amd.sh writes this name, and the AMD README tells operators to use it.
  test('the AMD core is pearl_core_hip.node', () => {
    expect(FILES[AMD]).toBe('pearl_core_hip.node');
    expect(LABELS[AMD]).toBe('AMD build (ROCm)');
  });
});

// What the CUDA 13 build is compiled for (the workflow's gencode list: sm_89
// and sm_120) against what the loader picks it for on its own (Blackwell only
// until the Ada fold under ptxas 13.3 has been benched on a 4090).
describe('cu13HasCodeFor / cu13AutoSelectsFor', () => {
  test('Blackwell has code and is selected', () => {
    expect(cu13HasCodeFor(RTX5090(0))).toBe(true);
    expect(cu13AutoSelectsFor(RTX5090(0))).toBe(true);
    expect(cu13HasCodeFor(card(0, '12.1', 610))).toBe(true);
  });

  test('Ada has code but is not selected yet', () => {
    expect(cu13HasCodeFor(RTX4090(0))).toBe(true);
    expect(cu13AutoSelectsFor(RTX4090(0))).toBe(false);
  });

  // 8.6 and 8.0 are Ampere: same major as Ada, no code in the build.
  test('Ampere and anything older has no code', () => {
    expect(cu13HasCodeFor(RTX3090(0))).toBe(false);
    expect(cu13AutoSelectsFor(RTX3090(0))).toBe(false);
    expect(cu13HasCodeFor(A100(0))).toBe(false);
    expect(cu13AutoSelectsFor(A100(0))).toBe(false);
    expect(cu13HasCodeFor(card(0, '7.5', 610))).toBe(false);
  });

  // 9.0 is Hopper: its sm_90a code is in the 12.8 build only.
  test('Hopper has no code', () => {
    expect(cu13HasCodeFor(H100(0))).toBe(false);
    expect(cu13AutoSelectsFor(H100(0))).toBe(false);
  });

  // 10.0 is data-center Blackwell (B200). It is Blackwell, but its sm_100
  // code is in the 12.8 build only, so it must not be taken for compute 12.x.
  test('B200 has no code', () => {
    expect(cu13HasCodeFor(B200(0))).toBe(false);
    expect(cu13AutoSelectsFor(B200(0))).toBe(false);
  });
});

// What each core is compiled for, read from the workflow that builds both
// (native-core.yml, its matrix's gencode lists), so the loader's choice cannot
// drift from them. The binary carries SASS only, no PTX: a card whose
// architecture is not on a core's list cannot run that core at all.
describe('the gencode lists the two cores are built from', () => {
  const wf = fs.readFileSync(
    path.join(__dirname, '..', '..', '.github', 'workflows', 'native-core.yml'), 'utf8');
  const matrix = wf.slice(wf.indexOf('include:'), wf.indexOf('\n    steps:'));
  const entries = matrix.split(/\n(?=\s+- os: )/).slice(1).map((block) => ({
    variant: (/variant: '([^']*)'/.exec(block) || [])[1],
    codes: [...block.matchAll(/^\s*-gencode arch=compute_\w+,code=(sm_\w+)\s*$/gm)]
      .map((m) => m[1]),
  }));
  const of = (variant) => entries.filter((e) => e.variant === variant);
  // One card of each architecture the app mines on, and the code it runs.
  const fleet = [
    [RTX2080TI(0), 'sm_75'], [A100(0), 'sm_80'], [RTX3090(0), 'sm_86'],
    [RTX4090(0), 'sm_89'], [H100(0), 'sm_90a'], [B200(0), 'sm_100'], [RTX5090(0), 'sm_120'],
  ];

  test('has the two 12.8 entries and the two CUDA 13 entries', () => {
    expect(of('')).toHaveLength(2);
    expect(of('-cu13')).toHaveLength(2);
  });

  // The 12.8 build is what every card falls back to, so it has code for all
  // of them; a B200 (compute 10.0) is one it always gets.
  test('the 12.8 build has code for every card, sm_100 included', () => {
    for (const e of of('')) {
      for (const [, sm] of fleet) expect(e.codes).toContain(sm);
    }
  });

  test('cu13HasCodeFor says what the CUDA 13 build was compiled for', () => {
    for (const e of of('-cu13')) {
      expect(e.codes).not.toContain('sm_100');
      for (const [c, sm] of fleet) expect([sm, cu13HasCodeFor(c)]).toEqual([sm, e.codes.includes(sm)]);
    }
  });
});

describe('parseCudaCards', () => {
  test('reads index, compute capability and driver major per card', () => {
    expect(parseCudaCards('0, 12.0, 610.57.04\r\n1, 8.9, 610.57.04\n')).toEqual([
      { index: 0, major: 12, minor: 0, driverMajor: 610 },
      { index: 1, major: 8, minor: 9, driverMajor: 610 },
    ]);
  });

  test('reads a Turing card (compute 7.5)', () => {
    expect(parseCudaCards('0, 7.5, 575.57.08')).toEqual([
      { index: 0, major: 7, minor: 5, driverMajor: 575 },
    ]);
  });

  // Two digits of major: a B200 is 10.0, not 1.0 or 0.0.
  test('reads a B200 (compute 10.0)', () => {
    expect(parseCudaCards('0, 10.0, 580.95.05')).toEqual([
      { index: 0, major: 10, minor: 0, driverMajor: 580 },
    ]);
  });

  test('keeps a card whose driver field does not parse, with no driver', () => {
    expect(parseCudaCards('0, 12.0, [N/A]')).toEqual([
      { index: 0, major: 12, minor: 0, driverMajor: null },
    ]);
  });

  // An error string, a blank line, a card reporting N/A for its capability, a
  // short row: none of it may be read as a card.
  test('drops rows that do not parse', () => {
    expect(parseCudaCards('\nField "compute_cap" is not a valid field\n0, [N/A], 610.1\nx, 12.0, 610.1\n0, 12.0')).toEqual([]);
  });

  test('is empty for nothing at all', () => {
    expect(parseCudaCards(undefined)).toEqual([]);
    expect(parseCudaCards(null)).toEqual([]);
  });
});

describe('pickCoreVariant', () => {
  test('an all-Blackwell rig on driver 580+ gets the CUDA 13 build', () => {
    expect(pickCoreVariant({ env: {}, cards: [RTX5090(0)], gpus: [{ index: 0 }] })).toEqual({
      variant: CU13, reason: 'driver 610, mining card is compute 12.0',
    });
  });

  test('says "every" for a multi-card rig and lists each capability once', () => {
    const cards = [RTX5090(0), card(1, '12.0', 610), card(2, '12.1', 610)];
    expect(pickCoreVariant({ env: {}, cards, gpus: [{ index: 0 }, { index: 1 }, { index: 2 }] }))
      .toEqual({ variant: CU13, reason: 'driver 610, every mining card is compute 12.0/12.1' });
  });

  test('exactly 580 is enough', () => {
    expect(pickCoreVariant({ env: {}, cards: [RTX5090(0, 580)], gpus: [] }).variant).toBe(CU13);
  });

  // The fleet this exists to protect: a CUDA 13 runtime cannot start below 580.
  test('an older driver keeps the 12.8 build, and says why', () => {
    expect(pickCoreVariant({ env: {}, cards: [RTX5090(0, 575)], gpus: [{ index: 0 }] })).toEqual({
      variant: CU12, reason: 'driver 575 (the CUDA 13 build needs 580+)',
    });
  });

  // The build carries sm_89 code, but the Ada fold under ptxas 13.3 has not
  // been benched, so Ada is not picked on its own; the reason says which.
  test('a 4090 rig keeps the 12.8 build: the CUDA 13 one is not selected for Ada yet', () => {
    expect(pickCoreVariant({ env: {}, cards: [RTX4090(0)], gpus: [{ index: 0 }] })).toEqual({
      variant: CU12,
      reason: 'GPU 0 is compute 8.9 (the CUDA 13 build has code for it but measured no faster on a 4090)',
    });
  });

  test('a 3090 rig keeps the 12.8 build: the CUDA 13 one has no code for it', () => {
    expect(pickCoreVariant({ env: {}, cards: [RTX3090(0)], gpus: [{ index: 0 }] })).toEqual({
      variant: CU12, reason: 'GPU 0 is compute 8.6 (the CUDA 13 build has no code for it)',
    });
  });

  // GA100 (A100, A800, A30, CMP 170HX) has sm_80 code in the 12.8 build only.
  test('an A100 rig keeps the 12.8 build: the CUDA 13 one has no code for it', () => {
    expect(pickCoreVariant({ env: {}, cards: [A100(0)], gpus: [{ index: 0 }] })).toEqual({
      variant: CU12, reason: 'GPU 0 is compute 8.0 (the CUDA 13 build has no code for it)',
    });
  });

  // Hopper (H100, H200) has sm_90a code in the 12.8 build only, on any driver.
  test('an H100 rig keeps the 12.8 build: the CUDA 13 one has no code for it', () => {
    expect(pickCoreVariant({ env: {}, cards: [H100(0)], gpus: [{ index: 0 }] })).toEqual({
      variant: CU12, reason: 'GPU 0 is compute 9.0 (the CUDA 13 build has no code for it)',
    });
    expect(pickCoreVariant({ env: {}, cards: [H100(0, 580)], gpus: [] })).toEqual({
      variant: CU12, reason: 'GPU 0 is compute 9.0 (the CUDA 13 build has no code for it)',
    });
  });

  // A B200 (compute 10.0) has sm_100 code in the 12.8 build only, so it loads
  // that build on every driver, including the 580+ ones that would put a 5090
  // on the CUDA 13 build.
  test('a B200 rig keeps the 12.8 build: the CUDA 13 one has no code for it', () => {
    expect(pickCoreVariant({ env: {}, cards: [B200(0)], gpus: [{ index: 0 }] })).toEqual({
      variant: CU12, reason: 'GPU 0 is compute 10.0 (the CUDA 13 build has no code for it)',
    });
    expect(pickCoreVariant({ env: {}, cards: [B200(0, 580), B200(1, 580)], gpus: [] })).toEqual({
      variant: CU12, reason: 'GPU 0 is compute 10.0 (the CUDA 13 build has no code for it)',
    });
    expect(pickCoreVariant({ env: {}, cards: [B200(0, 575)], gpus: [{ index: 0 }] })).toEqual({
      variant: CU12, reason: 'driver 575 (the CUDA 13 build needs 580+)',
    });
  });

  // One B200 beside a 5090 decides for the rig, as any card without code does.
  test('a 5090 + B200 rig keeps the 12.8 build', () => {
    const out = pickCoreVariant({
      env: {}, cards: [RTX5090(0), B200(1)], gpus: [{ index: 0 }, { index: 1 }],
    });
    expect(out.variant).toBe(CU12);
    expect(out.reason).toBe('GPU 1 is compute 10.0 (the CUDA 13 build has no code for it)');
  });

  // Turing's code is in the 12.8 build only (sm_75); the CUDA 13 one is sm_120.
  test('a 2080 Ti rig keeps the 12.8 build', () => {
    expect(pickCoreVariant({ env: {}, cards: [RTX2080TI(0)], gpus: [{ index: 0 }] })).toEqual({
      variant: CU12, reason: 'GPU 0 is compute 7.5 (the CUDA 13 build has no code for it)',
    });
  });

  // One addon serves every card in the process, so one card that cannot run
  // it decides for the rig.
  test('a mixed rig keeps the 12.8 build', () => {
    const out = pickCoreVariant({
      env: {}, cards: [RTX5090(0), RTX4090(1)], gpus: [{ index: 0 }, { index: 1 }],
    });
    expect(out.variant).toBe(CU12);
    expect(out.reason).toMatch(/^GPU 1 is compute 8\.9/);
  });

  test('a 5090 + 2080 Ti rig keeps the 12.8 build', () => {
    const out = pickCoreVariant({
      env: {}, cards: [RTX5090(0), RTX2080TI(1)], gpus: [{ index: 0 }, { index: 1 }],
    });
    expect(out.variant).toBe(CU12);
    expect(out.reason).toMatch(/^GPU 1 is compute 7\.5/);
  });

  // PEARL_GPU_INDEX narrows the mining list to one card; only that card counts.
  test('only the cards that will mine count', () => {
    const cards = [RTX4090(0), RTX5090(1)];
    expect(pickCoreVariant({ env: {}, cards, gpus: [{ index: 1 }] }).variant).toBe(CU13);
  });

  // No mining list: one core that picks its own card, which may be any of them.
  test('with no mining list every card has to qualify', () => {
    expect(pickCoreVariant({ env: {}, cards: [RTX5090(0), RTX4090(1)], gpus: [] }).variant).toBe(CU12);
    expect(pickCoreVariant({ env: {}, cards: [RTX5090(0), RTX5090(1)] }).variant).toBe(CU13);
  });

  test('a mining card nvidia-smi said nothing about keeps the 12.8 build', () => {
    expect(pickCoreVariant({ env: {}, cards: [RTX5090(0)], gpus: [{ index: 3 }] })).toEqual({
      variant: CU12, reason: 'GPU 3 compute capability unknown',
    });
  });

  test('nothing known about the cards keeps the 12.8 build', () => {
    expect(pickCoreVariant({ env: {}, cards: [] })).toEqual({
      variant: CU12, reason: 'GPU compute capability unknown',
    });
    // The shape a caller that predates the choice produces.
    expect(pickCoreVariant()).toEqual({ variant: CU12, reason: 'GPU compute capability unknown' });
    expect(pickCoreVariant({ cards: 'junk' }).variant).toBe(CU12);
  });

  test('an unreadable driver version keeps the 12.8 build', () => {
    expect(pickCoreVariant({ env: {}, cards: [card(0, '12.0', null)] })).toEqual({
      variant: CU12, reason: 'driver version unknown',
    });
  });

  describe('PEARL_CORE_VARIANT', () => {
    test('cu13 forces the CUDA 13 build past every check', () => {
      expect(pickCoreVariant({ env: { PEARL_CORE_VARIANT: 'cu13' }, cards: [RTX4090(0, 470)] }))
        .toEqual({ variant: CU13, reason: 'PEARL_CORE_VARIANT=cu13' });
    });

    test('cu12 forces the 12.8 build on a rig that qualifies, in any case', () => {
      expect(pickCoreVariant({ env: { PEARL_CORE_VARIANT: ' CU12 ' }, cards: [RTX5090(0)] }))
        .toEqual({ variant: CU12, reason: 'PEARL_CORE_VARIANT=cu12' });
    });

    // The AMD core is never chosen on its own, only asked for. With nvidia-smi
    // saying nothing (an AMD rig) it is what the operator gets; with NVIDIA cards
    // listed, the operator's word still stands.
    test('amd asks for the AMD build, whatever nvidia-smi said', () => {
      expect(pickCoreVariant({ env: { PEARL_CORE_VARIANT: 'AMD' }, cards: [] }))
        .toEqual({ variant: AMD, reason: 'PEARL_CORE_VARIANT=amd' });
      expect(pickCoreVariant({ env: { PEARL_CORE_VARIANT: 'amd' }, cards: [RTX5090(0)] }).variant).toBe(AMD);
    });

    test('nothing picks the AMD build without being asked', () => {
      for (const cards of [[], [RTX5090(0)], [RTX4090(0)], [RTX2080TI(0)], [card(0, '12.0', null)]]) {
        expect(pickCoreVariant({ env: {}, cards }).variant).not.toBe(AMD);
      }
    });

    test('blank is the same as unset', () => {
      expect(pickCoreVariant({ env: { PEARL_CORE_VARIANT: '  ' }, cards: [RTX5090(0)] }).variant).toBe(CU13);
    });

    // A typo must not quietly decide anything: it is ignored, and the log says so.
    test('anything else is ignored, and the reason says so', () => {
      expect(pickCoreVariant({ env: { PEARL_CORE_VARIANT: '13' }, cards: [RTX5090(0)] })).toEqual({
        variant: CU13,
        reason: 'PEARL_CORE_VARIANT=13 ignored (use cu12, cu13 or amd); driver 610, mining card is compute 12.0',
      });
      expect(pickCoreVariant({ env: { PEARL_CORE_VARIANT: 'x' }, cards: [] }).reason)
        .toBe('PEARL_CORE_VARIANT=x ignored (use cu12, cu13 or amd); GPU compute capability unknown');
    });
  });
});

describe('isRuntimeError', () => {
  // What pearl_host.cu says when cudaGetDeviceCount fails, which is where a
  // CUDA 13 runtime on a pre-580 driver stops.
  test('matches the core\'s own "no CUDA device" message', () => {
    expect(isRuntimeError(new Error('no CUDA device found — is an NVIDIA driver installed?'))).toBe(true);
  });

  test('matches the CUDA runtime\'s words for a runtime/driver mismatch', () => {
    for (const m of [
      'could not open GPU 0: CUDA driver version is insufficient for CUDA runtime version',
      'system has unsupported display driver / cuda driver combination',
      'forward compatibility was attempted on non supported HW',
      'no kernel image is available for execution on the device',
      'the provided PTX was compiled with an unsupported toolchain.',
      'device kernel image is invalid',
      'API call is not supported in the installed CUDA driver',
    ]) expect(isRuntimeError(new Error(m))).toBe(true);
  });

  // A card that is merely full is not a reason to change builds.
  test('does not match an ordinary card failure', () => {
    expect(isRuntimeError(new Error('GPU 0 has 1.2 GiB free, the profile needs 3.1 GiB'))).toBe(false);
    expect(isRuntimeError(new Error('GPU 4 was asked for (PEARL_GPU_INDEX) but this machine has 1'))).toBe(false);
  });

  test('accepts a bare string or nothing', () => {
    expect(isRuntimeError('no CUDA device found')).toBe(true);
    expect(isRuntimeError(null)).toBe(false);
    expect(isRuntimeError({ message: null })).toBe(false);
  });
});
