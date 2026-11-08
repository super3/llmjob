'use strict';

// The AMD Windows driver builds reported to move an idle card's VRAM into
// system RAM (llama.cpp discussion #23443): 32.0.31007.5012 up to, not
// including, Adrenalin 26.9.2's 32.0.32015.2008.

const {
  BAD_FROM, FIXED_IN, parseDriverVersion, compareDriverVersions, movesIdleVram, describeAmdDrivers,
} = require('../src/shared/amdDriver');

describe('parseDriverVersion', () => {
  it('reads numbers between dots', () => {
    expect(parseDriverVersion('32.0.31007.5012')).toEqual([32, 0, 31007, 5012]);
    expect(parseDriverVersion(' 31.0.24033.1003 ')).toEqual([31, 0, 24033, 1003]);
    expect(parseDriverVersion('7')).toEqual([7]);
  });

  it('is null for anything else', () => {
    for (const v of ['', null, undefined, 'N/A', '32.0.x.1', '32..1', '.32', '32.0.']) {
      expect([v, parseDriverVersion(v)]).toEqual([v, null]);
    }
  });
});

describe('compareDriverVersions', () => {
  it('compares part by part, as numbers', () => {
    expect(compareDriverVersions('32.0.31007.5012', '32.0.31007.5012')).toBe(0);
    expect(compareDriverVersions('32.0.31007.5013', '32.0.31007.5012')).toBe(1);
    expect(compareDriverVersions('32.0.9999.9999', '32.0.31007.5012')).toBe(-1);
    expect(compareDriverVersions('31.0.99999.0', '32.0.0.0')).toBe(-1);
  });

  it('counts a missing part as 0', () => {
    expect(compareDriverVersions('32.0', '32.0.0.0')).toBe(0);
    expect(compareDriverVersions('32.0.0.1', '32.0')).toBe(1);
  });

  it('is null when either is not a version', () => {
    expect(compareDriverVersions('N/A', '32.0')).toBeNull();
    expect(compareDriverVersions('32.0', 'N/A')).toBeNull();
  });
});

describe('movesIdleVram', () => {
  it('is true from the first bad build up to the fix', () => {
    expect([BAD_FROM, FIXED_IN]).toEqual(['32.0.31007.5012', '32.0.32015.2008']);
    for (const v of ['32.0.31007.5012', '32.0.31007.5013', '32.0.32015.2007', '32.0.31500.0']) {
      expect([v, movesIdleVram(v)]).toEqual([v, true]);
    }
  });

  it('is false before it, from the fix on, and for what is not a version', () => {
    for (const v of ['32.0.31007.5011', '32.0.22042.14002', '31.0.24033.1003', '32.0.32015.2008',
      '32.0.32015.2009', '33.0.0.0', 'N/A', '', null]) {
      expect([v, movesIdleVram(v)]).toEqual([v, false]);
    }
  });
});

describe('describeAmdDrivers', () => {
  const card = (driverVersion) => ({ name: 'AMD Radeon AI PRO R9700', driverVersion });

  it('says the version once, however many cards share it', () => {
    expect(describeAmdDrivers({ amd: [card('32.0.32015.2008'), card('32.0.32015.2008')] }))
      .toEqual([{ level: 'info', line: 'AMD graphics driver 32.0.32015.2008' }]);
  });

  it('warns about a build reported to move idle VRAM, naming the fix', () => {
    expect(describeAmdDrivers({ amd: [card('32.0.31007.5012'), card('32.0.31007.5012')] })).toEqual([
      { level: 'info', line: 'AMD graphics driver 32.0.31007.5012' },
      {
        level: 'warn',
        line: 'AMD graphics driver 32.0.31007.5012 is reported to move an idle card\'s VRAM to system RAM, '
          + 'which can slow or crash the PC (llama.cpp discussion #23443). Adrenalin 26.9.2 or newer fixes it.',
      },
    ]);
  });

  it('names each version when the cards differ, and warns only about the bad one', () => {
    const lines = describeAmdDrivers({ amd: [card('32.0.32015.2008'), card('32.0.31500.1')] });
    expect(lines[0]).toEqual({ level: 'info', line: 'AMD graphics driver 32.0.32015.2008, 32.0.31500.1' });
    expect(lines[1].line).toMatch(/^AMD graphics driver 32\.0\.31500\.1 is reported/);
  });

  // Linux's cards (sysfs) and an NVIDIA rig's answer have no driver version.
  it('says nothing when no version is known', () => {
    expect(describeAmdDrivers({ amd: [{ name: 'AMD Radeon RX 9070 XT', index: 0 }] })).toEqual([]);
    expect(describeAmdDrivers({ amd: [card(null)] })).toEqual([]);
    expect(describeAmdDrivers({ amd: [] })).toEqual([]);
    expect(describeAmdDrivers({ name: 'NVIDIA GeForce RTX 4090', count: 1 })).toEqual([]);
    expect(describeAmdDrivers(null)).toEqual([]);
  });
});
