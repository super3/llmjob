'use strict';

// The engine decision table. Pure, so every case here is exact: no filesystem,
// no PATH, no binary.

const {
  CHOICES, DEFAULT_MINER, BIN_NAME, DEV_FEE_PCT,
  isValidMiner, minerChoices, resolvePeakBin, pathDirsFrom, selectMiner,
} = require('../src/shared/minerSelect');

describe('constants', () => {
  test('auto is the default and the choices are the three engines', () => {
    expect(DEFAULT_MINER).toBe('auto');
    expect(CHOICES).toEqual(['auto', 'peak', 'native']);
    expect(BIN_NAME).toBe('peakminer');
  });

  // Quoted in the CLI help and the startup disclosure. If it ever moves, both
  // of those become wrong, so it is pinned here rather than left implicit.
  test('the disclosed dev fee is PeakMiner 2%', () => {
    expect(DEV_FEE_PCT).toBe(2.0);
  });

  test('isValidMiner accepts exactly the choices', () => {
    for (const c of CHOICES) expect(isValidMiner(c)).toBe(true);
    expect(isValidMiner('peakminer')).toBe(false);
    expect(isValidMiner('')).toBe(false);
    expect(isValidMiner(undefined)).toBe(false);
  });

  test('minerChoices renders them for an error message', () => {
    expect(minerChoices()).toBe('auto, peak, native');
  });
});

describe('pathDirsFrom', () => {
  test('splits a PATH and drops empties', () => {
    expect(pathDirsFrom('/usr/bin:/usr/local/bin')).toEqual(['/usr/bin', '/usr/local/bin']);
    expect(pathDirsFrom('/a::/b:')).toEqual(['/a', '/b']);
  });

  test('an absent PATH is no directories, not a crash', () => {
    expect(pathDirsFrom(undefined)).toEqual([]);
    expect(pathDirsFrom(null)).toEqual([]);
    expect(pathDirsFrom('')).toEqual([]);
  });

  test('honours a custom separator', () => {
    expect(pathDirsFrom('C:\\a;C:\\b', ';')).toEqual(['C:\\a', 'C:\\b']);
  });
});

describe('resolvePeakBin', () => {
  const exists = (set) => (p) => set.includes(p);

  test('an explicit --miner-bin wins and is used verbatim', () => {
    expect(resolvePeakBin({
      minerBin: '/opt/peak/pm', exists: exists(['/opt/peak/pm']),
    })).toBe('/opt/peak/pm');
  });

  // The point of naming a path is to get THAT binary. Falling through to a
  // different one found on PATH would run something the operator did not ask
  // for, under a name suggesting they had.
  test('an explicit path that is absent resolves to nothing, not a PATH hit', () => {
    expect(resolvePeakBin({
      minerBin: '/opt/peak/pm',
      pathDirs: ['/usr/bin'],
      exists: exists(['/usr/bin/peakminer']),
    })).toBe(null);
  });

  test('PEAK_MINER_BIN is consulted when no flag was given', () => {
    expect(resolvePeakBin({
      env: { PEAK_MINER_BIN: '/env/pm' }, exists: exists(['/env/pm']),
    })).toBe('/env/pm');
  });

  test('the flag beats the env var', () => {
    expect(resolvePeakBin({
      minerBin: '/flag/pm',
      env: { PEAK_MINER_BIN: '/env/pm' },
      exists: exists(['/flag/pm', '/env/pm']),
    })).toBe('/flag/pm');
  });

  test('falls back to the first PATH directory that has it', () => {
    expect(resolvePeakBin({
      pathDirs: ['/nope', '/usr/local/bin', '/usr/bin'],
      exists: exists(['/usr/local/bin/peakminer', '/usr/bin/peakminer']),
    })).toBe('/usr/local/bin/peakminer');
  });

  test('trailing slashes do not produce a doubled separator', () => {
    expect(resolvePeakBin({
      pathDirs: ['/usr/local/bin//'], exists: exists(['/usr/local/bin/peakminer']),
    })).toBe('/usr/local/bin/peakminer');
  });

  test('empty PATH entries are skipped', () => {
    expect(resolvePeakBin({
      pathDirs: ['', '/usr/bin'], exists: exists(['/usr/bin/peakminer']),
    })).toBe('/usr/bin/peakminer');
  });

  test('nothing anywhere is null', () => {
    expect(resolvePeakBin({ pathDirs: ['/usr/bin'], exists: () => false })).toBe(null);
    expect(resolvePeakBin()).toBe(null);
  });

  // With no `exists` injected nothing can be found, rather than everything
  // appearing to exist -- the safe direction for a probe that decides which
  // binary a rig runs.
  test('the default existence check finds nothing', () => {
    expect(resolvePeakBin({ pathDirs: ['/usr/bin'] })).toBe(null);
    expect(resolvePeakBin({ minerBin: '/opt/pm' })).toBe(null);
  });
});

describe('selectMiner', () => {
  const found = (p) => (q) => q === p;

  test('auto prefers PeakMiner when a binary is there', () => {
    const r = selectMiner({ pathDirs: ['/usr/bin'], exists: found('/usr/bin/peakminer') });
    expect(r.engine).toBe('peak');
    expect(r.bin).toBe('/usr/bin/peakminer');
    expect(r.reason).toContain('/usr/bin/peakminer');
  });

  // The whole reason 'auto' is a preference and not a requirement: PeakMiner is
  // proprietary and cannot be bundled, so a fresh rig has nothing installed and
  // must still mine.
  test('auto falls back to the built-in core when nothing is installed', () => {
    const r = selectMiner({ exists: () => false });
    expect(r.engine).toBe('native');
    expect(r.bin).toBe(null);
    expect(r.reason).toMatch(/zero-fee/);
  });

  test('auto is the default when no choice was given at all', () => {
    expect(selectMiner().engine).toBe('native');
    expect(selectMiner({}).engine).toBe('native');
  });

  test('native is forced even when PeakMiner is installed', () => {
    const r = selectMiner({
      miner: 'native', pathDirs: ['/usr/bin'], exists: found('/usr/bin/peakminer'),
    });
    expect(r.engine).toBe('native');
    expect(r.bin).toBe(null);
    expect(r.reason).toMatch(/forced/);
  });

  test('peak uses the binary when present', () => {
    const r = selectMiner({
      miner: 'peak', pathDirs: ['/usr/bin'], exists: found('/usr/bin/peakminer'),
    });
    expect(r.engine).toBe('peak');
  });

  // Asking for a named engine and silently getting a slower one would misreport
  // what the rig is doing, so this is an error rather than a fallback.
  test('peak with no binary is an error, not a silent downgrade', () => {
    const r = selectMiner({ miner: 'peak', exists: () => false });
    expect(r.engine).toBe(null);
    expect(r.bin).toBe(null);
    expect(r.reason).toMatch(/--miner-bin, PEAK_MINER_BIN, PATH/);
    expect(r.reason).toMatch(/cannot be bundled/);
  });
});
