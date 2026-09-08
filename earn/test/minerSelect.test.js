'use strict';

// The engine decision table. Pure, so every case here is exact: no filesystem,
// no PATH, no binary.

const {
  CHOICES, DEFAULT_MINER, BIN_NAMES, DEV_FEE_PCT,
  isValidMiner, minerChoices, resolveSrbBin, pathDirsFrom, selectMiner,
} = require('../src/shared/minerSelect');

describe('constants', () => {
  test('auto is the default and the choices are the three engines', () => {
    expect(DEFAULT_MINER).toBe('auto');
    expect(CHOICES).toEqual(['auto', 'srb', 'native']);
    // The release tarball ships it uppercase; the lowercase form is what a rig
    // that symlinks it onto PATH usually ends up with.
    expect(BIN_NAMES).toContain('SRBMiner-MULTI');
    expect(BIN_NAMES).toContain('srbminer-multi');
  });

  // Quoted in the CLI help and the startup disclosure. If it ever moves, both
  // of those become wrong, so it is pinned here rather than left implicit.
  test('the disclosed dev fee is SRBMiner 2%', () => {
    expect(DEV_FEE_PCT).toBe(2.0);
  });

  test('isValidMiner accepts exactly the choices', () => {
    for (const c of CHOICES) expect(isValidMiner(c)).toBe(true);
    expect(isValidMiner('srbminer')).toBe(false);
    expect(isValidMiner('')).toBe(false);
    expect(isValidMiner(undefined)).toBe(false);
  });

  test('minerChoices renders them for an error message', () => {
    expect(minerChoices()).toBe('auto, srb, native');
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

describe('resolveSrbBin', () => {
  const exists = (set) => (p) => set.includes(p);

  test('an explicit --miner-bin wins and is used verbatim', () => {
    expect(resolveSrbBin({
      minerBin: '/opt/srb/sm', exists: exists(['/opt/srb/sm']),
    })).toBe('/opt/srb/sm');
  });

  // The point of naming a path is to get THAT binary. Falling through to a
  // different one found on PATH would run something the operator did not ask
  // for, under a name suggesting they had.
  test('an explicit path that is absent resolves to nothing, not a PATH hit', () => {
    expect(resolveSrbBin({
      minerBin: '/opt/srb/sm',
      pathDirs: ['/usr/bin'],
      exists: exists(['/usr/bin/SRBMiner-MULTI']),
    })).toBe(null);
  });

  test('SRBMINER_BIN is consulted when no flag was given', () => {
    expect(resolveSrbBin({
      env: { SRBMINER_BIN: '/env/pm' }, exists: exists(['/env/pm']),
    })).toBe('/env/pm');
  });

  test('the flag beats the env var', () => {
    expect(resolveSrbBin({
      minerBin: '/flag/pm',
      env: { SRBMINER_BIN: '/env/pm' },
      exists: exists(['/flag/pm', '/env/pm']),
    })).toBe('/flag/pm');
  });

  test('falls back to the first PATH directory that has it', () => {
    expect(resolveSrbBin({
      pathDirs: ['/nope', '/usr/local/bin', '/usr/bin'],
      exists: exists(['/usr/local/bin/SRBMiner-MULTI', '/usr/bin/SRBMiner-MULTI']),
    })).toBe('/usr/local/bin/SRBMiner-MULTI');
  });

  test('trailing slashes do not produce a doubled separator', () => {
    expect(resolveSrbBin({
      pathDirs: ['/usr/local/bin//'], exists: exists(['/usr/local/bin/SRBMiner-MULTI']),
    })).toBe('/usr/local/bin/SRBMiner-MULTI');
  });

  test('empty PATH entries are skipped', () => {
    expect(resolveSrbBin({
      pathDirs: ['', '/usr/bin'], exists: exists(['/usr/bin/SRBMiner-MULTI']),
    })).toBe('/usr/bin/SRBMiner-MULTI');
  });

  // The release tarball ships SRBMiner-MULTI; a rig that symlinks it onto PATH
  // usually lowercases the name, so both spellings are searched.
  test('finds the lowercase spelling on PATH too', () => {
    expect(resolveSrbBin({
      pathDirs: ['/usr/local/bin'], exists: exists(['/usr/local/bin/srbminer-multi']),
    })).toBe('/usr/local/bin/srbminer-multi');
  });

  test('prefers the packaged spelling when both are present', () => {
    expect(resolveSrbBin({
      pathDirs: ['/usr/bin'],
      exists: exists(['/usr/bin/SRBMiner-MULTI', '/usr/bin/srbminer-multi']),
    })).toBe('/usr/bin/SRBMiner-MULTI');
  });

  test('nothing anywhere is null', () => {
    expect(resolveSrbBin({ pathDirs: ['/usr/bin'], exists: () => false })).toBe(null);
    expect(resolveSrbBin()).toBe(null);
  });

  // With no `exists` injected nothing can be found, rather than everything
  // appearing to exist -- the safe direction for a probe that decides which
  // binary a rig runs.
  test('the default existence check finds nothing', () => {
    expect(resolveSrbBin({ pathDirs: ['/usr/bin'] })).toBe(null);
    expect(resolveSrbBin({ minerBin: '/opt/pm' })).toBe(null);
  });
});

describe('selectMiner', () => {
  const found = (p) => (q) => q === p;

  test('auto prefers SRBMiner-Multi when a binary is there', () => {
    const r = selectMiner({ pathDirs: ['/usr/bin'], exists: found('/usr/bin/SRBMiner-MULTI') });
    expect(r.engine).toBe('srb');
    expect(r.bin).toBe('/usr/bin/SRBMiner-MULTI');
    expect(r.reason).toContain('/usr/bin/SRBMiner-MULTI');
  });

  // The whole reason 'auto' is a preference and not a requirement: SRBMiner-Multi is
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

  test('native is forced even when SRBMiner-Multi is installed', () => {
    const r = selectMiner({
      miner: 'native', pathDirs: ['/usr/bin'], exists: found('/usr/bin/SRBMiner-MULTI'),
    });
    expect(r.engine).toBe('native');
    expect(r.bin).toBe(null);
    expect(r.reason).toMatch(/forced/);
  });

  test('srb uses the binary when present', () => {
    const r = selectMiner({
      miner: 'srb', pathDirs: ['/usr/bin'], exists: found('/usr/bin/SRBMiner-MULTI'),
    });
    expect(r.engine).toBe('srb');
  });

  // Asking for a named engine and silently getting a slower one would misreport
  // what the rig is doing, so this is an error rather than a fallback.
  test('srb with no binary is an error, not a silent downgrade', () => {
    const r = selectMiner({ miner: 'srb', exists: () => false });
    expect(r.engine).toBe(null);
    expect(r.bin).toBe(null);
    expect(r.reason).toMatch(/--miner-bin, SRBMINER_BIN, PATH/);
    expect(r.reason).toMatch(/cannot be bundled/);
  });
});
