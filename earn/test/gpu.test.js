'use strict';

const { pickGpu, countGpus, parseGpuStats, parseMacGpu } = require('../src/shared/gpu');

describe('pickGpu', () => {
  test('picks the real GPU and skips the basic display adapter', () => {
    expect(pickGpu(['Microsoft Basic Display Adapter', 'NVIDIA GeForce RTX 4090'])).toBe('NVIDIA GeForce RTX 4090');
  });

  test('returns the first real adapter when several discrete cards are present', () => {
    expect(pickGpu(['NVIDIA GeForce RTX 4090', 'AMD Radeon RX 7900'])).toBe('NVIDIA GeForce RTX 4090');
  });

  test('prefers a discrete GPU over an integrated one listed first', () => {
    // The reported field case: an AMD APU enumerates before the RTX 4090.
    expect(pickGpu(['AMD Radeon(TM) Graphics', 'NVIDIA GeForce RTX 4090'])).toBe('NVIDIA GeForce RTX 4090');
    expect(pickGpu(['Intel(R) UHD Graphics 630', 'NVIDIA GeForce RTX 3080'])).toBe('NVIDIA GeForce RTX 3080');
    expect(pickGpu(['AMD Radeon(TM) Vega 8 Graphics', 'AMD Radeon RX 6800 XT'])).toBe('AMD Radeon RX 6800 XT');
  });

  test('falls back to an integrated GPU when it is the only real adapter', () => {
    expect(pickGpu(['Microsoft Basic Display Adapter', 'AMD Radeon(TM) Graphics'])).toBe('AMD Radeon(TM) Graphics');
    expect(pickGpu(['Intel(R) Iris(R) Xe Graphics'])).toBe('Intel(R) Iris(R) Xe Graphics');
  });

  test('trims whitespace and ignores blank/nullish entries', () => {
    expect(pickGpu([null, '', undefined, '   ', '  NVIDIA GeForce RTX 4090  '])).toBe('NVIDIA GeForce RTX 4090');
  });

  test('returns null when only virtual adapters are present', () => {
    expect(pickGpu(['Microsoft Basic Display Adapter', 'VMware SVGA 3D'])).toBeNull();
  });

  test('returns null for non-arrays', () => {
    expect(pickGpu(null)).toBeNull();
    expect(pickGpu(undefined)).toBeNull();
    expect(pickGpu('NVIDIA')).toBeNull();
  });
});

describe('countGpus', () => {
  test('counts discrete GPUs, ignoring an iGPU riding alongside', () => {
    expect(countGpus(['Intel UHD Graphics 770', 'NVIDIA GeForce RTX 3070', 'NVIDIA GeForce RTX 3070'])).toBe(2);
    expect(countGpus(Array(8).fill('NVIDIA GeForce RTX 3070'))).toBe(8);
  });

  test('an integrated-only machine counts as one miner', () => {
    expect(countGpus(['Intel UHD Graphics 770'])).toBe(1);
  });

  test('virtual adapters and junk count as zero', () => {
    expect(countGpus(['Microsoft Basic Display Adapter', ''])).toBe(0);
    expect(countGpus([])).toBe(0);
    expect(countGpus(null)).toBe(0);
    expect(countGpus([null])).toBe(0);
  });
});

describe('parseGpuStats', () => {
  test('parses one entry per card from the nvidia-smi CSV', () => {
    const out = '0, NVIDIA GeForce RTX 4090, 4096, 24564\n1, NVIDIA GeForce RTX 4060 Ti, 2000, 16380\n';
    expect(parseGpuStats(out)).toEqual([
      { index: 0, name: 'NVIDIA GeForce RTX 4090', usedMb: 4096, totalMb: 24564 },
      { index: 1, name: 'NVIDIA GeForce RTX 4060 Ti', usedMb: 2000, totalMb: 16380 },
    ]);
  });

  test('reads the numbers positionally so a name that holds extra commas can\'t misalign them', () => {
    // used/total are always the last two fields; the name is everything between.
    expect(parseGpuStats('2, GPU, X, 100, 8192')).toEqual([
      { index: 2, name: 'GPU,X', usedMb: 100, totalMb: 8192 },
    ]);
  });

  test('a blank name field becomes null', () => {
    expect(parseGpuStats('0, , 100, 8192')).toEqual([
      { index: 0, name: null, usedMb: 100, totalMb: 8192 },
    ]);
  });

  test('skips blank lines and rows that do not parse', () => {
    expect(parseGpuStats('\n0, RTX 4090, 4096, 24564\n\ngarbage\n1, RTX 4090, notanumber, 24564\n')).toEqual([
      { index: 0, name: 'RTX 4090', usedMb: 4096, totalMb: 24564 },
    ]);
  });

  test('returns an empty list for empty or nullish input', () => {
    expect(parseGpuStats('')).toEqual([]);
    expect(parseGpuStats(null)).toEqual([]);
    expect(parseGpuStats(undefined)).toEqual([]);
  });
});

describe('parseMacGpu', () => {
  const wrap = (entries) => JSON.stringify({ SPDisplaysDataType: entries });

  test('reads the chip name off an Apple silicon Mac', () => {
    expect(parseMacGpu(wrap([{ _name: 'Apple M3 Max', sppci_model: 'Apple M3 Max', sppci_cores: '40' }])))
      .toEqual({ name: 'Apple M3 Max', count: 1 });
  });

  // The two keys have swapped roles between macOS versions, so both are read —
  // a missed rename would silently cost the label.
  test('accepts an entry carrying only the older _name key', () => {
    expect(parseMacGpu(wrap([{ _name: 'Apple M1 Pro' }]))).toEqual({ name: 'Apple M1 Pro', count: 1 });
  });

  // Same rule as Windows: an Intel Mac listing its iGPU first must still report
  // the discrete card, which is the one that would run anything.
  test('prefers the discrete card on a dual-GPU Intel Mac', () => {
    expect(parseMacGpu(wrap([
      { sppci_model: 'Intel UHD Graphics 630' },
      { sppci_model: 'AMD Radeon Pro 5500M' },
    ]))).toEqual({ name: 'AMD Radeon Pro 5500M', count: 1 });
  });

  test('returns null for anything unusable', () => {
    expect(parseMacGpu('not json')).toBeNull();
    expect(parseMacGpu('')).toBeNull();
    expect(parseMacGpu(null)).toBeNull();
    expect(parseMacGpu(undefined)).toBeNull();
    expect(parseMacGpu(wrap([]))).toBeNull();
    expect(parseMacGpu(wrap([{}]))).toBeNull();
    expect(parseMacGpu(JSON.stringify({ SPDisplaysDataType: 'nope' }))).toBeNull();
    expect(parseMacGpu(JSON.stringify({}))).toBeNull();
    expect(parseMacGpu('null')).toBeNull();
  });
});
