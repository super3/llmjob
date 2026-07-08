'use strict';

const { pickGpu, countGpus } = require('../src/shared/gpu');

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
