'use strict';

const { pickGpu } = require('../src/shared/gpu');

describe('pickGpu', () => {
  test('picks the real GPU and skips the basic display adapter', () => {
    expect(pickGpu(['Microsoft Basic Display Adapter', 'NVIDIA GeForce RTX 4090'])).toBe('NVIDIA GeForce RTX 4090');
  });

  test('returns the first real adapter when several are present', () => {
    expect(pickGpu(['NVIDIA GeForce RTX 4090', 'AMD Radeon RX 7900'])).toBe('NVIDIA GeForce RTX 4090');
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
