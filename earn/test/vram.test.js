'use strict';

const { computeGpuLayers, requiredVramMb, hasEnoughVram } = require('../src/shared/vram');

const MODEL = { layers: 16, vramFullMb: 1600 }; // perLayer = 100 MB

describe('computeGpuLayers', () => {
  test('full offload when the budget covers the whole model', () => {
    expect(computeGpuLayers(24000, MODEL, 2048)).toBe(16); // lots free
    expect(computeGpuLayers(3648, MODEL, 2048)).toBe(16);  // budget exactly 1600
  });

  test('partial offload — only the layers that fit', () => {
    expect(computeGpuLayers(2848, MODEL, 2048)).toBe(8); // budget 800 → 8 layers
  });

  test('zero when nothing fits after the mining reserve', () => {
    expect(computeGpuLayers(2048, MODEL, 2048)).toBe(0); // budget 0
    expect(computeGpuLayers(1000, MODEL, 2048)).toBe(0); // negative budget
  });

  test('zero for an invalid model or non-numeric free VRAM', () => {
    expect(computeGpuLayers(24000, { layers: 0, vramFullMb: 1600 }, 0)).toBe(0);
    expect(computeGpuLayers(24000, { layers: 16, vramFullMb: 0 }, 0)).toBe(0);
    expect(computeGpuLayers('nope', MODEL, 0)).toBe(0);
    expect(computeGpuLayers(24000, null, 0)).toBe(0);
  });

  test('reserve defaults to 0 when omitted', () => {
    expect(computeGpuLayers(1600, MODEL)).toBe(16); // budget == full
  });
});

describe('requiredVramMb', () => {
  test('prefers an explicit minVramMb floor', () => {
    expect(requiredVramMb({ minVramMb: 6144, vramFullMb: 5800 })).toBe(6144);
  });
  test('falls back to vramFullMb when no explicit floor', () => {
    expect(requiredVramMb({ vramFullMb: 5800 })).toBe(5800);
  });
  test('zero when neither is set or the model is missing', () => {
    expect(requiredVramMb({})).toBe(0);
    expect(requiredVramMb(null)).toBe(0);
  });
});

describe('hasEnoughVram', () => {
  const M = { minVramMb: 6144 };
  test('true when free covers the requirement, incl. the exact boundary', () => {
    expect(hasEnoughVram(8000, M)).toBe(true);
    expect(hasEnoughVram(6144, M)).toBe(true);
  });
  test('false when free VRAM falls short', () => {
    expect(hasEnoughVram(6000, M)).toBe(false);
  });
  test('null when free VRAM cannot be measured', () => {
    expect(hasEnoughVram(null, M)).toBeNull();
    expect(hasEnoughVram('nope', M)).toBeNull();
  });
  test('always allowed when the model configures no floor', () => {
    expect(hasEnoughVram(0, {})).toBe(true);
    expect(hasEnoughVram(null, {})).toBe(true);
  });
});
