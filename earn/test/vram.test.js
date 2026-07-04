'use strict';

const { computeGpuLayers } = require('../src/shared/vram');

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
