'use strict';

const { computeGpuLayers, requiredVramMb, hasEnoughVram, pickLlmGpu } = require('../src/shared/vram');

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

describe('pickLlmGpu', () => {
  test('null for a non-array or empty input', () => {
    expect(pickLlmGpu(null)).toBeNull();
    expect(pickLlmGpu(undefined)).toBeNull();
    expect(pickLlmGpu('nope')).toBeNull();
    expect(pickLlmGpu([])).toBeNull();
  });

  test('picks the card with the most free VRAM (total − used)', () => {
    const cards = [
      { index: 0, usedMb: 3000, totalMb: 16000 }, // free 13000
      { index: 1, usedMb: 1000, totalMb: 16000 }, // free 15000  ← most
      { index: 2, usedMb: 8000, totalMb: 16000 }, // free 8000
    ];
    expect(pickLlmGpu(cards)).toEqual({ index: 1, freeMb: 15000 });
  });

  test('breaks ties on equal free VRAM by the lower index', () => {
    const cards = [
      { index: 2, usedMb: 3000, totalMb: 16000 },
      { index: 0, usedMb: 3000, totalMb: 16000 }, // same free, lower index ← wins
      { index: 1, usedMb: 3000, totalMb: 16000 },
    ];
    expect(pickLlmGpu(cards)).toEqual({ index: 0, freeMb: 13000 });
  });

  test('index 0 is a valid pick (not treated as falsy)', () => {
    expect(pickLlmGpu([{ index: 0, usedMb: 1000, totalMb: 16000 }])).toEqual({ index: 0, freeMb: 15000 });
  });

  test('skips unparseable cards (null entry, negative/NaN index, NaN used/total)', () => {
    const cards = [
      null,                                        // falsy entry
      { index: -1, usedMb: 0, totalMb: 16000 },    // negative index
      { index: 'x', usedMb: 0, totalMb: 16000 },   // NaN index
      { index: 1, usedMb: 'x', totalMb: 16000 },   // NaN used
      { index: 2, usedMb: 0, totalMb: 'x' },       // NaN total
      { index: 3, usedMb: 2000, totalMb: 16000 },  // valid → free 14000
    ];
    expect(pickLlmGpu(cards)).toEqual({ index: 3, freeMb: 14000 });
  });

  test('null when every card is unparseable', () => {
    expect(pickLlmGpu([null, { index: -1, usedMb: 0, totalMb: 8000 }, { index: 0, usedMb: 1, totalMb: 'x' }])).toBeNull();
  });

  test('clamps negative free VRAM (used > total) to zero', () => {
    expect(pickLlmGpu([{ index: 0, usedMb: 17000, totalMb: 16000 }])).toEqual({ index: 0, freeMb: 0 });
  });
});
