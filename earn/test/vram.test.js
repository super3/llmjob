'use strict';

const { ALL_LAYERS, computeGpuLayers, requiredVramMb, requiredFreeMb, hasEnoughVram, pickLlmGpu, beforeMiner, waitForFreeVram } = require('../src/shared/vram');

const MODEL = { layers: 16, vramFullMb: 1600 }; // perLayer = 100 MB

describe('computeGpuLayers', () => {
  // All-or-nothing by design: a partial offload leaves the remaining layers in
  // host RAM as unreclaimable memory, which OOM'd an 8 GB 13-GPU rig and got the
  // miner killed. A card that can't hold the whole model is skipped instead.
  test('full offload when the budget covers the whole model', () => {
    expect(computeGpuLayers(24000, MODEL, 2048)).toBe(ALL_LAYERS); // lots free
    expect(computeGpuLayers(3648, MODEL, 2048)).toBe(ALL_LAYERS);  // budget exactly 1600
  });

  test('zero — never a partial offload — when only part of the model would fit', () => {
    expect(computeGpuLayers(2848, MODEL, 2048)).toBe(0); // budget 800 of 1600 needed
    expect(computeGpuLayers(3647, MODEL, 2048)).toBe(0); // one MB short still means no
  });

  test('zero when nothing fits after the mining reserve', () => {
    expect(computeGpuLayers(2048, MODEL, 2048)).toBe(0); // budget 0
    expect(computeGpuLayers(1000, MODEL, 2048)).toBe(0); // negative budget
  });

  test('zero for an invalid model or non-numeric free VRAM', () => {
    expect(computeGpuLayers(24000, { vramFullMb: 0 }, 0)).toBe(0);
    expect(computeGpuLayers('nope', MODEL, 0)).toBe(0);
    expect(computeGpuLayers(24000, null, 0)).toBe(0);
  });

  test('reserve defaults to 0 when omitted', () => {
    expect(computeGpuLayers(1600, MODEL)).toBe(ALL_LAYERS); // budget == full
  });

  // The count we pass is deliberately bigger than any real model: llama.cpp
  // clamps it, which beats guessing a layer count that could come in too low.
  test('ALL_LAYERS is larger than any plausible layer count', () => {
    expect(ALL_LAYERS).toBeGreaterThan(200);
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

describe('requiredFreeMb', () => {
  // What a refusal should quote: the gate that actually blocked the card. When
  // co-running that is the offload budget (model + reserve), which exceeds the
  // preflight floor — reporting the floor told a user they needed less than they
  // did and refused them anyway.
  test('co-running: the model plus the mining reserve outranks the floor', () => {
    expect(requiredFreeMb({ vramFullMb: 3800, minVramMb: 4352 }, 2048)).toBe(5848);
  });

  test('llm-only: with no reserve the floor is the binding constraint', () => {
    expect(requiredFreeMb({ vramFullMb: 3800, minVramMb: 4352 }, 0)).toBe(4352);
    expect(requiredFreeMb({ vramFullMb: 3800, minVramMb: 4352 })).toBe(4352);
  });

  test('falls back to the full-offload figure when no floor is configured', () => {
    expect(requiredFreeMb({ vramFullMb: 3800 }, 512)).toBe(4312);
  });

  test('a model with neither figure needs nothing but the reserve', () => {
    expect(requiredFreeMb({}, 2048)).toBe(2048);
    expect(requiredFreeMb(null, 0)).toBe(0);
  });

  test('ignores an unparseable reserve', () => {
    expect(requiredFreeMb({ vramFullMb: 3800, minVramMb: 4352 }, 'x')).toBe(4352);
  });
});

describe('waitForFreeVram', () => {
  const card = (freeMb) => ({ index: 0, freeMb });
  const reads = (...free) => {
    const read = jest.fn();
    for (const f of free) read.mockResolvedValueOnce(f == null ? [] : [card(f)]);
    return read;
  };
  const pick = (cards) => cards[0] || null;

  test('returns as soon as the card has room, or there is no card', async () => {
    const read = reads(100, 500);
    await expect(waitForFreeVram({ needMb: 400, read, pick, stepMs: 0 })).resolves.toEqual(card(500));
    expect(read).toHaveBeenCalledTimes(2);
    await expect(waitForFreeVram({ needMb: 400, read: reads(null), pick, stepMs: 0 })).resolves.toBeNull();
  });

  test('stops at the deadline with the last reading', async () => {
    const read = reads(100, 100, 100);
    await expect(waitForFreeVram({ needMb: 400, read, pick, stepMs: 0, timeoutMs: 0 })).resolves.toEqual(card(100));
    expect(read).toHaveBeenCalledTimes(1);
  });

  test('stops after `tries` readings, or when no longer wanted', async () => {
    const read = reads(100, 100, 100);
    await expect(waitForFreeVram({ needMb: 400, read, pick, stepMs: 0, tries: 2 })).resolves.toBeUndefined();
    expect(read).toHaveBeenCalledTimes(2);
    const never = reads(500);
    await expect(waitForFreeVram({ needMb: 400, read: never, pick, stillWanted: () => false })).resolves.toBeUndefined();
    expect(never).not.toHaveBeenCalled();
  });
});

describe('minReserveMb', () => {
  const M = { vramFullMb: 1000, minVramMb: 1100, minReserveMb: 500 };

  test('requiredFreeMb charges the model\'s own reserve when the caller\'s is smaller', () => {
    expect(requiredFreeMb(M, 0)).toBe(1500);
    expect(requiredFreeMb(M, 200)).toBe(1500);
    expect(requiredFreeMb(M, 800)).toBe(1800);
  });

  test('computeGpuLayers sets the same reserve aside', () => {
    expect(computeGpuLayers(1500, M, 0)).toBe(ALL_LAYERS);
    expect(computeGpuLayers(1499, M, 0)).toBe(0);
    expect(computeGpuLayers(1799, M, 800)).toBe(0);
  });
});

describe('beforeMiner', () => {
  const cards = [
    { index: 0, name: 'A', usedMb: 3600, totalMb: 32607 },
    { index: 1, name: 'B', usedMb: 1000, totalMb: 12282 },
  ];
  test('gives the reserve back on the mining cards only', () => {
    expect(beforeMiner(cards, [0], 2600)).toEqual([
      { index: 0, name: 'A', usedMb: 1000, totalMb: 32607 },
      { index: 1, name: 'B', usedMb: 1000, totalMb: 12282 },
    ]);
    expect(cards[0].usedMb).toBe(3600);   // a copy, not the caller's list
  });
  test('never goes below nothing used', () => {
    expect(beforeMiner(cards, [1], 2600)[1].usedMb).toBe(0);
  });
  test('leaves cards it cannot read, and takes junk input', () => {
    expect(beforeMiner([null, { index: 0, usedMb: 'x' }], [0], 2600)).toEqual([null, { index: 0, usedMb: 'x' }]);
    expect(beforeMiner(null, null, null)).toEqual([]);
    expect(beforeMiner(cards, [0], undefined)[0].usedMb).toBe(3600);
  });
  // Counting the miner once: a model needs full + reserve of this figure, which
  // is needing full of the reading made with the miner running.
  test('a card that fits beside the running miner passes the usual check', () => {
    const M = { vramFullMb: 25100, minVramMb: 25670, minReserveMb: 2600 };
    const reading = { index: 0, usedMb: 32607 - 25100, totalMb: 32607 };
    const [c] = beforeMiner([reading], [0], 2600);
    expect(computeGpuLayers(c.totalMb - c.usedMb, M, 2600)).toBe(ALL_LAYERS);
    expect(computeGpuLayers(reading.totalMb - reading.usedMb, M, 2600)).toBe(0);
  });
});
