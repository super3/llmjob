'use strict';

const { planLlmInstances } = require('../src/shared/llmPlan');
const { ALL_LAYERS } = require('../src/shared/vram');

// A model with a 6 GB floor, 42 layers, ~5.8 GB full offload — like the shipped
// Gemma-4-E4B config.
const MODEL = { layers: 42, vramFullMb: 5800, minVramMb: 6144 };

describe('planLlmInstances', () => {
  test('one entry per eligible card, sorted by index, each fully offloaded', () => {
    const plan = planLlmInstances([
      { index: 1, usedMb: 1000, totalMb: 24000 }, // 23 GB free → full offload
      { index: 0, usedMb: 500, totalMb: 12000 },  // 11.5 GB free → full offload
    ], MODEL, 0);
    expect(plan.map((p) => p.index)).toEqual([0, 1]); // sorted
    expect(plan.every((p) => p.nGpuLayers === ALL_LAYERS)).toBe(true);
    expect(plan[0]).toEqual({ index: 0, freeMb: 11500, nGpuLayers: ALL_LAYERS });
  });

  test('skips cards without enough free VRAM for the model', () => {
    const plan = planLlmInstances([
      { index: 0, usedMb: 20000, totalMb: 24000 }, // 4 GB free → below the 6 GB floor
      { index: 1, usedMb: 1000, totalMb: 24000 },  // 23 GB free → fits
    ], MODEL, 0);
    expect(plan.map((p) => p.index)).toEqual([1]);
  });

  test('honours the mining reserve when deciding a card can hold the model', () => {
    // 8 GB free, reserve 2 GB → 6 GB budget ≈ full offload still fits the 5.8 GB model.
    const plan = planLlmInstances([{ index: 0, usedMb: 16000, totalMb: 24000 }], MODEL, 2048);
    expect(plan).toHaveLength(1);
    expect(plan[0].nGpuLayers).toBe(ALL_LAYERS);
  });

  test('drops a card when the reserve leaves no room for any layer', () => {
    // Exactly at the 6144 MB floor (passes hasEnoughVram) but a huge reserve
    // wipes the budget → 0 layers → not served.
    const plan = planLlmInstances([{ index: 0, usedMb: 24000 - 6144, totalMb: 24000 }], MODEL, 6144);
    expect(plan).toEqual([]);
  });

  test('returns [] when cards are measured but none fit', () => {
    const plan = planLlmInstances([
      { index: 0, usedMb: 22000, totalMb: 24000 }, // 2 GB free
      { index: 1, usedMb: 7000, totalMb: 8000 },   // 1 GB free
    ], MODEL, 0);
    expect(plan).toEqual([]);
  });

  test('falls back to one unknown-placement instance when no card is measurable', () => {
    expect(planLlmInstances(null, MODEL, 0)).toEqual([{ index: null, freeMb: null, nGpuLayers: ALL_LAYERS }]);
    expect(planLlmInstances([], MODEL, 0)).toEqual([{ index: null, freeMb: null, nGpuLayers: ALL_LAYERS }]);
    // entries present but unparseable (no numeric VRAM) → still "unmeasured"
    expect(planLlmInstances([{ index: 0 }, null], MODEL, 0)).toEqual([{ index: null, freeMb: null, nGpuLayers: ALL_LAYERS }]);
    // Unmeasurable VRAM still means full offload — llama.cpp decides placement,
    // and we never ask for a partial one.
    expect(planLlmInstances(null, {}, 0)).toEqual([{ index: null, freeMb: null, nGpuLayers: ALL_LAYERS }]);
  });

  test('ignores malformed card entries', () => {
    const plan = planLlmInstances([
      null,
      { index: -1, usedMb: 0, totalMb: 24000 },      // bad index
      { index: 'x', usedMb: 0, totalMb: 24000 },     // non-numeric index
      { index: 2, usedMb: 'n/a', totalMb: 24000 },   // non-numeric used
      { index: 0, usedMb: 1000, totalMb: 24000 },    // valid → kept
    ], MODEL, 0);
    expect(plan.map((p) => p.index)).toEqual([0]);
  });

  test('a model with no VRAM floor puts an instance on every measured card', () => {
    const plan = planLlmInstances([
      { index: 0, usedMb: 100, totalMb: 2000 },
      { index: 1, usedMb: 100, totalMb: 2000 },
    ], { layers: 10, vramFullMb: 1000 }, 0);
    expect(plan.map((p) => p.index)).toEqual([0, 1]);
  });

  // --llm-max-instances: an operator ceiling on top of VRAM eligibility. Note
  // this caps the FLEET SIZE only — the loading stampede is handled by starting
  // instances one at a time (LlmFleet), not by planning fewer of them.
  test('maxInstances truncates to the lowest GPU indices', () => {
    const cards = [
      { index: 0, usedMb: 1000, totalMb: 24000 },
      { index: 1, usedMb: 1000, totalMb: 24000 },
      { index: 2, usedMb: 1000, totalMb: 24000 },
    ];
    expect(planLlmInstances(cards, MODEL, 0, { maxInstances: 2 }).map((p) => p.index)).toEqual([0, 1]);
    // A cap at or above the eligible count changes nothing.
    expect(planLlmInstances(cards, MODEL, 0, { maxInstances: 3 }).map((p) => p.index)).toEqual([0, 1, 2]);
    expect(planLlmInstances(cards, MODEL, 0, { maxInstances: 9 }).map((p) => p.index)).toEqual([0, 1, 2]);
    // Zero/garbage still serves one card: serving from one beats serving none.
    expect(planLlmInstances(cards, MODEL, 0, { maxInstances: 0 }).map((p) => p.index)).toEqual([0]);
    expect(planLlmInstances(cards, MODEL, 0, { maxInstances: 'lots' }).map((p) => p.index)).toEqual([0, 1, 2]);
    // No cap set — null must NOT be read as 0 (Number(null) === 0 would shrink
    // every default fleet to a single card).
    expect(planLlmInstances(cards, MODEL, 0, { maxInstances: null }).map((p) => p.index)).toEqual([0, 1, 2]);
    expect(planLlmInstances(cards, MODEL, 0, {}).map((p) => p.index)).toEqual([0, 1, 2]);
  });
});
