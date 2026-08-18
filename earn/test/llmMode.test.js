'use strict';

const { MODES, DEFAULT_MODE, isValidMode, normalizeMode, resolvePlan } = require('../src/shared/llmMode');

describe('llmMode', () => {
  test('exposes the modes and an auto (co-run) default', () => {
    expect(MODES).toEqual(['mining', 'llm', 'auto']);
    expect(DEFAULT_MODE).toBe('auto');
    // The default must actually serve inference — a mining-only default left
    // headless rigs (HiveOS flight sheets pass no --mode) silently not serving.
    expect(resolvePlan(DEFAULT_MODE, { canMine: true, canLlm: true })).toEqual({ miner: true, llm: true });
  });

  test('isValidMode', () => {
    expect(isValidMode('auto')).toBe(true);
    expect(isValidMode('nope')).toBe(false);
  });

  const both = { canMine: true, canLlm: true };

  test('mining runs only the miner', () => {
    expect(resolvePlan('mining', both)).toEqual({ miner: true, llm: false });
  });

  test('llm pauses the miner and runs inference', () => {
    expect(resolvePlan('llm', both)).toEqual({ miner: false, llm: true });
  });

  test('auto co-runs miner + llm, and the retired both maps onto it', () => {
    expect(resolvePlan('both', both)).toEqual({ miner: true, llm: true });
    expect(resolvePlan('auto', both)).toEqual({ miner: true, llm: true });
  });

  test('respects what is actually possible', () => {
    expect(resolvePlan('both', { canMine: false, canLlm: true })).toEqual({ miner: false, llm: true });
    expect(resolvePlan('llm', { canLlm: false })).toEqual({ miner: false, llm: false });
    expect(resolvePlan('mining', { canMine: true })).toEqual({ miner: true, llm: false });
  });

  test('defaults to mining-only for an unknown mode and empty ctx', () => {
    expect(resolvePlan('bogus', both)).toEqual({ miner: true, llm: false });
    expect(resolvePlan('mining')).toEqual({ miner: false, llm: false });
  });

  // 'both' ("Mining+LLM") was retired in the v2 mocks: it resolved to exactly
  // what 'auto' resolves to, so it was a second name for one plan. A rig that
  // stored it must land on the same plan rather than falling through to
  // mining-only, which would drop the LLM with nothing to explain it.
  test('the retired both mode still resolves, as auto', () => {
    expect(MODES).not.toContain('both');
    expect(normalizeMode('both')).toBe('auto');
    expect(isValidMode('both')).toBe(true);
    const ctx = { canMine: true, canLlm: true };
    expect(resolvePlan('both', ctx)).toEqual(resolvePlan('auto', ctx));
    expect(resolvePlan('both', ctx)).toEqual({ miner: true, llm: true });
    // a mode that never existed still means mining-only
    expect(normalizeMode('turbo')).toBe('turbo');
    expect(isValidMode('turbo')).toBe(false);
  });
});
