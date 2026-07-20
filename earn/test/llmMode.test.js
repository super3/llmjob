'use strict';

const { MODES, DEFAULT_MODE, isValidMode, resolvePlan } = require('../src/shared/llmMode');

describe('llmMode', () => {
  test('exposes the modes and a mining default', () => {
    expect(MODES).toEqual(['mining', 'both', 'llm', 'auto']);
    expect(DEFAULT_MODE).toBe('mining');
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

  test('both and auto co-run miner + llm', () => {
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
});
