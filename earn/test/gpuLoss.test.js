'use strict';

const { cardKey, forVersion, remember } = require('../src/shared/gpuLoss');

describe('cardKey', () => {
  const cards = [{ index: 0, name: 'NVIDIA RTX PRO 4500 Blackwell' }, { index: 1, name: 'RTX 4070' }];

  // The bus id stays with the card when others are added or moved, and both
  // tools' spellings of it are one card.
  test('is the bus id, in one spelling, when there is one', () => {
    expect(cardKey(0, cards, { 0: '00000000:01:00.0' })).toBe('0:1:0.0');
    expect(cardKey(0, cards, {}, '0000:01:00.0')).toBe('0:1:0.0');
    // The one the server loaded on wins over nvidia-smi's for the planned card.
    expect(cardKey(0, cards, { 0: '00000000:01:00.0' }, '0000:02:00.0')).toBe('0:2:0.0');
  });

  test('is the number and name without one', () => {
    expect(cardKey(1, cards, { 1: 'unknown id' })).toBe('1:RTX 4070');
    expect(cardKey(1, cards)).toBe('1:RTX 4070');
    expect(cardKey(3, [null, { index: 3 }], null)).toBe('3:');
    expect(cardKey(3)).toBe('3:');
  });

  test('is one key for the card llama.cpp picked itself', () => {
    expect(cardKey(null, [], {})).toBe('auto');
    expect(cardKey(undefined)).toBe('auto');
  });
});

describe('forVersion', () => {
  test('keeps a record from this version', () => {
    const saved = { version: '0.6.5', cards: { '0:1:0.0': ['Qwen3.8-27B-UD-Q4_K_XL'] } };
    expect(forVersion(saved, '0.6.5')).toEqual(saved);
  });

  // An update may have fixed what went wrong, so it gets a fresh try.
  test('starts empty for another version, no file, or junk', () => {
    const empty = { version: '0.6.6', cards: {} };
    expect(forVersion({ version: '0.6.5', cards: { '0:1:0.0': ['x'] } }, '0.6.6')).toEqual(empty);
    expect(forVersion({}, '0.6.6')).toEqual(empty);
    expect(forVersion(null, '0.6.6')).toEqual(empty);
    expect(forVersion({ version: '0.6.6', cards: 'x' }, '0.6.6')).toEqual(empty);
    expect(forVersion({ version: '0.6.6', cards: { a: 'x', b: ['m', 3, null] } }, '0.6.6'))
      .toEqual({ version: '0.6.6', cards: { b: ['m'] } });
  });
});

describe('remember', () => {
  test('adds a model to a card once, leaving the old record alone', () => {
    const a = { version: '1', cards: {} };
    const b = remember(a, '0:1:0.0', 'Qwen');
    expect(b).toEqual({ version: '1', cards: { '0:1:0.0': ['Qwen'] } });
    expect(a.cards).toEqual({});
    const c = remember(b, '0:1:0.0', 'Gemma');
    expect(c.cards['0:1:0.0']).toEqual(['Qwen', 'Gemma']);
    expect(remember(c, '0:1:0.0', 'Gemma')).toBe(c);
  });
});
