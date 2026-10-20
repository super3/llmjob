'use strict';

const { saveKeys, lookupKeys, lostModels, forVersion, remember } = require('../src/shared/gpuLoss');

describe('saveKeys and lookupKeys', () => {
  const cards = [{ index: 0, name: 'NVIDIA RTX PRO 4500 Blackwell' }, { index: 1, name: 'RTX 4070' }];
  const PRO = '0:NVIDIA RTX PRO 4500 Blackwell';

  // The bus id stays with the card when others are added or moved, and both
  // tools' spellings of it are one card. The number and name is saved beside
  // it, marked, for a later start that gets no bus ids.
  test('save under the bus id, in one spelling, and the marked number and name', () => {
    expect(saveKeys(0, cards, { 0: '00000000:01:00.0' })).toEqual(['0:1:0.0', PRO + ' +bus']);
    expect(saveKeys(0, cards, {}, '0000:01:00.0')).toEqual(['0:1:0.0', PRO + ' +bus']);
    // The one the server loaded on wins over nvidia-smi's for the planned card.
    expect(saveKeys(0, cards, { 0: '00000000:01:00.0' }, '0000:02:00.0')).toEqual(['0:2:0.0', PRO + ' +bus']);
  });

  test('save under the number and name alone without a bus id', () => {
    expect(saveKeys(1, cards, { 1: 'unknown id' })).toEqual(['1:RTX 4070']);
    expect(saveKeys(1, cards)).toEqual(['1:RTX 4070']);
    expect(saveKeys(3, [null, { index: 3 }], null)).toEqual(['3:']);
    expect(saveKeys(3)).toEqual(['3:']);
    expect(saveKeys(null, [], {})).toEqual(['auto']);
    expect(saveKeys(undefined)).toEqual(['auto']);
  });

  test('look up by the bus id and an unmarked number and name, or by both number-and-name keys', () => {
    expect(lookupKeys(0, cards, { 0: '00000000:01:00.0' })).toEqual(['0:1:0.0', PRO]);
    expect(lookupKeys(0, cards, {})).toEqual([PRO, PRO + ' +bus']);
    expect(lookupKeys(null, [], {})).toEqual(['auto', 'auto +bus']);
  });

  const lost = (saved, looked) => lostModels(remember({ version: '1', cards: {} }, saved, 'Qwen'), looked);

  // The review's case: no nvidia-smi, so the start looks up 'auto', but
  // llama-server printed the bus id the model was on, and the loss was saved
  // under that alone. The model went back on the failing card.
  test('a loss saved with the bus id llama-server printed is found on a rig without nvidia-smi', () => {
    expect(lost(saveKeys(null, [], {}, '0000:01:00.0'), lookupKeys(null, [], {}))).toEqual(['Qwen']);
  });

  // nvidia-smi gave bus ids on one start and not the next, in either order.
  test('a loss is found whether or not the start that saved it had bus ids', () => {
    const bus = { 0: '00000000:01:00.0' };
    expect(lost(saveKeys(0, cards, bus), lookupKeys(0, cards, {}))).toEqual(['Qwen']);
    expect(lost(saveKeys(0, cards, {}), lookupKeys(0, cards, bus))).toEqual(['Qwen']);
    expect(lost(saveKeys(0, cards, bus), lookupKeys(0, cards, bus))).toEqual(['Qwen']);
  });

  // Numbers change when cards are moved. A start that knows the bus ids must
  // not hold back a healthy card that now has the failing card's number.
  test('a card moved to another number is not mistaken for the failing one', () => {
    const before = { 0: '00000000:01:00.0', 1: '00000000:02:00.0' };
    const after = { 0: '00000000:02:00.0', 1: '00000000:01:00.0' };
    const twins = [{ index: 0, name: 'RTX 3090' }, { index: 1, name: 'RTX 3090' }];
    const record = remember({ version: '1', cards: {} }, saveKeys(0, twins, before), 'Qwen');
    expect(lostModels(record, lookupKeys(0, twins, after))).toEqual([]);
    expect(lostModels(record, lookupKeys(1, twins, after))).toEqual(['Qwen']);
  });

  test('lostModels lists each model once, across the keys', () => {
    const record = { version: '1', cards: { a: ['Qwen'], b: ['Qwen', 'Gemma'] } };
    expect(lostModels(record, ['a', 'b', 'c'])).toEqual(['Qwen', 'Gemma']);
    expect(lostModels(record, [])).toEqual([]);
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
  test('adds a model under each key once, leaving the old record alone', () => {
    const a = { version: '1', cards: {} };
    const b = remember(a, ['0:1:0.0', '0:Pro +bus'], 'Qwen');
    expect(b).toEqual({ version: '1', cards: { '0:1:0.0': ['Qwen'], '0:Pro +bus': ['Qwen'] } });
    expect(a.cards).toEqual({});
    const c = remember(b, ['0:1:0.0'], 'Gemma');
    expect(c.cards['0:1:0.0']).toEqual(['Qwen', 'Gemma']);
    expect(c.cards['0:Pro +bus']).toEqual(['Qwen']);
    expect(remember(c, ['0:1:0.0', '0:Pro +bus'], 'Qwen')).toBe(c);
  });
});
