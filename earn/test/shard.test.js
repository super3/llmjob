'use strict';

const { pickShardPlan, cardFreeMb } = require('../src/shared/shard');

// minVramMb-only catalog fixture (ascending); the real catalog is exercised via
// config.test.js. Keeps these assertions independent of the shipped VRAM floors.
const CATALOG = [
  { id: 'small', minVramMb: 6000 },
  { id: 'mid', minVramMb: 20000 },
  { id: 'big', minVramMb: 30000 },
];
const card = (index, usedMb, totalMb) => ({ index, usedMb, totalMb });

describe('cardFreeMb', () => {
  test('free = total − used, clamped at zero, 0 on unparseable', () => {
    expect(cardFreeMb(card(0, 2000, 16000))).toBe(14000);
    expect(cardFreeMb(card(0, 17000, 16000))).toBe(0); // used > total
    expect(cardFreeMb(card(0, 'x', 16000))).toBe(0);
    expect(cardFreeMb(null)).toBe(0);
  });
});

describe('pickShardPlan', () => {
  test('null when there are not at least two usable cards', () => {
    expect(pickShardPlan(null, CATALOG)).toBeNull();
    expect(pickShardPlan([card(0, 2000, 16000)], CATALOG)).toBeNull();       // single card
    expect(pickShardPlan([card(0, 2000, 16000), card(1, 16000, 16000)], CATALOG)).toBeNull(); // 2nd card has 0 free
  });

  test('null for a missing/empty catalog', () => {
    expect(pickShardPlan([card(0, 2000, 16000), card(1, 2000, 16000)], null)).toBeNull();
    expect(pickShardPlan([card(0, 2000, 16000), card(1, 2000, 16000)], [])).toBeNull();
  });

  test('null when even all cards together cannot host the smallest model', () => {
    expect(pickShardPlan([card(0, 13000, 16000), card(1, 14000, 16000)], CATALOG)).toBeNull(); // 3000+2000 < 6000
  });

  test('null when a single card already fits the chosen model (no need to shard)', () => {
    // free 21000 + 5000 = 26000 → picks mid (20000); best card (21000) fits it alone
    expect(pickShardPlan([card(0, 3000, 24000), card(1, 3000, 8000)], CATALOG)).toBeNull();
  });

  test('shards the largest model that fits the aggregate across the biggest cards', () => {
    // two 16 GB cards, ~14 GB free each → aggregate 28000 → mid (20000); no single
    // card fits, so shard across both, weighted by free VRAM.
    const plan = pickShardPlan([card(0, 2000, 16000), card(1, 2000, 16000)], CATALOG);
    expect(plan).toMatchObject({
      model: { id: 'mid' },
      devices: [0, 1],
      tensorSplit: [14000, 14000],
      mainGpu: 0,
      freeMb: 28000,
    });
  });

  test('keeps a mining reserve per card and excludes non-serving cards with a 0 weight', () => {
    // three cards; the tiny one (index 2) is not needed and gets a 0 split entry.
    const plan = pickShardPlan(
      [card(0, 2000, 16000), card(1, 2000, 16000), card(2, 1000, 2000)], CATALOG, 2048);
    // free per big card 11952; aggregate over the two big ones covers mid (20000).
    // The tiny card (index 2, 0 free after reserve) is excluded from the shard.
    expect(plan.devices).toEqual([0, 1]);
    expect(plan.tensorSplit).toEqual([11952, 11952]); // only the two serving cards
    expect(plan.mainGpu).toBe(0);
  });

  test('orders the split vector by physical card index even when cards arrive unsorted', () => {
    const plan = pickShardPlan([card(2, 2000, 16000), card(0, 2000, 16000)], CATALOG);
    expect(plan.tensorSplit).toEqual([14000, 0, 14000]); // indices 0 and 2 serve; 1 is a gap
    expect(plan.devices).toEqual([0, 2]);
    expect(plan.mainGpu).toBe(0);
  });
});
