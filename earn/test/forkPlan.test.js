'use strict';

const {
  FP8_CERT_VERSION, COINS, FORK_NOTICE, fp8On, isFp8Job, cleanCardCoins, parseCardCoins, minesFp8, forkDrops,
  planCoins, planIsPlain, planLines, noticeCards, noticeLine,
} = require('../src/shared/forkPlan');

const RTX50 = { index: 0, major: 12, minor: 0 };
const RTX30 = { index: 1, major: 8, minor: 6 };
const base = { indices: [0, 1], cards: [RTX50, RTX30], coin: 'pearl', pearlOk: true, quantusOk: true };

describe('forkPlan', () => {
  test('constants and switches', () => {
    expect(FP8_CERT_VERSION).toBe(4);
    expect(COINS).toEqual(['pearl', 'quantus']);
    expect(FORK_NOTICE).toBe('Pearl\'s planned FP8 upgrade drops support for this card; it can mine Quantus instead.');
    expect(fp8On({ PEARL_FP8: '1' })).toBe(true);
    expect(fp8On({ PEARL_FP8: '0' })).toBe(false);
    expect(fp8On(undefined)).toBe(false);
    expect(isFp8Job(4)).toBe(true);
    expect(isFp8Job(5)).toBe(true);
    expect(isFp8Job(3)).toBe(false);
    expect(isFp8Job(null)).toBe(false);
  });

  test('cleanCardCoins keeps card indices with a known coin', () => {
    expect(cleanCardCoins({ 0: 'pearl', 1: 'quantus', 2: 'doge', x: 'pearl', '-1': 'pearl' })).toEqual({ 0: 'pearl', 1: 'quantus' });
    expect(cleanCardCoins(null)).toEqual({});
    expect(cleanCardCoins('0:pearl')).toEqual({});
  });

  test('parseCardCoins reads <card>:<coin> pairs', () => {
    expect(parseCardCoins('0:pearl, 1=Quantus')).toEqual({ map: { 0: 'pearl', 1: 'quantus' } });
    expect(parseCardCoins('0:doge').error).toMatch(/got "0:doge"/);
    expect(parseCardCoins('pearl').error).toMatch(/got "pearl"/);
    expect(parseCardCoins('').error).toMatch(/expected <card>:<coin>/);
    expect(parseCardCoins(null).error).toMatch(/expected <card>:<coin>/);
  });

  test('only an RTX 50 class card on an FP8 build mines Pearl after the fork', () => {
    expect(minesFp8(RTX50, true)).toBe(true);
    expect(minesFp8(RTX50, false)).toBe(false);
    expect(minesFp8(RTX30, true)).toBe(false);
    expect(minesFp8({ index: 0, major: 10, minor: 0 }, true)).toBe(false);
    expect(minesFp8(undefined, true)).toBe(false);
  });

  test('forkDrops: AMD, Intel and known pre-Blackwell cards, never an unknown one', () => {
    expect(forkDrops(undefined, 'amd')).toBe(true);
    expect(forkDrops(undefined, 'intel')).toBe(true);
    expect(forkDrops(RTX30, 'nvidia')).toBe(true);
    expect(forkDrops(RTX50, 'nvidia')).toBe(false);
    expect(forkDrops({ index: 0, major: 9, minor: 0 }, 'nvidia')).toBe(false);
    expect(forkDrops(undefined, 'nvidia')).toBe(false);
    expect(forkDrops(undefined, null)).toBe(false);
  });

  test('before the fork every card mines the rig\'s coin', () => {
    const plan = planCoins(base);
    expect(plan).toEqual({ whole: false, pearl: [0, 1], quantus: [], moved: [], stopped: [] });
    expect(planIsPlain(plan, 'pearl')).toBe(true);
    const q = planCoins({ ...base, coin: 'quantus' });
    expect(q).toEqual({ whole: false, pearl: [], quantus: [0, 1], moved: [], stopped: [] });
    expect(planIsPlain(q, 'quantus')).toBe(true);
    expect(planLines(plan)).toEqual([]);
  });

  test('a card\'s own choice beats the rig\'s coin', () => {
    const plan = planCoins({ ...base, cardCoins: { 1: 'quantus' } });
    expect(plan.pearl).toEqual([0]);
    expect(plan.quantus).toEqual([1]);
    expect(plan.moved).toEqual([]);
    expect(planIsPlain(plan, 'pearl')).toBe(false);
    const other = planCoins({ ...base, coin: 'quantus', cardCoins: { 0: 'pearl' } });
    expect(other.pearl).toEqual([0]);
    expect(other.quantus).toEqual([1]);
    expect(planIsPlain(other, 'quantus')).toBe(false);
  });

  test('a card whose coin has no address stops, and says which address', () => {
    const plan = planCoins({ ...base, cardCoins: { 1: 'quantus' }, quantusOk: false });
    expect(plan.stopped).toEqual([{ index: 1, why: 'no-quantus-address' }]);
    const noPearl = planCoins({ ...base, coin: 'quantus', cardCoins: { 0: 'pearl' }, pearlOk: false });
    expect(noPearl.stopped).toEqual([{ index: 0, why: 'no-pearl-address' }]);
    expect(planIsPlain(noPearl, 'quantus')).toBe(false);
    expect(planLines(plan)).toEqual([
      { level: 'error', line: 'GPU 1 is set to mine Quantus, but there is no Quantus address: not mining.' },
    ]);
    expect(planLines(noPearl)).toEqual([
      { level: 'error', line: 'GPU 0 is set to mine Pearl, but there is no Pearl address: not mining.' },
    ]);
  });

  test('once the fork is active, Pearl cards move to Quantus', () => {
    const plan = planCoins({ ...base, forkActive: true });
    expect(plan).toEqual({ whole: false, pearl: [], quantus: [0, 1], moved: [0, 1], stopped: [] });
    expect(planIsPlain(plan, 'pearl')).toBe(false);
    expect(planLines(plan)).toEqual([
      { level: 'warn', line: 'Pearl has switched to FP8, which GPU 0, 1 cannot mine. Mining Quantus instead.' },
    ]);
  });

  test('with the FP8 core an RTX 50 keeps Pearl and the rest move', () => {
    const plan = planCoins({ ...base, forkActive: true, fp8: true });
    expect(plan.pearl).toEqual([0]);
    expect(plan.quantus).toEqual([1]);
    expect(plan.moved).toEqual([1]);
  });

  test('with no Quantus address the fork stops the cards, with a reason', () => {
    const plan = planCoins({ ...base, forkActive: true, quantusOk: false });
    expect(plan.stopped).toEqual([{ index: 0, why: 'fork' }, { index: 1, why: 'fork' }]);
    expect(planLines(plan)).toEqual([{
      level: 'error',
      line: 'GPU 0, 1 cannot mine Pearl since its FP8 upgrade, and there is no Quantus address to mine to instead.'
        + ' Add one and start again.',
    }]);
  });

  test('a rig whose cards are not known is planned as one card', () => {
    const plan = planCoins({ indices: [], coin: 'pearl', pearlOk: true, quantusOk: true });
    expect(plan).toEqual({ whole: true, pearl: [null], quantus: [], moved: [], stopped: [] });
    const forked = planCoins({ indices: null, cards: null, coin: 'pearl', pearlOk: true, quantusOk: true, forkActive: true, fp8: true });
    expect(forked.quantus).toEqual([null]);
    expect(planLines(forked)[0].line).toBe('Pearl has switched to FP8, which this rig cannot mine. Mining Quantus instead.');
    const stopped = planCoins({ indices: [], coin: 'pearl', pearlOk: true, quantusOk: false, forkActive: true });
    expect(planLines(stopped)[0].line).toMatch(/^This rig cannot mine Pearl since its FP8 upgrade/);
  });

  test('the notice goes to cards on Pearl that the fork drops', () => {
    const plan = planCoins(base);
    expect(noticeCards({ plan, cards: base.cards, vendor: 'nvidia' })).toEqual([1]);
    expect(noticeCards({ plan, cards: base.cards, vendor: 'nvidia', env: {} })).toEqual([1]);
    // Not a card already on Quantus, and none once the fork is active or with the switch off.
    const picked = planCoins({ ...base, cardCoins: { 1: 'quantus' } });
    expect(noticeCards({ plan: picked, cards: base.cards, vendor: 'nvidia' })).toEqual([]);
    expect(noticeCards({ plan, cards: base.cards, vendor: 'nvidia', forkActive: true })).toEqual([]);
    expect(noticeCards({ plan, cards: base.cards, vendor: 'nvidia', env: { LLMJOB_FORK_NOTICE: '0' } })).toEqual([]);
    // An AMD rig: every card, though nvidia-smi lists none.
    expect(noticeCards({ plan, cards: null, vendor: 'amd' })).toEqual([0, 1]);
    // Cards nothing is known about are not told.
    expect(noticeCards({ plan, cards: [], vendor: 'nvidia' })).toEqual([]);
  });

  test('the notice line names the cards and, with no Quantus address, what to do', () => {
    expect(noticeLine([], true)).toBeNull();
    expect(noticeLine([0, 2], true)).toBe('GPU 0, 2: ' + FORK_NOTICE);
    expect(noticeLine([1], false)).toBe('GPU 1: ' + FORK_NOTICE + ' Add a Quantus address and it switches by itself.');
    expect(noticeLine([null], true)).toBe(FORK_NOTICE);
  });
});
