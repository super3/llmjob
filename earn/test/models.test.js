'use strict';

const { allModels, pickModel, ctxLadder, needsMmproj } = require('../src/shared/models');
const { LLM } = require('../src/shared/config');
const { requiredFreeMb } = require('../src/shared/vram');

// Two tiers with round numbers, so the assertions below are about the SELECTION
// RULE and not about whatever the shipped config happens to say today. The real
// config is exercised separately, at the bottom.
const BIG = { key: 'big', name: 'Big', vramFullMb: 20000, minVramMb: 22000, ctxSize: 262144 };
const MID = { key: 'mid', name: 'Mid', vramFullMb: 8000, minVramMb: 9000, ctxSize: 65536 };
const SMALL = { key: 'small', name: 'Small', vramFullMb: 3000, minVramMb: 3500, ctxSize: 32768 };
const LIST = [BIG, MID, SMALL];

describe('pickModel', () => {
  test('takes the largest model the card can actually hold', () => {
    expect(pickModel(40000, 0, LIST)).toBe(BIG);
    expect(pickModel(10000, 0, LIST)).toBe(MID);
    expect(pickModel(4000, 0, LIST)).toBe(SMALL);
  });

  test('falls back to the last entry when nothing else fits', () => {
    // Below even the small model's floor: it still returns the fallback rather
    // than null, because "no model" is not a state the caller can serve from —
    // the VRAM preflight in vram.js is what refuses to start it.
    expect(pickModel(100, 0, LIST)).toBe(SMALL);
  });

  test('charges the mining reserve against the card, so co-running demotes', () => {
    // 22000 free clears BIG's floor outright...
    expect(pickModel(22000, 0, LIST)).toBe(BIG);
    // ...but not once 4 GB is set aside for the miner, which is the whole point
    // of requiredFreeMb: the reserve is not available to the model.
    expect(pickModel(22000, 4096, LIST)).toBe(MID);
  });

  test('unmeasured VRAM gets the fallback, never a large model', () => {
    // null means "no NVIDIA / no driver", not "unlimited". Handing an unmeasured
    // machine a 17 GB download it may not be able to run is the failure this
    // guards; vram.hasEnoughVram makes the same call for the same reason.
    expect(pickModel(null, 0, LIST)).toBe(SMALL);
    expect(pickModel(undefined, 0, LIST)).toBe(SMALL);
    expect(pickModel(NaN, 0, LIST)).toBe(SMALL);
    expect(pickModel('not a number', 0, LIST)).toBe(SMALL);
  });

  test('the mining reserve defaults to zero when the caller omits it', () => {
    expect(pickModel(40000)).toBeTruthy();       // uses the shipped list
    expect(pickModel(40000, undefined, LIST)).toBe(BIG);
  });

  test('skips holes in the list rather than throwing', () => {
    expect(pickModel(40000, 0, [null, BIG, SMALL])).toBe(BIG);
  });

  test('an empty list yields null', () => {
    expect(pickModel(40000, 0, [])).toBeNull();
    expect(pickModel(null, 0, [])).toBeNull();
  });

  test('the boundary is inclusive — exactly enough is enough', () => {
    const need = requiredFreeMb(MID, 0);
    expect(pickModel(need, 0, [MID, SMALL])).toBe(MID);
    expect(pickModel(need - 1, 0, [MID, SMALL])).toBe(SMALL);
  });
});

// A tier measured at several contexts, with round numbers. 300 MiB of margin
// between its figure and its floor, like the shipped tier's ~570.
const SIZED = {
  key: 'sized', name: 'Sized', vramFullMb: 20000, minVramMb: 20300, ctxSize: 262144,
  ctxLadder: [262144, 131072, 65536, 32768],
  vramByCtxMb: { 262144: 20000, 131072: 14000, 65536: 11000, 32768: 9500 },
  minCtxSize: 65536,
  extraArgs: ['--jinja'],
};

describe('pickModel with per-context figures', () => {
  test('takes the largest context the card can hold', () => {
    expect(pickModel(30000, 0, [SIZED, SMALL])).toBe(SIZED);
    const mid = pickModel(16000, 0, [SIZED, SMALL]);
    expect(mid.ctxSize).toBe(131072);
    expect(mid.vramFullMb).toBe(14000);
    expect(mid.minVramMb).toBe(14300);
    expect(pickModel(12000, 0, [SIZED, SMALL]).ctxSize).toBe(65536);
  });

  test('prefers the big model at a smaller context over the small model', () => {
    // 14,000 + a 3,000 reserve fits 17,500; 20,000 + 3,000 does not.
    expect(pickModel(17500, 3000, [SIZED, SMALL]).key).toBe('sized');
    expect(pickModel(17500, 3000, [SIZED, SMALL]).ctxSize).toBe(131072);
  });

  test('never chooses a context below minCtxSize', () => {
    // 9,500 fits the 32768 rung, but the floor is 65536.
    expect(pickModel(10000, 0, [SIZED, SMALL])).toBe(SMALL);
  });

  test('the chosen model keeps only the ladder rungs at or below its context', () => {
    const m = pickModel(16000, 0, [SIZED, SMALL]);
    expect(m.ctxLadder).toEqual([131072, 65536, 32768]);
    expect(ctxLadder(m)[0]).toBe(131072);
  });

  test('keeps everything else, and leaves the configured tier untouched', () => {
    const m = pickModel(16000, 0, [SIZED, SMALL]);
    expect(m).not.toBe(SIZED);
    expect(m.key).toBe('sized');
    expect(m.extraArgs).toEqual(['--jinja']);
    expect(SIZED.ctxSize).toBe(262144);
    expect(SIZED.vramFullMb).toBe(20000);
    expect(SIZED.ctxLadder).toEqual([262144, 131072, 65536, 32768]);
  });

  test('skips rungs that have no figure', () => {
    const holes = Object.assign({}, SIZED, { vramByCtxMb: { 262144: 20000, 65536: 11000 } });
    expect(pickModel(16000, 0, [holes, SMALL]).ctxSize).toBe(65536);
  });

  test('with no minCtxSize every measured rung may be chosen', () => {
    const any = Object.assign({}, SIZED, { minCtxSize: undefined });
    expect(pickModel(10000, 0, [any, SMALL]).ctxSize).toBe(32768);
  });

  test('with no floor figures the margin is zero', () => {
    const bare = { key: 'bare', ctxLadder: [8192, 4096], vramByCtxMb: { 8192: 900, 4096: 600 } };
    const m = pickModel(700, 0, [bare, SMALL]);
    expect(m.ctxSize).toBe(4096);
    expect(m.vramFullMb).toBe(600);
    expect(m.minVramMb).toBe(600);
  });

  test('a model with only per-context figures is held to them at its top context too', () => {
    // It has no vramFullMb of its own, so returning it unchanged at 8192 would
    // leave it with no floor and hand it to a card with 500 MiB free.
    const bare = { key: 'bare', ctxLadder: [8192, 4096], vramByCtxMb: { 8192: 900, 4096: 600 } };
    expect(pickModel(500, 0, [bare, SMALL])).toBe(SMALL);
    const top = pickModel(1000, 0, [bare, SMALL]);
    expect(top.ctxSize).toBe(8192);
    expect(top.vramFullMb).toBe(900);
  });
});

describe('ctxLadder', () => {
  test('returns the model ladder when it has one', () => {
    expect(ctxLadder({ ctxLadder: [262144, 65536] })).toEqual([262144, 65536]);
  });

  test('a model without a ladder still yields one rung, so callers never branch', () => {
    expect(ctxLadder({ ctxSize: 32768 })).toEqual([32768]);
  });

  test('falls back to the global context size, then to empty', () => {
    expect(ctxLadder({})).toEqual([LLM.ctxSize]);
    expect(ctxLadder(null)).toEqual([LLM.ctxSize]);
  });

  test('drops junk rungs instead of asking llama-server for a zero context', () => {
    expect(ctxLadder({ ctxLadder: [262144, 0, -1, null, 65536], ctxSize: 1 })).toEqual([262144, 65536]);
  });

  test('returns a copy, so a caller cannot mutate the config', () => {
    const model = { ctxLadder: [262144, 65536] };
    ctxLadder(model).push(999);
    expect(model.ctxLadder).toEqual([262144, 65536]);
  });
});

describe('needsMmproj', () => {
  test('true only when both halves of the projector are configured', () => {
    expect(needsMmproj({ mmproj: { file: 'p.gguf', url: 'https://x/p.gguf' } })).toBe(true);
    expect(needsMmproj({ mmproj: { file: 'p.gguf' } })).toBe(false);
    expect(needsMmproj({ mmproj: { url: 'https://x/p.gguf' } })).toBe(false);
    expect(needsMmproj({})).toBe(false);
    expect(needsMmproj(null)).toBe(false);
  });
});

describe('the shipped config', () => {
  test('lists the big model first and the default last', () => {
    const list = allModels();
    expect(list[list.length - 1]).toBe(LLM.model);
    expect(list.length).toBeGreaterThan(1);
  });

  test('a 12 GB card keeps serving the default, a 32 GB card gets the vision model', () => {
    // The claim in the PR title, pinned: the point of the tier is that small
    // cards are NOT disturbed by it.
    const small = pickModel(12282 - 6795, LLM.miningReserveMb);   // the fleet's 4070, mining
    expect(small).toBe(LLM.model);

    const big = pickModel(32607, 0);                              // an idle 5090
    expect(big.key).toBe('qwen3.8-27b');
    expect(big.vision).toBe(true);
    expect(big.ctxSize).toBe(262144);
  });

  test('every tier carries what llama-server needs to start', () => {
    for (const m of allModels()) {
      expect(typeof m.name).toBe('string');
      expect(m.file).toMatch(/\.gguf$/);
      expect(m.url).toMatch(/^https:\/\//);
      expect(Number(m.vramFullMb)).toBeGreaterThan(0);
      expect(Number(m.layers)).toBeGreaterThan(0);
    }
  });

  test('a vision tier ships a projector, and the default does not pretend to', () => {
    const vision = allModels().filter((m) => m.vision);
    expect(vision.length).toBeGreaterThan(0);
    for (const m of vision) expect(needsMmproj(m)).toBe(true);
    expect(needsMmproj(LLM.model)).toBe(false);
  });

  test('the big model carries the MEASURED VRAM figure, not an estimate under it', () => {
    // 30,150 MiB measured on a 5090 at 262144 with a q8_0 KV cache. The value
    // here was once an estimate of 28,672 — 1,478 MiB UNDER the truth, which is
    // the direction that OOMs a node rather than merely idling it. Pinned so a
    // future edit cannot drift back below the measurement.
    const q = allModels().find((m) => m.key === 'qwen3.8-27b');
    expect(q.vramFullMb).toBeGreaterThanOrEqual(30150);
    expect(q.minVramMb).toBeGreaterThan(q.vramFullMb);
  });

  test('a 5090 cannot mine and serve this at 262144, so mining it is offered 131072', () => {
    // Measured on the card: llama at full settings holds 30,150 MiB, leaving
    // 2,457 free, while the miner needs 2,081 for its rank-128 profile plus the
    // ~500 MiB every CUDA process costs for its context — ~2,581, so it is short
    // by ~124 MiB. Verified empirically on the box, where the miner refuses with
    // "not enough free VRAM for the rank-128 profile".
    //
    // Budget against what CUDA sees (32,149) rather than what the card reports
    // (32,607): the driver reserves the difference.
    const TOTAL = 32149;
    const MINER = 2581;
    const q = allModels().find((m) => m.key === 'qwen3.8-27b');

    // Mining: free VRAM is what the miner has left, and that is below the
    // model's own resident size at 262144 before any reserve is even considered.
    // It used to fall back to Gemma here. It now gets the same model at 131072,
    // measured at 24,518 MiB, which fits beside the miner with room to spare.
    const whileMining = TOTAL - MINER;
    expect(whileMining).toBeLessThan(q.vramFullMb);
    const mining = pickModel(whileMining, LLM.miningReserveMb);
    expect(mining.key).toBe('qwen3.8-27b');
    expect(mining.ctxSize).toBe(131072);
    expect(mining.vramFullMb).toBe(24518);
    expect(whileMining).toBeGreaterThanOrEqual(requiredFreeMb(mining, LLM.miningReserveMb));

    // Idle: the card clears the full floor and gets the full context.
    expect(pickModel(TOTAL, 0)).toBe(q);
  });

  test('a 32 GB card that is mining gets the big model at 131072, whatever runs first', () => {
    const q = allModels().find((m) => m.key === 'qwen3.8-27b');
    // The CLI plans before the miner starts: an RTX PRO 4500 or 5090 with a
    // desktop on it, ~1 GB used.
    const before = pickModel(32607 - 1000, LLM.miningReserveMb);
    expect(before.key).toBe(q.key);
    expect(before.ctxSize).toBe(131072);
    // The app plans after the miner is running, so its ~2.6 GB is already gone
    // AND the reserve is charged again. It still fits.
    const after = pickModel(32607 - 1000 - 2600, LLM.miningReserveMb);
    expect(after.key).toBe(q.key);
    expect(after.ctxSize).toBe(131072);
  });

  test('a 24 GB card keeps serving the default, mining or not', () => {
    // The 32768 and 65536 rungs would fit a 4090 by a few hundred MiB. minCtxSize
    // keeps them as start-up fallbacks only, so a 24 GB card is not handed a
    // 27B model it can barely hold.
    expect(pickModel(24564 - 300, 0)).toBe(LLM.model);
    expect(pickModel(24564 - 300, LLM.miningReserveMb)).toBe(LLM.model);
  });

  test('the per-context figures are the measured sweep, and the top one matches vramFullMb', () => {
    const q = allModels().find((m) => m.key === 'qwen3.8-27b');
    expect(q.vramByCtxMb).toEqual({ 262144: 30150, 131072: 24518, 65536: 21702, 32768: 20324 });
    expect(q.vramByCtxMb[ctxLadder(q)[0]]).toBe(q.vramFullMb);
    for (const ctx of ctxLadder(q)) expect(q.vramByCtxMb[ctx]).toBeGreaterThan(0);
    expect(ctxLadder(q)).toContain(q.minCtxSize);
  });

  test('the flags that make 256K fit are actually configured', () => {
    // Without a quantised KV cache the model does not load at this context. If
    // someone drops these flags, the measured VRAM figure above is void — so the
    // two are pinned together.
    const q = allModels().find((m) => m.key === 'qwen3.8-27b');
    const a = q.extraArgs.join(' ');
    expect(a).toContain('--cache-type-k q8_0');
    expect(a).toContain('--cache-type-v q8_0');
    expect(a).toContain('-fa 1');
    expect(a).toContain('--jinja');
    // The MTP head ships inside the GGUF, so self-speculation needs no extra file.
    expect(a).toContain('--spec-type draft-mtp');
    // NOT -n: capping generation is what made AIME unmeasurable at 6400.
    expect(q.extraArgs).not.toContain('-n');
  });

  test('the big model ladder descends and ends somewhere a 24 GB card could live', () => {
    const q = allModels().find((m) => m.key === 'qwen3.8-27b');
    const ladder = ctxLadder(q);
    expect(ladder[0]).toBe(262144);
    for (let i = 1; i < ladder.length; i++) expect(ladder[i]).toBeLessThan(ladder[i - 1]);
    expect(ladder[ladder.length - 1]).toBeLessThanOrEqual(32768);
  });
});

// The fallbacks that only fire on a config shape the shipped one never has.
// They exist so this module cannot throw on a config without tiers — the state
// every release before this one was in — rather than as live behaviour.
describe('config-shape fallbacks', () => {
  function withConfig(LLMStub) {
    let mod;
    jest.isolateModules(() => {
      jest.doMock('../src/shared/config', () => ({ LLM: LLMStub }));
      mod = require('../src/shared/models');
    });
    return mod;
  }
  afterEach(() => jest.resetModules());

  test('a config with no tiers yields just the default model', () => {
    const model = { name: 'only', vramFullMb: 100 };
    const m = withConfig({ model, ctxSize: 4096 });
    expect(m.allModels()).toEqual([model]);
    expect(m.pickModel(999999, 0)).toBe(model);
  });

  test('a non-array tiers field is ignored rather than spread', () => {
    const model = { name: 'only', vramFullMb: 100 };
    const m = withConfig({ model, tiers: 'not an array', ctxSize: 4096 });
    expect(m.allModels()).toEqual([model]);
  });

  test('ctxLadder yields nothing when neither the model nor the config has a size', () => {
    const m = withConfig({ model: { name: 'only' } });
    expect(m.ctxLadder({})).toEqual([]);
    expect(m.ctxLadder({ ctxSize: 0 })).toEqual([]);
  });
});

describe('planAutoMode', () => {
  const { planAutoMode } = require('../src/shared/models');
  const big = { key: 'big', minVramMb: 30720, vramFullMb: 30150 };
  const small = { key: 'small', minVramMb: 6000, vramFullMb: 5000 };

  test('a card that can serve a bigger model alone than while mining goes demand-driven', () => {
    const p = planAutoMode(32133, 2048, [big, small]);
    expect(p.strategy).toBe('demand');
    expect(p.model.key).toBe('big');
  });

  test('when both choices agree it co-runs, so small cards are undisturbed', () => {
    const p = planAutoMode(8000, 2048, [big, small]);
    expect(p.strategy).toBe('corun');
    expect(p.model.key).toBe('small');
  });

  test('unknown VRAM falls back rather than guessing demand', () => {
    const p = planAutoMode(null, 2048, [big, small]);
    expect(p.strategy).toBe('corun');
  });

  test('defaults the mining reserve to zero when not given', () => {
    const p = planAutoMode(32133);
    expect(p.strategy).toBe('corun');   // no reserve means both choices agree
  });

  test('uses the shipped model list when none is passed', () => {
    const p = planAutoMode(32133, 2048);
    expect(['demand', 'corun']).toContain(p.strategy);
    expect(p.model).toBeTruthy();
  });

  test('the same model at a bigger context alone also goes demand-driven', () => {
    // SIZED fits 262144 alone (20,300) but only 131072 beside a 3,000 reserve.
    const p = planAutoMode(22000, 3000, [SIZED, SMALL]);
    expect(p.strategy).toBe('demand');
    expect(p.model).toBe(SIZED);
  });

  test('the same model at the same context co-runs', () => {
    const p = planAutoMode(17500, 3000, [SIZED, SMALL]);
    expect(p.strategy).toBe('corun');
    expect(p.model.ctxSize).toBe(131072);
  });

  test('a 32 GB card in CLI auto stays demand-driven at the full context', () => {
    // As before this change: alone it holds 262144, mining only 131072, so it
    // mines until asked and keeps the 5090 memory-clock lock while it does.
    const p = planAutoMode(32607 - 1000, LLM.miningReserveMb);
    expect(p.strategy).toBe('demand');
    expect(p.model.ctxSize).toBe(262144);
  });

  test('an empty model list yields corun with no model rather than throwing', () => {
    const p = planAutoMode(32133, 2048, []);
    expect(p.strategy).toBe('corun');
    expect(p.model).toBeNull();
  });
});
