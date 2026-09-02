'use strict';

const { allModels, pickModel, ctxLadder, needsMmproj, planAutoMode } = require('../src/shared/models');
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

  test('a 5090 cannot mine and serve this at 262144, and is not offered it', () => {
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
    // model's own resident size before any reserve is even considered.
    const whileMining = TOTAL - MINER;
    expect(whileMining).toBeLessThan(q.vramFullMb);
    // So 262144 is not on offer while mining -- which is what this test is about.
    // It does NOT follow that the tier is off the table: the card can host it at
    // 131072 alongside the miner, and refusing it outright was the bug that made
    // ctxLadder unreachable. The rung served is the highest that fits.
    const mining = pickModel(whileMining, LLM.miningReserveMb);
    expect(mining.key).toBe('qwen3.8-27b');
    expect(mining.ctxSize).toBe(131072);
    expect(mining.ctxSize).toBeLessThan(262144);

    // Idle: the card clears the floor and the tier is offered.
    expect(pickModel(TOTAL, 0).key).toBe('qwen3.8-27b');
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
    expect(p.coRunModel.key).toBe('small');
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

  test('an empty model list yields corun with no model rather than throwing', () => {
    const p = planAutoMode(32133, 2048, []);
    expect(p.strategy).toBe('corun');
    expect(p.model).toBeNull();
  });
});

describe('a card that cannot host the top rung is offered a lower one', () => {
  const { LLM } = require('../src/shared/config');
  const QWEN = LLM.tiers[0];

  test('a 24 GB card gets the tier at 65536 instead of being refused it', () => {
    // The bug: the gate priced the tier at its TOP rung, so a 24 GB card was
    // refused a model it can host at 65536 — and ctxLadder was dead code on
    // exactly the cards it was written for.
    const m = pickModel(24564, 0);
    expect(m.key).toBe(QWEN.key);
    expect(m.ctxSize).toBe(65536);
    expect(m.ctxLadder).toEqual([65536, 32768]);   // rungs above it are gone
  });

  test('the pinned copy carries corrected VRAM figures, not just a smaller ctx', () => {
    // llmPlan, computeGpuLayers and the refusal message all re-derive the
    // requirement from these two fields, so they must describe the rung served.
    const m = pickModel(24564, 0);
    expect(m.vramFullMb).toBeLessThan(QWEN.vramFullMb);
    expect(m.minVramMb - m.vramFullMb).toBe(QWEN.minVramMb - QWEN.vramFullMb);
    expect(requiredFreeMb(m, 0)).toBeLessThanOrEqual(24564);
  });

  test('a card that fits the top rung gets the config object itself, unpinned', () => {
    expect(pickModel(32607, 0)).toBe(QWEN);
  });

  test('a card too small for even the lowest rung still falls back', () => {
    expect(pickModel(8192, LLM.miningReserveMb)).toBe(LLM.model);
  });

  test('a tier with no per-token price is only ever offered at full context', () => {
    // Without mbPerCtxToken a rung cannot be priced, so the model keeps its
    // measured cost rather than being offered at a size we cannot cost.
    const unpriced = { key: 'x', vramFullMb: 30000, minVramMb: 30000, ctxSize: 262144,
      ctxLadder: [262144, 32768] };
    const small = { key: 's', vramFullMb: 1000, minVramMb: 1000 };
    expect(pickModel(20000, 0, [unpriced, small])).toBe(small);
  });

  test('auto stays demand-driven when the card can serve a BIGGER window alone', () => {
    // Both choices are now the same model, so comparing keys alone would say
    // "co-run" and give up half the context.
    const p = planAutoMode(32607, LLM.miningReserveMb);
    expect(p.strategy).toBe('demand');
    expect(p.model.ctxSize).toBe(262144);
    expect(p.coRunModel.ctxSize).toBe(131072);
  });

  test('auto co-runs when the window is the same either way', () => {
    const p = planAutoMode(24564, LLM.miningReserveMb);
    expect(p.strategy).toBe('corun');
    expect(p.model.ctxSize).toBe(p.coRunModel.ctxSize);
  });
});

describe('minOfferCtx separates admission from the startup ladder', () => {
  const { LLM } = require('../src/shared/config');

  test('a card below the offer floor keeps the small default', () => {
    // 22 GB can hold the tier at 32768, but a 27B model at a small window is not
    // obviously better for the fleet than the default at its full one — so the
    // floor keeps it on the default rather than the gate deciding silently.
    expect(pickModel(22000, 0)).toBe(LLM.model);
  });

  test('the rung below the floor is still available as a startup fallback', () => {
    // Admission and fallback are different questions: a node offered the tier can
    // still walk down to 32768 if it cannot start higher.
    expect(ctxLadder(LLM.tiers[0])).toContain(32768);
  });

  test('a tier without a floor is offered at any rung it fits', () => {
    const t = { key: 'x', vramFullMb: 30000, minVramMb: 30000, ctxSize: 262144,
      mbPerCtxToken: 0.042659, ctxLadder: [262144, 32768] };
    const small = { key: 's', vramFullMb: 1000, minVramMb: 1000 };
    const m = pickModel(21000, 0, [t, small]);
    expect(m.key).toBe('x');
    expect(m.ctxSize).toBe(32768);
  });
});

describe('rung pricing is defensive about missing measurements', () => {
  const SMALL = { key: 's', vramFullMb: 1000, minVramMb: 1000 };

  test('a tier with no measured cost is never priced below its top rung', () => {
    // Subtracting a rung's worth of KV cache from an absent measurement would
    // produce a negative requirement, which fits any card and OOMs it.
    const noCost = { key: 'n', minVramMb: 30000, ctxSize: 262144,
      mbPerCtxToken: 0.042659, ctxLadder: [262144, 65536] };
    expect(pickModel(25000, 0, [noCost, SMALL])).toBe(SMALL);
  });

  test('a tier with no floor figure still resolves a rung', () => {
    // minVramMb absent: headroom is just zero, not a crash.
    const noFloor = { key: 'f', vramFullMb: 30150, ctxSize: 262144,
      mbPerCtxToken: 0.042659, ctxLadder: [262144, 65536] };
    const m = pickModel(22000, 0, [noFloor, SMALL]);
    expect(m.key).toBe('f');
    expect(m.ctxSize).toBe(65536);
  });
});

test('a tier with no top rung recorded cannot be priced below it', () => {
  // ctxSize is what vramFullMb was measured AT. Without it there is no baseline
  // to subtract a rung's cache from, so the model keeps its full cost.
  const noTop = { key: 'u', vramFullMb: 30150, minVramMb: 30720,
    mbPerCtxToken: 0.042659, ctxLadder: [262144, 65536] };
  const SMALL = { key: 's', vramFullMb: 1000, minVramMb: 1000 };
  expect(pickModel(22000, 0, [noTop, SMALL])).toBe(SMALL);
});
