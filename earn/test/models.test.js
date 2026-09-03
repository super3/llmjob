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
    expect(pickModel(whileMining, LLM.miningReserveMb)).toBe(LLM.model);   // -> Gemma

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

// A tier shaped like the real one but with round numbers, so these assertions
// are about the RULE and not about whatever the shipped sweep says today. The
// shipped config is exercised separately, in the block below this one.
const TIER = {
  key: 'tier', name: 'Tier',
  vramFullMb: 30000, minVramMb: 30500,      // headroom 500
  ctxSize: 262144,
  ctxLadder: [262144, 131072, 65536, 32768],
  ctxVramMb: { 262144: 30000, 131072: 24000, 65536: 21000, 32768: 19000 },
  minOfferCtx: 65536,
};
const TINY = { key: 'tiny', name: 'Tiny', vramFullMb: 3000, minVramMb: 3500, ctxSize: 32768 };

describe('vramAtCtx', () => {
  const { vramAtCtx } = require('../src/shared/models');

  test('reads the measured figure for a priced rung', () => {
    expect(vramAtCtx(TIER, 131072)).toBe(24000);
    expect(vramAtCtx(TIER, 65536)).toBe(21000);
  });

  test('an UNPRICED rung costs the full model, which refuses it', () => {
    // The safe direction, and the reason this returns `full` rather than
    // interpolating: a guessed rung that comes in low OOMs a node, while one
    // that refuses merely leaves the node on the default.
    expect(vramAtCtx(TIER, 16384)).toBe(30000);
    expect(vramAtCtx(Object.assign({}, TIER, { ctxVramMb: { 262144: 30000 } }), 65536)).toBe(30000);
  });

  test('a junk figure in the table is refused, not trusted', () => {
    const junk = Object.assign({}, TIER, { ctxVramMb: { 65536: 0, 131072: -1, 32768: 'lots' } });
    expect(vramAtCtx(junk, 65536)).toBe(30000);
    expect(vramAtCtx(junk, 131072)).toBe(30000);
    expect(vramAtCtx(junk, 32768)).toBe(30000);
  });

  test('a model with no table cannot be priced below its top rung', () => {
    expect(vramAtCtx(TINY, 4096)).toBe(3000);
  });

  test('a model with no measured cost prices at zero, never negative', () => {
    // Pricing a rung by subtracting cache from an unknown total is how an
    // earlier approach could hand back a NEGATIVE requirement, which fits any
    // card. There is no such arithmetic now, and the guard is asserted anyway.
    expect(vramAtCtx({ ctxVramMb: { 4096: 10 } }, 4096)).toBe(0);
    expect(vramAtCtx(null, 4096)).toBe(0);
    expect(vramAtCtx(TIER, 0)).toBe(30000);
    expect(vramAtCtx(TIER, -1)).toBe(30000);
  });
});

describe('pinToFittingRung', () => {
  const { pinToFittingRung } = require('../src/shared/models');

  test('takes the highest rung that fits and pins the requirement to it', () => {
    const p = pinToFittingRung(TIER, 24500, 0);
    expect(p.ctxSize).toBe(131072);
    expect(p.vramFullMb).toBe(24000);
    expect(p.minVramMb).toBe(24500);          // measured + the model's own headroom
    // The ladder handed to llama-server starts where admission landed, so a
    // start failure walks DOWN from the admitted window rather than retrying one
    // the card was already refused.
    expect(p.ctxLadder).toEqual([131072, 65536, 32768]);
  });

  test('walks down to the next rung when the higher one does not fit', () => {
    expect(pinToFittingRung(TIER, 24499, 0).ctxSize).toBe(65536);
    expect(pinToFittingRung(TIER, 21500, 0).ctxSize).toBe(65536);
  });

  test('stops at minOfferCtx rather than walking the ladder to the bottom', () => {
    // 19,500 would host the 32768 rung, but a 27B at a small window is not
    // obviously a better use of a node than the default at its full one.
    expect(pinToFittingRung(TIER, 21499, 0)).toBeNull();
    expect(pinToFittingRung(TIER, 19500, 0)).toBeNull();
  });

  test('never returns the top rung — the caller has already tested it', () => {
    expect(pinToFittingRung(TIER, 999999, 0).ctxSize).toBe(131072);
  });

  test('charges the mining reserve against the rung too', () => {
    expect(pinToFittingRung(TIER, 24500, 0).ctxSize).toBe(131072);
    expect(pinToFittingRung(TIER, 24500, 2000).ctxSize).toBe(65536);
    // Enough reserve and no rung above the floor survives at all.
    expect(pinToFittingRung(TIER, 24500, 4000)).toBeNull();
  });

  test('a model with no ladder has nothing to walk; no floor walks it all', () => {
    expect(pinToFittingRung(TINY, 999999, 0)).toBeNull();
    const noFloor = Object.assign({}, TIER, { minOfferCtx: undefined });
    expect(pinToFittingRung(noFloor, 19500, 0).ctxSize).toBe(32768);
  });

  test('a missing VRAM field cannot produce a floor BELOW the measured cost', () => {
    // Neither field is optional in the shipped config -- both are asserted in
    // the block above -- but a tier that lost one must not end up with a
    // negative floor, which is a requirement that fits any card at all.
    const noFloor = Object.assign({}, TIER, { minVramMb: undefined });
    const p = pinToFittingRung(noFloor, 24500, 0);
    expect(p.minVramMb).toBe(p.vramFullMb);
    expect(p.minVramMb).toBeGreaterThan(0);

    // And with no measured cost the rung prices at zero, which computeGpuLayers
    // reads as "do not use this card" rather than as a model that fits anywhere.
    const noCost = { ctxLadder: [262144, 131072], ctxVramMb: { 131072: 500 }, minVramMb: 700 };
    const q = pinToFittingRung(noCost, 24500, 0);
    expect(q.vramFullMb).toBe(0);
    expect(q.minVramMb).toBe(700);
  });
  test('leaves the config object untouched', () => {
    const before = JSON.stringify(TIER);
    pinToFittingRung(TIER, 24500, 0);
    expect(JSON.stringify(TIER)).toBe(before);
  });
});

describe('a smaller window is offered only to a card serving ALONE', () => {
  const { planAutoMode } = require('../src/shared/models');
  const LIST = [TIER, TINY];

  test('a card that cannot host full context still gets the tier when it serves alone', () => {
    const m = pickModel(24500, 0, LIST);
    expect(m.key).toBe('tier');
    expect(m.ctxSize).toBe(131072);
  });

  test('the SAME card co-running is offered the tier only at full context', () => {
    // This is the distinction the first attempt at this lacked, and it is not a
    // policy nicety: any floor low enough to admit a 24 GB card at 65536 also
    // admits a MINING 32 GB card at 131072, which moved the whole 32 GB fleet
    // off the small default without anyone asking for it.
    expect(pickModel(24500, 2048, LIST)).toBe(TINY);
    expect(pickModel(24500, 1, LIST)).toBe(TINY);
  });

  test('a card that clears full context is unaffected either way', () => {
    // Object identity, not merely an equal shape: a card that was always
    // eligible gets the config object itself back.
    expect(pickModel(30500, 0, LIST)).toBe(TIER);
    expect(pickModel(32548, 2048, LIST)).toBe(TIER);
  });

  test('a card below the lowest offered rung keeps the default', () => {
    expect(pickModel(21499, 0, LIST)).toBe(TINY);
  });

  test('unmeasured VRAM is still never handed a reduced window', () => {
    expect(pickModel(null, 0, LIST)).toBe(TINY);
  });

  test('auto mode reads a pinned tier as worth waking for, not as a co-run', () => {
    const p = planAutoMode(24500, 2048, LIST);
    expect(p.strategy).toBe('demand');
    expect(p.model.key).toBe('tier');
    expect(p.model.ctxSize).toBe(131072);
  });
});

describe('the shipped config, card by card', () => {
  const q = () => allModels().find((m) => m.key === 'qwen3.8-27b');

  // Free VRAM measured on the machines, not taken from the box art. A 4090
  // reports 24,564 MiB and CUDA sees all of it; the Windows desktop on the
  // reference rig holds 124 (Brave 77, explorer 47), so ~24,440 is what
  // llama-server can actually have with the card to itself.
  const C4090 = 24440;
  const C5090 = 32149;              // 32,607 reported, ~458 reserved by the driver
  const MINER = 2581;               // 2,081 for the rank-128 profile + ~500 of CUDA context

  test('a 4090 serving alone gets the tier at 65536, with vision and MTP', () => {
    const m = pickModel(C4090, 0);
    expect(m.key).toBe('qwen3.8-27b');
    expect(m.ctxSize).toBe(65536);
    expect(m.vision).toBe(true);
    expect(needsMmproj(m)).toBe(true);
    expect(m.extraArgs.join(' ')).toContain('--spec-type draft-mtp');
    expect(m.extraArgs.join(' ')).toContain('--cache-type-k q8_0');
    // Priced from the sweep rather than from the top rung: 21,702 at 65536.
    expect(m.vramFullMb).toBe(21702);
    expect(m.minVramMb).toBe(21702 + 570);
    // And it fits with room, rather than on the edge.
    expect(C4090 - requiredFreeMb(m, 0)).toBeGreaterThan(2000);
  });

  test('a 4090 does NOT get it while mining — 24 GB cannot hold both', () => {
    // 21,702 plus the miner's 2,581 is 24,283 against 24,440: it "fits" by
    // 157 MiB, which is not a margin. Admission refuses it because a co-running
    // card is only ever offered the top rung, and that is the intended answer.
    expect(pickModel(C4090 - MINER, LLM.miningReserveMb)).toBe(LLM.model);
    expect(pickModel(C4090, LLM.miningReserveMb)).toBe(LLM.model);
  });

  test('the 5090 rows from the revert still hold, unchanged', () => {
    expect(pickModel(C5090, 0).ctxSize).toBe(262144);                        // idle
    expect(pickModel(C5090 - MINER, LLM.miningReserveMb)).toBe(LLM.model);   // mining
    // The regression the revert existed to undo: a mining 32 GB node must NOT
    // quietly move onto a 27B at half context.
    expect(pickModel(C5090, LLM.miningReserveMb)).toBe(LLM.model);
  });

  test('small cards are untouched', () => {
    expect(pickModel(12282 - 6795, LLM.miningReserveMb)).toBe(LLM.model);    // the fleet's 4070
    expect(pickModel(8192, 0)).toBe(LLM.model);
    expect(pickModel(16384, 0)).toBe(LLM.model);
  });

  test('every rung the tier may be admitted at carries a MEASURED figure', () => {
    // The invariant that keeps vramAtCtx from having to refuse a rung the ladder
    // actually offers: at or above the floor, a rung is priced.
    const t = q();
    for (const rung of ctxLadder(t)) {
      if (rung < t.minOfferCtx) continue;
      expect(Number(t.ctxVramMb[rung])).toBeGreaterThan(0);
    }
  });

  test('the measured table agrees with the curve the config fits', () => {
    // VRAM_MiB ~= 18967 + ctx * 0.042659, from the same sweep. A cross-check,
    // not a source — a rung is measured or it is refused — kept because a typo
    // in the table is otherwise invisible.
    const t = q();
    for (const [ctx, mb] of Object.entries(t.ctxVramMb)) {
      expect(Math.abs((18967 + Number(ctx) * 0.042659) - mb)).toBeLessThan(80);
    }
  });

  test('the top rung is priced at the same figure the tier ships', () => {
    const t = q();
    expect(t.ctxVramMb[t.ctxSize]).toBe(t.vramFullMb);
  });
});
