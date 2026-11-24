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
  const q = () => allModels().find((m) => m.key === 'qwen3.8-27b');

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
    expect(big.ctxSize).toBe(131072);
  });

  test('every tier carries what llama-server needs to start', () => {
    for (const m of allModels()) {
      expect(typeof m.name).toBe('string');
      expect(m.file).toMatch(/\.gguf$/);
      expect(m.url).toMatch(/^https:\/\//);
      expect(Number(m.vramFullMb)).toBeGreaterThan(0);
    }
  });

  test('a vision tier ships a projector, and the default does not pretend to', () => {
    const vision = allModels().filter((m) => m.vision);
    expect(vision.length).toBeGreaterThan(0);
    for (const m of vision) expect(needsMmproj(m)).toBe(true);
    expect(needsMmproj(LLM.model)).toBe(false);
  });

  test('the big model carries the MEASURED VRAM figure, not an estimate under it', () => {
    // A user's RTX PRO 4500 Blackwell held 29.1 GB (29,798 MiB) at 131072 with
    // the projector and MTP on, on the CUDA build. The old Q4_K_XL figure was
    // once an estimate 1,478 MiB under the truth, the direction that OOMs a node.
    expect(q().vramFullMb).toBeGreaterThanOrEqual(29798);
    expect(q().minVramMb).toBeGreaterThan(q().vramFullMb);
  });

  test('a 32 GB card that mines serves Q4_K_XL, measured before or after the miner starts', () => {
    // The CLI plans before the miner starts; the app plans after, with the
    // miner's ~2.6 GB already gone. An RTX PRO 4500 or 5090 with ~1 GB of desktop.
    // Q5_K_XL's 29,800 plus the 2,600 reserve is more than the card has.
    const corun = allModels().find((m) => m.key === 'qwen3.8-27b-corun');
    expect(pickModel(32607 - 1000, LLM.miningReserveMb)).toBe(corun);
    expect(pickModel(32607 - 1000 - 2600, LLM.miningReserveMb)).toBe(corun);
    expect(requiredFreeMb(q(), LLM.miningReserveMb)).toBeGreaterThan(32149);
  });

  test('a 32 GB card with the GPU to itself serves it, with what CUDA sees free', () => {
    // A 5090 reports 32,607 MiB and CUDA sees ~32,149; Windows' desktop takes
    // some of that too.
    expect(pickModel(32149, 0)).toBe(q());
    expect(pickModel(32607 - 1500, 0)).toBe(q());
  });

  test('a 24 GB card gets Qwen3.6-35B-A3B, mining or not', () => {
    const a3b = allModels().find((m) => m.key === 'qwen3.6-35b-a3b');
    expect(pickModel(24564, 0)).toBe(a3b);
    expect(pickModel(24564 - 500, LLM.miningReserveMb)).toBe(a3b);
  });

  test('a 48 GB card serves the big model beside the miner', () => {
    expect(pickModel(49140 - 1000, LLM.miningReserveMb)).toBe(q());
  });

  test('the flags the VRAM figure was measured with are configured', () => {
    // If someone drops the q8_0 cache, the measured figure above is void, so
    // the two are pinned together.
    const a = q().extraArgs.join(' ');
    expect(a).toContain('--cache-type-k q8_0');
    expect(a).toContain('--cache-type-v q8_0');
    expect(a).toContain('-fa 1');
    expect(a).toContain('--jinja');
    // The MTP head ships inside the GGUF, so self-speculation needs no extra file.
    expect(a).toContain('--spec-type draft-mtp');
    // NOT -n: capping generation is what made AIME unmeasurable at 6400.
    expect(q().extraArgs).not.toContain('-n');
  });

  test('the ladder starts at 131072 and descends to 32768', () => {
    const ladder = ctxLadder(q());
    expect(ladder[0]).toBe(131072);
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

  test('a 32 GB card in CLI auto goes demand-driven for the big model', () => {
    const p = planAutoMode(32607 - 1000, LLM.miningReserveMb);
    expect(p.strategy).toBe('demand');
    expect(p.model.key).toBe('qwen3.8-27b');
  });

  test('a 24 GB card in CLI auto co-runs Qwen3.6-35B-A3B', () => {
    const p = planAutoMode(24564 - 300, LLM.miningReserveMb);
    expect(p.strategy).toBe('corun');
    expect(p.model.key).toBe('qwen3.6-35b-a3b');
  });

  test('cards below 24 GB co-run Gemma in CLI auto', () => {
    for (const free of [8188 - 300, 12282 - 300, 16376 - 300, 20475 - 300]) {
      const p = planAutoMode(free, LLM.miningReserveMb);
      expect(p.strategy).toBe('corun');
      expect(p.model).toBe(LLM.model);
    }
  });

  test('a 96 GB card co-runs the big model beside the miner', () => {
    const p = planAutoMode(97887 - 500, LLM.miningReserveMb);
    expect(p.strategy).toBe('corun');
    expect(p.model.key).toBe('qwen3.8-27b');
  });

  test('an empty model list yields corun with no model rather than throwing', () => {
    const p = planAutoMode(32133, 2048, []);
    expect(p.strategy).toBe('corun');
    expect(p.model).toBeNull();
  });
});

// The founder's call of 24 Nov 2026: copy a user's setup for 32 GB cards.
// Qwen3.8-27B at UD-Q5_K_XL, 131072 context, the projector on, MTP on.
describe('the Qwen3.8 Q5_K_XL tier', () => {
  const { buildServerArgs } = require('../src/shared/llama');
  const q5 = () => allModels().find((m) => m.key === 'qwen3.8-27b');

  test('comes first, then the two tiers sized to run beside the miner, then Gemma', () => {
    expect(allModels().map((m) => m.key || 'default'))
      .toEqual(['qwen3.8-27b', 'qwen3.8-27b-corun', 'qwen3.6-35b-a3b', 'default']);
  });

  test('downloads the Unsloth UD-Q5_K_XL file and the F16 projector', () => {
    const m = q5();
    expect(m.name).toBe('Qwen3.8-27B-UD-Q5_K_XL');
    expect(m.file).toBe('Qwen3.8-27B-UD-Q5_K_XL.gguf');
    expect(m.url).toBe('https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-Q5_K_XL.gguf');
    expect(m.sha256).toMatch(/^[0-9a-f]{64}$/);
    expect(m.bytes).toBe(20976962464);
    expect(m.quant).toBe('Q5_K_XL');
    expect(m.mmproj.url).toBe('https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/mmproj-F16.gguf');
    // Its own file, so a card holding the Gemma copy keeps it.
    const files = allModels().map((x) => x.file);
    expect(new Set(files).size).toBe(files.length);
  });

  test('has no mining floor of its own: it is offered wherever it fits', () => {
    expect(q5().minReserveMb).toBeUndefined();
    expect(requiredFreeMb(q5(), 0)).toBe(q5().minVramMb);
  });

  test('starts llama-server with 131072, the projector, q8_0 KV, flash attention and MTP', () => {
    const m = q5();
    const args = buildServerArgs({
      modelPath: '/llm/' + m.file, ctxSize: ctxLadder(m)[0], alias: m.name, extraArgs: m.extraArgs,
      mmprojPath: '/llm/' + m.mmproj.file,
    });
    const s = args.join(' ');
    expect(s).toContain('--ctx-size 131072');
    expect(s).toContain('--n-gpu-layers 999');
    expect(s).toContain('--mmproj /llm/Qwen3.8-27B-mmproj-F16.gguf');
    expect(s).toContain('-fa 1');
    expect(s).toContain('--cache-type-k q8_0');
    expect(s).toContain('--spec-type draft-mtp');
    expect(s).toContain('--alias Qwen3.8-27B-UD-Q5_K_XL');
    // MTP comes from the layer inside this GGUF, so no draft model is passed.
    expect(args).not.toContain('--model-draft');
    expect(args).not.toContain('-md');
  });

  test('keeps a reasoning model\'s completion floor', () => {
    expect(q5().minCompletionTokens).toBe(512);
  });
});

// The founder's call of 24 Nov 2026 for cards that mine. Both VRAM figures are
// estimates until measured beside the miner (config.js says how each was made).
describe('the tiers sized to run beside the miner', () => {
  const { buildServerArgs } = require('../src/shared/llama');
  const corun = () => allModels().find((m) => m.key === 'qwen3.8-27b-corun');
  const a3b = () => allModels().find((m) => m.key === 'qwen3.6-35b-a3b');
  const R = LLM.miningReserveMb;
  // What nvidia-smi reports as each card's total.
  const CARDS = { 8: 8188, 12: 12282, 16: 16376, 20: 20475, 24: 24564, 32: 32607, 96: 97887 };

  test.each([8, 12, 16, 20])('a %i GB card keeps Gemma, mining or not', (gb) => {
    expect(pickModel(CARDS[gb] - 500, R)).toBe(LLM.model);
    expect(pickModel(CARDS[gb], 0)).toBe(LLM.model);
  });

  test('each needs its full size plus the miner reserve free, with or without a miner', () => {
    for (const m of [corun(), a3b()]) {
      expect(m.minReserveMb).toBe(R);
      expect(requiredFreeMb(m, 0)).toBe(m.vramFullMb + R);
      expect(m.minVramMb).toBeGreaterThan(m.vramFullMb);
    }
  });

  test('Q4_K_XL fits a 32 GB card beside the miner and not a 24 GB one', () => {
    expect(requiredFreeMb(corun(), R)).toBeLessThanOrEqual(32149 - 1000);
    expect(requiredFreeMb(corun(), R)).toBeGreaterThan(CARDS[24]);
    // At or above the 24,518 MiB the old sweep measured at 131072.
    expect(corun().vramFullMb).toBeGreaterThanOrEqual(24518);
  });

  test('the 35B fits a 24 GB card beside the miner with a desktop, and not a 20 GB one', () => {
    expect(requiredFreeMb(a3b(), R)).toBeLessThanOrEqual(CARDS[24] - 1000);
    expect(requiredFreeMb(a3b(), R)).toBeGreaterThan(CARDS[20]);
  });

  test('Q4_K_XL downloads the Unsloth file and shares the Q5_K_XL tier\'s projector', () => {
    const m = corun();
    expect(m.name).toBe('Qwen3.8-27B-UD-Q4_K_XL');
    expect(m.url).toBe('https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-Q4_K_XL.gguf');
    expect(m.bytes).toBe(17659407264);
    expect(m.sha256).toMatch(/^[0-9a-f]{64}$/);
    expect(m.mmproj).toEqual(allModels()[0].mmproj);
    expect(m.extraArgs).toEqual(allModels()[0].extraArgs);
    expect(ctxLadder(m)[0]).toBe(131072);
  });

  test('the 35B comes from the MTP repo, with its own projector', () => {
    const m = a3b();
    expect(m.name).toBe('Qwen3.6-35B-A3B-UD-IQ4_XS');
    expect(m.url).toBe('https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf');
    expect(m.bytes).toBe(18209036576);
    expect(m.sha256).toMatch(/^[0-9a-f]{64}$/);
    expect(m.mmproj.url).toBe('https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/resolve/main/mmproj-F16.gguf');
    expect(m.mmproj.file).not.toBe(corun().mmproj.file);
    expect(m.minCompletionTokens).toBe(512);
  });

  test('the 35B starts with 64K, the projector, q8_0 KV and MTP', () => {
    const m = a3b();
    const s = buildServerArgs({
      modelPath: '/llm/' + m.file, ctxSize: ctxLadder(m)[0], alias: m.name, extraArgs: m.extraArgs,
      mmprojPath: '/llm/' + m.mmproj.file,
    }).join(' ');
    expect(s).toContain('--ctx-size 65536');
    expect(s).toContain('--mmproj /llm/Qwen3.6-35B-A3B-mmproj-F16.gguf');
    expect(s).toContain('--cache-type-k q8_0');
    expect(s).toContain('--spec-type draft-mtp');
  });
});
