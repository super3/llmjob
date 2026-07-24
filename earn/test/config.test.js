'use strict';

const {
  REGIONS, DEFAULTS, MINER, ECON, LLM,
  regionFor, endpointFor, regionLabel, difficultyForCard,
} = require('../src/shared/config');

describe('config', () => {
  test('regionFor returns the matching region', () => {
    expect(regionFor('eu1')).toBe(REGIONS.eu1);
  });

  test('regionFor falls back to the default region for unknown input', () => {
    expect(regionFor('nope')).toBe(REGIONS[DEFAULTS.region]);
  });

  test('endpointFor returns the region endpoint and falls back', () => {
    expect(endpointFor('sg1')).toBe('sg1.alphapool.tech:5566');
    expect(endpointFor('???')).toBe(REGIONS.us2.endpoint);
  });

  test('all eight documented endpoints are present on port 5566', () => {
    const endpoints = Object.values(REGIONS).map((r) => r.endpoint);
    expect(endpoints).toHaveLength(8);
    expect(endpoints.every((e) => e.endsWith('.alphapool.tech:5566'))).toBe(true);
    expect(endpoints).toContain('hk1.alphapool.tech:5566');
  });

  test('regionLabel combines flag and label, with fallback', () => {
    expect(regionLabel('us2')).toBe('🇺🇸 us2');
    expect(regionLabel('xx')).toBe('🇺🇸 us2');
  });

  test('difficultyForCard maps card classes and falls back to the default', () => {
    expect(difficultyForCard('GPU #0 · RTX 5090')).toBe(1048576);
    expect(difficultyForCard('NVIDIA RTX PRO 6000 Blackwell Workstation Edition')).toBe(1048576);
    expect(difficultyForCard('RTX 4090')).toBe(524288);
    expect(difficultyForCard('RTX 4070')).toBe(262144);
    expect(difficultyForCard('RTX 3090')).toBe(262144);
    expect(difficultyForCard('RTX 3070')).toBe(131072);
    expect(difficultyForCard('A100')).toBe(131072);
    expect(difficultyForCard('RTX 2080')).toBe(16384);
    expect(difficultyForCard('V100')).toBe(4096);
    expect(difficultyForCard('something else')).toBe(DEFAULTS.difficulty);
    expect(difficultyForCard(null)).toBe(DEFAULTS.difficulty);
    // `pro 6000` must not catch the older, slower 6000-class cards.
    expect(difficultyForCard('NVIDIA RTX A6000')).toBe(DEFAULTS.difficulty);
    expect(difficultyForCard('Quadro RTX 6000')).toBe(DEFAULTS.difficulty);
  });

  test('LLM.models is a VRAM-ordered catalog incl. the two Qwen3.6 gateway models', () => {
    const ids = LLM.models.map((m) => m.id);
    expect(ids).toEqual(['gemma-4-e4b', 'qwen3.6-27b', 'qwen3.6-35b-a3b']);
    // ordered ascending by minVramMb so the picker's "largest that fits" is stable
    const floors = LLM.models.map((m) => m.minVramMb);
    expect(floors).toEqual([...floors].sort((a, b) => a - b));
    // exactly one default fallback (used when VRAM can't be measured)
    expect(LLM.models.filter((m) => m.default)).toHaveLength(1);
    expect(LLM.models[0]).toMatchObject({ default: true, name: LLM.model.name });
    // every model carries the fields placement + serving read
    for (const m of LLM.models) {
      expect(m).toMatchObject({
        id: expect.any(String), name: expect.any(String), file: expect.any(String),
        url: expect.stringContaining('http'), layers: expect.any(Number),
        minVramMb: expect.any(Number), quant: 'Q4_K_M',
        ctxSize: expect.any(Number), parallel: expect.any(Number),
      });
    }
    // the MoE row is the 35B A3B, batched a little (cheap ~3B-active decode)
    const moe = LLM.models.find((m) => m.id === 'qwen3.6-35b-a3b');
    expect(moe).toMatchObject({ kind: 'moe', parallel: 2 });
    expect(LLM.model).toMatchObject({ ctxSize: 4096, parallel: 1 });
  });

  test('engine and economics metadata are present', () => {
    expect(MINER).toMatchObject({ engine: 'alpha-miner', pow: 'pearlhash', devFeePct: 0, poolFeePct: 1 });
    expect(ECON).toMatchObject({ NET_TH: 61e6, DAILY_NET_PRL: 1.62e6, FEE: 0.99, PRL_USD: 0.30 });
    expect(DEFAULTS.difficulty).toBe(524288);
  });
});
