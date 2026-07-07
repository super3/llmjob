'use strict';

const {
  REGIONS, POOLS, DEFAULT_POOL, DEFAULTS, MINER, ECON,
  regionFor, endpointFor, regionLabel, difficultyForCard,
  poolFor, regionsFor, poolEndpointFor,
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

  test('pools: AlphaPool is the default with balances; HeroMiners is vardiff/suffix-style', () => {
    expect(DEFAULT_POOL).toBe('alphapool');
    expect(POOLS.alphapool).toMatchObject({ regions: REGIONS, workerStyle: 'flag', staticDifficulty: true, balances: true });
    expect(POOLS.herominers).toMatchObject({ workerStyle: 'suffix', staticDifficulty: false, balances: false, defaultRegion: 'ww' });
    const hero = Object.values(POOLS.herominers.regions).map((r) => r.endpoint);
    expect(hero).toHaveLength(5);
    expect(hero.every((e) => e.endsWith('pearl.herominers.com:1200'))).toBe(true);
  });

  test('poolFor / regionsFor fall back to the default pool', () => {
    expect(poolFor('herominers')).toBe(POOLS.herominers);
    expect(poolFor('nope')).toBe(POOLS.alphapool);
    expect(regionsFor('herominers')).toBe(POOLS.herominers.regions);
    expect(regionsFor(undefined)).toBe(REGIONS);
  });

  test('poolEndpointFor resolves per pool and falls back to the pool default region', () => {
    expect(poolEndpointFor('alphapool', 'eu1')).toBe('eu1.alphapool.tech:5566');
    expect(poolEndpointFor('herominers', 'de')).toBe('de.pearl.herominers.com:1200');
    expect(poolEndpointFor('herominers', 'us2')).toBe('pearl.herominers.com:1200'); // AlphaPool key → ww fallback
    expect(poolEndpointFor(undefined, undefined)).toBe('us2.alphapool.tech:5566');
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

  test('engine and economics metadata are present', () => {
    expect(MINER).toMatchObject({ engine: 'alpha-miner', pow: 'pearlhash', devFeePct: 0, poolFeePct: 1 });
    expect(ECON).toMatchObject({ NET_TH: 30.79e6, DAILY_NET_PRL: 1.2e6, FEE: 0.99, PRL_USD: 0.47 });
    expect(DEFAULTS.difficulty).toBe(524288);
  });
});
