'use strict';

const {
  REGIONS, DEFAULTS, MINER, ECON,
  regionFor, endpointFor, normalizeEndpoint, resolveEndpoint, splitEndpoint, regionLabel, difficultyForCard,
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

  test('engine and economics metadata are present', () => {
    expect(MINER).toMatchObject({ engine: 'alpha-miner', pow: 'pearlhash', devFeePct: 0, poolFeePct: 1 });
    expect(ECON).toMatchObject({ NET_TH: 61e6, DAILY_NET_PRL: 1.62e6, FEE: 0.99, PRL_USD: 0.30 });
    expect(DEFAULTS.difficulty).toBe(524288);
  });
});

// A hand-written endpoint override reaches the engine's --host directly, and
// alpha-miner 1.9.4 wants a bare host:port. Every older doc — and our own
// pre-1.9.4 vector — wrote it as `stratum+tcp://host:port`, so that form gets
// pasted in; passing it through made the engine resolve the scheme as part of
// the hostname and loop on "DNS lookup failed: No such host is known".
describe('normalizeEndpoint / resolveEndpoint', () => {
  test('strips a scheme, trailing slash and surrounding space', () => {
    expect(normalizeEndpoint('stratum+tcp://us1.alphapool.tech:5566')).toBe('us1.alphapool.tech:5566');
    expect(normalizeEndpoint('  us1.alphapool.tech:5566  ')).toBe('us1.alphapool.tech:5566');
    expect(normalizeEndpoint('tcp://eu1.alphapool.tech:5566')).toBe('eu1.alphapool.tech:5566');
    expect(normalizeEndpoint('us1.alphapool.tech:5566/')).toBe('us1.alphapool.tech:5566');
  });

  test('nothing usable becomes null, so the caller can fall back', () => {
    for (const v of ['', '   ', null, undefined]) expect(normalizeEndpoint(v)).toBeNull();
  });

  test('resolveEndpoint prefers a cleaned override, else the region', () => {
    expect(resolveEndpoint({ endpoint: 'stratum+tcp://us1.alphapool.tech:5566', region: 'eu1' }))
      .toBe('us1.alphapool.tech:5566');
    // A blank override must not win — that would point the miner at nothing.
    expect(resolveEndpoint({ endpoint: '   ', region: 'eu1' })).toBe(REGIONS.eu1.endpoint);
    expect(resolveEndpoint({ region: 'eu1' })).toBe(REGIONS.eu1.endpoint);
    expect(resolveEndpoint({})).toBe(REGIONS[DEFAULTS.region].endpoint);
    expect(resolveEndpoint()).toBe(REGIONS[DEFAULTS.region].endpoint);
  });
  // The split form is the whole point: a build that reads --host as host-only
  // and appends its own default port turned a combined `host:5566` into
  // `us2.alphapool.tech:5566:5566` and then failed DNS on it. Two field reports.
  test('splits host from port so a port can never be doubled', () => {
    expect(splitEndpoint('us2.alphapool.tech:5566')).toEqual({ host: 'us2.alphapool.tech', port: 5566 });
    expect(splitEndpoint('stratum+tcp://us1.alphapool.tech:5566'))
      .toEqual({ host: 'us1.alphapool.tech', port: 5566 });
  });

  // No port means no port — the engine keeps its own default rather than us
  // inventing one, and nothing usable stays null so callers can fall back.
  test('leaves a portless host alone and nulls an empty endpoint', () => {
    expect(splitEndpoint('us2.alphapool.tech')).toEqual({ host: 'us2.alphapool.tech', port: null });
    for (const v of ['', '   ', null, undefined]) {
      expect(splitEndpoint(v)).toEqual({ host: null, port: null });
    }
  });
});
