'use strict';

const {
  REGIONS, DEFAULTS, MINER, ECON,
  regionFor, endpointFor, normalizeEndpoint, resolveEndpoint, splitEndpoint, regionLabel,
  migrateRegion, LEGACY_REGIONS,
} = require('../src/shared/config');

describe('config', () => {
  test('regionFor returns the matching region', () => {
    expect(regionFor('de')).toBe(REGIONS.de);
  });

  test('regionFor falls back to the default region for unknown input', () => {
    expect(regionFor('nope')).toBe(REGIONS[DEFAULTS.region]);
  });

  test('endpointFor returns the region endpoint and falls back', () => {
    expect(endpointFor('sg')).toBe('sg.pearl.herominers.com:1200');
    expect(endpointFor('???')).toBe(REGIONS.us.endpoint);
  });

  // AlphaPool went with alpha-miner: it gates its stratum behind a GPU-solved
  // challenge and then dictates the mining geometry. HeroMiners sends a header
  // and a target and lets the miner pick its own, which is what the protocol
  // allows. Every region here was checked to resolve and accept a connection.
  test('every region is a HeroMiners Pearl endpoint on port 1200', () => {
    const endpoints = Object.values(REGIONS).map((r) => r.endpoint);
    expect(endpoints).toHaveLength(12);
    expect(endpoints.every((e) => e.endsWith('.pearl.herominers.com:1200'))).toBe(true);
    expect(endpoints).toContain('hk.pearl.herominers.com:1200');
    expect(endpoints.some((e) => /alphapool/.test(e))).toBe(false);
  });

  test('regionLabel combines flag and label, with fallback', () => {
    expect(regionLabel('us2')).toBe('🇺🇸 us2');
    expect(regionLabel('xx')).toBe('🇺🇸 us');
  });

  test('engine and economics metadata are present', () => {
    expect(MINER).toMatchObject({ engine: 'llmjob-pearl', pool: 'HeroMiners', pow: 'pearlhash', devFeePct: 0, poolFeePct: 0 });
    expect(ECON).toMatchObject({ NET_TH: 61e6, DAILY_NET_PRL: 1.62e6, FEE: 0.99, PRL_USD: 0.30 });
  });
});

// A hand-written endpoint override reaches the engine's --host directly, and
// alpha-miner 1.9.4 wants a bare host:port. Every older doc — and our own
// pre-1.9.4 vector — wrote it as `stratum+tcp://host:port`, so that form gets
// pasted in; passing it through made the engine resolve the scheme as part of
// the hostname and loop on "DNS lookup failed: No such host is known".
describe('normalizeEndpoint / resolveEndpoint', () => {
  test('strips a scheme, trailing slash and surrounding space', () => {
    expect(normalizeEndpoint('stratum+tcp://us.pearl.herominers.com:1200')).toBe('us.pearl.herominers.com:1200');
    expect(normalizeEndpoint('  us.pearl.herominers.com:1200  ')).toBe('us.pearl.herominers.com:1200');
    expect(normalizeEndpoint('tcp://de.pearl.herominers.com:1200')).toBe('de.pearl.herominers.com:1200');
    expect(normalizeEndpoint('us.pearl.herominers.com:1200/')).toBe('us.pearl.herominers.com:1200');
  });

  test('nothing usable becomes null, so the caller can fall back', () => {
    for (const v of ['', '   ', null, undefined]) expect(normalizeEndpoint(v)).toBeNull();
  });

  test('resolveEndpoint prefers a cleaned override, else the region', () => {
    expect(resolveEndpoint({ endpoint: 'stratum+tcp://us.pearl.herominers.com:1200', region: 'de' }))
      .toBe('us.pearl.herominers.com:1200');
    // A blank override must not win — that would point the miner at nothing.
    expect(resolveEndpoint({ endpoint: '   ', region: 'de' })).toBe(REGIONS.de.endpoint);
    expect(resolveEndpoint({ region: 'de' })).toBe(REGIONS.de.endpoint);
    expect(resolveEndpoint({})).toBe(REGIONS[DEFAULTS.region].endpoint);
    expect(resolveEndpoint()).toBe(REGIONS[DEFAULTS.region].endpoint);
  });
  // The split form is the whole point: a build that reads --host as host-only
  // and appends its own default port turned a combined `host:5566` into
  // `us2.pearl.herominers.com:1200:5566` and then failed DNS on it. Two field reports.
  test('splits host from port so a port can never be doubled', () => {
    expect(splitEndpoint('us2.pearl.herominers.com:1200')).toEqual({ host: 'us2.pearl.herominers.com', port: 1200 });
    expect(splitEndpoint('stratum+tcp://us.pearl.herominers.com:1200'))
      .toEqual({ host: 'us.pearl.herominers.com', port: 1200 });
  });

  // No port means no port — the engine keeps its own default rather than us
  // inventing one, and nothing usable stays null so callers can fall back.
  test('leaves a portless host alone and nulls an empty endpoint', () => {
    expect(splitEndpoint('us2.pearl.herominers.com')).toEqual({ host: 'us2.pearl.herominers.com', port: null });
    for (const v of ['', '   ', null, undefined]) {
      expect(splitEndpoint(v)).toEqual({ host: null, port: null });
    }
  });
});

// ── legacy region migration ─────────────────────────────────────────────────

describe('migrateRegion', () => {
  // Every 0.3.x install has an AlphaPool region in its settings file and none of
  // them exist now. Handing one to the Settings <select> leaves it BLANK rather
  // than erroring, and the renderer's own fallback then rewrote the choice —
  // so an upgrading rig changed continent without being told.
  test('every legacy id maps to a region that exists', () => {
    for (const [old, now] of Object.entries(LEGACY_REGIONS)) {
      expect(REGIONS[now]).toBeDefined();
      expect(migrateRegion(old)).toBe(now);
    }
  });

  // Mapped to the NEAREST live endpoint, not all onto the default: someone who
  // chose Singapore should still be mining in Singapore afterwards.
  test('keeps the user near where they chose', () => {
    expect(migrateRegion('sg1')).toBe('sg');
    expect(migrateRegion('hk1')).toBe('hk');
    expect(migrateRegion('in1')).toBe('sg');  // India -> the closest that answers
    expect(migrateRegion('eu1')).toBe('de');
    expect(migrateRegion('us1')).toBe('us');
  });

  // us2 exists on both pools, so it must NOT be translated.
  test('leaves an id that still exists alone', () => {
    for (const id of Object.keys(REGIONS)) expect(migrateRegion(id)).toBe(id);
    expect(migrateRegion('us2')).toBe('us2');
    expect(LEGACY_REGIONS.us2).toBeUndefined();
  });

  test('anything unrecognised becomes the default', () => {
    for (const v of ['junk', '', '   ', null, undefined, 0]) {
      expect(migrateRegion(v)).toBe(DEFAULTS.region);
    }
  });

  // A migrated id has to survive the rest of the pipeline, or the miner still
  // connects somewhere the user did not pick.
  test('a migrated id resolves to a real endpoint', () => {
    for (const old of Object.keys(LEGACY_REGIONS)) {
      const endpoint = resolveEndpoint({ region: migrateRegion(old) });
      expect(endpoint).toMatch(/\.pearl\.herominers\.com:1200$/);
    }
  });
});
