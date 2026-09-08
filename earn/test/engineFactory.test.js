'use strict';

// The one place that decides which engine a rig runs. Both entry points go
// through it, so the three-way outcome is pinned here rather than twice over in
// the CLI and main-process suites.

jest.mock('../src/main/pearlEngine', () => {
  class PearlEngine {
    constructor(opts) { this.opts = opts; PearlEngine.instances.push(this); }
  }
  PearlEngine.instances = [];
  return { PearlEngine };
});
jest.mock('../src/main/peakEngine', () => {
  class PeakEngine {
    constructor(opts) { this.opts = opts; PeakEngine.instances.push(this); }
  }
  PeakEngine.instances = [];
  return { PeakEngine, httpSummaryFetcher: jest.fn(() => 'FETCHER') };
});

const { PearlEngine } = require('../src/main/pearlEngine');
const { PeakEngine, httpSummaryFetcher } = require('../src/main/peakEngine');
const { createMinerEngine } = require('../src/main/engineFactory');

const CORE = () => ({});
const connect = jest.fn();
const spawn = jest.fn();
const readTemps = jest.fn();

beforeEach(() => {
  PearlEngine.instances.length = 0;
  PeakEngine.instances.length = 0;
  httpSummaryFetcher.mockClear();
});

function build(over = {}) {
  return createMinerEngine(Object.assign({
    settings: {}, createCore: CORE, env: {}, exists: () => false,
    connect, spawn, readTemps,
  }, over));
}

describe('PeakMiner selected', () => {
  test('builds a PeakEngine on the resolved binary', () => {
    const r = build({
      env: { PATH: '/usr/bin' }, exists: (p) => p === '/usr/bin/peakminer',
    });
    expect(r.miner).toBeInstanceOf(PeakEngine);
    expect(r.miner.opts.binPath).toBe('/usr/bin/peakminer');
    expect(r.miner.opts.spawn).toBe(spawn);
    expect(r.miner.opts.fetchSummary).toBe('FETCHER');
    expect(r.coreMissing).toBe(false);
    expect(PearlEngine.instances).toHaveLength(0);
  });

  test('an explicit --miner-bin is honoured', () => {
    const r = build({
      settings: { minerBin: '/opt/pm' }, exists: (p) => p === '/opt/pm',
    });
    expect(r.miner.opts.binPath).toBe('/opt/pm');
  });

  // "Why is this rig suddenly faster/slower" should be answerable from the log
  // alone, so the reason is always emitted, never only on the unusual paths.
  test('says why it chose PeakMiner', () => {
    const r = build({ env: { PATH: '/usr/bin' }, exists: () => true });
    expect(r.notes[0]).toMatch(/PeakMiner/);
  });

  test('PeakMiner wins over an available native core', () => {
    const r = build({ createCore: CORE, exists: () => true, env: { PATH: '/usr/bin' } });
    expect(r.miner).toBeInstanceOf(PeakEngine);
  });
});

describe('PeakMiner requested but absent', () => {
  test('is a refusal with no engine, whatever the core situation', () => {
    for (const createCore of [CORE, null]) {
      PeakEngine.instances.length = 0;
      PearlEngine.instances.length = 0;
      const r = build({ settings: { miner: 'peak' }, createCore, exists: () => false });
      expect(r.miner).toBe(null);
      expect(r.coreMissing).toBe(false);
      expect(r.notes[0]).toMatch(/no binary was found/);
      expect(PearlEngine.instances).toHaveLength(0);
      expect(PeakEngine.instances).toHaveLength(0);
    }
  });
});

describe('native selected', () => {
  test('builds a PearlEngine with the injected core and connect', () => {
    const r = build();
    expect(r.miner).toBeInstanceOf(PearlEngine);
    expect(r.miner.opts.createCore).toBe(CORE);
    expect(r.miner.opts.connect).toBe(connect);
    expect(r.miner.opts.readTemps).toBe(readTemps);
    expect(r.coreMissing).toBe(false);
  });

  test('--miner native ignores an installed PeakMiner', () => {
    const r = build({
      settings: { miner: 'native' }, env: { PATH: '/usr/bin' }, exists: () => true,
    });
    expect(r.miner).toBeInstanceOf(PearlEngine);
    expect(r.notes[0]).toMatch(/forced/);
  });
});

// The two callers have always differed here and both behaviours are load-bearing:
// the GUI wants the engine to exist so it can announce the problem in the miner
// log, the CLI wants to refuse up front so systemd sees a non-zero exit instead
// of a silent restart loop.
describe('missing native core', () => {
  test('requireCore=false still constructs, so the engine can announce it', () => {
    const r = build({ createCore: null, requireCore: false });
    expect(r.miner).toBeInstanceOf(PearlEngine);
    expect(r.coreMissing).toBe(true);
  });

  test('requireCore=true builds nothing and reports it', () => {
    const r = build({ createCore: null, requireCore: true });
    expect(r.miner).toBe(null);
    expect(r.coreMissing).toBe(true);
    expect(PearlEngine.instances).toHaveLength(0);
  });

  test('requireCore defaults to the GUI behaviour', () => {
    expect(build({ createCore: null }).miner).toBeInstanceOf(PearlEngine);
  });
});

describe('defaults', () => {
  test('called with nothing at all resolves to the native engine', () => {
    const r = createMinerEngine();
    expect(r.miner).toBeInstanceOf(PearlEngine);
    expect(r.notes).toHaveLength(1);
  });

  test('an absent PATH is not a crash', () => {
    expect(() => build({ env: {} })).not.toThrow();
  });

  // With no existence check injected, nothing is found -- so a caller that
  // forgets to pass one gets the built-in core, not a phantom PeakMiner.
  test('the default existence check finds nothing on PATH', () => {
    const r = createMinerEngine({ env: { PATH: '/usr/bin' }, createCore: CORE });
    expect(r.miner).toBeInstanceOf(PearlEngine);
  });
});
