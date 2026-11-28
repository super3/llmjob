'use strict';

const path = require('path');
const { EventEmitter } = require('events');
const run = require('../src/shared/neb/run');
const blake3 = require('../src/shared/miner/blake3');

const NEB = 'neb1kwgy6lnm774h2cckdag40sj6sw25cpustla0gu5e37n0mn30wnzsjvwr4n';

// A dataset file with one sample in each of two datasets.
function datasetFile() {
  const head = Buffer.alloc(8);
  head.write('NMDS');
  head.writeUInt32LE(2, 4);
  const set = () => {
    const h = Buffer.alloc(12);
    h.write('NMD2');
    h.writeUInt32LE(1, 4);
    h.writeUInt32LE(1, 8);
    return Buffer.concat([h, Buffer.alloc(785, 1)]);
  };
  return Buffer.concat([head, set(), set()]);
}

function fakeEngine(startReturns = true) {
  const e = new EventEmitter();
  e.start = jest.fn(() => startReturns);
  e.stop = jest.fn(() => e.emit('stopped', 0));
  return e;
}

function deps(over = {}) {
  const out = [];
  const errs = [];
  const engine = fakeEngine(over.startReturns);
  let clock = 1000000;
  const d = {
    settings: { nebAddress: NEB, worker: 'rig01', nebEndpoint: null, nebDataset: null },
    cards: [1],
    dataDir: path.join('/data', 'earn'),
    engineFor: jest.fn(() => engine),
    coreFactory: jest.fn(() => ({ label: 'neb_cuda' })),
    fs: { readFileSync: jest.fn(() => datasetFile()) },
    log: (line) => out.push(line),
    err: (line) => errs.push(line),
    logAt: jest.fn(),
    now: () => clock,
    resourcesPath: '/res',
    version: '9.9.9',
    proc: new EventEmitter(),
  };
  Object.assign(d, over.d);
  return { d, out, errs, engine, tick: (ms) => { clock += ms; } };
}

describe('splitNebPairs', () => {
  test('takes the NEB pairs out of --card-coin\'s value', () => {
    expect(run.splitNebPairs('0:pearl, 2:neb,1=NEB ,3:quantus,2:neb')).toEqual({ neb: [1, 2], rest: '0:pearl,3:quantus' });
    expect(run.splitNebPairs('5:neb')).toEqual({ neb: [5], rest: '' });
    expect(run.splitNebPairs('0:nebula,x:neb')).toEqual({ neb: [], rest: '0:nebula,x:neb' });
    expect(run.splitNebPairs(null)).toEqual({ neb: [], rest: '' });
  });
});

describe('takeNebCards', () => {
  const gpus = [{ index: 0, name: 'A' }, { index: 1, name: 'B' }, { index: 2, name: 'C' }];

  test('no NEB card: the rig as it is', () => {
    expect(run.takeNebCards({ gpus, nebCards: null })).toEqual({ cards: [], rest: gpus, unlisted: false, skipped: [] });
    expect(run.takeNebCards({})).toEqual({ cards: [], rest: [], unlisted: false, skipped: [] });
    expect(run.splitLines(run.takeNebCards({ gpus }))).toEqual([]);
  });

  test('the NEB cards come off the rig; one the rig lacks is skipped, with a line', () => {
    const split = run.takeNebCards({ gpus, nebCards: [1, 7] });
    expect(split).toEqual({ cards: [1], rest: [gpus[0], gpus[2]], unlisted: false, skipped: [7] });
    expect(run.splitLines(split)).toEqual(['warning: --card-coin gives GPU 7 to NEB, but this run has no such card']);
  });

  test('a rig whose cards are not listed: NEB takes the indices as given, and says no other coin shares it', () => {
    const split = run.takeNebCards({ gpus: [], nebCards: [0, 1] });
    expect(split).toEqual({ cards: [0, 1], rest: [], unlisted: true, skipped: [] });
    expect(run.splitLines(split)).toEqual(['NEB on GPU 0, 1: this rig\'s cards are not listed, so no other coin shares it']);
  });
});

describe('loadDatasets', () => {
  test('parses the file and hashes it', () => {
    const buf = datasetFile();
    const r = run.loadDatasets('/x/neb-datasets.bin', { readFileSync: () => buf });
    expect(r.datasets.map((d) => d.n)).toEqual([1, 1]);
    expect(r.hash).toBe(blake3.hash(buf).toString('hex'));
  });

  test('a missing file says where to get it; a wrong one says so', () => {
    const missing = { readFileSync: () => { throw Object.assign(new Error('no'), { code: 'ENOENT' }); } };
    expect(() => run.loadDatasets('/x/neb-datasets.bin', missing)).toThrow(
      'the NEB dataset file is not at /x/neb-datasets.bin (ENOENT). Download neb-datasets.bin from '
      + 'https://github.com/SilasEigen/neb/releases and put it there, or name it with --neb-dataset');
    expect(() => run.loadDatasets('/x/f', { readFileSync: () => Buffer.from('hello') }))
      .toThrow('/x/f is not a NEB dataset file this miner reads: not a NEB dataset file');
  });
});

describe('start', () => {
  test('loads the dataset from the data folder, starts the engine on the cards, and logs its lines', () => {
    const t = deps();
    expect(run.start(t.d)).toBe(t.engine);
    expect(t.d.fs.readFileSync).toHaveBeenCalledWith(path.join('/data', 'earn', 'neb-datasets.bin'));
    expect(t.d.coreFactory).toHaveBeenCalledWith({ resourcesPath: '/res', log: t.d.logAt });
    expect(t.d.engineFor).toHaveBeenCalledWith({ factory: { label: 'neb_cuda' } });
    expect(t.engine.start).toHaveBeenCalledWith({
      address: NEB, worker: 'rig01', endpoint: 'neb.rabidmining.com:3470', cards: [1],
      datasets: expect.any(Array), datasetHash: blake3.hash(datasetFile()).toString('hex'),
    });
    expect(t.out).toEqual(['neb:        GPU 1 → neb1kwgy…vwr4n on neb.rabidmining.com:3470']);

    t.engine.emit('log', { level: 'info', line: 'neb: logged in to the pool' });
    expect(t.d.logAt).toHaveBeenCalledWith('info', 'neb: logged in to the pool');
    t.engine.emit('error', new Error('already logged'));
    t.engine.emit('event', { type: 'connected', gpuIndex: 1 });
    // A status line every 10 s: the cards' rates and shares added up.
    t.tick(run.LOG_MS);
    t.engine.emit('event', { type: 'status', gpuIndex: 1, hashrate: 8e8, accepted: 3, rejected: 1 });
    t.engine.emit('event', { type: 'status', gpuIndex: 2, hashrate: 4e8, accepted: 1, rejected: 0 });
    t.tick(run.LOG_MS);
    t.engine.emit('event', { type: 'status', gpuIndex: 1, hashrate: 8e8, accepted: 4, rejected: 1 });
    t.engine.emit('stopped', 1);
    expect(t.out.slice(1)).toEqual([
      '⛏  neb 800.00 MH/s · 3 accepted · 1 rejected', '⛏  neb 1.20 GH/s · 5 accepted · 1 rejected', 'neb engine exited (code 1)',
    ]);
  });

  test('--neb-dataset and --neb-pool; every card when none is named', () => {
    const t = deps({ d: { cards: null } });
    Object.assign(t.d.settings, { nebDataset: '/my/file.bin', nebEndpoint: 'pool.example:3470' });
    run.start(t.d);
    expect(t.d.fs.readFileSync).toHaveBeenCalledWith('/my/file.bin');
    expect(t.out).toEqual(['neb:        every card → neb1kwgy…vwr4n on pool.example:3470']);
    expect(t.engine.start.mock.calls[0][0]).toMatchObject({ endpoint: 'pool.example:3470', cards: null });
  });

  test('no dataset file, no core, or an engine that will not start: null, and why', () => {
    const none = deps({ d: { fs: { readFileSync: () => { throw Object.assign(new Error('no'), { code: 'ENOENT' }); } } } });
    expect(run.start(none.d)).toBeNull();
    expect(none.errs[0]).toMatch(/^NEB: the NEB dataset file is not at /);
    expect(none.d.coreFactory).not.toHaveBeenCalled();

    const core = deps({ d: { coreFactory: () => null } });
    expect(run.start(core.d)).toBeNull();
    expect(core.errs).toEqual(['no NEB core loaded; the lines above say why. NEB mines nothing.']);

    const dead = deps({ startReturns: false });
    expect(run.start(dead.d)).toBeNull();
    expect(dead.errs).toEqual(['neb engine failed to start — see the error above']);
  });
});

describe('runAlone', () => {
  test('mines until a signal, and ends 0', async () => {
    const t = deps();
    const done = run.runAlone(t.d);
    expect(t.out.slice(0, 2)).toEqual(['LLMJob Earn CLI v9.9.9', 'coin:       NEB (Neural Engine Byte), experimental']);
    t.d.proc.emit('SIGINT');
    t.d.proc.emit('SIGTERM');
    expect(await done).toBe(0);
    expect(t.engine.stop).toHaveBeenCalledTimes(1);
    expect(t.out).toContain('shutting down…');
  });

  test('an engine that gives up on its own ends with its code', async () => {
    const t = deps();
    const done = run.runAlone(t.d);
    t.engine.emit('stopped', 1);
    expect(await done).toBe(1);
  });

  test('an engine that never starts ends 1', async () => {
    const t = deps({ d: { coreFactory: () => null } });
    expect(await run.runAlone(t.d)).toBe(1);
  });
});
