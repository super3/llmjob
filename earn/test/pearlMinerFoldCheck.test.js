'use strict';

// The rotation fold in the miner (--rotation-fold, PEARL_FOLD_ROT=1; off by
// default): a card whose core opens on it (core.tile) has 24 of the fold's
// hits recomputed before it is trusted, and every share after that. A wrong
// hit, a check that runs out of time or a fold that will not start puts the
// card back on the normal kernel, with one line that says why.
// shared/foldCheck has the check itself; pearlMiner.test.js the rest of the miner.

jest.mock('../src/main/gpuClocks', () => ({
  lockMemoryClock: jest.fn(() => ({ ok: true, error: null })),
  resetMemoryClock: jest.fn(() => ({ ok: true, error: null })),
}));

// Stand-in hits (marked `fake`) are right unless they say otherwise; a genuine
// hit is recomputed for real.
jest.mock('../src/shared/miner/checkHit', () => {
  const real = jest.requireActual('../src/shared/miner/checkHit');
  return {
    checkHit: jest.fn((hit, jobKey, profile) => (hit.fake ? hit.wrong || null : real.checkHit(hit, jobKey, profile))),
  };
});

const { EventEmitter } = require('events');
const { PearlMiner } = require('../src/main/pearlMiner');
const { checkHit } = require('../src/shared/miner/checkHit');
const foldCheck = require('../src/shared/foldCheck');
const coreOffsetTrial = require('../src/shared/coreOffsetTrial');
const { encode } = require('../src/shared/miner/stratum');
const {
  PROFILE, shareBound, profileForCore, penalizedAdjustmentFactor, difficultyAdjustmentFactor,
} = require('../src/shared/miner/pearlhash');
const { hitFor: genuine } = require('./helpers/pearlHit');

const { CHECK_HITS, CHECK_MS } = foldCheck;

const ADDR = 'prl1px5ervx6ftaegmdhqa5ajemh20j2uw7l9jt5j5s97rljp72yt3s8qncrxud';
const HEADER = '000000203a49fea8b6d42c60c543fe0f029749787679372495e5d2d1007e29e3'
  + '25e1c08065a0cedc057ba091aabd10476017b3d3f4e38eafb59707ded57f71e1feb7b5d155918b6a0dea0018';
// The fold's tile: one thread's sums, 12 rows by 16 columns.
const TILE = {
  rows: [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88],
  cols: [0, 1, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 49, 56, 57],
};
// m and n small enough to build real trees here, and big enough for the tile.
const SMALL = { ...PROFILE, m: 128, n: 64 };
const ROT = profileForCore(SMALL, { tile: TILE });
const MAX = (1n << 256n) - 1n;
// A pool target easy enough that nearly every hash is a share on the fold's tile, and one the normal
// kernel's tile (16 by 16, a third more work an attempt) can still scale.
const T = MAX / BigInt(Math.max(penalizedAdjustmentFactor(ROT), penalizedAdjustmentFactor(SMALL)));
const EASY = T.toString(16).padStart(64, '0');
const JOB = '00000000_2097152';
const AUTH = '{"id":1,"result":true,"error":null}\n';
// Genuine shares of the pool's job on the fold's tile: one right, one whose fold is wrong.
const GOOD = genuine(ROT, Buffer.from(HEADER, 'hex'), { bound: shareBound(T, ROT), jobId: JOB });
const WRONG = genuine(ROT, Buffer.from(HEADER, 'hex'), {
  bound: shareBound(T, ROT), jobId: JOB, tamper: (jp) => { jp[5] ^= 1; },
});
const MISMATCH = 'jackpot hash mismatch (fold/transcript)';
const CHECK_HEADER = Buffer.alloc(76, 0x5a);

const settings = { address: ADDR, worker: 'rig01', endpoint: 'us.pearl.herominers.com:1200', profile: SMALL };
const GPU = [{ index: 0, name: 'NVIDIA GeForce RTX 4090' }];
const CARD = 'GPU 0 (NVIDIA GeForce RTX 4090)';
const ON = CARD + ': rotation fold on (experimental). Checking 24 of its hits on the CPU before trusting it';
const PASSED = CARD + ': rotation fold passed its check (24 of 24 hits right). Mining with it';
const FAILED = (why, card = CARD) => card + ': the rotation fold failed its check (' + why + '). No share from it was '
  + 'sent, and this card mines on the normal kernel for the rest of this run. Please report this line '
  + '(earn/native/README.md, "A second RTX 40 fold").';

function makeSocket() {
  const s = new EventEmitter();
  s.written = [];
  s.write = jest.fn((d) => { s.written.push(d); return true; });
  s.destroy = jest.fn(() => { s.destroyed = true; });
  return s;
}

function jobLine(over = {}) {
  return encode({
    id: null, method: 'mining.notify',
    params: { job_id: JOB, header: HEADER, target: EASY, height: 103353, cert_version: 3, ...over },
  });
}

// A rig of one RTX 4090 whose core opens on the rotation fold whenever the
// miner lets it choose (cardK), as the CUDA core does with PEARL_FOLD_ROT=1.
// `over.core(core, opts)` dresses each core as it is made, or throws.
function rig(over = {}) {
  const t = { now: 0 };
  const made = [];
  const createCore = jest.fn((profile, opts) => {
    const c = new EventEmitter();
    c.setJob = jest.fn();
    c.stop = jest.fn();
    c.opts = opts;
    if (!over.silentDevice) c.device = { index: opts.deviceIndex, name: 'NVIDIA GeForce RTX 4090' };
    if (opts.cardK) {
      c.tile = TILE;
      c.kNote = 'rotation fold (experimental)';
    }
    if (over.core) over.core(c, opts);
    made.push(c);
    return c;
  });
  const b = { t, made, createCore, sock: null };
  const connect = () => { b.sock = makeSocket(); return b.sock; };
  const m = new PearlMiner({ connect, createCore, reconnectMs: 0, restartMs: 10, now: () => t.now });
  const logs = [];
  m.on('log', (l) => logs.push(l));
  m.on('error', () => {});
  Object.assign(b, { m, logs });
  b.start = (extra = {}) => {
    m.start({ ...settings, gpus: GPU, ...extra });
    b.sock.emit('connect');
    b.sock.emit('data', AUTH);
    b.sock.emit('data', jobLine());
    return b;
  };
  b.lines = (level) => logs
    .filter((l) => /rotation fold|normal kernel/.test(l.line) && (!level || l.level === level))
    .map((l) => l.line);
  b.submits = () => b.sock.written.map((w) => JSON.parse(w)).filter((x) => x.method === 'mining.submit');
  return b;
}

const jobs = (core) => core.setJob.mock.calls.map((c) => c[0].jobId);
const lastJob = (core) => core.setJob.mock.calls[core.setJob.mock.calls.length - 1][0];
// A stand-in hit of a check's job: right, or wrong with `wrong` as the reason.
const checkHitOf = (job, wrong) => ({ fake: true, jobId: job.jobId, jackpotHash: Buffer.alloc(32), wrong });

function tick(b, core, th = 330.5) {
  b.t.now += 500;
  core.emit('hashrate', th);
}

// Take a card through its check: its first sample starts it, 24 right hits pass it.
function pass(b, core) {
  tick(b, core);
  const job = lastJob(core);
  for (let n = 0; n < CHECK_HITS; n += 1) core.emit('hit', checkHitOf(job));
  return job;
}

beforeEach(() => {
  checkHit.mockClear();
});

describe('a card that opens on the rotation fold', () => {
  test('is named with the fold, and told it will be checked', () => {
    const b = rig().start();
    expect(b.createCore.mock.calls[0][1]).toMatchObject({ cardK: true, checkHits: true, deviceIndex: 0 });
    expect(b.logs.map((l) => l.line)).toEqual(expect.arrayContaining([
      'mining on GPU 0 · NVIDIA GeForce RTX 4090 · rotation fold (experimental)',
      ON,
    ]));
    // It mines the pool's job until its first hashrate sample.
    expect(jobs(b.made[0])).toEqual([JOB]);
  });

  test('checks 24 hits of a job of its own, then goes back to the pool\'s', () => {
    const b = rig().start();
    const [core] = b.made;
    tick(b, core, 0);
    expect(jobs(core)).toEqual([JOB]);
    tick(b, core, 330.5);
    const job = lastJob(core);
    // No block's header, and a target set for 3 hits a second at the card's own rate.
    expect(job.jobId).toBe('fold-check-1');
    expect(job.header).toEqual(CHECK_HEADER);
    expect(job.target).toBe(coreOffsetTrial.checkBound(330.5, difficultyAdjustmentFactor(ROT), 3));
    // A job the pool sends meanwhile waits for the check to end.
    b.sock.emit('data', jobLine({ job_id: 'next' }));
    tick(b, core);
    expect(jobs(core)).toEqual([JOB, 'fold-check-1']);
    for (let n = 0; n < CHECK_HITS - 1; n += 1) core.emit('hit', checkHitOf(job));
    expect(b.lines()).toEqual([expect.stringContaining('rotation fold (experimental)'), ON]);
    core.emit('hit', checkHitOf(job));
    expect(b.lines()).toEqual([expect.stringContaining('rotation fold (experimental)'), ON, PASSED]);
    expect(jobs(core)).toEqual([JOB, 'fold-check-1', 'next']);
    expect(checkHit).toHaveBeenCalledTimes(CHECK_HITS);
    // Nothing of the check went to the pool, and hits that arrive late change nothing.
    core.emit('hit', checkHitOf(job, MISMATCH));
    tick(b, core);
    expect(b.submits()).toEqual([]);
    expect(b.made).toHaveLength(1);
    expect(core.stop).not.toHaveBeenCalled();
  });

  test('has every share recomputed before it is sent, before the check and after', () => {
    const b = rig().start();
    const [core] = b.made;
    core.emit('hit', GOOD);
    expect(b.submits()).toHaveLength(1);
    pass(b, core);
    b.sock.emit('data', jobLine());
    core.emit('hit', GOOD);
    expect(b.submits()).toHaveLength(2);
    // Twice for the two shares, for real, and 24 times for the check.
    expect(checkHit).toHaveBeenCalledTimes(CHECK_HITS + 2);
    expect(b.lines('error')).toEqual([]);
  });

  test('goes back to the normal kernel at one wrong hit in its check', () => {
    const b = rig().start();
    const [core] = b.made;
    tick(b, core);
    const job = lastJob(core);
    for (let n = 0; n < 5; n += 1) core.emit('hit', checkHitOf(job));
    core.emit('hit', checkHitOf(job, MISMATCH));
    expect(b.lines('error')).toEqual([FAILED(MISMATCH)]);
    // The fold's core is stopped, and the card opens again with no choice of k.
    expect(core.stop).toHaveBeenCalledTimes(1);
    expect(b.made).toHaveLength(2);
    expect(b.made[1].opts).toMatchObject({ cardK: false, deviceIndex: 0 });
    expect(b.made[1].tile).toBeUndefined();
    // On fresh salts, and on the pool's job: no second check.
    expect(b.made[1].opts.saltBase).toBeGreaterThan(core.opts.saltBase);
    expect(jobs(b.made[1])).toEqual([JOB]);
    tick(b, b.made[1]);
    expect(jobs(b.made[1])).toEqual([JOB]);
    expect(b.m.devices()).toEqual([{ index: 0, name: 'NVIDIA GeForce RTX 4090' }]);
    // What the stopped core still reports is dropped: its check's hits, and a share.
    core.emit('hit', checkHitOf(job));
    core.emit('hit', GOOD);
    tick(b, core);
    expect(b.submits()).toEqual([]);
    expect(b.lines('error')).toHaveLength(1);
    expect(b.made).toHaveLength(2);
  });

  test('fails its check on a hit whose hash misses the check\'s target', () => {
    const b = rig().start();
    const [core] = b.made;
    tick(b, core);
    core.emit('hit', { ...checkHitOf(lastJob(core)), jackpotHash: Buffer.alloc(32, 0xff) });
    expect(b.lines('error')).toEqual([FAILED('the hash does not meet the check\'s target')]);
    expect(b.made).toHaveLength(2);
  });

  test('fails its check when it finds too few hits in a minute', () => {
    const b = rig().start();
    const [core] = b.made;
    tick(b, core);
    const job = lastJob(core);
    for (let n = 0; n < 3; n += 1) core.emit('hit', checkHitOf(job));
    b.t.now += CHECK_MS - 500;
    core.emit('hashrate', 330.5);
    expect(b.lines('error')).toEqual([]);
    tick(b, core);
    expect(b.lines('error')).toEqual([FAILED('only 3 of the 24 hits it needs in 60 s')]);
    expect(b.made[1].opts.cardK).toBe(false);
    expect(jobs(b.made[1])).toEqual([JOB]);
  });

  test('goes back to the normal kernel at a wrong share, and does not send it', () => {
    const b = rig().start();
    const [core] = b.made;
    pass(b, core);
    core.emit('hit', WRONG);
    expect(b.submits()).toEqual([]);
    expect(b.lines('error')).toEqual([FAILED(MISMATCH)]);
    expect(core.stop).toHaveBeenCalledTimes(1);
    expect(b.made[1].opts.cardK).toBe(false);
  });

  test('is not checked twice in a run: a card that passed comes back from a pause on the fold', () => {
    const b = rig().start();
    pass(b, b.made[0]);
    b.m.pauseCards([0]);
    b.m.resumeCards([0]);
    expect(b.made).toHaveLength(2);
    expect(b.made[1].opts.cardK).toBe(true);
    expect(jobs(b.made[1])).toEqual([JOB]);
    tick(b, b.made[1]);
    expect(jobs(b.made[1])).toEqual([JOB]);
    expect(b.lines().filter((l) => l === ON)).toHaveLength(1);
    // Its shares are still recomputed.
    b.made[1].emit('hit', GOOD);
    expect(b.submits()).toHaveLength(1);
    expect(checkHit).toHaveBeenCalledTimes(CHECK_HITS + 1);
  });

  test('starts its check again when it was paused in the middle of one', () => {
    const b = rig().start();
    const [core] = b.made;
    tick(b, core);
    const first = lastJob(core);
    for (let n = 0; n < CHECK_HITS - 1; n += 1) core.emit('hit', checkHitOf(first));
    b.m.pauseCards([0]);
    b.m.resumeCards([0]);
    const again = b.made[1];
    expect(b.lines().filter((l) => l === ON)).toHaveLength(2);
    tick(b, again);
    expect(lastJob(again).jobId).toBe('fold-check-2');
    // A hit of the first check, from the stopped core, counts for nothing.
    core.emit('hit', checkHitOf(first));
    expect(b.lines()).not.toContain(PASSED);
    for (let n = 0; n < CHECK_HITS; n += 1) again.emit('hit', checkHitOf(lastJob(again)));
    expect(b.lines()).toContain(PASSED);
    expect(jobs(again)).toEqual([JOB, 'fold-check-2', JOB]);
  });

  test('passes without a card to hand the job to when it was paused at its last hit', () => {
    const b = rig().start();
    const [core] = b.made;
    tick(b, core);
    const job = lastJob(core);
    for (let n = 0; n < CHECK_HITS - 1; n += 1) core.emit('hit', checkHitOf(job));
    b.m.pauseCards([0]);
    core.emit('hit', checkHitOf(job));
    expect(b.lines()).toContain(PASSED);
    expect(jobs(core)).toEqual([JOB, 'fold-check-1']);
  });

  test('opens on the normal kernel after a pause when a late share of its fold was wrong', () => {
    const b = rig().start();
    const [core] = b.made;
    pass(b, core);
    b.m.pauseCards([0]);
    core.emit('hit', WRONG);
    expect(b.lines('error')).toEqual([FAILED(MISMATCH)]);
    // Nothing to reopen while it is paused; more late hits say nothing more.
    expect(b.made).toHaveLength(1);
    core.emit('hit', GOOD);
    expect(b.submits()).toEqual([]);
    expect(b.lines('error')).toHaveLength(1);
    b.m.resumeCards([0]);
    expect(b.made[1].opts.cardK).toBe(false);
  });

  test('is a card down like any other when it will not open on the normal kernel either', () => {
    const b = rig({
      core: (c, opts) => {
        if (!opts.cardK) throw new Error('rotation fold: still asked for');
      },
    }).start();
    const [core] = b.made;
    const rates = [];
    b.m.on('hashrate', (th, device) => rates.push([th, device.index]));
    tick(b, core);
    core.emit('hit', checkHitOf(lastJob(core), MISMATCH));
    expect(b.lines('error')).toEqual([
      FAILED(MISMATCH),
      'could not open ' + CARD + ' again on the normal kernel: rotation fold: still asked for',
    ]);
    expect(rates[rates.length - 1]).toEqual([0, 0]);
    expect(b.m.devices()).toEqual([]);
    b.m.stop();
  });

  test('a core that names no card is checked and falls back the same way', () => {
    const b = rig({ silentDevice: true }).start({ gpus: [] });
    const [core] = b.made;
    expect(b.lines()).toEqual(['the GPU: rotation fold on (experimental). Checking 24 of its hits on the CPU before trusting it']);
    tick(b, core);
    core.emit('hit', checkHitOf(lastJob(core), MISMATCH));
    expect(b.lines('error')).toEqual([FAILED(MISMATCH, 'the GPU')]);
    expect(b.made[1].opts.cardK).toBe(false);
    // And when that second open fails.
    const c = rig({
      silentDevice: true,
      core: (x, opts) => {
        if (!opts.cardK) throw new Error('no CUDA device');
      },
    }).start({ gpus: [] });
    tick(c, c.made[0]);
    c.made[0].emit('hit', checkHitOf(lastJob(c.made[0]), MISMATCH));
    expect(c.lines('error')).toEqual([
      FAILED(MISMATCH, 'the GPU'),
      'could not open the GPU again on the normal kernel: no CUDA device',
    ]);
    c.m.stop();
  });
});

describe('a rotation fold that will not start', () => {
  // The core's own refusal (native/src/pearl_host_rot.inc, rot_create).
  const REFUSED = 'rotation fold: the driver refused the cubin (200)';

  test('opens the card on the normal kernel and says why, once', () => {
    const b = rig({
      core: (c, opts) => {
        if (opts.cardK) throw new Error(REFUSED);
      },
    }).start();
    expect(b.createCore.mock.calls.map((c) => c[1].cardK)).toEqual([true, false]);
    expect(b.lines()).toEqual([
      'GPU 0: rotation fold asked for but not used: the driver refused the cubin (200). Mining on the normal kernel',
    ]);
    expect(b.logs.find((l) => /rotation fold asked/.test(l.line)).level).toBe('warn');
    expect(b.m.devices()).toEqual([{ index: 0, name: 'NVIDIA GeForce RTX 4090' }]);
    expect(jobs(b.made[0])).toEqual([JOB]);
    // And stays there when the card is opened again.
    b.m.pauseCards([0]);
    b.m.resumeCards([0]);
    expect(b.createCore.mock.calls.map((c) => c[1].cardK)).toEqual([true, false, false]);
    expect(b.lines()).toHaveLength(1);
  });

  test('on a rig with no card list, and thrown as text', () => {
    const b = rig({
      core: (c, opts) => {
        if (opts.cardK) throw REFUSED;
      },
    }).start({ gpus: [] });
    expect(b.lines()).toEqual([
      'the GPU: rotation fold asked for but not used: the driver refused the cubin (200). Mining on the normal kernel',
    ]);
    expect(b.m.cardCount()).toBe(1);
  });

  test('any other refusal is the card\'s own, as before', () => {
    const b = rig({
      core: () => { throw new Error('not enough free VRAM for the rank-128 profile'); },
    });
    const errors = [];
    b.m.on('error', (e) => errors.push(e.message));
    b.m.start({ ...settings, gpus: GPU });
    expect(b.createCore).toHaveBeenCalledTimes(1);
    expect(errors).toEqual(['not enough free VRAM for the rank-128 profile']);
    expect(b.lines()).toEqual([]);
  });
});

describe('a card on the normal kernel', () => {
  test('has no fold check, and a stray hit of one is dropped', () => {
    const b = rig({ core: (c) => { delete c.tile; delete c.kNote; } }).start();
    const [core] = b.made;
    tick(b, core);
    expect(jobs(core)).toEqual([JOB]);
    core.emit('hit', { fake: true, jobId: 'fold-check-7', jackpotHash: Buffer.alloc(32) });
    expect(checkHit).not.toHaveBeenCalled();
    expect(b.submits()).toEqual([]);
    expect(b.lines()).toEqual([]);
  });
});
