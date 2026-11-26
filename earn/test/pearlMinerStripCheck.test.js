'use strict';

// The strip fold in the miner: an RX 7000 card (gfx1100, gfx1101) whose AMD
// core mines a smaller m than mainnet's (core.m). On Windows, where strips had
// not run when they were turned on, the card has 24 hits recomputed before it
// is trusted and every share after that, as a card on the rotation fold has,
// and it goes back to mainnet's m when its check fails or the pool rejects its
// shares. On Linux nothing changes. pearlMinerFoldCheck.test.js has the
// rotation fold's side of the same code.

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
const { CHECK_HITS, CHECK_MS } = require('../src/shared/foldCheck');
const { encode } = require('../src/shared/miner/stratum');
const { PROFILE, shareBound, profileForCore, penalizedAdjustmentFactor } = require('../src/shared/miner/pearlhash');
const { hitFor: genuine } = require('./helpers/pearlHit');

const ADDR = 'prl1px5ervx6ftaegmdhqa5ajemh20j2uw7l9jt5j5s97rljp72yt3s8qncrxud';
const HEADER = '000000203a49fea8b6d42c60c543fe0f029749787679372495e5d2d1007e29e3'
  + '25e1c08065a0cedc057ba091aabd10476017b3d3f4e38eafb59707ded57f71e1feb7b5d155918b6a0dea0018';
// The profile stands in for mainnet's, and half its m for the card's strips.
const SMALL = { ...PROFILE, m: 128, n: 64 };
const STRIP_M = 64;
const STRIP = profileForCore(SMALL, { k: SMALL.k, m: STRIP_M });
const MAX = (1n << 256n) - 1n;
const T = MAX / BigInt(Math.max(penalizedAdjustmentFactor(STRIP), penalizedAdjustmentFactor(SMALL)));
const EASY = T.toString(16).padStart(64, '0');
const JOB = '00000000_2097152';
const AUTH = '{"id":1,"result":true,"error":null}\n';
const AUTH_V2 = '{"id":1,"result":true,"error":null,"type":"v2"}\n';
// Genuine shares of the pool's job: at the strips' m, right and wrong, and at the profile's m.
const GOOD = genuine(STRIP, Buffer.from(HEADER, 'hex'), { bound: shareBound(T, STRIP), jobId: JOB });
const WRONG = genuine(STRIP, Buffer.from(HEADER, 'hex'), {
  bound: shareBound(T, STRIP), jobId: JOB, tamper: (jp) => { jp[5] ^= 1; },
});
const PLAIN = genuine(SMALL, Buffer.from(HEADER, 'hex'), { bound: shareBound(T, SMALL), jobId: JOB });
const MISMATCH = 'jackpot hash mismatch (fold/transcript)';

const settings = { address: ADDR, worker: 'rig01', endpoint: 'us.pearl.herominers.com:1200', profile: SMALL };
const NAME = 'AMD Radeon RX 7900 XTX';
const CARD = 'GPU 0 (' + NAME + ')';
const REPORT = 'Please report this line (earn/native/amd/README.md, "Testing on real hardware").';
const ON = CARD + ': strip fold on. Checking 24 of its hits on the CPU before trusting it';
const PASSED = CARD + ': strip fold passed its check (24 of 24 hits right). Mining with it';
const RECHECK = CARD + ': a share of the strip fold came out wrong (' + MISMATCH + '). It was not sent. '
  + 'Checking the fold again before trusting it';
const FAILED = (why) => CARD + ': the strip fold failed its check (' + why + '). No share from it was '
  + 'sent, and this card mines at mainnet\'s m (strips off) for the rest of this run. ' + REPORT;
const REJECTED = (why, card = CARD) => card + ': the pool rejected 3 shares of the strip fold ([23] ' + why
  + ') before it accepted 20. This card mines at mainnet\'s m (strips off) for the rest of this run. ' + REPORT;

function makeSocket() {
  const s = new EventEmitter();
  s.written = [];
  s.write = jest.fn((d) => { s.written.push(d); return true; });
  s.destroy = jest.fn(() => { s.destroyed = true; });
  return s;
}

function jobLine() {
  return encode({
    id: null, method: 'mining.notify',
    params: { job_id: JOB, header: HEADER, target: EASY, height: 103353, cert_version: 3 },
  });
}

// A rig of RX 7900 XTX cards whose core mines the strips' m whenever the miner
// lets it choose (cardK), as the AMD core does. `over.platform` is the PC's
// (Windows unless said), `over.cards` how many, `over.m` what the core reports
// in place of the strips' m, and `over.core(core, opts)` dresses each core as
// it is made, or throws.
function rig(over = {}) {
  const t = { now: 0 };
  const made = [];
  const createCore = jest.fn((profile, opts) => {
    const c = new EventEmitter();
    c.setJob = jest.fn();
    c.stop = jest.fn();
    c.opts = opts;
    c.device = { index: opts.deviceIndex, name: NAME };
    c.k = profile.k;
    c.m = opts.cardK ? over.m || STRIP_M : profile.m;
    if (over.core) over.core(c, opts);
    made.push(c);
    return c;
  });
  const b = { t, made, createCore, sock: null };
  const connect = () => { b.sock = makeSocket(); return b.sock; };
  const m = new PearlMiner({
    connect, createCore, reconnectMs: 0, restartMs: 10, now: () => t.now, platform: over.platform || 'win32',
  });
  const logs = [];
  m.on('log', (l) => logs.push(l));
  m.on('error', () => {});
  Object.assign(b, { m, logs });
  b.start = (auth = AUTH) => {
    const gpus = Array.from({ length: over.cards || 1 }, (_, index) => ({ index, name: NAME }));
    m.start({ ...settings, gpus });
    b.sock.emit('connect');
    b.sock.emit('data', auth);
    b.sock.emit('data', jobLine());
    return b;
  };
  b.lines = (level) => logs
    .filter((l) => /strip/.test(l.line) && (!level || l.level === level))
    .map((l) => l.line);
  b.submits = () => b.sock.written.map((w) => JSON.parse(w)).filter((x) => x.method === 'mining.submit');
  // Send one share from a core and have the pool answer it: true, or the reason it rejects it for.
  b.share = (core, answer, hit = GOOD) => {
    core.emit('hit', hit);
    const { id } = b.submits().pop();
    const reply = answer === true ? { id, result: true, error: null } : { id, result: null, error: [23, answer] };
    b.sock.emit('data', JSON.stringify(reply) + '\n');
  };
  return b;
}

const jobs = (core) => core.setJob.mock.calls.map((c) => c[0].jobId);
const lastJob = (core) => core.setJob.mock.calls[core.setJob.mock.calls.length - 1][0];
// A stand-in hit of a check's job: right, or wrong with `wrong` as the reason.
const checkHitOf = (job, wrong) => ({ fake: true, jobId: job.jobId, jackpotHash: Buffer.alloc(32), wrong });

function tick(b, core, th = 97.5) {
  b.t.now += 500;
  core.emit('hashrate', th);
}

// Take a card through its check: its first sample starts it, 24 right hits pass it.
function pass(b, core) {
  tick(b, core);
  const job = lastJob(core);
  for (let n = 0; n < CHECK_HITS; n += 1) core.emit('hit', checkHitOf(job));
}

beforeEach(() => {
  checkHit.mockClear();
});

describe('an RX 7000 card that folds strips on Windows', () => {
  test('is checked on 24 hits of a job of its own, then mines the pool\'s', () => {
    const b = rig().start();
    const [core] = b.made;
    expect(core.opts).toMatchObject({ cardK: true, deviceIndex: 0 });
    expect(b.logs.map((l) => l.line)).toEqual(expect.arrayContaining([
      'mining on GPU 0 · ' + NAME + ' · m ' + STRIP_M,
      ON,
    ]));
    expect(b.m.cores[0].profile).toEqual(STRIP);
    expect(jobs(core)).toEqual([JOB]);
    tick(b, core);
    const job = lastJob(core);
    expect(job.jobId).toBe('fold-check-1');
    for (let n = 0; n < CHECK_HITS; n += 1) core.emit('hit', checkHitOf(job));
    expect(b.lines()).toEqual([ON, PASSED]);
    expect(jobs(core)).toEqual([JOB, 'fold-check-1', JOB]);
    // Nothing of the check went to the pool.
    expect(b.submits()).toEqual([]);
    expect(b.made).toHaveLength(1);
  });

  test('has every share recomputed before it is sent, before the check and after', () => {
    const b = rig().start();
    const [core] = b.made;
    core.emit('hit', GOOD);
    pass(b, core);
    core.emit('hit', GOOD);
    expect(b.submits()).toHaveLength(2);
    expect(checkHit).toHaveBeenCalledTimes(CHECK_HITS + 2);
    expect(b.lines('error')).toEqual([]);
  });

  test('goes back to mainnet\'s m at one wrong hit in its check, with one line that says why', () => {
    const b = rig().start();
    const [core] = b.made;
    tick(b, core);
    const job = lastJob(core);
    core.emit('hit', checkHitOf(job));
    core.emit('hit', checkHitOf(job, MISMATCH));
    expect(b.lines('error')).toEqual([FAILED(MISMATCH)]);
    // The core is stopped, and the card opens again with no choice of its own: the profile's m.
    expect(core.stop).toHaveBeenCalledTimes(1);
    expect(b.made).toHaveLength(2);
    expect(b.made[1].opts).toMatchObject({ cardK: false, deviceIndex: 0 });
    expect(b.made[1].opts.saltBase).toBeGreaterThan(core.opts.saltBase);
    expect(b.m.cores[0].profile).toBe(SMALL);
    // On the pool's job, with no second check, and its shares go out as any card's do.
    tick(b, b.made[1]);
    expect(jobs(b.made[1])).toEqual([JOB]);
    checkHit.mockClear();
    b.share(b.made[1], true, PLAIN);
    expect(b.submits()).toHaveLength(1);
    expect(checkHit).not.toHaveBeenCalled();
    // What the stopped core still reports is dropped.
    core.emit('hit', GOOD);
    expect(b.submits()).toHaveLength(1);
    expect(b.lines('error')).toHaveLength(1);
  });

  test('goes back to mainnet\'s m when its check finds too few hits in a minute', () => {
    const b = rig().start();
    const [core] = b.made;
    tick(b, core);
    b.t.now += CHECK_MS;
    core.emit('hashrate', 97.5);
    expect(b.lines('error')).toEqual([FAILED('only 0 of the 24 hits it needs in 60 s')]);
    expect(b.made[1].opts.cardK).toBe(false);
  });

  test('is checked again at its first wrong share, and goes back to mainnet\'s m at the second', () => {
    const b = rig().start();
    const [core] = b.made;
    pass(b, core);
    core.emit('hit', WRONG);
    expect(b.lines('warn')).toEqual([RECHECK]);
    expect(b.made).toHaveLength(1);
    pass(b, core);
    expect(b.lines().filter((l) => l === PASSED)).toHaveLength(2);
    core.emit('hit', WRONG);
    expect(b.submits()).toEqual([]);
    expect(b.lines('error')).toEqual([FAILED(MISMATCH)]);
    expect(b.made[1].opts.cardK).toBe(false);
  });

  test('says so when the card will not open again at mainnet\'s m', () => {
    const b = rig({
      core: (c, opts) => {
        if (!opts.cardK) throw new Error('out of memory');
      },
    }).start();
    tick(b, b.made[0]);
    b.made[0].emit('hit', checkHitOf(lastJob(b.made[0]), MISMATCH));
    expect(b.logs.map((l) => l.line)).toContain(
      'could not open ' + CARD + ' again at mainnet\'s m (strips off): out of memory',
    );
  });

  test('leaves the other cards of the rig on their strips', () => {
    const b = rig({ cards: 2 }).start();
    tick(b, b.made[0]);
    b.made[0].emit('hit', checkHitOf(lastJob(b.made[0]), MISMATCH));
    expect(b.made.map((c) => [c.opts.deviceIndex, c.opts.cardK])).toEqual([[0, true], [1, true], [0, false]]);
    expect(b.made[1].stop).not.toHaveBeenCalled();
    // The second card's check starts at its own first sample.
    tick(b, b.made[1]);
    expect(lastJob(b.made[1]).jobId).toBe('fold-check-2');
  });
});

describe('the pool rejecting shares of a card that folds strips on Windows', () => {
  test('puts the card back at mainnet\'s m at the third, with one line that says why', () => {
    const b = rig().start();
    const [core] = b.made;
    pass(b, core);
    b.share(core, true);
    b.share(core, 'invalid share');
    b.share(core, 'invalid share');
    expect(b.made).toHaveLength(1);
    b.share(core, 'bad proof');
    expect(b.lines('error')).toEqual([REJECTED('bad proof')]);
    expect(core.stop).toHaveBeenCalledTimes(1);
    expect(b.made[1].opts).toMatchObject({ cardK: false, deviceIndex: 0 });
    expect(b.m.cores[0].profile).toBe(SMALL);
    expect(jobs(b.made[1])).toEqual([JOB]);
    // Once a card: a rejection after that, of a share from either core, changes nothing.
    b.share(b.made[1], 'invalid share', PLAIN);
    expect(b.lines('error')).toHaveLength(1);
    expect(b.made).toHaveLength(2);
  });

  test('counts a share still unanswered when the card left its strips for nothing', () => {
    const b = rig().start();
    const [core] = b.made;
    pass(b, core);
    core.emit('hit', GOOD);
    const late = b.submits().pop().id;
    b.share(core, 'invalid share');
    b.share(core, 'invalid share');
    b.share(core, 'invalid share');
    b.sock.emit('data', JSON.stringify({ id: late, result: null, error: [23, 'invalid share'] }) + '\n');
    expect(b.lines('error')).toEqual([REJECTED('invalid share')]);
    expect(b.made).toHaveLength(2);
  });

  test('does not count a stale, duplicate or low difficulty share', () => {
    const b = rig().start();
    const [core] = b.made;
    pass(b, core);
    for (const why of ['Stale share', 'duplicate share', 'Low difficulty share', 'share above target', 'job not found']) {
      b.share(core, why);
      b.share(core, why);
    }
    expect(b.lines('error')).toEqual([]);
    expect(b.made).toHaveLength(1);
  });

  test('does not count a compressed share before the pool has accepted one', () => {
    const b = rig().start(AUTH_V2);
    const [core] = b.made;
    pass(b, core);
    b.share(core, 'invalid share');
    b.share(core, 'invalid share');
    b.share(core, 'invalid share');
    // Those went toward the compression fallback, not this card's.
    expect(b.lines()).toEqual([ON, PASSED]);
    expect(b.made).toHaveLength(1);
  });

  test('counts a compressed share once the pool has accepted one', () => {
    const b = rig().start(AUTH_V2);
    const [core] = b.made;
    pass(b, core);
    b.share(core, true);
    b.share(core, 'invalid share');
    b.share(core, 'invalid share');
    b.share(core, 'invalid share');
    expect(b.lines('error')).toEqual([REJECTED('invalid share')]);
  });

  test('stops counting once the pool has accepted 20 of the card\'s shares', () => {
    const b = rig().start();
    const [core] = b.made;
    pass(b, core);
    b.share(core, 'invalid share');
    b.share(core, 'invalid share');
    for (let n = 0; n < 20; n += 1) b.share(core, true);
    b.share(core, 'invalid share');
    b.share(core, 'invalid share');
    expect(b.lines('error')).toEqual([]);
    expect(b.made).toHaveLength(1);
  });

  test('counts each card of a rig on its own', () => {
    const b = rig({ cards: 2 }).start();
    b.share(b.made[0], 'invalid share');
    b.share(b.made[0], 'invalid share');
    b.share(b.made[1], 'invalid share');
    b.share(b.made[1], 'invalid share');
    expect(b.made).toHaveLength(2);
    b.share(b.made[1], 'invalid share');
    expect(b.lines('error')).toEqual([REJECTED('invalid share', 'GPU 1 (' + NAME + ')')]);
    expect(b.made.map((c) => [c.opts.deviceIndex, c.opts.cardK])).toEqual([[0, true], [1, true], [1, false]]);
    expect(b.made[0].stop).not.toHaveBeenCalled();
  });
});

describe('where the strip fold is not checked', () => {
  test('on Linux the card mines its strips as before: no check, no recomputed share, no fallback', () => {
    const b = rig({ platform: 'linux' }).start();
    const [core] = b.made;
    expect(b.m.cores[0].profile).toEqual(STRIP);
    tick(b, core);
    expect(jobs(core)).toEqual([JOB]);
    for (let n = 0; n < 4; n += 1) b.share(core, 'invalid share');
    expect(b.submits()).toHaveLength(4);
    expect(checkHit).not.toHaveBeenCalled();
    expect(b.lines()).toEqual([]);
    expect(b.made).toHaveLength(1);
  });

  test('on Windows with PEARL_AMD_M=131072 the core keeps mainnet\'s m, and there is nothing to check', () => {
    const b = rig({ m: SMALL.m }).start();
    const [core] = b.made;
    expect(core.opts.cardK).toBe(true);
    expect(b.m.cores[0].profile).toBe(SMALL);
    tick(b, core);
    expect(jobs(core)).toEqual([JOB]);
    for (let n = 0; n < 4; n += 1) b.share(core, 'invalid share', PLAIN);
    expect(checkHit).not.toHaveBeenCalled();
    expect(b.lines()).toEqual([]);
    expect(b.made).toHaveLength(1);
  });
});
