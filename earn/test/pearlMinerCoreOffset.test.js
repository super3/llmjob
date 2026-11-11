'use strict';

// --mine-core-offset auto in the miner: a card on trial mines turns at offset 0
// and at each step up, every step first having its hits recomputed, and keeps
// the fastest step that is right (shared/coreOffsetTrial has the
// method; the shells' plan says which cards are on trial). pearlMiner.test.js
// has the rest of the miner.

// As in pearlMiner.test.js: a miner built without `clocks` must still never
// change a real card's clocks.
jest.mock('../src/main/gpuClocks', () => ({
  lockMemoryClock: jest.fn(() => ({ ok: true, error: null })),
  resetMemoryClock: jest.fn(() => ({ ok: true, error: null })),
}));

// A step's check recomputes 100 hits at about 5 ms each. Most tests here are
// about what the miner does with the answers, so they send stand-in hits
// (checkHitOf below, marked `fake`) that are right unless they say otherwise.
// A genuine hit is still recomputed for real.
jest.mock('../src/shared/miner/checkHit', () => {
  const real = jest.requireActual('../src/shared/miner/checkHit');
  return {
    checkHit: jest.fn((hit, jobKey, profile) => (hit.fake ? hit.wrong || null : real.checkHit(hit, jobKey, profile))),
  };
});

const { EventEmitter } = require('events');
const { PearlMiner, RESTART_MS } = require('../src/main/pearlMiner');
const coreOffsetTrial = require('../src/shared/coreOffsetTrial');
const { TURN_MS: MEM_TURN_MS } = require('../src/shared/memClockTrial');
const { TRIAL_SECS: SHAPE_TRIAL_SECS } = require('../src/shared/shapePick');
const { NO_OFFSETS } = require('../src/main/gpuClockOffset');
const { encode } = require('../src/shared/miner/stratum');
const {
  PROFILE, shareBound, profileForK, penalizedAdjustmentFactor, difficultyAdjustmentFactor,
} = require('../src/shared/miner/pearlhash');
const { hitFor: genuine } = require('./helpers/pearlHit');

const { TURN_MS, CHECK_HITS, CHECK_MS } = coreOffsetTrial;

const ADDR = 'prl1px5ervx6ftaegmdhqa5ajemh20j2uw7l9jt5j5s97rljp72yt3s8qncrxud';
const HEADER = '000000203a49fea8b6d42c60c543fe0f029749787679372495e5d2d1007e29e3'
  + '25e1c08065a0cedc057ba091aabd10476017b3d3f4e38eafb59707ded57f71e1feb7b5d155918b6a0dea0018';
// m and n small enough to build real trees here; k and rank are mainnet's.
const SMALL = { ...PROFILE, m: 64, n: 64 };
const MAX = (1n << 256n) - 1n;
// A pool target whose bound every hash meets, so a genuine hit is the first region.
const T = MAX / BigInt(penalizedAdjustmentFactor(SMALL));
const EASY = T.toString(16).padStart(64, '0');
const JOB = '00000000_2097152';
const AUTH = '{"id":1,"result":true,"error":null}\n';
const GOOD = genuine(SMALL, Buffer.from(HEADER, 'hex'), { bound: shareBound(T, SMALL), jobId: JOB });
const WRONG = genuine(SMALL, Buffer.from(HEADER, 'hex'), {
  bound: shareBound(T, SMALL), jobId: JOB, tamper: (jp) => { jp[5] ^= 1; },
});
const MISMATCH = 'jackpot hash mismatch (fold/transcript)';
// The header of a check's job, and a genuine hit for it.
const CHECK_HEADER = Buffer.alloc(76, 0x5a);
const realCheckHit = (jobId) => genuine(SMALL, CHECK_HEADER, { jobId });

const settings = { address: ADDR, worker: 'rig01', endpoint: 'us.pearl.herominers.com:1200', profile: SMALL };
const GPU = [{ index: 0, name: 'NVIDIA GeForce RTX 4090' }];
const TWO = [GPU[0], { index: 1, name: 'NVIDIA GeForce RTX 4090' }];
// rig01's RTX 4090 at 450 W, by offset (tuning/benchmark.md).
const RIG01 = { 0: 317.4, 50: 320.9, 100: 325.3, 150: 331.2 };

const START = (index = 0, steps = '+50, +100, +150', tail = '') => 'core clock offset trial on GPU ' + index
  + ': mining 75 s at each of 0, ' + steps + ' MHz and 0 again' + tail + '; each step first has 100 of its hits '
  + 'recomputed, and the fastest step that is right is kept';
const KEPT_150 = 'core clock offset kept at +150 MHz on GPU 0 while mining (+4.3%): 317.4 TH/s at 0, '
  + '320.9 at +50 MHz, 325.3 at +100 MHz, 331.2 at +150 MHz';
const NOT_RUN = (why) => 'core clock offset trial not run on GPU 0' + why + '. Mining continues at the card\'s own clock.';
const ENDED = 'core clock offset ended on GPU 0: the card stopped mining, and it gets none for the rest of this run';
const BACK = 'core clock offset back to 0 on GPU 0';
const RIGHTS = ': setting a clock offset needs root or administrator rights';
const DENIED = { ok: false, error: 'setting the clock offset: Insufficient Permissions', noRights: true };

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

// A rig whose cards answer NVML as rig01's RTX 4090 does: offset 0, allowed
// -1000 to 1000, and every set taken. `over.read(index)` and
// `over.set(index, mhz)` answer instead when they return something;
// `over.core(core, opts)` dresses each core as it is made.
function rig(over = {}) {
  const calls = [];
  const t = { now: 0 };
  const offsets = {
    readCoreOffset: jest.fn((index) => {
      calls.push(['read', index]);
      return (over.read && over.read(index)) || { ok: true, error: null, mhz: 0, min: -1000, max: 1000 };
    }),
    setCoreOffset: jest.fn((index, mhz) => {
      calls.push(['set', index, mhz]);
      return (over.set && over.set(index, mhz)) || { ok: true, error: null };
    }),
  };
  const clocks = {
    lockMemoryClock: jest.fn((index, mhz) => { calls.push(['lock', index, mhz]); return over.lock || { ok: true, error: null }; }),
    resetMemoryClock: jest.fn((index) => { calls.push(['reset', index]); return { ok: true, error: null }; }),
  };
  const made = [];
  const createCore = jest.fn((profile, opts) => {
    const c = new EventEmitter();
    c.setJob = jest.fn();
    c.stop = jest.fn();
    if (!over.silentDevice) c.device = { index: opts.deviceIndex, name: 'NVIDIA GeForce RTX 4090' };
    if (over.core) over.core(c, opts);
    made.push(c);
    return c;
  });
  if (over.clockOffsets) createCore.clockOffsets = over.clockOffsets;
  const b = { t, calls, made, createCore, offsets, clocks, sock: null };
  // A socket for each connection, as net.connect gives: b.sock is the latest.
  const connect = () => { b.sock = makeSocket(); return b.sock; };
  const m = new PearlMiner({
    connect, createCore: over.wrap ? over.wrap(createCore) : createCore, reconnectMs: 0, restartMs: 10, clocks, now: () => t.now,
    offsets: over.injectNone ? undefined : offsets,
  });
  const logs = [];
  m.on('log', (l) => logs.push(l));
  m.on('stopped', () => calls.push(['stopped']));
  m.on('error', () => {});
  Object.assign(b, { m, logs });
  // Log in and take a job, on the connection that is open now.
  b.login = (job = {}) => {
    b.sock.emit('connect');
    b.sock.emit('data', AUTH);
    b.sock.emit('data', jobLine(job));
    return b;
  };
  // Start mining with GPU 0 on trial, logged in and on a job.
  b.start = (extra = {}, job = {}) => {
    m.start({ ...settings, gpus: GPU, mineCoreOffsetTrial: [0], ...extra });
    return b.login(job);
  };
  b.lines = (level) => logs
    .filter((l) => /core clock offset|a hit came out wrong/.test(l.line) && (!level || l.level === level))
    .map((l) => l.line);
  // The offsets GPU `index` was set to, in order.
  b.sets = (index = 0) => calls.filter((c) => c[0] === 'set' && c[1] === index).map((c) => c[2]);
  b.submits = () => b.sock.written.map((w) => JSON.parse(w)).filter((x) => x.method === 'mining.submit');
  return b;
}

// The ids of the jobs a core was handed, in order.
const jobs = (core) => core.setJob.mock.calls.map((c) => c[0].jobId);
// The check's job a core is on, or null when the last job it was handed is the pool's.
function checkJob(core) {
  const { calls } = core.setJob.mock;
  const job = calls.length ? calls[calls.length - 1][0] : null;
  return job && String(job.jobId).startsWith('offset-check-') ? job : null;
}
// A stand-in hit of a check's job: right, or wrong with `wrong` as the reason.
const checkHitOf = (job, wrong) => ({ fake: true, jobId: job.jobId, jackpotHash: Buffer.alloc(32), wrong });

// One hashrate sample from `core`, half a second on.
function tick(b, core, th) {
  b.t.now += 500;
  core.emit('hashrate', th);
}

// Mine GPU `index`'s trial through: `core` reports `rates[mhz]` TH/s twice a
// second while its card is at offset `mhz` (`rates.close` in the last turn at
// 0, when given), and answers each step's check with 100 right hits, or with
// what `checks[mhz]` says:
//   { wrong: 'why', after: n }  n right hits, then a wrong one
//   { hits: n }                 n right hits, and no more
// Returns when the trial is over, or when `until()` says so.
function mine(b, core, rates, { checks = {}, index = 0, until = () => false } = {}) {
  const answered = new Set();
  let above = false;
  for (let i = 0; i < 4000; i += 1) {
    if (!b.m.coreOffsetTrials.has(index) || until()) return;
    const at = b.m.coreOffsetSet.get(index) || 0;
    above = above || at > 0;
    tick(b, core, at === 0 && above && rates.close != null ? rates.close : rates[at]);
    const job = checkJob(core);
    if (job && !answered.has(job.jobId)) {
      answered.add(job.jobId);
      const plan = checks[b.m.coreOffsetSet.get(index)] || {};
      const right = plan.after != null ? plan.after : (plan.hits != null ? plan.hits : CHECK_HITS);
      for (let n = 0; n < right; n += 1) core.emit('hit', checkHitOf(job));
      if (plan.wrong) core.emit('hit', checkHitOf(job, plan.wrong));
    }
  }
  throw new Error('the trial did not end');
}
// Until the card is in the hit check of step `mhz`, which is left unanswered.
const untilCheck = (b, core, mhz) => mine(b, core, RIG01, {
  checks: { [mhz]: { hits: 0 } }, until: () => !!checkJob(core),
});
// Until the card is `secs` into its turn at step `mhz`.
function untilTurn(b, core, mhz, secs = 20) {
  mine(b, core, RIG01, { until: () => b.m.coreOffsetSet.get(0) === mhz && !checkJob(core) });
  for (let i = 0; i < secs * 2; i += 1) tick(b, core, RIG01[mhz]);
}

describe('PearlMiner — the core clock offset trial', () => {
  describe('when a card opens', () => {
    // Off unless asked for: with no card in the plan, NVML is never called.
    test('a rig that did not ask is not touched', () => {
      for (const extra of [{ mineCoreOffsetTrial: [] }, { mineCoreOffsetTrial: undefined }]) {
        const b = rig().start(extra);
        for (let i = 0; i < 400; i += 1) tick(b, b.made[0], 317.4);
        b.made[0].emit('hit', GOOD);
        b.m.stop();
        expect(b.calls).toEqual([['stopped']]);
        expect(b.lines()).toEqual([]);
        expect(b.submits()).toHaveLength(1);
      }
    });

    // The set to 0 is where the trial starts from, and where a rig without the
    // rights hears so.
    test('a card on trial is read, set to offset 0, and says what it will do', () => {
      const b = rig().start();
      expect(b.calls).toEqual([['read', 0], ['set', 0, 0]]);
      expect(b.logs).toContainEqual({ level: 'info', line: START() });
      b.m.stop();
      // Stopped in its first turn, at offset 0: there is nothing to put back.
      expect(b.calls.slice(2)).toEqual([['stopped']]);
    });

    test('a card the plan does not name is left alone beside one it does', () => {
      const b = rig().start({ gpus: TWO, mineCoreOffsetTrial: [1] });
      expect(b.calls).toEqual([['read', 1], ['set', 1, 0]]);
      expect(b.lines()).toEqual([START(1)]);
      b.m.stop();
    });

    // The AMD core, or a pearl_core.node built before the offset calls: the
    // miner then has main/gpuClockOffset's NO_OFFSETS.
    test('a core without the offset calls gets no trial, with a warning', () => {
      const b = rig({ injectNone: true });
      expect(b.m.offsets).toBe(NO_OFFSETS);
      b.start();
      expect(b.logs.filter((l) => /core clock offset/.test(l.line)))
        .toEqual([{ level: 'warn', line: NOT_RUN(' (not supported by this core)') }]);
      mine(b, b.made[0], RIG01);
      expect(b.lines()).toHaveLength(1);
      b.m.stop();
    });

    // main/pearlCore hangs the addon's calls on the createCore it returns.
    test('the offset calls come with createCore unless they are injected', () => {
      const clockOffsets = {
        readCoreOffset: jest.fn(() => ({ ok: true, error: null, mhz: 0, min: -500, max: 500 })),
        setCoreOffset: jest.fn(() => ({ ok: true, error: null })),
      };
      const b = rig({ injectNone: true, clockOffsets });
      expect(b.m.offsets).toBe(clockOffsets);
      b.start();
      expect(clockOffsets.readCoreOffset).toHaveBeenCalledWith(0);
      expect(clockOffsets.setCoreOffset).toHaveBeenCalledWith(0, 0);
      b.m.stop();
      expect(new PearlMiner({}).offsets).toBe(NO_OFFSETS);
      expect(rig({ clockOffsets }).m.offsets).not.toBe(clockOffsets);
    });

    test('a card NVML will not read gets no trial, and the warning says what NVML said', () => {
      const b = rig({ read: () => ({ ok: false, error: 'reading the clock offset: Not Supported', noRights: false }) }).start();
      expect(b.lines('warn')).toEqual([NOT_RUN(' (reading the clock offset: Not Supported)')]);
      expect(b.calls).toEqual([['read', 0]]);
      b.m.stop();

      const denied = rig({ read: () => ({ ok: false, error: 'nvmlInit: Insufficient Permissions', noRights: true }) }).start();
      expect(denied.lines('warn')).toEqual([NOT_RUN(' (nvmlInit: Insufficient Permissions)' + RIGHTS)]);
      denied.m.stop();
    });

    // The common case on a rig that asked for the trial without the rights.
    test('without the rights to set an offset there is no trial, and one warning that says so', () => {
      const b = rig({ set: () => DENIED }).start();
      expect(b.calls).toEqual([['read', 0], ['set', 0, 0]]);
      expect(b.logs.filter((l) => /core clock offset/.test(l.line))).toEqual([{
        level: 'warn',
        line: NOT_RUN(' (setting the clock offset: Insufficient Permissions)' + RIGHTS),
      }]);
      for (let i = 0; i < 400; i += 1) tick(b, b.made[0], 317.4);
      b.m.stop();
      expect(b.calls.slice(2)).toEqual([['stopped']]);
      expect(b.lines()).toHaveLength(1);
    });

    test('a set NVML refuses for another reason is not put down to rights', () => {
      const b = rig({ set: () => ({ ok: false, error: 'setting the clock offset: Not Supported', noRights: false }) }).start();
      expect(b.lines('warn')).toEqual([NOT_RUN(' (setting the clock offset: Not Supported)')]);
      b.m.stop();
    });

    test('a card whose driver allows less than the first step gets no trial', () => {
      const b = rig({ read: () => ({ ok: true, error: null, mhz: 0, min: -30, max: 30 }) }).start();
      expect(b.lines('warn')).toEqual([NOT_RUN(': its driver allows at most +30 MHz, less than the first step of +50 MHz')]);
      expect(b.calls).toEqual([['read', 0]]);
      b.m.stop();
    });

    test('a card whose driver allows +100 MHz climbs to +100 and no further', () => {
      const b = rig({ read: () => ({ ok: true, error: null, mhz: 0, min: -100, max: 100 }) }).start();
      expect(b.lines()).toEqual([START(0, '+50, +100')]);
      mine(b, b.made[0], RIG01);
      expect(b.sets()).toEqual([0, 50, 100, 0, 100]);
      expect(b.lines().pop()).toBe('core clock offset kept at +100 MHz on GPU 0 while mining (+2.5%): '
        + '317.4 TH/s at 0, 320.9 at +50 MHz, 325.3 at +100 MHz');
      b.m.stop();
    });

    // A run killed hard cannot put its offset back, and another tool may have
    // set one. Either way the trial's turns at 0 have to be at 0.
    //
    // It is not put back afterwards: nothing says which of the two it was, and
    // an offset a killed run left has nothing checking what the card computes.
    // So the line is a warning, and says the offset is gone for good.
    test('a card found with an offset already set starts from 0, and the log warns that it is replaced', () => {
      const b = rig({ read: () => ({ ok: true, error: null, mhz: 150, min: -1000, max: 1000 }) }).start();
      expect(b.calls).toEqual([['read', 0], ['set', 0, 0]]);
      const HAD = 'GPU 0 had a core clock offset of +150 MHz, left by an earlier run or set by another tool. '
        + 'The trial replaces it: the card starts from 0, mines at what the trial keeps, and is at 0 when mining '
        + 'stops. The +150 MHz is not put back; set it again afterwards if it was yours.';
      expect(b.lines()).toEqual([HAD, START()]);
      expect(b.logs.find((l) => l.line === HAD).level).toBe('warn');
      // Nothing kept, and stopped: the card is still at 0, not at what it had.
      mine(b, b.made[0], { 0: 317.4, 50: 317.5, 100: 317.6, 150: 317.7 });
      b.m.stop();
      expect(b.sets().pop()).toBe(0);
      const under = rig({ read: () => ({ ok: true, error: null, mhz: -75, min: -1000, max: 1000 }) }).start();
      expect(under.lines()[0]).toMatch(/^GPU 0 had a core clock offset of -75 MHz, /);
      under.m.stop();
    });

    test('a core that does not name its card gets no trial', () => {
      const b = rig({ silentDevice: true }).start();
      expect(b.logs.filter((l) => /core clock offset/.test(l.line))).toEqual([{
        level: 'warn',
        line: 'core clock offset not tried: this pearl_core.node does not say which GPU it opened',
      }]);
      expect(b.calls).toEqual([]);
      b.m.stop();
    });

    // A hit check is about half a second of the miner's event loop, and every
    // card of a rig would otherwise check at the same moment.
    test('each card starts 15 seconds after the one before it', () => {
      const b = rig().start({ gpus: TWO, mineCoreOffsetTrial: [0, 1] });
      const firstSet = (index) => {
        const at = { 0: null, 1: null };
        for (let i = 0; i < 400 && at[index] == null; i += 1) {
          b.t.now += 500;
          for (const c of b.made) c.emit('hashrate', 317.4);
          for (const i2 of [0, 1]) if (at[i2] == null && b.sets(i2).includes(50)) at[i2] = b.t.now;
        }
        return at;
      };
      const at = firstSet(1);
      // The first card's trial starts on its first sample, the second's 15 s on.
      expect(at).toEqual({ 0: 500 + TURN_MS, 1: 500 + 15000 + TURN_MS });
      b.m.stop();
    });

    // Counted from when nothing else is in a card's way, not from when it
    // opened: memory clock trials on every card end on the same sample.
    test('cards whose memory clock trials end together still start 15 seconds apart', () => {
      const b = rig().start({
        gpus: TWO, mineCoreOffsetTrial: [0, 1], mineMemClockByIndex: { 0: 5001, 1: 5001 }, mineMemClockTrial: [0, 1],
      });
      const at = { 0: null, 1: null };
      for (let i = 0; i < 2000 && at[1] == null; i += 1) {
        b.t.now += 500;
        for (const c of b.made) c.emit('hashrate', 317.4);
        for (const index of [0, 1]) if (at[index] == null && b.sets(index).includes(50)) at[index] = b.t.now;
      }
      expect(b.m.memClockTrials.size).toBe(0);
      expect(at).toEqual({ 0: 4 * MEM_TURN_MS + TURN_MS, 1: 4 * MEM_TURN_MS + 15000 + TURN_MS });
      b.m.stop();
    });

    // An RTX 30 card may time its launch shapes when it opens
    // (shared/shapePick), a new shape every few seconds. The shells'
    // createCore says whether a core still is, from the core's 'shape' event.
    describe('a card timing its launch shapes', () => {
      const SHAPES = ['flip 256', 'band walk 8'];
      const shaped = (timingShapes) => {
        const b = rig({ core: (c) => { c.shapes = SHAPES; } });
        b.createCore.timingShapes = timingShapes;
        return b;
      };

      // The shape trial has no fixed length: each new job restarts its turn.
      test('waits for the shape trial however long it runs, and starts on the next sample', () => {
        let timing = true;
        const asked = [];
        const b = shaped((core) => { asked.push(core); return timing; }).start();
        const core = b.made[0];
        expect(b.lines()).toEqual([START(0, undefined, ', once its launch shapes are timed')]);
        // Well past the 180 s the miner used to guess.
        for (let i = 0; i < 800; i += 1) tick(b, core, 95.5);
        expect(b.t.now).toBe(400000);
        expect(b.m.coreOffsetTrials.get(0).trial).toBe(null);
        expect(b.sets()).toEqual([0]);
        expect(asked.every((c) => c === core)).toBe(true);
        timing = false;
        while (!b.sets().includes(50)) tick(b, core, 95.5);
        expect(b.t.now).toBe(400000 + 500 + TURN_MS);
        b.m.stop();
      });

      // A remembered shape, a forced one, trials off: nothing is timed.
      test('does not wait at all when no shape trial is under way', () => {
        const b = shaped(() => false).start();
        expect(b.lines()).toEqual([START()]);
        while (!b.sets().includes(50)) tick(b, b.made[0], 95.5);
        expect(b.t.now).toBe(500 + TURN_MS);
        b.m.stop();
      });

      // The miner and the shells' real wrapper together: the wait ends when
      // the core sends its 'shape' event, and a remembered shape means none.
      test('through shared/shapePick, the core\'s shape event is what ends the wait', () => {
        const { withShapePick } = require('../src/shared/shapePick');
        const data = {};
        const store = { get: (k) => data[k] || null, set: (k, v) => { data[k] = v; }, remove: (k) => { delete data[k]; } };
        const wrap = (create) => withShapePick(create, { store, gpus: GPU, now: () => 1, env: {} });
        const shape = (c) => { c.shapes = SHAPES; [c.shape] = SHAPES; };
        const b = rig({ core: shape, wrap }).start();
        const core = b.made[0];
        for (let i = 0; i < 800; i += 1) tick(b, core, 95.5);
        expect(b.sets()).toEqual([0]);
        core.emit('shape', JSON.stringify({ shape: SHAPES[1], rule: SHAPES[0], trial: [] }));
        while (!b.sets().includes(50)) tick(b, core, 95.5);
        expect(b.t.now).toBe(400000 + 500 + TURN_MS);
        b.m.stop();

        // The next start has the shape remembered, and its offset trial starts at once.
        const next = rig({ core: (c) => { c.shapes = SHAPES; [, c.shape] = SHAPES; }, wrap }).start();
        while (!next.sets().includes(50)) tick(next, next.made[0], 95.5);
        expect(next.t.now).toBe(500 + TURN_MS);
        next.m.stop();
      });

      test('with a memory clock trial as well, the start line names both', () => {
        const b = shaped(() => true).start({ mineMemClockByIndex: { 0: 5001 }, mineMemClockTrial: [0] });
        expect(b.lines()).toEqual([
          START(0, undefined, ', once the memory clock trial is over and its launch shapes are timed'),
        ]);
        b.m.stop();
      });
    });

    // A createCore that cannot say when the shape trial ends: the miner waits
    // the trial's 150 seconds and 30 more, as it did before it could ask.
    test('a card with launch shapes to time, and nothing to ask, starts after a fixed wait', () => {
      expect(SHAPE_TRIAL_SECS).toBe(150);
      const b = rig({ core: (c) => { c.shapes = ['flip 256', 'band walk 8']; } }).start();
      const core = b.made[0];
      while (!b.sets().includes(50)) tick(b, core, 95.5);
      // 180 s from its first sample, then the first turn.
      expect(b.t.now).toBe(500 + 180000 + TURN_MS);
      b.m.stop();
      // One shape, or none, is nothing to wait for.
      for (const shapes of [['flip 256'], [], undefined, 'flip 256']) {
        const one = rig({ core: (c) => { c.shapes = shapes; } }).start();
        while (!one.sets().includes(50)) tick(one, one.made[0], 95.5);
        expect(one.t.now).toBe(500 + TURN_MS);
        one.m.stop();
      }
    });
  });

  describe('the climb', () => {
    test('keeps the highest step when each was right and faster, says what it measured, and resets on stop', () => {
      const b = rig().start();
      const core = b.made[0];
      mine(b, core, RIG01);
      // 0 to start from, each step, 0 for the last turn, then the step kept.
      expect(b.sets()).toEqual([0, 50, 100, 150, 0, 150]);
      expect(b.lines()).toEqual([START(), KEPT_150]);
      expect(b.logs.find((l) => l.line === KEPT_150).level).toBe('info');
      // The card went to each check's job and back to the pool's.
      expect(jobs(core)).toEqual([JOB, 'offset-check-1', JOB, 'offset-check-2', JOB, 'offset-check-3', JOB]);
      // None of the 300 hits it checked went to the pool.
      expect(b.submits()).toEqual([]);
      // The trial is over: later samples change nothing.
      for (let i = 0; i < 400; i += 1) tick(b, core, 100);
      expect(b.sets()).toHaveLength(6);

      b.m.stop();
      expect(b.calls.slice(-2)).toEqual([['set', 0, 0], ['stopped']]);
      expect(b.lines().pop()).toBe(BACK);
    });

    test('keeps a lower step when the one above it was not faster', () => {
      const b = rig().start();
      mine(b, b.made[0], { ...RIG01, 150: 325.4 });
      expect(b.sets()).toEqual([0, 50, 100, 150, 0, 100]);
      expect(b.lines().pop()).toBe('core clock offset kept at +100 MHz on GPU 0 while mining (+2.5%): '
        + '317.4 TH/s at 0, 320.9 at +50 MHz, 325.3 at +100 MHz, 325.4 at +150 MHz');
      b.m.stop();
      expect(b.calls.slice(-2)).toEqual([['set', 0, 0], ['stopped']]);

      const first = rig().start();
      mine(first, first.made[0], { ...RIG01, 100: 321.0, 150: 321.1 });
      expect(first.sets()).toEqual([0, 50, 100, 150, 0, 50]);
      expect(first.lines().pop()).toBe('core clock offset kept at +50 MHz on GPU 0 while mining (+1.1%): '
        + '317.4 TH/s at 0, 320.9 at +50 MHz, 321.0 at +100 MHz, 321.1 at +150 MHz');
      first.m.stop();
    });

    test('leaves the card at 0 when no step was faster, and has nothing to put back on stop', () => {
      const b = rig().start();
      mine(b, b.made[0], { 0: 317.4, 50: 317.5, 100: 317.6, 150: 317.7 });
      expect(b.sets()).toEqual([0, 50, 100, 150, 0]);
      expect(b.lines().pop()).toBe('core clock offset left at 0 on GPU 0 (no step was faster by 0.3%): '
        + '317.4 TH/s at 0, 317.5 at +50 MHz, 317.6 at +100 MHz, 317.7 at +150 MHz');
      b.m.stop();
      expect(b.calls.slice(-2)).toEqual([['set', 0, 0], ['stopped']]);
      expect(b.sets()).toHaveLength(5);
    });

    test('with too few samples to compare, leaves the card at 0 and quotes no rates', () => {
      const b = rig().start();
      const core = b.made[0];
      tick(b, core, 317.4);
      // One sample a turn: the one that ends it. Each check gets its hits.
      for (let turn = 0; turn < 5; turn += 1) {
        b.t.now += TURN_MS;
        core.emit('hashrate', 317.4);
        const job = checkJob(core);
        for (let n = 0; job && n < CHECK_HITS; n += 1) core.emit('hit', checkHitOf(job));
      }
      expect(b.lines().pop()).toBe('core clock offset left at 0 on GPU 0 (too few hashrate samples to compare)');
      expect(b.sets()).toEqual([0, 50, 100, 150, 0]);
      b.m.stop();
    });

    test('a step with too few samples is quoted as no reading', () => {
      const b = rig().start();
      const core = b.made[0];
      mine(b, core, RIG01, { until: () => b.m.coreOffsetSet.get(0) === 100 });
      // The +100 turn: nothing but the sample that ends it.
      b.t.now += TURN_MS;
      core.emit('hashrate', 325.3);
      mine(b, core, RIG01);
      expect(b.lines().pop()).toBe('core clock offset kept at +50 MHz on GPU 0 while mining (+1.1%): '
        + '317.4 TH/s at 0, 320.9 at +50 MHz, no reading at +100 MHz, 331.2 at +150 MHz');
      b.m.stop();
    });
  });

  describe('a step\'s hit check', () => {
    // The check's job: a header of the miner's own, and a target set from the
    // card's rate so it hits about 30 times a second.
    test('hands the card a job of the miner\'s own, with a target easy for its rate', () => {
      const b = rig().start();
      const core = b.made[0];
      untilCheck(b, core, 50);
      expect(b.sets()).toEqual([0, 50]);
      const job = core.setJob.mock.calls[1][0];
      expect(job.jobId).toBe('offset-check-1');
      expect(job.header.equals(Buffer.alloc(76, 0x5a))).toBe(true);
      expect(job.target).toBe(coreOffsetTrial.checkBound(317.4, difficultyAdjustmentFactor(SMALL)));
      b.m.stop();
    });

    // End to end, with genuine hits recomputed for real. The check's bound is
    // opened up so the first region of the small test profile meets it.
    test('passes on 100 genuine hits, each recomputed', () => {
      const spy = jest.spyOn(coreOffsetTrial, 'checkBound').mockReturnValue(MAX);
      try {
        const b = rig().start();
        const core = b.made[0];
        untilCheck(b, core, 50);
        const hit = realCheckHit('offset-check-1');
        for (let n = 0; n < CHECK_HITS - 1; n += 1) core.emit('hit', hit);
        expect(checkJob(core)).not.toBe(null);
        core.emit('hit', hit);
        // Passed: back on the pool's job, still at +50, and nothing sent.
        expect(jobs(core)).toEqual([JOB, 'offset-check-1', JOB]);
        expect(b.sets()).toEqual([0, 50]);
        expect(b.submits()).toEqual([]);
        b.m.stop();
      } finally {
        spy.mockRestore();
      }
    });

    test('one genuine hit that recomputes wrong fails the step', () => {
      const spy = jest.spyOn(coreOffsetTrial, 'checkBound').mockReturnValue(MAX);
      try {
        const b = rig().start();
        const core = b.made[0];
        untilCheck(b, core, 50);
        core.emit('hit', realCheckHit('offset-check-1'));
        core.emit('hit', genuine(SMALL, CHECK_HEADER, { jobId: 'offset-check-1', tamper: (jp) => { jp[0] ^= 4; } }));
        expect(b.sets()).toEqual([0, 50, 0]);
        expect(jobs(core)).toEqual([JOB, 'offset-check-1', JOB]);
        mine(b, core, RIG01);
        expect(b.lines().pop()).toBe('core clock offset left at 0 on GPU 0 (no step passed its hit check): '
          + '317.4 TH/s at 0; +50 MHz failed its hit check (' + MISMATCH + ')');
        expect(b.submits()).toEqual([]);
        b.m.stop();
      } finally {
        spy.mockRestore();
      }
    });

    // The device reports a hit only under the job's target, so one above it is
    // a card that compares wrong.
    test('a hit whose hash is above the check\'s target fails the step', () => {
      const b = rig().start();
      const core = b.made[0];
      untilCheck(b, core, 50);
      core.emit('hit', realCheckHit('offset-check-1'));
      expect(b.sets()).toEqual([0, 50, 0]);
      mine(b, core, RIG01);
      expect(b.lines().pop()).toMatch(/; \+50 MHz failed its hit check \(the hash does not meet the check's target\)$/);
      b.m.stop();
    });

    test('a failed step ends the climb, and the card keeps the last step that passed', () => {
      const b = rig().start();
      const core = b.made[0];
      mine(b, core, RIG01, { checks: { 150: { wrong: MISMATCH, after: 57 } } });
      // +150 is set, fails, and is gone at once: no turn at it.
      expect(b.sets()).toEqual([0, 50, 100, 150, 0, 100]);
      expect(b.lines().pop()).toBe('core clock offset kept at +100 MHz on GPU 0 while mining (+2.5%): '
        + '317.4 TH/s at 0, 320.9 at +50 MHz, 325.3 at +100 MHz; +150 MHz failed its hit check (' + MISMATCH + ')');
      expect(jobs(core)).toEqual([JOB, 'offset-check-1', JOB, 'offset-check-2', JOB, 'offset-check-3', JOB]);
      expect(b.submits()).toEqual([]);
      b.m.stop();
    });

    // The trial's start line promises 100 hits recomputed, so 99 is not a pass.
    test('ends after 10 seconds, failed for want of hits, with fewer than the 100 it needs', () => {
      const few = rig().start();
      const core = few.made[0];
      untilCheck(few, core, 50);
      for (let n = 0; n < CHECK_HITS - 1; n += 1) core.emit('hit', checkHitOf(checkJob(core)));
      const began = few.t.now;
      while (checkJob(core)) tick(few, core, 320.9);
      expect(few.t.now - began).toBe(CHECK_MS);
      expect(few.sets()).toEqual([0, 50, 0]);
      mine(few, core, RIG01);
      expect(few.lines().pop()).toMatch(/; \+50 MHz failed its hit check \(only 99 of the 100 hits it needs in 10 s\)$/);
      few.m.stop();
    });

    // The core finishes the batches it had queued after its job changes, so a
    // few of a check's hits arrive once the check is over.
    test('hits that arrive after their check is over are dropped, right or wrong', () => {
      const b = rig().start();
      const core = b.made[0];
      mine(b, core, RIG01, { until: () => b.m.coreOffsetSet.get(0) === 50 });
      const before = b.logs.length;
      core.emit('hit', checkHitOf({ jobId: 'offset-check-1' }));
      core.emit('hit', checkHitOf({ jobId: 'offset-check-1' }, MISMATCH));
      expect(b.sets()).toEqual([0, 50]);
      // Nor do they count for the next step's check.
      untilCheck(b, core, 100);
      for (let n = 0; n < CHECK_HITS; n += 1) core.emit('hit', checkHitOf({ jobId: 'offset-check-1' }));
      expect(checkJob(core).jobId).toBe('offset-check-2');
      expect(b.logs.slice(before).map((l) => l.line)).toEqual([]);
      expect(b.submits()).toEqual([]);
      b.m.stop();
      // And after the run's trial is over, or from a card that never had one.
      core.emit('hit', checkHitOf({ jobId: 'offset-check-2' }));
      const none = rig().start({ mineCoreOffsetTrial: [] });
      none.made[0].emit('hit', checkHitOf({ jobId: 'offset-check-1' }));
      expect(none.submits()).toEqual([]);
      expect(none.logs.map((l) => l.line).filter((l) => /hit/.test(l))).toEqual([]);
      none.m.stop();
      const silent = rig({ silentDevice: true }).start();
      silent.made[0].emit('hit', checkHitOf({ jobId: 'offset-check-1' }));
      expect(silent.submits()).toEqual([]);
      silent.m.stop();
    });

    // No path leaves a trial behind a closed core (the tests of a core that
    // closes are below), so this makes one by hand.
    test('a trial whose card is somehow no longer mining ends at offset 0 rather than throw', () => {
      const b = rig().start();
      const core = b.made[0];
      untilCheck(b, core, 50);
      const job = checkJob(core);
      b.m.cores.length = 0;
      for (let n = 0; n < CHECK_HITS; n += 1) core.emit('hit', checkHitOf(job));
      expect(b.sets()).toEqual([0, 50, 0]);
      expect(b.m.coreOffsetTrials.size).toBe(0);
      expect(b.m.coreOffsetVerdicts.get(0)).toBe(0);
      b.m.stop();
    });

    test('a job the pool sends during a check is not lost: the card gets it when the check ends', () => {
      const b = rig().start({ gpus: TWO, mineCoreOffsetTrial: [0] });
      const [core, other] = b.made;
      untilCheck(b, core, 50);
      b.sock.emit('data', jobLine({ job_id: 'during_check' }));
      // The other card gets it at once; the card in its check keeps the check's job.
      expect(jobs(other)).toEqual([JOB, 'during_check']);
      expect(jobs(core)).toEqual([JOB, 'offset-check-1']);
      for (let n = 0; n < CHECK_HITS; n += 1) core.emit('hit', checkHitOf(checkJob(core)));
      expect(jobs(core)).toEqual([JOB, 'offset-check-1', 'during_check']);
      b.m.stop();
    });

    // Found for the pool's job just before the check's job took over.
    test('a real hit that arrives during a check is still recomputed and sent', () => {
      const b = rig().start();
      const core = b.made[0];
      untilCheck(b, core, 50);
      core.emit('hit', GOOD);
      expect(b.submits()).toHaveLength(1);
      expect(b.submits()[0].params.job_id).toBe(JOB);
      expect(checkJob(core)).not.toBe(null);
      b.m.stop();
    });

    // The existing handling of a hit for a job the pool has replaced.
    test('a hit for the job a new one replaced is still dropped and said, during a check and after', () => {
      const b = rig().start();
      const core = b.made[0];
      untilCheck(b, core, 50);
      b.sock.emit('data', jobLine({ job_id: 'next' }));
      core.emit('hit', GOOD);
      for (let n = 0; n < CHECK_HITS; n += 1) core.emit('hit', checkHitOf(checkJob(core)));
      core.emit('hit', GOOD);
      const dropped = b.logs.map((l) => l.line).filter((l) => /dropping a hit for the job replaced/.test(l));
      expect(dropped).toHaveLength(2);
      expect(b.submits()).toEqual([]);
      b.m.stop();
    });

    // The cores keep the job they have through a pool reconnect. A card whose
    // check ends meanwhile stays on the check's job, whose hits go nowhere,
    // until the pool sends one.
    test('a check that ends while the pool connection is down leaves the card on its job until a job comes', () => {
      jest.useFakeTimers();
      try {
        const b = rig().start();
        const core = b.made[0];
        untilCheck(b, core, 50);
        const job = checkJob(core);
        b.sock.emit('close');
        for (let n = 0; n < CHECK_HITS + 5; n += 1) core.emit('hit', checkHitOf(job));
        expect(jobs(core)).toEqual([JOB, 'offset-check-1']);
        expect(b.m.coreOffsetTrials.get(0).check).toBe(null);
        expect(b.logs.map((l) => l.line)).not.toContain('dropping a hit found while the pool connection is down');
        jest.runOnlyPendingTimers();
        b.login({ job_id: 'after_reconnect' });
        expect(jobs(core)).toEqual([JOB, 'offset-check-1', 'after_reconnect']);
        b.m.stop();
      } finally {
        jest.clearAllTimers();
        jest.useRealTimers();
      }
    });
  });

  describe('when NVML refuses mid-trial', () => {
    const refuse = (mhz, answer) => ({ set: (index, to) => (to === mhz ? answer : null) });
    const NOT_SUPPORTED = { ok: false, error: 'setting the clock offset: Not Supported', noRights: false };

    test('a first step that cannot be set ends the trial with a warning, at offset 0', () => {
      const b = rig(refuse(50, NOT_SUPPORTED)).start();
      mine(b, b.made[0], RIG01);
      expect(b.sets()).toEqual([0, 50]);
      expect(b.logs.filter((l) => l.level === 'warn')).toEqual([{
        level: 'warn',
        line: 'core clock offset trial ended on GPU 0 (+50 MHz: setting the clock offset: Not Supported). '
          + 'Mining continues at the card\'s own clock.',
      }]);
      expect(jobs(b.made[0])).toEqual([JOB]);
      b.m.stop();
      expect(b.calls.slice(-1)).toEqual([['stopped']]);
      expect(b.sets()).toHaveLength(2);
    });

    test('refused for want of rights, the warning says which', () => {
      const b = rig(refuse(50, DENIED)).start();
      mine(b, b.made[0], RIG01);
      expect(b.lines('warn')).toEqual(['core clock offset trial ended on GPU 0 (+50 MHz: setting the clock offset: '
        + 'Insufficient Permissions)' + RIGHTS + '. Mining continues at the card\'s own clock.']);
      b.m.stop();
    });

    test('a later step that cannot be set takes the card back to 0 and ends the trial', () => {
      const b = rig(refuse(100, NOT_SUPPORTED)).start();
      mine(b, b.made[0], RIG01);
      expect(b.sets()).toEqual([0, 50, 100, 0]);
      expect(b.lines('warn')).toHaveLength(1);
      expect(b.lines().filter((l) => /core clock offset (kept|left)/.test(l))).toEqual([]);
      // Not tried again, and nothing left to put back.
      for (let i = 0; i < 400; i += 1) tick(b, b.made[0], 317.4);
      b.m.stop();
      expect(b.sets()).toHaveLength(4);
    });

    // The last turn is at 0, so the step kept has to be set again.
    test('a kept step that cannot be set again leaves the card at 0', () => {
      let n = 0;
      const b = rig({ set: (index, to) => (to === 150 && (n += 1) === 2 ? NOT_SUPPORTED : null) }).start();
      mine(b, b.made[0], RIG01);
      expect(b.sets()).toEqual([0, 50, 100, 150, 0, 150]);
      expect(b.lines('warn')).toEqual(['core clock offset trial ended on GPU 0 (+150 MHz: setting the clock offset: '
        + 'Not Supported). Mining continues at the card\'s own clock.']);
      expect(b.m.coreOffsetVerdicts.get(0)).toBe(0);
      b.m.stop();
      expect(b.sets()).toHaveLength(6);
    });

    describe('a reset that fails', () => {
      const STUCK = (mhz) => 'could not set the core clock offset back to 0 on GPU 0 (setting the clock offset: Unknown '
        + 'Error). The card stays at ' + mhz + ' until the PC restarts or the NVIDIA driver reloads, or until mining '
        + 'next starts with the offset trial on, with the rights to set it.';
      const UNKNOWN = { ok: false, error: 'setting the clock offset: Unknown Error', noRights: false };
      // Every set to 0 after the one the trial starts with.
      const noReset = () => {
        let zeros = 0;
        return { set: (index, to) => (to === 0 && (zeros += 1) > 1 ? UNKNOWN : null) };
      };

      test('for the last turn ends the trial, and the error names what is left to do', () => {
        const b = rig(noReset()).start();
        mine(b, b.made[0], RIG01);
        expect(b.sets()).toEqual([0, 50, 100, 150, 0]);
        expect(b.logs.filter((l) => l.level === 'error')).toEqual([{ level: 'error', line: STUCK('+150 MHz') }]);
        expect(b.lines().filter((l) => /core clock offset (kept|left)/.test(l))).toEqual([]);
        // Its hits are still recomputed: the card may still be at +150.
        b.made[0].emit('hit', GOOD);
        expect(b.submits()).toHaveLength(1);
        b.m.stop();
        // Said once: stop does not try again.
        expect(b.sets()).toHaveLength(5);
        expect(b.calls.slice(-1)).toEqual([['stopped']]);
      });

      test('after a failed check still hands the card back the pool\'s job', () => {
        const b = rig(noReset()).start();
        const core = b.made[0];
        mine(b, core, RIG01, { checks: { 50: { wrong: MISMATCH, after: 3 } } });
        expect(b.sets()).toEqual([0, 50, 0]);
        expect(jobs(core)).toEqual([JOB, 'offset-check-1', JOB]);
        expect(b.lines('error')).toEqual([STUCK('+50 MHz')]);
        b.m.stop();
      });

      test('on stop is said, and the miner still stops', () => {
        const b = rig(noReset()).start();
        untilTurn(b, b.made[0], 100);
        b.m.stop();
        expect(b.lines('error')).toEqual([STUCK('+100 MHz')]);
        expect(b.calls.slice(-2)).toEqual([['set', 0, 0], ['stopped']]);
        expect(b.m.isRunning()).toBe(false);
      });
    });
  });

  // An offset is only kept while what the card computes is checked.
  describe('the hits of a card above its own clock', () => {
    const WRONG_LINE = 'GPU 0 (NVIDIA GeForce RTX 4090): a hit came out wrong (' + MISMATCH + '). No share was sent';

    test('are recomputed before they are sent: a right one goes out', () => {
      const b = rig().start();
      mine(b, b.made[0], RIG01);
      b.made[0].emit('hit', GOOD);
      b.made[0].emit('hit', GOOD);
      expect(b.submits()).toHaveLength(2);
      expect(b.lines('error')).toEqual([]);
      b.m.stop();
    });

    test('a wrong one is not sent, and the card goes back to 0 for the rest of the run', () => {
      const b = rig().start();
      const core = b.made[0];
      mine(b, core, RIG01);
      core.emit('hit', WRONG);
      expect(b.submits()).toEqual([]);
      expect(b.sets()).toEqual([0, 50, 100, 150, 0, 150, 0]);
      expect(b.logs.filter((l) => l.level === 'error')).toEqual([{
        level: 'error',
        line: WRONG_LINE + ', and this card gets no core clock offset for the rest of this run.',
      }]);
      expect(b.m.coreOffsetVerdicts.get(0)).toBe(0);
      // The card keeps mining, and its hits are still recomputed: a batch
      // computed at the offset can report after it is gone.
      expect(b.m.cores).toHaveLength(1);
      core.emit('hit', GOOD);
      expect(b.submits()).toHaveLength(1);
      core.emit('hit', WRONG);
      expect(b.submits()).toHaveLength(1);
      expect(b.lines('error').pop()).toBe(WRONG_LINE + '.');
      // No second trial, and nothing to put back on stop.
      for (let i = 0; i < 400; i += 1) tick(b, core, 317.4);
      b.m.stop();
      expect(b.sets()).toHaveLength(7);
    });

    test('a wrong one in the middle of the trial ends the trial', () => {
      const b = rig().start();
      const core = b.made[0];
      untilTurn(b, core, 100);
      core.emit('hit', WRONG);
      expect(b.submits()).toEqual([]);
      expect(b.sets()).toEqual([0, 50, 100, 0]);
      expect(b.lines('error')).toEqual([WRONG_LINE + ', and this card gets no core clock offset for the rest of this run.']);
      for (let i = 0; i < 400; i += 1) tick(b, core, 317.4);
      expect(b.sets()).toHaveLength(4);
      b.m.stop();
    });

    // In a check: the card goes back to the pool's job too.
    test('a wrong one during a check ends the trial and the check', () => {
      const b = rig().start();
      const core = b.made[0];
      untilCheck(b, core, 100);
      core.emit('hit', WRONG);
      expect(b.sets()).toEqual([0, 50, 100, 0]);
      expect(jobs(core)).toEqual([JOB, 'offset-check-1', JOB, 'offset-check-2', JOB]);
      b.m.stop();
    });

    // Before its first step a card is at its own clock, as any card is.
    test('are not recomputed while the card has not left offset 0', () => {
      const b = rig().start();
      tick(b, b.made[0], 317.4);
      b.made[0].emit('hit', WRONG);
      expect(b.submits()).toHaveLength(1);
      expect(b.lines('error')).toEqual([]);
      b.m.stop();
    });
  });

  describe('when the card\'s core closes in the same run', () => {
    beforeEach(() => jest.useFakeTimers());
    afterEach(() => {
      jest.clearAllTimers();
      jest.useRealTimers();
    });
    const fail = (b) => {
      b.made[b.made.length - 1].emit('error', new Error('GPU 0: misaligned address'));
      jest.advanceTimersByTime(RESTART_MS);
    };

    // A card that has just faulted is no card to run above its own clock.
    test('a fault in a step\'s turn puts the card back at 0 at once, and it is not tried again', () => {
      const b = rig().start();
      untilTurn(b, b.made[0], 50);
      b.made[0].emit('error', new Error('GPU 0: misaligned address'));
      // Before the restart, not at it.
      expect(b.sets()).toEqual([0, 50, 0]);
      expect(b.lines().pop()).toBe(ENDED);
      jest.advanceTimersByTime(RESTART_MS);
      expect(b.made).toHaveLength(2);
      mine(b, b.made[1], RIG01);
      for (let i = 0; i < 400; i += 1) tick(b, b.made[1], 317.4);
      expect(b.calls.filter((c) => c[0] === 'read')).toHaveLength(1);
      expect(b.sets()).toHaveLength(3);
      expect(b.lines().filter((l) => l === START())).toHaveLength(1);
      b.m.stop();
    });

    test('a fault during a check: the old core\'s hits are dropped, and the new core gets the pool\'s job', () => {
      const b = rig().start();
      const old = b.made[0];
      untilCheck(b, old, 50);
      const job = checkJob(old);
      fail(b);
      expect(b.sets()).toEqual([0, 50, 0]);
      for (let n = 0; n < CHECK_HITS; n += 1) old.emit('hit', checkHitOf(job));
      expect(jobs(old)).toEqual([JOB, 'offset-check-1']);
      expect(jobs(b.made[1])).toEqual([JOB]);
      b.m.stop();
    });

    test('a fault with an offset kept takes it off, and the card comes back at 0', () => {
      const b = rig().start();
      mine(b, b.made[0], RIG01);
      fail(b);
      expect(b.sets()).toEqual([0, 50, 100, 150, 0, 150, 0]);
      expect(b.lines().slice(-2)).toEqual([KEPT_150, ENDED]);
      for (let i = 0; i < 400; i += 1) tick(b, b.made[1], 317.4);
      b.m.stop();
      expect(b.sets()).toHaveLength(7);
      expect(b.calls.filter((c) => c[0] === 'read')).toHaveLength(1);
    });

    test('a fault after a trial that kept nothing changes nothing, and there is no second trial', () => {
      const b = rig().start();
      mine(b, b.made[0], { 0: 317.4, 50: 317.5, 100: 317.6, 150: 317.7 });
      const before = [b.calls.length, b.lines().length];
      fail(b);
      mine(b, b.made[1], RIG01);
      for (let i = 0; i < 400; i += 1) tick(b, b.made[1], 317.4);
      expect([b.calls.length, b.lines().length]).toEqual(before);
      b.m.stop();
    });

    test('a fault in the first turn, at offset 0, ends the trial too', () => {
      const b = rig().start();
      tick(b, b.made[0], 317.4);
      fail(b);
      expect(b.sets()).toEqual([0]);
      expect(b.lines()).toEqual([START(), ENDED]);
      mine(b, b.made[1], RIG01);
      for (let i = 0; i < 400; i += 1) tick(b, b.made[1], 317.4);
      expect(b.sets()).toEqual([0]);
      b.m.stop();
    });

    test('a card that could not be tried is not asked again, and warns once', () => {
      const b = rig({ set: () => DENIED }).start();
      fail(b);
      fail(b);
      expect(b.made).toHaveLength(3);
      expect(b.calls.filter((c) => c[0] === 'read')).toHaveLength(1);
      expect(b.lines('warn')).toHaveLength(1);
      b.m.stop();
    });

    // A model must never run on a card above its own clock.
    test('a card set aside for a prompt goes back to 0, and resumes without an offset', () => {
      const b = rig().start();
      mine(b, b.made[0], RIG01);
      b.m.pauseCards([0]);
      expect(b.sets().slice(-2)).toEqual([150, 0]);
      expect(b.lines().pop()).toBe(ENDED);
      b.m.resumeCards([0]);
      expect(b.made).toHaveLength(2);
      for (let i = 0; i < 400; i += 1) tick(b, b.made[1], 317.4);
      expect(b.sets()).toHaveLength(7);
      b.m.stop();
    });

    // Cards at another k are opened again at the profile's when a pool turns
    // their shares down (K_REJECT_LIMIT). No NVIDIA card mines another k today.
    test('a card opened again at another k goes back to 0', () => {
      const K4 = profileForK(SMALL, 4096);
      // A pool target whose bound at k 4096 every hash meets.
      const T4 = MAX / BigInt(penalizedAdjustmentFactor(K4));
      const b = rig({ core: (c, opts) => { c.k = opts.cardK ? 4096 : 2048; } })
        .start({}, { target: T4.toString(16).padStart(64, '0') });
      const core = b.made[0];
      untilTurn(b, core, 50);
      const hit = genuine(K4, Buffer.from(HEADER, 'hex'), { bound: shareBound(T4, K4), jobId: JOB });
      for (let n = 0; n < 3; n += 1) core.emit('hit', hit);
      for (const s of b.submits()) {
        b.sock.emit('data', JSON.stringify({ id: s.id, result: null, error: { code: 23, message: 'invalid share' } }) + '\n');
      }
      expect(b.made).toHaveLength(2);
      expect(b.m.cores.map((c) => c.profile.k)).toEqual([2048]);
      expect(b.sets()).toEqual([0, 50, 0]);
      expect(b.lines().pop()).toBe(ENDED);
      for (let i = 0; i < 400; i += 1) tick(b, b.made[1], 317.4);
      expect(b.sets()).toHaveLength(3);
      b.m.stop();
    });

    // The next start is a new run: what a card can do changes with heat.
    test('the next start tries the card again', () => {
      const b = rig().start();
      mine(b, b.made[0], RIG01);
      b.m.stop();
      b.start();
      expect(b.lines().filter((l) => l === START())).toHaveLength(2);
      mine(b, b.made[1], RIG01);
      expect(b.lines().filter((l) => l === KEPT_150)).toHaveLength(2);
      // A hit is recomputed in the new run only once the card leaves 0 again.
      b.m.stop();
      b.start();
      b.made[2].emit('hit', WRONG);
      expect(b.submits().filter((s) => s.params.job_id === JOB)).toHaveLength(1);
      b.m.stop();
    });
  });

  // START LLM while mining-only is up: a model is about to be served from the
  // card, and nothing checks what a model computes.
  describe('releaseMemClocks()', () => {
    test('in a step\'s turn: the card goes back to 0, the trial ends, and no card is tried after it', () => {
      jest.useFakeTimers();
      try {
        const b = rig().start();
        const core = b.made[0];
        untilTurn(b, core, 100);
        b.m.releaseMemClocks();
        expect(b.sets()).toEqual([0, 50, 100, 0]);
        expect(b.lines().pop()).toBe(BACK);
        mine(b, core, RIG01);
        for (let i = 0; i < 400; i += 1) tick(b, core, 317.4);
        expect(b.sets()).toHaveLength(4);
        // Nor a card that fails and is opened again.
        core.emit('error', new Error('GPU 0: misaligned address'));
        jest.advanceTimersByTime(RESTART_MS);
        expect(b.made).toHaveLength(2);
        expect(b.calls.filter((c) => c[0] === 'read')).toHaveLength(1);
        b.m.stop();
        expect(b.sets()).toHaveLength(4);
      } finally {
        jest.clearAllTimers();
        jest.useRealTimers();
      }
    });

    test('during a check: the card goes back to 0 and to the pool\'s job', () => {
      const b = rig().start();
      const core = b.made[0];
      untilCheck(b, core, 150);
      b.m.releaseMemClocks();
      expect(b.sets()).toEqual([0, 50, 100, 150, 0]);
      expect(jobs(core).slice(-2)).toEqual(['offset-check-3', JOB]);
      // The rest of the check's hits are dropped.
      core.emit('hit', checkHitOf({ jobId: 'offset-check-3' }, MISMATCH));
      expect(b.lines('error')).toEqual([]);
      expect(b.m.isRunning()).toBe(true);
      b.m.stop();
    });

    test('with an offset kept: the card goes back to 0 and mines on', () => {
      const b = rig().start();
      mine(b, b.made[0], RIG01);
      b.m.releaseMemClocks();
      expect(b.sets().slice(-2)).toEqual([150, 0]);
      expect(b.lines().pop()).toBe(BACK);
      expect(b.m.cores).toHaveLength(1);
      // Idempotent, and a later stop finds nothing left to put back.
      b.m.releaseMemClocks();
      b.m.stop();
      expect(b.sets()).toHaveLength(7);
    });

    test('before the trial\'s first step: the trial ends with nothing to put back', () => {
      const b = rig().start();
      tick(b, b.made[0], 317.4);
      b.m.releaseMemClocks();
      mine(b, b.made[0], RIG01);
      for (let i = 0; i < 400; i += 1) tick(b, b.made[0], 317.4);
      expect(b.sets()).toEqual([0]);
      expect(b.lines()).toEqual([START()]);
      b.m.stop();
    });
  });

  describe('stop()', () => {
    test('during a check puts the card back at 0 before stopped, and hands the stopped core no job', () => {
      const b = rig().start();
      const core = b.made[0];
      untilCheck(b, core, 100);
      b.m.stop();
      expect(b.calls.slice(-2)).toEqual([['set', 0, 0], ['stopped']]);
      expect(b.sets()).toEqual([0, 50, 100, 0]);
      expect(jobs(core)).toEqual([JOB, 'offset-check-1', JOB, 'offset-check-2']);
      expect(core.stop).toHaveBeenCalledTimes(1);
      expect(b.lines().pop()).toBe(BACK);
      // A hit still on its way from the stopped core goes nowhere.
      core.emit('hit', checkHitOf({ jobId: 'offset-check-2' }));
      expect(b.sets()).toHaveLength(4);
    });

    test('in a step\'s turn puts the card back at 0 before stopped', () => {
      const b = rig().start();
      untilTurn(b, b.made[0], 150);
      b.m.stop();
      expect(b.calls.slice(-2)).toEqual([['set', 0, 0], ['stopped']]);
      expect(b.sets()).toEqual([0, 50, 100, 150, 0]);
    });

    test('puts back only the cards this run changed', () => {
      const b = rig().start({ gpus: TWO, mineCoreOffsetTrial: [0] });
      untilTurn(b, b.made[0], 50);
      b.m.stop();
      expect(b.calls.filter((c) => c[1] === 1)).toEqual([]);
      expect(b.calls.slice(-2)).toEqual([['set', 0, 0], ['stopped']]);
    });
  });

  // --mine-mem-clock auto beside it: the memory clock trial goes first, so the
  // two never read each other's changes.
  describe('with a memory clock trial on the same card', () => {
    const BOTH = { mineMemClockByIndex: { 0: 5001 }, mineMemClockTrial: [0] };
    // The memory clock trial's four turns: 317.5 TH/s at the driver's clock,
    // `locked` with the lock.
    function memTrial(b, core, locked) {
      while (b.m.memClockTrials.has(0)) tick(b, core, b.m.memClockLocked.has(0) ? locked : 317.5);
    }

    test('waits for its verdict, then runs with the lock it kept still in place', () => {
      const b = rig().start(BOTH);
      const core = b.made[0];
      // The memory clock trial hands the card to the driver before its first turn.
      expect(b.calls).toEqual([['reset', 0], ['read', 0], ['set', 0, 0]]);
      expect(b.lines()).toEqual([START(0, '+50, +100, +150', ', once the memory clock trial is over')]);
      memTrial(b, core, 320.5);
      expect(b.t.now).toBe(4 * MEM_TURN_MS);
      // The memory clock trial alone moved the card: no offset yet.
      expect(b.calls.slice(3)).toEqual([['lock', 0, 5001], ['reset', 0], ['lock', 0, 5001]]);
      expect(b.logs.map((l) => l.line))
        .toContain('memory clock kept at 5001 MHz on GPU 0 while mining: 320.5 TH/s locked against 317.5 (+0.9%)');

      mine(b, core, RIG01);
      // Its first turn starts on the sample that ended the memory clock trial.
      expect(b.calls.slice(6)).toEqual([
        ['set', 0, 50], ['set', 0, 100], ['set', 0, 150], ['set', 0, 0], ['set', 0, 150],
      ]);
      expect(b.lines().pop()).toBe(KEPT_150);
      b.m.stop();
      expect(b.calls.slice(-3)).toEqual([['reset', 0], ['set', 0, 0], ['stopped']]);
    });

    test('starts after a memory clock trial that kept nothing, or could not run', () => {
      const slower = rig().start(BOTH);
      memTrial(slower, slower.made[0], 283.1);
      mine(slower, slower.made[0], RIG01);
      expect(slower.lines().pop()).toBe(KEPT_150);
      slower.m.stop();

      const noRights = rig({ lock: { ok: false, error: 'sudo: a password is required' } }).start(BOTH);
      memTrial(noRights, noRights.made[0], 320.5);
      expect(noRights.t.now).toBe(MEM_TURN_MS);
      mine(noRights, noRights.made[0], RIG01);
      expect(noRights.lines().pop()).toBe(KEPT_150);
      noRights.m.stop();
    });

    test('a fault during the memory clock trial ends the offset trial before it began', () => {
      jest.useFakeTimers();
      try {
        const b = rig().start(BOTH);
        for (let i = 0; i < 20; i += 1) tick(b, b.made[0], 317.5);
        b.made[0].emit('error', new Error('GPU 0: misaligned address'));
        jest.advanceTimersByTime(RESTART_MS);
        expect(b.lines().pop()).toBe(ENDED);
        // The memory clock trial starts over on the new core; the offset's does not.
        memTrial(b, b.made[1], 320.5);
        for (let i = 0; i < 400; i += 1) tick(b, b.made[1], 320.5);
        expect(b.sets()).toEqual([0]);
        b.m.stop();
      } finally {
        jest.clearAllTimers();
        jest.useRealTimers();
      }
    });
  });
});
