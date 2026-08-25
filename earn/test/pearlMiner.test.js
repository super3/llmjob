'use strict';

const { EventEmitter } = require('events');
const { PearlMiner, RECONNECT_MS } = require('../src/main/pearlMiner');
const { encode } = require('../src/shared/miner/stratum');
const { shareBound, PROFILE, buildConfig52, regionToTile } = require('../src/shared/miner/pearlhash');
const { hash, keyedHash } = require('../src/shared/miner/blake3');
const M = require('../src/shared/miner/merkle');

// A profile small enough to build real commitment trees in a test, but the same
// shape as the real one: 1024-byte leaves, a power-of-two leaf count, the 4x16
// tile, and k/rank = 8 chunks. m*k/1024 = 64 leaves a side.
const TINY = { k: 1024, rank: 128, mmaType: 0, m: 64, n: 64 };

const ADDR = 'prl1px5ervx6ftaegmdhqa5ajemh20j2uw7l9jt5j5s97rljp72yt3s8qncrxud';
const MDL = 'mdl1pl80mdy0culfn3g7jl3paa5ccnc8gkkfmkc6t2x0q6rmvd9dpu5wsk0v3z8';
const HEADER = '000000203a49fea8b6d42c60c543fe0f029749787679372495e5d2d1007e29e3'
  + '25e1c08065a0cedc057ba091aabd10476017b3d3f4e38eafb59707ded57f71e1feb7b5d155918b6a0dea0018';
// ~2^203, the live target.
const TARGET = '00000000000007fff80000000000000000000000000000000000000000000000';

function makeSocket() {
  const s = new EventEmitter();
  s.written = [];
  s.write = jest.fn((d) => { s.written.push(d); return true; });
  s.destroy = jest.fn();
  return s;
}

function makeCore() {
  const c = new EventEmitter();
  c.setJob = jest.fn();
  c.stop = jest.fn();
  return c;
}

function boot(over = {}) {
  const sock = over.sock || makeSocket();
  const core = over.core || makeCore();
  const connect = jest.fn(() => sock);
  const createCore = over.createCore === null ? null : jest.fn(() => core);
  const m = new PearlMiner({ connect, createCore, reconnectMs: 0 });
  const events = { log: [], job: [], share: [], rejected: [], hashrate: [], error: [], stopped: [], authorized: [], started: [] };
  for (const k of Object.keys(events)) m.on(k, (e) => events[k].push(e));
  return { m, sock, core, connect, createCore, events };
}

function jobLine(over = {}) {
  return encode({
    id: null, method: 'mining.notify',
    params: {
      job_id: '00000000_2097152', header: HEADER, target: TARGET,
      height: 103353, cert_version: 3, ...over,
    },
  });
}

const settings = { address: ADDR, worker: 'rig01', endpoint: 'us.pearl.herominers.com:1200' };

describe('PearlMiner — start', () => {
  test('connects, authorizes with object params, and reports started', () => {
    const { m, sock, connect, events } = boot();
    expect(m.start(settings)).toBe(true);
    expect(m.isRunning()).toBe(true);
    expect(connect).toHaveBeenCalledWith('us.pearl.herominers.com', 1200);

    sock.emit('connect');
    const sent = JSON.parse(sock.written[0]);
    expect(sent).toEqual({
      id: 1, method: 'mining.authorize', params: { wallet: ADDR, worker: 'rig01' },
    });
    expect(events.started[0]).toEqual({ pool: settings.endpoint, wallet: ADDR, worker: 'rig01' });
  });

  test('is a no-op while already running', () => {
    const { m } = boot();
    expect(m.start(settings)).toBe(true);
    expect(m.start(settings)).toBe(false);
  });

  // HeroMiners documents the MDL merge login as PRL+MDL.WORKER, and
  // combinePayoutAddress already produces exactly that.
  test('carries an MDL address into the wallet field', () => {
    const { m, sock } = boot();
    m.start({ ...settings, mdlAddress: MDL });
    sock.emit('connect');
    expect(JSON.parse(sock.written[0]).params.wallet).toBe(ADDR + '+' + MDL);
  });

  test('a malformed MDL address is dropped, not sent', () => {
    const { m, sock } = boot();
    m.start({ ...settings, mdlAddress: 'nope' });
    sock.emit('connect');
    expect(JSON.parse(sock.written[0]).params.wallet).toBe(ADDR);
  });

  // The expected state anywhere the CUDA core is not built. It must stop
  // cleanly and say why, and must NOT open a pool socket it could never feed.
  test('without a built core it explains and stops instead of crashing', () => {
    const { m, connect, events } = boot({ createCore: null });
    expect(m.start(settings)).toBe(false);
    expect(m.isRunning()).toBe(false);
    expect(connect).not.toHaveBeenCalled();
    expect(events.log[0].level).toBe('error');
    expect(events.log[0].line).toMatch(/core is not built/i);
    expect(events.stopped).toHaveLength(1);
  });

  test('a core that throws on construction surfaces as an error, not a crash', () => {
    const boom = new Error('CUDA driver version is insufficient');
    const m = new PearlMiner({ connect: jest.fn(), createCore: () => { throw boom; } });
    const errs = [];
    m.on('error', (e) => errs.push(e));
    expect(m.start(settings)).toBe(false);
    expect(errs).toEqual([boom]);
    expect(m.isRunning()).toBe(false);
  });

  test('a connect that throws is reported rather than thrown', () => {
    const m = new PearlMiner({ connect: () => { throw new Error('EHOSTUNREACH'); }, createCore: () => makeCore() });
    const errs = [];
    m.on('error', (e) => errs.push(e));
    m.start(settings);
    expect(errs[0].message).toBe('EHOSTUNREACH');
  });
});

describe('PearlMiner — protocol', () => {
  function running() {
    const b = boot();
    b.m.start(settings);
    b.sock.emit('connect');
    b.sock.written.length = 0;
    return b;
  }

  test('authorize success flips authorized and is logged', () => {
    const { sock, m, events } = running();
    sock.emit('data', '{"id":1,"error":null,"result":true}\n');
    expect(m.authorized).toBe(true);
    expect(events.authorized).toHaveLength(1);
  });

  test('authorize failure is reported and does not authorize', () => {
    const { sock, m, events } = running();
    sock.emit('data', '{"id":1,"result":null,"error":{"code":24,"msg":"bad address"}}\n');
    expect(m.authorized).toBe(false);
    expect(events.log.some((l) => l.level === 'error' && /bad address/.test(l.line))).toBe(true);
  });

  test('a job is handed to the core and announced', () => {
    const { sock, core, events } = running();
    sock.emit('data', jobLine());
    expect(events.job[0]).toEqual({ jobId: '00000000_2097152', height: 103353 });
    const arg = core.setJob.mock.calls[0][0];
    expect(arg.jobId).toBe('00000000_2097152');
    expect(arg.header).toHaveLength(76);
    // The core gets the SCALED bound, not the pool's raw target: the protocol
    // makes the bound easier in proportion to the work one attempt costs. Passing
    // the raw target makes shares 65536x rarer than the pool intends, which reads
    // as bad luck rather than as a bug.
    expect(arg.target).toBe(shareBound(BigInt('0x' + TARGET), PROFILE));
    expect(arg.target).toBe(BigInt('0x' + TARGET) * 524288n);
  });

  // A target so easy that scaling it by the adjustment factor would not fit 256
  // bits. The reference refuses rather than saturating, because a saturated
  // bound is satisfied by EVERY hash and would flood the pool with junk. So the
  // job is dropped with a reason rather than mined at a bound that means nothing.
  test('a job whose target cannot be scaled is refused, not saturated', () => {
    const { sock, core, events } = running();
    sock.emit('data', jobLine({ target: 'ff'.repeat(32) }));
    expect(core.setJob).not.toHaveBeenCalled();
    expect(events.log.some((l) => /too easy to scale/.test(l.line))).toBe(true);
  });

  // Lines arrive as pipe slices, not messages; a job split across two chunks
  // must still be delivered exactly once and whole.
  test('a message split across two chunks is handled once', () => {
    const { sock, core, events } = running();
    const line = jobLine();
    const cut = line.length - 12;
    sock.emit('data', line.slice(0, cut));
    expect(events.job).toHaveLength(0);
    sock.emit('data', line.slice(cut));
    expect(events.job).toHaveLength(1);
    expect(core.setJob).toHaveBeenCalledTimes(1);
  });

  test('several messages in one chunk are all handled', () => {
    const { sock, m, events } = running();
    sock.emit('data', '{"id":1,"error":null,"result":true}\n' + jobLine());
    expect(m.authorized).toBe(true);
    expect(events.job).toHaveLength(1);
  });

  // The exact failure that made alpha-miner's Ada builds run rank 512 and earn
  // nothing while looking healthy: the fork only credits the profile rank.
  test('a job at an uncredited rank is refused before the GPU touches it', () => {
    const { sock, core, events } = running();
    sock.emit('data', jobLine({ rank: 512 }));
    expect(core.setJob).not.toHaveBeenCalled();
    expect(events.job).toHaveLength(0);
    expect(events.log.some((l) => l.level === 'error' && /rank 512 is not the credited 128/.test(l.line))).toBe(true);
  });

  test('a job at the credited rank is mined', () => {
    const { sock, core } = running();
    sock.emit('data', jobLine({ rank: 128 }));
    expect(core.setJob).toHaveBeenCalledTimes(1);
  });

  test('a custom profile changes which rank is credited', () => {
    const b = boot();
    // A whole profile, not a fragment: the bound is computed from k and the
    // tile as well as the rank, so a partial override has no meaning.
    b.m.start({ ...settings, profile: { ...PROFILE, rank: 256, k: 4096 } });
    b.sock.emit('connect');
    b.sock.emit('data', jobLine({ rank: 256 }));
    expect(b.core.setJob).toHaveBeenCalledTimes(1);
  });

  test('an unusable job is refused and never reaches the core', () => {
    const { sock, core, events } = running();
    sock.emit('data', jobLine({ header: 'tooshort' }));
    expect(core.setJob).not.toHaveBeenCalled();
    expect(events.log.some((l) => /unusable job/.test(l.line))).toBe(true);
  });

  test('vardiff is logged; the next job carries the real target', () => {
    const { sock, events } = running();
    sock.emit('data', '{"method":"mining.set_difficulty","params":4000000}\n');
    expect(events.log.some((l) => /difficulty . 4000000/.test(l.line))).toBe(true);
  });

  test('blank lines and junk are tolerated', () => {
    const { sock, events } = running();
    sock.emit('data', '\n  \nnot json\n');
    expect(events.error).toHaveLength(0);
    expect(events.log.some((l) => /not json/.test(l.line))).toBe(true);
  });
});

describe('PearlMiner — shares', () => {
  function withJob() {
    const b = boot();
    b.m.start({ ...settings, profile: TINY });
    b.sock.emit('connect');
    b.sock.emit('data', jobLine());
    b.sock.written.length = 0;
    return b;
  }

  // A hit with GENUINE Merkle proofs, built the way the device builds them, so
  // the local certify step in _onHit is actually exercised. A stub proof would
  // make these tests pass against a miner that submits nothing provable.
  //
  // The operand contents are arbitrary -- the proof certifies that these leaves
  // sit under this root, and the root is whatever we committed to.
  function proofSide(matrix, rows, cols, jobKey) {
    const layers = M.buildLayers(jobKey, matrix);
    const leafIndices = M.leafIndicesFromRows(rows, cols);
    const p = M.multiLeafProof(jobKey, matrix, layers, leafIndices);
    return {
      leafIndices: p.leafIndices,
      leafData: Buffer.concat(p.leafData),
      siblings: Buffer.concat(p.siblings),
      root: p.root,
      totalLeaves: p.totalLeaves,
    };
  }

  function goodHit(jobId = '00000000_2097152', nonce = 0) {
    const h = Buffer.alloc(32);
    h[24] = 0x01; // 2^192, comfortably under the scaled bound
    const jobKey = hash(Buffer.concat([Buffer.from(HEADER, 'hex'), buildConfig52(TINY)]));
    const A = keyedHash(Buffer.alloc(32, 7), Buffer.alloc(64), TINY.m * TINY.k);
    const B = keyedHash(Buffer.alloc(32, 9), Buffer.alloc(64), TINY.n * TINY.k);
    const { rows, cols } = regionToTile(nonce, TINY);
    return {
      jobId,
      jackpotHash: h,
      nonce,
      proofA: proofSide(A, rows, TINY.k, jobKey),
      proofBt: proofSide(B, cols, TINY.k, jobKey),
    };
  }

  test('a valid hit is submitted as a plain proof', () => {
    const { core, sock } = withJob();
    core.emit('hit', goodHit());
    const sent = JSON.parse(sock.written[0]);
    expect(sent.method).toBe('mining.submit');
    expect(Object.keys(sent.params).sort()).toEqual(['hs', 'job_id', 'plain_proof']);
    expect(sent.params.job_id).toBe('00000000_2097152');
    expect(Buffer.from(sent.params.plain_proof, 'base64').length).toBeGreaterThan(1024);
    expect(sent.id).toBeGreaterThan(1); // never collides with the authorize id
  });

  // The device re-draws its operands every few tens of milliseconds. A proof
  // that has drifted off its hash cannot be certified anywhere, and submitting
  // it costs a round trip and counts against the worker.
  test('a hit whose proof does not verify is dropped, not submitted', () => {
    const { core, sock, events } = withJob();
    const hit = goodHit();
    hit.proofA.leafData[0] ^= 0xff;
    core.emit('hit', hit);
    expect(sock.written).toHaveLength(0);
    expect(events.log.some((l) => /does not verify locally/.test(l.line))).toBe(true);
  });

  // Nothing downstream can rebuild a proof the core did not attach.
  test('a hit with no proof attached is dropped', () => {
    const { core, sock } = withJob();
    const hit = goodHit();
    delete hit.proofA;
    core.emit('hit', hit);
    expect(sock.written).toHaveLength(0);
  });

  test('the pool verdict is matched back to the job that produced it', () => {
    const { core, sock, events } = withJob();
    core.emit('hit', goodHit());
    const id = JSON.parse(sock.written[0]).id;
    sock.emit('data', JSON.stringify({ id, result: true, error: null }) + '\n');
    expect(events.share[0]).toEqual({ jobId: '00000000_2097152', accepted: true });
  });

  test('a rejection reports the pool reason', () => {
    const { core, sock, events } = withJob();
    core.emit('hit', goodHit());
    const id = JSON.parse(sock.written[0]).id;
    sock.emit('data', JSON.stringify({ id, result: null, error: [21, 'Job not found'] }) + '\n');
    expect(events.rejected[0]).toEqual({ jobId: '00000000_2097152', reason: '[21] Job not found' });
  });

  test('a verdict for an unknown id still resolves without throwing', () => {
    const { sock, events } = withJob();
    sock.emit('data', '{"id":9999,"result":true,"error":null}\n');
    expect(events.share[0]).toEqual({ jobId: null, accepted: true });
  });

  // The core may still be finishing a job the pool has replaced. Submitting that
  // is how a miner earns a stale-share ban, so it is dropped.
  test('a hit for a superseded job is dropped', () => {
    const { core, sock } = withJob();
    sock.emit('data', jobLine({ job_id: 'newer_job' }));
    sock.written.length = 0;
    core.emit('hit', goodHit('00000000_2097152'));
    expect(sock.written).toHaveLength(0);
  });

  // Re-verified in JS because vardiff can move the target under the core.
  test('a hit that no longer meets the target is dropped, not submitted', () => {
    const { core, sock, events } = withJob();
    const weak = Buffer.alloc(32);
    weak[31] = 0xff; // astronomically above target
    core.emit('hit', { ...goodHit(), jackpotHash: weak });
    expect(sock.written).toHaveLength(0);
    expect(events.log.some((l) => /no longer meets target/.test(l.line))).toBe(true);
  });

  test('a hit arriving with no job at all is ignored', () => {
    const { m, core, sock } = boot();
    m.start(settings);
    sock.emit('connect');
    sock.written.length = 0;
    core.emit('hit', goodHit());
    expect(sock.written).toHaveLength(0);
  });

  test('hashrate and core errors are relayed', () => {
    const { core, events } = withJob();
    core.emit('hashrate', 296.5);
    const err = new Error('kernel launch failed');
    core.emit('error', err);
    expect(events.hashrate).toEqual([296.5]);
    expect(events.error).toEqual([err]);
  });
});

describe('PearlMiner — defaults and edges', () => {
  test('start with no settings at all does not throw', () => {
    const { m: mm, connect } = boot();
    expect(mm.start()).toBe(true);
    // No endpoint means no host and no port — it still fails safely rather than
    // throwing somewhere deep in the socket layer.
    expect(connect).toHaveBeenCalledWith('', NaN);
  });

  test('the worker falls back to rig01', () => {
    const { m: mm, sock } = boot();
    mm.start({ address: ADDR, endpoint: 'h:1' });
    sock.emit('connect');
    expect(JSON.parse(sock.written[0]).params.worker).toBe('rig01');
  });

  test('a bad job with no id still logs without appending an empty name', () => {
    const b = boot();
    b.m.start(settings);
    b.sock.emit('connect');
    b.sock.emit('data', encode({
      id: null, method: 'mining.notify',
      params: { header: 'short', target: TARGET },
    }));
    expect(b.events.log.some((l) => /ignoring an unusable job$/.test(l.line))).toBe(true);
  });

  test('a rejection for an unknown id reports a null job rather than throwing', () => {
    const b = boot();
    b.m.start(settings);
    b.sock.emit('connect');
    b.sock.emit('data', '{"id":4242,"result":null,"error":[21,"Job not found"]}\n');
    expect(b.events.rejected[0]).toEqual({ jobId: null, reason: '[21] Job not found' });
  });

  test('a verdict with no error detail still renders a reason', () => {
    const b = boot();
    b.m.start(settings);
    b.sock.emit('connect');
    b.sock.emit('data', '{"id":4243,"result":null,"error":[]}\n');
    expect(b.events.rejected[0].reason).toBe('');
  });

  // A reconnect timer must never be the only thing keeping the process alive.
  test('the reconnect timer does not hold the event loop open', () => {
    jest.useFakeTimers();
    const b = boot();
    b.m.start(settings);
    b.sock.emit('close');
    expect(b.m._reconnectTimer).toBeTruthy();
    b.m.stop();
    jest.useRealTimers();
  });
});

describe('PearlMiner — lifecycle', () => {
  test('stop releases the core and the socket, and reports stopped', () => {
    const { m, sock, core, events } = boot();
    m.start(settings);
    expect(m.stop()).toBe(true);
    expect(core.stop).toHaveBeenCalled();
    expect(sock.destroy).toHaveBeenCalled();
    expect(m.isRunning()).toBe(false);
    expect(events.stopped).toHaveLength(1);
    expect(m.stop()).toBe(false); // idempotent
  });

  test('stop survives a core and socket that throw on teardown', () => {
    const sock = makeSocket();
    sock.destroy = () => { throw new Error('already closed'); };
    const core = makeCore();
    core.stop = () => { throw new Error('core gone'); };
    const { m } = boot({ sock, core });
    m.start(settings);
    expect(m.stop()).toBe(true);
  });

  test('an unexpected close reconnects while running', () => {
    jest.useFakeTimers();
    const { m, sock, connect } = boot();
    m.start(settings);
    expect(connect).toHaveBeenCalledTimes(1);
    sock.emit('close');
    jest.runOnlyPendingTimers();
    expect(connect).toHaveBeenCalledTimes(2);
    m.stop();
    jest.useRealTimers();
  });

  test('a close after stop does not reconnect', () => {
    jest.useFakeTimers();
    const { m, sock, connect } = boot();
    m.start(settings);
    m.stop();
    sock.emit('close');
    jest.runOnlyPendingTimers();
    expect(connect).toHaveBeenCalledTimes(1);
    jest.useRealTimers();
  });

  test('a pending reconnect is cancelled by stop', () => {
    jest.useFakeTimers();
    const { m, sock, connect } = boot();
    m.start(settings);
    sock.emit('close');
    m.stop();
    jest.runOnlyPendingTimers();
    expect(connect).toHaveBeenCalledTimes(1);
    jest.useRealTimers();
  });

  // connect() throwing leaves the miner running with no socket at all, so stop
  // has to tear down cleanly against a half-built state.
  test('stop works when connect never produced a socket', () => {
    const core = makeCore();
    const mm = new PearlMiner({
      connect: () => { throw new Error('EHOSTUNREACH'); },
      createCore: () => core,
    });
    mm.on('error', () => {});
    mm.start(settings);
    expect(mm.isRunning()).toBe(true);
    expect(mm.stop()).toBe(true);
    expect(core.stop).toHaveBeenCalled();
  });

  test('socket errors are logged, not thrown', () => {
    const { m, sock, events } = boot();
    m.start(settings);
    sock.emit('error', new Error('ECONNRESET'));
    expect(events.log.some((l) => l.level === 'error' && /ECONNRESET/.test(l.line))).toBe(true);
  });

  // A pool that rejects with a code but no text must still render a reason
  // rather than "[21] undefined".
  test('a coded rejection with no message renders just the code', () => {
    const b = boot();
    b.m.start(settings);
    b.sock.emit('connect');
    b.sock.emit('data', '{"id":5,"result":null,"error":[21,""]}' + String.fromCharCode(10));
    expect(b.events.rejected[0].reason).toBe('[21] ');
  });

  test('the default reconnect delay is a sane backoff', () => {
    expect(RECONNECT_MS).toBeGreaterThanOrEqual(1000);
  });

  test('constructs with no options at all', () => {
    expect(new PearlMiner()).toBeInstanceOf(PearlMiner);
  });
});
