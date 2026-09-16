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

  // Which card found it has to be recorded at submit time: the pool's answer
  // comes back with only a submit id, and on a multi-card rig every share would
  // otherwise be credited to card 0.
  test('records the card whose core found the hit', () => {
    const core = makeCore();
    core.device = { index: 1, name: 'NVIDIA GeForce RTX 4070' };
    const b = boot({ core });
    b.m.start({ ...settings, profile: TINY, gpus: [{ index: 1, name: 'NVIDIA GeForce RTX 4070' }] });
    b.sock.emit('connect');
    b.sock.emit('data', jobLine());
    b.sock.written.length = 0;
    core.emit('hit', goodHit());
    const sent = JSON.parse(b.sock.written[0]);
    expect(b.m.pending.get(sent.id)).toEqual({ jobId: '00000000_2097152', index: 1 });
  });

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

  // A proof over the WRONG rows is internally consistent -- the leaves hash to
  // the root, because they are real leaves of the same tree -- so hashing alone
  // cannot catch it. The verifier reads each claimed row as bytes
  // [row*k, row*k+k) out of the leaves it was given, and answers "Failed to
  // extract strip" when they are not there.
  //
  // This is not a hypothetical. The device expanded the tile's row offset
  // against a hand-written mask that still said 4 rows after the tile grew to
  // 16, so it snapshotted rows a quarter of the way up the matrix. The fold was
  // right, the hash was right, the proof verified, and the pool rejected every
  // share.
  test('a proof describing the wrong rows is dropped', () => {
    const { core, sock, events } = withJob();
    const hit = goodHit();
    const jobKey = hash(Buffer.concat([Buffer.from(HEADER, 'hex'), buildConfig52(TINY)]));
    const A = keyedHash(Buffer.alloc(32, 7), Buffer.alloc(64), TINY.m * TINY.k);
    // Rows 16..31 rather than the 0..15 that region 0 names. Same tree, same
    // root, a perfectly valid Merkle proof -- of the wrong thing.
    const wrongRows = TINY.rows ? TINY.rows.map((r) => r + 16)
      : Array.from({ length: 16 }, (_, i) => i + 16);
    hit.proofA = proofSide(A, wrongRows, TINY.k, jobKey);
    core.emit('hit', hit);
    expect(sock.written).toHaveLength(0);
    expect(events.log.some((l) => /does not verify locally/.test(l.line))).toBe(true);
  });

  test('the pool verdict is matched back to the job that produced it', () => {
    const { core, sock, events } = withJob();
    core.emit('hit', goodHit());
    const id = JSON.parse(sock.written[0]).id;
    sock.emit('data', JSON.stringify({ id, result: true, error: null }) + '\n');
    expect(events.share[0]).toEqual({ jobId: '00000000_2097152', accepted: true, index: 0 });
  });

  test('a rejection reports the pool reason', () => {
    const { core, sock, events } = withJob();
    core.emit('hit', goodHit());
    const id = JSON.parse(sock.written[0]).id;
    sock.emit('data', JSON.stringify({ id, result: null, error: [21, 'Job not found'] }) + '\n');
    expect(events.rejected[0])
      .toEqual({ jobId: '00000000_2097152', reason: '[21] Job not found', index: 0 });
  });

  test('a verdict for an unknown id still resolves without throwing', () => {
    const { sock, events } = withJob();
    sock.emit('data', '{"id":9999,"result":true,"error":null}\n');
    expect(events.share[0]).toEqual({ jobId: null, accepted: true, index: 0 });
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
    expect(b.events.rejected[0]).toEqual({ jobId: null, reason: '[21] Job not found', index: 0 });
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

// Which card is mining is not something the host can work out for itself: the
// core opens a CUDA device, and CUDA's device list is not nvidia-smi's. So the
// core says, and the host repeats it. A rig that showed one card in the UI while
// another did the work (issue #226) is the failure this closes.
describe('PearlMiner — the cards the cores opened', () => {
  const withDevice = (device) => {
    const core = makeCore();
    if (device !== undefined) core.device = device;
    return boot({ core });
  };

  test('is taken from the core and said out loud', () => {
    const b = withDevice({ index: 1, name: 'NVIDIA GeForce RTX 4070' });
    b.m.start(settings);
    expect(b.m.devices()).toEqual([{ index: 1, name: 'NVIDIA GeForce RTX 4070' }]);
    expect(b.events.log.map((l) => l.line))
      .toContain('mining on GPU 1 · NVIDIA GeForce RTX 4070');
  });

  // A core built before the device choice existed says nothing. The host has to
  // stay usable against it -- an older pearl_core.node beside a newer app is the
  // normal state of a rig part-way through an upgrade -- so "unknown" falls back
  // to the behaviour that shipped before.
  test('is empty when the core does not report one, without a log line', () => {
    const b = withDevice(undefined);
    b.m.start(settings);
    expect(b.m.devices()).toEqual([]);
    expect(b.events.log.map((l) => l.line).some((l) => l.startsWith('mining on GPU'))).toBe(false);
  });

  test('ignores a device the core cannot describe', () => {
    for (const bad of [null, {}, { index: -1 }, { index: 1.5 }, { index: '0' }]) {
      const b = withDevice(bad);
      b.m.start(settings);
      expect(b.m.devices()).toEqual([]);
    }
  });

  // An index with no name still names something. "GPU 1" is a poor label but it
  // is a true one, and it keeps the UI from falling back to a card name that was
  // detected separately and may belong to a different card entirely.
  test('falls back to the bare index when the core gives no name', () => {
    const b = withDevice({ index: 2, name: '' });
    b.m.start(settings);
    expect(b.m.devices()).toEqual([{ index: 2, name: 'GPU 2' }]);
  });

  // A start that cannot build a core must not leave the last run's cards
  // standing: the UI would then label a rig that is mining nothing.
  test('is cleared when the core will not construct', () => {
    const core = makeCore();
    core.device = { index: 1, name: 'NVIDIA GeForce RTX 4070' };
    const b = boot({ core });
    b.m.start(settings);
    expect(b.m.devices()).toHaveLength(1);
    b.m.stop();
    b.m.createCore.mockImplementationOnce(() => { throw new Error('no CUDA device found'); });
    expect(b.m.start(settings)).toBe(false);
    expect(b.m.devices()).toEqual([]);
  });
});

// Mining on every card the rig has. One core per card, each on its own slice of
// the search space -- without that they would search the same operands and find
// the same shares, and the second card would earn nothing.
describe('PearlMiner — one core per card', () => {
  const GPUS = [
    { index: 0, name: 'NVIDIA RTX PRO 4500 Blackwell' },
    { index: 1, name: 'NVIDIA GeForce RTX 4070' },
  ];

  // A factory that hands out a fresh core per call and reports the card it was
  // asked for, which is what the real addon does.
  function fleet(over = {}) {
    const made = [];
    const sock = over.sock || makeSocket();
    const createCore = jest.fn((profile, opts) => {
      const c = makeCore();
      c.opts = opts;
      if (!over.silentDevice) c.device = { index: opts.deviceIndex, name: 'GPU' + opts.deviceIndex };
      made.push(c);
      return c;
    });
    const m = new PearlMiner({ connect: () => sock, createCore, reconnectMs: 0 });
    const events = { log: [], share: [], rejected: [], hashrate: [], error: [], stopped: [] };
    for (const k of Object.keys(events)) m.on(k, (...a) => events[k].push(a.length > 1 ? a : a[0]));
    return { m, sock, made, createCore, events };
  }

  test('starts a core on every card', () => {
    const b = fleet();
    expect(b.m.start({ ...settings, gpus: GPUS })).toBe(true);
    expect(b.made).toHaveLength(2);
    expect(b.m.devices().map((d) => d.index)).toEqual([0, 1]);
    expect(b.events.log.map((l) => l.line)).toEqual(expect.arrayContaining([
      'mining on GPU 0 · GPU0',
      'mining on GPU 1 · GPU1',
    ]));
  });

  // The whole point of the slice. Same base and stride on two cards means two
  // cards drawing the same operands, searching the same regions and submitting
  // the same shares -- the pool takes one and the second card earns nothing.
  test('gives each card its own slice of the search space', () => {
    const b = fleet();
    b.m.start({ ...settings, gpus: GPUS });
    expect(b.made.map((c) => c.opts)).toEqual([
      { saltBase: 0, saltStride: 2, deviceIndex: 0 },
      { saltBase: 1, saltStride: 2, deviceIndex: 1 },
    ]);
  });

  test('every card gets the job', () => {
    const b = fleet();
    b.m.start({ ...settings, gpus: GPUS });
    b.sock.emit('connect');
    b.sock.emit('data', jobLine());
    for (const c of b.made) expect(c.setJob).toHaveBeenCalledTimes(1);
  });

  test('stop releases every card', () => {
    const b = fleet();
    b.m.start({ ...settings, gpus: GPUS });
    b.m.stop();
    for (const c of b.made) expect(c.stop).toHaveBeenCalled();
  });

  // The rig's hashrate is the sum, and the pool is told the sum -- a share
  // submitted from one card still reports what the whole rig is doing.
  test('sums the hashrate across cards', () => {
    const b = fleet();
    b.m.start({ ...settings, gpus: GPUS });
    b.made[0].emit('hashrate', 100);
    b.made[1].emit('hashrate', 40);
    expect(b.m.totalHashrate()).toBe(140);
    expect(b.events.hashrate).toEqual([
      [100, { index: 0, name: 'GPU0' }],
      [40, { index: 1, name: 'GPU1' }],
    ]);
  });

  // A card with no room -- the local LLM has most of its VRAM -- must not take
  // the rest of the rig down with it. It used to: one core, one failure, no
  // mining at all.
  test('keeps mining on the cards that start when one refuses', () => {
    const b = fleet();
    b.createCore.mockImplementationOnce(() => { throw new Error('not enough free VRAM'); });
    expect(b.m.start({ ...settings, gpus: GPUS })).toBe(true);
    expect(b.m.devices().map((d) => d.index)).toEqual([1]);
    expect(b.events.log.map((l) => l.line))
      .toContain('skipping GPU 0 (NVIDIA RTX PRO 4500 Blackwell): not enough free VRAM');
    expect(b.events.error).toEqual([]);
  });

  test('a rig where no card starts reports the first reason and stops', () => {
    const b = fleet();
    b.createCore.mockImplementation(() => { throw new Error('not enough free VRAM'); });
    expect(b.m.start({ ...settings, gpus: GPUS })).toBe(false);
    expect(b.events.error.map((e) => e.message)).toEqual(['not enough free VRAM']);
    expect(b.m.isRunning()).toBe(false);
  });

  // An older pearl_core.node ignores the card we ask for and always opens CUDA's
  // device 0. Starting a second one would stack two searches on that one card
  // and report them as two cards.
  test('stops at one card when the core is too old to place itself', () => {
    const b = fleet({ silentDevice: true });
    expect(b.m.start({ ...settings, gpus: GPUS })).toBe(true);
    expect(b.made).toHaveLength(1);
    expect(b.events.log.map((l) => l.line).join(' ')).toMatch(/predates per-card mining/);
  });

  // No card list at all (no nvidia-smi, or not an NVIDIA rig): one core, no index
  // asked for, and the core picks its own card. Exactly what a single-card rig
  // did before any of this.
  test('falls back to a single self-placing core with no card list', () => {
    const b = fleet();
    b.m.start(settings);
    expect(b.made).toHaveLength(1);
    expect(b.made[0].opts).toEqual({ saltBase: 0, saltStride: 1 });
  });

  // A factory that returns nothing instead of throwing is the same answer: no
  // core. It used to be caught only because wiring a falsy core threw.
  test('treats a core that never materialises as a card that refused', () => {
    const b = fleet();
    b.createCore.mockImplementationOnce(() => null);
    expect(b.m.start({ ...settings, gpus: GPUS })).toBe(true);
    expect(b.m.devices().map((d) => d.index)).toEqual([1]);
    expect(b.events.log.map((l) => l.line).join(' ')).toMatch(/did not initialise/);
  });

  // Thrown strings and objects with no message are not errors we can quote, but
  // the card still has to say why it is not mining.
  test('says something useful when a card refuses without a message', () => {
    const b = fleet();
    b.createCore.mockImplementation(() => { throw 'CUDA_ERROR_NO_DEVICE'; });
    expect(b.m.start({ ...settings, gpus: GPUS })).toBe(false);
    expect(b.events.error.map((e) => e.message)).toEqual(['CUDA_ERROR_NO_DEVICE']);
  });

  // With no card list there is no card to name in the message, and a pinned card
  // nvidia-smi never listed has an index but no name.
  test('names what it can when a card it cannot describe refuses', () => {
    const noList = fleet();
    noList.createCore.mockImplementation(() => { throw new Error('no CUDA device found'); });
    noList.m.start(settings);
    expect(noList.events.log.map((l) => l.line))
      .toContain('skipping the GPU: no CUDA device found');

    const unnamed = fleet();
    unnamed.createCore.mockImplementation(() => { throw new Error('no CUDA device found'); });
    unnamed.m.start({ ...settings, gpus: [{ index: 3, name: null }] });
    expect(unnamed.events.log.map((l) => l.line))
      .toContain('skipping GPU 3: no CUDA device found');
  });

  // A tick with nothing in it must not turn the rig's hashrate into NaN, which
  // the UI would render as a blank where a number belongs.
  test('ignores a hashrate tick that carries no number', () => {
    const b = fleet();
    b.m.start({ ...settings, gpus: GPUS });
    b.made[0].emit('hashrate', 100);
    b.made[1].emit('hashrate', undefined);
    expect(b.m.totalHashrate()).toBe(100);
  });

  // The verdict comes back on the shared socket with only a submit id. Without
  // the card recorded against that id, every share on the rig would be credited
  // to card 0.
  test('credits a share to the card that found it', () => {
    const b = fleet();
    b.m.start({ ...settings, gpus: GPUS });
    b.m.pending.set(7, { jobId: 'j', index: 1 });
    b.sock.emit('data', JSON.stringify({ id: 7, result: true, error: null }) + '\n');
    expect(b.events.share[0]).toEqual({ jobId: 'j', accepted: true, index: 1 });
  });
});
