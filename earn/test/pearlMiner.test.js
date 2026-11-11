'use strict';

// The real gpuClocks runs nvidia-smi as root. A miner built without `clocks`
// falls back to it, so it is mocked for the whole file: a test that forgets to
// inject a fake must still never change a real card's clocks.
jest.mock('../src/main/gpuClocks', () => ({
  lockMemoryClock: jest.fn(() => ({ ok: true, error: null })),
  resetMemoryClock: jest.fn(() => ({ ok: true, error: null })),
}));

const { EventEmitter } = require('events');
const zlib = require('zlib');
const {
  PearlMiner, RECONNECT_MS, RESTART_MS, RESTART_LIMIT, STABLE_MS, SALT_RUN, GZIP_REJECT_LIMIT,
  K_REJECT_LIMIT, K_ANSWER_MS,
} = require('../src/main/pearlMiner');
const gpuClocks = require('../src/main/gpuClocks');
const { encode } = require('../src/shared/miner/stratum');
const { buildShareProof } = require('../src/shared/miner/shareProof');
const { decodePlainProof } = require('../src/shared/miner/plainProof');
const {
  shareBound, PROFILE, buildConfig52, regionToTile, profileForK,
} = require('../src/shared/miner/pearlhash');
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
  // As a real socket does: destroyed at once, closed later.
  s.destroy = jest.fn(() => { s.destroyed = true; });
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
  const events = { log: [], job: [], share: [], rejected: [], hashrate: [], error: [], stopped: [], authorized: [] };
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
  test('connects and authorizes with object params', () => {
    const { m, sock, connect } = boot();
    expect(m.start(settings)).toBe(true);
    expect(m.isRunning()).toBe(true);
    expect(connect).toHaveBeenCalledWith('us.pearl.herominers.com', 1200);

    sock.emit('connect');
    const sent = JSON.parse(sock.written[0]);
    // Asking for compressed proofs (see 'compressed share proofs'), and with no
    // agent given, naming none.
    expect(sent).toEqual({
      id: 1, method: 'mining.authorize', params: { wallet: ADDR, worker: 'rig01', type: 'v2' },
    });
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

  // Kryptex has no merge mining, and answered a login carrying an MDL address
  // with "Invalid login". There the rig mines Pearl alone, and says so.
  test('leaves an MDL address out of a Kryptex login', () => {
    const { m, sock, events } = boot();
    m.start({ ...settings, endpoint: 'prl-us.kryptex.network:7048', mdlAddress: MDL });
    sock.emit('connect');
    expect(JSON.parse(sock.written[0]).params.wallet).toBe(ADDR + '/rig01');
    expect(events.log).toContainEqual({ level: 'warn', line: 'Kryptex has no MDL merge mining: mining Pearl only' });
    m.stop();
    // Without one there is nothing to say.
    const quiet = boot();
    quiet.m.start({ ...settings, endpoint: 'prl-us.kryptex.network:7048', mdlAddress: 'nope' });
    expect(quiet.events.log.map((l) => l.line)).not.toContain('Kryptex has no MDL merge mining: mining Pearl only');
    quiet.m.stop();
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
    expect(events.log[0].line).toMatch(/^No Pearl core loaded — nothing to mine with./);
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
    expect(b.m.pending.get(sent.id)).toEqual({ jobId: '00000000_2097152', index: 1, gzip: false, k: TINY.k });
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

  // Our count has to be the pool's count. A verdict for a submit we never sent,
  // or a second verdict for one, is logged and counted nowhere.
  test('a verdict for an id with no submit waiting is not counted', () => {
    const { core, sock, events } = withJob();
    sock.emit('data', '{"id":9999,"result":true,"error":null}\n');
    sock.emit('data', '{"id":9998,"result":null,"error":[21,"Job not found"]}\n');
    expect(events.share).toHaveLength(0);
    expect(events.rejected).toHaveLength(0);
    expect(events.log.map((l) => l.line)).toEqual(expect.arrayContaining([
      'pool answered share id 9999, which this miner is not waiting on: accepted; not counted',
      'pool answered share id 9998, which this miner is not waiting on: [21] Job not found; not counted',
    ]));

    core.emit('hit', goodHit());
    const id = JSON.parse(sock.written[0]).id;
    sock.emit('data', JSON.stringify({ id, result: true, error: null }) + '\n');
    sock.emit('data', JSON.stringify({ id, result: true, error: null }) + '\n');
    expect(events.share).toHaveLength(1);
  });

  test('a submit answered with no error but result false counts as rejected', () => {
    const { core, sock, events } = withJob();
    core.emit('hit', goodHit());
    const id = JSON.parse(sock.written[0]).id;
    sock.emit('data', JSON.stringify({ id, result: false, error: null }) + '\n');
    expect(events.share).toHaveLength(0);
    expect(events.rejected[0])
      .toEqual({ jobId: '00000000_2097152', reason: 'no reason given (result false)', index: 0 });
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

  // Those hits are work the hashrate counts and the pool never sees, so each is
  // said, with whether the block changed.
  test('a dropped hit for the replaced job is logged with whether the block changed', () => {
    const { core, sock, events } = withJob();
    sock.emit('data', jobLine({ job_id: 'same_block' }));
    core.emit('hit', goodHit('00000000_2097152'));
    sock.emit('data', jobLine({ job_id: 'next_block', height: 999999 }));
    core.emit('hit', goodHit('same_block'));
    core.emit('hit', goodHit('long_gone'));
    const lines = events.log.filter((l) => /dropping a hit for the job replaced/.test(l.line)).map((l) => l.line);
    expect(lines).toHaveLength(2);
    expect(lines[0]).toMatch(/ms ago \(same block\)$/);
    expect(lines[1]).toMatch(/ms ago \(new block\)$/);
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

  // Pools close connections every few minutes. A hit found before the next
  // connection used to be written to the closed socket, which drops it without
  // a word, and its submit waited for an answer that could never come.
  test('a hit found while the pool connection is down is dropped and said', () => {
    const { m, core, sock, events } = withJob();
    core.emit('hit', goodHit());
    expect(m.pending.size).toBe(1);
    sock.written.length = 0;
    sock.emit('close');
    expect(m.pending.size).toBe(0);
    core.emit('hit', goodHit());
    expect(sock.written).toHaveLength(0);
    expect(events.log.map((l) => l.line)).toContain('dropping a hit found while the pool connection is down');
    m.stop();
  });

  // The next connection has jobs of its own. A share of the last one would
  // reach it before its first job, even before its login.
  test('the next connection is not sent a share of the last one\'s job', () => {
    jest.useFakeTimers();
    const socks = [makeSocket(), makeSocket()];
    const core = makeCore();
    const m = new PearlMiner({ connect: jest.fn(() => socks.shift()), createCore: () => core, reconnectMs: 0 });
    const [a, b] = socks;
    m.start({ ...settings, profile: TINY });
    a.emit('connect');
    a.emit('data', jobLine());
    a.emit('close');
    jest.runOnlyPendingTimers();
    b.emit('connect');
    b.written.length = 0;
    core.emit('hit', goodHit());
    expect(b.written).toHaveLength(0);
    // Its own job is mined and submitted as usual.
    b.emit('data', jobLine({ job_id: 'next_job' }));
    core.emit('hit', goodHit('next_job'));
    expect(JSON.parse(b.written[0]).params.job_id).toBe('next_job');
    m.stop();
    jest.useRealTimers();
  });

  // A rig's only core failing is not the end of mining: the card is opened again
  // after a while (see 'restarting a card that stopped'). Until then the failure
  // is in the log, not raised as an error, and the pool connection stays open
  // for the card to come back to.
  test('hashrate is relayed, and a failed only core is dropped without ending mining', () => {
    jest.useFakeTimers();
    try {
      const { m, core, events } = withJob();
      core.emit('hashrate', 296.5);
      core.emit('error', new Error('kernel launch failed'));
      expect(events.hashrate).toEqual([296.5]);
      expect(core.stop).toHaveBeenCalled();
      expect(events.error).toEqual([]);
      expect(events.stopped).toEqual([]);
      expect(m.isRunning()).toBe(true);
      m.stop();
    } finally {
      jest.useRealTimers();
    }
  });
});

// Kryptex's gzip extension as the miner runs it: asked for on every login,
// decided by each connection's own reply, and dropped for the rest of the run
// when a pool that agreed rejects compressed shares and accepts none.
describe('PearlMiner — compressed share proofs', () => {
  const JOB_KEY = hash(Buffer.concat([Buffer.from(HEADER, 'hex'), buildConfig52(TINY)]));
  // Verbatim, 2026-10-20: Kryptex asked with the type, and asked without it.
  const AUTH_V2 = '{"id":1,"result":true,"error":null,"type":"v2"}\n';
  const AUTH_PLAIN = '{"id":1,"result":true,"error":null}\n';
  const FALLBACK_LINE = 'the pool rejected 3 compressed shares and accepted none; '
    + 'reconnecting to send shares uncompressed for the rest of this run';

  // A hit with real proofs over the core's constant operand fill (every byte
  // 48), so it compresses the way a real share does. Built once: the miner
  // does not change a hit, so the same one can be found again and again.
  let theHit = null;
  function hit() {
    if (theHit) return theHit;
    const side = (rows) => {
      const mat = Buffer.alloc(TINY.m * TINY.k, 48);
      const p = M.multiLeafProof(JOB_KEY, mat, M.buildLayers(JOB_KEY, mat), M.leafIndicesFromRows(rows, TINY.k));
      return {
        leafIndices: p.leafIndices, leafData: Buffer.concat(p.leafData),
        siblings: Buffer.concat(p.siblings), root: p.root, totalLeaves: p.totalLeaves,
      };
    };
    const jackpotHash = Buffer.alloc(32);
    jackpotHash[24] = 0x01; // 2^192, comfortably under the scaled bound
    const { rows, cols } = regionToTile(0, TINY);
    theHit = { jobId: '00000000_2097152', jackpotHash, nonce: 0, proofA: side(rows), proofBt: side(cols) };
    return theHit;
  }

  // A miner with a fresh socket for every connection, so a reconnect can be
  // followed from one to the next, and a fresh core for every run, so a hit
  // after a restart is found once, as on a real rig.
  function rig({ agent, endpoint } = {}) {
    const socks = [];
    let core = null;
    const connect = jest.fn(() => { const s = makeSocket(); socks.push(s); return s; });
    const m = new PearlMiner({ connect, createCore: () => (core = makeCore()), reconnectMs: 0, agent });
    const lines = [];
    m.on('log', (l) => lines.push(l));
    m.start({ ...settings, profile: TINY, ...(endpoint ? { endpoint } : {}) });
    const sock = () => socks[socks.length - 1];
    const sent = (method) => sock().written.map((w) => JSON.parse(w)).filter((x) => x.method === method);
    return {
      m, socks, lines, sock,
      get core() { return core; },
      restart() { m.stop(); m.start({ ...settings, profile: TINY }); },
      said: () => lines.map((l) => l.line),
      authorize: () => sent('mining.authorize')[0],
      submits: () => sent('mining.submit'),
      // The newest socket connects, logs in with this reply, and gets a job.
      login(reply) {
        sock().emit('connect');
        sock().emit('data', reply);
        sock().emit('data', jobLine());
      },
      answer(id, ok) {
        sock().emit('data', JSON.stringify(ok ? { id, result: true, error: null }
          : { id, result: null, error: [23, 'invalid share'] }) + '\n');
      },
      // The socket closes, as one the miner destroyed does, and the reconnect runs.
      reconnect() {
        sock().emit('close');
        jest.runOnlyPendingTimers();
      },
    };
  }

  beforeEach(() => jest.useFakeTimers());
  afterEach(() => jest.useRealTimers());

  test('every login asks for them and names the miner, Kryptex\'s wallet/worker included', () => {
    const r = rig({ agent: 'llmjob-earn/0.6.6' });
    r.sock().emit('connect');
    expect(r.authorize().params).toEqual({ wallet: ADDR, worker: 'rig01', agent: 'llmjob-earn/0.6.6', type: 'v2' });
    r.m.stop();

    const kx = rig({ agent: 'llmjob-earn/0.6.6', endpoint: 'prl-us.kryptex.network:7048' });
    kx.sock().emit('connect');
    expect(kx.authorize().params)
      .toEqual({ wallet: ADDR + '/rig01', worker: 'rig01', agent: 'llmjob-earn/0.6.6', type: 'v2' });
    kx.m.stop();
  });

  test('a pool that says v2 is sent the proof gzipped, which unpacks to the plain one', () => {
    const r = rig();
    r.login(AUTH_V2);
    expect(r.m.gzip).toBe(true);
    // "authorized" alone on its line: rigs' health checks match it exactly.
    expect(r.said()).toEqual(expect.arrayContaining(['authorized', 'the pool takes compressed share proofs']));
    r.core.emit('hit', hit());
    const [submit] = r.submits();
    expect(Object.keys(submit.params).sort()).toEqual(['hs', 'job_id', 'plain_proof']);
    expect(zlib.gunzipSync(Buffer.from(submit.params.plain_proof, 'base64')))
      .toEqual(Buffer.from(buildShareProof(hit(), JOB_KEY, TINY), 'base64'));
    expect(r.m.pending.get(submit.id).gzip).toBe(true);
    r.m.stop();
  });

  test('a pool that does not say v2 is sent the plain proof', () => {
    const r = rig();
    r.login(AUTH_PLAIN);
    expect(r.m.gzip).toBe(false);
    expect(r.said()).toContain('authorized');
    r.core.emit('hit', hit());
    expect(r.submits()[0].params.plain_proof).toBe(buildShareProof(hit(), JOB_KEY, TINY));
    expect(r.said().some((l) => /compressed/.test(l))).toBe(false);
    r.m.stop();
  });

  // What one connection agreed says nothing about the next: a pool behind a
  // different address may not offer it, and would reject a compressed proof.
  test('the choice belongs to one connection: cleared on close, asked again, and decided by the new reply', () => {
    const r = rig();
    r.login(AUTH_V2);
    expect(r.m.gzip).toBe(true);
    r.reconnect();
    expect(r.socks).toHaveLength(2);
    expect(r.m.gzip).toBe(false);
    r.sock().emit('connect');
    expect(r.authorize().params.type).toBe('v2');
    r.sock().emit('data', AUTH_PLAIN);
    r.sock().emit('data', jobLine());
    r.core.emit('hit', hit());
    expect(r.submits()[0].params.plain_proof).toBe(buildShareProof(hit(), JOB_KEY, TINY));

    // And back the other way.
    r.reconnect();
    r.login(AUTH_V2);
    r.core.emit('hit', hit());
    expect(zlib.gunzipSync(Buffer.from(r.submits()[0].params.plain_proof, 'base64')))
      .toEqual(Buffer.from(buildShareProof(hit(), JOB_KEY, TINY), 'base64'));

    // A stop lets go of the connection, and its choice with it.
    r.m.stop();
    expect(r.m.gzip).toBe(false);
  });

  test('the sizes are logged on the first compressed share of each connection', () => {
    const r = rig();
    const sizeLines = () => r.said().filter((l) => /^sending compressed share proofs/.test(l));
    r.login(AUTH_V2);
    r.core.emit('hit', hit());
    r.core.emit('hit', hit());
    expect(r.submits()).toHaveLength(2);
    // Whole KB for the plain proof, tenths for the compressed one.
    expect(sizeLines()).toEqual([expect.stringMatching(/^sending compressed share proofs: \d+ KB -> \d\.\d KB$/)]);
    r.reconnect();
    r.login(AUTH_V2);
    r.core.emit('hit', hit());
    expect(sizeLines()).toHaveLength(2);
    r.m.stop();
  });

  test('compressed shares rejected with none accepted: the miner logs in again without asking, for the rest of the run', () => {
    expect(GZIP_REJECT_LIMIT).toBe(3);
    const r = rig();
    r.login(AUTH_V2);
    const first = r.sock();
    for (let i = 0; i < 3; i++) r.core.emit('hit', hit());
    const ids = r.submits().map((s) => s.id);
    r.answer(ids[0], false);
    r.answer(ids[1], false);
    expect(first.destroy).not.toHaveBeenCalled();
    r.answer(ids[2], false);
    expect(first.destroy).toHaveBeenCalledTimes(1);
    expect(r.lines).toContainEqual({ level: 'warn', line: FALLBACK_LINE });

    // The usual reconnect, and a login with no type: plain proofs from here.
    r.reconnect();
    r.sock().emit('connect');
    expect(r.authorize().params).toEqual({ wallet: ADDR, worker: 'rig01' });
    r.sock().emit('data', AUTH_PLAIN);
    r.sock().emit('data', jobLine());
    r.core.emit('hit', hit());
    expect(r.submits()[0].params.plain_proof).toBe(buildShareProof(hit(), JOB_KEY, TINY));

    // Every later connection of this run asks without it too.
    r.reconnect();
    r.sock().emit('connect');
    expect(r.authorize().params).not.toHaveProperty('type');

    // A new run asks again, and counts again: its first rejection is one, not
    // the fourth.
    r.restart();
    r.login(AUTH_V2);
    expect(r.authorize().params.type).toBe('v2');
    r.core.emit('hit', hit());
    r.answer(r.submits()[0].id, false);
    expect(r.sock().destroy).not.toHaveBeenCalled();
    r.m.stop();
  });

  // The fallback lets go of the socket at once, but it stays this.sock until
  // its close event comes. A write to it in between is lost without a word.
  test('a hit found after the fallback lets go of the socket, before it closes, is dropped and said', () => {
    const r = rig();
    r.login(AUTH_V2);
    for (let i = 0; i < 3; i++) r.core.emit('hit', hit());
    for (const s of r.submits()) r.answer(s.id, false);
    expect(r.sock().destroy).toHaveBeenCalledTimes(1);
    const written = r.sock().written.length;
    r.core.emit('hit', hit());
    expect(r.sock().written).toHaveLength(written);
    expect(r.m.pending.size).toBe(0);
    expect(r.said()).toContain('dropping a hit found while the pool connection is down');
    r.m.stop();
  });

  // An accepted plain share says nothing about whether a pool reads
  // compressed ones: a later connection may say v2 and then reject them all.
  test('only an accepted compressed share turns the fallback off', () => {
    const r = rig();
    r.login(AUTH_PLAIN);
    r.core.emit('hit', hit());
    r.answer(r.submits()[0].id, true);
    r.reconnect();
    r.login(AUTH_V2);
    for (let i = 0; i < 3; i++) r.core.emit('hit', hit());
    for (const s of r.submits()) r.answer(s.id, false);
    expect(r.sock().destroy).toHaveBeenCalledTimes(1);
    expect(r.said()).toContain(FALLBACK_LINE);
    r.m.stop();
  });

  // The pool behind the next run's connection may not be the one that read
  // them, so an accepted compressed share is per run too.
  test('a new run can fall back again, whatever the last one accepted', () => {
    const r = rig();
    r.login(AUTH_V2);
    r.core.emit('hit', hit());
    r.answer(r.submits()[0].id, true);
    r.restart();
    r.login(AUTH_V2);
    for (let i = 0; i < 3; i++) r.core.emit('hit', hit());
    for (const s of r.submits()) r.answer(s.id, false);
    expect(r.sock().destroy).toHaveBeenCalledTimes(1);
    expect(r.said()).toContain(FALLBACK_LINE);
    r.m.stop();
  });

  // The socket closes some time after the miner lets go of it, and verdicts
  // already on their way still arrive. They must not close it again.
  test('rejections arriving before the close do not close it twice', () => {
    const r = rig();
    r.login(AUTH_V2);
    for (let i = 0; i < 4; i++) r.core.emit('hit', hit());
    const reject = (s) => JSON.stringify({ id: s.id, result: null, error: [23, 'invalid share'] });
    r.sock().emit('data', r.submits().map(reject).join('\n') + '\n');
    expect(r.sock().destroy).toHaveBeenCalledTimes(1);
    expect(r.said().filter((l) => l === FALLBACK_LINE)).toHaveLength(1);
    r.m.stop();
  });

  test('one accepted compressed share shows the pool reads them, for the rest of the run', () => {
    const r = rig();
    r.login(AUTH_V2);
    for (let i = 0; i < 5; i++) r.core.emit('hit', hit());
    const ids = r.submits().map((s) => s.id);
    r.answer(ids[0], false);
    r.answer(ids[1], true);
    for (const id of ids.slice(2)) r.answer(id, false);
    expect(r.sock().destroy).not.toHaveBeenCalled();

    // Still so on the next connection.
    r.reconnect();
    r.login(AUTH_V2);
    for (let i = 0; i < 3; i++) r.core.emit('hit', hit());
    for (const s of r.submits()) r.answer(s.id, false);
    expect(r.sock().destroy).not.toHaveBeenCalled();
    expect(r.said()).not.toContain(FALLBACK_LINE);
    r.m.stop();
  });

  test('rejected plain shares say nothing about compression', () => {
    const r = rig();
    r.login(AUTH_PLAIN);
    for (let i = 0; i < 3; i++) r.core.emit('hit', hit());
    for (const s of r.submits()) r.answer(s.id, false);
    expect(r.sock().destroy).not.toHaveBeenCalled();
    expect(r.m.askGzip).toBe(true);
    r.m.stop();
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

  test('a rejection for an unknown id is not counted, and does not throw', () => {
    const b = boot();
    b.m.start(settings);
    b.sock.emit('connect');
    b.sock.emit('data', '{"id":4242,"result":null,"error":[21,"Job not found"]}\n');
    expect(b.events.rejected).toHaveLength(0);
  });

  // A pool error with no id answers no share, but it is how a pool says why it
  // is dropping the worker, so the reason has to reach the log.
  test('a pool error with no share id is logged with its reason, and not counted', () => {
    const b = boot();
    b.m.start(settings);
    b.sock.emit('connect');
    b.sock.emit('data', '{"id":null,"result":null,"error":[24,"Unauthorized worker"]}\n');
    expect(b.events.rejected).toHaveLength(0);
    expect(b.events.log).toContainEqual({
      level: 'error', line: 'pool answered with no share id: [24] Unauthorized worker; not counted',
    });
  });

  test('a verdict with no error detail still renders a reason', () => {
    const b = boot();
    b.m.start(settings);
    b.sock.emit('connect');
    b.m.pending.set(4243, { jobId: 'j', index: 0 });
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

  // A pool out of reach is retried every 5 s. Each retry used to log the socket
  // error and the close again, so a rig with no DNS filled its log with the same
  // two lines. Every retry still reports connect-failed, which main's one-time
  // DNS hint reads.
  test('a pool out of reach is said once, not on every retry', () => {
    jest.useFakeTimers();
    const socks = [];
    const connect = jest.fn(() => { const s = makeSocket(); socks.push(s); return s; });
    const m = new PearlMiner({ connect, createCore: () => makeCore(), reconnectMs: 0 });
    const lines = [];
    const failed = [];
    m.on('log', (l) => lines.push(l.line));
    m.on('connect-failed', (e) => failed.push(e));
    m.start(settings);
    const fail = (message) => {
      const s = socks[socks.length - 1];
      s.emit('error', new Error(message));
      s.emit('close');
      jest.runOnlyPendingTimers();
    };
    fail('getaddrinfo ENOTFOUND pool');
    fail('getaddrinfo ENOTFOUND pool');
    fail('getaddrinfo ENOTFOUND pool');
    expect(connect).toHaveBeenCalledTimes(4);
    expect(failed).toHaveLength(3);
    expect(lines.filter((l) => /^pool/.test(l))).toEqual([
      'pool socket error: getaddrinfo ENOTFOUND pool',
      'pool connection closed; reconnecting (until it connects, the retries are not logged)',
    ]);
    // A different error is news.
    fail('connect ECONNREFUSED');
    expect(lines.filter((l) => /ECONNREFUSED/.test(l))).toHaveLength(1);

    // Once the pool takes the login, the next drop is said again.
    lines.length = 0;
    const live = socks[socks.length - 1];
    live.emit('connect');
    live.emit('data', '{"id":1,"error":null,"result":true}\n');
    fail('connect ECONNREFUSED');
    fail('connect ECONNREFUSED');
    expect(lines.filter((l) => /^pool/.test(l))).toEqual([
      'pool socket error: connect ECONNREFUSED',
      'pool connection closed; reconnecting',
      'pool connection closed; reconnecting (until it connects, the retries are not logged)',
    ]);
    m.stop();
    jest.useRealTimers();
  });

  // stop() lets go of its socket, which closes later. A start in between has a
  // socket of its own, and that late close used to log it out and open a third.
  test('a late close from a socket stop() let go of leaves the next run alone', () => {
    jest.useFakeTimers();
    const a = makeSocket();
    const b = makeSocket();
    const connect = jest.fn().mockReturnValueOnce(a).mockReturnValueOnce(b);
    const m = new PearlMiner({ connect, createCore: () => makeCore(), reconnectMs: 0 });
    m.start(settings);
    m.stop();
    m.start(settings);
    // Nor is anything it still delivers heard.
    a.emit('data', '{"id":1,"error":null,"result":true}\n');
    expect(m.authorized).toBe(false);
    b.emit('connect');
    b.emit('data', '{"id":1,"error":null,"result":true}\n');
    a.emit('close');
    jest.runOnlyPendingTimers();
    expect(connect).toHaveBeenCalledTimes(2);
    expect(m.authorized).toBe(true);
    expect(m.sock).toBe(b);
    m.stop();
    jest.useRealTimers();
  });

  // A socket that closes inside destroy() closes while stop() runs.
  test('a socket that closes inside destroy does not reconnect', () => {
    jest.useFakeTimers();
    const sock = makeSocket();
    sock.destroy = jest.fn(() => sock.emit('close'));
    const { m, connect, events } = boot({ sock });
    m.start(settings);
    m.stop();
    jest.runOnlyPendingTimers();
    expect(connect).toHaveBeenCalledTimes(1);
    expect(events.log.map((l) => l.line)).not.toContain('pool connection closed; reconnecting');
    jest.useRealTimers();
  });

  // Half a line from the closed connection must not be glued onto the first
  // line of the next one: that line is usually the login's answer.
  test('a partial line from a closed connection is dropped', () => {
    jest.useFakeTimers();
    const a = makeSocket();
    const b = makeSocket();
    const connect = jest.fn().mockReturnValueOnce(a).mockReturnValueOnce(b);
    const m = new PearlMiner({ connect, createCore: () => makeCore(), reconnectMs: 0 });
    m.start(settings);
    a.emit('connect');
    a.emit('data', '{"id":null,"method":"mining.no');
    a.emit('close');
    jest.runOnlyPendingTimers();
    b.emit('connect');
    b.emit('data', '{"id":1,"error":null,"result":true}\n');
    expect(m.authorized).toBe(true);
    m.stop();
    jest.useRealTimers();
  });

  // A pool that rejects with a code but no text must still render a reason
  // rather than "[21] undefined".
  test('a coded rejection with no message renders just the code', () => {
    const b = boot();
    b.m.start(settings);
    b.sock.emit('connect');
    b.m.pending.set(5, { jobId: 'j', index: 0 });
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

// The Intel core says how much memory its card has and how much of it the core
// has allocated: core.device.memTotalMb and memUsedMb. The network board has no
// other source for an Intel card, so the miner hands both on with the card. The
// CUDA and AMD cores say neither, and their devices stay { index, name }.
describe('PearlMiner — a card\'s memory, as its core reports it', () => {
  const ARC = 'Intel(R) Arc(TM) A750 Graphics';
  const GPUS = [{ index: 0, name: 'Intel Arc A750' }, { index: 1, name: 'Intel Arc A750' }];

  beforeEach(() => jest.useFakeTimers());
  afterEach(() => {
    jest.clearAllTimers();
    jest.useRealTimers();
  });

  const withDevice = (device) => {
    const core = makeCore();
    core.device = device;
    const b = boot({ core });
    b.m.start(settings);
    return b;
  };

  // Two Arc cards. Each core reports its card, with `used` MB allocated on it:
  // one figure for each time a card is opened, the last one repeating.
  function rig(used = [1536]) {
    const made = [];
    const sock = makeSocket();
    const createCore = jest.fn((profile, opts) => {
      const c = makeCore();
      c.device = {
        index: opts.deviceIndex, name: ARC, fold: 'xmx8',
        memTotalMb: 8128, memUsedMb: used[Math.min(made.length, used.length - 1)],
      };
      made.push(c);
      return c;
    });
    const m = new PearlMiner({ connect: () => sock, createCore, reconnectMs: 0, restartMs: 10 });
    const ticks = [];
    m.on('hashrate', (th, device) => ticks.push([th, device]));
    m.on('log', () => {});
    m.on('error', () => {});
    m.start({ ...settings, gpus: GPUS });
    return { m, made, ticks };
  }

  test('is read with the card, in whole MB, and handed on with its hashrate', () => {
    const b = withDevice({ index: 0, name: ARC, fold: 'xmx8', memTotalMb: 8127.6, memUsedMb: 1535.5 });
    const ticks = [];
    b.m.on('hashrate', (th, device) => ticks.push([th, device]));
    expect(b.m.devices()).toEqual([{ index: 0, name: ARC, memTotalMb: 8128, memUsedMb: 1536 }]);
    b.core.emit('hashrate', 41.9);
    expect(ticks).toEqual([[41.9, { index: 0, name: ARC, memTotalMb: 8128, memUsedMb: 1536 }]]);
    b.m.stop();
  });

  // No size, no figures: a `used` alone says nothing anyone can show, and a
  // device without both keys is what every other core's card looks like.
  test('a core that gives no usable size reports no memory at all', () => {
    for (const memTotalMb of [undefined, null, 0, -1, NaN, Infinity, '8128']) {
      const b = withDevice({ index: 0, name: ARC, memTotalMb, memUsedMb: 1536 });
      const [device] = b.m.devices();
      expect(device).toEqual({ index: 0, name: ARC });
      expect(device).not.toHaveProperty('memTotalMb');
      expect(device).not.toHaveProperty('memUsedMb');
      b.m.stop();
    }
  });

  test('a size with no usable `used` beside it reads 0 used', () => {
    for (const memUsedMb of [undefined, null, 0, -5, NaN, 'lots']) {
      const b = withDevice({ index: 0, name: ARC, memTotalMb: 8128, memUsedMb });
      expect(b.m.devices()).toEqual([{ index: 0, name: ARC, memTotalMb: 8128, memUsedMb: 0 }]);
      b.m.stop();
    }
  });

  // A closed core has given the card's memory back. The hashrate of 0 that
  // says the card stopped says that too, or the board would show a card that
  // is not mining as still holding what it mined with.
  test('a card set aside for a prompt reads 0 used, and its new core\'s figure once it is back', () => {
    const b = rig([1536, 1536, 1792]);
    b.m.pauseCards([0]);
    expect(b.ticks).toEqual([[0, { index: 0, name: ARC, memTotalMb: 8128, memUsedMb: 0 }]]);
    // The other card is still mining with what it had.
    b.made[1].emit('hashrate', 41.8);
    expect(b.ticks.pop()).toEqual([41.8, { index: 1, name: ARC, memTotalMb: 8128, memUsedMb: 1536 }]);
    b.m.resumeCards([0]);
    b.made[2].emit('hashrate', 41.9);
    expect(b.ticks.pop()).toEqual([41.9, { index: 0, name: ARC, memTotalMb: 8128, memUsedMb: 1792 }]);
    b.m.stop();
  });

  test('a card whose core failed reads 0 used until it is opened again', () => {
    const b = rig();
    b.made[0].emit('error', new Error('GPU 0: device lost'));
    expect(b.ticks).toEqual([[0, { index: 0, name: ARC, memTotalMb: 8128, memUsedMb: 0 }]]);
    jest.advanceTimersByTime(10);
    b.made[2].emit('hashrate', 41.9);
    expect(b.ticks.pop()).toEqual([41.9, { index: 0, name: ARC, memTotalMb: 8128, memUsedMb: 1536 }]);
    b.m.stop();
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
      { saltBase: 0, saltStride: 2, deviceIndex: 0, cardK: true, checkHits: true },
      { saltBase: 1, saltStride: 2, deviceIndex: 1, cardK: true, checkHits: true },
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

  // An AMD rig whose OS lists more cards than HIP has (shared/gpu.planAmdGpus
  // plans from the OS's count): the extra index fails in the core, on its own,
  // with no restart, and the card HIP has mines.
  test('mines on the AMD cards HIP has when the OS listed one more', () => {
    const b = fleet();
    const real = b.createCore.getMockImplementation();
    b.createCore.mockImplementation((profile, opts) => {
      if (opts.deviceIndex === 1) {
        throw new Error('GPU 1 was asked for (PEARL_GPU_INDEX) but this machine has 1: valid indices are 0..0');
      }
      return real(profile, opts);
    });
    expect(b.m.start({ ...settings, gpus: [{ index: 0, name: null }, { index: 1, name: null }] })).toBe(true);
    expect(b.m.devices().map((d) => d.index)).toEqual([0]);
    // The prompt pause counts one mining card, not the two planned.
    expect(b.m.cardCount()).toBe(1);
    expect(b.made[0].opts).toMatchObject({ saltBase: 0, saltStride: 2, deviceIndex: 0 });
    expect(b.events.log).toContainEqual({
      level: 'info', line: 'skipping GPU 1: GPU 1 was asked for (PEARL_GPU_INDEX) but this machine has 1: valid indices are 0..0',
    });
    expect(b.events.log.map((l) => l.line).join(' ')).not.toMatch(/restarting/);
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
    expect(b.made[0].opts).toEqual({ saltBase: 0, saltStride: 1, cardK: true, checkHits: true });
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

// A core whose search fails (a CUDA fault such as the Windows Blackwell
// "misaligned address") never searches again. Leaving it in the rig meant the
// next pool job called setJob on it, the native core started a second search
// thread over the finished one, std::thread's assignment called std::terminate,
// and on Windows the whole app closed a few seconds after the error. The host
// now drops that card and carries on with the others.
describe('PearlMiner — a core that fails mid-run', () => {
  const GPUS = [
    { index: 0, name: 'NVIDIA GeForce RTX 5090' },
    { index: 1, name: 'NVIDIA GeForce RTX 4090' },
  ];
  const MISALIGNED = 'CUDA error during search: misaligned address';

  // A failed card's restart is a timer; fake ones keep a test from leaving a
  // real 30 s timer behind, and let a test run the restart when it wants to.
  beforeEach(() => jest.useFakeTimers());
  afterEach(() => {
    jest.clearAllTimers();
    jest.useRealTimers();
  });

  // One fresh core per card, reporting the card it was asked for (the real
  // addon's behaviour), and a clocks double that records the order of every
  // lock, reset and 'stopped'.
  function rig(over = {}) {
    const made = [];
    const calls = [];
    const sock = makeSocket();
    const clocks = {
      lockMemoryClock: jest.fn((index, mhz) => { calls.push(['lock', index, mhz]); return { ok: true, error: null }; }),
      resetMemoryClock: jest.fn((index) => { calls.push(['reset', index]); return { ok: true, error: null }; }),
    };
    const createCore = jest.fn((profile, opts) => {
      const c = makeCore();
      if (!over.silentDevice) {
        c.device = { index: opts.deviceIndex, name: GPUS[opts.deviceIndex].name };
      }
      made.push(c);
      return c;
    });
    const m = new PearlMiner({ connect: () => sock, createCore, reconnectMs: 0, clocks });
    const events = { log: [], hashrate: [], error: [], stopped: [] };
    for (const k of Object.keys(events)) m.on(k, (...a) => events[k].push(a.length > 1 ? a : a[0]));
    m.on('stopped', () => calls.push(['stopped']));
    const lines = () => events.log.map((l) => l.line);
    return { m, sock, made, calls, clocks, events, lines };
  }

  // A hashrate sample the dead core queued before failing must not bring the
  // card back into the rig's total.
  test('ignores a late hashrate sample from a core it already dropped', () => {
    const b = rig();
    b.m.start({ ...settings, gpus: GPUS });
    b.made[0].emit('hashrate', 107);
    b.made[1].emit('hashrate', 60);
    b.made[0].emit('error', new Error('GPU 0: ' + MISALIGNED));
    b.events.hashrate.length = 0;
    b.made[0].emit('hashrate', 107);
    expect(b.m.totalHashrate()).toBe(60);
    expect(b.m.hashrate).toBe(60);
    expect(b.events.hashrate).toEqual([]);
  });

  test('drops the failed card, says so once, and keeps mining on the other', () => {
    const b = rig();
    b.m.start({ ...settings, gpus: GPUS });
    b.sock.emit('connect');
    b.sock.emit('data', jobLine());
    b.made[0].emit('hashrate', 107);
    b.made[1].emit('hashrate', 60);
    b.events.hashrate.length = 0;

    // As the native core sends it: already naming its card.
    b.made[0].emit('error', new Error('GPU 0: ' + MISALIGNED));

    expect(b.made[0].stop).toHaveBeenCalledTimes(1);
    expect(b.made[1].stop).not.toHaveBeenCalled();
    expect(b.m.devices()).toEqual([{ index: 1, name: 'NVIDIA GeForce RTX 4090' }]);
    expect(b.m.isRunning()).toBe(true);
    // Logged once, with the card named once, and not escalated to an error:
    // the rig is still mining.
    expect(b.events.log.filter((l) => /stopped:/.test(l.line))).toEqual([{
      level: 'error',
      line: 'GPU 0 (NVIDIA GeForce RTX 5090) stopped: ' + MISALIGNED,
    }]);
    expect(b.events.error).toEqual([]);
    expect(b.events.stopped).toEqual([]);
    // Its hashrate leaves the rig total, and the UI hears it drop to zero.
    expect(b.m.totalHashrate()).toBe(60);
    expect(b.m.hashrate).toBe(60);
    expect(b.events.hashrate).toEqual([[0, { index: 0, name: 'NVIDIA GeForce RTX 5090' }]]);

    // The next job goes to the card that is left, and never to the failed one:
    // that setJob is what used to terminate the process.
    b.sock.emit('data', jobLine({ job_id: '00000000_2097153' }));
    expect(b.made[0].setJob).toHaveBeenCalledTimes(1);
    expect(b.made[1].setJob).toHaveBeenCalledTimes(2);

    // And stopping the rig later stops only what is still running.
    b.m.stop();
    expect(b.made[0].stop).toHaveBeenCalledTimes(1);
    expect(b.made[1].stop).toHaveBeenCalledTimes(1);
  });

  // With a memory clock plan the failed card is locked at the mining clock. It
  // is not mining any more, so it is released now, not when the rest of the rig
  // stops -- and not released a second time then.
  test('releases the failed card\'s memory clock at once, and only once', () => {
    const b = rig();
    b.m.start({ ...settings, gpus: GPUS, mineMemClockByIndex: { 0: 7001, 1: 7001 } });
    expect(b.calls).toEqual([['lock', 0, 7001], ['lock', 1, 7001]]);

    b.made[1].emit('error', new Error('GPU 1: ' + MISALIGNED));
    expect(b.calls.slice(2)).toEqual([['reset', 1]]);
    expect(b.lines()).toContain('memory clock released on GPU 1');

    b.m.stop();
    expect(b.calls.slice(3)).toEqual([['reset', 0], ['stopped']]);
  });

  // Every card left off for good is the same end as a start with no core: an
  // error that says why, then 'stopped', so the app shows mining has stopped
  // rather than a pool connection that mines nothing. Locks are released before
  // 'stopped', because the demand gate starts the LLM the moment it hears it.
  test('when the last card is left off, reports it and stops, releasing every lock first', () => {
    const b = rig();
    b.m.start({ ...settings, gpus: GPUS, mineMemClockByIndex: { 0: 7001, 1: 7001 } });
    b.sock.emit('connect');
    // From here on, no card opens again.
    const gone = new Error('no CUDA device found');
    b.m.createCore = jest.fn(() => { throw gone; });

    b.made[0].emit('error', new Error('GPU 0: ' + MISALIGNED));
    jest.advanceTimersByTime(RESTART_MS * RESTART_LIMIT);
    expect(b.lines()).toContain('leaving GPU 0 (NVIDIA GeForce RTX 5090) off: it stopped again within 10 '
      + 'minutes of each of 3 restarts. Stop and start mining to try it again.');
    // GPU 1 is still mining, so the rig is.
    expect(b.m.isRunning()).toBe(true);
    expect(b.events.error).toEqual([]);

    b.made[1].emit('error', new Error('GPU 1: ' + MISALIGNED));
    jest.advanceTimersByTime(RESTART_MS * (RESTART_LIMIT - 1));
    // Waiting on its last restart, the rig is still up.
    expect(b.m.isRunning()).toBe(true);
    jest.advanceTimersByTime(RESTART_MS);
    expect(b.m.createCore).toHaveBeenCalledTimes(2 * RESTART_LIMIT);
    // The error is why the card could not come back.
    expect(b.events.error).toEqual([gone]);
    // Marked as a failure, which PearlEngine turns into a non-zero exit code.
    expect(b.events.stopped).toEqual([{ failed: true }]);
    expect(b.m.isRunning()).toBe(false);
    expect(b.m.devices()).toEqual([]);
    expect(b.sock.destroy).toHaveBeenCalled();
    expect(b.calls.slice(2)).toEqual([['reset', 0], ['reset', 1], ['stopped']]);
    expect(b.lines()).toEqual(expect.arrayContaining([
      'GPU 0 (NVIDIA GeForce RTX 5090) stopped: ' + MISALIGNED,
      'GPU 1 (NVIDIA GeForce RTX 4090) stopped: ' + MISALIGNED,
      'could not restart GPU 1 (NVIDIA GeForce RTX 4090): no CUDA device found',
    ]));
  });

  // The native side can deliver one more error after the host has dropped the
  // core, and a stopped miner can hear one too. Neither is news.
  test('ignores an error from a core it has already dropped', () => {
    const b = rig();
    b.m.start({ ...settings, gpus: GPUS });
    b.made[0].emit('error', new Error('GPU 0: ' + MISALIGNED));
    b.made[0].emit('error', new Error('GPU 0: ' + MISALIGNED));
    expect(b.lines().filter((l) => /stopped:/.test(l))).toHaveLength(1);
    expect(b.made[0].stop).toHaveBeenCalledTimes(1);

    b.m.stop();
    b.made[1].emit('error', new Error('GPU 1: ' + MISALIGNED));
    expect(b.events.error).toEqual([]);
    // A stop somebody asked for is not a failure.
    expect(b.events.stopped).toEqual([{}]);
  });

  // A core that throws on stop is already gone; that must not stop the host
  // from dropping it and carrying on.
  test('tolerates a failed core that throws on stop', () => {
    const b = rig();
    b.m.start({ ...settings, gpus: GPUS });
    b.made[0].stop.mockImplementation(() => { throw new Error('already stopped'); });
    b.made[0].emit('error', new Error('GPU 0: ' + MISALIGNED));
    expect(b.m.devices().map((d) => d.index)).toEqual([1]);
    expect(b.m.isRunning()).toBe(true);
  });

  // An older core does not prefix its errors, and a core that will not name
  // its card cannot be named in the log either.
  test('quotes an unprefixed message as it is, and names no card it cannot', () => {
    const b = rig();
    b.m.start({ ...settings, gpus: GPUS });
    b.made[1].emit('error', new Error('an illegal memory access was encountered'));
    expect(b.lines()).toContain(
      'GPU 1 (NVIDIA GeForce RTX 4090) stopped: an illegal memory access was encountered',
    );

    const old = rig({ silentDevice: true });
    old.m.start(settings);
    old.made[0].emit('error', 'CUDA_ERROR_LAUNCH_FAILED');
    expect(old.lines()).toContain('the GPU stopped: CUDA_ERROR_LAUNCH_FAILED');
    expect(old.lines()).toContain('restarting the GPU in 30 s');
    expect(old.events.hashrate).toEqual([]);
  });
});

// A card whose search failed is opened again after RESTART_MS, the way a user
// would by stopping and starting mining, but for that card alone.
describe('PearlMiner — restarting a card that stopped', () => {
  const GPUS = [
    { index: 0, name: 'NVIDIA RTX PRO 4500 Blackwell' },
    { index: 1, name: 'NVIDIA GeForce RTX 4070' },
  ];
  const ILLEGAL = 'CUDA error during search: an illegal instruction was encountered';

  beforeEach(() => jest.useFakeTimers());
  afterEach(() => {
    jest.clearAllTimers();
    jest.useRealTimers();
  });

  function rig() {
    const made = [];
    const sock = makeSocket();
    const clocks = {
      lockMemoryClock: jest.fn(() => ({ ok: true, error: null })),
      resetMemoryClock: jest.fn(() => ({ ok: true, error: null })),
    };
    const createCore = jest.fn((profile, opts) => {
      const c = makeCore();
      c.device = { index: opts.deviceIndex, name: GPUS[opts.deviceIndex].name };
      made.push(c);
      return c;
    });
    const m = new PearlMiner({ connect: () => sock, createCore, reconnectMs: 0, clocks });
    const events = { log: [], hashrate: [], error: [], stopped: [] };
    for (const k of Object.keys(events)) m.on(k, (...a) => events[k].push(a.length > 1 ? a : a[0]));
    const lines = () => events.log.map((l) => l.line);
    return { m, sock, made, clocks, createCore, events, lines };
  }

  // The newest core made for a card.
  const coreFor = (b, index) => b.made.filter((c) => c.device.index === index).pop();

  test('opens the card again after a wait, on its own card, and hands it the current job', () => {
    const b = rig();
    b.m.start({ ...settings, gpus: GPUS, mineMemClockByIndex: { 0: 7001, 1: 7001 } });
    b.sock.emit('connect');
    b.sock.emit('data', jobLine());
    const first = b.made[1];
    first.emit('error', new Error('GPU 1: ' + ILLEGAL));
    expect(b.lines()).toContain('restarting GPU 1 (NVIDIA GeForce RTX 4070) in 30 s');

    // Nothing until the wait is over.
    jest.advanceTimersByTime(RESTART_MS - 1);
    expect(b.createCore).toHaveBeenCalledTimes(2);
    jest.advanceTimersByTime(1);
    expect(b.createCore).toHaveBeenCalledTimes(3);

    // The same card, in the same lane, on salts its first run never reaches.
    expect(b.createCore.mock.calls[2][1]).toEqual({
      deviceIndex: 1, saltStride: 2, saltBase: 1 + 2 * SALT_RUN, cardK: true, checkHits: true,
    });
    const again = coreFor(b, 1);
    expect(again).not.toBe(first);
    expect(b.m.devices()).toEqual([GPUS[0], GPUS[1]]);
    expect(b.lines().filter((l) => l === 'mining on GPU 1 · NVIDIA GeForce RTX 4070')).toHaveLength(2);
    // Locked again for mining, after the release at the failure.
    expect(b.clocks.lockMemoryClock.mock.calls).toEqual([[0, 7001], [1, 7001], [1, 7001]]);
    // The job the rig is on, so it searches now rather than at the next job.
    expect(again.setJob).toHaveBeenCalledTimes(1);
    expect(again.setJob.mock.calls[0][0].jobId).toBe('00000000_2097152');

    // It counts again: its hashrate is in the rig total.
    b.made[0].emit('hashrate', 180);
    again.emit('hashrate', 115);
    expect(b.m.totalHashrate()).toBe(295);
    // And the next job reaches it like any other card.
    b.sock.emit('data', jobLine({ job_id: '00000000_2097153' }));
    expect(again.setJob).toHaveBeenCalledTimes(2);
    expect(first.setJob).toHaveBeenCalledTimes(1);
  });

  // With the last card down, the rig waits for it with the pool connection open.
  // A job that arrives in the meantime is the one the card is handed.
  test('restarts the only card, keeping the pool connection, on the latest job', () => {
    const b = rig();
    b.m.start({ ...settings, gpus: [GPUS[1]] });
    b.sock.emit('connect');
    b.sock.emit('data', jobLine());
    b.made[0].emit('error', new Error('GPU 1: ' + ILLEGAL));
    expect(b.m.isRunning()).toBe(true);
    expect(b.m.devices()).toEqual([]);
    expect(b.sock.destroy).not.toHaveBeenCalled();

    b.sock.emit('data', jobLine({ job_id: '00000000_2097153' }));
    jest.advanceTimersByTime(RESTART_MS);
    expect(b.createCore.mock.calls[1][1]).toEqual({ deviceIndex: 1, saltStride: 1, saltBase: SALT_RUN, cardK: true, checkHits: true });
    expect(b.made[1].setJob.mock.calls[0][0].jobId).toBe('00000000_2097153');
    expect(b.events.error).toEqual([]);
    expect(b.events.stopped).toEqual([]);
  });

  // A core searches nothing until it has a job, and there is none to give it
  // yet; it gets the first one when the pool sends it.
  test('a card restarted before any job waits for the first one', () => {
    const b = rig();
    b.m.start({ ...settings, gpus: GPUS });
    b.made[0].emit('error', new Error('GPU 0: ' + ILLEGAL));
    jest.advanceTimersByTime(RESTART_MS);
    const again = coreFor(b, 0);
    expect(again.setJob).not.toHaveBeenCalled();
    b.sock.emit('connect');
    b.sock.emit('data', jobLine());
    expect(again.setJob).toHaveBeenCalledTimes(1);
  });

  // Each run gets salts of its own, however many times the card comes back.
  test('every run of a card searches its own salts', () => {
    const b = rig();
    b.m.start({ ...settings, gpus: GPUS });
    for (let run = 1; run <= RESTART_LIMIT; run++) {
      coreFor(b, 0).emit('error', new Error('GPU 0: ' + ILLEGAL));
      jest.advanceTimersByTime(RESTART_MS);
    }
    const bases = b.createCore.mock.calls
      .filter(([, o]) => o.deviceIndex === 0)
      .map(([, o]) => o.saltBase);
    expect(bases).toEqual([0, 2 * SALT_RUN, 4 * SALT_RUN, 6 * SALT_RUN]);
  });

  // A card that cannot mine for ten minutes at a time is not having a hiccup.
  // Leave it off, say why and how to try again, and keep the rest mining.
  test('leaves a card off once it keeps failing soon after coming back', () => {
    const b = rig();
    b.m.start({ ...settings, gpus: GPUS });
    for (let i = 0; i < RESTART_LIMIT; i++) {
      coreFor(b, 1).emit('error', new Error('GPU 1: ' + ILLEGAL));
      jest.advanceTimersByTime(RESTART_MS);
    }
    expect(b.createCore).toHaveBeenCalledTimes(2 + RESTART_LIMIT);

    // Fails again a minute after its third restart: that is the limit.
    jest.advanceTimersByTime(60 * 1000);
    coreFor(b, 1).emit('error', new Error('GPU 1: ' + ILLEGAL));
    expect(b.lines()).toContain('leaving GPU 1 (NVIDIA GeForce RTX 4070) off: it stopped again within 10 '
      + 'minutes of each of 3 restarts. Stop and start mining to try it again.');
    jest.advanceTimersByTime(STABLE_MS);
    expect(b.createCore).toHaveBeenCalledTimes(2 + RESTART_LIMIT);
    expect(b.m.devices()).toEqual([GPUS[0]]);
    expect(b.m.isRunning()).toBe(true);
    expect(b.events.error).toEqual([]);
  });

  // The other card waiting for its restart is still one of the run's cards, so
  // one card left off for good does not end the run and clear that restart.
  test('a card left off while the other waits for its restart does not end the run', () => {
    const b = rig();
    b.m.start({ ...settings, gpus: GPUS });
    for (let i = 0; i < RESTART_LIMIT; i++) {
      coreFor(b, 1).emit('error', new Error('GPU 1: ' + ILLEGAL));
      jest.advanceTimersByTime(RESTART_MS);
    }
    coreFor(b, 0).emit('error', new Error('GPU 0: ' + ILLEGAL));
    coreFor(b, 1).emit('error', new Error('GPU 1: ' + ILLEGAL));
    expect(b.lines()).toContain('leaving GPU 1 (NVIDIA GeForce RTX 4070) off: it stopped again within 10 '
      + 'minutes of each of 3 restarts. Stop and start mining to try it again.');
    expect([b.m.isRunning(), b.events.error, b.events.stopped]).toEqual([true, [], []]);
    jest.advanceTimersByTime(RESTART_MS);
    expect(b.createCore).toHaveBeenCalledTimes(3 + RESTART_LIMIT);
    expect(b.m.devices()).toEqual([GPUS[0]]);
    expect(b.m.isRunning()).toBe(true);
  });

  // A card that mined ten minutes since it came back was healthy, so a fault
  // after that starts the count over rather than adding to it. A rig whose card
  // faults every hour or so keeps getting it back.
  test('a card that mined a while since its last restart starts the count over', () => {
    const b = rig();
    b.m.start({ ...settings, gpus: GPUS });
    for (let i = 0; i < 2 * RESTART_LIMIT; i++) {
      jest.advanceTimersByTime(STABLE_MS);
      coreFor(b, 1).emit('error', new Error('GPU 1: ' + ILLEGAL));
      jest.advanceTimersByTime(RESTART_MS);
    }
    expect(b.createCore).toHaveBeenCalledTimes(2 + 2 * RESTART_LIMIT);
    expect(b.lines().some((l) => /^leaving/.test(l))).toBe(false);
    expect(b.m.devices()).toEqual([GPUS[0], GPUS[1]]);
  });

  // A card that will not open is a failed restart, and waits for the next one.
  test('a restart that cannot open the card says why and tries again', () => {
    const b = rig();
    b.m.start({ ...settings, gpus: GPUS });
    const real = b.createCore.getMockImplementation();
    b.m.createCore = jest.fn()
      .mockImplementationOnce(() => null)
      .mockImplementation(real);
    b.made[1].emit('error', new Error('GPU 1: ' + ILLEGAL));
    jest.advanceTimersByTime(RESTART_MS);
    expect(b.lines()).toContain('could not restart GPU 1 (NVIDIA GeForce RTX 4070): the Pearl core did not initialise');
    expect(b.lines().filter((l) => l === 'restarting GPU 1 (NVIDIA GeForce RTX 4070) in 30 s')).toHaveLength(2);
    jest.advanceTimersByTime(RESTART_MS);
    expect(b.m.devices()).toEqual([GPUS[0], GPUS[1]]);

    // A factory that throws something that is not an Error is quoted as it is.
    b.m.createCore = jest.fn(() => { throw 'CUDA_ERROR_NO_DEVICE'; });
    coreFor(b, 0).emit('error', new Error('GPU 0: ' + ILLEGAL));
    jest.advanceTimersByTime(RESTART_MS);
    expect(b.lines()).toContain('could not restart GPU 0 (NVIDIA RTX PRO 4500 Blackwell): CUDA_ERROR_NO_DEVICE');
  });

  // A stop somebody asked for cancels a restart that has not happened yet.
  test('stop() cancels a pending restart', () => {
    const b = rig();
    b.m.start({ ...settings, gpus: GPUS });
    b.made[1].emit('error', new Error('GPU 1: ' + ILLEGAL));
    b.m.stop();
    jest.advanceTimersByTime(RESTART_MS * 10);
    expect(b.createCore).toHaveBeenCalledTimes(2);
    expect(b.events.stopped).toEqual([{}]);
  });

  test('the wait is configurable, and defaults to 30 s', () => {
    expect(RESTART_MS).toBe(30000);
    expect(new PearlMiner({}).restartMs).toBe(RESTART_MS);
    const m = new PearlMiner({ restartMs: 5 });
    expect(m.restartMs).toBe(5);
  });
});

// The memory clock lock, taken outright (--mine-mem-clock <MHz>). The
// shells decide which cards and what clock (shared/memClock) and hand the miner
// { index: mhz }; the lock is taken per card as its core starts and released in
// stop(), before 'stopped', because the demand gate starts llama-server --
// which needs the full memory clock -- the moment it hears the miner stopped.
// A card the plan also lists for a trial earns its lock first: the RTX 5090 by
// default, and --mine-mem-clock auto ("the memory clock trial", below).
describe('PearlMiner — the memory clock lock', () => {
  const GPUS = [
    { index: 0, name: 'NVIDIA GeForce RTX 5090' },
    { index: 1, name: 'NVIDIA GeForce RTX 5090' },
  ];
  // A requested lock on both cards, as planMemClocks gives it for --mine-mem-clock.
  const MEM = { ...settings, mineMemClockByIndex: { 0: 7001, 1: 7001 } };
  // The same two cards on the RTX 5090 default, which puts both on trial.
  const DEFAULT = { ...MEM, mineMemClockDefault: true, mineMemClockTrial: [0, 1] };

  // A card that reports the index it was asked for, like the real addon, and a
  // clocks double that records every call in order alongside the miner's own
  // 'stopped', so "released before stopped" is something a test can check.
  function rig(over = {}) {
    const calls = [];
    const clocks = {
      lockMemoryClock: jest.fn((index, mhz) => { calls.push(['lock', index, mhz]); return over.lock || { ok: true, error: null }; }),
      resetMemoryClock: jest.fn((index) => { calls.push(['reset', index]); return over.reset || { ok: true, error: null }; }),
    };
    const createCore = jest.fn((profile, opts) => {
      const c = makeCore();
      if (!over.silentDevice) c.device = { index: opts.deviceIndex, name: 'GPU' + opts.deviceIndex };
      return c;
    });
    const m = new PearlMiner({ connect: () => makeSocket(), createCore, reconnectMs: 0, clocks });
    const logs = [];
    m.on('log', (l) => logs.push(l));
    m.on('stopped', () => calls.push(['stopped']));
    m.on('error', () => {});
    return { m, clocks, calls, logs, createCore, lines: () => logs.map((l) => l.line) };
  }

  test('locks every card that starts, and releases them all before stopped', () => {
    const b = rig();
    expect(b.m.start({ ...MEM, gpus: GPUS })).toBe(true);
    expect(b.calls).toEqual([['lock', 0, 7001], ['lock', 1, 7001]]);
    expect(b.lines()).toEqual(expect.arrayContaining([
      'memory clock locked at 7001 MHz on GPU 0 while mining',
      'memory clock locked at 7001 MHz on GPU 1 while mining',
    ]));

    b.m.stop();
    expect(b.calls.slice(2)).toEqual([['reset', 0], ['reset', 1], ['stopped']]);
    expect(b.lines()).toEqual(expect.arrayContaining([
      'memory clock released on GPU 0',
      'memory clock released on GPU 1',
    ]));
  });

  // Demand mode stops and restarts the same engine on every request. Each run
  // locks on start and releases on stop, and a released card is not released
  // twice.
  test('locks again on the next start, and releases only once per run', () => {
    const b = rig();
    b.m.start({ ...MEM, gpus: [GPUS[0]] });
    b.m.stop();
    b.m.stop();
    b.m.start({ ...MEM, gpus: [GPUS[0]] });
    b.m.stop();
    expect(b.calls).toEqual([
      ['lock', 0, 7001], ['reset', 0], ['stopped'],
      ['lock', 0, 7001], ['reset', 0], ['stopped'],
    ]);
  });

  // START LLM while mining-only runs: the GUI keeps the miner and asks for the
  // locks back before the model starts. Every locked card is released, mining
  // carries on, and the stop that follows has nothing left to release.
  test('releaseMemClocks() releases every card mid-run, and stop() does not release again', () => {
    const b = rig();
    b.m.start({ ...MEM, gpus: GPUS });
    b.m.releaseMemClocks();
    expect(b.m.isRunning()).toBe(true);
    expect(b.calls).toEqual([['lock', 0, 7001], ['lock', 1, 7001], ['reset', 0], ['reset', 1]]);
    expect(b.lines()).toEqual(expect.arrayContaining([
      'memory clock released on GPU 0',
      'memory clock released on GPU 1',
    ]));
    b.m.releaseMemClocks();
    b.m.stop();
    expect(b.calls.slice(4)).toEqual([['stopped']]);
  });

  // A card that fails after START LLM and is opened again (#252's restart)
  // must not take the lock back while the model is served on it. The next
  // start plans the locks afresh, so it locks again.
  test('a card restarted after releaseMemClocks() stays at the driver\'s clock', () => {
    jest.useFakeTimers();
    try {
      const b = rig();
      b.m.start({ ...MEM, gpus: GPUS });
      b.m.releaseMemClocks();
      const first = b.createCore.mock.results[0].value;
      first.emit('error', new Error('GPU 0: misaligned address'));
      jest.advanceTimersByTime(RESTART_MS);
      expect(b.createCore).toHaveBeenCalledTimes(3);
      expect(b.calls).toEqual([['lock', 0, 7001], ['lock', 1, 7001], ['reset', 0], ['reset', 1]]);

      b.m.stop();
      b.m.start({ ...MEM, gpus: [GPUS[0]] });
      expect(b.calls.slice(4)).toEqual([['stopped'], ['lock', 0, 7001]]);
      b.m.stop();
    } finally {
      jest.useRealTimers();
    }
  });

  test('with no plan touches nothing: no nvidia-smi, on start or stop', () => {
    for (const plan of [{}, { mineMemClockByIndex: {} }]) {
      const b = rig();
      b.m.start({ ...settings, ...plan, gpus: GPUS });
      b.m.stop();
      expect(b.clocks.lockMemoryClock).not.toHaveBeenCalled();
      expect(b.clocks.resetMemoryClock).not.toHaveBeenCalled();
      expect(b.lines().join(' ')).not.toMatch(/memory clock/);
    }
  });

  // The miner takes the plan as it is given: a card the plan has no entry for
  // is left at the driver's clock, in silence.
  test('locks only the cards the plan names', () => {
    const b = rig();
    b.m.start({ ...settings, gpus: GPUS, mineMemClockByIndex: { 1: 7001 } });
    expect(b.calls).toEqual([['lock', 1, 7001]]);
    expect(b.lines().filter((l) => /memory clock/.test(l))).toHaveLength(1);
    b.m.stop();
    expect(b.calls.slice(1)).toEqual([['reset', 1], ['stopped']]);
  });

  // The RTX 5090 default used to lock here. It is a trial now: nothing is
  // locked when the cards open (the trial's own tests are below).
  test('the RTX 5090 default locks nothing at the start', () => {
    const b = rig();
    b.m.start({ ...DEFAULT, gpus: GPUS });
    expect(b.clocks.lockMemoryClock).not.toHaveBeenCalled();
    expect(b.lines().filter((l) => /memory clock locked/.test(l))).toEqual([]);
    expect(b.lines().filter((l) => /^memory clock trial on GPU [01]: /.test(l))).toHaveLength(2);
    b.m.stop();
  });

  // No root and no sudoers rule is the usual reason. Mining carries on at the
  // default clock, the log says once what would fix it, and a card that was
  // never locked is not "released".
  test('a failed lock is said once, mining carries on, and it is not reset', () => {
    const b = rig({ lock: { ok: false, error: 'sudo: a password is required' } });
    expect(b.m.start({ ...MEM, gpus: [GPUS[0]] })).toBe(true);
    expect(b.m.isRunning()).toBe(true);
    const warned = b.logs.filter((l) => /memory clock/.test(l.line));
    expect(warned).toEqual([{
      level: 'warn',
      line: 'could not lock the memory clock on GPU 0 (sudo: a password is required): it needs root, '
        + 'or a sudoers NOPASSWD rule for nvidia-smi. Mining continues at the default memory clock.',
    }]);

    b.m.stop();
    expect(b.clocks.resetMemoryClock).not.toHaveBeenCalled();
  });

  // Only the cards that locked are released. On a mixed rig one card can take
  // the lock while another refuses it.
  test('releases exactly the cards that locked', () => {
    const b = rig();
    b.clocks.lockMemoryClock.mockImplementationOnce(() => ({ ok: false, error: 'not supported' }));
    b.m.start({ ...MEM, gpus: GPUS });
    b.m.stop();
    expect(b.clocks.resetMemoryClock.mock.calls).toEqual([[1]]);
  });

  // A card whose core refused is not mining. Locking it would only slow down
  // whatever holds it -- usually the local LLM, which is why it had no room.
  test('does not lock a card whose core did not start', () => {
    const b = rig();
    b.createCore.mockImplementationOnce(() => { throw new Error('not enough free VRAM'); });
    b.m.start({ ...MEM, gpus: GPUS });
    expect(b.clocks.lockMemoryClock.mock.calls).toEqual([[1, 7001]]);
  });

  test('a rig where no core starts locks nothing', () => {
    const b = rig();
    b.createCore.mockImplementation(() => { throw new Error('no CUDA device found'); });
    expect(b.m.start({ ...MEM, gpus: GPUS })).toBe(false);
    expect(b.clocks.lockMemoryClock).not.toHaveBeenCalled();
  });

  // An older core does not report its card, and it ignores the one it was
  // asked for, so that index is only a guess -- and a wrong one would lock a
  // card the rig is using for something else.
  test('does not guess from the card it asked for when the core does not say', () => {
    const b = rig({ silentDevice: true });
    b.m.start({ ...MEM, gpus: [{ index: 2, name: 'NVIDIA GeForce RTX 5090' }] });
    expect(b.clocks.lockMemoryClock).not.toHaveBeenCalled();
  });

  test('locks nothing when it cannot tell which card is mining', () => {
    const b = rig({ silentDevice: true });
    expect(b.m.start(MEM)).toBe(true);
    expect(b.clocks.lockMemoryClock).not.toHaveBeenCalled();
    expect(b.logs).toContainEqual({
      level: 'warn',
      line: 'memory clock left at default: this pearl_core.node does not say which GPU it opened',
    });
  });

  // The same old core on a 5090 rig that asked for nothing: the default
  // is skipped, and that is information, not a warning on every start.
  test('a default the core cannot take is one info line', () => {
    const b = rig({ silentDevice: true });
    expect(b.m.start(DEFAULT)).toBe(true);
    expect(b.clocks.lockMemoryClock).not.toHaveBeenCalled();
    expect(b.logs.filter((l) => /memory clock/.test(l.line))).toEqual([{
      level: 'info',
      line: 'memory clock left at default: this pearl_core.node does not say which GPU it opened',
    }]);
  });

  // The one failure that matters: the card stays locked. The miner still
  // stops, and the log says how to undo it by hand.
  test('a failed release is an error that names the fix, and the miner still stops', () => {
    const b = rig({ reset: { ok: false, error: 'sudo: a password is required' } });
    b.m.start({ ...MEM, gpus: [GPUS[0]] });
    b.m.stop();
    expect(b.m.isRunning()).toBe(false);
    expect(b.calls).toContainEqual(['stopped']);
    expect(b.logs).toContainEqual({
      level: 'error',
      line: 'could not release the memory clock on GPU 0 (sudo: a password is required). '
        + 'An LLM on it will decode slowly until you run: sudo nvidia-smi -i 0 -rmc',
    });
  });

  test('uses gpuClocks when no clocks are injected', () => {
    const core = makeCore();
    core.device = { index: 0, name: 'NVIDIA GeForce RTX 5090' };
    const m = new PearlMiner({ connect: () => makeSocket(), createCore: () => core });
    m.start(MEM);
    m.stop();
    expect(gpuClocks.lockMemoryClock).toHaveBeenCalledWith(0, 7001);
    expect(gpuClocks.resetMemoryClock).toHaveBeenCalledWith(0);
  });
});

describe('PearlMiner — pausing cards while the LLM answers a prompt', () => {
  const GPUS = [
    { index: 0, name: 'NVIDIA GeForce RTX 5090' },
    { index: 1, name: 'NVIDIA GeForce RTX 4070' },
  ];

  beforeEach(() => jest.useFakeTimers());
  afterEach(() => {
    jest.clearAllTimers();
    jest.useRealTimers();
  });

  function rig(over = {}) {
    const made = [];
    const sock = makeSocket();
    const clocks = {
      lockMemoryClock: jest.fn(() => ({ ok: true, error: null })),
      resetMemoryClock: jest.fn(() => ({ ok: true, error: null })),
    };
    const createCore = jest.fn((profile, opts) => {
      if (over.refuse && over.refuse(opts)) throw over.refuseWith !== undefined ? over.refuseWith : new Error('out of memory');
      const c = makeCore();
      if (!over.silentDevice) c.device = { index: opts.deviceIndex, name: GPUS[opts.deviceIndex].name };
      made.push(c);
      return c;
    });
    const m = new PearlMiner({ connect: () => sock, createCore, reconnectMs: 0, clocks });
    const events = { log: [], hashrate: [], error: [], stopped: [] };
    for (const k of Object.keys(events)) m.on(k, (...a) => events[k].push(a.length > 1 ? a : a[0]));
    const lines = () => events.log.map((l) => l.line);
    return { m, sock, made, clocks, createCore, events, lines };
  }
  const coreFor = (b, index) => b.made.filter((c) => c.device.index === index).pop();

  function mining(b, gpus = GPUS, extra = {}) {
    b.m.start({ ...settings, gpus, ...extra });
    b.sock.emit('connect');
    b.sock.emit('data', jobLine());
  }

  test('pauses one card and leaves the pool connection and the other card alone', () => {
    const b = rig();
    mining(b);
    b.made[0].emit('hashrate', 400);
    b.made[1].emit('hashrate', 115);

    expect(b.m.pauseCards([0])).toEqual([GPUS[0]]);
    expect(b.made[0].stop).toHaveBeenCalled();
    expect(b.made[1].stop).not.toHaveBeenCalled();
    expect(b.sock.destroy).not.toHaveBeenCalled();
    expect(b.m.isRunning()).toBe(true);
    expect(b.m.devices()).toEqual([GPUS[1]]);
    // Its hashrate goes to 0 rather than standing at its last tick.
    expect(b.events.hashrate).toContainEqual([0, GPUS[0]]);
    expect(b.m.totalHashrate()).toBe(115);
    // A job that arrives while paused goes to the card still mining only.
    b.sock.emit('data', jobLine({ job_id: '00000000_2097153' }));
    expect(b.made[0].setJob).toHaveBeenCalledTimes(1);
    expect(b.made[1].setJob).toHaveBeenCalledTimes(2);
  });

  // The shells' prompt pause counts these on an AMD rig (shared/promptPause): a
  // paused card and one waiting to restart are still the run's. A planned card
  // that would not start, one HIP doesn't have, is not.
  test('counts the cards mining, paused or waiting to restart, not one that never started', () => {
    const b = rig({ refuse: (opts) => opts.deviceIndex === 2 });
    expect(b.m.cardCount()).toBe(0);
    mining(b, [...GPUS, { index: 2, name: 'GPU 2' }]);
    expect(b.m.cardCount()).toBe(2);
    b.m.pauseCards([0]);
    expect(b.m.cardCount()).toBe(2);
    b.made[1].emit('error', new Error('an illegal memory access was encountered'));
    expect(b.lines()).toContain('restarting GPU 1 (NVIDIA GeForce RTX 4070) in 30 s');
    expect(b.m.cardCount()).toBe(2);
    b.m.stop();
    expect(b.m.cardCount()).toBe(0);
  });

  test('resumes the card on fresh salts and the current job, without logging it as a new start', () => {
    const b = rig();
    mining(b);
    b.m.pauseCards([0]);
    b.sock.emit('data', jobLine({ job_id: '00000000_2097153' }));
    const before = b.lines().length;

    expect(b.m.resumeCards([0])).toEqual([GPUS[0]]);
    const again = coreFor(b, 0);
    expect(again).not.toBe(b.made[0]);
    // A fresh run of salts, so the pool is never sent a share twice.
    expect(b.createCore.mock.calls[2][1]).toEqual({ deviceIndex: 0, saltStride: 2, saltBase: 0 + 2 * SALT_RUN, cardK: true, checkHits: true });
    expect(again.setJob.mock.calls[0][0].jobId).toBe('00000000_2097153');
    expect(b.m.devices()).toEqual([GPUS[1], GPUS[0]]);
    expect(b.lines().length).toBe(before);
    // Nothing is left paused, so a second resume does nothing.
    expect(b.m.resumeCards([0])).toEqual([]);
  });

  test('null means every card', () => {
    const b = rig();
    mining(b);
    expect(b.m.pauseCards(null)).toEqual(GPUS);
    expect(b.m.devices()).toEqual([]);
    // Still running, with the pool connection open, waiting for the prompt to end.
    expect(b.m.isRunning()).toBe(true);
    expect(b.m.resumeCards(null)).toEqual(GPUS);
    expect(b.m.devices()).toEqual(GPUS);
  });

  test('a card that is not mining, or a resume of a card not paused, is left alone', () => {
    const b = rig();
    mining(b, [GPUS[1]]);
    expect(b.m.pauseCards([0])).toEqual([]);
    expect(b.m.resumeCards([1])).toEqual([]);
    expect(b.made[0].stop).not.toHaveBeenCalled();
  });

  test('a core that will not name its card is paused only by an all-cards pause', () => {
    const b = rig({ silentDevice: true });
    b.m.start({ ...settings });
    b.sock.emit('connect');
    b.sock.emit('data', jobLine());
    expect(b.m.pauseCards([0])).toEqual([]);
    expect(b.m.pauseCards(null)).toEqual([null]);
    expect(b.events.hashrate).toEqual([]);
    expect(b.m.resumeCards([0])).toEqual([]);
    expect(b.m.resumeCards(null)).toEqual([null]);
    expect(b.createCore).toHaveBeenCalledTimes(2);
  });

  test('does nothing before a start or after a stop', () => {
    const b = rig();
    expect(b.m.pauseCards(null)).toEqual([]);
    expect(b.m.resumeCards(null)).toEqual([]);
    mining(b);
    b.m.pauseCards([0]);
    b.m.stop();
    expect(b.m.resumeCards(null)).toEqual([]);
    expect(b.m.paused.size).toBe(0);
  });

  test('releases a memory-clock lock on the card it pauses, and takes it again on resume', () => {
    const b = rig();
    mining(b, [GPUS[0]], { mineMemClockByIndex: { 0: 7001 } });
    b.m.pauseCards([0]);
    expect(b.clocks.resetMemoryClock).toHaveBeenCalledWith(0);
    b.m.resumeCards([0]);
    expect(b.clocks.lockMemoryClock.mock.calls).toEqual([[0, 7001], [0, 7001]]);
  });

  test('a card that will not open again is a card down: restarted later, and the rig keeps running', () => {
    let refuse = false;
    const b = rig({ refuse: () => refuse });
    mining(b);
    b.m.pauseCards([0]);
    refuse = true;
    expect(b.m.resumeCards([0])).toEqual([]);
    expect(b.lines()).toContain('could not resume GPU 0 (NVIDIA GeForce RTX 5090): out of memory');
    expect(b.lines()).toContain('restarting GPU 0 (NVIDIA GeForce RTX 5090) in 30 s');
    expect(b.m.isRunning()).toBe(true);
    refuse = false;
    jest.advanceTimersByTime(RESTART_MS);
    expect(b.m.devices()).toEqual([GPUS[1], GPUS[0]]);
  });

  describe('a resume refused for lack of VRAM', () => {
    const NO_VRAM = new Error('GPU 0: not enough free VRAM for the rank-128 profile: need ~2081 MiB, 900 MiB free.');

    test('stays paused and tries again, without counting a fault, until the memory is back', () => {
      let refuse = false;
      const b = rig({ refuse: (o) => refuse && o.deviceIndex === 0, refuseWith: NO_VRAM });
      mining(b, [GPUS[0]]);
      b.m.pauseCards([0]);
      refuse = true;
      expect(b.m.resumeCards([0])).toEqual([]);
      const warned = b.lines().filter((l) => l.startsWith('could not resume GPU 0'));
      expect(warned).toHaveLength(1);
      expect(warned[0]).toMatch(/Something else took the memory it gave up while the local LLM answered\. Trying again every 30 s\.$/);
      // Not a card down: no restart counting, and a one-card rig keeps its pool connection.
      expect(b.lines().some((l) => l.startsWith('restarting'))).toBe(false);
      expect(b.m.isRunning()).toBe(true);
      expect(b.events.stopped).toEqual([]);

      // Still short: tries again, and says nothing new.
      jest.advanceTimersByTime(RESTART_MS);
      expect(b.lines().filter((l) => l.startsWith('could not resume'))).toHaveLength(1);
      // The memory is back: it mines again, and says so.
      refuse = false;
      jest.advanceTimersByTime(RESTART_MS);
      expect(b.m.devices()).toEqual([GPUS[0]]);
      expect(b.lines()).toContain('GPU 0 (NVIDIA GeForce RTX 5090) has its memory back and is mining again');
      expect(b.m.cores[0].slot.fails).toBe(0);
    });

    test('a new prompt while it waits cancels the retry, so it does not start under the model', () => {
      let refuse = false;
      const b = rig({ refuse: () => refuse, refuseWith: NO_VRAM });
      mining(b, [GPUS[0]]);
      b.m.pauseCards([0]);
      refuse = true;
      b.m.resumeCards([0]);
      refuse = false;
      expect(b.m.pauseCards([0])).toEqual([]);
      jest.advanceTimersByTime(RESTART_MS * 3);
      expect(b.m.devices()).toEqual([]);
      expect(b.m.resumeCards([0])).toEqual([GPUS[0]]);
    });

    test('stop() drops the retry', () => {
      const b = rig({ refuse: (o) => b.m.paused.size > 0 && o.deviceIndex === 0, refuseWith: NO_VRAM });
      mining(b, [GPUS[0]]);
      b.m.pauseCards([0]);
      b.m.resumeCards([0]);
      b.m.stop();
      expect(b.m.resumeRetries.size).toBe(0);
      jest.advanceTimersByTime(RESTART_MS);
      expect(b.createCore).toHaveBeenCalledTimes(2);
    });

    test('a core that will not name its card is retried as every card', () => {
      let refuse = false;
      const b = rig({ silentDevice: true, refuse: () => refuse, refuseWith: NO_VRAM });
      b.m.start({ ...settings });
      b.sock.emit('connect');
      b.sock.emit('data', jobLine());
      b.m.pauseCards(null);
      refuse = true;
      b.m.resumeCards(null);
      expect(b.lines()).toContain('could not resume the GPU: ' + NO_VRAM.message
        + ' Something else took the memory it gave up while the local LLM answered. Trying again every 30 s.');
      refuse = false;
      jest.advanceTimersByTime(RESTART_MS);
      expect(b.m.cores).toHaveLength(1);
    });

    test('a pause of every card while an unnamed card waits cancels its retry', () => {
      let refuse = false;
      const b = rig({ silentDevice: true, refuse: () => refuse, refuseWith: NO_VRAM });
      b.m.start({ ...settings });
      b.sock.emit('connect');
      b.sock.emit('data', jobLine());
      b.m.pauseCards(null);
      refuse = true;
      b.m.resumeCards(null);
      refuse = false;
      b.m.pauseCards(null);
      jest.advanceTimersByTime(RESTART_MS);
      expect(b.m.cores).toHaveLength(0);
      expect(b.m.resumeCards(null)).toEqual([null]);
    });

    test('pausing another card leaves this card\'s retry alone', () => {
      let refuse = false;
      const b = rig({ refuse: (o) => refuse && o.deviceIndex === 0, refuseWith: NO_VRAM });
      mining(b);
      b.m.pauseCards([0]);
      refuse = true;
      b.m.resumeCards([0]);
      refuse = false;
      b.m.pauseCards([1]);
      jest.advanceTimersByTime(RESTART_MS);
      expect(b.m.devices()).toEqual([GPUS[0]]);
    });
  });

  test('a refusal that is not an Error is still named in the log', () => {
    let refuse = false;
    const b = rig({ refuse: () => refuse, refuseWith: 'no device' });
    mining(b);
    b.m.pauseCards([1]);
    refuse = true;
    b.m.resumeCards([1]);
    expect(b.lines()).toContain('could not resume GPU 1 (NVIDIA GeForce RTX 4070): no device');
  });

  test('a pause does not make a later fault count as a quick repeat', () => {
    const b = rig();
    mining(b, [GPUS[1]]);
    const startedAt = b.m.cores[0].slot.startedAt;
    jest.advanceTimersByTime(STABLE_MS);
    b.m.pauseCards([1]);
    b.m.resumeCards([1]);
    expect(b.m.cores[0].slot.startedAt).toBe(startedAt);
  });

  test('the last card failing while another is paused does not end the run', () => {
    const b = rig();
    mining(b);
    b.m.pauseCards([0]);
    // Fail card 1 enough times that it is left off.
    for (let i = 0; i <= RESTART_LIMIT; i++) {
      coreFor(b, 1).emit('error', new Error('GPU 1: boom'));
      jest.advanceTimersByTime(RESTART_MS);
    }
    expect(b.lines().some((l) => l.startsWith('leaving GPU 1'))).toBe(true);
    expect(b.m.isRunning()).toBe(true);
    expect(b.events.stopped).toEqual([]);
    expect(b.m.resumeCards([0])).toEqual([GPUS[0]]);
  });
});

// A core may mine a larger k on its card than the profile's (cardK; an RDNA 4
// card mines k 4096, everything else k 2048). Each card then has its own
// profile, and with it its own share bound, job_key and proofs: a rig whose
// cards mine different k must never send a share under the other k.
describe('PearlMiner — each card mines its own k', () => {
  // m and n small enough to build real trees here, at both k.
  const SMALL = { ...PROFILE, m: 64, n: 64 };
  const K4 = profileForK(SMALL, 4096);
  const T = BigInt('0x' + TARGET);
  const HEADER2 = HEADER.slice(0, -8) + '00000001';
  const AUTH_V2 = '{"id":1,"result":true,"error":null,"type":"v2"}\n';
  const AUTH_PLAIN = '{"id":1,"result":true,"error":null}\n';
  const FALLBACK = 'the pool turned down 3 shares at k 4096 (rejected, unanswered, or dropped with the connection) '
    + 'and accepted none; mining k 2048 on every card for the rest of this run';

  // A hit with genuine proofs, as the device builds them: over the constant
  // fill, under the job_key of `header` at this profile's k. Built once a
  // (header, k): the miner does not change a hit.
  const built = new Map();
  function hitFor(profile, { header = HEADER, jobId = '00000000_2097152', value = 1n << 192n } = {}) {
    const id = header + ':' + profile.k;
    if (!built.has(id)) {
      const jobKey = hash(Buffer.concat([Buffer.from(header, 'hex'), buildConfig52(profile)]));
      const side = (rows, dim) => {
        const mat = Buffer.alloc(dim * profile.k, 48);
        const p = M.multiLeafProof(jobKey, mat, M.buildLayers(jobKey, mat), M.leafIndicesFromRows(rows, profile.k));
        return {
          leafIndices: p.leafIndices, leafData: Buffer.concat(p.leafData),
          siblings: Buffer.concat(p.siblings), root: p.root, totalLeaves: p.totalLeaves,
        };
      };
      const { rows, cols } = regionToTile(0, profile);
      built.set(id, { nonce: 0, proofA: side(rows, profile.m), proofBt: side(cols, profile.n) });
    }
    const jackpotHash = Buffer.alloc(32);
    let v = value;
    for (let i = 0; i < 32; i++) { jackpotHash[i] = Number(v & 0xffn); v >>= 8n; }
    return { ...built.get(id), jobId, jackpotHash };
  }

  // The k a submitted share declares, read off its proof as a pool reads it.
  function kOf(submit) {
    const raw = Buffer.from(submit.params.plain_proof, 'base64');
    return decodePlainProof(submit.params.plain_proof, { gzip: raw[0] === 0x1f && raw[1] === 0x8b }).k;
  }

  // A rig of cards. Each card says what its core reports: `k` is the k it picks
  // when the miner lets it (cardK), and with cardK off it keeps the profile's;
  // no `k` is a core built before cores could choose. `fail` makes the card's
  // next core throw it. With `gpus: false` the rig has no card list, and its one
  // card's core (cards[0]) names no device. `mem` is the memory its core
  // reports, { memTotalMb, memUsedMb }; left out, the core reports none.
  // `miner` adds to what the miner is built with, and `more` to the settings
  // it starts with.
  function rig(cards, { gpus = true, miner = {}, more = {} } = {}) {
    const socks = [];
    const made = [];
    const connect = jest.fn(() => { const s = makeSocket(); socks.push(s); return s; });
    const createCore = jest.fn((profile, opts) => {
      const card = gpus ? cards.find((c) => c.index === opts.deviceIndex) : cards[0];
      if (card.fail) {
        const e = card.fail;
        card.fail = null;
        throw e;
      }
      const c = makeCore();
      if (gpus) c.device = { index: card.index, name: 'Radeon ' + card.index, ...card.mem };
      if (card.k !== undefined) c.k = opts.cardK ? card.k : profile.k;
      if (card.kNote) c.kNote = card.kNote;
      made.push({ core: c, opts, card });
      return c;
    });
    const m = new PearlMiner({ connect, createCore, reconnectMs: 0, restartMs: 10, ...miner });
    const lines = [];
    m.on('log', (l) => lines.push(l.line));
    m.on('error', () => {});
    const start = () => m.start({
      ...settings, profile: SMALL, gpus: gpus ? cards.map((c) => ({ index: c.index, name: 'Radeon ' + c.index })) : [],
      ...more,
    });
    start();
    const sock = () => socks[socks.length - 1];
    return {
      m, socks, made, lines, createCore, sock, start,
      // The newest core of a card (by index, or the only card).
      core: (index = cards[0].index) => made.filter((x) => x.card.index === index).pop().core,
      login(reply = AUTH_PLAIN, job) {
        sock().emit('connect');
        sock().emit('data', reply);
        sock().emit('data', jobLine(job));
      },
      job: (over) => sock().emit('data', jobLine(over)),
      submits: () => sock().written.map((w) => JSON.parse(w)).filter((x) => x.method === 'mining.submit'),
      answer(id, ok, msg = 'invalid share') {
        sock().emit('data', JSON.stringify(ok ? { id, result: true, error: null } : { id, result: null, error: [23, msg] }) + '\n');
      },
      // The socket closes, as one the pool hung up does, and the reconnect runs.
      reconnect() {
        sock().emit('close');
        jest.runOnlyPendingTimers();
      },
    };
  }

  test('opens every core with cardK, and a core that reports no k mines the profile as it did', () => {
    const r = rig([{ index: 0 }]);
    r.login();
    expect(r.createCore.mock.calls[0][0]).toBe(SMALL);
    expect(r.createCore.mock.calls[0][1].cardK).toBe(true);
    expect(r.m.cores[0].profile).toBe(SMALL);
    expect(r.core().setJob.mock.calls[0][0].target).toBe(shareBound(T, SMALL));
    expect(r.lines).toContain('mining on GPU 0 · Radeon 0');
    r.core().emit('hit', hitFor(SMALL));
    expect(kOf(r.submits()[0])).toBe(2048);
    r.m.stop();
  });

  test('a mixed rig gives each card the bound of its own k, and names k 4096 only where it is mined', () => {
    const r = rig([{ index: 0, k: 4096 }, { index: 1, k: 2048, kNote: 'k 2048 (PEARL_AMD_K=2048)' }, { index: 2, k: 2048 }]);
    r.login();
    expect(r.m.cores.map((c) => c.profile.k)).toEqual([4096, 2048, 2048]);
    expect(r.m.cores[0].profile).toEqual(K4);
    expect(r.m.cores[2].profile).toBe(SMALL);
    // A k 4096 attempt is twice the work, so its bound is twice as easy.
    expect(r.core(0).setJob.mock.calls[0][0].target).toBe(2n * shareBound(T, SMALL));
    expect(r.core(1).setJob.mock.calls[0][0].target).toBe(shareBound(T, SMALL));
    expect(r.lines).toEqual(expect.arrayContaining([
      'mining on GPU 0 · Radeon 0 · k 4096',
      'mining on GPU 1 · Radeon 1 · k 2048 (PEARL_AMD_K=2048)',
      'mining on GPU 2 · Radeon 2',
    ]));
    // Each card's rate is MACs a second at its own k, so they add up.
    r.core(0).emit('hashrate', 60);
    r.core(1).emit('hashrate', 30);
    r.core(2).emit('hashrate', 10);
    expect(r.m.totalHashrate()).toBe(100);
    r.m.stop();
  });

  test('each card\'s hits go out at its own k, and a hit certified for the other k is dropped, never sent', () => {
    const r = rig([{ index: 0, k: 4096 }, { index: 1, k: 2048 }]);
    r.login();
    r.core(0).emit('hit', hitFor(K4));
    r.core(1).emit('hit', hitFor(SMALL));
    const [s4, s2] = r.submits();
    expect([kOf(s4), kOf(s2)]).toEqual([4096, 2048]);
    expect(r.m.pending.get(s4.id)).toEqual({ jobId: '00000000_2097152', index: 0, gzip: false, k: 4096 });
    expect(r.m.pending.get(s2.id)).toEqual({ jobId: '00000000_2097152', index: 1, gzip: false, k: 2048 });
    // Each proof opens to its roots under its own k's job_key, and only that one.
    const key = (p) => hash(Buffer.concat([Buffer.from(HEADER, 'hex'), buildConfig52(p)]));
    const pp4 = decodePlainProof(s4.params.plain_proof);
    expect(M.verifyProof(key(K4), pp4.a.proof)).toBe(true);
    expect(M.verifyProof(key(SMALL), pp4.a.proof)).toBe(false);
    // Swapped round: each card handed the other k's hit.
    r.core(1).emit('hit', hitFor(K4));
    r.core(0).emit('hit', hitFor(SMALL));
    expect(r.submits()).toHaveLength(2);
    expect(r.lines.filter((l) => l === 'dropping a hit whose proof does not verify locally')).toHaveLength(2);
    r.m.stop();
  });

  test('a hash between the two bounds is sent from the k 4096 card and dropped from the k 2048 card', () => {
    const r = rig([{ index: 0, k: 4096 }, { index: 1, k: 2048 }]);
    r.login();
    const v = shareBound(T, K4);
    expect(v).toBeGreaterThan(shareBound(T, SMALL));
    r.core(0).emit('hit', hitFor(K4, { value: v }));
    r.core(1).emit('hit', hitFor(SMALL, { value: v }));
    expect(r.submits().map(kOf)).toEqual([4096]);
    expect(r.lines).toContain('dropping a hit that no longer meets target (vardiff moved)');
    r.m.stop();
  });

  test('a new job empties the job_keys, so each card certifies with the new header\'s', () => {
    const r = rig([{ index: 0, k: 4096 }, { index: 1, k: 2048 }]);
    r.login();
    r.core(0).emit('hit', hitFor(K4));
    expect(r.m.jobKeys.size).toBe(1);
    r.job({ job_id: '00000001_2097152', header: HEADER2 });
    expect(r.m.jobKeys.size).toBe(0);
    r.core(0).emit('hit', hitFor(K4, { header: HEADER2, jobId: '00000001_2097152' }));
    r.core(1).emit('hit', hitFor(SMALL, { header: HEADER2, jobId: '00000001_2097152' }));
    expect(r.submits().map(kOf)).toEqual([4096, 4096, 2048]);
    expect(r.m.jobKeys.size).toBe(2);
    r.m.stop();
  });

  test('a core that chose a k a share can\'t carry is stopped, and the card skipped', () => {
    const r = rig([{ index: 0, k: 3072 }, { index: 1, k: 2048 }]);
    expect(r.made[0].core.stop).toHaveBeenCalledTimes(1);
    expect(r.lines).toContain('skipping GPU 0 (Radeon 0): the core chose k 3072, which a share can\'t carry');
    expect(r.m.cores.map((c) => c.device.index)).toEqual([1]);
    r.m.stop();
  });

  test('a target too easy for k 4096 alone: those cards wait for the next job, said once, and the rest mine', () => {
    // Scaled by 2^19 it fits 256 bits; by 2^20 it doesn't.
    const easy = (((1n << 256n) - 1n) >> 19n).toString(16).padStart(64, '0');
    const r = rig([{ index: 0, k: 4096 }, { index: 1, k: 2048 }, { index: 2, k: 4096 }]);
    r.login(AUTH_PLAIN, { target: easy });
    expect(r.core(0).setJob).not.toHaveBeenCalled();
    expect(r.core(2).setJob).not.toHaveBeenCalled();
    expect(r.core(1).setJob).toHaveBeenCalledTimes(1);
    expect(r.lines.filter((l) => /too easy/.test(l))).toEqual([
      'pool target is too easy to scale for k 4096; not mining this job at that k',
    ]);
    // The next job is said again.
    r.job({ job_id: '00000001_2097152', target: easy });
    expect(r.lines.filter((l) => /too easy/.test(l))).toHaveLength(2);
    r.m.stop();
  });

  describe('opening a card again', () => {
    beforeEach(() => jest.useFakeTimers());
    afterEach(() => jest.useRealTimers());

    test('a restart and a resume read the card\'s k again, and a resume at another k says so', () => {
      const cards = [{ index: 0, k: 4096 }];
      const r = rig(cards);
      r.login();
      expect(r.core().setJob.mock.calls[0][0].target).toBe(2n * shareBound(T, SMALL));
      // Back after a fault with less memory free: the core stays at k 2048 and says why.
      cards[0].k = 2048;
      cards[0].kNote = 'k 2048: k 4096 needs ~1969 MiB and 900 MiB is free';
      r.core().emit('error', new Error('GPU 0: boom'));
      jest.advanceTimersByTime(10);
      expect(r.m.cores[0].profile).toBe(SMALL);
      expect(r.core().setJob.mock.calls[0][0].target).toBe(shareBound(T, SMALL));
      expect(r.lines).toContain('mining on GPU 0 · Radeon 0 · k 2048: k 4096 needs ~1969 MiB and 900 MiB is free');
      // A resume at another k than the card had says so, though a resume is quiet.
      cards[0].k = 4096;
      cards[0].kNote = '';
      r.m.pauseCards([0]);
      const said = r.lines.length;
      r.m.resumeCards([0]);
      expect(r.lines.slice(said)).toEqual(['mining on GPU 0 · Radeon 0 · k 4096']);
      expect(r.core().setJob.mock.calls[0][0].target).toBe(2n * shareBound(T, SMALL));
      // At the same k it stays quiet.
      r.m.pauseCards([0]);
      r.m.resumeCards([0]);
      expect(r.lines.slice(said + 1).filter((l) => l.startsWith('mining on'))).toEqual([]);
      r.m.stop();
    });
  });

  describe('when a pool turns k 4096 down', () => {
    beforeEach(() => jest.useFakeTimers());
    afterEach(() => jest.useRealTimers());

    test('three rejections and no accept: the k 4096 cards open again at k 2048 on fresh salts, on the job', () => {
      expect(K_REJECT_LIMIT).toBe(3);
      const r = rig([{ index: 0, k: 4096 }, { index: 1, k: 2048 }]);
      r.login();
      for (let i = 0; i < 3; i++) r.core(0).emit('hit', hitFor(K4));
      r.core(1).emit('hit', hitFor(SMALL));
      const ids = r.submits().map((s) => s.id);
      // A k 2048 share's rejection says nothing about k 4096.
      r.answer(ids[3], false);
      r.answer(ids[0], false);
      r.answer(ids[1], false);
      expect(r.lines).toContain('share rejected (k 4096): [23] invalid share');
      expect(r.lines).toContain('share rejected: [23] invalid share');
      expect(r.made).toHaveLength(2);
      const old = r.core(0);
      r.answer(ids[2], false);
      expect(r.lines).toContain(FALLBACK);
      expect(old.stop).toHaveBeenCalledTimes(1);
      expect(r.made).toHaveLength(3);
      expect(r.made[2].opts).toEqual({ deviceIndex: 0, saltStride: 2, saltBase: 0 + 2 * SALT_RUN, cardK: false, checkHits: true });
      expect(r.core(0).setJob.mock.calls[0][0].target).toBe(shareBound(T, SMALL));
      expect(r.m.cores.map((c) => c.profile.k)).toEqual([2048, 2048]);
      expect(r.lines.filter((l) => l === 'mining on GPU 0 · Radeon 0')).toHaveLength(1);
      // The k 2048 card and the connection are left alone.
      expect(r.made[1].core.stop).not.toHaveBeenCalled();
      expect(r.sock().destroy).not.toHaveBeenCalled();
      // A hit still on its way from the closed core goes out at the k it was found at.
      old.emit('hit', hitFor(K4));
      const late = r.submits().pop();
      expect(kOf(late)).toBe(4096);
      r.answer(late.id, false);
      expect(r.made).toHaveLength(3);
      // Opened again later, a card stays at k 2048 for the rest of the run.
      r.core(0).emit('error', new Error('GPU 0: boom'));
      jest.advanceTimersByTime(10);
      expect(r.made[3].opts.cardK).toBe(false);
      expect(r.m.cores.map((c) => c.profile.k)).toEqual([2048, 2048]);
      r.m.stop();
    });

    // The card's core closes and opens again, so its memory clock lock goes
    // with it, as at every other place a card's core closes. Left on, a card
    // part way through its memory clock trial started the trial over with the
    // miner still counting it as locked.
    test('a card opened again at k 2048 gives its memory clock lock back first', () => {
      const { TURN_MS } = require('../src/shared/memClockTrial');
      const calls = [];
      const t = { now: 0 };
      const ok = { ok: true, error: null };
      const clocks = {
        lockMemoryClock: (index, mhz) => { calls.push(['lock', index, mhz]); return ok; },
        resetMemoryClock: (index) => { calls.push(['reset', index]); return ok; },
      };
      const r = rig([{ index: 0, k: 4096 }], {
        miner: { clocks, now: () => t.now },
        more: { mineMemClockByIndex: { 0: 5001 }, mineMemClockTrial: [0] },
      });
      r.login();
      // Into the trial's second turn, the first locked one.
      t.now += TURN_MS;
      r.core().emit('hashrate', 100);
      expect(calls).toEqual([['reset', 0], ['lock', 0, 5001]]);
      expect([...r.m.memClockLocked]).toEqual([0]);

      for (let i = 0; i < 3; i++) r.core().emit('hit', hitFor(K4));
      for (const s of r.submits()) r.answer(s.id, false);
      expect(r.lines).toContain(FALLBACK);
      expect(r.made).toHaveLength(2);
      // The lock came off with the old core. The new core's trial then starts
      // as any does, with the card handed to the driver, and holds no lock.
      expect(calls.slice(2)).toEqual([['reset', 0], ['reset', 0]]);
      expect(r.lines).toContain('memory clock released on GPU 0');
      expect(r.m.memClockLocked.size).toBe(0);
      expect(r.m.memClockTrials.has(0)).toBe(true);
      r.m.stop();
      expect(calls).toHaveLength(4);
    });

    test('one accepted share at k 4096 shows the pool takes it: no rejection after it falls back', () => {
      const r = rig([{ index: 0, k: 4096 }]);
      r.login();
      for (let i = 0; i < 4; i++) r.core().emit('hit', hitFor(K4));
      const ids = r.submits().map((s) => s.id);
      r.answer(ids[0], true);
      for (const id of ids.slice(1)) r.answer(id, false);
      expect(r.m.kAccepted).toBe(true);
      expect(r.made).toHaveLength(1);
      expect(r.lines).not.toContain(FALLBACK);
      r.m.stop();
    });

    test('stale, duplicate and job-not-found rejections count toward neither fallback', () => {
      const r = rig([{ index: 0, k: 4096 }, { index: 1, k: 2048 }]);
      r.login(AUTH_V2);
      for (let i = 0; i < 3; i++) {
        r.core(0).emit('hit', hitFor(K4));
        r.core(1).emit('hit', hitFor(SMALL));
      }
      const reasons = ['Stale share', 'Duplicate share', 'Job not found', 'stale job', 'duplicate', 'job expired'];
      r.submits().forEach((s, i) => r.answer(s.id, false, reasons[i]));
      expect([r.m.kRejects, r.m.gzipRejects]).toEqual([0, 0]);
      expect(r.made).toHaveLength(2);
      expect(r.sock().destroy).not.toHaveBeenCalled();
      r.m.stop();
    });

    test('k 4096 rejections on a compressed connection count toward the k fallback only, so it goes first', () => {
      const r = rig([{ index: 0, k: 4096 }]);
      r.login(AUTH_V2);
      for (let i = 0; i < 3; i++) r.core().emit('hit', hitFor(K4));
      for (const s of r.submits()) r.answer(s.id, false);
      expect(r.lines).toContain(FALLBACK);
      expect([r.m.gzipRejects, r.m.askGzip]).toEqual([0, true]);
      expect(r.sock().destroy).not.toHaveBeenCalled();
      // At k 2048 with compression, rejections are the compression's, as before.
      for (let i = 0; i < 3; i++) r.core().emit('hit', hitFor(SMALL));
      for (const s of r.submits().slice(3)) r.answer(s.id, false);
      expect(r.m.askGzip).toBe(false);
      expect(r.sock().destroy).toHaveBeenCalledTimes(1);
      r.m.stop();
    });

    test('a new run lets cores choose their k again, and counts from nothing', () => {
      const r = rig([{ index: 0, k: 4096 }]);
      r.login();
      for (let i = 0; i < 3; i++) r.core().emit('hit', hitFor(K4));
      for (const s of r.submits()) r.answer(s.id, false);
      expect(r.m.cardK).toBe(false);
      r.m.stop();
      r.start();
      expect(r.createCore.mock.calls.pop()[1].cardK).toBe(true);
      expect([r.m.cardK, r.m.kRejects, r.m.kAccepted]).toEqual([true, 0, false]);
      r.login();
      r.core().emit('hit', hitFor(K4));
      r.answer(r.submits()[0].id, false);
      expect(r.m.kRejects).toBe(1);
      r.m.stop();
    });

    test('a pool that hangs up with a k 4096 share waiting: a strike a close, and three fall back', () => {
      const r = rig([{ index: 0, k: 4096 }, { index: 1, k: 2048 }]);
      for (let i = 0; i < 3; i++) {
        r.login();
        // Two shares waiting are one hang-up, not two.
        r.core(0).emit('hit', hitFor(K4));
        r.core(0).emit('hit', hitFor(K4));
        expect(r.m.kRejects).toBe(i);
        r.reconnect();
      }
      expect(r.lines).toContain(FALLBACK);
      // Reopened between connections: no job to give it until the next one.
      expect(r.core(0).setJob).not.toHaveBeenCalled();
      r.login();
      expect(r.core(0).setJob.mock.calls[0][0].target).toBe(shareBound(T, SMALL));
      r.m.stop();
    });

    test('a close the miner made, or with only k 2048 shares waiting, is no strike', () => {
      const r = rig([{ index: 0, k: 4096 }, { index: 1, k: 2048 }]);
      r.login(AUTH_V2);
      for (let i = 0; i < 3; i++) r.core(1).emit('hit', hitFor(SMALL));
      r.core(0).emit('hit', hitFor(K4));
      // Three compressed k 2048 rejections: the miner closes the socket itself.
      for (const s of r.submits().filter((x) => kOf(x) === 2048)) r.answer(s.id, false);
      expect(r.sock().destroy).toHaveBeenCalledTimes(1);
      r.reconnect();
      expect(r.m.kRejects).toBe(0);
      r.login();
      r.core(1).emit('hit', hitFor(SMALL));
      r.reconnect();
      expect(r.m.kRejects).toBe(0);
      r.m.stop();
    });

    test('a k 4096 share unanswered for K_ANSWER_MS is a strike, counted once', () => {
      expect(K_ANSWER_MS).toBe(30000);
      const r = rig([{ index: 0, k: 4096 }]);
      r.login();
      r.core().emit('hit', hitFor(K4));
      jest.advanceTimersByTime(K_ANSWER_MS - 1);
      expect(r.m.kRejects).toBe(0);
      jest.advanceTimersByTime(1);
      expect(r.m.kRejects).toBe(1);
      // Its late rejection is not a second strike.
      r.answer(r.submits()[0].id, false);
      expect(r.m.kRejects).toBe(1);
      // Nor is a hang-up while only a late share waits.
      r.core().emit('hit', hitFor(K4));
      jest.advanceTimersByTime(K_ANSWER_MS);
      expect(r.m.kRejects).toBe(2);
      r.reconnect();
      expect(r.m.kRejects).toBe(2);
      // An answer clears the wait: no strike comes after it.
      r.login();
      r.core().emit('hit', hitFor(K4));
      r.answer(r.submits()[0].id, true);
      jest.advanceTimersByTime(K_ANSWER_MS);
      expect(r.m.kRejects).toBe(2);
      // And stop() clears what is waiting.
      r.core().emit('hit', hitFor(K4));
      r.m.stop();
      jest.advanceTimersByTime(K_ANSWER_MS);
      expect(r.m.kRejects).toBe(2);
    });

    test('a card that will not open again at k 2048 is a card down, and comes back later', () => {
      const cards = [{ index: 0, k: 4096 }, { index: 1, k: 4096 }];
      const r = rig(cards);
      r.login();
      r.core(0).emit('hashrate', 60);
      r.core(1).emit('hashrate', 30);
      expect(r.m.hashrate).toBe(90);
      const rates = [];
      r.m.on('hashrate', (th, d) => rates.push([th, d.index]));
      for (let i = 0; i < 3; i++) r.core(0).emit('hit', hitFor(K4));
      cards[0].fail = new Error('not enough free VRAM for the rank-128 profile');
      for (const s of r.submits()) r.answer(s.id, false);
      expect(r.lines).toContain('could not open GPU 0 (Radeon 0) again at k 2048: not enough free VRAM for the rank-128 profile');
      // The card that is down leaves the total and the board; the one that opened again reports anew.
      expect(rates).toEqual([[0, 0]]);
      expect(r.m.hashrate).toBe(0);
      expect(r.lines).toContain('restarting GPU 0 (Radeon 0) in 0 s');
      expect(r.m.cores.map((c) => [c.device.index, c.profile.k])).toEqual([[1, 2048]]);
      jest.advanceTimersByTime(10);
      expect(r.m.cores.map((c) => [c.device.index, c.profile.k])).toEqual([[1, 2048], [0, 2048]]);
      r.m.stop();
    });

    // Its old core is closed and no new one opened, so it holds none of the
    // card's memory (see "a card's memory, as its core reports it").
    test('a card that will not open again reads 0 used, when its core reports its memory', () => {
      const cards = [{ index: 0, k: 4096, mem: { memTotalMb: 16304, memUsedMb: 2900 } }];
      const r = rig(cards);
      r.login();
      const ticks = [];
      r.m.on('hashrate', (th, d) => ticks.push([th, d]));
      for (let i = 0; i < 3; i++) r.core(0).emit('hit', hitFor(K4));
      cards[0].fail = new Error('not enough free VRAM for the rank-128 profile');
      for (const s of r.submits()) r.answer(s.id, false);
      expect(ticks).toEqual([[0, { index: 0, name: 'Radeon 0', memTotalMb: 16304, memUsedMb: 0 }]]);
      r.m.stop();
    });

    test('a core that names no card falls back the same way, and a throw that is not an Error is said too', () => {
      const cards = [{ k: 4096 }];
      const r = rig(cards, { gpus: false });
      r.login();
      expect(r.m.cores[0].device).toBeNull();
      for (let i = 0; i < 3; i++) r.core().emit('hit', hitFor(K4));
      cards[0].fail = 'no luck';
      for (const s of r.submits()) r.answer(s.id, false);
      expect(r.lines).toContain('could not open the GPU again at k 2048: no luck');
      jest.advanceTimersByTime(10);
      expect(r.m.cores.map((c) => c.profile.k)).toEqual([2048]);
      r.m.stop();
    });
  });
});

// A card whose fold has not run on its kind of card in a release (core.untested; on AMD, RX 6000 and Instinct
// cards) mines with every hit it would send recomputed first (shared/miner/checkHit). The hits here are
// genuine, built in JS as the device builds them; a wrong one has one transcript word changed, as a fold that
// is wrong on the card would give, with its proofs and seeds right.
describe('PearlMiner — checking the hits of an untested card', () => {
  const { hitFor: genuine } = require('./helpers/pearlHit');
  const { penalizedAdjustmentFactor } = require('../src/shared/miner/pearlhash');
  const SMALL = { ...PROFILE, m: 64, n: 64 };
  const MAX = (1n << 256n) - 1n;
  // A pool target whose bound every hash meets, so a hit is found in the first region.
  const T = MAX / BigInt(penalizedAdjustmentFactor(SMALL));
  const EASY = T.toString(16).padStart(64, '0');
  const JOB = '00000000_2097152';
  const opts = { bound: shareBound(T, SMALL), jobId: JOB };
  const GOOD = genuine(SMALL, Buffer.from(HEADER, 'hex'), opts);
  const WRONG = genuine(SMALL, Buffer.from(HEADER, 'hex'), { ...opts, tamper: (jp) => { jp[5] ^= 1; } });
  const FOLD = 'dot4 128x256, 8 waves of 64x64, v_dot4 from registers, hash in its own kernel, batch 256 column '
    + 'offsets, k 2048, gfx1030';
  const said = (where) => where + ': this kind of card has not mined in a release yet, so each of its hits is '
    + 'checked here before it is sent; the first was right';
  const wrong = (where) => where + ': a hit came out wrong (jackpot hash mismatch (fold/transcript)). No share '
    + 'was sent, and this card is off for the rest of this run. This kind of card has not mined in a release '
    + 'yet: please report this line (earn/native/amd/README.md, "Testing on real hardware").';

  beforeEach(() => jest.useFakeTimers());
  afterEach(() => {
    jest.clearAllTimers();
    jest.useRealTimers();
  });

  // Cards as [{ index, untested, fold }]; an `untested` left out is a core that does not say (an older one).
  // With `gpus: false` the rig has no card list and its one core names no card. A card's `mem` is the memory
  // its core reports, { memTotalMb, memUsedMb }; left out, the core reports none.
  function rig(cards, { gpus = true } = {}) {
    const sock = makeSocket();
    const made = [];
    const createCore = jest.fn((profile, o) => {
      const card = gpus ? cards.find((c) => c.index === o.deviceIndex) : cards[0];
      const c = makeCore();
      if (gpus) c.device = { index: card.index, name: 'Radeon ' + card.index, fold: card.fold, ...card.mem };
      if (card.untested !== undefined) c.untested = card.untested;
      made.push({ core: c, card });
      return c;
    });
    const m = new PearlMiner({ connect: () => sock, createCore, reconnectMs: 0, restartMs: 10 });
    const ev = { log: [], error: [], stopped: [], hashrate: [] };
    m.on('log', (l) => ev.log.push(l));
    m.on('error', (e) => ev.error.push(e));
    m.on('stopped', (e) => ev.stopped.push(e));
    m.on('hashrate', (th, d) => ev.hashrate.push([th, d ? d.index : null]));
    m.start({
      ...settings, profile: SMALL, gpus: gpus ? cards.map((c) => ({ index: c.index, name: 'Radeon ' + c.index })) : [],
    });
    sock.emit('connect');
    sock.emit('data', '{"id":1,"result":true,"error":null}\n');
    sock.emit('data', jobLine({ target: EASY }));
    sock.written.length = 0;
    return {
      m, sock, made, ev, createCore,
      submits: () => sock.written.map((w) => JSON.parse(w)).filter((x) => x.method === 'mining.submit'),
      core: (index = cards[0].index) => made.filter((x) => x.card.index === index).pop().core,
      lines: (level) => ev.log.filter((l) => !level || l.level === level).map((l) => l.line),
    };
  }

  test('opens every core with checkHits, so the core may open an untested card', () => {
    const r = rig([{ index: 0, untested: true, fold: FOLD }]);
    expect(r.createCore.mock.calls[0][1]).toMatchObject({ checkHits: true, cardK: true });
    r.m.stop();
  });

  test('an untested card\'s right hits are checked and sent, and the check is said once', () => {
    const r = rig([{ index: 0, untested: true, fold: FOLD }]);
    r.core().emit('hit', GOOD);
    r.core().emit('hit', GOOD);
    expect(r.submits()).toHaveLength(2);
    expect(r.lines()).toContain(said('GPU 0 (Radeon 0) · gfx1030'));
    expect(r.lines().filter((l) => l.includes('each of its hits is checked'))).toHaveLength(1);
    expect(r.lines('error')).toEqual([]);
    r.m.stop();
  });

  test('a wrong hit is not sent, its card stops for the run, and the other cards keep mining', () => {
    const r = rig([{ index: 0, untested: true, fold: FOLD }, { index: 1, untested: false }]);
    const bad = r.core(0);
    r.core(0).emit('hashrate', 20);
    r.core(1).emit('hashrate', 30);
    bad.emit('hit', WRONG);
    expect(r.submits()).toEqual([]);
    expect(bad.stop).toHaveBeenCalledTimes(1);
    expect(r.lines('error')).toEqual([wrong('GPU 0 (Radeon 0) · gfx1030')]);
    expect(r.ev.hashrate.pop()).toEqual([0, 0]);
    expect(r.m.hashrate).toBe(30);
    expect(r.m.cores.map((c) => c.device.index)).toEqual([1]);
    expect([r.m.isRunning(), r.ev.error, r.ev.stopped]).toEqual([true, [], []]);
    // Nothing more from that card, not even a right hit still on its way; no restart; the next job is the
    // other card's alone.
    bad.emit('hit', GOOD);
    jest.advanceTimersByTime(60 * 1000);
    expect(r.createCore).toHaveBeenCalledTimes(2);
    r.sock.emit('data', jobLine({ job_id: '00000001_2097152', target: EASY }));
    expect(bad.setJob).toHaveBeenCalledTimes(1);
    expect(r.core(1).setJob).toHaveBeenCalledTimes(2);
    // A tested card's hits are not checked: this one, wrong, goes out as it is.
    r.core(1).emit('hit', { ...WRONG, jobId: '00000001_2097152' });
    expect(r.submits()).toHaveLength(1);
    expect(r.submits()[0].params.job_id).toBe('00000001_2097152');
    r.m.stop();
  });

  test('a core that does not say whether it is untested (an older core) is not checked', () => {
    const r = rig([{ index: 0, fold: FOLD }]);
    r.core().emit('hit', WRONG);
    expect(r.submits()).toHaveLength(1);
    expect(r.lines().filter((l) => l.includes('each of its hits is checked'))).toEqual([]);
    r.m.stop();
  });

  test('the last card going wrong ends the run, with its line as the error', () => {
    const r = rig([{ index: 0, untested: true, fold: FOLD }]);
    r.core().emit('hit', WRONG);
    expect(r.ev.error.map((e) => e.message)).toEqual([wrong('GPU 0 (Radeon 0) · gfx1030')]);
    expect(r.ev.stopped).toEqual([{ failed: true }]);
    expect(r.m.isRunning()).toBe(false);
  });

  test('a card whose architecture the core could not read, or that names no card, is said as such', () => {
    const r = rig([{ index: 0, untested: true, fold: 'ref 32x64, dot products from LDS, hash in its own kernel' }]);
    r.core().emit('hit', GOOD);
    expect(r.lines()).toContain(said('GPU 0 (Radeon 0) · unknown architecture'));
    r.m.stop();
    const quiet = rig([{ untested: true }], { gpus: false });
    expect(quiet.m.cores[0].device).toBeNull();
    quiet.core().emit('hit', WRONG);
    expect(quiet.lines('error')).toEqual([wrong('the GPU · unknown architecture')]);
    expect(quiet.ev.hashrate).toEqual([]);
    expect(quiet.ev.stopped).toEqual([{ failed: true }]);
  });

  // The Intel core's cards are all checked. Its fold description begins with
  // the fold and has no gfx name: on an Arc A750 rig these lines used to say
  // "unknown architecture" and send a wrong hit's report to the AMD README.
  test('an Intel card is named by its fold, and a wrong hit is reported to the Intel README', () => {
    const XMX8 = 'xmx8 (DPAS 8x8x32, sub-group 8), hardware DPAS, col_batch 256, band 256 row windows, k 2048';
    const DOT = 'dot (int8 dot products, local memory), col_batch 256, band 64 row windows, k 2048';
    const r = rig([{ index: 0, untested: true, fold: XMX8 }, { index: 1, untested: true, fold: DOT }]);
    r.core(0).emit('hit', GOOD);
    expect(r.lines()).toContain(said('GPU 0 (Radeon 0) · Intel xmx8 fold'));
    r.core(1).emit('hit', WRONG);
    expect(r.lines('error')).toEqual(['GPU 1 (Radeon 1) · Intel dot fold: a hit came out wrong (jackpot hash mismatch '
      + '(fold/transcript)). No share was sent, and this card is off for the rest of this run. This kind of card has '
      + 'not mined in a release yet: please report this line (earn/native/sycl/README.md, "Untested cards").']);
    r.m.stop();
    // A word that only starts like a fold's name is not one.
    const other = rig([{ index: 0, untested: true, fold: 'xmx8x (something else)' }]);
    other.core().emit('hit', GOOD);
    expect(other.lines()).toContain(said('GPU 0 (Radeon 0) · unknown architecture'));
    other.m.stop();
  });

  // The Intel core also reports its card's memory. A card taken off for a
  // wrong hit has its core closed, so it holds none of it (see "a card's
  // memory, as its core reports it").
  test('a card stopped for a wrong hit reads 0 used, when its core reports its memory', () => {
    const mem = { memTotalMb: 8128, memUsedMb: 1536 };
    const r = rig([{ index: 0, untested: true, fold: 'xmx8', mem }, { index: 1, untested: true, fold: 'xmx8', mem }]);
    const ticks = [];
    r.m.on('hashrate', (th, d) => ticks.push([th, d]));
    r.core(0).emit('hit', WRONG);
    expect(ticks).toEqual([[0, { index: 0, name: 'Radeon 0', memTotalMb: 8128, memUsedMb: 0 }]]);
    r.m.stop();
  });

  test('a wrong hit from a paused card\'s stopped core keeps the card from coming back', () => {
    const r = rig([{ index: 0, untested: true, fold: FOLD }, { index: 1, untested: false }]);
    const old = r.core(0);
    r.m.pauseCards([0]);
    old.emit('hit', WRONG);
    expect(r.lines('error')).toEqual([wrong('GPU 0 (Radeon 0) · gfx1030')]);
    expect(r.m.resumeCards([0])).toEqual([]);
    expect(r.createCore).toHaveBeenCalledTimes(2);
    expect(r.m.isRunning()).toBe(true);
    r.m.stop();
  });

  test('a wrong hit from the core before a resume stops the card\'s new core', () => {
    const r = rig([{ index: 0, untested: true, fold: FOLD }, { index: 1, untested: false }]);
    const old = r.core(0);
    old.emit('hit', GOOD);
    r.m.pauseCards([0]);
    r.m.resumeCards([0]);
    const resumed = r.core(0);
    expect(resumed).not.toBe(old);
    // The check is said once a run, not again for the resumed core.
    resumed.emit('hit', GOOD);
    expect(r.submits()).toHaveLength(2);
    expect(r.lines().filter((l) => l.includes('each of its hits is checked'))).toHaveLength(1);
    old.emit('hit', WRONG);
    expect(resumed.stop).toHaveBeenCalledTimes(1);
    expect(r.m.cores.map((c) => c.device.index)).toEqual([1]);
    resumed.emit('hit', GOOD);
    expect(r.submits()).toHaveLength(2);
    r.m.stop();
  });

  test('a wrong hit from a failed core keeps its waiting restart from opening the card', () => {
    const r = rig([{ index: 0, untested: true, fold: FOLD }, { index: 1, untested: false }]);
    const old = r.core(0);
    old.emit('error', new Error('GPU 0: a fault'));
    expect(r.lines()).toContain('restarting GPU 0 (Radeon 0) in 0 s');
    old.emit('hit', WRONG);
    jest.advanceTimersByTime(10);
    expect(r.createCore).toHaveBeenCalledTimes(2);
    expect(r.m.cores.map((c) => c.device.index)).toEqual([1]);
    expect(r.m.isRunning()).toBe(true);
    r.m.stop();
  });

  test('on a one-card rig, that wrong hit ends the run: no card is left to wait for', () => {
    const r = rig([{ index: 0, untested: true, fold: FOLD }]);
    const old = r.core(0);
    old.emit('error', new Error('GPU 0: a fault'));
    expect(r.m.isRunning()).toBe(true);
    old.emit('hit', WRONG);
    expect(r.ev.error.map((e) => e.message)).toEqual([wrong('GPU 0 (Radeon 0) · gfx1030')]);
    expect(r.ev.stopped).toEqual([{ failed: true }]);
    jest.advanceTimersByTime(10);
    expect(r.createCore).toHaveBeenCalledTimes(1);
  });

  test('a wrong hit on the last mining card leaves the run up while another card waits for its restart', () => {
    const r = rig([{ index: 0, untested: true, fold: FOLD }, { index: 1, untested: false }]);
    r.core(1).emit('error', new Error('GPU 1: a fault'));
    r.core(0).emit('hit', WRONG);
    expect(r.lines('error')).toContain(wrong('GPU 0 (Radeon 0) · gfx1030'));
    expect([r.m.isRunning(), r.ev.error, r.ev.stopped]).toEqual([true, [], []]);
    jest.advanceTimersByTime(10);
    expect(r.createCore).toHaveBeenCalledTimes(3);
    expect(r.m.cores.map((c) => c.device.index)).toEqual([1]);
    r.m.stop();
  });

  // buildShareProof would drop such a hit too, but one at a time: the card would mine on and send nothing.
  test('a hit whose proof does not verify stops the card the same way, with the check\'s reason', () => {
    const r = rig([{ index: 0, untested: true, fold: FOLD }, { index: 1, untested: false }]);
    const leafData = Buffer.from(GOOD.proofA.leafData);
    leafData[0] ^= 1;
    const bad = r.core(0);
    bad.emit('hit', { ...GOOD, proofA: { ...GOOD.proofA, leafData } });
    expect(r.submits()).toEqual([]);
    expect(bad.stop).toHaveBeenCalledTimes(1);
    expect(r.lines('error')).toEqual(['GPU 0 (Radeon 0) · gfx1030: a hit came out wrong (Merkle proof does not verify). '
      + 'No share was sent, and this card is off for the rest of this run. This kind of card has not mined in a '
      + 'release yet: please report this line (earn/native/amd/README.md, "Testing on real hardware").']);
    expect(r.m.cores.map((c) => c.device.index)).toEqual([1]);
    r.m.stop();
  });
});

// The RTX 5090's default, and --mine-mem-clock auto: a card on trial mines four
// turns, with and without the lock, and keeps the lock only if it was faster
// (shared/memClockTrial has the method; the shells' plan says which cards are
// on trial). It is handed to the driver first, so the first turn is the
// driver's clock whatever an earlier run left on the card.
describe('PearlMiner — the memory clock trial', () => {
  const { TURN_MS } = require('../src/shared/memClockTrial');
  const { planMemClocks } = require('../src/shared/memClock');
  const GPU = [{ index: 0, name: 'NVIDIA GeForce RTX 4090' }];
  const AUTO = { ...settings, gpus: GPU, mineMemClockByIndex: { 0: 5001 }, mineMemClockTrial: [0] };
  const START_LINE = 'memory clock trial on GPU 0: mining 75 s at the driver\'s clock and '
    + '75 s locked at 5001 MHz, twice; the lock is kept only if it is faster';
  const NO_RIGHTS = { ok: false, error: 'sudo: a password is required' };

  // `store` is where trial answers are kept between starts, `env` the
  // environment the miner reads, and `now` the time the run starts at.
  function rig(over = {}) {
    const calls = [];
    const t = { now: over.now || 0 };
    const clocks = {
      lockMemoryClock: jest.fn((index, mhz) => { calls.push(['lock', index, mhz]); return over.lock || { ok: true, error: null }; }),
      resetMemoryClock: jest.fn((index) => { calls.push(['reset', index]); return over.reset || { ok: true, error: null }; }),
    };
    const made = [];
    const createCore = jest.fn((profile, opts) => {
      const c = makeCore();
      if (!over.silentDevice) c.device = { index: opts.deviceIndex, name: 'GPU' + opts.deviceIndex };
      made.push(c);
      return c;
    });
    const m = new PearlMiner({
      connect: () => makeSocket(), createCore, reconnectMs: 0, clocks, now: () => t.now,
      memClockStore: over.store, env: over.env || {},
    });
    const logs = [];
    m.on('log', (l) => logs.push(l));
    m.on('stopped', () => calls.push(['stopped']));
    m.on('error', () => {});
    const clockLines = () => logs.map((l) => l.line).filter((l) => /memory clock/.test(l));
    return { m, t, clocks, calls, logs, made, createCore, clockLines };
  }

  // One turn of the trial: the card reports `th` twice a second, and once more
  // on the second the turn ends.
  function turn(b, core, th) {
    const end = b.t.now + TURN_MS;
    while (b.t.now + 500 < end) {
      b.t.now += 500;
      core.emit('hashrate', th);
    }
    b.t.now = end;
    core.emit('hashrate', th);
  }
  const trial = (b, core, rates) => rates.forEach((th) => turn(b, core, th));

  // The reset comes first. A run that was killed leaves its lock on the card,
  // and a trial that began there would time the lock against itself.
  test('a card on trial is handed to the driver, starts at its clock, and says what it will do', () => {
    const b = rig();
    b.m.start(AUTO);
    expect(b.calls).toEqual([['reset', 0]]);
    expect(b.logs).toContainEqual({ level: 'info', line: START_LINE });
    b.m.stop();
    // Stopped in its first, unlocked turn: there is no lock to release.
    expect(b.calls.slice(1)).toEqual([['stopped']]);
  });

  test('keeps a lock that was faster, says what it measured, and releases it on stop', () => {
    const b = rig();
    b.m.start(AUTO);
    trial(b, b.made[0], [317.5, 320.5, 317.5, 320.5]);
    // Locked for the second and fourth turns, and still locked.
    expect(b.calls).toEqual([['reset', 0], ['lock', 0, 5001], ['reset', 0], ['lock', 0, 5001]]);
    expect(b.clockLines()).toEqual([
      START_LINE,
      'memory clock kept at 5001 MHz on GPU 0 while mining: 320.5 TH/s locked against 317.5 (+0.9%)',
    ]);
    // The trial is over: later samples change nothing.
    turn(b, b.made[0], 100);
    expect(b.calls).toHaveLength(4);

    b.m.stop();
    expect(b.calls.slice(4)).toEqual([['reset', 0], ['stopped']]);
    expect(b.clockLines().pop()).toBe('memory clock released on GPU 0');
  });

  // The 810 MHz speed on rig01's RTX 4090.
  test('gives back a lock that was slower, and leaves the card alone after', () => {
    const b = rig();
    b.m.start(AUTO);
    trial(b, b.made[0], [317.5, 283.1, 317.5, 283.1]);
    expect(b.calls).toEqual([['reset', 0], ['lock', 0, 5001], ['reset', 0], ['lock', 0, 5001], ['reset', 0]]);
    expect(b.clockLines()).toEqual([
      START_LINE,
      'memory clock left to the driver on GPU 0 (the lock was not ahead by 0.3%): '
        + '283.1 TH/s locked against 317.5 (-10.8%)',
    ]);
    b.m.stop();
    expect(b.calls.slice(5)).toEqual([['stopped']]);
  });

  test('with too few samples to compare, leaves the clock to the driver and quotes no rates', () => {
    const b = rig();
    b.m.start(AUTO);
    // One sample a turn: the one that ends it.
    for (let i = 0; i < 4; i += 1) {
      b.t.now += TURN_MS;
      b.made[0].emit('hashrate', 317.5);
    }
    expect(b.clockLines().pop())
      .toBe('memory clock left to the driver on GPU 0 (too few hashrate samples to compare)');
    expect(b.calls.pop()).toEqual(['reset', 0]);
    b.m.stop();
  });

  // A rig that asked for auto without the rights for it: the reset that
  // starts the trial fails, so the trial is not run and the rig hears why at
  // once, not a turn in. The card is not touched again this run.
  test('a card that cannot be handed to the driver gets no trial, with a warning, once', () => {
    const b = rig({ reset: NO_RIGHTS });
    b.m.start(AUTO);
    expect(b.calls).toEqual([['reset', 0]]);
    expect(b.logs.filter((l) => /memory clock/.test(l.line))).toEqual([{
      level: 'warn',
      line: 'memory clock trial not run on GPU 0 (sudo: a password is required): setting the memory clock needs '
        + 'root or administrator rights for nvidia-smi. Mining continues at the default memory clock.',
    }]);
    trial(b, b.made[0], [317.5, 320.5, 317.5, 320.5]);
    expect(b.calls).toHaveLength(1);
    b.m.stop();
    expect(b.calls.slice(1)).toEqual([['stopped']]);
  });

  // A driver that hands the clock back and then will not lock it.
  test('a lock that cannot be taken ends the trial with a warning, once', () => {
    const b = rig({ lock: NO_RIGHTS });
    b.m.start(AUTO);
    trial(b, b.made[0], [317.5, 320.5, 317.5, 320.5]);
    expect(b.calls).toEqual([['reset', 0], ['lock', 0, 5001]]);
    expect(b.logs.filter((l) => l.level === 'warn')).toEqual([{
      level: 'warn',
      line: 'memory clock trial ended on GPU 0 (sudo: a password is required): setting the memory clock needs '
        + 'root or administrator rights for nvidia-smi. Mining continues at the default memory clock.',
    }]);
    b.m.stop();
    expect(b.calls.slice(2)).toEqual([['stopped']]);
  });

  test('a lock that will not come off between turns ends the trial, and the log names the fix', () => {
    const b = rig({ reset: NO_RIGHTS });
    // The reset that starts the trial works; the ones after it do not.
    b.clocks.resetMemoryClock.mockImplementationOnce((index) => {
      b.calls.push(['reset', index]);
      return { ok: true, error: null };
    });
    b.m.start(AUTO);
    trial(b, b.made[0], [317.5, 320.5, 317.5, 320.5]);
    expect(b.calls).toEqual([['reset', 0], ['lock', 0, 5001], ['reset', 0]]);
    expect(b.logs.filter((l) => l.level === 'error')).toEqual([{
      level: 'error',
      line: 'could not release the memory clock on GPU 0 (sudo: a password is required). '
        + 'An LLM on it will decode slowly until you run: sudo nvidia-smi -i 0 -rmc',
    }]);
    b.m.stop();
  });

  test('stopping in a locked turn releases the lock before stopped', () => {
    const b = rig();
    b.m.start(AUTO);
    turn(b, b.made[0], 317.5);
    expect(b.calls).toEqual([['reset', 0], ['lock', 0, 5001]]);
    b.m.stop();
    expect(b.calls.slice(2)).toEqual([['reset', 0], ['stopped']]);
  });

  // START LLM while mining-only is up: a model is about to be served from the
  // card, so the trial must not take the lock on its next turn.
  test('releaseMemClocks() ends the trial, in a locked turn or an unlocked one', () => {
    const locked = rig();
    locked.m.start(AUTO);
    turn(locked, locked.made[0], 317.5);
    locked.m.releaseMemClocks();
    expect(locked.calls).toEqual([['reset', 0], ['lock', 0, 5001], ['reset', 0]]);
    trial(locked, locked.made[0], [320.5, 317.5, 320.5]);
    expect(locked.calls).toHaveLength(3);
    locked.m.stop();

    const unlocked = rig();
    unlocked.m.start(AUTO);
    unlocked.m.releaseMemClocks();
    trial(unlocked, unlocked.made[0], [317.5, 320.5, 317.5, 320.5]);
    expect(unlocked.calls).toEqual([['reset', 0]]);
    unlocked.m.stop();
  });

  // The miner takes the plan as it is given. No plan of the shells' mixes the
  // two today: a request locks every card, and the default and auto try
  // every card they name.
  test('a card not on trial is locked outright beside one that is', () => {
    const b = rig();
    b.m.start({
      ...settings,
      gpus: [GPU[0], { index: 1, name: 'NVIDIA GeForce RTX 4090' }],
      mineMemClockByIndex: { 0: 5001, 1: 5001 },
      mineMemClockTrial: [0],
    });
    expect(b.calls).toEqual([['reset', 0], ['lock', 1, 5001]]);
    // The locked card's samples move nobody's trial.
    turn(b, b.made[1], 400);
    expect(b.calls).toHaveLength(2);
    b.m.stop();
  });

  test('a core that does not name its card gets no trial', () => {
    const b = rig({ silentDevice: true });
    b.m.start(AUTO);
    expect(b.clockLines()).toEqual([
      'memory clock left at default: this pearl_core.node does not say which GPU it opened',
    ]);
    expect(b.calls).toEqual([]);
    b.m.stop();
  });

  test('times the trial by the wall clock unless a clock is injected', () => {
    expect(new PearlMiner({}).now).toBe(Date.now);
  });

  describe('when the card is opened again in the same run', () => {
    beforeEach(() => jest.useFakeTimers());
    afterEach(() => {
      jest.clearAllTimers();
      jest.useRealTimers();
    });
    const fail = (b) => {
      b.made[b.made.length - 1].emit('error', new Error('GPU 0: misaligned address'));
      jest.advanceTimersByTime(RESTART_MS);
    };

    test('a trial cut short starts over', () => {
      const b = rig();
      b.m.start(AUTO);
      turn(b, b.made[0], 317.5);
      fail(b);
      // The lock its second turn held is released with the card, and the new
      // trial hands the card to the driver again before its first turn.
      expect(b.calls).toEqual([['reset', 0], ['lock', 0, 5001], ['reset', 0], ['reset', 0]]);
      expect(b.made).toHaveLength(2);
      expect(b.clockLines().filter((l) => l === START_LINE)).toHaveLength(2);
      // The dead core's trial is gone; the new one starts from its first turn.
      trial(b, b.made[1], [317.5, 320.5, 317.5, 320.5]);
      expect(b.clockLines().pop())
        .toBe('memory clock kept at 5001 MHz on GPU 0 while mining: 320.5 TH/s locked against 317.5 (+0.9%)');
      b.m.stop();
    });

    test('a kept lock is taken again at once, with no second trial', () => {
      const b = rig();
      b.m.start(AUTO);
      trial(b, b.made[0], [317.5, 320.5, 317.5, 320.5]);
      fail(b);
      expect(b.calls.slice(4)).toEqual([['reset', 0], ['lock', 0, 5001]]);
      expect(b.clockLines().filter((l) => l === START_LINE)).toHaveLength(1);
      expect(b.clockLines().pop()).toBe('memory clock locked at 5001 MHz on GPU 0 while mining');
      b.m.stop();
    });

    test('a refused lock stays refused', () => {
      const b = rig();
      b.m.start(AUTO);
      trial(b, b.made[0], [317.5, 283.1, 317.5, 283.1]);
      const before = b.calls.length;
      fail(b);
      expect(b.made).toHaveLength(2);
      trial(b, b.made[1], [317.5, 320.5, 317.5, 320.5]);
      expect(b.calls).toHaveLength(before);
      expect(b.clockLines().filter((l) => l === START_LINE)).toHaveLength(1);
      b.m.stop();
    });

    // The next start is a new run: the power limit may have changed.
    test('the next start tries the card again', () => {
      const b = rig();
      b.m.start(AUTO);
      trial(b, b.made[0], [317.5, 283.1, 317.5, 283.1]);
      b.m.stop();
      b.m.start(AUTO);
      expect(b.clockLines().filter((l) => l === START_LINE)).toHaveLength(2);
      b.m.stop();
    });
  });

  // An RTX 5090 gets this trial at 7001 MHz without being asked. It used to be
  // locked as its core opened, on one run on one card. The settings here are
  // the plan's own (shared/memClock), wired as both shells wire them, so a
  // plan the miner would read another way fails here.
  describe('the RTX 5090 default', () => {
    const RTX5090 = [{ index: 0, name: 'NVIDIA GeForce RTX 5090' }];
    const wired = (plan, gpus) => ({
      ...settings, gpus,
      mineMemClockByIndex: plan.byIndex, mineMemClockDefault: plan.isDefault, mineMemClockTrial: plan.trial,
    });
    // What a one-5090 rig starts its miner with, for a request (none: the default).
    const planned = (over = {}) => wired(
      planMemClocks({ cards: [{ index: 0, major: 12, minor: 0, driverMajor: 610 }], gpus: RTX5090, ...over }), RTX5090);
    const DEFAULT = planned();
    const START_7001 = 'memory clock trial on GPU 0: mining 75 s at the driver\'s clock and '
      + '75 s locked at 7001 MHz, twice; the lock is kept only if it is faster';
    const NOT_RUN = 'memory clock trial not run on GPU 0 (sudo: a password is required): setting the memory clock '
      + 'needs root or administrator rights for nvidia-smi. Mining continues at the default memory clock.';
    const ENDED = 'memory clock trial ended on GPU 0 (sudo: a password is required): setting the memory clock needs '
      + 'root or administrator rights for nvidia-smi. Mining continues at the default memory clock.';
    const clockLogs = (b) => b.logs.filter((l) => /memory clock/.test(l.line));

    test('is planned as a trial, not a lock', () => {
      expect(DEFAULT).toMatchObject({
        mineMemClockByIndex: { 0: 7001 }, mineMemClockDefault: true, mineMemClockTrial: [0],
      });
    });

    test('locks nothing as the card opens: it is handed to the driver and tried', () => {
      const b = rig();
      b.m.start(DEFAULT);
      expect(b.calls).toEqual([['reset', 0]]);
      expect(clockLogs(b)).toEqual([{ level: 'info', line: START_7001 }]);
      b.m.stop();
      expect(b.calls.slice(1)).toEqual([['stopped']]);
    });

    // The measured card at its 600 W cap: 95.7 TH/s at the driver's clock and
    // 103.4 locked (main/gpuClocks).
    test('keeps the lock on a card that is faster with it, and says what it measured', () => {
      const b = rig();
      b.m.start(DEFAULT);
      trial(b, b.made[0], [95.7, 103.4, 95.7, 103.4]);
      expect(b.calls).toEqual([['reset', 0], ['lock', 0, 7001], ['reset', 0], ['lock', 0, 7001]]);
      expect(clockLogs(b)).toEqual([
        { level: 'info', line: START_7001 },
        { level: 'info', line: 'memory clock kept at 7001 MHz on GPU 0 while mining: 103.4 TH/s locked against 95.7 (+8.0%)' },
      ]);
      b.m.stop();
      expect(b.calls.slice(4)).toEqual([['reset', 0], ['stopped']]);
    });

    // A card where the two clocks are level, as the measured card's own
    // earlier run had them. Locked outright, it would have mined at 7001 MHz
    // for nothing.
    test('gives the lock back on a card that is not faster with it, which ends at the driver\'s clock', () => {
      const b = rig();
      b.m.start(DEFAULT);
      trial(b, b.made[0], [158, 158.2, 158, 158.2]);
      expect(b.calls).toEqual([['reset', 0], ['lock', 0, 7001], ['reset', 0], ['lock', 0, 7001], ['reset', 0]]);
      expect(b.clockLines().pop()).toBe('memory clock left to the driver on GPU 0 (the lock was not ahead by 0.3%): '
        + '158.2 TH/s locked against 158.0 (+0.1%)');
      expect(b.m.memClockLocked.size).toBe(0);
      b.m.stop();
      expect(b.calls.slice(5)).toEqual([['stopped']]);
    });

    test('with too few samples to compare, ends at the driver\'s clock', () => {
      const b = rig();
      b.m.start(DEFAULT);
      for (let i = 0; i < 4; i += 1) {
        b.t.now += TURN_MS;
        b.made[0].emit('hashrate', 103.4);
      }
      expect(b.clockLines().pop())
        .toBe('memory clock left to the driver on GPU 0 (too few hashrate samples to compare)');
      expect(b.calls.pop()).toEqual(['reset', 0]);
      expect(b.m.memClockLocked.size).toBe(0);
      b.m.stop();
    });

    // The common GUI rig: no rights to set clocks. Nobody asked for the trial,
    // so it reads one line of information, not a warning on every start, and
    // the card is not touched again.
    test('on a rig that may not set clocks, is one info line and no trial', () => {
      const b = rig({ reset: NO_RIGHTS, lock: NO_RIGHTS });
      b.m.start(DEFAULT);
      trial(b, b.made[0], [95.7, 103.4, 95.7, 103.4]);
      expect(b.calls).toEqual([['reset', 0]]);
      expect(clockLogs(b)).toEqual([{ level: 'info', line: NOT_RUN }]);
      b.m.stop();
      expect(b.calls.slice(1)).toEqual([['stopped']]);
    });

    test('a lock the driver will not take ends the trial in an info line, at the driver\'s clock', () => {
      const b = rig({ lock: NO_RIGHTS });
      b.m.start(DEFAULT);
      trial(b, b.made[0], [95.7, 103.4, 95.7, 103.4]);
      expect(b.calls).toEqual([['reset', 0], ['lock', 0, 7001]]);
      expect(clockLogs(b)).toEqual([{ level: 'info', line: START_7001 }, { level: 'info', line: ENDED }]);
      expect(b.m.memClockLocked.size).toBe(0);
      b.m.stop();
      expect(b.calls.slice(2)).toEqual([['stopped']]);
    });

    // START LLM while mining-only is up. A model is about to be served from
    // the card: the lock goes, and the trial takes no other.
    test('releaseMemClocks() ends it, and the card stays at the driver\'s clock', () => {
      const b = rig();
      b.m.start(DEFAULT);
      turn(b, b.made[0], 95.7);
      expect(b.calls).toEqual([['reset', 0], ['lock', 0, 7001]]);
      b.m.releaseMemClocks();
      trial(b, b.made[0], [103.4, 95.7, 103.4]);
      expect(b.calls).toEqual([['reset', 0], ['lock', 0, 7001], ['reset', 0]]);
      expect(b.m.memClockLocked.size).toBe(0);
      b.m.stop();
    });

    // On the default the plan names the 5090 alone. The 4090 beside it is not
    // locked, tried or reset.
    test('on a mixed rig, tries the 5090 and leaves the other card alone', () => {
      const gpus = [{ index: 0, name: 'NVIDIA GeForce RTX 4090' }, { index: 1, name: 'NVIDIA GeForce RTX 5090' }];
      const plan = planMemClocks({
        cards: [{ index: 0, major: 8, minor: 9, driverMajor: 610 }, { index: 1, major: 12, minor: 0, driverMajor: 610 }],
        gpus,
      });
      const b = rig();
      b.m.start(wired(plan, gpus));
      trial(b, b.made[1], [95.7, 103.4, 95.7, 103.4]);
      turn(b, b.made[0], 317.5);
      expect(b.calls).toEqual([['reset', 1], ['lock', 1, 7001], ['reset', 1], ['lock', 1, 7001]]);
      expect(b.clockLines().every((l) => l.includes(' on GPU 1'))).toBe(true);
      b.m.stop();
    });

    // The ways past the trial and off it, from the same plan function.
    test('a request for 7001 still locks at once and keeps the lock, whatever the card mines at', () => {
      const b = rig();
      b.m.start(planned({ requestedMhz: 7001 }));
      expect(b.calls).toEqual([['lock', 0, 7001]]);
      expect(b.clockLines()).toEqual(['memory clock locked at 7001 MHz on GPU 0 while mining']);
      trial(b, b.made[0], [103.4, 95.7, 103.4, 95.7]);
      expect(b.calls).toHaveLength(1);
      b.m.stop();
      expect(b.calls.slice(1)).toEqual([['reset', 0], ['stopped']]);
    });

    test('0 leaves the card alone: no reset, no trial, no lock', () => {
      const b = rig();
      b.m.start(planned({ requestedMhz: 0 }));
      trial(b, b.made[0], [95.7, 103.4, 95.7, 103.4]);
      b.m.stop();
      expect(b.calls).toEqual([['stopped']]);
      expect(b.clockLines()).toEqual([]);
    });

    test('a co-running LLM leaves the card alone: no reset, no trial, no lock', () => {
      for (const requestedMhz of [null, 'auto', 7001]) {
        const b = rig();
        b.m.start(planned({ requestedMhz, llmCoRuns: true }));
        trial(b, b.made[0], [95.7, 103.4, 95.7, 103.4]);
        b.m.stop();
        expect(b.calls).toEqual([['stopped']]);
        expect(b.clockLines()).toEqual([]);
      }
    });

    // `auto` runs the same trial on a 5090. It was asked for, so a rig that
    // cannot run it gets a warning.
    test('auto is the same trial, with a warning when it cannot run', () => {
      const b = rig();
      b.m.start(planned({ requestedMhz: 'auto' }));
      expect(clockLogs(b)).toEqual([{ level: 'info', line: START_7001 }]);
      b.m.stop();

      const noRights = rig({ reset: NO_RIGHTS });
      noRights.m.start(planned({ requestedMhz: 'auto' }));
      expect(clockLogs(noRights)).toEqual([{ level: 'warn', line: NOT_RUN }]);
      noRights.m.stop();
    });

    describe('when the card stops and is opened again in the same run', () => {
      beforeEach(() => jest.useFakeTimers());
      afterEach(() => {
        jest.clearAllTimers();
        jest.useRealTimers();
      });
      const fail = (b) => {
        b.made[b.made.length - 1].emit('error', new Error('GPU 0: misaligned address'));
        jest.advanceTimersByTime(RESTART_MS);
      };

      test('a card that stops in a locked turn gives the lock back, and is tried again when it is back', () => {
        const b = rig();
        b.m.start(DEFAULT);
        turn(b, b.made[0], 95.7);
        b.made[0].emit('error', new Error('GPU 0: misaligned address'));
        // Stopped: no lock, and no trial left to take one.
        expect(b.calls).toEqual([['reset', 0], ['lock', 0, 7001], ['reset', 0]]);
        expect(b.clockLines().pop()).toBe('memory clock released on GPU 0');
        expect([b.m.memClockLocked.size, b.m.memClockTrials.size]).toEqual([0, 0]);
        jest.advanceTimersByTime(RESTART_MS);
        expect(b.made).toHaveLength(2);
        expect(b.calls.slice(3)).toEqual([['reset', 0]]);
        expect(b.clockLines().filter((l) => l === START_7001)).toHaveLength(2);
        b.m.stop();
      });

      // The lock its trial kept is taken again with the card. If the driver
      // will not give it this time, that is information: nobody asked for it.
      test('a kept lock is taken again at once, and one that cannot be is an info line', () => {
        const b = rig();
        b.m.start(DEFAULT);
        trial(b, b.made[0], [95.7, 103.4, 95.7, 103.4]);
        fail(b);
        expect(b.calls.slice(4)).toEqual([['reset', 0], ['lock', 0, 7001]]);
        expect(b.clockLines().filter((l) => l === START_7001)).toHaveLength(1);
        expect(clockLogs(b).pop()).toEqual({ level: 'info', line: 'memory clock locked at 7001 MHz on GPU 0 while mining' });

        b.clocks.lockMemoryClock.mockImplementation(() => NO_RIGHTS);
        fail(b);
        expect(clockLogs(b).pop()).toEqual({
          level: 'info',
          line: 'memory clock left at the driver\'s default on GPU 0 (sudo: a password is required): '
            + 'the RTX 5090 lock needs root or administrator rights for nvidia-smi',
        });
        expect(b.m.memClockLocked.size).toBe(0);
        b.m.stop();
      });
    });
  });

  // A trial's answer is kept for a week a card, so the next start does not
  // try the card again (shared/memClockTrial; the shells keep the answers in a
  // file, as they keep the RTX 30 launch shapes). A trial on every start cost
  // a card where the lock pays half of its first five minutes each time, and
  // a rig whose mining runs are shorter than a trial never kept the lock.
  describe('remembering the answer between starts', () => {
    const { REMEMBER_MS } = require('../src/shared/memClockTrial');
    const { shapeFileStore } = require('../src/shared/shapePick');
    const NOW = Date.UTC(2026, 10, 11, 12);
    const DAY = 24 * 3600 * 1000;
    const KEY = 'gpu0 NVIDIA GeForce RTX 4090';
    const memStore = (data = {}) => ({ data, get: (k) => data[k] || null, set: (k, v) => { data[k] = v; } });
    // What a trial at 5001 MHz on the RTX 4090 left in the store on 11 November.
    const kept = (over = {}) => ({ keep: true, mhz: 5001, at: NOW, onTh: 320.5, offTh: 317.5, ...over });
    const notKept = (over = {}) => ({ keep: false, mhz: 5001, at: NOW, onTh: 283.1, offTh: 317.5, ...over });
    const KEPT_LINE = 'memory clock on GPU 0: its trial of 2026-11-11 kept the 5001 MHz lock, '
      + '320.5 TH/s locked against 317.5 (+0.9%). Remembered: no trial until 2026-11-18.';
    const NOT_KEPT_LINE = 'memory clock left to the driver on GPU 0: its trial of 2026-11-11 did not keep the '
      + '5001 MHz lock, 283.1 TH/s locked against 317.5 (-10.8%). Remembered: no trial until 2026-11-18.';

    test('a finished trial writes its answer down under the card\'s key, and says until when', () => {
      const store = memStore();
      const b = rig({ store, now: NOW });
      b.m.start(AUTO);
      trial(b, b.made[0], [317.5, 320.5, 317.5, 320.5]);
      expect(Object.keys(store.data)).toEqual([KEY]);
      expect(store.data[KEY]).toMatchObject({ keep: true, mhz: 5001, at: NOW + 4 * TURN_MS });
      expect(store.data[KEY].onTh).toBeCloseTo(320.5, 6);
      expect(store.data[KEY].offTh).toBeCloseTo(317.5, 6);
      expect(b.clockLines().pop()).toBe('memory clock kept at 5001 MHz on GPU 0 while mining: '
        + '320.5 TH/s locked against 317.5 (+0.9%); remembered until 2026-11-18');
      b.m.stop();

      const slower = rig({ store: memStore(), now: NOW });
      slower.m.start(AUTO);
      trial(slower, slower.made[0], [317.5, 283.1, 317.5, 283.1]);
      expect(slower.m.memClockStore.data[KEY]).toMatchObject({ keep: false, mhz: 5001 });
      expect(slower.clockLines().pop()).toBe('memory clock left to the driver on GPU 0 (the lock was not ahead '
        + 'by 0.3%): 283.1 TH/s locked against 317.5 (-10.8%); remembered until 2026-11-18');
      // The lock came off before the answer was written: "no lock" is never
      // remembered for a card that is still locked.
      expect(slower.calls.pop()).toEqual(['reset', 0]);
      slower.m.stop();
    });

    test('a remembered "kept" is locked at once on the next start, with no trial, and the log says why', () => {
      const store = memStore({ [KEY]: kept() });
      const b = rig({ store, now: NOW + DAY });
      b.m.start(AUTO);
      // No reset and no first turn at the driver's clock: straight to the lock.
      expect(b.calls).toEqual([['lock', 0, 5001]]);
      expect(b.logs.filter((l) => /memory clock/.test(l.line))).toEqual([
        { level: 'info', line: KEPT_LINE },
        { level: 'info', line: 'memory clock locked at 5001 MHz on GPU 0 while mining' },
      ]);
      // Slower with it now, but nothing is timed: the answer stands for its week.
      trial(b, b.made[0], [317.5, 283.1, 317.5, 283.1]);
      expect(b.calls).toHaveLength(1);
      expect(store.data[KEY]).toEqual(kept());
      b.m.stop();
      expect(b.calls.slice(1)).toEqual([['reset', 0], ['stopped']]);
    });

    // The card is handed back to the driver once, as a trial's start does: a
    // run that was killed while it held a lock leaves it on the card.
    test('a remembered "not kept" hands the card to the driver: no lock, no trial', () => {
      const store = memStore({ [KEY]: notKept() });
      const b = rig({ store, now: NOW + DAY });
      b.m.start(AUTO);
      expect(b.calls).toEqual([['reset', 0]]);
      trial(b, b.made[0], [317.5, 320.5, 317.5, 320.5]);
      expect(b.logs.filter((l) => /memory clock/.test(l.line))).toEqual([{ level: 'info', line: NOT_KEPT_LINE }]);
      expect(store.data[KEY]).toEqual(notKept());
      b.m.stop();
      expect(b.calls).toEqual([['reset', 0], ['stopped']]);
    });

    // No rights now, where the trial had them. The answer stands; the rig is
    // told the card may still be locked. A warning for a trial that was asked
    // for, information for the RTX 5090's default.
    test('a remembered "not kept" whose card cannot be handed back says so', () => {
      const LINE = 'could not hand the memory clock on GPU 0 back to the driver (sudo: a password is required): '
        + 'if an earlier run left it locked, it still is. Setting the memory clock needs root or administrator '
        + 'rights for nvidia-smi.';
      const store = memStore({ [KEY]: notKept() });
      const b = rig({ store, now: NOW + DAY, reset: NO_RIGHTS });
      b.m.start(AUTO);
      expect(b.logs.filter((l) => /memory clock/.test(l.line))).toEqual([
        { level: 'info', line: NOT_KEPT_LINE }, { level: 'warn', line: LINE },
      ]);
      expect(store.data[KEY]).toEqual(notKept());
      b.m.stop();
      expect(b.calls).toEqual([['reset', 0], ['stopped']]);

      const byDefault = rig({ store: memStore({ [KEY]: notKept() }), now: NOW + DAY, reset: NO_RIGHTS });
      byDefault.m.start({ ...AUTO, mineMemClockDefault: true });
      expect(byDefault.logs.filter((l) => /memory clock/.test(l.line)).pop()).toEqual({ level: 'info', line: LINE });
      byDefault.m.stop();
    });

    // The trial's last turn is a locked one. If that lock will not come off,
    // "no lock" is not true of the card, so it is not written down: for a week
    // every start would have left the card locked at the slower setting.
    test('a lock that was not kept and will not come off is not remembered as "no lock"', () => {
      const store = memStore();
      const b = rig({ store, now: NOW });
      b.m.start(AUTO);
      trial(b, b.made[0], [317.5, 283.1, 317.5]);
      b.clocks.resetMemoryClock.mockImplementation((index) => { b.calls.push(['reset', index]); return NO_RIGHTS; });
      turn(b, b.made[0], 283.1);
      expect(b.calls.pop()).toEqual(['reset', 0]);
      expect(store.data).toEqual({});
      expect(b.logs.filter((l) => /memory clock/.test(l.line)).slice(-2)).toEqual([
        {
          level: 'error',
          line: 'could not release the memory clock on GPU 0 (sudo: a password is required). '
            + 'An LLM on it will decode slowly until you run: sudo nvidia-smi -i 0 -rmc',
        },
        {
          level: 'info',
          line: 'memory clock trial on GPU 0 did not keep the lock (the lock was not ahead by 0.3%): '
            + '283.1 TH/s locked against 317.5 (-10.8%), but the lock is still on the card; nothing is remembered, '
            + 'so the next start tries again',
        },
      ]);
      b.m.stop();
    });

    test('after a week the card is tried again, and the new answer replaces the old one', () => {
      const store = memStore({ [KEY]: notKept() });
      const b = rig({ store, now: NOW + REMEMBER_MS });
      b.m.start(AUTO);
      expect(b.clockLines()).toEqual([START_LINE]);
      trial(b, b.made[0], [317.5, 320.5, 317.5, 320.5]);
      expect(store.data[KEY]).toMatchObject({ keep: true, mhz: 5001, at: NOW + REMEMBER_MS + 4 * TURN_MS });
      expect(b.clockLines().pop()).toMatch(/^memory clock kept at 5001 MHz .*; remembered until 2026-11-25$/);
      b.m.stop();
    });

    // The key is the slot and the model in it.
    test('a different card in the same slot is tried afresh', () => {
      const store = memStore({ [KEY]: kept() });
      const b = rig({ store, now: NOW + DAY });
      b.m.start({ ...AUTO, gpus: [{ index: 0, name: 'NVIDIA GeForce RTX 4080 SUPER' }] });
      expect(b.calls).toEqual([['reset', 0]]);
      expect(b.clockLines()).toEqual([START_LINE]);
      trial(b, b.made[0], [200, 199, 200, 199]);
      // The other card's answer is still its own.
      expect(store.data[KEY]).toEqual(kept());
      expect(store.data['gpu0 NVIDIA GeForce RTX 4080 SUPER']).toMatchObject({ keep: false, mhz: 5001 });
      b.m.stop();
    });

    test('an answer for another clock is not this lock\'s: the card is tried', () => {
      const store = memStore({ [KEY]: kept({ mhz: 7001 }) });
      const b = rig({ store, now: NOW + DAY });
      b.m.start(AUTO);
      expect(b.calls).toEqual([['reset', 0]]);
      expect(b.clockLines()).toEqual([START_LINE]);
      b.m.stop();
    });

    test('LLMJOB_MINE_MEM_CLOCK_RETRY=1 tries the card again now, and keeps the new answer', () => {
      const store = memStore({ [KEY]: notKept() });
      const b = rig({ store, now: NOW + DAY, env: { LLMJOB_MINE_MEM_CLOCK_RETRY: '1' } });
      b.m.start(AUTO);
      expect(b.clockLines()).toEqual([START_LINE]);
      trial(b, b.made[0], [317.5, 320.5, 317.5, 320.5]);
      expect(store.data[KEY]).toMatchObject({ keep: true, at: NOW + DAY + 4 * TURN_MS });
      b.m.stop();
    });

    // No answer is not a no. Each of these leaves the store as it was, so the
    // next start tries the card again.
    test('a trial that could not run or finish remembers nothing', () => {
      const tried = (over, run) => {
        const store = memStore();
        const b = rig({ store, now: NOW, ...over });
        b.m.start(AUTO);
        run(b);
        b.m.stop();
        return store.data;
      };
      const full = (b) => trial(b, b.made[0], [317.5, 320.5, 317.5, 320.5]);
      // No rights: the card could not be handed to the driver, or not locked.
      expect(tried({ reset: NO_RIGHTS }, full)).toEqual({});
      expect(tried({ lock: NO_RIGHTS }, full)).toEqual({});
      // Too few samples to compare.
      expect(tried({}, (b) => {
        for (let i = 0; i < 4; i += 1) {
          b.t.now += TURN_MS;
          b.made[0].emit('hashrate', 317.5);
        }
      })).toEqual({});
      // The card stopped, or mining did, part way through.
      expect(tried({}, (b) => {
        turn(b, b.made[0], 317.5);
        b.made[0].emit('error', new Error('GPU 0: misaligned address'));
      })).toEqual({});
      expect(tried({}, (b) => turn(b, b.made[0], 317.5))).toEqual({});
      // START LLM joined the miner.
      expect(tried({}, (b) => { turn(b, b.made[0], 317.5); b.m.releaseMemClocks(); full(b); })).toEqual({});
    });

    // A number is a request: the card is locked at it whatever its trial
    // found, and the store is neither read for it nor written.
    test('an explicit clock locks at once and ignores what is remembered', () => {
      const store = memStore({ [KEY]: notKept() });
      const get = jest.spyOn(store, 'get');
      const b = rig({ store, now: NOW + DAY });
      b.m.start({ ...settings, gpus: GPU, mineMemClockByIndex: { 0: 5001 } });
      expect(b.calls).toEqual([['lock', 0, 5001]]);
      expect(b.clockLines()).toEqual(['memory clock locked at 5001 MHz on GPU 0 while mining']);
      trial(b, b.made[0], [317.5, 283.1, 317.5, 283.1]);
      b.m.stop();
      expect(get).not.toHaveBeenCalled();
      expect(store.data[KEY]).toEqual(notKept());
    });

    // 0 and a co-running LLM give the miner no plan at all.
    test('with no plan the store is not touched', () => {
      const store = memStore({ [KEY]: kept() });
      const get = jest.spyOn(store, 'get');
      const b = rig({ store, now: NOW + DAY });
      b.m.start({ ...settings, gpus: GPU, mineMemClockByIndex: {}, mineMemClockTrial: undefined });
      b.m.stop();
      expect(get).not.toHaveBeenCalled();
      expect(b.calls).toEqual([['stopped']]);
    });

    // The RTX 5090's default uses the same memory. Nobody asked for its lock,
    // so a remembered one the driver will not give now is information.
    test('the RTX 5090 default takes its remembered answer too', () => {
      const gpus = [{ index: 0, name: 'NVIDIA GeForce RTX 5090' }];
      const plan = planMemClocks({ cards: [{ index: 0, major: 12, minor: 0, driverMajor: 610 }], gpus });
      const wired = {
        ...settings, gpus,
        mineMemClockByIndex: plan.byIndex, mineMemClockDefault: plan.isDefault, mineMemClockTrial: plan.trial,
      };
      const answer = { keep: true, mhz: 7001, at: NOW, onTh: 103.4, offTh: 95.7 };
      const LINE = 'memory clock on GPU 0: its trial of 2026-11-11 kept the 7001 MHz lock, '
        + '103.4 TH/s locked against 95.7 (+8.0%). Remembered: no trial until 2026-11-18.';

      const b = rig({ store: memStore({ 'gpu0 NVIDIA GeForce RTX 5090': answer }), now: NOW + DAY });
      b.m.start(wired);
      expect(b.calls).toEqual([['lock', 0, 7001]]);
      expect(b.clockLines()).toEqual([LINE, 'memory clock locked at 7001 MHz on GPU 0 while mining']);
      b.m.stop();

      const noRights = rig({ store: memStore({ 'gpu0 NVIDIA GeForce RTX 5090': answer }), now: NOW + DAY, lock: NO_RIGHTS });
      noRights.m.start(wired);
      expect(noRights.logs.filter((l) => /memory clock/.test(l.line))).toEqual([
        { level: 'info', line: LINE },
        {
          level: 'info',
          line: 'memory clock left at the driver\'s default on GPU 0 (sudo: a password is required): '
            + 'the RTX 5090 lock needs root or administrator rights for nvidia-smi',
        },
      ]);
      noRights.m.stop();
    });

    // The file is the user's to edit. Rates that are not numbers are left out
    // of the line; the answer itself still stands.
    test('a remembered answer whose rates are not usable is said without them', () => {
      const PLAIN = 'memory clock on GPU 0: its trial of 2026-11-11 kept the 5001 MHz lock. '
        + 'Remembered: no trial until 2026-11-18.';
      for (const rates of [{ onTh: 'fast', offTh: 317.5 }, { onTh: 320.5, offTh: null }, { onTh: 320.5, offTh: 0 }]) {
        const b = rig({ store: memStore({ [KEY]: kept(rates) }), now: NOW + DAY });
        b.m.start(AUTO);
        expect(b.clockLines()).toEqual([PLAIN, 'memory clock locked at 5001 MHz on GPU 0 while mining']);
        b.m.stop();
      }
    });

    // Both shells keep the answers in a JSON file, through the store the
    // launch shapes use. Two starts over one file, as two runs of the app are.
    test('through the shells\' file store, one run\'s answer is the next run\'s lock', () => {
      const files = {};
      const fileStore = () => shapeFileStore('/d/mem-clock-trials.json', {
        readSettings: (f) => JSON.parse(JSON.stringify(files[f] || {})),
        writeSettings: (f, data) => { files[f] = JSON.parse(JSON.stringify(data)); },
        fs: {}, log: () => {},
      });
      const first = rig({ store: fileStore(), now: NOW });
      first.m.start(AUTO);
      trial(first, first.made[0], [317.5, 320.5, 317.5, 320.5]);
      first.m.stop();
      expect(Object.keys(files['/d/mem-clock-trials.json'])).toEqual([KEY]);

      const next = rig({ store: fileStore(), now: NOW + DAY });
      next.m.start(AUTO);
      expect(next.calls).toEqual([['lock', 0, 5001]]);
      expect(next.clockLines()[0]).toMatch(/^memory clock on GPU 0: its trial of 2026-11-11 kept the 5001 MHz lock, /);
      next.m.stop();
    });

    describe('when the card is opened again in the same run', () => {
      beforeEach(() => jest.useFakeTimers());
      afterEach(() => {
        jest.clearAllTimers();
        jest.useRealTimers();
      });

      test('it keeps the remembered answer, and the line is not said twice', () => {
        const b = rig({ store: memStore({ [KEY]: kept() }), now: NOW + DAY });
        b.m.start(AUTO);
        b.made[0].emit('error', new Error('GPU 0: misaligned address'));
        jest.advanceTimersByTime(RESTART_MS);
        expect(b.made).toHaveLength(2);
        expect(b.calls).toEqual([['lock', 0, 5001], ['reset', 0], ['lock', 0, 5001]]);
        expect(b.clockLines().filter((l) => l === KEPT_LINE)).toHaveLength(1);
        b.m.stop();

        const not = rig({ store: memStore({ [KEY]: notKept() }), now: NOW + DAY });
        not.m.start(AUTO);
        not.made[0].emit('error', new Error('GPU 0: misaligned address'));
        jest.advanceTimersByTime(RESTART_MS);
        expect(not.made).toHaveLength(2);
        // Handed to the driver when the answer was recalled, and not again.
        expect(not.calls).toEqual([['reset', 0]]);
        expect(not.clockLines()).toEqual([NOT_KEPT_LINE]);
        not.m.stop();
      });
    });
  });
});
