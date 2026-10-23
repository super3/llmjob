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
} = require('../src/main/pearlMiner');
const gpuClocks = require('../src/main/gpuClocks');
const { encode } = require('../src/shared/miner/stratum');
const { buildShareProof } = require('../src/shared/miner/shareProof');
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
    expect(b.m.pending.get(sent.id)).toEqual({ jobId: '00000000_2097152', index: 1, gzip: false });
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
    expect(r.said()).toContain('authorized; the pool takes compressed share proofs');
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
      deviceIndex: 1, saltStride: 2, saltBase: 1 + 2 * SALT_RUN,
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
    expect(b.createCore.mock.calls[1][1]).toEqual({ deviceIndex: 1, saltStride: 1, saltBase: SALT_RUN });
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

// The memory clock lock (the RTX 5090 default, or --mine-mem-clock). The
// shells decide which cards and what clock (shared/memClock) and hand the miner
// { index: mhz }; the lock is taken per card as its core starts and released in
// stop(), before 'stopped', because the demand gate starts llama-server --
// which needs the full memory clock -- the moment it hears the miner stopped.
describe('PearlMiner — the memory clock lock', () => {
  const GPUS = [
    { index: 0, name: 'NVIDIA GeForce RTX 5090' },
    { index: 1, name: 'NVIDIA GeForce RTX 5090' },
  ];
  // A requested lock on both cards, as planMemClocks gives it for --mine-mem-clock.
  const MEM = { ...settings, mineMemClockByIndex: { 0: 7001, 1: 7001 } };
  // The same two cards on the RTX 5090 default.
  const DEFAULT = { ...MEM, mineMemClockDefault: true };

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
    b.m.start({ ...DEFAULT, gpus: GPUS });
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
      b.m.start({ ...DEFAULT, gpus: GPUS });
      b.m.releaseMemClocks();
      const first = b.createCore.mock.results[0].value;
      first.emit('error', new Error('GPU 0: misaligned address'));
      jest.advanceTimersByTime(RESTART_MS);
      expect(b.createCore).toHaveBeenCalledTimes(3);
      expect(b.calls).toEqual([['lock', 0, 7001], ['lock', 1, 7001], ['reset', 0], ['reset', 1]]);

      b.m.stop();
      b.m.start({ ...DEFAULT, gpus: [GPUS[0]] });
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

  // A mixed rig on the default: the plan names the 5090 and not the 4090
  // beside it, and the 4090 is left at the driver's clock, in silence.
  test('locks only the cards the plan names', () => {
    const b = rig();
    b.m.start({ ...settings, gpus: GPUS, mineMemClockByIndex: { 1: 7001 }, mineMemClockDefault: true });
    expect(b.calls).toEqual([['lock', 1, 7001]]);
    expect(b.lines().filter((l) => /memory clock/.test(l))).toHaveLength(1);
    b.m.stop();
    expect(b.calls.slice(1)).toEqual([['reset', 1], ['stopped']]);
  });

  // Nobody asked for the default, so the line says where it came from. It does
  // not name the off switch: these lines reach the GUI too, which has no flag,
  // so that is the shell's plan line (shared/memClock). A requested lock says
  // nothing extra.
  test('a default lock says so; a requested one does not', () => {
    const b = rig();
    b.m.start({ ...DEFAULT, gpus: [GPUS[0]] });
    expect(b.logs).toContainEqual({
      level: 'info',
      line: 'memory clock locked at 7001 MHz on GPU 0 while mining (the default on the RTX 5090)',
    });
    const asked = rig();
    asked.m.start({ ...MEM, gpus: [GPUS[0]] });
    expect(asked.lines()).toContain('memory clock locked at 7001 MHz on GPU 0 while mining');
  });

  // The common case: a GUI rig without the rights to set clocks. The user did
  // not ask for the lock, so losing it is one info line, not a warning on
  // every start -- and the card is not "released" at stop. The line names
  // neither sudo nor a flag: on Windows the fix is running as administrator.
  test('a failed default lock is one info line, and the card is not reset', () => {
    const b = rig({ lock: { ok: false, error: 'sudo: a password is required' } });
    expect(b.m.start({ ...DEFAULT, gpus: [GPUS[0]] })).toBe(true);
    expect(b.logs.filter((l) => /memory clock/.test(l.line))).toEqual([{
      level: 'info',
      line: 'memory clock left at the driver\'s default on GPU 0 (sudo: a password is required): '
        + 'the RTX 5090 lock needs root or administrator rights for nvidia-smi',
    }]);
    b.m.stop();
    expect(b.clocks.resetMemoryClock).not.toHaveBeenCalled();
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
    expect(b.createCore.mock.calls[2][1]).toEqual({ deviceIndex: 0, saltStride: 2, saltBase: 0 + 2 * SALT_RUN });
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
