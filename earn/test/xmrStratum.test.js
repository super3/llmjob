'use strict';

const s = require('../src/shared/xmr/stratum');

const SEED = 'ab'.repeat(32);
const BLOB = '1010b4a4c7c806' + 'cd'.repeat(69);   // 76 bytes, as Monero's are
const JOB = {
  blob: BLOB, job_id: 'e9c4b9', target: 'b88d0600', algo: 'rx/0', height: 3781596, seed_hash: SEED,
};

describe('requests', () => {
  // The worker name goes in pass and rigid, never after the address: on
  // HeroMiners "." starts a payment ID.
  test('login carries the address, the worker name twice, the agent and the algorithms', () => {
    const msg = s.buildLogin({ address: ' 87wTMj ', worker: 'rig01', agent: 'llmjob-earn/0.5.14' });
    expect(msg).toEqual({
      id: s.LOGIN_ID,
      jsonrpc: '2.0',
      method: 'login',
      params: { login: '87wTMj', pass: 'rig01', agent: 'llmjob-earn/0.5.14', rigid: 'rig01', algo: ['rx/0'] },
    });
  });

  test('login without a worker name or agent uses "x" and the app name', () => {
    const msg = s.buildLogin({ address: null, worker: '  ' });
    expect(msg.params).toMatchObject({ login: '', pass: 'x', rigid: 'x', agent: 'llmjob-earn' });
    expect(s.buildLogin({ address: 'a' }).params.pass).toBe('x');
  });

  // Kryptex names the worker from "address/worker" in login, as its Pearl pool
  // does. HeroMiners and any other pool keep the standard login.
  test('login on Kryptex puts the worker after a slash; other pools are unchanged', () => {
    const kx = s.buildLogin({ address: '87wTMj', worker: 'kx-xmr-test', host: 'xmr-us.kryptex.network' });
    expect(kx.params).toMatchObject({ login: '87wTMj/kx-xmr-test', pass: 'kx-xmr-test', rigid: 'kx-xmr-test' });
    expect(s.buildLogin({ address: '87wTMj', host: 'xmr.kryptex.network' }).params.login).toBe('87wTMj/x');
    expect(s.buildLogin({ address: '87wTMj', worker: 'rig01', host: 'de.monero.herominers.com' }).params.login).toBe('87wTMj');
    expect(s.buildLogin({ address: '87wTMj', worker: 'rig01', host: 'pool.notkryptex.network.example' }).params.login).toBe('87wTMj');
  });

  test('submit and keepalive name the session', () => {
    expect(s.buildSubmit(7, 'sess', { jobId: 'j', nonce: '3a000000', result: 'ff' })).toEqual({
      id: 7, jsonrpc: '2.0', method: 'submit', params: { id: 'sess', job_id: 'j', nonce: '3a000000', result: 'ff' },
    });
    expect(s.buildKeepalive(8, 'sess')).toEqual({ id: 8, jsonrpc: '2.0', method: 'keepalived', params: { id: 'sess' } });
  });

  test('encode writes one JSON line', () => {
    expect(s.encode({ a: 1 })).toBe('{"a":1}\n');
  });
});

describe('targets', () => {
  // XMRig's arithmetic: 2^64-1 / ((2^32-1) / t32). "b88d0600" is t32 =
  // 0x00068db8, which is difficulty 10000.
  test('a compact 32-bit target stands for its 64-bit target', () => {
    const t = s.parseTarget('b88d0600');
    expect(t).toBe(0xffffffffffffffffn / (0xffffffffn / 0x00068db8n));
    expect(s.targetDifficulty(t)).toBe(10000);
  });

  test('a 16-digit target is the 64-bit target itself, little-endian', () => {
    expect(s.parseTarget('0000000000000001')).toBe(1n << 56n);
    expect(s.parseTarget('FFFFFFFFFFFFFFFF')).toBe(0xffffffffffffffffn);
  });

  test('anything else is no target', () => {
    for (const bad of [null, '', 'b88d06', 'b88d0600aa', 'zz8d0600', '00000000', '0000000000000000']) {
      expect(s.parseTarget(bad)).toBeNull();
    }
  });

  test('the worker takes the 64-bit target as 16 big-endian hex digits', () => {
    expect(s.workerTarget(0x1234n)).toBe('0000000000001234');
    expect(s.workerTarget(s.parseTarget('ffffffffffffffff'))).toBe('ffffffffffffffff');
  });
});

describe('parseJob', () => {
  test('a good job, ready for the worker', () => {
    const j = s.parseJob(JOB);
    expect(j).toEqual({
      kind: 'job',
      jobId: 'e9c4b9',
      blob: BLOB,
      target: s.parseTarget('b88d0600'),
      difficulty: 10000,
      seedHash: SEED,
      height: 3781596,
      algo: 'rx/0',
      variant: 'v1',
      commit: false,
    });
  });

  test('upper-case hex is folded, and a job with no algo or height is rx/0 at an unknown height', () => {
    const j = s.parseJob(Object.assign({}, JOB, { blob: BLOB.toUpperCase(), seed_hash: SEED.toUpperCase(), algo: '', height: '7' }));
    expect(j).toMatchObject({ blob: BLOB, seedHash: SEED, algo: 'rx/0', height: null });
    expect(s.parseJob(Object.assign({}, JOB, { algo: undefined })).algo).toBe('rx/0');
    expect(s.parseJob(Object.assign({}, JOB, { job_id: 42 })).jobId).toBe('42');
  });

  test('says what is wrong with a job it cannot mine', () => {
    const bad = (patch) => s.parseJob(Object.assign({}, JOB, patch));
    expect(s.parseJob(null)).toEqual({ kind: 'bad-job', jobId: null, reason: 'no job_id' });
    expect(s.parseJob([])).toMatchObject({ kind: 'bad-job', reason: 'no job_id' });
    expect(bad({ job_id: '' }).reason).toBe('no job_id');
    expect(bad({ blob: 'abc' }).reason).toMatch(/^the blob is not 43-512 bytes/);
    expect(bad({ blob: 'zz'.repeat(76) }).reason).toMatch(/^the blob/);
    expect(bad({ blob: 'aa'.repeat(42) }).reason).toMatch(/^the blob/);
    expect(bad({ blob: 'aa'.repeat(513) }).reason).toMatch(/^the blob/);
    expect(bad({ blob: 5 }).reason).toMatch(/^the blob/);
    expect(bad({ target: 'nope' })).toEqual({ kind: 'bad-job', jobId: 'e9c4b9', reason: 'unreadable target "nope"' });
    expect(bad({ seed_hash: 'ab' }).reason).toBe('no RandomX seed_hash');
    expect(bad({ seed_hash: null }).reason).toBe('no RandomX seed_hash');
    expect(bad({ algo: 'rx/wow' }).reason).toBe('algorithm rx/wow is not one this miner mines');
  });

  // Monero's v17 fork moves to RandomX v2 with commitments; the table is
  // where its pool name will go, and the worker already mines both.
  test('only algorithms in the table are mined', () => {
    expect(Object.keys(s.ALGOS)).toEqual(['rx/0']);
    expect(s.DEFAULT_ALGO).toBe('rx/0');
    expect(s.NONCE_OFFSET).toBe(39);
    expect(s.MIN_BLOB_BYTES).toBe(43);
  });
});

describe('parseMessage', () => {
  const line = (o) => JSON.stringify(o);

  test('a login result with a session and a job', () => {
    const m = s.parseMessage(line({
      id: 1, jsonrpc: '2.0', error: null,
      result: { id: 'abc123', job: JOB, extensions: ['algo', 'keepalive'], status: 'OK' },
    }));
    expect(m.kind).toBe('login-ok');
    expect(m.sessionId).toBe('abc123');
    expect(m.job).toMatchObject({ kind: 'job', jobId: 'e9c4b9' });
    expect(m.extensions).toEqual(['algo', 'keepalive']);
  });

  test('a login result with no job or extensions', () => {
    const m = s.parseMessage(line({ id: 1, result: { id: 7 } }));
    expect(m).toEqual({ kind: 'login-ok', sessionId: '7', job: null, extensions: [] });
  });

  test('a refused login, with the pool\'s reason or without one', () => {
    expect(s.parseMessage(line({ id: 1, error: { code: -1, message: 'Invalid address used for login' }, result: null })))
      .toEqual({ kind: 'login-fail', error: { code: -1, message: 'Invalid address used for login' } });
    expect(s.parseMessage(line({ id: 1, error: null, result: { status: 'OK' } })))
      .toEqual({ kind: 'login-fail', error: { code: null, message: 'the pool sent no session id' } });
    expect(s.parseMessage(line({ id: 1, error: null, result: null })).kind).toBe('login-fail');
    expect(s.parseMessage(line({ id: 1, error: 'busy' })).error).toEqual({ code: null, message: 'busy' });
  });

  test('a job notification, good or bad', () => {
    expect(s.parseMessage(line({ jsonrpc: '2.0', method: 'job', params: JOB })).kind).toBe('job');
    expect(s.parseMessage(line({ method: 'job', params: { job_id: 'x' } })).kind).toBe('bad-job');
  });

  test('submit replies', () => {
    expect(s.parseMessage(line({ id: 12, error: null, result: { status: 'OK' } })))
      .toEqual({ kind: 'submit-accepted', id: 12 });
    expect(s.parseMessage(line({ id: 13, error: { code: -1, message: 'Low difficulty share' }, result: null })))
      .toEqual({ kind: 'submit-rejected', id: 13, error: { code: -1, message: 'Low difficulty share' } });
    expect(s.parseMessage(line({ id: 13, error: { message: null } })).error).toEqual({ code: null, message: '' });
  });

  test('keepalive replies are told apart by the ids the host sent them under', () => {
    const ids = new Set([20]);
    expect(s.parseMessage(line({ id: 20, error: null, result: { status: 'KEEPALIVED' } }), ids))
      .toEqual({ kind: 'keepalive', id: 20, error: null });
    expect(s.parseMessage(line({ id: 21, result: { status: 'OK' } }), ids).kind).toBe('submit-accepted');
  });

  test('an error the pool sends on its own', () => {
    expect(s.parseMessage(line({ id: null, error: { code: -1, message: 'Unauthenticated' } })))
      .toEqual({ kind: 'pool-error', error: { code: -1, message: 'Unauthenticated' } });
  });

  test('never throws on junk', () => {
    expect(s.parseMessage('')).toEqual({ kind: 'empty' });
    expect(s.parseMessage(null)).toEqual({ kind: 'empty' });
    expect(s.parseMessage('{nope')).toEqual({ kind: 'unparseable', raw: '{nope' });
    expect(s.parseMessage('[1]').kind).toBe('unknown');
    expect(s.parseMessage('3').kind).toBe('unknown');
    expect(s.parseMessage(line({ method: 'mining.notify' })).kind).toBe('unknown');
    expect(s.parseMessage(line({ id: 5 })).kind).toBe('unknown');
    expect(s.parseMessage(line({ id: null, result: true })).kind).toBe('unknown');
  });
});

describe('errorText', () => {
  test('names the code when there is one', () => {
    expect(s.errorText({ code: -1, message: 'Low difficulty share' })).toBe('[-1] Low difficulty share');
    expect(s.errorText({ code: null, message: 'x' })).toBe('x');
    expect(s.errorText({ code: 3, message: '' })).toBe('[3] unknown error');
    expect(s.errorText(null)).toBe('unknown error');
  });

  test('normalizeError keeps code and message', () => {
    expect(s.normalizeError(null)).toBeNull();
    expect(s.normalizeError([1, 'a'])).toEqual({ code: null, message: '1,a' });
  });
});
