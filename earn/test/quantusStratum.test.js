'use strict';

const s = require('../src/shared/quantus/stratum');
const { targetFromDifficulty } = require('../src/shared/quantus/qpow');

// LuckyPool's replies as recorded and published by not-only-mining-pool
// (engine/quantus/testdata/luckypool_login.json and luckypool_notify.json,
// Apache-2.0), verbatim.
const LUCKY_LOGIN = '{"id":1,"result":{"extensions":["keepalive"],"id":"7577b955-142a-4420-be97-cb3c54e84f4c","job":{'
  + '"algo":"qpow-poseidon2","difficulty":10000000000,"extranonce":"0004ee45","job_id":"22729-10000000000",'
  + '"mining_hash":"415c416d00ae68342369dbc22db3b8f805a1b1a0ea0848cf08116ab894e46753","seq":22729,'
  + '"target":"000000006df37f675ef6eadf5ab9a2072d44268d97df837e6748956e5c6c2117501e68855669e4b8356cf292464d9e16cc8d4655f8fb96f429b149fc87d74da4"},'
  + '"status":"OK"}}';
const LUCKY_NOTIFY = '{"jsonrpc":"2.0","method":"job","params":{"clean_jobs":true,"job":{"algo":"qpow-poseidon2",'
  + '"difficulty":10000000000,"extranonce":"0004f005","job_id":"22768-10000000000",'
  + '"mining_hash":"d8797e03ff633526267709d6d2f2e32652cf03c57a429e7ca90017a7d1ee0c8f","seq":22768,'
  + '"target":"000000006df37f675ef6eadf5ab9a2072d44268d97df837e6748956e5c6c2117501e68855669e4b8356cf292464d9e16cc8d4655f8fb96f429b149fc87d74da4"}}}';
// Captured 13 Nov 2026 with a login that is not an address (see the module).
const KRYPTEX_BAD_LOGIN = '{"id":1,"result":null,"error":{"code":-1,"message":"Invalid login"}}';
const HERO_BAD_LOGIN = '{"id":1,"error":{"code":-20,"message":"Invalid Quantus address"},"result":null}';

const JOB = {
  job_id: 'j1', mining_hash: 'ab'.repeat(32), target: '00000000ffffffff' + '00'.repeat(56), difficulty: 4294967297,
  extranonce: '0a0b', algo: 'qpow-poseidon2', seq: 7, height: 402800,
};

describe('requests', () => {
  test('login: the address, the worker in pass and rigid, and the agent', () => {
    const msg = s.buildLogin({ address: ' qzABC ', worker: 'rig01', agent: 'llmjob-earn/0.7.0', host: 'us2.quantus.herominers.com' });
    expect(msg).toEqual({
      id: 1, jsonrpc: '2.0', method: 'login',
      params: { login: 'qzABC', pass: 'rig01', agent: 'llmjob-earn/0.7.0', rigid: 'rig01' },
    });
  });

  test('login on Kryptex names the worker as address/worker', () => {
    expect(s.buildLogin({ address: 'qzABC', worker: 'rig01', host: 'qtc-us.kryptex.network' }).params.login).toBe('qzABC/rig01');
  });

  test('login with no worker, address or agent', () => {
    expect(s.buildLogin({}).params).toEqual({ login: '', pass: 'x', agent: 'llmjob-earn', rigid: 'x' });
  });

  test('submit and keepalive carry the session id', () => {
    expect(s.buildSubmit(5, 'sess', { jobId: 'j1', nonce: 'ff' })).toEqual({
      id: 5, jsonrpc: '2.0', method: 'submit', params: { id: 'sess', job_id: 'j1', nonce: 'ff' },
    });
    expect(s.buildKeepalive(6, 'sess')).toEqual({ id: 6, jsonrpc: '2.0', method: 'keepalived', params: { id: 'sess' } });
    expect(s.encode({ a: 1 })).toBe('{"a":1}\n');
  });
});

describe('LuckyPool\'s recorded replies', () => {
  test('the login reply: session, extensions and the first job', () => {
    const m = s.parseMessage(LUCKY_LOGIN);
    expect(m).toMatchObject({ kind: 'login-ok', sessionId: '7577b955-142a-4420-be97-cb3c54e84f4c', extensions: ['keepalive'] });
    expect(m.job).toMatchObject({
      kind: 'job', jobId: '22729-10000000000', difficulty: 10000000000n, seq: 22729, height: null, algo: 'qpow-poseidon2',
    });
    expect(m.job.miningHash.toString('hex')).toBe('415c416d00ae68342369dbc22db3b8f805a1b1a0ea0848cf08116ab894e46753');
    expect(m.job.extranonce.toString('hex')).toBe('0004ee45');
    // The target LuckyPool sent is (2^512 - 1) / difficulty.
    expect(m.job.target).toBe(targetFromDifficulty(10000000000n));
  });

  test('the job notification', () => {
    const m = s.parseMessage(LUCKY_NOTIFY);
    expect(m).toMatchObject({ kind: 'job', jobId: '22768-10000000000', seq: 22768 });
    expect(m.extranonce.toString('hex')).toBe('0004f005');
  });
});

describe('replies seen from Kryptex and HeroMiners', () => {
  test.each([['Kryptex', KRYPTEX_BAD_LOGIN, -1, 'Invalid login'], ['HeroMiners', HERO_BAD_LOGIN, -20, 'Invalid Quantus address']])(
    '%s refusing a login', (_, line, code, message) => {
      expect(s.parseMessage(line)).toEqual({ kind: 'login-fail', error: { code, message } });
    });
});

describe('jobs', () => {
  test('a job at the top level of params works too', () => {
    const m = s.parseJob(JOB);
    expect(m).toMatchObject({ kind: 'job', jobId: 'j1', difficulty: 4294967297n, seq: 7, height: 402800 });
    expect(m.target).toBe(0xffffffffn << 448n);
  });

  test('only a difficulty: the target is worked out', () => {
    const m = s.parseJob({ ...JOB, target: undefined, difficulty: '1000' });
    expect(m.target).toBe(targetFromDifficulty(1000n));
    expect(m.difficulty).toBe(1000n);
  });

  test('only a target: the difficulty is worked out', () => {
    const m = s.parseJob({ ...JOB, difficulty: undefined });
    expect(m.difficulty).toBe(4294967297n);
  });

  test('no algorithm and no extranonce are fine', () => {
    const m = s.parseJob({ ...JOB, algo: '', extranonce: undefined, seq: 'x' });
    expect(m).toMatchObject({ kind: 'job', algo: null, seq: null });
    expect(m.extranonce).toHaveLength(0);
  });

  test.each([
    [{ ...JOB, job_id: undefined }, 'no job_id'],
    [{ ...JOB, mining_hash: 'abc' }, 'mining_hash is not 32 bytes of hex'],
    [{ ...JOB, mining_hash: 5 }, 'mining_hash is not 32 bytes of hex'],
    [{ ...JOB, algo: 'rx/0' }, 'algorithm rx/0 is not one this miner mines'],
    [{ ...JOB, extranonce: 'zz' }, 'extranonce is not hex of 32 bytes or fewer'],
    [{ ...JOB, extranonce: '00'.repeat(33) }, 'extranonce is not hex of 32 bytes or fewer'],
    [{ ...JOB, target: 'ff' }, 'target is not 64 bytes of hex'],
    [{ ...JOB, target: '00'.repeat(64) }, 'the target is 0'],
    [{ ...JOB, difficulty: 0 }, 'unreadable difficulty 0'],
    [{ ...JOB, difficulty: 1.5 }, 'unreadable difficulty 1.5'],
    [{ ...JOB, target: undefined, difficulty: undefined }, 'no target and no difficulty'],
  ])('refused: %j', (job, reason) => {
    expect(s.parseJob(job)).toEqual({ kind: 'bad-job', jobId: job.job_id || null, reason });
  });

  test('params that are not an object', () => {
    expect(s.parseJob(null)).toEqual({ kind: 'bad-job', jobId: null, reason: 'no job_id' });
    expect(s.parseJob([1])).toEqual({ kind: 'bad-job', jobId: null, reason: 'no job_id' });
    expect(s.parseJob({ job: [1] })).toEqual({ kind: 'bad-job', jobId: null, reason: 'no job_id' });
  });

  test('difficulties', () => {
    expect(s.readDifficulty(' 12 ')).toBe(12n);
    expect(s.readDifficulty('0')).toBe(null);
    expect(s.readDifficulty('-1')).toBe(null);
    expect(s.readDifficulty(null)).toBe(null);
  });
});

describe('other replies', () => {
  test('submit accepted, as {"status":"OK"} or true', () => {
    expect(s.parseMessage('{"id":7,"result":{"status":"OK"},"error":null}')).toEqual({ kind: 'submit-accepted', id: 7 });
    expect(s.parseMessage('{"id":7,"result":true}')).toEqual({ kind: 'submit-accepted', id: 7 });
  });

  test('submit rejected, with a reason or without one', () => {
    expect(s.parseMessage('{"id":7,"result":null,"error":{"code":-1,"message":"Low difficulty share"}}'))
      .toEqual({ kind: 'submit-rejected', id: 7, error: { code: -1, message: 'Low difficulty share' } });
    expect(s.parseMessage('{"id":7,"result":false}'))
      .toEqual({ kind: 'submit-rejected', id: 7, error: { code: null, message: 'no reason given (result false)' } });
    expect(s.parseMessage('{"id":7,"error":null}'))
      .toEqual({ kind: 'submit-rejected', id: 7, error: { code: null, message: 'no reason given (result undefined)' } });
  });

  test('keepalive replies are told from submits by id', () => {
    const ids = new Set([9]);
    expect(s.parseMessage('{"id":9,"result":{"status":"KEEPALIVED"}}', ids)).toEqual({ kind: 'keepalive', id: 9, error: null });
  });

  test('a login reply with no session id fails', () => {
    expect(s.parseMessage('{"id":1,"result":{"status":"OK"}}'))
      .toEqual({ kind: 'login-fail', error: { code: null, message: 'the pool sent no session id' } });
    expect(s.parseMessage('{"id":1,"result":{"id":"x"}}')).toEqual({ kind: 'login-ok', sessionId: 'x', job: null, extensions: [] });
  });

  test('errors with no id, unknown methods, junk', () => {
    expect(s.parseMessage('{"id":null,"error":"banned"}')).toEqual({ kind: 'pool-error', error: { code: null, message: 'banned' } });
    expect(s.parseMessage('{"id":null,"result":1}')).toEqual({ kind: 'unknown', raw: '{"id":null,"result":1}' });
    expect(s.parseMessage('{"method":"mining.notify"}')).toEqual({ kind: 'unknown', raw: '{"method":"mining.notify"}' });
    expect(s.parseMessage('{"id":3}')).toEqual({ kind: 'unknown', raw: '{"id":3}' });
    expect(s.parseMessage('[1]')).toEqual({ kind: 'unknown', raw: '[1]' });
    expect(s.parseMessage('not json')).toEqual({ kind: 'unparseable', raw: 'not json' });
    expect(s.parseMessage('  ')).toEqual({ kind: 'empty' });
    expect(s.parseMessage(null)).toEqual({ kind: 'empty' });
    expect(s.errorText({ code: 1, message: 'm' })).toBe('[1] m');
  });
});
