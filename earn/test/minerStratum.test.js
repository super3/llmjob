'use strict';

const {
  HEADER_BYTES, AUTHORIZE_ID, GZIP_TYPE,
  buildAuthorize, isKryptexHost, buildSubmit, encode,
  decodeTarget, parseMessage, parseJob, parseDifficulty, normalizeError,
} = require('../src/shared/miner/stratum');

// The exact bytes captured from us.pearl.herominers.com:1200 on 2026-08-23.
const HEADER = '000000203a49fea8b6d42c60c543fe0f029749787679372495e5d2d1007e29e3'
  + '25e1c08065a0cedc057ba091aabd10476017b3d3f4e38eafb59707ded57f71e1feb7b5d155918b6a0dea0018';
const TARGET = '00000000000007fff80000000000000000000000000000000000000000000000';
const JOB_LINE = JSON.stringify({
  id: null, method: 'mining.notify',
  params: { job_id: '00000000_2097152', header: HEADER, target: TARGET, height: 103353, cert_version: 3 },
});

describe('buildAuthorize', () => {
  // The live pool rejects the array form with "params must be an object" and
  // rejects configure/subscribe outright, so the object-param authorize IS the
  // whole handshake. Getting this wrong is an instant, total failure to mine.
  test('is a single object-param authorize under the fixed id', () => {
    expect(buildAuthorize('prl1pabc', 'rig01')).toEqual({
      id: AUTHORIZE_ID,
      method: 'mining.authorize',
      params: { wallet: 'prl1pabc', worker: 'rig01', type: 'v2' },
    });
    // The same with HeroMiners' host.
    expect(buildAuthorize('prl1pabc', 'rig01', 'us2.pearl.herominers.com')).toEqual(buildAuthorize('prl1pabc', 'rig01'));
  });

  // Kryptex's gzip extension: "type":"v2" asks for compressed proofs. Every
  // pool is asked, since one that does not offer them just answers without the
  // type. The agent names this miner, as Kryptex's documented message does.
  test('asks for compressed proofs, and names the miner when given an agent', () => {
    expect(GZIP_TYPE).toBe('v2');
    expect(buildAuthorize('prl1pabc', 'rig01', 'us.pearl.herominers.com', { agent: 'llmjob-earn/0.6.6' }).params)
      .toEqual({ wallet: 'prl1pabc', worker: 'rig01', agent: 'llmjob-earn/0.6.6', type: 'v2' });
    // No agent: no agent key, rather than an empty one.
    expect(buildAuthorize('prl1pabc', 'rig01', 'us.pearl.herominers.com', { agent: '' }).params)
      .toEqual({ wallet: 'prl1pabc', worker: 'rig01', type: 'v2' });
  });

  // For a pool that said v2 and then rejected what we sent: the login without
  // the type, which is the captured message exactly.
  test('gzip false leaves the type out', () => {
    expect(buildAuthorize('prl1pabc', 'rig01', 'us.pearl.herominers.com', { gzip: false }).params)
      .toEqual({ wallet: 'prl1pabc', worker: 'rig01' });
    expect(buildAuthorize('prl1pabc', 'rig01', 'us.pearl.herominers.com', { agent: 'llmjob-earn/0.6.6', gzip: false }).params)
      .toEqual({ wallet: 'prl1pabc', worker: 'rig01', agent: 'llmjob-earn/0.6.6' });
  });

  test('Kryptex gets wallet/worker, since it ignores the worker field', () => {
    expect(buildAuthorize('prl1pabc', 'kx-test', 'prl-us.kryptex.network').params)
      .toEqual({ wallet: 'prl1pabc/kx-test', worker: 'kx-test', type: 'v2' });
    // The slash is kept with the agent and with or without the type.
    expect(encode(buildAuthorize('prl1pabc', 'kx-test', 'prl-us.kryptex.network', { agent: 'llmjob-earn/0.6.6' })))
      .toBe('{"id":1,"method":"mining.authorize","params":{"wallet":"prl1pabc/kx-test","worker":"kx-test",'
        + '"agent":"llmjob-earn/0.6.6","type":"v2"}}\n');
    expect(buildAuthorize('prl1pabc', 'kx-test', 'prl-us.kryptex.network', { gzip: false }).params)
      .toEqual({ wallet: 'prl1pabc/kx-test', worker: 'kx-test' });
    expect(buildAuthorize('prl1pabc', 'rig', 'PRL.KRYPTEX.NETWORK').params.wallet).toBe('prl1pabc/rig');
    // No worker name: the wallet alone, which Kryptex also takes.
    expect(buildAuthorize('prl1pabc', '', 'prl-us.kryptex.network').params.wallet).toBe('prl1pabc');
    // Another pool, or a name that only ends like Kryptex's, gets the standard message.
    expect(buildAuthorize('prl1pabc', 'rig', 'pool.example').params.wallet).toBe('prl1pabc');
    expect(buildAuthorize('prl1pabc', 'rig', 'notkryptex.network').params.wallet).toBe('prl1pabc');
    expect(isKryptexHost('kryptex.network')).toBe(true);
    expect(isKryptexHost(null)).toBe(false);
  });
});

describe('encode', () => {
  test('is newline-terminated JSON', () => {
    expect(encode({ a: 1 })).toBe('{"a":1}\n');
  });
});

describe('decodeTarget', () => {
  // Big-endian uint256. This target is ~2^203 (leading zero bytes then 0x07ff…),
  // which matches the pool's own "target=2^202" log for the same job.
  test('reads a 32-byte big-endian target as a BigInt', () => {
    const t = decodeTarget(TARGET);
    expect(t).toBe(BigInt('0x' + TARGET));
    // sanity: it is between 2^202 and 2^203.
    expect(t > (1n << 202n)).toBe(true);
    expect(t < (1n << 203n)).toBe(true);
  });

  test('rejects anything that is not exactly 64 hex chars', () => {
    expect(decodeTarget('00ff')).toBeNull();
    expect(decodeTarget(TARGET + 'ff')).toBeNull();
    expect(decodeTarget('g'.repeat(64))).toBeNull();
    expect(decodeTarget('')).toBeNull();
    expect(decodeTarget(null)).toBeNull();
    expect(decodeTarget(undefined)).toBeNull();
  });

  test('is case-insensitive', () => {
    expect(decodeTarget(TARGET.toUpperCase())).toBe(decodeTarget(TARGET));
  });
});

describe('parseMessage — the live job', () => {
  test('parses a real mining.notify into a searchable job', () => {
    const m = parseMessage(JOB_LINE);
    expect(m.kind).toBe('job');
    expect(m.jobId).toBe('00000000_2097152');
    expect(m.header).toBeInstanceOf(Buffer);
    expect(m.header).toHaveLength(HEADER_BYTES);
    expect(m.headerHex).toBe(HEADER);
    expect(m.target).toBe(decodeTarget(TARGET));
    expect(m.height).toBe(103353);
    expect(m.certVersion).toBe(3);
  });
});

describe('parseMessage — results', () => {
  test('authorize success and failure are told apart by id', () => {
    expect(parseMessage('{"id":1,"error":null,"result":true}')).toEqual({
      kind: 'auth-ok', id: 1, error: null, gzip: false,
    });
    expect(parseMessage('{"id":1,"error":{"code":24,"msg":"bad address"},"result":null}')).toEqual({
      kind: 'auth-fail', id: 1, error: { code: 24, message: 'bad address' },
    });
  });

  // The reply decides whether proofs go compressed: only "type":"v2". Once a
  // pool has said it, it unpacks every submit, and a plain proof is an invalid
  // share; without it, a compressed proof is.
  test('a login reply with "type":"v2" means compressed proofs, and nothing else does', () => {
    // Verbatim, 2026-10-20: Kryptex and HeroMiners asked with the type, then
    // Kryptex asked without it.
    expect(parseMessage('{"id":1,"result":true,"error":null,"type":"v2"}')).toEqual({
      kind: 'auth-ok', id: 1, error: null, gzip: true,
    });
    expect(parseMessage('{"id":1,"error":null,"result":true,"type":"v2"}').gzip).toBe(true);
    expect(parseMessage('{"id":1,"result":true,"error":null}').gzip).toBe(false);
    // Any other type is not the one we asked for.
    expect(parseMessage('{"id":1,"result":true,"error":null,"type":"v1"}').gzip).toBe(false);
    expect(parseMessage('{"id":1,"result":true,"error":null,"type":"V2"}').gzip).toBe(false);
    expect(parseMessage('{"id":1,"result":true,"error":null,"type":2}').gzip).toBe(false);
    expect(parseMessage('{"id":1,"result":true,"error":null,"type":null}').gzip).toBe(false);
    // A refused login is a refused login, whatever type it carries.
    expect(parseMessage('{"id":1,"result":null,"error":{"code":24,"msg":"bad address"},"type":"v2"}')).toEqual({
      kind: 'auth-fail', id: 1, error: { code: 24, message: 'bad address' },
    });
  });

  // Some pools answer with ONLY an error key and no result. That still has to
  // classify as a verdict, not fall through to 'unknown' and strand the submit.
  test('a response carrying only an error is still a verdict', () => {
    expect(parseMessage('{"id":9,"error":[21,"Job not found"]}')).toEqual({
      kind: 'submit-rejected', id: 9, error: { code: 21, message: 'Job not found' },
    });
    expect(parseMessage('{"id":1,"error":null}')).toEqual({ kind: 'auth-ok', id: 1, error: null, gzip: false });
  });

  // No error is not the same as accepted. A share counts only on result true.
  test('a submit answered with no error but no true result is rejected, saying what came back', () => {
    expect(parseMessage('{"id":7,"result":false,"error":null}')).toEqual({
      kind: 'submit-rejected', id: 7, error: { code: null, message: 'no reason given (result false)' },
    });
    expect(parseMessage('{"id":7,"result":null,"error":null}').error.message).toBe('no reason given (result null)');
    expect(parseMessage('{"id":7,"error":null}').error.message).toBe('no reason given (no result)');
    expect(parseMessage('{"id":7,"result":"OK"}').error.message).toBe('no reason given (result "OK")');
    expect(parseMessage('{"id":7,"result":{"status":"' + 'x'.repeat(200) + '"}}').error.message)
      .toHaveLength('no reason given (result '.length + 80 + 1);
  });

  // A method we do not handle must not be mistaken for a result, even though it
  // carries neither result nor error.
  test('an unhandled method is unknown, not a verdict', () => {
    expect(parseMessage('{"method":"mining.ping","params":[]}').kind).toBe('unknown');
  });

  test('a submit result is accepted or rejected by its own id', () => {
    expect(parseMessage('{"id":7,"result":true,"error":null}')).toEqual({
      kind: 'submit-accepted', id: 7, error: null,
    });
    // Both pools, verbatim (2026-10-17): HeroMiners, then Kryptex.
    expect(parseMessage('{"id":100,"error":null,"result":true}').kind).toBe('submit-accepted');
    expect(parseMessage('{"id":100,"result":true,"error":null}').kind).toBe('submit-accepted');
    // The reference notes the submit-error array form [code,"text"].
    expect(parseMessage('{"id":8,"result":null,"error":[21,"Job not found"]}')).toEqual({
      kind: 'submit-rejected', id: 8, error: { code: 21, message: 'Job not found' },
    });
  });
});

describe('parseMessage — set_difficulty', () => {
  test('reads vardiff as a number, array, or object', () => {
    expect(parseMessage('{"method":"mining.set_difficulty","params":2000000}'))
      .toEqual({ kind: 'difficulty', difficulty: 2000000 });
    expect(parseDifficulty([4000000])).toEqual({ kind: 'difficulty', difficulty: 4000000 });
    expect(parseDifficulty({ difficulty: 8000000 })).toEqual({ kind: 'difficulty', difficulty: 8000000 });
    expect(parseDifficulty('nonsense')).toEqual({ kind: 'difficulty', difficulty: null });
  });
});

describe('parseMessage — malformed input never throws', () => {
  test('junk, empty and non-objects are tagged, not thrown', () => {
    expect(parseMessage('not json').kind).toBe('unparseable');
    expect(parseMessage('').kind).toBe('empty');
    expect(parseMessage('   ').kind).toBe('empty');
    expect(parseMessage(null).kind).toBe('empty');
    expect(parseMessage('123').kind).toBe('unknown'); // valid JSON, not an object
    expect(parseMessage('null').kind).toBe('unknown');
    expect(parseMessage('{"method":"mining.hello"}').kind).toBe('unknown');
  });
});

describe('parseJob — rank', () => {
  // HeroMiners does not state a rank, so it is normally null and the pool is
  // trusted. A pool that DOES state one lets the host refuse uncredited work.
  test('is null when unstated and read through when present', () => {
    expect(parseJob({ job_id: 'j', header: HEADER, target: TARGET }).rank).toBeNull();
    expect(parseJob({ job_id: 'j', header: HEADER, target: TARGET, rank: 128 }).rank).toBe(128);
    expect(parseJob({ job_id: 'j', header: HEADER, target: TARGET, rank: 512 }).rank).toBe(512);
    expect(parseJob({ job_id: 'j', header: HEADER, target: TARGET, rank: 'x' }).rank).toBeNull();
  });
});

describe('parseJob — rejects unusable jobs', () => {
  test('a short header, bad target, or missing id is bad-job', () => {
    const ok = { job_id: 'j', header: HEADER, target: TARGET };
    expect(parseJob(ok).kind).toBe('job');
    expect(parseJob({ ...ok, header: 'abcd' }).kind).toBe('bad-job');
    expect(parseJob({ ...ok, header: 'zz'.repeat(HEADER_BYTES) }).kind).toBe('bad-job');
    expect(parseJob({ ...ok, target: 'short' }).kind).toBe('bad-job');
    expect(parseJob({ ...ok, job_id: undefined }).kind).toBe('bad-job');
    expect(parseJob(null).kind).toBe('bad-job');
    expect(parseJob([1, 2]).kind).toBe('bad-job');
  });

  test('bad-job still reports the id when there is one, for the log', () => {
    expect(parseJob({ job_id: 'j9', header: 'short', target: TARGET }))
      .toEqual({ kind: 'bad-job', jobId: 'j9' });
  });

  test('height and cert_version default to null when absent', () => {
    const m = parseJob({ job_id: 'j', header: HEADER, target: TARGET });
    expect(m.height).toBeNull();
    expect(m.certVersion).toBeNull();
  });
});

describe('normalizeError', () => {
  test('reduces object, array, scalar and null to one shape', () => {
    expect(normalizeError({ code: 20, msg: 'x' })).toEqual({ code: 20, message: 'x' });
    expect(normalizeError({ code: 1, message: 'y' })).toEqual({ code: 1, message: 'y' });
    expect(normalizeError({ code: 1 })).toEqual({ code: 1, message: '' });
    expect(normalizeError([21, 'Job not found'])).toEqual({ code: 21, message: 'Job not found' });
    expect(normalizeError([])).toEqual({ code: null, message: '' });
    expect(normalizeError('boom')).toEqual({ code: null, message: 'boom' });
    expect(normalizeError(null)).toBeNull();
    // A code-less object and a code-less array both have to survive.
    expect(normalizeError({ msg: 'no code' })).toEqual({ code: null, message: 'no code' });
    expect(normalizeError([null, 'no code'])).toEqual({ code: null, message: 'no code' });
  });
});

describe('buildSubmit', () => {
  // The proof IS the submission. It carries m, n, k, the noise rank and both
  // Merkle proofs with their row indices, and the verifier rebuilds the mining
  // configuration from those indices rather than being told it -- so there is
  // no nonce and there are no seeds to send.
  //
  // This used to send wallet, worker, nonce, sigma and b_seed as well. The pool
  // reads none of them, and that shape never produced an accepted share.
  test('sends the job id, the proof and nothing else', () => {
    const msg = buildSubmit(7, { jobId: 'j1', plainProof: '3q0=', hashrate: 5.08e13 });
    expect(msg).toEqual({
      id: 7,
      method: 'mining.submit',
      params: { job_id: 'j1', plain_proof: '3q0=', hs: 50800000000000 },
    });
  });

  // hs is advisory and the pool scores on shares, but a NaN or a negative would
  // still be a malformed message.
  test('a missing or nonsense hashrate submits as zero', () => {
    expect(buildSubmit(1, { jobId: 'j' }).params.hs).toBe(0);
    expect(buildSubmit(1, { jobId: 'j', hashrate: -5 }).params.hs).toBe(0);
  });
});

describe('constants', () => {
  test('header width matches the protocol', () => {
    expect(HEADER_BYTES).toBe(76);
  });
});
