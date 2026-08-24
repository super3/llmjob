'use strict';

const {
  HEADER_BYTES, TARGET_BYTES, AUTHORIZE_ID,
  buildAuthorize, buildSubmit, serializeProof, encode,
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
      params: { wallet: 'prl1pabc', worker: 'rig01' },
    });
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
      kind: 'auth-ok', id: 1, error: null,
    });
    expect(parseMessage('{"id":1,"error":{"code":24,"msg":"bad address"},"result":null}')).toEqual({
      kind: 'auth-fail', id: 1, error: { code: 24, message: 'bad address' },
    });
  });

  // Some pools answer with ONLY an error key and no result. That still has to
  // classify as a verdict, not fall through to 'unknown' and strand the submit.
  test('a response carrying only an error is still a verdict', () => {
    expect(parseMessage('{"id":9,"error":[21,"Job not found"]}')).toEqual({
      kind: 'submit-rejected', id: 9, error: { code: 21, message: 'Job not found' },
    });
    expect(parseMessage('{"id":1,"error":null}')).toEqual({ kind: 'auth-ok', id: 1, error: null });
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

describe('buildSubmit / serializeProof', () => {
  test('assembles the documented v2 object-param submit', () => {
    const msg = buildSubmit(7, {
      wallet: 'prl1pabc', worker: 'rig01', jobId: 'j1', nonce: 42,
      aSeed: 'aa', bSeed: 'bb', proof: Buffer.from([0xde, 0xad]),
    });
    expect(msg).toEqual({
      id: 7,
      method: 'mining.submit',
      params: {
        wallet: 'prl1pabc', worker: 'rig01', job_id: 'j1', nonce: 42,
        type: 'v2', sigma: 'aa', b_seed: 'bb', plain_proof: '3q0=',
      },
    });
  });

  test('serializeProof base64-encodes bytes and tolerates non-buffers', () => {
    expect(serializeProof(Buffer.from([0, 1, 2]))).toBe('AAEC');
    expect(serializeProof([0, 1, 2])).toBe('AAEC');
    expect(serializeProof(null)).toBe('');
  });
});

describe('constants', () => {
  test('header and target widths match the protocol', () => {
    expect(HEADER_BYTES).toBe(76);
    expect(TARGET_BYTES).toBe(32);
  });
});
