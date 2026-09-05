'use strict';

const nacl = require('tweetnacl');
const naclUtil = require('tweetnacl-util');
const crypto = require('crypto');
const {
  generateKeypair, fingerprint, adoptedNodeId, pingMessage, signMessage,
  buildJoinBody, buildTelemetry, signedBody, buildPingBody, NODE_ID_HEX,
} = require('../src/shared/node');

describe('generateKeypair / fingerprint', () => {
  test('makes a base64 Ed25519 keypair and a 16-hex fingerprint', () => {
    const kp = generateKeypair();
    expect(naclUtil.decodeBase64(kp.publicKey).length).toBe(32);
    expect(naclUtil.decodeBase64(kp.secretKey).length).toBe(64);
    expect(fingerprint(kp.publicKey)).toMatch(/^[0-9a-f]{16}$/);
  });

  test('fingerprint is stable per key and tolerates a nullish input', () => {
    expect(fingerprint('abc')).toBe(fingerprint('abc'));
    expect(fingerprint(null)).toMatch(/^[0-9a-f]{16}$/);
  });

  // 6 hex characters (24 bits) put two honest nodes on the same id with ~3%
  // probability at 1,000 nodes and 52% at 5,000 — and the loser was silently
  // unusable, its signed pings refused for a key mismatch. 16 characters
  // (64 bits) pushes the same 50% point past a billion nodes.
  test('is wide enough that the fleet will not collide', () => {
    expect(NODE_ID_HEX).toBe(16);
    expect(fingerprint('a')).not.toBe(fingerprint('b'));
  });

  // The server mints ids independently (nodeService.generateNodeFingerprint). If
  // the two ever disagree a client cannot address itself, so pin the exact
  // digest this end produces.
  test('is sha256(publicKey) truncated, which is what the server computes', () => {
    const expected = crypto.createHash('sha256').update('somekey').digest('hex').slice(0, 16);
    expect(fingerprint('somekey')).toBe(expected);
    expect(fingerprint('')).toBe(fingerprint(null));
  });
});

// The server is authoritative about which id a machine is enrolled under: it
// can legitimately answer with an id we did not compute (a machine whose old
// narrow id turns out to be taken by someone else's key enrolls on the wide one
// instead). Not adopting it left such a machine signing every later call as an
// id the server has no row for.
describe('adoptedNodeId', () => {
  test('adopts the id the server reports', () => {
    expect(adoptedNodeId('5840fc', 'a1b2c3d4e5f60789')).toBe('a1b2c3d4e5f60789');
    expect(adoptedNodeId('5840fc', ' a1b2c3d4e5f60789 ')).toBe('a1b2c3d4e5f60789');
  });

  test('keeps the local id when the server agrees or says nothing', () => {
    expect(adoptedNodeId('5840fc', '5840fc')).toBe('5840fc');
    expect(adoptedNodeId('5840fc', undefined)).toBe('5840fc');
    expect(adoptedNodeId('5840fc', null)).toBe('5840fc');
  });

  test('ignores anything that is not a plausible id', () => {
    // A garbled, proxied or hostile response must not rewrite our identity.
    for (const bad of ['', 'nope', '<script>', 'ABCDEF', '12345', 'f'.repeat(65), 42, {}]) {
      expect(adoptedNodeId('5840fc', bad)).toBe('5840fc');
    }
  });
});

describe('pingMessage / signMessage', () => {
  test('signs the challenge so the server (nacl.verify) accepts it', () => {
    const kp = generateKeypair();
    const msg = pingMessage('abc123', 1700000000000);
    expect(msg).toBe('abc123:1700000000000');
    const sig = signMessage(msg, kp.secretKey);
    const ok = nacl.sign.detached.verify(
      naclUtil.decodeUTF8(msg), naclUtil.decodeBase64(sig), naclUtil.decodeBase64(kp.publicKey),
    );
    expect(ok).toBe(true);
  });
});

describe('buildJoinBody', () => {
  test('passes through fields and honors a custom name', () => {
    expect(buildJoinBody({ token: 't', nodeId: 'abc123', publicKey: 'PK', name: '  My Rig ' }))
      .toEqual({ token: 't', nodeId: 'abc123', publicKey: 'PK', name: 'My Rig' });
  });
  test('defaults name to Node-<id> and token to empty; tolerates no args', () => {
    expect(buildJoinBody({ nodeId: 'abc123', publicKey: 'PK' }))
      .toEqual({ token: '', nodeId: 'abc123', publicKey: 'PK', name: 'Node-abc123' });
    expect(buildJoinBody()).toMatchObject({ token: '', name: 'Node-undefined' });
  });
});

describe('buildTelemetry', () => {
  test('maps live state, capabilities gated on ready', () => {
    expect(buildTelemetry({
      model: 'Gemma', quant: 'Q4_K_M', device: 'RTX 5090',
      vram: { totalMb: 32000, usedMb: 8000 }, tokensPerSec: 38.4, ready: true,
    })).toEqual({
      capabilities: ['chat'], activeJobs: 0, maxConcurrentJobs: 1,
      device: 'RTX 5090', vramTotal: 32000, vramUsed: 8000,
      model: 'Gemma', quant: 'Q4_K_M', tps: 38.4, name: null,
    });
  });

  test('carries the worker name for renames', () => {
    expect(buildTelemetry({ name: 'rig01' }).name).toBe('rig01');
  });

  test('nulls for missing data and empty capabilities when not ready', () => {
    expect(buildTelemetry()).toEqual({
      capabilities: [], activeJobs: 0, maxConcurrentJobs: 1,
      device: null, vramTotal: null, vramUsed: null, model: null, quant: null, tps: 0, name: null,
    });
    expect(buildTelemetry({ vram: { totalMb: NaN, usedMb: 5 }, ready: false }).vramTotal).toBeNull();
    expect(buildTelemetry({ vram: { totalMb: 5, usedMb: NaN } }).vramUsed).toBeNull();
  });

  test('reports activeJobs when serving cluster work', () => {
    expect(buildTelemetry({ activeJobs: 2 }).activeJobs).toBe(2);
    expect(buildTelemetry({ activeJobs: 'x' }).activeJobs).toBe(0);
  });
});

describe('signedBody', () => {
  test('signs the challenge and merges call-specific fields', () => {
    const kp = generateKeypair();
    const body = signedBody(
      { nodeId: 'abc123', publicKey: kp.publicKey, secretKey: kp.secretKey, timestamp: 1700000000000 },
      { chunkIndex: 3, content: 'hi', isFinal: true },
    );
    expect(body).toMatchObject({ nodeId: 'abc123', publicKey: kp.publicKey, timestamp: 1700000000000, chunkIndex: 3, content: 'hi', isFinal: true });
    const ok = nacl.sign.detached.verify(
      naclUtil.decodeUTF8('abc123:1700000000000'),
      naclUtil.decodeBase64(body.signature),
      naclUtil.decodeBase64(kp.publicKey),
    );
    expect(ok).toBe(true);
  });

  test('works with no extra fields', () => {
    const kp = generateKeypair();
    expect(signedBody({ nodeId: 'x', publicKey: kp.publicKey, secretKey: kp.secretKey, timestamp: 1 }))
      .toMatchObject({ nodeId: 'x', timestamp: 1 });
  });

  test('throws with no args (needs a secret key)', () => {
    expect(() => signedBody()).toThrow();
  });
});

describe('buildPingBody', () => {
  test('embeds a verifiable signature and folds in telemetry', () => {
    const kp = generateKeypair();
    const body = buildPingBody({
      nodeId: 'abc123', publicKey: kp.publicKey, secretKey: kp.secretKey,
      timestamp: 1700000000000, telemetry: { model: 'Gemma', tps: 5 },
    });
    expect(body).toMatchObject({ nodeId: 'abc123', publicKey: kp.publicKey, timestamp: 1700000000000, model: 'Gemma', tps: 5 });
    const ok = nacl.sign.detached.verify(
      naclUtil.decodeUTF8('abc123:1700000000000'),
      naclUtil.decodeBase64(body.signature),
      naclUtil.decodeBase64(kp.publicKey),
    );
    expect(ok).toBe(true);
  });

  test('works without telemetry', () => {
    const kp = generateKeypair();
    const body = buildPingBody({ nodeId: 'x', publicKey: kp.publicKey, secretKey: kp.secretKey, timestamp: 1 });
    expect(body.signature).toEqual(expect.any(String));
  });

  test('throws with no args (needs a secret key to sign)', () => {
    expect(() => buildPingBody()).toThrow();
  });
});
