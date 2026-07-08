'use strict';

const nacl = require('tweetnacl');
const naclUtil = require('tweetnacl-util');
const {
  generateKeypair, fingerprint, pingMessage, signMessage,
  buildJoinBody, buildTelemetry, buildPingBody,
} = require('../src/shared/node');

describe('generateKeypair / fingerprint', () => {
  test('makes a base64 Ed25519 keypair and a 6-hex fingerprint', () => {
    const kp = generateKeypair();
    expect(naclUtil.decodeBase64(kp.publicKey).length).toBe(32);
    expect(naclUtil.decodeBase64(kp.secretKey).length).toBe(64);
    expect(fingerprint(kp.publicKey)).toMatch(/^[0-9a-f]{6}$/);
  });

  test('fingerprint is stable per key and tolerates a nullish input', () => {
    expect(fingerprint('abc')).toBe(fingerprint('abc'));
    expect(fingerprint(null)).toMatch(/^[0-9a-f]{6}$/);
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
      model: 'Gemma', quant: 'Q4_K_M', tps: 38.4,
    });
  });

  test('nulls for missing data and empty capabilities when not ready', () => {
    expect(buildTelemetry()).toEqual({
      capabilities: [], activeJobs: 0, maxConcurrentJobs: 1,
      device: null, vramTotal: null, vramUsed: null, model: null, quant: null, tps: 0,
    });
    expect(buildTelemetry({ vram: { totalMb: NaN, usedMb: 5 }, ready: false }).vramTotal).toBeNull();
    expect(buildTelemetry({ vram: { totalMb: 5, usedMb: NaN } }).vramUsed).toBeNull();
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
