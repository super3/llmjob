'use strict';

const {
  isLikelyAntivirusBlock, describeLaunchError, isTlsTrustError, describeSetupError,
} = require('../src/shared/engineError');

describe('isLikelyAntivirusBlock', () => {
  test('never on non-Windows', () => {
    expect(isLikelyAntivirusBlock({ platform: 'linux', missing: true })).toBe(false);
  });
  test('defaults to false with no arguments', () => {
    expect(isLikelyAntivirusBlock()).toBe(false);
  });
  test('true on Windows when the binary vanished from disk', () => {
    expect(isLikelyAntivirusBlock({ platform: 'win32', missing: true })).toBe(true);
  });
  test('true on a UNKNOWN spawn error code', () => {
    expect(isLikelyAntivirusBlock({ platform: 'win32', err: { code: 'UNKNOWN' } })).toBe(true);
  });
  test('true when the message carries an AV-shaped code (ENOENT)', () => {
    expect(isLikelyAntivirusBlock({ platform: 'win32', err: { message: 'spawn ENOENT' } })).toBe(true);
  });
  test('true using errno when code is absent (EACCES)', () => {
    expect(isLikelyAntivirusBlock({ platform: 'win32', err: { errno: 'EACCES' } })).toBe(true);
  });
  test('false for an unrelated Windows error', () => {
    expect(isLikelyAntivirusBlock({ platform: 'win32', err: { code: 'EPIPE', message: 'broken pipe' } })).toBe(false);
  });
  test('false on Windows with no error and nothing missing', () => {
    expect(isLikelyAntivirusBlock({ platform: 'win32' })).toBe(false);
  });
});

describe('describeLaunchError', () => {
  test('antivirus case: flags it and gives allow-it guidance', () => {
    const d = describeLaunchError({ platform: 'win32', missing: true });
    expect(d.antivirus).toBe(true);
    expect(d.ui).toMatch(/antivirus/i);
    expect(d.log).toMatch(/Defender/);
  });

  test('non-antivirus case includes the underlying message', () => {
    const d = describeLaunchError({ platform: 'linux', err: { message: 'boom' } });
    expect(d.antivirus).toBe(false);
    expect(d.ui).toMatch(/see Logs/i);
    expect(d.log).toBe('failed to launch engine: boom');
  });

  test('non-antivirus case falls back to a generic detail when no error given', () => {
    const d = describeLaunchError({ platform: 'linux' });
    expect(d.log).toBe('failed to launch engine: unknown error');
  });

  test('tolerates being called with no arguments', () => {
    const d = describeLaunchError();
    expect(d.antivirus).toBe(false);
    expect(d.log).toBe('failed to launch engine: unknown error');
  });
});

describe('isTlsTrustError', () => {
  test('false with nothing to go on', () => {
    expect(isTlsTrustError(null)).toBe(false);
    expect(isTlsTrustError(new Error('connect ECONNREFUSED'))).toBe(false);
  });
  test('true on the code Node raises for a missing intermediate', () => {
    expect(isTlsTrustError({ code: 'UNABLE_TO_VERIFY_LEAF_SIGNATURE' })).toBe(true);
  });
  test('true on an intercepting proxy/antivirus root', () => {
    expect(isTlsTrustError({ code: 'SELF_SIGNED_CERT_IN_CHAIN' })).toBe(true);
    expect(isTlsTrustError({ code: 'DEPTH_ZERO_SELF_SIGNED_CERT' })).toBe(true);
  });
  test('reads errno when code is absent', () => {
    expect(isTlsTrustError({ errno: 'UNABLE_TO_GET_ISSUER_CERT_LOCALLY' })).toBe(true);
  });
  test('matches the OpenSSL wording when the code was dropped', () => {
    // Exactly what the rig in the bug report showed.
    expect(isTlsTrustError(new Error('unable to verify the first certificate'))).toBe(true);
    expect(isTlsTrustError(new Error('self-signed certificate in certificate chain'))).toBe(true);
  });
  test('false for rejections extra anchors must never paper over', () => {
    expect(isTlsTrustError({ code: 'CERT_HAS_EXPIRED' })).toBe(false);
    expect(isTlsTrustError({ code: 'ERR_TLS_CERT_ALTNAME_INVALID' })).toBe(false);
  });
});

describe('describeSetupError', () => {
  const URL = 'https://pearl.alphapool.tech/downloads/alpha-miner-1.8.3';
  const PATH = '/home/u/.config/LLMJob Earn/engine/alpha-miner';

  test('certificate failure: explains the cause and both fixes', () => {
    const d = describeSetupError({
      err: new Error('unable to verify the first certificate'),
      downloadUrl: URL,
      manualPath: PATH,
    });
    expect(d.tls).toBe(true);
    expect(d.ui).toMatch(/certificate/i);
    expect(d.log).toContain('unable to verify the first certificate');
    expect(d.log).toMatch(/proxy, VPN or antivirus/);
    expect(d.log).toContain('ca-certificates');
    // The manual route has to name the artifact AND where it must land — the
    // user who hit this downloaded it by hand and still got nowhere.
    expect(d.log).toContain('Manual install: download ' + URL + ' and save it as ' + PATH);
  });

  test('other failures keep the underlying message and the manual route', () => {
    const d = describeSetupError({ err: new Error('HTTP 404 for ' + URL), downloadUrl: URL });
    expect(d.tls).toBe(false);
    expect(d.ui).toMatch(/see Logs/i);
    expect(d.log).toContain('engine setup failed: HTTP 404');
    expect(d.log).toContain('Manual install: download ' + URL + ', then start again.');
  });

  test('without a URL there is no manual hint to give', () => {
    expect(describeSetupError({ err: 'boom' }).log).toBe('engine setup failed: boom.');
  });

  test('tolerates being called with no arguments', () => {
    expect(describeSetupError().log).toBe('engine setup failed: unknown error.');
  });
});
