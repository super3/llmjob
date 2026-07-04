'use strict';

const { isLikelyAntivirusBlock, describeLaunchError } = require('../src/shared/engineError');

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
