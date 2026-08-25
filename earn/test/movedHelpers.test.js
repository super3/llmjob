'use strict';

// Three small pure functions that used to live in shared/engine.js and
// shared/engineError.js. Those modules were entirely about downloading and
// launching alpha-miner and went with it; these three were never
// engine-specific, so they moved to the modules that actually use them.
//
// Their tests moved here with them. Each one is small enough to look obviously
// correct and has been wrong in production at least once, which is exactly the
// combination that earns a unit test.

jest.mock('child_process', () => ({ execFile: jest.fn(), spawn: jest.fn() }));

const { progressPercent, isTlsTrustError } = require('../src/main/io');
const { parseDriverMajor } = require('../src/main/probe');
const { isArchiveUrl, looksLikeArchive } = require('../src/main/llmEngineManager');

describe('progressPercent', () => {
  test('is a rounded percentage of the total', () => {
    expect(progressPercent(0, 200)).toBe(0);
    expect(progressPercent(50, 200)).toBe(25);
    expect(progressPercent(199, 200)).toBe(100); // rounds up at the very end
    expect(progressPercent(200, 200)).toBe(100);
  });

  // Servers may send no content-length at all. A percentage of an unknown total
  // is not 0% — it is "no idea", and the caller prints nothing rather than a
  // progress bar stuck at zero.
  test('an unknown or nonsense total has no percentage', () => {
    expect(progressPercent(10, 0)).toBeNull();
    expect(progressPercent(10, null)).toBeNull();
    expect(progressPercent(10, undefined)).toBeNull();
    expect(progressPercent(10, -5)).toBeNull();
  });

  // A proxy that reports a total smaller than what it then sends would
  // otherwise print 143%.
  test('clamps rather than exceeding 100 or dropping below 0', () => {
    expect(progressPercent(300, 200)).toBe(100);
    expect(progressPercent(-10, 200)).toBe(0);
  });
});

describe('isTlsTrustError', () => {
  // A missing trust anchor is worth retrying with more anchors; these are the
  // codes that mean exactly that.
  test('recognises a missing trust anchor by code', () => {
    for (const code of ['UNABLE_TO_VERIFY_LEAF_SIGNATURE', 'UNABLE_TO_GET_ISSUER_CERT',
      'SELF_SIGNED_CERT_IN_CHAIN', 'DEPTH_ZERO_SELF_SIGNED_CERT', 'CERT_UNTRUSTED']) {
      expect(isTlsTrustError({ code })).toBe(true);
    }
    expect(isTlsTrustError({ errno: 'CERT_UNTRUSTED' })).toBe(true);
  });

  // Some layers hand it on as a plain Error with the OpenSSL reason in the text
  // and no code at all.
  test('recognises it by wording when there is no code', () => {
    expect(isTlsTrustError(new Error('unable to verify the first certificate'))).toBe(true);
    expect(isTlsTrustError(new Error('self signed certificate in chain'))).toBe(true);
    expect(isTlsTrustError(new Error('unable to get local issuer certificate'))).toBe(true);
  });

  // Expiry and hostname mismatches are real rejections. Retrying with extra CAs
  // would paper over a certificate that genuinely should not be trusted.
  test('is not fooled by rejections no extra anchor should fix', () => {
    expect(isTlsTrustError(new Error('certificate has expired'))).toBe(false);
    expect(isTlsTrustError({ code: 'CERT_HAS_EXPIRED' })).toBe(false);
    expect(isTlsTrustError({ code: 'ERR_TLS_CERT_ALTNAME_INVALID' })).toBe(false);
    expect(isTlsTrustError(new Error('socket hang up'))).toBe(false);
    expect(isTlsTrustError(null)).toBe(false);
    expect(isTlsTrustError(undefined)).toBe(false);
    expect(isTlsTrustError({})).toBe(false);
  });
});

describe('parseDriverMajor', () => {
  test('takes the major version out of an nvidia-smi string', () => {
    expect(parseDriverMajor('576.88')).toBe(576);
    expect(parseDriverMajor('  580.65  ')).toBe(580);
    expect(parseDriverMajor('Driver Version: 550.120')).toBe(550);
  });

  // No driver is not driver zero: the caller must be able to tell "old" from
  // "could not read it", because only one of those is worth warning about.
  test('anything unreadable is null, not zero', () => {
    expect(parseDriverMajor('')).toBeNull();
    expect(parseDriverMajor(null)).toBeNull();
    expect(parseDriverMajor(undefined)).toBeNull();
    expect(parseDriverMajor('no nvidia-smi here')).toBeNull();
    expect(parseDriverMajor('576')).toBeNull(); // no minor → not a version string
  });
});

describe('archive detection', () => {
  test('recognises an archive by extension', () => {
    expect(isArchiveUrl('https://x/y.zip')).toBe(true);
    expect(isArchiveUrl('https://x/y.tar.gz')).toBe(true);
    expect(isArchiveUrl('https://x/y.tgz')).toBe(true);
    expect(isArchiveUrl('https://x/y.TGZ')).toBe(true);
    expect(isArchiveUrl('https://x/llama-server')).toBe(false);
    expect(isArchiveUrl('https://x/y.zip.sig')).toBe(false);
  });

  // By CONTENT, because the download lands under a format-neutral name. A file
  // that is really an archive sitting where the binary belongs must not be
  // trusted as the binary.
  test('recognises gzip and zip by their magic bytes', () => {
    expect(looksLikeArchive(Buffer.from([0x1f, 0x8b, 0x08, 0x00]))).toBe(true);
    expect(looksLikeArchive(Buffer.from([0x50, 0x4b, 0x03, 0x04]))).toBe(true);
    expect(looksLikeArchive(Buffer.from([0x7f, 0x45, 0x4c, 0x46]))).toBe(false); // ELF
    expect(looksLikeArchive(Buffer.from([0x1f]))).toBe(false);                   // truncated
    expect(looksLikeArchive(Buffer.from([0x50, 0x4b]))).toBe(false);
    expect(looksLikeArchive(Buffer.alloc(0))).toBe(false);
    expect(looksLikeArchive(null)).toBe(false);
    expect(looksLikeArchive('PK')).toBe(false); // not a Buffer
  });
});
