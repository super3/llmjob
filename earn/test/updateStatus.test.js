'use strict';

const { formatUpdate, clampPercent, describeUpdateError } = require('../src/shared/updateStatus');

describe('describeUpdateError', () => {
  // Verbatim from a user's log: electron-updater glues the whole response into
  // err.message — the 503's HTML body and every header, Set-Cookie included.
  const REAL_503 = '503 "method: GET url: https://github.com/super3/llmjob/releases.atom\n\n'
    + ' Data:\n <html><body><h1>503 Service Unavailable</h1>\nNo server is available to handle'
    + ' this request.\n</body></html>\n\n " Headers: { "cache-control": "no-cache",'
    + ' "set-cookie": [ "_gh_sess=fOk7Qwq0dNT%2BI2Yo0gKJ23Ck8j; path=/; HttpOnly" ] }';

  test('reduces a full HTTP response to the status and URL', () => {
    expect(describeUpdateError(new Error(REAL_503)))
      .toBe('HTTP 503 from https://github.com/super3/llmjob/releases.atom');
  });

  test('and never carries the body or the cookies through', () => {
    const out = describeUpdateError(new Error(REAL_503));
    expect(out).not.toMatch(/_gh_sess|<html>|set-cookie/i);
    expect(out.length).toBeLessThan(120);
  });

  test('a status with no URL still reads cleanly', () => {
    expect(describeUpdateError(new Error('429 rate limited'))).toBe('HTTP 429');
  });

  test('short messages pass through — including a bare string', () => {
    expect(describeUpdateError(new Error('net::ERR_HTTP2_SERVER_REFUSED_STREAM')))
      .toBe('net::ERR_HTTP2_SERVER_REFUSED_STREAM');
    expect(describeUpdateError('plain string failure')).toBe('plain string failure');
  });

  test('a long single-line message is capped', () => {
    const out = describeUpdateError(new Error('x'.repeat(500)));
    expect(out).toHaveLength(200);
    expect(out.endsWith('…')).toBe(true);
  });

  test('multi-line non-status messages keep only the first line', () => {
    expect(describeUpdateError(new Error('boom\n  at foo\n  at bar'))).toBe('boom');
  });

  test('nothing useful still says something', () => {
    expect(describeUpdateError(null)).toBe('unknown error');
    expect(describeUpdateError(new Error('   '))).toBe('unknown error');
  });
});

describe('clampPercent', () => {
  test('rounds a normal value', () => {
    expect(clampPercent(33.6)).toBe(34);
  });
  test('floors below zero to 0', () => {
    expect(clampPercent(-5)).toBe(0);
  });
  test('caps above 100 to 100', () => {
    expect(clampPercent(150)).toBe(100);
  });
  test('treats a non-number as 0', () => {
    expect(clampPercent('nope')).toBe(0);
  });
});

describe('formatUpdate', () => {
  test('checking', () => {
    expect(formatUpdate('checking')).toEqual({ phase: 'checking', text: 'Checking for updates…', show: true });
  });

  test('available with a version', () => {
    expect(formatUpdate('available', { version: '0.1.2' })).toEqual({
      phase: 'available', text: 'Update v0.1.2 available — downloading…', show: true,
    });
  });

  test('available without a version (empty payload)', () => {
    expect(formatUpdate('available', {})).toEqual({
      phase: 'available', text: 'Update available — downloading…', show: true,
    });
  });

  test('available with no payload at all', () => {
    expect(formatUpdate('available')).toEqual({
      phase: 'available', text: 'Update available — downloading…', show: true,
    });
  });

  test('progress with a percent', () => {
    expect(formatUpdate('progress', { percent: 42.4 })).toEqual({
      phase: 'progress', text: 'Downloading update… 42%', show: true,
    });
  });

  test('progress with no payload shows 0%', () => {
    expect(formatUpdate('progress')).toEqual({
      phase: 'progress', text: 'Downloading update… 0%', show: true,
    });
  });

  test('ready with a version reveals the restart affordance', () => {
    expect(formatUpdate('ready', { version: '0.1.2' })).toEqual({
      phase: 'ready', text: 'Update v0.1.2 ready', show: true, ready: true,
    });
  });

  test('ready without a version', () => {
    expect(formatUpdate('ready')).toEqual({
      phase: 'ready', text: 'Update ready', show: true, ready: true,
    });
  });

  test('none hides the bar', () => {
    expect(formatUpdate('none')).toEqual({ phase: 'none', text: '', show: false });
  });

  test('latest is a transient "up to date" message', () => {
    expect(formatUpdate('latest')).toEqual({
      phase: 'latest', text: 'You’re on the latest version', show: true, transient: true,
    });
  });

  // macOS: the ad-hoc-signed build has no in-app update path, so "Check for
  // updates" opens the Releases page and the bar confirms it. Deliberately not
  // an `error` — nothing failed — and transient, like `latest`.
  test('manual explains the macOS path without flagging a fault', () => {
    expect(formatUpdate('manual')).toEqual({
      phase: 'manual',
      text: 'Updates are manual on macOS — opening the Releases page',
      show: true,
      transient: true,
    });
  });

  test('error flags a fault', () => {
    expect(formatUpdate('error')).toEqual({
      phase: 'error', text: 'Update check failed — see Logs.', show: true, error: true,
    });
  });

  test('unknown phase falls back to idle', () => {
    expect(formatUpdate('wat')).toEqual({ phase: 'idle', text: '', show: false });
  });
});
