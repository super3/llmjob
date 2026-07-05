'use strict';

const { formatUpdate, clampPercent } = require('../src/shared/updateStatus');

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

  test('latest is a transient "up to date" message, with the version when known', () => {
    expect(formatUpdate('latest', { version: '0.1.8' })).toEqual({
      phase: 'latest', text: 'You’re on the latest version (v0.1.8)', show: true, transient: true,
    });
    expect(formatUpdate('latest')).toEqual({
      phase: 'latest', text: 'You’re on the latest version', show: true, transient: true,
    });
  });

  test('dev explains auto-update only runs installed', () => {
    expect(formatUpdate('dev')).toEqual({
      phase: 'dev', text: 'Auto-update works in the installed app.', show: true, transient: true,
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
