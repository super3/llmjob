'use strict';

const { DEFAULT_PREFS, IDLE_AFTER_SEC, IDLE_POLL_MS, normalizePrefs, idleAction } = require('../src/shared/appPrefs');

describe('normalizePrefs', () => {
  test('a fresh install closes to the tray, and starts nothing on its own', () => {
    expect(normalizePrefs(undefined)).toEqual({
      closeToTray: true, runAtStartup: false, mineWhenIdle: false, trayHintShown: false,
    });
    expect(normalizePrefs('nonsense')).toEqual(DEFAULT_PREFS);
  });

  test('keeps saved booleans, and falls back on anything else', () => {
    expect(normalizePrefs({ closeToTray: false, runAtStartup: 'yes', mineWhenIdle: true, extra: 1 })).toEqual({
      closeToTray: false, runAtStartup: false, mineWhenIdle: true, trayHintShown: false,
    });
  });

  test('the defaults cannot be changed by accident', () => {
    expect(Object.isFrozen(DEFAULT_PREFS)).toBe(true);
  });
});

describe('idleAction', () => {
  test('idle means five minutes without input, checked every two seconds', () => {
    expect(IDLE_AFTER_SEC).toBe(300);
    expect(IDLE_POLL_MS).toBe(2000);
  });

  test('pauses when someone is at the computer, and resumes when they leave', () => {
    expect(idleAction('active', false)).toBe('pause');
    expect(idleAction('idle', true)).toBe('resume');
    expect(idleAction('locked', true)).toBe('resume');
  });

  test('does nothing when nothing changed', () => {
    expect(idleAction('active', true)).toBeNull();
    expect(idleAction('idle', false)).toBeNull();
  });

  // Some Linux desktops cannot report idle time. Mining there beats never
  // starting.
  test('a system that cannot tell counts as idle', () => {
    expect(idleAction('unknown', true)).toBe('resume');
    expect(idleAction('unknown', false)).toBeNull();
  });
});
