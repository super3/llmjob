'use strict';

const os = require('os');
const { defaultWorker } = require('../src/shared/worker');
const { DEFAULTS } = require('../src/shared/config');

// The board keys a rig on (address, worker). A shared constant like "rig01"
// makes two machines on one payout address the SAME rig to the server — they
// overwrite each other's row, and if either is multi-GPU the other's row is
// dropped from the board entirely. Deriving from the hostname keeps them apart
// by default, which matters because nothing in either shell prompts for a name.
describe('defaultWorker', () => {
  test('uses this machine\'s hostname when none is passed', () => {
    const expected = String(os.hostname() || '').trim().toLowerCase().split('.')[0]
      .replace(/[^a-z0-9-]/g, '').replace(/^-+|-+$/g, '').slice(0, 32) || DEFAULTS.worker;
    expect(defaultWorker()).toBe(expected);
  });

  test('takes the first DNS label and lowercases it', () => {
    expect(defaultWorker('My-Rig.lan.example.com')).toBe('my-rig');
    expect(defaultWorker('BOX02')).toBe('box02');
  });

  test('strips characters a stratum worker token cannot carry', () => {
    expect(defaultWorker('  box_02!  ')).toBe('box02');
    expect(defaultWorker('rig #3')).toBe('rig3');
  });

  test('trims leading and trailing dashes', () => {
    expect(defaultWorker('--edge--')).toBe('edge');
  });

  test('caps the length', () => {
    expect(defaultWorker('a'.repeat(64))).toBe('a'.repeat(32));
  });

  test('falls back to the shared default only when there is no hostname at all', () => {
    // Nothing to tell two machines apart with, so the shared constant is honest.
    expect(defaultWorker('')).toBe(DEFAULTS.worker);
    expect(defaultWorker('   ')).toBe(DEFAULTS.worker);
    expect(defaultWorker(null)).toBe(defaultWorker()); // null → read the real hostname
  });

  // Stripping to ASCII erases a non-Latin hostname entirely. Returning the shared
  // constant there would put every such machine back on one name and recreate the
  // collision this function exists to prevent — for the users least likely to
  // notice, since the app never shows them the worker name it chose.
  test('derives a distinct token when sanitising leaves nothing usable', () => {
    const a = defaultWorker('Домашний-ПК');
    const b = defaultWorker('Рабочий-ПК');
    expect(a).toMatch(/^rig-[0-9a-f]{6}$/);
    expect(b).toMatch(/^rig-[0-9a-f]{6}$/);
    expect(a).not.toBe(b);
    expect(a).not.toBe(DEFAULTS.worker);
  });

  test('that token is stable for the same machine across runs', () => {
    expect(defaultWorker('Рабочий-ПК')).toBe(defaultWorker('Рабочий-ПК'));
  });

  test('distinguishes hostnames that sanitise down to the same single character', () => {
    expect(defaultWorker('工作站-1')).not.toBe(defaultWorker('服务器-1'));
  });

  test('a hostname that already sanitises cleanly is unchanged (nobody is renamed)', () => {
    expect(defaultWorker('DESKTOP-A1B2C3')).toBe('desktop-a1b2c3');
    expect(defaultWorker('rig9')).toBe('rig9');
  });

  test('two different hostnames give two different workers', () => {
    expect(defaultWorker('alpha')).not.toBe(defaultWorker('beta'));
  });
});
