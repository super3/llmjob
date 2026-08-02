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

  test('falls back to the shared default when the hostname is unusable', () => {
    // Both an unset hostname and one that sanitises to nothing must still yield a
    // valid worker name — an empty one would be rejected by the pool.
    expect(defaultWorker('')).toBe(DEFAULTS.worker);
    expect(defaultWorker('!!!')).toBe(DEFAULTS.worker);
    expect(defaultWorker('---')).toBe(DEFAULTS.worker);
    expect(defaultWorker(null)).toBe(defaultWorker()); // null → read the real hostname
  });

  test('two different hostnames give two different workers', () => {
    expect(defaultWorker('alpha')).not.toBe(defaultWorker('beta'));
  });
});
