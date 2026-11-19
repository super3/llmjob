'use strict';

const { amdWait } = require('../src/shared/amdWait');

// Issue #308: six RX 6000 cards on a 2-core G4400 held the load average near 7,
// because each card's thread busy-waited. The AMD core sleeps instead only where
// CPUs are few for the cards; elsewhere it keeps the spin it has always had.
describe('amdWait', () => {
  test('sleeps on a rig with fewer CPUs than cards plus one, and says why', () => {
    const w = amdWait({ cards: 6, cpus: 2, env: {} });
    expect(w.mode).toBe('block');
    expect(w.line).toBe('gpu wait:   block: 2 CPUs for 6 cards, so each card\'s thread sleeps between batches '
      + 'instead of taking a CPU core (--gpu-wait spin to change it)');
    expect(amdWait({ cards: 1, cpus: 1, env: {} }).line).toContain('1 CPU for 1 card,');
    expect(amdWait({ cards: 4, cpus: 4 }).mode).toBe('block');
  });

  test('leaves the spin where there are CPUs to spare', () => {
    expect(amdWait({ cards: 6, cpus: 7, env: {} })).toBeNull();
    expect(amdWait({ cards: 1, cpus: 16, env: {} })).toBeNull();
  });

  test('leaves a wait the user chose, and rigs with no AMD card or no CPU count', () => {
    expect(amdWait({ cards: 6, cpus: 2, env: { PEARL_WAIT: 'spin' } })).toBeNull();
    expect(amdWait({ cards: 0, cpus: 2, env: {} })).toBeNull();
    expect(amdWait({ cards: 6, cpus: 0, env: {} })).toBeNull();
    expect(amdWait({ cards: 6, cpus: null, env: {} })).toBeNull();
  });
});
