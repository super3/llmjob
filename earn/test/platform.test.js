'use strict';

const {
  NO_MINER, minerSupported, minerUnsupportedNote, autoUpdateSupported,
} = require('../src/shared/platform');

describe('minerSupported', () => {
  test('the two platforms AlphaPool builds alpha-miner for', () => {
    expect(minerSupported('win32')).toBe(true);
    expect(minerSupported('linux')).toBe(true);
  });

  test('macOS has no engine at any version', () => {
    expect(minerSupported('darwin')).toBe(false);
    expect(NO_MINER).toEqual(['darwin']);
  });

  // Everything downstream treats "not win32" as Linux (engine.enginePackage,
  // engineBinaryName, engineDownloadUrl), so an unknown platform resolving to a
  // Linux binary is the existing behaviour and is deliberately left alone —
  // only the platforms we actually ship for are gated.
  test('an unlisted platform is left as-is rather than guessed at', () => {
    expect(minerSupported('freebsd')).toBe(true);
    expect(minerSupported(undefined)).toBe(true);
  });
});

describe('minerUnsupportedNote', () => {
  test('says nothing where mining works, whatever the mode', () => {
    for (const mode of ['auto', 'mining', 'both', 'llm']) {
      expect(minerUnsupportedNote('win32', mode)).toBe('');
      expect(minerUnsupportedNote('linux', mode)).toBe('');
    }
  });

  test('says nothing on macOS in LLM mode — that mode never asked to mine', () => {
    expect(minerUnsupportedNote('darwin', 'llm')).toBe('');
  });

  test('explains the gap on macOS for the co-running modes, and what still runs', () => {
    for (const mode of ['auto', 'both']) {
      const note = minerUnsupportedNote('darwin', mode);
      expect(note).toMatch(/mining is not available on macOS/);
      expect(note).toMatch(/Windows and Linux only/);
      // The reassuring half matters as much as the refusal: an 'auto' start on a
      // Mac does bring the model up, and a note that only said "no mining" would
      // read as "nothing happened".
      expect(note).toMatch(/local LLM runs as usual/);
    }
  });

  test('mining-only mode is told where to go instead, since nothing will run', () => {
    const note = minerUnsupportedNote('darwin', 'mining');
    expect(note).toMatch(/mining is not available on macOS/);
    expect(note).toMatch(/Switch the compute mode to LLM/);
    expect(note).not.toMatch(/runs as usual/);
  });

  test('an unknown mode is treated as one that wanted to mine', () => {
    expect(minerUnsupportedNote('darwin', undefined)).toMatch(/mining is not available/);
  });
});

describe('autoUpdateSupported', () => {
  test('on everywhere electron-updater can actually install', () => {
    expect(autoUpdateSupported('win32')).toBe(true);
    expect(autoUpdateSupported('linux')).toBe(true);
  });

  // Squirrel.Mac requires the update's signature to match the running app's, and
  // the Mac build carries only an ad-hoc one — so a check there can only ever
  // end in a failure the user cannot act on.
  test('off on macOS, where the build is only ad-hoc signed', () => {
    expect(autoUpdateSupported('darwin')).toBe(false);
  });
});
