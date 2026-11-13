'use strict';

// The rotation fold's switch (shared/rotationFold): on by default; --no-rotation-fold
// on the CLI, LLMJOB_ROTATION_FOLD=0 in the desktop app's environment, or the rig's
// own PEARL_FOLD_ROT=0 turn it off. Both shells hand PEARL_FOLD_ROT (1 or 0) to the
// CUDA core.

const {
  CLI_ROTATION_FOLD_FLAG, CLI_NO_ROTATION_FOLD_FLAG, GUI_ROTATION_FOLD_ENV, CORE_ROTATION_FOLD_ENV,
  resolveRotationFold, readRotationFoldEnv,
} = require('../src/shared/rotationFold');
const { parseCliArgs, USAGE } = require('../src/shared/cliArgs');

describe('the rotation fold\'s switch', () => {
  test('the names: one flag, one variable for the app, one for the core', () => {
    expect([CLI_ROTATION_FOLD_FLAG, CLI_NO_ROTATION_FOLD_FLAG, GUI_ROTATION_FOLD_ENV, CORE_ROTATION_FOLD_ENV])
      .toEqual(['--rotation-fold', '--no-rotation-fold', 'LLMJOB_ROTATION_FOLD', 'PEARL_FOLD_ROT']);
    // The flags are the ones the CLI takes; the usage text documents the opt-out.
    expect(parseCliArgs(['--mode', 'llm', CLI_ROTATION_FOLD_FLAG]).settings.rotationFold).toBe(true);
    expect(parseCliArgs(['--mode', 'llm', CLI_NO_ROTATION_FOLD_FLAG]).settings.rotationFold).toBe(false);
    expect(USAGE).toContain('      ' + CLI_NO_ROTATION_FOLD_FLAG + ' ');
    expect(USAGE).toContain(CLI_ROTATION_FOLD_FLAG + ' turns it on');
  });

  test('nothing said is on, unless the rig set PEARL_FOLD_ROT itself', () => {
    expect([resolveRotationFold(true, {}), resolveRotationFold(false, {}), resolveRotationFold(null, {})])
      .toEqual([true, false, true]);
    expect(resolveRotationFold(null, null)).toBe(true);
    expect(resolveRotationFold(null, { PEARL_FOLD_ROT: '0' })).toBe(false);
    expect(resolveRotationFold(null, { PEARL_FOLD_ROT: '1' })).toBe(true);
    // Read as the core reads it: only 1 is on.
    expect(resolveRotationFold(null, { PEARL_FOLD_ROT: 'yes' })).toBe(false);
    // A flag wins over the rig's variable.
    expect(resolveRotationFold(true, { PEARL_FOLD_ROT: '0' })).toBe(true);
    expect(resolveRotationFold(false, { PEARL_FOLD_ROT: '1' })).toBe(false);
  });

  test('the app\'s variable is on unless it says 0', () => {
    expect(readRotationFoldEnv({})).toEqual({ on: true, warning: null });
    expect(readRotationFoldEnv(null)).toEqual({ on: true, warning: null });
    expect(readRotationFoldEnv({ LLMJOB_ROTATION_FOLD: '1' })).toEqual({ on: true, warning: null });
    expect(readRotationFoldEnv({ LLMJOB_ROTATION_FOLD: '0' })).toEqual({ on: false, warning: null });
    expect(readRotationFoldEnv({ LLMJOB_ROTATION_FOLD: ' 0 ' })).toEqual({ on: false, warning: null });
    expect(readRotationFoldEnv({ PEARL_FOLD_ROT: '0' })).toEqual({ on: false, warning: null });
    expect(readRotationFoldEnv({ LLMJOB_ROTATION_FOLD: '1', PEARL_FOLD_ROT: '0' })).toEqual({ on: true, warning: null });
  });

  // A typo must not change what the rig mines, and must not pass unseen.
  test('anything else keeps the default and says so', () => {
    for (const raw of ['true', 'yes', 'auto', '', '2']) {
      expect(readRotationFoldEnv({ LLMJOB_ROTATION_FOLD: raw })).toEqual({
        on: true,
        warning: 'LLMJOB_ROTATION_FOLD=' + raw + ' ignored (must be 1 or 0); the rotation fold stays on',
      });
    }
    expect(readRotationFoldEnv({ LLMJOB_ROTATION_FOLD: 'off', PEARL_FOLD_ROT: '0' })).toEqual({
      on: false,
      warning: 'LLMJOB_ROTATION_FOLD=off ignored (must be 1 or 0); the rotation fold stays off',
    });
  });
});
