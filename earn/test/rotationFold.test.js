'use strict';

// The rotation fold's switch (shared/rotationFold): --rotation-fold on the CLI,
// LLMJOB_ROTATION_FOLD=1 in the desktop app's environment. Both become
// PEARL_FOLD_ROT=1 for the CUDA core.

const {
  CLI_ROTATION_FOLD_FLAG, GUI_ROTATION_FOLD_ENV, CORE_ROTATION_FOLD_ENV, readRotationFoldEnv,
} = require('../src/shared/rotationFold');
const { parseCliArgs, USAGE } = require('../src/shared/cliArgs');

describe('the rotation fold\'s switch', () => {
  test('the names: one flag, one variable for the app, one for the core', () => {
    expect([CLI_ROTATION_FOLD_FLAG, GUI_ROTATION_FOLD_ENV, CORE_ROTATION_FOLD_ENV])
      .toEqual(['--rotation-fold', 'LLMJOB_ROTATION_FOLD', 'PEARL_FOLD_ROT']);
    // The flag is the one the CLI takes and documents.
    expect(parseCliArgs(['--mode', 'llm', CLI_ROTATION_FOLD_FLAG]).settings.rotationFold).toBe(true);
    expect(USAGE).toContain('      ' + CLI_ROTATION_FOLD_FLAG + ' ');
  });

  test('the app\'s variable is off unless it says 1', () => {
    expect(readRotationFoldEnv({})).toEqual({ on: false, warning: null });
    expect(readRotationFoldEnv(null)).toEqual({ on: false, warning: null });
    expect(readRotationFoldEnv({ LLMJOB_ROTATION_FOLD: '1' })).toEqual({ on: true, warning: null });
    expect(readRotationFoldEnv({ LLMJOB_ROTATION_FOLD: ' 1 ' })).toEqual({ on: true, warning: null });
    expect(readRotationFoldEnv({ LLMJOB_ROTATION_FOLD: '0' })).toEqual({ on: false, warning: null });
  });

  // A typo must not turn on a kernel nobody asked for, and must not pass unseen.
  test('anything else leaves it off and says so', () => {
    for (const raw of ['true', 'yes', 'auto', '', '2']) {
      expect(readRotationFoldEnv({ LLMJOB_ROTATION_FOLD: raw })).toEqual({
        on: false,
        warning: 'LLMJOB_ROTATION_FOLD=' + raw + ' ignored (must be 1 or 0); the rotation fold stays off',
      });
    }
  });
});
