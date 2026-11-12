'use strict';

// The switch for the rotation fold, a second mining kernel for RTX 30 and 40 cards
// (shared/foldCheck has what it is and how the miner checks it). Off unless
// asked for: --rotation-fold on the CLI, LLMJOB_ROTATION_FOLD=1 in the desktop
// app's environment (readRotationFoldEnv, read by main.js on every start).
// Both shells turn the request into PEARL_FOLD_ROT=1, the variable the CUDA
// core reads when it opens a card (native/src/pearl_host_rot.inc); a rig that
// sets that variable itself gets the same.
//
// Pure: the shells pass their environment in.

// The two switches, by the name each shell's log lines call them, and the
// core's own variable.
const CLI_ROTATION_FOLD_FLAG = '--rotation-fold';
const GUI_ROTATION_FOLD_ENV = 'LLMJOB_ROTATION_FOLD';
const CORE_ROTATION_FOLD_ENV = 'PEARL_FOLD_ROT';

// The desktop app's switch, from its environment. Returns { on, warning }:
// `on` is true only for 1, and `warning` is one line for the log when the
// variable is set to anything but 1 or 0. A typo leaves the fold off.
function readRotationFoldEnv(env) {
  const raw = env ? env[GUI_ROTATION_FOLD_ENV] : undefined;
  if (raw == null) return { on: false, warning: null };
  const text = String(raw).trim();
  if (text === '1') return { on: true, warning: null };
  if (text === '0') return { on: false, warning: null };
  return {
    on: false,
    warning: GUI_ROTATION_FOLD_ENV + '=' + raw + ' ignored (must be 1 or 0); the rotation fold stays off',
  };
}

module.exports = {
  CLI_ROTATION_FOLD_FLAG, GUI_ROTATION_FOLD_ENV, CORE_ROTATION_FOLD_ENV, readRotationFoldEnv,
};
