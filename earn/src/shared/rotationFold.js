'use strict';

// The switch for the rotation fold, a second mining kernel for RTX 30, 40 and 50 cards
// (shared/foldCheck has what it is and how the miner checks it). On by default
// since 13 November: on PR #293's Salad table it beat the old kernel on every RTX
// 30, 40 and 50 PC, by 2.7% to 9.7% (earn/tuning/benchmark.md). Turned off by
// --no-rotation-fold on the CLI, LLMJOB_ROTATION_FOLD=0 in the desktop app's
// environment (readRotationFoldEnv, read by main.js on every start), or
// PEARL_FOLD_ROT=0 set by the rig itself. Both shells hand the answer to the CUDA
// core as PEARL_FOLD_ROT (1 or 0), the variable it reads when it opens a card
// (native/src/pearl_host_rot.inc).
//
// On by default does not mean every card runs it. The core runs it only on
// compute 8.6, 8.9 and 12.0 cards whose cubin is there and has the memory, and
// the miner checks it on the CPU before trusting it; any other card, and one
// whose check fails, mines on the normal kernel. The AMD and Intel cores don't
// read the variable at all.
//
// Pure: the shells pass their environment in.

// The switches, by the name each shell's log lines call them, and the
// core's own variable.
const CLI_ROTATION_FOLD_FLAG = '--rotation-fold';
const CLI_NO_ROTATION_FOLD_FLAG = '--no-rotation-fold';
const GUI_ROTATION_FOLD_ENV = 'LLMJOB_ROTATION_FOLD';
const CORE_ROTATION_FOLD_ENV = 'PEARL_FOLD_ROT';

// Whether the fold is on, from a shell's request: true (asked for), false
// (turned off) or null (nothing said). Nothing said is on, unless the rig set
// PEARL_FOLD_ROT itself, which is then read as the core reads it (only 1 is on).
function resolveRotationFold(request, env) {
  if (request === true || request === false) return request;
  const own = env ? env[CORE_ROTATION_FOLD_ENV] : undefined;
  if (own != null) return own === '1';
  return true;
}

// The desktop app's switch, from its environment. Returns { on, warning }:
// `on` is false only for 0 (or the rig's own PEARL_FOLD_ROT=0), and `warning`
// is one line for the log when the variable is set to anything but 1 or 0. A
// typo keeps the default.
function readRotationFoldEnv(env) {
  const raw = env ? env[GUI_ROTATION_FOLD_ENV] : undefined;
  if (raw == null) return { on: resolveRotationFold(null, env), warning: null };
  const text = String(raw).trim();
  if (text === '1') return { on: true, warning: null };
  if (text === '0') return { on: false, warning: null };
  const on = resolveRotationFold(null, env);
  return {
    on,
    warning: GUI_ROTATION_FOLD_ENV + '=' + raw + ' ignored (must be 1 or 0); the rotation fold stays '
      + (on ? 'on' : 'off'),
  };
}

module.exports = {
  CLI_ROTATION_FOLD_FLAG, CLI_NO_ROTATION_FOLD_FLAG, GUI_ROTATION_FOLD_ENV, CORE_ROTATION_FOLD_ENV,
  resolveRotationFold, readRotationFoldEnv,
};
