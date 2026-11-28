'use strict';

// The CUDA core's patched kernel, checked in a process of its own before the
// miner opens the card.
//
// The core checks a patched kernel (a hand-written or patched cubin) against
// the compiled one before it uses it (earn/native/quantus/cuda/qc_cubin.inc).
// If the cubin faults in that check, the card's CUDA context is dead for the
// rest of the process: on Windows not even cudaDeviceReset or a second
// context brings it back. A core opened after that fails its first scan, and
// the engine stops the card. So the check runs first in a child process
// (`cubinCheck` in the addon), and the core is opened with the cubin only when
// the child says it passed. A child that faults just ends; the card then mines
// on the compiled kernel.
//
// The child is this program again: node or Electron (as node) with -e, or, in
// the CLI's single-file binary, which takes no -e, its own entry
// (src/cli/sea-entry.js), which calls runChild when CHECK_FILE is set.

const { createRequire } = require('module');

const CHECK_FILE = 'QUANTUS_CUBIN_CHECK_FILE';
const CHECK_CARDS = 'QUANTUS_CUBIN_CHECK_CARDS';
const CHILD_MS = 120000;

function seaBinary() {
  try {
    return require('node:sea').isSea();
  } catch (e) {
    return false;
  }
}

// The child: each card's check, one JSON line each, in order. It stops after a
// card whose check killed the process's CUDA context (lost): a card after it
// could be poisoned too, so the parent checks the rest in a new child.
function runChild(proc, req = createRequire(proc.execPath)) {
  const addon = req(proc.env[CHECK_FILE]);
  const cards = String(proc.env[CHECK_CARDS] || '').split(',').filter((s) => s !== '').map(Number);
  for (const device of cards) {
    const r = addon.cubinCheck(device);
    proc.stdout.write(JSON.stringify(r) + '\n');
    if (r && r.lost) break;
  }
}

function parseLines(text) {
  const out = [];
  for (const line of String(text || '').split('\n')) {
    try {
      const r = JSON.parse(line);
      if (r && typeof r === 'object' && Number.isInteger(r.device)) out.push(r);
    } catch (e) { /* not a result line */ }
  }
  return out;
}

// What createCore should pass for each card of `addon` (loaded from `file`):
// { cubin: true|false } once the child has checked the card, {} when there is
// nothing to check (an addon without cubinCheck, or a card with no patched
// kernel). Results are kept for the life of the process.
function cubinGate(addon, file, opts) {
  const log = opts.log || (() => {});
  const spawnSync = opts.spawnSync || require('child_process').spawnSync;
  const execPath = opts.execPath || process.execPath;
  const isSea = opts.isSea || seaBinary;
  const env = opts.env || process.env;
  const known = new Map();
  const able = Boolean(file) && typeof addon.cubinWanted === 'function' && typeof addon.cubinCheck === 'function';

  const wanted = (index) => {
    try {
      return Boolean(addon.cubinWanted(index));
    } catch (e) {
      return false;
    }
  };

  const settle = (device, ok, why) => {
    known.set(device, ok);
    if (ok) log('info', 'Quantus core: card ' + device + ': the patched kernel passed its check in a separate process');
    else log('warn', 'Quantus core: card ' + device + ': the patched kernel is not used (' + why + '); the compiled kernel runs');
  };

  // One child for the cards; a new one for those left after a card that killed
  // the child's context, or after a child that ended without an answer.
  const runChecks = (cards) => {
    let left = cards;
    while (left.length) {
      const childEnv = Object.assign({}, env, { ELECTRON_RUN_AS_NODE: '1', [CHECK_FILE]: file, [CHECK_CARDS]: left.join(',') });
      const args = isSea() ? [] : ['-e', 'require(' + JSON.stringify(__filename) + ').runChild(process)'];
      const r = spawnSync(execPath, args, {
        env: childEnv, encoding: 'utf8', timeout: CHILD_MS, windowsHide: true, stdio: ['ignore', 'pipe', 'inherit'],
      });
      const got = parseLines(r.stdout).filter((x) => left.includes(x.device) && !known.has(x.device));
      for (const x of got) settle(x.device, x.ok === true, x.why || (x.lost ? 'it faulted' : 'its check failed'));
      const last = got[got.length - 1];
      const rest = left.filter((i) => !known.has(i));
      if (rest.length && !(last && last.lost)) {
        // The child ended without answering for rest[0]: it died on that card.
        const how = r.error ? r.error.message : r.signal ? 'signal ' + r.signal : 'exit code ' + r.status;
        settle(rest[0], false, 'the check process ended (' + how + ')');
        rest.shift();
      }
      left = rest;
    }
  };

  return {
    prepare(indices) {
      if (!able) return;
      const cards = (Array.isArray(indices) ? indices : []).filter((i) => !known.has(i) && wanted(i));
      if (cards.length) runChecks(cards);
    },
    options(index) {
      if (!able) return {};
      if (!known.has(index)) {
        if (!wanted(index)) return {};
        runChecks([index]);
      }
      return { cubin: known.get(index) };
    },
  };
}

module.exports = { CHECK_FILE, CHECK_CARDS, CHILD_MS, cubinGate, runChild, parseLines, seaBinary };
