'use strict';

// Pure logic for the CLI under HiveOS: the HiveOS package updating itself, the
// trial a new version runs on, and the messages it posts to the HiveOS
// dashboard. The filesystem, download and process side is src/cli/hiveos.js.
//
// The package lives in /hive/miners/custom/llmjob-earn. HiveOS installs it once
// from the flight sheet's URL and never again while that URL stays the same, so
// the CLI updates it: it unpacks the release's package beside the installed one,
// swaps the two folders, and exits. HiveOS's miner loop then starts the new
// h-run.sh three seconds later. The old folder stays as llmjob-earn.prev until
// the new version has hashed. If it exits with an error before that, the new
// h-run.sh puts the old folder back and the old version skips the new one for a
// day.

// Written into the new package folder by the swap; removed when the new version
// has proven itself. h-run.sh reads it with sed, so it is KEY=VALUE lines.
const TRIAL_FILE = '.update-trial';
// Written by h-run.sh into the restored folder after a rollback, to hold back
// the version it rolled back from for a day.
const SKIP_FILE = '.update-skip';
// The old package folder, kept until the trial passes.
const PREV_SUFFIX = '.prev';

// A trial passes when the miner reports a hashrate. A run that mines nothing on
// purpose (every GPU off, or --mode llm) passes once it has run this long.
const TRIAL_IDLE_PASS_MS = 2 * 60 * 1000;
// And fails when a run that should mine has not hashed in this long.
const TRIAL_FAIL_MS = 10 * 60 * 1000;
// How often the CLI asks.
const TRIAL_CHECK_MS = 15 * 1000;
// The exit code for a failed trial, which h-run.sh rolls back on like any other
// error. 75 is EX_TEMPFAIL.
const TRIAL_FAILED_EXIT = 75;

// The same text is posted to the dashboard at most this often. HiveOS restarts a
// miner that exits every few seconds, and each start would post again.
const MESSAGE_REPEAT_MS = 60 * 60 * 1000;

// NVIDIA PCI device ids from 0x2900 up are Blackwell (B200 0x2901, RTX 5090
// 0x2b85, RTX 5080 0x2c02, RTX 5060 Ti 0x2d04, RTX PRO 6000 0x2bb1). Ada ends
// below 0x2900. Blackwell needs driver 570 or newer, and HiveOS's stable image
// ships 550, which does not see the card at all, so nvidia-smi can't be asked.
const BLACKWELL_FIRST_DEVICE_ID = 0x2900;
const MIN_DRIVER_BLACKWELL = 570;

// KEY=VALUE lines, as h-manifest.conf and the trial and skip files hold them.
// Blank lines and # comments are skipped; quotes around a value are dropped.
function parseKeyValues(text) {
  const out = {};
  for (const raw of String(text == null ? '' : text).split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith('#')) continue;
    const eq = line.indexOf('=');
    if (eq <= 0) continue;
    out[line.slice(0, eq).trim()] = line.slice(eq + 1).trim().replace(/^(["'])(.*)\1$/, '$2');
  }
  return out;
}

// The trial file's text.
function formatTrial({ from, to, atMs }) {
  return 'UPDATE_FROM=' + from + '\nUPDATE_TO=' + to + '\nUPDATE_AT=' + Math.floor(atMs / 1000) + '\n';
}

// The trial file read back, as { from, to } or null when it says nothing usable.
function parseTrial(text) {
  const kv = parseKeyValues(text);
  if (!kv.UPDATE_TO) return null;
  return { from: kv.UPDATE_FROM || null, to: kv.UPDATE_TO };
}

// Whether the skip file (written by h-run.sh as SKIP_VERSION and SKIP_UNTIL, in
// epoch seconds) holds back `version` at `nowMs`.
function isSkipped(skipText, version, nowMs) {
  const kv = parseKeyValues(skipText);
  const until = parseInt(kv.SKIP_UNTIL, 10);
  return !!kv.SKIP_VERSION && kv.SKIP_VERSION === version
    && Number.isFinite(until) && nowMs < until * 1000;
}

// The release asset that holds a version's HiveOS package.
function packageAssetName(version) {
  return 'llmjob-earn-' + version + '.tar.gz';
}

// What a trial should do now: 'pass', 'fail' or 'wait'. `hashing` is whether
// the miner has reported a hashrate; `mines` whether this run should.
function trialVerdict({ elapsedMs, hashing, mines }) {
  if (hashing) return 'pass';
  if (!mines) return elapsedMs >= TRIAL_IDLE_PASS_MS ? 'pass' : 'wait';
  return elapsedMs >= TRIAL_FAIL_MS ? 'fail' : 'wait';
}

// Whether to post a text now, given when it was last posted (ms, or null).
function shouldPost(lastAtMs, nowMs) {
  return !(Number.isFinite(lastAtMs) && nowMs - lastAtMs < MESSAGE_REPEAT_MS);
}

// The record of recent posts ({ text: atMs }) with `text` added at `nowMs` and
// anything old enough to post again dropped, so the file never grows.
function recordPost(posted, text, nowMs) {
  const out = {};
  for (const [t, at] of Object.entries(posted || {})) {
    if (!shouldPost(at, nowMs)) out[t] = at;
  }
  out[text] = nowMs;
  return out;
}

// The dashboard message for argument errors, which stop the CLI before it does
// anything. An address error points at the wallet template, anything else at
// Extra config, which is where every other flag comes from.
function startupMessage(errors) {
  const list = errors;
  const addr = list.find((e) => /^invalid Pearl address: |^--address is required/.test(e));
  if (addr) {
    const bad = /^invalid Pearl address: (.*)$/.exec(addr);
    return 'LLMJob Earn can\'t start: ' + (bad ? bad[1] + ' is not a Pearl address' : 'there is no wallet address')
      + '. Put your prl1p… address in the flight sheet\'s wallet template.';
  }
  return 'LLMJob Earn can\'t start: ' + list.join('; ') + '. Check Extra config arguments in the flight sheet.';
}

// The dashboard message for a core that is there but would not load. A glibc
// or libstdc++ version it lacks means the HiveOS image is too old; anything
// else is passed on as the loader said it.
function coreLoadMessage(line) {
  const text = String(line == null ? '' : line);
  const ver = /version [`'](GLIBC|GLIBCXX|CXXABI)_([0-9.]+)' not found/.exec(text);
  if (ver) {
    return 'LLMJob Earn can\'t mine: this HiveOS image is too old (the mining core needs '
      + ver[1] + '_' + ver[2] + '). Update the rig to the latest HiveOS image.';
  }
  return 'LLMJob Earn can\'t mine: ' + (text.length > 300 ? text.slice(0, 297) + '…' : text);
}

// The NVIDIA cards in `lspci -n` output that are Blackwell or newer, as their
// slots. Display (0300) and 3D (0302) controllers only.
function blackwellSlots(lspciText) {
  const slots = [];
  for (const line of String(lspciText == null ? '' : lspciText).split(/\r?\n/)) {
    const m = /^(\S+)\s+030[02]:\s+10de:([0-9a-f]{4})\b/i.exec(line.trim());
    if (m && parseInt(m[2], 16) >= BLACKWELL_FIRST_DEVICE_ID) slots.push(m[1]);
  }
  return slots;
}

// The driver's major version from /proc/driver/nvidia/version ("NVRM version:
// NVIDIA UNIX x86_64 Kernel Module  550.127.05  ..."), or null.
function driverMajor(procText) {
  const m = /Kernel Module(?:\s+for\s+\S+)?\s+(\d+)\.\d+/.exec(String(procText == null ? '' : procText));
  return m ? parseInt(m[1], 10) : null;
}

// The dashboard message for Blackwell cards on a driver too old for them, or
// null when there is nothing to say. With no driver loaded at all, the message
// still names the version needed.
function driverMessage(lspciText, procText) {
  const slots = blackwellSlots(lspciText);
  if (!slots.length) return null;
  const major = driverMajor(procText);
  if (major != null && major >= MIN_DRIVER_BLACKWELL) return null;
  return 'LLMJob Earn: ' + (slots.length === 1 ? 'the RTX 50 or other Blackwell card at ' + slots[0] + ' needs'
    : slots.length + ' RTX 50 or other Blackwell cards need') + ' NVIDIA driver ' + MIN_DRIVER_BLACKWELL
    + ' or newer, and this rig has ' + (major == null ? 'none loaded' : major)
    + '. Run nvidia-driver-update in Hive Shell.';
}

module.exports = {
  TRIAL_FILE,
  SKIP_FILE,
  PREV_SUFFIX,
  TRIAL_IDLE_PASS_MS,
  TRIAL_FAIL_MS,
  TRIAL_CHECK_MS,
  TRIAL_FAILED_EXIT,
  MESSAGE_REPEAT_MS,
  MIN_DRIVER_BLACKWELL,
  parseKeyValues,
  formatTrial,
  parseTrial,
  isSkipped,
  packageAssetName,
  trialVerdict,
  shouldPost,
  recordPost,
  startupMessage,
  coreLoadMessage,
  blackwellSlots,
  driverMajor,
  driverMessage,
};
