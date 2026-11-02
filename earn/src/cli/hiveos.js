'use strict';

// IO shell for the CLI under HiveOS -- the filesystem, download and process side
// of shared/hiveos.js, whose rules are tested there. h-run.sh sets LLMJOB_HIVEOS=1,
// and with it the CLI updates its whole package folder instead of only its own
// binary, runs a new version on trial, and posts problems to the HiveOS
// dashboard with HiveOS's own `message` command.

const fs = require('fs');
const path = require('path');
const { spawn, spawnSync, execFile } = require('child_process');
const { downloadFile } = require('../main/io');
const H = require('../shared/hiveos');

const ENV = 'LLMJOB_HIVEOS';
const MESSAGE_BIN = '/hive/bin/message';
// What was posted lately, so a miner HiveOS restarts every few seconds doesn't
// post on every start. /run is a tmpfs, so a reboot forgets it.
const MESSAGE_STATE = '/run/hive/llmjob-earn-messages.json';
const NVIDIA_VERSION = '/proc/driver/nvidia/version';

function isHiveos(env) {
  return (env || process.env)[ENV] === '1';
}

// The package folder: the one the running binary sits in.
function packageDir(execPath) {
  return path.dirname(execPath || process.execPath);
}

function readText(file) {
  try {
    return fs.readFileSync(file, 'utf8');
  } catch (e) {
    return '';
  }
}

// Post a message to this rig's HiveOS dashboard: type is success, info,
// warning or danger. Fire and forget, and never throws. False when it was not
// sent: not on HiveOS, or the same text went out within the hour.
function postMessage(type, text, opts) {
  const o = opts || {};
  const bin = o.messageBin || MESSAGE_BIN;
  const state = o.statePath || MESSAGE_STATE;
  const now = (o.now || Date.now)();
  if (!fs.existsSync(bin)) return false;
  let posted = {};
  try {
    posted = JSON.parse(fs.readFileSync(state, 'utf8')) || {};
  } catch (e) { /* nothing posted yet */ }
  if (!H.shouldPost(posted[text], now)) return false;
  try {
    fs.writeFileSync(state, JSON.stringify(H.recordPost(posted, text, now)));
  } catch (e) { /* posting matters more than remembering it */ }
  try {
    // Detached, so a CLI that is about to exit doesn't take the post with it.
    const child = spawn(bin, [type, text], { detached: true, stdio: 'ignore' });
    child.on('error', () => {});
    child.unref();
  } catch (e) {
    return false;
  }
  return true;
}

// The trial this package is on, as { from, to }, or null when it is on none.
function readTrial(dir) {
  return H.parseTrial(readText(path.join(dir, H.TRIAL_FILE)));
}

// Whether a rollback holds back `version` now.
function isSkipped(dir, version, nowMs) {
  return H.isSkipped(readText(path.join(dir, H.SKIP_FILE)), version, nowMs);
}

// The trial passed: the old folder can go.
function confirmTrial(dir) {
  fs.rmSync(path.join(dir, H.TRIAL_FILE), { force: true });
  fs.rmSync(dir + H.PREV_SUFFIX, { recursive: true, force: true });
}

// Where an update unpacks: beside the package, on the same filesystem, so the
// swap is two renames.
function stagingDir(dir, pid) {
  return path.join(path.dirname(dir), '.' + path.basename(dir) + '-update-' + pid);
}

// Remove what an update killed half way left behind.
function cleanStaging(dir) {
  const parent = path.dirname(dir);
  const prefix = '.' + path.basename(dir) + '-update-';
  let names = [];
  try {
    names = fs.readdirSync(parent);
  } catch (e) {
    return;
  }
  for (const n of names) {
    if (n.startsWith(prefix)) fs.rmSync(path.join(parent, n), { recursive: true, force: true });
  }
}

// h-config.sh writes the CLI's argument line inside the package folder (the
// manifest's CUSTOM_CONFIG_FILENAME). HiveOS writes it only when the miner
// starts, not between the loop's restarts, so the new folder needs a copy.
function carryConfig(dir, fresh) {
  const conf = H.parseKeyValues(readText(path.join(dir, 'h-manifest.conf'))).CUSTOM_CONFIG_FILENAME;
  if (!conf || !fs.existsSync(conf)) return;
  if (fs.realpathSync(path.dirname(conf)) === fs.realpathSync(dir)) {
    fs.copyFileSync(conf, path.join(fresh, path.basename(conf)));
  }
}

// Download and unpack `plan`'s package beside the installed one, check it is
// the version it says, and mark it as on trial. Returns the unpacked folder.
// Throws, with nothing left behind, when any step fails.
async function stagePackage(plan, dir, opts) {
  const o = opts || {};
  const stage = stagingDir(dir, o.pid || process.pid);
  const tarball = stage + '.tar.gz';
  fs.rmSync(stage, { recursive: true, force: true });
  fs.mkdirSync(stage, { recursive: true });
  try {
    await downloadFile(plan.packageUrl, tarball);
    const tar = spawnSync('tar', ['-xzf', tarball, '-C', stage], { encoding: 'utf8', timeout: 120000 });
    if (tar.error || tar.status !== 0) {
      throw new Error('could not unpack it (' + ((tar.error && tar.error.message)
        || String(tar.stderr || '').trim() || 'tar exited ' + tar.status) + ')');
    }
    // The package unpacks to a folder named like the installed one.
    const fresh = path.join(stage, path.basename(dir));
    const version = H.parseKeyValues(readText(path.join(fresh, 'h-manifest.conf'))).CUSTOM_VERSION;
    if (version !== plan.latestVersion) {
      throw new Error('it holds version ' + (version || 'none') + ', not ' + plan.latestVersion);
    }
    // The new CLI must at least run here before it replaces this one.
    const bin = path.join(fresh, path.basename(o.execPath || process.execPath));
    const ran = spawnSync(bin, ['--version'], { encoding: 'utf8', timeout: 30000 });
    const said = String(ran.stdout || '').trim();
    if (ran.error || ran.status !== 0 || said !== plan.latestVersion) {
      throw new Error('its CLI did not run here (' + ((ran.error && ran.error.message)
        || String(ran.stderr || '').trim() || said || 'exit ' + ran.status) + ')');
    }
    carryConfig(dir, fresh);
    fs.writeFileSync(path.join(fresh, H.TRIAL_FILE),
      H.formatTrial({ from: plan.currentVersion, to: plan.latestVersion, atMs: (o.now || Date.now)() }));
    return fresh;
  } catch (e) {
    fs.rmSync(stage, { recursive: true, force: true });
    throw e;
  } finally {
    fs.rmSync(tarball, { force: true });
  }
}

// Put the unpacked package in place of the installed one, keeping the old one
// as <dir>.prev until the new version's trial passes. A failed second rename
// puts the old folder back.
function swapIn(fresh, dir) {
  const prev = dir + H.PREV_SUFFIX;
  fs.rmSync(prev, { recursive: true, force: true });
  fs.renameSync(dir, prev);
  try {
    fs.renameSync(fresh, dir);
  } catch (e) {
    fs.renameSync(prev, dir);
    throw e;
  }
}

// Stage `plan`'s package and swap it in. The caller then exits, and HiveOS's
// miner loop starts the new h-run.sh.
async function updatePackage(plan, opts) {
  const o = opts || {};
  const dir = packageDir(o.execPath);
  const fresh = await stagePackage(plan, dir, o);
  try {
    swapIn(fresh, dir);
  } finally {
    fs.rmSync(path.dirname(fresh), { recursive: true, force: true });
  }
}

// The message for Blackwell cards on a driver too old for them, or null.
// Never rejects.
function checkDriver(opts) {
  const o = opts || {};
  return new Promise((resolve) => {
    execFile('lspci', ['-n'], { timeout: 10000 }, (err, stdout) => {
      if (err) return resolve(null);
      resolve(H.driverMessage(stdout, readText(o.versionFile || NVIDIA_VERSION)));
    });
  });
}

module.exports = {
  ENV,
  isHiveos,
  packageDir,
  postMessage,
  readTrial,
  isSkipped,
  confirmTrial,
  cleanStaging,
  stagePackage,
  swapIn,
  updatePackage,
  checkDriver,
};
