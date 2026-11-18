'use strict';

// ── Updates (electron-updater) ───────────────────────────────────────────────
// The desktop app's auto-update, moved out of main.js unchanged. The parts of
// main.js it reads and calls come through appContext.

const { app } = require('electron');
const { autoUpdater } = require('electron-updater');
const path = require('path');
const fs = require('fs');

const { ctx, share, late } = require('./appContext');
const { NETWORK } = require('../shared/config');
const { autoUpdateSupported } = require('../shared/platform');
const { formatUpdate, describeUpdateError } = require('../shared/updateStatus');
const { repairUpdateConfig } = require('../shared/updateConfig');

const send = late('send');
const loadSettings = late('loadSettings');
const persistSettings = late('persistSettings');
const endSession = late('endSession');
const refreshLoginItem = late('refreshLoginItem');
const openExternalSafe = late('openExternalSafe');

// Wire electron-updater to the renderer's update bar. autoUpdater pulls from the
// GitHub Releases feed (see build.publish); it only works in a packaged app, so
// main.js guards the call with app.isPackaged. Downloads happen automatically.
// With the window open, the user chooses when to restart via the
// 'app:update:install' channel; out of sight, it installs on its own (see
// installWhenHidden).
let manualUpdateCheck = false; // true while a user-initiated check is in flight
let updateTimer = null; // periodic re-check, so startup is not the only chance
let updateReady = false; // an update has downloaded and waits to install

// How long after an update's restart the new copy still counts as that restart,
// for starting in the tray (see resumeHidden in whenReady).
const UPDATE_RESTART_MS = 10 * 60 * 1000;

// Install the downloaded update now and restart into it. `hidden`: the window
// was out of sight, so the new copy starts in the tray too.
function installUpdate(hidden) {
  // Mining now, or paused waiting for the computer to go idle: resume after the
  // restart.
  const resume = !!ctx.session;
  // The relaunch needs this window to really close, not hide to the tray.
  ctx.quitting = true;
  try {
    // isSilent=true: install to the existing directory without re-showing the
    // assisted-installer wizard. isForceRunAfter=true: relaunch the app afterwards.
    autoUpdater.quitAndInstall(true, true);
  } catch (e) {
    ctx.quitting = false;
    send('miner:log', { level: 'error', line: 'update install failed: ' + e.message });
    return;
  }
  // An install the updater refused (no installer file, say) is reported through
  // its 'error' event and returns without quitting. The app carries on as it
  // was: a live START, and a window that closes to the tray.
  if (autoUpdater.quitAndInstallCalled === false) {
    ctx.quitting = false;
    return;
  }
  const restart = {};
  if (resume) restart.resumeMining = true;
  if (hidden) restart.resumeHidden = Date.now();
  if (resume || hidden) persistSettings(Object.assign({}, loadSettings(), restart));
  // Stop the LLM and miner now, before the installer relaunches the app.
  // llama-server binds a fixed port (8080); if the outgoing server still holds it,
  // the resumed instance's llama-server can't bind and the LLM silently fails to
  // start (the miner binds no port, so it's fine — exactly the "miner came back,
  // LLM didn't" symptom). Stopping here frees the port and VRAM before the new
  // instance starts. The quit itself comes on the next tick.
  endSession();
}

// An update installs when the app quits, and with the window closing to the
// tray the app rarely quits: a rig started at login could run an old build for
// months. So a downloaded update installs as soon as the window is out of sight
// (in the tray, or never shown after a launch at login), and the app restarts
// in the tray and picks its START back up. An open window keeps the "Update &
// restart" button instead.
//
// Not while a START is still bringing up the model: on a first run that is a
// download of several GB, which the restart would throw away. And not where the
// installer would ask for an administrator's approval, which nobody is there to
// give: an install for all users, in Program Files.
function installWhenHidden() {
  if (!updateReady || ctx.quitting || ctx.planRun) return;
  const win = ctx.win;
  if (win && !win.isDestroyed() && win.isVisible()) return;
  if (!canInstallQuietly()) return;
  installUpdate(true);
}

// Whether this user can write to the app's own folder, which is what an update
// replaces. Windows only: it is the install for all users that elevates. Tried
// with a real file, because fs.access ignores Windows' folder permissions.
function canInstallQuietly() {
  if (process.platform !== 'win32') return true;
  const probe = path.join(path.dirname(process.execPath), '.update-probe-' + process.pid);
  try {
    fs.writeFileSync(probe, '');
    fs.unlinkSync(probe);
    return true;
  } catch (e) {
    return false;
  }
}

// electron-updater reads resources/app-update.yml before every check. A crash or
// power cut during an install can leave that file unreadable (one Windows
// install's was all NUL bytes), and every check then fails until a reinstall.
// Point the updater at a good copy instead. The next update writes a good file
// again, and this then finds nothing to do.
function useGoodUpdateConfig() {
  try {
    const file = repairUpdateConfig({
      resourcesPath: process.resourcesPath, userDataPath: app.getPath('userData'), fs, join: path.join,
    });
    if (!file) return;
    autoUpdater.updateConfigPath = file;
    send('miner:log', { level: 'warn', line: 'the updater\'s app-update.yml was missing or damaged — using a good copy from the app\'s data folder' });
  } catch (e) {
    send('miner:log', { level: 'error', line: 'could not repair the updater\'s app-update.yml: ' + e.message });
  }
}

function setupUpdater() {
  const push = (phase, payload) => send('app:update', formatUpdate(phase, payload));
  useGoodUpdateConfig();
  autoUpdater.autoDownload = true;
  autoUpdater.autoInstallOnAppQuit = true;
  autoUpdater.on('checking-for-update', () => push('checking'));
  autoUpdater.on('update-available', (info) => { manualUpdateCheck = false; push('available', info); });
  // A manual check that finds nothing should say so; the automatic startup check
  // stays silent (no bar) when already current.
  autoUpdater.on('update-not-available', () => {
    if (manualUpdateCheck) { manualUpdateCheck = false; push('latest'); }
    else push('none');
  });
  autoUpdater.on('download-progress', (p) => push('progress', p));
  autoUpdater.on('update-downloaded', (info) => {
    updateReady = true;
    push('ready', info);
    installWhenHidden();
  });
  // An AppImage update moves to a new file named for its version and deletes
  // this one. Point a start-at-login entry at the new file now: after an install
  // on quit, the next start may be the login itself, and an entry naming the
  // deleted file would start nothing.
  autoUpdater.on('appimage-filename-updated', (file) => refreshLoginItem(file));
  autoUpdater.on('error', (err) => {
    manualUpdateCheck = false;
    push('error');
    send('miner:log', { level: 'error', line: 'update check failed: ' + describeUpdateError(err) });
  });
  // checkForUpdates() emits 'error' and THEN rethrows (AppUpdater.js), so the
  // handler above has already logged by the time the promise rejects. Logging
  // here too printed every network blip twice in the user's log.
  const check = () => autoUpdater.checkForUpdates().catch(() => {});
  check();
  // Re-check on a timer. The startup check used to be the only one for the
  // life of the process, so a rig that launched while GitHub's releases feed
  // was down never looked again — it would sit on a broken build until someone
  // restarted it, which is the opposite of what auto-update is for.
  updateTimer = setInterval(check, NETWORK.updateCheckIntervalMs);
  updateTimer.unref();
}

// User-initiated "Check for updates". In a dev/unpackaged run the real updater
// isn't wired, so walk the UI through checking → up-to-date so the button still
// gives feedback (the installed app runs a real check below).
function checkForUpdate() {
  // macOS: there is no in-app update to check for (the build is ad-hoc signed,
  // so Squirrel.Mac would refuse to install whatever it downloaded — see
  // shared/platform.autoUpdateSupported). Say so and open the Releases page,
  // which is the actual next step, rather than run a check that can only fail.
  if (!autoUpdateSupported(process.platform)) {
    send('app:update', formatUpdate('manual'));
    send('miner:log', { level: 'info', line: 'in-app updates are unavailable on macOS — opening ' + NETWORK.releasesUrl });
    openExternalSafe(NETWORK.releasesUrl);
    return;
  }
  if (!app.isPackaged) {
    send('app:update', formatUpdate('checking'));
    setTimeout(() => send('app:update', formatUpdate('latest')), 700);
    return;
  }
  manualUpdateCheck = true;
  send('app:update', formatUpdate('checking'));
  // Same as the startup check: the 'error' handler in setupUpdater has already
  // reset the flag, pushed the error state and logged the reason by the time
  // this rejects.
  autoUpdater.checkForUpdates().catch(() => {});
}

share({}, { installWhenHidden });

module.exports = { UPDATE_RESTART_MS, installUpdate, installWhenHidden, setupUpdater, checkForUpdate };
