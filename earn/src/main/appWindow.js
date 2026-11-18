'use strict';

// ── Window and tray ──────────────────────────────────────────────────────────
// The desktop app's window, its tray icon, and closing to the tray, moved out
// of main.js unchanged. The parts of main.js it reads and calls come through
// appContext.

const { app, BrowserWindow, Menu, Tray, Notification, nativeImage } = require('electron');
const { execFile } = require('child_process');
const path = require('path');

const { ctx, share, late } = require('./appContext');
const autostart = require('../shared/autostart');
const trayHost = require('../shared/trayHost');

const getPrefs = late('getPrefs');
const savePrefs = late('savePrefs');
const installWhenHidden = late('installWhenHidden');

// Started by the login item or the autostart entry: stay in the tray and start
// mining. Read once; a window made later (from the tray) shows as usual.
let hiddenLaunch = autostart.launchedHidden(process.argv);
let tray = null;
// Set on a real quit (the tray's Quit, an update, logging off), so the window's
// close handler lets the window close instead of hiding it.
let quitting = false;
// The Windows message a user's close arrives as (see the window's close handler).
const WM_SYSCOMMAND = 0x0112;
const SC_CLOSE = 0xf060;

let win = null;

function appIcon() {
  const dir = path.join(__dirname, '..', 'assets');
  return path.join(dir, process.platform === 'win32' ? 'icon.ico' : 'icon.png');
}

// The tray icon. The macOS menu bar wants a 16-point image, and icon.png is the
// 1024 px app icon, so it is scaled down there, with a 32 px copy for Retina.
function trayIcon() {
  if (process.platform !== 'darwin') return appIcon();
  const full = nativeImage.createFromPath(appIcon());
  const icon = full.resize({ width: 16, height: 16 });
  icon.addRepresentation({ scaleFactor: 2, buffer: full.resize({ width: 32, height: 32 }).toPNG() });
  return icon;
}

function createWindow() {
  win = new BrowserWindow({
    width: 620,
    height: 650,
    minWidth: 560,
    minHeight: 560,
    backgroundColor: '#fcfcfb',
    autoHideMenuBar: true,
    show: false,
    title: 'LLMJob Earn',
    icon: appIcon(),
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  // A page that loads, or loads again, is not listening for log lines until
  // its onLog says so (see send).
  win.webContents.on('did-start-loading', () => { ctx.logListening = false; });
  win.loadFile(path.join(__dirname, '..', 'renderer', 'index.html'));

  // A launch at login stays in the tray. Only when there is a tray to find it
  // in: without one, a hidden window could never be opened.
  const showOnLoad = !(hiddenLaunch && tray);
  hiddenLaunch = false;

  // Size the window to the rendered content so there's no default scrollbar and
  // no trailing whitespace, whatever the platform chrome / font metrics / DPI.
  // Do it before showing (window starts hidden) to avoid a resize flash. The
  // renderer keeps overflow-y:auto, so a transient taller state (update bar,
  // engine error) still scrolls rather than clipping.
  win.webContents.on('did-finish-load', () => {
    fitWindowToContent().finally(() => { if (showOnLoad && win && !win.isDestroyed()) win.show(); });
  });
  setTimeout(() => { if (showOnLoad && win && !win.isDestroyed() && !win.isVisible()) win.show(); }, 1500);

  // Closing the window hides it to the tray, and mining carries on, unless the
  // user turned that off in Settings or the app is really quitting.
  //
  // Only when the user closed it. On Windows the window's X, Alt+F4 and the
  // taskbar's Close all arrive as SC_CLOSE first. A close without one comes from
  // another program that wants the app gone: the installer before it replaces
  // the app's files, the uninstaller, Task Manager's End task. Hiding the window
  // from those left the app running until the installer force-killed it a second
  // later, so the miner never stopped cleanly. They quit it instead. A focused
  // window counts as the user's too, in case a close reaches it some other way.
  let userClose = false;
  if (process.platform === 'win32') {
    win.hookWindowMessage(WM_SYSCOMMAND, (wParam) => {
      if (Buffer.isBuffer(wParam) && (wParam.readUInt32LE(0) & 0xfff0) === SC_CLOSE) userClose = true;
    });
  }
  win.on('close', (e) => {
    const fromUser = process.platform !== 'win32' || userClose || win.isFocused();
    userClose = false;
    if (quitting || !fromUser || !tray || !getPrefs().closeToTray) return;
    e.preventDefault();
    win.hide();
    trayHint();
  });
  // Out of sight, a downloaded update can install.
  win.on('hide', installWhenHidden);
  // Logging off or shutting down on Windows: let the window close.
  win.on('session-end', () => { quitting = true; });

  // Right-click cut/copy/paste — Electron has no default context menu, so
  // pasting a payout address (or copying it) is otherwise mouse-inaccessible.
  win.webContents.on('context-menu', (_e, params) => {
    const f = params.editFlags;
    const items = [];
    if (params.isEditable || params.selectionText) {
      items.push(
        { role: 'cut', enabled: f.canCut },
        { role: 'copy', enabled: f.canCopy },
        { role: 'paste', enabled: f.canPaste },
        { type: 'separator' },
        { role: 'selectAll' }
      );
    }
    if (items.length && !win.isDestroyed()) Menu.buildFromTemplate(items).popup({ window: win });
  });
}

// Bring the window back: from the tray, or when the app is opened again while
// it runs (second-instance).
function showWindow() {
  if (!win || win.isDestroyed()) {
    createWindow();
    return;
  }
  if (win.isMinimized()) win.restore();
  win.show();
  win.focus();
}

function quitApp() {
  quitting = true;
  app.quit();
}

// The tray icon: the way back to a closed window, and the way to quit.
function createTray() {
  try {
    tray = new Tray(trayIcon());
  } catch (e) {
    // No tray on this desktop. Closing the window then quits, as before.
    tray = null;
    return;
  }
  tray.setToolTip('LLMJob Earn');
  tray.setContextMenu(Menu.buildFromTemplate([
    { label: 'Open LLMJob Earn', click: showWindow },
    { type: 'separator' },
    { label: 'Quit', click: quitApp },
  ]));
  tray.on('click', showWindow);
}

// GNOME draws no tray icon without an AppIndicator extension, though Electron
// makes one without complaint (see shared/trayHost). There, drop the icon, so
// closing the window quits rather than hiding it with no way back. Waits up to
// `waitMs` for the extension, which may still be loading at login.
const TRAY_WAIT_MS = 15000;
async function confirmTray(waitMs) {
  if (!tray || process.platform !== 'linux' || !trayHost.isGnome(process.env)) return;
  const deadline = Date.now() + waitMs;
  for (;;) {
    // false means no host; null means the bus couldn't be asked, which keeps
    // the icon, as before.
    if ((await trayHost.queryTrayHost(execFile)) !== false) return;
    if (Date.now() >= deadline) break;
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  tray.destroy();
  tray = null;
}

// The first time the window closes to the tray, say so once. Otherwise the app
// looks closed while the GPU is still working. Marked shown only once the
// notification was: one that never appeared is tried again on the next run.
let trayHintTried = false;
function trayHint() {
  if (getPrefs().trayHintShown || trayHintTried) return;
  trayHintTried = true;
  if (!Notification.isSupported()) {
    savePrefs({ trayHintShown: true });
    return;
  }
  const notice = new Notification({
    title: 'LLMJob Earn is still running',
    body: 'It keeps running in the tray. Click the tray icon to open it, or right-click it to quit. '
      + 'You can change this in Settings.',
  });
  notice.on('show', () => savePrefs({ trayHintShown: true }));
  notice.show();
}

// How far the frame may sit from the measured content before it is worth
// resizing. A pixel or two is invisible; chasing it is not, because the resize
// itself perturbs the layout (see below).
const FIT_TOLERANCE_PX = 2;

// Measure the rendered content (.app) and set the window's content area to it,
// so the miner view fits exactly. Best-effort — resolves regardless of errors.
//
// Only resizes when the frame is actually wrong. Re-applying a size that is
// already correct is not free: setContentSize round-trips the width back
// through getContentSize, and on a DPI-scaled display that round trip is not
// lossless, so a "no-op" fit still nudged the frame by a pixel. The renderer's
// ResizeObserver saw .app change and asked for another fit, which nudged it
// again — the window resized continuously on startup until a minimize/restore
// settled it, and any manual resize set it going again.
function fitWindowToContent() {
  if (!win || win.isDestroyed()) return Promise.resolve();
  return win.webContents
    .executeJavaScript('Math.ceil((document.querySelector(".app") || document.body).getBoundingClientRect().height)')
    .then((h) => {
      if (!win || win.isDestroyed() || !Number.isFinite(h) || h <= 0) return;
      const [width, current] = win.getContentSize();
      if (Math.abs(current - h) <= FIT_TOLERANCE_PX) return;
      win.setContentSize(width, h);
    })
    .catch(() => {});
}

share({
  win: { get: () => win },
  quitting: { get: () => quitting, set: (v) => { quitting = v; } },
  hiddenLaunch: { get: () => hiddenLaunch, set: (v) => { hiddenLaunch = v; } },
}, {});

module.exports = { TRAY_WAIT_MS, createWindow, showWindow, createTray, confirmTray, fitWindowToContent };
