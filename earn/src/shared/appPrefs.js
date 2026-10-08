'use strict';

// The desktop app's own behaviour, switched in Settings: what closing the window
// does, whether the app starts with the computer, and whether it mines only
// while nobody is using the computer. Pure, so the main process and its tests
// read them the same way.
//
// They live in their own file (preferences.json), not settings.json: every
// START rewrites settings.json from the renderer's mining fields, which would
// wipe these each time.

const DEFAULT_PREFS = Object.freeze({
  // Closing the window hides it to the tray and mining carries on. A miner that
  // stops when its window closes stops every time someone tidies their desktop.
  closeToTray: true,
  // Off until the user turns it on: a miner that starts itself with the
  // computer has to be something they chose.
  runAtStartup: false,
  mineWhenIdle: false,
  // The one-time "still running in the tray" notice has been shown. Not a
  // setting the user sees.
  trayHintShown: false,
});

// The computer counts as idle after this long without keyboard or mouse input.
const IDLE_AFTER_SEC = 5 * 60;
// How often the idle watcher looks. Short, so mining stops within a couple of
// seconds of someone coming back.
const IDLE_POLL_MS = 2000;

// Every known preference as a boolean, from whatever was saved: a missing,
// mistyped or hand-edited value falls back to its default.
function normalizePrefs(raw) {
  const saved = raw && typeof raw === 'object' ? raw : {};
  const prefs = {};
  for (const key of Object.keys(DEFAULT_PREFS)) {
    prefs[key] = typeof saved[key] === 'boolean' ? saved[key] : DEFAULT_PREFS[key];
  }
  return prefs;
}

// What the idle watcher does next: 'pause', 'resume', or null for nothing.
// `idleState` is Electron's powerMonitor.getSystemIdleState(IDLE_AFTER_SEC):
// 'active', 'idle', 'locked' or 'unknown'. A locked screen is someone who
// walked away. 'unknown' is a system that cannot tell (some Linux desktops):
// mining there rather than never starting.
function idleAction(idleState, paused) {
  const away = idleState !== 'active';
  if (paused && away) return 'resume';
  if (!paused && !away) return 'pause';
  return null;
}

module.exports = { DEFAULT_PREFS, IDLE_AFTER_SEC, IDLE_POLL_MS, normalizePrefs, idleAction };
