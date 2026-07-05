'use strict';

// Pure mapping from an electron-updater lifecycle phase to the small display
// model the renderer shows in its update bar. Kept free of Electron so it is
// fully unit-tested; main.js feeds it real autoUpdater events.

// Coerce a raw progress value to a 0-100 integer.
function clampPercent(p) {
  const n = Math.round(Number(p) || 0);
  if (n < 0) return 0;
  if (n > 100) return 100;
  return n;
}

// `phase` is one of: checking | available | progress | ready | none | latest |
// dev | error. `payload` carries { version } for available/ready/latest and
// { percent } for progress. Returns { phase, text, show, ready?, error?,
// transient? } — `show` drives the bar's visibility, `ready` reveals the restart
// button, `error` styles it as a fault, `transient` marks a message the renderer
// auto-dismisses (the "up to date" / dev results of a manual check).
function formatUpdate(phase, payload) {
  switch (phase) {
    case 'checking':
      return { phase, text: 'Checking for updates…', show: true };
    case 'available': {
      const v = payload && payload.version ? ' v' + payload.version : '';
      return { phase, text: 'Update' + v + ' available — downloading…', show: true };
    }
    case 'progress':
      return { phase, text: 'Downloading update… ' + clampPercent(payload && payload.percent) + '%', show: true };
    case 'ready': {
      const v = payload && payload.version ? ' v' + payload.version : '';
      return { phase, text: 'Update' + v + ' ready', show: true, ready: true };
    }
    case 'none':
      return { phase, text: '', show: false };
    case 'latest': {
      const v = payload && payload.version ? ' (v' + payload.version + ')' : '';
      return { phase, text: 'You’re on the latest version' + v, show: true, transient: true };
    }
    case 'dev':
      return { phase, text: 'Auto-update works in the installed app.', show: true, transient: true };
    case 'error':
      return { phase, text: 'Update check failed — see Logs.', show: true, error: true };
    default:
      return { phase: 'idle', text: '', show: false };
  }
}

module.exports = { formatUpdate, clampPercent };
