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
// manual | error. `payload` carries { version } for available/ready and
// { percent } for progress. Returns { phase, text, show, ready?, error?,
// transient? } — `show` drives the bar's visibility, `ready` reveals the restart
// button, `error` styles it as a fault, `transient` marks a message the renderer
// auto-dismisses (the "you're up to date" result of a manual check).
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
    case 'latest':
      return { phase, text: 'You’re on the latest version', show: true, transient: true };
    // macOS, where there is no in-app update path: the build is ad-hoc signed,
    // so Squirrel.Mac would refuse to install what it downloaded. Not an
    // `error` — nothing failed, the platform simply updates by hand — and
    // transient like `latest`, because main.js has already opened the Releases
    // page and the bar is only confirming where the user just went.
    case 'manual':
      return { phase, text: 'Updates are manual on macOS — opening the Releases page', show: true, transient: true };
    case 'error':
      return { phase, text: 'Update check failed — see Logs.', show: true, error: true };
    default:
      return { phase: 'idle', text: '', show: false };
  }
}

// A one-line description of why an update check failed, for the user's log.
//
// electron-updater glues the ENTIRE response into err.message — for a 503 that
// means the HTML body and every response header, Set-Cookie values included.
// Logged verbatim it is unreadable, and it travels wherever someone pastes
// their log for support. Reduce it to the part that identifies the failure.
//
// Handles the shapes seen in the wild:
//   503 "method: GET url: https://…/releases.atom … Data: <html>… Headers: {…}
//   net::ERR_HTTP2_SERVER_REFUSED_STREAM
function describeUpdateError(err) {
  const raw = String((err && err.message) || err || '').trim();
  if (!raw) return 'unknown error';
  const status = raw.match(/^(\d{3})\b/);
  if (status) {
    const url = raw.match(/url:\s*(\S+)/);
    return 'HTTP ' + status[1] + (url ? ' from ' + url[1] : '');
  }
  // Anything else: the first line only, capped. Long single-line messages do
  // exist (a stack glued onto the text), so the cap is not redundant.
  const first = raw.split('\n')[0].trim();
  return first.length > 200 ? first.slice(0, 199) + '…' : first;
}

module.exports = { formatUpdate, clampPercent, describeUpdateError };
