'use strict';

// Whether a GNOME desktop will show the app's tray icon.
//
// Electron puts the icon on the session bus as a StatusNotifierItem, the
// freedesktop tray protocol, or failing that in the older XEmbed tray. KDE,
// Xfce, Cinnamon, MATE and most others draw one or the other. GNOME draws
// neither without an AppIndicator extension (Ubuntu ships one, Fedora and Debian
// don't), and Electron can't tell: it makes the icon without complaint and the
// icon never appears. Closing the window would then hide it with no way back.
//
// So on GNOME the app asks the bus whether a StatusNotifier host, the program
// that draws these icons, is registered. Elsewhere the icon Electron makes is
// trusted, as before. Pure apart from the injected execFile.

function isGnome(env) {
  return /gnome/i.test(String((env && env.XDG_CURRENT_DESKTOP) || ''));
}

const WATCHER = 'org.kde.StatusNotifierWatcher';

// The same question through either of the two common D-Bus command-line tools:
// gdbus comes with GLib, dbus-send with D-Bus itself.
const QUERIES = [
  ['gdbus', ['call', '--session', '--dest', WATCHER, '--object-path', '/StatusNotifierWatcher',
    '--method', 'org.freedesktop.DBus.Properties.Get', WATCHER, 'IsStatusNotifierHostRegistered']],
  ['dbus-send', ['--session', '--print-reply', '--dest=' + WATCHER, '/StatusNotifierWatcher',
    'org.freedesktop.DBus.Properties.Get', 'string:' + WATCHER, 'string:IsStatusNotifierHostRegistered']],
];

const QUERY_TIMEOUT_MS = 2000;

// Resolves true when a host is registered, false when none is (no watcher on
// the bus counts as none), and null when the bus can't be asked: neither tool
// is installed, or the question timed out.
function queryTrayHost(execFile) {
  const ask = (i) => new Promise((resolve) => {
    if (i >= QUERIES.length) return resolve(null);
    const [cmd, args] = QUERIES[i];
    execFile(cmd, args, { timeout: QUERY_TIMEOUT_MS }, (err, stdout) => {
      if (err && err.code === 'ENOENT') return resolve(ask(i + 1));
      if (err && err.killed) return resolve(null);
      // gdbus prints (<true>,), dbus-send "variant boolean true".
      resolve(!err && /\btrue\b/.test(String(stdout)));
    });
  });
  return ask(0);
}

module.exports = { isGnome, queryTrayHost, QUERIES, QUERY_TIMEOUT_MS };
