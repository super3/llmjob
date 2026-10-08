'use strict';

// Starting the desktop app with the computer. Windows has a login item for it
// (Electron's app.setLoginItemSettings). Linux has no such call, so the app
// writes a freedesktop autostart entry, which every mainstream desktop reads
// at login. Pure: main.js does the file writes.

const path = require('path');

// Passed by the login item and the autostart entry: start in the tray, not on
// screen, and start mining.
const HIDDEN_ARG = '--hidden';

function launchedHidden(argv) {
  return Array.isArray(argv) && argv.includes(HIDDEN_ARG);
}

// ~/.config/autostart/llmjob-earn.desktop, or under $XDG_CONFIG_HOME.
function linuxAutostartFile(configHome) {
  return path.join(configHome, 'autostart', 'llmjob-earn.desktop');
}

// The entry's Exec line quotes the path, and inside quotes the spec reserves ",
// `, $ and \, each escaped with a backslash. An AppImage in a folder with a
// space or a dollar sign in its name must still launch.
function quoteExec(file) {
  return '"' + String(file).replace(/["`$\\]/g, (c) => '\\' + c) + '"';
}

// `execPath` is the AppImage itself ($APPIMAGE), which is what has to run: the
// binary inside it lives in a mount that is gone after a reboot.
function linuxDesktopEntry(execPath) {
  return [
    '[Desktop Entry]',
    'Type=Application',
    'Name=LLMJob Earn',
    'Comment=Mine Pearl with your GPU',
    'Exec=' + quoteExec(execPath) + ' ' + HIDDEN_ARG,
    'Terminal=false',
    'X-GNOME-Autostart-enabled=true',
    '',
  ].join('\n');
}

module.exports = { HIDDEN_ARG, launchedHidden, linuxAutostartFile, linuxDesktopEntry };
