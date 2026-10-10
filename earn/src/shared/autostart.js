'use strict';

// Starting the desktop app with the computer. Windows has a login item for it
// (Electron's app.setLoginItemSettings). Linux has no such call, so the app
// writes a freedesktop autostart entry, which every mainstream desktop reads
// at login. macOS gets a LaunchAgent: its login item can't pass --hidden, and
// from macOS 13 the app can't tell that the login item started it. Pure:
// main.js does the file writes.

const path = require('path');

// Passed by the login item and the autostart entry: start in the tray, not on
// screen, and start mining.
const HIDDEN_ARG = '--hidden';

// The LaunchAgent's name, the app's id.
const MAC_AGENT_LABEL = 'com.llmjob.earn';

function launchedHidden(argv) {
  return Array.isArray(argv) && argv.includes(HIDDEN_ARG);
}

// ~/.config/autostart/llmjob-earn.desktop, or under $XDG_CONFIG_HOME.
function linuxAutostartFile(configHome) {
  return path.join(configHome, 'autostart', 'llmjob-earn.desktop');
}

// The entry's Exec line quotes the path. Inside quotes the spec reserves ", `,
// $ and \, each escaped with a backslash. The file's general string escapes
// are undone first, so each of those backslashes is written doubled: a literal
// $ is \\$. And % starts a field code, so a literal one is %%. An AppImage in a
// folder with a space, a dollar sign or a percent sign in its name must still
// launch.
function quoteExec(file) {
  const quoted = '"' + String(file).replace(/["`$\\]/g, (c) => '\\' + c) + '"';
  return quoted.replace(/\\/g, '\\\\').replace(/%/g, '%%');
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

// An entry already on disk, pointed at the command in `fresh`. Only its Exec
// line changes, so a desktop's own switch for the entry (Hidden=true,
// X-GNOME-Autostart-enabled=false) stays as the user left it. An entry with no
// Exec line is replaced whole.
function retargetDesktopEntry(current, fresh) {
  const exec = fresh.split('\n').find((line) => line.startsWith('Exec='));
  const line = /^Exec=.*$/m;
  // A function, not a string: the path may hold $& or $1, which a replacement
  // string would expand.
  return line.test(current) ? current.replace(line, () => exec) : fresh;
}

// ~/Library/LaunchAgents/com.llmjob.earn.plist.
function macLaunchAgentFile(home) {
  return path.join(home, 'Library', 'LaunchAgents', MAC_AGENT_LABEL + '.plist');
}

function xmlText(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// Runs the app's binary (inside LLMJob Earn.app) with --hidden at login.
function macLaunchAgent(execPath) {
  return [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">',
    '<plist version="1.0">',
    '<dict>',
    '  <key>Label</key>',
    '  <string>' + MAC_AGENT_LABEL + '</string>',
    '  <key>ProgramArguments</key>',
    '  <array>',
    '    <string>' + xmlText(execPath) + '</string>',
    '    <string>' + HIDDEN_ARG + '</string>',
    '  </array>',
    '  <key>RunAtLoad</key>',
    '  <true/>',
    '  <key>ProcessType</key>',
    '  <string>Interactive</string>',
    '</dict>',
    '</plist>',
    '',
  ].join('\n');
}

module.exports = {
  HIDDEN_ARG, launchedHidden, linuxAutostartFile, linuxDesktopEntry, retargetDesktopEntry,
  macLaunchAgentFile, macLaunchAgent,
};
