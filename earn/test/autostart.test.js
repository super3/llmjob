'use strict';

const path = require('path');
const {
  HIDDEN_ARG, launchedHidden, linuxAutostartFile, linuxDesktopEntry, retargetDesktopEntry,
  macLaunchAgentFile, macLaunchAgent,
} = require('../src/shared/autostart');

// What a desktop makes of an Exec line, per the Desktop Entry spec: the
// general string escapes first (\\ is one backslash), then the quoting (inside
// quotes a backslash escapes ", `, $ and \), then the field codes (%% is %).
// Strict like GLib: any other escape, or a % that isn't %%, is refused.
function parseExec(line) {
  const value = line.slice('Exec='.length).replace(/\\(.)/g, (m, c) => {
    if (c === '\\') return '\\';
    throw new Error('invalid escape ' + m);
  });
  const args = [];
  let cur = '';
  let quoted = false;
  for (let i = 0; i < value.length; i++) {
    const c = value[i];
    if (quoted && c === '\\' && '"`$\\'.includes(value[i + 1])) cur += value[++i];
    else if (c === '"') quoted = !quoted;
    else if (c === ' ' && !quoted) {
      args.push(cur);
      cur = '';
    } else cur += c;
  }
  args.push(cur);
  return args.map((a) => a.replace(/%(.)/g, (m, c) => {
    if (c === '%') return '%';
    throw new Error('field code ' + m);
  }));
}

const execLine = (entry) => entry.split('\n').find((l) => l.startsWith('Exec='));

describe('autostart', () => {
  test('a launch at login is told apart by --hidden', () => {
    expect(HIDDEN_ARG).toBe('--hidden');
    expect(launchedHidden(['/opt/app', '--hidden'])).toBe(true);
    expect(launchedHidden(['/opt/app'])).toBe(false);
    expect(launchedHidden(undefined)).toBe(false);
  });

  test('the Linux entry lives in the autostart folder of the config home', () => {
    expect(linuxAutostartFile('/home/u/.config'))
      .toBe(path.join('/home/u/.config', 'autostart', 'llmjob-earn.desktop'));
  });

  test('the entry runs the AppImage hidden', () => {
    const entry = linuxDesktopEntry('/home/u/LLMJob-Earn.AppImage');
    expect(entry.split('\n')).toEqual([
      '[Desktop Entry]',
      'Type=Application',
      'Name=LLMJob Earn',
      'Comment=Mine Pearl with your GPU',
      'Exec="/home/u/LLMJob-Earn.AppImage" --hidden',
      'Terminal=false',
      'X-GNOME-Autostart-enabled=true',
      '',
    ]);
  });

  // Inside the quotes the spec reserves ", `, $ and \, and the file's string
  // escapes come off before the quoting does, so each escaping backslash is
  // doubled. A single one, as before, is an invalid escape GLib refuses.
  test('quotes a path with spaces and escapes the reserved characters', () => {
    const file = '/home/u/My Apps/$a"b`c\\d%e.AppImage';
    const line = execLine(linuxDesktopEntry(file));
    expect(line).toBe('Exec="/home/u/My Apps/\\\\$a\\\\"b\\\\`c\\\\\\\\d%%e.AppImage" --hidden');
    expect(parseExec(line)).toEqual([file, '--hidden']);
  });

  // An update renamed the AppImage. The entry follows it, and keeps a desktop's
  // own switch for it: rewriting it whole re-enabled an entry the user had
  // turned off.
  test('pointing an entry at a new AppImage changes only its Exec line', () => {
    const fresh = linuxDesktopEntry('/home/u/LLMJob-Earn-0.5.15.AppImage');
    const old = linuxDesktopEntry('/home/u/LLMJob-Earn-0.5.14.AppImage')
      .replace('X-GNOME-Autostart-enabled=true', 'X-GNOME-Autostart-enabled=false\nHidden=true');
    const next = retargetDesktopEntry(old, fresh);
    expect(execLine(next)).toBe('Exec="/home/u/LLMJob-Earn-0.5.15.AppImage" --hidden');
    expect(next).toContain('X-GNOME-Autostart-enabled=false\nHidden=true');
    expect(retargetDesktopEntry(next, fresh)).toBe(next);
  });

  test('a path with $& in it is written as it is, not expanded', () => {
    const fresh = linuxDesktopEntry('/home/u/$&$1/LLMJob.AppImage');
    const next = retargetDesktopEntry(linuxDesktopEntry('/old.AppImage'), fresh);
    expect(parseExec(execLine(next))).toEqual(['/home/u/$&$1/LLMJob.AppImage', '--hidden']);
  });

  test('an entry with no Exec line is replaced whole', () => {
    const fresh = linuxDesktopEntry('/home/u/LLMJob.AppImage');
    expect(retargetDesktopEntry('[Desktop Entry]\nName=broken\n', fresh)).toBe(fresh);
  });

  test('macOS: a LaunchAgent in the user\'s LaunchAgents folder runs the app hidden', () => {
    expect(macLaunchAgentFile('/Users/u'))
      .toBe(path.join('/Users/u', 'Library', 'LaunchAgents', 'com.llmjob.earn.plist'));
    const plist = macLaunchAgent('/Applications/R&D <test>/LLMJob Earn.app/Contents/MacOS/LLMJob Earn');
    expect(plist).toContain('<string>com.llmjob.earn</string>');
    expect(plist).toContain('<string>/Applications/R&amp;D &lt;test&gt;/LLMJob Earn.app/Contents/MacOS/LLMJob Earn</string>\n'
      + '    <string>--hidden</string>');
    expect(plist).toContain('<key>RunAtLoad</key>\n  <true/>');
  });
});
