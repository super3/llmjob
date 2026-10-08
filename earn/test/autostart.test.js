'use strict';

const path = require('path');
const { HIDDEN_ARG, launchedHidden, linuxAutostartFile, linuxDesktopEntry } = require('../src/shared/autostart');

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

  // Inside the quotes the spec reserves ", `, $ and \.
  test('quotes a path with spaces and escapes the reserved characters', () => {
    const entry = linuxDesktopEntry('/home/u/My Apps/$a"b`c\\d.AppImage');
    expect(entry).toContain('Exec="/home/u/My Apps/\\$a\\"b\\`c\\\\d.AppImage" --hidden');
  });
});
