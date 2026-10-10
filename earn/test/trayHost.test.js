'use strict';

const { isGnome, queryTrayHost, QUERIES, QUERY_TIMEOUT_MS } = require('../src/shared/trayHost');

// An execFile double: `answers` maps a command to (cb) => cb(err, stdout).
function fakeExec(answers) {
  return jest.fn((cmd, args, opts, cb) => answers[cmd](cb));
}
const missing = (cb) => cb(Object.assign(new Error('spawn ENOENT'), { code: 'ENOENT' }));

describe('trayHost', () => {
  test('GNOME is told by XDG_CURRENT_DESKTOP, Ubuntu\'s and Pop\'s included', () => {
    for (const d of ['GNOME', 'ubuntu:GNOME', 'pop:GNOME', 'GNOME-Classic:GNOME']) expect([d, isGnome({ XDG_CURRENT_DESKTOP: d })]).toEqual([d, true]);
    for (const d of ['KDE', 'XFCE', 'X-Cinnamon', 'MATE', '']) expect([d, isGnome({ XDG_CURRENT_DESKTOP: d })]).toEqual([d, false]);
    expect(isGnome({})).toBe(false);
    expect(isGnome(undefined)).toBe(false);
  });

  test('asks gdbus whether a StatusNotifier host is registered, briefly', async () => {
    const exec = fakeExec({ gdbus: (cb) => cb(null, '(<true>,)\n') });
    expect(await queryTrayHost(exec)).toBe(true);
    expect(exec).toHaveBeenCalledWith('gdbus', QUERIES[0][1], { timeout: QUERY_TIMEOUT_MS }, expect.any(Function));
    expect(QUERIES[0][1]).toContain('IsStatusNotifierHostRegistered');
  });

  // GNOME without an AppIndicator extension has no watcher on the bus at all.
  test('no host, and no watcher on the bus, both mean no tray', async () => {
    expect(await queryTrayHost(fakeExec({ gdbus: (cb) => cb(null, '(<false>,)\n') }))).toBe(false);
    const unknown = Object.assign(new Error('GDBus.Error:org.freedesktop.DBus.Error.ServiceUnknown'), { code: 1 });
    expect(await queryTrayHost(fakeExec({ gdbus: (cb) => cb(unknown, '') }))).toBe(false);
  });

  test('falls back to dbus-send when gdbus is not installed', async () => {
    const exec = fakeExec({ gdbus: missing, 'dbus-send': (cb) => cb(null, 'method return\n   variant       boolean true\n') });
    expect(await queryTrayHost(exec)).toBe(true);
    expect(exec.mock.calls.map((c) => c[0])).toEqual(['gdbus', 'dbus-send']);
    expect(await queryTrayHost(fakeExec({ gdbus: missing, 'dbus-send': (cb) => cb(null, 'variant boolean false') }))).toBe(false);
  });

  // Unknown keeps the tray, as before this check existed.
  test('cannot tell when neither tool is installed, or the bus does not answer in time', async () => {
    expect(await queryTrayHost(fakeExec({ gdbus: missing, 'dbus-send': missing }))).toBeNull();
    const slow = Object.assign(new Error('timed out'), { killed: true });
    expect(await queryTrayHost(fakeExec({ gdbus: (cb) => cb(slow, '') }))).toBeNull();
  });
});
