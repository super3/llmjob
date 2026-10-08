'use strict';

// The HiveOS custom-miner package only installs when one name lines up in four
// places: CUSTOM_NAME, the directory inside the tarball, the absolute paths in
// the manifest, and the stem of the versioned tarball filename. HiveOS derives
// the miner name by splitting a `<name>-<version>.tar.gz` install URL, then
// looks for `<name>/h-manifest.conf` in the archive — so publishing it as
// llmjob-earn-hiveos-<version>.tar.gz made every install fail with
// "No llmjob-earn-hiveos/h-manifest.conf". These are cheap text assertions
// because the packaging itself needs a built binary and a tar. The scripts
// themselves are run for real in hiveosScripts.test.js.

const fs = require('fs');
const path = require('path');
const { parseCliArgs } = require('../src/shared/cliArgs');

const earnRoot = path.join(__dirname, '..');
const manifestSrc = fs.readFileSync(path.join(earnRoot, 'hiveos', 'h-manifest.conf'), 'utf8');
const buildSrc = fs.readFileSync(path.join(earnRoot, 'scripts', 'build-hiveos.mjs'), 'utf8');

const NAME = 'llmjob-earn';

function manifestValue(key) {
  const m = new RegExp('^' + key + '=(.*)$', 'm').exec(manifestSrc);
  return m ? m[1].trim() : null;
}

describe('hiveos package naming', () => {
  test('CUSTOM_NAME is the name HiveOS installs under', () => {
    expect(manifestValue('CUSTOM_NAME')).toBe(NAME);
  });

  test('manifest paths live under CUSTOM_NAME', () => {
    expect(manifestValue('CUSTOM_CONFIG_FILENAME')).toBe(
      '/hive/miners/custom/' + NAME + '/' + NAME + '.conf');
    expect(manifestValue('CUSTOM_LOG_BASENAME')).toBe(
      '/var/log/miner/custom/' + NAME + '/' + NAME);
  });

  test('CUSTOM_VERSION is left blank for build-hiveos.mjs to stamp', () => {
    expect(manifestValue('CUSTOM_VERSION')).toBe('');
  });

  test('build script derives the package name from the manifest', () => {
    expect(buildSrc).toMatch(/CUSTOM_NAME=\(\.\+\)/);
    // Staged directory and tarball stem both come from that one value.
    expect(buildSrc).toContain("const pkgDir = join(stage, name)");
    expect(buildSrc).toContain("const out = join(dist, name + '-' + version + '.tar.gz')");
  });

  test('the versioned tarball is not published under the broken stem', () => {
    expect(buildSrc).not.toContain("'llmjob-earn-hiveos-'");
  });

  test('the unversioned legacy copy is still published', () => {
    // Flight sheets predating the versioned name point at this filename. HiveOS
    // reads "hiveos" as its version and "llmjob-earn" as the miner name.
    expect(buildSrc).toContain("'llmjob-earn-hiveos.tar.gz'");
  });

  test('a version with a dash is refused', () => {
    // HiveOS takes the text after the last '-' as the version, so 0.6.0-rc.1
    // would make the miner name llmjob-earn-0.6.0 and the install would fail.
    expect(buildSrc).toMatch(/if \(version\.includes\('-'\)\)/);
  });

  test('the cores are packed 0644, whatever mode they arrived with', () => {
    // v0.4.2-v0.4.5 shipped a world-writable pearl_core.node that a root
    // process loads.
    expect(buildSrc).toContain("chmodSync(join(pkgDir, 'pearl_core.node'), 0o644)");
    expect(buildSrc).toContain("chmodSync(join(pkgDir, 'pearl_core_cu13.node'), 0o644)");
  });
});

describe('hiveos flight sheet docs', () => {
  test('the Installation URL is versioned, not releases/latest', () => {
    // HiveOS keeps the old build while the file name stays the same, and
    // releases/latest/download/llmjob-earn-hiveos.tar.gz returned 404 for every
    // release from v0.5.0 to v0.5.12.
    expect(manifestSrc).toContain(
      'https://github.com/super3/llmjob/releases/download/v<version>/llmjob-earn-<version>.tar.gz');
    expect(manifestSrc).not.toContain('releases/latest');
  });

  test('the Extra config example is flags the CLI takes', () => {
    // The old example (--region eu1 --difficulty 131072) made the CLI exit at
    // once, and HiveOS restarted it every 3 s.
    const m = /Extra config\s+→ .*e\.g\. (.+)$/m.exec(manifestSrc);
    expect(m).not.toBeNull();
    const r = parseCliArgs(['--address', 'prl1pql8r6m4z9x7v2k0t3whu8e2snd4p6c'].concat(m[1].trim().split(/\s+/)));
    expect(r.errors).toEqual([]);
  });

  test('h-config.sh points the CLI at the stats file h-stats.sh reads', () => {
    const configSrc = fs.readFileSync(path.join(earnRoot, 'hiveos', 'h-config.sh'), 'utf8');
    const statsSrc = fs.readFileSync(path.join(earnRoot, 'hiveos', 'h-stats.sh'), 'utf8');
    const file = /--stats-file ([^\s"]+)/.exec(configSrc)[1];
    expect(statsSrc).toContain('${LLMJOB_EARN_STATS_FILE:-' + file + '}');
  });
});
