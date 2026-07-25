'use strict';

// The HiveOS custom-miner package only installs when one name lines up in four
// places: CUSTOM_NAME, the directory inside the tarball, the absolute paths in
// the manifest, and the stem of the versioned tarball filename. HiveOS derives
// the miner name by splitting a `<name>-<version>.tar.gz` install URL, then
// looks for `<name>/h-manifest.conf` in the archive — so publishing it as
// llmjob-earn-hiveos-<version>.tar.gz made every install fail with
// "No llmjob-earn-hiveos/h-manifest.conf". These are cheap text assertions
// because the packaging itself needs a built binary and a tar.

const fs = require('fs');
const path = require('path');

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
    // Flight sheets predating the versioned name point at this filename; with no
    // -<version> suffix to split on, HiveOS falls back to the tar's directory.
    expect(buildSrc).toContain("'llmjob-earn-hiveos.tar.gz'");
  });
});
