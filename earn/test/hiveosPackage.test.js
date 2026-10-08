'use strict';

// The HiveOS custom-miner package only installs when one name lines up in four
// places: CUSTOM_NAME, the directory inside the tarball, the absolute paths in
// the manifest, and the stem of the versioned tarball filename. HiveOS derives
// the miner name by splitting a `<name>-<version>.tar.gz` install URL, then
// looks for `<name>/h-manifest.conf` in the archive — so publishing it as
// llmjob-earn-hiveos-<version>.tar.gz made every install fail with
// "No llmjob-earn-hiveos/h-manifest.conf". Most of these are text assertions.
// "build-hiveos.mjs, run for real" below also runs the script on Linux, with
// stand-in binary and core files, so a break shows up in `npm test` and not
// first on a release run. The hook scripts are run in hiveosScripts.test.js.

const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');
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

// Runs a copy of build-hiveos.mjs in a temp dir laid out like earn/: its own
// package.json, the real hiveos/ files, and dist/ holding stand-ins for the CLI
// binary and cores, left 0666 the way the CI artifact arrives. Needs Linux for
// tar and file modes.
const tarOk = process.platform === 'linux'
  && spawnSync('tar', ['--version'], { stdio: 'ignore' }).status === 0;
(tarOk ? describe : describe.skip)('build-hiveos.mjs, run for real', () => {
  let tmp;
  beforeEach(() => {
    tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'hiveos-build-'));
    fs.mkdirSync(path.join(tmp, 'scripts'));
    fs.mkdirSync(path.join(tmp, 'hiveos'));
    fs.mkdirSync(path.join(tmp, 'dist'));
    fs.copyFileSync(path.join(earnRoot, 'scripts', 'build-hiveos.mjs'),
      path.join(tmp, 'scripts', 'build-hiveos.mjs'));
    for (const f of ['h-config.sh', 'h-run.sh', 'h-stats.sh', 'h-manifest.conf']) {
      fs.copyFileSync(path.join(earnRoot, 'hiveos', f), path.join(tmp, 'hiveos', f));
    }
    for (const f of ['llmjob-earn-cli-linux', 'pearl_core.node', 'pearl_core_cu13.node']) {
      const p = path.join(tmp, 'dist', f);
      fs.writeFileSync(p, 'stand-in ' + f + '\n');
      fs.chmodSync(p, 0o666);
    }
  });
  afterEach(() => {
    fs.rmSync(tmp, { recursive: true, force: true });
  });

  function build(version) {
    fs.writeFileSync(path.join(tmp, 'package.json'), JSON.stringify({ version }));
    return spawnSync(process.execPath, [path.join(tmp, 'scripts', 'build-hiveos.mjs')],
      { encoding: 'utf8' });
  }
  function tarballs() {
    return fs.readdirSync(path.join(tmp, 'dist')).filter((f) => f.endsWith('.tar.gz')).sort();
  }

  test('packs llmjob-earn/ with the right files, modes and version', () => {
    const r = build('9.8.7');
    expect(r.stderr).toBe('');
    expect(r.status).toBe(0);
    expect(tarballs()).toEqual(['llmjob-earn-9.8.7.tar.gz', 'llmjob-earn-hiveos.tar.gz']);

    const out = path.join(tmp, 'dist', 'llmjob-earn-9.8.7.tar.gz');
    const list = spawnSync('tar', ['-tvzf', out], { encoding: 'utf8' }).stdout;
    const modes = {};
    for (const line of list.trim().split('\n')) {
      const cols = line.trim().split(/\s+/);
      modes[cols[cols.length - 1]] = cols[0];
    }
    expect(modes).toEqual({
      'llmjob-earn/': expect.stringMatching(/^d/),
      'llmjob-earn/h-config.sh': '-rwxr-xr-x',
      'llmjob-earn/h-run.sh': '-rwxr-xr-x',
      'llmjob-earn/h-stats.sh': '-rwxr-xr-x',
      'llmjob-earn/h-manifest.conf': '-rw-r--r--',
      'llmjob-earn/llmjob-earn-cli-linux': '-rwxr-xr-x',
      'llmjob-earn/pearl_core.node': '-rw-r--r--',
      'llmjob-earn/pearl_core_cu13.node': '-rw-r--r--',
    });

    const manifest = spawnSync('tar', ['-xzOf', out, 'llmjob-earn/h-manifest.conf'],
      { encoding: 'utf8' }).stdout;
    expect(manifest).toMatch(/^CUSTOM_VERSION=9\.8\.7$/m);
    expect(manifest).toMatch(/^CUSTOM_NAME=llmjob-earn$/m);
    expect(fs.readFileSync(path.join(tmp, 'dist', 'llmjob-earn-hiveos.tar.gz')))
      .toEqual(fs.readFileSync(out));
  });

  test('a version with a dash builds nothing and says why', () => {
    const r = build('9.8.7-rc.1');
    expect(r.status).toBe(1);
    expect(r.stderr).toContain("contains '-'");
    expect(tarballs()).toEqual([]);
  });

  test('a missing pearl_core.node builds nothing', () => {
    fs.rmSync(path.join(tmp, 'dist', 'pearl_core.node'));
    const r = build('9.8.7');
    expect(r.status).toBe(1);
    expect(r.stderr).toContain('could not mine');
    expect(tarballs()).toEqual([]);
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
