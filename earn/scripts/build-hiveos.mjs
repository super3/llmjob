#!/usr/bin/env node
'use strict';

// Package the HiveOS custom-miner archive: the standalone CLI binary, its
// mining cores and the hiveos/ hook scripts, tarred as
// dist/llmjob-earn-<version>.tar.gz (plus an unversioned copy, see below) — the
// file a flight sheet's "Installation URL" points at. Run `npm run dist:cli`
// first (or pass an explicit binary path as the first argument). CI runs this
// right after dist:cli, so the tarball holds the same binary and cores the
// release publishes.

import { execFileSync } from 'node:child_process';
import { mkdirSync, copyFileSync, readFileSync, writeFileSync, chmodSync, rmSync, existsSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const dist = join(root, 'dist');
const bin = process.argv[2] ? resolve(process.argv[2]) : join(dist, 'llmjob-earn-cli-linux');
const version = JSON.parse(readFileSync(join(root, 'package.json'), 'utf8')).version;

// HiveOS takes the text after the LAST '-' of the file name as the version and
// the rest as the miner name. A version like 0.6.0-rc.1 would make the miner
// name "llmjob-earn-0.6.0", and HiveOS would refuse to install it.
if (version.includes('-')) {
  console.error('version ' + version + " contains '-': HiveOS would read the miner name as "
    + "llmjob-earn-" + version.slice(0, version.lastIndexOf('-')) + '. Use a version without one.');
  process.exit(1);
}

if (!existsSync(bin)) {
  console.error('CLI binary not found: ' + bin + ' — run `npm run dist:cli` first');
  process.exit(1);
}

// The custom-miner name, read from the manifest rather than repeated here: it
// has to be identical in the staged directory name, the tarball filename and
// CUSTOM_NAME, and deriving all three from one source keeps them that way.
const manifestSrc = readFileSync(join(root, 'hiveos', 'h-manifest.conf'), 'utf8');
const nameMatch = /^CUSTOM_NAME=(.+)$/m.exec(manifestSrc);
if (!nameMatch) {
  console.error('hiveos/h-manifest.conf has no CUSTOM_NAME');
  process.exit(1);
}
const name = nameMatch[1].trim();

// Stage <dist>/hiveos-stage/<name>/ — the directory name inside the tar must
// match CUSTOM_NAME for the HiveOS installer to place it correctly.
const stage = join(dist, 'hiveos-stage');
const pkgDir = join(stage, name);
rmSync(stage, { recursive: true, force: true });
mkdirSync(pkgDir, { recursive: true });

for (const f of ['h-config.sh', 'h-run.sh', 'h-stats.sh']) {
  copyFileSync(join(root, 'hiveos', f), join(pkgDir, f));
  chmodSync(join(pkgDir, f), 0o755);
}

// Stamp the package version into the manifest.
const manifest = manifestSrc.replace(/^CUSTOM_VERSION=.*$/m, 'CUSTOM_VERSION=' + version);
writeFileSync(join(pkgDir, 'h-manifest.conf'), manifest);
chmodSync(join(pkgDir, 'h-manifest.conf'), 0o644);

copyFileSync(bin, join(pkgDir, 'llmjob-earn-cli-linux'));
chmodSync(join(pkgDir, 'llmjob-earn-cli-linux'), 0o755);

// The native core sits beside the binary — the loader's first packaged-CLI
// candidate. A tarball without it reproduces the v0.4.1 bug where every rig
// installed a miner that could not mine, so its absence fails the build
// unless explicitly waived (ALLOW_MISSING_CORE=1, for script-only work).
//
// Modes are set, not copied: copyFileSync keeps the source's mode, and the CI
// artifact came out 0666, so v0.4.2-v0.4.5 shipped a world-writable core that
// HiveOS unpacks and a root process loads.
const core = join(dist, 'pearl_core.node');
if (existsSync(core)) {
  copyFileSync(core, join(pkgDir, 'pearl_core.node'));
  chmodSync(join(pkgDir, 'pearl_core.node'), 0o644);
} else if (process.env.ALLOW_MISSING_CORE === '1') {
  console.error('warning: packaging WITHOUT pearl_core.node (ALLOW_MISSING_CORE=1)');
} else {
  console.error('no ' + core + ' — this package could not mine. Run dist:cli with');
  console.error('vendor/native/pearl_core.node staged, or set ALLOW_MISSING_CORE=1.');
  process.exit(1);
}
// The CUDA 13 build rides along when the CLI build staged it. Optional, like
// it is there: without it a Blackwell rig mines on the 12.8 core above.
const cu13 = join(dist, 'pearl_core_cu13.node');
if (existsSync(cu13)) {
  copyFileSync(cu13, join(pkgDir, 'pearl_core_cu13.node'));
  chmodSync(join(pkgDir, 'pearl_core_cu13.node'), 0o644);
} else {
  console.error('warning: no pearl_core_cu13.node — RTX 50 rigs will use the CUDA 12.8 core');
}

// The tarball name carries the version. HiveOS reinstalls only when the
// Installation URL changes, and then reuses any archive of the same file name
// it already downloaded, even one from another URL. So only a new file name
// gets a rig a new build.
//
// The stem before that version must be exactly CUSTOM_NAME. HiveOS splits a
// `<name>-<version>.tar.gz` install URL to derive the miner name, then installs
// into /hive/miners/custom/<name>/ and reads <name>/h-manifest.conf from the
// archive. Naming the file llmjob-earn-hiveos-<version>.tar.gz made it derive
// "llmjob-earn-hiveos", which never matches the llmjob-earn/ directory inside
// the tar, and every install failed with "No llmjob-earn-hiveos/h-manifest.conf".
//
// The unversioned name is kept as a copy for flight sheets that point at
// releases/latest/download/llmjob-earn-hiveos.tar.gz. HiveOS reads "hiveos" as
// its version and "llmjob-earn" as the miner name, so it installs. Because the
// file name never changes, it only helps a first install: a rig that already
// has it says "Already installed" and keeps its old build.
const out = join(dist, name + '-' + version + '.tar.gz');
const legacy = join(dist, 'llmjob-earn-hiveos.tar.gz');
execFileSync('tar', ['-czf', out, '-C', stage, name]);
copyFileSync(out, legacy);
rmSync(stage, { recursive: true, force: true });
console.log('built ' + out + ' (v' + version + ') + legacy ' + legacy);
