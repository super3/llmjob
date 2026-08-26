#!/usr/bin/env node
'use strict';

// Package the HiveOS custom-miner archive: the standalone CLI binary plus the
// hiveos/ hook scripts, tarred as dist/llmjob-earn-hiveos.tar.gz — the file a
// flight sheet's "Installation URL" points at. Run `npm run dist:cli` first
// (or pass an explicit binary path as the first argument).

import { execFileSync } from 'node:child_process';
import { mkdirSync, copyFileSync, readFileSync, writeFileSync, chmodSync, rmSync, existsSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const dist = join(root, 'dist');
const bin = process.argv[2] ? resolve(process.argv[2]) : join(dist, 'llmjob-earn-cli-linux');
const version = JSON.parse(readFileSync(join(root, 'package.json'), 'utf8')).version;

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

copyFileSync(bin, join(pkgDir, 'llmjob-earn-cli-linux'));
chmodSync(join(pkgDir, 'llmjob-earn-cli-linux'), 0o755);

// The native core sits beside the binary — the loader's first packaged-CLI
// candidate. A tarball without it reproduces the v0.4.1 bug where every rig
// installed a miner that could not mine, so its absence fails the build
// unless explicitly waived (ALLOW_MISSING_CORE=1, for script-only work).
const core = join(dist, 'pearl_core.node');
if (existsSync(core)) {
  copyFileSync(core, join(pkgDir, 'pearl_core.node'));
} else if (process.env.ALLOW_MISSING_CORE === '1') {
  console.error('warning: packaging WITHOUT pearl_core.node (ALLOW_MISSING_CORE=1)');
} else {
  console.error('no ' + core + ' — this package could not mine. Run dist:cli with');
  console.error('vendor/native/pearl_core.node staged, or set ALLOW_MISSING_CORE=1.');
  process.exit(1);
}

// The tarball name carries the version: HiveOS rigs cache the download and can
// skip re-fetching a URL whose filename hasn't changed, leaving them stuck on an
// old build after a release. A per-release filename makes every update a fresh
// download.
//
// The stem before that version must be exactly CUSTOM_NAME. HiveOS splits a
// `<name>-<version>.tar.gz` install URL to derive the miner name, then installs
// into /hive/miners/custom/<name>/ and reads <name>/h-manifest.conf from the
// archive. Naming the file llmjob-earn-hiveos-<version>.tar.gz made it derive
// "llmjob-earn-hiveos", which never matches the llmjob-earn/ directory inside
// the tar, and every install failed with "No llmjob-earn-hiveos/h-manifest.conf".
//
// The unversioned name is kept as a copy so flight sheets that still point at
// releases/latest/…/llmjob-earn-hiveos.tar.gz keep installing — with no
// -<version> suffix to split on, HiveOS falls back to the directory in the tar.
const out = join(dist, name + '-' + version + '.tar.gz');
const legacy = join(dist, 'llmjob-earn-hiveos.tar.gz');
execFileSync('tar', ['-czf', out, '-C', stage, name]);
copyFileSync(out, legacy);
rmSync(stage, { recursive: true, force: true });
console.log('built ' + out + ' (v' + version + ') + legacy ' + legacy);
