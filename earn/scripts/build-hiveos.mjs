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

// Stage <dist>/hiveos-stage/llmjob-earn/ — the directory name inside the tar
// must match CUSTOM_NAME for the HiveOS installer to place it correctly.
const stage = join(dist, 'hiveos-stage');
const pkgDir = join(stage, 'llmjob-earn');
rmSync(stage, { recursive: true, force: true });
mkdirSync(pkgDir, { recursive: true });

for (const f of ['h-config.sh', 'h-run.sh', 'h-stats.sh']) {
  copyFileSync(join(root, 'hiveos', f), join(pkgDir, f));
  chmodSync(join(pkgDir, f), 0o755);
}

// Stamp the package version into the manifest.
const manifest = readFileSync(join(root, 'hiveos', 'h-manifest.conf'), 'utf8')
  .replace(/^CUSTOM_VERSION=.*$/m, 'CUSTOM_VERSION=' + version);
writeFileSync(join(pkgDir, 'h-manifest.conf'), manifest);

copyFileSync(bin, join(pkgDir, 'llmjob-earn-cli-linux'));
chmodSync(join(pkgDir, 'llmjob-earn-cli-linux'), 0o755);

// The tarball name carries the version: HiveOS rigs cache the download and can
// skip re-fetching a URL whose filename hasn't changed, leaving them stuck on an
// old build after a release. A per-release filename makes every update a fresh
// download. The unversioned name is kept as a copy so flight sheets that still
// point at releases/latest/…/llmjob-earn-hiveos.tar.gz keep installing.
const out = join(dist, 'llmjob-earn-hiveos-' + version + '.tar.gz');
const legacy = join(dist, 'llmjob-earn-hiveos.tar.gz');
execFileSync('tar', ['-czf', out, '-C', stage, 'llmjob-earn']);
copyFileSync(out, legacy);
rmSync(stage, { recursive: true, force: true });
console.log('built ' + out + ' (v' + version + ') + legacy ' + legacy);
