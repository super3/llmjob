#!/usr/bin/env node
'use strict';

// Package the headless CLI into a standalone single-file executable using
// Node's built-in Single Executable Applications (SEA). No external base binary
// is downloaded — we copy the running `node`, inject a bundled blob, and the
// result runs on a machine with no Node installed. The CLI then self-updates by
// pulling the newer binary from the GitHub release (see src/cli/selfUpdater.js).

import { build } from 'esbuild';
import { execFileSync } from 'node:child_process';
import { mkdirSync, copyFileSync, writeFileSync, chmodSync, existsSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const dist = join(root, 'dist');
const bundle = join(dist, 'cli-bundle.cjs');
const blob = join(dist, 'cli.blob');
const out = process.argv[2] ? resolve(process.argv[2]) : join(dist, 'llmjob-earn-cli-linux');

mkdirSync(dist, { recursive: true });

// 1) Bundle the CLI (+ its local requires and package.json) into one CJS file.
await build({
  entryPoints: [join(root, 'src/cli/sea-entry.js')],
  bundle: true,
  platform: 'node',
  target: 'node22',
  format: 'cjs',
  outfile: bundle,
  legalComments: 'none',
});

// 2) Generate the SEA blob from that bundle.
const seaConfig = join(dist, 'sea-config.json');
writeFileSync(seaConfig, JSON.stringify({ main: bundle, output: blob, disableExperimentalSEAWarning: true }));
execFileSync(process.execPath, ['--experimental-sea-config', seaConfig], { stdio: 'inherit' });

// 3) Copy the running node and inject the blob into it.
copyFileSync(process.execPath, out);
chmodSync(out, 0o755);
execFileSync('npx', [
  '--yes', 'postject', out, 'NODE_SEA_BLOB', blob,
  '--sentinel-fuse', 'NODE_SEA_FUSE_fce680ab2cc467b6e072b8b5df1996b2',
], { stdio: 'inherit' });

// The native core travels BESIDE the binary, never inside it: a .node cannot
// be dlopen'd out of a SEA snapshot, so bundling it would only hide it. CI
// stages it at vendor/native before this runs; a dev box without a CUDA build
// still gets a CLI (which will say, loudly and non-zero, that it cannot mine).
const core = join(root, 'vendor', 'native', 'pearl_core.node');
if (existsSync(core)) {
  copyFileSync(core, join(dist, 'pearl_core.node'));
  process.stdout.write('staged pearl_core.node beside the CLI\n');
} else {
  process.stdout.write('warning: no vendor/native/pearl_core.node — this CLI cannot mine\n');
}
process.stdout.write('built ' + out + '\n');
