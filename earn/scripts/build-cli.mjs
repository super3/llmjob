#!/usr/bin/env node
'use strict';

// Package the headless CLI into a standalone single-file executable using
// Node's built-in Single Executable Applications (SEA). No external base binary
// is downloaded — we copy the running `node`, inject a bundled blob, and the
// result runs on a machine with no Node installed. The CLI then self-updates by
// pulling the newer binary from the GitHub release (see src/cli/selfUpdater.js).

import { build } from 'esbuild';
import { execFileSync } from 'node:child_process';
import { mkdirSync, copyFileSync, writeFileSync, chmodSync, existsSync, readdirSync, rmSync } from 'node:fs';
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
// The CUDA 13 build (sm_120 only; see src/shared/coreVariant.js) travels the
// same way. Optional: without it a Blackwell rig loads the 12.8 core above,
// which is what every rig did before there were two.
const cu13 = join(root, 'vendor', 'native', 'pearl_core_cu13.node');
if (existsSync(cu13)) {
  copyFileSync(cu13, join(dist, 'pearl_core_cu13.node'));
  process.stdout.write('staged pearl_core_cu13.node beside the CLI\n');
} else {
  process.stdout.write('warning: no vendor/native/pearl_core_cu13.node — RTX 50 rigs will use the CUDA 12.8 core\n');
}
// The rotation fold's cubin (earn/native/src/pearl_rot.cu), which the 12.8 and
// CUDA 13 cores load from beside themselves when a rig asks for the fold
// (--rotation-fold). Optional: without it such a rig says the file is missing
// and mines on the release fold.
const rot = join(root, 'vendor', 'native', 'pearl_rot_sm89.cubin');
rmSync(join(dist, 'pearl_rot_sm89.cubin'), { force: true });
if (existsSync(rot)) {
  copyFileSync(rot, join(dist, 'pearl_rot_sm89.cubin'));
  chmodSync(join(dist, 'pearl_rot_sm89.cubin'), 0o644);
  process.stdout.write('staged pearl_rot_sm89.cubin beside the CLI\n');
} else {
  process.stdout.write('warning: no vendor/native/pearl_rot_sm89.cubin — --rotation-fold will mine on the release fold\n');
}
// The same fold's cubin for RTX 30 cards (sm_86), which the cores load on those.
const rot86 = join(root, 'vendor', 'native', 'pearl_rot_sm86.cubin');
rmSync(join(dist, 'pearl_rot_sm86.cubin'), { force: true });
if (existsSync(rot86)) {
  copyFileSync(rot86, join(dist, 'pearl_rot_sm86.cubin'));
  chmodSync(join(dist, 'pearl_rot_sm86.cubin'), 0o644);
  process.stdout.write('staged pearl_rot_sm86.cubin beside the CLI\n');
} else {
  process.stdout.write('warning: no vendor/native/pearl_rot_sm86.cubin — --rotation-fold on an RTX 30 card will mine on the release fold\n');
}
// And for RTX 50 cards (sm_120): the fold compiled from C, which rides in the CUDA 13 core's artifact.
const rot120 = join(root, 'vendor', 'native', 'pearl_rot_sm120.cubin');
rmSync(join(dist, 'pearl_rot_sm120.cubin'), { force: true });
if (existsSync(rot120)) {
  copyFileSync(rot120, join(dist, 'pearl_rot_sm120.cubin'));
  chmodSync(join(dist, 'pearl_rot_sm120.cubin'), 0o644);
  process.stdout.write('staged pearl_rot_sm120.cubin beside the CLI\n');
} else {
  process.stdout.write('warning: no vendor/native/pearl_rot_sm120.cubin — --rotation-fold on an RTX 50 card will mine on the release fold\n');
}
// And the RTX 5090's own (sm_120w: the 96 x 512 block, which only pays on a card its power cap holds). Without it
// a 5090 mines on pearl_rot_sm120.cubin.
const rot120w = join(root, 'vendor', 'native', 'pearl_rot_sm120w.cubin');
rmSync(join(dist, 'pearl_rot_sm120w.cubin'), { force: true });
if (existsSync(rot120w)) {
  copyFileSync(rot120w, join(dist, 'pearl_rot_sm120w.cubin'));
  chmodSync(join(dist, 'pearl_rot_sm120w.cubin'), 0o644);
  process.stdout.write('staged pearl_rot_sm120w.cubin beside the CLI\n');
}
// The AMD core (earn/native/amd, ROCm), loaded with --core amd. Optional the
// same way: without it an AMD rig has nothing to mine with, and every NVIDIA rig
// is unaffected.
const hip = join(root, 'vendor', 'native', 'pearl_core_hip.node');
if (existsSync(hip)) {
  copyFileSync(hip, join(dist, 'pearl_core_hip.node'));
  process.stdout.write('staged pearl_core_hip.node beside the CLI\n');
} else {
  process.stdout.write('warning: no vendor/native/pearl_core_hip.node — this CLI cannot mine on AMD cards\n');
}
// The Intel core (earn/native/sycl), for Intel Arc cards. It is a folder, not
// one file: the addon loads a host library and the SYCL runtime from beside
// itself, so the folder travels whole, as sycl/ beside the binary, where the
// loader looks (src/main/pearlCore.js). The release also gets it as one
// archive, pearl_core_sycl-linux.tar.gz, to unpack beside a binary that was
// downloaded alone. Optional like the AMD core: without it an Arc rig has
// nothing to mine with, and every other rig is unaffected.
const sycl = join(root, 'vendor', 'native', 'sycl');
const syclOut = join(dist, 'sycl');
rmSync(syclOut, { recursive: true, force: true });
rmSync(join(dist, 'pearl_core_sycl-linux.tar.gz'), { force: true });
if (existsSync(join(sycl, 'pearl_core_sycl.node'))) {
  mkdirSync(syclOut);
  for (const f of readdirSync(sycl)) {
    copyFileSync(join(sycl, f), join(syclOut, f));
    // The CI artifact arrives without modes. The check tool is the one program.
    chmodSync(join(syclOut, f), f === 'pearl_sycl_check' ? 0o755 : 0o644);
  }
  execFileSync('tar', ['-czf', join(dist, 'pearl_core_sycl-linux.tar.gz'), '-C', dist, 'sycl']);
  process.stdout.write('staged sycl/ (the Intel core) beside the CLI, and pearl_core_sycl-linux.tar.gz\n');
} else {
  process.stdout.write('warning: no vendor/native/sycl/pearl_core_sycl.node — this CLI cannot mine on Intel Arc cards\n');
}
// The RandomX CPU worker (earn/xmr-worker, built by build-xmr-worker.sh) goes
// beside the binary too: it is an executable of its own, run as a child
// process. Optional: without it --xmr-address says the worker is missing, and
// the GPU miner is unaffected.
const xmrWorker = join(root, 'vendor', 'native', 'llmjob-xmr-worker');
if (existsSync(xmrWorker)) {
  copyFileSync(xmrWorker, join(dist, 'llmjob-xmr-worker'));
  chmodSync(join(dist, 'llmjob-xmr-worker'), 0o755);
  process.stdout.write('staged llmjob-xmr-worker beside the CLI\n');
} else {
  process.stdout.write('warning: no vendor/native/llmjob-xmr-worker — this CLI cannot mine Monero on the CPU\n');
}
process.stdout.write('built ' + out + '\n');
