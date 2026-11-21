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
import { mkdirSync, copyFileSync, readFileSync, writeFileSync, chmodSync, rmSync, existsSync, readdirSync } from 'node:fs';
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

// The rotation fold's cubin, for --rotation-fold in Extra config: the CUDA
// cores load it from beside themselves. Optional like the CUDA 13 core:
// without it a rig that asks for the fold mines on the release fold.
const rot = join(dist, 'pearl_rot_sm89.cubin');
if (existsSync(rot)) {
  copyFileSync(rot, join(pkgDir, 'pearl_rot_sm89.cubin'));
  chmodSync(join(pkgDir, 'pearl_rot_sm89.cubin'), 0o644);
} else {
  console.error('warning: no pearl_rot_sm89.cubin — --rotation-fold will mine on the release fold');
}
// The same fold's cubin for RTX 30 cards.
const rot86 = join(dist, 'pearl_rot_sm86.cubin');
if (existsSync(rot86)) {
  copyFileSync(rot86, join(pkgDir, 'pearl_rot_sm86.cubin'));
  chmodSync(join(pkgDir, 'pearl_rot_sm86.cubin'), 0o644);
} else {
  console.error('warning: no pearl_rot_sm86.cubin — --rotation-fold on an RTX 30 card will mine on the release fold');
}
// And for RTX 50 cards.
const rot120 = join(dist, 'pearl_rot_sm120.cubin');
if (existsSync(rot120)) {
  copyFileSync(rot120, join(pkgDir, 'pearl_rot_sm120.cubin'));
  chmodSync(join(pkgDir, 'pearl_rot_sm120.cubin'), 0o644);
} else {
  console.error('warning: no pearl_rot_sm120.cubin — --rotation-fold on an RTX 50 card will mine on the release fold');
}
// And the RTX 5090's own (without it a 5090 mines on pearl_rot_sm120.cubin).
const rot120w = join(dist, 'pearl_rot_sm120w.cubin');
if (existsSync(rot120w)) {
  copyFileSync(rot120w, join(pkgDir, 'pearl_rot_sm120w.cubin'));
  chmodSync(join(pkgDir, 'pearl_rot_sm120w.cubin'), 0o644);
}

// The AMD core, for --core amd in Extra config. Optional like the CUDA 13
// core: without it the package mines on NVIDIA cards as before.
const hip = join(dist, 'pearl_core_hip.node');
if (existsSync(hip)) {
  copyFileSync(hip, join(pkgDir, 'pearl_core_hip.node'));
  chmodSync(join(pkgDir, 'pearl_core_hip.node'), 0o644);
} else {
  console.error('warning: no pearl_core_hip.node — this package cannot mine on AMD cards');
}

// The ROCm 7 HIP runtime that core loads, in rocm/ beside it (scripts/fetch-hip-runtime.sh),
// since HiveOS has none and its Ubuntu 20.04 image can't install ROCm 7. The core looks in
// /opt/rocm/lib first, so a rig that installed ROCm uses its own. Optional: without it the AMD
// core needs the rig to install the runtime, as before 9 November.
const hipRuntime = join(root, 'vendor', 'hip-runtime');
if (existsSync(hip) && existsSync(join(hipRuntime, 'libamdhip64.so.7'))) {
  const rocmDir = join(pkgDir, 'rocm');
  mkdirSync(rocmDir);
  for (const f of readdirSync(hipRuntime)) {
    copyFileSync(join(hipRuntime, f), join(rocmDir, f));
    chmodSync(join(rocmDir, f), 0o644);
  }
} else if (existsSync(hip)) {
  console.error('warning: no vendor/hip-runtime — AMD rigs need ROCm 7 installed to mine');
}

// The Intel core, for a rig of Intel Arc cards: sycl/ beside the binary, the
// addon with the host library and SYCL runtime it loads (dist:cli stages it).
// The CLI finds the Arc cards itself, so a flight sheet needs no --core. It
// needs only Intel's OpenCL or Level Zero driver on the rig, which HiveOS's
// Intel image has. Optional like the AMD core.
const sycl = join(dist, 'sycl');
if (existsSync(join(sycl, 'pearl_core_sycl.node'))) {
  const syclDir = join(pkgDir, 'sycl');
  mkdirSync(syclDir);
  for (const f of readdirSync(sycl)) {
    copyFileSync(join(sycl, f), join(syclDir, f));
    chmodSync(join(syclDir, f), f === 'pearl_sycl_check' ? 0o755 : 0o644);
  }
} else {
  console.error('warning: no sycl/pearl_core_sycl.node — this package cannot mine on Intel Arc cards');
}

// The RandomX CPU worker, for --xmr-address in Extra config. Optional like the
// CUDA 13 core: without it the rig mines Pearl as before.
const xmrWorker = join(dist, 'llmjob-xmr-worker');
if (existsSync(xmrWorker)) {
  copyFileSync(xmrWorker, join(pkgDir, 'llmjob-xmr-worker'));
  chmodSync(join(pkgDir, 'llmjob-xmr-worker'), 0o755);
} else {
  console.error('warning: no llmjob-xmr-worker — this package cannot mine Monero on the CPU');
}

// The tarball name carries the version. HiveOS reinstalls only when the
// Installation URL changes, and then reuses any archive of the same file name
// it already downloaded, even one from another URL. So only a new file name
// gets a rig a new build from HiveOS. The CLI updates the installed package
// itself from this versioned file (src/cli/hiveos.js).
//
// The stem before that version must be exactly CUSTOM_NAME. HiveOS splits a
// `<name>-<version>.tar.gz` install URL to derive the miner name, then installs
// into /hive/miners/custom/<name>/ and reads <name>/h-manifest.conf from the
// archive. Naming the file llmjob-earn-hiveos-<version>.tar.gz made it derive
// "llmjob-earn-hiveos", which never matches the llmjob-earn/ directory inside
// the tar, and every install failed with "No llmjob-earn-hiveos/h-manifest.conf".
//
// The unversioned copy is the one flight sheets use, at
// releases/latest/download/llmjob-earn-hiveos.tar.gz. HiveOS reads "hiveos" as
// its version and "llmjob-earn" as the miner name, so it installs. Its file
// name never changes, so HiveOS installs it once and says "Already installed"
// after that, which is what keeps it from undoing the CLI's own updates.
const out = join(dist, name + '-' + version + '.tar.gz');
const legacy = join(dist, 'llmjob-earn-hiveos.tar.gz');
execFileSync('tar', ['-czf', out, '-C', stage, name]);
copyFileSync(out, legacy);
rmSync(stage, { recursive: true, force: true });
console.log('built ' + out + ' (v' + version + ') + legacy ' + legacy);
