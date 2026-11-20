'use strict';

// Pure self-update logic for the headless CLI. The CLI ships to the GitHub
// Releases page as a standalone Linux executable (llmjob-earn-cli-linux); this
// module decides — from the running version and the GitHub "latest release" API
// response — whether a newer build exists and where to download it. All IO
// (HTTPS, filesystem, re-exec) lives in the shell (src/cli/selfUpdater.js) so
// this stays pure and fully unit-tested.

const { packageAssetName } = require('./hiveos');
const { CU12, CU13, AMD, FILES } = require('./coreVariant');
const version = require('./version');

const REPO = 'super3/llmjob';

// The Monero CPU worker's release asset, and its file beside the binary.
const XMR_WORKER = 'llmjob-xmr-worker';

// The files that belong to a release's CUDA cores but that it may not carry,
// as { key, file }: the plan's field for the file's URL, and the asset name,
// which is also its file name beside the binary. The updater handles them all
// one way (cli/selfUpdater): the old copy is removed before the update goes in,
// and a failed download leaves no file. A stale one would be loaded beside the
// new core: the CUDA 13 core ahead of the fresh 12.8 one on a 5090, a cubin
// as the new core's rotation fold or wide-tile fold. Without the file the
// loader says "not installed" and mines on what it has.
const OPTIONAL_ASSETS = [
  // The CUDA 13 build (shared/coreVariant). Missing from older releases and from
  // one whose CUDA 13 build failed in CI.
  { key: 'coreCu13Url', file: FILES[CU13] },
  // The rotation fold's cubins, which the CUDA cores load from beside
  // themselves when the fold is on (the default; --no-rotation-fold): RTX 40
  // (sm_89), RTX 30 (sm_86), RTX 50 (sm_120), and the RTX 5090's own, which
  // the cores take on a 5090 when it is there.
  { key: 'rotCubinUrl', file: 'pearl_rot_sm89.cubin' },
  { key: 'rotCubin86Url', file: 'pearl_rot_sm86.cubin' },
  { key: 'rotCubin120Url', file: 'pearl_rot_sm120.cubin' },
  { key: 'rotCubin120wUrl', file: 'pearl_rot_sm120w.cubin' },
  // RTX 20's wide-tile fold, which the cores load on RTX 20 cards when it is
  // there. It belongs to its release's core, as the rotation cubins do.
  { key: 't96CubinUrl', file: 'pearl_t96_sm75.cubin' },
];
const LATEST_RELEASE_API = 'https://api.github.com/repos/' + REPO + '/releases/latest';

// The published release-asset name for a platform. Only a Linux x64 binary is
// built today; other platforms have no standalone binary to self-update.
function assetNameFor(platform) {
  return platform === 'linux' ? 'llmjob-earn-cli-linux' : null;
}

// Drop a leading `v` and surrounding whitespace: `v0.1.11` -> `0.1.11`.
function normalizeVersion(v) {
  return String(v == null ? '' : v).trim().replace(/^v/i, '');
}

// Compare dotted numeric versions. Returns 1 if a>b, -1 if a<b, 0 if equal.
// Missing / non-numeric components count as 0, so it never throws on odd input
// (shared/version, not strict).
function compareVersions(a, b) {
  return version.compareVersions(a, b);
}

function isNewer(current, latest) {
  return compareVersions(latest, current) > 0;
}

// Reduce a GitHub "latest release" API object to what we need: the version and
// a { assetName -> downloadUrl } map.
function parseRelease(json) {
  const r = json || {};
  const version = normalizeVersion(r.tag_name || r.name || '');
  const assets = {};
  const list = Array.isArray(r.assets) ? r.assets : [];
  for (const a of list) {
    if (a && a.name) assets[a.name] = a.browser_download_url || null;
  }
  return { version, assets };
}

// Decide whether to update. Returns a plan describing the outcome; `reason` is
// one of: unsupported-platform, no-release, up-to-date, asset-missing,
// update-available. Only `update-available` carries a downloadUrl.
// `up-to-date` carries xmrWorkerUrl and coreHipUrl when this very release is
// the latest, so a CLI installed without the CPU worker or the AMD core, or
// whose download of one failed, can fetch it without waiting for the next
// release.
function planUpdate(opts) {
  const o = opts || {};
  const currentVersion = normalizeVersion(o.currentVersion);
  const rel = o.release || { version: '', assets: {} };
  const assetName = assetNameFor(o.platform);
  const base = { updateAvailable: false, currentVersion, latestVersion: rel.version };

  if (!assetName) return Object.assign(base, { reason: 'unsupported-platform' });
  if (!rel.version) return Object.assign(base, { reason: 'no-release' });
  if (!isNewer(currentVersion, rel.version)) {
    const same = compareVersions(currentVersion, rel.version) === 0;
    return Object.assign(base, {
      reason: 'up-to-date',
      xmrWorkerUrl: (same && rel.assets[XMR_WORKER]) || null,
      coreHipUrl: (same && rel.assets[FILES[AMD]]) || null,
    });
  }

  const downloadUrl = rel.assets[assetName] || null;
  if (!downloadUrl) return Object.assign(base, { reason: 'asset-missing' });

  // The native core ships as its own asset beside the CLI. It rides on the
  // plan so the updater can keep the pair in step: a new CLI driving an old
  // core is a silent version skew. Absent from older releases, hence nullable.
  const coreUrl = rel.assets[FILES[CU12]] || null;
  // The optional files (OPTIONAL_ASSETS), each null when the release has none.
  const optional = {};
  for (const { key, file } of OPTIONAL_ASSETS) optional[key] = rel.assets[file] || null;
  // The AMD core (pearl_core_hip.node, Linux releases from v0.6.8). Null on a
  // release without it; the updater then keeps the copy beside the binary,
  // since on an AMD rig it is the only core that can mine.
  const coreHipUrl = rel.assets[FILES[AMD]] || null;
  // The RandomX CPU worker (earn/xmr-worker) rides along too. Null on a
  // release without it, and then the copy beside the binary stays: the worker
  // states its protocol version when it starts, so an old one that no longer
  // fits is refused there, not run.
  const xmrWorkerUrl = rel.assets[XMR_WORKER] || null;
  // The HiveOS package, which the CLI updates whole under HiveOS: binary, cores
  // and hook scripts (shared/hiveos). Null until the release has it; the
  // package goes up after the binary.
  const packageUrl = rel.assets[packageAssetName(rel.version)] || null;

  return {
    updateAvailable: true,
    reason: 'update-available',
    currentVersion,
    latestVersion: rel.version,
    assetName,
    downloadUrl,
    coreUrl,
    ...optional,
    coreHipUrl,
    xmrWorkerUrl,
    packageUrl,
  };
}

module.exports = {
  LATEST_RELEASE_API,
  OPTIONAL_ASSETS,
  XMR_WORKER,
  assetNameFor,
  normalizeVersion,
  compareVersions,
  isNewer,
  parseRelease,
  planUpdate,
};
