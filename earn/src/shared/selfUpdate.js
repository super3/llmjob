'use strict';

// Pure self-update logic for the headless CLI. The CLI ships to the GitHub
// Releases page as a standalone Linux executable (llmjob-earn-cli-linux); this
// module decides — from the running version and the GitHub "latest release" API
// response — whether a newer build exists and where to download it. All IO
// (HTTPS, filesystem, re-exec) lives in the shell (src/cli/selfUpdater.js) so
// this stays pure and fully unit-tested.

const { packageAssetName } = require('./hiveos');

const REPO = 'super3/llmjob';
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
// Missing / non-numeric components count as 0, so it never throws on odd input.
function compareVersions(a, b) {
  const pa = normalizeVersion(a).split('.');
  const pb = normalizeVersion(b).split('.');
  const len = Math.max(pa.length, pb.length);
  for (let i = 0; i < len; i++) {
    const x = parseInt(pa[i], 10) || 0;
    const y = parseInt(pb[i], 10) || 0;
    if (x > y) return 1;
    if (x < y) return -1;
  }
  return 0;
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
      xmrWorkerUrl: (same && rel.assets['llmjob-xmr-worker']) || null,
      coreHipUrl: (same && rel.assets['pearl_core_hip.node']) || null,
    });
  }

  const downloadUrl = rel.assets[assetName] || null;
  if (!downloadUrl) return Object.assign(base, { reason: 'asset-missing' });

  // The native core ships as its own asset beside the CLI. It rides on the
  // plan so the updater can keep the pair in step: a new CLI driving an old
  // core is a silent version skew. Absent from older releases, hence nullable.
  const coreUrl = rel.assets['pearl_core.node'] || null;
  // The CUDA 13 build rides the same way (see shared/coreVariant). Null when a
  // release does not carry it -- older ones, or one whose CUDA 13 build failed
  // in CI -- and the updater then REMOVES any copy beside the binary, because a
  // stale one would be loaded ahead of the fresh 12.8 core on a 5090.
  const coreCu13Url = rel.assets['pearl_core_cu13.node'] || null;
  // The AMD core (pearl_core_hip.node, Linux releases from v0.6.8). Null on a
  // release without it; the updater then keeps the copy beside the binary,
  // since on an AMD rig it is the only core that can mine.
  const coreHipUrl = rel.assets['pearl_core_hip.node'] || null;
  // The RandomX CPU worker (earn/xmr-worker) rides along too. Null on a
  // release without it, and then the copy beside the binary stays: the worker
  // states its protocol version when it starts, so an old one that no longer
  // fits is refused there, not run.
  const xmrWorkerUrl = rel.assets['llmjob-xmr-worker'] || null;
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
    coreCu13Url,
    coreHipUrl,
    xmrWorkerUrl,
    packageUrl,
  };
}

module.exports = {
  REPO,
  LATEST_RELEASE_API,
  assetNameFor,
  normalizeVersion,
  compareVersions,
  isNewer,
  parseRelease,
  planUpdate,
};
