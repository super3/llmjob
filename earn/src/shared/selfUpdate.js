'use strict';

// Pure self-update logic for the headless CLI. The CLI ships to the GitHub
// Releases page as a standalone Linux executable (llmjob-earn-cli-linux); this
// module decides — from the running version and the GitHub "latest release" API
// response — whether a newer build exists and where to download it. All IO
// (HTTPS, filesystem, re-exec) lives in the shell (src/cli/selfUpdater.js) so
// this stays pure and fully unit-tested.

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
function planUpdate(opts) {
  const o = opts || {};
  const currentVersion = normalizeVersion(o.currentVersion);
  const rel = o.release || { version: '', assets: {} };
  const assetName = assetNameFor(o.platform);
  const base = { updateAvailable: false, currentVersion, latestVersion: rel.version };

  if (!assetName) return Object.assign(base, { reason: 'unsupported-platform' });
  if (!rel.version) return Object.assign(base, { reason: 'no-release' });
  if (!isNewer(currentVersion, rel.version)) return Object.assign(base, { reason: 'up-to-date' });

  const downloadUrl = rel.assets[assetName] || null;
  if (!downloadUrl) return Object.assign(base, { reason: 'asset-missing' });

  return {
    updateAvailable: true,
    reason: 'update-available',
    currentVersion,
    latestVersion: rel.version,
    assetName,
    downloadUrl,
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
