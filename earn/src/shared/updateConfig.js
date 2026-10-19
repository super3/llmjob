'use strict';

// The updater's settings file. electron-builder writes app-update.yml into the
// app's resources folder, and electron-updater reads it before every check. A
// crash or power cut while an update installs can leave that file full of NUL
// bytes. Every check then fails with "null byte is not allowed in input", and
// the app never updates again short of a reinstall. This keeps a copy of what the
// build writes, so the app can use that instead.

// What electron-builder writes for this app: build.publish from package.json,
// then updaterCacheDirName, the folder downloads are kept in (the package name
// plus "-updater"). No publisherName: the Windows build is not code-signed, so
// the build writes none and electron-updater checks no signature. If the build
// is ever signed, add publisherName here too, or a repaired install would skip
// that check. test/updateConfig.test.js holds this to package.json.
const UPDATE_CONFIG = [
  'owner: super3',
  'repo: llmjob',
  'provider: github',
  'releaseType: release',
  'updaterCacheDirName: llmjob-earn-updater',
  '',
].join('\n');

// Whether electron-updater can use this text. Not a YAML parse: js-yaml comes
// with electron-updater and is not ours to require. It catches what a broken
// write leaves behind (nothing, or NUL bytes, which YAML refuses anywhere) and a
// file without the one key every config has.
function usableUpdateConfig(text) {
  return typeof text === 'string' && !text.includes('\0') && /^provider:[ \t]*\S/m.test(text);
}

// Which config file electron-updater should read. null: the one in resources/ is
// fine, or there is no resources folder (not running under Electron). Otherwise
// writes the copy above to the user's data folder and returns its path. A failed
// write throws.
function repairUpdateConfig({ resourcesPath, userDataPath, fs, join }) {
  if (!resourcesPath) return null;
  let text = null;
  try {
    text = fs.readFileSync(join(resourcesPath, 'app-update.yml'), 'utf8');
  } catch (e) { /* missing or unreadable: repaired below */ }
  if (usableUpdateConfig(text)) return null;
  const file = join(userDataPath, 'app-update.yml');
  fs.writeFileSync(file, UPDATE_CONFIG);
  return file;
}

module.exports = { UPDATE_CONFIG, usableUpdateConfig, repairUpdateConfig };
