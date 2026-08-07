'use strict';

const path = require('path');
const {
  enginePath, engineFiles, engineDownloadUrl, engineArchiveName, isZipUrl, manualEnginePath,
  enginePackage,
} = require('../shared/engine');

// Ensures the alpha-miner engine is present, downloading and installing it on
// demand. All IO (filesystem, network download, zip extraction, chmod) is
// injected so the orchestration is fully unit-testable; main.js wires the real
// implementations.
class EngineManager {
  constructor({ dir, platform, gpu, version, urlBase, fs, download, extract, extractPackage, chmod } = {}) {
    this.dir = dir;
    this.platform = platform;
    this.gpu = gpu;
    this.version = version;
    this.urlBase = urlBase;
    this.fs = fs;
    this.download = download;
    this.extract = extract;
    this.extractPackage = extractPackage;
    this.chmod = chmod;
  }

  binaryPath() {
    return enginePath(this.dir, this.platform, this.gpu, this.version);
  }

  // A packaged engine counts as installed only when its core is there too — see
  // engineFiles. An interrupted extract leaves the launcher looking perfectly
  // installed, and nothing else would ever re-download it.
  isInstalled() {
    return engineFiles(this.dir, this.platform, this.gpu, this.version)
      .every((p) => this.fs.existsSync(p));
  }

  // Resolve to the engine path, downloading + installing it if missing.
  async ensure(onProgress) {
    const dest = this.binaryPath();
    if (this.isInstalled()) {
      // A cached binary can be present but lack the execute bit: the download
      // writes it 0o644 and only chmods afterwards, so an interrupted first run
      // (crash/kill/reboot between the rename and the chmod) — or a binary put
      // there by any other path — leaves a non-executable file that spawns with
      // EACCES forever, since this early return used to skip the chmod below.
      // Re-assert +x here so a stuck rig self-heals on the next start. Best
      // effort: a chmod failure (read-only dir, foreign owner) must not turn a
      // rig whose binary is already executable into a crash.
      this.ensureExecutable(dest);
      return dest;
    }

    this.fs.mkdirSync(this.dir, { recursive: true });

    // A packaged engine (1.9.1b's launcher + core tarball) installs as a tree,
    // not a file: extract it whole and make BOTH halves executable. The tarball
    // ships the core mode 644 while the launcher requires -x on it, so chmodding
    // only the launcher — the single-binary assumption everywhere else — leaves
    // every start failing with "core missing or not executable".
    const pkg = enginePackage(this.platform, this.version);
    if (pkg) {
      const archivePath = path.join(this.dir, pkg.archive);
      await this.download(engineDownloadUrl(this.platform, this.gpu, this.urlBase, this.version), archivePath, onProgress);
      await this.extractPackage(archivePath, this.dir);
      try { this.fs.unlinkSync(archivePath); } catch (e) { /* leftover archive is harmless */ }
      this.chmod(dest, 0o755);
      this.chmod(path.join(this.dir, pkg.dir, pkg.core), 0o755);
      return dest;
    }

    // A file the user downloaded by hand counts as installed — checked before we
    // go anywhere near the network, so a rig whose HTTPS is broken can be fixed
    // with a browser.
    if (!(await this.adoptManualDownload(dest))) {
      const url = engineDownloadUrl(this.platform, this.gpu, this.urlBase, this.version);

      if (isZipUrl(url)) {
        const zipPath = path.join(this.dir, 'engine.zip');
        await this.download(url, zipPath, onProgress);
        await this.extract(zipPath, dest);
        this.fs.unlinkSync(zipPath);
      } else {
        await this.download(url, dest, onProgress);
      }
    }

    if (this.platform !== 'win32') this.chmod(dest, 0o755);
    return dest;
  }

  // Install a hand-downloaded engine sitting in the engine dir, if there is one.
  // Returns whether it adopted something.
  //
  // Users hit by a failed download fetch the engine in a browser, which saves it
  // under the pool's own name — the unversioned `alpha-miner`, or the Windows
  // zip — never the versioned name the cache looks for. Without this the app
  // ignores the file, retries the download that already failed, and the user
  // reasonably reports that downloading it manually "changed nothing". The
  // archive is extracted (leaving the user's zip alone); a bare binary is
  // renamed into place, which is the install.
  async adoptManualDownload(dest) {
    const candidates = [
      manualEnginePath(this.dir, this.platform),
      path.join(this.dir, engineArchiveName(this.platform, this.gpu, this.version)),
    ];
    for (const src of candidates) {
      if (src === dest || !this.fs.existsSync(src)) continue;
      if (isZipUrl(src)) await this.extract(src, dest);
      else this.fs.renameSync(src, dest);
      return true;
    }
    return false;
  }

  // Grant the engine its execute bit (non-Windows only). Best effort by design:
  // this runs on the already-installed path where the file may sit on a
  // read-only mount or be owned by another user, and a throw there would break
  // a rig whose binary is already executable. The fresh-download path keeps its
  // own strict chmod so a genuine install failure still surfaces.
  ensureExecutable(dest) {
    if (this.platform === 'win32') return;
    try {
      this.chmod(dest, 0o755);
      // A packaged engine needs its core executable too — the launcher checks
      // -x on it and refuses to start otherwise, so a half-chmodded cache would
      // never self-heal without this.
      const pkg = enginePackage(this.platform, this.version);
      if (pkg) this.chmod(path.join(this.dir, pkg.dir, pkg.core), 0o755);
    } catch (e) {
      /* best effort — spawn will report EACCES if it truly isn't executable */
    }
  }
}

module.exports = { EngineManager };
