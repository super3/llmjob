'use strict';

const path = require('path');
const {
  enginePath, engineFiles, engineDownloadUrl, engineArchiveName, engineArchiveLauncher,
  manualEnginePath, enginePackage,
} = require('../shared/engine');

// Ensures the PeakMiner engine is present, downloading and installing it on
// demand. All IO (filesystem, network download, zip extraction, chmod) is
// injected so the orchestration is fully unit-testable; main.js wires the real
// implementations.
class EngineManager {
  constructor({ dir, platform, gpu, version, urlBase, fs, download, extractPackage, chmod } = {}) {
    this.dir = dir;
    this.platform = platform;
    this.gpu = gpu;
    this.version = version;
    this.urlBase = urlBase;
    this.fs = fs;
    this.download = download;
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

    // A described engine installs from its descriptor rather than by convention.
    // Two shapes today: an archive to unpack (Windows' zip), and a bare binary
    // to save as-is (the Linux build, which upstream publishes unarchived).
    const pkg = enginePackage(this.platform, this.version);
    if (pkg) {
      const url = engineDownloadUrl(this.platform, this.gpu, this.urlBase, this.version);
      if (pkg.saveAsIs) {
        // Nothing to unpack. Handing a plain ELF to the extractor would fail on
        // a file that is not an archive, so it goes straight to its final path.
        await this.download(url, dest, onProgress);
      } else {
        const archivePath = path.join(this.dir, pkg.archive);
        await this.download(url, archivePath, onProgress);
        await this.extractPackage(archivePath, this.dir);
        try { this.fs.unlinkSync(archivePath); } catch (e) { /* leftover archive is harmless */ }

        // The archive's own launcher name is not the name we cache it under:
        // upstream ships an unversioned `peakminer.exe`, and leaving it there
        // would make every future version bump a cache HIT — the next release
        // would find this exe at the expected path and never download. Rename it
        // so the cached name carries the version.
        //
        // Unconditional, with no null guard and no same-name guard: every
        // archive descriptor MUST declare an archiveLauncher that differs from
        // its launcher, and engine.test.js asserts both invariants directly.
        // Runtime checks here would be branches that can only ever be dead.
        this.fs.renameSync(
          path.join(this.dir, engineArchiveLauncher(this.platform, this.version)), dest);
      }
      // One rule for every described engine, whatever its shape: Windows has no
      // execute bit to grant, and chmodding there is at best a no-op, at worst a
      // throw that fails an otherwise perfect install.
      if (this.platform !== 'win32') this.chmod(dest, 0o755);
      return dest;
    }

    // A file the user downloaded by hand counts as installed — checked before we
    // go anywhere near the network, so a rig whose HTTPS is broken can be fixed
    // with a browser.
    if (!(await this.adoptManualDownload(dest))) {
      // No archive handling here. This branch runs only for a version no
      // descriptor claims, and every artifact named that way is a plain binary —
      // all the archive shapes live in PACKAGED and return above.
      await this.download(
        engineDownloadUrl(this.platform, this.gpu, this.urlBase, this.version), dest, onProgress);
    }

    if (this.platform !== 'win32') this.chmod(dest, 0o755);
    return dest;
  }

  // Install a hand-downloaded engine sitting in the engine dir, if there is one.
  // Returns whether it adopted something.
  //
  // Users hit by a failed download fetch the engine in a browser, which saves it
  // under upstream's own name — the unversioned `peakminer` — never the
  // versioned name the cache looks for. Without this the app ignores the file,
  // retries the download that already failed, and the user reasonably reports
  // that downloading it manually "changed nothing". Renaming it into place IS
  // the install. This path matters more now that an antivirus is a realistic
  // reason for the in-app download to fail, not just a broken proxy.
  async adoptManualDownload(dest) {
    const candidates = [
      manualEnginePath(this.dir, this.platform),
      path.join(this.dir, engineArchiveName(this.platform, this.gpu, this.version)),
    ];
    for (const src of candidates) {
      if (src === dest || !this.fs.existsSync(src)) continue;
      this.fs.renameSync(src, dest);
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
    } catch (e) {
      /* best effort — spawn will report EACCES if it truly isn't executable */
    }
  }
}

module.exports = { EngineManager };
