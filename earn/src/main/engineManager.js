'use strict';

const path = require('path');
const { enginePath, engineDownloadUrl, isZipUrl } = require('../shared/engine');

// Ensures the alpha-miner engine is present, downloading and installing it on
// demand. All IO (filesystem, network download, zip extraction, chmod) is
// injected so the orchestration is fully unit-testable; main.js wires the real
// implementations.
class EngineManager {
  constructor({ dir, platform, gpu, version, urlBase, fs, download, extract, chmod } = {}) {
    this.dir = dir;
    this.platform = platform;
    this.gpu = gpu;
    this.version = version;
    this.urlBase = urlBase;
    this.fs = fs;
    this.download = download;
    this.extract = extract;
    this.chmod = chmod;
  }

  binaryPath() {
    return enginePath(this.dir, this.platform, this.gpu, this.version);
  }

  isInstalled() {
    return this.fs.existsSync(this.binaryPath());
  }

  // Resolve to the engine path, downloading + installing it if missing.
  async ensure(onProgress) {
    const dest = this.binaryPath();
    if (this.isInstalled()) return dest;

    this.fs.mkdirSync(this.dir, { recursive: true });
    const url = engineDownloadUrl(this.platform, this.gpu, this.urlBase, this.version);

    if (isZipUrl(url)) {
      const zipPath = path.join(this.dir, 'engine.zip');
      await this.download(url, zipPath, onProgress);
      await this.extract(zipPath, dest);
      this.fs.unlinkSync(zipPath);
    } else {
      await this.download(url, dest, onProgress);
    }

    if (this.platform !== 'win32') this.chmod(dest, 0o755);
    return dest;
  }
}

module.exports = { EngineManager };
