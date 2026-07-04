'use strict';

const path = require('path');
const { LLM } = require('../shared/config');
const { resolveServerBinary } = require('../shared/llama');
const { isZipUrl } = require('../shared/engine');

// Ensures the llama.cpp `llama-server` binary and the GGUF model are present,
// downloading them on demand. Like EngineManager, all IO (fs, download, extract,
// chmod) is injected so the orchestration is fully unit-testable. `serverUrl` is
// the platform-specific llama-server download (a zip of releases, or a bare
// binary); the model URL comes from config.
class LlmEngineManager {
  constructor({ dir, platform, serverUrl, fs, download, extract, chmod } = {}) {
    this.dir = dir;
    this.platform = platform;
    this.serverUrl = serverUrl;
    this.fs = fs;
    this.download = download;
    this.extract = extract;
    this.chmod = chmod;
  }

  serverBinaryPath() {
    return path.join(this.dir, resolveServerBinary(null, this.platform));
  }

  modelPath() {
    return path.join(this.dir, LLM.model.file);
  }

  isServerInstalled() {
    return this.fs.existsSync(this.serverBinaryPath());
  }

  isModelInstalled() {
    return this.fs.existsSync(this.modelPath());
  }

  // Resolve to the server binary path, downloading + installing it if missing.
  async ensureServer(onProgress) {
    const dest = this.serverBinaryPath();
    if (this.isServerInstalled()) return dest;

    this.fs.mkdirSync(this.dir, { recursive: true });
    if (isZipUrl(this.serverUrl)) {
      const zipPath = path.join(this.dir, 'llama-server.zip');
      await this.download(this.serverUrl, zipPath, onProgress);
      await this.extract(zipPath, dest);
      this.fs.unlinkSync(zipPath);
    } else {
      await this.download(this.serverUrl, dest, onProgress);
    }

    if (this.platform !== 'win32') this.chmod(dest, 0o755);
    return dest;
  }

  // Resolve to the GGUF model path, downloading it (a plain file) if missing.
  async ensureModel(onProgress) {
    const dest = this.modelPath();
    if (this.isModelInstalled()) return dest;
    this.fs.mkdirSync(this.dir, { recursive: true });
    await this.download(LLM.model.url, dest, onProgress);
    return dest;
  }
}

module.exports = { LlmEngineManager };
