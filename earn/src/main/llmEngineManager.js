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
  constructor({ dir, platform, serverUrl, fs, download, extract, chmod, model } = {}) {
    this.dir = dir;
    this.platform = platform;
    this.serverUrl = serverUrl;
    this.fs = fs;
    this.download = download;
    this.extract = extract;
    this.chmod = chmod;
    // Which catalog model to fetch/serve. Defaults to the small model so callers
    // that don't select one (and the existing tests) keep their behavior; the
    // VRAM-tiered selection in main.js / earn-cli.js passes the chosen model.
    this.model = model || LLM.model;
  }

  serverBinaryPath() {
    return path.join(this.dir, resolveServerBinary(null, this.platform));
  }

  modelPath() {
    return path.join(this.dir, this.model.file);
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
    if (this.isServerInstalled()) {
      // A cached llama-server can be present but lack the execute bit: the
      // download writes it 0o644 and only chmods afterwards, so an interrupted
      // first install (crash/kill/reboot between the rename and the chmod) —
      // or a binary put there by any other path — leaves a non-executable file
      // that spawns with EACCES forever, since this early return used to skip
      // the chmod below. Re-assert +x here so a stuck node self-heals on the
      // next start. Best effort: a chmod failure (read-only dir, foreign owner)
      // must not turn a node whose binary is already executable into a crash.
      this.ensureExecutable(dest);
      return dest;
    }

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

  // Grant the llama-server its execute bit (non-Windows only). Best effort by
  // design: this runs on the already-installed path where the file may sit on a
  // read-only mount or be owned by another user, and a throw there would break
  // a node whose binary is already executable. The fresh-download path keeps its
  // own strict chmod so a genuine install failure still surfaces.
  ensureExecutable(dest) {
    if (this.platform === 'win32') return;
    try {
      this.chmod(dest, 0o755);
    } catch (e) {
      /* best effort — spawn will report EACCES if it truly isn't executable */
    }
  }

  // Resolve to the GGUF model path, downloading it (a plain file) if missing.
  async ensureModel(onProgress) {
    const dest = this.modelPath();
    if (this.isModelInstalled()) return dest;
    this.fs.mkdirSync(this.dir, { recursive: true });
    await this.download(this.model.url, dest, onProgress);
    return dest;
  }
}

module.exports = { LlmEngineManager };
