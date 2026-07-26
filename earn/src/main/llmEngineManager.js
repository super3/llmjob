'use strict';

const path = require('path');
const { LLM } = require('../shared/config');
const { resolveServerBinary } = require('../shared/llama');
const { isArchiveUrl, looksLikeArchive } = require('../shared/engine');

// Where an archive lands before it's extracted. One fixed name for every format:
// extractLlamaZip picks tar-vs-unzip from the file's MAGIC BYTES, not its
// extension, so nothing downstream needs the suffix. It deliberately does not
// start with the binary's own name — the archive must never be mistaken for,
// or overwrite, `llama-server` itself.
const ARCHIVE_TMP = 'llama-download.archive';

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

  // Installed means "a usable binary is there" — not merely "a file is there".
  // A file whose first bytes are gzip/zip magic is an un-extracted archive left
  // by the install bug that saved the .tar.gz as `llama-server`: it exists, it's
  // executable, and it can never run (ENOEXEC). Treat that as NOT installed so
  // ensureServer re-downloads and extracts properly, the same self-heal spirit
  // as the missing-execute-bit fix below. An unreadable file is treated as
  // installed — the old behaviour — so a permissions quirk can't force an
  // endless re-download.
  isServerInstalled() {
    const dest = this.serverBinaryPath();
    if (!this.fs.existsSync(dest)) return false;
    return !this.isArchiveFile(dest);
  }

  // Read the first bytes of `file` and report whether they're archive magic.
  // False when the file can't be read — see isServerInstalled.
  isArchiveFile(file) {
    try {
      const fd = this.fs.openSync(file, 'r');
      try {
        const head = Buffer.alloc(4);
        const read = this.fs.readSync(fd, head, 0, 4, 0);
        return looksLikeArchive(head.subarray(0, read));
      } finally {
        this.fs.closeSync(fd);
      }
    } catch (e) {
      return false;
    }
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
    if (isArchiveUrl(this.serverUrl)) {
      const archivePath = path.join(this.dir, ARCHIVE_TMP);
      await this.download(this.serverUrl, archivePath, onProgress);
      await this.extract(archivePath, dest);
      this.fs.unlinkSync(archivePath);
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
    await this.download(LLM.model.url, dest, onProgress);
    return dest;
  }
}

module.exports = { LlmEngineManager };
