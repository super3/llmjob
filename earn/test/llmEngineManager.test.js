'use strict';

const path = require('path');
const { LlmEngineManager } = require('../src/main/llmEngineManager');
const { LLM } = require('../src/shared/config');

function fsMock(existing = new Set()) {
  return {
    existsSync: (p) => existing.has(p),
    mkdirSync: jest.fn(),
    unlinkSync: jest.fn(),
  };
}

// An fs whose openSync/readSync serve `head` (a Buffer) as the file's first
// bytes — used to exercise the "cached file is really an un-extracted archive"
// self-heal. `readFail` makes openSync throw, the unreadable-file path.
function fsMockWithHead(existing, head, readFail) {
  const fs = fsMock(existing);
  fs.closeSync = jest.fn();
  fs.openSync = jest.fn(() => { if (readFail) throw new Error('EACCES'); return 7; });
  fs.readSync = jest.fn((fd, buf) => { head.copy(buf); return head.length; });
  return fs;
}

const GZIP_MAGIC = Buffer.from([0x1f, 0x8b, 0x08, 0x00]);
const ELF_MAGIC = Buffer.from([0x7f, 0x45, 0x4c, 0x46]);

describe('LlmEngineManager', () => {
  test('constructs with no arguments', () => {
    expect(() => new LlmEngineManager()).not.toThrow();
  });

  test('resolves paths from dir + platform binary + model file', () => {
    const m = new LlmEngineManager({ dir: '/eng', platform: 'win32', fs: fsMock() });
    expect(m.serverBinaryPath()).toBe(path.join('/eng', 'llama-server.exe'));
    expect(m.modelPath()).toBe(path.join('/eng', LLM.model.file));
  });

  test('ensureServer returns early and re-asserts +x when already installed off Windows', async () => {
    const dest = path.join('/eng', 'llama-server');
    const fs = fsMock(new Set([dest]));
    const download = jest.fn();
    const chmod = jest.fn();
    const m = new LlmEngineManager({ dir: '/eng', platform: 'linux', serverUrl: 'http://x/s.zip', fs, download, extract: jest.fn(), chmod });
    expect(await m.ensureServer()).toBe(dest);
    expect(download).not.toHaveBeenCalled();
    expect(fs.mkdirSync).not.toHaveBeenCalled();
    // A cached binary that lost its +x (e.g. an interrupted first install) gets
    // it back here, so the node stops crash-looping on spawn EACCES.
    expect(chmod).toHaveBeenCalledWith(dest, 0o755);
  });

  test('ensureServer does not chmod an already-installed binary on Windows', async () => {
    const dest = path.join('/eng', 'llama-server.exe');
    const fs = fsMock(new Set([dest]));
    const chmod = jest.fn();
    const m = new LlmEngineManager({ dir: '/eng', platform: 'win32', serverUrl: 'http://x/s.zip', fs, download: jest.fn(), extract: jest.fn(), chmod });
    expect(await m.ensureServer()).toBe(dest);
    expect(chmod).not.toHaveBeenCalled();
  });

  test('ensureServer swallows a failing chmod on the cached path (best effort)', async () => {
    const dest = path.join('/eng', 'llama-server');
    const fs = fsMock(new Set([dest]));
    const chmod = jest.fn(() => { throw new Error('EROFS: read-only file system'); });
    const m = new LlmEngineManager({ dir: '/eng', platform: 'linux', serverUrl: 'http://x/s.zip', fs, chmod });
    // Must still resolve: a chmod failure on an already-executable binary must
    // not turn a working node into a crash.
    expect(await m.ensureServer()).toBe(dest);
    expect(chmod).toHaveBeenCalledWith(dest, 0o755);
  });

  test('ensureServer downloads + extracts a zip and chmods off-Windows', async () => {
    const fs = fsMock();
    const download = jest.fn().mockResolvedValue();
    const extract = jest.fn().mockResolvedValue();
    const chmod = jest.fn();
    const m = new LlmEngineManager({ dir: '/eng', platform: 'linux', serverUrl: 'http://x/llama.zip', fs, download, extract, chmod });

    const p = await m.ensureServer();
    expect(fs.mkdirSync).toHaveBeenCalledWith('/eng', { recursive: true });
    expect(download).toHaveBeenCalledWith('http://x/llama.zip', path.join('/eng', 'llama-download.archive'), undefined);
    expect(extract).toHaveBeenCalledWith(path.join('/eng', 'llama-download.archive'), path.join('/eng', 'llama-server'));
    expect(fs.unlinkSync).toHaveBeenCalledWith(path.join('/eng', 'llama-download.archive'));
    expect(chmod).toHaveBeenCalledWith(path.join('/eng', 'llama-server'), 0o755);
    expect(p).toBe(path.join('/eng', 'llama-server'));
  });

  // The Linux/macOS regression: llama.cpp ships those platforms as .tar.gz, and
  // gating extraction on ".zip" alone saved the TARBALL as `llama-server` and
  // chmod +x'd it. execvp then rejects it (ENOEXEC), /bin/sh tries to parse the
  // gzip ("Syntax error: word unexpected") and exits 2 — every non-Windows rig.
  test.each([
    'http://x/llama-b9902-bin-ubuntu-vulkan-x64.tar.gz',
    'http://x/llama-macos-arm64.tgz',
  ])('ensureServer extracts %s instead of saving the archive as the binary', async (url) => {
    const fs = fsMock();
    const download = jest.fn().mockResolvedValue();
    const extract = jest.fn().mockResolvedValue();
    const chmod = jest.fn();
    const m = new LlmEngineManager({ dir: '/eng', platform: 'linux', serverUrl: url, fs, download, extract, chmod });

    const dest = path.join('/eng', 'llama-server');
    const archive = path.join('/eng', 'llama-download.archive');
    expect(await m.ensureServer()).toBe(dest);
    // The archive must NOT be downloaded straight to the binary path.
    expect(download).toHaveBeenCalledWith(url, archive, undefined);
    expect(extract).toHaveBeenCalledWith(archive, dest);
    expect(fs.unlinkSync).toHaveBeenCalledWith(archive);
    expect(chmod).toHaveBeenCalledWith(dest, 0o755);
  });

  test('a cached file that is really an un-extracted archive is not "installed"', async () => {
    const dest = path.join('/eng', 'llama-server');
    const fs = fsMockWithHead(new Set([dest]), GZIP_MAGIC);
    const download = jest.fn().mockResolvedValue();
    const extract = jest.fn().mockResolvedValue();
    const m = new LlmEngineManager({
      dir: '/eng', platform: 'linux', serverUrl: 'http://x/llama.tar.gz', fs, download, extract, chmod: jest.fn(),
    });

    // Rigs left over from the broken install have a gzip sitting at the binary
    // path; it must re-download and extract rather than spawn it forever.
    expect(m.isServerInstalled()).toBe(false);
    await m.ensureServer();
    expect(download).toHaveBeenCalled();
    expect(extract).toHaveBeenCalled();
    expect(fs.closeSync).toHaveBeenCalled();
  });

  test('a real binary is installed, and an unreadable one is left alone', () => {
    const dest = path.join('/eng', 'llama-server');
    const opts = { dir: '/eng', platform: 'linux', serverUrl: 'http://x/llama.tar.gz', chmod: jest.fn() };
    expect(new LlmEngineManager({ ...opts, fs: fsMockWithHead(new Set([dest]), ELF_MAGIC) }).isServerInstalled()).toBe(true);
    // Unreadable → keep the old behaviour (installed), so a permissions quirk
    // can't force an endless re-download.
    expect(new LlmEngineManager({ ...opts, fs: fsMockWithHead(new Set([dest]), ELF_MAGIC, true) }).isServerInstalled()).toBe(true);
    // Absent → not installed, without ever opening it.
    expect(new LlmEngineManager({ ...opts, fs: fsMockWithHead(new Set(), ELF_MAGIC) }).isServerInstalled()).toBe(false);
  });

  test('ensureServer downloads a bare binary and skips chmod on Windows', async () => {
    const fs = fsMock();
    const download = jest.fn().mockResolvedValue();
    const chmod = jest.fn();
    const m = new LlmEngineManager({ dir: '/eng', platform: 'win32', serverUrl: 'http://x/llama-server.exe', fs, download, extract: jest.fn(), chmod });

    await m.ensureServer();
    expect(download).toHaveBeenCalledWith('http://x/llama-server.exe', path.join('/eng', 'llama-server.exe'), undefined);
    expect(chmod).not.toHaveBeenCalled();
  });

  test('ensureModel returns early when installed, else downloads the GGUF', async () => {
    const modelP = path.join('/eng', LLM.model.file);

    const dl1 = jest.fn();
    const m1 = new LlmEngineManager({ dir: '/eng', platform: 'linux', fs: fsMock(new Set([modelP])), download: dl1 });
    expect(await m1.ensureModel()).toBe(modelP);
    expect(dl1).not.toHaveBeenCalled();

    const dl2 = jest.fn().mockResolvedValue();
    const fs2 = fsMock();
    const m2 = new LlmEngineManager({ dir: '/eng', platform: 'linux', fs: fs2, download: dl2 });
    expect(await m2.ensureModel()).toBe(modelP);
    expect(fs2.mkdirSync).toHaveBeenCalledWith('/eng', { recursive: true });
    expect(dl2).toHaveBeenCalledWith(LLM.model.url, modelP, undefined);
  });
});

// A node no longer serves one fixed model: a card large enough gets a bigger
// tier, so paths, install checks and downloads are all per-model. Every method
// keeps its old zero-argument meaning (the fleet default) so existing callers
// are untouched.
describe('LlmEngineManager — per-model paths and the vision projector', () => {
  const { LLM } = require('../src/shared/config');
  const VISION = {
    name: 'Big', file: 'big.gguf', url: 'https://h/big.gguf',
    mmproj: { file: 'big-mmproj.gguf', url: 'https://h/mmproj.gguf' },
  };
  // Built with path.join like every other assertion in this file: the separator
  // is the platform's, and hardcoding '/d/...' passes on Linux and fails on the
  // Windows runner.
  const WEIGHTS = path.join('/d', 'big.gguf');
  const PROJ = path.join('/d', 'big-mmproj.gguf');

  function mgr(present = []) {
    const seen = new Set(present);
    const download = jest.fn(async (url, dest) => { seen.add(dest); });
    return {
      download,
      m: new LlmEngineManager({
        dir: '/d', platform: 'linux', download,
        fs: { existsSync: (p) => seen.has(p), mkdirSync: jest.fn() },
      }),
    };
  }

  test('paths are per-model, so a big model cannot overwrite the small one', () => {
    const { m } = mgr();
    expect(m.modelPath(VISION)).toBe(WEIGHTS);
    expect(m.modelPath(LLM.model)).toBe(path.join('/d', LLM.model.file));
    expect(m.modelPath()).toBe(path.join('/d', LLM.model.file));   // unchanged default
    expect(m.modelPath(VISION)).not.toBe(m.modelPath());
  });

  test('mmprojPath is null for a model without a projector', () => {
    const { m } = mgr();
    expect(m.mmprojPath(VISION)).toBe(PROJ);
    expect(m.mmprojPath(LLM.model)).toBeNull();
    expect(m.mmprojPath()).toBeNull();
  });

  test('a vision model is not installed until BOTH files are there', () => {
    const weightsOnly = mgr([WEIGHTS]);
    expect(weightsOnly.m.isModelInstalled(VISION)).toBe(true);
    expect(weightsOnly.m.isMmprojInstalled(VISION)).toBe(false);

    const both = mgr([WEIGHTS, PROJ]);
    expect(both.m.isMmprojInstalled(VISION)).toBe(true);
  });

  test('a model with no projector is trivially satisfied', () => {
    const { m } = mgr();
    expect(m.isMmprojInstalled(LLM.model)).toBe(true);
    expect(m.isMmprojInstalled()).toBe(true);
  });

  test('ensureModel downloads the requested model, not always the default', async () => {
    const { m, download } = mgr();
    await expect(m.ensureModel(null, VISION)).resolves.toBe(WEIGHTS);
    expect(download).toHaveBeenCalledWith('https://h/big.gguf', WEIGHTS, null);
  });

  test('ensureMmproj fetches the projector separately from the weights', async () => {
    // Separate on purpose: a node whose 17 GB of weights are already on disk
    // must be able to pick up a 0.9 GB projector without re-downloading them.
    const { m, download } = mgr([WEIGHTS]);
    await expect(m.ensureMmproj(null, VISION)).resolves.toBe(PROJ);
    expect(download).toHaveBeenCalledTimes(1);
    expect(download).toHaveBeenCalledWith('https://h/mmproj.gguf', PROJ, null);
  });

  test('ensureMmproj is a no-op returning null for a text-only model', async () => {
    const { m, download } = mgr();
    await expect(m.ensureMmproj(null, LLM.model)).resolves.toBeNull();
    await expect(m.ensureMmproj()).resolves.toBeNull();
    expect(download).not.toHaveBeenCalled();
  });

  test('neither ensure re-downloads a file already present', async () => {
    const { m, download } = mgr([WEIGHTS, PROJ]);
    await m.ensureModel(null, VISION);
    await m.ensureMmproj(null, VISION);
    expect(download).not.toHaveBeenCalled();
  });
});
