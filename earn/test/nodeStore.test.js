'use strict';

jest.mock('fs');
jest.mock('os');

const fs = require('fs');
const os = require('os');
const path = require('path');
const store = require('../src/main/nodeStore');

// Build the expected paths with path.join so they match on every OS — the store
// uses path.join, which yields backslashes on Windows.
const STORE_DIR = path.join('/home/test', '.local', 'share', 'llmjob-earn');
const NODE_PATH = path.join(STORE_DIR, 'node.json');

describe('nodeStore', () => {
  let files;

  beforeEach(() => {
    jest.clearAllMocks();
    files = new Map();
    os.homedir.mockReturnValue('/home/test');
    fs.readFileSync.mockImplementation((p) => {
      if (files.has(p)) return files.get(p);
      const e = new Error('ENOENT: no such file'); e.code = 'ENOENT';
      throw e;
    });
    fs.writeFileSync.mockImplementation((p, data) => { files.set(p, data); });
    fs.renameSync.mockImplementation((from, to) => { files.set(to, files.get(from)); files.delete(from); });
    fs.mkdirSync.mockImplementation(() => {});
  });

  it('nodePath is under the shared store dir', () => {
    expect(store.nodePath()).toBe(NODE_PATH);
  });

  describe('loadNode', () => {
    it('returns null when the file is missing', () => {
      expect(store.loadNode()).toBeNull();
    });

    it('parses an existing identity', () => {
      files.set(NODE_PATH, JSON.stringify({ nodeId: 'n1', publicKey: 'pk' }));
      expect(store.loadNode()).toEqual({ nodeId: 'n1', publicKey: 'pk' });
    });

    it('returns null on invalid JSON', () => {
      files.set(NODE_PATH, 'not json{');
      expect(store.loadNode()).toBeNull();
    });
  });

  describe('saveNode', () => {
    // Replaced whole, through a temp file renamed over it. Written in place, a
    // power cut could leave a torn node.json, which reads as no identity, and
    // getOrCreateNode would then make a new key and unlink the machine.
    it('creates the dir and replaces node.json whole, readable by its owner only', () => {
      files.set(NODE_PATH, JSON.stringify({ nodeId: 'old' }));
      store.saveNode({ nodeId: 'n1' });
      expect(fs.mkdirSync).toHaveBeenCalledWith(STORE_DIR, { recursive: true });
      const tmp = NODE_PATH + '.' + process.pid + '.tmp';
      const call = fs.writeFileSync.mock.calls[0];
      expect(call[0]).toBe(tmp);
      expect(JSON.parse(call[1])).toEqual({ nodeId: 'n1' });
      expect(call[2]).toEqual({ mode: 0o600 });
      expect(fs.renameSync).toHaveBeenCalledWith(tmp, NODE_PATH);
      expect([...files.keys()]).toEqual([NODE_PATH]);
      expect(store.loadNode()).toEqual({ nodeId: 'n1' });
    });

    it('throws when it cannot be saved, as before', () => {
      fs.writeFileSync.mockImplementation(() => { throw Object.assign(new Error('disk full'), { code: 'ENOSPC' }); });
      expect(() => store.saveNode({ nodeId: 'n1' })).toThrow('disk full');
    });
  });

  describe('migrateFrom', () => {
    it('does nothing when an identity already exists', () => {
      files.set(NODE_PATH, JSON.stringify({ publicKey: 'pk', secretKey: 'sk' }));
      expect(store.migrateFrom('/legacy/node.json')).toBe(false);
      expect(fs.writeFileSync).not.toHaveBeenCalled();
    });

    it('does nothing when no legacy path is given', () => {
      expect(store.migrateFrom(null)).toBe(false);
    });

    it('copies a full legacy identity into the shared store', () => {
      files.set('/legacy/node.json', JSON.stringify({ publicKey: 'pk', secretKey: 'sk', nodeId: 'n1' }));
      expect(store.migrateFrom('/legacy/node.json')).toBe(true);
      expect(store.loadNode()).toMatchObject({ publicKey: 'pk', secretKey: 'sk' });
    });

    it('ignores a legacy file without a keypair', () => {
      files.set('/legacy/node.json', JSON.stringify({ nodeId: 'n1' }));
      expect(store.migrateFrom('/legacy/node.json')).toBe(false);
      expect(store.loadNode()).toBeNull();
    });

    it('ignores an unreadable legacy file', () => {
      expect(store.migrateFrom('/legacy/missing.json')).toBe(false);
    });
  });

  describe('getOrCreateNode', () => {
    it('creates and persists a new identity on first use', () => {
      const node = store.getOrCreateNode();
      expect(node.publicKey).toBeTruthy();
      expect(node.secretKey).toBeTruthy();
      expect(node.nodeId).toBeTruthy();
      expect(node.connected).toBe(false);
      // Persisted, and stable across a reload.
      expect(store.loadNode()).toEqual(node);
    });

    it('returns the existing identity without rewriting it', () => {
      const first = store.getOrCreateNode();
      fs.writeFileSync.mockClear();
      const again = store.getOrCreateNode();
      expect(again).toEqual(first);
      expect(fs.writeFileSync).not.toHaveBeenCalled();
    });

    it('regenerates when the stored identity is missing its keypair', () => {
      files.set(NODE_PATH, JSON.stringify({ nodeId: 'stale' }));
      const node = store.getOrCreateNode();
      expect(node.publicKey).toBeTruthy();
      expect(node.secretKey).toBeTruthy();
    });
  });
});
