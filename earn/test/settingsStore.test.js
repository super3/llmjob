'use strict';

const { readSettings, writeSettings, writeFileAtomic } = require('../src/shared/settingsStore');

const TMP = '/s.json.' + process.pid + '.tmp';

// An fs that keeps files in a Map, so a test reads back what landed.
function memFs(files = new Map()) {
  return {
    files,
    existsSync: (p) => files.has(p),
    readFileSync: (p) => files.get(p),
    writeFileSync: jest.fn((p, body) => { files.set(p, body); }),
    renameSync: jest.fn((from, to) => { files.set(to, files.get(from)); files.delete(from); }),
    unlinkSync: jest.fn((p) => { files.delete(p); }),
  };
}

const err = (code, message) => Object.assign(new Error(message || code), { code });

describe('readSettings', () => {
  it('returns {} when the file does not exist', () => {
    const fs = { existsSync: () => false, readFileSync: () => { throw new Error('should not read'); } };
    expect(readSettings('/nope.json', { fs })).toEqual({});
  });

  it('parses an existing settings file', () => {
    const fs = { existsSync: () => true, readFileSync: () => '{"address":"prl1abc","region":"eu1"}' };
    expect(readSettings('/s.json', { fs })).toEqual({ address: 'prl1abc', region: 'eu1' });
  });

  it('logs and returns {} on a corrupt file', () => {
    const fs = { existsSync: () => true, readFileSync: () => 'not json{' };
    const logs = [];
    expect(readSettings('/s.json', { fs, log: (m) => logs.push(m) })).toEqual({});
    expect(logs).toHaveLength(1);
    expect(logs[0]).toMatch(/Could not read settings/);
  });

  it('tolerates a missing logger (default noop)', () => {
    const fs = { existsSync: () => true, readFileSync: () => 'bad' };
    expect(readSettings('/s.json', { fs })).toEqual({});
  });

  // Every caller reads keys off the answer: loadSettings().shareLlm threw on a
  // file holding a bare null.
  it('logs and returns {} for JSON that is not an object', () => {
    for (const body of ['null', '[1,2]', '"text"', '7']) {
      const logs = [];
      const fs = { existsSync: () => true, readFileSync: () => body };
      expect(readSettings('/s.json', { fs, log: (m) => logs.push(m) })).toEqual({});
      expect(logs[0]).toBe('Could not read settings (/s.json): not a JSON object');
    }
  });
});

describe('writeSettings', () => {
  it('writes pretty JSON and returns the data', () => {
    const fs = memFs();
    const data = { address: 'prl1abc' };
    expect(writeSettings('/s.json', data, { fs })).toBe(data);
    expect(JSON.parse(fs.files.get('/s.json'))).toEqual(data);
    expect(fs.files.get('/s.json')).toBe(JSON.stringify(data, null, 2));
  });

  // main writes preferences.json and gpu-lost.json this way too, and the line
  // used to say "settings" for all of them.
  it('logs a write failure naming the file, instead of throwing', () => {
    const fs = memFs();
    fs.writeFileSync.mockImplementation(() => { throw err('ENOSPC', 'disk full'); });
    const logs = [];
    expect(() => writeSettings('/prefs.json', { a: 1 }, { fs, log: (m) => logs.push(m) })).not.toThrow();
    expect(logs).toEqual(['Could not save /prefs.json: disk full']);
  });

  it('tolerates a missing logger on failure (default noop)', () => {
    const fs = memFs();
    fs.writeFileSync.mockImplementation(() => { throw err('ENOSPC', 'disk full'); });
    expect(() => writeSettings('/s.json', { a: 1 }, { fs })).not.toThrow();
  });
});

// A file written in place and cut off by a power cut is a torn file: settings
// lost, or node.json read as no identity. Replaced whole, the file is the old
// one or the new one.
describe('writeFileAtomic', () => {
  it('writes a temp file named for this process, then renames it over', () => {
    const fs = memFs(new Map([['/s.json', 'old']]));
    writeFileAtomic('/s.json', 'new', { fs });
    expect(fs.writeFileSync).toHaveBeenCalledWith(TMP, 'new', undefined);
    expect(fs.renameSync).toHaveBeenCalledWith(TMP, '/s.json');
    expect([...fs.files]).toEqual([['/s.json', 'new']]);
  });

  it('passes a mode on, for a file only its owner may read', () => {
    const fs = memFs();
    writeFileAtomic('/s.json', 'secret', { fs, mode: 0o600 });
    expect(fs.writeFileSync).toHaveBeenCalledWith(TMP, 'secret', { mode: 0o600 });
  });

  // Windows refuses the rename while another process holds the file open.
  it('writes in place when the rename is refused, and removes the temp file', () => {
    for (const code of ['EPERM', 'EACCES', 'EBUSY']) {
      const fs = memFs(new Map([['/s.json', 'old']]));
      fs.renameSync.mockImplementation(() => { throw err(code); });
      writeFileAtomic('/s.json', 'new', { fs, mode: 0o600 });
      expect(fs.writeFileSync).toHaveBeenLastCalledWith('/s.json', 'new', { mode: 0o600 });
      expect([...fs.files]).toEqual([['/s.json', 'new']]);
    }
  });

  // A temp file that could not be written in full is not left behind, and the
  // live file is not touched.
  it('throws any other failure, leaving the live file as it was', () => {
    const fs = memFs(new Map([['/s.json', 'old']]));
    fs.writeFileSync.mockImplementation((p) => { fs.files.set(p, 'half'); throw err('ENOSPC', 'disk full'); });
    expect(() => writeFileAtomic('/s.json', 'new', { fs })).toThrow('disk full');
    expect([...fs.files]).toEqual([['/s.json', 'old']]);

    const gone = memFs(new Map([['/s.json', 'old']]));
    gone.renameSync.mockImplementation(() => { throw err('ENOENT'); });
    gone.unlinkSync.mockImplementation(() => { throw err('ENOENT'); });
    expect(() => writeFileAtomic('/s.json', 'new', { fs: gone })).toThrow('ENOENT');
    expect(gone.files.get('/s.json')).toBe('old');
  });
});
