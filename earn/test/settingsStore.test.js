'use strict';

const { readSettings, writeSettings } = require('../src/shared/settingsStore');

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
});

describe('writeSettings', () => {
  it('writes pretty JSON and returns the data', () => {
    let written;
    const fs = { writeFileSync: (file, body) => { written = { file, body }; } };
    const data = { address: 'prl1abc' };
    expect(writeSettings('/s.json', data, { fs })).toBe(data);
    expect(written.file).toBe('/s.json');
    expect(JSON.parse(written.body)).toEqual(data);
  });

  it('logs on a write failure instead of throwing', () => {
    const fs = { writeFileSync: () => { throw new Error('disk full'); } };
    const logs = [];
    expect(() => writeSettings('/s.json', { a: 1 }, { fs, log: (m) => logs.push(m) })).not.toThrow();
    expect(logs[0]).toMatch(/Could not save settings/);
  });

  it('tolerates a missing logger on failure (default noop)', () => {
    const fs = { writeFileSync: () => { throw new Error('disk full'); } };
    expect(() => writeSettings('/s.json', { a: 1 }, { fs })).not.toThrow();
  });
});
