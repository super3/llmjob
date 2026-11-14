'use strict';

// The app's log on disk (src/main/logFile.js), on a real temp folder.

const fs = require('fs');
const os = require('os');
const path = require('path');
const { makeLogFile, MAX_BYTES } = require('../src/main/logFile');

let dir;
beforeEach(() => { dir = fs.mkdtempSync(path.join(os.tmpdir(), 'earnlog-')); });
afterEach(() => { fs.rmSync(dir, { recursive: true, force: true }); });

const at = () => new Date('2026-11-14T13:05:01.123Z');
const read = (f) => fs.readFileSync(path.join(dir, 'logs', f), 'utf8');

describe('the log file', () => {
  it('appends each line with a time and level, making the folder', () => {
    const log = makeLogFile({ fs, dir: path.join(dir, 'logs'), now: at });
    log.write('info', 'pearl: started');
    log.write('error', 'could not start mining');
    log.write(undefined, 42);
    expect(log.file).toBe(path.join(dir, 'logs', 'earn.log'));
    expect(read('earn.log')).toBe(
      '2026-11-14T13:05:01.123Z INFO  pearl: started\n'
      + '2026-11-14T13:05:01.123Z ERROR could not start mining\n'
      + '2026-11-14T13:05:01.123Z INFO  42\n');
  });

  it('keeps writing after the lines already there, and moves a full file to earn.1.log', () => {
    fs.mkdirSync(path.join(dir, 'logs'));
    fs.writeFileSync(path.join(dir, 'logs', 'earn.log'), 'x'.repeat(90) + '\n');
    const log = makeLogFile({ fs, dir: path.join(dir, 'logs'), now: at, maxBytes: 120 });
    log.write('info', 'a');
    expect(read('earn.1.log')).toBe('x'.repeat(90) + '\n');
    expect(read('earn.log')).toMatch(/INFO {2}a\n$/);
    log.write('info', 'b');
    expect(read('earn.log').split('\n').filter(Boolean)).toHaveLength(2);
    expect(MAX_BYTES).toBe(1024 * 1024);
  });

  it('stamps lines with the real time by default', () => {
    const log = makeLogFile({ fs, dir: path.join(dir, 'logs') });
    log.write('warn', 'w');
    expect(read('earn.log')).toMatch(/^\d{4}-\d\d-\d\dT[\d:.]+Z WARN {2}w\n$/);
  });

  it('never throws when the disk refuses', () => {
    const broken = {
      mkdirSync: () => { throw new Error('EROFS'); },
      writeFileSync: () => { throw new Error('EROFS'); },
      unlinkSync: () => { throw new Error('ENOENT'); },
      readFileSync: () => { throw new Error('ENOENT'); },
    };
    const log = makeLogFile({ fs: broken, dir: '/nowhere' });
    expect(() => log.write('info', 'x')).not.toThrow();
    expect(() => log.markMining({})).not.toThrow();
    expect(() => log.clearMining()).not.toThrow();
    expect(log.takeCrash()).toBeNull();
  });
});

describe('a run that closed while mining', () => {
  it('is reported once, with what the marker said', () => {
    const log = makeLogFile({ fs, dir: path.join(dir, 'logs'), now: at });
    expect(log.takeCrash()).toBeNull();
    log.markMining({ version: '0.7.2' });
    // The next launch finds it.
    const next = makeLogFile({ fs, dir: path.join(dir, 'logs'), now: at });
    expect(next.takeCrash()).toEqual({ startedAt: '2026-11-14T13:05:01.123Z', version: '0.7.2' });
    expect(next.takeCrash()).toBeNull();
  });

  it('is not reported after a clean stop', () => {
    const log = makeLogFile({ fs, dir: path.join(dir, 'logs'), now: at });
    log.markMining({});
    log.clearMining();
    expect(log.takeCrash()).toBeNull();
  });

  it('still counts when the marker is unreadable or not an object', () => {
    const log = makeLogFile({ fs, dir: path.join(dir, 'logs'), now: at });
    log.markMining({});
    fs.writeFileSync(path.join(dir, 'logs', 'mining.json'), '{half');
    expect(log.takeCrash()).toEqual({});
    fs.writeFileSync(path.join(dir, 'logs', 'mining.json'), 'null');
    expect(log.takeCrash()).toEqual({});
  });
});
