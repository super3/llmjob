'use strict';

const fs = require('fs');
const path = require('path');
const { POOL_THREADS, widenThreadPool } = require('../src/shared/threadPool');

const SRC = path.join(__dirname, '..', 'src');
const read = (...p) => fs.readFileSync(path.join(SRC, ...p), 'utf8');

describe('widenThreadPool', () => {
  test('sets a pool big enough for a scan a card on a large rig when nobody chose a size', () => {
    expect(POOL_THREADS).toBeGreaterThanOrEqual(16);
    const env = {};
    expect(widenThreadPool(env)).toBe(true);
    expect(env.UV_THREADPOOL_SIZE).toBe(String(POOL_THREADS));
    const blank = { UV_THREADPOOL_SIZE: ' ' };
    expect(widenThreadPool(blank)).toBe(true);
    expect(blank.UV_THREADPOOL_SIZE).toBe(String(POOL_THREADS));
  });

  test('keeps a size the user chose', () => {
    const env = { UV_THREADPOOL_SIZE: '8' };
    expect(widenThreadPool(env)).toBe(false);
    expect(env.UV_THREADPOOL_SIZE).toBe('8');
  });

  // libuv reads the variable when the pool is first used, so each shell calls this before its other requires.
  test('both shells call it first, before any module that could use the pool', () => {
    for (const file of [['cli', 'earn-cli.js'], ['main', 'main.js']]) {
      const text = read(...file);
      const call = text.indexOf("require('../shared/threadPool').widenThreadPool(process.env);");
      expect([file.join('/'), call > 0]).toEqual([file.join('/'), true]);
      expect([file.join('/'), text.indexOf('require(') === text.indexOf("require('../shared/threadPool')")]).toEqual([file.join('/'), true]);
    }
  });
});
