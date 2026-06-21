'use strict';

// Coverage is enforced at 100% on the pure logic modules (shared/*) and the
// process supervisor (minerManager). The Electron entry points (main.js,
// preload.js) and the renderer are thin shells that require an Electron/DOM
// runtime and are intentionally excluded from coverage.
module.exports = {
  rootDir: '.',
  testEnvironment: 'node',
  collectCoverage: true,
  collectCoverageFrom: ['src/shared/**/*.js', 'src/main/minerManager.js', 'src/main/engineManager.js'],
  coverageThreshold: {
    global: { branches: 100, functions: 100, lines: 100, statements: 100 },
  },
  testMatch: ['<rootDir>/test/**/*.test.js'],
};
