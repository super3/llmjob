module.exports = {
  testEnvironment: 'node',
  coverageDirectory: 'coverage',
  collectCoverageFrom: [
    'src/**/*.js',
    '!src/cli.js', // CLI will be tested separately
    '!src/index.js' // Just exports
  ],
  testMatch: [
    '**/tests/**/*.test.js',
    '!**/tests/cli.test.js' // Skip CLI integration tests for now
  ],
  coverageThreshold: {
    global: {
      branches: 85,
      functions: 95,
      lines: 95,
      statements: 95
    }
  },
  testTimeout: 10000
};