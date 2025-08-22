module.exports = {
  testEnvironment: 'node',
  coverageDirectory: 'coverage',
  collectCoverageFrom: [
    'server/src/**/*.js',
    '!server/src/index.js', // Exclude main server file from coverage
  ],
  testMatch: [
    '**/server/tests/**/*.test.js'
  ],
  coverageThreshold: {
    global: {
      branches: 50,
      functions: 50,
      lines: 50,
      statements: 50
    }
  },
  setupFilesAfterEnv: ['./server/tests/setup.js']
};