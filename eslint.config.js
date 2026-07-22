'use strict';

// Flat ESLint config covering the Node server and the earn desktop/CLI sources.
// The repo previously had no linter at all, while CI ran `npm run lint
// --if-present` (a silent no-op); this makes that step real.

const js = require('@eslint/js');
const globals = require('globals');

module.exports = [
  {
    ignores: ['**/node_modules/**', '**/coverage/**', '**/dist/**'],
  },
  js.configs.recommended,
  {
    // Default: Node CommonJS (server, and the earn main/CLI/shared modules).
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: 'commonjs',
      globals: { ...globals.node },
    },
    rules: {
      // Catch dead imports/vars — the main reason for turning linting on — but
      // don't churn on the many intentional (req, res, next) params or on
      // caught-but-unused errors.
      'no-unused-vars': ['error', { args: 'none', caughtErrors: 'none', varsIgnorePattern: '^_' }],
      // Best-effort empty catches are a deliberate pattern in the IO shells.
      'no-empty': ['error', { allowEmptyCatch: true }],
    },
  },
  {
    // The Electron renderer runs in a browser context (no Node integration).
    files: ['earn/src/renderer/**/*.js'],
    languageOptions: {
      globals: { ...globals.browser },
    },
  },
  {
    // The preload bridge sees both the Node require graph and the DOM.
    files: ['earn/src/main/preload.js'],
    languageOptions: {
      globals: { ...globals.browser },
    },
  },
  {
    // Jest drives the test files.
    files: ['server/tests/**/*.js', 'earn/test/**/*.js'],
    languageOptions: {
      globals: { ...globals.jest },
    },
  },
];
