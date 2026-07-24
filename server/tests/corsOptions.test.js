'use strict';

const { isAllowedOrigin, corsOrigin } = require('../src/corsOptions');

describe('isAllowedOrigin', () => {
  test('allows a missing origin (non-browser request)', () => {
    expect(isAllowedOrigin(undefined)).toBe(true);
    expect(isAllowedOrigin('')).toBe(true);
    expect(isAllowedOrigin(null)).toBe(true);
  });

  test('allows our own site + Railway prod origins', () => {
    expect(isAllowedOrigin('https://llmjob.com')).toBe(true);
    expect(isAllowedOrigin('https://www.llmjob.com')).toBe(true);
    expect(isAllowedOrigin('https://llmjob-production.up.railway.app')).toBe(true);
  });

  test('allows localhost and Railway PR previews', () => {
    expect(isAllowedOrigin('http://localhost')).toBe(true);
    expect(isAllowedOrigin('http://localhost:8080')).toBe(true);
    expect(isAllowedOrigin('http://127.0.0.1:3000')).toBe(true);
    expect(isAllowedOrigin('https://llmjob-llmjob-pr-131.up.railway.app')).toBe(true);
  });

  test('refuses other websites and look-alikes', () => {
    expect(isAllowedOrigin('https://evil.com')).toBe(false);
    expect(isAllowedOrigin('https://llmjob.com.evil.com')).toBe(false);
    expect(isAllowedOrigin('https://notllmjob-production.up.railway.app')).toBe(false);
    expect(isAllowedOrigin('http://llmjob.com')).toBe(false); // http, not https
  });
});

describe('corsOrigin (cors callback form)', () => {
  test('resolves true for an allowed origin and false otherwise, never errors', () => {
    const seen = [];
    corsOrigin('https://llmjob.com', (err, ok) => seen.push([err, ok]));
    corsOrigin('https://evil.com', (err, ok) => seen.push([err, ok]));
    corsOrigin(undefined, (err, ok) => seen.push([err, ok]));
    expect(seen).toEqual([[null, true], [null, false], [null, true]]);
  });
});
