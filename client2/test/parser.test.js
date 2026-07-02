'use strict';

const { parseHashrateValue, parseLine } = require('../src/shared/parser');

describe('parseHashrateValue', () => {
  test('converts units to TH/s', () => {
    expect(parseHashrateValue('1', 'TH/s')).toBe(1);
    expect(parseHashrateValue('1000', 'GH/s')).toBeCloseTo(1, 9);
    expect(parseHashrateValue('2', 'MH/s')).toBeCloseTo(2e-6, 12);
    expect(parseHashrateValue('5', 'kH/s')).toBeCloseTo(5e-9, 15);
    expect(parseHashrateValue('7', 'H/s')).toBeCloseTo(7e-12, 18);
  });

  test('unknown unit assumed TH/s, bad number is zero', () => {
    expect(parseHashrateValue('5', 'PH/s')).toBe(5);
    expect(parseHashrateValue('abc', 'TH/s')).toBe(0);
  });
});

describe('parseLine', () => {
  test('returns null for empty/blank/null input', () => {
    expect(parseLine('')).toBeNull();
    expect(parseLine(null)).toBeNull();
    expect(parseLine('   ')).toBeNull();
  });

  test('parses a connection line with a worker', () => {
    expect(parseLine('connected to us2.alphapool.tech:5566 · worker rig01')).toEqual({
      type: 'connected', endpoint: 'us2.alphapool.tech:5566', worker: 'rig01',
    });
  });

  test('parses a connection line without a worker', () => {
    expect(parseLine('connected to eu1.alphapool.tech:5566')).toEqual({
      type: 'connected', endpoint: 'eu1.alphapool.tech:5566', worker: null,
    });
  });

  test('parses accepted and rejected shares', () => {
    expect(parseLine('accepted share #14,820 diff 524.3K gpu0 12ms')).toEqual({
      type: 'share', status: 'accepted', index: 14820,
    });
    expect(parseLine('rejected share')).toEqual({
      type: 'share', status: 'rejected', index: null,
    });
  });

  test('parses a full hashrate report with load and power', () => {
    expect(parseLine('hashrate 354.1 TH/s load 92% 318W')).toEqual({
      type: 'hashrate', hashrate: 354.1, load: 92, power: 318,
    });
  });

  test('parses a bare hashrate with unit conversion and null load/power', () => {
    expect(parseLine('speed 1000 GH/s')).toEqual({
      type: 'hashrate', hashrate: 1, load: null, power: null,
    });
  });

  test('returns null for unrecognized lines', () => {
    expect(parseLine('just some random log output')).toBeNull();
  });
});
