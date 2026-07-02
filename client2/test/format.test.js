'use strict';

const { pad2, formatUptime, formatHashrate, formatInt, formatLogTime } = require('../src/shared/format');

describe('format', () => {
  test('pad2 zero-pads', () => {
    expect(pad2(5)).toBe('05');
    expect(pad2(12)).toBe('12');
  });

  test('formatUptime renders h/m/s, omitting zero hours and clamping', () => {
    expect(formatUptime(8048)).toBe('2h 14m 08s');
    expect(formatUptime(65)).toBe('01m 05s');
    expect(formatUptime(-5)).toBe('00m 00s');
    expect(formatUptime('nan')).toBe('00m 00s');
  });

  test('formatHashrate fixes to one decimal', () => {
    expect(formatHashrate(354.137)).toBe('354.1');
    expect(formatHashrate('x')).toBe('0.0');
  });

  test('formatInt groups thousands', () => {
    expect(formatInt(14820)).toBe('14,820');
    expect(formatInt('x')).toBe('0');
  });

  test('formatLogTime accepts Date and timestamp', () => {
    expect(formatLogTime(new Date(0))).toMatch(/^\d{1,2}:\d{2}:\d{2}$/);
    expect(formatLogTime(0)).toMatch(/^\d{1,2}:\d{2}:\d{2}$/);
  });
});
