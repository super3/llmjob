'use strict';

const {
  pad2, formatUptime, formatHashrate, formatInt, formatLogTime, formatDeviceLabel,
} = require('../src/shared/format');

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

// The device label names every card that is mining. A rig mines on all of them,
// so one name is the honest answer only when there is one card -- showing one
// name for two working cards is the same lie issue #226 was about.
describe('formatDeviceLabel', () => {
  test('one card is its own name', () => {
    expect(formatDeviceLabel(['NVIDIA GeForce RTX 4090'])).toBe('NVIDIA GeForce RTX 4090');
  });

  // A 13-card rig would otherwise print the same string thirteen times.
  test('identical cards are counted, not repeated', () => {
    expect(formatDeviceLabel(['RTX 4090', 'RTX 4090'])).toBe('2x RTX 4090');
    expect(formatDeviceLabel(Array(13).fill('RTX 3060'))).toBe('13x RTX 3060');
  });

  test('different cards are listed', () => {
    expect(formatDeviceLabel(['NVIDIA RTX PRO 4500 Blackwell', 'NVIDIA GeForce RTX 4070']))
      .toBe('NVIDIA RTX PRO 4500 Blackwell + NVIDIA GeForce RTX 4070');
  });

  // Before the first status every bucket is unnamed, and the caller falls back
  // to what it detected. Null says "nothing to name" rather than inventing one.
  test('is null when there is nothing to name', () => {
    expect(formatDeviceLabel([])).toBeNull();
    expect(formatDeviceLabel([null, '', '   '])).toBeNull();
    expect(formatDeviceLabel(null)).toBeNull();
    expect(formatDeviceLabel(undefined)).toBeNull();
  });

  test('ignores the cards that have no name yet', () => {
    expect(formatDeviceLabel(['RTX 4090', null])).toBe('RTX 4090');
  });
});
