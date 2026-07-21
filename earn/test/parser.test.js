'use strict';

const { numField, gpuName, gpuIndex, parseLine } = require('../src/shared/parser');

// Real sample lines from alpha-miner 1.8.6 stdout.
const STATUS = '2026-07-03T23:31:35.680Z level=INFO ver=1.8.6 gpu=0:NVIDIA GeForce RTX 4090 component=miner status attempts=100 hits=3 accepted=5 rejected=1 dropped=0 hashrate_th_s=286.86 tmac_s=286.86 share_equiv_th_s=332.01 ctemp=71c cclk=2355MHz mclk=10251MHz power=449W';
const CONNECTED = '2026-07-03T23:31:13.958Z level=INFO ver=1.8.6 gpu=0:NVIDIA GeForce RTX 4090 component=pool connected host=us2.alphapool.tech port=5566 tls=false';
const CUDA = '2026-07-03T23:31:13.794Z level=INFO ver=1.8.6 gpu=system component=cuda scheduling=blocking-sync';

describe('numField', () => {
  test('reads an integer field', () => {
    expect(numField('a=1 accepted=5 b=2', 'accepted')).toBe(5);
  });
  test('reads a float and ignores a trailing unit', () => {
    expect(numField(STATUS, 'hashrate_th_s')).toBe(286.86);
    expect(numField(STATUS, 'power')).toBe(449);
  });
  test('returns null when the field is absent', () => {
    expect(numField('a=1', 'missing')).toBeNull();
  });
});

describe('gpuName', () => {
  test('extracts the device name and strips the index', () => {
    expect(gpuName(STATUS)).toBe('NVIDIA GeForce RTX 4090');
  });
  test('treats the "system" placeholder as no GPU', () => {
    expect(gpuName(CUDA)).toBeNull();
  });
  test('returns null when there is no gpu field', () => {
    expect(gpuName('level=INFO component=pool connected')).toBeNull();
  });
});

describe('gpuIndex', () => {
  test('reads the 0-based card index from gpu=<index>:<name>', () => {
    expect(gpuIndex(STATUS)).toBe(0);
    expect(gpuIndex('ver=1.8.6 gpu=3:NVIDIA GeForce RTX 4060 Ti component=miner')).toBe(3);
  });
  test('returns null with no index (gpu=system, name-only, or no gpu field)', () => {
    expect(gpuIndex(CUDA)).toBeNull();
    expect(gpuIndex('gpu=NVIDIA GeForce RTX 4090 component=miner')).toBeNull();
    expect(gpuIndex('component=pool connected')).toBeNull();
  });
});

describe('parseLine', () => {
  test('parses a miner status line into card index + hashrate + cumulative counts + gpu', () => {
    expect(parseLine(STATUS)).toEqual({
      type: 'status',
      gpuIndex: 0,
      hashrate: 286.86,
      accepted: 5,
      rejected: 1,
      power: 449,
      gpu: 'NVIDIA GeForce RTX 4090',
    });
  });

  test('parses a pool connection line', () => {
    expect(parseLine(CONNECTED)).toEqual({
      type: 'connected',
      gpuIndex: 0,
      endpoint: 'us2.alphapool.tech:5566',
      gpu: 'NVIDIA GeForce RTX 4090',
    });
  });

  test('returns null for unrecognized, empty and nullish lines', () => {
    expect(parseLine(CUDA)).toBeNull();
    expect(parseLine('   ')).toBeNull();
    expect(parseLine('')).toBeNull();
    expect(parseLine(null)).toBeNull();
    expect(parseLine(undefined)).toBeNull();
  });
});
