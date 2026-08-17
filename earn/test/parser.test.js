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
      temp: 71,
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

// alpha-miner 1.9.4 dropped the key=value logs entirely and renders a live
// stats table. The rig still mined perfectly with the old parser — it just
// reported 0.0 TH/s and $0.00/day in the UI, which is why these are pinned
// against output captured from a real 5090 rather than invented.
describe('parseLine on the 1.9.4 stats table', () => {
  const ROW = ' #0  RTX 5090            313.95 TH/s   71C   44%     575W   0.546    2437    13801      2   0   0';

  test('reads hashrate, counters, power and temp from a device row', () => {
    expect(parseLine(ROW)).toEqual({
      type: 'status',
      gpuIndex: 0,
      hashrate: 313.95,
      accepted: 2,
      rejected: 0,
      power: 575,
      temp: 71,
      gpu: 'RTX 5090',
    });
  });

  // The table's own Total line carries no #N, so a multi-GPU rig accumulates
  // per card instead of double-counting the summary.
  test('ignores the Total row and the table rules', () => {
    expect(parseLine('     Total               313.95 TH/s                 575W   0.546          2   0   0')).toBeNull();
    expect(parseLine('────────────────────────────────')).toBeNull();
  });

  test('tracks the second card on a multi-GPU rig', () => {
    const r = parseLine(' #1  RTX 4090            286.86 TH/s   86C   61%     449W   0.638    2100    10501     41   2   0');
    expect(r.gpuIndex).toBe(1);
    expect(r.accepted).toBe(41);
    expect(r.rejected).toBe(2);
    expect(r.temp).toBe(86);
  });

  test('normalises non-terabyte hashrate units', () => {
    expect(parseLine('#0 A  500 GH/s 1 0 0').hashrate).toBeCloseTo(0.5, 6);
    expect(parseLine('#0 A  2000000 H/s 1 0 0').hashrate).toBeCloseTo(2e-6, 12);
  });

  // A row missing its fan/clock/counter columns must still yield the hashrate
  // rather than failing the whole line — the strict-layout version of this
  // parser is what silently showed zero.
  test('survives a row with columns missing', () => {
    expect(parseLine('#0  RTX 5090  120.5 TH/s')).toEqual({
      type: 'status', gpuIndex: 0, hashrate: 120.5,
      accepted: null, rejected: null, power: null, temp: null, gpu: 'RTX 5090',
    });
  });

  test('returns null for a #N line with no hashrate, and nulls an empty name', () => {
    expect(parseLine('#0  starting up')).toBeNull();
    expect(parseLine('#0 42.0 TH/s 1 0 0').gpu).toBeNull();
  });

  test('strips ANSI colour before matching', () => {
    const r = parseLine('\u001b[32m #0  RTX 5090  313.95 TH/s   71C   575W  2 0 0\u001b[0m');
    expect(r.hashrate).toBe(313.95);
    expect(r.gpu).toBe('RTX 5090');
  });

  test('reads the plainer 1.9.4 connection line', () => {
    expect(parseLine('[stratum] connected to us1.alphapool.tech:5566')).toEqual({
      type: 'connected', gpuIndex: null, endpoint: 'us1.alphapool.tech:5566', gpu: null,
    });
  });
});

// The engine retries every 5s and reprints the same line, naming neither the
// host it tried nor whether the name resolved — which is how a field report
// arrived as eight identical "No such host is known" lines. Classifying it lets
// the app say the useful thing once.
describe('parseLine on 1.9.4 connection failures', () => {
  test('flags a DNS failure however the platform words it', () => {
    for (const reason of [
      'DNS lookup failed: No such host is known.',
      'getaddrinfo ENOTFOUND us1.alphapool.tech',
      'Name or service not known',
    ]) {
      expect(parseLine('[stratum] connect failed: ' + reason)).toEqual({
        type: 'connect-failed', reason, dns: true,
      });
    }
  });

  // A refused connection is the pool's problem, not the rig's resolver, and must
  // not be reported as a DNS fault.
  test('a non-DNS failure is carried through unflagged', () => {
    expect(parseLine('[stratum] connect failed: connection refused'))
      .toEqual({ type: 'connect-failed', reason: 'connection refused', dns: false });
  });
});
