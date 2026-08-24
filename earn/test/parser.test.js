'use strict';

const { parseStatsRow, parseLine } = require('../src/shared/parser');

// Every fixture below is a line captured verbatim from a real PeakMiner 2.11.0
// run against us.pearl.herominers.com:1200 on an RTX 4090 — including the two
// failure cases, which were forced by pointing the miner at a hostname that does
// not resolve and at a closed local port. Fixtures invented from documentation
// are how the previous engine's parser came to be written against a format the
// binary had already stopped emitting, so these are kept as-is on purpose.
const CONNECTED = '2026-08-23 23:38:40  INFO connected us.pearl.herominers.com:1200  diff —  ping 1059ms';
const ACCEPTED = '2026-08-23 23:30:25 accepted   GPU 0  lat 82ms  diff 9.01 PH  effort 50%';
const ROW = '  0  RTX 4090  296.5 TH/s       3 / 0      78°C   48%   449W  660.4 GH/W  10251MHz   2340MHz';
const TOTAL = 'Total          296.5 TH/s       3 / 0     eff 100.0%    449W  660.4 GH/W';
const DNS = '2026-08-23 23:39:42 ERROR failed to connect to no-such-pool.invalid:1200: No such host is known. (os error 11001)';
const REFUSED = '2026-08-23 23:40:09 ERROR failed to connect to 127.0.0.1:59999: '
  + 'No connection could be made because the target machine actively refused it. (os error 10061)';

describe('parseLine — pool connection', () => {
  test('reads the endpoint off the connect line', () => {
    expect(parseLine(CONNECTED)).toEqual({
      type: 'connected', gpuIndex: null, endpoint: 'us.pearl.herominers.com:1200', gpu: null,
    });
  });

  // At connect time the pool has not assigned a difficulty and the field is a
  // literal em dash. Reading it would put NaN on the dashboard.
  test('does not try to read the placeholder difficulty', () => {
    expect(parseLine(CONNECTED)).not.toHaveProperty('difficulty');
  });
});

describe('parseLine — shares', () => {
  test('an accepted share names its card', () => {
    expect(parseLine(ACCEPTED)).toEqual({ type: 'share', gpuIndex: 0, accepted: true });
  });

  test('a second card is attributed to that card', () => {
    expect(parseLine('2026-08-23 23:30:25 accepted   GPU 3  lat 82ms  diff 9.01 PH  effort 50%'))
      .toEqual({ type: 'share', gpuIndex: 3, accepted: true });
  });
});

describe('parseLine — status table', () => {
  test('parses a card row into hashrate, shares and telemetry', () => {
    expect(parseLine(ROW)).toEqual({
      type: 'status',
      gpuIndex: 0,
      hashrate: 296.5,
      accepted: 3,
      rejected: 0,
      power: 449,
      temp: 78,
      gpu: 'RTX 4090',
    });
  });

  // The Total row repeats the same hashrate and share counts with no index. A
  // multi-GPU rig that counted it would double every figure it displays.
  test('skips the Total row', () => {
    expect(parseLine(TOTAL)).toBeNull();
  });

  test('normalises every hashrate unit to TH/s', () => {
    const at = (s) => parseStatsRow('0  GPU  ' + s + '   1 / 0').hashrate;
    expect(at('296.5 TH/s')).toBeCloseTo(296.5, 6);
    expect(at('1.0 PH/s')).toBeCloseTo(1e3, 6);
    expect(at('2.0 EH/s')).toBeCloseTo(2e6, 6);
    expect(at('500 GH/s')).toBeCloseTo(0.5, 6);
    expect(at('500 MH/s')).toBeCloseTo(5e-4, 6);
    expect(at('500 KH/s')).toBeCloseTo(5e-7, 6);
    expect(at('500 H/s')).toBeCloseTo(5e-10, 6);
  });

  // Read outwards from the hashrate rather than matching fixed columns: a rig
  // that reports no fan, no clocks or no temperature must still yield its
  // hashrate instead of failing the whole row and showing zero.
  test('survives a row missing its optional columns', () => {
    expect(parseStatsRow('1  RTX 3090  100.0 TH/s   9 / 2')).toEqual({
      type: 'status',
      gpuIndex: 1,
      hashrate: 100,
      accepted: 9,
      rejected: 2,
      power: null,
      temp: null,
      gpu: 'RTX 3090',
    });
  });

  test('a row with no share pair leaves the counters null rather than zero', () => {
    const r = parseStatsRow('0  RTX 4090  296.5 TH/s');
    expect(r.accepted).toBeNull();
    expect(r.rejected).toBeNull();
    expect(r.hashrate).toBeCloseTo(296.5, 6);
  });

  // The clock columns are bare integers too, so a naive "last three numbers"
  // read would report 10251 accepted shares and 2340 rejected.
  test('takes shares from the ok/inv pair, not from the trailing clocks', () => {
    const r = parseLine(ROW);
    expect(r.accepted).toBe(3);
    expect(r.rejected).toBe(0);
  });

  test('rejects non-rows', () => {
    expect(parseStatsRow('not a row at all')).toBeNull();
    expect(parseStatsRow('0  RTX 4090  no hashrate here')).toBeNull();
  });

  test('a nameless row still reports its hashrate', () => {
    expect(parseStatsRow('0  296.5 TH/s   1 / 0').gpu).toBeNull();
  });
});

describe('parseLine — connection failures', () => {
  // The endpoint is host:port, so the reason begins after the SECOND colon. A
  // lazy match splits at the first and reports the host as
  // "no-such-pool.invalid" with the port glued to the front of the reason.
  test('separates a host:port endpoint from the reason', () => {
    expect(parseLine(DNS)).toEqual({
      type: 'connect-failed',
      endpoint: 'no-such-pool.invalid:1200',
      reason: 'No such host is known. (os error 11001)',
      dns: true,
    });
  });

  test('flags a refused connection as reachable-but-closed, not DNS', () => {
    const e = parseLine(REFUSED);
    expect(e.type).toBe('connect-failed');
    expect(e.endpoint).toBe('127.0.0.1:59999');
    expect(e.dns).toBe(false);
  });

  // Eight identical retry lines that never say name resolution is the problem is
  // exactly what two users reported. The flag is what lets the UI say it.
  test('recognises the other spellings of a resolver failure', () => {
    const dnsOf = (reason) => parseLine('ERROR failed to connect to h:1: ' + reason).dns;
    expect(dnsOf('DNS lookup failed')).toBe(true);
    expect(dnsOf('Name or service not known')).toBe(true);
    expect(dnsOf('name not known')).toBe(true);
    expect(dnsOf('getaddrinfo ENOTFOUND')).toBe(true);
    expect(dnsOf('os error 11001')).toBe(true);
    expect(dnsOf('Connection reset by peer')).toBe(false);
  });
});

describe('parseLine — passthrough', () => {
  test('returns null for lines with nothing structured in them', () => {
    expect(parseLine('2026-08-23 23:38:40  INFO new job 00000000_2097152')).toBeNull();
    expect(parseLine('2026-08-23 23:38:40  INFO vardiff 9.01 PH')).toBeNull();
    expect(parseLine('  pool us.pearl.herominers.com:1200    uptime 00:00:10    ping 1059ms')).toBeNull();
    expect(parseLine('')).toBeNull();
    expect(parseLine(null)).toBeNull();
    expect(parseLine(undefined)).toBeNull();
  });

  // The banner is drawn with box characters and the licence line carries
  // parentheses and colons; neither may be mistaken for a row or an error.
  test('ignores the startup banner', () => {
    expect(parseLine('|  _ \\ ___  __ _| | _|  \\/  (_)_ __   ___ _ __')).toBeNull();
    expect(parseLine('# high-performance GPU miner · v2.11.0')).toBeNull();
    expect(parseLine('───────────────────────────────────────────────')).toBeNull();
  });

  test('strips ANSI colour before matching', () => {
    expect(parseLine('[32m' + ACCEPTED + '[0m'))
      .toEqual({ type: 'share', gpuIndex: 0, accepted: true });
  });
});
