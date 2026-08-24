'use strict';

const { STATUS_INTERVAL_SECS, resolveBinary, buildArgs } = require('../src/shared/minerArgs');
const { DEFAULTS, endpointFor } = require('../src/shared/config');

const ADDR = 'prl1px5ervx6ftaegmdhqa5ajemh20j2uw7l9jt5j5s97rljp72yt3s8qncrxud';
const MDL = 'mdl1pl80mdy0culfn3g7jl3paa5ccnc8gkkfmkc6t2x0q6rmvd9dpu5wsk0v3z8';

// Read the value that follows a flag, so assertions do not depend on argument
// order and a reordering does not silently pass while the vector changed.
function val(args, flag) {
  const i = args.indexOf(flag);
  return i === -1 ? undefined : args[i + 1];
}

describe('resolveBinary', () => {
  test('an explicit path always wins', () => {
    expect(resolveBinary('/opt/peakminer', 'linux')).toBe('/opt/peakminer');
    expect(resolveBinary('C:\\pm.exe', 'win32')).toBe('C:\\pm.exe');
  });

  test('falls back to the platform default', () => {
    expect(resolveBinary(null, 'win32')).toBe('peakminer.exe');
    expect(resolveBinary(null, 'linux')).toBe('peakminer');
    expect(resolveBinary(undefined, undefined)).toBe('peakminer');
  });
});

describe('buildArgs', () => {
  test('builds the documented PeakMiner vector', () => {
    const args = buildArgs({ address: ADDR, worker: 'rig01', region: 'us1', difficulty: 524288 });
    expect(val(args, '-c')).toBe('pearl');
    expect(val(args, '-o')).toBe('stratum+tcp://us.pearl.herominers.com:1200');
    expect(val(args, '-u')).toBe(ADDR);
    expect(val(args, '-w')).toBe('rig01');
    expect(val(args, '-p')).toBe('x;d=524288');
    expect(val(args, '-d')).toBe('0');
  });

  // Upstream's default status interval is 60s, which would leave the dashboard
  // blank for a minute after every start.
  test('asks for a status table often enough to drive a live UI', () => {
    expect(STATUS_INTERVAL_SECS).toBeLessThanOrEqual(15);
    expect(val(buildArgs({ address: ADDR }), '-i')).toBe(String(STATUS_INTERVAL_SECS));
  });

  // Tips are prose printed under the table; upstream documents this flag for
  // log scrapers specifically.
  test('suppresses the usage tips that would otherwise land in the log', () => {
    expect(buildArgs({ address: ADDR })).toContain('--no-tips');
  });

  // The API defaults to ON, listening on 127.0.0.1:4068. We read the log, so a
  // listening socket the user never asked for is pure liability — and two rigs
  // on one box would collide on the port.
  test('disables the HTTP stats API', () => {
    expect(val(buildArgs({ address: ADDR }), '-a')).toBe('0');
  });

  test('defaults worker, difficulty and region when unset', () => {
    const args = buildArgs({ address: ADDR });
    expect(val(args, '-w')).toBe(DEFAULTS.worker);
    expect(val(args, '-p')).toBe('x;d=' + DEFAULTS.difficulty);
    expect(val(args, '-o')).toBe('stratum+tcp://' + endpointFor(DEFAULTS.region));
  });

  test('an empty worker is omitted rather than sent blank', () => {
    expect(buildArgs({ address: ADDR, worker: '' })).not.toContain('-w');
  });

  test('selects the requested card', () => {
    expect(val(buildArgs({ address: ADDR, gpuIndex: 2 }), '-d')).toBe('2');
    expect(val(buildArgs({ address: ADDR, gpuIndex: 0 }), '-d')).toBe('0');
  });

  // HeroMiners documents its merge-mining login as PRL+MDL.WORKER, which is
  // exactly what combinePayoutAddress produces — and -u is sent verbatim, so it
  // survives intact.
  test('carries a merge-mining address through unchanged', () => {
    expect(val(buildArgs({ address: ADDR, mdlAddress: MDL }), '-u')).toBe(ADDR + '+' + MDL);
  });

  test('a malformed MDL address is dropped rather than sent', () => {
    expect(val(buildArgs({ address: ADDR, mdlAddress: 'not-an-address' }), '-u')).toBe(ADDR);
  });

  // A user pasting `stratum+tcp://host:port` into the endpoint override must not
  // produce `stratum+tcp://stratum+tcp://host:port`, which is the shape that
  // made the previous engine try to resolve the scheme as part of the hostname.
  test('normalises a scheme already present in an endpoint override', () => {
    const args = buildArgs({ address: ADDR, endpoint: 'stratum+tcp://pool.example:1200' });
    expect(val(args, '-o')).toBe('stratum+tcp://pool.example:1200');
  });

  test('an endpoint override beats the region', () => {
    const args = buildArgs({ address: ADDR, region: 'us1', endpoint: 'pool.example:9999' });
    expect(val(args, '-o')).toBe('stratum+tcp://pool.example:9999');
  });

  // There is exactly one URL argument and no port flag, so the doubled-port
  // failure that produced "pool us2.alphapool.tech:5566:5566" cannot recur.
  test('never emits a separate port argument', () => {
    const args = buildArgs({ address: ADDR, region: 'us1' });
    expect(args).not.toContain('--port');
    expect(args.filter((a) => a === '-o')).toHaveLength(1);
  });

  test('tolerates being called with nothing at all', () => {
    const args = buildArgs();
    expect(val(args, '-c')).toBe('pearl');
    expect(val(args, '-u')).toBe('');
  });
});
