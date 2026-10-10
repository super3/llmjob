'use strict';

const http = require('http');
const { createLanShare, lanAddresses } = require('../src/main/lanShare');
const { createServeGate } = require('../src/main/autoGate');

const v4 = (address, internal = false) => ({ family: 'IPv4', address, internal });

describe('lanAddresses', () => {
  test('keeps the addresses other devices can reach, home networks first', () => {
    const ifaces = {
      'vEthernet (WSL)': [v4('172.24.80.1')],
      Tailscale: [v4('100.101.102.103')],
      Loopback: [v4('127.0.0.1', true)],
      Ethernet: [{ family: 'IPv6', address: 'fe80::1', internal: false }, v4('192.168.0.220')],
      'No DHCP': [v4('169.254.10.20')],
      Office: [v4('10.0.0.7')],
      // 172.15 is not one of the private 172.16-31 ranges.
      Odd: [v4('172.15.0.1')],
    };
    expect(lanAddresses(ifaces)).toEqual(['192.168.0.220', '10.0.0.7', '172.24.80.1', '100.101.102.103', '172.15.0.1']);
  });

  test('copes with no interfaces at all', () => {
    expect(lanAddresses(null)).toEqual([]);
    expect(lanAddresses({ eth0: null })).toEqual([]);
  });
});

// A gate that records what it was built with instead of opening a socket.
function fakeGates() {
  const made = [];
  const makeGate = (opts) => {
    const g = { opts, start: jest.fn(() => g), stop: jest.fn() };
    made.push(g);
    return g;
  };
  return { made, makeGate };
}

describe('createLanShare', () => {
  const ifaces = () => ({ Ethernet: [v4('192.168.0.220')] });

  test('opens one gate while wanted, and closes it when not', () => {
    const { made, makeGate } = fakeGates();
    const log = jest.fn();
    const onChange = jest.fn();
    const share = createLanShare({ upstreamUrl: () => null, isLlmReady: () => true, log, onChange, makeGate, interfaces: ifaces });
    expect(share.status()).toBeNull();

    share.sync(true, { name: 'gemma', ctxSize: 32768 });
    share.sync(true, { name: 'gemma', ctxSize: 32768 });
    expect(made).toHaveLength(1);
    expect(made[0].start).toHaveBeenCalledTimes(1);
    expect(made[0].opts).toMatchObject({ port: 8000, modelName: 'gemma', ctxSize: 32768 });
    // No host: the gate's own default, every interface.
    expect(made[0].opts.host).toBeUndefined();
    expect(log).toHaveBeenCalledWith('sharing the local LLM on your network at http://192.168.0.220:8000/v1');
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(share.status()).toEqual({ urls: ['http://192.168.0.220:8000/v1'], error: null });

    share.sync(false);
    share.sync(false);
    expect(made[0].stop).toHaveBeenCalledTimes(1);
    expect(log).toHaveBeenLastCalledWith('stopped sharing the local LLM on the network');
    expect(onChange).toHaveBeenCalledTimes(2);
    expect(share.status()).toBeNull();
  });

  test('forwards to the port the fleet really bound, or 8080 before it says', () => {
    const { made, makeGate } = fakeGates();
    let url = null;
    const share = createLanShare({ upstreamUrl: () => url, makeGate, interfaces: ifaces });
    share.sync(true);
    expect(made[0].opts.modelName).toBeUndefined();
    expect(made[0].opts.upstreamPort()).toBe(8080);
    url = 'http://127.0.0.1:8081';
    expect(made[0].opts.upstreamPort()).toBe(8081);
    url = 'http://127.0.0.1';
    expect(made[0].opts.upstreamPort()).toBe(8080);
  });

  test('says when the computer has no network address', () => {
    const { makeGate } = fakeGates();
    const log = jest.fn();
    const share = createLanShare({ upstreamUrl: () => null, log, makeGate, interfaces: () => ({}) });
    share.sync(true);
    expect(log).toHaveBeenCalledWith('sharing the local LLM on your network at port 8000 (no network address yet)');
    expect(share.status()).toEqual({ urls: [], error: null });
  });

  test('reports a port that is taken, and tries again on the next sync', () => {
    const { made, makeGate } = fakeGates();
    const onChange = jest.fn();
    const share = createLanShare({ upstreamUrl: () => null, onChange, makeGate, interfaces: ifaces });
    share.sync(true);
    made[0].opts.onListenError(Object.assign(new Error('listen EADDRINUSE'), { code: 'EADDRINUSE' }));
    expect(made[0].stop).toHaveBeenCalled();
    expect(share.status()).toEqual({ urls: [], error: 'Port 8000 is already in use' });
    expect(onChange).toHaveBeenCalledTimes(2);

    share.sync(true);
    expect(made).toHaveLength(2);
    expect(share.status()).toEqual({ urls: ['http://192.168.0.220:8000/v1'], error: null });
    made[1].opts.onListenError(new Error('permission denied'));
    expect(share.status()).toEqual({ urls: [], error: 'permission denied' });

    // Switching off clears the error.
    share.sync(false);
    expect(share.status()).toBeNull();
    expect(onChange).toHaveBeenCalledTimes(5);
  });

  test('ignores a late error from a gate it already replaced', () => {
    const { made, makeGate } = fakeGates();
    const share = createLanShare({ upstreamUrl: () => null, makeGate, interfaces: ifaces });
    share.sync(true);
    share.sync(false);
    share.sync(true);
    made[0].opts.onListenError(Object.assign(new Error('listen EADDRINUSE'), { code: 'EADDRINUSE' }));
    expect(made[1].stop).not.toHaveBeenCalled();
    expect(share.status()).toEqual({ urls: ['http://192.168.0.220:8000/v1'], error: null });
  });

  test('constructs with no options', () => {
    expect(createLanShare().status()).toBeNull();
  });
});

describe('createLanShare with real sockets', () => {
  let upstream; let upPort;

  beforeAll(async () => {
    upstream = http.createServer((req, res) => {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ path: req.url }));
    });
    await new Promise((r) => upstream.listen(0, '127.0.0.1', r));
    upPort = upstream.address().port;
  });
  afterAll(() => upstream.close());

  // A port nothing holds right now.
  async function freePort() {
    const s = http.createServer();
    await new Promise((r) => s.listen(0, '0.0.0.0', r));
    const { port } = s.address();
    await new Promise((r) => s.close(r));
    return port;
  }

  function post(port, path) {
    return new Promise((resolve, reject) => {
      const req = http.request({ host: '127.0.0.1', port, path, method: 'POST' }, (r) => {
        const c = []; r.on('data', (d) => c.push(d));
        r.on('end', () => resolve({ status: r.statusCode, body: Buffer.concat(c).toString() }));
      });
      req.on('error', reject);
      req.end('{}');
    });
  }

  test('serves llama-server on every interface, on its own port', async () => {
    const port = await freePort();
    const gates = [];
    const share = createLanShare({
      port,
      upstreamUrl: () => 'http://127.0.0.1:' + upPort,
      isLlmReady: () => true,
      makeGate: (o) => { const g = createServeGate(o); gates.push(g); return g; },
    });
    share.sync(true, { name: 'gemma', ctxSize: 32768 });
    const server = gates[0].server.server;
    await new Promise((r) => server.once('listening', r));
    expect(server.address().address).toBe('0.0.0.0');

    const res = await post(port, '/v1/chat/completions');
    expect(res.status).toBe(200);
    expect(JSON.parse(res.body)).toEqual({ path: '/v1/chat/completions' });

    share.sync(false);
    await new Promise((r) => server.once('close', r));
  });

  test('reports a port another program holds', async () => {
    const blocker = http.createServer();
    await new Promise((r) => blocker.listen(0, '0.0.0.0', r));
    const { port } = blocker.address();
    const onChange = jest.fn();
    const share = createLanShare({ port, upstreamUrl: () => null, isLlmReady: () => true, onChange });
    share.sync(true);
    await new Promise((r) => { onChange.mockImplementation(() => { if (share.status().error) r(); }); });
    expect(share.status()).toEqual({ urls: [], error: 'Port ' + port + ' is already in use' });
    await new Promise((r) => blocker.close(r));
  });
});
