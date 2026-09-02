'use strict';

const http = require('http');
const { LlmGateServer } = require('../src/main/llmGateServer');
const { LlmGate } = require('../src/shared/llmGate');

function get(port, path) {
  return new Promise((resolve, reject) => {
    http.get({ host: '127.0.0.1', port, path }, (r) => {
      const c = []; r.on('data', (d) => c.push(d));
      r.on('end', () => resolve({ status: r.statusCode, body: Buffer.concat(c).toString() }));
    }).on('error', reject);
  });
}
function post(port, path, payload) {
  return new Promise((resolve, reject) => {
    const req = http.request({ host: '127.0.0.1', port, path, method: 'POST',
      headers: { 'Content-Type': 'application/json' } }, (r) => {
      const c = []; r.on('data', (d) => c.push(d));
      r.on('end', () => resolve({ status: r.statusCode, body: Buffer.concat(c).toString() }));
    });
    req.on('error', reject);
    req.end(payload);
  });
}

describe('LlmGateServer', () => {
  let upstream; let upPort; let gs; let gsPort; let started;

  beforeAll(async () => {
    upstream = http.createServer((req, res) => {
      if (req.url === '/slow-stream') {
        res.writeHead(200, { 'Content-Type': 'text/event-stream' });
        res.write('data: one\n\n');
        setTimeout(() => { res.write('data: two\n\n'); res.end(); }, 30);
        return;
      }
      const c = []; req.on('data', (d) => c.push(d));
      req.on('end', () => {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ ok: true, saw: Buffer.concat(c).toString(), path: req.url }));
      });
    });
    await new Promise((r) => upstream.listen(0, '127.0.0.1', r));
    upPort = upstream.address().port;

    started = 0;
    let ready = false;
    const gate = new LlmGate({
      idleMs: 10_000,
      isLlmReady: () => ready,
      // A real load takes seconds; the delay here is what proves the request is
      // HELD rather than refused while the model comes up.
      startLlm: () => new Promise((r) => setTimeout(() => { started += 1; ready = true; r(); }, 40)),
      stopLlm: async () => { ready = false; },
      startMiner: async () => {}, stopMiner: async () => {},
    });
    gs = new LlmGateServer({ port: 0, upstreamPort: upPort, gate, modelName: 'Qwen3.8-27B-UD-Q4_K_XL' });
    gs.server = http.createServer((req, res) => { gs._handle(req, res).catch(() => res.destroy()); });
    await new Promise((r) => gs.server.listen(0, '127.0.0.1', r));
    gsPort = gs.server.address().port;
  });

  afterAll(() => { if (gs.server) gs.server.close(); upstream.close(); });

  test('a probe is answered from gate state and does NOT start the model', async () => {
    const r = await get(gsPort, '/health');
    expect(r.status).toBe(200);
    expect(JSON.parse(r.body)).toMatchObject({ status: 'loading' });
    expect(started).toBe(0);   // the point: no wake
  });

  test('/v1/models reports the model name while the model is stopped', async () => {
    const r = await get(gsPort, '/v1/models');
    expect(JSON.parse(r.body).data[0].id).toBe('Qwen3.8-27B-UD-Q4_K_XL');
    expect(started).toBe(0);
  });

  test('an inference request is held during the load, then forwarded with its body', async () => {
    const r = await post(gsPort, '/v1/chat/completions', JSON.stringify({ hello: 'world' }));
    expect(r.status).toBe(200);
    const j = JSON.parse(r.body);
    expect(j.ok).toBe(true);
    expect(JSON.parse(j.saw)).toEqual({ hello: 'world' });   // body survived the wait
    expect(j.path).toBe('/v1/chat/completions');
    expect(started).toBe(1);
  });

  test('once up, a probe passes through to the real server', async () => {
    const r = await get(gsPort, '/health');
    expect(JSON.parse(r.body).path).toBe('/health');   // came from upstream, not the gate
  });

  test('a streamed response is piped, not buffered', async () => {
    const r = await get(gsPort, '/slow-stream');
    expect(r.body).toBe('data: one\n\ndata: two\n\n');
  });
});

const { pickHeaders } = require('../src/main/llmGateServer');

describe('pickHeaders', () => {
  test('drops hop-by-hop headers and host, keeps the rest', () => {
    const out = pickHeaders({
      host: 'x', connection: 'keep-alive', 'transfer-encoding': 'chunked',
      'content-type': 'application/json', authorization: 'Bearer k',
    });
    expect(out).toEqual({ 'content-type': 'application/json', authorization: 'Bearer k' });
  });
  test('tolerates no headers', () => { expect(pickHeaders()).toEqual({}); });
});

describe('LlmGateServer edges', () => {
  test('an unlisted passive path reports gate state', async () => {
    const gate = new LlmGate({ isLlmReady: () => false });
    const gs = new LlmGateServer({ port: 0, gate });
    const s = http.createServer((req, res) => gs._handle(req, res));
    await new Promise((r) => s.listen(0, '127.0.0.1', r));
    const r = await get(s.address().port, '/props');
    expect(JSON.parse(r.body)).toEqual({ gate: 'MINING', llm_up: false });
    s.close();
  });

  test('an upstream that is not listening becomes a 502, not a hang', async () => {
    const gate = new LlmGate({ isLlmReady: () => true, startLlm: async () => {} });
    // port 1 is not listening
    const gs = new LlmGateServer({ port: 0, upstreamPort: 1, gate });
    const s = http.createServer((req, res) => gs._handle(req, res));
    await new Promise((r) => s.listen(0, '127.0.0.1', r));
    const r = await post(s.address().port, '/v1/chat/completions', '{}');
    expect(r.status).toBe(502);
    expect(JSON.parse(r.body).error.message).toMatch(/upstream/);
    s.close();
  });

  test('an LLM that will not start becomes a 503 rather than a dropped connection', async () => {
    const gate = new LlmGate({
      isLlmReady: () => false,
      startLlm: async () => { throw new Error('no VRAM'); },
      stopMiner: async () => {},
    });
    const gs = new LlmGateServer({ port: 0, gate });
    const s = http.createServer((req, res) => gs._handle(req, res));
    await new Promise((r) => s.listen(0, '127.0.0.1', r));
    const r = await post(s.address().port, '/v1/chat/completions', '{}');
    expect(r.status).toBe(503);
    expect(JSON.parse(r.body).error.message).toMatch(/no VRAM/);
    s.close();
  });

  test('start() binds and arms the idle timer; stop() unwinds both', async () => {
    let released = 0;
    const gate = new LlmGate({
      idleMs: 0, isLlmReady: () => true,
      startLlm: async () => {}, stopLlm: async () => { released += 1; },
      startMiner: async () => {}, stopMiner: async () => {},
    });
    await gate.ensureServing();
    const gs = new LlmGateServer({ port: 0, gate, log: () => {} }).start();
    await new Promise((r) => setTimeout(r, 2300));   // one tick of the 2s timer
    expect(released).toBe(1);
    gs.stop();
    expect(gs.server).toBeNull();
    expect(gs.timer).toBeNull();
  }, 10000);
});

describe('LlmGateServer via start()', () => {
  test('serves a real request through the bound server', async () => {
    const up = http.createServer((req, res) => {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end('{"via":"upstream"}');
    });
    await new Promise((r) => up.listen(0, '127.0.0.1', r));
    const gate = new LlmGate({
      idleMs: 60_000, isLlmReady: () => true,
      startLlm: async () => {}, stopLlm: async () => {},
      startMiner: async () => {}, stopMiner: async () => {},
    });
    const gs = new LlmGateServer({ port: 0, host: '127.0.0.1',
      upstreamPort: up.address().port, gate, log: () => {} }).start();
    await new Promise((r) => gs.server.on('listening', r));
    const r = await post(gs.server.address().port, '/v1/chat/completions', '{}');
    expect(JSON.parse(r.body)).toEqual({ via: 'upstream' });
    gs.stop(); up.close();
  }, 10000);
});

describe('LlmGateServer defaults', () => {
  test('constructs with no options at all', () => {
    const gs = new LlmGateServer();
    expect(gs.port).toBe(8000);          // the documented public port
    expect(gs.host).toBe('0.0.0.0');
    expect(gs.upstreamPort).toBe(8080);
    expect(gs.modelName).toBe('local');
    expect(gs.gate).toBeTruthy();        // builds its own gate when not given one
    expect(() => gs.log('x')).not.toThrow();
    gs.stop();                           // stopping before start must be safe
  });
});

describe('LlmGateServer internals', () => {
  const fakeRes = () => {
    const r = { headersSent: false, chunks: [], status: null, headers: null };
    r.writeHead = (s, h) => { r.status = s; r.headers = h; r.headersSent = true; };
    r.end = (b) => { if (b) r.chunks.push(String(b)); };
    r.on = () => {};
    return r;
  };

  test('a probe answered while the model IS up reports ok', () => {
    const gs = new LlmGateServer({ gate: new LlmGate({ isLlmReady: () => true }) });
    const res = fakeRes();
    gs._passive({ url: '/health' }, res);
    expect(JSON.parse(res.chunks[0]).status).toBe('ok');
  });

  test('a request with no url does not throw', () => {
    const gs = new LlmGateServer({ gate: new LlmGate({}) });
    const res = fakeRes();
    gs._passive({}, res);
    expect(res.status).toBe(200);
  });

  test('/models is answered as well as /v1/models', () => {
    const gs = new LlmGateServer({ gate: new LlmGate({}), modelName: 'M' });
    const res = fakeRes();
    gs._passive({ url: '/models' }, res);
    expect(JSON.parse(res.chunks[0]).data[0].id).toBe('M');
  });

  test('an upstream failure AFTER headers are sent does not try to write them again', async () => {
    // Streaming has already begun when the upstream dies: writeHead would throw.
    const gs = new LlmGateServer({ port: 0, upstreamPort: 1, gate: new LlmGate({}) });
    const res = fakeRes();
    res.headersSent = true;
    gs._forward({ method: 'POST', url: '/v1/chat/completions', headers: {}, pipe: () => {} }, res, '{}');
    await new Promise((r) => setTimeout(r, 60));
    expect(res.status).toBeNull();                       // no second writeHead
    expect(res.chunks.join('')).toMatch(/upstream/);
  });

  test('an unref-less timer handle is tolerated', () => {
    const real = global.setInterval;
    global.setInterval = () => ({});   // no unref, as under some fake-timer setups
    try {
      const gs = new LlmGateServer({ port: 0, host: '127.0.0.1', gate: new LlmGate({}) });
      expect(() => gs.start()).not.toThrow();
      gs.stop();
    } finally { global.setInterval = real; }
  });

  test('a release that rejects on the idle tick is swallowed, not thrown at the loop', async () => {
    const gate = new LlmGate({
      idleMs: 0, isLlmReady: () => true,
      startLlm: async () => {}, stopLlm: async () => { throw new Error('stop failed'); },
      startMiner: async () => {}, stopMiner: async () => {},
    });
    await gate.ensureServing();
    const gs = new LlmGateServer({ port: 0, host: '127.0.0.1', gate, log: () => {} }).start();
    await new Promise((r) => setTimeout(r, 2300));
    gs.stop();
  }, 10000);
});

describe('LlmGateServer remaining paths', () => {
  test('an idle tick with nothing to release does nothing', async () => {
    const gate = new LlmGate({ idleMs: 10_000, isLlmReady: () => false });   // MINING
    const gs = new LlmGateServer({ port: 0, host: '127.0.0.1', gate, log: () => {} }).start();
    await new Promise((r) => setTimeout(r, 2300));   // a tick where shouldRelease() is false
    expect(gate.state).toBe('MINING');
    gs.stop();
  }, 10000);

  test('a wake failure after headers are sent does not write them twice', async () => {
    const gate = new LlmGate({
      isLlmReady: () => false,
      startLlm: async () => { throw new Error('no VRAM'); },
      stopMiner: async () => {},
    });
    const gs = new LlmGateServer({ port: 0, gate });
    const res = {
      headersSent: true, chunks: [], status: null,
      writeHead: (s) => { res.status = s; },
      end: (b) => { if (b) res.chunks.push(String(b)); },
      on: () => {},
    };
    const req = Object.assign(
      (async function* () { /* empty body */ }()),
      { url: '/v1/chat/completions', method: 'POST', headers: {} },
    );
    await gs._handle(req, res);
    expect(res.status).toBeNull();                   // not written a second time
    expect(res.chunks.join('')).toMatch(/no VRAM/);
  });
});

describe('probes never refresh the idle clock', () => {
  test('polling /health while SERVING does not defer the flip back to mining', async () => {
    const up = http.createServer((req, res) => { res.writeHead(200); res.end('{"ok":1}'); });
    await new Promise((r) => up.listen(0, '127.0.0.1', r));
    let t = 0;
    const gate = new LlmGate({
      idleMs: 1000, now: () => t, isLlmReady: () => true,
      startLlm: async () => {}, stopLlm: async () => {},
      startMiner: async () => {}, stopMiner: async () => {},
    });
    await gate.ensureServing();
    const gs = new LlmGateServer({ port: 0, upstreamPort: up.address().port, gate });
    const s = http.createServer((req, res) => gs._handle(req, res));
    await new Promise((r) => s.listen(0, '127.0.0.1', r));

    t = 900;
    await get(s.address().port, '/health');     // a monitor polls
    t = 1100;
    // If the probe had counted as activity, idleFor() would have reset at t=900
    // and this would be false — which is how a polled node never stops serving.
    expect(gate.shouldRelease()).toBe(true);
    s.close(); up.close();
  }, 10000);
});
