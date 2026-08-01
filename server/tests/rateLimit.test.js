// Per-client rate limiting for the endpoints that cost real money: the
// unauthenticated free-chat proxy (OpenRouter credit) and the OpenAI gateway
// (a node's GPU time). Before this there was no throttle anywhere on the server.
const request = require('supertest');
const express = require('express');
const { rateLimit, clientKey } = require('../src/middleware/rateLimit');

describe('rateLimit middleware', () => {
  // A deterministic clock so window rollover is tested without waiting.
  function fakeClock(start = 1000) {
    let t = start;
    return { now: () => t, advance: (ms) => { t += ms; } };
  }

  function appWith(limiter) {
    const app = express();
    app.use(express.json());
    app.post('/x', limiter, (req, res) => res.json({ ok: true }));
    return app;
  }

  it('allows requests up to the limit and rejects the next one with 429', async () => {
    const app = appWith(rateLimit({ windowMs: 60000, max: 2, keyFn: () => 'c1' }));

    await request(app).post('/x').expect(200);
    await request(app).post('/x').expect(200);

    const blocked = await request(app).post('/x').expect(429);
    expect(blocked.body.error.type).toBe('rate_limit_error');
    expect(blocked.body.error.message).toMatch(/Too many requests/);
    expect(blocked.headers['retry-after']).toBeDefined();
  });

  it('reports Retry-After in whole seconds, never below 1', async () => {
    const clock = fakeClock();
    const app = appWith(rateLimit({ windowMs: 60000, max: 1, now: clock.now, keyFn: () => 'c1' }));

    await request(app).post('/x').expect(200);
    const first = await request(app).post('/x').expect(429);
    expect(Number(first.headers['retry-after'])).toBe(60);

    // Right at the edge of the window the remainder rounds to less than a second;
    // the header must still say "wait", not "0".
    clock.advance(59999);
    const edge = await request(app).post('/x').expect(429);
    expect(Number(edge.headers['retry-after'])).toBe(1);
  });

  it('counts each client separately', async () => {
    let who = 'a';
    const app = appWith(rateLimit({ windowMs: 60000, max: 1, keyFn: () => who }));

    await request(app).post('/x').expect(200); // a's only request
    who = 'b';
    await request(app).post('/x').expect(200); // b is unaffected by a
    who = 'a';
    await request(app).post('/x').expect(429);
  });

  it('starts a fresh allowance once the window rolls over', async () => {
    const clock = fakeClock();
    const app = appWith(rateLimit({ windowMs: 1000, max: 1, now: clock.now, keyFn: () => 'c1' }));

    await request(app).post('/x').expect(200);
    await request(app).post('/x').expect(429);

    clock.advance(1001);
    await request(app).post('/x').expect(200);
  });

  it('sweeps expired windows so the map tracks active clients only', async () => {
    const clock = fakeClock();
    const limiter = rateLimit({ windowMs: 1000, now: clock.now, keyFn: (req) => req.headers['x-who'] });
    const app = appWith(limiter);

    await request(app).post('/x').set('x-who', 'one').expect(200);
    await request(app).post('/x').set('x-who', 'two').expect(200);
    expect(limiter.hits.size).toBe(2);

    // Both windows expire; the next request should evict them, not accumulate.
    clock.advance(1001);
    await request(app).post('/x').set('x-who', 'three').expect(200);
    expect(limiter.hits.size).toBe(1);
    expect(limiter.hits.has('three')).toBe(true);
  });

  it('applies defaults when constructed with no options', async () => {
    const limiter = rateLimit();
    const app = appWith(limiter);
    await request(app).post('/x').expect(200);
    expect(limiter.hits.size).toBe(1);
  });

  it('still limits when the response cannot set headers', () => {
    const limiter = rateLimit({ max: 1, keyFn: () => 'k' });
    const json = jest.fn();
    const res = { status: jest.fn(() => ({ json })) };
    const next = jest.fn();

    limiter({}, res, next);          // first call passes through
    limiter({}, res, next);          // second is refused
    expect(next).toHaveBeenCalledTimes(1);
    expect(res.status).toHaveBeenCalledWith(429);
  });

  describe('clientKey', () => {
    it('prefers req.ip', () => {
      expect(clientKey({ ip: '1.2.3.4', socket: { remoteAddress: '5.6.7.8' } })).toBe('1.2.3.4');
    });

    it('falls back to the socket address', () => {
      expect(clientKey({ socket: { remoteAddress: '5.6.7.8' } })).toBe('5.6.7.8');
    });

    // A request with no identity must land in a bucket, not bypass the limit.
    it('falls back to a shared bucket rather than going unlimited', () => {
      expect(clientKey({})).toBe('unknown');
      expect(clientKey(null)).toBe('unknown');
      expect(clientKey({ socket: {} })).toBe('unknown');
    });
  });
});
