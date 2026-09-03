// The marketing pages are served at extensionless URLs (/chat, not /chat.html).
// Builds the real site into dist/ and drives the Express app against it, so both
// halves are covered together: express.static's `extensions` option resolving
// /chat to dist/chat.html, and the 301 that retires the old .html URLs.
const { execFileSync } = require('node:child_process');
const path = require('node:path');
const request = require('supertest');
const { app } = require('../src/index');

const ROOT = path.join(__dirname, '../..');

beforeAll(() => {
  execFileSync('node', ['site/build-site.mjs'], { cwd: ROOT, stdio: 'ignore' });
});

describe('extensionless page URLs', () => {
  it('serves a page at its extensionless path', async () => {
    const res = await request(app).get('/chat');
    expect(res.status).toBe(200);
    expect(res.headers['content-type']).toMatch(/text\/html/);
    expect(res.text).toContain('<title>LLMJob Chat');
  });

  it('still serves the home page at /', async () => {
    const res = await request(app).get('/');
    expect(res.status).toBe(200);
    expect(res.text).toContain('<title>LLMJob');
  });

  it('links between pages carry no .html', async () => {
    const res = await request(app).get('/');
    expect(res.text).toContain('href="/chat"');
    expect(res.text).not.toMatch(/href="[^"]*\.html"/);
  });
});

describe('legacy .html URLs', () => {
  it('redirects /chat.html to /chat', async () => {
    const res = await request(app).get('/chat.html');
    expect(res.status).toBe(301);
    expect(res.headers.location).toBe('/chat');
  });

  it('redirects /index.html to the root', async () => {
    const res = await request(app).get('/index.html');
    expect(res.status).toBe(301);
    expect(res.headers.location).toBe('/');
  });

  it('keeps the query string', async () => {
    const res = await request(app).get('/docs.html?section=keys');
    expect(res.status).toBe(301);
    expect(res.headers.location).toBe('/docs?section=keys');
  });

  it('leaves non-GET requests alone', async () => {
    const res = await request(app).post('/chat.html');
    expect(res.status).toBe(404);
  });
});
