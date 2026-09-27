// The marketing pages are served at extensionless URLs (/network, not
// /network.html). Builds the real site into dist/ and drives the Express app
// against it, so both halves are covered together: express.static's
// `extensions` option resolving /network to dist/network.html, and the 301 that
// retires the old .html URLs.
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
    const res = await request(app).get('/network');
    expect(res.status).toBe(200);
    expect(res.headers['content-type']).toMatch(/text\/html/);
    expect(res.text).toContain('<title>LLMJob Network');
  });

  // The chat page was removed. Its URL now serves a small page that sends the
  // visitor to the home page (site/redirects.json), so old links still work.
  it('sends the retired chat page to the home page', async () => {
    const res = await request(app).get('/chat');
    expect(res.status).toBe(200);
    expect(res.text).toContain('<meta http-equiv="refresh" content="0; url=/" />');
    expect(res.text).toContain('location.replace("/")');
    expect(res.text).toContain('<meta name="robots" content="noindex" />');
    // Nothing on the site links to it any more.
    const home = await request(app).get('/');
    expect(home.text).not.toContain('href="/chat"');
  });

  it('still serves the home page at /', async () => {
    const res = await request(app).get('/');
    expect(res.status).toBe(200);
    expect(res.text).toContain('<title>LLMJob');
  });

  it('links between pages carry no .html', async () => {
    const res = await request(app).get('/');
    expect(res.text).toContain('href="/network"');
    expect(res.text).not.toMatch(/href="[^"]*\.html"/);
  });
});

describe('legacy .html URLs', () => {
  it('redirects /network.html to /network', async () => {
    const res = await request(app).get('/network.html');
    expect(res.status).toBe(301);
    expect(res.headers.location).toBe('/network');
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
    const res = await request(app).post('/network.html');
    expect(res.status).toBe(404);
  });
});
